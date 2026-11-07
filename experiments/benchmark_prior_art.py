"""
Benchmark Deep-Koopman-KANDy against open-source prior-art Koopman methods.

Prior art with public code (verified 2026-07):
  * Coote & Colbrook (2026) "Residual-Guided Dictionary Learning ..." [pyresdmd]
    https://github.com/GeorgeCoote/pyresdmd            -- used directly.
  * Nehma & Tiwari (2024) "Leveraging KANs For Enhanced Deep Koopman ..."
    https://github.com/tiwari-research-group/Koopman-with-KANs
    -- their code is control-focused notebooks; we FAITHFULLY REIMPLEMENT the
    core architecture (KAN enc/dec + UNCONSTRAINED linear Koopman, no stability,
    no control) with our local KANLinear (no pykan dependency).
  * EDMD-DL (Li et al. 2017): dictionary Psi=[x, MLP(x)], closed-form K.

These methods produce PREDICTORS/spectra, not symbolic dictionaries, so the
common axis is PREDICTION.  Each method rolls out in its NATIVE mode -- we do NOT
give the baselines our autoregressive re-encoding:
  * ours    : autoregressive  D o M o E   (re-encode every step), stable A=K-L^T L
  * nehma   : pure-latent linear  dec(K^n enc(x0))
  * coote   : pure-latent EDMD     C (Psi(x) K^n)
  * edmd_dl : state-readout + re-lift  [Psi(x) K]_{:d}

Metrics (per method, on a held-out test set):
  * one-step MSE (raw + variance-normalized)
  * NRMSE vs horizon (capped at 10 = "diverged")
  * VPT: first step NRMSE crosses 0.5 and 1.0, reported in STEPS and (where a
    clean Lyapunov exponent exists) in Lyapunov times tau_Lambda
  * trainable params; training wall-time

Systems: lorenz, cat_map, standard_map, ikeda, ks, mackey_glass.
Torus maps (cat, standard) are embedded to R^4 via (cos,sin) as in the paper;
KS and Mackey-Glass reuse the committed cached datasets.

Usage:
    python experiments/benchmark_prior_art.py                       # all 6, 3 seeds
    python experiments/benchmark_prior_art.py --systems lorenz --seeds 1
    python experiments/benchmark_prior_art.py --smoke               # fast sanity
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

# Cap intra-op threads: on many-core CPUs the default over-subscribes for these
# many-small-op KAN workloads and thrashes (~3x slower). 8 is a good sweet spot.
torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))

# --- paths -----------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

_PYRESDMD = os.environ.get(
    "PYRESDMD_PATH",
    "/tmp/claude-1000/-home-kslote-Desktop-deep-koopman-KAN-code-base-deep-koopman-KANDy/"
    "48d1d603-92bb-4d0e-89e0-47c76e5e65a3/scratchpad/extrepos/pyresdmd",
)
if os.path.isdir(_PYRESDMD) and _PYRESDMD not in sys.path:
    sys.path.insert(0, _PYRESDMD)

from models import KANAutoEncoder          # noqa: E402
from data import simulate_lorenz           # noqa: E402


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===========================================================================
# Simulators (self-contained) and cached-data loaders
# ===========================================================================
def _lorenz(seed, n_traj=40, n_steps=3000):
    tr = simulate_lorenz(n_traj=n_traj, n_steps=n_steps, dt=0.01,
                         burn_in=200, device="cpu")
    return tr.detach().cpu().numpy().astype(np.float32)


_CAT = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float64)


def _cat_map(seed, n_traj=120, n_steps=320):
    rng = np.random.default_rng(seed)
    xy = rng.random((n_traj, 2))
    out = np.zeros((n_traj, n_steps, 2), dtype=np.float64)
    for t in range(n_steps):
        out[:, t, :] = xy
        xy = (xy @ _CAT.T) % 1.0
    # torus embed -> R^4  (cos/sin of 2*pi*coord)
    x, y = out[..., 0], out[..., 1]
    emb = np.stack([np.cos(2 * np.pi * x), np.sin(2 * np.pi * x),
                    np.cos(2 * np.pi * y), np.sin(2 * np.pi * y)], axis=-1)
    return emb.astype(np.float32)


def _standard_map(seed, n_traj=120, n_steps=620, K=0.97):
    rng = np.random.default_rng(seed)
    x = 2 * np.pi * rng.random(n_traj)
    y = 2 * np.pi * rng.random(n_traj)
    out = np.zeros((n_traj, n_steps, 2), dtype=np.float64)
    for t in range(n_steps):
        out[:, t, 0] = x
        out[:, t, 1] = y
        y = (y + K * np.sin(x)) % (2 * np.pi)
        x = (x + y) % (2 * np.pi)
    th, p = out[..., 0], out[..., 1]
    emb = np.stack([np.cos(th), np.sin(th), np.cos(p), np.sin(p)], axis=-1)
    return emb.astype(np.float32)


def _ikeda(seed, n_traj=120, n_steps=620, u=0.9, burn_in=500):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5, n_traj)
    y = rng.uniform(-0.5, 0.5, n_traj)
    for _ in range(burn_in):
        r2 = x * x + y * y
        t = 0.4 - 6.0 / (1.0 + r2)
        ct, st = np.cos(t), np.sin(t)
        x, y = 1.0 + u * (x * ct - y * st), u * (x * st + y * ct)
    out = np.zeros((n_traj, n_steps, 2), dtype=np.float64)
    for k in range(n_steps):
        out[:, k, 0] = x
        out[:, k, 1] = y
        r2 = x * x + y * y
        t = 0.4 - 6.0 / (1.0 + r2)
        ct, st = np.cos(t), np.sin(t)
        x, y = 1.0 + u * (x * ct - y * st), u * (x * st + y * ct)
    return out.astype(np.float32)


def _load_cached(path, key=None):
    d = torch.load(os.path.join(_ROOT, path), map_location="cpu")
    if isinstance(d, dict):
        d = d[key if key else "traj"]
    return d.detach().cpu().numpy().astype(np.float32)


# ===========================================================================
# System registry
#   simulate(seed) -> (n_traj, T, d)   [cached systems ignore seed for data]
#   cached: split trajectories into train/test instead of fresh seeds
# ===========================================================================
SYSTEMS = {
    "lorenz": dict(
        state_dim=3, dt=0.01, lambda_max=0.9056, n_roll=None, cached=False,
        simulate=_lorenz, mid=8, latent=16, dict_size=16, dict_hidden=32,
        edmd_lift=5, edmd_hidden=64, epochs=200),
    "cat_map": dict(
        state_dim=4, dt=1.0, lambda_max=0.9624, n_roll=30, cached=False,
        simulate=_cat_map, mid=16, latent=24, dict_size=24, dict_hidden=32,
        edmd_lift=8, edmd_hidden=64, epochs=200),
    "standard_map": dict(
        state_dim=4, dt=1.0, lambda_max=None, n_roll=60, cached=False,
        simulate=_standard_map, mid=16, latent=32, dict_size=32, dict_hidden=32,
        edmd_lift=8, edmd_hidden=64, epochs=200),
    "ikeda": dict(
        state_dim=2, dt=1.0, lambda_max=None, n_roll=60, cached=False,
        simulate=_ikeda, mid=8, latent=16, dict_size=16, dict_hidden=32,
        edmd_lift=5, edmd_hidden=64, epochs=200),
    "ks": dict(
        state_dim=64, dt=0.25, lambda_max=0.043, n_roll=180, cached=True,
        simulate=lambda s: _load_cached("results/ks_stable/ks_data.pt"),
        mid=48, latent=64, dict_size=64, dict_hidden=32,
        edmd_lift=24, edmd_hidden=128, epochs=100),
    "mackey_glass": dict(
        state_dim=64, dt=4.0, lambda_max=0.0086, n_roll=87, cached=True,
        simulate=lambda s: _load_cached(
            "results/mackey_glass/mg_data_N64_dt4_T800_n24.pt", "traj"),
        mid=48, latent=64, dict_size=64, dict_hidden=32,
        edmd_lift=24, edmd_hidden=128, epochs=100),
}


def make_pairs(traj):
    d = traj.shape[-1]
    return traj[:, :-1, :].reshape(-1, d), traj[:, 1:, :].reshape(-1, d)


# ===========================================================================
# Metrics
# ===========================================================================
def one_step_errors(predict_onestep, X, Y):
    with torch.no_grad():
        Yhat = predict_onestep(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()
    se = (Yhat - Y) ** 2
    raw = float(se.mean())
    var = np.maximum(Y.var(axis=0), 1e-8)
    return raw, float((se / var).mean())


def nrmse_curve(rollout, test_traj, n_steps, n_ics, std_ref):
    T = test_traj.shape[1]
    n_steps = min(n_steps, T - 1)
    rng = np.random.default_rng(0)
    idx = rng.choice(test_traj.shape[0],
                     size=min(n_ics, test_traj.shape[0]), replace=False)
    curves = []
    for i in idx:
        pred = rollout(test_traj[i, 0, :], n_steps)          # (n_steps+1, d)
        true = test_traj[i, : n_steps + 1, :]
        curves.append((np.abs(pred - true) / std_ref).mean(axis=1))
    mean = np.mean(np.stack(curves, axis=0), axis=0)
    return np.minimum(np.nan_to_num(mean, nan=10.0, posinf=10.0), 10.0)


def horizon_crossings(mean_nrmse, dt, lambda_max):
    out = {}
    steps_per_tau = (1.0 / lambda_max) / dt if lambda_max else None
    for thr in (0.5, 1.0):
        c = np.where(mean_nrmse > thr)[0]
        steps = float(c[0]) if len(c) else float("inf")
        out[f"vpt_{thr}_steps"] = steps
        out[f"vpt_{thr}_tau"] = (steps / steps_per_tau
                                 if (steps_per_tau and np.isfinite(steps))
                                 else (float("inf") if steps_per_tau else None))
    return out


def n_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


# ===========================================================================
# Method 1 -- ours (KAN AE + stable generator A = Omega - L^T L)
# ===========================================================================
class StableGenerator(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.skew = nn.Parameter(0.01 * torch.randn(n * (n - 1) // 2))
        self.Ldiag = nn.Parameter(0.01 * torch.randn(n))
        self.Llow = nn.Parameter(0.01 * torch.randn(n * (n - 1) // 2))

    def A(self):
        n = self.n
        Om = torch.zeros(n, n, device=self.skew.device)
        iu = torch.triu_indices(n, n, 1)
        Om[iu[0], iu[1]] = self.skew
        Om = Om - Om.T
        L = torch.zeros(n, n, device=self.Ldiag.device)
        L[range(n), range(n)] = self.Ldiag
        il = torch.tril_indices(n, n, -1)
        L[il[0], il[1]] = self.Llow
        return Om - L.T @ L


class OursModel(nn.Module):
    def __init__(self, state_dim, mid, latent, grl1):
        super().__init__()
        self.ae = KANAutoEncoder(state_dim, mid, latent, grid_range_l1=grl1)
        self.gen = StableGenerator(latent)

    def propagate(self, z, dt):
        return z @ torch.matrix_exp(self.gen.A() * dt).T


class NehmaModel(nn.Module):
    def __init__(self, state_dim, mid, latent, grl1):
        super().__init__()
        self.ae = KANAutoEncoder(state_dim, mid, latent, grid_range_l1=grl1)
        self.K = nn.Linear(latent, latent, bias=False)

    def propagate(self, z, dt=None):
        return self.K(z)


def train_ae_koopman(model, X, Y, dt, epochs, lr, batch, device, seed,
                     progress=None):
    """Paper 3-term loss: predx + latent-consistency + 0.5*recon.

    progress: optional callback(epoch, epochs, loss, elapsed_s) called once per
    epoch.  Purely for console feedback -- training is unchanged when omitted.
    """
    set_seed(seed)
    model.to(device).train()
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = Xt.shape[0]
    t0 = time.time()
    for _ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            xb, yb = Xt[idx], Yt[idx]
            z_t, x_rec = model.ae(xb)
            z_tp1 = model.propagate(z_t, dt)
            x_pred = model.ae.decoder(z_tp1)
            z_enc, _ = model.ae(yb)
            loss = (((x_pred - yb) ** 2).mean()
                    + ((z_tp1 - z_enc) ** 2).mean()
                    + 0.5 * ((x_rec - xb) ** 2).mean())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if progress is not None:
            progress(_ep + 1, epochs, float(loss.detach()), time.time() - t0)
    return time.time() - t0


def ours_predictors(model, dt, device):
    model.eval()

    def predict_onestep(X):
        with torch.no_grad():
            z, _ = model.ae(X.to(device))
            return model.ae.decoder(model.propagate(z, dt))

    def rollout(x0, n_steps):
        x = torch.as_tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
        out = [x.squeeze(0).cpu().numpy()]
        with torch.no_grad():
            for _ in range(n_steps):
                z, _ = model.ae(x)
                x = model.ae.decoder(model.propagate(z, dt))
                out.append(x.squeeze(0).cpu().numpy())
        return np.stack(out, axis=0)

    return predict_onestep, rollout


def nehma_predictors(model, device):
    model.eval()

    def predict_onestep(X):
        with torch.no_grad():
            z, _ = model.ae(X.to(device))
            return model.ae.decoder(model.K(z))

    def rollout(x0, n_steps):
        x = torch.as_tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            z, _ = model.ae(x)
            out = [model.ae.decoder(z).squeeze(0).cpu().numpy()]
            for _ in range(n_steps):
                z = model.K(z)
                out.append(model.ae.decoder(z).squeeze(0).cpu().numpy())
        return np.stack(out, axis=0)

    return predict_onestep, rollout


# ===========================================================================
# Method 3 -- Coote & Colbrook (2026) ResDMD trainable dictionary [pyresdmd]
# ===========================================================================
def run_coote(X, Y, cfg, epochs, lr, device, seed):
    from pyresdmd.dicts import ReLUDictionary, TrainableDictionary
    from pyresdmd.compute.spectra import EDMD
    set_seed(seed)
    d = cfg["state_dim"]
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    relu = ReLUDictionary(input_dim=d, n_functions=cfg["dict_size"],
                          hidden_dim=cfg["dict_hidden"], hidden_layers=1)
    td = TrainableDictionary(relu).to(device)
    t0 = time.time()
    td.fit(Xt, Yt, epochs=epochs, lr=lr, patience=max(10, epochs // 5),
           test_size=0.3, batch_size=0.2, shuffle=True)
    wall = time.time() - t0

    def lift(x):
        feats = td.evaluate(x)
        ones = torch.ones(x.shape[0], 1, device=x.device)
        return torch.cat([ones, x, feats], dim=1)

    with torch.no_grad():
        PsiX, PsiY = lift(Xt), lift(Yt)
        W = torch.ones(PsiX.shape[0], device=device) / PsiX.shape[0]
        K = EDMD(PsiX, PsiY, W)
        C = torch.linalg.lstsq(PsiX, Xt).solution
    npar = n_params(td)

    def predict_onestep(Xq):
        with torch.no_grad():
            return (lift(Xq.to(device)) @ K) @ C

    def rollout(x0, n_steps):
        x = torch.as_tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            psi = lift(x)
            out = [(psi @ C).squeeze(0).cpu().numpy()]
            for _ in range(n_steps):
                psi = psi @ K
                out.append((psi @ C).squeeze(0).cpu().numpy())
        return np.stack(out, axis=0)

    return predict_onestep, rollout, npar, wall


# ===========================================================================
# Method 4 -- EDMD-DL (Li et al. 2017)
# ===========================================================================
def run_edmd_dl(X, Y, cfg, epochs, lr, device, seed):
    set_seed(seed)
    d = cfg["state_dim"]
    hid = cfg.get("edmd_hidden", 64)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    mlp = nn.Sequential(nn.Linear(d, hid), nn.SiLU(),
                        nn.Linear(hid, hid), nn.SiLU(),
                        nn.Linear(hid, cfg.get("edmd_lift", 5))).to(device)

    def lift(x):
        return torch.cat([x, mlp(x)], dim=1)

    def solveK(Px, Py):
        return torch.linalg.lstsq(Px, Py).solution

    opt = torch.optim.AdamW(mlp.parameters(), lr=lr)
    n = Xt.shape[0]
    batch = 2048
    t0 = time.time()
    for _ in range(epochs):
        with torch.no_grad():
            K = solveK(lift(Xt), lift(Yt)).detach()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            Px, Py = lift(Xt[idx]), lift(Yt[idx])
            loss = ((Px @ K - Py) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        K = solveK(lift(Xt), lift(Yt)).detach()
    wall = time.time() - t0
    npar = int(sum(p.numel() for p in mlp.parameters()))

    def predict_onestep(Xq):
        with torch.no_grad():
            return (lift(Xq.to(device)) @ K)[:, :d]

    def rollout(x0, n_steps):
        x = torch.as_tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
        out = [x.squeeze(0).cpu().numpy()]
        with torch.no_grad():
            for _ in range(n_steps):
                x = (lift(x) @ K)[:, :d]
                out.append(x.squeeze(0).cpu().numpy())
        return np.stack(out, axis=0)

    return predict_onestep, rollout, npar, wall


# ===========================================================================
# Driver
# ===========================================================================
def get_data(cfg, seed, smoke):
    """Return (train_traj, test_traj).  Cached systems split by trajectory;
    generated systems use fresh train/test seeds."""
    if cfg["cached"]:
        traj = cfg["simulate"](seed)
        n = traj.shape[0]
        n_te = max(1, n // 5)
        test, train = traj[:n_te], traj[n_te:]
    else:
        set_seed(seed)
        train = cfg["simulate"](seed)
        set_seed(seed + 1000)
        test = cfg["simulate"](seed + 1000)
    if smoke:
        train, test = train[:, :120, :], test[:, :120, :]
    return train, test


def bench_system(name, cfg, seed, device, smoke, n_ics=48):
    dt = cfg["dt"]
    epochs = 3 if smoke else cfg["epochs"]
    d = cfg["state_dim"]
    train_traj, test_traj = get_data(cfg, seed, smoke)
    Xtr, Ytr = make_pairs(train_traj)
    if smoke:
        Xtr, Ytr = Xtr[:2000], Ytr[:2000]
    Xte, Yte = make_pairs(test_traj)
    std_ref = np.maximum(test_traj.reshape(-1, d).std(axis=0), 1e-8)
    flat = train_traj.reshape(-1, d)
    grl1 = [float(flat.min()) - 1.0, float(flat.max()) + 1.0]
    n_roll = 10 if smoke else (cfg["n_roll"] or
                               int(3 * (1.0 / cfg["lambda_max"]) / dt))
    n_roll = min(n_roll, test_traj.shape[1] - 1)

    rows, curves = [], {}

    def record(method, po, ro, npar, wall):
        raw, norm = one_step_errors(po, Xte, Yte)
        mn = nrmse_curve(ro, test_traj, n_roll, (4 if smoke else n_ics), std_ref)
        cr = horizon_crossings(mn, dt, cfg["lambda_max"])
        two = int(min(2 * (1.0 / cfg["lambda_max"]) / dt, len(mn) - 1)) \
            if cfg["lambda_max"] else len(mn) - 1
        row = dict(system=name, method=method, seed=seed,
                   one_step_mse=raw, one_step_nmse=norm,
                   nrmse_end=float(mn[two]), n_params=npar,
                   wall_time_s=round(wall, 1), n_roll=n_roll, **cr)
        rows.append(row)
        curves[method] = mn.tolist()
        tau = cr["vpt_1.0_tau"]
        taus = f"{tau:.2f}tau" if tau not in (None, float("inf")) else "-"
        print(f"  [{method:>8}] 1step_nmse={norm:.3e} "
              f"VPT0.5={cr['vpt_0.5_steps']:.0f}st VPT1.0={cr['vpt_1.0_steps']:.0f}st "
              f"({taus}) params={npar} wall={wall:.0f}s")

    def safe(method, fn):
        try:
            fn()
        except Exception as e:                       # keep the long run alive
            import traceback
            traceback.print_exc()
            print(f"  [{method:>8}] FAILED: {type(e).__name__}: {e}")

    def _ours():
        m = OursModel(d, cfg["mid"], cfg["latent"], grl1)
        w = train_ae_koopman(m, Xtr, Ytr, dt, epochs, 1e-3, 512, device, seed)
        po, ro = ours_predictors(m, dt, device)
        record("ours", po, ro, n_params(m), w)

    def _nehma():
        m = NehmaModel(d, cfg["mid"], cfg["latent"], grl1)
        w = train_ae_koopman(m, Xtr, Ytr, dt, epochs, 1e-3, 512, device, seed)
        po, ro = nehma_predictors(m, device)
        record("nehma", po, ro, n_params(m), w)

    def _coote():
        po, ro, npar, w = run_coote(Xtr, Ytr, cfg, epochs, 1e-3, device, seed)
        record("coote", po, ro, npar, w)

    def _edmd():
        po, ro, npar, w = run_edmd_dl(Xtr, Ytr, cfg, epochs, 1e-3, device, seed)
        record("edmd_dl", po, ro, npar, w)

    safe("ours", _ours)
    safe("nehma", _nehma)
    safe("coote", _coote)
    safe("edmd_dl", _edmd)
    return rows, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS.keys()))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(_ROOT, "results", "benchmark_prior_art"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    import csv
    csv_path = os.path.join(args.outdir, "results.csv")
    fields = ["system", "method", "seed", "one_step_mse", "one_step_nmse",
              "nrmse_end", "vpt_0.5_steps", "vpt_1.0_steps",
              "vpt_0.5_tau", "vpt_1.0_tau", "n_roll", "n_params", "wall_time_s"]

    def flush(rows, curves):
        with open(csv_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k) for k in fields})
        with open(os.path.join(args.outdir, "nrmse_curves.json"), "w") as f:
            json.dump(curves, f, indent=2)

    all_rows, all_curves = [], {}
    for name in args.systems:
        cfg = SYSTEMS[name]
        for seed in range(args.seeds):
            print(f"\n=== {name} | seed {seed} | epochs "
                  f"{3 if args.smoke else cfg['epochs']} | {args.device} ===")
            try:
                rows, curves = bench_system(name, cfg, seed, args.device, args.smoke)
                all_rows.extend(rows)
                all_curves[f"{name}_seed{seed}"] = curves
            except Exception as e:                     # never lose prior systems
                import traceback
                traceback.print_exc()
                print(f"=== {name} seed {seed} FAILED: {e} ===")
            flush(all_rows, all_curves)                # incremental save

    print(f"\nWrote {csv_path}\n\n=== SUMMARY (1-step norm-MSE | VPT0.5,1.0 steps) ===")
    for r in all_rows:
        print(f"{r['system']:>13} {r['method']:>8}  nmse={r['one_step_nmse']:.3e}  "
              f"VPT0.5={r['vpt_0.5_steps']:>5.0f}  VPT1.0={r['vpt_1.0_steps']:>5.0f}  "
              f"params={r['n_params']:>7}")


if __name__ == "__main__":
    main()
