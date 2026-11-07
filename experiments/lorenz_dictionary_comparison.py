"""Lorenz: does the Deep-Koopman-KANDy dictionary help EDMD, and does SINDy
recover the equations from it?

Self-contained.  One system (Lorenz), one run, one table.

    python experiments/lorenz_dictionary_comparison.py
    python experiments/lorenz_dictionary_comparison.py --epochs 40   # ~2 min

Pipeline
    1. simulate Lorenz, split train/test
    2. train Deep-Koopman-KANDy (KAN autoencoder + stable generator)
    3. read the dictionary T off the encoder with the paper's decomposition
       (manifold_analysis.full_decomposition_nd, target_type="latent"): each
       second-layer manifold node v_k is factored as v_k = h_k(g_k), with g_k
       recovered on the level sets and h_k from the arc-length gradient
       identity h'(g) = (grad f . grad g)/||grad g||^2; T is the union of the
       surviving monomials
    4. compare five predictors on identical data and metrics:
         Deep-Koopman-KANDy   decoder o exp(K dt) o encoder, nothing re-fit
         EDMD(T)              EDMD with T, K re-fit by least squares
         EDMD                 EDMD with the full degree-2 library
         EDMD-DL              EDMD with a learned dictionary [x, MLP(x)]
         ResDMD               EDMD on the degree-2 library, keeping only
                              eigenpairs whose residual is below tol
    5. hand T to SINDy and print the equations it returns

Metrics
    1-step nMSE      squared error of x_{t+1}, per-coordinate variance normalised
    NRMSE @ 1 tau_L  rollout error one Lyapunov time out (tau_L = 1/0.9056)
    VPT_0.5          first rollout step where mean NRMSE exceeds 0.5

Note on degree: the readout uses degree 2, which is where the Lorenz target
{x, y, z, xy, xz} lives.  That library has only 9 atoms, so the discovered
union often fills it -- in which case EDMD(T) and EDMD coincide by
construction, and the table says so.  Degree 3 leaves room for selection to be
wrong but is not the paper's target degree.

Environment: run this with the project venv, which has pysindy 2.1 working
natively:

    ./venv/bin/python experiments/lorenz_dictionary_comparison.py

(The system anaconda env has pysindy 1.7.5, which is incompatible with its
scikit-learn 1.8 -- pysindy's estimators predate sklearn's __sklearn_tags__
protocol.  If pysindy fails to import or fit, this script falls back to its own
STLSQ and says so loudly, so the table still completes either way.)
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
warnings.filterwarnings("ignore")

from data import simulate_lorenz                                # noqa: E402
from manifold_analysis import full_decomposition_nd             # noqa: E402
from models import KANAutoEncoder, PyKANAutoEncoder             # noqa: E402

sys.path.insert(0, os.path.join(_ROOT, "experiments"))
from lorenz_stable import DeepKoopmanCTStable, train as train_stable  # noqa: E402

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT = 0.01
LAMBDA_MAX = 0.9056                     # largest Lyapunov exponent
TAU_L = 1.0 / LAMBDA_MAX                # Lyapunov time (~1.10 time units)
DEGREE = 3
VARS = ["x", "y", "z"]
TRUE_EQ = {                             # atom -> (dx/dt, dy/dt, dz/dt)
    "x": (-SIGMA, RHO, 0.0),
    "y": (SIGMA, -1.0, 0.0),
    "z": (0.0, 0.0, -BETA),
    "x*z": (0.0, -1.0, 0.0),
    "x*y": (0.0, 0.0, 1.0),
}


# ===========================================================================
# 1. Data
# ===========================================================================
def lorenz_data(seed, n_train_traj=60, n_test_traj=12, n_steps=6000):
    """Paper's Lorenz setup (60 x 6000 at dt=0.01), plus a held-out test block.

    lorenz_stable.train splits the trajectories it is given 48/12 into
    train/val, so the test trajectories here are never seen during training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    traj = simulate_lorenz(n_traj=n_train_traj + n_test_traj, n_steps=n_steps,
                           dt=DT, burn_in=300,
                           device="cpu").numpy().astype(np.float64)
    return traj[:n_train_traj], traj[n_train_traj:]


def pairs(traj):
    return traj[:, :-1, :].reshape(-1, 3), traj[:, 1:, :].reshape(-1, 3)


def lorenz_rhs(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return np.stack([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z],
                    axis=1)


# ===========================================================================
# 2. Deep-Koopman-KANDy  (the repo's model and training, imported verbatim)
# ===========================================================================
# Lorenz KAN settings, matching experiments/spline_regularization_sweep.py
KAN_KWARGS = dict(grid_size=5, spline_order=3,
                  grid_range_l1=[-30, 50], grid_range_l2=[-10, 10])


def load_trained(path, device="cpu"):
    """Load an already-trained Deep-Koopman-KANDy checkpoint.

    This is the default path: the paper's Lorenz model is already trained and
    committed, so there is no reason to retrain it here.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    mid, latent = cfg["mid"], cfg["latent"]
    ae = PyKANAutoEncoder(3, mid, latent, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, latent)
    model.load_state_dict(ck["model_state_dict"])
    return model.eval().to(device), mid, latent


def build_model(latent, mid, device="cpu"):
    """Two-layer KANAutoEncoder + lorenz_stable.DeepKoopmanCTStable.

    The KANLinear backbone (local kan.py) rather than PyKAN: it is far faster,
    and it exposes encoder.kan1 / encoder.kan2, so the readout can address the
    second-layer manifold directly (target_type="latent").
    """
    ae = KANAutoEncoder(3, mid, latent, **KAN_KWARGS)
    return DeepKoopmanCTStable(ae, latent).to(device)


def build_and_train(traj, latent, mid, epochs, seed, device="cpu"):
    """Architecture and training loop both imported, not reimplemented."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ae = KANAutoEncoder(3, mid, latent, **KAN_KWARGS).to(device)
    model = train_stable(ae, latent, torch.tensor(traj, dtype=torch.float32),
                         epochs=epochs, device=device)
    return model.eval()


def kandy_step(model):
    """One step of decoder o exp(A dt) o encoder."""
    def step(x):
        with torch.no_grad():
            t = torch.tensor(np.atleast_2d(x), dtype=torch.float32)
            return model.forward_one_step(t, DT)["x_tp1_pred"].numpy()
    return step


# ===========================================================================
# 3. Dictionary
# ===========================================================================
def monomials(degree=DEGREE):
    """Degree-1..degree power tuples over (x, y, z), with names."""
    out = []
    for total in range(1, degree + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                c = total - a - b
                out.append((a, b, c))
    names = []
    for p in out:
        parts = [f"{v}^{e}" if e > 1 else v for v, e in zip(VARS, p) if e]
        names.append("*".join(parts))
    return out, names


def lift(X, powers, scale=None):
    """[1, atoms...] with optional per-column scaling for conditioning."""
    cols = [np.ones(len(X))] + [np.prod(X ** np.array(p, dtype=float), axis=1)
                                for p in powers]
    P = np.stack(cols, axis=1)
    if scale is None:
        scale = P.std(axis=0)
        scale[scale < 1e-12] = 1.0
    return P / scale, scale


def extract_dictionary(model, X, alpha=5e-3, h_degree=4, n_samples=4000):
    """The paper's readout: level-set + arc-length decomposition of the
    SECOND-LAYER manifold of the two-layer KAN encoder.

    This is not a regression of the latent against monomials.  For each
    second-layer node of the two-layer KAN encoder,
    manifold_analysis.full_decomposition_nd factors it as

            v_k(x) = h_k( g_k(x) )

    with g_k the sparse inner function recovered on the level sets, and h_k the
    1-D outer function recovered from the arc-length gradient identity
    h'(g) = (grad f . grad g) / ||grad g||^2 followed by term-by-term
    integration.  (For the PyKAN encoder the second layer's output IS the
    second-layer manifold itself.  The
    dictionary T is the union over nodes of the monomials
    surviving that decomposition (the "ideal" generators), which is Phase 3 of
    Algorithm 1.

    Returns (powers, names, stats).
    """
    idx = np.random.default_rng(0).choice(len(X), min(n_samples, len(X)),
                                          replace=False)
    xt = torch.tensor(X[idx], dtype=torch.float32)

    atoms, per_node, r2_g, r2_hg, n_affine, n_failed = {}, [], [], [], 0, 0
    for k in range(model.latent_dim):
        try:
            dec = full_decomposition_nd(model, xt, k, VARS,
                                        target_type="latent",
                                        max_degree=DEGREE, lasso_alpha=alpha,
                                        h_degree=h_degree)
        except Exception:
            # A node whose level sets are degenerate (||grad g|| ~ 0 over the
            # sample) leaves the outer-function fit with no valid points.  Skip
            # it and report the count rather than dropping the whole readout.
            n_failed += 1
            continue
        gens = [g for g in dec["ideal"]["generators"] if sum(g["power"]) > 0]
        for g in gens:
            atoms[tuple(g["power"])] = g["name"]
        per_node.append(len(gens))
        r2_g.append(dec["g_recon_r2"])
        r2_hg.append(dec["recon_r2"])
        n_affine += bool(dec["outer"].get("used_affine"))

    powers = sorted(atoms, key=lambda p: (sum(p), p))
    stats = dict(per_node_median=float(np.median(per_node)) if per_node else 0.0,
                 r2_g_median=float(np.median(r2_g)) if r2_g else float("nan"),
                 r2_hg_median=float(np.median(r2_hg)) if r2_hg else float("nan"),
                 n_affine=n_affine, n_nodes=model.latent_dim,
                 n_failed=n_failed)
    return powers, [atoms[p] for p in powers], stats


# ===========================================================================
# 4. Predictors.  Each returns step(x_batch) -> x_next_batch.
# ===========================================================================
def fit_edmd(X, Y, powers):
    PX, scale = lift(X, powers)
    PY, _ = lift(Y, powers, scale)
    K = np.linalg.lstsq(PX, PY, rcond=None)[0]
    B = np.linalg.lstsq(PX, X, rcond=None)[0]

    def step(x):
        P, _ = lift(np.atleast_2d(x), powers, scale)
        return P @ K @ B
    return step, K, PX, PY


def fit_resdmd(X, Y, powers, tol=0.1):
    """EDMD on the same library, keeping only low-residual eigenpairs."""
    step, K, PX, PY = fit_edmd(X, Y, powers)
    n = len(PX)
    G, A, L = PX.T @ PX / n, PX.T @ PY / n, PY.T @ PY / n
    lam, V = np.linalg.eig(K)
    res = []
    for i in range(len(lam)):
        g, l = V[:, i], lam[i]
        num = g.conj() @ (L - np.conj(l) * A - l * A.conj().T
                          + abs(l) ** 2 * G) @ g
        res.append(float(np.sqrt(max(np.real(num / (g.conj() @ G @ g)), 0.0))))
    res = np.array(res)

    good = res < tol
    if good.sum() < 2:                       # keep the best few if tol is harsh
        good = np.zeros_like(res, bool)
        good[np.argsort(res)[:max(2, len(res) // 4)]] = True
    Vg = V[:, good]
    Kf = np.real(Vg @ np.diag(lam[good]) @ np.linalg.pinv(Vg))

    _, scale = lift(X, powers)
    B = np.linalg.lstsq(lift(X, powers, scale)[0], X, rcond=None)[0]

    def step(x):
        P, _ = lift(np.atleast_2d(x), powers, scale)
        return P @ Kf @ B
    return step, res, int(good.sum())


def fit_edmd_dl(X, Y, lift_dim, epochs, seed, hidden=64, lr=1e-3, batch=512):
    """Li et al. (2017): dictionary [1, x, MLP(x)], learned with K in the loop."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(),
                        nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, lift_dim))
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)

    def psi(t):
        return torch.cat([torch.ones(len(t), 1), t, net(t)], dim=1)

    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            i = perm[s:s + batch]
            PX, PY = psi(Xt[i]), psi(Yt[i])
            K = torch.linalg.lstsq(PX, PY).solution      # K in closed form
            loss = ((PX @ K - PY) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        PX, PY = psi(Xt), psi(Yt)
        K = torch.linalg.lstsq(PX, PY).solution
        B = torch.linalg.lstsq(PX, Xt).solution

    def step(x):
        with torch.no_grad():
            t = torch.tensor(np.atleast_2d(x), dtype=torch.float32)
            return (psi(t) @ K @ B).numpy()
    return step, 4 + lift_dim


# ===========================================================================
# 5. Metrics
# ===========================================================================
def one_step_nmse(step, X, Y):
    P = step(X)
    return float(((P - Y) ** 2 / np.maximum(Y.var(axis=0), 1e-8)).mean())


def rollout_metrics(step, test_traj, n_steps, std_ref, n_ics=24):
    curves = []
    for i in range(min(n_ics, len(test_traj))):
        x = test_traj[i, 0, :].copy()
        pred = [x]
        for _ in range(n_steps):
            x = np.clip(np.nan_to_num(step(x)[0], nan=1e6), -1e6, 1e6)
            pred.append(x)
        pred = np.array(pred)
        true = test_traj[i, :n_steps + 1, :]
        curves.append((np.abs(pred - true) / std_ref).mean(axis=1))
    mean = np.minimum(np.mean(curves, axis=0), 10.0)

    at_tau = int(min(TAU_L / DT, len(mean) - 1))
    cross = np.where(mean > 0.5)[0]
    vpt = float(cross[0]) if len(cross) else float("inf")
    return float(mean[at_tau]), vpt


# ===========================================================================
# 6. SINDy on the discovered dictionary
# ===========================================================================
def run_sindy(X, xdot, powers, names, threshold=0.05):
    """Fit SINDy with T as its library.  Returns (equations, used_pysindy)."""
    Theta = np.stack([np.prod(X ** np.array(p, dtype=float), axis=1)
                      for p in powers], axis=1)
    try:
        # pysindy's own STLSQ, applied to exactly the atoms of T.  We build the
        # design matrix ourselves rather than via CustomLibrary: that library
        # applies each function across every combination of state variables,
        # which would silently expand T into a different, larger dictionary.
        import pysindy as ps
        opt = ps.STLSQ(threshold=threshold)
        opt.fit(Theta, xdot)
        return np.asarray(opt.coef_).T, names, True         # (n_atoms, 3)
    except Exception as exc:
        print(f"  ! pysindy unavailable ({type(exc).__name__}: "
              f"{str(exc)[:70]}...)")
        print("  ! falling back to built-in STLSQ.  For real pysindy, run "
              "this with ./venv/bin/python")
        Xi = np.linalg.lstsq(Theta, xdot, rcond=None)[0]
        for _ in range(20):
            small = np.abs(Xi) < threshold
            Xi_new = np.zeros_like(Xi)
            for j in range(3):
                big = ~small[:, j]
                if big.any():
                    Xi_new[big, j] = np.linalg.lstsq(Theta[:, big], xdot[:, j],
                                                     rcond=None)[0]
            if np.allclose(Xi_new, Xi):
                break
            Xi = Xi_new
        return Xi, names, False


def _norm(name):
    """manifold_analysis writes cross terms as 'xy'; monomials() as 'x*y'."""
    return name.replace("*", "").replace("^", "")


def report_sindy(coef, names):
    eqs, exact = [], True
    for j, lhs in enumerate(["dx/dt", "dy/dt", "dz/dt"]):
        terms = {_norm(names[i]): coef[i, j] for i in range(len(names))
                 if abs(coef[i, j]) > 1e-3}
        truth = {_norm(k): v[j] for k, v in TRUE_EQ.items() if abs(v[j]) > 0}
        extra = set(terms) - set(truth)
        missing = set(truth) - set(terms)
        err = max([abs(terms[k] - v) / abs(v) for k, v in truth.items()
                   if k in terms] or [np.inf])
        if extra or missing or err > 0.02:
            exact = False
        eqs.append((lhs, " ".join(f"{c:+.4f} {n}" for n, c in
                                  sorted(terms.items(), key=lambda kv: -abs(kv[1]))),
                    sorted(extra), sorted(missing), err))
    return eqs, exact


# ===========================================================================
# 7. Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=os.path.join(_ROOT, "results", "lorenz_stable",
                                         "stable_8_32_model.pt"),
                    help="trained Deep-Koopman-KANDy checkpoint to read the "
                         "dictionary off (default: the committed Lorenz model)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--traj", type=int, default=60,
                    help="trajectories (>=60: lorenz_stable.train splits 48/12)")
    ap.add_argument("--steps", type=int, default=6000, help="steps per trajectory")
    ap.add_argument("--quick", action="store_true",
                    help="reduced data/epochs for iteration (~4 min instead of "
                         "~2 h); NOT the paper's training budget")
    ap.add_argument("--no-cache", action="store_true",
                    help="retrain even if a cached model exists")
    ap.add_argument("--latent", type=int, default=5,
                    help="latent dim (5 = the paper's dictionary-recovery model)")
    ap.add_argument("--mid", type=int, default=8)
    ap.add_argument("--lasso-alpha", type=float, default=1e-5,
                    help="Lasso weight for the inner function")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resdmd-tol", type=float, default=0.1)
    ap.add_argument("--roll-taus", type=float, default=3.0,
                    help="rollout length in Lyapunov times (default 3)")
    args = ap.parse_args()
    if args.quick:
        args.steps, args.epochs = 1200, 30

    print("=" * 74)
    print("Lorenz: Deep-Koopman-KANDy dictionary vs EDMD / EDMD-DL / ResDMD")
    print("=" * 74)

    train, test = lorenz_data(args.seed, n_train_traj=args.traj,
                              n_steps=args.steps)
    Xtr, Ytr = pairs(train)
    Xte, Yte = pairs(test)
    std_ref = np.maximum(test.reshape(-1, 3).std(axis=0), 1e-8)
    n_roll = int(args.roll_taus * TAU_L / DT)
    print(f"\ndata: {len(Xtr):,} train pairs, {len(Xte):,} test pairs, "
          f"rollout {n_roll} steps ({args.roll_taus:g} tau_L)")

    print(f"\n[1/4] loading trained Deep-Koopman-KANDy")
    model, mid, latent = load_trained(args.checkpoint)
    args.mid, args.latent = mid, latent
    print(f"    {os.path.relpath(args.checkpoint, _ROOT)}  "
          f"(PyKAN [3,{mid},{latent}], already trained)")

    print("\n[2/4] reading the dictionary off the second-layer manifold")
    print("      (level-set inner g + arc-length outer h, per node)")
    T_powers, T_names, st = extract_dictionary(model, Xtr,
                                              alpha=args.lasso_alpha)
    all_powers, all_names = monomials()
    print(f"    T = {len(T_powers)} of {len(all_powers)} degree-{DEGREE} atoms "
          f"(median {st['per_node_median']:.0f} per node over "
          f"{st['n_nodes']} nodes)")
    print(f"    decomposition quality: median R2(g) = {st['r2_g_median']:.3f}, "
          f"median R2(h.g) = {st['r2_hg_median']:.3f}; "
          f"{st['n_affine']}/{st['n_nodes']} nodes have affine h")
    if st["n_failed"]:
        print(f"    WARNING: {st['n_failed']}/{st['n_nodes']} nodes had "
              f"degenerate level sets and were skipped")
    print(f"    {', '.join(T_names)}")
    target = {_norm(a) for a in ("x", "y", "z", "x*y", "x*z")}
    hit = target & {_norm(a) for a in T_names}
    print(f"    vs the known target {sorted(target)}: "
          f"recall {len(hit)}/{len(target)}, "
          f"{len({_norm(a) for a in T_names} - target)} extra")
    if len(T_powers) == len(all_powers):
        print("    NOTE: T fills the degree-2 library, so EDMD(T) and EDMD "
              "are the same fit.")

    print("\n[3/4] fitting predictors")
    rows = []

    def add(name, step, natoms, note=""):
        nm = one_step_nmse(step, Xte, Yte)
        nr, vpt = rollout_metrics(step, test, n_roll, std_ref)
        rows.append((name, natoms, nm, nr, vpt, note))
        print(f"    {name:<22} done")

    add("Deep-Koopman-KANDy", kandy_step(model), model.latent_dim, "latent dim")
    add("EDMD(T)", fit_edmd(Xtr, Ytr, T_powers)[0], len(T_powers))
    add("EDMD", fit_edmd(Xtr, Ytr, all_powers)[0], len(all_powers))
    dl_step, dl_atoms = fit_edmd_dl(Xtr, Ytr, max(4, len(T_powers)),
                                    min(args.epochs, 50), args.seed)
    add("EDMD-DL", dl_step, dl_atoms)
    res_step, res, n_keep = fit_resdmd(Xtr, Ytr, all_powers, args.resdmd_tol)
    add("ResDMD", res_step, len(all_powers),
        f"kept {n_keep}/{len(res)} eigenpairs with residual < "
        f"{args.resdmd_tol} (median residual {np.median(res):.3f}). "
        f"Residual filtering certifies the spectrum, it is not a way to build "
        f"a predictor: discarding modes also discards subspace the state "
        f"readout needs, so the rollout degrades. Keeping every mode makes "
        f"this row identical to EDMD.")

    print("\n[4/4] SINDy on the discovered dictionary T")
    coef, sindy_names, used_pysindy = run_sindy(Xtr, lorenz_rhs(Xtr),
                                                T_powers, T_names)
    eqs, exact = report_sindy(coef, sindy_names)

    # ---------------- table ----------------
    print("\n" + "=" * 74)
    print(f"{'method':<22}{'atoms':>7}{'1-step nMSE':>14}"
          f"{'NRMSE@1tau':>12}{'VPT0.5':>9}")
    print("-" * 74)
    for name, natoms, nm, nr, vpt, note in rows:
        v = f">{n_roll}" if not np.isfinite(vpt) else f"{vpt:.0f}"
        print(f"{name:<22}{natoms:>7}{nm:>14.3e}{nr:>12.3f}{v:>9}")
    print("-" * 74)
    print(f"VPT0.5 = first rollout step with mean NRMSE > 0.5, in steps "
          f"(dt={DT}); 1 tau_L = {int(TAU_L / DT)} steps.")
    print(f'">{n_roll}" means the rollout never crossed 0.5 within the '
          f"{n_roll}-step ({args.roll_taus:g} tau_L) window -- censored by the")
    print("   measurement window, not infinite.  Raise --roll-taus to resolve it.")
    import textwrap
    for name, natoms, nm, nr, vpt, note in rows:
        if note:
            print(textwrap.fill(f"  {name}: {note}", 74,
                                subsequent_indent="    "))

    print(f"\nSINDy with T  ({'pysindy' if used_pysindy else 'built-in STLSQ'})"
          f" -> {'RECOVERS LORENZ' if exact else 'does not match Lorenz'}")
    for lhs, rhs, extra, missing, err in eqs:
        print(f"  {lhs} = {rhs}")
        if extra or missing:
            print(f"        extra={extra} missing={missing}")
        print(f"        max relative coefficient error: {err:.2e}")
    print("\ntrue Lorenz:  dx/dt = 10(y-x),  dy/dt = x(28-z)-y,  "
          "dz/dt = xy - (8/3)z")
    print("=" * 74)


if __name__ == "__main__":
    main()
