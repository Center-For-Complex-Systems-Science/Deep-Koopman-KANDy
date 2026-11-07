"""EDMD vs EDMD-DL vs ResDMD, using the dictionary learned by Deep-Koopman-KANDy.

Lorenz and KS only.  One code path:

  1. Train a Deep-Koopman-KANDy model (same config as the paper benchmark).
  2. Read off its dictionary T: per latent coordinate, a LassoLars fit of that
     coordinate against a degree-2 monomial library; keep atoms at >=1% of the
     largest |coeff| (the paper's rule); union across coordinates.
  3. Score three EDMD-family methods on the same pairs and metrics:

       EDMD(T)     least squares on T                      <- our dictionary
       EDMD(poly)  least squares on the full degree-2 library   [control]
       EDMD-DL     dictionary [x, MLP(x)] learned end to end, size-matched to
                   |T| (Li et al. 2017)
       ResDMD(T)   the EDMD(T) operator, with Colbrook-Townsend residuals
                   certifying which eigenpairs are trustworthy

ResDMD residuals are computed directly from the Gram matrices rather than via
pyresdmd (that package is not a dependency of this repo, and pysindy-style
version drift is not worth it for eight lines):

    res(lam, g)^2 = g* (L - conj(lam) A - lam A* + |lam|^2 G) g / (g* G g)
    G = Psi_X* Psi_X,  A = Psi_X* Psi_Y,  L = Psi_Y* Psi_Y

Metrics match benchmark_prior_art.py: one-step normalized MSE and VPT_0.5 (the
first rollout step where mean NRMSE crosses 0.5).

    python experiments/edmd_family_benchmark.py                  # both systems
    python experiments/edmd_family_benchmark.py --systems lorenz
    python experiments/edmd_family_benchmark.py --smoke          # fast check
    python experiments/edmd_family_benchmark.py --epochs 40      # ~2 min total

Runtime is dominated by step 1: training the encoder takes ~8 min for Lorenz
and ~2 min for KS on CPU, while the readout and all three EDMD-family methods
together take under a minute.  Trained models are cached under results/, so a
second run of the same configuration starts at step 2 (use --no-cache to force
retraining, --epochs to trade accuracy for time).
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
warnings.filterwarnings("ignore")

from sklearn.linear_model import LassoLars                        # noqa: E402

from benchmark_prior_art import (                                 # noqa: E402
    SYSTEMS, get_data, horizon_crossings, make_pairs, n_params,
    nrmse_curve, one_step_errors, run_edmd_dl, train_ae_koopman, OursModel,
)
from discovered_dictionary_ablation import (                      # noqa: E402
    FeatureMap, make_edmd, monomial_powers, power_name,
)

OUTDIR = os.path.join(_ROOT, "results", "edmd_family_benchmark")
DEGREE = 2          # KS advection is quadratic; Lorenz's target is quadratic


# ---------------------------------------------------------------------------
# Step 2: read the dictionary off the trained encoder
# ---------------------------------------------------------------------------
def discover_T(model, X, degree, n_samples=3000, n_coords=None, alpha=1e-4,
               rel=0.01, cap=None, device="cpu"):
    """Union of per-coordinate LassoLars supports.  Returns list of powers."""
    rng = np.random.default_rng(0)
    Xs = np.asarray(X[rng.choice(X.shape[0], min(n_samples, X.shape[0]),
                                 replace=False)], dtype=np.float64)
    with torch.no_grad():
        Z = model.ae.encoder(torch.as_tensor(Xs, dtype=torch.float32,
                                             device=device)).cpu().numpy()

    powers = monomial_powers(Xs.shape[1], degree)
    M = np.stack([np.prod(Xs ** np.asarray(p, dtype=np.float64), axis=1)
                  for p in powers], axis=1)
    mu, sd = M.mean(axis=0), M.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Msc = (M - mu) / sd

    coords = range(Z.shape[1]) if n_coords is None else \
        rng.choice(Z.shape[1], min(n_coords, Z.shape[1]), replace=False)

    freq, per_coord = {}, []
    for k in coords:
        f = Z[:, k].astype(np.float64)
        fs = f.std() or 1.0
        est = LassoLars(alpha=alpha, fit_intercept=True)
        est.fit(Msc, (f - f.mean()) / fs)
        c = np.abs(est.coef_)
        if c.max() <= 0:
            continue
        keep = np.where(c >= rel * c.max())[0]
        per_coord.append(len(keep))
        for i in keep:
            freq[powers[i]] = freq.get(powers[i], 0) + 1

    ordered = sorted(freq, key=lambda p: (-freq[p], sum(p), p))
    if cap:
        ordered = ordered[:cap]
    return ordered, {
        "n_union": len(freq), "n_kept": len(ordered),
        "n_library": len(powers),
        "per_coord_median": float(np.median(per_coord)) if per_coord else 0.0,
        "n_coords_scanned": len(per_coord),
    }


# ---------------------------------------------------------------------------
# ResDMD residuals for the EDMD operator on a given dictionary
# ---------------------------------------------------------------------------
def resdmd_residuals(Xtr, Ytr, powers, Xref):
    fm = FeatureMap(powers, Xref)
    PX, PY = fm(Xtr), fm(Ytr)
    n = PX.shape[0]
    G, A, L = PX.T @ PX / n, PX.T @ PY / n, PY.T @ PY / n
    K = np.linalg.solve(G + 1e-10 * np.eye(len(G)), A)
    lam, V = np.linalg.eig(K)
    res = []
    for i in range(len(lam)):
        g, l = V[:, i], lam[i]
        num = g.conj() @ (L - np.conj(l) * A - l * A.conj().T
                          + abs(l) ** 2 * G) @ g
        den = g.conj() @ G @ g
        res.append(float(np.sqrt(max(np.real(num / den), 0.0))))
    return np.array(lam), np.array(res)


# ---------------------------------------------------------------------------
def run_system(name, seed, device, smoke, use_cache=True,
               epochs_override=None):
    cfg = SYSTEMS[name]
    d, dt = cfg["state_dim"], cfg["dt"]
    epochs = 3 if smoke else (epochs_override or cfg["epochs"])

    train_traj, test_traj = get_data(cfg, seed, smoke)
    Xtr, Ytr = make_pairs(train_traj)
    Xte, Yte = make_pairs(test_traj)
    if smoke:
        Xtr, Ytr = Xtr[:2000], Ytr[:2000]
    std_ref = np.maximum(test_traj.reshape(-1, d).std(axis=0), 1e-8)
    flat = train_traj.reshape(-1, d)
    grl1 = [float(flat.min()) - 1.0, float(flat.max()) + 1.0]
    n_roll = 10 if smoke else (cfg["n_roll"] or
                               int(3 * (1.0 / cfg["lambda_max"]) / dt))
    n_roll = min(n_roll, test_traj.shape[1] - 1)
    n_ics = 4 if smoke else 32

    rows = []

    def record(method, po, ro, npar, wall, natoms, extra=None):
        def po_t(Xt):
            # EDMD predictors are numpy (FeatureMap np.asarray's the tensor);
            # EDMD-DL's is torch.  Either way, hand back a tensor.
            out = po(Xt)
            return out if torch.is_tensor(out) else \
                torch.as_tensor(np.asarray(out), dtype=torch.float32)
        raw, norm = one_step_errors(po_t, Xte, Yte)
        mn = nrmse_curve(ro, test_traj, n_roll, n_ics, std_ref)
        cr = horizon_crossings(mn, dt, cfg["lambda_max"])
        row = dict(system=name, method=method, seed=seed, n_atoms=natoms,
                   one_step_nmse=norm, vpt_05_steps=cr["vpt_0.5_steps"],
                   n_params=npar, wall_s=round(wall, 1), n_roll=n_roll)
        row.update(extra or {})
        rows.append(row)
        print(f"  [{method:>10}] atoms={natoms:>4} nmse={norm:.3e} "
              f"VPT0.5={cr['vpt_0.5_steps']:>5.0f} wall={wall:.0f}s")
        return row

    # -- 1. train Deep-Koopman-KANDy, 2. read off its dictionary ------------
    print(f"\n=== {name} (seed {seed}) ===")
    model = OursModel(d, cfg["mid"], cfg["latent"], grl1).to(device)
    ckpt = os.path.join(OUTDIR, f"model_{name}_s{seed}_e{epochs}.pt")

    if use_cache and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  loaded cached model from {ckpt}\n"
              f"  (pass --no-cache to retrain)")
    else:
        print(f"  training Deep-Koopman-KANDy: {epochs} epochs on "
              f"{len(Xtr):,} pairs -- this is the slow step "
              f"(~{'8 min' if name == 'lorenz' else '2 min'} on CPU)")

        def show(ep, tot, loss, el):
            every = max(1, tot // 20)
            if ep % every == 0 or ep == tot:
                eta = el / ep * (tot - ep)
                print(f"    epoch {ep:>4}/{tot}  loss {loss:.3e}  "
                      f"{el:.0f}s elapsed, ~{eta:.0f}s left", flush=True)

        t0 = time.time()
        train_ae_koopman(model, Xtr, Ytr, dt, epochs, 1e-3, 512, device, seed,
                         progress=show)
        print(f"  trained in {time.time() - t0:.0f}s")
        os.makedirs(OUTDIR, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
    model.eval()

    # KS: 64 state dims -> 2145 degree-2 atoms; scan a subset of coordinates
    # and cap the union so the benchmark stays quick.
    kw = dict(n_coords=32, cap=200) if name == "ks" else {}
    if smoke:
        kw = dict(n_coords=4, cap=25)
    t0 = time.time()
    T, info = discover_T(model, Xtr, DEGREE, device=device, **kw)
    print(f"  T: {info['n_kept']} atoms (union {info['n_union']} of "
          f"{info['n_library']}; median {info['per_coord_median']:.0f}/coord "
          f"over {info['n_coords_scanned']} coords) in {time.time() - t0:.0f}s")

    # -- 3. the three methods ----------------------------------------------
    po, ro, npar, w, na = make_edmd(Xtr, Ytr, T, Xtr)
    record("EDMD(T)", po, ro, npar, w, na)

    full = monomial_powers(d, DEGREE)
    if len(full) <= 400:
        po, ro, npar, w, na = make_edmd(Xtr, Ytr, full, Xtr)
        record("EDMD(poly)", po, ro, npar, w, na)
    else:
        print(f"  [EDMD(poly)] skipped: {len(full)} atoms is not a quick fit")

    cfg_dl = dict(cfg)
    cfg_dl["edmd_lift"] = max(4, len(T) - d)      # size-match to |T|
    po, ro, npar, w = run_edmd_dl(Xtr, Ytr, cfg_dl, epochs, 1e-3, device, seed)
    record("EDMD-DL", po, ro, npar, w, cfg_dl["edmd_lift"] + d)

    lam, res = resdmd_residuals(Xtr, Ytr, T, Xtr)
    cert = float(np.mean(res < 0.1))
    print(f"  [  ResDMD(T)] {len(lam)} eigenpairs, residual median "
          f"{np.median(res):.3f} min {res.min():.3f}; "
          f"{cert * 100:.0f}% certified at tol 0.1")
    rows.append(dict(system=name, method="ResDMD(T)", seed=seed,
                     n_atoms=len(T), resdmd_res_median=float(np.median(res)),
                     resdmd_res_min=float(res.min()), resdmd_frac_cert=cert))
    return rows, {"T": [power_name(p, [f"x{i}" for i in range(d)]) for p in T],
                  **info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["lorenz", "ks"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override training epochs (default: 200 lorenz / "
                         "100 ks). The encoder training is ~95%% of runtime.")
    ap.add_argument("--no-cache", action="store_true",
                    help="retrain even if a cached model exists")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)   # never look stuck in a pipe

    all_rows, dicts = [], {}
    for name in args.systems:
        for seed in range(args.seeds):
            r, info = run_system(name, seed, args.device, args.smoke,
                                 use_cache=not args.no_cache,
                                 epochs_override=args.epochs)
            all_rows.extend(r)
            dicts[f"{name}_s{seed}"] = info

    import csv
    fields = ["system", "method", "seed", "n_atoms", "one_step_nmse",
              "vpt_05_steps", "n_params", "wall_s", "n_roll",
              "resdmd_res_median", "resdmd_res_min", "resdmd_frac_cert"]
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    with open(os.path.join(OUTDIR, "dictionaries.json"), "w") as f:
        json.dump(dicts, f, indent=2)

    print(f"\nWrote {OUTDIR}/results.csv")
    print(f"\n{'system':>8} {'method':>11} {'atoms':>6} {'1step nMSE':>12} "
          f"{'VPT0.5':>7}")
    for r in all_rows:
        if "one_step_nmse" not in r:
            continue
        print(f"{r['system']:>8} {r['method']:>11} {r['n_atoms']:>6} "
              f"{r['one_step_nmse']:>12.3e} {r['vpt_05_steps']:>7.0f}")


if __name__ == "__main__":
    main()
