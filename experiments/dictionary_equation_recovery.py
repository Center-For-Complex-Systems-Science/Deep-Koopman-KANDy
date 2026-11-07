"""Did SINDy and KANDy recover the governing equations from T?

The discovered-dictionary ablation scores prediction.  This script scores
*recovery*: given the discovered dictionary T (and the deg-D and target
libraries as controls), does the downstream method return the Lorenz system

    xdot = sigma (y - x)          sigma = 10
    ydot = x (rho - z) - y        rho   = 28
    zdot = x y - beta z           beta  = 8/3

SINDy here is the STLSQ implementation in discovered_dictionary_ablation.py
(pysindy 1.7.5 is broken against the installed scikit-learn 1.8.0; see the note
there).  For SINDy this is a direct read of the coefficient matrix: support
(which atoms are active) and values (how close to truth).

For KANDy the question is subtler.  A KAN layer is a sum of univariate spline
edges, so it is only *symbolic* if those edges are affine -- then the layer is
a linear regression on the library and its slopes are the equation
coefficients.  We therefore evaluate every edge phi_ij on the data range (the
layer is additive, so phi_ij is recovered exactly by varying feature j alone
from the zero baseline), score how affine it is, and read the implied
coefficient from the fitted slope.  A KAN layer with strongly non-affine edges
may predict well while encoding no closed-form equation at all.

    python experiments/dictionary_equation_recovery.py [--seeds 3]
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
warnings.filterwarnings("ignore")

from benchmark_prior_art import SYSTEMS, get_data, make_pairs   # noqa: E402
from data import lorenz_rhs                                     # noqa: E402
from discovered_dictionary_ablation import (                    # noqa: E402
    DEGREE, ORACLE, VAR_NAMES, FeatureMap, monomial_powers, power_name,
    stlsq,
)

ABL = os.path.join(_ROOT, "results", "discovered_dictionary_ablation")
OUTDIR = os.path.join(_ROOT, "results", "dictionary_equation_recovery")

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
# True Lorenz coefficients keyed by (power tuple) -> per-equation value
TRUE = {
    (1, 0, 0): (-SIGMA, RHO, 0.0),      # x
    (0, 1, 0): (SIGMA, -1.0, 0.0),      # y
    (0, 0, 1): (0.0, 0.0, -BETA),       # z
    (1, 0, 1): (0.0, -1.0, 0.0),        # x z
    (1, 1, 0): (0.0, 0.0, 1.0),         # x y
}
EQ_NAMES = ["xdot", "ydot", "zdot"]


def parse_atom(name, var_names):
    """'x^2*y' -> (2, 1, 0)"""
    p = [0] * len(var_names)
    for part in name.split("*"):
        part = part.strip()
        if not part or part == "1":
            continue
        if "^" in part:
            v, e = part.split("^")
            p[var_names.index(v)] += int(e)
        else:
            p[var_names.index(part)] += 1
    return tuple(p)


def load_discovered(system, seed, var_names):
    path = os.path.join(ABL, "dictionaries.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path)).get(f"{system}_s{seed}")
    if d is None:
        return None
    return [parse_atom(a, var_names) for a in d["atoms"]]


def sindy_fit(Theta, target, thresholds=(1e-4, 1e-3, 1e-2, 1e-1)):
    """Same STLSQ selection as the ablation: threshold picked on held-out data."""
    n = Theta.shape[0]
    cut = int(0.8 * n)
    best, best_err = None, np.inf
    for thr in thresholds:
        Xi = stlsq(Theta[:cut], target[:cut], thr)
        if not np.any(np.abs(Xi) > 0):
            continue
        err = float(np.mean((Theta[cut:] @ Xi - target[cut:]) ** 2))
        if np.isfinite(err) and err < best_err:
            best, best_err = (Xi, thr), err
    return best if best else (np.linalg.lstsq(Theta, target, rcond=None)[0], 0.0)


def score_equations(powers, coefs, var_names, tol_rel=0.02, active_tol=1e-3):
    """Compare a coefficient matrix against the true Lorenz equations."""
    idx = {tuple(p): i for i, p in enumerate(powers)}
    rows, exact = [], True
    for e in range(3):
        true_terms = {p: v[e] for p, v in TRUE.items() if abs(v[e]) > 0}
        got = {}
        for p, i in idx.items():
            if sum(p) == 0:
                continue
            c = float(coefs[i, e])
            if abs(c) > active_tol:
                got[p] = c
        fp = sorted(set(got) - set(true_terms))
        fn = sorted(set(true_terms) - set(got))
        errs = [abs(got[p] - v) / abs(v) for p, v in true_terms.items() if p in got]
        rows.append({
            "eq": EQ_NAMES[e],
            "n_active": len(got),
            "false_pos": [power_name(p, var_names) for p in fp],
            "false_neg": [power_name(p, var_names) for p in fn],
            "max_rel_err": max(errs) if errs else float("nan"),
            "terms": {power_name(p, var_names): round(c, 4)
                      for p, c in sorted(got.items(), key=lambda kv: -abs(kv[1]))},
        })
        if fp or fn or (errs and max(errs) > tol_rel):
            exact = False
    return rows, exact


def kandy_edges(model, fm, X, powers, var_names, n_grid=101):
    """Recover each univariate edge of the KAN layer and score its affineness.

    The layer is additive in its inputs, so phi_ij(t) = f(e_j t) - f(0) exactly.
    Returns per-edge affine R^2 and the implied raw-space coefficient.
    """
    F = fm(X)
    lo, hi = F.min(axis=0), F.max(axis=0)
    n_feat = F.shape[1]
    dev = next(model.parameters()).device

    with torch.no_grad():
        base = model(torch.zeros(1, n_feat, dtype=torch.float32, device=dev)
                     ).cpu().numpy()[0]

    r2 = np.zeros((n_feat, base.shape[0]))
    slope = np.zeros((n_feat, base.shape[0]))
    for j in range(n_feat):
        grid = np.linspace(lo[j], hi[j], n_grid)
        probe = np.zeros((n_grid, n_feat), dtype=np.float32)
        probe[:, j] = grid
        with torch.no_grad():
            phi = model(torch.as_tensor(probe, device=dev)).cpu().numpy() - base
        for e in range(base.shape[0]):
            y = phi[:, e]
            A = np.stack([grid, np.ones_like(grid)], axis=1)
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ beta
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2[j, e] = 1.0 if ss_tot < 1e-18 else 1 - np.sum((y - pred) ** 2) / ss_tot
            # fm scales columns by fm.scale; slope is w.r.t. the scaled feature
            slope[j, e] = beta[0] / fm.scale[j]
    return r2, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kandy-epochs", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    system = "lorenz"
    cfg = SYSTEMS[system]
    var_names = VAR_NAMES[system]
    degree = DEGREE[system]
    report = {}

    for seed in range(args.seeds):
        train_traj, _ = get_data(cfg, seed, smoke=False)
        Xtr, _ = make_pairs(train_traj)
        xdot = lorenz_rhs(torch.as_tensor(Xtr, dtype=torch.float64)).numpy()

        libs = {"poly": monomial_powers(3, degree),
                "target": [tuple(p) for p in ORACLE[system]]}
        disc = load_discovered(system, seed, var_names)
        if disc:
            libs["disc"] = disc

        print(f"\n================ Lorenz, seed {seed} ================")
        for tag in ("disc", "poly", "target"):
            if tag not in libs:
                continue
            powers = libs[tag]
            fm = FeatureMap(powers, Xtr)
            Theta = fm(Xtr)
            Xi_scaled, thr = sindy_fit(Theta, xdot)
            # undo the column scaling to get raw-space coefficients
            Xi = Xi_scaled / fm.scale[:, None]
            rows, exact = score_equations(fm.powers, Xi, var_names)

            print(f"\n-- SINDy / {tag} library ({fm.n_atoms} atoms, "
                  f"STLSQ threshold {thr:g}) -> "
                  f"{'EXACT' if exact else 'not exact'}")
            for r in rows:
                terms = "  ".join(f"{c:+.4f}*{n}" for n, c in r["terms"].items())
                print(f"   {r['eq']} = {terms}")
                if r["false_pos"] or r["false_neg"]:
                    print(f"        extra={r['false_pos']} missing={r['false_neg']}")
                print(f"        max relative coefficient error: "
                      f"{r['max_rel_err']:.2e}")
            report[f"sindy_{tag}_seed{seed}"] = {"exact": exact, "rows": rows,
                                                 "threshold": thr}

        # ---- KANDy on the discovered library -----------------------------
        tag = "disc" if "disc" in libs else "poly"
        powers = libs[tag]
        # Train the same layer the ablation trains, but keep the module so its
        # edges can be inspected (make_kandy returns only predictors).
        from discovered_dictionary_ablation import KandyLayer
        fm = FeatureMap(powers, Xtr)
        F = fm(Xtr)
        mu, sd = F.mean(axis=0), F.std(axis=0)
        sd[sd < 1e-8] = 1.0
        torch.manual_seed(seed)
        np.random.seed(seed)
        kan = KandyLayer(F.shape[1], 3, mu, sd).to(args.device)
        Ft = torch.as_tensor(F, dtype=torch.float32, device=args.device)
        Yt = torch.as_tensor(xdot, dtype=torch.float32, device=args.device)
        opt = torch.optim.AdamW(kan.parameters(), lr=1e-3)
        for _ in range(args.kandy_epochs):
            perm = torch.randperm(Ft.shape[0], device=args.device)
            for s in range(0, Ft.shape[0], 512):
                i = perm[s:s + 512]
                loss = ((kan(Ft[i]) - Yt[i]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(kan.parameters(), 5.0)
                opt.step()
        kan.eval()
        r2, slope = kandy_edges(kan, fm, Xtr, powers, var_names)

        print(f"\n-- KANDy / {tag} library ({fm.n_atoms} atoms) edge analysis")
        aff = []
        for e in range(3):
            live = [j for j in range(len(fm.powers))
                    if sum(fm.powers[j]) > 0 and abs(slope[j, e]) > 1e-3]
            if live:
                worst = min(r2[j, e] for j in live)
                med = float(np.median([r2[j, e] for j in live]))
            else:
                worst, med = float("nan"), float("nan")
            aff.append((med, worst))
            terms = sorted(live, key=lambda j: -abs(slope[j, e]))[:6]
            desc = "  ".join(f"{slope[j, e]:+.3f}*{power_name(fm.powers[j], var_names)}"
                             f"(R2af={r2[j, e]:.3f})" for j in terms)
            print(f"   {EQ_NAMES[e]}: {len(live)} live edges, affine R2 "
                  f"median {med:.4f} worst {worst:.4f}")
            print(f"        {desc}")
        # If the edges are affine the layer IS a linear model: score its slopes
        rows_k, exact_k = score_equations(fm.powers, slope, var_names)
        print(f"   implied-equation check: {'EXACT' if exact_k else 'not exact'}")
        for r in rows_k:
            if r["false_pos"] or r["false_neg"]:
                print(f"     {r['eq']}: extra={r['false_pos']} "
                      f"missing={r['false_neg']} "
                      f"max_rel_err={r['max_rel_err']:.2e}")
            else:
                print(f"     {r['eq']}: support exact, "
                      f"max_rel_err={r['max_rel_err']:.2e}")
        report[f"kandy_{tag}_seed{seed}"] = {
            "exact": exact_k, "rows": rows_k,
            "affine_r2_median": [a[0] for a in aff],
            "affine_r2_worst": [a[1] for a in aff],
        }

    with open(os.path.join(OUTDIR, "recovery.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nWrote {os.path.join(OUTDIR, 'recovery.json')}")


if __name__ == "__main__":
    main()
