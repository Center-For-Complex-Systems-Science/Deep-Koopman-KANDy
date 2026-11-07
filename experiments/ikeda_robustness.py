"""
Ikeda map: readout robustness study  (NeurIPS reviewer EbT5-Q4).

Question under test
-------------------
"How sensitive are the recovered dictionaries to the choice of Lasso
regularization, polynomial degree, readout basis, and pruning threshold,
particularly on the non-polynomial systems?"

Ikeda is the paper's misspecified-readout showcase: the map depends on
r^2 = x^2 + y^2 only through cos/sin of t(r^2) = 0.4 - 6/(1 + r^2), so a
degree-3 polynomial Lasso readout is wrong by construction.  The paper's claim
(Table tab:ikeda-levelset-decomposition) is that this degrades *parsimony*, not
*expressivity*: R^2(h o g) stays in [0.92, 0.99] and the foliation coordinate is
still the radial one.

The claim this script tests is the stronger, sweep-level version:

    the recovered foliation coordinate g is STRUCTURALLY stable across
    lambda / degree / basis / prune-tau; what moves is only the density of g
    and the complexity of h.

Axes
----
  A. Lasso lambda in {1e-6 ... 1e-2}            (post-hoc, degree 3, poly basis)
  B. polynomial degree D in {1..5}              (post-hoc, alpha = 1e-3)
  C. readout basis: poly (misspecified, paper default)
                  / radial-trig (correctly specified)
                  / Gaussian RBF                (post-hoc, alpha = 1e-3)
  D. PyKAN-style prune threshold tau (prune -> retrain 100 ep @ 5e-4 ->
     re-decompose; the Lorenz tab:prune-sweep protocol).  The Lorenz grid
     {0, .01, .03, .05} is extended to {.1, .2, .3, .5} because this encoder has
     no edge with an attribution score inside the Lorenz range.  The score is
     PyKAN's normalized edge attribution B_ij, reimplemented for KANLinear in
     experiments/kan_attribution.py -- NOT the unnormalized spline-weight
     magnitude that ikeda_minimal.prune_model uses, which lives on a different
     scale and would make tau incomparable to Lorenz.

Axes A-C are post-hoc: they re-run the level-set decomposition on an already
trained encoder.  No retraining.  Only axis D retrains.

Structural-stability metrics (basis independent, so axis C is comparable)
------------------------------------------------------------------------
  R2_radial   : R^2 of regressing g onto a degree-4 polynomial in r^2.
                "Is g still a function of the radial coordinate alone?"
  |rho|(g,r2) : |Spearman| between g and r^2 -- monotone version of the above,
                immune to how the outer h reparametrizes the level sets.
  quad_ratio  : min(|c_x2|,|c_y2|) / max(|c_x2|,|c_y2|) with the same-sign flag.
                This is the paper's own metric (reported as 0.66 for z_1) and
                only exists for the polynomial basis.

Cost metrics (reviewer question on "adding too many candidate functions")
-------------------------------------------------------------------------
  P           : design-matrix column count BEFORE Lasso
  t_decomp    : wall time of the decomposition step (Lasso + gradient identity
                + h recovery), total and per latent coordinate

Run:
    python experiments/ikeda_robustness.py                # trains base model
    python experiments/ikeda_robustness.py --skip-train   # reuse checkpoint
    python experiments/ikeda_robustness.py --skip-tau     # axes A-C only
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training import set_seed
from ikeda import generate_ikeda, IkedaPairDataset
import ikeda_minimal as IM
from readout_bases import (
    poly_basis, ikeda_radial_trig_basis, rbf_basis,
    decompose_with_basis, r2_against_span, spearman_abs,
)
from kan_attribution import edge_attribution, score_summary

# =====================================================================
# Configuration
# =====================================================================

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'results' / 'ikeda_robustness'

SEED = 0                    # paper's "best seed" for the Ikeda level-set figure
N_ANALYSIS_PTS = 10000      # matches experiments/ikeda_levelset_analysis.py
DATA_SEED_TRAIN = 42        # generate_ikeda seed used by ikeda_minimal.py
DATA_SEED_ANALYSIS = 0      # generate_ikeda seed used by ikeda_levelset_analysis.py

# Paper defaults for this system
PAPER_ALPHA = 1e-3
PAPER_DEGREE = 3
H_DEGREE = 3                # ikeda_levelset_analysis.py passes h_degree=3
SPARSITY_THRESHOLD = 1e-4

# Axis A: Lasso lambda (mirrors the Lorenz tab:lasso-sweep grid)
ALPHA_SWEEP = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
# Axis B: polynomial degree
DEGREE_SWEEP = [1, 2, 3, 4, 5]
# Axis C: readout basis
BASIS_SWEEP = ['poly', 'radial_trig', 'rbf']
RBF_PER_DIM = 6             # 6x6 = 36 centers + constant -> P = 37
# Axis D: prune threshold (Lorenz protocol: prune, retrain, re-decompose).
# The first four values are the Lorenz tab:prune-sweep grid.  They are extended
# past 0.05 because the theory-matched [2,4,4] encoder turns out to have NO edge
# whose PyKAN attribution score falls in the Lorenz range (min score ~0.19), so
# the Lorenz grid alone would prune nothing and the axis would be uninformative.
TAU_SWEEP = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
LORENZ_TAU_GRID = [0.0, 0.01, 0.03, 0.05]
RETRAIN_EPOCHS = 100
RETRAIN_LR = 5e-4

# The paper's "target" monomials for the Ikeda inner function; anything else
# active in the polynomial basis is counted as a false positive / extra atom.
IKEDA_TARGET_ATOMS = {'x', 'y', 'x^2', 'y^2', 'xy'}

# Paper's headline numbers to reproduce (tab:ikeda-levelset-decomposition)
PAPER_R2_HG = [0.915, 0.941, 0.986, 0.968]
PAPER_QUAD_RATIO = 0.66     # reported for z_1

torch.set_num_threads(12)


# =====================================================================
# Base model: train or load
# =====================================================================

def train_base_model(traj_torch, seed=SEED):
    """Paper config: [2,4,4] KANLinear encoder/decoder + stable propagator.

    This is exactly experiments/ikeda_minimal.py's train_one_seed, called here
    so the robustness study is self-contained and seeded.
    """
    return IM.train_one_seed(seed, traj_torch)


def load_or_train(traj_torch, skip_train):
    ckpt_path = OUTDIR / f'base_model_seed{SEED}.pt'
    if skip_train and ckpt_path.exists():
        print(f"Loading base model from {ckpt_path}")
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = IM.IkedaMinimalKoopman()
        model.load_state_dict(ck['model_state_dict'])
        return model.eval()
    print(f"Training base model [2,{IM.MID_DIM},{IM.LATENT_DIM}] seed {SEED} "
          f"({IM.EPOCHS} epochs)...")
    model = train_base_model(traj_torch)
    torch.save({'model_state_dict': model.state_dict(), 'seed': SEED,
                'config': {'mid': IM.MID_DIM, 'latent': IM.LATENT_DIM,
                           'epochs': IM.EPOCHS, 'lr': IM.LR}},
               ckpt_path)
    print(f"Saved {ckpt_path}")
    return model.eval()


# =====================================================================
# Encoder gradients (chart = the state (x, y) itself)
# =====================================================================

def encoder_grad(model, pts, target_idx, chunk=2000):
    """d z_k / d(x, y) and z_k values, via autograd."""
    gs, fs = [], []
    for i in range(0, len(pts), chunk):
        xi = torch.tensor(pts[i:i + chunk], dtype=torch.float32,
                          requires_grad=True)
        fi = model.encoder(xi)[:, target_idx]
        fi.sum().backward()
        gs.append(xi.grad.detach().numpy())
        fs.append(fi.detach().numpy())
    return np.concatenate(gs), np.concatenate(fs)


def precompute_gradients(model, pts, latent_dim):
    """Gradients/values for every latent dim; reused across all settings."""
    return [encoder_grad(model, pts, k) for k in range(latent_dim)]


# =====================================================================
# One sweep setting
# =====================================================================

def build_basis(kind, pts, degree):
    if kind == 'poly':
        return poly_basis(pts, ['x', 'y'], max_degree=degree)
    if kind == 'radial_trig':
        return ikeda_radial_trig_basis(pts)
    if kind == 'rbf':
        return rbf_basis(pts, n_per_dim=RBF_PER_DIM, eps_scale=1.0)
    raise ValueError(kind)


def run_setting(basis, grads, pts, alpha, solver='cd'):
    """Decompose every latent dim under one (basis, alpha) setting.

    Returns (per_dim_rows, aggregate_row).
    """
    r2_ref_cols = np.column_stack([  # degree-4 polynomial in r^2
        (pts[:, 0] ** 2 + pts[:, 1] ** 2) ** k for k in (1, 2, 3, 4)])
    r2_true = pts[:, 0] ** 2 + pts[:, 1] ** 2

    rows = []
    t_total = 0.0
    for k, (grad_f, f_vals) in enumerate(grads):
        t0 = time.perf_counter()
        res = decompose_with_basis(basis, f_vals, grad_f, alpha,
                                   sparsity_threshold=SPARSITY_THRESHOLD,
                                   h_degree=H_DEGREE, solver=solver)
        dt = time.perf_counter() - t0
        t_total += dt

        names = res['names']
        coeffs = res['coeffs']
        cx2 = cy2 = float('nan')
        if 'x^2' in names and 'y^2' in names:
            cx2 = float(coeffs[names.index('x^2')])
            cy2 = float(coeffs[names.index('y^2')])
        if np.isnan(cx2):
            quad_ratio = float('nan')
            same_sign = False
        else:
            quad_ratio = (min(abs(cx2), abs(cy2))
                          / max(abs(cx2), abs(cy2), 1e-12))
            same_sign = bool(cx2 * cy2 > 0)
        radial_sig = bool(same_sign and quad_ratio > 0.5
                          and abs(cx2) > 0.01 and abs(cy2) > 0.01)

        extra = [n for n in res['active_names']
                 if n != '1' and n not in IKEDA_TARGET_ATOMS]
        n_fp = len(extra) if basis_is_poly(names) else float('nan')

        rows.append({
            'dim': k,
            'r2_g': res['r2_g'],
            'r2_hg': res['r2_hg'],
            'n_active': res['n_active'],
            'n_dominant': res['n_dominant'],
            'used_affine': int(res['used_affine']),
            'degenerate': int(res['degenerate']),
            'c_x2': cx2, 'c_y2': cy2,
            'quad_ratio': quad_ratio,
            'same_sign': int(same_sign),
            'radial_signature': int(radial_sig),
            'R2_radial': r2_against_span(res['g_vals'], r2_ref_cols),
            'spearman_g_r2': spearman_abs(res['g_vals'], r2_true),
            'n_fp': n_fp,
            'top_atoms': ';'.join(
                f"{n}({c:+.4f})" for n, c in sorted(
                    zip(res['active_names'], res['active_coeffs']),
                    key=lambda p: -abs(p[1]))[:6]),
            't_decomp_s': dt,
        })

    agg = aggregate(rows)
    agg['t_decomp_total_s'] = t_total
    agg['t_decomp_per_dim_s'] = t_total / max(len(rows), 1)
    return rows, agg


def basis_is_poly(names):
    return 'x^2' in names or names == ['1', 'y', 'x']


def _med(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.median(vals)) if vals else float('nan')


def aggregate(rows):
    r2hg = [r['r2_hg'] for r in rows]
    return {
        'r2_hg_median': _med(r2hg),
        'r2_hg_min': float(np.min(r2hg)),
        'r2_hg_max': float(np.max(r2hg)),
        'n_dims_r2hg_gt_0.9': int(sum(1 for v in r2hg if v > 0.9)),
        'n_active_median': _med([r['n_active'] for r in rows]),
        'n_dominant_median': _med([r['n_dominant'] for r in rows]),
        'R2_radial_median': _med([r['R2_radial'] for r in rows]),
        'R2_radial_max': float(np.max([r['R2_radial'] for r in rows])),
        'spearman_median': _med([r['spearman_g_r2'] for r in rows]),
        'spearman_max': float(np.max([r['spearman_g_r2'] for r in rows])),
        'n_radial_signature': int(sum(r['radial_signature'] for r in rows)),
        'quad_ratio_best': (float(np.nanmax([r['quad_ratio'] for r in rows]))
                            if not all(np.isnan(r['quad_ratio']) for r in rows)
                            else float('nan')),
        'n_fp_median': _med([r['n_fp'] for r in rows]),
        'n_affine_h': int(sum(r['used_affine'] for r in rows)),
        'n_degenerate': int(sum(r['degenerate'] for r in rows)),
    }


# =====================================================================
# Axis D: prune + retrain + re-decompose
# =====================================================================

def one_step_mse(model, traj_torch, n_val_traj=40):
    """One-step state prediction MSE on the held-out trajectories."""
    dl = DataLoader(IkedaPairDataset(traj_torch[-n_val_traj:]), batch_size=4096)
    model.eval()
    se, n = 0.0, 0
    with torch.no_grad():
        for a, b in dl:
            pred = model.forward_one_step(a)['xy_np1_pred']
            se += float(((pred - b) ** 2).sum())
            n += b.numel()
    return se / n


def encoder_layers(model):
    return [model.encoder.kan1, model.encoder.kan2]


def attribution_scores(model, pts, n=4000):
    """PyKAN-style edge attribution B_ij for the encoder (paper's tau score).

    See experiments/kan_attribution.py for why the raw KANLinear spline-weight
    magnitude cannot be used with the Lorenz tau grid.
    """
    x = torch.tensor(pts[:n], dtype=torch.float32)
    scores = edge_attribution(encoder_layers(model), x)
    return {f'kan{i+1}': s for i, s in enumerate(scores)}


def prune_at(model, tau, pts):
    """Zero every encoder edge whose attribution score is <= tau.

    Matches PyKAN's prune_edge (mask = score > tau) and hence the Lorenz
    tab:prune-sweep protocol, unlike ikeda_minimal.prune_model's unnormalized
    spline-weight-magnitude proxy.
    """
    scores = attribution_scores(model, pts)
    n_pruned, n_total = 0, 0
    with torch.no_grad():
        for layer, sc in zip(encoder_layers(model), scores.values()):
            mask = torch.tensor(sc <= tau)
            for oi in range(layer.out_features):
                for ii in range(layer.in_features):
                    n_total += 1
                    if mask[oi, ii]:
                        layer.spline_weight[oi, ii, :] = 0.0
                        if getattr(layer, 'spline_scaler', None) is not None:
                            layer.spline_scaler[oi, ii] = 0.0
                        layer.base_weight[oi, ii] = 0.0
                        n_pruned += 1
    return n_pruned, n_total, scores


def retrain(model, traj_torch, epochs=RETRAIN_EPOCHS, lr=RETRAIN_LR):
    n_train = int(0.8 * traj_torch.shape[0])
    dl = DataLoader(IkedaPairDataset(traj_torch[:n_train]),
                    batch_size=IM.BATCH_SIZE, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=IM.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()
    for ep in range(1, epochs + 1):
        model.train()
        for a, b in dl:
            out = model.forward_one_step(a)
            z_true = model.encoder(b)
            loss = (IM.GAMMA_PREDX * mse(out['xy_np1_pred'], b)
                    + IM.ALPHA_LATENT * mse(out['z_np1_pred'], z_true)
                    + IM.BETA_RECON * mse(out['xy_n_rec'], a)
                    + IM.L1_SPLINE_WEIGHT * IM.spline_l1_reg(model))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), IM.GRAD_CLIP)
            opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs:
            print(f"      retrain ep {ep:3d}  loss {float(loss):.5f}", flush=True)
    return model.eval()


# =====================================================================
# I/O helpers
# =====================================================================

def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval='')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}")


def fmt(v, n=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '--'
    return f"{v:.{n}f}"


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-tau', action='store_true')
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    # ---- data -----------------------------------------------------
    traj_train = torch.from_numpy(generate_ikeda(
        n_traj=IM.N_TRAJ, n_steps=IM.N_STEPS, burn_in=IM.BURN_IN,
        seed=DATA_SEED_TRAIN))
    traj_analysis = generate_ikeda(n_traj=IM.N_TRAJ, n_steps=IM.N_STEPS,
                                   burn_in=IM.BURN_IN, seed=DATA_SEED_ANALYSIS)
    flat = traj_analysis.reshape(-1, 2)
    rng = np.random.RandomState(42)
    pts = flat[rng.permutation(flat.shape[0])[:N_ANALYSIS_PTS]].astype(np.float64)
    print(f"Analysis points: {pts.shape}  x[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
          f"y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]")

    # ---- base model ------------------------------------------------
    model = load_or_train(traj_train, args.skip_train)
    latent_dim = IM.LATENT_DIM
    base_mse = one_step_mse(model, traj_train)
    print(f"Base one-step MSE: {base_mse:.6e}")

    grads = precompute_gradients(model, pts, latent_dim)

    # ==================================================================
    # Reproduction check against tab:ikeda-levelset-decomposition
    # ==================================================================
    print("\n" + "=" * 72)
    print("REPRODUCTION CHECK  (alpha=1e-3, degree 3, polynomial basis)")
    print("=" * 72)
    basis_paper = build_basis('poly', pts, PAPER_DEGREE)
    repro_rows, repro_agg = run_setting(basis_paper, grads, pts, PAPER_ALPHA)
    print(f"{'dim':>4} {'R2(g)':>8} {'R2(hog)':>8} {'#act':>5} {'affine':>7} "
          f"{'c_x2':>9} {'c_y2':>9} {'ratio':>6} {'R2_rad':>7} {'|rho|':>6}")
    for r in repro_rows:
        print(f"z[{r['dim']}] {r['r2_g']:8.3f} {r['r2_hg']:8.3f} "
              f"{r['n_active']:5d} {'yes' if r['used_affine'] else 'no':>7} "
              f"{r['c_x2']:+9.4f} {r['c_y2']:+9.4f} {r['quad_ratio']:6.3f} "
              f"{r['R2_radial']:7.3f} {r['spearman_g_r2']:6.3f}")
    got = sorted(r['r2_hg'] for r in repro_rows)
    want = sorted(PAPER_R2_HG)
    in_band = all(0.90 <= v <= 0.995 for v in got)
    best_ratio = repro_agg['quad_ratio_best']
    print(f"\n  paper R2(hog) sorted: {[f'{v:.3f}' for v in want]}")
    print(f"  ours  R2(hog) sorted: {[f'{v:.3f}' for v in got]}")
    print(f"  all in [0.90, 0.995]? {in_band}")
    print(f"  best matched-quadratic ratio: {fmt(best_ratio)} (paper: {PAPER_QUAD_RATIO})")
    print(f"  dims with radial signature: {repro_agg['n_radial_signature']}/{latent_dim}")
    print(f"  median R2_radial: {fmt(repro_agg['R2_radial_median'])}  "
          f"max: {fmt(repro_agg['R2_radial_max'])}")

    reproduced = in_band and repro_agg['n_radial_signature'] >= 1
    print(f"\n  REPRODUCED: {reproduced}")
    if not reproduced:
        print("  WARNING: base model does not match the paper's headline "
              "decomposition. Sweep numbers below describe THIS model, and the "
              "discrepancy is reported rather than smoothed over.")

    write_csv(OUTDIR / 'reproduction_per_dim.csv', repro_rows)

    all_rows = []      # aggregate rows, one per setting
    detail_rows = []   # per-dim rows, all settings

    def record(axis, setting_label, extra, rows, agg, basis, alpha):
        agg_row = {'axis': axis, 'setting': setting_label,
                   'P': basis.n_atoms, 'alpha': alpha, **extra, **agg}
        all_rows.append(agg_row)
        for r in rows:
            detail_rows.append({'axis': axis, 'setting': setting_label,
                                'P': basis.n_atoms, 'alpha': alpha, **r})
        return agg_row

    # ==================================================================
    # Axis A: Lasso lambda
    # ==================================================================
    print("\n" + "=" * 72)
    print("AXIS A: Lasso lambda  (degree 3, polynomial basis)")
    print("=" * 72)
    axis_a = []
    for alpha in ALPHA_SWEEP:
        rows, agg = run_setting(basis_paper, grads, pts, alpha)
        a = record('lambda', f'{alpha:g}', {'degree': PAPER_DEGREE,
                                            'basis': 'poly'},
                   rows, agg, basis_paper, alpha)
        axis_a.append(a)
        print(f"  a={alpha:>8.0e} P={basis_paper.n_atoms:3d} "
              f"R2hg med={fmt(agg['r2_hg_median'])} min={fmt(agg['r2_hg_min'])} "
              f"|#act={agg['n_active_median']:.1f} FP={fmt(agg['n_fp_median'],1)} "
              f"|R2rad med={fmt(agg['R2_radial_median'])} max={fmt(agg['R2_radial_max'])} "
              f"|rho med={fmt(agg['spearman_median'])} "
              f"|radial-sig={agg['n_radial_signature']} "
              f"|affine h={agg['n_affine_h']} "
              f"|t={agg['t_decomp_total_s']:.2f}s")
    write_csv(OUTDIR / 'axis_lambda.csv', axis_a)

    # LassoLars cross-check at the paper setting (cheap sanity check that the
    # coordinate-descent density is not a solver artifact -- it is for n=64
    # systems, but should not be at P<=35).
    rows_l, agg_l = run_setting(basis_paper, grads, pts, PAPER_ALPHA,
                                solver='lars')
    print(f"\n  LassoLars cross-check @ a={PAPER_ALPHA:g}: "
          f"#act median={agg_l['n_active_median']:.1f} "
          f"(CD: {axis_a[ALPHA_SWEEP.index(PAPER_ALPHA)]['n_active_median']:.1f}), "
          f"R2hg median={fmt(agg_l['r2_hg_median'])}, "
          f"R2rad median={fmt(agg_l['R2_radial_median'])}")
    record('solver', 'LassoLars', {'degree': PAPER_DEGREE, 'basis': 'poly'},
           rows_l, agg_l, basis_paper, PAPER_ALPHA)

    # ==================================================================
    # Axis B: polynomial degree
    # ==================================================================
    print("\n" + "=" * 72)
    print(f"AXIS B: polynomial degree  (alpha = {PAPER_ALPHA:g})")
    print("=" * 72)
    axis_b = []
    for D in DEGREE_SWEEP:
        b = build_basis('poly', pts, D)
        rows, agg = run_setting(b, grads, pts, PAPER_ALPHA)
        a = record('degree', str(D), {'degree': D, 'basis': 'poly'},
                   rows, agg, b, PAPER_ALPHA)
        axis_b.append(a)
        print(f"  D={D} P={b.n_atoms:3d} "
              f"R2hg med={fmt(agg['r2_hg_median'])} min={fmt(agg['r2_hg_min'])} "
              f"|#act={agg['n_active_median']:.1f} FP={fmt(agg['n_fp_median'],1)} "
              f"|R2rad med={fmt(agg['R2_radial_median'])} max={fmt(agg['R2_radial_max'])} "
              f"|rho med={fmt(agg['spearman_median'])} "
              f"|radial-sig={agg['n_radial_signature']} "
              f"|t={agg['t_decomp_total_s']:.2f}s ({agg['t_decomp_per_dim_s']:.3f}/dim)")
    write_csv(OUTDIR / 'axis_degree.csv', axis_b)

    # Does adding candidates HURT recognition? (FP count vs P at fixed alpha)
    print("\n  candidate-count cost:  P -> median FP -> median #active")
    prev_fp = None
    hurt_at = None
    for a in axis_b:
        fp = a['n_fp_median']
        print(f"    D={a['setting']}  P={a['P']:3d}  FP={fmt(fp,1)}  "
              f"#act={a['n_active_median']:.1f}  "
              f"R2rad_med={fmt(a['R2_radial_median'])}")
        if prev_fp is not None and not np.isnan(fp) and fp > prev_fp and hurt_at is None:
            hurt_at = a['setting']
        prev_fp = fp
    print(f"    first degree where FP count rises with P: "
          f"{hurt_at if hurt_at else 'none'}")

    # ==================================================================
    # Axis C: readout basis
    # ==================================================================
    print("\n" + "=" * 72)
    print(f"AXIS C: readout basis  (alpha = {PAPER_ALPHA:g})")
    print("=" * 72)
    axis_c = []
    for kind in BASIS_SWEEP:
        b = build_basis(kind, pts, PAPER_DEGREE)
        rows, agg = run_setting(b, grads, pts, PAPER_ALPHA)
        a = record('basis', kind, {'degree': PAPER_DEGREE if kind == 'poly' else -1,
                                   'basis': kind},
                   rows, agg, b, PAPER_ALPHA)
        axis_c.append(a)
        print(f"  {kind:<12} P={b.n_atoms:3d} "
              f"R2hg med={fmt(agg['r2_hg_median'])} min={fmt(agg['r2_hg_min'])} "
              f"|#act={agg['n_active_median']:.1f} "
              f"|R2rad med={fmt(agg['R2_radial_median'])} max={fmt(agg['R2_radial_max'])} "
              f"|rho med={fmt(agg['spearman_median'])} max={fmt(agg['spearman_max'])} "
              f"|affine h={agg['n_affine_h']} "
              f"|t={agg['t_decomp_total_s']:.2f}s")
        # which atoms does the correctly-specified basis actually pick?
        if kind == 'radial_trig':
            for r in rows:
                print(f"      z[{r['dim']}] top: {r['top_atoms']}")
    write_csv(OUTDIR / 'axis_basis.csv', axis_c)

    # ==================================================================
    # Axis D: prune threshold tau
    # ==================================================================
    axis_d = []
    if not args.skip_tau:
        print("\n" + "=" * 72)
        print("AXIS D: prune threshold tau  (prune -> retrain -> re-decompose)")
        print("=" * 72)
        base_state = {k: v.clone() for k, v in model.state_dict().items()}
        base_scores = attribution_scores(model, pts)
        summ = score_summary(base_scores)
        print(f"  encoder attribution scores (PyKAN B_ij, {sum(s.size for s in base_scores.values())} edges): "
              f"min={summ['min']:.3f} p25={summ['p25']:.3f} med={summ['median']:.3f} "
              f"max={summ['max']:.3f}")
        print(f"  fraction <= 0.01/0.03/0.05: {summ['frac_below_0.01']:.2f}/"
              f"{summ['frac_below_0.03']:.2f}/{summ['frac_below_0.05']:.2f}  "
              f"(Lorenz uses tau=0.03)")
        for tau in TAU_SWEEP:
            print(f"\n  tau = {tau}")
            m = IM.IkedaMinimalKoopman()
            m.load_state_dict(base_state)
            m.eval()
            if tau > 0:
                n_pruned, n_total, _ = prune_at(m, tau, pts)
                print(f"    pruned {n_pruned}/{n_total} encoder edges")
                if n_pruned > 0:
                    m = retrain(m, traj_train)
                else:
                    print("    (nothing pruned -> model is byte-identical to tau=0; "
                          "skipping retrain)")
            else:
                n_pruned = 0
                n_total = sum(l.out_features * l.in_features
                              for l in encoder_layers(m))
            mse_tau = one_step_mse(m, traj_train)
            g_tau = precompute_gradients(m, pts, latent_dim)
            rows, agg = run_setting(basis_paper, g_tau, pts, PAPER_ALPHA)
            a = record('tau', f'{tau:g}',
                       {'degree': PAPER_DEGREE, 'basis': 'poly',
                        'n_pruned': n_pruned, 'n_edges': n_total,
                        'frac_pruned': n_pruned / max(n_total, 1),
                        'in_lorenz_grid': int(tau in LORENZ_TAU_GRID),
                        'one_step_mse': mse_tau,
                        'mse_ratio': mse_tau / max(base_mse, 1e-30)},
                       rows, agg, basis_paper, PAPER_ALPHA)
            axis_d.append(a)
            print(f"    MSE {mse_tau:.4e} ({mse_tau/base_mse:.2f}x base) "
                  f"| R2hg med={fmt(agg['r2_hg_median'])} min={fmt(agg['r2_hg_min'])} "
                  f"| #act={agg['n_active_median']:.1f} "
                  f"| R2rad med={fmt(agg['R2_radial_median'])} "
                  f"max={fmt(agg['R2_radial_max'])} "
                  f"| radial-sig={agg['n_radial_signature']} "
                  f"| best quad ratio={fmt(agg['quad_ratio_best'])}")
            torch.save({'model_state_dict': m.state_dict(), 'tau': tau},
                       OUTDIR / f'model_tau{tau:g}.pt')
        write_csv(OUTDIR / 'axis_tau.csv', axis_d)

    # ==================================================================
    # Save everything
    # ==================================================================
    write_csv(OUTDIR / 'all_settings.csv', all_rows)
    write_csv(OUTDIR / 'per_dim_detail.csv', detail_rows)

    config = {
        'system': 'ikeda',
        'architecture': f'[2,{IM.MID_DIM},{IM.LATENT_DIM}] KANLinear enc/dec '
                        '+ stable propagator expm(K - L^T L)',
        'kan_implementation': 'KANLinear (kan.py), NOT pykan',
        'seed': SEED,
        'train_epochs': IM.EPOCHS, 'train_lr': IM.LR,
        'n_traj': IM.N_TRAJ, 'n_steps': IM.N_STEPS, 'burn_in': IM.BURN_IN,
        'data_seed_train': DATA_SEED_TRAIN,
        'data_seed_analysis': DATA_SEED_ANALYSIS,
        'n_analysis_pts': N_ANALYSIS_PTS,
        'paper_alpha': PAPER_ALPHA, 'paper_degree': PAPER_DEGREE,
        'h_degree': H_DEGREE, 'sparsity_threshold': SPARSITY_THRESHOLD,
        'alpha_sweep': ALPHA_SWEEP, 'degree_sweep': DEGREE_SWEEP,
        'basis_sweep': BASIS_SWEEP, 'tau_sweep': TAU_SWEEP,
        'rbf_per_dim': RBF_PER_DIM,
        'retrain_epochs': RETRAIN_EPOCHS, 'retrain_lr': RETRAIN_LR,
        'lorenz_tau_grid': LORENZ_TAU_GRID,
        'tau_score': 'PyKAN edge attribution B_ij (kan_attribution.py), '
                     'not the unnormalized KANLinear spline-weight magnitude',
        'target_atoms': sorted(IKEDA_TARGET_ATOMS),
        'base_one_step_mse': base_mse,
    }
    with open(OUTDIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    summary = {
        'config': config,
        'reproduction': {
            'paper_r2_hg': PAPER_R2_HG,
            'ours_r2_hg': [r['r2_hg'] for r in repro_rows],
            'all_in_band_0.90_0.995': bool(in_band),
            'best_quad_ratio': best_ratio,
            'paper_quad_ratio': PAPER_QUAD_RATIO,
            'n_radial_signature': repro_agg['n_radial_signature'],
            'reproduced': bool(reproduced),
            'aggregate': repro_agg,
        },
        'axis_lambda': axis_a,
        'axis_degree': axis_b,
        'axis_basis': axis_c,
        'axis_tau': axis_d,
        'attribution_score_summary': (score_summary(attribution_scores(model, pts))
                                      if not args.skip_tau else None),
        'degree_fp_rise_at': hurt_at,
    }
    with open(OUTDIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote {OUTDIR / 'summary.json'}")
    print("Done.")


if __name__ == '__main__':
    main()
