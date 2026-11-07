"""
Standard map (kappa = 0.97): readout robustness study  (NeurIPS reviewer EbT5-Q4).

Question under test
-------------------
"How sensitive are the recovered dictionaries to the choice of Lasso
regularization, polynomial degree, readout basis, and pruning threshold,
particularly on the non-polynomial systems?"

The standard map is the second non-polynomial system: the encoder takes the
trig embedding (cos th, sin th, cos p, sin p) of the angles, so a *polynomial*
readout in those four variables is exactly a *Fourier* readout in the angles.
The paper's claim (main text Eq. 13) is that the learned observables span the
|m| + |n| <= 2 Fourier atoms

    {cos th, sin th, cos p, sin p,
     cos th cos p, sin th sin p, cos th sin p, sin th cos p,
     cos th sin th, cos p sin p}

which, in the poly-in-trig chart, are the 10 non-square degree-2 monomials.

Axes
----
  A. Lasso lambda in {1e-6 ... 1e-2}, run at BOTH degree 2 (the paper default,
     which spans exactly |m|+|n| <= 2) and degree 3.
  B. polynomial degree D in {1..5}                     (alpha = 1e-5, paper value)
  C. readout basis: poly-in-trig (paper default)
                  / direct Fourier features e^{i(m th + n p)}, |m|+|n| <= 3
                  / periodic (von Mises) RBF                (alpha = 1e-5)
  D. prune tau -- NOT RUN.  See the note printed at the end: this encoder is
     KANLinear-based and the paper's tau protocol is PyKAN edge attribution.

Charts
------
The polynomial basis is differentiated in the trig chart (cx, sx, cy, sy), which
is the paper's protocol.  The Fourier and RBF bases are differentiated in the
angle chart (theta, p).  The gradient identity h' = <grad f, grad g>/||grad g||^2
is chart-covariant provided f and g are differentiated in the same chart, so
both are valid; each row records which chart it used.

Structural-stability metrics
----------------------------
  recall/precision/jaccard vs the Eq.-13 atom set, computed on the UNION of
  dominant atoms across latent dims (the Lorenz protocol in
  experiments/pykan_pruned_sweep.py).
  R2_eq13 : basis-independent.  R^2 of OLS-regressing the recovered g onto the
            span of the 10 Eq.-13 Fourier atoms.  If g still lives in that span,
            this is ~1 no matter which dictionary produced it -- so it is the
            metric that makes the RBF and Fourier rows comparable to the
            polynomial rows.

Cost metrics
------------
  P        : design-matrix column count before Lasso
  t_decomp : wall time of the decomposition step, total and per latent coord.

Run:
    python experiments/standard_map_robustness.py
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

from training import set_seed
from standard_map import simulate_standard_map
from standard_map_stable import StableKoopmanKAN
from readout_bases import (
    poly_basis, torus_fourier_basis, torus_rbf_basis,
    decompose_with_basis, r2_against_span, set_metrics,
)

# =====================================================================
# Configuration
# =====================================================================

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'results' / 'standard_map_robustness'
CKPT = ROOT / 'results' / 'standard_map' / 'standard_map_stable_32_128.pt'

SEED = 42
N_ANALYSIS_PTS = 8000       # matches experiments/standard_map_analysis.py
N_TRAJ, N_STEPS = 200, 2000

# Paper defaults for this system (experiments/standard_map_analysis.py)
PAPER_ALPHA = 1e-5
PAPER_DEGREE = 2            # degree 2 in (cx,sx,cy,sy) == |m|+|n| <= 2
H_DEGREE = 3
SPARSITY_THRESHOLD = 1e-4

ALPHA_SWEEP = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
DEGREE_SWEEP = [1, 2, 3, 4, 5]
BASIS_SWEEP = ['poly_trig', 'fourier', 'rbf']
FOURIER_MAX_ORDER = 3
RBF_PER_DIM = 8             # 8x8 = 64 von Mises centers + constant -> P = 65
RBF_KAPPA = 4.0

TRIG_VARS = ['cx', 'sx', 'cy', 'sy']

# Eq. 13 atom set, in poly-in-trig naming (see build_polynomial_dict_nd naming).
EQ13_POLY = {'cx', 'sx', 'cy', 'sy',
             'cxcy', 'sxsy', 'cxsy', 'sxcy',
             'cxsx', 'cysy'}
# NOTE: the literal Eq.-13 list above is *not* the complete |m|+|n| <= 2 family
# it claims to be.  cx*sx = sin(2th)/2 and cy*sy = sin(2p)/2 are the SINE parts
# of the (2,0) and (0,2) modes; the COSINE parts cos(2th), cos(2p) are missing
# from the list.  In the poly-in-trig chart those are exactly cx^2/sx^2 and
# cy^2/sy^2 (cos 2th = 2cx^2 - 1 = 1 - 2sx^2).  So we score against both:
#   EQ13_POLY      -- the literal main-text list (10 atoms)
#   EQ13_POLY_FULL -- the complete |m|+|n| <= 2 family (14 monomials)
# and report where the difference matters.
EQ13_POLY_FULL = EQ13_POLY | {'cx^2', 'sx^2', 'cy^2', 'sy^2'}
# Same set as direct Fourier atoms: cos/sin of (m,n) with 1 <= |m|+|n| <= 2.
EQ13_FOURIER = set()
for _mn in [(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)]:
    EQ13_FOURIER.add(f'cos({_mn[0]},{_mn[1]})')
    EQ13_FOURIER.add(f'sin({_mn[0]},{_mn[1]})')

torch.set_num_threads(12)


# =====================================================================
# Model + gradients
# =====================================================================

def load_model():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = StableKoopmanKAN(cfg['mid'], cfg['latent'])
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model, cfg


def trig_chart_grad(model, angles, target_idx, chunk=2000):
    """d z_k / d(cx, sx, cy, sy): differentiate the KAN stack directly.

    Bypasses the encoder's internal trig embedding so the chart is the trig
    embedding itself -- this is what experiments/standard_map_analysis.py does
    and what makes the recovered monomials read as Fourier atoms.
    """
    gs, fs = [], []
    for i in range(0, len(angles), chunk):
        b = torch.tensor(angles[i:i + chunk], dtype=torch.float32)
        x, y = b[:, 0], b[:, 1]
        t = torch.stack([torch.cos(x), torch.sin(x),
                         torch.cos(y), torch.sin(y)], dim=-1).requires_grad_(True)
        z = model.encoder.kan2(model.encoder.kan1(t))[:, target_idx]
        z.sum().backward()
        gs.append(t.grad.numpy())
        fs.append(z.detach().numpy())
    return np.concatenate(gs), np.concatenate(fs)


def angle_chart_grad(model, angles, target_idx, chunk=2000):
    """d z_k / d(theta, p): differentiate through the trig embedding."""
    gs, fs = [], []
    for i in range(0, len(angles), chunk):
        b = torch.tensor(angles[i:i + chunk], dtype=torch.float32,
                         requires_grad=True)
        z = model.encoder(b)[:, target_idx]
        z.sum().backward()
        gs.append(b.grad.numpy())
        fs.append(z.detach().numpy())
    return np.concatenate(gs), np.concatenate(fs)


# =====================================================================
# One sweep setting
# =====================================================================

def run_setting(basis, grads, ref_M, ref_atoms, alpha, latent_dim, solver='cd',
                ref_atoms_full=None):
    """Decompose every latent dim; return per-dim rows + aggregate row."""
    rows = []
    union_all, union_dom = set(), set()
    t_total = 0.0

    for k, (grad_f, f_vals) in enumerate(grads):
        t0 = time.perf_counter()
        res = decompose_with_basis(basis, f_vals, grad_f, alpha,
                                   sparsity_threshold=SPARSITY_THRESHOLD,
                                   h_degree=H_DEGREE, solver=solver)
        dt = time.perf_counter() - t0
        t_total += dt

        act = set(res['active_names']) - {'1'}
        dom = set(res['dominant_names']) - {'1'}
        union_all |= act
        union_dom |= dom

        in_ref = len(dom & ref_atoms) if ref_atoms else 0
        rows.append({
            'dim': k,
            'r2_g': res['r2_g'],
            'r2_hg': res['r2_hg'],
            'n_active': res['n_active'],
            'n_dominant': res['n_dominant'],
            'used_affine': int(res['used_affine']),
            'degenerate': int(res['degenerate']),
            'n_in_ref': in_ref,
            'n_extra': len(dom - ref_atoms) if ref_atoms else float('nan'),
            'frac_in_ref': in_ref / max(len(dom), 1) if ref_atoms else float('nan'),
            'R2_eq13': r2_against_span(res['g_vals'], ref_M),
            'top_atoms': ';'.join(
                f"{n}({c:+.4f})" for n, c in sorted(
                    zip(res['active_names'], res['active_coeffs']),
                    key=lambda p: -abs(p[1]))[:6]),
            't_decomp_s': dt,
        })

    r2hg = np.array([r['r2_hg'] for r in rows])
    m_all = set_metrics(union_all, ref_atoms) if ref_atoms else {}
    m_dom = set_metrics(union_dom, ref_atoms) if ref_atoms else {}
    m_full = set_metrics(union_dom, ref_atoms_full) if ref_atoms_full else {}

    agg = {
        'r2_hg_median': float(np.median(r2hg)),
        'r2_hg_min': float(r2hg.min()),
        'r2_hg_p10': float(np.percentile(r2hg, 10)),
        'n_dims_r2hg_gt_0.9': int((r2hg > 0.9).sum()),
        'n_dims_r2hg_gt_0.8': int((r2hg > 0.8).sum()),
        'n_active_median': float(np.median([r['n_active'] for r in rows])),
        'n_dominant_median': float(np.median([r['n_dominant'] for r in rows])),
        'R2_eq13_median': float(np.median([r['R2_eq13'] for r in rows])),
        'R2_eq13_p90': float(np.percentile([r['R2_eq13'] for r in rows], 90)),
        'n_affine_h': int(sum(r['used_affine'] for r in rows)),
        'n_degenerate': int(sum(r['degenerate'] for r in rows)),
        'union_recall': m_dom.get('recall', float('nan')),
        'union_precision': m_dom.get('precision', float('nan')),
        'union_jaccard': m_dom.get('jaccard', float('nan')),
        'union_n_extra': m_dom.get('n_extra', float('nan')),
        'union_n_detected': m_dom.get('n_detected', float('nan')),
        'full2_recall': m_full.get('recall', float('nan')),
        'full2_precision': m_full.get('precision', float('nan')),
        'full2_jaccard': m_full.get('jaccard', float('nan')),
        'full2_n_extra': m_full.get('n_extra', float('nan')),
        'full2_missing': (';'.join(sorted(ref_atoms_full - union_dom))
                          if ref_atoms_full else ''),
        'union_recall_allactive': m_all.get('recall', float('nan')),
        'union_jaccard_allactive': m_all.get('jaccard', float('nan')),
        'union_n_extra_allactive': m_all.get('n_extra', float('nan')),
        'extra_atoms': ';'.join(sorted(union_dom - ref_atoms)) if ref_atoms else '',
        'missing_atoms': ';'.join(sorted(ref_atoms - union_dom)) if ref_atoms else '',
        't_decomp_total_s': t_total,
        't_decomp_per_dim_s': t_total / max(latent_dim, 1),
    }
    return rows, agg


# =====================================================================
# I/O
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
    ap.add_argument('--alpha-degrees', type=int, nargs='+', default=[2, 3],
                    help='degrees at which to run the lambda sweep')
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    model, cfg = load_model()
    latent_dim = cfg['latent']
    K = cfg['K']
    print(f"Loaded {CKPT.name}: [4,{cfg['mid']},{latent_dim}], kappa={K}")

    M = model.koopman_matrix().numpy()
    eig_mag = np.abs(np.linalg.eigvals(M))
    sing = np.linalg.svd(M, compute_uv=False)
    print(f"  |lambda| in [{eig_mag.min():.4f}, {eig_mag.max():.4f}], "
          f"#>0.99 = {(eig_mag > 0.99).sum()}")
    print(f"  sigma   in [{sing.min():.4f}, {sing.max():.4f}], "
          f"#>0.99 = {(sing > 0.99).sum()}, #>0.95 = {(sing > 0.95).sum()}")

    # ---- data -----------------------------------------------------
    traj = simulate_standard_map(n_traj=N_TRAJ, n_steps=N_STEPS, K=K)
    xy = traj.reshape(-1, 2).numpy()
    rng = np.random.RandomState(0)
    angles = xy[rng.choice(len(xy), N_ANALYSIS_PTS, replace=False)].astype(np.float64)
    trig = np.column_stack([np.cos(angles[:, 0]), np.sin(angles[:, 0]),
                            np.cos(angles[:, 1]), np.sin(angles[:, 1])])

    # Reference span: the 10 Eq.-13 Fourier atoms, evaluated once.
    ref_basis = poly_basis(trig, TRIG_VARS, max_degree=2)
    ref_cols = [ref_basis.names.index(n) for n in sorted(EQ13_POLY_FULL)]
    REF_M = ref_basis.M[:, ref_cols]   # span of the complete |m|+|n| <= 2 family

    # ---- gradients, one set per chart -----------------------------
    print("Precomputing encoder gradients (2 charts x "
          f"{latent_dim} latent dims)...")
    t0 = time.perf_counter()
    grads_trig = [trig_chart_grad(model, angles, k) for k in range(latent_dim)]
    grads_ang = [angle_chart_grad(model, angles, k) for k in range(latent_dim)]
    print(f"  {time.perf_counter() - t0:.1f}s")

    all_rows, detail_rows = [], []

    def record(axis, setting, extra, rows, agg, P, alpha, chart):
        row = {'axis': axis, 'setting': setting, 'P': P, 'alpha': alpha,
               'chart': chart, **extra, **agg}
        all_rows.append(row)
        for r in rows:
            detail_rows.append({'axis': axis, 'setting': setting, 'P': P,
                                'alpha': alpha, 'chart': chart, **r})
        return row

    # ==================================================================
    # Reproduction check vs Eq. 13
    # ==================================================================
    print("\n" + "=" * 78)
    print(f"REPRODUCTION CHECK  (alpha={PAPER_ALPHA:g}, degree {PAPER_DEGREE}, "
          "poly-in-trig)")
    print("=" * 78)
    basis_paper = poly_basis(trig, TRIG_VARS, max_degree=PAPER_DEGREE)
    repro_rows, repro_agg = run_setting(basis_paper, grads_trig, REF_M,
                                        EQ13_POLY, PAPER_ALPHA, latent_dim,
                                        ref_atoms_full=EQ13_POLY_FULL)
    print(f"  P = {basis_paper.n_atoms} atoms: {basis_paper.names}")
    print(f"  R2(h o g): median {fmt(repro_agg['r2_hg_median'])}, "
          f"p10 {fmt(repro_agg['r2_hg_p10'])}, min {fmt(repro_agg['r2_hg_min'])}, "
          f"{repro_agg['n_dims_r2hg_gt_0.9']}/{latent_dim} dims > 0.9")
    print(f"  union(dominant) vs Eq.13: recall {fmt(repro_agg['union_recall'])}, "
          f"precision {fmt(repro_agg['union_precision'])}, "
          f"jaccard {fmt(repro_agg['union_jaccard'])}, "
          f"extra {repro_agg['union_n_extra']}")
    print(f"  extra atoms   : {repro_agg['extra_atoms'] or '(none)'}")
    print(f"  missing atoms : {repro_agg['missing_atoms'] or '(none)'}")
    print(f"  union(dominant) vs COMPLETE |m|+|n|<=2 family (14 monomials): "
          f"recall {fmt(repro_agg['full2_recall'])}, "
          f"precision {fmt(repro_agg['full2_precision'])}, "
          f"jaccard {fmt(repro_agg['full2_jaccard'])}, "
          f"extra {repro_agg['full2_n_extra']}")
    print(f"  R2_fourier2 (g projected on the |m|+|n|<=2 span): "
          f"median {fmt(repro_agg['R2_eq13_median'])}")
    for r in repro_rows[:6]:
        print(f"    z[{r['dim']}] R2g={r['r2_g']:.3f} R2hg={r['r2_hg']:.3f} "
              f"n={r['n_active']} {r['top_atoms']}")

    reproduced = (repro_agg['union_recall'] >= 0.9
                  and repro_agg['r2_hg_median'] > 0.9)
    print(f"\n  REPRODUCED (recall>=0.9 of Eq.13 and median R2(hog)>0.9): "
          f"{reproduced}")
    if not reproduced:
        print("  WARNING: base model does not reproduce the paper's Eq.-13 "
              "recovery; sweep numbers describe THIS model.")
    write_csv(OUTDIR / 'reproduction_per_dim.csv', repro_rows)

    # ==================================================================
    # Axis A: Lasso lambda
    # ==================================================================
    axis_a = []
    for D in args.alpha_degrees:
        print("\n" + "=" * 78)
        print(f"AXIS A: Lasso lambda  (poly-in-trig, degree {D})")
        print("=" * 78)
        b = poly_basis(trig, TRIG_VARS, max_degree=D)
        for alpha in ALPHA_SWEEP:
            rows, agg = run_setting(b, grads_trig, REF_M, EQ13_POLY, alpha,
                                    latent_dim, ref_atoms_full=EQ13_POLY_FULL)
            a = record('lambda', f'{alpha:g}', {'degree': D, 'basis': 'poly_trig'},
                       rows, agg, b.n_atoms, alpha, 'trig')
            axis_a.append(a)
            print(f"  a={alpha:>8.0e} P={b.n_atoms:3d} "
                  f"R2hg med={fmt(agg['r2_hg_median'])} p10={fmt(agg['r2_hg_p10'])} "
                  f"|#act={agg['n_active_median']:.1f} "
                  f"|rec={fmt(agg['union_recall'])} prec={fmt(agg['union_precision'])} "
                  f"jac={fmt(agg['union_jaccard'])} extra={agg['union_n_extra']} "
                  f"|full2: rec={fmt(agg['full2_recall'])} "
                  f"prec={fmt(agg['full2_precision'])} "
                  f"jac={fmt(agg['full2_jaccard'])} extra={agg['full2_n_extra']} "
                  f"|R2f2 med={fmt(agg['R2_eq13_median'])} "
                  f"|affine={agg['n_affine_h']} "
                  f"|t={agg['t_decomp_total_s']:.1f}s")
            if agg['missing_atoms']:
                print(f"           missing: {agg['missing_atoms']}")
    write_csv(OUTDIR / 'axis_lambda.csv', axis_a)

    # LassoLars cross-check at the paper setting
    rows_l, agg_l = run_setting(basis_paper, grads_trig, REF_M, EQ13_POLY,
                                PAPER_ALPHA, latent_dim, solver='lars',
                                ref_atoms_full=EQ13_POLY_FULL)
    print(f"\n  LassoLars cross-check @ a={PAPER_ALPHA:g}, D={PAPER_DEGREE}: "
          f"#act median={agg_l['n_active_median']:.1f} "
          f"(CD: {repro_agg['n_active_median']:.1f}), "
          f"recall={fmt(agg_l['union_recall'])} "
          f"(CD: {fmt(repro_agg['union_recall'])}), "
          f"R2hg median={fmt(agg_l['r2_hg_median'])}")
    record('solver', 'LassoLars', {'degree': PAPER_DEGREE, 'basis': 'poly_trig'},
           rows_l, agg_l, basis_paper.n_atoms, PAPER_ALPHA, 'trig')

    # ==================================================================
    # Axis B: polynomial degree
    # ==================================================================
    print("\n" + "=" * 78)
    print(f"AXIS B: polynomial degree  (alpha = {PAPER_ALPHA:g}, poly-in-trig)")
    print("=" * 78)
    axis_b = []
    for D in DEGREE_SWEEP:
        b = poly_basis(trig, TRIG_VARS, max_degree=D)
        rows, agg = run_setting(b, grads_trig, REF_M, EQ13_POLY, PAPER_ALPHA,
                                latent_dim, ref_atoms_full=EQ13_POLY_FULL)
        a = record('degree', str(D), {'degree': D, 'basis': 'poly_trig'},
                   rows, agg, b.n_atoms, PAPER_ALPHA, 'trig')
        axis_b.append(a)
        print(f"  D={D} P={b.n_atoms:3d} "
              f"R2hg med={fmt(agg['r2_hg_median'])} p10={fmt(agg['r2_hg_p10'])} "
              f"|#act={agg['n_active_median']:.1f} "
              f"|rec={fmt(agg['union_recall'])} prec={fmt(agg['union_precision'])} "
              f"jac={fmt(agg['union_jaccard'])} extra={agg['union_n_extra']} "
              f"|full2: rec={fmt(agg['full2_recall'])} "
              f"prec={fmt(agg['full2_precision'])} "
              f"jac={fmt(agg['full2_jaccard'])} extra={agg['full2_n_extra']} "
              f"|R2f2 med={fmt(agg['R2_eq13_median'])} "
              f"|t={agg['t_decomp_total_s']:.1f}s "
              f"({agg['t_decomp_per_dim_s']:.3f}/dim)")
    write_csv(OUTDIR / 'axis_degree.csv', axis_b)

    print("\n  candidate-count cost:  P -> extra atoms -> precision "
          "(vs the complete |m|+|n|<=2 family)")
    prev_extra, hurt_at = None, None
    for a in axis_b:
        print(f"    D={a['setting']}  P={a['P']:3d}  extra={a['full2_n_extra']}  "
              f"prec={fmt(a['full2_precision'])}  "
              f"rec={fmt(a['full2_recall'])}  "
              f"#act={a['n_active_median']:.1f}  "
              f"R2f2={fmt(a['R2_eq13_median'])}")
        if prev_extra is not None and a['full2_n_extra'] > prev_extra and hurt_at is None:
            hurt_at = a['setting']
        prev_extra = a['full2_n_extra']
    print(f"    first degree where extra-atom count rises with P: "
          f"{hurt_at if hurt_at else 'none'}")

    # ==================================================================
    # Axis C: readout basis
    # ==================================================================
    print("\n" + "=" * 78)
    print(f"AXIS C: readout basis  (alpha = {PAPER_ALPHA:g})")
    print("=" * 78)
    axis_c = []
    basis_specs = {
        'poly_trig': (lambda: poly_basis(trig, TRIG_VARS, max_degree=PAPER_DEGREE),
                      grads_trig, EQ13_POLY, EQ13_POLY_FULL, 'trig'),
        # EQ13_FOURIER is already the complete |m|+|n| <= 2 family in real form
        'fourier': (lambda: torus_fourier_basis(angles, max_order=FOURIER_MAX_ORDER),
                    grads_ang, EQ13_FOURIER, EQ13_FOURIER, 'angle'),
        'rbf': (lambda: torus_rbf_basis(angles, n_per_dim=RBF_PER_DIM,
                                        kappa=RBF_KAPPA),
                grads_ang, set(), None, 'angle'),
    }
    for kind in BASIS_SWEEP:
        make, grads, ref_atoms, ref_full, chart = basis_specs[kind]
        b = make()
        rows, agg = run_setting(b, grads, REF_M, ref_atoms, PAPER_ALPHA,
                                latent_dim, ref_atoms_full=ref_full)
        a = record('basis', kind, {'degree': PAPER_DEGREE if kind == 'poly_trig' else -1,
                                   'basis': kind}, rows, agg, b.n_atoms,
                   PAPER_ALPHA, chart)
        axis_c.append(a)
        print(f"  {kind:<10} P={b.n_atoms:3d} chart={chart:<6} "
              f"R2hg med={fmt(agg['r2_hg_median'])} p10={fmt(agg['r2_hg_p10'])} "
              f"|#act={agg['n_active_median']:.1f} "
              f"|rec={fmt(agg['union_recall'])} prec={fmt(agg['union_precision'])} "
              f"jac={fmt(agg['union_jaccard'])} extra={agg['union_n_extra']} "
              f"|full2: rec={fmt(agg['full2_recall'])} "
              f"prec={fmt(agg['full2_precision'])} "
              f"|R2f2 med={fmt(agg['R2_eq13_median'])} "
              f"p90={fmt(agg['R2_eq13_p90'])} "
              f"|affine={agg['n_affine_h']} "
              f"|t={agg['t_decomp_total_s']:.1f}s")
        if agg['extra_atoms']:
            print(f"      extra   : {agg['extra_atoms']}")
        if agg['missing_atoms']:
            print(f"      missing : {agg['missing_atoms']}")
    write_csv(OUTDIR / 'axis_basis.csv', axis_c)

    # ==================================================================
    # Axis D: prune tau -- documented as not applicable
    # ==================================================================
    tau_note = (
        "Prune-tau axis NOT run for the standard map. The paper's tau protocol "
        "(appendix Eq. edge-importance) is a PyKAN-style layerwise attribution "
        "on activation variance, B_ij = mean_k r_kij, applied to a PyKAN "
        "encoder. The committed standard-map encoder "
        "(results/standard_map/standard_map_stable_32_128.pt) is KANLinear "
        "(kan.py), which exposes no PyKAN activation graph and therefore no "
        "attribution scores; the KANLinear surrogate used for Ikeda "
        "(spline_weight magnitude) is a different score and would not be the "
        "same tau axis. Retraining a PyKAN [4,32,128] standard-map encoder "
        "purely to add this axis was judged out of scope: the tau axis is "
        "already covered on two systems (Lorenz tab:prune-sweep and Ikeda, "
        "results/ikeda_robustness/axis_tau.csv)."
    )
    print("\n" + "=" * 78)
    print("AXIS D: prune tau")
    print("=" * 78)
    print("  " + tau_note.replace('. ', '.\n  '))

    # ==================================================================
    # Save
    # ==================================================================
    write_csv(OUTDIR / 'all_settings.csv', all_rows)
    write_csv(OUTDIR / 'per_dim_detail.csv', detail_rows)

    config = {
        'system': 'standard_map',
        'kappa': K,
        'checkpoint': str(CKPT.relative_to(ROOT)),
        'architecture': f"[4,{cfg['mid']},{latent_dim}] KANLinear enc/dec on the "
                        "(cos th, sin th, cos p, sin p) embedding + stable "
                        "propagator expm(K - L^T L)",
        'kan_implementation': 'KANLinear (kan.py), NOT pykan',
        'seed': SEED, 'n_analysis_pts': N_ANALYSIS_PTS,
        'n_traj': N_TRAJ, 'n_steps': N_STEPS,
        'paper_alpha': PAPER_ALPHA, 'paper_degree': PAPER_DEGREE,
        'h_degree': H_DEGREE, 'sparsity_threshold': SPARSITY_THRESHOLD,
        'alpha_sweep': ALPHA_SWEEP, 'alpha_sweep_degrees': args.alpha_degrees,
        'degree_sweep': DEGREE_SWEEP, 'basis_sweep': BASIS_SWEEP,
        'fourier_max_order': FOURIER_MAX_ORDER,
        'rbf_per_dim': RBF_PER_DIM, 'rbf_kappa': RBF_KAPPA,
        'eq13_poly_atoms': sorted(EQ13_POLY),
        'eq13_poly_full_atoms': sorted(EQ13_POLY_FULL),
        'eq13_fourier_atoms': sorted(EQ13_FOURIER),
        'eig_mag_min': float(eig_mag.min()), 'eig_mag_max': float(eig_mag.max()),
        'n_sigma_gt_0.99': int((sing > 0.99).sum()),
        'n_sigma_gt_0.95': int((sing > 0.95).sum()),
        'tau_axis_note': tau_note,
    }
    with open(OUTDIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    summary = {
        'config': config,
        'reproduction': {'aggregate': repro_agg, 'reproduced': bool(reproduced)},
        'axis_lambda': axis_a,
        'axis_degree': axis_b,
        'axis_basis': axis_c,
        'axis_tau': tau_note,
        'degree_extra_rise_at': hurt_at,
    }
    with open(OUTDIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote {OUTDIR / 'summary.json'}")
    print("Done.")


if __name__ == '__main__':
    main()
