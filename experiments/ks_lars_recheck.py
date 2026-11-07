"""
KS level-set dictionary re-check: coordinate-descent Lasso vs exact LARS.

WHY THIS EXPERIMENT EXISTS
--------------------------
On Mackey-Glass we discovered that sklearn's coordinate-descent (CD) Lasso
returns DENSE near-optima on collinear, high-dimensional polynomial designs
even when it reports full convergence, while LassoLars (exact LARS path)
returns a genuinely sparse solution at the SAME alpha with an identical R^2
(MG: 254 nonzeros CD vs 12 nonzeros LARS, both R^2 = 0.9993).

The KS claims in notes.md / the rebuttal draft rest on the CD-Lasso run inside
`manifold_analysis.recover_inner_function_nd` (Lasso, alpha=1e-4, max_iter=
50000) that reported "each latent dim uses 110-190 active polynomial terms of
2145 degree-2 monomials", concluding "the dictionary is dense: the high-D
lift foliates but does not sparsely symbolize."

This script re-runs the inner-function recovery for ALL 128 latent dims with
BOTH solvers at the same alpha and asks: does the density survive LARS?

WHAT IT DOES
------------
1. Loads the trained KS Koopman-KAN checkpoint (wide [64,96,128] if present,
   else the committed narrow [64,32,128]) -- no retraining.
2. Samples N_SAMPLE snapshots and builds the degree-2 design in the 64 grid
   variables ONCE (2145 monomials), reused for all 128 latent coordinates.
3. Per latent dim: fits the inner g with (a) CD Lasso (max_iter=100000,
   recording n_iter_/dual gap/ConvergenceWarning) and (b) LassoLars, at the
   SAME alpha. Records nonzero counts and R^2 under both.
4. For the LassoLars inner, runs the identical gradient-identity outer-function
   pipeline as manifold_analysis.full_decomposition_nd: R^2(affine),
   R^2(h o g), and the outer gain.
5. Classifies the surviving LARS atoms: constant / linear u_i / diagonal
   u_i^2 / cross u_i u_j, and for the cross terms the periodic spatial
   separation |i-j| (advection u u_x lives at separation 1).
6. ALIASING CONTRAST: for representative grid points, fits an OLS LINEAR
   surrogate on the 64 snapshot values to the advection target
   n_i = u_i (u_{i+1} - u_{i-1}) / (2 dx). A spatial snapshot has no temporal
   extent, so u_t is not linearly readable and the MG-style
   representation-induced linearization should NOT occur -> low/moderate R^2.
   Sanity control: the same surrogate on the (linear) u_xx target must give
   R^2 = 1.

PRIMARY MODEL: the WIDE checkpoint results/ks_stable_wide/ks_stable_model.pt
([64,96,128], 500 epochs, one-step loss) -- the run the paper cites. Outputs go
to results/ks_stable_wide/lars_recheck/. The committed narrow [64,32,128]
checkpoint can be re-checked with --narrow as a secondary robustness point
(-> results/ks_stable/lars_recheck_narrow/).

Outputs: summary.json + lars_recheck.png/pdf.
Does NOT retrain, does NOT edit docs/*.tex, does NOT overwrite existing results.
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root
sys.path.insert(0, _HERE)                     # experiments/ (for ks_stable)

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Lasso, LassoLars
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import KANAutoEncoder
from manifold_analysis import (build_polynomial_dict_nd, grad_polynomial_nd,
                               compute_network_gradient_nd,
                               recover_outer_function_from_raw, eval_h)

# Reuse the exact stable-Koopman class + constants from the training script.
from ks_stable import DeepKoopmanCTStable

# =====================================================================
# Configuration
# =====================================================================

ROOT = Path(__file__).resolve().parent.parent
WIDE_DIR = ROOT / 'results' / 'ks_stable_wide'      # [64,96,128], 500 epochs
NARROW_DIR = ROOT / 'results' / 'ks_stable'         # [64,32,128], 100 epochs
DATA_PATH = NARROW_DIR / 'ks_data.pt'               # (30, 1000, 64)
# Primary (headline) analysis is the WIDE model -> results/ks_stable_wide/.
# The narrow checkpoint is only a secondary robustness point.
OUTDIR = WIDE_DIR / 'lars_recheck'
OUTDIR_NARROW = NARROW_DIR / 'lars_recheck_narrow'

N_GRID = 64
L_DOMAIN = 22.0
DX = L_DOMAIN / N_GRID
LATENT_SIZE = 128
N_TRAIN_TRAJ = 24            # decomposition used the training trajectories

N_SAMPLE = 4000              # snapshots for the level-set analysis
SEED = 0

# The alpha the ORIGINAL KS decomposition used (ks_stable.py:
# full_decomposition_nd(..., max_degree=2, lasso_alpha=1e-4, h_degree=4)).
LASSO_ALPHA = 1e-4
MAX_DEGREE = 2
H_DEGREE = 4

CD_MAX_ITER = 100000         # original used 50000; we give CD every chance
SPARSITY_THRESHOLD = 1e-4    # the original criterion, on UNSCALED coefficients
N_JOBS = 12                  # parallel CD-Lasso fits (design is memmapped)

# Grid points for the aliasing contrast (spread around the periodic domain).
ALIAS_POINTS = [8, 20, 32, 44, 56]

# The Mackey-Glass reference number for the analogous surrogate test.
MG_ALIAS_R2 = 0.9993


# =====================================================================
# Model / data loading (no retraining)
# =====================================================================

def load_model_and_data(prefer_wide=True):
    """Load the KS checkpoint (wide if available) and a snapshot sample."""
    wide_ckpt = WIDE_DIR / 'ks_stable_model.pt'
    narrow_ckpt = NARROW_DIR / 'ks_stable_model.pt'

    if prefer_wide and wide_ckpt.exists():
        ckpt_path, which = wide_ckpt, 'wide'
    else:
        ckpt_path, which = narrow_ckpt, 'narrow'

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    mid = cfg.get('mid_size', 96)
    latent = cfg.get('latent_size', LATENT_SIZE)

    # Same construction as ks_stable.py's --skip-train branch.
    ae = KANAutoEncoder(N_GRID, mid, latent,
                        grid_range_l1=[-5, 5], grid_range_l2=[-10, 10])
    model = DeepKoopmanCTStable(ae, latent)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    hist = ckpt.get('history', None)
    final_val = float(hist['val_total'][-1]) if hist else float('nan')

    traj = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    # The original decomposition ran on the TRAINING trajectories.
    flat = traj[:N_TRAIN_TRAJ].reshape(-1, N_GRID)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(flat.shape[0])[:N_SAMPLE]
    x_data = flat[idx].contiguous()

    info = {
        'checkpoint': str(ckpt_path.relative_to(ROOT)),
        'which_model': which,
        'arch': [N_GRID, mid, latent],
        'epochs': cfg.get('epochs', None),
        'final_val_total': final_val,
        'n_snapshots': int(x_data.shape[0]),
        'pool_snapshots': int(flat.shape[0]),
    }
    return model, x_data, traj, latent, info


# =====================================================================
# Dual-solver level-set decomposer (design matrix built ONCE)
# =====================================================================

def _unscale(coef_std, intercept_std, col_mean, col_std, f_mean, f_std,
             const_idx):
    """Map standardized-space coefficients back to raw monomial coefficients,
    folding the intercept into the constant term (identical to
    manifold_analysis.recover_inner_function_nd)."""
    coeffs = f_std * coef_std / col_std
    intercept = f_mean + f_std * (intercept_std -
                                  np.sum(coef_std * col_mean / col_std))
    coeffs = coeffs.copy()
    coeffs[const_idx] += intercept
    return coeffs


def _r2(y, pred):
    ss = np.sum((y - pred) ** 2)
    tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss / max(tot, 1e-12))


def _cd_fit_one(Ms, f_scaled, alpha, max_iter):
    """One coordinate-descent Lasso fit; returns coef, intercept, diagnostics.

    Run in a joblib worker (Ms is memmapped, not copied)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        las = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=True)
        las.fit(Ms, f_scaled)
        conv_warn = any(issubclass(w.category, ConvergenceWarning)
                        for w in caught)
    n_iter = int(np.atleast_1d(las.n_iter_)[0])
    dual_gap = float(np.atleast_1d(las.dual_gap_)[0])
    return las.coef_.copy(), float(las.intercept_), n_iter, dual_gap, conv_warn


class DualDecomposer:
    """Level-set inner/outer decomposition with the degree-2 design built once.

    Inner g is fit with BOTH coordinate-descent Lasso and LassoLars at the same
    alpha; the outer h pipeline (gradient identity) runs on the LARS inner,
    exactly as manifold_analysis.full_decomposition_nd does on the CD inner.
    """

    def __init__(self, model, x_data, alpha=LASSO_ALPHA, max_degree=MAX_DEGREE):
        self.model = model
        self.x = x_data
        self.xn = x_data.numpy().astype(float)
        self.alpha = alpha
        self.var_names = [f'u_{j}' for j in range(N_GRID)]

        t0 = time.time()
        self.M, self.names, self.powers = build_polynomial_dict_nd(
            self.xn, self.var_names, max_degree=max_degree)
        self.build_time = time.time() - t0

        self.cm = self.M.mean(0)
        self.cs = self.M.std(0)
        self.cs[self.cs < 1e-12] = 1.0
        self.Ms = (self.M - self.cm) / self.cs
        self.const_idx = self.powers.index(tuple([0] * N_GRID))
        self.pow_arr = np.asarray(self.powers)
        self.deg = self.pow_arr.sum(1)

        with torch.no_grad():
            self.z_all = model.ae.encoder(x_data).numpy().astype(float)
        self.z_var = self.z_all.var(0)

    # ---- targets -----------------------------------------------------
    def target(self, k):
        f = self.z_all[:, k]
        fm = float(f.mean())
        fs = float(f.std()) or 1.0
        return f, fm, fs

    # ---- inner fits --------------------------------------------------
    def fit_lars(self, k):
        f, fm, fs = self.target(k)
        ll = LassoLars(alpha=self.alpha, fit_intercept=True)
        ll.fit(self.Ms, (f - fm) / fs)
        coeffs = _unscale(ll.coef_, ll.intercept_, self.cm, self.cs, fm, fs,
                          self.const_idx)
        return self._summarize_inner(f, coeffs, ll.coef_)

    def cd_targets(self):
        """Standardized targets for all dims (for the parallel CD sweep)."""
        out = []
        for k in range(self.z_all.shape[1]):
            f, fm, fs = self.target(k)
            out.append((f - fm) / fs)
        return out

    def finish_cd(self, k, coef_std, intercept_std):
        f, fm, fs = self.target(k)
        coeffs = _unscale(coef_std, intercept_std, self.cm, self.cs, fm, fs,
                          self.const_idx)
        return self._summarize_inner(f, coeffs, coef_std)

    def _summarize_inner(self, f, coeffs, coef_std):
        """Nonzero counts + R^2, raw and under the original 1e-4 threshold."""
        g_raw = self.M @ coeffs
        r2_raw = _r2(f, g_raw)

        mask = np.abs(coeffs) > SPARSITY_THRESHOLD
        coeffs_thr = np.where(mask, coeffs, 0.0)
        g_thr = self.M @ coeffs_thr
        r2_thr = _r2(f, g_thr)

        return {
            'coeffs': coeffs_thr,          # thresholded, as the original does
            'coeffs_raw': coeffs,
            'nnz_exact': int(np.sum(np.abs(coef_std) > 0)),
            'nnz_1e4': int(mask.sum()),
            'r2_raw': r2_raw,
            'r2_thresh': r2_thr,
        }

    # ---- outer h (gradient identity), on a given inner ----------------
    def outer_pipeline(self, k, coeffs):
        grad_f, f_vals = compute_network_gradient_nd(self.model, self.x, k,
                                                     'latent')
        grad_g = grad_polynomial_nd(self.xn, coeffs, self.powers)
        g_vals = self.M @ coeffs
        dot = np.sum(grad_f * grad_g, axis=1)
        nsq = np.sum(grad_g ** 2, axis=1)
        outer = recover_outer_function_from_raw(dot, nsq, g_vals, f_vals,
                                                 poly_degree=H_DEGREE)
        f_hat = eval_h(g_vals, outer)
        r2_hog = _r2(f_vals, f_hat)

        A = np.column_stack([g_vals, np.ones_like(g_vals)])
        c, *_ = np.linalg.lstsq(A, f_vals, rcond=None)
        r2_aff = _r2(f_vals, A @ c)

        return {
            'r2_affine': r2_aff,
            'r2_hog': r2_hog,
            'outer_gain': r2_hog - r2_aff,
            'used_affine': bool(outer.get('used_affine', True)),
        }

    # ---- does the quadratic part earn its place? ----------------------
    def linear_baselines(self, k, coeffs):
        """Two controls that decide 'sparse near-linear filter' (B) vs
        'sparse WITH genuine advection quadratics' (C):

          r2_linear_ols  : unpenalized OLS on ALL 64 linear terms + intercept.
                           The strongest possible purely-linear readout of the
                           snapshot -- an upper bound on narrative B.
          r2_lars_linpart: the recovered LARS dictionary with every degree-2
                           atom zeroed. Measures the quadratic contribution
                           WITHIN the discovered dictionary.
        """
        f = self.z_all[:, k]
        r2_ols_in, r2_ols_out = ols_r2(self.xn, f)

        lin_only = np.where(self.deg <= 1, coeffs, 0.0)
        # refit the constant so the linear part is not penalised by an offset
        g_lin = self.M @ lin_only
        A = np.column_stack([g_lin, np.ones_like(g_lin)])
        c, *_ = np.linalg.lstsq(A, f, rcond=None)
        r2_linpart = _r2(f, A @ c)

        g_full = self.M @ coeffs
        r2_full = _r2(f, g_full)
        return {
            'r2_linear_ols': r2_ols_in,
            'r2_linear_ols_holdout': r2_ols_out,
            'r2_lars_linear_part': r2_linpart,
            'quad_gain_vs_linpart': r2_full - r2_linpart,
            'quad_gain_vs_ols': r2_full - r2_ols_in,
        }

    # ---- atom classification -----------------------------------------
    def classify_atoms(self, coeffs):
        """Split the surviving atoms into const / linear / u_i^2 / u_i u_j and
        report the periodic separation of the cross terms."""
        act = np.where(np.abs(coeffs) > SPARSITY_THRESHOLD)[0]
        n_const = n_lin = n_sq = n_cross = 0
        m_const = m_lin = m_sq = m_cross = 0.0
        sep_counts, sep_mass = {}, {}
        cross_examples = []
        linear_examples = []

        for i in act:
            c = float(coeffs[i])
            p = self.pow_arr[i]
            d = int(p.sum())
            if d == 0:
                n_const += 1; m_const += abs(c); continue
            nz = np.nonzero(p)[0]
            if d == 1:
                n_lin += 1; m_lin += abs(c)
                linear_examples.append((self.names[i], c))
            elif len(nz) == 1:                      # u_i^2
                n_sq += 1; m_sq += abs(c)
            else:                                   # u_i u_j
                n_cross += 1; m_cross += abs(c)
                a, b = int(nz[0]), int(nz[1])
                sep = abs(a - b)
                sep = min(sep, N_GRID - sep)        # periodic domain
                sep_counts[sep] = sep_counts.get(sep, 0) + 1
                sep_mass[sep] = sep_mass.get(sep, 0.0) + abs(c)
                cross_examples.append((self.names[i], c, sep))

        cross_examples.sort(key=lambda t: -abs(t[1]))
        linear_examples.sort(key=lambda t: -abs(t[1]))
        tot_mass = m_const + m_lin + m_sq + m_cross + 1e-30
        return {
            'n_const': n_const, 'n_linear': n_lin,
            'n_square': n_sq, 'n_cross': n_cross,
            'mass_frac_linear': m_lin / tot_mass,
            'mass_frac_square': m_sq / tot_mass,
            'mass_frac_cross': m_cross / tot_mass,
            'mass_frac_quadratic': (m_sq + m_cross) / tot_mass,
            'cross_sep_counts': {int(k): int(v) for k, v in sorted(sep_counts.items())},
            'cross_sep_mass': {int(k): float(v) for k, v in sorted(sep_mass.items())},
            'top_cross': [(n, float(c), int(s)) for n, c, s in cross_examples[:6]],
            'top_linear': [(n, float(c)) for n, c in linear_examples[:6]],
        }


# =====================================================================
# Aliasing contrast: is the KS nonlinearity linearly readable from a snapshot?
# =====================================================================

def separation_enrichment(sep_counts, n_dims):
    """Normalize cross-term separation counts by the number of AVAILABLE pairs.

    On a periodic 64-point grid there are N pairs at each separation
    s = 1..N/2-1 and N/2 pairs at s = N/2, so raw counts must be divided by
    availability before "near-diagonal enrichment" means anything. Enrichment
    = (rate at separation s) / (overall rate over all pairs). Advection u u_x
    is a separation-1 stencil, so a clean advection dictionary would put nearly
    all its enrichment at s = 1.
    """
    avail = {s: (N_GRID if s < N_GRID // 2 else N_GRID // 2)
             for s in range(1, N_GRID // 2 + 1)}
    total_pairs = sum(avail.values())          # = C(64,2) = 2016
    total_atoms = sum(sep_counts.values())
    base_rate = total_atoms / max(total_pairs * n_dims, 1)
    enr = {}
    for s, a in avail.items():
        rate = sep_counts.get(s, 0) / (a * n_dims)
        enr[s] = rate / base_rate if base_rate > 0 else 0.0
    near = sum(sep_counts.get(s, 0) for s in (1, 2))
    near_avail = avail[1] + avail[2]
    far = total_atoms - near
    far_avail = total_pairs - near_avail
    return {
        'enrichment_by_separation': {int(s): float(v) for s, v in enr.items()},
        'enrichment_sep1': float(enr.get(1, 0.0)),
        'enrichment_sep1_2': float((near / max(near_avail * n_dims, 1)) /
                                   max(base_rate, 1e-30)),
        'frac_atoms_at_sep1': float(sep_counts.get(1, 0) / max(total_atoms, 1)),
        'frac_atoms_at_sep_le2': float(near / max(total_atoms, 1)),
        'frac_pairs_at_sep_le2': float(near_avail / total_pairs),
        'far_atom_count': int(far), 'far_pair_count': int(far_avail),
        'base_rate': float(base_rate),
    }


def ols_r2(X, y, split=True):
    """In-sample (and held-out) R^2 of an OLS linear fit with intercept."""
    A = np.column_stack([X, np.ones(len(y))])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2_in = _r2(y, A @ c)
    if not split:
        return r2_in, float('nan')
    n = len(y) // 2
    c2, *_ = np.linalg.lstsq(A[:n], y[:n], rcond=None)
    r2_out = _r2(y[n:], A[n:] @ c2)
    return r2_in, r2_out


def aliasing_contrast(x_np):
    """Fit LINEAR surrogates on the 64 snapshot values to nonlinear/linear
    spatial targets.

    Advection target  n_i = u_i (u_{i+1} - u_{i-1}) / (2 dx)   -- genuinely
    quadratic; a snapshot carries no temporal information, so unlike the MG
    delay-window case there is no aliasing route by which a linear readout can
    reproduce it.

    Control target    (u_xx)_i = (u_{i+1} - 2 u_i + u_{i-1}) / dx^2  -- exactly
    linear in the snapshot, so the surrogate must return R^2 = 1. This confirms
    the low advection R^2 is a property of the target, not of the fit.
    """
    n_pts = x_np.shape[1]
    up = np.roll(x_np, -1, axis=1)     # u_{i+1}
    um = np.roll(x_np, 1, axis=1)      # u_{i-1}
    advection = x_np * (up - um) / (2 * DX)
    uxx = (up - 2 * x_np + um) / DX ** 2

    per_point = {}
    adv_all, ctl_all = [], []
    for i in range(n_pts):
        r2_in, r2_out = ols_r2(x_np, advection[:, i])
        r2c_in, _ = ols_r2(x_np, uxx[:, i], split=False)
        adv_all.append(r2_in)
        ctl_all.append(r2c_in)
        if i in ALIAS_POINTS:
            per_point[i] = {
                'advection_r2_in': r2_in,
                'advection_r2_holdout': r2_out,
                'uxx_control_r2': r2c_in,
            }
    return {
        'representative_points': per_point,
        'advection_r2_all64': {
            'min': float(np.min(adv_all)), 'median': float(np.median(adv_all)),
            'mean': float(np.mean(adv_all)), 'max': float(np.max(adv_all)),
        },
        'uxx_control_r2_all64': {
            'min': float(np.min(ctl_all)), 'median': float(np.median(ctl_all)),
        },
        'mg_reference_r2': MG_ALIAS_R2,
    }


# =====================================================================
# Figure
# =====================================================================

def make_figure(rows, alias, info, agg, outdir):
    cd_nnz = np.array([r['cd']['nnz_1e4'] for r in rows], float)
    lars_nnz = np.array([r['lars']['nnz_1e4'] for r in rows], float)
    cd_r2 = np.array([r['cd']['r2_thresh'] for r in rows], float)
    lars_r2 = np.array([r['lars']['r2_thresh'] for r in rows], float)
    dims = np.array([r['dim'] for r in rows])
    alive = np.array([not r['dead'] for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # (1) nonzeros per dim, both solvers
    ax = axes[0, 0]
    ax.plot(dims, cd_nnz, 'o-', ms=3, lw=0.9, color='crimson',
            label=f'CD Lasso (median {np.median(cd_nnz):.0f})')
    ax.plot(dims, lars_nnz, 's-', ms=3, lw=0.9, color='steelblue',
            label=f'LassoLars (median {np.median(lars_nnz):.0f})')
    ax.axhspan(110, 190, color='gray', alpha=0.15,
               label='claimed 110-190 range')
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel('latent dim $k$')
    ax.set_ylabel('# active monomials  ($|c| > 10^{-4}$)')
    ax.set_title(f'Inner $g$ sparsity, $\\alpha={LASSO_ALPHA:g}$, '
                 f'2145 degree-2 monomials')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # (2) distributions
    ax = axes[0, 1]
    bins = np.logspace(0, np.log10(max(cd_nnz.max(), 10)), 30)
    ax.hist(np.clip(cd_nnz, 1, None), bins=bins, alpha=0.6, color='crimson',
            label='CD Lasso')
    ax.hist(np.clip(lars_nnz, 1, None), bins=bins, alpha=0.6, color='steelblue',
            label='LassoLars')
    ax.set_xscale('log')
    ax.set_xlabel('# active monomials')
    ax.set_ylabel('# latent dims')
    ax.set_title('Active-term distribution over 128 dims')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # (3) R^2: CD vs LARS vs purely-linear controls
    ax = axes[0, 2]
    ols_r2v = np.array([r['baselines']['r2_linear_ols'] for r in rows], float)
    linpart = np.array([r['baselines']['r2_lars_linear_part'] for r in rows], float)
    o = np.argsort(-lars_r2)
    ax.plot(lars_r2[o], 's-', ms=2.5, lw=1.0, color='steelblue',
            label=f'LassoLars deg-2 (med {np.median(lars_r2):.4f})')
    ax.plot(cd_r2[o], 'o-', ms=2.5, lw=1.0, color='crimson', alpha=0.7,
            label=f'CD Lasso deg-2 (med {np.median(cd_r2):.4f})')
    ax.plot(ols_r2v[o], '-', lw=1.2, color='darkgreen',
            label=f'OLS, 64 LINEAR terms (med {np.median(ols_r2v):.4f})')
    ax.plot(linpart[o], '--', lw=1.0, color='gray',
            label=f'LARS linear part only (med {np.median(linpart):.4f})')
    ax.set_xlabel('latent dim (sorted by LARS $R^2$)')
    ax.set_ylabel('$R^2(g)$')
    ax.set_title('Does the quadratic part earn its place?')
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(alpha=0.2)

    # (4) atom composition under LARS
    ax = axes[1, 0]
    comp = np.array([[r['atoms']['n_linear'], r['atoms']['n_square'],
                      r['atoms']['n_cross']] for r in rows], float)
    order = np.argsort(-comp.sum(1))
    bottom = np.zeros(len(rows))
    for j, (lbl, col) in enumerate([('linear $u_i$', '#4C78A8'),
                                    ('diagonal $u_i^2$', '#F58518'),
                                    ('cross $u_i u_j$', '#54A24B')]):
        ax.bar(range(len(rows)), comp[order, j], bottom=bottom, color=col,
               width=1.0, label=lbl)
        bottom += comp[order, j]
    ax.set_xlabel('latent dim (sorted by # atoms)')
    ax.set_ylabel('# atoms (LassoLars)')
    ax.set_title('What survives LARS')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis='y')

    # (5) cross-term spatial separation, normalized by available pairs
    #     (advection u u_x is a separation-1 stencil)
    ax = axes[1, 1]
    enr = agg['separation_enrichment']['enrichment_by_separation']
    if enr:
        ks_ = sorted(int(s) for s in enr)
        vals = [enr[s] if s in enr else enr[str(s)] for s in ks_]
        cols = ['#d62728' if s <= 2 else '#54A24B' for s in ks_]
        ax.bar(ks_, vals, color=cols)
        ax.axhline(1.0, color='k', ls='--', lw=1,
                   label='uniform over all pairs')
        ax.legend(fontsize=8)
        ax.set_ylabel('enrichment  (rate / uniform rate)')
    else:
        ax.text(0.5, 0.5, 'no cross terms survived', ha='center',
                transform=ax.transAxes)
    ax.set_xlabel('periodic separation $|i-j|$')
    ax.set_title(f"Cross-term geometry under LARS\n"
                 f"sep-1 enrichment "
                 f"{agg['separation_enrichment']['enrichment_sep1']:.2f}x, but "
                 f"{100 * (1 - agg['separation_enrichment']['frac_atoms_at_sep_le2']):.0f}% "
                 f"of atoms sit at $|i-j| > 2$")
    ax.grid(alpha=0.2, axis='y')

    # (6) aliasing contrast
    ax = axes[1, 2]
    # keys may be ints (fresh run) or strings (reloaded from JSON)
    pts = sorted(alias['representative_points'], key=lambda s: int(s))
    vals = [alias['representative_points'][p]['advection_r2_in'] for p in pts]
    hold = [alias['representative_points'][p]['advection_r2_holdout'] for p in pts]
    ctl = [alias['representative_points'][p]['uxx_control_r2'] for p in pts]
    xx = np.arange(len(pts))
    ax.bar(xx - 0.26, vals, width=0.26, color='#4C78A8',
           label='$u\\,u_x$ target, in-sample')
    ax.bar(xx, hold, width=0.26, color='#9ecae1',
           label='$u\\,u_x$ target, held out')
    ax.bar(xx + 0.26, ctl, width=0.26, color='#BAB0AC',
           label='$u_{xx}$ control (linear)')
    ax.axhline(MG_ALIAS_R2, color='crimson', ls='--', lw=1.5,
               label=f'MG Hill surrogate ({MG_ALIAS_R2})')
    ax.set_xticks(xx)
    ax.set_xticklabels([f'$i={p}$' for p in pts])
    ax.set_ylabel('$R^2$ of LINEAR surrogate')
    ax.set_ylim(0, 1.08)
    ax.set_title('Aliasing diagnostic: is the nonlinearity\n'
                 'linearly readable from the state?')
    ax.legend(fontsize=8, loc='center right')
    ax.grid(alpha=0.2, axis='y')

    fig.suptitle(f"KS level-set dictionary re-check ({info['which_model']} model "
                 f"{info['arch']}, {info['n_snapshots']} snapshots)", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / 'lars_recheck.png', dpi=190, bbox_inches='tight')
    fig.savefig(outdir / 'lars_recheck.pdf', bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    global N_SAMPLE
    ap = argparse.ArgumentParser()
    ap.add_argument('--dims', type=int, default=None,
                    help='limit number of latent dims (smoke test)')
    ap.add_argument('--n-sample', type=int, default=N_SAMPLE)
    ap.add_argument('--narrow', action='store_true',
                    help='force the committed narrow checkpoint')
    ap.add_argument('--jobs', type=int, default=N_JOBS)
    ap.add_argument('--skip-cd', action='store_true',
                    help='skip the coordinate-descent sweep (LARS only)')
    ap.add_argument('--outdir', type=str, default=None,
                    help='override output directory')
    ap.add_argument('--replot', action='store_true',
                    help='recompute aggregates + figure from an existing '
                         'summary.json (no model load, no refitting)')
    args = ap.parse_args()

    N_SAMPLE = args.n_sample
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = OUTDIR_NARROW if args.narrow else OUTDIR

    t_start = time.time()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        replot(outdir)
        return

    print('=' * 72, flush=True)
    print('KS level-set dictionary re-check: CD-Lasso vs exact LARS', flush=True)
    print('=' * 72, flush=True)

    model, x_data, traj, latent, info = load_model_and_data(
        prefer_wide=not args.narrow)
    print(f"Model: {info['which_model']} {info['arch']} from {info['checkpoint']}",
          flush=True)
    print(f"  epochs={info['epochs']}  final val_total={info['final_val_total']:.6f}",
          flush=True)
    print(f"  snapshots: {info['n_snapshots']} of {info['pool_snapshots']} "
          f"(training trajectories, seed {SEED})", flush=True)

    dec = DualDecomposer(model, x_data, alpha=LASSO_ALPHA)
    n_terms = dec.M.shape[1]
    print(f"  design: {dec.M.shape[0]} x {n_terms} degree-{MAX_DEGREE} monomials "
          f"(built in {dec.build_time:.1f}s)", flush=True)

    n_dims = latent if args.dims is None else min(args.dims, latent)

    # ---- CD-Lasso sweep (parallel; design memmapped by joblib) --------
    cd_raw = {}
    t_cd = 0.0
    if not args.skip_cd:
        print(f"\n--- Coordinate-descent Lasso (max_iter={CD_MAX_ITER}), "
              f"{n_dims} dims on {args.jobs} workers ---", flush=True)
        targets = dec.cd_targets()[:n_dims]
        t0 = time.time()
        res = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(_cd_fit_one)(dec.Ms, targets[k], LASSO_ALPHA, CD_MAX_ITER)
            for k in range(n_dims))
        t_cd = time.time() - t0
        for k, r in enumerate(res):
            cd_raw[k] = r
        print(f"  CD sweep: {t_cd:.1f}s "
              f"({t_cd / max(n_dims, 1):.2f}s/dim)", flush=True)

    # ---- LARS + outer pipeline ---------------------------------------
    print(f"\n--- LassoLars + gradient-identity outer h, {n_dims} dims ---",
          flush=True)
    rows = []
    t_lars = t_outer = 0.0
    for k in range(n_dims):
        dead = bool(dec.z_var[k] < 1e-8)

        t0 = time.time()
        lars = dec.fit_lars(k)
        t_lars += time.time() - t0

        if k in cd_raw:
            coef_std, icept, n_iter, dual_gap, warned = cd_raw[k]
            cd = dec.finish_cd(k, coef_std, icept)
            cd.update({'n_iter': n_iter, 'dual_gap': dual_gap,
                       'converged': bool((not warned) and n_iter < CD_MAX_ITER)})
        else:
            cd = {'nnz_exact': -1, 'nnz_1e4': -1, 'r2_raw': float('nan'),
                  'r2_thresh': float('nan'), 'n_iter': -1, 'dual_gap': float('nan'),
                  'converged': None, 'coeffs': None, 'coeffs_raw': None}

        t0 = time.time()
        try:
            outer = dec.outer_pipeline(k, lars['coeffs'])
        except Exception as e:
            outer = {'r2_affine': float('nan'), 'r2_hog': float('nan'),
                     'outer_gain': float('nan'), 'used_affine': None,
                     'error': str(e)}
        t_outer += time.time() - t0

        atoms = dec.classify_atoms(lars['coeffs'])
        base = dec.linear_baselines(k, lars['coeffs'])

        rows.append({
            'dim': k,
            'z_var': float(dec.z_var[k]),
            'dead': dead,
            'baselines': base,
            'cd': {kk: vv for kk, vv in cd.items()
                   if kk not in ('coeffs', 'coeffs_raw')},
            'lars': {kk: vv for kk, vv in lars.items()
                     if kk not in ('coeffs', 'coeffs_raw')},
            'outer': outer,
            'atoms': atoms,
        })

        if (k + 1) % 16 == 0 or k == n_dims - 1:
            print(f"  {k+1}/{n_dims} dims  "
                  f"(last: CD nnz {cd['nnz_1e4']}, LARS nnz {lars['nnz_1e4']}, "
                  f"R2 {cd['r2_thresh']:.4f} vs {lars['r2_thresh']:.4f}, "
                  f"R2(hog) {outer['r2_hog']:.4f})", flush=True)
    print(f"  LARS fits {t_lars:.1f}s, outer pipeline {t_outer:.1f}s", flush=True)

    # ---- Aliasing contrast -------------------------------------------
    print('\n--- Aliasing contrast (linear surrogate for u*u_x) ---', flush=True)
    alias = aliasing_contrast(dec.xn)
    for p in sorted(alias['representative_points']):
        d = alias['representative_points'][p]
        print(f"  i={p:2d}:  advection R2 = {d['advection_r2_in']:.4f} "
              f"(holdout {d['advection_r2_holdout']:.4f})   "
              f"u_xx control R2 = {d['uxx_control_r2']:.6f}", flush=True)
    print(f"  all 64 points, advection R2: "
          f"median {alias['advection_r2_all64']['median']:.4f}, "
          f"range [{alias['advection_r2_all64']['min']:.4f}, "
          f"{alias['advection_r2_all64']['max']:.4f}]", flush=True)
    print(f"  MG reference (Hill surrogate on delay window): {MG_ALIAS_R2}",
          flush=True)

    # Attractor dimensionality: context for how easily a quadratic target can
    # be linearly aliased on a low-dimensional attractor.
    Uc = dec.xn - dec.xn.mean(0)
    sv = np.linalg.svd(Uc, compute_uv=False)
    ev = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    pod = {'n_modes_99pct': int(np.searchsorted(ev, 0.99) + 1),
           'n_modes_999pct': int(np.searchsorted(ev, 0.999) + 1),
           'n_modes_9999pct': int(np.searchsorted(ev, 0.9999) + 1)}
    print(f"  POD: {pod['n_modes_99pct']} modes for 99% variance, "
          f"{pod['n_modes_999pct']} for 99.9%, "
          f"{pod['n_modes_9999pct']} for 99.99% (of 64)", flush=True)
    alias['pod_dimension'] = pod

    agg = compute_aggregates(rows, n_dims, n_terms)
    print_aggregates(agg, rows, n_dims)

    wall = time.time() - t_start
    summary = {
        'experiment': 'ks_lars_recheck',
        'question': ('Does the KS "dense dictionary" (110-190 active degree-2 '
                     'monomials per latent dim) survive an exact LARS solver, '
                     'or is it a coordinate-descent artifact as on Mackey-Glass?'),
        'model': info,
        'config': {
            'alpha': LASSO_ALPHA, 'max_degree': MAX_DEGREE,
            'h_degree': H_DEGREE, 'cd_max_iter': CD_MAX_ITER,
            'cd_max_iter_original': 50000,
            'sparsity_threshold': SPARSITY_THRESHOLD,
            'n_sample': int(dec.M.shape[0]), 'n_monomials': int(n_terms),
            'seed': SEED, 'L': L_DOMAIN, 'dx': DX, 'N_grid': N_GRID,
        },
        'aggregates': agg,
        'aliasing_contrast': alias,
        'per_dim': rows,
        'timing_sec': {'cd_sweep': t_cd, 'lars_fits': t_lars,
                       'outer_pipeline': t_outer, 'wall_clock': wall},
    }
    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  Saved {outdir / 'summary.json'}", flush=True)

    make_figure(rows, alias, info, agg, outdir)
    print(f"  Saved {outdir / 'lars_recheck.png'} / .pdf", flush=True)
    print(f"\n  Wall clock: {wall:.1f}s", flush=True)


def compute_aggregates(rows, n_dims, n_terms):
    """Roll the per-dim table up into the numbers the verdict rests on."""
    cd_nnz = np.array([r['cd']['nnz_1e4'] for r in rows], float)
    lars_nnz = np.array([r['lars']['nnz_1e4'] for r in rows], float)
    cd_r2 = np.array([r['cd']['r2_thresh'] for r in rows], float)
    lars_r2 = np.array([r['lars']['r2_thresh'] for r in rows], float)
    hog = np.array([r['outer']['r2_hog'] for r in rows], float)
    aff = np.array([r['outer']['r2_affine'] for r in rows], float)
    gain = np.array([r['outer']['outer_gain'] for r in rows], float)
    n_cross = np.array([r['atoms']['n_cross'] for r in rows], float)
    n_lin = np.array([r['atoms']['n_linear'] for r in rows], float)
    n_sq = np.array([r['atoms']['n_square'] for r in rows], float)
    qmass = np.array([r['atoms']['mass_frac_quadratic'] for r in rows], float)
    r2_ols = np.array([r['baselines']['r2_linear_ols'] for r in rows], float)
    r2_linpart = np.array([r['baselines']['r2_lars_linear_part'] for r in rows], float)
    qgain_lp = np.array([r['baselines']['quad_gain_vs_linpart'] for r in rows], float)
    qgain_ols = np.array([r['baselines']['quad_gain_vs_ols'] for r in rows], float)

    def stats(a):
        a = np.asarray(a, float)
        good = a[np.isfinite(a)]
        if good.size == 0:
            return {}
        return {'min': float(good.min()), 'p25': float(np.percentile(good, 25)),
                'median': float(np.median(good)),
                'p75': float(np.percentile(good, 75)),
                'max': float(good.max()), 'mean': float(good.mean())}

    sep_agg = {}
    for r in rows:
        for s, c in r['atoms']['cross_sep_counts'].items():
            sep_agg[int(s)] = sep_agg.get(int(s), 0) + int(c)

    n_conv = sum(1 for r in rows if r['cd'].get('converged') is True)
    n_cdrun = sum(1 for r in rows if r['cd'].get('converged') is not None)

    agg = {
        'n_dims': n_dims,
        'n_monomials': int(n_terms),
        'cd_nnz': stats(cd_nnz), 'lars_nnz': stats(lars_nnz),
        'cd_r2': stats(cd_r2), 'lars_r2': stats(lars_r2),
        'r2_hog': stats(hog), 'r2_affine': stats(aff), 'outer_gain': stats(gain),
        'n_linear_atoms': stats(n_lin), 'n_square_atoms': stats(n_sq),
        'n_cross_atoms': stats(n_cross),
        'quadratic_mass_frac': stats(qmass),
        'r2_linear_ols': stats(r2_ols),
        'r2_lars_linear_part': stats(r2_linpart),
        'quad_gain_vs_linear_part': stats(qgain_lp),
        'quad_gain_vs_linear_ols': stats(qgain_ols),
        'dims_where_quad_gain_gt_0p01': int(np.sum(qgain_ols > 0.01)),
        'cd_converged_dims': f'{n_conv}/{n_cdrun}',
        'cd_max_n_iter': int(max([r['cd']['n_iter'] for r in rows] + [-1])),
        'cd_max_dual_gap': float(np.nanmax([r['cd']['dual_gap'] for r in rows])),
        'total_cross_atoms': int(n_cross.sum()),
        'cross_sep_counts_all_dims': {int(k): int(v) for k, v in sorted(sep_agg.items())},
        'dims_with_zero_cross': int(np.sum(n_cross == 0)),
        'separation_enrichment': separation_enrichment(sep_agg, n_dims),
        'sparsity_reduction_median': float(np.median(cd_nnz) / max(np.median(lars_nnz), 1e-9)),
    }
    return agg


def print_aggregates(agg, rows, n_dims):
    print('\n' + '=' * 72, flush=True)
    print('AGGREGATES', flush=True)
    print('=' * 72, flush=True)
    print(f"  CD-Lasso  nnz: min {agg['cd_nnz']['min']:.0f}  "
          f"median {agg['cd_nnz']['median']:.0f}  max {agg['cd_nnz']['max']:.0f}"
          f"   (converged {agg['cd_converged_dims']})", flush=True)
    print(f"  LassoLars nnz: min {agg['lars_nnz']['min']:.0f}  "
          f"median {agg['lars_nnz']['median']:.0f}  max {agg['lars_nnz']['max']:.0f}",
          flush=True)
    print(f"  R2(g): CD median {agg['cd_r2']['median']:.5f}  "
          f"LARS median {agg['lars_r2']['median']:.5f}", flush=True)
    print(f"  R2(affine) median {agg['r2_affine']['median']:.4f}  "
          f"R2(h o g) median {agg['r2_hog']['median']:.4f}  "
          f"gain median {agg['outer_gain']['median']:.4f}", flush=True)
    print(f"  LARS atoms: linear median {agg['n_linear_atoms']['median']:.0f}, "
          f"u_i^2 median {agg['n_square_atoms']['median']:.0f}, "
          f"cross median {agg['n_cross_atoms']['median']:.0f} "
          f"(total {agg['total_cross_atoms']}, "
          f"{agg['dims_with_zero_cross']} dims with none)", flush=True)
    print(f"  cross-term separations (all dims): "
          f"{agg['cross_sep_counts_all_dims']}", flush=True)
    print(f"  LINEAR-ONLY controls: OLS(64 linear) median R2 "
          f"{agg['r2_linear_ols']['median']:.5f}; LARS linear part median R2 "
          f"{agg['r2_lars_linear_part']['median']:.5f}", flush=True)
    se = agg['separation_enrichment']
    print(f"  near-diagonal enrichment (pair-count normalized): "
          f"sep-1 {se['enrichment_sep1']:.2f}x, sep<=2 {se['enrichment_sep1_2']:.2f}x; "
          f"{100 * se['frac_atoms_at_sep_le2']:.1f}% of cross atoms at |i-j|<=2 "
          f"(vs {100 * se['frac_pairs_at_sep_le2']:.1f}% of available pairs)",
          flush=True)
    print(f"  quadratic gain over best linear readout: median "
          f"{agg['quad_gain_vs_linear_ols']['median']:.5f}, max "
          f"{agg['quad_gain_vs_linear_ols']['max']:.5f} "
          f"({agg['dims_where_quad_gain_gt_0p01']}/{n_dims} dims gain > 0.01)",
          flush=True)

    # top dims by |cross-term| presence, for the report
    print('\n  Example LARS dictionaries (top 8 dims by latent variance):',
          flush=True)
    order = np.argsort(-np.array([r['z_var'] for r in rows]))[:8]
    for k in order:
        r = rows[int(k)]
        terms = []
        for nm, c in r['atoms']['top_linear'][:3]:
            terms.append(f'{c:+.3f}*{nm}')
        for nm, c, s in r['atoms']['top_cross'][:3]:
            terms.append(f'{c:+.3f}*{nm}[sep{s}]')
        print(f"   z[{r['dim']:3d}] var={r['z_var']:.3f} "
              f"nnz {r['lars']['nnz_1e4']:3d} (CD {r['cd']['nnz_1e4']:3d}) "
              f"R2g {r['lars']['r2_thresh']:.4f} R2hog {r['outer']['r2_hog']:.4f} "
              f"R2lin {r['baselines']['r2_linear_ols']:.4f} | "
              + '  '.join(terms), flush=True)


def replot(outdir):
    """Recompute aggregates and regenerate the figure from a saved summary.json
    (no model, no refitting) -- used when the reporting layer changes."""
    with open(outdir / 'summary.json') as f:
        summary = json.load(f)
    rows = summary['per_dim']
    n_dims = len(rows)
    n_terms = summary['config']['n_monomials']
    agg = compute_aggregates(rows, n_dims, n_terms)
    print_aggregates(agg, rows, n_dims)
    summary['aggregates'] = agg
    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    make_figure(rows, summary['aliasing_contrast'], summary['model'], agg, outdir)
    print(f"\n  Replotted from {outdir / 'summary.json'}", flush=True)


if __name__ == '__main__':
    main()
