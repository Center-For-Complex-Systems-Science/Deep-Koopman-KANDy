"""
Geometric companion to the KS LassoLars re-check.

The re-check (experiments/ks_lars_recheck.py) established the NUMBERS: at the
same alpha=1e-4, a converged coordinate-descent Lasso keeps ~160 monomials per
latent coordinate while exact LARS keeps ~81, at identical R^2. This script
supplies the GEOMETRY behind those numbers, in the visual language the paper
already uses (results/ks_stable_wide/ks_foliation.png, ks_splines.png):

1. lars_foliation_z{k}.png -- the published 4-panel foliation figure, but with
   g = the LARS-recovered inner function:
     (a) POD-projected attractor colored by the observable z_k
     (b) level sets of g foliating the attractor in POD1-POD2
     (c) z_k vs g with the recovered outer h
     (d) z_k(t) along a validation trajectory
   Plus lars_foliation_compare_z{k}.png: the CD-Lasso g and the LassoLars g
   banded at the SAME numeric levels on the SAME projection. The hypothesis
   under test is that the two solvers describe the SAME activation manifold --
   the foliation is solver-independent, only the atom list differs. Quantified
   by corr(g_CD, g_LARS) over all 128 dims.

2. lars_splines_z{k}.png -- the layer-1 univariate splines phi^(1)_{j,i}(u_i)
   that generate a dim's LARS atoms, for the top-2 hidden nodes j by actual
   layer-2 contribution, at the input sites i carrying that dim's strongest
   atoms, plus the layer-2 splines Phi^(2)_{k,j}. This shows the spline SHAPES
   whose composition produces a quadratic neighbour stencil: a curved layer-2
   acting on a near-linear layer-1 mixture yields products, whereas curvature
   already in layer 1 would mean something else. The script measures the
   curvature of every plotted spline (1 - R^2 of a linear fit) and reports
   which pattern is actually present.

3. lars_atom_pattern.png -- coupling_structure.png-style 64x64 |c_ij| heatmaps
   of the quadratic coefficients, CD vs LARS, for two dims: does LARS keep the
   near-diagonal advection band and drop scattered far pairs, or does it thin
   everything uniformly?

No retraining. Reuses the decomposer, loaders and alpha from ks_lars_recheck,
so the design matrix and sample are identical to the re-check run.
Writes to results/ks_stable_wide/lars_recheck/. Does NOT edit docs/*.tex.
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root
sys.path.insert(0, _HERE)                     # experiments/

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from manifold_analysis import (build_polynomial_dict_nd, grad_polynomial_nd,
                               compute_network_gradient_nd,
                               recover_outer_function_from_raw, eval_h)

from ks_lars_recheck import (DualDecomposer, load_model_and_data, _cd_fit_one,
                             LASSO_ALPHA, CD_MAX_ITER, H_DEGREE,
                             SPARSITY_THRESHOLD, N_GRID, L_DOMAIN, OUTDIR)

# =====================================================================
# Configuration
# =====================================================================

# z19 is the published foliation exemplar (results/ks_stable_wide/ks_foliation.png);
# z91 and z92 carry the strongest separation-1 neighbour atoms found by LARS
# (z91: -0.042*u_59u_60, z92: +0.037*u_4u_5).
FOLIATION_DIMS = [19, 91, 92]
HEATMAP_DIMS = [19, 91]          # coupling_structure-style CD vs LARS panels

N_BANDS = 6                       # level-set bands, as in ks_foliation.png
BAND_FRAC = 0.03                  # half-width as a fraction of the g range
# The inner g is FIT on the same 4000 random snapshots as the re-check, but the
# attractor is DISPLAYED using contiguous validation trajectories: random
# snapshots project to a diffuse cloud, whereas consecutive states trace the
# filaments that make a foliation legible (this is what ks_foliation.png shows).
DISPLAY_TRAJ = [24, 25, 26]       # validation trajectories used for panels a/b
N_SITES_EXTRA = 3                 # extra linear-atom sites in the spline grid
CACHE = 'lars_geometry_cache.npz'


# =====================================================================
# Inner/outer helpers
# =====================================================================

def outer_for(dec, k, coeffs):
    """Recover the outer h for a given inner g (same gradient identity as
    manifold_analysis.full_decomposition_nd)."""
    grad_f, f_vals = compute_network_gradient_nd(dec.model, dec.x, k, 'latent')
    grad_g = grad_polynomial_nd(dec.xn, coeffs, dec.powers)
    g_vals = dec.M @ coeffs
    dot = np.sum(grad_f * grad_g, axis=1)
    nsq = np.sum(grad_g ** 2, axis=1)
    outer = recover_outer_function_from_raw(dot, nsq, g_vals, f_vals,
                                            poly_degree=H_DEGREE)
    f_hat = eval_h(g_vals, outer)
    ss = np.sum((f_vals - f_hat) ** 2)
    tot = np.sum((f_vals - f_vals.mean()) ** 2)
    return g_vals, f_vals, outer, float(1 - ss / max(tot, 1e-12))


def quad_matrix(coeffs, powers_arr, names=None):
    """64x64 matrix of |quadratic coefficients|; diagonal holds u_i^2."""
    Q = np.zeros((N_GRID, N_GRID))
    act = np.where(np.abs(coeffs) > SPARSITY_THRESHOLD)[0]
    for i in act:
        p = powers_arr[i]
        if p.sum() != 2:
            continue
        nz = np.nonzero(p)[0]
        c = abs(float(coeffs[i]))
        if len(nz) == 1:
            Q[nz[0], nz[0]] = c
        else:
            Q[nz[0], nz[1]] = c
            Q[nz[1], nz[0]] = c
    return Q


def near_diag_fraction(Q, width=2):
    """Fraction of |quadratic| energy within `width` of the diagonal, on a
    PERIODIC grid (so wrap-around neighbours count)."""
    tot = np.abs(Q).sum()
    if tot < 1e-15:
        return 0.0
    ii, jj = np.indices(Q.shape)
    sep = np.abs(ii - jj)
    sep = np.minimum(sep, N_GRID - sep)
    return float(np.abs(Q)[sep <= width].sum() / tot)


# =====================================================================
# KAN spline extraction
# =====================================================================

def edge_spline(layer, j, i, xs):
    """Evaluate the univariate edge function phi_{j,i} of a KANLinear layer.

    phi_{j,i}(x) = base_weight[j,i] * silu(x) + sum_k w_spline[j,i,k] B_k(x)
    -- exactly the per-edge term KANLinear.forward sums over i.
    """
    with torch.no_grad():
        xt = torch.zeros(len(xs), layer.in_features)
        xt[:, i] = torch.tensor(xs, dtype=torch.float32)
        B = layer.b_splines(xt)[:, i, :]                      # (n, n_basis)
        w = layer.scaled_spline_weight[j, i]                  # (n_basis,)
        base = layer.base_weight[j, i] * layer.base_activation(
            torch.tensor(xs, dtype=torch.float32))
        return (base + B @ w).numpy()


def curvature(xs, ys):
    """1 - R^2 of a straight-line fit: 0 = perfectly linear edge, 1 = no
    linear content at all. The number that decides 'near-linear layer 1'."""
    A = np.column_stack([xs, np.ones_like(xs)])
    c, *_ = np.linalg.lstsq(A, ys, rcond=None)
    res = ys - A @ c
    tot = np.sum((ys - ys.mean()) ** 2)
    return float(np.sum(res ** 2) / max(tot, 1e-15))


def hidden_activations(model, x_data):
    with torch.no_grad():
        return model.ae.encoder.kan1(x_data).numpy()


def top_hidden_nodes(model, x_data, k, n_top=2):
    """Rank hidden nodes j by the actual spread of their layer-2 contribution
    Phi^(2)_{k,j}(a_j) on the data -- not by raw weight magnitude."""
    a = hidden_activations(model, x_data)
    kan2 = model.ae.encoder.kan2
    contrib_std = np.zeros(a.shape[1])
    for j in range(a.shape[1]):
        lo, hi = np.percentile(a[:, j], [1, 99])
        if hi - lo < 1e-9:
            continue
        grid = np.linspace(lo, hi, 64)
        vals = edge_spline(kan2, k, j, grid)
        # spread of the edge output over the observed activation range
        contrib_std[j] = float(np.std(np.interp(a[:, j], grid, vals)))
    order = np.argsort(-contrib_std)[:n_top]
    return [int(j) for j in order], contrib_std


# =====================================================================
# Figure 1: level-set foliation (style of ks_foliation.png)
# =====================================================================

def band_levels(g, n=N_BANDS):
    qs = np.quantile(g, np.linspace(0.08, 0.92, n))
    return qs, (g.max() - g.min()) * BAND_FRAC


def plot_foliation(dec, k, coeffs, disp, z_traj, t_axis, outdir, r2_hog,
                   n_atoms):
    """4-panel foliation figure. The inner g and outer h come from the fit
    sample; panels (a)/(b) are drawn on the contiguous display trajectories."""
    g, f, outer, _ = outer_for(dec, k, coeffs)
    pod = disp['pod']
    g_d = disp['M'] @ coeffs
    z_d = disp['z'][:, k]
    qs, band = band_levels(g_d)
    cols = plt.cm.plasma(np.linspace(0, 0.92, len(qs)))

    fig = plt.figure(figsize=(17, 11.5))

    # (a) attractor in POD space, colored by the observable
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    sc = ax.scatter(pod[:, 0], pod[:, 1], pod[:, 2], c=z_d, s=4, cmap='viridis')
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.10)
    ax.set_xlabel('POD 1'); ax.set_ylabel('POD 2'); ax.set_zlabel('POD 3')
    ax.set_title(f'(a) KS attractor (POD proj) colored by observable '
                 f'$z_{{{k}}}$', fontsize=12)

    # (b) level sets of g foliate the attractor
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(pod[:, 0], pod[:, 1], c='lightgray', s=5)
    for c_, col in zip(qs, cols):
        m = np.abs(g_d - c_) < band
        ax.scatter(pod[m, 0], pod[m, 1], color=col, s=9, label=f'g≈{c_:.2f}')
    ax.set_xlabel('POD 1'); ax.set_ylabel('POD 2')
    ax.set_title(f'(b) Level sets of $g$ (LARS inner, {n_atoms} atoms) '
                 f'foliate the attractor  [R²(h∘g)={r2_hog:.3f}]', fontsize=12)
    ax.legend(fontsize=7, loc='upper right')

    # (c) outer function
    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(g, f, s=5, alpha=0.3, color='steelblue', label='$z_k$ vs $g$')
    gg = np.linspace(g.min(), g.max(), 300)
    ax.plot(gg, eval_h(gg, outer), 'r-', lw=2, label='$h(g)$ recovered')
    ax.set_xlabel('$g(\\mathbf{u})$ (inner)')
    ax.set_ylabel(f'$z_{{{k}}}$')
    ax.set_title('(c) Outer function $h(g)$  '
                 '(near-linear ⇒ observable ≈ linear)', fontsize=12)
    ax.legend(fontsize=9)

    # (d) the observable along a trajectory
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t_axis, z_traj, 'k-', lw=1)
    ax.set_xlabel('time'); ax.set_ylabel(f'$z_{{{k}}}(t)$')
    ax.set_title('(d) Observable is a slow collective coordinate '
                 'along a trajectory', fontsize=12)

    fig.suptitle(f'KS: activation manifold & level-set foliation of the '
                 f'attractor (observable $z_{{{k}}}$, LassoLars degree-2 '
                 f'readout)', fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / f'lars_foliation_z{k}.png', dpi=140,
                bbox_inches='tight')
    fig.savefig(outdir / f'lars_foliation_z{k}.pdf', bbox_inches='tight')
    plt.close(fig)
    return g_d


def plot_foliation_compare(k, g_cd, g_lars, pod, outdir, n_cd, n_lars, corr):
    """Same projection, same numeric band levels, two solvers."""
    qs, band = band_levels(g_lars)     # LARS levels used for BOTH panels
    cols = plt.cm.plasma(np.linspace(0, 0.92, len(qs)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), sharex=True, sharey=True)
    for ax, g, lbl, n_at in [(axes[0], g_cd, 'CD Lasso', n_cd),
                             (axes[1], g_lars, 'LassoLars', n_lars)]:
        ax.scatter(pod[:, 0], pod[:, 1], c='lightgray', s=5)
        for c_, col in zip(qs, cols):
            m = np.abs(g - c_) < band
            ax.scatter(pod[m, 0], pod[m, 1], color=col, s=9,
                       label=f'g≈{c_:.2f}')
        ax.set_xlabel('POD 1')
        ax.set_title(f'{lbl} inner $g$ — {n_at} active monomials', fontsize=12)
    axes[0].set_ylabel('POD 2')
    axes[1].legend(fontsize=7, loc='upper right')
    fig.suptitle(f'Same activation manifold, different atom lists: level sets '
                 f'of $g$ at identical levels, $z_{{{k}}}$   '
                 f'[corr($g_{{CD}}$, $g_{{LARS}}$) = {corr:.5f}]', fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / f'lars_foliation_compare_z{k}.png', dpi=140,
                bbox_inches='tight')
    fig.savefig(outdir / f'lars_foliation_compare_z{k}.pdf', bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# Figure 2: the splines behind the atoms (style of ks_splines.png)
# =====================================================================

def plot_splines(model, dec, k, coeffs, sites, pair_sites, outdir):
    """Layer-1 edge splines at the atom sites for the top-2 hidden nodes,
    plus the layer-2 edge spline that reads those nodes into z_k."""
    x_np = dec.xn
    a = hidden_activations(model, dec.x)
    nodes, contrib = top_hidden_nodes(model, dec.x, k, n_top=2)
    kan1 = model.ae.encoder.kan1
    kan2 = model.ae.encoder.kan2

    ncol = len(sites) + 1
    fig, axes = plt.subplots(len(nodes), ncol, figsize=(3.1 * ncol, 5.4))
    axes = np.atleast_2d(axes)
    curv = {'layer1': [], 'layer2': [], 'by_edge': {}}

    for r, j in enumerate(nodes):
        for c, i in enumerate(sites):
            lo, hi = np.percentile(x_np[:, i], [1, 99])
            xs = np.linspace(lo, hi, 300)
            ys = edge_spline(kan1, j, i, xs)
            cv = curvature(xs, ys)
            curv['layer1'].append(cv)
            curv['by_edge'][f'{j}_{i}'] = cv
            ax = axes[r, c]
            ax.axhline(0, color='lightgray', ls=':', lw=1)
            ax.plot(xs, ys, 'b-', lw=2)
            ax.set_title(f'$\\phi^{{(1)}}_{{{j},{i}}}(u_{{{i}}})$'
                         f'   [curv {cv:.2f}]', fontsize=11)
            ax.tick_params(labelsize=8)

        lo, hi = np.percentile(a[:, j], [1, 99])
        xs = np.linspace(lo, hi, 300)
        ys = edge_spline(kan2, k, j, xs)
        cv = curvature(xs, ys)
        curv['layer2'].append(cv)
        ax = axes[r, ncol - 1]
        ax.axhline(0, color='lightgray', ls=':', lw=1)
        ax.plot(xs, ys, 'r-', lw=2)
        ax.set_title(f'$\\Phi^{{(2)}}_{{{k},{j}}}(u_{{{j}}})$'
                     f'   [curv {cv:.2f}]', fontsize=11)
        ax.tick_params(labelsize=8)

    dom = nodes[0]
    cp = [curv['by_edge'][f'{dom}_{i}'] for i in pair_sites] or [float('nan')]
    co = [curv['by_edge'][f'{dom}_{i}'] for i in sites if i not in pair_sites]
    fig.suptitle(f'KS layer-1 splines feeding $z_{{{k}}}$  '
                 f'(top-2 hidden nodes {nodes} × sites {sites}; '
                 f'right column = layer-2 edge)\n'
                 f'dominant node {dom}: curvature at the pair sites '
                 f'{pair_sites} = {np.mean(cp):.3f} vs {np.mean(co):.3f} at the '
                 f'other sites; its layer-2 edge = {curv["layer2"][0]:.3f}',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / f'lars_splines_z{k}.png', dpi=140, bbox_inches='tight')
    fig.savefig(outdir / f'lars_splines_z{k}.pdf', bbox_inches='tight')
    plt.close(fig)
    return nodes, curv


# =====================================================================
# Figure 3: quadratic coupling, CD vs LARS
# =====================================================================

def plot_atom_pattern(dims, coeff_cd, coeff_lars, powers_arr, outdir):
    fig, axes = plt.subplots(len(dims), 2, figsize=(13, 5.6 * len(dims)))
    axes = np.atleast_2d(axes)
    stats = {}
    for r, k in enumerate(dims):
        Qc = quad_matrix(coeff_cd[k], powers_arr)
        Ql = quad_matrix(coeff_lars[k], powers_arr)
        vmax = max(np.abs(Qc).max(), np.abs(Ql).max(), 1e-12)
        for c, (Q, lbl) in enumerate([(Qc, 'CD Lasso'), (Ql, 'LassoLars')]):
            ax = axes[r, c]
            im = ax.imshow(np.abs(Q), cmap='hot', origin='lower',
                           vmin=0, vmax=vmax, aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.82, label='$|c_{ij}|$')
            nq = int((np.abs(np.triu(Q)) > 0).sum())
            df = near_diag_fraction(Q)
            ax.set_xlabel('grid point $i$'); ax.set_ylabel('grid point $j$')
            ax.set_title(f'$z_{{{k}}}$  {lbl}: {nq} quadratic atoms, '
                         f'near-diag energy {df:.2f}', fontsize=11)
            stats[f'z{k}_{lbl.split()[0]}'] = {
                'n_quadratic_atoms': nq, 'near_diag_energy_frac': df}
    fig.suptitle('Quadratic coupling $|c_{ij}|$: coordinate descent vs exact '
                 'LARS (same $\\alpha$)', fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / 'lars_atom_pattern.png', dpi=150, bbox_inches='tight')
    fig.savefig(outdir / 'lars_atom_pattern.pdf', bbox_inches='tight')
    plt.close(fig)
    return stats


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--corr-dims', type=int, default=128,
                    help='how many dims to fit with BOTH solvers for the '
                         'corr(g_CD, g_LARS) statistic')
    ap.add_argument('--outdir', type=str, default=str(OUTDIR))
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print('=' * 72, flush=True)
    print('KS LARS re-check: geometry (foliation, splines, coupling)', flush=True)
    print('=' * 72, flush=True)

    model, x_data, traj, latent, info = load_model_and_data(prefer_wide=True)
    print(f"Model: {info['which_model']} {info['arch']}", flush=True)
    dec = DualDecomposer(model, x_data, alpha=LASSO_ALPHA)
    powers_arr = dec.pow_arr
    n_dims = min(args.corr_dims, latent)

    # ---- inner fits with both solvers -------------------------------
    # The figure dims always need both solvers; --corr-dims controls how many
    # dims get the (slow) CD fit purely for the correlation statistic. Results
    # are cached per dim so a second pass only fits what is missing.
    figure_dims = sorted(set(FOLIATION_DIMS) | set(HEATMAP_DIMS))
    needed = sorted(set(figure_dims) | set(range(n_dims)))

    n_terms = dec.M.shape[1]
    C_cd = np.zeros((latent, n_terms))
    C_lars = np.zeros((latent, n_terms))
    have_cd = np.zeros(latent, bool)
    have_lars = np.zeros(latent, bool)

    cache = outdir / CACHE
    if cache.exists() and not args.no_cache:
        blob = np.load(cache)
        C_cd, C_lars = blob['C_cd'], blob['C_lars']
        have_cd, have_lars = blob['have_cd'], blob['have_lars']
        print(f"  loaded cached fits from {cache.name} "
              f"({int(have_cd.sum())} CD, {int(have_lars.sum())} LARS)",
              flush=True)

    todo_lars = [k for k in needed if not have_lars[k]]
    if todo_lars:
        print(f"\n--- LassoLars, {len(todo_lars)} dims ---", flush=True)
        for k in todo_lars:
            C_lars[k] = dec.fit_lars(k)['coeffs']
            have_lars[k] = True
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    todo_cd = [k for k in needed if not have_cd[k]]
    if todo_cd:
        print(f"--- CD Lasso, {len(todo_cd)} dims on {args.jobs} workers "
              f"(this is the slow part) ---", flush=True)
        targets = dec.cd_targets()
        res = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(_cd_fit_one)(dec.Ms, targets[k], LASSO_ALPHA, CD_MAX_ITER)
            for k in todo_cd)
        for k, (coef, icept, *_rest) in zip(todo_cd, res):
            C_cd[k] = dec.finish_cd(k, coef, icept)['coeffs']
            have_cd[k] = True

    if todo_lars or todo_cd:
        np.savez_compressed(cache, C_cd=C_cd, C_lars=C_lars,
                            have_cd=have_cd, have_lars=have_lars)
        print(f"  cached to {cache.name} ({time.time() - t0:.0f}s)", flush=True)

    # ---- corr(g_CD, g_LARS) -----------------------------------------
    both = np.where(have_cd & have_lars)[0]
    G_cd = dec.M @ C_cd[both].T
    G_lars = dec.M @ C_lars[both].T
    corr_by_dim = {}
    for col, k in enumerate(both):
        corr_by_dim[int(k)] = float(np.corrcoef(G_cd[:, col],
                                                G_lars[:, col])[0, 1])
    corrs = np.array([corr_by_dim[int(k)] for k in both])
    print('\n--- corr(g_CD, g_LARS) across dims ---', flush=True)
    print(f"  median {np.median(corrs):.6f}  min {corrs.min():.6f}  "
          f"max {corrs.max():.6f}  (n={len(corrs)} dims)", flush=True)
    print(f"  dims with corr < 0.999: {int((corrs < 0.999).sum())}", flush=True)

    # ---- shared geometry --------------------------------------------
    # Display set: contiguous validation trajectories, so the POD projection
    # shows the attractor's filaments rather than a diffuse cloud.
    X_disp = torch.cat([traj[i] for i in DISPLAY_TRAJ], dim=0).contiguous()
    xd = X_disp.numpy().astype(float)
    Ud = xd - xd.mean(0)
    _, _, Vtd = np.linalg.svd(Ud, full_matrices=False)
    M_disp, _, _ = build_polynomial_dict_nd(
        xd, [f'u_{j}' for j in range(N_GRID)], max_degree=2)
    with torch.no_grad():
        z_disp = model.ae.encoder(X_disp).numpy()
    disp = {'pod': Ud @ Vtd[:3].T, 'M': M_disp, 'z': z_disp}
    print(f"  display set: {xd.shape[0]} contiguous snapshots from "
          f"validation trajectories {DISPLAY_TRAJ}", flush=True)

    # a validation trajectory for panel (d)
    val0 = traj[DISPLAY_TRAJ[0]]
    with torch.no_grad():
        z_val = model.ae.encoder(val0).numpy()
    t_axis = np.arange(val0.shape[0]) * 0.25

    summary = {'corr_g_cd_lars': {
        'median': float(np.median(corrs)), 'min': float(corrs.min()),
        'max': float(corrs.max()), 'n_dims': int(len(corrs)),
        'n_below_0.999': int((corrs < 0.999).sum()),
        'argmin_dim': int(both[int(np.argmin(corrs))])}, 'dims': {}}

    # ---- per-dim figures --------------------------------------------
    for k in FOLIATION_DIMS:
        cl, cc = C_lars[k], C_cd[k]
        n_lars = int((np.abs(cl) > SPARSITY_THRESHOLD).sum())
        n_cd = int((np.abs(cc) > SPARSITY_THRESHOLD).sum())

        _, _, _, r2_hog = outer_for(dec, k, cl)
        g_lars = plot_foliation(dec, k, cl, disp, z_val[:, k], t_axis, outdir,
                                r2_hog, n_lars)
        g_cd = disp['M'] @ cc
        plot_foliation_compare(k, g_cd, g_lars, disp['pod'], outdir, n_cd,
                               n_lars, corr_by_dim[int(k)])

        # sites: the input indices carrying this dim's strongest atoms
        act = np.where(np.abs(cl) > SPARSITY_THRESHOLD)[0]
        order = act[np.argsort(-np.abs(cl[act]))]
        sites, pair_sites = [], []
        for idx in order:
            p = powers_arr[idx]
            if p.sum() == 2 and len(np.nonzero(p)[0]) == 2:
                a_, b_ = np.nonzero(p)[0]
                sep = min(abs(a_ - b_), N_GRID - abs(a_ - b_))
                if sep == 1 and not pair_sites:
                    pair_sites = [int(a_), int(b_)]
            if p.sum() == 1 and len(sites) < N_SITES_EXTRA:
                sites.append(int(np.argmax(p)))
            if pair_sites and len(sites) >= N_SITES_EXTRA:
                break
        sites = pair_sites + [s for s in sites if s not in pair_sites]

        nodes, curv = plot_splines(model, dec, k, cl, sites, pair_sites, outdir)
        dom = nodes[0]
        curv_pair = [curv['by_edge'][f'{dom}_{i}'] for i in pair_sites]
        curv_other = [curv['by_edge'][f'{dom}_{i}'] for i in sites
                      if i not in pair_sites]
        summary['dims'][f'z{k}'] = {
            'dominant_node': int(dom),
            'layer1_curv_at_pair_sites': [float(c) for c in curv_pair],
            'layer1_curv_at_other_sites': [float(c) for c in curv_other],
            'layer2_curv_dominant_node': float(curv['layer2'][0]),
            'n_atoms_lars': n_lars, 'n_atoms_cd': n_cd,
            'r2_hog_lars': r2_hog, 'corr_g': corr_by_dim[int(k)],
            'neighbour_pair_sites': pair_sites, 'spline_sites': sites,
            'top_hidden_nodes': nodes,
            'layer1_curvature_median': float(np.median(curv['layer1'])),
            'layer1_curvature_max': float(np.max(curv['layer1'])),
            'layer2_curvature': [float(c) for c in curv['layer2']],
        }
        print(f"  z[{k}]: LARS {n_lars} atoms (CD {n_cd}), R²(h∘g)={r2_hog:.4f}, "
              f"corr(g)={corr_by_dim[int(k)]:.5f}, pair sites {pair_sites}, "
              f"nodes {nodes}", flush=True)
        print(f"          dominant node {dom}: layer-1 curvature at pair sites "
              f"{[round(c, 3) for c in curv_pair]} vs other sites "
              f"{[round(c, 3) for c in curv_other]}; layer-2 edge curvature "
              f"{curv['layer2'][0]:.3f}", flush=True)

    # ---- coupling heatmaps ------------------------------------------
    stats = plot_atom_pattern(HEATMAP_DIMS, {k: C_cd[k] for k in HEATMAP_DIMS},
                              {k: C_lars[k] for k in HEATMAP_DIMS},
                              powers_arr, outdir)
    summary['coupling'] = stats
    print('\n--- quadratic coupling (near-diagonal energy fraction) ---',
          flush=True)
    for key, v in stats.items():
        print(f"  {key}: {v['n_quadratic_atoms']} atoms, "
              f"near-diag {v['near_diag_energy_frac']:.3f}", flush=True)

    summary['wall_clock_sec'] = time.time() - t0
    with open(outdir / 'geometry_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  Saved {outdir / 'geometry_summary.json'}", flush=True)
    print(f"  Wall clock: {time.time() - t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
