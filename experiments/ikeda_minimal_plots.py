"""
Ikeda [2,4,4] publication plots.

Produces:
  results/ikeda_minimal/dictionary_proof.png       — single best seed (6-panel)
  results/ikeda_minimal/dictionary_proof_all_seeds.png — 4 seeds (4x4 panel)
  results/ikeda_minimal/all_channels.png           — all 4 inner functions for one seed
  results/ikeda_minimal/levelset_with_h.png        — level-set Lasso + h(g) decomposition

Requires: trained models from ikeda_minimal.py (pruned_seed_*.pt)

Run from project root:
    python experiments/ikeda_minimal_plots.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Lasso
from numpy.linalg import lstsq

from ikeda import generate_ikeda
from ikeda_minimal import (IkedaMinimalKoopman, build_poly_dict_2d,
                            LASSO_ALPHA, LASSO_MAX_DEGREE)
from manifold_analysis_foliation import extract_inner_functions
from manifold_analysis import (
    recover_inner_function_nd, eval_polynomial_nd, grad_polynomial_nd,
    recover_outer_function_from_raw, eval_h
)

OUTDIR = Path('results/ikeda_minimal')
G_MAX = 6.0  # clip x-axis to attractor range


def load_data(n_sample=25000):
    traj = generate_ikeda(n_traj=200, n_steps=2000, burn_in=500, seed=42)
    pts = traj.reshape(-1, 2)
    np.random.seed(0)
    idx = np.random.choice(len(pts), n_sample, replace=False)
    return pts[idx]


def analyze_seed(model, pts_sub, channel, names, M_dict, M_scaled, col_mean, col_std):
    """Run Lasso + gradient identity on one model/channel. Returns all plot data."""
    x_np, y_np = pts_sub[:, 0], pts_sub[:, 1]
    r2 = x_np**2 + y_np**2

    x_data = torch.from_numpy(pts_sub)
    u_vals, per_edge = extract_inner_functions(model.encoder, x_data)

    # Lasso on u[channel]
    target = u_vals[:, channel]
    t_mean, t_std = target.mean(), max(target.std(), 1e-12)
    t_scaled = (target - t_mean) / t_std
    lasso = Lasso(alpha=LASSO_ALPHA, max_iter=50000, fit_intercept=True)
    lasso.fit(M_scaled, t_scaled)
    coeffs = t_std * lasso.coef_ / col_std
    intercept = t_mean + t_std * (lasso.intercept_ - np.sum(lasso.coef_ * col_mean / col_std))
    mask = np.abs(coeffs) > 1e-4
    active = [(names[i], float(coeffs[i])) for i in np.where(mask)[0]]
    active.sort(key=lambda x: -abs(x[1]))
    x2_c = coeffs[names.index('x^2')]
    y2_c = coeffs[names.index('y^2')]
    ratio = min(abs(x2_c), abs(y2_c)) / max(abs(x2_c), abs(y2_c), 1e-12)

    # Gradient identity
    x_t = torch.from_numpy(pts_sub).float().requires_grad_(True)
    u_all = model.encoder.kan1(x_t)
    u_ch = u_all[:, channel]
    grad_f = torch.autograd.grad(u_ch.sum(), x_t, create_graph=False)[0].detach().numpy()
    grad_g = np.stack([2 * x_np, 2 * y_np], axis=-1)
    grad_g_norm_sq = (grad_g**2).sum(-1)
    h_prime = (grad_f * grad_g).sum(-1) / np.maximum(grad_g_norm_sq, 1e-8)
    u_np = u_ch.detach().numpy()
    good = (grad_g_norm_sq > 0.5) & (r2 <= G_MAX)

    g_s = r2[good][np.argsort(r2[good])]
    hp_s = h_prime[good][np.argsort(r2[good])]
    u_s = u_np[good][np.argsort(r2[good])]

    be = np.linspace(g_s.min(), g_s.max(), 81)
    bc, bm_hp, bm_u, bstd_u = [], [], [], []
    for i in range(80):
        m = (g_s >= be[i]) & (g_s < be[i + 1])
        if m.sum() > 15:
            bc.append((be[i] + be[i + 1]) / 2)
            bm_hp.append(np.median(hp_s[m]))
            bm_u.append(u_s[m].mean())
            bstd_u.append(u_s[m].std())
    bc = np.array(bc)
    bm_hp = np.array(bm_hp)
    bm_u = np.array(bm_u)
    bstd_u = np.array(bstd_u)

    # Trig fit
    t_bc = 0.4 - 6.0 / (1.0 + bc)
    A_trig = np.column_stack([np.cos(t_bc), np.sin(t_bc), np.ones(len(bc))])
    sol, _, _, _ = lstsq(A_trig, bm_u, rcond=None)
    pred_trig = A_trig @ sol
    r2_trig = 1 - np.sum((bm_u - pred_trig)**2) / max(np.sum((bm_u - bm_u.mean())**2), 1e-12)
    dt_dg = 6.0 / (1.0 + bc)**2
    hp_theory = (-sol[0] * np.sin(t_bc) + sol[1] * np.cos(t_bc)) * dt_dg
    g_fine = np.linspace(bc.min(), bc.max(), 300)
    t_fine = 0.4 - 6.0 / (1.0 + g_fine)
    h_fine = sol[0] * np.cos(t_fine) + sol[1] * np.sin(t_fine) + sol[2]

    return {
        'u_vals': u_vals, 'per_edge': per_edge,
        'active': active, 'x2_c': x2_c, 'y2_c': y2_c, 'ratio': ratio,
        'r2': r2, 'good': good, 'h_prime': h_prime,
        'bc': bc, 'bm_hp': bm_hp, 'bm_u': bm_u, 'bstd_u': bstd_u,
        'sol': sol, 'r2_trig': r2_trig, 'hp_theory': hp_theory,
        'g_fine': g_fine, 'h_fine': h_fine,
    }


# =====================================================================
# Single-seed 6-panel figure
# =====================================================================

def plot_single_seed(seed, channel, pts_sub, names, M_dict, M_scaled, col_mean, col_std):
    model = IkedaMinimalKoopman()
    ckpt = torch.load(OUTDIR / f'pruned_seed_{seed}.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    r = analyze_seed(model, pts_sub, channel, names, M_dict, M_scaled, col_mean, col_std)
    x_np, y_np = pts_sub[:, 0], pts_sub[:, 1]

    fig = plt.figure(figsize=(18, 12))
    gs_fig = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # (a) Edge splines
    ax = fig.add_subplot(gs_fig[0, 0])
    psi_x = r['per_edge'][:, channel, 0]
    psi_y = r['per_edge'][:, channel, 1]
    sx, sy = np.argsort(x_np), np.argsort(y_np)
    ax.scatter(x_np[sx], psi_x[sx], s=0.5, alpha=0.1, c='steelblue', rasterized=True, label='psi(x)')
    ax.scatter(y_np[sy], psi_y[sy], s=0.5, alpha=0.1, c='#d62728', rasterized=True, label='psi(y)')
    c2x = np.polyfit(x_np, psi_x, 2)
    c2y = np.polyfit(y_np, psi_y, 2)
    xf = np.linspace(x_np.min(), x_np.max(), 200)
    yf = np.linspace(y_np.min(), y_np.max(), 200)
    ax.plot(xf, np.polyval(c2x, xf), 'b-', lw=2.5, label=f'{c2x[0]:.3f}t^2+{c2x[1]:.3f}t+{c2x[2]:.3f}')
    ax.plot(yf, np.polyval(c2y, yf), 'r-', lw=2.5, label=f'{c2y[0]:.3f}t^2+{c2y[1]:.3f}t+{c2y[2]:.3f}')
    ax.set_xlabel('x or y'); ax.set_ylabel('Edge activation')
    ax.set_title(f'(a) Edge splines for u_{channel}'); ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

    # (b) u on attractor
    ax = fig.add_subplot(gs_fig[0, 1])
    sc = ax.scatter(x_np, y_np, s=0.5, alpha=0.3, c=r['u_vals'][:, channel], cmap='RdBu_r', rasterized=True)
    plt.colorbar(sc, ax=ax, label=f'u_{channel}', shrink=0.8)
    theta = np.linspace(0, 2 * np.pi, 200)
    for rv in [1, 2, 4]:
        ax.plot(np.sqrt(rv) * np.cos(theta), np.sqrt(rv) * np.sin(theta), 'k-', lw=0.8, alpha=0.4)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'(b) u_{channel} on attractor'); ax.set_aspect('equal')

    # (c) u vs r^2
    ax = fig.add_subplot(gs_fig[0, 2])
    ax.scatter(r['r2'], r['u_vals'][:, channel], s=0.5, alpha=0.05, c='gray', rasterized=True)
    ax.plot(r['bc'], r['bm_u'], 'b-', lw=2.5, label='binned mean')
    ax.fill_between(r['bc'], r['bm_u'] - r['bstd_u'], r['bm_u'] + r['bstd_u'], alpha=0.15, color='blue')
    ax.set_xlim(0, G_MAX); ax.set_xlabel('g = x^2 + y^2'); ax.set_ylabel(f'u_{channel}')
    ax.set_title(f'(c) Scatter collapse (ratio={r["ratio"]:.3f})'); ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    # (d) Lasso bar chart
    ax = fig.add_subplot(gs_fig[1, 0])
    bar_names = [n for n, c in r['active'] if n != '1'][:8]
    bar_vals = [c for n, c in r['active'] if n != '1'][:8]
    bar_colors = ['#2ca02c' if n in ('x^2', 'y^2') else '#aaaaaa' for n in bar_names]
    ax.barh(range(len(bar_names)), bar_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(bar_names))); ax.set_yticklabels(bar_names)
    ax.set_title(f'(d) Lasso: x^2={r["x2_c"]:+.3f}, y^2={r["y2_c"]:+.3f}')
    ax.grid(True, alpha=0.15, axis='x'); ax.invert_yaxis()

    # (e) h'(g)
    ax = fig.add_subplot(gs_fig[1, 1])
    good = r['good']
    ax.scatter(r['r2'][good], r['h_prime'][good], s=0.3, alpha=0.02, c='gray', rasterized=True)
    ax.plot(r['bc'], r['bm_hp'], 'b-', lw=2.5, label="h'(g)")
    ax.plot(r['bc'], r['hp_theory'], 'r--', lw=2, label="trig fit h'")
    ax.set_xlim(0, G_MAX); ax.set_xlabel('g = x^2 + y^2'); ax.set_ylabel("h'(g)")
    ax.set_title("(e) Gradient identity"); ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    # (f) h(g)
    ax = fig.add_subplot(gs_fig[1, 2])
    s = r['sol']
    ax.plot(r['bc'], r['bm_u'], 'b-', lw=3, label=f'u_{channel} (learned)')
    ax.plot(r['g_fine'], r['h_fine'], 'r--', lw=2.5, label=f'{s[0]:.2f}cos+{s[1]:.2f}sin+{s[2]:.2f}')
    ax.fill_between(r['bc'], r['bm_u'] - r['bstd_u'], r['bm_u'] + r['bstd_u'], alpha=0.1, color='blue')
    ax.set_xlim(0, G_MAX); ax.set_xlabel('g = x^2 + y^2'); ax.set_ylabel('h(g)')
    ax.set_title(f'(f) Outer function R^2 = {r["r2_trig"]:.3f}'); ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    fig.suptitle(f'Ikeda [2,4,4] seed {seed}: Compositional Dictionary Extraction\n'
                 'Inner: g = x^2 + y^2 (Lasso)    '
                 'Outer: h = a cos(0.4-6/(1+g)) + b sin(0.4-6/(1+g)) + c (gradient identity)',
                 fontsize=13, y=1.02)
    fig.savefig(OUTDIR / 'dictionary_proof.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'dictionary_proof.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved dictionary_proof.png (seed {seed})')


# =====================================================================
# Multi-seed 4x4 figure
# =====================================================================

def plot_all_seeds(seeds, pts_sub, results_json, names, M_dict, M_scaled, col_mean, col_std):
    fig = plt.figure(figsize=(20, 5 * len(seeds)))
    gs_fig = fig.add_gridspec(len(seeds), 4, hspace=0.4, wspace=0.3)
    x_np, y_np = pts_sub[:, 0], pts_sub[:, 1]
    theta = np.linspace(0, 2 * np.pi, 200)

    for row, seed in enumerate(seeds):
        model = IkedaMinimalKoopman()
        ckpt = torch.load(OUTDIR / f'pruned_seed_{seed}.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        sr = results_json['per_seed'][f'seed_{seed}']
        CH = sr['gradient_identity']['channel']
        r = analyze_seed(model, pts_sub, CH, names, M_dict, M_scaled, col_mean, col_std)

        # (col 0) u on attractor
        ax = fig.add_subplot(gs_fig[row, 0])
        sc = ax.scatter(x_np, y_np, s=0.5, alpha=0.3, c=r['u_vals'][:, CH], cmap='RdBu_r', rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        for rv in [1, 2, 4]:
            ax.plot(np.sqrt(rv) * np.cos(theta), np.sqrt(rv) * np.sin(theta), 'k-', lw=0.7, alpha=0.3)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'Seed {seed}: u_{CH} on attractor', fontsize=10); ax.set_aspect('equal')

        # (col 1) Lasso bar chart
        ax = fig.add_subplot(gs_fig[row, 1])
        bar_names = [n for n, c in r['active'] if n != '1'][:8]
        bar_vals = [c for n, c in r['active'] if n != '1'][:8]
        bar_colors = ['#2ca02c' if n in ('x^2', 'y^2') else '#aaaaaa' for n in bar_names]
        ax.barh(range(len(bar_names)), bar_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(bar_names))); ax.set_yticklabels(bar_names, fontsize=9)
        ax.set_title(f'Lasso: x^2={r["x2_c"]:+.3f}, y^2={r["y2_c"]:+.3f}\nratio={r["ratio"]:.3f}', fontsize=10)
        ax.grid(True, alpha=0.15, axis='x'); ax.invert_yaxis()

        # (col 2) h'(g)
        ax = fig.add_subplot(gs_fig[row, 2])
        good = r['good']
        ax.scatter(r['r2'][good], r['h_prime'][good], s=0.3, alpha=0.03, c='gray', rasterized=True)
        ax.plot(r['bc'], r['bm_hp'], 'b-', lw=2.5, label="h'(g)")
        ax.plot(r['bc'], r['hp_theory'], 'r--', lw=2, label="trig fit h'")
        ax.set_xlim(0, G_MAX); ax.set_xlabel('g = x^2 + y^2'); ax.set_ylabel("h'(g)")
        ax.set_title("h'(g) via gradient identity", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

        # (col 3) h(g)
        ax = fig.add_subplot(gs_fig[row, 3])
        s = r['sol']
        ax.plot(r['bc'], r['bm_u'], 'b-', lw=3, label=f'u_{CH} (learned)')
        ax.plot(r['g_fine'], r['h_fine'], 'r--', lw=2.5,
                label=f'{s[0]:.2f}cos + {s[1]:.2f}sin + {s[2]:.2f}')
        ax.fill_between(r['bc'], r['bm_u'] - r['bstd_u'], r['bm_u'] + r['bstd_u'], alpha=0.1, color='blue')
        ax.set_xlim(0, G_MAX); ax.set_xlabel('g = x^2 + y^2'); ax.set_ylabel('h(g)')
        ax.set_title(f'Seed {seed}: R^2 = {r["r2_trig"]:.3f}', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

        del model
        torch.cuda.empty_cache()

    fig.suptitle('Ikeda [2,4,4]: Dictionary Extraction Stability (seeds 0,1,2,4)', fontsize=14, y=1.01)
    fig.savefig(OUTDIR / 'dictionary_proof_all_seeds.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'dictionary_proof_all_seeds.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved dictionary_proof_all_seeds.png')


# =====================================================================
# All channels figure — one row per inner function
# =====================================================================

def plot_all_channels(seed, pts_sub):
    """4 rows (one per u_j) x 5 columns: psi(x), psi(y), attractor, u vs r^2, classification."""
    model = IkedaMinimalKoopman()
    ckpt = torch.load(OUTDIR / f'pruned_seed_{seed}.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    x_data = torch.from_numpy(pts_sub)
    u_vals, per_edge = extract_inner_functions(model.encoder, x_data)
    x_np, y_np = pts_sub[:, 0], pts_sub[:, 1]
    r2 = x_np**2 + y_np**2
    theta_c = np.linspace(0, 2 * np.pi, 200)
    n_ch = u_vals.shape[1]

    fig, axes = plt.subplots(n_ch, 5, figsize=(25, 5 * n_ch))

    for j in range(n_ch):
        psi_x = per_edge[:, j, 0]
        psi_y = per_edge[:, j, 1]
        u_j = u_vals[:, j]

        c2x = np.polyfit(x_np, psi_x, 2)
        c2y = np.polyfit(y_np, psi_y, 2)

        # Linear fit for classification
        A_lin = np.column_stack([x_np, y_np, np.ones(len(x_np))])
        sol_lin, _, _, _ = lstsq(A_lin, u_j, rcond=None)
        r2_lin = 1 - np.sum((u_j - A_lin @ sol_lin)**2) / np.sum((u_j - u_j.mean())**2)

        ratio = min(abs(c2x[0]), abs(c2y[0])) / max(abs(c2x[0]), abs(c2y[0]), 1e-12)
        same_sign = c2x[0] * c2y[0] > 0

        # Col 0: psi(x) — separate subplot
        ax = axes[j, 0]
        sx = np.argsort(x_np)
        ax.scatter(x_np[sx], psi_x[sx], s=0.3, alpha=0.08, c='steelblue', rasterized=True)
        xf = np.linspace(x_np.min(), x_np.max(), 200)
        ax.plot(xf, np.polyval(c2x, xf), 'b-', lw=2.5,
                label=f'{c2x[0]:+.3f}x^2{c2x[1]:+.3f}x{c2x[2]:+.3f}')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel(f'psi_{{j={j}}}(x)', fontsize=10)
        if j == 0:
            ax.set_title('psi(x) edge spline', fontsize=11)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.15)

        # Col 1: psi(y) — separate subplot
        ax = axes[j, 1]
        sy = np.argsort(y_np)
        ax.scatter(y_np[sy], psi_y[sy], s=0.3, alpha=0.08, c='#d62728', rasterized=True)
        yf = np.linspace(y_np.min(), y_np.max(), 200)
        ax.plot(yf, np.polyval(c2y, yf), 'r-', lw=2.5,
                label=f'{c2y[0]:+.3f}y^2{c2y[1]:+.3f}y{c2y[2]:+.3f}')
        ax.set_xlabel('y', fontsize=10)
        ax.set_ylabel(f'psi_{{j={j}}}(y)', fontsize=10)
        if j == 0:
            ax.set_title('psi(y) edge spline', fontsize=11)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.15)

        # Col 2: u_j on attractor
        ax = axes[j, 2]
        sc = ax.scatter(x_np, y_np, s=0.3, alpha=0.3, c=u_j, cmap='RdBu_r', rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        for rv in [1, 2, 4]:
            ax.plot(np.sqrt(rv) * np.cos(theta_c), np.sqrt(rv) * np.sin(theta_c),
                    'k-', lw=0.6, alpha=0.3)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        if j == 0:
            ax.set_title('u_j on attractor', fontsize=11)
        ax.set_aspect('equal')

        # Col 3: u vs r^2 or u vs linear fit
        ax = axes[j, 3]
        if r2_lin > 0.95:
            # Linear channel — plot u vs ax+by
            pred_lin = sol_lin[0] * x_np + sol_lin[1] * y_np + sol_lin[2]
            ax.scatter(pred_lin, u_j, s=0.3, alpha=0.05, c='gray', rasterized=True)
            lims = [min(pred_lin.min(), u_j.min()), max(pred_lin.max(), u_j.max())]
            ax.plot(lims, lims, 'r-', lw=2, label='y = x')
            ax.set_xlabel(f'{sol_lin[0]:+.2f}x + {sol_lin[1]:+.2f}y + {sol_lin[2]:+.2f}')
            ax.set_ylabel(f'u[{j}]')
            ax.set_title(f'u[{j}] linear fit (R^2={r2_lin:.3f})', fontsize=10)
            ax.legend(fontsize=9)
        else:
            # Plot u vs r^2
            ax.scatter(r2, u_j, s=0.3, alpha=0.05, c='gray', rasterized=True)
            sort_r2 = np.argsort(r2)
            r2_s = r2[sort_r2]; u_s = u_j[sort_r2]
            be = np.linspace(r2_s.min(), min(r2_s.max(), G_MAX), 61)
            bc, bm = [], []
            for i in range(len(be) - 1):
                m = (r2_s >= be[i]) & (r2_s < be[i + 1])
                if m.sum() > 15:
                    bc.append((be[i] + be[i + 1]) / 2)
                    bm.append(u_s[m].mean())
            bc, bm = np.array(bc), np.array(bm)
            ax.plot(bc, bm, 'b-', lw=2.5, label='binned mean')

            # Trig fit
            if len(bc) > 3:
                t_bc = 0.4 - 6.0 / (1.0 + bc)
                A_trig = np.column_stack([np.cos(t_bc), np.sin(t_bc), np.ones(len(bc))])
                sol_t, _, _, _ = lstsq(A_trig, bm, rcond=None)
                g_fine = np.linspace(bc.min(), bc.max(), 200)
                t_fine = 0.4 - 6.0 / (1.0 + g_fine)
                h_fine = sol_t[0] * np.cos(t_fine) + sol_t[1] * np.sin(t_fine) + sol_t[2]
                r2_trig = 1 - np.sum((bm - A_trig @ sol_t)**2) / max(np.sum((bm - bm.mean())**2), 1e-12)
                ax.plot(g_fine, h_fine, 'r--', lw=2, label=f'trig R^2={r2_trig:.2f}')

            ax.set_xlim(0, G_MAX)
            ax.set_xlabel('r^2 = x^2+y^2')
            ax.set_ylabel(f'u[{j}]')
            ax.set_title(f'u[{j}] vs r^2 (ratio={ratio:.2f})', fontsize=10)
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

        # Col 4: classification text
        ax = axes[j, 4]
        if r2_lin > 0.95:
            ax.text(0.5, 0.5,
                    f'LINEAR\nu[{j}] = {sol_lin[0]:+.3f}x\n'
                    f'       + {sol_lin[1]:+.3f}y\n'
                    f'       + {sol_lin[2]:+.3f}\n'
                    f'R^2 = {r2_lin:.4f}\n\nDictionary: x, y',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes,
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#d4edda', alpha=0.8))
        elif ratio > 0.7 and same_sign:
            ax.text(0.5, 0.5,
                    f'r^2 CANDIDATE\nx^2 coeff: {c2x[0]:+.4f}\n'
                    f'y^2 coeff: {c2y[0]:+.4f}\n'
                    f'ratio: {ratio:.3f}\n'
                    f'same sign: {same_sign}',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes,
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#cce5ff', alpha=0.8))
        else:
            ax.text(0.5, 0.5,
                    f'MIXED\nratio={ratio:.3f}\nsame sign: {same_sign}\n'
                    f'R^2_lin={r2_lin:.3f}',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes,
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.8))
        ax.set_axis_off()

    fig.suptitle(f'Ikeda [2,4,4] Seed {seed}: All Inner Functions — Dictionary Extraction',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'all_channels.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'all_channels.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved all_channels.png (seed {seed})')


# =====================================================================
# Level-set decomposition figure (manifold_analysis.py pipeline)
# =====================================================================

def plot_levelset_decomposition(seed, pts_sub):
    """Run full_decomposition from manifold_analysis.py on inner functions."""
    model = IkedaMinimalKoopman()
    ckpt = torch.load(OUTDIR / f'pruned_seed_{seed}.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    var_names = ['x', 'y']
    x_np, y_np = pts_sub[:, 0], pts_sub[:, 1]
    n_ch = 4

    # Run decomposition on all channels, find linear and r^2 channels
    all_decomp = []
    for j in range(n_ch):
        x_ag = torch.from_numpy(pts_sub).float().requires_grad_(True)
        u_out = model.encoder.kan1(x_ag)
        f_j = u_out[:, j]
        grad_f = torch.autograd.grad(f_j.sum(), x_ag, create_graph=False)[0].detach().numpy()
        f_vals = f_j.detach().numpy()

        inner = recover_inner_function_nd(pts_sub, var_names, f_vals,
                                           max_degree=3, alpha=1e-3)
        g_vals = eval_polynomial_nd(pts_sub, inner['coeffs'], inner['powers'])
        grad_g = grad_polynomial_nd(pts_sub, inner['coeffs'], inner['powers'])

        dot = np.sum(grad_f * grad_g, axis=1)
        norm_sq = np.sum(grad_g**2, axis=1)
        outer = recover_outer_function_from_raw(dot, norm_sq, g_vals, f_vals, poly_degree=4)
        f_hat = eval_h(g_vals, outer)

        r2_g = inner['r2']
        r2_hog = 1 - np.sum((f_vals - f_hat)**2) / max(np.sum((f_vals - f_vals.mean())**2), 1e-12)

        active = [(inner['names'][i], float(inner['coeffs'][i]))
                  for i in range(len(inner['coeffs'])) if abs(inner['coeffs'][i]) > 1e-4]
        active.sort(key=lambda x: -abs(x[1]))

        x2_c = inner['coeffs'][inner['names'].index('x^2')] if 'x^2' in inner['names'] else 0
        y2_c = inner['coeffs'][inner['names'].index('y^2')] if 'y^2' in inner['names'] else 0
        ratio = min(abs(x2_c), abs(y2_c)) / max(abs(x2_c), abs(y2_c), 1e-12)
        same = float(x2_c) * float(y2_c) > 0

        # Linear fit
        A_lin = np.column_stack([x_np, y_np, np.ones(len(x_np))])
        sol_lin, _, _, _ = lstsq(A_lin, f_vals, rcond=None)
        r2_lin = 1 - np.sum((f_vals - A_lin @ sol_lin)**2) / np.sum((f_vals - f_vals.mean())**2)

        all_decomp.append({
            'j': j, 'r2_g': r2_g, 'r2_hog': r2_hog, 'r2_lin': r2_lin,
            'x2': float(x2_c), 'y2': float(y2_c), 'ratio': ratio, 'same': same,
            'active': active, 'inner': inner, 'outer': outer,
            'g_vals': g_vals, 'f_vals': f_vals, 'f_hat': f_hat,
        })

    lin_ch = max(all_decomp, key=lambda r: r['r2_lin'])
    r2_candidates = [r for r in all_decomp if r['same'] and r['ratio'] > 0.5]
    r2_ch = max(r2_candidates, key=lambda r: r['ratio']) if r2_candidates else max(all_decomp, key=lambda r: r['ratio'])

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for row, (r, label) in enumerate([(lin_ch, 'Linear'), (r2_ch, 'Compositional')]):
        sort_g = np.argsort(r['g_vals'])
        gs = r['g_vals'][sort_g]; fs = r['f_vals'][sort_g]; fhs = r['f_hat'][sort_g]
        be = np.linspace(gs.min(), gs.max(), 81)
        bc, bmf, bmfh = [], [], []
        for i in range(80):
            m = (gs >= be[i]) & (gs < be[i + 1])
            if m.sum() > 15:
                bc.append((be[i] + be[i + 1]) / 2)
                bmf.append(fs[m].mean())
                bmfh.append(fhs[m].mean())
        bc, bmf, bmfh = np.array(bc), np.array(bmf), np.array(bmfh)

        # Col 0: f vs g
        ax = axes[row, 0]
        ax.scatter(r['g_vals'], r['f_vals'], s=0.3, alpha=0.04, c='gray', rasterized=True)
        ax.plot(bc, bmf, 'b-', lw=2.5, label=f'u[{r["j"]}] binned mean')
        ax.plot(bc, bmfh, 'r--', lw=2.5, label='h(g) from gradient identity')
        ax.set_xlabel('g(x,y) [Lasso polynomial]')
        ax.set_ylabel(f'u[{r["j"]}]')
        ax.set_title(f'{label}: u[{r["j"]}] vs g\n'
                     f'R^2(g)={r["r2_g"]:.4f}, R^2(hog)={r["r2_hog"]:.4f}', fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

        # Col 1: h(g)
        ax = axes[row, 1]
        g_fine = np.linspace(r['g_vals'].min(), r['g_vals'].max(), 300)
        h_fine = eval_h(g_fine, r['outer'])
        ax.plot(g_fine, h_fine, 'r-', lw=3, label='h(g) [degree-4 poly]')
        ax.plot(g_fine, g_fine, 'k:', lw=1.5, alpha=0.4, label='h(g) = g (identity)')
        ax.set_xlabel('g'); ax.set_ylabel('h(g)')
        is_affine = abs(r['r2_hog'] - r['r2_g']) < 0.01
        ax.set_title(f'Outer function h(g)\n{"affine" if is_affine else "nonlinear"}', fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

        # Col 2: Lasso dictionary
        ax = axes[row, 2]
        act = [(n, c) for n, c in r['active'] if n != '1'][:8]
        bar_n = [n for n, c in act]
        bar_v = [c for n, c in act]
        bar_c = ['#2ca02c' if n in ('x^2', 'y^2') else '#ff9800' if n in ('x', 'y') else '#aaaaaa'
                 for n in bar_n]
        ax.barh(range(len(bar_n)), bar_v, color=bar_c, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(bar_n)))
        ax.set_yticklabels(bar_n, fontsize=11)
        ax.set_xlabel('Lasso coefficient')
        ax.set_title(f'x^2={r["x2"]:+.3f}, y^2={r["y2"]:+.3f}, ratio={r["ratio"]:.3f}', fontsize=11)
        ax.grid(True, alpha=0.15, axis='x')
        ax.invert_yaxis()

    fig.suptitle(f'Ikeda [2,4,4] Seed {seed}: Level-Set Decomposition u_j = h(g(x,y))\n'
                 'Green = x^2, y^2 (r^2)    Orange = x, y (linear)    '
                 'h is affine: Lasso polynomial captures the full structure',
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'levelset_with_h.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'levelset_with_h.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved levelset_with_h.png (seed {seed})')


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    results_json = json.load(open(OUTDIR / 'results.json'))
    pts_sub = load_data()
    x_np, y_np = pts_sub[:, 0], pts_sub[:, 1]

    M_dict, names = build_poly_dict_2d(x_np, y_np, max_degree=LASSO_MAX_DEGREE)
    col_mean = M_dict.mean(0)
    col_std = M_dict.std(0)
    col_std[col_std < 1e-12] = 1.0
    M_scaled = (M_dict - col_mean) / col_std

    # Single best seed (seed 1, trig R2=0.960, channel u[2])
    print("--- Single-seed figure (seed 1) ---")
    plot_single_seed(1, 2, pts_sub, names, M_dict, M_scaled, col_mean, col_std)

    # All seeds except edge case (seed 3)
    print("\n--- Multi-seed figure (seeds 0,1,2,4) ---")
    plot_all_seeds([0, 1, 2, 4], pts_sub, results_json, names, M_dict, M_scaled, col_mean, col_std)

    # All 4 channels for seed 1
    print("\n--- All channels figure (seed 1) ---")
    plot_all_channels(1, pts_sub)

    # Level-set decomposition for seed 1
    print("\n--- Level-set decomposition figure (seed 1) ---")
    plot_levelset_decomposition(1, pts_sub)

    print("\nDone.")
