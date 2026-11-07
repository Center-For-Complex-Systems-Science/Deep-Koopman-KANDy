"""
Publication figure: Lorenz attractor with Koopman rollout and level sets.

Three-panel figure:
    (a) True attractor colored by learned Koopman observable g(x,y,z)
    (b) Rollout comparison: true vs predicted trajectory in 3D
    (c) True attractor with level-set bands highlighted
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path

from models import PyKANAutoEncoder
from data import simulate_lorenz
from training import set_seed
from manifold_analysis import full_decomposition, eval_polynomial

from lorenz_stable import DeepKoopmanCTStable

OUTDIR = Path('results/lorenz_stable')
DT = 0.01
LAMBDA_MAX = 0.9056
TAU_L = 1.0 / LAMBDA_MAX

# ── Classic Lorenz viewing angle ──
ELEV, AZIM = 22, -55


def load_model():
    ckpt = torch.load(OUTDIR / 'stable_8_32_model.pt',
                      map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    ae = PyKANAutoEncoder(3, cfg['mid'], cfg['latent'], grid=5, k=3)
    model = DeepKoopmanCTStable(ae, cfg['latent'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def rollout(model, x0, n_steps):
    xs = [x0]
    x = x0
    with torch.no_grad():
        for _ in range(n_steps):
            out = model.forward_one_step(x.unsqueeze(0), DT)
            x = out['x_tp1_pred'].squeeze(0)
            xs.append(x)
    return torch.stack(xs).numpy()


def colored_line_3d(ax, pts, c, cmap, norm, lw=0.6, alpha=0.9):
    """Plot a 3D line colored by scalar array c."""
    segments = np.concatenate([pts[:-1, np.newaxis, :],
                               pts[1:, np.newaxis, :]], axis=1)
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, alpha=alpha)
    lc.set_array(c[:-1])
    lc.set_linewidth(lw)
    ax.add_collection3d(lc)


def style_ax(ax):
    """Publication-quality 3D axis."""
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel('$x$', fontsize=10, labelpad=1)
    ax.set_ylabel('$y$', fontsize=10, labelpad=1)
    ax.set_zlabel('$z$', fontsize=10, labelpad=1)
    ax.tick_params(labelsize=6, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.88, 0.88, 0.88, 0.4))
        axis.line.set_linewidth(0.3)
    ax.grid(True, alpha=0.12, lw=0.25)
    # Consistent limits across panels
    ax.set_xlim(-22, 22)
    ax.set_ylim(-28, 28)
    ax.set_zlim(2, 50)


if __name__ == '__main__':
    set_seed(42)

    print("Loading model...")
    model, cfg = load_model()

    print("Generating data...")
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=300)
    val_traj = traj[48:]

    # ── Find best decomposition (prefer cross-terms) ──
    flat = traj.reshape(-1, 3)
    idx = torch.randperm(flat.shape[0])[:5000]
    x_sample = flat[idx]

    print("Running decompositions...")
    best_ti, best_score = -1, -999
    best_dr = None
    for ti in range(32):
        dr = full_decomposition(model, x_sample, ti, target_type='latent',
                                max_degree=3, lasso_alpha=1e-5, h_degree=4)
        cm = dict(zip(dr['inner']['active_names'], dr['inner']['active_coeffs']))
        has_cross = any(abs(cm.get(t, 0)) > 0.01 for t in ['xy', 'xz', 'yz'])
        score = dr['recon_r2'] + (0.05 if has_cross else 0)
        if score > best_score:
            best_score = score
            best_ti = ti
            best_dr = dr
    print(f"  Best: z[{best_ti}], R²(h∘g) = {best_dr['recon_r2']:.4f}")

    # ── Dense attractor data (all val trajectories) ──
    all_pts = val_traj.reshape(-1, 3).numpy()
    g_all = eval_polynomial(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
                            best_dr['inner']['coeffs'], best_dr['inner']['powers'])

    # ── Rollout (use 2 Lyapunov times — still accurate) ──
    n_tau = 2
    steps = int(n_tau * TAU_L / DT)
    true_traj = val_traj[0, :steps + 1].numpy()
    pred_traj = rollout(model, val_traj[0, 0], steps)

    # ── Build g expression for annotation ──
    pairs = sorted(zip(best_dr['inner']['active_names'],
                       best_dr['inner']['active_coeffs']),
                   key=lambda x: -abs(x[1]))
    expr_parts = []
    for name, coeff in pairs:
        if name == '1':
            continue
        if len(expr_parts) == 0:
            expr_parts.append(f'{coeff:+.2f}\\,{name}')
        else:
            expr_parts.append(f'{coeff:+.2f}\\,{name}')
        if len(expr_parts) >= 3:
            break
    g_expr = ' '.join(expr_parts) + ' + \\cdots'

    # ═══════════════════════════════════════════════════════════════════
    #  Figure
    # ═══════════════════════════════════════════════════════════════════
    print("Rendering figure...")

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.4,
        'figure.dpi': 150,
    })

    fig = plt.figure(figsize=(16, 5.2))

    g_vmin, g_vmax = np.percentile(g_all, 1), np.percentile(g_all, 99)
    g_norm = Normalize(vmin=g_vmin, vmax=g_vmax)
    cmap_g = plt.cm.coolwarm

    # ── Subsample and depth-sort for all attractor panels ──
    n_show = min(50000, len(all_pts))
    sub = np.random.choice(len(all_pts), n_show, replace=False)

    # ================================================================
    # (a) Attractor colored by Koopman observable g
    # ================================================================
    ax1 = fig.add_subplot(131, projection='3d')
    style_ax(ax1)

    # Plot as a trajectory (line) for smoothness — use first 5 val trajs
    for ti_v in range(min(8, val_traj.shape[0])):
        pts_v = val_traj[ti_v].numpy()
        g_v = eval_polynomial(pts_v[:, 0], pts_v[:, 1], pts_v[:, 2],
                              best_dr['inner']['coeffs'], best_dr['inner']['powers'])
        colored_line_3d(ax1, pts_v, g_v, cmap_g, g_norm, lw=0.25, alpha=0.6)

    cbar = fig.colorbar(ScalarMappable(norm=g_norm, cmap=cmap_g), ax=ax1,
                        shrink=0.55, aspect=18, pad=0.06, location='left')
    cbar.set_label('$g(x,y,z)$', fontsize=9, rotation=90)
    cbar.ax.tick_params(labelsize=6)

    ax1.set_title(r'$\mathbf{(a)}$  Koopman observable',
                  fontsize=10, pad=6, loc='left')
    # Annotate expression below title
    ax1.text2D(0.50, 0.93,
               f'$z_{{{best_ti}}}$:  $g \\approx {g_expr}$',
               transform=ax1.transAxes, fontsize=7.5, ha='center',
               fontstyle='italic', color='#444444')

    # ================================================================
    # (b) Rollout: true vs predicted
    # ================================================================
    ax2 = fig.add_subplot(132, projection='3d')
    style_ax(ax2)

    # Faint attractor background
    ax2.scatter(all_pts[sub, 0], all_pts[sub, 1], all_pts[sub, 2],
                c='#d5d5d5', s=0.08, alpha=0.15, rasterized=True,
                edgecolors='none')

    # True trajectory — solid blue
    ax2.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
             color='#2166ac', lw=1.0, alpha=0.9, label='True', zorder=5)

    # Predicted — dashed red
    ax2.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
             color='#b2182b', lw=1.0, alpha=0.9, ls='--',
             label='Predicted', zorder=6)

    # Start/end markers
    ax2.scatter(*true_traj[0], c='black', s=35, zorder=10, marker='o',
                edgecolors='white', linewidths=0.6)
    ax2.scatter(*true_traj[-1], c='#2166ac', s=25, zorder=10, marker='s',
                edgecolors='black', linewidths=0.4)
    ax2.scatter(*pred_traj[-1], c='#b2182b', s=25, zorder=10, marker='s',
                edgecolors='black', linewidths=0.4)

    ax2.legend(fontsize=8, loc='upper right', framealpha=0.85,
               edgecolor='none', handlelength=1.8, borderpad=0.4)
    ax2.set_title(r'$\mathbf{(b)}$  Koopman rollout'
                  f' ({n_tau} $\\tau_\\Lambda$)',
                  fontsize=10, pad=6, loc='left')

    # ================================================================
    # (c) Level-set bands
    # ================================================================
    ax3 = fig.add_subplot(133, projection='3d')
    style_ax(ax3)

    # Background attractor
    ax3.scatter(all_pts[sub, 0], all_pts[sub, 1], all_pts[sub, 2],
                c='#e2e2e2', s=0.08, alpha=0.2, rasterized=True,
                edgecolors='none')

    # 5 level-set bands with curated colors
    g_lo = np.percentile(g_all, 12)
    g_hi = np.percentile(g_all, 88)
    levels = np.linspace(g_lo, g_hi, 5)
    band_w = (g_hi - g_lo) * 0.028
    colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']

    for lev, col in zip(levels, colors):
        mask = np.abs(g_all - lev) < band_w
        if mask.sum() < 30:
            continue
        ax3.scatter(all_pts[mask, 0], all_pts[mask, 1], all_pts[mask, 2],
                    c=col, s=1.8, alpha=0.75, rasterized=True,
                    edgecolors='none', label=f'$g = {lev:.1f}$')

    ax3.legend(fontsize=7, loc='upper right', framealpha=0.85,
               edgecolor='none', handlelength=1.0, markerscale=4,
               borderpad=0.4, labelspacing=0.3)
    ax3.set_title(r'$\mathbf{(c)}$  Level sets of $g$',
                  fontsize=10, pad=6, loc='left')

    # ── Layout ──
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.88,
                        wspace=0.08)
    fig.suptitle('Lorenz System:  Stable Koopman with KAN Autoencoder',
                 fontsize=13, fontweight='bold', y=0.96)

    fig.savefig(OUTDIR / 'fig_lorenz_paper.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTDIR / 'fig_lorenz_paper.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUTDIR}/fig_lorenz_paper.{{png,pdf}}")
