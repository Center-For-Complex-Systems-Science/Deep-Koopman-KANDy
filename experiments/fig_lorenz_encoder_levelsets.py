"""
Compare level sets of g(x,y,z) vs the actual encoder output f(x,y,z).

Two-panel figure for each analyzed latent dim:
    (a) Level sets of g  (polynomial inner function)
    (b) Level sets of f  (encoder output, i.e. the true learned observable)

This lets you see whether h is genuinely nonlinear — if the bands
shift position or spacing between panels, h is doing real work.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from models import PyKANAutoEncoder
from data import simulate_lorenz
from training import set_seed
from manifold_analysis import full_decomposition, eval_polynomial

from lorenz_stable import DeepKoopmanCTStable

OUTDIR = Path('results/lorenz_stable')
DT = 0.01

ELEV, AZIM = 22, -55
COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']


def load_model():
    ckpt = torch.load(OUTDIR / 'stable_8_32_model.pt',
                      map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    ae = PyKANAutoEncoder(3, cfg['mid'], cfg['latent'], grid=5, k=3)
    model = DeepKoopmanCTStable(ae, cfg['latent'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def style_ax(ax):
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
    ax.set_xlim(-22, 22)
    ax.set_ylim(-28, 28)
    ax.set_zlim(2, 50)


def plot_level_bands(ax, all_pts, vals, sub, title, label_prefix):
    """Draw background attractor + 5 level-set bands."""
    ax.scatter(all_pts[sub, 0], all_pts[sub, 1], all_pts[sub, 2],
               c='#e2e2e2', s=0.08, alpha=0.2, rasterized=True,
               edgecolors='none')

    lo = np.percentile(vals, 12)
    hi = np.percentile(vals, 88)
    levels = np.linspace(lo, hi, 5)
    band_w = (hi - lo) * 0.028

    for lev, col in zip(levels, COLORS):
        mask = np.abs(vals - lev) < band_w
        if mask.sum() < 30:
            continue
        ax.scatter(all_pts[mask, 0], all_pts[mask, 1], all_pts[mask, 2],
                   c=col, s=1.8, alpha=0.75, rasterized=True,
                   edgecolors='none',
                   label=f'${label_prefix} = {lev:.1f}$')

    ax.legend(fontsize=7, loc='upper right', framealpha=0.85,
              edgecolor='none', handlelength=1.0, markerscale=4,
              borderpad=0.4, labelspacing=0.3)
    style_ax(ax)
    ax.set_title(title, fontsize=10, pad=6, loc='left')


if __name__ == '__main__':
    set_seed(42)

    print("Loading model...")
    model, cfg = load_model()

    print("Generating data...")
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=300)
    val_traj = traj[48:]
    all_pts = val_traj.reshape(-1, 3).numpy()
    all_pts_t = torch.tensor(all_pts, dtype=torch.float32)

    n_show = min(50000, len(all_pts))
    sub = np.random.choice(len(all_pts), n_show, replace=False)

    # Get encoder outputs for all points
    print("Computing encoder outputs...")
    with torch.no_grad():
        f_all = model.ae.encoder(all_pts_t).numpy()  # (N, latent_dim)

    # Run decomposition on a subsample
    flat = traj.reshape(-1, 3)
    idx = torch.randperm(flat.shape[0])[:5000]
    x_sample = flat[idx]

    # Find top 4 latent dims by decomposition quality
    print("Running decompositions...")
    results = []
    for ti in range(cfg['latent']):
        dr = full_decomposition(model, x_sample, ti, target_type='latent',
                                max_degree=3, lasso_alpha=1e-5, h_degree=4)
        results.append((ti, dr))
        print(f"  z[{ti}]: R²(g)={dr['g_recon_r2']:.4f}, "
              f"R²(h∘g)={dr['recon_r2']:.4f}, "
              f"gap={dr['recon_r2'] - dr['g_recon_r2']:.4f}")

    # Sort by largest h∘g vs g gap (where h matters most)
    results.sort(key=lambda x: -(x[1]['recon_r2'] - x[1]['g_recon_r2']))
    top = results[:4]

    # ── Figure: 4 rows x 2 cols ──
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.4,
        'figure.dpi': 150,
    })

    fig = plt.figure(figsize=(11, 20))

    for row, (ti, dr) in enumerate(top):
        g_all_ti = eval_polynomial(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
                                   dr['inner']['coeffs'], dr['inner']['powers'])
        f_all_ti = f_all[:, ti]

        r2_g = dr['g_recon_r2']
        r2_hg = dr['recon_r2']

        # (left) Level sets of g
        ax_g = fig.add_subplot(4, 2, 2 * row + 1, projection='3d')
        plot_level_bands(ax_g, all_pts, g_all_ti, sub,
                         rf'$\mathbf{{({chr(97 + 2*row)})}}$  $z_{{{ti}}}$: '
                         rf'level sets of $g$   (R$^2$={r2_g:.3f})',
                         'g')

        # (right) Level sets of f (encoder)
        ax_f = fig.add_subplot(4, 2, 2 * row + 2, projection='3d')
        plot_level_bands(ax_f, all_pts, f_all_ti, sub,
                         rf'$\mathbf{{({chr(98 + 2*row)})}}$  $z_{{{ti}}}$: '
                         rf'level sets of $f$   (R$^2$={r2_hg:.3f})',
                         'f')

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.95,
                        wspace=0.05, hspace=0.15)
    fig.suptitle('Level Sets:  $g$ (polynomial)  vs  $f$ (encoder)',
                 fontsize=14, fontweight='bold', y=0.98)

    outpath = OUTDIR / 'fig_encoder_vs_g_levelsets'
    fig.savefig(str(outpath) + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(str(outpath) + '.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {outpath}.{{png,pdf}}")
    print("\nTop 4 dims by h-gap (largest nonlinear h contribution):")
    for ti, dr in top:
        print(f"  z[{ti}]: R²(g)={dr['g_recon_r2']:.4f} -> "
              f"R²(h∘g)={dr['recon_r2']:.4f}  "
              f"(gap={dr['recon_r2'] - dr['g_recon_r2']:.4f})")
