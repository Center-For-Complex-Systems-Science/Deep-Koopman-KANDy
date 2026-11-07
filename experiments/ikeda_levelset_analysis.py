"""
Ikeda [2,4,4] model: manifold_analysis level-set decomposition.

This is Figure 6. in the supplement of the paper.

Runs full_decomposition_nd on all 4 latent dims, then plots:
  - Level-set bands on the 2D attractor (like fig_lorenz_paper panel c)
  - Summary table of inner g, outer h, R² for each dim
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42

from training import set_seed
from ikeda import generate_ikeda, ikeda_attractor_sample
from manifold_analysis import (
    full_decomposition_nd, eval_polynomial_nd, eval_h,
)

# Reuse model definitions from ikeda_minimal
from ikeda_minimal import (
    IkedaMinimalKoopman, LATENT_DIM,
)

OUTDIR = Path('results/ikeda_minimal')
SEED = 0  # best seed


# =====================================================================
# Wrapper: give the Ikeda model an .ae.encoder interface
# =====================================================================

class AEWrapper:
    """Minimal wrapper so compute_network_gradient_nd can find model.ae.encoder."""
    def __init__(self, model):
        self.encoder = model.encoder
        self.decoder = model.decoder


class ModelWrapper:
    """Wrap IkedaMinimalKoopman so full_decomposition_nd works."""
    def __init__(self, model):
        self.ae = AEWrapper(model)
        self._model = model


def load_model(seed=0):
    ckpt = torch.load(OUTDIR / f'model_seed_{seed}.pt',
                      map_location='cpu', weights_only=False)
    model = IkedaMinimalKoopman()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


if __name__ == '__main__':
    set_seed(42)

    print("Loading model...")
    model = load_model(SEED)
    wrapped = ModelWrapper(model)

    print("Generating data...")
    traj = generate_ikeda(n_traj=200, n_steps=2000, burn_in=500, seed=0)
    flat = traj.reshape(-1, 2)
    # Subsample for analysis
    rng = np.random.RandomState(42)
    idx = rng.permutation(flat.shape[0])[:10000]
    x_sample = torch.from_numpy(flat[idx]).float()

    var_names = ['x', 'y']

    # ── Run full_decomposition_nd on all latent dims ──
    print("\nRunning full_decomposition_nd on all 4 latent dims...")
    results = {}
    for ti in range(LATENT_DIM):
        print(f"  z[{ti}]...", end=" ", flush=True)
        dr = full_decomposition_nd(wrapped, x_sample, ti, var_names,
                                   target_type='latent', max_degree=3,
                                   lasso_alpha=1e-3, h_degree=3)
        results[ti] = dr
        print(f"R²(g)={dr['g_recon_r2']:.4f}  R²(h∘g)={dr['recon_r2']:.4f}  "
              f"active: {dr['inner']['active_names']}")

    # ── Summary table ──
    print("\n" + "="*70)
    print(f"{'dim':>4s}  {'R²(g)':>8s}  {'R²(h∘g)':>8s}  {'affine?':>7s}  {'active terms'}")
    print("-"*70)
    for ti in range(LATENT_DIM):
        dr = results[ti]
        aff = "yes" if dr['outer'].get('used_affine', False) else "no"
        terms = ", ".join(dr['inner']['active_names'][:6])
        print(f"z[{ti}]  {dr['g_recon_r2']:8.4f}  {dr['recon_r2']:8.4f}  {aff:>7s}  {terms}")
    print("="*70)

    # ── Dense attractor data for plotting ──
    print("\nGenerating attractor for plotting...")
    att_pts = ikeda_attractor_sample(n_points=100000, u=0.9, burn_in=1000)
    att_torch = torch.from_numpy(att_pts).float()

    # ══════════════════════════════════════════════════════════════════
    #  Figure: 4-panel level-set plot (one per latent dim)
    # ══════════════════════════════════════════════════════════════════
    print("Rendering level-set figure...")

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.4,
        'figure.dpi': 300,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    band_colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']

    for ti, ax in zip(range(LATENT_DIM), axes.flat):
        dr = results[ti]

        # Evaluate g on the dense attractor
        g_att = eval_polynomial_nd(att_pts, dr['inner']['coeffs'],
                                   dr['inner']['powers'])

        # Background attractor in grey
        ax.scatter(att_pts[:, 0], att_pts[:, 1], s=0.05, alpha=0.15,
                   c='#d5d5d5', rasterized=True, edgecolors='none')

        # Level-set bands
        g_lo = np.percentile(g_att, 12)
        g_hi = np.percentile(g_att, 88)
        levels = np.linspace(g_lo, g_hi, 5)
        band_w = (g_hi - g_lo) * 0.035

        for lev, col in zip(levels, band_colors):
            mask = np.abs(g_att - lev) < band_w
            if mask.sum() < 20:
                continue
            ax.scatter(att_pts[mask, 0], att_pts[mask, 1],
                       c=col, s=0.6, alpha=0.7, rasterized=True,
                       edgecolors='none', label=f'$g = {lev:.2f}$')

        # Build expression for title
        pairs = sorted(zip(dr['inner']['active_names'],
                           dr['inner']['active_coeffs']),
                       key=lambda p: -abs(p[1]))
        expr_parts = []
        for name, coeff in pairs:
            if name == '1':
                continue
            expr_parts.append(f'{coeff:+.3f}{name}')
            if len(expr_parts) >= 3:
                break
        g_expr = ' '.join(expr_parts)
        if len(pairs) > 3:
            g_expr += ' + ...'

        aff_tag = " [affine h]" if dr['outer'].get('used_affine', False) else ""
        ax.set_title(f"$z_{{{ti}}}$:  $g \\approx$ {g_expr}\n"
                     f"$R^2(g)$={dr['g_recon_r2']:.3f},  "
                     f"$R^2(h \\circ g)$={dr['recon_r2']:.3f}{aff_tag}",
                     fontsize=9)
        ax.set_xlabel('$x$', fontsize=20)
        ax.set_ylabel('$y$', fontsize=20)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc='upper right', framealpha=0.8,
                  edgecolor='none', markerscale=6, handlelength=1,
                  labelspacing=0.25, borderpad=0.3)

    fig.suptitle('Ikeda [2,4,4]: Level Sets of Inner Function $g$ (manifold_analysis)',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = OUTDIR / 'levelset_manifold_analysis'
    fig.savefig(str(outpath) + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(str(outpath) + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {outpath}.{{png,pdf}}")

    # ══════════════════════════════════════════════════════════════════
    #  Figure 1b: h(g) on attractor | level sets | h'(g) (degree 2)
    # ══════════════════════════════════════════════════════════════════
    print("Rendering unconstrained h(g) + level-set figure...")

    PLOT_DIMS = [0, 1, 2, 3]
    fig1b, axes1b = plt.subplots(len(PLOT_DIMS), 3, figsize=(16, 4.2 * len(PLOT_DIMS)))

    for row, ti in enumerate(PLOT_DIMS):
        dr = results[ti]
        outer = dr['outer']

        # Evaluate g on dense attractor
        g_att = eval_polynomial_nd(att_pts, dr['inner']['coeffs'],
                                   dr['inner']['powers'])
        # Evaluate h(g) on dense attractor
        hg_att = eval_h(g_att, outer)

        # ── Column 0: attractor colored by h(g) ──
        ax0 = axes1b[row, 0]
        vmin, vmax = np.percentile(hg_att, 2), np.percentile(hg_att, 98)
        sc = ax0.scatter(att_pts[:, 0], att_pts[:, 1], s=0.10, alpha=0.8,
                         c=hg_att, cmap='coolwarm', vmin=vmin, vmax=vmax,
                         rasterized=True, edgecolors='none')
        plt.colorbar(sc, ax=ax0, shrink=0.8, label='$h(g)$', pad=0.02)
        ax0.set_xlabel('$x$', fontsize=20)
        ax0.set_ylabel('$y$', fontsize=20)
        ax0.set_title(f"$z_{{{ti}}}$: $h \\circ g$   "
                      f"$R^2 = {dr['recon_r2']:.3f}$", fontsize=20)
        ax0.set_aspect('equal')
        ax0.tick_params(labelsize=10)

        # ── Column 1: level-set bands ──
        ax1 = axes1b[row, 1]
        ax1.scatter(att_pts[:, 0], att_pts[:, 1], s=0.05, alpha=0.12,
                    c='#d5d5d5', rasterized=True, edgecolors='none')

        g_lo = np.percentile(g_att, 12)
        g_hi = np.percentile(g_att, 88)
        levels = np.linspace(g_lo, g_hi, 5)
        band_w = (g_hi - g_lo) * 0.035

        for lev, col in zip(levels, band_colors):
            mask = np.abs(g_att - lev) < band_w
            if mask.sum() < 20:
                continue
            ax1.scatter(att_pts[mask, 0], att_pts[mask, 1],
                        c=col, s=0.6, alpha=0.7, rasterized=True,
                        edgecolors='none', label=f'$g = {lev:.2f}$')

        ax1.set_xlabel('$x$', fontsize=20)
        ax1.set_ylabel('$y$', fontsize=20)
        ax1.set_title(f"$z_{{{ti}}}$: level sets of $g$", fontsize=20)
        ax1.set_aspect('equal')
        ax1.tick_params(labelsize=10)

        ax1.legend(fontsize=12, loc='upper right', framealpha=0.8,
                   edgecolor='none', markerscale=6, handlelength=1,
                   labelspacing=0.25, borderpad=0.3)

        # ── Column 2: h'(g) gradient identity ──
        ax2 = axes1b[row, 2]
        ax2.scatter(outer['g_samples'], outer['h_prime_samples'],
                    s=0.3, alpha=0.15, c='grey', rasterized=True, label="$h'$ data")
        if len(outer['h_bin_centers']) > 0:
            ax2.plot(outer['h_bin_centers'] + outer['g_center'],
                     outer['h_bin_medians'],
                     'ro', ms=3, zorder=5, label="binned median")
        ax2.plot(outer['g_plot'], outer['h_prime_fit'],
                 'b-', lw=1.5, zorder=6, label="$h'$ fit")

        ax2.set_title(f"$z_{{{ti}}}$: $h'(g)$", fontsize=20)
        ax2.set_xlabel('$g$', fontsize=20)
        ax2.set_ylabel("$h'(g)$", fontsize=20)
        ax2.legend(fontsize=14, loc='best')
        ax2.tick_params(labelsize=10)
        ax2.grid(True, alpha=0.2)
        if ti == 3:
            ax2.set_xlim(left=1.4)

    fig1b.tight_layout()

    outpath1b = OUTDIR / 'hg_levelset_combined'
    fig1b.savefig(str(outpath1b) + '.png', dpi=300, bbox_inches='tight')
    fig1b.savefig(str(outpath1b) + '.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig1b)
    print(f"Saved {outpath1b}.{{png,pdf}}")


    print("\nDone.")
