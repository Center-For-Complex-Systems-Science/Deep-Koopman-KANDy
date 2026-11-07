"""
Publication version of the Lorenz attractor figure (fig:lorenz_attractor).

Redraws fig_lorenz_paper.py in the house style of ks_paper_figures.py and
ikeda_paper_figures.py. That script and its outputs, and the hand-edited
docs/figs/fig_lorenz_paper_fixed2.pdf, are left untouched; this one writes new
files to results/lorenz_stable/paper_figs/. No training.

The analysis is the same: checkpoint, seed, RNG call order, decomposition
settings, z selection, percentiles, band width and rollout. The order matters:
constructing the PyKAN encoder/decoder reseeds the global RNGs, so the data and
samples below depend on load_model() running first.

The hand edits in fixed2 are carried over: no suptitle or panel titles, no
g-expression annotation, levels labeled g = c_i rather than numerically.

Panels: (a) attractor colored by g, (b) true vs predicted rollout over
~2 tau_Lambda, (c) level-set bands g ~ c_i.

Sizing: NeurIPS \\textwidth = 5.5 in; fonts are Times-like (STIX) to match the
body text. Dense clouds are rasterized inside vector PDFs. pdf.fonttype = 42
embeds TrueType, so the PDF contains no Type 3 fonts.

Usage (from anywhere):
    python experiments/lorenz_paper_figures.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tempfile
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from models import PyKANAutoEncoder
from data import simulate_lorenz
from training import set_seed
from manifold_analysis import full_decomposition, eval_polynomial
from lorenz_stable import DeepKoopmanCTStable
from fig_lorenz_paper import rollout, DT, TAU_L, ELEV, AZIM

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / 'results' / 'lorenz_stable' / 'stable_8_32_model.pt'
OUTDIR = ROOT / 'results' / 'lorenz_stable' / 'paper_figs'
FIG_NAME = 'lorenz_koopman_attractor'

# Drawn at the width the KS and Ikeda figures crop to, so that all of them scale
# by the same factor at width=\textwidth and their fonts print at the same size.
FIG_WIDTH = 5.0
FIG_HEIGHT = 1.7
N_TAU = 2                # rollout length in Lyapunov times

# Semantic colors shared with ks_paper_figures.py / ikeda_paper_figures.py
BLUE = '#0F4D92'
RED = '#B64342'
ATTRACTOR_GREY = '#DADADA'
BAND_COLORS = ['#D55E00', '#E69F00', '#009E73', '#0072B2', '#CC79A7']


def apply_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'legend.fontsize': 7,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'grid.color': '#E2E2E2',
        'grid.linewidth': 0.3,
        'lines.linewidth': 1.2,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 600,
        'figure.dpi': 150,
    })


def load_model():
    """Build in a temp cwd: the PyKAN constructor writes to ./model."""
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            ae = PyKANAutoEncoder(3, cfg['mid'], cfg['latent'], grid=5, k=3)
        finally:
            os.chdir(cwd)
    model = DeepKoopmanCTStable(ae, cfg['latent'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def compute():
    """Same steps, order and seeds as fig_lorenz_paper.py."""
    set_seed(42)
    model = load_model()

    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=300)
    val_traj = traj[48:]

    flat = traj.reshape(-1, 3)
    idx = torch.randperm(flat.shape[0])[:5000]
    x_sample = flat[idx]

    best_ti, best_score, best_dr = -1, -999, None
    for ti in range(32):
        dr = full_decomposition(model, x_sample, ti, target_type='latent',
                                max_degree=3, lasso_alpha=1e-5, h_degree=4)
        cm = dict(zip(dr['inner']['active_names'], dr['inner']['active_coeffs']))
        has_cross = any(abs(cm.get(t, 0)) > 0.01 for t in ['xy', 'xz', 'yz'])
        score = dr['recon_r2'] + (0.05 if has_cross else 0)
        if score > best_score:
            best_ti, best_score, best_dr = ti, score, dr

    all_pts = val_traj.reshape(-1, 3).numpy()
    g_all = eval_polynomial(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
                            best_dr['inner']['coeffs'],
                            best_dr['inner']['powers'])

    steps = int(N_TAU * TAU_L / DT)
    true_traj = val_traj[0, :steps + 1].numpy()
    pred_traj = rollout(model, val_traj[0, 0], steps)

    sub = np.random.choice(len(all_pts), min(50000, len(all_pts)),
                           replace=False)

    # Printed so a run can be checked against the published figure.
    pairs = sorted(zip(best_dr['inner']['active_names'],
                       best_dr['inner']['active_coeffs']),
                   key=lambda p: -abs(p[1]))
    terms = [f'{c:+.2f} {n}' for n, c in pairs if n != '1'][:3]
    levels = np.linspace(*np.percentile(g_all, [12, 88]), 5)
    print(f"Best: z[{best_ti}], R2(hog) = {best_dr['recon_r2']:.4f}", flush=True)
    print(f"  g ~ {' '.join(terms)} + ...", flush=True)
    print(f"  g 1-99%: {np.percentile(g_all, 1):.3f} to "
          f"{np.percentile(g_all, 99):.3f}", flush=True)
    print(f"  levels: {np.round(levels, 2)}", flush=True)
    print(f"  rollout: {steps} steps; end true {true_traj[-1].round(2)}, "
          f"pred {pred_traj[-1].round(2)}", flush=True)

    return dict(inner=best_dr['inner'], val_traj=val_traj, all_pts=all_pts,
                g_all=g_all, true_traj=true_traj, pred_traj=pred_traj, sub=sub)


def style_ax(ax):
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(-22, 22)
    ax.set_ylim(-28, 28)
    ax.set_zlim(2, 50)
    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([-20, 0, 20])
    ax.set_zticks([10, 30, 50])
    ax.set_xlabel('$x$', labelpad=-9)
    ax.set_ylabel('$y$', labelpad=-9)
    ax.set_zlabel('$z$', labelpad=-12)
    ax.tick_params(axis='both', pad=-3.5, labelsize=6.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('#CFCFCF')
        axis.pane.set_linewidth(0.3)
        axis.line.set_linewidth(0.5)
    ax.set_box_aspect(None, zoom=1.0)
    ax.patch.set_alpha(0)                # neighbors' labels stay visible


def colored_line_3d(ax, pts, c, cmap, norm, lw, alpha):
    segments = np.concatenate([pts[:-1, np.newaxis, :],
                               pts[1:, np.newaxis, :]], axis=1)
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, alpha=alpha,
                          rasterized=True)
    lc.set_array(c[:-1])
    lc.set_linewidth(lw)
    ax.add_collection3d(lc)


def plot_observable(ax, d, g_norm, cmap):
    for pts in d['val_traj'][:8].numpy():
        g_v = eval_polynomial(pts[:, 0], pts[:, 1], pts[:, 2],
                              d['inner']['coeffs'], d['inner']['powers'])
        colored_line_3d(ax, pts, g_v, cmap, g_norm, lw=0.12, alpha=0.6)


def plot_rollout(ax, d):
    pts, sub = d['all_pts'], d['sub']
    true, pred = d['true_traj'], d['pred_traj']
    ax.computed_zorder = False           # keep markers above the curves
    ax.scatter(pts[sub, 0], pts[sub, 1], pts[sub, 2], c=ATTRACTOR_GREY,
               s=0.03, alpha=0.45, rasterized=True, edgecolors='none', zorder=1)
    ax.plot(*true.T, color=BLUE, lw=0.8, zorder=3)
    ax.plot(*pred.T, color=RED, lw=0.8, ls=(0, (3, 1.5)), zorder=4)
    ax.scatter(*true[0], c='k', s=7, marker='o', edgecolors='white',
               linewidths=0.3, depthshade=False, zorder=5)
    ax.scatter(*true[-1], c=BLUE, s=6, marker='s', edgecolors='k',
               linewidths=0.3, depthshade=False, zorder=5)
    ax.scatter(*pred[-1], c=RED, s=6, marker='s', edgecolors='k',
               linewidths=0.3, depthshade=False, zorder=5)


def plot_level_sets(ax, d):
    pts, sub, g_all = d['all_pts'], d['sub'], d['g_all']
    ax.scatter(pts[sub, 0], pts[sub, 1], pts[sub, 2], c=ATTRACTOR_GREY,
               s=0.03, alpha=0.45, rasterized=True, edgecolors='none')
    g_lo, g_hi = np.percentile(g_all, 12), np.percentile(g_all, 88)
    band_w = (g_hi - g_lo) * 0.028
    for lev, col in zip(np.linspace(g_lo, g_hi, 5), BAND_COLORS):
        mask = np.abs(g_all - lev) < band_w
        if mask.sum() < 30:
            continue
        ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2], c=col, s=0.2,
                   alpha=0.75, rasterized=True, edgecolors='none')


def make_figure(d):
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    boxes = [(0.065, 0.0, 0.30, 0.9), (0.375, 0.0, 0.30, 0.9),
             (0.685, 0.0, 0.30, 0.9)]
    ax_a, ax_b, ax_c = [fig.add_axes(b, projection='3d') for b in boxes]
    for ax in (ax_a, ax_b, ax_c):
        style_ax(ax)

    g_norm = Normalize(*np.percentile(d['g_all'], [1, 99]))
    cmap = plt.cm.viridis
    plot_observable(ax_a, d, g_norm, cmap)
    plot_rollout(ax_b, d)
    plot_level_sets(ax_c, d)

    cax = fig.add_axes([0.05, 0.2, 0.012, 0.56])
    cb = fig.colorbar(ScalarMappable(norm=g_norm, cmap=cmap), cax=cax)
    cb.outline.set_linewidth(0.5)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.ax.tick_params(width=0.5, length=2, labelsize=6.5)
    cb.locator = MaxNLocator(4)
    cb.update_ticks()
    cb.set_label('$g(x,y,z)$', fontsize=7.5, labelpad=1)

    for letter, x in zip('abc', (0.0, 0.375, 0.685)):
        fig.text(x, 0.985, f'({letter})', fontsize=9, fontweight='bold',
                 va='top', ha='left')

    fig.legend(handles=[Line2D([], [], color=BLUE, lw=0.9, label='True'),
                        Line2D([], [], color=RED, lw=0.9, ls=(0, (3, 1.5)),
                               label='Predicted')],
               loc='upper left', bbox_to_anchor=(0.42, 1.0), ncol=2,
               handlelength=2.0, handletextpad=0.4, columnspacing=1.0,
               borderaxespad=0)
    fig.legend(handles=[Line2D([], [], ls='', marker='s', ms=3.5, color=col,
                               label=f'$c_{i}$')
                        for i, col in enumerate(BAND_COLORS, start=1)],
               loc='upper left', bbox_to_anchor=(0.775, 1.0),
               ncol=5, handlelength=0.7, handletextpad=0.3, columnspacing=0.7,
               borderaxespad=0)
    fig.text(0.735, 0.975, '$g =$', fontsize=7, va='top', ha='left')

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = OUTDIR / f'{FIG_NAME}.{ext}'
        fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
        print(f"  Saved {path.relative_to(ROOT)}", flush=True)
    plt.close(fig)


def main():
    apply_style()
    make_figure(compute())


if __name__ == '__main__':
    main()
