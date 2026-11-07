"""
Publication version of the Ikeda level-set figure (Figure 6, supplement).

Redraws hg_levelset_combined from ikeda_levelset_analysis.py in the house style
of ks_paper_figures.py. That script and its outputs are left untouched; this
one writes new files. The analysis is the same: same checkpoint, seeds, sample,
full_decomposition_nd settings, attractor, level-set percentiles and band width.
Only the presentation changes. No training.

Columns: (a) attractor colored by h(g), (b) level sets of g, (c) h'(g) from the
gradient identity. Rows: latent dimensions z_0..z_3.

Sizing: NeurIPS \\textwidth = 5.5 in; fonts are Times-like (STIX) to match the
body text, 6-9 pt at print size. Point clouds are rasterized inside vector PDFs.
pdf.fonttype = 42 embeds TrueType, so the PDF contains no Type 3 fonts.

Usage (from anywhere):
    python experiments/ikeda_paper_figures.py
    python experiments/ikeda_paper_figures.py --copy-to-docs   # also docs/figs/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import shutil
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from training import set_seed
from ikeda import generate_ikeda, ikeda_attractor_sample
from manifold_analysis import (full_decomposition_nd, eval_polynomial_nd,
                               eval_h)
from ikeda_minimal import IkedaMinimalKoopman, LATENT_DIM
from ikeda_levelset_analysis import ModelWrapper

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / 'results' / 'ikeda_minimal' / 'model_seed_0.pt'
OUTDIR = ROOT / 'results' / 'ikeda_minimal' / 'paper_figs'
DOCS_FIGS = ROOT / 'docs' / 'figs'
FIG_NAME = 'ikeda_hg_levelset'

TEXT_WIDTH = 5.5         # NeurIPS \textwidth (in)
FIG_HEIGHT = 6.2

# Semantic colors shared with ks_paper_figures.py (the caption names red and
# blue for the binned medians and the fit).
BLUE = '#0F4D92'
RED = '#B64342'
NEUTRAL = '#9E9E9E'
ATTRACTOR_GREY = '#DADADA'
# Okabe-Ito, in the hue order of the original figure; yellow is skipped because
# it does not read on white.
BAND_COLORS = ['#D55E00', '#E69F00', '#009E73', '#0072B2', '#CC79A7']


def apply_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 1.2,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 600,
        'figure.dpi': 150,
    })


def load_model():
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = IkedaMinimalKoopman()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def compute():
    """Same steps, order and seeds as ikeda_levelset_analysis.py."""
    set_seed(42)
    wrapped = ModelWrapper(load_model())

    traj = generate_ikeda(n_traj=200, n_steps=2000, burn_in=500, seed=0)
    flat = traj.reshape(-1, 2)
    rng = np.random.RandomState(42)
    idx = rng.permutation(flat.shape[0])[:10000]
    x_sample = torch.from_numpy(flat[idx]).float()

    results = {}
    for ti in range(LATENT_DIM):
        results[ti] = full_decomposition_nd(
            wrapped, x_sample, ti, ['x', 'y'], target_type='latent',
            max_degree=3, lasso_alpha=1e-3, h_degree=3)

    print(f"{'dim':>4s}  {'R2(g)':>8s}  {'R2(hog)':>8s}  {'affine':>6s}", flush=True)
    for ti, dr in results.items():
        print(f"z[{ti}]  {dr['g_recon_r2']:8.4f}  {dr['recon_r2']:8.4f}  "
              f"{str(dr['outer'].get('used_affine', False)):>6s}", flush=True)

    att_pts = ikeda_attractor_sample(n_points=100000, u=0.9, burn_in=1000)
    return results, att_pts


def save(fig, name, copy_to_docs):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = OUTDIR / f'{name}.{ext}'
        fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
        print(f"  Saved {path.relative_to(ROOT)}", flush=True)
        if copy_to_docs:
            shutil.copy(path, DOCS_FIGS / path.name)
            print(f"  Copied to docs/figs/{path.name}", flush=True)
    plt.close(fig)


def column_header(ax, letter, text):
    ax.text(0.0, 1.2, f'({letter})', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom', ha='right')
    ax.text(0.04, 1.2, text, transform=ax.transAxes, fontsize=8,
            va='bottom', ha='left')


def row_title(ax, text):
    ax.text(0.0, 1.03, text, transform=ax.transAxes, fontsize=7.5,
            va='bottom', ha='left')


def plot_attractor_h(ax, att_pts, hg_att):
    vmin, vmax = np.percentile(hg_att, 2), np.percentile(hg_att, 98)
    sc = ax.scatter(att_pts[:, 0], att_pts[:, 1], s=0.04, c=hg_att,
                    cmap='viridis', vmin=vmin, vmax=vmax, rasterized=True,
                    edgecolors='none')
    cax = ax.inset_axes([1.06, 0.08, 0.07, 0.84])
    cb = ax.figure.colorbar(sc, cax=cax)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2, labelsize=6.5)
    cb.locator = MaxNLocator(3)
    cb.update_ticks()


def plot_level_sets(ax, att_pts, g_att):
    ax.scatter(att_pts[:, 0], att_pts[:, 1], s=0.03, c=ATTRACTOR_GREY,
               rasterized=True, edgecolors='none')
    g_lo, g_hi = np.percentile(g_att, 12), np.percentile(g_att, 88)
    band_w = (g_hi - g_lo) * 0.035

    handles = []
    for lev, col in zip(np.linspace(g_lo, g_hi, 5), BAND_COLORS):
        mask = np.abs(g_att - lev) < band_w
        if mask.sum() < 20:
            continue
        ax.scatter(att_pts[mask, 0], att_pts[mask, 1], c=col, s=0.08,
                   rasterized=True, edgecolors='none')
        handles.append(Line2D([], [], ls='', marker='s', ms=3.5, color=col,
                              label=f'{lev:.2f}'))
    # Reversed so the legend reads high-to-low, like the colorbar beside it.
    ax.legend(handles=handles[::-1], title='$g$', loc='center left',
              bbox_to_anchor=(1.0, 0.5), fontsize=6.5, title_fontsize=7.5,
              handletextpad=0.1, labelspacing=0.3, borderaxespad=0.2,
              borderpad=0)


def plot_h_prime(ax, outer):
    ax.scatter(outer['g_samples'], outer['h_prime_samples'], s=0.4,
               c=NEUTRAL, alpha=0.3, rasterized=True, edgecolors='none')
    if len(outer['h_bin_centers']) > 0:
        ax.plot(outer['h_bin_centers'] + outer['g_center'],
                outer['h_bin_medians'], 'o', color=RED, ms=1.6, mew=0, zorder=5)
    ax.plot(outer['g_plot'], outer['h_prime_fit'], color=BLUE, lw=1.1, zorder=6)
    ax.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
    ax.set_ylabel("$h'(g)$", labelpad=1)


def make_figure(results, att_pts, copy_to_docs):
    fig, axes = plt.subplots(
        LATENT_DIM, 3, figsize=(TEXT_WIDTH, FIG_HEIGHT),
        gridspec_kw={'width_ratios': [1.0, 1.0, 1.25], 'wspace': 0.62,
                     'hspace': 0.34})

    x_lim = (att_pts[:, 0].min() - 0.1, att_pts[:, 0].max() + 0.1)
    y_lim = (att_pts[:, 1].min() - 0.1, att_pts[:, 1].max() + 0.1)

    for ti in range(LATENT_DIM):
        dr = results[ti]
        g_att = eval_polynomial_nd(att_pts, dr['inner']['coeffs'],
                                   dr['inner']['powers'])
        hg_att = eval_h(g_att, dr['outer'])
        ax_h, ax_g, ax_d = axes[ti]

        plot_attractor_h(ax_h, att_pts, hg_att)
        plot_level_sets(ax_g, att_pts, g_att)
        plot_h_prime(ax_d, dr['outer'])
        if ti == 3:                      # same x-window as the original figure
            ax_d.set_xlim(left=1.4)

        row_title(ax_h, f"$z_{ti}$,  $R^2 = {dr['recon_r2']:.3f}$")
        row_title(ax_g, f"$z_{ti}$")
        row_title(ax_d, f"$z_{ti}$")

        for ax in (ax_h, ax_g):
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')
            ax.set_anchor('W')
            ax.set_xticks([0, 1])
            ax.set_yticks([-2, -1, 0, 1])
        ax_h.set_ylabel('$y$', labelpad=1)
        ax_g.set_ylabel('$y$', labelpad=1)
        if ti == LATENT_DIM - 1:
            ax_h.set_xlabel('$x$', labelpad=1)
            ax_g.set_xlabel('$x$', labelpad=1)
            ax_d.set_xlabel('$g$', labelpad=1)
        else:
            ax_h.tick_params(labelbottom=False)
            ax_g.tick_params(labelbottom=False)

    column_header(axes[0, 0], 'a', '$h \\circ g$ on the attractor')
    column_header(axes[0, 1], 'b', 'Level sets of $g$')
    column_header(axes[0, 2], 'c', "Outer derivative $h'(g)$")

    handles = [
        Line2D([], [], ls='', marker='o', ms=2.5, mew=0, color=NEUTRAL,
               label="$h'$ samples"),
        Line2D([], [], ls='', marker='o', ms=2.5, mew=0, color=RED,
               label='Binned median'),
        Line2D([], [], color=BLUE, lw=1.1, label="Cubic fit"),
    ]
    axes[-1, 2].legend(handles=handles, loc='upper center', ncol=3,
                       bbox_to_anchor=(0.45, -0.3), handlelength=1.2,
                       handletextpad=0.3, columnspacing=0.8, borderaxespad=0)

    save(fig, FIG_NAME, copy_to_docs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--copy-to-docs', action='store_true',
                        help='also copy the outputs into docs/figs/')
    args = parser.parse_args()

    apply_style()
    results, att_pts = compute()
    make_figure(results, att_pts, args.copy_to_docs)


if __name__ == '__main__':
    main()
