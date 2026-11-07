"""
Publication figures for the KS appendix (NeurIPS): space-time rollout and NRMSE.

Loads the paper model -- the WIDE [64, 96, 128] KAN autoencoder, 500 epochs,
results/ks_stable_wide/ks_stable_model.pt -- and the cached KS data
(results/ks_stable/ks_data.pt), then redraws only the two rollout figures
produced by ks_stable.py (plot_spacetime_rollout, plot_nrmse). No training.

Rollout and NRMSE definitions are imported from ks_stable.py unchanged, so the
horizon printed here (first mean-NRMSE crossing of 1.0) is the same quantity
quoted in docs/appendix.tex (1.45 tau_L).

Sizing: NeurIPS \\textwidth = 5.5 in; fonts are Times-like (STIX) to match the
body text, 7-9 pt at print size. Heatmaps are rasterized inside vector PDFs.

Usage:
    python experiments/ks_paper_figures.py
    python experiments/ks_paper_figures.py --copy-to-docs   # also write docs/figs/
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

from models import KANAutoEncoder
from ks_stable import (DeepKoopmanCTStable, rollout_ks, compute_nrmse,
                       INPUT_SIZE, LATENT_SIZE, L_DOMAIN, N_GRID, DT_SAVE,
                       TAU_L, N_TRAIN)

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / 'results' / 'ks_stable_wide' / 'ks_stable_model.pt'
DATA = ROOT / 'results' / 'ks_stable' / 'ks_data.pt'
OUTDIR = ROOT / 'results' / 'ks_stable_wide' / 'paper_figs'
DOCS_FIGS = ROOT / 'docs' / 'figs'

N_ROLLOUT = 400          # 100 time units = 4.3 tau_L (same as ks_stable.py)
SPACETIME_TRAJ = 0       # validation trajectory shown in the space-time figure
TEXT_WIDTH = 5.5         # NeurIPS \textwidth (in)

# Semantic colors (scientific-figure-making PALETTE)
BLUE = '#0F4D92'
BLUE_LIGHT = '#3775BA'
RED = '#B64342'
NEUTRAL = '#9E9E9E'


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
    cfg = ckpt['config']
    assert cfg['mid_size'] == 96 and cfg['epochs'] == 500, \
        f"expected the wide [64,96,128] 500-epoch model, got {cfg}"
    ae = KANAutoEncoder(INPUT_SIZE, cfg['mid_size'], LATENT_SIZE,
                        grid_range_l1=[-5, 5], grid_range_l2=[-10, 10])
    model = DeepKoopmanCTStable(ae, LATENT_SIZE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


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


def panel_label(ax, letter, x=-0.07, y=1.0):
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom', ha='right')


# =====================================================================
# Figure 1: space-time rollout (true / predicted / absolute error)
# =====================================================================

def plot_spacetime(true, pred, horizon_tau, copy_to_docs):
    """Three stacked x-t panels, time on the horizontal axis in tau_L."""
    err = np.abs(true - pred)
    t_tau = np.arange(true.shape[0]) * DT_SAVE / TAU_L
    x = np.linspace(0, L_DOMAIN, N_GRID, endpoint=False)
    vmax = np.ceil(max(np.abs(true).max(), np.abs(pred).max()))

    fig, axes = plt.subplots(3, 1, figsize=(TEXT_WIDTH, 3.6), sharex=True,
                             gridspec_kw={'hspace': 0.18})
    fields = [(true, 'True $u(x,t)$', 'RdBu_r', -vmax, vmax),
              (pred, 'Predicted $\\hat u(x,t)$', 'RdBu_r', -vmax, vmax),
              (err, 'Error $|u-\\hat u|$', 'magma', 0, np.ceil(err.max()))]

    for i, (ax, (data, label, cmap, lo, hi)) in enumerate(zip(axes, fields)):
        im = ax.pcolormesh(t_tau, x, data.T, cmap=cmap, vmin=lo, vmax=hi,
                           shading='nearest', rasterized=True)
        ax.set_ylabel('$x$', labelpad=1)
        ax.set_yticks([0, 11, 22])
        ax.set_ylim(0, L_DOMAIN)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(True)
        ax.axvline(horizon_tau, color='k' if i < 2 else 'w',
                   ls='--', lw=0.8, alpha=0.9)
        ax.text(0.005, 0.93, label, transform=ax.transAxes, fontsize=7.5,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none',
                          alpha=0.85))
        cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03, aspect=10)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(width=0.5, length=2, labelsize=6.5)
        cb.set_ticks([lo, 0, hi] if lo < 0 else [0, hi / 2, hi])
        panel_label(ax, 'abc'[i], x=-0.075)

    axes[0].text(horizon_tau, 1.03, f'{horizon_tau:.2f}$\\,\\tau_L$',
                 transform=axes[0].get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=7)
    axes[-1].set_xlim(0, t_tau[-1])
    axes[-1].set_xlabel('Time ($\\tau_L$)')
    save(fig, 'ks_spacetime_rollout', copy_to_docs)


# =====================================================================
# Figure 2: NRMSE vs time over the validation trajectories
# =====================================================================

def plot_nrmse(all_nrmse, copy_to_docs):
    """(a) linear scale with per-trajectory curves, (b) log-scale growth."""
    n_val = all_nrmse.shape[0]
    mean, std = all_nrmse.mean(0), all_nrmse.std(0)
    t_tau = np.arange(all_nrmse.shape[1]) * DT_SAVE / TAU_L
    cross = np.where(mean > 1.0)[0]
    horizon_tau = t_tau[cross[0]]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.95),
                             gridspec_kw={'wspace': 0.32,
                                          'width_ratios': [1.35, 1]})

    # (a) linear
    ax = axes[0]
    for k in range(n_val):
        ax.plot(t_tau, all_nrmse[k], color=NEUTRAL, lw=0.5, alpha=0.7,
                label='Validation trajectory' if k == 0 else None)
    ax.fill_between(t_tau, np.maximum(mean - std, 0), mean + std,
                    color=BLUE_LIGHT, alpha=0.18, lw=0, label='Mean $\\pm$ 1 s.d.')
    ax.plot(t_tau, mean, color=BLUE, lw=1.4, label='Mean NRMSE')
    ax.axhline(1.0, color=RED, ls='--', lw=0.8, label='NRMSE $=1$')
    ax.axvline(horizon_tau, color='k', ls=':', lw=0.8)
    ax.plot(horizon_tau, 1.0, 'o', ms=3.5, mfc='white', mec='k', mew=0.8,
            zorder=5)
    ax.annotate(f'{horizon_tau:.2f}$\\,\\tau_L$', xy=(horizon_tau, 1.0),
                xytext=(horizon_tau + 0.25, 0.3), fontsize=7,
                arrowprops=dict(arrowstyle='-|>', lw=0.6, color='k',
                                mutation_scale=6))
    ax.set_xlim(0, t_tau[-1])
    ax.set_ylim(0, 2.0)
    ax.set_xlabel('Time ($\\tau_L$)')
    ax.set_ylabel('NRMSE')
    panel_label(ax, 'a', x=-0.11, y=1.02)

    # (b) log
    ax = axes[1]
    ax.fill_between(t_tau[1:], np.maximum(mean - std, 1e-4)[1:],
                    (mean + std)[1:], color=BLUE_LIGHT, alpha=0.18, lw=0)
    ax.semilogy(t_tau[1:], mean[1:], color=BLUE, lw=1.4)
    ax.axhline(1.0, color=RED, ls='--', lw=0.8)
    ax.axvline(horizon_tau, color='k', ls=':', lw=0.8)
    ax.set_xlim(0, t_tau[-1])
    ax.set_ylim(1e-3, 5)
    ax.set_xlabel('Time ($\\tau_L$)')
    ax.set_ylabel('NRMSE')
    panel_label(ax, 'b', x=-0.16, y=1.02)

    # One shared legend above both panels
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(0.5, 0.97), ncol=4, handlelength=1.4,
               columnspacing=1.2, borderaxespad=0.0)

    save(fig, 'ks_nrmse', copy_to_docs)
    return horizon_tau


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--copy-to-docs', action='store_true',
                        help='also copy the outputs into docs/figs/')
    args = parser.parse_args()

    apply_style()
    model, cfg = load_model()
    print(f"Loaded {CKPT.relative_to(ROOT)}: [{cfg['input_size']}, "
          f"{cfg['mid_size']}, {cfg['latent_size']}], {cfg['epochs']} epochs",
          flush=True)

    traj = torch.load(DATA, map_location='cpu', weights_only=True)
    val_traj = traj[N_TRAIN:]

    print(f"Rolling out {val_traj.shape[0]} validation trajectories "
          f"x {N_ROLLOUT} steps...", flush=True)
    preds, all_nrmse = [], []
    for i in range(val_traj.shape[0]):
        pred = rollout_ks(model, val_traj[i, 0], N_ROLLOUT, DT_SAVE)
        preds.append(pred.numpy())
        all_nrmse.append(compute_nrmse(pred, val_traj[i, :N_ROLLOUT + 1]))
    all_nrmse = np.array(all_nrmse)

    horizon_tau = plot_nrmse(all_nrmse, args.copy_to_docs)
    plot_spacetime(val_traj[SPACETIME_TRAJ, :N_ROLLOUT + 1].numpy(),
                   preds[SPACETIME_TRAJ], horizon_tau, args.copy_to_docs)

    mean = all_nrmse.mean(0)
    step = int(np.where(mean > 1.0)[0][0])
    print(f"Horizon (first mean NRMSE > 1): {horizon_tau:.2f} tau_L = "
          f"{step * DT_SAVE:.1f} time units = {step} Koopman steps", flush=True)


if __name__ == '__main__':
    main()
