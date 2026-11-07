"""
Eigenmode–dictionary interaction for the [3,5,5] PyKAN stable Lorenz model.

The KAN encoder discovers a 5D invariant subspace that can be described
in two bases:
  1. Eigenmodes — dynamically independent damped oscillators
  2. Dictionary — physically interpretable monomials {x, y, z, xy, xz}

This script computes how the two bases relate: which eigenmodes carry
which dictionary terms, and vice versa.

Run from project root:
    python experiments/eigenmode_dictionary.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

from data import simulate_lorenz
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable

OUTDIR = Path("results/lorenz_5d_second_test")
DT = 0.01
MID = 5
LATENT = 5
LAMBDA_MAX = 0.9056
TAU_L = 1.0 / LAMBDA_MAX
POLY_NAMES = ["x", "y", "z", "xy", "xz"]


def main():
    print("=" * 70)
    print("  EIGENMODE–DICTIONARY INTERACTION")
    print("=" * 70)

    # Load model
    ae = PyKANAutoEncoder(3, MID, LATENT, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, LATENT)
    ckpt = torch.load(OUTDIR / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    enc = model.ae.encoder

    # Data
    set_seed(42)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    x_flat = traj[:48].reshape(-1, 3)

    # Psi activations on a large sample
    idx = np.random.choice(len(x_flat), 10000, replace=False)
    x_data = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    x_np = x_data.numpy()
    with torch.no_grad():
        enc(x_data)
        psi = enc.acts[1].numpy()

    # Fit K_psi
    pair_idx = np.random.choice(len(x_flat) - 1, 50000, replace=False)
    x_t = torch.tensor(x_flat[pair_idx].numpy(), dtype=torch.float32)
    x_tp1 = torch.tensor(x_flat[pair_idx + 1].numpy(), dtype=torch.float32)
    with torch.no_grad():
        enc(x_t); psi_t = enc.acts[1].numpy()
        enc(x_tp1); psi_tp1 = enc.acts[1].numpy()
    psi_t_wc = np.column_stack([np.ones(len(psi_t)), psi_t])
    K_full, _, _, _ = np.linalg.lstsq(psi_t_wc, psi_tp1, rcond=None)
    K_psi = K_full[1:, :]

    # Eigendecomposition
    eigvals, V = np.linalg.eig(K_psi)
    V_inv = np.linalg.inv(V)
    mu = np.log(eigvals.astype(complex)) / DT
    order = np.argsort(-np.abs(mu.imag))

    # Modal coordinates
    modal = psi @ V_inv.T

    # Polynomial observables
    x, y, z = x_np[:, 0], x_np[:, 1], x_np[:, 2]
    poly_data = np.column_stack([x, y, z, x * y, x * z])

    # Mode labels
    mode_labels = []
    for rank in range(LATENT):
        oidx = order[rank]
        freq = mu[oidx].imag / (2 * np.pi)
        decay = mu[oidx].real
        mode_labels.append(f"M{rank} ({freq:+.1f} Hz)")

    # ── Compute correlations ──

    # Re(mode) vs polynomial
    corr_mode_poly = np.zeros((LATENT, len(POLY_NAMES)))
    for rank in range(LATENT):
        modal_re = modal[:, order[rank]].real
        for p in range(len(POLY_NAMES)):
            corr_mode_poly[rank, p] = np.corrcoef(modal_re, poly_data[:, p])[0, 1]

    # |mode amplitude| vs polynomial (for complex modes)
    amp_corr = np.zeros((LATENT, len(POLY_NAMES)))
    for rank in range(LATENT):
        amp = np.abs(modal[:, order[rank]])
        for p in range(len(POLY_NAMES)):
            amp_corr[rank, p] = np.corrcoef(amp, poly_data[:, p])[0, 1]

    # Regression: polynomial = sum of eigenmodes
    modal_real_parts = np.column_stack([modal[:, order[r]].real for r in range(LATENT)])
    modal_wc = np.column_stack([np.ones(len(modal_real_parts)), modal_real_parts])

    regression_r2 = {}
    regression_coeffs = {}
    for pi, pn in enumerate(POLY_NAMES):
        target = poly_data[:, pi]
        coeffs, _, _, _ = np.linalg.lstsq(modal_wc, target, rcond=None)
        pred = modal_wc @ coeffs
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        regression_r2[pn] = r2
        regression_coeffs[pn] = coeffs[1:]  # skip intercept

    # Print results
    print(f"\n  Correlations: Re(mode) vs polynomial observable")
    print(f"  {'Mode':>15}", end="")
    for pn in POLY_NAMES:
        print(f"  {pn:>6}", end="")
    print()
    print("  " + "-" * 50)
    for rank in range(LATENT):
        print(f"  {mode_labels[rank]:>15}", end="")
        for p in range(len(POLY_NAMES)):
            print(f"  {corr_mode_poly[rank, p]:>6.3f}", end="")
        print()

    print(f"\n  Amplitude correlations: |mode| vs polynomial")
    print(f"  {'Mode':>15}", end="")
    for pn in POLY_NAMES:
        print(f"  {pn:>6}", end="")
    print()
    print("  " + "-" * 50)
    for rank in range(LATENT):
        print(f"  {mode_labels[rank]:>15}", end="")
        for p in range(len(POLY_NAMES)):
            print(f"  {amp_corr[rank, p]:>6.3f}", end="")
        print()

    print(f"\n  Regression: observable = Σ c_k · Re(mode_k)")
    for pn in POLY_NAMES:
        c = regression_coeffs[pn]
        r2 = regression_r2[pn]
        terms = []
        for r in range(LATENT):
            if abs(c[r]) > 0.5:
                terms.append(f"{c[r]:+.1f}·M{r}")
        print(f"    {pn:>4}: R²={r2:.4f}  {' '.join(terms)}")

    # ── Get a trajectory for time series ──
    traj_seg = traj[0, :3000].numpy()
    x_seg = torch.tensor(traj_seg, dtype=torch.float32)
    with torch.no_grad():
        enc(x_seg)
        psi_seg = enc.acts[1].numpy()
    modal_seg = psi_seg @ V_inv.T
    t = np.arange(len(traj_seg)) * DT

    # ══════════════════════════════════════════════════════════════
    # FIGURE: Hero plot
    # ══════════════════════════════════════════════════════════════

    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35,
                  height_ratios=[1.2, 1, 1, 1, 1])

    # ── Row 0: Correlation heatmaps side by side ──

    # Panel (0,0): Re(mode) vs polynomial correlation
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(corr_mode_poly, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(POLY_NAMES)))
    ax.set_xticklabels(POLY_NAMES, fontsize=10)
    ax.set_yticks(range(LATENT))
    ax.set_yticklabels(mode_labels, fontsize=8)
    for i in range(LATENT):
        for j in range(len(POLY_NAMES)):
            v = corr_mode_poly[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Re(eigenmode) vs observable\n(correlation)", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel (0,1): |mode amplitude| vs polynomial
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(amp_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(POLY_NAMES)))
    ax.set_xticklabels(POLY_NAMES, fontsize=10)
    ax.set_yticks(range(LATENT))
    ax.set_yticklabels(mode_labels, fontsize=8)
    for i in range(LATENT):
        for j in range(len(POLY_NAMES)):
            v = amp_corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.5 else "black")
    ax.set_title("|eigenmode amplitude| vs observable\n(spatial structure)", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel (0,2): Regression R² bar chart
    ax = fig.add_subplot(gs[0, 2])
    r2_vals = [regression_r2[pn] for pn in POLY_NAMES]
    colors_bar = ["steelblue" if pn in {"x", "y", "z"} else "coral" for pn in POLY_NAMES]
    ax.bar(range(len(POLY_NAMES)), r2_vals, color=colors_bar, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(POLY_NAMES)))
    ax.set_xticklabels(POLY_NAMES, fontsize=10)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title("Observable = Σ eigenmodes\n(regression R²)", fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.3)
    for i, (pn, r2) in enumerate(zip(POLY_NAMES, r2_vals)):
        ax.text(i, r2 + 0.01, f"{r2:.3f}", ha="center", fontsize=8)

    # ── Rows 1-2: Attractor colored by eigenmode (Re) and by dictionary term ──
    for rank in range(LATENT):
        # Row 1: eigenmode on attractor
        ax = fig.add_subplot(gs[1, rank] if rank < 3 else gs[2, rank - 3],
                              projection="3d")
        oidx = order[rank]
        c = modal[:, oidx].real
        vmax = np.percentile(np.abs(c), 95)
        ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                  c=c, cmap="coolwarm", s=0.3, alpha=0.3,
                  vmin=-vmax, vmax=vmax)
        freq = mu[oidx].imag / (2 * np.pi)
        ax.set_title(f"Eigenmode M{rank}\n{freq:+.1f} Hz, Re(μ)={mu[oidx].real:.1f}",
                    fontsize=9)
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=5)

    # Fill remaining slots with dictionary terms on attractor
    dict_plot = [("xy", x * y), ("xz", x * z)]
    for di, (dname, dvals) in enumerate(dict_plot):
        ax = fig.add_subplot(gs[2, 2 + di] if 2 + di < 3 else gs[2, 2],
                              projection="3d")
        vmax = np.percentile(np.abs(dvals), 95)
        ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                  c=dvals, cmap="coolwarm", s=0.3, alpha=0.3,
                  vmin=-vmax, vmax=vmax)
        ax.set_title(f"Dictionary: {dname}", fontsize=9)
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=5)

    # ── Rows 3-4: Time series — eigenmode vs dictionary term ──

    # Row 3: eigenmode time series
    for rank in range(LATENT):
        col = rank if rank < 3 else rank
        ax = fig.add_subplot(gs[3, rank] if rank < 3 else gs[4, rank - 3])
        oidx = order[rank]
        mk = modal_seg[:, oidx]
        freq = mu[oidx].imag / (2 * np.pi)
        ax.plot(t, mk.real, "b-", lw=0.6, alpha=0.8, label="Re(mode)")
        if abs(mu[oidx].imag) > 0.1:
            ax.plot(t, np.abs(mk), "k--", lw=0.4, alpha=0.4, label="|mode|")
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_title(f"M{rank} ({freq:+.1f} Hz)", fontsize=9)
        ax.set_xlim(0, 5)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    # Row 4: dictionary term time series (for comparison)
    for di, (dname, dcol) in enumerate([("xy", 3), ("xz", 4)]):
        ax = fig.add_subplot(gs[4, 2 + di] if 2 + di < 3 else gs[4, 2])
        ax.plot(t, traj_seg[:, 0] * traj_seg[:, 1] if dname == "xy"
                else traj_seg[:, 0] * traj_seg[:, 2],
                "coral", lw=0.6, alpha=0.8)
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_title(f"Dictionary: {dname}(t)", fontsize=9)
        ax.set_xlim(0, 5)
        ax.tick_params(labelsize=6)

    fig.suptitle("Eigenmode–Dictionary Interaction\n"
                 "Eigenmodes (dynamically independent) vs Dictionary terms "
                 "(physically interpretable)\nspan the same 5D Koopman-invariant subspace",
                 fontsize=13, y=1.01)
    fig.savefig(OUTDIR / "eigenmode_dictionary.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTDIR / "eigenmode_dictionary.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved eigenmode_dictionary.png/pdf")

    # ── Save results ──
    save_data = {
        "mode_labels": mode_labels,
        "eigenvalues": [
            {"mode": rank,
             "abs_lambda": float(abs(eigvals[order[rank]])),
             "re_mu": float(mu[order[rank]].real),
             "freq_hz": float(mu[order[rank]].imag / (2 * np.pi))}
            for rank in range(LATENT)
        ],
        "correlation_re_mode_vs_poly": {
            pn: [float(corr_mode_poly[r, p]) for r in range(LATENT)]
            for p, pn in enumerate(POLY_NAMES)
        },
        "correlation_amp_vs_poly": {
            pn: [float(amp_corr[r, p]) for r in range(LATENT)]
            for p, pn in enumerate(POLY_NAMES)
        },
        "regression_r2": {pn: float(v) for pn, v in regression_r2.items()},
        "regression_coeffs": {
            pn: [float(c) for c in regression_coeffs[pn]]
            for pn in POLY_NAMES
        },
    }
    with open(OUTDIR / "eigenmode_dictionary.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved eigenmode_dictionary.json")


if __name__ == "__main__":
    main()
