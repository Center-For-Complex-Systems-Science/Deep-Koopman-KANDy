"""
Plot the eigenmode structure of the 8x8 psi-space Koopman matrix.

Visualizes:
  1. The full pipeline: input → Layer 1 splines → K_psi eigenmodes → frequencies
  2. Eigenvector participation: which psi's feed each mode
  3. Modal coordinates on the Lorenz attractor (3D, colored by each mode)
  4. Time series of modal coordinates showing oscillation + decay
  5. Eigenvalue spectrum in the complex plane

Run from project root:
    python experiments/plot_eigenmode_structure.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
from pathlib import Path
from scipy.linalg import logm
from numpy.polynomial import polynomial as P

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

from data import simulate_lorenz
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable

OUTDIR = Path("results/spline_basis_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)
DT = 0.01
MID = 8
VAR = ["x", "y", "z"]


def load_model():
    latent_dim = 32
    ae = PyKANAutoEncoder(3, 8, latent_dim, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, latent_dim)
    ckpt = torch.load("results/lorenz_stable/stable_8_32_model.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, latent_dim


def get_psi(enc, x_data):
    with torch.no_grad():
        enc(x_data)
        return enc.acts[1].numpy()


def fit_kpsi(enc, train_traj):
    x_flat = train_traj.reshape(-1, 3)
    pair_idx = np.random.choice(len(x_flat) - 1, 50000, replace=False)
    x_t = torch.tensor(x_flat[pair_idx].numpy(), dtype=torch.float32)
    x_tp1 = torch.tensor(x_flat[pair_idx + 1].numpy(), dtype=torch.float32)
    psi_t = get_psi(enc, x_t)
    psi_tp1 = get_psi(enc, x_tp1)
    psi_t_wc = np.column_stack([np.ones(len(psi_t)), psi_t])
    K_full, _, _, _ = np.linalg.lstsq(psi_t_wc, psi_tp1, rcond=None)
    return K_full[1:, :], K_full[0, :]


def extract_edges(enc, x_data):
    x_np = x_data.numpy()
    x_mean = torch.tensor(x_np.mean(axis=0), dtype=torch.float32)
    # Baseline
    with torch.no_grad():
        enc(x_mean.unsqueeze(0))
        baseline = enc.acts[1].numpy().flatten()
    edges = {}
    for i in range(3):
        xi = np.linspace(x_np[:, i].min(), x_np[:, i].max(), 300)
        probe = x_mean.unsqueeze(0).repeat(300, 1)
        probe[:, i] = torch.tensor(xi, dtype=torch.float32)
        with torch.no_grad():
            enc(probe)
            l1 = enc.acts[1].numpy()
        for j in range(MID):
            edges[(j, i)] = (xi, l1[:, j] - baseline[j])
    return edges


def main():
    model, latent_dim = load_model()
    enc = model.ae.encoder

    set_seed(42)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:48]
    x_flat = train_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 8000, replace=False)
    x_data = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    x_np = x_data.numpy()

    # Fit K_psi and decompose
    K_psi, b_psi = fit_kpsi(enc, train_traj)
    eigvals, V = np.linalg.eig(K_psi)
    V_inv = np.linalg.inv(V)
    A_psi = logm(K_psi) / DT
    mu = np.log(eigvals) / DT

    # Sort: conjugate pairs grouped, then real modes
    order = np.argsort(-np.abs(mu.imag))

    # Get psi and modal coordinates on data
    psi_data = get_psi(enc, x_data)
    modal_data = psi_data @ V_inv.T  # (N, 8) complex

    # Get edges for spline plots
    edges = extract_edges(enc, x_data)

    # Classify edge linearity
    lin_r2 = np.zeros((MID, 3))
    for j in range(MID):
        for i in range(3):
            xi, eo = edges[(j, i)]
            c = P.polyfit(xi, eo, 1)
            pred = P.polyval(xi, c)
            ss_res = np.sum((eo - pred)**2)
            ss_tot = np.sum((eo - eo.mean())**2)
            lin_r2[j, i] = 1 - ss_res / max(ss_tot, 1e-12)

    # Get a trajectory segment for time series
    traj_seg = traj[0, :2000].numpy()  # single trajectory, 2000 steps
    x_seg = torch.tensor(traj_seg, dtype=torch.float32)
    psi_seg = get_psi(enc, x_seg)
    modal_seg = psi_seg @ V_inv.T
    t_seg = np.arange(len(traj_seg)) * DT

    # ══════════════════════════════════════════════════════════════
    # FIGURE 1: Hero figure — full pipeline
    # ══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)

    # ── Row 0: Layer-1 edge functions (top 4 most nonlinear psi nodes) ──
    # Rank psi nodes by max nonlinearity across inputs
    nl_scores = [min(lin_r2[j, :]) for j in range(MID)]
    top_psi = np.argsort(nl_scores)[:4]

    for col, j in enumerate(top_psi):
        ax = fig.add_subplot(gs[0, col])
        for i, (color, vn) in enumerate(zip(["#e74c3c", "#2ecc71", "#3498db"], VAR)):
            xi, eo = edges[(j, i)]
            r2 = lin_r2[j, i]
            lw = 2.5 if r2 < 0.95 else 1.0
            alpha = 1.0 if r2 < 0.95 else 0.4
            label = f"$\\phi({vn})$" if r2 < 0.95 else f"$\\phi({vn})$ [lin]"
            ax.plot(xi, eo, color=color, lw=lw, alpha=alpha, label=label)
        ax.axhline(0, color="gray", ls=":", alpha=0.3)
        ax.set_title(f"$\\psi_{j}$ edges (min linR²={nl_scores[j]:.2f})", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.tick_params(labelsize=7)

    # ── Row 1: Eigenvector participation heatmap ──
    ax_heat = fig.add_subplot(gs[1, :3])
    participation = np.zeros((MID, MID))
    mode_labels = []
    for rank, oidx in enumerate(order):
        v = V[:, oidx]
        v_norm = np.abs(v) / np.max(np.abs(v))
        participation[:, rank] = v_norm
        freq = mu[oidx].imag / (2 * np.pi)
        decay = mu[oidx].real
        mode_labels.append(f"M{rank}\n{freq:+.1f}Hz\nRe={decay:.1f}")

    im = ax_heat.imshow(participation, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax_heat.set_yticks(range(MID))
    ax_heat.set_yticklabels([f"$\\psi_{j}$" for j in range(MID)], fontsize=9)
    ax_heat.set_xticks(range(MID))
    ax_heat.set_xticklabels(mode_labels, fontsize=7)
    ax_heat.set_title("Eigenvector Participation: |V[ψ_j, mode]| (normalized)", fontsize=11)
    plt.colorbar(im, ax=ax_heat, shrink=0.8)

    # Annotate cells
    for i in range(MID):
        for j in range(MID):
            val = participation[i, j]
            if val > 0.3:
                ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if val > 0.6 else "black")

    # ── Row 1, col 3: Eigenvalue spectrum ──
    ax_spec = fig.add_subplot(gs[1, 3])
    theta = np.linspace(0, 2*np.pi, 100)
    ax_spec.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.2, lw=1)
    colors_mode = plt.cm.tab10(np.linspace(0, 1, MID))
    for rank, oidx in enumerate(order):
        lam = eigvals[oidx]
        ax_spec.plot(lam.real, lam.imag, "o", color=colors_mode[rank],
                    ms=10, label=f"M{rank}", zorder=5)
        freq = mu[oidx].imag / (2*np.pi)
        ax_spec.annotate(f"{freq:+.1f}Hz", (lam.real, lam.imag),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(5, 3), textcoords="offset points")
    ax_spec.set_xlim(0.9, 1.01)
    ax_spec.set_ylim(-0.15, 0.15)
    ax_spec.set_xlabel("Re(λ)", fontsize=9)
    ax_spec.set_ylabel("Im(λ)", fontsize=9)
    ax_spec.set_title("Discrete eigenvalues", fontsize=10)
    ax_spec.set_aspect("equal")
    ax_spec.grid(True, alpha=0.3)
    ax_spec.tick_params(labelsize=7)

    # ── Row 2: Modal coordinates on the attractor (3D) ──
    # Show 4 modes: the 3 oscillatory pairs (real part) + slow real mode
    show_modes = [0, 2, 4, 7]  # 2.0Hz, 1.4Hz, 0.9Hz, slow
    for col, rank in enumerate(show_modes):
        ax3d = fig.add_subplot(gs[2, col], projection="3d")
        oidx = order[rank]
        modal_k = modal_data[:, oidx]
        freq = mu[oidx].imag / (2*np.pi)
        decay = mu[oidx].real

        # Color by real part of modal coordinate
        c_vals = modal_k.real
        vmax = np.percentile(np.abs(c_vals), 95)
        sc = ax3d.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                         c=c_vals, cmap="coolwarm", s=0.3, alpha=0.3,
                         vmin=-vmax, vmax=vmax)
        ax3d.set_xlabel("x", fontsize=7)
        ax3d.set_ylabel("y", fontsize=7)
        ax3d.set_zlabel("z", fontsize=7)
        ax3d.set_title(f"Mode {rank}: {freq:+.1f} Hz\nRe(μ)={decay:.1f}",
                       fontsize=9)
        ax3d.view_init(elev=22, azim=-55)
        ax3d.tick_params(labelsize=5)

    # ── Row 3: Time series of modal coordinates ──
    for col, rank in enumerate(show_modes):
        ax_ts = fig.add_subplot(gs[3, col])
        oidx = order[rank]
        modal_k = modal_seg[:, oidx]
        freq = mu[oidx].imag / (2*np.pi)

        ax_ts.plot(t_seg, modal_k.real, "b-", lw=0.8, alpha=0.8, label="Re")
        if abs(mu[oidx].imag) > 0.1:
            ax_ts.plot(t_seg, modal_k.imag, "r-", lw=0.8, alpha=0.5, label="Im")
            ax_ts.plot(t_seg, np.abs(modal_k), "k--", lw=0.5, alpha=0.4, label="|z|")

        ax_ts.set_xlabel("t (s)", fontsize=8)
        ax_ts.set_ylabel(f"Mode {rank}", fontsize=8)
        ax_ts.set_title(f"M{rank} time series ({freq:+.1f} Hz)", fontsize=9)
        ax_ts.tick_params(labelsize=6)
        ax_ts.legend(fontsize=6)
        ax_ts.set_xlim(0, 5)

    fig.suptitle("Eigenmode Structure of Learned Koopman Operator in Spline Basis\n"
                 "(x,y,z) → Layer-1 ψ → K_ψ eigenmodes → damped oscillators",
                 fontsize=14, y=1.01)
    fig.savefig(OUTDIR / "eigenmode_structure.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTDIR / "eigenmode_structure.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eigenmode_structure.png/pdf")

    # ══════════════════════════════════════════════════════════════
    # FIGURE 2: Compact pipeline diagram
    # ══════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 8))

    # Top row: the 8 psi nodes — show their most nonlinear edge
    for j in range(MID):
        ax = axes2[0, j] if j < 4 else axes2[1, j - 4]
        # Find most nonlinear input
        most_nl = np.argmin(lin_r2[j, :])
        xi, eo = edges[(j, most_nl)]
        ax.plot(xi, eo, "b-", lw=2)
        ax.axhline(0, color="gray", ls=":", alpha=0.3)
        r2 = lin_r2[j, most_nl]
        tag = "NONLINEAR" if r2 < 0.95 else "~linear"
        ax.set_title(f"$\\psi_{j}$: $\\phi$({VAR[most_nl]})  [{tag}]\n"
                     f"linR²={r2:.3f}", fontsize=9)
        ax.tick_params(labelsize=6)

        # Show which modes this psi feeds (from participation matrix)
        mode_str = []
        for rank in range(MID):
            p = participation[j, rank]
            if p > 0.3:
                freq = mu[order[rank]].imag / (2*np.pi)
                mode_str.append(f"M{rank}({freq:+.1f}Hz):{p:.1f}")
        if mode_str:
            ax.text(0.02, 0.02, "\n".join(mode_str), transform=ax.transAxes,
                   fontsize=5, va="bottom", fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7))

    fig2.suptitle("8 Spline Coordinates → Eigenmode Participation", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(OUTDIR / "psi_to_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved psi_to_modes.png")

    # ══════════════════════════════════════════════════════════════
    # FIGURE 3: Layer-2 edge functions (phi_{k,j}(psi_j))
    # Show for top 8 latent dims by variance
    # ══════════════════════════════════════════════════════════════
    z_data = get_psi(enc, x_data)  # This is Layer-1; need Layer-2
    with torch.no_grad():
        enc(x_data)
        if hasattr(enc, 'acts') and len(enc.acts) > 2:
            z_l2 = enc.acts[2].numpy()  # Layer-2 output = latent
        else:
            z_l2 = enc(x_data).numpy()

    z_var = np.var(z_l2, axis=0)
    top_latent = np.argsort(-z_var)[:8]

    fig3, axes3 = plt.subplots(8, MID, figsize=(2.5 * MID, 2.5 * 8))

    for row_idx, k in enumerate(top_latent):
        for j in range(MID):
            ax = axes3[row_idx, j]
            # Scatter z_k vs psi_j, with binned median
            sort_idx = np.argsort(psi_data[:, j])
            psi_s = psi_data[sort_idx, j]
            z_s = z_l2[sort_idx, k]

            n_bins = 40
            be = np.linspace(psi_s.min(), psi_s.max(), n_bins + 1)
            bc, bm = [], []
            for b in range(n_bins):
                mask = (psi_s >= be[b]) & (psi_s < be[b+1])
                if mask.sum() > 5:
                    bc.append((be[b] + be[b+1]) / 2)
                    bm.append(np.median(z_s[mask]))

            if bc:
                ax.plot(bc, bm, "b-", lw=2)

                # Check linearity
                if len(bc) > 2:
                    bc_arr, bm_arr = np.array(bc), np.array(bm)
                    c = P.polyfit(bc_arr, bm_arr, 1)
                    pred = P.polyval(bc_arr, c)
                    ss_res = np.sum((bm_arr - pred)**2)
                    ss_tot = np.sum((bm_arr - bm_arr.mean())**2)
                    r2 = 1 - ss_res / max(ss_tot, 1e-12)
                    color = "red" if r2 < 0.9 else "black"
                    ax.set_title(f"z[{k}]←ψ{j} R²={r2:.2f}",
                                fontsize=7, color=color)
                else:
                    ax.set_title(f"z[{k}]←ψ{j}", fontsize=7)
            else:
                ax.set_title(f"z[{k}]←ψ{j} [empty]", fontsize=7)

            ax.tick_params(labelsize=5)

    fig3.suptitle("Layer-2 Edge Functions: $z_k$ vs $\\psi_j$ (binned median)\n"
                  "Red title = nonlinear (linR² < 0.9)",
                  fontsize=13, y=1.005)
    fig3.tight_layout()
    fig3.savefig(OUTDIR / "layer2_edges_detail.png", dpi=120, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved layer2_edges_detail.png")

    print(f"\n  All figures in {OUTDIR}/")


if __name__ == "__main__":
    main()
