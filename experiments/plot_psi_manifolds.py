"""
Plot Layer-1 additive manifolds and per-edge univariate splines.

For each mid-node j (0..7), the KAN Layer-1 computes:
    psi_j(x,y,z) = phi_{j,0}(x) + phi_{j,1}(y) + phi_{j,2}(z)

This script:
  1. Plots all 24 univariate edge functions phi_{j,i} (8 mid-nodes x 3 inputs)
     showing the actual learned spline, not a symbolic fit
  2. Plots the 3D level-set manifold psi_j = c for each mid-node
  3. Shows the Lorenz attractor colored by psi_j value

Run from project root:
    python experiments/plot_psi_manifolds.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import simulate_lorenz
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable

OUTDIR = Path("results/spline_basis_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)
DT = 0.01
MID_SIZE = 8
VAR_NAMES = ["x", "y", "z"]


def load_model():
    latent_dim = 32
    ae = PyKANAutoEncoder(3, 8, latent_dim, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, latent_dim)
    ckpt = torch.load("results/lorenz_stable/stable_8_32_model.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, latent_dim


def extract_edge_functions(model, x_data):
    """Extract each univariate edge function phi_{j,i}(x_i) by sweeping
    one input at a time while holding others at their mean.

    Returns:
        edges: dict (j, i) -> (xi_range, edge_output)
        psi_data: (N, 8) full Layer-1 activations on real data
    """
    enc = model.ae.encoder
    x_np = x_data.numpy()
    n_probe = 500

    edges = {}
    for i in range(3):
        xi_min, xi_max = x_np[:, i].min(), x_np[:, i].max()
        xi_range = np.linspace(xi_min, xi_max, n_probe)

        x_probe = torch.zeros(n_probe, 3)
        for k in range(3):
            x_probe[:, k] = float(x_np[:, k].mean())
        x_probe[:, i] = torch.tensor(xi_range, dtype=torch.float32)

        with torch.no_grad():
            enc(x_probe)
            l1_out = enc.acts[1].numpy()  # (n_probe, 8)

        # To isolate phi_{j,i}(x_i), we need to subtract the contribution
        # from the other inputs at their mean values.
        # At the mean point, get the baseline:
        x_mean = torch.zeros(1, 3)
        for k in range(3):
            x_mean[0, k] = float(x_np[:, k].mean())
        with torch.no_grad():
            enc(x_mean)
            baseline = enc.acts[1].numpy().flatten()  # (8,)

        for j in range(MID_SIZE):
            # phi_{j,i}(x_i) ≈ l1_out[:, j] - baseline[j] + phi_{j,i}(mean_i)
            # The sweep output includes all 3 inputs at mean except input i.
            # So l1_out[:, j] = phi_{j,i}(x_i) + phi_{j,k≠i}(mean_k)
            # We want just phi_{j,i}(x_i), so subtract baseline and add back
            # phi_{j,i}(mean_i) which equals baseline contribution from i.
            # Actually, the cleanest: just report l1_out[:, j] - baseline[j]
            # This gives phi_{j,i}(x_i) - phi_{j,i}(mean_i), i.e. centered.
            edges[(j, i)] = (xi_range, l1_out[:, j] - baseline[j])

    # Full activations on real data
    with torch.no_grad():
        enc(x_data)
        psi_data = enc.acts[1].numpy()

    return edges, psi_data


def plot_all_edges(edges):
    """Plot 1: All 24 univariate edge functions (8 rows x 3 cols)."""
    fig, axes = plt.subplots(MID_SIZE, 3, figsize=(12, 2.5 * MID_SIZE))

    for j in range(MID_SIZE):
        for i in range(3):
            ax = axes[j, i]
            xi_range, edge_out = edges[(j, i)]
            ax.plot(xi_range, edge_out, "b-", lw=2)
            ax.axhline(0, color="gray", ls=":", alpha=0.3)
            ax.set_title(f"$\\phi_{{{j},{i}}}$({VAR_NAMES[i]})", fontsize=10)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_xlabel(f"${VAR_NAMES[i]}$", fontsize=10)

    for i in range(3):
        axes[0, i].set_title(f"Input: ${VAR_NAMES[i]}$\n"
                              f"$\\phi_{{0,{i}}}$({VAR_NAMES[i]})", fontsize=10)

    fig.suptitle("Layer-1 Univariate Edge Functions (centered at mean)",
                 fontsize=14, y=1.002)
    fig.tight_layout()
    fig.savefig(OUTDIR / "layer1_edges.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTDIR / "layer1_edges.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved layer1_edges.png/pdf")


def plot_psi_manifolds(model, x_data, psi_data):
    """Plot 2: For each mid-node j, show:
      - Row of 3 univariate functions (already in plot_all_edges)
      - 3D scatter of attractor colored by psi_j value
      - Level-set bands at constant psi_j values
    """
    x_np = x_data.numpy()

    fig, axes = plt.subplots(MID_SIZE, 2, figsize=(14, 3.5 * MID_SIZE),
                              subplot_kw={"projection": "3d"})

    for j in range(MID_SIZE):
        psi_j = psi_data[:, j]
        psi_min, psi_max = psi_j.min(), psi_j.max()
        psi_range = psi_max - psi_min

        # Left: attractor colored by psi_j
        ax = axes[j, 0]
        sc = ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                        c=psi_j, cmap="coolwarm", s=0.5, alpha=0.3)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_zlabel("z", fontsize=8)
        ax.set_title(f"$\\psi_{j}$(x,y,z) on attractor", fontsize=10)
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=6)
        plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)

        # Right: level-set bands — color points by which band they're in
        n_bands = 5
        band_edges = np.linspace(psi_min, psi_max, n_bands + 1)
        ax2 = axes[j, 1]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_bands))

        for b in range(n_bands):
            mask = (psi_j >= band_edges[b]) & (psi_j < band_edges[b + 1])
            if mask.sum() > 0:
                ax2.scatter(x_np[mask, 0], x_np[mask, 1], x_np[mask, 2],
                           c=[colors[b]], s=0.5, alpha=0.4,
                           label=f"$\\psi_{j}$=[{band_edges[b]:.1f},{band_edges[b+1]:.1f})")

        ax2.set_xlabel("x", fontsize=8)
        ax2.set_ylabel("y", fontsize=8)
        ax2.set_zlabel("z", fontsize=8)
        ax2.set_title(f"$\\psi_{j}$ level-set bands", fontsize=10)
        ax2.view_init(elev=22, azim=-55)
        ax2.tick_params(labelsize=6)
        ax2.legend(fontsize=6, loc="upper right", markerscale=5)

    fig.suptitle("Layer-1 Additive Manifolds: $\\psi_j(x,y,z) = \\phi_{j,0}(x) + \\phi_{j,1}(y) + \\phi_{j,2}(z)$",
                 fontsize=14, y=1.001)
    fig.tight_layout()
    fig.savefig(OUTDIR / "psi_manifolds.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved psi_manifolds.png")


def plot_psi_summary(edges, psi_data, x_data):
    """Plot 3: Compact summary — one row per mid-node.
    4 columns: phi(x), phi(y), phi(z), psi histogram.
    """
    x_np = x_data.numpy()
    fig, axes = plt.subplots(MID_SIZE, 4, figsize=(16, 2.5 * MID_SIZE))

    for j in range(MID_SIZE):
        # Columns 0-2: univariate edges
        for i in range(3):
            ax = axes[j, i]
            xi_range, edge_out = edges[(j, i)]
            ax.plot(xi_range, edge_out, "b-", lw=2)
            ax.axhline(0, color="gray", ls=":", alpha=0.3)

            # Classify nonlinearity
            # Linear R²: fit a line and check residual
            from numpy.polynomial import polynomial as P
            c_lin = P.polyfit(xi_range, edge_out, 1)
            pred_lin = P.polyval(xi_range, c_lin)
            ss_res_lin = np.sum((edge_out - pred_lin) ** 2)
            ss_tot = np.sum((edge_out - edge_out.mean()) ** 2)
            r2_lin = 1 - ss_res_lin / max(ss_tot, 1e-12)

            # Edge range as fraction of total psi range
            edge_range = edge_out.max() - edge_out.min()
            psi_range = psi_data[:, j].max() - psi_data[:, j].min()
            contribution = edge_range / max(psi_range, 1e-12)

            color = "red" if r2_lin < 0.95 and contribution > 0.05 else "black"
            nl_tag = "NL" if r2_lin < 0.95 else "~lin"
            ax.set_title(f"$\\phi_{{{j},{i}}}$({VAR_NAMES[i]})  "
                          f"[{nl_tag}, {contribution:.0%}]",
                          fontsize=8, color=color)
            ax.tick_params(labelsize=6)

        # Column 3: histogram of psi_j values on attractor
        ax = axes[j, 3]
        ax.hist(psi_data[:, j], bins=50, color="steelblue", alpha=0.7, density=True)
        ax.set_title(f"$\\psi_{j}$ distribution", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle("Layer-1 Summary: Edge Functions & Nonlinearity",
                 fontsize=14, y=1.001)
    fig.tight_layout()
    fig.savefig(OUTDIR / "psi_summary.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTDIR / "psi_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved psi_summary.png/pdf")


def main():
    print("=" * 70)
    print("  LAYER-1 MANIFOLD VISUALIZATION")
    print("=" * 70)

    model, latent_dim = load_model()

    set_seed(42)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:48]
    x_flat = train_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_data = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    print(f"  Data: {x_data.shape}")

    # Extract edge functions and psi activations
    print("\nExtracting edge functions...")
    edges, psi_data = extract_edge_functions(model, x_data)

    # Print nonlinearity summary
    print(f"\n  {'Node':>6} {'Input':>6} {'Range':>8} {'LinR2':>7} {'Type':>6}")
    print("  " + "-" * 40)
    from numpy.polynomial import polynomial as P
    for j in range(MID_SIZE):
        for i in range(3):
            xi_range, edge_out = edges[(j, i)]
            c_lin = P.polyfit(xi_range, edge_out, 1)
            pred_lin = P.polyval(xi_range, c_lin)
            ss_res = np.sum((edge_out - pred_lin) ** 2)
            ss_tot = np.sum((edge_out - edge_out.mean()) ** 2)
            r2_lin = 1 - ss_res / max(ss_tot, 1e-12)
            edge_range = edge_out.max() - edge_out.min()
            nl = "NL" if r2_lin < 0.95 else "lin"
            print(f"  psi{j}  {VAR_NAMES[i]:>6}  {edge_range:>8.3f}  {r2_lin:>7.3f}  {nl:>6}")

    # Plot 1: All edges
    print("\nPlotting edge functions...")
    plot_all_edges(edges)

    # Plot 2: 3D manifolds
    print("Plotting 3D manifolds...")
    plot_psi_manifolds(model, x_data, psi_data)

    # Plot 3: Summary
    print("Plotting summary...")
    plot_psi_summary(edges, psi_data, x_data)

    print(f"\n  All outputs in {OUTDIR}/")


if __name__ == "__main__":
    main()
