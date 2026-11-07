"""
Full analysis of the best model: prune@0.03 + retrain.

Generates:
  1. eigenvalues.png/pdf — discrete + generator eigenvalue spectrum
  2. rollout.png/pdf — x,y,z time series vs truth
  3. nrmse.png/pdf — NRMSE vs Lyapunov time (linear + log)
  4. eigenmode_dictionary.png/pdf — how dictionary maps onto eigenmodes
  5. encoder_diagram.png/pdf — PyKAN network plot
  6. decoder_diagram.png/pdf — PyKAN network plot

Run from project root:
    python experiments/lorenz_5d_best_model_plots.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import json
from pathlib import Path
from numpy.polynomial import polynomial as P
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data import simulate_lorenz
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable

OUTDIR = Path("results/lorenz_5d_best_model")
OUTDIR.mkdir(parents=True, exist_ok=True)
DT = 0.01
MID = 5
LATENT = 5
LAMBDA_MAX = 0.9056
TAU_L = 1.0 / LAMBDA_MAX
POLY_NAMES = ["x", "y", "z", "xy", "xz"]


def load_model():
    ae = PyKANAutoEncoder(3, MID, LATENT, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, LATENT)
    ckpt = torch.load("results/lorenz_5d_third_test/model_pruned_0.03.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def ols_r2(X, y):
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res / max(ss_tot, 1e-12), coeffs


def main():
    print("=" * 70)
    print("  BEST MODEL PLOTS: prune@0.03 + retrain")
    print("=" * 70)

    model = load_model()
    enc = model.ae.encoder
    dec = model.ae.decoder

    set_seed(42)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    val_traj = traj[48:]
    train_traj = traj[:48]

    x_flat = train_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 10000, replace=False)
    x_data = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    x_np = x_data.numpy()

    # ══════════════════════════════════════════════════════════════
    # 1. EIGENVALUES
    # ══════════════════════════════════════════════════════════════
    print("\n  1. Eigenvalue spectrum...")

    A = model.generator_matrix().detach().cpu().numpy()
    M = model.propagator_matrix(DT).cpu().numpy()
    eigs_M = np.linalg.eigvals(M)
    eigs_A = np.linalg.eigvals(A)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.3, label='Unit circle')
    ax.scatter(eigs_M.real, eigs_M.imag, s=80, c='steelblue',
               edgecolors='black', linewidths=0.5, zorder=3)
    for lam in eigs_M:
        if abs(lam.imag) > 0.001:
            freq = np.angle(lam) / (2 * np.pi * DT)
            ax.annotate(f'{freq:+.1f} Hz', (lam.real, lam.imag),
                        fontsize=7, ha='left', xytext=(3, 3),
                        textcoords='offset points')
    ax.set_xlabel(r'Re($\lambda$)', fontsize=12)
    ax.set_ylabel(r'Im($\lambda$)', fontsize=12)
    ax.set_title(f'Discrete Eigenvalues $\\exp(A \\cdot dt)$\n'
                 f'$|\\lambda|$ range: [{np.abs(eigs_M).min():.4f}, '
                 f'{np.abs(eigs_M).max():.4f}]', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axvline(0, color='red', ls='--', lw=1, alpha=0.5, label='Re=0 (stability)')
    ax.scatter(eigs_A.real, eigs_A.imag, s=80, c='#F44336',
               edgecolors='black', linewidths=0.5, zorder=3)
    for mu in eigs_A:
        if abs(mu.imag) > 0.01:
            freq = mu.imag / (2 * np.pi)
            ax.annotate(f'{freq:+.1f} Hz', (mu.real, mu.imag),
                        fontsize=7, ha='left', xytext=(3, 3),
                        textcoords='offset points')
        else:
            ax.annotate(f'Re={mu.real:.1f}', (mu.real, mu.imag),
                        fontsize=7, ha='left', xytext=(3, 3),
                        textcoords='offset points')
    ax.set_xlabel(r'Re($\mu$)', fontsize=12)
    ax.set_ylabel(r'Im($\mu$)', fontsize=12)
    ax.set_title(f'Generator $A = K - L^\\top L$\n'
                 f'max Re($\\mu$) = {np.max(eigs_A.real):.4f}', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Stable Koopman Spectrum — Pruned [3,5,5] (Lorenz)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'eigenvalues.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'eigenvalues.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"     Saved eigenvalues.png/pdf")
    print(f"     |λ| range: [{np.abs(eigs_M).min():.4f}, {np.abs(eigs_M).max():.4f}]")

    # ══════════════════════════════════════════════════════════════
    # 2. ROLLOUT
    # ══════════════════════════════════════════════════════════════
    print("\n  2. Rollout...")

    n_tau = 10
    steps_per_tau = int(TAU_L / DT)
    n_steps = n_tau * steps_per_tau

    all_pred, all_true = [], []
    with torch.no_grad():
        for i in range(val_traj.shape[0]):
            x = val_traj[i, 0]
            xs = [x]
            for _ in range(n_steps):
                out = model.forward_one_step(x.unsqueeze(0), DT)
                x = out['x_tp1_pred'].squeeze(0)
                xs.append(x)
            all_pred.append(torch.stack(xs))
            all_true.append(val_traj[i, :n_steps + 1])

    preds = torch.stack(all_pred)
    trues = torch.stack(all_true)
    t_tau = np.arange(n_steps + 1) * DT / TAU_L

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    true_np = trues[0].numpy()
    pred_np = preds[0].detach().numpy()
    for i, (ax, lab, ct, cp) in enumerate(zip(
            axes, ['x', 'y', 'z'],
            ['#1f77b4', '#2ca02c', '#ff7f0e'],
            ['#d62728', '#9467bd', '#8c564b'])):
        ax.plot(t_tau, true_np[:, i], color=ct, lw=1, alpha=0.8, label=f'True ${lab}$')
        ax.plot(t_tau, pred_np[:, i], color=cp, lw=1, alpha=0.8, ls='--',
                label=f'Predicted ${lab}$')
        ax.set_ylabel(f'${lab}(t)$', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)
        for tm in range(1, n_tau + 1):
            ax.axvline(tm, color='gray', ls=':', alpha=0.3, lw=0.5)
    axes[-1].set_xlabel(r'Time [$\tau_\Lambda$]', fontsize=12)
    fig.suptitle(r'Pruned [3,5,5] Koopman Rollout ($\tau_\Lambda \approx 1.1$ s)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'rollout.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'rollout.pdf', bbox_inches='tight')
    plt.close(fig)
    print("     Saved rollout.png/pdf")

    # ══════════════════════════════════════════════════════════════
    # 3. NRMSE
    # ══════════════════════════════════════════════════════════════
    print("\n  3. NRMSE...")

    std_comp = trues.reshape(-1, 3).std(dim=0)
    rmse = (preds - trues).pow(2).mean(dim=0).sqrt()
    nrmse_per = rmse / std_comp.unsqueeze(0)
    nrmse = nrmse_per.mean(dim=1).numpy()
    nrmse_x = nrmse_per[:, 0].numpy()
    nrmse_y = nrmse_per[:, 1].numpy()
    nrmse_z = nrmse_per[:, 2].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(t_tau, nrmse, 'k-', lw=2, label='NRMSE (all)')
    ax.plot(t_tau, nrmse_x, '-', color='#1f77b4', lw=1, alpha=0.7, label='$x$')
    ax.plot(t_tau, nrmse_y, '-', color='#2ca02c', lw=1, alpha=0.7, label='$y$')
    ax.plot(t_tau, nrmse_z, '-', color='#ff7f0e', lw=1, alpha=0.7, label='$z$')
    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5, label='NRMSE = 1')
    ax.fill_between(t_tau, 0, nrmse, where=nrmse < 1.0, color='green', alpha=0.08)
    cross = np.where(nrmse > 1.0)[0]
    if len(cross) > 0:
        tc = t_tau[cross[0]]
        ax.annotate(f'NRMSE=1 at {tc:.1f} $\\tau_\\Lambda$',
                    xy=(tc, 1.0), xytext=(tc + 0.5, 1.3),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    ax.set_xlabel(r'Time [$\tau_\Lambda$]', fontsize=12)
    ax.set_ylabel('NRMSE', fontsize=12)
    ax.set_title('Normalized RMSE', fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim(0, n_tau)
    ax.set_ylim(0, min(2.0, nrmse.max() * 1.1))
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.semilogy(t_tau, nrmse, 'k-', lw=2)
    ax.semilogy(t_tau, nrmse_x, '-', color='#1f77b4', lw=1, alpha=0.7)
    ax.semilogy(t_tau, nrmse_y, '-', color='#2ca02c', lw=1, alpha=0.7)
    ax.semilogy(t_tau, nrmse_z, '-', color='#ff7f0e', lw=1, alpha=0.7)
    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel(r'Time [$\tau_\Lambda$]', fontsize=12)
    ax.set_ylabel('NRMSE (log)', fontsize=12)
    ax.set_title('NRMSE Growth', fontsize=12)
    ax.set_xlim(0, n_tau)
    ax.grid(True, alpha=0.2)

    horizon = t_tau[cross[0]] if len(cross) > 0 else float('inf')
    fig.suptitle(f'Pruned [3,5,5] Prediction Horizon: {horizon:.1f} $\\tau_\\Lambda$ '
                 f'({horizon * TAU_L:.1f} s)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'nrmse.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'nrmse.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"     Saved nrmse.png/pdf (horizon: {horizon:.1f} tau_L)")

    # ══════════════════════════════════════════════════════════════
    # 4. EIGENMODE-DICTIONARY INTERACTION
    # ══════════════════════════════════════════════════════════════
    print("\n  4. Eigenmode-dictionary interaction...")

    # Get psi activations
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

    eigvals_K, V = np.linalg.eig(K_psi)
    V_inv = np.linalg.inv(V)
    mu = np.log(eigvals_K.astype(complex)) / DT
    order = np.argsort(-np.abs(mu.imag))

    modal = psi @ V_inv.T
    x, y, z = x_np[:, 0], x_np[:, 1], x_np[:, 2]
    poly_data = np.column_stack([x, y, z, x * y, x * z])

    # Correlations
    corr_re = np.zeros((LATENT, len(POLY_NAMES)))
    corr_amp = np.zeros((LATENT, len(POLY_NAMES)))
    for rank in range(LATENT):
        for p in range(len(POLY_NAMES)):
            corr_re[rank, p] = np.corrcoef(modal[:, order[rank]].real,
                                            poly_data[:, p])[0, 1]
            corr_amp[rank, p] = np.corrcoef(np.abs(modal[:, order[rank]]),
                                             poly_data[:, p])[0, 1]

    # Regression
    modal_wc = np.column_stack([np.ones(len(psi)),
                                 *[modal[:, order[r]].real for r in range(LATENT)]])
    reg_r2 = {}
    for pi, pn in enumerate(POLY_NAMES):
        r2, _ = ols_r2(modal_wc, poly_data[:, pi])
        reg_r2[pn] = r2

    # Mode labels
    mode_labels = []
    for rank in range(LATENT):
        freq = mu[order[rank]].imag / (2 * np.pi)
        mode_labels.append(f"M{rank} ({freq:+.1f} Hz)")

    # Time series
    traj_seg = traj[0, :3000].numpy()
    x_seg = torch.tensor(traj_seg, dtype=torch.float32)
    with torch.no_grad():
        enc(x_seg)
        psi_seg = enc.acts[1].numpy()
    modal_seg = psi_seg @ V_inv.T
    t_seg = np.arange(len(traj_seg)) * DT

    # ── Plot ──
    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35,
                  height_ratios=[1.2, 1, 1, 1, 1])

    # Row 0: correlation heatmaps + regression R²
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(corr_re, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(POLY_NAMES)))
    ax.set_xticklabels(POLY_NAMES, fontsize=10)
    ax.set_yticks(range(LATENT))
    ax.set_yticklabels(mode_labels, fontsize=8)
    for i in range(LATENT):
        for j in range(len(POLY_NAMES)):
            v = corr_re[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Re(eigenmode) vs observable", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(corr_amp, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(POLY_NAMES)))
    ax.set_xticklabels(POLY_NAMES, fontsize=10)
    ax.set_yticks(range(LATENT))
    ax.set_yticklabels(mode_labels, fontsize=8)
    for i in range(LATENT):
        for j in range(len(POLY_NAMES)):
            v = corr_amp[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.5 else "black")
    ax.set_title("|eigenmode amplitude| vs observable", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[0, 2])
    r2_vals = [reg_r2[pn] for pn in POLY_NAMES]
    colors_bar = ["steelblue" if pn in {"x", "y", "z"} else "coral" for pn in POLY_NAMES]
    ax.bar(range(len(POLY_NAMES)), r2_vals, color=colors_bar, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(POLY_NAMES)))
    ax.set_xticklabels(POLY_NAMES, fontsize=10)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title("Observable = Σ eigenmodes (R²)", fontsize=10)
    ax.set_ylim(0.5, 1.05)
    for i, r2 in enumerate(r2_vals):
        ax.text(i, r2 + 0.01, f"{r2:.3f}", ha="center", fontsize=8)

    # Rows 1-2: attractor colored by eigenmodes
    for rank in range(LATENT):
        row = 1 if rank < 3 else 2
        col = rank if rank < 3 else rank - 3
        ax = fig.add_subplot(gs[row, col], projection="3d")
        c = modal[:, order[rank]].real
        vmax = np.percentile(np.abs(c), 95)
        ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                  c=c, cmap="coolwarm", s=0.3, alpha=0.3,
                  vmin=-vmax, vmax=vmax)
        freq = mu[order[rank]].imag / (2 * np.pi)
        ax.set_title(f"Mode {rank}: {freq:+.1f} Hz\nRe(μ)={mu[order[rank]].real:.1f}",
                    fontsize=9)
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=5)

    # Fill remaining with dictionary terms
    for di, (dn, dv) in enumerate([("xy", x * y), ("xz", x * z)]):
        ax = fig.add_subplot(gs[2, 2 + di] if 2 + di < 3 else gs[2, 2],
                              projection="3d")
        vmax = np.percentile(np.abs(dv), 95)
        ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                  c=dv, cmap="coolwarm", s=0.3, alpha=0.3,
                  vmin=-vmax, vmax=vmax)
        ax.set_title(f"Dictionary: {dn}", fontsize=9)
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=5)

    # Rows 3-4: time series
    for rank in range(LATENT):
        row = 3 if rank < 3 else 4
        col = rank if rank < 3 else rank - 3
        ax = fig.add_subplot(gs[row, col])
        mk = modal_seg[:, order[rank]]
        freq = mu[order[rank]].imag / (2 * np.pi)
        ax.plot(t_seg, mk.real, "b-", lw=0.6, alpha=0.8, label="Re")
        if abs(mu[order[rank]].imag) > 0.1:
            ax.plot(t_seg, np.abs(mk), "k--", lw=0.4, alpha=0.4, label="|z|")
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_title(f"M{rank} ({freq:+.1f} Hz)", fontsize=9)
        ax.set_xlim(0, 5)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    # Dictionary time series in remaining slots
    for di, dn in enumerate(["xy", "xz"]):
        if 2 + di < 3:
            ax = fig.add_subplot(gs[4, 2 + di])
        else:
            ax = fig.add_subplot(gs[4, 2])
        if dn == "xy":
            ax.plot(t_seg, traj_seg[:, 0] * traj_seg[:, 1], "coral", lw=0.6)
        else:
            ax.plot(t_seg, traj_seg[:, 0] * traj_seg[:, 2], "coral", lw=0.6)
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_title(f"{dn}(t)", fontsize=9)
        ax.set_xlim(0, 5)
        ax.tick_params(labelsize=6)

    fig.suptitle("Eigenmode–Dictionary Interaction (Pruned [3,5,5])\n"
                 "Eigenmodes vs dictionary $\\{x,y,z,xy,xz\\}$",
                 fontsize=13, y=1.01)
    fig.savefig(OUTDIR / 'eigenmode_dictionary.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'eigenmode_dictionary.pdf', bbox_inches='tight')
    plt.close(fig)
    print("     Saved eigenmode_dictionary.png/pdf")

    # ══════════════════════════════════════════════════════════════
    # 5-6. PyKAN diagrams
    # ══════════════════════════════════════════════════════════════
    print("\n  5. Encoder diagram...")
    try:
        enc.save_act = True
        with torch.no_grad():
            enc(x_data[:500])
        enc.plot()
        plt.savefig(OUTDIR / 'encoder_diagram.png', dpi=150, bbox_inches='tight')
        plt.savefig(OUTDIR / 'encoder_diagram.pdf', bbox_inches='tight')
        plt.close('all')
        print("     Saved encoder_diagram.png/pdf")
    except Exception as e:
        print(f"     Failed: {e}")

    print("  6. Decoder diagram...")
    try:
        dec.save_act = True
        with torch.no_grad():
            z_sample = enc(x_data[:500])
            dec(z_sample)
        dec.plot()
        plt.savefig(OUTDIR / 'decoder_diagram.png', dpi=150, bbox_inches='tight')
        plt.savefig(OUTDIR / 'decoder_diagram.pdf', bbox_inches='tight')
        plt.close('all')
        print("     Saved decoder_diagram.png/pdf")
    except Exception as e:
        print(f"     Failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # Save summary
    # ══════════════════════════════════════════════════════════════
    summary = {
        "model": "prune@0.03 + retrain",
        "architecture": [3, MID, LATENT],
        "eigenvalues_discrete": [
            {"real": float(l.real), "imag": float(l.imag), "abs": float(abs(l))}
            for l in eigs_M
        ],
        "eigenvalues_generator": [
            {"real": float(l.real), "imag": float(l.imag), "freq_hz": float(l.imag/(2*np.pi))}
            for l in eigs_A
        ],
        "prediction_horizon_tau": float(horizon),
        "koopman_psi_eigenvalues": [
            {"mode": rank, "freq_hz": float(mu[order[rank]].imag/(2*np.pi)),
             "decay": float(mu[order[rank]].real),
             "abs_lambda": float(abs(eigvals_K[order[rank]]))}
            for rank in range(LATENT)
        ],
        "regression_r2": {pn: float(v) for pn, v in reg_r2.items()},
    }
    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary.json")

    # Print eigenvalue table
    print(f"\n  Eigenvalue table:")
    print(f"  {'Mode':>5} {'|λ|':>8} {'Re(μ)':>8} {'freq Hz':>8}")
    for rank in range(LATENT):
        oidx = order[rank]
        print(f"  M{rank:>3} {abs(eigvals_K[oidx]):>8.4f} "
              f"{mu[oidx].real:>8.2f} {mu[oidx].imag/(2*np.pi):>8.3f}")

    print(f"\n  All plots saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
