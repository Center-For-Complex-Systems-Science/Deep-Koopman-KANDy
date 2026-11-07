"""
Lorenz prediction plots with Lyapunov-time NRMSE.

Generates:
    1. Multi-step rollout vs truth (x, y, z components, x-axis in Lyapunov times)
    2. 3D phase portrait overlay
    3. NRMSE vs time in Lyapunov times (with shaded divergence region)
    4. Multiple initial conditions for statistics
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from data import simulate_lorenz
from models import PureDeepKANAutoEncoder, DeepKoopmanCT
from training import set_seed, rollout

OUTDIR = Path('figs')

# Lorenz largest Lyapunov exponent (Sprott 2003)
LAMBDA_MAX = 0.9056  # /s
DT = 0.01
TAU_L = 1.0 / LAMBDA_MAX  # ≈ 1.104 s
STEPS_PER_TAU = int(TAU_L / DT)  # ≈ 110 steps


def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = torch.load(os.path.join(base, 'old', 'kan_pure_3_3_10_model.pt'),
                      map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    ae = PureDeepKANAutoEncoder(
        input_size=3, mid_size=cfg['mid_size'],
        bottleneck_size=cfg['latent_dim'],
        grid_size=5, spline_order=3,
        grid_range_l1=[-30, 50], grid_range_l2=[-10, 10],
    )
    model = DeepKoopmanCT(ae, latent_dim=cfg['latent_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def compute_nrmse(pred, true):
    """Normalized RMSE: RMSE / std(true) at each time step.

    pred, true: (n_traj, T, 3)
    Returns: (T,) NRMSE averaged over components and trajectories.
    """
    # Per-component std computed over all trajectories and time
    std_per_comp = true.reshape(-1, 3).std(dim=0)  # (3,)

    # Squared error at each (traj, time, comp)
    se = (pred - true) ** 2  # (n_traj, T, 3)

    # RMSE per (time, comp), averaged over trajectories
    rmse = se.mean(dim=0).sqrt()  # (T, 3)

    # Normalize by component std
    nrmse_per_comp = rmse / std_per_comp.unsqueeze(0)  # (T, 3)

    # Average over components
    return nrmse_per_comp.mean(dim=1).numpy()  # (T,)


def multi_rollout(model, traj, n_steps):
    """Rollout from multiple initial conditions.

    Returns:
        preds: (n_traj, n_steps+1, 3)
        trues: (n_traj, n_steps+1, 3)
    """
    n_traj = traj.shape[0]
    preds = []
    trues = []

    with torch.no_grad():
        for i in range(n_traj):
            x0 = traj[i, 0]
            xs = rollout(model, x0, DT, n_steps)
            preds.append(xs)
            trues.append(traj[i, :n_steps + 1])

    return torch.stack(preds), torch.stack(trues)


if __name__ == '__main__':
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    model = load_model()

    print("Generating Lorenz trajectories...")
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=500)
    val_traj = traj[48:]  # 12 validation trajectories

    # Rollout for 10 Lyapunov times
    n_tau = 10
    n_steps = n_tau * STEPS_PER_TAU
    print(f"Rolling out {val_traj.shape[0]} trajectories for {n_tau} Lyapunov times "
          f"({n_steps} steps)...")

    preds, trues = multi_rollout(model, val_traj, n_steps)
    t_tau = np.arange(n_steps + 1) * DT / TAU_L  # time in Lyapunov units

    # NRMSE
    nrmse = compute_nrmse(preds, trues)

    # Per-component NRMSE
    std_comp = trues.reshape(-1, 3).std(dim=0)
    rmse_comp = (preds - trues).pow(2).mean(dim=0).sqrt()  # (T, 3)
    nrmse_x = (rmse_comp[:, 0] / std_comp[0]).numpy()
    nrmse_y = (rmse_comp[:, 1] / std_comp[1]).numpy()
    nrmse_z = (rmse_comp[:, 2] / std_comp[2]).numpy()

    # ================================================================
    # Figure 1: Trajectory comparison (single trajectory)
    # ================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ['x', 'y', 'z']
    colors_true = ['#1f77b4', '#2ca02c', '#ff7f0e']
    colors_pred = ['#d62728', '#9467bd', '#8c564b']

    traj_idx = 0
    true_np = trues[traj_idx].numpy()
    pred_np = preds[traj_idx].detach().numpy()

    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(t_tau, true_np[:, i], color=colors_true[i], lw=1, alpha=0.8,
                label=f'True ${lab}$')
        ax.plot(t_tau, pred_np[:, i], color=colors_pred[i], lw=1, alpha=0.8,
                ls='--', label=f'Predicted ${lab}$')
        ax.set_ylabel(f'${lab}(t)$', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)

        # Shade Lyapunov time markers
        for tau_mark in range(1, n_tau + 1):
            ax.axvline(tau_mark, color='gray', ls=':', alpha=0.3, lw=0.5)

    axes[-1].set_xlabel(r'Time  [$\tau_\Lambda$  (Lyapunov times)]', fontsize=12)
    fig.suptitle(r'Koopman Rollout vs Truth  ($\tau_\Lambda = 1/\lambda_{max} \approx 1.1$ s)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'lorenz_rollout_lyapunov.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'lorenz_rollout_lyapunov.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved lorenz_rollout_lyapunov.png/pdf")

    # ================================================================
    # Figure 2: NRMSE vs Lyapunov time
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: linear scale
    ax = axes[0]
    ax.plot(t_tau, nrmse, 'k-', lw=2, label='NRMSE (all)')
    ax.plot(t_tau, nrmse_x, '-', color='#1f77b4', lw=1, alpha=0.7, label='$x$')
    ax.plot(t_tau, nrmse_y, '-', color='#2ca02c', lw=1, alpha=0.7, label='$y$')
    ax.plot(t_tau, nrmse_z, '-', color='#ff7f0e', lw=1, alpha=0.7, label='$z$')
    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5, label='NRMSE = 1 (climatology)')
    ax.fill_between(t_tau, 0, nrmse, where=nrmse < 1.0,
                     color='green', alpha=0.08, label='Skillful region')
    ax.fill_between(t_tau, 0, np.maximum(nrmse, 1.0), where=nrmse >= 1.0,
                     color='red', alpha=0.08)
    ax.set_xlabel(r'Time  [$\tau_\Lambda$]', fontsize=12)
    ax.set_ylabel('NRMSE', fontsize=12)
    ax.set_title('Normalized RMSE vs Lyapunov Time', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(0, n_tau)
    ax.set_ylim(0, min(2.0, nrmse.max() * 1.1))
    ax.grid(True, alpha=0.2)

    # Mark where NRMSE crosses 1
    cross_idx = np.where(nrmse > 1.0)[0]
    if len(cross_idx) > 0:
        t_cross = t_tau[cross_idx[0]]
        ax.annotate(f'NRMSE=1 at {t_cross:.1f} $\\tau_\\Lambda$',
                    xy=(t_cross, 1.0), xytext=(t_cross + 0.5, 1.3),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')

    # Right: log scale
    ax = axes[1]
    ax.semilogy(t_tau, nrmse, 'k-', lw=2, label='NRMSE (all)')
    ax.semilogy(t_tau, nrmse_x, '-', color='#1f77b4', lw=1, alpha=0.7, label='$x$')
    ax.semilogy(t_tau, nrmse_y, '-', color='#2ca02c', lw=1, alpha=0.7, label='$y$')
    ax.semilogy(t_tau, nrmse_z, '-', color='#ff7f0e', lw=1, alpha=0.7, label='$z$')
    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5)

    # Fit exponential growth in early phase (NRMSE < 0.5)
    early = nrmse < 0.5
    if early.sum() > 10:
        t_early = t_tau[early]
        n_early = nrmse[early]
        valid = n_early > 1e-4
        if valid.sum() > 5:
            log_n = np.log(n_early[valid])
            t_e = t_early[valid]
            slope, intercept = np.polyfit(t_e, log_n, 1)
            ax.semilogy(t_tau[:len(t_early)],
                        np.exp(intercept + slope * t_tau[:len(t_early)]),
                        'r:', lw=1.5, alpha=0.7,
                        label=f'Exp fit: $e^{{{slope:.2f} t/\\tau_\\Lambda}}$')

    ax.set_xlabel(r'Time  [$\tau_\Lambda$]', fontsize=12)
    ax.set_ylabel('NRMSE (log)', fontsize=12)
    ax.set_title('NRMSE Growth (log scale)', fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim(0, n_tau)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f'Prediction Skill  '
                 r'($\lambda_{max}$' f' = {LAMBDA_MAX:.4f}/s, '
                 r'$\tau_\Lambda$' f' = {TAU_L:.2f} s, '
                 f'{val_traj.shape[0]} trajectories)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'lorenz_nrmse_lyapunov.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'lorenz_nrmse_lyapunov.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved lorenz_nrmse_lyapunov.png/pdf")

    # ================================================================
    # Figure 3: 3D phase portrait
    # ================================================================
    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121, projection='3d')
    ax.plot(true_np[:, 0], true_np[:, 1], true_np[:, 2],
            'b-', lw=0.5, alpha=0.6, label='True')
    ax.plot(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2],
            'r-', lw=0.5, alpha=0.6, label='Predicted')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_zlabel('$z$')
    ax.set_title('Full Rollout')
    ax.legend(fontsize=8)

    # Zoomed: first 2 Lyapunov times (where prediction is good)
    n_zoom = 2 * STEPS_PER_TAU
    ax = fig.add_subplot(122, projection='3d')
    ax.plot(true_np[:n_zoom, 0], true_np[:n_zoom, 1], true_np[:n_zoom, 2],
            'b-', lw=1, alpha=0.8, label='True')
    ax.plot(pred_np[:n_zoom, 0], pred_np[:n_zoom, 1], pred_np[:n_zoom, 2],
            'r--', lw=1, alpha=0.8, label='Predicted')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_zlabel('$z$')
    ax.set_title(r'First 2 $\tau_\Lambda$')
    ax.legend(fontsize=8)

    fig.suptitle('Lorenz Attractor: True vs Koopman Rollout', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'lorenz_phase_portrait.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'lorenz_phase_portrait.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved lorenz_phase_portrait.png/pdf")

    # Summary
    cross_tau = t_tau[cross_idx[0]] if len(cross_idx) > 0 else float('inf')
    print(f"\n  Prediction horizon (NRMSE < 1): {cross_tau:.2f} Lyapunov times")
    print(f"  = {cross_tau * TAU_L:.2f} seconds")
    print(f"  = {int(cross_tau * STEPS_PER_TAU)} steps")
