"""
Vortex shedding (single cylinder) with stable Koopman KAN model.

Applies the POD + stable Deep Koopman pipeline to 2D Navier-Stokes vorticity
fields from laminar vortex shedding behind a single cylinder at Re=100.
The high-dimensional vorticity field (200x100 = 20,000 grid points) is first
reduced via POD (proper orthogonal decomposition / SVD) to r=32 modes.
A KANAutoEncoder then lifts the r-dimensional POD coordinates to a 64-dim
latent space where the dynamics are approximately linear under the stable
Koopman propagator A = K - L^T L (guaranteeing |eigenvalues| <= 1).

The vortex shedding at Re=100 has Strouhal number St ~ 0.17, giving a
shedding period T_s ~ D/(St*U) ~ 5.9 time units.

Pipeline:
    1. Simulate 2D wake -> vorticity snapshots omega(x,y,t)
    2. POD: SVD of snapshot matrix -> keep top r modes -> POD coords a(t)
    3. KANAutoEncoder(r, mid, latent) + DeepKoopmanCTStable(latent) on a(t)
    4. Rollout: predict POD coords, reconstruct omega = omega_mean + U_r @ a_pred

The vortex shedding period at Re=100 is T ~ 5 time units (St ~ 0.2).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models import KANAutoEncoder
from data import PairDataset
from training import set_seed

from two_cylinder_wake import TwoCylinderWakeSolver, Params

# =====================================================================
# Configuration
# =====================================================================

OUTDIR = Path('results/wake_stable')

# Simulation parameters
NX = 200
NY = 100
DT_SIM = 0.004       # solver timestep
T_END = 400.0         # total simulation time (longer for more training data)
RE = 300.0            # Reynolds number (past limit cycle → richer dynamics)
DT_SAVE = 0.2         # snapshot save interval
T_START = 40.0        # discard initial transient

# POD
R_POD = 32            # number of POD modes to retain

# Windowing for trajectory-style training
WINDOW_LEN = 75       # steps per window (= 15 time units, ~3 shedding periods)

# Model architecture
INPUT_SIZE = R_POD    # 32 POD modes
MID_SIZE = 32
LATENT_SIZE = 64      # 2x lifting

# Training
EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
GRAD_CLIP = 1.0

# Loss weights
GAMMA_PREDX = 1.0
ALPHA_LATENT = 1.0
BETA_RECON = 0.5

# Vortex shedding period estimate (Strouhal ~ 0.2 for Re=100 cylinder wake)
T_SHED = 5.0


# =====================================================================
# Stable Deep Koopman model (A = K - L^T L parameterization)
# =====================================================================

class DeepKoopmanCTStable(nn.Module):
    """Continuous-time Deep Koopman with stable generator.

    A = K - L^T L  where:
        K = -K^T              (skew-symmetric, drives rotation/oscillation)
        L^T L >= 0            (PSD, drives damping/contraction)

    => Re(eigenvalues of A) <= 0  => |eigenvalues of expm(A*dt)| <= 1

    When L -> 0, recovers the unitary (area-preserving) case.
    Params: n(n-1)/2 (skew) + n(n+1)/2 (lower-tri L) = n^2.
    """

    def __init__(self, autoencoder, latent_dim):
        super().__init__()
        self.ae = autoencoder
        self.latent_dim = latent_dim
        n = latent_dim

        # Skew-symmetric part K (rotation): upper triangle
        n_skew = n * (n - 1) // 2
        self.skew_params = nn.Parameter(0.01 * torch.randn(n_skew))

        # Lower triangular L (damping): L^T L is PSD
        self.L_diag = nn.Parameter(0.01 * torch.randn(n))
        n_lower = n * (n - 1) // 2
        self.L_lower = nn.Parameter(0.01 * torch.randn(n_lower))

    def _get_skew(self):
        n = self.latent_dim
        K = torch.zeros(n, n, device=self.skew_params.device)
        idx = torch.triu_indices(n, n, offset=1)
        K[idx[0], idx[1]] = self.skew_params
        return K - K.T

    def _get_L(self):
        n = self.latent_dim
        L = torch.zeros(n, n, device=self.L_diag.device)
        L[range(n), range(n)] = self.L_diag
        idx = torch.tril_indices(n, n, offset=-1)
        L[idx[0], idx[1]] = self.L_lower
        return L

    def generator_matrix(self):
        K = self._get_skew()
        L = self._get_L()
        return K - L.T @ L

    def damping_matrix(self):
        """L^T L -- the PSD damping component."""
        L = self._get_L()
        return (L.T @ L).detach()

    def expm_propagate(self, z_t, dt):
        A = self.generator_matrix()
        M = torch.matrix_exp(A * dt)
        return z_t @ M.T

    def propagator_matrix(self, dt):
        A = self.generator_matrix()
        return torch.matrix_exp(A * dt).detach()

    def forward_one_step(self, x_t, dt):
        z_t, x_t_rec = self.ae(x_t)
        z_tp1_pred = self.expm_propagate(z_t, dt)
        x_tp1_pred = self.ae.decoder(z_tp1_pred)
        return {
            "z_t": z_t,
            "x_t_rec": x_t_rec,
            "z_tp1_pred": z_tp1_pred,
            "x_tp1_pred": x_tp1_pred,
        }


# =====================================================================
# Simulation + POD
# =====================================================================

def run_simulation_and_pod(data_path):
    """Run wake solver, compute POD, cache everything to disk."""
    print("--- Running 2D wake simulation ---", flush=True)
    p = Params(nx=NX, ny=NY, dt=DT_SIM, t_end=T_END, re=RE, ygap=0.0,
               poisson_iters=100)
    solver = TwoCylinderWakeSolver(p)

    n_sim_steps = int(math.ceil(T_END / DT_SIM))
    save_interval = max(1, int(round(DT_SAVE / DT_SIM)))
    save_after_step = int(round(T_START / DT_SIM))

    # Collect snapshots
    snapshots = []
    times = []

    for step_i in range(n_sim_steps):
        solver.step()

        if step_i >= save_after_step and (step_i - save_after_step) % save_interval == 0:
            omega = solver.vorticity().copy()  # (ny, nx)
            snapshots.append(omega.ravel())     # flatten to (ny*nx,)
            times.append(solver.time)

        if (step_i + 1) % max(1, n_sim_steps // 20) == 0:
            print(f"  t = {solver.time:.1f} / {T_END:.1f}  "
                  f"({len(snapshots)} snapshots saved)", flush=True)

    snapshots = np.array(snapshots, dtype=np.float64)  # (n_snap, ny*nx)
    times = np.array(times)
    n_snap = snapshots.shape[0]
    print(f"  Simulation complete: {n_snap} snapshots, shape {snapshots.shape}",
          flush=True)

    # --- POD ---
    print("--- Computing POD ---", flush=True)
    # Snapshot matrix: (n_points, n_snapshots)
    omega_matrix = snapshots.T  # (ny*nx, n_snap)
    omega_mean = omega_matrix.mean(axis=1, keepdims=True)  # (ny*nx, 1)
    omega_centered = omega_matrix - omega_mean

    # Economy SVD
    U, S, Vt = np.linalg.svd(omega_centered, full_matrices=False)

    # Keep top r modes
    r = R_POD
    U_r = U[:, :r]                           # (ny*nx, r)  -- spatial modes
    a_pod = (U_r.T @ omega_centered).T       # (n_snap, r)  -- POD coordinates

    # Energy captured
    energy = np.cumsum(S**2) / np.sum(S**2)
    print(f"  POD: {r} modes capture {energy[r-1]*100:.1f}% of energy", flush=True)
    print(f"  Singular values (top 10): {S[:10]}", flush=True)
    print(f"  POD coord range: [{a_pod.min():.2f}, {a_pod.max():.2f}]", flush=True)

    # Save everything
    np.savez_compressed(
        data_path,
        snapshots=snapshots.astype(np.float32),
        x=solver.x,
        y=solver.y,
        times=times,
        omega_mean=omega_mean.ravel().astype(np.float32),
        U_pod=U_r.astype(np.float32),
        S_pod=S.astype(np.float32),
        a_pod=a_pod.astype(np.float32),
        # Cylinder geometry for plotting
        c1=np.array(solver.c1),
        c2=np.array(solver.c2),
        r_cyl=np.array([solver.r]),
        nx=np.array([NX]),
        ny=np.array([NY]),
    )
    print(f"  Saved data to {data_path}", flush=True)

    return {
        'snapshots': snapshots.astype(np.float32),
        'x': solver.x, 'y': solver.y,
        'times': times,
        'omega_mean': omega_mean.ravel().astype(np.float32),
        'U_pod': U_r.astype(np.float32),
        'S_pod': S.astype(np.float32),
        'a_pod': a_pod.astype(np.float32),
        'c1': np.array(solver.c1),
        'c2': np.array(solver.c2),
        'r_cyl': solver.r,
    }


def load_data(data_path):
    """Load cached simulation + POD data."""
    d = np.load(data_path)
    return {
        'snapshots': d['snapshots'],
        'x': d['x'], 'y': d['y'],
        'times': d['times'],
        'omega_mean': d['omega_mean'],
        'U_pod': d['U_pod'],
        'S_pod': d['S_pod'],
        'a_pod': d['a_pod'],
        'c1': d['c1'],
        'c2': d['c2'],
        'r_cyl': float(d['r_cyl'][0]) if d['r_cyl'].ndim > 0 else float(d['r_cyl']),
    }


# =====================================================================
# Training
# =====================================================================

def prepare_trajectories(a_pod):
    """Window the POD time series with 50% overlap for more training data.

    Returns:
        traj: (n_windows, window_len, r) float32 tensor
    """
    n_snap, r = a_pod.shape
    stride = WINDOW_LEN // 2  # 50% overlap
    windows = []
    for start in range(0, n_snap - WINDOW_LEN + 1, stride):
        windows.append(a_pod[start:start + WINDOW_LEN])
    traj = torch.tensor(np.array(windows), dtype=torch.float32)
    print(f"  Windowed data: {traj.shape} "
          f"({traj.shape[0]} windows x {WINDOW_LEN} steps x {r} POD modes, "
          f"stride={stride})", flush=True)
    return traj


def train_model(ae, traj, epochs=EPOCHS, lr=LR, device='cpu'):
    """Train the stable Koopman model on POD coordinate trajectories."""
    n_windows = traj.shape[0]
    n_train = max(1, n_windows - 3)  # keep 3 windows for validation
    n_val = n_windows - n_train

    print(f"  Train/val split: {n_train}/{n_val} windows", flush=True)

    train_dl = DataLoader(PairDataset(traj[:n_train].to(device)),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(PairDataset(traj[n_train:].to(device)),
                        batch_size=BATCH_SIZE)

    model = DeepKoopmanCTStable(ae, LATENT_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_skew = LATENT_SIZE * (LATENT_SIZE - 1) // 2
    n_damp = LATENT_SIZE + LATENT_SIZE * (LATENT_SIZE - 1) // 2
    print(f"  Stable Koopman [{INPUT_SIZE},{MID_SIZE},{LATENT_SIZE}]: "
          f"{n_params:,} params ({n_skew} skew + {n_damp} damping = "
          f"{n_skew + n_damp} generator)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    history = {
        'train_total': [], 'train_predx': [], 'train_latent': [], 'train_recon': [],
        'val_total': [], 'val_predx': [], 'val_latent': [], 'val_recon': [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        losses_t, losses_p, losses_l, losses_r = [], [], [], []

        for x_t, x_tp1 in train_dl:
            out = model.forward_one_step(x_t, DT_SAVE)
            z_true = model.ae.encoder(x_tp1)

            loss_predx = mse(out['x_tp1_pred'], x_tp1)
            loss_latent = mse(out['z_tp1_pred'], z_true)
            loss_recon = mse(out['x_t_rec'], x_t)
            loss = (GAMMA_PREDX * loss_predx
                    + ALPHA_LATENT * loss_latent
                    + BETA_RECON * loss_recon)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            losses_t.append(loss.item())
            losses_p.append(loss_predx.item())
            losses_l.append(loss_latent.item())
            losses_r.append(loss_recon.item())

        scheduler.step()

        # Validation
        model.eval()
        vloss_t, vloss_p, vloss_l, vloss_r = [], [], [], []
        with torch.no_grad():
            for x_t, x_tp1 in val_dl:
                out = model.forward_one_step(x_t, DT_SAVE)
                z_true = model.ae.encoder(x_tp1)

                loss_predx = mse(out['x_tp1_pred'], x_tp1)
                loss_latent = mse(out['z_tp1_pred'], z_true)
                loss_recon = mse(out['x_t_rec'], x_t)
                loss = (GAMMA_PREDX * loss_predx
                        + ALPHA_LATENT * loss_latent
                        + BETA_RECON * loss_recon)

                vloss_t.append(loss.item())
                vloss_p.append(loss_predx.item())
                vloss_l.append(loss_latent.item())
                vloss_r.append(loss_recon.item())

        history['train_total'].append(float(np.mean(losses_t)))
        history['train_predx'].append(float(np.mean(losses_p)))
        history['train_latent'].append(float(np.mean(losses_l)))
        history['train_recon'].append(float(np.mean(losses_r)))
        history['val_total'].append(float(np.mean(vloss_t)))
        history['val_predx'].append(float(np.mean(vloss_p)))
        history['val_latent'].append(float(np.mean(vloss_l)))
        history['val_recon'].append(float(np.mean(vloss_r)))

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            M = model.propagator_matrix(DT_SAVE).cpu().numpy()
            eigs = np.linalg.eigvals(M)
            mag_range = f'[{np.abs(eigs).min():.4f}, {np.abs(eigs).max():.4f}]'
            print(f"  Ep {ep:3d} | train {np.mean(losses_t):.6f} "
                  f"(pred {np.mean(losses_p):.6f}, lat {np.mean(losses_l):.6f}, "
                  f"rec {np.mean(losses_r):.6f}) | val {np.mean(vloss_t):.6f} | "
                  f"|lam| {mag_range}", flush=True)

    return model.cpu().eval(), history


# =====================================================================
# Autoregressive rollout (in POD space)
# =====================================================================

def rollout_pod(model, a0, n_steps, dt):
    """Autoregressive Koopman rollout in POD coordinate space.

    Args:
        model: DeepKoopmanCTStable (on CPU)
        a0: (r,) initial POD coordinate vector
        n_steps: number of forward steps
        dt: time between snapshots

    Returns:
        (n_steps+1, r) tensor of predicted POD coordinates
    """
    with torch.no_grad():
        # Encode ONCE, then propagate purely in latent space
        z = model.ae.encoder(a0.unsqueeze(0))
        preds = [model.ae.decoder(z).squeeze(0)]
        for _ in range(n_steps):
            z = model.expm_propagate(z, dt)
            preds.append(model.ae.decoder(z).squeeze(0))
    return torch.stack(preds)


def reconstruct_vorticity(a_pod_pred, U_pod, omega_mean,
                          a_pod_mean=None, a_pod_std=None):
    """Reconstruct vorticity from POD coordinates.

    Args:
        a_pod_pred: (n_steps, r) predicted POD coordinates (numpy)
                    If normalized, pass a_pod_mean/a_pod_std to denormalize.
        U_pod: (n_points, r) POD spatial modes (numpy)
        omega_mean: (n_points,) mean vorticity (numpy)
        a_pod_mean: (r,) normalization mean (optional)
        a_pod_std: (r,) normalization std (optional)

    Returns:
        (n_steps, n_points) reconstructed vorticity
    """
    a = a_pod_pred
    if a_pod_mean is not None and a_pod_std is not None:
        a = a * a_pod_std[None, :] + a_pod_mean[None, :]
    return omega_mean[None, :] + a @ U_pod.T


def compute_nrmse_pod(pred, true):
    """Compute NRMSE of POD coefficients at each timestep.

    Args:
        pred: (T, r) predicted POD coefficients (tensor or numpy)
        true: (T, r) true POD coefficients (tensor or numpy)

    Returns:
        nrmse: (T,) NRMSE per timestep (numpy)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(true, torch.Tensor):
        true = true.numpy()
    std_all = true.ravel().std()
    if std_all < 1e-12:
        std_all = 1.0
    rmse = np.sqrt(np.mean((pred - true)**2, axis=1))
    return rmse / std_all


def compute_nrmse_field(pred_omega, true_omega):
    """Compute NRMSE of full vorticity fields at each timestep.

    Args:
        pred_omega: (T, n_points)
        true_omega: (T, n_points)

    Returns:
        nrmse: (T,) NRMSE per timestep (numpy)
    """
    std_all = true_omega.ravel().std()
    if std_all < 1e-12:
        std_all = 1.0
    rmse = np.sqrt(np.mean((pred_omega - true_omega)**2, axis=1))
    return rmse / std_all


# =====================================================================
# Helper: add cylinder patches
# =====================================================================

def add_cylinders(ax, data):
    """Draw cylinder circles on an axis."""
    c1, c2, r_cyl = data['c1'], data['c2'], data['r_cyl']
    for cx, cy in [c1, c2]:
        patch = mpatches.Circle(
            (cx, cy), r_cyl,
            facecolor='0.65', edgecolor='black', linewidth=1.2, zorder=5,
        )
        ax.add_patch(patch)


# =====================================================================
# Plot 1: Vorticity comparison (hero figure)
# =====================================================================

def plot_vorticity_comparison(model, data):
    """6-panel figure: true (top) vs predicted (bottom) vorticity at 6 times."""
    print("1. Vorticity comparison...", flush=True)

    x, y = data['x'], data['y']
    nx, ny = len(x), len(y)
    a_pod = data['a_pod']
    U_pod = data['U_pod']
    omega_mean = data['omega_mean']
    times = data['times']
    snapshots = data['snapshots']

    # Target times for comparison
    target_times = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    # Find the validation window start. Use a starting snapshot that is
    # within the validation portion of the data.
    n_windows = a_pod.shape[0] // WINDOW_LEN
    n_train_windows = max(1, n_windows - 3)
    val_start_idx = n_train_windows * WINDOW_LEN

    # Find the snapshot index closest to our first target time
    # that gives us enough runway for all target times
    t0_target = target_times[0]
    t_last_target = target_times[-1]

    # Start rollout from val_start
    a0 = torch.tensor(a_pod[val_start_idx], dtype=torch.float32)
    t0_actual = times[val_start_idx]
    n_rollout_needed = int(math.ceil((t_last_target - t0_actual) / DT_SAVE)) + 10

    # If validation window starts after our target times, use a different approach:
    # find the closest snapshot to 50.0 and roll out from there
    if t0_actual > t0_target:
        # Find snapshot closest to target_times[0] - a bit of margin
        start_t = t0_target - 5.0
        start_idx = np.argmin(np.abs(times - start_t))
        a0 = torch.tensor(a_pod[start_idx], dtype=torch.float32)
        t0_actual = times[start_idx]
        n_rollout_needed = int(math.ceil((t_last_target - t0_actual) / DT_SAVE)) + 10

    n_rollout_needed = min(n_rollout_needed, a_pod.shape[0] - 1)

    # Koopman rollout
    a_pred = rollout_pod(model, a0, n_rollout_needed, DT_SAVE).numpy()

    # Collect panels
    fig, axes = plt.subplots(2, 6, figsize=(30, 8))

    extent = [x[0], x[-1], y[0], y[-1]]
    vmin, vmax = -6.0, 6.0

    for col, t_target in enumerate(target_times):
        # True snapshot
        snap_idx = np.argmin(np.abs(times - t_target))
        omega_true = snapshots[snap_idx].reshape(ny, nx)

        # Predicted: find the rollout step corresponding to this time
        rollout_step = int(round((times[snap_idx] - t0_actual) / DT_SAVE))
        rollout_step = max(0, min(rollout_step, a_pred.shape[0] - 1))
        omega_pred_flat = reconstruct_vorticity(
            a_pred[rollout_step:rollout_step+1], U_pod, omega_mean,
            data.get('a_pod_mean'), data.get('a_pod_std'),
        )[0]
        omega_pred = omega_pred_flat.reshape(ny, nx)

        # Top row: true
        ax = axes[0, col]
        ax.imshow(
            np.clip(omega_true, vmin, vmax),
            origin='lower', extent=extent, cmap='RdBu_r',
            vmin=vmin, vmax=vmax, interpolation='bilinear', aspect='auto',
        )
        add_cylinders(ax, data)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'$t = {t_target:.0f}$', fontsize=14)
        if col == 0:
            ax.set_ylabel('True', fontsize=14, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Bottom row: predicted
        ax = axes[1, col]
        ax.imshow(
            np.clip(omega_pred, vmin, vmax),
            origin='lower', extent=extent, cmap='RdBu_r',
            vmin=vmin, vmax=vmax, interpolation='bilinear', aspect='auto',
        )
        add_cylinders(ax, data)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel('Predicted', fontsize=14, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle('Vorticity: True vs Koopman Rollout', fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'vorticity_comparison.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'vorticity_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved vorticity_comparison.png/pdf", flush=True)


# =====================================================================
# Plot 2: POD energy spectrum
# =====================================================================

def plot_pod_spectrum(data):
    """Singular values and cumulative energy."""
    print("2. POD energy spectrum...", flush=True)

    S = data['S_pod']
    energy = np.cumsum(S**2) / np.sum(S**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: singular values (log scale)
    ax = axes[0]
    ax.semilogy(np.arange(1, len(S) + 1), S, 'ko-', markersize=4, lw=1.5)
    ax.axvline(R_POD, color='red', ls='--', lw=1, alpha=0.7,
               label=f'$r = {R_POD}$')
    ax.set_xlabel('Mode index $k$', fontsize=12)
    ax.set_ylabel('Singular value $\\sigma_k$', fontsize=12)
    ax.set_title('POD Singular Values', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, min(100, len(S)))

    # Right: cumulative energy
    ax = axes[1]
    ax.plot(np.arange(1, len(energy) + 1), energy * 100, 'b-', lw=2)
    ax.axhline(99.0, color='green', ls='--', lw=1, alpha=0.7, label='99%')
    ax.axvline(R_POD, color='red', ls='--', lw=1, alpha=0.7,
               label=f'$r = {R_POD}$ ({energy[R_POD-1]*100:.1f}%)')
    # Find where 99% energy is
    idx_99 = np.searchsorted(energy, 0.99)
    ax.plot(idx_99 + 1, 99.0, 'go', markersize=8, zorder=5)
    ax.annotate(f'{idx_99 + 1} modes for 99%',
                xy=(idx_99 + 1, 99.0), xytext=(idx_99 + 10, 96),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    ax.set_xlabel('Number of modes', fontsize=12)
    ax.set_ylabel('Cumulative energy (%)', fontsize=12)
    ax.set_title('POD Energy Fraction', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, min(100, len(energy)))
    ax.set_ylim(50, 100.5)

    fig.tight_layout()
    fig.savefig(OUTDIR / 'pod_spectrum.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'pod_spectrum.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved pod_spectrum.png/pdf "
          f"(r={R_POD} captures {energy[R_POD-1]*100:.1f}%)", flush=True)


# =====================================================================
# Plot 3: POD spatial modes
# =====================================================================

def plot_pod_modes(data):
    """Show the first 6 spatial POD modes as vorticity-like images."""
    print("3. POD spatial modes...", flush=True)

    x, y = data['x'], data['y']
    nx, ny = len(x), len(y)
    U_pod = data['U_pod']
    extent = [x[0], x[-1], y[0], y[-1]]

    n_show = min(6, U_pod.shape[1])
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes_flat = axes.flatten()

    for k in range(n_show):
        ax = axes_flat[k]
        mode_k = U_pod[:, k].reshape(ny, nx)
        vmax_k = np.abs(mode_k).max()
        ax.imshow(
            mode_k, origin='lower', extent=extent,
            cmap='RdBu_r', vmin=-vmax_k, vmax=vmax_k,
            interpolation='bilinear', aspect='auto',
        )
        add_cylinders(ax, data)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'POD Mode {k+1}', fontsize=12)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle('First 6 POD Spatial Modes', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'pod_modes.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'pod_modes.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved pod_modes.png/pdf", flush=True)


# =====================================================================
# Plot 4: Centerline space-time
# =====================================================================

def plot_centerline_spacetime(model, data):
    """Space-time heatmaps of omega(x, y=0, t) for true and predicted."""
    print("4. Centerline space-time...", flush=True)

    x, y = data['x'], data['y']
    nx, ny = len(x), len(y)
    a_pod = data['a_pod']
    U_pod = data['U_pod']
    omega_mean = data['omega_mean']
    times = data['times']
    snapshots = data['snapshots']

    # Centerline: find y-index closest to y=0
    j_center = np.argmin(np.abs(y))
    print(f"  Centerline at y[{j_center}] = {y[j_center]:.3f}", flush=True)

    # Use a chunk of the time series for space-time plot
    # Start from a reasonable time, roll out for ~50 time units
    n_rollout = min(250, a_pod.shape[0] - 1)  # 250 steps = 50 time units
    start_idx = 0  # start from beginning of saved data
    a0 = torch.tensor(a_pod[start_idx], dtype=torch.float32)

    a_pred = rollout_pod(model, a0, n_rollout, DT_SAVE).numpy()

    # True centerline: omega(x, y=0, t)
    true_center = np.zeros((n_rollout + 1, nx))
    pred_center = np.zeros((n_rollout + 1, nx))

    for step_i in range(n_rollout + 1):
        snap_idx = start_idx + step_i
        if snap_idx < snapshots.shape[0]:
            omega_true = snapshots[snap_idx].reshape(ny, nx)
            true_center[step_i, :] = omega_true[j_center, :]

        omega_pred_flat = reconstruct_vorticity(
            a_pred[step_i:step_i+1], U_pod, omega_mean,
            data.get('a_pod_mean'), data.get('a_pod_std'),
        )[0]
        omega_pred = omega_pred_flat.reshape(ny, nx)
        pred_center[step_i, :] = omega_pred[j_center, :]

    t_axis = times[start_idx:start_idx + n_rollout + 1]
    if len(t_axis) < n_rollout + 1:
        t_axis = np.arange(n_rollout + 1) * DT_SAVE + times[start_idx]

    vmax = max(np.abs(true_center).max(), np.abs(pred_center).max())
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    titles = ['True $\\omega(x, y{=}0, t)$', 'Predicted $\\omega(x, y{=}0, t)$']
    data_arrays = [true_center, pred_center]

    for ax, title, arr in zip(axes, titles, data_arrays):
        im = ax.pcolormesh(
            x, t_axis, arr,
            cmap='RdBu_r', vmin=vmin, vmax=vmax,
            shading='auto', rasterized=True,
        )
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, shrink=0.8, label='$\\omega$')

    axes[0].set_ylabel('Time $t$', fontsize=12)

    fig.suptitle('Centerline Vorticity: Space-Time Diagram', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'centerline_spacetime.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'centerline_spacetime.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved centerline_spacetime.png/pdf", flush=True)


# =====================================================================
# Plot 5: POD coefficient time series
# =====================================================================

def plot_pod_timeseries(model, data):
    """Time series of top 4 POD coefficients: true vs predicted."""
    print("5. POD coefficient time series...", flush=True)

    a_pod = data['a_pod']
    times = data['times']

    # Rollout from beginning
    n_rollout = min(500, a_pod.shape[0] - 1)
    a0 = torch.tensor(a_pod[0], dtype=torch.float32)
    a_pred = rollout_pod(model, a0, n_rollout, DT_SAVE).numpy()
    a_true = a_pod[:n_rollout + 1]
    t_axis = times[:n_rollout + 1]

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    for k, ax in enumerate(axes):
        ax.plot(t_axis, a_true[:, k], 'b-', lw=1.2, alpha=0.8, label='True')
        ax.plot(t_axis, a_pred[:, k], 'r--', lw=1.2, alpha=0.8, label='Predicted')
        ax.set_ylabel(f'$a_{k+1}(t)$', fontsize=12)
        ax.set_title(f'POD Mode {k+1}', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time $t$', fontsize=12)
    fig.suptitle('POD Coefficients: True vs Koopman Prediction', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'pod_timeseries.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'pod_timeseries.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved pod_timeseries.png/pdf", flush=True)


# =====================================================================
# Plot 6: Eigenvalue spectrum
# =====================================================================

def plot_eigenvalues(model):
    """Discrete and generator eigenvalues of the stable Koopman model."""
    print("6. Eigenvalue spectrum...", flush=True)

    A = model.generator_matrix().detach().cpu().numpy()
    M = model.propagator_matrix(DT_SAVE).cpu().numpy()
    eigs_M = np.linalg.eigvals(M)
    eigs_A = np.linalg.eigvals(A)

    mags = np.abs(eigs_M)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Discrete eigenvalues on complex plane
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.3, label='Unit circle')
    sc = ax.scatter(eigs_M.real, eigs_M.imag, s=40, c=mags,
                    cmap='viridis', edgecolors='black', linewidths=0.5,
                    zorder=3, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='$|\\lambda|$', shrink=0.8)
    ax.set_xlabel(r'Re($\lambda$)', fontsize=12)
    ax.set_ylabel(r'Im($\lambda$)', fontsize=12)
    ax.set_title(f'Discrete: expm($A \\cdot \\Delta t$)\n'
                 f'$|\\lambda|$ range: [{mags.min():.4f}, {mags.max():.4f}]',
                 fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    n_near_unit = np.sum(mags > 0.99)
    print(f"  Discrete |lam| range: [{mags.min():.6f}, {mags.max():.6f}]", flush=True)
    print(f"  # eigenvalues with |lam| > 0.99: {n_near_unit}", flush=True)
    print(f"  All |lam| <= 1? {np.all(mags <= 1.0 + 1e-6)}", flush=True)

    # Generator eigenvalues
    ax = axes[1]
    re_A = eigs_A.real
    im_A = eigs_A.imag
    colors_A = np.abs(eigs_A)
    ax.axvline(0, color='red', ls='--', lw=1, alpha=0.5,
               label='Re=0 (stability boundary)')
    sc2 = ax.scatter(re_A, im_A, s=40, c=colors_A, cmap='plasma',
                     edgecolors='black', linewidths=0.5, zorder=3)
    plt.colorbar(sc2, ax=ax, label='$|\\mu|$', shrink=0.8)
    ax.set_xlabel(r'Re($\mu$)', fontsize=12)
    ax.set_ylabel(r'Im($\mu$)', fontsize=12)
    max_re = np.max(re_A)
    ax.set_title(f'Generator: $A = K - L^T L$\n'
                 f'max Re($\\mu$) = {max_re:.4f} (should be $\\leq$ 0)',
                 fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    print(f"  Generator Re(mu) range: [{re_A.min():.6f}, {max_re:.6f}]", flush=True)

    # Damping analysis
    D = model.damping_matrix().cpu().numpy()
    damp_eigs = np.linalg.eigvalsh(D)
    print(f"  Damping L^T L eigenvalues: min={damp_eigs.min():.6f}, "
          f"max={damp_eigs.max():.6f}", flush=True)

    # Frequency analysis: Im(eig)/2pi relates to physical oscillation freq
    # Vortex shedding: expect a prominent frequency near St ~ 0.2 -> f ~ 0.2 Hz
    freqs_gen = np.abs(im_A) / (2 * np.pi)
    print(f"  Generator frequencies: top 5 = "
          f"{np.sort(freqs_gen)[::-1][:5]}", flush=True)

    fig.suptitle('Stable Koopman Spectrum (Wake)', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'eigenvalues.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'eigenvalues.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved eigenvalues.png/pdf", flush=True)


# =====================================================================
# Plot 7: NRMSE over time
# =====================================================================

def plot_nrmse(model, data):
    """NRMSE of POD coefficients and reconstructed vorticity over rollout time."""
    print("7. NRMSE over time...", flush=True)

    a_pod = data['a_pod']
    U_pod = data['U_pod']
    omega_mean = data['omega_mean']
    times = data['times']
    snapshots = data['snapshots']

    # Roll out from multiple starting points in validation region
    n_windows = a_pod.shape[0] // WINDOW_LEN
    n_train_windows = max(1, n_windows - 3)

    # Use multiple starting points for robustness
    start_indices = []
    for w in range(n_train_windows, n_windows):
        start_indices.append(w * WINDOW_LEN)
    # Also try a few points from the training region
    for w in range(0, n_train_windows, max(1, n_train_windows // 3)):
        start_indices.append(w * WINDOW_LEN)

    n_rollout = min(200, a_pod.shape[0] // 2)  # 40 time units

    all_nrmse_pod = []
    all_nrmse_field = []

    for si in start_indices:
        if si + n_rollout >= a_pod.shape[0]:
            continue
        a0 = torch.tensor(a_pod[si], dtype=torch.float32)
        a_pred = rollout_pod(model, a0, n_rollout, DT_SAVE).numpy()
        a_true = a_pod[si:si + n_rollout + 1]

        nrmse_pod = compute_nrmse_pod(a_pred, a_true)
        all_nrmse_pod.append(nrmse_pod)

        # Field NRMSE
        true_omega = snapshots[si:si + n_rollout + 1]
        pred_omega = reconstruct_vorticity(a_pred, U_pod, omega_mean,
                                           data.get('a_pod_mean'), data.get('a_pod_std'))
        nrmse_field = compute_nrmse_field(pred_omega, true_omega)
        all_nrmse_field.append(nrmse_field)

    all_nrmse_pod = np.array(all_nrmse_pod)
    all_nrmse_field = np.array(all_nrmse_field)
    mean_nrmse_pod = all_nrmse_pod.mean(axis=0)
    std_nrmse_pod = all_nrmse_pod.std(axis=0)
    mean_nrmse_field = all_nrmse_field.mean(axis=0)
    std_nrmse_field = all_nrmse_field.std(axis=0)

    t_axis = np.arange(n_rollout + 1) * DT_SAVE
    t_shed = t_axis / T_SHED  # in shedding periods

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # POD coefficient NRMSE
    ax = axes[0]
    ax.plot(t_shed, mean_nrmse_pod, 'b-', lw=2, label='POD coefficients')
    ax.fill_between(t_shed,
                    mean_nrmse_pod - std_nrmse_pod,
                    mean_nrmse_pod + std_nrmse_pod,
                    alpha=0.15, color='steelblue')
    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5, label='NRMSE = 1')
    cross_pod = np.where(mean_nrmse_pod > 1.0)[0]
    if len(cross_pod) > 0:
        tc = t_shed[cross_pod[0]]
        ax.annotate(f'NRMSE=1 at {tc:.1f} $T_s$',
                    xy=(tc, 1.0), xytext=(tc + 0.5, 1.3),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    ax.set_xlabel(f'Time [shedding periods $T_s \\approx {T_SHED:.0f}$]', fontsize=12)
    ax.set_ylabel('NRMSE', fontsize=12)
    ax.set_title('POD Coefficient NRMSE', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, t_shed[-1])
    ax.set_ylim(0, min(3.0, mean_nrmse_pod.max() * 1.1))
    ax.grid(True, alpha=0.2)

    # Vorticity field NRMSE
    ax = axes[1]
    ax.plot(t_shed, mean_nrmse_field, 'r-', lw=2, label='Vorticity field')
    ax.fill_between(t_shed,
                    mean_nrmse_field - std_nrmse_field,
                    mean_nrmse_field + std_nrmse_field,
                    alpha=0.15, color='salmon')
    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5, label='NRMSE = 1')
    cross_field = np.where(mean_nrmse_field > 1.0)[0]
    if len(cross_field) > 0:
        tc = t_shed[cross_field[0]]
        ax.annotate(f'NRMSE=1 at {tc:.1f} $T_s$',
                    xy=(tc, 1.0), xytext=(tc + 0.5, 1.3),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    ax.set_xlabel(f'Time [shedding periods $T_s \\approx {T_SHED:.0f}$]', fontsize=12)
    ax.set_ylabel('NRMSE', fontsize=12)
    ax.set_title('Reconstructed Vorticity NRMSE', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, t_shed[-1])
    ax.set_ylim(0, min(3.0, mean_nrmse_field.max() * 1.1))
    ax.grid(True, alpha=0.2)

    horizon_pod = t_shed[cross_pod[0]] if len(cross_pod) > 0 else float('inf')
    horizon_field = t_shed[cross_field[0]] if len(cross_field) > 0 else float('inf')

    fig.suptitle(f'Prediction Horizon: {horizon_pod:.1f} $T_s$ (POD), '
                 f'{horizon_field:.1f} $T_s$ (field)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'nrmse.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'nrmse.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved nrmse.png/pdf (POD horizon: {horizon_pod:.1f} T_s, "
          f"field horizon: {horizon_field:.1f} T_s)", flush=True)

    return mean_nrmse_pod, horizon_pod


# =====================================================================
# Plot 8: Reconstruction quality
# =====================================================================

def plot_reconstruction(model, data):
    """Autoencoder round-trip: POD -> encode -> decode -> POD -> vorticity."""
    print("8. Reconstruction quality...", flush=True)

    x, y = data['x'], data['y']
    nx, ny = len(x), len(y)
    a_pod = data['a_pod']
    U_pod = data['U_pod']
    omega_mean = data['omega_mean']
    snapshots = data['snapshots']

    # Pick 4 random validation snapshots
    n_windows = a_pod.shape[0] // WINDOW_LEN
    n_train_windows = max(1, n_windows - 3)
    val_start = n_train_windows * WINDOW_LEN
    val_end = a_pod.shape[0]
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(np.arange(val_start, val_end), size=4, replace=False)

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    extent = [x[0], x[-1], y[0], y[-1]]
    vmin, vmax = -6.0, 6.0

    overall_mse = 0.0

    for row, snap_idx in enumerate(sample_indices):
        a_in = torch.tensor(a_pod[snap_idx:snap_idx+1], dtype=torch.float32)
        with torch.no_grad():
            z, a_recon = model.ae(a_in)
        a_recon_np = a_recon.numpy()[0]

        # True vorticity (from full snapshots)
        omega_true = snapshots[snap_idx].reshape(ny, nx)

        # Denormalize for reconstruction
        a_pod_mean = data.get('a_pod_mean')
        a_pod_std_v = data.get('a_pod_std')

        # POD-reconstructed (no KAN, just r modes)
        a_denorm = a_pod[snap_idx] * a_pod_std_v + a_pod_mean if a_pod_mean is not None else a_pod[snap_idx]
        omega_pod = (omega_mean + U_pod @ a_denorm).reshape(ny, nx)

        # KAN round-trip reconstructed
        a_recon_denorm = a_recon_np * a_pod_std_v + a_pod_mean if a_pod_mean is not None else a_recon_np
        omega_kan = (omega_mean + U_pod @ a_recon_denorm).reshape(ny, nx)

        mse_i = np.mean((omega_pod - omega_kan)**2)
        overall_mse += mse_i

        # Column 0: true vorticity
        ax = axes[row, 0]
        ax.imshow(np.clip(omega_true, vmin, vmax), origin='lower', extent=extent,
                  cmap='RdBu_r', vmin=vmin, vmax=vmax,
                  interpolation='bilinear', aspect='auto')
        add_cylinders(ax, data)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title('True $\\omega$', fontsize=12)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Column 1: POD-only reconstruction
        ax = axes[row, 1]
        ax.imshow(np.clip(omega_pod, vmin, vmax), origin='lower', extent=extent,
                  cmap='RdBu_r', vmin=vmin, vmax=vmax,
                  interpolation='bilinear', aspect='auto')
        add_cylinders(ax, data)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(f'POD ({R_POD} modes)', fontsize=12)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Column 2: KAN round-trip
        ax = axes[row, 2]
        ax.imshow(np.clip(omega_kan, vmin, vmax), origin='lower', extent=extent,
                  cmap='RdBu_r', vmin=vmin, vmax=vmax,
                  interpolation='bilinear', aspect='auto')
        add_cylinders(ax, data)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(f'KAN Round-Trip (MSE={mse_i:.4f})', fontsize=12)
        else:
            ax.set_title(f'MSE={mse_i:.4f}', fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

    overall_mse /= 4
    fig.suptitle(f'Autoencoder Reconstruction Quality (mean MSE = {overall_mse:.5f})',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'reconstruction.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'reconstruction.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved reconstruction.png/pdf (mean MSE = {overall_mse:.5f})", flush=True)

    return overall_mse


# =====================================================================
# Loss curves
# =====================================================================

def plot_loss_curves(history):
    """Training and validation loss curves."""
    print("  Plotting loss curves...", flush=True)
    epochs = np.arange(1, len(history['train_total']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    labels = [('total', 'Total Loss'), ('predx', 'Prediction Loss'),
              ('latent', 'Latent Consistency'), ('recon', 'Reconstruction')]

    for ax, (key, title) in zip(axes.flatten(), labels):
        ax.plot(epochs, history[f'train_{key}'], label='Train', lw=1.5)
        ax.plot(epochs, history[f'val_{key}'], label='Val', lw=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title(title)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Wake Stable Koopman Training', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'loss_curves.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'loss_curves.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved loss_curves.png/pdf", flush=True)


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Two-cylinder wake with stable Koopman KAN model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--skip-train', action='store_true',
                        help='Load existing checkpoint instead of training')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Load cached simulation data instead of re-running')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    print("=" * 70, flush=True)
    print("Two-Cylinder Wake: Stable Koopman with KANLinear Autoencoder", flush=True)
    print("=" * 70, flush=True)
    print(f"  Grid: {NX} x {NY} ({NX*NY:,} points)", flush=True)
    print(f"  Re = {RE}, dt_sim = {DT_SIM}, dt_save = {DT_SAVE}", flush=True)
    print(f"  POD modes: r = {R_POD}", flush=True)
    print(f"  Model: KANAutoEncoder({INPUT_SIZE}, {MID_SIZE}, {LATENT_SIZE})", flush=True)
    print(f"  Generator: A = K - L^T L  (stable, latent_dim={LATENT_SIZE})", flush=True)
    print(f"  Device: {args.device}", flush=True)
    print(f"  Shedding period: T_s ~ {T_SHED} time units", flush=True)

    # ---- Simulation + POD ----
    data_path = OUTDIR / 'wake_data.npz'
    ckpt_path = OUTDIR / 'wake_stable_model.pt'

    if args.skip_sim and data_path.exists():
        print(f"\n--- Loading cached data from {data_path} ---", flush=True)
        data = load_data(data_path)
        print(f"  Snapshots: {data['snapshots'].shape}", flush=True)
        print(f"  POD coords: {data['a_pod'].shape}", flush=True)
        energy = np.cumsum(data['S_pod']**2) / np.sum(data['S_pod']**2)
        print(f"  {R_POD} modes capture {energy[R_POD-1]*100:.1f}% energy", flush=True)
    else:
        print(f"\n--- Running simulation + POD ---", flush=True)
        data = run_simulation_and_pod(data_path)

    a_pod_raw = data['a_pod']
    n_snap = a_pod_raw.shape[0]
    print(f"  Total snapshots: {n_snap}", flush=True)
    print(f"  Raw POD coord range: [{a_pod_raw.min():.2f}, {a_pod_raw.max():.2f}]", flush=True)

    # ---- Normalize POD coordinates for KAN ----
    a_pod_mean = a_pod_raw.mean(axis=0)
    a_pod_std = a_pod_raw.std(axis=0)
    a_pod_std[a_pod_std < 1e-8] = 1.0  # avoid division by zero
    a_pod = (a_pod_raw - a_pod_mean) / a_pod_std
    print(f"  Normalized POD coord range: [{a_pod.min():.2f}, {a_pod.max():.2f}]", flush=True)

    # Store normalization params for reconstruction and replace a_pod with normalized
    data['a_pod_mean'] = a_pod_mean
    data['a_pod_std'] = a_pod_std
    data['a_pod_raw'] = a_pod_raw
    data['a_pod'] = a_pod  # all plot functions now use normalized coords

    # ---- Prepare trajectories ----
    print("\n--- Preparing trajectory windows ---", flush=True)
    traj = prepare_trajectories(a_pod)
    n_windows = traj.shape[0]
    n_train_windows = max(1, n_windows - 3)
    n_val_windows = n_windows - n_train_windows
    print(f"  Windows: {n_windows} total ({n_train_windows} train, "
          f"{n_val_windows} val)", flush=True)

    # ---- Train or load model ----
    if args.skip_train and ckpt_path.exists():
        print(f"\n--- Loading checkpoint: {ckpt_path} ---", flush=True)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        ae = KANAutoEncoder(
            INPUT_SIZE, MID_SIZE, LATENT_SIZE,
            grid_range_l1=[-4, 4], grid_size=10,
            grid_range_l2=[-8, 8],
        )
        model = DeepKoopmanCTStable(ae, LATENT_SIZE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        history = ckpt.get('history', None)
        if history is not None:
            print(f"  Loaded model (final val loss: "
                  f"{history['val_total'][-1]:.6f})", flush=True)
    else:
        print(f"\n--- Training ({args.epochs} epochs) ---", flush=True)
        ae = KANAutoEncoder(
            INPUT_SIZE, MID_SIZE, LATENT_SIZE,
            grid_range_l1=[-4, 4], grid_size=10,
            grid_range_l2=[-8, 8],
        ).to(args.device)
        model, history = train_model(ae, traj, epochs=args.epochs,
                                     lr=LR, device=args.device)

        torch.save({
            'model_state_dict': model.state_dict(),
            'dt': DT_SAVE,
            'history': history,
            'config': {
                'input_size': INPUT_SIZE,
                'mid_size': MID_SIZE,
                'latent_size': LATENT_SIZE,
                'r_pod': R_POD,
                'nx': NX, 'ny': NY,
                're': RE,
                'dt_save': DT_SAVE,
                'epochs': args.epochs,
                'lr': LR,
                'batch_size': BATCH_SIZE,
            },
        }, ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path}", flush=True)

        # Loss curves
        plot_loss_curves(history)

    # ---- Save config ----
    config = {
        'nx': NX, 'ny': NY, 're': RE,
        'dt_sim': DT_SIM, 'dt_save': DT_SAVE,
        't_end': T_END, 't_start': T_START,
        'r_pod': R_POD,
        'window_len': WINDOW_LEN,
        'input_size': INPUT_SIZE, 'mid_size': MID_SIZE,
        'latent_size': LATENT_SIZE,
        'epochs': args.epochs, 'lr': LR,
        'batch_size': BATCH_SIZE, 'grad_clip': GRAD_CLIP,
        'weight_decay': WEIGHT_DECAY,
        'gamma_predx': GAMMA_PREDX, 'alpha_latent': ALPHA_LATENT,
        'beta_recon': BETA_RECON,
        'n_windows': n_windows,
        'n_train_windows': n_train_windows,
        'n_val_windows': n_val_windows,
        't_shed': T_SHED,
        'device': args.device,
    }
    with open(OUTDIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ---- Generate all plots ----
    print("\n--- Generating plots ---", flush=True)

    # 1. Vorticity comparison (hero figure)
    plot_vorticity_comparison(model, data)

    # 2. POD energy spectrum
    plot_pod_spectrum(data)

    # 3. POD mode shapes
    plot_pod_modes(data)

    # 4. Centerline space-time
    plot_centerline_spacetime(model, data)

    # 5. POD coefficient time series
    plot_pod_timeseries(model, data)

    # 6. Eigenvalue spectrum
    plot_eigenvalues(model)

    # 7. NRMSE over time
    mean_nrmse, horizon = plot_nrmse(model, data)

    # 8. Reconstruction quality
    recon_mse = plot_reconstruction(model, data)

    # ---- Summary ----
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  System:  Two-cylinder wake, Re={RE}, {NX}x{NY} grid", flush=True)
    print(f"  POD:     {R_POD} modes", flush=True)
    print(f"  Model:   KANAutoEncoder({INPUT_SIZE}, {MID_SIZE}, {LATENT_SIZE}) "
          f"+ StableKoopman", flush=True)
    print(f"  Horizon: {horizon:.1f} shedding periods "
          f"({horizon * T_SHED:.1f} time units)", flush=True)
    print(f"  Recon MSE: {recon_mse:.5f}", flush=True)
    if history is not None:
        print(f"  Final train loss: {history['train_total'][-1]:.6f}", flush=True)
        print(f"  Final val loss:   {history['val_total'][-1]:.6f}", flush=True)
    print(f"  All results saved to: {OUTDIR.resolve()}/", flush=True)
    print("=" * 70, flush=True)
