"""
Kuramoto-Sivashinsky equation with stable Koopman KAN model.

The KS equation  u_t + u*u_x + u_xx + u_xxxx = 0  exhibits spatiotemporal
chaos on domain [0, L] with periodic BCs. This experiment applies the
stable Deep Koopman model (A = K - L^T L, guaranteeing |lambda| <= 1) with
a KANLinear autoencoder to LIFT the 64-dimensional PDE state to a 128-dim
latent space where the dynamics are approximately linear.

The KS equation is a canonical benchmark for Koopman/DMD model reduction.
Here we do the opposite: lift to higher dimensions, then analyze the
learned dictionary via level-set decomposition.

Simulator: ETDRK4 spectral method (Kassam & Trefethen 2005).
Model:     KANAutoEncoder(64, 96, 128) + DeepKoopmanCTStable(128).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from models import KANAutoEncoder
from data import PairDataset
from training import set_seed

# =====================================================================
# Configuration
# =====================================================================

OUTDIR = Path('results/ks_stable')

# KS parameters
L_DOMAIN = 22.0          # domain length (chaotic regime for L > ~17)
N_GRID = 64              # number of Fourier modes / grid points
DT_SIM = 0.005           # ETDRK4 integration timestep (conservative for stability)
DT_SAVE = 0.25           # save interval for Koopman model
BURN_IN = 100.0           # transient to discard (time units)
T_TOTAL = 250.0           # simulation length after burn-in (time units)
N_TRAJ = 30               # number of trajectories

# KS Lyapunov time for L=22 (Edson et al. 2019, Pathak et al. 2018)
LAMBDA_MAX_KS = 0.043     # largest Lyapunov exponent (/time unit)
TAU_L = 1.0 / LAMBDA_MAX_KS  # ~23.3 time units

# Model architecture
INPUT_SIZE = N_GRID       # 64 grid points
MID_SIZE = 96             # m: single hidden layer = compositional richness (paper Eq. 7).
                          # WIDER than the 64-D input so layer 2 has enough additive units
                          # to compose advection cross-terms u_i u_{i+/-1} (was 32 -- a
                          # bottleneck below both input n=64 and dictionary d=128).
LATENT_SIZE = 128         # d: dictionary size (number of Koopman observables z_k), 2x lift

# Training
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
GRAD_CLIP = 1.0
N_TRAIN = 24              # training trajectories (out of 30)
N_VAL = N_TRAJ - N_TRAIN  # 6 validation trajectories

# Multi-step (autoregressive) rollout training.
# The eval horizon is an 80-step autoregressive rollout, but a one-step loss never
# sees its own compounding error. We add a latent-rollout loss: encode once, apply
# M^j in latent space, decode each step, and match the true future x_{t+j}. K ramps
# from 1 to ROLLOUT_HORIZON after a warmup so early training stays stable.
ROLLOUT_HORIZON = 16      # max latent-rollout length K_max in the training loss
ROLLOUT_WARMUP = 20       # epochs of pure one-step (K=1) before ramping K up
ROLLOUT_DISCOUNT = 0.9    # per-step weight gamma^(j-1) on the multi-step terms

# Loss weights
GAMMA_PREDX = 1.0
ALPHA_LATENT = 1.0
BETA_RECON = 0.5


# =====================================================================
# Stable Deep Koopman model (copied from lorenz_stable.py)
# =====================================================================

class DeepKoopmanCTStable(nn.Module):
    """Continuous-time Deep Koopman with stable generator.

    A = K - L^T L  where:
        K = -K^T              (skew-symmetric, drives rotation/oscillation)
        L^T L >= 0            (PSD, drives damping/contraction)

    => Re(eigenvalues of A) <= 0  => |eigenvalues of expm(A·dt)| <= 1

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
# KS Simulator: ETDRK4 spectral method (Kassam & Trefethen 2005)
# =====================================================================

def simulate_ks(n_traj=30, L=22.0, N=64, dt_sim=0.5, dt_save=0.25,
                burn_in=100.0, T=250.0, seed=None):
    """Simulate Kuramoto-Sivashinsky equation using ETDRK4.

    KS equation: u_t = -u*u_x - u_xx - u_xxxx
    In Fourier space: uhat_t = -(ik/2)*FFT(u^2) + k^2*uhat - k^4*uhat

    Args:
        n_traj: number of independent trajectories
        L: domain length
        N: number of grid points (must be even)
        dt_sim: ETDRK4 timestep
        dt_save: save interval for output
        burn_in: transient time to discard
        T: simulation time after burn-in
        seed: random seed for initial conditions

    Returns:
        data: torch.Tensor of shape (n_traj, n_steps, N)
              where n_steps = int(T / dt_save)
    """
    if seed is not None:
        np.random.seed(seed)

    # Spatial grid
    dx = L / N
    x = np.arange(N) * dx

    # Wavenumbers (standard FFT ordering)
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # shape (N,)

    # Linear operator in Fourier space: L(k) = k^2 - k^4
    L_k = k**2 - k**4

    # Precompute ETDRK4 coefficients (Kassam & Trefethen 2005)
    # Uses contour integral for numerical stability
    M_contour = 32
    r = np.exp(2j * np.pi * (np.arange(1, M_contour + 1) - 0.5) / M_contour)

    # Coefficients are computed per wavenumber
    E = np.exp(L_k * dt_sim)                         # (N,)
    E2 = np.exp(L_k * dt_sim / 2)                    # (N,)

    Q = np.zeros(N, dtype=complex)
    f1 = np.zeros(N, dtype=complex)
    f2 = np.zeros(N, dtype=complex)
    f3 = np.zeros(N, dtype=complex)

    for i in range(N):
        LR = dt_sim * L_k[i] + r                     # (M_contour,)
        Q[i] = dt_sim * np.mean(np.real((np.exp(LR / 2) - 1) / LR))
        f1[i] = dt_sim * np.mean(np.real(
            (-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3))
        f2[i] = dt_sim * np.mean(np.real(
            (2 + LR + np.exp(LR) * (-2 + LR)) / LR**3))
        f3[i] = dt_sim * np.mean(np.real(
            (-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3))

    # Make coefficients real (they should be for real-valued u)
    E = E.real
    E2 = E2.real
    Q = Q.real
    f1 = f1.real
    f2 = f2.real
    f3 = f3.real

    # 2/3 dealiasing mask: zero modes with |wavenumber| > N/3
    dealias = np.ones(N, dtype=bool)
    kmax_d = N // 3
    dealias[kmax_d + 1: N - kmax_d] = False

    # Nonlinear term: N(uhat) = -(ik/2) * FFT(IFFT(uhat)^2), dealiased
    ik_half = 0.5j * k

    def nonlinear(uhat):
        uhat_d = uhat * dealias  # dealias input
        u = np.real(np.fft.ifft(uhat_d))
        return -ik_half * np.fft.fft(u**2) * dealias  # dealias output

    # Time parameters
    n_burn = int(np.ceil(burn_in / dt_sim))
    n_save_ratio = max(1, int(round(dt_save / dt_sim)))
    n_save_steps = int(T / dt_save)
    n_total_after_burn = n_save_steps * n_save_ratio

    print(f"  KS simulator: L={L}, N={N}, dt_sim={dt_sim}, dt_save={dt_save}", flush=True)
    print(f"  burn_in steps={n_burn}, save steps={n_save_steps}, "
          f"save ratio={n_save_ratio}", flush=True)

    all_data = np.zeros((n_traj, n_save_steps, N))

    def etdrk4_step(uhat):
        Nu = nonlinear(uhat)
        a_ = E2 * uhat + Q * Nu
        Na = nonlinear(a_)
        b_ = E2 * uhat + Q * Na
        Nb = nonlinear(b_)
        c_ = E2 * a_ + Q * (2 * Nb - Nu)
        Nc = nonlinear(c_)
        return (E * uhat + Nu * f1 + 2 * (Na + Nb) * f2 + Nc * f3) * dealias

    MAX_AMP = 5.0  # reject trajectories exceeding this (attractor has |u| ~ 3)

    for traj_idx in range(n_traj):
        success = False
        for attempt in range(20):
            # Initial condition: single low-amplitude cosine + small noise
            u0 = (0.2 + 0.1 * np.random.randn()) * np.cos(2 * np.pi * x / L)
            u0 += 0.01 * np.random.randn(N)
            uhat = np.fft.fft(u0) * dealias
            blown = False

            # Burn-in (check every 100 steps)
            for step_b in range(n_burn):
                uhat = etdrk4_step(uhat)
                if step_b % 100 == 0:
                    u_check = np.fft.ifft(uhat).real
                    if np.any(np.isnan(u_check)) or np.max(np.abs(u_check)) > MAX_AMP:
                        blown = True
                        break

            if blown:
                continue

            # Production run (check every save)
            save_idx = 0
            for step in range(n_total_after_burn):
                if step % n_save_ratio == 0 and save_idx < n_save_steps:
                    u_real = np.fft.ifft(uhat).real
                    if np.max(np.abs(u_real)) > MAX_AMP:
                        blown = True
                        break
                    all_data[traj_idx, save_idx, :] = u_real
                    save_idx += 1

                uhat = etdrk4_step(uhat)

            if blown:
                continue

            success = True
            break

        if not success:
            print(f"  WARNING: Traj {traj_idx+1} failed after 20 attempts!", flush=True)

        if (traj_idx + 1) % 10 == 0 or traj_idx == 0:
            u_snap = np.real(np.fft.ifft(uhat))
            print(f"  Traj {traj_idx + 1}/{n_traj}: "
                  f"u range [{u_snap.min():.2f}, {u_snap.max():.2f}]", flush=True)

    return torch.tensor(all_data, dtype=torch.float32)


# =====================================================================
# Training
# =====================================================================

class SequenceDataset(torch.utils.data.Dataset):
    """Sliding windows of length (horizon+1) from trajectory data.

    traj: (n_traj, n_steps, d)  ->  each item: (horizon+1, d).
    Used for the multi-step (autoregressive) rollout loss. With horizon=1 this
    reduces to consecutive (x_t, x_{t+1}) pairs (as a length-2 window).
    """

    def __init__(self, traj, horizon):
        self.traj = traj
        self.n_traj, self.n_steps, self.d = traj.shape
        self.horizon = int(horizon)
        self.per_traj = self.n_steps - self.horizon
        assert self.per_traj > 0, "trajectory shorter than rollout horizon"
        self.n_windows = self.n_traj * self.per_traj

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        traj_id = idx // self.per_traj
        t = idx % self.per_traj
        return self.traj[traj_id, t:t + self.horizon + 1]  # (horizon+1, d)


def koopman_rollout_loss(model, seq, dt, K, discount):
    """Multi-step latent-rollout loss on windows ``seq`` of shape (B, H+1, d),
    using the first ``K`` steps.

    Encode x_0 once, propagate z_j = M^j z_0 in latent space (M computed once and
    reused), decode each step, and match the true future x_{t+j} with a per-step
    discount gamma^(j-1). Returns (total, pred, latent, recon) scalar tensors,
    where ``pred``/``latent`` are discount-weighted averages over the K steps.

    K=1 reproduces the original one-step objective. The eval protocol
    (rollout_ks) is the re-encoding autoregressive rollout; enforcing accuracy of
    M^j here is what curbs the per-step error that compounds over that rollout.
    """
    mse = nn.functional.mse_loss
    x0 = seq[:, 0]
    z = model.ae.encoder(x0)
    recon = mse(model.ae.decoder(z), x0)

    A = model.generator_matrix()
    M = torch.matrix_exp(A * dt)              # computed once, reused across steps

    pred_terms = z.new_zeros(())
    lat_terms = z.new_zeros(())
    wsum = 0.0
    for j in range(1, K + 1):
        z = z @ M.T                           # z_j = M^j z_0 (autoregressive in latent)
        xj = seq[:, j]
        w = discount ** (j - 1)
        pred_terms = pred_terms + w * mse(model.ae.decoder(z), xj)
        lat_terms = lat_terms + w * mse(z, model.ae.encoder(xj))
        wsum += w

    pred = pred_terms / wsum
    latent = lat_terms / wsum
    total = GAMMA_PREDX * pred + ALPHA_LATENT * latent + BETA_RECON * recon
    return total, pred, latent, recon


def train_model(ae, traj, epochs=EPOCHS, lr=1e-3, device='cpu',
                rollout_horizon=ROLLOUT_HORIZON, rollout_warmup=ROLLOUT_WARMUP,
                rollout_discount=ROLLOUT_DISCOUNT):
    """Train the stable Koopman model on KS data with a curriculum multi-step
    rollout loss (K grows 1 -> rollout_horizon after rollout_warmup epochs)."""
    K_max = max(1, int(rollout_horizon))
    train_dl = DataLoader(SequenceDataset(traj[:N_TRAIN], K_max),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(SequenceDataset(traj[N_TRAIN:], K_max),
                        batch_size=BATCH_SIZE)

    def k_schedule(ep):
        """K=1 during warmup, then ramp linearly to K_max over remaining epochs."""
        if K_max <= 1 or ep <= rollout_warmup:
            return 1
        frac = (ep - rollout_warmup) / max(1, epochs - rollout_warmup)
        return int(min(K_max, 1 + round(frac * (K_max - 1))))

    model = DeepKoopmanCTStable(ae, LATENT_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_skew = LATENT_SIZE * (LATENT_SIZE - 1) // 2
    n_damp = LATENT_SIZE + LATENT_SIZE * (LATENT_SIZE - 1) // 2
    print(f"  Stable Koopman [{INPUT_SIZE},{MID_SIZE},{LATENT_SIZE}]: "
          f"{n_params:,} params ({n_skew} skew + {n_damp} damping = "
          f"{n_skew + n_damp} generator)", flush=True)
    print(f"  Rollout curriculum: K 1 -> {K_max} after {rollout_warmup} warmup "
          f"epochs, discount {rollout_discount}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {
        'train_total': [], 'train_predx': [], 'train_latent': [], 'train_recon': [],
        'val_total': [], 'val_predx': [], 'val_latent': [], 'val_recon': [], 'K': [],
    }

    for ep in range(1, epochs + 1):
        K = k_schedule(ep)
        model.train()
        losses_t, losses_p, losses_l, losses_r = [], [], [], []

        for seq in train_dl:
            seq = seq.to(device)
            loss, loss_p, loss_l, loss_r = koopman_rollout_loss(
                model, seq, DT_SAVE, K, rollout_discount)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            losses_t.append(loss.item())
            losses_p.append(loss_p.item())
            losses_l.append(loss_l.item())
            losses_r.append(loss_r.item())

        scheduler.step()

        # Validation at the same rollout length K
        model.eval()
        vloss_t, vloss_p, vloss_l, vloss_r = [], [], [], []
        with torch.no_grad():
            for seq in val_dl:
                seq = seq.to(device)
                loss, loss_p, loss_l, loss_r = koopman_rollout_loss(
                    model, seq, DT_SAVE, K, rollout_discount)
                vloss_t.append(loss.item())
                vloss_p.append(loss_p.item())
                vloss_l.append(loss_l.item())
                vloss_r.append(loss_r.item())

        history['train_total'].append(float(np.mean(losses_t)))
        history['train_predx'].append(float(np.mean(losses_p)))
        history['train_latent'].append(float(np.mean(losses_l)))
        history['train_recon'].append(float(np.mean(losses_r)))
        history['val_total'].append(float(np.mean(vloss_t)))
        history['val_predx'].append(float(np.mean(vloss_p)))
        history['val_latent'].append(float(np.mean(vloss_l)))
        history['val_recon'].append(float(np.mean(vloss_r)))
        history['K'].append(K)

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            M = model.propagator_matrix(DT_SAVE).cpu().numpy()
            eigs = np.linalg.eigvals(M)
            mag_range = f'[{np.abs(eigs).min():.4f}, {np.abs(eigs).max():.4f}]'
            print(f"  Ep {ep:3d} | K={K:2d} | train {np.mean(losses_t):.5f} "
                  f"(pred {np.mean(losses_p):.5f}, lat {np.mean(losses_l):.5f}, "
                  f"rec {np.mean(losses_r):.5f}) | val {np.mean(vloss_t):.5f} | "
                  f"|lam| {mag_range}", flush=True)

    return model.cpu().eval(), history


# =====================================================================
# Autoregressive rollout
# =====================================================================

def rollout_ks(model, x0, n_steps, dt):
    """Autoregressive Koopman rollout for KS snapshots.

    Args:
        model: DeepKoopmanCTStable (on CPU or GPU)
        x0: (N_GRID,) initial snapshot
        n_steps: number of steps to roll forward
        dt: time step between snapshots

    Returns:
        (n_steps+1, N_GRID) tensor of predicted snapshots
    """
    xs = [x0]
    x = x0
    with torch.no_grad():
        for _ in range(n_steps):
            out = model.forward_one_step(x.unsqueeze(0), dt)
            x = out['x_tp1_pred'].squeeze(0)
            xs.append(x)
    return torch.stack(xs)


def compute_nrmse(pred, true):
    """Compute spatially-averaged NRMSE at each timestep.

    Args:
        pred: (T, N) predicted snapshots
        true: (T, N) true snapshots

    Returns:
        nrmse: (T,) NRMSE per timestep
    """
    std_spatial = true.reshape(-1).std()
    if std_spatial < 1e-12:
        std_spatial = 1.0
    rmse = (pred - true).pow(2).mean(dim=1).sqrt()
    return (rmse / std_spatial).numpy()


# =====================================================================
# Plot 1: Space-time rollout
# =====================================================================

def plot_spacetime_rollout(model, val_traj, dt):
    """Space-time heatmap: true vs predicted vs error."""
    print("1. Space-time rollout...", flush=True)

    n_rollout = 400  # 400 steps = 100 time units at dt_save=0.25
    x0 = val_traj[0, 0]

    pred = rollout_ks(model, x0, n_rollout, dt)
    true = val_traj[0, :n_rollout + 1]

    pred_np = pred.numpy()
    true_np = true.numpy()
    error_np = np.abs(true_np - pred_np)

    t_axis = np.arange(n_rollout + 1) * DT_SAVE
    x_axis = np.linspace(0, L_DOMAIN, N_GRID, endpoint=False)

    vmax = max(np.abs(true_np).max(), np.abs(pred_np).max())
    vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    titles = ['True $u(x, t)$', 'Predicted $u(x, t)$', '$|u_{true} - u_{pred}|$']
    data_list = [true_np, pred_np, error_np]
    cmaps = ['RdBu_r', 'RdBu_r', 'hot']
    vmins = [vmin, vmin, 0]
    vmaxs = [vmax, vmax, error_np.max()]

    for ax, title, data, cmap, vm_lo, vm_hi in zip(
            axes, titles, data_list, cmaps, vmins, vmaxs):
        im = ax.pcolormesh(x_axis, t_axis, data,
                           cmap=cmap, vmin=vm_lo, vmax=vm_hi,
                           shading='auto', rasterized=True)
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, shrink=0.8)

    axes[0].set_ylabel('Time $t$', fontsize=12)

    # Mark Lyapunov times on y-axis
    for tau_i in range(1, int(t_axis[-1] / TAU_L) + 1):
        for ax in axes:
            ax.axhline(tau_i * TAU_L, color='gray', ls=':', alpha=0.3, lw=0.5)

    fig.suptitle(f'KS Space-Time Rollout ({n_rollout} Koopman steps, '
                 f'{n_rollout * DT_SAVE:.0f} time units)', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'spacetime_rollout.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'spacetime_rollout.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved spacetime_rollout.png/pdf", flush=True)


# =====================================================================
# Plot 2: Snapshot comparisons
# =====================================================================

def plot_snapshots(model, val_traj, dt):
    """Spatial profiles at selected times."""
    print("2. Snapshot comparisons...", flush=True)

    n_rollout = 400
    x0 = val_traj[0, 0]
    pred = rollout_ks(model, x0, n_rollout, dt)
    true = val_traj[0, :n_rollout + 1]

    # Snapshot times in time units: 0, 10, 25, 50, 75, 100
    snapshot_times = [0, 10, 25, 50, 75, 100]
    snapshot_steps = [int(t / DT_SAVE) for t in snapshot_times]

    x_axis = np.linspace(0, L_DOMAIN, N_GRID, endpoint=False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for idx, (ax, t_val, step) in enumerate(
            zip(axes_flat, snapshot_times, snapshot_steps)):
        if step >= true.shape[0]:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
            continue

        u_true = true[step].numpy()
        u_pred = pred[step].numpy()

        rmse = np.sqrt(np.mean((u_true - u_pred)**2))
        std = np.std(u_true) if np.std(u_true) > 1e-12 else 1.0
        nrmse = rmse / std

        ax.plot(x_axis, u_true, 'b-', lw=1.5, alpha=0.8, label='True')
        ax.plot(x_axis, u_pred, 'r--', lw=1.5, alpha=0.8, label='Predicted')
        ax.set_title(f'$t = {t_val}$  (NRMSE = {nrmse:.4f})', fontsize=11)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(x)$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle('KS Snapshot Comparison: True vs Predicted', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'snapshots.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'snapshots.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved snapshots.png/pdf", flush=True)


# =====================================================================
# Plot 3: NRMSE over time
# =====================================================================

def plot_nrmse(model, val_traj, dt):
    """NRMSE vs time (linear and log scale), averaged over validation set."""
    print("3. NRMSE over time...", flush=True)

    n_rollout = 400  # 100 time units
    all_nrmse = []

    for i in range(val_traj.shape[0]):
        x0 = val_traj[i, 0]
        pred = rollout_ks(model, x0, n_rollout, dt)
        true_i = val_traj[i, :n_rollout + 1]
        nrmse_i = compute_nrmse(pred, true_i)
        all_nrmse.append(nrmse_i)

    all_nrmse = np.array(all_nrmse)  # (n_val, n_rollout+1)
    mean_nrmse = all_nrmse.mean(axis=0)
    std_nrmse = all_nrmse.std(axis=0)

    t_steps = np.arange(n_rollout + 1)
    t_time = t_steps * DT_SAVE
    t_tau = t_time / TAU_L

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax = axes[0]
    ax.plot(t_tau, mean_nrmse, 'k-', lw=2, label='Mean NRMSE')
    ax.fill_between(t_tau, mean_nrmse - std_nrmse, mean_nrmse + std_nrmse,
                    alpha=0.15, color='steelblue')
    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5, label='NRMSE = 1')
    ax.fill_between(t_tau, 0, mean_nrmse, where=mean_nrmse < 1.0,
                    color='green', alpha=0.08)
    cross = np.where(mean_nrmse > 1.0)[0]
    if len(cross) > 0:
        tc = t_tau[cross[0]]
        ax.annotate(f'NRMSE=1 at {tc:.2f} $\\tau_L$',
                    xy=(tc, 1.0), xytext=(tc + 0.3, 1.3),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    ax.set_xlabel(r'Time [$\tau_L$]  ($\tau_L \approx 23$ time units)', fontsize=12)
    ax.set_ylabel('NRMSE', fontsize=12)
    ax.set_title('Normalized RMSE (linear)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, t_tau[-1])
    ax.set_ylim(0, min(3.0, mean_nrmse.max() * 1.1))
    ax.grid(True, alpha=0.2)

    # x-axis: also show Koopman steps
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    step_ticks = np.array([0, 100, 200, 300, 400])
    ax_top.set_xticks(step_ticks * DT_SAVE / TAU_L)
    ax_top.set_xticklabels([str(s) for s in step_ticks])
    ax_top.set_xlabel('Koopman steps', fontsize=10)

    # Log scale
    ax = axes[1]
    ax.semilogy(t_tau, mean_nrmse, 'k-', lw=2)
    ax.fill_between(t_tau,
                    np.maximum(mean_nrmse - std_nrmse, 1e-4),
                    mean_nrmse + std_nrmse,
                    alpha=0.15, color='steelblue')
    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel(r'Time [$\tau_L$]', fontsize=12)
    ax.set_ylabel('NRMSE (log)', fontsize=12)
    ax.set_title('NRMSE Growth (log scale)', fontsize=12)
    ax.set_xlim(0, t_tau[-1])
    ax.grid(True, alpha=0.2)

    horizon_time = t_time[cross[0]] if len(cross) > 0 else float('inf')
    horizon_tau = t_tau[cross[0]] if len(cross) > 0 else float('inf')
    fig.suptitle(f'KS Prediction Horizon: {horizon_tau:.2f} $\\tau_L$ '
                 f'({horizon_time:.1f} time units)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'nrmse.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'nrmse.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved nrmse.png/pdf (horizon: {horizon_tau:.2f} tau_L = "
          f"{horizon_time:.1f} time units)", flush=True)

    return mean_nrmse, horizon_tau


# =====================================================================
# Plot 4: Eigenvalue spectrum
# =====================================================================

def plot_eigenvalues(model):
    """Discrete and generator eigenvalues of the stable Koopman model."""
    print("4. Eigenvalue spectrum...", flush=True)

    A = model.generator_matrix().detach().cpu().numpy()
    M = model.propagator_matrix(DT_SAVE).cpu().numpy()
    eigs_M = np.linalg.eigvals(M)
    eigs_A = np.linalg.eigvals(A)

    mags = np.abs(eigs_M)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

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

    fig.suptitle('Stable Koopman Spectrum (KS)', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'eigenvalues.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'eigenvalues.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved eigenvalues.png/pdf", flush=True)


# =====================================================================
# Plot 5: Latent spatial modes
# =====================================================================

def plot_latent_modes(model, val_traj):
    """Spatial sensitivity of top latent dimensions via autograd."""
    print("5. Latent spatial modes...", flush=True)

    # Gather a batch of data for gradient computation
    flat = val_traj.reshape(-1, N_GRID)
    idx = torch.randperm(flat.shape[0])[:2000]
    x_batch = flat[idx]

    # Compute encoder output to rank by variance
    with torch.no_grad():
        z_all = model.ae.encoder(x_batch)  # (2000, 128)
    z_var = z_all.var(dim=0).numpy()
    top_dims = np.argsort(z_var)[::-1][:8]

    print(f"  Top 8 latent dims by variance: {top_dims.tolist()}", flush=True)
    print(f"  Variance: {z_var[top_dims].tolist()}", flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes_flat = axes.flatten()
    x_axis = np.linspace(0, L_DOMAIN, N_GRID, endpoint=False)

    for plot_idx, (ax, dim_k) in enumerate(zip(axes_flat, top_dims)):
        # Compute mean gradient dz_k/du_j via autograd
        grads = []
        chunk = 500
        for i in range(0, x_batch.shape[0], chunk):
            xi = x_batch[i:i+chunk].detach().clone().requires_grad_(True)
            z = model.ae.encoder(xi)
            z_k = z[:, dim_k].sum()
            z_k.backward()
            grads.append(xi.grad.detach().numpy())

        mean_grad = np.concatenate(grads, axis=0).mean(axis=0)  # (N_GRID,)

        # Also compute correlation
        with torch.no_grad():
            z_k_vals = z_all[:, dim_k].numpy()
        x_np = x_batch.numpy()
        corr = np.array([np.corrcoef(x_np[:, j], z_k_vals)[0, 1]
                         for j in range(N_GRID)])

        ax.plot(x_axis, mean_grad / np.abs(mean_grad).max(),
                'b-', lw=1.5, alpha=0.8, label='Gradient (norm.)')
        ax.plot(x_axis, corr, 'r--', lw=1, alpha=0.7, label='Correlation')
        ax.set_title(f'$z_{{{dim_k}}}$  (var={z_var[dim_k]:.3f})', fontsize=10)
        ax.set_xlabel('$x$')
        ax.set_ylabel('Sensitivity')
        ax.grid(True, alpha=0.2)
        if plot_idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Latent Spatial Modes (Top 8 by Variance)', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'latent_modes.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'latent_modes.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved latent_modes.png/pdf", flush=True)

    return top_dims


# =====================================================================
# Plot 6: Decomposition summary
# =====================================================================

def run_decompositions(model, traj, n_decomp=None):
    """Run full_decomposition_nd for all latent dims."""
    print("6. Decomposition analysis...", flush=True)

    from manifold_analysis import full_decomposition_nd

    flat = traj.reshape(-1, N_GRID)
    # Subsample for tractability (degree-2 on 64 vars gives ~2145 terms)
    n_sample = min(5000, flat.shape[0])
    idx = torch.randperm(flat.shape[0])[:n_sample]
    x_data = flat[idx]

    var_names = [f'u_{j}' for j in range(N_GRID)]

    if n_decomp is None:
        n_decomp = LATENT_SIZE

    decomps = []
    for ti in range(n_decomp):
        try:
            dr = full_decomposition_nd(
                model, x_data, ti, var_names,
                target_type='latent',
                max_degree=2,
                lasso_alpha=1e-4,
                h_degree=4,
            )
            decomps.append(dr)
        except Exception as e:
            print(f"  z[{ti}]: decomposition failed: {e}", flush=True)
            decomps.append({
                'label': f'z[{ti}]',
                'target_idx': ti,
                'recon_r2': float('nan'),
                'g_recon_r2': float('nan'),
                'inner': {'active_names': [], 'active_coeffs': np.array([])},
            })

        if (ti + 1) % 16 == 0 or ti == n_decomp - 1:
            n_good = sum(1 for d in decomps
                         if not np.isnan(d.get('recon_r2', float('nan')))
                         and d['recon_r2'] > 0.5)
            print(f"  Decomposed {ti + 1}/{n_decomp} dims "
                  f"({n_good} with R^2 > 0.5)", flush=True)

    return decomps, x_data


def plot_decomposition_summary(decomps):
    """Bar chart of R^2(hog) for each latent dim."""
    r2s = []
    for d in decomps:
        val = d.get('recon_r2', float('nan'))
        r2s.append(val if not np.isnan(val) else 0.0)

    r2s = np.array(r2s)
    order = np.argsort(r2s)[::-1]

    fig, ax = plt.subplots(figsize=(16, 5))
    colors = ['steelblue' if r > 0.8 else 'orange' if r > 0.5 else 'gray'
              for r in r2s[order]]
    ax.bar(range(len(r2s)), r2s[order], color=colors, edgecolor='none')
    ax.axhline(0.9, color='red', ls='--', alpha=0.5, label='$R^2 = 0.9$')
    ax.axhline(0.5, color='orange', ls='--', alpha=0.3, label='$R^2 = 0.5$')
    ax.set_xlabel('Latent dim (sorted by $R^2$)', fontsize=12)
    ax.set_ylabel('$R^2(h \\circ g)$', fontsize=12)
    ax.set_title(f'KS Dictionary Decomposition Quality '
                 f'({sum(r2s > 0.9)}/{len(r2s)} dims with $R^2 > 0.9$)',
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, len(r2s) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'decomposition_summary.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'decomposition_summary.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved decomposition_summary.png/pdf", flush=True)

    # Print dictionary table
    print("\n  Dictionary recovery table:", flush=True)
    print(f"  {'Dim':>4}  {'R2(g)':>8}  {'R2(hog)':>8}  "
          f"{'#active':>7}  top terms", flush=True)
    print("  " + "-" * 70, flush=True)
    for ti in order[:20]:
        dr = decomps[ti]
        r2g = dr.get('g_recon_r2', float('nan'))
        r2h = dr.get('recon_r2', float('nan'))
        if np.isnan(r2h) or r2h < 0.1:
            continue
        inner = dr.get('inner', {})
        active_names = inner.get('active_names', [])
        active_coeffs = inner.get('active_coeffs', np.array([]))
        n_active = len(active_names)

        if n_active > 0 and len(active_coeffs) > 0:
            pairs = sorted(zip(active_names, active_coeffs),
                           key=lambda x: -abs(x[1]))[:5]
            expr = '  '.join(f'{c:+.3f}*{n}' for n, c in pairs)
        else:
            expr = '(none)'

        print(f"  z[{ti:3d}]  {r2g:8.3f}  {r2h:8.3f}  {n_active:7d}  {expr}",
              flush=True)


# =====================================================================
# Plot 7: Quadratic coupling for best dims
# =====================================================================

def plot_quadratic_coupling(decomps):
    """Show linear and quadratic spatial structure for best-decomposed dims."""
    print("7. Quadratic coupling analysis...", flush=True)

    # Find top 4 dims by R^2
    r2s = [d.get('recon_r2', 0.0) for d in decomps]
    ranked = np.argsort(r2s)[::-1]
    targets = ranked[:4]

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))

    for row, ti in enumerate(targets):
        dr = decomps[ti]
        inner = dr.get('inner', {})
        names = inner.get('active_names', [])
        coeffs = inner.get('active_coeffs', np.array([]))
        powers = inner.get('powers', [])
        all_coeffs = inner.get('coeffs', np.zeros(1))
        all_names = inner.get('names', [])
        all_powers = inner.get('powers', [])

        # Separate linear and quadratic terms
        linear_coeffs = np.zeros(N_GRID)
        quad_matrix = np.zeros((N_GRID, N_GRID))

        for name, coeff, pw in zip(all_names, all_coeffs, all_powers):
            if abs(coeff) < 1e-10:
                continue
            total_deg = sum(pw)
            if total_deg == 1:
                # Linear term u_j
                for j, p in enumerate(pw):
                    if p == 1:
                        linear_coeffs[j] = coeff
            elif total_deg == 2:
                # Quadratic: either u_j^2 or u_i*u_j
                nonzero = [(j, p) for j, p in enumerate(pw) if p > 0]
                if len(nonzero) == 1:
                    j, p = nonzero[0]
                    quad_matrix[j, j] = coeff  # u_j^2
                elif len(nonzero) == 2:
                    j1, p1 = nonzero[0]
                    j2, p2 = nonzero[1]
                    quad_matrix[j1, j2] = abs(coeff)
                    quad_matrix[j2, j1] = abs(coeff)

        r2h = dr.get('recon_r2', 0.0)

        # Left: linear coefficient spatial profile
        ax = axes[row, 0]
        x_axis = np.arange(N_GRID)
        ax.bar(x_axis, linear_coeffs, color='steelblue', edgecolor='none')
        ax.set_xlabel('Grid point $j$')
        ax.set_ylabel('$c_j$ (linear)')
        ax.set_title(f'$z[{ti}]$ Linear Coefficients '
                     f'($R^2 = {r2h:.3f}$)', fontsize=11)
        ax.grid(True, alpha=0.2)

        # Right: quadratic coupling heatmap
        ax = axes[row, 1]
        qmax = np.abs(quad_matrix).max()
        if qmax < 1e-12:
            qmax = 1.0
        im = ax.imshow(np.abs(quad_matrix), cmap='hot', aspect='auto',
                       vmin=0, vmax=qmax, origin='lower')
        plt.colorbar(im, ax=ax, shrink=0.8, label='$|c_{ij}|$')
        ax.set_xlabel('Grid point $i$')
        ax.set_ylabel('Grid point $j$')
        ax.set_title(f'$z[{ti}]$ Quadratic Coupling $|c_{{ij}}|$',
                     fontsize=11)

        # Check for near-diagonal structure (KS nonlinearity u*u_x)
        diag_energy = 0.0
        total_energy = np.abs(quad_matrix).sum()
        for offset in [-2, -1, 0, 1, 2]:
            diag_energy += np.abs(np.diag(quad_matrix, offset)).sum()
        if total_energy > 1e-12:
            diag_frac = diag_energy / total_energy
            ax.set_title(f'$z[{ti}]$ Quadratic $|c_{{ij}}|$ '
                         f'(diag frac={diag_frac:.2f})', fontsize=11)
            print(f"  z[{ti}]: diagonal fraction of quadratic energy: "
                  f"{diag_frac:.3f}", flush=True)

    fig.suptitle('KS Quadratic Coupling Structure (Top 4 Dims)', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'coupling_structure.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'coupling_structure.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved coupling_structure.png/pdf", flush=True)


# =====================================================================
# Plot 8: Reconstruction examples
# =====================================================================

def plot_levelset_native(model, val_traj, dt, n_dims=4, decomps=None):
    """Native space-time level-set foliation for the top-variance observables.

    Six panels per latent dim (mirrors the Lorenz/Ikeda level-set figures, in the
    PDE's native x-t representation):
      (1) reconstruction h(g(u)) over time     (2) latent z_k over time
      (3) z_k vs h(g) affine check              (4) level-set bands: mean u(x) at fixed g
      (5) u(x,t) with h(g) level contours       (6) linear part of g = spatial mode
    Self-contained: decomposes its own top dims if `decomps` is not supplied, so it
    runs even under --skip-decomp.
    """
    print("7b. Native level-set foliation...", flush=True)
    from manifold_analysis import (full_decomposition_nd, eval_polynomial_nd,
                                    eval_h)

    flat = val_traj.reshape(-1, N_GRID)
    with torch.no_grad():
        zv = model.ae.encoder(flat).numpy().var(0)
    dims = list(np.argsort(zv)[::-1][:n_dims])

    xg = np.linspace(0, L_DOMAIN, N_GRID, endpoint=False)
    traj = val_traj[0].numpy()[:400]                       # one trajectory, 100 t.u.
    tt = np.arange(len(traj)) * dt
    with torch.no_grad():
        z_all = model.ae.encoder(
            torch.tensor(traj, dtype=torch.float32)).numpy()
    var_names = [f'u_{j}' for j in range(N_GRID)]
    x_sub = flat[torch.randperm(flat.shape[0])[:5000]]

    for kdim in dims:
        kdim = int(kdim)
        dr = decomps[kdim] if (decomps is not None and kdim < len(decomps)
                               and not np.isnan(decomps[kdim].get('recon_r2', np.nan))) else None
        if dr is None:
            dr = full_decomposition_nd(model, x_sub, kdim, var_names,
                                       target_type='latent', max_degree=2,
                                       lasso_alpha=1e-4, h_degree=4)
        coeffs = np.asarray(dr['inner']['coeffs'])
        powers = np.asarray(dr['inner']['powers'])
        g_s, f_s, fh_s = dr['g_vals'], dr['f_vals'], dr['f_hat']
        x_s = dr['x_np']
        linear = np.zeros(N_GRID)
        for c, p in zip(coeffs, powers):
            if p.sum() == 1:
                linear[int(np.argmax(p))] += c

        z_t = z_all[:, kdim]

        # POD projection of the attractor (for the level-set panel)
        Uc = x_s - x_s.mean(0)
        _, _, Vt = np.linalg.svd(Uc, full_matrices=False)
        aP = Uc @ Vt[:2].T

        # --- Koopman coherence: self-predictability of g vs lag, on TRUE data ---
        # If g is (near) a Koopman eigenfunction, g(t+tau) stays a linear function
        # of g(t) even past the Lyapunov time (the flow maps level sets to level
        # sets). We compare g's self-predictability R^2(tau) to that of the leading
        # POD amplitude as a generic-mode baseline; small-lag values are trivially
        # high (any smooth signal), so the discriminating region is tau ~ tau_L.
        vflat = val_traj.numpy()                              # (n_val, T, N)
        g_curves = np.stack([eval_polynomial_nd(vflat[i], coeffs, powers)
                             for i in range(vflat.shape[0])])  # (n_val, T)
        a1_curves = (vflat - vflat.reshape(-1, N_GRID).mean(0)) @ Vt[0]

        def r2_lag(sig, lag):
            x0 = sig[:, :-lag].ravel(); x1 = sig[:, lag:].ravel()
            s, ic = np.polyfit(x0, x1, 1)
            return 1.0 - np.sum((x1 - (s * x0 + ic)) ** 2) / \
                max(np.var(x1) * x1.size, 1e-12)

        lags = np.unique(np.linspace(1, int(2 * TAU_L / dt), 24).astype(int))
        lag_tau = lags * dt / TAU_L
        r2_g = np.array([r2_lag(g_curves, L_) for L_ in lags])
        r2_a1 = np.array([r2_lag(a1_curves, L_) for L_ in lags])
        slope1 = np.polyfit(g_curves[:, :-1].ravel(),
                            g_curves[:, 1:].ravel(), 1)[0]
        coh_tauL = float(np.interp(1.0, lag_tau, r2_g))

        fig, ax = plt.subplots(2, 3, figsize=(19, 10))
        ext = [0, L_DOMAIN, tt[0], tt[-1]]
        # (1) observable as a slow variable
        ax[0, 0].imshow(np.tile(z_t[:, None], (1, N_GRID)), aspect='auto',
                        origin='lower', extent=ext, cmap='coolwarm')
        ax[0, 0].set_title(f'(1) Observable $z_{{{kdim}}}$ over time (slow variable)')
        ax[0, 0].set_xlabel('x'); ax[0, 0].set_ylabel('Time')
        # (2) level sets as spatial-profile classes
        qs = np.quantile(g_s, [.1, .3, .5, .7, .9])
        band = (g_s.max() - g_s.min()) * .04
        cols = plt.cm.turbo(np.linspace(0, 1, len(qs)))
        for c_, col in zip(qs, cols):
            m = np.abs(g_s - c_) < band
            if m.sum() < 3:
                continue
            mu, sd = x_s[m].mean(0), x_s[m].std(0)
            ax[0, 1].plot(xg, mu, color=col, lw=2, label=f'g≈{c_:.2f}')
            ax[0, 1].fill_between(xg, mu - sd, mu + sd, color=col, alpha=.15)
        ax[0, 1].set_xlabel('x'); ax[0, 1].set_ylabel('u(x)')
        ax[0, 1].set_title('(2) Level sets = spatial-profile classes')
        ax[0, 1].legend(fontsize=7)
        # (3) affine check
        ax[0, 2].scatter(fh_s, f_s, s=5, alpha=.3, color='steelblue')
        lims = [min(f_s.min(), fh_s.min()), max(f_s.max(), fh_s.max())]
        ax[0, 2].plot(lims, lims, 'r--', lw=1, label='y=x (affine h)')
        ax[0, 2].set_xlabel(r'$h(g(\mathbf{u}))$')
        ax[0, 2].set_ylabel(f'$z_{{{kdim}}}$')
        ax[0, 2].set_title(f'(3) $z_{{{kdim}}}$ vs $h(g)$  '
                           f'(R²={dr["recon_r2"]:.4f})')
        ax[0, 2].legend(fontsize=8)
        # (4) level sets foliate the (POD-projected) attractor
        ax[1, 0].scatter(aP[:, 0], aP[:, 1], c='lightgray', s=5)
        for c_, col in zip(qs, cols):
            m = np.abs(g_s - c_) < band
            ax[1, 0].scatter(aP[m, 0], aP[m, 1], color=col, s=9)
        ax[1, 0].set_xlabel('POD 1'); ax[1, 0].set_ylabel('POD 2')
        ax[1, 0].set_title('(4) Level sets foliate the attractor (POD proj)')
        # (5) Koopman coherence: does the flow map level sets to level sets?
        ax[1, 1].plot(lag_tau, r2_g, 'b-o', ms=3,
                      label=f'observable $g$ (1-step slope {slope1:.3f})')
        ax[1, 1].plot(lag_tau, r2_a1, color='gray', ls='--',
                      label='leading POD amp (baseline)')
        ax[1, 1].axvline(1, color='r', ls=':', label=r'$\tau_L$')
        ax[1, 1].set_xlabel(r'lag $\tau/\tau_L$')
        ax[1, 1].set_ylabel(r'$R^2(g_{t+\tau}\,|\,g_t)$')
        ax[1, 1].set_ylim(0, 1.02)
        ax[1, 1].set_title('(5) Koopman coherence: level-set advection vs lag')
        ax[1, 1].legend(fontsize=7)
        # (6) linear part of g = spatial mode
        ax[1, 2].bar(np.arange(N_GRID), linear, color='steelblue')
        ax[1, 2].axhline(0, color='k', lw=.6)
        ax[1, 2].set_xlabel('grid point i')
        ax[1, 2].set_ylabel('coefficient of $u_i$')
        ax[1, 2].set_title('(6) Linear part of g: spatial mode')
        fig.suptitle(f'KS Level-Set Analysis  z[{kdim}]  '
                     f'R²(h∘g)={dr["recon_r2"]:.4f}   '
                     f'coherence R²@$\\tau_L$={coh_tauL:.2f}', fontsize=13)
        fig.tight_layout()
        fig.savefig(OUTDIR / f'levelset_z{kdim}.png', dpi=140,
                    bbox_inches='tight')
        fig.savefig(OUTDIR / f'levelset_z{kdim}.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved levelset_z{kdim}.png (R²(h∘g)={dr['recon_r2']:.4f}, "
              f"coherence R²@τ_L={coh_tauL:.2f})", flush=True)


def plot_reconstruction(model, val_traj):
    """Encoder->decoder round-trip reconstruction quality."""
    print("8. Reconstruction examples...", flush=True)

    flat = val_traj.reshape(-1, N_GRID)
    idx = torch.randperm(flat.shape[0])[:4]
    samples = flat[idx]

    with torch.no_grad():
        z, recon = model.ae(samples)

    x_axis = np.linspace(0, L_DOMAIN, N_GRID, endpoint=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        u_orig = samples[i].numpy()
        u_recon = recon[i].numpy()
        mse_i = np.mean((u_orig - u_recon)**2)

        ax.plot(x_axis, u_orig, 'b-', lw=1.5, alpha=0.8, label='Original')
        ax.plot(x_axis, u_recon, 'r--', lw=1.5, alpha=0.8,
                label='Reconstructed')
        ax.fill_between(x_axis, u_orig, u_recon, alpha=0.1, color='gray')
        ax.set_title(f'Snapshot {i + 1}  (MSE = {mse_i:.6f})', fontsize=11)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(x)$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # Overall reconstruction MSE
    with torch.no_grad():
        all_flat = val_traj.reshape(-1, N_GRID)
        sub_idx = torch.randperm(all_flat.shape[0])[:5000]
        _, all_recon = model.ae(all_flat[sub_idx])
        overall_mse = nn.MSELoss()(all_recon, all_flat[sub_idx]).item()

    fig.suptitle(f'KS Autoencoder Reconstruction '
                 f'(overall MSE = {overall_mse:.6f})', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'reconstruction.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'reconstruction.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved reconstruction.png/pdf (overall MSE = {overall_mse:.6f})",
          flush=True)

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

    fig.suptitle('KS Stable Koopman Training', fontsize=14)
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
        description='KS equation with stable Koopman KAN model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--mid', type=int, default=MID_SIZE,
                        help='Hidden dictionary-layer width (default %(default)s)')
    parser.add_argument('--rollout', type=int, default=ROLLOUT_HORIZON,
                        help='Max multi-step rollout length K_max (1 = one-step)')
    parser.add_argument('--rollout-warmup', type=int, default=ROLLOUT_WARMUP,
                        help='Epochs of pure one-step training before ramping K')
    parser.add_argument('--discount', type=float, default=ROLLOUT_DISCOUNT,
                        help='Per-step discount gamma^(j-1) on multi-step terms')
    parser.add_argument('--outdir', default=str(OUTDIR),
                        help='Output directory for checkpoint/figures')
    parser.add_argument('--data', default=str(OUTDIR / 'ks_data.pt'),
                        help='Cached KS data path (reused across runs)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Load existing checkpoint instead of training')
    parser.add_argument('--skip-decomp', action='store_true',
                        help='Skip the decomposition analysis (slow for 128 dims)')
    args = parser.parse_args()

    # Apply CLI overrides to module-level config (functions read these at call time)
    MID_SIZE = args.mid
    ROLLOUT_HORIZON = args.rollout
    ROLLOUT_WARMUP = args.rollout_warmup
    ROLLOUT_DISCOUNT = args.discount
    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    print("=" * 70, flush=True)
    print("KS Stable Koopman with KANLinear Autoencoder", flush=True)
    print("=" * 70, flush=True)
    print(f"  Domain: L={L_DOMAIN}, N={N_GRID}, dt_save={DT_SAVE}", flush=True)
    print(f"  Model:  KANAutoEncoder({INPUT_SIZE}, {MID_SIZE}, {LATENT_SIZE})",
          flush=True)
    print(f"  Generator: A = K - L^T L  (stable, latent_dim={LATENT_SIZE})",
          flush=True)
    print(f"  Device: {args.device}", flush=True)
    print(f"  Lyapunov time: tau_L ~ {TAU_L:.1f} time units", flush=True)

    # ---- Generate KS data ----
    print("\n--- Generating KS data ---", flush=True)
    ckpt_path = OUTDIR / 'ks_stable_model.pt'
    data_path = Path(args.data)

    if data_path.exists():
        print(f"  Loading cached data from {data_path}", flush=True)
        traj = torch.load(data_path, map_location='cpu', weights_only=True)
    else:
        traj = simulate_ks(
            n_traj=N_TRAJ, L=L_DOMAIN, N=N_GRID,
            dt_sim=DT_SIM, dt_save=DT_SAVE,
            burn_in=BURN_IN, T=T_TOTAL, seed=42,
        )
        torch.save(traj, data_path)
        print(f"  Saved KS data to {data_path}", flush=True)

    n_traj_actual, n_steps, n_grid = traj.shape
    print(f"  Data shape: {traj.shape} "
          f"(n_traj={n_traj_actual}, n_steps={n_steps}, N={n_grid})", flush=True)
    print(f"  u range: [{traj.min():.2f}, {traj.max():.2f}]", flush=True)

    traj_cpu = traj.clone()
    train_traj = traj[:N_TRAIN]
    val_traj = traj[N_TRAIN:]

    # ---- Train or load model ----
    if args.skip_train and ckpt_path.exists():
        print(f"\n--- Loading checkpoint: {ckpt_path} ---", flush=True)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        mid_ckpt = ckpt.get('config', {}).get('mid_size', MID_SIZE)
        ae = KANAutoEncoder(
            INPUT_SIZE, mid_ckpt, LATENT_SIZE,
            grid_range_l1=[-5, 5],
            grid_range_l2=[-10, 10],
        )
        model = DeepKoopmanCTStable(ae, LATENT_SIZE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        history = ckpt.get('history', None)
        if history is not None:
            print(f"  Loaded model (final val loss: "
                  f"{history['val_total'][-1]:.5f})", flush=True)
    else:
        print(f"\n--- Training ({args.epochs} epochs) ---", flush=True)
        ae = KANAutoEncoder(
            INPUT_SIZE, MID_SIZE, LATENT_SIZE,
            grid_range_l1=[-5, 5],
            grid_range_l2=[-10, 10],
        ).to(args.device)
        traj_device = traj.to(args.device)
        model, history = train_model(ae, traj_device, epochs=args.epochs,
                                     lr=LR, device=args.device,
                                     rollout_horizon=args.rollout,
                                     rollout_warmup=args.rollout_warmup,
                                     rollout_discount=args.discount)

        torch.save({
            'model_state_dict': model.state_dict(),
            'dt': DT_SAVE,
            'history': history,
            'config': {
                'input_size': INPUT_SIZE,
                'mid_size': MID_SIZE,
                'latent_size': LATENT_SIZE,
                'L': L_DOMAIN,
                'N': N_GRID,
                'dt_save': DT_SAVE,
                'epochs': args.epochs,
                'lr': LR,
                'batch_size': BATCH_SIZE,
                'rollout_horizon': args.rollout,
                'rollout_warmup': args.rollout_warmup,
                'rollout_discount': args.discount,
                'grid_range_l1': [-5, 5],
                'grid_range_l2': [-10, 10],
                'seed': 42,
            },
        }, ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path}", flush=True)

        # Loss curves
        plot_loss_curves(history)

    # ---- Save config (training runs only: a --skip-train replot must not
    # overwrite the training record with this invocation's CLI defaults) ----
    if not args.skip_train:
        config = {
            'L': L_DOMAIN, 'N': N_GRID, 'dt_sim': DT_SIM, 'dt_save': DT_SAVE,
            'burn_in': BURN_IN, 'T': T_TOTAL, 'n_traj': N_TRAJ,
            'input_size': INPUT_SIZE, 'mid_size': MID_SIZE,
            'latent_size': LATENT_SIZE,
            'epochs': args.epochs, 'lr': LR,
            'batch_size': BATCH_SIZE, 'grad_clip': GRAD_CLIP,
            'weight_decay': WEIGHT_DECAY,
            'gamma_predx': GAMMA_PREDX, 'alpha_latent': ALPHA_LATENT,
            'beta_recon': BETA_RECON,
            'rollout_horizon': args.rollout, 'rollout_warmup': args.rollout_warmup,
            'rollout_discount': args.discount,
            'grid_range_l1': [-5, 5], 'grid_range_l2': [-10, 10],
            'seed': 42,
            'n_train': N_TRAIN, 'n_val': N_VAL,
            'tau_L': TAU_L, 'lambda_max_ks': LAMBDA_MAX_KS,
            'device': args.device,
        }
        with open(OUTDIR / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    # ---- Generate all plots ----
    print("\n--- Generating plots ---", flush=True)

    # 1. Space-time rollout
    plot_spacetime_rollout(model, val_traj, DT_SAVE)

    # 2. Snapshot comparisons
    plot_snapshots(model, val_traj, DT_SAVE)

    # 3. NRMSE over time
    mean_nrmse, horizon = plot_nrmse(model, val_traj, DT_SAVE)

    # 4. Eigenvalue spectrum
    plot_eigenvalues(model)

    # 5. Latent spatial modes
    top_dims = plot_latent_modes(model, val_traj)

    # 6-7. Decomposition and quadratic coupling (can be slow)
    decomps = None
    if not args.skip_decomp:
        decomps, x_data = run_decompositions(model, train_traj)
        plot_decomposition_summary(decomps)

        # 7. Quadratic coupling for best dims
        plot_quadratic_coupling(decomps)
    else:
        print("  Skipping decomposition analysis (--skip-decomp)", flush=True)

    # 7b. Native space-time level-set foliation for top observables
    #     (self-contained: decomposes its own top dims, reuses decomps if present)
    plot_levelset_native(model, val_traj, DT_SAVE, n_dims=4, decomps=decomps)

    # 8. Reconstruction examples
    recon_mse = plot_reconstruction(model, val_traj)

    # ---- Summary ----
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model: KANAutoEncoder({INPUT_SIZE}, {MID_SIZE}, {LATENT_SIZE}) "
          f"+ StableKoopman", flush=True)
    print(f"  KS domain: L={L_DOMAIN}, N={N_GRID}", flush=True)
    print(f"  Prediction horizon: {horizon:.2f} tau_L "
          f"({horizon * TAU_L:.1f} time units)", flush=True)
    print(f"  Reconstruction MSE: {recon_mse:.6f}", flush=True)
    if history is not None:
        print(f"  Final train loss: {history['train_total'][-1]:.5f}", flush=True)
        print(f"  Final val loss:   {history['val_total'][-1]:.5f}", flush=True)
    print(f"  All results saved to: {OUTDIR.resolve()}/", flush=True)
    print("=" * 70, flush=True)
