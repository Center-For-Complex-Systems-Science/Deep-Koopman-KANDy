"""
Arnold Cat Map with ERGODIC Koopman propagator.

Same as cat_map_stable.py but with one key change: the propagator has a
FIXED eigenvalue at lambda=1 corresponding to the constant eigenfunction.

For a mixing system, the constant function f(x,y) = 1 is the ONLY
Koopman eigenfunction. This experiment pins that eigenvalue by construction:
  - One latent dimension is always = 1 (constant observable)
  - The propagator has a 1x1 block with eigenvalue exactly 1
  - The remaining (n-1) dims have the stable constraint |lambda| <= 1

The expected result: the model learns that long-time prediction converges
to the ergodic mean (correct!) while short-time prediction decays at the
Lyapunov rate (also correct). This is the complete Koopman picture for mixing.

Scientific question: what happens when a finite-dimensional Koopman model is
applied to a system with no discrete spectrum? Expected outcomes:
    1. Phase space: model rollout will NOT fill the torus uniformly
    2. Prediction: error grows exponentially at the Lyapunov rate (~1 step)
    3. Eigenvalues: all INSIDE the unit circle (damping admits limitation)
    4. Dictionary: R^2(h o g) << 1 (no polynomial eigenfunctions exist)
    5. Mixing: model fails to reproduce stretching/folding beyond 1-2 steps

This is a NEGATIVE RESULT confirming theoretical limitations of finite-dimensional
Koopman methods for systems with continuous spectrum.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from pathlib import Path

from kan import KANLinear
from training import set_seed

OUTDIR = Path('results/cat_map_ergodic')

# Lyapunov exponent of the Arnold cat map
LAMBDA_MAX = np.log((3 + np.sqrt(5)) / 2)  # ~0.9624

# =====================================================================
# Cat map simulation
# =====================================================================

CAT_MATRIX = np.array([[2, 1], [1, 1]], dtype=np.float64)


def simulate_cat_map(n_traj=200, n_steps=500, seed=42):
    """Generate Arnold cat map trajectories on [0,1)^2.

    Returns:
        traj: (n_traj, n_steps, 2) tensor with values in [0,1)
    """
    np.random.seed(seed)
    data = np.zeros((n_traj, n_steps, 2))
    for i in range(n_traj):
        xy = np.random.rand(2)
        for t in range(n_steps):
            data[i, t] = xy
            xy = (CAT_MATRIX @ xy) % 1.0
    return torch.tensor(data, dtype=torch.float32)


# =====================================================================
# Dataset
# =====================================================================

class MapPairDataset(Dataset):
    """(state_n, state_{n+1}) pairs from discrete map trajectories."""
    def __init__(self, traj):
        # traj: (n_traj, n_steps, 2) with values in [0,1)
        self.x = traj[:, :-1].reshape(-1, 2)
        self.y = traj[:, 1:].reshape(-1, 2)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# =====================================================================
# Torus embedding and angular loss
# =====================================================================

def torus_embed(xy):
    """(batch, 2) in [0,1)^2 -> (batch, 4): [cos(2pi*x), sin(2pi*x), cos(2pi*y), sin(2pi*y)]"""
    theta = 2 * np.pi * xy
    return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)


def angular_mse(pred, true):
    """MSE on the torus [0,1)^2 -- wraps around boundaries."""
    diff = pred - true
    diff = diff - torch.round(diff)  # wrap to [-0.5, 0.5)
    return (diff ** 2).mean()


# =====================================================================
# Stable propagator: M = expm(K - L^T L), |eigenvalues| <= 1
# =====================================================================

class ErgodicPropagator(nn.Module):
    """Discrete-time propagator with one FIXED eigenvalue at lambda=1.

    Structure:
        M = block_diag(M_stable, [1])

    where M_stable = expm(K - L^T L) is the (dim-1)x(dim-1) stable block
    (all |eigenvalues| <= 1), and the last dimension has eigenvalue exactly 1
    (the constant/ergodic eigenfunction).

    The last latent dim is always = 1 (appended by the encoder), and the
    propagator preserves it exactly.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dim_stable = dim - 1  # the stable block size
        n = self.dim_stable
        n_skew = n * (n - 1) // 2
        self.skew_params = nn.Parameter(0.01 * torch.randn(n_skew))
        self.L_diag = nn.Parameter(0.01 * torch.randn(n))
        n_lower = n * (n - 1) // 2
        self.L_lower = nn.Parameter(0.01 * torch.randn(n_lower))

    def _get_skew(self):
        n = self.dim_stable
        K = torch.zeros(n, n, device=self.skew_params.device)
        idx = torch.triu_indices(n, n, offset=1)
        K[idx[0], idx[1]] = self.skew_params
        return K - K.T

    def _get_L(self):
        n = self.dim_stable
        L = torch.zeros(n, n, device=self.L_diag.device)
        L[range(n), range(n)] = self.L_diag
        idx = torch.tril_indices(n, n, offset=-1)
        L[idx[0], idx[1]] = self.L_lower
        return L

    def generator_matrix(self):
        """Generator for the stable block only (dim-1 x dim-1)."""
        K = self._get_skew()
        L = self._get_L()
        return K - L.T @ L

    def get_stable_matrix(self):
        """Return the (dim-1)x(dim-1) stable block."""
        return torch.matrix_exp(self.generator_matrix())

    def get_matrix(self):
        """Return the full dim x dim propagator with lambda=1 in last entry."""
        M_stable = self.get_stable_matrix()
        M = torch.zeros(self.dim, self.dim, device=M_stable.device)
        M[:self.dim_stable, :self.dim_stable] = M_stable
        M[-1, -1] = 1.0  # fixed ergodic eigenvalue
        return M

    def forward(self, z):
        return z @ self.get_matrix().T


# =====================================================================
# Encoder / Decoder (torus-aware, like standard_map_unitary.py)
# =====================================================================

class CatMapEncoder(nn.Module):
    """Two-layer KAN encoder for the torus [0,1)^2.

    Input: (x, y) in [0,1)^2
    Embeds as (cos(2pi*x), sin(2pi*x), cos(2pi*y), sin(2pi*y))
    then applies two KAN layers.

    Appends a constant 1 as the last latent dimension — the ergodic
    eigenfunction. The KAN outputs (latent_size - 1) learned dims.
    """
    def __init__(self, mid_size, latent_size, grid_size=5, spline_order=3):
        super().__init__()
        self.latent_size = latent_size
        kan_kw = dict(grid_size=grid_size, spline_order=spline_order,
                      grid_range=[-1.5, 1.5])
        self.kan1 = KANLinear(4, mid_size, **kan_kw)
        # Output latent_size - 1 learned dims (the last dim is the constant 1)
        self.kan2 = KANLinear(mid_size, latent_size - 1,
                              grid_size=grid_size, spline_order=spline_order,
                              grid_range=[-5, 5])

    def forward(self, xy):
        h = torus_embed(xy)
        z_learned = self.kan2(self.kan1(h))  # (batch, latent_size - 1)
        # Append constant 1 — the ergodic eigenfunction
        ones = torch.ones(z_learned.shape[0], 1, device=z_learned.device)
        return torch.cat([z_learned, ones], dim=-1)  # (batch, latent_size)


class CatMapDecoder(nn.Module):
    """Two-layer KAN decoder back to torus [0,1)^2.

    Outputs (h1, h2, h3, h4) -> x = atan2(h2, h1)/(2pi) mod 1,
                                  y = atan2(h4, h3)/(2pi) mod 1.
    """
    def __init__(self, latent_size, mid_size, grid_size=5, spline_order=3):
        super().__init__()
        kan_kw = dict(grid_size=grid_size, spline_order=spline_order,
                      grid_range=[-5, 5])
        self.kan1 = KANLinear(latent_size, mid_size, **kan_kw)
        self.kan2 = KANLinear(mid_size, 4,
                              grid_size=grid_size, spline_order=spline_order,
                              grid_range=[-5, 5])

    def forward(self, z):
        h = self.kan2(self.kan1(z))
        x = torch.atan2(h[:, 1], h[:, 0]) / (2 * np.pi)
        y = torch.atan2(h[:, 3], h[:, 2]) / (2 * np.pi)
        x = x % 1.0
        y = y % 1.0
        return torch.stack([x, y], dim=-1)


# =====================================================================
# Full model
# =====================================================================

class CatMapKoopman(nn.Module):
    """Discrete Koopman with stable propagator for the Arnold cat map."""
    def __init__(self, mid_size, latent_dim, grid_size=5, spline_order=3):
        super().__init__()
        self.encoder = CatMapEncoder(mid_size, latent_dim, grid_size, spline_order)
        self.decoder = CatMapDecoder(latent_dim, mid_size, grid_size, spline_order)
        self.prop = ErgodicPropagator(latent_dim)
        self.latent_dim = latent_dim

    def forward_one_step(self, xy):
        z = self.encoder(xy)
        xy_rec = self.decoder(z)
        z_next = self.prop(z)
        xy_next = self.decoder(z_next)
        return {
            'z_n': z, 'xy_n_rec': xy_rec,
            'z_np1_pred': z_next, 'xy_np1_pred': xy_next,
        }

    def koopman_matrix(self):
        return self.prop.get_matrix().detach()


# =====================================================================
# Wrapper for manifold_analysis compatibility
# =====================================================================

class _AEWrapper:
    """Minimal wrapper so manifold_analysis sees model.ae.encoder(x).

    The encoder here takes 4D torus-embedded inputs (cos/sin), NOT raw (x,y).
    This is what full_decomposition_nd needs when var_names are torus coords.
    """
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder


class CatMapKoopmanForAnalysis:
    """Adapter that presents the CatMapKoopman model to manifold_analysis.

    model.ae.encoder expects (batch, 4) torus-embedded input.
    """
    def __init__(self, model):
        self._model = model
        self.ae = _AEWrapper(
            encoder=model.encoder._torus_encoder,
            decoder=model.decoder,
        )


class _TorusEncoder(nn.Module):
    """Encoder that takes 4D torus coords directly (no embedding step).
    Appends constant 1 for the ergodic dim."""
    def __init__(self, kan1, kan2):
        super().__init__()
        self.kan1 = kan1
        self.kan2 = kan2

    def forward(self, h):
        z_learned = self.kan2(self.kan1(h))
        ones = torch.ones(z_learned.shape[0], 1, device=z_learned.device)
        return torch.cat([z_learned, ones], dim=-1)


# =====================================================================
# Training
# =====================================================================

# --- Hyperparameters ---
N_TRAJ = 200
N_STEPS = 500
MID_SIZE = 24
LATENT_DIM = 1024
GRID_SIZE = 5
SPLINE_ORDER = 3
EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 2048
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
RECON_WEIGHT = 0.5


def train(model, traj, epochs=EPOCHS, lr=LR, device='cpu'):
    """Train the CatMapKoopman model."""
    n_train = int(0.8 * traj.shape[0])
    train_dl = DataLoader(MapPairDataset(traj[:n_train]),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(MapPairDataset(traj[n_train:]),
                        batch_size=BATCH_SIZE)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CatMapKoopman [4,{MID_SIZE},{LATENT_DIM}]: {n_params:,} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    history = {'train': [], 'val': []}

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xy_n, xy_np1 in train_dl:
            xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
            out = model.forward_one_step(xy_n)
            z_true = model.encoder(xy_np1)

            loss_pred = angular_mse(out['xy_np1_pred'], xy_np1)
            loss_latent = mse(out['z_np1_pred'], z_true)
            loss_recon = angular_mse(out['xy_n_rec'], xy_n)
            loss = loss_pred + loss_latent + RECON_WEIGHT * loss_recon

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())
        scheduler.step()

        history['train'].append(float(np.mean(losses)))

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            model.eval()
            vloss = []
            with torch.no_grad():
                for xy_n, xy_np1 in val_dl:
                    xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
                    out = model.forward_one_step(xy_n)
                    z_true = model.encoder(xy_np1)
                    l = (angular_mse(out['xy_np1_pred'], xy_np1)
                         + mse(out['z_np1_pred'], z_true)
                         + RECON_WEIGHT * angular_mse(out['xy_n_rec'], xy_n))
                    vloss.append(l.item())
            history['val'].append(float(np.mean(vloss)))

            M = model.koopman_matrix().cpu().numpy()
            eigs = np.linalg.eigvals(M)
            mags = np.abs(eigs)
            print(f"  Ep {ep:3d} | train {history['train'][-1]:.5f} | "
                  f"val {history['val'][-1]:.5f} | "
                  f"|lambda| [{mags.min():.4f}, {mags.max():.4f}]", flush=True)

    return model.cpu().eval(), history


# =====================================================================
# 1. Phase space rollout
# =====================================================================

def batched_rollout(model, ics_tensor, n_steps, batch_size=512):
    """Autoregressive rollout for many ICs in parallel (batched).

    Args:
        model: CatMapKoopman
        ics_tensor: (n_ics, 2) initial conditions in [0,1)^2
        n_steps: number of steps to roll out
        batch_size: max ICs to process in one batch (memory control)

    Returns:
        all_pts: (n_ics * n_steps, 2) numpy array of all visited points
    """
    n_ics = ics_tensor.shape[0]
    all_pts = []
    with torch.no_grad():
        for start in range(0, n_ics, batch_size):
            end = min(start + batch_size, n_ics)
            xy = ics_tensor[start:end].clone()  # (batch, 2)
            batch_pts = []
            for t in range(n_steps):
                batch_pts.append(xy.numpy().copy())
                out = model.forward_one_step(xy)
                xy = out['xy_np1_pred']
            all_pts.append(np.concatenate(batch_pts, axis=0))
            done = min(end, n_ics)
            if done % 200 == 0 or done == n_ics:
                print(f"    {done}/{n_ics} ICs done", flush=True)
    return np.concatenate(all_pts, axis=0)


def plot_phase_space(model, outdir):
    """Plot true vs model phase space rollouts."""
    print("1. Phase space rollout...", flush=True)
    n_ics = 200
    n_steps = 5000

    # True phase space (vectorized)
    np.random.seed(99)
    ics = np.random.rand(n_ics, 2)
    true_pts = []
    xy = ics.copy()
    for t in range(n_steps):
        true_pts.append(xy.copy())
        xy = (xy @ CAT_MATRIX.T) % 1.0  # vectorized: (n_ics, 2)
    true_pts = np.concatenate(true_pts, axis=0)

    # Model rollout (batched)
    ics_tensor = torch.tensor(ics, dtype=torch.float32)
    model_pts = batched_rollout(model, ics_tensor, n_steps, batch_size=200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sub = np.random.choice(len(true_pts), min(200000, len(true_pts)), replace=False)
    axes[0].scatter(true_pts[sub, 0], true_pts[sub, 1], s=0.05, alpha=0.2,
                    c='black', rasterized=True)
    axes[0].set_title('True Cat Map Phase Space', fontsize=13)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')

    sub2 = np.random.choice(len(model_pts), min(200000, len(model_pts)), replace=False)
    axes[1].scatter(model_pts[sub2, 0], model_pts[sub2, 1], s=0.05, alpha=0.2,
                    c='steelblue', rasterized=True)
    axes[1].set_title('Stable Koopman Rollout', fontsize=13)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('$x$')

    fig.suptitle(f'Arnold Cat Map: {n_ics} ICs x {n_steps} steps\n'
                 f'Purely continuous spectrum -- model cannot sustain ergodic coverage',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / 'phase_space.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / 'phase_space.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved phase_space.png/pdf", flush=True)


# =====================================================================
# 2. Short-horizon prediction accuracy
# =====================================================================

def plot_prediction_accuracy(model, outdir):
    """Plot angular prediction error vs horizon, compared to persistence."""
    print("2. Prediction accuracy vs horizon...", flush=True)
    n_ics = 1000
    max_horizon = 20

    np.random.seed(77)
    ics = np.random.rand(n_ics, 2).astype(np.float32)

    # True k-step trajectories (vectorized)
    true_trajs = np.zeros((n_ics, max_horizon + 1, 2))
    xy = ics.copy()
    for t in range(max_horizon + 1):
        true_trajs[:, t] = xy
        if t < max_horizon:
            xy = (xy @ CAT_MATRIX.T) % 1.0

    # Model k-step predictions (batched autoregressive)
    model_trajs = np.zeros((n_ics, max_horizon + 1, 2))
    with torch.no_grad():
        xy = torch.tensor(ics)  # (n_ics, 2)
        for t in range(max_horizon + 1):
            model_trajs[:, t] = xy.numpy()
            if t < max_horizon:
                out = model.forward_one_step(xy)
                xy = out['xy_np1_pred']

    # Compute angular error at each horizon
    horizons = np.arange(1, max_horizon + 1)
    model_errors = np.zeros(max_horizon)
    persist_errors = np.zeros(max_horizon)

    for k in range(max_horizon):
        # Model error
        diff_m = model_trajs[:, k + 1] - true_trajs[:, k + 1]
        diff_m = diff_m - np.round(diff_m)
        model_errors[k] = np.mean(diff_m ** 2)

        # Persistence error (predict = initial condition)
        diff_p = true_trajs[:, 0] - true_trajs[:, k + 1]
        diff_p = diff_p - np.round(diff_p)
        persist_errors[k] = np.mean(diff_p ** 2)

    # Theoretical random-guess baseline: for uniform on [0,1), E[(x-y mod 1)^2] = 1/12 per dim
    random_baseline = 1.0 / 6.0  # two dimensions

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(horizons, model_errors, 'o-', color='steelblue', lw=2, ms=6,
                label='Model prediction')
    ax.semilogy(horizons, persist_errors, 's--', color='gray', lw=1.5, ms=5,
                alpha=0.6, label='Persistence (predict $x_0$)')
    ax.axhline(random_baseline, color='red', ls=':', alpha=0.5,
               label=f'Random guess (1/6 = {random_baseline:.4f})')

    # Mark Lyapunov time
    lyap_time = 1.0 / LAMBDA_MAX
    ax.axvline(lyap_time, color='green', ls='--', alpha=0.5,
               label=f'Lyapunov time = {lyap_time:.2f} steps')

    ax.set_xlabel('Prediction horizon (steps)', fontsize=12)
    ax.set_ylabel('Angular MSE', fontsize=12)
    ax.set_title('Arnold Cat Map: Prediction Error vs Horizon\n'
                 f'$\\lambda_{{\\max}} = {LAMBDA_MAX:.3f}$, '
                 f'Lyapunov time $\\approx$ {lyap_time:.2f} steps',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, which='both')
    ax.set_xlim(0.5, max_horizon + 0.5)
    fig.tight_layout()
    fig.savefig(outdir / 'prediction_accuracy.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / 'prediction_accuracy.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved prediction_accuracy.png/pdf", flush=True)

    return model_errors, persist_errors


# =====================================================================
# 3. Eigenvalue spectrum
# =====================================================================

def plot_eigenvalues(model, outdir):
    """Eigenvalues on the complex plane + magnitude histogram."""
    print("3. Eigenvalue spectrum...", flush=True)
    M = model.koopman_matrix().cpu().numpy()
    eigs = np.linalg.eigvals(M)
    mags = np.abs(eigs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: complex plane
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.3, label='Unit circle')
    ax.scatter(eigs.real, eigs.imag, s=60, c='steelblue', edgecolors='black',
               linewidths=0.5, zorder=3)
    ax.set_xlabel('Re($\\lambda$)', fontsize=12)
    ax.set_ylabel('Im($\\lambda$)', fontsize=12)
    ax.set_title('Koopman Eigenvalues', fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

    # Right: magnitude histogram
    ax = axes[1]
    ax.hist(mags, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(1.0, color='red', ls='--', lw=1.5, alpha=0.7, label='$|\\lambda|=1$')
    ax.set_xlabel('$|\\lambda|$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Eigenvalue Magnitude Distribution', fontsize=13)
    ax.legend(fontsize=10)

    fig.suptitle('Arnold Cat Map: Stable Koopman Eigenvalues\n'
                 'Area-preserving system with continuous spectrum -- '
                 'model should damp ($|\\lambda|<1$)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / 'eigenvalues.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / 'eigenvalues.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  |lambda| range: [{mags.min():.6f}, {mags.max():.6f}]", flush=True)
    print(f"  Mean |lambda|: {mags.mean():.6f}", flush=True)
    n_near_unit = np.sum(mags > 0.99)
    print(f"  # eigenvalues with |lambda| > 0.99: {n_near_unit}/{len(mags)}", flush=True)
    print("  Saved eigenvalues.png/pdf", flush=True)

    return eigs, mags


# =====================================================================
# 4. Eigenvalue magnitude decay
# =====================================================================

def plot_eigenvalue_decay(eigs, outdir):
    """Sorted eigenvalue magnitudes -- shows damping profile."""
    print("4. Eigenvalue magnitude decay...", flush=True)
    mags = np.sort(np.abs(eigs))[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(mags) + 1), mags, 'o-', color='steelblue', ms=5, lw=1.5)
    ax.axhline(1.0, color='red', ls='--', alpha=0.5, label='$|\\lambda|=1$ (unit circle)')
    ax.set_xlabel('Index (sorted)', fontsize=12)
    ax.set_ylabel('$|\\lambda|$', fontsize=12)
    ax.set_title('Arnold Cat Map: Eigenvalue Magnitude Decay\n'
                 'All $|\\lambda|<1$ implies damping -- '
                 'model honestly admits it cannot represent continuous spectrum',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.5, len(mags) + 0.5)
    fig.tight_layout()
    fig.savefig(outdir / 'eigenvalue_decay.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / 'eigenvalue_decay.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved eigenvalue_decay.png/pdf", flush=True)


# =====================================================================
# 5. Rollout error growth
# =====================================================================

def plot_error_growth(model, outdir):
    """Trajectory-wise angular error vs step number with Lyapunov reference."""
    print("5. Error growth analysis...", flush=True)
    n_ics = 100
    n_steps = 30

    np.random.seed(55)
    ics = np.random.rand(n_ics, 2).astype(np.float32)

    errors = np.zeros((n_ics, n_steps))

    with torch.no_grad():
        xy_true = ics.copy()  # (n_ics, 2)
        xy_model = torch.tensor(ics)  # (n_ics, 2)
        for t in range(n_steps):
            # Advance truth (vectorized)
            xy_true = (xy_true @ CAT_MATRIX.T) % 1.0
            # Advance model (batched)
            out = model.forward_one_step(xy_model)
            xy_model = out['xy_np1_pred']
            # Angular error per IC
            diff = xy_model.numpy() - xy_true.astype(np.float32)
            diff = diff - np.round(diff)
            errors[:, t] = np.sqrt(np.mean(diff ** 2, axis=1))

    steps = np.arange(1, n_steps + 1)
    mean_err = errors.mean(axis=0)

    # Theoretical exponential growth (saturates at ~0.29 = sqrt(1/12))
    # E[|x - y|] for uniform on [0,1) is 1/sqrt(12) ~ 0.289
    saturation = 1.0 / np.sqrt(12)
    theory = np.minimum(errors[0, 0] * np.exp(LAMBDA_MAX * (steps - 1)),
                        saturation)

    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(n_ics):
        ax.plot(steps, errors[i], color='steelblue', alpha=0.1, lw=0.5)
    ax.plot(steps, mean_err, color='darkblue', lw=2.5, label='Mean error')
    ax.plot(steps, theory, 'r--', lw=2, alpha=0.7,
            label=f'$\\sim e^{{\\lambda_{{\\max}} n}}$, $\\lambda_{{\\max}}={LAMBDA_MAX:.3f}$')
    ax.axhline(saturation, color='gray', ls=':', alpha=0.5,
               label=f'Saturation ($1/\\sqrt{{12}} \\approx {saturation:.3f}$)')

    ax.set_xlabel('Step number', fontsize=12)
    ax.set_ylabel('RMSE (angular)', fontsize=12)
    ax.set_title('Arnold Cat Map: Rollout Error Growth\n'
                 '100 ICs, thin lines = individual, thick = mean',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, which='both')
    ax.set_xlim(0.5, n_steps + 0.5)
    fig.tight_layout()
    fig.savefig(outdir / 'error_growth.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / 'error_growth.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved error_growth.png/pdf", flush=True)


# =====================================================================
# 6. Mixing visualization
# =====================================================================

def plot_mixing(model, outdir):
    """Visualize stretching/folding: colored grid under cat map vs model."""
    print("6. Mixing visualization...", flush=True)
    n_grid = 50
    steps_to_show = [1, 2, 3, 5, 10, 20]

    # Initial colored grid on [0,1)^2
    gx = np.linspace(0, 1, n_grid, endpoint=False)
    gy = np.linspace(0, 1, n_grid, endpoint=False)
    xx, yy = np.meshgrid(gx, gy)
    pts0 = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)

    # Color by initial position (HSV based on angle from center)
    cx = pts0[:, 0] - 0.5
    cy = pts0[:, 1] - 0.5
    hue = (np.arctan2(cy, cx) / (2 * np.pi) + 0.5) % 1.0
    sat = np.ones_like(hue) * 0.9
    val = np.ones_like(hue) * 0.9
    colors = hsv_to_rgb(np.stack([hue, sat, val], axis=-1))

    # Compute true trajectories
    max_step = max(steps_to_show)
    true_trajs = [pts0.copy()]
    pts = pts0.copy()
    for t in range(max_step):
        pts = (CAT_MATRIX @ pts.T).T % 1.0
        true_trajs.append(pts.copy())

    # Compute model trajectories (batched -- all 2500 points at once)
    model_trajs = [pts0.copy().astype(np.float32)]
    with torch.no_grad():
        xy = torch.tensor(pts0, dtype=torch.float32)
        for t in range(max_step):
            out = model.forward_one_step(xy)
            xy = out['xy_np1_pred']
            model_trajs.append(xy.numpy().copy())

    # Plot: 2 rows x len(steps_to_show) columns
    # Top row: true, bottom row: model
    n_panels = len(steps_to_show)
    fig, axes = plt.subplots(2, n_panels, figsize=(3.5 * n_panels, 7))

    for col, step in enumerate(steps_to_show):
        # True
        ax = axes[0, col]
        ax.scatter(true_trajs[step][:, 0], true_trajs[step][:, 1],
                   c=colors, s=3, alpha=0.8, rasterized=True)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'True, step {step}', fontsize=10)
        if col == 0:
            ax.set_ylabel('True', fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

        # Model
        ax = axes[1, col]
        m_pts = model_trajs[step]
        # Clip to [0,1) for display (model may go slightly out)
        m_pts_clip = m_pts % 1.0
        ax.scatter(m_pts_clip[:, 0], m_pts_clip[:, 1],
                   c=colors, s=3, alpha=0.8, rasterized=True)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Model, step {step}', fontsize=10)
        if col == 0:
            ax.set_ylabel('Model', fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Arnold Cat Map: Mixing Visualization\n'
                 f'{n_grid}x{n_grid} grid, colored by initial position',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / 'mixing.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / 'mixing.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved mixing.png/pdf", flush=True)


# =====================================================================
# 7. Dictionary recovery attempt
# =====================================================================

def plot_dictionary(model, outdir):
    """Run manifold decomposition on torus-embedded coordinates.

    For a system with continuous spectrum, we expect POOR R^2 --
    no polynomial observable should be a good eigenfunction.
    """
    print("7. Dictionary recovery attempt...", flush=True)

    try:
        from manifold_analysis import full_decomposition_nd
    except ImportError:
        print("  WARNING: manifold_analysis not available, skipping dictionary.", flush=True)
        return None

    # Generate uniform data on the torus and embed
    np.random.seed(42)
    n_pts = 5000
    xy_raw = np.random.rand(n_pts, 2).astype(np.float32)
    xy_tensor = torch.tensor(xy_raw)
    h_data = torus_embed(xy_tensor)  # (n_pts, 4)
    h_np = h_data.numpy()

    # Build a wrapper model that manifold_analysis can use
    # The encoder's kan1 and kan2 accept 4D torus input directly
    torus_enc = _TorusEncoder(model.encoder.kan1, model.encoder.kan2)

    class _FakeAE:
        def __init__(self, enc):
            self.encoder = enc

    class _FakeModel:
        def __init__(self, ae):
            self.ae = ae

    fake_model = _FakeModel(_FakeAE(torus_enc))

    var_names = ['cos_x', 'sin_x', 'cos_y', 'sin_y']
    latent_dim = model.latent_dim

    r2_scores = []
    g_r2_scores = []
    results_all = []

    for idx in range(latent_dim):
        try:
            res = full_decomposition_nd(
                fake_model, h_data, idx,
                var_names=var_names,
                target_type='latent',
                max_degree=3,
                lasso_alpha=1e-4,
                h_degree=4,
            )
            r2_scores.append(res['recon_r2'])
            g_r2_scores.append(res['g_recon_r2'])
            results_all.append(res)
            if idx < 5 or res['recon_r2'] > 0.5:
                print(f"    z[{idx}]: R2(g)={res['g_recon_r2']:.4f}, "
                      f"R2(h o g)={res['recon_r2']:.4f}, "
                      f"ideal={res['ideal']}", flush=True)
        except Exception as e:
            print(f"    z[{idx}]: FAILED -- {e}", flush=True)
            r2_scores.append(np.nan)
            g_r2_scores.append(np.nan)
            results_all.append(None)

    r2_arr = np.array(r2_scores)
    g_r2_arr = np.array(g_r2_scores)

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(latent_dim)
    bars_h = ax.bar(x_pos - 0.2, r2_arr, 0.4, label='$R^2(h \\circ g)$',
                    color='steelblue', alpha=0.8)
    bars_g = ax.bar(x_pos + 0.2, g_r2_arr, 0.4, label='$R^2(g)$',
                    color='coral', alpha=0.8)
    ax.axhline(0.0, color='black', lw=0.5)
    ax.set_xlabel('Latent dimension index', fontsize=12)
    ax.set_ylabel('$R^2$', fontsize=12)
    ax.set_title('Arnold Cat Map: Dictionary Recovery\n'
                 'Continuous spectrum $\\Rightarrow$ no polynomial eigenfunctions '
                 '$\\Rightarrow$ low $R^2$',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_xticks(x_pos)
    fig.tight_layout()
    fig.savefig(outdir / 'dictionary.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / 'dictionary.pdf', bbox_inches='tight')
    plt.close(fig)

    # Print summary
    valid = ~np.isnan(r2_arr)
    print(f"\n  Dictionary recovery summary:", flush=True)
    print(f"    Mean R2(h o g) = {np.nanmean(r2_arr):.4f}", flush=True)
    print(f"    Max  R2(h o g) = {np.nanmax(r2_arr):.4f}", flush=True)
    print(f"    Mean R2(g)     = {np.nanmean(g_r2_arr):.4f}", flush=True)
    print(f"    # dims with R2(h o g) > 0.9: "
          f"{np.sum(r2_arr[valid] > 0.9)}/{np.sum(valid)}", flush=True)
    print(f"    # dims with R2(h o g) > 0.5: "
          f"{np.sum(r2_arr[valid] > 0.5)}/{np.sum(valid)}", flush=True)
    print("  Saved dictionary.png/pdf", flush=True)

    return {'r2_hog': r2_arr.tolist(), 'r2_g': g_r2_arr.tolist()}


# =====================================================================
# Loss curve plot
# =====================================================================

def plot_loss_curve(history, outdir):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history['train']) + 1), history['train'],
            label='Train', color='steelblue', lw=1.5)
    # val is recorded every 10 epochs + ep 1 + last
    val_epochs = [1] + list(range(10, EPOCHS + 1, 10))
    if EPOCHS not in val_epochs:
        val_epochs.append(EPOCHS)
    val_epochs = val_epochs[:len(history['val'])]
    ax.plot(val_epochs, history['val'],
            label='Val', color='coral', lw=1.5, marker='o', ms=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Arnold Cat Map: Training Loss', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(outdir / 'loss_curve.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved loss_curve.png", flush=True)


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arnold Cat Map: Stable Koopman')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--skip-train', action='store_true',
                        help='Load saved model instead of training')
    args = parser.parse_args()

    EPOCHS = args.epochs
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(0)

    print("=" * 60, flush=True)
    print("Arnold Cat Map: Stable Koopman KAN", flush=True)
    print("=" * 60, flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"Lyapunov exponent: {LAMBDA_MAX:.4f}", flush=True)
    print(f"Lyapunov time: {1.0/LAMBDA_MAX:.2f} steps", flush=True)
    print(f"Config: mid={MID_SIZE}, latent={LATENT_DIM}, epochs={EPOCHS}", flush=True)
    print(flush=True)

    # --- Simulate ---
    print("Simulating cat map trajectories...", flush=True)
    traj = simulate_cat_map(n_traj=N_TRAJ, n_steps=N_STEPS, seed=42)
    print(f"  Trajectories: {traj.shape}", flush=True)
    print(f"  Range: x=[{traj[:,:,0].min():.4f}, {traj[:,:,0].max():.4f}], "
          f"y=[{traj[:,:,1].min():.4f}, {traj[:,:,1].max():.4f}]", flush=True)

    # --- Train or load ---
    ckpt_path = OUTDIR / 'cat_map_model.pt'
    model = CatMapKoopman(MID_SIZE, LATENT_DIM, GRID_SIZE, SPLINE_ORDER)

    if args.skip_train and ckpt_path.exists():
        print(f"\nLoading model from {ckpt_path}...", flush=True)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        history = ckpt.get('history', {'train': [], 'val': []})
    else:
        print(f"\nTraining...", flush=True)
        model, history = train(model, traj, epochs=EPOCHS, lr=LR,
                               device=args.device)

        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'config': {
                'mid': MID_SIZE, 'latent': LATENT_DIM,
                'grid_size': GRID_SIZE, 'spline_order': SPLINE_ORDER,
                'epochs': EPOCHS, 'lr': LR,
                'n_traj': N_TRAJ, 'n_steps': N_STEPS,
            },
        }, ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path}", flush=True)

    # --- Save config ---
    config = {
        'system': 'arnold_cat_map',
        'map_matrix': [[2, 1], [1, 1]],
        'lyapunov_exponent': float(LAMBDA_MAX),
        'lyapunov_time_steps': float(1.0 / LAMBDA_MAX),
        'mid_size': MID_SIZE,
        'latent_dim': LATENT_DIM,
        'grid_size': GRID_SIZE,
        'spline_order': SPLINE_ORDER,
        'epochs': EPOCHS,
        'lr': LR,
        'batch_size': BATCH_SIZE,
        'n_traj': N_TRAJ,
        'n_steps': N_STEPS,
        'recon_weight': RECON_WEIGHT,
        'propagator': 'StablePropagator (M = expm(K - L^T L), |lambda| <= 1)',
        'encoder_input': 'torus embedding (cos(2pi*x), sin(2pi*x), cos(2pi*y), sin(2pi*y))',
        'decoder_output': 'atan2 recovery to [0,1)^2',
    }
    with open(OUTDIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # --- Generate all plots ---
    print(f"\n{'='*60}", flush=True)
    print("Generating plots...", flush=True)
    print(f"{'='*60}", flush=True)

    # Loss curve
    if history['train']:
        plot_loss_curve(history, OUTDIR)

    # 1. Phase space rollout
    plot_phase_space(model, OUTDIR)

    # 2. Prediction accuracy
    model_errors, persist_errors = plot_prediction_accuracy(model, OUTDIR)

    # 3. Eigenvalue spectrum
    eigs, mags = plot_eigenvalues(model, OUTDIR)

    # 4. Eigenvalue decay
    plot_eigenvalue_decay(eigs, OUTDIR)

    # 5. Error growth
    plot_error_growth(model, OUTDIR)

    # 6. Mixing visualization
    plot_mixing(model, OUTDIR)

    # 7. Dictionary recovery
    dict_results = plot_dictionary(model, OUTDIR)

    # --- Summary ---
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY: Arnold Cat Map -- Stable Koopman KAN", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\nSystem properties:", flush=True)
    print(f"  Map matrix: [[2,1],[1,1]]", flush=True)
    print(f"  Lyapunov exponent: {LAMBDA_MAX:.4f}", flush=True)
    print(f"  Koopman spectrum: PURELY CONTINUOUS (no eigenfunctions except constants)", flush=True)

    print(f"\nModel properties:", flush=True)
    print(f"  Architecture: KAN [4,{MID_SIZE},{LATENT_DIM}] encoder/decoder", flush=True)
    print(f"  Propagator: expm(K - L^T L), |lambda| <= 1", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}", flush=True)

    print(f"\nEigenvalue analysis:", flush=True)
    print(f"  |lambda| range: [{mags.min():.6f}, {mags.max():.6f}]", flush=True)
    print(f"  Mean |lambda|: {mags.mean():.6f}", flush=True)
    print(f"  # with |lambda| > 0.99: {np.sum(mags > 0.99)}/{len(mags)}", flush=True)
    print(f"  # with |lambda| > 0.9:  {np.sum(mags > 0.9)}/{len(mags)}", flush=True)
    all_inside = np.all(mags < 1.0 + 1e-6)
    print(f"  All inside unit circle: {all_inside}", flush=True)

    print(f"\nPrediction (1-step angular MSE): {model_errors[0]:.6f}", flush=True)
    print(f"Prediction (5-step angular MSE): {model_errors[4]:.6f}", flush=True)
    print(f"Random baseline: {1.0/6.0:.6f}", flush=True)

    if dict_results is not None:
        r2s = [v for v in dict_results['r2_hog'] if not np.isnan(v)]
        if r2s:
            print(f"\nDictionary recovery:", flush=True)
            print(f"  Mean R2(h o g): {np.mean(r2s):.4f}", flush=True)
            print(f"  Max  R2(h o g): {np.max(r2s):.4f}", flush=True)
            print(f"  Compare: Lorenz R2 ~ 0.99, KS R2 ~ 1.0", flush=True)
            print(f"  Cat map should be much lower (continuous spectrum)", flush=True)

    print(f"\nExpected outcomes (continuous spectrum):", flush=True)
    print(f"  [1] Phase space: model rollout should NOT fill torus uniformly", flush=True)
    print(f"  [2] Prediction: error saturates within ~1-2 Lyapunov times", flush=True)
    print(f"  [3] Eigenvalues: should be INSIDE unit circle (damping)", flush=True)
    print(f"  [4] Dictionary: R2 << 1 (no polynomial eigenfunctions)", flush=True)
    print(f"  [5] Mixing: model fails to reproduce stretching/folding beyond 1-2 steps", flush=True)

    print(f"\nAll results saved to {OUTDIR}/", flush=True)
