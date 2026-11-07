"""
Standard Map: Deep Koopman KAN with eigenvalue convergence study.

The Standard Map:
    y_{n+1} = y_n + K sin(x_n)     (mod 2π)
    x_{n+1} = x_n + y_{n+1}         (mod 2π)

This is a discrete-time area-preserving map on the torus [0,2π)².

Strategy:
    1. Train KANLinear models at increasing latent dims [4, 8, 16, 32, 64]
    2. Track Koopman eigenvalue convergence
    3. Once stable, run manifold/level-set decomposition
    4. Check: does KAN discover Fourier-like observables?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from kan import KANLinear
from training import set_seed

OUTDIR = Path('results/standard_map')


# =====================================================================
# Standard Map simulator
# =====================================================================

@torch.no_grad()
def simulate_standard_map(n_traj=200, n_steps=2000, K=0.97, device='cpu'):
    """Simulate Standard Map trajectories on the torus [0,2π)².

    Returns:
        traj: (n_traj, n_steps, 2) — columns are (x, y)
    """
    # Random initial conditions on the torus
    x = 2 * np.pi * torch.rand(n_traj, device=device)
    y = 2 * np.pi * torch.rand(n_traj, device=device)

    traj = torch.zeros(n_traj, n_steps, 2, device=device)
    for t in range(n_steps):
        traj[:, t, 0] = x
        traj[:, t, 1] = y
        # Standard map step
        y_new = (y + K * torch.sin(x)) % (2 * np.pi)
        x_new = (x + y_new) % (2 * np.pi)
        x, y = x_new, y_new

    return traj


class MapPairDataset(Dataset):
    """Dataset of (state_n, state_{n+1}) pairs from map trajectories."""
    def __init__(self, traj):
        self.traj = traj
        self.n_traj, self.n_steps, self.d = traj.shape
        self.n_pairs = self.n_traj * (self.n_steps - 1)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        traj_id = idx // (self.n_steps - 1)
        t = idx % (self.n_steps - 1)
        return self.traj[traj_id, t], self.traj[traj_id, t + 1]


# =====================================================================
# Discrete-time Deep Koopman with KAN
# =====================================================================

class KANEncoder(nn.Module):
    """Two-layer KAN encoder for the torus.

    Input: (x, y) ∈ [0, 2π)² — raw angles.
    We feed (cos x, sin x, cos y, sin y) to handle periodicity.
    """
    def __init__(self, mid_size, latent_size, grid_size=5, spline_order=3):
        super().__init__()
        # 4 inputs: cos(x), sin(x), cos(y), sin(y)
        kan_kw = dict(grid_size=grid_size, spline_order=spline_order,
                      grid_range=[-1.5, 1.5])
        self.kan1 = KANLinear(4, mid_size, **kan_kw)
        self.kan2 = KANLinear(mid_size, latent_size,
                              grid_size=grid_size, spline_order=spline_order,
                              grid_range=[-5, 5])

    def forward(self, xy):
        # Lift to trig embedding for periodicity
        x, y = xy[:, 0], xy[:, 1]
        h = torch.stack([torch.cos(x), torch.sin(x),
                         torch.cos(y), torch.sin(y)], dim=-1)
        return self.kan2(self.kan1(h))


class KANDecoder(nn.Module):
    """Two-layer KAN decoder back to the torus."""
    def __init__(self, latent_size, mid_size, grid_size=5, spline_order=3):
        super().__init__()
        kan_kw = dict(grid_size=grid_size, spline_order=spline_order,
                      grid_range=[-5, 5])
        self.kan1 = KANLinear(latent_size, mid_size, **kan_kw)
        # Output 4: cos(x), sin(x), cos(y), sin(y) — then recover angles via atan2
        self.kan2 = KANLinear(mid_size, 4,
                              grid_size=grid_size, spline_order=spline_order,
                              grid_range=[-5, 5])

    def forward(self, z):
        h = self.kan2(self.kan1(z))
        # Recover angles from (cos, sin) pairs
        x = torch.atan2(h[:, 1], h[:, 0])  # atan2(sin x, cos x)
        y = torch.atan2(h[:, 3], h[:, 2])
        # Map to [0, 2π)
        x = x % (2 * np.pi)
        y = y % (2 * np.pi)
        return torch.stack([x, y], dim=-1)


class DiscreteKoopmanKAN(nn.Module):
    """Discrete-time Deep Koopman model for maps.

    z_{n+1} = M z_n  (linear propagation, no matrix exponential)
    """
    def __init__(self, mid_size, latent_dim, grid_size=5, spline_order=3):
        super().__init__()
        self.encoder = KANEncoder(mid_size, latent_dim, grid_size, spline_order)
        self.decoder = KANDecoder(latent_dim, mid_size, grid_size, spline_order)
        self.M = nn.Linear(latent_dim, latent_dim, bias=False)
        self.latent_dim = latent_dim

    def forward_one_step(self, xy_n):
        z_n = self.encoder(xy_n)
        xy_n_rec = self.decoder(z_n)
        z_np1_pred = self.M(z_n)
        xy_np1_pred = self.decoder(z_np1_pred)
        return {
            'z_n': z_n,
            'xy_n_rec': xy_n_rec,
            'z_np1_pred': z_np1_pred,
            'xy_np1_pred': xy_np1_pred,
        }

    def koopman_matrix(self):
        return self.M.weight.detach()


# =====================================================================
# Training
# =====================================================================

def angular_mse(pred, true):
    """MSE on the torus: accounts for wraparound at 2π."""
    diff = pred - true
    # Wrap to [-π, π]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return (diff ** 2).mean()


def train_model(mid_size, latent_dim, traj, epochs=200, lr=1e-3,
                batch_size=2048, device='cpu'):
    n_train = int(0.8 * traj.shape[0])
    train_dl = DataLoader(MapPairDataset(traj[:n_train]),
                          batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(MapPairDataset(traj[n_train:]),
                        batch_size=batch_size)

    model = DiscreteKoopmanKAN(mid_size, latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {'train': [], 'val': [], 'eigs': []}

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xy_n, xy_np1 in train_dl:
            xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
            out = model.forward_one_step(xy_n)
            z_np1_true = model.encoder(xy_np1)

            loss_pred = angular_mse(out['xy_np1_pred'], xy_np1)
            loss_latent = nn.functional.mse_loss(out['z_np1_pred'], z_np1_true)
            loss_recon = angular_mse(out['xy_n_rec'], xy_n)
            loss = loss_pred + loss_latent + 0.5 * loss_recon

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        vloss = []
        with torch.no_grad():
            for xy_n, xy_np1 in val_dl:
                xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
                out = model.forward_one_step(xy_n)
                z_np1_true = model.encoder(xy_np1)
                l = (angular_mse(out['xy_np1_pred'], xy_np1)
                     + nn.functional.mse_loss(out['z_np1_pred'], z_np1_true)
                     + 0.5 * angular_mse(out['xy_n_rec'], xy_n))
                vloss.append(l.item())

        history['train'].append(float(np.mean(losses)))
        history['val'].append(float(np.mean(vloss)))

        # Track eigenvalues
        eigs = np.linalg.eigvals(model.koopman_matrix().cpu().numpy())
        history['eigs'].append(eigs)

        if ep == 1 or ep % 25 == 0 or ep == epochs:
            eig_mags = np.sort(np.abs(eigs))[::-1][:5]
            print(f"  Ep {ep:3d} | train {history['train'][-1]:.5f} "
                  f"| val {history['val'][-1]:.5f} "
                  f"| |λ|: {eig_mags.round(4)}", flush=True)

    model = model.cpu().eval()
    return model, history


# =====================================================================
# Eigenvalue convergence study
# =====================================================================

def eigenvalue_convergence(traj, configs, device='cpu'):
    """Train models at increasing sizes and track eigenvalue convergence."""
    all_results = {}

    for cfg in configs:
        mid = cfg['mid']
        latent = cfg['latent']
        name = f'[4,{mid},{latent}]'
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        model, history = train_model(
            mid, latent, traj, epochs=cfg.get('epochs', 200),
            lr=cfg.get('lr', 1e-3), device=device,
        )

        # Final eigenvalues
        eigs = np.linalg.eigvals(model.koopman_matrix().numpy())
        eigs_sorted = eigs[np.argsort(-np.abs(eigs))]

        all_results[name] = {
            'model': model,
            'history': history,
            'eigs': eigs_sorted,
            'mid': mid,
            'latent': latent,
        }

        print(f"\n  Final eigenvalues (top 10 by |λ|):")
        for i, e in enumerate(eigs_sorted[:10]):
            print(f"    λ_{i}: {e:.4f}  |λ|={abs(e):.4f}  arg={np.degrees(np.angle(e)):+.1f}°")

    return all_results


def plot_convergence(all_results, K):
    """Plot eigenvalue convergence across model sizes."""
    names = list(all_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Eigenvalues in complex plane
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.2)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    for name, col in zip(names, colors):
        eigs = all_results[name]['eigs']
        ax.scatter(eigs.real, eigs.imag, s=40, c=[col], alpha=0.7,
                   label=name, edgecolors='black', linewidths=0.3)
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title('Koopman Eigenvalues')
    ax.set_aspect('equal')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Panel 2: Eigenvalue magnitudes across sizes
    ax = axes[1]
    for i, (name, col) in enumerate(zip(names, colors)):
        eigs = all_results[name]['eigs']
        mags = np.sort(np.abs(eigs))[::-1]
        ax.plot(mags, 'o-', color=col, markersize=3, lw=1, label=name)
    ax.axhline(1.0, color='red', ls='--', alpha=0.3)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('|λ|')
    ax.set_title('Eigenvalue Magnitudes')
    ax.legend(fontsize=7)

    # Panel 3: Convergence — do eigenvalues stabilize?
    ax = axes[2]
    # Compare consecutive sizes: how much do top eigenvalues change?
    if len(names) > 1:
        convergence = []
        for i in range(1, len(names)):
            prev_eigs = all_results[names[i-1]]['eigs']
            curr_eigs = all_results[names[i]]['eigs']
            # Match by nearest
            n_compare = min(len(prev_eigs), len(curr_eigs), 10)
            diffs = []
            for j in range(n_compare):
                # Find nearest eigenvalue in curr to prev[j]
                d = np.min(np.abs(prev_eigs[j] - curr_eigs))
                diffs.append(d)
            convergence.append(np.mean(diffs))

        ax.bar(range(len(convergence)), convergence, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(convergence)))
        ax.set_xticklabels([f'{names[i]}→{names[i+1]}' for i in range(len(convergence))],
                           fontsize=7, rotation=20)
        ax.set_ylabel('Mean eigenvalue shift')
        ax.set_title('Convergence (smaller = more stable)')

    fig.suptitle(f'Standard Map K={K}: Eigenvalue Convergence Study', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'eigenvalue_convergence.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'eigenvalue_convergence.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved eigenvalue_convergence.png/pdf")


def plot_phase_space(traj, K):
    """Plot the standard map phase space."""
    fig, ax = plt.subplots(figsize=(8, 8))
    t_np = traj[:20].reshape(-1, 2).numpy()
    ax.scatter(t_np[:, 0], t_np[:, 1], s=0.1, alpha=0.3, c='black', rasterized=True)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_title(f'Standard Map Phase Space (K={K})', fontsize=13)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(OUTDIR / 'phase_space.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved phase_space.png")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=float, default=0.97, help='Kick strength')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    K = args.K

    print(f"Standard Map: K={K}")
    print(f"Device: {args.device}")

    # Simulate
    print("\nSimulating...")
    traj = simulate_standard_map(n_traj=200, n_steps=2000, K=K, device=args.device)
    traj = traj.cpu()
    print(f"  Trajectories: {traj.shape}")

    # Phase space
    plot_phase_space(traj, K)

    # Eigenvalue convergence study
    configs = [
        {'mid': 4,  'latent': 4,  'epochs': 200},
        {'mid': 8,  'latent': 8,  'epochs': 200},
        {'mid': 8,  'latent': 16, 'epochs': 250},
        {'mid': 16, 'latent': 32, 'epochs': 300},
        {'mid': 16, 'latent': 64, 'epochs': 300},
    ]

    results = eigenvalue_convergence(traj, configs, device=args.device)
    plot_convergence(results, K)

    # Save eigenvalue data
    eig_data = {}
    for name, r in results.items():
        eig_data[name] = {
            'eigs': r['eigs'].tolist(),
            'val_loss': r['history']['val'][-1],
            'mid': r['mid'],
            'latent': r['latent'],
        }

    import json
    with open(OUTDIR / 'eigenvalue_data.json', 'w') as f:
        json.dump(eig_data, f, indent=2, default=lambda x: str(x))
    print(f"\nSaved eigenvalue_data.json")

    # Print convergence summary
    print(f"\n{'='*60}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'Val Loss':>10} {'# |λ|>0.99':>12} {'# |λ|>0.9':>12} {'Max |λ|':>10}")
    print("-" * 60)
    for name, r in results.items():
        mags = np.abs(r['eigs'])
        print(f"{name:<15} {r['history']['val'][-1]:>10.5f} "
              f"{np.sum(mags > 0.99):>12} {np.sum(mags > 0.9):>12} "
              f"{mags.max():>10.4f}")

    print(f"\nAll results in {OUTDIR}/")
