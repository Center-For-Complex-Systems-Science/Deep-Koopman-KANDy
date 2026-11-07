"""
Standard Map: lock in discovered Fourier basis + sweep additional modes.

The encoder is hybrid:
    z = [fixed_fourier(x,y), KAN_learned(cos x, sin x, cos y, sin y)]

where fixed_fourier contains the 12 modes we already discovered:
    {cos x, sin x, cos y, sin y,
     cos(x+y), sin(x+y), cos(x-y), sin(x-y),
     cos 2x, sin 2x, cos 2y, sin 2y}

The KAN learns additional higher-order modes. We sweep the number of
extra learned dims: 0, 8, 16, 32, 64.

Hypothesis: as M increases, eigenvalues move toward the unit circle
and the chaotic sea fills in.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from kan import KANLinear
from training import set_seed
from standard_map import simulate_standard_map, MapPairDataset, angular_mse

OUTDIR = Path('results/standard_map/fourier_sweep')


# =====================================================================
# Fixed Fourier feature layer
# =====================================================================

class FixedFourierFeatures(nn.Module):
    """Compute the 12 discovered Fourier modes analytically (no learning)."""

    def __init__(self):
        super().__init__()
        self.n_features = 12

    def forward(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        return torch.stack([
            torch.cos(x), torch.sin(x),         # (1,0), fundamental
            torch.cos(y), torch.sin(y),         # (0,1)
            torch.cos(x + y), torch.sin(x + y), # (1,1)
            torch.cos(x - y), torch.sin(x - y), # (1,-1)
            torch.cos(2 * x), torch.sin(2 * x), # (2,0)
            torch.cos(2 * y), torch.sin(2 * y), # (0,2)
        ], dim=-1)


# =====================================================================
# Hybrid encoder: fixed Fourier + learned KAN
# =====================================================================

class HybridFourierKANEncoder(nn.Module):
    """z = [fixed_fourier(x,y), KAN(trig_embed(x,y))]

    The first 12 dims are the locked-in Fourier basis.
    The remaining dims are learned by a two-layer KAN.
    """

    def __init__(self, n_extra, mid_size=8, grid_size=5, spline_order=3):
        super().__init__()
        self.fourier = FixedFourierFeatures()
        self.n_fixed = self.fourier.n_features
        self.n_extra = n_extra
        self.latent_dim = self.n_fixed + n_extra

        if n_extra > 0:
            kan_kw = dict(grid_size=grid_size, spline_order=spline_order)
            self.kan1 = KANLinear(4, mid_size, grid_range=[-1.5, 1.5], **kan_kw)
            self.kan2 = KANLinear(mid_size, n_extra, grid_range=[-5, 5], **kan_kw)

    def forward(self, xy):
        f_fixed = self.fourier(xy)

        if self.n_extra == 0:
            return f_fixed

        # Trig embed for KAN input
        x, y = xy[:, 0], xy[:, 1]
        trig = torch.stack([torch.cos(x), torch.sin(x),
                            torch.cos(y), torch.sin(y)], dim=-1)
        f_learned = self.kan2(self.kan1(trig))

        return torch.cat([f_fixed, f_learned], dim=-1)


class HybridDecoder(nn.Module):
    """Decode from full latent (fixed + learned) back to torus."""

    def __init__(self, latent_dim, mid_size=8, grid_size=5, spline_order=3):
        super().__init__()
        kan_kw = dict(grid_size=grid_size, spline_order=spline_order)
        self.kan1 = KANLinear(latent_dim, mid_size, grid_range=[-5, 5], **kan_kw)
        self.kan2 = KANLinear(mid_size, 4, grid_range=[-5, 5], **kan_kw)

    def forward(self, z):
        h = self.kan2(self.kan1(z))
        x = torch.atan2(h[:, 1], h[:, 0]) % (2 * np.pi)
        y = torch.atan2(h[:, 3], h[:, 2]) % (2 * np.pi)
        return torch.stack([x, y], dim=-1)


class HybridKoopman(nn.Module):
    """Discrete Koopman with hybrid Fourier+KAN encoder."""

    def __init__(self, n_extra, mid_size=8):
        super().__init__()
        self.encoder = HybridFourierKANEncoder(n_extra, mid_size)
        self.latent_dim = self.encoder.latent_dim
        self.decoder = HybridDecoder(self.latent_dim, mid_size)
        self.M = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

    def forward_one_step(self, xy):
        z = self.encoder(xy)
        xy_rec = self.decoder(z)
        z_next = self.M(z)
        xy_next = self.decoder(z_next)
        return {
            'z_n': z, 'xy_n_rec': xy_rec,
            'z_np1_pred': z_next, 'xy_np1_pred': xy_next,
        }

    def koopman_matrix(self):
        return self.M.weight.detach()


# =====================================================================
# Training
# =====================================================================

def train(n_extra, traj, epochs=300, lr=1e-3, mid_size=8, device='cpu'):
    n_train = int(0.8 * traj.shape[0])
    train_dl = DataLoader(MapPairDataset(traj[:n_train]),
                          batch_size=2048, shuffle=True, drop_last=True)
    val_dl = DataLoader(MapPairDataset(traj[n_train:]),
                        batch_size=2048)

    model = HybridKoopman(n_extra, mid_size).to(device)
    latent_dim = model.latent_dim
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  12 fixed + {n_extra} learned = {latent_dim} latent dims, {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xy_n, xy_np1 in train_dl:
            xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
            out = model.forward_one_step(xy_n)
            z_true = model.encoder(xy_np1)
            loss = (angular_mse(out['xy_np1_pred'], xy_np1)
                    + mse(out['z_np1_pred'], z_true)
                    + 0.5 * angular_mse(out['xy_n_rec'], xy_n))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        scheduler.step()

        if ep == 1 or ep % 50 == 0 or ep == epochs:
            model.eval()
            vloss = []
            with torch.no_grad():
                for xy_n, xy_np1 in val_dl:
                    xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
                    out = model.forward_one_step(xy_n)
                    z_true = model.encoder(xy_np1)
                    l = (angular_mse(out['xy_np1_pred'], xy_np1)
                         + mse(out['z_np1_pred'], z_true)
                         + 0.5 * angular_mse(out['xy_n_rec'], xy_n))
                    vloss.append(l.item())
            eigs = np.linalg.eigvals(model.koopman_matrix().cpu().numpy())
            mags = np.sort(np.abs(eigs))[::-1][:5]
            n_on_circle = np.sum(np.abs(eigs) > 0.95)
            print(f"  Ep {ep:3d} | val {np.mean(vloss):.5f} "
                  f"| |λ|>0.95: {n_on_circle}/{latent_dim} "
                  f"| top: {mags.round(3)}", flush=True)

    return model.cpu().eval()


# =====================================================================
# Rollout and phase space plot
# =====================================================================

def rollout_phase_space(model, K, n_ics=200, n_steps=5000):
    """Rollout model and return scatter points."""
    torch.manual_seed(42)
    x0 = 2 * np.pi * torch.rand(n_ics)
    y0 = 2 * np.pi * torch.rand(n_ics)

    pts = []
    with torch.no_grad():
        for ic in range(n_ics):
            xy = torch.stack([x0[ic], y0[ic]]).unsqueeze(0)
            for t in range(n_steps):
                pts.append(xy.squeeze().numpy())
                out = model.forward_one_step(xy)
                xy = out['xy_np1_pred']
    return np.array(pts)


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=float, default=0.97)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    K = args.K

    print(f"Standard Map Fourier Sweep: K={K}")
    print(f"Fixed basis: 12 Fourier modes (|m|+|n| ≤ 2)")

    traj = simulate_standard_map(n_traj=200, n_steps=2000, K=K, device=args.device).cpu()

    # True phase space for comparison
    traj_true = simulate_standard_map(n_traj=200, n_steps=5000, K=K).cpu()
    true_pts = traj_true.reshape(-1, 2).numpy()

    # Sweep: 0 extra (pure Fourier), 8, 16, 32, 64 extra learned dims
    extras = [0, 8, 16, 32, 64]
    results = {}

    for n_extra in extras:
        print(f"\n{'='*60}")
        print(f"n_extra = {n_extra} (total latent = {12 + n_extra})")
        print(f"{'='*60}")

        model = train(n_extra, traj, epochs=300, device=args.device)
        eigs = np.linalg.eigvals(model.koopman_matrix().numpy())

        print(f"  Rolling out phase space...")
        model_pts = rollout_phase_space(model, K, n_ics=200, n_steps=5000)

        results[n_extra] = {
            'model': model,
            'eigs': eigs,
            'model_pts': model_pts,
        }

    # ---- Plot: phase space comparison across sizes ----
    n_configs = len(extras)
    fig, axes = plt.subplots(1, n_configs + 1, figsize=(4 * (n_configs + 1), 4))

    # True
    ax = axes[0]
    sub = np.random.choice(len(true_pts), min(80000, len(true_pts)), replace=False)
    ax.scatter(true_pts[sub, 0], true_pts[sub, 1], s=0.05, alpha=0.2,
               c='black', rasterized=True)
    ax.set_title('True', fontsize=11)
    ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')

    for i, n_extra in enumerate(extras):
        ax = axes[i + 1]
        pts = results[n_extra]['model_pts']
        sub = np.random.choice(len(pts), min(80000, len(pts)), replace=False)
        ax.scatter(pts[sub, 0], pts[sub, 1], s=0.05, alpha=0.2,
                   c='steelblue', rasterized=True)
        eigs = results[n_extra]['eigs']
        n_circle = np.sum(np.abs(eigs) > 0.95)
        ax.set_title(f'12+{n_extra} ({n_circle} on circle)', fontsize=10)
        ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$')

    fig.suptitle(f'Standard Map K={K}: Phase Space vs Fourier Basis Size', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'phase_space_sweep.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'phase_space_sweep.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved phase_space_sweep.png/pdf")

    # ---- Plot: eigenvalue convergence ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.2)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(extras)))
    for n_extra, col in zip(extras, colors):
        eigs = results[n_extra]['eigs']
        ax.scatter(eigs.real, eigs.imag, s=30, c=[col], alpha=0.7,
                   label=f'12+{n_extra}', edgecolors='black', linewidths=0.3)
    ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')
    ax.set_title('Eigenvalues in Complex Plane')
    ax.set_aspect('equal')
    ax.legend(fontsize=7)

    ax = axes[1]
    for n_extra, col in zip(extras, colors):
        eigs = results[n_extra]['eigs']
        mags = np.sort(np.abs(eigs))[::-1]
        ax.plot(mags, 'o-', color=col, markersize=3, lw=1, label=f'12+{n_extra}')
    ax.axhline(1.0, color='red', ls='--', alpha=0.3)
    ax.axhline(0.95, color='orange', ls=':', alpha=0.3)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('|λ|')
    ax.set_title('Eigenvalue Magnitudes')
    ax.legend(fontsize=7)

    fig.suptitle(f'Standard Map K={K}: Eigenvalue Convergence with Fourier Basis Size',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'eigenvalue_sweep.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'eigenvalue_sweep.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved eigenvalue_sweep.png/pdf")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'|λ|>0.95':>10} {'|λ|>0.99':>10} {'Max |λ|':>10}")
    print("-" * 50)
    for n_extra in extras:
        eigs = results[n_extra]['eigs']
        mags = np.abs(eigs)
        print(f"12+{n_extra:<11} {np.sum(mags>0.95):>10} "
              f"{np.sum(mags>0.99):>10} {mags.max():>10.4f}")

    print(f"\nAll results in {OUTDIR}/")
