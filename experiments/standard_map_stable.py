"""
Standard Map with stable Koopman propagator (|λ| ≤ 1).

Propagator: M = expm(K - L^T L)  where:
    K = -K^T              (skew-symmetric, drives rotation)
    L^T L >= 0            (PSD, drives damping)

    => Re(eigenvalues of K - L^T L) <= 0
    => |eigenvalues of M| <= 1

Same parameterization as Lorenz, KS, cat map, and vortex shedding experiments.
When L -> 0: M = expm(K) is orthogonal (|λ| = 1, area-preserving).
When L != 0: selective damping allowed.

For the Standard Map:
    - KAM tori modes should have |λ| ≈ 1 (persistent)
    - Chaotic sea modes should have |λ| < 1 (mixing → effective damping)
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

OUTDIR = Path('results/standard_map')


# =====================================================================
# Stable propagator: M = expm(K - L^T L), |λ| ≤ 1
# =====================================================================

class StablePropagator(nn.Module):
    """M = expm(A) where A = K - L^T L.

    K = -K^T (skew-symmetric, rotation)
    L^T L >= 0 (PSD, damping)

    => Re(eigenvalues of A) <= 0 => |eigenvalues of M| <= 1.
    Params: n(n-1)/2 (skew) + n(n+1)/2 (lower-tri L) = n^2.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        n_skew = dim * (dim - 1) // 2
        self.skew_params = nn.Parameter(0.01 * torch.randn(n_skew))
        self.L_diag = nn.Parameter(0.01 * torch.randn(dim))
        n_lower = dim * (dim - 1) // 2
        self.L_lower = nn.Parameter(0.01 * torch.randn(n_lower))

    def _get_skew(self):
        n = self.dim
        K = torch.zeros(n, n, device=self.skew_params.device)
        idx = torch.triu_indices(n, n, offset=1)
        K[idx[0], idx[1]] = self.skew_params
        return K - K.T

    def _get_L(self):
        n = self.dim
        L = torch.zeros(n, n, device=self.L_diag.device)
        L[range(n), range(n)] = self.L_diag
        idx = torch.tril_indices(n, n, offset=-1)
        L[idx[0], idx[1]] = self.L_lower
        return L

    def generator_matrix(self):
        K = self._get_skew()
        L = self._get_L()
        return K - L.T @ L

    def get_matrix(self):
        return torch.matrix_exp(self.generator_matrix())

    def forward(self, z):
        return z @ self.get_matrix().T


# =====================================================================
# Encoder / Decoder
# =====================================================================

class KANEncoder(nn.Module):
    def __init__(self, mid_size, latent_size):
        super().__init__()
        self.kan1 = KANLinear(4, mid_size, grid_size=5, spline_order=3,
                              grid_range=[-1.5, 1.5])
        self.kan2 = KANLinear(mid_size, latent_size, grid_size=5, spline_order=3,
                              grid_range=[-5, 5])

    def forward(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        h = torch.stack([torch.cos(x), torch.sin(x),
                         torch.cos(y), torch.sin(y)], dim=-1)
        return self.kan2(self.kan1(h))


class KANDecoder(nn.Module):
    def __init__(self, latent_size, mid_size):
        super().__init__()
        self.kan1 = KANLinear(latent_size, mid_size, grid_size=5, spline_order=3,
                              grid_range=[-5, 5])
        self.kan2 = KANLinear(mid_size, 4, grid_size=5, spline_order=3,
                              grid_range=[-5, 5])

    def forward(self, z):
        h = self.kan2(self.kan1(z))
        x = torch.atan2(h[:, 1], h[:, 0]) % (2 * np.pi)
        y = torch.atan2(h[:, 3], h[:, 2]) % (2 * np.pi)
        return torch.stack([x, y], dim=-1)


class StableKoopmanKAN(nn.Module):
    def __init__(self, mid_size, latent_dim):
        super().__init__()
        self.encoder = KANEncoder(mid_size, latent_dim)
        self.decoder = KANDecoder(latent_dim, mid_size)
        self.prop = StablePropagator(latent_dim)
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
# Training
# =====================================================================

def train(mid_size, latent_dim, traj, epochs=400, lr=1e-3, device='cpu'):
    n_train = int(0.8 * traj.shape[0])
    train_dl = DataLoader(MapPairDataset(traj[:n_train]),
                          batch_size=2048, shuffle=True, drop_last=True)
    val_dl = DataLoader(MapPairDataset(traj[n_train:]),
                        batch_size=2048)

    model = StableKoopmanKAN(mid_size, latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Stable Koopman [4,{mid_size},{latent_dim}]: {n_params:,} params")

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

            M = model.koopman_matrix().cpu().numpy()
            eigs = np.linalg.eigvals(M)
            mags = np.abs(eigs)

            print(f"  Ep {ep:3d} | val {np.mean(vloss):.5f} "
                  f"| |λ| [{mags.min():.4f}, {mags.max():.4f}]", flush=True)

    return model.cpu().eval()


# =====================================================================
# Rollout + plotting
# =====================================================================

def rollout(model, n_ics=200, n_steps=5000):
    torch.manual_seed(42)
    x0 = 2 * np.pi * torch.rand(n_ics)
    y0 = 2 * np.pi * torch.rand(n_ics)
    pts = []
    with torch.no_grad():
        for ic in range(n_ics):
            xy = torch.stack([x0[ic], y0[ic]]).unsqueeze(0)
            for t in range(n_steps):
                if t % 2 == 0:
                    pts.append(xy.squeeze().numpy().copy())
                out = model.forward_one_step(xy)
                xy = out['xy_np1_pred']
            if (ic + 1) % 50 == 0:
                print(f"    {ic+1}/{n_ics}", flush=True)
    return np.array(pts)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=float, default=0.97)
    parser.add_argument('--mid', type=int, default=32)
    parser.add_argument('--latent', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    K = args.K

    print(f"Standard Map K={K}: Stable Koopman (A = K - L^T L)")
    print(f"Device: {args.device}")

    traj = simulate_standard_map(n_traj=200, n_steps=2000, K=K,
                                  device=args.device).cpu()

    model = train(args.mid, args.latent, traj, epochs=args.epochs,
                  device=args.device)

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'mid': args.mid, 'latent': args.latent, 'K': K},
    }, OUTDIR / f'standard_map_stable_{args.mid}_{args.latent}.pt')
    print("Saved checkpoint")

    # Eigenvalue analysis
    M = model.koopman_matrix().numpy()
    eigs = np.linalg.eigvals(M)
    mags = np.abs(eigs)
    A = model.prop.generator_matrix().detach().cpu().numpy()
    eigs_A = np.linalg.eigvals(A)

    print(f"\n|λ| range: [{mags.min():.4f}, {mags.max():.4f}]")
    print(f"# with |λ| > 0.99: {np.sum(mags > 0.99)}")
    print(f"# with |λ| > 0.95: {np.sum(mags > 0.95)}")
    print(f"Generator Re(μ) range: [{eigs_A.real.min():.4f}, {eigs_A.real.max():.4f}]")
    print(f"All |λ| ≤ 1? {np.all(mags <= 1.0 + 1e-6)}")

    # Rollout
    print(f"\nRolling out 200 ICs × 5000 steps...")
    model_pts = rollout(model)

    true_traj = simulate_standard_map(n_traj=200, n_steps=5000, K=K).reshape(-1, 2).numpy()

    # Plot: phase space comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sub = np.random.choice(len(true_traj), min(150000, len(true_traj)), replace=False)
    axes[0].scatter(true_traj[sub, 0], true_traj[sub, 1], s=0.05, alpha=0.2,
                    c='black', rasterized=True)
    axes[0].set_title('True Phase Space', fontsize=13)
    axes[0].set_xlim(0, 2*np.pi); axes[0].set_ylim(0, 2*np.pi)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')

    sub2 = np.random.choice(len(model_pts), min(150000, len(model_pts)), replace=False)
    axes[1].scatter(model_pts[sub2, 0], model_pts[sub2, 1], s=0.05, alpha=0.2,
                    c='steelblue', rasterized=True)
    axes[1].set_title(f'Stable Koopman [{args.mid},{args.latent}]', fontsize=13)
    axes[1].set_xlim(0, 2*np.pi); axes[1].set_ylim(0, 2*np.pi)
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('$x$')

    fig.suptitle(f'Standard Map K={K}: Stable Koopman (A = K - L$^T$L)\n'
                 f'|λ| range: [{mags.min():.4f}, {mags.max():.4f}]', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'phase_space_stable.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'phase_space_stable.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved phase_space_stable.png/pdf")

    # Eigenvalue plot (two panels like other experiments)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.3)
    sc = ax.scatter(eigs.real, eigs.imag, s=40, c=mags, cmap='RdYlGn',
                    vmin=0, vmax=1, edgecolors='black', linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label='|λ|')
    ax.set_xlabel(r'Re($\lambda$)'); ax.set_ylabel(r'Im($\lambda$)')
    ax.set_title(f'Discrete Eigenvalues expm(A)\n'
                 f'|λ| range: [{mags.min():.4f}, {mags.max():.4f}]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.axvline(0, color='red', ls='--', lw=1, alpha=0.5)
    sc2 = ax.scatter(eigs_A.real, eigs_A.imag, s=40, c=np.abs(eigs_A),
                     cmap='RdYlGn', edgecolors='black', linewidths=0.5, zorder=3)
    plt.colorbar(sc2, ax=ax, label='|μ|')
    ax.set_xlabel(r'Re($\mu$)'); ax.set_ylabel(r'Im($\mu$)')
    ax.set_title(f'Generator Eigenvalues A = K - L$^T$L\n'
                 f'max Re(μ) = {eigs_A.real.max():.4f}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    fig.suptitle(f'Stable Koopman Spectrum (Standard Map K={K})', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'eigenvalues_stable.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'eigenvalues_stable.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved eigenvalues_stable.png/pdf")

    print(f"\nAll results in {OUTDIR}/")
