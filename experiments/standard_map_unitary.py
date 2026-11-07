"""
Standard Map with unitary Koopman propagator.

Forces all eigenvalues onto the unit circle via M = exp(iH) where H is
a real skew-symmetric matrix (so M is orthogonal/area-preserving).

This prevents the damping that empties the chaotic sea.
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
# Orthogonal propagator via Cayley transform of skew-symmetric matrix
# =====================================================================

class OrthogonalPropagator(nn.Module):
    """M = (I - A)(I + A)^{-1} where A is skew-symmetric.

    Guarantees M^T M = I (orthogonal), so all eigenvalues have |λ| = 1.
    For area-preserving maps this is the correct constraint.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Skew-symmetric: parameterize upper triangle only
        n_params = dim * (dim - 1) // 2
        self.triu_vals = nn.Parameter(0.01 * torch.randn(n_params))

    def get_skew(self):
        A = torch.zeros(self.dim, self.dim, device=self.triu_vals.device)
        idx = torch.triu_indices(self.dim, self.dim, offset=1)
        A[idx[0], idx[1]] = self.triu_vals
        return A - A.T  # skew-symmetric

    def get_matrix(self):
        A = self.get_skew()
        I = torch.eye(self.dim, device=A.device)
        return torch.linalg.solve(I + A, I - A)

    def forward(self, z):
        M = self.get_matrix()
        return z @ M.T


# =====================================================================
# Encoder / Decoder (same as standard_map.py)
# =====================================================================

class KANEncoder(nn.Module):
    def __init__(self, mid_size, latent_size, grid_size=5, spline_order=3):
        super().__init__()
        kan_kw = dict(grid_size=grid_size, spline_order=spline_order,
                      grid_range=[-1.5, 1.5])
        self.kan1 = KANLinear(4, mid_size, **kan_kw)
        self.kan2 = KANLinear(mid_size, latent_size,
                              grid_size=grid_size, spline_order=spline_order,
                              grid_range=[-5, 5])

    def forward(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        h = torch.stack([torch.cos(x), torch.sin(x),
                         torch.cos(y), torch.sin(y)], dim=-1)
        return self.kan2(self.kan1(h))


class KANDecoder(nn.Module):
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
        x = torch.atan2(h[:, 1], h[:, 0]) % (2 * np.pi)
        y = torch.atan2(h[:, 3], h[:, 2]) % (2 * np.pi)
        return torch.stack([x, y], dim=-1)


class UnitaryKoopmanKAN(nn.Module):
    """Discrete Koopman with orthogonal (unitary) propagator."""
    def __init__(self, mid_size, latent_dim):
        super().__init__()
        self.encoder = KANEncoder(mid_size, latent_dim)
        self.decoder = KANDecoder(latent_dim, mid_size)
        self.prop = OrthogonalPropagator(latent_dim)
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

def train(mid_size, latent_dim, traj, epochs=300, lr=1e-3, device='cpu'):
    n_train = int(0.8 * traj.shape[0])
    train_dl = DataLoader(MapPairDataset(traj[:n_train]),
                          batch_size=2048, shuffle=True, drop_last=True)
    val_dl = DataLoader(MapPairDataset(traj[n_train:]),
                        batch_size=2048)

    model = UnitaryKoopmanKAN(mid_size, latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Unitary Koopman [4,{mid_size},{latent_dim}]: {n_params:,} params")

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

        if ep == 1 or ep % 25 == 0 or ep == epochs:
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
            mag_range = f'[{np.abs(eigs).min():.4f}, {np.abs(eigs).max():.4f}]'
            print(f"  Ep {ep:3d} | val {np.mean(vloss):.5f} "
                  f"| |λ| range: {mag_range}", flush=True)

    return model.cpu().eval()


# =====================================================================
# Rollout
# =====================================================================

def rollout_phase_space(model, n_ics=500, n_steps=10000):
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
            if (ic + 1) % 100 == 0:
                print(f"    {ic+1}/{n_ics} ICs done", flush=True)

    return np.array(pts)


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=float, default=0.97)
    parser.add_argument('--mid', type=int, default=16)
    parser.add_argument('--latent', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    K = args.K

    print(f"Standard Map K={K}: Unitary Koopman")
    print(f"Device: {args.device}")

    # Simulate
    traj = simulate_standard_map(n_traj=200, n_steps=2000, K=K,
                                  device=args.device).cpu()

    # Train unitary model
    model = train(args.mid, args.latent, traj, epochs=args.epochs,
                  device=args.device)

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'mid': args.mid, 'latent': args.latent, 'K': K},
    }, f'old/standard_map_unitary_{args.mid}_{args.latent}.pt')
    print(f"Saved checkpoint")

    # Eigenvalue check
    M = model.koopman_matrix().numpy()
    eigs = np.linalg.eigvals(M)
    print(f"\nEigenvalue magnitudes: min={np.abs(eigs).min():.6f}, "
          f"max={np.abs(eigs).max():.6f}")
    print(f"All on unit circle? {np.allclose(np.abs(eigs), 1.0, atol=1e-4)}")

    # Rollout
    print(f"\nRolling out 500 ICs × 10,000 steps...")
    model_pts = rollout_phase_space(model, n_ics=500, n_steps=10000)

    # True phase space
    traj_true = simulate_standard_map(n_traj=500, n_steps=10000, K=K).cpu()
    true_pts = traj_true.reshape(-1, 2).numpy()

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sub = np.random.choice(len(true_pts), min(200000, len(true_pts)), replace=False)
    axes[0].scatter(true_pts[sub, 0], true_pts[sub, 1], s=0.05, alpha=0.2,
                    c='black', rasterized=True)
    axes[0].set_title('True Phase Space', fontsize=13)
    axes[0].set_xlim(0, 2*np.pi); axes[0].set_ylim(0, 2*np.pi)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')

    sub2 = np.random.choice(len(model_pts), min(200000, len(model_pts)), replace=False)
    axes[1].scatter(model_pts[sub2, 0], model_pts[sub2, 1], s=0.05, alpha=0.2,
                    c='steelblue', rasterized=True)
    axes[1].set_title('Unitary Koopman Rollout', fontsize=13)
    axes[1].set_xlim(0, 2*np.pi); axes[1].set_ylim(0, 2*np.pi)
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('$x$')

    fig.suptitle(f'Standard Map K={K}: Unitary Propagator (|λ|=1 enforced)\n'
                 f'500 ICs × 10,000 steps', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'phase_space_unitary.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'phase_space_unitary.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved phase_space_unitary.png/pdf")

    # Eigenvalue plot
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.3)
    ax.scatter(eigs.real, eigs.imag, s=50, c='steelblue', edgecolors='black',
               linewidths=0.5, zorder=3)
    ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')
    ax.set_title(f'Unitary Koopman Eigenvalues (all |λ|=1)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'eigenvalues_unitary.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved eigenvalues_unitary.png")

    print(f"\nAll results in {OUTDIR}/")
