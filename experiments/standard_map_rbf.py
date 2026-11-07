"""
Standard Map: RBF level-set analysis + RBF-based KAN.

1. Fit torus-aware RBF implicit surface to the chaotic sea boundary
2. Compare true vs model rollout boundaries
3. Train a KAN with RBF base activation and compare to SiLU
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
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening, binary_fill_holes

from kan import KANLinear
from training import set_seed
from standard_map import simulate_standard_map, MapPairDataset, angular_mse

OUTDIR = Path('results/standard_map')
K_PARAM = 0.97


# =====================================================================
# RBF level-set fitting (adapted from user's code)
# =====================================================================

def wrap_2pi(a):
    return np.mod(a, 2.0 * np.pi)


def embed_torus(x, p):
    """Map (x, p) on torus to R^4: (cos x, sin x, cos p, sin p)."""
    return np.column_stack([np.cos(x), np.sin(x), np.cos(p), np.sin(p)])


def build_occupancy_mask(xs, ps, grid_size=320, sigma=1.6, threshold_ratio=0.02):
    """Build binary mask of occupied phase space from scatter data."""
    edges = np.linspace(0.0, 2*np.pi, grid_size + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    H, _, _ = np.histogram2d(ps, xs, bins=[edges, edges])
    Hs = gaussian_filter(H, sigma=sigma, mode='wrap')

    mask = Hs > (threshold_ratio * Hs.max())
    mask = binary_closing(mask)
    mask = binary_opening(mask)
    mask = binary_fill_holes(mask)

    return centers, centers, H, Hs, mask


def rbf_features_torus(X, centers, sigma):
    """Torus-aware RBF features using embedding distance."""
    Xe = embed_torus(X[:, 0], X[:, 1])
    Ce = embed_torus(centers[:, 0], centers[:, 1])
    d2 = ((Xe[:, None, :] - Ce[None, :, :]) ** 2).sum(axis=2)
    Phi = np.exp(-d2 / (2.0 * sigma * sigma))
    return np.hstack([np.ones((X.shape[0], 1)), X, Phi])


def fit_rbf_level_set(xs, ps, grid_size=320, n_inside=6000, n_outside=6000,
                       n_centers=320, ridge=1e-3, seed=13):
    """Fit RBF implicit function: g < 0 = chaotic sea, g > 0 = islands."""
    x_centers, p_centers, H, Hs, mask = build_occupancy_mask(xs, ps, grid_size)

    rng = np.random.default_rng(seed)
    inside = np.argwhere(mask)
    outside = np.argwhere(~mask)

    n_in = min(n_inside, len(inside))
    n_out = min(n_outside, len(outside))

    in_idx = inside[rng.choice(len(inside), size=n_in, replace=False)]
    out_idx = outside[rng.choice(len(outside), size=n_out, replace=False)]

    pts = np.vstack([in_idx, out_idx])
    labels = np.hstack([-np.ones(n_in), np.ones(n_out)])

    X_train = np.column_stack([x_centers[pts[:, 1]], p_centers[pts[:, 0]]])

    centers = X_train[rng.choice(len(X_train), size=min(n_centers, len(X_train)),
                                  replace=False)]

    # Auto-select RBF width
    probe = X_train[rng.choice(len(X_train), size=min(1200, len(X_train)),
                                replace=False)]
    probe_e = embed_torus(probe[:, 0], probe[:, 1])
    center_e = embed_torus(centers[:, 0], centers[:, 1])
    d = ((probe_e[:, None, :] - center_e[None, :, :]) ** 2).sum(axis=2) ** 0.5
    sigma = np.median(np.min(d, axis=1)) * 2.3

    Phi = rbf_features_torus(X_train, centers, sigma)
    w = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(Phi.shape[1]), Phi.T @ labels)

    scores = Phi @ w
    in_acc = np.mean(scores[:n_in] < 0.0)
    out_acc = np.mean(scores[n_in:] > 0.0)

    return {
        'centers': centers, 'sigma': sigma, 'w': w,
        'x_grid': x_centers, 'p_grid': p_centers, 'mask': mask,
        'inside_acc': in_acc, 'outside_acc': out_acc,
    }


def evaluate_rbf_grid(x_centers, p_centers, rbf_result):
    """Evaluate RBF level-set on grid."""
    Xg, Pg = np.meshgrid(x_centers, p_centers)
    pts = np.column_stack([Xg.ravel(), Pg.ravel()])
    G = (rbf_features_torus(pts, rbf_result['centers'], rbf_result['sigma'])
         @ rbf_result['w']).reshape(Xg.shape)
    return Xg, Pg, G


# =====================================================================
# RBF base activation for KAN
# =====================================================================

class RBFActivation(nn.Module):
    """Gaussian RBF activation: exp(-x²)."""
    def forward(self, x):
        return torch.exp(-x * x)


# =====================================================================
# Unitary Koopman model (reuse from standard_map_unitary.py)
# =====================================================================

class OrthogonalPropagator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        n_params = dim * (dim - 1) // 2
        self.triu_vals = nn.Parameter(0.01 * torch.randn(n_params))

    def get_matrix(self):
        A = torch.zeros(self.dim, self.dim, device=self.triu_vals.device)
        idx = torch.triu_indices(self.dim, self.dim, offset=1)
        A[idx[0], idx[1]] = self.triu_vals
        A = A - A.T
        I = torch.eye(self.dim, device=A.device)
        return torch.linalg.solve(I + A, I - A)

    def forward(self, z):
        return z @ self.get_matrix().T


class KANEncoder(nn.Module):
    def __init__(self, mid_size, latent_size, base_activation=nn.SiLU):
        super().__init__()
        self.kan1 = KANLinear(4, mid_size, grid_size=5, spline_order=3,
                              grid_range=[-1.5, 1.5], base_activation=base_activation)
        self.kan2 = KANLinear(mid_size, latent_size, grid_size=5, spline_order=3,
                              grid_range=[-5, 5], base_activation=base_activation)

    def forward(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        h = torch.stack([torch.cos(x), torch.sin(x),
                         torch.cos(y), torch.sin(y)], dim=-1)
        return self.kan2(self.kan1(h))


class KANDecoder(nn.Module):
    def __init__(self, latent_size, mid_size, base_activation=nn.SiLU):
        super().__init__()
        self.kan1 = KANLinear(latent_size, mid_size, grid_size=5, spline_order=3,
                              grid_range=[-5, 5], base_activation=base_activation)
        self.kan2 = KANLinear(mid_size, 4, grid_size=5, spline_order=3,
                              grid_range=[-5, 5], base_activation=base_activation)

    def forward(self, z):
        h = self.kan2(self.kan1(z))
        x = torch.atan2(h[:, 1], h[:, 0]) % (2 * np.pi)
        y = torch.atan2(h[:, 3], h[:, 2]) % (2 * np.pi)
        return torch.stack([x, y], dim=-1)


class UnitaryKoopmanKAN(nn.Module):
    def __init__(self, mid_size, latent_dim, base_activation=nn.SiLU):
        super().__init__()
        self.encoder = KANEncoder(mid_size, latent_dim, base_activation)
        self.decoder = KANDecoder(latent_dim, mid_size, base_activation)
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

def train_model(mid_size, latent_dim, traj, base_activation=nn.SiLU,
                epochs=300, lr=1e-3, device='cpu', label=''):
    n_train = int(0.8 * traj.shape[0])
    train_dl = DataLoader(MapPairDataset(traj[:n_train]),
                          batch_size=2048, shuffle=True, drop_last=True)
    val_dl = DataLoader(MapPairDataset(traj[n_train:]),
                        batch_size=2048)

    model = UnitaryKoopmanKAN(mid_size, latent_dim, base_activation).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {label}: {n_params:,} params")

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
            print(f"    Ep {ep:3d} | val {np.mean(vloss):.5f}", flush=True)

    return model.cpu().eval()


def rollout(model, n_ics=300, n_steps=8000):
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
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(0)

    print(f"Standard Map K={K_PARAM}: RBF Level-Set + RBF KAN")

    # ---- Simulate true orbit ----
    print("\nSimulating true chaotic orbit...")
    # Long single orbit for RBF fitting
    x, p = 0.1, 0.1
    xs, ps = [x], [p]
    for _ in range(120000):
        p = wrap_2pi(p + K_PARAM * np.sin(x))
        x = wrap_2pi(x + p)
        xs.append(x)
        ps.append(p)
    xs, ps = np.array(xs[5000:]), np.array(ps[5000:])
    print(f"  {len(xs):,} orbit points")

    # ---- Fit RBF to true chaotic sea ----
    print("\nFitting RBF level-set to true chaotic sea...")
    rbf_true = fit_rbf_level_set(xs, ps)
    print(f"  Inside acc: {rbf_true['inside_acc']:.3f}, "
          f"Outside acc: {rbf_true['outside_acc']:.3f}")

    # ---- Train two models: SiLU vs RBF base ----
    traj = simulate_standard_map(n_traj=200, n_steps=2000, K=K_PARAM,
                                  device=args.device).cpu()

    mid, latent = 16, 64

    print(f"\nTraining SiLU model [{mid},{latent}]...")
    model_silu = train_model(mid, latent, traj, base_activation=nn.SiLU,
                              epochs=300, device=args.device, label='SiLU')

    print(f"\nTraining RBF model [{mid},{latent}]...")
    model_rbf = train_model(mid, latent, traj, base_activation=RBFActivation,
                             epochs=300, device=args.device, label='RBF')

    # ---- Rollout both models ----
    print("\nRolling out SiLU model...")
    pts_silu = rollout(model_silu)
    print(f"  {len(pts_silu):,} points")

    print("Rolling out RBF model...")
    pts_rbf = rollout(model_rbf)
    print(f"  {len(pts_rbf):,} points")

    # ---- Fit RBF to model rollout data ----
    print("\nFitting RBF level-set to SiLU rollout...")
    rbf_silu = fit_rbf_level_set(pts_silu[:, 0], pts_silu[:, 1])
    print(f"  Inside acc: {rbf_silu['inside_acc']:.3f}")

    print("Fitting RBF level-set to RBF-KAN rollout...")
    rbf_rbf = fit_rbf_level_set(pts_rbf[:, 0], pts_rbf[:, 1])
    print(f"  Inside acc: {rbf_rbf['inside_acc']:.3f}")

    # ---- Evaluate level-set functions on grid ----
    grid = np.linspace(0, 2*np.pi, 200)
    Xg_true, Pg_true, G_true = evaluate_rbf_grid(rbf_true['x_grid'],
                                                    rbf_true['p_grid'], rbf_true)
    Xg_silu, Pg_silu, G_silu = evaluate_rbf_grid(rbf_silu['x_grid'],
                                                    rbf_silu['p_grid'], rbf_silu)
    Xg_rbf, Pg_rbf, G_rbf = evaluate_rbf_grid(rbf_rbf['x_grid'],
                                                  rbf_rbf['p_grid'], rbf_rbf)

    # ---- Plot: 3-panel comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # True
    ax = axes[0]
    ax.scatter(xs[::4], ps[::4], s=0.1, alpha=0.15, c='black', rasterized=True)
    ax.contour(Xg_true, Pg_true, G_true, levels=[0.0], colors='red', linewidths=2)
    ax.set_title('True Chaotic Sea + RBF Boundary', fontsize=11)
    ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
    ax.set_xlabel('$x$'); ax.set_ylabel('$p$')
    ax.set_aspect('equal')

    # SiLU model
    ax = axes[1]
    sub = np.random.choice(len(pts_silu), min(80000, len(pts_silu)), replace=False)
    ax.scatter(pts_silu[sub, 0], pts_silu[sub, 1], s=0.1, alpha=0.15,
               c='steelblue', rasterized=True)
    ax.contour(Xg_true, Pg_true, G_true, levels=[0.0], colors='red',
               linewidths=1.5, linestyles='--')
    ax.contour(Xg_silu, Pg_silu, G_silu, levels=[0.0], colors='green',
               linewidths=1.5)
    ax.set_title('SiLU KAN: rollout + boundaries\n(red=true, green=model)', fontsize=10)
    ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
    ax.set_xlabel('$x$')
    ax.set_aspect('equal')

    # RBF model
    ax = axes[2]
    sub = np.random.choice(len(pts_rbf), min(80000, len(pts_rbf)), replace=False)
    ax.scatter(pts_rbf[sub, 0], pts_rbf[sub, 1], s=0.1, alpha=0.15,
               c='darkorange', rasterized=True)
    ax.contour(Xg_true, Pg_true, G_true, levels=[0.0], colors='red',
               linewidths=1.5, linestyles='--')
    ax.contour(Xg_rbf, Pg_rbf, G_rbf, levels=[0.0], colors='green',
               linewidths=1.5)
    ax.set_title('RBF KAN: rollout + boundaries\n(red=true, green=model)', fontsize=10)
    ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
    ax.set_xlabel('$x$')
    ax.set_aspect('equal')

    fig.suptitle(f'Standard Map K={K_PARAM}: RBF Level-Set Comparison\n'
                 f'SiLU vs RBF Base Activation (unitary, latent={latent})', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'rbf_comparison.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'rbf_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved rbf_comparison.png/pdf")

    # ---- Plot: occupancy masks ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mask, title in zip(axes,
                                [rbf_true['mask'], rbf_silu['mask'], rbf_rbf['mask']],
                                ['True', 'SiLU KAN', 'RBF KAN']):
        ax.imshow(mask, origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi],
                  aspect='equal', alpha=0.7, cmap='Blues')
        ax.set_title(f'{title} Occupancy', fontsize=11)
        ax.set_xlabel('$x$'); ax.set_ylabel('$p$')
    fig.suptitle(f'Phase Space Occupancy Masks (K={K_PARAM})', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'occupancy_masks.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved occupancy_masks.png")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    # Compute overlap between model occupancy and true occupancy
    true_mask = rbf_true['mask']
    silu_mask = rbf_silu['mask']
    rbf_mask = rbf_rbf['mask']

    silu_iou = np.sum(true_mask & silu_mask) / np.sum(true_mask | silu_mask)
    rbf_iou = np.sum(true_mask & rbf_mask) / np.sum(true_mask | rbf_mask)

    print(f"  SiLU KAN occupancy IoU with true: {silu_iou:.3f}")
    print(f"  RBF  KAN occupancy IoU with true: {rbf_iou:.3f}")
    print(f"  (IoU = intersection / union of occupied pixels)")

    print(f"\nAll results in {OUTDIR}/")
