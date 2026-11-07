"""
Reproduce Table 5: angular MSE vs. prediction horizon for the Arnold cat map.

Trains the [4, 24, 1024] KAN Koopman model with the stable propagator from
scratch, then reports Model, Persistence, and Random errors at steps 1, 3, 5,
and 7+. The "7+" row is the mean over steps 7 to MAX_HORIZON.

All three columns use the same metric: squared wrapped distance on the torus,
averaged over the coordinates (x, y). On this scale Persistence and Random are
both 1/12 ~ 0.083.

Usage (from the repository root):
    python experiments/table_5.py                    # train, then evaluate
    python experiments/table_5.py --checkpoint results/cat_map_standard/cat_map_model.pt
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kan import KANLinear

OUTDIR = Path('results/table_5')

CAT_MATRIX = np.array([[2, 1], [1, 1]], dtype=np.float64)

# Training configuration (matches results/cat_map_standard/config.json)
N_TRAJ = 200
N_STEPS = 500
MID_SIZE = 24
LATENT_DIM = 1024
GRID_SIZE = 5
SPLINE_ORDER = 3
EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 2048
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
RECON_WEIGHT = 0.5

# Evaluation configuration
N_ICS = 1000
EVAL_SEED = 77
MAX_HORIZON = 20
TABLE_STEPS = (1, 3, 5)
SATURATED_FROM = 7

# Table 5 as printed in the paper, for comparison: (model, persistence, random)
PUBLISHED = {'1': (2.36e-6, 0.082, 0.085), '3': (5.53e-5, 0.082, 0.082),
             '5': (2.54e-3, 0.084, 0.084), '7+': (8.22e-2, 0.084, 0.083)}


def simulate_cat_map(n_traj, n_steps, seed=42):
    """Return (n_traj, n_steps, 2) cat map trajectories on [0,1)^2."""
    np.random.seed(seed)
    data = np.zeros((n_traj, n_steps, 2))
    for i in range(n_traj):
        xy = np.random.rand(2)
        for t in range(n_steps):
            data[i, t] = xy
            xy = (CAT_MATRIX @ xy) % 1.0
    return torch.tensor(data, dtype=torch.float32)


def pair_dataset(traj):
    """Build a dataset of (state_n, state_{n+1}) pairs."""
    return TensorDataset(traj[:, :-1].reshape(-1, 2), traj[:, 1:].reshape(-1, 2))


def wrap(diff):
    """Map differences on the unit torus to [-0.5, 0.5]."""
    return diff - diff.round() if torch.is_tensor(diff) else diff - np.round(diff)


def angular_mse(pred, true):
    """Mean squared wrapped distance, averaged over samples and coordinates."""
    return (wrap(pred - true) ** 2).mean()


class StablePropagator(nn.Module):
    """Propagator M = expm(K - L^T L), with K skew-symmetric, so |lambda| <= 1."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        n_off = dim * (dim - 1) // 2
        self.skew_params = nn.Parameter(0.01 * torch.randn(n_off))
        self.L_diag = nn.Parameter(0.01 * torch.randn(dim))
        self.L_lower = nn.Parameter(0.01 * torch.randn(n_off))

    def get_matrix(self):
        n = self.dim
        device = self.skew_params.device
        K = torch.zeros(n, n, device=device)
        idx = torch.triu_indices(n, n, offset=1)
        K[idx[0], idx[1]] = self.skew_params
        L = torch.zeros(n, n, device=device)
        L[range(n), range(n)] = self.L_diag
        idx = torch.tril_indices(n, n, offset=-1)
        L[idx[0], idx[1]] = self.L_lower
        return torch.matrix_exp((K - K.T) - L.T @ L)

    def forward(self, z):
        return z @ self.get_matrix().T


class CatMapEncoder(nn.Module):
    """Embed (x, y) as (cos, sin) pairs on the torus, then apply two KAN layers."""

    def __init__(self, mid_size, latent_size, grid_size, spline_order):
        super().__init__()
        self.kan1 = KANLinear(4, mid_size, grid_size=grid_size,
                              spline_order=spline_order, grid_range=[-1.5, 1.5])
        self.kan2 = KANLinear(mid_size, latent_size, grid_size=grid_size,
                              spline_order=spline_order, grid_range=[-5, 5])

    def forward(self, xy):
        theta = 2 * np.pi * xy
        h = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        return self.kan2(self.kan1(h))


class CatMapDecoder(nn.Module):
    """Apply two KAN layers, then recover the angles on [0,1)^2 with atan2."""

    def __init__(self, latent_size, mid_size, grid_size, spline_order):
        super().__init__()
        self.kan1 = KANLinear(latent_size, mid_size, grid_size=grid_size,
                              spline_order=spline_order, grid_range=[-5, 5])
        self.kan2 = KANLinear(mid_size, 4, grid_size=grid_size,
                              spline_order=spline_order, grid_range=[-5, 5])

    def forward(self, z):
        h = self.kan2(self.kan1(z))
        x = torch.atan2(h[:, 1], h[:, 0]) / (2 * np.pi) % 1.0
        y = torch.atan2(h[:, 3], h[:, 2]) / (2 * np.pi) % 1.0
        return torch.stack([x, y], dim=-1)


class CatMapKoopman(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CatMapEncoder(MID_SIZE, LATENT_DIM, GRID_SIZE, SPLINE_ORDER)
        self.decoder = CatMapDecoder(LATENT_DIM, MID_SIZE, GRID_SIZE, SPLINE_ORDER)
        self.prop = StablePropagator(LATENT_DIM)

    def loss(self, xy_n, xy_np1):
        z = self.encoder(xy_n)
        z_next = self.prop(z)
        return (angular_mse(self.decoder(z_next), xy_np1)
                + nn.functional.mse_loss(z_next, self.encoder(xy_np1))
                + RECON_WEIGHT * angular_mse(self.decoder(z), xy_n))

    def step(self, xy):
        return self.decoder(self.prop(self.encoder(xy)))


def train(model, traj, epochs, device):
    """Train on the first 80% of trajectories and track loss on the rest."""
    n_train = int(0.8 * traj.shape[0])
    train_dl = DataLoader(pair_dataset(traj[:n_train]), batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True)
    val_dl = DataLoader(pair_dataset(traj[n_train:]), batch_size=BATCH_SIZE)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xy_n, xy_np1 in train_dl:
            loss = model.loss(xy_n.to(device), xy_np1.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())
        scheduler.step()

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                val = np.mean([model.loss(a.to(device), b.to(device)).item()
                               for a, b in val_dl])
            print(f"  Epoch {ep:3d} | train {np.mean(losses):.3e} | val {val:.3e}",
                  flush=True)

    return model.cpu().eval()


def evaluate(model):
    """Return {row label: (model, persistence, random)} for Table 5."""
    rng = np.random.RandomState(EVAL_SEED)
    ics = rng.rand(N_ICS, 2).astype(np.float32)

    true_xy = ics.astype(np.float64)
    model_xy = torch.tensor(ics)
    errors = np.zeros((MAX_HORIZON, 3))

    with torch.no_grad():
        for k in range(MAX_HORIZON):
            true_xy = (true_xy @ CAT_MATRIX.T) % 1.0
            model_xy = model.step(model_xy)
            errors[k] = (angular_mse(model_xy.numpy(), true_xy),
                         angular_mse(ics, true_xy),
                         angular_mse(rng.rand(N_ICS, 2), true_xy))

    rows = {str(k): tuple(errors[k - 1]) for k in TABLE_STEPS}
    rows[f'{SATURATED_FROM}+'] = tuple(errors[SATURATED_FROM - 1:].mean(axis=0))
    return rows


def print_table(rows):
    print(f"\n{'Steps':<6}{'Model MSE':>12}{'Persistence':>13}{'Random':>9}"
          f"   | {'Published':>9}{'Persist.':>10}{'Random':>8}")
    for label, (m, p, r) in rows.items():
        pm, pp, pr = PUBLISHED[label]
        print(f"{label:<6}{m:>12.2e}{p:>13.3f}{r:>9.3f}"
              f"   | {pm:>9.2e}{pp:>10.3f}{pr:>8.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--checkpoint', type=Path,
                        help='evaluate this saved model instead of training')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = CatMapKoopman()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
    else:
        traj = simulate_cat_map(N_TRAJ, N_STEPS)
        print(f"Training for {args.epochs} epochs on {args.device}...", flush=True)
        model = train(model, traj, args.epochs, args.device)
        torch.save({'model_state_dict': model.state_dict()}, OUTDIR / 'model.pt')

    rows = evaluate(model)
    print_table(rows)

    with open(OUTDIR / 'table_5.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['steps', 'model_mse', 'persistence', 'random'])
        writer.writerows([label, *values] for label, values in rows.items())
    print(f"\nSaved {OUTDIR / 'table_5.csv'}")


if __name__ == '__main__':
    main()
