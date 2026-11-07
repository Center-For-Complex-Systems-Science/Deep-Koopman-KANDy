"""
Reproduce Table 6: effect of model size on cat-map prediction.

Trains the KAN Koopman model from table_5.py at each size in the table, then
reports parameters, validation loss, max |lambda|, and forecast horizon.

Each size has its own checkpoint in results/table_6/. An existing checkpoint is
loaded rather than retrained, so no run replaces a trained model. Delete a
checkpoint to retrain that size.

Usage (from the repository root):
    python experiments/table_6.py                       # all three sizes
    python experiments/table_6.py --arch 4,16,32 4,32,64
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from table_5 import (BATCH_SIZE, CAT_MATRIX, EVAL_SEED, GRID_SIZE, MAX_HORIZON,
                     N_ICS, N_STEPS, N_TRAJ, SPLINE_ORDER, CatMapDecoder,
                     CatMapEncoder, CatMapKoopman, StablePropagator,
                     angular_mse, pair_dataset, simulate_cat_map, train)

OUTDIR = Path('results/table_6')

# (mid size, latent size): training epochs. The epoch count for [4,32,64] is not
# recorded anywhere; 150 is the default in cat_map_stable.py.
ARCHS = {(16, 32): 150, (32, 64): 150, (24, 1024): 200}

# The horizon is the number of steps before the model's angular MSE first reaches
# this value. error_curves.csv has the full curves for any other threshold.
HORIZON_THRESHOLD = 3e-2

# Table 6 as printed in the paper: (parameters, val loss, max |lambda|, horizon)
PUBLISHED = {(16, 32): (12544, 7e-3, 0.985, 2),
             (32, 64): (47616, 7e-4, 0.990, 3),
             (24, 1024): (1542016, 6e-5, 0.958, 6)}


class SizedCatMapKoopman(CatMapKoopman):
    """CatMapKoopman with its layer sizes passed in."""

    def __init__(self, mid_size, latent_dim):
        nn.Module.__init__(self)
        self.encoder = CatMapEncoder(mid_size, latent_dim, GRID_SIZE, SPLINE_ORDER)
        self.decoder = CatMapDecoder(latent_dim, mid_size, GRID_SIZE, SPLINE_ORDER)
        self.prop = StablePropagator(latent_dim)


def arch_name(arch):
    return f'4_{arch[0]}_{arch[1]}'


def parse_arch(text):
    """Convert '4,16,32' to (16, 32)."""
    arch = tuple(int(v) for v in text.split(',')[1:])
    if arch not in ARCHS:
        raise argparse.ArgumentTypeError(f'{text} is not a row of Table 6')
    return arch


def val_loss(model, traj):
    """Training loss on the held-out 20% of trajectories."""
    n_train = int(0.8 * traj.shape[0])
    val_dl = DataLoader(pair_dataset(traj[n_train:]), batch_size=BATCH_SIZE)
    with torch.no_grad():
        return float(np.mean([model.loss(a, b).item() for a, b in val_dl]))


def error_curve(model):
    """Model angular MSE at steps 1 to MAX_HORIZON, using Table 5's test points."""
    ics = np.random.RandomState(EVAL_SEED).rand(N_ICS, 2).astype(np.float32)
    true_xy = ics.astype(np.float64)
    model_xy = torch.tensor(ics)
    errors = np.zeros(MAX_HORIZON)

    with torch.no_grad():
        for k in range(MAX_HORIZON):
            true_xy = (true_xy @ CAT_MATRIX.T) % 1.0
            model_xy = model.step(model_xy)
            errors[k] = angular_mse(model_xy.numpy(), true_xy)
    return errors


def horizon(errors):
    above = np.nonzero(errors >= HORIZON_THRESHOLD)[0]
    return int(above[0]) if len(above) else len(errors)


def load_or_train(arch, traj, selected, device):
    """Return the model for this size, or None if it is neither saved nor selected."""
    path = OUTDIR / f'model_{arch_name(arch)}.pt'
    torch.manual_seed(0)
    model = SizedCatMapKoopman(*arch)

    if path.exists():
        print(f"Loading {path}", flush=True)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        return model.eval()
    if arch not in selected:
        return None

    print(f"Training [4,{arch[0]},{arch[1]}] for {ARCHS[arch]} epochs on {device}...",
          flush=True)
    model = train(model, traj, ARCHS[arch], device)
    torch.save({'model_state_dict': model.state_dict(), 'epochs': ARCHS[arch]}, path)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--arch', type=parse_arch, nargs='+', default=list(ARCHS),
                        help='sizes to train, such as 4,16,32 (default: all)')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    traj = simulate_cat_map(N_TRAJ, N_STEPS)

    rows, curves = [], []
    for arch in ARCHS:
        model = load_or_train(arch, traj, args.arch, args.device)
        if model is None:
            continue
        eigs = np.linalg.eigvals(model.prop.get_matrix().detach().double().numpy())
        errors = error_curve(model)
        rows.append([f'[4,{arch[0]},{arch[1]}]',
                     sum(p.numel() for p in model.parameters()),
                     val_loss(model, traj), np.abs(eigs).max(), horizon(errors)])
        curves.append([rows[-1][0], *errors])

        print(f"\n{'':<10}{'Params':>10}{'Val loss':>11}{'max|lam|':>10}{'Horizon':>9}")
        print(f"{'Measured':<10}{rows[-1][1]:>10}{rows[-1][2]:>11.1e}"
              f"{rows[-1][3]:>10.3f}{rows[-1][4]:>9}")
        p = PUBLISHED[arch]
        print(f"{'Published':<10}{p[0]:>10}{p[1]:>11.1e}{p[2]:>10.3f}{p[3]:>9}\n",
              flush=True)

    with open(OUTDIR / 'table_6.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['architecture', 'parameters', 'val_loss', 'max_abs_lambda',
                         'horizon'])
        writer.writerows(rows)
    with open(OUTDIR / 'error_curves.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['architecture', *range(1, MAX_HORIZON + 1)])
        writer.writerows(curves)
    print(f"Saved {OUTDIR / 'table_6.csv'} and {OUTDIR / 'error_curves.csv'}")


if __name__ == '__main__':
    main()
