"""
Ikeda map data generation and Phase 1 attractor verification.

The Ikeda map:
    t_n = 0.4 - 6 / (1 + x_n^2 + y_n^2)
    x_{n+1} = 1 + u * (x_n cos(t_n) - y_n sin(t_n))
    y_{n+1} = u * (x_n sin(t_n) + y_n cos(t_n))

with u = 0.9.

The nonlinearity factors through g(x,y) = x^2 + y^2.  Each output
coordinate is a bilinear combination of (x,y) with coefficients
cos(t(g)), sin(t(g)) that are univariate functions of g.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset


# =====================================================================
# Ikeda map step
# =====================================================================

def ikeda_step(x, y, u=0.9):
    """One step of the Ikeda map.  x, y can be arrays."""
    r2 = x**2 + y**2
    t = 0.4 - 6.0 / (1.0 + r2)
    ct, st = np.cos(t), np.sin(t)
    x_new = 1.0 + u * (x * ct - y * st)
    y_new = u * (x * st + y * ct)
    return x_new, y_new


def ikeda_step_torch(x, y, u=0.9):
    """Torch version for use in training loops."""
    r2 = x**2 + y**2
    t = 0.4 - 6.0 / (1.0 + r2)
    ct, st = torch.cos(t), torch.sin(t)
    x_new = 1.0 + u * (x * ct - y * st)
    y_new = u * (x * st + y * ct)
    return x_new, y_new


# =====================================================================
# Trajectory generation
# =====================================================================

def generate_ikeda(n_traj=200, n_steps=2000, u=0.9, burn_in=500, seed=0):
    """Generate Ikeda map trajectories on the attractor.

    Args:
        n_traj: number of independent trajectories
        n_steps: recorded steps per trajectory (after burn-in)
        u: Ikeda parameter (0.9 gives chaotic attractor)
        burn_in: discard this many initial steps
        seed: random seed

    Returns:
        traj: (n_traj, n_steps, 2) numpy float32 array
    """
    rng = np.random.RandomState(seed)
    # Sample ICs in rough attractor bounding box
    x = rng.uniform(-0.5, 1.5, size=n_traj)
    y = rng.uniform(-2.0, 1.0, size=n_traj)

    # Burn in
    for _ in range(burn_in):
        x, y = ikeda_step(x, y, u=u)

    # Record
    traj = np.zeros((n_traj, n_steps, 2), dtype=np.float32)
    for t in range(n_steps):
        traj[:, t, 0] = x
        traj[:, t, 1] = y
        x, y = ikeda_step(x, y, u=u)

    return traj


def ikeda_attractor_sample(n_points=100000, u=0.9, burn_in=1000, seed=42):
    """Single long trajectory for visualization.

    Returns:
        pts: (n_points, 2) numpy array on the attractor
    """
    rng = np.random.RandomState(seed)
    x, y = rng.uniform(-0.5, 1.5), rng.uniform(-2.0, 1.0)
    for _ in range(burn_in):
        x, y = ikeda_step(x, y, u=u)
    pts = np.zeros((n_points, 2), dtype=np.float32)
    for i in range(n_points):
        pts[i, 0], pts[i, 1] = x, y
        x, y = ikeda_step(x, y, u=u)
    return pts


# =====================================================================
# Pair dataset (matches data.py PairDataset interface)
# =====================================================================

class IkedaPairDataset(Dataset):
    """Dataset of (state_n, state_{n+1}) pairs from Ikeda trajectories."""
    def __init__(self, traj):
        """traj: numpy or torch (n_traj, n_steps, 2)"""
        if isinstance(traj, np.ndarray):
            traj = torch.from_numpy(traj)
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
# Phase 1: Attractor verification plot
# =====================================================================

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path

    FIGDIR = Path('figures')
    FIGDIR.mkdir(exist_ok=True)

    print("Phase 1: Ikeda attractor verification")

    # Generate attractor sample
    pts = ikeda_attractor_sample(n_points=100000, u=0.9)
    print(f"  Attractor: {pts.shape[0]} points")
    print(f"  x range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"  y range: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    print(f"  r² range: [{(pts**2).sum(1).min():.3f}, {(pts**2).sum(1).max():.3f}]")

    # Verify trajectory generation interface
    traj = generate_ikeda(n_traj=10, n_steps=500, u=0.9, burn_in=500, seed=0)
    print(f"  Trajectory batch: {traj.shape}")

    # Verify PairDataset interface
    ds = IkedaPairDataset(traj)
    x_t, x_tp1 = ds[0]
    print(f"  PairDataset: {len(ds)} pairs, sample shapes {x_t.shape}, {x_tp1.shape}")

    # Verify one-step prediction
    x0, y0 = float(x_t[0]), float(x_t[1])
    x1_true, y1_true = ikeda_step(x0, y0)
    print(f"  Step check: ({x0:.4f},{y0:.4f}) → ({x1_true:.4f},{y1_true:.4f})")
    print(f"  Dataset next: ({float(x_tp1[0]):.4f},{float(x_tp1[1]):.4f})")

    # Plot attractor
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a): plain attractor
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.05, alpha=0.3, c='black', rasterized=True)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Ikeda Attractor ($u = 0.9$)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Panel (b): colored by r² = x² + y²
    r2 = pts[:, 0]**2 + pts[:, 1]**2
    ax = axes[1]
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=0.05, alpha=0.3,
                    c=r2, cmap='viridis', rasterized=True)
    plt.colorbar(sc, ax=ax, label='$r^2 = x^2 + y^2$', shrink=0.8)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Colored by $r^2$ (foliation coordinate)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    fig.suptitle('Phase 1: Ikeda Map Data Verification', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'ikeda_attractor.png', dpi=200, bbox_inches='tight')
    fig.savefig(FIGDIR / 'ikeda_attractor.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {FIGDIR / 'ikeda_attractor.png'}")
    print("Phase 1 complete.")
