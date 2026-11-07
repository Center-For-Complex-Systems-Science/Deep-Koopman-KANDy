import torch
from torch.utils.data import Dataset


def lorenz_rhs(x, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Lorenz system right-hand side. x: (..., 3)"""
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    dx1 = sigma * (x2 - x1)
    dx2 = x1 * (rho - x3) - x2
    dx3 = x1 * x2 - beta * x3
    return torch.stack([dx1, dx2, dx3], dim=-1)


@torch.no_grad()
def rk4_step(x, dt, rhs_fn):
    k1 = rhs_fn(x)
    k2 = rhs_fn(x + 0.5 * dt * k1)
    k3 = rhs_fn(x + 0.5 * dt * k2)
    k4 = rhs_fn(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


@torch.no_grad()
def simulate_lorenz(
    n_traj=50,
    n_steps=5000,
    dt=0.01,
    burn_in=200,
    init_box=([-20, -30, 0], [20, 30, 50]),
    device="cpu",
):
    """
    Returns:
        traj: (n_traj, n_steps, 3) after burn-in
    """
    lo = torch.tensor(init_box[0], dtype=torch.float32, device=device)
    hi = torch.tensor(init_box[1], dtype=torch.float32, device=device)

    x0 = lo + (hi - lo) * torch.rand(n_traj, 3, device=device)
    x = x0.clone()

    for _ in range(burn_in):
        x = rk4_step(x, dt, lorenz_rhs)

    traj = torch.zeros(n_traj, n_steps, 3, device=device)
    for t in range(n_steps):
        traj[:, t, :] = x
        x = rk4_step(x, dt, lorenz_rhs)

    return traj


class PairDataset(Dataset):
    """Dataset of consecutive (x_t, x_{t+1}) pairs from trajectory data."""

    def __init__(self, traj: torch.Tensor):
        """
        traj: (n_traj, n_steps, 3)
        """
        self.traj = traj
        self.n_traj, self.n_steps, self.d = traj.shape
        self.n_pairs = self.n_traj * (self.n_steps - 1)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        traj_id = idx // (self.n_steps - 1)
        t = idx % (self.n_steps - 1)
        x_t = self.traj[traj_id, t]
        x_tp1 = self.traj[traj_id, t + 1]
        return x_t, x_tp1
