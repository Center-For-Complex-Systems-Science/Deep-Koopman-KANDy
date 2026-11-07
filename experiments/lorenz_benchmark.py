"""
Lorenz Benchmark: 8 methods x 6 metrics.

Compares DMD, EDMD-poly, EDMD-learned, MLP-Koopman, 1L-KAN-Koopman,
2L-KAN-Koopman (unconstrained), 2L-KAN-Koopman (stable), and KANDy
on the Lorenz system across one-step MSE, rollout horizon, support
recovery (P/R/F1), sparsity, seed variance, and KAN HP variance.

Run from project root:
    python experiments/lorenz_benchmark.py
"""

import sys, os, json, csv, time, traceback, gc, warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import simulate_lorenz, PairDataset, lorenz_rhs
from models import KANAutoEncoder, DeepKoopmanCT
from kan import KANLinear
from training import train_deep_koopman_ct, rollout, set_seed
from manifold_analysis import full_decomposition

# Import stable model from experiments/lorenz_stable.py
from lorenz_stable import DeepKoopmanCTStable

# KANDy imports are deferred to avoid the kan.py namespace conflict.
# The local kan.py (KANLinear) shadows PyKAN's kan module, and kandy/core.py
# does `from kan import KAN` at the top level. We work around this by
# temporarily manipulating sys.path at import time inside the function
# that needs KANDy.

# ── Configuration ─────────────────────────────────────────────────────
OUTDIR = Path(PROJECT_ROOT) / "results" / "lorenz_benchmark"
OUTDIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT = 0.01
N_TRAJ = 60
N_STEPS = 6000
BURN_IN = 500
N_TRAIN = 48  # 80% of 60

LAMBDA_MAX = 0.906   # Lorenz max Lyapunov exponent (/s)
TAU_L = 1.0 / LAMBDA_MAX  # ~1.103 s
STEPS_PER_TAU = int(TAU_L / DT)  # ~110
ROLLOUT_TAUS = 10
ROLLOUT_STEPS = ROLLOUT_TAUS * STEPS_PER_TAU
N_ROLLOUT_ICS = 10

SEEDS = [0, 1, 2, 3, 4]
KAN_GRID_SIZES = [3, 5, 8]
KAN_SPLINE_ORDERS = [2, 3, 4]

TRUE_SUPPORT = {"x", "y", "z", "xy", "xz"}

# Name mapping from PolynomialLift to canonical
KANDY_NAME_MAP = {
    "x_0": "x", "x_1": "y", "x_2": "z",
    "x_0^2": "x^2", "x_0*x_1": "xy", "x_0*x_2": "xz",
    "x_1^2": "y^2", "x_1*x_2": "yz", "x_2^2": "z^2",
}

# Name mapping from manifold_analysis output to canonical
# (manifold_analysis already uses x, y, z, xy, xz, etc.)
def canonicalize_name(name):
    """Map any name variant to canonical form."""
    name = name.strip()
    # Already canonical for manifold_analysis output
    if name in {"x", "y", "z", "xy", "xz", "yz", "x^2", "y^2", "z^2", "1"}:
        return name
    # KANDy / PolynomialLift names
    if name in KANDY_NAME_MAP:
        return KANDY_NAME_MAP[name]
    # EDMD feature names are already canonical
    return name


# ── Inline model definitions ─────────────────────────────────────────

class MLPAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, X):
        Y = self.encoder(X)
        Xr = self.decoder(Y)
        return Y, Xr


class SingleLayerKANAutoEncoder(nn.Module):
    def __init__(self, input_size, latent_size, grid_size=5, spline_order=3):
        super().__init__()
        self.encoder = KANLinear(input_size, latent_size, grid_size=grid_size,
                                 spline_order=spline_order, grid_range=[-30, 50])
        self.decoder = KANLinear(latent_size, input_size, grid_size=grid_size,
                                 spline_order=spline_order, grid_range=[-10, 10])

    def forward(self, X):
        Y = self.encoder(X)
        Xr = self.decoder(Y)
        return Y, Xr


# ── Helper: polynomial lifting for EDMD ───────────────────────────────

EDMD_FEATURE_NAMES = ["x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]

def poly_lift_degree2(X):
    """X: (N,3) -> Z: (N,9) = [x, y, z, x^2, xy, xz, y^2, yz, z^2]."""
    x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    return np.column_stack([x, y, z, x**2, x*y, x*z, y**2, y*z, z**2])


# ── Metric Computation ────────────────────────────────────────────────

def compute_one_step_mse_deep(model, val_traj, dt, device):
    """One-step MSE for deep Koopman models (DeepKoopmanCT or DeepKoopmanCTStable)."""
    model.eval()
    ds = PairDataset(val_traj)
    dl = DataLoader(ds, batch_size=4096, shuffle=False)
    total_se = 0.0
    n = 0
    with torch.no_grad():
        for x_t, x_tp1 in dl:
            x_t = x_t.to(device)
            x_tp1 = x_tp1.to(device)
            out = model.forward_one_step(x_t, dt)
            se = (out["x_tp1_pred"] - x_tp1).pow(2).sum().item()
            total_se += se
            n += x_t.shape[0] * x_t.shape[1]
    return total_se / n


def compute_rollout_nrmse(pred, true):
    """Compute NRMSE per step, averaged across components.
    pred, true: (n_steps+1, 3) numpy arrays."""
    std_comp = true.reshape(-1, 3).std(axis=0)
    std_comp = np.maximum(std_comp, 1e-8)
    rmse = np.sqrt(np.mean((pred - true)**2, axis=1, keepdims=True))
    # Per-component NRMSE
    per_comp = np.sqrt((pred - true)**2) / std_comp[None, :]
    nrmse = per_comp.mean(axis=1)
    return nrmse


def compute_rollout_horizon_deep(model, val_traj, dt, device, n_ics=N_ROLLOUT_ICS):
    """Rollout horizon in Lyapunov times for deep Koopman models."""
    model.eval()
    n_val = val_traj.shape[0]
    max_steps = min(ROLLOUT_STEPS, val_traj.shape[1] - 1)
    ics = np.random.choice(n_val, size=min(n_ics, n_val), replace=False)
    horizons = []
    all_nrmse = []
    with torch.no_grad():
        for ic_idx in ics:
            x0 = val_traj[ic_idx, 0].to(device)
            true_traj = val_traj[ic_idx, :max_steps + 1].numpy()
            # Autoregressive rollout
            xs = [x0.cpu().numpy()]
            x = x0
            for _ in range(max_steps):
                out = model.forward_one_step(x.unsqueeze(0), dt)
                x = out["x_tp1_pred"].squeeze(0)
                xs.append(x.cpu().numpy())
            pred_traj = np.array(xs)
            nrmse = compute_rollout_nrmse(pred_traj, true_traj)
            all_nrmse.append(nrmse)
            # Find first time NRMSE > 0.5
            exceed = np.where(nrmse > 0.5)[0]
            if len(exceed) > 0:
                h = exceed[0] * DT / TAU_L
            else:
                h = max_steps * DT / TAU_L
            horizons.append(h)
    # Pad shorter NRMSE arrays to common length for averaging
    max_len = max(len(n) for n in all_nrmse)
    padded = [np.pad(n, (0, max_len - len(n)), constant_values=np.nan) for n in all_nrmse]
    mean_nrmse = np.nanmean(padded, axis=0)
    return float(np.mean(horizons)), mean_nrmse


def compute_support_metrics(discovered, true=TRUE_SUPPORT):
    """Compute precision, recall, F1 for support recovery."""
    discovered = set(discovered)
    true = set(true)
    tp = len(discovered & true)
    precision = tp / max(len(discovered), 1)
    recall = tp / max(len(true), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return {"precision": precision, "recall": recall, "f1": f1,
            "discovered": sorted(discovered), "true": sorted(true)}


# ── Basis sufficiency metric ─────────────────────────────────────────
# The right question: can a fixed small basis explain ALL latent dims?
# For each candidate basis, compute min R² across latent dims.
# The Lorenz basis {x,y,z,xy,xz} should be sufficient (high min-R²).

ALL_DEGREE2_MONOMIALS = ["x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]

# Candidate bases to test (nested, increasing size)
CANDIDATE_BASES = [
    ("linear",       ["x", "y", "z"]),
    ("Lorenz",       ["x", "y", "z", "xy", "xz"]),
    ("Lorenz+yz",    ["x", "y", "z", "xy", "xz", "yz"]),
    ("full-deg2",    ["x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]),
]


def compute_basis_sufficiency(coeffs_per_dim, r2_per_dim, names_per_dim,
                              r2_threshold=0.8):
    """Given Lasso decomposition of each latent dim, test which candidate
    bases are sufficient to explain all well-reconstructed dims.

    For each candidate basis B, restrict coefficients to B and recompute R².
    Report min-R² across dims (the worst-explained dim determines sufficiency).

    Returns dict with per-basis results and the smallest sufficient basis.
    """
    results = {}
    for basis_name, basis_terms in CANDIDATE_BASES:
        restricted_r2s = []
        for zi in range(len(coeffs_per_dim)):
            if r2_per_dim[zi] < r2_threshold:
                continue  # skip poorly reconstructed dims
            coeffs = coeffs_per_dim[zi]      # dict: name -> coeff
            # R² using only basis terms
            # We need the original data to recompute, but we can approximate:
            # the full-basis R² minus the contribution of out-of-basis terms
            # A simpler proxy: fraction of coefficient energy in-basis
            in_basis_energy = sum(coeffs.get(t, 0.0)**2 for t in basis_terms)
            total_energy = sum(v**2 for k, v in coeffs.items() if k != "1")
            if total_energy < 1e-15:
                restricted_r2s.append(1.0)
            else:
                # Energy ratio as proxy for R² contribution
                restricted_r2s.append(in_basis_energy / total_energy)

        if restricted_r2s:
            results[basis_name] = {
                "min_energy_ratio": float(np.min(restricted_r2s)),
                "mean_energy_ratio": float(np.mean(restricted_r2s)),
                "n_dims": len(restricted_r2s),
                "basis_size": len(basis_terms),
                "basis": basis_terms,
            }
        else:
            results[basis_name] = {
                "min_energy_ratio": 0.0, "mean_energy_ratio": 0.0,
                "n_dims": 0, "basis_size": len(basis_terms),
                "basis": basis_terms,
            }

    # Find smallest sufficient basis (min energy ratio > 0.95)
    smallest_sufficient = None
    for basis_name, basis_terms in CANDIDATE_BASES:
        if results[basis_name]["min_energy_ratio"] > 0.95:
            smallest_sufficient = basis_name
            break

    return results, smallest_sufficient


def get_support_deep_koopman(model, train_traj, latent_dim, device):
    """Extract support from a deep Koopman model via full_decomposition.

    Returns:
        support: set of discovered monomials (union, for backward compat)
        avg_sparsity: mean active terms per dim
        basis_results: dict from compute_basis_sufficiency
        decomp_details: per-dim coefficients and R² for further analysis
    """
    x_flat = train_traj.reshape(-1, 3)
    n = x_flat.shape[0]
    idx = np.random.choice(n, size=min(5000, n), replace=False)
    x_sub = x_flat[idx]
    x_tensor = torch.tensor(x_sub, dtype=torch.float32)

    all_names = set()
    sparsities = []
    coeffs_per_dim = []
    r2_per_dim = []
    names_per_dim = []
    model_cpu = model.cpu().eval()

    for zi in range(latent_dim):
        try:
            res = full_decomposition(model_cpu, x_tensor, zi,
                                     target_type="latent", max_degree=2)
            r2 = res.get("recon_r2", 0.0)
            r2_per_dim.append(r2)

            # Build coeff dict
            cdict = dict(zip(
                [canonicalize_name(n) for n in res["inner"]["names"]],
                res["inner"]["coeffs"]
            ))
            coeffs_per_dim.append(cdict)
            names_per_dim.append(res["inner"]["names"])

            active = res["inner"]["active_names"]
            n_active = len([nm for nm in active if nm != "1"])
            sparsities.append(n_active)
            if r2 > 0.8:
                for name in active:
                    cn = canonicalize_name(name)
                    if cn != "1":
                        all_names.add(cn)
        except Exception as e:
            print(f"    full_decomposition failed for z[{zi}]: {e}")
            sparsities.append(0)
            coeffs_per_dim.append({})
            r2_per_dim.append(0.0)
            names_per_dim.append([])

    model.to(device)
    avg_sparsity = float(np.mean(sparsities)) if sparsities else 0.0

    # Basis sufficiency analysis
    basis_results, smallest_sufficient = compute_basis_sufficiency(
        coeffs_per_dim, r2_per_dim, names_per_dim)

    return all_names, avg_sparsity, basis_results, smallest_sufficient


# ── Method Implementations ────────────────────────────────────────────

def run_dmd(train_traj, val_traj, seed):
    """Method 1: Dynamic Mode Decomposition (linear X_{t+1} = A X_t)."""
    print(f"\n{'='*60}")
    print(f"  DMD (seed={seed})")
    print(f"{'='*60}")
    set_seed(seed)

    # Build X_t, X_{t+1} pairs from training data
    X_t = train_traj[:, :-1, :].reshape(-1, 3).numpy()
    X_tp1 = train_traj[:, 1:, :].reshape(-1, 3).numpy()

    # Fit A via least squares: A = X_tp1.T @ X_t @ pinv(X_t.T @ X_t)
    A = X_tp1.T @ np.linalg.pinv(X_t.T)  # (3, 3)
    print(f"  A matrix:\n{A}")

    # One-step MSE on val
    Xv_t = val_traj[:, :-1, :].reshape(-1, 3).numpy()
    Xv_tp1 = val_traj[:, 1:, :].reshape(-1, 3).numpy()
    pred = (A @ Xv_t.T).T
    one_step_mse = float(np.mean((pred - Xv_tp1)**2))
    print(f"  One-step MSE: {one_step_mse:.6f}")

    # Rollout
    n_val = val_traj.shape[0]
    max_steps = min(ROLLOUT_STEPS, val_traj.shape[1] - 1)
    ics = np.random.choice(n_val, size=min(N_ROLLOUT_ICS, n_val), replace=False)
    horizons = []
    all_nrmse = []
    for ic_idx in ics:
        true_traj = val_traj[ic_idx, :max_steps + 1].numpy()
        x = true_traj[0].copy()
        xs = [x.copy()]
        for _ in range(max_steps):
            x = A @ x
            xs.append(x.copy())
        pred_traj = np.array(xs)
        nrmse = compute_rollout_nrmse(pred_traj, true_traj)
        all_nrmse.append(nrmse)
        exceed = np.where(nrmse > 0.5)[0]
        h = exceed[0] * DT / TAU_L if len(exceed) > 0 else max_steps * DT / TAU_L
        horizons.append(h)
    rollout_horizon = float(np.mean(horizons))
    mean_nrmse = np.mean(all_nrmse, axis=0)
    print(f"  Rollout horizon: {rollout_horizon:.3f} tau_L")

    # Support: DMD is always {x, y, z}
    support = {"x", "y", "z"}
    support_metrics = compute_support_metrics(support)
    # Basis sufficiency: linear basis has energy ratio 1.0, Lorenz also 1.0
    # (DMD only sees linear terms)
    support_metrics["basis_sufficiency"] = {
        "linear":    {"min_energy_ratio": 1.0, "mean_energy_ratio": 1.0, "n_dims": 3, "basis_size": 3, "basis": ["x","y","z"]},
        "Lorenz":    {"min_energy_ratio": 1.0, "mean_energy_ratio": 1.0, "n_dims": 3, "basis_size": 5, "basis": ["x","y","z","xy","xz"]},
        "Lorenz+yz": {"min_energy_ratio": 1.0, "mean_energy_ratio": 1.0, "n_dims": 3, "basis_size": 6, "basis": ["x","y","z","xy","xz","yz"]},
        "full-deg2": {"min_energy_ratio": 1.0, "mean_energy_ratio": 1.0, "n_dims": 3, "basis_size": 9, "basis": ALL_DEGREE2_MONOMIALS},
    }
    support_metrics["smallest_sufficient"] = "linear"
    sparsity = 3.0

    return {
        "one_step_mse": one_step_mse,
        "rollout_horizon": rollout_horizon,
        "rollout_nrmse": mean_nrmse.tolist(),
        "support": support_metrics,
        "sparsity": sparsity,
        "A": A.tolist(),
    }


def run_edmd_poly(train_traj, val_traj, seed):
    """Method 2: Extended DMD with degree-2 polynomial dictionary."""
    print(f"\n{'='*60}")
    print(f"  EDMD-poly (seed={seed})")
    print(f"{'='*60}")
    set_seed(seed)

    X_t = train_traj[:, :-1, :].reshape(-1, 3).numpy()
    X_tp1 = train_traj[:, 1:, :].reshape(-1, 3).numpy()

    Z_t = poly_lift_degree2(X_t)    # (N, 9)
    Z_tp1 = poly_lift_degree2(X_tp1)  # (N, 9)

    # K = Z_tp1.T @ pinv(Z_t.T)  shape (9, 9)
    K = Z_tp1.T @ np.linalg.pinv(Z_t.T)
    print(f"  K matrix shape: {K.shape}")

    # One-step MSE on val — use only state rows (0,1,2)
    Xv_t = val_traj[:, :-1, :].reshape(-1, 3).numpy()
    Xv_tp1 = val_traj[:, 1:, :].reshape(-1, 3).numpy()
    Zv_t = poly_lift_degree2(Xv_t)
    Zv_tp1_pred = (K @ Zv_t.T).T
    # Extract state prediction from first 3 columns
    pred_state = Zv_tp1_pred[:, :3]
    one_step_mse = float(np.mean((pred_state - Xv_tp1)**2))
    print(f"  One-step MSE: {one_step_mse:.6f}")

    # Rollout: re-lift at each step
    n_val = val_traj.shape[0]
    max_steps = min(ROLLOUT_STEPS, val_traj.shape[1] - 1)
    ics = np.random.choice(n_val, size=min(N_ROLLOUT_ICS, n_val), replace=False)
    horizons = []
    all_nrmse = []
    for ic_idx in ics:
        true_traj = val_traj[ic_idx, :max_steps + 1].numpy()
        x = true_traj[0].copy()
        xs = [x.copy()]
        for _ in range(max_steps):
            z = poly_lift_degree2(x.reshape(1, -1)).flatten()
            z_next = K @ z
            x = z_next[:3].copy()
            xs.append(x.copy())
        pred_traj = np.array(xs)
        nrmse = compute_rollout_nrmse(pred_traj, true_traj)
        all_nrmse.append(nrmse)
        exceed = np.where(nrmse > 0.5)[0]
        h = exceed[0] * DT / TAU_L if len(exceed) > 0 else max_steps * DT / TAU_L
        horizons.append(h)
    rollout_horizon = float(np.mean(horizons))
    mean_nrmse = np.mean(all_nrmse, axis=0)
    print(f"  Rollout horizon: {rollout_horizon:.3f} tau_L")

    # Support: threshold K - I to find which features drive dynamics
    # K ≈ I + dt*L, so K-I isolates the generator contributions.
    # Threshold the state-prediction rows (0-2) of K-I.
    K_dev = K - np.eye(K.shape[0])
    K_dev_state = K_dev[:3, :]  # (3, 9)
    threshold = 0.01 * np.abs(K_dev_state).max()
    active_cols = set()
    nonzero_counts = []
    for row in range(K_dev.shape[0]):
        row_thresh = 0.01 * np.abs(K_dev[row]).max()
        nz = np.sum(np.abs(K_dev[row]) > row_thresh)
        nonzero_counts.append(nz)
    for col in range(9):
        if np.any(np.abs(K_dev_state[:, col]) > threshold):
            active_cols.add(EDMD_FEATURE_NAMES[col])
    support = {canonicalize_name(n) for n in active_cols if canonicalize_name(n) != "1"}
    support_metrics = compute_support_metrics(support)
    sparsity = float(np.mean(nonzero_counts))

    # Basis sufficiency for EDMD: energy in each candidate basis per row of K-I
    edmd_coeffs_per_dim = []
    edmd_r2_per_dim = []
    for row in range(K_dev.shape[0]):
        cdict = dict(zip(EDMD_FEATURE_NAMES, K_dev[row]))
        edmd_coeffs_per_dim.append(cdict)
        edmd_r2_per_dim.append(1.0)  # all rows are "well-fit" for EDMD
    basis_results, smallest_suff = compute_basis_sufficiency(
        edmd_coeffs_per_dim, edmd_r2_per_dim, EDMD_FEATURE_NAMES)
    support_metrics["basis_sufficiency"] = basis_results
    support_metrics["smallest_sufficient"] = smallest_suff

    print(f"  Support: {support}")
    print(f"  P={support_metrics['precision']:.3f} R={support_metrics['recall']:.3f} F1={support_metrics['f1']:.3f}")
    for bname, br in basis_results.items():
        print(f"    basis '{bname}' (k={br['basis_size']}): "
              f"min_energy={br['min_energy_ratio']:.3f}  "
              f"mean_energy={br['mean_energy_ratio']:.3f}")
    if smallest_suff:
        print(f"  Smallest sufficient basis: {smallest_suff}")

    return {
        "one_step_mse": one_step_mse,
        "rollout_horizon": rollout_horizon,
        "rollout_nrmse": mean_nrmse.tolist(),
        "support": support_metrics,
        "sparsity": sparsity,
    }


def run_mlp_koopman(train_traj, val_traj, seed, latent_dim=10, hidden=64, epochs=80):
    """Method 4: MLP-Koopman."""
    print(f"\n{'='*60}")
    print(f"  MLP-Koopman (seed={seed})")
    print(f"{'='*60}")

    ae = MLPAutoEncoder(3, hidden, latent_dim)
    traj_full = torch.cat([train_traj, val_traj], dim=0)
    model, history, tr_traj, vl_traj, dt = train_deep_koopman_ct(
        ae, traj=traj_full, latent_dim=latent_dim, epochs=epochs,
        lr=1e-3, seed=seed, batch_size=2048, grad_clip=1.0,
        dt=DT, n_traj=N_TRAJ,
    )
    model = model.to(DEVICE).eval()

    one_step_mse = compute_one_step_mse_deep(model, vl_traj.to(DEVICE), DT, DEVICE)
    print(f"  One-step MSE: {one_step_mse:.6f}")

    rollout_horizon, mean_nrmse = compute_rollout_horizon_deep(
        model, vl_traj, DT, DEVICE)
    print(f"  Rollout horizon: {rollout_horizon:.3f} tau_L")

    # Support recovery
    support, sparsity, basis_results, smallest_suff = get_support_deep_koopman(
        model, tr_traj, latent_dim, DEVICE)
    support_metrics = compute_support_metrics(support)
    support_metrics["basis_sufficiency"] = basis_results
    support_metrics["smallest_sufficient"] = smallest_suff
    print(f"  Support (union): {support}")
    print(f"  F1: {support_metrics['f1']:.3f}, Sparsity: {sparsity:.2f}")
    for bname, br in basis_results.items():
        print(f"    basis '{bname}' (k={br['basis_size']}): "
              f"min_energy={br['min_energy_ratio']:.3f}  "
              f"mean_energy={br['mean_energy_ratio']:.3f}  "
              f"n_dims={br['n_dims']}")
    if smallest_suff:
        print(f"  Smallest sufficient basis: {smallest_suff}")

    del model, ae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    return {
        "one_step_mse": one_step_mse,
        "rollout_horizon": rollout_horizon,
        "rollout_nrmse": mean_nrmse.tolist(),
        "support": support_metrics,
        "sparsity": sparsity,
    }


def run_1l_kan_koopman(train_traj, val_traj, seed, latent_dim=10,
                        grid_size=5, spline_order=3, epochs=80):
    """Method 5: Single-layer KAN-Koopman."""
    print(f"\n{'='*60}")
    print(f"  1L-KAN-Koopman (seed={seed}, gs={grid_size}, so={spline_order})")
    print(f"{'='*60}")

    ae = SingleLayerKANAutoEncoder(3, latent_dim,
                                    grid_size=grid_size, spline_order=spline_order)
    traj_full = torch.cat([train_traj, val_traj], dim=0)
    model, history, tr_traj, vl_traj, dt = train_deep_koopman_ct(
        ae, traj=traj_full, latent_dim=latent_dim, epochs=epochs,
        lr=1e-3, seed=seed, batch_size=2048, grad_clip=1.0,
        dt=DT, n_traj=N_TRAJ,
    )
    model = model.to(DEVICE).eval()

    one_step_mse = compute_one_step_mse_deep(model, vl_traj.to(DEVICE), DT, DEVICE)
    print(f"  One-step MSE: {one_step_mse:.6f}")

    rollout_horizon, mean_nrmse = compute_rollout_horizon_deep(
        model, vl_traj, DT, DEVICE)
    print(f"  Rollout horizon: {rollout_horizon:.3f} tau_L")

    support, sparsity, basis_results, smallest_suff = get_support_deep_koopman(
        model, tr_traj, latent_dim, DEVICE)
    support_metrics = compute_support_metrics(support)
    support_metrics["basis_sufficiency"] = basis_results
    support_metrics["smallest_sufficient"] = smallest_suff
    print(f"  Support (union): {support}")
    print(f"  F1: {support_metrics['f1']:.3f}, Sparsity: {sparsity:.2f}")
    for bname, br in basis_results.items():
        print(f"    basis '{bname}' (k={br['basis_size']}): "
              f"min_energy={br['min_energy_ratio']:.3f}  "
              f"mean_energy={br['mean_energy_ratio']:.3f}  "
              f"n_dims={br['n_dims']}")
    if smallest_suff:
        print(f"  Smallest sufficient basis: {smallest_suff}")

    del model, ae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    return {
        "one_step_mse": one_step_mse,
        "rollout_horizon": rollout_horizon,
        "rollout_nrmse": mean_nrmse.tolist(),
        "support": support_metrics,
        "sparsity": sparsity,
    }


def run_2l_kan_koopman(train_traj, val_traj, seed, latent_dim=10, mid_size=8,
                        grid_size=5, spline_order=3, epochs=80):
    """Method 6: Two-layer KAN-Koopman (unconstrained)."""
    print(f"\n{'='*60}")
    print(f"  2L-KAN-Koopman unconstrained (seed={seed}, gs={grid_size}, so={spline_order})")
    print(f"{'='*60}")

    ae = KANAutoEncoder(3, mid_size, latent_dim,
                         grid_size=grid_size, spline_order=spline_order,
                         grid_range_l1=[-30, 50], grid_range_l2=[-10, 10])
    traj_full = torch.cat([train_traj, val_traj], dim=0)
    model, history, tr_traj, vl_traj, dt = train_deep_koopman_ct(
        ae, traj=traj_full, latent_dim=latent_dim, epochs=epochs,
        lr=1e-3, seed=seed, batch_size=2048, grad_clip=1.0,
        dt=DT, n_traj=N_TRAJ,
    )
    model = model.to(DEVICE).eval()

    one_step_mse = compute_one_step_mse_deep(model, vl_traj.to(DEVICE), DT, DEVICE)
    print(f"  One-step MSE: {one_step_mse:.6f}")

    rollout_horizon, mean_nrmse = compute_rollout_horizon_deep(
        model, vl_traj, DT, DEVICE)
    print(f"  Rollout horizon: {rollout_horizon:.3f} tau_L")

    support, sparsity, basis_results, smallest_suff = get_support_deep_koopman(
        model, tr_traj, latent_dim, DEVICE)
    support_metrics = compute_support_metrics(support)
    support_metrics["basis_sufficiency"] = basis_results
    support_metrics["smallest_sufficient"] = smallest_suff
    print(f"  Support (union): {support}")
    print(f"  F1: {support_metrics['f1']:.3f}, Sparsity: {sparsity:.2f}")
    for bname, br in basis_results.items():
        print(f"    basis '{bname}' (k={br['basis_size']}): "
              f"min_energy={br['min_energy_ratio']:.3f}  "
              f"mean_energy={br['mean_energy_ratio']:.3f}  "
              f"n_dims={br['n_dims']}")
    if smallest_suff:
        print(f"  Smallest sufficient basis: {smallest_suff}")

    # Return model and traj for EDMD-learned reuse (only for seed 0)
    result = {
        "one_step_mse": one_step_mse,
        "rollout_horizon": rollout_horizon,
        "rollout_nrmse": mean_nrmse.tolist(),
        "support": support_metrics,
        "sparsity": sparsity,
    }

    # Keep model for EDMD-learned if seed == 0
    if seed == 0:
        result["_model"] = model.cpu().eval()
        result["_tr_traj"] = tr_traj
        result["_vl_traj"] = vl_traj
    else:
        del model, ae
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    return result


def run_2l_kan_koopman_stable(train_traj, val_traj, seed, latent_dim=10,
                                mid_size=8, grid_size=5, spline_order=3,
                                epochs=150):
    """Method 7: Two-layer KAN-Koopman with stable generator (A = K - L^T L)."""
    print(f"\n{'='*60}")
    print(f"  2L-KAN-Koopman STABLE (seed={seed}, gs={grid_size}, so={spline_order})")
    print(f"{'='*60}")
    set_seed(seed)

    ae = KANAutoEncoder(3, mid_size, latent_dim,
                         grid_size=grid_size, spline_order=spline_order,
                         grid_range_l1=[-30, 50], grid_range_l2=[-10, 10])
    ae = ae.to(DEVICE)

    # Combine train+val for the split that train_deep_koopman_ct would do
    traj_full = torch.cat([train_traj, val_traj], dim=0).to(DEVICE)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    val_dl = DataLoader(PairDataset(traj_full[N_TRAIN:]),
                        batch_size=2048)

    model = DeepKoopmanCTStable(ae, latent_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x_t, x_tp1 in train_dl:
            x_t, x_tp1 = x_t.to(DEVICE), x_tp1.to(DEVICE)
            out = model.forward_one_step(x_t, DT)
            z_true = model.ae.encoder(x_tp1)
            loss = (mse(out["x_tp1_pred"], x_tp1)
                    + mse(out["z_tp1_pred"], z_true)
                    + 0.5 * mse(out["x_t_rec"], x_t))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        scheduler.step()

        if ep == 1 or ep % 20 == 0 or ep == epochs:
            model.eval()
            vloss = []
            with torch.no_grad():
                for x_t, x_tp1 in val_dl:
                    x_t, x_tp1 = x_t.to(DEVICE), x_tp1.to(DEVICE)
                    out = model.forward_one_step(x_t, DT)
                    z_true = model.ae.encoder(x_tp1)
                    l = (mse(out["x_tp1_pred"], x_tp1)
                         + mse(out["z_tp1_pred"], z_true)
                         + 0.5 * mse(out["x_t_rec"], x_t))
                    vloss.append(l.item())
            print(f"    Ep {ep:3d} | train {np.mean(losses):.5f} | val {np.mean(vloss):.5f}")

    model = model.eval()

    # Metrics
    val_traj_dev = traj_full[N_TRAIN:].cpu()
    one_step_mse = compute_one_step_mse_deep(model, val_traj_dev.to(DEVICE), DT, DEVICE)
    print(f"  One-step MSE: {one_step_mse:.6f}")

    rollout_horizon, mean_nrmse = compute_rollout_horizon_deep(
        model, val_traj_dev, DT, DEVICE)
    print(f"  Rollout horizon: {rollout_horizon:.3f} tau_L")

    tr_traj_cpu = traj_full[:N_TRAIN].cpu()
    support, sparsity = get_support_deep_koopman(
        model, tr_traj_cpu, latent_dim, DEVICE)
    support_metrics = compute_support_metrics(support)
    print(f"  Support: {support}")
    print(f"  F1: {support_metrics['f1']:.3f}, Sparsity: {sparsity:.2f}")

    del model, ae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    return {
        "one_step_mse": one_step_mse,
        "rollout_horizon": rollout_horizon,
        "rollout_nrmse": mean_nrmse.tolist(),
        "support": support_metrics,
        "sparsity": sparsity,
    }


def run_edmd_learned(kan_model, tr_traj, vl_traj, val_traj_orig, seed):
    """Method 3: EDMD with the 2L-KAN encoder as dictionary (from method 6)."""
    print(f"\n{'='*60}")
    print(f"  EDMD-learned (seed={seed}, using frozen 2L-KAN encoder)")
    print(f"{'='*60}")
    set_seed(seed)

    encoder = kan_model.ae.encoder
    decoder = kan_model.ae.decoder
    latent_dim = kan_model.K.weight.shape[0]

    # Build Z_t, Z_tp1 from training data
    X_t = tr_traj[:, :-1, :].reshape(-1, 3)
    X_tp1 = tr_traj[:, 1:, :].reshape(-1, 3)

    with torch.no_grad():
        Z_t = encoder(X_t).numpy()
        Z_tp1 = encoder(X_tp1).numpy()

    # Fit K via least squares
    K = Z_tp1.T @ np.linalg.pinv(Z_t.T)
    print(f"  K shape: {K.shape}")

    # One-step MSE on val
    Xv_t = vl_traj[:, :-1, :].reshape(-1, 3)
    Xv_tp1 = vl_traj[:, 1:, :].reshape(-1, 3)
    with torch.no_grad():
        Zv_t = encoder(Xv_t).numpy()
    Zv_tp1_pred = (K @ Zv_t.T).T
    with torch.no_grad():
        pred_state = decoder(torch.tensor(Zv_tp1_pred, dtype=torch.float32)).numpy()
    one_step_mse = float(np.mean((pred_state - Xv_tp1.numpy())**2))
    print(f"  One-step MSE: {one_step_mse:.6f}")

    # Rollout: encode once, propagate, decode each step
    n_val = val_traj_orig.shape[0]
    max_steps = min(ROLLOUT_STEPS, val_traj_orig.shape[1] - 1)
    ics = np.random.choice(n_val, size=min(N_ROLLOUT_ICS, n_val), replace=False)
    horizons = []
    all_nrmse = []
    for ic_idx in ics:
        true_traj = val_traj_orig[ic_idx, :max_steps + 1].numpy()
        x0 = val_traj_orig[ic_idx, 0]
        with torch.no_grad():
            z = encoder(x0.unsqueeze(0)).numpy().flatten()
        xs = [true_traj[0]]
        for _ in range(max_steps):
            z = K @ z
            with torch.no_grad():
                x_pred = decoder(torch.tensor(z, dtype=torch.float32).unsqueeze(0))
                xs.append(x_pred.squeeze(0).numpy())
        pred_traj = np.array(xs)
        nrmse = compute_rollout_nrmse(pred_traj, true_traj)
        all_nrmse.append(nrmse)
        exceed = np.where(nrmse > 0.5)[0]
        h = exceed[0] * DT / TAU_L if len(exceed) > 0 else max_steps * DT / TAU_L
        horizons.append(h)
    rollout_horizon = float(np.mean(horizons))
    mean_nrmse = np.mean(all_nrmse, axis=0)
    print(f"  Rollout horizon: {rollout_horizon:.3f} tau_L")

    # Support: reuse the deep Koopman support extraction on the KAN model
    support, sparsity, basis_results, smallest_suff = get_support_deep_koopman(
        kan_model, tr_traj, latent_dim, "cpu")
    support_metrics = compute_support_metrics(support)
    support_metrics["basis_sufficiency"] = basis_results
    support_metrics["smallest_sufficient"] = smallest_suff
    print(f"  Support (union): {support}")
    print(f"  F1: {support_metrics['f1']:.3f}, Sparsity: {sparsity:.2f}")
    for bname, br in basis_results.items():
        print(f"    basis '{bname}' (k={br['basis_size']}): "
              f"min_energy={br['min_energy_ratio']:.3f}  "
              f"mean_energy={br['mean_energy_ratio']:.3f}")
    if smallest_suff:
        print(f"  Smallest sufficient basis: {smallest_suff}")

    return {
        "one_step_mse": one_step_mse,
        "rollout_horizon": rollout_horizon,
        "rollout_nrmse": mean_nrmse.tolist(),
        "support": support_metrics,
        "sparsity": sparsity,
    }


def _import_kandy():
    """Import KANDy with path manipulation to avoid local kan.py conflict.

    The local kan.py defines KANLinear but shadows PyKAN's 'kan' package
    that kandy/core.py needs (it does `from kan import KAN`).  We temporarily
    rename kan.py on disk so Python finds the real pykan package, then rename
    it back.  This is the same strategy used in models._import_pykan but we
    need it at module level for kandy.
    """
    kan_py = os.path.join(PROJECT_ROOT, "kan.py")
    kan_tmp = os.path.join(PROJECT_ROOT, "kan.py._tmp_benchmark")
    renamed = False
    try:
        if os.path.exists(kan_py):
            os.rename(kan_py, kan_tmp)
            renamed = True
        # Also clear any cached 'kan' module so Python re-discovers it
        saved_kan = sys.modules.pop("kan", None)
        # Clear kandy modules too in case of stale imports
        to_remove = [k for k in sys.modules if k == "kandy" or k.startswith("kandy.")]
        for k in to_remove:
            del sys.modules[k]

        from kandy import KANDy as _KANDy, PolynomialLift as _PL
    finally:
        if renamed:
            os.rename(kan_tmp, kan_py)
        # Restore local kan module
        if saved_kan is not None:
            sys.modules["kan"] = saved_kan
        elif "kan" in sys.modules:
            # Remove the pykan 'kan' so it doesn't shadow local kan.py later
            del sys.modules["kan"]
    return _KANDy, _PL


def run_kandy(train_traj, val_traj, seed, grid=5, k=3, steps=500):
    """Method 8: KANDy (KAN + polynomial lift, derivative supervision)."""
    print(f"\n{'='*60}")
    print(f"  KANDy (seed={seed})")
    print(f"{'='*60}")

    KANDy, PolynomialLift = _import_kandy()

    # Flatten training trajectories into state snapshots
    X = train_traj.reshape(-1, 3).numpy()
    X_dot = lorenz_rhs(torch.tensor(X, dtype=torch.float32)).numpy()

    lift = PolynomialLift(degree=2)
    kandy_model = KANDy(lift=lift, grid=grid, k=k, steps=steps, seed=seed)
    kandy_model.fit(X, X_dot)

    # One-step MSE: predict x_dot, then x_{t+1} = x_t + dt * x_dot_pred
    Xv = val_traj[:, :-1, :].reshape(-1, 3).numpy()
    Xv_tp1 = val_traj[:, 1:, :].reshape(-1, 3).numpy()
    Xv_dot_pred = kandy_model.predict(Xv)
    Xv_pred = Xv + DT * Xv_dot_pred
    one_step_mse = float(np.mean((Xv_pred - Xv_tp1)**2))
    print(f"  One-step MSE: {one_step_mse:.6f}")

    # Rollout
    n_val = val_traj.shape[0]
    max_steps = min(ROLLOUT_STEPS, val_traj.shape[1] - 1)
    ics = np.random.choice(n_val, size=min(N_ROLLOUT_ICS, n_val), replace=False)
    horizons = []
    all_nrmse = []
    for ic_idx in ics:
        true_traj = val_traj[ic_idx, :max_steps + 1].numpy()
        x0 = true_traj[0]
        pred = kandy_model.rollout(x0, T=max_steps + 1, dt=DT)
        nrmse = compute_rollout_nrmse(pred, true_traj)
        all_nrmse.append(nrmse)
        exceed = np.where(nrmse > 0.5)[0]
        h = exceed[0] * DT / TAU_L if len(exceed) > 0 else max_steps * DT / TAU_L
        horizons.append(h)
    rollout_horizon = float(np.mean(horizons))
    mean_nrmse = np.mean(all_nrmse, axis=0)
    print(f"  Rollout horizon: {rollout_horizon:.3f} tau_L")

    # Support: threshold A matrix
    A = kandy_model.get_A()
    if A is not None:
        threshold = 0.01 * np.abs(A).max()
        active_cols = set()
        nonzero_counts = []
        feat_names = lift.feature_names
        for row in range(A.shape[0]):
            row_thresh = 0.01 * np.abs(A[row]).max()
            nz = np.sum(np.abs(A[row]) > row_thresh)
            nonzero_counts.append(nz)
        for col in range(A.shape[1]):
            if np.any(np.abs(A[:, col]) > threshold):
                cn = canonicalize_name(feat_names[col])
                if cn != "1":
                    active_cols.add(cn)
        sparsity = float(np.mean(nonzero_counts))
    else:
        active_cols = set()
        sparsity = 0.0

    support_metrics = compute_support_metrics(active_cols)

    # Basis sufficiency for KANDy: energy in each candidate basis per row of A
    if A is not None:
        kandy_coeffs = []
        kandy_r2s = []
        for row in range(A.shape[0]):
            cdict = {canonicalize_name(feat_names[j]): A[row, j]
                     for j in range(A.shape[1])}
            kandy_coeffs.append(cdict)
            kandy_r2s.append(1.0)
        basis_results, smallest_suff = compute_basis_sufficiency(
            kandy_coeffs, kandy_r2s, feat_names)
    else:
        basis_results, smallest_suff = {}, None
    support_metrics["basis_sufficiency"] = basis_results
    support_metrics["smallest_sufficient"] = smallest_suff

    print(f"  Support: {active_cols}")
    print(f"  F1: {support_metrics['f1']:.3f}, Sparsity: {sparsity:.2f}")
    if basis_results:
        for bname, br in basis_results.items():
            print(f"    basis '{bname}' (k={br['basis_size']}): "
                  f"min_energy={br['min_energy_ratio']:.3f}")
    if smallest_suff:
        print(f"  Smallest sufficient basis: {smallest_suff}")

    return {
        "one_step_mse": one_step_mse,
        "rollout_horizon": rollout_horizon,
        "rollout_nrmse": mean_nrmse.tolist(),
        "support": support_metrics,
        "sparsity": sparsity,
    }


# ── Plotting ──────────────────────────────────────────────────────────

METHOD_NAMES = [
    "DMD", "EDMD-poly", "EDMD-learned", "MLP-Koopman",
    "1L-KAN", "2L-KAN", "2L-KAN-stable", "KANDy",
]

ALL_MONOMIALS = ["x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]


def plot_comparison_table(results, path):
    """Plot 1: Formatted table of all metrics."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")

    col_labels = ["Method", "1-Step MSE", "Horizon (tau_L)",
                  "Precision", "Recall", "F1", "Sparsity"]
    cell_data = []

    for name in METHOD_NAMES:
        if name not in results:
            cell_data.append([name] + ["--"] * 6)
            continue
        r = results[name]
        # Gather seed results
        seed_mses = [r["seeds"][s]["one_step_mse"] for s in r["seeds"]]
        seed_horizons = [r["seeds"][s]["rollout_horizon"] for s in r["seeds"]]

        mse_str = f"{np.mean(seed_mses):.4e} +/- {np.std(seed_mses):.1e}"
        hor_str = f"{np.mean(seed_horizons):.2f} +/- {np.std(seed_horizons):.2f}"

        # Support from seed 0
        s0 = r["seeds"]["0"]
        p = s0["support"]["precision"]
        rec = s0["support"]["recall"]
        f1 = s0["support"]["f1"]
        sp = s0["sparsity"]

        cell_data.append([
            name, mse_str, hor_str,
            f"{p:.2f}", f"{rec:.2f}", f"{f1:.2f}", f"{sp:.1f}",
        ])

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, len(cell_data) + 1):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)

    fig.suptitle("Lorenz Benchmark: 8 Methods x 6 Metrics", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rollout_curves(results, path):
    """Plot 2: NRMSE vs Lyapunov time for all methods (seed=0)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    t_tau = np.arange(ROLLOUT_STEPS + 1) * DT / TAU_L

    colors = plt.cm.tab10(np.linspace(0, 1, len(METHOD_NAMES)))
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    for i, name in enumerate(METHOD_NAMES):
        if name not in results:
            continue
        s0 = results[name]["seeds"].get("0")
        if s0 is None or "rollout_nrmse" not in s0:
            continue
        nrmse = np.array(s0["rollout_nrmse"])
        ax.plot(t_tau[:len(nrmse)], nrmse, label=name,
                color=colors[i], ls=linestyles[i], lw=1.5)

    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Threshold (0.5)")
    ax.set_xlabel(r"Time ($\tau_\Lambda$)", fontsize=12)
    ax.set_ylabel("NRMSE", fontsize=12)
    ax.set_title("Rollout NRMSE vs Lyapunov Time (seed=0)", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(0, 2.0)
    ax.set_xlim(0, ROLLOUT_TAUS)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_support_heatmap(results, path):
    """Plot 3: Heatmap — methods (rows) x monomials (cols).
    Cell = fraction of seeds that found each monomial."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data = np.zeros((len(METHOD_NAMES), len(ALL_MONOMIALS)))
    for i, name in enumerate(METHOD_NAMES):
        if name not in results:
            continue
        for s_key, s_val in results[name]["seeds"].items():
            discovered = set(s_val["support"]["discovered"])
            for j, mono in enumerate(ALL_MONOMIALS):
                if mono in discovered:
                    data[i, j] += 1
        n_seeds = len(results[name]["seeds"])
        if n_seeds > 0:
            data[i] /= n_seeds

    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ALL_MONOMIALS)))
    ax.set_xticklabels(ALL_MONOMIALS, fontsize=10)
    ax.set_yticks(range(len(METHOD_NAMES)))
    ax.set_yticklabels(METHOD_NAMES, fontsize=10)

    # Annotate cells
    for i in range(len(METHOD_NAMES)):
        for j in range(len(ALL_MONOMIALS)):
            val = data[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color=color)

    # Highlight true support columns
    for j, mono in enumerate(ALL_MONOMIALS):
        if mono in TRUE_SUPPORT:
            ax.axvline(j - 0.5, color="green", lw=2, alpha=0.3)
            ax.axvline(j + 0.5, color="green", lw=2, alpha=0.3)

    plt.colorbar(im, label="Fraction of seeds finding monomial")
    ax.set_title("Support Recovery Heatmap", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_kan_hp_heatmap(results, path):
    """Plot 4: KAN hyperparameter sweep — one-step MSE as f(grid_size, spline_order)."""
    kan_methods = ["1L-KAN", "2L-KAN", "2L-KAN-stable"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, name in zip(axes, kan_methods):
        if name not in results or "hp_sweep" not in results[name]:
            ax.set_title(f"{name} (no data)")
            continue

        sweep = results[name]["hp_sweep"]
        data = np.full((len(KAN_GRID_SIZES), len(KAN_SPLINE_ORDERS)), np.nan)
        for gi, gs in enumerate(KAN_GRID_SIZES):
            for si, so in enumerate(KAN_SPLINE_ORDERS):
                key = f"gs{gs}_so{so}"
                if key in sweep:
                    data[gi, si] = sweep[key]

        # Use log scale for better visualization
        data_log = np.log10(np.where(np.isnan(data), 1.0, data))
        im = ax.imshow(data_log, cmap="viridis", aspect="auto")

        ax.set_xticks(range(len(KAN_SPLINE_ORDERS)))
        ax.set_xticklabels(KAN_SPLINE_ORDERS)
        ax.set_yticks(range(len(KAN_GRID_SIZES)))
        ax.set_yticklabels(KAN_GRID_SIZES)
        ax.set_xlabel("Spline Order")
        ax.set_ylabel("Grid Size")
        ax.set_title(f"{name}")

        # Annotate with actual values
        for gi in range(len(KAN_GRID_SIZES)):
            for si in range(len(KAN_SPLINE_ORDERS)):
                if not np.isnan(data[gi, si]):
                    ax.text(si, gi, f"{data[gi, si]:.2e}",
                            ha="center", va="center", fontsize=7,
                            color="white" if data_log[gi, si] < data_log[~np.isnan(data_log)].mean() else "black")

        plt.colorbar(im, ax=ax, label="log10(MSE)")

    fig.suptitle("KAN Hyperparameter Sweep: One-Step MSE (seed=0)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main Benchmark Loop ──────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 70)
    print("  LORENZ BENCHMARK: 8 METHODS x 6 METRICS")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Rollout: {ROLLOUT_TAUS} tau_L = {ROLLOUT_STEPS} steps")
    print(f"  Output: {OUTDIR}")
    print()

    # ── Generate shared data ──────────────────────────────────────────
    print("Generating Lorenz data...")
    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=N_STEPS, dt=DT,
                           burn_in=BURN_IN, device="cpu")
    train_traj = traj[:N_TRAIN].contiguous()
    val_traj = traj[N_TRAIN:].contiguous()
    print(f"  train: {train_traj.shape}, val: {val_traj.shape}")

    results = {}

    # ==================================================================
    # Method 1: DMD
    # ==================================================================
    results["DMD"] = {"seeds": {}, "hp_sweep": {}}
    for seed in SEEDS:
        try:
            r = run_dmd(train_traj, val_traj, seed)
            results["DMD"]["seeds"][str(seed)] = r
        except Exception as e:
            print(f"  DMD seed={seed} FAILED: {e}")
            traceback.print_exc()

    # ==================================================================
    # Method 2: EDMD-poly
    # ==================================================================
    results["EDMD-poly"] = {"seeds": {}, "hp_sweep": {}}
    for seed in SEEDS:
        try:
            r = run_edmd_poly(train_traj, val_traj, seed)
            results["EDMD-poly"]["seeds"][str(seed)] = r
        except Exception as e:
            print(f"  EDMD-poly seed={seed} FAILED: {e}")
            traceback.print_exc()

    # ==================================================================
    # Method 4: MLP-Koopman
    # ==================================================================
    results["MLP-Koopman"] = {"seeds": {}, "hp_sweep": {}}
    for seed in SEEDS:
        try:
            r = run_mlp_koopman(train_traj, val_traj, seed,
                                latent_dim=10, hidden=64, epochs=80)
            results["MLP-Koopman"]["seeds"][str(seed)] = r
        except Exception as e:
            print(f"  MLP-Koopman seed={seed} FAILED: {e}")
            traceback.print_exc()

    # ==================================================================
    # Method 5: 1-Layer KAN-Koopman
    # ==================================================================
    results["1L-KAN"] = {"seeds": {}, "hp_sweep": {}}
    for seed in SEEDS:
        try:
            r = run_1l_kan_koopman(train_traj, val_traj, seed,
                                    latent_dim=10, grid_size=5, spline_order=3,
                                    epochs=80)
            results["1L-KAN"]["seeds"][str(seed)] = r
        except Exception as e:
            print(f"  1L-KAN seed={seed} FAILED: {e}")
            traceback.print_exc()

    # HP sweep for 1L-KAN (seed=0 only)
    for gs in KAN_GRID_SIZES:
        for so in KAN_SPLINE_ORDERS:
            if gs == 5 and so == 3:
                # Already computed in seed loop
                if "0" in results["1L-KAN"]["seeds"]:
                    results["1L-KAN"]["hp_sweep"][f"gs{gs}_so{so}"] = \
                        results["1L-KAN"]["seeds"]["0"]["one_step_mse"]
                continue
            key = f"gs{gs}_so{so}"
            try:
                print(f"\n  1L-KAN HP sweep: grid={gs}, order={so}")
                r = run_1l_kan_koopman(train_traj, val_traj, seed=0,
                                        latent_dim=10, grid_size=gs,
                                        spline_order=so, epochs=80)
                results["1L-KAN"]["hp_sweep"][key] = r["one_step_mse"]
            except Exception as e:
                print(f"  1L-KAN HP {key} FAILED: {e}")
                traceback.print_exc()

    # ==================================================================
    # Method 6: 2-Layer KAN-Koopman (unconstrained)
    # ==================================================================
    results["2L-KAN"] = {"seeds": {}, "hp_sweep": {}}
    saved_model_for_edmd = None
    saved_tr_traj = None
    saved_vl_traj = None

    for seed in SEEDS:
        try:
            r = run_2l_kan_koopman(train_traj, val_traj, seed,
                                    latent_dim=10, mid_size=8,
                                    grid_size=5, spline_order=3, epochs=80)
            # Extract model for EDMD-learned (seed=0 only)
            if "_model" in r:
                saved_model_for_edmd = r.pop("_model")
                saved_tr_traj = r.pop("_tr_traj")
                saved_vl_traj = r.pop("_vl_traj")
            results["2L-KAN"]["seeds"][str(seed)] = r
        except Exception as e:
            print(f"  2L-KAN seed={seed} FAILED: {e}")
            traceback.print_exc()

    # HP sweep for 2L-KAN (seed=0 only)
    for gs in KAN_GRID_SIZES:
        for so in KAN_SPLINE_ORDERS:
            if gs == 5 and so == 3:
                if "0" in results["2L-KAN"]["seeds"]:
                    results["2L-KAN"]["hp_sweep"][f"gs{gs}_so{so}"] = \
                        results["2L-KAN"]["seeds"]["0"]["one_step_mse"]
                continue
            key = f"gs{gs}_so{so}"
            try:
                print(f"\n  2L-KAN HP sweep: grid={gs}, order={so}")
                r = run_2l_kan_koopman(train_traj, val_traj, seed=0,
                                        latent_dim=10, mid_size=8,
                                        grid_size=gs, spline_order=so, epochs=80)
                # Discard model refs from hp sweep
                r.pop("_model", None)
                r.pop("_tr_traj", None)
                r.pop("_vl_traj", None)
                results["2L-KAN"]["hp_sweep"][key] = r["one_step_mse"]
            except Exception as e:
                print(f"  2L-KAN HP {key} FAILED: {e}")
                traceback.print_exc()

    # ==================================================================
    # Method 3: EDMD-learned (uses saved 2L-KAN model from seed=0)
    # ==================================================================
    results["EDMD-learned"] = {"seeds": {}, "hp_sweep": {}}
    if saved_model_for_edmd is not None:
        for seed in SEEDS:
            try:
                r = run_edmd_learned(saved_model_for_edmd, saved_tr_traj,
                                     saved_vl_traj, val_traj, seed)
                results["EDMD-learned"]["seeds"][str(seed)] = r
            except Exception as e:
                print(f"  EDMD-learned seed={seed} FAILED: {e}")
                traceback.print_exc()
        del saved_model_for_edmd, saved_tr_traj, saved_vl_traj
        gc.collect()
    else:
        print("\n  WARNING: No saved 2L-KAN model available for EDMD-learned!")

    # ==================================================================
    # Method 7: 2-Layer KAN-Koopman (stable)
    # ==================================================================
    results["2L-KAN-stable"] = {"seeds": {}, "hp_sweep": {}}
    for seed in SEEDS:
        try:
            r = run_2l_kan_koopman_stable(train_traj, val_traj, seed,
                                            latent_dim=10, mid_size=8,
                                            grid_size=5, spline_order=3,
                                            epochs=150)
            results["2L-KAN-stable"]["seeds"][str(seed)] = r
        except Exception as e:
            print(f"  2L-KAN-stable seed={seed} FAILED: {e}")
            traceback.print_exc()

    # HP sweep for 2L-KAN-stable (seed=0 only)
    for gs in KAN_GRID_SIZES:
        for so in KAN_SPLINE_ORDERS:
            if gs == 5 and so == 3:
                if "0" in results["2L-KAN-stable"]["seeds"]:
                    results["2L-KAN-stable"]["hp_sweep"][f"gs{gs}_so{so}"] = \
                        results["2L-KAN-stable"]["seeds"]["0"]["one_step_mse"]
                continue
            key = f"gs{gs}_so{so}"
            try:
                print(f"\n  2L-KAN-stable HP sweep: grid={gs}, order={so}")
                r = run_2l_kan_koopman_stable(train_traj, val_traj, seed=0,
                                                latent_dim=10, mid_size=8,
                                                grid_size=gs, spline_order=so,
                                                epochs=150)
                results["2L-KAN-stable"]["hp_sweep"][key] = r["one_step_mse"]
            except Exception as e:
                print(f"  2L-KAN-stable HP {key} FAILED: {e}")
                traceback.print_exc()

    # ==================================================================
    # Method 8: KANDy
    # ==================================================================
    results["KANDy"] = {"seeds": {}, "hp_sweep": {}}
    for seed in SEEDS:
        try:
            r = run_kandy(train_traj, val_traj, seed)
            results["KANDy"]["seeds"][str(seed)] = r
        except Exception as e:
            print(f"  KANDy seed={seed} FAILED: {e}")
            traceback.print_exc()

    # ==================================================================
    # Save results JSON (strip numpy arrays for serialization)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)

    results_path = OUTDIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved {results_path}")

    # CSV: one row per (method, seed) with all scalar metrics
    csv_path = OUTDIR / "results.csv"
    csv_fields = [
        "method", "seed", "one_step_mse", "rollout_horizon",
        "precision", "recall", "f1", "sparsity",
        "basis_linear_min", "basis_lorenz_min", "basis_full_d2_min",
        "smallest_sufficient_basis",
        "discovered_support", "wall_time",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for name in METHOD_NAMES:
            if name not in results or not results[name]["seeds"]:
                continue
            for seed_key, sv in results[name]["seeds"].items():
                bs = sv["support"].get("basis_sufficiency", {})
                writer.writerow({
                    "method": name,
                    "seed": seed_key,
                    "one_step_mse": sv["one_step_mse"],
                    "rollout_horizon": sv["rollout_horizon"],
                    "precision": sv["support"]["precision"],
                    "recall": sv["support"]["recall"],
                    "f1": sv["support"]["f1"],
                    "sparsity": sv["sparsity"],
                    "basis_linear_min": bs.get("linear", {}).get("min_energy_ratio", ""),
                    "basis_lorenz_min": bs.get("Lorenz", {}).get("min_energy_ratio", ""),
                    "basis_full_d2_min": bs.get("full-deg2", {}).get("min_energy_ratio", ""),
                    "smallest_sufficient_basis": sv["support"].get("smallest_sufficient", ""),
                    "discovered_support": ";".join(sv["support"]["discovered"]),
                    "wall_time": sv.get("wall_time", ""),
                })
    print(f"  Saved {csv_path}")

    # CSV: KAN hyperparameter sweep
    hp_csv_path = OUTDIR / "kan_hp_sweep.csv"
    hp_fields = ["method", "grid_size", "spline_order", "one_step_mse"]
    with open(hp_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hp_fields)
        writer.writeheader()
        for name in METHOD_NAMES:
            if name not in results or "hp_sweep" not in results[name]:
                continue
            for key, mse_val in results[name]["hp_sweep"].items():
                # key format: "gs{g}_so{s}"
                parts = key.replace("gs", "").replace("so", "").split("_")
                if len(parts) == 2:
                    writer.writerow({
                        "method": name,
                        "grid_size": parts[0],
                        "spline_order": parts[1],
                        "one_step_mse": mse_val,
                    })
    print(f"  Saved {hp_csv_path}")

    # ==================================================================
    # Generate plots
    # ==================================================================
    print("\nGenerating plots...")
    plot_comparison_table(results, OUTDIR / "comparison_table.png")
    plot_rollout_curves(results, OUTDIR / "rollout_curves.png")
    plot_support_heatmap(results, OUTDIR / "support_heatmap.png")
    plot_kan_hp_heatmap(results, OUTDIR / "kan_hp_heatmap.png")

    # ==================================================================
    # Print summary
    # ==================================================================
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"\n  Results: {results_path}")
    print(f"  Plots:   {OUTDIR}")

    # Summary table with basis sufficiency
    print(f"\n{'Method':<18} {'1-Step MSE':>14} {'Horizon':>10} {'F1':>6} "
          f"{'Spars':>6} {'Lin':>6} {'Lorenz':>7} {'Full-d2':>8} {'Smallest':>10}")
    print("-" * 95)
    for name in METHOD_NAMES:
        if name not in results or not results[name]["seeds"]:
            print(f"{name:<18} {'--':>14} {'--':>10}")
            continue
        seed_mses = [results[name]["seeds"][s]["one_step_mse"]
                     for s in results[name]["seeds"]]
        seed_hors = [results[name]["seeds"][s]["rollout_horizon"]
                     for s in results[name]["seeds"]]
        s0 = results[name]["seeds"].get("0", list(results[name]["seeds"].values())[0])
        f1 = s0["support"]["f1"]
        sp = s0["sparsity"]
        bs = s0["support"].get("basis_sufficiency", {})
        lin_e = bs.get("linear", {}).get("min_energy_ratio", 0)
        lor_e = bs.get("Lorenz", {}).get("min_energy_ratio", 0)
        fd2_e = bs.get("full-deg2", {}).get("min_energy_ratio", 0)
        smallest = s0["support"].get("smallest_sufficient", "--")
        print(f"{name:<18} {np.mean(seed_mses):>14.4e} {np.mean(seed_hors):>10.2f} "
              f"{f1:>6.2f} {sp:>6.1f} {lin_e:>6.3f} {lor_e:>7.3f} {fd2_e:>8.3f} "
              f"{str(smallest):>10}")


if __name__ == "__main__":
    main()
