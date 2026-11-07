"""
Ikeda Map Deep Koopman with KAN autoencoder and stable propagator.

The Ikeda map:
    t_n = 0.4 - 6 / (1 + x_n^2 + y_n^2)
    x_{n+1} = 1 + u (x_n cos t_n - y_n sin t_n)
    y_{n+1} = u (x_n sin t_n + y_n cos t_n)

The nonlinearity factors through g(x,y) = x^2 + y^2.  Each output
coordinate is bilinear in (x,y) with coefficients cos(t(g)), sin(t(g))
that are univariate functions of g.  This is *regime (2): co-foliation
with two active channels per latent dim*, not regime (1) channel
sparsity, as in Lorenz.

Phases:
  Training:     KAN encoder [2,6,8] + StablePropagator + pruning + retrain
  Phase 4:      Foliation discovery — test layer-1 inner functions u_j
                against r^2 = x^2 + y^2 (the foliation coordinate)
  Phase 5:      h∘g decomposition via manifold_analysis gradient method
  Phase 6-7:    Prediction horizon + eigenvalue spectrum

Architecture:
  Encoder: KANLinear [2, 6, 8], grid ranges L1=[-3,3], L2=[-5,5]
  Decoder: KANLinear [8, 6, 2], grid ranges L1=[-5,5], L2=[-3,3]
  Propagator: StablePropagator (M = expm(K - L^T L), |lam| <= 1)

Usage:
  python experiments/ikeda_koopman.py              # train + analyze
  python experiments/ikeda_koopman.py --analyze    # analyze existing checkpoints
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from kan import KANLinear
from training import set_seed
from ikeda import generate_ikeda, IkedaPairDataset, ikeda_step
from manifold_analysis import (
    full_decomposition_nd,
    eval_h,
)
from plotting import inspect_kanlinear

# =====================================================================
# Configuration
# =====================================================================

OUTDIR = Path(__file__).resolve().parent.parent / 'results' / 'ikeda'

# Architecture
INPUT_DIM = 2
MID_DIM = 6
LATENT_DIM = 8
GRID_SIZE = 5
SPLINE_ORDER = 3

# Data
N_TRAJ = 200
N_STEPS = 2000
BURN_IN = 500

# Training
EPOCHS = 400
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 2048
GAMMA_PREDX = 1.0    # prediction loss weight
ALPHA_LATENT = 1.0    # latent consistency weight
BETA_RECON = 0.5      # reconstruction loss weight
L1_SPLINE_WEIGHT = 0.001  # spline regularization weight
GRAD_CLIP = 1.0

# Pruning + retraining
PRUNE_THRESHOLD = 0.03
RETRAIN_EPOCHS = 100
RETRAIN_LR = 5e-4

# Prediction horizon
HORIZON_N_ICS = 50
HORIZON_MAX_STEPS = 200
HORIZON_MSE_THRESHOLD = 0.1

# Level-set analysis
# Degree 5 + alpha 1e-5 gives 21/21 active terms (fully dense) —
# the gradient formula blows up on such polynomials. Use degree 3
# with moderate alpha so g is sparser and ∇g is well-conditioned.
LASSO_MAX_DEGREE = 3
LASSO_ALPHA = 1e-3

# Seeds
SEEDS = [0, 1, 2, 3, 4]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =====================================================================
# StablePropagator: M = expm(K - L^T L), |lam| <= 1
# =====================================================================

class StablePropagator(nn.Module):
    """M = expm(A) where A = K - L^T L.

    K = -K^T (skew-symmetric, rotation)
    L^T L >= 0 (PSD, damping)

    => Re(eigenvalues of A) <= 0 => |eigenvalues of M| <= 1.
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
# Encoder / Decoder (KANLinear, no angular encoding)
# =====================================================================

class IkedaKANEncoder(nn.Module):
    """Two-layer KAN encoder: [2, 6, 8]."""
    def __init__(self):
        super().__init__()
        self.kan1 = KANLinear(INPUT_DIM, MID_DIM, grid_size=GRID_SIZE,
                              spline_order=SPLINE_ORDER, grid_range=[-3, 3])
        self.kan2 = KANLinear(MID_DIM, LATENT_DIM, grid_size=GRID_SIZE,
                              spline_order=SPLINE_ORDER, grid_range=[-5, 5])

    def forward(self, x):
        return self.kan2(self.kan1(x))


class IkedaKANDecoder(nn.Module):
    """Two-layer KAN decoder: [8, 6, 2]."""
    def __init__(self):
        super().__init__()
        self.kan1 = KANLinear(LATENT_DIM, MID_DIM, grid_size=GRID_SIZE,
                              spline_order=SPLINE_ORDER, grid_range=[-5, 5])
        self.kan2 = KANLinear(MID_DIM, INPUT_DIM, grid_size=GRID_SIZE,
                              spline_order=SPLINE_ORDER, grid_range=[-3, 3])

    def forward(self, z):
        return self.kan2(self.kan1(z))


class IkedaStableKoopman(nn.Module):
    """Full model: KAN encoder + StablePropagator + KAN decoder."""
    def __init__(self):
        super().__init__()
        self.encoder = IkedaKANEncoder()
        self.decoder = IkedaKANDecoder()
        self.prop = StablePropagator(LATENT_DIM)
        self.latent_dim = LATENT_DIM

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
# L1 spline regularization
# =====================================================================

def spline_l1_reg(model):
    """Sum of mean absolute spline weights across all KANLinear layers."""
    reg = 0.0
    for module in model.modules():
        if isinstance(module, KANLinear):
            reg = reg + module.spline_weight.abs().mean()
    return reg


# =====================================================================
# Training loop
# =====================================================================

def train_one_seed(seed, traj, epochs, lr, device):
    """Train model for one seed. Returns (model, val_loss, history)."""
    set_seed(seed)

    n_train = int(0.8 * traj.shape[0])
    train_ds = IkedaPairDataset(traj[:n_train])
    val_ds = IkedaPairDataset(traj[n_train:])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = IkedaStableKoopman().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model [2,{MID_DIM},{LATENT_DIM}]: {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}

    for ep in range(1, epochs + 1):
        model.train()
        ep_losses = []
        for xy_n, xy_np1 in train_dl:
            xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
            out = model.forward_one_step(xy_n)
            z_true = model.encoder(xy_np1)

            loss_pred = mse(out['xy_np1_pred'], xy_np1)
            loss_latent = mse(out['z_np1_pred'], z_true)
            loss_recon = mse(out['xy_n_rec'], xy_n)
            loss_reg = spline_l1_reg(model)

            loss = (GAMMA_PREDX * loss_pred
                    + ALPHA_LATENT * loss_latent
                    + BETA_RECON * loss_recon
                    + L1_SPLINE_WEIGHT * loss_reg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ep_losses.append(loss.item())
        scheduler.step()

        # Validation
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            model.eval()
            vloss = []
            with torch.no_grad():
                for xy_n, xy_np1 in val_dl:
                    xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
                    out = model.forward_one_step(xy_n)
                    z_true = model.encoder(xy_np1)
                    l = (GAMMA_PREDX * mse(out['xy_np1_pred'], xy_np1)
                         + ALPHA_LATENT * mse(out['z_np1_pred'], z_true)
                         + BETA_RECON * mse(out['xy_n_rec'], xy_n))
                    vloss.append(l.item())
            val_mean = float(np.mean(vloss))

            M = model.koopman_matrix().cpu().numpy()
            eigs = np.linalg.eigvals(M)
            mags = np.abs(eigs)

            print(f"  Ep {ep:3d} | train {np.mean(ep_losses):.5f} "
                  f"| val {val_mean:.5f} "
                  f"| |lam| [{mags.min():.4f}, {mags.max():.4f}]", flush=True)

        history['train_loss'].append(float(np.mean(ep_losses)))
        # Compute val loss every epoch for the history (lightweight)
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            history['val_loss'].append(val_mean)
        else:
            history['val_loss'].append(history['val_loss'][-1] if history['val_loss'] else float('nan'))

    model = model.cpu().eval()
    final_val = history['val_loss'][-1]
    return model, final_val, history


# =====================================================================
# Pruning
# =====================================================================

def compute_edge_importance(layer):
    """Compute importance score for each edge in a KANLinear layer.

    importance[out, in] = mean(|spline_weight[out, in, :]|) + |spline_scaler[out, in]|
    """
    with torch.no_grad():
        imp = layer.spline_weight.abs().mean(dim=-1)  # (out, in)
        if hasattr(layer, 'spline_scaler') and layer.spline_scaler is not None:
            imp = imp + layer.spline_scaler.abs()
    return imp


def prune_layer(layer, threshold):
    """Zero out edges with importance < threshold. Returns number of pruned edges."""
    imp = compute_edge_importance(layer)
    mask = imp < threshold  # (out, in) boolean
    n_pruned = int(mask.sum().item())
    with torch.no_grad():
        for out_i in range(layer.out_features):
            for in_i in range(layer.in_features):
                if mask[out_i, in_i]:
                    layer.spline_weight[out_i, in_i, :] = 0.0
                    if hasattr(layer, 'spline_scaler') and layer.spline_scaler is not None:
                        layer.spline_scaler[out_i, in_i] = 0.0
                    layer.base_weight[out_i, in_i] = 0.0
    return n_pruned


def prune_model(model, threshold=PRUNE_THRESHOLD):
    """Prune all KANLinear layers in the model. Returns total edges pruned."""
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            n = prune_layer(module, threshold)
            total_edges = module.out_features * module.in_features
            print(f"    Pruned {name}: {n}/{total_edges} edges removed")
            total += n
    return total


def retrain_pruned(model, traj, epochs, lr, device):
    """Retrain a pruned model for additional epochs at lower learning rate."""
    n_train = int(0.8 * traj.shape[0])
    train_ds = IkedaPairDataset(traj[:n_train])
    val_ds = IkedaPairDataset(traj[n_train:])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        ep_losses = []
        for xy_n, xy_np1 in train_dl:
            xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
            out = model.forward_one_step(xy_n)
            z_true = model.encoder(xy_np1)

            loss_pred = mse(out['xy_np1_pred'], xy_np1)
            loss_latent = mse(out['z_np1_pred'], z_true)
            loss_recon = mse(out['xy_n_rec'], xy_n)
            loss_reg = spline_l1_reg(model)

            loss = (GAMMA_PREDX * loss_pred
                    + ALPHA_LATENT * loss_latent
                    + BETA_RECON * loss_recon
                    + L1_SPLINE_WEIGHT * loss_reg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ep_losses.append(loss.item())
        scheduler.step()

        if ep == 1 or ep % 25 == 0 or ep == epochs:
            model.eval()
            vloss = []
            with torch.no_grad():
                for xy_n, xy_np1 in val_dl:
                    xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
                    out = model.forward_one_step(xy_n)
                    z_true = model.encoder(xy_np1)
                    l = (GAMMA_PREDX * mse(out['xy_np1_pred'], xy_np1)
                         + ALPHA_LATENT * mse(out['z_np1_pred'], z_true)
                         + BETA_RECON * mse(out['xy_n_rec'], xy_n))
                    vloss.append(l.item())
            print(f"    Retrain ep {ep:3d} | train {np.mean(ep_losses):.5f} "
                  f"| val {np.mean(vloss):.5f}", flush=True)

    return model.cpu().eval()


# =====================================================================
# Prediction horizon
# =====================================================================

def compute_prediction_horizon(model, val_traj, n_ics, max_steps, threshold):
    """Roll out model and compare to true Ikeda iteration.

    Returns prediction horizon (first step where mean MSE > threshold)
    for each of n_ics initial conditions from val_traj.
    """
    model.eval()
    # Pick n_ics starting points from the validation trajectories
    n_val = val_traj.shape[0]
    n_steps_traj = val_traj.shape[1]
    rng = np.random.RandomState(42)
    ic_traj_ids = rng.randint(0, n_val, size=n_ics)
    ic_step_ids = rng.randint(0, n_steps_traj, size=n_ics)

    horizons = []
    mse_per_step = np.zeros(max_steps)
    count_per_step = np.zeros(max_steps)

    with torch.no_grad():
        for ic_idx in range(n_ics):
            # True initial condition
            x_true = val_traj[ic_traj_ids[ic_idx], ic_step_ids[ic_idx]].numpy()
            x_model = torch.tensor(x_true, dtype=torch.float32).unsqueeze(0)

            horizon_found = False
            for step in range(max_steps):
                # True next step
                x_new, y_new = ikeda_step(float(x_true[0]), float(x_true[1]))
                x_true = np.array([x_new, y_new], dtype=np.float32)

                # Model next step
                out = model.forward_one_step(x_model)
                x_model = out['xy_np1_pred']
                x_model_np = x_model.squeeze().numpy()

                err = float(np.mean((x_model_np - x_true) ** 2))
                mse_per_step[step] += err
                count_per_step[step] += 1

                if not horizon_found and err > threshold:
                    horizons.append(step)
                    horizon_found = True

            if not horizon_found:
                horizons.append(max_steps)

    mean_mse_per_step = mse_per_step / np.maximum(count_per_step, 1)
    mean_horizon = float(np.mean(horizons))
    return mean_horizon, horizons, mean_mse_per_step


# =====================================================================
# Model adapter for manifold_analysis compatibility
# =====================================================================

class _AEAdapter:
    """Makes IkedaStableKoopman look like DeepKoopmanCT for manifold_analysis."""
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder


def _wrap_for_manifold(model):
    """Wrap IkedaStableKoopman so manifold_analysis.compute_network_gradient_nd works.

    manifold_analysis expects model.ae.encoder with .kan1 and .kan2.
    """
    return type('ModelWrap', (), {'ae': _AEAdapter(model.encoder, model.decoder)})()


# =====================================================================
# Phase 4: Foliation analysis
# =====================================================================

def extract_layer1_activations(model, pts):
    """Extract layer-1 inner functions u_j(x,y) = Σ_i ψ_{j,i}(x_i).

    Returns:
        u: (N, mid_dim) numpy array of layer-1 outputs
        edge_info: dict from inspect_kanlinear with per-edge activations
    """
    model.eval()
    with torch.no_grad():
        u = model.encoder.kan1(pts)
    edge_info = inspect_kanlinear(model.encoder.kan1, pts)
    return u.numpy(), edge_info


def foliation_alignment_score(u_j_values, r2_values, degree=4):
    """Test whether u_j is constant on level sets of r^2.

    Uses a degree-4 polynomial of r^2 as a flexible 1D smoother.

    Returns:
        R2_radial: fraction of variance of u_j explained by r^2 alone
        fit: fitted values
    """
    poly = PolynomialFeatures(degree=degree)
    r2_feat = poly.fit_transform(r2_values.reshape(-1, 1))
    reg = LinearRegression().fit(r2_feat, u_j_values)
    r2_score = reg.score(r2_feat, u_j_values)
    fit = reg.predict(r2_feat)
    return float(r2_score), fit


def classify_edge_function(x_in, y_edge):
    """Classify a single KAN edge function as linear vs quadratic.

    Returns dict with R2_lin, R2_quad, leading quadratic coefficient.
    """
    coeffs_lin = np.polyfit(x_in, y_edge, 1)
    fit_lin = np.polyval(coeffs_lin, x_in)
    ss_tot = np.sum((y_edge - y_edge.mean())**2)
    r2_lin = 1.0 - np.sum((y_edge - fit_lin)**2) / max(ss_tot, 1e-12)

    coeffs_quad = np.polyfit(x_in, y_edge, 2)
    fit_quad = np.polyval(coeffs_quad, x_in)
    r2_quad = 1.0 - np.sum((y_edge - fit_quad)**2) / max(ss_tot, 1e-12)

    return {
        'r2_lin': float(r2_lin),
        'r2_quad': float(r2_quad),
        'quad_coeff': float(coeffs_quad[0]),
        'is_quadratic': r2_quad > 0.9 and r2_lin < 0.8 and abs(coeffs_quad[0]) > 0.01,
    }


def channel_activity_count(model, threshold=0.03):
    """Count active outer splines |A_k| for each latent dim k.

    |A_k| = #{j : ||Φ_{k,j}||_inf > threshold}

    Returns:
        activity: list of int, one per latent dim
        importance: (latent_dim, mid_dim) array of edge importances
    """
    kan2 = model.encoder.kan2
    imp = kan2.spline_weight.abs().mean(dim=-1)  # (latent_dim, mid_dim)
    if hasattr(kan2, 'spline_scaler') and kan2.spline_scaler is not None:
        imp = imp + kan2.spline_scaler.abs()
    imp_np = imp.detach().numpy()

    activity = [int(np.sum(imp_np[k] > threshold)) for k in range(imp_np.shape[0])]
    return activity, imp_np


def run_foliation_analysis(model, pts, outdir):
    """Phase 4: test whether layer-1 inner functions factor through r^2.

    This is the decisive diagnostic. The proposition predicts:
      - At least 2 channels with R2_radial > 0.95 (carrying cos(t(r²)), sin(t(r²)))
      - |A_k| ≈ 2 (regime 2, not regime 1)

    Returns dict with all foliation metrics.
    """
    x_np = pts[:, 0].numpy()
    y_np = pts[:, 1].numpy()
    r2 = x_np**2 + y_np**2

    u_np, edge_info = extract_layer1_activations(model, pts)
    mid_dim = u_np.shape[1]

    # 4a: classify each layer-1 edge function
    edge_classifications = []
    for j in range(mid_dim):
        for i in range(2):
            x_in = edge_info['x_in'][:, i].numpy()
            y_edge = edge_info['postacts'][:, j, i].numpy()
            cls = classify_edge_function(x_in, y_edge)
            cls['node'] = j
            cls['input'] = i
            cls['var'] = 'x' if i == 0 else 'y'
            edge_classifications.append(cls)

    # 4b: foliation alignment for each layer-1 node
    foliation_scores = []
    for j in range(mid_dim):
        r2_radial, fit = foliation_alignment_score(u_np[:, j], r2)
        foliation_scores.append({
            'node': j,
            'R2_radial': r2_radial,
            'std': float(np.std(u_np[:, j])),
        })

    # 4c: channel activity count
    activity, layer2_imp = channel_activity_count(model)

    # Print results
    print("\n  Phase 4a: Layer-1 edge functions")
    for ec in edge_classifications:
        tag = ' *** QUADRATIC' if ec['is_quadratic'] else ''
        print(f"    ψ[{ec['node']},{ec['input']}]({ec['var']}): "
              f"R2_lin={ec['r2_lin']:.3f}, R2_quad={ec['r2_quad']:.3f}, "
              f"a={ec['quad_coeff']:.4f}{tag}")

    print("\n  Phase 4b: Foliation alignment (u_j vs r²)")
    for fs in foliation_scores:
        tag = ' *** ALIGNED' if fs['R2_radial'] > 0.8 else ''
        print(f"    u[{fs['node']}]: R2_radial = {fs['R2_radial']:.4f}, "
              f"std = {fs['std']:.4f}{tag}")

    n_aligned = sum(1 for fs in foliation_scores if fs['R2_radial'] > 0.8)
    print(f"    Channels with R2_radial > 0.8: {n_aligned}/{mid_dim}")

    print("\n  Phase 4c: Channel activity |A_k| (tau=0.03)")
    for k, ak in enumerate(activity):
        active_js = [j for j in range(mid_dim) if layer2_imp[k, j] > 0.03]
        print(f"    z[{k}]: |A_k| = {ak}, channels = {active_js}")

    activity_hist = np.bincount(activity, minlength=mid_dim + 1)
    mode_ak = int(np.argmax(activity_hist))
    print(f"    |A_k| histogram: {activity_hist.tolist()}")
    print(f"    Mode of |A_k|: {mode_ak}")

    # 4d: Visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, mid_dim, hspace=0.35, wspace=0.3)

    # Top row: attractor colored by u_j
    for j in range(mid_dim):
        ax = fig.add_subplot(gs[0, j])
        sc = ax.scatter(x_np, y_np, s=0.1, alpha=0.3, c=u_np[:, j],
                        cmap='coolwarm', rasterized=True)
        ax.set_title(f'$u_{j}$\nR²={foliation_scores[j]["R2_radial"]:.3f}',
                     fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel('$y$', fontsize=8)

    # Middle row: u_j vs r² (the money plot — scatter should collapse to a curve)
    for j in range(mid_dim):
        ax = fig.add_subplot(gs[1, j])
        ax.scatter(r2, u_np[:, j], s=0.1, alpha=0.15, c='steelblue', rasterized=True)
        # Overlay the polynomial fit
        r2_sorted = np.sort(r2)
        _, fit_vals = foliation_alignment_score(u_np[:, j], r2)
        order = np.argsort(r2)
        ax.plot(r2[order], fit_vals[order], 'r-', lw=1.5, alpha=0.8)
        ax.set_xlabel('$r^2$', fontsize=7)
        ax.set_title(f'$u_{j}$ vs $r^2$', fontsize=8)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel('$u_j$', fontsize=8)

    # Bottom row: contour plot of r² overlaid on attractor
    # Show which u_j's have quadratic edges
    for j in range(mid_dim):
        ax = fig.add_subplot(gs[2, j])
        # Color by u_j with r² contours overlaid
        sc = ax.scatter(x_np, y_np, s=0.1, alpha=0.3, c=u_np[:, j],
                        cmap='coolwarm', rasterized=True)
        # Add r² contours
        xg = np.linspace(x_np.min(), x_np.max(), 100)
        yg = np.linspace(y_np.min(), y_np.max(), 100)
        Xg, Yg = np.meshgrid(xg, yg)
        R2g = Xg**2 + Yg**2
        ax.contour(Xg, Yg, R2g, levels=6, colors='black', linewidths=0.5, alpha=0.5)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel('$y$', fontsize=8)
        ax.set_xlabel('$x$', fontsize=7)

        # Tag quadratic edges
        x_edges = [ec for ec in edge_classifications
                   if ec['node'] == j and ec['is_quadratic']]
        if x_edges:
            vars_quad = [ec['var'] for ec in x_edges]
            ax.set_title(f'$u_{j}$ + $r^2$ contours\nquad: {",".join(vars_quad)}',
                         fontsize=7, color='red')
        else:
            ax.set_title(f'$u_{j}$ + $r^2$ contours', fontsize=7)

    fig.suptitle('Phase 4: Foliation Discovery\n'
                 'Middle row: u_j vs r² — collapse to curve = foliation found',
                 fontsize=12)
    fig.savefig(outdir / 'foliation_discovery.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / 'foliation_discovery.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {outdir / 'foliation_discovery.png'}")

    return {
        'edge_classifications': edge_classifications,
        'foliation_scores': foliation_scores,
        'channel_activity': activity,
        'layer2_importance': layer2_imp.tolist(),
        'n_aligned_r2': n_aligned,
        'mode_Ak': mode_ak,
    }


# =====================================================================
# Phase 5a: Layer-1 inner function decomposition
# =====================================================================

def run_layer1_decomposition(model, pts, outdir, max_degree=3, alpha=1e-3):
    """h∘g decomposition on layer-1 outputs u_j.

    For Ikeda, u_j = ψ_{j,0}(x) + ψ_{j,1}(y) should approximate
    a_j·(x²+y²) + linear.  The Lasso inner function should show
    x² and y² coefficients with similar magnitude and same sign
    (the r² = x²+y² signature).

    This is distinct from the z_k decomposition: z_k is bilinear
    (cos(t(r²))·x, sin(t(r²))·y mixing) so its Lasso is dense.
    But u_j is additive, so its quadratic part IS sparse.
    """
    from manifold_analysis import (
        recover_inner_function_nd, eval_polynomial_nd,
        grad_polynomial_nd, recover_outer_function_from_raw,
        eval_h as _eval_h, analyze_ideal,
    )

    pts_np = pts.numpy() if isinstance(pts, torch.Tensor) else pts
    pts_t = torch.tensor(pts_np, dtype=torch.float32)
    mid_dim = MID_DIM

    results = []
    for j in range(mid_dim):
        # Compute u_j and ∇u_j via autograd
        xi = pts_t.detach().clone().requires_grad_(True)
        u = model.encoder.kan1(xi)
        uj = u[:, j]
        uj.sum().backward()
        grad_uj = xi.grad.detach().numpy()
        uj_np = uj.detach().numpy()

        # Inner function via Lasso
        inner = recover_inner_function_nd(pts_np, ['x', 'y'], uj_np,
                                           max_degree=max_degree, alpha=alpha)

        # Outer function via gradient formula
        g_vals = eval_polynomial_nd(pts_np, inner['coeffs'], inner['powers'])
        grad_g = grad_polynomial_nd(pts_np, inner['coeffs'], inner['powers'])

        dot = np.sum(grad_uj * grad_g, axis=1)
        norm_sq = np.sum(grad_g**2, axis=1)
        outer = recover_outer_function_from_raw(dot, norm_sq, g_vals, uj_np,
                                                 poly_degree=4)

        f_hat = _eval_h(g_vals, outer)
        ss_res = np.sum((uj_np - f_hat)**2)
        ss_tot = np.sum((uj_np - uj_np.mean())**2)
        r2_hg = 1.0 - ss_res / max(ss_tot, 1e-12)

        ideal = analyze_ideal(inner)

        # Extract x² and y² coefficients for r² signature check
        x2_coeff = y2_coeff = 0.0
        for nm, cf in zip(inner['active_names'], inner['active_coeffs']):
            if nm == 'x^2':
                x2_coeff = float(cf)
            elif nm == 'y^2':
                y2_coeff = float(cf)

        r2_ratio = min(abs(x2_coeff), abs(y2_coeff)) / max(abs(x2_coeff), abs(y2_coeff), 1e-12)
        r2_same_sign = (x2_coeff * y2_coeff > 0) and abs(x2_coeff) > 0.01 and abs(y2_coeff) > 0.01
        r2_signature = r2_same_sign and r2_ratio > 0.5

        result = {
            'node': j,
            'r2_g': float(inner['r2']),
            'r2_hg': float(r2_hg),
            'n_active': len(inner['active']),
            'active_names': list(inner['active_names']),
            'active_coeffs': [float(c) for c in inner['active_coeffs']],
            'h_affine': bool(outer.get('used_affine', False)),
            'x2_coeff': x2_coeff,
            'y2_coeff': y2_coeff,
            'r2_ratio': float(r2_ratio),
            'r2_same_sign': r2_same_sign,
            'r2_signature': r2_signature,
            'ideal_type': ideal['type'],
        }
        results.append(result)

        tag = ' *** r² FOUND' if r2_signature else ''
        print(f"    u[{j}]: R²(g)={inner['r2']:.4f}, R²(h∘g)={r2_hg:.4f}, "
              f"x²={x2_coeff:+.4f}, y²={y2_coeff:+.4f}, "
              f"ratio={r2_ratio:.3f}, active={inner['active_names']}{tag}")

    # Summary
    n_r2 = sum(1 for r in results if r['r2_signature'])
    print(f"\n  Layer-1 nodes with r² = x²+y² signature: {n_r2}/{mid_dim}")

    return results


# =====================================================================
# Phase 5b: h∘g decomposition on latent dims via manifold_analysis
# =====================================================================

def run_decomposition(model, traj, outdir, max_degree=3, lasso_alpha=1e-4,
                      h_degree=4, n_subsample=8000):
    """Run full h∘g decomposition on each latent dim z_k.

    Uses manifold_analysis.full_decomposition_nd which:
      1. Lasso recovers sparse inner polynomial g(x,y)
      2. Gradient formula recovers outer function h: h'(g) = ∇f·∇g / ||∇g||²
      3. Ideal analysis identifies the Koopman observables

    This is what the other agent's lasso_analysis_latent should have been.
    """
    # Flatten and subsample
    all_pts = traj.reshape(-1, 2)
    if len(all_pts) > n_subsample:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(all_pts), n_subsample, replace=False)
        all_pts = all_pts[idx]

    x_data = torch.tensor(all_pts, dtype=torch.float32)
    model_wrap = _wrap_for_manifold(model)

    results = []
    for k in range(LATENT_DIM):
        print(f"    z[{k}]...", end=' ', flush=True)
        try:
            decomp = full_decomposition_nd(
                model_wrap, x_data, k, var_names=['x', 'y'],
                target_type='latent', max_degree=max_degree,
                lasso_alpha=lasso_alpha, h_degree=h_degree,
            )
            print(f"R²(g)={decomp['g_recon_r2']:.4f}, "
                  f"R²(h∘g)={decomp['recon_r2']:.4f}, "
                  f"active: {decomp['inner']['active_names']}")
            results.append(decomp)
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(None)

    # Summary table
    print(f"\n  {'Dim':>4s} | {'R²(g)':>8s} | {'R²(h∘g)':>9s} | {'#terms':>6s} | "
          f"{'h affine?':>9s} | Active terms")
    print(f"  {'-'*4} | {'-'*8} | {'-'*9} | {'-'*6} | {'-'*9} | {'-'*40}")
    for k, d in enumerate(results):
        if d is None:
            print(f"  z[{k}] | {'FAIL':>8s} |")
            continue
        inner = d['inner']
        outer = d['outer']
        active_str = ", ".join(
            f"{n}({c:+.3f})" for n, c in
            zip(inner['active_names'], inner['active_coeffs'])
            if n != '1'
        )
        if len(active_str) > 50:
            active_str = active_str[:47] + '...'
        print(f"  z[{k}] | {d['g_recon_r2']:8.4f} | {d['recon_r2']:9.4f} | "
              f"{len(inner['active']):6d} | "
              f"{'yes' if outer.get('used_affine', False) else 'no':>9s} | "
              f"{active_str}")

    # Decomposition plots for top dims by R²(h∘g)
    valid = [(k, d) for k, d in enumerate(results) if d is not None]
    valid.sort(key=lambda kd: kd[1]['recon_r2'], reverse=True)
    n_plot = min(4, len(valid))

    if n_plot > 0:
        fig, axes = plt.subplots(n_plot, 4, figsize=(16, 3.5 * n_plot))
        if n_plot == 1:
            axes = axes[None, :]

        for row, (k, d) in enumerate(valid[:n_plot]):
            x_np = d['x_np']
            f_vals = d['f_vals']
            g_vals = d['g_vals']
            f_hat = d['f_hat']
            outer = d['outer']
            inner = d['inner']

            # Panel (a): attractor colored by z_k
            ax = axes[row, 0]
            sc = ax.scatter(x_np[:, 0], x_np[:, 1], s=0.3, alpha=0.3,
                            c=f_vals, cmap='coolwarm', rasterized=True)
            ax.set_ylabel(f'z[{k}]', fontsize=10, fontweight='bold')
            ax.set_title('$f = z_k(x,y)$', fontsize=9)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)

            # Panel (b): f vs g scatter — clip to data range
            ax = axes[row, 1]
            ax.scatter(g_vals, f_vals, s=0.3, alpha=0.15, c='steelblue',
                       rasterized=True)
            # Zoom to where the data actually lives (1st–99th percentile)
            g_lo, g_hi = np.percentile(g_vals, [1, 99])
            g_pad = 0.05 * (g_hi - g_lo)
            ax.set_xlim(g_lo - g_pad, g_hi + g_pad)
            ax.set_xlabel('$g(x,y)$', fontsize=8)
            ax.set_ylabel('$f$', fontsize=8)
            ax.set_title(f'$f$ vs $g$  (R²(g)={d["g_recon_r2"]:.3f})\n'
                         f'{len(inner["active"])} active terms', fontsize=9)
            ax.tick_params(labelsize=7)

            # Panel (c): outer function h' — clip to reasonable range
            ax = axes[row, 2]
            h_type = 'affine' if outer.get('used_affine', False) else 'nonlinear'
            if 'g_samples' in outer and len(outer['g_samples']) > 0:
                hp = outer['h_prime_samples']
                gs = outer['g_samples']
                # Clip h' to IQR-based range so outliers don't destroy the plot
                hp_med = np.median(hp)
                hp_iqr = np.percentile(hp, 75) - np.percentile(hp, 25)
                if hp_iqr < 1e-6:
                    hp_iqr = np.std(hp)
                hp_lo = hp_med - 5 * max(hp_iqr, 0.1)
                hp_hi = hp_med + 5 * max(hp_iqr, 0.1)
                ax.scatter(gs, hp, s=0.3, alpha=0.1, c='gray',
                           rasterized=True, label="$h'$ samples")
                if 'g_plot' in outer:
                    hp_fit = outer['h_prime_fit']
                    ax.plot(outer['g_plot'], hp_fit, 'r-', lw=2, label="$h'$ fit")
                ax.set_ylim(hp_lo, hp_hi)
                ax.legend(fontsize=6)
            ax.set_xlabel('$g$', fontsize=8)
            ax.set_ylabel("$h'(g)$", fontsize=8)
            ax.set_title(f"Outer $h'(g)$ [{h_type}]", fontsize=9)
            ax.tick_params(labelsize=7)

            # Panel (d): reconstruction f̂ = h(g) vs true f
            ax = axes[row, 3]
            # Clip f_hat to the range of f_vals to avoid outlier-driven axes
            f_hat_clip = np.clip(f_hat, f_vals.min() - 0.5, f_vals.max() + 0.5)
            ax.scatter(f_vals, f_hat_clip, s=0.3, alpha=0.15, c='steelblue',
                       rasterized=True)
            lims = [f_vals.min(), f_vals.max()]
            ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
            ax.set_xlim(lims[0] - 0.1, lims[1] + 0.1)
            ax.set_ylim(lims[0] - 0.1, lims[1] + 0.1)
            ax.set_xlabel('True $f$', fontsize=8)
            ax.set_ylabel('$\\hat{f} = h(g)$', fontsize=8)
            ax.set_title(f'R²(h∘g)={d["recon_r2"]:.4f}', fontsize=9)
            ax.tick_params(labelsize=7)
            # Annotate active terms
            terms = [n for n in inner['active_names'] if n != '1']
            if terms:
                ax.text(0.02, 0.98, 'g: ' + ', '.join(terms[:5]),
                        transform=ax.transAxes, fontsize=6, va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('Phase 5: h∘g Decomposition via Gradient Method\n'
                     "(top latent dims by R²(h∘g))", fontsize=12)
        fig.tight_layout()
        fig.savefig(outdir / 'decomposition.png', dpi=200, bbox_inches='tight')
        fig.savefig(outdir / 'decomposition.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Saved {outdir / 'decomposition.png'}")

    # Serialize results
    serializable = []
    for k, d in enumerate(results):
        if d is None:
            serializable.append({'dim': k, 'status': 'failed'})
            continue
        serializable.append({
            'dim': k,
            'r2_g': float(d['g_recon_r2']),
            'r2_hg': float(d['recon_r2']),
            'n_active': len(d['inner']['active']),
            'active_names': list(d['inner']['active_names']),
            'active_coeffs': [float(c) for c in d['inner']['active_coeffs']],
            'h_affine': bool(d['outer'].get('used_affine', False)),
            'ideal_type': d['ideal']['type'] if d.get('ideal') else 'unknown',
        })

    return results, serializable


# =====================================================================
# Eigenvalue plot
# =====================================================================

def plot_eigenvalues(all_eigs, outdir, seeds):
    """Plot Koopman spectra for all seeds on one figure."""
    n_seeds = len(seeds)
    fig, axes = plt.subplots(1, n_seeds, figsize=(4 * n_seeds, 4), squeeze=False)
    axes = axes[0]

    theta = np.linspace(0, 2 * np.pi, 200)

    for i, (s, eigs) in enumerate(zip(seeds, all_eigs)):
        ax = axes[i]
        mags = np.abs(eigs)
        ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.3)
        sc = ax.scatter(eigs.real, eigs.imag, s=50, c=mags, cmap='RdYlGn',
                        vmin=0, vmax=1, edgecolors='black', linewidths=0.5, zorder=3)
        ax.set_xlabel(r'Re($\lambda$)')
        if i == 0:
            ax.set_ylabel(r'Im($\lambda$)')
        ax.set_title(f'Seed {s}\n'
                     f'|lam| [{mags.min():.3f}, {mags.max():.3f}]', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)

    fig.suptitle('Ikeda Map: Koopman Spectrum (Stable Propagator)', fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / 'eigenvalues.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / 'eigenvalues.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outdir / 'eigenvalues.png'}")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ikeda Deep Koopman KAN')
    parser.add_argument('--analyze', action='store_true',
                        help='Skip training, analyze existing checkpoints')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single seed to analyze (default: best by val loss)')
    parser.add_argument('--device', default=DEVICE)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Ikeda Map: Deep Koopman with KAN + Stable Propagator")
    print(f"  Architecture: [{INPUT_DIM}, {MID_DIM}, {LATENT_DIM}]")
    print(f"  Device: {args.device}")
    print(f"  Mode: {'analyze only' if args.analyze else 'train + analyze'}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Generate data (shared across seeds for fair comparison)
    # ------------------------------------------------------------------
    print("\nGenerating Ikeda trajectories...")
    traj = generate_ikeda(n_traj=N_TRAJ, n_steps=N_STEPS, burn_in=BURN_IN, seed=42)
    print(f"  Shape: {traj.shape}")
    print(f"  x range: [{traj[:,:,0].min():.3f}, {traj[:,:,0].max():.3f}]")
    print(f"  y range: [{traj[:,:,1].min():.3f}, {traj[:,:,1].max():.3f}]")

    traj_torch = torch.from_numpy(traj)
    n_train = int(0.8 * N_TRAJ)
    val_traj = traj_torch[n_train:]

    # ------------------------------------------------------------------
    # Training (skip if --analyze and checkpoints exist)
    # ------------------------------------------------------------------
    if not args.analyze:
        all_eigs = []
        for seed in SEEDS:
            print(f"\n{'='*70}")
            print(f"SEED {seed}")
            print(f"{'='*70}")

            print(f"\n--- Training (seed={seed}) ---")
            model, val_loss, history = train_one_seed(
                seed, traj_torch, epochs=EPOCHS, lr=LR, device=args.device)
            print(f"  Final val loss: {val_loss:.5f}")

            ckpt_path = OUTDIR / f'ikeda_seed_{seed}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_dim': INPUT_DIM, 'mid_dim': MID_DIM, 'latent_dim': LATENT_DIM,
                    'epochs': EPOCHS, 'lr': LR, 'seed': seed,
                    'gamma': GAMMA_PREDX, 'alpha': ALPHA_LATENT, 'beta': BETA_RECON,
                    'l1_weight': L1_SPLINE_WEIGHT,
                },
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

            # Pruning
            print(f"\n--- Pruning (threshold={PRUNE_THRESHOLD}) ---")
            n_pruned = prune_model(model, threshold=PRUNE_THRESHOLD)
            print(f"  Total edges pruned: {n_pruned}")

            # Retrain
            print(f"\n--- Retraining ({RETRAIN_EPOCHS} epochs, lr={RETRAIN_LR}) ---")
            model = retrain_pruned(model, traj_torch, epochs=RETRAIN_EPOCHS,
                                   lr=RETRAIN_LR, device=args.device)

            pruned_path = OUTDIR / f'ikeda_pruned_seed_{seed}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_dim': INPUT_DIM, 'mid_dim': MID_DIM, 'latent_dim': LATENT_DIM,
                    'prune_threshold': PRUNE_THRESHOLD, 'retrain_epochs': RETRAIN_EPOCHS,
                    'retrain_lr': RETRAIN_LR, 'seed': seed,
                    'n_pruned': n_pruned,
                },
            }, pruned_path)
            print(f"  Saved {pruned_path}")

            M = model.koopman_matrix().cpu().numpy()
            eigs = np.linalg.eigvals(M)
            all_eigs.append(eigs)

        plot_eigenvalues(all_eigs, OUTDIR, SEEDS)

    # ------------------------------------------------------------------
    # Analysis: load best checkpoint and run proper diagnostics
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("ANALYSIS: Foliation Discovery + h∘g Decomposition")
    print(f"{'='*70}")

    # Determine which seed to analyze
    if args.seed is not None:
        analyze_seed = args.seed
    else:
        # Find best seed by val loss from checkpoint metadata
        best_loss = float('inf')
        analyze_seed = SEEDS[0]
        for s in SEEDS:
            ckpt_path = OUTDIR / f'ikeda_seed_{s}.pt'
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                vl = ckpt.get('val_loss', float('inf'))
                if vl < best_loss:
                    best_loss = vl
                    analyze_seed = s

    # Load pruned model
    pruned_path = OUTDIR / f'ikeda_pruned_seed_{analyze_seed}.pt'
    print(f"\nLoading pruned model: {pruned_path}")
    ckpt = torch.load(pruned_path, map_location='cpu', weights_only=False)
    model = IkedaStableKoopman()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Seed: {analyze_seed}")

    # Eigenvalue info
    M = model.koopman_matrix().cpu().numpy()
    eigs = np.linalg.eigvals(M)
    mags = np.abs(eigs)
    print(f"  Koopman |lam|: [{mags.min():.4f}, {mags.max():.4f}]")

    # Prediction horizon
    print(f"\n--- Prediction Horizon ---")
    mean_horizon, horizons, mse_curve = compute_prediction_horizon(
        model, val_traj, n_ics=HORIZON_N_ICS,
        max_steps=HORIZON_MAX_STEPS, threshold=HORIZON_MSE_THRESHOLD)
    print(f"  Mean prediction horizon: {mean_horizon:.1f} steps "
          f"(MSE threshold={HORIZON_MSE_THRESHOLD})")

    # Prepare attractor data for analysis
    train_traj_np = traj[:n_train]
    all_pts = train_traj_np.reshape(-1, 2)
    # Subsample for analysis speed
    rng = np.random.RandomState(0)
    n_sub = min(20000, len(all_pts))
    idx = rng.choice(len(all_pts), n_sub, replace=False)
    pts = torch.tensor(all_pts[idx], dtype=torch.float32)

    # ------------------------------------------------------------------
    # Phase 4: Foliation analysis
    # ------------------------------------------------------------------
    print(f"\n--- Phase 4: Foliation Discovery ---")
    foliation_results = run_foliation_analysis(model, pts, OUTDIR)

    # ------------------------------------------------------------------
    # Phase 5a: Layer-1 inner function decomposition (the key result)
    # ------------------------------------------------------------------
    print(f"\n--- Phase 5a: Layer-1 h∘g Decomposition ---")
    print(f"  (Testing whether u_j = ψ_{'{j,0}'}(x) + ψ_{'{j,1}'}(y) ≈ a·(x²+y²) + linear)")
    layer1_decomp = run_layer1_decomposition(model, pts, OUTDIR,
                                              max_degree=3, alpha=1e-3)

    # ------------------------------------------------------------------
    # Phase 5b: z_k decomposition via manifold_analysis
    # ------------------------------------------------------------------
    print(f"\n--- Phase 5b: z_k h∘g Decomposition (manifold_analysis) ---")
    print(f"  (Expected: dense g because z_k is bilinear in cos(t(r²))·x, sin(t(r²))·y)")
    decomp_results, decomp_serializable = run_decomposition(
        model, train_traj_np, OUTDIR,
        max_degree=LASSO_MAX_DEGREE, lasso_alpha=LASSO_ALPHA,
        h_degree=4)

    # ------------------------------------------------------------------
    # Eigenvalue plot for this model
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    theta_c = np.linspace(0, 2 * np.pi, 200)

    ax = axes[0]
    ax.plot(np.cos(theta_c), np.sin(theta_c), 'k:', alpha=0.3)
    sc = ax.scatter(eigs.real, eigs.imag, s=60, c=mags, cmap='RdYlGn',
                    vmin=0, vmax=1, edgecolors='black', linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label='|λ|')
    ax.set_xlabel(r'Re($\lambda$)'); ax.set_ylabel(r'Im($\lambda$)')
    ax.set_title(f'Koopman Eigenvalues (seed {analyze_seed})\n'
                 f'|λ| ∈ [{mags.min():.4f}, {mags.max():.4f}]')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

    A = model.prop.generator_matrix().detach().cpu().numpy()
    eigs_A = np.linalg.eigvals(A)
    ax = axes[1]
    ax.axvline(0, color='red', ls='--', lw=1, alpha=0.5)
    sc2 = ax.scatter(eigs_A.real, eigs_A.imag, s=60, c=np.abs(eigs_A),
                     cmap='RdYlGn', edgecolors='black', linewidths=0.5, zorder=3)
    plt.colorbar(sc2, ax=ax, label='|μ|')
    ax.set_xlabel(r'Re($\mu$)'); ax.set_ylabel(r'Im($\mu$)')
    ax.set_title(f'Generator A = K − L$^T$L\nmax Re(μ) = {eigs_A.real.max():.4f}')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

    fig.suptitle(f'Ikeda Koopman Spectrum (seed {analyze_seed})', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'eigenvalues_best.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'eigenvalues_best.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {OUTDIR / 'eigenvalues_best.png'}")

    # ------------------------------------------------------------------
    # Save all results
    # ------------------------------------------------------------------
    analysis_results = {
        'system': 'ikeda',
        'u': 0.9,
        'analyze_seed': analyze_seed,
        'mean_prediction_horizon': mean_horizon,
        'eigenvalue_mags_min': float(mags.min()),
        'eigenvalue_mags_max': float(mags.max()),
        'foliation': {
            'edge_classifications': foliation_results['edge_classifications'],
            'foliation_scores': foliation_results['foliation_scores'],
            'channel_activity': foliation_results['channel_activity'],
            'n_aligned_r2': foliation_results['n_aligned_r2'],
            'mode_Ak': foliation_results['mode_Ak'],
        },
        'layer1_decomposition': layer1_decomp,
        'decomposition': decomp_serializable,
    }

    # numpy bool/int/float -> python native for JSON
    def _sanitize(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    with open(OUTDIR / 'analysis_results.json', 'w') as f:
        json.dump(_sanitize(analysis_results), f, indent=2)
    print(f"\nSaved {OUTDIR / 'analysis_results.json'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Seed: {analyze_seed}")
    print(f"  Prediction horizon: {mean_horizon:.1f} steps")
    print(f"  Koopman |lam|: [{mags.min():.4f}, {mags.max():.4f}]")
    print(f"  Foliation: {foliation_results['n_aligned_r2']}/{MID_DIM} "
          f"layer-1 nodes aligned with r² (R²>0.8)")
    print(f"  Mode |A_k|: {foliation_results['mode_Ak']}")

    # Best decomposition result
    valid_decomps = [d for d in decomp_serializable if d.get('r2_hg') is not None]
    if valid_decomps:
        best_decomp = max(valid_decomps, key=lambda d: d['r2_hg'])
        print(f"  Best h∘g: z[{best_decomp['dim']}] with "
              f"R²(g)={best_decomp['r2_g']:.4f}, R²(h∘g)={best_decomp['r2_hg']:.4f}")
        print(f"    Active terms: {best_decomp['active_names']}")
        print(f"    h type: {'affine' if best_decomp['h_affine'] else 'nonlinear'}")

    # Layer-1 r² signature check (the key result)
    n_r2_sig = sum(1 for r in layer1_decomp if r['r2_signature'])
    print(f"  Layer-1 r² signature: {n_r2_sig}/{MID_DIM} nodes have x²≈y² (same sign, ratio>0.5)")
    for r in layer1_decomp:
        if r['r2_signature']:
            print(f"    u[{r['node']}]: x²={r['x2_coeff']:+.4f}, y²={r['y2_coeff']:+.4f}, "
                  f"ratio={r['r2_ratio']:.3f}")

    # z_k decomposition (expected dense for bilinear regime)
    if valid_decomps:
        median_n_active = np.median([d['n_active'] for d in valid_decomps])
        print(f"  z_k Lasso: median {median_n_active:.0f} active terms "
              f"(expected: dense, because z_k is bilinear)")
    else:
        print(f"  z_k decomposition: no valid results")

    print(f"\nAll results in {OUTDIR}/")
    print("Done.")
