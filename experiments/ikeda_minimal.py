"""
Ikeda Map with minimal architecture [2, 4, 4].

The dictionary {x, y, cos(t(r²)), sin(t(r²))} has 4 terms, so the
architecture matches: 4 mid nodes for inner functions, 4 latent dims
for the Koopman observables.

Same training pipeline as ikeda_koopman.py but with the theory-matched
architecture. Includes full foliation + gradient identity analysis.

Run from project root:
    python experiments/ikeda_minimal.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Lasso
from numpy.linalg import lstsq

from kan import KANLinear
from training import set_seed
from ikeda import generate_ikeda, IkedaPairDataset, ikeda_step
from plotting import inspect_kanlinear


def extract_inner_functions(encoder, x_data):
    """Extract layer-1 inner functions u_j and per-edge activations.

    Replaces the manifold_analysis_foliation dependency with our
    existing plotting.inspect_kanlinear.

    Returns:
        u_values: (N, mid) numpy array of layer-1 outputs
        per_edge: (N, mid, n_in) numpy array of per-edge activations
    """
    import torch
    if isinstance(x_data, np.ndarray):
        x_data = torch.from_numpy(x_data).float()
    encoder.eval()
    with torch.no_grad():
        u = encoder.kan1(x_data).cpu().numpy()
    info = inspect_kanlinear(encoder.kan1, x_data)
    per_edge = info['postacts'].numpy()  # (N, out=mid, in=n_in)
    return u, per_edge

# =====================================================================
# Configuration
# =====================================================================

OUTDIR = Path('results/ikeda_minimal')
OUTDIR.mkdir(parents=True, exist_ok=True)

# Architecture — theory-matched
INPUT_DIM = 2
MID_DIM = 4
LATENT_DIM = 4
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
GAMMA_PREDX = 1.0
ALPHA_LATENT = 1.0
BETA_RECON = 0.5
L1_SPLINE_WEIGHT = 0.001
GRAD_CLIP = 1.0

# Pruning
PRUNE_THRESHOLD = 0.03
RETRAIN_EPOCHS = 100
RETRAIN_LR = 5e-4

# Prediction horizon
HORIZON_N_ICS = 50
HORIZON_MAX_STEPS = 200
HORIZON_MSE_THRESHOLD = 0.1

# Lasso
LASSO_MAX_DEGREE = 3
LASSO_ALPHA = 1e-3

SEEDS = [0, 1, 2, 3, 4]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =====================================================================
# Model
# =====================================================================

class StablePropagator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        n_skew = dim * (dim - 1) // 2
        self.skew_params = nn.Parameter(0.01 * torch.randn(n_skew))
        self.L_diag = nn.Parameter(0.01 * torch.randn(dim))
        self.L_lower = nn.Parameter(0.01 * torch.randn(n_skew))

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
        return self._get_skew() - self._get_L().T @ self._get_L()

    def get_matrix(self):
        return torch.matrix_exp(self.generator_matrix())

    def forward(self, z):
        return z @ self.get_matrix().T


class IkedaMinimalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan1 = KANLinear(INPUT_DIM, MID_DIM, grid_size=GRID_SIZE,
                              spline_order=SPLINE_ORDER, grid_range=[-3, 3])
        self.kan2 = KANLinear(MID_DIM, LATENT_DIM, grid_size=GRID_SIZE,
                              spline_order=SPLINE_ORDER, grid_range=[-5, 5])

    def forward(self, x):
        return self.kan2(self.kan1(x))


class IkedaMinimalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan1 = KANLinear(LATENT_DIM, MID_DIM, grid_size=GRID_SIZE,
                              spline_order=SPLINE_ORDER, grid_range=[-5, 5])
        self.kan2 = KANLinear(MID_DIM, INPUT_DIM, grid_size=GRID_SIZE,
                              spline_order=SPLINE_ORDER, grid_range=[-3, 3])

    def forward(self, z):
        return self.kan2(self.kan1(z))


class IkedaMinimalKoopman(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IkedaMinimalEncoder()
        self.decoder = IkedaMinimalDecoder()
        self.prop = StablePropagator(LATENT_DIM)
        self.latent_dim = LATENT_DIM

    def forward_one_step(self, xy):
        z = self.encoder(xy)
        xy_rec = self.decoder(z)
        z_next = self.prop(z)
        xy_next = self.decoder(z_next)
        return {'z_n': z, 'xy_n_rec': xy_rec,
                'z_np1_pred': z_next, 'xy_np1_pred': xy_next}

    def koopman_matrix(self):
        return self.prop.get_matrix().detach()


# =====================================================================
# Training
# =====================================================================

def spline_l1_reg(model):
    reg = 0.0
    for m in model.modules():
        if isinstance(m, KANLinear):
            reg = reg + m.spline_weight.abs().mean()
    return reg


def train_one_seed(seed, traj):
    set_seed(seed)
    n_train = int(0.8 * traj.shape[0])
    train_dl = DataLoader(IkedaPairDataset(traj[:n_train]),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(IkedaPairDataset(traj[n_train:]),
                        batch_size=BATCH_SIZE)

    model = IkedaMinimalKoopman().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model [2,{MID_DIM},{LATENT_DIM}]: {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    mse = nn.MSELoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xy_n, xy_np1 in train_dl:
            xy_n, xy_np1 = xy_n.to(DEVICE), xy_np1.to(DEVICE)
            out = model.forward_one_step(xy_n)
            z_true = model.encoder(xy_np1)
            loss = (GAMMA_PREDX * mse(out['xy_np1_pred'], xy_np1)
                    + ALPHA_LATENT * mse(out['z_np1_pred'], z_true)
                    + BETA_RECON * mse(out['xy_n_rec'], xy_n)
                    + L1_SPLINE_WEIGHT * spline_l1_reg(model))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())
        sched.step()

        if ep == 1 or ep % 50 == 0 or ep == EPOCHS:
            model.eval()
            vl = []
            with torch.no_grad():
                for xy_n, xy_np1 in val_dl:
                    xy_n, xy_np1 = xy_n.to(DEVICE), xy_np1.to(DEVICE)
                    out = model.forward_one_step(xy_n)
                    z_true = model.encoder(xy_np1)
                    l = (GAMMA_PREDX * mse(out['xy_np1_pred'], xy_np1)
                         + ALPHA_LATENT * mse(out['z_np1_pred'], z_true)
                         + BETA_RECON * mse(out['xy_n_rec'], xy_n))
                    vl.append(l.item())
            M = model.koopman_matrix().cpu().numpy()
            mags = np.abs(np.linalg.eigvals(M))
            print(f"  Ep {ep:3d} | train {np.mean(losses):.5f} "
                  f"| val {np.mean(vl):.5f} "
                  f"| |lam| [{mags.min():.4f}, {mags.max():.4f}]", flush=True)

    return model.cpu().eval()


# =====================================================================
# Pruning + retraining
# =====================================================================

def prune_model(model):
    total = 0
    for name, m in model.named_modules():
        if isinstance(m, KANLinear):
            with torch.no_grad():
                imp = m.spline_weight.abs().mean(dim=-1)
                if hasattr(m, 'spline_scaler') and m.spline_scaler is not None:
                    imp = imp + m.spline_scaler.abs()
                mask = imp < PRUNE_THRESHOLD
                n = int(mask.sum().item())
                for oi in range(m.out_features):
                    for ii in range(m.in_features):
                        if mask[oi, ii]:
                            m.spline_weight[oi, ii, :] = 0.0
                            if hasattr(m, 'spline_scaler') and m.spline_scaler is not None:
                                m.spline_scaler[oi, ii] = 0.0
                            m.base_weight[oi, ii] = 0.0
            total_edges = m.out_features * m.in_features
            print(f"    Pruned {name}: {n}/{total_edges} edges")
            total += n
    return total


def retrain_pruned(model, traj):
    n_train = int(0.8 * traj.shape[0])
    train_dl = DataLoader(IkedaPairDataset(traj[:n_train]),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(IkedaPairDataset(traj[n_train:]), batch_size=BATCH_SIZE)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=RETRAIN_LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=RETRAIN_EPOCHS)
    mse = nn.MSELoss()

    for ep in range(1, RETRAIN_EPOCHS + 1):
        model.train()
        losses = []
        for xy_n, xy_np1 in train_dl:
            xy_n, xy_np1 = xy_n.to(DEVICE), xy_np1.to(DEVICE)
            out = model.forward_one_step(xy_n)
            z_true = model.encoder(xy_np1)
            loss = (GAMMA_PREDX * mse(out['xy_np1_pred'], xy_np1)
                    + ALPHA_LATENT * mse(out['z_np1_pred'], z_true)
                    + BETA_RECON * mse(out['xy_n_rec'], xy_n)
                    + L1_SPLINE_WEIGHT * spline_l1_reg(model))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())
        sched.step()
        if ep == 1 or ep % 25 == 0 or ep == RETRAIN_EPOCHS:
            model.eval()
            vl = []
            with torch.no_grad():
                for xy_n, xy_np1 in val_dl:
                    xy_n, xy_np1 = xy_n.to(DEVICE), xy_np1.to(DEVICE)
                    out = model.forward_one_step(xy_n)
                    z_true = model.encoder(xy_np1)
                    l = (GAMMA_PREDX * mse(out['xy_np1_pred'], xy_np1)
                         + ALPHA_LATENT * mse(out['z_np1_pred'], z_true)
                         + BETA_RECON * mse(out['xy_n_rec'], xy_n))
                    vl.append(l.item())
            print(f"    Retrain ep {ep:3d} | train {np.mean(losses):.5f} "
                  f"| val {np.mean(vl):.5f}", flush=True)
    return model.cpu().eval()


# =====================================================================
# Prediction horizon
# =====================================================================

def compute_horizon(model, val_traj):
    model.eval()
    rng = np.random.RandomState(42)
    n_val = val_traj.shape[0]
    ic_traj = rng.randint(0, n_val, size=HORIZON_N_ICS)
    ic_step = rng.randint(0, val_traj.shape[1], size=HORIZON_N_ICS)
    horizons = []
    with torch.no_grad():
        for i in range(HORIZON_N_ICS):
            x_true = val_traj[ic_traj[i], ic_step[i]].numpy()
            x_model = torch.tensor(x_true, dtype=torch.float32).unsqueeze(0)
            found = False
            for step in range(HORIZON_MAX_STEPS):
                xn, yn = ikeda_step(float(x_true[0]), float(x_true[1]))
                x_true = np.array([xn, yn], dtype=np.float32)
                out = model.forward_one_step(x_model)
                x_model = out['xy_np1_pred']
                err = float(np.mean((x_model.squeeze().numpy() - x_true)**2))
                if not found and err > HORIZON_MSE_THRESHOLD:
                    horizons.append(step)
                    found = True
            if not found:
                horizons.append(HORIZON_MAX_STEPS)
    return float(np.mean(horizons)), horizons


# =====================================================================
# Analysis: edge splines + Lasso + gradient identity
# =====================================================================

def build_poly_dict_2d(x, y, max_degree=3):
    terms, names = [], []
    for total in range(max_degree + 1):
        for p in range(total + 1):
            q = total - p
            term = (x ** p) * (y ** q)
            parts = []
            if p == 1: parts.append("x")
            elif p > 1: parts.append(f"x^{p}")
            if q == 1: parts.append("y")
            elif q > 1: parts.append(f"y^{q}")
            name = "".join(parts) if parts else "1"
            terms.append(term)
            names.append(name)
    return np.column_stack(terms), names


def analyze_model(model, pts_sub):
    """Full analysis: edge splines, Lasso on u_j, gradient identity."""
    x_data = torch.from_numpy(pts_sub)
    u_vals, per_edge = extract_inner_functions(model.encoder, x_data)
    x_np, y_np = pts_sub[:, 0], pts_sub[:, 1]
    r2 = x_np**2 + y_np**2

    n_ch = u_vals.shape[1]
    results = {'channels': [], 'gradient_identity': []}

    # --- Per-channel: edge spline fits + Lasso ---
    M_dict, names = build_poly_dict_2d(x_np, y_np, max_degree=LASSO_MAX_DEGREE)
    col_mean = M_dict.mean(0)
    col_std = M_dict.std(0)
    col_std[col_std < 1e-12] = 1.0
    M_scaled = (M_dict - col_mean) / col_std

    for j in range(n_ch):
        psi_x = per_edge[:, j, 0]
        psi_y = per_edge[:, j, 1]

        # Quadratic fits
        c2_x = np.polyfit(x_np, psi_x, 2)
        c2_y = np.polyfit(y_np, psi_y, 2)
        ratio = min(abs(c2_x[0]), abs(c2_y[0])) / max(abs(c2_x[0]), abs(c2_y[0]), 1e-12)
        same_sign = c2_x[0] * c2_y[0] > 0

        # Lasso
        target = u_vals[:, j]
        t_mean, t_std = target.mean(), max(target.std(), 1e-12)
        t_scaled = (target - t_mean) / t_std
        lasso = Lasso(alpha=LASSO_ALPHA, max_iter=50000, fit_intercept=True)
        lasso.fit(M_scaled, t_scaled)
        coeffs = t_std * lasso.coef_ / col_std
        intercept = t_mean + t_std * (lasso.intercept_ - np.sum(lasso.coef_ * col_mean / col_std))
        pred = M_dict @ coeffs + intercept
        ss_res = np.sum((target - pred)**2)
        ss_tot = np.sum((target - target.mean())**2)
        r2_lasso = 1 - ss_res / max(ss_tot, 1e-12)

        mask = np.abs(coeffs) > 1e-4
        active = [(names[i], float(coeffs[i])) for i in np.where(mask)[0]]
        active.sort(key=lambda x: -abs(x[1]))

        x2_c = coeffs[names.index('x^2')]
        y2_c = coeffs[names.index('y^2')]
        lasso_ratio = min(abs(x2_c), abs(y2_c)) / max(abs(x2_c), abs(y2_c), 1e-12)

        results['channels'].append({
            'channel': j,
            'x2_edge': float(c2_x[0]), 'y2_edge': float(c2_y[0]),
            'edge_ratio': float(ratio), 'same_sign': bool(same_sign),
            'r2_lasso': float(r2_lasso), 'n_active': int(mask.sum()),
            'x2_lasso': float(x2_c), 'y2_lasso': float(y2_c),
            'lasso_ratio': float(lasso_ratio),
            'active': active,
        })

    # --- Gradient identity on best r² channel ---
    # Pick channel with highest lasso_ratio where same_sign is True
    r2_channels = [c for c in results['channels']
                   if c['same_sign'] and c['lasso_ratio'] > 0.5
                   and abs(c['x2_lasso']) > 0.01]
    if r2_channels:
        best_ch = max(r2_channels, key=lambda c: c['lasso_ratio'])['channel']
    else:
        best_ch = max(results['channels'], key=lambda c: c['lasso_ratio'])['channel']

    results['best_r2_channel'] = best_ch

    x_t = torch.from_numpy(pts_sub).float().requires_grad_(True)
    u_all = model.encoder.kan1(x_t)
    u_ch = u_all[:, best_ch]
    grad_f = torch.autograd.grad(u_ch.sum(), x_t, create_graph=False)[0].detach().numpy()

    grad_g = np.stack([2 * x_np, 2 * y_np], axis=-1)
    grad_g_norm_sq = (grad_g**2).sum(-1)
    h_prime = (grad_f * grad_g).sum(-1) / np.maximum(grad_g_norm_sq, 1e-8)
    u_np = u_ch.detach().numpy()
    good = grad_g_norm_sq > 0.5

    g_good = r2[good]
    hp_good = h_prime[good]
    u_good = u_np[good]
    sort_idx = np.argsort(g_good)
    g_s, hp_s, u_s = g_good[sort_idx], hp_good[sort_idx], u_good[sort_idx]

    n_bins = 80
    be = np.linspace(g_s.min(), g_s.max(), n_bins + 1)
    bc, bm_hp, bm_u = [], [], []
    for i in range(n_bins):
        m = (g_s >= be[i]) & (g_s < be[i + 1])
        if m.sum() > 15:
            bc.append((be[i] + be[i + 1]) / 2)
            bm_hp.append(np.median(hp_s[m]))
            bm_u.append(u_s[m].mean())
    bc, bm_hp, bm_u = np.array(bc), np.array(bm_hp), np.array(bm_u)

    # Integrate h' -> h
    dg = np.diff(bc)
    h_int = np.concatenate([[0], np.cumsum(bm_hp[:-1] * dg)])
    h_int += bm_u[0]
    r2_int = 1 - np.sum((bm_u - h_int)**2) / max(np.sum((bm_u - bm_u.mean())**2), 1e-12)

    # Trig fit: h(g) = a*cos(t(g)) + b*sin(t(g)) + c
    t_bc = 0.4 - 6.0 / (1.0 + bc)
    A_trig = np.column_stack([np.cos(t_bc), np.sin(t_bc), np.ones(len(bc))])
    sol, _, _, _ = lstsq(A_trig, bm_u, rcond=None)
    pred_trig = A_trig @ sol
    r2_trig = 1 - np.sum((bm_u - pred_trig)**2) / max(np.sum((bm_u - bm_u.mean())**2), 1e-12)

    dt_dg = 6.0 / (1.0 + bc)**2
    hp_theory = (-sol[0] * np.sin(t_bc) + sol[1] * np.cos(t_bc)) * dt_dg

    results['gradient_identity'] = {
        'channel': best_ch,
        'a_cos': float(sol[0]), 'b_sin': float(sol[1]), 'c_const': float(sol[2]),
        'r2_trig': float(r2_trig), 'r2_integrated': float(r2_int),
        'bc': bc.tolist(), 'bm_u': bm_u.tolist(),
        'pred_trig': pred_trig.tolist(), 'h_int': h_int.tolist(),
        'bm_hp': bm_hp.tolist(), 'hp_theory': hp_theory.tolist(),
    }

    return results


# =====================================================================
# Plotting
# =====================================================================

def plot_results(results, seed):
    gi = results['gradient_identity']
    bc = np.array(gi['bc'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ch = gi['channel']

    ax = axes[0]
    ax.plot(bc, gi['bm_hp'], 'b-', lw=2.5, label="h'(g) recovered")
    ax.plot(bc, gi['hp_theory'], 'r--', lw=2, label="h'(g) from trig fit")
    ax.set_xlabel('g = x^2 + y^2')
    ax.set_ylabel("h'(g)")
    ax.set_title("(a) Outer derivative h'(g)")
    ax.legend(); ax.grid(True, alpha=0.15)

    ax = axes[1]
    ax.plot(bc, gi['bm_u'], 'b-', lw=2.5, label=f'u_{ch} (binned)')
    ax.plot(bc, gi['pred_trig'], 'r--', lw=2.5,
            label=f"{gi['a_cos']:.2f}cos(t) + {gi['b_sin']:.2f}sin(t) + {gi['c_const']:.2f}")
    ax.set_xlabel('g = x^2 + y^2')
    ax.set_ylabel('h(g)')
    ax.set_title(f"(b) Trig fit R^2 = {gi['r2_trig']:.3f}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    ax = axes[2]
    ax.plot(bc, gi['bm_u'], 'b-', lw=2.5, label=f'u_{ch} (actual)')
    ax.plot(bc, gi['h_int'], 'r--', lw=2.5, label="h(g) integrated")
    ax.set_xlabel('g = x^2 + y^2')
    ax.set_ylabel('h(g)')
    ax.set_title(f"(c) Integrated R^2 = {gi['r2_integrated']:.3f}")
    ax.legend(); ax.grid(True, alpha=0.15)

    fig.suptitle(f'Ikeda [2,4,4] seed {seed}: u_{ch} = h(x^2+y^2),  '
                 f'h = a*cos(t(g)) + b*sin(t(g)) + c', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / f'outer_h_seed_{seed}.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / f'outer_h_seed_{seed}.pdf', bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("=" * 70)
    print(f"  Ikeda Map: Minimal Architecture [{INPUT_DIM}, {MID_DIM}, {LATENT_DIM}]")
    print(f"  Device: {DEVICE}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    traj = generate_ikeda(n_traj=N_TRAJ, n_steps=N_STEPS, burn_in=BURN_IN, seed=42)
    traj_torch = torch.from_numpy(traj)
    n_train = int(0.8 * N_TRAJ)
    val_traj = traj_torch[n_train:]

    pts_sub = traj.reshape(-1, 2)
    np.random.seed(0)
    idx = np.random.choice(len(pts_sub), 20000, replace=False)
    pts_sub = pts_sub[idx]

    print(f"  Data: {traj.shape}")
    print(f"  x: [{traj[:,:,0].min():.3f}, {traj[:,:,0].max():.3f}]")
    print(f"  y: [{traj[:,:,1].min():.3f}, {traj[:,:,1].max():.3f}]")

    all_results = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        # Train
        model = train_one_seed(seed, traj_torch)
        torch.save({'model_state_dict': model.state_dict(), 'seed': seed},
                   OUTDIR / f'model_seed_{seed}.pt')

        # Prune + retrain
        print(f"\n  Pruning (threshold={PRUNE_THRESHOLD})...")
        n_pruned = prune_model(model)
        print(f"  Total pruned: {n_pruned}")
        model = retrain_pruned(model, traj_torch)
        torch.save({'model_state_dict': model.state_dict(), 'seed': seed,
                     'n_pruned': n_pruned},
                   OUTDIR / f'pruned_seed_{seed}.pt')

        # Prediction horizon
        hor_mean, hor_list = compute_horizon(model, val_traj)
        print(f"\n  Horizon: {hor_mean:.1f} [{min(hor_list)}, {max(hor_list)}]")

        # Eigenvalues
        M = model.koopman_matrix().numpy()
        eigs = np.linalg.eigvals(M)
        mags = np.abs(eigs)
        print(f"  |lam|: [{mags.min():.4f}, {mags.max():.4f}]")

        # Analysis
        print(f"\n  Running analysis...")
        analysis = analyze_model(model, pts_sub)
        analysis['horizon_mean'] = hor_mean
        analysis['horizon_list'] = hor_list
        analysis['n_pruned'] = n_pruned
        analysis['eigenvalues_real'] = eigs.real.tolist()
        analysis['eigenvalues_imag'] = eigs.imag.tolist()

        # Print summary
        print(f"\n  Channel analysis:")
        for c in analysis['channels']:
            top3 = ', '.join([f"{n}({v:+.3f})" for n, v in c['active'][:3]])
            print(f"    u[{c['channel']}]: x2={c['x2_lasso']:+.4f} y2={c['y2_lasso']:+.4f} "
                  f"ratio={c['lasso_ratio']:.3f} R2={c['r2_lasso']:.4f} [{top3}]")

        gi = analysis['gradient_identity']
        print(f"\n  Gradient identity (channel {gi['channel']}):")
        print(f"    h(g) = {gi['a_cos']:.3f}*cos(t(g)) + {gi['b_sin']:.3f}*sin(t(g)) + {gi['c_const']:.3f}")
        print(f"    Trig fit R^2 = {gi['r2_trig']:.4f}")

        # Plot
        plot_results(analysis, seed)

        # Store (strip large arrays for JSON)
        store = {k: v for k, v in analysis.items() if k != 'gradient_identity'}
        gi_store = {k: v for k, v in gi.items()
                    if k not in ('bc', 'bm_u', 'pred_trig', 'h_int', 'bm_hp', 'hp_theory')}
        store['gradient_identity'] = gi_store
        all_results[f'seed_{seed}'] = store

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for sname, sr in all_results.items():
        gi = sr['gradient_identity']
        print(f"  {sname}: horizon={sr['horizon_mean']:.1f} "
              f"pruned={sr['n_pruned']} "
              f"trig_R2={gi['r2_trig']:.3f} "
              f"h={gi['a_cos']:.2f}cos+{gi['b_sin']:.2f}sin+{gi['c_const']:.2f}")

    with open(OUTDIR / 'results.json', 'w') as f:
        json.dump({'config': {'mid': MID_DIM, 'latent': LATENT_DIM,
                              'epochs': EPOCHS, 'prune': PRUNE_THRESHOLD},
                   'per_seed': all_results}, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR / 'results.json'}")
    print("  Done.")
