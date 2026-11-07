"""
Ikeda Map: PyKAN backend with sparsity + pruning sweep.

Uses the PyKAN backend (not KANLinear) for the encoder/decoder, which
gives access to PyKAN's built-in:
  - Sparsity regularization via model.get_reg() (L1 + entropy)
  - Structured pruning via importance scoring
  - Grid updates from data (update_grid_from_samples)

Sweeps over:
  - lamb (sparsity weight): [0, 0.001, 0.01, 0.05]
  - prune_threshold: [0.01, 0.03, 0.05, 0.1]

For each (lamb, threshold), trains, prunes, retrains, then evaluates:
  - Prediction horizon
  - Layer-1 foliation alignment (R²_radial of u_j vs r²)
  - Channel activity |A_k|
  - Layer-1 r² signature (x²≈y² same sign)

The Ikeda map is discrete-time, so the propagator is
  M = expm(K - L^T L)   (stable, |λ| ≤ 1)
applied as z_{n+1} = M z_n (no dt parameter).

Usage:
  python experiments/ikeda_pykan_sweep.py
  python experiments/ikeda_pykan_sweep.py --device cpu
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from models import _import_pykan
from training import set_seed
from ikeda import generate_ikeda, IkedaPairDataset, ikeda_step
from manifold_analysis import recover_inner_function_nd

OUTDIR = Path(__file__).resolve().parent.parent / 'results' / 'ikeda_pykan'

# Architecture
MID = 6
LATENT = 8

# Data (must use full dataset — PyKAN underperforms badly with less data)
N_TRAJ = 200
N_STEPS = 2000
BURN_IN = 500

# Training (PyKAN is ~5x slower per batch than KANLinear, so fewer epochs)
EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 2048

# Loss weights
GAMMA = 1.0   # prediction
ALPHA = 1.0   # latent consistency
BETA = 0.5    # reconstruction

# Sweep
LAMB_VALUES = [0.0, 0.001, 0.01, 0.05]
PRUNE_THRESHOLDS = [0.01, 0.03, 0.05, 0.1]
RETRAIN_EPOCHS = 100
RETRAIN_LR = 5e-4

# PyKAN's .to(device) does not reliably move internal spline grids
# to CUDA, causing device-mismatch errors. Force CPU.
DEVICE = 'cpu'


# =====================================================================
# Stable propagator (same as ikeda_koopman.py)
# =====================================================================

class StablePropagator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        n_skew = dim * (dim - 1) // 2
        self.skew_params = nn.Parameter(0.01 * torch.randn(n_skew))
        self.L_diag = nn.Parameter(0.01 * torch.randn(dim))
        self.L_lower = nn.Parameter(0.01 * torch.randn(dim * (dim - 1) // 2))

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


# =====================================================================
# PyKAN-based Koopman model
# =====================================================================

class IkedaPyKANKoopman(nn.Module):
    def __init__(self, mid=MID, latent=LATENT, grid=5, k=3):
        super().__init__()
        KAN = _import_pykan()
        self.encoder = KAN(width=[2, mid, latent], grid=grid, k=k, seed=0)
        self.decoder = KAN(width=[latent, mid, 2], grid=grid, k=k, seed=1)
        self.prop = StablePropagator(latent)
        self.latent_dim = latent

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

def train_model(traj_torch, lamb, device, seed=0):
    set_seed(seed)
    n_train = int(0.8 * traj_torch.shape[0])
    train_dl = DataLoader(IkedaPairDataset(traj_torch[:n_train]),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(IkedaPairDataset(traj_torch[n_train:]),
                        batch_size=BATCH_SIZE)

    model = IkedaPyKANKoopman().to(device)
    # Disable save_act during training unless we need regularization
    model.encoder.save_act = (lamb > 0)
    model.decoder.save_act = False

    print(f"    Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Grid initialization from data
    x_sample = traj_torch[:n_train].reshape(-1, 2)
    idx = np.random.choice(len(x_sample), min(3000, len(x_sample)), replace=False)
    x_init = x_sample[idx].to(device)
    with torch.no_grad():
        model.encoder.update_grid_from_samples(x_init)
        z_init = model.encoder(x_init)
        model.decoder.update_grid_from_samples(z_init)
    print(f"    Grids initialized from {len(x_init)} samples")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    mse = nn.MSELoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_losses = []
        for xy_n, xy_np1 in train_dl:
            xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
            out = model.forward_one_step(xy_n)
            z_true = model.encoder(xy_np1)

            loss = (GAMMA * mse(out['xy_np1_pred'], xy_np1)
                    + ALPHA * mse(out['z_np1_pred'], z_true)
                    + BETA * mse(out['xy_n_rec'], xy_n))

            # Sparsity regularization
            if lamb > 0 and model.encoder.save_act:
                reg = model.encoder.get_reg(
                    'edge_forward_spline_n', lamb_l1=1.0, lamb_entropy=2.0,
                    lamb_coef=0.0, lamb_coefdiff=0.0)
                loss = loss + lamb * reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())
        sched.step()

        # Re-update grids periodically
        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                model.encoder.update_grid_from_samples(x_init)
                z_new = model.encoder(x_init)
                model.decoder.update_grid_from_samples(z_new)

        if ep == 1 or ep % 50 == 0 or ep == EPOCHS:
            model.eval()
            vloss = []
            with torch.no_grad():
                for xy_n, xy_np1 in val_dl:
                    xy_n, xy_np1 = xy_n.to(device), xy_np1.to(device)
                    out = model.forward_one_step(xy_n)
                    z_true = model.encoder(xy_np1)
                    l = (GAMMA * mse(out['xy_np1_pred'], xy_np1)
                         + ALPHA * mse(out['z_np1_pred'], z_true)
                         + BETA * mse(out['xy_n_rec'], xy_n))
                    vloss.append(l.item())
            M = model.koopman_matrix().cpu().numpy()
            if np.any(np.isnan(M)) or np.any(np.isinf(M)):
                print(f"    Ep {ep:3d} | NaN/Inf in Koopman matrix — stopping early")
                break
            mags = np.abs(np.linalg.eigvals(M))
            print(f"    Ep {ep:3d} | train {np.mean(ep_losses):.5f} "
                  f"| val {np.mean(vloss):.5f} | |λ| [{mags.min():.3f}, {mags.max():.3f}]",
                  flush=True)

    return model.cpu().eval(), float(np.mean(vloss))


# =====================================================================
# Pruning (edge importance via activation variance)
# =====================================================================

def get_edge_importance(kan_model, x_sample):
    """Compute per-edge importance from spline post-activation magnitudes.

    PyKAN stores spline_postacts[l] as (N, out, in) after a forward pass
    with save_act=True. Edge importance = mean absolute activation.
    """
    kan_model.save_act = True
    kan_model.eval()
    with torch.no_grad():
        kan_model(x_sample)

    importances = []
    for l in range(len(kan_model.width_in) - 1):
        acts = kan_model.spline_postacts[l]  # (N, out, in)
        imp = acts.abs().mean(dim=0)  # (out, in)
        importances.append(imp.cpu().numpy())

    kan_model.save_act = False
    return importances


def measure_activity(model, x_sample, threshold):
    """Measure edge importance and channel activity without modifying weights.

    Returns:
        imps: list of (out, in) importance arrays per layer
        activity: list of |A_k| counts from layer-2
        n_below: total edges below threshold
    """
    imps = get_edge_importance(model.encoder, x_sample)

    n_below = sum(int((imp < threshold).sum()) for imp in imps)

    # Channel activity from layer-2 (index l=1)
    if len(imps) > 1:
        imp2 = imps[1]  # (latent, mid)
        activity = [int(np.sum(imp2[k] > threshold)) for k in range(imp2.shape[0])]
    else:
        activity = [0] * LATENT

    return imps, activity, n_below


    # retrain_model removed — sparsity is enforced via lamb during initial
    # training, not via post-hoc weight zeroing + retraining.


# =====================================================================
# Evaluation
# =====================================================================

def compute_horizon(model, val_traj, n_ics=50, max_steps=200, threshold=0.1):
    model.eval()
    rng = np.random.RandomState(42)
    horizons = []
    with torch.no_grad():
        for _ in range(n_ics):
            ti = rng.randint(0, val_traj.shape[0])
            si = rng.randint(0, val_traj.shape[1])
            x_true = val_traj[ti, si].numpy().copy()
            x_model = val_traj[ti, si].unsqueeze(0)
            horizon = max_steps
            for step in range(max_steps):
                x_new, y_new = ikeda_step(float(x_true[0]), float(x_true[1]))
                x_true = np.array([x_new, y_new], dtype=np.float32)
                out = model.forward_one_step(x_model)
                x_model = out['xy_np1_pred']
                err = float(np.mean((x_model.squeeze().numpy() - x_true)**2))
                if err > threshold:
                    horizon = step
                    break
            horizons.append(horizon)
    return float(np.mean(horizons)), horizons


def foliation_metrics(model, pts_np):
    """Compute R²_radial for each layer-1 output and r² signature."""
    pts = torch.tensor(pts_np, dtype=torch.float32)

    # Get layer-1 output via PyKAN internal activations
    model.encoder.save_act = True
    model.encoder.eval()
    with torch.no_grad():
        model.encoder(pts)
    # acts[0] = input, acts[1] = layer-1 output
    u = model.encoder.acts[1].cpu().numpy()  # (N, mid)
    model.encoder.save_act = False

    r2 = pts_np[:, 0]**2 + pts_np[:, 1]**2
    mid = u.shape[1]

    results = []
    for j in range(mid):
        poly = PolynomialFeatures(degree=4)
        feat = poly.fit_transform(r2.reshape(-1, 1))
        reg = LinearRegression().fit(feat, u[:, j])
        r2_radial = reg.score(feat, u[:, j])

        # Lasso for r² signature
        inner = recover_inner_function_nd(pts_np, ['x', 'y'], u[:, j],
                                           max_degree=3, alpha=1e-3)
        x2c = y2c = 0.0
        for nm, cf in zip(inner['active_names'], inner['active_coeffs']):
            if nm == 'x^2': x2c = float(cf)
            elif nm == 'y^2': y2c = float(cf)
        ratio = min(abs(x2c), abs(y2c)) / max(abs(x2c), abs(y2c), 1e-12)
        same_sign = (x2c * y2c > 0) and abs(x2c) > 0.01 and abs(y2c) > 0.01
        sig = same_sign and ratio > 0.5

        results.append({
            'node': j, 'R2_radial': float(r2_radial),
            'x2': x2c, 'y2': y2c, 'ratio': ratio, 'r2_sig': sig,
            'n_active': len(inner['active']),
        })

    return results


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=DEVICE)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Ikeda PyKAN Sparsity + Pruning Sweep")
    print(f"  Architecture: [2, {MID}, {LATENT}]")
    print(f"  lamb values: {LAMB_VALUES}")
    print(f"  prune thresholds: {PRUNE_THRESHOLDS}")
    print(f"  Device: {args.device}")
    print("=" * 70)

    # Data
    traj = generate_ikeda(n_traj=N_TRAJ, n_steps=N_STEPS, burn_in=BURN_IN, seed=42)
    traj_torch = torch.from_numpy(traj)
    n_train = int(0.8 * N_TRAJ)
    val_traj = traj_torch[n_train:]
    train_pts = traj[:n_train].reshape(-1, 2)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(train_pts), min(8000, len(train_pts)), replace=False)
    eval_pts = train_pts[idx]
    eval_pts_t = torch.tensor(eval_pts, dtype=torch.float32)

    all_results = []

    for lamb in LAMB_VALUES:
        print(f"\n{'='*70}")
        print(f"LAMB = {lamb}")
        print(f"{'='*70}")

        # Train
        model, val_loss_pre = train_model(traj_torch, lamb, args.device)
        print(f"  Pre-prune val loss: {val_loss_pre:.5f}")

        # Save pre-prune checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {'mid': MID, 'latent': LATENT, 'lamb': lamb, 'backend': 'PyKAN'},
            'val_loss': val_loss_pre,
        }, OUTDIR / f'ikeda_pykan_lamb{lamb}.pt')

        # Horizon pre-prune
        horizon_pre, _ = compute_horizon(model, val_traj)
        print(f"  Pre-prune horizon: {horizon_pre:.1f} steps")

        # Foliation metrics on the trained model
        fol = foliation_metrics(model, eval_pts)
        n_r2_sig = sum(1 for f in fol if f['r2_sig'])
        best_r2_rad = max(f['R2_radial'] for f in fol)
        print(f"  r² signature: {n_r2_sig}/{MID} nodes, best R²_radial={best_r2_rad:.4f}")
        for f in fol:
            if f['r2_sig']:
                print(f"    u[{f['node']}]: x²={f['x2']:+.4f}, y²={f['y2']:+.4f}, "
                      f"ratio={f['ratio']:.3f}, #terms={f['n_active']}")

        # Measure channel activity at each threshold (no weight modification)
        for tau in PRUNE_THRESHOLDS:
            imps, activity, n_below = measure_activity(model, eval_pts_t, tau)
            mode_ak = int(np.argmax(np.bincount(activity, minlength=MID + 1)))
            print(f"  tau={tau:.2f}: {n_below} edges below, |A_k|={activity}, mode={mode_ak}")

            result = {
                'lamb': lamb,
                'tau': tau,
                'val_loss': val_loss_pre,
                'horizon': horizon_pre,
                'n_below_threshold': n_below,
                'activity': activity,
                'mode_Ak': mode_ak,
                'n_r2_signature': n_r2_sig,
                'best_R2_radial': best_r2_rad,
                'foliation': fol,
            }
            all_results.append(result)

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'lamb':>6s} | {'tau':>5s} | {'val_loss':>8s} | {'horizon':>7s} | "
          f"{'below':>5s} | {'|A_k|':>5s} | {'r² sig':>6s} | {'best R²':>7s}")
    print(f"{'-'*6} | {'-'*5} | {'-'*8} | {'-'*7} | {'-'*5} | {'-'*5} | {'-'*6} | {'-'*7}")

    for r in all_results:
        print(f"{r['lamb']:6.3f} | {r['tau']:5.2f} | {r['val_loss']:8.5f} | "
              f"{r['horizon']:7.1f} | {r['n_below_threshold']:5d} | {r['mode_Ak']:5d} | "
              f"{r['n_r2_signature']:6d} | {r['best_R2_radial']:7.4f}")

    # ── Heatmap plot ───────────────────────────────────────────────
    n_lamb = len(LAMB_VALUES)
    n_tau = len(PRUNE_THRESHOLDS)

    metrics = {
        'horizon': np.zeros((n_lamb, n_tau)),
        'mode_Ak': np.zeros((n_lamb, n_tau)),
        'n_r2_sig': np.zeros((n_lamb, n_tau)),
        'best_R2': np.zeros((n_lamb, n_tau)),
        'val_loss': np.zeros((n_lamb, n_tau)),
    }

    for r in all_results:
        i = LAMB_VALUES.index(r['lamb'])
        j = PRUNE_THRESHOLDS.index(r['tau'])
        metrics['horizon'][i, j] = r['horizon']
        metrics['mode_Ak'][i, j] = r['mode_Ak']
        metrics['n_r2_sig'][i, j] = r['n_r2_signature']
        metrics['best_R2'][i, j] = r['best_R2_radial']
        metrics['val_loss'][i, j] = r['val_loss']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    titles = ['Prediction Horizon', 'Mode |A_k|', 'r² Signature Count',
              'Best R²_radial', 'Val Loss', 'Horizon × r² score']
    data = [metrics['horizon'], metrics['mode_Ak'], metrics['n_r2_sig'],
            metrics['best_R2'], metrics['val_loss'],
            metrics['horizon'] * metrics['n_r2_sig']]
    cmaps = ['YlGn', 'RdYlGn_r', 'YlGn', 'YlGn', 'RdYlGn_r', 'YlGn']

    lamb_labels = [str(l) for l in LAMB_VALUES]
    tau_labels = [str(t) for t in PRUNE_THRESHOLDS]

    for idx, (ax, title, d, cmap) in enumerate(zip(axes.flat, titles, data, cmaps)):
        im = ax.imshow(d, aspect='auto', cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n_tau)); ax.set_xticklabels(tau_labels, fontsize=8)
        ax.set_yticks(range(n_lamb)); ax.set_yticklabels(lamb_labels, fontsize=8)
        ax.set_xlabel('prune threshold τ', fontsize=9)
        ax.set_ylabel('sparsity λ', fontsize=9)
        ax.set_title(title, fontsize=10)
        for ii in range(n_lamb):
            for jj in range(n_tau):
                val = d[ii, jj]
                fmt = f'{val:.0f}' if val > 1 else f'{val:.3f}'
                ax.text(jj, ii, fmt, ha='center', va='center', fontsize=7)

    fig.suptitle('Ikeda PyKAN: Sparsity × Pruning Sweep', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'sweep_heatmap.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'sweep_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {OUTDIR / 'sweep_heatmap.png'}")

    # Save all results
    def _sanitize(obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_sanitize(v) for v in obj]
        return obj

    with open(OUTDIR / 'sweep_results.json', 'w') as f:
        json.dump(_sanitize(all_results), f, indent=2)
    print(f"Saved {OUTDIR / 'sweep_results.json'}")

    # Best configuration: maximize horizon × foliation quality
    best = max(all_results, key=lambda r: r['horizon'] * (r['n_r2_signature'] + 0.1))
    print(f"\nBest config: lamb={best['lamb']}, tau={best['tau']}")
    print(f"  Horizon: {best['horizon']:.1f}, r² sig: {best['n_r2_signature']}/{MID}, "
          f"|A_k| mode: {best['mode_Ak']}, val_loss: {best['val_loss']:.5f}")
    print(f"\nAll results in {OUTDIR}/")
