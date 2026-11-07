"""
Lorenz system with stable Koopman propagator (spectral radius <= 1).

Generator:  A = K - L^T L   where K = -K^T (skew), L^T L >= 0 (PSD)
    => Re(eigenvalues of A) <= 0
    => |eigenvalues of expm(A·dt)| <= 1

This is strictly more general than the unitary constraint:
    - L = 0  =>  A = K (skew-symmetric)  =>  |λ| = 1  (unitary/standard map)
    - L != 0 =>  damping allowed          =>  |λ| <= 1  (dissipative/Lorenz)

Parameter count: n(n-1)/2 (skew) + n(n+1)/2 (lower-tri L) = n^2, same as
unconstrained, but structurally guarantees stability.

Generates all paper figures:
    1. PyKAN model.plot() for encoder and decoder
    2. Rollout vs truth (time series)
    3. NRMSE in Lyapunov times
    4. Manifold decomposition summary
    5. Layer-1 spline function identification
    6. Level-set visualizations for best dims
    7. 3D attractor overlay (true vs predicted)
    8. Eigenvalue spectrum (with unit circle reference)
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
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.optimize import curve_fit

from models import PyKANAutoEncoder
from data import simulate_lorenz, PairDataset
from training import set_seed
from manifold_analysis import full_decomposition

OUTDIR = Path('results/lorenz_stable')
LAMBDA_MAX = 0.9056  # Lorenz largest Lyapunov exponent (/s)
DT = 0.01
TAU_L = 1.0 / LAMBDA_MAX


# =====================================================================
# Stable Deep Koopman model (spectral radius <= 1)
# =====================================================================

class DeepKoopmanCTStable(nn.Module):
    """Continuous-time Deep Koopman with stable generator.

    A = K - L^T L  where:
        K = -K^T              (skew-symmetric, drives rotation/oscillation)
        L^T L >= 0            (PSD, drives damping/contraction)

    => Re(eigenvalues of A) <= 0  => |eigenvalues of expm(A·dt)| <= 1

    When L -> 0, recovers the unitary (area-preserving) case.
    Params: n(n-1)/2 (skew) + n(n+1)/2 (lower-tri L) = n^2.
    """

    def __init__(self, autoencoder, latent_dim):
        super().__init__()
        self.ae = autoencoder
        self.latent_dim = latent_dim
        n = latent_dim

        # Skew-symmetric part K (rotation): upper triangle
        n_skew = n * (n - 1) // 2
        self.skew_params = nn.Parameter(0.01 * torch.randn(n_skew))

        # Lower triangular L (damping): L^T L is PSD
        self.L_diag = nn.Parameter(0.01 * torch.randn(n))
        n_lower = n * (n - 1) // 2
        self.L_lower = nn.Parameter(0.01 * torch.randn(n_lower))

    def _get_skew(self):
        n = self.latent_dim
        K = torch.zeros(n, n, device=self.skew_params.device)
        idx = torch.triu_indices(n, n, offset=1)
        K[idx[0], idx[1]] = self.skew_params
        return K - K.T

    def _get_L(self):
        n = self.latent_dim
        L = torch.zeros(n, n, device=self.L_diag.device)
        L[range(n), range(n)] = self.L_diag
        idx = torch.tril_indices(n, n, offset=-1)
        L[idx[0], idx[1]] = self.L_lower
        return L

    def generator_matrix(self):
        K = self._get_skew()
        L = self._get_L()
        return K - L.T @ L

    def damping_matrix(self):
        """L^T L — the PSD damping component."""
        L = self._get_L()
        return (L.T @ L).detach()

    def expm_propagate(self, z_t, dt):
        A = self.generator_matrix()
        M = torch.matrix_exp(A * dt)
        return z_t @ M.T

    def propagator_matrix(self, dt):
        A = self.generator_matrix()
        return torch.matrix_exp(A * dt).detach()

    def forward_one_step(self, x_t, dt):
        z_t, x_t_rec = self.ae(x_t)
        z_tp1_pred = self.expm_propagate(z_t, dt)
        x_tp1_pred = self.ae.decoder(z_tp1_pred)
        return {
            "z_t": z_t,
            "x_t_rec": x_t_rec,
            "z_tp1_pred": z_tp1_pred,
            "x_tp1_pred": x_tp1_pred,
        }


# =====================================================================
# Training
# =====================================================================

def train(ae, latent_dim, traj, epochs=150, lr=1e-3, device='cpu'):
    n_train = 48
    train_dl = DataLoader(PairDataset(traj[:n_train]),
                          batch_size=2048, shuffle=True, drop_last=True)
    val_dl = DataLoader(PairDataset(traj[n_train:]),
                        batch_size=2048)

    model = DeepKoopmanCTStable(ae, latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_skew = latent_dim * (latent_dim - 1) // 2
    n_damp = latent_dim + latent_dim * (latent_dim - 1) // 2
    print(f"  Stable Koopman [3,8,{latent_dim}]: {n_params:,} params "
          f"({n_skew} skew + {n_damp} damping = {n_skew + n_damp} generator)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    # Grid init.  PyKAN adapts its grid to the data; KANAutoEncoder (local
    # kan.py) fixes its grid at construction via grid_range_l1/l2 and has no
    # such method, so grid updates are skipped for that backbone.
    adaptive_grid = hasattr(ae.encoder, 'update_grid_from_samples')
    x_sample = traj[:n_train].reshape(-1, 3)[:2000].to(device)
    if adaptive_grid:
        with torch.no_grad():
            z_sample = ae.encoder(x_sample)
        ae.encoder.update_grid_from_samples(x_sample)
        ae.decoder.update_grid_from_samples(z_sample)
        print('  Grids initialized')

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x_t, x_tp1 in train_dl:
            x_t, x_tp1 = x_t.to(device), x_tp1.to(device)
            out = model.forward_one_step(x_t, DT)
            z_true = model.ae.encoder(x_tp1)
            loss = (mse(out['x_tp1_pred'], x_tp1)
                    + mse(out['z_tp1_pred'], z_true)
                    + 0.5 * mse(out['x_t_rec'], x_t))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        scheduler.step()

        # Grid re-init
        if adaptive_grid and ep % 20 == 0:
            with torch.no_grad():
                idx = torch.randperm(x_sample.shape[0])[:1000]
                ae.encoder.update_grid_from_samples(x_sample[idx])
                z_up = ae.encoder(x_sample[idx])
                ae.decoder.update_grid_from_samples(z_up)

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            model.eval()
            vloss = []
            with torch.no_grad():
                for x_t, x_tp1 in val_dl:
                    x_t, x_tp1 = x_t.to(device), x_tp1.to(device)
                    out = model.forward_one_step(x_t, DT)
                    z_true = model.ae.encoder(x_tp1)
                    l = (mse(out['x_tp1_pred'], x_tp1)
                         + mse(out['z_tp1_pred'], z_true)
                         + 0.5 * mse(out['x_t_rec'], x_t))
                    vloss.append(l.item())

            M = model.propagator_matrix(DT).cpu().numpy()
            eigs = np.linalg.eigvals(M)
            mag_range = f'[{np.abs(eigs).min():.4f}, {np.abs(eigs).max():.4f}]'
            print(f"  Ep {ep:3d} | train {np.mean(losses):.5f} | "
                  f"val {np.mean(vloss):.5f} | |λ| {mag_range}", flush=True)

    return model.cpu().eval()


# =====================================================================
# 1. PyKAN network diagrams
# =====================================================================

def plot_pykan_networks(model, x_data, latent_dim):
    print("1. PyKAN network diagrams...")
    z_labels = [f'$z_{{{k}}}$' for k in range(latent_dim)]

    with torch.no_grad():
        model.ae.encoder(x_data)
        z = model.ae.encoder(x_data)
        model.ae.decoder(z)

    model.ae.encoder.plot(
        beta=3,
        in_vars=[r'$x$', r'$y$', r'$z$'],
        out_vars=z_labels,
        title=r'Encoder:  $(x, y, z) \to \mathbf{z}$',
    )
    plt.savefig(OUTDIR / 'encoder.png', dpi=200, bbox_inches='tight')
    plt.savefig(OUTDIR / 'encoder.pdf', bbox_inches='tight')
    plt.close('all')

    model.ae.decoder.plot(
        beta=3,
        in_vars=z_labels,
        out_vars=[r'$\hat{x}$', r'$\hat{y}$', r'$\hat{z}$'],
        title=r'Decoder:  $\mathbf{z} \to (\hat{x}, \hat{y}, \hat{z})$',
    )
    plt.savefig(OUTDIR / 'decoder.png', dpi=200, bbox_inches='tight')
    plt.savefig(OUTDIR / 'decoder.pdf', bbox_inches='tight')
    plt.close('all')
    print("  Saved encoder/decoder .png/.pdf")


# =====================================================================
# 2-3. Rollout and NRMSE in Lyapunov times
# =====================================================================

def rollout_trajectory(model, x0, n_steps):
    """Autoregressive rollout from initial condition."""
    xs = [x0]
    x = x0
    with torch.no_grad():
        for _ in range(n_steps):
            out = model.forward_one_step(x.unsqueeze(0), DT)
            x = out['x_tp1_pred'].squeeze(0)
            xs.append(x)
    return torch.stack(xs)


def plot_rollout_and_nrmse(model, traj):
    print("2-3. Rollout and NRMSE...")

    val_traj = traj[48:]
    n_tau = 10
    steps_per_tau = int(TAU_L / DT)
    n_steps = n_tau * steps_per_tau

    all_pred, all_true = [], []
    with torch.no_grad():
        for i in range(val_traj.shape[0]):
            x = val_traj[i, 0]
            xs = [x]
            for _ in range(n_steps):
                out = model.forward_one_step(x.unsqueeze(0), DT)
                x = out['x_tp1_pred'].squeeze(0)
                xs.append(x)
            all_pred.append(torch.stack(xs))
            all_true.append(val_traj[i, :n_steps + 1])

    preds = torch.stack(all_pred)
    trues = torch.stack(all_true)
    t_tau = np.arange(n_steps + 1) * DT / TAU_L

    # NRMSE
    std_comp = trues.reshape(-1, 3).std(dim=0)
    rmse = (preds - trues).pow(2).mean(dim=0).sqrt()
    nrmse_per = rmse / std_comp.unsqueeze(0)
    nrmse = nrmse_per.mean(dim=1).numpy()
    nrmse_x = nrmse_per[:, 0].numpy()
    nrmse_y = nrmse_per[:, 1].numpy()
    nrmse_z = nrmse_per[:, 2].numpy()

    # --- Rollout figure ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    true_np = trues[0].numpy()
    pred_np = preds[0].detach().numpy()
    for i, (ax, lab, ct, cp) in enumerate(zip(
            axes, ['x', 'y', 'z'],
            ['#1f77b4', '#2ca02c', '#ff7f0e'],
            ['#d62728', '#9467bd', '#8c564b'])):
        ax.plot(t_tau, true_np[:, i], color=ct, lw=1, alpha=0.8, label=f'True ${lab}$')
        ax.plot(t_tau, pred_np[:, i], color=cp, lw=1, alpha=0.8, ls='--',
                label=f'Predicted ${lab}$')
        ax.set_ylabel(f'${lab}(t)$', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)
        for tm in range(1, n_tau + 1):
            ax.axvline(tm, color='gray', ls=':', alpha=0.3, lw=0.5)
    axes[-1].set_xlabel(r'Time [$\tau_\Lambda$]', fontsize=12)
    fig.suptitle(r'Stable Koopman Rollout ($\tau_\Lambda \approx 1.1$ s)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'rollout.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'rollout.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved rollout.png/pdf")

    # --- NRMSE figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(t_tau, nrmse, 'k-', lw=2, label='NRMSE (all)')
    ax.plot(t_tau, nrmse_x, '-', color='#1f77b4', lw=1, alpha=0.7, label='$x$')
    ax.plot(t_tau, nrmse_y, '-', color='#2ca02c', lw=1, alpha=0.7, label='$y$')
    ax.plot(t_tau, nrmse_z, '-', color='#ff7f0e', lw=1, alpha=0.7, label='$z$')
    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5, label='NRMSE = 1')
    ax.fill_between(t_tau, 0, nrmse, where=nrmse < 1.0, color='green', alpha=0.08)
    cross = np.where(nrmse > 1.0)[0]
    if len(cross) > 0:
        tc = t_tau[cross[0]]
        ax.annotate(f'NRMSE=1 at {tc:.1f} $\\tau_\\Lambda$',
                    xy=(tc, 1.0), xytext=(tc + 0.5, 1.3),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    ax.set_xlabel(r'Time [$\tau_\Lambda$]', fontsize=12)
    ax.set_ylabel('NRMSE', fontsize=12)
    ax.set_title('Normalized RMSE', fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim(0, n_tau)
    ax.set_ylim(0, min(2.0, nrmse.max() * 1.1))
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.semilogy(t_tau, nrmse, 'k-', lw=2)
    ax.semilogy(t_tau, nrmse_x, '-', color='#1f77b4', lw=1, alpha=0.7)
    ax.semilogy(t_tau, nrmse_y, '-', color='#2ca02c', lw=1, alpha=0.7)
    ax.semilogy(t_tau, nrmse_z, '-', color='#ff7f0e', lw=1, alpha=0.7)
    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel(r'Time [$\tau_\Lambda$]', fontsize=12)
    ax.set_ylabel('NRMSE (log)', fontsize=12)
    ax.set_title('NRMSE Growth', fontsize=12)
    ax.set_xlim(0, n_tau)
    ax.grid(True, alpha=0.2)

    horizon = t_tau[cross[0]] if len(cross) > 0 else float('inf')
    fig.suptitle(f'Prediction Horizon: {horizon:.1f} $\\tau_\\Lambda$ '
                 f'({horizon * TAU_L:.1f} s)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'nrmse.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'nrmse.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved nrmse.png/pdf (horizon: {horizon:.1f} tau_L)")

    return preds, trues


# =====================================================================
# 4. Manifold decomposition summary
# =====================================================================

def run_decompositions(model, x_data, latent_dim):
    print("4. Manifold decompositions...")
    decomps = []
    for ti in range(latent_dim):
        dr = full_decomposition(model, x_data, ti, target_type='latent',
                                max_degree=3, lasso_alpha=1e-5, h_degree=4)
        decomps.append(dr)
    return decomps


def plot_decomposition_summary(decomps, latent_dim):
    r2s = [d['recon_r2'] for d in decomps]
    r2g = [d['g_recon_r2'] for d in decomps]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(latent_dim), r2s, color='steelblue', alpha=0.8)
    axes[0].axhline(0.9, color='red', ls='--', alpha=0.5)
    axes[0].set_xlabel('Latent dim')
    axes[0].set_ylabel(r'R$^2$(h$\circ$g)')
    axes[0].set_title('Decomposition Quality')

    xy_coeffs, xz_coeffs = [], []
    for d in decomps:
        cm = dict(zip(d['inner']['active_names'], d['inner']['active_coeffs']))
        xy_coeffs.append(abs(cm.get('xy', 0)))
        xz_coeffs.append(abs(cm.get('xz', 0)))

    axes[1].bar(np.arange(latent_dim) - 0.2, xy_coeffs, 0.35,
                label='|xy coeff|', color='#9C27B0', alpha=0.8)
    axes[1].bar(np.arange(latent_dim) + 0.2, xz_coeffs, 0.35,
                label='|xz coeff|', color='#F44336', alpha=0.8)
    axes[1].set_xlabel('Latent dim')
    axes[1].set_ylabel('|coefficient|')
    axes[1].set_title('Cross-Term Recovery')
    axes[1].legend()

    fig.suptitle(f'Lorenz Dictionary Recovery (Stable PyKAN [3,8,{latent_dim}])',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'decomposition_summary.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'decomposition_summary.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved decomposition_summary.png/pdf")


# =====================================================================
# 5. Layer-1 spline function identification
# =====================================================================

def identify_spline_functions(model, x_data):
    print("5. Layer-1 spline identification...")

    enc = model.ae.encoder
    x_np = x_data.numpy()
    var_names = ['x', 'y', 'z']

    candidates = {
        'linear': (lambda x, a, b: a * x + b, 2),
        'quadratic': (lambda x, a, b, c: a * x**2 + b * x + c, 3),
        'cubic': (lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, 4),
        'sin': (lambda x, a, b, c, d: a * np.sin(b * x + c) + d, 4),
        'cos': (lambda x, a, b, c, d: a * np.cos(b * x + c) + d, 4),
        'tanh': (lambda x, a, b, c, d: a * np.tanh(b * x + c) + d, 4),
        'exp': (lambda x, a, b, c: a * np.exp(b * x) + c, 3),
        'abs': (lambda x, a, b, c: a * np.abs(x - b) + c, 3),
    }

    mid_size = 8
    fig, axes = plt.subplots(mid_size, 3, figsize=(15, 3 * mid_size))
    results = {}

    for i, vn in enumerate(var_names):
        xi_range = np.linspace(x_np[:, i].min(), x_np[:, i].max(), 500)

        for j in range(mid_size):
            ax = axes[j, i]

            x_probe = torch.zeros(500, 3)
            for k in range(3):
                x_probe[:, k] = float(x_np[:, k].mean())
            x_probe[:, i] = torch.tensor(xi_range, dtype=torch.float32)

            with torch.no_grad():
                enc(x_probe)
                if hasattr(enc, 'acts') and len(enc.acts) > 1:
                    l1_out = enc.acts[1].numpy()
                    edge_out = l1_out[:, j]
                else:
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
                    continue

            edge_range = edge_out.max() - edge_out.min()
            if edge_range < 0.01:
                ax.text(0.5, 0.5, 'dead', transform=ax.transAxes,
                        ha='center', fontsize=8, color='gray')
                results[(j, i)] = ('dead', 0, {})
                continue

            best_name, best_r2, best_params, best_pred = 'unknown', -999, {}, None

            for cname, (cfn, nparams) in candidates.items():
                try:
                    p0 = [1.0] * nparams
                    popt, _ = curve_fit(cfn, xi_range, edge_out, p0=p0,
                                        maxfev=5000)
                    pred = cfn(xi_range, *popt)
                    ss_res = np.sum((edge_out - pred) ** 2)
                    ss_tot = np.sum((edge_out - edge_out.mean()) ** 2)
                    r2 = 1 - ss_res / max(ss_tot, 1e-12)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_name = cname
                        best_params = dict(zip(['a', 'b', 'c', 'd'][:nparams], popt))
                        best_pred = pred
                except Exception:
                    continue

            results[(j, i)] = (best_name, best_r2, best_params)

            ax.plot(xi_range, edge_out, 'b-', lw=1.5, alpha=0.8)
            if best_r2 > 0.5 and best_pred is not None:
                ax.plot(xi_range, best_pred, 'r--', lw=1, alpha=0.7)

            label = f'{best_name} R²={best_r2:.3f}'
            if best_name in ('sin', 'cos') and best_r2 > 0.9:
                b_val = best_params.get('b', 0)
                label = f'{best_name}({b_val:.2f}·{vn}) R²={best_r2:.3f}'
            elif best_name == 'linear' and best_r2 > 0.9:
                a_val = best_params.get('a', 0)
                label = f'{a_val:.3f}·{vn} R²={best_r2:.3f}'

            ax.set_title(f'psi({vn}->{j}): {label}', fontsize=8)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_title(f'Input: ${vn}$\npsi({vn}->{j}): {label}', fontsize=8)

    fig.suptitle('KAN Layer 1: Learned Univariate Splines (Stable Model)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'spline_identification.png', dpi=150, bbox_inches='tight')
    fig.savefig(OUTDIR / 'spline_identification.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved spline_identification.png/pdf")

    print("\n  Layer 1 spline summary:")
    print(f"  {'Edge':>10}  {'Function':>10}  {'R2':>6}  Key params")
    print("  " + "-" * 50)
    for j in range(mid_size):
        for i, vn in enumerate(var_names):
            fname, r2, params = results.get((j, i), ('?', 0, {}))
            if fname == 'dead':
                continue
            p_str = ', '.join(f'{k}={v:.3f}' for k, v in list(params.items())[:2])
            print(f"  psi({vn}->{j})  {fname:>10}  {r2:6.3f}  {p_str}")

    return results


# =====================================================================
# 6. Level-set visualizations
# =====================================================================

def plot_level_sets(decomps, x_np, targets=None):
    print("6. Level-set visualizations...")

    if targets is None:
        ranked = sorted(range(len(decomps)), key=lambda i: -decomps[i]['recon_r2'])
        targets = ranked[:4]

    for ti in targets:
        dr = decomps[ti]
        inner = dr['inner']
        ideal = dr['ideal']
        f_vals = dr['f_vals']
        g_vals = dr['g_vals']
        f_hat = dr['f_hat']
        outer = dr['outer']

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        sub = np.random.choice(len(f_vals), min(3000, len(f_vals)), replace=False)

        # Row 1: Lasso coefficients, f vs g, h'(g), reconstruction
        ax = axes[0, 0]
        anames = inner['active_names']
        acoeffs = inner['active_coeffs']
        if len(anames) > 0:
            order = np.argsort(np.abs(acoeffs))[::-1][:12]
            ns = [anames[i] for i in order]
            cs = [acoeffs[i] for i in order]
            colors = ['#d62728' if c < 0 else '#1f77b4' for c in cs]
            ax.barh(range(len(ns)), cs, color=colors, edgecolor='k', lw=0.5)
            ax.set_yticks(range(len(ns)))
            ax.set_yticklabels(ns, fontsize=8)
            ax.invert_yaxis()
        ax.set_xlabel('Coefficient')
        ax.set_title(f'z[{ti}] Ideal Generators\nR²(g)={inner["r2"]:.4f}', fontsize=10)

        ax = axes[0, 1]
        ax.scatter(g_vals[sub], f_vals[sub], s=1, alpha=0.2, rasterized=True)
        ax.set_xlabel('g(x,y,z)')
        ax.set_ylabel(f'z[{ti}]')
        ax.set_title('f vs g')

        ax = axes[0, 2]
        gs = outer['g_samples']
        hp = outer['h_prime_samples']
        sub2 = np.random.choice(len(gs), min(3000, len(gs)), replace=False)
        ax.scatter(gs[sub2], hp[sub2], s=1, alpha=0.15, c='steelblue', rasterized=True)
        if 'h_bin_centers' in outer and len(outer['h_bin_centers']) > 0:
            ax.plot(outer['h_bin_centers'] + outer['g_center'],
                    outer['h_bin_medians'], 'ko', ms=3)
        ax.plot(outer['g_plot'], outer['h_prime_fit'], 'r-', lw=2)
        ax.set_xlabel('g')
        ax.set_ylabel("h'(g)")
        ax.set_title("Outer Derivative h'(g)")

        ax = axes[0, 3]
        ax.scatter(f_hat[sub], f_vals[sub], s=1, alpha=0.2, rasterized=True)
        lim = [min(f_hat.min(), f_vals.min()), max(f_hat.max(), f_vals.max())]
        ax.plot(lim, lim, 'r--', lw=1)
        ax.set_xlabel('h(g(x))')
        ax.set_ylabel(f'z[{ti}]')
        tag = ' [affine]' if outer.get('used_affine') else ''
        ax.set_title(f'R²(h∘g)={dr["recon_r2"]:.4f}{tag}')

        # Row 2: Lorenz colored by f, g, level-set slices, residual
        for col, (title, vals, cmap) in enumerate([
            (f'z[{ti}] (network)', f_vals, 'coolwarm'),
            ('g(x,y,z)', g_vals, 'coolwarm'),
            ('Level-set slices', None, None),
            ('residual f - h(g)', f_vals - f_hat, 'RdBu_r'),
        ]):
            ax = axes[1, col]
            if col == 2:
                ax.scatter(x_np[sub, 0], x_np[sub, 1], c='lightgray',
                           s=0.5, alpha=0.1, rasterized=True)
                g_min, g_max = np.percentile(g_vals, 10), np.percentile(g_vals, 90)
                levels = np.linspace(g_min, g_max, 7)[1:-1]
                cmap_lev = plt.cm.viridis
                for k, lev in enumerate(levels):
                    tol = (g_max - g_min) * 0.03
                    mask = np.abs(g_vals - lev) < tol
                    if mask.sum() > 10:
                        ax.scatter(x_np[mask, 0], x_np[mask, 1],
                                   c=[cmap_lev(k / len(levels))],
                                   s=4, alpha=0.6, label=f'g={lev:.1f}',
                                   rasterized=True)
                ax.set_title('Level-set slices', fontsize=10)
                ax.legend(fontsize=5, markerscale=3)
            else:
                sc = ax.scatter(x_np[sub, 0], x_np[sub, 1], c=vals[sub],
                                s=2, alpha=0.5, cmap=cmap, rasterized=True)
                ax.set_title(title, fontsize=10)
                plt.colorbar(sc, ax=ax, shrink=0.8)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')

        expr = ideal['expression'][:50]
        fig.suptitle(f'z[{ti}]:  f = h(g)   |   g = {expr}', fontsize=12)
        fig.tight_layout()
        fig.savefig(OUTDIR / f'levelset_z{ti}.png', dpi=150, bbox_inches='tight')
        fig.savefig(OUTDIR / f'levelset_z{ti}.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved levelset_z{ti}.png/pdf")


# =====================================================================
# 7. 3D attractor overlay (true vs predicted)
# =====================================================================

def plot_attractor_overlay(model, traj):
    print("7. 3D attractor overlay...")

    val_traj = traj[48:]

    # Pick a trajectory and rollout for several Lyapunov times
    n_tau_short = 4  # short horizon where prediction is still good
    steps_short = int(n_tau_short * TAU_L / DT)

    # Also do a longer rollout to show divergence
    n_tau_long = 8
    steps_long = int(n_tau_long * TAU_L / DT)

    fig = plt.figure(figsize=(18, 7))

    # --- Panel 1: short-horizon overlay (accurate region) ---
    ax1 = fig.add_subplot(131, projection='3d')
    true_short = val_traj[0, :steps_short + 1].numpy()
    pred_short = rollout_trajectory(model, val_traj[0, 0], steps_short).numpy()

    ax1.plot(true_short[:, 0], true_short[:, 1], true_short[:, 2],
             color='#1f77b4', lw=0.8, alpha=0.8, label='True')
    ax1.plot(pred_short[:, 0], pred_short[:, 1], pred_short[:, 2],
             color='#d62728', lw=0.8, alpha=0.8, ls='--', label='Predicted')
    ax1.set_xlabel('$x$'); ax1.set_ylabel('$y$'); ax1.set_zlabel('$z$')
    ax1.set_title(f'Short horizon ({n_tau_short} $\\tau_\\Lambda$)', fontsize=11)
    ax1.legend(fontsize=8, loc='upper left')

    # --- Panel 2: long-horizon overlay ---
    ax2 = fig.add_subplot(132, projection='3d')
    true_long = val_traj[0, :steps_long + 1].numpy()
    pred_long = rollout_trajectory(model, val_traj[0, 0], steps_long).numpy()

    ax2.plot(true_long[:, 0], true_long[:, 1], true_long[:, 2],
             color='#1f77b4', lw=0.5, alpha=0.5, label='True')
    ax2.plot(pred_long[:, 0], pred_long[:, 1], pred_long[:, 2],
             color='#d62728', lw=0.5, alpha=0.5, ls='--', label='Predicted')
    ax2.set_xlabel('$x$'); ax2.set_ylabel('$y$'); ax2.set_zlabel('$z$')
    ax2.set_title(f'Long horizon ({n_tau_long} $\\tau_\\Lambda$)', fontsize=11)
    ax2.legend(fontsize=8, loc='upper left')

    # --- Panel 3: attractor from many rollouts ---
    ax3 = fig.add_subplot(133, projection='3d')

    # Plot true attractor (gray background)
    true_all = val_traj.reshape(-1, 3).numpy()
    sub = np.random.choice(len(true_all), min(20000, len(true_all)), replace=False)
    ax3.scatter(true_all[sub, 0], true_all[sub, 1], true_all[sub, 2],
                s=0.3, alpha=0.05, c='gray', rasterized=True, label='True attractor')

    # Overlay several predicted trajectories
    colors = ['#d62728', '#ff7f0e', '#9467bd', '#2ca02c']
    for ic in range(min(4, val_traj.shape[0])):
        pred_ic = rollout_trajectory(model, val_traj[ic, 0], steps_long).numpy()
        ax3.plot(pred_ic[:, 0], pred_ic[:, 1], pred_ic[:, 2],
                 color=colors[ic % len(colors)], lw=0.4, alpha=0.6)
    ax3.set_xlabel('$x$'); ax3.set_ylabel('$y$'); ax3.set_zlabel('$z$')
    ax3.set_title('Predicted on attractor', fontsize=11)

    fig.suptitle('Lorenz Attractor: Stable Koopman Predictions', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'attractor_overlay.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'attractor_overlay.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved attractor_overlay.png/pdf")


# =====================================================================
# 8. Eigenvalue spectrum
# =====================================================================

def plot_eigenvalues(model):
    print("8. Eigenvalue spectrum...")

    A = model.generator_matrix().detach().cpu().numpy()
    M = model.propagator_matrix(DT).cpu().numpy()
    eigs_M = np.linalg.eigvals(M)
    eigs_A = np.linalg.eigvals(A)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Discrete eigenvalues (of M = expm(A*dt))
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.3)
    ax.scatter(eigs_M.real, eigs_M.imag, s=50, c='steelblue',
               edgecolors='black', linewidths=0.5, zorder=3)
    ax.set_xlabel(r'Re($\lambda$)'); ax.set_ylabel(r'Im($\lambda$)')
    ax.set_title(f'Discrete Eigenvalues expm(A dt)\n'
                 f'|$\\lambda$| range: [{np.abs(eigs_M).min():.4f}, '
                 f'{np.abs(eigs_M).max():.4f}]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Generator eigenvalues (of A = K - L^T L, should have Re(mu) <= 0)
    ax = axes[1]
    ax.axvline(0, color='red', ls='--', lw=1, alpha=0.5, label='Re=0 (stability boundary)')
    ax.scatter(eigs_A.real, eigs_A.imag, s=50, c='#F44336',
               edgecolors='black', linewidths=0.5, zorder=3)
    ax.set_xlabel(r'Re($\mu$)'); ax.set_ylabel(r'Im($\mu$)')
    max_re = np.max(eigs_A.real)
    ax.set_title(f'Generator Eigenvalues A = K - L$^T$L\n'
                 f'max Re($\\mu$) = {max_re:.4f} (should be $\\leq$ 0)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Stable Koopman Spectrum (Lorenz)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'eigenvalues.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'eigenvalues.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved eigenvalues.png/pdf")
    print(f"  |lambda| range: [{np.abs(eigs_M).min():.6f}, "
          f"{np.abs(eigs_M).max():.6f}]")
    print(f"  All |lambda| <= 1? "
          f"{np.all(np.abs(eigs_M) <= 1.0 + 1e-6)}")
    print(f"  Generator Re(mu) range: [{eigs_A.real.min():.6f}, "
          f"{eigs_A.real.max():.6f}]")

    # Damping analysis
    D = model.damping_matrix().cpu().numpy()
    damp_eigs = np.linalg.eigvalsh(D)
    print(f"  Damping L^T L eigenvalues: min={damp_eigs.min():.6f}, "
          f"max={damp_eigs.max():.6f}")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mid', type=int, default=8)
    parser.add_argument('--latent', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-train', action='store_true',
                        help='Load existing checkpoint instead of training')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(0)

    print(f"Lorenz Stable Koopman (PyKAN [3,{args.mid},{args.latent}])")
    print(f"Generator: A = K - L^T L  (skew + PSD damping)")
    print(f"Device: {args.device}")

    # Simulate data
    print("\nGenerating Lorenz data...")
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=300,
                           device=args.device)
    traj_cpu = traj.cpu()

    ckpt_path = OUTDIR / f'stable_{args.mid}_{args.latent}_model.pt'

    if args.skip_train and ckpt_path.exists():
        print(f"\nLoading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        ae = PyKANAutoEncoder(3, cfg['mid'], cfg['latent'], grid=5, k=3)
        model = DeepKoopmanCTStable(ae, cfg['latent'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
    else:
        print(f"\nTraining...")
        ae = PyKANAutoEncoder(3, args.mid, args.latent, grid=5, k=3).to(args.device)
        model = train(ae, args.latent, traj, epochs=args.epochs, device=args.device)

        torch.save({
            'model_state_dict': model.state_dict(),
            'dt': DT,
            'config': {'mid': args.mid, 'latent': args.latent},
        }, ckpt_path)
        print(f"  Saved {ckpt_path}")

    # Prepare analysis data
    flat = traj_cpu.reshape(-1, 3)
    idx = torch.randperm(flat.shape[0])[:5000]
    x_data = flat[idx]
    x_np = x_data.numpy()

    # 1. Network diagrams
    plot_pykan_networks(model, x_data, args.latent)

    # 2-3. Rollout and NRMSE
    preds, trues = plot_rollout_and_nrmse(model, traj_cpu)

    # 4. Decompositions
    decomps = run_decompositions(model, x_data, args.latent)
    plot_decomposition_summary(decomps, args.latent)

    # Print dictionary table
    print("\n  Dictionary recovery table:")
    print(f"  {'Dim':>4}  {'R2(g)':>8}  {'R2(hog)':>8}  g (top terms)")
    print("  " + "-" * 60)
    for ti, dr in enumerate(decomps):
        if dr['recon_r2'] > 0.95:
            pairs = sorted(zip(dr['inner']['active_names'],
                               dr['inner']['active_coeffs']),
                           key=lambda x: -abs(x[1]))[:5]
            expr = '  '.join(f'{c:+.3f}*{n}' for n, c in pairs)
            print(f"  z[{ti:2d}]  {dr['g_recon_r2']:8.3f}  "
                  f"{dr['recon_r2']:8.3f}  {expr}")

    # 5. Spline identification
    spline_results = identify_spline_functions(model, x_data)

    # 6. Level sets for best dims
    ranked = sorted(range(args.latent),
                    key=lambda i: -decomps[i]['recon_r2'])
    plot_level_sets(decomps, x_np, targets=ranked[:4])

    # 7. 3D attractor overlay
    plot_attractor_overlay(model, traj_cpu)

    # 8. Eigenvalues
    plot_eigenvalues(model)

    print(f"\nAll results in {OUTDIR}/")
