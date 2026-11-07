"""
Complete Lorenz results for PyKAN [3,8,32] Deep Koopman model.

Generates all paper figures:
    1. PyKAN model.plot() for encoder and decoder
    2. Rollout vs truth
    3. NRMSE in Lyapunov times
    4. Manifold decomposition for all latent dims
    5. Layer-1 spline function identification (sin, cos, tanh, poly, etc.)
    6. Level-set visualizations for best dims
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

from models import PyKANAutoEncoder, DeepKoopmanCT
from data import simulate_lorenz
from training import set_seed
from manifold_analysis import full_decomposition

OUTDIR = Path('results/lorenz_pykan')
LAMBDA_MAX = 0.9056  # Lorenz largest Lyapunov exponent (/s)
DT = 0.01
TAU_L = 1.0 / LAMBDA_MAX


def load_model():
    ckpt = torch.load('old/pykan_3_8_32_model.pt', map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    ae = PyKANAutoEncoder(3, cfg['mid_size'], cfg['latent_dim'], grid=5, k=3)
    model = DeepKoopmanCT(ae, cfg['latent_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


# =====================================================================
# 1. PyKAN model.plot() for encoder and decoder
# =====================================================================

def plot_pykan_networks(model, x_data):
    print("1. PyKAN network diagrams...")

    z_labels = [f'$z_{{{k}}}$' for k in range(32)]

    # Need a forward pass for activation stats
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
    print("  Saved encoder.png/pdf")

    model.ae.decoder.plot(
        beta=3,
        in_vars=z_labels,
        out_vars=[r'$\hat{x}$', r'$\hat{y}$', r'$\hat{z}$'],
        title=r'Decoder:  $\mathbf{z} \to (\hat{x}, \hat{y}, \hat{z})$',
    )
    plt.savefig(OUTDIR / 'decoder.png', dpi=200, bbox_inches='tight')
    plt.savefig(OUTDIR / 'decoder.pdf', bbox_inches='tight')
    plt.close('all')
    print("  Saved decoder.png/pdf")


# =====================================================================
# 2-3. Rollout and NRMSE in Lyapunov times
# =====================================================================

def plot_rollout_and_nrmse(model, traj):
    print("2-3. Rollout and NRMSE...")

    val_traj = traj[48:]
    n_tau = 10
    steps_per_tau = int(TAU_L / DT)
    n_steps = n_tau * steps_per_tau

    # Multi-trajectory rollout
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
    rmse = (preds - trues).pow(2).mean(dim=0).sqrt()  # (T, 3)
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
    fig.suptitle(r'Koopman Rollout ($\tau_\Lambda \approx 1.1$ s)', fontsize=13)
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
    print(f"  Saved nrmse.png/pdf (horizon: {horizon:.1f} τ_Λ)")


# =====================================================================
# 4. Manifold decomposition summary
# =====================================================================

def run_decompositions(model, x_data):
    print("4. Manifold decompositions...")
    decomps = []
    for ti in range(32):
        dr = full_decomposition(model, x_data, ti, target_type='latent',
                                max_degree=3, lasso_alpha=1e-5, h_degree=4)
        decomps.append(dr)
    return decomps


def plot_decomposition_summary(decomps):
    latent_dim = len(decomps)
    r2s = [d['recon_r2'] for d in decomps]
    r2g = [d['g_recon_r2'] for d in decomps]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(latent_dim), r2s, color='steelblue', alpha=0.8)
    axes[0].axhline(0.9, color='red', ls='--', alpha=0.5)
    axes[0].set_xlabel('Latent dim')
    axes[0].set_ylabel('R²(h∘g)')
    axes[0].set_title('Decomposition Quality')

    # Cross-term presence
    xy_coeffs = []
    xz_coeffs = []
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

    fig.suptitle('Lorenz Dictionary Recovery (PyKAN [3,8,32])', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'decomposition_summary.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'decomposition_summary.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved decomposition_summary.png/pdf")


# =====================================================================
# 5. Layer-1 spline function identification
# =====================================================================

def identify_spline_functions(model, x_data):
    """Sweep each input through layer 1 and fit candidate functions."""
    print("5. Layer-1 spline identification...")

    enc = model.ae.encoder
    x_np = x_data.numpy()
    var_names = ['x', 'y', 'z']

    # Candidate functions for fitting
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

    # For PyKAN, we evaluate the full encoder layer by sweeping one input
    # We need to get the intermediate output after layer 0 (first KAN layer)
    # PyKAN's KAN stores activations internally after forward pass
    mid_size = 8

    fig, axes = plt.subplots(mid_size, 3, figsize=(15, 3 * mid_size))
    results = {}

    for i, vn in enumerate(var_names):
        xi_range = np.linspace(x_np[:, i].min(), x_np[:, i].max(), 500)

        for j in range(mid_size):
            ax = axes[j, i]

            # Build probe: sweep input i, others at mean
            x_probe = torch.zeros(500, 3)
            for k in range(3):
                x_probe[:, k] = float(x_np[:, k].mean())
            x_probe[:, i] = torch.tensor(xi_range, dtype=torch.float32)

            # Get layer-1 output
            with torch.no_grad():
                enc(x_probe)  # full forward to populate internals
                # PyKAN stores postactivations; get layer 0 output
                if hasattr(enc, 'acts') and len(enc.acts) > 1:
                    l1_out = enc.acts[1].numpy()  # (500, mid_size)
                    edge_out = l1_out[:, j]
                else:
                    # Fallback: can't get intermediate — skip
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
                    continue

            # Skip dead edges
            edge_range = edge_out.max() - edge_out.min()
            if edge_range < 0.01:
                ax.text(0.5, 0.5, 'dead', transform=ax.transAxes,
                        ha='center', fontsize=8, color='gray')
                results[(j, i)] = ('dead', 0, {})
                continue

            # Fit candidates
            best_name = 'unknown'
            best_r2 = -999
            best_params = {}

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
                except:
                    continue

            results[(j, i)] = (best_name, best_r2, best_params)

            # Plot
            ax.plot(xi_range, edge_out, 'b-', lw=1.5, alpha=0.8)
            if best_r2 > 0.5:
                ax.plot(xi_range, best_pred, 'r--', lw=1, alpha=0.7)

            label = f'{best_name} R²={best_r2:.3f}'
            if best_name in ('sin', 'cos') and best_r2 > 0.9:
                b_val = best_params.get('b', 0)
                label = f'{best_name}({b_val:.2f}·{vn}) R²={best_r2:.3f}'
            elif best_name == 'linear' and best_r2 > 0.9:
                a_val = best_params.get('a', 0)
                label = f'{a_val:.3f}·{vn} R²={best_r2:.3f}'

            ax.set_title(f'ψ({vn}→{j}): {label}', fontsize=8)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_title(f'Input: ${vn}$\nψ({vn}→{j}): {label}', fontsize=8)

    fig.suptitle('KAN Layer 1: Learned Univariate Splines with Function ID', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'spline_identification.png', dpi=150, bbox_inches='tight')
    fig.savefig(OUTDIR / 'spline_identification.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved spline_identification.png/pdf")

    # Print summary
    print("\n  Layer 1 spline summary:")
    print(f"  {'Edge':>10}  {'Function':>10}  {'R²':>6}  Key params")
    print("  " + "-" * 50)
    for j in range(mid_size):
        for i, vn in enumerate(var_names):
            fname, r2, params = results.get((j, i), ('?', 0, {}))
            if fname == 'dead':
                continue
            p_str = ', '.join(f'{k}={v:.3f}' for k, v in list(params.items())[:2])
            print(f"  ψ({vn}→{j})  {fname:>10}  {r2:6.3f}  {p_str}")

    return results


# =====================================================================
# 6. Level-set visualizations for best dims
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
            ('h(g) reconstruction', f_hat, 'coolwarm'),
            ('residual f−h(g)', f_vals - f_hat, 'RdBu_r'),
        ]):
            ax = axes[1, col]
            if col == 2:
                # Level-set slices instead
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
# Main
# =====================================================================

if __name__ == '__main__':
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    print("Loading PyKAN [3,8,32] model...")
    model, cfg = load_model()

    print("Generating Lorenz data...")
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=300)
    flat = traj.reshape(-1, 3)
    idx = torch.randperm(flat.shape[0])[:5000]
    x_data = flat[idx]
    x_np = x_data.numpy()

    # 1. Network diagrams
    plot_pykan_networks(model, x_data)

    # 2-3. Rollout and NRMSE
    plot_rollout_and_nrmse(model, traj)

    # 4. Decompositions
    decomps = run_decompositions(model, x_data)
    plot_decomposition_summary(decomps)

    # Print dictionary table
    print("\n  Dictionary recovery table:")
    print(f"  {'Dim':>4}  {'R²(g)':>8}  {'R²(h∘g)':>8}  g (top terms)")
    print("  " + "-" * 60)
    for ti, dr in enumerate(decomps):
        if dr['recon_r2'] > 0.95:
            pairs = sorted(zip(dr['inner']['active_names'],
                               dr['inner']['active_coeffs']),
                           key=lambda x: -abs(x[1]))[:5]
            expr = '  '.join(f'{c:+.3f}·{n}' for n, c in pairs)
            print(f"  z[{ti:2d}]  {dr['g_recon_r2']:8.3f}  {dr['recon_r2']:8.3f}  {expr}")

    # 5. Spline identification
    spline_results = identify_spline_functions(model, x_data)

    # 6. Level sets for best dims (top 4 by R²)
    ranked = sorted(range(32), key=lambda i: -decomps[i]['recon_r2'])
    plot_level_sets(decomps, x_np, targets=ranked[:4])

    print(f"\nAll results in {OUTDIR}/")
