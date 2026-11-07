"""
Standard Map: dictionary recovery and manifold visualization.

Uses the [16,64] trained model. The encoder maps (x,y) → (cos x, sin x, cos y, sin y)
internally, so polynomial decomposition on those 4 trig variables gives Fourier modes.

Degree-2 polynomials in {cx, sx, cy, sy} = all Fourier harmonics up to |m|+|n| ≤ 2.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from training import set_seed
from manifold_analysis import (
    build_polynomial_dict_nd, recover_inner_function_nd,
    grad_polynomial_nd, eval_polynomial_nd,
)
from dictionary_viability import gradient_alignment, orthogonal_residual

# Import the model from standard_map.py
from standard_map import DiscreteKoopmanKAN, simulate_standard_map

OUTDIR = Path('results/standard_map')


def load_model():
    ckpt = torch.load('old/standard_map_16_64.pt', map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = DiscreteKoopmanKAN(cfg['mid'], cfg['latent'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def trig_embed(xy):
    """Map (x, y) → (cos x, sin x, cos y, sin y)."""
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([np.cos(x), np.sin(x), np.cos(y), np.sin(y)])


def fourier_name(powers):
    """Convert polynomial powers in (cx, sx, cy, sy) to Fourier mode label."""
    pcx, psx, pcy, psy = powers
    # Products of cos/sin give Fourier modes via product-to-sum identities
    # For readability, just show the monomial
    parts = []
    for p, name in zip(powers, ['cx', 'sx', 'cy', 'sy']):
        if p == 1:
            parts.append(name)
        elif p > 1:
            parts.append(f'{name}^{p}')
    return '·'.join(parts) if parts else '1'


def compute_encoder_gradient(model, xy_tensor, target_idx):
    """Compute gradient of latent dim w.r.t. trig-embedded inputs."""
    N = xy_tensor.shape[0]
    chunk = 2000
    grad_list, f_list = [], []

    for i in range(0, N, chunk):
        xi = xy_tensor[i:i+chunk].detach().clone().requires_grad_(True)
        fi = model.encoder(xi)[:, target_idx]
        fi.sum().backward()
        # Gradient w.r.t. (x, y) — but we want w.r.t. trig embedding
        # So compute via chain rule: d/d(cx,sx,cy,sy) from d/d(x,y)
        grad_xy = xi.grad.detach().numpy()  # (chunk, 2)

        # Jacobian of trig embedding: d(cx,sx,cy,sy)/d(x,y)
        x_np = xi[:, 0].detach().numpy()
        y_np = xi[:, 1].detach().numpy()
        # dcx/dx = -sin(x), dsx/dx = cos(x), etc.
        # grad_trig = grad_xy @ J^{-T} where J = d(trig)/d(xy)
        # But J is 4x2, not invertible. Instead compute grad_f w.r.t trig directly.
        grad_list.append(grad_xy)
        f_list.append(fi.detach().numpy())

    return np.concatenate(grad_list), np.concatenate(f_list)


def compute_trig_gradient(model, xy_tensor, target_idx):
    """Compute gradient of latent dim w.r.t. (cx, sx, cy, sy) via autograd."""
    N = xy_tensor.shape[0]
    chunk = 2000
    grad_list, f_list = [], []

    for i in range(0, N, chunk):
        xy_i = xy_tensor[i:i+chunk].detach()
        x, y = xy_i[:, 0], xy_i[:, 1]
        # Build trig embedding as differentiable tensor
        trig = torch.stack([torch.cos(x), torch.sin(x),
                            torch.cos(y), torch.sin(y)], dim=-1).requires_grad_(True)
        # Forward through KAN layers directly (bypass the trig embedding in encoder)
        z = model.encoder.kan2(model.encoder.kan1(trig))
        fi = z[:, target_idx]
        fi.sum().backward()
        grad_list.append(trig.grad.detach().numpy())
        f_list.append(fi.detach().numpy())

    return np.concatenate(grad_list), np.concatenate(f_list)


def analyze_and_plot(model, traj, K):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    latent_dim = model.latent_dim
    xy_np = traj.reshape(-1, 2).numpy()

    # Subsample
    idx = np.random.choice(len(xy_np), min(8000, len(xy_np)), replace=False)
    xy_sub = xy_np[idx]
    xy_tensor = torch.tensor(xy_sub, dtype=torch.float32)

    # Trig-embed the data
    trig_data = trig_embed(xy_sub)
    var_names = ['cx', 'sx', 'cy', 'sy']

    # ---- Decompose each latent dim ----
    print(f"\nManifold decomposition (degree-2 in trig vars):")
    print(f"{'Dim':>4}  {'R²(g)':>8}  {'R²(hg)':>8}  {'#':>3}  Top terms (Fourier modes)")
    print("-" * 75)

    decomps = []
    for ti in range(latent_dim):
        # Get f values and gradients
        grad_f, f_vals = compute_trig_gradient(model, xy_tensor, ti)

        # Recover inner polynomial in trig variables
        inner = recover_inner_function_nd(trig_data, var_names, f_vals,
                                           max_degree=2, alpha=1e-5)

        # Evaluate g and grad_g
        g_vals = eval_polynomial_nd(trig_data, inner['coeffs'], inner['powers'])
        grad_g = grad_polynomial_nd(trig_data, inner['coeffs'], inner['powers'])

        # Outer function recovery (simplified — use affine)
        from numpy.linalg import lstsq
        A_aff = np.column_stack([g_vals, np.ones(len(g_vals))])
        c_aff, _, _, _ = lstsq(A_aff, f_vals, rcond=None)
        f_hat = c_aff[0] * g_vals + c_aff[1]

        ss_res = np.sum((f_vals - f_hat) ** 2)
        ss_tot = np.sum((f_vals - f_vals.mean()) ** 2)
        recon_r2 = 1 - ss_res / max(ss_tot, 1e-12)

        # Alignment
        align = gradient_alignment(grad_f, grad_g)

        decomps.append({
            'target_idx': ti,
            'inner': inner,
            'f_vals': f_vals,
            'g_vals': g_vals,
            'f_hat': f_hat,
            'g_recon_r2': inner['r2'],
            'recon_r2': recon_r2,
            'grad_f': grad_f,
            'grad_g': grad_g,
            'alignment': align,
        })

        if recon_r2 > 0.8 or inner['r2'] > 0.8:
            top = sorted(zip(inner['active_names'], inner['active_coeffs']),
                         key=lambda x: -abs(x[1]))[:5]
            top_str = '  '.join(f'{c:+.3f}·{fourier_name(inner["powers"][inner["active"][j]])}'
                               if j < len(inner['active']) else ''
                               for j, (n, c) in enumerate(top))
            # Simpler: just use names
            top_str = '  '.join(f'{c:+.3f}·{n}' for n, c in top)
            print(f'z[{ti:2d}]  {inner["r2"]:8.3f}  {recon_r2:8.3f}  '
                  f'{len(inner["active"]):3d}  {top_str}')

    # ---- Summary: dictionary across all dims ----
    all_terms = {}
    for d in decomps:
        for name, coeff in zip(d['inner']['active_names'], d['inner']['active_coeffs']):
            if name == '1':
                continue
            if name not in all_terms or abs(coeff) > abs(all_terms[name]):
                all_terms[name] = coeff

    # Sort by max coefficient
    sorted_terms = sorted(all_terms.items(), key=lambda x: -abs(x[1]))
    print(f"\nDICTIONARY (union across all dims, sorted by max |coeff|):")
    for name, coeff in sorted_terms[:15]:
        # Count how many dims have this term
        count = sum(1 for d in decomps
                    if name in d['inner']['active_names'])
        print(f"  {name:>10}  max|c|={abs(coeff):.4f}  in {count}/{latent_dim} dims")

    # ---- Plot manifolds for best dims ----
    ranked = sorted(range(latent_dim), key=lambda i: -decomps[i]['recon_r2'])
    targets = ranked[:6]

    print(f"\nPlotting manifolds for z{targets}...")

    for ti in targets:
        d = decomps[ti]
        f_vals = d['f_vals']
        g_vals = d['g_vals']
        f_hat = d['f_hat']
        inner = d['inner']

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        # Row 1: decomposition
        # Lasso coefficients
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
        ax.set_title(f'z[{ti}] Polynomial (Fourier) Terms\n'
                     f'R²(g)={inner["r2"]:.4f}', fontsize=10)

        # f vs g
        ax = axes[0, 1]
        sub = np.random.choice(len(f_vals), min(3000, len(f_vals)), replace=False)
        ax.scatter(g_vals[sub], f_vals[sub], s=1, alpha=0.2, rasterized=True)
        ax.set_xlabel('g(cx,sx,cy,sy)')
        ax.set_ylabel(f'z[{ti}]')
        ax.set_title('f vs g')

        # f vs h(g)
        ax = axes[0, 2]
        ax.scatter(f_hat[sub], f_vals[sub], s=1, alpha=0.2, rasterized=True)
        lim = [min(f_hat.min(), f_vals.min()), max(f_hat.max(), f_vals.max())]
        ax.plot(lim, lim, 'r--', lw=1)
        ax.set_xlabel('h(g)')
        ax.set_ylabel(f'z[{ti}]')
        ax.set_title(f'R²(h∘g)={d["recon_r2"]:.4f}')

        # Row 2: phase space manifolds
        # Torus colored by f
        ax = axes[1, 0]
        sc = ax.scatter(xy_sub[sub, 0], xy_sub[sub, 1], c=f_vals[sub],
                        s=2, alpha=0.5, cmap='coolwarm', rasterized=True)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title(f'z[{ti}] on torus')
        ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
        plt.colorbar(sc, ax=ax, shrink=0.8)

        # Torus colored by g
        ax = axes[1, 1]
        sc = ax.scatter(xy_sub[sub, 0], xy_sub[sub, 1], c=g_vals[sub],
                        s=2, alpha=0.5, cmap='coolwarm', rasterized=True)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title('g(cx,sx,cy,sy) on torus')
        ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
        plt.colorbar(sc, ax=ax, shrink=0.8)

        # Level-set slices on torus
        ax = axes[1, 2]
        ax.scatter(xy_sub[sub, 0], xy_sub[sub, 1], c='lightgray',
                   s=0.5, alpha=0.1, rasterized=True)
        g_min, g_max = np.percentile(g_vals, 10), np.percentile(g_vals, 90)
        levels = np.linspace(g_min, g_max, 7)[1:-1]
        cmap_lev = plt.cm.viridis
        for k, lev in enumerate(levels):
            tol = (g_max - g_min) * 0.03
            mask = np.abs(g_vals - lev) < tol
            if mask.sum() > 10:
                ax.scatter(xy_sub[mask, 0], xy_sub[mask, 1],
                           c=[cmap_lev(k / len(levels))],
                           s=4, alpha=0.6, label=f'g={lev:.2f}', rasterized=True)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title('Level-set curves on torus')
        ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
        ax.legend(fontsize=6, markerscale=3)

        top_terms = sorted(zip(inner['active_names'], inner['active_coeffs']),
                           key=lambda x: -abs(x[1]))[:4]
        expr = '  '.join(f'{c:+.3f}·{n}' for n, c in top_terms)
        fig.suptitle(f'Standard Map z[{ti}]:  g = {expr}', fontsize=12)
        fig.tight_layout()
        fig.savefig(OUTDIR / f'manifold_z{ti}.png', dpi=150, bbox_inches='tight')
        fig.savefig(OUTDIR / f'manifold_z{ti}.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved manifold_z{ti}.png/pdf")

    # ---- Combined manifold plot ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for col, ti in enumerate(targets[:6]):
        ax = axes[col // 3, col % 3]
        d = decomps[ti]
        sc = ax.scatter(xy_sub[sub, 0], xy_sub[sub, 1], c=d['g_vals'][sub],
                        s=2, alpha=0.5, cmap='coolwarm', rasterized=True)
        top = sorted(zip(d['inner']['active_names'], d['inner']['active_coeffs']),
                     key=lambda x: -abs(x[1]))[:3]
        top_str = ' '.join(f'{c:+.2f}·{n}' for n, c in top if n != '1')
        ax.set_title(f'z[{ti}]: {top_str}\nR²={d["recon_r2"]:.3f}', fontsize=9)
        ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        plt.colorbar(sc, ax=ax, shrink=0.7)

    fig.suptitle(f'Standard Map K={K}: Koopman Observable Manifolds', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'manifolds_combined.png', dpi=150, bbox_inches='tight')
    fig.savefig(OUTDIR / 'manifolds_combined.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved manifolds_combined.png/pdf")

    return decomps


# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=float, default=0.97)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    set_seed(42)
    K = args.K

    # First check if model exists, if not train it
    ckpt_path = 'old/standard_map_16_64.pt'
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path}. Training [16,64] model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        traj = simulate_standard_map(n_traj=200, n_steps=2000, K=K, device=device).cpu()
        from standard_map import train_model
        model, history = train_model(16, 64, traj, epochs=300, device=device)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {'mid': 16, 'latent': 64, 'K': K},
        }, ckpt_path)
        print(f"Saved {ckpt_path}")
    else:
        model, cfg = load_model()
        K = cfg.get('K', K)

    print(f"Standard Map K={K}")

    traj = simulate_standard_map(n_traj=200, n_steps=2000, K=K)
    decomps = analyze_and_plot(model, traj, K)

    print(f"\nAll results in {OUTDIR}/")
