"""
Standard Map: manifold/level-set decomposition comparison across propagator types.

Compares dictionary recovery for:
    1. Unconstrained  — M = nn.Linear (original)
    2. Stable         — M = expm(K - L^T L), |λ| ≤ 1
    3. Unitary        — M = Cayley(A), |λ| = 1

All three share the same KANEncoder(kan1, kan2) with trig embedding,
so the decomposition pipeline is identical.
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
    recover_inner_function_nd, eval_polynomial_nd, grad_polynomial_nd,
)
from dictionary_viability import gradient_alignment

from standard_map import DiscreteKoopmanKAN, simulate_standard_map
from standard_map_stable import StableKoopmanKAN
from standard_map_unitary import UnitaryKoopmanKAN

OUTDIR = Path('results/standard_map')

# Default checkpoints
CHECKPOINTS = {
    'unconstrained': 'old/standard_map_16_64.pt',
    'stable':        'results/standard_map/standard_map_stable_32_128.pt',
    'unitary':       'old/standard_map_unitary_32_256.pt',
}

MODEL_CLASSES = {
    'unconstrained': DiscreteKoopmanKAN,
    'stable':        StableKoopmanKAN,
    'unitary':       UnitaryKoopmanKAN,
}


# =====================================================================
# Model loading
# =====================================================================

def load_model(name, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    cls = MODEL_CLASSES[name]
    model = cls(cfg['mid'], cfg['latent'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded {name}: [{cfg['mid']},{cfg['latent']}] from {ckpt_path}")
    return model, cfg


# =====================================================================
# Gradient computation (same for all — encoder structure is identical)
# =====================================================================

def compute_trig_gradient(model, xy_tensor, target_idx):
    """Gradient of latent dim w.r.t. (cx, sx, cy, sy) via autograd."""
    N = xy_tensor.shape[0]
    chunk = 2000
    grad_list, f_list = [], []

    for i in range(0, N, chunk):
        xy_i = xy_tensor[i:i+chunk].detach()
        x, y = xy_i[:, 0], xy_i[:, 1]
        trig = torch.stack([torch.cos(x), torch.sin(x),
                            torch.cos(y), torch.sin(y)], dim=-1).requires_grad_(True)
        z = model.encoder.kan2(model.encoder.kan1(trig))
        fi = z[:, target_idx]
        fi.sum().backward()
        grad_list.append(trig.grad.detach().numpy())
        f_list.append(fi.detach().numpy())

    return np.concatenate(grad_list), np.concatenate(f_list)


def trig_embed(xy):
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([np.cos(x), np.sin(x), np.cos(y), np.sin(y)])


def fourier_name(powers):
    parts = []
    for p, name in zip(powers, ['cx', 'sx', 'cy', 'sy']):
        if p == 1:
            parts.append(name)
        elif p > 1:
            parts.append(f'{name}^{p}')
    return '·'.join(parts) if parts else '1'


# =====================================================================
# Decomposition
# =====================================================================

def decompose_model(model, xy_sub, xy_tensor, trig_data, var_names):
    """Run level-set decomposition on all latent dims of a model."""
    latent_dim = model.latent_dim
    decomps = []

    for ti in range(latent_dim):
        grad_f, f_vals = compute_trig_gradient(model, xy_tensor, ti)

        inner = recover_inner_function_nd(trig_data, var_names, f_vals,
                                          max_degree=2, alpha=1e-5)

        g_vals = eval_polynomial_nd(trig_data, inner['coeffs'], inner['powers'])
        grad_g = grad_polynomial_nd(trig_data, inner['coeffs'], inner['powers'])

        # Affine outer
        from numpy.linalg import lstsq
        A_aff = np.column_stack([g_vals, np.ones(len(g_vals))])
        c_aff, _, _, _ = lstsq(A_aff, f_vals, rcond=None)
        f_hat = c_aff[0] * g_vals + c_aff[1]

        ss_res = np.sum((f_vals - f_hat) ** 2)
        ss_tot = np.sum((f_vals - f_vals.mean()) ** 2)
        recon_r2 = 1 - ss_res / max(ss_tot, 1e-12)

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

    return decomps


def print_decomposition(name, decomps):
    """Print summary table for a model's decomposition."""
    latent_dim = len(decomps)
    print(f"\n{'='*70}")
    print(f"  {name.upper()} — Manifold Decomposition (degree-2 in trig vars)")
    print(f"{'='*70}")
    print(f"{'Dim':>4}  {'R²(g)':>8}  {'R²(hg)':>8}  {'#':>3}  Top terms")
    print("-" * 70)

    good = 0
    for d in decomps:
        ti = d['target_idx']
        inner = d['inner']
        if d['recon_r2'] > 0.8 or inner['r2'] > 0.8:
            good += 1
            top = sorted(zip(inner['active_names'], inner['active_coeffs']),
                         key=lambda x: -abs(x[1]))[:5]
            top_str = '  '.join(f'{c:+.3f}·{n}' for n, c in top)
            print(f'z[{ti:2d}]  {inner["r2"]:8.3f}  {d["recon_r2"]:8.3f}  '
                  f'{len(inner["active"]):3d}  {top_str}')

    print(f"\n  {good}/{latent_dim} dims with R² > 0.8")
    return good


def get_dictionary(decomps):
    """Extract union dictionary across all dims."""
    all_terms = {}
    for d in decomps:
        for name, coeff in zip(d['inner']['active_names'],
                               d['inner']['active_coeffs']):
            if name == '1':
                continue
            if name not in all_terms or abs(coeff) > abs(all_terms[name]):
                all_terms[name] = coeff
    return sorted(all_terms.items(), key=lambda x: -abs(x[1]))


# =====================================================================
# Per-model manifold plots (same format as standard_map_analysis.py)
# =====================================================================

def plot_manifolds(name, decomps, xy_sub, outdir):
    """Plot individual + combined manifold plots for one model."""
    ranked = sorted(range(len(decomps)), key=lambda i: -decomps[i]['recon_r2'])
    targets = ranked[:6]
    sub = np.random.choice(len(xy_sub), min(3000, len(xy_sub)), replace=False)

    for ti in targets:
        d = decomps[ti]
        f_vals, g_vals, f_hat = d['f_vals'], d['g_vals'], d['f_hat']
        inner = d['inner']

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        # Row 1, col 0: Lasso coefficients
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

        # Row 1, col 1: f vs g
        ax = axes[0, 1]
        ax.scatter(g_vals[sub], f_vals[sub], s=1, alpha=0.2, rasterized=True)
        ax.set_xlabel('g(cx,sx,cy,sy)')
        ax.set_ylabel(f'z[{ti}]')
        ax.set_title('f vs g')

        # Row 1, col 2: f vs h(g)
        ax = axes[0, 2]
        ax.scatter(f_hat[sub], f_vals[sub], s=1, alpha=0.2, rasterized=True)
        lim = [min(f_hat.min(), f_vals.min()), max(f_hat.max(), f_vals.max())]
        ax.plot(lim, lim, 'r--', lw=1)
        ax.set_xlabel('h(g)')
        ax.set_ylabel(f'z[{ti}]')
        ax.set_title(f'R²(h∘g)={d["recon_r2"]:.4f}')

        # Row 2, col 0: torus colored by f
        ax = axes[1, 0]
        sc = ax.scatter(xy_sub[sub, 0], xy_sub[sub, 1], c=f_vals[sub],
                        s=2, alpha=0.5, cmap='coolwarm', rasterized=True)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title(f'z[{ti}] on torus')
        ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
        plt.colorbar(sc, ax=ax, shrink=0.8)

        # Row 2, col 1: torus colored by g
        ax = axes[1, 1]
        sc = ax.scatter(xy_sub[sub, 0], xy_sub[sub, 1], c=g_vals[sub],
                        s=2, alpha=0.5, cmap='coolwarm', rasterized=True)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title('g(cx,sx,cy,sy) on torus')
        ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
        plt.colorbar(sc, ax=ax, shrink=0.8)

        # Row 2, col 2: level-set curves
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
        fig.suptitle(f'{name} z[{ti}]:  g = {expr}', fontsize=12)
        fig.tight_layout()
        fig.savefig(outdir / f'manifold_{name}_z{ti}.png', dpi=150, bbox_inches='tight')
        fig.savefig(outdir / f'manifold_{name}_z{ti}.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved manifold_{name}_z{ti}.png/pdf")

    # Combined 2x3
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

    fig.suptitle(f'{name}: Koopman Observable Manifolds', fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / f'manifolds_combined_{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(outdir / f'manifolds_combined_{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved manifolds_combined_{name}.png/pdf")


# =====================================================================
# Comparison plot across models
# =====================================================================

def plot_comparison(all_decomps, outdir):
    """Side-by-side comparison: R² distributions and dictionary overlap."""
    names = list(all_decomps.keys())
    n_models = len(names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: R²(h∘g) distribution per model
    ax = axes[0]
    colors = {'unconstrained': '#1f77b4', 'stable': '#2ca02c', 'unitary': '#d62728'}
    for name in names:
        r2s = sorted([d['recon_r2'] for d in all_decomps[name]], reverse=True)
        ax.plot(r2s, 'o-', markersize=3, lw=1, color=colors.get(name, 'gray'),
                label=f'{name} (n={len(r2s)})')
    ax.axhline(0.8, color='gray', ls='--', alpha=0.3, label='R²=0.8')
    ax.set_xlabel('Latent dim (sorted)')
    ax.set_ylabel('R²(h∘g)')
    ax.set_title('Reconstruction Quality per Dim')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 2: # dims above threshold
    ax = axes[1]
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    x_pos = np.arange(len(thresholds))
    width = 0.8 / n_models
    for i, name in enumerate(names):
        r2s = [d['recon_r2'] for d in all_decomps[name]]
        counts = [sum(1 for r in r2s if r > t) for t in thresholds]
        ax.bar(x_pos + i * width, counts, width, label=name,
               color=colors.get(name, 'gray'), edgecolor='k', lw=0.5)
    ax.set_xticks(x_pos + width * (n_models - 1) / 2)
    ax.set_xticklabels([f'>{t}' for t in thresholds])
    ax.set_xlabel('R²(h∘g) threshold')
    ax.set_ylabel('# latent dims')
    ax.set_title('Interpretable Dimensions')
    ax.legend(fontsize=8)

    # Panel 3: Dictionary comparison — which Fourier terms appear?
    ax = axes[2]
    # Collect dictionaries
    dicts = {}
    for name in names:
        terms = get_dictionary(all_decomps[name])
        dicts[name] = {t: c for t, c in terms}

    # Union of top terms
    all_term_names = set()
    for d in dicts.values():
        all_term_names |= set(list(d.keys())[:10])
    all_term_names = sorted(all_term_names)[:12]

    y_pos = np.arange(len(all_term_names))
    for i, name in enumerate(names):
        vals = [abs(dicts[name].get(t, 0)) for t in all_term_names]
        ax.barh(y_pos + i * width, vals, width, label=name,
                color=colors.get(name, 'gray'), edgecolor='k', lw=0.5)
    ax.set_yticks(y_pos + width * (n_models - 1) / 2)
    ax.set_yticklabels(all_term_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('max |coefficient|')
    ax.set_title('Dictionary Terms (union)')
    ax.legend(fontsize=8)

    fig.suptitle('Standard Map: Propagator Comparison — Dictionary Recovery', fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / 'comparison_constrained.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / 'comparison_constrained.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved comparison_constrained.png/pdf")


# =====================================================================
# Eigenvalue comparison
# =====================================================================

def plot_eigenvalue_comparison(models, outdir):
    """Eigenvalue spectra side by side."""
    names = list(models.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    colors = {'unconstrained': '#1f77b4', 'stable': '#2ca02c', 'unitary': '#d62728'}
    theta = np.linspace(0, 2 * np.pi, 200)

    for ax, name in zip(axes, names):
        model = models[name]
        M = model.koopman_matrix().numpy()
        eigs = np.linalg.eigvals(M)
        mags = np.abs(eigs)

        ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.3)
        sc = ax.scatter(eigs.real, eigs.imag, s=30, c=mags, cmap='RdYlGn',
                        vmin=0, vmax=1.05, edgecolors='black', linewidths=0.3, zorder=3)
        ax.set_xlabel(r'Re($\lambda$)')
        ax.set_ylabel(r'Im($\lambda$)')
        ax.set_title(f'{name}\n|λ| ∈ [{mags.min():.3f}, {mags.max():.3f}]')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        plt.colorbar(sc, ax=ax, label='|λ|', shrink=0.8)

    fig.suptitle('Koopman Eigenvalue Spectra by Propagator Type', fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / 'eigenvalue_comparison.png', dpi=200, bbox_inches='tight')
    fig.savefig(outdir / 'eigenvalue_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved eigenvalue_comparison.png/pdf")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=float, default=0.97)
    parser.add_argument('--unconstrained', type=str, default=CHECKPOINTS['unconstrained'])
    parser.add_argument('--stable', type=str, default=CHECKPOINTS['stable'])
    parser.add_argument('--unitary', type=str, default=CHECKPOINTS['unitary'])
    parser.add_argument('--skip', nargs='*', default=[], help='Model names to skip')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    K = args.K

    # Simulate shared data
    print(f"Standard Map K={K}: Constrained Propagator Comparison")
    print("Simulating...")
    traj = simulate_standard_map(n_traj=200, n_steps=2000, K=K)
    xy_np = traj.reshape(-1, 2).numpy()
    idx = np.random.choice(len(xy_np), min(8000, len(xy_np)), replace=False)
    xy_sub = xy_np[idx]
    xy_tensor = torch.tensor(xy_sub, dtype=torch.float32)
    trig_data = trig_embed(xy_sub)
    var_names = ['cx', 'sx', 'cy', 'sy']

    # Load models
    ckpt_map = {
        'unconstrained': args.unconstrained,
        'stable': args.stable,
        'unitary': args.unitary,
    }

    models = {}
    all_decomps = {}

    for name in ['unconstrained', 'stable', 'unitary']:
        if name in args.skip:
            print(f"\n  Skipping {name}")
            continue
        path = ckpt_map[name]
        if not os.path.exists(path):
            print(f"\n  {name}: checkpoint not found at {path}, skipping")
            continue

        model, cfg = load_model(name, path)
        models[name] = model

        print(f"\n  Decomposing {name} (latent_dim={model.latent_dim})...")
        decomps = decompose_model(model, xy_sub, xy_tensor, trig_data, var_names)
        all_decomps[name] = decomps

        n_good = print_decomposition(name, decomps)
        plot_manifolds(name, decomps, xy_sub, OUTDIR)

        # Print dictionary
        terms = get_dictionary(decomps)
        print(f"\n  DICTIONARY ({name}):")
        for term_name, coeff in terms[:15]:
            count = sum(1 for d in decomps
                        if term_name in d['inner']['active_names'])
            print(f"    {term_name:>10}  max|c|={abs(coeff):.4f}  "
                  f"in {count}/{model.latent_dim} dims")

    # Comparison plots
    if len(all_decomps) >= 2:
        plot_comparison(all_decomps, OUTDIR)

    if len(models) >= 2:
        plot_eigenvalue_comparison(models, OUTDIR)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<16} {'Latent':>6} {'R²>0.8':>7} {'R²>0.9':>7} "
          f"{'R²>0.95':>8} {'Med R²':>7}")
    print("-" * 55)
    for name, decomps in all_decomps.items():
        r2s = [d['recon_r2'] for d in decomps]
        n = len(r2s)
        print(f"{name:<16} {n:>6} {sum(1 for r in r2s if r>0.8):>7} "
              f"{sum(1 for r in r2s if r>0.9):>7} "
              f"{sum(1 for r in r2s if r>0.95):>8} "
              f"{np.median(r2s):>7.3f}")

    print(f"\nAll results in {OUTDIR}/")
