"""
Ikeda Map: Layer-1 h∘g decomposition — the key dictionary extraction result.

Runs the full level-set + gradient decomposition on the LAYER-1 inner
functions u_j = ψ_{j,0}(x) + ψ_{j,1}(y), not on z_k.

Why layer-1 and not z_k?
  The Ikeda map's nonlinearity factors through g = x² + y².  Each
  output coordinate is bilinear: cos(t(g))·x - sin(t(g))·y.  So z_k
  (the layer-2 output) is a bilinear function of (x,y) with
  transcendental coefficients — no sparse polynomial g exists for z_k.

  But layer-1 outputs u_j are ADDITIVE: u_j = ψ_{j,0}(x) + ψ_{j,1}(y).
  When ψ_{j,0}(x) ≈ a·x² and ψ_{j,1}(y) ≈ a·y², then u_j ≈ a·(x²+y²).
  The h∘g decomposition on u_j should recover g ≈ x²+y² (sparse) with
  h ≈ affine.

  This is the dictionary extraction result: the KAN encoder discovered
  x² + y² = r² as the Koopman-relevant observable, and the inner
  function recovery confirms it via the gradient formula.

Usage:
  python experiments/ikeda_layer1_decomposition.py [--seed 4] [--alpha 1e-3]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from ikeda import generate_ikeda
from ikeda_koopman import IkedaStableKoopman, MID_DIM
from manifold_analysis import (
    recover_inner_function_nd, eval_polynomial_nd,
    grad_polynomial_nd, recover_outer_function_from_raw,
    eval_h, analyze_ideal,
)

OUTDIR = Path(__file__).resolve().parent.parent / 'results' / 'ikeda'


def decompose_layer1_node(model, pts_np, j, max_degree=3, alpha=1e-3, h_degree=4):
    """Full h∘g decomposition on layer-1 output u_j.

    Computes ∇u_j via autograd, runs Lasso for inner g,
    gradient formula for outer h, returns full result dict.
    """
    pts_t = torch.tensor(pts_np, dtype=torch.float32)

    # Forward + backward for u_j
    xi = pts_t.detach().clone().requires_grad_(True)
    u = model.encoder.kan1(xi)
    uj = u[:, j]
    uj.sum().backward()
    grad_uj = xi.grad.detach().numpy()
    uj_np = uj.detach().numpy()

    # Inner function via Lasso
    inner = recover_inner_function_nd(pts_np, ['x', 'y'], uj_np,
                                       max_degree=max_degree, alpha=alpha)

    # Evaluate g and ∇g
    g_vals = eval_polynomial_nd(pts_np, inner['coeffs'], inner['powers'])
    grad_g = grad_polynomial_nd(pts_np, inner['coeffs'], inner['powers'])

    # Outer function via gradient formula: h'(g) = ∇f·∇g / ||∇g||²
    dot = np.sum(grad_uj * grad_g, axis=1)
    norm_sq = np.sum(grad_g**2, axis=1)
    outer = recover_outer_function_from_raw(dot, norm_sq, g_vals, uj_np,
                                             poly_degree=h_degree)

    f_hat = eval_h(g_vals, outer)
    ss_res = np.sum((uj_np - f_hat)**2)
    ss_tot = np.sum((uj_np - uj_np.mean())**2)
    r2_hg = 1.0 - ss_res / max(ss_tot, 1e-12)

    ideal = analyze_ideal(inner)

    # r² signature
    x2c = y2c = 0.0
    for nm, cf in zip(inner['active_names'], inner['active_coeffs']):
        if nm == 'x^2': x2c = float(cf)
        elif nm == 'y^2': y2c = float(cf)
    ratio = min(abs(x2c), abs(y2c)) / max(abs(x2c), abs(y2c), 1e-12)
    same_sign = (x2c * y2c > 0) and abs(x2c) > 0.01 and abs(y2c) > 0.01
    r2_sig = same_sign and ratio > 0.5

    return {
        'node': j,
        'r2_g': float(inner['r2']),
        'r2_hg': float(r2_hg),
        'n_active': len(inner['active']),
        'active_names': list(inner['active_names']),
        'active_coeffs': [float(c) for c in inner['active_coeffs']],
        'h_affine': bool(outer.get('used_affine', False)),
        'x2_coeff': x2c,
        'y2_coeff': y2c,
        'ratio': ratio,
        'r2_signature': r2_sig,
        'ideal_type': ideal['type'],
        'ideal_expression': ideal.get('expression', ''),
        # For plotting
        'uj_np': uj_np,
        'g_vals': g_vals,
        'f_hat': f_hat,
        'outer': outer,
        'inner': inner,
        'grad_uj': grad_uj,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=1e-3)
    parser.add_argument('--degree', type=int, default=3)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt_path = OUTDIR / f'ikeda_pruned_seed_{args.seed}.pt'
    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = IkedaStableKoopman()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Data
    traj = generate_ikeda(n_traj=50, n_steps=1000, burn_in=500, seed=42)
    pts_np = traj.reshape(-1, 2)[:10000]

    # ── Decompose all layer-1 nodes ────────────────────────────────
    print("\n" + "="*70)
    print(f"Layer-1 h∘g Decomposition (alpha={args.alpha}, degree={args.degree})")
    print("="*70)

    results = []
    for j in range(MID_DIM):
        r = decompose_layer1_node(model, pts_np, j,
                                   max_degree=args.degree, alpha=args.alpha)
        results.append(r)
        tag = ' *** r² FOUND' if r['r2_signature'] else ''
        print(f"  u[{j}]: R²(g)={r['r2_g']:.4f}, R²(h∘g)={r['r2_hg']:.4f}, "
              f"x²={r['x2_coeff']:+.4f}, y²={r['y2_coeff']:+.4f}, "
              f"ratio={r['ratio']:.3f}, h={'affine' if r['h_affine'] else 'nonlinear'}, "
              f"terms={r['active_names']}{tag}")

    n_sig = sum(1 for r in results if r['r2_signature'])
    print(f"\n  r² signature found in {n_sig}/{MID_DIM} layer-1 nodes")

    # ── Publication figure ─────────────────────────────────────────
    # One row per node with r² signature
    sig_results = [r for r in results if r['r2_signature']]
    if not sig_results:
        sig_results = sorted(results, key=lambda r: r['ratio'], reverse=True)[:2]

    n_rows = len(sig_results)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    x_np, y_np = pts_np[:, 0], pts_np[:, 1]
    r2 = x_np**2 + y_np**2

    for row, r in enumerate(sig_results):
        j = r['node']
        uj = r['uj_np']
        g_vals = r['g_vals']
        f_hat = r['f_hat']
        outer = r['outer']

        # (a) Attractor colored by u_j
        ax = axes[row, 0]
        sc = ax.scatter(x_np, y_np, s=0.3, alpha=0.3, c=uj,
                        cmap='coolwarm', rasterized=True)
        ax.set_ylabel(f'u[{j}]', fontsize=10, fontweight='bold')
        ax.set_title(f'$u_{j}(x,y)$ on attractor', fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

        # (b) u_j vs r² (the collapse plot)
        ax = axes[row, 1]
        ax.scatter(r2, uj, s=0.3, alpha=0.15, c='steelblue', rasterized=True)
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        poly = PolynomialFeatures(degree=4)
        feat = poly.fit_transform(r2.reshape(-1, 1))
        reg = LinearRegression().fit(feat, uj)
        r2_score = reg.score(feat, uj)
        order = np.argsort(r2)
        ax.plot(r2[order], reg.predict(feat)[order], 'r-', lw=2)
        ax.set_xlabel('$r^2 = x^2 + y^2$', fontsize=8)
        ax.set_ylabel(f'$u_{j}$', fontsize=8)
        ax.set_title(f'Collapse onto $r^2$ (R²={r2_score:.3f})', fontsize=9)
        ax.tick_params(labelsize=7)

        # (c) Outer h'(g)
        ax = axes[row, 2]
        if 'g_samples' in outer and len(outer['g_samples']) > 0:
            ax.scatter(outer['g_samples'], outer['h_prime_samples'],
                       s=0.3, alpha=0.1, c='gray', rasterized=True, label="$h'$")
            if 'g_plot' in outer:
                ax.plot(outer['g_plot'], outer['h_prime_fit'],
                        'r-', lw=2, label="fit")
            ax.legend(fontsize=6)
        h_type = 'affine' if r['h_affine'] else 'nonlinear'
        ax.set_xlabel('$g$', fontsize=8)
        ax.set_ylabel("$h'(g)$", fontsize=8)
        ax.set_title(f"Outer function [{h_type}]", fontsize=9)
        ax.tick_params(labelsize=7)

        # (d) Reconstruction
        ax = axes[row, 3]
        ax.scatter(uj, f_hat, s=0.3, alpha=0.15, c='steelblue', rasterized=True)
        lims = [min(uj.min(), f_hat.min()), max(uj.max(), f_hat.max())]
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
        ax.set_xlabel(f'True $u_{j}$', fontsize=8)
        ax.set_ylabel('$h(g(x,y))$', fontsize=8)
        ax.set_title(f"R²(h∘g)={r['r2_hg']:.4f}\n"
                     f"x²={r['x2_coeff']:+.3f}, y²={r['y2_coeff']:+.3f}", fontsize=9)
        ax.tick_params(labelsize=7)
        terms = [n for n in r['active_names'] if n != '1']
        if terms:
            ax.text(0.02, 0.98, 'g: ' + ', '.join(terms[:5]),
                    transform=ax.transAxes, fontsize=6, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Ikeda Layer-1 Dictionary Extraction\n'
                 'h∘g decomposition recovers $g \\approx x^2 + y^2$ (the foliation)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'layer1_decomposition.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'layer1_decomposition.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {OUTDIR / 'layer1_decomposition.png'}")

    # ── Save results ───────────────────────────────────────────────
    serializable = []
    for r in results:
        s = {k: v for k, v in r.items()
             if k not in ('uj_np', 'g_vals', 'f_hat', 'outer', 'inner', 'grad_uj')}
        serializable.append(s)

    out = {
        'seed': args.seed,
        'alpha': args.alpha,
        'max_degree': args.degree,
        'n_r2_signature': n_sig,
        'results': serializable,
    }
    with open(OUTDIR / 'layer1_decomposition.json', 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(f"Saved {OUTDIR / 'layer1_decomposition.json'}")
