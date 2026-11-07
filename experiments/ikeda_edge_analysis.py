"""
Ikeda Map: Layer-1 edge function analysis and foliation alignment.

Loads a trained pruned Ikeda model and inspects the KAN encoder's
layer-1 edge functions ψ_{j,i}(x_i) to determine:
  1. Which edges are quadratic (ψ ≈ a·x_i² + b·x_i + c)
  2. Whether quadratic edges pair x² and y² into r² = x² + y²
  3. Foliation alignment: R²_radial of each layer-1 node u_j against r²
  4. Channel activity |A_k| in layer-2 (regime diagnostic)

The Ikeda nonlinearity factors through g(x,y) = x² + y², so the
proposition predicts that the encoder discovers this foliation.

Usage:
  python experiments/ikeda_edge_analysis.py [--seed 4]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from plotting import inspect_kanlinear
from ikeda import generate_ikeda
from ikeda_koopman import IkedaStableKoopman

OUTDIR = Path(__file__).resolve().parent.parent / 'results' / 'ikeda'


def classify_edge(x_in, y_edge):
    """Fit linear and quadratic to a single edge function."""
    ss_tot = np.sum((y_edge - y_edge.mean())**2)
    if ss_tot < 1e-12:
        return {'r2_lin': 1.0, 'r2_quad': 1.0, 'quad_coeff': 0.0,
                'lin_slope': 0.0, 'is_quadratic': False}

    c_lin = np.polyfit(x_in, y_edge, 1)
    r2_lin = 1.0 - np.sum((y_edge - np.polyval(c_lin, x_in))**2) / ss_tot

    c_quad = np.polyfit(x_in, y_edge, 2)
    r2_quad = 1.0 - np.sum((y_edge - np.polyval(c_quad, x_in))**2) / ss_tot

    return {
        'r2_lin': float(r2_lin),
        'r2_quad': float(r2_quad),
        'quad_coeff': float(c_quad[0]),
        'lin_slope': float(c_lin[0]),
        'is_quadratic': r2_quad > 0.9 and r2_lin < 0.8 and abs(c_quad[0]) > 0.01,
    }


def foliation_r2(u_vals, r2_vals, degree=4):
    """R² of u_j regressed on polynomial(r²)."""
    poly = PolynomialFeatures(degree=degree)
    feat = poly.fit_transform(r2_vals.reshape(-1, 1))
    reg = LinearRegression().fit(feat, u_vals)
    return float(reg.score(feat, u_vals)), reg.predict(feat)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4)
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
    pts = torch.tensor(pts_np, dtype=torch.float32)
    x_np, y_np = pts_np[:, 0], pts_np[:, 1]
    r2 = x_np**2 + y_np**2

    # ── Layer-1 edge functions ──────────────────────────────────────
    info = inspect_kanlinear(model.encoder.kan1, pts)
    mid_dim = model.encoder.kan1.out_features

    print("\n" + "="*70)
    print("LAYER-1 EDGE FUNCTIONS: ψ_{j,i}(x_i)")
    print("="*70)

    edge_results = []
    for j in range(mid_dim):
        for i in range(2):
            x_in = info['x_in'][:, i].numpy()
            y_edge = info['postacts'][:, j, i].numpy()
            var = 'x' if i == 0 else 'y'
            cls = classify_edge(x_in, y_edge)
            cls['node'] = j
            cls['input'] = i
            cls['var'] = var
            cls['variance'] = float(np.var(y_edge))
            edge_results.append(cls)

            tag = ' *** QUADRATIC' if cls['is_quadratic'] else ''
            print(f"  ψ[{j},{i}]({var}): R²_lin={cls['r2_lin']:.3f}, "
                  f"R²_quad={cls['r2_quad']:.3f}, a={cls['quad_coeff']:.4f}, "
                  f"var={cls['variance']:.4f}{tag}")

    # ── Layer-1 node foliation alignment ────────────────────────────
    with torch.no_grad():
        u = model.encoder.kan1(pts).numpy()

    print("\n" + "="*70)
    print("FOLIATION ALIGNMENT: u_j vs r² = x² + y²")
    print("="*70)

    for j in range(mid_dim):
        r2_score, fit = foliation_r2(u[:, j], r2)
        tag = ' *** ALIGNED' if r2_score > 0.8 else ''
        print(f"  u[{j}]: R²_radial = {r2_score:.4f}, "
              f"std = {np.std(u[:, j]):.4f}{tag}")

    # ── Lasso alpha sweep on u[3] (best r² node) ───────────────────
    print("\n" + "="*70)
    print("LASSO ALPHA SWEEP on u[3] (strongest r² node)")
    print("="*70)

    from manifold_analysis import recover_inner_function_nd
    u3 = u[:, 3]
    for alpha in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
        inner = recover_inner_function_nd(pts_np, ['x', 'y'], u3,
                                           max_degree=3, alpha=alpha)
        # Extract x² and y² coefficients
        x2c = y2c = 0.0
        for nm, cf in zip(inner['active_names'], inner['active_coeffs']):
            if nm == 'x^2': x2c = float(cf)
            elif nm == 'y^2': y2c = float(cf)
        ratio = min(abs(x2c), abs(y2c)) / max(abs(x2c), abs(y2c), 1e-12)
        print(f"  alpha={alpha:.0e}: R²={inner['r2']:.4f}, #active={len(inner['active'])}, "
              f"x²={x2c:+.4f}, y²={y2c:+.4f}, ratio={ratio:.3f}, "
              f"terms={inner['active_names']}")

    # ── Channel activity |A_k| ─────────────────────────────────────
    print("\n" + "="*70)
    print("LAYER-2 CHANNEL ACTIVITY |A_k| (threshold sweep)")
    print("="*70)

    kan2 = model.encoder.kan2
    imp = kan2.spline_weight.abs().mean(dim=-1)
    if hasattr(kan2, 'spline_scaler') and kan2.spline_scaler is not None:
        imp = imp + kan2.spline_scaler.abs()
    imp_np = imp.detach().numpy()

    for tau in [0.01, 0.03, 0.05, 0.1, 0.2]:
        counts = [int(np.sum(imp_np[k] > tau)) for k in range(imp_np.shape[0])]
        hist = np.bincount(counts, minlength=mid_dim + 1)
        print(f"  tau={tau:.2f}: |A_k| = {counts}, mode={int(np.argmax(hist))}")

    # ── Visualization ──────────────────────────────────────────────
    fig, axes = plt.subplots(mid_dim, 4, figsize=(16, 2.5 * mid_dim))

    for j in range(mid_dim):
        # Col 0: ψ_{j,0}(x)
        ax = axes[j, 0]
        x_in = info['x_in'][:, 0].numpy()
        y_edge = info['postacts'][:, j, 0].numpy()
        order = np.argsort(x_in)
        ax.plot(x_in[order], y_edge[order], 'b-', lw=1.5, alpha=0.7)
        cls_x = [e for e in edge_results if e['node'] == j and e['input'] == 0][0]
        color = 'red' if cls_x['is_quadratic'] else 'black'
        ax.set_title(f"ψ[{j},0](x)\na={cls_x['quad_coeff']:.3f}", fontsize=8, color=color)
        ax.tick_params(labelsize=6)
        if j == 0: ax.set_xlabel('x', fontsize=7)

        # Col 1: ψ_{j,1}(y)
        ax = axes[j, 1]
        x_in = info['x_in'][:, 1].numpy()
        y_edge = info['postacts'][:, j, 1].numpy()
        order = np.argsort(x_in)
        ax.plot(x_in[order], y_edge[order], 'r-', lw=1.5, alpha=0.7)
        cls_y = [e for e in edge_results if e['node'] == j and e['input'] == 1][0]
        color = 'red' if cls_y['is_quadratic'] else 'black'
        ax.set_title(f"ψ[{j},1](y)\na={cls_y['quad_coeff']:.3f}", fontsize=8, color=color)
        ax.tick_params(labelsize=6)
        if j == 0: ax.set_xlabel('y', fontsize=7)

        # Col 2: u_j vs r²
        ax = axes[j, 2]
        ax.scatter(r2, u[:, j], s=0.2, alpha=0.1, c='steelblue', rasterized=True)
        r2_score, fit = foliation_r2(u[:, j], r2)
        order = np.argsort(r2)
        ax.plot(r2[order], fit[order], 'r-', lw=1.5)
        ax.set_title(f"u[{j}] vs r²\nR²_radial={r2_score:.3f}", fontsize=8)
        ax.set_xlabel('r²', fontsize=7)
        ax.tick_params(labelsize=6)

        # Col 3: attractor colored by u_j
        ax = axes[j, 3]
        sc = ax.scatter(x_np, y_np, s=0.2, alpha=0.2, c=u[:, j],
                        cmap='coolwarm', rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(f"u[{j}] on attractor", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(f'Ikeda Layer-1 Edge Analysis (seed {args.seed})', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'edge_analysis.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTDIR / 'edge_analysis.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {OUTDIR / 'edge_analysis.png'}")
