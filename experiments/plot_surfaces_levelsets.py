"""
Surface and level-set comparison: True f vs Lasso h∘g vs KANDy h∘g.

For each test case, shows 2D slices as filled contours + level-set lines:
  Column 1: True f
  Column 2: Lasso inner g → gradient outer h → h(g)
  Column 3: KANDy inner g → gradient outer h → h(g)

Both methods use the SAME gradient-based outer function recovery.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from itertools import combinations_with_replacement
from collections import Counter

from manifold_analysis import (
    recover_inner_function,
    recover_outer_function,
    eval_polynomial,
    grad_polynomial,
    eval_h,
    build_polynomial_dict,
)
from models import _import_pykan
PyKAN = _import_pykan()

OUTDIR = Path('results/surfaces_levelsets')
OUTDIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

# =====================================================================
# Polynomial lift for KANDy
# =====================================================================

def poly_lift(X, degree=3):
    n = X.shape[1]
    combos = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n), d):
            combos.append(combo)
    cols, names = [], []
    for combo in combos:
        col = np.ones(X.shape[0], dtype=X.dtype)
        for idx in combo:
            col *= X[:, idx]
        cols.append(col)
        c = Counter(combo)
        parts = []
        for idx, power in sorted(c.items()):
            vname = ['x', 'y', 'z'][idx] if idx < 3 else f'x_{idx}'
            parts.append(vname if power == 1 else f'{vname}^{power}')
        names.append('*'.join(parts))
    return np.column_stack(cols), names


# =====================================================================
# KANDy inner function + gradient outer
# =====================================================================

def kandy_inner(X, f_vals, grad_f, degree=3, steps=200, grid=15,
                max_train=8000):
    """KANDy inner g on polynomial dict, then gradient outer h."""
    theta, names = poly_lift(X, degree)
    m = theta.shape[1]

    # Subsample for training
    N = len(theta)
    if N > max_train:
        idx = np.random.permutation(N)[:max_train]
        theta_sub, f_sub = theta[idx], f_vals[idx]
    else:
        theta_sub, f_sub = theta, f_vals

    Ns = len(theta_sub)
    n1 = int(Ns * 0.70)
    n2 = int(Ns * 0.85)

    def _t(a):
        return torch.tensor(a, dtype=torch.float32)

    dataset = {
        'train_input': _t(theta_sub[:n1]),
        'train_label': _t(f_sub[:n1, None]),
        'test_input': _t(theta_sub[n1:n2]),
        'test_label': _t(f_sub[n1:n2, None]),
    }

    kan = PyKAN(width=[m, 1], grid=grid, k=3, seed=42)
    kan.fit(dataset, opt='LBFGS', steps=steps, lamb=0.01)

    # Predict g on ALL data
    with torch.no_grad():
        g_vals = kan(_t(theta)).numpy().flatten()

    # Gradient of g via finite differences through the KAN
    x_, y_, z_ = X[:, 0], X[:, 1], X[:, 2]
    eps = 1e-4
    grad_g = np.zeros_like(X)
    for d in range(3):
        Xp = X.copy(); Xp[:, d] += eps
        Xm = X.copy(); Xm[:, d] -= eps
        tp, _ = poly_lift(Xp, degree)
        tm, _ = poly_lift(Xm, degree)
        with torch.no_grad():
            gp = kan(_t(tp)).numpy().flatten()
            gm = kan(_t(tm)).numpy().flatten()
        grad_g[:, d] = (gp - gm) / (2 * eps)

    # R²(g)
    ss_res = np.sum((f_vals - g_vals)**2)
    ss_tot = np.sum((f_vals - f_vals.mean())**2)
    r2_g = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Gradient outer h (SAME method as Lasso)
    grad_g_tuple = (grad_g[:, 0], grad_g[:, 1], grad_g[:, 2])
    outer = recover_outer_function(grad_f, grad_g_tuple, g_vals, f_vals,
                                    poly_degree=4)

    # Full reconstruction
    f_hat = eval_h(g_vals, outer)
    ss_res2 = np.sum((f_vals - f_hat)**2)
    r2_hg = 1.0 - ss_res2 / max(ss_tot, 1e-12)

    # Also need a function to evaluate g on a grid
    def eval_g_grid(X_grid):
        theta_grid, _ = poly_lift(X_grid, degree)
        with torch.no_grad():
            return kan(_t(theta_grid)).numpy().flatten()

    return {
        'g_vals': g_vals,
        'r2_g': r2_g,
        'r2_hg': r2_hg,
        'outer': outer,
        'f_hat': f_hat,
        'eval_g': eval_g_grid,
        'kan': kan,
    }


# =====================================================================
# Lasso inner function + gradient outer
# =====================================================================

def lasso_inner(X, f_vals, grad_f, degree=3, alpha=1e-5):
    """Lasso inner g, then gradient outer h."""
    x_, y_, z_ = X[:, 0], X[:, 1], X[:, 2]

    inner = recover_inner_function(x_, y_, z_, f_vals,
                                    max_degree=degree, alpha=alpha)
    g_vals = eval_polynomial(x_, y_, z_, inner['coeffs'], inner['powers'])
    grad_g = grad_polynomial(x_, y_, z_, inner['coeffs'], inner['powers'])

    outer = recover_outer_function(grad_f, grad_g, g_vals, f_vals,
                                    poly_degree=4)
    f_hat = eval_h(g_vals, outer)
    ss_res = np.sum((f_vals - f_hat)**2)
    ss_tot = np.sum((f_vals - f_vals.mean())**2)
    r2_hg = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Function to evaluate g on a grid
    def eval_g_grid(X_grid):
        return eval_polynomial(X_grid[:, 0], X_grid[:, 1], X_grid[:, 2],
                               inner['coeffs'], inner['powers'])

    return {
        'g_vals': g_vals,
        'r2_g': inner['r2'],
        'r2_hg': r2_hg,
        'outer': outer,
        'f_hat': f_hat,
        'eval_g': eval_g_grid,
        'inner': inner,
    }


# =====================================================================
# Data
# =====================================================================

N = 50_000
x_data = np.random.uniform(-20, 20, N)
y_data = np.random.uniform(-30, 30, N)
z_data = np.random.uniform(0, 50, N)
X_data = np.column_stack([x_data, y_data, z_data])

TEST_CASES = [
    {
        'name': r'$f = \cos(x/20 + y/40)$',
        'label': 'cos',
        'f': np.cos(x_data/20 + y_data/40),
        'grad_f': np.column_stack([
            -np.sin(x_data/20 + y_data/40) / 20,
            -np.sin(x_data/20 + y_data/40) / 40,
            np.zeros(N),
        ]),
        'f_grid': lambda xx, yy, zz: np.cos(xx/20 + yy/40),
        'slice_axes': ('x', 'y'),
        'slice_fix': ('z', 25.0),
        'true_g_str': r'$g = x/20 + y/40$',
        'true_h_str': r'$h = \cos$',
    },
    {
        'name': r'$f = \exp(-(x^2+y^2)/200)$',
        'label': 'exp',
        'f': np.exp(-(x_data**2 + y_data**2) / 200),
        'grad_f': np.column_stack([
            np.exp(-(x_data**2 + y_data**2) / 200) * (-x_data / 100),
            np.exp(-(x_data**2 + y_data**2) / 200) * (-y_data / 100),
            np.zeros(N),
        ]),
        'f_grid': lambda xx, yy, zz: np.exp(-(xx**2 + yy**2) / 200),
        'slice_axes': ('x', 'y'),
        'slice_fix': ('z', 25.0),
        'true_g_str': r'$g = -(x^2+y^2)/200$',
        'true_h_str': r'$h = \exp$',
    },
    {
        'name': r'$f = \tanh(z/10 - 2)$',
        'label': 'tanh',
        'f': np.tanh(z_data/10 - 2),
        'grad_f': np.column_stack([
            np.zeros(N),
            np.zeros(N),
            (1 - np.tanh(z_data/10 - 2)**2) / 10,
        ]),
        'f_grid': lambda xx, yy, zz: np.tanh(zz/10 - 2),
        'slice_axes': ('x', 'z'),
        'slice_fix': ('y', 0.0),
        'true_g_str': r'$g = z/10 - 2$',
        'true_h_str': r'$h = \tanh$',
    },
    {
        'name': r'$f = (xy/100)^3$',
        'label': 'cube',
        'f': (x_data * y_data / 100)**3,
        'grad_f': np.column_stack([
            3 * (x_data * y_data / 100)**2 * y_data / 100,
            3 * (x_data * y_data / 100)**2 * x_data / 100,
            np.zeros(N),
        ]),
        'f_grid': lambda xx, yy, zz: (xx * yy / 100)**3,
        'slice_axes': ('x', 'y'),
        'slice_fix': ('z', 25.0),
        'true_g_str': r'$g = xy/100$',
        'true_h_str': r'$h = (\cdot)^3$',
    },
]


# =====================================================================
# Run both methods on all test cases
# =====================================================================

all_results = {}

for tc in TEST_CASES:
    print(f"\n{'='*60}")
    print(f"  {tc['label']}: {tc['name']}")
    print(f"{'='*60}")

    print("  Running Lasso...")
    rl = lasso_inner(X_data, tc['f'], tc['grad_f'], degree=3, alpha=1e-5)
    print(f"    R²(g)={rl['r2_g']:.4f}  R²(h∘g)={rl['r2_hg']:.4f}")

    print("  Running KANDy...")
    rk = kandy_inner(X_data, tc['f'], tc['grad_f'], degree=3,
                     steps=200, grid=15)
    print(f"    R²(g)={rk['r2_g']:.4f}  R²(h∘g)={rk['r2_hg']:.4f}")

    all_results[tc['label']] = {'tc': tc, 'lasso': rl, 'kandy': rk}


# =====================================================================
# Grid evaluation helper
# =====================================================================

def make_grid(tc, n_pts=200):
    """Build 2D meshgrid for the appropriate slice."""
    ax1_name, ax2_name = tc['slice_axes']
    fix_name, fix_val = tc['slice_fix']

    ranges = {'x': (-20, 20), 'y': (-30, 30), 'z': (0, 50)}
    ax1_range = ranges[ax1_name]
    ax2_range = ranges[ax2_name]

    a1 = np.linspace(*ax1_range, n_pts)
    a2 = np.linspace(*ax2_range, n_pts)
    A1, A2 = np.meshgrid(a1, a2)

    # Build (N_grid, 3) array
    grid = np.zeros((n_pts * n_pts, 3))
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    grid[:, axis_map[ax1_name]] = A1.ravel()
    grid[:, axis_map[ax2_name]] = A2.ravel()
    grid[:, axis_map[fix_name]] = fix_val

    xx = grid[:, 0].reshape(n_pts, n_pts)
    yy = grid[:, 1].reshape(n_pts, n_pts)
    zz = grid[:, 2].reshape(n_pts, n_pts)

    return A1, A2, grid, xx, yy, zz, ax1_name, ax2_name, fix_name, fix_val


def eval_hog_on_grid(result, grid):
    """Evaluate h(g(x)) on a grid using the stored eval_g and outer h."""
    g_grid = result['eval_g'](grid)
    f_grid = eval_h(g_grid, result['outer'])
    return f_grid, g_grid


# =====================================================================
# Big comparison figure: one row per test case, 3 columns
# =====================================================================

def plot_comparison(all_results):
    n_cases = len(all_results)
    fig, axes = plt.subplots(n_cases, 6, figsize=(28, 4.2 * n_cases))

    for row, (label, res) in enumerate(all_results.items()):
        tc = res['tc']
        rl = res['lasso']
        rk = res['kandy']

        A1, A2, grid, xx, yy, zz, ax1, ax2, fix_name, fix_val = make_grid(tc)
        n = A1.shape[0]

        # True f on grid
        F_true = tc['f_grid'](xx, yy, zz)

        # Lasso h∘g on grid
        F_lasso, G_lasso = eval_hog_on_grid(rl, grid)
        F_lasso = F_lasso.reshape(n, n)
        G_lasso = G_lasso.reshape(n, n)

        # KANDy h∘g on grid
        F_kandy, G_kandy = eval_hog_on_grid(rk, grid)
        F_kandy = F_kandy.reshape(n, n)
        G_kandy = G_kandy.reshape(n, n)

        # Shared colorbar range from true f
        vmin, vmax = F_true.min(), F_true.max()
        n_levels = 12

        # --- Col 0: True f surface ---
        ax = axes[row, 0]
        cf = ax.contourf(A1, A2, F_true, levels=n_levels, cmap='RdBu_r',
                         vmin=vmin, vmax=vmax)
        ax.contour(A1, A2, F_true, levels=n_levels, colors='k',
                   linewidths=0.4, alpha=0.5)
        ax.set_title(f'True $f$', fontsize=10)
        ax.set_xlabel(f'${ax1}$')
        ax.set_ylabel(f'${ax2}$')
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

        # --- Col 1: True f level sets (contour only) ---
        ax = axes[row, 1]
        cs = ax.contour(A1, A2, F_true, levels=n_levels, cmap='RdBu_r')
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
        ax.set_title(f'True $f$ level sets', fontsize=10)
        ax.set_xlabel(f'${ax1}$')
        ax.set_ylabel(f'${ax2}$')

        # --- Col 2: Lasso h∘g surface ---
        ax = axes[row, 2]
        cf = ax.contourf(A1, A2, F_lasso, levels=n_levels, cmap='RdBu_r',
                         vmin=vmin, vmax=vmax)
        ax.contour(A1, A2, F_lasso, levels=n_levels, colors='k',
                   linewidths=0.4, alpha=0.5)
        ax.set_title(f'Lasso $h \\circ g$   R²={rl["r2_hg"]:.4f}', fontsize=10)
        ax.set_xlabel(f'${ax1}$')
        ax.set_ylabel(f'${ax2}$')
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

        # --- Col 3: Lasso level sets ---
        ax = axes[row, 3]
        cs = ax.contour(A1, A2, F_lasso, levels=n_levels, cmap='RdBu_r')
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
        # Overlay true level sets in gray for comparison
        ax.contour(A1, A2, F_true, levels=n_levels, colors='gray',
                   linewidths=0.5, linestyles='dashed', alpha=0.6)
        ax.set_title(f'Lasso level sets (gray=true)', fontsize=10)
        ax.set_xlabel(f'${ax1}$')
        ax.set_ylabel(f'${ax2}$')

        # --- Col 4: KANDy h∘g surface ---
        ax = axes[row, 4]
        cf = ax.contourf(A1, A2, F_kandy, levels=n_levels, cmap='RdBu_r',
                         vmin=vmin, vmax=vmax)
        ax.contour(A1, A2, F_kandy, levels=n_levels, colors='k',
                   linewidths=0.4, alpha=0.5)
        ax.set_title(f'KANDy $h \\circ g$   R²={rk["r2_hg"]:.4f}', fontsize=10)
        ax.set_xlabel(f'${ax1}$')
        ax.set_ylabel(f'${ax2}$')
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

        # --- Col 5: KANDy level sets ---
        ax = axes[row, 5]
        cs = ax.contour(A1, A2, F_kandy, levels=n_levels, cmap='RdBu_r')
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
        ax.contour(A1, A2, F_true, levels=n_levels, colors='gray',
                   linewidths=0.5, linestyles='dashed', alpha=0.6)
        ax.set_title(f'KANDy level sets (gray=true)', fontsize=10)
        ax.set_xlabel(f'${ax1}$')
        ax.set_ylabel(f'${ax2}$')

        # Row label on left
        axes[row, 0].annotate(
            tc['name'], xy=(-0.35, 0.5),
            xycoords='axes fraction', fontsize=11, fontweight='bold',
            ha='center', va='center', rotation=90)

    plt.tight_layout(w_pad=1.0, h_pad=1.5)
    plt.savefig(OUTDIR / 'comparison_6col.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTDIR / 'comparison_6col.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTDIR}/comparison_6col.png")


# =====================================================================
# Inner function g comparison (surfaces of g itself, not h∘g)
# =====================================================================

def plot_inner_functions(all_results):
    """Show Lasso g vs KANDy g (the raw inner functions before h)."""
    n_cases = len(all_results)
    fig, axes = plt.subplots(n_cases, 4, figsize=(20, 4.2 * n_cases))

    for row, (label, res) in enumerate(all_results.items()):
        tc = res['tc']
        rl = res['lasso']
        rk = res['kandy']

        A1, A2, grid, xx, yy, zz, ax1, ax2, fix_name, fix_val = make_grid(tc)
        n = A1.shape[0]

        G_lasso = rl['eval_g'](grid).reshape(n, n)
        G_kandy = rk['eval_g'](grid).reshape(n, n)

        # Shared range
        gmin = min(G_lasso.min(), G_kandy.min())
        gmax = max(G_lasso.max(), G_kandy.max())
        n_levels = 12

        # --- Col 0: Lasso g surface ---
        ax = axes[row, 0]
        cf = ax.contourf(A1, A2, G_lasso, levels=n_levels, cmap='viridis')
        ax.contour(A1, A2, G_lasso, levels=n_levels, colors='k',
                   linewidths=0.4, alpha=0.5)
        ax.set_title(f'Lasso $g$   R²(g)={rl["r2_g"]:.4f}', fontsize=10)
        ax.set_xlabel(f'${ax1}$')
        ax.set_ylabel(f'${ax2}$')
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

        # --- Col 1: Lasso g level sets ---
        ax = axes[row, 1]
        cs = ax.contour(A1, A2, G_lasso, levels=n_levels, cmap='viridis')
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
        ax.set_title(f'Lasso $g$ level sets', fontsize=10)
        ax.set_xlabel(f'${ax1}$')
        ax.set_ylabel(f'${ax2}$')

        # --- Col 2: KANDy g surface ---
        ax = axes[row, 2]
        cf = ax.contourf(A1, A2, G_kandy, levels=n_levels, cmap='plasma')
        ax.contour(A1, A2, G_kandy, levels=n_levels, colors='k',
                   linewidths=0.4, alpha=0.5)
        ax.set_title(f'KANDy $g$   R²(g)={rk["r2_g"]:.4f}', fontsize=10)
        ax.set_xlabel(f'${ax1}$')
        ax.set_ylabel(f'${ax2}$')
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

        # --- Col 3: KANDy g level sets ---
        ax = axes[row, 3]
        cs = ax.contour(A1, A2, G_kandy, levels=n_levels, cmap='plasma')
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
        ax.set_title(f'KANDy $g$ level sets', fontsize=10)
        ax.set_xlabel(f'${ax1}$')
        ax.set_ylabel(f'${ax2}$')

        # Row label
        axes[row, 0].annotate(
            tc['name'], xy=(-0.35, 0.5),
            xycoords='axes fraction', fontsize=11, fontweight='bold',
            ha='center', va='center', rotation=90)

    plt.tight_layout(w_pad=1.0, h_pad=1.5)
    plt.savefig(OUTDIR / 'inner_g_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTDIR / 'inner_g_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTDIR}/inner_g_comparison.png")


# =====================================================================
# Outer function h comparison
# =====================================================================

def plot_outer_functions(all_results):
    """Show recovered h curves for Lasso and KANDy side by side."""
    n_cases = len(all_results)
    fig, axes = plt.subplots(1, n_cases, figsize=(4.5 * n_cases, 4))

    for col, (label, res) in enumerate(all_results.items()):
        tc = res['tc']
        ol = res['lasso']['outer']
        ok = res['kandy']['outer']

        ax = axes[col]

        # Lasso h curve
        ax.plot(ol['g_plot'], ol['h_fit_plot'], 'b-', lw=2.5,
                label=f'Lasso h  (R²={res["lasso"]["r2_hg"]:.4f})')

        # KANDy h curve
        ax.plot(ok['g_plot'], ok['h_fit_plot'], 'r--', lw=2.5,
                label=f'KANDy h  (R²={res["kandy"]["r2_hg"]:.4f})')

        # Scatter of (g, f) for reference
        idx = np.random.permutation(len(res['lasso']['g_vals']))[:2000]
        ax.scatter(res['lasso']['g_vals'][idx], tc['f'][idx],
                   s=1, alpha=0.15, color='gray', zorder=0)

        ax.set_xlabel('$g$', fontsize=11)
        ax.set_ylabel('$h(g) \\approx f$', fontsize=11)
        ax.set_title(f'{tc["name"]}\n{tc["true_g_str"]}, {tc["true_h_str"]}',
                     fontsize=10)
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(OUTDIR / 'outer_h_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTDIR / 'outer_h_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTDIR}/outer_h_comparison.png")


# =====================================================================
# Main
# =====================================================================

print("\n" + "=" * 60)
print("  SUMMARY TABLE")
print("=" * 60)
print(f"{'Case':<8} | {'Lasso R²(g)':>12} {'Lasso R²(h∘g)':>14} | "
      f"{'KANDy R²(g)':>12} {'KANDy R²(h∘g)':>14}")
print("-" * 75)
for label, res in all_results.items():
    rl = res['lasso']
    rk = res['kandy']
    print(f"{label:<8} | {rl['r2_g']:12.4f} {rl['r2_hg']:14.4f} | "
          f"{rk['r2_g']:12.4f} {rk['r2_hg']:14.4f}")

print("\nGenerating plots...")
plot_comparison(all_results)
plot_inner_functions(all_results)
plot_outer_functions(all_results)
print(f"\nAll plots saved to {OUTDIR}/")
