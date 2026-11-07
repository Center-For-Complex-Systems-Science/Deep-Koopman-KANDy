"""
Benchmark: Lasso vs KANDy when the true function involves sin of a monomial.

Both methods get the SAME degree-3 monomial dictionary.
Both use the SAME gradient-based outer h recovery.

Key test: f = sin(xy/200).  The true inner function is g = xy (a monomial
in the dictionary), and h = sin.  Lasso must approximate sin via polynomials;
KANDy's spline edge on the xy feature can learn sin directly.

Test cases:
  1. f = sin(xy/200)           — g = xy/200 (bilinear), h = sin
  2. f = exp(-(x²+y²)/200)    — g = -(x²+y²)/200,     h = exp   [baseline]
  3. f = tanh(z/10 - 2)        — g = z/10 - 2,          h = tanh  [baseline]
  4. f = (xy/100)³             — g = xy/100,             h = cube  [baseline]

Data: N=50,000 uniform on [-20,20] × [-30,30] × [0,50]
Dictionary: degree-3 monomials (19 features)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations_with_replacement
from collections import Counter

from manifold_analysis import (
    recover_inner_function,
    recover_outer_function,
    eval_polynomial,
    grad_polynomial,
    eval_h,
)
from models import _import_pykan
PyKAN = _import_pykan()

OUTDIR = Path('results/benchmark_sine')
OUTDIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

# =====================================================================
# Monomial dictionary
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
# KANDy inner + gradient outer
# =====================================================================

def run_kandy(X, f_vals, grad_f, degree=3, steps=300, grid=20,
              max_train=10000, lamb=0.001):
    theta, names = poly_lift(X, degree)
    m = theta.shape[1]

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
    kan.fit(dataset, opt='LBFGS', steps=steps, lamb=lamb)

    # Predict on ALL data
    with torch.no_grad():
        g_vals = kan(_t(theta)).numpy().flatten()

    # Gradient via finite differences
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

    # Gradient outer h
    grad_g_tuple = (grad_g[:, 0], grad_g[:, 1], grad_g[:, 2])
    outer = recover_outer_function(grad_f, grad_g_tuple, g_vals, f_vals,
                                    poly_degree=4)
    f_hat = eval_h(g_vals, outer)
    ss_res2 = np.sum((f_vals - f_hat)**2)
    r2_hg = 1.0 - ss_res2 / max(ss_tot, 1e-12)

    def eval_g_grid(X_grid):
        tg, _ = poly_lift(X_grid, degree)
        with torch.no_grad():
            return kan(_t(tg)).numpy().flatten()

    return {
        'g_vals': g_vals, 'r2_g': r2_g, 'r2_hg': r2_hg,
        'outer': outer, 'f_hat': f_hat, 'eval_g': eval_g_grid,
    }


# =====================================================================
# Lasso inner + gradient outer
# =====================================================================

def run_lasso(X, f_vals, grad_f, degree=3, alpha=1e-5):
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

    def eval_g_grid(X_grid):
        return eval_polynomial(X_grid[:, 0], X_grid[:, 1], X_grid[:, 2],
                               inner['coeffs'], inner['powers'])

    return {
        'g_vals': g_vals, 'r2_g': inner['r2'], 'r2_hg': r2_hg,
        'outer': outer, 'f_hat': f_hat, 'eval_g': eval_g_grid,
        'inner': inner,
    }


# =====================================================================
# Data + test cases
# =====================================================================

N = 50_000
x = np.random.uniform(-20, 20, N)
y = np.random.uniform(-30, 30, N)
z = np.random.uniform(0, 50, N)
X = np.column_stack([x, y, z])

TEST_CASES = [
    {
        'name': r'$\sin(xy/200)$',
        'label': 'sin_xy',
        'f': np.sin(x * y / 200),
        'grad_f': np.column_stack([
            np.cos(x * y / 200) * y / 200,
            np.cos(x * y / 200) * x / 200,
            np.zeros(N),
        ]),
        'f_grid': lambda xx, yy, zz: np.sin(xx * yy / 200),
        'slice_axes': ('x', 'y'), 'slice_fix': ('z', 25.0),
        'true_g': r'$xy/200$', 'true_h': r'$\sin$',
    },
    {
        'name': r'$\exp(-(x^2+y^2)/200)$',
        'label': 'exp',
        'f': np.exp(-(x**2 + y**2) / 200),
        'grad_f': np.column_stack([
            np.exp(-(x**2 + y**2) / 200) * (-x / 100),
            np.exp(-(x**2 + y**2) / 200) * (-y / 100),
            np.zeros(N),
        ]),
        'f_grid': lambda xx, yy, zz: np.exp(-(xx**2 + yy**2) / 200),
        'slice_axes': ('x', 'y'), 'slice_fix': ('z', 25.0),
        'true_g': r'$-(x^2+y^2)/200$', 'true_h': r'$\exp$',
    },
    {
        'name': r'$\tanh(z/10 - 2)$',
        'label': 'tanh',
        'f': np.tanh(z / 10 - 2),
        'grad_f': np.column_stack([
            np.zeros(N),
            np.zeros(N),
            (1 - np.tanh(z / 10 - 2)**2) / 10,
        ]),
        'f_grid': lambda xx, yy, zz: np.tanh(zz / 10 - 2),
        'slice_axes': ('x', 'z'), 'slice_fix': ('y', 0.0),
        'true_g': r'$z/10-2$', 'true_h': r'$\tanh$',
    },
    {
        'name': r'$(xy/100)^3$',
        'label': 'cube',
        'f': (x * y / 100)**3,
        'grad_f': np.column_stack([
            3 * (x * y / 100)**2 * y / 100,
            3 * (x * y / 100)**2 * x / 100,
            np.zeros(N),
        ]),
        'f_grid': lambda xx, yy, zz: (xx * yy / 100)**3,
        'slice_axes': ('x', 'y'), 'slice_fix': ('z', 25.0),
        'true_g': r'$xy/100$', 'true_h': r'$(\cdot)^3$',
    },
]


# =====================================================================
# Run
# =====================================================================

all_results = {}

for tc in TEST_CASES:
    print(f"\n{'='*60}")
    print(f"  {tc['label']}: {tc['name']}")
    print(f"{'='*60}")

    rl = run_lasso(X, tc['f'], tc['grad_f'])
    print(f"  Lasso:  R²(g)={rl['r2_g']:.4f}  R²(h∘g)={rl['r2_hg']:.4f}")
    if 'inner' in rl:
        active = list(zip(rl['inner']['active_names'],
                          rl['inner']['active_coeffs']))
        top = sorted(active, key=lambda t: abs(t[1]), reverse=True)[:5]
        print(f"          top terms: {', '.join(f'{c:+.4f}·{n}' for n,c in top)}")

    rk = run_kandy(X, tc['f'], tc['grad_f'])
    print(f"  KANDy:  R²(g)={rk['r2_g']:.4f}  R²(h∘g)={rk['r2_hg']:.4f}")

    all_results[tc['label']] = {'tc': tc, 'lasso': rl, 'kandy': rk}


# =====================================================================
# Summary
# =====================================================================

print("\n" + "=" * 80)
print("SUMMARY: Lasso vs KANDy (degree-3 monomial dictionary)")
print("=" * 80)
print(f"{'Case':<12} | {'Lasso R²(g)':>12} {'R²(h∘g)':>10} | "
      f"{'KANDy R²(g)':>12} {'R²(h∘g)':>10} | {'winner':>8}")
print("-" * 80)
for label, res in all_results.items():
    rl, rk = res['lasso'], res['kandy']
    if rk['r2_hg'] > rl['r2_hg'] + 0.005:
        w = 'KANDy'
    elif rl['r2_hg'] > rk['r2_hg'] + 0.005:
        w = 'Lasso'
    else:
        w = 'tie'
    print(f"{label:<12} | {rl['r2_g']:12.4f} {rl['r2_hg']:10.4f} | "
          f"{rk['r2_g']:12.4f} {rk['r2_hg']:10.4f} | {w:>8}")


# =====================================================================
# Plots: 4 rows × 7 cols
#   col 0: true f surface
#   col 1: true f level sets
#   col 2: Lasso h∘g surface
#   col 3: Lasso h∘g level sets (gray = true)
#   col 4: KANDy h∘g surface
#   col 5: KANDy h∘g level sets (gray = true)
#   col 6: outer h curves (both methods + data scatter)
# =====================================================================

def make_grid(tc, n_pts=200):
    ax1_name, ax2_name = tc['slice_axes']
    fix_name, fix_val = tc['slice_fix']
    ranges = {'x': (-20, 20), 'y': (-30, 30), 'z': (0, 50)}
    a1 = np.linspace(*ranges[ax1_name], n_pts)
    a2 = np.linspace(*ranges[ax2_name], n_pts)
    A1, A2 = np.meshgrid(a1, a2)
    grid = np.zeros((n_pts * n_pts, 3))
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    grid[:, axis_map[ax1_name]] = A1.ravel()
    grid[:, axis_map[ax2_name]] = A2.ravel()
    grid[:, axis_map[fix_name]] = fix_val
    xx = grid[:, 0].reshape(n_pts, n_pts)
    yy = grid[:, 1].reshape(n_pts, n_pts)
    zz = grid[:, 2].reshape(n_pts, n_pts)
    return A1, A2, grid, xx, yy, zz, ax1_name, ax2_name


n_cases = len(all_results)
fig, axes = plt.subplots(n_cases, 7, figsize=(32, 4.2 * n_cases))

for row, (label, res) in enumerate(all_results.items()):
    tc = res['tc']
    rl, rk = res['lasso'], res['kandy']
    A1, A2, grid, xx, yy, zz, ax1, ax2 = make_grid(tc)
    n = A1.shape[0]

    F_true = tc['f_grid'](xx, yy, zz)

    F_lasso = eval_h(rl['eval_g'](grid), rl['outer']).reshape(n, n)
    F_kandy = eval_h(rk['eval_g'](grid), rk['outer']).reshape(n, n)

    vmin, vmax = F_true.min(), F_true.max()
    n_lev = 12

    # Col 0: true f
    ax = axes[row, 0]
    cf = ax.contourf(A1, A2, F_true, levels=n_lev, cmap='RdBu_r',
                     vmin=vmin, vmax=vmax)
    ax.contour(A1, A2, F_true, levels=n_lev, colors='k', linewidths=0.4, alpha=0.5)
    ax.set_title('True $f$', fontsize=10)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # Col 1: true f level sets
    ax = axes[row, 1]
    cs = ax.contour(A1, A2, F_true, levels=n_lev, cmap='RdBu_r')
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
    ax.set_title('True level sets', fontsize=10)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')

    # Col 2: Lasso h∘g
    ax = axes[row, 2]
    cf = ax.contourf(A1, A2, F_lasso, levels=n_lev, cmap='RdBu_r',
                     vmin=vmin, vmax=vmax)
    ax.contour(A1, A2, F_lasso, levels=n_lev, colors='k', linewidths=0.4, alpha=0.5)
    ax.set_title(f'Lasso $h\\circ g$  R²={rl["r2_hg"]:.4f}', fontsize=10)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # Col 3: Lasso level sets + true (gray dashed)
    ax = axes[row, 3]
    cs = ax.contour(A1, A2, F_lasso, levels=n_lev, cmap='RdBu_r')
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
    ax.contour(A1, A2, F_true, levels=n_lev, colors='gray',
               linewidths=0.5, linestyles='dashed', alpha=0.6)
    ax.set_title('Lasso levels (gray=true)', fontsize=10)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')

    # Col 4: KANDy h∘g
    ax = axes[row, 4]
    cf = ax.contourf(A1, A2, F_kandy, levels=n_lev, cmap='RdBu_r',
                     vmin=vmin, vmax=vmax)
    ax.contour(A1, A2, F_kandy, levels=n_lev, colors='k', linewidths=0.4, alpha=0.5)
    ax.set_title(f'KANDy $h\\circ g$  R²={rk["r2_hg"]:.4f}', fontsize=10)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # Col 5: KANDy level sets + true
    ax = axes[row, 5]
    cs = ax.contour(A1, A2, F_kandy, levels=n_lev, cmap='RdBu_r')
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
    ax.contour(A1, A2, F_true, levels=n_lev, colors='gray',
               linewidths=0.5, linestyles='dashed', alpha=0.6)
    ax.set_title('KANDy levels (gray=true)', fontsize=10)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')

    # Col 6: outer h curves
    ax = axes[row, 6]
    idx_s = np.random.permutation(len(rl['g_vals']))[:3000]
    ax.scatter(rl['g_vals'][idx_s], tc['f'][idx_s], s=0.5, alpha=0.1,
               color='gray', zorder=0, label='data (Lasso g)')
    ax.plot(rl['outer']['g_plot'], rl['outer']['h_fit_plot'],
            'b-', lw=2.5, label=f'Lasso h (R²={rl["r2_hg"]:.3f})')
    ax.plot(rk['outer']['g_plot'], rk['outer']['h_fit_plot'],
            'r--', lw=2.5, label=f'KANDy h (R²={rk["r2_hg"]:.3f})')
    ax.set_xlabel('$g$'); ax.set_ylabel('$h(g)$')
    ax.set_title(f'Outer $h$: {tc["true_h"]}', fontsize=10)
    ax.legend(fontsize=7, loc='best')

    # Row label
    axes[row, 0].annotate(
        f'{tc["name"]}\n{tc["true_g"]}, {tc["true_h"]}',
        xy=(-0.45, 0.5), xycoords='axes fraction',
        fontsize=10, fontweight='bold', ha='center', va='center', rotation=90)

plt.tight_layout(w_pad=0.8, h_pad=1.5)
plt.savefig(OUTDIR / 'sine_benchmark.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTDIR / 'sine_benchmark.pdf', bbox_inches='tight')
plt.close()
print(f"\nSaved {OUTDIR}/sine_benchmark.png")
