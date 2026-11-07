"""
Full benchmark: Lasso h∘g vs KAN h∘g on all test cases.

Runs recover_inner_function (Lasso) and recover_inner_function_kan
independently on each case, applies the same gradient outer h to both,
reports R²(g), R²(h∘g), used_affine, and wall time.

Test cases split into three tiers:
  A. Polynomial inner functions (Lasso should win)
  B. Non-polynomial univariate inner functions (KAN should win)
  C. Non-polynomial bivariate inner functions (both struggle)

Data: N=50,000 uniform on [-5,5]³, degree-3 monomial dictionary.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from manifold_analysis import (
    recover_inner_function,
    recover_inner_function_kan,
    recover_outer_function,
    build_polynomial_dict,
    eval_polynomial,
    grad_polynomial,
    eval_h,
)

OUTDIR = Path('results/benchmark_full_pipeline')
OUTDIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

DEGREE = 3
N = 50_000


# =====================================================================
# Data
# =====================================================================

x = np.random.uniform(-5, 5, N)
y = np.random.uniform(-5, 5, N)
z = np.random.uniform(-5, 5, N)


# =====================================================================
# Test cases
# =====================================================================

TEST_CASES = [
    # --- A: Polynomial inner (Lasso territory) ---
    {
        'name': r'cos(x/5 + y/10)',
        'tier': 'A',
        'f': np.cos(x/5 + y/10),
        'grad_f': np.column_stack([
            -np.sin(x/5 + y/10) / 5,
            -np.sin(x/5 + y/10) / 10,
            np.zeros(N)]),
        'true_g': 'x/5 + y/10', 'true_h': 'cos',
        'f_grid': lambda xx, yy, zz: np.cos(xx/5 + yy/10),
        'axes': ('x', 'y'), 'fix': ('z', 0.0),
    },
    {
        'name': r'exp(-(x²+y²)/10)',
        'tier': 'A',
        'f': np.exp(-(x**2 + y**2) / 10),
        'grad_f': np.column_stack([
            np.exp(-(x**2 + y**2) / 10) * (-x / 5),
            np.exp(-(x**2 + y**2) / 10) * (-y / 5),
            np.zeros(N)]),
        'true_g': '-(x²+y²)/10', 'true_h': 'exp',
        'f_grid': lambda xx, yy, zz: np.exp(-(xx**2 + yy**2) / 10),
        'axes': ('x', 'y'), 'fix': ('z', 0.0),
    },
    {
        'name': r'tanh(z - 2)',
        'tier': 'A',
        'f': np.tanh(z - 2),
        'grad_f': np.column_stack([
            np.zeros(N),
            np.zeros(N),
            1 - np.tanh(z - 2)**2]),
        'true_g': 'z - 2', 'true_h': 'tanh',
        'f_grid': lambda xx, yy, zz: np.tanh(zz - 2),
        'axes': ('x', 'z'), 'fix': ('y', 0.0),
    },
    {
        'name': r'(xy/5)³',
        'tier': 'A',
        'f': (x * y / 5)**3,
        'grad_f': np.column_stack([
            3 * (x * y / 5)**2 * y / 5,
            3 * (x * y / 5)**2 * x / 5,
            np.zeros(N)]),
        'true_g': 'xy/5', 'true_h': '(·)³',
        'f_grid': lambda xx, yy, zz: (xx * yy / 5)**3,
        'axes': ('x', 'y'), 'fix': ('z', 0.0),
    },
    # --- B: Non-polynomial univariate inner (KAN territory) ---
    {
        'name': r'exp(-sin(x)²)',
        'tier': 'B',
        'f': np.exp(-np.sin(x)**2),
        'grad_f': np.column_stack([
            np.exp(-np.sin(x)**2) * (-2 * np.sin(x) * np.cos(x)),
            np.zeros(N),
            np.zeros(N)]),
        'true_g': 'sin(x)²', 'true_h': 'exp(-·)',
        'f_grid': lambda xx, yy, zz: np.exp(-np.sin(xx)**2),
        'axes': ('x', 'y'), 'fix': ('z', 0.0),
    },
    {
        'name': r'tanh(x + cos(z))',
        'tier': 'B',
        'f': np.tanh(x + np.cos(z)),
        'grad_f': np.column_stack([
            1 - np.tanh(x + np.cos(z))**2,
            np.zeros(N),
            -(1 - np.tanh(x + np.cos(z))**2) * np.sin(z)]),
        'true_g': 'x + cos(z)', 'true_h': 'tanh',
        'f_grid': lambda xx, yy, zz: np.tanh(xx + np.cos(zz)),
        'axes': ('x', 'z'), 'fix': ('y', 0.0),
    },
    {
        'name': r'sin(x) · z',
        'tier': 'B',
        'f': np.sin(x) * z,
        'grad_f': np.column_stack([
            np.cos(x) * z,
            np.zeros(N),
            np.sin(x)]),
        'true_g': 'sin(x)·z', 'true_h': 'identity',
        'f_grid': lambda xx, yy, zz: np.sin(xx) * zz,
        'axes': ('x', 'z'), 'fix': ('y', 0.0),
    },
    {
        'name': r'cos(y)²',
        'tier': 'B',
        'f': np.cos(y)**2,
        'grad_f': np.column_stack([
            np.zeros(N),
            -2 * np.cos(y) * np.sin(y),
            np.zeros(N)]),
        'true_g': 'cos(y)', 'true_h': '(·)²',
        'f_grid': lambda xx, yy, zz: np.cos(yy)**2,
        'axes': ('x', 'y'), 'fix': ('z', 0.0),
    },
    # --- C: Non-polynomial bivariate inner (hard for both) ---
    {
        'name': r'sin(sin(y) + x²)',
        'tier': 'C',
        'f': np.sin(np.sin(y) + x**2),
        'grad_f': np.column_stack([
            np.cos(np.sin(y) + x**2) * 2 * x,
            np.cos(np.sin(y) + x**2) * np.cos(y),
            np.zeros(N)]),
        'true_g': 'sin(y) + x²', 'true_h': 'sin',
        'f_grid': lambda xx, yy, zz: np.sin(np.sin(yy) + xx**2),
        'axes': ('x', 'y'), 'fix': ('z', 0.0),
    },
    {
        'name': r'sin(x · cos(y))',
        'tier': 'C',
        'f': np.sin(x * np.cos(y)),
        'grad_f': np.column_stack([
            np.cos(x * np.cos(y)) * np.cos(y),
            np.cos(x * np.cos(y)) * (-x * np.sin(y)),
            np.zeros(N)]),
        'true_g': 'x·cos(y)', 'true_h': 'sin',
        'f_grid': lambda xx, yy, zz: np.sin(xx * np.cos(yy)),
        'axes': ('x', 'y'), 'fix': ('z', 0.0),
    },
    {
        'name': r'exp(-sin(x)·sin(y))',
        'tier': 'C',
        'f': np.exp(-np.sin(x) * np.sin(y)),
        'grad_f': np.column_stack([
            np.exp(-np.sin(x) * np.sin(y)) * (-np.cos(x) * np.sin(y)),
            np.exp(-np.sin(x) * np.sin(y)) * (-np.sin(x) * np.cos(y)),
            np.zeros(N)]),
        'true_g': 'sin(x)·sin(y)', 'true_h': 'exp(-·)',
        'f_grid': lambda xx, yy, zz: np.exp(-np.sin(xx) * np.sin(yy)),
        'axes': ('x', 'y'), 'fix': ('z', 0.0),
    },
]


# =====================================================================
# Run one method
# =====================================================================

def run_lasso(tc):
    t0 = time.time()
    inner = recover_inner_function(x, y, z, tc['f'],
                                    max_degree=DEGREE, alpha=1e-5)
    g_vals = eval_polynomial(x, y, z, inner['coeffs'], inner['powers'])
    grad_g = grad_polynomial(x, y, z, inner['coeffs'], inner['powers'])
    outer = recover_outer_function(tc['grad_f'], grad_g, g_vals, tc['f'],
                                    poly_degree=4)
    f_hat = eval_h(g_vals, outer)
    ss_res = np.sum((tc['f'] - f_hat)**2)
    ss_tot = np.sum((tc['f'] - tc['f'].mean())**2)
    r2_hg = 1.0 - ss_res / max(ss_tot, 1e-12)
    elapsed = time.time() - t0

    # Grid evaluator for plots
    def eval_g_grid(X_grid):
        return eval_polynomial(X_grid[:, 0], X_grid[:, 1], X_grid[:, 2],
                               inner['coeffs'], inner['powers'])

    return {
        'r2_g': inner['r2'],
        'r2_hg': r2_hg,
        'affine': outer['used_affine'],
        'n_active': len(inner['active']),
        'active': inner['active_names'],
        'time': elapsed,
        'g_vals': g_vals,
        'f_hat': f_hat,
        'outer': outer,
        'eval_g': eval_g_grid,
    }


def run_kan(tc):
    t0 = time.time()
    inner = recover_inner_function_kan(x, y, z, tc['f'],
                                        max_degree=DEGREE,
                                        kan_steps=500, kan_grid=20,
                                        kan_lamb=0.0, max_train=15000)
    g_vals = inner['g_vals']
    grad_g = inner['grad_g']
    outer = recover_outer_function(tc['grad_f'],
                                    (grad_g[:, 0], grad_g[:, 1], grad_g[:, 2]),
                                    g_vals, tc['f'], poly_degree=4)
    f_hat = eval_h(g_vals, outer)
    ss_res = np.sum((tc['f'] - f_hat)**2)
    ss_tot = np.sum((tc['f'] - tc['f'].mean())**2)
    r2_hg = 1.0 - ss_res / max(ss_tot, 1e-12)
    elapsed = time.time() - t0
    return {
        'r2_g': inner['r2'],
        'r2_hg': r2_hg,
        'affine': outer['used_affine'],
        'n_active': len(inner['active']),
        'active': inner['active_names'],
        'time': elapsed,
        'g_vals': g_vals,
        'f_hat': f_hat,
        'outer': outer,
        'eval_g': inner['eval_g'],
    }


# =====================================================================
# Run all
# =====================================================================

results = []

for i, tc in enumerate(TEST_CASES):
    print(f"\n[{i+1}/{len(TEST_CASES)}] {tc['name']}  (tier {tc['tier']})")

    rl = run_lasso(tc)
    print(f"  Lasso:  R²(g)={rl['r2_g']:8.4f}  R²(h∘g)={rl['r2_hg']:8.4f}  "
          f"affine={rl['affine']}  time={rl['time']:.1f}s")

    rk = run_kan(tc)
    print(f"  KAN:    R²(g)={rk['r2_g']:8.4f}  R²(h∘g)={rk['r2_hg']:8.4f}  "
          f"affine={rk['affine']}  time={rk['time']:.1f}s")

    winner = 'Lasso' if rl['r2_hg'] > rk['r2_hg'] + 0.005 else \
             'KAN' if rk['r2_hg'] > rl['r2_hg'] + 0.005 else 'tie'
    print(f"  → {winner}")

    results.append({
        'name': tc['name'], 'tier': tc['tier'],
        'true_g': tc['true_g'], 'true_h': tc['true_h'],
        'lasso': rl, 'kan': rk, 'winner': winner,
        'tc': tc,
    })


# =====================================================================
# Summary table
# =====================================================================

print("\n" + "=" * 105)
print("FULL BENCHMARK: Lasso h∘g vs KAN h∘g  (degree-3 dict, [-5,5]³, N=50k)")
print("=" * 105)
print(f"{'Case':<25} {'Tier':>4} | {'Lasso R²(g)':>11} {'R²(h∘g)':>9} {'aff':>4} | "
      f"{'KAN R²(g)':>11} {'R²(h∘g)':>9} {'aff':>4} | {'Winner':>7}")
print("-" * 105)

tier_scores = {'A': [0, 0], 'B': [0, 0], 'C': [0, 0]}

for r in results:
    rl, rk = r['lasso'], r['kan']
    a_l = 'Y' if rl['affine'] else 'N'
    a_k = 'Y' if rk['affine'] else 'N'
    print(f"{r['name']:<25} {r['tier']:>4} | "
          f"{rl['r2_g']:11.4f} {rl['r2_hg']:9.4f} {a_l:>4} | "
          f"{rk['r2_g']:11.4f} {rk['r2_hg']:9.4f} {a_k:>4} | "
          f"{r['winner']:>7}")
    if r['winner'] == 'Lasso':
        tier_scores[r['tier']][0] += 1
    elif r['winner'] == 'KAN':
        tier_scores[r['tier']][1] += 1

print("-" * 105)
for tier, (l, k) in tier_scores.items():
    total = sum(1 for r in results if r['tier'] == tier)
    ties = total - l - k
    print(f"  Tier {tier}: Lasso {l}, KAN {k}, tie {ties}")

lasso_mean = np.mean([r['lasso']['r2_hg'] for r in results])
kan_mean = np.mean([r['kan']['r2_hg'] for r in results])
print(f"\n  Overall mean R²(h∘g): Lasso={lasso_mean:.4f}  KAN={kan_mean:.4f}")


# =====================================================================
# Bar chart
# =====================================================================

fig, ax = plt.subplots(figsize=(14, 5))
names = [r['name'] for r in results]
tiers = [r['tier'] for r in results]
lr2 = [r['lasso']['r2_hg'] for r in results]
kr2 = [r['kan']['r2_hg'] for r in results]

xpos = np.arange(len(results))
w = 0.35

bars_l = ax.bar(xpos - w/2, lr2, w, label='Lasso h∘g', color='steelblue')
bars_k = ax.bar(xpos + w/2, kr2, w, label='KAN h∘g', color='darkorange')

ax.set_xticks(xpos)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('R²(h∘g)')
ax.set_title('Full Pipeline Benchmark: Lasso vs KAN Inner Function Recovery')
ax.legend()
ax.set_ylim(-0.1, 1.1)
ax.axhline(1.0, color='gray', ls=':', lw=0.5)

# Tier separators
tier_bounds = []
for i in range(1, len(tiers)):
    if tiers[i] != tiers[i-1]:
        tier_bounds.append(i - 0.5)
for b in tier_bounds:
    ax.axvline(b, color='gray', ls='--', lw=0.8, alpha=0.5)

# Tier labels
tier_centers = {}
for i, t in enumerate(tiers):
    tier_centers.setdefault(t, []).append(i)
for t, indices in tier_centers.items():
    cx = np.mean(indices)
    labels = {'A': 'Polynomial\ninner', 'B': 'Non-poly\nunivariate', 'C': 'Non-poly\nbivariate'}
    ax.text(cx, -0.07, f'Tier {t}: {labels[t]}', ha='center', fontsize=7,
            transform=ax.get_xaxis_transform())

plt.tight_layout()
plt.savefig(OUTDIR / 'full_benchmark.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTDIR / 'full_benchmark.pdf', bbox_inches='tight')
plt.close()
print(f"\nSaved {OUTDIR}/full_benchmark.png")


# =====================================================================
# Per-case 7-column plots: true f, true levels, Lasso h∘g, Lasso levels,
#                          KAN h∘g, KAN levels, outer h curves
# =====================================================================

def make_grid(tc, n_pts=200):
    ax1_name, ax2_name = tc['axes']
    fix_name, fix_val = tc['fix']
    R = 5
    a1 = np.linspace(-R, R, n_pts)
    a2 = np.linspace(-R, R, n_pts)
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


print("\nGenerating per-case level-set plots...")

for i, r in enumerate(results):
    tc = r['tc']
    rl, rk = r['lasso'], r['kan']
    A1, A2, grid, xx, yy, zz, ax1, ax2 = make_grid(tc)
    n = A1.shape[0]

    F_true = tc['f_grid'](xx, yy, zz)

    # Evaluate h∘g on grid for both methods
    g_l_grid = rl['eval_g'](grid)
    F_lasso = eval_h(g_l_grid, rl['outer']).reshape(n, n)

    g_k_grid = rk['eval_g'](grid[:, 0], grid[:, 1], grid[:, 2])
    F_kan = eval_h(g_k_grid, rk['outer']).reshape(n, n)

    vmin, vmax = F_true.min(), F_true.max()
    n_lev = 12

    fig, axes_row = plt.subplots(1, 7, figsize=(32, 4))

    # Col 0: true f
    ax = axes_row[0]
    cf = ax.contourf(A1, A2, F_true, levels=n_lev, cmap='RdBu_r',
                     vmin=vmin, vmax=vmax)
    ax.contour(A1, A2, F_true, levels=n_lev, colors='k', linewidths=0.4, alpha=0.5)
    ax.set_title('True $f$', fontsize=10)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # Col 1: true level sets
    ax = axes_row[1]
    cs = ax.contour(A1, A2, F_true, levels=n_lev, cmap='RdBu_r')
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
    ax.set_title('True level sets', fontsize=10)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')

    # Col 2: Lasso h∘g
    ax = axes_row[2]
    cf = ax.contourf(A1, A2, F_lasso, levels=n_lev, cmap='RdBu_r',
                     vmin=vmin, vmax=vmax)
    ax.contour(A1, A2, F_lasso, levels=n_lev, colors='k', linewidths=0.4, alpha=0.5)
    ax.set_title(f'Lasso $h\\circ g$  R²={rl["r2_hg"]:.4f}', fontsize=9)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # Col 3: Lasso level sets + true (gray)
    ax = axes_row[3]
    cs = ax.contour(A1, A2, F_lasso, levels=n_lev, cmap='RdBu_r')
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
    ax.contour(A1, A2, F_true, levels=n_lev, colors='gray',
               linewidths=0.5, linestyles='dashed', alpha=0.6)
    ax.set_title('Lasso levels (gray=true)', fontsize=9)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')

    # Col 4: KAN h∘g
    ax = axes_row[4]
    cf = ax.contourf(A1, A2, F_kan, levels=n_lev, cmap='RdBu_r',
                     vmin=vmin, vmax=vmax)
    ax.contour(A1, A2, F_kan, levels=n_lev, colors='k', linewidths=0.4, alpha=0.5)
    ax.set_title(f'KAN $h\\circ g$  R²={rk["r2_hg"]:.4f}', fontsize=9)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # Col 5: KAN level sets + true (gray)
    ax = axes_row[5]
    cs = ax.contour(A1, A2, F_kan, levels=n_lev, cmap='RdBu_r')
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
    ax.contour(A1, A2, F_true, levels=n_lev, colors='gray',
               linewidths=0.5, linestyles='dashed', alpha=0.6)
    ax.set_title('KAN levels (gray=true)', fontsize=9)
    ax.set_xlabel(f'${ax1}$'); ax.set_ylabel(f'${ax2}$')

    # Col 6: outer h curves
    ax = axes_row[6]
    # Lasso h curve
    ax.plot(rl['outer']['g_plot'], rl['outer']['h_fit_plot'],
            'b-', lw=2.5, label=f'Lasso h ({"aff" if rl["affine"] else "nonlin"})')
    # KAN h curve
    ax.plot(rk['outer']['g_plot'], rk['outer']['h_fit_plot'],
            'r--', lw=2.5, label=f'KAN h ({"aff" if rk["affine"] else "nonlin"})')
    # Scatter: (g, f) for Lasso
    idx_s = np.random.permutation(len(rl['g_vals']))[:2000]
    ax.scatter(rl['g_vals'][idx_s], tc['f'][idx_s], s=0.5, alpha=0.1,
               color='steelblue', zorder=0, label='Lasso g')
    # Scatter: (g, f) for KAN
    idx_s2 = np.random.permutation(len(rk['g_vals']))[:2000]
    ax.scatter(rk['g_vals'][idx_s2], tc['f'][idx_s2], s=0.5, alpha=0.1,
               color='darkorange', zorder=0, label='KAN g')
    ax.set_xlabel('$g$'); ax.set_ylabel('$h(g) \\approx f$')
    ax.set_title('Outer $h$', fontsize=10)
    ax.legend(fontsize=6, loc='best')

    fig.suptitle(f'{tc["name"]}  (Tier {tc["tier"]}: g={tc["true_g"]}, h={tc["true_h"]})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    fname = f'case_{i+1:02d}_{tc["tier"]}'
    plt.savefig(OUTDIR / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTDIR / f'{fname}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}.png")
