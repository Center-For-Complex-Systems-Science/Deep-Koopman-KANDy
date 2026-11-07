"""
Reproduce the paper's h∘g decomposition results.

Data: N=50,000 uniform on [-20,20] × [-30,30] × [0,50]
Dictionary: degree-3 polynomial
Lasso: alpha=1e-5

Test cases:
  1. f = cos(x/20 + y/40),      g = x/20 + y/40 (linear),    h = cos
  2. f = exp(-(x²+y²)/200),     g = -(x²+y²)/200 (quadratic), h = exp
  3. f = tanh(z/10 - 2),        g = z/10 - 2 (linear),        h = tanh
  4. f = (xy/100)³,             g = xy/100 (bilinear),         h = (·)³
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from manifold_analysis import (
    recover_inner_function,
    recover_outer_function,
    eval_polynomial,
    grad_polynomial,
    eval_h,
    build_polynomial_dict,
)

OUTDIR = Path('results/reproduce_paper_hog')
OUTDIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

# =====================================================================
# Data
# =====================================================================
N = 50_000
x = np.random.uniform(-20, 20, N)
y = np.random.uniform(-30, 30, N)
z = np.random.uniform(0, 50, N)

# =====================================================================
# Test cases with analytic gradients
# =====================================================================
TEST_CASES = [
    {
        'name': r'$\cos(x/20 + y/40)$',
        'label': 'cos',
        'f': np.cos(x/20 + y/40),
        'grad_f': np.column_stack([
            -np.sin(x/20 + y/40) / 20,
            -np.sin(x/20 + y/40) / 40,
            np.zeros(N),
        ]),
        'true_g': 'x/20 + y/40',
        'true_h': 'cos',
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
        'true_g': '-(x²+y²)/200',
        'true_h': 'exp',
    },
    {
        'name': r'$\tanh(z/10 - 2)$',
        'label': 'tanh',
        'f': np.tanh(z/10 - 2),
        'grad_f': np.column_stack([
            np.zeros(N),
            np.zeros(N),
            (1 - np.tanh(z/10 - 2)**2) / 10,
        ]),
        'true_g': 'z/10 - 2',
        'true_h': 'tanh',
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
        'true_g': 'xy/100',
        'true_h': '(·)³',
    },
]

# =====================================================================
# Run each test case
# =====================================================================

results = []

for tc in TEST_CASES:
    print(f"\n{'='*60}")
    print(f"  {tc['name']}")
    print(f"  true g = {tc['true_g']},  true h = {tc['true_h']}")
    print(f"{'='*60}")

    f_vals = tc['f']
    grad_f = tc['grad_f']

    # Step 1: Lasso inner function (degree-3, alpha=1e-5)
    inner = recover_inner_function(x, y, z, f_vals,
                                    max_degree=3, alpha=1e-5)

    print(f"  Inner g: R² = {inner['r2']:.6f}")
    print(f"  Active terms ({len(inner['active'])}): ", end='')
    for name, coeff in zip(inner['active_names'], inner['active_coeffs']):
        print(f"  {coeff:+.6f}·{name}", end='')
    print()

    # Step 2: Evaluate g and ∇g
    g_vals = eval_polynomial(x, y, z, inner['coeffs'], inner['powers'])
    grad_g = grad_polynomial(x, y, z, inner['coeffs'], inner['powers'])

    # Step 3: Outer function via gradient formula
    outer = recover_outer_function(grad_f, grad_g, g_vals, f_vals,
                                    poly_degree=4)

    # Step 4: Full reconstruction
    f_hat = eval_h(g_vals, outer)
    ss_res = np.sum((f_vals - f_hat)**2)
    ss_tot = np.sum((f_vals - f_vals.mean())**2)
    r2_hg = 1.0 - ss_res / max(ss_tot, 1e-12)

    print(f"  Outer h: affine = {outer['used_affine']}")
    print(f"  Full h∘g: R² = {r2_hg:.6f}")

    results.append({
        'tc': tc,
        'inner': inner,
        'outer': outer,
        'g_vals': g_vals,
        'f_hat': f_hat,
        'r2_g': inner['r2'],
        'r2_hg': r2_hg,
    })

# =====================================================================
# Summary table
# =====================================================================

print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)
print(f"{'Test case':<25} {'R²(g)':>10} {'R²(h∘g)':>10} {'#active':>8} {'h type':>10}")
print("-" * 75)
for r in results:
    tc = r['tc']
    aff = 'affine' if r['outer']['used_affine'] else 'nonlinear'
    print(f"{tc['label']:<25} {r['r2_g']:10.6f} {r['r2_hg']:10.6f} "
          f"{len(r['inner']['active']):8d} {aff:>10}")

# =====================================================================
# Four-panel plot per test case
# =====================================================================

for i, r in enumerate(results):
    tc = r['tc']
    inner = r['inner']
    outer = r['outer']
    f_vals = tc['f']
    g_vals = r['g_vals']

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    # Panel 1: g vs f scatter
    ax = axes[0]
    ax.scatter(f_vals, g_vals, s=0.5, alpha=0.2, rasterized=True)
    ax.set_xlabel('f (true)')
    ax.set_ylabel('g (Lasso)')
    ax.set_title(f'Inner: R²(g)={r["r2_g"]:.4f}')

    # Panel 2: h' samples and fit
    ax = axes[1]
    g_v = outer['g_samples']
    hp = outer['h_prime_samples']
    ax.scatter(g_v, hp, s=0.5, alpha=0.1, color='gray', rasterized=True,
               label="h' samples")
    if len(outer.get('h_bin_centers', [])) > 0:
        ax.plot(outer['h_bin_centers'] + outer['g_center'],
                outer['h_bin_medians'], 'r-', lw=2, label='binned median')
    ax.plot(outer['g_plot'], outer['h_prime_fit'], 'b--', lw=1.5,
            label="h' fit")
    ax.set_xlabel('g')
    ax.set_ylabel("h'(g)")
    ax.set_title("Derivative h'")
    ax.legend(fontsize=7)

    # Panel 3: h curve (g vs f, with fit overlay)
    ax = axes[2]
    ax.scatter(g_v, outer['f_valid'], s=0.5, alpha=0.1, color='gray',
               rasterized=True, label='data')
    ax.plot(outer['g_plot'], outer['h_fit_plot'], 'r-', lw=2, label='h fit')
    ax.set_xlabel('g')
    ax.set_ylabel('f')
    ax.set_title(f'Outer h: {"affine" if outer["used_affine"] else "nonlinear"}')
    ax.legend(fontsize=7)

    # Panel 4: f vs f_hat
    ax = axes[3]
    f_hat = r['f_hat']
    ax.scatter(f_vals, f_hat, s=0.5, alpha=0.2, rasterized=True)
    lo, hi = f_vals.min(), f_vals.max()
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    ax.set_xlabel('f (true)')
    ax.set_ylabel('f̂ = h(g)')
    ax.set_title(f'Reconstruction: R²={r["r2_hg"]:.4f}')

    fig.suptitle(f'{tc["name"]}   (true g = {tc["true_g"]}, true h = {tc["true_h"]})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTDIR / f'{tc["label"]}_panels.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTDIR / f'{tc["label"]}_panels.pdf', bbox_inches='tight')
    plt.close()

print(f"\nPlots saved to {OUTDIR}/")
