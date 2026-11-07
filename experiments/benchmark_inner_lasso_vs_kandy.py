"""
Benchmark: Lasso vs KANDy for inner function recovery in h∘g decomposition.

The manifold analysis decomposes each encoder output as f(x,y,z) = h(g(x,y,z)).

Step 1 (INNER): recover g from a polynomial dictionary
Step 2 (OUTER): recover h via gradient formula  h'(g) = ∇f·∇g / ||∇g||²

This benchmark tests whether replacing Lasso (step 1) with a single-layer KAN
on the SAME dictionary improves recovery — especially when the true inner
function involves non-polynomial transformations of dictionary terms.

Key insight: Lasso fits  g = Σ cᵢ · θᵢ  (LINEAR in dictionary terms).
KANDy fits  g = Σ ψᵢ(θᵢ)  (NONLINEAR spline functions of dictionary terms).

If the true function is f = h(sin(xy)), then:
  - Lasso: g ≈ c₁·xy + c₂·(xy)³ + ...  (polynomial approx of sin → bad)
  - KANDy: g ≈ ψ(xy) where ψ ≈ sin     (spline captures shape → good)

Both feed into the SAME gradient-based outer function recovery for h.

Test cases (synthetic, with known ground truth):
  1. f = sin(xy)               — trig of cross-term
  2. f = exp(-(x²+y²)/20)      — exp of quadratic
  3. f = tanh(xz/10)           — saturating nonlinearity
  4. f = cos(x+2y) + 0.5·z    — trig of linear combo + additive term
  5. f = (xy)³                 — power of cross-term (both should handle)
"""

import sys, os, time
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations_with_replacement
from collections import Counter

from models import _import_pykan
PyKAN = _import_pykan()

OUTDIR = Path('results/benchmark_inner_recovery')
OUTDIR.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)


# =====================================================================
# Polynomial dictionary (shared by both methods)
# =====================================================================

def poly_lift(X, degree=2):
    """Build degree-d monomial features. Returns (features, names)."""
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
# Gradient computation (autograd on a known function)
# =====================================================================

def compute_grad_f(f_fn, X_np):
    """Compute f(X) and ∇f via autograd on a Python function."""
    X_t = torch.tensor(X_np, dtype=torch.float32, requires_grad=True)
    f_t = f_fn(X_t)
    f_t.sum().backward()
    return f_t.detach().numpy(), X_t.grad.numpy()


# =====================================================================
# Outer function recovery (same for both methods)
# =====================================================================

def recover_outer(grad_f, grad_g, g_vals, f_vals, poly_degree=4):
    """Gradient-based outer function recovery: h'(g) = ∇f·∇g / ||∇g||²."""
    dot = np.sum(grad_f * grad_g, axis=1)
    norm_sq = np.sum(grad_g**2, axis=1)

    # Filter small gradients
    threshold = max(1e-4, np.percentile(norm_sq, 5))
    valid = norm_sq > threshold
    h_prime = dot[valid] / norm_sq[valid]
    g_v = g_vals[valid]
    f_v = f_vals[valid]

    # Outlier removal
    med = np.median(h_prime)
    iqr = np.percentile(h_prime, 75) - np.percentile(h_prime, 25)
    if iqr < 1e-10:
        iqr = np.std(h_prime)
    inlier = np.abs(h_prime - med) < 3 * max(iqr, 0.1)
    h_prime = h_prime[inlier]
    g_v = g_v[inlier]
    f_v = f_v[inlier]

    # Binned-median polynomial fit for h'
    n_bins = min(80, len(g_v) // 20)
    g_center = np.median(g_v)
    g_sh = g_v - g_center

    bin_edges = np.linspace(g_sh.min(), g_sh.max(), n_bins + 1)
    bc, bm = [], []
    for i in range(n_bins):
        mask = (g_sh >= bin_edges[i]) & (g_sh < bin_edges[i + 1])
        if mask.sum() >= 5:
            bc.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
            bm.append(np.median(h_prime[mask]))
    bc, bm = np.array(bc), np.array(bm)

    if len(bc) > poly_degree + 1:
        V = np.vander(bc, N=poly_degree + 1, increasing=True)
        hp_c, _, _, _ = np.linalg.lstsq(V, bm, rcond=None)
    else:
        V = np.vander(g_sh, N=poly_degree + 1, increasing=True)
        hp_c, _, _, _ = np.linalg.lstsq(V, h_prime, rcond=None)

    # Integrate h'→h
    h_c = np.zeros(poly_degree + 2)
    for k in range(poly_degree + 1):
        h_c[k + 1] = hp_c[k] / (k + 1)

    # Set constant via reference point
    ref = np.argmin(np.abs(g_sh))
    V_int = np.vander(g_sh, N=poly_degree + 2, increasing=True)
    h_rec = V_int @ h_c
    h_at_ref = np.vander(np.array([g_v[ref] - g_center]), N=poly_degree + 2,
                          increasing=True) @ h_c
    C = f_v[ref] - h_at_ref[0]
    h_c[0] = C
    h_rec += C

    # Affine fallback
    A = np.column_stack([g_v, np.ones(len(g_v))])
    c_aff, _, _, _ = np.linalg.lstsq(A, f_v, rcond=None)
    h_aff = c_aff[0] * g_v + c_aff[1]

    ss_poly = np.sum((f_v - h_rec)**2)
    ss_aff = np.sum((f_v - h_aff)**2)
    use_affine = ss_poly > ss_aff

    if use_affine:
        h_rec = h_aff

    return g_v, f_v, h_rec, use_affine


def eval_h_on_full(g_all, g_v, f_v, h_rec, use_affine):
    """Evaluate h on the full dataset by interpolation."""
    if use_affine:
        A = np.column_stack([g_v, np.ones(len(g_v))])
        c, _, _, _ = np.linalg.lstsq(A, f_v, rcond=None)
        return c[0] * g_all + c[1]
    else:
        # Polynomial interpolation from the valid subset
        A = np.column_stack([g_v, np.ones(len(g_v))])
        c, _, _, _ = np.linalg.lstsq(A, h_rec, rcond=None)
        return c[0] * g_all + c[1]


# =====================================================================
# Method 1: Lasso inner function
# =====================================================================

def inner_lasso(X, f_vals, degree=2, alpha=1e-5):
    """Lasso regression on polynomial dictionary."""
    from sklearn.linear_model import Lasso

    theta, names = poly_lift(X, degree)
    m = theta.shape[1]

    # Standardize
    col_mean = theta.mean(axis=0)
    col_std = theta.std(axis=0)
    col_std[col_std < 1e-12] = 1.0
    theta_s = (theta - col_mean) / col_std
    f_mean, f_std = f_vals.mean(), f_vals.std()
    if f_std < 1e-12:
        f_std = 1.0
    f_s = (f_vals - f_mean) / f_std

    lasso = Lasso(alpha=alpha, max_iter=50000, fit_intercept=True)
    lasso.fit(theta_s, f_s)

    # Unscale
    coeffs = f_std * lasso.coef_ / col_std
    intercept = f_mean + f_std * (lasso.intercept_ - np.sum(lasso.coef_ * col_mean / col_std))

    # Predict g
    g_vals = theta @ coeffs + intercept

    # g gradient via finite differences on the polynomial
    # (analytic would be better but this is simpler for the benchmark)
    eps = 1e-4
    grad_g = np.zeros_like(X)
    for d in range(X.shape[1]):
        Xp = X.copy(); Xp[:, d] += eps
        Xm = X.copy(); Xm[:, d] -= eps
        tp, _ = poly_lift(Xp, degree)
        tm, _ = poly_lift(Xm, degree)
        grad_g[:, d] = ((tp @ coeffs + intercept) - (tm @ coeffs + intercept)) / (2 * eps)

    # R² of g alone
    ss_res = np.sum((f_vals - g_vals)**2)
    ss_tot = np.sum((f_vals - f_vals.mean())**2)
    r2_g = 1.0 - ss_res / max(ss_tot, 1e-12)

    active = np.where(np.abs(coeffs) > 1e-4)[0]
    return {
        'g_vals': g_vals,
        'grad_g': grad_g,
        'r2_g': r2_g,
        'n_active': len(active),
        'active_names': [names[i] for i in active],
        'active_coeffs': coeffs[active],
    }


# =====================================================================
# Method 2: KANDy inner function (single-layer KAN on same dictionary)
# =====================================================================

def inner_kandy(X, f_vals, degree=2, steps=100, grid=10, max_train=5000):
    """Single-layer KAN regression on polynomial dictionary."""
    theta, names = poly_lift(X, degree)
    m = theta.shape[1]

    # Subsample for KAN training
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

    kan = PyKAN(width=[m, 1], grid=grid, k=3, seed=SEED)
    kan.fit(dataset, opt='LBFGS', steps=steps, lamb=0.01)

    # Predict g on ALL data
    with torch.no_grad():
        g_vals = kan(_t(theta)).numpy().flatten()

    # g gradient via autograd through the KAN
    grad_g = np.zeros_like(X)
    eps = 1e-4
    for d in range(X.shape[1]):
        Xp = X.copy(); Xp[:, d] += eps
        Xm = X.copy(); Xm[:, d] -= eps
        tp, _ = poly_lift(Xp, degree)
        tm, _ = poly_lift(Xm, degree)
        with torch.no_grad():
            gp = kan(_t(tp)).numpy().flatten()
            gm = kan(_t(tm)).numpy().flatten()
        grad_g[:, d] = (gp - gm) / (2 * eps)

    # R² of g alone
    ss_res = np.sum((f_vals - g_vals)**2)
    ss_tot = np.sum((f_vals - f_vals.mean())**2)
    r2_g = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Active features via activation variance
    kan.save_act = True
    with torch.no_grad():
        kan(dataset['train_input'])
    acts = kan.acts[0]
    act_var = acts.var(dim=0).numpy()
    thresh = 0.01 * act_var.max()
    active = [i for i in range(m) if act_var[i] > thresh]

    # Extract symbolic formula
    formula = "N/A"
    try:
        kan.auto_symbolic()
        exprs, syms = kan.symbolic_formula()
        if exprs:
            import sympy as sp
            sub_map = {sp.Symbol(str(s)): sp.Symbol(n) for s, n in zip(syms, names)}
            formula = str(sp.expand(sp.sympify(exprs[0]).xreplace(sub_map)))
    except Exception as e:
        formula = f"failed: {e}"

    return {
        'g_vals': g_vals,
        'grad_g': grad_g,
        'r2_g': r2_g,
        'n_active': len(active),
        'active_names': [names[i] for i in active],
        'formula': formula,
    }


# =====================================================================
# Test cases
# =====================================================================

def make_data(n=10000, box=(-5, 5)):
    """Uniform random data in a box (avoids attractor correlation issues)."""
    X = np.random.uniform(box[0], box[1], (n, 3))
    return X


TEST_CASES = [
    {
        'name': 'sin(xy)',
        'fn': lambda X: torch.sin(X[:, 0] * X[:, 1]),
        'fn_np': lambda X: np.sin(X[:, 0] * X[:, 1]),
        'true_inner': 'xy',
        'true_outer': 'sin',
    },
    {
        'name': 'exp(-(x²+y²)/20)',
        'fn': lambda X: torch.exp(-(X[:, 0]**2 + X[:, 1]**2) / 20),
        'fn_np': lambda X: np.exp(-(X[:, 0]**2 + X[:, 1]**2) / 20),
        'true_inner': 'x²+y²',
        'true_outer': 'exp(-·/20)',
    },
    {
        'name': 'tanh(xz/10)',
        'fn': lambda X: torch.tanh(X[:, 0] * X[:, 2] / 10),
        'fn_np': lambda X: np.tanh(X[:, 0] * X[:, 2] / 10),
        'true_inner': 'xz',
        'true_outer': 'tanh(·/10)',
    },
    {
        'name': 'cos(x+2y)',
        'fn': lambda X: torch.cos(X[:, 0] + 2 * X[:, 1]),
        'fn_np': lambda X: np.cos(X[:, 0] + 2 * X[:, 1]),
        'true_inner': 'x+2y',
        'true_outer': 'cos',
    },
    {
        'name': '(xy)³/1000',
        'fn': lambda X: (X[:, 0] * X[:, 1])**3 / 1000,
        'fn_np': lambda X: (X[:, 0] * X[:, 1])**3 / 1000,
        'true_inner': 'xy',
        'true_outer': '(·)³/1000',
    },
]


# =====================================================================
# Run one test case
# =====================================================================

def run_test(tc, X):
    """Run Lasso and KANDy inner recovery, then gradient outer, for one test."""
    name = tc['name']
    f_vals = tc['fn_np'](X)
    f_vals_t, grad_f = compute_grad_f(tc['fn'], X)

    results = {}
    for method_name, inner_fn in [('Lasso', inner_lasso), ('KANDy', inner_kandy)]:
        t0 = time.time()
        inner = inner_fn(X, f_vals)
        t_inner = time.time() - t0

        # Step 2: gradient-based outer recovery (SAME for both)
        t0 = time.time()
        g_v, f_v, h_rec, use_aff = recover_outer(
            grad_f, inner['grad_g'], inner['g_vals'], f_vals)
        t_outer = time.time() - t0

        # Full h∘g on the valid subset
        ss_res = np.sum((f_v - h_rec)**2)
        ss_tot = np.sum((f_v - f_v.mean())**2)
        r2_hg = 1.0 - ss_res / max(ss_tot, 1e-12)

        results[method_name] = {
            'r2_g': inner['r2_g'],
            'r2_hg': r2_hg,
            'n_active': inner['n_active'],
            'active_names': inner['active_names'],
            'used_affine': use_aff,
            'time_inner': t_inner,
            'time_outer': t_outer,
            'g_vals': inner['g_vals'],
            'f_vals': f_vals,
        }
        if 'formula' in inner:
            results[method_name]['formula'] = inner['formula']

    return results


# =====================================================================
# Summary table and plots
# =====================================================================

def print_results(all_results):
    print("\n" + "=" * 100)
    print("INNER FUNCTION BENCHMARK: Lasso vs KANDy (same polynomial dict, same gradient outer)")
    print("=" * 100)
    print(f"{'Test case':<22} | {'':^42s} | {'':^42s}")
    print(f"{'':22s} | {'LASSO':^42s} | {'KANDy':^42s}")
    print(f"{'':22s} | {'R²(g)':>8} {'R²(h∘g)':>8} {'#act':>5} "
          f"{'affine':>7} {'time':>6} | {'R²(g)':>8} {'R²(h∘g)':>8} {'#act':>5} "
          f"{'affine':>7} {'time':>6}")
    print("-" * 100)

    for name, res in all_results.items():
        rl = res['Lasso']
        rk = res['KANDy']
        print(f"{name:<22} | "
              f"{rl['r2_g']:8.4f} {rl['r2_hg']:8.4f} {rl['n_active']:5d} "
              f"{'yes' if rl['used_affine'] else 'no':>7} {rl['time_inner']:5.1f}s | "
              f"{rk['r2_g']:8.4f} {rk['r2_hg']:8.4f} {rk['n_active']:5d} "
              f"{'yes' if rk['used_affine'] else 'no':>7} {rk['time_inner']:5.1f}s")

    # Feature comparison
    print("\n" + "=" * 100)
    print("ACTIVE FEATURES")
    print("=" * 100)
    for name, res in all_results.items():
        rl = res['Lasso']
        rk = res['KANDy']
        print(f"{name:<22} | Lasso: {sorted(rl['active_names'])}")
        print(f"{'':22s} | KANDy: {sorted(rk['active_names'])}")
        if 'formula' in rk:
            print(f"{'':22s} | formula: {rk['formula']}")
        print()


def plot_results(all_results):
    n = len(all_results)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10), squeeze=False)

    for col, (name, res) in enumerate(all_results.items()):
        rl = res['Lasso']
        rk = res['KANDy']
        f = rl['f_vals']

        # Row 0: Lasso g vs f
        ax = axes[0, col]
        ax.scatter(f, rl['g_vals'], s=1, alpha=0.3, rasterized=True)
        lo, hi = f.min(), f.max()
        ax.plot([lo, hi], [lo, hi], 'r--', lw=0.8)
        ax.set_title(f'Lasso g  R²={rl["r2_g"]:.4f}', fontsize=9)
        if col == 0:
            ax.set_ylabel('g (Lasso)')

        # Row 1: KANDy g vs f
        ax = axes[1, col]
        ax.scatter(f, rk['g_vals'], s=1, alpha=0.3, color='darkorange', rasterized=True)
        ax.plot([lo, hi], [lo, hi], 'r--', lw=0.8)
        ax.set_title(f'KANDy g  R²={rk["r2_g"]:.4f}', fontsize=9)
        if col == 0:
            ax.set_ylabel('g (KANDy)')

        # Row 2: R² bar comparison
        ax = axes[2, col]
        x = [0, 1]
        bars = ax.bar(x, [rl['r2_hg'], rk['r2_hg']], color=['steelblue', 'darkorange'])
        ax.set_xticks(x)
        ax.set_xticklabels(['Lasso\nh∘g', 'KANDy\nh∘g'])
        ax.set_ylim(min(rl['r2_hg'], rk['r2_hg']) - 0.1, 1.05)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.axhline(1.0, color='gray', ls=':', lw=0.5)
        for bar, val in zip(bars, [rl['r2_hg'], rk['r2_hg']]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTDIR / 'inner_recovery_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTDIR / 'inner_recovery_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved plots to {OUTDIR}/inner_recovery_comparison.png")


# =====================================================================
# Main
# =====================================================================

def main():
    print("Generating uniform test data (10k points, [-5,5]³)...")
    X = make_data(n=10000, box=(-5, 5))
    print(f"  shape: {X.shape}")

    all_results = {}
    for tc in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"TEST: {tc['name']}")
        print(f"  true inner: {tc['true_inner']},  true outer: {tc['true_outer']}")
        print(f"{'='*60}")

        res = run_test(tc, X)
        all_results[tc['name']] = res

        for method in ['Lasso', 'KANDy']:
            r = res[method]
            print(f"  {method:6s}: R²(g)={r['r2_g']:.4f}  R²(h∘g)={r['r2_hg']:.4f}  "
                  f"#active={r['n_active']}  affine_h={r['used_affine']}  "
                  f"time={r['time_inner']:.1f}s")

    print_results(all_results)
    plot_results(all_results)
    print(f"\nDone. Results in {OUTDIR}/")


if __name__ == '__main__':
    main()
