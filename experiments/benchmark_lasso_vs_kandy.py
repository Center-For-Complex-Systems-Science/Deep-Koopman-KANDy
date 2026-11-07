"""
Benchmark: Lasso vs KANDy for inner function estimation.

For each latent dimension z[k] of a trained deep Koopman encoder, we compare
two methods of recovering the inner function g(x,y,z) such that z[k] ≈ h(g(x)):

  1. LASSO: Sparse polynomial regression via L1 penalty
     (current manifold_analysis.recover_inner_function)

  2. KANDy: Single-layer KAN on a PolynomialLift, with auto_symbolic extraction
     (kandy.KANDy with PolynomialLift(degree=2))

Metrics:
  - R² of the full approximation to f_vals = encoder(x)[k]
  - Number of active features (sparsity)
  - Which cross-terms are identified
  - Wall-clock time

Uses the Lorenz stable model (results/lorenz_stable/).
"""

import sys, os, time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations_with_replacement
from collections import Counter

from models import PyKANAutoEncoder
from data import simulate_lorenz
from training import set_seed
from manifold_analysis import (
    recover_inner_function,
    compute_network_gradient,
    recover_outer_function,
    eval_polynomial,
    grad_polynomial,
    eval_h,
)

# Import PyKAN's KAN class (not the local kan.py KANLinear)
from models import _import_pykan
PyKAN = _import_pykan()

OUTDIR = Path('results/benchmark_lasso_vs_kandy')
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DT = 0.01


# =====================================================================
# Load or rebuild the stable Lorenz model
# =====================================================================

def load_model():
    """Load the Lorenz stable model and generate matching data."""
    from experiments.lorenz_stable import DeepKoopmanCTStable

    model_path = Path('results/lorenz_stable/stable_8_32_model.pt')
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run experiments/lorenz_stable.py first."
        )

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    mid_size = cfg['mid']
    latent_dim = cfg['latent']
    print(f"  Config: mid={mid_size}, latent={latent_dim}")

    ae = PyKANAutoEncoder(3, mid_size, latent_dim, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, latent_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Generate Lorenz data (same parameters as training)
    set_seed(SEED)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=500)
    x_all = traj.reshape(-1, 3)

    # Subsample for speed
    N = min(20000, x_all.shape[0])
    idx = torch.randperm(x_all.shape[0])[:N]
    x_data = x_all[idx]

    return model, x_data


# =====================================================================
# Method 1: Lasso inner function (current pipeline)
# =====================================================================

def run_lasso(model, x_data, target_idx, max_degree=2):
    """Run the Lasso inner function recovery + outer h for one latent dim."""
    t0 = time.time()

    x_t = x_data if isinstance(x_data, torch.Tensor) else torch.tensor(x_data, dtype=torch.float32)
    x_np = x_data.numpy() if isinstance(x_data, torch.Tensor) else x_data
    x_, y_, z_ = x_np[:, 0], x_np[:, 1], x_np[:, 2]

    # Get encoder output + gradient
    grad_f, f_vals = compute_network_gradient(model, x_t, target_idx, 'latent')

    # Inner function via Lasso
    inner = recover_inner_function(x_, y_, z_, f_vals, max_degree=max_degree,
                                    alpha=1e-5)

    # Outer function via gradient formula
    g_vals = eval_polynomial(x_, y_, z_, inner['coeffs'], inner['powers'])
    grad_g = grad_polynomial(x_, y_, z_, inner['coeffs'], inner['powers'])
    outer = recover_outer_function(grad_f, grad_g, g_vals, f_vals, poly_degree=4)

    # Full reconstruction
    f_hat = eval_h(g_vals, outer)
    ss_res = np.sum((f_vals - f_hat) ** 2)
    ss_tot = np.sum((f_vals - f_vals.mean()) ** 2)
    full_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    elapsed = time.time() - t0

    return {
        'method': 'Lasso',
        'target_idx': target_idx,
        'inner_r2': inner['r2'],
        'full_r2': full_r2,
        'n_active': len(inner['active']),
        'active_names': inner['active_names'],
        'active_coeffs': inner['active_coeffs'].tolist(),
        'used_affine': outer.get('used_affine', False),
        'elapsed': elapsed,
        'f_vals': f_vals,
        'f_hat': f_hat,
    }


# =====================================================================
# Method 2: KANDy single-layer KAN on PolynomialLift
# =====================================================================

def poly_lift(X, degree=2):
    """PolynomialLift: build degree-d monomial features from X (N, n).

    Returns (features, feature_names).
    """
    if X.ndim == 1:
        X = X[None, :]
    n = X.shape[1]
    combos = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n), d):
            combos.append(combo)

    cols = []
    names = []
    for combo in combos:
        col = np.ones(X.shape[0], dtype=X.dtype)
        for idx in combo:
            col = col * X[:, idx]
        cols.append(col)
        c = Counter(combo)
        parts = []
        for idx, power in sorted(c.items()):
            vname = ['x', 'y', 'z'][idx] if idx < 3 else f'x_{idx}'
            parts.append(vname if power == 1 else f'{vname}^{power}')
        names.append('*'.join(parts))

    return np.column_stack(cols), names


def run_kandy(model, x_data, target_idx, max_degree=2, steps=100, grid=10,
              max_train=5000):
    """Fit a single-layer PyKAN on PolynomialLift features to approximate encoder(x)[k]."""
    t0 = time.time()

    x_t = x_data if isinstance(x_data, torch.Tensor) else torch.tensor(x_data, dtype=torch.float32)
    x_np = x_data.numpy() if isinstance(x_data, torch.Tensor) else x_data

    # Get encoder output (the "target" we're trying to approximate)
    with torch.no_grad():
        z_all = model.ae.encoder(x_t)
        f_vals = z_all[:, target_idx].numpy()

    # Build polynomial lift features
    theta, feat_names = poly_lift(x_np, degree=max_degree)
    m = theta.shape[1]  # lift dimension

    # Subsample for KAN training speed (LBFGS on full 20k is slow)
    N = len(theta)
    if N > max_train:
        idx = np.random.permutation(N)[:max_train]
        theta_sub = theta[idx]
        f_sub = f_vals[idx]
    else:
        theta_sub = theta
        f_sub = f_vals

    # Chronological split (70/15/15) on subsampled data
    Ns = len(theta_sub)
    n1 = int(Ns * 0.70)
    n2 = int(Ns * 0.85)

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    dataset = {
        'train_input': _t(theta_sub[:n1]),
        'train_label': _t(f_sub[:n1, None]),
        'test_input': _t(theta_sub[n1:n2]),
        'test_label': _t(f_sub[n1:n2, None]),
    }

    # Build single-layer KAN: width=[m, 1]
    kan_model = PyKAN(width=[m, 1], grid=grid, k=3, seed=SEED)

    # Train with LBFGS
    kan_model.fit(dataset, opt='LBFGS', steps=steps, lamb=0.01)

    # Predict on ALL data
    with torch.no_grad():
        f_hat = kan_model(_t(theta)).numpy().flatten()
    ss_res = np.sum((f_vals - f_hat) ** 2)
    ss_tot = np.sum((f_vals - f_vals.mean()) ** 2)
    full_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Extract symbolic formula
    formula_str = "N/A"
    try:
        kan_model.save_act = True
        with torch.no_grad():
            kan_model(dataset['train_input'])
        kan_model.auto_symbolic()
        exprs, syms = kan_model.symbolic_formula()
        if exprs:
            import sympy as sp
            sub_map = {sp.Symbol(str(s)): sp.Symbol(n)
                      for s, n in zip(syms, feat_names)}
            formula = sp.sympify(exprs[0]).xreplace(sub_map)
            formula_str = str(sp.expand(formula))
    except Exception as e:
        formula_str = f"extraction failed: {e}"

    # Identify active features via edge activation variance
    active_names = []
    n_active = 0
    try:
        kan_model.save_act = True
        with torch.no_grad():
            kan_model(dataset['train_input'])
        # Check which input edges have significant activation variance
        acts = kan_model.acts[0]  # (N, m) — layer 0 post-activations
        act_var = acts.var(dim=0).numpy()
        threshold = 0.01 * act_var.max()
        for i in range(m):
            if act_var[i] > threshold:
                active_names.append(feat_names[i])
                n_active += 1
    except Exception:
        n_active = -1

    elapsed = time.time() - t0

    return {
        'method': 'KANDy',
        'target_idx': target_idx,
        'inner_r2': None,  # KANDy doesn't separate inner/outer
        'full_r2': full_r2,
        'n_active': n_active,
        'active_names': active_names,
        'formula': formula_str,
        'elapsed': elapsed,
        'f_vals': f_vals,
        'f_hat': f_hat,
    }


# =====================================================================
# Comparison table
# =====================================================================

def print_comparison(results_lasso, results_kandy):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("BENCHMARK: Lasso h∘g  vs  KANDy Ψ(φ(x))  on Lorenz encoder latent dims")
    print("=" * 90)
    print(f"{'z[k]':>5} | {'Lasso R²':>10} {'h∘g R²':>10} {'#act':>5} {'time':>6} | "
          f"{'KANDy R²':>10} {'#act':>5} {'time':>6} | {'winner':>7}")
    print("-" * 90)

    lasso_wins, kandy_wins = 0, 0
    for rl, rk in zip(results_lasso, results_kandy):
        k = rl['target_idx']
        lr2 = rl['full_r2']
        kr2 = rk['full_r2']

        if lr2 > kr2 + 0.005:
            winner = "Lasso"
            lasso_wins += 1
        elif kr2 > lr2 + 0.005:
            winner = "KANDy"
            kandy_wins += 1
        else:
            winner = "tie"

        print(f"z[{k}] | "
              f"{rl['inner_r2']:10.4f} {lr2:10.4f} {rl['n_active']:5d} {rl['elapsed']:5.1f}s | "
              f"{kr2:10.4f} {rk['n_active']:5d} {rk['elapsed']:5.1f}s | "
              f"{winner:>7}")

    print("-" * 90)

    lr2_all = np.array([r['full_r2'] for r in results_lasso])
    kr2_all = np.array([r['full_r2'] for r in results_kandy])
    lt_all = sum(r['elapsed'] for r in results_lasso)
    kt_all = sum(r['elapsed'] for r in results_kandy)

    print(f"{'MEAN':>5} | {'':>10} {lr2_all.mean():10.4f} {'':>5} {lt_all:5.1f}s | "
          f"{kr2_all.mean():10.4f} {'':>5} {kt_all:5.1f}s | "
          f"L:{lasso_wins} K:{kandy_wins}")

    # Feature comparison
    print("\n" + "=" * 90)
    print("FEATURE COMPARISON: which cross-terms does each method find?")
    print("=" * 90)
    LORENZ_TERMS = {'x', 'y', 'z', 'xy', 'xz'}  # known Koopman observables

    for rl, rk in zip(results_lasso, results_kandy):
        k = rl['target_idx']
        l_names = set(rl['active_names'])
        k_names = set(rk['active_names'])

        # Normalize names for comparison
        def normalize(names):
            """Map polynomial names to a common format."""
            mapping = {
                'x_0': 'x', 'x_1': 'y', 'x_2': 'z',
                'x_0^2': 'x²', 'x_1^2': 'y²', 'x_2^2': 'z²',
                'x_0*x_1': 'xy', 'x_0*x_2': 'xz', 'x_1*x_2': 'yz',
            }
            return {mapping.get(n, n) for n in names}

        k_norm = normalize(k_names)
        l_cross = {n for n in l_names if len(n) > 1 and n != '1' and '^' not in n}
        k_cross = {n for n in k_norm if len(n) > 1 and n != '1' and '²' not in n}

        l_str = str(sorted(l_names))
        k_str = str(sorted(k_norm))
        print(f"z[{k}] | Lasso: {l_str:40s} | KANDy: {k_str}")

    # KANDy formulas
    print("\n" + "=" * 90)
    print("KANDy SYMBOLIC FORMULAS")
    print("=" * 90)
    for rk in results_kandy:
        k = rk['target_idx']
        print(f"z[{k}] = {rk.get('formula', 'N/A')}")


# =====================================================================
# Scatter plots
# =====================================================================

def plot_scatter(results_lasso, results_kandy):
    """f_true vs f_hat scatter plots for both methods."""
    n_dims = len(results_lasso)
    fig, axes = plt.subplots(2, n_dims, figsize=(3.2 * n_dims, 6),
                              squeeze=False)

    for col, (rl, rk) in enumerate(zip(results_lasso, results_kandy)):
        k = rl['target_idx']
        for row, (res, label) in enumerate([(rl, 'Lasso'), (rk, 'KANDy')]):
            ax = axes[row, col]
            f = res['f_vals']
            fh = res['f_hat']
            ax.scatter(f, fh, s=1, alpha=0.3, rasterized=True)
            lo, hi = f.min(), f.max()
            ax.plot([lo, hi], [lo, hi], 'r--', lw=0.8)
            ax.set_title(f'{label} z[{k}]  R²={res["full_r2"]:.4f}', fontsize=9)
            if col == 0:
                ax.set_ylabel(r'$\hat{f}$')
            if row == 1:
                ax.set_xlabel(r'$f$ (encoder)')

    plt.tight_layout()
    plt.savefig(OUTDIR / 'scatter_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTDIR / 'scatter_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plots to {OUTDIR}/scatter_comparison.png")


def plot_r2_bar(results_lasso, results_kandy):
    """Bar chart comparing R² across latent dims."""
    dims = [r['target_idx'] for r in results_lasso]
    lr2 = [r['full_r2'] for r in results_lasso]
    kr2 = [r['full_r2'] for r in results_kandy]

    x = np.arange(len(dims))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w/2, lr2, w, label='Lasso h∘g', color='steelblue')
    ax.bar(x + w/2, kr2, w, label='KANDy Ψ(φ)', color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels([f'z[{d}]' for d in dims])
    ax.set_ylabel('R²')
    ax.set_title('Inner Function Recovery: Lasso vs KANDy')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', ls=':', lw=0.5)

    plt.tight_layout()
    plt.savefig(OUTDIR / 'r2_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTDIR / 'r2_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved R² bar chart to {OUTDIR}/r2_comparison.png")


# =====================================================================
# Main
# =====================================================================

def main():
    set_seed(SEED)
    print("Loading Lorenz stable model...")
    model, x_data = load_model()
    latent_dim = model.latent_dim
    print(f"  Data: {x_data.shape[0]} points, latent_dim={latent_dim}")

    # Run on a subset of dims (32 is a lot; pick the 8 with highest variance)
    with torch.no_grad():
        z_all = model.ae.encoder(x_data)
        z_var = z_all.var(dim=0).numpy()
    top_dims = np.argsort(z_var)[::-1][:8].tolist()
    print(f"  Top-8 dims by variance: {top_dims}")

    results_lasso = []
    results_kandy = []

    for k in top_dims:
        print(f"\n--- z[{k}] ---")

        print(f"  Running Lasso...")
        rl = run_lasso(model, x_data, k, max_degree=2)
        print(f"    inner R²={rl['inner_r2']:.4f}, h∘g R²={rl['full_r2']:.4f}, "
              f"#active={rl['n_active']}, affine_h={rl['used_affine']}, "
              f"time={rl['elapsed']:.1f}s")

        print(f"  Running KANDy...")
        rk = run_kandy(model, x_data, k, max_degree=2, steps=100, grid=10)
        print(f"    R²={rk['full_r2']:.4f}, #active={rk['n_active']}, "
              f"time={rk['elapsed']:.1f}s")

        results_lasso.append(rl)
        results_kandy.append(rk)

    # Print comparison
    print_comparison(results_lasso, results_kandy)

    # Plots
    plot_scatter(results_lasso, results_kandy)
    plot_r2_bar(results_lasso, results_kandy)

    print(f"\nAll results saved to {OUTDIR}/")


if __name__ == '__main__':
    main()
