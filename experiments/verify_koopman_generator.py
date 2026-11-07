"""Verify that the symbolic observables + Koopman generator reproduce true dynamics.

The deep Koopman model learns:
    z_i = h_i(g_i(x))       (observables, from level-set analysis)
    dz/dt = A z              (linear evolution, from generator matrix)

The chain rule gives the true derivative:
    dz_i/dt = h'(g_i(x)) * ∇g_i(x) · F(x)

where F(x) is the known RHS of the dynamical system.

This script compares the two on test data:
    Koopman prediction:  dz_i/dt = Σ_j A[i,j] z_j
    Chain-rule truth:    dz_i/dt = h'(g_i) * ∇g_i · F(x)

If they agree, the trained model has successfully learned a finite-dimensional
Koopman-invariant subspace with interpretable symbolic observables.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import PassthroughAutoEncoder, DeepKANAutoEncoder, DeepKoopmanCT
from data import simulate_mixing, mixing_rhs, simulate_lorenz, lorenz_rhs
from manifold_analysis import (
    full_decomposition_nd, full_decomposition,
    eval_polynomial_nd, grad_polynomial_nd,
    eval_polynomial, grad_polynomial,
    eval_h,
)
from training import set_seed


# =====================================================================
# Helper: evaluate h' (derivative of outer function)
# =====================================================================

def eval_h_prime(g_vals, outer_result):
    """Evaluate h'(g) from the stored outer function decomposition.

    For affine h:  h(g) = a*g + b  →  h'(g) = a  (constant)
    For polynomial h: h'(g) is stored as h_prime_coeffs centered at g_center.
    """
    if outer_result.get('used_affine', False):
        return np.full_like(g_vals, outer_result['affine_coeffs'][0])
    g_shifted = g_vals - outer_result['g_center']
    V = np.vander(g_shifted, N=len(outer_result['h_prime_coeffs']), increasing=True)
    return V @ outer_result['h_prime_coeffs']


# =====================================================================
# Helper: evaluate symbolic z_i and dz_i/dt
# =====================================================================

def eval_symbolic_z(x_data, decomp_results, dim_indices, n_passthrough=0):
    """Evaluate symbolic z_i = h_i(g_i(x)) for all dims.

    Args:
        x_data: (N, d) numpy array
        decomp_results: dict {idx: decomposition_result}
        dim_indices: list of latent dim indices
        n_passthrough: number of passthrough dims (z_i = x_i for i < n_passthrough)

    Returns:
        z_symbolic: (N, len(dim_indices)) array
    """
    N = x_data.shape[0]
    z = np.zeros((N, len(dim_indices)))

    for col, idx in enumerate(dim_indices):
        if idx < n_passthrough:
            z[:, col] = x_data[:, idx]
        else:
            r = decomp_results[idx]
            g_vals = eval_polynomial_nd(x_data, r['inner']['coeffs'], r['inner']['powers'])
            z[:, col] = eval_h(g_vals, r['outer'])

    return z


def eval_chain_rule_dz(x_data, decomp_results, rhs_fn, dim_indices,
                        n_passthrough=0):
    """Compute dz_i/dt via chain rule: h'(g_i) * ∇g_i · F(x).

    Args:
        x_data: (N, d) numpy array
        decomp_results: dict {idx: decomposition_result}
        rhs_fn: function x_tensor -> dx_tensor  (the true RHS)
        dim_indices: list of latent dim indices
        n_passthrough: number of passthrough dims

    Returns:
        dz_chain: (N, len(dim_indices)) array
    """
    N, d = x_data.shape
    dz = np.zeros((N, len(dim_indices)))

    # Compute F(x) — the true dynamics
    x_t = torch.tensor(x_data, dtype=torch.float32)
    F = rhs_fn(x_t).numpy()  # (N, d)

    for col, idx in enumerate(dim_indices):
        if idx < n_passthrough:
            # dz_i/dt = F_i(x) directly
            dz[:, col] = F[:, idx]
        else:
            r = decomp_results[idx]
            inner = r['inner']
            outer = r['outer']

            # g(x) and ∇g(x)
            g_vals = eval_polynomial_nd(x_data, inner['coeffs'], inner['powers'])
            grad_g = grad_polynomial_nd(x_data, inner['coeffs'], inner['powers'])  # (N, d)

            # h'(g)
            hp = eval_h_prime(g_vals, outer)

            # ∇g · F
            grad_dot_F = np.sum(grad_g * F, axis=1)

            # dz/dt = h'(g) * ∇g · F
            dz[:, col] = hp * grad_dot_F

    return dz


def eval_koopman_dz(z_symbolic, A):
    """Compute dz/dt = z @ A^T from the Koopman generator.

    The generator weight has shape (latent_dim, latent_dim) and the model
    uses z_{t+1} = z_t @ expm(A*dt)^T, so dz/dt = z @ A^T.
    """
    return z_symbolic @ A.T


# =====================================================================
# Mixing system verification
# =====================================================================

def verify_mixing():
    print("=" * 70)
    print("KOOPMAN GENERATOR VERIFICATION: Quadratic Coupling System")
    print("=" * 70)

    # Load model
    ckpt_path = os.path.join(os.path.dirname(__file__), "..",
                              "old", "mixing_passthrough_strong_model.pt")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    n_state = cfg['n_state']
    lift_size = cfg['lift_size']
    latent_dim = cfg['latent_dim']
    sigma = cfg['sigma']
    omega = cfg['omega']
    beta_param = cfg['beta']
    alpha_param = cfg['alpha']

    ae = PassthroughAutoEncoder(
        input_size=n_state, mid_size=10, hidden_size=40,
        lift_size=lift_size, grid_size=8, spline_order=3,
        grid_range_l1=[-3, 3], grid_range_l2=[-10, 10],
    )
    model = DeepKoopmanCT(ae, latent_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded model: latent_dim={latent_dim}, n_state={n_state}, lift={lift_size}")
    print(f"Parameters: σ={sigma}, ω={omega}, β={beta_param}, α={alpha_param}")

    # Generate test data
    set_seed(42)  # Different seed from training
    traj = simulate_mixing(
        n_traj=50, n_steps=2000, dt=0.01,
        sigma=sigma, omega=omega, beta=beta_param, alpha=alpha_param,
    )
    x_data = traj[:, ::5, :].reshape(-1, 3)[:10000]
    x_np = x_data.numpy()
    print(f"Test data: {x_np.shape[0]} points")

    # Run level-set analysis on lift dims
    print("\nRunning level-set analysis on lift dims...")
    var_names = ["x1", "x2", "x3"]
    decomp = {}
    for idx in range(n_state, latent_dim):
        decomp[idx] = full_decomposition_nd(
            model, x_data, idx,
            var_names=var_names,
            target_type='latent',
            max_degree=3,
            lasso_alpha=1e-5,
            h_degree=4,
        )
        r2 = decomp[idx]['recon_r2']
        ht = 'affine' if decomp[idx]['outer'].get('used_affine', False) else 'poly'
        names = decomp[idx]['inner']['active_names']
        coeffs = decomp[idx]['inner']['active_coeffs']
        top = sorted(zip(names, coeffs), key=lambda x: abs(x[1]), reverse=True)
        sig = [(n, c) for n, c in top if n != '1'][:3]
        terms = ", ".join(f"{n}({c:+.3f})" for n, c in sig)
        print(f"  z[{idx}]: R²(h∘g)={r2:.4f}, h={ht}, {terms}")

    # Get generator matrix
    A = model.generator_matrix().detach().numpy()
    dim_indices = list(range(latent_dim))

    # Compute symbolic z values
    z_sym = eval_symbolic_z(x_np, decomp, dim_indices, n_passthrough=n_state)

    # Also compute neural network z for comparison
    with torch.no_grad():
        z_nn = model.ae.encoder(x_data).numpy()

    # Compute dz/dt via both methods
    rhs_fn = lambda x: mixing_rhs(x, sigma=sigma, omega=omega,
                                    beta=beta_param, alpha=alpha_param)
    dz_chain = eval_chain_rule_dz(x_np, decomp, rhs_fn, dim_indices,
                                    n_passthrough=n_state)
    dz_koopman = eval_koopman_dz(z_sym, A)

    # Also: dz/dt from Koopman using neural network z (should be better)
    dz_koopman_nn = eval_koopman_dz(z_nn, A)

    # Compare
    print("\n" + "-" * 70)
    print("COMPARISON: chain-rule dz/dt vs Koopman dz/dt")
    print("-" * 70)
    print(f"{'dim':>5s}  {'R²(sym)':>10s}  {'R²(nn)':>10s}  "
          f"{'relErr(sym)':>12s}  {'relErr(nn)':>12s}  type")
    print("-" * 70)

    r2_sym_list = []
    r2_nn_list = []
    for col, idx in enumerate(dim_indices):
        # Chain-rule truth
        dz_true = dz_chain[:, col]

        # Koopman prediction (symbolic z)
        dz_pred_sym = dz_koopman[:, col]

        # Koopman prediction (neural z)
        dz_pred_nn = dz_koopman_nn[:, col]

        # R² (symbolic)
        ss_res_sym = np.sum((dz_true - dz_pred_sym) ** 2)
        ss_tot = np.sum((dz_true - dz_true.mean()) ** 2)
        r2_sym = 1.0 - ss_res_sym / max(ss_tot, 1e-12)

        # R² (neural)
        ss_res_nn = np.sum((dz_true - dz_pred_nn) ** 2)
        r2_nn = 1.0 - ss_res_nn / max(ss_tot, 1e-12)

        # Relative error
        denom = np.sqrt(np.sum(dz_true ** 2))
        re_sym = np.sqrt(ss_res_sym) / max(denom, 1e-12)
        re_nn = np.sqrt(ss_res_nn) / max(denom, 1e-12)

        label = "passthrough" if idx < n_state else "lift"
        print(f"z[{idx:>2d}]  {r2_sym:>10.4f}  {r2_nn:>10.4f}  "
              f"{re_sym:>12.4f}  {re_nn:>12.4f}  {label}")

        r2_sym_list.append(r2_sym)
        r2_nn_list.append(r2_nn)

    print(f"\nMean R² (symbolic): {np.mean(r2_sym_list):.4f}")
    print(f"Mean R² (neural):   {np.mean(r2_nn_list):.4f}")

    # Analytical generator for comparison
    print("\n" + "-" * 70)
    print("ANALYTICAL vs LEARNED GENERATOR EIGENVALUES")
    print("-" * 70)
    eigenvalues = np.linalg.eigvals(A)
    ev_sorted = sorted(eigenvalues, key=lambda x: x.real)
    print("Learned:")
    for i, ev in enumerate(ev_sorted):
        print(f"  λ[{i}] = {ev.real:+.4f} {ev.imag:+.4f}j")
    print(f"\nExpected:")
    print(f"  {-sigma:+.4f} ± {omega:.4f}j  (oscillator)")
    print(f"  {-beta_param:+.4f}              (x3 decay)")
    print(f"  {-2*sigma:+.4f} ± {2*omega:.4f}j  (2nd harmonic)")
    print(f"  {-2*sigma:+.4f}              (zero-freq quadratic)")

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Koopman Generator Verification: Quadratic Coupling System", fontsize=14)

    for col, idx in enumerate(dim_indices):
        ax = axes.flat[col]
        dz_true = dz_chain[:, col]
        dz_pred = dz_koopman[:, col]

        # Subsample for plotting
        step = max(1, len(dz_true) // 2000)
        ax.scatter(dz_true[::step], dz_pred[::step], s=1, alpha=0.3)
        lims = [min(dz_true.min(), dz_pred.min()),
                max(dz_true.max(), dz_pred.max())]
        ax.plot(lims, lims, 'r-', linewidth=1)
        label = "pass" if idx < n_state else "lift"
        ax.set_title(f"z[{idx}] ({label})\nR²={r2_sym_list[col]:.3f}", fontsize=10)
        ax.set_xlabel("chain-rule dz/dt", fontsize=8)
        ax.set_ylabel("Koopman dz/dt", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), "..", "figs",
                             "verify_koopman_mixing.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    print(f"\nPlot saved to {fig_path}")
    plt.close()

    return r2_sym_list, r2_nn_list


# =====================================================================
# Lorenz system verification
# =====================================================================

def verify_lorenz():
    print("\n\n" + "=" * 70)
    print("KOOPMAN GENERATOR VERIFICATION: Lorenz System")
    print("=" * 70)

    # Load model
    ckpt_path = os.path.join(os.path.dirname(__file__), "..",
                              "old", "kan_deep_model.pt")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    latent_dim = cfg['latent_dim']

    ae = DeepKANAutoEncoder(
        input_size=3, mid_size=8, hidden_size=32,
        bottleneck_size=latent_dim, grid_size=5, spline_order=3,
        grid_range_l1=[-30, 50], grid_range_l2=[-10, 10],
    )
    model = DeepKoopmanCT(ae, latent_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded model: latent_dim={latent_dim}")

    # Generate test data
    set_seed(42)
    traj = simulate_lorenz(n_traj=30, n_steps=3000, dt=0.01, burn_in=500)
    x_data = traj[:, ::5, :].reshape(-1, 3)[:10000]
    x_np = x_data.numpy()
    print(f"Test data: {x_np.shape[0]} points")

    # Run level-set analysis on all latent dims
    print("\nRunning level-set analysis on all latent dims...")
    decomp = {}
    for idx in range(latent_dim):
        decomp[idx] = full_decomposition(
            model, x_data, idx,
            target_type='latent',
            max_degree=3,
            lasso_alpha=1e-5,
            h_degree=6,
        )
        r2 = decomp[idx]['recon_r2']
        ht = 'affine' if decomp[idx]['outer'].get('used_affine', False) else 'poly'
        names = decomp[idx]['inner']['active_names']
        coeffs = decomp[idx]['inner']['active_coeffs']
        top = sorted(zip(names, coeffs), key=lambda x: abs(x[1]), reverse=True)
        sig = [(n, c) for n, c in top if n != '1'][:3]
        terms = ", ".join(f"{n}({c:+.3f})" for n, c in sig)
        print(f"  z[{idx}]: R²(h∘g)={r2:.4f}, h={ht}, {terms}")

    # Get generator matrix
    A = model.generator_matrix().detach().numpy()
    dim_indices = list(range(latent_dim))

    # Compute symbolic z values (Lorenz uses 3D polynomial functions)
    N = x_np.shape[0]
    x_, y_, z_ = x_np[:, 0], x_np[:, 1], x_np[:, 2]
    z_sym = np.zeros((N, latent_dim))
    for col, idx in enumerate(dim_indices):
        r = decomp[idx]
        g_vals = eval_polynomial(x_, y_, z_, r['inner']['coeffs'], r['inner']['powers'])
        z_sym[:, col] = eval_h(g_vals, r['outer'])

    # Neural network z
    with torch.no_grad():
        z_nn = model.ae.encoder(x_data).numpy()

    # Compute true F(x)
    x_t = torch.tensor(x_np, dtype=torch.float32)
    F = lorenz_rhs(x_t).numpy()

    # Chain-rule dz/dt for each latent dim
    dz_chain = np.zeros((N, latent_dim))
    for col, idx in enumerate(dim_indices):
        r = decomp[idx]
        inner = r['inner']
        outer = r['outer']

        g_vals = eval_polynomial(x_, y_, z_, inner['coeffs'], inner['powers'])
        gx, gy, gz = grad_polynomial(x_, y_, z_, inner['coeffs'], inner['powers'])

        hp = eval_h_prime(g_vals, outer)

        grad_dot_F = gx * F[:, 0] + gy * F[:, 1] + gz * F[:, 2]
        dz_chain[:, col] = hp * grad_dot_F

    # Koopman dz/dt
    dz_koopman = eval_koopman_dz(z_sym, A)
    dz_koopman_nn = eval_koopman_dz(z_nn, A)

    # Compare
    print("\n" + "-" * 70)
    print("COMPARISON: chain-rule dz/dt vs Koopman dz/dt")
    print("-" * 70)
    print(f"{'dim':>5s}  {'R²(sym)':>10s}  {'R²(nn)':>10s}  "
          f"{'relErr(sym)':>12s}  {'relErr(nn)':>12s}  {'R²(h∘g)':>8s}")
    print("-" * 70)

    r2_sym_list = []
    r2_nn_list = []
    r2_fit_list = []
    for col, idx in enumerate(dim_indices):
        dz_true = dz_chain[:, col]
        dz_pred_sym = dz_koopman[:, col]
        dz_pred_nn = dz_koopman_nn[:, col]

        ss_res_sym = np.sum((dz_true - dz_pred_sym) ** 2)
        ss_res_nn = np.sum((dz_true - dz_pred_nn) ** 2)
        ss_tot = np.sum((dz_true - dz_true.mean()) ** 2)
        r2_sym = 1.0 - ss_res_sym / max(ss_tot, 1e-12)
        r2_nn = 1.0 - ss_res_nn / max(ss_tot, 1e-12)

        denom = np.sqrt(np.sum(dz_true ** 2))
        re_sym = np.sqrt(ss_res_sym) / max(denom, 1e-12)
        re_nn = np.sqrt(ss_res_nn) / max(denom, 1e-12)

        r2_fit = decomp[idx]['recon_r2']

        print(f"z[{idx:>2d}]  {r2_sym:>10.4f}  {r2_nn:>10.4f}  "
              f"{re_sym:>12.4f}  {re_nn:>12.4f}  {r2_fit:>8.4f}")

        r2_sym_list.append(r2_sym)
        r2_nn_list.append(r2_nn)
        r2_fit_list.append(r2_fit)

    print(f"\nMean R² (symbolic): {np.mean(r2_sym_list):.4f}")
    print(f"Mean R² (neural):   {np.mean(r2_nn_list):.4f}")

    # Eigenvalues
    print("\n" + "-" * 70)
    print("KOOPMAN GENERATOR EIGENVALUES")
    print("-" * 70)
    eigenvalues = np.linalg.eigvals(A)
    for i, ev in enumerate(sorted(eigenvalues, key=lambda x: x.real)):
        print(f"  λ[{i}] = {ev.real:+.4f} {ev.imag:+.4f}j")

    # Plot
    n_cols = 5
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
    fig.suptitle("Koopman Generator Verification: Lorenz System", fontsize=14)

    for col, idx in enumerate(dim_indices):
        ax = axes.flat[col]
        dz_true = dz_chain[:, col]
        dz_pred = dz_koopman[:, col]

        step = max(1, len(dz_true) // 2000)
        ax.scatter(dz_true[::step], dz_pred[::step], s=1, alpha=0.3)
        lims = [min(dz_true.min(), dz_pred.min()),
                max(dz_true.max(), dz_pred.max())]
        ax.plot(lims, lims, 'r-', linewidth=1)
        ax.set_title(f"z[{idx}]\nR²={r2_sym_list[col]:.3f}", fontsize=10)
        ax.set_xlabel("chain-rule dz/dt", fontsize=8)
        ax.set_ylabel("Koopman dz/dt", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), "..", "figs",
                             "verify_koopman_lorenz.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    print(f"\nPlot saved to {fig_path}")
    plt.close()

    return r2_sym_list, r2_nn_list


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    r2_mix_sym, r2_mix_nn = verify_mixing()
    r2_lor_sym, r2_lor_nn = verify_lorenz()

    print("\n\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"\nQuadratic Coupling:")
    print(f"  Mean R² (symbolic observables): {np.mean(r2_mix_sym):.4f}")
    print(f"  Mean R² (neural network z):     {np.mean(r2_mix_nn):.4f}")
    print(f"\nLorenz:")
    print(f"  Mean R² (symbolic observables): {np.mean(r2_lor_sym):.4f}")
    print(f"  Mean R² (neural network z):     {np.mean(r2_lor_nn):.4f}")
    print(f"\nInterpretation:")
    print(f"  R² close to 1: Koopman generator + symbolic observables")
    print(f"  faithfully reproduce the true dynamics via dz/dt = Az.")
    print(f"  Gap between 'symbolic' and 'neural': information lost in")
    print(f"  discretizing the neural network into polynomial expressions.")
