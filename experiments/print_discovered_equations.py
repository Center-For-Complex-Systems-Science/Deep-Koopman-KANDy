"""Print the full discovered symbolic equations.

For each latent dim:
    1. The observable:  z_i = h_i(g_i(x))
    2. The Koopman ODE: dz_i/dt = Σ_j A[i,j] z_j

Then compare the Koopman ODE with the true dz_i/dt from chain rule.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from models import PassthroughAutoEncoder, DeepKANAutoEncoder, DeepKoopmanCT
from data import simulate_mixing, simulate_lorenz
from manifold_analysis import (
    full_decomposition_nd, full_decomposition, eval_h,
)
from training import set_seed


def format_polynomial(names, coeffs, threshold=1e-3):
    """Format a sparse polynomial as a readable expression."""
    pairs = sorted(zip(names, coeffs), key=lambda x: abs(x[1]), reverse=True)
    parts = []
    for name, c in pairs:
        if abs(c) < threshold:
            continue
        if name == '1':
            parts.append(f"{c:+.4f}")
        else:
            parts.append(f"{c:+.4f}·{name}")
    return " ".join(parts) if parts else "0"


def format_h(outer):
    """Format the outer function h."""
    if outer.get('used_affine', False):
        a, b = outer['affine_coeffs']
        return f"h(u) = {a:.4f}·u + {b:.4f}"
    else:
        coeffs = outer['h_coeffs']
        center = outer['g_center']
        parts = []
        for k, c in enumerate(coeffs):
            if abs(c) < 1e-6:
                continue
            if k == 0:
                parts.append(f"{c:.4f}")
            elif k == 1:
                parts.append(f"{c:+.4f}·(u{-center:+.2f})")
            else:
                parts.append(f"{c:+.4f}·(u{-center:+.2f})^{k}")
        return "h(u) = " + " ".join(parts) if parts else "h(u) = 0"


def format_koopman_ode(A, i, z_labels):
    """Format dz_i/dt = Σ A[i,j] z_j as a string."""
    parts = []
    order = np.argsort(np.abs(A[i, :]))[::-1]
    for j in order:
        c = A[i, j]
        if abs(c) < 1e-4:
            continue
        parts.append(f"{c:+.4f}·{z_labels[j]}")
    return " ".join(parts) if parts else "0"


def print_mixing_equations():
    print("=" * 80)
    print("DISCOVERED EQUATIONS: Quadratic Coupling System")
    print("=" * 80)

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

    # Generate data for analysis
    set_seed(0)
    traj = simulate_mixing(
        n_traj=100, n_steps=3000, dt=0.01,
        sigma=sigma, omega=omega, beta=beta_param, alpha=alpha_param,
    )
    x_data = traj[:, ::5, :].reshape(-1, 3)[:20000]

    # Run level-set analysis
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

    # Get generator matrix
    A = model.generator_matrix().detach().numpy()

    # --- Part 1: Discovered observables ---
    print("\n" + "─" * 80)
    print("PART 1: DISCOVERED OBSERVABLES")
    print("─" * 80)
    print(f"\nTrue system: dx1/dt = -σx1 + ωx2")
    print(f"             dx2/dt = -ωx1 - σx2")
    print(f"             dx3/dt = -βx3 + αx1x2")
    print(f"  (σ={sigma}, ω={omega}, β={beta_param}, α={alpha_param})")
    print(f"\nExact Koopman-invariant subspace: {{x1, x2, x3, x1², x2², x1x2}}")

    z_labels = []
    z_expressions = []

    for idx in range(latent_dim):
        if idx < n_state:
            label = var_names[idx]
            z_labels.append(label)
            z_expressions.append(label)
            print(f"\n  z[{idx}] = {label}  (passthrough)")
        else:
            r = decomp[idx]
            inner = r['inner']
            outer = r['outer']

            g_expr = format_polynomial(inner['active_names'], inner['active_coeffs'])
            h_expr = format_h(outer)

            # Create a short label based on dominant non-constant terms
            names = inner['active_names']
            coeffs = inner['active_coeffs']
            top = sorted(zip(names, coeffs), key=lambda x: abs(x[1]), reverse=True)
            dominant = [n for n, c in top if n != '1' and abs(c) > 0.01][:2]
            short = "+".join(dominant) if dominant else "?"
            label = f"lift_{idx-n_state}({short})"
            z_labels.append(label)

            full_expr = f"h({g_expr})" if not outer.get('used_affine', False) else g_expr
            z_expressions.append(full_expr)

            r2 = r['recon_r2']
            print(f"\n  z[{idx}] ≈ {label}   [R²(h∘g)={r2:.4f}]")
            print(f"    g(x) = {g_expr}")
            print(f"    {h_expr}")

    # --- Part 2: Koopman generator ODEs ---
    print("\n\n" + "─" * 80)
    print("PART 2: KOOPMAN GENERATOR ODEs  (dz_i/dt = Σ A[i,j] z_j)")
    print("─" * 80)

    for i in range(latent_dim):
        ode = format_koopman_ode(A, i, z_labels)
        dim_label = var_names[i] if i < n_state else f"lift_{i-n_state}"
        print(f"\n  d({z_labels[i]})/dt = {ode}")

    # --- Part 3: True vs Koopman for passthrough dims ---
    print("\n\n" + "─" * 80)
    print("PART 3: COMPARISON (passthrough dims — exact observable, true ODE known)")
    print("─" * 80)

    print(f"\n  TRUE:    dx1/dt = {-sigma:+.4f}·x1 {omega:+.4f}·x2")
    ode_x1 = format_koopman_ode(A, 0, z_labels)
    print(f"  LEARNED: dx1/dt = {ode_x1}")

    print(f"\n  TRUE:    dx2/dt = {-omega:+.4f}·x1 {-sigma:+.4f}·x2")
    ode_x2 = format_koopman_ode(A, 1, z_labels)
    print(f"  LEARNED: dx2/dt = {ode_x2}")

    print(f"\n  TRUE:    dx3/dt = {-beta_param:+.4f}·x3 {alpha_param:+.4f}·x1x2")
    ode_x3 = format_koopman_ode(A, 2, z_labels)
    print(f"  LEARNED: dx3/dt = {ode_x3}")

    # --- Part 4: Generator matrix ---
    print("\n\n" + "─" * 80)
    print("PART 4: GENERATOR MATRIX A")
    print("─" * 80)
    header = "        " + "  ".join(f"{z_labels[j]:>12s}" for j in range(latent_dim))
    print(header)
    for i in range(latent_dim):
        row = f"{z_labels[i]:>7s} " + "  ".join(f"{A[i,j]:>12.4f}" for j in range(latent_dim))
        print(row)

    # Eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    print(f"\nEigenvalues:")
    for i, ev in enumerate(sorted(eigenvalues, key=lambda x: x.real)):
        print(f"  λ[{i}] = {ev.real:+.4f} {ev.imag:+.4f}j")


def print_lorenz_equations():
    print("\n\n" + "=" * 80)
    print("DISCOVERED EQUATIONS: Lorenz System")
    print("=" * 80)

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

    # Generate data
    set_seed(0)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=0.01, burn_in=500)
    x_data = traj[:, ::5, :].reshape(-1, 3)[:20000]

    # Run level-set analysis
    decomp = {}
    for idx in range(latent_dim):
        decomp[idx] = full_decomposition(
            model, x_data, idx,
            target_type='latent',
            max_degree=3,
            lasso_alpha=1e-5,
            h_degree=6,
        )

    # Get generator matrix
    A = model.generator_matrix().detach().numpy()

    # --- Part 1: Discovered observables ---
    print("\n" + "─" * 80)
    print("PART 1: DISCOVERED OBSERVABLES")
    print("─" * 80)
    print(f"\nTrue system: dx/dt = σ(y-x),  dy/dt = x(ρ-z)-y,  dz/dt = xy-βz")
    print(f"  (σ=10, ρ=28, β=8/3)")
    print(f"\nKnown Koopman-relevant observables: {{x, y, z, xy, xz}}")

    z_labels = []
    for idx in range(latent_dim):
        r = decomp[idx]
        inner = r['inner']
        outer = r['outer']

        g_expr = format_polynomial(inner['active_names'], inner['active_coeffs'])
        h_expr = format_h(outer)

        names = inner['active_names']
        coeffs = inner['active_coeffs']
        top = sorted(zip(names, coeffs), key=lambda x: abs(x[1]), reverse=True)
        dominant = [n for n, c in top if n != '1' and abs(c) > 0.01][:2]
        short = "+".join(dominant) if dominant else "?"
        label = f"z{idx}({short})"
        z_labels.append(label)

        r2 = r['recon_r2']
        g_r2 = r['inner']['r2']
        gain = r2 - g_r2
        h_type = "NONLINEAR" if gain > 0.02 else "affine"

        print(f"\n  z[{idx}] ≈ {label}   [R²(g)={g_r2:.4f}, R²(h∘g)={r2:.4f}, h={h_type}]")
        print(f"    g(x,y,z) = {g_expr}")
        if not outer.get('used_affine', False) and gain > 0.02:
            print(f"    {h_expr}")

    # --- Part 2: Koopman generator ODEs ---
    print("\n\n" + "─" * 80)
    print("PART 2: KOOPMAN GENERATOR ODEs")
    print("─" * 80)

    for i in range(latent_dim):
        ode = format_koopman_ode(A, i, z_labels)
        print(f"\n  d({z_labels[i]})/dt = {ode}")

    # --- Part 3: Generator matrix ---
    print("\n\n" + "─" * 80)
    print("PART 3: EIGENVALUES")
    print("─" * 80)
    eigenvalues = np.linalg.eigvals(A)
    for i, ev in enumerate(sorted(eigenvalues, key=lambda x: x.real)):
        print(f"  λ[{i}] = {ev.real:+.4f} {ev.imag:+.4f}j")


if __name__ == "__main__":
    print_mixing_equations()
    print_lorenz_equations()
