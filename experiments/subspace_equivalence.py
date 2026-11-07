"""
Test: do the learned spline observables span the same Koopman-invariant
subspace as the polynomial dictionary {x, y, z, xy, xz}?

If the spline basis and polynomial basis span the same subspace on the
attractor, then:
  - Each polynomial observable can be written as a linear combination of psi's
  - Each psi can be written as a linear combination of polynomial observables
  - Both projections have R² ~ 1

This would mean the KAN discovered a different BASIS for the same invariant
subspace — not the polynomial one, but an equivalent nonlinearly-warped one.

Run from project root:
    python experiments/subspace_equivalence.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import simulate_lorenz
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable

OUTDIR = Path("results/spline_basis_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)
DT = 0.01


def load_model():
    latent_dim = 32
    ae = PyKANAutoEncoder(3, 8, latent_dim, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, latent_dim)
    ckpt = torch.load("results/lorenz_stable/stable_8_32_model.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, latent_dim


def ols_r2(X, y):
    """OLS regression of y on X, return R², coefficients."""
    coeffs, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return r2, coeffs, pred


def main():
    print("=" * 70)
    print("  SUBSPACE EQUIVALENCE TEST")
    print("=" * 70)

    model, latent_dim = load_model()

    set_seed(42)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:48]
    x_flat = train_traj.reshape(-1, 3)

    # Use a large sample for stable regression
    idx = np.random.choice(len(x_flat), 10000, replace=False)
    x_data = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    x_np = x_data.numpy()
    print(f"  Data: {x_data.shape}")

    # ── Get spline activations (Layer-1 output) ──
    enc = model.ae.encoder
    with torch.no_grad():
        enc(x_data)
        psi = enc.acts[1].numpy()  # (N, 8)
    print(f"  Psi (Layer-1 output): {psi.shape}")

    # ── Build polynomial observables ──
    x, y, z = x_np[:, 0], x_np[:, 1], x_np[:, 2]

    poly_lorenz = np.column_stack([x, y, z, x*y, x*z])
    poly_lorenz_names = ["x", "y", "z", "xy", "xz"]

    poly_full_d2 = np.column_stack([x, y, z, x**2, x*y, x*z, y**2, y*z, z**2])
    poly_full_names = ["x", "y", "z", "x²", "xy", "xz", "y²", "yz", "z²"]

    # Add intercept to regression matrices
    psi_with_const = np.column_stack([np.ones(len(psi)), psi])
    poly_lorenz_wc = np.column_stack([np.ones(len(x)), poly_lorenz])
    poly_full_wc = np.column_stack([np.ones(len(x)), poly_full_d2])

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Can the 8 psi's explain each polynomial observable?
    #   poly_k = a_0 + sum_j a_j * psi_j
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  TEST 1: Polynomial → Spline projection")
    print("  Can {psi_0,...,psi_7} explain each of {x, y, z, xy, xz}?")
    print(f"{'='*70}")

    print(f"\n  {'Observable':>12} {'R²(8 psi)':>10} {'R²(8 psi + products)':>22}")
    print("  " + "-" * 50)

    # Also test with psi products (degree-2 in psi space)
    psi_products = []
    psi_prod_names = []
    for i in range(8):
        for j in range(i, 8):
            psi_products.append(psi[:, i] * psi[:, j])
            psi_prod_names.append(f"psi{i}*psi{j}" if i != j else f"psi{i}²")
    psi_full = np.column_stack([np.ones(len(psi)), psi] + [p.reshape(-1, 1) for p in psi_products])

    test1_r2_linear = []
    test1_r2_full = []
    for k, name in enumerate(poly_lorenz_names):
        target = poly_lorenz[:, k]
        r2_lin, _, _ = ols_r2(psi_with_const, target)
        r2_full, _, _ = ols_r2(psi_full, target)
        test1_r2_linear.append(r2_lin)
        test1_r2_full.append(r2_full)
        print(f"  {name:>12} {r2_lin:>10.6f} {r2_full:>22.6f}")

    print(f"\n  Min R² (linear psi): {min(test1_r2_linear):.6f}")
    print(f"  Min R² (psi + products): {min(test1_r2_full):.6f}")

    # Also test the spurious terms
    print(f"\n  Spurious polynomial terms:")
    for k, name in enumerate(poly_full_names):
        if name in poly_lorenz_names:
            continue
        target = poly_full_d2[:, k]
        r2_lin, _, _ = ols_r2(psi_with_const, target)
        r2_full, _, _ = ols_r2(psi_full, target)
        print(f"  {name:>12} {r2_lin:>10.6f} {r2_full:>22.6f}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: Can the polynomial basis explain each psi?
    #   psi_j = b_0 + sum_k b_k * poly_k
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  TEST 2: Spline → Polynomial projection")
    print("  Can {x, y, z, xy, xz} explain each psi_j?")
    print(f"{'='*70}")

    print(f"\n  {'Psi node':>10} {'R²(Lorenz)':>12} {'R²(full d2)':>13} {'Residual var':>14}")
    print("  " + "-" * 55)

    test2_r2_lorenz = []
    test2_r2_full = []
    for j in range(8):
        target = psi[:, j]
        r2_lor, _, _ = ols_r2(poly_lorenz_wc, target)
        r2_full, _, pred_full = ols_r2(poly_full_wc, target)
        residual_var = np.var(target - pred_full)
        test2_r2_lorenz.append(r2_lor)
        test2_r2_full.append(r2_full)
        print(f"  psi_{j:>5} {r2_lor:>12.6f} {r2_full:>13.6f} {residual_var:>14.6f}")

    print(f"\n  Min R² (Lorenz basis): {min(test2_r2_lorenz):.6f}")
    print(f"  Min R² (full deg-2):   {min(test2_r2_full):.6f}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: Subspace angles
    # Compute principal angles between span(psi) and span(poly_lorenz)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  TEST 3: Principal angles between subspaces")
    print(f"{'='*70}")

    # Center the data
    psi_c = psi - psi.mean(axis=0)
    poly_c = poly_lorenz - poly_lorenz.mean(axis=0)

    # QR decomposition for orthonormal bases
    Q_psi, _ = np.linalg.qr(psi_c)      # (N, 8)
    Q_poly, _ = np.linalg.qr(poly_c)    # (N, 5)

    # SVD of Q_poly^T @ Q_psi gives cosines of principal angles
    M = Q_poly.T @ Q_psi  # (5, 8)
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    # sigma = cos(principal angles), should be 5 values
    angles_deg = np.arccos(np.clip(sigma, -1, 1)) * 180 / np.pi

    print(f"\n  Principal angles (degrees) between span(psi_8) and span(poly_Lorenz_5):")
    for i, (s, a) in enumerate(zip(sigma, angles_deg)):
        print(f"    angle_{i}: {a:6.2f}°  (cos = {s:.6f})")

    print(f"\n  All angles < 1° means the subspaces are nearly identical.")
    print(f"  Max angle: {angles_deg.max():.2f}°")

    # ══════════════════════════════════════════════════════════════════
    # TEST 4: One-step Koopman consistency
    # If psi spans the invariant subspace, then psi(x_{t+1}) should be
    # a linear function of psi(x_t).
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  TEST 4: Koopman linearity in spline basis")
    print("  Is psi(x_{t+1}) = K_psi @ psi(x_t)?")
    print(f"{'='*70}")

    # Get consecutive pairs
    x_t = train_traj[:, :-1, :].reshape(-1, 3)
    x_tp1 = train_traj[:, 1:, :].reshape(-1, 3)

    # Subsample
    n_pairs = min(50000, len(x_t))
    pair_idx = np.random.choice(len(x_t), n_pairs, replace=False)
    x_t_sub = torch.tensor(x_t[pair_idx].numpy(), dtype=torch.float32)
    x_tp1_sub = torch.tensor(x_tp1[pair_idx].numpy(), dtype=torch.float32)

    with torch.no_grad():
        enc(x_t_sub)
        psi_t = enc.acts[1].numpy()
        enc(x_tp1_sub)
        psi_tp1 = enc.acts[1].numpy()

    # Fit K_psi via least squares: psi_{t+1} = K_psi @ psi_t
    psi_t_wc = np.column_stack([np.ones(len(psi_t)), psi_t])
    print(f"\n  {'Psi node':>10} {'R²(linear Koopman)':>20}")
    print("  " + "-" * 35)

    koopman_r2s = []
    for j in range(8):
        r2, _, _ = ols_r2(psi_t_wc, psi_tp1[:, j])
        koopman_r2s.append(r2)
        print(f"  psi_{j:>5} {r2:>20.6f}")

    print(f"\n  Min Koopman R²: {min(koopman_r2s):.6f}")
    print(f"  Mean Koopman R²: {np.mean(koopman_r2s):.6f}")

    # Compare: Koopman linearity in POLYNOMIAL basis
    x_t_np = x_t[pair_idx].numpy()
    x_tp1_np = x_tp1[pair_idx].numpy()
    poly_t = np.column_stack([
        np.ones(n_pairs),
        x_t_np[:, 0], x_t_np[:, 1], x_t_np[:, 2],
        x_t_np[:, 0] * x_t_np[:, 1], x_t_np[:, 0] * x_t_np[:, 2],
    ])
    poly_tp1 = np.column_stack([
        x_tp1_np[:, 0], x_tp1_np[:, 1], x_tp1_np[:, 2],
        x_tp1_np[:, 0] * x_tp1_np[:, 1], x_tp1_np[:, 0] * x_tp1_np[:, 2],
    ])

    print(f"\n  Comparison: Koopman linearity in polynomial basis {{x,y,z,xy,xz}}:")
    poly_koopman_r2s = []
    for k, name in enumerate(poly_lorenz_names):
        r2, _, _ = ols_r2(poly_t, poly_tp1[:, k])
        poly_koopman_r2s.append(r2)
        print(f"  {name:>10} {r2:>20.6f}")

    print(f"\n  Min poly Koopman R²: {min(poly_koopman_r2s):.6f}")
    print(f"  Mean poly Koopman R²: {np.mean(poly_koopman_r2s):.6f}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Test 1 — poly → psi projection:")
    print(f"    Can 8 psi's explain {{x,y,z,xy,xz}}?  min R² = {min(test1_r2_linear):.6f}")
    print(f"    With psi products?                   min R² = {min(test1_r2_full):.6f}")
    print(f"\n  Test 2 — psi → poly projection:")
    print(f"    Can {{x,y,z,xy,xz}} explain each psi?  min R² = {min(test2_r2_lorenz):.6f}")
    print(f"    Can full deg-2 explain each psi?       min R² = {min(test2_r2_full):.6f}")
    print(f"\n  Test 3 — principal angles:")
    print(f"    Max angle between span(psi) and span(poly): {angles_deg.max():.2f}°")
    print(f"\n  Test 4 — Koopman linearity:")
    print(f"    Spline basis min R²:     {min(koopman_r2s):.6f}")
    print(f"    Polynomial basis min R²: {min(poly_koopman_r2s):.6f}")

    if angles_deg.max() < 5.0 and min(test1_r2_linear) > 0.95:
        print(f"\n  CONCLUSION: The subspaces appear EQUIVALENT.")
        print(f"  The KAN learned a nonlinearly-warped basis for the same")
        print(f"  Koopman-invariant subspace that {{x,y,z,xy,xz}} spans.")
    elif min(test1_r2_full) > 0.95:
        print(f"\n  CONCLUSION: Subspaces equivalent at degree-2 in psi space.")
        print(f"  Linear psi alone is insufficient; need psi products.")
    else:
        print(f"\n  CONCLUSION: Subspaces are NOT equivalent.")
        print(f"  The KAN learned a different invariant subspace.")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Test 1 bar chart
    ax = axes[0, 0]
    x_pos = np.arange(len(poly_lorenz_names))
    ax.bar(x_pos - 0.15, test1_r2_linear, 0.3, label="8 psi (linear)", color="steelblue")
    ax.bar(x_pos + 0.15, test1_r2_full, 0.3, label="psi + products", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(poly_lorenz_names)
    ax.set_ylabel("R²")
    ax.set_title("Test 1: Can psi explain each polynomial?")
    ax.legend()
    ax.set_ylim(0.8, 1.005)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.3)

    # Panel 2: Test 2 bar chart
    ax = axes[0, 1]
    x_pos = np.arange(8)
    ax.bar(x_pos - 0.15, test2_r2_lorenz, 0.3, label="Lorenz {x,y,z,xy,xz}", color="steelblue")
    ax.bar(x_pos + 0.15, test2_r2_full, 0.3, label="Full deg-2", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"ψ{j}" for j in range(8)])
    ax.set_ylabel("R²")
    ax.set_title("Test 2: Can polynomials explain each ψ?")
    ax.legend()
    ax.set_ylim(0.8, 1.005)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.3)

    # Panel 3: Principal angles
    ax = axes[1, 0]
    ax.bar(range(len(angles_deg)), angles_deg, color="steelblue")
    ax.set_xlabel("Principal angle index")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Test 3: Principal angles between subspaces")
    ax.axhline(5, color="red", ls="--", alpha=0.5, label="5° threshold")
    ax.legend()

    # Panel 4: Koopman linearity comparison
    ax = axes[1, 1]
    x_pos = np.arange(max(8, 5))
    ax.bar(np.arange(8) - 0.15, koopman_r2s, 0.3, label="Spline basis (ψ)", color="coral")
    ax.bar(np.arange(5) + 0.15, poly_koopman_r2s, 0.3, label="Poly {x,y,z,xy,xz}", color="steelblue")
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([f"ψ{j}" for j in range(8)])
    ax.set_ylabel("R²")
    ax.set_title("Test 4: Koopman linearity (one-step R²)")
    ax.legend()
    ax.set_ylim(0.9, 1.005)

    fig.suptitle("Subspace Equivalence: Learned Splines vs Polynomial Dictionary", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "subspace_equivalence.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTDIR / "subspace_equivalence.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved subspace_equivalence.png/pdf")


if __name__ == "__main__":
    main()
