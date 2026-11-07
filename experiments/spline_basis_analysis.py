"""
Spline-basis dictionary analysis for stable Lorenz model.

Instead of fitting the encoder output with polynomial monomials {x, y, z, xy, ...},
use the LEARNED Layer-1 spline functions as the dictionary basis. Layer 2 of the KAN
computes z_k = sum_j phi_{k,j}(psi_j), where psi_j are Layer-1 outputs. The products
psi_i * psi_j through the mixing layer are the actual observables — NOT x*y.

This script:
  1. Extracts Layer-1 spline activations psi_0...psi_7 (8 mid nodes, each a fn of x,y,z)
  2. Plots Layer-2 splines: each edge phi_{k,j}(psi_j) for every latent dim k
  3. Builds a dictionary from products of Layer-1 outputs: {psi_i, psi_i*psi_j}
  4. Fits each latent dim with Lasso on this spline-product dictionary
  5. Compares R² and sparsity to the polynomial dictionary approach

Run from project root:
    python experiments/spline_basis_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from itertools import combinations_with_replacement
from scipy.optimize import curve_fit
from sklearn.linear_model import Lasso

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import simulate_lorenz
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition

OUTDIR = Path("results/spline_basis_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)
DT = 0.01
TRUE_SUPPORT = {"x", "y", "z", "xy", "xz"}

# Symbolic candidate library for identifying Layer-1 splines
SYMBOLIC_CANDIDATES = {
    "linear":    (lambda x, a, b: a * x + b, 2),
    "quadratic": (lambda x, a, b, c: a * x**2 + b * x + c, 3),
    "cubic":     (lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, 4),
    "sin":       (lambda x, a, b, c, d: a * np.sin(b * x + c) + d, 4),
    "cos":       (lambda x, a, b, c, d: a * np.cos(b * x + c) + d, 4),
    "tanh":      (lambda x, a, b, c, d: a * np.tanh(b * x + c) + d, 4),
    "exp":       (lambda x, a, b, c: a * np.exp(np.clip(b * x, -10, 10)) + c, 3),
    "abs":       (lambda x, a, b, c: a * np.abs(x - b) + c, 3),
}


def load_model():
    """Load the stable Lorenz checkpoint."""
    latent_dim = 32
    ae = PyKANAutoEncoder(3, 8, latent_dim, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, latent_dim)
    ckpt = torch.load("results/lorenz_stable/stable_8_32_model.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, latent_dim


def get_layer1_activations(model, x_data):
    """Get Layer-1 output activations (the 8 mid-node values).

    For PyKAN encoder, a forward pass populates model.ae.encoder.acts,
    where acts[1] is the post-Layer-1 activation (shape: N x mid_size).
    """
    enc = model.ae.encoder
    with torch.no_grad():
        enc(x_data)
        if hasattr(enc, 'acts') and len(enc.acts) > 1:
            psi = enc.acts[1].numpy()  # (N, 8)
        else:
            raise RuntimeError("Cannot access Layer-1 activations from PyKAN encoder")
    return psi


def get_layer2_activations(model, x_data):
    """Get Layer-2 output activations (the latent_dim values).

    acts[2] is the post-Layer-2 activation (shape: N x latent_dim).
    """
    enc = model.ae.encoder
    with torch.no_grad():
        enc(x_data)
        if hasattr(enc, 'acts') and len(enc.acts) > 2:
            z = enc.acts[2].numpy()
        else:
            z = enc(x_data).numpy()
    return z


def identify_layer1_splines(model, x_data):
    """Identify what symbolic function each Layer-1 edge computes.

    Returns:
        spline_info: dict (j, i) -> (name, r2, params)
            j = mid node index (0..7), i = input index (0=x, 1=y, 2=z)
    """
    enc = model.ae.encoder
    x_np = x_data.numpy()
    var_names = ["x", "y", "z"]
    mid_size = 8
    results = {}

    for i, vn in enumerate(var_names):
        xi_range = np.linspace(x_np[:, i].min(), x_np[:, i].max(), 500)
        x_probe = torch.zeros(500, 3)
        for k in range(3):
            x_probe[:, k] = float(x_np[:, k].mean())
        x_probe[:, i] = torch.tensor(xi_range, dtype=torch.float32)

        with torch.no_grad():
            enc(x_probe)
            if hasattr(enc, 'acts') and len(enc.acts) > 1:
                l1_out = enc.acts[1].numpy()  # (500, 8)
            else:
                continue

        for j in range(mid_size):
            edge_out = l1_out[:, j]
            edge_range = edge_out.max() - edge_out.min()
            if edge_range < 0.01:
                results[(j, i)] = ("dead", 0.0, {})
                continue

            best_name, best_r2, best_params = "unknown", -999, {}
            for cname, (cfn, nparams) in SYMBOLIC_CANDIDATES.items():
                try:
                    p0 = [1.0] * nparams
                    popt, _ = curve_fit(cfn, xi_range, edge_out, p0=p0, maxfev=5000)
                    pred = cfn(xi_range, *popt)
                    ss_res = np.sum((edge_out - pred) ** 2)
                    ss_tot = np.sum((edge_out - edge_out.mean()) ** 2)
                    r2 = 1 - ss_res / max(ss_tot, 1e-12)
                    if r2 > best_r2:
                        best_r2, best_name = r2, cname
                        best_params = dict(zip(["a", "b", "c", "d"][:nparams], popt))
                except Exception:
                    continue
            results[(j, i)] = (best_name, best_r2, best_params)

    return results


def plot_layer2_splines(model, x_data, latent_dim):
    """Plot Layer-2 spline functions: phi_{k,j}(psi_j) for each latent dim k.

    Each subplot shows how latent dim k depends on mid-node j,
    with psi_j varying over its data range and all other psi held at mean.
    """
    enc = model.ae.encoder
    mid_size = 8

    # Get Layer-1 activations on real data to know the psi ranges
    psi_data = get_layer1_activations(model, x_data)  # (N, 8)
    psi_means = psi_data.mean(axis=0)
    psi_mins = psi_data.min(axis=0)
    psi_maxs = psi_data.max(axis=0)

    # For each mid node j, sweep psi_j while holding others at mean
    n_probe = 300
    edge_responses = {}  # (k, j) -> (psi_range, response)

    for j in range(mid_size):
        psi_range = np.linspace(psi_mins[j], psi_maxs[j], n_probe)
        psi_probe = np.tile(psi_means, (n_probe, 1))
        psi_probe[:, j] = psi_range
        psi_tensor = torch.tensor(psi_probe, dtype=torch.float32)

        # Pass through Layer-2 only
        with torch.no_grad():
            # PyKAN: we need to call the second layer directly
            # enc.act_fun[1] is Layer-2's activation function
            # But PyKAN doesn't expose layers separately easily.
            # Instead: do full forward, then do it with psi_j at mean → subtract
            # Actually, simplest: use the full encoder but inject Layer-1 outputs.
            # PyKAN stores acts after forward. We can use the symbolic structure.
            pass

    # Alternative approach: finite-difference on the full encoder
    # Vary one psi_j at a time by perturbing the input to Layer-2
    # Since we can't easily inject into Layer-2 of PyKAN, we'll compute
    # the Layer-2 edge functions numerically from data.

    # For each latent dim k, compute correlation with each psi_j
    z_data = get_layer2_activations(model, x_data)  # (N, latent_dim)

    # Plot: for each latent dim, show z_k vs psi_j scatter (top 3 by correlation)
    n_plot_dims = min(latent_dim, 16)  # plot top 16 latent dims by R²

    # Get R² from full_decomposition to rank dims
    x_sub_idx = np.random.choice(len(x_data), min(3000, len(x_data)), replace=False)
    x_sub = x_data[x_sub_idx]
    psi_sub = psi_data[x_sub_idx]
    z_sub = z_data[x_sub_idx]

    # Rank by variance (active dims)
    z_var = np.var(z_sub, axis=0)
    ranked_dims = np.argsort(-z_var)[:n_plot_dims]

    fig, axes = plt.subplots(n_plot_dims, mid_size, figsize=(2.5 * mid_size, 2.5 * n_plot_dims))
    if n_plot_dims == 1:
        axes = axes[None, :]

    for row_idx, k in enumerate(ranked_dims):
        for j in range(mid_size):
            ax = axes[row_idx, j]

            # Sort by psi_j for clean plotting
            sort_idx = np.argsort(psi_sub[:, j])
            psi_sorted = psi_sub[sort_idx, j]
            z_sorted = z_sub[sort_idx, k]

            # Bin and take median for a clean curve
            n_bins = 50
            bin_edges = np.linspace(psi_sorted.min(), psi_sorted.max(), n_bins + 1)
            bin_centers = []
            bin_medians = []
            for b in range(n_bins):
                mask = (psi_sorted >= bin_edges[b]) & (psi_sorted < bin_edges[b + 1])
                if mask.sum() > 5:
                    bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                    bin_medians.append(np.median(z_sorted[mask]))

            if bin_centers:
                ax.plot(bin_centers, bin_medians, "b-", lw=1.5, alpha=0.8)

            # Correlation
            if np.std(psi_sub[:, j]) > 1e-8 and np.std(z_sub[:, k]) > 1e-8:
                corr = np.corrcoef(psi_sub[:, j], z_sub[:, k])[0, 1]
            else:
                corr = 0.0

            ax.set_title(f"z[{k}] vs psi[{j}]\nr={corr:.2f}", fontsize=7)
            ax.tick_params(labelsize=5)
            if row_idx == 0:
                ax.set_xlabel(f"psi_{j}", fontsize=7)

    fig.suptitle("Layer-2 Mixing: z_k vs psi_j (binned median)", fontsize=14, y=1.001)
    fig.tight_layout()
    fig.savefig(OUTDIR / "layer2_splines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved layer2_splines.png ({n_plot_dims} x {mid_size} grid)")

    return ranked_dims


def build_spline_product_dictionary(psi_data, max_degree=2):
    """Build dictionary from products of Layer-1 activations.

    psi_data: (N, mid_size) — the 8 mid-node activations
    Returns:
        M: (N, n_features) design matrix
        names: list of feature names like 'psi0', 'psi0*psi1', etc.
    """
    N, d = psi_data.shape
    features = []
    names = []

    # Constant
    features.append(np.ones(N))
    names.append("1")

    # Linear: psi_j
    for j in range(d):
        features.append(psi_data[:, j])
        names.append(f"psi{j}")

    # Degree 2: psi_i * psi_j
    if max_degree >= 2:
        for i in range(d):
            for j in range(i, d):
                features.append(psi_data[:, i] * psi_data[:, j])
                if i == j:
                    names.append(f"psi{i}^2")
                else:
                    names.append(f"psi{i}*psi{j}")

    return np.column_stack(features), names


def fit_with_spline_dictionary(z_data, psi_data, latent_dim, alpha=1e-5):
    """Fit each latent dim with Lasso on the spline-product dictionary.

    Returns per-dim results: coefficients, R², active terms.
    """
    M, names = build_spline_product_dictionary(psi_data, max_degree=2)

    # Normalize
    col_mean = M.mean(axis=0)
    col_std = M.std(axis=0)
    col_std[col_std < 1e-12] = 1.0
    M_scaled = (M - col_mean) / col_std

    results = []
    for k in range(latent_dim):
        target = z_data[:, k]
        t_mean = target.mean()
        t_std = target.std()
        if t_std < 1e-12:
            results.append({"r2": 0.0, "active": [], "n_active": 0, "coeffs": {}})
            continue

        t_scaled = (target - t_mean) / t_std

        lasso = Lasso(alpha=alpha, max_iter=50000, fit_intercept=True)
        lasso.fit(M_scaled, t_scaled)

        # Unscale coefficients
        coeffs_orig = t_std * lasso.coef_ / col_std
        intercept = t_mean + t_std * (lasso.intercept_ - np.sum(lasso.coef_ * col_mean / col_std))

        # Prediction and R²
        pred = M @ coeffs_orig + intercept
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - t_mean) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)

        # Active terms (excluding constant)
        threshold = 1e-4
        active = [(names[j], coeffs_orig[j])
                  for j in range(len(names))
                  if abs(coeffs_orig[j]) > threshold and names[j] != "1"]
        active.sort(key=lambda x: -abs(x[1]))

        cdict = dict(zip(names, coeffs_orig))
        cdict["1"] = intercept

        results.append({
            "r2": r2,
            "active": active,
            "n_active": len(active),
            "coeffs": cdict,
        })

    return results, names


def map_spline_products_to_state(spline_info, mid_size=8):
    """Map psi products back to state-space meaning.

    Given Layer-1 spline identifications, determine what each psi_i*psi_j
    corresponds to in terms of (x, y, z) interactions.

    Returns: dict mapping 'psiI*psiJ' -> description string
    """
    var_names = ["x", "y", "z"]
    mappings = {}

    for j in range(mid_size):
        # Which input variables does psi_j depend on?
        deps = []
        for i in range(3):
            info = spline_info.get((j, i), ("dead", 0, {}))
            fname, r2, params = info
            if fname != "dead" and r2 > 0.5:
                deps.append((i, var_names[i], fname, r2))

        # Sort by R² to find dominant input
        deps.sort(key=lambda x: -x[3])
        if deps:
            dominant = deps[0]
            mappings[f"psi{j}"] = f"{dominant[2]}({dominant[1]})"
        else:
            mappings[f"psi{j}"] = "dead"

    # Products
    for i in range(mid_size):
        for j2 in range(i, mid_size):
            m_i = mappings.get(f"psi{i}", "?")
            m_j = mappings.get(f"psi{j2}", "?")
            if i == j2:
                mappings[f"psi{i}^2"] = f"({m_i})^2"
            else:
                mappings[f"psi{i}*psi{j2}"] = f"{m_i} * {m_j}"

    return mappings


def main():
    print("=" * 70)
    print("  SPLINE-BASIS DICTIONARY ANALYSIS")
    print("=" * 70)

    # Load model and data
    model, latent_dim = load_model()
    print(f"  Model: PyKAN [3, 8, {latent_dim}] stable")

    set_seed(42)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:48]
    x_flat = train_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_data = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    print(f"  Data: {x_data.shape}")

    # ══════════════════════════════════════════════════════════════════
    # 1. Identify Layer-1 spline functions
    # ══════════════════════════════════════════════════════════════════
    print("\n1. Identifying Layer-1 splines...")
    spline_info = identify_layer1_splines(model, x_data)

    print(f"\n  {'Edge':>12}  {'Function':>10}  {'R2':>6}")
    print("  " + "-" * 35)
    for j in range(8):
        for i, vn in enumerate(["x", "y", "z"]):
            fname, r2, params = spline_info.get((j, i), ("?", 0, {}))
            if fname != "dead":
                print(f"  psi({vn}->{j})  {fname:>10}  {r2:6.3f}")

    # ══════════════════════════════════════════════════════════════════
    # 2. Plot Layer-2 splines
    # ══════════════════════════════════════════════════════════════════
    print("\n2. Plotting Layer-2 mixing structure...")
    psi_data = get_layer1_activations(model, x_data)
    z_data = get_layer2_activations(model, x_data)
    ranked_dims = plot_layer2_splines(model, x_data, latent_dim)

    # ══════════════════════════════════════════════════════════════════
    # 3. Spline-product dictionary vs polynomial dictionary
    # ══════════════════════════════════════════════════════════════════
    print("\n3. Comparing dictionaries...")

    # Spline-product dictionary: {psi_i, psi_i*psi_j}
    spline_results, spline_names = fit_with_spline_dictionary(
        z_data, psi_data, latent_dim, alpha=1e-5)

    # Polynomial dictionary (for comparison): use full_decomposition
    poly_r2s = []
    poly_sparsities = []
    for k in range(latent_dim):
        try:
            res = full_decomposition(model, x_data, k,
                                      target_type="latent", max_degree=2)
            poly_r2s.append(res.get("recon_r2", 0.0))
            active = [n for n in res["inner"]["active_names"] if n != "1"]
            poly_sparsities.append(len(active))
        except Exception:
            poly_r2s.append(0.0)
            poly_sparsities.append(0)

    # Print comparison
    print(f"\n  {'Dim':>4}  {'Poly R2':>8}  {'Poly #':>6}  "
          f"{'Spline R2':>10}  {'Spline #':>8}  Top spline terms")
    print("  " + "-" * 85)

    n_spline_better = 0
    for k in range(latent_dim):
        pr2 = poly_r2s[k]
        pn = poly_sparsities[k]
        sr = spline_results[k]
        sr2 = sr["r2"]
        sn = sr["n_active"]

        if sr2 > pr2 + 0.001:
            n_spline_better += 1

        top_terms = sr["active"][:4]
        top_str = ", ".join(f"{n}={c:.3f}" for n, c in top_terms)

        marker = " *" if sr2 > pr2 + 0.01 else ""
        print(f"  z[{k:2d}]  {pr2:8.3f}  {pn:6d}  {sr2:10.3f}  {sn:8d}  {top_str}{marker}")

    # Summary stats
    spline_r2s = [sr["r2"] for sr in spline_results]
    spline_spars = [sr["n_active"] for sr in spline_results]

    print(f"\n  Summary:")
    print(f"    Polynomial dict: mean R²={np.mean(poly_r2s):.4f}, "
          f"mean sparsity={np.mean(poly_sparsities):.1f}/9")
    print(f"    Spline dict:     mean R²={np.mean(spline_r2s):.4f}, "
          f"mean sparsity={np.mean(spline_spars):.1f}/{len(spline_names)-1}")
    print(f"    Spline better in {n_spline_better}/{latent_dim} dims")

    # ══════════════════════════════════════════════════════════════════
    # 4. Map spline products back to state-space meaning
    # ══════════════════════════════════════════════════════════════════
    print("\n4. State-space interpretation of spline products...")
    name_map = map_spline_products_to_state(spline_info)

    print(f"\n  Spline-product -> state-space mapping:")
    for sname, meaning in sorted(name_map.items()):
        if "dead" not in meaning:
            print(f"    {sname:>15}  =  {meaning}")

    # For well-reconstructed dims, show the dictionary in state-space terms
    print(f"\n  Discovered dictionary (state-space interpretation):")
    print(f"  {'Dim':>4}  {'R2':>6}  Terms (state-space)")
    print("  " + "-" * 70)
    for k in range(latent_dim):
        sr = spline_results[k]
        if sr["r2"] < 0.9:
            continue
        terms = sr["active"][:5]
        interpreted = []
        for tname, tcoeff in terms:
            meaning = name_map.get(tname, tname)
            interpreted.append(f"{tcoeff:+.3f}*{meaning}")
        print(f"  z[{k:2d}]  {sr['r2']:.3f}  {'  '.join(interpreted)}")

    # ══════════════════════════════════════════════════════════════════
    # 5. Comparison plot
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: R² comparison
    ax = axes[0]
    dims = np.arange(latent_dim)
    ax.bar(dims - 0.2, poly_r2s, 0.35, label="Polynomial", color="steelblue", alpha=0.8)
    ax.bar(dims + 0.2, spline_r2s, 0.35, label="Spline-product", color="coral", alpha=0.8)
    ax.axhline(0.9, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("R²")
    ax.set_title("Reconstruction quality")
    ax.legend(fontsize=9)

    # Panel 2: Sparsity comparison
    ax = axes[1]
    ax.bar(dims - 0.2, poly_sparsities, 0.35, label="Polynomial", color="steelblue", alpha=0.8)
    ax.bar(dims + 0.2, spline_spars, 0.35, label="Spline-product", color="coral", alpha=0.8)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("# active terms")
    ax.set_title("Sparsity (lower = better)")
    ax.legend(fontsize=9)

    # Panel 3: R² vs sparsity scatter
    ax = axes[2]
    ax.scatter(poly_sparsities, poly_r2s, label="Polynomial", color="steelblue",
               alpha=0.6, s=30)
    ax.scatter(spline_spars, spline_r2s, label="Spline-product", color="coral",
               alpha=0.6, s=30)
    ax.set_xlabel("# active terms")
    ax.set_ylabel("R²")
    ax.set_title("R² vs sparsity tradeoff")
    ax.legend(fontsize=9)

    fig.suptitle("Polynomial vs Spline-Product Dictionary", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "dictionary_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved dictionary_comparison.png")

    # Save results
    import json
    save_data = {
        "polynomial": {"r2": poly_r2s, "sparsity": poly_sparsities},
        "spline": {
            "r2": spline_r2s,
            "sparsity": spline_spars,
            "per_dim": [
                {"r2": sr["r2"], "n_active": sr["n_active"],
                 "top_terms": [(n, float(c)) for n, c in sr["active"][:5]]}
                for sr in spline_results
            ],
        },
        "spline_info": {
            f"psi({['x','y','z'][i]}->{j})": {"fn": fname, "r2": r2}
            for (j, i), (fname, r2, _) in spline_info.items()
            if fname != "dead"
        },
        "name_map": {k: v for k, v in name_map.items() if "dead" not in v},
    }
    with open(OUTDIR / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved results.json")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
