"""
Visualize the level-set + gradient + ideal decomposition
of KAN Deep Koopman latent dimensions.

For each analyzed dimension, produces:
  - Lasso coefficient bar chart (the "ideal generators")
  - Level-set contours: f vs g in 2D slices of (x,y,z)
  - h'(g) scatter + polynomial fit  (outer derivative)
  - h(g) scatter + integrated fit   (outer function)
  - Full reconstruction quality: f̂ = h(g) vs true f
  - Lorenz attractor colored by reconstructed f̂
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from models import DeepKANAutoEncoder, DeepKoopmanCT
from data import simulate_lorenz
from manifold_analysis import full_decomposition, eval_polynomial, eval_h


# ── Helpers ──────────────────────────────────────────────────────────

def load_model():
    ckpt = torch.load(
        os.path.join(os.path.dirname(__file__), "..", "kan_deep_model.pt"),
        map_location="cpu", weights_only=False,
    )
    cfg = ckpt["config"]
    ae = DeepKANAutoEncoder(
        input_size=3, mid_size=cfg["mid_size"],
        hidden_size=cfg["hidden_size"], bottleneck_size=cfg["latent_dim"],
        grid_size=5, spline_order=3,
        grid_range_l1=[-30, 50], grid_range_l2=[-10, 10],
    )
    model = DeepKoopmanCT(ae, latent_dim=cfg["latent_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Per-target plot suite ────────────────────────────────────────────

def plot_decomposition(res, save_prefix=""):
    """Generate the full plot suite for one decomposition result."""
    label = res['label']
    inner = res['inner']
    outer = res['outer']
    ideal = res['ideal']
    f_vals = res['f_vals']
    g_vals = res['g_vals']
    f_hat = res['f_hat']
    x_np = res['x_np']
    x_, y_, z_ = x_np[:, 0], x_np[:, 1], x_np[:, 2]

    # =================================================================
    # Panel 1: Ideal generators — Lasso coefficient bar chart
    # =================================================================
    fig1, ax = plt.subplots(figsize=(10, 4))
    active_names = inner['active_names']
    active_coeffs = inner['active_coeffs']

    if len(active_names) > 0:
        order = np.argsort(np.abs(active_coeffs))[::-1]
        names_sorted = [active_names[i] for i in order]
        coeffs_sorted = [active_coeffs[i] for i in order]
        colors = ['#d62728' if c < 0 else '#1f77b4' for c in coeffs_sorted]
        bars = ax.bar(range(len(names_sorted)), coeffs_sorted, color=colors,
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(names_sorted)))
        ax.set_xticklabels(names_sorted, fontsize=11, rotation=45, ha='right')
        ax.axhline(0, color='black', lw=0.5)
    ax.set_ylabel("Lasso coefficient", fontsize=12)
    ax.set_title(f"{label}:  Ideal Generators  (Lasso, R²={inner['r2']:.4f})\n"
                 f"Type: {ideal['type']}   |   g(x,y,z) = {ideal['expression']}",
                 fontsize=11)
    fig1.tight_layout()
    fig1.savefig(f"{save_prefix}ideal_generators.png", dpi=150,
                 bbox_inches="tight")
    print(f"  Saved {save_prefix}ideal_generators.png")

    # =================================================================
    # Panel 2: Level-set contours — f and g in 2D slices
    # =================================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))

    # Choose slice values near median of each variable
    z_med = np.median(z_)
    y_med = np.median(y_)
    x_med = np.median(x_)

    slice_specs = [
        ("x", "y", z_, z_med, x_, y_, 0, 1),   # fix z
        ("x", "z", y_, y_med, x_, z_, 0, 2),   # fix y
        ("y", "z", x_, x_med, y_, z_, 1, 2),   # fix x
    ]

    for col, (xlab, ylab, fix_var, fix_val, h_var, v_var,
              h_idx, v_idx) in enumerate(slice_specs):

        # Select points near the slice
        tol = np.std(fix_var) * 0.15
        mask = np.abs(fix_var - fix_val) < tol

        if mask.sum() < 50:
            tol = np.std(fix_var) * 0.3
            mask = np.abs(fix_var - fix_val) < tol

        hv, vv = h_var[mask], v_var[mask]
        fv, gv = f_vals[mask], g_vals[mask]

        # Top row: true f
        ax = axes2[0, col]
        sc = ax.scatter(hv, vv, c=fv, s=4, alpha=0.6, cmap='coolwarm',
                        rasterized=True)
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        fix_name = ["x", "y", "z"][[2, 1, 0][col]]
        ax.set_title(f"f = {label}   ({fix_name}≈{fix_val:.1f})",
                     fontsize=10)
        plt.colorbar(sc, ax=ax, shrink=0.8)

        # Bottom row: g (inner polynomial)
        ax = axes2[1, col]
        sc = ax.scatter(hv, vv, c=gv, s=4, alpha=0.6, cmap='coolwarm',
                        rasterized=True)
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(f"g(x,y,z)   ({fix_name}≈{fix_val:.1f})",
                     fontsize=10)
        plt.colorbar(sc, ax=ax, shrink=0.8)

    fig2.suptitle(f"{label}:  Level-Set Contours — f (top) vs g (bottom)",
                  fontsize=13, y=1.01)
    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}level_sets.png", dpi=150,
                 bbox_inches="tight")
    print(f"  Saved {save_prefix}level_sets.png")

    # =================================================================
    # Panel 3: Outer function  h'(g) and h(g)
    # =================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(17, 5))

    # 3a: h'(g) scatter + binned medians + fit
    ax = axes3[0]
    g_samp = outer['g_samples']
    hp_samp = outer['h_prime_samples']
    n = len(g_samp)
    idx = np.random.choice(n, min(3000, n), replace=False)
    ax.scatter(g_samp[idx], hp_samp[idx], s=2, alpha=0.15, color='steelblue',
               rasterized=True)
    # Show binned medians
    if 'h_bin_centers' in outer:
        bc = outer['h_bin_centers'] + outer['g_center']
        bm = outer['h_bin_medians']
        ax.plot(bc, bm, 'ko', ms=3, label="binned median", zorder=5)
    ax.plot(outer['g_plot'], outer['h_prime_fit'], 'r-', lw=2,
            label="polynomial fit")
    ax.set_xlabel("g(x,y,z)", fontsize=11)
    ax.set_ylabel("h'(g)", fontsize=11)
    ax.set_title(f"Outer Derivative  h'(g)", fontsize=11)
    ax.legend(fontsize=9)

    # 3b: h(g) — outer function
    ax = axes3[1]
    ax.scatter(g_samp[idx], outer['f_valid'][idx], s=2, alpha=0.15,
               color='steelblue', label="data: f vs g", rasterized=True)
    h_label = ("h(g) affine" if outer.get('used_affine', False)
               else "h(g) = ∫h'dg")
    ax.plot(outer['g_plot'], outer['h_fit_plot'], 'r-', lw=2,
            label=h_label)
    ax.set_xlabel("g(x,y,z)", fontsize=11)
    ax.set_ylabel("f = h(g)", fontsize=11)
    ax.set_title(f"Outer Function  h(g)", fontsize=11)
    ax.legend(fontsize=9)

    # 3c: Reconstruction scatter  f̂ vs f
    ax = axes3[2]
    ax.scatter(f_hat, f_vals, s=2, alpha=0.2, rasterized=True)
    lim = [min(f_hat.min(), f_vals.min()), max(f_hat.max(), f_vals.max())]
    ax.plot(lim, lim, 'r--', lw=1.5)
    ax.set_xlabel("f̂ = h(g(x,y,z))", fontsize=11)
    ax.set_ylabel(f"f = {label}", fontsize=11)
    affine_tag = " [affine]" if outer.get('used_affine') else ""
    ax.set_title(f"Reconstruction  R²={res['recon_r2']:.4f}{affine_tag}",
                 fontsize=11)

    fig3.suptitle(
        f"{label}:  Outer Function Recovery  "
        f"(g alone R²={res['g_recon_r2']:.4f},  "
        f"h∘g R²={res['recon_r2']:.4f})",
        fontsize=12, y=1.02)
    fig3.tight_layout()
    fig3.savefig(f"{save_prefix}outer_function.png", dpi=150,
                 bbox_inches="tight")
    print(f"  Saved {save_prefix}outer_function.png")

    # =================================================================
    # Panel 4: Lorenz attractor colored by f, g, f̂, residual
    # =================================================================
    fig4 = plt.figure(figsize=(20, 5))

    targets = [
        (f"f = {label}", f_vals),
        ("g(x,y,z) [inner]", g_vals),
        ("f̂ = h(g) [recon]", f_hat),
        ("residual f − f̂", f_vals - f_hat),
    ]
    for i, (title, vals) in enumerate(targets):
        ax = fig4.add_subplot(1, 4, i + 1, projection='3d')
        sub = np.random.choice(len(x_np), min(3000, len(x_np)), replace=False)
        sc = ax.scatter(x_np[sub, 0], x_np[sub, 1], x_np[sub, 2],
                        c=vals[sub], s=2, alpha=0.5,
                        cmap='coolwarm' if i < 3 else 'RdBu_r',
                        rasterized=True)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_zlabel("z", fontsize=8)
        ax.set_title(title, fontsize=9)
        plt.colorbar(sc, ax=ax, shrink=0.6)

    fig4.suptitle(f"{label}:  Lorenz Attractor Coloring", fontsize=12)
    fig4.tight_layout()
    fig4.savefig(f"{save_prefix}lorenz_colored.png", dpi=150,
                 bbox_inches="tight")
    print(f"  Saved {save_prefix}lorenz_colored.png")

    plt.close('all')


# ── Summary comparison plot ──────────────────────────────────────────

def plot_summary(results, save_path="level_set_summary.png"):
    """Comparative summary across all analyzed dimensions."""
    n = len(results)

    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes[:, np.newaxis]

    for col, res in enumerate(results):
        label = res['label']
        inner = res['inner']
        ideal = res['ideal']

        # Top: ideal generators bar chart (compact)
        ax = axes[0, col]
        anames = inner['active_names']
        acoeffs = inner['active_coeffs']
        if len(anames) > 0:
            order = np.argsort(np.abs(acoeffs))[::-1][:10]  # top 10
            names_s = [anames[i] for i in order]
            coeffs_s = [acoeffs[i] for i in order]
            colors = ['#d62728' if c < 0 else '#1f77b4' for c in coeffs_s]
            ax.barh(range(len(names_s)), coeffs_s, color=colors,
                    edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(names_s)))
            ax.set_yticklabels(names_s, fontsize=10)
            ax.invert_yaxis()
        ax.set_xlabel("coefficient", fontsize=10)
        ax.set_title(f"{label}\ng = {ideal['expression'][:50]}...\n"
                     f"R²(g)={inner['r2']:.3f}  R²(h∘g)={res['recon_r2']:.3f}",
                     fontsize=9)

        # Bottom: f vs f̂ scatter
        ax = axes[1, col]
        f_vals = res['f_vals']
        f_hat = res['f_hat']
        idx = np.random.choice(len(f_vals), min(3000, len(f_vals)),
                               replace=False)
        ax.scatter(f_hat[idx], f_vals[idx], s=2, alpha=0.3, rasterized=True)
        lim = [min(f_hat.min(), f_vals.min()),
               max(f_hat.max(), f_vals.max())]
        ax.plot(lim, lim, 'r--', lw=1.5)
        ax.set_xlabel("f̂ = h(g)", fontsize=10)
        ax.set_ylabel(f"f = {label}", fontsize=10)
        ax.set_title(f"Reconstruction  R²={res['recon_r2']:.4f}",
                     fontsize=10)

    fig.suptitle("Level-Set + Gradient Decomposition:  f(x,y,z) = h(g(x,y,z))",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved summary: {save_path}")


# ── Known observable comparison ──────────────────────────────────────

def plot_observable_comparison(results, x_np, save_path="level_set_vs_observables.png"):
    """Compare recovered g to the known Lorenz observables."""
    x_, y_, z_ = x_np[:, 0], x_np[:, 1], x_np[:, 2]
    obs = {
        'x': x_, 'y': y_, 'z': z_,
        'xy': x_*y_, 'xz': x_*z_, 'yz': y_*z_,
        'x^2': x_**2, 'y^2': y_**2, 'z^2': z_**2,
    }

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes[:, np.newaxis]

    for col, res in enumerate(results):
        g_vals = res['g_vals']
        f_vals = res['f_vals']
        label = res['label']

        # Find best-matching observable for g
        best_r2, best_name = -1, "?"
        for oname, oval in obs.items():
            A = np.column_stack([oval, np.ones(len(oval))])
            c, _, _, _ = np.linalg.lstsq(A, g_vals, rcond=None)
            pred = A @ c
            ss = np.sum((g_vals - pred)**2) / max(np.sum((g_vals - g_vals.mean())**2), 1e-12)
            r2 = 1 - ss
            if r2 > best_r2:
                best_r2 = r2
                best_name = oname
                best_pred = pred
                best_coeffs = c

        # Top: g vs best observable
        ax = axes[0, col]
        idx = np.random.choice(len(g_vals), min(3000, len(g_vals)),
                               replace=False)
        ax.scatter(obs[best_name][idx], g_vals[idx], s=2, alpha=0.3,
                   rasterized=True)
        ax.set_xlabel(f"true {best_name}", fontsize=11)
        ax.set_ylabel("g(x,y,z)", fontsize=11)
        ax.set_title(f"{label}: g ≈ {best_coeffs[0]:.3f}·{best_name}"
                     f" + {best_coeffs[1]:.3f}\nR²={best_r2:.4f}",
                     fontsize=10)
        # fit line
        xr = np.array([obs[best_name].min(), obs[best_name].max()])
        ax.plot(xr, best_coeffs[0] * xr + best_coeffs[1], 'r-', lw=2)

        # Bottom: multi-observable regression on g
        ax = axes[1, col]
        all_obs = np.column_stack([v for v in obs.values()])
        all_obs_aug = np.column_stack([all_obs, np.ones(len(all_obs))])
        c_multi, _, _, _ = np.linalg.lstsq(all_obs_aug, g_vals, rcond=None)
        pred_multi = all_obs_aug @ c_multi
        r2_multi = 1 - np.sum((g_vals - pred_multi)**2) / \
                   max(np.sum((g_vals - g_vals.mean())**2), 1e-12)

        obs_names_list = list(obs.keys())
        sig = [(obs_names_list[i], c_multi[i])
               for i in range(len(obs_names_list))
               if abs(c_multi[i]) > 0.001]
        sig.sort(key=lambda t: abs(t[1]), reverse=True)
        sig_str = " + ".join(f"{c:.3f}·{n}" for n, c in sig[:5])

        ax.scatter(pred_multi[idx], g_vals[idx], s=2, alpha=0.3,
                   rasterized=True)
        lim = [min(pred_multi.min(), g_vals.min()),
               max(pred_multi.max(), g_vals.max())]
        ax.plot(lim, lim, 'r--', lw=1.5)
        ax.set_xlabel("multi-obs prediction", fontsize=10)
        ax.set_ylabel("g(x,y,z)", fontsize=10)
        ax.set_title(f"g ≈ {sig_str}\nR²={r2_multi:.4f}", fontsize=9)

    fig.suptitle("Recovered Inner Function g  vs  Known Lorenz Observables",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = load_model()

    # Generate data
    traj = simulate_lorenz(n_traj=20, n_steps=5000, dt=0.01,
                           burn_in=300, device="cpu")
    x_data = traj.reshape(-1, 3)
    idx = torch.randperm(x_data.shape[0])[:5000]
    x_sub = x_data[idx]

    # Analyze these targets:
    #   z[7]  — captures xy (R²=0.862 with stepwise regression)
    #   z[6]  — best interpretable: {z, x, xy, xz} R²=0.998
    #   z[0]  — primarily z (R²=0.914)
    #   z[5]  — z-dominated, cleanest R²=0.993 from {z, x}

    targets = [
        (7, 'latent', "z7"),
        (6, 'latent', "z6"),
        (0, 'latent', "z0"),
        (5, 'latent', "z5"),
    ]

    results = []
    for tidx, ttype, prefix in targets:
        print(f"\n{'='*70}")
        print(f"  Analyzing {ttype} dim {tidx}")
        print(f"{'='*70}")

        res = full_decomposition(
            model, x_sub, tidx, target_type=ttype,
            max_degree=3, lasso_alpha=None, h_degree=4,
        )
        results.append(res)

        # Print ideal summary
        ideal = res['ideal']
        print(f"  Inner function g:  {ideal['expression']}")
        print(f"  Type: {ideal['type']}")
        print(f"  # generators: {ideal['n_active']}, "
              f"# cross-terms: {ideal['n_cross_terms']}")
        print(f"  R²(g alone): {res['g_recon_r2']:.4f}")
        print(f"  R²(h∘g):     {res['recon_r2']:.4f}")

        # Per-target detailed plots
        plot_decomposition(res, save_prefix=f"levelset_{prefix}_")

    # Summary plots
    plot_summary(results)
    plot_observable_comparison(results, x_sub.numpy())

    plt.show()
    print("\nDone.")
