"""
Complete Lorenz analysis: model → spectrum → manifold → splines → eigenmodes.

Single script showing everything about the pre-trained KAN Deep Koopman model.

Sections:
    1. Load model and data
    2. Network architecture summary
    3. Koopman generator spectrum and eigenvalues
    4. One-step prediction quality
    5. Multi-step rollout
    6. Manifold decomposition f = h(g) for each latent dim
    7. Sparsity sweep (which terms survive at each regularization)
    8. KAN spline activations (what each edge learns)
    9. Gradient alignment and geometric viability
   10. Level-set visualizations
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from data import simulate_lorenz
from models import PureDeepKANAutoEncoder, DeepKoopmanCT
from training import set_seed, rollout
from manifold_analysis import (
    full_decomposition, eval_polynomial, eval_h,
    compute_network_gradient, recover_inner_function,
    build_polynomial_dict, grad_polynomial,
)
from dictionary_viability import (
    compute_viability_metrics, compute_verdict, sparsity_sweep,
    gradient_alignment, orthogonal_residual, single_valuedness,
)
from plotting import plot_kan_model

OUTDIR = Path('figs')


# =====================================================================
# 1. Load model
# =====================================================================

def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = torch.load(os.path.join(base, 'old', 'kan_pure_3_3_10_model.pt'),
                      map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    ae = PureDeepKANAutoEncoder(
        input_size=3, mid_size=cfg['mid_size'],
        bottleneck_size=cfg['latent_dim'],
        grid_size=5, spline_order=3,
        grid_range_l1=[-30, 50], grid_range_l2=[-10, 10],
    )
    model = DeepKoopmanCT(ae, latent_dim=cfg['latent_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg, ckpt.get('dt', 0.01)


# =====================================================================
# 2. Network architecture
# =====================================================================

def print_architecture(model, cfg):
    print("=" * 60)
    print("NETWORK ARCHITECTURE (Pure KAN — no ReLU/Dense)")
    print("=" * 60)
    print(f"  Encoder: x ∈ R³")
    print(f"    → KAN1: KANLinear(3 → {cfg['mid_size']})  [univariate splines]")
    print(f"    → KAN2: KANLinear({cfg['mid_size']} → {cfg['latent_dim']})  [mixing → latent]")
    print(f"    → z ∈ R^{cfg['latent_dim']}")
    print(f"  Koopman: z_{{t+1}} = z_t @ expm(A·dt)^T,  A ∈ R^{{{cfg['latent_dim']}×{cfg['latent_dim']}}}")
    print(f"  Decoder: z → KAN → KAN → x̂ ∈ R³")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")


# =====================================================================
# 3. Koopman spectrum
# =====================================================================

def analyze_spectrum(model, dt):
    print("\n" + "=" * 60)
    print("KOOPMAN GENERATOR SPECTRUM")
    print("=" * 60)

    A = model.generator_matrix().detach().numpy()
    eigs_A = np.linalg.eigvals(A)
    order = np.argsort(-np.abs(eigs_A))
    eigs_A = eigs_A[order]

    P = np.linalg.matrix_power(np.eye(len(A)), 1)  # just for expm
    from scipy.linalg import expm
    P = expm(A * dt)
    eigs_P = np.linalg.eigvals(P)
    eigs_P = eigs_P[np.argsort(-np.abs(eigs_P))]

    print(f"\n  {'Mode':>4}  {'Generator μ':>20}  {'|μ|':>8}  {'Propagator λ':>20}  {'|λ|':>8}  {'Decay /s':>10}  {'Freq Hz':>10}")
    print("  " + "-" * 90)
    for i in range(len(eigs_A)):
        mu = eigs_A[i]
        lam = eigs_P[i]
        decay = mu.real
        freq = mu.imag / (2 * np.pi)
        print(f"  {i:4d}  {mu:20.4f}  {abs(mu):8.4f}  {lam:20.4f}  {abs(lam):8.4f}  {decay:10.3f}  {freq:10.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(eigs_A.real, eigs_A.imag, s=100, marker='o',
               facecolors='none', edgecolors='steelblue', lw=2.5, zorder=3)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)
    for e in eigs_A:
        if abs(e.imag) > 0.1:
            ax.annotate(f'{e.real:.1f}{e.imag:+.1f}i', (e.real, e.imag),
                       fontsize=7, textcoords='offset points', xytext=(5, 5))
    ax.set_xlabel('Re(μ)')
    ax.set_ylabel('Im(μ)')
    ax.set_title('Generator A Eigenvalues')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.2)
    ax.scatter(eigs_P.real, eigs_P.imag, s=100, marker='o',
               facecolors='none', edgecolors='steelblue', lw=2.5, zorder=3)
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title('Propagator expm(A·dt) Eigenvalues')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    fig.suptitle('Koopman Spectrum', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'spectrum.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return eigs_A, eigs_P


# =====================================================================
# 4-5. Prediction and rollout
# =====================================================================

def analyze_prediction(model, traj, dt):
    print("\n" + "=" * 60)
    print("PREDICTION QUALITY")
    print("=" * 60)

    # One-step on validation data
    val = traj[48:].reshape(-1, 3)
    with torch.no_grad():
        out = model.forward_one_step(val[:-1], dt)
        pred = out['x_tp1_pred']
        true = val[1:]
        mse = torch.mean((pred - true) ** 2).item()
        recon = out['x_t_rec']
        recon_mse = torch.mean((recon - val[:-1]) ** 2).item()
    print(f"  One-step prediction MSE: {mse:.6f}")
    print(f"  Reconstruction MSE:      {recon_mse:.6f}")

    # Multi-step rollout
    x0 = traj[0, 0]
    xs_pred = rollout(model, x0, dt, 500)
    xs_true = traj[0, :501]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    labels = ['x', 'y', 'z']
    for i, lab in enumerate(labels):
        axes[0, 0].plot(xs_true[:, i].numpy(), alpha=0.8, lw=1)
    axes[0, 0].set_title('True trajectory')
    axes[0, 0].legend(labels)

    for i, lab in enumerate(labels):
        axes[0, 1].plot(xs_pred[:, i].detach().numpy(), alpha=0.8, lw=1)
    axes[0, 1].set_title('Predicted rollout (500 steps)')
    axes[0, 1].legend(labels)

    # 3D phase portrait
    ax3d = fig.add_subplot(2, 2, 3, projection='3d')
    t = xs_true.numpy()
    ax3d.plot(t[:, 0], t[:, 1], t[:, 2], 'b-', alpha=0.5, lw=0.5)
    p = xs_pred.detach().numpy()
    ax3d.plot(p[:, 0], p[:, 1], p[:, 2], 'r-', alpha=0.5, lw=0.5)
    ax3d.set_title('Phase portrait')

    # Per-step error
    err = torch.sqrt(torch.sum((xs_pred[:len(xs_true)] - xs_true[:len(xs_pred)]) ** 2, dim=-1))
    axes[1, 1].semilogy(err.detach().numpy())
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('||error||')
    axes[1, 1].set_title('Rollout error vs time')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'prediction.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# 6. Manifold decomposition for all latent dims
# =====================================================================

def analyze_manifolds(model, x_data):
    print("\n" + "=" * 60)
    print("MANIFOLD DECOMPOSITION: f(x,y,z) = h(g(x,y,z))")
    print("=" * 60)

    latent_dim = model.K.net.weight.shape[0]
    x_np = x_data.numpy()

    print(f"\n  {'Dim':>4}  {'R²(g)':>8}  {'R²(h∘g)':>8}  {'Affine':>6}  {'#Terms':>6}  {'Type':<25}  Top Terms")
    print("  " + "-" * 100)

    all_decomps = []
    for ti in range(latent_dim):
        dr = full_decomposition(model, x_data, ti, target_type='latent',
                                max_degree=3, h_degree=4)
        all_decomps.append(dr)

        ideal = dr['ideal']
        top_terms = dr['inner']['active_names'][:5]
        affine = 'yes' if dr['outer'].get('used_affine', False) else 'no'
        print(f"  z[{ti:2d}]  {dr['g_recon_r2']:8.4f}  {dr['recon_r2']:8.4f}  "
              f"{affine:>6}  {ideal['n_active']:6d}  {ideal['type']:<25}  {top_terms}")

    return all_decomps


# =====================================================================
# 7. Sparsity sweep on best target
# =====================================================================

def analyze_sparsity(x_data, decomps):
    print("\n" + "=" * 60)
    print("SPARSITY SWEEP")
    print("=" * 60)

    # Find best target
    best_idx = max(range(len(decomps)), key=lambda i: decomps[i]['recon_r2'])
    best = decomps[best_idx]
    print(f"  Best target: z[{best_idx}] (R²(h∘g) = {best['recon_r2']:.4f})")

    x_np = x_data.numpy()
    ss = sparsity_sweep(x_np, best['f_vals'], max_degree=3)

    print(f"\n  {'Alpha':>10}  {'#Terms':>7}  {'R²':>8}  Active")
    print("  " + "-" * 70)
    for s in ss['sweep']:
        terms = ', '.join(s['active_names'][:6])
        if len(s['active_names']) > 6:
            terms += f' (+{len(s["active_names"])-6})'
        print(f"  {s['alpha']:10.1e}  {s['n_active']:7d}  {s['r2']:8.4f}  {terms}")

    elbow = ss['elbow']
    print(f"\n  Elbow: {elbow['n_active']} terms at α={elbow['alpha']:.1e} → R²={elbow['r2']:.4f}")
    print(f"  Sparsity ratio: {ss['sparsity_ratio']:.3f} ({elbow['n_active']}/{ss['n_total_terms']})")
    print(f"  Active at elbow: {elbow['active_names']}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    alphas = [s['alpha'] for s in ss['sweep']]
    r2s = [s['r2'] for s in ss['sweep']]
    n_terms = [s['n_active'] for s in ss['sweep']]

    axes[0].semilogx(alphas, r2s, 'o-', color='steelblue', lw=1.5)
    axes[0].axhline(0.9, color='red', ls='--', alpha=0.5)
    axes[0].axvline(elbow['alpha'], color='green', ls=':', alpha=0.7)
    axes[0].set_xlabel('Lasso α')
    axes[0].set_ylabel('R²')
    axes[0].set_title(f'z[{best_idx}]: Sparsity–Accuracy Tradeoff')
    axes[0].set_ylim(-0.1, 1.05)

    axes[1].semilogx(alphas, n_terms, 's-', color='coral', lw=1.5)
    axes[1].axhline(elbow['n_active'], color='green', ls=':', alpha=0.7)
    axes[1].set_xlabel('Lasso α')
    axes[1].set_ylabel('Active Terms')
    axes[1].set_title(f'Sparsity (total possible = {ss["n_total_terms"]})')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'sparsity_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return best_idx, ss


# =====================================================================
# 8. KAN spline activations
# =====================================================================

def analyze_splines(model, x_data):
    print("\n" + "=" * 60)
    print("KAN SPLINE ACTIVATIONS")
    print("=" * 60)

    enc = model.ae.encoder
    kan1 = enc.kan1
    kan2 = enc.kan2
    x = x_data.detach()
    x_np = x.numpy()
    var_names = ['x', 'y', 'z']

    # Layer 1: evaluate each edge
    with torch.no_grad():
        l1_out = kan1(x)

    in1, out1 = kan1.in_features, kan1.out_features
    print(f"\n  Layer 1: {in1} inputs → {out1} nodes")

    # Plot spline curves for each input
    fig, axes = plt.subplots(out1, in1, figsize=(4 * in1, 3 * out1))
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i in range(in1):
        xi_range = np.linspace(x_np[:, i].min(), x_np[:, i].max(), 300)
        xi_t = torch.tensor(xi_range, dtype=torch.float32)
        x_probe = torch.zeros(300, in1)
        # Set other inputs to their mean
        for j in range(in1):
            if j == i:
                x_probe[:, j] = xi_t
            else:
                x_probe[:, j] = float(x_np[:, j].mean())

        with torch.no_grad():
            bases = kan1.b_splines(x_probe)
            ssw = kan1.scaled_spline_weight
            base_act = kan1.base_activation(x_probe)

        for j in range(out1):
            spline = (bases[:, i, :] * ssw[j, i, :].unsqueeze(0)).sum(-1)
            base = kan1.base_weight[j, i] * base_act[:, i]
            edge_out = (spline + base).detach().numpy()

            ax = axes[j, i] if out1 > 1 else axes[i]
            ax.plot(xi_range, edge_out, color=colors[i], lw=1.5)
            ax.set_xlabel(var_names[i], fontsize=8)
            ax.set_ylabel(f'ψ_{{{j},{i}}}', fontsize=8)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_title(f'Input: {var_names[i]}', fontsize=9)

    fig.suptitle('KAN Layer 1: Learned Univariate Splines', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'splines_layer1.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Layer 2: mixing → latent (no ReLU/Dense in pure KAN)
    print(f"  Layer 2: {kan2.in_features} inputs → {kan2.out_features} nodes (mixing → latent)")
    with torch.no_grad():
        l2_out = kan2(l1_out)  # This IS the latent representation

    # Layer 2 weight structure: which layer-1 nodes feed which latent dims
    # Use base_weight + spline_scaler norms as proxy for edge importance
    bw2 = kan2.base_weight.detach().numpy()  # (latent, mid)
    ss2 = kan2.spline_scaler.detach().numpy() if hasattr(kan2, 'spline_scaler') else np.zeros_like(bw2)
    importance = np.abs(bw2) + np.abs(ss2)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(importance, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel(f'Layer 1 node (0-{kan2.in_features-1})')
    ax.set_ylabel(f'Latent dimension z[k]')
    ax.set_title('KAN Layer 2 Edge Importance:\nLayer 1 → Latent Mapping')
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'layer2_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return l1_out, l2_out


# =====================================================================
# 8b. PyKAN-style network diagram
# =====================================================================

def plot_kan_diagrams(model, x_data):
    print("\n  Generating PyKAN-style network diagrams...")

    latent_dim = model.K.net.weight.shape[0]
    z_labels = [f'z{k}' for k in range(latent_dim)]

    # Encoder: x,y,z → ... → z[0..9]
    fig = plot_kan_model(
        model.ae.encoder, x_data,
        plot_mode='total',
        in_vars=['x', 'y', 'z'],
        title='KAN Encoder:  (x, y, z) → latent z',
        figsize=(14, 10),
    )
    fig.savefig(OUTDIR / 'kan_encoder.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved kan_encoder.png")

    # Full autoencoder: x,y,z → z → x̂,ŷ,ẑ
    fig = plot_kan_model(
        model.ae, x_data,
        plot_mode='total',
        in_vars=['x', 'y', 'z'],
        title='KAN Autoencoder:  (x, y, z) → z → (x̂, ŷ, ẑ)',
        figsize=(16, 14),
    )
    fig.savefig(OUTDIR / 'kan_autoencoder.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved kan_autoencoder.png")


# =====================================================================
# 9. Geometric viability for best targets
# =====================================================================

def analyze_viability(decomps):
    print("\n" + "=" * 60)
    print("GEOMETRIC VIABILITY DIAGNOSTICS")
    print("=" * 60)

    print(f"\n  {'Dim':>4}  {'R²(h∘g)':>8}  {'Align':>7}  {'Orth':>7}  {'Sparse':>7}  {'Terms':>5}  Verdict")
    print("  " + "-" * 65)

    for i, dr in enumerate(decomps):
        vm = compute_viability_metrics(dr, run_sparsity_sweep=True, max_degree=3)
        v = compute_verdict(vm)
        mt = vm.get('min_terms_r2_above_90', '?')
        print(f"  z[{i:2d}]  {vm['r2_hg']:8.4f}  "
              f"{vm['alignment']['mean']:7.4f}  "
              f"{vm['orthogonal_residual']['mean_r_perp_norm']:7.4f}  "
              f"{vm.get('sparsity_ratio', float('nan')):7.3f}  "
              f"{str(mt):>5}  {v}")


# =====================================================================
# 10. Level-set visualization for selected targets
# =====================================================================

def plot_level_sets(decomps, targets=(6, 7)):
    print("\n" + "=" * 60)
    print("LEVEL-SET VISUALIZATIONS")
    print("=" * 60)

    for ti in targets:
        dr = decomps[ti]
        label = dr['label']
        inner = dr['inner']
        outer = dr['outer']
        ideal = dr['ideal']
        f_vals = dr['f_vals']
        g_vals = dr['g_vals']
        f_hat = dr['f_hat']
        x_np = dr['x_np']

        print(f"\n  {label}: g = {ideal['expression'][:60]}")
        print(f"    R²(g)={inner['r2']:.4f}, R²(h∘g)={dr['recon_r2']:.4f}")

        fig, axes = plt.subplots(2, 4, figsize=(20, 9))

        # Row 1: Lasso coefficients, f vs g, h'(g), h(g)
        # Lasso coefficients
        ax = axes[0, 0]
        anames = inner['active_names']
        acoeffs = inner['active_coeffs']
        if len(anames) > 0:
            order = np.argsort(np.abs(acoeffs))[::-1][:10]
            names_s = [anames[i] for i in order]
            coeffs_s = [acoeffs[i] for i in order]
            colors = ['#d62728' if c < 0 else '#1f77b4' for c in coeffs_s]
            ax.barh(range(len(names_s)), coeffs_s, color=colors, edgecolor='k', lw=0.5)
            ax.set_yticks(range(len(names_s)))
            ax.set_yticklabels(names_s, fontsize=9)
            ax.invert_yaxis()
        ax.set_xlabel('Coefficient')
        ax.set_title(f'{label}: Ideal Generators\nR²(g)={inner["r2"]:.4f}', fontsize=10)

        # f vs g scatter
        ax = axes[0, 1]
        idx = np.random.choice(len(f_vals), min(3000, len(f_vals)), replace=False)
        ax.scatter(g_vals[idx], f_vals[idx], s=1, alpha=0.2, rasterized=True)
        ax.set_xlabel('g(x,y,z)')
        ax.set_ylabel(f'f = {label}')
        ax.set_title('f vs g (inner)')

        # h'(g)
        ax = axes[0, 2]
        g_s = outer['g_samples']
        hp = outer['h_prime_samples']
        n = len(g_s)
        idx2 = np.random.choice(n, min(3000, n), replace=False)
        ax.scatter(g_s[idx2], hp[idx2], s=1, alpha=0.15, c='steelblue', rasterized=True)
        if 'h_bin_centers' in outer and len(outer['h_bin_centers']) > 0:
            ax.plot(outer['h_bin_centers'] + outer['g_center'],
                    outer['h_bin_medians'], 'ko', ms=3)
        ax.plot(outer['g_plot'], outer['h_prime_fit'], 'r-', lw=2)
        ax.set_xlabel('g')
        ax.set_ylabel("h'(g)")
        ax.set_title("Outer Derivative h'(g)")

        # f vs h(g)
        ax = axes[0, 3]
        ax.scatter(f_hat[idx], f_vals[idx], s=1, alpha=0.2, rasterized=True)
        lim = [min(f_hat.min(), f_vals.min()), max(f_hat.max(), f_vals.max())]
        ax.plot(lim, lim, 'r--', lw=1)
        ax.set_xlabel('h(g(x,y,z))')
        ax.set_ylabel(f'{label}')
        tag = ' [affine]' if outer.get('used_affine') else ''
        ax.set_title(f'Reconstruction R²={dr["recon_r2"]:.4f}{tag}')

        # Row 2: Lorenz colored by f, g, h(g), residual
        targets_color = [
            (f'{label}', f_vals, 'coolwarm'),
            ('g(x,y,z)', g_vals, 'coolwarm'),
            ('h(g) recon', f_hat, 'coolwarm'),
            ('residual f−h(g)', f_vals - f_hat, 'RdBu_r'),
        ]
        sub = np.random.choice(len(x_np), min(3000, len(x_np)), replace=False)
        for col, (title, vals, cmap) in enumerate(targets_color):
            ax = axes[1, col]
            sc = ax.scatter(x_np[sub, 0], x_np[sub, 1], c=vals[sub],
                           s=2, alpha=0.5, cmap=cmap, rasterized=True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(title, fontsize=9)
            plt.colorbar(sc, ax=ax, shrink=0.8)

        fig.suptitle(f'{label}:  f(x,y,z) = h(g(x,y,z))  —  '
                     f'g = {ideal["expression"][:40]}', fontsize=12)
        fig.tight_layout()
        fig.savefig(OUTDIR / f'levelset_z{ti}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved levelset_z{ti}.png")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    # 1. Load
    print("Loading model...")
    model, cfg, dt = load_model()

    # 2. Architecture
    print_architecture(model, cfg)

    # 3. Generate data
    print("\nGenerating Lorenz data...")
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=dt, burn_in=500)
    flat = traj.reshape(-1, 3)
    idx = torch.randperm(flat.shape[0])[:5000]
    x_data = flat[idx]
    print(f"  Trajectories: {traj.shape}, Analysis sample: {x_data.shape}")

    # 3. Spectrum
    eigs_A, eigs_P = analyze_spectrum(model, dt)

    # 4-5. Prediction
    analyze_prediction(model, traj, dt)

    # 6. Manifold decomposition
    decomps = analyze_manifolds(model, x_data)

    # 7. Sparsity sweep
    best_idx, ss = analyze_sparsity(x_data, decomps)

    # 8. Splines
    analyze_splines(model, x_data)

    # 8b. PyKAN-style diagrams
    plot_kan_diagrams(model, x_data)

    # 9. Viability
    analyze_viability(decomps)

    # 10. Level sets
    plot_level_sets(decomps, targets=(best_idx, 6, 7))

    print(f"\nAll outputs in {OUTDIR}/")
    print("Done.")
