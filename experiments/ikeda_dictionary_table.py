"""
Ikeda Map: Dictionary recovery table across seeds and Lasso alphas.

For each seed, loads the pruned model and runs:
  1. Layer-1 r² signature check (x² and y² coefficient ratio)
  2. Layer-1 foliation alignment (R²_radial of u_j vs r²)
  3. z_k h∘g decomposition (expected dense for bilinear regime)

Reports a summary table matching the format of lorenz_5d_dictionary_table.py.

Usage:
  python experiments/ikeda_dictionary_table.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from ikeda import generate_ikeda
from ikeda_koopman import IkedaStableKoopman, MID_DIM, LATENT_DIM, SEEDS
from manifold_analysis import (
    recover_inner_function_nd, full_decomposition_nd,
)

OUTDIR = Path(__file__).resolve().parent.parent / 'results' / 'ikeda'


def foliation_r2(u_vals, r2_vals, degree=4):
    poly = PolynomialFeatures(degree=degree)
    feat = poly.fit_transform(r2_vals.reshape(-1, 1))
    reg = LinearRegression().fit(feat, u_vals)
    return float(reg.score(feat, u_vals))


def layer1_r2_signature(model, pts_np, alpha=1e-3, max_degree=3):
    """Check which layer-1 nodes have the x²≈y² (r²) signature."""
    pts = torch.tensor(pts_np, dtype=torch.float32)
    with torch.no_grad():
        u = model.encoder.kan1(pts).numpy()

    r2 = pts_np[:, 0]**2 + pts_np[:, 1]**2
    results = []
    for j in range(u.shape[1]):
        # Foliation alignment
        r2_radial = foliation_r2(u[:, j], r2)

        # Lasso inner function
        inner = recover_inner_function_nd(pts_np, ['x', 'y'], u[:, j],
                                           max_degree=max_degree, alpha=alpha)
        x2c = y2c = 0.0
        for nm, cf in zip(inner['active_names'], inner['active_coeffs']):
            if nm == 'x^2': x2c = float(cf)
            elif nm == 'y^2': y2c = float(cf)

        ratio = min(abs(x2c), abs(y2c)) / max(abs(x2c), abs(y2c), 1e-12)
        same_sign = (x2c * y2c > 0) and abs(x2c) > 0.01 and abs(y2c) > 0.01
        sig = same_sign and ratio > 0.5

        results.append({
            'node': j,
            'R2_radial': r2_radial,
            'x2_coeff': x2c,
            'y2_coeff': y2c,
            'ratio': ratio,
            'r2_signature': sig,
            'n_active': len(inner['active']),
        })
    return results


def zk_decomposition(model, pts_np, max_degree=5, alpha=1e-5, n_sub=5000):
    """h∘g decomposition on z_k (expected dense)."""
    from ikeda_koopman import _AEAdapter
    if len(pts_np) > n_sub:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(pts_np), n_sub, replace=False)
        pts_np = pts_np[idx]

    pts = torch.tensor(pts_np, dtype=torch.float32)
    model_wrap = type('W', (), {
        'ae': _AEAdapter(model.encoder, model.decoder)
    })()

    results = []
    for k in range(LATENT_DIM):
        try:
            decomp = full_decomposition_nd(
                model_wrap, pts, k, var_names=['x', 'y'],
                target_type='latent', max_degree=max_degree,
                lasso_alpha=alpha, h_degree=4)
            results.append({
                'dim': k,
                'r2_g': float(decomp['g_recon_r2']),
                'r2_hg': float(decomp['recon_r2']),
                'n_active': len(decomp['inner']['active']),
                'h_affine': bool(decomp['outer'].get('used_affine', False)),
            })
        except Exception as e:
            results.append({'dim': k, 'r2_g': float('nan'), 'r2_hg': float('nan'),
                            'n_active': 0, 'h_affine': True})
    return results


if __name__ == '__main__':
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Shared data
    traj = generate_ikeda(n_traj=200, n_steps=2000, burn_in=500, seed=42)
    n_train = int(0.8 * 200)
    train_pts = traj[:n_train].reshape(-1, 2)

    # Subsample for speed
    rng = np.random.RandomState(0)
    idx = rng.choice(len(train_pts), min(10000, len(train_pts)), replace=False)
    pts_sub = train_pts[idx]

    all_seed_results = {}

    print("="*80)
    print("Ikeda Dictionary Recovery Table")
    print("="*80)

    for seed in SEEDS:
        ckpt_path = OUTDIR / f'ikeda_pruned_seed_{seed}.pt'
        if not ckpt_path.exists():
            print(f"\n  Seed {seed}: checkpoint not found, skipping")
            continue

        print(f"\n{'─'*80}")
        print(f"Seed {seed}")
        print(f"{'─'*80}")

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = IkedaStableKoopman()
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # Layer-1 r² signature
        print("  Layer-1 r² signature:")
        l1 = layer1_r2_signature(model, pts_sub)
        n_sig = sum(1 for r in l1 if r['r2_signature'])
        for r in l1:
            tag = ' ***' if r['r2_signature'] else ''
            print(f"    u[{r['node']}]: R²_rad={r['R2_radial']:.3f}, "
                  f"x²={r['x2_coeff']:+.4f}, y²={r['y2_coeff']:+.4f}, "
                  f"ratio={r['ratio']:.3f}{tag}")
        print(f"    Nodes with r² signature: {n_sig}/{MID_DIM}")

        # z_k decomposition
        print("  z_k decomposition:")
        zk = zk_decomposition(model, pts_sub)
        med_active = np.median([r['n_active'] for r in zk])
        for r in zk:
            print(f"    z[{r['dim']}]: R²(g)={r['r2_g']:.4f}, "
                  f"R²(h∘g)={r['r2_hg']:.4f}, #active={r['n_active']}")
        print(f"    Median active terms: {med_active:.0f}")

        all_seed_results[seed] = {
            'layer1': l1,
            'zk': zk,
            'n_r2_signature': n_sig,
            'median_zk_active': float(med_active),
        }

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("SUMMARY ACROSS SEEDS")
    print(f"{'='*80}")
    print(f"{'Seed':>4s} | {'r² nodes':>8s} | {'med R²_rad':>10s} | "
          f"{'med z_k terms':>13s} | {'best R²(h∘g)':>12s}")
    print(f"{'-'*4} | {'-'*8} | {'-'*10} | {'-'*13} | {'-'*12}")

    for seed in SEEDS:
        if seed not in all_seed_results:
            continue
        sr = all_seed_results[seed]
        best_r2_rad = max(r['R2_radial'] for r in sr['layer1'])
        best_r2_hg = max(r['r2_hg'] for r in sr['zk'] if not np.isnan(r['r2_hg']))
        print(f"  {seed:2d} | {sr['n_r2_signature']:8d} | {best_r2_rad:10.4f} | "
              f"{sr['median_zk_active']:13.0f} | {best_r2_hg:12.4f}")

    # Save
    out = {'seeds': {int(k): v for k, v in all_seed_results.items()}}
    with open(OUTDIR / 'dictionary_table.json', 'w') as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else bool(o) if isinstance(o, np.bool_) else o)
    print(f"\nSaved {OUTDIR / 'dictionary_table.json'}")
