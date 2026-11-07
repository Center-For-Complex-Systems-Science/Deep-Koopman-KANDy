"""
Experiment: Sparsity-regularized encoder + decoder analysis.

Two questions:
  1. Does adding L1 sparsity regularization to KAN spline weights force the
     encoder to carve out dedicated latent dims for cross-terms (xy, xz)?
  2. What polynomial structure does the DECODER learn in z-space? Is the
     decoder doing nonlinear work that the encoder analysis misses?

Run from project root:
    python experiments/sparsity_decoder_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import simulate_lorenz, PairDataset, lorenz_rhs
from models import KANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition
from sklearn.linear_model import Lasso

# ── Config ──
DT = 0.01
N_TRAJ = 60
N_TRAIN = 48
LATENT_DIM = 19
TRUE_SUPPORT = {"x", "y", "z", "xy", "xz"}
NAMES_D2 = ["1", "x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]
OUTDIR = Path("results/sparsity_decoder_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"


# ── Data ──
def generate_data():
    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:N_TRAIN].contiguous()
    val_traj = traj[N_TRAIN:].contiguous()
    return train_traj, val_traj


# ── Training with optional sparsity regularization ──
def train_model(train_traj, val_traj, sparsity_weight=0.0, epochs=150, seed=0):
    """Train 2L-KAN-stable with optional L1 sparsity on KAN spline weights."""
    set_seed(seed)
    ae = KANAutoEncoder(3, 8, LATENT_DIM, grid_size=5, spline_order=3,
                         grid_range_l1=[-30, 50], grid_range_l2=[-10, 10])
    traj_full = torch.cat([train_traj, val_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    val_dl = DataLoader(PairDataset(traj_full[N_TRAIN:]), batch_size=2048)

    model = DeepKoopmanCTStable(ae, LATENT_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x_t, x_tp1 in train_dl:
            out = model.forward_one_step(x_t, DT)
            z_true = model.ae.encoder(x_tp1)
            loss = (mse_fn(out["x_tp1_pred"], x_tp1)
                    + mse_fn(out["z_tp1_pred"], z_true)
                    + 0.5 * mse_fn(out["x_t_rec"], x_t))

            # L1 sparsity on KAN spline weights in encoder
            if sparsity_weight > 0:
                l1 = 0.0
                for name, param in model.ae.encoder.named_parameters():
                    if "spline_weight" in name or "base_weight" in name:
                        l1 = l1 + param.abs().mean()
                loss = loss + sparsity_weight * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        if ep == 1 or ep % 50 == 0 or ep == epochs:
            model.eval()
            vloss = []
            with torch.no_grad():
                for x_t, x_tp1 in val_dl:
                    out = model.forward_one_step(x_t, DT)
                    z_true = model.ae.encoder(x_tp1)
                    l = (mse_fn(out["x_tp1_pred"], x_tp1)
                         + mse_fn(out["z_tp1_pred"], z_true)
                         + 0.5 * mse_fn(out["x_t_rec"], x_t))
                    vloss.append(l.item())
            print(f"    Ep {ep:3d} | train {np.mean(losses):.5f} | val {np.mean(vloss):.5f}")

    model.eval()
    return model


# ── Encoder analysis ──
def analyze_encoder(model, x_sub):
    """Level-set decomposition of each latent dim. Returns support set and per-dim coefficients."""
    model_cpu = model.cpu().eval()
    all_coeffs = []
    support = set()
    sparsities = []

    for zi in range(LATENT_DIM):
        try:
            res = full_decomposition(model_cpu, x_sub, zi,
                                      target_type="latent", max_degree=2)
            r2 = res.get("recon_r2", 0.0)
            coeffs = res["inner"]["coeffs"]
            names = res["inner"]["names"]
            coeff_dict = dict(zip(names, coeffs))
            row = [coeff_dict.get(n, 0.0) for n in NAMES_D2]
            all_coeffs.append(row)

            active = [n for n in res["inner"]["active_names"] if n != "1"]
            sparsities.append(len(active))
            if r2 > 0.8:
                support.update(active)
        except Exception as e:
            all_coeffs.append([0.0] * len(NAMES_D2))
            sparsities.append(0)

    return support, np.array(all_coeffs), sparsities


# ── Decoder analysis ──
def analyze_decoder(model, x_sub):
    """Analyze decoder: what polynomial of z reconstructs each state variable?"""
    model_cpu = model.cpu().eval()

    with torch.no_grad():
        z_sub = model_cpu.ae.encoder(x_sub).numpy()
        x_rec = model_cpu.ae.decoder(torch.tensor(z_sub, dtype=torch.float32)).numpy()
    x_true = x_sub.numpy()

    recon_mse = np.mean((x_rec - x_true) ** 2)
    d = z_sub.shape[1]

    # Build linear features in z
    N = len(z_sub)
    lin_feats = [np.ones((N, 1))]
    lin_names = ["1"]
    for i in range(d):
        lin_feats.append(z_sub[:, i:i+1])
        lin_names.append(f"z{i}")
    M_lin = np.column_stack(lin_feats) if lin_feats else np.ones((N, 1))
    # Fix: column_stack needs proper shape
    M_lin = np.column_stack([f.ravel() for f in lin_feats])

    # Build quadratic features in z
    quad_feats = []
    quad_names = []
    for i in range(d):
        for j in range(i, d):
            quad_feats.append(z_sub[:, i] * z_sub[:, j])
            quad_names.append(f"z{i}*z{j}" if i != j else f"z{i}^2")
    M_quad = np.column_stack([M_lin] + [q.reshape(-1, 1) for q in quad_feats])
    all_names = lin_names + quad_names

    results = {"recon_mse": recon_mse, "dims": {}}

    for dim, dim_name in enumerate(["x", "y", "z"]):
        target = x_true[:, dim]

        # Linear fit
        coeffs_lin, _, _, _ = np.linalg.lstsq(M_lin, target, rcond=None)
        pred_lin = M_lin @ coeffs_lin
        ss_res_lin = np.sum((target - pred_lin) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r2_lin = 1 - ss_res_lin / ss_tot

        # Quadratic fit via Lasso
        lasso = Lasso(alpha=1e-4, max_iter=50000, fit_intercept=False)
        lasso.fit(M_quad, target)
        pred_quad = M_quad @ lasso.coef_
        ss_res_quad = np.sum((target - pred_quad) ** 2)
        r2_quad = 1 - ss_res_quad / ss_tot

        # Count active quadratic terms
        n_lin_terms = len(lin_names)
        quad_active = [(abs(lasso.coef_[n_lin_terms + j]), quad_names[j], lasso.coef_[n_lin_terms + j])
                       for j in range(len(quad_names)) if abs(lasso.coef_[n_lin_terms + j]) > 1e-4]
        quad_active.sort(reverse=True)

        results["dims"][dim_name] = {
            "r2_linear": r2_lin,
            "r2_quadratic": r2_quad,
            "n_quad_active": len(quad_active),
            "top_quad": quad_active[:10],
        }

    return results


# ── Composed model analysis ──
def analyze_composed(model, x_sub):
    """What polynomial of (x,y,z) does decoder(expm(A·dt)·encoder(x)) compute?"""
    model_cpu = model.cpu().eval()

    with torch.no_grad():
        out = model_cpu.forward_one_step(x_sub, DT)
        x_pred = out["x_tp1_pred"].numpy()

    x_np = x_sub.numpy()
    x, y, z = x_np[:, 0:1], x_np[:, 1:2], x_np[:, 2:3]
    M = np.column_stack([np.ones_like(x), x, y, z,
                          x**2, x*y, x*z, y**2, y*z, z**2])
    state_names = ["1", "x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]

    results = {}
    for dim, dim_name in enumerate(["x", "y", "z"]):
        target = x_pred[:, dim]
        lasso = Lasso(alpha=1e-6, max_iter=50000, fit_intercept=False)
        lasso.fit(M, target)
        pred = M @ lasso.coef_
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        active = [(abs(lasso.coef_[j]), state_names[j], lasso.coef_[j])
                  for j in range(len(state_names)) if abs(lasso.coef_[j]) > 1e-5]
        active.sort(reverse=True)
        results[dim_name] = {"r2": r2, "terms": active}

    return results


# ── Main ──
def main():
    train_traj, val_traj = generate_data()

    # Sample points for analysis
    x_flat = train_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)

    # True one-step coefficients for reference
    print("True one-step map (Euler, dt=0.01):")
    print(f"  x_{{t+1}} = {1-10*DT:.4f}*x + {10*DT:.4f}*y")
    print(f"  y_{{t+1}} = {28*DT:.4f}*x + {1-DT:.4f}*y + {-DT:.4f}*xz")
    print(f"  z_{{t+1}} = {DT:.4f}*xy + {1-8/3*DT:.4f}*z")

    # ════════════════════════════════════════════════════════════
    # Sweep sparsity weights: 0 (baseline), 0.001, 0.01, 0.1
    # ════════════════════════════════════════════════════════════
    sparsity_weights = [0.0, 0.001, 0.01, 0.1]

    all_results = {}
    for sw in sparsity_weights:
        tag = f"sw={sw}"
        print(f"\n{'='*70}")
        print(f"  SPARSITY WEIGHT = {sw}")
        print(f"{'='*70}")

        model = train_model(train_traj, val_traj, sparsity_weight=sw)

        # ── One-step MSE ──
        ds = PairDataset(val_traj)
        dl = DataLoader(ds, batch_size=4096)
        total_se = 0.0
        n = 0
        with torch.no_grad():
            for xt, xtp1 in dl:
                out = model.forward_one_step(xt, DT)
                total_se += (out["x_tp1_pred"] - xtp1).pow(2).sum().item()
                n += xt.shape[0] * xt.shape[1]
        one_step_mse = total_se / n
        print(f"\n  One-step MSE: {one_step_mse:.4e}")

        # ── Encoder analysis ──
        print(f"\n  ENCODER ANALYSIS:")
        enc_support, enc_coeffs, enc_sparsities = analyze_encoder(model, x_sub)

        # Print coefficient table
        print(f"  {'dim':>4} | " + " ".join(f"{n:>7}" for n in NAMES_D2))
        print(f"  " + "-" * 95)
        for zi in range(LATENT_DIM):
            row = enc_coeffs[zi]
            lin_e = sum(row[j]**2 for j in [1, 2, 3])
            cross_e = sum(row[j]**2 for j in [5, 6])
            ratio = cross_e / max(lin_e, 1e-15)
            mark = " *" if ratio > 0.05 else ""
            row_str = " ".join(f"{v:>7.4f}" for v in row)
            print(f"  z[{zi:2d}] | {row_str}{mark}")

        tp = len(enc_support & TRUE_SUPPORT)
        enc_prec = tp / max(len(enc_support), 1)
        enc_rec = tp / len(TRUE_SUPPORT)
        enc_f1 = 2 * enc_prec * enc_rec / max(enc_prec + enc_rec, 1e-10)
        print(f"\n  Encoder support: {sorted(enc_support)}")
        print(f"  P={enc_prec:.3f} R={enc_rec:.3f} F1={enc_f1:.3f}")
        print(f"  Avg sparsity: {np.mean(enc_sparsities):.1f}")

        # ── Decoder analysis ──
        print(f"\n  DECODER ANALYSIS:")
        dec_results = analyze_decoder(model, x_sub)
        print(f"  Reconstruction MSE: {dec_results['recon_mse']:.6e}")
        for dim_name in ["x", "y", "z"]:
            d = dec_results["dims"][dim_name]
            print(f"  {dim_name}: R²(linear)={d['r2_linear']:.6f}  "
                  f"R²(quadratic)={d['r2_quadratic']:.6f}  "
                  f"#quad_active={d['n_quad_active']}")
            if d["top_quad"]:
                top3 = d["top_quad"][:3]
                print(f"       top quad: " +
                      ", ".join(f"{n}={v:.4f}" for _, n, v in top3))

        # ── Composed model analysis ──
        print(f"\n  COMPOSED MODEL (full one-step map in state space):")
        comp_results = analyze_composed(model, x_sub)
        for dim_name in ["x", "y", "z"]:
            c = comp_results[dim_name]
            terms_str = ", ".join(f"{n}={v:.6f}" for _, n, v in c["terms"][:6])
            print(f"  {dim_name}_hat: R²={c['r2']:.6f}  {terms_str}")

        all_results[sw] = {
            "one_step_mse": one_step_mse,
            "enc_support": sorted(enc_support),
            "enc_prec": enc_prec, "enc_rec": enc_rec, "enc_f1": enc_f1,
            "enc_avg_sparsity": float(np.mean(enc_sparsities)),
            "dec_linear_r2": {dn: dec_results["dims"][dn]["r2_linear"] for dn in ["x","y","z"]},
            "dec_quad_r2": {dn: dec_results["dims"][dn]["r2_quadratic"] for dn in ["x","y","z"]},
            "dec_n_quad": {dn: dec_results["dims"][dn]["n_quad_active"] for dn in ["x","y","z"]},
            "comp_r2": {dn: comp_results[dn]["r2"] for dn in ["x","y","z"]},
            "comp_terms": {dn: [(n, v) for _, n, v in comp_results[dn]["terms"]]
                           for dn in ["x","y","z"]},
        }

    # ════════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Sparsity wt':>12} {'MSE':>10} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"{'Spars':>6} {'Dec lin R²':>12} {'Dec quad R²':>12}")
    print(f"  " + "-" * 80)
    for sw in sparsity_weights:
        r = all_results[sw]
        dec_lin = np.mean(list(r["dec_linear_r2"].values()))
        dec_quad = np.mean(list(r["dec_quad_r2"].values()))
        print(f"  {sw:>12.3f} {r['one_step_mse']:>10.4e} {r['enc_prec']:>6.3f} "
              f"{r['enc_rec']:>6.3f} {r['enc_f1']:>6.3f} {r['enc_avg_sparsity']:>6.1f} "
              f"{dec_lin:>12.6f} {dec_quad:>12.6f}")

    # Save
    import json
    with open(OUTDIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {OUTDIR / 'results.json'}")


if __name__ == "__main__":
    main()
