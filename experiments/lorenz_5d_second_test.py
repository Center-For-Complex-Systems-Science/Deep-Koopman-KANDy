"""
Lorenz [3,5,5] PyKAN test — FIXED grid initialization.

The first lorenz_5d_test.py failed (val loss 1.31, MSE=0.71) because
PyKAN's spline grids were initialized at their default range while the
Lorenz data spans [-30, 50]. This version fixes the issue by:

  1. Calling update_grid_from_samples() with real training data before
     training begins, so the spline knots cover the actual data range.
  2. Periodically re-updating the grid during training (every 50 epochs)
     to track the evolving activation distributions.

Run from project root:
    python experiments/lorenz_5d_second_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from scipy.linalg import logm
from numpy.polynomial import polynomial as P
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data import simulate_lorenz, PairDataset
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition

OUTDIR = Path("results/lorenz_5d_second_test")
OUTDIR.mkdir(parents=True, exist_ok=True)
DT = 0.01
MID = 5
LATENT = 5
N_TRAJ = 60
N_TRAIN = 48
EPOCHS = 200
VAR = ["x", "y", "z"]

LAMBDA_MAX = 0.906
TAU_L = 1.0 / LAMBDA_MAX


def ols_r2(X, y):
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res / max(ss_tot, 1e-12), coeffs


def main():
    print("=" * 70)
    print(f"  LORENZ [3, {MID}, {LATENT}] PyKAN — FIXED GRID")
    print("=" * 70)

    # ── Data ──
    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:N_TRAIN].contiguous()
    val_traj = traj[N_TRAIN:].contiguous()
    traj_full = torch.cat([train_traj, val_traj], dim=0)
    print(f"  Data: train={train_traj.shape}, val={val_traj.shape}")

    # ── Build model ──
    set_seed(0)
    ae = PyKANAutoEncoder(3, MID, LATENT, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, LATENT)
    enc = model.ae.encoder
    dec = model.ae.decoder
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── FIX: Initialize grids from actual data ──
    print("\n  Initializing grids from training data...")
    x_sample = train_traj.reshape(-1, 3)
    # Subsample for grid init
    idx = np.random.choice(len(x_sample), min(5000, len(x_sample)), replace=False)
    x_init = x_sample[idx]

    with torch.no_grad():
        # Update encoder grids
        enc.update_grid_from_samples(x_init)
        # Get encoder output for decoder grid init
        z_init = enc(x_init)
        dec.update_grid_from_samples(z_init)
    print("  Grids initialized.")

    # ── Train ──
    print(f"\n  Training for {EPOCHS} epochs...")
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    val_dl = DataLoader(PairDataset(traj_full[N_TRAIN:]), batch_size=2048)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    mse_fn = nn.MSELoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x_t, x_tp1 in train_dl:
            out = model.forward_one_step(x_t, DT)
            z_true = model.ae.encoder(x_tp1)
            loss = (mse_fn(out["x_tp1_pred"], x_tp1)
                    + mse_fn(out["z_tp1_pred"], z_true)
                    + 0.5 * mse_fn(out["x_t_rec"], x_t))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        # Re-update grids periodically
        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                enc.update_grid_from_samples(x_init)
                z_init_new = enc(x_init)
                dec.update_grid_from_samples(z_init_new)

        if ep == 1 or ep % 20 == 0 or ep == EPOCHS:
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
    torch.save({"model_state_dict": model.state_dict(),
                "config": {"mid": MID, "latent": LATENT, "backend": "PyKAN",
                            "grid_fix": True}},
               OUTDIR / "model.pt")
    print(f"  Saved model to {OUTDIR / 'model.pt'}")

    # ── Prediction metrics ──
    ds = PairDataset(val_traj)
    dl = DataLoader(ds, batch_size=4096)
    total_se, n = 0.0, 0
    with torch.no_grad():
        for x_t, x_tp1 in dl:
            out = model.forward_one_step(x_t, DT)
            total_se += (out["x_tp1_pred"] - x_tp1).pow(2).sum().item()
            n += x_t.shape[0] * x_t.shape[1]
    mse = total_se / n

    steps_per_tau = int(TAU_L / DT)
    max_steps = min(10 * steps_per_tau, val_traj.shape[1] - 1)
    horizons = []
    with torch.no_grad():
        for ic in range(val_traj.shape[0]):
            true = val_traj[ic, :max_steps+1].numpy()
            x = val_traj[ic, 0]
            xs = [x.numpy()]
            for _ in range(max_steps):
                out = model.forward_one_step(x.unsqueeze(0), DT)
                x = out["x_tp1_pred"].squeeze(0)
                xs.append(x.numpy())
            pred = np.array(xs)
            std = true.reshape(-1, 3).std(axis=0)
            std = np.maximum(std, 1e-8)
            nrmse = (np.sqrt((pred - true)**2) / std[None, :]).mean(axis=1)
            exceed = np.where(nrmse > 0.5)[0]
            if len(exceed) > 0:
                horizons.append(exceed[0] * DT / TAU_L)
            else:
                horizons.append(max_steps * DT / TAU_L)

    print(f"\n  One-step MSE:    {mse:.4e}")
    print(f"  Rollout horizon: {np.mean(horizons):.2f} ± {np.std(horizons):.2f} tau_L")

    # ── Level-set decomposition ──
    print(f"\n  Level-set decomposition (degree=3, alpha=1e-5):")
    x_flat = train_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)

    print(f"  {'Dim':>4}  {'R2(g)':>8}  {'R2(hog)':>8}  g (top terms)")
    print("  " + "-" * 80)

    TRUE_SET = {"x", "y", "z", "xy", "xz"}

    decomp_results = []
    for zi in range(LATENT):
        dr = full_decomposition(model, x_sub, zi, target_type='latent',
                                max_degree=3, lasso_alpha=1e-5, h_degree=4)

        # All active terms sorted by magnitude (exclude constant)
        all_pairs = sorted(zip(dr['inner']['active_names'],
                               dr['inner']['active_coeffs']),
                           key=lambda x: -abs(x[1]))
        all_pairs = [(n, c) for n, c in all_pairs if n != "1"]

        top6 = all_pairs[:6]
        g_expr = '  '.join(f'{c:+.4f}*{n}' for n, c in top6)

        h_coeffs = dr['outer'].get('h_coeffs', None)
        h_str = ""
        if h_coeffs is not None:
            for deg, c in enumerate(h_coeffs):
                if deg == 0: continue
                if abs(c) > 1e-4:
                    h_str += f"{c:+.2f}u^{deg} " if deg > 1 else f"{c:+.2f}u "
        h_str = h_str.strip() or "const"

        n_true = sum(1 for n, c in all_pairs if n in TRUE_SET)
        n_fp = len(all_pairs) - n_true
        largest_fp = max((abs(c) for n, c in all_pairs if n not in TRUE_SET), default=0)

        print(f"  z[{zi}]  R²(g)={dr['g_recon_r2']:7.3f}  R²(h∘g)={dr['recon_r2']:.3f}  "
              f"active={len(all_pairs)} (true={n_true}, FP={n_fp}, maxFP={largest_fp:.4f})")
        print(f"         g = {g_expr}")

        decomp_results.append({
            "dim": zi, "r2_g": float(dr['g_recon_r2']),
            "r2_hg": float(dr['recon_r2']), "h": h_str,
            "top_terms": [(n, float(c)) for n, c in top6],
            "all_terms": [(n, float(c)) for n, c in all_pairs],
            "n_true": n_true, "n_fp": n_fp,
            "largest_fp": float(largest_fp),
        })

    # ── Spline analysis ──
    print(f"\n  Layer-1 spline linearity:")
    x_mean = torch.tensor(x_flat.numpy().mean(axis=0), dtype=torch.float32)
    with torch.no_grad():
        enc(x_mean.unsqueeze(0))
        baseline = enc.acts[1].numpy().flatten()

    n_nl = 0
    for j in range(MID):
        for i in range(3):
            xi = np.linspace(x_flat[:, i].numpy().min(),
                             x_flat[:, i].numpy().max(), 300)
            probe = x_mean.unsqueeze(0).repeat(300, 1)
            probe[:, i] = torch.tensor(xi, dtype=torch.float32)
            with torch.no_grad():
                enc(probe)
                eo = enc.acts[1].numpy()[:, j] - baseline[j]
            c = P.polyfit(xi, eo, 1)
            pred = P.polyval(xi, c)
            ss = np.sum((eo - pred)**2)
            st = np.sum((eo - eo.mean())**2)
            r2 = 1 - ss / max(st, 1e-12)
            nl = "NL" if r2 < 0.95 else ""
            if nl: n_nl += 1
            print(f"    psi{j}({VAR[i]}): linR²={r2:.3f} {nl}")

    print(f"\n  Nonlinear edges: {n_nl}/15")

    # ── K_psi eigendecomposition ──
    print(f"\n  K_psi eigendecomposition:")
    pair_idx = np.random.choice(len(x_flat) - 1, 50000, replace=False)
    x_t = torch.tensor(x_flat[pair_idx].numpy(), dtype=torch.float32)
    x_tp1 = torch.tensor(x_flat[pair_idx + 1].numpy(), dtype=torch.float32)

    with torch.no_grad():
        enc(x_t); psi_t = enc.acts[1].numpy()
        enc(x_tp1); psi_tp1 = enc.acts[1].numpy()

    psi_t_wc = np.column_stack([np.ones(len(psi_t)), psi_t])
    K_full, _, _, _ = np.linalg.lstsq(psi_t_wc, psi_tp1, rcond=None)
    K_psi = K_full[1:, :]

    eigvals = np.linalg.eigvals(K_psi)
    mu = np.log(eigvals.astype(complex)) / DT
    order = np.argsort(-np.abs(mu.imag))

    print(f"  {'Mode':>5} {'|λ|':>8} {'Re(μ)':>8} {'freq Hz':>8}")
    for rank, oidx in enumerate(order):
        print(f"  M{rank:>3} {abs(eigvals[oidx]):>8.4f} {mu[oidx].real:>8.2f} "
              f"{mu[oidx].imag/(2*np.pi):>8.3f}")

    # Koopman linearity
    print(f"\n  Koopman linearity in psi basis:")
    psi_r2s = []
    for j in range(MID):
        r2, _ = ols_r2(psi_t_wc, psi_tp1[:, j])
        psi_r2s.append(r2)
        print(f"    psi{j}: R² = {r2:.6f}")
    print(f"  Min: {min(psi_r2s):.6f}")

    # ── Save results ──
    eigenvalues_info = []
    for rank, oidx in enumerate(order):
        eigenvalues_info.append({
            "mode": rank,
            "abs_lambda": float(abs(eigvals[oidx])),
            "re_mu": float(mu[oidx].real),
            "freq_hz": float(mu[oidx].imag / (2 * np.pi)),
        })

    spline_linearity = []
    # Re-collect for saving
    for j in range(MID):
        for i in range(3):
            xi = np.linspace(x_flat[:, i].numpy().min(),
                             x_flat[:, i].numpy().max(), 300)
            probe = x_mean.unsqueeze(0).repeat(300, 1)
            probe[:, i] = torch.tensor(xi, dtype=torch.float32)
            with torch.no_grad():
                enc(probe)
                eo = enc.acts[1].numpy()[:, j] - baseline[j]
            c = P.polyfit(xi, eo, 1)
            pred = P.polyval(xi, c)
            ss = np.sum((eo - pred)**2)
            st = np.sum((eo - eo.mean())**2)
            r2 = 1 - ss / max(st, 1e-12)
            spline_linearity.append({
                "node": j, "input": VAR[i], "lin_r2": float(r2),
                "nonlinear": r2 < 0.95,
            })

    save_data = {
        "backend": "PyKAN", "grid_fix": True,
        "architecture": [3, MID, LATENT],
        "n_params": sum(p.numel() for p in model.parameters()),
        "one_step_mse": mse,
        "rollout_horizon_tau": float(np.mean(horizons)),
        "rollout_horizon_std": float(np.std(horizons)),
        "nonlinear_edges": n_nl,
        "koopman_linearity_min": float(min(psi_r2s)),
        "koopman_linearity_per_psi": [float(r) for r in psi_r2s],
        "eigenvalues": eigenvalues_info,
        "spline_linearity": spline_linearity,
        "decomposition": decomp_results,
    }
    with open(OUTDIR / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR / 'results.json'}")

    # ══════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════
    print(f"\n  Generating plots...")

    # Get psi data for plots
    with torch.no_grad():
        enc(x_sub)
        psi_data = enc.acts[1].numpy()
    x_np = x_sub.numpy()

    # ── Plot 1: PyKAN encoder network diagram ──
    print("  1. PyKAN encoder diagram...")
    try:
        enc.save_act = True
        with torch.no_grad():
            enc(x_sub[:500])
        fig_enc = enc.plot()
        plt.savefig(OUTDIR / "encoder_diagram.png", dpi=150, bbox_inches="tight")
        plt.savefig(OUTDIR / "encoder_diagram.pdf", bbox_inches="tight")
        plt.close('all')
        print("     Saved encoder_diagram.png/pdf")
    except Exception as e:
        print(f"     encoder diagram failed: {e}")

    # ── Plot 2: PyKAN decoder network diagram ──
    print("  2. PyKAN decoder diagram...")
    try:
        dec = model.ae.decoder
        dec.save_act = True
        with torch.no_grad():
            z_sample = enc(x_sub[:500])
            dec(z_sample)
        fig_dec = dec.plot()
        plt.savefig(OUTDIR / "decoder_diagram.png", dpi=150, bbox_inches="tight")
        plt.savefig(OUTDIR / "decoder_diagram.pdf", bbox_inches="tight")
        plt.close('all')
        print("     Saved decoder_diagram.png/pdf")
    except Exception as e:
        print(f"     decoder diagram failed: {e}")

    # ── Plot 3: Layer-1 edge functions (5 rows x 3 cols) ──
    print("  3. Layer-1 edge functions...")
    fig3, axes3 = plt.subplots(MID, 3, figsize=(12, 2.5 * MID))
    for j in range(MID):
        for i in range(3):
            ax = axes3[j, i]
            xi = np.linspace(x_np[:, i].min(), x_np[:, i].max(), 300)
            probe = x_mean.unsqueeze(0).repeat(300, 1)
            probe[:, i] = torch.tensor(xi, dtype=torch.float32)
            with torch.no_grad():
                enc(probe)
                eo = enc.acts[1].numpy()[:, j] - baseline[j]
            ax.plot(xi, eo, "b-", lw=2)
            # Linear fit overlay
            c_lin = P.polyfit(xi, eo, 1)
            ax.plot(xi, P.polyval(xi, c_lin), "r--", lw=1, alpha=0.5)
            ax.axhline(0, color="gray", ls=":", alpha=0.3)
            ss = np.sum((eo - P.polyval(xi, c_lin))**2)
            st = np.sum((eo - eo.mean())**2)
            r2 = 1 - ss / max(st, 1e-12)
            color = "red" if r2 < 0.95 else "black"
            ax.set_title(f"$\\psi_{j}$({VAR[i]}) linR²={r2:.2f}",
                        fontsize=9, color=color)
            ax.tick_params(labelsize=6)
    fig3.suptitle("Layer-1 Edge Functions [3,5,5] PyKAN (grid-fixed)",
                  fontsize=13)
    fig3.tight_layout()
    fig3.savefig(OUTDIR / "layer1_edges.png", dpi=150, bbox_inches="tight")
    fig3.savefig(OUTDIR / "layer1_edges.pdf", bbox_inches="tight")
    plt.close(fig3)
    print("     Saved layer1_edges.png/pdf")

    # ── Plot 4: Psi manifolds on attractor (5 rows: edges + attractor + level sets) ──
    print("  4. Psi manifolds...")
    fig4 = plt.figure(figsize=(22, 4 * MID))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(MID, 5, figure=fig4, width_ratios=[1, 1, 1, 1.3, 1.3],
                  wspace=0.3, hspace=0.35)

    for j in range(MID):
        # Cols 0-2: edge functions
        for i in range(3):
            ax = fig4.add_subplot(gs[j, i])
            xi = np.linspace(x_np[:, i].min(), x_np[:, i].max(), 300)
            probe = x_mean.unsqueeze(0).repeat(300, 1)
            probe[:, i] = torch.tensor(xi, dtype=torch.float32)
            with torch.no_grad():
                enc(probe)
                eo = enc.acts[1].numpy()[:, j] - baseline[j]
            ax.plot(xi, eo, "b-", lw=2)
            ax.axhline(0, color="gray", ls=":", alpha=0.3)
            c_lin = P.polyfit(xi, eo, 1)
            ss = np.sum((eo - P.polyval(xi, c_lin))**2)
            st = np.sum((eo - eo.mean())**2)
            r2 = 1 - ss / max(st, 1e-12)
            color = "red" if r2 < 0.95 else "black"
            ax.set_title(f"$\\phi_{{{j},{i}}}$({VAR[i]}) [linR²={r2:.2f}]",
                        fontsize=8, color=color)
            ax.tick_params(labelsize=6)

        # Col 3: attractor colored by psi_j
        ax3d = fig4.add_subplot(gs[j, 3], projection="3d")
        c = psi_data[:, j]
        vmax = np.percentile(np.abs(c), 95)
        ax3d.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                     c=c, cmap="coolwarm", s=0.5, alpha=0.3,
                     vmin=-vmax, vmax=vmax)
        ax3d.set_title(f"$\\psi_{j}$ on attractor", fontsize=10)
        ax3d.view_init(elev=22, azim=-55)
        ax3d.tick_params(labelsize=5)
        ax3d.set_xlabel("x", fontsize=7)
        ax3d.set_ylabel("y", fontsize=7)
        ax3d.set_zlabel("z", fontsize=7)

        # Col 4: level-set bands
        ax3d2 = fig4.add_subplot(gs[j, 4], projection="3d")
        psi_j = psi_data[:, j]
        n_bands = 5
        band_edges = np.linspace(psi_j.min(), psi_j.max(), n_bands + 1)
        colors_band = plt.cm.viridis(np.linspace(0.1, 0.9, n_bands))
        for b in range(n_bands):
            mask = (psi_j >= band_edges[b]) & (psi_j < band_edges[b + 1])
            if mask.sum() > 0:
                ax3d2.scatter(x_np[mask, 0], x_np[mask, 1], x_np[mask, 2],
                             c=[colors_band[b]], s=0.5, alpha=0.4)
        ax3d2.set_title(f"$\\psi_{j}$ level sets", fontsize=10)
        ax3d2.view_init(elev=22, azim=-55)
        ax3d2.tick_params(labelsize=5)
        ax3d2.set_xlabel("x", fontsize=7)
        ax3d2.set_ylabel("y", fontsize=7)
        ax3d2.set_zlabel("z", fontsize=7)

    fig4.suptitle("Layer-1 Splines + Manifolds [3,5,5] PyKAN (grid-fixed)\n"
                  "$\\psi_j = \\phi_{j,0}(x) + \\phi_{j,1}(y) + \\phi_{j,2}(z)$",
                  fontsize=14, y=1.01)
    fig4.savefig(OUTDIR / "psi_manifolds.png", dpi=120, bbox_inches="tight")
    fig4.savefig(OUTDIR / "psi_manifolds.pdf", bbox_inches="tight")
    plt.close(fig4)
    print("     Saved psi_manifolds.png/pdf")

    # ── Plot 5: Layer-2 splines (z_k vs psi_j) ──
    print("  5. Layer-2 splines...")
    with torch.no_grad():
        enc(x_sub)
        if len(enc.acts) > 2:
            z_data = enc.acts[2].numpy()
        else:
            z_data = enc(x_sub).numpy()

    fig5, axes5 = plt.subplots(LATENT, MID, figsize=(4 * MID, 4 * LATENT))
    for k in range(LATENT):
        for j in range(MID):
            ax = axes5[k, j]
            # Scatter
            ax.scatter(psi_data[:, j], z_data[:, k], s=0.3, alpha=0.03, color="gray")
            # Binned median
            si = np.argsort(psi_data[:, j])
            ps = psi_data[si, j]
            zs = z_data[si, k]
            be = np.linspace(ps.min(), ps.max(), 60)
            bc, bm = [], []
            for b in range(len(be)-1):
                m = (ps >= be[b]) & (ps < be[b+1])
                if m.sum() > 5:
                    bc.append((be[b]+be[b+1])/2)
                    bm.append(np.median(zs[m]))
            if len(bc) > 2:
                bc_a, bm_a = np.array(bc), np.array(bm)
                ax.plot(bc_a, bm_a, "b-", lw=2.5)
                c_lin = P.polyfit(bc_a, bm_a, 1)
                pred = P.polyval(bc_a, c_lin)
                ss = np.sum((bm_a - pred)**2)
                st = np.sum((bm_a - bm_a.mean())**2)
                r2 = 1 - ss / max(st, 1e-12)
                color = "red" if r2 < 0.9 else "black"
                ax.set_title(f"$z_{k}$ ← $\\psi_{j}$ R²={r2:.2f}",
                            fontsize=9, color=color)
            ax.tick_params(labelsize=6)
    fig5.suptitle("Layer-2 Edge Functions (blue=median, gray=scatter, red=nonlinear)",
                  fontsize=13)
    fig5.tight_layout()
    fig5.savefig(OUTDIR / "layer2_edges.png", dpi=120, bbox_inches="tight")
    fig5.savefig(OUTDIR / "layer2_edges.pdf", bbox_inches="tight")
    plt.close(fig5)
    print("     Saved layer2_edges.png/pdf")

    # ── Plot 6: Attractor colored by each psi (compact strip) ──
    print("  6. Attractor colored by psi...")
    fig6, axes6 = plt.subplots(1, MID, figsize=(4 * MID, 4),
                                subplot_kw={"projection": "3d"})
    for j in range(MID):
        ax = axes6[j]
        c = psi_data[:, j]
        vmax = np.percentile(np.abs(c), 95)
        ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                  c=c, cmap="coolwarm", s=0.3, alpha=0.3,
                  vmin=-vmax, vmax=vmax)
        ax.set_title(f"$\\psi_{j}$", fontsize=12)
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=5)
    fig6.suptitle("Attractor colored by Layer-1 activations", fontsize=13)
    fig6.tight_layout()
    fig6.savefig(OUTDIR / "attractor_psi.png", dpi=120, bbox_inches="tight")
    plt.close(fig6)
    print("     Saved attractor_psi.png")

    # ── Plot 7: Eigenmode structure ──
    print("  7. Eigenmode structure...")
    eigvals_full, V = np.linalg.eig(K_psi)
    V_inv = np.linalg.inv(V)
    modal_data = psi_data @ V_inv.T

    # Get a trajectory segment for time series
    traj_seg = traj[0, :2000].numpy()
    x_seg = torch.tensor(traj_seg, dtype=torch.float32)
    with torch.no_grad():
        enc(x_seg)
        psi_seg = enc.acts[1].numpy()
    modal_seg = psi_seg @ V_inv.T
    t_seg = np.arange(len(traj_seg)) * DT

    fig7 = plt.figure(figsize=(18, 12))
    gs7 = GridSpec(3, MID, figure=fig7, hspace=0.4, wspace=0.3)

    # Row 0: participation heatmap + eigenvalues
    ax_heat = fig7.add_subplot(gs7[0, :MID-1])
    participation = np.zeros((MID, MID))
    mode_labels = []
    for rank, oidx in enumerate(order):
        v = V[:, oidx]
        participation[:, rank] = np.abs(v) / np.max(np.abs(v))
        freq = mu[oidx].imag / (2 * np.pi)
        mode_labels.append(f"M{rank}\n{freq:+.1f}Hz\nRe={mu[oidx].real:.1f}")
    im = ax_heat.imshow(participation, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax_heat.set_yticks(range(MID))
    ax_heat.set_yticklabels([f"$\\psi_{j}$" for j in range(MID)])
    ax_heat.set_xticks(range(MID))
    ax_heat.set_xticklabels(mode_labels, fontsize=8)
    ax_heat.set_title("Eigenvector Participation", fontsize=11)
    for ii in range(MID):
        for jj in range(MID):
            v = participation[ii, jj]
            if v > 0.3:
                ax_heat.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if v > 0.6 else "black")
    plt.colorbar(im, ax=ax_heat, shrink=0.8)

    ax_spec = fig7.add_subplot(gs7[0, MID-1])
    theta = np.linspace(0, 2*np.pi, 100)
    ax_spec.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.2)
    colors_m = plt.cm.tab10(np.linspace(0, 1, MID))
    for rank, oidx in enumerate(order):
        lam = eigvals_full[oidx]
        ax_spec.plot(lam.real, lam.imag, "o", color=colors_m[rank], ms=10)
        freq = mu[oidx].imag / (2*np.pi)
        ax_spec.annotate(f"M{rank}\n{freq:+.1f}Hz",
                        (lam.real, lam.imag), fontsize=6, ha="left")
    ax_spec.set_xlim(0.9, 1.01)
    ax_spec.set_aspect("equal")
    ax_spec.set_title("Eigenvalues", fontsize=10)
    ax_spec.grid(True, alpha=0.3)

    # Row 1: attractor colored by each mode
    for rank in range(MID):
        ax = fig7.add_subplot(gs7[1, rank], projection="3d")
        oidx = order[rank]
        c = modal_data[:, oidx].real
        vmax = np.percentile(np.abs(c), 95)
        ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                  c=c, cmap="coolwarm", s=0.3, alpha=0.3,
                  vmin=-vmax, vmax=vmax)
        freq = mu[oidx].imag / (2*np.pi)
        ax.set_title(f"Mode {rank}: {freq:+.1f} Hz", fontsize=9)
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=5)

    # Row 2: time series
    for rank in range(MID):
        ax = fig7.add_subplot(gs7[2, rank])
        oidx = order[rank]
        mk = modal_seg[:, oidx]
        freq = mu[oidx].imag / (2*np.pi)
        ax.plot(t_seg, mk.real, "b-", lw=0.8, label="Re")
        if abs(mu[oidx].imag) > 0.1:
            ax.plot(t_seg, mk.imag, "r-", lw=0.8, alpha=0.5, label="Im")
        ax.set_title(f"M{rank} ({freq:+.1f} Hz)", fontsize=9)
        ax.set_xlim(0, 5)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    fig7.suptitle("Eigenmode Structure [3,5,5] PyKAN (grid-fixed)",
                  fontsize=14, y=1.01)
    fig7.savefig(OUTDIR / "eigenmode_structure.png", dpi=200, bbox_inches="tight")
    fig7.savefig(OUTDIR / "eigenmode_structure.pdf", bbox_inches="tight")
    plt.close(fig7)
    print("     Saved eigenmode_structure.png/pdf")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  PyKAN [3,5,5] + stable, grid-fixed")
    print(f"  MSE: {mse:.4e}  (was 0.71 without grid fix)")
    print(f"  Horizon: {np.mean(horizons):.2f} tau_L  (was 0.21)")
    print(f"  Nonlinear edges: {n_nl}/15")
    print(f"  Koopman linearity: min R²={min(psi_r2s):.4f}")
    print(f"\n  Plots saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
