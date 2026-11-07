"""
This is the main experiment reported in the paper.

Lorenz [3,5,5] PyKAN — Third attempt: prune weak edges before decomposition.

The second test showed many near-zero edges in the encoder/decoder diagrams.
Pruning these before running the level-set decomposition should:
  1. Clarify which psi nodes are actually active per latent dim
  2. Reduce false positives in the polynomial Lasso (fewer nonlinear
     compositions to approximate)
  3. Potentially improve R²(h∘g) by removing noise from dead edges

The model is trained identically to lorenz_5d_second_test.py (grid-fixed
PyKAN [3,5,5] + stable). After training, we prune the encoder edges
at various thresholds and re-run the decomposition.

Run from project root:
    python experiments/lorenz_5d_third_test.py
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
import copy
from numpy.polynomial import polynomial as P
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

OUTDIR = Path("results/lorenz_5d_third_test")
OUTDIR.mkdir(parents=True, exist_ok=True)
DT = 0.01
MID = 5
LATENT = 5
N_TRAJ = 60
N_TRAIN = 48
EPOCHS = 200
VAR = ["x", "y", "z"]
TRUE_SET = {"x", "y", "z", "xy", "xz"}
POLY_NAMES = ["x", "y", "z", "xy", "xz"]

LAMBDA_MAX = 0.9056
TAU_L = 1.0 / LAMBDA_MAX


def train_model(train_traj, val_traj):
    """Train PyKAN [3,5,5] with grid fix (identical to second test)."""
    set_seed(0)
    ae = PyKANAutoEncoder(3, MID, LATENT, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, LATENT)
    enc = model.ae.encoder
    dec = model.ae.decoder

    # Grid fix
    x_sample = train_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_sample), min(5000, len(x_sample)), replace=False)
    x_init = x_sample[idx]
    with torch.no_grad():
        enc.update_grid_from_samples(x_init)
        z_init = enc(x_init)
        dec.update_grid_from_samples(z_init)

    traj_full = torch.cat([train_traj, val_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    mse_fn = nn.MSELoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
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
        sched.step()
        if ep % 50 == 0:
            enc.update_grid_from_samples(x_init)
            with torch.no_grad():
                z_new = enc(x_init)
            dec.update_grid_from_samples(z_new)
        if ep == 1 or ep % 40 == 0 or ep == EPOCHS:
            model.eval()
            vl_dl = DataLoader(PairDataset(traj_full[N_TRAIN:]), batch_size=2048)
            vloss = []
            with torch.no_grad():
                for x_t, x_tp1 in vl_dl:
                    out = model.forward_one_step(x_t, DT)
                    z_true = model.ae.encoder(x_tp1)
                    l = (mse_fn(out["x_tp1_pred"], x_tp1)
                         + mse_fn(out["z_tp1_pred"], z_true)
                         + 0.5 * mse_fn(out["x_t_rec"], x_t))
                    vloss.append(l.item())
            print(f"    Ep {ep:3d} | val {np.mean(vloss):.5f}")

    model.eval()
    return model


def eval_prediction(model, val_traj):
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
            true = val_traj[ic, :max_steps + 1].numpy()
            x = val_traj[ic, 0]
            xs = [x.numpy()]
            for _ in range(max_steps):
                out = model.forward_one_step(x.unsqueeze(0), DT)
                x = out["x_tp1_pred"].squeeze(0)
                xs.append(x.numpy())
            pred = np.array(xs)
            std = true.reshape(-1, 3).std(axis=0)
            std = np.maximum(std, 1e-8)
            nrmse = (np.sqrt((pred - true) ** 2) / std[None, :]).mean(axis=1)
            exceed = np.where(nrmse > 0.5)[0]
            if len(exceed) > 0:
                horizons.append(exceed[0] * DT / TAU_L)
            else:
                horizons.append(max_steps * DT / TAU_L)

    return mse, float(np.mean(horizons)), float(np.std(horizons))


def run_decomposition(model, x_sub, label=""):
    """Run level-set decomposition on all latent dims."""
    print(f"\n  {label}")
    results = []
    for zi in range(LATENT):
        dr = full_decomposition(model, x_sub, zi, target_type='latent',
                                max_degree=3, lasso_alpha=1e-5, h_degree=4)

        all_pairs = sorted(zip(dr['inner']['active_names'],
                               dr['inner']['active_coeffs']),
                           key=lambda x: -abs(x[1]))
        all_pairs = [(n, c) for n, c in all_pairs if n != "1"]
        top6 = all_pairs[:6]

        n_true = sum(1 for n, c in all_pairs if n in TRUE_SET)
        n_fp = len(all_pairs) - n_true
        largest_fp = max((abs(c) for n, c in all_pairs if n not in TRUE_SET), default=0)

        g_expr = '  '.join(f'{c:+.4f}*{n}' for n, c in top6)
        print(f"  z[{zi}] R²(g)={dr['g_recon_r2']:7.3f} R²(h∘g)={dr['recon_r2']:.3f} "
              f"active={len(all_pairs)} (true={n_true}, FP={n_fp})")
        print(f"       g = {g_expr}")

        results.append({
            "dim": zi,
            "r2_g": float(dr['g_recon_r2']),
            "r2_hg": float(dr['recon_r2']),
            "top_terms": [(n, float(c)) for n, c in top6],
            "all_terms": [(n, float(c)) for n, c in all_pairs],
            "n_true": n_true, "n_fp": n_fp,
            "largest_fp": float(largest_fp),
        })
    return results


def count_active_edges(enc):
    """Count non-pruned edges in the encoder."""
    enc.save_act = True
    # Dummy forward to populate activations
    dummy = torch.randn(10, 3)
    with torch.no_grad():
        enc(dummy)

    total = 0
    active = 0
    for l in range(len(enc.width_in) - 1):
        n_in = int(enc.width_in[l])
        n_out = int(enc.width_out[l + 1])
        for i in range(n_in):
            for j in range(n_out):
                total += 1
                # Check if edge is masked out
                mask = enc.act_fun[l].mask
                idx = j * n_in + i
                if idx < len(mask) and mask[idx] > 0.5:
                    active += 1
    return active, total


def main():
    print("=" * 70)
    print("  LORENZ [3,5,5] PyKAN — THIRD TEST: PRUNE THEN DECOMPOSE")
    print("=" * 70)

    # Data
    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:N_TRAIN].contiguous()
    val_traj = traj[N_TRAIN:].contiguous()

    # Train (or load from second test)
    model_path = Path("results/lorenz_5d_second_test/model.pt")
    if model_path.exists():
        print(f"\n  Loading pre-trained model from {model_path}")
        ae = PyKANAutoEncoder(3, MID, LATENT, grid=5, k=3)
        model = DeepKoopmanCTStable(ae, LATENT)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
    else:
        print(f"\n  Training fresh model...")
        model = train_model(train_traj, val_traj)

    enc = model.ae.encoder

    # Prediction baseline (before pruning)
    mse_pre, hor_pre, hor_std = eval_prediction(model, val_traj)
    print(f"\n  Pre-prune: MSE={mse_pre:.4e}  Horizon={hor_pre:.2f}±{hor_std:.2f} tau_L")

    # Analysis data
    x_flat = train_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)

    # ══════════════════════════════════════════════════════════════
    # Pre-prune decomposition (baseline)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  BASELINE (no pruning)")
    print(f"{'='*60}")
    decomp_baseline = run_decomposition(model, x_sub, "Before pruning")

    # ══════════════════════════════════════════════════════════════
    # Prune encoder and re-decompose at multiple thresholds
    # ══════════════════════════════════════════════════════════════
    RETRAIN_EPOCHS = 100
    prune_thresholds = [0.01, 0.03, 0.05]
    all_prune_results = {}

    for thresh in prune_thresholds:
        print(f"\n{'='*60}")
        print(f"  PRUNE THRESHOLD = {thresh}")
        print(f"{'='*60}")

        # Reload fresh model for each threshold
        ae = PyKANAutoEncoder(3, MID, LATENT, grid=5, k=3)
        model = DeepKoopmanCTStable(ae, LATENT)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        enc = model.ae.encoder
        dec = model.ae.decoder

        # Populate activations and compute edge scores for pruning
        enc.save_act = True
        with torch.no_grad():
            enc(x_sub[:2000])
        try:
            enc.attribute(plot=False)
        except Exception:
            pass

        # Prune encoder edges
        enc.prune_edge(threshold=thresh, log_history=False)

        # Also prune decoder
        dec.save_act = True
        with torch.no_grad():
            z_sample = enc(x_sub[:2000])
            dec(z_sample)
        try:
            dec.attribute(plot=False)
        except Exception:
            pass
        try:
            dec.prune_edge(threshold=thresh, log_history=False)
        except Exception:
            print(f"  Decoder prune failed, skipping")

        # Check prediction BEFORE retrain
        mse_pruned, hor_pruned, _ = eval_prediction(model, val_traj)
        print(f"  After prune (before retrain): MSE={mse_pruned:.4e} ({mse_pruned/mse_pre:.1f}x)")

        # ── Retrain with frozen mask ──
        print(f"  Retraining for {RETRAIN_EPOCHS} epochs with pruned edges frozen...")
        traj_full = torch.cat([train_traj, val_traj], dim=0)
        train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                              batch_size=2048, shuffle=True, drop_last=True)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=RETRAIN_EPOCHS)
        mse_fn = nn.MSELoss()

        for ep in range(1, RETRAIN_EPOCHS + 1):
            model.train()
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
            sched.step()
            if ep % 25 == 0:
                model.eval()
                vl_dl = DataLoader(PairDataset(traj_full[N_TRAIN:]), batch_size=2048)
                vloss = []
                with torch.no_grad():
                    for x_t, x_tp1 in vl_dl:
                        out = model.forward_one_step(x_t, DT)
                        z_true = model.ae.encoder(x_tp1)
                        l = (mse_fn(out["x_tp1_pred"], x_tp1)
                             + mse_fn(out["z_tp1_pred"], z_true)
                             + 0.5 * mse_fn(out["x_t_rec"], x_t))
                        vloss.append(l.item())
                print(f"    Ep {ep:3d} | val {np.mean(vloss):.5f}")

        model.eval()

        # Check prediction AFTER retrain
        mse_post, hor_post, hor_std_post = eval_prediction(model, val_traj)
        print(f"  After retrain: MSE={mse_post:.4e} ({mse_post/mse_pre:.2f}x)  "
              f"Horizon={hor_post:.2f}±{hor_std_post:.2f} tau_L")

        # Save pruned+retrained encoder diagram
        try:
            enc.save_act = True
            with torch.no_grad():
                enc(x_sub[:500])
            enc.plot()
            plt.savefig(OUTDIR / f"encoder_pruned_retrained_{thresh}.png",
                        dpi=150, bbox_inches="tight")
            plt.close('all')
            print(f"  Saved encoder_pruned_retrained_{thresh}.png")
        except Exception as e:
            print(f"  Encoder plot failed: {e}")

        # Decomposition after prune + retrain
        decomp = run_decomposition(model, x_sub,
                                    f"After prune({thresh}) + retrain({RETRAIN_EPOCHS}ep)")

        all_prune_results[thresh] = {
            "mse_pruned": mse_pruned,
            "mse_retrained": mse_post,
            "horizon": hor_post,
            "horizon_std": hor_std_post,
            "mse_ratio_pruned": mse_pruned / mse_pre,
            "mse_ratio_retrained": mse_post / mse_pre,
            "decomposition": decomp,
        }

        # Save checkpoint
        torch.save({"model_state_dict": model.state_dict(),
                     "config": {"mid": MID, "latent": LATENT,
                                "prune_threshold": thresh,
                                "retrain_epochs": RETRAIN_EPOCHS}},
                    OUTDIR / f"model_pruned_{thresh}.pt")

        del model, ae, enc, dec

    # ══════════════════════════════════════════════════════════════
    # Summary table
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: Effect of pruning on dictionary recovery")
    print(f"{'='*80}")

    print(f"\n  {'Threshold':>10} {'MSE(prune)':>11} {'MSE(retrain)':>13} {'ratio':>6} "
          f"{'Horizon':>8} {'med R²(h∘g)':>12} {'#FP':>5} {'#true':>6}")
    print("  " + "-" * 80)

    # Baseline
    r2s_base = [d['r2_hg'] for d in decomp_baseline]
    fps_base = [d['n_fp'] for d in decomp_baseline]
    tps_base = [d['n_true'] for d in decomp_baseline]
    print(f"  {'none':>10} {'--':>11} {mse_pre:>13.4e} {'1.00':>6} {hor_pre:>8.2f} "
          f"{np.median(r2s_base):>12.3f} {np.mean(fps_base):>5.1f} {np.mean(tps_base):>6.1f}")

    for thresh in prune_thresholds:
        pr = all_prune_results[thresh]
        r2s = [d['r2_hg'] for d in pr['decomposition']]
        fps = [d['n_fp'] for d in pr['decomposition']]
        tps = [d['n_true'] for d in pr['decomposition']]
        print(f"  {thresh:>10.2f} {pr['mse_pruned']:>11.4e} {pr['mse_retrained']:>13.4e} "
              f"{pr['mse_ratio_retrained']:>6.2f} {pr['horizon']:>8.2f} "
              f"{np.median(r2s):>12.3f} {np.mean(fps):>5.1f} {np.mean(tps):>6.1f}")

    # ══════════════════════════════════════════════════════════════
    # Plot: R²(h∘g) and #FP vs prune threshold
    # ══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    thresholds = [0.0] + prune_thresholds
    mses = [mse_pre] + [all_prune_results[t]['mse_retrained'] for t in prune_thresholds]
    med_r2s = [np.median(r2s_base)] + [
        np.median([d['r2_hg'] for d in all_prune_results[t]['decomposition']])
        for t in prune_thresholds]
    mean_fps = [np.mean(fps_base)] + [
        np.mean([d['n_fp'] for d in all_prune_results[t]['decomposition']])
        for t in prune_thresholds]
    horizons = [hor_pre] + [all_prune_results[t]['horizon'] for t in prune_thresholds]

    ax = axes[0]
    ax.plot(thresholds, mses, "o-", color="steelblue", lw=2)
    ax.set_xlabel("Prune threshold", fontsize=11)
    ax.set_ylabel("One-step MSE", fontsize=11)
    ax.set_title("Prediction quality vs pruning", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(thresholds, med_r2s, "o-", color="coral", lw=2)
    ax.set_xlabel("Prune threshold", fontsize=11)
    ax.set_ylabel("Median R²(h∘g)", fontsize=11)
    ax.set_title("Decomposition quality vs pruning", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(thresholds, mean_fps, "o-", color="green", lw=2)
    ax.set_xlabel("Prune threshold", fontsize=11)
    ax.set_ylabel("Mean #FP per dim", fontsize=11)
    ax.set_title("False positives vs pruning", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Edge Pruning on Dictionary Recovery\n"
                 "PyKAN [3,5,5] Stable Koopman (Lorenz)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "prune_sweep.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTDIR / "prune_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved prune_sweep.png/pdf")

    # ══════════════════════════════════════════════════════════════
    # Per-dim decomposition at best threshold
    # ══════════════════════════════════════════════════════════════
    # Find threshold that maximizes median R²(h∘g) while keeping retrained MSE < 2x baseline
    best_thresh = 0.0
    best_r2 = np.median(r2s_base)
    for thresh in prune_thresholds:
        pr = all_prune_results[thresh]
        if pr['mse_ratio_retrained'] > 2.0:
            continue
        r2 = np.median([d['r2_hg'] for d in pr['decomposition']])
        if r2 > best_r2:
            best_r2 = r2
            best_thresh = thresh

    print(f"\n  Best threshold (MSE < 2x): {best_thresh} (R²={best_r2:.3f})")

    # Save
    save_data = {
        "model_source": str(model_path),
        "pre_prune_mse": mse_pre,
        "pre_prune_horizon": hor_pre,
        "baseline_decomposition": decomp_baseline,
        "prune_results": {
            str(t): {
                "mse_pruned": all_prune_results[t]["mse_pruned"],
                "mse_retrained": all_prune_results[t]["mse_retrained"],
                "horizon": all_prune_results[t]["horizon"],
                "mse_ratio_pruned": all_prune_results[t]["mse_ratio_pruned"],
                "mse_ratio_retrained": all_prune_results[t]["mse_ratio_retrained"],
                "decomposition": all_prune_results[t]["decomposition"],
            }
            for t in prune_thresholds
        },
        "best_threshold": best_thresh,
    }
    with open(OUTDIR / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved results.json")

    print(f"\n  All outputs in {OUTDIR}/")


if __name__ == "__main__":
    main()
