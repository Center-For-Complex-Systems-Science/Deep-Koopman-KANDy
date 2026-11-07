"""

Mostly, a failed experiment.

Ablation row H: PyKAN [3,5,5] + stable + prune@0.03 + retrain.

Uses the PyKAN backend (with grid updates) — the same model that achieved
Jaccard=1.0 in lorenz_5d_third_test.py. Runs across 5 seeds to match
the ablation_lorenz.py experiment.

Pipeline per seed:
  1. Train PyKAN [3,5,5] + stable for 200 epochs (with grid updates)
  2. Prune encoder+decoder edges via PyKAN's attribute + prune_edge
  3. Retrain 100 epochs at lr=5e-4
  4. Extract dictionary at Lasso alpha=5e-3

Run from project root:
    python experiments/ablation_lorenz_pruned.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json, time, gc
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import simulate_lorenz, PairDataset
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition

OUTDIR = Path("results/ablation_lorenz")
OUTDIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUTDIR / "model_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
N_TRAJ = 60
N_TRAIN = 48
LATENT = 5
MID = 5
# Seed 42 matches the working lorenz_5d_third_test.py model
# (data_seed=42, model_seed=0 in that script, but here we use the
#  model seed directly since data is generated once with set_seed(42))
SEEDS = [42, 0, 1, 2, 3, 4]

LAMBDA_MAX = 0.906
TAU_L = 1.0 / LAMBDA_MAX
STEPS_PER_TAU = int(TAU_L / DT)
N_ROLLOUT_ICS = 10

TRUE_DICT = ["x", "y", "z", "xy", "xz"]
TRUE_SET = set(TRUE_DICT)
FALSE_TERMS = ["x^2", "y^2", "z^2", "yz"]

TRAIN_EPOCHS = 200
RETRAIN_EPOCHS = 100
PRUNE_THRESHOLD = 0.03
LASSO_ALPHA = 5e-3


def compute_nrmse(pred, true):
    std = true.reshape(-1, 3).std(axis=0)
    std = np.maximum(std, 1e-8)
    return (np.sqrt((pred - true)**2) / std[None, :]).mean(axis=1)


def eval_prediction(model, vl_traj):
    ds = PairDataset(vl_traj)
    dl = DataLoader(ds, batch_size=4096)
    total_se, n = 0.0, 0
    with torch.no_grad():
        for x_t, x_tp1 in dl:
            out = model.forward_one_step(x_t, DT)
            total_se += (out["x_tp1_pred"] - x_tp1).pow(2).sum().item()
            n += x_t.shape[0] * x_t.shape[1]
    one_step_mse = total_se / n

    steps_2tau = 2 * STEPS_PER_TAU
    max_steps = min(steps_2tau, vl_traj.shape[1] - 1)
    ics = np.random.choice(vl_traj.shape[0], min(N_ROLLOUT_ICS, vl_traj.shape[0]), replace=False)
    rollout_errors = []
    with torch.no_grad():
        for ic in ics:
            true = vl_traj[ic, :max_steps+1].numpy()
            x = vl_traj[ic, 0]
            xs = [x.numpy()]
            for _ in range(max_steps):
                out = model.forward_one_step(x.unsqueeze(0), DT)
                x = out["x_tp1_pred"].squeeze(0)
                xs.append(x.numpy())
            pred = np.array(xs)
            nrmse = compute_nrmse(pred, true)
            rollout_errors.append(float(nrmse[min(steps_2tau, len(nrmse)-1)]))
    rollout_2tau = float(np.mean(rollout_errors))

    return one_step_mse, rollout_2tau


def eval_symbolic(model, tr_traj, lasso_alpha=LASSO_ALPHA):
    x_flat = tr_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    model_cpu = model.cpu().eval()

    detected = set()
    r2_hgs = []
    table = []

    for zi in range(LATENT):
        try:
            dr = full_decomposition(model_cpu, x_sub, zi, target_type='latent',
                                     max_degree=3, lasso_alpha=lasso_alpha, h_degree=4)
            r2_hg = dr['recon_r2']
            r2_hgs.append(r2_hg)

            all_pairs = sorted(zip(dr['inner']['active_names'],
                                   dr['inner']['active_coeffs']),
                               key=lambda x: -abs(x[1]))
            all_pairs = [(n, float(c)) for n, c in all_pairs if n != "1"]

            top5 = all_pairs[:5]
            true_terms = [(n, c) for n, c in all_pairs if n in TRUE_SET]
            fp_terms = [(n, c) for n, c in all_pairs if n not in TRUE_SET]

            h_coeffs = dr['outer'].get('h_coeffs', None)
            h_str = ""
            if h_coeffs is not None:
                for deg, c in enumerate(h_coeffs):
                    if deg == 0: continue
                    if abs(c) > 1e-4:
                        h_str += f"{c:+.2f}u^{deg} " if deg > 1 else f"{c:+.2f}u "
            h_str = h_str.strip() or "const"

            table.append({
                "dim": zi, "r2_g": float(dr['g_recon_r2']),
                "r2_hg": r2_hg, "h": h_str,
                "top5": top5, "true_terms": true_terms,
                "fp_terms": fp_terms, "n_true": len(true_terms),
                "n_fp": len(fp_terms), "all_terms": all_pairs,
            })

            if r2_hg > 0.8:
                detected |= {n for n, c in top5}
        except Exception as e:
            r2_hgs.append(0.0)
            table.append({"dim": zi, "r2_hg": 0.0, "top5": [],
                          "true_terms": [], "fp_terms": [],
                          "n_true": 0, "n_fp": 0, "all_terms": [],
                          "r2_g": 0.0, "h": "ERR"})

    tp_results = {t: t in detected for t in TRUE_DICT}
    fp_results = {t: t in detected for t in FALSE_TERMS}
    false_positives = detected - TRUE_SET
    tp = len(detected & TRUE_SET)
    prec = tp / max(len(detected), 1)
    rec = tp / len(TRUE_SET)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    jaccard = tp / max(len(TRUE_SET) + len(false_positives), 1)

    return {
        "median_r2_hg": float(np.median(r2_hgs)),
        "det_rate": rec,
        "tp_results": tp_results,
        "fp_results": fp_results,
        "false_positives": sorted(false_positives),
        "n_fp": len(false_positives),
        "support_f1": f1,
        "support_precision": prec,
        "support_recall": rec,
        "jaccard": jaccard,
        "table": table,
    }


def train_prune_retrain(train_traj, val_traj, seed):
    """PyKAN [3,5,5] with SiLU base, grid updates → prune → retrain.

    Returns model plus intermediate results:
      - sym_pruned_only: decomposition results AFTER pruning but BEFORE retraining
      - sym_retrained: decomposition results AFTER pruning + retraining
    """
    set_seed(seed)
    ae = PyKANAutoEncoder(3, MID, LATENT, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, LATENT)
    enc = model.ae.encoder
    dec = model.ae.decoder

    # Grid initialization from data
    x_sample = train_traj.reshape(-1, 3)
    grid_idx = np.random.choice(len(x_sample), min(5000, len(x_sample)), replace=False)
    x_init = x_sample[grid_idx]
    with torch.no_grad():
        enc.update_grid_from_samples(x_init)
        z_init = enc(x_init)
        dec.update_grid_from_samples(z_init)

    traj_full = torch.cat([train_traj, val_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_EPOCHS)
    mse_fn = nn.MSELoss()

    # Phase 1: Train
    for ep in range(1, TRAIN_EPOCHS + 1):
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
        # Grid update every 50 epochs
        if ep % 50 == 0:
            enc.update_grid_from_samples(x_init)
            with torch.no_grad():
                z_new = enc(x_init)
            dec.update_grid_from_samples(z_new)
            print(f"        Train Ep {ep}")

    model.eval()

    # ── Plot: AFTER TRAINING (before pruning) ──
    x_plot = x_sample[np.random.choice(len(x_sample), 500, replace=False)]
    def save_kan_plots(enc, dec, x_plot, stage, seed):
        try:
            enc.save_act = True
            with torch.no_grad():
                enc(x_plot)
            enc.plot()
            plt.savefig(PLOT_DIR / f"encoder_seed{seed}_{stage}.png",
                        dpi=120, bbox_inches="tight")
            plt.close("all")
            print(f"        Saved encoder_seed{seed}_{stage}.png")
        except Exception as e:
            print(f"        Encoder plot failed ({stage}): {e}")
        try:
            dec.save_act = True
            with torch.no_grad():
                z_plot = enc(x_plot)
                dec(z_plot)
            dec.plot()
            plt.savefig(PLOT_DIR / f"decoder_seed{seed}_{stage}.png",
                        dpi=120, bbox_inches="tight")
            plt.close("all")
            print(f"        Saved decoder_seed{seed}_{stage}.png")
        except Exception as e:
            print(f"        Decoder plot failed ({stage}): {e}")

    save_kan_plots(enc, dec, x_plot, "1_trained", seed)

    # Phase 2: Prune via PyKAN's attribute + prune_edge
    # Populate activations
    x_sub = x_sample[np.random.choice(len(x_sample), 2000, replace=False)]
    enc.save_act = True
    with torch.no_grad():
        enc(x_sub)
    try:
        enc.attribute(plot=False)
    except Exception as e:
        print(f"        enc.attribute failed: {e}")

    try:
        enc.prune_edge(threshold=PRUNE_THRESHOLD, log_history=False)
        n_pruned = 0
        n_total = 0
        for l in range(len(enc.act_fun)):
            mask = enc.act_fun[l].mask.detach()
            n_total += mask.numel()
            n_pruned += (mask < 0.5).sum().item()
        print(f"        Encoder: pruned {n_pruned}/{n_total} edges")
    except Exception as e:
        print(f"        enc.prune_edge failed: {e}")

    # Prune decoder too
    dec.save_act = True
    with torch.no_grad():
        z_sub = enc(x_sub)
        dec(z_sub)
    try:
        dec.attribute(plot=False)
        dec.prune_edge(threshold=PRUNE_THRESHOLD, log_history=False)
        n_pruned_d = 0
        n_total_d = 0
        for l in range(len(dec.act_fun)):
            mask = dec.act_fun[l].mask.detach()
            n_total_d += mask.numel()
            n_pruned_d += (mask < 0.5).sum().item()
        print(f"        Decoder: pruned {n_pruned_d}/{n_total_d} edges")
    except Exception as e:
        print(f"        dec.prune_edge failed: {e}")

    # ── Plot: AFTER PRUNING (before retraining) ──
    save_kan_plots(enc, dec, x_plot, "2_pruned", seed)

    # ── Decomposition: PRUNED ONLY (no retrain) ──
    print(f"        Decomposition on pruned model (no retrain)...")
    mse_pruned, rollout_pruned = eval_prediction(model, traj_full[N_TRAIN:].cpu())
    sym_pruned = eval_symbolic(model, traj_full[:N_TRAIN].cpu(), lasso_alpha=LASSO_ALPHA)
    print(f"        Pruned-only: MSE={mse_pruned:.4e}  Jac={sym_pruned['jaccard']:.3f}  "
          f"Rec={sym_pruned['support_recall']:.0%}")

    # Phase 3: Retrain
    opt2 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=RETRAIN_EPOCHS)

    for ep in range(1, RETRAIN_EPOCHS + 1):
        model.train()
        for x_t, x_tp1 in train_dl:
            out = model.forward_one_step(x_t, DT)
            z_true = model.ae.encoder(x_tp1)
            loss = (mse_fn(out["x_tp1_pred"], x_tp1)
                    + mse_fn(out["z_tp1_pred"], z_true)
                    + 0.5 * mse_fn(out["x_t_rec"], x_t))
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
        sched2.step()
        if ep % 25 == 0:
            print(f"        Retrain Ep {ep}")

    model.eval()

    # ── Plot: AFTER RETRAINING ──
    save_kan_plots(enc, dec, x_plot, "3_retrained", seed)

    return (model, traj_full[:N_TRAIN].cpu(), traj_full[N_TRAIN:].cpu(),
            {"mse_pruned": mse_pruned, "sym_pruned": sym_pruned})


def main():
    t_start = time.time()
    print("=" * 70)
    print("  ABLATION ROW H: PyKAN [3,5,5] + stable + prune@0.03 + retrain")
    print(f"  Lasso alpha={LASSO_ALPHA} for dictionary extraction")
    print("=" * 70)

    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:N_TRAIN].contiguous()
    val_traj = traj[N_TRAIN:].contiguous()
    print(f"  Data: train={train_traj.shape}, val={val_traj.shape}")

    seed_results = []

    for seed in SEEDS:
        print(f"\n    seed={seed}")
        t0 = time.time()

        model, tr, vl, pruned_info = train_prune_retrain(train_traj, val_traj, seed)
        n_params = sum(p.numel() for p in model.ae.parameters())

        # Retrained model metrics
        mse, rollout_2tau = eval_prediction(model, vl)
        sym = eval_symbolic(model, tr, lasso_alpha=LASSO_ALPHA)

        elapsed = time.time() - t0

        # Pruned-only metrics (from before retraining)
        sp = pruned_info["sym_pruned"]

        result = {
            "mse": mse, "rollout_2tau": rollout_2tau,
            "n_params": n_params, "wall_time": elapsed,
            **sym,
            # Pruned-only (no retrain) metrics
            "pruned_only_mse": pruned_info["mse_pruned"],
            "pruned_only_jaccard": sp["jaccard"],
            "pruned_only_recall": sp["support_recall"],
            "pruned_only_precision": sp["support_precision"],
            "pruned_only_n_fp": sp["n_fp"],
        }
        seed_results.append(result)

        print(f"    RETRAINED: MSE={mse:.4e}  Roll@2τ={rollout_2tau:.3f}  "
              f"R²(h∘g)={sym['median_r2_hg']:.3f}  "
              f"Jaccard={sym['jaccard']:.3f}  "
              f"Rec={sym['support_recall']:.0%}  Prec={sym['support_precision']:.2f}  "
              f"FP={sym['n_fp']}  ({elapsed:.0f}s)")
        print(f"    PRUNED-ONLY: MSE={pruned_info['mse_pruned']:.4e}  "
              f"Jaccard={sp['jaccard']:.3f}  "
              f"Rec={sp['support_recall']:.0%}  FP={sp['n_fp']}")

        # Print dictionary
        print(f"    Dictionary (alpha={LASSO_ALPHA}):")
        for d in sym['table']:
            true_str = " ".join(f"{c:+.4f}*{n}" for n, c in d['true_terms'])
            fp_str = " ".join(f"{c:+.4f}*{n}" for n, c in d['fp_terms'][:3])
            print(f"      z[{d['dim']}] R²={d['r2_hg']:.3f} TRUE: {true_str}")
            if d['fp_terms']:
                print(f"             FP: {fp_str}")

        del model
        gc.collect()

    # ── Summary ──
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: Variant H (PyKAN + stable + prune@0.03 + retrain)")
    print(f"{'='*80}")

    mses = [r['mse'] for r in seed_results]
    rolls = [r['rollout_2tau'] for r in seed_results]
    r2s = [r['median_r2_hg'] for r in seed_results]
    jacs = [r['jaccard'] for r in seed_results]
    precs = [r['support_precision'] for r in seed_results]
    recs = [r['support_recall'] for r in seed_results]
    fps = [r['n_fp'] for r in seed_results]

    # Pruned-only metrics
    po_jacs = [r['pruned_only_jaccard'] for r in seed_results]
    po_recs = [r['pruned_only_recall'] for r in seed_results]
    po_precs = [r['pruned_only_precision'] for r in seed_results]
    po_mses = [r['pruned_only_mse'] for r in seed_results]

    print(f"\n  {'':>20} {'RETRAINED':>15} {'PRUNED ONLY':>15}")
    print(f"  {'MSE':>20} {np.mean(mses):.4e}±{np.std(mses):.1e} {np.mean(po_mses):.4e}±{np.std(po_mses):.1e}")
    print(f"  {'Roll@2τ':>20} {np.mean(rolls):.3f}±{np.std(rolls):.3f}")
    print(f"  {'R²(h∘g)':>20} {np.mean(r2s):.3f}±{np.std(r2s):.3f}")
    print(f"  {'Jaccard':>20} {np.mean(jacs):.3f}±{np.std(jacs):.3f} {np.mean(po_jacs):.3f}±{np.std(po_jacs):.3f}")
    print(f"  {'Precision':>20} {np.mean(precs):.2f}±{np.std(precs):.2f} {np.mean(po_precs):.2f}±{np.std(po_precs):.2f}")
    print(f"  {'Recall':>20} {np.mean(recs):.2f}±{np.std(recs):.2f} {np.mean(po_recs):.2f}±{np.std(po_recs):.2f}")
    print(f"  {'#FP':>20} {np.mean(fps):.1f}±{np.std(fps):.1f}")

    print(f"\n  Per-term detection (across {len(SEEDS)} seeds):")
    for t in TRUE_DICT:
        rate = np.mean([r['tp_results'][t] for r in seed_results])
        print(f"    {t:>4}: {rate:.0%}")
    print(f"  False positives:")
    for t in FALSE_TERMS:
        rate = np.mean([r['fp_results'][t] for r in seed_results])
        if rate > 0:
            print(f"    {t:>4}: {rate:.0%}")

    print(f"\n  ABLATION TABLE ROW (LaTeX):")
    print(f"  H & PyKAN + stable + prune & {seed_results[0]['n_params']} & "
          f"${np.mean(mses):.2e} \\pm {np.std(mses):.0e}$ & "
          f"${np.mean(rolls):.2f} \\pm {np.std(rolls):.2f}$ & "
          f"${np.mean(jacs):.2f} \\pm {np.std(jacs):.2f}$ & "
          f"${np.mean(precs):.2f}$ & "
          f"${np.mean(recs):.2f}$ \\\\")

    # Save
    save_data = {
        "variant": "H: PyKAN [3,5,5] + stable + prune@0.03 + retrain",
        "backend": "PyKAN",
        "prune_threshold": PRUNE_THRESHOLD,
        "lasso_alpha": LASSO_ALPHA,
        "train_epochs": TRAIN_EPOCHS,
        "retrain_epochs": RETRAIN_EPOCHS,
        "n_params": seed_results[0]['n_params'],
        "mse_mean": float(np.mean(mses)),
        "mse_std": float(np.std(mses)),
        "rollout_2tau_mean": float(np.mean(rolls)),
        "rollout_2tau_std": float(np.std(rolls)),
        "jaccard_mean": float(np.mean(jacs)),
        "jaccard_std": float(np.std(jacs)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "per_term": {t: float(np.mean([r['tp_results'][t] for r in seed_results]))
                     for t in TRUE_DICT},
        "per_fp": {t: float(np.mean([r['fp_results'][t] for r in seed_results]))
                   for t in FALSE_TERMS},
        "seeds": [{
            "mse": r['mse'],
            "rollout_2tau": r['rollout_2tau'],
            "median_r2_hg": r['median_r2_hg'],
            "jaccard": r['jaccard'],
            "precision": r['support_precision'],
            "recall": r['support_recall'],
            "n_fp": r['n_fp'],
            "false_positives": r['false_positives'],
            "wall_time": r['wall_time'],
            "dictionary": r['table'],
        } for r in seed_results],
    }

    with open(OUTDIR / "results_pruned_pykan.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR / 'results_pruned_pykan.json'}")

    elapsed = time.time() - t_start
    print(f"  Total time: {elapsed/60:.0f} minutes")


if __name__ == "__main__":
    main()
