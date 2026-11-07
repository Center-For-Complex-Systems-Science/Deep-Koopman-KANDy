"""
PyKAN pruned model: Lasso alpha sweep + multi-seed validation.

Stage 1: Fix seed=0, prune@0.03, sweep Lasso alpha to find best.
Stage 2: Best alpha from Stage 1, run 5 seeds [0,1,2,3,4].

Same data (seed 42), same ICs as ablation_lorenz.py.
Reports the Sparse Recovery Score = recall × min(1, |target| / terms_per_dim).

Run from project root:
    python experiments/pykan_pruned_sweep.py
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

from data import simulate_lorenz, PairDataset
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition

OUTDIR = Path("results/pykan_pruned_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
N_TRAJ = 60
N_TRAIN = 48
LATENT = 5
MID = 5
BURN_IN = 500

TRAIN_EPOCHS = 200
RETRAIN_EPOCHS = 100
PRUNE_THRESHOLD = 0.03

LAMBDA_MAX = 0.906
TAU_L = 1.0 / LAMBDA_MAX
STEPS_PER_TAU = int(TAU_L / DT)
N_ROLLOUT_ICS = 10

TRUE_DICT = ["x", "y", "z", "xy", "xz"]
TRUE_SET = set(TRUE_DICT)
N_TARGET = len(TRUE_SET)

# Stage 1: alpha sweep at seed=0
ALPHA_SWEEP = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

# Stage 2: best alpha across seeds
SEEDS = [0, 1, 2, 3, 4]


def compute_nrmse(pred, true):
    std = true.reshape(-1, 3).std(axis=0)
    std = np.maximum(std, 1e-8)
    return (np.sqrt((pred - true)**2) / std[None, :]).mean(axis=1)


def train_prune_retrain(train_traj, val_traj, seed):
    """Train PyKAN [3,5,5] + stable, grid updates, prune@0.03, retrain."""
    set_seed(seed)
    ae = PyKANAutoEncoder(3, MID, LATENT, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, LATENT)
    enc = model.ae.encoder
    dec = model.ae.decoder

    x_sample = train_traj.reshape(-1, 3)
    grid_idx = np.random.choice(len(x_sample), min(5000, len(x_sample)), replace=False)
    x_init = x_sample[grid_idx]
    with torch.no_grad():
        enc.update_grid_from_samples(x_init)
        dec.update_grid_from_samples(enc(x_init))

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
        if ep % 50 == 0:
            enc.update_grid_from_samples(x_init)
            with torch.no_grad():
                dec.update_grid_from_samples(enc(x_init))

    model.eval()

    # Phase 2: Prune
    x_sub = x_sample[np.random.choice(len(x_sample), 2000, replace=False)]
    enc.save_act = True
    with torch.no_grad():
        enc(x_sub)
    try:
        enc.attribute(plot=False)
        enc.prune_edge(threshold=PRUNE_THRESHOLD, log_history=False)
    except Exception as e:
        print(f"      Prune failed: {e}")

    try:
        dec.save_act = True
        with torch.no_grad():
            dec(enc(x_sub))
        dec.attribute(plot=False)
        dec.prune_edge(threshold=PRUNE_THRESHOLD, log_history=False)
    except Exception:
        pass

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

    model.eval()
    return model, traj_full[:N_TRAIN].cpu(), traj_full[N_TRAIN:].cpu()


def eval_prediction(model, vl_traj):
    ds = PairDataset(vl_traj)
    dl = DataLoader(ds, batch_size=4096)
    total_se, n = 0.0, 0
    with torch.no_grad():
        for x_t, x_tp1 in dl:
            out = model.forward_one_step(x_t, DT)
            total_se += (out["x_tp1_pred"] - x_tp1).pow(2).sum().item()
            n += x_t.shape[0] * x_t.shape[1]
    mse = total_se / n

    steps_2tau = 2 * STEPS_PER_TAU
    max_steps = min(steps_2tau, vl_traj.shape[1] - 1)
    ics = np.random.choice(vl_traj.shape[0], min(N_ROLLOUT_ICS, vl_traj.shape[0]), replace=False)
    errors = []
    with torch.no_grad():
        for ic in ics:
            true = vl_traj[ic, :max_steps+1].numpy()
            x = vl_traj[ic, 0]
            xs = [x.numpy()]
            for _ in range(max_steps):
                out = model.forward_one_step(x.unsqueeze(0), DT)
                x = out["x_tp1_pred"].squeeze(0)
                xs.append(x.numpy())
            nrmse = compute_nrmse(np.array(xs), true)
            errors.append(float(nrmse[min(steps_2tau, len(nrmse)-1)]))
    return mse, float(np.mean(errors))


def eval_dictionary(model, tr_traj, lasso_alpha):
    """Level-set decomposition at given alpha. Returns per-dim and union metrics."""
    x_flat = tr_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    model_cpu = model.cpu().eval()

    union_detected = set()
    dim_results = []

    for zi in range(LATENT):
        try:
            dr = full_decomposition(model_cpu, x_sub, zi, target_type='latent',
                                     max_degree=3, lasso_alpha=lasso_alpha, h_degree=4)
            pairs = sorted(zip(dr['inner']['active_names'],
                               dr['inner']['active_coeffs']),
                           key=lambda x: -abs(x[1]))
            all_terms = [(n, float(c)) for n, c in pairs if n != "1"]
            top5 = all_terms[:5]

            detected_this_dim = {n for n, c in top5}
            if dr['recon_r2'] > 0.8:
                union_detected |= detected_this_dim

            true_in_dim = detected_this_dim & TRUE_SET
            fp_in_dim = detected_this_dim - TRUE_SET

            dim_results.append({
                "dim": zi, "r2_hg": float(dr['recon_r2']),
                "n_terms": len(detected_this_dim),
                "true_terms": sorted(true_in_dim),
                "fp_terms": sorted(fp_in_dim),
                "all_terms": [(n, c) for n, c in all_terms[:8]],
            })
        except Exception:
            dim_results.append({
                "dim": zi, "r2_hg": 0.0, "n_terms": 0,
                "true_terms": [], "fp_terms": [], "all_terms": [],
            })

    # Union metrics
    tp = len(union_detected & TRUE_SET)
    recall = tp / N_TARGET
    precision = tp / max(len(union_detected), 1)
    jaccard = tp / max(len(union_detected | TRUE_SET), 1)
    n_fp = len(union_detected - TRUE_SET)

    # Per-dim sparsity
    terms_per_dim = [d["n_terms"] for d in dim_results]
    mean_tpd = np.mean(terms_per_dim) if terms_per_dim else 0

    # Sparse Recovery Score
    sparsity_penalty = min(1.0, N_TARGET / max(mean_tpd, 1e-10))
    sparse_score = recall * sparsity_penalty

    return {
        "recall": recall, "precision": precision, "jaccard": jaccard,
        "n_fp": n_fp, "mean_terms_per_dim": mean_tpd,
        "sparse_score": sparse_score,
        "detected": sorted(union_detected),
        "dim_results": dim_results,
    }


def main():
    t_start = time.time()
    print("=" * 70)
    print("  PyKAN Pruned Model: Alpha Sweep + Multi-Seed Validation")
    print(f"  Prune threshold: {PRUNE_THRESHOLD}")
    print("=" * 70)

    # Shared data (matches ablation_lorenz.py)
    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT, burn_in=BURN_IN, device="cpu")
    train_traj = traj[:N_TRAIN].contiguous()
    val_traj = traj[N_TRAIN:].contiguous()

    # ══════════════════════════════════════════════════════════════
    # STAGE 1: Alpha sweep at seed=0
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Alpha sweep (seed=0, prune@{PRUNE_THRESHOLD})")
    print(f"{'='*60}")

    print(f"\n  Training model (seed=0)...")
    t0 = time.time()
    model, tr, vl = train_prune_retrain(train_traj, val_traj, seed=0)
    mse, nrmse_2tau = eval_prediction(model, vl)
    print(f"  Trained in {time.time()-t0:.0f}s, MSE={mse:.4e}")

    stage1_results = []
    print(f"\n  {'alpha':>8} {'Recall':>7} {'Jac':>6} {'t/dim':>6} {'Score':>7} {'FP':>4} detected")
    print(f"  " + "-"*65)

    for alpha in ALPHA_SWEEP:
        d = eval_dictionary(model, tr, alpha)
        stage1_results.append({"alpha": alpha, **d})
        print(f"  {alpha:>8.0e} {d['recall']:>7.0%} {d['jaccard']:>6.3f} "
              f"{d['mean_terms_per_dim']:>6.1f} {d['sparse_score']:>7.3f} "
              f"{d['n_fp']:>4} {d['detected']}")

    # Find best alpha (max sparse_score, break ties by fewer FPs)
    best = max(stage1_results, key=lambda r: (r['sparse_score'], -r['n_fp']))
    best_alpha = best['alpha']
    print(f"\n  Best alpha: {best_alpha:.0e} (Score={best['sparse_score']:.3f})")

    del model
    gc.collect()

    # ══════════════════════════════════════════════════════════════
    # STAGE 2: Multi-seed validation at best alpha
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STAGE 2: 5 seeds at alpha={best_alpha:.0e}")
    print(f"{'='*60}")

    seed_results = []

    for seed in SEEDS:
        print(f"\n  seed={seed}")
        t0 = time.time()

        model, tr, vl = train_prune_retrain(train_traj, val_traj, seed)
        mse, nrmse_2tau = eval_prediction(model, vl)
        d = eval_dictionary(model, tr, best_alpha)
        elapsed = time.time() - t0

        result = {
            "seed": seed, "mse": mse, "nrmse_2tau": nrmse_2tau,
            "alpha": best_alpha, "wall_time": elapsed, **d,
        }
        seed_results.append(result)

        print(f"    MSE={mse:.4e} NRMSE@2τ={nrmse_2tau:.3f} "
              f"Rec={d['recall']:.0%} Jac={d['jaccard']:.3f} "
              f"Score={d['sparse_score']:.3f} t/d={d['mean_terms_per_dim']:.1f} "
              f"FP={d['n_fp']} ({elapsed:.0f}s)")

        # Print per-dim dictionary
        for dd in d['dim_results']:
            terms_str = " ".join(f"{c:+.4f}*{n}" for n, c in dd['all_terms'][:6])
            print(f"      z[{dd['dim']}] R²={dd['r2_hg']:.3f} [{dd['n_terms']} terms] {terms_str}")

        del model
        gc.collect()

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    mses = [r['mse'] for r in seed_results]
    nrmses = [r['nrmse_2tau'] for r in seed_results]
    recs = [r['recall'] for r in seed_results]
    jacs = [r['jaccard'] for r in seed_results]
    scores = [r['sparse_score'] for r in seed_results]
    tpds = [r['mean_terms_per_dim'] for r in seed_results]
    fps = [r['n_fp'] for r in seed_results]

    print(f"\n  Best alpha: {best_alpha:.0e}")
    print(f"  MSE:          {np.mean(mses):.4e} ± {np.std(mses):.1e}")
    print(f"  NRMSE@2τ:     {np.mean(nrmses):.3f} ± {np.std(nrmses):.3f}")
    print(f"  Recall:       {np.mean(recs):.2f} ± {np.std(recs):.2f}")
    print(f"  Jaccard:      {np.mean(jacs):.3f} ± {np.std(jacs):.3f}")
    print(f"  Sparse Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print(f"  Terms/dim:    {np.mean(tpds):.1f} ± {np.std(tpds):.1f}")
    print(f"  #FP:          {np.mean(fps):.1f} ± {np.std(fps):.1f}")

    # Per-term detection
    print(f"\n  Per-term detection across {len(SEEDS)} seeds:")
    for t in TRUE_DICT:
        rate = np.mean([t in r['detected'] for r in seed_results])
        print(f"    {t:>4}: {rate:.0%}")

    # Ablation table row
    print(f"\n  ABLATION TABLE ROW:")
    print(f"  PyKAN+prune@{PRUNE_THRESHOLD}+retrain & "
          f"${np.mean(mses):.2e} \\pm {np.std(mses):.0e}$ & "
          f"${np.mean(nrmses):.2f} \\pm {np.std(nrmses):.2f}$ & "
          f"${np.mean(jacs):.2f} \\pm {np.std(jacs):.2f}$ & "
          f"${np.mean([r['precision'] for r in seed_results]):.2f}$ & "
          f"${np.mean(recs):.2f}$ & "
          f"${np.mean(scores):.3f} \\pm {np.std(scores):.3f}$ \\\\")

    # Save
    save_data = {
        "prune_threshold": PRUNE_THRESHOLD,
        "best_alpha": best_alpha,
        "stage1": stage1_results,
        "stage2": seed_results,
        "summary": {
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "recall_mean": float(np.mean(recs)),
            "jaccard_mean": float(np.mean(jacs)),
            "jaccard_std": float(np.std(jacs)),
            "sparse_score_mean": float(np.mean(scores)),
            "sparse_score_std": float(np.std(scores)),
            "terms_per_dim_mean": float(np.mean(tpds)),
        },
    }
    with open(OUTDIR / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR / 'results.json'}")

    elapsed = time.time() - t_start
    print(f"  Total time: {elapsed/60:.0f} minutes")


if __name__ == "__main__":
    main()
