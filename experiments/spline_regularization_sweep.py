"""
Spline Regularization Sweep for Deep-Koopman-KANDy.

Tests L1 + entropy regularization on KAN spline weights to suppress
false positive monomials in the level-set dictionary recovery.

Regularization:
    l1_per_edge = spline_weight.abs().mean(dim=-1)   # (out, in)
    l1_total = l1_per_edge.sum()
    p = l1_per_edge / l1_total
    entropy = -sum(p * log(p))
    reg_loss = l1_total + entropy

Sweep: reg_weight in {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1}
5 seeds each.

Run from project root:
    python experiments/spline_regularization_sweep.py
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
from models import KANAutoEncoder
from kan import KANLinear
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition

OUTDIR = Path("results/spline_regularization_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
N_TRAJ = 60
N_TRAIN = 48
LATENT = 5
MID = 5
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 200

LAMBDA_MAX = 0.906
TAU_L = 1.0 / LAMBDA_MAX
STEPS_PER_TAU = int(TAU_L / DT)
N_ROLLOUT_ICS = 10

TRUE_DICT = ["x", "y", "z", "xy", "xz"]
TRUE_SET = set(TRUE_DICT)
FALSE_TERMS = ["x^2", "y^2", "z^2", "yz"]

REG_WEIGHTS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

KAN_KWARGS = dict(grid_size=5, spline_order=3,
                  grid_range_l1=[-30, 50], grid_range_l2=[-10, 10])


def get_spline_regularization(ae):
    """L1 + entropy regularization on KAN spline weights."""
    reg = 0.0
    for module in ae.modules():
        if isinstance(module, KANLinear):
            l1_per_edge = module.spline_weight.abs().mean(-1)
            l1_total = l1_per_edge.sum()
            p = l1_per_edge / (l1_total + 1e-10)
            entropy = -torch.sum(p * (p + 1e-10).log())
            reg = reg + l1_total + entropy
    return reg


def compute_nrmse(pred, true):
    std = true.reshape(-1, 3).std(axis=0)
    std = np.maximum(std, 1e-8)
    return (np.sqrt((pred - true)**2) / std[None, :]).mean(axis=1)


def train_and_evaluate(train_traj, val_traj, seed, reg_weight):
    set_seed(seed)
    ae = KANAutoEncoder(3, MID, LATENT, **KAN_KWARGS)
    traj_full = torch.cat([train_traj, val_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    model = DeepKoopmanCTStable(ae, LATENT)
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
            if reg_weight > 0:
                loss = loss + reg_weight * get_spline_regularization(model.ae)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()

    # One-step MSE
    vl_traj = traj_full[N_TRAIN:].cpu()
    ds = PairDataset(vl_traj)
    dl = DataLoader(ds, batch_size=4096)
    total_se, n = 0.0, 0
    with torch.no_grad():
        for x_t, x_tp1 in dl:
            out = model.forward_one_step(x_t, DT)
            total_se += (out["x_tp1_pred"] - x_tp1).pow(2).sum().item()
            n += x_t.shape[0] * x_t.shape[1]
    one_step_mse = total_se / n

    # NRMSE@2tau
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
    nrmse_2tau = float(np.mean(rollout_errors))

    # Dictionary extraction
    tr_traj = traj_full[:N_TRAIN].cpu()
    x_flat = tr_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    model_cpu = model.cpu().eval()

    detected = set()
    r2_hgs = []
    dim_details = []

    for zi in range(LATENT):
        dr = full_decomposition(model_cpu, x_sub, zi, target_type='latent',
                                max_degree=3, lasso_alpha=1e-5, h_degree=4)
        r2_hgs.append(dr['recon_r2'])
        pairs = sorted(zip(dr['inner']['active_names'],
                           dr['inner']['active_coeffs']),
                       key=lambda x: -abs(x[1]))
        top = [(n, float(c)) for n, c in pairs if n != "1"][:5]
        all_terms = [(n, float(c)) for n, c in pairs if n != "1"]

        if dr['recon_r2'] > 0.8:
            detected |= {n for n, c in top}

        true_terms = [(n, c) for n, c in all_terms if n in TRUE_SET]
        fp_terms = [(n, c) for n, c in all_terms if n not in TRUE_SET]
        dim_details.append({
            "dim": zi, "r2_hg": float(dr['recon_r2']),
            "true_terms": true_terms, "fp_terms": fp_terms,
        })

    tp_results = {t: t in detected for t in TRUE_DICT}
    fp_results = {t: t in detected for t in FALSE_TERMS}
    false_positives = detected - TRUE_SET

    tp = len(detected & TRUE_SET)
    prec = tp / max(len(detected), 1)
    rec = tp / len(TRUE_SET)
    jaccard = tp / max(len(detected | TRUE_SET), 1)

    del model, ae
    gc.collect()

    return {
        "one_step_mse": one_step_mse,
        "nrmse_2tau": nrmse_2tau,
        "median_r2_hg": float(np.median(r2_hgs)),
        "jaccard": jaccard,
        "precision": prec,
        "recall": rec,
        "n_fp": len(false_positives),
        "false_positives": sorted(false_positives),
        "detected": sorted(detected),
        "tp_results": tp_results,
        "fp_results": fp_results,
        "dim_details": dim_details,
    }


def main():
    t_start = time.time()
    print("=" * 70)
    print("  SPLINE REGULARIZATION SWEEP")
    print(f"  Reg weights: {REG_WEIGHTS}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:N_TRAIN].contiguous()
    val_traj = traj[N_TRAIN:].contiguous()

    all_results = {}

    for rw in REG_WEIGHTS:
        print(f"\n{'='*60}")
        print(f"  reg_weight = {rw}")
        print(f"{'='*60}")

        seed_results = []
        for seed in SEEDS:
            print(f"    seed={seed}", end="", flush=True)
            t0 = time.time()
            r = train_and_evaluate(train_traj, val_traj, seed, rw)
            elapsed = time.time() - t0
            r["wall_time"] = elapsed
            seed_results.append(r)
            print(f"  MSE={r['one_step_mse']:.4e}  Jac={r['jaccard']:.3f}  "
                  f"Rec={r['recall']:.0%}  FP={r['n_fp']}  ({elapsed:.0f}s)")

        all_results[str(rw)] = seed_results

    # Summary table
    print(f"\n\n{'='*90}")
    print("  SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"\n  {'reg_wt':>8} {'MSE':>14} {'NRMSE@2τ':>12} {'Jaccard':>12} "
          f"{'Prec':>8} {'Recall':>8} {'#FP':>5}")
    print("  " + "-"*75)

    for rw in REG_WEIGHTS:
        sr = all_results[str(rw)]
        mses = [r['one_step_mse'] for r in sr]
        nrmses = [r['nrmse_2tau'] for r in sr]
        jacs = [r['jaccard'] for r in sr]
        precs = [r['precision'] for r in sr]
        recs = [r['recall'] for r in sr]
        fps = [r['n_fp'] for r in sr]
        print(f"  {rw:>8.3f} {np.mean(mses):.4e}±{np.std(mses):.0e} "
              f"{np.mean(nrmses):.2f}±{np.std(nrmses):.2f} "
              f"{np.mean(jacs):.3f}±{np.std(jacs):.3f} "
              f"{np.mean(precs):.2f}±{np.std(precs):.2f} "
              f"{np.mean(recs):.2f}±{np.std(recs):.2f} "
              f"{np.mean(fps):.1f}")

    # Per-term detection rates
    print(f"\n  Per-term detection rates:")
    print(f"  {'reg_wt':>8}", end="")
    for t in TRUE_DICT + FALSE_TERMS:
        print(f"  {t:>5}", end="")
    print()
    print("  " + "-"*70)
    for rw in REG_WEIGHTS:
        sr = all_results[str(rw)]
        print(f"  {rw:>8.3f}", end="")
        for t in TRUE_DICT:
            rate = np.mean([r['tp_results'][t] for r in sr])
            print(f"  {rate:>5.0%}", end="")
        for t in FALSE_TERMS:
            rate = np.mean([r['fp_results'][t] for r in sr])
            print(f"  {rate:>5.0%}", end="")
        print()

    # Save
    save_data = {}
    for rw in REG_WEIGHTS:
        sr = all_results[str(rw)]
        save_data[str(rw)] = {
            "mse_mean": float(np.mean([r['one_step_mse'] for r in sr])),
            "mse_std": float(np.std([r['one_step_mse'] for r in sr])),
            "nrmse_2tau_mean": float(np.mean([r['nrmse_2tau'] for r in sr])),
            "jaccard_mean": float(np.mean([r['jaccard'] for r in sr])),
            "jaccard_std": float(np.std([r['jaccard'] for r in sr])),
            "precision_mean": float(np.mean([r['precision'] for r in sr])),
            "recall_mean": float(np.mean([r['recall'] for r in sr])),
            "n_fp_mean": float(np.mean([r['n_fp'] for r in sr])),
            "per_term": {t: float(np.mean([r['tp_results'][t] for r in sr]))
                         for t in TRUE_DICT},
            "per_fp": {t: float(np.mean([r['fp_results'][t] for r in sr]))
                       for t in FALSE_TERMS},
            "seeds": [{
                "one_step_mse": r["one_step_mse"],
                "nrmse_2tau": r["nrmse_2tau"],
                "jaccard": r["jaccard"],
                "precision": r["precision"],
                "recall": r["recall"],
                "n_fp": r["n_fp"],
                "false_positives": r["false_positives"],
                "detected": r["detected"],
                "dim_details": r["dim_details"],
            } for r in sr],
        }

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR / 'results.json'}")

    elapsed = time.time() - t_start
    print(f"  Total time: {elapsed/60:.0f} minutes")


if __name__ == "__main__":
    main()
