"""
Quick GPU regularization sweep with full coefficient output.

Tests reg weights [0, 0.0001, 0.0005, 0.001, 0.005, 0.01] at seed 0
on GPU and prints the complete dictionary with all coefficients.

Run from project root:
    python experiments/gpu_reg_sweep_detailed.py
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
import warnings
warnings.filterwarnings("ignore")

from data import simulate_lorenz, PairDataset
from models import KANAutoEncoder
from kan import KANLinear
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition

OUTDIR = Path("results/gpu_reg_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
LATENT = 5
MID = 5
TRUE = {"x", "y", "z", "xy", "xz"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KAN_KWARGS = dict(grid_size=5, spline_order=3,
                  grid_range_l1=[-30, 50], grid_range_l2=[-10, 10])

REG_WEIGHTS = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
SEED = 0


def get_spline_reg(ae):
    reg = 0.0
    for m in ae.modules():
        if isinstance(m, KANLinear):
            l1 = m.spline_weight.abs().mean(-1)
            total = l1.sum()
            p = l1 / (total + 1e-10)
            reg = reg + total - torch.sum(p * (p + 1e-10).log())
    return reg


def main():
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    set_seed(42)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:48].contiguous()
    val_traj = traj[48:].contiguous()
    traj_full = torch.cat([train_traj, val_traj], dim=0)

    all_results = {}

    for rw in REG_WEIGHTS:
        print(f"\n{'='*70}")
        print(f"  reg_weight = {rw}")
        print(f"{'='*70}")

        set_seed(SEED)
        ae = KANAutoEncoder(3, MID, LATENT, **KAN_KWARGS)
        model = DeepKoopmanCTStable(ae, LATENT).to(DEVICE)
        train_dl = DataLoader(PairDataset(traj_full[:48]),
                              batch_size=2048, shuffle=True, drop_last=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        mse_fn = nn.MSELoss()

        for ep in range(1, 201):
            model.train()
            for x_t, x_tp1 in train_dl:
                x_t, x_tp1 = x_t.to(DEVICE), x_tp1.to(DEVICE)
                out = model.forward_one_step(x_t, DT)
                z_true = model.ae.encoder(x_tp1)
                loss = (mse_fn(out["x_tp1_pred"], x_tp1)
                        + mse_fn(out["z_tp1_pred"], z_true)
                        + 0.5 * mse_fn(out["x_t_rec"], x_t))
                if rw > 0:
                    loss = loss + rw * get_spline_reg(model.ae)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

        model = model.cpu().eval()

        # One-step MSE
        ds = PairDataset(val_traj)
        dl = DataLoader(ds, batch_size=4096)
        total_se, n = 0.0, 0
        with torch.no_grad():
            for x_t, x_tp1 in dl:
                out = model.forward_one_step(x_t, DT)
                total_se += (out["x_tp1_pred"] - x_tp1).pow(2).sum().item()
                n += x_t.shape[0] * x_t.shape[1]
        mse = total_se / n

        # Decomposition
        x_flat = train_traj.reshape(-1, 3)
        idx = np.random.choice(len(x_flat), 5000, replace=False)
        x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)

        detected = set()
        dim_results = []

        for zi in range(LATENT):
            dr = full_decomposition(model, x_sub, zi, target_type='latent',
                                    max_degree=3, lasso_alpha=1e-5, h_degree=4)
            pairs = sorted(zip(dr['inner']['active_names'],
                               dr['inner']['active_coeffs']),
                           key=lambda x: -abs(x[1]))
            all_terms = [(n, float(c)) for n, c in pairs if n != "1"]
            top5 = all_terms[:5]

            if dr['recon_r2'] > 0.8:
                detected |= {n for n, c in top5}

            true_terms = [(n, c) for n, c in all_terms if n in TRUE]
            fp_terms = [(n, c) for n, c in all_terms if n not in TRUE]

            dim_results.append({
                "dim": zi,
                "r2_hg": float(dr['recon_r2']),
                "true_terms": true_terms,
                "fp_terms": fp_terms,
            })

        tp = detected & TRUE
        fp = detected - TRUE
        jac = len(tp) / max(len(detected | TRUE), 1)
        rec = len(tp) / len(TRUE)

        print(f"\n  MSE: {mse:.4e}  Jaccard: {jac:.3f}  Recall: {rec:.0%}")
        print(f"  Detected: {sorted(detected)}")
        print(f"  TRUE: {sorted(tp)}  FP: {sorted(fp)}")

        # Full coefficient table
        print(f"\n  {'dim':>4} {'R²':>6} | TRUE terms                                    | FP terms")
        print(f"  " + "-"*90)

        for d in dim_results:
            true_str = "  ".join(f"{c:+.4f}*{n}" for n, c in d['true_terms'])
            fp_str = "  ".join(f"{c:+.4f}*{n}" for n, c in d['fp_terms'][:5])

            print(f"  z[{d['dim']}] {d['r2_hg']:>6.3f} | {true_str}")
            if d['fp_terms']:
                print(f"              | FP: {fp_str}")

        # Bar chart view
        print(f"\n  Coefficient magnitudes:")
        for d in dim_results:
            print(f"\n  z[{d['dim']}] R²={d['r2_hg']:.3f}:")
            all_t = d['true_terms'] + d['fp_terms']
            all_t.sort(key=lambda x: -abs(x[1]))
            for n, c in all_t[:8]:
                status = "TRUE" if n in TRUE else "FP"
                bar = "█" * int(min(abs(c) / 0.005, 40))
                print(f"    {c:>+10.6f}  {n:>8}  {status:>4}  {bar}")

        all_results[str(rw)] = {
            "mse": mse, "jaccard": jac, "recall": rec,
            "detected": sorted(detected),
            "fp": sorted(fp),
            "dims": dim_results,
        }

        del model, ae
        torch.cuda.empty_cache()

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY (GPU, seed={SEED})")
    print(f"{'='*70}")
    print(f"\n  {'reg':>8} {'MSE':>10} {'Jaccard':>8} {'Recall':>7} {'FPs'}")
    print(f"  " + "-"*50)
    for rw in REG_WEIGHTS:
        r = all_results[str(rw)]
        print(f"  {rw:>8.4f} {r['mse']:>10.4e} {r['jaccard']:>8.3f} "
              f"{r['recall']:>7.0%} {r['fp']}")

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR / 'results.json'}")


if __name__ == "__main__":
    main()
