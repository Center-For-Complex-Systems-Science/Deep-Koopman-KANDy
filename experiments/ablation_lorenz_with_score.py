"""
Re-run of ablation_lorenz.py that also computes the Sparse Recovery Score.

Score = recall × min(1, |target| / mean_terms_per_dim)

Same variants, same seeds, same data. Adds terms_per_dim and sparse_score
to the output.

Run from project root:
    python experiments/ablation_lorenz_with_score.py
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
from models import KANAutoEncoder, KANEncoder, KANDecoder, DeepKoopmanCT
from kan import KANLinear
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition

OUTDIR = Path("results/ablation_lorenz_score")
OUTDIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
N_TRAJ = 60
N_TRAIN = 48
LATENT = 5
MID = 5
SEEDS = [0, 1, 2, 3, 4]

LAMBDA_MAX = 0.906
TAU_L = 1.0 / LAMBDA_MAX
STEPS_PER_TAU = int(TAU_L / DT)
N_ROLLOUT_ICS = 10

TRUE_DICT = ["x", "y", "z", "xy", "xz"]
TRUE_SET = set(TRUE_DICT)
FALSE_TERMS = ["x^2", "y^2", "z^2", "yz"]
N_TARGET = len(TRUE_SET)
KAN_KWARGS = dict(grid_size=5, spline_order=3,
                  grid_range_l1=[-30, 50], grid_range_l2=[-10, 10])


# ── Autoencoder variants (same as ablation_lorenz.py) ──

class SingleLayerKANAutoEncoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.encoder = KANLinear(input_size, latent_size,
                                  grid_size=5, spline_order=3, grid_range=[-30, 50])
        self.decoder = KANLinear(latent_size, input_size,
                                  grid_size=5, spline_order=3, grid_range=[-10, 10])
    def forward(self, X):
        Y = self.encoder(X)
        return Y, self.decoder(Y)

class MLPAutoEncoder(nn.Module):
    def __init__(self, input_size, mid_size, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, mid_size), nn.SiLU(),
                                      nn.Linear(mid_size, latent_size))
        self.decoder = nn.Sequential(nn.Linear(latent_size, mid_size), nn.SiLU(),
                                      nn.Linear(mid_size, input_size))
    def forward(self, X):
        Y = self.encoder(X)
        return Y, self.decoder(Y)

class MLPEncoder(nn.Module):
    def __init__(self, input_size, mid_size, latent_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, mid_size), nn.SiLU(),
                                  nn.Linear(mid_size, latent_size))
    def forward(self, x): return self.net(x)

class MLPDecoder(nn.Module):
    def __init__(self, latent_size, mid_size, output_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_size, mid_size), nn.SiLU(),
                                  nn.Linear(mid_size, output_size))
    def forward(self, x): return self.net(x)

class HybridKANEncMLPDec(nn.Module):
    def __init__(self, input_size, mid_size, latent_size):
        super().__init__()
        self.encoder = KANEncoder(input_size, mid_size, latent_size, **KAN_KWARGS)
        self.decoder = MLPDecoder(latent_size, mid_size, input_size)
    def forward(self, X):
        Y = self.encoder(X)
        return Y, self.decoder(Y)

class HybridMLPEncKANDec(nn.Module):
    def __init__(self, input_size, mid_size, latent_size):
        super().__init__()
        self.encoder = MLPEncoder(input_size, mid_size, latent_size)
        self.decoder = KANDecoder(latent_size, mid_size, input_size, **KAN_KWARGS)
    def forward(self, X):
        Y = self.encoder(X)
        return Y, self.decoder(Y)


# ── Training ──

def train_stable(ae, train_traj, val_traj, seed, epochs=200):
    set_seed(seed)
    traj_full = torch.cat([train_traj, val_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    model = DeepKoopmanCTStable(ae, LATENT)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse_fn = nn.MSELoss()
    for ep in range(1, epochs + 1):
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
    model.eval()
    return model, traj_full[:N_TRAIN].cpu(), traj_full[N_TRAIN:].cpu()

def train_unconstrained(ae, train_traj, val_traj, seed, epochs=200):
    set_seed(seed)
    traj_full = torch.cat([train_traj, val_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    model = DeepKoopmanCT(ae, LATENT)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse_fn = nn.MSELoss()
    for ep in range(1, epochs + 1):
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
    model.eval()
    return model, traj_full[:N_TRAIN].cpu(), traj_full[N_TRAIN:].cpu()


# ── Evaluation ──

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

def eval_symbolic(model, tr_traj, top_n=5):
    x_flat = tr_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    model_cpu = model.cpu().eval()

    detected = set()
    r2_hgs = []
    terms_per_dim = []
    dim_details = []

    for zi in range(LATENT):
        try:
            dr = full_decomposition(model_cpu, x_sub, zi, target_type='latent',
                                     max_degree=3, lasso_alpha=1e-5, h_degree=4)
            r2_hg = dr['recon_r2']
            r2_hgs.append(r2_hg)
            pairs = sorted(zip(dr['inner']['active_names'],
                               dr['inner']['active_coeffs']),
                           key=lambda x: -abs(x[1]))
            top = [(nm, float(c)) for nm, c in pairs if nm != "1"][:top_n]
            detected_this = {nm for nm, c in top}
            terms_per_dim.append(len(detected_this))

            if r2_hg > 0.8:
                detected |= detected_this

            true_in = sorted(detected_this & TRUE_SET)
            fp_in = sorted(detected_this - TRUE_SET)
            dim_details.append({
                "dim": zi, "r2_hg": float(r2_hg),
                "n_terms": len(detected_this),
                "true": true_in, "fp": fp_in,
                "top_terms": top,
            })
        except Exception:
            r2_hgs.append(0.0)
            terms_per_dim.append(0)
            dim_details.append({"dim": zi, "r2_hg": 0.0, "n_terms": 0,
                                "true": [], "fp": [], "top_terms": []})

    tp_results = {t: t in detected for t in TRUE_DICT}
    fp_results = {t: t in detected for t in FALSE_TERMS}
    false_positives = detected - TRUE_SET
    tp = len(detected & TRUE_SET)
    prec = tp / max(len(detected), 1)
    rec = tp / len(TRUE_SET)
    jaccard = tp / max(len(detected | TRUE_SET), 1)

    mean_tpd = float(np.mean(terms_per_dim)) if terms_per_dim else 0
    sparsity_penalty = min(1.0, N_TARGET / max(mean_tpd, 1e-10))
    sparse_score = rec * sparsity_penalty

    return {
        "median_r2_hg": float(np.median(r2_hgs)),
        "det_rate": rec, "support_precision": prec, "support_recall": rec,
        "support_f1": 2*prec*rec/max(prec+rec, 1e-10),
        "jaccard": jaccard,
        "tp_results": tp_results, "fp_results": fp_results,
        "false_positives": sorted(false_positives), "n_fp": len(false_positives),
        "mean_terms_per_dim": mean_tpd,
        "sparse_score": sparse_score,
        "dim_details": dim_details,
    }


# ── Variants ──

def make_A(s): return KANAutoEncoder(3, MID, LATENT, **KAN_KWARGS), "stable"
def make_B(s): return SingleLayerKANAutoEncoder(3, LATENT), "stable"
def make_C(s): return MLPAutoEncoder(3, MID, LATENT), "stable"
def make_D(s): return HybridKANEncMLPDec(3, MID, LATENT), "stable"
def make_E(s): return HybridMLPEncKANDec(3, MID, LATENT), "stable"
def make_F(s): return KANAutoEncoder(3, MID, LATENT, **KAN_KWARGS), "unconstrained"

VARIANTS = {
    "A: 2L-KAN + stable":       (make_A, True),
    "B: 1L-KAN + stable":       (make_B, True),
    "C: MLP + stable":          (make_C, True),
    "D: KAN-enc + MLP-dec":     (make_D, True),
    "E: MLP-enc + KAN-dec":     (make_E, True),
    "F: 2L-KAN + unconstrained":(make_F, True),
}


def main():
    t_start = time.time()
    print("=" * 80)
    print("  ABLATION WITH SPARSE RECOVERY SCORE")
    print("=" * 80)

    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:N_TRAIN].contiguous()
    val_traj = traj[N_TRAIN:].contiguous()

    all_results = {}

    for vname, (make_fn, do_symbolic) in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"  {vname}")
        print(f"{'='*60}")

        seed_results = []
        for seed in SEEDS:
            print(f"    seed={seed}", end="", flush=True)
            t0 = time.time()
            ae, prop_type = make_fn(seed)
            n_params = sum(p.numel() for p in ae.parameters())
            if prop_type == "stable":
                model, tr, vl = train_stable(ae, train_traj, val_traj, seed)
            else:
                model, tr, vl = train_unconstrained(ae, train_traj, val_traj, seed)

            mse, rollout_2tau = eval_prediction(model, vl)
            sym = eval_symbolic(model, tr)
            elapsed = time.time() - t0

            result = {"mse": mse, "rollout_2tau": rollout_2tau,
                      "n_params": n_params, "wall_time": elapsed, **sym}
            seed_results.append(result)

            print(f"  MSE={mse:.4e} Jac={sym['jaccard']:.3f} "
                  f"Score={sym['sparse_score']:.3f} t/d={sym['mean_terms_per_dim']:.1f} "
                  f"Rec={sym['support_recall']:.0%} ({elapsed:.0f}s)")

            del model, ae
            gc.collect()

        all_results[vname] = seed_results

    # ── Summary ──
    print(f"\n\n{'='*90}")
    print("  SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"\n  {'Variant':<28} {'MSE':>12} {'NRMSE@2τ':>10} {'Jaccard':>10} "
          f"{'Score':>8} {'t/d':>5} {'Rec':>5} {'Prec':>5}")
    print("  " + "-"*85)

    for vname in VARIANTS:
        sr = all_results[vname]
        mses = [r['mse'] for r in sr]
        rolls = [r['rollout_2tau'] for r in sr]
        jacs = [r['jaccard'] for r in sr]
        scores = [r['sparse_score'] for r in sr]
        tpds = [r['mean_terms_per_dim'] for r in sr]
        recs = [r['support_recall'] for r in sr]
        precs = [r['support_precision'] for r in sr]

        print(f"  {vname:<28} {np.mean(mses):.4e} {np.mean(rolls):>10.2f} "
              f"{np.mean(jacs):.3f}±{np.std(jacs):.3f} "
              f"{np.mean(scores):.3f}±{np.std(scores):.3f} "
              f"{np.mean(tpds):>5.1f} {np.mean(recs):>5.0%} {np.mean(precs):>5.2f}")

    # Save
    save_data = {}
    for vname in VARIANTS:
        sr = all_results[vname]
        save_data[vname] = {
            "n_params": sr[0]['n_params'],
            "mse_mean": float(np.mean([r['mse'] for r in sr])),
            "jaccard_mean": float(np.mean([r['jaccard'] for r in sr])),
            "jaccard_std": float(np.std([r['jaccard'] for r in sr])),
            "sparse_score_mean": float(np.mean([r['sparse_score'] for r in sr])),
            "sparse_score_std": float(np.std([r['sparse_score'] for r in sr])),
            "mean_terms_per_dim": float(np.mean([r['mean_terms_per_dim'] for r in sr])),
            "recall_mean": float(np.mean([r['support_recall'] for r in sr])),
            "precision_mean": float(np.mean([r['support_precision'] for r in sr])),
            "per_term": {t: float(np.mean([r['tp_results'][t] for r in sr]))
                         for t in TRUE_DICT},
            "per_fp": {t: float(np.mean([r['fp_results'][t] for r in sr]))
                       for t in FALSE_TERMS},
            "seeds": [{
                "mse": r['mse'], "rollout_2tau": r['rollout_2tau'],
                "jaccard": r['jaccard'], "sparse_score": r['sparse_score'],
                "mean_terms_per_dim": r['mean_terms_per_dim'],
                "recall": r['support_recall'], "precision": r['support_precision'],
                "n_fp": r['n_fp'], "detected": sorted(set().union(*[set(d['true']+d['fp']) for d in r['dim_details']])),
                "dim_details": r['dim_details'],
            } for r in sr],
        }

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR / 'results.json'}")
    print(f"  Total time: {(time.time()-t_start)/60:.0f} minutes")


if __name__ == "__main__":
    main()
