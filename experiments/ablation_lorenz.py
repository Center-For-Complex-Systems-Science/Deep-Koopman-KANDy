"""
Controlled ablation study for Deep-Koopman-KANDy on Lorenz.

Variants:
  A: Full Deep-Koopman-KANDy  (2L-KAN encoder/decoder + stable propagator)
  B: 1-layer KAN encoder/decoder + stable propagator
  C: Parameter-matched MLP encoder/decoder + stable propagator
  D: KAN encoder + MLP decoder + stable propagator
  E: MLP encoder + KAN decoder + stable propagator
  F: Full KAN encoder/decoder + unconstrained propagator
  G: Full model, prediction only (no level-set recovery)

All variants: latent_dim=5, same data, same optimizer, same loss weights.
5 seeds each.

Metrics:
  - One-step MSE
  - Rollout error at 2 tau_Lambda
  - Median R²(h∘g) across latent dims
  - Support detection: for each of {x,y,z,xy,xz}, detected as top-5 term?
  - False positive detection: {x²,y²,z²,yz}
  - Active terms per latent dim

Run from project root:
    python experiments/ablation_lorenz.py
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
from models import KANAutoEncoder, KANEncoder, KANDecoder, DeepKoopmanCT
from kan import KANLinear
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition

OUTDIR = Path("results/ablation_lorenz")
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
KAN_KWARGS = dict(grid_size=5, spline_order=3,
                  grid_range_l1=[-30, 50], grid_range_l2=[-10, 10])


# ── Autoencoder variants ─────────────────────────────────────

class SingleLayerKANAutoEncoder(nn.Module):
    """Variant B: 1-layer KAN (no mid layer, no cross-terms possible)."""
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.encoder = KANLinear(input_size, latent_size,
                                  grid_size=5, spline_order=3, grid_range=[-30, 50])
        self.decoder = KANLinear(latent_size, input_size,
                                  grid_size=5, spline_order=3, grid_range=[-10, 10])

    def forward(self, X):
        Y = self.encoder(X)
        Xr = self.decoder(Y)
        return Y, Xr


class MLPAutoEncoder(nn.Module):
    """Variant C: parameter-matched MLP."""
    def __init__(self, input_size, mid_size, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, mid_size), nn.SiLU(),
            nn.Linear(mid_size, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, mid_size), nn.SiLU(),
            nn.Linear(mid_size, input_size),
        )

    def forward(self, X):
        Y = self.encoder(X)
        Xr = self.decoder(Y)
        return Y, Xr


class MLPEncoder(nn.Module):
    def __init__(self, input_size, mid_size, latent_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, mid_size), nn.SiLU(),
            nn.Linear(mid_size, latent_size),
        )

    def forward(self, x):
        return self.net(x)


class MLPDecoder(nn.Module):
    def __init__(self, latent_size, mid_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, mid_size), nn.SiLU(),
            nn.Linear(mid_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class HybridKANEncMLPDec(nn.Module):
    """Variant D: KAN encoder + MLP decoder."""
    def __init__(self, input_size, mid_size, latent_size):
        super().__init__()
        self.encoder = KANEncoder(input_size, mid_size, latent_size, **KAN_KWARGS)
        self.decoder = MLPDecoder(latent_size, mid_size, input_size)

    def forward(self, X):
        Y = self.encoder(X)
        Xr = self.decoder(Y)
        return Y, Xr


class HybridMLPEncKANDec(nn.Module):
    """Variant E: MLP encoder + KAN decoder."""
    def __init__(self, input_size, mid_size, latent_size):
        super().__init__()
        self.encoder = MLPEncoder(input_size, mid_size, latent_size)
        self.decoder = KANDecoder(latent_size, mid_size, input_size, **KAN_KWARGS)

    def forward(self, X):
        Y = self.encoder(X)
        Xr = self.decoder(Y)
        return Y, Xr


# ── Training ─────────────────────────────────────────────────

def train_stable(ae, train_traj, val_traj, seed, epochs=200):
    """Train with stable propagator A = K - L^T L."""
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
        if ep % 50 == 0:
            print(f"        Ep {ep}")

    model.eval()
    return model, traj_full[:N_TRAIN].cpu(), traj_full[N_TRAIN:].cpu()


def train_unconstrained(ae, train_traj, val_traj, seed, epochs=200):
    """Train with unconstrained propagator (standard DeepKoopmanCT)."""
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
        if ep % 50 == 0:
            print(f"        Ep {ep}")

    model.eval()
    return model, traj_full[:N_TRAIN].cpu(), traj_full[N_TRAIN:].cpu()


# ── Evaluation ───────────────────────────────────────────────

def compute_nrmse(pred, true):
    std = true.reshape(-1, 3).std(axis=0)
    std = np.maximum(std, 1e-8)
    return (np.sqrt((pred - true)**2) / std[None, :]).mean(axis=1)


def eval_prediction(model, vl_traj):
    """One-step MSE and rollout error at 2 tau_L."""
    ds = PairDataset(vl_traj)
    dl = DataLoader(ds, batch_size=4096)
    total_se, n = 0.0, 0
    with torch.no_grad():
        for x_t, x_tp1 in dl:
            out = model.forward_one_step(x_t, DT)
            total_se += (out["x_tp1_pred"] - x_tp1).pow(2).sum().item()
            n += x_t.shape[0] * x_t.shape[1]
    one_step_mse = total_se / n

    # Rollout at 2 tau_L
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
            # Error at 2 tau_L (or end)
            rollout_errors.append(float(nrmse[min(steps_2tau, len(nrmse)-1)]))
    rollout_2tau = float(np.mean(rollout_errors))

    return one_step_mse, rollout_2tau


def eval_symbolic(model, tr_traj, top_n=5):
    """Level-set decomposition → detection metrics."""
    x_flat = tr_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    model_cpu = model.cpu().eval()

    detected = set()
    r2_hgs = []
    sparsities = []
    table = []

    for zi in range(LATENT):
        try:
            dr = full_decomposition(model_cpu, x_sub, zi, target_type='latent',
                                     max_degree=3, lasso_alpha=1e-5, h_degree=4)
            r2_hg = dr['recon_r2']
            r2_hgs.append(r2_hg)

            pairs = sorted(zip(dr['inner']['active_names'],
                               dr['inner']['active_coeffs']),
                           key=lambda x: -abs(x[1]))
            top = [(nm, c) for nm, c in pairs if nm != "1"][:top_n]
            n_active = len([nm for nm, c in pairs if nm != "1" and abs(c) > 1e-4])
            sparsities.append(n_active)

            h_coeffs = dr['outer'].get('h_coeffs', None)
            h_str = ""
            if h_coeffs is not None:
                for deg, c in enumerate(h_coeffs):
                    if deg == 0: continue
                    if abs(c) > 1e-4:
                        h_str += f"{c:+.2f}u^{deg} " if deg > 1 else f"{c:+.2f}u "
            h_str = h_str.strip() or "const"

            table.append({"dim": zi, "r2_hg": r2_hg, "top": top, "h": h_str})

            if r2_hg > 0.8:
                detected |= {nm for nm, c in top}
        except Exception as e:
            r2_hgs.append(0.0)
            sparsities.append(0)
            table.append({"dim": zi, "r2_hg": 0.0, "top": [], "h": f"ERR"})

    # Detection results
    tp_results = {t: t in detected for t in TRUE_DICT}
    fp_results = {t: t in detected for t in FALSE_TERMS}
    false_positives = detected - TRUE_SET
    det_rate = sum(tp_results.values()) / len(TRUE_DICT)
    support_f1 = 0.0
    tp = len(detected & TRUE_SET)
    prec = tp / max(len(detected), 1)
    rec = tp / len(TRUE_SET)
    if prec + rec > 0:
        support_f1 = 2 * prec * rec / (prec + rec)

    return {
        "median_r2_hg": float(np.median(r2_hgs)),
        "mean_sparsity": float(np.mean(sparsities)),
        "det_rate": det_rate,
        "tp_results": tp_results,
        "fp_results": fp_results,
        "false_positives": sorted(false_positives),
        "n_fp": len(false_positives),
        "support_f1": support_f1,
        "support_precision": prec,
        "support_recall": rec,
        "table": table,
    }


# ── Variant definitions ──────────────────────────────────────

def make_variant_A(seed):
    """Full: 2L-KAN + stable."""
    ae = KANAutoEncoder(3, MID, LATENT, **KAN_KWARGS)
    return ae, "stable"

def make_variant_B(seed):
    """1-layer KAN + stable."""
    ae = SingleLayerKANAutoEncoder(3, LATENT)
    return ae, "stable"

def make_variant_C(seed):
    """MLP + stable."""
    ae = MLPAutoEncoder(3, MID, LATENT)
    return ae, "stable"

def make_variant_D(seed):
    """KAN encoder + MLP decoder + stable."""
    ae = HybridKANEncMLPDec(3, MID, LATENT)
    return ae, "stable"

def make_variant_E(seed):
    """MLP encoder + KAN decoder + stable."""
    ae = HybridMLPEncKANDec(3, MID, LATENT)
    return ae, "stable"

def make_variant_F(seed):
    """Full KAN + unconstrained."""
    ae = KANAutoEncoder(3, MID, LATENT, **KAN_KWARGS)
    return ae, "unconstrained"

def make_variant_G(seed):
    """Full KAN + stable (prediction only, skip symbolic)."""
    ae = KANAutoEncoder(3, MID, LATENT, **KAN_KWARGS)
    return ae, "stable"


VARIANTS = {
    "A: 2L-KAN + stable":       (make_variant_A, True),
    "B: 1L-KAN + stable":       (make_variant_B, True),
    "C: MLP + stable":          (make_variant_C, True),
    "D: KAN-enc + MLP-dec":     (make_variant_D, True),
    "E: MLP-enc + KAN-dec":     (make_variant_E, True),
    "F: 2L-KAN + unconstrained":(make_variant_F, True),
    "G: Full (pred only)":      (make_variant_G, False),  # skip symbolic
}


# ── Main ─────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 80)
    print("  ABLATION STUDY: Deep-Koopman-KANDy on Lorenz")
    print("=" * 80)

    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:N_TRAIN].contiguous()
    val_traj = traj[N_TRAIN:].contiguous()
    print(f"  Data: train={train_traj.shape}, val={val_traj.shape}")
    print(f"  Variants: {len(VARIANTS)}, Seeds: {SEEDS}")

    all_results = {}

    for vname, (make_fn, do_symbolic) in VARIANTS.items():
        print(f"\n{'='*70}")
        print(f"  {vname}")
        print(f"{'='*70}")

        seed_results = []
        for seed in SEEDS:
            print(f"\n    seed={seed}")
            t0 = time.time()

            ae, prop_type = make_fn(seed)
            n_params = sum(p.numel() for p in ae.parameters())

            if prop_type == "stable":
                model, tr, vl = train_stable(ae, train_traj, val_traj, seed)
            else:
                model, tr, vl = train_unconstrained(ae, train_traj, val_traj, seed)

            # Prediction metrics
            mse, rollout_2tau = eval_prediction(model, vl)

            # Symbolic metrics
            if do_symbolic:
                sym = eval_symbolic(model, tr)
            else:
                sym = {
                    "median_r2_hg": None, "mean_sparsity": None,
                    "det_rate": None, "tp_results": None, "fp_results": None,
                    "false_positives": [], "n_fp": None,
                    "support_f1": None, "support_precision": None,
                    "support_recall": None, "table": [],
                }

            elapsed = time.time() - t0
            result = {
                "mse": mse, "rollout_2tau": rollout_2tau,
                "n_params": n_params, "wall_time": elapsed,
                **sym,
            }
            seed_results.append(result)

            sym_str = ""
            if do_symbolic:
                sym_str = (f"  R²(h∘g)={sym['median_r2_hg']:.3f}  "
                           f"Det={sym['det_rate']:.0%}  FP={sym['n_fp']}")
            print(f"    MSE={mse:.4e}  Roll@2τ={rollout_2tau:.3f}"
                  f"  params={n_params}{sym_str}  ({elapsed:.0f}s)")

            # Print decomposition table for seed 0
            if seed == 0 and do_symbolic and sym['table']:
                for row in sym['table']:
                    top_str = '  '.join(f'{c:+.4f}*{n}' for n, c in row['top'][:5])
                    print(f"      z[{row['dim']}] R²={row['r2_hg']:.3f}  h={row['h']}")
                    print(f"        g = {top_str}")

            del model, ae
            gc.collect()

        all_results[vname] = seed_results

    # ══════════════════════════════════════════════════════════════
    # Summary tables
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*100}")
    print("  TABLE X: Main ablation results")
    print(f"{'='*100}")

    print(f"\n  {'Variant':<28} {'Params':>6} {'1-Step MSE':>12} {'Roll@2τ':>9} "
          f"{'R²(h∘g)':>8} {'Det%':>5} {'F1':>5} {'#FP':>4}")
    print("  " + "-" * 85)

    for vname in VARIANTS:
        sr = all_results[vname]
        mses = [r['mse'] for r in sr]
        rolls = [r['rollout_2tau'] for r in sr]
        params = sr[0]['n_params']

        mse_str = f"{np.mean(mses):.2e}±{np.std(mses):.0e}"
        roll_str = f"{np.mean(rolls):.3f}±{np.std(rolls):.2f}"

        if sr[0]['median_r2_hg'] is not None:
            r2s = [r['median_r2_hg'] for r in sr]
            dets = [r['det_rate'] for r in sr]
            f1s = [r['support_f1'] for r in sr]
            fps = [r['n_fp'] for r in sr]
            r2_str = f"{np.mean(r2s):.3f}"
            det_str = f"{np.mean(dets):.0%}"
            f1_str = f"{np.mean(f1s):.2f}"
            fp_str = f"{np.mean(fps):.1f}"
        else:
            r2_str = det_str = f1_str = fp_str = "--"

        print(f"  {vname:<28} {params:>6} {mse_str:>12} {roll_str:>9} "
              f"{r2_str:>8} {det_str:>5} {f1_str:>5} {fp_str:>4}")

    # ── Detection heatmap ──
    print(f"\n\n{'='*100}")
    print("  TABLE Z: Per-term detection rates")
    print(f"{'='*100}")

    print(f"\n  {'Variant':<28}", end="")
    for t in TRUE_DICT:
        print(f" {t:>4}", end="")
    print("  |", end="")
    for t in FALSE_TERMS:
        print(f" {t:>4}", end="")
    print()
    print("  " + "-" * 80)

    for vname in VARIANTS:
        sr = all_results[vname]
        if sr[0]['tp_results'] is None:
            print(f"  {vname:<28}  (prediction only)")
            continue

        print(f"  {vname:<28}", end="")
        for t in TRUE_DICT:
            rate = np.mean([r['tp_results'][t] for r in sr])
            mark = "  ✓ " if rate == 1.0 else f"{rate:.0%}" if rate > 0 else "  · "
            print(f" {mark:>4}", end="")
        print("  |", end="")
        for t in FALSE_TERMS:
            rate = np.mean([r['fp_results'][t] for r in sr])
            mark = f"{rate:.0%}" if rate > 0 else "  · "
            print(f" {mark:>4}", end="")
        print()

    # ── Save ──
    save_data = {}
    for vname in VARIANTS:
        sr = all_results[vname]
        save_data[vname] = {
            "n_params": sr[0]['n_params'],
            "mse_mean": float(np.mean([r['mse'] for r in sr])),
            "mse_std": float(np.std([r['mse'] for r in sr])),
            "rollout_2tau_mean": float(np.mean([r['rollout_2tau'] for r in sr])),
            "rollout_2tau_std": float(np.std([r['rollout_2tau'] for r in sr])),
            "seeds": [{
                "mse": r['mse'], "rollout_2tau": r['rollout_2tau'],
                "median_r2_hg": r['median_r2_hg'],
                "det_rate": r['det_rate'], "n_fp": r['n_fp'],
                "support_f1": r['support_f1'],
                "false_positives": r['false_positives'],
                "wall_time": r['wall_time'],
            } for r in sr],
        }
        if sr[0]['tp_results'] is not None:
            save_data[vname]["per_term"] = {
                t: float(np.mean([r['tp_results'][t] for r in sr]))
                for t in TRUE_DICT
            }
            save_data[vname]["per_fp"] = {
                t: float(np.mean([r['fp_results'][t] for r in sr]))
                for t in FALSE_TERMS
            }

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR / 'results.json'}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vnames = [v for v in VARIANTS]
    short_names = ["A:2L-KAN\n+stable", "B:1L-KAN\n+stable", "C:MLP\n+stable",
                   "D:KAN-enc\nMLP-dec", "E:MLP-enc\nKAN-dec", "F:2L-KAN\n+unconstr",
                   "G:Full\n(pred only)"]

    # Panel 1: One-step MSE
    ax = axes[0]
    mse_means = [np.mean([r['mse'] for r in all_results[v]]) for v in vnames]
    mse_stds = [np.std([r['mse'] for r in all_results[v]]) for v in vnames]
    ax.bar(range(len(vnames)), mse_means, yerr=mse_stds, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(vnames)))
    ax.set_xticklabels(short_names, fontsize=7)
    ax.set_ylabel("One-step MSE")
    ax.set_title("Prediction: One-step MSE")
    ax.set_yscale("log")

    # Panel 2: Rollout error at 2 tau_L
    ax = axes[1]
    roll_means = [np.mean([r['rollout_2tau'] for r in all_results[v]]) for v in vnames]
    roll_stds = [np.std([r['rollout_2tau'] for r in all_results[v]]) for v in vnames]
    ax.bar(range(len(vnames)), roll_means, yerr=roll_stds, color='coral', alpha=0.8)
    ax.set_xticks(range(len(vnames)))
    ax.set_xticklabels(short_names, fontsize=7)
    ax.set_ylabel("NRMSE @ 2τ_Λ")
    ax.set_title("Rollout Error at 2 Lyapunov Times")

    # Panel 3: Detection heatmap
    ax = axes[2]
    all_terms = TRUE_DICT + FALSE_TERMS
    sym_variants = [v for v in vnames if all_results[v][0]['tp_results'] is not None]
    sym_short = [short_names[i] for i, v in enumerate(vnames) if all_results[v][0]['tp_results'] is not None]
    data = np.zeros((len(sym_variants), len(all_terms)))
    for i, v in enumerate(sym_variants):
        sr = all_results[v]
        for j, t in enumerate(all_terms):
            if t in TRUE_DICT:
                data[i, j] = np.mean([r['tp_results'][t] for r in sr])
            else:
                data[i, j] = np.mean([r['fp_results'][t] for r in sr])

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(all_terms)))
    ax.set_xticklabels(all_terms, fontsize=9)
    ax.set_yticks(range(len(sym_variants)))
    ax.set_yticklabels(sym_short, fontsize=7)
    for i in range(len(sym_variants)):
        for j in range(len(all_terms)):
            v = data[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if v < 0.5 else "black")
    # Separator line between true and false terms
    ax.axvline(len(TRUE_DICT) - 0.5, color="black", lw=2)
    ax.set_title("Dictionary Detection (left=true, right=FP)")

    fig.suptitle("Lorenz Ablation: Deep-Koopman-KANDy", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "ablation_lorenz.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTDIR / "ablation_lorenz.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ablation_lorenz.png/pdf")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.0f} minutes")


if __name__ == "__main__":
    main()
