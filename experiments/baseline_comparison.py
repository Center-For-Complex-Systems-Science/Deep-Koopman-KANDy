"""
Baseline comparison: dictionary detection rate on Lorenz.

Methods:
  1. DMD — linear, no lifting
  2. EDMD-poly (degree 2) — oracle polynomial dictionary
  3. EDMD-poly (degree 3) — larger oracle dictionary
  4. MLP-Koopman [3,64,32,5] — deep Koopman with MLP autoencoder
  5. 2L-KAN-stable [3,5,5] — our method

Detection metric: for each target term in {x, y, z, xy, xz}, does it appear
as a dominant term (top 5 by |coefficient|) in ANY latent dim / K-I row
with R²(h∘g) > 0.8?

Run from project root:
    python experiments/baseline_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json, time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import simulate_lorenz, PairDataset, lorenz_rhs
from models import KANAutoEncoder #, PyKANAutoEncoder, DeepKoopmanCT
from kan import KANLinear
from training import train_deep_koopman_ct, set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition

OUTDIR = Path("results/baseline_comparison")
OUTDIR.mkdir(parents=True, exist_ok=True)
DT = 0.01
N_TRAJ = 60
N_TRAIN = 48
LATENT = 5
TRUE_DICT = ["x", "y", "z", "xy", "xz"]
TRUE_SET = set(TRUE_DICT)
FALSE_TERMS = ["x^2", "y^2", "z^2", "yz"]  # known spurious terms to track
ALL_TERMS = ["x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]
SEEDS = [0, 1, 2, 3, 4]

LAMBDA_MAX = 0.906
TAU_L = 1.0 / LAMBDA_MAX
STEPS_PER_TAU = int(TAU_L / DT)
ROLLOUT_TAUS = 10
ROLLOUT_STEPS = ROLLOUT_TAUS * STEPS_PER_TAU
N_ROLLOUT_ICS = 10

EDMD_NAMES_D2 = ["x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]
EDMD_NAMES_D3 = EDMD_NAMES_D2 + [
    "x^3", "x^2y", "x^2z", "xy^2", "xyz", "xz^2", "y^3", "y^2z", "yz^2", "z^3"
]


# ── Data ──
def generate_data():
    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    return traj[:N_TRAIN].contiguous(), traj[N_TRAIN:].contiguous()


def poly_lift(X, degree=2):
    x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    feats = [x, y, z, x**2, x*y, x*z, y**2, y*z, z**2]
    if degree >= 3:
        feats += [x**3, x**2*y, x**2*z, x*y**2, x*y*z, x*z**2,
                  y**3, y**2*z, y*z**2, z**3]
    return np.column_stack(feats)


# ── Rollout helpers ──
def compute_nrmse(pred, true):
    std = true.reshape(-1, 3).std(axis=0)
    std = np.maximum(std, 1e-8)
    per_comp = np.sqrt((pred - true)**2) / std[None, :]
    return per_comp.mean(axis=1)


def rollout_horizon(nrmse_curve, threshold=0.5):
    exceed = np.where(nrmse_curve > threshold)[0]
    if len(exceed) > 0:
        return exceed[0] * DT / TAU_L
    return len(nrmse_curve) * DT / TAU_L


# ── Detection metric ──
def detect_terms_deep_koopman(model, x_data, latent_dim, top_n=5):
    """Run level-set decomposition on each latent dim.
    Return per-dim table and detection set."""
    detected = set()
    table = []
    for zi in range(latent_dim):
        try:
            dr = full_decomposition(model, x_data, zi, target_type='latent',
                                     max_degree=3, lasso_alpha=1e-5, h_degree=4)
            r2_g = dr['g_recon_r2']
            r2_hg = dr['recon_r2']
            pairs = sorted(zip(dr['inner']['active_names'],
                               dr['inner']['active_coeffs']),
                           key=lambda x: -abs(x[1]))
            # Top N non-constant terms
            top = [(n, c) for n, c in pairs if n != "1"][:top_n]
            top_names = {n for n, c in top}

            h_coeffs = dr['outer'].get('h_coeffs', None)
            h_str = ""
            if h_coeffs is not None:
                for deg, c in enumerate(h_coeffs):
                    if deg == 0: continue
                    if abs(c) > 1e-4:
                        if deg == 1: h_str += f"{c:+.2f}u "
                        else: h_str += f"{c:+.2f}u^{deg} "
            h_str = h_str.strip() or "const"

            table.append({
                "dim": zi, "r2_g": r2_g, "r2_hg": r2_hg,
                "top": top, "h": h_str,
            })

            if r2_hg > 0.8:
                detected |= top_names
        except Exception as e:
            table.append({"dim": zi, "r2_g": 0, "r2_hg": 0, "top": [], "h": f"ERR: {e}"})

    return detected, table


def detect_terms_edmd(K, feature_names):
    """Detect terms from K-I matrix thresholding."""
    K_dev = K - np.eye(K.shape[0])
    K_state = K_dev[:3, :]  # state-prediction rows
    threshold = 0.01 * np.abs(K_state).max()

    detected = set()
    for col in range(len(feature_names)):
        if np.any(np.abs(K_state[:, col]) > threshold):
            detected.add(feature_names[col])
    detected.discard("1")
    return detected


def detection_rate(detected, true=TRUE_SET):
    """For each true term, is it detected? Also track false positives."""
    results = {}
    for term in TRUE_DICT:
        results[term] = term in detected
    rate = sum(results.values()) / len(TRUE_DICT)

    # False positives: detected terms not in true dictionary
    false_positives = detected - true
    fp_results = {}
    for term in FALSE_TERMS:
        fp_results[term] = term in detected

    return results, rate, false_positives, fp_results


# ── MLP autoencoder ──
class MLPAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, X):
        Y = self.encoder(X)
        Xr = self.decoder(Y)
        return Y, Xr


# ── Method runners ──

def run_dmd(train_traj, val_traj, seed):
    set_seed(seed)
    X_t = train_traj[:, :-1, :].reshape(-1, 3).numpy()
    X_tp1 = train_traj[:, 1:, :].reshape(-1, 3).numpy()
    A = X_tp1.T @ np.linalg.pinv(X_t.T)

    # One-step MSE
    Xv_t = val_traj[:, :-1, :].reshape(-1, 3).numpy()
    Xv_tp1 = val_traj[:, 1:, :].reshape(-1, 3).numpy()
    pred = (A @ Xv_t.T).T
    mse = float(np.mean((pred - Xv_tp1)**2))

    # Rollout
    max_steps = min(ROLLOUT_STEPS, val_traj.shape[1] - 1)
    ics = np.random.choice(val_traj.shape[0], min(N_ROLLOUT_ICS, val_traj.shape[0]), replace=False)
    horizons = []
    for ic in ics:
        true = val_traj[ic, :max_steps+1].numpy()
        x = true[0].copy()
        xs = [x.copy()]
        for _ in range(max_steps):
            x = A @ x
            xs.append(x.copy())
        nrmse = compute_nrmse(np.array(xs), true)
        horizons.append(rollout_horizon(nrmse))
    horizon = float(np.mean(horizons))

    detected = {"x", "y", "z"}
    det_results, det_rate, false_positives, fp_results = detection_rate(detected)

    return {
        "mse": mse, "horizon": horizon,
        "detected": detected, "det_results": det_results, "det_rate": det_rate,
        "false_positives": false_positives, "fp_results": fp_results,
        "table": [{"dim": "A", "r2_g": 1.0, "r2_hg": 1.0,
                   "top": [("x", A[0,0]), ("y", A[0,1]), ("z", A[0,2])],
                   "h": "linear"}],
    }


def run_edmd(train_traj, val_traj, seed, degree=2):
    set_seed(seed)
    names = EDMD_NAMES_D2 if degree == 2 else EDMD_NAMES_D3
    X_t = train_traj[:, :-1, :].reshape(-1, 3).numpy()
    X_tp1 = train_traj[:, 1:, :].reshape(-1, 3).numpy()
    Z_t = poly_lift(X_t, degree)
    Z_tp1 = poly_lift(X_tp1, degree)
    K = Z_tp1.T @ np.linalg.pinv(Z_t.T)

    # One-step MSE
    Xv_t = val_traj[:, :-1, :].reshape(-1, 3).numpy()
    Xv_tp1 = val_traj[:, 1:, :].reshape(-1, 3).numpy()
    Zv_t = poly_lift(Xv_t, degree)
    pred = (K @ Zv_t.T).T[:, :3]
    mse = float(np.mean((pred - Xv_tp1)**2))

    # Rollout
    max_steps = min(ROLLOUT_STEPS, val_traj.shape[1] - 1)
    ics = np.random.choice(val_traj.shape[0], min(N_ROLLOUT_ICS, val_traj.shape[0]), replace=False)
    horizons = []
    for ic in ics:
        true = val_traj[ic, :max_steps+1].numpy()
        x = true[0].copy()
        xs = [x.copy()]
        for _ in range(max_steps):
            z = poly_lift(x.reshape(1, -1), degree).flatten()
            x = (K @ z)[:3].copy()
            xs.append(x.copy())
        nrmse = compute_nrmse(np.array(xs), true)
        horizons.append(rollout_horizon(nrmse))
    horizon = float(np.mean(horizons))

    detected = detect_terms_edmd(K, names)
    det_results, det_rate, false_positives, fp_results = detection_rate(detected)

    return {
        "mse": mse, "horizon": horizon,
        "detected": detected, "det_results": det_results, "det_rate": det_rate,
        "false_positives": false_positives, "fp_results": fp_results,
        "table": [],
    }


def run_mlp_koopman(train_traj, val_traj, seed, hidden=64, epochs=80):
    ae = MLPAutoEncoder(3, hidden, LATENT)
    traj_full = torch.cat([train_traj, val_traj], dim=0)
    model, history, tr_traj, vl_traj, dt = train_deep_koopman_ct(
        ae, traj=traj_full, latent_dim=LATENT, epochs=epochs,
        lr=1e-3, seed=seed, batch_size=2048, grad_clip=1.0,
        dt=DT, n_traj=N_TRAJ,
    )
    model = model.cpu().eval()

    # One-step MSE
    ds = PairDataset(vl_traj)
    dl = DataLoader(ds, batch_size=4096)
    total_se, n = 0.0, 0
    with torch.no_grad():
        for x_t, x_tp1 in dl:
            out = model.forward_one_step(x_t, DT)
            total_se += (out["x_tp1_pred"] - x_tp1).pow(2).sum().item()
            n += x_t.shape[0] * x_t.shape[1]
    mse = total_se / n

    # Rollout
    max_steps = min(ROLLOUT_STEPS, vl_traj.shape[1] - 1)
    ics = np.random.choice(vl_traj.shape[0], min(N_ROLLOUT_ICS, vl_traj.shape[0]), replace=False)
    horizons = []
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
            horizons.append(rollout_horizon(nrmse))
    horizon = float(np.mean(horizons))

    # Detection
    x_flat = tr_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx], dtype=torch.float32)
    detected, table = detect_terms_deep_koopman(model, x_sub, LATENT)
    det_results, det_rate, false_positives, fp_results = detection_rate(detected)

    del model, ae
    return {
        "mse": mse, "horizon": horizon,
        "detected": detected, "det_results": det_results, "det_rate": det_rate,
        "false_positives": false_positives, "fp_results": fp_results,
        "table": table,
    }


def run_kan_stable(train_traj, val_traj, seed, epochs=200):
    set_seed(seed)
    ae = KANAutoEncoder(3, 5, LATENT, grid_size=5, spline_order=3,
                         grid_range_l1=[-30, 50], grid_range_l2=[-10, 10])
    traj_full = torch.cat([train_traj, val_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    val_dl = DataLoader(PairDataset(traj_full[N_TRAIN:]), batch_size=2048)

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
            print(f"      Ep {ep}")

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
    mse = total_se / n

    # Rollout
    max_steps = min(ROLLOUT_STEPS, vl_traj.shape[1] - 1)
    ics = np.random.choice(vl_traj.shape[0], min(N_ROLLOUT_ICS, vl_traj.shape[0]), replace=False)
    horizons = []
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
            horizons.append(rollout_horizon(nrmse))
    horizon = float(np.mean(horizons))

    # Detection
    tr_traj = traj_full[:N_TRAIN].cpu()
    x_flat = tr_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    detected, table = detect_terms_deep_koopman(model, x_sub, LATENT)
    det_results, det_rate, false_positives, fp_results = detection_rate(detected)

    del model, ae
    return {
        "mse": mse, "horizon": horizon,
        "detected": detected, "det_results": det_results, "det_rate": det_rate,
        "false_positives": false_positives, "fp_results": fp_results,
        "table": table,
    }


# ── Main ──

def main():
    train_traj, val_traj = generate_data()
    print(f"Data: train={train_traj.shape}, val={val_traj.shape}")

    methods = {
        "DMD": lambda s: run_dmd(train_traj, val_traj, s),
        "EDMD-d2": lambda s: run_edmd(train_traj, val_traj, s, degree=2),
        "EDMD-d3": lambda s: run_edmd(train_traj, val_traj, s, degree=3),
        "MLP-Koopman": lambda s: run_mlp_koopman(train_traj, val_traj, s),
        "KAN-stable": lambda s: run_kan_stable(train_traj, val_traj, s),
    }

    all_results = {}

    for name, run_fn in methods.items():
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        seed_results = []
        for seed in SEEDS:
            print(f"\n    seed={seed}")
            t0 = time.time()
            r = run_fn(seed)
            elapsed = time.time() - t0
            r["wall_time"] = elapsed
            seed_results.append(r)

            n_fp = len(r['false_positives'])
            print(f"    MSE={r['mse']:.4e}  Horizon={r['horizon']:.2f} tau_L  "
                  f"Detection={r['det_rate']:.0%}  FP={n_fp}  ({elapsed:.0f}s)")
            print(f"    Detected: {sorted(r['detected'])}")
            for term in TRUE_DICT:
                mark = "✓" if r['det_results'][term] else "✗"
                print(f"      {mark} {term}")
            if r['false_positives']:
                print(f"    False positives: {sorted(r['false_positives'])}")
                for term in FALSE_TERMS:
                    if r['fp_results'][term]:
                        print(f"      ✗ FP: {term}")

            # Print decomposition table for first seed
            if seed == 0 and r['table']:
                print(f"\n    Decomposition table (seed=0):")
                for row in r['table']:
                    top_str = '  '.join(f'{c:+.4f}*{n}' for n, c in row.get('top', [])[:5])
                    print(f"      z[{row['dim']}] R²(g)={row['r2_g']:.3f} "
                          f"R²(h∘g)={row['r2_hg']:.3f}  h={row['h']}")
                    print(f"        g = {top_str}")

        all_results[name] = seed_results

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*90}")
    print("  SUMMARY: DETECTION RATE OF DICTIONARY TERMS")
    print(f"{'='*90}")

    print(f"\n  {'Method':<15} {'MSE':>12} {'Horizon':>10} {'Det%':>6} ", end="")
    for t in TRUE_DICT:
        print(f" {t:>4}", end="")
    print("  |", end="")
    for t in FALSE_TERMS:
        print(f" {t:>4}", end="")
    print(f"  {'#FP':>4}")
    print("  " + "-" * 95)

    for name in methods:
        sr = all_results[name]
        mses = [r['mse'] for r in sr]
        hors = [r['horizon'] for r in sr]
        rates = [r['det_rate'] for r in sr]

        # Per-term detection rate across seeds
        term_rates = {}
        for t in TRUE_DICT:
            term_rates[t] = np.mean([r['det_results'][t] for r in sr])

        # False positive rates
        fp_rates = {}
        for t in FALSE_TERMS:
            fp_rates[t] = np.mean([r['fp_results'][t] for r in sr])
        avg_fp = np.mean([len(r['false_positives']) for r in sr])

        mse_str = f"{np.mean(mses):.2e}" if np.std(mses) < 1e-10 else f"{np.mean(mses):.2e}±{np.std(mses):.0e}"
        hor_str = f"{np.mean(hors):.2f}" if np.std(hors) < 0.01 else f"{np.mean(hors):.2f}±{np.std(hors):.1f}"

        print(f"  {name:<15} {mse_str:>12} {hor_str:>10} {np.mean(rates):>5.0%} ", end="")
        for t in TRUE_DICT:
            r = term_rates[t]
            mark = f"{r:.0%}" if r < 1.0 else "  ✓ "
            print(f" {mark:>4}", end="")
        print("  |", end="")
        for t in FALSE_TERMS:
            r = fp_rates[t]
            mark = f"{r:.0%}" if r > 0.0 else "  · "
            print(f" {mark:>4}", end="")
        print(f"  {avg_fp:>4.1f}")

    # Save
    save_data = {}
    for name in methods:
        sr = all_results[name]
        save_data[name] = {
            "mse_mean": float(np.mean([r['mse'] for r in sr])),
            "mse_std": float(np.std([r['mse'] for r in sr])),
            "horizon_mean": float(np.mean([r['horizon'] for r in sr])),
            "horizon_std": float(np.std([r['horizon'] for r in sr])),
            "detection_rate": float(np.mean([r['det_rate'] for r in sr])),
            "per_term": {t: float(np.mean([r['det_results'][t] for r in sr]))
                         for t in TRUE_DICT},
            "per_fp": {t: float(np.mean([r['fp_results'][t] for r in sr]))
                       for t in FALSE_TERMS},
            "avg_false_positives": float(np.mean([len(r['false_positives']) for r in sr])),
            "seeds": [{
                "mse": r['mse'], "horizon": r['horizon'],
                "detected": sorted(r['detected']),
                "false_positives": sorted(r['false_positives']),
                "det_rate": r['det_rate'],
            } for r in sr],
        }

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR / 'results.json'}")

    # ══════════════════════════════════════════════════════════════
    # Detection heatmap
    # ══════════════════════════════════════════════════════════════
    method_names = list(methods.keys())

    # Heatmap with ALL terms: true dict + false positive candidates
    fig, ax = plt.subplots(figsize=(12, 4.5))
    data = np.zeros((len(method_names), len(ALL_TERMS)))
    for i, name in enumerate(method_names):
        for j, term in enumerate(ALL_TERMS):
            if term in TRUE_DICT:
                data[i, j] = save_data[name]["per_term"][term]
            elif term in FALSE_TERMS:
                data[i, j] = save_data[name]["per_fp"][term]

    # Custom colormap: green for true positives, red for false positives
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ALL_TERMS)))
    ax.set_xticklabels(ALL_TERMS, fontsize=11)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=10)

    for i in range(len(method_names)):
        for j in range(len(ALL_TERMS)):
            v = data[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if v < 0.5 else "black")

    # Mark true terms vs false positive terms
    for j, term in enumerate(ALL_TERMS):
        if term in TRUE_SET:
            ax.text(j, -0.7, "TRUE", ha="center", va="center",
                    fontsize=7, color="green", fontweight="bold")
        else:
            ax.text(j, -0.7, "FP", ha="center", va="center",
                    fontsize=7, color="red", fontweight="bold")

    plt.colorbar(im, label="Detection rate across 5 seeds")
    ax.set_title("Dictionary Term Detection: True Positives vs False Positives", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "detection_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTDIR / "detection_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved detection_heatmap.png/pdf")


if __name__ == "__main__":
    main()
