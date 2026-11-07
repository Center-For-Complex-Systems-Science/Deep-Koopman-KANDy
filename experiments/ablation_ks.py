"""
Controlled ablation study for Deep-Koopman-KANDy on the KS equation.

Same 7 variants as ablation_lorenz.py, adapted for the 64D PDE state.

Variants:
  A: Full Deep-Koopman-KANDy  (2L-KAN [64,32,128] + stable)
  B: 1-layer KAN [64,128] + stable
  C: Parameter-matched MLP [64,32,128] + stable
  D: KAN encoder + MLP decoder + stable
  E: MLP encoder + KAN decoder + stable
  F: Full KAN + unconstrained propagator
  G: Full model, prediction only (no decomposition)

Metrics:
  - One-step MSE
  - Prediction horizon (time units until NRMSE > 1.0)
  - Median R²(h∘g) across latent dims
  - Diagonal fraction of quadratic coupling matrix

Data: reuses results/ks_stable/ks_data.pt (30 trajectories, 1000 steps, 64 grid points)
3 seeds (KS training is expensive).

Run from project root:
    python experiments/ablation_ks.py
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

from models import KANAutoEncoder, KANEncoder, KANDecoder, DeepKoopmanCT
from kan import KANLinear
from data import PairDataset
from training import set_seed
from manifold_analysis import full_decomposition_nd

OUTDIR = Path("results/ablation_ks")
OUTDIR.mkdir(parents=True, exist_ok=True)

# KS parameters (match ks_stable.py)
N_GRID = 64
DT_SAVE = 0.25
LAMBDA_MAX_KS = 0.043
TAU_L = 1.0 / LAMBDA_MAX_KS  # ~23.3 time units
N_TRAJ = 30
N_TRAIN = 24

# Architecture
INPUT_SIZE = N_GRID       # 64
MID_SIZE = 32
LATENT_SIZE = 128

# Training
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 512
GRAD_CLIP = 1.0
GAMMA_PREDX = 1.0
ALPHA_LATENT = 1.0
BETA_RECON = 0.5

SEEDS = [0, 1, 2]

# Decomposition: only do top N_DECOMP dims (128 is too expensive for ablation)
N_DECOMP = 32

KAN_KWARGS = dict(grid_size=5, spline_order=3,
                  grid_range_l1=[-5, 5], grid_range_l2=[-10, 10])


# ── Stable Koopman (copied from ks_stable.py to avoid import issues) ──

class DeepKoopmanCTStable(nn.Module):
    def __init__(self, autoencoder, latent_dim):
        super().__init__()
        self.ae = autoencoder
        self.latent_dim = latent_dim
        n = latent_dim
        self.skew_params = nn.Parameter(torch.randn(n * (n - 1) // 2) * 0.01)
        self.L_diag = nn.Parameter(torch.ones(n) * 0.1)
        self.L_lower = nn.Parameter(torch.randn(n * (n - 1) // 2) * 0.01)

    def _get_skew(self):
        n = self.latent_dim
        K = torch.zeros(n, n, device=self.skew_params.device)
        idx = torch.triu_indices(n, n, offset=1)
        K[idx[0], idx[1]] = self.skew_params
        K = K - K.T
        return K

    def _get_L(self):
        n = self.latent_dim
        L = torch.zeros(n, n, device=self.L_diag.device)
        L[range(n), range(n)] = self.L_diag
        idx = torch.tril_indices(n, n, offset=-1)
        L[idx[0], idx[1]] = self.L_lower
        return L

    def generator_matrix(self):
        K = self._get_skew()
        L = self._get_L()
        return K - L.T @ L

    def expm_propagate(self, z_t, dt):
        A = self.generator_matrix()
        M = torch.matrix_exp(A * dt)
        return z_t @ M.T

    def forward_one_step(self, x_t, dt):
        z_t, x_t_rec = self.ae(x_t)
        z_tp1_pred = self.expm_propagate(z_t, dt)
        x_tp1_pred = self.ae.decoder(z_tp1_pred)
        return {
            "z_t": z_t, "x_t_rec": x_t_rec,
            "z_tp1_pred": z_tp1_pred, "x_tp1_pred": x_tp1_pred,
        }


# ── Autoencoder variants ──

class SingleLayerKANAutoEncoder(nn.Module):
    """Variant B: 1-layer KAN."""
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.encoder = KANLinear(input_size, latent_size,
                                  grid_size=5, spline_order=3, grid_range=[-5, 5])
        self.decoder = KANLinear(latent_size, input_size,
                                  grid_size=5, spline_order=3, grid_range=[-10, 10])

    def forward(self, X):
        return self.encoder(X), self.decoder(self.encoder(X))


class MLPAutoEncoder(nn.Module):
    """Variant C: MLP."""
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
        return Y, self.decoder(Y)


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
        return Y, self.decoder(Y)


class HybridMLPEncKANDec(nn.Module):
    """Variant E: MLP encoder + KAN decoder."""
    def __init__(self, input_size, mid_size, latent_size):
        super().__init__()
        self.encoder = MLPEncoder(input_size, mid_size, latent_size)
        self.decoder = KANDecoder(latent_size, mid_size, input_size, **KAN_KWARGS)

    def forward(self, X):
        Y = self.encoder(X)
        return Y, self.decoder(Y)


# ── Training ──

def train_model(ae, traj, seed, prop_type="stable", epochs=EPOCHS):
    set_seed(seed)
    train_dl = DataLoader(PairDataset(traj[:N_TRAIN]),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    if prop_type == "stable":
        model = DeepKoopmanCTStable(ae, LATENT_SIZE)
    else:
        model = DeepKoopmanCT(ae, LATENT_SIZE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        for x_t, x_tp1 in train_dl:
            out = model.forward_one_step(x_t, DT_SAVE)
            z_true = model.ae.encoder(x_tp1)
            loss = (GAMMA_PREDX * mse_fn(out["x_tp1_pred"], x_tp1)
                    + ALPHA_LATENT * mse_fn(out["z_tp1_pred"], z_true)
                    + BETA_RECON * mse_fn(out["x_t_rec"], x_t))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sched.step()
        if ep % 25 == 0:
            print(f"        Ep {ep}", flush=True)

    model.eval()
    return model


# ── Evaluation ──

def compute_nrmse(pred, true):
    std = true.reshape(-1, N_GRID).std(axis=0).mean()
    std = max(std, 1e-8)
    return np.sqrt(np.mean((pred - true)**2, axis=1)) / std


def eval_prediction(model, val_traj):
    """One-step MSE and prediction horizon in time units."""
    ds = PairDataset(val_traj)
    dl = DataLoader(ds, batch_size=2048)
    total_se, n = 0.0, 0
    with torch.no_grad():
        for x_t, x_tp1 in dl:
            out = model.forward_one_step(x_t, DT_SAVE)
            total_se += (out["x_tp1_pred"] - x_tp1).pow(2).sum().item()
            n += x_t.shape[0] * x_t.shape[1]
    one_step_mse = total_se / n

    # Rollout horizon: NRMSE > 1.0
    n_val = val_traj.shape[0]
    max_steps = min(400, val_traj.shape[1] - 1)  # up to 100 time units
    horizons = []
    with torch.no_grad():
        for ic in range(n_val):
            true = val_traj[ic, :max_steps+1].numpy()
            x = val_traj[ic, 0]
            xs = [x.numpy()]
            for _ in range(max_steps):
                out = model.forward_one_step(x.unsqueeze(0), DT_SAVE)
                x = out["x_tp1_pred"].squeeze(0)
                xs.append(x.numpy())
            pred = np.array(xs)
            nrmse = compute_nrmse(pred, true)
            exceed = np.where(nrmse > 1.0)[0]
            if len(exceed) > 0:
                horizons.append(exceed[0] * DT_SAVE)
            else:
                horizons.append(max_steps * DT_SAVE)

    return one_step_mse, float(np.mean(horizons))


def eval_symbolic(model, train_traj, n_decomp=N_DECOMP):
    """Decomposition → R²(h∘g) and diagonal fraction of quadratic coupling."""
    flat = train_traj.reshape(-1, N_GRID)
    n_sample = min(3000, flat.shape[0])
    idx = torch.randperm(flat.shape[0])[:n_sample]
    x_data = flat[idx]
    var_names = [f'u_{j}' for j in range(N_GRID)]

    model_cpu = model.cpu().eval()

    r2_hgs = []
    diag_fracs = []

    for ti in range(n_decomp):
        try:
            dr = full_decomposition_nd(
                model_cpu, x_data, ti, var_names,
                target_type='latent', max_degree=2,
                lasso_alpha=1e-4, h_degree=4,
            )
            r2 = dr.get('recon_r2', 0.0)
            if np.isnan(r2):
                r2 = 0.0
            r2_hgs.append(r2)

            # Diagonal fraction of quadratic coupling
            inner = dr.get('inner', {})
            all_names = inner.get('names', [])
            all_coeffs = inner.get('coeffs', np.zeros(1))
            all_powers = inner.get('powers', [])

            quad_matrix = np.zeros((N_GRID, N_GRID))
            for name, coeff, pw in zip(all_names, all_coeffs, all_powers):
                if abs(coeff) < 1e-10:
                    continue
                total_deg = sum(pw)
                if total_deg == 2:
                    nonzero = [(j, p) for j, p in enumerate(pw) if p > 0]
                    if len(nonzero) == 1:
                        j, p = nonzero[0]
                        quad_matrix[j, j] = coeff
                    elif len(nonzero) == 2:
                        j1, _ = nonzero[0]
                        j2, _ = nonzero[1]
                        quad_matrix[j1, j2] = abs(coeff)
                        quad_matrix[j2, j1] = abs(coeff)

            total_energy = np.abs(quad_matrix).sum()
            if total_energy > 1e-12:
                diag_energy = 0.0
                for offset in [-2, -1, 0, 1, 2]:
                    diag_energy += np.abs(np.diag(quad_matrix, offset)).sum()
                diag_fracs.append(diag_energy / total_energy)
            else:
                diag_fracs.append(0.0)

        except Exception as e:
            r2_hgs.append(0.0)
            diag_fracs.append(0.0)

        if (ti + 1) % 8 == 0:
            print(f"        decomposed {ti+1}/{n_decomp}", flush=True)

    return {
        "median_r2_hg": float(np.median(r2_hgs)),
        "mean_r2_hg": float(np.mean(r2_hgs)),
        "median_diag_frac": float(np.median(diag_fracs)),
        "mean_diag_frac": float(np.mean(diag_fracs)),
        "n_good": int(sum(1 for r in r2_hgs if r > 0.5)),
    }


# ── Variant definitions ──

def make_A(seed):
    return KANAutoEncoder(INPUT_SIZE, MID_SIZE, LATENT_SIZE, **KAN_KWARGS), "stable"

def make_B(seed):
    return SingleLayerKANAutoEncoder(INPUT_SIZE, LATENT_SIZE), "stable"

def make_C(seed):
    return MLPAutoEncoder(INPUT_SIZE, MID_SIZE, LATENT_SIZE), "stable"

def make_D(seed):
    return HybridKANEncMLPDec(INPUT_SIZE, MID_SIZE, LATENT_SIZE), "stable"

def make_E(seed):
    return HybridMLPEncKANDec(INPUT_SIZE, MID_SIZE, LATENT_SIZE), "stable"

def make_F(seed):
    return KANAutoEncoder(INPUT_SIZE, MID_SIZE, LATENT_SIZE, **KAN_KWARGS), "unconstrained"

def make_G(seed):
    return KANAutoEncoder(INPUT_SIZE, MID_SIZE, LATENT_SIZE, **KAN_KWARGS), "stable"


VARIANTS = {
    "A: 2L-KAN + stable":        (make_A, True),
    "B: 1L-KAN + stable":        (make_B, True),
    "C: MLP + stable":           (make_C, True),
    "D: KAN-enc + MLP-dec":      (make_D, True),
    "E: MLP-enc + KAN-dec":      (make_E, True),
    "F: 2L-KAN + unconstrained": (make_F, True),
    "G: Full (pred only)":       (make_G, False),
}


# ── Main ──

def main():
    t_start = time.time()
    print("=" * 80)
    print("  ABLATION STUDY: Deep-Koopman-KANDy on KS Equation")
    print("=" * 80)

    # Load pre-simulated data
    data_path = Path("results/ks_stable/ks_data.pt")
    if data_path.exists():
        traj = torch.load(data_path, map_location="cpu", weights_only=False)
        print(f"  Loaded KS data: {traj.shape}")
    else:
        # Need to simulate — import from ks_stable
        print("  KS data not found, simulating...")
        sys.path.insert(0, "experiments")
        from ks_stable import simulate_ks
        set_seed(42)
        traj = simulate_ks(n_traj=N_TRAJ, L=22.0, N=N_GRID,
                           dt_sim=0.005, dt_save=DT_SAVE,
                           burn_in=100.0, T=250.0, seed=42)
        torch.save(traj, data_path)

    train_traj = traj[:N_TRAIN].contiguous()
    val_traj = traj[N_TRAIN:].contiguous()
    print(f"  Train: {train_traj.shape}, Val: {val_traj.shape}")
    print(f"  Seeds: {SEEDS}, Decomp dims: {N_DECOMP}")

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

            model = train_model(ae, traj, seed, prop_type)

            # Prediction
            mse, horizon = eval_prediction(model, val_traj)
            print(f"    MSE={mse:.4e}  Horizon={horizon:.1f} t.u.  params={n_params}")

            # Symbolic
            if do_symbolic:
                sym = eval_symbolic(model, train_traj)
                print(f"    R²(h∘g)={sym['median_r2_hg']:.3f}  "
                      f"diag_frac={sym['median_diag_frac']:.3f}  "
                      f"good={sym['n_good']}/{N_DECOMP}")
            else:
                sym = {"median_r2_hg": None, "mean_r2_hg": None,
                       "median_diag_frac": None, "mean_diag_frac": None,
                       "n_good": None}

            elapsed = time.time() - t0
            result = {"mse": mse, "horizon": horizon,
                      "n_params": n_params, "wall_time": elapsed, **sym}
            seed_results.append(result)
            print(f"    ({elapsed:.0f}s)")

            del model, ae
            gc.collect()

        all_results[vname] = seed_results

    # ── Summary ──
    print(f"\n\n{'='*90}")
    print("  TABLE: KS Ablation Results")
    print(f"{'='*90}")
    print(f"  {'Variant':<28} {'Params':>7} {'MSE':>12} {'Horizon':>12} "
          f"{'R²(h∘g)':>9} {'DiagFrac':>9} {'#good':>6}")
    print("  " + "-"*85)

    for vname in VARIANTS:
        sr = all_results[vname]
        mses = [r['mse'] for r in sr]
        hors = [r['horizon'] for r in sr]
        params = sr[0]['n_params']

        mse_str = f"{np.mean(mses):.2e}±{np.std(mses):.0e}"
        hor_str = f"{np.mean(hors):.1f}±{np.std(hors):.1f}"

        if sr[0]['median_r2_hg'] is not None:
            r2s = [r['median_r2_hg'] for r in sr]
            dfs = [r['median_diag_frac'] for r in sr]
            goods = [r['n_good'] for r in sr]
            r2_str = f"{np.mean(r2s):.3f}"
            df_str = f"{np.mean(dfs):.3f}"
            g_str = f"{np.mean(goods):.0f}/{N_DECOMP}"
        else:
            r2_str = df_str = g_str = "--"

        print(f"  {vname:<28} {params:>7} {mse_str:>12} {hor_str:>12} "
              f"{r2_str:>9} {df_str:>9} {g_str:>6}")

    # Save
    save_data = {}
    for vname in VARIANTS:
        sr = all_results[vname]
        save_data[vname] = {
            "n_params": sr[0]['n_params'],
            "mse_mean": float(np.mean([r['mse'] for r in sr])),
            "mse_std": float(np.std([r['mse'] for r in sr])),
            "horizon_mean": float(np.mean([r['horizon'] for r in sr])),
            "horizon_std": float(np.std([r['horizon'] for r in sr])),
            "seeds": [dict(r) for r in sr],
        }

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR / 'results.json'}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/3600:.1f} hours")


if __name__ == "__main__":
    main()
