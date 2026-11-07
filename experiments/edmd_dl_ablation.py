"""
EDMD with Dictionary Learning (Li, Dietrich, Bollt & Kevrekidis 2017).

Structural differences from MLP+Stable:
  - No decoder: state reconstruction is a fixed linear projection
  - K is recomputed as closed-form EDMD least-squares at each step
  - Loss is only the linear-residual ||psi(x_{t+1}) - K psi(x_t)||^2
  - No stability constraint on K
  - Dictionary includes raw state by construction: psi(x) = [x, f(x)]

After training, PySR is run on the learned dictionary components f_k(x)
to extract symbolic expressions, scored against {x, y, z, xy, xz}.

Same data (seed 42), same model seeds [0,1,2,3,4] as the external ablation.

Run from project root:
    python experiments/edmd_dl_ablation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import csv, json, time, logging
from pathlib import Path

import pysr  # noqa: F401 — before torch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import simulate_lorenz, lorenz_rhs
from training import set_seed

OUTDIR = Path("results/edmd_dl_ablation")
OUTDIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
N_TRAJ = 60
N_TRAIN = 48
STATE_DIM = 3
LIFT_DIM = 5          # learned dictionary components (on top of raw state)
DICT_DIM = STATE_DIM + LIFT_DIM  # total dictionary size = 8
SEEDS = [0, 1, 2, 3, 4]
BURN_IN = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 200          # match Deep-Koopman-KANDy budget
LR = 1e-3
BATCH_SIZE = 2048
K_REFRESH_EVERY = 1   # recompute K every epoch (faithful to Li et al.)

LAMBDA_MAX = 0.906
TAU_L = 1.0 / LAMBDA_MAX
STEPS_PER_TAU = int(TAU_L / DT)
N_ROLLOUT_ICS = 10

TRUE_DICT = ["x", "y", "z", "xy", "xz"]
TRUE_SET = set(TRUE_DICT)
COEFF_THRESHOLD = 1e-2

PYSR_OPERATORS = ["+", "-", "*"]
PYSR_MAX_COMPLEXITY = 20
PYSR_POPULATIONS = 40
PYSR_ITERATIONS = 200
PYSR_SEEDS_PER_LATENT = 5
PYSR_R2_THRESHOLD = 0.95

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# EDMD-DL Model
# ═══════════════════════════════════════════════════════════════════════════

class DictionaryNetwork(nn.Module):
    """Learned dictionary: psi(x) = [x, f(x)] where f is an MLP.

    The raw state x is always included by construction, so reconstruction
    is a fixed linear projection (first STATE_DIM components).
    """

    def __init__(self, state_dim, lift_dim, hidden=64):
        super().__init__()
        self.state_dim = state_dim
        self.lift_dim = lift_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, lift_dim),
        )

    def forward(self, x):
        """Returns psi(x) = [x, f(x)], shape (batch, state_dim + lift_dim)."""
        f = self.net(x)
        return torch.cat([x, f], dim=-1)

    def lift_only(self, x):
        """Returns just the learned components f(x), shape (batch, lift_dim)."""
        return self.net(x)


class EDMDDL(nn.Module):
    """EDMD with Dictionary Learning.

    Training alternates:
      1. Gradient step on dictionary network parameters (minimize ||psi(x') - K psi(x)||^2)
      2. Refresh K as closed-form least-squares: K = Psi_Y @ pinv(Psi_X)

    No decoder. No stability constraint. K is always optimal given psi.
    """

    def __init__(self, state_dim, lift_dim, hidden=64):
        super().__init__()
        self.dict_net = DictionaryNetwork(state_dim, lift_dim, hidden)
        self.dict_dim = state_dim + lift_dim
        # K is a buffer, not a parameter — updated analytically
        self.register_buffer('K', torch.eye(self.dict_dim))

    def psi(self, x):
        return self.dict_net(x)

    def predict_psi(self, x):
        """Predict psi(x_{t+1}) = K @ psi(x_t)."""
        psi_t = self.psi(x)
        return psi_t @ self.K.T

    def predict_state(self, x):
        """Predict x_{t+1} from x_t: first state_dim components of K @ psi(x_t)."""
        psi_next = self.predict_psi(x)
        return psi_next[:, :self.dict_net.state_dim]

    @torch.no_grad()
    def refresh_K(self, x_t, x_tp1):
        """Recompute K as closed-form EDMD solution."""
        psi_t = self.psi(x_t)      # (N, dict_dim)
        psi_tp1 = self.psi(x_tp1)  # (N, dict_dim)
        # K = Psi_Y^T @ pinv(Psi_X^T) = (Psi_X^T @ Psi_X)^{-1} @ Psi_X^T @ Psi_Y
        # Using pseudoinverse for numerical stability
        self.K.copy_(psi_tp1.T @ torch.linalg.pinv(psi_t.T))


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def compute_nrmse(pred, true):
    std = true.reshape(-1, 3).std(axis=0)
    std = np.maximum(std, 1e-8)
    return (np.sqrt((pred - true)**2) / std[None, :]).mean(axis=1)


def eval_prediction(model, test_traj):
    """One-step MSE and NRMSE@2tau."""
    x_t = test_traj[:, :-1, :].reshape(-1, 3)
    x_tp1 = test_traj[:, 1:, :].reshape(-1, 3)

    model.eval()
    with torch.no_grad():
        x_pred = model.predict_state(x_t)
    mse = float((x_pred - x_tp1).pow(2).mean().item())

    steps_2tau = 2 * STEPS_PER_TAU
    max_steps = min(steps_2tau, test_traj.shape[1] - 1)
    ics = np.random.choice(test_traj.shape[0], min(N_ROLLOUT_ICS, test_traj.shape[0]), replace=False)
    rollout_errors = []
    with torch.no_grad():
        for ic in ics:
            true = test_traj[ic, :max_steps+1].numpy()
            x = test_traj[ic, 0].clone()
            xs = [x.numpy().copy()]
            for _ in range(max_steps):
                x = model.predict_state(x.unsqueeze(0)).squeeze(0)
                xs.append(x.numpy().copy())
            pred = np.array(xs)
            nrmse = compute_nrmse(pred, true)
            rollout_errors.append(float(nrmse[min(steps_2tau, len(nrmse)-1)]))
    return mse, float(np.mean(rollout_errors))


def score_dictionary(detected_monomials):
    detected = set(detected_monomials)
    tp = len(detected & TRUE_SET)
    precision = tp / max(len(detected), 1)
    recall = tp / len(TRUE_SET)
    jaccard = tp / max(len(detected | TRUE_SET), 1)
    return {
        "jaccard": jaccard, "precision": precision, "recall": recall,
        "detected": sorted(detected), "n_detected": len(detected),
        "n_tp": tp, "n_fp": len(detected - TRUE_SET),
    }


def extract_monomials_from_sympy(expr):
    import sympy as sp
    canonical_map = {
        "x0": "x", "x1": "y", "x2": "z",
        "x0**2": "x^2", "x0*x1": "xy", "x0*x2": "xz",
        "x1**2": "y^2", "x1*x2": "yz", "x2**2": "z^2",
        "x1*x0": "xy", "x2*x0": "xz", "x2*x1": "yz",
    }
    if expr is None:
        return set()
    expanded = sp.expand(expr)
    monomials = set()
    terms = expanded.as_ordered_terms() if expanded.is_Add else [expanded]
    for term in terms:
        coeff, mono = term.as_coeff_Mul()
        if abs(float(coeff)) < COEFF_THRESHOLD:
            continue
        mono_str = str(mono)
        if mono_str == "1" or mono == sp.S.One:
            continue
        canonical = canonical_map.get(mono_str, mono_str)
        if canonical in {"x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2",
                          "x^3", "x^2y", "x^2z", "xy^2", "xyz", "xz^2",
                          "y^3", "y^2z", "yz^2", "z^3"}:
            monomials.add(canonical)
    return monomials


def run_pysr_on_lift(model, x_train, x_test, method_name="", seed=0):
    """Run PySR on each LEARNED dictionary component f_k(x).

    Note: we only run PySR on the learned components f_k, not on the raw
    state components (which are x, y, z by construction — trivially recovered).
    """
    from pysr import PySRRegressor

    with torch.no_grad():
        f_train = model.dict_net.lift_only(x_train).numpy()
        f_test = model.dict_net.lift_only(x_test).numpy()
    x_train_np = x_train.numpy()
    x_test_np = x_test.numpy()

    # Raw state {x, y, z} is always in the dictionary by construction
    detected = {"x", "y", "z"}
    pysr_log = []

    for k in range(LIFT_DIM):
        log.info(f"    PySR on f[{k}] ({method_name})")
        target_train = f_train[:, k]
        target_test = f_test[:, k]
        t_mean, t_std = target_train.mean(), max(target_train.std(), 1e-12)
        target_train_s = (target_train - t_mean) / t_std
        target_test_s = (target_test - t_mean) / t_std

        best_expr, best_loss, best_r2, best_complexity = None, float("inf"), -999, 999
        all_results = []

        for ps in range(PYSR_SEEDS_PER_LATENT):
            pysr_seed = seed * 1000 + k * 100 + ps
            pmodel = PySRRegressor(
                binary_operators=PYSR_OPERATORS, unary_operators=[],
                maxsize=PYSR_MAX_COMPLEXITY, populations=PYSR_POPULATIONS,
                niterations=PYSR_ITERATIONS, random_state=pysr_seed,
                parallelism="serial", progress=False, verbosity=0,
                temp_equation_file=True,
            )
            pmodel.fit(x_train_np, target_train_s, variable_names=["x0", "x1", "x2"])

            eqs = pmodel.equations_
            if eqs is not None and len(eqs) > 0:
                for _, row in eqs.iterrows():
                    try:
                        pred_test = pmodel.predict(x_test_np, index=int(row.name))
                        ss_res = np.sum((target_test_s - pred_test)**2)
                        ss_tot = np.sum((target_test_s - target_test_s.mean())**2)
                        r2 = 1 - ss_res / max(ss_tot, 1e-12)
                        complexity = int(row['complexity'])
                        loss = float(row['loss'])
                        all_results.append({
                            "pysr_seed": ps, "complexity": complexity,
                            "loss": loss, "r2_test": r2,
                            "equation": str(row['equation']),
                        })
                        if r2 >= PYSR_R2_THRESHOLD:
                            if complexity < best_complexity or (complexity == best_complexity and loss < best_loss):
                                best_expr = row.get('sympy_format', None)
                                best_loss, best_r2, best_complexity = loss, r2, complexity
                        elif best_r2 < PYSR_R2_THRESHOLD:
                            if loss < best_loss:
                                best_expr = row.get('sympy_format', None)
                                best_loss, best_r2, best_complexity = loss, r2, complexity
                    except Exception:
                        pass

        k_monomials = extract_monomials_from_sympy(best_expr)
        detected |= k_monomials
        pysr_log.append({
            "latent_dim": k, "best_expr": str(best_expr),
            "best_r2": best_r2, "best_complexity": best_complexity,
            "monomials": sorted(k_monomials),
        })
        log.info(f"      best: {best_expr} (R²={best_r2:.3f}, monomials={sorted(k_monomials)})")

    return detected, pysr_log


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("  EDMD-DL Ablation (Li, Dietrich, Bollt & Kevrekidis 2017)")
    log.info(f"  Device: {DEVICE}")
    log.info(f"  Dictionary: [x, y, z, f_1(x)..f_{LIFT_DIM}(x)] = {DICT_DIM}D")
    log.info("=" * 70)

    # Data (same as external ablation)
    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT, burn_in=BURN_IN, device="cpu")
    train_traj = traj[:N_TRAIN].contiguous()
    test_traj = traj[N_TRAIN:].contiguous()

    # Build (x_t, x_{t+1}) pairs
    x_t_all = train_traj[:, :-1, :].reshape(-1, 3)
    x_tp1_all = train_traj[:, 1:, :].reshape(-1, 3)

    csv_path = OUTDIR / "results.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "method", "seed", "one_step_mse", "nrmse_2tau",
        "jaccard", "precision", "recall", "n_detected", "n_tp", "n_fp",
        "detected", "n_params", "train_epochs",
    ])
    csv_writer.writeheader()

    jsonl_path = OUTDIR / "expressions.jsonl"
    jsonl_file = open(jsonl_path, "w")

    all_results = []

    for seed in SEEDS:
        log.info(f"\n{'='*60}")
        log.info(f"  SEED {seed}")
        log.info(f"{'='*60}")
        t0 = time.time()

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = EDMDDL(STATE_DIM, LIFT_DIM, hidden=64).to(DEVICE)
        n_params = sum(p.numel() for p in model.dict_net.parameters())
        log.info(f"  Dictionary network params: {n_params}")

        train_ds = TensorDataset(x_t_all, x_tp1_all)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        # Only optimize dictionary network parameters (K is refreshed analytically)
        opt = torch.optim.AdamW(model.dict_net.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        for ep in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_t_batch, x_tp1_batch in train_dl:
                x_t_batch = x_t_batch.to(DEVICE)
                x_tp1_batch = x_tp1_batch.to(DEVICE)

                # Forward: compute dictionary values
                psi_t = model.psi(x_t_batch)
                psi_tp1 = model.psi(x_tp1_batch)

                # Predict using current K
                psi_tp1_pred = psi_t @ model.K.T

                # Loss: ||psi(x') - K psi(x)||^2
                loss = (psi_tp1 - psi_tp1_pred).pow(2).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.dict_net.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            sched.step()

            # Refresh K analytically using a large sample
            if ep % K_REFRESH_EVERY == 0:
                sample_idx = np.random.choice(len(x_t_all), min(50000, len(x_t_all)), replace=False)
                x_t_sample = x_t_all[sample_idx].to(DEVICE)
                x_tp1_sample = x_tp1_all[sample_idx].to(DEVICE)
                model.refresh_K(x_t_sample, x_tp1_sample)

            if ep % 50 == 0:
                log.info(f"    Ep {ep}: loss={epoch_loss/n_batches:.6f}")

        model = model.cpu().eval()

        # Final K refresh on full training data
        model.refresh_K(x_t_all, x_tp1_all)

        # Prediction metrics
        mse, nrmse_2tau = eval_prediction(model, test_traj)
        log.info(f"  MSE={mse:.4e} NRMSE@2τ={nrmse_2tau:.3f}")

        # PySR on learned dictionary components
        x_test_flat = test_traj.reshape(-1, 3)
        tr_idx = np.random.choice(len(x_t_all), min(5000, len(x_t_all)), replace=False)
        te_idx = np.random.choice(len(x_test_flat), min(2000, len(x_test_flat)), replace=False)

        detected, pysr_log = run_pysr_on_lift(
            model, x_t_all[tr_idx], x_test_flat[te_idx],
            method_name="EDMD-DL", seed=seed,
        )

        scores = score_dictionary(detected)
        elapsed = time.time() - t0

        log.info(f"  Jaccard={scores['jaccard']:.3f} Rec={scores['recall']:.0%} "
                 f"detected={scores['detected']} ({elapsed:.0f}s)")

        # K spectrum
        eigs = np.linalg.eigvals(model.K.numpy())
        log.info(f"  K spectrum: |λ| range=[{np.abs(eigs).min():.4f}, {np.abs(eigs).max():.4f}]")

        result = {
            "method": "EDMD-DL+PySR", "seed": seed,
            "one_step_mse": mse, "nrmse_2tau": nrmse_2tau,
            "n_params": n_params, "train_epochs": EPOCHS,
            **scores,
        }
        all_results.append(result)

        csv_writer.writerow(result)
        csv_file.flush()
        json.dump({"method": "EDMD-DL+PySR", "seed": seed, "pysr_log": pysr_log},
                   jsonl_file, default=str)
        jsonl_file.write("\n")

        del model
        torch.cuda.empty_cache()

    csv_file.close()
    jsonl_file.close()

    # Summary
    log.info(f"\n{'='*70}")
    log.info("  SUMMARY")
    log.info(f"{'='*70}")
    jacs = [r['jaccard'] for r in all_results]
    recs = [r['recall'] for r in all_results]
    mses = [r['one_step_mse'] for r in all_results]
    log.info(f"  Jaccard: {np.mean(jacs):.3f} ± {np.std(jacs):.3f}")
    log.info(f"  Recall:  {np.mean(recs):.2f} ± {np.std(recs):.2f}")
    log.info(f"  MSE:     {np.mean(mses):.4e} ± {np.std(mses):.1e}")
    log.info(f"  Total:   {(time.time()-t_start)/3600:.1f} hours")
    log.info(f"  Saved:   {csv_path}")


if __name__ == "__main__":
    main()
