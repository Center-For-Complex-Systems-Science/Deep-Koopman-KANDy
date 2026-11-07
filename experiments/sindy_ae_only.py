"""
SINDy-AE only — runs just the SINDy-AE + PySR portion of the external
ablation for 5 seeds. Reuses EDMD and Deep-Koopman-KANDy results from
the existing results.csv.

Run from project root:
    python experiments/sindy_ae_only.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import csv, json, time, logging
from pathlib import Path

# PySR before torch
import pysr  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import simulate_lorenz, lorenz_rhs
from training import set_seed
from sindy_autoencoder import SindyAutoencoder, sindy_loss

OUTDIR = Path("results/sindy_ae_only")
OUTDIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
N_TRAJ = 60
N_TRAIN = 48
LATENT = 5
SEEDS = [0, 1, 2, 3, 4]
BURN_IN = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAMBDA_MAX = 0.906
TAU_L = 1.0 / LAMBDA_MAX
STEPS_PER_TAU = int(TAU_L / DT)
N_ROLLOUT_ICS = 10

TRUE_DICT = ["x", "y", "z", "xy", "xz"]
TRUE_SET = set(TRUE_DICT)
COEFF_THRESHOLD = 1e-2

SINDY_PRE_EPOCHS = 2001
SINDY_REFINE_EPOCHS = 1001
SINDY_THRESHOLD = 0.1
SINDY_THRESHOLD_EVERY = 500
SINDY_LR = 1e-3
SINDY_WIDTHS = [64, 32]
SINDY_LOSS_WEIGHTS = {
    'decoder': 1.0,
    'sindy_z': 1e-4,
    'sindy_x': 1e-4,
    'sindy_regularization': 1e-5,
}

PYSR_OPERATORS = ["+", "-", "*"]
PYSR_MAX_COMPLEXITY = 20
PYSR_POPULATIONS = 40
PYSR_ITERATIONS = 200
PYSR_SEEDS_PER_LATENT = 5
PYSR_R2_THRESHOLD = 0.95

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def compute_nrmse(pred, true):
    std = true.reshape(-1, 3).std(axis=0)
    std = np.maximum(std, 1e-8)
    return (np.sqrt((pred - true)**2) / std[None, :]).mean(axis=1)


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


def extract_monomials_from_sympy(expr, var_names=("x0", "x1", "x2"), canonical_map=None):
    import sympy as sp
    if canonical_map is None:
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


def run_pysr_on_encoder(encoder_fn, x_train, x_test, latent_dim, method_name="", seed=0):
    from pysr import PySRRegressor
    with torch.no_grad():
        z_train = encoder_fn(x_train).numpy()
        z_test = encoder_fn(x_test).numpy()
    x_train_np = x_train.numpy()
    x_test_np = x_test.numpy()

    detected = set()
    pysr_log = []

    for k in range(latent_dim):
        log.info(f"    PySR on z[{k}] ({method_name})")
        target_train = z_train[:, k]
        target_test = z_test[:, k]
        t_mean, t_std = target_train.mean(), max(target_train.std(), 1e-12)
        target_train_s = (target_train - t_mean) / t_std
        target_test_s = (target_test - t_mean) / t_std

        best_expr, best_loss, best_r2, best_complexity = None, float("inf"), -999, 999
        all_results = []

        for ps in range(PYSR_SEEDS_PER_LATENT):
            pysr_seed = seed * 1000 + k * 100 + ps
            model = PySRRegressor(
                binary_operators=PYSR_OPERATORS, unary_operators=[],
                maxsize=PYSR_MAX_COMPLEXITY, populations=PYSR_POPULATIONS,
                niterations=PYSR_ITERATIONS, random_state=pysr_seed,
                parallelism="serial", progress=False, verbosity=0,
                temp_equation_file=True,
            )
            model.fit(x_train_np, target_train_s, variable_names=["x0", "x1", "x2"])

            eqs = model.equations_
            if eqs is not None and len(eqs) > 0:
                for _, row in eqs.iterrows():
                    try:
                        pred_test = model.predict(x_test_np, index=int(row.name))
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


def eval_prediction_sindy(model, test_traj, test_xdot):
    x_flat = test_traj[:, :-1, :].reshape(-1, 3)
    x_tp1_flat = test_traj[:, 1:, :].reshape(-1, 3)
    dx_flat = test_xdot[:, :-1, :].reshape(-1, 3)
    model.eval()
    with torch.no_grad():
        out = model(x_flat, dx_flat)
        x_pred = x_flat + DT * out['dx_decode']
    one_step_mse = float((x_pred - x_tp1_flat).pow(2).mean().item())

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
                dx = lorenz_rhs(x.unsqueeze(0)).squeeze(0)
                out = model(x.unsqueeze(0), dx.unsqueeze(0))
                x = x + DT * out['dx_decode'].squeeze(0)
                xs.append(x.numpy().copy())
            pred = np.array(xs)
            nrmse = compute_nrmse(pred, true)
            rollout_errors.append(float(nrmse[min(steps_2tau, len(nrmse)-1)]))
    return one_step_mse, float(np.mean(rollout_errors))


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("  SINDy-AE Only — 5 seeds")
    log.info(f"  Device: {DEVICE}")
    log.info("=" * 70)

    # Data (same as external ablation)
    set_seed(42)
    traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT, burn_in=BURN_IN, device="cpu")
    xdot = lorenz_rhs(traj)
    train_traj = traj[:N_TRAIN].contiguous()
    test_traj = traj[N_TRAIN:].contiguous()
    train_xdot = xdot[:N_TRAIN].contiguous()
    test_xdot = xdot[N_TRAIN:].contiguous()

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

        model = SindyAutoencoder(
            input_dim=3, latent_dim=LATENT, widths=SINDY_WIDTHS,
            activation='sigmoid', poly_order=3, model_order=1,
            coefficient_init='constant',
        ).to(DEVICE)

        x_train = train_traj.reshape(-1, 3)
        dx_train = train_xdot.reshape(-1, 3)
        train_dl = DataLoader(TensorDataset(x_train, dx_train),
                              batch_size=8192, shuffle=True, drop_last=True)
        opt = torch.optim.Adam(model.parameters(), lr=SINDY_LR)

        # Phase 1
        log.info(f"  Phase 1: {SINDY_PRE_EPOCHS} epochs")
        for ep in range(1, SINDY_PRE_EPOCHS + 1):
            model.train()
            for x_batch, dx_batch in train_dl:
                x_batch, dx_batch = x_batch.to(DEVICE), dx_batch.to(DEVICE)
                out = model(x_batch, dx_batch)
                total, _, _ = sindy_loss(out, {'x': x_batch, 'dx': dx_batch},
                                          model, SINDY_LOSS_WEIGHTS, model_order=1)
                opt.zero_grad()
                total.backward()
                opt.step()
            if ep % SINDY_THRESHOLD_EVERY == 0:
                model.apply_thresholding(SINDY_THRESHOLD)
                active = model.active_terms().sum().item()
                log.info(f"    Ep {ep}: {active:.0f} active terms")

        # Phase 2
        log.info(f"  Phase 2: {SINDY_REFINE_EPOCHS} epochs refinement")
        refine_weights = dict(SINDY_LOSS_WEIGHTS)
        refine_weights['sindy_regularization'] = 0.0
        for ep in range(1, SINDY_REFINE_EPOCHS + 1):
            model.train()
            for x_batch, dx_batch in train_dl:
                x_batch, dx_batch = x_batch.to(DEVICE), dx_batch.to(DEVICE)
                out = model(x_batch, dx_batch)
                _, _, ref_loss = sindy_loss(out, {'x': x_batch, 'dx': dx_batch},
                                            model, refine_weights, model_order=1)
                opt.zero_grad()
                ref_loss.backward()
                opt.step()
            if ep % 1000 == 0:
                log.info(f"    Refine ep {ep}")

        model = model.cpu().eval()

        # Prediction
        mse, nrmse_2tau = eval_prediction_sindy(model, test_traj, test_xdot)
        log.info(f"  MSE={mse:.4e} NRMSE@2τ={nrmse_2tau:.3f}")

        # PySR
        x_test = test_traj.reshape(-1, 3)
        tr_idx = np.random.choice(len(x_train), min(5000, len(x_train)), replace=False)
        te_idx = np.random.choice(len(x_test), min(2000, len(x_test)), replace=False)

        detected, pysr_log = run_pysr_on_encoder(
            model.encode, x_train[tr_idx], x_test[te_idx],
            LATENT, method_name="SINDy-AE", seed=seed,
        )

        scores = score_dictionary(detected)
        n_params = sum(p.numel() for p in model.parameters())
        elapsed = time.time() - t0

        log.info(f"  Jaccard={scores['jaccard']:.3f} Rec={scores['recall']:.0%} "
                 f"detected={scores['detected']} ({elapsed:.0f}s)")

        result = {"method": "SINDy-AE+PySR", "seed": seed,
                  "one_step_mse": mse, "nrmse_2tau": nrmse_2tau,
                  "n_params": n_params,
                  "train_epochs": SINDY_PRE_EPOCHS + SINDY_REFINE_EPOCHS,
                  **scores}
        all_results.append(result)

        csv_writer.writerow(result)
        csv_file.flush()
        json.dump({"method": "SINDy-AE+PySR", "seed": seed, "pysr_log": pysr_log},
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
