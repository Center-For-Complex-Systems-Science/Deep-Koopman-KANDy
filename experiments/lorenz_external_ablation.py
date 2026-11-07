"""
Symbolic Dictionary Recovery Benchmark on the Lorenz System
============================================================

Purpose
-------
This experiment tests the central interpretability claim of Deep-Koopman-KANDy:
that the combination of a two-layer KAN encoder with level-set manifold analysis
recovers the minimal Koopman-invariant polynomial dictionary {x, y, z, xy, xz}
more cleanly than alternative pipelines that also aim to produce symbolic forms
for learned latent coordinates.

The comparison is scoped specifically to the recovery of symbolic expressions
for the ENCODER COORDINATES z_k(x), not to the recovery of latent dynamics.
All three methods compared here are forced through the same scoring pipeline
(Jaccard / precision / recall against the target dictionary) so that the
comparison is apples-to-apples on the interpretability-of-the-lifting axis.

Methods Compared
----------------
1. Deep-Koopman-KANDy (this work).
   Two-layer KAN encoder/decoder trained with the stable Koopman generator
   A = K - L^T L. Symbolic recovery via level-set manifold analysis: each
   latent coordinate z_k is decomposed as f(x) = h(g(x)) with inner function
   g recovered by Lasso on a degree-3 monomial basis and outer function h
   recovered via the gradient formula h'(g) = (grad f . grad g) / ||grad g||^2.
   (Protocol matches Table 1 of the main paper.)

2. Deep-Koopman (MLP) + PySR on encoder.
   The "MLP + stable" ablation row from Table 1 of the main paper, trained
   identically to Deep-Koopman-KANDy but with MLP encoder/decoder. After
   training, the encoder is frozen and PySR is run independently on each
   latent coordinate z_k(x) to produce a symbolic expression. This isolates
   the contribution of the KAN architecture to symbolic recoverability.

3. SINDy-AE (Champion, Lusch, Kutz, Brunton, PNAS 2019) + PySR on encoder.
   The full SINDy-AE model is trained on the same Lorenz data using the
   reference implementation at github.com/kpchamp/SindyAutoencoders. After
   training, the SINDy equation for z-dot is discarded and PySR is run on
   each latent coordinate z_k(x) from the trained encoder. NOTE: this is NOT
   SINDy-AE's native output; SINDy-AE natively outputs a sparse equation for
   latent dynamics, not for the encoder. We reroute its output through PySR
   so the comparison is commensurable. This choice is explicit and
   conservative -- it tests whether SINDy-AE's joint objective yields an
   encoder whose coordinates happen to be recoverable in symbolic form.

Data
----
Lorenz system (sigma=10, rho=28, beta=8/3). 60 trajectories of 6000 steps
with RK4 at dt=0.01, 500-step burn-in, initial conditions drawn uniformly
from [-20, 20] x [-30, 30] x [0, 50]. First 80% used for training, 20% held
out for test. Identical training data across all three methods.

Derivatives: SINDy-AE requires x-dot. Analytical derivatives from the RK4
integrator are provided rather than finite-differenced from snapshots. This
is the charitable choice and removes a confound reviewers would otherwise
raise. The Deep-Koopman methods use the existing (x_t, x_{t+1}) snapshot
pairs with no derivative access.

Symbolic Regression Protocol (methods 2 and 3)
----------------------------------------------
PySR is run with an identical configuration for both methods:
- Operator set: {+, -, *} (sufficient for the polynomial target; no
  transcendentals, which would invite PySR to prefer short trigonometric
  surrogates for xy and xz).
- Maximum complexity: 20.
- Populations: 40, iterations: 200.
- Loss: MSE against the frozen encoder's output z_k(x) on the training set.
- For each latent coordinate z_k, PySR is run with 5 independent seeds; the
  returned expression is selected as the one with the lowest loss at the
  lowest complexity tier that achieves R^2 >= 0.95 on a held-out validation
  split of the encoder outputs. If no expression clears 0.95, the lowest-
  loss expression at any complexity is taken.

Model Selection
---------------
Each of the three methods is trained with 5 random seeds, matching the
protocol of Table 1 in the main paper. Metrics are reported as mean +- std
over seeds. This is a conservative evaluation of SINDy-AE, whose reference
protocol uses best-of-10 with manual model selection; we note this
explicitly in the paper text.

Metrics (extending Table 1)
---------------------------
For each method, after symbolic expressions have been obtained for every
latent coordinate (via level-set analysis for method 1, via PySR for
methods 2 and 3), the union of nonzero monomial terms across all latent
coordinates is compared to the target set {x, y, z, xy, xz} via:
- Jaccard similarity: |recovered ∩ target| / |recovered ∪ target|
- Precision: |recovered ∩ target| / |recovered|
- Recall: |recovered ∩ target| / |target|

A monomial is considered "recovered" for methods 2 and 3 if it appears in
any PySR-returned expression with coefficient magnitude above 1e-2 after
standardization (matching the threshold used for the level-set analysis
in the main paper).

One-step MSE and NRMSE at 2*tau_Lambda are also reported for continuity
with Table 1, though these are secondary -- the primary claim tested by
this experiment is dictionary recoverability, not predictive accuracy.

Output
------
Produces one extended row per method for Table 1 in the main paper, plus
a supplementary table listing the top-k PySR-recovered expressions per
latent coordinate for methods 2 and 3. Raw PySR outputs are logged for
reproducibility.

Expected Outcome (prior, not a commitment)
------------------------------------------
- Method 1 (ours): Jaccard = 1.0, matching Table 1.
- Method 2 (MLP + PySR): Jaccard ≈ 0.4-0.6. Linear terms (x, y, z) likely
  recovered; cross-terms (xy, xz) likely missed because MLP latents are
  typically tangled affine mixtures where polynomial structure does not
  dominate the PySR loss landscape.
- Method 3 (SINDy-AE + PySR): Jaccard ≈ 0.5-0.7. SINDy-AE's joint objective
  pressures the encoder toward latents with sparse polynomial dynamics,
  which indirectly biases them toward polynomial-friendly forms, but the
  encoder itself is not regularized for sparse symbolic structure.

Deviations from these priors are informative either way and should be
reported faithfully.

References
----------
- Champion, Lusch, Kutz, Brunton. "Data-driven discovery of coordinates
  and governing equations." PNAS 116(45): 22445-22451, 2019.
  Code: github.com/kpchamp/SindyAutoencoders
- Cranmer. "Interpretable Machine Learning for Science with PySR and
  SymbolicRegression.jl." arXiv:2305.01582, 2023.
  Code: github.com/MilesCranmer/PySR
"""

import sys, os, csv, json, time, logging
from pathlib import Path

# PySR / juliacall must be imported BEFORE torch to avoid segfault.
# See https://github.com/pytorch/pytorch/issues/78829
import pysr  # noqa: F401 — early import to initialize Julia before torch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

import matplotlib
matplotlib.use("Agg")

from data import simulate_lorenz, PairDataset, lorenz_rhs
from models import KANAutoEncoder
from kan import KANLinear
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition
from sindy_autoencoder import SindyAutoencoder, sindy_loss

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTDIR = Path("results/lorenz_external_ablation")
OUTDIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
N_TRAJ = 60
N_TRAIN = 48        # 80% of 60
LATENT = 5
MID = 5
SEEDS = [0, 1, 2, 3, 4]
BURN_IN = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAMBDA_MAX = 0.906
TAU_L = 1.0 / LAMBDA_MAX
STEPS_PER_TAU = int(TAU_L / DT)
N_ROLLOUT_ICS = 10

TRUE_DICT = ["x", "y", "z", "xy", "xz"]
TRUE_SET = set(TRUE_DICT)
COEFF_THRESHOLD = 1e-2  # monomial recovery threshold after standardization

# Deep-Koopman-KANDy training (KANLinear backend with spline L1+entropy regularization)
DK_KANDY_EPOCHS = 200
DK_KANDY_SPLINE_REG = 0.001  # L1 + entropy on spline weights (optimal from multi-seed sweep)
DK_KANDY_LASSO_ALPHA = 1e-5  # level-set Lasso alpha
KAN_KWARGS = dict(grid_size=5, spline_order=3,
                  grid_range_l1=[-30, 50], grid_range_l2=[-10, 10])

# MLP + Stable training (matches Table 1 variant C)
MLP_EPOCHS = 200

# SINDy-AE training (Champion et al. published protocol)
SINDY_PRE_EPOCHS = 5001
SINDY_REFINE_EPOCHS = 5001
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

# PySR config (identical for methods 2 and 3)
PYSR_OPERATORS = ["+", "-", "*"]
PYSR_MAX_COMPLEXITY = 20
PYSR_POPULATIONS = 40
PYSR_ITERATIONS = 200
PYSR_SEEDS_PER_LATENT = 5
PYSR_R2_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_data(data_seed):
    """Generate Lorenz data deterministically from data_seed.
    Returns traj (N_TRAJ, 6000, 3) and xdot (N_TRAJ, 6000, 3)."""
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)

    traj = simulate_lorenz(
        n_traj=N_TRAJ, n_steps=6000, dt=DT,
        burn_in=BURN_IN, device="cpu",
    )

    # Analytical derivatives at every snapshot
    xdot = lorenz_rhs(traj)

    # Sanity checks
    assert traj.shape == (N_TRAJ, 6000, 3), f"Bad traj shape: {traj.shape}"
    assert xdot.shape == (N_TRAJ, 6000, 3), f"Bad xdot shape: {xdot.shape}"
    assert not torch.isnan(traj).any(), "NaN in traj"
    assert not torch.isnan(xdot).any(), "NaN in xdot"

    # Assert ICs within bounds
    ics = traj[:, 0, :]  # after burn-in these are on the attractor
    # (post-burn-in ICs won't be in the init_box, but should be finite)
    assert torch.isfinite(ics).all(), "Non-finite ICs"

    # Verify analytical derivatives against a time-centered finite difference.
    # A forward difference approximates the derivative at the interval midpoint,
    # not exactly at the left endpoint, so compare against the trapezoidal
    # average of the endpoint derivatives.
    fd_xdot = (traj[:, 1:, :] - traj[:, :-1, :]) / DT
    ana_xdot_mid = 0.5 * (xdot[:, :-1, :] + xdot[:, 1:, :])
    discrepancy = (fd_xdot - ana_xdot_mid).abs().mean().item()
    log.info(
        "  Derivative check (centered): mean |FD - avg analytical| = "
        f"{discrepancy:.6f}"
    )
    assert discrepancy < 0.2, f"Derivative discrepancy too large: {discrepancy}"

    return traj, xdot


def split_data(traj, xdot):
    """80/20 train/test split by trajectory."""
    train_traj = traj[:N_TRAIN].contiguous()
    test_traj = traj[N_TRAIN:].contiguous()
    train_xdot = xdot[:N_TRAIN].contiguous()
    test_xdot = xdot[N_TRAIN:].contiguous()
    return train_traj, test_traj, train_xdot, test_xdot


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION METRICS (shared across all methods)
# ═══════════════════════════════════════════════════════════════════════════

def compute_nrmse(pred, true):
    std = true.reshape(-1, 3).std(axis=0)
    std = np.maximum(std, 1e-8)
    return (np.sqrt((pred - true)**2) / std[None, :]).mean(axis=1)


def eval_prediction_koopman(model, test_traj):
    """One-step MSE and NRMSE@2tau for a DeepKoopmanCT-like model."""
    ds = PairDataset(test_traj)
    dl = DataLoader(ds, batch_size=4096)
    total_se, n = 0.0, 0
    with torch.no_grad():
        for x_t, x_tp1 in dl:
            out = model.forward_one_step(x_t, DT)
            total_se += (out["x_tp1_pred"] - x_tp1).pow(2).sum().item()
            n += x_t.shape[0] * x_t.shape[1]
    one_step_mse = total_se / n

    steps_2tau = 2 * STEPS_PER_TAU
    max_steps = min(steps_2tau, test_traj.shape[1] - 1)
    ics = np.random.choice(test_traj.shape[0], min(N_ROLLOUT_ICS, test_traj.shape[0]), replace=False)
    rollout_errors = []
    with torch.no_grad():
        for ic in ics:
            true = test_traj[ic, :max_steps+1].numpy()
            x = test_traj[ic, 0]
            xs = [x.numpy()]
            for _ in range(max_steps):
                out = model.forward_one_step(x.unsqueeze(0), DT)
                x = out["x_tp1_pred"].squeeze(0)
                xs.append(x.numpy())
            pred = np.array(xs)
            nrmse = compute_nrmse(pred, true)
            rollout_errors.append(float(nrmse[min(steps_2tau, len(nrmse)-1)]))
    nrmse_2tau = float(np.mean(rollout_errors))

    return one_step_mse, nrmse_2tau


def eval_prediction_sindy(model, test_traj, test_xdot):
    """One-step MSE for SINDy-AE: x_{t+1} = x_t + dt * dx_decode."""
    x_flat = test_traj[:, :-1, :].reshape(-1, 3)
    x_tp1_flat = test_traj[:, 1:, :].reshape(-1, 3)
    dx_flat = test_xdot[:, :-1, :].reshape(-1, 3)

    model.eval()
    with torch.no_grad():
        out = model(x_flat, dx_flat)
        x_pred = x_flat + DT * out['dx_decode']
    one_step_mse = float((x_pred - x_tp1_flat).pow(2).mean().item())

    # Rollout via SINDy dynamics
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
    nrmse_2tau = float(np.mean(rollout_errors))

    return one_step_mse, nrmse_2tau


# ═══════════════════════════════════════════════════════════════════════════
# SCORING PIPELINE (shared)
# ═══════════════════════════════════════════════════════════════════════════

def score_dictionary(detected_monomials):
    """Compute Jaccard, precision, recall against TRUE_SET."""
    detected = set(detected_monomials)
    tp = len(detected & TRUE_SET)
    precision = tp / max(len(detected), 1)
    recall = tp / len(TRUE_SET)
    union = len(detected | TRUE_SET)
    jaccard = tp / max(union, 1)
    return {
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "detected": sorted(detected),
        "n_detected": len(detected),
        "n_tp": tp,
        "n_fp": len(detected - TRUE_SET),
    }


def extract_monomials_from_sympy(expr, var_names=("x0", "x1", "x2"),
                                  canonical_map=None):
    """Extract monomials from a SymPy expression with |coeff| > COEFF_THRESHOLD.

    Returns a set of canonical monomial names.
    """
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

    syms = [sp.Symbol(v) for v in var_names]
    expanded = sp.expand(expr)

    monomials = set()
    if expanded.is_Add:
        terms = expanded.as_ordered_terms()
    else:
        terms = [expanded]

    for term in terms:
        coeff, mono = term.as_coeff_Mul()
        if abs(float(coeff)) < COEFF_THRESHOLD:
            continue
        # Convert monomial to string key
        mono_str = str(mono)
        if mono_str == "1" or mono == sp.S.One:
            continue
        # Try canonical mapping
        canonical = canonical_map.get(mono_str, mono_str)
        if canonical in {"x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2",
                          "x^3", "x^2y", "x^2z", "xy^2", "xyz", "xz^2",
                          "y^3", "y^2z", "yz^2", "z^3"}:
            monomials.add(canonical)

    return monomials


# ═══════════════════════════════════════════════════════════════════════════
# PySR WRAPPER (used by methods 2 and 3)
# ═══════════════════════════════════════════════════════════════════════════

def run_pysr_on_encoder(encoder_fn, x_train, x_test, latent_dim,
                         method_name="", seed=0):
    """Run PySR on each latent coordinate of a frozen encoder.

    Args:
        encoder_fn: callable, x (N,3) -> z (N, latent_dim)
        x_train: (N_train, 3) tensor
        x_test: (N_test, 3) tensor
        latent_dim: int
        method_name: str for logging
        seed: base seed for PySR

    Returns:
        detected_monomials: set of canonical names
        pysr_log: list of dicts (one per latent dim) for jsonl
    """
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

        # Standardize target
        t_mean, t_std = target_train.mean(), target_train.std()
        if t_std < 1e-12:
            t_std = 1.0
        target_train_s = (target_train - t_mean) / t_std
        target_test_s = (target_test - t_mean) / t_std

        best_expr = None
        best_loss = float("inf")
        best_r2 = -999
        best_complexity = 999
        all_results = []

        for ps in range(PYSR_SEEDS_PER_LATENT):
            pysr_seed = seed * 1000 + k * 100 + ps

            model = PySRRegressor(
                binary_operators=PYSR_OPERATORS,
                unary_operators=[],
                maxsize=PYSR_MAX_COMPLEXITY,
                populations=PYSR_POPULATIONS,
                niterations=PYSR_ITERATIONS,
                random_state=pysr_seed,
                deterministic=True,
                parallelism="serial",
                progress=False,
                verbosity=0,
                temp_equation_file=True,
            )

            model.fit(x_train_np, target_train_s,
                      variable_names=["x0", "x1", "x2"])

            # Evaluate all equations on test set
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
                            "pysr_seed": ps,
                            "complexity": complexity,
                            "loss": loss,
                            "r2_test": r2,
                            "equation": str(row['equation']),
                            "sympy": str(row.get('sympy_format', row['equation'])),
                        })

                        # Selection: lowest complexity with R² >= 0.95
                        if r2 >= PYSR_R2_THRESHOLD:
                            if (complexity < best_complexity or
                                (complexity == best_complexity and loss < best_loss)):
                                best_expr = row.get('sympy_format', None)
                                best_loss = loss
                                best_r2 = r2
                                best_complexity = complexity
                        elif best_r2 < PYSR_R2_THRESHOLD:
                            # Fallback: best loss at any complexity
                            if loss < best_loss:
                                best_expr = row.get('sympy_format', None)
                                best_loss = loss
                                best_r2 = r2
                                best_complexity = complexity
                    except Exception:
                        pass

        # Extract monomials from best expression
        k_monomials = extract_monomials_from_sympy(best_expr)
        detected |= k_monomials

        pysr_log.append({
            "latent_dim": k,
            "best_expr": str(best_expr),
            "best_r2": best_r2,
            "best_complexity": best_complexity,
            "monomials": sorted(k_monomials),
            "all_results": all_results,
        })

        log.info(f"      best: {best_expr} (R²={best_r2:.3f}, "
                 f"complexity={best_complexity}, monomials={sorted(k_monomials)})")

    return detected, pysr_log


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 1: Deep-Koopman-KANDy
# ═══════════════════════════════════════════════════════════════════════════

def _spline_regularization(ae):
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


def run_deep_koopman_kandy(train_traj, test_traj, model_seed):
    """Train KANLinear [3,5,5] + stable with spline regularization, level-set recovery."""
    log.info(f"  [Deep-Koopman-KANDy] Training (seed={model_seed})")

    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    ae = KANAutoEncoder(3, MID, LATENT, **KAN_KWARGS)
    model = DeepKoopmanCTStable(ae, LATENT).to(DEVICE)

    traj_full = torch.cat([train_traj, test_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DK_KANDY_EPOCHS)
    mse_fn = nn.MSELoss()

    for ep in range(1, DK_KANDY_EPOCHS + 1):
        model.train()
        for x_t, x_tp1 in train_dl:
            x_t, x_tp1 = x_t.to(DEVICE), x_tp1.to(DEVICE)
            out = model.forward_one_step(x_t, DT)
            z_true = model.ae.encoder(x_tp1)
            loss = (mse_fn(out["x_tp1_pred"], x_tp1)
                    + mse_fn(out["z_tp1_pred"], z_true)
                    + 0.5 * mse_fn(out["x_t_rec"], x_t)
                    + DK_KANDY_SPLINE_REG * _spline_regularization(model.ae))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 50 == 0:
            log.info(f"    Ep {ep}")

    model.eval()

    # Prediction (on CPU for level-set analysis compatibility)
    model_cpu = model.cpu()
    mse, nrmse_2tau = eval_prediction_koopman(model_cpu, test_traj)
    assert nrmse_2tau < 5.0, f"KANDy NRMSE sanity check failed: {nrmse_2tau}"

    # Dictionary via level-set analysis (already on CPU)
    x_flat = train_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub_t = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)

    detected = set()
    decomp_log = []
    for zi in range(LATENT):
        dr = full_decomposition(model_cpu, x_sub_t, zi, target_type='latent',
                                max_degree=3, lasso_alpha=DK_KANDY_LASSO_ALPHA, h_degree=4)
        pairs = sorted(zip(dr['inner']['active_names'],
                           dr['inner']['active_coeffs']),
                       key=lambda x: -abs(x[1]))
        top = [(n, float(c)) for n, c in pairs if n != "1"][:5]
        if dr['recon_r2'] > 0.8:
            detected |= {n for n, c in top}

        decomp_log.append({
            "latent_dim": zi,
            "r2_hg": float(dr['recon_r2']),
            "top_terms": top,
        })

    scores = score_dictionary(detected)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "method": "Deep-Koopman-KANDy",
        "one_step_mse": mse,
        "nrmse_2tau": nrmse_2tau,
        "n_params": n_params,
        "train_epochs": DK_KANDY_EPOCHS,
        **scores,
        "decomp_log": decomp_log,
    }


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 2: MLP + Stable + PySR
# ═══════════════════════════════════════════════════════════════════════════

class MLPAutoEncoder(nn.Module):
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


def run_mlp_stable(train_traj, test_traj, model_seed):
    """Train MLP + stable, then run PySR on encoder."""
    log.info(f"  [MLP+Stable] Training (seed={model_seed})")

    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    ae = MLPAutoEncoder(3, MID, LATENT)
    traj_full = torch.cat([train_traj, test_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    model = DeepKoopmanCTStable(ae, LATENT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    mse_fn = nn.MSELoss()

    for ep in range(1, MLP_EPOCHS + 1):
        model.train()
        for x_t, x_tp1 in train_dl:
            x_t, x_tp1 = x_t.to(DEVICE), x_tp1.to(DEVICE)
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
            log.info(f"    Ep {ep}")

    model = model.cpu().eval()

    # Prediction
    mse, nrmse_2tau = eval_prediction_koopman(model, test_traj)
    assert nrmse_2tau < 5.0, f"MLP NRMSE sanity check failed: {nrmse_2tau}"

    # PySR on frozen encoder
    x_train_flat = train_traj.reshape(-1, 3)
    x_test_flat = test_traj.reshape(-1, 3)
    # Subsample for PySR (full dataset is too large)
    tr_idx = np.random.choice(len(x_train_flat), min(5000, len(x_train_flat)), replace=False)
    te_idx = np.random.choice(len(x_test_flat), min(2000, len(x_test_flat)), replace=False)

    def encoder_fn(x):
        return model.ae.encoder(x)

    detected, pysr_log = run_pysr_on_encoder(
        encoder_fn,
        x_train_flat[tr_idx],
        x_test_flat[te_idx],
        LATENT,
        method_name="MLP+Stable",
        seed=model_seed,
    )

    scores = score_dictionary(detected)
    n_params = sum(p.numel() for p in model.parameters())

    # Assert PySR produced expressions for every latent
    assert len(pysr_log) == LATENT, f"PySR missing results: got {len(pysr_log)}"

    return {
        "method": "MLP+Stable+PySR",
        "one_step_mse": mse,
        "nrmse_2tau": nrmse_2tau,
        "n_params": n_params,
        "train_epochs": MLP_EPOCHS,
        **scores,
        "pysr_log": pysr_log,
    }


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 3: SINDy-AE + PySR
# ═══════════════════════════════════════════════════════════════════════════

def run_sindy_ae(train_traj, test_traj, train_xdot, test_xdot, model_seed):
    """Train SINDy-AE with Champion et al. protocol, then PySR on encoder."""
    log.info(f"  [SINDy-AE] Training (seed={model_seed})")

    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    model = SindyAutoencoder(
        input_dim=3,
        latent_dim=LATENT,
        widths=SINDY_WIDTHS,
        activation='sigmoid',
        poly_order=3,
        model_order=1,
        coefficient_init='constant',
    ).to(DEVICE)

    x_train = train_traj.reshape(-1, 3)
    dx_train = train_xdot.reshape(-1, 3)
    x_test = test_traj.reshape(-1, 3)
    dx_test = test_xdot.reshape(-1, 3)

    train_ds = TensorDataset(x_train, dx_train)
    train_dl = DataLoader(train_ds, batch_size=2048, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=SINDY_LR)

    # Phase 1: Pre-threshold training
    log.info(f"    Phase 1: {SINDY_PRE_EPOCHS} epochs with L1 regularization")
    for ep in range(1, SINDY_PRE_EPOCHS + 1):
        model.train()
        for x_batch, dx_batch in train_dl:
            x_batch, dx_batch = x_batch.to(DEVICE), dx_batch.to(DEVICE)
            out = model(x_batch, dx_batch)
            total, parts, _ = sindy_loss(
                out, {'x': x_batch, 'dx': dx_batch}, model,
                SINDY_LOSS_WEIGHTS, model_order=1,
            )
            opt.zero_grad()
            total.backward()
            opt.step()

        if ep % SINDY_THRESHOLD_EVERY == 0:
            model.apply_thresholding(SINDY_THRESHOLD)
            active = model.active_terms().sum().item()
            log.info(f"    Ep {ep}: thresholding, {active:.0f} active terms")

    # Phase 2: Refinement (L1 weight → 0)
    log.info(f"    Phase 2: {SINDY_REFINE_EPOCHS} epochs refinement (no L1)")
    refine_weights = dict(SINDY_LOSS_WEIGHTS)
    refine_weights['sindy_regularization'] = 0.0

    for ep in range(1, SINDY_REFINE_EPOCHS + 1):
        model.train()
        for x_batch, dx_batch in train_dl:
            x_batch, dx_batch = x_batch.to(DEVICE), dx_batch.to(DEVICE)
            out = model(x_batch, dx_batch)
            _, _, refinement_loss = sindy_loss(
                out, {'x': x_batch, 'dx': dx_batch}, model,
                refine_weights, model_order=1,
            )
            opt.zero_grad()
            refinement_loss.backward()
            opt.step()

        if ep % 1000 == 0:
            log.info(f"    Refine ep {ep}")

    model = model.cpu().eval()

    # Prediction
    mse, nrmse_2tau = eval_prediction_sindy(model, test_traj, test_xdot)
    assert nrmse_2tau < 10.0, f"SINDy-AE NRMSE sanity check failed: {nrmse_2tau}"

    # PySR on frozen encoder
    tr_idx = np.random.choice(len(x_train), min(5000, len(x_train)), replace=False)
    te_idx = np.random.choice(len(x_test), min(2000, len(x_test)), replace=False)

    def encoder_fn(x):
        return model.encode(x)

    detected, pysr_log = run_pysr_on_encoder(
        encoder_fn,
        x_train[tr_idx],
        x_test[te_idx],
        LATENT,
        method_name="SINDy-AE",
        seed=model_seed,
    )

    scores = score_dictionary(detected)
    n_params = sum(p.numel() for p in model.parameters())

    assert len(pysr_log) == LATENT, f"PySR missing results: got {len(pysr_log)}"

    return {
        "method": "SINDy-AE+PySR",
        "one_step_mse": mse,
        "nrmse_2tau": nrmse_2tau,
        "n_params": n_params,
        "train_epochs": SINDY_PRE_EPOCHS + SINDY_REFINE_EPOCHS,
        **scores,
        "pysr_log": pysr_log,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PySR SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def pysr_smoke_test():
    """Verify PySR works before running the full experiment."""
    log.info("PySR smoke test...")
    from pysr import PySRRegressor

    x = np.random.randn(200, 1)
    y = x[:, 0]**2 + x[:, 0]
    model = PySRRegressor(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=10,
        populations=10,
        niterations=20,
        random_state=0,
        deterministic=True,
        parallelism="serial",
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )
    model.fit(x, y, variable_names=["x"])
    pred = model.predict(x)
    r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
    log.info(f"  PySR smoke test: R²={r2:.3f}, best={model.get_best()['equation']}")
    assert r2 > 0.9, f"PySR smoke test failed: R²={r2}"
    log.info("  PySR smoke test PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("  Symbolic Dictionary Recovery Benchmark — Lorenz System")
    log.info(f"  Device: {DEVICE}")
    log.info("=" * 70)

    # Smoke test PySR
    pysr_smoke_test()

    # CSV output
    csv_path = OUTDIR / "results.csv"
    csv_fields = [
        "method", "seed", "data_seed", "model_seed",
        "one_step_mse", "nrmse_2tau",
        "jaccard", "precision", "recall",
        "n_detected", "n_tp", "n_fp", "detected",
        "n_params", "train_epochs",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # JSONL for PySR/decomposition logs
    jsonl_path = OUTDIR / "expressions.jsonl"
    jsonl_file = open(jsonl_path, "w")

    all_results = {m: [] for m in [
        "EDMD-poly (oracle)", "Deep-Koopman-KANDy", "MLP+Stable+PySR", "SINDy-AE+PySR"
    ]}

    # Generate data once (matches ablation_lorenz.py: set_seed(42))
    data_seed = 42
    log.info(f"  Generating shared data (data_seed={data_seed})...")
    traj, xdot = generate_data(data_seed)
    train_traj, test_traj, train_xdot, test_xdot = split_data(traj, xdot)

    # ── EDMD-poly (oracle): deterministic, run once ──
    log.info(f"\n  EDMD-poly (oracle) — degree-3 polynomial dictionary")
    EDMD_NAMES = ["x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2",
                  "x^3", "x^2y", "x^2z", "xy^2", "xyz", "xz^2", "y^3", "y^2z", "yz^2", "z^3"]

    def _poly_lift_d3(X):
        x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        return np.column_stack([
            x, y, z, x**2, x*y, x*z, y**2, y*z, z**2,
            x**3, x**2*y, x**2*z, x*y**2, x*y*z, x*z**2, y**3, y**2*z, y*z**2, z**3,
        ])

    X_t = train_traj[:, :-1, :].reshape(-1, 3).numpy()
    X_tp1 = train_traj[:, 1:, :].reshape(-1, 3).numpy()
    Z_t = _poly_lift_d3(X_t)
    Z_tp1 = _poly_lift_d3(X_tp1)
    K_edmd = Z_tp1.T @ np.linalg.pinv(Z_t.T)

    # One-step MSE
    Xv_t = test_traj[:, :-1, :].reshape(-1, 3).numpy()
    Xv_tp1 = test_traj[:, 1:, :].reshape(-1, 3).numpy()
    Zv_t = _poly_lift_d3(Xv_t)
    edmd_pred = (K_edmd @ Zv_t.T).T[:, :3]
    edmd_mse = float(np.mean((edmd_pred - Xv_tp1)**2))

    # NRMSE@2tau via rollout
    steps_2tau = 2 * STEPS_PER_TAU
    max_steps = min(steps_2tau, test_traj.shape[1] - 1)
    edmd_rollout_errors = []
    for ic in range(min(N_ROLLOUT_ICS, test_traj.shape[0])):
        true = test_traj[ic, :max_steps+1].numpy()
        x = true[0].copy()
        xs = [x.copy()]
        for _ in range(max_steps):
            zz = _poly_lift_d3(x.reshape(1, -1)).flatten()
            x = (K_edmd @ zz)[:3].copy()
            xs.append(x.copy())
        pred = np.array(xs)
        nrmse = compute_nrmse(pred, true)
        edmd_rollout_errors.append(float(nrmse[min(steps_2tau, len(nrmse)-1)]))
    edmd_nrmse_2tau = float(np.mean(edmd_rollout_errors))

    # Dictionary via K-I thresholding
    K_dev = K_edmd - np.eye(K_edmd.shape[0])
    K_dev_state = K_dev[:3, :]
    edmd_thresh = 0.01 * np.abs(K_dev_state).max()
    edmd_detected = set()
    for col in range(len(EDMD_NAMES)):
        if np.any(np.abs(K_dev_state[:, col]) > edmd_thresh):
            edmd_detected.add(EDMD_NAMES[col])
    edmd_scores = score_dictionary(edmd_detected)

    log.info(f"  EDMD-poly: MSE={edmd_mse:.4e} NRMSE@2τ={edmd_nrmse_2tau:.3f} "
             f"Jaccard={edmd_scores['jaccard']:.3f} detected={edmd_scores['detected']}")

    # Store as single result (deterministic — same for all seeds)
    edmd_result = {
        "method": "EDMD-poly (oracle)",
        "one_step_mse": edmd_mse,
        "nrmse_2tau": edmd_nrmse_2tau,
        "n_params": K_edmd.size,
        "train_epochs": 0,
        **edmd_scores,
    }
    for _ in SEEDS:
        all_results["EDMD-poly (oracle)"].append(edmd_result)

    csv_writer.writerow({
        "method": edmd_result["method"], "seed": "all",
        "data_seed": data_seed, "model_seed": "N/A",
        "one_step_mse": edmd_result["one_step_mse"],
        "nrmse_2tau": edmd_result["nrmse_2tau"],
        "jaccard": edmd_result["jaccard"], "precision": edmd_result["precision"],
        "recall": edmd_result["recall"], "n_detected": edmd_result["n_detected"],
        "n_tp": edmd_result["n_tp"], "n_fp": edmd_result["n_fp"],
        "detected": ";".join(edmd_result["detected"]),
        "n_params": edmd_result["n_params"], "train_epochs": 0,
    })
    csv_file.flush()

    for seed in SEEDS:
        log.info(f"\n{'='*60}")
        log.info(f"  SEED {seed}")
        log.info(f"{'='*60}")

        model_seed = seed
        log.info(f"  data_seed={data_seed}, model_seed={model_seed}")

        # --- Method 1: KANDy ---
        t0 = time.time()
        r1 = run_deep_koopman_kandy(train_traj, test_traj, model_seed)
        r1["wall_time"] = time.time() - t0
        all_results["Deep-Koopman-KANDy"].append(r1)
        log.info(f"  [Deep-Koopman-KANDy] MSE={r1['one_step_mse']:.4e} NRMSE@2τ={r1['nrmse_2tau']:.3f} "
                 f"Jaccard={r1['jaccard']:.3f} ({r1['wall_time']:.0f}s)")

        csv_writer.writerow({
            "method": r1["method"], "seed": seed,
            "data_seed": data_seed, "model_seed": model_seed,
            "one_step_mse": r1["one_step_mse"], "nrmse_2tau": r1["nrmse_2tau"],
            "jaccard": r1["jaccard"], "precision": r1["precision"],
            "recall": r1["recall"], "n_detected": r1["n_detected"],
            "n_tp": r1["n_tp"], "n_fp": r1["n_fp"],
            "detected": ";".join(r1["detected"]),
            "n_params": r1["n_params"], "train_epochs": r1["train_epochs"],
        })
        json.dump({"method": r1["method"], "seed": seed,
                    "decomp_log": r1.get("decomp_log", [])},
                   jsonl_file, default=str)
        jsonl_file.write("\n")
        csv_file.flush()

        # --- Method 2: MLP + Stable + PySR ---
        t0 = time.time()
        r2 = run_mlp_stable(train_traj, test_traj, model_seed)
        r2["wall_time"] = time.time() - t0
        all_results["MLP+Stable+PySR"].append(r2)
        log.info(f"  [MLP+PySR] MSE={r2['one_step_mse']:.4e} NRMSE@2τ={r2['nrmse_2tau']:.3f} "
                 f"Jaccard={r2['jaccard']:.3f} ({r2['wall_time']:.0f}s)")

        csv_writer.writerow({
            "method": r2["method"], "seed": seed,
            "data_seed": data_seed, "model_seed": model_seed,
            "one_step_mse": r2["one_step_mse"], "nrmse_2tau": r2["nrmse_2tau"],
            "jaccard": r2["jaccard"], "precision": r2["precision"],
            "recall": r2["recall"], "n_detected": r2["n_detected"],
            "n_tp": r2["n_tp"], "n_fp": r2["n_fp"],
            "detected": ";".join(r2["detected"]),
            "n_params": r2["n_params"], "train_epochs": r2["train_epochs"],
        })
        json.dump({"method": r2["method"], "seed": seed,
                    "pysr_log": r2.get("pysr_log", [])},
                   jsonl_file, default=str)
        jsonl_file.write("\n")
        csv_file.flush()

        # --- Method 3: SINDy-AE + PySR ---
        t0 = time.time()
        r3 = run_sindy_ae(train_traj, test_traj, train_xdot, test_xdot, model_seed)
        r3["wall_time"] = time.time() - t0
        all_results["SINDy-AE+PySR"].append(r3)
        log.info(f"  [SINDy+PySR] MSE={r3['one_step_mse']:.4e} NRMSE@2τ={r3['nrmse_2tau']:.3f} "
                 f"Jaccard={r3['jaccard']:.3f} ({r3['wall_time']:.0f}s)")

        csv_writer.writerow({
            "method": r3["method"], "seed": seed,
            "data_seed": data_seed, "model_seed": model_seed,
            "one_step_mse": r3["one_step_mse"], "nrmse_2tau": r3["nrmse_2tau"],
            "jaccard": r3["jaccard"], "precision": r3["precision"],
            "recall": r3["recall"], "n_detected": r3["n_detected"],
            "n_tp": r3["n_tp"], "n_fp": r3["n_fp"],
            "detected": ";".join(r3["detected"]),
            "n_params": r3["n_params"], "train_epochs": r3["train_epochs"],
        })
        json.dump({"method": r3["method"], "seed": seed,
                    "pysr_log": r3.get("pysr_log", [])},
                   jsonl_file, default=str)
        jsonl_file.write("\n")
        csv_file.flush()

    csv_file.close()
    jsonl_file.close()

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE (markdown, pasteable into paper)
    # ═══════════════════════════════════════════════════════════════════
    log.info(f"\n\n{'='*90}")
    log.info("  RESULTS TABLE (markdown)")
    log.info(f"{'='*90}\n")

    header = (
        "| Method | Epochs | Params | 1-Step MSE | NRMSE@2τ_Λ | "
        "Jaccard ↑ | Precision ↑ | Recall ↑ |"
    )
    sep = "|" + "|".join(["---"] * 8) + "|"
    print(header)
    print(sep)

    for method_name in ["Deep-Koopman-KANDy", "MLP+Stable+PySR", "SINDy-AE+PySR"]:
        rs = all_results[method_name]
        mses = [r["one_step_mse"] for r in rs]
        nrmses = [r["nrmse_2tau"] for r in rs]
        jacs = [r["jaccard"] for r in rs]
        precs = [r["precision"] for r in rs]
        recs = [r["recall"] for r in rs]
        epochs = rs[0]["train_epochs"]
        params = rs[0]["n_params"]

        row = (
            f"| {method_name} | {epochs} | {params:,} | "
            f"${np.mean(mses):.2e} \\pm {np.std(mses):.0e}$ | "
            f"${np.mean(nrmses):.2f} \\pm {np.std(nrmses):.2f}$ | "
            f"${np.mean(jacs):.2f} \\pm {np.std(jacs):.2f}$ | "
            f"${np.mean(precs):.2f} \\pm {np.std(precs):.2f}$ | "
            f"${np.mean(recs):.2f} \\pm {np.std(recs):.2f}$ |"
        )
        print(row)

    # Sanity checks on final scores
    for method_name, rs in all_results.items():
        for r in rs:
            assert 0 <= r["recall"] <= 1, f"{method_name} recall out of range: {r['recall']}"

    elapsed = time.time() - t_start
    log.info(f"\n  Total time: {elapsed/3600:.1f} hours")
    log.info(f"  Results: {csv_path}")
    log.info(f"  Expressions: {jsonl_path}")

    # Flag discrepancies
    log.info(f"\n  DISCREPANCY NOTE: burn_in=500 used (matches code), "
             f"paper text may say 300. Verify and update paper.")


if __name__ == "__main__":
    main()
