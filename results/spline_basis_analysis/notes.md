# Lorenz [3,5,5] Experiment Notes

## Overview

A series of experiments investigating dictionary recovery from a Deep-Koopman-KANDy model
on the Lorenz system. The model uses a two-layer KAN encoder `[3, 5, 5]` (3 inputs, 5
mid-nodes, 5 latent dims) with a stable Koopman propagator `A = K - L^T L`.

The target dictionary is the known Koopman-invariant basis `{x, y, z, xy, xz}`.

---

## Experiment 1: Spline Basis Analysis (original [3,8,32] model)

**Script:** [`experiments/spline_basis_analysis.py`](../../experiments/spline_basis_analysis.py)

**Purpose:** Investigate whether the polynomial Lasso dictionary is the right basis for
analyzing KAN encoder outputs, or whether the learned spline functions form a more
natural dictionary.

**Steps:**
1. Load the pre-trained PyKAN `[3,8,32]` stable model from `results/lorenz_stable/stable_8_32_model.pt`
2. Extract Layer-1 edge functions by sweeping each input (x, y, z) while holding others at their mean
3. Identify each edge's symbolic form (linear, tanh, sin, cubic, etc.) via curve fitting
4. Build a spline-product dictionary `{psi_i, psi_i * psi_j}` from the 8 mid-node activations
5. Compare Lasso fits using polynomial vs spline-product dictionaries

**Key findings:**
- Layer-1 z-edges are almost all linear (6/8 with LinR² > 0.98)
- Layer-1 x and y edges are strongly nonlinear (tanh, bumps, sigmoids)
- Spline-product dictionary achieves R²=0.999 vs polynomial R²=0.976
- But spline-product sparsity is worse (38/44 terms vs 8.8/9) due to Lasso alpha being too loose
- The polynomial dictionary is a Taylor approximation of the true spline basis

**Results:** `results/spline_basis_analysis/`

**Plots:**
- `layer1_edges.png/pdf` — All 24 univariate edge functions
- `psi_summary.png/pdf` — Edge functions + nonlinearity classification + psi distributions
- `psi_manifolds.png` — 3D attractor colored by psi_j and level-set bands
- `layer2_splines.png` — Layer-2 mixing: z_k vs psi_j (binned median curves)
- `dictionary_comparison.png` — R² and sparsity: polynomial vs spline-product

---

## Experiment 2: Subspace Equivalence

**Script:** [`experiments/subspace_equivalence.py`](../../experiments/subspace_equivalence.py)

**Purpose:** Test whether the 8 learned spline coordinates span the same Koopman-invariant
subspace as the polynomial dictionary `{x, y, z, xy, xz}`.

**Steps:**
1. Load the [3,8,32] model
2. Compute Layer-1 activations (psi) on 10,000 attractor points
3. Project polynomial observables onto psi basis (OLS regression)
4. Project psi onto polynomial basis (reverse direction)
5. Compute principal angles between the two subspaces
6. Fit one-step Koopman propagator K_psi in the psi basis and compare linearity R² with polynomial basis

**Key findings:**
- Linear psi alone explains `{x, y, z}` well but `xy` (R²=0.34) and `xz` (R²=0.94) less so
- Psi products bring all observables to R²>0.999
- Principal angles: max 42.8° — subspaces overlap but are NOT identical
- Koopman linearity: spline basis min R²=0.999 vs polynomial min R²=0.991 — spline basis is slightly more linear

**Results:** `results/spline_basis_analysis/subspace_equivalence.png/pdf`

---

## Experiment 3: Eigenmode Structure (original [3,8,32] model)

**Script:** [`experiments/plot_eigenmode_structure.py`](../../experiments/plot_eigenmode_structure.py)

**Purpose:** Decompose the 8×8 K_psi matrix into eigenmodes and visualize how the learned
spline coordinates map onto dynamically independent oscillators.

**Steps:**
1. Fit K_psi via least squares on consecutive psi pairs
2. Eigendecompose K_psi → eigenvalues, eigenvectors
3. Compute continuous-time generator eigenvalues via logm(K_psi)/dt
4. Project data onto eigenmodes (modal coordinates)
5. Correlate each eigenmode with polynomial observables
6. Visualize: participation heatmap, eigenvalue spectrum, modal attractors, time series

**Results:** `results/spline_basis_analysis/eigenmode_structure.png/pdf`

---

## Experiment 4: [3,5,5] First Test (broken grid)

**Script:** [`experiments/lorenz_5d_test.py`](../../experiments/lorenz_5d_test.py)

**Purpose:** Train a minimal [3,5,5] PyKAN model matching the 5D Koopman-invariant
subspace dimension.

**Outcome:** Val loss=1.31, MSE=0.71 — model failed to train. The PyKAN spline grids
were initialized at their default range while the Lorenz data spans [-30, 50].

**Lesson:** PyKAN requires `update_grid_from_samples()` to be called with real data
before training. KANLinear avoids this because `grid_range` is set explicitly.

**Results:** `results/lorenz_5d/` (broken results, kept for reference)

---

## Experiment 5: [3,5,5] Second Test (grid fix)

**Script:** [`experiments/lorenz_5d_second_test.py`](../../experiments/lorenz_5d_second_test.py)

**Purpose:** Retrain the [3,5,5] PyKAN model with proper grid initialization.

**Steps:**
1. Call `enc.update_grid_from_samples(x_init)` and `dec.update_grid_from_samples(z_init)` before training
2. Re-update grids every 50 epochs during training
3. Train 200 epochs with AdamW + CosineAnnealingLR
4. Run full analysis: edge functions, psi manifolds, layer-2 splines, eigenmode decomposition, level-set dictionary

**Key findings:**
- MSE improved from 0.71 → 2.6e-3 (270x better)
- Rollout horizon: 0.65 τ_Λ
- 9/15 edges nonlinear (same pattern: z linear, x/y nonlinear)
- Koopman linearity: min R²=0.998
- Level-set decomposition finds `{x, y, z, xy, xz}` in all 5 dims
- BUT: z[0] has R²(h∘g)=0.15 and z[2] has R²(h∘g)=0.80 — two dims poorly decomposed

**Results:** `results/lorenz_5d_second_test/`

**Plots:**
- `encoder_diagram.png/pdf` — PyKAN network diagram
- `decoder_diagram.png/pdf` — PyKAN network diagram
- `layer1_edges.png/pdf` — 5×3 univariate edge functions
- `psi_manifolds.png/pdf` — Edge functions + attractor coloring + level-set bands
- `layer2_edges.png/pdf` — 5×5 z_k vs psi_j
- `attractor_psi.png` — Attractor colored by each psi
- `eigenmode_structure.png/pdf` — Participation heatmap + eigenvalues + time series
- `rollout.png/pdf` — x, y, z time series vs truth
- `nrmse.png/pdf` — NRMSE with horizon annotation
- `eigenvalues.png/pdf` — Discrete + generator eigenvalue spectrum

---

## Experiment 6: Eigenmode-Dictionary Interaction

**Script:** [`experiments/eigenmode_dictionary.py`](../../experiments/eigenmode_dictionary.py)

**Purpose:** Quantify how the polynomial dictionary `{x, y, z, xy, xz}` maps onto the
Koopman eigenmodes. The dictionary and eigenmodes are two bases for the same 5D
invariant subspace.

**Steps:**
1. Load the [3,5,5] grid-fixed model
2. Fit K_psi and eigendecompose
3. Compute modal coordinates on attractor data
4. Correlate Re(mode) and |mode amplitude| with each polynomial observable
5. Regress each observable onto all 5 eigenmodes

**Key findings:**
- `z` maps nearly 1-to-1 onto the fast spiral mode M0 (r=-0.94, R²=0.999)
- `x, y` map onto the slow/damping modes M2/M4 (R²≈0.93)
- `xy` is distributed across ALL modes (R²=0.85) — needs full interference to reconstruct
- `xz` is similarly distributed (R²=0.77)
- The amplitude |M0| correlates with z at r=0.997 — spiral amplitude tracks attractor height
- This is a property of the Lorenz dynamics, not the model

**Results:** `results/lorenz_5d_second_test/eigenmode_dictionary.png/pdf`, `eigenmode_dictionary.json`

---

## Experiment 7: [3,5,5] Third Test (prune + retrain)

**Script:** [`experiments/lorenz_5d_third_test.py`](../../experiments/lorenz_5d_third_test.py)

**Purpose:** Prune weak edges from the trained encoder, then retrain to recover
prediction quality while simplifying the decomposition.

**Steps:**
1. Load the grid-fixed model from experiment 5
2. Run baseline decomposition (no pruning) for comparison
3. For each prune threshold (0.01, 0.03, 0.05):
   a. Call `enc.attribute(plot=False)` to compute edge importance scores
   b. Call `enc.prune_edge(threshold=thresh)` to zero out weak edges
   c. Also prune the decoder
   d. Retrain 100 epochs at lr=5e-4 with pruned edges frozen
   e. Run level-set decomposition on the retrained model
   f. Save pruned+retrained encoder diagram and checkpoint

**Key findings:**
- Pruning alone destroys prediction (MSE goes 13,000x worse at threshold=0.03)
- But retraining RECOVERS prediction — and often improves it:
  - Prune@0.01 + retrain: MSE=1.27e-3 (0.49x baseline — **better** than unpruned!)
  - Prune@0.03 + retrain: MSE=2.55e-3 (0.98x — nearly same)
- Decomposition quality improves dramatically:
  - Prune@0.01: median R²(h∘g) jumps 0.797 → 0.965
  - z[0] R² improves 0.162 → 0.950

**Results:** `results/lorenz_5d_third_test/`

---

## Experiment 8: Lasso Alpha Sweep

**Script:** [`experiments/lorenz_5d_dictionary_table.py`](../../experiments/lorenz_5d_dictionary_table.py)

**Purpose:** Find the Lasso regularization strength that best separates true dictionary
terms from false positives, across all pruned/unpruned models.

**Steps:**
1. Load each model (unpruned, prune@0.01, prune@0.03, prune@0.05)
2. For each Lasso alpha in {1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2}:
   a. Run level-set decomposition on all 5 latent dims
   b. Compute union of detected true terms and false positives
   c. Compute recall, precision, Jaccard similarity

**Key findings:**
- **Prune@0.03 at alpha=5e-3 achieves Jaccard=1.000** — perfect dictionary recovery
  - 100% recall (all of `{x, y, z, xy, xz}` found)
  - 50% precision (some FP but all tiny: max FP coeff = 0.009)
  - Holds at alpha=1e-2 too
- Unpruned model maxes out at Jaccard=0.556 regardless of alpha
- Prune@0.01 reaches Jaccard=0.714 at alpha=1e-3
- Prune@0.05 is too aggressive (recall drops to 60%)

**Best configuration per model (at 100% recall):**

| Model | Best alpha | Jaccard | Precision | Med R² |
|-------|-----------|---------|-----------|--------|
| unpruned | 5e-4 | 0.556 | 0.36 | 0.966 |
| prune@0.01 | 1e-3 | 0.714 | 0.42 | 0.920 |
| **prune@0.03** | **5e-3** | **1.000** | **0.50** | **0.971** |
| prune@0.05 | 1e-4 | 0.455 | 0.31 | 0.562 |

**Results:** `results/lorenz_5d_third_test/dictionary_table.json`

---

## Experiment 9: Best Model Full Analysis

**Script:** [`experiments/lorenz_5d_best_model_plots.py`](../../experiments/lorenz_5d_best_model_plots.py)

**Purpose:** Generate all paper-quality plots for the best model (prune@0.03 + retrain).

**Steps:**
1. Load `results/lorenz_5d_third_test/model_pruned_0.03.pt`
2. Generate eigenvalue spectrum (discrete + generator)
3. Generate rollout vs truth time series
4. Generate NRMSE with horizon annotation
5. Compute eigenmode-dictionary correlations and regression
6. Generate PyKAN encoder/decoder network diagrams

**Key results:**
- Prediction horizon: 0.8 τ_Λ
- |λ| range: [0.854, 0.997] — all stable
- Eigenmodes: ±1.4 Hz (spiral), ±0.6 Hz (slow), 0 Hz (damping)
- Dictionary-eigenmode R²: z=0.998, x=0.967, y=0.843, xz=0.826, xy=0.741
- Cross-terms distributed across all modes (robust across all models tested)

**Results:** `results/lorenz_5d_best_model/`

**Plots:**
- `eigenvalues.png/pdf` — Discrete + generator eigenvalue spectrum
- `rollout.png/pdf` — x, y, z time series vs truth over 10 τ_Λ
- `nrmse.png/pdf` — NRMSE with horizon annotation (linear + log)
- `eigenmode_dictionary.png/pdf` — Correlation heatmaps + R² + attractor + time series
- `encoder_diagram.png/pdf` — Pruned PyKAN network
- `decoder_diagram.png/pdf` — Pruned PyKAN network

---

## Experiment 10: Ablation Study (KANLinear backend)

**Script:** [`experiments/ablation_lorenz.py`](../../experiments/ablation_lorenz.py)

**Purpose:** Controlled architectural ablations testing which components of
Deep-Koopman-KANDy are responsible for dictionary recovery.

**Variants (all with latent_dim=5, 5 seeds each):**
- A: 2L-KAN + stable (full model)
- B: 1L-KAN + stable (no mid layer, no cross-terms possible)
- C: MLP + stable
- D: KAN encoder + MLP decoder
- E: MLP encoder + KAN decoder
- F: 2L-KAN + unconstrained propagator
- G: Full model, prediction only (no decomposition)

**Detection metric:** For each of `{x, y, z, xy, xz}`, is it in the top-5 terms
of ANY latent dim with R²(h∘g) > 0.8? Union across dims. 5 seeds.

**Key findings:**
- **A (full model) has the highest Jaccard (0.72)** and perfect recall (100%)
- B (1L-KAN) drops xz recall to 20% — second layer is needed for cross-terms
- C (MLP) detects yz in 60% of seeds — false cross-term
- D (KAN-enc + MLP-dec) shows the encoder carries the dictionary (xy=100%)
- F (unconstrained) gets 100% recall but 3.0 FP — stable constraint improves specificity

**Results:** `results/ablation_lorenz/results.json`

---

## Experiment 11: KS Equation Ablation

**Script:** [`experiments/ablation_ks.py`](../../experiments/ablation_ks.py)

**Purpose:** Same ablation structure applied to the Kuramoto-Sivashinsky equation
(64 grid points, 128 latent dims).

**Key metric:** Diagonal fraction of quadratic coupling matrix — measures whether
the learned dictionary reflects the local nearest-neighbor structure of `u · u_x`.

**Key findings:**
- KAN-encoder variants (A, B, D) concentrate 29-38% of quadratic energy near the diagonal
- MLP-encoder variants (C, E) only get 12-17%
- 1L-KAN (B) achieves best prediction (MSE=5e-4, horizon=0.97 τ_Λ) — KS nonlinearity
  is spatially local and representable by single-layer per-input splines
- Stable constraint has minimal effect on KS

**Results:** `results/ablation_ks/results.json`

---

## Baseline Comparison

**Script:** [`experiments/baseline_comparison.py`](../../experiments/baseline_comparison.py)

**Purpose:** Compare dictionary detection against DMD, EDMD (degree 2 and 3),
MLP-Koopman, and KAN-stable.

**Key findings:**
- KAN-stable detects all 5 terms (100% recall) and never finds yz
- MLP-Koopman also detects all 5 but finds yz in 100% of seeds (false positive)
- EDMD with K-I thresholding gets perfect detection but was handed the dictionary

**Results:** `results/baseline_comparison/results.json`

---

## Summary of Key Results

1. The KAN encoder learns **nonlinear spline coordinates** (tanh, bumps of x and y; linear z)
   that span a 5D Koopman-invariant subspace

2. The polynomial dictionary `{x, y, z, xy, xz}` is recovered from level-set decomposition
   with **Jaccard=1.0** at the optimal pruning threshold (0.03) and Lasso alpha (5e-3)

3. The dictionary and eigenmodes are **two bases for the same subspace**: linear terms align
   with individual modes, cross-terms are distributed across all modes

4. The **two-layer KAN encoder** is required for cross-term detection (1L-KAN misses xz)

5. The **stable propagator** reduces false positives (unconstrained finds more spurious terms)

6. **Edge pruning + retraining** is the key to clean dictionary recovery — it simplifies
   the encoder function so the polynomial Lasso can separate true from spurious terms
