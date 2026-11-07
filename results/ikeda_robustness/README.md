# Ikeda map: readout robustness study (NeurIPS reviewer EbT5-Q4)

Produced by `experiments/ikeda_robustness.py` (helpers: `experiments/readout_bases.py`,
`experiments/kan_attribution.py`). CPU, seeded, ~25 min end to end.

## Model

`[2,4,4]` **KANLinear** encoder/decoder + stable propagator `expm(K - LᵀL)`, seed 0,
400 epochs — i.e. exactly `experiments/ikeda_minimal.py`'s configuration, which is
what produced Table `tab:ikeda-levelset-decomposition`. **Not** PyKAN.
Base one-step MSE `9.818e-4`.

## Reproduction of `tab:ikeda-levelset-decomposition`

| dim | R²(h∘g) ours | paper | c_x² ours | paper | c_y² ours | paper | #active | FP | h affine ours | paper |
|-----|------|-------|-------|-------|-------|-------|---------|----|------|-------|
| z₀ | 0.912 | 0.915 | −0.0804 | −0.081 | −0.0114 | −0.011 | 9 | 3 | yes | **no** |
| z₁ | 0.941 | 0.941 | −0.0762 | −0.076 | −0.0499 | −0.050 | 10 | 4 | yes | yes |
| z₂ | 0.985 | 0.986 | +0.0121 | +0.012 | −0.0016 | — | 7 | 1 | no | no |
| z₃ | 0.967 | 0.968 | +0.0849 | +0.085 | +0.0200 | +0.020 | 7 | 2 | yes | yes |

Matched-quadratic ratio on z₁: **0.656** (paper 0.66). Reproduction is exact to
three decimals except the z₀ `h`-affine flag, which is a borderline
polynomial-vs-affine selection inside `recover_outer_function_from_raw`.

## Metrics

- `R2_radial` — R² of regressing the recovered `g` onto a degree-4 polynomial in
  r² = x²+y². Basis independent, so the `basis` axis is comparable.
- `spearman_g_r2` — |Spearman ρ(g, r²)|, monotone version of the above.
- `quad_ratio` — min(|c_x²|,|c_y²|)/max(...), the paper's own proxy. Polynomial
  basis only.
- `P` — design-matrix columns before Lasso; `t_decomp_s` — wall time of the
  decomposition step (Lasso + gradient identity + h recovery), per dim and total.

## Headline

- **λ (4 decades), degree D ≥ 3, and τ inside the Lorenz-calibrated grid** leave the
  radial/non-radial partition of the four latent coordinates completely unchanged
  (z₁, z₃ radial with R2_radial 0.91–0.97; z₀, z₂ non-radial at 0.08–0.27).
- What moves is density and fit: #active 10 → 6 and R²(h∘g) 0.976 → 0.908 as λ
  goes 1e-6 → 1e-2, with false positives falling 4 → 1.
- **The paper's 0.66 coefficient ratio is not itself λ-invariant**: it sweeps
  0.497 → 0.955 monotonically with λ. The *identity* of the radial coordinate is
  stable; its coefficient-ratio *value* is a function of the regularization.
- **τ**: the PyKAN attribution scores on this encoder have min 0.193, so
  τ ∈ {0.01, 0.03, 0.05} (and 0.1) prune nothing and the dictionary is bit-identical.
  Beyond that, pruning costs 3–10× one-step MSE and permutes which latent index
  carries the radial coordinate.

## Files

- `config.json`, `summary.json` — configuration and all aggregates
- `axis_lambda.csv`, `axis_degree.csv`, `axis_basis.csv`, `axis_tau.csv` — one row per setting
- `all_settings.csv` — every setting, every axis
- `per_dim_detail.csv` — one row per (setting, latent dim)
- `reproduction_per_dim.csv` — the paper-config decomposition
- `base_model_seed0.pt`, `model_tau*.pt` — checkpoints
- `run.log`
