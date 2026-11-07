# Standard map (κ = 0.97): readout robustness study (NeurIPS reviewer EbT5-Q4)

Produced by `experiments/standard_map_robustness.py` (helper:
`experiments/readout_bases.py`). CPU, seeded, post-hoc only — no retraining.

## Model

Committed checkpoint `results/standard_map/standard_map_stable_32_128.pt`:
`[4,32,128]` **KANLinear** encoder/decoder on the (cos θ, sin θ, cos p, sin p)
embedding + stable propagator `expm(K − LᵀL)`. 128 latent coordinates.

## Reference atom sets

The paper's Eq. 13 lists 10 atoms and calls them "exactly e^{i(mθ+np)} with
|m|+|n| ≤ 2". **That enumeration is incomplete.** cos θ sin θ = ½sin 2θ and
cos p sin p = ½sin 2p are the *sine* parts of the (2,0) and (0,2) modes; the
*cosine* parts cos 2θ, cos 2p are missing. In the poly-in-trig chart those are
cx²/sx² and cy²/sy² (cos 2θ = 2cx² − 1 = 1 − 2sx²).

So every table scores against two reference sets:

- `union_*` — the literal Eq.-13 list (10 atoms)
- `full2_*` — the complete |m|+|n| ≤ 2 family (14 non-constant degree-2 monomials)

The four atoms the encoder recovers "extra" relative to Eq. 13 are exactly the
missing cos 2θ / cos 2p modes, so precision is 0.714 against the literal list and
**1.000** against the complete family.

`R2_eq13` in the CSVs is the basis-independent structural metric: R² of
OLS-regressing the recovered `g` onto the span of the complete |m|+|n| ≤ 2 family.

## Headline

- **λ is inert.** Across 1e-6 → 1e-2 at the paper degree D = 2: recall 1.000,
  precision 1.000, extras 0, R2_fourier2 1.000, R²(h∘g) median 0.976–0.977,
  #active 14 → 13. Perfect invariance.
- **D is the only knob that matters.** D = 1 under-fits (recall 0.286,
  R²(h∘g) 0.481). D = 2 is exact. D ≥ 3 keeps recall at 1.000 but precision
  collapses 1.000 → 0.412 → 0.203 → 0.126 as extras grow 0 → 20 → 55 → 97,
  roughly linearly in P, while R²(h∘g) gains only +0.023.
- **Basis-independent structure holds.** R2_fourier2 ≥ 0.978 in all three bases
  (poly-in-trig 1.000, Fourier |m|+|n| ≤ 3 0.981, periodic RBF 0.979), so `g`
  stays inside the |m|+|n| ≤ 2 Fourier span no matter which dictionary found it.
- **Tail caveat:** 19/128 latent coordinates have R²(h∘g) < 0.9 at the paper
  setting (min −0.62). These are the strongly damped coordinates; the paper
  reports no per-dim minimum.
- **Cost caveat:** coordinate-descent Lasso wall time is *not* monotone in P and
  is not reproducible run to run. Two identical runs of D = 4 (P = 70) took
  27.4 s/dim and 0.500 s/dim; D = 5 (P = 126) took 0.77 and 0.078 s/dim. The
  variance is a CD convergence stall on ill-conditioned columns, not a size
  effect. Quote the decomposition as seconds-scale per setting (D ≤ 3 is
  ≤ 0.5 s/dim in every run) and note the tail rather than the mean.

## Prune-τ axis

Not run — see `tau_axis_note` in `config.json`. The paper's τ protocol is PyKAN
`attribute()` + `prune_edge(τ)`; this encoder is KANLinear and exposes no PyKAN
activation graph. The τ axis is covered on Lorenz (`tab:prune-sweep`) and on
Ikeda (`results/ikeda_robustness/axis_tau.csv`, using the PyKAN attribution
score reimplemented for KANLinear in `experiments/kan_attribution.py`).

## Files

- `config.json`, `summary.json`
- `axis_lambda.csv` (both D = 2 and D = 3), `axis_degree.csv`, `axis_basis.csv`
- `all_settings.csv`, `per_dim_detail.csv`, `reproduction_per_dim.csv`
- `run.log`
