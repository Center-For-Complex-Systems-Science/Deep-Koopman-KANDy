"""
Readout-basis library + generic level-set decomposition for the robustness study
(NeurIPS reviewer EbT5-Q4).

The paper's level-set decomposition (`manifold_analysis.full_decomposition_nd`)
hard-codes a *polynomial* readout dictionary.  The reviewer asks how sensitive
the recovered dictionaries are to the choice of readout basis, so we need to be
able to swap the dictionary while keeping everything else (Lasso, gradient
identity for h, sparsification, R^2 accounting) byte-identical to the paper
pipeline.

Design
------
A `Basis` bundles three things evaluated on the SAME chart in which the network
gradient is taken:

    M      : (N, P) design matrix
    names  : P readable atom labels
    dM     : (N, P, d) per-atom gradient w.r.t. the chart coordinates

`decompose_with_basis` then runs exactly the paper's recipe:

    1. standardize columns + target, Lasso(alpha), unscale, fold intercept
    2. hard-threshold |coeff| < sparsity_threshold  -> active set
    3. g = M @ c,  grad_g = sum_i c_i dM[:, i, :]
    4. h' = (grad_f . grad_g) / ||grad_g||^2, binned-median polynomial fit,
       integrate, affine fallback     (manifold_analysis.recover_outer_function_from_raw)
    5. R^2(g), R^2(h o g)

Charts
------
Ikeda: the chart is the state (x, y) itself for every basis.
Standard map: the polynomial basis lives in the trig chart (cx, sx, cy, sy) --
this is what turns degree-2 monomials into Fourier atoms and is the paper's
protocol -- while the Fourier and RBF bases live in the angle chart (theta, p).
The gradient identity h' = <grad f, grad g>/||grad g||^2 is chart-covariant as
long as f and g are differentiated in the *same* chart, so both are valid; the
scripts record which chart each row used.
"""

import numpy as np
from sklearn.linear_model import Lasso, LassoLars

from manifold_analysis import recover_outer_function_from_raw


# =====================================================================
# Basis container
# =====================================================================

class Basis:
    """A readout dictionary evaluated on a fixed chart.

    Attributes:
        M     : (N, P) design matrix
        names : list of P atom labels
        dM    : (N, P, d) gradient of each atom w.r.t. the d chart coordinates
        const_idx : index of the constant atom (or None)
    """

    def __init__(self, M, names, dM, const_idx=None):
        self.M = M
        self.names = list(names)
        self.dM = dM
        self.const_idx = const_idx

    @property
    def n_atoms(self):
        return self.M.shape[1]


# =====================================================================
# Polynomial basis (the paper default)
# =====================================================================

def _enum_powers(d, max_deg):
    if d == 1:
        for p in range(max_deg + 1):
            yield (p,)
    else:
        for p in range(max_deg + 1):
            for rest in _enum_powers(d - 1, max_deg - p):
                yield (p,) + rest


def poly_basis(data, var_names, max_degree=3):
    """Monomials of total degree <= max_degree in the chart coordinates.

    Naming matches `manifold_analysis.build_polynomial_dict_nd` exactly so that
    atom names are directly comparable with the paper's tables.
    """
    N, d = data.shape
    cols, names, dcols, const_idx = [], [], [], None

    for pw in _enum_powers(d, max_degree):
        term = np.ones(N)
        parts = []
        for dim, p in enumerate(pw):
            if p > 0:
                term = term * data[:, dim] ** p
            if p == 1:
                parts.append(var_names[dim])
            elif p > 1:
                parts.append(f"{var_names[dim]}^{p}")
        name = "".join(parts) if parts else "1"
        if name == "1":
            const_idx = len(names)

        grad = np.zeros((N, d))
        for dim in range(d):
            if pw[dim] == 0:
                continue
            dterm = np.full(N, float(pw[dim]))
            for j in range(d):
                if j == dim:
                    dterm = dterm * data[:, j] ** max(pw[j] - 1, 0)
                elif pw[j] > 0:
                    dterm = dterm * data[:, j] ** pw[j]
            grad[:, dim] = dterm

        cols.append(term)
        names.append(name)
        dcols.append(grad)

    M = np.column_stack(cols)
    dM = np.stack(dcols, axis=1)  # (N, P, d)
    return Basis(M, names, dM, const_idx)


# =====================================================================
# Ikeda: correctly-specified radial / trigonometric basis
# =====================================================================

def ikeda_radial_trig_basis(data):
    """Correctly-specified family for the Ikeda map.

    The true Ikeda observable dictionary is {x, y, cos t(r^2), sin t(r^2)}
    with t(r^2) = 0.4 - 6/(1 + r^2) and r^2 = x^2 + y^2, combined bilinearly.
    This basis contains those atoms plus the bilinear products the map actually
    forms, so the polynomial misspecification of the paper default is removed.
    """
    x, y = data[:, 0], data[:, 1]
    N = data.shape[0]
    r2 = x ** 2 + y ** 2
    t = 0.4 - 6.0 / (1.0 + r2)
    ct, st = np.cos(t), np.sin(t)
    # dt/dr2 = 6/(1+r2)^2 ; dr2/dx = 2x, dr2/dy = 2y
    dt_dr2 = 6.0 / (1.0 + r2) ** 2
    dt_dx, dt_dy = dt_dr2 * 2 * x, dt_dr2 * 2 * y

    ones = np.ones(N)
    zeros = np.zeros(N)

    atoms = []

    def add(name, val, gx, gy):
        atoms.append((name, val, gx, gy))

    add("1", ones, zeros, zeros)
    add("x", x, ones, zeros)
    add("y", y, zeros, ones)
    add("xy", x * y, y, x)
    add("r2", r2, 2 * x, 2 * y)
    add("cos_t", ct, -st * dt_dx, -st * dt_dy)
    add("sin_t", st, ct * dt_dx, ct * dt_dy)
    add("x*cos_t", x * ct, ct + x * (-st * dt_dx), x * (-st * dt_dy))
    add("x*sin_t", x * st, st + x * (ct * dt_dx), x * (ct * dt_dy))
    add("y*cos_t", y * ct, y * (-st * dt_dx), ct + y * (-st * dt_dy))
    add("y*sin_t", y * st, y * (ct * dt_dx), st + y * (ct * dt_dy))
    # second harmonic of the same foliation coordinate
    add("cos_2t", np.cos(2 * t), -2 * np.sin(2 * t) * dt_dx,
        -2 * np.sin(2 * t) * dt_dy)
    add("sin_2t", np.sin(2 * t), 2 * np.cos(2 * t) * dt_dx,
        2 * np.cos(2 * t) * dt_dy)

    names = [a[0] for a in atoms]
    M = np.column_stack([a[1] for a in atoms])
    dM = np.stack([np.column_stack([a[2], a[3]]) for a in atoms], axis=1)
    return Basis(M, names, dM, const_idx=names.index("1"))


# =====================================================================
# Gaussian RBF basis (Euclidean chart -- used for Ikeda)
# =====================================================================

def rbf_basis(data, n_per_dim=6, eps_scale=1.0, seed=0):
    """Isotropic Gaussian RBFs on a grid covering the data range, + constant.

    phi_c(u) = exp(-||u - c||^2 / (2 s^2)),  s = eps_scale * grid spacing.
    grad phi_c = -(u - c)/s^2 * phi_c
    """
    N, d = data.shape
    lo = np.percentile(data, 1, axis=0)
    hi = np.percentile(data, 99, axis=0)
    axes = [np.linspace(lo[j], hi[j], n_per_dim) for j in range(d)]
    grids = np.meshgrid(*axes, indexing="ij")
    centers = np.column_stack([gg.ravel() for gg in grids])  # (C, d)
    spacing = np.mean([(hi[j] - lo[j]) / max(n_per_dim - 1, 1) for j in range(d)])
    s = max(eps_scale * spacing, 1e-6)

    cols, names, dcols = [np.ones(N)], ["1"], [np.zeros((N, d))]
    for ci, c in enumerate(centers):
        diff = data - c[None, :]                    # (N, d)
        phi = np.exp(-np.sum(diff ** 2, axis=1) / (2 * s ** 2))
        cols.append(phi)
        names.append(f"rbf{ci}")
        dcols.append(-(diff / s ** 2) * phi[:, None])

    M = np.column_stack(cols)
    dM = np.stack(dcols, axis=1)
    return Basis(M, names, dM, const_idx=0)


# =====================================================================
# Standard map: direct Fourier features on the torus (angle chart)
# =====================================================================

def fourier_modes(max_order=3):
    """Canonical (m, n) index pairs with 1 <= |m| + |n| <= max_order.

    Canonical means m > 0, or (m == 0 and n > 0) -- this picks one member of
    each conjugate pair {(m,n), (-m,-n)} so cos/sin span the real 2-D mode.
    """
    modes = []
    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            s = abs(m) + abs(n)
            if s < 1 or s > max_order:
                continue
            if m > 0 or (m == 0 and n > 0):
                modes.append((m, n))
    return sorted(modes, key=lambda mn: (abs(mn[0]) + abs(mn[1]), mn))


def torus_fourier_basis(angles, max_order=3):
    """{cos(m*theta + n*p), sin(m*theta + n*p)} for 1 <= |m|+|n| <= max_order.

    Chart coordinates are the raw angles (theta, p).
    """
    th, p = angles[:, 0], angles[:, 1]
    N = angles.shape[0]
    cols, names, dcols = [np.ones(N)], ["1"], [np.zeros((N, 2))]
    for (m, n) in fourier_modes(max_order):
        arg = m * th + n * p
        c, s = np.cos(arg), np.sin(arg)
        cols.append(c)
        names.append(f"cos({m},{n})")
        dcols.append(np.column_stack([-m * s, -n * s]))
        cols.append(s)
        names.append(f"sin({m},{n})")
        dcols.append(np.column_stack([m * c, n * c]))
    M = np.column_stack(cols)
    dM = np.stack(dcols, axis=1)
    return Basis(M, names, dM, const_idx=0)


def torus_rbf_basis(angles, n_per_dim=8, kappa=4.0):
    """Periodic (von Mises product) RBFs on the torus, + constant.

    phi_c(th, p) = exp(kappa*(cos(th - th_c) - 1) + kappa*(cos(p - p_c) - 1))
    d/dth = -kappa*sin(th - th_c) * phi ;  likewise for p.
    """
    th, p = angles[:, 0], angles[:, 1]
    N = angles.shape[0]
    grid = np.linspace(0, 2 * np.pi, n_per_dim, endpoint=False)
    cols, names, dcols = [np.ones(N)], ["1"], [np.zeros((N, 2))]
    ci = 0
    for tc in grid:
        for pc in grid:
            dth, dp = th - tc, p - pc
            phi = np.exp(kappa * (np.cos(dth) - 1.0) + kappa * (np.cos(dp) - 1.0))
            cols.append(phi)
            names.append(f"vm{ci}")
            dcols.append(np.column_stack([-kappa * np.sin(dth) * phi,
                                          -kappa * np.sin(dp) * phi]))
            ci += 1
    M = np.column_stack(cols)
    dM = np.stack(dcols, axis=1)
    return Basis(M, names, dM, const_idx=0)


# =====================================================================
# Generic decomposition (paper recipe, arbitrary basis)
# =====================================================================

def decompose_with_basis(basis, f_vals, grad_f, alpha,
                         sparsity_threshold=1e-4, h_degree=3,
                         solver="cd", max_iter=50000):
    """Level-set decomposition f ~= h(g) with `basis` as the readout dictionary.

    Args:
        basis  : Basis (M, names, dM) on the same chart as grad_f
        f_vals : (N,) network output values
        grad_f : (N, d) network gradient in the chart
        alpha  : Lasso penalty (on standardized columns/target)
        solver : 'cd' (sklearn Lasso, coordinate descent -- the paper default)
                 or 'lars' (LassoLars, exact path; cross-check)

    Returns dict with inner (coeffs / active set / R^2(g)), outer (h), and
    R^2(h o g).
    """
    M, names, dM = basis.M, basis.names, basis.dM
    N, P = M.shape

    col_mean = M.mean(axis=0)
    col_std = M.std(axis=0)
    col_std[col_std < 1e-12] = 1.0
    M_scaled = (M - col_mean) / col_std

    f_mean = float(f_vals.mean())
    f_std = float(f_vals.std())
    if f_std < 1e-12:
        f_std = 1.0
    f_scaled = (f_vals - f_mean) / f_std

    if solver == "lars":
        reg = LassoLars(alpha=alpha, fit_intercept=True, max_iter=8000)
    else:
        reg = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=True)
    reg.fit(M_scaled, f_scaled)

    coeffs = f_std * reg.coef_ / col_std
    intercept = f_mean + f_std * (reg.intercept_
                                  - np.sum(reg.coef_ * col_mean / col_std))
    if basis.const_idx is not None:
        coeffs[basis.const_idx] += intercept
        offset = 0.0
    else:
        offset = intercept

    mask = np.abs(coeffs) > sparsity_threshold
    coeffs_sparse = np.where(mask, coeffs, 0.0)

    g_vals = M @ coeffs_sparse + offset
    ss_tot = np.sum((f_vals - f_vals.mean()) ** 2)
    r2_g = 1.0 - np.sum((f_vals - g_vals) ** 2) / max(ss_tot, 1e-12)

    grad_g = np.einsum("p,npd->nd", coeffs_sparse, dM)
    dot = np.sum(grad_f * grad_g, axis=1)
    norm_sq = np.sum(grad_g ** 2, axis=1)

    degenerate = (not np.isfinite(norm_sq).all()
                  or float(np.max(norm_sq)) < 1e-12
                  or float(np.std(g_vals)) < 1e-12)
    outer = None
    if degenerate:
        # g collapsed to a constant -- no foliation to read out.
        f_hat = np.full_like(f_vals, f_vals.mean())
        r2_hg = 0.0
        used_affine = True
    else:
        try:
            outer = recover_outer_function_from_raw(dot, norm_sq, g_vals, f_vals,
                                                    poly_degree=h_degree)
        except (IndexError, ValueError, np.linalg.LinAlgError):
            # Too few points survive the ||grad g|| / IQR filters for the
            # gradient identity; fall back to the affine h the paper's routine
            # would itself have selected.
            outer = None
        if outer is None:
            A_aff = np.column_stack([g_vals, np.ones(len(g_vals))])
            c_aff, *_ = np.linalg.lstsq(A_aff, f_vals, rcond=None)
            outer = {"used_affine": True, "affine_coeffs": c_aff,
                     "h_coeffs": np.array([c_aff[1], c_aff[0]]),
                     "g_center": 0.0, "fallback": True}
        from manifold_analysis import eval_h
        f_hat = eval_h(g_vals, outer)
        r2_hg = 1.0 - np.sum((f_vals - f_hat) ** 2) / max(ss_tot, 1e-12)
        used_affine = bool(outer.get("used_affine", False))

    active = np.where(mask)[0]
    # "dominant" = within 1% of the largest standardized contribution; this is
    # the same relative-strength filter analyze_ideal() uses.
    contrib = np.abs(coeffs_sparse) * col_std
    max_contrib = contrib.max() if contrib.size else 0.0
    dominant = np.array([i for i in active
                         if max_contrib > 0 and contrib[i] / max_contrib >= 0.01],
                        dtype=int)

    return {
        "coeffs": coeffs_sparse,
        "names": names,
        "active": active,
        "active_names": [names[i] for i in active],
        "active_coeffs": coeffs_sparse[active],
        "dominant_names": [names[i] for i in dominant],
        "n_active": int(mask.sum()),
        "n_dominant": int(len(dominant)),
        "r2_g": float(r2_g),
        "r2_hg": float(r2_hg),
        "used_affine": used_affine,
        "degenerate": bool(degenerate),
        "g_vals": g_vals,
        "f_vals": f_vals,
        "outer": outer,
        "alpha": float(alpha),
    }


# =====================================================================
# Structural-stability metrics (basis independent)
# =====================================================================

def r2_against_span(g_vals, ref_M):
    """R^2 of OLS-regressing g onto a reference span (constant included).

    This is the basis-independent test of "did we recover the same foliation
    coordinate?": if g still lives in the reference span, R^2 ~ 1 regardless of
    which dictionary produced it.
    """
    if np.std(g_vals) < 1e-12:
        return 0.0
    A = np.column_stack([np.ones(len(g_vals)), ref_M])
    coef, *_ = np.linalg.lstsq(A, g_vals, rcond=None)
    pred = A @ coef
    ss_res = np.sum((g_vals - pred) ** 2)
    ss_tot = np.sum((g_vals - g_vals.mean()) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def spearman_abs(a, b):
    """|Spearman rho| -- monotone association, insensitive to the outer h."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    if denom < 1e-12:
        return 0.0
    return float(abs(np.sum(ra * rb) / denom))


def set_metrics(detected, reference):
    """Union-level Jaccard / precision / recall against a reference atom set.

    Mirrors the Lorenz protocol in experiments/pykan_pruned_sweep.py.
    """
    detected = set(detected) - {"1"}
    reference = set(reference)
    tp = len(detected & reference)
    return {
        "recall": tp / max(len(reference), 1),
        "precision": tp / max(len(detected), 1),
        "jaccard": tp / max(len(detected | reference), 1),
        "n_extra": len(detected - reference),
        "n_detected": len(detected),
    }
