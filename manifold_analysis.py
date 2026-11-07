"""
Level Set + Gradient + Ideal analysis for KAN Deep Koopman manifolds.

Given a trained two-layer KAN encoder that maps (x, y, z) → latent,
decompose each latent dimension / mixing node as:

    f(x, y, z) = h( g(x, y, z) )

where:
    g  = sparse polynomial inner function  (recovered via Lasso on level sets)
    h  = 1-D outer function                (recovered via gradient formula)

The "ideal" of the decomposition is the set of monomials surviving Lasso
selection — these are the Koopman observables the network discovered.

References:
    research_materials/arc_length_clean.md
    research_materials/level_sets.md
"""

import numpy as np
import torch
from sklearn.linear_model import Lasso, LassoCV
from scipy.optimize import least_squares


# =====================================================================
# 1. Polynomial dictionary
# =====================================================================

def build_polynomial_dict(x, y, z, max_degree=3, include_trig=False):
    """Build a polynomial feature dictionary for 3 variables.

    Returns:
        M : (N, n_terms) design matrix
        names : list of str, monomial labels
        powers : list of (p, q, r) tuples
    """
    N = len(x)
    terms, names, powers = [], [], []

    for total_deg in range(max_degree + 1):
        for p in range(total_deg + 1):
            for q in range(total_deg - p + 1):
                r = total_deg - p - q
                term = (x ** p) * (y ** q) * (z ** r)
                terms.append(term)

                # Build readable name
                parts = []
                if p == 1: parts.append("x")
                elif p > 1: parts.append(f"x^{p}")
                if q == 1: parts.append("y")
                elif q > 1: parts.append(f"y^{q}")
                if r == 1: parts.append("z")
                elif r > 1: parts.append(f"z^{r}")
                name = "".join(parts) if parts else "1"
                names.append(name)
                powers.append((p, q, r))

    M = np.column_stack(terms)
    return M, names, powers


def eval_polynomial(x, y, z, coeffs, powers):
    """Evaluate sparse polynomial g(x,y,z) = Σ c_i x^p y^q z^r."""
    result = np.zeros_like(x, dtype=float)
    for c, (p, q, r) in zip(coeffs, powers):
        if abs(c) > 1e-15:
            result += c * (x ** p) * (y ** q) * (z ** r)
    return result


def grad_polynomial(x, y, z, coeffs, powers):
    """Compute ∇g analytically.  Returns (dg/dx, dg/dy, dg/dz)."""
    gx = np.zeros_like(x, dtype=float)
    gy = np.zeros_like(x, dtype=float)
    gz = np.zeros_like(x, dtype=float)
    for c, (p, q, r) in zip(coeffs, powers):
        if abs(c) < 1e-15:
            continue
        if p > 0:
            gx += c * p * (x ** max(p - 1, 0)) * (y ** q) * (z ** r)
        if q > 0:
            gy += c * (x ** p) * q * (y ** max(q - 1, 0)) * (z ** r)
        if r > 0:
            gz += c * (x ** p) * (y ** q) * r * (z ** max(r - 1, 0))
    return gx, gy, gz


# =====================================================================
# 1b. N-dimensional polynomial dictionary (generalizes 3D versions)
# =====================================================================

def build_polynomial_dict_nd(data, var_names, max_degree=3):
    """Build a polynomial feature dictionary for N variables.

    Args:
        data : (N, d) numpy array of state variables
        var_names : list of d variable name strings
        max_degree : max total polynomial degree

    Returns:
        M : (N, n_terms) design matrix
        names : list of str, monomial labels
        powers : list of d-tuples
    """
    from itertools import product as iproduct
    N, d = data.shape
    terms, names, powers_list = [], [], []

    # Enumerate all d-tuples (p1,...,pd) with sum <= max_degree
    def _enum_powers(d, max_deg):
        if d == 1:
            for p in range(max_deg + 1):
                yield (p,)
        else:
            for p in range(max_deg + 1):
                for rest in _enum_powers(d - 1, max_deg - p):
                    yield (p,) + rest

    for pw in _enum_powers(d, max_degree):
        term = np.ones(N, dtype=float)
        parts = []
        for dim, p in enumerate(pw):
            if p > 0:
                term *= data[:, dim] ** p
            if p == 1:
                parts.append(var_names[dim])
            elif p > 1:
                parts.append(f"{var_names[dim]}^{p}")
        name = "".join(parts) if parts else "1"
        terms.append(term)
        names.append(name)
        powers_list.append(pw)

    M = np.column_stack(terms)
    return M, names, powers_list


def eval_polynomial_nd(data, coeffs, powers):
    """Evaluate sparse polynomial g(x1,...,xd) = Σ c_i Π x_j^{p_j}."""
    N = data.shape[0]
    result = np.zeros(N, dtype=float)
    for c, pw in zip(coeffs, powers):
        if abs(c) > 1e-15:
            term = np.ones(N, dtype=float)
            for dim, p in enumerate(pw):
                if p > 0:
                    term *= data[:, dim] ** p
            result += c * term
    return result


def grad_polynomial_nd(data, coeffs, powers):
    """Compute ∇g analytically for d-dimensional polynomial.

    Returns:
        grads : (N, d) numpy array
    """
    N, d = data.shape
    grads = np.zeros((N, d), dtype=float)
    for c, pw in zip(coeffs, powers):
        if abs(c) < 1e-15:
            continue
        for dim in range(d):
            if pw[dim] > 0:
                dterm = np.ones(N, dtype=float) * c * pw[dim]
                for j in range(d):
                    if j == dim:
                        dterm *= data[:, j] ** max(pw[j] - 1, 0)
                    else:
                        if pw[j] > 0:
                            dterm *= data[:, j] ** pw[j]
                grads[:, dim] += dterm
    return grads


def recover_inner_function_nd(data, var_names, f_vals, max_degree=3,
                               alpha=None, sparsity_threshold=1e-4):
    """Recover sparse polynomial inner function for d-dimensional state.

    Same algorithm as recover_inner_function but generalized to d variables.
    """
    M, names, powers = build_polynomial_dict_nd(data, var_names, max_degree)

    col_mean = np.mean(M, axis=0)
    col_std = np.std(M, axis=0)
    col_std[col_std < 1e-12] = 1.0
    M_scaled = (M - col_mean) / col_std

    f_mean = np.mean(f_vals)
    f_std = np.std(f_vals)
    if f_std < 1e-12:
        f_std = 1.0
    f_scaled = (f_vals - f_mean) / f_std

    alpha_used = alpha if alpha is not None else 1e-5

    lasso = Lasso(alpha=alpha_used, max_iter=50000, fit_intercept=True)
    lasso.fit(M_scaled, f_scaled)

    coeffs_orig = f_std * lasso.coef_ / col_std
    intercept_orig = f_mean + f_std * (lasso.intercept_ - np.sum(lasso.coef_ * col_mean / col_std))

    # Fold intercept into constant term (all-zeros power tuple)
    zero_pw = tuple(0 for _ in range(data.shape[1]))
    const_idx = powers.index(zero_pw)
    coeffs_orig[const_idx] += intercept_orig

    mask = np.abs(coeffs_orig) > sparsity_threshold
    coeffs_sparse = coeffs_orig.copy()
    coeffs_sparse[~mask] = 0.0

    pred = M @ coeffs_sparse
    ss_res = np.sum((f_vals - pred) ** 2)
    ss_tot = np.sum((f_vals - f_vals.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    active = np.where(mask)[0]

    return {
        'coeffs': coeffs_sparse,
        'names': names,
        'powers': powers,
        'active': active,
        'active_names': [names[i] for i in active],
        'active_coeffs': coeffs_sparse[active],
        'r2': r2,
        'alpha': alpha_used,
    }


def full_decomposition_nd(model, x_data, target_idx, var_names,
                           target_type='latent', max_degree=3,
                           lasso_alpha=None, h_degree=4):
    """N-dimensional version of full_decomposition.

    Args:
        model : trained DeepKoopmanCT
        x_data : (N, d) tensor or numpy array
        target_idx : which dimension to analyze
        var_names : list of d variable name strings
        target_type : 'latent' or 'mixing'
        max_degree, lasso_alpha, h_degree : analysis parameters

    Returns:
        dict with inner, outer, ideal, reconstruction results
    """
    x_np = x_data.numpy() if isinstance(x_data, torch.Tensor) else x_data
    d = x_np.shape[1]

    label = f"z[{target_idx}]" if target_type == 'latent' else f"v[{target_idx}]"

    # Step 1: Evaluate f and ∇f via autograd
    x_t = x_data if isinstance(x_data, torch.Tensor) else torch.tensor(x_data, dtype=torch.float32)
    grad_f, f_vals = compute_network_gradient_nd(model, x_t, target_idx, target_type)

    # Step 2: Recover inner function via Lasso
    inner = recover_inner_function_nd(x_np, var_names, f_vals,
                                       max_degree=max_degree, alpha=lasso_alpha)

    # Step 3: Evaluate g and ∇g
    g_vals = eval_polynomial_nd(x_np, inner['coeffs'], inner['powers'])
    grad_g_mat = grad_polynomial_nd(x_np, inner['coeffs'], inner['powers'])

    # Step 4: Recover outer function (uses dot product / norm² formula)
    grad_g_tuple = tuple(grad_g_mat[:, i] for i in range(d))
    dot = np.sum(grad_f * grad_g_mat, axis=1)
    norm_sq = np.sum(grad_g_mat**2, axis=1)
    outer = recover_outer_function_from_raw(dot, norm_sq, g_vals, f_vals,
                                             poly_degree=h_degree)

    # Step 5: Ideal analysis
    ideal = analyze_ideal(inner)

    # Step 6: Full reconstruction
    f_hat = eval_h(g_vals, outer)
    recon_mse = np.mean((f_vals - f_hat) ** 2)
    recon_r2 = 1.0 - np.sum((f_vals - f_hat) ** 2) / \
               max(np.sum((f_vals - f_vals.mean()) ** 2), 1e-12)

    return {
        'label': label,
        'target_idx': target_idx,
        'target_type': target_type,
        'f_vals': f_vals,
        'x_np': x_np,
        'inner': inner,
        'g_vals': g_vals,
        'grad_f': grad_f,
        'outer': outer,
        'ideal': ideal,
        'f_hat': f_hat,
        'recon_mse': recon_mse,
        'recon_r2': recon_r2,
        'g_recon_r2': inner['r2'],
    }


def compute_network_gradient_nd(model, x_data, target_idx, target_type='latent'):
    """Compute ∇f w.r.t. state via autograd. Works for any input dimension."""
    enc = model.ae.encoder
    chunk = 2000
    f_vals_list = []
    grad_list = []

    # Resolve the inner KAN encoder (handles PassthroughEncoder wrapper)
    kan_enc = getattr(enc, 'kan_encoder', enc)

    for i in range(0, len(x_data), chunk):
        xi = x_data[i:i+chunk].detach().clone().requires_grad_(True)
        if target_type == 'latent':
            fi = enc(xi)[:, target_idx]
        elif target_type == 'mixing':
            ui = kan_enc.kan1(xi)
            vi = kan_enc.kan2(ui)
            fi = vi[:, target_idx]
        elif target_type == 'relu_mixing':
            ui = kan_enc.kan1(xi)
            vi = kan_enc.kan2(ui)
            rvi = kan_enc.relu(vi)
            fi = rvi[:, target_idx]
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        fi_sum = fi.sum()
        fi_sum.backward()
        grad_list.append(xi.grad.detach().numpy())
        f_vals_list.append(fi.detach().numpy())

    return np.concatenate(grad_list, axis=0), np.concatenate(f_vals_list, axis=0)


# =====================================================================
# 2. Lasso inner function recovery
# =====================================================================

def recover_inner_function(x, y, z, f_vals, max_degree=3,
                           alpha=None, sparsity_threshold=1e-4):
    """Recover sparse inner polynomial g(x,y,z) via Lasso.

    Uses the level-set formulation: fit g so that g(x,y,z) ≈ f(x,y,z)
    with L1 penalty to enforce sparsity.  When h ≈ linear (as in the
    Lorenz latent dims), g directly approximates f and the surviving
    monomials are the Koopman observables.

    Two-stage approach:
      1. OLS fit → compute residual scale
      2. Lasso with alpha scaled to residual (ensures cross-terms survive)

    Args:
        x, y, z : (N,) state variables
        f_vals  : (N,) target function values
        max_degree : max total polynomial degree
        alpha : Lasso regularization strength (None → auto-select)
        sparsity_threshold : coefficients below this are zeroed

    Returns:
        result : dict with keys:
            'coeffs'   : full coefficient vector
            'names'    : monomial names
            'powers'   : (p,q,r) tuples
            'active'   : indices of surviving terms
            'active_names' : names of surviving terms
            'active_coeffs': coefficients of surviving terms
            'r2'       : R² of the fit
            'alpha'    : Lasso alpha used
    """
    M, names, powers = build_polynomial_dict(x, y, z, max_degree)

    # Standardize columns AND target for balanced Lasso
    col_mean = np.mean(M, axis=0)
    col_std = np.std(M, axis=0)
    col_std[col_std < 1e-12] = 1.0
    M_scaled = (M - col_mean) / col_std

    f_mean = np.mean(f_vals)
    f_std = np.std(f_vals)
    if f_std < 1e-12:
        f_std = 1.0
    f_scaled = (f_vals - f_mean) / f_std

    if alpha is None:
        # Use a very small alpha to preserve weak-but-real cross-terms.
        # On normalized data, 1e-5 keeps terms contributing ~0.1% of variance.
        alpha_used = 1e-5
    else:
        alpha_used = alpha

    lasso = Lasso(alpha=alpha_used, max_iter=50000, fit_intercept=True)
    lasso.fit(M_scaled, f_scaled)

    # Unscale coefficients back to original space
    # y_scaled = M_scaled @ coef + intercept
    # (f - f_mean)/f_std = ((M - col_mean)/col_std) @ coef + intercept
    # f = f_std * (M/col_std @ coef - col_mean/col_std @ coef + intercept) + f_mean
    # f = Σ_i (f_std * coef_i / col_std_i) * M_i + [f_mean + f_std*(intercept - Σ coef_i*col_mean_i/col_std_i)]
    coeffs_orig = f_std * lasso.coef_ / col_std
    intercept_orig = f_mean + f_std * (lasso.intercept_ - np.sum(lasso.coef_ * col_mean / col_std))

    # Fold intercept into the constant term (powers (0,0,0))
    const_idx = powers.index((0, 0, 0))
    coeffs_orig[const_idx] += intercept_orig

    # Sparsify
    mask = np.abs(coeffs_orig) > sparsity_threshold
    coeffs_sparse = coeffs_orig.copy()
    coeffs_sparse[~mask] = 0.0

    # R²
    pred = M @ coeffs_sparse
    ss_res = np.sum((f_vals - pred) ** 2)
    ss_tot = np.sum((f_vals - f_vals.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    active = np.where(mask)[0]

    return {
        'coeffs': coeffs_sparse,
        'names': names,
        'powers': powers,
        'active': active,
        'active_names': [names[i] for i in active],
        'active_coeffs': coeffs_sparse[active],
        'r2': r2,
        'alpha': alpha_used,
    }


# =====================================================================
# 3. Neural network gradient via autograd
# =====================================================================

def compute_network_gradient(model, x_data, target_idx, target_type='latent'):
    """Compute ∇f w.r.t. (x, y, z) via autograd.

    Args:
        model : DeepKoopmanCT
        x_data : (N, 3) tensor
        target_idx : which output dimension
        target_type : 'latent' (z_k), 'mixing' (v_k), or 'relu_mixing'

    Returns:
        grad_f : (N, 3) numpy array  [df/dx, df/dy, df/dz]
        f_vals : (N,) numpy array
    """
    enc = model.ae.encoder
    has_dense = hasattr(enc, 'dense') and hasattr(enc, 'relu')

    def _forward_target(enc, xi, target_idx, target_type):
        """Compute scalar target from encoder, handling both Dense and Pure KAN."""
        if target_type == 'latent':
            if has_dense:
                u = enc.kan1(xi)
                v = enc.kan2(u)
                rv = enc.relu(v)
                out = enc.dense(rv)
            else:
                out = enc(xi)
            return out[:, target_idx]
        elif target_type == 'mixing':
            u = enc.kan1(xi)
            v = enc.kan2(u)
            return v[:, target_idx]
        elif target_type == 'relu_mixing':
            u = enc.kan1(xi)
            v = enc.kan2(u)
            rv = enc.relu(v)
            return rv[:, target_idx]
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    # Compute gradient via backward in chunks for memory
    chunk = 2000
    f_vals_list = []
    grad_list = []

    for i in range(0, len(x_data), chunk):
        xi = x_data[i:i+chunk].detach().clone().requires_grad_(True)
        fi = _forward_target(enc, xi, target_idx, target_type)

        fi_sum = fi.sum()
        fi_sum.backward()
        grad_list.append(xi.grad.detach().numpy())
        f_vals_list.append(fi.detach().numpy())

    return np.concatenate(grad_list, axis=0), np.concatenate(f_vals_list, axis=0)


# =====================================================================
# 4. Outer function recovery via gradient formula
# =====================================================================

def recover_outer_function(grad_f, grad_g, g_vals, f_vals,
                           poly_degree=4, grad_norm_threshold=1e-2):
    """Recover h' and integrate to get h.

    h'(g(x,y,z)) = (∇f · ∇g) / ||∇g||²

    Strategy:
      1. Compute h' samples from gradient formula
      2. Fit h' as polynomial (binned-median for robustness)
      3. Integrate to get h
      4. If h∘g is worse than a simple affine h, fall back to affine

    Args:
        grad_f : (N, 3) gradient of network output
        grad_g : tuple (dg/dx, dg/dy, dg/dz), each (N,)
        g_vals : (N,) inner function values
        f_vals : (N,) network output values
        poly_degree : degree for h' polynomial fit
        grad_norm_threshold : filter points with small ||∇g||

    Returns:
        result dict
    """
    gx, gy, gz = grad_g

    # ∇f · ∇g
    dot = grad_f[:, 0] * gx + grad_f[:, 1] * gy + grad_f[:, 2] * gz

    # ||∇g||²
    norm_sq = gx**2 + gy**2 + gz**2

    # Filter small gradients (more aggressive percentile-based threshold)
    norm_thresh = max(grad_norm_threshold**2,
                      np.percentile(norm_sq, 5))
    valid = norm_sq > norm_thresh
    h_prime = dot[valid] / norm_sq[valid]
    g_valid = g_vals[valid]
    f_valid = f_vals[valid]

    # Filter outliers in h' (tighter: 3× IQR)
    h_prime_median = np.median(h_prime)
    h_prime_iqr = np.percentile(h_prime, 75) - np.percentile(h_prime, 25)
    if h_prime_iqr < 1e-10:
        h_prime_iqr = np.std(h_prime)
    inlier = np.abs(h_prime - h_prime_median) < 3 * max(h_prime_iqr, 0.1)
    h_prime = h_prime[inlier]
    g_valid = g_valid[inlier]
    f_valid = f_valid[inlier]

    # ----- Binned-median fit for robustness -----
    # Bin by g, take median h' per bin → smooth 1D curve, then fit polynomial
    n_bins = min(80, len(g_valid) // 20)
    g_center = np.median(g_valid)
    g_shifted = g_valid - g_center

    bin_edges = np.linspace(g_shifted.min(), g_shifted.max(), n_bins + 1)
    bin_centers, bin_medians = [], []
    for i in range(n_bins):
        mask_bin = (g_shifted >= bin_edges[i]) & (g_shifted < bin_edges[i+1])
        if mask_bin.sum() >= 5:
            bin_centers.append(0.5 * (bin_edges[i] + bin_edges[i+1]))
            bin_medians.append(np.median(h_prime[mask_bin]))
    bin_centers = np.array(bin_centers)
    bin_medians = np.array(bin_medians)

    # Fit h'(z) as polynomial on binned medians
    if len(bin_centers) > poly_degree + 1:
        V_bin = np.vander(bin_centers, N=poly_degree + 1, increasing=True)
        h_prime_coeffs, _, _, _ = np.linalg.lstsq(V_bin, bin_medians,
                                                    rcond=None)
    else:
        V = np.vander(g_shifted, N=poly_degree + 1, increasing=True)
        h_prime_coeffs, _, _, _ = np.linalg.lstsq(V, h_prime, rcond=None)

    # Integrate: h(z) = C + Σ b_k/(k+1) (z-g_center)^(k+1)
    h_coeffs_integrated = np.zeros(poly_degree + 2)
    for k in range(poly_degree + 1):
        h_coeffs_integrated[k + 1] = h_prime_coeffs[k] / (k + 1)

    # Set integration constant via median reference
    ref_idx = np.argmin(np.abs(g_shifted))
    g0 = g_valid[ref_idx]
    f0 = f_valid[ref_idx]

    # Evaluate h at all valid points
    V_int = np.vander(g_shifted, N=poly_degree + 2, increasing=True)
    h_reconstructed = V_int @ h_coeffs_integrated

    # Adjust constant so h(g0) = f0
    h_at_ref = np.vander(np.array([g0 - g_center]), N=poly_degree + 2,
                          increasing=True) @ h_coeffs_integrated
    C = f0 - h_at_ref[0]
    h_coeffs_integrated[0] = C
    h_reconstructed += C

    # ----- Affine fallback -----
    # If the polynomial h∘g is worse than a simple affine f ≈ a·g + b,
    # fall back to affine (this happens when g already ≈ f).
    A_aff = np.column_stack([g_valid, np.ones(len(g_valid))])
    c_aff, _, _, _ = np.linalg.lstsq(A_aff, f_valid, rcond=None)
    h_affine = c_aff[0] * g_valid + c_aff[1]

    ss_poly = np.sum((f_valid - h_reconstructed) ** 2)
    ss_affine = np.sum((f_valid - h_affine) ** 2)

    use_affine = ss_poly > ss_affine
    if use_affine:
        h_reconstructed = h_affine

    # Evaluate h' fit for plotting
    g_plot = np.linspace(g_valid.min(), g_valid.max(), 500)
    V_plot = np.vander(g_plot - g_center, N=poly_degree + 1, increasing=True)
    h_prime_fit = V_plot @ h_prime_coeffs

    # Evaluate h for plotting
    if use_affine:
        h_fit_plot = c_aff[0] * g_plot + c_aff[1]
    else:
        V_int_plot = np.vander(g_plot - g_center, N=poly_degree + 2,
                                increasing=True)
        h_fit_plot = V_int_plot @ h_coeffs_integrated

    # Full valid mask
    full_valid = np.zeros(len(grad_f), dtype=bool)
    valid_idx = np.where(valid)[0]
    full_valid[valid_idx[inlier]] = True

    return {
        'g_samples': g_valid,
        'h_prime_samples': h_prime,
        'h_prime_coeffs': h_prime_coeffs,
        'h_coeffs': h_coeffs_integrated,
        'g_center': g_center,
        'g0': g0,
        'f0': f0,
        'C': C,
        'h_reconstructed': h_reconstructed,
        'f_valid': f_valid,
        'g_plot': g_plot,
        'h_prime_fit': h_prime_fit,
        'h_fit_plot': h_fit_plot,
        'h_bin_centers': bin_centers,
        'h_bin_medians': bin_medians,
        'valid_mask': full_valid,
        'used_affine': use_affine,
        'affine_coeffs': c_aff,
    }


def recover_outer_function_from_raw(dot, norm_sq, g_vals, f_vals,
                                    poly_degree=4, grad_norm_threshold=1e-2):
    """Recover outer function from pre-computed ∇f·∇g and ||∇g||².

    Same algorithm as recover_outer_function but accepts raw dot products
    and squared norms directly (works for any dimension).
    """
    norm_thresh = max(grad_norm_threshold**2,
                      np.percentile(norm_sq, 5))
    valid = norm_sq > norm_thresh
    h_prime = dot[valid] / norm_sq[valid]
    g_valid = g_vals[valid]
    f_valid = f_vals[valid]

    h_prime_median = np.median(h_prime)
    h_prime_iqr = np.percentile(h_prime, 75) - np.percentile(h_prime, 25)
    if h_prime_iqr < 1e-10:
        h_prime_iqr = np.std(h_prime)
    inlier = np.abs(h_prime - h_prime_median) < 3 * max(h_prime_iqr, 0.1)
    h_prime = h_prime[inlier]
    g_valid = g_valid[inlier]
    f_valid = f_valid[inlier]

    n_bins = min(80, len(g_valid) // 20)
    g_center = np.median(g_valid)
    g_shifted = g_valid - g_center

    bin_edges = np.linspace(g_shifted.min(), g_shifted.max(), n_bins + 1)
    bin_centers, bin_medians = [], []
    for i in range(n_bins):
        mask_bin = (g_shifted >= bin_edges[i]) & (g_shifted < bin_edges[i+1])
        if mask_bin.sum() >= 5:
            bin_centers.append(0.5 * (bin_edges[i] + bin_edges[i+1]))
            bin_medians.append(np.median(h_prime[mask_bin]))
    bin_centers = np.array(bin_centers)
    bin_medians = np.array(bin_medians)

    if len(bin_centers) > poly_degree + 1:
        V_bin = np.vander(bin_centers, N=poly_degree + 1, increasing=True)
        h_prime_coeffs, _, _, _ = np.linalg.lstsq(V_bin, bin_medians, rcond=None)
    else:
        V = np.vander(g_shifted, N=poly_degree + 1, increasing=True)
        h_prime_coeffs, _, _, _ = np.linalg.lstsq(V, h_prime, rcond=None)

    h_coeffs_integrated = np.zeros(poly_degree + 2)
    for k in range(poly_degree + 1):
        h_coeffs_integrated[k + 1] = h_prime_coeffs[k] / (k + 1)

    ref_idx = np.argmin(np.abs(g_shifted))
    g0 = g_valid[ref_idx]
    f0 = f_valid[ref_idx]

    V_int = np.vander(g_shifted, N=poly_degree + 2, increasing=True)
    h_reconstructed = V_int @ h_coeffs_integrated

    h_at_ref = np.vander(np.array([g0 - g_center]), N=poly_degree + 2,
                          increasing=True) @ h_coeffs_integrated
    C = f0 - h_at_ref[0]
    h_coeffs_integrated[0] = C
    h_reconstructed += C

    A_aff = np.column_stack([g_valid, np.ones(len(g_valid))])
    c_aff, _, _, _ = np.linalg.lstsq(A_aff, f_valid, rcond=None)
    h_affine = c_aff[0] * g_valid + c_aff[1]

    ss_poly = np.sum((f_valid - h_reconstructed) ** 2)
    ss_affine = np.sum((f_valid - h_affine) ** 2)

    use_affine = ss_poly > ss_affine
    if use_affine:
        h_reconstructed = h_affine

    g_plot = np.linspace(g_valid.min(), g_valid.max(), 500)
    V_plot = np.vander(g_plot - g_center, N=poly_degree + 1, increasing=True)
    h_prime_fit = V_plot @ h_prime_coeffs

    if use_affine:
        h_fit_plot = c_aff[0] * g_plot + c_aff[1]
    else:
        V_int_plot = np.vander(g_plot - g_center, N=poly_degree + 2, increasing=True)
        h_fit_plot = V_int_plot @ h_coeffs_integrated

    full_valid = np.zeros(len(f_vals), dtype=bool)
    valid_idx = np.where(valid)[0]
    full_valid[valid_idx[inlier]] = True

    return {
        'g_samples': g_valid,
        'h_prime_samples': h_prime,
        'h_prime_coeffs': h_prime_coeffs,
        'h_coeffs': h_coeffs_integrated,
        'g_center': g_center,
        'g0': g0, 'f0': f0, 'C': C,
        'h_reconstructed': h_reconstructed,
        'f_valid': f_valid,
        'g_plot': g_plot,
        'h_prime_fit': h_prime_fit,
        'h_fit_plot': h_fit_plot,
        'h_bin_centers': bin_centers,
        'h_bin_medians': bin_medians,
        'valid_mask': full_valid,
        'used_affine': use_affine,
        'affine_coeffs': c_aff,
    }


def eval_h(g_vals, h_result):
    """Evaluate the recovered outer function h at new g values."""
    if h_result.get('used_affine', False):
        c = h_result['affine_coeffs']
        return c[0] * g_vals + c[1]
    g_shifted = g_vals - h_result['g_center']
    V = np.vander(g_shifted, N=len(h_result['h_coeffs']), increasing=True)
    return V @ h_result['h_coeffs']


# =====================================================================
# 5. Ideal analysis
# =====================================================================

def analyze_ideal(inner_result, threshold_frac=0.01):
    """Analyze the ideal structure of the inner function.

    The "ideal" here means: which monomials survive Lasso selection?
    These are the generators of the polynomial ideal defining the
    Koopman observable.

    For example, if g ≈ c₁·xy + c₂·xz + c₃·z + c₀, the ideal
    generators are {xy, xz, z, 1} — these are the observables
    the network discovered.

    Returns:
        dict with ideal analysis
    """
    coeffs = inner_result['active_coeffs']
    names = inner_result['active_names']
    powers = [inner_result['powers'][i] for i in inner_result['active']]

    if len(coeffs) == 0:
        return {'generators': [], 'description': 'empty ideal'}

    # Sort by absolute coefficient magnitude
    order = np.argsort(np.abs(coeffs))[::-1]
    max_coeff = np.abs(coeffs[order[0]])

    generators = []
    for i in order:
        frac = np.abs(coeffs[i]) / max_coeff
        if frac < threshold_frac:
            continue
        generators.append({
            'name': names[i],
            'coeff': float(coeffs[i]),
            'power': powers[i],
            'relative_strength': float(frac),
        })

    # Classify the type of observable
    non_const = [g for g in generators if g['name'] != '1']
    total_degree = max((sum(g['power']) for g in non_const), default=0)
    n_mixed = sum(1 for g in non_const
                  if sum(1 for p in g['power'] if p > 0) > 1)

    if n_mixed > 0:
        obs_type = "multivariate (cross-terms present)"
    elif total_degree > 1:
        obs_type = "polynomial (no cross-terms)"
    elif total_degree == 1:
        obs_type = "linear"
    else:
        obs_type = "constant"

    # Build symbolic expression
    parts = []
    for g in generators:
        c = g['coeff']
        n = g['name']
        if n == '1':
            parts.append(f"{c:+.4f}")
        else:
            parts.append(f"{c:+.4f}·{n}")
    expression = " ".join(parts)

    return {
        'generators': generators,
        'type': obs_type,
        'expression': expression,
        'n_active': len(generators),
        'n_cross_terms': n_mixed,
    }


# =====================================================================
# 6. Full pipeline
# =====================================================================

def full_decomposition(model, x_data, target_idx, target_type='latent',
                       max_degree=3, lasso_alpha=None, h_degree=4):
    """Run the complete level-set + gradient + ideal pipeline.

    Args:
        model : trained DeepKoopmanCT
        x_data : (N, 3) tensor of Lorenz states
        target_idx : which dimension to analyze
        target_type : 'latent' or 'mixing'
        max_degree : polynomial degree for inner function
        lasso_alpha : Lasso penalty (None → cross-validate)
        h_degree : polynomial degree for outer function fit

    Returns:
        dict with all results: inner, outer, ideal, reconstruction
    """
    x_np = x_data.numpy() if isinstance(x_data, torch.Tensor) else x_data
    x_, y_, z_ = x_np[:, 0], x_np[:, 1], x_np[:, 2]

    label = f"z[{target_idx}]" if target_type == 'latent' else f"v[{target_idx}]"

    # Step 1: Evaluate f and ∇f
    grad_f, f_vals = compute_network_gradient(
        model, x_data if isinstance(x_data, torch.Tensor)
        else torch.tensor(x_data, dtype=torch.float32),
        target_idx, target_type)

    # Step 2: Recover inner function via Lasso
    inner = recover_inner_function(x_, y_, z_, f_vals,
                                    max_degree=max_degree, alpha=lasso_alpha)

    # Step 3: Evaluate g and ∇g
    g_vals = eval_polynomial(x_, y_, z_, inner['coeffs'], inner['powers'])
    grad_g = grad_polynomial(x_, y_, z_, inner['coeffs'], inner['powers'])

    # Step 4: Recover outer function
    outer = recover_outer_function(grad_f, grad_g, g_vals, f_vals,
                                    poly_degree=h_degree)

    # Step 5: Ideal analysis
    ideal = analyze_ideal(inner)

    # Step 6: Full reconstruction f̂ = h(g(x,y,z))
    f_hat = eval_h(g_vals, outer)
    recon_mse = np.mean((f_vals - f_hat) ** 2)
    recon_r2 = 1.0 - np.sum((f_vals - f_hat) ** 2) / \
               max(np.sum((f_vals - f_vals.mean()) ** 2), 1e-12)

    # Also: direct polynomial reconstruction (g alone, no h)
    g_recon_r2 = inner['r2']

    return {
        'label': label,
        'target_idx': target_idx,
        'target_type': target_type,
        'f_vals': f_vals,
        'x_np': x_np,
        'inner': inner,
        'g_vals': g_vals,
        'grad_f': grad_f,
        'grad_g': grad_g,
        'outer': outer,
        'ideal': ideal,
        'f_hat': f_hat,
        'recon_mse': recon_mse,
        'recon_r2': recon_r2,
        'g_recon_r2': g_recon_r2,
    }
