"""
Dictionary-viability diagnostics for the level-set + gradient decomposition.

Given a decomposition f(x) = h(g(x)), tests whether the scalar compositional
structure is geometrically consistent, numerically stable, and low-residual.

Metrics implemented:
    1. Gradient alignment score
    2. Orthogonal residual score
    3. Conditioning / denominator stability score
    4. Single-valuedness of h' as a function of g
    5. Sparse support stability across repeated runs
    6. Subspace stability across repeated runs (optional)
"""

import numpy as np
from itertools import combinations
from manifold_analysis import (
    build_polynomial_dict,
    recover_inner_function,
)


# =====================================================================
# 1. Gradient alignment score
# =====================================================================

def gradient_alignment(grad_f, grad_g, eps=1e-10):
    """Compute pointwise gradient alignment score.

    align(x) = |∇f · ∇g| / (||∇f|| ||∇g|| + eps)

    Args:
        grad_f : (N, d) gradient of network output
        grad_g : (N, d) gradient of inner polynomial
        eps : numerical stability constant

    Returns:
        dict with pointwise values and summary statistics
    """
    if isinstance(grad_g, tuple):
        grad_g = np.column_stack(grad_g)

    dot = np.sum(grad_f * grad_g, axis=1)
    norm_f = np.linalg.norm(grad_f, axis=1)
    norm_g = np.linalg.norm(grad_g, axis=1)

    align = np.abs(dot) / (norm_f * norm_g + eps)
    align = np.clip(align, 0.0, 1.0)

    return {
        'values': align,
        'mean': float(np.mean(align)),
        'median': float(np.median(align)),
        'p10': float(np.percentile(align, 10)),
        'p90': float(np.percentile(align, 90)),
        'std': float(np.std(align)),
    }


# =====================================================================
# 2. Orthogonal residual score
# =====================================================================

def orthogonal_residual(grad_f, grad_g, eps=1e-10):
    """Compute the component of ∇f not explained by scalar factorization.

    proj_coeff(x) = (∇f · ∇g) / (||∇g||² + eps)
    r_perp(x) = || ∇f - proj_coeff(x) ∇g ||
    r_perp_norm(x) = r_perp(x) / (||∇f|| + eps)

    Args:
        grad_f : (N, d) gradient of network output
        grad_g : (N, d) gradient of inner polynomial
        eps : numerical stability constant

    Returns:
        dict with pointwise and summary statistics
    """
    if isinstance(grad_g, tuple):
        grad_g = np.column_stack(grad_g)

    dot = np.sum(grad_f * grad_g, axis=1)
    norm_sq_g = np.sum(grad_g ** 2, axis=1)
    norm_f = np.linalg.norm(grad_f, axis=1)

    proj_coeff = dot / (norm_sq_g + eps)
    projected = proj_coeff[:, None] * grad_g
    residual_vec = grad_f - projected

    r_perp = np.linalg.norm(residual_vec, axis=1)
    r_perp_norm = r_perp / (norm_f + eps)

    return {
        'r_perp': r_perp,
        'r_perp_norm': r_perp_norm,
        'mean_r_perp': float(np.mean(r_perp)),
        'median_r_perp': float(np.median(r_perp)),
        'mean_r_perp_norm': float(np.mean(r_perp_norm)),
        'median_r_perp_norm': float(np.median(r_perp_norm)),
        'p90_r_perp_norm': float(np.percentile(r_perp_norm, 90)),
        'p95_r_perp_norm': float(np.percentile(r_perp_norm, 95)),
    }


# =====================================================================
# 3. Conditioning / denominator stability score
# =====================================================================

def conditioning_score(grad_f, grad_g, h_prime_samples=None, eps=1e-10):
    """Assess ill-conditioning of the chain-rule estimator.

    Tracks:
        - fraction of points with small ||∇g||²
        - distribution of ||∇g||²
        - condition ratio ||∇f|| / (||∇g|| + eps)
        - h' distribution statistics if provided

    Args:
        grad_f : (N, d) gradient of network output
        grad_g : (N, d) gradient of inner polynomial
        h_prime_samples : (M,) optional h' values from outer recovery
        eps : numerical stability constant

    Returns:
        dict with conditioning diagnostics
    """
    if isinstance(grad_g, tuple):
        grad_g = np.column_stack(grad_g)

    norm_f = np.linalg.norm(grad_f, axis=1)
    norm_g = np.linalg.norm(grad_g, axis=1)
    norm_sq_g = norm_g ** 2

    # Low-gradient fraction (multiple thresholds)
    thresh_5pct = np.percentile(norm_sq_g, 5)
    low_grad_frac_1e2 = float(np.mean(norm_sq_g < 1e-2))
    low_grad_frac_1e4 = float(np.mean(norm_sq_g < 1e-4))
    low_grad_frac_5pct = 0.05  # by definition

    # Condition ratio
    cond_ratio = norm_f / (norm_g + eps)

    result = {
        'norm_sq_g_mean': float(np.mean(norm_sq_g)),
        'norm_sq_g_median': float(np.median(norm_sq_g)),
        'norm_sq_g_p5': float(np.percentile(norm_sq_g, 5)),
        'norm_sq_g_p10': float(np.percentile(norm_sq_g, 10)),
        'low_grad_frac_1e2': low_grad_frac_1e2,
        'low_grad_frac_1e4': low_grad_frac_1e4,
        'cond_ratio_mean': float(np.mean(cond_ratio)),
        'cond_ratio_median': float(np.median(cond_ratio)),
        'cond_ratio_p95': float(np.percentile(cond_ratio, 95)),
        'cond_ratio_max_clipped': float(np.percentile(cond_ratio, 99)),
    }

    if h_prime_samples is not None and len(h_prime_samples) > 0:
        result['h_prime_mean'] = float(np.mean(h_prime_samples))
        result['h_prime_median'] = float(np.median(h_prime_samples))
        result['h_prime_std'] = float(np.std(h_prime_samples))
        result['h_prime_iqr'] = float(
            np.percentile(h_prime_samples, 75) -
            np.percentile(h_prime_samples, 25)
        )

    return result


# =====================================================================
# 4. Single-valuedness of h' as a function of g
# =====================================================================

def single_valuedness(g_vals, h_prime_vals, n_bins=50, min_bin_count=5):
    """Test whether h' is approximately single-valued as a function of g.

    Bins points by g, computes within-bin variance/IQR of h'.

    Args:
        g_vals : (M,) inner function values (valid points only)
        h_prime_vals : (M,) estimated h' values
        n_bins : number of bins
        min_bin_count : minimum points per bin to include

    Returns:
        dict with within-bin variance statistics
    """
    if len(g_vals) < n_bins * 2:
        n_bins = max(5, len(g_vals) // 10)

    bin_edges = np.linspace(np.min(g_vals), np.max(g_vals), n_bins + 1)

    within_bin_vars = []
    within_bin_iqrs = []
    within_bin_mads = []
    bin_counts = []
    bin_centers = []
    bin_medians = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (g_vals >= lo) & (g_vals <= hi)
        else:
            mask = (g_vals >= lo) & (g_vals < hi)

        n_in_bin = mask.sum()
        if n_in_bin < min_bin_count:
            continue

        hp_bin = h_prime_vals[mask]
        within_bin_vars.append(float(np.var(hp_bin)))
        iqr = float(np.percentile(hp_bin, 75) - np.percentile(hp_bin, 25))
        within_bin_iqrs.append(iqr)
        within_bin_mads.append(float(np.median(np.abs(hp_bin - np.median(hp_bin)))))
        bin_counts.append(int(n_in_bin))
        bin_centers.append(0.5 * (lo + hi))
        bin_medians.append(float(np.median(hp_bin)))

    within_bin_vars = np.array(within_bin_vars) if within_bin_vars else np.array([np.nan])
    within_bin_iqrs = np.array(within_bin_iqrs) if within_bin_iqrs else np.array([np.nan])
    within_bin_mads = np.array(within_bin_mads) if within_bin_mads else np.array([np.nan])
    bin_counts = np.array(bin_counts) if bin_counts else np.array([0])

    # Weighted average by bin count
    total_count = bin_counts.sum()
    if total_count > 0:
        weights = bin_counts / total_count
        weighted_var = float(np.sum(weights * within_bin_vars))
    else:
        weighted_var = float('nan')

    return {
        'mean_within_bin_var': float(np.nanmean(within_bin_vars)),
        'median_within_bin_var': float(np.nanmedian(within_bin_vars)),
        'weighted_within_bin_var': weighted_var,
        'mean_within_bin_iqr': float(np.nanmean(within_bin_iqrs)),
        'median_within_bin_mad': float(np.nanmedian(within_bin_mads)),
        'n_valid_bins': int(len(within_bin_vars)),
        'bin_centers': np.array(bin_centers),
        'bin_medians': np.array(bin_medians),
        'bin_counts': bin_counts,
        'within_bin_vars': within_bin_vars,
    }


# =====================================================================
# 5. Sparse support stability across repeated runs
# =====================================================================

def support_stability(run_results, name_key='active_names',
                      coeff_key='active_coeffs'):
    """Compare recovered sparse supports across multiple runs.

    Args:
        run_results : list of inner-result dicts from repeated decompositions
        name_key : key for active term names in each result
        coeff_key : key for active term coefficients

    Returns:
        dict with selection frequencies, coefficient stats, Jaccard similarities
    """
    n_runs = len(run_results)
    if n_runs < 2:
        return {'n_runs': n_runs, 'note': 'need >= 2 runs for stability'}

    # Collect all unique terms and per-run info
    all_terms = set()
    run_supports = []
    run_coeff_maps = []

    for res in run_results:
        names = res[name_key] if isinstance(res[name_key], list) else list(res[name_key])
        coeffs = res[coeff_key] if isinstance(res[coeff_key], (list, np.ndarray)) else [res[coeff_key]]
        support = set(names)
        all_terms.update(support)
        run_supports.append(support)
        coeff_map = {}
        for n, c in zip(names, coeffs):
            coeff_map[n] = float(c)
        run_coeff_maps.append(coeff_map)

    all_terms = sorted(all_terms)

    # Selection frequency
    selection_freq = {}
    for term in all_terms:
        freq = sum(1 for s in run_supports if term in s) / n_runs
        selection_freq[term] = float(freq)

    # Coefficient statistics per term
    coeff_stats = {}
    for term in all_terms:
        vals = [cm[term] for cm in run_coeff_maps if term in cm]
        if len(vals) >= 2:
            signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in vals]
            sign_stability = float(abs(sum(signs)) / len(signs))
            coeff_stats[term] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'cv': float(np.std(vals) / (abs(np.mean(vals)) + 1e-15)),
                'sign_stability': sign_stability,
                'n_selected': len(vals),
            }
        elif len(vals) == 1:
            coeff_stats[term] = {
                'mean': float(vals[0]),
                'std': 0.0,
                'cv': 0.0,
                'sign_stability': 1.0,
                'n_selected': 1,
            }

    # Pairwise Jaccard similarity
    jaccards = []
    for i, j in combinations(range(n_runs), 2):
        inter = len(run_supports[i] & run_supports[j])
        union = len(run_supports[i] | run_supports[j])
        jacc = inter / union if union > 0 else 1.0
        jaccards.append(jacc)

    jaccard_matrix = np.ones((n_runs, n_runs))
    idx = 0
    for i, j in combinations(range(n_runs), 2):
        jaccard_matrix[i, j] = jaccards[idx]
        jaccard_matrix[j, i] = jaccards[idx]
        idx += 1

    return {
        'n_runs': n_runs,
        'all_terms': all_terms,
        'selection_freq': selection_freq,
        'coeff_stats': coeff_stats,
        'mean_jaccard': float(np.mean(jaccards)),
        'median_jaccard': float(np.median(jaccards)),
        'std_jaccard': float(np.std(jaccards)),
        'jaccard_matrix': jaccard_matrix,
    }


# =====================================================================
# 6. Subspace stability across repeated runs
# =====================================================================

def subspace_stability(run_results, all_names=None):
    """Compare recovered observable spans across runs using principal angles.

    Even when coordinate-level supports differ, the spanned subspace
    may be stable.

    Args:
        run_results : list of inner-result dicts
        all_names : shared monomial name list (if None, built from union)

    Returns:
        dict with principal angle statistics
    """
    n_runs = len(run_results)
    if n_runs < 2:
        return {'n_runs': n_runs, 'note': 'need >= 2 runs for subspace stability'}

    # Build coefficient vectors in a shared monomial basis
    if all_names is None:
        all_names_set = set()
        for res in run_results:
            all_names_set.update(res.get('names', []))
        all_names = sorted(all_names_set)

    name_to_idx = {n: i for i, n in enumerate(all_names)}
    d = len(all_names)

    coeff_vecs = []
    for res in run_results:
        vec = np.zeros(d)
        names = res.get('active_names', [])
        coeffs = res.get('active_coeffs', [])
        for n, c in zip(names, coeffs):
            if n in name_to_idx:
                vec[name_to_idx[n]] = float(c)
        coeff_vecs.append(vec)

    C = np.array(coeff_vecs)  # (n_runs, d)

    # Pairwise principal angles between 1D subspaces (just the angle between vectors)
    angles = []
    for i, j in combinations(range(n_runs), 2):
        vi, vj = C[i], C[j]
        ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
        if ni < 1e-12 or nj < 1e-12:
            angles.append(np.pi / 2)
            continue
        cos_angle = np.clip(np.dot(vi, vj) / (ni * nj), -1, 1)
        angles.append(float(np.arccos(abs(cos_angle))))

    # SVD of the coefficient matrix to check span dimension
    if C.shape[0] >= 2 and np.linalg.norm(C) > 1e-12:
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        # Fraction of variance in first k components
        total_var = np.sum(S ** 2)
        cumvar = np.cumsum(S ** 2) / max(total_var, 1e-12)
        effective_dim = int(np.searchsorted(cumvar, 0.95) + 1)
    else:
        S = np.array([0.0])
        effective_dim = 0

    return {
        'n_runs': n_runs,
        'mean_angle_rad': float(np.mean(angles)) if angles else float('nan'),
        'median_angle_rad': float(np.median(angles)) if angles else float('nan'),
        'mean_angle_deg': float(np.degrees(np.mean(angles))) if angles else float('nan'),
        'effective_dim': effective_dim,
        'singular_values': S.tolist(),
    }


# =====================================================================
# 7. Sparsity sweep: Lasso alpha → (n_terms, R²) Pareto front
# =====================================================================

def sparsity_sweep(x_data, f_vals, var_names=None, max_degree=3,
                   alphas=None, sparsity_threshold=1e-4):
    """Sweep Lasso alpha to produce the sparsity-accuracy tradeoff curve.

    The key discriminator: dictionary-friendly systems maintain high R²
    even at high sparsity (few active terms). Systems without a sparse
    dictionary need many terms for comparable fit.

    Args:
        x_data : (N, d) numpy array of state variables
        f_vals : (N,) target function values
        var_names : list of d variable name strings (None → 3D xyz)
        max_degree : polynomial degree
        alphas : list of Lasso alpha values to sweep
        sparsity_threshold : coefficients below this are zeroed

    Returns:
        dict with sweep results including Pareto front
    """
    if alphas is None:
        alphas = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4,
                  1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]

    sweep = []
    for alpha in alphas:
        inner = recover_inner_function(
            x_data[:, 0], x_data[:, 1], x_data[:, 2], f_vals,
            max_degree=max_degree, alpha=alpha,
            sparsity_threshold=sparsity_threshold)

        n_active = len(inner['active'])
        r2 = inner['r2']
        active_names = list(inner['active_names'])
        sweep.append({
            'alpha': float(alpha),
            'n_active': int(n_active),
            'r2': float(r2),
            'active_names': active_names,
        })

    # Find the "elbow": best R² with fewest terms
    # Specifically: smallest n_active such that R² > 0.9
    sparse_viable = [s for s in sweep if s['r2'] > 0.9]
    if sparse_viable:
        elbow = min(sparse_viable, key=lambda s: s['n_active'])
    else:
        # No alpha gives R² > 0.9 with any sparsity — still report best
        elbow = max(sweep, key=lambda s: s['r2'])

    # Compute sparsity ratio: n_active_at_elbow / total_possible_terms
    d = x_data.shape[1]
    # Total terms in degree-d polynomial
    from math import comb
    n_total_terms = comb(d + max_degree, max_degree)
    sparsity_ratio = elbow['n_active'] / n_total_terms if n_total_terms > 0 else 1.0

    return {
        'sweep': sweep,
        'elbow': elbow,
        'sparsity_ratio': float(sparsity_ratio),
        'n_total_terms': n_total_terms,
        'min_terms_r2_above_90': elbow['n_active'] if elbow['r2'] > 0.9 else None,
    }


# =====================================================================
# Main: compute all viability metrics for one decomposition
# =====================================================================

def compute_viability_metrics(decomp_result, run_sparsity_sweep=True,
                              var_names=None, max_degree=3):
    """Compute all dictionary-viability metrics from a decomposition result.

    Args:
        decomp_result : dict returned by full_decomposition or full_decomposition_nd
        run_sparsity_sweep : if True, run Lasso alpha sweep for sparsity analysis
        var_names : variable names for sparsity sweep (None → infer)
        max_degree : polynomial degree for sparsity sweep

    Returns:
        dict with all viability metrics
    """
    grad_f = decomp_result['grad_f']
    inner = decomp_result['inner']
    outer = decomp_result['outer']
    f_vals = decomp_result['f_vals']
    g_vals = decomp_result['g_vals']
    x_np = decomp_result['x_np']

    # Compute grad_g in (N, d) form
    if 'grad_g' in decomp_result:
        grad_g_raw = decomp_result['grad_g']
        if isinstance(grad_g_raw, tuple):
            grad_g = np.column_stack(grad_g_raw)
        else:
            grad_g = grad_g_raw
    else:
        from manifold_analysis import grad_polynomial
        grad_g_tuple = grad_polynomial(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                                        inner['coeffs'], inner['powers'])
        grad_g = np.column_stack(grad_g_tuple)

    # 1. Alignment
    align = gradient_alignment(grad_f, grad_g)

    # 2. Orthogonal residual
    orth = orthogonal_residual(grad_f, grad_g)

    # 3. Conditioning
    h_prime_samples = outer.get('h_prime_samples', None)
    cond = conditioning_score(grad_f, grad_g, h_prime_samples)

    # 4. Single-valuedness
    g_valid = outer.get('g_samples', None)
    h_prime = outer.get('h_prime_samples', None)
    if g_valid is not None and h_prime is not None and len(g_valid) > 10:
        sv = single_valuedness(g_valid, h_prime)
    else:
        sv = {'mean_within_bin_var': float('nan'), 'note': 'insufficient data'}

    # 7. Sparsity sweep
    if run_sparsity_sweep:
        ss = sparsity_sweep(x_np, f_vals, var_names=var_names,
                            max_degree=max_degree)
    else:
        ss = None

    result = {
        'alignment': align,
        'orthogonal_residual': orth,
        'conditioning': cond,
        'single_valuedness': sv,
        'r2_g': float(inner['r2']),
        'r2_hg': float(decomp_result['recon_r2']),
        'used_affine': bool(outer.get('used_affine', False)),
        'n_active_terms': int(len(inner['active'])),
        'active_terms': list(inner['active_names']),
    }

    if ss is not None:
        result['sparsity_sweep'] = ss
        result['sparsity_ratio'] = ss['sparsity_ratio']
        result['min_terms_r2_above_90'] = ss['min_terms_r2_above_90']
    else:
        result['sparsity_ratio'] = float('nan')
        result['min_terms_r2_above_90'] = None

    return result


# =====================================================================
# Compute stability metrics across repeated runs
# =====================================================================

def compute_run_stability(run_decomp_results):
    """Compute support and subspace stability across repeated decompositions.

    Args:
        run_decomp_results : list of decomposition result dicts

    Returns:
        dict with support stability and subspace stability metrics
    """
    inner_results = [r['inner'] for r in run_decomp_results]

    sup_stab = support_stability(inner_results)
    sub_stab = subspace_stability(inner_results)

    # Also aggregate per-run viability metrics
    per_run_metrics = [compute_viability_metrics(r) for r in run_decomp_results]

    r2_g_vals = [m['r2_g'] for m in per_run_metrics]
    r2_hg_vals = [m['r2_hg'] for m in per_run_metrics]
    align_vals = [m['alignment']['mean'] for m in per_run_metrics]
    orth_vals = [m['orthogonal_residual']['mean_r_perp_norm'] for m in per_run_metrics]

    return {
        'support_stability': sup_stab,
        'subspace_stability': sub_stab,
        'per_run_metrics': per_run_metrics,
        'r2_g': {'mean': float(np.mean(r2_g_vals)),
                  'std': float(np.std(r2_g_vals))},
        'r2_hg': {'mean': float(np.mean(r2_hg_vals)),
                   'std': float(np.std(r2_hg_vals))},
        'alignment': {'mean': float(np.mean(align_vals)),
                      'std': float(np.std(align_vals))},
        'orthogonal_residual_norm': {'mean': float(np.mean(orth_vals)),
                                      'std': float(np.std(orth_vals))},
    }


# =====================================================================
# Verdict logic
# =====================================================================

def compute_verdict(metrics, stability=None):
    """Compute a coarse viability verdict from metrics.

    The central criterion is SPARSITY: a viable dictionary means a small
    number of monomial terms suffice for good fit. High R² with many terms
    is polynomial overfitting, not dictionary discovery.

    Args:
        metrics : dict from compute_viability_metrics (single run)
                  or aggregated stats
        stability : dict from compute_run_stability (optional)

    Returns:
        str : one of 'viable', 'partially_viable', 'subspace_only', 'not_viable'
    """
    align_mean = metrics.get('alignment', {}).get('mean', 0.0)
    orth_mean = metrics.get('orthogonal_residual', {}).get('mean_r_perp_norm', 1.0)
    r2_hg = metrics.get('r2_hg', 0.0)
    r2_g = metrics.get('r2_g', 0.0)
    sv_var = metrics.get('single_valuedness', {}).get('weighted_within_bin_var', float('inf'))
    if np.isnan(sv_var):
        sv_var = float('inf')

    # Sparsity: the primary discriminator
    sparsity_ratio = metrics.get('sparsity_ratio', float('nan'))
    min_terms = metrics.get('min_terms_r2_above_90', None)

    # Sparse means: can get R² > 0.9 with at most 50% of available terms
    # AND the number of terms is small in absolute terms
    if not np.isnan(sparsity_ratio):
        is_sparse = sparsity_ratio < 0.5 and (min_terms is not None and min_terms <= 8)
    else:
        # Fallback: use n_active_terms directly
        n_active = metrics.get('n_active_terms', 99)
        is_sparse = n_active <= 8

    good_fit = r2_hg > 0.9
    good_alignment = align_mean > 0.8
    low_orth_residual = orth_mean < 0.3
    good_sv = sv_var < 1.0

    # Geometric consistency
    geometric_ok = good_alignment and low_orth_residual and good_sv
    partial_geometric = (good_alignment or low_orth_residual) and not geometric_ok

    # Support stability from repeated runs
    if stability is not None:
        mean_jaccard = stability.get('support_stability', {}).get('mean_jaccard', 0.0)
        subspace_angle = stability.get('subspace_stability', {}).get('mean_angle_deg', 90.0)
        if np.isnan(subspace_angle):
            subspace_angle = 90.0
        stable_support = mean_jaccard > 0.7
        stable_subspace = subspace_angle < 15.0
    else:
        stable_support = True  # assume stable if single run
        stable_subspace = True

    # Verdict hierarchy: sparsity gates everything
    if not is_sparse:
        # Not sparse → cannot be viable, regardless of fit quality
        if stable_subspace and good_fit:
            return 'subspace_only'
        return 'not_viable'

    if good_fit and geometric_ok and stable_support:
        return 'viable'
    elif good_fit and partial_geometric:
        if not stable_support and stable_subspace:
            return 'subspace_only'
        return 'partially_viable'
    elif stable_subspace and not stable_support:
        return 'subspace_only'
    else:
        return 'not_viable'


