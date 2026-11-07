"""
Approximate Vanishing Ideal (AVI) for dictionary recovery.

Given data points and a target function f, finds polynomial relations
f ≈ g(x) by computing the SVD of the augmented evaluation matrix
[Φ(x) | -f] and extracting the null space.

The near-zero singular values correspond to approximate polynomial
relations among the monomials and f. No regularization parameter needed —
the singular value gap is the natural threshold.
"""

import numpy as np
from manifold_analysis import build_polynomial_dict, build_polynomial_dict_nd


def recover_vanishing_ideal(x_data, f_vals, var_names=None, max_degree=3,
                             sv_threshold='auto', verbose=True):
    """Recover polynomial relations via Approximate Vanishing Ideal.

    Algorithm:
        1. Build monomial evaluation matrix Φ in state variables
        2. Augment: A = [Φ | -f]  (N × (n_terms + 1))
        3. SVD(A) = U Σ V^T
        4. Right singular vectors with σ ≈ 0 are vanishing polynomials
        5. Those with significant weight on the f-column give f ≈ Σ c_j m_j

    Args:
        x_data : (N, d) numpy array of state variables
        f_vals : (N,) target function values
        var_names : list of variable name strings
        max_degree : maximum polynomial degree
        sv_threshold : 'auto' or float. Cutoff for "near-zero" singular values.
        verbose : print diagnostics

    Returns:
        dict with:
            - coeffs: sparse coefficient vector
            - active_names, active_coeffs: nonzero terms
            - r2: reconstruction quality
            - singular_values: full spectrum
            - vanishing_polynomials: list of all recovered relations
            - sv_gap: location and size of the singular value gap
    """
    N, d = x_data.shape

    # Build monomial dictionary
    if var_names is None:
        if d == 3:
            var_names = ['x', 'y', 'z']
        else:
            var_names = [f'x{i}' for i in range(d)]

    if d == 3 and var_names == ['x', 'y', 'z']:
        M, names, powers = build_polynomial_dict(
            x_data[:, 0], x_data[:, 1], x_data[:, 2], max_degree)
    else:
        M, names, powers = build_polynomial_dict_nd(x_data, var_names, max_degree)

    n_terms = M.shape[1]

    # Normalize columns for numerical stability
    col_norms = np.linalg.norm(M, axis=0)
    col_norms[col_norms < 1e-12] = 1.0
    M_norm = M / col_norms

    f_norm_factor = np.linalg.norm(f_vals)
    if f_norm_factor < 1e-12:
        f_norm_factor = 1.0
    f_normed = f_vals / f_norm_factor

    # Augment: A = [Φ_normalized | -f_normalized]
    A = np.column_stack([M_norm, -f_normed])

    # SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Find singular value gap
    log_s = np.log10(S + 1e-16)
    gaps = np.diff(log_s)  # negative values = drops
    # The biggest drop indicates the gap between signal and null space
    gap_idx = np.argmin(gaps)  # index of biggest drop
    gap_size = -gaps[gap_idx]  # in log10 units

    if sv_threshold == 'auto':
        # Threshold: geometric mean between the singular values across the gap
        if gap_size > 1.0:  # at least 10× drop
            sv_threshold = np.sqrt(S[gap_idx] * S[gap_idx + 1])
        else:
            # No clear gap — use relative threshold
            sv_threshold = S[0] * 1e-4

    if verbose:
        print(f"  SVD: {len(S)} singular values")
        print(f"  Range: [{S[-1]:.2e}, {S[0]:.2e}]")
        print(f"  Gap at index {gap_idx}: {S[gap_idx]:.2e} → {S[gap_idx+1]:.2e} "
              f"({gap_size:.1f} decades)")
        print(f"  Threshold: {sv_threshold:.2e}")
        print(f"  Null space dim: {np.sum(S < sv_threshold)}")

    # ---- Core idea: SVD reveals which monomials are needed ----
    #
    # The monomial matrix M has rank structure: if f lives in a
    # k-dimensional subspace of monomials, then [M | -f] has the
    # same rank as M restricted to those k monomials.
    #
    # Strategy: use SVD of M alone to find the effective rank,
    # then regress f onto the top-k monomial principal components.
    # The monomials that DON'T contribute (zero loading on top-k PCs)
    # are the ones excluded from the dictionary.

    # SVD of monomial matrix alone (without f)
    U_m, S_m, Vt_m = np.linalg.svd(M_norm, full_matrices=False)

    ss_tot = np.sum((f_vals - f_vals.mean()) ** 2)

    # Project f onto each principal component to find which matter
    f_centered = f_vals - f_vals.mean()
    pc_loadings = U_m.T @ f_centered  # (n_terms,) — how much f aligns with each PC

    # Cumulative R² as we add PCs (ordered by |loading|, not by σ)
    loading_order = np.argsort(-np.abs(pc_loadings))

    # Sweep: find minimum PCs needed for R² > 0.99
    vanishing = []
    best_r2 = -999
    best_k = n_terms
    for k in range(1, n_terms + 1):
        use = loading_order[:k]
        # Reconstruct f from top-k PCs
        proj = U_m[:, use] @ (U_m[:, use].T @ f_vals)
        ss_res = np.sum((f_vals - proj) ** 2)
        r2_k = 1.0 - ss_res / max(ss_tot, 1e-12)
        vanishing.append({'k': k, 'r2': r2_k, 'pc_indices': use.copy()})
        if r2_k > best_r2:
            best_r2 = r2_k
        if r2_k > 0.99 and best_k == n_terms:
            best_k = k

    if verbose:
        print(f"  PCs needed for R²>0.99: {best_k}/{n_terms}")
        print(f"  R² with {best_k} PCs: {vanishing[best_k-1]['r2']:.6f}")

    # Now identify which MONOMIALS contribute to the top-k PCs
    # The right singular vectors Vt_m tell us monomial loadings per PC
    use_pcs = loading_order[:best_k]
    monomial_importance = np.zeros(n_terms)
    for pc in use_pcs:
        # Weight by how much this PC contributes to f
        weight = abs(pc_loadings[pc])
        monomial_importance += weight * np.abs(Vt_m[pc, :])

    # Regression: f onto monomial matrix (full least-squares, then threshold)
    coeffs_full, _, _, _ = np.linalg.lstsq(M, f_vals, rcond=None)
    pred_full = M @ coeffs_full
    r2 = 1.0 - np.sum((f_vals - pred_full) ** 2) / max(ss_tot, 1e-12)

    # Use monomial_importance to rank terms — threshold at elbow
    importance_sorted = np.sort(monomial_importance)[::-1]
    cumulative = np.cumsum(importance_sorted) / importance_sorted.sum()
    # Keep terms that account for 95% of total importance
    n_keep = np.searchsorted(cumulative, 0.95) + 1
    importance_threshold = importance_sorted[min(n_keep, len(importance_sorted)-1)]

    if verbose:
        print(f"  Monomial importance: {n_keep} terms capture 95% of loading")

    # Zero out unimportant monomials
    coeffs = coeffs_full.copy()
    unimportant = monomial_importance < importance_threshold
    coeffs[unimportant] = 0.0

    # Re-fit with only important monomials for clean coefficients
    important_mask = ~unimportant
    if important_mask.sum() > 0:
        M_sub = M[:, important_mask]
        c_sub, _, _, _ = np.linalg.lstsq(M_sub, f_vals, rcond=None)
        coeffs = np.zeros(n_terms)
        coeffs[important_mask] = c_sub

    pred_sparse = M @ coeffs
    r2_sparse = 1.0 - np.sum((f_vals - pred_sparse) ** 2) / max(ss_tot, 1e-12)

    active = np.where(np.abs(coeffs) > 1e-10)[0]
    active_names = [names[i] for i in active]
    active_coeffs = coeffs[active]

    # Sort by magnitude
    order = np.argsort(-np.abs(active_coeffs))
    active_names = [active_names[i] for i in order]
    active_coeffs = active_coeffs[order]

    if verbose:
        print(f"  R² (full): {r2:.6f}")
        print(f"  R² (sparse, {len(active)} terms): {r2_sparse:.6f}")
        print(f"  Active: {active_names}")

    return {
        'coeffs': coeffs,
        'coeffs_full': coeffs,
        'names': names,
        'powers': powers,
        'active': active,
        'active_names': active_names,
        'active_coeffs': active_coeffs,
        'r2': float(r2_sparse),
        'r2_full': float(r2),
        'singular_values': S,
        'sv_threshold': sv_threshold,
        'sv_gap_idx': gap_idx,
        'sv_gap_size': gap_size,
        'vanishing_polynomials': vanishing,
        'n_pcs_needed': best_k,
        'pc_r2_curve': [(v['k'], v['r2']) for v in vanishing],
    }


def plot_singular_values(result, title='', save_path=None):
    """Plot singular value spectrum with gap annotation."""
    import matplotlib.pyplot as plt

    S = result['singular_values']
    gap_idx = result['sv_gap_idx']
    thresh = result['sv_threshold']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: singular values (log scale)
    ax = axes[0]
    ax.semilogy(S, 'o-', color='steelblue', markersize=4, lw=1.5)
    ax.axhline(thresh, color='red', ls='--', alpha=0.7,
               label=f'Threshold: {thresh:.1e}')
    ax.axvline(gap_idx + 0.5, color='green', ls=':', alpha=0.5,
               label=f'Gap at {gap_idx}')

    # Shade null space
    null_idx = np.where(S < thresh)[0]
    if len(null_idx) > 0:
        ax.axvspan(null_idx[0] - 0.5, len(S) - 0.5, alpha=0.1, color='red',
                   label=f'Null space ({len(null_idx)} dims)')

    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Value Spectrum')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Right: recovered coefficients
    ax = axes[1]
    active_names = result['active_names']
    active_coeffs = result['active_coeffs']
    if len(active_names) > 0:
        colors = ['#d62728' if c < 0 else '#1f77b4' for c in active_coeffs]
        ax.barh(range(len(active_names)), active_coeffs, color=colors,
                edgecolor='black', lw=0.5)
        ax.set_yticks(range(len(active_names)))
        ax.set_yticklabels(active_names, fontsize=9)
        ax.invert_yaxis()
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Recovered Polynomial (R²={result["r2"]:.4f})')

    fig.suptitle(title or 'Approximate Vanishing Ideal', fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig
