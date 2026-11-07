"""
Foliation discovery diagnostics for 2-layer KAN encoders.

Given a trained two-layer KAN encoder (kan1: n_in -> mid, kan2: mid -> latent),
decompose the encoder's computation into inner functions u_j = sum_i psi_{j,i}(x_i)
and outer functions Phi_{k,j}(u_j), then test whether the inner functions factor
through a radial coordinate r^2 = x^2 + y^2 (i.e., the encoder discovered a
foliation of the state space).

This is designed for 2D maps (e.g., Ikeda, standard map) where the nonlinearity
factors through r^2.  The key diagnostic: if u_j vs r^2 collapses to a curve,
the KAN has discovered the foliation without being told about it.

Functions:
    extract_inner_functions    -- layer-1 outputs u_j and per-edge activations
    extract_outer_functions    -- layer-2 per-edge contributions Phi_{k,j}(u_j)
    channel_activity           -- count active channels per latent dim
    foliation_alignment_score  -- test whether u_j factors through r^2
    foliation_analysis         -- full pipeline combining all of the above
    plot_foliation_discovery   -- 3-row diagnostic figure
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


# =====================================================================
# 1. Extract layer-1 inner functions
# =====================================================================

def _compute_per_edge_activations(kan_layer, x):
    """Compute per-edge activations for a KANLinear layer.

    For each input i and output j, compute the individual edge contribution:
        edge_{j,i}(x) = base_activation(x_i) * base_weight[j,i]
                       + spline(x_i) * spline_scaler[j,i]

    Handles frozen edges (contributes zero) and symbolic edge functions.

    Args:
        kan_layer: KANLinear module
        x: (N, in_features) tensor

    Returns:
        per_edge: (N, out_features, in_features) tensor of edge contributions
    """
    N = x.size(0)
    n_in = kan_layer.in_features
    n_out = kan_layer.out_features

    # Check for frozen edges
    has_frozen = (hasattr(kan_layer, '_frozen_mask')
                  and kan_layer._frozen_mask is not None
                  and kan_layer._frozen_mask.any())

    # Determine effective weights (zero out frozen edges)
    if has_frozen:
        free = ~kan_layer._frozen_mask  # (out, in) bool
        bw = kan_layer.base_weight * free
        ssw = kan_layer.scaled_spline_weight * free.unsqueeze(-1)
    else:
        bw = kan_layer.base_weight
        ssw = kan_layer.scaled_spline_weight

    # Compute B-spline basis: (N, in_features, grid_size + spline_order)
    bases = kan_layer.b_splines(x)

    # Base activation applied element-wise: (N, in_features)
    base_act = kan_layer.base_activation(x)

    # Allocate output: (N, n_out, n_in)
    per_edge = torch.zeros(N, n_out, n_in, device=x.device, dtype=x.dtype)

    for j in range(n_out):
        for i in range(n_in):
            # Spline contribution using effective (possibly masked) weights
            spline_val = (bases[:, i, :] * ssw[j, i, :]).sum(-1)

            # Base contribution using effective weights
            base_val = base_act[:, i] * bw[j, i]

            per_edge[:, j, i] = base_val + spline_val

    # Add symbolic edge contributions (these replace the frozen learnable part)
    if hasattr(kan_layer, '_symbolic_fns') and kan_layer._symbolic_fns:
        for (out_idx, in_idx), fn in kan_layer._symbolic_fns.items():
            per_edge[:, out_idx, in_idx] = (
                per_edge[:, out_idx, in_idx] + fn(x[:, in_idx])
            )

    return per_edge


def extract_inner_functions(encoder, x_data):
    """Extract layer-1 inner functions from a 2-layer KAN encoder.

    The inner functions are u_j(x) = sum_i psi_{j,i}(x_i) for j=1,...,mid.

    Args:
        encoder: trained KAN encoder with .kan1 and .kan2 attributes
        x_data: (N, n_in) tensor or numpy array of input points

    Returns:
        u_values: (N, mid) numpy array of inner function values
        per_edge: (N, mid, n_in) numpy array of per-edge activation tensor
    """
    if isinstance(x_data, np.ndarray):
        x_data = torch.from_numpy(x_data).float()

    encoder.eval()
    device = next(encoder.parameters()).device
    x_data = x_data.to(device)

    with torch.no_grad():
        # Full layer-1 output
        u = encoder.kan1(x_data)  # (N, mid)

        # Per-edge activations
        edge_acts = _compute_per_edge_activations(encoder.kan1, x_data)

    return u.cpu().numpy(), edge_acts.cpu().numpy()


# =====================================================================
# 2. Extract layer-2 outer functions
# =====================================================================

def extract_outer_functions(encoder, x_data):
    """Extract layer-2 per-edge contributions Phi_{k,j}(u_j).

    For each latent dim k and hidden dim j, compute the contribution of
    u_j to z_k through the layer-2 edge (j -> k).

    Args:
        encoder: trained KAN encoder with .kan1 and .kan2 attributes
        x_data: (N, n_in) tensor or numpy array

    Returns:
        outer_contributions: (N, latent, mid) numpy array of per-edge
            contributions from layer 2
    """
    if isinstance(x_data, np.ndarray):
        x_data = torch.from_numpy(x_data).float()

    encoder.eval()
    device = next(encoder.parameters()).device
    x_data = x_data.to(device)

    with torch.no_grad():
        # Get intermediate activations (layer-1 output)
        u = encoder.kan1(x_data)  # (N, mid)

        # Per-edge activations through layer 2
        edge_acts = _compute_per_edge_activations(encoder.kan2, u)
        # shape: (N, latent, mid)

    return edge_acts.cpu().numpy()


# =====================================================================
# 3. Channel activity analysis
# =====================================================================

def channel_activity(encoder, x_data, threshold=0.03):
    """Count active channels per latent dimension.

    A channel j is active for latent dim k if the max absolute value of
    Phi_{k,j}(u_j) over the data exceeds the threshold.

    Args:
        encoder: trained KAN encoder
        x_data: (N, n_in) tensor or numpy array
        threshold: minimum max|Phi_{k,j}| to consider channel active

    Returns:
        active_counts: (latent_dim,) int array of |A_k| per dim
        active_masks: (latent_dim, mid_dim) boolean array
    """
    outer = extract_outer_functions(encoder, x_data)
    # outer shape: (N, latent, mid)

    # Max absolute contribution over the dataset for each (k, j) pair
    max_abs = np.max(np.abs(outer), axis=0)  # (latent, mid)

    active_masks = max_abs > threshold
    active_counts = active_masks.sum(axis=1)  # (latent,)

    return active_counts, active_masks


# =====================================================================
# 4. Foliation alignment score
# =====================================================================

def foliation_alignment_score(u_j_values, r2_values, n_bins=50):
    """Test whether the inner function u_j factors through r^2 = x^2 + y^2.

    Two diagnostics:
      1. Binned R^2: partition r^2 into equal-frequency bins, measure how much
         within-bin variance of u_j is reduced relative to total variance.
      2. Spline R^2: fit a 1D smoothing spline u_j ~ f(r^2) and report R^2.

    If both are high (> 0.95), u_j is essentially a function of r^2 alone,
    meaning the encoder discovered the foliation.

    Args:
        u_j_values: (N,) array of inner function values
        r2_values: (N,) array of r^2 = x^2 + y^2 values
        n_bins: number of equal-frequency bins for the binned diagnostic

    Returns:
        dict with:
            r2_radial: binned R^2 (variance-reduction metric)
            r2_spline: spline regression R^2
            bin_means: (n_bins,) mean u_j per bin
            bin_r2_centers: (n_bins,) center r^2 value per bin
            residual_per_bin: (n_bins,) within-bin variance per bin
    """
    N = len(u_j_values)
    total_var = np.var(u_j_values)

    if total_var < 1e-15:
        # Constant function -- trivially radial
        return {
            'r2_radial': 1.0,
            'r2_spline': 1.0,
            'bin_means': np.zeros(n_bins),
            'bin_r2_centers': np.zeros(n_bins),
            'residual_per_bin': np.zeros(n_bins),
        }

    # --- Binned R^2 ---
    # Equal-frequency bins based on r^2 quantiles
    bin_edges = np.quantile(r2_values, np.linspace(0, 1, n_bins + 1))
    # Ensure unique edges (can happen with repeated r^2 values)
    bin_edges[-1] += 1e-10

    bin_indices = np.digitize(r2_values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_means = np.zeros(n_bins)
    bin_r2_centers = np.zeros(n_bins)
    residual_per_bin = np.zeros(n_bins)
    weighted_within_var = 0.0
    total_count = 0

    for b in range(n_bins):
        mask = bin_indices == b
        count = mask.sum()
        if count == 0:
            continue
        vals = u_j_values[mask]
        bin_means[b] = vals.mean()
        bin_r2_centers[b] = r2_values[mask].mean()
        within_var = vals.var()
        residual_per_bin[b] = within_var
        weighted_within_var += within_var * count
        total_count += count

    mean_within_var = weighted_within_var / max(total_count, 1)
    r2_radial = 1.0 - mean_within_var / total_var

    # --- Spline R^2 ---
    # Sort by r^2 for spline fitting
    sort_idx = np.argsort(r2_values)
    r2_sorted = r2_values[sort_idx]
    u_sorted = u_j_values[sort_idx]

    try:
        # Smoothing spline with moderate smoothing
        s_param = len(r2_sorted) / 10.0
        spline = UnivariateSpline(r2_sorted, u_sorted, s=s_param, k=3)
        u_pred = spline(r2_values)
        ss_res = np.sum((u_j_values - u_pred) ** 2)
        ss_tot = np.sum((u_j_values - u_j_values.mean()) ** 2)
        r2_spline = 1.0 - ss_res / max(ss_tot, 1e-15)
    except Exception:
        # Fallback if spline fitting fails (e.g., too few unique points)
        r2_spline = r2_radial

    return {
        'r2_radial': float(r2_radial),
        'r2_spline': float(r2_spline),
        'bin_means': bin_means,
        'bin_r2_centers': bin_r2_centers,
        'residual_per_bin': residual_per_bin,
    }


# =====================================================================
# 5. Full foliation analysis pipeline
# =====================================================================

def foliation_analysis(encoder, x_data, threshold=0.03):
    """Full foliation discovery pipeline for a 2-layer KAN encoder.

    Extracts inner/outer functions, identifies active channels, and tests
    each active inner function for alignment with the r^2 foliation.

    Args:
        encoder: trained 2-layer KAN encoder with .kan1 and .kan2 attributes
        x_data: (N, 2) tensor or numpy array of (x, y) points on the attractor
        threshold: pruning threshold for channel activity

    Returns:
        dict with:
            inner_functions: (N, mid) array of u_j values
            per_edge_inner: (N, mid, n_in) array of layer-1 edge activations
            outer_contributions: (N, latent, mid) array of layer-2 edge activations
            active_counts: (latent,) array of |A_k| per dim
            active_masks: (latent, mid) boolean array
            foliation_scores: list of dicts, one per hidden dim j, each
                containing the foliation_alignment_score results (or None
                if channel j is not active for any latent dim)
            summary: dict with aggregate statistics
    """
    if isinstance(x_data, np.ndarray):
        x_np = x_data.astype(np.float32)
        x_tensor = torch.from_numpy(x_np)
    else:
        x_tensor = x_data.float()
        x_np = x_data.cpu().numpy().astype(np.float32)

    # Compute r^2 = x^2 + y^2
    r2 = x_np[:, 0] ** 2 + x_np[:, 1] ** 2

    # Extract inner functions
    inner_functions, per_edge_inner = extract_inner_functions(encoder, x_tensor)
    # inner_functions: (N, mid), per_edge_inner: (N, mid, n_in)

    # Extract outer functions
    outer_contributions = extract_outer_functions(encoder, x_tensor)
    # (N, latent, mid)

    # Channel activity
    active_counts, active_masks = channel_activity(encoder, x_tensor, threshold)
    mid_dim = inner_functions.shape[1]

    # Determine which hidden channels are active for ANY latent dim
    any_active = active_masks.any(axis=0)  # (mid,)

    # Foliation alignment for each hidden channel
    foliation_scores = []
    for j in range(mid_dim):
        if any_active[j]:
            score = foliation_alignment_score(inner_functions[:, j], r2)
        else:
            score = None
        foliation_scores.append(score)

    # Summary statistics
    active_indices = np.where(any_active)[0]
    n_active = len(active_indices)
    if n_active > 0:
        radial_scores = [foliation_scores[j]['r2_radial'] for j in active_indices]
        spline_scores = [foliation_scores[j]['r2_spline'] for j in active_indices]
        mean_radial = float(np.mean(radial_scores))
        mean_spline = float(np.mean(spline_scores))
        n_radial_aligned = int(np.sum(np.array(radial_scores) > 0.95))
        n_spline_aligned = int(np.sum(np.array(spline_scores) > 0.95))
    else:
        mean_radial = 0.0
        mean_spline = 0.0
        n_radial_aligned = 0
        n_spline_aligned = 0

    summary = {
        'n_active_channels': n_active,
        'total_channels': mid_dim,
        'active_channel_indices': active_indices.tolist(),
        'mean_radial_r2': mean_radial,
        'mean_spline_r2': mean_spline,
        'n_radial_aligned': n_radial_aligned,
        'n_spline_aligned': n_spline_aligned,
        'latent_dim': outer_contributions.shape[1],
        'active_counts_per_latent': active_counts.tolist(),
    }

    return {
        'inner_functions': inner_functions,
        'per_edge_inner': per_edge_inner,
        'outer_contributions': outer_contributions,
        'active_counts': active_counts,
        'active_masks': active_masks,
        'foliation_scores': foliation_scores,
        'summary': summary,
    }


# =====================================================================
# 6. Diagnostic plotting
# =====================================================================

def plot_foliation_discovery(encoder, x_data, save_path, threshold=0.03):
    """Create 3-row foliation discovery diagnostic figure.

    Top row:    scatter of u_j(x, y) over the attractor for each active
                hidden channel j, colored by u_j value.
    Middle row: u_j vs r^2 scatter with smoothing spline overlaid.
                This is the money plot -- if the encoder discovered the
                foliation, the scatter collapses to a curve.
    Bottom row: contour plot of z_k over the attractor with circular
                r^2 contours overlaid for reference.

    Args:
        encoder: trained 2-layer KAN encoder
        x_data: (N, 2) tensor or numpy array on the attractor
        save_path: path stem for saving (will append .png and .pdf)
        threshold: channel activity threshold
    """
    # Run the full analysis
    results = foliation_analysis(encoder, x_data, threshold)

    if isinstance(x_data, np.ndarray):
        x_np = x_data.astype(np.float32)
    else:
        x_np = x_data.cpu().numpy().astype(np.float32)

    r2 = x_np[:, 0] ** 2 + x_np[:, 1] ** 2

    inner = results['inner_functions']      # (N, mid)
    outer = results['outer_contributions']  # (N, latent, mid)
    active_masks = results['active_masks']  # (latent, mid)
    foliation_scores = results['foliation_scores']
    summary = results['summary']

    # Select active channels (up to 6 for display)
    active_js = summary['active_channel_indices']
    display_js = active_js[:6]
    n_display = len(display_js)

    if n_display == 0:
        print("  No active channels found -- skipping foliation plot.")
        return results

    # Select latent dims to show in bottom row (up to n_display)
    latent_dim = outer.shape[1]
    # Pick latent dims with most active channels
    n_active_per_latent = results['active_counts']
    latent_order = np.argsort(-n_active_per_latent)
    display_ks = latent_order[:n_display]

    # Compute full encoder output for bottom row
    if isinstance(x_data, np.ndarray):
        x_tensor = torch.from_numpy(x_data.astype(np.float32))
    else:
        x_tensor = x_data.float()
    encoder.eval()
    device = next(encoder.parameters()).device
    with torch.no_grad():
        z_full = encoder(x_tensor.to(device)).cpu().numpy()  # (N, latent)

    # Subsample for scatter plots if dataset is very large
    N = len(x_np)
    max_scatter = 20000
    if N > max_scatter:
        rng = np.random.RandomState(42)
        scatter_idx = rng.permutation(N)[:max_scatter]
    else:
        scatter_idx = np.arange(N)

    # --- Create figure ---
    fig, axes = plt.subplots(3, n_display, figsize=(4.0 * n_display, 11),
                             squeeze=False)

    # Top row: u_j colored over the attractor
    for col, j in enumerate(display_js):
        ax = axes[0, col]
        u_j = inner[scatter_idx, j]
        sc = ax.scatter(x_np[scatter_idx, 0], x_np[scatter_idx, 1],
                        s=0.3, alpha=0.4, c=u_j, cmap='coolwarm',
                        rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        score = foliation_scores[j]
        ax.set_title(f'$u_{{{j}}}(x,y)$\n'
                     f'$R^2_{{rad}}$={score["r2_radial"]:.3f}',
                     fontsize=10)
        ax.set_xlabel('$x$', fontsize=9)
        ax.set_ylabel('$y$', fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    # Middle row: u_j vs r^2 (the money plot)
    for col, j in enumerate(display_js):
        ax = axes[1, col]
        u_j = inner[:, j]
        score = foliation_scores[j]

        # Scatter
        ax.scatter(r2[scatter_idx], u_j[scatter_idx],
                   s=0.3, alpha=0.2, c='steelblue', rasterized=True)

        # Spline fit overlay
        sort_idx = np.argsort(r2)
        r2_sorted = r2[sort_idx]
        u_sorted = u_j[sort_idx]
        try:
            s_param = len(r2_sorted) / 10.0
            spline = UnivariateSpline(r2_sorted, u_sorted, s=s_param, k=3)
            r2_grid = np.linspace(r2.min(), r2.max(), 500)
            ax.plot(r2_grid, spline(r2_grid), 'r-', lw=1.5, label='spline fit')
            ax.legend(fontsize=7, loc='best')
        except Exception:
            pass

        # Bin means as markers
        ax.scatter(score['bin_r2_centers'], score['bin_means'],
                   c='orange', s=15, zorder=5, edgecolors='black',
                   linewidths=0.5, label='bin means')

        ax.set_title(f'$u_{{{j}}}$ vs $r^2$\n'
                     f'$R^2_{{spline}}$={score["r2_spline"]:.3f}',
                     fontsize=10)
        ax.set_xlabel('$r^2 = x^2 + y^2$', fontsize=9)
        ax.set_ylabel(f'$u_{{{j}}}$', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    # Bottom row: z_k contours over the attractor with r^2 reference circles
    for col in range(n_display):
        ax = axes[2, col]
        if col < len(display_ks):
            k = display_ks[col]
            z_k = z_full[scatter_idx, k]

            sc = ax.scatter(x_np[scatter_idx, 0], x_np[scatter_idx, 1],
                            s=0.3, alpha=0.4, c=z_k, cmap='viridis',
                            rasterized=True)
            plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)

            # Overlay circular r^2 contours for reference
            r2_quantiles = np.quantile(r2, [0.2, 0.4, 0.6, 0.8])
            theta = np.linspace(0, 2 * np.pi, 200)
            for rq in r2_quantiles:
                r = np.sqrt(rq)
                ax.plot(r * np.cos(theta), r * np.sin(theta),
                        'k--', alpha=0.3, lw=0.8)

            ax.set_title(f'$z_{{{k}}}$  ($|A_{{{k}}}|$={results["active_counts"][k]})',
                         fontsize=10)
            ax.set_xlabel('$x$', fontsize=9)
            ax.set_ylabel('$y$', fontsize=9)
            ax.set_aspect('equal')
        else:
            ax.axis('off')
        ax.tick_params(labelsize=7)

    # Overall title
    fig.suptitle(
        f'Foliation Discovery: {summary["n_active_channels"]}/{summary["total_channels"]} '
        f'active channels, '
        f'{summary["n_radial_aligned"]}/{summary["n_active_channels"]} radially aligned '
        f'($R^2_{{rad}} > 0.95$)',
        fontsize=12, y=1.01)

    fig.tight_layout()

    # Save
    save_path = str(save_path)
    if save_path.endswith('.png') or save_path.endswith('.pdf'):
        stem = save_path.rsplit('.', 1)[0]
    else:
        stem = save_path

    fig.savefig(stem + '.png', dpi=200, bbox_inches='tight')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved foliation plot: {stem}.png / .pdf")

    return results
