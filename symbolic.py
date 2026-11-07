"""
Global Symbolic Consistency Refinement for KANDy.

Extracts interpretable symbolic expressions from trained KANLinear layers
by jointly fitting symbolic functions to per-edge activations with a
global consistency constraint (cosheaf-inspired).

For each output node j of a KANLinear(in_dim → out_dim), the output is:
    h_j = Σ_i φ_{j,i}(x_i)

where φ_{j,i} are learned spline+base activation functions on each edge.

This module:
1. Decomposes the KANLinear into per-edge activations
2. For each output node, enumerates symbolic assignments for its edges
3. Jointly fits parameters to minimize:
       ||Σ_i f_i(x_i; θ) - h_j||²  +  λ · Σ_i ||f_i - φ_i||²
   (global consistency)                (spline fidelity)
"""

import numpy as np
import torch
from itertools import product as cartesian_product
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Symbolic function library
# (name, callable(x, *params) -> array, n_params, initial_params)
# ---------------------------------------------------------------------------
def _safe_exp(x):
    return np.exp(np.clip(x, -20, 20))


SYMBOLIC_LIBRARY = [
    ("a*x+b",
     lambda x, a, b: a * x + b,
     2, [1.0, 0.0]),
    ("a*x²+b*x+c",
     lambda x, a, b, c: a * x**2 + b * x + c,
     3, [0.01, 1.0, 0.0]),
    ("a*x³+b*x²+c*x+d",
     lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
     4, [1e-4, 0.01, 1.0, 0.0]),
    ("a*sin(c*x+d)+b",
     lambda x, a, b, c, d: a * np.sin(c * x + d) + b,
     4, [1.0, 0.0, 0.1, 0.0]),
    ("a*cos(c*x+d)+b",
     lambda x, a, b, c, d: a * np.cos(c * x + d) + b,
     4, [1.0, 0.0, 0.1, 0.0]),
    ("a*exp(b*x)+c",
     lambda x, a, b, c: a * _safe_exp(b * x) + c,
     3, [0.1, 0.01, 0.0]),
    ("a*tanh(b*x+c)+d",
     lambda x, a, b, c, d: a * np.tanh(b * x + c) + d,
     4, [1.0, 0.1, 0.0, 0.0]),
    ("const",
     lambda x, a: np.full_like(x, a, dtype=float),
     1, [0.0]),
]


# ---------------------------------------------------------------------------
# Per-edge activation extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_edge_activations(kan_linear, x_data):
    """Decompose KANLinear output into per-edge activation values.

    For edge (i→j):
        φ_{j,i}(x_i) = base_weight[j,i] · SiLU(x_i)
                      + Σ_k scaled_spline_weight[j,i,k] · B_k(x_i)

    Args:
        kan_linear: trained KANLinear module
        x_data: (N, in_features) tensor

    Returns:
        edge_acts: (out_features, in_features, N) ndarray — per-edge values
        h_data: (N, out_features) ndarray — full layer outputs
    """
    x = x_data.detach()
    base_act = kan_linear.base_activation(x)                        # (N, in)
    base_contrib = base_act.unsqueeze(1) * kan_linear.base_weight.unsqueeze(0)  # (N, out, in)

    bsplines = kan_linear.b_splines(x)                               # (N, in, n_coeffs)
    ssw = kan_linear.scaled_spline_weight                            # (out, in, n_coeffs)
    spline_contrib = torch.einsum('bik,jik->bji', bsplines, ssw)    # (N, out, in)

    edge_acts = (base_contrib + spline_contrib).permute(1, 2, 0)    # (out, in, N)
    h_data = edge_acts.sum(dim=1).T                                  # (N, out)

    return edge_acts.cpu().numpy(), h_data.cpu().numpy()


# ---------------------------------------------------------------------------
# Local per-edge fitting (used for initialization)
# ---------------------------------------------------------------------------
def _fit_single_edge(x_i, phi_vals, library):
    """Fit each candidate in the library to one edge's activation curve."""
    y_var = np.var(phi_vals)
    if y_var < 1e-12:
        return [("const", [float(np.mean(phi_vals))], 1.0)]

    results = []
    for name, func, n_params, init in library:
        try:
            def resid(p, f=func):
                return f(x_i, *p) - phi_vals
            res = least_squares(resid, init[:n_params], method='lm', max_nfev=500)
            r2 = 1.0 - np.sum(res.fun**2) / (y_var * len(phi_vals))
            results.append((name, res.x.tolist(), float(r2)))
        except Exception:
            pass

    results.sort(key=lambda r: r[2], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Global consistency refinement
# ---------------------------------------------------------------------------
def global_consistency_refinement(
    kan_linear, x_data, library=None, lam=0.1, n_samples=2000,
    input_names=None, verbose=True, top_k=3,
):
    """Extract symbolic expressions with global consistency constraint.

    Two-level optimization (as described in the theory):
      Outer: discrete selection of function type per edge (pruned to top_k
             local candidates, so top_k^in_dim assignments per node)
      Inner: continuous optimization of parameters via least-squares

    Args:
        kan_linear: trained KANLinear layer
        x_data: (N, in_features) tensor of representative inputs
        library: list of (name, func, n_params, init); default SYMBOLIC_LIBRARY
        lam: weight for spline-fidelity regularizer
        n_samples: subsample size for efficiency
        input_names: e.g. ["x", "y", "z"]
        verbose: print per-node results
        top_k: keep top-k local candidates per edge to prune search space

    Returns:
        list of dicts (one per output node) with keys:
            edges, formula, consistency_mse, r2
    """
    if library is None:
        library = SYMBOLIC_LIBRARY
    if input_names is None:
        input_names = [f"x{i}" for i in range(kan_linear.in_features)]

    N = x_data.shape[0]
    if N > n_samples:
        idx = torch.randperm(N, device=x_data.device)[:n_samples]
        x_sub = x_data[idx]
    else:
        x_sub = x_data

    edge_acts, h_data = extract_edge_activations(kan_linear, x_sub)
    x_np = x_sub.cpu().numpy()
    out_dim, in_dim, n_pts = edge_acts.shape

    if verbose:
        print(f"Global symbolic refinement: {out_dim} nodes, "
              f"top-{top_k} pruning → {top_k**in_dim} assignments/node, "
              f"{n_pts} data points")

    # STEP 1: Local fits for each edge — used for pruning and initialization
    # local_ranked[j][i] = [(lib_idx, name, params, r2), ...] sorted by r2
    local_ranked = [[[] for _ in range(in_dim)] for _ in range(out_dim)]

    for j in range(out_dim):
        for i in range(in_dim):
            fits = _fit_single_edge(x_np[:, i], edge_acts[j, i], library)
            for name, params, r2 in fits:
                lib_idx = next(k for k, (n, *_) in enumerate(library) if n == name)
                local_ranked[j][i].append((lib_idx, name, params, r2))

    results = []

    for j in range(out_dim):
        h_target = h_data[:, j]
        h_var = np.var(h_target)

        # Prune: keep only top_k candidates per edge
        candidates_per_edge = []
        for i in range(in_dim):
            ranked = local_ranked[j][i][:top_k]
            candidates_per_edge.append(ranked)

        best_loss = np.inf
        best_info = None

        # STEP 2: Enumerate pruned assignments (top_k^in_dim)
        for combo in cartesian_product(*candidates_per_edge):
            # combo is tuple of (lib_idx, name, local_params, local_r2) per edge
            edge_specs = []
            param_counts = []
            init_parts = []

            for i, (lib_idx, name, local_params, _) in enumerate(combo):
                _, func, n_p, default_init = library[lib_idx]
                edge_specs.append((i, name, func, n_p))
                param_counts.append(n_p)
                init_parts.extend(local_params if local_params else default_init[:n_p])

            offsets = [0] + list(np.cumsum(param_counts))
            init_arr = np.array(init_parts, dtype=float)

            def _make_residuals(specs=edge_specs, offs=offsets,
                                ht=h_target, ea_j=edge_acts[j]):
                def residuals(params):
                    pred = np.zeros(len(ht))
                    fid = []
                    for k, (i, name, func, n_p) in enumerate(specs):
                        p = params[offs[k]:offs[k + 1]]
                        fval = func(x_np[:, i], *p)
                        pred += fval
                        fid.append(fval - ea_j[i])
                    cons = pred - ht
                    fid_all = np.concatenate(fid) * np.sqrt(lam)
                    return np.concatenate([cons, fid_all])
                return residuals

            try:
                res = least_squares(
                    _make_residuals(), init_arr,
                    method='trf', max_nfev=1000, ftol=1e-6, xtol=1e-6,
                )
                cons_mse = float(np.mean(res.fun[:len(h_target)]**2))
            except Exception:
                continue

            if cons_mse < best_loss:
                best_loss = cons_mse
                best_info = {
                    'node': j,
                    'edge_specs': edge_specs,
                    'offsets': offsets,
                    'fitted_params': res.x.copy(),
                    'consistency_mse': cons_mse,
                }

        # Format output
        if best_info is not None:
            edges = []
            formula_parts = []
            for k, (i, name, func, n_p) in enumerate(best_info['edge_specs']):
                p = best_info['fitted_params'][best_info['offsets'][k]:best_info['offsets'][k + 1]]
                edges.append({
                    'input_idx': i,
                    'input_name': input_names[i],
                    'func_name': name,
                    'params': p.tolist(),
                })
                formula_parts.append(_format_func(name, p, input_names[i]))

            formula = " + ".join(formula_parts)
            r2 = 1.0 - best_loss / max(h_var, 1e-12)

            best_info['edges'] = edges
            best_info['formula'] = formula
            best_info['r2'] = float(r2)

            if verbose:
                status = f"R²={r2:.4f}" if r2 > 0.5 else f"R²={r2:.4f} (low)"
                print(f"  h[{j:2d}] = {formula}  [{status}]")

        results.append(best_info)

    return results


# ---------------------------------------------------------------------------
# Full encoder lift summary
# ---------------------------------------------------------------------------
def summarize_encoder_lift(
    kan_encoder, x_data, library=None, lam=0.1, n_samples=5000,
):
    """Extract the full interpretable encoder mapping.

    Encoder: z = Dense(ReLU(KANLinear(x)))

    Extracts symbolic forms for the KANLinear edges, then shows
    how the Dense layer linearly combines them into each latent dim.
    """
    print("=" * 70)
    print("STEP 1: Symbolic extraction of KANLinear edge functions")
    print("=" * 70)
    node_results = global_consistency_refinement(
        kan_encoder.kan, x_data, library=library, lam=lam,
        n_samples=n_samples, input_names=["x", "y", "z"],
    )

    W = kan_encoder.dense.weight.detach().cpu().numpy()   # (latent, hidden)
    b = kan_encoder.dense.bias.detach().cpu().numpy()      # (latent,)
    latent_dim, hidden_dim = W.shape

    print("\n" + "=" * 70)
    print("STEP 2: Full encoder lift  z_k = Σ_j W[k,j] · ReLU(h_j) + b_k")
    print("=" * 70)

    for k in range(latent_dim):
        print(f"\n--- z[{k}]  (bias = {b[k]:.4f}) ---")
        weights_k = W[k]
        order = np.argsort(np.abs(weights_k))[::-1]
        shown = 0
        for j in order:
            w = weights_k[j]
            if abs(w) < 0.01:
                continue
            r = node_results[j]
            if r is None:
                continue
            tag = f"R²={r['r2']:.3f}" if r['r2'] is not None else ""
            print(f"  {w:+.4f} · ReLU( {r['formula']} )  [{tag}]")
            shown += 1
            if shown >= 5:
                print(f"  ... ({np.sum(np.abs(weights_k) >= 0.01) - shown} more terms)")
                break

    return node_results


# ---------------------------------------------------------------------------
# Two-layer KAN: deep encoder lift with level set analysis
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate_layer_outputs(encoder, x_data, n_samples=2000):
    """Run data through the two-layer encoder and capture intermediates."""
    device = x_data.device
    N = x_data.shape[0]
    if N > n_samples:
        idx = torch.randperm(N, device=device)[:n_samples]
        x_sub = x_data[idx]
    else:
        x_sub = x_data

    u = encoder.kan1(x_sub)   # layer 1 output: (N, mid_size)
    v = encoder.kan2(u)        # layer 2 output: (N, hidden_size)

    return x_sub, u, v


def _level_set_analysis(kan2, u_data, v_data, l1_results, input_names,
                        top_k_edges=5, verbose=True):
    """Analyze layer 2 edge functions to identify mixing/product terms.

    For each output node k of layer 2, the output is:
        v_k = Σ_j Φ_{k,j}(u_j)

    where u_j are layer 1 outputs (already symbolically characterized).

    We identify which layer 1 nodes each layer 2 node depends on most
    strongly (by edge activation variance), and characterize the
    functional form of the composition.

    If Φ_{k,j}(u_j) is roughly linear and u_j ≈ f(x_i), then v_k
    inherits the univariate structure. But if Φ_{k,j} is nonlinear
    and multiple u_j contribute, the composition creates cross-terms.
    """
    edge_acts, h_data = extract_edge_activations(kan2, u_data)
    # edge_acts: (hidden_size, mid_size, N)
    # h_data: (N, hidden_size)

    u_np = u_data.cpu().numpy()
    hidden_size, mid_size, n_pts = edge_acts.shape

    if verbose:
        print(f"\nLevel set analysis: {hidden_size} output nodes, "
              f"{mid_size} layer-1 inputs")

    node_analyses = []

    for k in range(hidden_size):
        h_var = np.var(h_data[:, k])
        if h_var < 1e-10:
            node_analyses.append(None)
            continue

        # Measure each edge's contribution by variance
        edge_vars = np.array([np.var(edge_acts[k, j]) for j in range(mid_size)])
        total_var = edge_vars.sum()
        if total_var < 1e-10:
            node_analyses.append(None)
            continue

        frac = edge_vars / total_var
        dominant = np.argsort(frac)[::-1]

        # Characterize: which layer-1 nodes matter?
        active_edges = [(j, frac[j]) for j in dominant if frac[j] > 0.05]

        # For each active edge, fit a simple symbolic form
        edge_fits = []
        for j, contribution in active_edges:
            fits = _fit_single_edge(u_np[:, j], edge_acts[k, j], SYMBOLIC_LIBRARY)
            best = fits[0] if fits else ("?", [], 0.0)
            l1_formula = l1_results[j]['formula'] if l1_results[j] else f"u[{j}]"
            edge_fits.append({
                'l1_node': j,
                'contribution': float(contribution),
                'edge_func': best[0],
                'edge_params': best[1],
                'edge_r2': best[2],
                'l1_formula': l1_formula,
            })

        # Determine if this node is doing mixing
        n_active = len(active_edges)
        is_mixing = n_active >= 2

        analysis = {
            'node': k,
            'n_active_inputs': n_active,
            'is_mixing': is_mixing,
            'edge_fits': edge_fits,
            'h_var': float(h_var),
        }
        node_analyses.append(analysis)

        if verbose and h_var > 0.01:
            tag = "MIXING" if is_mixing else "pass-through"
            parts = []
            for ef in edge_fits[:top_k_edges]:
                j = ef['l1_node']
                pct = ef['contribution'] * 100
                fname = ef['edge_func']
                r2 = ef['edge_r2']
                parts.append(f"u[{j}]({pct:.0f}%, {fname}, R²={r2:.3f})")
            summary = " + ".join(parts)
            print(f"  v[{k:2d}] [{tag:12s}] = {summary}")

    return node_analyses


def summarize_deep_encoder_lift(
    encoder, x_data, library=None, lam=0.1, n_samples=2000,
):
    """Extract the full interpretable two-layer KAN encoder mapping.

    Encoder: KANLinear_1(x) → KANLinear_2(u) → ReLU → Dense → z

    Step 1: Symbolic extraction of layer 1 edges (univariate functions of x,y,z)
    Step 2: Level set analysis of layer 2 (how it mixes layer 1 outputs)
    Step 3: Show how Dense combines layer 2 outputs into latent dims
    """
    # Evaluate intermediates
    x_sub, u_data, v_data = _evaluate_layer_outputs(encoder, x_data, n_samples)

    # --- Layer 1: univariate symbolic extraction ---
    print("=" * 70)
    print("LAYER 1: Univariate basis functions  u_j = Σ_i ψ_{j,i}(x_i)")
    print("=" * 70)
    l1_results = global_consistency_refinement(
        encoder.kan1, x_sub, library=library, lam=lam,
        n_samples=n_samples, input_names=["x", "y", "z"],
    )

    # --- Layer 2: mixing analysis ---
    print("\n" + "=" * 70)
    print("LAYER 2: Mixing functions  v_k = Σ_j Φ_{k,j}(u_j)")
    print("=" * 70)
    l2_analysis = _level_set_analysis(
        encoder.kan2, u_data, v_data, l1_results,
        input_names=["x", "y", "z"],
    )

    # --- Dense layer: final combination (only for encoders with ReLU+Dense) ---
    if hasattr(encoder, 'dense'):
        W = encoder.dense.weight.detach().cpu().numpy()
        b = encoder.dense.bias.detach().cpu().numpy()
        latent_dim, hidden_dim = W.shape

        print("\n" + "=" * 70)
        print("LATENT DIMS: z_k = Σ_j W[k,j] · ReLU(v_j) + b_k")
        print("=" * 70)

        for k in range(latent_dim):
            print(f"\n--- z[{k}]  (bias = {b[k]:.4f}) ---")
            weights_k = W[k]
            order = np.argsort(np.abs(weights_k))[::-1]
            shown = 0
            for j in order:
                w = weights_k[j]
                if abs(w) < 0.01:
                    continue
                a = l2_analysis[j]
                if a is None:
                    continue
                tag = "MIX" if a['is_mixing'] else "uni"
                inputs = ", ".join(
                    f"u[{ef['l1_node']}]({ef['contribution']*100:.0f}%)"
                    for ef in a['edge_fits'][:3]
                )
                print(f"  {w:+.4f} · ReLU(v[{j}])  [{tag}] depends on: {inputs}")
                shown += 1
                if shown >= 5:
                    remaining = np.sum(np.abs(weights_k) >= 0.01) - shown
                    if remaining > 0:
                        print(f"  ... ({remaining} more terms)")
                    break
    else:
        print("\n" + "=" * 70)
        print("LATENT DIMS: z_k = v_k  (pure KAN, no Dense layer)")
        print("=" * 70)
        print("Each latent dim z[k] IS layer 2 output v[k] directly.")

    return l1_results, l2_analysis


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_func(name, params, var):
    p = params

    def _c(v, threshold=1e-4):
        """Format coefficient, suppress near-zero."""
        if abs(v) < threshold:
            return "0"
        return f"{v:.4f}"

    if name == "const":
        return _c(p[0])
    elif name == "a*x+b":
        return f"{_c(p[0])}·{var} + {_c(p[1])}"
    elif name == "a*x²+b*x+c":
        return f"{_c(p[0])}·{var}² + {_c(p[1])}·{var} + {_c(p[2])}"
    elif name == "a*x³+b*x²+c*x+d":
        return f"{p[0]:.6f}·{var}³ + {_c(p[1])}·{var}² + {_c(p[2])}·{var} + {_c(p[3])}"
    elif name == "a*sin(c*x+d)+b":
        return f"{_c(p[0])}·sin({_c(p[2])}·{var} + {_c(p[3])}) + {_c(p[1])}"
    elif name == "a*cos(c*x+d)+b":
        return f"{_c(p[0])}·cos({_c(p[2])}·{var} + {_c(p[3])}) + {_c(p[1])}"
    elif name == "a*exp(b*x)+c":
        return f"{_c(p[0])}·exp({_c(p[1])}·{var}) + {_c(p[2])}"
    elif name == "a*tanh(b*x+c)+d":
        return f"{_c(p[0])}·tanh({_c(p[1])}·{var} + {_c(p[2])}) + {_c(p[3])}"
    return f"{name}({var})"
