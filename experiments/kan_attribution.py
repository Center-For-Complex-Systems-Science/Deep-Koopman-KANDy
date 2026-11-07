"""
PyKAN-style edge attribution scores for KANLinear encoders.

Why this exists
---------------
The paper's prune threshold tau (appendix "Edge importance and the prune rule",
Eq. edge-importance) is PyKAN's `attribute()` + `prune_edge(threshold=tau)`:
a layerwise-relevance-propagation-style attribution on activation *variance*,

    edge_score^(l)_{k,i,j} = E_l[i,j] * subnode_score^(l)_{k,i}
                             / (S_l[i] + eps)
    B^(l)_{i,j}            = mean_k edge_score^(l)_{k,i,j}

with E_l[i,j] = std over the batch of the post-activation of edge (j -> i) and
S_l[i] = std over the batch of node i's summed value.  Because each edge score
is *divided by the scale of the node it feeds*, the scores are normalized and
tau ~ 0.03 is meaningful across models.

The Lorenz tau sweep (tab:prune-sweep) used exactly this, via a PyKAN encoder.
The Ikeda [2,4,4] encoder is KANLinear (kan.py), whose only built-in importance
proxy is the raw spline-weight magnitude -- an *unnormalized* score that on the
trained Ikeda model sits at O(1) for every edge, so the Lorenz tau grid
{0.01, 0.03, 0.05} prunes nothing and the axis would be vacuous.

This module reimplements the PyKAN recursion (mirroring
kan/MultKAN.py::attribute, einsum('ij,ki,i->kij')) on top of
plotting.inspect_kanlinear, so the same tau grid transfers to KANLinear models.
"""

import numpy as np
import torch

from plotting import inspect_kanlinear


def _layer_scales(layer, x):
    """(edge_actscale (out,in), subnode_actscale (out,), layer_output (N,out))."""
    info = inspect_kanlinear(layer, x)
    postacts = info['postacts']                 # (N, out, in), base + spline
    edge_actscale = postacts.std(dim=0)         # (out, in)
    out = layer(x)                              # (N, out)
    subnode_actscale = out.std(dim=0)           # (out,)
    return edge_actscale, subnode_actscale, out


@torch.no_grad()
def edge_attribution(layers, x):
    """PyKAN attribution scores B^(l)_{i,j} for a stack of KANLinear layers.

    Args:
        layers : list of KANLinear, applied in order (layers[0] sees x)
        x      : (N, in_features) input batch -- scores are data dependent

    Returns:
        list of (out, in) numpy arrays, one per layer, same ordering as `layers`.
    """
    edge_scales, subnode_scales = [], []
    h = x
    for layer in layers:
        e, s, h = _layer_scales(layer, h)
        edge_scales.append(e)
        subnode_scales.append(s)

    depth = len(layers)
    out_dim = edge_scales[-1].shape[0]
    node_score = torch.eye(out_dim)             # (out_dim, out_dim)

    edge_scores = [None] * depth
    for l in range(depth - 1, -1, -1):
        # subnode_score == node_score here: KANLinear has no multiplication nodes
        es = torch.einsum('ij,ki,i->kij', edge_scales[l], node_score,
                          1.0 / (subnode_scales[l] + 1e-4))
        edge_scores[l] = es.mean(dim=0).cpu().numpy()   # mean over output dims
        node_score = es.sum(dim=1)                      # (out_dim, in)

    return edge_scores


@torch.no_grad()
def prune_edges_by_attribution(model, x, tau, layer_filter=None):
    """Zero every KANLinear edge whose attribution score is <= tau.

    Mirrors PyKAN's prune_edge: mask = (edge_score > tau) & old_mask, and the
    masked edge's activation is frozen to zero (here by zeroing its spline and
    base weights, which is how ikeda_minimal.prune_model realizes the mask).

    Args:
        model        : nn.Module containing KANLinear submodules
        x            : (N, d) input batch used to compute the data-dependent scores
        tau          : prune threshold
        layer_filter : optional fn(name) -> bool, restricts which submodule
                       stacks are pruned (e.g. lambda n: n.startswith('encoder'))

    Returns:
        dict with n_pruned, n_total, and the score arrays per stack.
    """
    from kan import KANLinear

    # Group KANLinear layers into forward stacks by their parent module, so the
    # relevance recursion runs over an actual composition.
    stacks = {}
    for name, mod in model.named_modules():
        if isinstance(mod, KANLinear):
            parent = name.rsplit('.', 1)[0] if '.' in name else ''
            stacks.setdefault(parent, []).append((name, mod))

    n_pruned = n_total = 0
    all_scores = {}
    for parent, entries in stacks.items():
        if layer_filter is not None and not layer_filter(parent):
            continue
        entries.sort(key=lambda e: e[0])        # kan1 before kan2
        layers = [m for _, m in entries]
        try:
            scores = edge_attribution(layers, x)
        except RuntimeError:
            # Shape mismatch means this stack is not fed by x (e.g. the decoder,
            # whose input is the latent code); skip it.
            # Adding error message to avoid silent failures.
            print("Error: Shape mismatch means this stack is not fed ")
            continue
        for (name, mod), sc in zip(entries, scores):
            all_scores[name] = sc
            mask = torch.tensor(sc <= tau)
            for oi in range(mod.out_features):
                for ii in range(mod.in_features):
                    n_total += 1
                    if mask[oi, ii]:
                        mod.spline_weight[oi, ii, :] = 0.0
                        if getattr(mod, 'spline_scaler', None) is not None:
                            mod.spline_scaler[oi, ii] = 0.0
                        mod.base_weight[oi, ii] = 0.0
                        n_pruned += 1
    return {'n_pruned': n_pruned, 'n_total': n_total, 'scores': all_scores}


def score_summary(scores):
    """Flatten a {name: (out,in) array} dict into summary statistics."""
    flat = np.concatenate([s.ravel() for s in scores.values()]) if scores else np.zeros(0)
    if flat.size == 0:
        return {}
    return {
        'min': float(flat.min()), 'p25': float(np.percentile(flat, 25)),
        'median': float(np.median(flat)), 'p75': float(np.percentile(flat, 75)),
        'max': float(flat.max()),
        'frac_below_0.01': float((flat <= 0.01).mean()),
        'frac_below_0.03': float((flat <= 0.03).mean()),
        'frac_below_0.05': float((flat <= 0.05).mean()),
    }
