"""
Pruning Stability Test for Deep-Koopman-KANDy.

Tests whether the pruning pattern (which edges survive) is stable across:
  1. Different model seeds (same data)
  2. Different data seeds (different ICs)

For each (data_seed, model_seed) pair:
  - Train PyKAN [3,5,5] + stable for 200 epochs (with grid updates)
  - Compute edge importance via enc.attribute()
  - Prune at threshold 0.03
  - Record which edges were pruned (the binary mask)
  - Save model.plot() for the encoder (before and after pruning)
  - Save activation distributions and bar plots per edge

If the pruning pattern is identical across seeds, the architecture
discovery is deterministic — only the weight values differ.

Run from project root:
    python experiments/pruning_stability_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json, time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import simulate_lorenz, PairDataset
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable

OUTDIR = Path("results/pruning_stability")
OUTDIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
N_TRAJ = 60
N_TRAIN = 48
LATENT = 5
MID = 5
TRAIN_EPOCHS = 200
PRUNE_THRESHOLD = 0.03

# Test matrix: vary data seed and model seed independently
DATA_SEEDS = [42, 43, 44]
MODEL_SEEDS = [0, 1, 2, 3]


def train_model(traj, model_seed):
    """Train PyKAN [3,5,5] + stable with grid updates."""
    set_seed(model_seed)
    ae = PyKANAutoEncoder(3, MID, LATENT, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, LATENT)
    enc = model.ae.encoder
    dec = model.ae.decoder

    x_sample = traj[:N_TRAIN].reshape(-1, 3)
    grid_idx = np.random.choice(len(x_sample), min(5000, len(x_sample)), replace=False)
    x_init = x_sample[grid_idx]
    with torch.no_grad():
        enc.update_grid_from_samples(x_init)
        z_init = enc(x_init)
        dec.update_grid_from_samples(z_init)

    train_dl = DataLoader(PairDataset(traj[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_EPOCHS)
    mse_fn = nn.MSELoss()

    for ep in range(1, TRAIN_EPOCHS + 1):
        model.train()
        for x_t, x_tp1 in train_dl:
            out = model.forward_one_step(x_t, DT)
            z_true = model.ae.encoder(x_tp1)
            loss = (mse_fn(out["x_tp1_pred"], x_tp1)
                    + mse_fn(out["z_tp1_pred"], z_true)
                    + 0.5 * mse_fn(out["x_t_rec"], x_t))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 50 == 0:
            enc.update_grid_from_samples(x_init)
            with torch.no_grad():
                z_new = enc(x_init)
            dec.update_grid_from_samples(z_new)

    model.eval()
    return model


def get_edge_masks(enc):
    """Extract binary pruning mask as a list of 2D arrays per layer.

    Returns list of (n_out, n_in) arrays — same layout as edge_scores.
    Raw mask is (n_in, n_out), so we transpose.
    """
    masks = []
    for l in range(len(enc.act_fun)):
        mask = enc.act_fun[l].mask.detach().numpy()
        n_in = int(enc.width_in[l])
        n_out = int(enc.width_out[l + 1])
        # Raw mask: (n_in, n_out) where mask[i,j] = input i -> output j
        # Reshape then transpose to get (n_out, n_in) matching edge_scores
        masks.append(mask.reshape(n_in, n_out).T)
    return masks


def get_edge_scores(enc):
    """Extract edge importance scores as a list of 2D arrays per layer."""
    scores = []
    if enc.edge_scores is not None:
        for s in enc.edge_scores:
            scores.append(s.detach().numpy())
    return scores


def plot_encoder(enc, x_data, tag, outdir):
    """Save PyKAN model.plot() for the encoder."""
    try:
        enc.save_act = True
        with torch.no_grad():
            enc(x_data[:500])
        enc.plot()
        plt.savefig(outdir / f"encoder_{tag}.png", dpi=120, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        print(f"      encoder plot failed: {e}")


def plot_activation_distributions(enc, x_data, tag, outdir):
    """Plot distribution of activations for each edge."""
    enc.save_act = True
    with torch.no_grad():
        enc(x_data)

    for layer_idx in range(len(enc.act_fun)):
        n_in = int(enc.width_in[layer_idx])
        n_out = int(enc.width_out[layer_idx + 1])

        # Get pre-activations and post-activations
        preacts = enc.acts[layer_idx].detach().numpy()  # (N, n_in)
        postacts = enc.acts[layer_idx + 1].detach().numpy()  # (N, n_out)

        # Per-edge activation: approximate as postacts contribution
        # For a cleaner view, compute per-edge spline output
        # But acts only gives per-node values. Use edge_actscale if available.
        try:
            actscale = enc.edge_actscale[layer_idx].detach().numpy()  # (n_out, n_in)
        except (AttributeError, IndexError):
            actscale = np.ones((n_out, n_in))

        # Distribution plot: one subplot per edge
        fig, axes = plt.subplots(n_out, n_in, figsize=(3 * n_in, 2.5 * n_out),
                                  squeeze=False)

        in_labels = ["x", "y", "z"] if layer_idx == 0 else [f"ψ{j}" for j in range(n_in)]
        out_labels = [f"ψ{j}" for j in range(n_out)] if layer_idx == 0 else [f"z[{k}]" for k in range(n_out)]

        for j in range(n_out):
            for i in range(n_in):
                ax = axes[j, i]
                # Activation scale as proxy for edge importance
                scale = actscale[j, i] if actscale.ndim == 2 else 1.0

                # Get the mask — raw shape is (n_in, n_out)
                mask_val = enc.act_fun[layer_idx].mask.detach().numpy()
                mask_raw = mask_val.reshape(n_in, n_out)  # mask_raw[i,j] = input i -> output j
                is_active = mask_raw[i, j] > 0.5

                ax.hist(preacts[:, i], bins=50, alpha=0.6, color="steelblue" if is_active else "lightgray",
                        density=True)
                ax.set_title(f"{in_labels[i]}→{out_labels[j]}\nscale={scale:.3f}",
                             fontsize=7,
                             color="black" if is_active else "red")
                ax.tick_params(labelsize=5)
                if not is_active:
                    ax.set_facecolor("#fff0f0")

        fig.suptitle(f"Layer {layer_idx} Activation Distributions ({tag})\n"
                     f"Red background = pruned edge", fontsize=10)
        fig.tight_layout()
        fig.savefig(outdir / f"activations_L{layer_idx}_{tag}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)


def plot_edge_importance_bars(enc, tag, outdir):
    """Bar plot of edge importance scores with neuron labels on x-axis."""
    scores = get_edge_scores(enc)
    masks = get_edge_masks(enc)

    if not scores:
        print(f"      No edge scores available for {tag}")
        return

    for layer_idx, (score, mask) in enumerate(zip(scores, masks)):
        n_out, n_in = mask.shape

        in_labels = ["x", "y", "z"] if layer_idx == 0 else [f"ψ{j}" for j in range(n_in)]
        out_labels = [f"ψ{j}" for j in range(n_out)] if layer_idx == 0 else [f"z[{k}]" for k in range(n_out)]

        # Flatten for bar plot
        edge_labels = []
        edge_scores_flat = []
        edge_active = []

        for j in range(n_out):
            for i in range(n_in):
                edge_labels.append(f"{in_labels[i]}→{out_labels[j]}")
                edge_scores_flat.append(float(score[j, i]) if score.ndim == 2 else float(score[j * n_in + i]))
                edge_active.append(mask[j, i] > 0.5)

        fig, ax = plt.subplots(figsize=(max(12, len(edge_labels) * 0.5), 5))
        colors = ["steelblue" if a else "lightcoral" for a in edge_active]
        bars = ax.bar(range(len(edge_labels)), edge_scores_flat, color=colors,
                       edgecolor="black", linewidth=0.3)
        ax.axhline(PRUNE_THRESHOLD, color="red", ls="--", lw=1.5,
                    label=f"Threshold={PRUNE_THRESHOLD}")
        ax.set_xticks(range(len(edge_labels)))
        ax.set_xticklabels(edge_labels, rotation=90, fontsize=7)
        ax.set_ylabel("Edge Importance Score", fontsize=10)
        ax.set_title(f"Layer {layer_idx} Edge Importance ({tag})\n"
                     f"Blue=kept, Red=pruned", fontsize=11)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / f"importance_L{layer_idx}_{tag}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)


def main():
    t_start = time.time()
    print("=" * 70)
    print("  PRUNING STABILITY TEST")
    print(f"  Data seeds: {DATA_SEEDS}")
    print(f"  Model seeds: {MODEL_SEEDS}")
    print(f"  Prune threshold: {PRUNE_THRESHOLD}")
    print("=" * 70)

    all_masks = {}  # (data_seed, model_seed) -> masks
    all_scores = {}
    summary = []

    for ds in DATA_SEEDS:
        # Generate data
        print(f"\n  Data seed={ds}:")
        set_seed(ds)
        traj = simulate_lorenz(n_traj=N_TRAJ, n_steps=6000, dt=DT,
                               burn_in=500, device="cpu")
        x_data = traj[:N_TRAIN].reshape(-1, 3)
        x_sub = x_data[np.random.choice(len(x_data), 2000, replace=False)]
        x_sub_t = torch.tensor(x_sub.numpy(), dtype=torch.float32)

        for ms in MODEL_SEEDS:
            tag = f"d{ds}_m{ms}"
            print(f"    Model seed={ms} ({tag})")
            t0 = time.time()

            model = train_model(traj, ms)
            enc = model.ae.encoder
            elapsed_train = time.time() - t0
            print(f"      Trained in {elapsed_train:.0f}s")

            # Plot encoder BEFORE pruning
            plot_encoder(enc, x_sub_t, f"{tag}_before", OUTDIR)

            # Compute edge importance and prune
            enc.save_act = True
            with torch.no_grad():
                enc(x_sub_t)
            try:
                enc.attribute(plot=False)
            except Exception as e:
                print(f"      attribute failed: {e}")

            # Record scores BEFORE pruning
            scores_before = get_edge_scores(enc)
            all_scores[(ds, ms)] = scores_before

            # Plot importance bars BEFORE pruning
            plot_edge_importance_bars(enc, f"{tag}_before", OUTDIR)

            # Prune
            try:
                enc.prune_edge(threshold=PRUNE_THRESHOLD, log_history=False)
            except Exception as e:
                print(f"      prune failed: {e}")

            # Record masks AFTER pruning
            masks = get_edge_masks(enc)
            all_masks[(ds, ms)] = masks

            n_pruned_total = sum((m < 0.5).sum() for m in masks)
            n_total = sum(m.size for m in masks)
            print(f"      Pruned {n_pruned_total}/{n_total} edges")

            # Plot encoder AFTER pruning
            plot_encoder(enc, x_sub_t, f"{tag}_after", OUTDIR)

            # Plot activation distributions AFTER pruning
            plot_activation_distributions(enc, x_sub_t, f"{tag}_after", OUTDIR)

            # Plot importance bars AFTER pruning (scores don't change, but mask does)
            plot_edge_importance_bars(enc, f"{tag}_after", OUTDIR)

            summary.append({
                "data_seed": ds, "model_seed": ms,
                "n_pruned": int(n_pruned_total), "n_total": int(n_total),
                "masks": [m.tolist() for m in masks],
                "train_time": elapsed_train,
            })

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ═══════════════════════════════════════════════════════════════
    # STABILITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print("  STABILITY ANALYSIS")
    print(f"{'='*70}")

    # Compare masks across model seeds (same data)
    print(f"\n  Same data, different model seeds:")
    for ds in DATA_SEEDS:
        masks_this_data = [(ms, all_masks[(ds, ms)]) for ms in MODEL_SEEDS
                           if (ds, ms) in all_masks]
        if len(masks_this_data) < 2:
            continue

        ref_ms, ref_masks = masks_this_data[0]
        print(f"\n    data_seed={ds}, reference=model_seed={ref_ms}:")
        for ms, masks in masks_this_data[1:]:
            identical = all(np.array_equal(r, m) for r, m in zip(ref_masks, masks))
            diffs = sum(np.sum(r != m) for r, m in zip(ref_masks, masks))
            print(f"      vs model_seed={ms}: {'IDENTICAL' if identical else f'{diffs} edges differ'}")

    # Compare masks across data seeds (same model seed)
    print(f"\n  Same model seed, different data:")
    for ms in MODEL_SEEDS:
        masks_this_model = [(ds, all_masks[(ds, ms)]) for ds in DATA_SEEDS
                            if (ds, ms) in all_masks]
        if len(masks_this_model) < 2:
            continue

        ref_ds, ref_masks = masks_this_model[0]
        print(f"\n    model_seed={ms}, reference=data_seed={ref_ds}:")
        for ds, masks in masks_this_model[1:]:
            identical = all(np.array_equal(r, m) for r, m in zip(ref_masks, masks))
            diffs = sum(np.sum(r != m) for r, m in zip(ref_masks, masks))
            print(f"      vs data_seed={ds}: {'IDENTICAL' if identical else f'{diffs} edges differ'}")

    # Global: are ALL masks identical?
    all_mask_list = list(all_masks.values())
    if len(all_mask_list) >= 2:
        ref = all_mask_list[0]
        all_identical = all(
            all(np.array_equal(r, m) for r, m in zip(ref, other))
            for other in all_mask_list[1:]
        )
        print(f"\n  ALL masks identical across all seeds? {'YES' if all_identical else 'NO'}")

    # ═══════════════════════════════════════════════════════════════
    # Heatmap: pruning mask across all runs
    # ═══════════════════════════════════════════════════════════════
    for layer_idx in range(2):  # 2 layers in [3,5,5]
        n_runs = len(all_mask_list)
        if n_runs == 0:
            continue

        n_out, n_in = all_mask_list[0][layer_idx].shape
        in_labels = ["x", "y", "z"] if layer_idx == 0 else [f"ψ{j}" for j in range(n_in)]
        out_labels = [f"ψ{j}" for j in range(n_out)] if layer_idx == 0 else [f"z[{k}]" for k in range(n_out)]

        # Build matrix: rows = runs, cols = edges (flattened)
        n_edges = n_out * n_in
        mask_matrix = np.zeros((n_runs, n_edges))
        run_labels = []
        for idx, ((ds, ms), masks) in enumerate(all_masks.items()):
            mask_matrix[idx] = masks[layer_idx].flatten()
            run_labels.append(f"d{ds}_m{ms}")

        edge_labels = [f"{in_labels[i]}→{out_labels[j]}"
                       for j in range(n_out) for i in range(n_in)]

        fig, ax = plt.subplots(figsize=(max(10, n_edges * 0.6), max(4, n_runs * 0.4)))
        im = ax.imshow(mask_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_edges))
        ax.set_xticklabels(edge_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n_runs))
        ax.set_yticklabels(run_labels, fontsize=8)
        for i in range(n_runs):
            for j in range(n_edges):
                v = mask_matrix[i, j]
                ax.text(j, i, "●" if v > 0.5 else "○",
                        ha="center", va="center", fontsize=8,
                        color="black" if v > 0.5 else "red")
        ax.set_title(f"Layer {layer_idx} Pruning Mask Across Seeds\n"
                     f"● = active, ○ = pruned (threshold={PRUNE_THRESHOLD})",
                     fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(OUTDIR / f"mask_stability_L{layer_idx}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════
    # Threshold sweep: how does the mask change across thresholds?
    # Uses the stored edge scores — no retraining needed.
    # ═══════════════════════════════════════════════════════════════
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]

    print(f"\n  THRESHOLD SWEEP (using stored edge scores):")
    print(f"  {'threshold':>10}", end="")
    for ds in DATA_SEEDS:
        for ms in MODEL_SEEDS:
            if (ds, ms) in all_scores:
                print(f"  d{ds}m{ms}", end="")
    print(f"  {'agree?':>8}")
    print(f"  " + "-" * 80)

    for layer_idx in range(2):
        print(f"\n  Layer {layer_idx}:")
        for thresh in thresholds:
            masks_at_thresh = []
            counts = []
            for ds in DATA_SEEDS:
                for ms in MODEL_SEEDS:
                    if (ds, ms) not in all_scores:
                        continue
                    scores = all_scores[(ds, ms)]
                    if layer_idx < len(scores):
                        s = scores[layer_idx]
                        mask = (s > thresh).astype(int)
                        masks_at_thresh.append(mask)
                        n_active = mask.sum()
                        n_total = mask.size
                        counts.append(f"{n_active:>3}/{n_total}")

            # Check agreement
            if len(masks_at_thresh) >= 2:
                all_same = all(np.array_equal(masks_at_thresh[0], m)
                               for m in masks_at_thresh[1:])
            else:
                all_same = True

            print(f"  {thresh:>10.3f}", end="")
            for c in counts:
                print(f"  {c:>6}", end="")
            print(f"  {'  YES' if all_same else '  NO':>8}")

    # Score distribution plot: overlay all runs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for layer_idx in range(2):
        ax = axes[layer_idx]
        for (ds, ms), scores in all_scores.items():
            if layer_idx < len(scores):
                s = scores[layer_idx].flatten()
                ax.hist(s, bins=30, alpha=0.3, label=f"d{ds}_m{ms}")
        for thresh in [0.01, 0.03, 0.05, 0.1]:
            ax.axvline(thresh, color="red", ls="--", lw=0.8, alpha=0.6)
            ax.text(thresh, ax.get_ylim()[1] * 0.9, f"{thresh}",
                    fontsize=7, ha="left", color="red")
        ax.set_xlabel("Edge Importance Score", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Layer {layer_idx} Score Distribution", fontsize=11)
        ax.legend(fontsize=6)
    fig.suptitle("Edge Importance Score Distributions Across All Runs\n"
                 "Red lines = candidate thresholds", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Saved score_distributions.png")
    print(f"\n  Saved all plots to {OUTDIR}/")

    # Save summary
    # Include flattened scores for post-hoc analysis
    for entry in summary:
        ds, ms = entry["data_seed"], entry["model_seed"]
        if (ds, ms) in all_scores:
            entry["edge_scores"] = [s.tolist() for s in all_scores[(ds, ms)]]

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved results.json")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.0f} minutes")


if __name__ == "__main__":
    main()
