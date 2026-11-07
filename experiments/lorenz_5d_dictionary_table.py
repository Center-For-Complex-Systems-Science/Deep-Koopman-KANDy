"""
Reproduce the dictionary recovery table for the pruned+retrained model.

Loads the prune@0.03 + retrained model and runs level-set decomposition
at Lasso alpha=5e-3 to produce the paper table.

Also runs the full alpha sweep across all pruned/unpruned models
for the supplementary material.

Run from project root:
    python experiments/lorenz_5d_dictionary_table.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from data import simulate_lorenz
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition

OUTDIR = Path("results/lorenz_5d_third_test")
OUTDIR.mkdir(parents=True, exist_ok=True)
LATENT = 5
TRUE = {"x", "y", "z", "xy", "xz"}
LASSO_ALPHAS = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

MODELS = {
    "unpruned":   "results/lorenz_5d_second_test/model.pt",
    "prune@0.01": "results/lorenz_5d_third_test/model_pruned_0.01.pt",
    "prune@0.03": "results/lorenz_5d_third_test/model_pruned_0.03.pt",
    "prune@0.05": "results/lorenz_5d_third_test/model_pruned_0.05.pt",
}


def load_model(path):
    ae = PyKANAutoEncoder(3, 5, LATENT, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, LATENT)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_decomposition(model, x_sub, alpha):
    results = []
    for zi in range(LATENT):
        dr = full_decomposition(model, x_sub, zi, target_type='latent',
                                max_degree=3, lasso_alpha=alpha, h_degree=4)
        pairs = sorted(zip(dr['inner']['active_names'],
                           dr['inner']['active_coeffs']),
                       key=lambda x: -abs(x[1]))
        pairs = [(n, c) for n, c in pairs if n != "1"]
        true_terms = [(n, c) for n, c in pairs if n in TRUE]
        fp_terms = [(n, c) for n, c in pairs if n not in TRUE]

        h_coeffs = dr['outer'].get('h_coeffs', None)
        h_str = ""
        if h_coeffs is not None:
            for deg, c in enumerate(h_coeffs):
                if deg == 0:
                    continue
                if abs(c) > 1e-4:
                    h_str += f"{c:+.2f}u^{deg} " if deg > 1 else f"{c:+.2f}u "
        h_str = h_str.strip() or "const"

        results.append({
            "dim": zi,
            "r2_g": float(dr['g_recon_r2']),
            "r2_hg": float(dr['recon_r2']),
            "h": h_str,
            "true_terms": [(n, float(c)) for n, c in true_terms],
            "fp_terms": [(n, float(c)) for n, c in fp_terms],
            "n_true": len(true_terms),
            "n_fp": len(fp_terms),
        })
    return results


def print_table(decomp, title=""):
    print(f"\n  {title}")
    print(f"  {'Dim':>4} {'R²(h∘g)':>8} | {'TRUE terms':>45} | {'FP terms'}")
    print(f"  " + "-" * 90)
    all_true = set()
    all_fp = set()
    for d in decomp:
        true_str = "  ".join(f"{c:+.4f}*{n}" for n, c in d['true_terms'])
        fp_str = "  ".join(f"{c:+.4f}*{n}" for n, c in d['fp_terms'])
        print(f"  z[{d['dim']}] {d['r2_hg']:>8.3f} | {true_str:>45} | {fp_str}")
        for n, c in d['true_terms']:
            all_true.add(n)
        for n, c in d['fp_terms']:
            all_fp.add(n)
    tp = len(all_true & TRUE)
    fp = len(all_fp)
    rec = tp / len(TRUE)
    prec = tp / max(tp + fp, 1)
    jac = tp / max(len(all_true | TRUE) + fp - tp, 1)
    print(f"\n  Union TRUE: {sorted(all_true & TRUE)}")
    print(f"  Union FP:   {sorted(all_fp)}")
    print(f"  Recall={rec:.0%}  Precision={prec:.2f}  Jaccard={jac:.3f}")
    return rec, prec, jac


def main():
    # Data
    set_seed(42)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=0.01, burn_in=500, device="cpu")
    x_flat = traj[:48].reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)

    # ══════════════════════════════════════════════════════════════
    # MAIN TABLE: prune@0.03 at alpha=5e-3
    # ══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  MAIN TABLE: prune@0.03 + retrain, Lasso alpha=5e-3")
    print("=" * 70)

    model = load_model(MODELS["prune@0.03"])
    decomp = run_decomposition(model, x_sub, alpha=5e-3)
    print_table(decomp, "Prune@0.03 + Retrain, alpha=5e-3")
    del model

    # ══════════════════════════════════════════════════════════════
    # FULL SWEEP: all models × all alphas
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*90}")
    print(f"  FULL SWEEP: all models × all alphas")
    print(f"{'='*90}")

    all_results = {}
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"  {model_name}: not found, skipping")
            continue
        model = load_model(model_path)
        model_results = {}
        for alpha in LASSO_ALPHAS:
            decomp = run_decomposition(model, x_sub, alpha)
            all_true = set()
            all_fp = set()
            r2s = []
            for d in decomp:
                r2s.append(d['r2_hg'])
                for n, c in d['true_terms']:
                    all_true.add(n)
                for n, c in d['fp_terms']:
                    all_fp.add(n)
            tp = len(all_true & TRUE)
            fp = len(all_fp)
            rec = tp / len(TRUE)
            prec = tp / max(tp + fp, 1)
            jac = tp / max(len(all_true | TRUE) + fp - tp, 1)
            model_results[str(alpha)] = {
                "med_r2": float(np.median(r2s)),
                "recall": rec,
                "precision": prec,
                "jaccard": jac,
                "mean_true": float(np.mean([d['n_true'] for d in decomp])),
                "mean_fp": float(np.mean([d['n_fp'] for d in decomp])),
                "decomposition": decomp,
            }
        all_results[model_name] = model_results
        del model

    # Summary table
    print(f"\n  {'Model':>14} {'alpha':>7} {'med R²':>7} {'#true':>6} {'#FP':>5} "
          f"{'Recall':>7} {'Prec':>6} {'Jaccard':>8}")
    print("  " + "-" * 65)
    for model_name in MODELS:
        if model_name not in all_results:
            continue
        for alpha in LASSO_ALPHAS:
            r = all_results[model_name][str(alpha)]
            print(f"  {model_name:>14} {alpha:>7.0e} {r['med_r2']:>7.3f} "
                  f"{r['mean_true']:>6.1f} {r['mean_fp']:>5.1f} "
                  f"{r['recall']:>7.0%} {r['precision']:>6.2f} {r['jaccard']:>8.3f}")
        print()

    # Best per model
    print(f"\n  BEST (max Jaccard at 100% recall):")
    print(f"  {'Model':>14} {'alpha':>7} {'Jaccard':>8} {'Prec':>6} {'R²':>7} {'#FP':>5}")
    print("  " + "-" * 55)
    for model_name in MODELS:
        if model_name not in all_results:
            continue
        best_alpha, best_jac = None, 0
        for alpha in LASSO_ALPHAS:
            r = all_results[model_name][str(alpha)]
            if r['recall'] >= 1.0 and r['jaccard'] > best_jac:
                best_jac = r['jaccard']
                best_alpha = alpha
        if best_alpha:
            r = all_results[model_name][str(best_alpha)]
            print(f"  {model_name:>14} {best_alpha:>7.0e} {r['jaccard']:>8.3f} "
                  f"{r['precision']:>6.2f} {r['med_r2']:>7.3f} {r['mean_fp']:>5.1f}")

    # Save
    with open(OUTDIR / "dictionary_table.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR / 'dictionary_table.json'}")


if __name__ == "__main__":
    main()
