"""
Level-set decomposition of the [3,5,5] stable Koopman model on Lorenz.

Reproduces the table:
  - max_degree=3, lasso_alpha=1e-5
  - Full data and 95th percentile crop
  - Prints g(x) dominant terms, R²(g), R²(h∘g), h(u) polynomial

Run from project root:
    python experiments/lorenz_5d_decomposition.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

from data import simulate_lorenz
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable
from manifold_analysis import full_decomposition

OUTDIR = Path("results/lorenz_5d")
DT = 0.01
LATENT = 5


def load_model():
    ae = PyKANAutoEncoder(3, 5, LATENT, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, LATENT)
    ckpt = torch.load(OUTDIR / "model_5_5.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def format_h(h_coeffs):
    """Format h(u) polynomial, omitting constant."""
    terms = []
    for deg, c in enumerate(h_coeffs):
        if deg == 0:
            continue  # skip constant
        if abs(c) < 1e-5:
            continue
        if deg == 1:
            terms.append(f"{c:+.2f}u")
        else:
            terms.append(f"{c:+.2f}u^{deg}")
    return " ".join(terms) if terms else "const"


def run_decomposition(model, x_sub, label=""):
    print(f"\n  {label}")
    print(f"  {'Dim':>4}  {'R2(g)':>8}  {'R2(hog)':>8}  {'h(u)':>25}  g (top terms)")
    print("  " + "-" * 95)

    for zi in range(LATENT):
        dr = full_decomposition(model, x_sub, zi, target_type='latent',
                                max_degree=3, lasso_alpha=1e-5, h_degree=4)

        # g top terms
        pairs = sorted(zip(dr['inner']['active_names'],
                           dr['inner']['active_coeffs']),
                       key=lambda x: -abs(x[1]))[:6]
        g_expr = '  '.join(f'{c:+.4f}*{n}' for n, c in pairs)

        # h polynomial
        h_coeffs = dr['outer'].get('h_coeffs', None)
        h_str = format_h(h_coeffs) if h_coeffs is not None else "N/A"

        print(f"  z[{zi}]  {dr['g_recon_r2']:8.3f}  {dr['recon_r2']:8.3f}  "
              f"{h_str:>25}  {g_expr}")


def main():
    print("=" * 70)
    print("  LORENZ [3,5,5] LEVEL-SET DECOMPOSITION")
    print("=" * 70)

    model = load_model()

    set_seed(42)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:48]
    x_flat = train_traj.reshape(-1, 3)

    # Full data
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    run_decomposition(model, x_sub, label="max_degree=3, lasso_alpha=1e-5, full data")

    # 2.5-97.5 percentile crop
    mask = np.ones(len(x_flat), dtype=bool)
    for i in range(3):
        lo, hi = np.percentile(x_flat[:, i].numpy(), [2.5, 97.5])
        mask &= (x_flat[:, i].numpy() >= lo) & (x_flat[:, i].numpy() <= hi)
    x_cropped = x_flat[mask]
    idx2 = np.random.choice(len(x_cropped), min(5000, len(x_cropped)), replace=False)
    x_sub2 = torch.tensor(x_cropped[idx2].numpy(), dtype=torch.float32)
    print(f"\n  Cropped: {len(x_flat)} -> {len(x_cropped)} points")
    run_decomposition(model, x_sub2, label="max_degree=3, lasso_alpha=1e-5, 2.5-97.5% crop")


if __name__ == "__main__":
    main()
