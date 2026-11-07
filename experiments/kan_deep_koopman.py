"""KANDy two-layer experiment: Deep KAN Koopman on Lorenz system.

Uses a two-layer KAN encoder following the KAT construction:
  Layer 1: KANLinear(3 → mid)    — univariate basis functions ψ_{j,i}(x_i)
  Layer 2: KANLinear(mid → hidden) — mixing layer, enables cross-terms (xy, xz)
  Then:    ReLU → Dense(hidden → latent) → Koopman matrix

After training, we extract:
  - Layer 1: symbolic univariate functions (same as single-layer case)
  - Layer 2: level set analysis to identify mixing/product terms
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models import DeepKANAutoEncoder
from training import train_deep_koopman_ct
from plotting import make_plots
from symbolic import summarize_deep_encoder_lift


if __name__ == "__main__":
    latent_dim = 10
    mid_size = 8       # layer 1 output: univariate basis functions
    hidden_size = 32   # layer 2 output: mixed observables

    ae = DeepKANAutoEncoder(
        input_size=3,
        mid_size=mid_size,
        hidden_size=hidden_size,
        bottleneck_size=latent_dim,
        grid_size=5,
        spline_order=3,
        grid_range_l1=[-30, 50],
        grid_range_l2=[-10, 10],
    )

    print(f"DeepKAN AE params: {sum(p.numel() for p in ae.parameters()):,}")

    model, history, train_traj, val_traj, dt = train_deep_koopman_ct(
        autoencoder=ae,
        dt=0.01,
        n_traj=60,
        n_steps=6000,
        burn_in=300,
        batch_size=2048,
        epochs=80,
        lr=1e-3,
        latent_dim=latent_dim,
        alpha_latent=1.0,
        beta_recon=0.5,
        gamma_predx=1.0,
        grad_clip=1.0,
        seed=0,
    )

    # Save model
    save_path = os.path.join(os.path.dirname(__file__), "..", "kan_deep_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'dt': dt,
        'config': {
            'latent_dim': latent_dim,
            'mid_size': mid_size,
            'hidden_size': hidden_size,
        },
    }, save_path)
    print(f"Model saved to {save_path}")

    make_plots(model, history, train_traj, val_traj, dt, rollout_steps=2500)

    # --- Symbolic extraction ---
    print("\n\n" + "=" * 70)
    print("SYMBOLIC EXTRACTION: TWO-LAYER KAN KOOPMAN LIFT")
    print("=" * 70)

    device = next(model.parameters()).device
    x_states = train_traj.reshape(-1, 3).to(device)

    summarize_deep_encoder_lift(
        model.ae.encoder, x_states, lam=0.1, n_samples=2000,
    )
