"""Train Pure KAN Deep Koopman on Lorenz with [3,3,10] encoder."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models import PureDeepKANAutoEncoder, DeepKoopmanCT
from training import train_deep_koopman_ct, set_seed
from plotting import make_plots

set_seed(0)

ae = PureDeepKANAutoEncoder(
    input_size=3,
    mid_size=3,
    bottleneck_size=10,
    grid_size=5,
    spline_order=3,
    grid_range_l1=[-30, 50],
    grid_range_l2=[-10, 10],
)

print(f"Encoder: KANLinear(3→3) → KANLinear(3→10)")
print(f"Decoder: KANLinear(10→3) → KANLinear(3→3)")
print(f"Parameters: {sum(p.numel() for p in ae.parameters()):,}")

model, history, train_traj, val_traj, dt = train_deep_koopman_ct(
    ae,
    dt=0.01,
    n_traj=60,
    n_steps=6000,
    burn_in=500,
    batch_size=2048,
    epochs=150,
    lr=1e-3,
    latent_dim=10,
    alpha_latent=1.0,
    beta_recon=0.5,
    gamma_predx=1.0,
    grad_clip=1.0,
    seed=0,
)

# Save
ckpt = {
    'model_state_dict': model.state_dict(),
    'history': history,
    'dt': dt,
    'config': {'latent_dim': 10, 'mid_size': 3},
}
save_path = os.path.join(os.path.dirname(__file__), '..', 'old', 'kan_pure_3_3_10_model.pt')
torch.save(ckpt, save_path)
print(f"\nSaved to {save_path}")
print(f"Final val loss: {history['val_total'][-1]:.5f}")
