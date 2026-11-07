"""Train PyKAN [3,8,32] Deep Koopman on Lorenz and test dictionary recovery."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import PyKANAutoEncoder, DeepKoopmanCT
from data import simulate_lorenz, PairDataset
from training import set_seed
from manifold_analysis import full_decomposition

set_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ae = PyKANAutoEncoder(3, 8, 32, grid=5, k=3)
model = DeepKoopmanCT(ae, latent_dim=32).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'PyKAN [3,8,32] + Koopman(32x32): {n_params:,} params')
print(f'Device: {device}')

traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=0.01, burn_in=300, device=device)
n_train = 48
train_dl = DataLoader(PairDataset(traj[:n_train]), batch_size=2048,
                      shuffle=True, drop_last=True)
val_dl = DataLoader(PairDataset(traj[n_train:]), batch_size=2048)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
mse = nn.MSELoss()

# Grid init
x_sample = traj[:n_train].reshape(-1, 3)[:2000]
with torch.no_grad():
    z_sample = ae.encoder(x_sample)
ae.encoder.update_grid_from_samples(x_sample)
ae.decoder.update_grid_from_samples(z_sample)
print('Grids initialized')

for ep in range(1, 151):
    model.train()
    losses = []
    for x_t, x_tp1 in train_dl:
        x_t, x_tp1 = x_t.to(device), x_tp1.to(device)
        out = model.forward_one_step(x_t, 0.01)
        z_true = model.ae.encoder(x_tp1)
        loss = (mse(out['x_tp1_pred'], x_tp1)
                + mse(out['z_tp1_pred'], z_true)
                + 0.5 * mse(out['x_t_rec'], x_t))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    if ep % 20 == 0:
        with torch.no_grad():
            idx = torch.randperm(x_sample.shape[0])[:1000]
            ae.encoder.update_grid_from_samples(x_sample[idx])
            z_up = ae.encoder(x_sample[idx])
            ae.decoder.update_grid_from_samples(z_up)

    if ep == 1 or ep % 10 == 0 or ep == 150:
        model.eval()
        vloss = []
        with torch.no_grad():
            for x_t, x_tp1 in val_dl:
                x_t, x_tp1 = x_t.to(device), x_tp1.to(device)
                out = model.forward_one_step(x_t, 0.01)
                z_true = model.ae.encoder(x_tp1)
                l = (mse(out['x_tp1_pred'], x_tp1)
                     + mse(out['z_tp1_pred'], z_true)
                     + 0.5 * mse(out['x_t_rec'], x_t))
                vloss.append(l.item())
        print(f'Ep {ep:3d} | train {np.mean(losses):.5f} | val {np.mean(vloss):.5f}',
              flush=True)

# Save
model_cpu = model.cpu().eval()
torch.save({
    'model_state_dict': model_cpu.state_dict(),
    'dt': 0.01,
    'config': {'latent_dim': 32, 'mid_size': 8},
}, 'old/pykan_3_8_32_model.pt')
print('Saved old/pykan_3_8_32_model.pt')

# Decomposition
flat = traj.reshape(-1, 3).cpu()
idx = torch.randperm(flat.shape[0])[:5000]
x_data = flat[idx]

print(f'\nDICTIONARY RECOVERY')
print(f'{"Dim":>4}  {"R²(g)":>8}  {"R²(h∘g)":>8}  Cross-terms')
print('-' * 70)

cross_all = {}
for ti in range(32):
    dr = full_decomposition(model_cpu, x_data, ti, target_type='latent',
                            max_degree=3, lasso_alpha=1e-5, h_degree=4)
    names = dr['inner']['active_names']
    coeffs = dr['inner']['active_coeffs']
    cm = dict(zip(names, coeffs))

    interesting = {t: cm[t] for t in ['xy', 'xz', 'yz']
                   if t in cm and abs(cm[t]) > 0.005}
    for t, c in interesting.items():
        cross_all[t] = max(cross_all.get(t, 0), abs(c))

    if dr['recon_r2'] > 0.95 or interesting:
        int_str = '  '.join(f'{t}={c:+.4f}' for t, c in interesting.items()) \
                  if interesting else 'linear'
        print(f'z[{ti:2d}]  {dr["g_recon_r2"]:8.3f}  {dr["recon_r2"]:8.3f}  {int_str}')

print(f'\nCross-terms found (max |coeff|): {cross_all}')
