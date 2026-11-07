"""
Plot PyKAN-style network diagrams for the trained KANLinear Lorenz model.

Strategy: create PyKAN KAN objects with matching widths, distill our trained
KANLinear encoder/decoder into them (fit input→output pairs), then call
PyKAN's model.plot() for the nice LaTeX diagrams.
"""

import os, sys

# Import pykan from site-packages (avoid local kan.py conflict)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_hidden = []
for p in list(sys.path):
    if os.path.isfile(os.path.join(p, 'kan.py')) or p == _project_root:
        sys.path.remove(p)
        _hidden.append(p)
if '' in sys.path:
    sys.path.remove('')
    _hidden.append('')
if 'kan' in sys.modules:
    del sys.modules['kan']

from kan import KAN

# Restore paths
for p in reversed(_hidden):
    sys.path.insert(0, p)
if 'kan' in sys.modules:
    del sys.modules['kan']

sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from data import simulate_lorenz
from models import PureDeepKANAutoEncoder, DeepKoopmanCT
from training import set_seed

OUTDIR = Path('figs')


def load_model():
    ckpt = torch.load(os.path.join(_project_root, 'old', 'kan_pure_3_3_10_model.pt'),
                      map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    ae = PureDeepKANAutoEncoder(
        input_size=3, mid_size=cfg['mid_size'],
        bottleneck_size=cfg['latent_dim'],
        grid_size=5, spline_order=3,
        grid_range_l1=[-30, 50], grid_range_l2=[-10, 10],
    )
    model = DeepKoopmanCT(ae, latent_dim=cfg['latent_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def distill_into_pykan(teacher_fn, pykan_model, x_train, epochs=200, lr=1e-2):
    """Train a PyKAN model to match teacher_fn(x_train) outputs."""
    with torch.no_grad():
        y_target = teacher_fn(x_train)

    opt = torch.optim.Adam(pykan_model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        y_pred = pykan_model(x_train)
        loss = mse(y_pred, y_target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 50 == 0 or ep == 1:
            print(f"    Distill ep {ep:3d}: loss={loss.item():.6f}")

    # Final forward pass to populate internal activation stats for plotting
    with torch.no_grad():
        pykan_model(x_train)

    return loss.item()


if __name__ == '__main__':
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    print("Loading trained KANLinear model...")
    model, cfg = load_model()
    enc = model.ae.encoder  # PureDeepKANEncoder: kan1(3→3) → kan2(3→10)
    dec = model.ae.decoder  # PureDeepKANDecoder: kan1(10→3) → kan2(3→3)

    print("Generating Lorenz data...")
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=0.01, burn_in=500)
    flat = traj.reshape(-1, 3)
    idx = torch.randperm(flat.shape[0])[:2000]
    x_data = flat[idx]

    # Get latent representations for decoder distillation
    with torch.no_grad():
        z_data = enc(x_data)

    # ---- Distill encoder into PyKAN ----
    print("\nDistilling encoder: KAN([3, 3, 10])...")
    pykan_enc = KAN(width=[3, 3, 10], grid=5, k=3, seed=0)
    pykan_enc(x_data[:10])  # init
    pykan_enc.update_grid_from_samples(x_data)
    enc_loss = distill_into_pykan(enc, pykan_enc, x_data, epochs=300, lr=1e-2)
    print(f"  Final encoder distillation loss: {enc_loss:.6f}")

    # ---- Distill decoder into PyKAN ----
    print("\nDistilling decoder: KAN([10, 3, 3])...")
    pykan_dec = KAN(width=[10, 3, 3], grid=5, k=3, seed=1)
    pykan_dec(z_data[:10])  # init
    pykan_dec.update_grid_from_samples(z_data)
    dec_loss = distill_into_pykan(dec, pykan_dec, z_data, epochs=300, lr=1e-2)
    print(f"  Final decoder distillation loss: {dec_loss:.6f}")

    # ---- Plot encoder ----
    print("\nPlotting encoder...")
    # Run representative data through for activation stats
    with torch.no_grad():
        pykan_enc(x_data)

    enc_fig = pykan_enc.plot(
        beta=3,
        in_vars=[r'$x$', r'$y$', r'$z$'],
        out_vars=[r'$z_0$', r'$z_1$', r'$z_2$', r'$z_3$', r'$z_4$',
                  r'$z_5$', r'$z_6$', r'$z_7$', r'$z_8$', r'$z_9$'],
        title=r'Encoder:  $(x, y, z) \to \mathbf{z}$',
    )

    # pykan.plot() creates its own figure — save via plt
    plt.savefig(OUTDIR / 'pykan_encoder.png', dpi=200, bbox_inches='tight')
    plt.savefig(OUTDIR / 'pykan_encoder.pdf', bbox_inches='tight')
    plt.close('all')
    print(f"  Saved pykan_encoder.png/pdf")

    # ---- Plot decoder ----
    print("Plotting decoder...")
    with torch.no_grad():
        pykan_dec(z_data)

    dec_fig = pykan_dec.plot(
        beta=3,
        in_vars=[r'$z_0$', r'$z_1$', r'$z_2$', r'$z_3$', r'$z_4$',
                 r'$z_5$', r'$z_6$', r'$z_7$', r'$z_8$', r'$z_9$'],
        out_vars=[r'$\hat{x}$', r'$\hat{y}$', r'$\hat{z}$'],
        title=r'Decoder:  $\mathbf{z} \to (\hat{x}, \hat{y}, \hat{z})$',
    )

    plt.savefig(OUTDIR / 'pykan_decoder.png', dpi=200, bbox_inches='tight')
    plt.savefig(OUTDIR / 'pykan_decoder.pdf', bbox_inches='tight')
    plt.close('all')
    print(f"  Saved pykan_decoder.png/pdf")

    print(f"\nDone. Figures in {OUTDIR}/")
