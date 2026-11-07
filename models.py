"""
Deep Koopman models with KAN encoder/decoder.

Two KAN backends:
    1. KANLinear (local kan.py) — custom B-spline layer, used for training
    2. PyKAN (pip install pykan) — for pruning, symbolic regression, plotting

Both feed into DeepKoopmanCT which learns the Koopman generator A and
propagates via z_{t+1} = z_t @ expm(A·dt)^T.
"""

import torch
import torch.nn as nn
from kan import KANLinear


# =====================================================================
# KANLinear-based autoencoder (two-layer, pure KAN)
# =====================================================================

class KANEncoder(nn.Module):
    """Two-layer KAN encoder: KANLinear(input→mid) → KANLinear(mid→latent).

    Layer 1 learns univariate basis functions ψ_{j,i}(x_i).
    Layer 2 mixes them, enabling cross-term discovery (e.g. xy, xz).
    """

    def __init__(
        self,
        input_size,
        mid_size,
        latent_size,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range_l1=[-30, 50],
        grid_range_l2=[-10, 10],
    ):
        super().__init__()
        kan_common = dict(
            grid_size=grid_size, spline_order=spline_order,
            scale_noise=scale_noise, scale_base=scale_base,
            scale_spline=scale_spline, base_activation=base_activation,
            grid_eps=grid_eps,
        )
        self.kan1 = KANLinear(input_size, mid_size,
                              grid_range=grid_range_l1, **kan_common)
        self.kan2 = KANLinear(mid_size, latent_size,
                              grid_range=grid_range_l2, **kan_common)

    def forward(self, x):
        return self.kan2(self.kan1(x))


class KANDecoder(nn.Module):
    """Two-layer KAN decoder: KANLinear(latent→mid) → KANLinear(mid→output)."""

    def __init__(
        self,
        latent_size,
        mid_size,
        output_size,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range_l1=[-10, 10],
        grid_range_l2=[-10, 10],
    ):
        super().__init__()
        kan_common = dict(
            grid_size=grid_size, spline_order=spline_order,
            scale_noise=scale_noise, scale_base=scale_base,
            scale_spline=scale_spline, base_activation=base_activation,
            grid_eps=grid_eps,
        )
        self.kan1 = KANLinear(latent_size, mid_size,
                              grid_range=grid_range_l1, **kan_common)
        self.kan2 = KANLinear(mid_size, output_size,
                              grid_range=grid_range_l2, **kan_common)

    def forward(self, x):
        return self.kan2(self.kan1(x))


class KANAutoEncoder(nn.Module):
    """Two-layer KAN autoencoder.

    Architecture: [input_size, mid_size, latent_size] per side.
    Pure B-spline composition — all nonlinearity from learned splines.
    """

    def __init__(self, input_size, mid_size, latent_size, **kwargs):
        super().__init__()
        self.encoder = KANEncoder(input_size, mid_size, latent_size, **kwargs)
        self.decoder = KANDecoder(latent_size, mid_size, input_size, **kwargs)

    def forward(self, X):
        Y = self.encoder(X)
        Xr = self.decoder(Y)
        return Y, Xr


# =====================================================================
# PyKAN-based autoencoder (for pruning, symbolic regression, plotting)
# =====================================================================

def _import_pykan():
    """Import PyKAN's KAN class, handling the local kan.py conflict."""
    import sys, os
    project_root = os.path.dirname(os.path.abspath(__file__))
    hidden = []
    for p in list(sys.path):
        if os.path.isfile(os.path.join(p, 'kan.py')) or p == project_root:
            sys.path.remove(p)
            hidden.append(p)
    if '' in sys.path:
        sys.path.remove('')
        hidden.append('')
    if 'kan' in sys.modules:
        del sys.modules['kan']
    from kan import KAN
    for p in reversed(hidden):
        sys.path.insert(0, p)
    if 'kan' in sys.modules:
        del sys.modules['kan']
    return KAN


class PyKANAutoEncoder(nn.Module):
    """Two-layer autoencoder using PyKAN's KAN class.

    Supports .prune(), .suggest_symbolic(), .auto_symbolic(), .plot().
    Architecture: KAN([input_size, mid_size, latent_size]) per side.
    """

    def __init__(self, input_size, mid_size, latent_size, grid=5, k=3):
        super().__init__()
        KAN = _import_pykan()
        self.encoder = KAN(width=[input_size, mid_size, latent_size],
                           grid=grid, k=k, seed=0)
        self.decoder = KAN(width=[latent_size, mid_size, input_size],
                           grid=grid, k=k, seed=1)

    def forward(self, X):
        Y = self.encoder(X)
        Xr = self.decoder(Y)
        return Y, Xr


# =====================================================================
# Deep Koopman model
# =====================================================================

class DeepKoopmanCT(nn.Module):
    """Continuous-time Deep Koopman model.

    Wraps any autoencoder (KANAutoEncoder or PyKANAutoEncoder) and learns
    generator matrix A. Propagates latent states via z_{t+1} = z_t @ expm(A·dt)^T.

    The autoencoder must have .encoder and .decoder attributes and return
    (encoded, reconstructed) from forward().
    """

    def __init__(self, autoencoder, latent_dim: int):
        super().__init__()
        self.ae = autoencoder
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

    def generator_matrix(self):
        return self.K.weight

    def expm_propagate(self, z_t, dt):
        A = self.generator_matrix()
        M = torch.matrix_exp(A * dt)
        return z_t @ M.T

    def forward_one_step(self, x_t, dt):
        z_t, x_t_rec = self.ae(x_t)
        z_tp1_pred = self.expm_propagate(z_t, dt)
        x_tp1_pred = self.ae.decoder(z_tp1_pred)
        return {
            "z_t": z_t,
            "x_t_rec": x_t_rec,
            "z_tp1_pred": z_tp1_pred,
            "x_tp1_pred": x_tp1_pred,
        }

    def load_state_dict(self, state_dict, **kwargs):
        # Handle old checkpoints that used Knet wrapper (K.net.weight → K.weight)
        if 'K.net.weight' in state_dict and 'K.weight' not in state_dict:
            state_dict['K.weight'] = state_dict.pop('K.net.weight')
        super().load_state_dict(state_dict, **kwargs)
