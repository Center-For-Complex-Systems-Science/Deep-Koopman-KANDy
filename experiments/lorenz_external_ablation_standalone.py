"""

"We're going to need a bigger boat." - Shark man from Jaws

I am rewritting this script to run on an A100 because it is taking forever.

Symbolic Dictionary Recovery Benchmark on the Lorenz System
============================================================

Purpose
-------
This experiment tests the central interpretability claim of Deep-Koopman-KANDy:
that the combination of a two-layer KAN encoder with level-set manifold analysis
recovers the minimal Koopman-invariant polynomial dictionary {x, y, z, xy, xz}
more cleanly than alternative pipelines that also aim to produce symbolic forms
for learned latent coordinates.

The comparison is scoped specifically to the recovery of symbolic expressions
for the ENCODER COORDINATES z_k(x), not to the recovery of latent dynamics.
All three methods compared here are forced through the same scoring pipeline
(Jaccard / precision / recall against the target dictionary) so that the
comparison is apples-to-apples on the interpretability-of-the-lifting axis.

Methods Compared
----------------
1. Deep-Koopman-KANDy (this work).
   Two-layer KAN encoder/decoder trained with the stable Koopman generator
   A = K - L^T L. Symbolic recovery via level-set manifold analysis: each
   latent coordinate z_k is decomposed as f(x) = h(g(x)) with inner function
   g recovered by Lasso on a degree-3 monomial basis and outer function h
   recovered via the gradient formula h'(g) = (grad f . grad g) / ||grad g||^2.
   (Protocol matches Table 1 of the main paper.)

2. Deep-Koopman (MLP) + PySR on encoder.
   The "MLP + stable" ablation row from Table 1 of the main paper, trained
   identically to Deep-Koopman-KANDy but with MLP encoder/decoder. After
   training, the encoder is frozen and PySR is run independently on each
   latent coordinate z_k(x) to produce a symbolic expression. This isolates
   the contribution of the KAN architecture to symbolic recoverability.

3. SINDy-AE (Champion, Lusch, Kutz, Brunton, PNAS 2019) + PySR on encoder.
   The full SINDy-AE model is trained on the same Lorenz data using the
   reference implementation at github.com/kpchamp/SindyAutoencoders. After
   training, the SINDy equation for z-dot is discarded and PySR is run on
   each latent coordinate z_k(x) from the trained encoder. NOTE: this is NOT
   SINDy-AE's native output; SINDy-AE natively outputs a sparse equation for
   latent dynamics, not for the encoder. We reroute its output through PySR
   so the comparison is commensurable. This choice is explicit and
   conservative -- it tests whether SINDy-AE's joint objective yields an
   encoder whose coordinates happen to be recoverable in symbolic form.

Data
----
Lorenz system (sigma=10, rho=28, beta=8/3). 60 trajectories of 6000 steps
with RK4 at dt=0.01, 500-step burn-in, initial conditions drawn uniformly
from [-20, 20] x [-30, 30] x [0, 50]. First 80% used for training, 20% held
out for test. Identical training data across all three methods.

Derivatives: SINDy-AE requires x-dot. Analytical derivatives from the RK4
integrator are provided rather than finite-differenced from snapshots. This
is the charitable choice and removes a confound reviewers would otherwise
raise. The Deep-Koopman methods use the existing (x_t, x_{t+1}) snapshot
pairs with no derivative access.

Symbolic Regression Protocol (methods 2 and 3)
----------------------------------------------
PySR is run with an identical configuration for both methods:
- Operator set: {+, -, *} (sufficient for the polynomial target; no
  transcendentals, which would invite PySR to prefer short trigonometric
  surrogates for xy and xz).
- Maximum complexity: 20.
- Populations: 40, iterations: 200.
- Loss: MSE against the frozen encoder's output z_k(x) on the training set.
- For each latent coordinate z_k, PySR is run with 5 independent seeds; the
  returned expression is selected as the one with the lowest loss at the
  lowest complexity tier that achieves R^2 >= 0.95 on a held-out validation
  split of the encoder outputs. If no expression clears 0.95, the lowest-
  loss expression at any complexity is taken.

Model Selection
---------------
Each of the three methods is trained with 5 random seeds, matching the
protocol of Table 1 in the main paper. Metrics are reported as mean +- std
over seeds. This is a conservative evaluation of SINDy-AE, whose reference
protocol uses best-of-10 with manual model selection; we note this
explicitly in the paper text.

Metrics (extending Table 1)
---------------------------
For each method, after symbolic expressions have been obtained for every
latent coordinate (via level-set analysis for method 1, via PySR for
methods 2 and 3), the union of nonzero monomial terms across all latent
coordinates is compared to the target set {x, y, z, xy, xz} via:
- Jaccard similarity: |recovered ∩ target| / |recovered ∪ target|
- Precision: |recovered ∩ target| / |recovered|
- Recall: |recovered ∩ target| / |target|

A monomial is considered "recovered" for methods 2 and 3 if it appears in
any PySR-returned expression with coefficient magnitude above 1e-2 after
standardization (matching the threshold used for the level-set analysis
in the main paper).

One-step MSE and NRMSE at 2*tau_Lambda are also reported for continuity
with Table 1, though these are secondary -- the primary claim tested by
this experiment is dictionary recoverability, not predictive accuracy.

Output
------
Produces one extended row per method for Table 1 in the main paper, plus
a supplementary table listing the top-k PySR-recovered expressions per
latent coordinate for methods 2 and 3. Raw PySR outputs are logged for
reproducibility.

Expected Outcome (prior, not a commitment)
------------------------------------------
- Method 1 (ours): Jaccard = 1.0, matching Table 1.
- Method 2 (MLP + PySR): Jaccard ≈ 0.4-0.6. Linear terms (x, y, z) likely
  recovered; cross-terms (xy, xz) likely missed because MLP latents are
  typically tangled affine mixtures where polynomial structure does not
  dominate the PySR loss landscape.
- Method 3 (SINDy-AE + PySR): Jaccard ≈ 0.5-0.7. SINDy-AE's joint objective
  pressures the encoder toward latents with sparse polynomial dynamics,
  which indirectly biases them toward polynomial-friendly forms, but the
  encoder itself is not regularized for sparse symbolic structure.

Deviations from these priors are informative either way and should be
reported faithfully.

References
----------
- Champion, Lusch, Kutz, Brunton. "Data-driven discovery of coordinates
  and governing equations." PNAS 116(45): 22445-22451, 2019.
  Code: github.com/kpchamp/SindyAutoencoders
- Cranmer. "Interpretable Machine Learning for Science with PySR and
  SymbolicRegression.jl." arXiv:2305.01582, 2023.
  Code: github.com/MilesCranmer/PySR
"""

import sys, os, csv, json, time, logging
from pathlib import Path

from sklearn.linear_model import Lasso

# PySR / juliacall must be imported BEFORE torch to avoid segfault.
# See https://github.com/pytorch/pytorch/issues/78829
import pysr  # noqa: F401 — early import to initialize Julia before torch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

import matplotlib
matplotlib.use("Agg")

from data import simulate_lorenz, PairDataset, lorenz_rhs
from models import KANAutoEncoder
from kan import KANLinear

#from lorenz_stable import DeepKoopmanCTStable
#from manifold_analysis import full_decomposition
#from sindy_autoencoder import SindyAutoencoder, sindy_loss
from math import comb

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SINDy library construction
# ---------------------------------------------------------------------------

def compute_library_size(n, poly_order, include_sine=False):
    """Number of terms in a polynomial library of n variables up to poly_order.

    Matches the enumeration used in sindy_library (constant + monomials of
    degree 1..poly_order, using combinations-with-repetition ordering, plus
    optional sin(z_i) terms).
    """
    size = 1  # constant term
    for d in range(1, poly_order + 1):
        size += comb(n + d - 1, d)
    if include_sine:
        size += n
    return size


def sindy_library(z, poly_order, include_sine=False):
    """Polynomial SINDy library up to poly_order (max 5), optional sine terms.

    Args:
        z: Tensor of shape (batch, latent_dim).
        poly_order: int in {1, 2, 3, 4, 5}.
        include_sine: if True, appends sin(z_i) for each i.

    Returns:
        Tensor of shape (batch, library_dim). Term ordering matches the
        original TF implementation so SINDy coefficient matrices are directly
        transferable.
    """
    batch = z.shape[0]
    latent_dim = z.shape[1]
    library = [torch.ones(batch, device=z.device, dtype=z.dtype)]

    for i in range(latent_dim):
        library.append(z[:, i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(z[:, i] * z[:, j])

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(z[:, i] * z[:, j] * z[:, k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for q in range(p, latent_dim):
                            library.append(
                                z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q]
                            )

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:, i]))

    return torch.stack(library, dim=1)


def sindy_library_order2(z, dz, poly_order, include_sine=False):
    """Second-order SINDy library: same enumeration over concatenation [z, dz]."""
    z_combined = torch.cat([z, dz], dim=1)
    return sindy_library(z_combined, poly_order, include_sine)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class SindyAutoencoder(nn.Module):
    """PyTorch SINDy Autoencoder (Champion et al. 2019).

    Args:
        input_dim: Dimension of raw state x.
        latent_dim: Dimension of reduced coordinates z.
        widths: List of hidden-layer widths for encoder (decoder uses reversed).
                Example: [64, 32] means encoder is input_dim -> 64 -> 32 -> latent_dim.
        activation: One of 'elu', 'relu', 'sigmoid', 'linear'.
        poly_order: Max polynomial degree in the SINDy library (<= 5).
        include_sine: If True, adds sin(z_i) terms to the library.
        model_order: 1 for systems dz/dt = f(z); 2 for ddz/dt^2 = f(z, dz).
        coefficient_init: 'xavier', 'constant' (1.0), or 'normal'.
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        widths,
        activation='sigmoid',
        poly_order=3,
        include_sine=False,
        model_order=1,
        coefficient_init='constant',
    ):
        super().__init__()

        if activation not in {'elu', 'relu', 'sigmoid', 'linear'}:
            raise ValueError(f"Unknown activation: {activation!r}")
        if poly_order not in {1, 2, 3, 4, 5}:
            raise ValueError(f"poly_order must be in 1..5, got {poly_order}")
        if model_order not in {1, 2}:
            raise ValueError(f"model_order must be 1 or 2, got {model_order}")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation_name = activation
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.model_order = model_order

        # Library dimension depends on whether we stack [z, dz] for order 2
        n_vars = latent_dim if model_order == 1 else 2 * latent_dim
        self.library_dim = compute_library_size(n_vars, poly_order, include_sine)

        # --- Encoder: input_dim -> widths[0] -> ... -> widths[-1] -> latent_dim
        encoder_dims = [input_dim] + list(widths) + [latent_dim]
        self.encoder_layers = nn.ModuleList([
            nn.Linear(encoder_dims[i], encoder_dims[i + 1])
            for i in range(len(encoder_dims) - 1)
        ])

        # --- Decoder: latent_dim -> widths[-1] -> ... -> widths[0] -> input_dim
        decoder_dims = [latent_dim] + list(reversed(widths)) + [input_dim]
        self.decoder_layers = nn.ModuleList([
            nn.Linear(decoder_dims[i], decoder_dims[i + 1])
            for i in range(len(decoder_dims) - 1)
        ])

        # Xavier uniform init, zero biases — matches tf.contrib.layers.xavier_initializer
        for layer in list(self.encoder_layers) + list(self.decoder_layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # --- SINDy coefficients (trainable)
        if coefficient_init == 'xavier':
            coefs = torch.empty(self.library_dim, latent_dim)
            nn.init.xavier_uniform_(coefs)
        elif coefficient_init == 'constant':
            coefs = torch.ones(self.library_dim, latent_dim)
        elif coefficient_init == 'normal':
            coefs = torch.randn(self.library_dim, latent_dim)
        else:
            raise ValueError(f"Unknown coefficient_init: {coefficient_init!r}")
        self.sindy_coefficients = nn.Parameter(coefs)

        # --- Coefficient mask (non-trainable buffer, updated by apply_thresholding)
        self.register_buffer(
            'coefficient_mask',
            torch.ones(self.library_dim, latent_dim),
        )

    # ------------------------------------------------------------------
    # Activation + its derivatives
    # ------------------------------------------------------------------

    def _activation(self, pre):
        name = self.activation_name
        if name == 'elu':
            return F.elu(pre)
        if name == 'relu':
            return F.relu(pre)
        if name == 'sigmoid':
            return torch.sigmoid(pre)
        return pre  # linear

    def _activation_deriv(self, pre):
        """First derivative of activation at pre-activation values."""
        name = self.activation_name
        if name == 'elu':
            # ELU'(x) = 1 if x > 0 else exp(x); min(exp(x), 1) captures both
            return torch.minimum(torch.exp(pre), torch.ones_like(pre))
        if name == 'relu':
            return (pre > 0).to(pre.dtype)
        if name == 'sigmoid':
            s = torch.sigmoid(pre)
            return s * (1.0 - s)
        return torch.ones_like(pre)  # linear

    def _activation_deriv2(self, pre):
        """Second derivative of activation at pre-activation values."""
        name = self.activation_name
        if name == 'elu':
            # ELU''(x) = exp(x) for x < 0, 0 for x >= 0.
            # (Undefined at x=0; the original code uses x < 0 strictly.)
            return torch.exp(pre) * (pre < 0).to(pre.dtype)
        if name == 'relu':
            return torch.zeros_like(pre)  # second deriv is 0 almost everywhere
        if name == 'sigmoid':
            s = torch.sigmoid(pre)
            return s * (1.0 - s) * (1.0 - 2.0 * s)
        return torch.zeros_like(pre)  # linear

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, x):
        h = x
        for layer in self.encoder_layers[:-1]:
            h = self._activation(layer(h))
        return self.encoder_layers[-1](h)

    def decode(self, z):
        h = z
        for layer in self.decoder_layers[:-1]:
            h = self._activation(layer(h))
        return self.decoder_layers[-1](h)

    # ------------------------------------------------------------------
    # Analytical derivatives via chain rule
    # ------------------------------------------------------------------

    def _network_derivative(self, input_, dx, layers):
        """Propagate first derivative dx/dt through a stack of linear layers
        interleaved with the configured activation.

        Returns dz/dt, where z = layers(input_) and input_ evolves with dx.
        Faithful to the TF1 implementation in z_derivative().

        Note on weight shapes: nn.Linear stores weight as (out, in), so the
        matmul x @ W in the original TF code corresponds to x @ weight.T here.
        """
        dz = dx
        if self.activation_name == 'linear':
            for layer in layers:
                dz = dz @ layer.weight.T
            return dz

        h = input_
        for layer in layers[:-1]:
            pre = layer(h)  # h @ W.T + b
            act_deriv = self._activation_deriv(pre)
            dz = act_deriv * (dz @ layer.weight.T)
            h = self._activation(pre)
        dz = dz @ layers[-1].weight.T
        return dz

    def _network_derivative_order2(self, input_, dx, ddx, layers):
        """Propagate first and second derivatives through a stack of layers.

        Returns (dz, ddz). Faithful to z_derivative_order2() in the original.
        """
        dz = dx
        ddz = ddx

        if self.activation_name == 'linear':
            for layer in layers:
                dz = dz @ layer.weight.T
                ddz = ddz @ layer.weight.T
            return dz, ddz

        h = input_
        for layer in layers[:-1]:
            pre = layer(h)
            dz_pre = dz @ layer.weight.T   # pre-activation first derivative
            ddz_pre = ddz @ layer.weight.T

            act1 = self._activation_deriv(pre)
            act2 = self._activation_deriv2(pre)

            # d/dt of activation(pre) = act'(pre) * dz_pre
            dz = act1 * dz_pre
            # d^2/dt^2 of activation(pre) = act''(pre) * (dz_pre)^2 + act'(pre) * ddz_pre
            ddz = act2 * dz_pre.pow(2) + act1 * ddz_pre

            h = self._activation(pre)

        # Final linear layer: derivatives pass through unchanged by weight.T
        dz = dz @ layers[-1].weight.T
        ddz = ddz @ layers[-1].weight.T
        return dz, ddz

    def z_derivative(self, x, dx):
        """Encoder-side first derivative: dz/dt given x(t) and dx/dt."""
        return self._network_derivative(x, dx, self.encoder_layers)

    def z_derivative_order2(self, x, dx, ddx):
        """Encoder-side first and second derivatives."""
        return self._network_derivative_order2(x, dx, ddx, self.encoder_layers)

    def x_derivative_decode(self, z, dz):
        """Decoder-side: predicted dx/dt from predicted dz/dt."""
        return self._network_derivative(z, dz, self.decoder_layers)

    def x_derivative_decode_order2(self, z, dz, ddz):
        """Decoder-side first and second derivatives."""
        return self._network_derivative_order2(z, dz, ddz, self.decoder_layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, dx, ddx=None):
        """
        Args:
            x: (batch, input_dim) state snapshots.
            dx: (batch, input_dim) first time-derivatives of x.
            ddx: (batch, input_dim) second time-derivatives — required iff
                 model_order == 2.

        Returns:
            Dict of tensors. Keys depend on model_order:
              Order 1: z, x_decode, dz, dz_predict, dx_decode, Theta
              Order 2: z, x_decode, dz, ddz, ddz_predict, dx_decode,
                       ddx_decode, Theta
        """
        z = self.encode(x)
        x_decode = self.decode(z)
        masked_coefs = self.coefficient_mask * self.sindy_coefficients

        if self.model_order == 1:
            dz = self.z_derivative(x, dx)
            Theta = sindy_library(z, self.poly_order, self.include_sine)
            dz_predict = Theta @ masked_coefs
            dx_decode = self.x_derivative_decode(z, dz_predict)
            return {
                'z': z,
                'x_decode': x_decode,
                'dz': dz,
                'dz_predict': dz_predict,
                'dx_decode': dx_decode,
                'Theta': Theta,
            }

        # Order 2
        if ddx is None:
            raise ValueError("ddx is required when model_order == 2")
        dz, ddz = self.z_derivative_order2(x, dx, ddx)
        Theta = sindy_library_order2(z, dz, self.poly_order, self.include_sine)
        ddz_predict = Theta @ masked_coefs
        dx_decode, ddx_decode = self.x_derivative_decode_order2(z, dz, ddz_predict)
        return {
            'z': z,
            'x_decode': x_decode,
            'dz': dz,
            'ddz': ddz,
            'ddz_predict': ddz_predict,
            'dx_decode': dx_decode,
            'ddx_decode': ddx_decode,
            'Theta': Theta,
        }

    # ------------------------------------------------------------------
    # Sequential thresholding (L0 proxy)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply_thresholding(self, threshold):
        """Zero out mask entries whose |coefficient| falls below threshold.

        Call periodically during training (e.g. every 500 epochs). Once a mask
        entry is zero, subsequent gradients through that coefficient are
        suppressed, giving a hard L0 sparsity pattern without requiring a
        separate optimizer.

        After thresholding, it is standard to continue training with the
        L1 regularization weight set to zero (the 'refinement' loss below).
        """
        keep = self.sindy_coefficients.abs() >= threshold
        self.coefficient_mask.mul_(keep.to(self.coefficient_mask.dtype))

    @torch.no_grad()
    def active_terms(self):
        """Return (library_dim, latent_dim) boolean tensor of surviving terms."""
        return self.coefficient_mask.bool()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sindy_loss(outputs, targets, model, loss_weights, model_order=1):
    """Compute SINDy-AE composite loss.

    Args:
        outputs: dict returned by SindyAutoencoder.forward.
        targets: dict with at least 'x' and 'dx'; also 'ddx' if model_order==2.
        model: the SindyAutoencoder instance (needed for masked coefficients).
        loss_weights: dict with keys 'decoder', 'sindy_z', 'sindy_x',
                      'sindy_regularization'.
        model_order: 1 or 2.

    Returns:
        (total_loss, component_losses_dict, refinement_total_loss)
        The refinement loss drops the L1 regularization term — use after
        sequential thresholding has fixed the sparsity pattern.
    """
    x = targets['x']
    dx = targets['dx']

    loss_decoder = (x - outputs['x_decode']).pow(2).mean()

    if model_order == 1:
        loss_sindy_z = (outputs['dz'] - outputs['dz_predict']).pow(2).mean()
        loss_sindy_x = (dx - outputs['dx_decode']).pow(2).mean()
    else:
        ddx = targets['ddx']
        loss_sindy_z = (outputs['ddz'] - outputs['ddz_predict']).pow(2).mean()
        loss_sindy_x = (ddx - outputs['ddx_decode']).pow(2).mean()

    masked_coefs = model.coefficient_mask * model.sindy_coefficients
    loss_reg = masked_coefs.abs().mean()

    total = (
        loss_weights['decoder'] * loss_decoder
        + loss_weights['sindy_z'] * loss_sindy_z
        + loss_weights['sindy_x'] * loss_sindy_x
        + loss_weights['sindy_regularization'] * loss_reg
    )

    total_refinement = (
        loss_weights['decoder'] * loss_decoder
        + loss_weights['sindy_z'] * loss_sindy_z
        + loss_weights['sindy_x'] * loss_sindy_x
    )

    losses = {
        'decoder': loss_decoder,
        'sindy_z': loss_sindy_z,
        'sindy_x': loss_sindy_x,
        'sindy_regularization': loss_reg,
    }
    return total, losses, total_refinement


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)

    # First-order smoke test (Lorenz-shaped)
    model = SindyAutoencoder(
        input_dim=128, latent_dim=3, widths=[64, 32],
        activation='sigmoid', poly_order=3, model_order=1,
    )
    x = torch.randn(16, 128)
    dx = torch.randn(16, 128)
    out = model(x, dx)
    total, parts, refinement = sindy_loss(
        out, {'x': x, 'dx': dx}, model,
        loss_weights={
            'decoder': 1.0, 'sindy_z': 1e-4,
            'sindy_x': 1e-4, 'sindy_regularization': 1e-5,
        },
        model_order=1,
    )
    total.backward()
    print(f"Order 1 OK. library_dim={model.library_dim}, loss={total.item():.4f}")
    print(f"  decoder={parts['decoder'].item():.4f}, "
          f"sindy_z={parts['sindy_z'].item():.4f}, "
          f"sindy_x={parts['sindy_x'].item():.4f}")

    # Second-order smoke test (pendulum-shaped)
    model2 = SindyAutoencoder(
        input_dim=51 * 51, latent_dim=1, widths=[128, 64],
        activation='sigmoid', poly_order=3, model_order=2,
    )
    x2 = torch.randn(8, 51 * 51)
    dx2 = torch.randn(8, 51 * 51)
    ddx2 = torch.randn(8, 51 * 51)
    out2 = model2(x2, dx2, ddx2)
    total2, _, _ = sindy_loss(
        out2, {'x': x2, 'dx': dx2, 'ddx': ddx2}, model2,
        loss_weights={
            'decoder': 1.0, 'sindy_z': 1e-4,
            'sindy_x': 1e-4, 'sindy_regularization': 1e-5,
        },
        model_order=2,
    )
    total2.backward()
    print(f"Order 2 OK. library_dim={model2.library_dim}, loss={total2.item():.4f}")

    # Thresholding smoke test
    before = model.coefficient_mask.sum().item()
    model.apply_thresholding(threshold=0.5)
    after = model.coefficient_mask.sum().item()
    print(f"Thresholding: {before:.0f} -> {after:.0f} active mask entries")

def compute_network_gradient_nd(model, x_data, target_idx, target_type='latent'):
    """Compute ∇f w.r.t. state via autograd. Works for any input dimension."""
    enc = model.ae.encoder
    chunk = 2000
    f_vals_list = []
    grad_list = []

    # Resolve the inner KAN encoder (handles PassthroughEncoder wrapper)
    kan_enc = getattr(enc, 'kan_encoder', enc)

    for i in range(0, len(x_data), chunk):
        xi = x_data[i:i+chunk].detach().clone().requires_grad_(True)
        if target_type == 'latent':
            fi = enc(xi)[:, target_idx]
        elif target_type == 'mixing':
            ui = kan_enc.kan1(xi)
            vi = kan_enc.kan2(ui)
            fi = vi[:, target_idx]
        elif target_type == 'relu_mixing':
            ui = kan_enc.kan1(xi)
            vi = kan_enc.kan2(ui)
            rvi = kan_enc.relu(vi)
            fi = rvi[:, target_idx]
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        fi_sum = fi.sum()
        fi_sum.backward()
        grad_list.append(xi.grad.detach().numpy())
        f_vals_list.append(fi.detach().numpy())

    return np.concatenate(grad_list, axis=0), np.concatenate(f_vals_list, axis=0)

def eval_h(g_vals, h_result):
    """Evaluate the recovered outer function h at new g values."""
    if h_result.get('used_affine', False):
        c = h_result['affine_coeffs']
        return c[0] * g_vals + c[1]
    g_shifted = g_vals - h_result['g_center']
    V = np.vander(g_shifted, N=len(h_result['h_coeffs']), increasing=True)
    return V @ h_result['h_coeffs']

def recover_inner_function(x, y, z, f_vals, max_degree=3,
                           alpha=None, sparsity_threshold=1e-4):
    """Recover sparse inner polynomial g(x,y,z) via Lasso.

    Uses the level-set formulation: fit g so that g(x,y,z) ≈ f(x,y,z)
    with L1 penalty to enforce sparsity.  When h ≈ linear (as in the
    Lorenz latent dims), g directly approximates f and the surviving
    monomials are the Koopman observables.

    Two-stage approach:
      1. OLS fit → compute residual scale
      2. Lasso with alpha scaled to residual (ensures cross-terms survive)

    Args:
        x, y, z : (N,) state variables
        f_vals  : (N,) target function values
        max_degree : max total polynomial degree
        alpha : Lasso regularization strength (None → auto-select)
        sparsity_threshold : coefficients below this are zeroed

    Returns:
        result : dict with keys:
            'coeffs'   : full coefficient vector
            'names'    : monomial names
            'powers'   : (p,q,r) tuples
            'active'   : indices of surviving terms
            'active_names' : names of surviving terms
            'active_coeffs': coefficients of surviving terms
            'r2'       : R² of the fit
            'alpha'    : Lasso alpha used
    """
    M, names, powers = build_polynomial_dict(x, y, z, max_degree)

    # Standardize columns AND target for balanced Lasso
    col_mean = np.mean(M, axis=0)
    col_std = np.std(M, axis=0)
    col_std[col_std < 1e-12] = 1.0
    M_scaled = (M - col_mean) / col_std

    f_mean = np.mean(f_vals)
    f_std = np.std(f_vals)
    if f_std < 1e-12:
        f_std = 1.0
    f_scaled = (f_vals - f_mean) / f_std

    if alpha is None:
        # Use a very small alpha to preserve weak-but-real cross-terms.
        # On normalized data, 1e-5 keeps terms contributing ~0.1% of variance.
        alpha_used = 1e-5
    else:
        alpha_used = alpha

    lasso = Lasso(alpha=alpha_used, max_iter=50000, fit_intercept=True)
    lasso.fit(M_scaled, f_scaled)

    # Unscale coefficients back to original space
    # y_scaled = M_scaled @ coef + intercept
    # (f - f_mean)/f_std = ((M - col_mean)/col_std) @ coef + intercept
    # f = f_std * (M/col_std @ coef - col_mean/col_std @ coef + intercept) + f_mean
    # f = Σ_i (f_std * coef_i / col_std_i) * M_i + [f_mean + f_std*(intercept - Σ coef_i*col_mean_i/col_std_i)]
    coeffs_orig = f_std * lasso.coef_ / col_std
    intercept_orig = f_mean + f_std * (lasso.intercept_ - np.sum(lasso.coef_ * col_mean / col_std))

    # Fold intercept into the constant term (powers (0,0,0))
    const_idx = powers.index((0, 0, 0))
    coeffs_orig[const_idx] += intercept_orig

    # Sparsify
    mask = np.abs(coeffs_orig) > sparsity_threshold
    coeffs_sparse = coeffs_orig.copy()
    coeffs_sparse[~mask] = 0.0

    # R²
    pred = M @ coeffs_sparse
    ss_res = np.sum((f_vals - pred) ** 2)
    ss_tot = np.sum((f_vals - f_vals.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    active = np.where(mask)[0]

    return {
        'coeffs': coeffs_sparse,
        'names': names,
        'powers': powers,
        'active': active,
        'active_names': [names[i] for i in active],
        'active_coeffs': coeffs_sparse[active],
        'r2': r2,
        'alpha': alpha_used,
    }



def recover_outer_function(grad_f, grad_g, g_vals, f_vals,
                           poly_degree=4, grad_norm_threshold=1e-2):
    """Recover h' and integrate to get h.

    h'(g(x,y,z)) = (∇f · ∇g) / ||∇g||²

    Strategy:
      1. Compute h' samples from gradient formula
      2. Fit h' as polynomial (binned-median for robustness)
      3. Integrate to get h
      4. If h∘g is worse than a simple affine h, fall back to affine

    Args:
        grad_f : (N, 3) gradient of network output
        grad_g : tuple (dg/dx, dg/dy, dg/dz), each (N,)
        g_vals : (N,) inner function values
        f_vals : (N,) network output values
        poly_degree : degree for h' polynomial fit
        grad_norm_threshold : filter points with small ||∇g||

    Returns:
        result dict
    """
    gx, gy, gz = grad_g

    # ∇f · ∇g
    dot = grad_f[:, 0] * gx + grad_f[:, 1] * gy + grad_f[:, 2] * gz

    # ||∇g||²
    norm_sq = gx**2 + gy**2 + gz**2

    # Filter small gradients (more aggressive percentile-based threshold)
    norm_thresh = max(grad_norm_threshold**2,
                      np.percentile(norm_sq, 5))
    valid = norm_sq > norm_thresh
    h_prime = dot[valid] / norm_sq[valid]
    g_valid = g_vals[valid]
    f_valid = f_vals[valid]

    # Filter outliers in h' (tighter: 3× IQR)
    h_prime_median = np.median(h_prime)
    h_prime_iqr = np.percentile(h_prime, 75) - np.percentile(h_prime, 25)
    if h_prime_iqr < 1e-10:
        h_prime_iqr = np.std(h_prime)
    inlier = np.abs(h_prime - h_prime_median) < 3 * max(h_prime_iqr, 0.1)
    h_prime = h_prime[inlier]
    g_valid = g_valid[inlier]
    f_valid = f_valid[inlier]

    # ----- Binned-median fit for robustness -----
    # Bin by g, take median h' per bin → smooth 1D curve, then fit polynomial
    n_bins = min(80, len(g_valid) // 20)
    g_center = np.median(g_valid)
    g_shifted = g_valid - g_center

    bin_edges = np.linspace(g_shifted.min(), g_shifted.max(), n_bins + 1)
    bin_centers, bin_medians = [], []
    for i in range(n_bins):
        mask_bin = (g_shifted >= bin_edges[i]) & (g_shifted < bin_edges[i+1])
        if mask_bin.sum() >= 5:
            bin_centers.append(0.5 * (bin_edges[i] + bin_edges[i+1]))
            bin_medians.append(np.median(h_prime[mask_bin]))
    bin_centers = np.array(bin_centers)
    bin_medians = np.array(bin_medians)

    # Fit h'(z) as polynomial on binned medians
    if len(bin_centers) > poly_degree + 1:
        V_bin = np.vander(bin_centers, N=poly_degree + 1, increasing=True)
        h_prime_coeffs, _, _, _ = np.linalg.lstsq(V_bin, bin_medians,
                                                    rcond=None)
    else:
        V = np.vander(g_shifted, N=poly_degree + 1, increasing=True)
        h_prime_coeffs, _, _, _ = np.linalg.lstsq(V, h_prime, rcond=None)

    # Integrate: h(z) = C + Σ b_k/(k+1) (z-g_center)^(k+1)
    h_coeffs_integrated = np.zeros(poly_degree + 2)
    for k in range(poly_degree + 1):
        h_coeffs_integrated[k + 1] = h_prime_coeffs[k] / (k + 1)

    # Set integration constant via median reference
    ref_idx = np.argmin(np.abs(g_shifted))
    g0 = g_valid[ref_idx]
    f0 = f_valid[ref_idx]

    # Evaluate h at all valid points
    V_int = np.vander(g_shifted, N=poly_degree + 2, increasing=True)
    h_reconstructed = V_int @ h_coeffs_integrated

    # Adjust constant so h(g0) = f0
    h_at_ref = np.vander(np.array([g0 - g_center]), N=poly_degree + 2,
                          increasing=True) @ h_coeffs_integrated
    C = f0 - h_at_ref[0]
    h_coeffs_integrated[0] = C
    h_reconstructed += C

    # ----- Affine fallback -----
    # If the polynomial h∘g is worse than a simple affine f ≈ a·g + b,
    # fall back to affine (this happens when g already ≈ f).
    A_aff = np.column_stack([g_valid, np.ones(len(g_valid))])
    c_aff, _, _, _ = np.linalg.lstsq(A_aff, f_valid, rcond=None)
    h_affine = c_aff[0] * g_valid + c_aff[1]

    ss_poly = np.sum((f_valid - h_reconstructed) ** 2)
    ss_affine = np.sum((f_valid - h_affine) ** 2)

    use_affine = ss_poly > ss_affine
    if use_affine:
        h_reconstructed = h_affine

    # Evaluate h' fit for plotting
    g_plot = np.linspace(g_valid.min(), g_valid.max(), 500)
    V_plot = np.vander(g_plot - g_center, N=poly_degree + 1, increasing=True)
    h_prime_fit = V_plot @ h_prime_coeffs

    # Evaluate h for plotting
    if use_affine:
        h_fit_plot = c_aff[0] * g_plot + c_aff[1]
    else:
        V_int_plot = np.vander(g_plot - g_center, N=poly_degree + 2,
                                increasing=True)
        h_fit_plot = V_int_plot @ h_coeffs_integrated

    # Full valid mask
    full_valid = np.zeros(len(grad_f), dtype=bool)
    valid_idx = np.where(valid)[0]
    full_valid[valid_idx[inlier]] = True

    return {
        'g_samples': g_valid,
        'h_prime_samples': h_prime,
        'h_prime_coeffs': h_prime_coeffs,
        'h_coeffs': h_coeffs_integrated,
        'g_center': g_center,
        'g0': g0,
        'f0': f0,
        'C': C,
        'h_reconstructed': h_reconstructed,
        'f_valid': f_valid,
        'g_plot': g_plot,
        'h_prime_fit': h_prime_fit,
        'h_fit_plot': h_fit_plot,
        'h_bin_centers': bin_centers,
        'h_bin_medians': bin_medians,
        'valid_mask': full_valid,
        'used_affine': use_affine,
        'affine_coeffs': c_aff,
    }

def full_decomposition(model, x_data, target_idx, target_type='latent',
                       max_degree=3, lasso_alpha=None, h_degree=4):
    """Run the complete level-set + gradient + ideal pipeline.

    Args:
        model : trained DeepKoopmanCT
        x_data : (N, 3) tensor of Lorenz states
        target_idx : which dimension to analyze
        target_type : 'latent' or 'mixing'
        max_degree : polynomial degree for inner function
        lasso_alpha : Lasso penalty (None → cross-validate)
        h_degree : polynomial degree for outer function fit

    Returns:
        dict with all results: inner, outer, ideal, reconstruction
    """
    x_np = x_data.numpy() if isinstance(x_data, torch.Tensor) else x_data
    x_, y_, z_ = x_np[:, 0], x_np[:, 1], x_np[:, 2]

    label = f"z[{target_idx}]" if target_type == 'latent' else f"v[{target_idx}]"

    # Step 1: Evaluate f and ∇f
    grad_f, f_vals = compute_network_gradient(
        model, x_data if isinstance(x_data, torch.Tensor)
        else torch.tensor(x_data, dtype=torch.float32),
        target_idx, target_type)

    # Step 2: Recover inner function via Lasso
    inner = recover_inner_function(x_, y_, z_, f_vals,
                                    max_degree=max_degree, alpha=lasso_alpha)

    # Step 3: Evaluate g and ∇g
    g_vals = eval_polynomial(x_, y_, z_, inner['coeffs'], inner['powers'])
    grad_g = grad_polynomial(x_, y_, z_, inner['coeffs'], inner['powers'])

    # Step 4: Recover outer function
    outer = recover_outer_function(grad_f, grad_g, g_vals, f_vals,
                                    poly_degree=h_degree)

    # Step 5: Ideal analysis
    #ideal = analyze_ideal(inner)

    # Step 6: Full reconstruction f̂ = h(g(x,y,z))
    f_hat = eval_h(g_vals, outer)
    recon_mse = np.mean((f_vals - f_hat) ** 2)
    recon_r2 = 1.0 - np.sum((f_vals - f_hat) ** 2) / \
               max(np.sum((f_vals - f_vals.mean()) ** 2), 1e-12)

    # Also: direct polynomial reconstruction (g alone, no h)
    g_recon_r2 = inner['r2']

    return {
        'label': label,
        'target_idx': target_idx,
        'target_type': target_type,
        'f_vals': f_vals,
        'x_np': x_np,
        'inner': inner,
        'g_vals': g_vals,
        'grad_f': grad_f,
        'grad_g': grad_g,
        'outer': outer,
        #'ideal': ideal,
        'f_hat': f_hat,
        'recon_mse': recon_mse,
        'recon_r2': recon_r2,
        'g_recon_r2': g_recon_r2,
    }
# =====================================================================
# 3. Neural network gradient via autograd
# =====================================================================

def compute_network_gradient(model, x_data, target_idx, target_type='latent'):
    """Compute ∇f w.r.t. (x, y, z) via autograd.

    Args:
        model : DeepKoopmanCT
        x_data : (N, 3) tensor
        target_idx : which output dimension
        target_type : 'latent' (z_k), 'mixing' (v_k), or 'relu_mixing'

    Returns:
        grad_f : (N, 3) numpy array  [df/dx, df/dy, df/dz]
        f_vals : (N,) numpy array
    """
    enc = model.ae.encoder
    has_dense = hasattr(enc, 'dense') and hasattr(enc, 'relu')

    def _forward_target(enc, xi, target_idx, target_type):
        """Compute scalar target from encoder, handling both Dense and Pure KAN."""
        if target_type == 'latent':
            if has_dense:
                u = enc.kan1(xi)
                v = enc.kan2(u)
                rv = enc.relu(v)
                out = enc.dense(rv)
            else:
                out = enc(xi)
            return out[:, target_idx]
        elif target_type == 'mixing':
            u = enc.kan1(xi)
            v = enc.kan2(u)
            return v[:, target_idx]
        elif target_type == 'relu_mixing':
            u = enc.kan1(xi)
            v = enc.kan2(u)
            rv = enc.relu(v)
            return rv[:, target_idx]
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    # Compute gradient via backward in chunks for memory
    chunk = 2000
    f_vals_list = []
    grad_list = []

    for i in range(0, len(x_data), chunk):
        xi = x_data[i:i+chunk].detach().clone().requires_grad_(True)
        fi = _forward_target(enc, xi, target_idx, target_type)

        fi_sum = fi.sum()
        fi_sum.backward()
        grad_list.append(xi.grad.detach().numpy())
        f_vals_list.append(fi.detach().numpy())

    return np.concatenate(grad_list, axis=0), np.concatenate(f_vals_list, axis=0)


# =====================================================================
# 1. Polynomial dictionary
# =====================================================================

def build_polynomial_dict(x, y, z, max_degree=3, include_trig=False):
    """Build a polynomial feature dictionary for 3 variables.

    Returns:
        M : (N, n_terms) design matrix
        names : list of str, monomial labels
        powers : list of (p, q, r) tuples
    """
    N = len(x)
    terms, names, powers = [], [], []

    for total_deg in range(max_degree + 1):
        for p in range(total_deg + 1):
            for q in range(total_deg - p + 1):
                r = total_deg - p - q
                term = (x ** p) * (y ** q) * (z ** r)
                terms.append(term)

                # Build readable name
                parts = []
                if p == 1: parts.append("x")
                elif p > 1: parts.append(f"x^{p}")
                if q == 1: parts.append("y")
                elif q > 1: parts.append(f"y^{q}")
                if r == 1: parts.append("z")
                elif r > 1: parts.append(f"z^{r}")
                name = "".join(parts) if parts else "1"
                names.append(name)
                powers.append((p, q, r))

    M = np.column_stack(terms)
    return M, names, powers


def eval_polynomial(x, y, z, coeffs, powers):
    """Evaluate sparse polynomial g(x,y,z) = Σ c_i x^p y^q z^r."""
    result = np.zeros_like(x, dtype=float)
    for c, (p, q, r) in zip(coeffs, powers):
        if abs(c) > 1e-15:
            result += c * (x ** p) * (y ** q) * (z ** r)
    return result


def grad_polynomial(x, y, z, coeffs, powers):
    """Compute ∇g analytically.  Returns (dg/dx, dg/dy, dg/dz)."""
    gx = np.zeros_like(x, dtype=float)
    gy = np.zeros_like(x, dtype=float)
    gz = np.zeros_like(x, dtype=float)
    for c, (p, q, r) in zip(coeffs, powers):
        if abs(c) < 1e-15:
            continue
        if p > 0:
            gx += c * p * (x ** max(p - 1, 0)) * (y ** q) * (z ** r)
        if q > 0:
            gy += c * (x ** p) * q * (y ** max(q - 1, 0)) * (z ** r)
        if r > 0:
            gz += c * (x ** p) * (y ** q) * r * (z ** max(r - 1, 0))
    return gx, gy, gz


# =====================================================================
# 1b. N-dimensional polynomial dictionary (generalizes 3D versions)
# =====================================================================

def build_polynomial_dict_nd(data, var_names, max_degree=3):
    """Build a polynomial feature dictionary for N variables.

    Args:
        data : (N, d) numpy array of state variables
        var_names : list of d variable name strings
        max_degree : max total polynomial degree

    Returns:
        M : (N, n_terms) design matrix
        names : list of str, monomial labels
        powers : list of d-tuples
    """
    from itertools import product as iproduct
    N, d = data.shape
    terms, names, powers_list = [], [], []

    # Enumerate all d-tuples (p1,...,pd) with sum <= max_degree
    def _enum_powers(d, max_deg):
        if d == 1:
            for p in range(max_deg + 1):
                yield (p,)
        else:
            for p in range(max_deg + 1):
                for rest in _enum_powers(d - 1, max_deg - p):
                    yield (p,) + rest

    for pw in _enum_powers(d, max_degree):
        term = np.ones(N, dtype=float)
        parts = []
        for dim, p in enumerate(pw):
            if p > 0:
                term *= data[:, dim] ** p
            if p == 1:
                parts.append(var_names[dim])
            elif p > 1:
                parts.append(f"{var_names[dim]}^{p}")
        name = "".join(parts) if parts else "1"
        terms.append(term)
        names.append(name)
        powers_list.append(pw)

    M = np.column_stack(terms)
    return M, names, powers_list


def eval_polynomial_nd(data, coeffs, powers):
    """Evaluate sparse polynomial g(x1,...,xd) = Σ c_i Π x_j^{p_j}."""
    N = data.shape[0]
    result = np.zeros(N, dtype=float)
    for c, pw in zip(coeffs, powers):
        if abs(c) > 1e-15:
            term = np.ones(N, dtype=float)
            for dim, p in enumerate(pw):
                if p > 0:
                    term *= data[:, dim] ** p
            result += c * term
    return result


def grad_polynomial_nd(data, coeffs, powers):
    """Compute ∇g analytically for d-dimensional polynomial.

    Returns:
        grads : (N, d) numpy array
    """
    N, d = data.shape
    grads = np.zeros((N, d), dtype=float)
    for c, pw in zip(coeffs, powers):
        if abs(c) < 1e-15:
            continue
        for dim in range(d):
            if pw[dim] > 0:
                dterm = np.ones(N, dtype=float) * c * pw[dim]
                for j in range(d):
                    if j == dim:
                        dterm *= data[:, j] ** max(pw[j] - 1, 0)
                    else:
                        if pw[j] > 0:
                            dterm *= data[:, j] ** pw[j]
                grads[:, dim] += dterm
    return grads


def recover_inner_function_nd(data, var_names, f_vals, max_degree=3,
                               alpha=None, sparsity_threshold=1e-4):
    """Recover sparse polynomial inner function for d-dimensional state.

    Same algorithm as recover_inner_function but generalized to d variables.
    """
    M, names, powers = build_polynomial_dict_nd(data, var_names, max_degree)

    col_mean = np.mean(M, axis=0)
    col_std = np.std(M, axis=0)
    col_std[col_std < 1e-12] = 1.0
    M_scaled = (M - col_mean) / col_std

    f_mean = np.mean(f_vals)
    f_std = np.std(f_vals)
    if f_std < 1e-12:
        f_std = 1.0
    f_scaled = (f_vals - f_mean) / f_std

    alpha_used = alpha if alpha is not None else 1e-5

    lasso = Lasso(alpha=alpha_used, max_iter=50000, fit_intercept=True)
    lasso.fit(M_scaled, f_scaled)

    coeffs_orig = f_std * lasso.coef_ / col_std
    intercept_orig = f_mean + f_std * (lasso.intercept_ - np.sum(lasso.coef_ * col_mean / col_std))

    # Fold intercept into constant term (all-zeros power tuple)
    zero_pw = tuple(0 for _ in range(data.shape[1]))
    const_idx = powers.index(zero_pw)
    coeffs_orig[const_idx] += intercept_orig

    mask = np.abs(coeffs_orig) > sparsity_threshold
    coeffs_sparse = coeffs_orig.copy()
    coeffs_sparse[~mask] = 0.0

    pred = M @ coeffs_sparse
    ss_res = np.sum((f_vals - pred) ** 2)
    ss_tot = np.sum((f_vals - f_vals.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    active = np.where(mask)[0]

    return {
        'coeffs': coeffs_sparse,
        'names': names,
        'powers': powers,
        'active': active,
        'active_names': [names[i] for i in active],
        'active_coeffs': coeffs_sparse[active],
        'r2': r2,
        'alpha': alpha_used,
    }


def recover_outer_function_from_raw(dot, norm_sq, g_vals, f_vals,
                                    poly_degree=4, grad_norm_threshold=1e-2):
    """Recover outer function from pre-computed ∇f·∇g and ||∇g||².

    Same algorithm as recover_outer_function but accepts raw dot products
    and squared norms directly (works for any dimension).
    """
    norm_thresh = max(grad_norm_threshold**2,
                      np.percentile(norm_sq, 5))
    valid = norm_sq > norm_thresh
    h_prime = dot[valid] / norm_sq[valid]
    g_valid = g_vals[valid]
    f_valid = f_vals[valid]

    h_prime_median = np.median(h_prime)
    h_prime_iqr = np.percentile(h_prime, 75) - np.percentile(h_prime, 25)
    if h_prime_iqr < 1e-10:
        h_prime_iqr = np.std(h_prime)
    inlier = np.abs(h_prime - h_prime_median) < 3 * max(h_prime_iqr, 0.1)
    h_prime = h_prime[inlier]
    g_valid = g_valid[inlier]
    f_valid = f_valid[inlier]

    n_bins = min(80, len(g_valid) // 20)
    g_center = np.median(g_valid)
    g_shifted = g_valid - g_center

    bin_edges = np.linspace(g_shifted.min(), g_shifted.max(), n_bins + 1)
    bin_centers, bin_medians = [], []
    for i in range(n_bins):
        mask_bin = (g_shifted >= bin_edges[i]) & (g_shifted < bin_edges[i+1])
        if mask_bin.sum() >= 5:
            bin_centers.append(0.5 * (bin_edges[i] + bin_edges[i+1]))
            bin_medians.append(np.median(h_prime[mask_bin]))
    bin_centers = np.array(bin_centers)
    bin_medians = np.array(bin_medians)

    if len(bin_centers) > poly_degree + 1:
        V_bin = np.vander(bin_centers, N=poly_degree + 1, increasing=True)
        h_prime_coeffs, _, _, _ = np.linalg.lstsq(V_bin, bin_medians, rcond=None)
    else:
        V = np.vander(g_shifted, N=poly_degree + 1, increasing=True)
        h_prime_coeffs, _, _, _ = np.linalg.lstsq(V, h_prime, rcond=None)

    h_coeffs_integrated = np.zeros(poly_degree + 2)
    for k in range(poly_degree + 1):
        h_coeffs_integrated[k + 1] = h_prime_coeffs[k] / (k + 1)

    ref_idx = np.argmin(np.abs(g_shifted))
    g0 = g_valid[ref_idx]
    f0 = f_valid[ref_idx]

    V_int = np.vander(g_shifted, N=poly_degree + 2, increasing=True)
    h_reconstructed = V_int @ h_coeffs_integrated

    h_at_ref = np.vander(np.array([g0 - g_center]), N=poly_degree + 2,
                          increasing=True) @ h_coeffs_integrated
    C = f0 - h_at_ref[0]
    h_coeffs_integrated[0] = C
    h_reconstructed += C

    A_aff = np.column_stack([g_valid, np.ones(len(g_valid))])
    c_aff, _, _, _ = np.linalg.lstsq(A_aff, f_valid, rcond=None)
    h_affine = c_aff[0] * g_valid + c_aff[1]

    ss_poly = np.sum((f_valid - h_reconstructed) ** 2)
    ss_affine = np.sum((f_valid - h_affine) ** 2)

    use_affine = ss_poly > ss_affine
    if use_affine:
        h_reconstructed = h_affine

    g_plot = np.linspace(g_valid.min(), g_valid.max(), 500)
    V_plot = np.vander(g_plot - g_center, N=poly_degree + 1, increasing=True)
    h_prime_fit = V_plot @ h_prime_coeffs

    if use_affine:
        h_fit_plot = c_aff[0] * g_plot + c_aff[1]
    else:
        V_int_plot = np.vander(g_plot - g_center, N=poly_degree + 2, increasing=True)
        h_fit_plot = V_int_plot @ h_coeffs_integrated

    full_valid = np.zeros(len(f_vals), dtype=bool)
    valid_idx = np.where(valid)[0]
    full_valid[valid_idx[inlier]] = True

    return {
        'g_samples': g_valid,
        'h_prime_samples': h_prime,
        'h_prime_coeffs': h_prime_coeffs,
        'h_coeffs': h_coeffs_integrated,
        'g_center': g_center,
        'g0': g0, 'f0': f0, 'C': C,
        'h_reconstructed': h_reconstructed,
        'f_valid': f_valid,
        'g_plot': g_plot,
        'h_prime_fit': h_prime_fit,
        'h_fit_plot': h_fit_plot,
        'h_bin_centers': bin_centers,
        'h_bin_medians': bin_medians,
        'valid_mask': full_valid,
        'used_affine': use_affine,
        'affine_coeffs': c_aff,
    }


def full_decomposition_nd(model, x_data, target_idx, var_names,
                           target_type='latent', max_degree=3,
                           lasso_alpha=None, h_degree=4):
    """N-dimensional version of full_decomposition.

    Args:
        model : trained DeepKoopmanCT
        x_data : (N, d) tensor or numpy array
        target_idx : which dimension to analyze
        var_names : list of d variable name strings
        target_type : 'latent' or 'mixing'
        max_degree, lasso_alpha, h_degree : analysis parameters

    Returns:
        dict with inner, outer, ideal, reconstruction results
    """
    x_np = x_data.numpy() if isinstance(x_data, torch.Tensor) else x_data
    d = x_np.shape[1]

    label = f"z[{target_idx}]" if target_type == 'latent' else f"v[{target_idx}]"

    # Step 1: Evaluate f and ∇f via autograd
    x_t = x_data if isinstance(x_data, torch.Tensor) else torch.tensor(x_data, dtype=torch.float32)
    grad_f, f_vals = compute_network_gradient_nd(model, x_t, target_idx, target_type)

    # Step 2: Recover inner function via Lasso
    inner = recover_inner_function_nd(x_np, var_names, f_vals,
                                       max_degree=max_degree, alpha=lasso_alpha)

    # Step 3: Evaluate g and ∇g
    g_vals = eval_polynomial_nd(x_np, inner['coeffs'], inner['powers'])
    grad_g_mat = grad_polynomial_nd(x_np, inner['coeffs'], inner['powers'])

    # Step 4: Recover outer function (uses dot product / norm² formula)
    dot = np.sum(grad_f * grad_g_mat, axis=1)
    norm_sq = np.sum(grad_g_mat**2, axis=1)
    outer = recover_outer_function_from_raw(dot, norm_sq, g_vals, f_vals,
                                             poly_degree=h_degree)

    # # Step 5: Ideal analysis
    # ideal = analyze_ideal(inner)

    # Step 6: Full reconstruction
    f_hat = eval_h(g_vals, outer)
    recon_mse = np.mean((f_vals - f_hat) ** 2)
    recon_r2 = 1.0 - np.sum((f_vals - f_hat) ** 2) / \
               max(np.sum((f_vals - f_vals.mean()) ** 2), 1e-12)

    return {
        'label': label,
        'target_idx': target_idx,
        'target_type': target_type,
        'f_vals': f_vals,
        'x_np': x_np,
        'inner': inner,
        'g_vals': g_vals,
        'grad_f': grad_f,
        'outer': outer,
        #'ideal': ideal,
        'f_hat': f_hat,
        'recon_mse': recon_mse,
        'recon_r2': recon_r2,
        'g_recon_r2': inner['r2'],
    }


# =====================================================================
# Stable Deep Koopman model (spectral radius <= 1)
# =====================================================================

class DeepKoopmanCTStable(nn.Module):
    """Continuous-time Deep Koopman with stable generator.

    A = K - L^T L  where:
        K = -K^T              (skew-symmetric, drives rotation/oscillation)
        L^T L >= 0            (PSD, drives damping/contraction)

    => Re(eigenvalues of A) <= 0  => |eigenvalues of expm(A·dt)| <= 1

    When L -> 0, recovers the unitary (area-preserving) case.
    Params: n(n-1)/2 (skew) + n(n+1)/2 (lower-tri L) = n^2.
    """

    def __init__(self, autoencoder, latent_dim):
        super().__init__()
        self.ae = autoencoder
        self.latent_dim = latent_dim
        n = latent_dim

        # Skew-symmetric part K (rotation): upper triangle
        n_skew = n * (n - 1) // 2
        self.skew_params = nn.Parameter(0.01 * torch.randn(n_skew))

        # Lower triangular L (damping): L^T L is PSD
        self.L_diag = nn.Parameter(0.01 * torch.randn(n))
        n_lower = n * (n - 1) // 2
        self.L_lower = nn.Parameter(0.01 * torch.randn(n_lower))

    def _get_skew(self):
        n = self.latent_dim
        K = torch.zeros(n, n, device=self.skew_params.device)
        idx = torch.triu_indices(n, n, offset=1)
        K[idx[0], idx[1]] = self.skew_params
        return K - K.T

    def _get_L(self):
        n = self.latent_dim
        L = torch.zeros(n, n, device=self.L_diag.device)
        L[range(n), range(n)] = self.L_diag
        idx = torch.tril_indices(n, n, offset=-1)
        L[idx[0], idx[1]] = self.L_lower
        return L

    def generator_matrix(self):
        K = self._get_skew()
        L = self._get_L()
        return K - L.T @ L

    def damping_matrix(self):
        """L^T L — the PSD damping component."""
        L = self._get_L()
        return (L.T @ L).detach()

    def expm_propagate(self, z_t, dt):
        A = self.generator_matrix()
        M = torch.matrix_exp(A * dt)
        return z_t @ M.T

    def propagator_matrix(self, dt):
        A = self.generator_matrix()
        return torch.matrix_exp(A * dt).detach()

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTDIR = Path("results/lorenz_external_ablation")
OUTDIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
N_TRAJ = 60
N_TRAIN = 48        # 80% of 60
LATENT = 5
MID = 5
SEEDS = [0, 1, 2, 3, 4]
BURN_IN = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAMBDA_MAX = 0.906
TAU_L = 1.0 / LAMBDA_MAX
STEPS_PER_TAU = int(TAU_L / DT)
N_ROLLOUT_ICS = 10

TRUE_DICT = ["x", "y", "z", "xy", "xz"]
TRUE_SET = set(TRUE_DICT)
COEFF_THRESHOLD = 1e-2  # monomial recovery threshold after standardization

# Deep-Koopman-KANDy training (KANLinear backend with spline L1+entropy regularization)
DK_KANDY_EPOCHS = 200
DK_KANDY_SPLINE_REG = 0.001  # L1 + entropy on spline weights (optimal from multi-seed sweep)
DK_KANDY_LASSO_ALPHA = 1e-5  # level-set Lasso alpha
KAN_KWARGS = dict(grid_size=5, spline_order=3,
                  grid_range_l1=[-30, 50], grid_range_l2=[-10, 10])

# MLP + Stable training (matches Table 1 variant C)
MLP_EPOCHS = 200

# SINDy-AE training (Champion et al. published protocol)
SINDY_PRE_EPOCHS = 5001
SINDY_REFINE_EPOCHS = 5001
SINDY_THRESHOLD = 0.1
SINDY_THRESHOLD_EVERY = 500
SINDY_LR = 1e-3
SINDY_WIDTHS = [64, 32]
SINDY_LOSS_WEIGHTS = {
    'decoder': 1.0,
    'sindy_z': 1e-4,
    'sindy_x': 1e-4,
    'sindy_regularization': 1e-5,
}

# PySR config (identical for methods 2 and 3)
PYSR_OPERATORS = ["+", "-", "*"]
PYSR_MAX_COMPLEXITY = 20
PYSR_POPULATIONS = 40
PYSR_ITERATIONS = 200
PYSR_SEEDS_PER_LATENT = 5
PYSR_R2_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_data(data_seed):
    """Generate Lorenz data deterministically from data_seed.
    Returns traj (N_TRAJ, 6000, 3) and xdot (N_TRAJ, 6000, 3)."""
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)

    traj = simulate_lorenz(
        n_traj=N_TRAJ, n_steps=6000, dt=DT,
        burn_in=BURN_IN, device="cpu",
    )

    # Analytical derivatives at every snapshot
    xdot = lorenz_rhs(traj)

    # Sanity checks
    assert traj.shape == (N_TRAJ, 6000, 3), f"Bad traj shape: {traj.shape}"
    assert xdot.shape == (N_TRAJ, 6000, 3), f"Bad xdot shape: {xdot.shape}"
    assert not torch.isnan(traj).any(), "NaN in traj"
    assert not torch.isnan(xdot).any(), "NaN in xdot"

    # Assert ICs within bounds
    ics = traj[:, 0, :]  # after burn-in these are on the attractor
    # (post-burn-in ICs won't be in the init_box, but should be finite)
    assert torch.isfinite(ics).all(), "Non-finite ICs"

    # Verify analytical derivatives against a time-centered finite difference.
    # A forward difference approximates the derivative at the interval midpoint,
    # not exactly at the left endpoint, so compare against the trapezoidal
    # average of the endpoint derivatives.
    fd_xdot = (traj[:, 1:, :] - traj[:, :-1, :]) / DT
    ana_xdot_mid = 0.5 * (xdot[:, :-1, :] + xdot[:, 1:, :])
    discrepancy = (fd_xdot - ana_xdot_mid).abs().mean().item()
    log.info(
        "  Derivative check (centered): mean |FD - avg analytical| = "
        f"{discrepancy:.6f}"
    )
    assert discrepancy < 0.2, f"Derivative discrepancy too large: {discrepancy}"

    return traj, xdot


def split_data(traj, xdot):
    """80/20 train/test split by trajectory."""
    train_traj = traj[:N_TRAIN].contiguous()
    test_traj = traj[N_TRAIN:].contiguous()
    train_xdot = xdot[:N_TRAIN].contiguous()
    test_xdot = xdot[N_TRAIN:].contiguous()
    return train_traj, test_traj, train_xdot, test_xdot


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION METRICS (shared across all methods)
# ═══════════════════════════════════════════════════════════════════════════

def compute_nrmse(pred, true):
    std = true.reshape(-1, 3).std(axis=0)
    std = np.maximum(std, 1e-8)
    return (np.sqrt((pred - true)**2) / std[None, :]).mean(axis=1)


def eval_prediction_koopman(model, test_traj):
    """One-step MSE and NRMSE@2tau for a DeepKoopmanCT-like model."""
    ds = PairDataset(test_traj)
    dl = DataLoader(ds, batch_size=4096)
    total_se, n = 0.0, 0
    with torch.no_grad():
        for x_t, x_tp1 in dl:
            out = model.forward_one_step(x_t, DT)
            total_se += (out["x_tp1_pred"] - x_tp1).pow(2).sum().item()
            n += x_t.shape[0] * x_t.shape[1]
    one_step_mse = total_se / n

    steps_2tau = 2 * STEPS_PER_TAU
    max_steps = min(steps_2tau, test_traj.shape[1] - 1)
    ics = np.random.choice(test_traj.shape[0], min(N_ROLLOUT_ICS, test_traj.shape[0]), replace=False)
    rollout_errors = []
    with torch.no_grad():
        for ic in ics:
            true = test_traj[ic, :max_steps+1].numpy()
            x = test_traj[ic, 0]
            xs = [x.numpy()]
            for _ in range(max_steps):
                out = model.forward_one_step(x.unsqueeze(0), DT)
                x = out["x_tp1_pred"].squeeze(0)
                xs.append(x.numpy())
            pred = np.array(xs)
            nrmse = compute_nrmse(pred, true)
            rollout_errors.append(float(nrmse[min(steps_2tau, len(nrmse)-1)]))
    nrmse_2tau = float(np.mean(rollout_errors))

    return one_step_mse, nrmse_2tau


def eval_prediction_sindy(model, test_traj, test_xdot):
    """One-step MSE for SINDy-AE: x_{t+1} = x_t + dt * dx_decode."""
    x_flat = test_traj[:, :-1, :].reshape(-1, 3)
    x_tp1_flat = test_traj[:, 1:, :].reshape(-1, 3)
    dx_flat = test_xdot[:, :-1, :].reshape(-1, 3)

    model.eval()
    with torch.no_grad():
        out = model(x_flat, dx_flat)
        x_pred = x_flat + DT * out['dx_decode']
    one_step_mse = float((x_pred - x_tp1_flat).pow(2).mean().item())

    # Rollout via SINDy dynamics
    steps_2tau = 2 * STEPS_PER_TAU
    max_steps = min(steps_2tau, test_traj.shape[1] - 1)
    ics = np.random.choice(test_traj.shape[0], min(N_ROLLOUT_ICS, test_traj.shape[0]), replace=False)
    rollout_errors = []
    with torch.no_grad():
        for ic in ics:
            true = test_traj[ic, :max_steps+1].numpy()
            x = test_traj[ic, 0].clone()
            xs = [x.numpy().copy()]
            for _ in range(max_steps):
                dx = lorenz_rhs(x.unsqueeze(0)).squeeze(0)
                out = model(x.unsqueeze(0), dx.unsqueeze(0))
                x = x + DT * out['dx_decode'].squeeze(0)
                xs.append(x.numpy().copy())
            pred = np.array(xs)
            nrmse = compute_nrmse(pred, true)
            rollout_errors.append(float(nrmse[min(steps_2tau, len(nrmse)-1)]))
    nrmse_2tau = float(np.mean(rollout_errors))

    return one_step_mse, nrmse_2tau


# ═══════════════════════════════════════════════════════════════════════════
# SCORING PIPELINE (shared)
# ═══════════════════════════════════════════════════════════════════════════

def score_dictionary(detected_monomials):
    """Compute Jaccard, precision, recall against TRUE_SET."""
    detected = set(detected_monomials)
    tp = len(detected & TRUE_SET)
    precision = tp / max(len(detected), 1)
    recall = tp / len(TRUE_SET)
    union = len(detected | TRUE_SET)
    jaccard = tp / max(union, 1)
    return {
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "detected": sorted(detected),
        "n_detected": len(detected),
        "n_tp": tp,
        "n_fp": len(detected - TRUE_SET),
    }


def extract_monomials_from_sympy(expr, var_names=("x0", "x1", "x2"),
                                  canonical_map=None):
    """Extract monomials from a SymPy expression with |coeff| > COEFF_THRESHOLD.

    Returns a set of canonical monomial names.
    """
    import sympy as sp

    if canonical_map is None:
        canonical_map = {
            "x0": "x", "x1": "y", "x2": "z",
            "x0**2": "x^2", "x0*x1": "xy", "x0*x2": "xz",
            "x1**2": "y^2", "x1*x2": "yz", "x2**2": "z^2",
            "x1*x0": "xy", "x2*x0": "xz", "x2*x1": "yz",
        }

    if expr is None:
        return set()

    syms = [sp.Symbol(v) for v in var_names]
    expanded = sp.expand(expr)

    monomials = set()
    if expanded.is_Add:
        terms = expanded.as_ordered_terms()
    else:
        terms = [expanded]

    for term in terms:
        coeff, mono = term.as_coeff_Mul()
        if abs(float(coeff)) < COEFF_THRESHOLD:
            continue
        # Convert monomial to string key
        mono_str = str(mono)
        if mono_str == "1" or mono == sp.S.One:
            continue
        # Try canonical mapping
        canonical = canonical_map.get(mono_str, mono_str)
        if canonical in {"x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2",
                          "x^3", "x^2y", "x^2z", "xy^2", "xyz", "xz^2",
                          "y^3", "y^2z", "yz^2", "z^3"}:
            monomials.add(canonical)

    return monomials


# ═══════════════════════════════════════════════════════════════════════════
# PySR WRAPPER (used by methods 2 and 3)
# ═══════════════════════════════════════════════════════════════════════════

def run_pysr_on_encoder(encoder_fn, x_train, x_test, latent_dim,
                         method_name="", seed=0):
    """Run PySR on each latent coordinate of a frozen encoder.

    Args:
        encoder_fn: callable, x (N,3) -> z (N, latent_dim)
        x_train: (N_train, 3) tensor
        x_test: (N_test, 3) tensor
        latent_dim: int
        method_name: str for logging
        seed: base seed for PySR

    Returns:
        detected_monomials: set of canonical names
        pysr_log: list of dicts (one per latent dim) for jsonl
    """
    from pysr import PySRRegressor

    with torch.no_grad():
        z_train = encoder_fn(x_train).numpy()
        z_test = encoder_fn(x_test).numpy()
    x_train_np = x_train.numpy()
    x_test_np = x_test.numpy()

    detected = set()
    pysr_log = []

    for k in range(latent_dim):
        log.info(f"    PySR on z[{k}] ({method_name})")
        target_train = z_train[:, k]
        target_test = z_test[:, k]

        # Standardize target
        t_mean, t_std = target_train.mean(), target_train.std()
        if t_std < 1e-12:
            t_std = 1.0
        target_train_s = (target_train - t_mean) / t_std
        target_test_s = (target_test - t_mean) / t_std

        best_expr = None
        best_loss = float("inf")
        best_r2 = -999
        best_complexity = 999
        all_results = []

        for ps in range(PYSR_SEEDS_PER_LATENT):
            pysr_seed = seed * 1000 + k * 100 + ps

            model = PySRRegressor(
                binary_operators=PYSR_OPERATORS,
                unary_operators=[],
                maxsize=PYSR_MAX_COMPLEXITY,
                populations=PYSR_POPULATIONS,
                niterations=PYSR_ITERATIONS,
                random_state=pysr_seed,
                deterministic=True,
                parallelism="serial",
                progress=False,
                verbosity=0,
                temp_equation_file=True,
            )

            model.fit(x_train_np, target_train_s,
                      variable_names=["x0", "x1", "x2"])

            # Evaluate all equations on test set
            eqs = model.equations_
            if eqs is not None and len(eqs) > 0:
                for _, row in eqs.iterrows():
                    try:
                        pred_test = model.predict(x_test_np, index=int(row.name))
                        ss_res = np.sum((target_test_s - pred_test)**2)
                        ss_tot = np.sum((target_test_s - target_test_s.mean())**2)
                        r2 = 1 - ss_res / max(ss_tot, 1e-12)
                        complexity = int(row['complexity'])
                        loss = float(row['loss'])

                        all_results.append({
                            "pysr_seed": ps,
                            "complexity": complexity,
                            "loss": loss,
                            "r2_test": r2,
                            "equation": str(row['equation']),
                            "sympy": str(row.get('sympy_format', row['equation'])),
                        })

                        # Selection: lowest complexity with R² >= 0.95
                        if r2 >= PYSR_R2_THRESHOLD:
                            if (complexity < best_complexity or
                                (complexity == best_complexity and loss < best_loss)):
                                best_expr = row.get('sympy_format', None)
                                best_loss = loss
                                best_r2 = r2
                                best_complexity = complexity
                        elif best_r2 < PYSR_R2_THRESHOLD:
                            # Fallback: best loss at any complexity
                            if loss < best_loss:
                                best_expr = row.get('sympy_format', None)
                                best_loss = loss
                                best_r2 = r2
                                best_complexity = complexity
                    except Exception:
                        pass

        # Extract monomials from best expression
        k_monomials = extract_monomials_from_sympy(best_expr)
        detected |= k_monomials

        pysr_log.append({
            "latent_dim": k,
            "best_expr": str(best_expr),
            "best_r2": best_r2,
            "best_complexity": best_complexity,
            "monomials": sorted(k_monomials),
            "all_results": all_results,
        })

        log.info(f"      best: {best_expr} (R²={best_r2:.3f}, "
                 f"complexity={best_complexity}, monomials={sorted(k_monomials)})")

    return detected, pysr_log


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 1: Deep-Koopman-KANDy
# ═══════════════════════════════════════════════════════════════════════════

def _spline_regularization(ae):
    """L1 + entropy regularization on KAN spline weights."""
    reg = 0.0
    for module in ae.modules():
        if isinstance(module, KANLinear):
            l1_per_edge = module.spline_weight.abs().mean(-1)
            l1_total = l1_per_edge.sum()
            p = l1_per_edge / (l1_total + 1e-10)
            entropy = -torch.sum(p * (p + 1e-10).log())
            reg = reg + l1_total + entropy
    return reg


def run_deep_koopman_kandy(train_traj, test_traj, model_seed):
    """Train KANLinear [3,5,5] + stable with spline regularization, level-set recovery."""
    log.info(f"  [Deep-Koopman-KANDy] Training (seed={model_seed})")

    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    ae = KANAutoEncoder(3, MID, LATENT, **KAN_KWARGS)
    model = DeepKoopmanCTStable(ae, LATENT).to(DEVICE)

    traj_full = torch.cat([train_traj, test_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DK_KANDY_EPOCHS)
    mse_fn = nn.MSELoss()

    for ep in range(1, DK_KANDY_EPOCHS + 1):
        model.train()
        for x_t, x_tp1 in train_dl:
            x_t, x_tp1 = x_t.to(DEVICE), x_tp1.to(DEVICE)
            out = model.forward_one_step(x_t, DT)
            z_true = model.ae.encoder(x_tp1)
            loss = (mse_fn(out["x_tp1_pred"], x_tp1)
                    + mse_fn(out["z_tp1_pred"], z_true)
                    + 0.5 * mse_fn(out["x_t_rec"], x_t)
                    + DK_KANDY_SPLINE_REG * _spline_regularization(model.ae))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 50 == 0:
            log.info(f"    Ep {ep}")

    model.eval()

    # Prediction (on CPU for level-set analysis compatibility)
    model_cpu = model.cpu()
    mse, nrmse_2tau = eval_prediction_koopman(model_cpu, test_traj)
    assert nrmse_2tau < 5.0, f"KANDy NRMSE sanity check failed: {nrmse_2tau}"

    # Dictionary via level-set analysis (already on CPU)
    x_flat = train_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 5000, replace=False)
    x_sub_t = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)

    detected = set()
    decomp_log = []
    for zi in range(LATENT):
        dr = full_decomposition(model_cpu, x_sub_t, zi, target_type='latent',
                                max_degree=3, lasso_alpha=DK_KANDY_LASSO_ALPHA, h_degree=4)
        pairs = sorted(zip(dr['inner']['active_names'],
                           dr['inner']['active_coeffs']),
                       key=lambda x: -abs(x[1]))
        top = [(n, float(c)) for n, c in pairs if n != "1"][:5]
        if dr['recon_r2'] > 0.8:
            detected |= {n for n, c in top}

        decomp_log.append({
            "latent_dim": zi,
            "r2_hg": float(dr['recon_r2']),
            "top_terms": top,
        })

    scores = score_dictionary(detected)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "method": "Deep-Koopman-KANDy",
        "one_step_mse": mse,
        "nrmse_2tau": nrmse_2tau,
        "n_params": n_params,
        "train_epochs": DK_KANDY_EPOCHS,
        **scores,
        "decomp_log": decomp_log,
    }


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 2: MLP + Stable + PySR
# ═══════════════════════════════════════════════════════════════════════════

class MLPAutoEncoder(nn.Module):
    def __init__(self, input_size, mid_size, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, mid_size), nn.SiLU(),
            nn.Linear(mid_size, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, mid_size), nn.SiLU(),
            nn.Linear(mid_size, input_size),
        )

    def forward(self, X):
        Y = self.encoder(X)
        return Y, self.decoder(Y)


def run_mlp_stable(train_traj, test_traj, model_seed):
    """Train MLP + stable, then run PySR on encoder."""
    log.info(f"  [MLP+Stable] Training (seed={model_seed})")

    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    ae = MLPAutoEncoder(3, MID, LATENT)
    traj_full = torch.cat([train_traj, test_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:N_TRAIN]),
                          batch_size=2048, shuffle=True, drop_last=True)
    model = DeepKoopmanCTStable(ae, LATENT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    mse_fn = nn.MSELoss()

    for ep in range(1, MLP_EPOCHS + 1):
        model.train()
        for x_t, x_tp1 in train_dl:
            x_t, x_tp1 = x_t.to(DEVICE), x_tp1.to(DEVICE)
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
            log.info(f"    Ep {ep}")

    model = model.cpu().eval()

    # Prediction
    mse, nrmse_2tau = eval_prediction_koopman(model, test_traj)
    assert nrmse_2tau < 5.0, f"MLP NRMSE sanity check failed: {nrmse_2tau}"

    # PySR on frozen encoder
    x_train_flat = train_traj.reshape(-1, 3)
    x_test_flat = test_traj.reshape(-1, 3)
    # Subsample for PySR (full dataset is too large)
    tr_idx = np.random.choice(len(x_train_flat), min(5000, len(x_train_flat)), replace=False)
    te_idx = np.random.choice(len(x_test_flat), min(2000, len(x_test_flat)), replace=False)

    def encoder_fn(x):
        return model.ae.encoder(x)

    detected, pysr_log = run_pysr_on_encoder(
        encoder_fn,
        x_train_flat[tr_idx],
        x_test_flat[te_idx],
        LATENT,
        method_name="MLP+Stable",
        seed=model_seed,
    )

    scores = score_dictionary(detected)
    n_params = sum(p.numel() for p in model.parameters())

    # Assert PySR produced expressions for every latent
    assert len(pysr_log) == LATENT, f"PySR missing results: got {len(pysr_log)}"

    return {
        "method": "MLP+Stable+PySR",
        "one_step_mse": mse,
        "nrmse_2tau": nrmse_2tau,
        "n_params": n_params,
        "train_epochs": MLP_EPOCHS,
        **scores,
        "pysr_log": pysr_log,
    }


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 3: SINDy-AE + PySR
# ═══════════════════════════════════════════════════════════════════════════

def run_sindy_ae(train_traj, test_traj, train_xdot, test_xdot, model_seed):
    """Train SINDy-AE with Champion et al. protocol, then PySR on encoder."""
    log.info(f"  [SINDy-AE] Training (seed={model_seed})")

    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    model = SindyAutoencoder(
        input_dim=3,
        latent_dim=LATENT,
        widths=SINDY_WIDTHS,
        activation='sigmoid',
        poly_order=3,
        model_order=1,
        coefficient_init='constant',
    ).to(DEVICE)

    x_train = train_traj.reshape(-1, 3)
    dx_train = train_xdot.reshape(-1, 3)
    x_test = test_traj.reshape(-1, 3)
    dx_test = test_xdot.reshape(-1, 3)

    train_ds = TensorDataset(x_train, dx_train)
    train_dl = DataLoader(train_ds, batch_size=2048, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=SINDY_LR)

    # Phase 1: Pre-threshold training
    log.info(f"    Phase 1: {SINDY_PRE_EPOCHS} epochs with L1 regularization")
    for ep in range(1, SINDY_PRE_EPOCHS + 1):
        model.train()
        for x_batch, dx_batch in train_dl:
            x_batch, dx_batch = x_batch.to(DEVICE), dx_batch.to(DEVICE)
            out = model(x_batch, dx_batch)
            total, parts, _ = sindy_loss(
                out, {'x': x_batch, 'dx': dx_batch}, model,
                SINDY_LOSS_WEIGHTS, model_order=1,
            )
            opt.zero_grad()
            total.backward()
            opt.step()

        if ep % SINDY_THRESHOLD_EVERY == 0:
            model.apply_thresholding(SINDY_THRESHOLD)
            active = model.active_terms().sum().item()
            log.info(f"    Ep {ep}: thresholding, {active:.0f} active terms")

    # Phase 2: Refinement (L1 weight → 0)
    log.info(f"    Phase 2: {SINDY_REFINE_EPOCHS} epochs refinement (no L1)")
    refine_weights = dict(SINDY_LOSS_WEIGHTS)
    refine_weights['sindy_regularization'] = 0.0

    for ep in range(1, SINDY_REFINE_EPOCHS + 1):
        model.train()
        for x_batch, dx_batch in train_dl:
            x_batch, dx_batch = x_batch.to(DEVICE), dx_batch.to(DEVICE)
            out = model(x_batch, dx_batch)
            _, _, refinement_loss = sindy_loss(
                out, {'x': x_batch, 'dx': dx_batch}, model,
                refine_weights, model_order=1,
            )
            opt.zero_grad()
            refinement_loss.backward()
            opt.step()

        if ep % 1000 == 0:
            log.info(f"    Refine ep {ep}")

    model = model.cpu().eval()

    # Prediction
    mse, nrmse_2tau = eval_prediction_sindy(model, test_traj, test_xdot)
    assert nrmse_2tau < 10.0, f"SINDy-AE NRMSE sanity check failed: {nrmse_2tau}"

    # PySR on frozen encoder
    tr_idx = np.random.choice(len(x_train), min(5000, len(x_train)), replace=False)
    te_idx = np.random.choice(len(x_test), min(2000, len(x_test)), replace=False)

    def encoder_fn(x):
        return model.encode(x)

    detected, pysr_log = run_pysr_on_encoder(
        encoder_fn,
        x_train[tr_idx],
        x_test[te_idx],
        LATENT,
        method_name="SINDy-AE",
        seed=model_seed,
    )

    scores = score_dictionary(detected)
    n_params = sum(p.numel() for p in model.parameters())

    assert len(pysr_log) == LATENT, f"PySR missing results: got {len(pysr_log)}"

    return {
        "method": "SINDy-AE+PySR",
        "one_step_mse": mse,
        "nrmse_2tau": nrmse_2tau,
        "n_params": n_params,
        "train_epochs": SINDY_PRE_EPOCHS + SINDY_REFINE_EPOCHS,
        **scores,
        "pysr_log": pysr_log,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PySR SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def pysr_smoke_test():
    """Verify PySR works before running the full experiment."""
    log.info("PySR smoke test...")
    from pysr import PySRRegressor

    x = np.random.randn(200, 1)
    y = x[:, 0]**2 + x[:, 0]
    model = PySRRegressor(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=10,
        populations=10,
        niterations=20,
        random_state=0,
        deterministic=True,
        parallelism="serial",
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )
    model.fit(x, y, variable_names=["x"])
    pred = model.predict(x)
    r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
    log.info(f"  PySR smoke test: R²={r2:.3f}, best={model.get_best()['equation']}")
    assert r2 > 0.9, f"PySR smoke test failed: R²={r2}"
    log.info("  PySR smoke test PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("  Symbolic Dictionary Recovery Benchmark — Lorenz System")
    log.info(f"  Device: {DEVICE}")
    log.info("=" * 70)

    # Smoke test PySR
    pysr_smoke_test()

    # CSV output
    csv_path = OUTDIR / "results.csv"
    csv_fields = [
        "method", "seed", "data_seed", "model_seed",
        "one_step_mse", "nrmse_2tau",
        "jaccard", "precision", "recall",
        "n_detected", "n_tp", "n_fp", "detected",
        "n_params", "train_epochs",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # JSONL for PySR/decomposition logs
    jsonl_path = OUTDIR / "expressions.jsonl"
    jsonl_file = open(jsonl_path, "w")

    all_results = {m: [] for m in [
        "EDMD-poly (oracle)", "Deep-Koopman-KANDy", "MLP+Stable+PySR", "SINDy-AE+PySR"
    ]}

    # Generate data once (matches ablation_lorenz.py: set_seed(42))
    data_seed = 42
    log.info(f"  Generating shared data (data_seed={data_seed})...")
    traj, xdot = generate_data(data_seed)
    train_traj, test_traj, train_xdot, test_xdot = split_data(traj, xdot)

    # ── EDMD-poly (oracle): deterministic, run once ──
    log.info(f"\n  EDMD-poly (oracle) — degree-3 polynomial dictionary")
    EDMD_NAMES = ["x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2",
                  "x^3", "x^2y", "x^2z", "xy^2", "xyz", "xz^2", "y^3", "y^2z", "yz^2", "z^3"]

    def _poly_lift_d3(X):
        x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        return np.column_stack([
            x, y, z, x**2, x*y, x*z, y**2, y*z, z**2,
            x**3, x**2*y, x**2*z, x*y**2, x*y*z, x*z**2, y**3, y**2*z, y*z**2, z**3,
        ])

    X_t = train_traj[:, :-1, :].reshape(-1, 3).numpy()
    X_tp1 = train_traj[:, 1:, :].reshape(-1, 3).numpy()
    Z_t = _poly_lift_d3(X_t)
    Z_tp1 = _poly_lift_d3(X_tp1)
    K_edmd = Z_tp1.T @ np.linalg.pinv(Z_t.T)

    # One-step MSE
    Xv_t = test_traj[:, :-1, :].reshape(-1, 3).numpy()
    Xv_tp1 = test_traj[:, 1:, :].reshape(-1, 3).numpy()
    Zv_t = _poly_lift_d3(Xv_t)
    edmd_pred = (K_edmd @ Zv_t.T).T[:, :3]
    edmd_mse = float(np.mean((edmd_pred - Xv_tp1)**2))

    # NRMSE@2tau via rollout
    steps_2tau = 2 * STEPS_PER_TAU
    max_steps = min(steps_2tau, test_traj.shape[1] - 1)
    edmd_rollout_errors = []
    for ic in range(min(N_ROLLOUT_ICS, test_traj.shape[0])):
        true = test_traj[ic, :max_steps+1].numpy()
        x = true[0].copy()
        xs = [x.copy()]
        for _ in range(max_steps):
            zz = _poly_lift_d3(x.reshape(1, -1)).flatten()
            x = (K_edmd @ zz)[:3].copy()
            xs.append(x.copy())
        pred = np.array(xs)
        nrmse = compute_nrmse(pred, true)
        edmd_rollout_errors.append(float(nrmse[min(steps_2tau, len(nrmse)-1)]))
    edmd_nrmse_2tau = float(np.mean(edmd_rollout_errors))

    # Dictionary via K-I thresholding
    K_dev = K_edmd - np.eye(K_edmd.shape[0])
    K_dev_state = K_dev[:3, :]
    edmd_thresh = 0.01 * np.abs(K_dev_state).max()
    edmd_detected = set()
    for col in range(len(EDMD_NAMES)):
        if np.any(np.abs(K_dev_state[:, col]) > edmd_thresh):
            edmd_detected.add(EDMD_NAMES[col])
    edmd_scores = score_dictionary(edmd_detected)

    log.info(f"  EDMD-poly: MSE={edmd_mse:.4e} NRMSE@2τ={edmd_nrmse_2tau:.3f} "
             f"Jaccard={edmd_scores['jaccard']:.3f} detected={edmd_scores['detected']}")

    # Store as single result (deterministic — same for all seeds)
    edmd_result = {
        "method": "EDMD-poly (oracle)",
        "one_step_mse": edmd_mse,
        "nrmse_2tau": edmd_nrmse_2tau,
        "n_params": K_edmd.size,
        "train_epochs": 0,
        **edmd_scores,
    }
    for _ in SEEDS:
        all_results["EDMD-poly (oracle)"].append(edmd_result)

    csv_writer.writerow({
        "method": edmd_result["method"], "seed": "all",
        "data_seed": data_seed, "model_seed": "N/A",
        "one_step_mse": edmd_result["one_step_mse"],
        "nrmse_2tau": edmd_result["nrmse_2tau"],
        "jaccard": edmd_result["jaccard"], "precision": edmd_result["precision"],
        "recall": edmd_result["recall"], "n_detected": edmd_result["n_detected"],
        "n_tp": edmd_result["n_tp"], "n_fp": edmd_result["n_fp"],
        "detected": ";".join(edmd_result["detected"]),
        "n_params": edmd_result["n_params"], "train_epochs": 0,
    })
    csv_file.flush()

    for seed in SEEDS:
        log.info(f"\n{'='*60}")
        log.info(f"  SEED {seed}")
        log.info(f"{'='*60}")

        model_seed = seed
        log.info(f"  data_seed={data_seed}, model_seed={model_seed}")

        # --- Method 1: KANDy ---
        t0 = time.time()
        r1 = run_deep_koopman_kandy(train_traj, test_traj, model_seed)
        r1["wall_time"] = time.time() - t0
        all_results["Deep-Koopman-KANDy"].append(r1)
        log.info(f"  [Deep-Koopman-KANDy] MSE={r1['one_step_mse']:.4e} NRMSE@2τ={r1['nrmse_2tau']:.3f} "
                 f"Jaccard={r1['jaccard']:.3f} ({r1['wall_time']:.0f}s)")

        csv_writer.writerow({
            "method": r1["method"], "seed": seed,
            "data_seed": data_seed, "model_seed": model_seed,
            "one_step_mse": r1["one_step_mse"], "nrmse_2tau": r1["nrmse_2tau"],
            "jaccard": r1["jaccard"], "precision": r1["precision"],
            "recall": r1["recall"], "n_detected": r1["n_detected"],
            "n_tp": r1["n_tp"], "n_fp": r1["n_fp"],
            "detected": ";".join(r1["detected"]),
            "n_params": r1["n_params"], "train_epochs": r1["train_epochs"],
        })
        json.dump({"method": r1["method"], "seed": seed,
                    "decomp_log": r1.get("decomp_log", [])},
                   jsonl_file, default=str)
        jsonl_file.write("\n")
        csv_file.flush()

        # --- Method 2: MLP + Stable + PySR ---
        t0 = time.time()
        r2 = run_mlp_stable(train_traj, test_traj, model_seed)
        r2["wall_time"] = time.time() - t0
        all_results["MLP+Stable+PySR"].append(r2)
        log.info(f"  [MLP+PySR] MSE={r2['one_step_mse']:.4e} NRMSE@2τ={r2['nrmse_2tau']:.3f} "
                 f"Jaccard={r2['jaccard']:.3f} ({r2['wall_time']:.0f}s)")

        csv_writer.writerow({
            "method": r2["method"], "seed": seed,
            "data_seed": data_seed, "model_seed": model_seed,
            "one_step_mse": r2["one_step_mse"], "nrmse_2tau": r2["nrmse_2tau"],
            "jaccard": r2["jaccard"], "precision": r2["precision"],
            "recall": r2["recall"], "n_detected": r2["n_detected"],
            "n_tp": r2["n_tp"], "n_fp": r2["n_fp"],
            "detected": ";".join(r2["detected"]),
            "n_params": r2["n_params"], "train_epochs": r2["train_epochs"],
        })
        json.dump({"method": r2["method"], "seed": seed,
                    "pysr_log": r2.get("pysr_log", [])},
                   jsonl_file, default=str)
        jsonl_file.write("\n")
        csv_file.flush()

        # --- Method 3: SINDy-AE + PySR ---
        t0 = time.time()
        r3 = run_sindy_ae(train_traj, test_traj, train_xdot, test_xdot, model_seed)
        r3["wall_time"] = time.time() - t0
        all_results["SINDy-AE+PySR"].append(r3)
        log.info(f"  [SINDy+PySR] MSE={r3['one_step_mse']:.4e} NRMSE@2τ={r3['nrmse_2tau']:.3f} "
                 f"Jaccard={r3['jaccard']:.3f} ({r3['wall_time']:.0f}s)")

        csv_writer.writerow({
            "method": r3["method"], "seed": seed,
            "data_seed": data_seed, "model_seed": model_seed,
            "one_step_mse": r3["one_step_mse"], "nrmse_2tau": r3["nrmse_2tau"],
            "jaccard": r3["jaccard"], "precision": r3["precision"],
            "recall": r3["recall"], "n_detected": r3["n_detected"],
            "n_tp": r3["n_tp"], "n_fp": r3["n_fp"],
            "detected": ";".join(r3["detected"]),
            "n_params": r3["n_params"], "train_epochs": r3["train_epochs"],
        })
        json.dump({"method": r3["method"], "seed": seed,
                    "pysr_log": r3.get("pysr_log", [])},
                   jsonl_file, default=str)
        jsonl_file.write("\n")
        csv_file.flush()

    csv_file.close()
    jsonl_file.close()

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE (markdown, pasteable into paper)
    # ═══════════════════════════════════════════════════════════════════
    log.info(f"\n\n{'='*90}")
    log.info("  RESULTS TABLE (markdown)")
    log.info(f"{'='*90}\n")

    header = (
        "| Method | Epochs | Params | 1-Step MSE | NRMSE@2τ_Λ | "
        "Jaccard ↑ | Precision ↑ | Recall ↑ |"
    )
    sep = "|" + "|".join(["---"] * 8) + "|"
    print(header)
    print(sep)

    for method_name in ["Deep-Koopman-KANDy", "MLP+Stable+PySR", "SINDy-AE+PySR"]:
        rs = all_results[method_name]
        mses = [r["one_step_mse"] for r in rs]
        nrmses = [r["nrmse_2tau"] for r in rs]
        jacs = [r["jaccard"] for r in rs]
        precs = [r["precision"] for r in rs]
        recs = [r["recall"] for r in rs]
        epochs = rs[0]["train_epochs"]
        params = rs[0]["n_params"]

        row = (
            f"| {method_name} | {epochs} | {params:,} | "
            f"${np.mean(mses):.2e} \\pm {np.std(mses):.0e}$ | "
            f"${np.mean(nrmses):.2f} \\pm {np.std(nrmses):.2f}$ | "
            f"${np.mean(jacs):.2f} \\pm {np.std(jacs):.2f}$ | "
            f"${np.mean(precs):.2f} \\pm {np.std(precs):.2f}$ | "
            f"${np.mean(recs):.2f} \\pm {np.std(recs):.2f}$ |"
        )
        print(row)

    # Sanity checks on final scores
    for method_name, rs in all_results.items():
        for r in rs:
            assert 0 <= r["recall"] <= 1, f"{method_name} recall out of range: {r['recall']}"

    elapsed = time.time() - t_start
    log.info(f"\n  Total time: {elapsed/3600:.1f} hours")
    log.info(f"  Results: {csv_path}")
    log.info(f"  Expressions: {jsonl_path}")

    # Flag discrepancies
    log.info(f"\n  DISCREPANCY NOTE: burn_in=500 used (matches code), "
             f"paper text may say 300. Verify and update paper.")


if __name__ == "__main__":
    main()
