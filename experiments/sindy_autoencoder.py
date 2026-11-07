"""
SINDy Autoencoder — PyTorch port of Champion, Lusch, Kutz, Brunton (PNAS 2019).

Reference: github.com/kpchamp/SindyAutoencoders (original TF1 implementation)

This port preserves two deliberate choices from the original:

1. Derivatives of encoder/decoder outputs w.r.t. input are computed analytically
   via the chain rule through the linear layers, not with torch.autograd.grad.
   The original authors do this for speed — per-sample autograd on a batch is
   slow, and the analytical form is closed for the activation families they
   support (ELU, ReLU, sigmoid, linear).

2. The L0 sparsity proxy is implemented as a coefficient_mask buffer that gets
   updated OUTSIDE the training loop (typically every N epochs, zeroing
   coefficients below a threshold). The mask is a buffer, not a parameter —
   it participates in forward but not in backprop.

Usage (first-order, e.g. Lorenz):

    model = SindyAutoencoder(
        input_dim=128, latent_dim=3, widths=[64, 32],
        poly_order=3, activation='sigmoid', model_order=1,
    )
    outputs = model(x, dx)
    total, parts, refinement_total = sindy_loss(
        outputs, {'x': x, 'dx': dx}, model,
        loss_weights={'decoder': 1.0, 'sindy_z': 1e-4,
                      'sindy_x': 1e-4, 'sindy_regularization': 1e-5},
        model_order=1,
    )
    total.backward()

Sequential thresholding (call periodically, e.g. every 500 epochs):

    model.apply_thresholding(threshold=0.1)

Post-training encoder extraction (for downstream symbolic regression):

    model.eval()
    with torch.no_grad():
        z = model.encode(x_test)  # shape (N, latent_dim)
"""

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