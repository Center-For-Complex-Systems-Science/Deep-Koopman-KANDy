import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from training import rollout


def make_plots(model, history, train_traj, val_traj, dt, rollout_steps=2000):
    epochs = np.arange(1, len(history["train_total"]) + 1)

    # Loss curves
    plt.figure()
    plt.plot(epochs, history["train_total"], label="train total")
    plt.plot(epochs, history["val_total"], label="val total")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Total loss")
    plt.legend(); plt.tight_layout()

    plt.figure()
    plt.plot(epochs, history["train_predx"], label="train pred x_{t+1}")
    plt.plot(epochs, history["val_predx"], label="val pred x_{t+1}")
    plt.xlabel("epoch"); plt.ylabel("mse"); plt.title("One-step prediction loss")
    plt.legend(); plt.tight_layout()

    plt.figure()
    plt.plot(epochs, history["train_latent"], label="train latent")
    plt.plot(epochs, history["val_latent"], label="val latent")
    plt.xlabel("epoch"); plt.ylabel("mse"); plt.title("Latent consistency loss")
    plt.legend(); plt.tight_layout()

    plt.figure()
    plt.plot(epochs, history["train_recon"], label="train recon")
    plt.plot(epochs, history["val_recon"], label="val recon")
    plt.xlabel("epoch"); plt.ylabel("mse"); plt.title("Reconstruction loss")
    plt.legend(); plt.tight_layout()

    # One-step prediction trace
    model.eval()
    device = next(model.parameters()).device

    x_true = val_traj[0]
    T = min(2000, x_true.shape[0] - 1)
    x_t = torch.from_numpy(x_true[:T].numpy()).to(device)
    x_tp1 = torch.from_numpy(x_true[1:T+1].numpy()).to(device)

    with torch.no_grad():
        out = model.forward_one_step(x_t, dt)
        x_pred = out["x_tp1_pred"].cpu().numpy()

    x_tp1_np = x_tp1.cpu().numpy()

    for i, label in enumerate(["x", "y", "z"]):
        plt.figure()
        plt.plot(x_tp1_np[:, i], label=f"true {label}")
        plt.plot(x_pred[:, i], label=f"pred {label}")
        plt.title(f"One-step prediction ({label} component)")
        plt.xlabel("t index"); plt.ylabel(label)
        plt.legend(); plt.tight_layout()

    # Multi-step rollout vs ground truth
    x0 = torch.from_numpy(val_traj[0, 0].numpy()).to(device)
    with torch.no_grad():
        x_roll = rollout(model, x0, dt, steps=rollout_steps).cpu().numpy()

    x_gt = val_traj[0, :rollout_steps+1].numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_gt[:, 0], x_gt[:, 1], x_gt[:, 2], label="ground truth")
    ax.plot(x_roll[:, 0], x_roll[:, 1], x_roll[:, 2], label="model rollout")
    ax.set_title("Lorenz trajectory rollout (3D)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(x_gt[:, 0], x_gt[:, 2], label="gt (x,z)")
    plt.plot(x_roll[:, 0], x_roll[:, 2], label="rollout (x,z)")
    plt.xlabel("x"); plt.ylabel("z")
    plt.title("Phase projection: x vs z")
    plt.legend(); plt.tight_layout()

    plt.show()

import numpy as np
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def inspect_kanlinear(layer, x, beta=3.0):
    """
    Reconstruct per-edge quantities for a fast KANLinear layer.

    Returns
    -------
    dict with:
        x_in          : (B, in_features)
        postspline    : (B, out_features, in_features)
        postacts      : (B, out_features, in_features)   # base + spline + symbolic
        alpha         : (out_features, in_features)      # edge opacity
        numeric_mask  : (out_features, in_features)
        symbolic_mask : (out_features, in_features)
    """
    x = x.reshape(-1, layer.in_features)
    B = x.size(0)

    frozen = getattr(layer, "_frozen_mask", None)
    if frozen is None:
        frozen = torch.zeros(
            layer.out_features,
            layer.in_features,
            dtype=torch.bool,
            device=x.device,
        )

    numeric_mask = (~frozen).float()

    # Numeric edge pieces
    bw = layer.base_weight * numeric_mask
    ssw = layer.scaled_spline_weight * numeric_mask.unsqueeze(-1)

    base_vals = layer.base_activation(x)[:, None, :] * bw[None, :, :]   # (B, O, I)

    bases = layer.b_splines(x)                                           # (B, I, C)
    spline_vals = torch.einsum("bic,oic->boi", bases, ssw)               # (B, O, I)

    # Symbolic edge pieces
    symbolic_vals = torch.zeros_like(base_vals)
    symbolic_mask = torch.zeros(
        layer.out_features,
        layer.in_features,
        dtype=torch.float32,
        device=x.device,
    )

    if hasattr(layer, "_symbolic_fns") and layer._symbolic_fns:
        for (out_idx, in_idx), fn in layer._symbolic_fns.items():
            symbolic_vals[:, out_idx, in_idx] = fn(x[:, in_idx])
            symbolic_mask[out_idx, in_idx] = 1.0

    postacts = base_vals + spline_vals + symbolic_vals

    # Simple forward importance for opacity
    # Normalize per layer so the figure is readable.
    score = postacts.abs().mean(dim=0)
    score = score / score.max().clamp_min(1e-8)
    alpha = torch.tanh(beta * score)

    return {
        "x_in": x.detach().cpu(),
        "postspline": spline_vals.detach().cpu(),
        "postacts": postacts.detach().cpu(),
        "alpha": alpha.detach().cpu(),
        "numeric_mask": numeric_mask.detach().cpu(),
        "symbolic_mask": symbolic_mask.detach().cpu(),
    }


@torch.no_grad()
def collect_kan_traces(model, x, kan_cls=None, beta=3.0):
    """
    Runs one forward pass and records KANLinear layers in actual execution order.
    """
    if kan_cls is None:
        # infer the class from the first matching module name
        kan_modules = [m for m in model.modules() if m.__class__.__name__ == "KANLinear"]
    else:
        kan_modules = [m for m in model.modules() if isinstance(m, kan_cls)]

    if not kan_modules:
        raise ValueError("No KANLinear layers found in model.")

    traces = []
    hooks = []

    def hook(module, inputs, output):
        x_in = inputs[0]
        traces.append((module, inspect_kanlinear(module, x_in, beta=beta)))

    for m in kan_modules:
        hooks.append(m.register_forward_hook(hook))

    was_training = model.training
    try:
        model.eval()
        _ = model(x)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    return traces


@torch.no_grad()
def plot_kan_model(
    model,
    x,
    *,
    kan_cls=None,
    plot_mode="spline",   # "spline" (PyKAN-like) or "total"
    beta=3.0,
    figsize=(12, 8),
    title=None,
    in_vars=None,
    out_vars=None,
    sample=False,
):
    """
    Plot a PyKAN-style diagram for a chained model made of KANLinear layers.

    Parameters
    ----------
    model : nn.Module
    x     : representative batch, shape (..., in_features)
    plot_mode : "spline" or "total"
        "spline" mimics PyKAN's edge mini-plots more closely.
        "total" plots base + spline + symbolic per edge.
    """
    traces = collect_kan_traces(model, x, kan_cls=kan_cls, beta=beta)

    widths = [traces[0][0].in_features] + [m.out_features for m, _ in traces]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, right = 0.06, 0.94
    top, bottom = 0.92, 0.08
    ys = np.linspace(top, bottom, len(widths))

    xs_per_row = []
    for n in widths:
        xs = np.linspace(left, right, n) if n > 1 else np.array([0.5])
        xs_per_row.append(xs)

    # Draw layer nodes
    for row_idx, xs in enumerate(xs_per_row):
        y = ys[row_idx]
        ax.scatter(xs, np.full_like(xs, y), s=80, color="black", zorder=5)

    # Labels
    if in_vars is not None:
        for xi, name in zip(xs_per_row[0], in_vars):
            ax.text(xi, ys[0] + 0.04, str(name), ha="center", va="bottom", fontsize=10)

    if out_vars is not None:
        for xo, name in zip(xs_per_row[-1], out_vars):
            ax.text(xo, ys[-1] - 0.04, str(name), ha="center", va="top", fontsize=10)

    if title is not None:
        ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=14)

    # Draw each KAN layer
    for l, (module, tr) in enumerate(traces):
        n_in = module.in_features
        n_out = module.out_features

        y_in, y_out = ys[l], ys[l + 1]
        y_mid = 0.5 * (y_in + y_out)

        n_edges = n_in * n_out
        xslots = np.linspace(left, right, n_edges) if n_edges > 1 else np.array([0.5])

        box_h = min(0.11, abs(y_in - y_out) * 0.45)
        box_w = min(0.08, 0.7 / max(n_edges, 1))

        for i in range(n_in):
            for j in range(n_out):
                idx = i * n_out + j
                xc = xslots[idx]

                alpha = float(tr["alpha"][j, i])
                sym = bool(tr["symbolic_mask"][j, i] > 0)
                num = bool(tr["numeric_mask"][j, i] > 0)

                if sym and num:
                    color = "purple"
                elif sym:
                    color = "red"
                elif num:
                    color = "black"
                else:
                    color = "lightgray"

                # connection lines
                ax.plot(
                    [xs_per_row[l][i], xc],
                    [y_in, y_mid + box_h / 2],
                    color=color,
                    alpha=alpha,
                    lw=1.3,
                )
                ax.plot(
                    [xc, xs_per_row[l + 1][j]],
                    [y_mid - box_h / 2, y_out],
                    color=color,
                    alpha=alpha,
                    lw=1.3,
                )

                # mini edge plot
                iax = fig.add_axes([xc - box_w / 2, y_mid - box_h / 2, box_w, box_h])

                xin = tr["x_in"][:, i].numpy()
                if plot_mode == "spline":
                    yedge = tr["postspline"][:, j, i].numpy()
                elif plot_mode == "total":
                    yedge = tr["postacts"][:, j, i].numpy()
                else:
                    raise ValueError("plot_mode must be 'spline' or 'total'")

                order = np.argsort(xin)
                iax.plot(xin[order], yedge[order], color=color, lw=1.0)
                if sample:
                    iax.scatter(xin[order], yedge[order], color=color, s=3)

                iax.set_xticks([])
                iax.set_yticks([])
                for spine in iax.spines.values():
                    spine.set_color(color)
                    spine.set_alpha(alpha)

    return fig