"""
Publication-quality Standard Map: true dynamics vs stable Koopman rollout.

Two-pass neon rendering on a light background, side by side.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from training import set_seed
from standard_map_stable import StableKoopmanKAN

TAU = 2.0 * np.pi
OUTDIR = Path("results/standard_map")

# ── palette ────────────────────────────────────────────────────────────────────
NEON = LinearSegmentedColormap.from_list("neon", [
    "#ff0aaa",   # hot pink
    "#7700ff",   # violet
    "#0055ff",   # royal blue
    "#00c8ff",   # sky cyan
    "#00ffaa",   # mint
    "#aaff00",   # chartreuse
    "#ffdd00",   # gold
    "#ff6600",   # orange
    "#ff0aaa",   # wrap back to pink
], N=2048)


def wrap(y):
    """[0, 2pi) -> [-pi, pi) for plotting."""
    return ((y + np.pi) % TAU) - np.pi


def load_stable_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = StableKoopmanKAN(cfg["mid"], cfg["latent"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


# ── rollouts ───────────────────────────────────────────────────────────────────

def true_rollout(K, n_theta, n_p, n_steps, discard):
    """Vectorized true Standard Map on the torus."""
    x0 = np.linspace(0, TAU, n_theta, endpoint=False, dtype=np.float32)
    y0 = np.linspace(0, TAU, n_p, endpoint=True, dtype=np.float32)
    xg, yg = np.meshgrid(x0, y0, indexing="xy")
    x = xg.ravel().copy()
    y = yg.ravel().copy()

    # colour key: initial momentum mapped to [0, 1]
    c_orbit = 1.0 - ((wrap(y) + np.pi) / TAU).astype(np.float32)

    pts_x, pts_p, pts_c = [], [], []
    for n in range(n_steps):
        y = (y + K * np.sin(x)) % TAU
        x = (x + y) % TAU
        if n >= discard:
            pts_x.append(x.copy())
            pts_p.append(wrap(y))
            pts_c.append(c_orbit)

    return np.concatenate(pts_x), np.concatenate(pts_p), np.concatenate(pts_c)


def model_rollout(model, K, n_theta, n_p, n_steps, discard):
    """Batched autoregressive rollout through the Koopman model."""
    x0 = np.linspace(0, TAU, n_theta, endpoint=False, dtype=np.float32)
    y0 = np.linspace(0, TAU, n_p, endpoint=True, dtype=np.float32)
    xg, yg = np.meshgrid(x0, y0, indexing="xy")
    x_flat = xg.ravel()
    y_flat = yg.ravel()

    c_orbit = 1.0 - ((wrap(y_flat) + np.pi) / TAU).astype(np.float32)

    xy = torch.tensor(np.column_stack([x_flat, y_flat]), dtype=torch.float32)

    pts_x, pts_p, pts_c = [], [], []
    with torch.no_grad():
        for n in range(n_steps):
            out = model.forward_one_step(xy)
            xy = out["xy_np1_pred"]
            if n >= discard:
                xy_np = xy.numpy()
                pts_x.append(xy_np[:, 0].copy())
                pts_p.append(wrap(xy_np[:, 1]))
                pts_c.append(c_orbit)
            if (n + 1) % 500 == 0:
                print(f"    step {n+1}/{n_steps}", flush=True)

    return np.concatenate(pts_x), np.concatenate(pts_p), np.concatenate(pts_c)


# ── rendering ──────────────────────────────────────────────────────────────────

def render_panel(ax, theta, p, c, title=None, subtitle=None):
    """Two-pass scatter: soft halo + crisp core."""
    rgba = NEON(c)

    ax.scatter(theta, p, s=0.35, c=rgba, alpha=0.08,
               linewidths=0, rasterized=True)          # soft halo
    ax.scatter(theta, p, s=0.06, c=rgba, alpha=0.85,
               linewidths=0, rasterized=True)          # crisp core

    # ── axes ──────────────────────────────────────────────────────────────
    ax.set_xlim(0, TAU)
    ax.set_ylim(-np.pi, np.pi)

    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, TAU])
    ax.set_xticklabels(
        ["$0$", r"$\frac{\pi}{2}$", r"$\pi$",
         r"$\frac{3\pi}{2}$", r"$2\pi$"],
        fontsize=25, color="#222233",
    )
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels(
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "$0$",
         r"$\frac{\pi}{2}$", r"$\pi$"],
        fontsize=25, color="#222233",
    )

    for spine in ax.spines.values():
        spine.set_color("#aaaabb")
        spine.set_linewidth(0.7)
    ax.tick_params(colors="#444455", length=4, width=0.7)
    ax.grid(color="#ccccda", linewidth=0.5, linestyle="-", alpha=0.7)

    ax.set_xlabel(r"$\theta$", fontsize=28, color="#111122", labelpad=10)
    ax.set_ylabel(r"$p$", fontsize=28, color="#111122", labelpad=10, rotation=0)

    if title:
        ax.set_title(title, fontsize=22, color="#111133", pad=14, fontweight="medium")

    if subtitle:
        ax.text(
            0.985, 0.02, subtitle,
            transform=ax.transAxes, fontsize=11,
            color="#888899", ha="right", va="bottom",
        )


# ── main ───────────────────────────────────────────────────────────────────────

def build_figure(
    ckpt_path    = None,
    K            = 0.97,
    n_theta      = 72,
    n_p          = 48,
    n_steps      = 2200,
    discard      = 200,
    dpi          = 240,
):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(0)

    if ckpt_path is None:
        ckpt_path = OUTDIR / "standard_map_stable_32_128.pt"
    model, cfg = load_stable_model(ckpt_path)
    K = cfg.get("K", K)
    mid, latent = cfg["mid"], cfg["latent"]
    print(f"K = {K}   model [{mid}, {latent}]")

    n_orbits = n_theta * n_p
    n_kept = n_steps - discard
    print(f"{n_orbits:,} orbits x {n_steps} steps ({n_kept} kept)")

    # ── true map ──────────────────────────────────────────────────────────
    print("True rollout ...")
    true_t, true_p, true_c = true_rollout(K, n_theta, n_p, n_steps, discard)
    print(f"  {len(true_t):,} points")

    # ── model rollout ─────────────────────────────────────────────────────
    print("Model rollout ...")
    mod_t, mod_p, mod_c = model_rollout(model, K, n_theta, n_p, n_steps, discard)
    print(f"  {len(mod_t):,} points")

    # ── eigenvalue stats for annotation ───────────────────────────────────
    M = model.koopman_matrix().numpy()
    eigs = np.linalg.eigvals(M)
    mags = np.abs(eigs)

    # ── figure ────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(30, 9.5), facecolor="white",
    )
    ax1.set_facecolor("#e8e8ec")
    ax2.set_facecolor("#e8e8ec")

    render_panel(
        ax1, true_t, true_p, true_c,
        #title="True Standard Map",
        #subtitle="KAM tori  \u00b7  island chains  \u00b7  chaotic sea",
    )
    render_panel(
        ax2, mod_t, mod_p, mod_c,
        #title=f"Stable Koopman Rollout  [{mid}, {latent}]",
        # subtitle=(
        #     r"$M = \exp\!(K - L^\top\! L)$"
        #     f",   "
        #     r"$|\lambda|$"
        #     f" $\\in$ [{mags.min():.3f}, {mags.max():.3f}]"
        # ),
    )

    # fig.suptitle(
    #     f"Chirikov Standard Map  \u2014  $K = {K}$",
    #     fontsize=30, color="#111133", y=0.99, fontweight="light",
    # )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("png", "pdf", "svg"):
        out = OUTDIR / f"standard_map_hero_two.{ext}"
        fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight")
        print(f"Saved -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
