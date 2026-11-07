"""
Ikeda foliation analysis: inner function decomposition + gradient identity.

Loads a trained pruned Ikeda model and:
1. Extracts per-edge spline functions psi_{j,i} for each inner function u_j
2. Fits quadratics to identify which channels learn x^2 + y^2
3. Runs Lasso on inner functions to recover g = x^2 + y^2
4. Applies the arc-length gradient formula to recover h'(g)
5. Fits h(g) = a*cos(t(g)) + b*sin(t(g)) + c where t(g) = 0.4 - 6/(1+g)
6. Produces publication figures

Run from project root:
    python experiments/ikeda_foliation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Lasso
from numpy.linalg import lstsq
from scipy.interpolate import interp1d

from ikeda import generate_ikeda
from ikeda_koopman import IkedaStableKoopman, build_poly_dict_2d
from manifold_analysis_foliation import extract_inner_functions, extract_outer_functions, channel_activity

FIGDIR = Path('figures')
FIGDIR.mkdir(exist_ok=True)
OUTDIR = Path('results/ikeda')
SEED = 0
N_SAMPLE = 30000


def load_model(seed=SEED):
    model = IkedaStableKoopman()
    ckpt = torch.load(OUTDIR / f'ikeda_pruned_seed_{seed}.pt',
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def load_data(n_sample=N_SAMPLE):
    traj = generate_ikeda(n_traj=200, n_steps=2000, burn_in=500, seed=42)
    pts = traj.reshape(-1, 2)
    np.random.seed(0)
    idx = np.random.choice(len(pts), n_sample, replace=False)
    return pts[idx]


# =====================================================================
# 1. Edge spline analysis: fit quadratics to each psi_{j,i}
# =====================================================================

def edge_spline_analysis(model, pts_sub):
    x_data = torch.from_numpy(pts_sub)
    u_vals, per_edge = extract_inner_functions(model.encoder, x_data)
    x_vals, y_vals = pts_sub[:, 0], pts_sub[:, 1]
    r2 = x_vals**2 + y_vals**2

    n_channels = u_vals.shape[1]
    results = []

    for j in range(n_channels):
        psi_x = per_edge[:, j, 0]
        psi_y = per_edge[:, j, 1]

        c2_x = np.polyfit(x_vals, psi_x, 2)
        c2_y = np.polyfit(y_vals, psi_y, 2)

        pred_x = np.polyval(c2_x, x_vals)
        pred_y = np.polyval(c2_y, y_vals)
        r2_x = 1 - np.sum((psi_x - pred_x)**2) / max(np.sum((psi_x - psi_x.mean())**2), 1e-12)
        r2_y = 1 - np.sum((psi_y - pred_y)**2) / max(np.sum((psi_y - psi_y.mean())**2), 1e-12)

        ratio = min(abs(c2_x[0]), abs(c2_y[0])) / max(abs(c2_x[0]), abs(c2_y[0]), 1e-12)
        same_sign = c2_x[0] * c2_y[0] > 0

        results.append({
            'channel': j,
            'x2_coeff': c2_x[0], 'x1_coeff': c2_x[1], 'x0_coeff': c2_x[2],
            'y2_coeff': c2_y[0], 'y1_coeff': c2_y[1], 'y0_coeff': c2_y[2],
            'r2_quad_x': r2_x, 'r2_quad_y': r2_y,
            'ratio': ratio, 'same_sign': same_sign,
        })

    return results, u_vals, per_edge


def plot_edge_splines(model, pts_sub, channels=(1, 5)):
    results, u_vals, per_edge = edge_spline_analysis(model, pts_sub)
    x_vals, y_vals = pts_sub[:, 0], pts_sub[:, 1]
    r2 = x_vals**2 + y_vals**2

    fig, axes = plt.subplots(len(channels), 3, figsize=(16, 5 * len(channels)))
    if len(channels) == 1:
        axes = axes[np.newaxis, :]

    for row, ch in enumerate(channels):
        r = results[ch]
        psi_x = per_edge[:, ch, 0]
        psi_y = per_edge[:, ch, 1]

        # psi(x)
        ax = axes[row, 0]
        sx = np.argsort(x_vals)
        ax.scatter(x_vals[sx], psi_x[sx], s=0.3, alpha=0.1, c='gray', rasterized=True)
        x_fine = np.linspace(x_vals.min(), x_vals.max(), 200)
        c2_x = [r['x2_coeff'], r['x1_coeff'], r['x0_coeff']]
        ax.plot(x_fine, np.polyval(c2_x, x_fine), 'r-', lw=2,
                label=f"{c2_x[0]:.3f}x^2 + {c2_x[1]:.3f}x + {c2_x[2]:.3f}")
        ax.set_xlabel('x'); ax.set_ylabel(f'psi_{ch},0(x)')
        ax.set_title(f'u[{ch}]: psi(x), quad R^2={r["r2_quad_x"]:.4f}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

        # psi(y)
        ax = axes[row, 1]
        sy = np.argsort(y_vals)
        ax.scatter(y_vals[sy], psi_y[sy], s=0.3, alpha=0.1, c='gray', rasterized=True)
        y_fine = np.linspace(y_vals.min(), y_vals.max(), 200)
        c2_y = [r['y2_coeff'], r['y1_coeff'], r['y0_coeff']]
        ax.plot(y_fine, np.polyval(c2_y, y_fine), 'r-', lw=2,
                label=f"{c2_y[0]:.3f}y^2 + {c2_y[1]:.3f}y + {c2_y[2]:.3f}")
        ax.set_xlabel('y'); ax.set_ylabel(f'psi_{ch},1(y)')
        ax.set_title(f'u[{ch}]: psi(y), quad R^2={r["r2_quad_y"]:.4f}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

        # u vs r^2
        ax = axes[row, 2]
        ax.scatter(r2, u_vals[:, ch], s=0.3, alpha=0.05, c='gray', rasterized=True)
        c1 = np.polyfit(r2, u_vals[:, ch], 1)
        r2_fine = np.linspace(r2.min(), r2.max(), 200)
        ax.plot(r2_fine, np.polyval(c1, r2_fine), 'b-', lw=2,
                label=f'{c1[0]:.4f}*r^2 + {c1[1]:.3f}')
        pred_lin = np.polyval(c1, r2)
        r2_lin = 1 - np.sum((u_vals[:, ch] - pred_lin)**2) / max(np.sum((u_vals[:, ch] - u_vals[:, ch].mean())**2), 1e-12)
        ax.set_xlabel('r^2 = x^2 + y^2'); ax.set_ylabel(f'u_{ch}')
        ax.set_title(f'u[{ch}] vs r^2, linear R^2={r2_lin:.4f}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    fig.suptitle('Edge Spline Functions: What Did the KAN Learn?', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'ikeda_edge_splines.png', dpi=200, bbox_inches='tight')
    fig.savefig(FIGDIR / 'ikeda_edge_splines.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR / "ikeda_edge_splines.png"}')
    return results


# =====================================================================
# 2. Lasso on inner functions
# =====================================================================

def lasso_inner_functions(model, pts_sub, max_degree=3, alphas=(1e-5, 1e-4, 1e-3, 1e-2)):
    x_data = torch.from_numpy(pts_sub)
    u_vals, _ = extract_inner_functions(model.encoder, x_data)
    x_np, y_np = pts_sub[:, 0], pts_sub[:, 1]

    M_dict, names = build_poly_dict_2d(x_np, y_np, max_degree=max_degree)
    col_mean = M_dict.mean(0)
    col_std = M_dict.std(0)
    col_std[col_std < 1e-12] = 1.0
    M_scaled = (M_dict - col_mean) / col_std

    all_results = {}
    for j in range(u_vals.shape[1]):
        target = u_vals[:, j]
        t_mean, t_std = target.mean(), max(target.std(), 1e-12)
        t_scaled = (target - t_mean) / t_std

        channel_results = []
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=50000, fit_intercept=True)
            lasso.fit(M_scaled, t_scaled)
            coeffs = t_std * lasso.coef_ / col_std
            intercept = t_mean + t_std * (lasso.intercept_ - np.sum(lasso.coef_ * col_mean / col_std))
            pred = M_dict @ coeffs + intercept
            ss_res = np.sum((target - pred)**2)
            ss_tot = np.sum((target - target.mean())**2)
            r2 = 1 - ss_res / max(ss_tot, 1e-12)

            mask = np.abs(coeffs) > 1e-4
            active = [(names[i], float(coeffs[i])) for i in np.where(mask)[0]]
            active.sort(key=lambda x: -abs(x[1]))

            x2_c = coeffs[names.index('x^2')]
            y2_c = coeffs[names.index('y^2')]
            ratio = min(abs(x2_c), abs(y2_c)) / max(abs(x2_c), abs(y2_c), 1e-12)

            channel_results.append({
                'alpha': alpha, 'r2': r2, 'n_active': int(mask.sum()),
                'active': active, 'x2': float(x2_c), 'y2': float(y2_c),
                'ratio': ratio, 'same_sign': bool(x2_c * y2_c > 0),
                'coeffs': coeffs, 'intercept': intercept,
            })
        all_results[j] = channel_results

    return all_results, names, M_dict


# =====================================================================
# 3. Gradient identity: recover h'(g) and h(g)
# =====================================================================

def gradient_identity(model, pts_sub, channel=1, n_bins=80):
    """Apply the arc-length gradient formula to recover h(g) where
    u_channel = h(g) and g = x^2 + y^2."""

    x_t = torch.from_numpy(pts_sub).float().requires_grad_(True)
    u_all = model.encoder.kan1(x_t)
    u_ch = u_all[:, channel]

    # grad f = grad u_channel
    grad_f = torch.autograd.grad(u_ch.sum(), x_t, create_graph=False,
                                  retain_graph=True)[0].detach().numpy()

    x_np, y_np = pts_sub[:, 0], pts_sub[:, 1]
    r2 = x_np**2 + y_np**2

    # grad g = (2x, 2y)
    grad_g = np.stack([2 * x_np, 2 * y_np], axis=-1)
    grad_g_norm_sq = (grad_g**2).sum(-1)

    # h'(g) = (grad_f . grad_g) / ||grad_g||^2
    dot = (grad_f * grad_g).sum(-1)
    h_prime = dot / np.maximum(grad_g_norm_sq, 1e-8)

    u_np = u_ch.detach().numpy()
    good = grad_g_norm_sq > 0.5

    g_good = r2[good]
    hp_good = h_prime[good]
    u_good = u_np[good]

    sort_idx = np.argsort(g_good)
    g_s = g_good[sort_idx]
    hp_s = hp_good[sort_idx]
    u_s = u_good[sort_idx]

    # Bin
    be = np.linspace(g_s.min(), g_s.max(), n_bins + 1)
    bc, bm_hp, bm_u = [], [], []
    for i in range(n_bins):
        m = (g_s >= be[i]) & (g_s < be[i + 1])
        if m.sum() > 15:
            bc.append((be[i] + be[i + 1]) / 2)
            bm_hp.append(np.median(hp_s[m]))
            bm_u.append(u_s[m].mean())
    bc = np.array(bc)
    bm_hp = np.array(bm_hp)
    bm_u = np.array(bm_u)

    # Integrate h'(g) -> h(g)
    dg = np.diff(bc)
    h_int = np.concatenate([[0], np.cumsum(bm_hp[:-1] * dg)])
    h_int += bm_u[0]

    r2_int = 1 - np.sum((bm_u - h_int)**2) / max(np.sum((bm_u - bm_u.mean())**2), 1e-12)

    # Trig fit: h(g) = a*cos(t(g)) + b*sin(t(g)) + c
    t_bc = 0.4 - 6.0 / (1.0 + bc)
    A = np.column_stack([np.cos(t_bc), np.sin(t_bc), np.ones(len(bc))])
    sol, _, _, _ = lstsq(A, bm_u, rcond=None)
    pred_trig = A @ sol
    r2_trig = 1 - np.sum((bm_u - pred_trig)**2) / max(np.sum((bm_u - bm_u.mean())**2), 1e-12)

    # Theoretical h'(g) from trig fit
    dt_dg = 6.0 / (1.0 + bc)**2
    hp_theory = (-sol[0] * np.sin(t_bc) + sol[1] * np.cos(t_bc)) * dt_dg

    return {
        'bc': bc, 'bm_hp': bm_hp, 'bm_u': bm_u,
        'h_int': h_int, 'r2_int': r2_int,
        'pred_trig': pred_trig, 'sol': sol, 'r2_trig': r2_trig,
        'hp_theory': hp_theory, 't_bc': t_bc,
        'g_good': g_good, 'hp_good': hp_good, 'u_good': u_good,
    }


def plot_outer_function(result, channel=1):
    r = result
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) h'(g)
    ax = axes[0]
    ax.plot(r['bc'], r['bm_hp'], 'b-', lw=2.5, label="h'(g) recovered")
    ax.plot(r['bc'], r['hp_theory'], 'r--', lw=2, label="h'(g) from trig fit")
    ax.set_xlabel('g = x^2 + y^2', fontsize=12)
    ax.set_ylabel("h'(g)", fontsize=12)
    ax.set_title("(a) Outer derivative h'(g)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.15)

    # (b) h(g) trig fit
    ax = axes[1]
    ax.plot(r['bc'], r['bm_u'], 'b-', lw=2.5, label=f'u_{channel} (binned mean)')
    s = r['sol']
    ax.plot(r['bc'], r['pred_trig'], 'r--', lw=2.5,
            label=f"{s[0]:.2f}cos(t) + {s[1]:.2f}sin(t) + {s[2]:.2f}")
    ax.set_xlabel('g = x^2 + y^2', fontsize=12)
    ax.set_ylabel('h(g)', fontsize=12)
    ax.set_title(f"(b) h(g) trig fit, R^2 = {r['r2_trig']:.3f}", fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    # (c) Integrated h(g)
    ax = axes[2]
    ax.plot(r['bc'], r['bm_u'], 'b-', lw=2.5, label=f'u_{channel} (actual)')
    ax.plot(r['bc'], r['h_int'], 'r--', lw=2.5, label="h(g) integrated from h'")
    ax.set_xlabel('g = x^2 + y^2', fontsize=12)
    ax.set_ylabel('h(g)', fontsize=12)
    ax.set_title(f"(c) Integrated h(g), R^2 = {r['r2_int']:.3f}", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"Ikeda: u_{channel} = h(x^2+y^2),  "
        f"h(g) = a*cos(t(g)) + b*sin(t(g)) + c,  "
        f"t(g) = 0.4 - 6/(1+g)",
        fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'ikeda_outer_h_trig.png', dpi=200, bbox_inches='tight')
    fig.savefig(FIGDIR / 'ikeda_outer_h_trig.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR / "ikeda_outer_h_trig.png"}')


# =====================================================================
# 4. Foliation 2D plot: u colored over attractor with r^2 circles
# =====================================================================

def plot_foliation_2d(model, pts_sub, channels=(1, 5)):
    x_data = torch.from_numpy(pts_sub)
    u_vals, _ = extract_inner_functions(model.encoder, x_data)
    r2_data = pts_sub[:, 0]**2 + pts_sub[:, 1]**2
    theta = np.linspace(0, 2 * np.pi, 200)

    with torch.no_grad():
        z_actual = model.encoder(x_data).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) u[channels[0]] over attractor
    ch = channels[0]
    ax = axes[0, 0]
    sc = ax.scatter(pts_sub[:, 0], pts_sub[:, 1], s=0.3, alpha=0.3,
                    c=u_vals[:, ch], cmap='viridis', rasterized=True)
    plt.colorbar(sc, ax=ax, label=f'u_{ch} value', shrink=0.8)
    for rv in [1, 2, 4, 8]:
        ax.plot(np.sqrt(rv) * np.cos(theta), np.sqrt(rv) * np.sin(theta),
                'r-', lw=1, alpha=0.5)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'(a) u_{ch}(x,y) over attractor + r^2 circles')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)

    # (b) r^2 reference
    ax = axes[0, 1]
    sc = ax.scatter(pts_sub[:, 0], pts_sub[:, 1], s=0.3, alpha=0.3,
                    c=r2_data, cmap='viridis', rasterized=True,
                    vmin=np.percentile(r2_data, 1), vmax=np.percentile(r2_data, 99))
    plt.colorbar(sc, ax=ax, label='r^2 = x^2+y^2', shrink=0.8)
    for rv in [1, 2, 4, 8]:
        ax.plot(np.sqrt(rv) * np.cos(theta), np.sqrt(rv) * np.sin(theta),
                'r-', lw=1, alpha=0.5)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('(b) r^2 over attractor (reference)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)

    # (c) u[channels[1]] over attractor
    ch2 = channels[1]
    ax = axes[1, 0]
    sc = ax.scatter(pts_sub[:, 0], pts_sub[:, 1], s=0.3, alpha=0.3,
                    c=u_vals[:, ch2], cmap='viridis', rasterized=True)
    plt.colorbar(sc, ax=ax, label=f'u_{ch2} value', shrink=0.8)
    for rv in [1, 2, 4, 8]:
        ax.plot(np.sqrt(rv) * np.cos(theta), np.sqrt(rv) * np.sin(theta),
                'r-', lw=1, alpha=0.5)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'(c) u_{ch2}(x,y) over attractor + r^2 circles')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)

    # (d) u[channels[0]] vs r^2 scatter
    ax = axes[1, 1]
    sort_idx = np.argsort(r2_data)
    ax.scatter(r2_data, u_vals[:, channels[0]], s=0.3, alpha=0.05,
               c='gray', rasterized=True)
    r2_sorted = r2_data[sort_idx]
    u_sorted = u_vals[sort_idx, channels[0]]
    n_bins = 100
    be = np.linspace(r2_sorted.min(), r2_sorted.max(), n_bins + 1)
    bc, bm = [], []
    for i in range(n_bins):
        m = (r2_sorted >= be[i]) & (r2_sorted < be[i + 1])
        if m.sum() > 5:
            bc.append((be[i] + be[i + 1]) / 2)
            bm.append(u_sorted[m].mean())
    ax.plot(bc, bm, 'g-', lw=2.5, label='binned mean')
    ax.set_xlabel('r^2 = x^2 + y^2'); ax.set_ylabel(f'u_{channels[0]}')
    ax.set_title(f'(d) u_{channels[0]} vs r^2')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.15)

    fig.suptitle('Ikeda: Does the encoder follow the r^2 foliation?', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'ikeda_foliation_2d.png', dpi=200, bbox_inches='tight')
    fig.savefig(FIGDIR / 'ikeda_foliation_2d.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR / "ikeda_foliation_2d.png"}')


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  Ikeda Foliation Analysis")
    print("=" * 70)

    model = load_model()
    pts_sub = load_data()
    print(f"  Data: {pts_sub.shape[0]} points")
    print(f"  x range: [{pts_sub[:,0].min():.3f}, {pts_sub[:,0].max():.3f}]")
    print(f"  y range: [{pts_sub[:,1].min():.3f}, {pts_sub[:,1].max():.3f}]")

    # --- Edge spline analysis ---
    print("\n--- Edge Spline Analysis ---")
    spline_results = plot_edge_splines(model, pts_sub, channels=(1, 5))
    for r in spline_results:
        print(f"  u[{r['channel']}]: x^2={r['x2_coeff']:+.4f}  y^2={r['y2_coeff']:+.4f}  "
              f"ratio={r['ratio']:.3f}  same_sign={r['same_sign']}")

    # --- Lasso on inner functions ---
    print("\n--- Lasso on Inner Functions (degree 3) ---")
    lasso_results, names, M_dict = lasso_inner_functions(
        model, pts_sub, max_degree=3, alphas=(1e-5, 1e-4, 1e-3, 1e-2))
    for j in [1, 5]:
        print(f"\n  u[{j}]:")
        for lr in lasso_results[j]:
            top = ', '.join([f"{n}({c:+.3f})" for n, c in lr['active'][:5]])
            print(f"    alpha={lr['alpha']:.0e}: R^2={lr['r2']:.4f} n={lr['n_active']:>2} "
                  f"x^2={lr['x2']:+.3f} y^2={lr['y2']:+.3f} ratio={lr['ratio']:.3f} [{top}]")

    # --- Gradient identity on u[1] ---
    print("\n--- Gradient Identity: u[1] = h(x^2 + y^2) ---")
    grad_result = gradient_identity(model, pts_sub, channel=1)
    s = grad_result['sol']
    print(f"  h(g) = {s[0]:.3f}*cos(t(g)) + {s[1]:.3f}*sin(t(g)) + {s[2]:.3f}")
    print(f"  Trig fit R^2 = {grad_result['r2_trig']:.4f}")
    print(f"  Integrated R^2 = {grad_result['r2_int']:.4f}")
    plot_outer_function(grad_result, channel=1)

    # --- Foliation 2D plot ---
    print("\n--- Foliation 2D Plot ---")
    plot_foliation_2d(model, pts_sub, channels=(1, 5))

    print("\nDone.")
