"""
Train a [3, 5, 5] PyKAN stable Koopman on Lorenz and run the full
spline-basis + eigenmode analysis.

Run from project root:
    python experiments/lorenz_5d_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from scipy.linalg import logm
from numpy.polynomial import polynomial as P
from sklearn.linear_model import Lasso
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

from data import simulate_lorenz, PairDataset
from models import PyKANAutoEncoder
from training import set_seed
from lorenz_stable import DeepKoopmanCTStable

OUTDIR = Path("results/lorenz_5d")
OUTDIR.mkdir(parents=True, exist_ok=True)
DT = 0.01
MID = 5
LATENT = 5
VAR = ["x", "y", "z"]
TRUE_SUPPORT = {"x", "y", "z", "xy", "xz"}

SYMBOLIC_CANDIDATES = {
    "linear":    (lambda x, a, b: a * x + b, 2),
    "quadratic": (lambda x, a, b, c: a * x**2 + b * x + c, 3),
    "cubic":     (lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, 4),
    "sin":       (lambda x, a, b, c, d: a * np.sin(b * x + c) + d, 4),
    "cos":       (lambda x, a, b, c, d: a * np.cos(b * x + c) + d, 4),
    "tanh":      (lambda x, a, b, c, d: a * np.tanh(b * x + c) + d, 4),
    "exp":       (lambda x, a, b, c: a * np.exp(np.clip(b * x, -10, 10)) + c, 3),
    "abs":       (lambda x, a, b, c: a * np.abs(x - b) + c, 3),
}


def ols_r2(X, y):
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res / max(ss_tot, 1e-12), coeffs


def get_psi(enc, x_data):
    with torch.no_grad():
        enc(x_data)
        return enc.acts[1].numpy()


def get_latent(enc, x_data):
    with torch.no_grad():
        enc(x_data)
        if len(enc.acts) > 2:
            return enc.acts[2].numpy()
        return enc(x_data).numpy()


def main():
    print("=" * 70)
    print(f"  LORENZ [3, {MID}, {LATENT}] STABLE KOOPMAN TEST")
    print("=" * 70)

    # ── Data ──
    set_seed(42)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=DT, burn_in=500, device="cpu")
    train_traj = traj[:48].contiguous()
    val_traj = traj[48:].contiguous()
    print(f"  Data: train={train_traj.shape}, val={val_traj.shape}")

    # ── Train ──
    print(f"\n  Training PyKAN [{3},{MID},{LATENT}] + stable Koopman...")
    set_seed(0)
    ae = PyKANAutoEncoder(3, MID, LATENT, grid=5, k=3)
    traj_full = torch.cat([train_traj, val_traj], dim=0)
    train_dl = DataLoader(PairDataset(traj_full[:48]), batch_size=2048,
                          shuffle=True, drop_last=True)
    val_dl = DataLoader(PairDataset(traj_full[48:]), batch_size=2048)

    model = DeepKoopmanCTStable(ae, LATENT)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    mse_fn = nn.MSELoss()

    for ep in range(1, 201):
        model.train()
        losses = []
        for x_t, x_tp1 in train_dl:
            out = model.forward_one_step(x_t, DT)
            z_true = model.ae.encoder(x_tp1)
            loss = (mse_fn(out["x_tp1_pred"], x_tp1)
                    + mse_fn(out["z_tp1_pred"], z_true)
                    + 0.5 * mse_fn(out["x_t_rec"], x_t))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        if ep == 1 or ep % 40 == 0 or ep == 200:
            model.eval()
            vloss = []
            with torch.no_grad():
                for x_t, x_tp1 in val_dl:
                    out = model.forward_one_step(x_t, DT)
                    z_true = model.ae.encoder(x_tp1)
                    l = (mse_fn(out["x_tp1_pred"], x_tp1)
                         + mse_fn(out["z_tp1_pred"], z_true)
                         + 0.5 * mse_fn(out["x_t_rec"], x_t))
                    vloss.append(l.item())
            print(f"    Ep {ep:3d} | train {np.mean(losses):.5f} | val {np.mean(vloss):.5f}")

    model.eval()
    torch.save({"model_state_dict": model.state_dict(),
                "config": {"mid": MID, "latent": LATENT}},
               OUTDIR / "model_5_5.pt")
    print(f"  Saved model to {OUTDIR / 'model_5_5.pt'}")

    # ── Analysis data ──
    enc = model.ae.encoder
    x_flat = train_traj.reshape(-1, 3)
    idx = np.random.choice(len(x_flat), 8000, replace=False)
    x_data = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    x_np = x_data.numpy()
    psi_data = get_psi(enc, x_data)
    z_data = get_latent(enc, x_data)

    # ══════════════════════════════════════════════════════════════
    # 1. Layer-1 edge functions
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  1. LAYER-1 EDGE FUNCTIONS")
    print(f"{'='*70}")

    x_mean = torch.tensor(x_np.mean(axis=0), dtype=torch.float32)
    with torch.no_grad():
        enc(x_mean.unsqueeze(0))
        baseline = enc.acts[1].numpy().flatten()

    edges = {}
    lin_r2 = np.zeros((MID, 3))
    for i in range(3):
        xi = np.linspace(x_np[:, i].min(), x_np[:, i].max(), 300)
        probe = x_mean.unsqueeze(0).repeat(300, 1)
        probe[:, i] = torch.tensor(xi, dtype=torch.float32)
        with torch.no_grad():
            enc(probe)
            l1 = enc.acts[1].numpy()
        for j in range(MID):
            eo = l1[:, j] - baseline[j]
            edges[(j, i)] = (xi, eo)
            c = P.polyfit(xi, eo, 1)
            pred = P.polyval(xi, c)
            ss = np.sum((eo - pred)**2)
            st = np.sum((eo - eo.mean())**2)
            lin_r2[j, i] = 1 - ss / max(st, 1e-12)

    # Identify symbolic functions
    spline_info = {}
    print(f"\n  {'Node':>6} {'Input':>6} {'Range':>8} {'LinR²':>7} {'Best fit':>10} {'R²':>7}")
    print("  " + "-" * 52)
    for j in range(MID):
        for i in range(3):
            xi, eo = edges[(j, i)]
            er = eo.max() - eo.min()
            best_name, best_r2 = "unknown", -999
            for cname, (cfn, nparams) in SYMBOLIC_CANDIDATES.items():
                try:
                    p0 = [1.0] * nparams
                    popt, _ = curve_fit(cfn, xi, eo, p0=p0, maxfev=5000)
                    pred = cfn(xi, *popt)
                    ss = np.sum((eo - pred)**2)
                    st = np.sum((eo - eo.mean())**2)
                    r2 = 1 - ss / max(st, 1e-12)
                    if r2 > best_r2:
                        best_r2, best_name = r2, cname
                except Exception:
                    pass
            spline_info[(j, i)] = (best_name, best_r2)
            nl = "NL" if lin_r2[j, i] < 0.95 else ""
            print(f"  psi{j}  {VAR[i]:>6}  {er:>8.3f}  {lin_r2[j,i]:>7.3f}  "
                  f"{best_name:>10}  {best_r2:>7.3f}  {nl}")

    # ══════════════════════════════════════════════════════════════
    # 2. K_psi eigendecomposition
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  2. K_psi EIGENDECOMPOSITION (5x5)")
    print(f"{'='*70}")

    pair_idx = np.random.choice(len(x_flat) - 1, 50000, replace=False)
    x_t = torch.tensor(x_flat[pair_idx].numpy(), dtype=torch.float32)
    x_tp1 = torch.tensor(x_flat[pair_idx + 1].numpy(), dtype=torch.float32)
    psi_t = get_psi(enc, x_t)
    psi_tp1 = get_psi(enc, x_tp1)
    psi_t_wc = np.column_stack([np.ones(len(psi_t)), psi_t])
    K_full, _, _, _ = np.linalg.lstsq(psi_t_wc, psi_tp1, rcond=None)
    K_psi = K_full[1:, :]

    eigvals, V = np.linalg.eig(K_psi)
    V_inv = np.linalg.inv(V)
    mu = np.log(eigvals.astype(complex)) / DT
    order = np.argsort(-np.abs(mu.imag))

    print(f"\n  K_psi =")
    for i in range(MID):
        print(f"    [{' '.join(f'{K_psi[i,j]:>8.4f}' for j in range(MID))}]")

    print(f"\n  {'Mode':>5} {'Re(λ)':>8} {'Im(λ)':>8} {'|λ|':>8} "
          f"{'Re(μ)':>8} {'Im(μ)':>8} {'freq Hz':>8}")
    print("  " + "-" * 55)
    for rank, oidx in enumerate(order):
        lam = eigvals[oidx]
        m = mu[oidx]
        print(f"  M{rank:>3} {lam.real:>8.4f} {lam.imag:>8.4f} {abs(lam):>8.4f} "
              f"{m.real:>8.2f} {m.imag:>8.2f} {m.imag/(2*np.pi):>8.3f}")

    # Eigenvector participation
    print(f"\n  Eigenvector participation |V[psi_j, mode]| (normalized):")
    header = "        " + "  ".join(f"M{r}" for r in range(MID))
    print(f"  {header}")
    for j in range(MID):
        row = f"  psi{j}:  "
        for rank in range(MID):
            v = V[:, order[rank]]
            v_n = np.abs(v) / np.max(np.abs(v))
            row += f" {v_n[j]:.3f}"
        print(row)

    # Koopman linearity R²
    print(f"\n  Koopman linearity (psi_{{t+1}} = K @ psi_t):")
    for j in range(MID):
        r2, _ = ols_r2(psi_t_wc, psi_tp1[:, j])
        print(f"    psi{j}: R² = {r2:.6f}")

    # ══════════════════════════════════════════════════════════════
    # 3. Subspace equivalence
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  3. SUBSPACE EQUIVALENCE (5 psi vs 5 poly)")
    print(f"{'='*70}")

    x, y, z = x_np[:, 0], x_np[:, 1], x_np[:, 2]
    poly_lorenz = np.column_stack([x, y, z, x*y, x*z])
    psi_wc = np.column_stack([np.ones(len(psi_data)), psi_data])
    poly_wc = np.column_stack([np.ones(len(x)), poly_lorenz])

    print(f"\n  Poly → Psi: can 5 psi's explain each polynomial?")
    for k, name in enumerate(["x", "y", "z", "xy", "xz"]):
        r2, _ = ols_r2(psi_wc, poly_lorenz[:, k])
        print(f"    {name:>4}: R² = {r2:.6f}")

    print(f"\n  Psi → Poly: can {{x,y,z,xy,xz}} explain each psi?")
    for j in range(MID):
        r2, _ = ols_r2(poly_wc, psi_data[:, j])
        print(f"    psi{j}: R² = {r2:.6f}")

    # Principal angles
    psi_c = psi_data - psi_data.mean(axis=0)
    poly_c = poly_lorenz - poly_lorenz.mean(axis=0)
    Q_psi, _ = np.linalg.qr(psi_c)
    Q_poly, _ = np.linalg.qr(poly_c)
    _, sigma, _ = np.linalg.svd(Q_poly.T @ Q_psi, full_matrices=False)
    angles = np.arccos(np.clip(sigma, -1, 1)) * 180 / np.pi

    print(f"\n  Principal angles (5 psi vs 5 poly):")
    for i, a in enumerate(angles):
        print(f"    angle_{i}: {a:.2f}°")
    print(f"  Max angle: {angles.max():.2f}°")

    # ══════════════════════════════════════════════════════════════
    # 4. Polynomial Koopman comparison
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  4. KOOPMAN LINEARITY: SPLINE vs POLYNOMIAL BASIS")
    print(f"{'='*70}")

    xt_np = x_flat[pair_idx].numpy()
    xtp1_np = x_flat[pair_idx + 1].numpy()
    poly_t = np.column_stack([
        np.ones(len(xt_np)),
        xt_np[:, 0], xt_np[:, 1], xt_np[:, 2],
        xt_np[:, 0]*xt_np[:, 1], xt_np[:, 0]*xt_np[:, 2]])
    poly_tp1 = np.column_stack([
        xtp1_np[:, 0], xtp1_np[:, 1], xtp1_np[:, 2],
        xtp1_np[:, 0]*xtp1_np[:, 1], xtp1_np[:, 0]*xtp1_np[:, 2]])

    print(f"\n  {'Basis':>15} {'dim':>4}", end="")
    for name in ["x", "y", "z", "xy", "xz"]:
        print(f"  {name:>8}", end="")
    print(f"  {'min':>8}  {'mean':>8}")

    # Polynomial Koopman
    poly_r2s = []
    print(f"  {'poly{x,y,z,xy,xz}':>15} {5:>4}", end="")
    for k in range(5):
        r2, _ = ols_r2(poly_t, poly_tp1[:, k])
        poly_r2s.append(r2)
        print(f"  {r2:>8.5f}", end="")
    print(f"  {min(poly_r2s):>8.5f}  {np.mean(poly_r2s):>8.5f}")

    # Spline Koopman
    psi_r2s = []
    print(f"  {'spline(psi)':>15} {MID:>4}", end="")
    for j in range(MID):
        r2, _ = ols_r2(psi_t_wc, psi_tp1[:, j])
        psi_r2s.append(r2)
        print(f"  {r2:>8.5f}", end="")
    print(f"  {min(psi_r2s):>8.5f}  {np.mean(psi_r2s):>8.5f}")

    # ══════════════════════════════════════════════════════════════
    # 5. Plots
    # ══════════════════════════════════════════════════════════════
    print(f"\n  Generating plots...")

    # Figure 1: Edge functions (5 rows x 3 cols)
    fig1, axes1 = plt.subplots(MID, 3, figsize=(12, 2.5 * MID))
    for j in range(MID):
        for i in range(3):
            ax = axes1[j, i]
            xi, eo = edges[(j, i)]
            ax.plot(xi, eo, "b-", lw=2)
            ax.axhline(0, color="gray", ls=":", alpha=0.3)
            r2 = lin_r2[j, i]
            tag = "NL" if r2 < 0.95 else "lin"
            color = "red" if r2 < 0.95 else "black"
            ax.set_title(f"$\\psi_{j}$({VAR[i]}) [{tag}, linR²={r2:.2f}]",
                        fontsize=9, color=color)
            ax.tick_params(labelsize=6)
    fig1.suptitle(f"Layer-1 Edge Functions [3, {MID}, {LATENT}]", fontsize=13)
    fig1.tight_layout()
    fig1.savefig(OUTDIR / "edges.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Attractor colored by each psi
    fig2, axes2 = plt.subplots(1, MID, figsize=(4 * MID, 4),
                                subplot_kw={"projection": "3d"})
    for j in range(MID):
        ax = axes2[j]
        c = psi_data[:, j]
        vmax = np.percentile(np.abs(c), 95)
        ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                  c=c, cmap="coolwarm", s=0.3, alpha=0.3,
                  vmin=-vmax, vmax=vmax)
        ax.set_title(f"$\\psi_{j}$", fontsize=11)
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=5)
    fig2.suptitle(f"Attractor colored by Layer-1 activations", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(OUTDIR / "attractor_psi.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: Eigenmode hero figure
    modal_data = psi_data @ V_inv.T
    traj_seg = traj[0, :2000].numpy()
    x_seg = torch.tensor(traj_seg, dtype=torch.float32)
    psi_seg = get_psi(enc, x_seg)
    modal_seg = psi_seg @ V_inv.T
    t_seg = np.arange(len(traj_seg)) * DT

    fig3 = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, MID, figure=fig3, hspace=0.4, wspace=0.3)

    # Row 0: Participation heatmap + eigenvalue spectrum
    ax_heat = fig3.add_subplot(gs[0, :MID-1])
    participation = np.zeros((MID, MID))
    mode_labels = []
    for rank, oidx in enumerate(order):
        v = V[:, oidx]
        participation[:, rank] = np.abs(v) / np.max(np.abs(v))
        freq = mu[oidx].imag / (2 * np.pi)
        mode_labels.append(f"M{rank}\n{freq:+.1f}Hz\nRe={mu[oidx].real:.1f}")
    im = ax_heat.imshow(participation, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax_heat.set_yticks(range(MID))
    ax_heat.set_yticklabels([f"$\\psi_{j}$" for j in range(MID)])
    ax_heat.set_xticks(range(MID))
    ax_heat.set_xticklabels(mode_labels, fontsize=8)
    ax_heat.set_title("Eigenvector Participation", fontsize=11)
    for i in range(MID):
        for j in range(MID):
            v = participation[i, j]
            if v > 0.3:
                ax_heat.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if v > 0.6 else "black")
    plt.colorbar(im, ax=ax_heat, shrink=0.8)

    ax_spec = fig3.add_subplot(gs[0, MID-1])
    theta = np.linspace(0, 2*np.pi, 100)
    ax_spec.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.2)
    colors_m = plt.cm.tab10(np.linspace(0, 1, MID))
    for rank, oidx in enumerate(order):
        lam = eigvals[oidx]
        ax_spec.plot(lam.real, lam.imag, "o", color=colors_m[rank], ms=10)
        freq = mu[oidx].imag / (2*np.pi)
        ax_spec.annotate(f"M{rank}\n{freq:+.1f}Hz",
                        (lam.real, lam.imag), fontsize=6, ha="left")
    ax_spec.set_xlim(0.9, 1.01)
    ax_spec.set_aspect("equal")
    ax_spec.set_title("Eigenvalues", fontsize=10)
    ax_spec.grid(True, alpha=0.3)

    # Row 1: Attractor colored by each mode
    for rank in range(MID):
        ax = fig3.add_subplot(gs[1, rank], projection="3d")
        oidx = order[rank]
        c = modal_data[:, oidx].real
        vmax = np.percentile(np.abs(c), 95)
        ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                  c=c, cmap="coolwarm", s=0.3, alpha=0.3,
                  vmin=-vmax, vmax=vmax)
        freq = mu[oidx].imag / (2*np.pi)
        ax.set_title(f"Mode {rank}: {freq:+.1f} Hz", fontsize=9)
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=5)

    # Row 2: Time series
    for rank in range(MID):
        ax = fig3.add_subplot(gs[2, rank])
        oidx = order[rank]
        mk = modal_seg[:, oidx]
        freq = mu[oidx].imag / (2*np.pi)
        ax.plot(t_seg, mk.real, "b-", lw=0.8, label="Re")
        if abs(mu[oidx].imag) > 0.1:
            ax.plot(t_seg, mk.imag, "r-", lw=0.8, alpha=0.5, label="Im")
        ax.set_title(f"M{rank} ({freq:+.1f} Hz)", fontsize=9)
        ax.set_xlim(0, 5)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    fig3.suptitle(f"Eigenmode Structure [{3},{MID},{LATENT}] Stable Koopman",
                  fontsize=14, y=1.01)
    fig3.savefig(OUTDIR / "eigenmode_structure.png", dpi=200, bbox_inches="tight")
    fig3.savefig(OUTDIR / "eigenmode_structure.pdf", bbox_inches="tight")
    plt.close(fig3)

    # Figure 4: Layer-2 edges (5 latent x 5 psi)
    fig4, axes4 = plt.subplots(LATENT, MID, figsize=(3 * MID, 3 * LATENT))
    for k in range(LATENT):
        for j in range(MID):
            ax = axes4[k, j]
            si = np.argsort(psi_data[:, j])
            ps = psi_data[si, j]
            zs = z_data[si, k]
            be = np.linspace(ps.min(), ps.max(), 40)
            bc, bm = [], []
            for b in range(len(be)-1):
                m = (ps >= be[b]) & (ps < be[b+1])
                if m.sum() > 5:
                    bc.append((be[b]+be[b+1])/2)
                    bm.append(np.median(zs[m]))
            if len(bc) > 2:
                bc_a, bm_a = np.array(bc), np.array(bm)
                ax.plot(bc_a, bm_a, "b-", lw=2)
                c = P.polyfit(bc_a, bm_a, 1)
                pred = P.polyval(bc_a, c)
                ss = np.sum((bm_a-pred)**2)
                st = np.sum((bm_a-bm_a.mean())**2)
                r2 = 1 - ss/max(st, 1e-12)
                color = "red" if r2 < 0.9 else "black"
                ax.set_title(f"z[{k}]←ψ{j} R²={r2:.2f}", fontsize=8, color=color)
            ax.tick_params(labelsize=5)
    fig4.suptitle("Layer-2 Edge Functions (red = nonlinear)", fontsize=12)
    fig4.tight_layout()
    fig4.savefig(OUTDIR / "layer2_edges.png", dpi=120, bbox_inches="tight")
    plt.close(fig4)

    print(f"\n  Saved all plots to {OUTDIR}/")

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Architecture: PyKAN [{3},{MID},{LATENT}] + stable A=K-L^TL")
    print(f"  Koopman linearity (psi basis): min R²={min(psi_r2s):.6f}")
    print(f"  Koopman linearity (poly basis): min R²={min(poly_r2s):.6f}")
    print(f"  Principal angles: {', '.join(f'{a:.1f}°' for a in angles)}")
    print(f"  Max angle: {angles.max():.1f}°")

    n_nl = sum(1 for j in range(MID) for i in range(3) if lin_r2[j, i] < 0.95)
    print(f"  Nonlinear edges: {n_nl}/{MID*3}")
    print(f"\n  Eigenvalues:")
    for rank, oidx in enumerate(order):
        m = mu[oidx]
        print(f"    M{rank}: μ = {m.real:+.2f}{m.imag:+.2f}j  "
              f"freq={m.imag/(2*np.pi):+.3f} Hz  |λ|={abs(eigvals[oidx]):.4f}")


if __name__ == "__main__":
    main()
