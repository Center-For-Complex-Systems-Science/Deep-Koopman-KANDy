"""
KS inner-Lasso basis test: physical monomials vs. Fourier basis.

Question (post-hoc, no retraining): on the trained wide KS model, does swapping
the inner-Lasso dictionary from per-grid-point monomials to a translation-
invariant Fourier basis (i) sparsify the inner g, and (ii) make the recovered
outer function h NON-affine?

Method. For a linear coordinate change v = R u (R real-orthonormal DFT; R = I is
the physical basis), the observable's gradient transforms as grad_v f = R grad_u f
(R^{-T} = R for orthonormal R). We compute grad_u f by autograd through the
encoder (which takes the physical state), transform it, fit the degree-2 inner
Lasso in v-coordinates, and run the same gradient-identity outer recovery
h'(g) = grad_f . grad_g / ||grad_g||^2.  Identical pipeline in both bases.

Run:  python experiments/ks_fourier_basis_test.py
"""
import os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import KANAutoEncoder
from ks_stable import DeepKoopmanCTStable          # main() is __main__-guarded
from manifold_analysis import (compute_network_gradient_nd,
                                recover_inner_function_nd,
                                eval_polynomial_nd, grad_polynomial_nd,
                                recover_outer_function_from_raw)

N = 64
CKPT = os.path.join(_ROOT, "results/ks_stable_wide/ks_stable_model.pt")
DATA = os.path.join(_ROOT, "results/ks_stable/ks_data.pt")
N_SAMPLE = 5000
N_DIMS = 8
MAXDEG = 2
ALPHA = 1e-4
HDEG = 4


def real_dft_matrix(n):
    """n x n real ORTHONORMAL DFT: rows are {const, cos_k, sin_k, ..., Nyquist}."""
    j = np.arange(n)
    rows, names = [], []
    rows.append(np.ones(n) / np.sqrt(n)); names.append("c0")
    for k in range(1, n // 2):
        rows.append(np.sqrt(2.0 / n) * np.cos(2 * np.pi * k * j / n)); names.append(f"c{k}")
        rows.append(np.sqrt(2.0 / n) * np.sin(2 * np.pi * k * j / n)); names.append(f"s{k}")
    rows.append(np.cos(np.pi * j) / np.sqrt(n)); names.append(f"c{n//2}")  # Nyquist
    R = np.stack(rows)
    return R, names


def r2(y, yhat):
    ss = np.sum((y - yhat) ** 2)
    tot = max(np.sum((y - y.mean()) ** 2), 1e-12)
    return 1.0 - ss / tot


def decompose(model, x_data, R, names, k):
    """Degree-2 inner Lasso + gradient-identity outer recovery in coords v = R u."""
    x_np = x_data.numpy()
    grad_f, f_vals = compute_network_gradient_nd(model, x_data, k, "latent")   # (N,64),(N,)
    v = x_np @ R.T                    # v_n = R u_n
    grad_f_v = grad_f @ R.T           # grad_v f = R grad_u f

    inner = recover_inner_function_nd(v, names, f_vals, max_degree=MAXDEG, alpha=ALPHA)
    g = eval_polynomial_nd(v, inner["coeffs"], inner["powers"])
    gg = grad_polynomial_nd(v, inner["coeffs"], inner["powers"])
    dot = np.sum(grad_f_v * gg, axis=1)
    nsq = np.sum(gg ** 2, axis=1)
    outer = recover_outer_function_from_raw(dot, nsq, g, f_vals, poly_degree=HDEG)

    # affine and polynomial-h reconstructions on all samples
    ca = outer["affine_coeffs"]
    r2_aff = r2(f_vals, ca[0] * g + ca[1])
    gsh = g - outer["g_center"]
    Vp = np.vander(gsh, N=len(outer["h_coeffs"]), increasing=True)
    r2_poly = r2(f_vals, Vp @ outer["h_coeffs"])
    return {
        "n_active": len(inner["active_names"]),
        "r2_g": inner["r2"],
        "r2_aff": r2_aff,
        "r2_poly": r2_poly,
        "gain": r2_poly - r2_aff,
        "used_affine": bool(outer["used_affine"]),
        "top": inner["active_names"][:8],
    }


def main():
    print("Loading wide KS model:", CKPT, flush=True)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    mid = ckpt.get("config", {}).get("mid_size", 96)
    ae = KANAutoEncoder(N, mid, 128, grid_range_l1=[-5, 5], grid_range_l2=[-10, 10])
    model = DeepKoopmanCTStable(ae, 128)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    traj = torch.load(DATA, map_location="cpu", weights_only=True)
    flat = traj.reshape(-1, N)
    torch.manual_seed(0)
    idx = torch.randperm(flat.shape[0])[:N_SAMPLE]
    x_data = flat[idx].float()
    print(f"data: {tuple(traj.shape)} -> {tuple(x_data.shape)} samples", flush=True)

    R_fourier, fnames = real_dft_matrix(N)
    assert np.allclose(R_fourier @ R_fourier.T, np.eye(N), atol=1e-8), "DFT not orthonormal"
    R_phys, pnames = np.eye(N), [f"u_{j}" for j in range(N)]

    with torch.no_grad():
        zvar = model.ae.encoder(x_data).numpy().var(0)
    dims = [int(d) for d in np.argsort(zvar)[::-1][:N_DIMS]]

    print("\n" + "=" * 96)
    print(f"{'dim':>4} {'basis':>8} {'#atoms':>7} {'R2(g)':>8} {'R2(affine)':>11} "
          f"{'R2(poly h)':>11} {'gain':>8} {'affine?':>8}")
    print("=" * 96)
    agg = {"phys": [], "fourier": []}
    for k in dims:
        for tag, (R, names) in (("phys", (R_phys, pnames)),
                                ("fourier", (R_fourier, fnames))):
            d = decompose(model, x_data, R, names, k)
            agg[tag].append(d)
            print(f"{k:>4} {tag:>8} {d['n_active']:>7} {d['r2_g']:>8.3f} "
                  f"{d['r2_aff']:>11.4f} {d['r2_poly']:>11.4f} {d['gain']:>8.4f} "
                  f"{str(d['used_affine']):>8}")
        print("-" * 96)

    print("\nSUMMARY (median over %d dims):" % N_DIMS)
    for tag in ("phys", "fourier"):
        A = agg[tag]
        med = lambda key: float(np.median([a[key] for a in A]))
        n_nonaffine = sum(1 for a in A if not a["used_affine"])
        print(f"  {tag:>8}:  #atoms={med('n_active'):.0f}  R2(h.g)~={med('r2_poly'):.3f}  "
              f"median gain={med('gain'):.4f}  non-affine dims={n_nonaffine}/{N_DIMS}")
    print("\nTop Fourier atoms for the first analyzed dim:", agg["fourier"][0]["top"])


if __name__ == "__main__":
    main()
