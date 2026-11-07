"""Discovered-dictionary ablation: hand T to EDMD, SINDy and KANDy.

This experiment and table is not in the paper. However, we leave it in this
repository for completeness. Tl;dr Deep-Koopman-KANDy leans the dictionary;
in this script we pass the dictionary to the methods who use it.
Reviewer hdc8 (Q4) asks what the *discovered* dictionary T does when it is used
for prediction rather than interpretation.  In the paper T is post-hoc: the
generator K is fit once in Phase 1 and never re-fit, and every reported
prediction is the autoregressive rollout D o exp(K dt) o E.  This script tests
the other branch -- T as the input library of a downstream method -- which
requires re-fitting, because T collects the *inner* supports and drops the
outer maps h_k, so span(T) is not the learned latent span.

Arms per system (all on the same data/metrics as benchmark_prior_art.py):

  ours          full Deep-Koopman-KANDy rollout (reference; nothing re-fit)
  edmd_disc     EDMD, K re-fit by least squares on span(T), re-lifting rollout
  edmd_poly     EDMD on the full degree-D monomial library      [control]
  edmd_oracle   EDMD on the known target dictionary             [lorenz only]
  edmdlin_*     same fits, but propagated inside the lift (Psi_{n+1} = K Psi_n)
                with no re-lifting -- the actual finite-dimensional Koopman
                model, and the only arm that tests dictionary invariance
  sindy_disc    SINDy (STLSQ) with T as its library
  sindy_poly    SINDy (STLSQ) on the full degree-D library      [control]
  sindy_oracle  SINDy on the known target dictionary            [lorenz only]
  kandy_disc    single KAN layer over the T atoms
  kandy_state   single KAN layer over the raw state             [native KANDy]

The controls are the point: "EDMD on T scores X" is uninterpretable without
the generic library of the same degree to compare against.  What the table
tests is whether the encoder's discovered atoms are a *better input* than the
untargeted library -- the claim the Discussion makes when it positions the
method as supplying candidate structure to SINDy/KANDy/EDMD.

Notes on faithfulness:
  * SINDy is sequentially-thresholded least squares (STLSQ) run on the
    specified library.  It is implemented in this file rather than imported:
    pysindy 1.7.5 raises ValueError against the installed scikit-learn 1.8.0 at
    every entry point (optimizer base unpacks a 5-tuple from _preprocess_data,
    whose arity changed).  Same algorithm
    (continuous-time with analytic xdot for Lorenz; discrete-time x_{n+1} for
    the maps).  The STLSQ threshold is chosen per arm on a held-out slice of
    the training data, so no arm is handicapped by a bad threshold.
  * Atom selection follows the paper's convention (manifold_analysis.
    analyze_ideal): keep atoms whose |coeff| is at least 1% of the largest in
    that coordinate.  The alternative scale-aware rule (|coeff| * std(atom)) is
    available via --rule scaled and is reported alongside in
    dictionaries.json, because the two disagree sharply: on Lorenz the raw rule
    keeps a median of 7 atoms per coordinate at R^2(g) = 0.980, the scale-aware
    rule keeps 18 at 0.995, from the same fit.  The extra atoms buy ~1.5% of
    variance, so the sparse description is a fidelity choice, not an artifact --
    but the union under the scale-aware rule saturates the degree-3 library and
    would make the discovered and generic arms the same experiment.
  * EDMD reconstructs the state by least squares from the lift.  The edmd_*
    arms re-lift at every rollout step, which is how EDMD is used in practice;
    the edmdlin_* arms do not.  The distinction is load-bearing on the cat map:
    its one-step map is exactly degree-3 in the cos/sin embedding, so a
    re-lifting rollout iterates an exact nonlinear map and looks closed, while
    the lift-space rollout exposes the absent invariance.
  * On the maps, SINDy and EDMD given the same library are the same estimator
    up to STLSQ thresholding (EDMD's propagator composed with its linear state
    readout is the regression Psi(X) -> Y that discrete SINDy fits), so those
    rows agree by construction.  They differ on Lorenz, where SINDy fits xdot
    and EDMD fits the dt-flow map.

Usage:
    python experiments/discovered_dictionary_ablation.py --smoke
    python experiments/discovered_dictionary_ablation.py \
        --systems lorenz ikeda standard_map cat_map --seeds 3
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_prior_art import (                                  # noqa: E402
    SYSTEMS, get_data, horizon_crossings, make_pairs, n_params,
    nrmse_curve, one_step_errors, ours_predictors, set_seed,
    train_ae_koopman, OursModel,
)
from manifold_analysis import recover_inner_function_nd            # noqa: E402
from data import lorenz_rhs                                        # noqa: E402
from kan import KANLinear                                          # noqa: E402

OUTDIR = os.path.join(_ROOT, "results", "discovered_dictionary_ablation")

# Readout degree per system.  Lorenz uses degree 3 rather than the degree-2
# target: at degree 2 the library holds only 9 atoms, the union saturates it,
# and the discovered and generic arms become the same experiment.  Degree 3
# (19 atoms) is the setting where selection can actually be wrong.  The two
# maps are in the cos/sin embedding, where low-order monomials *are* the
# Fourier atoms, and run at degree 3 (34 atoms) for the same reason.  Ikeda
# keeps the paper's degree 3, which in two variables is only 9 atoms: the
# discovered union saturates it, so Ikeda has no disc-vs-poly contrast by
# construction.  That is the paper's own "dense but structured" finding for
# Ikeda, not a defect of this experiment, and it is reported as such.
DEGREE = {"lorenz": 3, "ikeda": 3, "standard_map": 3, "cat_map": 3}

# Known target dictionary, where one exists (Lorenz only).
ORACLE = {"lorenz": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]}

VAR_NAMES = {
    "lorenz": ["x", "y", "z"],
    "ikeda": ["x", "y"],
    "standard_map": ["c1", "s1", "c2", "s2"],
    "cat_map": ["c1", "s1", "c2", "s2"],
}


# ===========================================================================
# Monomial features
# ===========================================================================
def monomial_powers(d, degree, include_const=False):
    """All power tuples with total degree in [1, degree] (or [0, degree]).

    Enumerated as multisets of variable indices rather than as the full product
    grid: the grid is (degree+1)**d, which is unusable for a 64-dimensional
    state such as KS.
    """
    out = []
    lo = 0 if include_const else 1
    for deg in range(lo, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), deg):
            p = [0] * d
            for i in combo:
                p[i] += 1
            out.append(tuple(p))
    return sorted(out, key=lambda p: (sum(p), p))


def power_name(p, names):
    parts = [f"{n}^{e}" if e > 1 else n for n, e in zip(names, p) if e]
    return "1" if not parts else "*".join(parts)


class FeatureMap:
    """Monomial lift with a constant column and per-column scaling.

    Column scaling matters: degree-3 monomials of the Lorenz state span five
    orders of magnitude, and an unscaled least-squares solve is ill-conditioned
    enough to change the answer.
    """

    def __init__(self, powers, X_ref):
        self.powers = [tuple(p) for p in powers]
        self.has_const = any(sum(p) == 0 for p in self.powers)
        if not self.has_const:
            self.powers = [tuple(0 for _ in self.powers[0])] + self.powers
        raw = self._raw(X_ref)
        s = raw.std(axis=0)
        s[s < 1e-12] = 1.0
        self.scale = s

    def _raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        cols = [np.prod(X ** np.asarray(p, dtype=np.float64), axis=1)
                for p in self.powers]
        return np.stack(cols, axis=1)

    def __call__(self, X):
        return self._raw(X) / self.scale

    def __len__(self):
        return len(self.powers)

    @property
    def n_atoms(self):
        """Atom count excluding the constant column."""
        return sum(1 for p in self.powers if sum(p) > 0)


def _lstsq(A, B, ridge=1e-10):
    """Ridge-stabilised least squares; A (n,p), B (n,m) -> (p,m)."""
    p = A.shape[1]
    G = A.T @ A + ridge * np.eye(p)
    return np.linalg.solve(G, A.T @ B)


# ===========================================================================
# Phase 2 readout: extract the discovered dictionary T from the encoder
# ===========================================================================
def discover_dictionary(model, X, system, degree, alpha=1e-5, rel_thresh=0.01,
                        n_samples=4000, device="cpu", rule="raw"):
    """Union of the per-coordinate inner-function supports (Algorithm 1, Ph. 2-3).

    Returns (powers, info).  Atoms are kept when their weight is at least
    `rel_thresh` of the largest weight in that latent coordinate, where the
    weight is |coeff| under the paper's convention (rule="raw") or
    |coeff| * std(atom) under the scale-aware alternative (rule="scaled").
    Both unions are reported in the returned info.
    """
    names = VAR_NAMES[system]
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False)
    Xs = np.asarray(X[idx], dtype=np.float64)

    model.eval()
    with torch.no_grad():
        Z = model.ae.encoder(
            torch.as_tensor(Xs, dtype=torch.float32, device=device)
        ).cpu().numpy()

    unions = {"raw": {}, "scaled": {}}
    per_coord, r2_restricted = [], {"raw": [], "scaled": []}
    for k in range(Z.shape[1]):
        f = Z[:, k].astype(np.float64)
        inner = recover_inner_function_nd(Xs, names, f, max_degree=degree,
                                          alpha=alpha)
        coeffs = np.asarray(inner["coeffs"], dtype=np.float64)
        powers = [tuple(p) for p in inner["powers"]]
        atom_std = np.array([
            np.std(np.prod(Xs ** np.asarray(p, dtype=np.float64), axis=1))
            if sum(p) > 0 else 0.0
            for p in powers
        ])
        w = {"raw": np.abs(coeffs).copy(),
             "scaled": np.abs(coeffs) * atom_std}
        for r in w:                                  # never select the constant
            w[r][[i for i, p in enumerate(powers) if sum(p) == 0]] = 0.0

        entry = {"coord": k, "r2_g": float(inner["r2"])}
        for r in ("raw", "scaled"):
            if w[r].max() <= 0:
                entry[f"n_atoms_{r}"] = 0
                continue
            keep = np.where(w[r] >= rel_thresh * w[r].max())[0]
            for i in keep:
                unions[r][powers[i]] = max(unions[r].get(powers[i], 0.0),
                                           float(w[r][i] / w[r].max()))
            entry[f"n_atoms_{r}"] = len(keep)
            if r == rule:
                entry["atoms"] = [power_name(powers[i], names) for i in keep]
            # R^2 of the thresholded support, refit by OLS (fair to each rule)
            A = np.concatenate([
                np.stack([np.prod(Xs ** np.asarray(powers[i], dtype=np.float64),
                                  axis=1) for i in keep], axis=1),
                np.ones((Xs.shape[0], 1))], axis=1)
            beta, *_ = np.linalg.lstsq(A, f, rcond=None)
            ss = np.sum((f - A @ beta) ** 2)
            r2_restricted[r].append(
                1.0 - ss / max(np.sum((f - f.mean()) ** 2), 1e-12))
        entry["n_atoms"] = entry[f"n_atoms_{rule}"]
        per_coord.append(entry)

    union = unions[rule]
    ordered = sorted(union.keys(), key=lambda p: (-union[p], sum(p), p))
    counts = [c["n_atoms"] for c in per_coord]
    info = {
        "system": system, "degree": degree, "rule": rule, "alpha": alpha,
        "rel_thresh": rel_thresh,
        "n_union": len(ordered),
        "n_union_raw": len(unions["raw"]),
        "n_union_scaled": len(unions["scaled"]),
        "n_full_library": len(monomial_powers(Xs.shape[1], degree)),
        "atoms": [power_name(p, names) for p in ordered],
        "atoms_scaled": sorted(power_name(p, names) for p in unions["scaled"]),
        "per_coord_atoms_median": float(np.median(counts)) if counts else 0.0,
        "per_coord_atoms_min": int(np.min(counts)) if counts else 0,
        "per_coord_atoms_max": int(np.max(counts)) if counts else 0,
        "r2_g_median": float(np.median([c["r2_g"] for c in per_coord])),
        "r2_support_median_raw": float(np.median(r2_restricted["raw"])),
        "r2_support_median_scaled": float(np.median(r2_restricted["scaled"])),
        "per_coord": per_coord,
    }
    if system in ORACLE:
        tgt = {power_name(tuple(p), names) for p in ORACLE[system]}
        got = set(info["atoms"])
        info["target_recall"] = len(tgt & got) / len(tgt)
        info["target_jaccard"] = len(tgt & got) / max(len(tgt | got), 1)
    return ordered, info


# ===========================================================================
# Downstream method 1: EDMD with a given library (K re-fit by least squares)
# ===========================================================================
def make_edmd(Xtr, Ytr, powers, Xref, relift=True):
    """EDMD on a given library.

    relift=True re-lifts the decoded state at every rollout step, which is how
    EDMD is used in practice.  relift=False propagates inside the lift,
    Psi_{n+1} = K Psi_n, which is the actual finite-dimensional Koopman model
    and the only variant that tests dictionary *invariance*: with re-lifting one
    iterates an exact nonlinear map whenever the one-step map lies in the
    library span (the cat map in the cos/sin embedding is exactly this case),
    so a re-lifting rollout can look closed when no finite closure exists.
    """
    t0 = time.time()
    fm = FeatureMap(powers, Xref)
    PX, PY = fm(Xtr), fm(Ytr)
    K = _lstsq(PX, PY)                 # lift-space propagator
    B = _lstsq(PX, np.asarray(Xtr, dtype=np.float64))   # lift -> state readout
    wall = time.time() - t0

    def onestep_np(X):
        return fm(X) @ K @ B

    def _guard(a):
        if not np.all(np.isfinite(a)):
            a = np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(a, -1e6, 1e6)

    def rollout_relift(x0, n_steps):
        x = np.asarray(x0, dtype=np.float64)[None, :]
        out = [x[0].copy()]
        for _ in range(n_steps):
            x = _guard(fm(x) @ K @ B)
            out.append(x[0].copy())
        return np.stack(out, axis=0)

    def rollout_lift(x0, n_steps):
        psi = fm(np.asarray(x0, dtype=np.float64)[None, :])
        out = [np.asarray(x0, dtype=np.float64).copy()]
        for _ in range(n_steps):
            psi = _guard(psi @ K)
            out.append(_guard(psi @ B)[0].copy())
        return np.stack(out, axis=0)

    ro = rollout_relift if relift else rollout_lift
    return onestep_np, ro, K.size + B.size, wall, fm.n_atoms


# ===========================================================================
# Downstream method 2: SINDy (pysindy STLSQ) with a given library
# ===========================================================================
def _rk4(rhs, x, dt):
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def stlsq(Theta, Y, threshold, alpha=1e-5, max_iter=20):
    """Sequentially-thresholded least squares (Brunton et al. 2016).

    Implemented here rather than imported: pysindy 1.7.5 is incompatible with
    the installed scikit-learn (its optimizer base unpacks a 5-tuple from
    sklearn's _preprocess_data, which changed arity), so every pysindy entry
    point raises ValueError in this environment.  This is the same algorithm --
    ridge least squares, zero out coefficients below `threshold`, refit on the
    surviving support, iterate to a fixed point.
    """
    n_feat, n_out = Theta.shape[1], Y.shape[1]
    G = Theta.T @ Theta + alpha * np.eye(n_feat)
    Xi = np.linalg.solve(G, Theta.T @ Y)
    for _ in range(max_iter):
        small = np.abs(Xi) < threshold
        if not small.any():
            break
        Xi_new = np.zeros_like(Xi)
        for j in range(n_out):
            big = ~small[:, j]
            if not big.any():
                continue
            Tb = Theta[:, big]
            Gb = Tb.T @ Tb + alpha * np.eye(Tb.shape[1])
            Xi_new[big, j] = np.linalg.solve(Gb, Tb.T @ Y[:, j])
        if np.allclose(Xi_new, Xi):
            Xi = Xi_new
            break
        Xi = Xi_new
    return Xi


def make_sindy(Xtr, target, powers, Xref, dt, discrete,
               thresholds=(1e-4, 1e-3, 1e-2, 1e-1)):
    """STLSQ on the given library; threshold picked on a held-out train slice."""
    t0 = time.time()
    fm = FeatureMap(powers, Xref)
    Theta = fm(Xtr)
    Y = np.asarray(target, dtype=np.float64)

    n = Theta.shape[0]
    cut = int(0.8 * n)
    best, best_err = None, np.inf
    for thr in thresholds:
        Xi = stlsq(Theta[:cut], Y[:cut], thr)
        if not np.any(np.abs(Xi) > 0):                 # thresholded everything
            continue
        err = float(np.mean((Theta[cut:] @ Xi - Y[cut:]) ** 2))
        if np.isfinite(err) and err < best_err:
            best, best_err = (Xi, thr), err
    if best is None:
        Xi, thr = _lstsq(Theta, Y), 0.0
    else:
        Xi, thr = best
    wall = time.time() - t0

    def rhs(x):
        return fm(x) @ Xi

    if discrete:
        def onestep_np(X):
            return rhs(np.asarray(X, dtype=np.float64))

        def step(x):
            return rhs(x)
    else:
        def onestep_np(X):
            return _rk4(rhs, np.asarray(X, dtype=np.float64), dt)

        def step(x):
            return _rk4(rhs, x, dt)

    def rollout(x0, n_steps):
        x = np.asarray(x0, dtype=np.float64)[None, :]
        out = [x[0].copy()]
        for _ in range(n_steps):
            x = step(x)
            if not np.all(np.isfinite(x)):
                x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            x = np.clip(x, -1e6, 1e6)
            out.append(x[0].copy())
        return np.stack(out, axis=0)

    n_active = int(np.sum(np.abs(Xi) > 0))
    return onestep_np, rollout, n_active, wall, fm.n_atoms, thr


# ===========================================================================
# Downstream method 3: KANDy -- one KAN layer over the library features
# ===========================================================================
class KandyLayer(nn.Module):
    """Single KAN layer mapping library features to the dynamics target.

    KANDy replaces SINDy's sparse regression with a KAN layer; the native form
    reads the raw state, the ablated form reads the discovered atoms.
    """

    def __init__(self, n_feat, out_dim, mean, std, grid=8):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(std, dtype=torch.float32))
        self.layer = KANLinear(n_feat, out_dim, grid_size=grid,
                               grid_range=[-3.0, 3.0])

    def forward(self, feats):
        return self.layer((feats - self.mu) / self.sd)


def make_kandy(Xtr, target, powers, Xref, dt, discrete, device, seed,
               epochs=200, batch=512, lr=1e-3):
    set_seed(seed)
    t0 = time.time()
    if powers is None:                        # native KANDy: raw state input
        fm = None
        F = np.asarray(Xtr, dtype=np.float64)

        def featurize(X):
            return np.asarray(X, dtype=np.float64)
        n_atoms = F.shape[1]
    else:
        fm = FeatureMap(powers, Xref)
        F = fm(Xtr)

        def featurize(X):
            return fm(X)
        n_atoms = fm.n_atoms

    mu, sd = F.mean(axis=0), F.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Y = np.asarray(target, dtype=np.float64)

    model = KandyLayer(F.shape[1], Y.shape[1], mu, sd).to(device)
    Ft = torch.as_tensor(F, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = Ft.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            loss = ((model(Ft[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    wall = time.time() - t0
    model.eval()

    def rhs(x):
        with torch.no_grad():
            f = torch.as_tensor(featurize(x), dtype=torch.float32, device=device)
            return model(f).cpu().numpy().astype(np.float64)

    if discrete:
        def onestep_np(X):
            return rhs(np.asarray(X, dtype=np.float64))

        def step(x):
            return rhs(x)
    else:
        def onestep_np(X):
            return _rk4(rhs, np.asarray(X, dtype=np.float64), dt)

        def step(x):
            return _rk4(rhs, x, dt)

    def rollout(x0, n_steps):
        x = np.asarray(x0, dtype=np.float64)[None, :]
        out = [x[0].copy()]
        for _ in range(n_steps):
            x = step(x)
            if not np.all(np.isfinite(x)):
                x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            x = np.clip(x, -1e6, 1e6)
            out.append(x[0].copy())
        return np.stack(out, axis=0)

    return onestep_np, rollout, n_params(model), wall, n_atoms


# ===========================================================================
# Per-system driver
# ===========================================================================
def bench_system(name, seed, device, smoke, rule="raw", n_ics=48,
                 reuse_dicts=None, arms=None):
    cfg = SYSTEMS[name]
    dt, d = cfg["dt"], cfg["state_dim"]
    degree = DEGREE[name]
    epochs = 3 if smoke else cfg["epochs"]
    kandy_epochs = 3 if smoke else 200

    train_traj, test_traj = get_data(cfg, seed, smoke)
    Xtr, Ytr = make_pairs(train_traj)
    if smoke:
        Xtr, Ytr = Xtr[:2000], Ytr[:2000]
    Xte, Yte = make_pairs(test_traj)
    std_ref = np.maximum(test_traj.reshape(-1, d).std(axis=0), 1e-8)
    flat = train_traj.reshape(-1, d)
    grl1 = [float(flat.min()) - 1.0, float(flat.max()) + 1.0]
    n_roll = 10 if smoke else (cfg["n_roll"] or
                               int(3 * (1.0 / cfg["lambda_max"]) / dt))
    n_roll = min(n_roll, test_traj.shape[1] - 1)

    # Continuous-time systems regress xdot; maps regress x_{n+1}.
    continuous = (name == "lorenz")
    if continuous:
        target = lorenz_rhs(torch.as_tensor(Xtr, dtype=torch.float64)).numpy()
    else:
        target = np.asarray(Ytr, dtype=np.float64)

    rows, curves, dicts = [], {}, {}

    def record(method, onestep_np, ro, npar, wall, n_atoms, extra=None):
        def po(Xt):
            return torch.as_tensor(onestep_np(Xt.cpu().numpy()),
                                   dtype=torch.float32)
        raw, norm = one_step_errors(po, Xte, Yte)
        mn = nrmse_curve(ro, test_traj, n_roll, (4 if smoke else n_ics), std_ref)
        cr = horizon_crossings(mn, dt, cfg["lambda_max"])
        two = int(min(2 * (1.0 / cfg["lambda_max"]) / dt, len(mn) - 1)) \
            if cfg["lambda_max"] else len(mn) - 1
        row = dict(system=name, method=method, seed=seed, degree=degree,
                   n_atoms=n_atoms, one_step_mse=raw, one_step_nmse=norm,
                   nrmse_end=float(mn[two]), n_params=npar,
                   wall_time_s=round(wall, 1), n_roll=n_roll, **cr)
        row.update(extra or {})
        rows.append(row)
        curves[method] = mn.tolist()
        print(f"  [{method:>12}] atoms={n_atoms:>3} 1step_nmse={norm:.3e} "
              f"VPT0.5={cr['vpt_0.5_steps']:>4.0f}st "
              f"VPT1.0={cr['vpt_1.0_steps']:>4.0f}st wall={wall:.0f}s")

    def safe(method, fn):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [{method:>12}] FAILED: {type(e).__name__}: {e}")

    def want(arm):
        return arms is None or arm.split("_")[0] in arms

    # -- Phase 1: train the model, then read out its dictionary -------------
    print(f"\n=== {name} (seed {seed}, degree {degree}) ===")
    if reuse_dicts is not None:
        # Re-score downstream arms against a previously saved T without
        # retraining the encoder (used when only a downstream method changed).
        saved = json.load(open(reuse_dicts))[f"{name}_s{seed}"]
        inv = {power_name(p, VAR_NAMES[name]): p
               for p in monomial_powers(d, degree)}
        T_powers = [inv[a] for a in saved["atoms"] if a in inv]
        dicts[seed] = saved
        print(f"  reusing saved T: {len(T_powers)} atoms "
              f"(from {reuse_dicts})")
        full_powers = monomial_powers(d, degree)
        libraries = [("disc", T_powers), ("poly", full_powers)]
        if name in ORACLE:
            libraries.append(("oracle", ORACLE[name]))
        for tag, powers in libraries:
            if not powers:
                continue
            if want("edmd"):
                def _e(tag=tag, powers=powers):
                    po, ro, npar, w, na = make_edmd(Xtr, Ytr, powers, Xtr)
                    record(f"edmd_{tag}", po, ro, npar, w, na)

                def _el(tag=tag, powers=powers):
                    po, ro, npar, w, na = make_edmd(Xtr, Ytr, powers, Xtr,
                                                    relift=False)
                    record(f"edmdlin_{tag}", po, ro, npar, w, na)
                safe(f"edmd_{tag}", _e)
                safe(f"edmdlin_{tag}", _el)
            if want("sindy"):
                def _s(tag=tag, powers=powers):
                    po, ro, npar, w, na, thr = make_sindy(
                        Xtr, target, powers, Xtr, dt, discrete=not continuous)
                    record(f"sindy_{tag}", po, ro, npar, w, na,
                           extra={"stlsq_threshold": thr})
                safe(f"sindy_{tag}", _s)
        return rows, curves, dicts

    model = OursModel(d, cfg["mid"], cfg["latent"], grl1)
    wall_ours = train_ae_koopman(model, Xtr, Ytr, dt, epochs, 1e-3, 512,
                                 device, seed)
    po_ours, ro_ours = ours_predictors(model, dt, device)

    def _ours():
        raw, norm = one_step_errors(po_ours, Xte, Yte)
        mn = nrmse_curve(ro_ours, test_traj, n_roll, (4 if smoke else n_ics),
                         std_ref)
        cr = horizon_crossings(mn, dt, cfg["lambda_max"])
        two = int(min(2 * (1.0 / cfg["lambda_max"]) / dt, len(mn) - 1)) \
            if cfg["lambda_max"] else len(mn) - 1
        rows.append(dict(system=name, method="ours", seed=seed, degree=degree,
                         n_atoms=cfg["latent"], one_step_mse=raw,
                         one_step_nmse=norm, nrmse_end=float(mn[two]),
                         n_params=n_params(model),
                         wall_time_s=round(wall_ours, 1), n_roll=n_roll, **cr))
        curves["ours"] = mn.tolist()
        print(f"  [{'ours':>12}] latent={cfg['latent']:>3} "
              f"1step_nmse={norm:.3e} VPT0.5={cr['vpt_0.5_steps']:>4.0f}st "
              f"VPT1.0={cr['vpt_1.0_steps']:>4.0f}st wall={wall_ours:.0f}s")

    safe("ours", _ours)

    t_disc = time.time()
    T_powers, dict_info = discover_dictionary(model, Xtr, name, degree,
                                              device=device, rule=rule)
    dict_info["readout_wall_s"] = round(time.time() - t_disc, 1)
    dict_info["seed"] = seed
    dicts[seed] = dict_info
    print(f"  discovered T ({rule}): {dict_info['n_union']} atoms of "
          f"{dict_info['n_full_library']} (median {dict_info['per_coord_atoms_median']:.0f}"
          f"/coord, R2(g) {dict_info['r2_g_median']:.3f}, "
          f"R2(support) {dict_info['r2_support_median_' + rule]:.3f}; "
          f"scaled rule would give {dict_info['n_union_scaled']})")
    if "target_recall" in dict_info:
        print(f"    vs known target: recall {dict_info['target_recall']:.2f} "
              f"Jaccard {dict_info['target_jaccard']:.2f}")
    print(f"    {', '.join(dict_info['atoms'][:14])}"
          f"{' ...' if len(dict_info['atoms']) > 14 else ''}")

    full_powers = monomial_powers(d, degree)
    libraries = [("disc", T_powers), ("poly", full_powers)]
    if name in ORACLE:
        libraries.append(("oracle", ORACLE[name]))

    # -- Downstream arms ----------------------------------------------------
    for tag, powers in libraries:
        if not powers:
            print(f"  [edmd_{tag}] skipped: empty library")
            continue

        def _edmd(tag=tag, powers=powers):
            po, ro, npar, w, na = make_edmd(Xtr, Ytr, powers, Xtr)
            record(f"edmd_{tag}", po, ro, npar, w, na)

        def _edmd_lin(tag=tag, powers=powers):
            po, ro, npar, w, na = make_edmd(Xtr, Ytr, powers, Xtr, relift=False)
            record(f"edmdlin_{tag}", po, ro, npar, w, na)

        def _sindy(tag=tag, powers=powers):
            po, ro, npar, w, na, thr = make_sindy(
                Xtr, target, powers, Xtr, dt, discrete=not continuous)
            record(f"sindy_{tag}", po, ro, npar, w, na,
                   extra={"stlsq_threshold": thr})

        safe(f"edmd_{tag}", _edmd)
        safe(f"edmdlin_{tag}", _edmd_lin)
        safe(f"sindy_{tag}", _sindy)

    def _kandy_disc():
        po, ro, npar, w, na = make_kandy(Xtr, target, T_powers, Xtr, dt,
                                         discrete=not continuous, device=device,
                                         seed=seed, epochs=kandy_epochs)
        record("kandy_disc", po, ro, npar, w, na)

    def _kandy_state():
        po, ro, npar, w, na = make_kandy(Xtr, target, None, Xtr, dt,
                                         discrete=not continuous, device=device,
                                         seed=seed, epochs=kandy_epochs)
        record("kandy_state", po, ro, npar, w, na)

    if T_powers:
        safe("kandy_disc", _kandy_disc)
    safe("kandy_state", _kandy_state)

    return rows, curves, dicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+",
                    default=["lorenz", "ikeda", "standard_map", "cat_map"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reuse-dicts", default=None,
                    help="path to a dictionaries.json; skips encoder training "
                         "and re-scores downstream arms against the saved T")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="restrict to these arm families, e.g. --arms sindy")
    ap.add_argument("--rule", choices=["raw", "scaled"], default="raw",
                    help="atom-selection rule: raw |coeff| (paper) or "
                         "scale-aware |coeff|*std(atom)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    outdir = OUTDIR + (f"_{args.tag}" if args.tag else "")
    os.makedirs(outdir, exist_ok=True)

    all_rows, all_curves, all_dicts = [], {}, {}
    for name in args.systems:
        for seed in range(args.seeds):
            rows, curves, dicts = bench_system(name, seed, args.device,
                                               args.smoke, rule=args.rule,
                                               reuse_dicts=args.reuse_dicts,
                                               arms=args.arms)
            all_rows.extend(rows)
            all_curves[f"{name}_s{seed}"] = curves
            all_dicts[f"{name}_s{seed}"] = dicts[seed]

    fields = ["system", "method", "seed", "degree", "n_atoms", "one_step_mse",
              "one_step_nmse", "nrmse_end", "vpt_0.5_steps", "vpt_1.0_steps",
              "vpt_0.5_tau", "vpt_1.0_tau", "n_params", "wall_time_s",
              "n_roll", "stlsq_threshold"]
    csv_path = os.path.join(outdir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    with open(os.path.join(outdir, "dictionaries.json"), "w") as f:
        json.dump(all_dicts, f, indent=2)
    with open(os.path.join(outdir, "nrmse_curves.json"), "w") as f:
        json.dump(all_curves, f)

    print(f"\nWrote {csv_path}")
    print("\n=== SUMMARY (median over seeds) ===")
    print(f"{'system':>13} {'method':>13} {'atoms':>6} {'1step_nmse':>12} "
          f"{'VPT0.5':>7} {'VPT1.0':>7}")
    for name in args.systems:
        methods = []
        for r in all_rows:
            if r["system"] == name and r["method"] not in methods:
                methods.append(r["method"])
        for m in methods:
            sub = [r for r in all_rows if r["system"] == name and r["method"] == m]
            med = lambda k: float(np.median([s[k] for s in sub]))   # noqa: E731
            print(f"{name:>13} {m:>13} {med('n_atoms'):>6.0f} "
                  f"{med('one_step_nmse'):>12.3e} "
                  f"{med('vpt_0.5_steps'):>7.0f} {med('vpt_1.0_steps'):>7.0f}")


if __name__ == "__main__":
    main()
