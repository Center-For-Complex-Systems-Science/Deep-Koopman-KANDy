#!/usr/bin/env python3
"""
Two-cylinder wake simulation in 2D incompressible Navier-Stokes.

This is a compact educational solver that uses:
- finite differences on a regular Cartesian grid
- Chorin pressure projection
- Brinkman penalization for two circular cylinders
- a weak outlet sponge to reduce reflections

It is designed to reproduce the kind of red/blue vorticity images often
shown for laminar unsteady wakes behind two cylinders.

Dependencies:
    numpy, matplotlib

Example:
    python two_cylinder_wake.py --nx 360 --ny 192 --re 100 --t-end 160 \
        --snapshots 10 70 140 --output wake.png

Animation example:
    python two_cylinder_wake.py --animate --fps 20 --output wake.mp4

Notes:
- This is not a production CFD code. It is meant to be readable and easy to run.
- The most image-like regime is usually Re in the 80-140 range.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Params:
    nx: int = 320
    ny: int = 160
    lx: float = 25.0
    ly: float = 10.0
    re: float = 100.0
    u_in: float = 1.0
    rho: float = 1.0
    dt: float = 0.002
    t_end: float = 150.0
    poisson_iters: int = 60
    penal_eta: float = 3e-4
    sponge_sigma: float = 1.5
    inlet_perturb: float = 0.03
    perturb_freq: float = 0.12
    output: str = "two_cylinder_wake.png"
    animate: bool = False
    fps: int = 20
    snapshot_times: tuple[float, ...] = (10.0, 70.0, 140.0)
    show: bool = False
    seed: int = 0
    cmap: str = "RdBu_r"
    omega_clip: float = 6.0

    # geometry in units of cylinder diameter D=1
    D: float = 1.0
    cx: float = 2.7
    ygap: float = 2.5  # center-to-center spacing in y


class TwoCylinderWakeSolver:
    def __init__(self, p: Params):
        self.p = p
        self.dx = p.lx / (p.nx - 1)
        self.dy = p.ly / (p.ny - 1)
        self.nu = p.u_in * p.D / p.re

        x = np.linspace(0.0, p.lx, p.nx)
        y = np.linspace(-0.5 * p.ly, 0.5 * p.ly, p.ny)
        self.x, self.y = x, y
        self.X, self.Y = np.meshgrid(x, y)

        # cylinder centers
        self.c1 = (p.cx, 0.5 * p.ygap)
        self.c2 = (p.cx, -0.5 * p.ygap)
        r = 0.5 * p.D
        self.r = r

        d1 = np.sqrt((self.X - self.c1[0]) ** 2 + (self.Y - self.c1[1]) ** 2)
        d2 = np.sqrt((self.X - self.c2[0]) ** 2 + (self.Y - self.c2[1]) ** 2)
        solid = (d1 <= r) | (d2 <= r)
        # smooth edge helps stability a bit
        edge = 0.08 * p.D
        phi1 = 0.5 * (1.0 - np.tanh((d1 - r) / edge))
        phi2 = 0.5 * (1.0 - np.tanh((d2 - r) / edge))
        self.chi = np.clip(phi1 + phi2, 0.0, 1.0)
        self.solid = solid

        # outlet sponge to damp fluctuations near the right boundary
        sponge_start = p.lx - 4.0 * p.D
        xi = np.clip((self.X - sponge_start) / max(p.lx - sponge_start, 1e-12), 0.0, 1.0)
        self.sponge = p.sponge_sigma * xi**2

        # state arrays
        self.u = np.full((p.ny, p.nx), p.u_in, dtype=float)
        self.v = np.zeros((p.ny, p.nx), dtype=float)
        self.pfield = np.zeros((p.ny, p.nx), dtype=float)

        # Start near rest in the body region.
        self.u[self.solid] = 0.0
        self.v[self.solid] = 0.0

        # Slightly smoothed start-up helps the wake develop cleanly.
        startup = 0.5 * (1.0 + np.tanh((self.X - (self.c1[0] - 1.5 * p.D)) / (0.5 * p.D)))
        self.u *= startup

        self.time = 0.0
        self.rng = np.random.default_rng(p.seed)

    def apply_velocity_bc(self, u: np.ndarray, v: np.ndarray, t: float) -> None:
        p = self.p

        # Left inflow: uniform stream + tiny time-periodic cross-stream perturbation.
        inflow_v = p.inlet_perturb * p.u_in * np.sin(2.0 * np.pi * p.perturb_freq * t)
        u[:, 0] = p.u_in
        v[:, 0] = inflow_v

        # Right outflow: zero-gradient.
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]

        # Top/bottom free-slip-ish BC:
        # normal velocity = 0, tangential gradient = 0
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        v[0, :] = 0.0
        v[-1, :] = 0.0

        # Brinkman no-slip inside cylinders.
        u[self.solid] = 0.0
        v[self.solid] = 0.0

    def apply_pressure_bc(self, pfield: np.ndarray) -> None:
        # Neumann on left/top/bottom, Dirichlet reference at outlet.
        pfield[:, 0] = pfield[:, 1]
        pfield[0, :] = pfield[1, :]
        pfield[-1, :] = pfield[-2, :]
        pfield[:, -1] = 0.0

    def ddx(self, a: np.ndarray) -> np.ndarray:
        out = np.empty_like(a)
        out[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * self.dx)
        out[:, 0] = (a[:, 1] - a[:, 0]) / self.dx
        out[:, -1] = (a[:, -1] - a[:, -2]) / self.dx
        return out

    def ddy(self, a: np.ndarray) -> np.ndarray:
        out = np.empty_like(a)
        out[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * self.dy)
        out[0, :] = (a[1, :] - a[0, :]) / self.dy
        out[-1, :] = (a[-1, :] - a[-2, :]) / self.dy
        return out

    def laplacian(self, a: np.ndarray) -> np.ndarray:
        out = np.zeros_like(a)
        out[1:-1, 1:-1] = (
            (a[1:-1, 2:] - 2.0 * a[1:-1, 1:-1] + a[1:-1, :-2]) / self.dx**2
            + (a[2:, 1:-1] - 2.0 * a[1:-1, 1:-1] + a[:-2, 1:-1]) / self.dy**2
        )
        # Fill boundaries by copying nearest interior estimate.
        out[:, 0] = out[:, 1]
        out[:, -1] = out[:, -2]
        out[0, :] = out[1, :]
        out[-1, :] = out[-2, :]
        return out

    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.ddx(u) + self.ddy(v)

    def pressure_projection(self, u_star: np.ndarray, v_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rhs = (self.p.rho / self.p.dt) * self.divergence(u_star, v_star)
        pfield = self.pfield.copy()
        dx2 = self.dx**2
        dy2 = self.dy**2
        denom = 2.0 * (dx2 + dy2)

        for _ in range(self.p.poisson_iters):
            pn = pfield.copy()
            pfield[1:-1, 1:-1] = (
                (pn[1:-1, 2:] + pn[1:-1, :-2]) * dy2
                + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx2
                - rhs[1:-1, 1:-1] * dx2 * dy2
            ) / denom
            self.apply_pressure_bc(pfield)

        u = u_star.copy()
        v = v_star.copy()
        u[1:-1, 1:-1] -= (self.p.dt / self.p.rho) * (pfield[1:-1, 2:] - pfield[1:-1, :-2]) / (2.0 * self.dx)
        v[1:-1, 1:-1] -= (self.p.dt / self.p.rho) * (pfield[2:, 1:-1] - pfield[:-2, 1:-1]) / (2.0 * self.dy)
        self.apply_velocity_bc(u, v, self.time)
        self.pfield = pfield
        return u, v

    def step(self) -> None:
        p = self.p
        u = self.u
        v = self.v

        self.apply_velocity_bc(u, v, self.time)

        du_dx = self.ddx(u)
        du_dy = self.ddy(u)
        dv_dx = self.ddx(v)
        dv_dy = self.ddy(v)

        lap_u = self.laplacian(u)
        lap_v = self.laplacian(v)

        # Conservative enough for readability; central advection is acceptable here
        # because the Re and dt are modest.
        adv_u = u * du_dx + v * du_dy
        adv_v = u * dv_dx + v * dv_dy

        u_star = u + p.dt * (-adv_u + self.nu * lap_u)
        v_star = v + p.dt * (-adv_v + self.nu * lap_v)

        # Brinkman penalization toward solid-body velocity = 0.
        fac = p.dt * self.chi / p.penal_eta
        u_star = (u_star - fac * 0.0) / (1.0 + fac)
        v_star = (v_star - fac * 0.0) / (1.0 + fac)

        # Simple outlet sponge nudging back toward inflow.
        if np.any(self.sponge > 0.0):
            u_star += p.dt * self.sponge * (p.u_in - u_star)
            v_star += p.dt * self.sponge * (0.0 - v_star)

        self.apply_velocity_bc(u_star, v_star, self.time + p.dt)
        self.u, self.v = self.pressure_projection(u_star, v_star)
        self.time += p.dt

    def vorticity(self) -> np.ndarray:
        return self.ddx(self.v) - self.ddy(self.u)


def make_figure(solver: TwoCylinderWakeSolver, snapshots: List[np.ndarray], times: Iterable[float], output: Path) -> None:
    p = solver.p
    times = list(times)
    n = len(snapshots)

    if n == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10.5, 3.4), constrained_layout=True)
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 3.1), constrained_layout=True)
        axes = np.atleast_1d(axes)

    extent = [solver.x[0], solver.x[-1], solver.y[0], solver.y[-1]]
    vmax = p.omega_clip
    vmin = -vmax

    for ax, omega, t in zip(axes, snapshots, times):
        ax.imshow(
            omega,
            origin="lower",
            extent=extent,
            cmap=p.cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            aspect="auto",
        )
        for cx, cy in [solver.c1, solver.c2]:
            ax.add_patch(patches.Circle((cx, cy), solver.r, facecolor="0.65", edgecolor="black", linewidth=1.4))
        ax.set_xlim(0.0, p.lx)
        ax.set_ylim(-0.5 * p.ly, 0.5 * p.ly)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$t={t:.0f}$", fontsize=20)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {output}")


def make_animation(solver: TwoCylinderWakeSolver, output: Path) -> None:
    p = solver.p
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    extent = [solver.x[0], solver.x[-1], solver.y[0], solver.y[-1]]

    omega = solver.vorticity()
    im = ax.imshow(
        omega,
        origin="lower",
        extent=extent,
        cmap=p.cmap,
        vmin=-p.omega_clip,
        vmax=p.omega_clip,
        interpolation="bilinear",
        aspect="auto",
        animated=True,
    )
    for cx, cy in [solver.c1, solver.c2]:
        ax.add_patch(patches.Circle((cx, cy), solver.r, facecolor="0.65", edgecolor="black", linewidth=1.4))
    title = ax.set_title("", fontsize=18)
    ax.set_xlim(0.0, p.lx)
    ax.set_ylim(-0.5 * p.ly, 0.5 * p.ly)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    frames = max(1, int(round(p.t_end / p.dt)))
    steps_per_frame = max(1, int(round(1.0 / (p.fps * p.dt))))
    nframes = max(1, frames // steps_per_frame)

    def update(_frame_idx: int):
        for _ in range(steps_per_frame):
            if solver.time < p.t_end - 1e-12:
                solver.step()
        omega = solver.vorticity()
        im.set_array(np.clip(omega, -p.omega_clip, p.omega_clip))
        title.set_text(rf"$t={solver.time:.1f}$")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=nframes, interval=1000 / p.fps, blit=True)

    if output.suffix.lower() == ".gif":
        ani.save(output, writer=animation.PillowWriter(fps=p.fps))
    else:
        ani.save(output, writer="ffmpeg", fps=p.fps, dpi=160)
    print(f"Saved animation to {output}")


def run_simulation(p: Params) -> None:
    solver = TwoCylinderWakeSolver(p)

    if p.animate:
        make_animation(solver, Path(p.output))
        if p.show:
            plt.show()
        return

    snapshot_times = list(p.snapshot_times)
    snapshot_idx = 0
    snapshots: List[np.ndarray] = []
    saved_times: List[float] = []
    nsteps = int(math.ceil(p.t_end / p.dt))

    for step in range(nsteps):
        solver.step()

        while snapshot_idx < len(snapshot_times) and solver.time >= snapshot_times[snapshot_idx] - 1e-12:
            snapshots.append(np.clip(solver.vorticity(), -p.omega_clip, p.omega_clip).copy())
            saved_times.append(snapshot_times[snapshot_idx])
            snapshot_idx += 1

        if (step + 1) % max(1, nsteps // 20) == 0:
            print(f"t = {solver.time:.2f} / {p.t_end:.2f}")

    if not snapshots:
        snapshots = [np.clip(solver.vorticity(), -p.omega_clip, p.omega_clip)]
        saved_times = [solver.time]

    make_figure(solver, snapshots, saved_times, Path(p.output))
    if p.show:
        plt.show()


def parse_args() -> Params:
    parser = argparse.ArgumentParser(description="Two-cylinder laminar wake simulation.")
    parser.add_argument("--nx", type=int, default=320, help="Grid points in x.")
    parser.add_argument("--ny", type=int, default=160, help="Grid points in y.")
    parser.add_argument("--lx", type=float, default=25.0, help="Domain length.")
    parser.add_argument("--ly", type=float, default=10.0, help="Domain height.")
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number based on D and U_in.")
    parser.add_argument("--u-in", type=float, default=1.0, help="Inflow speed.")
    parser.add_argument("--dt", type=float, default=0.002, help="Time step.")
    parser.add_argument("--t-end", type=float, default=150.0, help="Final simulation time.")
    parser.add_argument("--poisson-iters", type=int, default=60, help="Pressure Poisson iterations per step.")
    parser.add_argument("--penal-eta", type=float, default=3e-4, help="Penalization time scale.")
    parser.add_argument("--sponge-sigma", type=float, default=1.5, help="Outlet sponge strength.")
    parser.add_argument("--inlet-perturb", type=float, default=0.03, help="Cross-stream inlet perturbation amplitude as fraction of U_in.")
    parser.add_argument("--perturb-freq", type=float, default=0.12, help="Inlet perturbation frequency.")
    parser.add_argument("--D", type=float, default=1.0, help="Cylinder diameter.")
    parser.add_argument("--cx", type=float, default=2.7, help="Cylinder center x-position.")
    parser.add_argument("--ygap", type=float, default=2.5, help="Vertical center-to-center cylinder spacing.")
    parser.add_argument("--omega-clip", type=float, default=6.0, help="Vorticity color scale clip.")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap.")
    parser.add_argument("--animate", action="store_true", help="Save an animation instead of a snapshot figure.")
    parser.add_argument("--fps", type=int, default=20, help="Animation FPS.")
    parser.add_argument("--snapshots", type=float, nargs="*", default=[10.0, 70.0, 140.0], help="Snapshot times for still figure.")
    parser.add_argument("--output", default="two_cylinder_wake.png", help="Output figure or animation path.")
    parser.add_argument("--show", action="store_true", help="Display the figure window at the end.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (reserved for future use).")
    args = parser.parse_args()

    return Params(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        re=args.re,
        u_in=args.u_in,
        dt=args.dt,
        t_end=args.t_end,
        poisson_iters=args.poisson_iters,
        penal_eta=args.penal_eta,
        sponge_sigma=args.sponge_sigma,
        inlet_perturb=args.inlet_perturb,
        perturb_freq=args.perturb_freq,
        D=args.D,
        cx=args.cx,
        ygap=args.ygap,
        omega_clip=args.omega_clip,
        cmap=args.cmap,
        animate=args.animate,
        fps=args.fps,
        snapshot_times=tuple(args.snapshots),
        output=args.output,
        show=args.show,
        seed=args.seed,
    )


if __name__ == "__main__":
    run_simulation(parse_args())
