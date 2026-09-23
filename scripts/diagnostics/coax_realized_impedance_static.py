#!/usr/bin/env python3
"""What characteristic impedance does the REALIZED cross-section have?

A coaxial line's characteristic impedance is set by the shape of its
cross-section: `Z0 = eta0 / (sqrt(eps_r) * G)`, with `G = C/eps` the geometric
factor of the two conductors. For smooth circles that is `2 pi / ln(b/a)`. For
the staircased pair the lattice actually builds it is not, and the difference is
real physics, not an artefact -- a staircased coax at four cells across its
annulus genuinely has a different impedance from the smooth one it approximates.

This measures that, from the PEC EDGE MASKS the lane hands the solver and
nothing else. Two nodes joined by a shorted transverse edge are one conductor;
the pin is held at 1 V, the wall at 0 V, Laplace is solved on the same lattice
in the plane, and `G` comes from the field energy. It touches neither the
S-parameter extractor, nor the annular resistor, nor
`coaxial_tem_characteristic_impedance`.

**Why this exists.** The one-port `Z0` the oracle reports is
`R_dut (1 - Gamma)/(1 + Gamma)` with `R_dut` the DECLARED load, and the annular
resistor's conductivity is built from `ln(shell_inner/a)`. The load's realized
resistance and the line's impedance therefore carry the same discrete geometric
factor, and it cancels: that number reproduces the closed form at every mesh
whatever the lattice built. It is a real check of one thing -- that the
resistor's calibration and the declared annulus agree -- and it is not a
measurement of the realized line. This is.

Method and the numbers it reproduces are from the blind review of the change
that introduced it.

Usage::

    PYTHONPATH=. python scripts/diagnostics/coax_realized_impedance_static.py \\
        [--out <dir>] [--rungs 3.789288121451007 4 6 9]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
from collections import deque

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.pec import realized_pec_edge_masks  # noqa: E402
from rfx.sources.coaxial_port import (  # noqa: E402
    PTFE_EPS_R, SMA_OUTER_RADIUS, SMA_PIN_RADIUS,
    coaxial_tem_characteristic_impedance, stamp_coaxial_line,
)
from rfx.sources.sources import GaussianPulse  # noqa: E402

ETA0 = 376.730313668
CENTRE = (0.004, 0.004)
DOMAIN = (0.008, 0.008, 0.012)
ANNULUS = SMA_OUTER_RADIUS - SMA_PIN_RADIUS
DEFAULT_RUNGS = (3.789288121451007, 4.0, 6.0, 9.0)


def build(dx: float, outer_radius: float | None = None,
          shell_thickness_m: float | None = None):
    """The realized conductor cells at one cell size."""
    sim = Simulation(freq_max=40e9, domain=DOMAIN, boundary="cpml", dx=dx)
    sim.add_coaxial_port((CENTRE[0], CENTRE[1], DOMAIN[2] / 2.0), face="top",
                         pin_length=5e-3,
                         waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    grid = sim._build_grid()
    materials, _, _ = sim._build_materials(grid)
    kw = dict(center_xy=CENTRE, z_lo_index=int(grid.pad_z_lo) + 4,
              z_hi_index=int(grid.shape[2]) - int(grid.pad_z_hi) - 2,
              pin_radius=SMA_PIN_RADIUS,
              outer_radius=float(SMA_OUTER_RADIUS if outer_radius is None
                                 else outer_radius))
    if shell_thickness_m is not None:
        kw["shell_thickness_m"] = float(shell_thickness_m)
    _, shell_inner, cells = stamp_coaxial_line(grid, materials, **kw)
    return grid, np.asarray(cells), float(shell_inner)


def geometric_factor(grid, cells) -> tuple[float, int, int]:
    """``G = C/eps`` for the cross-section the shorted edges build."""
    edges = [np.asarray(e) for e in
             realized_pec_edge_masks(cells, sheets=(), wires=())]
    k = int(grid.shape[2]) // 2
    mx, my = edges[0][:, :, k], edges[1][:, :, k]
    nx, ny = mx.shape

    parent = np.arange(nx * ny)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(nx):
        for j in range(ny):
            if mx[i, j] and i + 1 < nx:
                union(i * ny + j, (i + 1) * ny + j)
            if my[i, j] and j + 1 < ny:
                union(i * ny + j, i * ny + j + 1)
    root = np.array([find(n) for n in range(nx * ny)]).reshape(nx, ny)

    i0 = int(round(CENTRE[0] / grid.dx)) + int(grid.pad_x_lo)
    j0 = int(round(CENTRE[1] / grid.dx)) + int(grid.pad_y_lo)
    pin = root == root[i0, j0]
    metal = np.zeros((nx, ny), dtype=bool)
    metal |= mx | np.roll(mx, 1, axis=0)
    metal |= my | np.roll(my, 1, axis=1)
    outside = metal & ~pin
    if not outside.any():
        raise RuntimeError("the cross-section has no outer conductor")
    wi, wj = np.argwhere(outside)[0]
    wall = root == root[wi, wj]
    if (wall & pin).any():
        raise RuntimeError("pin and wall are one conductor: the line is shorted")

    free = ~(pin | wall)
    seed = next(((i0 + d, j0) for d in range(1, nx) if free[i0 + d, j0]), None)
    if seed is None:
        raise RuntimeError("no dielectric between pin and wall")
    seen = np.zeros((nx, ny), dtype=bool)
    q = deque([seed])
    seen[seed] = True
    while q:
        ci, cj = q.popleft()
        for ni, nj in ((ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1)):
            if 0 <= ni < nx and 0 <= nj < ny and free[ni, nj] and not seen[ni, nj]:
                seen[ni, nj] = True
                q.append((ni, nj))

    ids = np.argwhere(seen)
    idx = -np.ones((nx, ny), dtype=int)
    for n, (i, j) in enumerate(ids):
        idx[i, j] = n

    from scipy.sparse import csr_matrix, lil_matrix
    from scipy.sparse.linalg import spsolve
    A = lil_matrix((len(ids), len(ids)))
    rhs = np.zeros(len(ids))
    for n, (i, j) in enumerate(ids):
        A[n, n] = 4.0
        for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if idx[ni, nj] >= 0:
                A[n, idx[ni, nj]] = -1.0
            elif pin[ni, nj]:
                rhs[n] += 1.0
    phi = spsolve(csr_matrix(A), rhs)

    f = np.zeros((nx, ny))
    for n, (i, j) in enumerate(ids):
        f[i, j] = phi[n]
    f[pin] = 1.0
    f[wall] = 0.0
    live = seen | pin | wall
    G = 0.0
    for di, dj in ((1, 0), (0, 1)):
        a = live[:nx - di, :ny - dj] & live[di:, dj:]
        d = f[:nx - di, :ny - dj] - f[di:, dj:]
        G += float(np.sum((d * a) ** 2))
    return G, int(pin.sum()), int(wall.sum())


def measure(rungs) -> dict:
    declared = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS, SMA_OUTER_RADIUS, float(PTFE_EPS_R))
    g_continuum = 2.0 * np.pi / float(np.log(SMA_OUTER_RADIUS / SMA_PIN_RADIUS))
    rows = []
    for rung in rungs:
        dx = ANNULUS / float(rung)
        grid, cells, shell_inner = build(dx)
        G, n_pin, n_wall = geometric_factor(grid, cells)
        z0 = ETA0 / (np.sqrt(float(PTFE_EPS_R)) * G)
        rows.append({
            "rung_annulus_cells": float(rung), "dx_m": dx,
            "geometric_factor": G,
            "geometric_factor_continuum": g_continuum,
            "z0_realized_ohm": float(z0),
            "z0_declared_ohm": float(declared),
            "frac_vs_declared": float(abs(z0 - declared) / declared),
            "shell_inner_radius_m": shell_inner,
            "pin_nodes": n_pin, "wall_nodes": n_wall,
        })
    return {"declared_z_tem_ohm": float(declared),
            "geometric_factor_continuum": g_continuum, "rungs": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rungs", nargs="*", type=float, default=list(DEFAULT_RUNGS))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    rec = measure(args.rungs)
    print(f"declared Z_TEM = {rec['declared_z_tem_ohm']:.4f} ohm, "
          f"continuum G = {rec['geometric_factor_continuum']:.5f}")
    print(f"{'annulus cells':>14} {'dx um':>8} {'G':>9} {'Z0 realized':>12} "
          f"{'from declared':>14}")
    for r in rec["rungs"]:
        print(f"{r['rung_annulus_cells']:14.3f} {r['dx_m']*1e6:8.2f} "
              f"{r['geometric_factor']:9.5f} {r['z0_realized_ohm']:12.4f} "
              f"{r['frac_vs_declared']*100:13.2f} %")

    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        rec["schema"] = "rfx.coax_realized_impedance_static"
        rec["schema_version"] = 1
        rec["run_id"] = args.run_id
        rec["utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        path = out / "coax_realized_impedance_static.json"
        path.write_text(json.dumps(rec, indent=1))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
