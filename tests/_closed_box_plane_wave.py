"""The closed-box plane-wave scenes and the test's own face-read level.

The reference scene: dx 2.5 mm, domain 55 x 55 x 40 mm, CPML 8, a closed TF/SF
box with margin 3 driven along +x by a differentiated Gaussian (f0 0.6 GHz,
bandwidth 0.95), float32. Its 1-D line holds the pulse at the face read nodes
until step 862 and then falls below 1e-6 of its peak.

Every level here is formed from ``record_face_values``' rows with numpy:
``r(n) = max(I_E(n) / P_E, I_H(n) / P_H)``, float64 division of the float32
per-step maxima by their float32 peaks.
"""
import math

import numpy as np

from rfx import Box, Simulation
from rfx.sources.tfsf import tfsf_from_declaration
from rfx.sources.tfsf_discrete_plane_wave import box_bounds
from rfx.stepping.incident_line import record_face_values

TOL = 1e-6
DX = 0.0025
F0 = 0.6e9
REFERENCE = dict(f0=F0, bandwidth=0.95, polarization="ez", theta_deg=0, phi_deg=0,
                 direction="+x", margin=3, closed_box=True, waveform="differentiated_gaussian")
EPS0 = 8.8541878128e-12
C0 = 299792458.0


def declare(change=None):
    """The reference scene's plane wave, with ``change`` applied to its declaration."""
    sim = Simulation(freq_max=1.5e9, domain=(.055, .055, .040), dx=DX, boundary="cpml",
                     cpml_layers=8, snap="declared")
    sim.add_tfsf_source(**{**REFERENCE, **(change or {})})
    return sim


def line(sim, n, *, line_steps="run"):
    """The line the runner builds for ``n`` steps (``line_steps`` n on run, None on forward)."""
    grid = sim._build_grid()
    cfg, state = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=n,
                                       line_steps=n if line_steps == "run" else None)
    return cfg, state, grid


def recorded_maxima(sim, n):
    """``(I_H, I_E)`` per step from ``record_face_values``, and the realized direction."""
    cfg, state, grid = line(sim, n)
    (e_line, h_line), _ = record_face_values(cfg, state, grid.dx, grid.dt)
    i_e = np.max(np.abs(np.asarray(e_line)), axis=1)
    i_h = np.max(np.abs(np.asarray(h_line)), axis=1)
    return i_h, i_e, tuple(int(v) for v in cfg.m)


def level(i_h, i_e):
    """r(n): each kind by its own peak, float64 on the host."""
    e = np.asarray(i_e).astype(np.float64) / np.float64(np.max(i_e))
    h = np.asarray(i_h).astype(np.float64) / np.float64(np.max(i_h))
    return np.maximum(e, h)


def window_starts(n):
    """The admission window [round(0.5 N), N) and the early window [round(0.25 N), N)."""
    return {"admission": int(round(0.5 * n)), "early": int(round(0.25 * n))}


def pulse_end(r):
    """1 + the last step above ``TOL``."""
    above = np.flatnonzero(r > TOL)
    return int(above[-1]) + 1 if above.size else 0


def window_class(start, value, n_end):
    """At most ``TOL``; a pulse refusal (opens before ``n_end``); a floor (opens at or after)."""
    if value <= TOL:
        return "off"
    return "pulse" if start < n_end else "floor"


def windows(i_h, i_e, lengths, n_end):
    """{(N, window): (start, level, class)} over the prefixes of one record."""
    table = {}
    for n in lengths:
        r = level(i_h[:n], i_e[:n])
        for name, start in window_starts(n).items():
            value = float(np.max(r[start:n]))
            table[(n, name)] = (start, value, window_class(start, value, n_end))
    return table


def same_level(a, b):
    """The same class, and 0.5 <= a / b <= 2 where both are at most ``TOL``."""
    if a[2] != b[2]:
        return False
    return a[2] != "off" or 0.5 <= a[1] / b[1] <= 2.0


def slotted_box(z_fraction=0.62):
    """The reference plane wave on a lossy slotted PEC box.

    The walls' outer faces lie 2 cells inside the TF/SF faces; the walls are
    one cell thick. The fill puts the PEC box's lowest mode (E along z) at
    1.5 f0 with dielectric Q 300. The -x wall has a slot 3 cells along y and
    1 along z. Returns the model, the identification probes and the interior
    node bounds.
    """
    sim = Simulation(freq_max=1.5e9, domain=(.055, .055, .040), dx=DX, boundary="cpml",
                     cpml_layers=8, snap="declared")
    grid = sim._build_grid()
    tf = box_bounds(grid.shape, grid.cpml_layers, REFERENCE["margin"])
    outer = {a: (getattr(tf, a + "_lo") + 2, getattr(tf, a + "_hi") - 2) for a in "xyz"}
    inner = {a: (lo + 1, hi - 1) for a, (lo, hi) in outer.items()}
    node = {a: tuple(grid.node_of(a, i) for i in inner[a]) for a in "xyz"}
    wall = {a: tuple(grid.node_of(a, i) for i in outer[a]) for a in "xyz"}
    a_m, d_m, b_m = (node[a][1] - node[a][0] for a in "xyz")
    f_r = 1.5 * F0
    eps_r = (C0 / (2 * f_r)) ** 2 * (1 / a_m ** 2 + 1 / d_m ** 2)
    sim.add_material("fill", eps_r=eps_r, sigma=2 * math.pi * f_r * EPS0 * eps_r / 300)
    sim.add(Box(tuple(node[a][0] for a in "xyz"), tuple(node[a][1] for a in "xyz")), material="fill")
    (x0, x1), (y0, y1), (z0, z1) = (wall[a] for a in "xyz")
    (xi0, xi1), (yi0, yi1), (zi0, zi1) = (node[a] for a in "xyz")
    for lo, hi in (((xi1, y0, z0), (x1, y1, z1)), ((x0, y0, z0), (x1, yi0, z1)),
                   ((x0, yi1, z0), (x1, y1, z1)), ((x0, y0, z0), (x1, y1, zi0)),
                   ((x0, y0, zi1), (x1, y1, z1))):
        sim.add(Box(lo, hi), material="pec")
    sy = inner["y"][0] + (inner["y"][1] - inner["y"][0] - 3) // 2
    sz = inner["z"][0] + (inner["z"][1] - inner["z"][0] - 1) // 2
    (sy0, sy1), (sz0, sz1) = ((grid.node_of("y", sy), grid.node_of("y", sy + 3)),
                              (grid.node_of("z", sz), grid.node_of("z", sz + 1)))
    for lo, hi in (((x0, yi0, zi0), (xi0, sy0, zi1)), ((x0, sy1, zi0), (xi0, yi1, zi1)),
                   ((x0, sy0, zi0), (xi0, sy1, sz0)), ((x0, sy0, sz1), (xi0, sy1, zi1))):
        sim.add(Box(lo, hi), material="pec")
    sim.add_tfsf_source(**REFERENCE)
    corner = (xi0, yi0, zi0)
    probes = tuple((tuple(c + f * s for c, f, s in zip(corner, frac, (a_m, d_m, b_m))), comp)
                   for frac, comp in (((0.62, 0.38, z_fraction), "ez"),
                                      ((0.30, 0.62, z_fraction), "hy")))
    for position, component in probes:
        sim.add_probe(position=position, component=component)
    return sim, probes, inner
