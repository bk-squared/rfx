"""Issue #786 — the W4R port-driven P-C resonator, copied verbatim-in-behaviour.

PROVENANCE. Every constant and every line of :func:`build_sim` below is a
functional copy of ``validation/research/multiband_nu/{fixtures,
w4r_port_supraconvergence}.py`` on branch ``agent/multiband-nu-envelope``
(PR #785), which is the code that produced
``results/w4r_supraconvergence.json`` — the measurement issue #786 is about.
That branch is NOT modified by this lane; the copy exists so this lane's
harness is self-contained and rerunnable from ``main``'s solver (verified:
``git diff main agent/multiband-nu-envelope -- rfx/`` touches only
``rfx/api/_preflight.py`` advisories and ``rfx/api/__init__.py``
re-exports, no solver code).

The copy is CHECKED, not assumed: ``d0_reproduce.py`` re-runs the uniform
ladder and compares every rung against the committed PR #785 numbers.

Sole additions over the original ``measure()``:

* the raw probe time series and ``dt`` are returned so an INDEPENDENT
  frequency estimator can be run on the identical record (discriminator
  D4c) without paying for the FDTD run twice;
* ``src_amp`` / ``src_positions`` / ``with_trace`` knobs, all defaulting to
  the PR #785 values, so D2 and D3 change exactly one thing each.
"""

from __future__ import annotations

import time

import numpy as np

from rfx import Simulation, Box, GaussianPulse

C0 = 299792458.0

# --- P-C geometry (fixtures.py, PR #785) -------------------------------
PC_A = 27e-3
PC_B = 22.5e-3
PC_H_SUB = 1.5e-3
PC_H_TRACE_BAND = 1.5e-3
PC_AIR1 = 4.5e-3
PC_H_UPPER = 1.5e-3
PC_AIR2 = 4.5e-3
PC_EPS_SUB = 4.3
PC_EPS_UPPER = 2.2
PC_DX0 = 0.75e-3
PC_DZF0 = 0.25e-3
RATIO_CAP = 1.4
PC_TOTAL_H = PC_H_SUB + PC_H_TRACE_BAND + PC_AIR1 + PC_H_UPPER + PC_AIR2

# Declared trace footprint (the PEC node set the half-cell margins target).
TRACE_X = (6.75e-3, 20.25e-3)
TRACE_Y = (9.0e-3, 13.5e-3)
TRACE_Z = (PC_H_SUB, PC_H_SUB + PC_H_TRACE_BAND)

# --- W4R instrument (w4r_port_supraconvergence.py, PR #785) ------------
BAND = (4.0e9, 6.5e9)
F_MAX = 12e9
T_TOTAL = 20e-9
Q_MIN = 30.0
SRC_P = (6.75e-3, 11.25e-3, 0.75e-3)
SRC_M = (20.25e-3, 11.25e-3, 0.75e-3)
PRB = (18.0e-3, 11.25e-3, 0.75e-3)
WAVEFORM = dict(f0=6e9, bandwidth=0.9)
SUBPIXEL = True

SCALES = (0.5, 0.6, 0.75, 1.0, 1.5)
REF_SCALE = 0.25


def _sym_air_band(length: float, dzf: float, cap: float = RATIO_CAP):
    up = []
    d = dzf
    while d * cap <= 4 * dzf + 1e-15:
        d = d * cap
        up.append(d)
    ramp_len = 2 * sum(up)
    plateau_d = up[-1] if up else dzf
    rem = length - ramp_len
    if rem < 0:
        while up and rem < 0:
            up = up[:-1]
            ramp_len = 2 * sum(up)
            plateau_d = up[-1] if up else dzf
            rem = length - ramp_len
    n_plateau = max(0, int(np.floor(rem / plateau_d)))
    residual = rem - n_plateau * plateau_d
    cells = up + [plateau_d] * n_plateau + up[::-1]
    if residual > 1e-12:
        if n_plateau:
            cells = (up + [plateau_d + residual / n_plateau] * n_plateau
                     + up[::-1])
        elif up:
            top = up[-1] + residual / 2
            cells = up[:-1] + [top, top] + up[::-1][1:]
        else:
            cells = [length]
    s = sum(cells)
    mid = len(cells) // 2
    cells[mid] += length - s
    assert abs(sum(cells) - length) < 1e-9
    return cells


def pc_dz_profile_sym(scale: float) -> np.ndarray:
    dzf = PC_DZF0 * scale
    prof: list[float] = []
    for band_len in (PC_H_SUB, PC_H_TRACE_BAND):
        n = int(round(band_len / dzf))
        assert abs(n * dzf - band_len) < 1e-9, (band_len, dzf)
        prof += [dzf] * n
    prof += _sym_air_band(PC_AIR1, dzf)
    prof += [dzf] * int(round(PC_H_UPPER / dzf))
    prof += _sym_air_band(PC_AIR2, dzf)
    out = np.asarray(prof, dtype=np.float64)
    assert abs(out.sum() - PC_TOTAL_H) < 1e-9
    return out


def pc_uniform_profile(scale: float) -> np.ndarray:
    dzf = PC_DZF0 * scale
    n = int(round(PC_TOTAL_H / dzf))
    assert abs(n * dzf - PC_TOTAL_H) < 1e-9
    return np.full(n, dzf, dtype=np.float64)


def n_steps_for(scale: float, dz_min: float) -> int:
    dx = PC_DX0 * scale
    dt = 0.99 / (C0 * np.sqrt(2.0 / dx ** 2 + 1.0 / dz_min ** 2))
    return int(round(T_TOTAL / dt))


def build_sim(scale: float, dz_profile: np.ndarray, antisym: bool = True,
              with_trace: bool = True, src_amp: float = 1.0,
              src_p=SRC_P, src_m=SRC_M, prb=PRB) -> Simulation:
    """PR #785 ``build_sim`` with knife-edge-free (half-cell margin) drawing.

    ``with_trace=False`` deletes ONLY the PEC trace (D2 control); every
    other declaration, the port pair and the probe are untouched.
    """
    assert abs(dz_profile.sum() - PC_TOTAL_H) < 1e-9
    dx = PC_DX0 * scale
    dzf = PC_DZF0 * scale
    assert abs(round(PC_A / dx) * dx - PC_A) < 1e-9
    sim = Simulation(
        freq_max=F_MAX, domain=(PC_A, PC_B, PC_TOTAL_H),
        dx=dx, boundary="pec", dz_profile=dz_profile,
    )
    sim.add_material("sub", eps_r=PC_EPS_SUB, sigma=0.0)
    sim.add_material("upper", eps_r=PC_EPS_UPPER, sigma=0.0)
    sim.add(Box((0, 0, 0), (PC_A, PC_B, PC_H_SUB)), material="sub")
    z_up0 = PC_H_SUB + PC_H_TRACE_BAND + PC_AIR1
    sim.add(Box((0, 0, z_up0), (PC_A, PC_B, z_up0 + PC_H_UPPER)),
            material="upper")
    if with_trace:
        sim.add(Box((TRACE_X[0] - dx / 2, TRACE_Y[0] - dx / 2,
                     TRACE_Z[0] - dzf / 2),
                    (TRACE_X[1] + dx / 2, TRACE_Y[1] + dx / 2,
                     TRACE_Z[1] + dzf / 2)),
                material="pec")
    sim.add_source(src_p, "ez",
                   waveform=GaussianPulse(amplitude=+src_amp, **WAVEFORM),
                   amplitude_kind="current")
    sim.add_source(src_m, "ez",
                   waveform=GaussianPulse(
                       amplitude=((-src_amp) if antisym else src_amp),
                       **WAVEFORM),
                   amplitude_kind="current")
    sim.add_probe(prb, "ez")
    return sim


def modes_of(result):
    modes = result.find_resonances(freq_range=(3e9, 9e9))
    return sorted([m for m in modes if abs(m.Q) > Q_MIN], key=lambda m: m.freq)


def target_line(modes, band=BAND):
    in_band = [m for m in modes if band[0] <= m.freq <= band[1]]
    if not in_band:
        return float("nan"), 0.0, {"in_band": []}
    top = max(in_band, key=lambda m: abs(m.amplitude))
    others = [abs(m.amplitude) for m in in_band if m is not top]
    dom = (abs(top.amplitude) / max(others)) if others else float("inf")
    return float(top.freq), float(dom), {
        "in_band": [(float(m.freq), float(m.Q), float(abs(m.amplitude)))
                    for m in in_band]}


def measure(scale: float, multiband: bool = False, subpixel: bool | None = None,
            keep_series: bool = False, band=BAND, **build_kw) -> dict:
    """One rung. Returns the PR #785 row plus (optionally) the raw record."""
    prof = pc_dz_profile_sym(scale) if multiband else pc_uniform_profile(scale)
    t0 = time.time()
    sim = build_sim(scale, prof, **build_kw)
    sim.preflight()
    n_steps = n_steps_for(scale, float(prof.min()))
    result = sim.run(n_steps=n_steps,
                     subpixel_smoothing=(SUBPIXEL if subpixel is None
                                         else subpixel))
    wall = time.time() - t0
    modes = modes_of(result)
    f_t, dom, info = target_line(modes, band)
    row = {
        "scale": scale, "multiband": multiband,
        "nz": len(prof), "n_steps": n_steps,
        "cells": int(round(PC_A / (PC_DX0 * scale)))
                 * int(round(PC_B / (PC_DX0 * scale))) * len(prof),
        "modes": [(float(m.freq), float(m.Q), float(abs(m.amplitude)))
                  for m in modes[:8]],
        "f_target": f_t, "dominance": dom, "in_band": info["in_band"],
        "dt": float(result.dt), "wallclock_s": wall,
    }
    if keep_series:
        row["_series"] = np.asarray(result.time_series).ravel().astype(np.float64)
    return row


def _describe():
    """Print the fixture's realized cell counts per ladder scale.

    This module is a library (the lane's harnesses import it); the guard
    exists because the #737 example-fidelity gate imports every 'audited'
    script and requires module scope not to be a script body.
    """
    for s in list(SCALES) + [REF_SCALE]:
        dx, dzf = PC_DX0 * s, PC_DZF0 * s
        prof = pc_uniform_profile(s)
        print("s=%-6s dx=%.5f mm dz=%.5f mm  nz=%3d  cells=%9d  n_steps=%6d"
              % (s, dx * 1e3, dzf * 1e3, len(prof),
                 int(round(PC_A / dx)) * int(round(PC_B / dx)) * len(prof),
                 n_steps_for(s, dzf)))


if __name__ == "__main__":
    _describe()
