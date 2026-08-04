#!/usr/bin/env python3
"""Build the rfx-side committed reference for the #490 lane-2 MSL phase referee.

Runs ``compute_msl_s_matrix`` on the SAME substrate/trace ``tests/
test_msl_port_integration.py`` already validates (RO4350B, eps_r=3.66,
h_sub=254um, W_TRACE=600um, L_LINE=10mm, PORT_MARGIN=2mm), but at
DX=50um instead of that fixture's committed DX=80um.

Why DX=50um and not the committed DX=80um: at DX=80um this exact
substrate (h_sub/dx = 254/80 = 3.175 cells) is the recorded
"mixed-cell danger zone" that ``scripts/diagnostics/
build_msl_notch_openems_comparison.py`` found produces a NON-PHYSICAL
openEMS MSL-port extraction on the SAME substrate (its own docstring:
"at dx=80 um the substrate is only 3.175 cells ... the openEMS MSL-port
extraction is NON-PHYSICAL (|S11|^2+|S21|^2 up to 8.9)"; "dx=50 um
gives 5.08 substrate cells where BOTH solvers are passive, so it is
the only valid matched-mesh comparison"). This is a directly
applicable do-not-repeat for issue #490 lane 2 (matched-dx MSL phase
referee): reusing DX=80um here would repeat the SAME recorded openEMS
failure this repo already diagnosed once. DX=50um gives 254/50 = 5.08
substrate cells here too (identical substrate), so the same precedent
applies.

Writes a JSON fixture (freqs, S, Z0, beta, reference-plane geometry)
that ``validation/research/msl_phase_referee/openems_msl_phase_referee.py``
reads for its own Stage B comparison -- keeping that script openEMS-only
(no rfx import), matching the coax/floquet referee precedent.

Usage::

    python scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py \\
        --output tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from rfx.api import Simulation
from rfx.api._sparams import _resolve_msl_auto_offsets
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n

# ---------------------------------------------------------------------------
# Geometry -- identical to tests/test_msl_port_integration.py except DX.
# ---------------------------------------------------------------------------
EPS_R = 3.66          # RO4350B
H_SUB = 254e-6        # substrate thickness, metres
W_TRACE = 600e-6      # trace width, metres
L_LINE = 10e-3        # thru-line length
PORT_MARGIN = 2e-3    # feed -> domain edge clearance
DX = 50e-6            # matched-mesh choice (see module docstring)
F_MAX = 5e9

LX = L_LINE + 2 * PORT_MARGIN
LY = W_TRACE + 2 * (2 * H_SUB + 8 * DX)
LZ = H_SUB + 1.5e-3

N_FREQS = 30
NUM_PERIODS = 12


def _build_sim() -> Simulation:
    sim = Simulation(
        freq_max=F_MAX,
        domain=(LX, LY, LZ),
        dx=DX,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (LX, LY, H_SUB)), material="ro4350b")
    y_centre = LY / 2.0
    trace_y_lo = y_centre - W_TRACE / 2.0
    trace_y_hi = y_centre + W_TRACE / 2.0
    sim.add(
        Box((0.0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + DX)),
        material="pec",
    )
    sim.add_msl_port(
        position=(PORT_MARGIN, y_centre, 0.0),
        width=W_TRACE, height=H_SUB, direction="+x", impedance=50.0,
    )
    sim.add_msl_port(
        position=(PORT_MARGIN + L_LINE, y_centre, 0.0),
        width=W_TRACE, height=H_SUB, direction="-x", impedance=50.0,
    )
    return sim


def _reference_plane_geometry(sim: Simulation) -> dict:
    """Physical x-coordinate of each port's probe-0 plane (issue #478/#469
    ``n_probe_offset`` resolution) -- the ACTUAL plane rfx's reported S11/S21
    are referenced to (``compute_msl_s_matrix``'s ``a_fwd_d``/``b_ref_d`` use
    ``v_per_port[driven][0]``, i.e. probe 0, not the feed plane). This is
    the geometry the openEMS Stage B referee must land its own
    ``ref_plane_shift`` on for a like-for-like phase comparison.
    """
    grid = sim._build_grid()
    entries = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), grid)
    out = {}
    for pe in entries:
        x_feed, y_centre, z_lo = pe.position
        mp = MSLPort(
            feed_x=float(x_feed), y_lo=float(y_centre - pe.width / 2),
            y_hi=float(y_centre + pe.width / 2), z_lo=float(z_lo),
            z_hi=float(z_lo + pe.height), direction=pe.direction,
            impedance=pe.impedance, excitation=pe.waveform,
        )
        xs = msl_probe_x_coords_n(
            grid, mp, n_probes=pe.n_probes,
            n_offset_cells=pe.n_probe_offset, n_spacing_cells=pe.n_probe_spacing,
        )
        out[pe.name] = {
            "feed_x_m": float(x_feed),
            "direction": pe.direction,
            "n_probe_offset": int(pe.n_probe_offset),
            "n_probe_spacing": int(pe.n_probe_spacing),
            "n_probes": int(pe.n_probes),
            "probe0_x_m": float(xs[0]),
        }
    interior = {
        ax: int(getattr(grid, f"n{ax}") - getattr(grid, f"pad_{ax}_lo") - getattr(grid, f"pad_{ax}_hi"))
        for ax in ("x", "y", "z")
    }
    out["_grid"] = {
        "shape": [int(n) for n in grid.shape],
        "dx": float(grid.dx),
        "pad_x_lo": int(grid.pad_x_lo), "pad_x_hi": int(grid.pad_x_hi),
        "pad_y_lo": int(grid.pad_y_lo), "pad_y_hi": int(grid.pad_y_hi),
        "pad_z_lo": int(grid.pad_z_lo), "pad_z_hi": int(grid.pad_z_hi),
        "interior_cells": interior,
        "rasterized_clear_m": {
            ax: interior[ax] * float(grid.dx) for ax in ("x", "y", "z")
        },
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json")
    args = p.parse_args(argv)

    sim = _build_sim()
    ref_geom = _reference_plane_geometry(sim)

    t0 = time.time()
    result = sim.compute_msl_s_matrix(n_freqs=N_FREQS, num_periods=NUM_PERIODS)
    elapsed = time.time() - t0

    S = np.asarray(jax.device_get(result.S))
    Z0 = np.asarray(jax.device_get(result.Z0))
    beta = np.asarray(jax.device_get(result.beta))
    freqs = np.asarray(jax.device_get(result.freqs))

    s11 = S[0, 0, :]
    s21 = S[1, 0, :]

    out = {
        "meta": {
            "purpose": "rfx-side committed reference for #490 lane 2 (MSL phase referee)",
            "eps_r": EPS_R, "h_sub_m": H_SUB, "w_trace_m": W_TRACE,
            "l_line_m": L_LINE, "port_margin_m": PORT_MARGIN, "dx_m": DX,
            "f_max_hz": F_MAX, "n_freqs": N_FREQS, "num_periods": NUM_PERIODS,
            "elapsed_s": round(elapsed, 1),
        },
        "reference_plane_geometry": ref_geom,
        "freqs_hz": freqs.tolist(),
        "s11": [[float(c.real), float(c.imag)] for c in s11],
        "s21": [[float(c.real), float(c.imag)] for c in s21],
        "z0_port0": [[float(c.real), float(c.imag)] for c in Z0[0, :]],
        "z0_port1": [[float(c.real), float(c.imag)] for c in Z0[1, :]],
        "beta_first_port": [[float(c.real), float(c.imag)] for c in beta],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"elapsed={elapsed:.1f}s")
    print(f"mean|S11|={np.mean(np.abs(s11)):.4f} mean|S21|={np.mean(np.abs(s21)):.4f}")
    print(f"written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
