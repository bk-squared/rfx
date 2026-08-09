#!/usr/bin/env python3
"""Produce the cv06b notch-e4 rfx leg fixture (``msl_stub_notch_rfx_dx50.json``).

THE MISSING PRODUCER (issue #519)
---------------------------------
``tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json`` was committed as
evidence with NO committed producer — the matched dx=50 µm geometry survived
only as prose in the fixture meta::

    "matched openEMS dx=50um (l_line=5mm, margin=1mm, stub=12mm,
     eps_r=3.66, h_sub=254um, W=600um)"

A fixture with no committed producer cannot be honestly regenerated (PR #516
review finding F7). This script IS that producer. It reconstructs the matched
geometry from the meta prose, imitating the canonical in-repo example
``validation/crossval/06b_msl_notch_filter_uniform.py::_build_sim`` (identical
stack: ro4350b substrate box, PEC trace running the full x-extent through the
CPML per commit 8882ef1, open stub at LX/2, ``add_msl_port`` +x/-x at
PORT_MARGIN, BoundarySpec x/y=cpml z=(pec,cpml), cpml_layers=8, F_MAX=7 GHz) —
with the dx=50 leg's L_LINE=5 mm, PORT_MARGIN=1 mm, DX=50 µm instead of
cv06b's 30 mm / 2 mm / 80 µm — then runs
``compute_msl_s_matrix(n_freqs=100, num_periods=20.0)`` and writes the fixture
in the committed JSON schema
(meta / freqs_ghz / s11_mag / s21_mag / re_z0_median_ohm / notch, float32
magnitudes, no trailing newline).

This is the rfx home repo: ``sim.preflight(strict=False)`` runs and its output
is part of the evidence (quote it with any |S| number). The ring-down settling
witness (``settling_db``, −40 dB rule), the per-bin ``reliable`` mask, the
S-assembly rule and the passivity-projection clip amounts are printed so the
fixture numbers carry their honesty context; they are intentionally NOT added
to the fixture schema (schema is locked by the committed evidence chain).

After (re)producing the fixture, refresh the derived summary with::

    python scripts/diagnostics/build_msl_notch_openems_comparison.py

and recommit fixture + comparison_summary.json TOGETHER
(``test_msl_notch_e4_comparison_gates.py::test_committed_summary_matches_rederived``
is a byte-equality lock between them).

Runtime: two ~1.6M-cell FDTD drives (the committed evidence commit recorded
~65 min on CPU).

Usage::

    python scripts/diagnostics/build_msl_notch_rfx_dx50.py \
        [--output tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = _REPO_ROOT / "tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json"

# Matched-geometry constants — the fixture meta prose, made executable.
# Shared with cv06b (validation/crossval/06b_msl_notch_filter_uniform.py):
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
STUB_LEN = 12e-3
F_MAX = 7e9
# dx=50 leg specifics (cv06b ships 30mm / 2mm / 80µm instead):
L_LINE = 5e-3
PORT_MARGIN = 1e-3
DX = 50e-6

N_FREQS = 100
NUM_PERIODS = 20.0


def build_sim() -> Simulation:
    """The cv06b ``_build_sim`` construction at the dx=50 matched geometry."""
    LX = L_LINE + 2 * PORT_MARGIN
    # Lateral box: W + 2·(2·h_sub + 8·dx) on the MSL side, plus stub_length
    # on the +y side to fit the open-circuit stub.
    msl_clearance = 2 * (2 * H_SUB + 8 * DX)
    LY = W_TRACE + msl_clearance + STUB_LEN + 2 * (2 * H_SUB + 8 * DX)
    LZ = H_SUB + 1.5e-3

    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")

    # Place trace at y where there's clearance below + stub above.
    y_trace = (2 * H_SUB + 8 * DX) + W_TRACE / 2.0
    trace_y_lo = y_trace - W_TRACE / 2.0
    trace_y_hi = y_trace + W_TRACE / 2.0

    # Main microstrip line (full LX so it goes through CPML — required for
    # MSL port termination, see commit 8882ef1 on msl_port_integration test).
    sim.add(
        Box((0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + DX)),
        material="pec",
    )

    # Open-circuit stub branching off the main line at x = LX/2.
    stub_x_centre = LX / 2.0
    stub_x_lo = stub_x_centre - W_TRACE / 2.0
    stub_x_hi = stub_x_centre + W_TRACE / 2.0
    sim.add(
        Box((stub_x_lo, trace_y_hi, H_SUB),
            (stub_x_hi, trace_y_hi + STUB_LEN, H_SUB + DX)),
        material="pec",
    )

    sim.add_msl_port(
        position=(PORT_MARGIN, y_trace, 0.0),
        width=W_TRACE, height=H_SUB,
        direction="+x", impedance=50.0,
    )
    sim.add_msl_port(
        position=(PORT_MARGIN + L_LINE, y_trace, 0.0),
        width=W_TRACE, height=H_SUB,
        direction="-x", impedance=50.0,
    )
    return sim


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(_DEFAULT_OUT))
    args = p.parse_args(argv)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path

    print("=" * 70)
    print("notch-e4 rfx leg producer — matched openEMS dx=50um geometry (#519)")
    print("=" * 70)
    print(f"eps_r={EPS_R}, h_sub={H_SUB*1e6:.0f}um, W={W_TRACE*1e6:.0f}um, "
          f"stub={STUB_LEN*1e3:.0f}mm")
    print(f"l_line={L_LINE*1e3:.0f}mm, margin={PORT_MARGIN*1e3:.0f}mm, "
          f"dx={DX*1e6:.0f}um (h_sub/dx={H_SUB/DX:.2f} cells)")
    print(f"n_freqs={N_FREQS}, num_periods={NUM_PERIODS:g}")
    print()

    sim = build_sim()

    print("Preflight (verbatim, part of the evidence):")
    sim.preflight(strict=False)
    print()

    print("Running rfx 2-port S-matrix sweep (2 drives)...")
    t0 = time.time()
    res = sim.compute_msl_s_matrix(n_freqs=N_FREQS, num_periods=NUM_PERIODS)
    print(f"  ... done in {time.time() - t0:.1f}s")
    print()

    f = np.asarray(res.freqs)
    s11_mag = np.abs(np.asarray(res.S[0, 0, :]))
    s21_mag = np.abs(np.asarray(res.S[1, 0, :]))
    z0 = np.asarray(res.Z0[0, :])

    # Honesty context (printed, not committed — schema is locked).
    settling = getattr(res, "settling_db", None)
    if settling is not None:
        print(f"settling_db per drive (rule: <= -40 dB): "
              f"{np.asarray(settling).tolist()}")
    reliable = getattr(res, "reliable", None)
    if reliable is not None:
        rel = np.asarray(reliable)
        bad = np.where(~np.all(rel, axis=0))[0]
        print(f"reliable: {bad.size}/{rel.shape[1]} bins flagged"
              + (f" at f_ghz={[round(float(f[i])/1e9, 4) for i in bad]}"
                 if bad.size else ""))
    print(f"assembly: {getattr(res, 'assembly', None)}")
    pc = getattr(res, "passivity_correction", None)
    if pc is not None:
        pc = np.asarray(pc)
        print(f"passivity_correction: max={float(pc.max()):.6f} "
              f"({int((pc > 0).sum())} bins clipped)")
    print()

    s21_db = 20 * np.log10(s21_mag + 1e-30)
    i_notch = int(np.argmin(s21_db))
    f_notch_ghz = float(f[i_notch]) / 1e9
    depth_db = float(s21_db[i_notch])
    re_z0_median = float(np.median(z0.real))
    max_energy_sum = float((s11_mag ** 2 + s21_mag ** 2).max())

    print(f"notch: f={f_notch_ghz:.4f} GHz, depth={depth_db:.2f} dB")
    print(f"re_z0_median_ohm={re_z0_median:.2f}, "
          f"max(|S11|^2+|S21|^2)={max_energy_sum:.4f}")

    record = {
        "meta": {
            "solver": "rfx",
            "api": "Simulation.compute_msl_s_matrix (add_msl_port)",
            "geometry": (
                "matched openEMS dx=50um (l_line=5mm, margin=1mm, stub=12mm, "
                "eps_r=3.66, h_sub=254um, W=600um)"
            ),
            "dx_um": DX * 1e6,
            "h_sub_cells": round(H_SUB / DX, 2),
            "n_freqs": N_FREQS,
            "num_periods": NUM_PERIODS,
        },
        "freqs_ghz": (f.astype(np.float32) / np.float32(1e9)).tolist(),
        "s11_mag": s11_mag.astype(np.float32).tolist(),
        "s21_mag": s21_mag.astype(np.float32).tolist(),
        "re_z0_median_ohm": round(re_z0_median, 2),
        "notch": {
            "f_ghz": round(f_notch_ghz, 4),
            "depth_db": round(depth_db, 2),
        },
    }
    out_path.write_text(json.dumps(record, indent=2))
    print(f"\nWrote {out_path}")
    print("Now refresh the derived summary (byte-locked to this fixture):\n"
          "  python scripts/diagnostics/build_msl_notch_openems_comparison.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
