#!/usr/bin/env python3
"""THRU singular-value dx ladder — one rung per invocation.

Measures sv(f), the largest singular value of the 2x2 S-matrix per bin, of
the 2-port wire THRU fixture behind
``tests/test_lumped_twoport_vi_validation_battery.py::_build_thru`` at dx,
dx/2 or dx/4, with the CPML physical thickness and the physical run time
held constant.  What is held, what varies, the outcome table and the
validity gates are pre-declared in
``docs/design_notes/thru_singular_value_dx_ladder_predeclaration.md``; that
note binds.  This script only measures and writes one JSON.  It carries NO
verdict logic — reading the ladder is a separate round.

Usage::

    python scripts/diagnostics/thru_singular_value_dx_ladder.py \
        --dx-divisor 1 --output out/rung_dx.json

``--dx-divisor 1`` is the battery fixture byte for byte and must reproduce
the recorded ``sv_max = 1.003227`` (gate G1 of the note).  ``--dx-divisor 4``
is a long run; it belongs on the GPU lane
(``scripts/vessl_thru_singular_value_dx_ladder.yaml``), never on CPU.

Same extractor as the fixture: ``sim.run(compute_s_params=True)`` routes the
2-port wire set through ``compute_lumped_wire_s_matrix_via_scan``.  The
per-drive settling witness is taken from the driver's own per-drive forward
via a read-only spy on ``Simulation._forward_from_materials`` (the raw dict
it returns carries the probe ``time_series``), so the driver is untouched.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import jax  # noqa: E402

import rfx  # noqa: E402
from rfx import Box, Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.sources.sources import GaussianPulse  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture constants — verbatim from the battery module.  The dx rung must
# reproduce the recorded number; any edit here breaks gate G1 on purpose.
# ---------------------------------------------------------------------------
DX0_M = 0.5e-3
DOMAIN_M = (0.032, 0.020, 0.010)
FREQ_MAX_HZ = 10e9
CPML_LAYERS0 = 8
H_M = 1.0e-3          # trace height above ground = wire port extent
W_M = 5.0e-3          # trace width
X1_M = 0.008
X2_M = 0.024
Y_MID_M = DOMAIN_M[1] / 2
N_STEPS0 = 4000
FREQS_HZ = np.linspace(3e9, 7e9, 9)
Z0_OHM = 50.0
PULSE_F0_HZ = 5e9
PULSE_BW = 0.8

# Physical lengths the fixture states in cells (note section 2.1).
OVERHANG_M = DX0_M          # trace x-overhang past each port column: 0.5 mm
CPML_THICKNESS_M = CPML_LAYERS0 * DX0_M   # 4.0 mm, held

# Witness probes (note section 5): mid-gap Ez under each port and mid-line.
PROBE_Z_M = 0.5e-3
PROBE_XS_M = (X1_M, 0.5 * (X1_M + X2_M), X2_M)
PROBE_LABELS = ("port1_gap", "mid_line", "port2_gap")

BATTERY_SV_MAX = 1.003227
BATTERY_CODES = ["pec_faces_finite_pec",
                 "wire_port_dead_extent_cells",
                 "wire_port_dead_extent_cells"]
SETTLING_TAIL_FRACTION = 0.10


def build_rung(divisor: int) -> tuple[Simulation, dict]:
    """The THRU at dx = DX0/divisor with the sheet trace and physical overhang."""
    dx = DX0_M / divisor
    cpml_layers = int(round(CPML_THICKNESS_M / dx))
    sim = Simulation(
        freq_max=FREQ_MAX_HZ,
        domain=DOMAIN_M,
        dx=dx,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=cpml_layers,
    )
    # One-cell PEC sheet on top of the wire spans; overhang held at 0.5 mm.
    trace_lo = (X1_M - OVERHANG_M, Y_MID_M - W_M / 2, H_M)
    trace_hi = (X2_M + OVERHANG_M, Y_MID_M + W_M / 2, H_M + dx)
    sim.add(Box(trace_lo, trace_hi), material="pec")
    pulse = GaussianPulse(f0=PULSE_F0_HZ, bandwidth=PULSE_BW)
    sim.add_port(position=(X1_M, Y_MID_M, 0.0), component="ez",
                 impedance=Z0_OHM, extent=H_M, waveform=pulse,
                 direction="-x")
    sim.add_port(position=(X2_M, Y_MID_M, 0.0), component="ez",
                 impedance=Z0_OHM, extent=H_M, waveform=pulse,
                 direction="+x")
    for x in PROBE_XS_M:
        sim.add_probe(position=(x, Y_MID_M, PROBE_Z_M), component="ez")
    geom = {
        "dx_m": dx,
        "cpml_layers": cpml_layers,
        "cpml_thickness_m": cpml_layers * dx,
        "trace_box_lo_m": list(trace_lo),
        "trace_box_hi_m": list(trace_hi),
        "trace_thickness_cells": 1,
        "overhang_m": OVERHANG_M,
        "overhang_cells": int(round(OVERHANG_M / dx)),
        "port_extent_m": H_M,
    }
    return sim, geom


def settling_db(ts: np.ndarray | None) -> tuple[float, list[float]]:
    """Worst-probe tail/peak power ratio in dB (MSL-lane definition)."""
    if ts is None:
        return float("nan"), []
    p = np.asarray(ts, dtype=float) ** 2
    if p.ndim != 2 or p.shape[0] < 10:
        return float("nan"), []
    tail = max(1, int(p.shape[0] * SETTLING_TAIL_FRACTION))
    end = p[-tail:, :].mean(axis=0)
    peak = p.max(axis=0)
    tiny = np.finfo(float).tiny
    per_probe = 10.0 * np.log10((end + tiny) / (peak + tiny))
    return float(np.max(per_probe)), [float(v) for v in per_probe]


def git_sha(override: str | None) -> str:
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 — provenance only, never fatal
        return "unknown"


def rasterization_witness(sim: Simulation, grid) -> dict:
    """G4: finite-PEC cell count, wire port cells / live cells."""
    from rfx.sources.sources import WirePort, _wire_port_live_cells
    sheet_specs: list = []
    _m, _d, _l, pec_mask, _a, _b, _c = sim._assemble_materials(
        grid, sheet_specs=sheet_specs)
    pec_cells = int(np.asarray(pec_mask).sum()) if pec_mask is not None else -1
    ports = []
    for pe in sim._ports:
        end = list(pe.position)
        end[2] += pe.extent
        wp = WirePort(start=pe.position, end=tuple(end), component=pe.component,
                      impedance=pe.impedance, excitation=pe.waveform)
        cells, live_flags, n_live = _wire_port_live_cells(grid, wp, pec_mask)
        ports.append({"n_cells": int(len(cells)), "n_live": int(n_live),
                      "live_flags": [bool(f) for f in live_flags]})
    return {
        "grid_shape": [int(grid.nx), int(grid.ny), int(grid.nz)],
        "n_cells_total": int(grid.nx) * int(grid.ny) * int(grid.nz),
        "finite_pec_cells": pec_cells,
        "wire_ports": ports,
    }


def run_rung(divisor: int, git_sha_override: str | None) -> dict:
    t_start = time.time()
    sim, geom = build_rung(divisor)

    # Run-time rule: same physical time as the fixture, from the grids' own dt.
    grid = sim._build_grid()
    dt = float(grid.dt)
    dt0 = float(build_rung(1)[0]._build_grid().dt)
    n_steps = int(round(N_STEPS0 * dt0 / dt))
    t_total_s = n_steps * dt

    report = sim.preflight()
    preflight_msgs = [str(i) for i in report]
    for msg in preflight_msgs:
        print(f"[rung dx/{divisor}] preflight (verbatim): {msg}")
    codes = sorted(str(getattr(i, "code", None)) for i in report)
    remaining = list(codes)
    battery_codes_present = True
    for c in BATTERY_CODES:
        if c in remaining:
            remaining.remove(c)
        else:
            battery_codes_present = False
    extra_codes = remaining

    raster = rasterization_witness(sim, grid)

    # Spy: capture the per-drive probe record the production driver's own
    # forward returns.  Pass-through; the driver and its arguments are untouched.
    captured: list[tuple[int | None, np.ndarray | None]] = []
    orig_forward = sim._forward_from_materials

    def _spy(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        if kwargs.get("_return_raw_port_sparams") and isinstance(out, dict):
            ts = out.get("time_series")
            captured.append((kwargs.get("_sparam_drive_idx"),
                             None if ts is None else np.asarray(ts, dtype=float)))
        return out

    sim._forward_from_materials = _spy  # type: ignore[method-assign]

    t_run0 = time.time()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        result = sim.run(n_steps=n_steps, compute_s_params=True,
                         s_param_freqs=FREQS_HZ)
    t_run = time.time() - t_run0
    warn_msgs = [f"{w.category.__name__}: {w.message}" for w in wlist]
    for w in warn_msgs:
        print(f"[rung dx/{divisor}] warning (verbatim): {w}")

    S = np.asarray(result.s_params).astype(np.complex128)
    assert S.shape == (2, 2, len(FREQS_HZ)), S.shape
    assert np.all(np.isfinite(S)), "non-finite S"
    sv_pairs = [np.linalg.svd(S[:, :, k], compute_uv=False) for k in
                range(S.shape[2])]
    sv_max_per_bin = [float(p[0]) for p in sv_pairs]
    sv_min_per_bin = [float(p[1]) for p in sv_pairs]
    k_max = int(np.argmax(sv_max_per_bin))
    sv_max = sv_max_per_bin[k_max]
    k3 = int(np.argmin(np.abs(FREQS_HZ - 3e9)))
    excess_3ghz = sv_max_per_bin[k3] - 1.0
    diffs = np.diff(sv_max_per_bin)
    monotone_decreasing = bool(np.all(diffs < 0))

    s11, s22, s21, s12 = S[0, 0], S[1, 1], S[1, 0], S[0, 1]
    col_power = [np.abs(s11) ** 2 + np.abs(s21) ** 2,
                 np.abs(s22) ** 2 + np.abs(s12) ** 2]
    recip_abs = np.abs(s21 - s12)

    drives = []
    for idx, ts in captured:
        worst, per_probe = settling_db(ts)
        drives.append({"drive_idx": None if idx is None else int(idx),
                       "settling_db": worst,
                       "settling_db_per_probe": per_probe,
                       "record_shape": None if ts is None else list(ts.shape)})
    main_ts = getattr(result, "time_series", None)
    main_ts = None if main_ts is None else np.asarray(main_ts, dtype=float)
    main_worst, main_per_probe = settling_db(main_ts)

    field_dtype = None
    try:
        fd = sim._resolve_field_dtype()
        field_dtype = None if fd is None else str(np.dtype(fd))
    except Exception:  # noqa: BLE001 — provenance only
        field_dtype = "unresolved"

    out = {
        "study": "thru_singular_value_dx_ladder",
        "predeclaration": "docs/design_notes/thru_singular_value_dx_ladder_predeclaration.md",
        "rung": {"dx_divisor": divisor, **geom, "n_steps": n_steps,
                 "dt_s": dt, "dt0_s": dt0, "physical_time_s": t_total_s,
                 "n_steps_fixture": N_STEPS0},
        "fixture": {"domain_m": list(DOMAIN_M), "freq_max_hz": FREQ_MAX_HZ,
                    "trace_width_m": W_M, "trace_height_m": H_M,
                    "port_x_m": [X1_M, X2_M], "z0_ohm": Z0_OHM,
                    "pulse": {"f0_hz": PULSE_F0_HZ, "bandwidth": PULSE_BW},
                    "battery_sv_max": BATTERY_SV_MAX},
        "preflight": {"codes": codes,
                      "battery_codes_present": battery_codes_present,
                      "extra_codes": extra_codes,
                      "messages_verbatim": preflight_msgs},
        "warnings_verbatim": warn_msgs,
        "rasterization": raster,
        "freqs_hz": [float(f) for f in FREQS_HZ],
        "s_matrix": {
            "layout": "S[i][j][k] = {re, im} at freqs_hz[k]; i = receive, j = drive",
            "re": np.real(S).tolist(), "im": np.imag(S).tolist(),
        },
        "abs_s": {"s11": np.abs(s11).tolist(), "s22": np.abs(s22).tolist(),
                  "s21": np.abs(s21).tolist(), "s12": np.abs(s12).tolist()},
        "singular_values": {"max_per_bin": sv_max_per_bin,
                            "min_per_bin": sv_min_per_bin,
                            "sv_max": sv_max,
                            "sv_max_freq_hz": float(FREQS_HZ[k_max]),
                            "excess_3ghz": excess_3ghz,
                            "monotone_decreasing_in_f": monotone_decreasing,
                            "delta_vs_battery_sv_max": sv_max - BATTERY_SV_MAX},
        "column_power": [c.tolist() for c in col_power],
        "reciprocity_abs": recip_abs.tolist(),
        "reciprocity_abs_max": float(recip_abs.max()),
        "settling": {"per_drive": drives,
                     "main_pass_settling_db": main_worst,
                     "main_pass_per_probe": main_per_probe,
                     "probe_labels": list(PROBE_LABELS),
                     "definition": "max over probes of 10*log10(mean(E^2, last 10%) / max(E^2))"},
        "wall_time_s": {"run": t_run, "total": time.time() - t_start},
        "provenance": {
            "git_sha": git_sha(git_sha_override),
            "rfx_version": getattr(rfx, "__version__", "?"),
            "rfx_file": os.path.relpath(rfx.__file__, str(REPO))
            if rfx.__file__.startswith(str(REPO)) else rfx.__file__,
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
            "x64_enabled": bool(jax.config.x64_enabled),
            "field_dtype": field_dtype,
            "python": sys.version.split()[0],
            "hostname": platform.node(),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dx-divisor", type=int, choices=(1, 2, 4), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--git-sha", default=None,
                    help="provenance override when the tree has no .git "
                         "(tarball fetch); default: git rev-parse HEAD")
    args = ap.parse_args()

    print(f"rfx from {rfx.__file__}; jax {jax.__version__} "
          f"backend {jax.default_backend()} x64 {jax.config.x64_enabled}")
    out = run_rung(args.dx_divisor, args.git_sha)

    # Persist first (feedback_persist_before_the_optional_stage), then print.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    os.replace(tmp, args.output)
    print(f"wrote {args.output}")

    sv = out["singular_values"]
    with np.printoptions(precision=6, suppress=True):
        print(f"dx/{args.dx_divisor}: dx={out['rung']['dx_m']*1e3:.4f} mm "
              f"cpml={out['rung']['cpml_layers']} n_steps={out['rung']['n_steps']} "
              f"cells={out['rasterization']['n_cells_total']}")
        print(f"sv_max per bin = {np.array(sv['max_per_bin'])}")
        print(f"sv_max = {sv['sv_max']:.7f} at {sv['sv_max_freq_hz']/1e9:.2f} GHz; "
              f"excess@3GHz = {sv['excess_3ghz']:+.7f}; "
              f"delta vs battery 1.003227 = {sv['delta_vs_battery_sv_max']:+.2e}")
        print(f"|S21| = {np.array(out['abs_s']['s21'])}")
        print(f"|S11| = {np.array(out['abs_s']['s11'])}")
        print(f"settling_db per drive = "
              f"{[d['settling_db'] for d in out['settling']['per_drive']]}; "
              f"main pass = {out['settling']['main_pass_settling_db']}")
        print(f"preflight codes = {out['preflight']['codes']} "
              f"(battery codes present: {out['preflight']['battery_codes_present']})")
        print(f"wall time: run {out['wall_time_s']['run']:.1f} s, "
              f"total {out['wall_time_s']['total']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
