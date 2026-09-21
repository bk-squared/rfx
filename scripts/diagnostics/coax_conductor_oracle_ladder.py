#!/usr/bin/env python3
"""The oracle's four comparisons as a mesh ladder, on two boards.

``tests/oracle/test_coax_conductor_realization.py`` asserts a 1 % bar on ONE
mesh. The v2 accuracy bar does not allow that: "A comparison made on a mesh
that has not been shown to converge (at least two refinements, trend stated)
says nothing either way — not a pass, not a defect." So the gate is not
entitled to its reading in either direction, and this supplies the refinement
it skipped.

It also separates two things the gate's fixture confounds. That fixture is the
committed AD gate's board — 8x8x12 mm, 3 probes, 13 bins over 6-12 GHz — and it
realizes **3.789 annulus cells**, below the lane's own "four or more"
recommendation. The diagnostic that produced the before-numbers used a
different board: 8x8x60 mm, 12 probes, 81 bins over 4-12 GHz. Running the same
rungs on BOTH boards tells the mesh apart from the board; running only the
long one would not.

Pre-declared, before any solve, as three distinguishable outcomes:

* **A** — the comparisons come inside their bars as the mesh refines, on both
  boards, and 3.789 is the rung that sits outside. Then the trend is in the
  mesh.
* **B** — they come inside on the long board at every rung including 3.789, and
  stay outside on the 12 mm board at every rung. Then the trend is in the
  board, not the mesh.
* **C** — they stay outside their bars at every rung on both boards, or they
  cross in and back out with refinement. Then neither board's number is a
  converged one and the quantity has not been shown to settle.

**LEADER TO FILL** — what the outcome means, and what the gate's fixture
should be. This script writes numbers; it changes no bar, no test and no
shipped code.

Usage (one case per invocation; the generator shards them)::

    PYTHONPATH=. python scripts/diagnostics/coax_conductor_oracle_ladder.py \\
        --lane thru --board thru_long --rung 6 --out <run-dir> --run-id <id>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax  # noqa: E402

import rfx  # noqa: E402
from rfx.api import Simulation  # noqa: E402
from rfx.sources.coaxial_port import (  # noqa: E402
    PTFE_EPS_R, coaxial_tem_characteristic_impedance,
)
from rfx.sources.sources import GaussianPulse  # noqa: E402

sys.path.insert(0, str(REPO / "tests" / "oracle"))
from test_coax_conductor_realization import (  # noqa: E402
    BETA_FRAC, COLUMN_POWER_MAX, COLUMN_POWER_MIN, C0, Z0_FRAC,
    _beta_from_s21_phase,
)

FREQ_MAX = 40.0e9
CPML_LAYERS = 16
DRIVE = dict(f0=8.0e9, bandwidth=1.2)
RECORD_UNITS = 12.0          # the battery's own default record length

# The rung the gate's board realizes when the cell size is left to freq_max.
# Carried as a rung so the gate's own fixture lands on the same trend line.
GATE_RUNG = 3.789288121451007

BOARDS = {
    # The diagnostic's boards — the ones the arm 0-5 numbers were measured on.
    "thru_long": dict(
        lane="thru", domain=(0.008, 0.008, 0.060),
        probes=dict(probe_count=12, probe_start_cells=8, probe_spacing_cells=4),
        freqs=np.linspace(4.0e9, 12.0e9, 81)),
    "load_long": dict(
        lane="load", domain=(0.008, 0.008, 0.040), probes=dict(probe_count=12),
        freqs=np.linspace(4.0e9, 12.0e9, 81)),
    # The gate's own boards — the committed AD fixtures the oracle borrowed.
    "thru_gate": dict(
        lane="thru", domain=(0.008, 0.008, 0.012),
        probes=dict(probe_count=3, probe_start_cells=4, probe_spacing_cells=2),
        freqs=np.linspace(6.0e9, 12.0e9, 13)),
    "load_gate": dict(
        lane="load", domain=(0.008, 0.008, 0.020), probes=dict(probe_count=9),
        freqs=np.linspace(6.0e9, 12.0e9, 13)),
}


def port_radii() -> tuple[float, float]:
    """The registered port's radii, from the API's own defaults."""
    sim = Simulation(freq_max=FREQ_MAX, domain=(0.008, 0.008, 0.060),
                     boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.030), face="top", pin_length=5.0e-3)
    p = sim._coaxial_ports[0]
    return float(p.pin_radius), float(p.outer_radius)


def dx_of(rung: float) -> float:
    a, b = port_radii()
    return (b - a) / float(rung)


def record_steps(grid, units: float) -> int:
    """``units`` one-way traversals of the z extent at the fill's phase
    velocity, in whole timesteps rounded up to a multiple of 100. The battery's
    own rule, so the same PHYSICAL record is used at every rung."""
    nz_phys = int(grid.shape[2]) - int(grid.pad_z_lo) - int(grid.pad_z_hi)
    lz = nz_phys * float(grid.dx)
    v = C0 / math.sqrt(float(PTFE_EPS_R))
    return int(math.ceil(units * lz / v / float(grid.dt) / 100.0) * 100)


def build(board: str, rung: float) -> Simulation:
    spec = BOARDS[board]
    domain = spec["domain"]
    sim = Simulation(freq_max=FREQ_MAX, domain=domain, boundary="cpml",
                     cpml_layers=CPML_LAYERS, dx=dx_of(rung))
    sim.add_coaxial_port((domain[0] / 2.0, domain[1] / 2.0, domain[2] / 2.0),
                         face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(**DRIVE))
    return sim


def measure_thru(board: str, rung: float) -> dict:
    spec = BOARDS[board]
    freqs_in = spec["freqs"]
    sim = build(board, rung)
    grid = sim._build_grid()
    n_steps = record_steps(grid, RECORD_UNITS)
    res = sim.compute_coaxial_two_port(n_steps=n_steps, freqs=freqs_in,
                                       **spec["probes"])
    freqs = np.asarray(res.freqs, dtype=float)
    S = np.asarray(res.s_params)
    planes = np.asarray(res.reference_planes, dtype=float)
    l12 = float(abs(planes[0] - planes[1]))
    beta_analytic = 2.0 * np.pi * freqs * math.sqrt(float(PTFE_EPS_R)) / C0

    # Both estimators, both reported whatever either one reads: the gate stops
    # at the first assertion and so never printed the second.
    try:
        beta_phase, v_p = _beta_from_s21_phase(freqs, S[1, 0, :], l12)
        phase_worst = float(np.max(np.abs(beta_phase / beta_analytic - 1.0)))
        phase_note = None
    except AssertionError as exc:          # the unwrapping guard, not a solve failure
        phase_worst, v_p, phase_note = float("nan"), float("nan"), str(exc)

    gamma = np.asarray(res.gamma)
    beta_fit = np.imag(gamma)
    beta_fit = beta_fit.mean(axis=tuple(range(beta_fit.ndim - 1)))
    pencil_worst = float(np.max(np.abs(beta_fit / beta_analytic - 1.0)))

    col = np.sum(np.abs(S) ** 2, axis=0)

    # The probe array's own span, which a dx ladder shortens because the lane
    # places planes by cell index (rfx/sparams/coax.py). Recorded per rung so a
    # replay can see how much phase the estimators actually had to work with.
    probes = spec["probes"]
    span_cells = int(probes.get("probe_spacing_cells", 0)) * (
        int(probes.get("probe_count", 1)) - 1)
    span_m = span_cells * float(grid.dx)
    f_mid = float(freqs[len(freqs) // 2])
    span_rad = 2.0 * np.pi * f_mid * math.sqrt(float(PTFE_EPS_R)) / C0 * span_m

    out = {
        "n_steps": n_steps, "record_units": RECORD_UNITS,
        "reference_plane_separation_m": l12,
        "probe_span_cells": span_cells,
        "probe_span_m": span_m,
        "probe_span_rad_at_band_centre": float(span_rad),
        "probe_span_wavelengths_at_band_centre": float(span_rad / (2.0 * np.pi)),
        "band_centre_hz": f_mid,
        # The complex S itself, so a replay re-derives every comparison rather
        # than restating the numbers this script already computed.
        "s_params_real": np.real(S).astype(float).tolist(),
        "s_params_imag": np.imag(S).astype(float).tolist(),
        "annulus_cells": float(getattr(res, "annulus_cells", float("nan"))),
        "beta_ratio_s21_phase_worst": phase_worst,
        "beta_ratio_matrix_pencil_worst": pencil_worst,
        "s21_phase_note": phase_note,
        "phase_velocity_m_per_s": v_p,
        "phase_velocity_analytic_m_per_s": C0 / math.sqrt(float(PTFE_EPS_R)),
        "max_column_power": float(col.max()),
        "min_column_power": float(col.min()),
        "abs_s21": np.abs(S[1, 0, :]).astype(float).tolist(),
        "abs_s11": np.abs(S[0, 0, :]).astype(float).tolist(),
        "freqs_hz": freqs.tolist(),
        "status": str(res.status),
        "settling_db": np.asarray(res.settling_db, dtype=float).tolist(),
    }
    out["beta_within_bar"] = bool(
        np.isfinite(phase_worst) and phase_worst <= BETA_FRAC
        and pencil_worst <= BETA_FRAC)
    out["column_power_within_bar"] = bool(
        out["max_column_power"] <= COLUMN_POWER_MAX
        and out["min_column_power"] >= COLUMN_POWER_MIN)
    return out


def measure_load(board: str, rung: float, load_ohm: float) -> dict:
    spec = BOARDS[board]
    sim = build(board, rung)
    grid = sim._build_grid()
    n_steps = record_steps(grid, RECORD_UNITS)
    res = sim.compute_coaxial_line_reflection(
        termination="matched", dut_impedance=load_ohm, n_steps=n_steps,
        freqs=spec["freqs"], **spec["probes"])
    a, b = port_radii()
    declared = coaxial_tem_characteristic_impedance(a, b, float(PTFE_EPS_R))
    z0 = np.asarray(res.z0_numerical_ohm)
    finite = np.isfinite(np.real(z0))
    measured = float(np.median(np.real(z0)[finite])) if finite.any() else float("nan")
    frac = abs(measured - declared) / declared
    s11 = np.asarray(res.s11)
    return {
        "n_steps": n_steps, "record_units": RECORD_UNITS,
        "load_ohm": float(load_ohm),
        "annulus_cells": float(getattr(res, "annulus_cells", float("nan"))),
        "z0_ohm": measured, "declared_z_tem_ohm": declared,
        "frac_vs_declared": frac, "within_bar": bool(frac <= Z0_FRAC),
        "n_finite_bins": int(finite.sum()), "n_bins": int(finite.size),
        # Complex Gamma and the per-bin Z0, so the replay re-derives the median
        # from the reflection rather than trusting the scalar above.
        "s11_real": np.real(s11).astype(float).tolist(),
        "s11_imag": np.imag(s11).astype(float).tolist(),
        "z0_per_bin_real": np.real(z0).astype(float).tolist(),
        "freqs_hz": np.asarray(res.freqs, dtype=float).tolist(),
        "abs_gamma": np.abs(s11).astype(float).tolist(),
        "status": str(getattr(res, "status", "")),
        "settling_db": np.asarray(getattr(res, "settling_db", []),
                                  dtype=float).tolist(),
    }


def assemble(src_dir: Path, out_path: Path) -> int:
    """Merge the per-rung run records into the one committed record.

    Every case ran as its own VESSL job (wall time is the largest case, not
    their sum), so the record is assembled from their JSONs rather than
    produced by one long job. The assembler refuses a mixed-commit set: a
    record whose rungs came from different trees would compare a ladder
    against itself.
    """
    files = sorted(src_dir.rglob("ladder_*.json"))
    if not files:
        raise SystemExit(f"no ladder_*.json under {src_dir}")
    cases, commits = [], set()
    for f in files:
        d = json.loads(f.read_text())
        if "measured" not in d:
            print(f"  skipping {f.name}: no measurement (the job died before "
                  "its solve returned)")
            continue
        commits.add(d["commit"])
        cases.append(d)
    if len(commits) != 1:
        raise SystemExit(
            f"the records span {len(commits)} commits {sorted(commits)}; a "
            "ladder assembled across trees compares nothing")
    for c in cases:
        m = c["measured"]
        if "s_params_real" not in m:
            continue
        S = (np.asarray(m["s_params_real"], dtype=float)
             + 1j * np.asarray(m["s_params_imag"], dtype=float))
        freqs = np.asarray(m["freqs_hz"], dtype=float)
        col = np.sum(np.abs(S) ** 2, axis=0).min(axis=0)
        k = int(np.argmin(col))
        s21_db = 20.0 * np.log10(np.abs(S[1, 0, :]))
        s11 = np.abs(S[0, 0, :])
        m["min_column_power_bin_hz"] = float(freqs[k])
        m["min_column_power_bin_index"] = k
        m["worst_s21_db"] = float(s21_db.min())
        m["worst_s21_db_bin_hz"] = float(freqs[int(np.argmin(s21_db))])
        m["worst_s11_db"] = float(20.0 * np.log10(max(float(s11.max()), 1e-30)))
        m["n_bins_below_column_power_min"] = int((col < COLUMN_POWER_MIN).sum())

    first = cases[0]
    rec = {
        "schema": "rfx.coax_conductor_long_board_record", "schema_version": 1,
        "commit": first["commit"],
        "what": ("the coaxial line's two constants as a mesh ladder on the "
                 "diagnostic's own boards, under the PEC-edge conductor "
                 "realization; the fast replay re-derives every comparison "
                 "from the complex S stored here"),
        "bars": first["bars"],
        "jax_version": first["jax_version"],
        "numpy_version": first["numpy_version"],
        "python": first["python"],
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "cases": sorted(cases, key=lambda c: (c["board"], c.get("measured", {})
                                              .get("load_ohm", 0.0),
                                              c["rung_annulus_cells"])),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1))
    boards = sorted({c["board"] for c in cases})
    print(f"[assemble] {len(cases)} cases, boards {boards}, "
          f"commit {rec['commit'][:8]} -> {out_path}")
    for c in rec["cases"]:
        m = c["measured"]
        tag = (f"{c['board']} r{c['rung_annulus_cells']:.4g}"
               + (f" {m['load_ohm']:g}ohm" if "load_ohm" in m else ""))
        if "load_ohm" in m:
            print(f"  {tag:34s} Z0 {m['z0_ohm']:.3f} "
                  f"({m['frac_vs_declared']*100:.2f} %)")
        else:
            print(f"  {tag:34s} beta {m['beta_ratio_s21_phase_worst']*100:.2f} / "
                  f"{m['beta_ratio_matrix_pencil_worst']*100:.2f} %, "
                  f"power [{m['min_column_power']:.5f}, {m['max_column_power']:.5f}] "
                  f"(min at {m.get('min_column_power_bin_hz', float('nan'))/1e9:.2f} GHz), "
                  f"worst |S21| {m.get('worst_s21_db', float('nan')):.3f} dB, "
                  f"worst |S11| {m.get('worst_s11_db', float('nan')):.1f} dB, "
                  f"probe span {m['probe_span_rad_at_band_centre']:.3f} rad")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=sorted(BOARDS))
    ap.add_argument("--rung", type=float)
    ap.add_argument("--load-ohm", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--record-units", type=float, default=None,
                    help="one-way traversals of the z extent to record; the\n"
                         "default is the battery's 12. A finer mesh leaves the\n"
                         "box less settled at the same PHYSICAL record, so this\n"
                         "is how that is measured rather than argued.")
    ap.add_argument("--assemble", default=None,
                    help="merge every ladder_*.json under this directory into "
                         "the single committed record written to --out")
    args = ap.parse_args()

    if args.assemble:
        return assemble(Path(args.assemble), Path(args.out))
    global RECORD_UNITS
    if args.record_units is not None:
        RECORD_UNITS = float(args.record_units)
    if args.board is None or args.rung is None:
        raise SystemExit("--board and --rung are required unless --assemble")

    spec = BOARDS[args.board]
    if spec["lane"] == "load" and args.load_ohm is None:
        raise SystemExit("--load-ohm is required on a one-port board")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                  text=True).strip()
    a, b = port_radii()
    rec = {
        "schema": "rfx.coax_conductor_oracle_ladder", "schema_version": 1,
        "commit": sha, "run_id": args.run_id,
        "board": args.board, "lane": spec["lane"],
        "rung_annulus_cells": float(args.rung),
        "dx_m": dx_of(args.rung), "dx_um": dx_of(args.rung) * 1e6,
        "domain_m": list(spec["domain"]), "probes": dict(spec["probes"]),
        "freqs_hz": [float(spec["freqs"][0]), float(spec["freqs"][-1])],
        "n_freqs": int(spec["freqs"].size),
        "pin_radius_m": a, "outer_radius_m": b,
        "freq_max_hz": FREQ_MAX, "cpml_layers": CPML_LAYERS, "drive": DRIVE,
        "bars": {"beta_frac": BETA_FRAC, "column_power_max": COLUMN_POWER_MAX,
                 "z0_frac": Z0_FRAC},
        "rfx_file": str(Path(rfx.__file__).resolve()),
        "jax_version": jax.__version__, "numpy_version": np.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "python": sys.version.split()[0], "platform": platform.platform(),
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    tag = args.board if args.load_ohm is None else f"{args.board}-{args.load_ohm:g}"
    path = out_dir / f"ladder_{tag}_r{args.rung:g}.json"
    path.write_text(json.dumps(rec, indent=1))      # persisted BEFORE the solve

    if spec["lane"] == "thru":
        rec["measured"] = m = measure_thru(args.board, args.rung)
    else:
        rec["measured"] = m = measure_load(args.board, args.rung, args.load_ohm)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)

    head = f"[ladder] {args.board} rung {args.rung:g} (dx {rec['dx_um']:.2f} um)"
    if spec["lane"] == "thru":
        print(f"{head} realized {m['annulus_cells']:.4f} cells, "
              f"{m['n_steps']} steps, status {m['status']}, "
              f"settling {m['settling_db']}")
        print(f"{head} beta (S21 phase) {m['beta_ratio_s21_phase_worst']*100:.2f} %, "
              f"(pencil) {m['beta_ratio_matrix_pencil_worst']*100:.2f} % "
              f"-> within the {BETA_FRAC*100:.0f} % bar: {m['beta_within_bar']}")
        if m["s21_phase_note"]:
            print(f"{head} S21-phase estimate unavailable: {m['s21_phase_note']}")
        print(f"{head} column power [{m['min_column_power']:.5f}, "
              f"{m['max_column_power']:.5f}] -> within bar: "
              f"{m['column_power_within_bar']}")
    else:
        print(f"{head} realized {m['annulus_cells']:.4f} cells, "
              f"{m['n_steps']} steps, status {m['status']}")
        print(f"{head} {m['load_ohm']:g} ohm load: Z0 {m['z0_ohm']:.3f} ohm, "
              f"{m['frac_vs_declared']*100:.2f} % from the declared "
              f"{m['declared_z_tem_ohm']:.3f} -> within bar: {m['within_bar']}")
    print(f"[ladder] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
