#!/usr/bin/env python3
"""Microstrip chain battery — the measurement driver (reduced v2.0 form).

Runs the pre-declaration in
``docs/design_notes/msl_chain_battery_predeclaration.md`` against
``Simulation.compute_msl_s_matrix`` on a uniform mesh with ``mode="laplace"``
ports and the RAW S (``enforce_passivity=False``, the shipped default).

Every stage writes ONE JSON into ``--out``, persisted before anything optional
runs, and carries its own provenance: the commit (no fallback — the driver
refuses to write a record it cannot stamp), the ``rfx`` package path it
imported, library versions, the device, the realized geometry it asserted, the
verbatim preflight text, every warning, wall time and peak memory.

Stages::

    --stage solve    --dut {notch,thru} --rung {100,50,25}
    --stage pilot    --rung 100                 record-length choice
    --stage identity --rung 100                 forward identity, criterion 1(2)
    --stage adfd     --rung 100                 AD against a float64-loss FD
    --stage plane    --rung 50                  reference-plane invariance
    --assemble                                  arithmetic only, no FDTD

The assembler joins the stage JSONs into
``tests/fixtures/msl_chain_battery/fixture.json``; the replay test
``tests/oracle/test_msl_chain_battery.py`` re-derives every assembled number
from the stored S and compares it against the contract's bar.

This driver computes numbers. It writes no verdict sentence: where a reading
of the numbers belongs, the fixture carries the measurement and the
pre-declared threshold side by side and nothing else.

Usage (from a clean checkout; the ``rfx`` import must resolve to this tree)::

    PYTHONPATH=. python scripts/diagnostics/msl_chain_battery_measure.py \
        --stage solve --dut notch --rung 25 --out <run-dir> --run-id <id>
    PYTHONPATH=. python scripts/diagnostics/msl_chain_battery_measure.py \
        --assemble --out <run-dir> \
        --fixture-out tests/fixtures/msl_chain_battery/fixture.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import rfx  # noqa: E402
from rfx import Box, Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.sources.sources import GaussianPulse  # noqa: E402
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff  # noqa: E402

from tests._x64_compat import enable_x64  # noqa: E402  SCOPED x64 only

SCHEMA = "rfx.msl_chain_battery"
SCHEMA_VERSION = 1
PREDECLARATION = "docs/design_notes/msl_chain_battery_predeclaration.md"
CONTRACT = "docs/design_notes/chain_closure_contract.md"
DRIVER = "scripts/diagnostics/msl_chain_battery_measure.py"
ARTIFACT = "tests/fixtures/msl_chain_battery/fixture.json"

C0 = 299792458.0

# ---------------------------------------------------------------------------
# The board — every patterned dimension a whole number of cells at 100 um, so
# no rung rounds a dimension. Pre-declaration, section "Board".
# ---------------------------------------------------------------------------

EPS_R = 3.66
H_SUB = 300e-6          # substrate thickness
W_TRACE = 600e-6        # trace width (zero-thickness PEC sheet)
W_STUB = 600e-6
L_STUB = 12e-3          # open-circuit stub, notch DUT only
L_LINE = 20e-3          # line length between the two port planes
PORT_MARGIN = 3e-3      # x margin beyond each port plane
MARGIN_LAT = 3e-3       # y margin below the trace and beyond the stub open end
AIR = 1.5e-3            # air above the substrate
CPML_LAYERS = 8

F_LO = 1e9
F_HI = 7e9
N_FREQS = 121
FREQS = np.linspace(F_LO, F_HI, N_FREQS)

RUNGS_UM = (100, 50, 25)
DUTS = ("notch", "thru")

# Record length. The pre-declaration fixes it "per rung from the settling
# witness of a short pilot at 100 um"; the pilot stage measures the witness
# against this ladder and the chosen value is written into every record.
PILOT_NUM_PERIODS = (10.0, 20.0, 30.0, 40.0)
DEFAULT_NUM_PERIODS = 20.0

# The drive. The shipped default for an MSL port is
# GaussianPulse(f0=freq_max/2, bandwidth=0.8); a differentiated Gaussian's
# spectrum is |S(f)| ~ f*exp(-(f/(f0*bw))^2), which at 7 GHz is ~39 dB below
# its own peak for that setting. The pilot measures both drives over the
# declared 1-7 GHz band and records the per-bin reliability of each; the one
# used by the battery is named in every record.
DRIVES = {
    "default": dict(f0=F_HI / 2.0, bandwidth=0.8),
    "wide": dict(f0=4e9, bandwidth=1.0),
}
DEFAULT_DRIVE = "default"

# Reference-plane displacement for stage 3(b), in cells at its own rung.
PLANE_SHIFT_CELLS = 10

# AD/FD stage. theta scales the substrate's eps_r region of eps_override.
AD_THETA0 = 1.0
AD_FD_H = 1e-3
AD_BAND = (3e9, 5e9)          # objective 1 averages |S21|^2 over this band
MIN_FD_ULP_SPAN = 1.0e4       # below this the FD reference resolves nothing

# The bar (contract, "The v2.0 battery for lumped/wire, MSL and coax").
# Recorded beside each measurement; never applied as a verdict here.
BAR = {
    "magnitude_db": 2.0,
    "frequency_frac": 0.01,
    "column_power_max": 1.02,
    "reciprocity": 0.02,
    "ad_fd_rel": 0.05,
    "identity_rtol": 1e-5,
    "identity_atol": 1e-7,
    "settling_db": -40.0,
}


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def git_sha() -> str:
    """The commit this tree is at. No fallback: a record that cannot name its
    own commit is not a record, so this raises rather than writing 'unknown'.
    """
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                      text=True, stderr=subprocess.PIPE).strip()
    except Exception as exc:  # noqa: BLE001 — re-raised immediately, see above
        raise RuntimeError(
            f"cannot read the commit of {REPO}: {exc}. Every record this driver "
            "writes is stamped with its commit and there is no fallback value; "
            "run from a git checkout (a copied tree needs "
            "`git config --global --add safe.directory`)."
        ) from exc
    if not out:
        raise RuntimeError(f"`git rev-parse HEAD` returned nothing in {REPO}")
    return out


def peak_memory() -> dict:
    """Host peak RSS and, where the backend reports one, device peak bytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    dev = {}
    for d in jax.devices():
        stats = getattr(d, "memory_stats", None)
        if stats is None:
            continue
        try:
            s = stats() or {}
        except Exception:  # noqa: BLE001 — a diagnostic, never fatal
            continue
        dev[str(d)] = {k: int(v) for k, v in s.items()
                       if isinstance(v, (int, float)) and "bytes" in k}
    return {"host_peak_rss_bytes": int(rss), "device": dev}


def provenance(args) -> dict:
    return {
        "commit": git_sha(),
        "run_id": args.run_id,
        "rfx_file": str(Path(rfx.__file__).resolve()),
        "rfx_version": getattr(rfx, "__version__", "?"),
        "repo": str(REPO),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _log(msg: str) -> None:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[msl-battery {stamp}] {msg}", flush=True)


def _log_witness(tag: str, res) -> None:
    """One readable line per solve while the job is still running: the record's
    settling, how much of the band the extractor called reliable, the worst
    passivity excess and the sampled |S21| minimum."""
    S = np.asarray(res.S)
    settling = np.asarray(res.settling_db) if res.settling_db is not None else np.array([np.nan])
    reliable = (float(np.mean(np.asarray(res.reliable))) if res.reliable is not None
                else float("nan"))
    excess = (float(np.max(res.sigma_max_excess)) if res.sigma_max_excess is not None
              else float("nan"))
    col = float(np.max(np.sum(np.abs(S) ** 2, axis=0)))
    k = int(np.argmin(np.abs(S[1, 0, :])))
    _log(f"{tag}: settling {np.array2string(settling, precision=2)} dB | reliable "
         f"{reliable*100:.1f} % | max sigma_max excess {excess:.4g} | max column power "
         f"{col:.5f} | min |S21| {abs(S[1, 0, k]):.5g} at "
         f"{np.asarray(res.freqs)[k]/1e9:.4f} GHz")


def _write(path: Path, obj: dict) -> None:
    """Persist atomically, BEFORE anything optional runs (printing is not
    persisting), then drop the compile cache so the next case does not inherit
    this one's XLA programs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False))
    os.replace(tmp, path)
    jax.clear_caches()
    _log(f"wrote {path}")


def _c(a) -> dict:
    """A complex array as JSON: real and imaginary parts, plus its shape."""
    arr = np.asarray(a)
    return {"shape": list(arr.shape),
            "real": np.real(arr).astype(float).tolist(),
            "imag": np.imag(arr).astype(float).tolist()}


def _f(a):
    if a is None:
        return None
    return np.asarray(a).astype(float).tolist()


def _b(a):
    if a is None:
        return None
    return np.asarray(a).astype(bool).tolist()


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------

def preflight_record(sim) -> dict:
    """The preflight report, verbatim text and structured findings.

    Captured with warnings silenced so the auto-mesh UserWarning does not land
    in the solve's own warning list twice; the findings themselves are the
    report's items, not warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    return {
        "n_findings": len(report),
        "ok": bool(report.ok),
        "text": [str(i) for i in report],
        "findings": [{"code": getattr(i, "code", "uncoded"),
                      "severity": getattr(i, "severity", "warning"),
                      "message": str(i)} for i in report],
    }


def _dedupe(wlist) -> list[dict]:
    """Every warning, deduplicated by text and counted. Never suppressed."""
    seen: dict[str, int] = {}
    for w in wlist:
        key = f"{w.category.__name__}: {w.message}"
        seen[key] = seen.get(key, 0) + 1
    return [{"warning": k, "count": n} for k, n in seen.items()]


class _Captured:
    """Run a block with every warning recorded rather than shown once."""

    def __enter__(self):
        self._ctx = warnings.catch_warnings(record=True)
        self._list = self._ctx.__enter__()
        warnings.simplefilter("always")
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        self.warnings = _dedupe(self._list)
        self._ctx.__exit__(*exc)
        return False


# ---------------------------------------------------------------------------
# the board
# ---------------------------------------------------------------------------

def domain() -> tuple[float, float, float]:
    lx = L_LINE + 2 * PORT_MARGIN
    ly = MARGIN_LAT + W_TRACE + L_STUB + MARGIN_LAT
    lz = H_SUB + AIR
    return lx, ly, lz


def build_sim(dx: float, dut: str, *, drive: str = DEFAULT_DRIVE,
              probe_offset: int | None = None, probe_spacing: int | None = None,
              precision: str = "float32") -> Simulation:
    """The board of the pre-declaration at one cell size.

    The ground is the PEC wall at the bottom of the domain — on the uniform
    lane that is ``Boundary(lo="pec")`` on z, which is the same object the
    non-uniform lane spells ``pec_faces={"zlo"}``: a wall, so the return
    current is not carried by a finite sheet.

    The trace and the stub are ZERO-THICKNESS PEC sheets (equal z corners).
    Drawn with unequal corners the same Box is a volume and realizes a
    cell-thick slab with walls at two z planes, which is not this board.
    """
    if dut not in DUTS:
        raise ValueError(f"unknown dut {dut!r}; expected one of {DUTS}")
    lx, ly, lz = domain()
    sim = Simulation(
        freq_max=F_HI, domain=(lx, ly, lz), dx=dx, precision=precision,
        cpml_layers=CPML_LAYERS,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("substrate", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, H_SUB)), material="substrate")

    trace_y_lo = MARGIN_LAT
    trace_y_hi = MARGIN_LAT + W_TRACE
    sim.add(Box((0.0, trace_y_lo, H_SUB), (lx, trace_y_hi, H_SUB)), material="pec")

    if dut == "notch":
        xc = lx / 2.0
        sim.add(Box((xc - W_STUB / 2.0, trace_y_hi, H_SUB),
                    (xc + W_STUB / 2.0, trace_y_hi + L_STUB, H_SUB)),
                material="pec")

    y_centre = 0.5 * (trace_y_lo + trace_y_hi)
    wf = GaussianPulse(**DRIVES[drive])
    probe_kw = {}
    if probe_offset is not None:
        probe_kw["n_probe_offset"] = int(probe_offset)
    if probe_spacing is not None:
        probe_kw["n_probe_spacing"] = int(probe_spacing)
    sim.add_msl_port(position=(PORT_MARGIN, y_centre, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0,
                     waveform=wf, **probe_kw)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_centre, 0.0), width=W_TRACE,
                     height=H_SUB, direction="-x", impedance=50.0,
                     waveform=wf, **probe_kw)
    return sim


def declared(dx: float, dut: str) -> dict:
    lx, ly, lz = domain()
    z0_hj, eps_eff = hammerstad_jensen_z0_eps_eff(W_TRACE, H_SUB, EPS_R)
    rec = {
        "dut": dut,
        "dx_m": dx,
        "dx_um": dx * 1e6,
        "eps_r": EPS_R,
        "h_sub_m": H_SUB,
        "w_trace_m": W_TRACE,
        "l_line_m": L_LINE,
        "port_margin_m": PORT_MARGIN,
        "lateral_margin_m": MARGIN_LAT,
        "air_m": AIR,
        "domain_m": [lx, ly, lz],
        "cpml_layers": CPML_LAYERS,
        "substrate_cells_under_strip": H_SUB / dx,
        "trace_cells_across": W_TRACE / dx,
        "port_plane_x_m": [PORT_MARGIN, PORT_MARGIN + L_LINE],
        "freqs_hz": [F_LO, F_HI, N_FREQS],
        "hj_z0_ohm": z0_hj,
        "hj_eps_eff": eps_eff,
    }
    if dut == "notch":
        rec["w_stub_m"] = W_STUB
        rec["l_stub_m"] = L_STUB
        rec["stub_x_centre_m"] = lx / 2.0
        rec["stub_cells_long"] = L_STUB / dx
        rec["f_notch_analytic_hz"] = C0 / (4.0 * L_STUB * math.sqrt(eps_eff))
    return rec


# ---------------------------------------------------------------------------
# realized geometry — measured from the ONE realization function, before any
# FDTD step. Imitates validation/crossval/06b_msl_notch_filter_uniform.py's
# realized_metal / assert_realized_metal on this board (no import: that script
# pins a different board and enforce_passivity=True).
# ---------------------------------------------------------------------------

def realized_geometry(sim: Simulation, dx: float, dut: str) -> dict:
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid

    grid = sim._build_grid()
    sheets: list = []
    wires: list = []
    assembled = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
    materials, pec_mask = assembled[0], assembled[3]
    if pec_mask is None and not sheets and not wires:
        raise RuntimeError("realized_geometry: this build has no conductor at all")
    edges = realized_pec_edge_masks(pec_mask, sheets=tuple(sheets), wires=tuple(wires),
                                    periodic=sim._periodic_flags())
    mx, my, _mz = (np.asarray(e) for e in edges)
    gc = coords_from_uniform_grid(grid)
    nodes = (np.asarray(gc.x), np.asarray(gc.y), np.asarray(gc.z))

    planes = realized_wall_planes(edges, 2)
    if len(planes) != 1:
        raise RuntimeError(
            "realized_geometry: the metal of this board is zero-thickness sheets on "
            f"ONE node plane, but the realization has tangential walls at z planes "
            f"{planes} — two planes is what a cell-thick volume trace realizes")
    k = int(planes[0])

    ex_per_row = mx[:, :, k].sum(axis=0)
    if not ex_per_row.any():
        raise RuntimeError("realized_geometry: no Ex edge on the sheet plane — no "
                           "metal runs along the propagation axis")
    trace_rows = np.flatnonzero(ex_per_row == ex_per_row.max())
    if trace_rows.size < 2 or trace_rows.max() - trace_rows.min() + 1 != trace_rows.size:
        raise RuntimeError(
            f"realized_geometry: the longest Ex rows are {trace_rows.tolist()} — the "
            "main line does not realize as one contiguous block of rows")
    j0, j1 = int(trace_rows.min()), int(trace_rows.max())
    n_rows = j1 - j0 + 1

    # The substrate under the strip, counted as CELLS from the realized eps
    # array along the column through the trace centre.
    eps = np.asarray(materials.eps_r)
    i_mid = eps.shape[0] // 2
    j_mid = (j0 + j1) // 2
    col = eps[i_mid, j_mid, :]
    sub_cells = int(np.count_nonzero(col > 1.0 + 1e-6))

    rec = {
        "grid_shape": [int(s) for s in grid.shape],
        "n_cells": int(np.prod(grid.shape)),
        "dt_s": float(grid.dt),
        "sheet_plane_k": k,
        "sheet_plane_z_m": float(nodes[2][k]),
        "n_pec_sheets": len(sheets),
        "n_pec_volume_cells": 0 if pec_mask is None else int(np.asarray(pec_mask).sum()),
        "trace_rows": [j0, j1],
        "trace_n_rows": n_rows,
        "trace_y_m": [float(nodes[1][j0]), float(nodes[1][j1])],
        # GEOMETRIC width: the node span the sheet's longitudinal edges occupy.
        "trace_w_geometric_m": float(nodes[1][j1] - nodes[1][j0]),
        # ELECTRICAL width: n_rows * dx, the quasi-TEM filament convention.
        "trace_w_electrical_m": n_rows * dx,
        "substrate_cells_under_strip": sub_cells,
        "n_ex_edges": int(mx.sum()),
        "n_ey_edges": int(my.sum()),
    }

    if dut == "notch":
        stub_cols = np.flatnonzero(my[:, j1:, k].any(axis=1))
        if stub_cols.size == 0:
            raise RuntimeError("realized_geometry: no Ey edge above the main line — "
                               "the stub is not connected to the trace")
        i0, i1 = int(stub_cols.min()), int(stub_cols.max())
        stub_edges = np.flatnonzero(my[i0:i1 + 1, :, k].any(axis=0))
        stub_open = int(stub_edges.max()) + 1
        y_centre = 0.5 * (float(nodes[1][j0]) + float(nodes[1][j1]))
        rec.update({
            "stub_cols": [i0, i1],
            "stub_n_cols": i1 - i0 + 1,
            "stub_x_m": [float(nodes[0][i0]), float(nodes[0][i1])],
            "stub_w_geometric_m": float(nodes[0][i1] - nodes[0][i0]),
            "stub_w_electrical_m": (i1 - i0 + 1) * dx,
            "stub_len_m": float(nodes[1][stub_open]) - float(nodes[1][j1]),
            "stub_len_centreline_m": float(nodes[1][stub_open]) - y_centre,
            "stub_open_node_j": stub_open,
        })
    else:
        if my[:, j1:, k].any():
            raise RuntimeError("realized_geometry: the thru control realized Ey edges "
                               "above the main line — it has a stub")

    entries = sim._resolve_msl_probe_entries(grid)
    ports = []
    for e in entries:
        axis = 0 if e.direction[1] == "x" else 1
        sgn = 1.0 if e.direction[0] == "+" else -1.0
        feed = e.position[axis]
        xs = [feed + sgn * (e.n_probe_offset + n * e.n_probe_spacing) * dx
              for n in range(e.n_probes)]
        ports.append({
            "name": e.name,
            "direction": e.direction,
            "feed_plane_m": float(feed),
            "n_probe_offset": int(e.n_probe_offset),
            "n_probe_spacing": int(e.n_probe_spacing),
            "n_probes": int(e.n_probes),
            "probe_planes_m": [float(x) for x in xs],
            "waveform": {"f0": float(e.waveform.f0), "bandwidth": float(e.waveform.bandwidth),
                         "amplitude": float(e.waveform.amplitude),
                         "cutoff": float(e.waveform.cutoff)},
        })
    rec["ports"] = ports
    return rec


def assert_realized(sim: Simulation, dx: float, dut: str) -> dict:
    """Refuse to solve unless the realized board IS the declared board.

    No FDTD step runs here. A PEC sheet is solved about 0.35 cell wider at each
    free edge, so the tolerance on a width is one cell; what this catches is a
    lost or gained row, a sheet snapped to the wrong node plane, a foil
    re-declared as a volume, and a mesh on which h_sub stops landing on a node.
    """
    m = realized_geometry(sim, dx, dut)
    d = declared(dx, dut)
    problems = []
    if m["n_pec_sheets"] != (2 if dut == "notch" else 1):
        problems.append(f"{m['n_pec_sheets']} PEC sheet(s) classified, expected "
                        f"{2 if dut == 'notch' else 1}")
    if m["n_pec_volume_cells"]:
        problems.append(f"{m['n_pec_volume_cells']} PEC VOLUME cell(s) realized; both "
                        "metal entries must be zero-thickness sheets")
    if abs(m["sheet_plane_z_m"] - H_SUB) > 1e-12:
        problems.append(f"sheet plane at z={m['sheet_plane_z_m']*1e6:.4f} um, declared "
                        f"substrate top {H_SUB*1e6:.4f} um")
    if m["substrate_cells_under_strip"] != round(H_SUB / dx):
        problems.append(f"{m['substrate_cells_under_strip']} substrate cells under the "
                        f"strip, declared {round(H_SUB / dx)}")
    if abs(m["trace_w_geometric_m"] - W_TRACE) > dx:
        problems.append(f"realized trace width {m['trace_w_geometric_m']*1e6:.2f} um is "
                        f"more than one cell ({dx*1e6:.1f} um) from the declared "
                        f"{W_TRACE*1e6:.1f} um")
    if dut == "notch":
        if abs(m["stub_w_geometric_m"] - W_STUB) > dx:
            problems.append(f"realized stub width {m['stub_w_geometric_m']*1e6:.2f} um is "
                            f"more than one cell from the declared {W_STUB*1e6:.1f} um")
        if abs(m["stub_len_m"] - L_STUB) > dx:
            problems.append(f"realized stub length {m['stub_len_m']*1e6:.2f} um is more "
                            f"than one cell from the declared {L_STUB*1e6:.1f} um")
        if m["trace_n_rows"] != m["stub_n_cols"]:
            problems.append(f"trace realizes {m['trace_n_rows']} rows and the stub "
                            f"{m['stub_n_cols']} — they are declared the same width")
    for p in m["ports"]:
        if abs(p["feed_plane_m"] - round(p["feed_plane_m"] / dx) * dx) > 1e-12:
            problems.append(f"port {p['name']} feed plane {p['feed_plane_m']} m is not on "
                            "a node line")
    if problems:
        raise RuntimeError(
            "assert_realized: the realized board is not the declared board — refusing "
            "to solve. " + "; ".join(problems) + f" [declared: {d}] [measured: {m}]")
    return m


# ---------------------------------------------------------------------------
# solving
# ---------------------------------------------------------------------------

def _result_record(res) -> dict:
    """Everything the S-matrix result carries that a later reader needs."""
    pc = []
    for c in (res.probe_clearance or ()):
        pc.append({k: (float(v) if isinstance(v, (int, float, np.floating)) else
                       (list(map(float, v)) if isinstance(v, (list, tuple)) else str(v)))
                   for k, v in vars(c).items()})
    return {
        "freqs_hz": _f(res.freqs),
        "S": _c(res.S),
        "port_names": list(res.port_names),
        "Z0": _c(res.Z0),
        "beta": _c(res.beta),
        "reference_impedances": _f(res.reference_impedances),
        "reliable": _b(res.reliable),
        "settling_db": _f(res.settling_db),
        "sigma_max_excess": _f(res.sigma_max_excess),
        "cond_a": _f(res.cond_a),
        "beta_railed": _b(res.beta_railed),
        "assembly": res.assembly,
        "probe_clearance": pc,
        "passivity_correction": _f(res.passivity_correction),
        "S_raw": None if res.S_raw is None else _c(res.S_raw),
    }


def solve(sim, *, num_periods: float, freqs=None, **kw):
    return sim.compute_msl_s_matrix(
        freqs=jnp.asarray(FREQS if freqs is None else freqs),
        num_periods=float(num_periods), **kw)


def _closest_divisor(n: int, target: int) -> int:
    """Divisor of ``n`` nearest ``target`` — checkpoint_segments must divide
    n_steps exactly (padding would shift the DFT accumulator windows)."""
    best = 1
    for d in range(1, int(n ** 0.5) + 1):
        if n % d == 0:
            for cand in (d, n // d):
                if abs(cand - target) < abs(best - target):
                    best = cand
    return best


def _base(args, stage: str, dut: str | None, dx: float | None) -> dict:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "dut": dut,
        "rung_um": None if dx is None else dx * 1e6,
        "predeclaration": PREDECLARATION,
        "contract": CONTRACT,
        "driver": DRIVER,
        "bar": BAR,
        "provenance": provenance(args),
    }


# ---------------------------------------------------------------------------
# stage: cost estimate (printed before any solve)
# ---------------------------------------------------------------------------

def cost_estimate(dx: float, dut: str, num_periods: float) -> dict:
    sim = build_sim(dx, dut)
    grid = sim._build_grid()
    n = int(np.prod(grid.shape))
    n_steps = int(grid.num_timesteps(num_periods=num_periods))
    # Field carry (6 arrays) + the CPML psi slabs, which are n_layers deep on
    # each absorbing face rather than full arrays.
    nx, ny, nz = (int(s) for s in grid.shape)
    psi = 4 * CPML_LAYERS * (8 * nx * nz + 8 * nx * ny + 8 * ny * nz)
    fields = 6 * n * 4
    est = {
        "grid_shape": [nx, ny, nz],
        "n_cells": n,
        "dt_s": float(grid.dt),
        "num_periods": float(num_periods),
        "n_steps": n_steps,
        "n_drives": 2,
        "cell_steps": n * n_steps * 2,
        "field_bytes_f32": fields,
        "cpml_psi_bytes_f32": psi,
        "carry_bytes_f32": fields + psi,
        "forward_peak_estimate_bytes": 3 * (fields + psi) + 5 * n * 4,
    }
    print(f"[cost] dx={dx*1e6:.0f} um {dut}: {n:,} cells, {n_steps:,} steps x 2 drives "
          f"= {est['cell_steps']:.3e} cell-steps; carry "
          f"{est['carry_bytes_f32']/2**30:.3f} GiB; forward peak estimate "
          f"{est['forward_peak_estimate_bytes']/2**30:.3f} GiB", flush=True)
    return est


# ---------------------------------------------------------------------------
# stage: pilot
# ---------------------------------------------------------------------------

def stage_pilot(args, out: Path) -> None:
    dx = args.rung * 1e-6
    rec = _base(args, "pilot", "notch", dx)
    rec["declared"] = declared(dx, "notch")
    rec["num_periods_ladder"] = list(PILOT_NUM_PERIODS)
    rec["drives"] = {k: dict(v) for k, v in DRIVES.items()}
    rec["cases"] = []
    _write(out, rec)

    for drive in DRIVES:
        for npd in PILOT_NUM_PERIODS:
            sim = build_sim(dx, "notch", drive=drive)
            geo = assert_realized(sim, dx, "notch")
            pf = preflight_record(sim)
            est = cost_estimate(dx, "notch", npd)
            _log(f"pilot drive={drive} num_periods={npd}")
            with _Captured() as cap:
                res = solve(sim, num_periods=npd)
            _log_witness(f"pilot drive={drive} num_periods={npd}", res)
            case = {
                "drive": drive,
                "num_periods": float(npd),
                "realized": geo,
                "preflight": pf,
                "cost": est,
                "warnings": cap.warnings,
                "wall_s": cap.wall,
                "peak_memory": peak_memory(),
                "result": _result_record(res),
            }
            rec["cases"].append(case)
            _write(out, rec)          # persist after EVERY case


# ---------------------------------------------------------------------------
# stage: solve
# ---------------------------------------------------------------------------

def stage_solve(args, out: Path) -> None:
    dx = args.rung * 1e-6
    dut = args.dut
    sim = build_sim(dx, dut, drive=args.drive)
    geo = assert_realized(sim, dx, dut)
    pf = preflight_record(sim)
    est = cost_estimate(dx, dut, args.num_periods)

    rec = _base(args, "solve", dut, dx)
    rec.update({
        "drive": args.drive,
        "num_periods": float(args.num_periods),
        "declared": declared(dx, dut),
        "realized": geo,
        "preflight": pf,
        "cost": est,
    })
    _log(f"solve dut={dut} dx={dx*1e6:.0f} um num_periods={args.num_periods}")
    with _Captured() as cap:
        res = solve(sim, num_periods=args.num_periods)
    _log_witness(f"solve {dut} {dx*1e6:.0f} um", res)
    rec["warnings"] = cap.warnings
    rec["wall_s"] = cap.wall
    rec["peak_memory"] = peak_memory()
    rec["result"] = _result_record(res)
    _write(out, rec)


# ---------------------------------------------------------------------------
# stage: forward identity (criterion 1(2))
# ---------------------------------------------------------------------------

def _own_eps(sim, dtype=jnp.float32):
    """The simulation's OWN permittivity array, as the forward lane would build
    it. ``eps_override`` REPLACES the array, so handing this back is the no-op
    whose result must equal the untraced call's."""
    grid = sim._build_grid()
    materials = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    return jnp.asarray(np.asarray(materials.eps_r), dtype=dtype)


def stage_identity(args, out: Path) -> None:
    dx = args.rung * 1e-6
    sim = build_sim(dx, "notch", drive=args.drive)
    geo = assert_realized(sim, dx, "notch")
    pf = preflight_record(sim)
    cost_estimate(dx, "notch", args.num_periods)

    rec = _base(args, "identity", "notch", dx)
    rec.update({"drive": args.drive, "num_periods": float(args.num_periods),
                "declared": declared(dx, "notch"), "realized": geo, "preflight": pf})

    _log("identity: plain call")
    with _Captured() as cap_plain:
        res_plain = solve(sim, num_periods=args.num_periods)
    _log_witness("identity plain", res_plain)
    rec["plain"] = _result_record(res_plain)
    rec["plain_warnings"] = cap_plain.warnings
    rec["plain_wall_s"] = cap_plain.wall
    _write(out, rec)

    eps = _own_eps(sim)
    rec["eps_override"] = {"shape": [int(s) for s in eps.shape],
                           "dtype": str(eps.dtype),
                           "min": float(np.min(np.asarray(eps))),
                           "max": float(np.max(np.asarray(eps))),
                           "n_above_vacuum": int(np.count_nonzero(np.asarray(eps) > 1.0 + 1e-6))}
    _log("identity: no-op eps_override call")
    with _Captured() as cap_ov:
        res_ov = solve(sim, num_periods=args.num_periods, eps_override=eps)
    _log_witness("identity eps_override", res_ov)
    rec["override"] = _result_record(res_ov)
    rec["override_warnings"] = cap_ov.warnings
    rec["override_wall_s"] = cap_ov.wall

    a = np.asarray(res_plain.S)
    b = np.asarray(res_ov.S)
    d = np.abs(a - b)
    rec["difference"] = {
        "max_abs": float(d.max()),
        "max_abs_per_entry": [[float(d[i, j].max()) for j in range(d.shape[1])]
                              for i in range(d.shape[0])],
        "argmax_flat": int(np.argmax(d)),
        "max_rel": float((d / np.maximum(np.abs(a), 1e-300)).max()),
        "allclose_at_bar": bool(np.allclose(a, b, rtol=BAR["identity_rtol"],
                                            atol=BAR["identity_atol"])),
    }
    rec["peak_memory"] = peak_memory()
    _write(out, rec)


# ---------------------------------------------------------------------------
# stage: AD against a float64-loss FD
# ---------------------------------------------------------------------------

def _substrate_mask(sim) -> np.ndarray:
    grid = sim._build_grid()
    materials = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    eps = np.asarray(materials.eps_r)
    return eps > 1.0 + 1e-6


def _band_mean_s21_sq(S, freqs, lo, hi):
    """Band-mean |S21|^2 over [lo, hi]. A smooth scalar that depends on eps in
    every substrate cell, and is not passivity-pinned the way sum_ij|S_ij|^2
    is. The band mask is taken on the CONCRETE frequency grid, so it is a
    static index set under tracing."""
    m = (np.asarray(freqs) >= lo) & (np.asarray(freqs) <= hi)
    idx = jnp.asarray(np.flatnonzero(m))
    return jnp.mean(jnp.abs(S[1, 0, :][idx]) ** 2)


def _re_s21_s11_conj_at(S, k):
    return jnp.real(S[1, 0, k] * jnp.conj(S[0, 0, k]))


def _fd_ulp_span(f_plus: float, f_minus: float, dtype) -> float:
    """Resolving power of a central difference, in ULPs of ``dtype``.

    ``dtype`` is the dtype the LOSS was computed in, not the container the
    values arrived in: ``float(jnp_scalar)`` is always a Python float, so
    keying off the value alone measures float64 even for a float32 loss. Same
    expression as the gate in tests/unit/autodiff/test_msl_ad_fd_converged.py.
    """
    ulp = float(np.spacing(np.asarray(abs(0.5 * (f_plus + f_minus)), dtype=dtype)))
    return abs(f_plus - f_minus) / ulp


def stage_adfd(args, out: Path) -> None:
    dx = args.rung * 1e-6
    sim = build_sim(dx, "notch", drive=args.drive)
    geo = assert_realized(sim, dx, "notch")
    pf = preflight_record(sim)
    d = declared(dx, "notch")
    grid = sim._build_grid()
    n_steps = int(grid.num_timesteps(num_periods=args.num_periods))
    segments = _closest_divisor(n_steps, int(np.sqrt(n_steps)))
    assert n_steps % segments == 0
    k_notch = int(np.argmin(np.abs(FREQS - d["f_notch_analytic_hz"])))

    rec = _base(args, "adfd", "notch", dx)
    rec.update({
        "drive": args.drive,
        "num_periods": float(args.num_periods),
        "declared": d, "realized": geo, "preflight": pf,
        "theta0": AD_THETA0, "fd_h": AD_FD_H,
        "n_steps": n_steps, "checkpoint_segments": segments,
        "min_fd_ulp_span": MIN_FD_ULP_SPAN,
        "objectives": {
            "band_mean_s21_sq": {"band_hz": list(AD_BAND),
                                 "what": "mean over the band of |S21(f)|^2"},
            "re_s21_s11_conj_at_notch": {
                "bin_index": k_notch, "bin_hz": float(FREQS[k_notch]),
                "anchor": "the bin nearest the analytic quarter-wave notch",
                "what": "Re(S21 conj(S11)) at that bin"},
        },
        "cases": [],
    })

    eps32 = _own_eps(sim, jnp.float32)
    mask32 = jnp.asarray(_substrate_mask(sim))
    rec["design_variable"] = {
        "what": "theta multiplies eps_r in the substrate region of eps_override",
        "n_cells_scaled": int(np.count_nonzero(np.asarray(mask32))),
        "n_cells_total": int(np.asarray(mask32).size),
    }
    _write(out, rec)

    def _obj_factory(simx, epsx, maskx, which):
        def objective(theta):
            eps = jnp.where(maskx, epsx * theta, epsx)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = simx.compute_msl_s_matrix(freqs=jnp.asarray(FREQS),
                                              num_periods=float(args.num_periods),
                                              eps_override=eps,
                                              checkpoint_segments=segments)
            if which == "band_mean_s21_sq":
                return _band_mean_s21_sq(r.S, FREQS, *AD_BAND)
            return _re_s21_s11_conj_at(r.S, k_notch)
        return objective

    for which in ("band_mean_s21_sq", "re_s21_s11_conj_at_notch"):
        _log(f"adfd {which}: AD (float32)")
        obj32 = _obj_factory(sim, eps32, mask32, which)
        with _Captured() as cap_ad:
            loss, g = jax.value_and_grad(obj32)(jnp.float32(AD_THETA0))
        case = {
            "objective": which,
            "ad": {"loss": float(loss), "grad": float(g),
                   "loss_dtype": str(jnp.asarray(loss).dtype),
                   "wall_s": cap_ad.wall, "warnings": cap_ad.warnings},
        }
        rec["cases"].append(case)
        _write(out, rec)

        _log(f"adfd {which}: FD (float64 fields and loss, scoped x64)")
        with _Captured() as cap_fd:
            with enable_x64():
                sim64 = build_sim(dx, "notch", drive=args.drive, precision="float64")
                grid64 = sim64._build_grid()
                if (grid64.shape, grid64.dx, grid64.dt) != (grid.shape, grid.dx, grid.dt):
                    raise RuntimeError("the float64 referee did not build the same "
                                       "discrete rig as the float32 run")
                eps64 = _own_eps(sim64, jnp.float64)
                mask64 = jnp.asarray(_substrate_mask(sim64))
                obj64 = _obj_factory(sim64, eps64, mask64, which)

                def _loss64(theta):
                    arr = obj64(jnp.float64(theta))
                    if arr.dtype != jnp.float64:
                        raise RuntimeError(
                            f"the FD reference did not run in float64 (got {arr.dtype}): "
                            "JAX truncates a float64 request to float32 when x64 is off, "
                            "so the scoped context failed to engage. A float32 reference "
                            "resolves far too few ULPs to judge a gradient.")
                    return float(arr), arr.dtype

                f_plus, loss_dtype = _loss64(AD_THETA0 + AD_FD_H)
                f_minus, _ = _loss64(AD_THETA0 - AD_FD_H)
        g_fd = (f_plus - f_minus) / (2.0 * AD_FD_H)
        span = _fd_ulp_span(f_plus, f_minus, loss_dtype)
        interpretable = bool(span >= MIN_FD_ULP_SPAN)
        case["fd"] = {
            "f_plus": f_plus, "f_minus": f_minus, "h": AD_FD_H,
            "grad": g_fd, "loss_dtype": str(loss_dtype),
            "ulp_span": span, "wall_s": cap_fd.wall, "warnings": cap_fd.warnings,
        }
        # The resolving-power statement comes BEFORE the accuracy number, so a
        # comparator failure is recorded as a comparator failure.
        case["comparator"] = {"ulp_span": span, "floor": MIN_FD_ULP_SPAN,
                              "interpretable": interpretable}
        if interpretable:
            denom = max(abs(g_fd), 1e-300)
            case["rel_err"] = abs(float(g) - g_fd) / denom
        else:
            case["rel_err"] = None
            case["rel_err_note"] = "not interpretable: FD span below the ULP floor"
        _write(out, rec)

    rec["peak_memory"] = peak_memory()
    _write(out, rec)


# ---------------------------------------------------------------------------
# stage: reference-plane invariance
# ---------------------------------------------------------------------------

def stage_plane(args, out: Path) -> None:
    dx = args.rung * 1e-6
    base = build_sim(dx, "notch", drive=args.drive)
    entries = base._resolve_msl_probe_entries(base._build_grid())
    offsets = {int(e.n_probe_offset) for e in entries}
    spacings = {int(e.n_probe_spacing) for e in entries}
    if len(offsets) != 1 or len(spacings) != 1:
        raise RuntimeError(f"the two ports resolved different ladders: offsets {offsets}, "
                           f"spacings {spacings} — the shift would not be common")
    off0 = offsets.pop()
    spacing = spacings.pop()
    off1 = off0 + PLANE_SHIFT_CELLS

    rec = _base(args, "plane", "notch", dx)
    rec.update({
        "drive": args.drive, "num_periods": float(args.num_periods),
        "declared": declared(dx, "notch"),
        "shift_cells": PLANE_SHIFT_CELLS,
        "shift_m": PLANE_SHIFT_CELLS * dx,
        "n_probe_spacing": spacing,
        "n_probe_offsets": [off0, off1],
        "arms": [],
    })
    _write(out, rec)

    for tag, off in (("base", off0), ("shifted", off1)):
        sim = build_sim(dx, "notch", drive=args.drive, probe_offset=off,
                        probe_spacing=spacing)
        geo = assert_realized(sim, dx, "notch")
        pf = preflight_record(sim)
        cost_estimate(dx, "notch", args.num_periods)
        _log(f"plane arm={tag} n_probe_offset={off}")
        with _Captured() as cap:
            res = solve(sim, num_periods=args.num_periods)
        _log_witness(f"plane arm={tag}", res)
        rec["arms"].append({
            "tag": tag, "n_probe_offset": off,
            "realized": geo, "preflight": pf,
            "warnings": cap.warnings, "wall_s": cap.wall,
            "result": _result_record(res),
        })
        _write(out, rec)

    # The displacement each arm's probe planes actually moved, measured from
    # the realized positions rather than assumed from the cell count.
    p0 = rec["arms"][0]["realized"]["ports"]
    p1 = rec["arms"][1]["realized"]["ports"]
    rec["realized_plane_displacement_m"] = [
        abs(b["probe_planes_m"][0] - a["probe_planes_m"][0]) for a, b in zip(p0, p1)]
    rec["peak_memory"] = peak_memory()
    _write(out, rec)


# ---------------------------------------------------------------------------
# assemble — arithmetic only, no FDTD
# ---------------------------------------------------------------------------

def _S(rec_result) -> np.ndarray:
    d = rec_result["S"]
    return np.asarray(d["real"], dtype=float) + 1j * np.asarray(d["imag"], dtype=float)


def _beta(rec_result) -> np.ndarray:
    d = rec_result["beta"]
    return np.asarray(d["real"], dtype=float) + 1j * np.asarray(d["imag"], dtype=float)


def parabolic_min(freqs: np.ndarray, y: np.ndarray) -> dict:
    """Minimum of ``y`` over ``freqs``: the raw bin, and the vertex of the
    parabola through that bin and its two neighbours. Both are kept — the raw
    bin is what the grid can say, the interpolated value is an estimate whose
    error the grid bounds."""
    k = int(np.argmin(y))
    out = {"bin_index": k, "bin_hz": float(freqs[k]), "bin_value": float(y[k])}
    if 0 < k < len(y) - 1:
        y0, y1, y2 = float(y[k - 1]), float(y[k]), float(y[k + 1])
        denom = y0 - 2.0 * y1 + y2
        delta = 0.0 if denom == 0.0 else 0.5 * (y0 - y2) / denom
        step = float(freqs[k + 1] - freqs[k])
        out["interp_hz"] = float(freqs[k]) + delta * step
        out["interp_delta_bins"] = float(delta)
        out["interp_value"] = y1 - 0.25 * (y0 - y2) * delta
    else:
        out["interp_hz"] = float(freqs[k])
        out["interp_delta_bins"] = 0.0
        out["interp_value"] = float(y[k])
        out["at_band_edge"] = True
    return out


def crossing_width(freqs: np.ndarray, y_db: np.ndarray, level_db: float,
                   k: int) -> dict:
    """Width of the contiguous run around bin ``k`` where ``y_db <= level_db``,
    with the two edges linearly interpolated between the bracketing bins."""
    n = len(y_db)
    if y_db[k] > level_db:
        return {"level_db": level_db, "width_hz": 0.0, "lo_hz": None, "hi_hz": None}
    lo = k
    while lo > 0 and y_db[lo - 1] <= level_db:
        lo -= 1
    hi = k
    while hi < n - 1 and y_db[hi + 1] <= level_db:
        hi += 1

    def _edge(i_in, i_out):
        if i_out < 0 or i_out >= n:
            return None
        a, b = float(y_db[i_in]), float(y_db[i_out])
        if a == b:
            return float(freqs[i_in])
        t = (level_db - a) / (b - a)
        return float(freqs[i_in]) + t * (float(freqs[i_out]) - float(freqs[i_in]))

    f_lo = _edge(lo, lo - 1)
    f_hi = _edge(hi, hi + 1)
    width = None if (f_lo is None or f_hi is None) else f_hi - f_lo
    return {"level_db": level_db, "lo_hz": f_lo, "hi_hz": f_hi,
            "lo_bin": lo, "hi_bin": hi, "width_hz": width,
            "clipped_at_band_edge": bool(f_lo is None or f_hi is None)}


def power_metrics(S: np.ndarray) -> dict:
    """Column power, its excess over the bar, reciprocity and power closure.

    The closure residual ``1 - sum_i |S_i1|^2`` is what the absorber and
    radiation take on a lossless board. It is recorded with its curve.
    """
    col = np.sum(np.abs(S) ** 2, axis=0)            # (n_ports, n_freqs)
    recip = np.abs(S[1, 0, :] - S[0, 1, :])
    smax = float(np.abs(S).max())
    return {
        "column_power": col.astype(float).tolist(),
        "max_column_power": float(col.max()),
        "argmax_column_power": [int(i) for i in np.unravel_index(int(np.argmax(col)), col.shape)],
        "power_closure": (1.0 - col).astype(float).tolist(),
        "reciprocity_abs": recip.astype(float).tolist(),
        "max_abs_s": smax,
        "reciprocity_metric": float(recip.max() / max(smax, 1e-300)),
    }


def referee_context() -> dict:
    """The committed openEMS / Palace / rfx notch records, read as they stand.

    They were measured on a DIFFERENT board — h_sub = 254 um against this
    battery's 300 um, which moves the quarter-wave notch by about 0.6 % on the
    closed form alone — so they are carried as context and compared with
    nothing. The pre-declaration says so; this function does no arithmetic on
    them beyond reading the numbers out of their own files.
    """
    root = REPO / "tests" / "fixtures" / "msl_notch_e4"
    out: dict = {"note": "a different board (h_sub = 254 um); context, not a comparison",
                 "records": {}}
    for tag, name, path in (("openems", "openEMS", "msl_stub_notch_openems_dx50.json"),
                            ("palace", "Palace FEM", "msl_stub_notch_palace_referee.json"),
                            ("rfx_dx50", "rfx", "msl_stub_notch_rfx_dx50.json")):
        p = root / path
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        entry = {"solver": name, "path": f"tests/fixtures/msl_notch_e4/{path}",
                 "meta": d.get("meta")}
        if "notch" in d:
            entry["notch"] = d["notch"]
        if "referee" in d:
            entry["referee"] = d["referee"]
        out["records"][tag] = entry
    return out


def _load_stage(out: Path, name: str) -> dict | None:
    p = out / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def stage_assemble(args, out: Path, fixture_out: Path) -> None:
    fix = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "predeclaration": PREDECLARATION,
        "contract": CONTRACT,
        "driver": DRIVER,
        "artifact": ARTIFACT,
        "bar": BAR,
        "assembled_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "assembler_commit": git_sha(),
        "board": {k: v for k, v in declared(100e-6, "notch").items()
                  if k not in ("dx_m", "dx_um", "substrate_cells_under_strip",
                               "trace_cells_across", "stub_cells_long")},
        "freqs_hz": FREQS.astype(float).tolist(),
        "solves": {},
        "ladder": {},
        "identity": None,
        "adfd": None,
        "plane": None,
        "pilot": None,
        "referee_context": referee_context(),
    }

    pilot = _load_stage(out, "pilot_100um.json")
    if pilot is not None:
        fix["pilot"] = {
            "provenance": pilot["provenance"],
            "cases": [{
                "drive": c["drive"],
                "num_periods": c["num_periods"],
                "settling_db": c["result"]["settling_db"],
                "reliable_fraction": float(np.mean(np.asarray(c["result"]["reliable"])))
                if c["result"]["reliable"] is not None else None,
                "max_sigma_max_excess": float(np.max(c["result"]["sigma_max_excess"]))
                if c["result"]["sigma_max_excess"] is not None else None,
                "wall_s": c["wall_s"],
            } for c in pilot["cases"]],
        }

    for dut in DUTS:
        for um in RUNGS_UM:
            rec = _load_stage(out, f"solve_{dut}_{um}um.json")
            if rec is None:
                continue
            S = _S(rec["result"])
            freqs = np.asarray(rec["result"]["freqs_hz"], dtype=float)
            s21_db = 20.0 * np.log10(np.maximum(np.abs(S[1, 0, :]), 1e-300))
            s11_db = 20.0 * np.log10(np.maximum(np.abs(S[0, 0, :]), 1e-300))
            entry = {
                "dut": dut,
                "rung_um": um,
                "num_periods": rec["num_periods"],
                "drive": rec["drive"],
                "provenance": rec["provenance"],
                "realized": rec["realized"],
                "declared": rec["declared"],
                "preflight_text": rec["preflight"]["text"],
                "warnings": rec["warnings"],
                "wall_s": rec["wall_s"],
                "peak_memory": rec["peak_memory"],
                "S": rec["result"]["S"],
                "freqs_hz": rec["result"]["freqs_hz"],
                "Z0": rec["result"]["Z0"],
                "beta": rec["result"]["beta"],
                "settling_db": rec["result"]["settling_db"],
                "reliable": rec["result"]["reliable"],
                "sigma_max_excess": rec["result"]["sigma_max_excess"],
                "beta_railed": rec["result"]["beta_railed"],
                "assembly": rec["result"]["assembly"],
                "reference_impedances": rec["result"]["reference_impedances"],
                "s21_db": s21_db.astype(float).tolist(),
                "s11_db": s11_db.astype(float).tolist(),
                "power": power_metrics(S),
                "z0_real_median_ohm": float(np.median(np.real(
                    np.asarray(rec["result"]["Z0"]["real"], dtype=float)))),
            }
            entry["settled"] = bool(rec["result"]["settling_db"] is not None
                                    and np.all(np.asarray(rec["result"]["settling_db"])
                                               <= BAR["settling_db"]))
            entry["column_power_within_bar"] = bool(
                entry["power"]["max_column_power"] <= BAR["column_power_max"])
            entry["reciprocity_within_bar"] = bool(
                entry["power"]["reciprocity_metric"] <= BAR["reciprocity"])
            if dut == "notch":
                notch = parabolic_min(freqs, np.abs(S[1, 0, :]))
                entry["notch"] = notch
                entry["notch_depth_db"] = float(s21_db[notch["bin_index"]])
                entry["stopband"] = crossing_width(freqs, s21_db, -10.0,
                                                   notch["bin_index"])
                f_an = rec["declared"]["f_notch_analytic_hz"]
                entry["f_notch_analytic_hz"] = f_an
                entry["notch_vs_analytic_frac"] = {
                    "bin": abs(notch["bin_hz"] - f_an) / f_an,
                    "interp": abs(notch["interp_hz"] - f_an) / f_an,
                }
                # The same closed form on the REALIZED stub and trace width, so
                # a reader can see how much of any offset is the lattice.
                _, eps_eff_real = hammerstad_jensen_z0_eps_eff(
                    rec["realized"]["trace_w_electrical_m"], H_SUB, EPS_R)
                f_an_real = C0 / (4.0 * rec["realized"]["stub_len_centreline_m"]
                                  * math.sqrt(eps_eff_real))
                entry["f_notch_analytic_realized_hz"] = f_an_real
                entry["notch_vs_analytic_realized_frac"] = {
                    "bin": abs(notch["bin_hz"] - f_an_real) / f_an_real,
                    "interp": abs(notch["interp_hz"] - f_an_real) / f_an_real,
                }
            fix["solves"][f"{dut}_{um}um"] = entry

    # --- the dx ladder ----------------------------------------------------
    for dut in DUTS:
        keys = [f"{dut}_{um}um" for um in RUNGS_UM]
        have = [k for k in keys if k in fix["solves"]]
        if len(have) < 2:
            continue
        lad: dict = {"rungs_um": [int(k.split("_")[-1][:-2]) for k in have]}
        if dut == "notch":
            fs = [fix["solves"][k]["notch"]["interp_hz"] for k in have]
            lad["notch_interp_hz"] = fs
            lad["notch_bin_hz"] = [fix["solves"][k]["notch"]["bin_hz"] for k in have]
            diffs = [abs(fs[i + 1] - fs[i]) for i in range(len(fs) - 1)]
            lad["successive_diff_hz"] = diffs
            lad["successive_diff_ratio"] = (
                None if len(diffs) < 2 or diffs[0] == 0.0 else diffs[1] / diffs[0])
            lad["notch_frac_vs_finest"] = [abs(f - fs[-1]) / fs[-1] for f in fs]
        # |S21| and |S11| against the finest rung, off the notch core.
        fine = fix["solves"][have[-1]]
        for key, name in (("s21_db", "s21"), ("s11_db", "s11")):
            fine_db = np.asarray(fine[key], dtype=float)
            rows = []
            for k in have:
                cur = np.asarray(fix["solves"][k][key], dtype=float)
                d = np.abs(cur - fine_db)
                core = np.zeros(len(d), dtype=bool)
                if dut == "notch":
                    core = fine_db <= -20.0        # the notch's -20 dB core
                outside = ~core
                rows.append({
                    "rung": k,
                    "max_db_diff_vs_finest": float(d.max()),
                    "max_db_diff_vs_finest_outside_notch_core": float(d[outside].max())
                    if outside.any() else None,
                    "n_bins_in_core": int(core.sum()),
                })
            lad[f"{name}_vs_finest"] = rows
        lad["max_column_power"] = [fix["solves"][k]["power"]["max_column_power"]
                                   for k in have]
        lad["z0_real_median_ohm"] = [fix["solves"][k]["z0_real_median_ohm"] for k in have]
        lad["settled"] = [fix["solves"][k]["settled"] for k in have]
        fix["ladder"][dut] = lad

    # --- forward identity --------------------------------------------------
    ident = _load_stage(out, "identity_100um.json")
    if ident is not None:
        a = _S(ident["plain"])
        b = _S(ident["override"])
        fix["identity"] = {
            "provenance": ident["provenance"],
            "num_periods": ident["num_periods"],
            "rung_um": ident["rung_um"],
            "plain_S": ident["plain"]["S"],
            "override_S": ident["override"]["S"],
            "freqs_hz": ident["plain"]["freqs_hz"],
            "max_abs_diff": float(np.abs(a - b).max()),
            "max_abs_diff_per_entry": ident["difference"]["max_abs_per_entry"],
            "rtol": BAR["identity_rtol"], "atol": BAR["identity_atol"],
            "allclose_at_bar": ident["difference"]["allclose_at_bar"],
            "plain_settling_db": ident["plain"]["settling_db"],
            "override_settling_db": ident["override"]["settling_db"],
            "warnings": {"plain": ident["plain_warnings"],
                         "override": ident["override_warnings"]},
        }

    # --- AD vs FD ----------------------------------------------------------
    ad = _load_stage(out, "adfd_100um.json")
    if ad is not None:
        fix["adfd"] = {
            "provenance": ad["provenance"],
            "num_periods": ad["num_periods"],
            "rung_um": ad["rung_um"],
            "theta0": ad["theta0"], "fd_h": ad["fd_h"],
            "checkpoint_segments": ad["checkpoint_segments"],
            "n_steps": ad["n_steps"],
            "min_fd_ulp_span": ad["min_fd_ulp_span"],
            "objectives": ad["objectives"],
            "design_variable": ad.get("design_variable"),
            "cases": ad["cases"],
            "bar": BAR["ad_fd_rel"],
        }

    # --- reference-plane invariance ---------------------------------------
    pl = _load_stage(out, "plane_50um.json")
    if pl is not None and len(pl["arms"]) == 2:
        a_rec, b_rec = pl["arms"][0], pl["arms"][1]
        Sa, Sb = _S(a_rec["result"]), _S(b_rec["result"])
        beta_a = _beta(a_rec["result"])
        beta_b = _beta(b_rec["result"])
        delta = float(pl["realized_plane_displacement_m"][0])
        mag_a = 20.0 * np.log10(np.maximum(np.abs(Sa), 1e-300))
        mag_b = 20.0 * np.log10(np.maximum(np.abs(Sb), 1e-300))
        dmag = np.abs(mag_a - mag_b)
        core = np.abs(Sa[1, 0, :]) <= 10 ** (-20.0 / 20.0)
        outside = ~core
        # Moving both planes by the same Delta rotates S11 by 2*beta*Delta and
        # S21 by beta*(Delta_1 + Delta_2) = 2*beta*Delta here.
        rot11 = np.angle(Sb[0, 0, :] * np.conj(Sa[0, 0, :]))
        rot21 = np.angle(Sb[1, 0, :] * np.conj(Sa[1, 0, :]))
        # Each arm's OWN fitted beta gives its own prediction. They are stored
        # separately: the two arms fit beta from different probe ladders, so a
        # gap between the two predictions is itself a reading of the fit.
        pred = 2.0 * np.real(beta_a) * delta
        pred_shifted = 2.0 * np.real(beta_b) * delta
        fix["plane"] = {
            "provenance": pl["provenance"],
            "num_periods": pl["num_periods"], "rung_um": pl["rung_um"],
            "shift_cells": pl["shift_cells"],
            "displacement_m": pl["realized_plane_displacement_m"],
            "n_probe_offsets": pl["n_probe_offsets"],
            "n_probe_spacing": pl["n_probe_spacing"],
            "freqs_hz": a_rec["result"]["freqs_hz"],
            "base_S": a_rec["result"]["S"], "shifted_S": b_rec["result"]["S"],
            "base_beta": a_rec["result"]["beta"], "shifted_beta": b_rec["result"]["beta"],
            "base_settling_db": a_rec["result"]["settling_db"],
            "shifted_settling_db": b_rec["result"]["settling_db"],
            "mag_diff_db": dmag.astype(float).tolist(),
            "max_mag_diff_db": float(dmag.max()),
            "max_mag_diff_db_outside_notch_core": float(dmag[:, :, outside].max())
            if outside.any() else None,
            "n_bins_in_core": int(core.sum()),
            "rotation_s11_rad": rot11.astype(float).tolist(),
            "rotation_s21_rad": rot21.astype(float).tolist(),
            "predicted_rotation_rad": pred.astype(float).tolist(),
            "predicted_rotation_shifted_beta_rad": pred_shifted.astype(float).tolist(),
            "rotation_s11_residual_rad": np.abs(
                np.angle(np.exp(1j * (rot11 - pred)))).astype(float).tolist(),
            "rotation_s21_residual_rad": np.abs(
                np.angle(np.exp(1j * (rot21 - pred)))).astype(float).tolist(),
            "bar_magnitude_db": BAR["magnitude_db"],
        }

    fixture_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = fixture_out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(fix, indent=1))
    os.replace(tmp, fixture_out)
    _log(f"wrote {fixture_out}")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("solve", "pilot", "identity", "adfd", "plane"))
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--dut", choices=DUTS, default="notch")
    ap.add_argument("--rung", type=int, choices=RUNGS_UM, default=100)
    ap.add_argument("--drive", choices=tuple(DRIVES), default=DEFAULT_DRIVE)
    ap.add_argument("--num-periods", type=float, default=DEFAULT_NUM_PERIODS)
    ap.add_argument("--out", required=True, help="directory for the stage JSONs")
    ap.add_argument("--fixture-out", default=str(REPO / ARTIFACT))
    ap.add_argument("--run-id", default=None, help="the compute run id, recorded as-is")
    ap.add_argument("--estimate-only", action="store_true",
                    help="print the cell/step/byte estimate and exit")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.estimate_only:
        for um in RUNGS_UM:
            for dut in DUTS:
                cost_estimate(um * 1e-6, dut, args.num_periods)
        return 0

    if args.assemble:
        stage_assemble(args, out, Path(args.fixture_out))
        return 0

    if args.stage is None:
        ap.error("one of --stage or --assemble is required")

    um = args.rung
    if args.stage == "solve":
        stage_solve(args, out / f"solve_{args.dut}_{um}um.json")
    elif args.stage == "pilot":
        stage_pilot(args, out / f"pilot_{um}um.json")
    elif args.stage == "identity":
        stage_identity(args, out / f"identity_{um}um.json")
    elif args.stage == "adfd":
        stage_adfd(args, out / f"adfd_{um}um.json")
    elif args.stage == "plane":
        stage_plane(args, out / f"plane_{um}um.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
