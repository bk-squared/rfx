#!/usr/bin/env python3
"""Coaxial chain battery — the measurement driver (reduced v2.0 form).

Runs the pre-declaration in ``docs/design_notes/coax_chain_battery_predeclaration.md``
against ``Simulation.compute_coaxial_two_port`` (dielectric bead, thru) and
``Simulation.compute_coaxial_line_reflection`` (short, open, 25 and 100 ohm
loads), both through the ``eps_scale`` design channel.

Every stage writes ONE JSON into ``--out``, persisted before anything optional
runs, and carries its own provenance: the commit (no fallback — the driver
refuses to write a record it cannot stamp), the ``rfx`` package path it
imported, library versions, the device, the realized geometry it asserted
BEFORE solving, the verbatim preflight text, every warning, wall time and peak
memory.

Stages::

    --stage pilot         --rung 4                  record length + drive check
    --stage solve  --dut {bead,thru,short,open,r25,r100} --rung {4,6,9}
    --stage identity      --rung 4                  forward identity, criterion 1(2)
    --stage adfd-twoport  --rung 4                  AD against a float64-loss FD
    --stage adfd-oneport  --rung 4                  the same on the one-port lane
    --stage plane         --rung 6                  reference-plane invariance
    --assemble                                      arithmetic only, no FDTD

The rung is named by the annulus cell count it realizes: the cell size is
``(outer_radius - pin_radius) / rung``, so ``--rung 9`` is the finest.

The assembler joins the stage JSONs into
``tests/fixtures/coax_chain_battery/fixture.json``; the replay test
``tests/oracle/test_coax_chain_battery.py`` re-derives every assembled number
from the stored S and compares it against the contract's bar.

This driver computes numbers. It writes no verdict sentence: where a reading of
the numbers belongs, the fixture carries the measurement and the pre-declared
threshold side by side and nothing else.

Usage (from a clean checkout; the ``rfx`` import must resolve to this tree)::

    PYTHONPATH=. python scripts/diagnostics/coax_chain_battery_measure.py \
        --stage solve --dut bead --rung 9 --out <run-dir> --run-id <id>
    PYTHONPATH=. python scripts/diagnostics/coax_chain_battery_measure.py \
        --assemble --out <run-dir> \
        --fixture-out tests/fixtures/coax_chain_battery/fixture.json
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
from rfx.api import Simulation  # noqa: E402
from rfx.sources.sources import GaussianPulse  # noqa: E402
from rfx.sources.coaxial_port import (  # noqa: E402
    PTFE_EPS_R,
    coaxial_tem_characteristic_impedance,
    stamp_coaxial_annular_resistor,
    stamp_coaxial_line,
    stamp_coaxial_short_plane,
)

SCHEMA = "rfx.coax_chain_battery"
SCHEMA_VERSION = 1
PREDECLARATION = "docs/design_notes/coax_chain_battery_predeclaration.md"
CONTRACT = "docs/design_notes/chain_closure_contract.md"
DRIVER = "scripts/diagnostics/coax_chain_battery_measure.py"
ARTIFACT = "tests/fixtures/coax_chain_battery/fixture.json"

C0 = 299792458.0

# ---------------------------------------------------------------------------
# The line — the repository's existing coax fixtures, unchanged.
# Two-port: tests/unit/sparams/test_coax_two_port_smatrix.py::_sim.
# One-port: tests/unit/sparams/test_coaxial_line_reflection.py::_run.
# The radii, the fill and the analytic Z_TEM are read from the port the API
# builds and from rfx.sources.coaxial_port; nothing here re-declares them.
# ---------------------------------------------------------------------------

DOMAIN_TWOPORT = (0.008, 0.008, 0.060)
DOMAIN_ONEPORT = (0.008, 0.008, 0.040)
FREQ_MAX = 40.0e9
CPML_LAYERS = 16                 # the Simulation default both fixtures use

F_LO, F_HI, N_FREQS = 4.0e9, 12.0e9, 81
FREQS = np.linspace(F_LO, F_HI, N_FREQS)

TWO_PORT_DUTS = ("bead", "thru")
ONE_PORT_DUTS = ("short", "open", "r25", "r100")
DUTS = TWO_PORT_DUTS + ONE_PORT_DUTS
ONE_PORT_LOAD_OHM = {"r25": 25.0, "r100": 100.0}

# Rungs, named by the annulus cell count each realizes. The annulus (outer
# minus inner radius) is what the mesh has to resolve, so the ladder is set on
# it rather than on a cell size: dx = (b - a) / rung, a ratio-1.5 ladder around
# the committed fixture's own 3.79 cells. The finest is the claims rung.
RUNGS = (4, 6, 9)
CLAIMS_RUNG = 9

# Record length. Fixed in units of a one-way traversal of the z domain at the
# fill's phase velocity, so the same PHYSICAL record is used at every rung
# (n_steps then scales with 1/dx by itself). The pilot measures the settling
# witness against this ladder; the chosen value is written into every record,
# and the assembler reads it back from the records rather than from a default.
RECORD_UNIT_LADDER = (4.0, 8.0, 12.0, 16.0)
DEFAULT_RECORD_UNITS = 12.0

# The drive. The committed two-port fixture's own pulse. A differentiated
# Gaussian's spectrum is |S(f)| ~ f exp(-(f/(f0 bw))^2); at this setting the
# 4-12 GHz band sits within 4.3 dB of the pulse's own peak at both edges
# (closed form below, measured per bin by the pilot). The wider alternative is
# carried so the pilot's band-edge check has something to compare against.
DRIVES = {
    "fixture": dict(f0=8.0e9, bandwidth=1.2),
    "wide": dict(f0=8.0e9, bandwidth=1.6),
}
DEFAULT_DRIVE = "fixture"

PROBE_COUNT = 12
PROBE_START_CELLS = 8
PROBE_SPACING_CELLS = 4

# The bead: eps_r multiplied by 4 over an axial length of 6 mm centred between
# the two probe arrays. sqrt(4) = 2, so the section's impedance is Z_TEM / 2
# and its phase constant 2 beta_line, whatever the fill is.
BEAD_EPS_SCALE = 4.0
BEAD_LENGTH_M = 6.0e-3
BEAD_MASK_VARIANTS = ("full_cross_section", "annulus_only")
BEAD_MASK = "full_cross_section"          # the pre-declaration's wording

# Stage 3(b). The pre-declaration asks for two runs differing only in
# ``reference_plane_axial_index_offset``. That keyword belongs to the
# DEPRECATED ``compute_coaxial_s_matrix`` and is not a parameter of
# ``compute_coaxial_two_port``, whose reference planes are its own feed planes
# (``rfx/sparams/coax.py``: ref_top_m / ref_bot_m from z_feed_top / z_feed_bot)
# and are not settable. The plane change this lane CAN express without touching
# the grid is to translate the DUT along the line: the feeds do not move, so
# port 1's electrical distance to the bead falls by Delta and port 2's rises by
# Delta. Predicted: |S| invariant, angle(S11) rotates by +2 beta Delta,
# angle(S22) by -2 beta Delta, angle(S21) unchanged (d1 + d2 is conserved).
# Everything else — grid, cell count, step count, probes, drive — is identical
# between the two arms.
PLANE_SHIFT_CELLS = 4

# --- the AD stages ---------------------------------------------------------
# ``compute_coaxial_two_port`` and ``compute_coaxial_line_reflection`` expose no
# ``checkpoint_segments``, so ``rfx.simulation.run``'s legacy scan keeps every
# step's carry and the reverse-mode tape costs O(n_steps * |carry|). The carry
# holds the DFT plane accumulators as well as the fields and the CPML psi
# slabs, which is why both committed AD gates run at ONE frequency
# (tests/unit/autodiff/test_coax_two_port_ad.py, test_coax_end_to_end_ad.py).
# The battery's AD stages therefore run on a SHORTER z domain and a reduced
# frequency set; the cell size, the cross-section, the fill and the design
# channel are the battery's own. Every reduced value is recorded in the stage
# JSON beside the estimate that forced it.
AD_RUNG = 4
AD_DOMAIN_TWOPORT = (0.008, 0.008, 0.026)
AD_DOMAIN_ONEPORT = (0.008, 0.008, 0.020)   # the committed one-port AD fixture
AD_FREQS = np.array([5.0e9, 8.0e9, 11.0e9])
AD_PROBE_COUNT = 6
AD_PROBE_START_CELLS = 6
AD_PROBE_SPACING_CELLS = 3
AD_ONE_PORT_PROBE_COUNT = 9                 # the committed one-port AD fixture
AD_RECORD_UNITS = 9.0              # the longest record whose tape fits the budget below
AD_TAPE_BUDGET_BYTES = 30 * 2 ** 30         # the a6000-1 preset's 48 GB, with room

# theta channels. Two-port: theta multiplies eps_r inside the bead, so
# theta0 = 4.0 is the battery's own bead. One-port: theta is ADDED to eps_r in
# a mid-line slab, the committed one-port AD gate's own channel, theta0 = 0.
AD_THETA0_TWOPORT = BEAD_EPS_SCALE
AD_FD_H_TWOPORT = 8.0e-3                    # 2e-3 of theta0, the committed gate's ratio
AD_THETA0_ONEPORT = 0.0
AD_FD_H_ONEPORT = 2.0e-2                    # the committed one-port gate's own h
AD_ONE_PORT_SLAB_CELLS = 6
MIN_FD_ULP_SPAN = 1.0e4                     # below this the FD reference resolves nothing

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
    # PI ruling 2026-09-21: a quantity that is near zero by construction is not
    # compared in dB from rung to rung. The thru's |S11| is held to this upper
    # bound at every bin; the bead's |S11| is compared within the magnitude bar
    # only outside the cores of its reflection zeros, a core being a bin where
    # the ANALYTIC |S11| is below this level.
    "deep_null_db": -20.0,
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
    print(f"[coax-battery {stamp}] {msg}", flush=True)


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


def settling_witness(res) -> dict:
    """The ring-down witness, and whether this call produced one at all.

    ``compute_coaxial_two_port`` computes ``settling_db`` from a concrete time
    series and SKIPS it whenever ``eps_scale`` is given (it cannot read a traced
    array back), so every bead record leaves it NaN; the one-port result carries
    no ``settling_db`` field at all. Where the energy witness is absent the
    contract admits one substitute — record-length invariance, measured by the
    doubled-record arms this driver submits beside the claims rung — so the
    absence is recorded here rather than papered over with the thru's value.
    """
    raw = getattr(res, "settling_db", None)
    if raw is None:
        return {"settling_db": None, "has_energy_witness": False,
                "why": "this lane's result carries no settling_db field"}
    arr = np.asarray(raw, dtype=float)
    finite = bool(np.all(np.isfinite(arr)))
    return {
        "settling_db": [None if not np.isfinite(v) else float(v) for v in np.atleast_1d(arr)],
        "has_energy_witness": finite,
        "why": None if finite else (
            "settling_db is NaN: compute_coaxial_two_port skips the ring-down "
            "witness on the eps_scale path, which every bead record takes"),
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


def preflight_record(sim) -> dict:
    """The preflight report, verbatim text and structured findings."""
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


# ---------------------------------------------------------------------------
# the line
# ---------------------------------------------------------------------------

def dx_of(rung: int) -> float:
    """The cell size a rung names: ``(outer - pin) / rung``, read from the
    port's own radii rather than from a number written here."""
    a, b = port_radii()
    return (b - a) / float(rung)


def port_radii() -> tuple[float, float]:
    """The registered port's radii, read from the API's own defaults."""
    sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN_TWOPORT, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3)
    p = sim._coaxial_ports[0]
    return float(p.pin_radius), float(p.outer_radius)


def build_sim(rung: int, dut: str, *, drive: str = DEFAULT_DRIVE,
              domain: tuple[float, float, float] | None = None) -> Simulation:
    """The committed coax fixture at one cell size.

    The two lanes use their own fixture's z extent; ``domain`` overrides it for
    the AD stages, whose reduced board is recorded with the measurement.
    """
    if dut not in DUTS:
        raise ValueError(f"unknown dut {dut!r}; expected one of {DUTS}")
    if domain is None:
        domain = DOMAIN_TWOPORT if dut in TWO_PORT_DUTS else DOMAIN_ONEPORT
    dx = dx_of(rung)
    sim = Simulation(freq_max=FREQ_MAX, domain=domain, boundary="cpml",
                     cpml_layers=CPML_LAYERS, dx=dx)
    sim.add_coaxial_port((domain[0] / 2.0, domain[1] / 2.0, domain[2] / 2.0),
                         face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(**DRIVES[drive]))
    return sim


def declared(rung: int, dut: str,
             domain: tuple[float, float, float] | None = None) -> dict:
    a, b = port_radii()
    dx = dx_of(rung)
    if domain is None:
        domain = DOMAIN_TWOPORT if dut in TWO_PORT_DUTS else DOMAIN_ONEPORT
    z_tem = coaxial_tem_characteristic_impedance(a, b)
    rec = {
        "dut": dut,
        "lane": "two_port" if dut in TWO_PORT_DUTS else "one_port",
        "rung_annulus_cells": rung,
        "dx_m": dx,
        "dx_um": dx * 1e6,
        "domain_m": list(domain),
        "freq_max_hz": FREQ_MAX,
        "cpml_layers": CPML_LAYERS,
        "pin_radius_m": a,
        "outer_radius_m": b,
        "annulus_m": b - a,
        "annulus_cells": (b - a) / dx,
        "fill_eps_r": float(PTFE_EPS_R),
        "z_tem_ohm": z_tem,
        "probe_count": PROBE_COUNT,
        "probe_start_cells": PROBE_START_CELLS,
        "probe_spacing_cells": PROBE_SPACING_CELLS,
        "freqs_hz": [F_LO, F_HI, N_FREQS],
    }
    if dut == "bead":
        rec.update({
            "bead_eps_scale": BEAD_EPS_SCALE,
            "bead_length_m": BEAD_LENGTH_M,
            "bead_cells": BEAD_LENGTH_M / dx,
            "bead_mask": BEAD_MASK,
            "bead_fill_eps_r": float(PTFE_EPS_R) * BEAD_EPS_SCALE,
            "bead_z_tem_ohm": coaxial_tem_characteristic_impedance(
                a, b, float(PTFE_EPS_R) * BEAD_EPS_SCALE),
        })
    if dut in ONE_PORT_LOAD_OHM:
        rec["dut_impedance_ohm"] = ONE_PORT_LOAD_OHM[dut]
    return rec


# ---------------------------------------------------------------------------
# realized geometry — measured before any FDTD step, from the same stamping
# helpers the extractor calls. The axial layout is REPLICATED here from
# rfx/sparams/coax.py; every replicated plane is checked against the value the
# result itself reports (``reference_planes``, ``annulus_cells``) after the
# solve, so a drift between the two shows up as a failure rather than as a
# fixture that quietly describes a different line.
# ---------------------------------------------------------------------------

def axial_layout(grid, lane: str, probes: tuple[int, int, int] | None = None) -> dict:
    """The feed / source / probe indices the extractor derives internally.

    ``probes`` is ``(count, start_cells, spacing_cells)``; the AD stages pass
    their own reduced ladder, everything else takes the lane's defaults.
    """
    count, start, spacing = probes or (PROBE_COUNT, PROBE_START_CELLS,
                                       PROBE_SPACING_CELLS)
    nz = int(grid.shape[2])
    pad_lo, pad_hi = int(grid.pad_z_lo), int(grid.pad_z_hi)
    if lane == "two_port":
        z_hi_coax_top = nz - pad_hi - 2
        z_feed_top = z_hi_coax_top - 1
        z_src_top = z_hi_coax_top - 3
        z_lo_coax_bot = pad_lo + 2
        z_feed_bot = z_lo_coax_bot + 1
        z_src_bot = z_lo_coax_bot + 3
        probes_top = sorted(z_src_top - start - spacing * k for k in range(count))
        probes_bot = sorted(z_src_bot + start + spacing * k for k in range(count))
        return {
            "lane": lane, "nz": nz, "pad_z_lo": pad_lo, "pad_z_hi": pad_hi,
            "z_lo_coax": z_lo_coax_bot, "z_hi_coax": z_hi_coax_top,
            "z_feed_bot": z_feed_bot, "z_feed_top": z_feed_top,
            "z_src_bot": z_src_bot, "z_src_top": z_src_top,
            "probes_bot": probes_bot, "probes_top": probes_top,
            "probe_gap_cells": probes_top[0] - probes_bot[-1],
        }
    z_dut = pad_lo + 4                      # dut_offset_cells default
    z_hi_coax = nz - pad_hi - 2
    z_feed = z_hi_coax - 1
    z_src = z_hi_coax - 3
    probe_z = [z_dut + start + spacing * k for k in range(count)]
    return {
        "lane": lane, "nz": nz, "pad_z_lo": pad_lo, "pad_z_hi": pad_hi,
        "z_dut": z_dut, "z_hi_coax": z_hi_coax, "z_feed": z_feed, "z_src": z_src,
        "probes": probe_z,
    }


def bead_indices(layout: dict, dx: float, *, shift_cells: int = 0) -> tuple[int, int]:
    """The bead's [start, stop) axial cell indices: ``round(6 mm / dx)`` cells
    centred between the two probe arrays, translated by ``shift_cells``."""
    n_bead = int(round(BEAD_LENGTH_M / dx))
    if n_bead < 1:
        raise RuntimeError(f"the bead rounds to {n_bead} cells at dx = {dx:.6g} m")
    mid = 0.5 * (layout["probes_bot"][-1] + layout["probes_top"][0])
    z0 = int(round(mid - 0.5 * n_bead)) + int(shift_cells)
    return z0, z0 + n_bead


def cross_section_masks(grid, center_xy, a: float, b: float) -> dict:
    """Radial masks in the x-y plane, on the same node convention the stamps
    use (``x = (i - pad_x_lo) * dx``)."""
    dx = float(grid.dx)
    i = np.arange(int(grid.shape[0]))
    j = np.arange(int(grid.shape[1]))
    x = (i - int(grid.pad_x_lo)) * dx - float(center_xy[0])
    y = (j - int(grid.pad_y_lo)) * dx - float(center_xy[1])
    r = np.hypot(x[:, None], y[None, :])
    shell_thickness = min(dx, 0.5 * (b - a))
    shell_inner = b - shell_thickness
    return {
        "r": r,
        "pin": r <= a,
        "fill": (r <= shell_inner) & (r > a),
        "shell": (r <= b) & (r > shell_inner),
        "shell_inner_radius": shell_inner,
    }


def bead_eps_scale(grid, center_xy, a: float, b: float, layout: dict, dx: float, *,
                   variant: str = BEAD_MASK, shift_cells: int = 0) -> np.ndarray:
    """``eps_scale`` for the bead: 4 inside it, 1 everywhere else."""
    if variant not in BEAD_MASK_VARIANTS:
        raise ValueError(f"unknown bead mask {variant!r}; expected {BEAD_MASK_VARIANTS}")
    z0, z1 = bead_indices(layout, dx, shift_cells=shift_cells)
    scale = np.ones(tuple(int(s) for s in grid.shape), dtype=np.float32)
    if variant == "full_cross_section":
        scale[:, :, z0:z1] = BEAD_EPS_SCALE
    else:
        # The dielectric annulus alone: the pin and the shell are PEC (their
        # eps_r is irrelevant there) and outside the shell the line is screened.
        fill = cross_section_masks(grid, center_xy, a, b)["fill"]
        scale[:, :, z0:z1] = np.where(fill[:, :, None], BEAD_EPS_SCALE, 1.0)
    return scale


def realized_geometry(sim: Simulation, rung: int, dut: str, *,
                      shift_cells: int = 0,
                      bead_mask: str = BEAD_MASK,
                      probes: tuple[int, int, int] | None = None) -> dict:
    """What the grid and the stamps actually build, measured not assumed."""
    grid = sim._build_grid()
    lane = "two_port" if dut in TWO_PORT_DUTS else "one_port"
    layout = axial_layout(grid, lane, probes)
    port = sim._coaxial_ports[0]
    a, b = float(port.pin_radius), float(port.outer_radius)
    center_xy = (float(port.position[0]), float(port.position[1]))
    dz = float(grid.dx)

    materials, _, _ = sim._build_materials(grid)
    if lane == "two_port":
        materials, shell_inner = stamp_coaxial_line(
            grid, materials, center_xy=center_xy,
            z_lo_index=layout["z_lo_coax"], z_hi_index=layout["z_hi_coax"],
            pin_radius=a, outer_radius=b)
        for z in (layout["z_feed_top"], layout["z_feed_bot"]):
            materials = stamp_coaxial_annular_resistor(
                grid, materials, center_xy=center_xy, z_index=z, pin_radius=a,
                outer_radius=b, target_impedance=coaxial_tem_characteristic_impedance(a, b),
                shell_inner_radius=shell_inner)
        ref_planes_m = [(layout["z_feed_top"] - grid.pad_z_lo) * dz,
                        (layout["z_feed_bot"] - grid.pad_z_lo) * dz]
        probe_planes_m = {
            "bot": [(z - grid.pad_z_lo) * dz for z in layout["probes_bot"]],
            "top": [(z - grid.pad_z_lo) * dz for z in layout["probes_top"]],
        }
    else:
        materials, shell_inner = stamp_coaxial_line(
            grid, materials, center_xy=center_xy,
            z_lo_index=layout["z_dut"], z_hi_index=layout["z_hi_coax"],
            pin_radius=a, outer_radius=b)
        materials = stamp_coaxial_annular_resistor(
            grid, materials, center_xy=center_xy, z_index=layout["z_feed"],
            pin_radius=a, outer_radius=b,
            target_impedance=coaxial_tem_characteristic_impedance(a, b),
            shell_inner_radius=shell_inner)
        if dut == "short":
            materials = stamp_coaxial_short_plane(
                grid, materials, center_xy=center_xy, z_index=layout["z_dut"],
                outer_radius=b)
        elif dut in ONE_PORT_LOAD_OHM:
            materials = stamp_coaxial_annular_resistor(
                grid, materials, center_xy=center_xy, z_index=layout["z_dut"],
                pin_radius=a, outer_radius=b,
                target_impedance=ONE_PORT_LOAD_OHM[dut],
                shell_inner_radius=shell_inner)
        ref_planes_m = [(layout["z_dut"] - grid.pad_z_lo) * dz]
        probe_planes_m = {"line": [(z - grid.pad_z_lo) * dz for z in layout["probes"]]}

    eps = np.asarray(materials.eps_r)
    sig = np.asarray(materials.sigma)
    masks = cross_section_masks(grid, center_xy, a, b)
    # A z index strictly inside the line and away from every stamped plane.
    z_probe = int(layout["probes_bot"][-1] if lane == "two_port" else layout["probes"][-1])
    r = masks["r"]
    pin_cells = int(masks["pin"].sum())
    shell_cells = int(masks["shell"].sum())
    fill_cells = int(masks["fill"].sum())
    pec_slice = sig[:, :, z_probe] > 1.0
    fill_slice = masks["fill"]
    rec = {
        "grid_shape": [int(s) for s in grid.shape],
        "n_cells": int(np.prod(grid.shape)),
        "dx_m": dz,
        "dt_s": float(grid.dt),
        "pads": [int(grid.pad_x_lo), int(grid.pad_y_lo), int(grid.pad_z_lo),
                 int(grid.pad_z_hi)],
        "layout": {k: v for k, v in layout.items()},
        "annulus_cells": (b - a) / dz,
        "pin_radius_m": a,
        "outer_radius_m": b,
        "shell_inner_radius_m": float(shell_inner),
        # Rasterized counts in one cross-section, and the radii the raster
        # actually reaches — the declared radii are continuous, these are not.
        "pin_cells_cross_section": pin_cells,
        "shell_cells_cross_section": shell_cells,
        "fill_cells_cross_section": fill_cells,
        "realized_pin_radius_max_m": float(r[masks["pin"]].max()) if pin_cells else None,
        "realized_fill_radius_min_m": float(r[fill_slice].min()) if fill_cells else None,
        "realized_fill_radius_max_m": float(r[fill_slice].max()) if fill_cells else None,
        "realized_annulus_cells_radial": float(
            (r[fill_slice].max() - r[masks["pin"]].max()) / dz) if fill_cells and pin_cells else None,
        "fill_eps_r_realized": float(np.median(eps[:, :, z_probe][fill_slice]))
        if fill_cells else None,
        "n_pec_cells_at_probe_plane": int(pec_slice.sum()),
        "reference_planes_m": ref_planes_m,
        "probe_planes_m": probe_planes_m,
        "z_probe_index_used_for_cross_section": z_probe,
        "waveform": {"f0": float(port.excitation.f0),
                     "bandwidth": float(port.excitation.bandwidth),
                     "amplitude": float(port.excitation.amplitude),
                     "cutoff": float(port.excitation.cutoff)},
    }
    if lane == "two_port":
        line_len = (layout["z_hi_coax"] - layout["z_lo_coax"]) * dz
        rec["line_length_m"] = line_len
        rec["feed_to_feed_m"] = (layout["z_feed_top"] - layout["z_feed_bot"]) * dz
        if dut == "bead":
            z0, z1 = bead_indices(layout, dz, shift_cells=shift_cells)
            rec.update({
                "bead_mask": bead_mask,
                "bead_shift_cells": int(shift_cells),
                "bead_z_cells": [z0, z1],
                "bead_n_cells": z1 - z0,
                "bead_z_m": [(z0 - grid.pad_z_lo) * dz, (z1 - grid.pad_z_lo) * dz],
                "bead_length_realized_m": (z1 - z0) * dz,
                # What the analytic referee needs: the line length between each
                # feed plane and the bead's near face.
                "d_port1_to_bead_m": (layout["z_feed_top"] - z1) * dz,
                "d_port2_to_bead_m": (z0 - layout["z_feed_bot"]) * dz,
                "bead_inside_probe_gap": bool(
                    z0 > layout["probes_bot"][-1] and z1 < layout["probes_top"][0]),
            })
    else:
        rec["line_length_m"] = (layout["z_hi_coax"] - layout["z_dut"]) * dz
    return rec


def assert_realized(sim: Simulation, rung: int, dut: str, **kw) -> dict:  # noqa: D401
    """Refuse to solve unless the realized line IS the declared line."""
    m = realized_geometry(sim, rung, dut, **kw)
    d = declared(rung, dut, domain=tuple(float(v) for v in sim._domain))
    problems = []
    if abs(m["annulus_cells"] - rung) > 1e-9:
        problems.append(f"annulus realizes {m['annulus_cells']:.6f} cells, rung is {rung}")
    if m["fill_eps_r_realized"] is None or abs(
            m["fill_eps_r_realized"] - float(PTFE_EPS_R)) > 1e-6:
        problems.append(f"the fill realizes eps_r = {m['fill_eps_r_realized']}, the "
                        f"stamp declares {float(PTFE_EPS_R)}")
    if m["pin_cells_cross_section"] < 1 or m["shell_cells_cross_section"] < 1:
        problems.append(f"the cross-section rasterizes {m['pin_cells_cross_section']} pin "
                        f"and {m['shell_cells_cross_section']} shell cells")
    if m["n_pec_cells_at_probe_plane"] < 1:
        problems.append("no PEC cell at the probe plane — the line is not conducting there")
    lay = m["layout"]
    if m["layout"]["lane"] == "two_port":
        if lay["probe_gap_cells"] <= 0:
            problems.append(f"the two probe arrays overlap: gap {lay['probe_gap_cells']} cells")
        if dut == "bead":
            if not m["bead_inside_probe_gap"]:
                problems.append(
                    f"the bead at cells {m['bead_z_cells']} is not strictly between the "
                    f"probe arrays (bottom ends {lay['probes_bot'][-1]}, top starts "
                    f"{lay['probes_top'][0]})")
            if abs(m["bead_length_realized_m"] - BEAD_LENGTH_M) > m["dx_m"]:
                problems.append(
                    f"the bead realizes {m['bead_length_realized_m']*1e3:.4f} mm, more "
                    f"than one cell from the declared {BEAD_LENGTH_M*1e3:.4f} mm")
    if problems:
        raise RuntimeError(
            "assert_realized: the realized line is not the declared line — refusing to "
            "solve. " + "; ".join(problems) + f" [declared: {d}] [measured: {m}]")
    return m


def cross_check_result_against_layout(rec_realized: dict, res, lane: str) -> dict:
    """The planes THIS driver replicated against the planes the RESULT reports.

    The two arithmetics are independent — one is in ``rfx/sparams/coax.py``,
    the other in ``axial_layout`` above — so agreement is a check and not a
    tautology. A disagreement means the driver is describing a different line
    from the one that was solved.
    """
    out = {"annulus_cells_reported": float(res.annulus_cells),
           "annulus_cells_replicated": rec_realized["annulus_cells"]}
    out["annulus_cells_agree"] = bool(
        abs(out["annulus_cells_reported"] - out["annulus_cells_replicated"]) < 1e-9)
    if lane == "two_port":
        rep = np.asarray(res.reference_planes, dtype=float)
        mine = np.asarray(rec_realized["reference_planes_m"], dtype=float)
        out["reference_planes_reported_m"] = rep.tolist()
        out["reference_planes_replicated_m"] = mine.tolist()
        out["reference_planes_max_abs_diff_m"] = float(np.max(np.abs(rep - mine)))
        out["reference_planes_agree"] = bool(out["reference_planes_max_abs_diff_m"] < 1e-12)
    if not out["annulus_cells_agree"] or not out.get("reference_planes_agree", True):
        raise RuntimeError(
            "cross_check_result_against_layout: the layout this driver replicated is not "
            f"the layout the extractor used — {out}")
    return out


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

def record_steps(grid, units: float) -> int:
    """``units`` one-way traversals of the z domain at the fill's phase
    velocity, in whole timesteps rounded up to a multiple of 100."""
    nz_phys = int(grid.shape[2]) - int(grid.pad_z_lo) - int(grid.pad_z_hi)
    lz = nz_phys * float(grid.dx)
    v = C0 / math.sqrt(float(PTFE_EPS_R))
    n = units * lz / v / float(grid.dt)
    return int(math.ceil(n / 100.0) * 100)


def cost_estimate(grid, n_steps: int, n_freqs: int, n_planes: int,
                  n_drives: int) -> dict:
    """Cells, steps and the bytes the forward carry and the reverse tape cost.

    The carry is the six field arrays, the CPML psi slabs (``n_layers`` deep on
    each absorbing face) and the DFT plane accumulators. The reverse-mode tape
    keeps one carry per step, because neither coax lane exposes
    ``checkpoint_segments``.
    """
    nx, ny, nz = (int(s) for s in grid.shape)
    n = nx * ny * nz
    fields = 6 * n * 4
    psi = 4 * CPML_LAYERS * 8 * (nx * nz + nx * ny + ny * nz)
    dft = n_planes * nx * ny * n_freqs * 8
    carry = fields + psi + dft
    est = {
        "grid_shape": [nx, ny, nz], "n_cells": n, "dt_s": float(grid.dt),
        "n_steps": int(n_steps), "n_drives": int(n_drives),
        "n_freqs": int(n_freqs), "n_dft_planes": int(n_planes),
        "cell_steps": n * int(n_steps) * int(n_drives),
        "field_bytes_f32": fields, "cpml_psi_bytes_f32": psi,
        "dft_accumulator_bytes": dft, "carry_bytes": carry,
        "reverse_tape_estimate_bytes": carry * int(n_steps),
    }
    print(f"[cost] {nx}x{ny}x{nz} = {n:,} cells, {n_steps:,} steps x {n_drives} "
          f"drive(s) = {est['cell_steps']:.3e} cell-steps; carry "
          f"{carry / 2 ** 30:.4f} GiB; reverse tape estimate "
          f"{est['reverse_tape_estimate_bytes'] / 2 ** 30:.2f} GiB", flush=True)
    return est


def drive_spectrum_rel_db(f_hz: float, f0: float, bandwidth: float) -> float:
    """A differentiated Gaussian's amplitude at ``f_hz``, in dB below its own
    peak. ``s(t) = -2 (t-t0)/tau exp(-((t-t0)/tau)^2)`` with
    ``tau = 1/(f0 bw pi)`` transforms to ``|S(f)| ~ f exp(-(f/(f0 bw))^2)``,
    whose peak sits at ``f0 bw / sqrt(2)``. Closed form, no measurement."""
    w = f0 * bandwidth
    peak_f = w / math.sqrt(2.0)
    amp = f_hz * math.exp(-(f_hz / w) ** 2)
    peak = peak_f * math.exp(-0.5)
    return 20.0 * math.log10(max(amp, 1e-300) / peak)


# ---------------------------------------------------------------------------
# solving
# ---------------------------------------------------------------------------

def _two_port_record(res) -> dict:
    return {
        "lane": "two_port",
        "freqs_hz": _f(res.freqs),
        "S": _c(res.s_params),
        "port_names": list(res.port_names),
        "reference_planes_m": _f(res.reference_planes),
        "cond_a": _f(res.cond_a),
        "recurrence_residual": _f(res.recurrence_residual),
        "fit_residual": _f(res.fit_residual),
        "gamma": _c(res.gamma),
        "annulus_cells": float(res.annulus_cells),
        "settling": settling_witness(res),
        "status": str(res.status),
    }


def _one_port_record(res) -> dict:
    return {
        "lane": "one_port",
        "freqs_hz": _f(res.freqs),
        "S11": _c(res.s11),
        "gamma": _c(res.gamma),
        "recurrence_residual": _f(res.recurrence_residual),
        "fit_residual": _f(res.fit_residual),
        "annulus_cells": float(res.annulus_cells),
        "z0_numerical_ohm": _c(res.z0_numerical_ohm),
        "termination": str(res.termination),
        "settling": settling_witness(res),
        "status": str(res.status),
    }


def solve_two_port(sim, *, n_steps: int, freqs=None, eps_scale=None,
                   probe_count: int = PROBE_COUNT,
                   probe_start_cells: int = PROBE_START_CELLS,
                   probe_spacing_cells: int = PROBE_SPACING_CELLS):
    return sim.compute_coaxial_two_port(
        n_steps=int(n_steps),
        freqs=jnp.asarray(FREQS if freqs is None else freqs),
        probe_count=probe_count, probe_start_cells=probe_start_cells,
        probe_spacing_cells=probe_spacing_cells, eps_scale=eps_scale)


def solve_one_port(sim, dut: str, *, n_steps: int, freqs=None, eps_scale=None,
                   probe_count: int = PROBE_COUNT):
    kw = {}
    if dut == "short":
        kw["termination"] = "short"
    elif dut == "open":
        kw["termination"] = "open"
    else:
        kw["termination"] = "matched"
        kw["dut_impedance"] = ONE_PORT_LOAD_OHM[dut]
    return sim.compute_coaxial_line_reflection(
        n_steps=int(n_steps), freqs=jnp.asarray(FREQS if freqs is None else freqs),
        probe_count=probe_count, eps_scale=eps_scale, **kw)


def _log_two_port(tag: str, res) -> None:
    S = np.asarray(res.s_params)
    col = float(np.max(np.sum(np.abs(S) ** 2, axis=0)))
    k = int(np.argmin(np.abs(S[0, 0, :])))
    _log(f"{tag}: status {res.status} | settling "
         f"{np.array2string(np.asarray(res.settling_db, dtype=float), precision=2)} dB "
         f"| max cond(A) {float(np.max(res.cond_a)):.4g} | max rec.resid "
         f"{float(np.max(res.recurrence_residual)):.4g} | max column power {col:.5f} "
         f"| min |S11| {abs(S[0, 0, k]):.5g} at {np.asarray(res.freqs)[k]/1e9:.4f} GHz")


def _log_one_port(tag: str, res) -> None:
    g = np.asarray(res.s11)
    _log(f"{tag}: status {res.status} | max rec.resid "
         f"{float(np.max(res.recurrence_residual)):.4g} | max fit.resid "
         f"{float(np.max(res.fit_residual)):.4g} | |S11| "
         f"[{float(np.abs(g).min()):.4f}, {float(np.abs(g).max()):.4f}]")


def _base(args, stage: str, dut: str | None, rung: int | None) -> dict:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "dut": dut,
        "rung_annulus_cells": rung,
        "predeclaration": PREDECLARATION,
        "contract": CONTRACT,
        "driver": DRIVER,
        "bar": BAR,
        "provenance": provenance(args),
    }


def _dut_eps_scale(sim, rung: int, dut: str, *, shift_cells: int = 0,
                   variant: str = BEAD_MASK, as_jnp: bool = True,
                   probes: tuple[int, int, int] | None = None):
    """The design-channel array a DUT needs, or None where it needs none."""
    if dut != "bead":
        return None
    grid = sim._build_grid()
    port = sim._coaxial_ports[0]
    a, b = float(port.pin_radius), float(port.outer_radius)
    layout = axial_layout(grid, "two_port", probes)
    arr = bead_eps_scale(grid, (float(port.position[0]), float(port.position[1])),
                         a, b, layout, float(grid.dx), variant=variant,
                         shift_cells=shift_cells)
    return jnp.asarray(arr) if as_jnp else arr


# ---------------------------------------------------------------------------
# stage: pilot — record length, drive, and the bead-mask control
# ---------------------------------------------------------------------------

def stage_pilot(args, out: Path) -> None:
    rung = args.rung
    rec = _base(args, "pilot", "bead", rung)
    rec["declared"] = declared(rung, "bead")
    rec["record_unit_ladder"] = list(RECORD_UNIT_LADDER)
    rec["record_unit_definition"] = (
        "one unit = one traversal of the physical z extent at c / sqrt(fill eps_r); "
        "n_steps is that time in whole steps, rounded up to a multiple of 100")
    rec["drives"] = {k: dict(v) for k, v in DRIVES.items()}
    rec["drive_spectrum_rel_db"] = {
        k: {"band_lo": drive_spectrum_rel_db(F_LO, v["f0"], v["bandwidth"]),
            "band_hi": drive_spectrum_rel_db(F_HI, v["f0"], v["bandwidth"])}
        for k, v in DRIVES.items()}
    rec["cases"] = []
    _write(out, rec)

    # 1. the record-length ladder, on the battery's own drive
    for units in RECORD_UNIT_LADDER:
        sim = build_sim(rung, "bead", drive=DEFAULT_DRIVE)
        geo = assert_realized(sim, rung, "bead")
        pf = preflight_record(sim)
        grid = sim._build_grid()
        n_steps = record_steps(grid, units)
        est = cost_estimate(grid, n_steps, N_FREQS, 2 * 2 * PROBE_COUNT, 2)
        eps = _dut_eps_scale(sim, rung, "bead")
        _log(f"pilot record_units={units} n_steps={n_steps}")
        with _Captured() as cap:
            res = solve_two_port(sim, n_steps=n_steps, eps_scale=eps)
        _log_two_port(f"pilot units={units}", res)
        rec["cases"].append({
            "kind": "record_length", "drive": DEFAULT_DRIVE,
            "bead_mask": BEAD_MASK, "record_units": float(units), "n_steps": n_steps,
            "realized": geo, "preflight": pf, "cost": est,
            "cross_check": cross_check_result_against_layout(geo, res, "two_port"),
            "warnings": cap.warnings, "wall_s": cap.wall,
            "peak_memory": peak_memory(), "result": _two_port_record(res),
        })
        _write(out, rec)

    # 2. the wider drive at the chosen record length — the band-edge check
    for drive in DRIVES:
        if drive == DEFAULT_DRIVE:
            continue
        sim = build_sim(rung, "bead", drive=drive)
        geo = assert_realized(sim, rung, "bead")
        pf = preflight_record(sim)
        grid = sim._build_grid()
        n_steps = record_steps(grid, args.record_units)
        eps = _dut_eps_scale(sim, rung, "bead")
        _log(f"pilot drive={drive} n_steps={n_steps}")
        with _Captured() as cap:
            res = solve_two_port(sim, n_steps=n_steps, eps_scale=eps)
        _log_two_port(f"pilot drive={drive}", res)
        rec["cases"].append({
            "kind": "drive", "drive": drive, "bead_mask": BEAD_MASK,
            "record_units": float(args.record_units), "n_steps": n_steps,
            "realized": geo, "preflight": pf,
            "cross_check": cross_check_result_against_layout(geo, res, "two_port"),
            "warnings": cap.warnings, "wall_s": cap.wall,
            "peak_memory": peak_memory(), "result": _two_port_record(res),
        })
        _write(out, rec)

    # 3. the bead-mask control. The pre-declaration says "the full
    #    cross-section"; the physical bead is the dielectric annulus alone.
    #    Outside the shell the line is screened, so the two should agree — this
    #    measures whether they do rather than assuming it.
    for variant in BEAD_MASK_VARIANTS:
        if variant == BEAD_MASK:
            continue
        sim = build_sim(rung, "bead", drive=DEFAULT_DRIVE)
        geo = assert_realized(sim, rung, "bead", bead_mask=variant)
        grid = sim._build_grid()
        n_steps = record_steps(grid, args.record_units)
        eps = _dut_eps_scale(sim, rung, "bead", variant=variant)
        _log(f"pilot bead_mask={variant} n_steps={n_steps}")
        with _Captured() as cap:
            res = solve_two_port(sim, n_steps=n_steps, eps_scale=eps)
        _log_two_port(f"pilot mask={variant}", res)
        rec["cases"].append({
            "kind": "bead_mask", "drive": DEFAULT_DRIVE, "bead_mask": variant,
            "record_units": float(args.record_units), "n_steps": n_steps,
            "realized": geo, "preflight": preflight_record(sim),
            "cross_check": cross_check_result_against_layout(geo, res, "two_port"),
            "warnings": cap.warnings, "wall_s": cap.wall,
            "peak_memory": peak_memory(), "result": _two_port_record(res),
        })
        _write(out, rec)

    # 4. the AD boards' own settling, forward-only. The AD stages run on a
    #    shorter z domain; whether THAT record settles is a separate question
    #    from whether the battery's does, and it is cheap to answer here.
    for lane, dut, dom, pc, ps, pp in (
        ("two_port", "bead", AD_DOMAIN_TWOPORT, AD_PROBE_COUNT,
         AD_PROBE_START_CELLS, AD_PROBE_SPACING_CELLS),
        ("one_port", "short", AD_DOMAIN_ONEPORT, AD_ONE_PORT_PROBE_COUNT, None, None),
    ):
        sim = build_sim(AD_RUNG, dut, drive=DEFAULT_DRIVE, domain=dom)
        grid = sim._build_grid()
        n_steps = record_steps(grid, AD_RECORD_UNITS)
        _log(f"pilot ad-board {lane} n_steps={n_steps}")
        with _Captured() as cap:
            if lane == "two_port":
                probes = (pc, ps, pp)
                geo = assert_realized(sim, AD_RUNG, dut, probes=probes)
                eps = _dut_eps_scale(sim, AD_RUNG, dut, probes=probes)
                res = solve_two_port(sim, n_steps=n_steps, freqs=AD_FREQS,
                                     eps_scale=eps, probe_count=pc,
                                     probe_start_cells=ps, probe_spacing_cells=pp)
                result = _two_port_record(res)
                _log_two_port("pilot ad-board two_port", res)
                est = cost_estimate(grid, n_steps, len(AD_FREQS), 2 * 2 * pc, 2)
            else:
                geo = assert_realized(sim, AD_RUNG, dut,
                                      probes=(pc, PROBE_START_CELLS, PROBE_SPACING_CELLS))
                res = solve_one_port(sim, dut, n_steps=n_steps, freqs=AD_FREQS,
                                     probe_count=pc)
                result = _one_port_record(res)
                _log_one_port("pilot ad-board one_port", res)
                est = cost_estimate(grid, n_steps, len(AD_FREQS), 2 * pc, 1)
        rec["cases"].append({
            "kind": "ad_board", "lane": lane, "dut": dut, "domain_m": list(dom),
            "record_units": AD_RECORD_UNITS, "n_steps": n_steps,
            "realized": geo, "cost": est, "warnings": cap.warnings,
            "wall_s": cap.wall, "peak_memory": peak_memory(), "result": result,
        })
        _write(out, rec)


# ---------------------------------------------------------------------------
# stage: solve
# ---------------------------------------------------------------------------

def stage_solve(args, out: Path) -> None:
    rung, dut = args.rung, args.dut
    lane = "two_port" if dut in TWO_PORT_DUTS else "one_port"
    sim = build_sim(rung, dut, drive=args.drive)
    geo = assert_realized(sim, rung, dut)
    pf = preflight_record(sim)
    grid = sim._build_grid()
    n_steps = record_steps(grid, args.record_units)
    n_planes = (2 * 2 * PROBE_COUNT) if lane == "two_port" else (2 * PROBE_COUNT)
    est = cost_estimate(grid, n_steps, N_FREQS, n_planes, 2 if lane == "two_port" else 1)

    rec = _base(args, "solve", dut, rung)
    rec.update({
        "lane": lane, "drive": args.drive,
        "record_units": float(args.record_units), "n_steps": n_steps,
        "declared": declared(rung, dut), "realized": geo, "preflight": pf, "cost": est,
    })
    _log(f"solve dut={dut} rung={rung} n_steps={n_steps}")
    with _Captured() as cap:
        if lane == "two_port":
            eps = _dut_eps_scale(sim, rung, dut)
            res = solve_two_port(sim, n_steps=n_steps, eps_scale=eps)
            _log_two_port(f"solve {dut} rung {rung}", res)
            rec["result"] = _two_port_record(res)
        else:
            res = solve_one_port(sim, dut, n_steps=n_steps)
            _log_one_port(f"solve {dut} rung {rung}", res)
            rec["result"] = _one_port_record(res)
    rec["cross_check"] = cross_check_result_against_layout(geo, res, lane)
    rec["warnings"] = cap.warnings
    rec["wall_s"] = cap.wall
    rec["peak_memory"] = peak_memory()
    _write(out, rec)


# ---------------------------------------------------------------------------
# stage: forward identity (criterion 1(2))
# ---------------------------------------------------------------------------

def stage_identity(args, out: Path) -> None:
    """Two pairs, because this lane's ``eps_scale`` channel makes them different
    questions.

    ARM A (thru): ``eps_scale = None`` against a no-op ``eps_scale`` of ones.
    That is the contract's own sentence — "S under a no-op traced override
    equals the untraced call" — and it is the only pair on this lane where the
    two calls run DIFFERENT code: ``compute_coaxial_two_port`` dispatches on
    ``eps_scale is not None``, so the ones array routes the voltage extraction,
    the assembly and the two-drive solve through their jnp cores while ``None``
    keeps the validated numpy path.

    ARM B (bead): the same bead handed over as a numpy array and as a jnp
    array. That is what the pre-declaration asks for literally. Both arms take
    the jnp path (the dispatch keys off ``is not None``, not off the container),
    so this pair measures the container, not the code path.
    """
    rung = args.rung
    rec = _base(args, "identity", None, rung)
    rec.update({"drive": args.drive, "record_units": float(args.record_units),
                "arms": []})
    _write(out, rec)

    for tag, dut, left, right, what in (
        ("thru_none_vs_ones", "thru", "eps_scale=None", "eps_scale=ones",
         "the untraced numpy path against the no-op jnp path"),
        ("bead_numpy_vs_jnp", "bead", "eps_scale=numpy bead", "eps_scale=jnp bead",
         "the same bead in two containers; both take the jnp path"),
    ):
        sim = build_sim(rung, dut, drive=args.drive)
        geo = assert_realized(sim, rung, dut)
        pf = preflight_record(sim)
        grid = sim._build_grid()
        n_steps = record_steps(grid, args.record_units)
        if dut == "thru":
            eps_left, eps_right = None, jnp.ones(tuple(int(s) for s in grid.shape),
                                                 dtype=jnp.float32)
        else:
            eps_left = _dut_eps_scale(sim, rung, dut, as_jnp=False)
            eps_right = jnp.asarray(eps_left)

        _log(f"identity {tag}: left ({left})")
        with _Captured() as cap_l:
            res_l = solve_two_port(build_sim(rung, dut, drive=args.drive),
                                   n_steps=n_steps, eps_scale=eps_left)
        _log_two_port(f"identity {tag} left", res_l)
        _log(f"identity {tag}: right ({right})")
        with _Captured() as cap_r:
            res_r = solve_two_port(build_sim(rung, dut, drive=args.drive),
                                   n_steps=n_steps, eps_scale=eps_right)
        _log_two_port(f"identity {tag} right", res_r)

        a = np.asarray(res_l.s_params)
        b = np.asarray(res_r.s_params)
        d = np.abs(a - b)
        rec["arms"].append({
            "tag": tag, "dut": dut, "left": left, "right": right, "what": what,
            "n_steps": n_steps, "declared": declared(rung, dut), "realized": geo,
            "preflight": pf,
            "cross_check": cross_check_result_against_layout(geo, res_l, "two_port"),
            "left_result": _two_port_record(res_l),
            "right_result": _two_port_record(res_r),
            "left_warnings": cap_l.warnings, "right_warnings": cap_r.warnings,
            "left_wall_s": cap_l.wall, "right_wall_s": cap_r.wall,
            "difference": {
                "max_abs": float(d.max()),
                "max_abs_per_entry": [[float(d[i, j].max()) for j in range(d.shape[1])]
                                      for i in range(d.shape[0])],
                "max_rel": float((d / np.maximum(np.abs(a), 1e-300)).max()),
                "max_abs_magnitude_diff": float(np.abs(np.abs(a) - np.abs(b)).max()),
                "allclose_at_bar": bool(np.allclose(a, b, rtol=BAR["identity_rtol"],
                                                    atol=BAR["identity_atol"])),
            },
        })
        _write(out, rec)

    rec["peak_memory"] = peak_memory()
    _write(out, rec)


# ---------------------------------------------------------------------------
# stage: AD against a float64-loss FD
# ---------------------------------------------------------------------------

def _fd_ulp_span(f_plus: float, f_minus: float, dtype) -> float:
    """Resolving power of a central difference, in ULPs of ``dtype``.

    ``dtype`` is the dtype the LOSS was computed in, not the container the
    values arrived in: ``float(jnp_scalar)`` is always a Python float, so keying
    off the value alone measures float64 even for a float32 loss. Same
    expression as the committed gates' own helper.
    """
    ulp = float(np.spacing(np.asarray(abs(0.5 * (f_plus + f_minus)), dtype=dtype)))
    return abs(f_plus - f_minus) / ulp


def _objectives_two_port(S, k_anchor: int) -> dict:
    """The three scalars stage 3(a) differentiates, as jnp expressions."""
    return {
        "band_mean_s21_sq": jnp.mean(jnp.abs(S[1, 0, :]) ** 2),
        "band_mean_s11_sq": jnp.mean(jnp.abs(S[0, 0, :]) ** 2),
        "re_s21_s11_conj_at_max_s11": jnp.real(S[1, 0, k_anchor]
                                               * jnp.conj(S[0, 0, k_anchor])),
    }


def _objectives_one_port(g, k_anchor: int) -> dict:
    return {
        "band_mean_s11_sq": jnp.mean(jnp.abs(g) ** 2),
        "re_s11_at_max_s11": jnp.real(g[k_anchor]),
    }


def stage_adfd(args, out: Path, lane: str) -> None:
    rung = AD_RUNG
    dut = "bead" if lane == "two_port" else "short"
    domain = AD_DOMAIN_TWOPORT if lane == "two_port" else AD_DOMAIN_ONEPORT
    probe_count = AD_PROBE_COUNT if lane == "two_port" else AD_ONE_PORT_PROBE_COUNT
    theta0 = AD_THETA0_TWOPORT if lane == "two_port" else AD_THETA0_ONEPORT
    fd_h = AD_FD_H_TWOPORT if lane == "two_port" else AD_FD_H_ONEPORT

    probes = ((probe_count, AD_PROBE_START_CELLS, AD_PROBE_SPACING_CELLS)
              if lane == "two_port"
              else (probe_count, PROBE_START_CELLS, PROBE_SPACING_CELLS))
    sim = build_sim(rung, dut, drive=args.drive, domain=domain)
    geo = assert_realized(sim, rung, dut, probes=probes)
    pf = preflight_record(sim)
    grid = sim._build_grid()
    shape = tuple(int(s) for s in grid.shape)
    n_steps = record_steps(grid, AD_RECORD_UNITS)
    n_planes = (2 * 2 * probe_count) if lane == "two_port" else (2 * probe_count)
    est = cost_estimate(grid, n_steps, len(AD_FREQS), n_planes,
                        2 if lane == "two_port" else 1)

    rec = _base(args, f"adfd-{lane.replace('_', '')}", dut, rung)
    rec.update({
        "lane": lane, "drive": args.drive, "domain_m": list(domain),
        "record_units": AD_RECORD_UNITS, "n_steps": n_steps,
        "freqs_hz": AD_FREQS.tolist(), "probe_count": probe_count,
        "declared": declared(rung, dut, domain=domain), "realized": geo,
        "preflight": pf, "cost": est,
        "theta0": theta0, "fd_h": fd_h, "min_fd_ulp_span": MIN_FD_ULP_SPAN,
        "device": [str(d) for d in jax.devices()],
        "device_kind": [getattr(d, "device_kind", "?") for d in jax.devices()],
        "reduced_because": (
            "neither coax lane exposes checkpoint_segments, so the reverse-mode "
            "tape keeps one full carry per step; the board's z extent and the "
            "frequency set are reduced to fit it, the cell size and cross-section "
            "are the battery's own. The estimate that forced it is in `cost`."),
        "tape_budget_bytes": AD_TAPE_BUDGET_BYTES,
        "cases": [],
    })

    if lane == "two_port":
        layout = axial_layout(grid, "two_port", probes)
        z0, z1 = bead_indices(layout, float(grid.dx))
        mask = np.zeros(shape, dtype=bool)
        mask[:, :, z0:z1] = True
        rec["design_variable"] = {
            "what": "theta REPLACES the eps_r multiplier inside the bead "
                    "(eps_scale = theta there, 1 elsewhere); theta0 is the "
                    "battery's own bead value",
            "bead_z_cells": [z0, z1], "n_cells_scaled": int(mask.sum()),
            "n_cells_total": int(mask.size), "bead_mask": BEAD_MASK,
        }
    else:
        nz = shape[2]
        lo, hi = nz // 2 - AD_ONE_PORT_SLAB_CELLS // 2, nz // 2 + AD_ONE_PORT_SLAB_CELLS // 2
        mask = np.zeros(shape, dtype=bool)
        mask[:, :, lo:hi] = True
        rec["design_variable"] = {
            "what": "theta is ADDED to the eps_r multiplier in a mid-line slab "
                    "(eps_scale = 1 + theta there, 1 elsewhere) — the committed "
                    "one-port AD gate's own channel",
            "slab_z_cells": [lo, hi], "n_cells_scaled": int(mask.sum()),
            "n_cells_total": int(mask.size),
        }
    _write(out, rec)

    mask_j = jnp.asarray(mask)

    def _eps_scale(theta, dtype):
        base = jnp.ones(shape, dtype=dtype)
        if lane == "two_port":
            return jnp.where(mask_j, jnp.asarray(theta, dtype=dtype), base)
        return base + mask_j * jnp.asarray(theta, dtype=dtype)

    def _forward(theta, dtype=jnp.float32):
        eps = _eps_scale(theta, dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if lane == "two_port":
                r = sim.compute_coaxial_two_port(
                    n_steps=n_steps, freqs=jnp.asarray(AD_FREQS),
                    probe_count=probe_count, probe_start_cells=AD_PROBE_START_CELLS,
                    probe_spacing_cells=AD_PROBE_SPACING_CELLS, eps_scale=eps)
                return r.s_params
            r = sim.compute_coaxial_line_reflection(
                termination="short", n_steps=n_steps, freqs=jnp.asarray(AD_FREQS),
                probe_count=probe_count, eps_scale=eps)
            return r.s11

    # The anchor bin is chosen from a CONCRETE forward at theta0, so it is a
    # static index under tracing and the same bin for AD and FD.
    _log("adfd: concrete forward at theta0 (anchor bin and settling witness)")
    with _Captured() as cap0:
        S0 = np.asarray(_forward(theta0))
    ref = S0[0, 0, :] if lane == "two_port" else S0
    k_anchor = int(np.argmax(np.abs(ref)))
    rec["anchor"] = {"bin_index": k_anchor, "bin_hz": float(AD_FREQS[k_anchor]),
                     "what": "the bin of largest |S11| at theta0",
                     "abs_s11_at_theta0": float(np.abs(ref[k_anchor]))}
    rec["forward_at_theta0"] = (_c(S0) if lane == "two_port" else _c(S0))
    rec["forward_warnings"] = cap0.warnings
    rec["forward_wall_s"] = cap0.wall
    _write(out, rec)

    names = (("band_mean_s21_sq", "band_mean_s11_sq", "re_s21_s11_conj_at_max_s11")
             if lane == "two_port" else ("band_mean_s11_sq", "re_s11_at_max_s11"))

    def _obj(theta, which, dtype=jnp.float32):
        S = _forward(theta, dtype)
        table = (_objectives_two_port(S, k_anchor) if lane == "two_port"
                 else _objectives_one_port(S, k_anchor))
        return table[which]

    ad_results = {}
    for which in names:
        _log(f"adfd {which}: AD (float32)")
        with _Captured() as cap_ad:
            loss, g = jax.value_and_grad(lambda t, w=which: _obj(t, w))(
                jnp.asarray(theta0, dtype=jnp.float32))
        ad_results[which] = {
            "loss": float(loss), "grad": float(g),
            "loss_dtype": str(jnp.asarray(loss).dtype),
            "wall_s": cap_ad.wall, "warnings": cap_ad.warnings,
        }
        rec["cases"].append({"objective": which, "ad": ad_results[which]})
        _write(out, rec)

    # ONE pair of float64 forward solves serves every objective: the finite
    # difference is a property of the run at theta +/- h, not of the reduction
    # taken afterwards.
    try:
        from jax import enable_x64
    except ImportError:                      # older JAX (< ~0.4.31)
        from tests._x64_compat import enable_x64

    _log(f"adfd: FD reference at theta0 +/- {fd_h} (float64 loss, scoped x64)")
    with _Captured() as cap_fd:
        with enable_x64():
            S_plus = _forward(theta0 + fd_h)
            S_minus = _forward(theta0 - fd_h)
            if S_plus.dtype not in (jnp.complex128,):
                raise RuntimeError(
                    f"the FD reference did not run in complex128 (got {S_plus.dtype}): "
                    "JAX truncates a float64 request to float32 when x64 is off, so the "
                    "scoped context failed to engage. A float32 reference resolves far "
                    "too few ULPs to judge a gradient.")
            tbl_p = (_objectives_two_port(S_plus, k_anchor) if lane == "two_port"
                     else _objectives_one_port(S_plus, k_anchor))
            tbl_m = (_objectives_two_port(S_minus, k_anchor) if lane == "two_port"
                     else _objectives_one_port(S_minus, k_anchor))
            fd_values = {w: (float(tbl_p[w]), float(tbl_m[w]), str(tbl_p[w].dtype))
                         for w in names}
            S_plus_rec, S_minus_rec = _c(np.asarray(S_plus)), _c(np.asarray(S_minus))
    rec["fd_forward"] = {"S_plus": S_plus_rec, "S_minus": S_minus_rec,
                         "wall_s": cap_fd.wall, "warnings": cap_fd.warnings}

    for case in rec["cases"]:
        which = case["objective"]
        f_plus, f_minus, dtype_name = fd_values[which]
        dtype = np.float64 if dtype_name == "float64" else np.float32
        g_fd = (f_plus - f_minus) / (2.0 * fd_h)
        span = _fd_ulp_span(f_plus, f_minus, dtype)
        interpretable = bool(span >= MIN_FD_ULP_SPAN)
        case["fd"] = {"f_plus": f_plus, "f_minus": f_minus, "h": fd_h,
                      "grad": g_fd, "loss_dtype": dtype_name, "ulp_span": span}
        # The resolving-power statement comes BEFORE the accuracy number, so a
        # comparator failure is recorded as a comparator failure.
        case["comparator"] = {"ulp_span": span, "floor": MIN_FD_ULP_SPAN,
                              "interpretable": interpretable}
        if interpretable:
            case["rel_err"] = abs(case["ad"]["grad"] - g_fd) / max(abs(g_fd), 1e-300)
            case["same_sign"] = bool(case["ad"]["grad"] * g_fd > 0)
        else:
            case["rel_err"] = None
            case["same_sign"] = None
            case["rel_err_note"] = "not interpretable: FD span below the ULP floor"
    rec["peak_memory"] = peak_memory()
    _write(out, rec)


# ---------------------------------------------------------------------------
# stage: reference-plane invariance
# ---------------------------------------------------------------------------

def stage_plane(args, out: Path) -> None:
    rung = args.rung
    rec = _base(args, "plane", "bead", rung)
    dx = dx_of(rung)
    rec.update({
        "drive": args.drive, "record_units": float(args.record_units),
        "shift_cells": PLANE_SHIFT_CELLS, "shift_m": PLANE_SHIFT_CELLS * dx,
        "what_moves": (
            "the DUT, not the grid. compute_coaxial_two_port has no "
            "reference_plane_axial_index_offset (that keyword belongs to the "
            "deprecated compute_coaxial_s_matrix); its reference planes ARE its feed "
            "planes and are not settable. Translating the bead by Delta leaves the "
            "feeds, the grid, the probes, the drive and the step count identical and "
            "changes port 1's electrical distance to the DUT by -Delta and port 2's by "
            "+Delta."),
        "predicted": (
            "|S| invariant; angle(S11) rotates by +2 beta Delta, angle(S22) by "
            "-2 beta Delta, angle(S21) unchanged"),
        "arms": [],
    })
    _write(out, rec)

    for tag, shift in (("base", 0), ("shifted", PLANE_SHIFT_CELLS)):
        sim = build_sim(rung, "bead", drive=args.drive)
        geo = assert_realized(sim, rung, "bead", shift_cells=shift)
        pf = preflight_record(sim)
        grid = sim._build_grid()
        n_steps = record_steps(grid, args.record_units)
        eps = _dut_eps_scale(sim, rung, "bead", shift_cells=shift)
        _log(f"plane arm={tag} shift={shift} cells n_steps={n_steps}")
        with _Captured() as cap:
            res = solve_two_port(sim, n_steps=n_steps, eps_scale=eps)
        _log_two_port(f"plane arm={tag}", res)
        rec["arms"].append({
            "tag": tag, "shift_cells": shift, "n_steps": n_steps,
            "realized": geo, "preflight": pf,
            "cross_check": cross_check_result_against_layout(geo, res, "two_port"),
            "warnings": cap.warnings, "wall_s": cap.wall,
            "result": _two_port_record(res),
        })
        _write(out, rec)

    a_geo, b_geo = rec["arms"][0]["realized"], rec["arms"][1]["realized"]
    rec["realized_displacement_m"] = {
        "port1": b_geo["d_port1_to_bead_m"] - a_geo["d_port1_to_bead_m"],
        "port2": b_geo["d_port2_to_bead_m"] - a_geo["d_port2_to_bead_m"],
    }
    rec["peak_memory"] = peak_memory()
    _write(out, rec)


# ---------------------------------------------------------------------------
# analytic referees — arithmetic only
# ---------------------------------------------------------------------------

def tem_beta(freqs: np.ndarray, eps_r: float) -> np.ndarray:
    """Lossless TEM phase constant ``omega sqrt(eps_r) / c``."""
    return 2.0 * np.pi * np.asarray(freqs, dtype=float) * math.sqrt(eps_r) / C0


def bead_referee(freqs: np.ndarray, *, a: float, b: float, eps_fill: float,
                 eps_scale: float, length_m: float,
                 d_port1_m: float | None = None,
                 d_port2_m: float | None = None) -> dict:
    """A lossless TEM section of impedance ``Z_TEM / sqrt(eps_scale)`` and
    electrical length ``sqrt(eps_scale) beta0 L`` between two ``Z_TEM`` lines.

    ``S11 = Gamma (1 - e^{-2j theta}) / (1 - Gamma^2 e^{-2j theta})``,
    ``S21 = (1 - Gamma^2) e^{-j theta} / (1 - Gamma^2 e^{-2j theta})``,
    ``Gamma = (Z2 - Z1) / (Z2 + Z1)``. With ``d_port1_m`` / ``d_port2_m`` the
    result is also referred to the feed planes, the extra line being lossless
    ``Z_TEM``: ``S11 e^{-2j beta1 d1}``, ``S21 e^{-j beta1 (d1 + d2)}``.
    """
    freqs = np.asarray(freqs, dtype=float)
    z1 = coaxial_tem_characteristic_impedance(a, b, eps_fill)
    z2 = coaxial_tem_characteristic_impedance(a, b, eps_fill * eps_scale)
    gam = (z2 - z1) / (z2 + z1)
    beta1 = tem_beta(freqs, eps_fill)
    beta2 = tem_beta(freqs, eps_fill * eps_scale)
    theta = beta2 * length_m
    e1 = np.exp(-1j * theta)
    e2 = e1 ** 2
    den = 1.0 - gam ** 2 * e2
    s11 = gam * (1.0 - e2) / den
    s21 = (1.0 - gam ** 2) * e1 / den
    out = {
        "z1_ohm": z1, "z2_ohm": z2, "gamma": gam, "length_m": length_m,
        "eps_fill": eps_fill, "eps_scale": eps_scale,
        "theta_rad": theta.tolist(),
        "beta1_rad_per_m": beta1.tolist(), "beta2_rad_per_m": beta2.tolist(),
        "S11_at_bead": _c(s11), "S21_at_bead": _c(s21),
        "abs_S11": np.abs(s11).tolist(), "abs_S21": np.abs(s21).tolist(),
        # theta = n pi are the reflection zeros: a deep null by construction.
        "reflection_zero_hz": [
            n * C0 / (2.0 * length_m * math.sqrt(eps_fill * eps_scale))
            for n in range(1, 8)
            if float(freqs.min()) <= n * C0 / (2.0 * length_m
                                               * math.sqrt(eps_fill * eps_scale))
            <= float(freqs.max())],
    }
    if d_port1_m is not None and d_port2_m is not None:
        s11_p = s11 * np.exp(-2j * beta1 * d_port1_m)
        s22_p = s11 * np.exp(-2j * beta1 * d_port2_m)
        s21_p = s21 * np.exp(-1j * beta1 * (d_port1_m + d_port2_m))
        out.update({
            "d_port1_m": d_port1_m, "d_port2_m": d_port2_m,
            "S11_at_reference_plane": _c(s11_p),
            "S22_at_reference_plane": _c(s22_p),
            "S21_at_reference_plane": _c(s21_p),
        })
    return out


def one_port_referee(dut: str, *, a: float, b: float, eps_fill: float) -> dict:
    """``Gamma = -1`` (short), ``+1`` (open, idealized: no fringing or
    radiation at the open end) and ``(R - Z0) / (R + Z0)`` for a resistive load,
    with ``Z0`` the line's own analytic ``Z_TEM``."""
    z0 = coaxial_tem_characteristic_impedance(a, b, eps_fill)
    if dut == "short":
        g, what = -1.0, "PEC short plane at the reference plane"
    elif dut == "open":
        g, what = 1.0, ("ideal open; the realized open end fringes and radiates, "
                        "which this closed form does not carry")
    else:
        r = ONE_PORT_LOAD_OHM[dut]
        g, what = (r - z0) / (r + z0), f"resistive load {r} ohm against Z_TEM"
    return {"z0_ohm": z0, "gamma": g, "abs_gamma": abs(g), "what": what}


# ---------------------------------------------------------------------------
# assemble — arithmetic only, no FDTD
# ---------------------------------------------------------------------------

def _S(block) -> np.ndarray:
    return (np.asarray(block["real"], dtype=float)
            + 1j * np.asarray(block["imag"], dtype=float))


def power_metrics(S: np.ndarray) -> dict:
    """Column power, reciprocity and power closure, with their curves."""
    col = np.sum(np.abs(S) ** 2, axis=0)
    recip = np.abs(S[1, 0, :] - S[0, 1, :])
    smax = float(np.abs(S).max())
    return {
        "column_power": col.astype(float).tolist(),
        "max_column_power": float(col.max()),
        "argmax_column_power": [int(i) for i in
                                np.unravel_index(int(np.argmax(col)), col.shape)],
        "power_closure": (1.0 - col).astype(float).tolist(),
        "reciprocity_abs": recip.astype(float).tolist(),
        "max_abs_s": smax,
        "reciprocity_metric": float(recip.max() / max(smax, 1e-300)),
    }


def _vertex_delta(y0: float, y1: float, y2: float) -> float:
    denom = y0 - 2.0 * y1 + y2
    return 0.0 if denom == 0.0 else 0.5 * (y0 - y2) / denom


def parabolic_min(freqs: np.ndarray, y: np.ndarray) -> dict:
    """Locate the minimum of ``|S11|`` from the raw bin and from the vertex of
    the parabola through ``|S11|^2`` at that bin and its neighbours.

    Near a simple reflection zero ``S11 ~ a (f - f0)``, so ``|S11|^2`` IS a
    parabola there and ``|S11|`` is a V — which makes the squared form the
    estimator the physics justifies and the magnitude form a check on it.
    """
    k = int(np.argmin(y))
    step = float(freqs[1] - freqs[0])
    out = {"bin_index": k, "bin_hz": float(freqs[k]), "bin_value": float(y[k]),
           "bin_width_hz": step}
    if 0 < k < len(y) - 1:
        sq = np.asarray(y, dtype=float) ** 2
        d_sq = _vertex_delta(float(sq[k - 1]), float(sq[k]), float(sq[k + 1]))
        d_mag = _vertex_delta(float(y[k - 1]), float(y[k]), float(y[k + 1]))
        out["interp_hz"] = float(freqs[k]) + d_sq * step
        out["interp_delta_bins"] = float(d_sq)
        out["interp_on"] = "|S11|^2"
        out["interp_hz_on_magnitude"] = float(freqs[k]) + d_mag * step
        out["estimator_spread_hz"] = abs(out["interp_hz"] - out["interp_hz_on_magnitude"])
    else:
        out.update({"interp_hz": float(freqs[k]), "interp_delta_bins": 0.0,
                    "interp_on": "|S11|^2", "interp_hz_on_magnitude": float(freqs[k]),
                    "estimator_spread_hz": 0.0, "at_band_edge": True})
    return out


def richardson(f_coarse: float, f_fine: float, order: int, ratio: float) -> float:
    """``f_fine + (f_fine - f_coarse) / (ratio**order - 1)``.

    The limit a sequence converging at ``order`` in the cell size would reach.
    Arithmetic on two numbers; it asserts nothing about the order the sequence
    actually has, which is what the successive-difference ratio is for.
    """
    return f_fine + (f_fine - f_coarse) / (ratio ** order - 1.0)


def fixture_provenance(p: dict, index: dict | None, stage_file: str) -> dict:
    """The provenance a committed artifact may carry.

    The stage JSONs record absolute paths because that is what a person
    debugging a job needs. A fixture is committed to a public repository, so the
    machine paths are reduced to the question they exist to answer — did
    ``import rfx`` resolve to this run's own tree — and the run is named by its
    compute id rather than by a directory on one pod.
    """
    out = {k: p[k] for k in (
        "commit", "jax_version", "numpy_version", "jax_default_backend",
        "jax_devices", "jax_enable_x64", "python", "platform", "utc",
        "rfx_version") if k in p}
    rfx_file, repo = p.get("rfx_file", ""), p.get("repo", "")
    out["rfx_import_tail"] = "/".join(Path(rfx_file).parts[-2:]) if rfx_file else None
    out["rfx_resolved_inside_the_run_tree"] = bool(
        repo and rfx_file and rfx_file.startswith(repo.rstrip("/") + "/"))
    entry = (index or {}).get(stage_file, {})
    out["compute_run_id"] = entry.get("vessl_run_id")
    out["compute_run_dir"] = entry.get("run_dir")
    return out


def _load_stage(out: Path, name: str) -> dict | None:
    p = out / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def stage_assemble(args, out: Path, fixture_out: Path) -> None:
    index = json.loads(Path(args.run_index).read_text()) if args.run_index else None
    a, b = port_radii()
    fix = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "predeclaration": PREDECLARATION, "contract": CONTRACT, "driver": DRIVER,
        "artifact": ARTIFACT, "bar": BAR,
        "assembled_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "assembler_commit": git_sha(),
        "line": {
            "pin_radius_m": a, "outer_radius_m": b, "annulus_m": b - a,
            "fill_eps_r": float(PTFE_EPS_R),
            "z_tem_ohm": coaxial_tem_characteristic_impedance(a, b),
            "domain_two_port_m": list(DOMAIN_TWOPORT),
            "domain_one_port_m": list(DOMAIN_ONEPORT),
            "freq_max_hz": FREQ_MAX, "cpml_layers": CPML_LAYERS,
            "bead_eps_scale": BEAD_EPS_SCALE, "bead_length_m": BEAD_LENGTH_M,
            "rungs_annulus_cells": list(RUNGS), "claims_rung": CLAIMS_RUNG,
        },
        "freqs_hz": FREQS.astype(float).tolist(),
        "solves": {}, "ladder": {}, "identity": None,
        "adfd": {}, "plane": None, "pilot": None,
    }

    # --- pilot -------------------------------------------------------------
    pilot = _load_stage(out, f"pilot_rung{AD_RUNG}.json")
    if pilot is not None:
        fix["pilot"] = {
            "provenance": fixture_provenance(pilot["provenance"], index,
                                             f"pilot_rung{AD_RUNG}.json"),
            "record_unit_definition": pilot["record_unit_definition"],
            "drive_spectrum_rel_db": pilot["drive_spectrum_rel_db"],
            "cases": [{
                "kind": c["kind"], "drive": c.get("drive"),
                "bead_mask": c.get("bead_mask"), "lane": c.get("lane"),
                "record_units": c["record_units"], "n_steps": c["n_steps"],
                "settling": c["result"].get("settling"),
                "status": c["result"]["status"],
                "max_recurrence_residual": float(np.max(c["result"]["recurrence_residual"])),
                "max_fit_residual": float(np.max(c["result"]["fit_residual"])),
                "max_cond_a": (float(np.max(c["result"]["cond_a"]))
                               if c["result"].get("cond_a") is not None else None),
                "wall_s": c["wall_s"],
            } for c in pilot["cases"]],
        }
        # The two bead masks side by side, on the S they each produced.
        masks = {c["bead_mask"]: c for c in pilot["cases"]
                 if c["kind"] in ("record_length", "bead_mask")
                 and c.get("record_units") == args.record_units}
        if len(masks) == 2:
            k0, k1 = BEAD_MASK_VARIANTS
            s0, s1 = _S(masks[k0]["result"]["S"]), _S(masks[k1]["result"]["S"])
            fix["pilot"]["bead_mask_comparison"] = {
                "masks": [k0, k1],
                "max_abs_diff": float(np.abs(s0 - s1).max()),
                "max_db_diff": float(np.max(np.abs(
                    20.0 * np.log10(np.maximum(np.abs(s0), 1e-300))
                    - 20.0 * np.log10(np.maximum(np.abs(s1), 1e-300))))),
                "what": ("the pre-declaration's full-cross-section bead against the "
                         "dielectric annulus alone; outside the shell the line is "
                         "screened, so this measures whether that matters"),
            }

    # --- solves ------------------------------------------------------------
    for dut in DUTS:
        lane = "two_port" if dut in TWO_PORT_DUTS else "one_port"
        for rung in RUNGS:
            rec = _load_stage(out, f"solve_{dut}_rung{rung}.json")
            if rec is None:
                continue
            res = rec["result"]
            freqs = np.asarray(res["freqs_hz"], dtype=float)
            entry = {
                "dut": dut, "lane": lane, "rung_annulus_cells": rung,
                "n_steps": rec["n_steps"], "record_units": rec["record_units"],
                "drive": rec["drive"],
                "provenance": fixture_provenance(rec["provenance"], index,
                                                 f"solve_{dut}_rung{rung}.json"),
                "declared": rec["declared"], "realized": rec["realized"],
                "cross_check": rec["cross_check"],
                "preflight_text": rec["preflight"]["text"],
                "warnings": rec["warnings"], "wall_s": rec["wall_s"],
                "peak_memory": rec["peak_memory"],
                "freqs_hz": res["freqs_hz"], "status": res["status"],
                "annulus_cells": res["annulus_cells"],
                "recurrence_residual": res["recurrence_residual"],
                "fit_residual": res["fit_residual"],
                "gamma": res["gamma"],
                "resolution": {
                    "dx_m": rec["declared"]["dx_m"],
                    "annulus_cells": rec["declared"]["annulus_cells"],
                    "cells_per_guided_wavelength_at_f_max": (
                        C0 / (float(freqs.max()) * math.sqrt(float(PTFE_EPS_R)))
                        / rec["declared"]["dx_m"]),
                    "f_max_hz": float(freqs.max()),
                    "smallest_resolved_dimension_m": rec["declared"]["annulus_m"],
                },
            }
            if lane == "two_port":
                S = _S(res["S"])
                entry.update({
                    "S": res["S"], "Z0_reference": None,
                    "reference_planes_m": res["reference_planes_m"],
                    "settling": res["settling"], "cond_a": res["cond_a"],
                    "s11_db": (20.0 * np.log10(np.maximum(np.abs(S[0, 0, :]), 1e-300))
                               ).astype(float).tolist(),
                    "s21_db": (20.0 * np.log10(np.maximum(np.abs(S[1, 0, :]), 1e-300))
                               ).astype(float).tolist(),
                    "s22_db": (20.0 * np.log10(np.maximum(np.abs(S[1, 1, :]), 1e-300))
                               ).astype(float).tolist(),
                    "power": power_metrics(S),
                    "settled": (
                        bool(np.all(np.asarray(res["settling"]["settling_db"],
                                               dtype=float) <= BAR["settling_db"]))
                        if res["settling"]["has_energy_witness"] else None),
                })
                entry["column_power_within_bar"] = bool(
                    entry["power"]["max_column_power"] <= BAR["column_power_max"])
                entry["reciprocity_within_bar"] = bool(
                    entry["power"]["reciprocity_metric"] <= BAR["reciprocity"])
                if dut == "bead":
                    real = rec["realized"]
                    ref_dec = bead_referee(
                        freqs, a=a, b=b, eps_fill=float(PTFE_EPS_R),
                        eps_scale=BEAD_EPS_SCALE, length_m=BEAD_LENGTH_M,
                        d_port1_m=real["d_port1_to_bead_m"],
                        d_port2_m=real["d_port2_to_bead_m"])
                    ref_real = bead_referee(
                        freqs, a=a, b=b, eps_fill=float(PTFE_EPS_R),
                        eps_scale=BEAD_EPS_SCALE,
                        length_m=real["bead_length_realized_m"],
                        d_port1_m=real["d_port1_to_bead_m"],
                        d_port2_m=real["d_port2_to_bead_m"])
                    entry["referee_declared_length"] = ref_dec
                    entry["referee_realized_length"] = ref_real
                    entry["reflection_zero_measured"] = parabolic_min(
                        freqs, np.abs(S[0, 0, :]))
                    # The deep-null ruling: dB distance to the referee only
                    # where the ANALYTIC |S11| is above the null level.
                    for tag, refr in (("declared", ref_dec), ("realized", ref_real)):
                        an11 = np.asarray(refr["abs_S11"], dtype=float)
                        an21 = np.asarray(refr["abs_S21"], dtype=float)
                        core = 20.0 * np.log10(np.maximum(an11, 1e-300)) <= BAR["deep_null_db"]
                        d11 = np.abs(20.0 * np.log10(np.maximum(np.abs(S[0, 0, :]), 1e-300))
                                     - 20.0 * np.log10(np.maximum(an11, 1e-300)))
                        d21 = np.abs(20.0 * np.log10(np.maximum(np.abs(S[1, 0, :]), 1e-300))
                                     - 20.0 * np.log10(np.maximum(an21, 1e-300)))
                        entry[f"vs_referee_{tag}"] = {
                            "s11_db_diff": d11.astype(float).tolist(),
                            "s21_db_diff": d21.astype(float).tolist(),
                            "null_core": core.astype(bool).tolist(),
                            "n_bins_in_null_core": int(core.sum()),
                            "max_s11_db_diff_outside_core": (
                                float(d11[~core].max()) if (~core).any() else None),
                            "max_s21_db_diff": float(d21.max()),
                            "reflection_zero_analytic_hz": refr["reflection_zero_hz"],
                        }
                        zeros = refr["reflection_zero_hz"]
                        if zeros:
                            f_an = zeros[0]
                            entry[f"vs_referee_{tag}"]["zero_frac"] = {
                                "bin": abs(entry["reflection_zero_measured"]["bin_hz"]
                                           - f_an) / f_an,
                                "interp": abs(entry["reflection_zero_measured"]["interp_hz"]
                                              - f_an) / f_an,
                                "analytic_hz": f_an,
                            }
                else:
                    entry["thru_reflection_floor_db"] = float(np.max(entry["s11_db"]))
                    entry["thru_reflection_within_deep_null_bound"] = bool(
                        entry["thru_reflection_floor_db"] <= BAR["deep_null_db"])
            else:
                g = _S(res["S11"])
                entry.update({
                    "S11": res["S11"],
                    "reference_planes_m": rec["realized"]["reference_planes_m"],
                    "z0_numerical_ohm": res["z0_numerical_ohm"],
                    "termination": res["termination"],
                    "abs_s11": np.abs(g).astype(float).tolist(),
                    "s11_db": (20.0 * np.log10(np.maximum(np.abs(g), 1e-300))
                               ).astype(float).tolist(),
                    "max_abs_s11": float(np.abs(g).max()),
                    "passive": bool(np.all(np.abs(g) <= 1.0 + 0.02)),
                })
                refr = one_port_referee(dut, a=a, b=b, eps_fill=float(PTFE_EPS_R))
                entry["referee"] = refr
                entry["vs_referee"] = {
                    "abs_diff": np.abs(np.abs(g) - refr["abs_gamma"]).astype(float).tolist(),
                    "max_abs_diff": float(np.abs(np.abs(g) - refr["abs_gamma"]).max()),
                    "db_diff": np.abs(
                        20.0 * np.log10(np.maximum(np.abs(g), 1e-300))
                        - 20.0 * np.log10(max(refr["abs_gamma"], 1e-300))
                    ).astype(float).tolist(),
                    "max_db_diff": float(np.max(np.abs(
                        20.0 * np.log10(np.maximum(np.abs(g), 1e-300))
                        - 20.0 * np.log10(max(refr["abs_gamma"], 1e-300))))),
                }
            fix["solves"][f"{dut}_rung{rung}"] = entry

    # --- record-length invariance, the settling substitute -----------------
    # The contract admits ONE substitute where a lane emits no energy witness:
    # double the record window at a fixed absorber, and read the shift in
    # max|S| and the column power. Every bead record and every one-port record
    # is in that position (see `settling_witness`), so the claims rung is run
    # twice and the two are compared here. The thru, which DOES carry the
    # energy witness, is run twice as well: it is the control that says whether
    # the substitute and the witness agree on the same record.
    fix["record_length_invariance"] = {}
    for dut in DUTS:
        base = _load_stage(out, f"solve_{dut}_rung{CLAIMS_RUNG}.json")
        doubled = _load_stage(out, f"solve_{dut}_rung{CLAIMS_RUNG}_double.json")
        if base is None or doubled is None:
            continue
        lane = "two_port" if dut in TWO_PORT_DUTS else "one_port"
        rb, rd = base["result"], doubled["result"]
        if lane == "two_port":
            Sb, Sd = _S(rb["S"]), _S(rd["S"])
            shift = float(np.abs(np.abs(Sb) - np.abs(Sd)).max())
            col_b = float(np.max(np.sum(np.abs(Sb) ** 2, axis=0)))
            col_d = float(np.max(np.sum(np.abs(Sd) ** 2, axis=0)))
        else:
            Sb, Sd = _S(rb["S11"]), _S(rd["S11"])
            shift = float(np.abs(np.abs(Sb) - np.abs(Sd)).max())
            col_b = float(np.max(np.abs(Sb) ** 2))
            col_d = float(np.max(np.abs(Sd) ** 2))
        fix["record_length_invariance"][dut] = {
            "lane": lane, "rung_annulus_cells": CLAIMS_RUNG,
            "provenance": fixture_provenance(
                doubled["provenance"], index,
                f"solve_{dut}_rung{CLAIMS_RUNG}_double.json"),
            "record_units": [base["record_units"], doubled["record_units"]],
            "n_steps": [base["n_steps"], doubled["n_steps"]],
            "record_ratio": doubled["n_steps"] / base["n_steps"],
            "max_abs_shift": shift,
            # The contract's form of this substitute: the shift below a tenth
            # of the magnitude bar, expressed in amplitude at the worst bin.
            "shift_bound": 10 ** (BAR["magnitude_db"] / 20.0) - 1.0,
            "shift_bound_tenth": (10 ** (BAR["magnitude_db"] / 20.0) - 1.0) / 10.0,
            "max_column_power": [col_b, col_d],
            "energy_witness": [rb["settling"], rd["settling"]],
            "S_doubled": (rd["S"] if lane == "two_port" else rd["S11"]),
            "freqs_hz": rd["freqs_hz"],
        }

    # --- the dx ladder ------------------------------------------------------
    for dut in DUTS:
        keys = [f"{dut}_rung{r}" for r in RUNGS]
        have = [k for k in keys if k in fix["solves"]]
        if len(have) < 2:
            continue
        rungs = [fix["solves"][k]["rung_annulus_cells"] for k in have]
        ratio = rungs[-1] / rungs[-2] if len(rungs) >= 2 else None
        lane = fix["solves"][have[0]]["lane"]
        lad: dict = {"rungs_annulus_cells": rungs, "refinement_ratio": ratio,
                     "lane": lane}
        fine = fix["solves"][have[-1]]
        if lane == "two_port":
            entries = {"s11": (0, 0), "s21": (1, 0), "s22": (1, 1)}
            fine_S = _S(fine["S"])
            core = np.zeros(len(fine["freqs_hz"]), dtype=bool)
            if dut == "bead" and "vs_referee_realized" in fine:
                core = np.asarray(fine["vs_referee_realized"]["null_core"], dtype=bool)
            for name, (i, j) in entries.items():
                fine_db = 20.0 * np.log10(np.maximum(np.abs(fine_S[i, j, :]), 1e-300))
                rows = []
                for k in have:
                    cur_S = _S(fix["solves"][k]["S"])
                    cur_db = 20.0 * np.log10(np.maximum(np.abs(cur_S[i, j, :]), 1e-300))
                    d = np.abs(cur_db - fine_db)
                    d_abs = np.abs(np.abs(cur_S[i, j, :]) - np.abs(fine_S[i, j, :]))
                    outside = ~core if name == "s11" else np.ones(len(d), bool)
                    rows.append({
                        "rung": k,
                        "max_db_diff_vs_finest": float(d.max()),
                        "max_db_diff_vs_finest_outside_null_core": (
                            float(d[outside].max()) if outside.any() else None),
                        # The same difference in amplitude. A 2 dB bar on a
                        # quantity sitting 30 dB down is not the test it is on a
                        # quantity near 0 dB, and only the pair shows which case
                        # a row is.
                        "max_abs_diff_vs_finest": float(d_abs.max()),
                        "finest_min_abs": float(np.abs(fine_S[i, j, :]).min()),
                        "finest_max_abs": float(np.abs(fine_S[i, j, :]).max()),
                        "n_bins_in_null_core": int(core.sum()),
                    })
                lad[f"{name}_vs_finest"] = rows
            lad["max_column_power"] = [fix["solves"][k]["power"]["max_column_power"]
                                       for k in have]
            lad["settled"] = [fix["solves"][k]["settled"] for k in have]
            if dut == "bead":
                fs = [fix["solves"][k]["reflection_zero_measured"]["interp_hz"]
                      for k in have]
                lad["reflection_zero_interp_hz"] = fs
                lad["reflection_zero_bin_hz"] = [
                    fix["solves"][k]["reflection_zero_measured"]["bin_hz"] for k in have]
                diffs = [abs(fs[i + 1] - fs[i]) for i in range(len(fs) - 1)]
                lad["successive_diff_hz"] = diffs
                lad["successive_diff_ratio"] = (
                    None if len(diffs) < 2 or diffs[0] == 0.0 else diffs[1] / diffs[0])
                lad["zero_frac_vs_finest"] = [abs(f - fs[-1]) / fs[-1] for f in fs]
                if len(fs) >= 2 and ratio:
                    lad["extrapolation"] = {
                        "kind": "arithmetic only — no claim about the sequence's "
                                "actual order",
                        "from_rungs": [have[-2], have[-1]],
                        "refinement_ratio": ratio,
                        "formula": "f_fine + (f_fine - f_coarse) / (ratio**order - 1)",
                        **{f"order_{p}_hz": richardson(fs[-2], fs[-1], p, ratio)
                           for p in (1, 2)},
                    }
        else:
            fine_abs = np.asarray(fine["abs_s11"], dtype=float)
            rows = []
            for k in have:
                cur = np.asarray(fix["solves"][k]["abs_s11"], dtype=float)
                d_db = np.abs(20.0 * np.log10(np.maximum(cur, 1e-300))
                              - 20.0 * np.log10(np.maximum(fine_abs, 1e-300)))
                rows.append({
                    "rung": k,
                    "max_db_diff_vs_finest": float(d_db.max()),
                    "max_abs_diff_vs_finest": float(np.abs(cur - fine_abs).max()),
                    "finest_min_abs": float(fine_abs.min()),
                    "finest_max_abs": float(fine_abs.max()),
                })
            lad["s11_vs_finest"] = rows
            lad["max_abs_s11"] = [fix["solves"][k]["max_abs_s11"] for k in have]

        # The support matrix asks for one cell size: the coarsest rung whose
        # every compared quantity sits inside the bar against the finest. Pure
        # arithmetic over the thresholds already in `bar` — a boolean per rung,
        # not a recommendation sentence.
        inside = []
        for i, k in enumerate(have):
            row = {"rung": k}
            if lane == "two_port":
                for name in ("s11", "s21", "s22"):
                    worst = lad[f"{name}_vs_finest"][i][
                        "max_db_diff_vs_finest_outside_null_core"]
                    row[f"{name}_within_{BAR['magnitude_db']:g}dB"] = (
                        None if worst is None else bool(worst <= BAR["magnitude_db"]))
                if dut == "bead":
                    row["zero_within_1pct"] = bool(
                        lad["zero_frac_vs_finest"][i] <= BAR["frequency_frac"])
            else:
                row[f"s11_within_{BAR['magnitude_db']:g}dB"] = bool(
                    lad["s11_vs_finest"][i]["max_db_diff_vs_finest"]
                    <= BAR["magnitude_db"])
            row["all_inside_bar"] = all(v for v in row.values() if isinstance(v, bool))
            row["resolution"] = fix["solves"][k]["resolution"]
            inside.append(row)
        lad["rung_within_bar_vs_finest"] = inside
        qualifying = [r for r in inside if r["all_inside_bar"]]
        lad["coarsest_rung_within_bar"] = qualifying[0]["rung"] if qualifying else None
        fix["ladder"][dut] = lad

    # --- forward identity ---------------------------------------------------
    ident = _load_stage(out, f"identity_rung{AD_RUNG}.json")
    if ident is not None:
        fix["identity"] = {
            "provenance": fixture_provenance(ident["provenance"], index,
                                             f"identity_rung{AD_RUNG}.json"),
            "rtol": BAR["identity_rtol"], "atol": BAR["identity_atol"],
            "arms": [{
                "tag": arm["tag"], "dut": arm["dut"], "left": arm["left"],
                "right": arm["right"], "what": arm["what"], "n_steps": arm["n_steps"],
                "rung_annulus_cells": ident["rung_annulus_cells"],
                "freqs_hz": arm["left_result"]["freqs_hz"],
                "left_S": arm["left_result"]["S"], "right_S": arm["right_result"]["S"],
                "left_settling": arm["left_result"]["settling"],
                "right_settling": arm["right_result"]["settling"],
                "left_status": arm["left_result"]["status"],
                "right_status": arm["right_result"]["status"],
                **arm["difference"],
                "warnings": {"left": arm["left_warnings"], "right": arm["right_warnings"]},
            } for arm in ident["arms"]],
        }

    # --- AD vs FD -----------------------------------------------------------
    for lane, name in (("two_port", "adfd-twoport"), ("one_port", "adfd-oneport")):
        ad = _load_stage(out, f"{name}_rung{AD_RUNG}.json")
        if ad is None:
            continue
        fix["adfd"][lane] = {
            "provenance": fixture_provenance(ad["provenance"], index,
                                             f"{name}_rung{AD_RUNG}.json"),
            "lane": lane, "dut": ad["dut"], "domain_m": ad["domain_m"],
            "rung_annulus_cells": ad["rung_annulus_cells"], "n_steps": ad["n_steps"],
            "record_units": ad["record_units"], "freqs_hz": ad["freqs_hz"],
            "probe_count": ad["probe_count"], "theta0": ad["theta0"],
            "fd_h": ad["fd_h"], "min_fd_ulp_span": ad["min_fd_ulp_span"],
            "reduced_because": ad["reduced_because"], "cost": ad["cost"],
            "design_variable": ad["design_variable"], "anchor": ad["anchor"],
            "cases": ad["cases"], "bar": BAR["ad_fd_rel"],
            "peak_memory": ad.get("peak_memory"),
        }

    # --- reference-plane invariance ----------------------------------------
    pl = _load_stage(out, "plane_rung6.json")
    if pl is not None and len(pl["arms"]) == 2:
        a_rec, b_rec = pl["arms"][0], pl["arms"][1]
        Sa, Sb = _S(a_rec["result"]["S"]), _S(b_rec["result"]["S"])
        freqs = np.asarray(a_rec["result"]["freqs_hz"], dtype=float)
        delta = pl["shift_m"]
        beta = tem_beta(freqs, float(PTFE_EPS_R))
        mag_a = 20.0 * np.log10(np.maximum(np.abs(Sa), 1e-300))
        mag_b = 20.0 * np.log10(np.maximum(np.abs(Sb), 1e-300))
        dmag = np.abs(mag_a - mag_b)
        rot11 = np.angle(Sb[0, 0, :] * np.conj(Sa[0, 0, :]))
        rot22 = np.angle(Sb[1, 1, :] * np.conj(Sa[1, 1, :]))
        rot21 = np.angle(Sb[1, 0, :] * np.conj(Sa[1, 0, :]))
        pred = 2.0 * beta * delta
        core = np.zeros(len(freqs), dtype=bool)
        fine_bead = fix["solves"].get(f"bead_rung{pl['rung_annulus_cells']}")
        if fine_bead and "vs_referee_realized" in fine_bead:
            core = np.asarray(fine_bead["vs_referee_realized"]["null_core"], dtype=bool)
        outside = ~core
        fix["plane"] = {
            "provenance": fixture_provenance(pl["provenance"], index, "plane_rung6.json"),
            "rung_annulus_cells": pl["rung_annulus_cells"], "n_steps": a_rec["n_steps"],
            "shift_cells": pl["shift_cells"], "shift_m": delta,
            "what_moves": pl["what_moves"], "predicted": pl["predicted"],
            "realized_displacement_m": pl["realized_displacement_m"],
            "freqs_hz": a_rec["result"]["freqs_hz"],
            "base_S": a_rec["result"]["S"], "shifted_S": b_rec["result"]["S"],
            "base_settling": a_rec["result"]["settling"],
            "shifted_settling": b_rec["result"]["settling"],
            "analytic_beta_rad_per_m": beta.astype(float).tolist(),
            "mag_diff_db": dmag.astype(float).tolist(),
            "max_mag_diff_db": float(dmag.max()),
            "max_mag_diff_db_outside_null_core": (
                float(dmag[:, :, outside].max()) if outside.any() else None),
            "n_bins_in_null_core": int(core.sum()),
            "rotation_s11_rad": rot11.astype(float).tolist(),
            "rotation_s22_rad": rot22.astype(float).tolist(),
            "rotation_s21_rad": rot21.astype(float).tolist(),
            "predicted_rotation_rad": pred.astype(float).tolist(),
            "rotation_s11_residual_rad": np.abs(
                np.angle(np.exp(1j * (rot11 - pred)))).astype(float).tolist(),
            "rotation_s22_residual_rad": np.abs(
                np.angle(np.exp(1j * (rot22 + pred)))).astype(float).tolist(),
            "rotation_s21_residual_rad": np.abs(rot21).astype(float).tolist(),
            # The same residuals against the opposite sign. Which one is small
            # says which phase convention the extractor carries; neither being
            # small is the finding.
            "rotation_s11_residual_opposite_sign_rad": np.abs(
                np.angle(np.exp(1j * (rot11 + pred)))).astype(float).tolist(),
            "rotation_s22_residual_opposite_sign_rad": np.abs(
                np.angle(np.exp(1j * (rot22 - pred)))).astype(float).tolist(),
            "max_rotation_s11_residual_rad": float(np.max(np.abs(
                np.angle(np.exp(1j * (rot11 - pred)))))),
            "max_rotation_s11_residual_opposite_sign_rad": float(np.max(np.abs(
                np.angle(np.exp(1j * (rot11 + pred)))))),
            "bar_magnitude_db": BAR["magnitude_db"],
        }

    fixture_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = fixture_out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(fix, indent=1))
    os.replace(tmp, fixture_out)
    _log(f"wrote {fixture_out}")


# ---------------------------------------------------------------------------

STAGES = ("pilot", "solve", "identity", "adfd-twoport", "adfd-oneport", "plane")


def stage_layout_only() -> int:
    """Print the layout and the cost at every rung, build no fields, solve
    nothing. What this answers: does the bead fit between the probe arrays, and
    what does each stage cost."""
    for rung in RUNGS:
        for dut in ("bead", "short"):
            sim = build_sim(rung, dut)
            grid = sim._build_grid()
            lane = "two_port" if dut in TWO_PORT_DUTS else "one_port"
            geo = assert_realized(sim, rung, dut)
            n_steps = record_steps(grid, DEFAULT_RECORD_UNITS)
            n_planes = (2 * 2 * PROBE_COUNT) if lane == "two_port" else (2 * PROBE_COUNT)
            print(f"--- rung {rung} ({dut}, {lane}): dx = {geo['dx_m']*1e6:.3f} um, "
                  f"annulus {geo['annulus_cells']:.4f} cells, "
                  f"fill eps_r {geo['fill_eps_r_realized']}")
            if dut == "bead":
                print(f"    bead {geo['bead_z_cells']} = "
                      f"{geo['bead_length_realized_m']*1e3:.4f} mm, gap "
                      f"{geo['layout']['probe_gap_cells']} cells, d1 "
                      f"{geo['d_port1_to_bead_m']*1e3:.3f} mm, d2 "
                      f"{geo['d_port2_to_bead_m']*1e3:.3f} mm")
            cost_estimate(grid, n_steps, N_FREQS, n_planes, 2 if lane == "two_port" else 1)
    for lane, dut, dom, pc, ps, pp in (
        ("two_port", "bead", AD_DOMAIN_TWOPORT, AD_PROBE_COUNT,
         AD_PROBE_START_CELLS, AD_PROBE_SPACING_CELLS),
        ("one_port", "short", AD_DOMAIN_ONEPORT, AD_ONE_PORT_PROBE_COUNT,
         PROBE_START_CELLS, PROBE_SPACING_CELLS),
    ):
        sim = build_sim(AD_RUNG, dut, domain=dom)
        grid = sim._build_grid()
        geo = assert_realized(sim, AD_RUNG, dut, probes=(pc, ps, pp))
        n_steps = record_steps(grid, AD_RECORD_UNITS)
        n_planes = (2 * 2 * pc) if lane == "two_port" else (2 * pc)
        print(f"--- AD board {lane} (rung {AD_RUNG}, z = {dom[2]*1e3:.1f} mm)")
        if dut == "bead":
            print(f"    bead {geo['bead_z_cells']}, gap "
                  f"{geo['layout']['probe_gap_cells']} cells, inside gap "
                  f"{geo['bead_inside_probe_gap']}")
        est = cost_estimate(grid, n_steps, len(AD_FREQS), n_planes,
                            2 if lane == "two_port" else 1)
        fits = est["reverse_tape_estimate_bytes"] <= AD_TAPE_BUDGET_BYTES
        print(f"    tape estimate {est['reverse_tape_estimate_bytes']/2**30:.2f} GiB "
              f"vs budget {AD_TAPE_BUDGET_BYTES/2**30:.1f} GiB: "
              f"{'fits' if fits else 'DOES NOT FIT'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--dut", choices=DUTS, default="bead")
    ap.add_argument("--rung", type=int, choices=RUNGS, default=RUNGS[0])
    ap.add_argument("--drive", choices=tuple(DRIVES), default=DEFAULT_DRIVE)
    ap.add_argument("--record-units", type=float, default=DEFAULT_RECORD_UNITS)
    ap.add_argument("--out", help="directory for the stage JSONs")
    ap.add_argument("--fixture-out", default=str(REPO / ARTIFACT))
    ap.add_argument("--tag", default=None,
                    help="suffix for this stage's output file; 'double' marks the "
                         "doubled-record arm of the record-length-invariance check")
    ap.add_argument("--run-id", default=None, help="the compute run id, recorded as-is")
    ap.add_argument("--run-index", default=None,
                    help="assemble only: a JSON mapping each stage file to its "
                         "compute run id and run directory name")
    ap.add_argument("--layout-only", action="store_true",
                    help="print the layout and the cost at every rung and exit")
    args = ap.parse_args()

    if args.layout_only:
        return stage_layout_only()

    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.assemble:
        stage_assemble(args, out, Path(args.fixture_out))
        return 0

    if args.stage is None:
        ap.error("one of --stage, --assemble or --layout-only is required")

    suffix = f"_{args.tag}" if args.tag else ""
    if args.stage == "solve":
        stage_solve(args, out / f"solve_{args.dut}_rung{args.rung}{suffix}.json")
    elif args.stage == "pilot":
        stage_pilot(args, out / f"pilot_rung{args.rung}.json")
    elif args.stage == "identity":
        stage_identity(args, out / f"identity_rung{args.rung}.json")
    elif args.stage == "adfd-twoport":
        stage_adfd(args, out / f"adfd-twoport_rung{AD_RUNG}.json", "two_port")
    elif args.stage == "adfd-oneport":
        stage_adfd(args, out / f"adfd-oneport_rung{AD_RUNG}.json", "one_port")
    elif args.stage == "plane":
        stage_plane(args, out / f"plane_rung{args.rung}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
