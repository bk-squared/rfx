#!/usr/bin/env python3
"""Lumped / wire port chain battery — the measurement driver (reduced v2.0 form).

Runs the pre-declaration in
``docs/design_notes/lumped_wire_chain_battery_predeclaration.md`` against the
differentiable ``Simulation.forward(port_s11_freqs=...)`` S11, for a lumped
port (``add_port(..., extent=None)``) and a wire port (``add_port(...,
extent=h)``). The S matrix of ``run(compute_s_params=True)`` is a numpy
post-process and is not in the v2.0 chain; it is never called here.

The fixture is a parallel-plate channel one cell wide between PEC plates and
magnetic side walls, which carries an exact TEM wave, so the port's input
impedance is the closed form of a terminated line and the referee is arithmetic
rather than another solver.

Every stage writes ONE JSON into ``--out``, persisted before anything optional
runs, and carries its own provenance: the commit (no fallback — the driver
refuses to write a record it cannot stamp), the ``rfx`` package path it
imported, library versions, the device, the realized geometry it asserted, the
verbatim preflight text, every warning, wall time and peak memory.

Stages::

    --stage pilot    --kind wire --rung 1000      record length and drive choice
    --stage solve    --kind {lumped,wire} --dut {short,open,res_half,res_double,matched}
                     --rung {1000,500,250}
    --stage identity --kind {lumped,wire} --rung 1000   criterion 1(2)
    --stage adfd-r   --kind {lumped,wire} --rung 1000   AD vs f64 FD in R
    --stage adfd-eps --kind {lumped,wire} --rung 1000   AD vs f64 FD in eps_r
    --assemble                                          arithmetic only, no FDTD

``--all-duts`` and ``--all-rungs`` loop a solve stage inside one process, which
is how the campaign fits into a handful of jobs: each case persists its own
JSON as it finishes, so a job that dies half way leaves the cases it completed.

The assembler joins the stage JSONs into
``tests/fixtures/lumped_wire_chain_battery/fixture.json``; the replay test
``tests/oracle/test_lumped_wire_chain_battery.py`` re-derives every assembled
number from the stored S11 and compares it against the contract's bar.

This driver computes numbers. It writes no verdict sentence: where a reading of
the numbers belongs, the fixture carries the measurement and the pre-declared
threshold side by side and nothing else.

Usage (from a clean checkout; the ``rfx`` import must resolve to this tree)::

    PYTHONPATH=. python scripts/diagnostics/lumped_wire_chain_battery_measure.py \\
        --stage solve --kind wire --all-duts --all-rungs --out <run-dir> --run-id <id>
    PYTHONPATH=. python scripts/diagnostics/lumped_wire_chain_battery_measure.py \\
        --assemble --out <run-dir> \\
        --fixture-out tests/fixtures/lumped_wire_chain_battery/fixture.json
"""
from __future__ import annotations

import argparse
import cmath
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

from tests._x64_compat import enable_x64  # noqa: E402  SCOPED x64 only

SCHEMA = "rfx.lumped_wire_chain_battery"
SCHEMA_VERSION = 1
PREDECLARATION = "docs/design_notes/lumped_wire_chain_battery_predeclaration.md"
CONTRACT = "docs/design_notes/chain_closure_contract.md"
DRIVER = "scripts/diagnostics/lumped_wire_chain_battery_measure.py"
ARTIFACT = "tests/fixtures/lumped_wire_chain_battery/fixture.json"

C0 = 299792458.0
ETA0 = 376.730313668

# ---------------------------------------------------------------------------
# The channel. Pre-declaration, section "Fixture".
#
# h and w are counted in CELLS, so the line's Zc is the same at every rung and
# only its length in cells changes. The port's own reference impedance is that
# Zc, which collapses the closed form to S11 = Gamma_L * exp(-2 j beta L).
# ---------------------------------------------------------------------------

L_LINE = 30e-3                  # declared port plane to termination plane
N_H = {"lumped": 1, "wire": 4}  # plate gap in cells, per port kind
EPS_R_AIR = 1.0
EPS_R_FILL = 2.2                # the permittivity-gradient leg only

F_LO = 1e9
F_HI = 10e9
N_FREQS = 91
FREQS = np.linspace(F_LO, F_HI, N_FREQS)

RUNGS_UM = (1000, 500, 250)     # dx = 1.0, 0.5, 0.25 mm
KINDS = ("lumped", "wire")
DUTS = ("short", "open", "res_half", "res_double", "matched")
RESISTIVE = {"res_half": 0.5, "res_double": 2.0, "matched": 1.0}
CLAIMS_RUNG_UM = 250
COARSEST_RUNG_UM = 1000

# The port node. The pre-declaration puts the port AT the magnetic wall; rfx
# refuses that arrangement and says why (see DEVIATIONS below), so the port
# sits on node 1 and the wall's zeroed H half-cell is the open behind it.
PORT_NODE = 1

# Record length. The pilot measures the witness against this ladder and the
# chosen value is written into every record.
PILOT_NUM_PERIODS = (10.0, 20.0, 40.0, 80.0)
DEFAULT_NUM_PERIODS = 40.0

# The drive. The shipped default for a port is GaussianPulse(f0=freq_max/2,
# bandwidth=0.8); a differentiated Gaussian's spectrum is
# |S(f)| ~ f*exp(-(f/(f0*bw))^2), which at 10 GHz is ~39 dB below its own peak
# for that setting. The pilot measures both drives over the declared 1-10 GHz
# band; the one used by the battery is named in every record.
DRIVES = {
    "default": dict(f0=F_HI / 2.0, bandwidth=0.8),
    "wide": dict(f0=5e9, bandwidth=1.6),
}
DEFAULT_DRIVE = "wide"

# AD/FD stage.
AD_FD_REL_H = 1e-3              # relative central-difference step
AD_BAND = (4e9, 6e9)            # objective 1 averages |S11|^2 over this band
AD_MID_BIN_HZ = 5e9             # objective 2 reads Re(S11) at this bin
MIN_FD_ULP_SPAN = 1.0e4         # below this the FD reference resolves nothing

# The bar (contract, "The v2.0 battery for lumped/wire, MSL and coax") plus the
# two numbers this family's pre-declaration adds: the passivity bound that
# replaces reciprocity and power closure for a one-port, and the matched
# control's upper bound. Recorded beside each measurement; never applied as a
# verdict here.
BAR = {
    "magnitude_db": 2.0,
    "frequency_frac": 0.01,
    "passivity_max": 1.02,
    "matched_floor_db": -20.0,
    "ad_fd_rel": 0.05,
    "identity_rtol": 1e-5,
    "identity_atol": 1e-7,
    # Criterion 2's record-length substitute: one tenth of the magnitude gate.
    "record_doubling_db": 0.2,
}

# Deviations from the pre-declaration, recorded rather than silently applied.
DEVIATIONS = [
    {
        "what": "the port sits one cell in from the magnetic wall, on node 1",
        "predeclaration_says": ("the port spans the plate gap at one end of the "
                                "line, backed by a magnetic wall"),
        "why": (
            "rfx refuses the port AT the wall and says so in preflight: "
            "'sits on the PMC x_lo plane. The outgoing tangential H is zeroed "
            "every step by apply_pmc_faces, so no wave radiates - the probe "
            "records silent zero field. Offset by one cell off the plane to let "
            "the Yee curl run normally.' Measured with the port on node 0, every "
            "DUT returns |S11| = 1.000000 at every bin."),
        "what_it_costs": (
            "nothing electrically: apply_pmc_faces zeroes hy[0], the half-cell "
            "between node 0 and node 1, so no current flows behind the port. "
            "That is a zero-length open, not a one-cell stub."),
    },
    {
        "what": "the one-cell-wide channel is realized with PMC side walls, not two cells",
        "predeclaration_says": ("if the solver's one-cell-wide magnetic-wall channel "
                                "does not hold the TEM field, the width is realized "
                                "as two cells or with periodic side walls"),
        "why": ("it does hold it once the port is off the x_lo wall. apply_pmc_faces "
                "on a y face zeroes hx and hz, not hy, so the TEM pair (Ez, Hy) is "
                "untouched by the side walls."),
        "what_it_costs": "nothing; the width stays one cell at every rung.",
    },
    {
        "what": "the open termination's realized plane is half a cell short of L",
        "predeclaration_says": "L = 30 mm, port cell centre to the termination plane",
        "why": (
            "a short is pinned on an E node (apply_pec_faces zeroes the tangential E "
            "on the x_hi node) and an open on an H half-node (apply_pmc_faces zeroes "
            "hy[-2], the last interior half-cell). The two cannot both land on an "
            "integer number of cells from the port node."),
        "what_it_costs": (
            "the open's realized length is (N_L - 0.5) cells and shrinks toward "
            "30 mm as the mesh refines: 29.5, 29.75, 29.875 mm. Every analytic "
            "comparison in this fixture uses the REALIZED length, which is exact "
            "at every rung; the declared-length comparison is recorded beside it."),
    },
]


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
    print(f"[lw-battery {stamp}] {msg}", flush=True)


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


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------

def preflight_record(sim) -> dict:
    """The preflight report, verbatim text and structured findings.

    Captured with warnings silenced so the report's own findings do not land in
    the solve's warning list twice; the solve itself runs with
    ``skip_preflight=True`` for the same reason, and this record is the one
    place the preflight text lives.
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
# the closed forms — plain functions, unit-tested in the replay test against a
# hand-computed value. Nothing here touches the solver.
# ---------------------------------------------------------------------------

def line_zc(kind: str, eps_r: float = EPS_R_AIR) -> float:
    """Characteristic impedance of the channel: ``eta0 * h / w / sqrt(eps_r)``
    with ``h`` = ``N_H[kind]`` cells and ``w`` = one cell, so the ratio is a
    whole number and Zc does not move with the cell size."""
    return ETA0 * N_H[kind] / math.sqrt(eps_r)


def line_beta(freq_hz, eps_r: float = EPS_R_AIR):
    """TEM phase constant ``omega sqrt(eps_r) / c``. Continuum, not the
    lattice's numerical dispersion — that difference is recorded separately."""
    return 2.0 * math.pi * np.asarray(freq_hz, dtype=float) * math.sqrt(eps_r) / C0


def zin_terminated_line(zc: float, beta, length_m: float, z_load):
    """Input impedance of a lossless line of length ``length_m`` terminated in
    ``z_load``. ``z_load=None`` means an open, ``0.0`` a short.

    ``Zin = Zc (ZL + j Zc tan(beta L)) / (Zc + j ZL tan(beta L))``.
    """
    t = np.tan(np.asarray(beta, dtype=float) * length_m)
    if z_load is None:
        return zc / (1j * t)
    zl = complex(z_load)
    return zc * (zl + 1j * zc * t) / (zc + 1j * zl * t)


def s11_from_zin(zin, zref: float):
    """``(Zin - Zref) / (Zin + Zref)``."""
    return (zin - zref) / (zin + zref)


def gamma_load(z_load, zc: float) -> complex:
    """``(ZL - Zc)/(ZL + Zc)``; +1 for an open, -1 for a short."""
    if z_load is None:
        return 1.0 + 0.0j
    zl = complex(z_load)
    return (zl - zc) / (zl + zc)


def s11_closed_form(zc: float, beta, length_m: float, z_load, zref: float):
    """The battery's referee. With ``zref == zc`` this equals
    ``gamma_load * exp(-2j beta L)`` exactly, which is the identity the replay
    test checks the implementation against."""
    return s11_from_zin(zin_terminated_line(zc, beta, length_m, z_load), zref)


def ds11_dr(zc: float, beta, length_m: float, r_ohm: float, zref: float):
    """``d S11 / d R`` for a resistive load, from the chain rule on the two
    closed forms above:

        dS11/dZin = 2 Zref / (Zin + Zref)^2
        dZin/dR   = Zc^2 (1 + t^2) / (Zc + j R t)^2,   t = tan(beta L)

    With ``zref == zc`` the product reduces to ``exp(-2j beta L) * 2 Zc /
    (R + Zc)^2``, which is the derivative of ``Gamma_L exp(-2j beta L)`` — the
    cross-check the replay test runs.
    """
    t = np.tan(np.asarray(beta, dtype=float) * length_m)
    zin = zc * (r_ohm + 1j * zc * t) / (zc + 1j * r_ohm * t)
    dzin = zc ** 2 * (1.0 + t ** 2) / (zc + 1j * r_ohm * t) ** 2
    return 2.0 * zref * dzin / (zin + zref) ** 2


def ds11_deps_short(zc0: float, eps_r: float, beta, length_m: float, zref: float):
    """``d S11 / d eps_r`` for a SHORT-terminated filled line, with ``Zref``
    held at the value the port was built with (it is a static float on the
    port, not a traced quantity).

        Zc(eps)   = zc0 / sqrt(eps),        dZc/deps   = -Zc / (2 eps)
        beta(eps) = omega sqrt(eps) / c,    dbeta/deps =  beta / (2 eps)
        Zin       = j Zc tan(beta L)
        dZin/deps = j Zc / (2 eps) * [ L beta (1 + t^2) - t ]
    """
    zc = zc0 / math.sqrt(eps_r)
    b = np.asarray(beta, dtype=float)
    t = np.tan(b * length_m)
    zin = 1j * zc * t
    dzin = 1j * zc / (2.0 * eps_r) * (length_m * b * (1.0 + t ** 2) - t)
    return 2.0 * zref * dzin / (zin + zref) ** 2


def fit_electrical_length(s11_short, freqs, eps_r: float) -> dict:
    """The line's OWN electrical length, fitted from a short's phase.

    A short referenced to the line's own Zc gives ``S11 = -exp(-2 j beta L)``,
    so the unwrapped angle is ``pi - 2 beta L`` and is linear in frequency with
    slope ``-4 pi sqrt(eps_r) L / c``. A least-squares straight line through
    ``unwrap(angle(S11))`` against frequency therefore returns the length the
    LATTICE realizes, which is not the length that was drawn: the numerical
    dispersion and the half-cell the port and the termination sit on both land
    in it.

    The fit is `numpy.polyfit(freqs, unwrap(angle(S11)), 1)` over the whole
    band, unweighted. Its residual is reported so a reader can see whether the
    angle really was a straight line — on a lossless line it is, and a large
    residual would mean the record is not the line this fit assumes.
    """
    ang = np.unwrap(np.angle(np.asarray(s11_short)))
    f = np.asarray(freqs, dtype=float)
    slope, intercept = np.polyfit(f, ang, 1)
    resid = ang - (slope * f + intercept)
    length = -float(slope) * C0 / (4.0 * math.pi * math.sqrt(eps_r))
    return {
        "length_m": length,
        "slope_rad_per_hz": float(slope),
        "intercept_rad": float(intercept),
        "max_abs_residual_rad": float(np.abs(resid).max()),
        "rms_residual_rad": float(np.sqrt(np.mean(resid ** 2))),
        "eps_r": eps_r,
        "n_bins": int(f.size),
        "fit": ("least squares straight line through unwrap(angle(S11)) against "
                "frequency over the whole band, unweighted; "
                "L = -slope * c / (4 pi sqrt(eps_r)), from "
                "angle = pi - 2 beta L with beta = omega sqrt(eps_r) / c"),
        "what_it_is_not": ("not the drawn length. The lattice's numerical "
                           "dispersion and the half-cell the port and the "
                           "termination sit on are both inside it."),
    }


def numerical_beta(freq_hz, dx: float, dt: float, eps_r: float = EPS_R_AIR):
    """The Yee lattice's own phase constant along the propagation axis for a
    wave with no transverse variation:

        sin(beta dx / 2) = (dx / (c dt / sqrt(eps_r))) * sin(omega dt / 2)

    Recorded beside the continuum ``line_beta`` so a reader can see how much of
    any frequency offset the lattice already explains. It is a reference, not a
    correction applied to anything.
    """
    w = 2.0 * math.pi * np.asarray(freq_hz, dtype=float)
    s = (dx * math.sqrt(eps_r) / (C0 * dt)) * np.sin(w * dt / 2.0)
    s = np.clip(s, -1.0, 1.0)
    return 2.0 * np.arcsin(s) / dx


# ---------------------------------------------------------------------------
# the fixture's layout in cells — one function, so the declared and the
# asserted geometry cannot drift apart
# ---------------------------------------------------------------------------

def layout(kind: str, dut: str, dx: float) -> dict:
    """Where every plane lands, in nodes and in metres.

    ``n_l`` is the declared length in cells. A short is realized by PEC on the
    x_hi node, an open by PMC zeroing the last interior Hy half-cell, and a
    resistive load by lumped elements on a node with the PMC wall one cell
    beyond it, so each DUT gets the node count that puts its own termination
    plane where the pre-declaration asks.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown port kind {kind!r}; expected one of {KINDS}")
    if dut not in DUTS:
        raise ValueError(f"unknown dut {dut!r}; expected one of {DUTS}")
    n_l = int(round(L_LINE / dx))
    if not math.isclose(n_l * dx, L_LINE, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"L_LINE={L_LINE} is not a whole number of {dx} m cells")
    n_h = N_H[kind]
    if dut == "short":
        # PEC on the x_hi node zeroes the tangential E there.
        n_nodes = n_l + 2
        i_term = n_nodes - 1
        length_cells = float(i_term - PORT_NODE)
        term_note = "PEC on the x_hi node plane"
        i_rlc = None
    elif dut == "open":
        # PMC zeroes hy[-2]: the current vanishes half a cell inboard of x_hi.
        n_nodes = n_l + 2
        i_term = None
        length_cells = float(n_nodes - 2) + 0.5 - PORT_NODE
        term_note = "PMC on x_hi, so the last interior Hy half-cell carries no current"
        i_rlc = None
    else:
        # The elements sit one cell inboard of the wall, so the wall's zeroed
        # Hy half-cell is immediately beyond them and nothing follows the load.
        n_nodes = n_l + 3
        i_rlc = n_nodes - 2
        i_term = i_rlc
        length_cells = float(i_rlc - PORT_NODE)
        term_note = "lumped R on the node one cell inboard of the PMC x_hi wall"
    return {
        "kind": kind,
        "dut": dut,
        "dx_m": dx,
        "dx_um": dx * 1e6,
        "n_l_declared": n_l,
        "n_nodes_x": n_nodes,
        "n_cells_x": n_nodes - 1,
        "n_h_cells": n_h,
        "n_w_cells": 1,
        "i_port": PORT_NODE,
        "i_rlc": i_rlc,
        "i_term_node": i_term,
        "termination": term_note,
        "length_cells_realized": length_cells,
        "length_m_realized": length_cells * dx,
        "length_m_declared": L_LINE,
        "length_frac_from_declared": abs(length_cells * dx - L_LINE) / L_LINE,
        "domain_m": [(n_nodes - 1) * dx, dx, n_h * dx],
        "gap_h_m": n_h * dx,
        "width_w_m": dx,
        "x_hi_boundary": "pec" if dut == "short" else "pmc",
    }


def build_sim(kind: str, dut: str, dx: float, *, drive: str = DEFAULT_DRIVE,
              eps_r: float = EPS_R_AIR, precision: str = "float32") -> Simulation:
    """The channel of the pre-declaration at one cell size.

    The plates are the PEC walls on z, the side walls the PMC walls on y, and
    the port's own backing the PMC wall on x_lo. Nothing is drawn as geometry
    except the dielectric fill, so there is no rasterized conductor to lose.
    """
    lay = layout(kind, dut, dx)
    lx, ly, lz = lay["domain_m"]
    sim = Simulation(
        freq_max=F_HI, domain=(lx, ly, lz), dx=dx, precision=precision,
        boundary=BoundarySpec(
            x=Boundary(lo="pmc", hi=lay["x_hi_boundary"]),
            y=Boundary(lo="pmc", hi="pmc"),
            z=Boundary(lo="pec", hi="pec")),
    )
    if eps_r != EPS_R_AIR:
        sim.add_material("fill", eps_r=eps_r)
        sim.add(Box((0.0, 0.0, 0.0), (lx, ly, lz)), material="fill")
    zc = line_zc(kind, eps_r)
    wf = GaussianPulse(**DRIVES[drive])
    x_port = PORT_NODE * dx
    extra = {} if kind == "lumped" else {"extent": lz}
    sim.add_port(position=(x_port, 0.0, 0.0), component="ez",
                 impedance=zc, waveform=wf, **extra)
    if lay["i_rlc"] is not None:
        r_total = RESISTIVE[dut] * zc
        for k in range(lay["n_h_cells"]):
            sim.add_lumped_rlc(position=(lay["i_rlc"] * dx, 0.0, k * dx),
                               component="ez", R=r_total / lay["n_h_cells"],
                               topology="parallel")
    return sim


def declared(kind: str, dut: str, dx: float, eps_r: float = EPS_R_AIR) -> dict:
    lay = dict(layout(kind, dut, dx))
    zc = line_zc(kind, eps_r)
    lay.update({
        "eps_r": eps_r,
        "zc_ohm": zc,
        "zref_ohm": zc,
        "z_load_ohm": (None if dut == "open"
                       else (0.0 if dut == "short" else RESISTIVE[dut] * zc)),
        "r_total_ohm": (RESISTIVE[dut] * zc if dut in RESISTIVE else None),
        "r_per_cell_ohm": (RESISTIVE[dut] * zc / lay["n_h_cells"]
                           if dut in RESISTIVE else None),
        "gamma_load": None,
        "freqs_hz": [F_LO, F_HI, N_FREQS],
        "cells_per_wavelength_at_f_hi": (C0 / (F_HI * math.sqrt(eps_r))) / dx,
    })
    g = gamma_load(lay["z_load_ohm"] if dut != "short" else 0.0, zc)
    lay["gamma_load"] = {"real": g.real, "imag": g.imag, "abs": abs(g)}
    return lay


# ---------------------------------------------------------------------------
# realized geometry — measured from the built grid before any FDTD step, and
# from the port spec the solve itself returns
# ---------------------------------------------------------------------------

def realized_grid(sim: Simulation, kind: str, dut: str, dx: float) -> dict:
    """What the SIMULATION holds, resolved onto the grid it will build.

    Every index here comes from the position the simulation itself carries
    (``sim._ports``, ``sim._lumped_rlc``) and not from ``layout``, so the
    comparison in ``assert_realized_grid`` is between two independent things.
    Reading the declared node back out of ``layout`` and asserting it equals
    ``layout`` is the tautology this function exists to avoid: measured that
    way, the guard accepted a port built on node 0 and a load moved a whole
    cell, because both sides moved together.
    """
    grid = sim._build_grid()
    shape = [int(s) for s in grid.shape]
    ports = list(getattr(sim, "_ports", []))
    elements = list(getattr(sim, "_lumped_rlc", []))
    out = {
        "grid_shape_nodes": shape,
        "n_cells": int(np.prod([max(s - 1, 1) for s in shape])),
        "dt_s": float(grid.dt),
        "courant_c_dt_over_dx": float(C0 * grid.dt / dx),
        "n_ports": len(ports),
        "boundary_faces": {
            "pec": sorted(getattr(grid, "pec_faces", set()) or set()),
            "pmc": sorted(getattr(grid, "pmc_faces", set()) or set()),
        },
    }
    if ports:
        p = ports[0]
        out["port_index"] = [int(v) for v in grid.position_to_index(p.position)]
        out["port_position_m"] = [float(v) for v in p.position]
        out["port_component"] = p.component
        out["port_impedance_ohm"] = float(p.impedance)
        out["port_extent_m"] = None if p.extent is None else float(p.extent)
        out["port_extent_cells"] = (None if p.extent is None
                                    else int(round(p.extent / dx)))
        out["port_excite"] = bool(p.excite)
    out["n_rlc_elements"] = len(elements)
    if elements:
        out["rlc_indices"] = [[int(v) for v in grid.position_to_index(e.position)]
                              for e in elements]
        out["rlc_positions_m"] = [[float(v) for v in e.position] for e in elements]
        out["rlc_values_ohm"] = [float(e.R) for e in elements]
        out["rlc_components"] = [e.component for e in elements]
    return out


def assert_realized_grid(sim: Simulation, kind: str, dut: str, dx: float,
                        eps_r: float = EPS_R_AIR) -> dict:
    """Refuse to solve unless the built grid IS the declared channel.

    ``eps_r`` is the filling the caller BUILT with. It has to be passed, because
    the port's reference impedance is the filled line's Zc: a stage that builds
    an eps_r = 2.2 line and asks this function for the air line's declaration is
    comparing two different channels. That mismatch is what the first run of the
    permittivity stage hit once this guard stopped comparing the layout to
    itself.

    No FDTD step runs here. What it catches: a domain that rounded to a
    different node count, a port or element that snapped to the wrong node
    (``position_to_index`` rounds to the NEAREST node, so a half-cell offset in
    a position silently moves an element one cell), and a boundary face that did
    not come out of the BoundarySpec the way it was written.
    """
    m = realized_grid(sim, kind, dut, dx)
    lay = layout(kind, dut, dx)
    d = declared(kind, dut, dx, eps_r)
    problems = []
    m["eps_r_declared"] = eps_r
    want_shape = [lay["n_nodes_x"], 2, lay["n_h_cells"] + 1]
    if m["grid_shape_nodes"] != want_shape:
        problems.append(f"grid is {m['grid_shape_nodes']} nodes, declared {want_shape}")
    if m["n_ports"] != 1:
        problems.append(f"{m['n_ports']} port(s) registered, declared 1")
    else:
        if m["port_index"] != [PORT_NODE, 0, 0]:
            problems.append(f"the port resolved to node {m['port_index']}, declared "
                            f"{[PORT_NODE, 0, 0]}")
        if not math.isclose(m["port_impedance_ohm"], d["zref_ohm"], rel_tol=1e-9):
            problems.append(f"port impedance {m['port_impedance_ohm']} ohm, declared "
                            f"{d['zref_ohm']}")
        want_extent = None if kind == "lumped" else lay["n_h_cells"]
        if m["port_extent_cells"] != want_extent:
            problems.append(f"port extent {m['port_extent_cells']} cell(s), declared "
                            f"{want_extent} for a {kind} port")
        if not m["port_excite"]:
            problems.append("the port is passive; this battery drives its one port")
    want_pec = {"z_lo", "z_hi"} | ({"x_hi"} if dut == "short" else set())
    want_pmc = {"x_lo", "y_lo", "y_hi"} | (set() if dut == "short" else {"x_hi"})
    if set(m["boundary_faces"]["pec"]) != want_pec:
        problems.append(f"PEC faces {m['boundary_faces']['pec']}, declared {sorted(want_pec)}")
    if set(m["boundary_faces"]["pmc"]) != want_pmc:
        problems.append(f"PMC faces {m['boundary_faces']['pmc']}, declared {sorted(want_pmc)}")
    want_n = lay["n_h_cells"] if lay["i_rlc"] is not None else 0
    if m["n_rlc_elements"] != want_n:
        problems.append(f"{m['n_rlc_elements']} load element(s), declared {want_n}")
    elif want_n:
        want = [[lay["i_rlc"], 0, k] for k in range(lay["n_h_cells"])]
        if m["rlc_indices"] != want:
            problems.append(f"the load resolved to nodes {m['rlc_indices']}, "
                            f"declared {want}")
        total = sum(m["rlc_values_ohm"])
        if not math.isclose(total, d["r_total_ohm"], rel_tol=1e-9):
            problems.append(f"the load sums to {total} ohm, declared "
                            f"{d['r_total_ohm']}")
    if problems:
        raise RuntimeError(
            "assert_realized_grid: the built channel is not the declared channel "
            "— refusing to solve. " + "; ".join(problems)
            + f" [declared: {lay}] [measured: {m}]")
    return m


def port_spec_record(result, kind: str, dut: str, dx: float) -> dict:
    """The port specification the solve itself carried, and the check that it
    is the declared port.

    This runs AFTER the solve because the resolved port object only exists
    inside the compiled result. The solve is seconds long, so the cost of
    finding a wrong port here rather than before is a few seconds; what matters
    is that no RECORD is written with a port the driver did not verify.
    """
    lay = layout(kind, dut, dx)
    specs = result.lumped_port_sparams if kind == "lumped" else result.wire_port_sparams
    if not specs:
        raise RuntimeError(
            f"port_spec_record: forward() returned no {kind} port accumulator. "
            f"lumped={bool(result.lumped_port_sparams)} "
            f"wire={bool(result.wire_port_sparams)} — the port was registered as "
            "the other kind, so the S11 would not be this fixture's.")
    if len(specs) != 1:
        raise RuntimeError(f"port_spec_record: {len(specs)} ports, declared 1")
    spec = specs[0][0]
    # The two specs name their cell differently: a lumped port carries the one
    # cell it occupies as (i, j, k), a wire port the MIDPOINT of its live run as
    # (mid_i, mid_j, mid_k) plus the run itself in live_cells.
    if kind == "lumped":
        cell = [int(spec.i), int(spec.j), int(spec.k)]
    else:
        cell = [int(spec.mid_i), int(spec.mid_j), int(spec.mid_k)]
    rec = {
        "kind": kind,
        "port_cell": cell,
        "component": spec.component,
        "impedance_ohm": float(spec.impedance),
        "n_freqs": int(np.asarray(spec.freqs).size),
    }
    problems = []
    if not math.isclose(rec["impedance_ohm"], declared(kind, dut, dx)["zref_ohm"],
                        rel_tol=1e-9):
        problems.append(f"port impedance {rec['impedance_ohm']} ohm, declared "
                        f"{declared(kind, dut, dx)['zref_ohm']}")
    if kind == "lumped":
        if cell != [PORT_NODE, 0, 0]:
            problems.append(f"port cell {cell}, declared {[PORT_NODE, 0, 0]}")
    else:
        live = tuple(getattr(spec, "live_cells", ()) or ())
        rec["live_cells"] = [[int(v) for v in c] for c in live]
        rec["n_live"] = len(live)
        rec["excite"] = bool(getattr(spec, "excite", True))
        want = [[PORT_NODE, 0, k] for k in range(lay["n_h_cells"])]
        if rec["live_cells"] != want:
            problems.append(f"live wire cells {rec['live_cells']}, declared {want} "
                            "— the gap is not bridged by the declared run")
        if cell[0] != PORT_NODE or cell[1] != 0:
            problems.append(f"the wire port's midpoint cell is {cell}, declared "
                            f"column {[PORT_NODE, 0]}")
        if not rec["excite"]:
            problems.append("the wire port came back passive; the driven whole-port "
                            "diagonal is the only validated wire reading")
    if problems:
        raise RuntimeError("port_spec_record: the realized port is not the declared "
                           "port — refusing to write this record. "
                           + "; ".join(problems))
    return rec


# ---------------------------------------------------------------------------
# solving
# ---------------------------------------------------------------------------

def solve_s11(sim, *, num_periods: float, freqs=None, **kw):
    """One ``forward(port_s11_freqs=...)`` call. ``skip_preflight=True``
    because ``preflight_record`` already captured the report; leaving it on
    would put the same findings into the solve's warning list a second time."""
    return sim.forward(port_s11_freqs=jnp.asarray(FREQS if freqs is None else freqs),
                       num_periods=float(num_periods), skip_preflight=True, **kw)


def n_steps_for(sim, num_periods: float) -> int:
    return int(sim._build_grid().num_timesteps(num_periods=num_periods))


def _witness_line(tag: str, s11: np.ndarray, extra: str = "") -> None:
    mag = np.abs(s11)
    k = int(np.argmax(mag))
    _log(f"{tag}: max|S11| {mag[k]:.5f} at {FREQS[k] / 1e9:.3f} GHz | "
         f"min {mag.min():.5g} | mean {mag.mean():.5f}{extra}")


def _s11_of(result) -> np.ndarray:
    s = np.asarray(result.s_params)
    if s.ndim != 1:
        s = s.reshape(-1)
    if s.size != N_FREQS:
        raise RuntimeError(f"S11 came back with {s.size} bins, declared {N_FREQS}")
    return s


def _base(args, stage: str, kind: str | None, dut: str | None,
          dx: float | None) -> dict:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "kind": kind,
        "dut": dut,
        "rung_um": None if dx is None else round(dx * 1e6),
        "predeclaration": PREDECLARATION,
        "contract": CONTRACT,
        "driver": DRIVER,
        "bar": BAR,
        "deviations": DEVIATIONS,
        "provenance": provenance(args),
    }


def cost_estimate(kind: str, dut: str, dx: float, num_periods: float) -> dict:
    sim = build_sim(kind, dut, dx)
    grid = sim._build_grid()
    nodes = [int(s) for s in grid.shape]
    n = int(np.prod(nodes))
    n_steps = n_steps_for(sim, num_periods)
    est = {
        "grid_shape_nodes": nodes,
        "n_nodes": n,
        "dt_s": float(grid.dt),
        "num_periods": float(num_periods),
        "n_steps": n_steps,
        "node_steps": n * n_steps,
        "field_bytes_f32": 6 * n * 4,
    }
    print(f"[cost] {kind} {dut} dx={dx * 1e6:.0f} um: {n:,} nodes, {n_steps:,} steps "
          f"= {est['node_steps']:.3e} node-steps", flush=True)
    return est


# ---------------------------------------------------------------------------
# stage: pilot — record length and drive
# ---------------------------------------------------------------------------

def drive_spectrum_rel_db(f_hz: float, f0: float, bandwidth: float) -> float:
    """A differentiated Gaussian's amplitude at ``f_hz``, in dB below its own
    peak. ``|S(f)| ~ f exp(-(f/(f0 bw))^2)``, whose peak sits at
    ``f0 bw / sqrt(2)``. Closed form, no measurement."""
    w = f0 * bandwidth
    peak_f = w / math.sqrt(2.0)
    amp = f_hz * math.exp(-((f_hz / w) ** 2))
    peak = peak_f * math.exp(-0.5)
    return 20.0 * math.log10(max(amp, 1e-300) / peak)


def stage_pilot(args, out: Path) -> None:
    dx = args.rung * 1e-6
    kind, dut = args.kind, "open"
    rec = _base(args, "pilot", kind, dut, dx)
    rec["declared"] = declared(kind, dut, dx)
    rec["num_periods_ladder"] = list(PILOT_NUM_PERIODS)
    rec["drives"] = {k: dict(v) for k, v in DRIVES.items()}
    rec["drive_spectrum_rel_db"] = {
        k: {"at_band_lo": drive_spectrum_rel_db(F_LO, v["f0"], v["bandwidth"]),
            "at_band_hi": drive_spectrum_rel_db(F_HI, v["f0"], v["bandwidth"]),
            "peak_hz": v["f0"] * v["bandwidth"] / math.sqrt(2.0)}
        for k, v in DRIVES.items()}
    rec["cases"] = []
    _write(out, rec)

    for drive in DRIVES:
        for npd in PILOT_NUM_PERIODS:
            sim = build_sim(kind, dut, dx, drive=drive)
            geo = assert_realized_grid(sim, kind, dut, dx)
            pf = preflight_record(sim)
            est = cost_estimate(kind, dut, dx, npd)
            with _Captured() as cap:
                res = solve_s11(sim, num_periods=npd)
            s11 = _s11_of(res)
            _witness_line(f"pilot drive={drive} periods={npd}", s11)
            rec["cases"].append({
                "drive": drive,
                "num_periods": float(npd),
                "n_steps": est["n_steps"],
                "realized_grid": geo,
                "port_spec": port_spec_record(res, kind, dut, dx),
                "preflight": pf,
                "cost": est,
                "warnings": cap.warnings,
                "wall_s": cap.wall,
                "peak_memory": peak_memory(),
                "s11": _c(s11),
                "abs_s11": _f(np.abs(s11)),
                "max_abs_s11": float(np.abs(s11).max()),
            })
            _write(out, rec)          # persist after EVERY case


# ---------------------------------------------------------------------------
# stage: solve
# ---------------------------------------------------------------------------

def _one_solve(args, kind: str, dut: str, dx: float, out: Path) -> None:
    sim = build_sim(kind, dut, dx, drive=args.drive)
    geo = assert_realized_grid(sim, kind, dut, dx)
    pf = preflight_record(sim)
    est = cost_estimate(kind, dut, dx, args.num_periods)

    rec = _base(args, "solve", kind, dut, dx)
    rec.update({
        "drive": args.drive,
        "num_periods": float(args.num_periods),
        "declared": declared(kind, dut, dx),
        "realized_grid": geo,
        "preflight": pf,
        "cost": est,
    })
    _log(f"solve {kind} {dut} dx={dx * 1e6:.0f} um periods={args.num_periods}")
    with _Captured() as cap:
        res = solve_s11(sim, num_periods=args.num_periods)
    s11 = _s11_of(res)
    _witness_line(f"solve {kind} {dut} {dx * 1e6:.0f} um", s11)
    rec["port_spec"] = port_spec_record(res, kind, dut, dx)
    rec["warnings"] = cap.warnings
    rec["wall_s"] = cap.wall
    rec["peak_memory"] = peak_memory()
    rec["s11"] = _c(s11)
    rec["n_steps"] = est["n_steps"]
    _write(out, rec)                  # persist BEFORE the doubled record

    # Criterion 2's settling substitute: forward() emits no energy monitor, so
    # the record is doubled at a fixed absorber and the shift is the witness.
    sim2 = build_sim(kind, dut, dx, drive=args.drive)
    with _Captured() as cap2:
        res2 = solve_s11(sim2, num_periods=2.0 * args.num_periods)
    s11d = _s11_of(res2)
    _witness_line(f"solve {kind} {dut} {dx * 1e6:.0f} um DOUBLED", s11d)
    d = np.abs(s11d - s11)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-300))
    mag_db_d = 20.0 * np.log10(np.maximum(np.abs(s11d), 1e-300))
    rec["doubled"] = {
        "num_periods": 2.0 * args.num_periods,
        "n_steps": n_steps_for(sim2, 2.0 * args.num_periods),
        "s11": _c(s11d),
        "max_abs_diff": float(d.max()),
        "argmax_bin": int(np.argmax(d)),
        "max_abs_db_shift": float(np.abs(mag_db_d - mag_db).max()),
        "wall_s": cap2.wall,
        "warnings": cap2.warnings,
        "what_this_substitutes": (
            "forward(port_s11_freqs=...) emits no settling witness; the "
            "contract's admissible substitute is record-length invariance — "
            "double the window and hold the magnitude shift below one tenth of "
            "the magnitude gate."),
    }
    rec["peak_memory"] = peak_memory()
    _write(out, rec)


def stage_solve(args, out_dir: Path) -> None:
    duts = DUTS if args.all_duts else (args.dut,)
    rungs = RUNGS_UM if args.all_rungs else (args.rung,)
    for dut in duts:
        for um in rungs:
            _one_solve(args, args.kind, dut, um * 1e-6,
                       out_dir / f"solve_{args.kind}_{dut}_{um}um.json")


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
    kind, dut = args.kind, args.dut
    sim = build_sim(kind, dut, dx, drive=args.drive)
    geo = assert_realized_grid(sim, kind, dut, dx)
    pf = preflight_record(sim)

    rec = _base(args, "identity", kind, dut, dx)
    rec.update({"drive": args.drive, "num_periods": float(args.num_periods),
                "declared": declared(kind, dut, dx), "realized_grid": geo,
                "preflight": pf})

    _log("identity: plain call")
    with _Captured() as cap_plain:
        res_plain = solve_s11(sim, num_periods=args.num_periods)
    a = _s11_of(res_plain)
    _witness_line("identity plain", a)
    rec["port_spec"] = port_spec_record(res_plain, kind, dut, dx)
    rec["plain_s11"] = _c(a)
    rec["plain_warnings"] = cap_plain.warnings
    rec["plain_wall_s"] = cap_plain.wall
    _write(out, rec)

    eps = _own_eps(sim)
    rec["eps_override"] = {"shape": [int(s) for s in eps.shape],
                           "dtype": str(eps.dtype),
                           "min": float(np.min(np.asarray(eps))),
                           "max": float(np.max(np.asarray(eps))),
                           "n_above_vacuum": int(np.count_nonzero(
                               np.asarray(eps) > 1.0 + 1e-6))}
    _log("identity: no-op eps_override call")
    with _Captured() as cap_ov:
        res_ov = solve_s11(sim, num_periods=args.num_periods, eps_override=eps)
    b = _s11_of(res_ov)
    _witness_line("identity eps_override", b)
    rec["override_s11"] = _c(b)
    rec["override_warnings"] = cap_ov.warnings
    rec["override_wall_s"] = cap_ov.wall

    d = np.abs(a - b)
    rec["difference"] = {
        "max_abs": float(d.max()),
        "argmax_bin": int(np.argmax(d)),
        "max_rel": float((d / np.maximum(np.abs(a), 1e-300)).max()),
        "allclose_at_bar": bool(np.allclose(a, b, rtol=BAR["identity_rtol"],
                                            atol=BAR["identity_atol"])),
    }
    rec["peak_memory"] = peak_memory()
    _write(out, rec)


# ---------------------------------------------------------------------------
# stage: AD against a float64-loss FD
# ---------------------------------------------------------------------------

def _fd_ulp_span(f_plus: float, f_minus: float, dtype) -> float:
    """Resolving power of a central difference, in ULPs of ``dtype``.

    ``dtype`` is the dtype the LOSS was computed in, not the container the
    values arrived in: ``float(jnp_scalar)`` is always a Python float, so
    keying off the value alone measures float64 even for a float32 loss. Same
    expression as the gate in tests/unit/autodiff/test_msl_ad_fd_converged.py.
    """
    ulp = float(np.spacing(np.asarray(abs(0.5 * (f_plus + f_minus)), dtype=dtype)))
    return abs(f_plus - f_minus) / ulp


_AD_BAND_IDX = np.flatnonzero((FREQS >= AD_BAND[0]) & (FREQS <= AD_BAND[1]))
_AD_MID_BIN = int(np.argmin(np.abs(FREQS - AD_MID_BIN_HZ)))


def _objective(s11, which: str):
    """Two scalars of the differentiable S11. The band mask is taken on the
    CONCRETE frequency grid, so it is a static index set under tracing."""
    if which == "band_mean_s11_sq":
        return jnp.mean(jnp.abs(s11[jnp.asarray(_AD_BAND_IDX)]) ** 2)
    return jnp.real(s11[_AD_MID_BIN])


def _ad_fd_case(which: str, theta0: float, f32_loss, f64_loss, h_rel: float) -> dict:
    """One objective: reverse-mode AD in float32 against a central difference
    whose LOSS runs in float64, with the comparator's resolving power recorded
    before its accuracy."""
    _log(f"adfd {which}: AD (float32)")
    with _Captured() as cap_ad:
        loss, g = jax.value_and_grad(f32_loss)(jnp.float32(theta0))
    case = {
        "objective": which,
        "theta0": float(theta0),
        "fd_rel_h": h_rel,
        "ad": {"loss": float(loss), "grad": float(g),
               "loss_dtype": str(jnp.asarray(loss).dtype),
               "grad_finite": bool(np.isfinite(float(g))),
               "wall_s": cap_ad.wall, "warnings": cap_ad.warnings},
    }
    h = abs(theta0) * h_rel
    _log(f"adfd {which}: FD (float64 fields and loss, scoped x64)")
    with _Captured() as cap_fd:
        with enable_x64():
            f_plus, loss_dtype = f64_loss(theta0 + h)
            f_minus, _ = f64_loss(theta0 - h)
    g_fd = (f_plus - f_minus) / (2.0 * h)
    span = _fd_ulp_span(f_plus, f_minus, loss_dtype)
    interpretable = bool(span >= MIN_FD_ULP_SPAN)
    case["fd"] = {"f_plus": f_plus, "f_minus": f_minus, "h": h,
                  "grad": g_fd, "loss_dtype": str(loss_dtype),
                  "ulp_span": span, "wall_s": cap_fd.wall,
                  "warnings": cap_fd.warnings}
    # The resolving-power statement comes BEFORE the accuracy number, so a
    # comparator failure is recorded as a comparator failure.
    case["comparator"] = {"ulp_span": span, "floor": MIN_FD_ULP_SPAN,
                          "interpretable": interpretable}
    if interpretable:
        case["rel_err"] = abs(float(g) - g_fd) / max(abs(g_fd), 1e-300)
    else:
        case["rel_err"] = None
        case["rel_err_note"] = "not interpretable: FD span below the ULP floor"
    return case


def _pairwise(case: dict) -> dict:
    """The three pairwise distances the pre-declaration's fourth falsifier names:
    AD, FD and the closed-form derivative against each other.

    Each is ``|a - b| / max(|a|, |b|)``, which is symmetric and does not pick
    one of the pair as the reference — the falsifier is written pairwise, so
    the arithmetic is too. A pair in which both numbers are near zero is still
    reported; the magnitudes are beside it.
    """
    vals = {"ad": case["ad"]["grad"], "fd": case["fd"]["grad"],
            "closed_form": case.get("closed_form", {}).get("grad")}
    out = {"grads": vals, "rel": {}}
    names = [k for k, v in vals.items() if v is not None]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            denom = max(abs(vals[a]), abs(vals[b]), 1e-300)
            out["rel"][f"{a}_vs_{b}"] = abs(vals[a] - vals[b]) / denom
    out["max_rel"] = max(out["rel"].values()) if out["rel"] else None
    out["bar"] = BAR["ad_fd_rel"]
    out["interpretable"] = case["comparator"]["interpretable"]
    return out


def _f64_guard(arr):
    if arr.dtype != jnp.float64:
        raise RuntimeError(
            f"the FD reference did not run in float64 (got {arr.dtype}): JAX "
            "truncates a float64 request to float32 when x64 is off, so the "
            "scoped context failed to engage. A float32 reference resolves far "
            "too few ULPs to judge a gradient.")
    return float(arr), arr.dtype


def _bin_context(s11_meas: np.ndarray, s11_an: np.ndarray, which: str) -> dict:
    """Where on the curve each objective's derivative was taken.

    ``Re(S11)`` has derivative ``-sin(angle) d(angle)/dtheta``, so near a phase
    extremum the same physical line gives very different derivatives for a small
    difference in phase. The magnitude and angle of both curves at the bin are
    recorded so that a disagreement between the solver's derivative and the
    closed form's can be read against the phase offset that produced it, rather
    than only as a percentage.
    """
    def _at(arr, k):
        return {"abs": float(np.abs(arr[k])), "angle_rad": float(np.angle(arr[k])),
                "real": float(np.real(arr[k])), "imag": float(np.imag(arr[k])),
                "sin_angle": float(np.sin(np.angle(arr[k])))}
    if which == "band_mean_s11_sq":
        idx = _AD_BAND_IDX
        return {
            "bins": [int(i) for i in idx],
            "band_hz": list(AD_BAND),
            "measured_mean_abs_sq": float(np.mean(np.abs(s11_meas[idx]) ** 2)),
            "analytic_mean_abs_sq": float(np.mean(np.abs(s11_an[idx]) ** 2)),
        }
    k = _AD_MID_BIN
    return {"bin_index": k, "bin_hz": float(FREQS[k]),
            "measured": _at(s11_meas, k), "analytic": _at(s11_an, k),
            "angle_diff_rad": float(np.angle(s11_meas[k]) - np.angle(s11_an[k]))}


def _fitted_length(args, kind: str, dx: float, eps_r: float) -> dict:
    """Solve the SHORT of this channel at this rung and fit its length.

    One extra solve per AD stage, so the fitted length comes from the same
    commit, the same drive and the same record length as the gradients it sits
    beside rather than from another job's record.
    """
    sim = build_sim(kind, "short", dx, drive=args.drive, eps_r=eps_r)
    assert_realized_grid(sim, kind, "short", dx, eps_r)
    with _Captured():
        res = solve_s11(sim, num_periods=args.num_periods)
    s11 = _s11_of(res)
    fit = fit_electrical_length(s11, FREQS, eps_r)
    fit["declared_length_m"] = layout(kind, "short", dx)["length_m_realized"]
    fit["frac_from_declared"] = abs(
        fit["length_m"] - fit["declared_length_m"]) / fit["declared_length_m"]
    fit["s11_short"] = _c(s11)
    _log(f"fitted electrical length at {dx * 1e6:.0f} um, eps_r={eps_r}: "
         f"{fit['length_m'] * 1e3:.5f} mm against the declared "
         f"{fit['declared_length_m'] * 1e3:.5f} mm "
         f"({fit['frac_from_declared'] * 100:.3f} %), rms residual "
         f"{fit['rms_residual_rad']:.3e} rad")
    return fit


def stage_adfd_r(args, out: Path) -> None:
    """theta = the total load resistance, entered through ``rlc_values_override``."""
    dx = args.rung * 1e-6
    kind, dut = args.kind, args.dut
    if dut not in RESISTIVE:
        raise ValueError(f"--stage adfd-r needs a resistive DUT, got {dut!r}")
    lay = layout(kind, dut, dx)
    d = declared(kind, dut, dx)
    theta0 = d["r_total_ohm"]
    n_h = lay["n_h_cells"]

    sim = build_sim(kind, dut, dx, drive=args.drive)
    geo = assert_realized_grid(sim, kind, dut, dx)
    pf = preflight_record(sim)

    rec = _base(args, "adfd-r", kind, dut, dx)
    rec.update({
        "drive": args.drive, "num_periods": float(args.num_periods),
        "declared": d, "realized_grid": geo, "preflight": pf,
        "design_variable": {
            "what": "theta is the TOTAL load resistance in ohms; each of the "
                    "gap's cells carries theta / n_h through rlc_values_override",
            "n_elements": n_h, "theta0_ohm": theta0},
        "min_fd_ulp_span": MIN_FD_ULP_SPAN,
        "objectives": {
            "band_mean_s11_sq": {"band_hz": list(AD_BAND),
                                 "bins": [int(i) for i in _AD_BAND_IDX],
                                 "what": "mean over the band of |S11(f)|^2"},
            "re_s11_at_mid_band": {"bin_index": _AD_MID_BIN,
                                   "bin_hz": float(FREQS[_AD_MID_BIN]),
                                   "what": "Re(S11) at that bin"},
        },
        "cases": [],
    })
    _write(out, rec)

    fit = _fitted_length(args, kind, dx, EPS_R_AIR)
    rec["fitted_electrical_length"] = fit
    _write(out, rec)

    # The curve the objectives are scalars OF, at theta0. A gradient reported
    # without the S it differentiates cannot be read (workspace rule R5).
    with _Captured():
        s11_theta0 = _s11_of(solve_s11(sim, num_periods=args.num_periods))
    an_theta0 = s11_closed_form(d["zc_ohm"], line_beta(FREQS),
                                lay["length_m_realized"], theta0, d["zref_ohm"])
    rec["s11_at_theta0"] = _c(s11_theta0)
    rec["analytic_at_theta0"] = _c(an_theta0)
    _write(out, rec)

    def _make(simx, which):
        def loss(theta):
            ov = {i: {"R": theta / n_h} for i in range(n_h)}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = simx.forward(port_s11_freqs=jnp.asarray(FREQS),
                                 num_periods=float(args.num_periods),
                                 skip_preflight=True, rlc_values_override=ov)
            return _objective(r.s_params.reshape(-1), which)
        return loss

    for which in ("band_mean_s11_sq", "re_s11_at_mid_band"):
        f32 = _make(sim, which)

        def f64(theta, which=which):
            sim64 = build_sim(kind, dut, dx, drive=args.drive, precision="float64")
            g64, g32 = sim64._build_grid(), sim._build_grid()
            if (tuple(g64.shape), g64.dx, g64.dt) != (tuple(g32.shape), g32.dx, g32.dt):
                raise RuntimeError("the float64 referee did not build the same "
                                   "discrete rig as the float32 run")
            return _f64_guard(_make(sim64, which)(jnp.float64(theta)))

        case = _ad_fd_case(which, theta0, f32, f64, AD_FD_REL_H)
        # The closed form's own derivative, as a third witness beside AD and FD,
        # on BOTH lengths: the one that was drawn and the one the lattice
        # realizes. Which of the two carries criterion 3a is not settled here.
        case["closed_form"] = _closed_form_dr(kind, dut, dx, theta0, which,
                                              label="declared")
        case["closed_form_fitted_length"] = _closed_form_dr(
            kind, dut, dx, theta0, which, length_m=fit["length_m"],
            label="fitted from the short's phase at this rung")
        case["pairwise"] = _pairwise(case)
        case["pairwise_fitted_length"] = _pairwise(
            {**case, "closed_form": case["closed_form_fitted_length"]})
        case["bin_context"] = _bin_context(s11_theta0, an_theta0, which)
        rec["cases"].append(case)
        _write(out, rec)

    rec["peak_memory"] = peak_memory()
    _write(out, rec)


def _closed_form_dr(kind, dut, dx, r_ohm, which, length_m=None, label="") -> dict:
    """d(objective)/dR from the closed forms, on a stated line length."""
    lay = layout(kind, dut, dx)
    d = declared(kind, dut, dx)
    zc, zref = d["zc_ohm"], d["zref_ohm"]
    L = lay["length_m_realized"] if length_m is None else float(length_m)
    beta = line_beta(FREQS)
    s = s11_closed_form(zc, beta, L, r_ohm, zref)
    ds = ds11_dr(zc, beta, L, r_ohm, zref)
    if which == "band_mean_s11_sq":
        val = float(np.mean(np.abs(s[_AD_BAND_IDX]) ** 2))
        grad = float(np.mean(2.0 * np.real(np.conj(s[_AD_BAND_IDX]) * ds[_AD_BAND_IDX])))
    else:
        val = float(np.real(s[_AD_MID_BIN]))
        grad = float(np.real(ds[_AD_MID_BIN]))
    return {"loss": val, "grad": grad, "length_m": L, "length_source": label or "declared",
            "what": "the analytic derivative of the same objective on the stated "
                    "line length; a reference, compared with nothing here"}


def stage_adfd_eps(args, out: Path, dut: str = "short") -> None:
    """theta scales the permittivity of the filled line.

    Two terminations, and the difference between them is the point. On a SHORT
    the line is lossless and referenced to its own Zc, so ``|S11| = 1`` at every
    frequency: the band mean of ``|S11|^2`` is the constant 1 and its derivative
    is identically zero. That leg is the pre-declared one and is kept as
    measured. On the RESISTIVE termination the same objective moves with eps,
    so its derivative is a number AD and FD can both be wrong about.
    """
    dx = args.rung * 1e-6
    kind = args.kind
    d = declared(kind, dut, dx, eps_r=EPS_R_FILL)
    sim = build_sim(kind, dut, dx, drive=args.drive, eps_r=EPS_R_FILL)
    geo = assert_realized_grid(sim, kind, dut, dx, EPS_R_FILL)
    pf = preflight_record(sim)
    eps32 = _own_eps(sim, jnp.float32)
    n_filled = int(np.count_nonzero(np.asarray(eps32) > 1.0 + 1e-6))

    rec = _base(args, "adfd-eps" if dut == "short" else "adfd-eps-res",
                kind, dut, dx)
    rec.update({
        "drive": args.drive, "num_periods": float(args.num_periods),
        "declared": d, "realized_grid": geo, "preflight": pf,
        "eps_r_fill": EPS_R_FILL,
        "design_variable": {
            "what": "theta MULTIPLIES the whole eps_override array, so theta=1 is "
                    "the eps_r = 2.2 line and d/dtheta = eps_r * d/d(eps_r)",
            "n_cells_above_vacuum": n_filled,
            "n_cells_total": int(np.asarray(eps32).size), "theta0": 1.0},
        "min_fd_ulp_span": MIN_FD_ULP_SPAN,
        "objectives": {
            "band_mean_s11_sq": {
                "band_hz": list(AD_BAND),
                "bins": [int(i) for i in _AD_BAND_IDX],
                "what": "mean over the band of |S11(f)|^2",
                "degenerate_on_this_dut": bool(dut == "short"),
                "why": ("a short referenced to the line's own Zc gives |S11| = 1 "
                        "at every frequency, so this objective is the constant 1 "
                        "and its derivative is identically zero")
                if dut == "short" else
                ("a resistive termination makes |S11| depend on eps through both "
                 "Zc and beta, so this objective has a derivative to compare"),
            },
            "re_s11_at_mid_band": {"bin_index": _AD_MID_BIN,
                                   "bin_hz": float(FREQS[_AD_MID_BIN]),
                                   "what": "Re(S11) at that bin"},
        },
        "cases": [],
    })
    _write(out, rec)

    fit = _fitted_length(args, kind, dx, EPS_R_FILL)
    rec["fitted_electrical_length"] = fit
    _write(out, rec)

    with _Captured():
        s11_theta0 = _s11_of(solve_s11(sim, num_periods=args.num_periods))
    _zc_fill = ETA0 * N_H[kind] / math.sqrt(EPS_R_FILL)
    _zl = 0.0 if dut == "short" else d["r_total_ohm"]
    an_theta0 = s11_from_zin(
        zin_terminated_line(_zc_fill, line_beta(FREQS, EPS_R_FILL),
                            layout(kind, dut, dx)["length_m_realized"], _zl),
        d["zref_ohm"])
    rec["s11_at_theta0"] = _c(s11_theta0)
    rec["analytic_at_theta0"] = _c(an_theta0)
    _write(out, rec)

    def _make(simx, epsx, which):
        def loss(theta):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = simx.forward(port_s11_freqs=jnp.asarray(FREQS),
                                 num_periods=float(args.num_periods),
                                 skip_preflight=True, eps_override=epsx * theta)
            return _objective(r.s_params.reshape(-1), which)
        return loss

    objectives = (("band_mean_s11_sq", "re_s11_at_mid_band") if dut == "short"
                  else ("band_mean_s11_sq",))
    for which in objectives:
        f32 = _make(sim, eps32, which)

        def f64(theta, which=which):
            sim64 = build_sim(kind, dut, dx, drive=args.drive, eps_r=EPS_R_FILL,
                              precision="float64")
            g64, g32 = sim64._build_grid(), sim._build_grid()
            if (tuple(g64.shape), g64.dx, g64.dt) != (tuple(g32.shape), g32.dx, g32.dt):
                raise RuntimeError("the float64 referee did not build the same "
                                   "discrete rig as the float32 run")
            eps64 = _own_eps(sim64, jnp.float64)
            return _f64_guard(_make(sim64, eps64, which)(jnp.float64(theta)))

        case = _ad_fd_case(which, 1.0, f32, f64, AD_FD_REL_H)
        case["closed_form"] = _closed_form_deps(kind, dut, dx, which,
                                                label="declared")
        case["closed_form_fitted_length"] = _closed_form_deps(
            kind, dut, dx, which, length_m=fit["length_m"],
            label="fitted from the short's phase at this rung")
        case["pairwise"] = _pairwise(case)
        case["pairwise_fitted_length"] = _pairwise(
            {**case, "closed_form": case["closed_form_fitted_length"]})
        case["bin_context"] = _bin_context(s11_theta0, an_theta0, which)
        rec["cases"].append(case)
        _write(out, rec)

    rec["peak_memory"] = peak_memory()
    _write(out, rec)


def _closed_form_deps(kind, dut, dx, which, length_m=None, label="") -> dict:
    """d(objective)/dtheta from the closed forms, where theta scales eps_r, so
    the chain rule carries a factor of ``eps_r`` over ``d/d(eps_r)``.

    ``dut`` selects the termination: a short (the pre-declared leg, whose
    band-mean objective is identically constant) or the resistive load.
    """
    lay = layout(kind, dut, dx)
    d = declared(kind, dut, dx, eps_r=EPS_R_FILL)
    zc0, zref = ETA0 * N_H[kind], d["zref_ohm"]
    L = lay["length_m_realized"] if length_m is None else float(length_m)
    beta = line_beta(FREQS, EPS_R_FILL)
    zc = zc0 / math.sqrt(EPS_R_FILL)
    if dut == "short":
        s = s11_from_zin(zin_terminated_line(zc, beta, L, 0.0), zref)
        ds = ds11_deps_short(zc0, EPS_R_FILL, beta, L, zref) * EPS_R_FILL
    else:
        # A resistive load: Zc moves with eps and ZL does not, so the derivative
        # is taken by a central difference of the SAME closed form in float64.
        # Arithmetic on a formula, not on a solve.
        r = d["r_total_ohm"]
        h = 1e-6

        def _s_at(e):
            zc_e = zc0 / math.sqrt(e)
            return s11_from_zin(
                zin_terminated_line(zc_e, line_beta(FREQS, e), L, r), zref)
        s = _s_at(EPS_R_FILL)
        ds = (_s_at(EPS_R_FILL * (1.0 + h)) - _s_at(EPS_R_FILL * (1.0 - h))) / (2.0 * h)
    if which == "band_mean_s11_sq":
        val = float(np.mean(np.abs(s[_AD_BAND_IDX]) ** 2))
        grad = float(np.mean(2.0 * np.real(np.conj(s[_AD_BAND_IDX]) * ds[_AD_BAND_IDX])))
    else:
        val = float(np.real(s[_AD_MID_BIN]))
        grad = float(np.real(ds[_AD_MID_BIN]))
    return {"loss": val, "grad": grad, "length_m": L, "eps_r": EPS_R_FILL,
            "dut": dut, "length_source": label or "declared",
            "what": "the analytic derivative of the same objective w.r.t. the "
                    "same scaling theta; a reference, compared with nothing here"}


# ---------------------------------------------------------------------------
# stage: what the run() path returns on the same build
#
# NOT in the v2.0 chain. The contract scopes this family to the differentiable
# S11 of forward(port_s11_freqs=...) and says so in as many words; the S matrix
# of run(compute_s_params=True) is a numpy post-process. This stage exists
# because the two disagree on this channel in a way worth recording, and a
# number nobody wrote down is a number the next session measures again.
# ---------------------------------------------------------------------------

def stage_notinchain(args, out: Path) -> None:
    dx = args.rung * 1e-6
    kind = args.kind
    rec = _base(args, "notinchain", kind, None, dx)
    rec.update({
        "drive": args.drive, "num_periods": float(args.num_periods),
        "scope": ("run(compute_s_params=True) is NOT in the v2.0 chain for this "
                  "family. Nothing here is compared against a bar and nothing "
                  "here gates anything; it is recorded because the two paths "
                  "disagree on this channel."),
        "cases": [],
    })
    _write(out, rec)

    for dut in DUTS:
        sim_f = build_sim(kind, dut, dx, drive=args.drive)
        assert_realized_grid(sim_f, kind, dut, dx)
        n_steps = n_steps_for(sim_f, args.num_periods)
        with _Captured() as cap_f:
            s_fwd = _s11_of(solve_s11(sim_f, num_periods=args.num_periods))
        sim_r = build_sim(kind, dut, dx, drive=args.drive)
        with _Captured() as cap_r:
            res_r = sim_r.run(n_steps=n_steps, compute_s_params=True,
                              s_param_freqs=FREQS, skip_preflight=True)
        s_run = np.asarray(res_r.s_params).reshape(-1)
        d = np.abs(s_run - s_fwd)
        _log(f"notinchain {kind} {dut}: forward max|S11| {np.abs(s_fwd).max():.6f} | "
             f"run max|S11| {np.abs(s_run).max():.6f} | max|diff| {d.max():.6f}")
        rec["cases"].append({
            "dut": dut,
            "n_steps": n_steps,
            "forward_s11": _c(s_fwd),
            "run_s11": _c(s_run),
            "forward_abs": _f(np.abs(s_fwd)),
            "run_abs": _f(np.abs(s_run)),
            "run_abs_is_exactly_one": bool(np.all(np.abs(s_run) == 1.0)),
            "max_abs_difference": float(d.max()),
            "forward_warnings": cap_f.warnings,
            "run_warnings": cap_r.warnings,
        })
        _write(out, rec)
    rec["peak_memory"] = peak_memory()
    _write(out, rec)


# ---------------------------------------------------------------------------
# assemble — arithmetic only, no FDTD
# ---------------------------------------------------------------------------

def _cx(block) -> np.ndarray:
    return (np.asarray(block["real"], dtype=float)
            + 1j * np.asarray(block["imag"], dtype=float))


def phase_crossings(freqs: np.ndarray, s11: np.ndarray) -> dict:
    """Frequencies at which the unwrapped ``angle(S11)`` passes a multiple of
    pi, found by linear interpolation between the bracketing bins.

    On a lossless line the angle is ``angle(Gamma_L) - 2 beta L``, which is
    linear in frequency, so a linear interpolation between two bins costs far
    less than the bin width; the bin width is recorded beside every crossing so
    a reader can see what the sweep could resolve on its own.
    """
    ang = np.unwrap(np.angle(s11))
    step = float(freqs[1] - freqs[0])
    out = {"bin_width_hz": step, "unwrapped_angle_rad": ang.astype(float).tolist(),
           "crossings": []}
    lo, hi = ang.min(), ang.max()
    m_lo = int(math.floor(lo / math.pi)) - 1
    m_hi = int(math.ceil(hi / math.pi)) + 1
    for m in range(m_lo, m_hi + 1):
        level = m * math.pi
        for k in range(len(ang) - 1):
            a, b = ang[k], ang[k + 1]
            if (a - level) == 0.0:
                f = float(freqs[k])
            elif (a - level) * (b - level) < 0.0:
                f = float(freqs[k]) + (level - a) / (b - a) * step
            else:
                continue
            out["crossings"].append({
                "multiple_of_pi": m,
                "level_rad": level,
                "hz": f,
                "between_bins": [k, k + 1],
                "bin_width_hz": step,
            })
    return out


def analytic_crossings(zc: float, length_m: float, z_load, zref: float,
                       eps_r: float, f_lo: float, f_hi: float) -> list[dict]:
    """The same crossings from the closed form, solved rather than sampled.

    With ``zref == zc`` the angle is ``angle(Gamma_L) - 2 beta L`` exactly, so
    ``m pi = angle(Gamma_L) - 2 omega sqrt(eps_r) L / c`` has a closed solution
    in frequency for every integer ``m``.
    """
    g = gamma_load(z_load, zc)
    phi = cmath.phase(g)
    k = 2.0 * 2.0 * math.pi * math.sqrt(eps_r) * length_m / C0     # d(2 beta L)/df
    out = []
    m = -4000
    while m < 4000:
        f = (phi - m * math.pi) / k
        if f_lo <= f <= f_hi:
            out.append({"multiple_of_pi": m, "hz": f})
        m += 1
    return sorted(out, key=lambda r: r["hz"])


def match_crossings(measured: list[dict], analytic: list[dict]) -> list[dict]:
    """Pair each analytic crossing with the nearest measured one of the same
    parity (0 or pi), and record the fractional frequency distance."""
    rows = []
    for a in analytic:
        same = [m for m in measured if (m["multiple_of_pi"] - a["multiple_of_pi"]) % 2 == 0]
        if not same:
            rows.append({"analytic_hz": a["hz"], "multiple_of_pi": a["multiple_of_pi"],
                         "measured_hz": None, "frac": None})
            continue
        best = min(same, key=lambda m: abs(m["hz"] - a["hz"]))
        rows.append({
            "analytic_hz": a["hz"],
            "multiple_of_pi": a["multiple_of_pi"],
            "measured_hz": best["hz"],
            "measured_multiple_of_pi": best["multiple_of_pi"],
            "frac": abs(best["hz"] - a["hz"]) / a["hz"],
            "bin_width_frac": best["bin_width_hz"] / a["hz"],
        })
    return rows


def _solve_entry(rec: dict) -> dict:
    kind, dut = rec["kind"], rec["dut"]
    dx = rec["declared"]["dx_m"]
    s11 = _cx(rec["s11"])
    freqs = FREQS
    mag = np.abs(s11)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-300))
    d = rec["declared"]
    zc, zref = d["zc_ohm"], d["zref_ohm"]
    z_load = 0.0 if dut == "short" else d["z_load_ohm"]
    L_real = d["length_m_realized"]

    beta = line_beta(freqs)
    an_real = s11_closed_form(zc, beta, L_real, z_load, zref)
    an_decl = s11_closed_form(zc, beta, L_LINE, z_load, zref)
    beta_num = numerical_beta(freqs, dx, rec["realized_grid"]["dt_s"])
    an_lattice = s11_closed_form(zc, beta_num, L_real, z_load, zref)

    entry = {
        "kind": kind, "dut": dut, "rung_um": rec["rung_um"],
        "drive": rec["drive"], "num_periods": rec["num_periods"],
        "n_steps": rec["n_steps"],
        "declared": d,
        "realized_grid": rec["realized_grid"],
        "port_spec": rec["port_spec"],
        "preflight_text": rec["preflight"]["text"],
        "warnings": rec["warnings"],
        "wall_s": rec["wall_s"],
        "peak_memory": rec["peak_memory"],
        "freqs_hz": freqs.astype(float).tolist(),
        "s11": rec["s11"],
        "abs_s11": mag.astype(float).tolist(),
        "s11_db": mag_db.astype(float).tolist(),
        "max_abs_s11": float(mag.max()),
        "argmax_abs_s11_bin": int(np.argmax(mag)),
        "min_abs_s11": float(mag.min()),
        "mean_abs_s11": float(mag.mean()),
        "resolution": {
            "dx_m": dx,
            "cells_per_wavelength_at_f_hi": (C0 / F_HI) / dx,
            "cells_across_gap": d["n_h_cells"],
            "cells_along_line": d["length_cells_realized"],
        },
        "record_doubling": {
            # The doubled record's own S11 is kept, so the shift below can be
            # recomputed from the two curves rather than trusted as a number.
            "s11": rec["doubled"]["s11"],
            "max_abs_diff": rec["doubled"]["max_abs_diff"],
            "max_abs_db_shift": rec["doubled"]["max_abs_db_shift"],
            "n_steps": rec["doubled"]["n_steps"],
            "bar_db": BAR["record_doubling_db"],
            "within_bar": bool(rec["doubled"]["max_abs_db_shift"]
                               <= BAR["record_doubling_db"]),
        } if "doubled" in rec else None,
        "passivity": {
            "max_abs_s11": float(mag.max()),
            "bar": BAR["passivity_max"],
            "within_bar": bool(mag.max() <= BAR["passivity_max"]),
        },
        "referee": {
            "analytic_realized_length": _c(an_real),
            "analytic_declared_length": _c(an_decl),
            "analytic_lattice_beta": _c(an_lattice),
            "length_m_realized": L_real,
            "length_m_declared": L_LINE,
            "what": ("the same closed form on three references: the realized "
                     "line length, the declared 30 mm, and the realized length "
                     "with the Yee lattice's own beta. Recorded together because "
                     "which one a lattice measurement should be read against is "
                     "a question this battery does not settle."),
        },
    }

    # Magnitude against the closed form. |Gamma_L| is constant in frequency, so
    # for a reflecting DUT the comparison is a curve against a constant and the
    # dB distance is the bar's own quantity.
    #
    # The matched control is the exception the PI ruled on: its closed form is
    # identically zero, so a dB difference against it is a difference against a
    # floor constant and says nothing (it reads ~6000 dB). That case is judged
    # by its own upper bound below and carries no dB comparison at all — not a
    # comparison recorded as failing, which would be a number nobody may read.
    if dut == "matched":
        entry["magnitude_vs_analytic"] = {
            "applies": False,
            "why": ("the closed form is identically zero for a matched load, so "
                    "there is no magnitude to be within 2 dB of. The pre-declaration "
                    "holds this case to an upper bound instead; see matched_floor."),
            "abs_gamma_load": 0.0,
            "max_abs_diff": float(np.abs(mag - np.abs(an_real)).max()),
        }
    else:
        an_mag_db = 20.0 * np.log10(np.maximum(np.abs(an_real), 1e-300))
        entry["magnitude_vs_analytic"] = {
            "applies": True,
            "abs_gamma_load": abs(complex(d["gamma_load"]["real"],
                                          d["gamma_load"]["imag"])),
            "max_abs_db_diff": float(np.abs(mag_db - an_mag_db).max()),
            "argmax_bin": int(np.argmax(np.abs(mag_db - an_mag_db))),
            "max_abs_diff": float(np.abs(mag - np.abs(an_real)).max()),
            "bar_db": BAR["magnitude_db"],
            "within_bar": bool(np.abs(mag_db - an_mag_db).max() <= BAR["magnitude_db"]),
        }

    if dut == "matched":
        # A deep null: held to an upper bound, never compared in dB rung to rung.
        entry["matched_floor"] = {
            "max_abs_s11": float(mag.max()),
            "max_db": float(mag_db.max()),
            "bar_db": BAR["matched_floor_db"],
            "within_bar": bool(mag_db.max() <= BAR["matched_floor_db"]),
            "what": ("the port's own reflection floor at this cell size — the "
                     "number a user needs in order to know how well a port of "
                     "this kind can be matched."),
        }
    else:
        meas = phase_crossings(freqs, s11)
        an_cross = analytic_crossings(zc, L_real, z_load, zref, EPS_R_AIR, F_LO, F_HI)
        an_cross_decl = analytic_crossings(zc, L_LINE, z_load, zref, EPS_R_AIR,
                                           F_LO, F_HI)
        rows = match_crossings(meas["crossings"], an_cross)
        fracs = [r["frac"] for r in rows if r["frac"] is not None]
        # The slope of the unwrapped angle, measured and analytic. The crossing
        # test above cannot see a conjugated S: angle = phi - 2 beta L and
        # angle = phi + 2 beta L cross multiples of pi at the SAME frequencies,
        # so a flipped time convention would pass it unchanged. This pair of
        # numbers is where that would show. Recorded as a fact with no
        # threshold — the pre-declaration's phase test is the crossings, and
        # inventing a second criterion here is not this driver's to do.
        ang = np.unwrap(np.angle(s11))
        slope_meas = float(np.polyfit(freqs, ang, 1)[0])
        slope_an = float(np.polyfit(freqs, np.unwrap(np.angle(an_real)), 1)[0])
        entry["phase"] = {
            "angle_slope_rad_per_hz": {
                "measured": slope_meas,
                "analytic_realized_length": slope_an,
                "ratio": (slope_meas / slope_an) if slope_an != 0.0 else None,
                "same_sign": bool(slope_meas * slope_an > 0.0),
                "what": ("a least-squares slope of the unwrapped angle over the "
                         "whole band. The crossing comparison is blind to a "
                         "conjugated S because both conventions cross multiples "
                         "of pi at the same frequencies; this is not."),
            },
            "measured": meas,
            "analytic_realized_length": an_cross,
            "analytic_declared_length": an_cross_decl,
            "matched_realized": rows,
            "matched_declared": match_crossings(meas["crossings"], an_cross_decl),
            "max_frac": max(fracs) if fracs else None,
            "bar_frac": BAR["frequency_frac"],
            "within_bar": bool(fracs and max(fracs) <= BAR["frequency_frac"]),
            "n_crossings": len(rows),
        }
    return entry


def _ladder(fix: dict, kind: str, dut: str) -> dict | None:
    keys = [f"{kind}_{dut}_{um}um" for um in RUNGS_UM]
    have = [k for k in keys if k in fix["solves"]]
    if len(have) < 2:
        return None
    fine = fix["solves"][have[-1]]
    fine_db = np.asarray(fine["s11_db"], dtype=float)
    fine_abs = np.asarray(fine["abs_s11"], dtype=float)
    lad: dict = {
        "rungs_um": [int(k.rsplit("_", 1)[-1][:-2]) for k in have],
        "finest": have[-1],
        "max_abs_s11": [fix["solves"][k]["max_abs_s11"] for k in have],
        "record_doubling_db": [fix["solves"][k]["record_doubling"]["max_abs_db_shift"]
                               if fix["solves"][k]["record_doubling"] else None
                               for k in have],
        "length_m_realized": [fix["solves"][k]["declared"]["length_m_realized"]
                              for k in have],
        "rows": [],
    }
    for k in have:
        cur = fix["solves"][k]
        cur_db = np.asarray(cur["s11_db"], dtype=float)
        cur_abs = np.asarray(cur["abs_s11"], dtype=float)
        row = {
            "rung": k,
            "max_abs_diff_vs_finest": float(np.abs(cur_abs - fine_abs).max()),
            "passivity_within_bar": cur["passivity"]["within_bar"],
            "resolution": cur["resolution"],
        }
        # The flags that decide the recommended cell size are listed by name,
        # never collected by matching the key text: a renamed key would then
        # drop out of the decision silently and every rung would qualify.
        deciding = ["passivity_within_bar"]
        if dut == "matched":
            # A deep null is compared in AMPLITUDE, not in dB: the floor halves
            # with the cell size, and a ratio of dB values would read ~1 for a
            # sequence that is in fact first order.
            row["matched_floor_abs"] = cur["matched_floor"]["max_abs_s11"]
            row["matched_floor_db"] = cur["matched_floor"]["max_db"]
            row["matched_floor_within_bar"] = cur["matched_floor"]["within_bar"]
            deciding.append("matched_floor_within_bar")
        else:
            row["max_db_diff_vs_finest"] = float(np.abs(cur_db - fine_db).max())
            row["magnitude_within_2dB_vs_finest"] = bool(
                np.abs(cur_db - fine_db).max() <= BAR["magnitude_db"])
            row["magnitude_within_2dB_vs_analytic"] = \
                cur["magnitude_vs_analytic"]["within_bar"]
            fine_cross = [r["measured_hz"] for r in fine["phase"]["matched_realized"]]
            cur_cross = [r["measured_hz"] for r in cur["phase"]["matched_realized"]]
            pairs = [(a, b) for a, b in zip(cur_cross, fine_cross)
                     if a is not None and b is not None]
            row["crossing_frac_vs_finest"] = (
                max(abs(a - b) / b for a, b in pairs) if pairs else None)
            row["crossing_within_1pct_vs_finest"] = bool(
                pairs and max(abs(a - b) / b for a, b in pairs) <= BAR["frequency_frac"])
            row["crossing_frac_vs_analytic"] = cur["phase"]["max_frac"]
            row["crossing_within_1pct_vs_analytic"] = cur["phase"]["within_bar"]
            deciding += ["magnitude_within_2dB_vs_finest",
                         "crossing_within_1pct_vs_finest"]
        row["deciding_flags"] = deciding
        row["all_inside_bar"] = all(bool(row[f]) for f in deciding)
        lad["rows"].append(row)
    # The support matrix asks for one cell size: the coarsest rung whose every
    # quantity sits inside the bar against the finest. Arithmetic over the
    # thresholds already in `bar` — a boolean per rung, not a recommendation.
    qualifying = [r for r in lad["rows"] if r["all_inside_bar"]]
    lad["coarsest_rung_within_bar"] = qualifying[0]["rung"] if qualifying else None
    # Successive differences and their ratio, on the quantity the DUT has.
    if dut == "matched":
        seq = [fix["solves"][k]["matched_floor"]["max_abs_s11"] for k in have]
    else:
        seq = []
        for k in have:
            c = [r["measured_hz"] for r in fix["solves"][k]["phase"]["matched_realized"]
                 if r["measured_hz"] is not None]
            seq.append(c[0] if c else None)
    lad["ladder_sequence"] = seq
    lad["ladder_quantity"] = ("the matched control's worst |S11| in amplitude"
                              if dut == "matched"
                              else "the lowest measured phase crossing, in Hz")
    if all(v is not None for v in seq) and len(seq) >= 3:
        diffs = [abs(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)]
        lad["successive_diff"] = diffs
        lad["successive_diff_ratio"] = (None if diffs[0] == 0.0 else diffs[1] / diffs[0])
        lad["richardson"] = {
            f"order_{p}": seq[-1] + (seq[-1] - seq[-2]) / (2.0 ** p - 1.0)
            for p in (1, 2)}
        lad["richardson_note"] = ("arithmetic only — no claim about the order the "
                                  "sequence actually has; the successive-difference "
                                  "ratio beside it is what speaks to that")
    return lad


def fixture_provenance(p: dict, index: dict | None, stage_file: str) -> dict:
    """The provenance a committed artifact may carry.

    The stage JSONs record absolute paths because that is what a person
    debugging a job needs. A fixture is committed to a public repository, so the
    machine paths are reduced to the question they exist to answer — did
    ``import rfx`` resolve to this run's own tree, or to something installed
    elsewhere — and the run is named by its compute id rather than by a
    directory on one pod.
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


def openems_context() -> dict:
    """The repository's recorded openEMS lumped-port comparison, read as it
    stands.

    The producers are ``scripts/diagnostics/build_lumped_openems_sparameter_
    comparison.py`` and ``build_lumped_openems_sweep_comparison.py``. They write
    into ``.omx/physics-gate/...``, which this repository does not track, so
    until this battery committed one no clean clone had a copy. The committed
    file is read as it stands and quoted with its own scope limits; nothing in
    this artifact is compared against it.
    """
    root = REPO / "tests" / "fixtures" / "lumped_wire_chain_battery"
    candidates = ["lumped_openems_pec_box_sweep_comparison.json"]
    out: dict = {
        "producers": [
            "scripts/diagnostics/build_lumped_openems_sparameter_comparison.py",
            "scripts/diagnostics/build_lumped_openems_sweep_comparison.py",
        ],
        "producer_default_output_dir": ".omx/physics-gate/latest-lumped-openems-generic-comparison",
        "records": {},
        "note": None,
    }
    for name in candidates:
        p = root / name
        if p.exists():
            out["records"][name] = json.loads(p.read_text())
    if out["records"]:
        out["note"] = (
            "context, not a comparison. The record is on a two-port 50 ohm PEC "
            "box at 0.8-1.8 GHz with dx = 5 mm, a different fixture from this "
            "battery's parallel-plate line, and nothing in this artifact is "
            "compared against it. It carries its own claim_scope and "
            "completion_decision; read those before quoting it. Its _committed_as "
            "block records that it was found in an untracked runtime directory "
            "and what was edited to commit it.")
    else:
        out["note"] = (
            "no stored openEMS lumped comparison is committed in this tree. Both "
            "producers write into the untracked .omx/ runtime directory, so "
            "neither leaves a committed artifact on its own. The pre-declaration "
            "lists this comparison as CONTEXT for stage 3d and compares nothing "
            "against it; the referee this battery uses is the closed form above.")
    return out


KNOWN_LOAD_RECORD = "scripts/diagnostics/lumped_port_known_load_line.json"
# The name the known-load record carries when a battery job re-runs its
# producer and copies the output into the run directory.
KNOWN_LOAD_REPRODUCTION = "lumped_port_known_load_line.json"


def port_kind_ab(out_dir: Path | None = None, index: dict | None = None) -> dict:
    """The A/B the lumped leg turns on, read from the record its own script
    wrote rather than recomputed here.

    ``scripts/diagnostics/lumped_port_known_load_line.py`` declares ONE cell of
    ONE line two ways — a lumped port and a wire port, differing by `extent=dx`
    on `add_port` — in front of three loads whose reflection is an exact number.
    Nothing in this function solves anything or changes a digit; it lifts the
    three loads, five bins, both port kinds, their complex S11 and both
    `V/(Zc I)` columns into the artifact so a reader of the fixture does not
    have to run the script.

    The committed record names the commit it was written at, which need not be
    the battery's. When the battery's own job re-ran the producer, that output
    sits in ``out_dir`` under ``KNOWN_LOAD_REPRODUCTION`` and is lifted beside the
    record as ``reproduction``, complex S11 and commit included, so the record
    is carried together with a run at the battery's commit that reproduced it.
    """
    path = REPO / KNOWN_LOAD_RECORD
    out = {"producer": "scripts/diagnostics/lumped_port_known_load_line.py",
           "record": KNOWN_LOAD_RECORD}
    if not path.exists():
        out["present"] = False
        out["note"] = ("the record has not been produced in this tree; run the "
                       "producer with no arguments to write it")
        return out
    d = json.loads(path.read_text())
    out["present"] = True
    out["commit"] = d.get("commit")
    out["channel"] = d.get("channel")
    out["freqs_hz"] = d.get("freqs_hz")
    out["what"] = (
        "one cell of one line declared two ways. The only difference between "
        "the two rows of each load is extent=dx on add_port. Zref is the line's "
        "own Zc, so |S11| is |Gamma_L| at every bin whatever the length and "
        "whatever beta.")
    out["loads"] = {}
    for name, e in d.get("loads", {}).items():
        row = {"r_over_zc": e["r_over_zc"], "r_ohm": e["r_ohm"],
               "closed_form_abs_s11": e["closed_form_abs_s11"]}
        for kind, m in e["ports"].items():
            row[kind] = {
                "abs_s11": m["abs_s11"],
                "s11_real": m["s11_real"],
                "s11_imag": m["s11_imag"],
                "v_over_zc_i_real": m["v_over_zc_i_real"],
                "v_over_zc_i_imag": m["v_over_zc_i_imag"],
                "nonpassive_warning": m["nonpassive_warning"],
                "max_abs_from_closed_form": float(np.max(np.abs(
                    np.asarray(m["abs_s11"]) - e["closed_form_abs_s11"]))),
            }
        out["loads"][name] = row

    rep_path = None if out_dir is None else Path(out_dir) / KNOWN_LOAD_REPRODUCTION
    if rep_path is not None and rep_path.exists():
        r = json.loads(rep_path.read_text())
        entry = (index or {}).get(KNOWN_LOAD_REPRODUCTION, {})
        out["reproduction"] = {
            "commit": r.get("commit"),
            "rfx_resolved_inside_this_tree": r.get("rfx_resolved_inside_this_tree"),
            "compute_run_id": entry.get("vessl_run_id"),
            "compute_run_dir": entry.get("run_dir"),
            "freqs_hz": r.get("freqs_hz"),
            "loads": {name: {kind: {"s11_real": m["s11_real"],
                                    "s11_imag": m["s11_imag"]}
                             for kind, m in e["ports"].items()}
                      for name, e in r.get("loads", {}).items()},
            "what": ("the same producer, run with no arguments by a battery job "
                     "at the battery's commit, lifted as written. The record "
                     "above is carried because this run reproduced it."),
        }
    return out


def _degenerate_objective_evidence(fix: dict) -> dict:
    """What the ULP-span floor did on an objective with no derivative.

    Derived from the records already in the fixture: for every leg whose
    objective the driver marked degenerate, the span the comparator reported and
    the floor it was held to. The floor is computed on the two LOSS values, so
    it passes when those differ by many ULPs even though their DIFFERENCE is
    round-off. This block states that with the numbers beside it; it changes no
    threshold and gates nothing.
    """
    rows = []
    for name, block in sorted(fix.get("adfd", {}).items()):
        obj = block.get("objectives", {}).get("band_mean_s11_sq", {})
        if not obj.get("degenerate_on_this_dut"):
            continue
        for case in block["cases"]:
            if case["objective"] != "band_mean_s11_sq":
                continue
            rows.append({
                "leg": name, "rung_um": block["rung_um"], "dut": block["dut"],
                "ulp_span": case["fd"]["ulp_span"],
                "floor": block["min_fd_ulp_span"],
                "span_above_floor": bool(case["fd"]["ulp_span"]
                                         >= block["min_fd_ulp_span"]),
                "ad_grad": case["ad"]["grad"],
                "fd_grad": case["fd"]["grad"],
                "closed_form_grad": case["closed_form"]["grad"],
                "ad_loss": case["ad"]["loss"],
                "rel_err_ad_vs_fd": case["rel_err"],
            })
    return {
        "rows": rows,
        "what_the_floor_is": ("|f_plus - f_minus| in ULPs of the loss, held to "
                              "MIN_FD_ULP_SPAN"),
        "what_it_did_here": ("passed at every rung listed above while both "
                             "gradients were round-off around a derivative that "
                             "is analytically zero"),
        "why": ("the floor is computed on the two LOSS values, which do differ "
                "by many ULPs. It is their DIFFERENCE that is round-off, and the "
                "floor as this repository defines it does not look at that."),
        "nothing_is_regated_here": True,
    }


def _load_stage(out: Path, name: str) -> dict | None:
    p = out / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def stage_assemble(args, out: Path, fixture_out: Path) -> None:
    index = json.loads(Path(args.run_index).read_text()) if args.run_index else None
    fix = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "predeclaration": PREDECLARATION,
        "contract": CONTRACT,
        "driver": DRIVER,
        "artifact": ARTIFACT,
        "bar": BAR,
        "deviations": DEVIATIONS,
        "assembled_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "assembler_commit": git_sha(),
        "channel": {
            "l_line_m": L_LINE, "n_h_cells": N_H, "n_w_cells": 1,
            "eta0": ETA0, "eps_r_air": EPS_R_AIR, "eps_r_fill": EPS_R_FILL,
            "zc_ohm": {k: line_zc(k) for k in KINDS},
            "port_node": PORT_NODE,
            "band_hz": [F_LO, F_HI, N_FREQS],
            "rungs_um": list(RUNGS_UM),
            "duts": list(DUTS),
            "resistive_r_over_zc": dict(RESISTIVE),
        },
        "freqs_hz": FREQS.astype(float).tolist(),
        "solves": {},
        "ladder": {},
        "identity": {},
        "adfd": {},
        "pilot": None,
        "openems_context": openems_context(),
        "port_kind_ab": port_kind_ab(out, index),
        "not_in_chain_observations": {},
    }

    for kind in KINDS:
        for dut in DUTS:
            for um in RUNGS_UM:
                name = f"solve_{kind}_{dut}_{um}um.json"
                rec = _load_stage(out, name)
                if rec is None:
                    continue
                entry = _solve_entry(rec)
                entry["provenance"] = fixture_provenance(rec["provenance"], index, name)
                fix["solves"][f"{kind}_{dut}_{um}um"] = entry

    for kind in KINDS:
        for dut in DUTS:
            lad = _ladder(fix, kind, dut)
            if lad is not None:
                fix["ladder"][f"{kind}_{dut}"] = lad

    pilot = _load_stage(out, f"pilot_{args.kind}_{COARSEST_RUNG_UM}um.json")
    if pilot is not None:
        fix["pilot"] = {
            "provenance": fixture_provenance(
                pilot["provenance"], index,
                f"pilot_{args.kind}_{COARSEST_RUNG_UM}um.json"),
            "kind": pilot["kind"], "dut": pilot["dut"], "rung_um": pilot["rung_um"],
            "drives": pilot["drives"],
            "drive_spectrum_rel_db": pilot["drive_spectrum_rel_db"],
            "cases": [{
                "drive": c["drive"], "num_periods": c["num_periods"],
                "n_steps": c["n_steps"], "max_abs_s11": c["max_abs_s11"],
                "abs_s11": c["abs_s11"], "wall_s": c["wall_s"],
                "warnings": c["warnings"],
            } for c in pilot["cases"]],
        }
        # What doubling the record moved, inside the pilot's own ladder.
        for drive in {c["drive"] for c in pilot["cases"]}:
            byp = {c["num_periods"]: _cx(c["s11"]) for c in pilot["cases"]
                   if c["drive"] == drive}
            shifts = []
            for p in sorted(byp):
                if 2.0 * p in byp:
                    a, b = byp[p], byp[2.0 * p]
                    da = np.abs(20.0 * np.log10(np.maximum(np.abs(b), 1e-300))
                                - 20.0 * np.log10(np.maximum(np.abs(a), 1e-300)))
                    shifts.append({"num_periods": p, "doubled_to": 2.0 * p,
                                   "max_abs_diff": float(np.abs(b - a).max()),
                                   "max_abs_db_shift": float(da.max())})
            fix["pilot"].setdefault("record_doubling", {})[drive] = shifts

    for kind in KINDS:
        ident = _load_stage(out, f"identity_{kind}_{COARSEST_RUNG_UM}um.json")
        if ident is not None:
            a = _cx(ident["plain_s11"])
            b = _cx(ident["override_s11"])
            fix["identity"][kind] = {
                "provenance": fixture_provenance(
                    ident["provenance"], index,
                    f"identity_{kind}_{COARSEST_RUNG_UM}um.json"),
                "kind": kind, "dut": ident["dut"], "rung_um": ident["rung_um"],
                "num_periods": ident["num_periods"],
                "plain_s11": ident["plain_s11"],
                "override_s11": ident["override_s11"],
                "eps_override": ident["eps_override"],
                "max_abs_diff": float(np.abs(a - b).max()),
                "rtol": BAR["identity_rtol"], "atol": BAR["identity_atol"],
                "allclose_at_bar": ident["difference"]["allclose_at_bar"],
                "warnings": {"plain": ident["plain_warnings"],
                             "override": ident["override_warnings"]},
            }
        # Every rung an AD leg was measured at, not only the coarsest. The
        # pre-declaration places stage 3a at the coarsest rung; a finer rung is
        # extra evidence about the same leg, so it is collected under its own
        # key rather than replacing the declared one.
        for leg in ("adfd-r", "adfd-eps", "adfd-eps-res"):
          for um in RUNGS_UM:
            name = f"{leg}_{kind}_{um}um.json"
            ad = _load_stage(out, name)
            if ad is None:
                continue
            fix["adfd"][f"{kind}_{leg}_{um}um"] = {
                "provenance": fixture_provenance(ad["provenance"], index, name),
                "kind": kind, "dut": ad["dut"], "rung_um": ad["rung_um"],
                "leg": leg,
                "predeclared_rung": bool(um == COARSEST_RUNG_UM),
                "num_periods": ad["num_periods"],
                "design_variable": ad["design_variable"],
                "objectives": ad["objectives"],
                "min_fd_ulp_span": ad["min_fd_ulp_span"],
                "s11_at_theta0": ad.get("s11_at_theta0"),
                "analytic_at_theta0": ad.get("analytic_at_theta0"),
                "fitted_electrical_length": ad.get("fitted_electrical_length"),
                "cases": ad["cases"],
                "bar": BAR["ad_fd_rel"],
                "what_the_ulp_span_does_not_say": (
                    "the span is |f_plus - f_minus| in ULPs of the loss, so it "
                    "answers whether the two LOSS values are resolved from each "
                    "other. It does not answer whether the DERIVATIVE is: an "
                    "objective whose true derivative is zero gives two losses "
                    "millions of ULPs apart whose difference is round-off, and "
                    "the span passes. Read each case's closed_form gradient and "
                    "the loss beside it before reading its rel_err."),
            }

    for kind in KINDS:
        nic = _load_stage(out, f"notinchain_{kind}_{COARSEST_RUNG_UM}um.json")
        if nic is None:
            continue
        # Derived here, from the stored arrays: is the run() path's S11 the
        # same complex array for every load? That is a sharper statement than
        # a magnitude near one, and it is the one the numbers support.
        runs = {c["dut"]: _cx(c["run_s11"]) for c in nic["cases"]}
        fwds = {c["dut"]: _cx(c["forward_s11"]) for c in nic["cases"]}
        ref = next(iter(runs.values())) if runs else None
        same = {d: bool(np.array_equal(v, ref)) for d, v in runs.items()}
        fwd_spread = (float(max(np.abs(v).max() for v in fwds.values())
                            - min(np.abs(v).min() for v in fwds.values()))
                      if fwds else None)
        fix["not_in_chain_observations"][kind] = {
            "provenance": fixture_provenance(
                nic["provenance"], index,
                f"notinchain_{kind}_{COARSEST_RUNG_UM}um.json"),
            "scope": nic["scope"],
            "rung_um": nic["rung_um"],
            "num_periods": nic["num_periods"],
            "cases": nic["cases"],
            "run_path_load_independence": {
                "identical_to_first_dut": same,
                "all_identical": bool(all(same.values())),
                "run_abs_min": (float(min(np.abs(v).min() for v in runs.values()))
                                if runs else None),
                "run_abs_max": (float(max(np.abs(v).max() for v in runs.values()))
                                if runs else None),
                "forward_abs_spread_across_duts": fwd_spread,
                "what": ("whether run(compute_s_params=True) returned the SAME "
                         "complex array for a short, an open, R = Zc/2, R = Zc "
                         "and R = 2 Zc on this channel, and what range the "
                         "forward path covered over the same five loads"),
            },
        }

    fix["degenerate_objective_evidence"] = _degenerate_objective_evidence(fix)

    fixture_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = fixture_out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(fix, indent=1))
    os.replace(tmp, fixture_out)
    _log(f"wrote {fixture_out} — {len(fix['solves'])} solves, "
         f"{len(fix['ladder'])} ladders")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("pilot", "solve", "identity",
                                        "adfd-r", "adfd-eps", "adfd-eps-res",
                                        "notinchain"))
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--kind", choices=KINDS, default="wire")
    ap.add_argument("--dut", choices=DUTS, default="res_double")
    ap.add_argument("--all-duts", action="store_true",
                    help="solve stage: loop every DUT inside this process")
    ap.add_argument("--rung", type=int, choices=RUNGS_UM, default=COARSEST_RUNG_UM)
    ap.add_argument("--all-rungs", action="store_true",
                    help="solve stage: loop every rung inside this process")
    ap.add_argument("--drive", choices=tuple(DRIVES), default=DEFAULT_DRIVE)
    ap.add_argument("--num-periods", type=float, default=DEFAULT_NUM_PERIODS)
    ap.add_argument("--out", required=True, help="directory for the stage JSONs")
    ap.add_argument("--fixture-out", default=str(REPO / ARTIFACT))
    ap.add_argument("--run-id", default=None, help="the compute run id, recorded as-is")
    ap.add_argument("--run-index", default=None,
                    help="assemble only: a JSON mapping each stage file to its "
                         "compute run id and run directory name")
    ap.add_argument("--estimate-only", action="store_true",
                    help="print the node/step estimate and exit")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.estimate_only:
        for kind in KINDS:
            for um in RUNGS_UM:
                for dut in DUTS:
                    cost_estimate(kind, dut, um * 1e-6, args.num_periods)
        return 0

    if args.assemble:
        stage_assemble(args, out, Path(args.fixture_out))
        return 0

    if args.stage is None:
        ap.error("one of --stage or --assemble is required")

    um = args.rung
    if args.stage == "pilot":
        stage_pilot(args, out / f"pilot_{args.kind}_{um}um.json")
    elif args.stage == "solve":
        stage_solve(args, out)
    elif args.stage == "identity":
        stage_identity(args, out / f"identity_{args.kind}_{um}um.json")
    elif args.stage == "adfd-r":
        stage_adfd_r(args, out / f"adfd-r_{args.kind}_{um}um.json")
    elif args.stage == "adfd-eps":
        stage_adfd_eps(args, out / f"adfd-eps_{args.kind}_{um}um.json", "short")
    elif args.stage == "adfd-eps-res":
        stage_adfd_eps(args, out / f"adfd-eps-res_{args.kind}_{um}um.json",
                       "res_double")
    elif args.stage == "notinchain":
        stage_notinchain(args, out / f"notinchain_{args.kind}_{um}um.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
