#!/usr/bin/env python3
"""Issue #498 / #517 step 2 — the refplane-instrumented mixed-lane measurement.

**REPORT-ONLY.** This script measures; it does not decide, does not gate, and
does not pin. Its predeclaration is
``docs/design_notes/mixed_refplane_predeclaration.md`` (committed
BEFORE any run, on ``meas/498-517-mixed-referee``). Read that first: every
falsifier, budget and outcome below is quoted from it, not invented here.

======================================================================
WHAT IT DOES
======================================================================
One two-drive FDTD run on the committed #488 mixed lumped/wire<->MSL fixture
(``tests/unit/sparams/test_mixed_port_sparam.py::_base_sim/_add_feed/_add_msl``, verbatim)
with exactly TWO declared deviations:

  1. ``add_port(..., direction="-x", reference_plane_cells=10)`` on the lw
     feed. ``direction`` is the OUTWARD normal and ``outboard_sign = -1`` for
     ``"+"`` (``rfx/probes/refplane.py``), so ``"-x"`` points the planes INTO
     the DUT, toward the MSL port: slot 0 at x = 2.80 mm (index 43), slot 1 at
     x = 3.60 mm (index 53).
  2. ``n_probes=3`` on the MSL port -> ladder 4.72 / 4.40 / 4.08 mm.

and dumps, per frequency bin and per drive: the complex ``S_raw`` AND ``S``,
the reference-plane ``|out/inc|`` on both drives, ``|b_msl|`` with the
factor-2 convention written out, every flux surface individually and signed,
the reciprocity residual, ``cond(A)``, ``settling_db``, and the preflight
output verbatim. It then EVALUATES F1 / F2 / F3 exactly as predeclared and
prints which side each falsifier came out against — with
``NON_DISCRIMINATING`` and ``UNRESOLVED`` as legal, expected outcomes.

======================================================================
HOW THE PLANES ARE REACHED (and the reviewer blocker that shaped it)
======================================================================
``compute_mixed_s_matrix()`` v1 raises ``NotImplementedError`` on
``add_port(reference_plane_cells=...)``, and that rejection is coupled to
``docs/guides/sparameter_support_matrix.{md,json}`` by the parity gate. This
script therefore does NOT lift the guard: it monkeypatches
``Simulation._forward_from_materials`` for the duration of one call, swapping
``self._ports`` for ``dataclasses.replace(pe, reference_plane_cells=N)``
copies (each entry's ``excite`` flag preserved) and restoring them
immediately. Zero shipped-code change; the guard, its message and the support
matrix stay untouched.

**Reviewer blocker B1, applied.** The predeclaration's mechanism as first
written could not work: ``rfx/api/_execute.py`` registers the reference-plane
specs only under ``_sparam_drive_idx is not None``, and the mixed lane's drive
loop never passes it (it selects the driven port with per-run ``excite``
flags). Measured on this exact fixture, with ``reference_plane_cells=10``
already swapped onto ``sim._ports``::

    drive_idx=None : wire_refplane -> None
    drive_idx=0    : wire_refplane -> tuple len=2
    drive_idx=999  : wire_refplane -> tuple len=2

and the naive repair is wrong too: ``_excite_this_port`` IGNORES ``pe.excite``
once ``_sparam_drive_idx`` is set, so passing ``0`` on the MSL-driven run
would excite the lw port and destroy run 1 — the very drive F2's ``M2`` is
measured on. The shape used here (and measured, see
``_POSITIVE_CONTROL_NOTE``) is: pass the driven lw port's own sparam index on
an lw-driven run, and an OUT-OF-RANGE sentinel
(``_NO_LW_DRIVE_SENTINEL``) on a run where no lw port is driven — no lw
source is built, the planes still register. The wrapper asserts BOTH on every
run: ``raw["wire_refplane"]`` is not None, and the lw accumulator's own
``excite`` flag matches the lane's per-run intent. That positive control is
cheap and runs at ``num_periods=4`` as well as at 60.

======================================================================
CONVENTIONS (fixed BEFORE the run, reviewer correction 3)
======================================================================
``refplane_split`` returns ``w_plus = 0.5*(v + zc*i_corr)`` — the half is
ALREADY inside — and ``_b_msl`` is ``(V0 - Z_hj*I)/(2*sqrt(Z_hj))`` — the two
is ALREADY inside. Writing ``V = V+ + V-``, ``I = (V+ - V-)/Z``:
``out == V+`` exactly and ``b_msl == V-/sqrt(Z)``. Both are full-amplitude
power waves whose ``|.|^2`` is the wave power in the ``Re(V conj(I))``
convention with NO 1/2 — the same convention ``flux_spectrum`` returns
(``integral Re(E x H*).n dA``). Hence::

    R3 = |b_msl| / ( |out| / sqrt(Re Zc) )        <=>   |b_msl|^2  vs  |out|^2 / Re(Zc)

NOT ``|out|^2/(4 Re Zc)``, which under-reads by exactly 4 in power. Pinned by
``tests/unit/sparams/test_mixed_refplane_measurement.py`` (pure NumPy, written against the
wrong form first).

======================================================================
WHAT THIS SCRIPT REFUSES TO DO
======================================================================
* It refuses to write any path that reads as a record, gate, reference,
  snapshot, baseline, tolerance or committed fixture (``_assert_report_only``).
  The only thing it writes is one diagnostic JSON under its own directory.
* It refuses to pin the lumped/wire ("lw") diagonal. ``|S00|`` is RECORDED
  (raw, per bin, alongside everything else) and is never compared to a
  threshold, never asserted, never used to decide a falsifier. F4 — the lw
  diagonal — is NOT a falsifier on this run and is deliberately absent from
  ``FALSIFIERS``; it is carried as a prediction only, in
  ``_F4_PREDICTION_NOT_A_FALSIFIER`` (reviewer non-blocking item: relabel).
  The PI's sequencing decision on #776/#778 + the parked #683 flip is
  undecided; see §10 of the predeclaration.
* It never compares ``result.S``. Only ``S_raw`` is comparator-eligible
  (``enforce_passivity=True`` makes the shipped matrix a joint SVD
  projection).

======================================================================
REVIEWER NON-BLOCKING ITEMS APPLIED
======================================================================
* Every flux surface is recorded BOTH ways: the lane-default ``flux_spectrum``
  path (comparable with the lane's own ``box_lw`` / ``plane_msl``) and
  ``exact_f64=True``. **R1 quotes the exact_f64 numbers**; the default-path
  numbers are recorded beside them and their relative gap is printed, because
  under complex64 a PARTIAL subnormal flush corrupts small-magnitude sums well
  inside R1's [0.95, 1.03] window (``rfx/probes/probes.py``'s own note).
* F1's two numerator terms are not the same KIND of quantity: ``P_line(+x)``
  is a line-mode power at a reference plane while ``P_line(-x)`` is a FULL
  cross-section flux that also catches substrate/air radiation. A matching
  ``+x`` full-cross-section plane at x = 2.56 mm is therefore also registered
  and R1 is reported in two forms — ``R1`` (the predeclared, mixed-kind form,
  which is the comparator) and ``R1_symmetric`` (both branches full
  cross-section, REPORTED ONLY, to size the asymmetry).
* ``settling_db`` under ``n_probes=3`` is a max over 3 witness planes, not the
  committed run's 5. A max over a subset can only be LOWER, i.e. flattering,
  so it is NOT bit-comparable with the committed -122.57 / -119.93 dB. Printed
  next to that quote.
* F2 measures the reflection of the WHOLE -x-side load — feed transition PLUS
  2.0 mm of line PLUS the declared open end at x = 0 — referred to two planes,
  not "the feed transition's reflection". The magnitude-equality argument only
  needs the uniform lossless line between 2.80 mm and 4.72 mm, so F2's logic
  is unchanged; the label is corrected so the open end is not later treated as
  a surprise.
* F1's outcome space is CLOSED (reviewer blocker B2): the fourth quadrant
  (Im/Re within class, R1 out of window, miss NOT attributable to the y-faces)
  is predeclared as ``SIDE_D_UNATTRIBUTED_MISS`` -> report and STOP, and the
  y-face attribution carries a numeric rule (``_F1_YFACE_SHARE`` of the miss
  in the +/-y pair AND the x/z-only box reconciling inside the window).

Usage
-----
    cd <worktree> && PYTHONPATH=<worktree> JAX_PLATFORMS=cpu \\
        python3 scripts/diagnostics/mixed_refplane_measurement.py \\
            [--num-periods 60] [--out DIR] [--no-write]

``--num-periods 4`` is the plumbing smoke, NOT the measurement: at that record
length the ring-down witness reads ~ -1 dB, far above the -40 dB settling
rule, so every falsifier MUST come back ``UNRESOLVED``. The measurement is
``--num-periods 60``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses as _dc
import io
import json
import os
import re
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

from rfx import Box, Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.probes.probes import flux_spectrum  # noqa: E402
from rfx.probes.refplane import (  # noqa: E402
    refplane_beta,
    refplane_centered_current,
    refplane_split,
    refplane_zc_two_plane,
)
from rfx.sources.sources import GaussianPulse  # noqa: E402

OUT_DIR = REPO / "scripts" / "diagnostics" / "mixed_refplane_measurement"
ARTIFACT_NAME = "mixed_refplane_measurement.json"

# ---------------------------------------------------------------------------
# Fixture — verbatim from tests/unit/sparams/test_mixed_port_sparam.py (_base_sim,
# _add_feed, _add_msl), plus the two declared deviations.
# ---------------------------------------------------------------------------
_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
_DX = 80e-6
_DOMAIN = (8e-3, 3e-3, 754e-6)
_CPML_LAYERS = 8
_FREQ_MAX = 5e9
_X_FEED = 2.0e-3
_X_MSL = 5.5e-3

# --- the two declared deviations -------------------------------------------
_FEED_DIRECTION = "-x"        # deviation 1: planes point INTO the DUT
_REFPLANE_N = 10              # deviation 1: N (slot 0) and 2N (slot 1)
_N_PROBES = 3                 # deviation 2: ladder 4.72 / 4.40 / 4.08 mm

# --- predeclared extra flux surfaces ---------------------------------------
_FLUX_MX_NAME = "_mixed_refplane_flux_lw_mx"
_FLUX_MX_X = 1.44e-3          # predeclared -x full cross-section (index 26)
_FLUX_PX_NAME = "_mixed_refplane_flux_lw_px"
_FLUX_PX_X = 2.56e-3          # symmetric +x full cross-section, REPORTED ONLY

# --- constants quoted from the predeclaration -------------------------------
_SETTLING_WITNESS_DB = -40.0  # mirrors rfx/api/_sparams.py::_SETTLING_WITNESS_DB
_Z0_HJ_COMMITTED = 47.89479996289313
_F1_ZC_IM_RE_MAX = 0.03       # refplane.py::_ZC_IM_RE_WARN_RATIO
_F1_R1_WINDOW = (0.95, 1.03)  # measured inter-surface offsets 0.976 / 1.026
_F1_YFACE_SHARE = 0.70        # B2: share of the miss the y-pair must carry
_F2_B_FLOOR = 0.01
_F2_B_MAX = 0.15              # above this the anchor cannot discriminate
_F2_SMALL = 0.05              # "both small" ceiling
_F2_ALT_TOL = 0.05
_F2_ALT_S22 = (0.4324, 0.4342, 0.4387, 0.4437, 0.4478)   # committed-JSON row
_F3_WINDOW = (0.95, 1.03)
_F3_ANCHOR_TOL = 0.02

_NO_LW_DRIVE_SENTINEL = 10_000_000   # out of range for any real port count

_POSITIVE_CONTROL_NOTE = (
    "B1 positive control: every drive asserts raw['wire_refplane'] is not "
    "None AND that the lw accumulator's excite flag equals the lane's own "
    "per-run intent (True only on the lw-driven run), so the instrumentation "
    "cannot silently change what is excited."
)

_F4_PREDICTION_NOT_A_FALSIFIER = {
    "id": "F4",
    "status": "PREDICTION ONLY — NOT A FALSIFIER ON THIS RUN",
    "why": (
        "The lw diagonal is the port's reflection toward its OWN source; it "
        "is not on the line and no quantity on main measures it (per-cell "
        "pre-injection frame). Nothing in this run tests it."
    ),
    "band_quoted_not_pinned": "0.21 +/- 0.03",
    "arithmetic_open_question": (
        "The plan's 0.211-0.219 and 'flux witness <= 3.5% at |S00|=0.21' do "
        "not reproduce from the committed JSON: at |S00|=0.21 the flux "
        "reciprocity deviation computes to 4.61% (9.76% at 0.38), and an "
        "independent bracket 1 - r0/r1 gives 0.2136-0.2203. Recorded, not "
        "resolved."
    ),
    "may_be_pinned": False,
}

DO_NOT_PIN = (
    "any lumped/wire (lw) diagonal value — shipped 0.38-0.40, predicted "
    "0.21 +/- 0.03, or sqrt(n_live)-rescaled variants",
    "reciprocity_tol = 0.06",
    "the docs' 9.0% / 55% quotes and the support-matrix known_limits text",
    "the mixed lane's reference_plane_cells rejection itself",
    "|S22| = 0.02-0.03, whatever F2 returns",
    "Zc_meas / beta measured on this run",
    "cond(A), settling_db, wall clock, the inter-surface offsets",
    "num_periods = 60, the 5-bin frequency set, n_probes = 3",
)

# Path fragments that make an output look like a committed record or gate.
_FORBIDDEN_PATH_TOKENS = (
    "reference", "record", "gate", "golden", "snapshot", "baseline",
    "tolerance", "expected", "support_matrix", "known_limits", "fixture",
)
_FORBIDDEN_PATH_PARENTS = ("tests", "docs", "validation", "rfx")


# ===========================================================================
# Report-only guards
# ===========================================================================

def _assert_report_only(path: Path) -> Path:
    """Refuse to write anything that reads as a record / gate / reference.

    Report-only is a property of the OUTPUT, not of the intent, so it is
    enforced here rather than asserted in prose. Raises ``RuntimeError`` on
    any path whose name or parents mark it as committed evidence something
    else consumes.
    """
    p = Path(path).resolve()
    rel = p.relative_to(REPO) if str(p).startswith(str(REPO)) else p
    parts_lower = [q.lower() for q in rel.parts]
    for parent in _FORBIDDEN_PATH_PARENTS:
        if parts_lower and parts_lower[0] == parent:
            raise RuntimeError(
                f"report-only: refusing to write under {parent}/ — this "
                f"script produces diagnostics, not committed evidence "
                f"({rel})."
            )
    stem = p.name.lower()
    for token in _FORBIDDEN_PATH_TOKENS:
        if token in stem:
            raise RuntimeError(
                f"report-only: refusing to write {rel} — the name contains "
                f"{token!r}, which reads as a record/gate/reference. This "
                "measurement pins nothing."
            )
    return p


def _assert_no_lw_diagonal_pin(payload: dict) -> None:
    """Fail loudly if the artifact ever grows an lw-diagonal comparison.

    The lw diagonal may be RECORDED (it is, per bin, raw) but never compared,
    asserted, or bounded — the PI's sequencing decision on #776/#778 + the
    parked #683 flip is undecided. Any key that pairs the lw diagonal with a
    verdict / expectation / tolerance is a pin, and this refuses it.
    """
    bad = []

    def _walk(node, trail):
        if isinstance(node, dict):
            for k, v in node.items():
                key = str(k).lower()
                looks_lw = ("s00" in key or "lw_diag" in key
                            or "s_lw_lw" in key)
                looks_pin = any(t in key for t in (
                    "expected", "verdict", "pass", "fail", "tol",
                    "gate", "reference", "assert", "budget", "window"))
                if looks_lw and looks_pin:
                    bad.append("/".join(trail + [str(k)]))
                _walk(v, trail + [str(k)])
        elif isinstance(node, list):
            for n, v in enumerate(node):
                _walk(v, trail + [str(n)])

    _walk(payload, [])
    if bad:
        raise RuntimeError(
            "report-only: the artifact pairs the lw diagonal with a "
            f"verdict/expectation at {bad} — that is a pin, and §10 of the "
            "predeclaration forbids it until the PI sequencing decision."
        )


# ===========================================================================
# Fixture builders
# ===========================================================================

def build_fixture(*, feed_direction: str = _FEED_DIRECTION,
                  n_probes: int = _N_PROBES,
                  register_extra_flux: bool = True) -> Simulation:
    """The committed #488 mixed fixture with the two declared deviations.

    ``reference_plane_cells`` is deliberately NOT set here — the mixed lane
    rejects it at entry. The planes are reached by
    :func:`refplane_instrumentation`, which injects it inside
    ``_forward_from_materials`` only.
    """
    lx, ly, lz = _DOMAIN
    sim = Simulation(
        freq_max=_FREQ_MAX, domain=(lx, ly, lz), dx=_DX,
        cpml_layers=_CPML_LAYERS,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB + _DX)), material="pec")
    # Deviation 1 lives in `direction`; reference_plane_cells is injected
    # by the instrumentation, never by the fixture.
    sim.add_port(position=(_X_FEED, y_c, 0.0), component="ez",
                 impedance=50.0, extent=_H_SUB, direction=feed_direction)
    sim.add_msl_port(position=(_X_MSL, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction="-x", impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5),
                     n_probe_offset=10, n_probe_spacing=4,
                     n_probes=n_probes)
    if register_extra_flux:
        # Pre-registered, full cross-section (no size/center). The mixed lane
        # saves and restores self._flux_monitors and looks up only its OWN
        # names, so these ride along name-keyed in raw["flux_monitors"].
        freqs = np.linspace(1e9, 4e9, 5)
        for nm, xc in ((_FLUX_MX_NAME, _FLUX_MX_X),
                       (_FLUX_PX_NAME, _FLUX_PX_X)):
            sim.add_flux_monitor(axis="x", coordinate=float(xc),
                                 freqs=jnp.asarray(freqs), name=nm)
    return sim


# ===========================================================================
# Instrumentation (monkeypatch only; zero shipped-code change)
# ===========================================================================

class RefplaneCapture:
    """Per-drive raw payloads captured from ``_forward_from_materials``."""

    def __init__(self) -> None:
        self.runs: list[dict] = []
        self.grid = None
        self.positive_control: list[dict] = []


@contextlib.contextmanager
def refplane_instrumentation(n_cells: int = _REFPLANE_N):
    """Swap ``reference_plane_cells`` on and pass a legal ``_sparam_drive_idx``.

    See the module docstring (blocker B1) for why both halves are required and
    why the sentinel — not ``0`` — is used on a run with no lw drive.
    """
    cap = RefplaneCapture()
    orig = Simulation._forward_from_materials

    def patched(self, grid, materials, debye_spec, lorentz_spec, **kw):
        lw_slots = [k for k, pe in enumerate(self._ports)
                    if pe.impedance != 0.0]
        driven = [k for k in lw_slots if self._ports[k].excite]
        if len(driven) > 1:
            raise RuntimeError(
                "refplane instrumentation: more than one lw port is excited on "
                "this run; the single-drive assumption behind "
                "_sparam_drive_idx does not hold.")
        drive_idx = (lw_slots.index(driven[0]) if driven
                     else _NO_LW_DRIVE_SENTINEL)
        saved = list(self._ports)
        try:
            self._ports = [
                _dc.replace(pe, reference_plane_cells=int(n_cells))
                if pe.impedance != 0.0 else pe
                for pe in saved
            ]
            kw["_sparam_drive_idx"] = drive_idx
            raw = orig(self, grid, materials, debye_spec, lorentz_spec, **kw)
        finally:
            self._ports = saved
        if not isinstance(raw, dict):
            return raw
        # --- B1 positive control, on EVERY drive -------------------------
        rp = raw.get("wire_refplane")
        if rp is None or len(rp) == 0:
            raise AssertionError(
                "refplane instrumentation: raw['wire_refplane'] is empty — the "
                "reference planes did not register. Do not trust anything "
                "downstream.")
        wire = raw.get("wire") or ()
        excite_flags = [bool(s.excite) for s, _ in wire]
        expect = [bool(pe.excite) for pe in saved if pe.impedance != 0.0]
        if excite_flags != expect:
            raise AssertionError(
                "refplane instrumentation: the drive index changed WHAT IS "
                f"EXCITED (accumulator excite={excite_flags}, lane intent="
                f"{expect}). Refusing to continue.")
        cap.positive_control.append({
            "drive_idx": int(drive_idx),
            "n_refplane_specs": int(len(rp)),
            "lw_excite_flags": excite_flags,
            "lane_intent": expect,
        })
        cap.grid = grid
        cap.runs.append(raw)
        return raw

    Simulation._forward_from_materials = patched
    try:
        yield cap
    finally:
        Simulation._forward_from_materials = orig


# ===========================================================================
# Reference-plane extraction (pure NumPy on the raw accumulators)
# ===========================================================================

def extract_refplane(raw: dict, freqs: np.ndarray, dt: float, dx: float,
                     port_index: int = 0) -> dict:
    """Cook one drive's ``raw['wire_refplane']`` into the recorded channel.

    Returns raw accumulators uncooked as well as ``i_corr``, ``Zc``, ``beta``
    and the ``(out, inc)`` pair at BOTH slots.
    """
    rp = raw["wire_refplane"]
    by_slot = {int(s.plane_slot): (s, a) for s, a in rp
               if int(s.port_index) == port_index}
    if set(by_slot) != {0, 1}:
        raise AssertionError(
            f"expected slots {{0, 1}} for port {port_index}, got "
            f"{sorted(by_slot)}")
    out: dict = {"slots": {}}
    sign = int(by_slot[0][0].outboard_sign)
    v, im, ip = {}, {}, {}
    for slot, (spec, accs) in by_slot.items():
        v[slot] = np.asarray(accs[0], dtype=np.complex128)
        im[slot] = np.asarray(accs[1], dtype=np.complex128)
        ip[slot] = np.asarray(accs[2], dtype=np.complex128)
        out["slots"][slot] = {
            "plane_index": int(spec.plane_index),
            "n_cells_outboard": int(spec.n_cells_outboard),
            "outboard_sign": int(spec.outboard_sign),
            "line_axis": int(spec.line_axis),
            "plane_v": v[slot],
            "plane_im": im[slot],
            "plane_ip": ip[slot],
        }
    i_corr = {s: refplane_centered_current(im[s], ip[s], freqs, dt)
              for s in (0, 1)}
    zc = refplane_zc_two_plane(v[0], i_corr[0], v[1], i_corr[1])
    for s in (0, 1):
        o, i_ = refplane_split(v[s], i_corr[s], zc, sign)
        out["slots"][s]["i_corr"] = i_corr[s]
        out["slots"][s]["out"] = o
        out["slots"][s]["inc"] = i_
    sep = float(by_slot[1][0].n_cells_outboard
                - by_slot[0][0].n_cells_outboard) * dx
    out["zc"] = zc
    out["beta"] = refplane_beta(out["slots"][0]["out"],
                                out["slots"][1]["out"], sep)
    out["separation_m"] = sep
    out["outboard_sign"] = sign
    # (|out|^2 - |inc|^2)/Re(Zc) — the zero-free-parameter net line power at
    # the plane, in the Re(V I*) (no 1/2) convention flux_spectrum returns.
    o0, i0 = out["slots"][0]["out"], out["slots"][0]["inc"]
    out["p_line_slot0"] = (np.abs(o0) ** 2 - np.abs(i0) ** 2) / zc.real
    out["out_over_inc"] = {
        s: np.abs(out["slots"][s]["out"]) / np.maximum(
            np.abs(out["slots"][s]["inc"]), np.finfo(float).tiny)
        for s in (0, 1)
    }
    return out


def b_msl_probe0(v0, i0, z0_hj):
    """``_b_msl`` reproduced verbatim: ``(V0 - Z_hj I)/(2 sqrt(Z_hj))``.

    The factor 2 is written out here on purpose — it is exactly what makes
    ``b_msl`` a FULL-amplitude power wave (``b = V-/sqrt(Z)``), which is why
    F3 divides ``|out|`` by ``sqrt(Re Zc)`` and NOT by ``2 sqrt(Re Zc)``.
    """
    return (np.asarray(v0, dtype=np.complex128)
            - float(z0_hj) * np.asarray(i0, dtype=np.complex128)) / (
                2.0 * np.sqrt(float(z0_hj)))


def a_msl_probe0(v0, i0, z0_hj):
    """The matching incident wave ``(V0 + Z_hj I)/(2 sqrt(Z_hj))``."""
    return (np.asarray(v0, dtype=np.complex128)
            + float(z0_hj) * np.asarray(i0, dtype=np.complex128)) / (
                2.0 * np.sqrt(float(z0_hj)))


# ===========================================================================
# Falsifier evaluators — pure, synthetic-testable, predeclared verdicts
# ===========================================================================

def _settling_ok(settling_db) -> bool:
    s = np.asarray(settling_db, dtype=float)
    return bool(s.size and np.all(np.isfinite(s))
                and np.all(s <= _SETTLING_WITNESS_DB))


def _unresolved(fid: str, reason: str, numbers: dict) -> dict:
    return {"falsifier": fid, "verdict": "UNRESOLVED", "resolved": False,
            "side": None, "reason": reason, "numbers": numbers}


def evaluate_f1(*, zc_im_re, r1, r1_xz, y_face_contrib, box_net,
                settling_db) -> dict:
    """F1 — plane fidelity + bidirectional flux accounting (lw drive).

    Outcome space is CLOSED (reviewer blocker B2):

    ``UNRESOLVED``              settling above -40 dB on either drive, or the
                                box net is not a usable discriminator (a
                                non-finite or vanishing denominator).
    ``SIDE_B_PLANE_BAD``        max |Im Zc/Re Zc| > 0.03 -> the plane is near
                                field contaminated; report and STOP.
    ``SIDE_A_INSTRUMENT_GOOD``  ratio in window at all bins AND Im/Re in class.
    ``SIDE_C_BOX_UNDERCAPTURE`` Im/Re in class, R1 out of window, and the miss
                                is attributable to the box's +/-0.24 mm y half
                                width (narrower than the trace's 0.30 mm): the
                                +/-y pair carries >= _F1_YFACE_SHARE of the
                                miss at every bin AND the x/z-only box
                                reconciles inside the window. Report, CONTINUE
                                (F2 and F3 never use the box).
    ``SIDE_D_UNATTRIBUTED_MISS`` the fourth quadrant: Im/Re in class, R1 out of
                                window, y attribution INVALID. Report and STOP,
                                miss recorded as unattributed.
    """
    zc_im_re = np.asarray(zc_im_re, dtype=float)
    r1 = np.asarray(r1, dtype=float)
    r1_xz = np.asarray(r1_xz, dtype=float)
    y_face_contrib = np.asarray(y_face_contrib, dtype=float)
    box_net = np.asarray(box_net, dtype=float)
    lo, hi = _F1_R1_WINDOW
    numbers = {
        "zc_im_re_max": float(np.max(np.abs(zc_im_re))) if zc_im_re.size else float("nan"),
        "r1": r1.tolist(),
        "r1_symmetric_or_xz": r1_xz.tolist(),
        "window": [lo, hi],
        "zc_im_re_class_boundary": _F1_ZC_IM_RE_MAX,
    }
    if not _settling_ok(settling_db):
        return _unresolved("F1", (
            "settling precondition failed: the ring-down witness is at or "
            f"above {_SETTLING_WITNESS_DB} dB on at least one drive, so every "
            "DFT-derived number of that drive is a truncation artifact"),
            numbers)
    if (not np.all(np.isfinite(r1))) or np.any(box_net == 0.0):
        return _unresolved("F1", (
            "discriminator below budget: the box net power is zero or "
            "non-finite at some bin, so R1 is not a ratio of measured "
            "powers"), numbers)
    if float(np.max(np.abs(zc_im_re))) > _F1_ZC_IM_RE_MAX:
        return {"falsifier": "F1", "verdict": "SIDE_B_PLANE_BAD",
                "resolved": True, "side": "plane (the instrument)",
                "reason": (
                    f"max |Im Zc/Re Zc| = {np.max(np.abs(zc_im_re)):.4f} > "
                    f"{_F1_ZC_IM_RE_MAX} — the N=10 plane pair is near-field "
                    "contaminated on this fixture. STOP: no S claim, no F2, "
                    "no F3."),
                "numbers": numbers}
    if bool(np.all((r1 >= lo) & (r1 <= hi))):
        return {"falsifier": "F1", "verdict": "SIDE_A_INSTRUMENT_GOOD",
                "resolved": True, "side": None,
                "reason": (
                    "R1 in window at all bins and |Im Zc/Re Zc| within the "
                    "measured class boundary — the plane pair is on the "
                    "uniform line and both launch branches are accounted "
                    "for. Proceed to F2 and F3."),
                "numbers": numbers}
    miss = box_net - (r1 * box_net)          # box_net - numerator
    denom = np.maximum(np.abs(miss), np.finfo(float).tiny)
    y_share = np.abs(y_face_contrib) / denom
    numbers["miss"] = miss.tolist()
    numbers["y_face_contrib"] = y_face_contrib.tolist()
    numbers["y_share"] = y_share.tolist()
    numbers["y_share_required"] = _F1_YFACE_SHARE
    y_ok = bool(np.all(y_share >= _F1_YFACE_SHARE))
    xz_ok = bool(np.all((r1_xz >= lo) & (r1_xz <= hi)))
    if y_ok and xz_ok:
        return {"falsifier": "F1", "verdict": "SIDE_C_BOX_UNDERCAPTURE",
                "resolved": True, "side": "box (declared geometry limit)",
                "reason": (
                    "R1 outside the window, but the +/-y face pair carries "
                    f">= {_F1_YFACE_SHARE:.0%} of the miss at every bin and "
                    "the x/z-only box reconciles inside the window — the "
                    "box's +/-0.24 mm y half-width is narrower than the "
                    "trace's 0.30 mm half-width. Declared geometry "
                    "limitation, NOT a plane defect. CONTINUE: F2 and F3 "
                    "never use the box."),
                "numbers": numbers}
    return {"falsifier": "F1", "verdict": "SIDE_D_UNATTRIBUTED_MISS",
            "resolved": True, "side": "unattributed",
            "reason": (
                "R1 outside the window with |Im Zc/Re Zc| inside its class "
                "boundary, and the miss is NOT attributable to the box's y "
                f"faces (y_share_min = {float(np.min(y_share)):.3f} vs "
                f"{_F1_YFACE_SHARE}, x/z-only box in window = {xz_ok}). "
                "Predeclared fourth quadrant: report and STOP, miss recorded "
                "as unattributed."),
            "numbers": numbers}


def f2_budget(zc_meas_re_band_mean: float, z0_hj: float = _Z0_HJ_COMMITTED) -> float:
    """``B = |(Zc - Z0_hj)/(Zc + Z0_hj)| + 0.01`` — computed from the run."""
    zc = float(zc_meas_re_band_mean)
    return float(abs((zc - z0_hj) / (zc + z0_hj))) + _F2_B_FLOOR


def evaluate_f2(*, m2, s22, zc_meas_re_band_mean, settling_db,
                z0_hj: float = _Z0_HJ_COMMITTED,
                s22_alt=_F2_ALT_S22) -> dict:
    """F2 — MSL-diagonal same-run referee (two-sided).

    ``M2 = |out/inc|`` at refplane slot 0 on the MSL drive (measured Zc)
    versus the shipped ``|S22|_raw = |b/a|`` at MSL probe 0 (analytic HJ Z0).
    Both are the reflection of the WHOLE -x-side load — feed transition PLUS
    2.0 mm of line PLUS the declared open end at x = 0 — referred to two
    planes; on the uniform lossless line between 2.80 mm and 4.72 mm their
    MAGNITUDES must agree.

    Verdicts: ``UNRESOLVED`` (settling), ``NON_DISCRIMINATING_ANCHOR``
    (B > 0.15 — the discriminator is below its own budget),
    ``CONSISTENT_NON_DISCRIMINATING`` (explicitly NOT "vindicated"),
    ``MSL_DIAGONAL_CONVICTED``, ``REPORTED_NO_ATTRIBUTION``.
    """
    m2 = np.asarray(m2, dtype=float)
    s22 = np.asarray(s22, dtype=float)
    alt = np.asarray(s22_alt, dtype=float)
    b = f2_budget(zc_meas_re_band_mean, z0_hj)
    numbers = {"M2": m2.tolist(), "abs_S22_raw": s22.tolist(),
               "B": b, "B_max": _F2_B_MAX,
               "zc_meas_re_band_mean": float(zc_meas_re_band_mean),
               "z0_hj": float(z0_hj),
               "S22_alt_predeclared": alt.tolist()}
    if not _settling_ok(settling_db):
        return _unresolved("F2", (
            "settling precondition failed: the ring-down witness is at or "
            f"above {_SETTLING_WITNESS_DB} dB on at least one drive"),
            numbers)
    if b > _F2_B_MAX:
        return {"falsifier": "F2", "verdict": "NON_DISCRIMINATING_ANCHOR",
                "resolved": False, "side": None,
                "reason": (
                    f"B = {b:.4f} > {_F2_B_MAX}: the analytic HJ anchor is "
                    "too far from the measured line impedance to "
                    "discriminate at all. No attribution is claimed."),
                "numbers": numbers}
    d = np.abs(m2 - s22)
    if bool(np.all(d <= b)) and bool(np.all(m2 <= _F2_SMALL)) \
            and bool(np.all(s22 <= _F2_SMALL)):
        return {"falsifier": "F2",
                "verdict": "CONSISTENT_NON_DISCRIMINATING",
                "resolved": False, "side": None,
                "reason": (
                    f"|M2 - |S22|| <= B = {b:.4f} at every bin with both "
                    f"<= {_F2_SMALL}. Predeclared as CONSISTENT, "
                    "NON-DISCRIMINATING ON THE ANCHOR — explicitly NOT 'the "
                    "MSL diagonal is vindicated': a small-vs-small agreement "
                    "at this budget carries no information about a "
                    "0.02-0.03 quantity."),
                "numbers": numbers}
    if bool(np.all((m2 - s22) > b)) and bool(np.all(np.abs(m2 - alt) <= _F2_ALT_TOL)):
        return {"falsifier": "F2", "verdict": "MSL_DIAGONAL_CONVICTED",
                "resolved": True, "side": "MSL diagonal (probe-0 extractor / HJ anchor)",
                "reason": (
                    f"M2 - |S22| > B = {b:.4f} at every bin and M2 sits "
                    f"within {_F2_ALT_TOL} of the predeclared "
                    "reciprocity-implied row ~0.43. The flux gap sits on the "
                    "MSL side; the shipped lw 0.38-0.40 is not the "
                    "residual's source."),
                "numbers": numbers}
    return {"falsifier": "F2", "verdict": "REPORTED_NO_ATTRIBUTION",
            "resolved": False, "side": None,
            "reason": (
                "M2 is neither within B of the shipped diagonal at "
                "small-vs-small, nor near the predeclared ~0.43 alternative. "
                "Third declared possibility: report the number, attribute "
                "nothing — both channels suspect, only the external referee "
                "can discriminate."),
            "numbers": numbers}


def evaluate_f3(*, r3, zc_meas_re_band_mean, settling_db,
                denominator=None, signal_floor: float = 0.0,
                z0_hj: float = _Z0_HJ_COMMITTED) -> dict:
    """F3 — MSL receive-channel referee (thru between 2.80 mm and 4.72 mm).

    ``R3 = |b_msl(probe 0, lw run)| / (|out(slot 0, lw run)| / sqrt(Re Zc))``.

    Verdicts: ``UNRESOLVED`` (settling, or the plane-wave denominator below
    ``signal_floor`` — the discriminator below its budget),
    ``MSL_RECEIVE_AGREES``, ``MSL_ANCHOR_CONVICTED``,
    ``MSL_EXTRACTOR_CONVICTED``.
    """
    r3 = np.asarray(r3, dtype=float)
    lo, hi = _F3_WINDOW
    anchor = float(np.sqrt(float(zc_meas_re_band_mean) / float(z0_hj)))
    numbers = {"R3": r3.tolist(), "window": [lo, hi],
               "anchor_prediction_sqrt_Zc_over_Zhj": anchor,
               "anchor_tol": _F3_ANCHOR_TOL,
               "zc_meas_re_band_mean": float(zc_meas_re_band_mean)}
    if not _settling_ok(settling_db):
        return _unresolved("F3", (
            "settling precondition failed: the ring-down witness is at or "
            f"above {_SETTLING_WITNESS_DB} dB on at least one drive"),
            numbers)
    if denominator is not None:
        den = np.abs(np.asarray(denominator, dtype=float))
        numbers["denominator"] = den.tolist()
        numbers["signal_floor"] = float(signal_floor)
        if not np.all(np.isfinite(r3)) or np.any(den <= signal_floor):
            return _unresolved("F3", (
                "discriminator below budget: the plane-wave denominator "
                f"|out|/sqrt(Re Zc) falls to or below the signal floor "
                f"{signal_floor:.3e} at some bin, so R3 is a ratio of noise"),
                numbers)
    if bool(np.all((r3 >= lo) & (r3 <= hi))):
        return {"falsifier": "F3", "verdict": "MSL_RECEIVE_AGREES",
                "resolved": True, "side": None,
                "reason": (
                    "R3 in window at all bins — the shipped MSL receive "
                    "extractor agrees with the #313-validated plane wave. "
                    "The lane's |S10| magnitude gap is NOT in the MSL "
                    "receive channel. This pins no number; it says where the "
                    "residual is not."),
                "numbers": numbers}
    if float(np.max(np.abs(r3 - anchor))) <= _F3_ANCHOR_TOL:
        return {"falsifier": "F3", "verdict": "MSL_ANCHOR_CONVICTED",
                "resolved": True, "side": "MSL anchor (analytic HJ Z0)",
                "reason": (
                    f"R3 outside the window but equal to sqrt(Re Zc/Z_hj) = "
                    f"{anchor:.4f} within {_F3_ANCHOR_TOL} at every bin — the "
                    "extractor is fine and the HJ anchor is the cause."),
                "numbers": numbers}
    return {"falsifier": "F3", "verdict": "MSL_EXTRACTOR_CONVICTED",
            "resolved": True, "side": "MSL receive extractor (probe 0)",
            "reason": (
                "R3 outside the window and NOT equal to sqrt(Re Zc/Z_hj) "
                f"= {anchor:.4f} within {_F3_ANCHOR_TOL} — the MSL wave "
                "magnitude at probe 0 is off by exactly R3 (per-bin factor "
                "recorded). Comparator finding; pins no number."),
            "numbers": numbers}


FALSIFIERS = ("F1", "F2", "F3")   # F4 is NOT here — see the module docstring.


# ===========================================================================
# Flux bookkeeping
# ===========================================================================

def _flux_both_ways(mon) -> tuple[np.ndarray, np.ndarray]:
    """(lane-default, exact_f64) flux for one monitor.

    Recorded both ways because under complex64 a PARTIAL subnormal flush
    corrupts small-magnitude sums well inside R1's window, and the #304
    warning alone only fires on a TOTAL flush.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = np.asarray(flux_spectrum(mon), dtype=np.float64)
        exact = np.asarray(flux_spectrum(mon, exact_f64=True),
                           dtype=np.float64)
    return default, exact


_FACE_RE = re.compile(r"^_mixed_flux_lw(\d+)_([xyz])([+-])(.+)$")


def collect_box_faces(fmon: dict, port: int = 0) -> list[dict]:
    """Every lw box face, INDIVIDUALLY and SIGNED, both flux paths.

    The per-face record is what makes an F1 miss attributable; the lane itself
    only keeps the net.
    """
    faces = []
    for nm in sorted(fmon):
        m = _FACE_RE.match(nm)
        if not m or int(m.group(1)) != port:
            continue
        default, exact = _flux_both_ways(fmon[nm])
        sgn = 1.0 if m.group(3) == "+" else -1.0
        faces.append({
            "name": nm, "axis": m.group(2), "sign": sgn,
            "coordinate_m": float(m.group(4)),
            "flux_default": default, "flux_exact_f64": exact,
            "signed_default": sgn * default, "signed_exact_f64": sgn * exact,
        })
    if len(faces) != 5:
        raise AssertionError(
            f"expected the lane's 5-face box for lw port {port}, got "
            f"{len(faces)}: {[f['name'] for f in faces]}")
    return faces


# ===========================================================================
# cond(A) — REPORTED, NEVER GATED
# ===========================================================================

def cond_a_raw_and_column_normalized(*, v_lw, i_lw, v0_msl, i_msl, z0_lw,
                                     n_live_lw, z0_hj_msl, wire_mode,
                                     drive_plan) -> dict:
    """``cond(A)`` raw and column-normalized, comparable with the committed
    27.1 / 2.01 on this fixture.

    The wave-amplitude matrix ``A`` is not a shipped object — the shipped
    mixed lane is single-ratio — so this reuses the COMMITTED step-1 helpers
    (``scripts/diagnostics/i517_mixed_solve_vs_ratio_measurement.py``:
    ``_uniform_wave_amplitudes`` / ``_column_normalized_cond``) rather than
    re-deriving them, and the raw conditioning comes from the shipped
    ``msl_solve_s_from_waves``. cond(A) is on the do-not-gate list; if the
    step-1 helper cannot be imported this returns the reason instead of
    failing the measurement.
    """
    import importlib.util as _ilu
    step1 = REPO / "scripts" / "diagnostics" / \
        "i517_mixed_solve_vs_ratio_measurement.py"
    try:
        spec = _ilu.spec_from_file_location("_i517_step1", step1)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:                              # pragma: no cover
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    wave_a, wave_b = mod._uniform_wave_amplitudes(
        v_lw, i_lw, v0_msl, i_msl, np.asarray(z0_lw),
        np.asarray(n_live_lw), np.asarray(z0_hj_msl), wire_mode, drive_plan)
    from rfx.api._sparams import msl_solve_s_from_waves
    _S, cond_raw = msl_solve_s_from_waves(wave_a, wave_b)
    return {
        "available": True,
        "cond_a_raw": np.asarray(cond_raw, dtype=float).tolist(),
        "cond_a_column_normalized": np.asarray(
            mod._column_normalized_cond(wave_a), dtype=float).tolist(),
        "committed_reference_for_comparison_only": {
            "raw_max": 27.119385129670142,
            "column_normalized_max": 2.0088847691019205,
        },
        "note": "REPORTED, NEVER GATED (do-not-pin list entry 8).",
    }


# ===========================================================================
# Bookkeeping
# ===========================================================================

def run_bookkeeping() -> dict:
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:                                    # pragma: no cover
        sha = "unknown"
    return {
        "git_sha": sha,
        "jax_version": jax.__version__,
        "jax_enable_x64_env": os.environ.get("JAX_ENABLE_X64", "<unset>"),
        "jax_x64_enabled": bool(jax.config.x64_enabled),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS", "<unset>"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def _c2(a) -> list:
    a = np.asarray(a, dtype=np.complex128)
    return [[float(z.real), float(z.imag)] for z in np.ravel(a)]


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-periods", type=float, default=60.0)
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--no-write", action="store_true",
                    help="evaluate and print; write nothing")
    args = ap.parse_args(argv)

    num_periods = float(args.num_periods)
    is_measurement = num_periods >= 60.0
    freqs = np.linspace(1e9, 4e9, 5)

    print("=" * 78)
    print("ISSUE #498 / #517 — MIXED-LANE REFERENCE-PLANE MEASUREMENT "
          "(REPORT-ONLY)")
    print("=" * 78)
    print("Predeclaration: docs/design_notes/"
          "mixed_refplane_predeclaration.md")
    print(f"num_periods = {num_periods:g}"
          + ("" if is_measurement else
             "   <-- REDUCED SETTINGS: THIS IS A PLUMBING SMOKE, NOT THE "
             "MEASUREMENT."))
    if not is_measurement:
        print("    At this record length the ring-down witness sits far "
              "above the -40 dB")
        print("    settling rule, so EVERY falsifier verdict below MUST read "
              "UNRESOLVED.")
        print("    The measurement is num_periods = 60.")
    print(f"Nothing here may be pinned. Do-not-pin list: {len(DO_NOT_PIN)} "
          "entries (in the artifact).")
    print()

    sim = build_fixture()
    t0 = time.time()
    stdout_buf = io.StringIO()
    # Capture the flux-channel override's own ill_cond / neg_power masks the
    # way the step-1 script does: capture what production computes, never
    # re-derive it.
    import rfx.api._sparams as _sparams_mod
    _orig_override = _sparams_mod._mixed_flux_magnitude_override
    masks: dict = {}

    def _capturing_override(S_wave_, box_lw_, plane_msl_, drive_plan_,
                            msl_away_signs_, n_lw_, ill_cond_floor=0.05):
        out = _orig_override(S_wave_, box_lw_, plane_msl_, drive_plan_,
                             msl_away_signs_, n_lw_,
                             ill_cond_floor=ill_cond_floor)
        masks["ill_cond"] = np.asarray(out[1]).tolist()
        masks["neg_power"] = np.asarray(out[2]).tolist()
        masks["ill_cond_floor"] = float(ill_cond_floor)
        masks["msl_away_signs"] = [float(x) for x in msl_away_signs_]
        return out

    _sparams_mod._mixed_flux_magnitude_override = _capturing_override
    with refplane_instrumentation(_REFPLANE_N) as cap:
        with warnings.catch_warnings(record=True) as wr:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(stdout_buf):
                result, diag = sim.compute_mixed_s_matrix(
                    freqs=freqs, num_periods=num_periods,
                    skip_preflight=False, return_diagnostics=True,
                )
            caught = [str(w.message) for w in wr]
    _sparams_mod._mixed_flux_magnitude_override = _orig_override
    wall_clock_s = time.time() - t0
    preflight_text = stdout_buf.getvalue()

    print("=" * 78)
    print("PREFLIGHT (verbatim)")
    print("=" * 78)
    print(preflight_text)
    print("=" * 78)
    print(f"WARNINGS (verbatim, {len(caught)})")
    print("=" * 78)
    for w in caught:
        print(f"  - {w}")
    print()
    print("B1 POSITIVE CONTROL (per drive)")
    for pc in cap.positive_control:
        print(f"  {pc}")
    print(f"  {_POSITIVE_CONTROL_NOTE}")
    print()

    grid = cap.grid
    dt, dx = float(grid.dt), float(grid.dx)
    drive_plan = list(diag["drive_plan"])
    n_lw = int(np.asarray(diag["v_lw"]).shape[1])
    z0_hj = float(np.asarray(diag["z0_hj_msl"])[0])
    settling = np.asarray(result.settling_db, dtype=float)

    # --- comparator-eligible S ------------------------------------------
    S_raw = np.asarray(result.S_raw if result.S_raw is not None else result.S)
    S_shipped = np.asarray(result.S)
    S_wave = (None if result.S_wave is None
              else np.asarray(result.S_wave))
    passivity_correction = (None if result.passivity_correction is None
                            else np.asarray(result.passivity_correction))

    # --- reference-plane channel, both drives ----------------------------
    rp_runs = [extract_refplane(raw, freqs, dt, dx, port_index=0)
               for raw in cap.runs]
    lw_run = next(i for i, (fam, _l) in enumerate(drive_plan) if fam == "lw")
    msl_run = next(i for i, (fam, _l) in enumerate(drive_plan) if fam == "msl")
    rp_lw, rp_msl = rp_runs[lw_run], rp_runs[msl_run]
    zc = rp_lw["zc"]
    zc_im_re = np.abs(zc.imag / zc.real)
    zc_re_band_mean = float(np.mean(zc.real))

    # --- flux surfaces ----------------------------------------------------
    flux_records = []
    for run_idx, raw in enumerate(cap.runs):
        fmon = raw["flux_monitors"]
        faces = collect_box_faces(fmon, port=0)
        box_default = np.sum([f["signed_default"] for f in faces], axis=0)
        box_exact = np.sum([f["signed_exact_f64"] for f in faces], axis=0)
        y_default = np.sum([f["signed_default"] for f in faces
                            if f["axis"] == "y"], axis=0)
        y_exact = np.sum([f["signed_exact_f64"] for f in faces
                          if f["axis"] == "y"], axis=0)
        xz_exact = np.sum([f["signed_exact_f64"] for f in faces
                           if f["axis"] != "y"], axis=0)
        msl_nm = next(nm for nm in fmon if nm.startswith("_mixed_flux_msl"))
        msl_d, msl_e = _flux_both_ways(fmon[msl_nm])
        mx_d, mx_e = _flux_both_ways(fmon[_FLUX_MX_NAME])
        px_d, px_e = _flux_both_ways(fmon[_FLUX_PX_NAME])
        flux_records.append({
            "faces": faces,
            "box_net_default": box_default, "box_net_exact_f64": box_exact,
            "box_y_pair_default": y_default, "box_y_pair_exact_f64": y_exact,
            "box_xz_only_exact_f64": xz_exact,
            "plane_msl_name": msl_nm,
            "plane_msl_default": msl_d, "plane_msl_exact_f64": msl_e,
            "plane_mx_default": mx_d, "plane_mx_exact_f64": mx_e,
            "plane_px_default": px_d, "plane_px_exact_f64": px_e,
        })

    # capture fidelity: the per-face box must reproduce the lane's own net
    lane_box = np.asarray(diag["box_lw_flux"])[:, 0, :]
    fid_box = float(np.max(np.abs(
        np.asarray([fr["box_net_default"] for fr in flux_records]) - lane_box)
        / np.maximum(np.abs(lane_box), np.finfo(float).tiny)))
    lane_plane = np.asarray(diag["plane_msl_flux"])[:, 0, :]
    fid_plane = float(np.max(np.abs(
        np.asarray([fr["plane_msl_default"] for fr in flux_records])
        - lane_plane) / np.maximum(np.abs(lane_plane), np.finfo(float).tiny)))
    print(f"capture fidelity (per-face box net vs lane box_lw): "
          f"max rel dev {fid_box:.3e}")
    print(f"capture fidelity (msl plane vs lane plane_msl):     "
          f"max rel dev {fid_plane:.3e}")
    if max(fid_box, fid_plane) > 1e-9:
        raise AssertionError(
            "the re-derived flux surfaces diverge from the lane's own — do "
            "not trust anything downstream")

    # --- F1 ---------------------------------------------------------------
    p_line_px = rp_lw["p_line_slot0"]
    p_line_mx = -flux_records[lw_run]["plane_mx_exact_f64"]
    box_net = flux_records[lw_run]["box_net_exact_f64"]
    r1 = (p_line_px + p_line_mx) / box_net
    # REPORTED-ONLY symmetric variant: both branches full cross-section.
    r1_sym = ((flux_records[lw_run]["plane_px_exact_f64"] + p_line_mx)
              / box_net)
    # The B2 attribution denominator: the box rebuilt WITHOUT its y faces.
    xz_box = flux_records[lw_run]["box_xz_only_exact_f64"]
    r1_xz = (p_line_px + p_line_mx) / np.where(xz_box != 0.0, xz_box, 1.0)
    f1 = evaluate_f1(zc_im_re=zc_im_re, r1=r1, r1_xz=r1_xz,
                     y_face_contrib=flux_records[lw_run]["box_y_pair_exact_f64"],
                     box_net=box_net, settling_db=settling)

    # --- F2 / F3 ----------------------------------------------------------
    v0_msl = np.asarray(diag["v0_msl"])
    i_msl = np.asarray(diag["i_msl"])
    b_msl_lw = b_msl_probe0(v0_msl[lw_run, 0], i_msl[lw_run, 0], z0_hj)
    b_msl_msl = b_msl_probe0(v0_msl[msl_run, 0], i_msl[msl_run, 0], z0_hj)
    a_msl_msl = a_msl_probe0(v0_msl[msl_run, 0], i_msl[msl_run, 0], z0_hj)
    a_msl_lw = a_msl_probe0(v0_msl[lw_run, 0], i_msl[lw_run, 0], z0_hj)

    m2 = rp_msl["out_over_inc"][0]
    s22_raw = np.abs(S_raw[n_lw, n_lw, :])
    f2 = evaluate_f2(m2=m2, s22=s22_raw,
                     zc_meas_re_band_mean=zc_re_band_mean,
                     settling_db=settling, z0_hj=z0_hj)

    out_lw0 = rp_lw["slots"][0]["out"]
    denom_f3 = np.abs(out_lw0) / np.sqrt(zc.real)
    r3 = np.abs(b_msl_lw) / np.maximum(denom_f3, np.finfo(float).tiny)
    f3 = evaluate_f3(r3=r3, zc_meas_re_band_mean=zc_re_band_mean,
                     settling_db=settling, denominator=denom_f3,
                     signal_floor=0.0, z0_hj=z0_hj)

    # --- witnesses --------------------------------------------------------
    from rfx.api._sparams import _mixed_reciprocity_deviation
    rec_raw = _mixed_reciprocity_deviation(S_raw)
    rec_wave = (None if S_wave is None
                else _mixed_reciprocity_deviation(np.asarray(S_wave)))

    # ==================== printed table ==================================
    print()
    print("=" * 78)
    print("PER-BIN TABLE  (S_raw is the ONLY comparator-eligible S; every "
          "number REPORT-ONLY)")
    print("=" * 78)
    hdr = (f"{'f[GHz]':>7} {'|S00|r':>8} {'|S22|r':>8} {'|S10|r':>8} "
           f"{'M2':>9} {'R3':>9} {'ReZc':>8} {'Im/Re':>8} {'beta/k0':>8} "
           f"{'R1':>9} {'R1sym':>9}")
    print(hdr)
    print("-" * len(hdr))
    k0 = 2.0 * np.pi * freqs / 299792458.0
    for k, f in enumerate(freqs):
        print(f"{f/1e9:7.2f} {abs(S_raw[0,0,k]):8.4f} "
              f"{abs(S_raw[n_lw,n_lw,k]):8.4f} {abs(S_raw[n_lw,0,k]):8.4f} "
              f"{m2[k]:9.4f} {r3[k]:9.4f} {zc.real[k]:8.3f} "
              f"{zc_im_re[k]:8.4f} {rp_lw['beta'][k]/k0[k]:8.4f} "
              f"{r1[k]:9.4f} {r1_sym[k]:9.4f}")
    print()
    print(f"|b_msl| (lw drive, probe 0)  : "
          f"{np.abs(b_msl_lw).tolist()}")
    print("   convention: _b_msl = (V0 - Z_hj I)/(2 sqrt(Z_hj)); the 2 is "
          "INSIDE, so |b|^2 is the")
    print("   wave power in the Re(V I*) (no 1/2) convention and F3 divides "
          "|out| by sqrt(Re Zc),")
    print("   NOT by 2 sqrt(Re Zc). |out|^2/(4 Re Zc) would under-read by "
          "exactly 4 in power.")
    print(f"|out| (lw, slot 0)           : {np.abs(out_lw0).tolist()}")
    print(f"|out|/sqrt(Re Zc)            : {denom_f3.tolist()}")
    print(f"P_line(+x) slot 0            : {p_line_px.tolist()}")
    print(f"P_line(-x) = -flux(1.44 mm)  : {p_line_mx.tolist()}")
    print(f"P_box_net (exact_f64)        : {box_net.tolist()}")
    print(f"box y-pair (exact_f64)       : "
          f"{flux_records[lw_run]['box_y_pair_exact_f64'].tolist()}")
    for fr_i, fr in enumerate(flux_records):
        print(f"run {fr_i} ({drive_plan[fr_i][0]}-driven) faces:")
        for fc in fr["faces"]:
            print(f"    {fc['name']:<34} sgn {fc['sign']:+.0f} "
                  f"exact_f64 {np.array2string(fc['flux_exact_f64'], precision=6)}")
        print(f"    plane_msl  {np.array2string(fr['plane_msl_exact_f64'], precision=6)}")
        print(f"    plane -x   {np.array2string(fr['plane_mx_exact_f64'], precision=6)}")
        print(f"    plane +x   {np.array2string(fr['plane_px_exact_f64'], precision=6)}")
    print()
    print(f"settling_db per drive        : {settling.tolist()}")
    print("   NOTE: n_probes=3, so this is a max over 3 witness planes, not "
          "the committed run's 5.")
    print("   A max over a subset can only be LOWER (flattering), so it is "
          "NOT bit-comparable")
    print("   with the committed -122.57 / -119.93 dB at num_periods=60.")
    n_live_lw = [int(len(spec.live_cells)) for spec, _ in
                 (cap.runs[lw_run].get("wire") or ())]
    cond_rec = cond_a_raw_and_column_normalized(
        v_lw=np.asarray(diag["v_lw"]), i_lw=np.asarray(diag["i_lw"]),
        v0_msl=v0_msl, i_msl=i_msl,
        z0_lw=np.asarray(result.z0_ref)[:n_lw],
        n_live_lw=n_live_lw or [1],
        z0_hj_msl=np.asarray(diag["z0_hj_msl"]),
        wire_mode=True, drive_plan=drive_plan)
    print(f"n_live_lw                    : {n_live_lw}")
    print(f"cond(A) raw                  : {cond_rec.get('cond_a_raw')}")
    print(f"cond(A) column-normalized    : "
          f"{cond_rec.get('cond_a_column_normalized')}")
    print("   (comparable with the committed 27.1 / 2.01; REPORTED, NEVER "
          "GATED)")
    print(f"ill_cond / neg_power masks   : "
          f"{masks.get('ill_cond')} / {masks.get('neg_power')}")
    print(f"reciprocity (S_raw)          : {rec_raw}")
    print(f"reciprocity (wave channel)   : {rec_wave}")
    print(f"passivity_correction         : "
          f"{None if passivity_correction is None else passivity_correction.tolist()}")
    print(f"wall clock                   : {wall_clock_s:.2f} s")
    print()
    print("=" * 78)
    print("FALSIFIER VERDICTS (predeclared; 'UNRESOLVED' and "
          "'NON_DISCRIMINATING' are legal outcomes)")
    print("=" * 78)
    for verdict in (f1, f2, f3):
        side = verdict["side"] or "— (no side; nothing convicted)"
        print(f"{verdict['falsifier']}: {verdict['verdict']}")
        print(f"    came out against : {side}")
        print(f"    reason           : {verdict['reason']}")
    print()
    print("F4 is NOT a falsifier on this run — prediction only, not pinned: "
          f"{_F4_PREDICTION_NOT_A_FALSIFIER['band_quoted_not_pinned']}")
    if not is_measurement:
        bad = [v["falsifier"] for v in (f1, f2, f3)
               if v["verdict"] != "UNRESOLVED"]
        if bad:
            raise AssertionError(
                f"reduced-settings smoke: {bad} did NOT read UNRESOLVED, but "
                "the settling precondition cannot be met at "
                f"num_periods={num_periods:g}. The evaluators are wrong.")
        print("SMOKE CHECK: all three falsifiers read UNRESOLVED, as required "
              "at reduced settings.")

    # ==================== artifact =======================================
    payload = {
        "what_this_is": (
            "REPORT-ONLY diagnostic record of the issue #498/#517 "
            "reference-plane-instrumented mixed-lane run. Not a reference, "
            "not a gate, not a fixture. Pins nothing."),
        "predeclaration": (
            "docs/design_notes/mixed_refplane_predeclaration.md"),
        "is_the_measurement": is_measurement,
        "reduced_settings_note": (
            None if is_measurement else
            f"num_periods={num_periods:g} is a plumbing smoke, NOT the "
            "measurement; its falsifier verdicts are UNRESOLVED by "
            "construction."),
        "do_not_pin": list(DO_NOT_PIN),
        "f4_prediction_not_a_falsifier": _F4_PREDICTION_NOT_A_FALSIFIER,
        "bookkeeping": {**run_bookkeeping(), "wall_clock_s": wall_clock_s,
                        "field_dtype": str(np.asarray(S_raw).dtype)},
        "fixture": {
            "eps_r": _EPS_R, "h_sub": _H_SUB, "w_trace": _W_TRACE, "dx": _DX,
            "domain": list(_DOMAIN), "cpml_layers": _CPML_LAYERS,
            "freq_max": _FREQ_MAX, "x_feed": _X_FEED, "x_msl": _X_MSL,
            "freqs_hz": freqs.tolist(), "num_periods": num_periods,
            "declared_deviations": {
                "feed_direction": _FEED_DIRECTION,
                "reference_plane_cells": _REFPLANE_N,
                "n_probes": _N_PROBES,
            },
            "grid_shape": [int(v) for v in grid.shape],
            "dt": dt, "dx_grid": dx,
        },
        "instrumentation": {
            "mechanism": (
                "monkeypatch Simulation._forward_from_materials: swap "
                "self._ports for dataclasses.replace(pe, "
                "reference_plane_cells=N) preserving excite, pass a legal "
                "_sparam_drive_idx, restore. Zero shipped-code change; the "
                "mixed lane's reference_plane_cells rejection and the "
                "support-matrix text stay untouched."),
            "positive_control": cap.positive_control,
            "positive_control_note": _POSITIVE_CONTROL_NOTE,
            "no_lw_drive_sentinel": _NO_LW_DRIVE_SENTINEL,
        },
        "capture_fidelity": {
            "box_faces_vs_lane_box_lw_max_rel_dev": fid_box,
            "msl_plane_vs_lane_plane_msl_max_rel_dev": fid_plane,
        },
        # The openEMS referee's input contract (see
        # scripts/diagnostics/probe_fed_msl_openems_referee.py, "Required
        # keys"): top-level ``freqs_hz`` plus ``s_raw`` nested [2][2][n_freqs]
        # of [re, im], index 0 = lumped/wire family, index 1 = MSL. Emitted
        # natively so the referee can consume this artifact directly; VESSL
        # 369367257607 refused an earlier artifact that carried the same
        # content only under the names below, and needed
        # scripts/diagnostics/mixed_refplane_artifact_to_referee_contract.py to
        # re-shape it. Same numbers, two spellings, no projected S either way.
        "freqs_hz": [float(f) for f in np.asarray(freqs).tolist()],
        "s_raw": [[[[float(S_raw[i, j, k].real), float(S_raw[i, j, k].imag)]
                    for k in range(S_raw.shape[2])]
                   for j in range(S_raw.shape[1])]
                  for i in range(S_raw.shape[0])],
        "s_matrix": {
            "S_raw": _c2(S_raw), "S_raw_shape": list(S_raw.shape),
            "S_shipped_post_passivity": _c2(S_shipped),
            "S_wave": None if S_wave is None else _c2(S_wave),
            "passivity_correction": (
                None if passivity_correction is None
                else passivity_correction.tolist()),
            "port_names": list(result.port_names),
            "port_families": list(result.port_families),
            "z0_ref": np.asarray(result.z0_ref).tolist(),
            "magnitude_channel": result.magnitude_channel,
            "comparator_rule": (
                "S_raw only. result.S is a joint SVD projection under "
                "enforce_passivity=True and is never compared."),
            "lw_diagonal_note": (
                "|S00| is recorded raw and is NEVER compared, asserted or "
                "bounded here. F4 is a prediction, not a result."),
        },
        "refplane": {
            "lw_drive_run": lw_run, "msl_drive_run": msl_run,
            "runs": [
                {
                    "drive": drive_plan[i][0],
                    "outboard_sign": rp["outboard_sign"],
                    "separation_m": rp["separation_m"],
                    "zc": _c2(rp["zc"]),
                    "zc_im_over_re": np.abs(rp["zc"].imag / rp["zc"].real).tolist(),
                    "beta": np.asarray(rp["beta"]).tolist(),
                    "slow_wave_ratio_beta_over_k0": (
                        np.asarray(rp["beta"]) / k0).tolist(),
                    "p_line_slot0": rp["p_line_slot0"].tolist(),
                    "out_over_inc": {str(s): v.tolist()
                                     for s, v in rp["out_over_inc"].items()},
                    "slots": {
                        str(s): {
                            "plane_index": rp["slots"][s]["plane_index"],
                            "n_cells_outboard": rp["slots"][s]["n_cells_outboard"],
                            "plane_v": _c2(rp["slots"][s]["plane_v"]),
                            "plane_im": _c2(rp["slots"][s]["plane_im"]),
                            "plane_ip": _c2(rp["slots"][s]["plane_ip"]),
                            "i_corr": _c2(rp["slots"][s]["i_corr"]),
                            "out": _c2(rp["slots"][s]["out"]),
                            "inc": _c2(rp["slots"][s]["inc"]),
                        } for s in (0, 1)
                    },
                } for i, rp in enumerate(rp_runs)
            ],
        },
        "msl_channel": {
            "z0_hj_msl": z0_hj,
            "v0_msl": _c2(v0_msl), "i_msl": _c2(i_msl),
            "v0_i_shape": list(np.asarray(v0_msl).shape),
            "b_msl_lw_drive_probe0": _c2(b_msl_lw),
            "a_msl_lw_drive_probe0": _c2(a_msl_lw),
            "b_msl_msl_drive_probe0": _c2(b_msl_msl),
            "a_msl_msl_drive_probe0": _c2(a_msl_msl),
            "abs_S22_raw": s22_raw.tolist(),
            "z0_msl_fit_abs_diagnostic_only": np.asarray(
                diag["z0_msl_fit_abs"]).tolist(),
            "convention": (
                "_b_msl = (V0 - Z_hj I)/(2 sqrt(Z_hj)) — the 2 is INSIDE, so "
                "b = V-/sqrt(Z) is a FULL-amplitude power wave whose |b|^2 "
                "is power in the Re(V I*) (no 1/2) convention."),
        },
        "flux": [
            {
                "drive": drive_plan[i][0],
                "faces": [
                    {"name": fc["name"], "axis": fc["axis"],
                     "sign": fc["sign"], "coordinate_m": fc["coordinate_m"],
                     "flux_default": fc["flux_default"].tolist(),
                     "flux_exact_f64": fc["flux_exact_f64"].tolist()}
                    for fc in fr["faces"]
                ],
                "box_net_default": fr["box_net_default"].tolist(),
                "box_net_exact_f64": fr["box_net_exact_f64"].tolist(),
                "box_y_pair_exact_f64": fr["box_y_pair_exact_f64"].tolist(),
                "box_xz_only_exact_f64": fr["box_xz_only_exact_f64"].tolist(),
                "plane_msl_default": fr["plane_msl_default"].tolist(),
                "plane_msl_exact_f64": fr["plane_msl_exact_f64"].tolist(),
                "plane_mx_1p44mm_default": fr["plane_mx_default"].tolist(),
                "plane_mx_1p44mm_exact_f64": fr["plane_mx_exact_f64"].tolist(),
                "plane_px_2p56mm_default": fr["plane_px_default"].tolist(),
                "plane_px_2p56mm_exact_f64": fr["plane_px_exact_f64"].tolist(),
            } for i, fr in enumerate(flux_records)
        ],
        "flux_convention": (
            "flux_spectrum returns integral Re(E x H*).n dA with NO 1/2. "
            "R1 quotes the exact_f64 numbers; the lane-default (complex64) "
            "path is recorded beside them because a PARTIAL subnormal flush "
            "corrupts small sums inside R1's window."),
        "witnesses": {
            "cond_a": cond_rec,
            "n_live_lw": n_live_lw,
            "flux_override_masks": masks,
            "settling_db": settling.tolist(),
            "settling_rule_db": _SETTLING_WITNESS_DB,
            "settling_caveat": (
                "n_probes=3 makes settling_db a max over 3 witness planes, "
                "not the committed run's 5; a max over a subset can only be "
                "lower, so it is NOT bit-comparable with the committed "
                "-122.57 / -119.93 dB."),
            "reciprocity_S_raw": (
                None if rec_raw is None
                else {"pair": list(rec_raw[0]), "max_dev": float(rec_raw[1])}),
            "reciprocity_wave": (
                None if rec_wave is None
                else {"pair": list(rec_wave[0]), "max_dev": float(rec_wave[1])}),
            "s21_power_witness": np.asarray(
                result.s21_power_witness).tolist(),
            "reliable": (None if result.reliable is None
                         else np.asarray(result.reliable).tolist()),
            "beta_railed": (None if result.beta_railed is None
                            else np.asarray(result.beta_railed).tolist()),
        },
        "f1_inputs": {
            "R1": r1.tolist(),
            "R1_symmetric_reported_only": r1_sym.tolist(),
            "R1_xz_only_box": r1_xz.tolist(),
            "P_line_plus_x": p_line_px.tolist(),
            "P_line_minus_x": p_line_mx.tolist(),
            "asymmetry_note": (
                "P_line(+x) is a LINE-MODE power at the reference plane; "
                "P_line(-x) is a FULL cross-section flux that also catches "
                "substrate/air radiation. R1 is therefore mixed-kind by "
                "predeclaration and R1_symmetric (both full cross-section) is "
                "reported alongside to size the one-sided systematic."),
        },
        "falsifiers": {"F1": f1, "F2": f2, "F3": f3},
        "preflight_text": preflight_text,
        "warnings": caught,
    }
    _assert_no_lw_diagonal_pin(payload)

    if args.no_write:
        print("--no-write: artifact not written.")
        return 0
    out_dir = _assert_report_only(Path(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _assert_report_only(out_dir / ARTIFACT_NAME)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"artifact -> {out_path}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
