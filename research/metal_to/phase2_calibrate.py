"""Phase-2 calibration — the empty-line reference, insertion loss, and the dx/2 mesh.

Closes the two gaps that block scoring anything at all (joint review, 2026-08-27).

G-A. NOTHING IN THE REPO PRODUCED IL.
    ``score_dualband`` is written entirely on INSERTION LOSS RELATIVE TO THE
    EMPTY LINE,

        IL(f) = 20 log10|S21_empty(f)| - 20 log10|S21_dut(f)|   [dB, positive
                                                                 = attenuation]

    and ``score_dualband.check_validity`` gates on ``empty_cal_max_db``. But
    ``phase2_fixture.solve`` returns ABSOLUTE ``s21_db`` only, there was no
    empty-line solve, no IL helper and no ``empty_cal_max_db`` anywhere. Every
    number in the phase was therefore one hand-written subtraction away from
    being scored, with the sign convention as the thing to get wrong. Worse,
    ``LY`` changed when the fixture went two-sided (20.696 -> 27.559 mm), so the
    Stage-0 assumption "empty line measured at 0.00 dB" is a statement about a
    fixture that no longer exists and has to be RE-MEASURED before any IL is
    quotable.

    This module supplies :func:`empty_reference` (solve the bare two-sided
    fixture once, cache it), :func:`insertion_loss` (the subtraction, in the one
    direction ``score_dualband`` means) and :func:`score_design` (solve -> IL ->
    frozen score, assembled here so no caller assembles it by hand).

G-B. THE QUOTABLE ROBUSTNESS NUMBER NEEDS THE dx/2 MESH.
    ``robust_eval`` is explicit that the coarse mesh cannot express the +-50 um
    PCB etch tolerance: its only representable offset is +-127 um = 2.54x the
    spec, which tests a board nobody would ship. At dx/2 = 63.5 um the offset is
    1.27x and quotable. ``phase2_fixture`` now takes ``dx`` and derives every
    cell count from frozen PHYSICAL planes (see its ``Mesh`` block); this module
    adds the piece that makes a coarse design testable there --
    :func:`refine_mask`, exact 2x2 cell replication -- and the cross-mesh
    equality check that says the two meshes really do describe one rectangle.

WHAT MAKES A dx/2 NUMBER COMPARABLE, AND WHAT STILL DOES NOT
------------------------------------------------------------
Comparable, and asserted by the smoke below:
  * identical design-box rectangle, to the nanometre, at both meshes;
  * identical realized conductor -- the 635 um trace the coarse mesh actually
    rasterizes, resolved by 10 fine cells instead of 5 coarse ones, and
    ``T_METAL`` thick at both;
  * a coarse design maps to the fine mesh by whole-cell replication, so the
    refined design IS the coarse design, not a re-drawing of it.

Not yet controlled, and each one is a reason a fine-mesh number will not
reproduce a coarse-mesh one exactly (they are mesh convergence, which is the
point of running both -- but they must not be read as etch effects):
  * Yee staircase at the substrate interface. The coarse fixture's own preflight
    says it: 2 substrate cells at dx = 127 um, ">5% Z0 staircase error expected,
    refine to dx <= 64 um". The fine mesh is precisely the mesh preflight asks
    for, so Z0 SHOULD move between the two, by roughly the amount preflight
    predicts. If it moves much more, that is a finding, not a nuisance.
  * the MSL ports keep their validated definition (nominal centre, ``W_TRACE``
    wide), so the fine mesh's port window covers 9 of the 10 realized trace cell
    rows. Read the port Z0 back at both meshes.
  * a live coarse solve was seen reporting "Z0 for MSL port 'msl_0' = 13.72 ohm
    deviates 89.3% from analytic 47.89 ohm" on a deliberately truncated 2-period
    record. That is almost certainly an N-probe transient artifact of the short
    window and not a geometry problem -- but it has never been controlled at a
    SETTLED window, so :func:`empty_reference` captures every solver warning
    into the cached record and :attr:`EmptyReference.z0_warnings` surfaces them.
    A calibration whose Z0 readback is still off at 90 periods is not a
    calibration.

COST
----
The empty-line solve is the same price as any other solve at that mesh, and it
is paid ONCE per (dx, window, frequency grid) -- hence the cache.
:func:`cost_estimate` prints the arithmetic, anchored on the Stage-0 measured
83 s / 90 periods on one 4090.

Run the smoke (CPU; builds both meshes, no expensive solve):
    JAX_PLATFORMS=cpu PYTHONPATH=<repo root> python research/metal_to/phase2_calibrate.py

Add ``--toy-solve`` to also run one real 2-period 3-frequency empty solve at the
coarse mesh, which proves the solve/cache path end to end. That is a smoke, not
a calibration: 2 periods does not settle and its numbers are not quotable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import phase2_fixture as fx                     # noqa: E402
import score_dualband as sd                     # noqa: E402
from phase2_fixture import DX, SIDES, Mesh      # noqa: E402
from score_dualband import SCORE, Thresholds    # noqa: E402

__all__ = [
    "CACHE_DIR", "FIXTURE_VERSION",
    "fixture_geometry", "reference_key",
    "EmptyReference", "empty_reference",
    "ILResult", "insertion_loss", "empty_calibration_max_db",
    "ScoredDesign", "score_design",
    "refine_mask", "coarsen_mask", "refine_mask_to_box",
    "assert_boxes_agree", "cost_estimate",
]

# ---------------------------------------------------------------------------
# 0. Cache location and provenance
# ---------------------------------------------------------------------------
CACHE_DIR = Path(os.environ.get(
    "EMPTY_REF_DIR", str(HERE / "out_vessl" / "empty_ref")))

#: Bump this when the FIXTURE changes in a way the geometry dict below does not
#: already capture (a boundary spec, a port definition, an excitation rule).
#: Every cached reference records it and a mismatch is a cache miss, not a
#: silent reuse -- the whole failure mode this cache could introduce is scoring
#: a new fixture against an old fixture's empty line.
FIXTURE_VERSION = "phase2-two-sided-2026-08-27"

#: Stage-0 measurement anchor for :func:`cost_estimate` (NOTE_stage0_window.md,
#: jobs 369367256474-478): the ORIGINAL one-sided fixture, 90 periods,
#: 41 299 steps, 954 180 cells, 161 frequencies, two port drives, 83 s on one
#: 4090. Cost is taken proportional to cells x steps x drives; the frequency
#: count rides along in the DFT accumulation and is a small correction, so it is
#: reported but not scaled on.
_ANCHOR_CELLS_MEASURED = 279 * 180 * 19          # 954 180, the ORIGINAL fixture
_ANCHOR_STEPS = 41299
_ANCHOR_WALL_S = 83.0


# ---------------------------------------------------------------------------
# 1. Geometry fingerprint -- what a cached empty line is a reference FOR
# ---------------------------------------------------------------------------
def fixture_geometry(dx: float = DX) -> dict:
    """Every number that decides what ``build_sim(freqs, dx)`` produces.

    This is the cache key's payload and it is deliberately verbose: a cached
    empty-line reference is only valid for the fixture it was solved on, and
    the way that goes wrong is a geometry constant moving while the file name
    stays the same. Compare dicts, not names.
    """
    m = fx.mesh(dx)
    b = m.physical_bounds()
    return {
        "fixture_version": FIXTURE_VERSION,
        "eps_r": fx.EPS_R, "h_sub_m": fx.H_SUB, "w_trace_m": fx.W_TRACE,
        "l_line_m": fx.L_LINE, "port_margin_m": fx.PORT_MARGIN,
        "f_max_hz": fx.F_MAX, "cpml_layers": fx.CPML_LAYERS,
        "port_impedance_ohm": 50.0,
        "boundary": "x=cpml,y=cpml,z=(lo:pec,hi:cpml)",
        "lx_m": fx.LX, "ly_m": fx.LY, "lz_m": fx.LZ,
        "dx_m": m.dx, "refine": m.refine,
        "trace_box_y_lo_m": fx.TRACE_Y_LO,
        "trace_box_y_hi_m": m.trace_box_y_hi,
        "y_trace_m": fx.Y_TRACE,
        "t_metal_m": fx.T_METAL,
        "cells": {"nx_box": m.nx_box, "ny_side": m.ny_side,
                  "n_trace": m.n_trace_cells, "edge_margin": m.edge_margin_cells,
                  "ny_total": m.ny_cells, "n_metal": m.n_metal_cells},
        "planes_m": {k: float(v) for k, v in b.items()},
    }


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)


def _freq_fingerprint(freqs_hz) -> dict:
    f = np.asarray(freqs_hz, dtype=np.float64).ravel()
    if f.size == 0:
        raise ValueError("empty frequency grid")
    if np.any(np.diff(f) <= 0):
        raise ValueError("frequency grid must be strictly ascending")
    mhz = [int(round(x / 1e6)) for x in f]
    if len(set(mhz)) != len(mhz):
        raise ValueError(
            "two frequencies round to the same integer MHz; the scoring grid "
            "is indexed in integer MHz and would collide")
    return {"n": int(f.size),
            "sha256": hashlib.sha256(f.tobytes()).hexdigest(),
            "mhz": mhz}


def reference_key(freqs_hz, num_periods: float, dx: float = DX) -> str:
    """Cache key: (dx, window, frequency grid, fixture geometry). 16 hex chars."""
    payload = {
        "geometry": fixture_geometry(dx),
        "num_periods": float(num_periods),
        "freqs": _freq_fingerprint(freqs_hz),
    }
    return hashlib.sha256(_canon(payload).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 2. The empty-line reference
# ---------------------------------------------------------------------------
@dataclass
class EmptyReference:
    """One solved bare two-sided fixture, and where it came from."""

    key: str
    path: Path
    cached: bool
    dx: float
    num_periods: float
    freqs_mhz: np.ndarray
    s21_db: np.ndarray
    s11_db: np.ndarray
    empty_cal_max_db: float
    record: dict
    geometry: dict
    warnings_text: list = field(default_factory=list)

    @property
    def settled(self) -> bool:
        return bool(self.record.get("settled", False))

    @property
    def quotable(self) -> bool:
        """Settled AND inside the frozen ``EMPTY_CAL_MAX_DB`` gate.

        ``score_dualband.check_validity`` gates the DESIGN's settling and the
        reference's ``empty_cal_max_db``, but nothing gates the REFERENCE's own
        settling -- an unsettled empty line is a ring-down artifact subtracted
        from every design in the campaign, with no term anywhere that would
        show it. :func:`score_design` refuses one by default.
        """
        return (self.settled
                and self.empty_cal_max_db <= sd.EMPTY_CAL_MAX_DB)

    @property
    def z0_warnings(self) -> list:
        """Solver warnings mentioning Z0 -- minor (e) of the joint review.

        A settled empty line whose port Z0 readback is far from the analytic
        Hammerstad-Jensen value is not a usable calibration, whatever its
        |S21| looks like.
        """
        return [w for w in self.warnings_text if "reported Z0" in w]

    def summary(self) -> str:
        s = self.record.get("settling_worst_db")
        return (f"empty line  dx={self.dx*1e6:.2f} um  {self.num_periods:g} "
                f"periods  {self.freqs_mhz.size} freqs  "
                f"|IL_empty|max = {self.empty_cal_max_db:.4f} dB "
                f"(gate {sd.EMPTY_CAL_MAX_DB:.2f})  settling "
                f"{'n/a' if s is None else f'{s:.1f} dB'}  "
                f"quotable={'YES' if self.quotable else 'NO'}  "
                f"{'CACHED' if self.cached else 'solved'}  key={self.key}")


def _empty_path(key: str, num_periods: float, dx: float,
                cache_dir: Path) -> Path:
    return Path(cache_dir) / (f"empty_dx{dx*1e6:.2f}um_"
                              f"p{float(num_periods):g}_{key}.json")


def empty_reference(freqs_hz, num_periods: float, dx: float = DX,
                    cache_dir: Path | str = CACHE_DIR,
                    force: bool = False,
                    solver: Callable | None = None,
                    verbose: bool = True) -> EmptyReference:
    """Solve (or reload) the BARE two-sided fixture -- the IL denominator.

    Parameters
    ----------
    freqs_hz : the DFT bins. Must be the SAME grid the DUT is solved on;
        :func:`score_design` refuses to subtract two different grids.
    num_periods : the record window, in periods of ``F_MAX``. Stage-0 measured
        45 for descent and 90 for verification ON THE ONE-SIDED FIXTURE. LY
        changed, so those windows are inherited, not proven, until an empty
        reference at this fixture is seen to settle -- which is exactly what
        ``record['settling_worst_db']`` in the cached JSON records.
    dx : mesh. A reference is valid for ONE mesh; the key includes it.
    force : re-solve and overwrite an existing cache entry.
    solver : test hook. ``solver(sim, freqs_hz, num_periods) -> record dict``,
        defaulting to ``phase2_fixture.solve``. Used by the smoke to exercise
        the assembly without an FDTD run; a record produced this way is written
        with ``"synthetic": True`` so it can never be mistaken for a solve.

    Notes
    -----
    The cache is keyed on the FIXTURE GEOMETRY, not on a file name, and the
    stored geometry is re-compared on load. If the fixture moves, the key
    changes and you get a cache MISS -- which is the correct behaviour and the
    reason ``LY`` changing invalidated the Stage-0 "empty line = 0.00 dB"
    assumption instead of silently keeping it.
    """
    cache_dir = Path(cache_dir)
    geom = fixture_geometry(dx)
    key = reference_key(freqs_hz, num_periods, dx)
    path = _empty_path(key, num_periods, dx, cache_dir)
    f_mhz = np.asarray(_freq_fingerprint(freqs_hz)["mhz"], dtype=int)

    if path.exists() and not force:
        d = json.loads(path.read_text())
        if _canon(d.get("geometry")) != _canon(geom):
            raise RuntimeError(
                f"cached empty reference {path} carries a DIFFERENT fixture "
                f"geometry than the current one, under the same key. Delete it "
                f"and re-solve; do not score against it.")
        rec = d["record"]
        got = np.asarray(rec["freqs_MHz"], dtype=int)
        if not np.array_equal(got, f_mhz):
            raise RuntimeError(f"cached empty reference {path} was solved on a "
                               f"different frequency grid")
        return EmptyReference(
            key=key, path=path, cached=True, dx=float(dx),
            num_periods=float(num_periods), freqs_mhz=got,
            s21_db=np.asarray(rec["s21_db"], dtype=float),
            s11_db=np.asarray(rec["s11_db"], dtype=float),
            empty_cal_max_db=float(d["empty_cal_max_db"]),
            record=rec, geometry=d["geometry"],
            warnings_text=list(d.get("warnings", [])))

    fixture = fx.build_sim(freqs_hz, dx=dx)
    pre = [str(m) for m in fixture.sim.preflight()]
    if verbose:
        print(f"[empty] dx={dx*1e6:.2f} um  grid="
              f"{tuple(fixture.sim._build_grid().shape)}  "
              f"{len(pre)} preflight message(s)  periods={num_periods:g}  "
              f"freqs={f_mhz.size}")

    run = solver if solver is not None else fx.solve
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec = run(fixture.sim, freqs_hz, float(num_periods))
    wall = time.time() - t0
    wtext = [str(w.message) for w in caught]

    got = np.asarray(rec["freqs_MHz"], dtype=int)
    if not np.array_equal(got, f_mhz):
        raise RuntimeError("the solver returned a different frequency grid "
                           "than it was asked for")
    cal = empty_calibration_max_db(rec["s21_db"])

    out = {
        "key": key, "dx_m": float(dx), "num_periods": float(num_periods),
        "empty_cal_max_db": cal,
        "empty_cal_gate_db": sd.EMPTY_CAL_MAX_DB,
        "empty_cal_pass": bool(cal <= sd.EMPTY_CAL_MAX_DB),
        "synthetic": solver is not None,
        "wall_s": round(wall, 1),
        "geometry": geom,
        "preflight": pre,
        "warnings": wtext,
        "record": rec,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    if verbose:
        print(f"[empty] |IL_empty|max = {cal:.4f} dB "
              f"(gate {sd.EMPTY_CAL_MAX_DB:.2f} dB -> "
              f"{'PASS' if cal <= sd.EMPTY_CAL_MAX_DB else 'FAIL'})  "
              f"settling {rec.get('settling_worst_db')}  {wall:.1f} s  "
              f"-> {path}")
        if not rec.get("settled", False):
            print(f"[empty] !! NOT SETTLED ({rec.get('settling_worst_db')} dB, "
                  f"need <= {sd.SETTLING_MAX_DB:.0f}). This reference is a "
                  f"ring-down artifact and would be subtracted from every "
                  f"design in the campaign. Do not score against it.")
        for w in wtext:
            print(f"[empty] solver warning: {str(w)[:160]}")

    return EmptyReference(
        key=key, path=path, cached=False, dx=float(dx),
        num_periods=float(num_periods), freqs_mhz=got,
        s21_db=np.asarray(rec["s21_db"], dtype=float),
        s11_db=np.asarray(rec["s11_db"], dtype=float),
        empty_cal_max_db=cal, record=rec, geometry=geom, warnings_text=wtext)


# ---------------------------------------------------------------------------
# 3. Insertion loss -- the one subtraction, in the one direction
# ---------------------------------------------------------------------------
class ILResult(NamedTuple):
    """``il_db`` is what ``score_dualband.score`` wants; the rest is the gate."""

    il_db: np.ndarray
    empty_cal_max_db: float


def empty_calibration_max_db(empty_s21_db) -> float:
    """``max |20 log10|S21_empty||`` -- how far the bare line is from 0 dB.

    ``score_dualband.EMPTY_CAL_MAX_DB`` gates this at 0.10 dB. It is a check on
    the REFERENCE, not on any design: a lossless through-line in a matched
    fixture must transmit at 0 dB, and every dB it does not is a dB of fixture
    artifact that the IL subtraction will carry into every design's score with
    the opposite sign. (The one-sided fixture's measured floor was 0.011 dB.)
    """
    e = np.asarray(empty_s21_db, dtype=float).ravel()
    if e.size == 0:
        raise ValueError("empty reference has no samples")
    if not np.all(np.isfinite(e)):
        raise ValueError("empty reference contains non-finite dB values")
    return float(np.max(np.abs(e)))


def insertion_loss(dut_s21_db, empty_s21_db) -> ILResult:
    """IL(f) = |S21_empty|_dB - |S21_dut|_dB. POSITIVE MEANS ATTENUATION.

    That is the sign ``score_dualband`` is written in ("IL(f) = 20*log10|
    S21_empty(f)| - 20*log10|S21_dut(f)| [dB, positive = attenuation]"), and it
    is the sign every threshold in the frozen metric assumes: ``r_req_db =
    20`` is 20 dB of REJECTION, ``il_pass_db = 1`` is 1 dB of passband LOSS.
    Get it backwards and a perfect filter scores as a catastrophe, silently,
    because both signs produce plausible-looking numbers.

    Both arguments are ABSOLUTE dB traces on the SAME frequency grid --
    ``phase2_fixture.solve``'s ``s21_db`` for the design and
    :attr:`EmptyReference.s21_db` for the reference.

    Returns
    -------
    ILResult(il_db, empty_cal_max_db) -- the trace to score, and the reference's
    own calibration number to hand to ``score_dualband.check_validity``.
    """
    d = np.asarray(dut_s21_db, dtype=float).ravel()
    e = np.asarray(empty_s21_db, dtype=float).ravel()
    if d.shape != e.shape:
        raise ValueError(
            f"IL needs the design and the empty line on the SAME grid: "
            f"{d.size} design samples vs {e.size} empty samples")
    if not np.all(np.isfinite(d)):
        raise ValueError("design S21 trace contains non-finite dB values")
    return ILResult(il_db=e - d, empty_cal_max_db=empty_calibration_max_db(e))


# ---------------------------------------------------------------------------
# 4. solve -> IL -> frozen score, assembled once
# ---------------------------------------------------------------------------
@dataclass
class ScoredDesign:
    """One design, solved, calibrated and scored. ``result`` is the frozen one."""

    result: sd.Result
    validity: sd.Validity
    il_db: np.ndarray
    freqs_mhz: np.ndarray
    record: dict
    empty_key: str
    empty_cal_max_db: float
    n_boxes: int
    label: str = "design"

    def as_response(self) -> dict:
        """The dict ``robust_eval.robust_score`` accepts as one etch field.

        Carries the raw trace, so ``robust_score`` reports FREE-RUNNING notch
        drift rather than falling back to the frozen metric's in-band centres
        (which pin at a band edge when a notch walks out of its band).
        """
        return {
            "freqs_mhz": np.asarray(self.freqs_mhz),
            "il_db": np.asarray(self.il_db),
            "s11_db": np.asarray(self.record["s11_db"], dtype=float),
            "s21_db_abs": np.asarray(self.record["s21_db"], dtype=float),
            "validity": self.validity,
        }

    def summary(self) -> str:
        r = self.result
        return (f"{self.label}: M={r.M:.2f} [S_L={r.S_L:.2f} S_U={r.S_U:.2f} "
                f"S_G={r.S_G:.2f} S_P={r.S_P:.2f}] Omega={r.Omega:+.2f} "
                f"R_L={r.R_L_raw:.1f} R_U={r.R_U_raw:.1f} dB  "
                f"boxes={self.n_boxes}  "
                f"valid={'yes' if self.validity.ok else 'NO'}")


def score_design(mask=None, *, freqs_hz, num_periods: float, dx: float = DX,
                 boxes=None, thr: Thresholds = SCORE,
                 empty: EmptyReference | None = None,
                 cache_dir: Path | str = CACHE_DIR,
                 solver: Callable | None = None,
                 label: str = "design",
                 require_settled_empty: bool = True,
                 verbose: bool = True) -> ScoredDesign:
    """Build -> solve -> calibrate -> score, in one call.

    This exists so that no caller ever writes the subtraction. Give it a
    two-sided mask (``{'lo':..., 'hi':...}`` or a ``(2, nx, ny)`` array) or an
    explicit list of PEC boxes, and it returns the FROZEN metric's ``Result``
    with a fully populated ``Validity`` attached -- including
    ``empty_cal_max_db``, which nothing else in the repo was computing, so the
    ``check_validity`` gate on it could never fire.

    The empty-line reference is fetched (or solved once and cached) at the same
    ``(dx, freqs, num_periods)``; a mismatch is an exception, not a broadcast.

    ``passivity_correction`` is taken from ``phase2_fixture.solve``'s record,
    which maps the solver's ``None`` (== nothing needed projecting == perfectly
    passive) to zeros. Passing the raw ``None`` through ``np.asarray(...,
    dtype=float)`` yields ``nan``, ``nan <= 0.05`` is False, and the cleanest
    possible run is declared NOT QUOTABLE -- defect D5 of the joint review, live
    in ``xval1_imperative.py`` at the time of writing. Never build this record
    by hand.
    """
    fixture = fx.build_sim(freqs_hz, dx=dx)
    grid = fixture.sim._build_grid()
    box = fx.design_box(grid, fixture.mesh)

    if mask is not None and boxes is not None:
        raise ValueError("give a mask or boxes, not both")
    if mask is not None:
        boxes = fx.boxes_from_mask(mask, box)
    n_boxes = fx.add_pec_boxes(fixture.sim, boxes) if boxes else 0

    if empty is None:
        empty = empty_reference(freqs_hz, num_periods, dx=dx,
                                cache_dir=cache_dir, solver=solver,
                                verbose=verbose)
    if abs(empty.dx - dx) > 1e-15 or abs(empty.num_periods - num_periods) > 1e-9:
        raise ValueError(
            f"empty reference is dx={empty.dx*1e6:.2f} um / "
            f"{empty.num_periods:g} periods but the design is solved at "
            f"dx={dx*1e6:.2f} um / {num_periods:g} periods")
    if require_settled_empty and not empty.settled:
        raise ValueError(
            f"the empty-line reference at {empty.path} is NOT SETTLED "
            f"(settling {empty.record.get('settling_worst_db')} dB, need <= "
            f"{sd.SETTLING_MAX_DB:.0f}). Subtracting an unsettled reference "
            f"puts its ring-down into every design's IL with no term that "
            f"would show it. Re-solve at a longer window, or pass "
            f"require_settled_empty=False and label the result NOT QUOTABLE.")

    run = solver if solver is not None else fx.solve
    rec = run(fixture.sim, freqs_hz, float(num_periods))

    f_mhz = np.asarray(rec["freqs_MHz"], dtype=int)
    if not np.array_equal(f_mhz, np.asarray(empty.freqs_mhz, dtype=int)):
        raise ValueError("design and empty reference are on different "
                         "frequency grids -- IL would be meaningless")

    il, cal = insertion_loss(rec["s21_db"], empty.s21_db)
    il_clipped = np.minimum(il, thr.r_cap_db)
    validity = sd.check_validity(
        rec["settling_db"], rec["passivity_correction"], rec["reliable"],
        f_mhz, il_clipped, thr=thr, empty_cal_max_db=cal)
    result = sd.score(f_mhz, il,
                      s11_db=np.asarray(rec["s11_db"], dtype=float),
                      s21_db_abs=np.asarray(rec["s21_db"], dtype=float),
                      thr=thr, validity=validity)
    out = ScoredDesign(result=result, validity=validity, il_db=il,
                       freqs_mhz=f_mhz, record=rec, empty_key=empty.key,
                       empty_cal_max_db=cal, n_boxes=n_boxes, label=label)
    if verbose:
        print("[score] " + out.summary())
    return out


# ---------------------------------------------------------------------------
# 5. Mesh refinement of a mask  (gap G-B)
# ---------------------------------------------------------------------------
def _sides(mask) -> dict | None:
    """Two-sided mask -> {'lo','hi'}; a plain 2-D mask -> None (handle direct)."""
    if isinstance(mask, Mapping):
        missing = set(SIDES) - set(mask)
        if missing:
            raise KeyError(f"two-sided mask is missing side(s) {sorted(missing)}")
        extra = set(mask) - set(SIDES)
        if extra:
            raise KeyError(f"two-sided mask has unknown side(s) {sorted(extra)}")
        return {n: np.asarray(mask[n]) for n in SIDES}
    a = np.asarray(mask)
    if a.ndim == 3 and a.shape[0] == 2:
        return {"lo": a[0], "hi": a[1]}
    if a.ndim == 2:
        return None
    raise TypeError(f"mask must be 2-D, (2, nx, ny) or a two-sided mapping; "
                    f"got shape {a.shape}")


def _refine_2d(m: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(m), factor, axis=0), factor, axis=1)


def refine_mask(mask, factor: int = 2):
    """Map a coarse-mesh mask onto a ``factor``x finer mesh by cell replication.

    Each coarse cell becomes a ``factor x factor`` block of fine cells, which
    occupies EXACTLY the same physical rectangle (``phase2_fixture.mesh``
    guarantees the two meshes' design boxes coincide plane for plane). So the
    refined design is not an approximation of the coarse design -- it is the
    same metal, re-sampled. That is what lets a design optimized at dx be
    ETCH-TESTED at dx/2, where +-1 cell = 63.5 um is 1.27x the +-50 um PCB
    tolerance instead of the coarse mesh's unquotable 2.54x.

    Accepts a two-sided mapping, a ``(2, nx, ny)`` array or a bare 2-D mask,
    and returns the same form.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    s = _sides(mask)
    if s is None:
        return _refine_2d(mask, factor)
    out = {n: _refine_2d(s[n], factor) for n in SIDES}
    if isinstance(mask, Mapping):
        return out
    return np.stack([out["lo"], out["hi"]])


def _coarsen_2d(m: np.ndarray, factor: int, rule: str) -> np.ndarray:
    a = np.asarray(m)
    nx, ny = a.shape
    if nx % factor or ny % factor:
        raise ValueError(f"shape {a.shape} is not divisible by {factor}")
    blocks = (a >= 0.5).reshape(nx // factor, factor, ny // factor, factor)
    if rule == "all":
        return blocks.all(axis=(1, 3)).astype(np.uint8)
    if rule == "any":
        return blocks.any(axis=(1, 3)).astype(np.uint8)
    if rule == "majority":
        return (blocks.mean(axis=(1, 3)) >= 0.5).astype(np.uint8)
    raise ValueError("rule must be 'all', 'any' or 'majority'")


def coarsen_mask(mask, factor: int = 2, rule: str = "all"):
    """Inverse of :func:`refine_mask` for a mask that came FROM one.

    Exists to make the round trip assertable. It is NOT a way to bring an
    etch-perturbed fine mask back to the coarse mesh -- an eroded or dilated
    fine mask is a sub-coarse-cell structure by construction, and the three
    rules ('all', 'any', 'majority') would disagree about it, which is exactly
    the information the fine mesh was introduced to keep.
    """
    factor = int(factor)
    s = _sides(mask)
    if s is None:
        return _coarsen_2d(mask, factor, rule)
    out = {n: _coarsen_2d(s[n], factor, rule) for n in SIDES}
    if isinstance(mask, Mapping):
        return out
    return np.stack([out["lo"], out["hi"]])


def refine_mask_to_box(mask, coarse_box, fine_box):
    """Refine ``mask`` from ``coarse_box``'s mesh to ``fine_box``'s, checked.

    Verifies the shapes against both design boxes and that the two boxes
    describe the same physical rectangle, so a mask cannot be silently refined
    into a differently-placed region.
    """
    fx.assert_same_physical_bounds(coarse_box.mesh, fine_box.mesh)
    factor = fine_box.mesh.refine // coarse_box.mesh.refine
    if factor * coarse_box.mesh.refine != fine_box.mesh.refine or factor < 1:
        raise ValueError(
            f"fine mesh refine={fine_box.mesh.refine} is not an integer "
            f"multiple of coarse refine={coarse_box.mesh.refine}")
    s = _sides(mask)
    if s is None:
        raise TypeError("refine_mask_to_box needs a two-sided mask")
    for n in SIDES:
        if s[n].shape != coarse_box.side(n).shape:
            raise ValueError(f"side '{n}' mask shape {s[n].shape} != coarse "
                             f"box shape {coarse_box.side(n).shape}")
    out = {n: _refine_2d(s[n], factor) for n in SIDES}
    for n in SIDES:
        if out[n].shape != fine_box.side(n).shape:
            raise ValueError(f"refined side '{n}' shape {out[n].shape} != fine "
                             f"box shape {fine_box.side(n).shape}")
    return out


def assert_boxes_agree(coarse_box, fine_box, tol_m: float = 1e-12) -> dict:
    """The two design boxes must occupy the same physical rectangle.

    Checks the REALIZED ``BoxSide.extent_m`` on each grid -- i.e. the numbers
    that will be handed to ``rfx.Box`` -- not only the mesh bookkeeping. ``z``
    is compared too and is expected to match, because ``T_METAL`` is fixed.
    """
    fx.assert_same_physical_bounds(coarse_box.mesh, fine_box.mesh, tol_m)
    out, bad = {}, []
    for name in SIDES:
        a, b = coarse_box.side(name).extent_m, fine_box.side(name).extent_m
        for axis, ax in enumerate("xyz"):
            for edge in (0, 1):
                d = b[axis][edge] - a[axis][edge]
                out[f"{name}.{ax}{'lo' if edge == 0 else 'hi'}"] = (
                    a[axis][edge], b[axis][edge], d)
                if abs(d) > tol_m:
                    bad.append(f"side '{name}' {ax}"
                               f"{'lo' if edge == 0 else 'hi'}: "
                               f"{a[axis][edge]*1e6:.3f} vs "
                               f"{b[axis][edge]*1e6:.3f} um "
                               f"(delta {d*1e9:.1f} nm)")
    if bad:
        raise AssertionError("realized design boxes differ between meshes:\n  "
                             + "\n  ".join(bad))
    return out


# ---------------------------------------------------------------------------
# 6. Cost
# ---------------------------------------------------------------------------
def cost_estimate(dx: float = DX, num_periods: float = 90.0,
                  n_freqs: int = 123, grid=None) -> dict:
    """Wall-clock estimate for ONE solve at ``dx``, anchored on Stage-0.

    Anchor (NOTE_stage0_window.md): the ORIGINAL one-sided fixture, 954 180
    cells, 41 299 steps, two port drives, 83 s on one 4090. Cost is taken
    proportional to cells x steps; the DFT accumulation scales with n_freqs and
    is reported separately rather than folded in, because it is a small
    fraction of a 41 000-step run and pretending to know the constant would be
    false precision.

    The estimate is for a GPU. It is NOT a CPU estimate; the smoke's 2-period
    toy solve runs on CPU and is orders of magnitude off this line.
    """
    if grid is None:
        fixture = fx.build_sim(np.array([5.0e9]), dx=dx)
        grid = fixture.sim._build_grid()
    cells = int(np.prod(grid.shape))
    dt = float(grid.dt)
    n_steps = int(round(float(num_periods) * (1.0 / fx.F_MAX) / dt))
    cell_steps = cells * n_steps
    anchor_cs = _ANCHOR_CELLS_MEASURED * _ANCHOR_STEPS
    return {
        "dx_um": dx * 1e6, "grid": tuple(int(s) for s in grid.shape),
        "cells": cells, "dt_ps": dt * 1e12, "n_steps": n_steps,
        "record_ns": n_steps * dt * 1e9,
        "dft_res_MHz": (1.0 / (n_steps * dt)) / 1e6 if n_steps else float("nan"),
        "cell_steps": cell_steps,
        "vs_anchor": cell_steps / anchor_cs,
        "wall_est_s": _ANCHOR_WALL_S * cell_steps / anchor_cs,
        "n_freqs": int(n_freqs),
    }


def _fmt_cost(c: dict) -> str:
    return (f"dx={c['dx_um']:6.2f} um  grid={str(c['grid']):>18s}  "
            f"{c['cells']:>10,d} cells  dt={c['dt_ps']:.4f} ps  "
            f"{c['n_steps']:>7,d} steps ({c['record_ns']:.2f} ns, "
            f"res {c['dft_res_MHz']:.0f} MHz)  "
            f"{c['cell_steps']:.3e} cell-steps = {c['vs_anchor']:6.2f}x the "
            f"Stage-0 anchor  -> ~{c['wall_est_s']:.0f} s "
            f"({c['wall_est_s']/60:.1f} min) per solve on one 4090")


# ---------------------------------------------------------------------------
# 7. Smoke
# ---------------------------------------------------------------------------
def _synthetic_solver(bands_db: float = 0.0, floor_db: float = 0.011,
                      settling_db: float = -120.0):
    """A ``solver``-shaped callable that fabricates a plausible record.

    Exists so the ASSEMBLY (IL sign, validity gates, frozen score) can be
    exercised without an FDTD run. ``bands_db = 0`` is a bare line;
    ``bands_db > 0`` is a dual-band notch of that depth. Anything it produces is
    marked ``"synthetic": True`` all the way into the cache file, and it touches
    no rfx internals, so it cannot drift with the solver.
    """
    def run(sim, freqs_hz, num_periods):
        f = np.asarray(freqs_hz, dtype=np.float64)
        mhz = np.array([int(round(x / 1e6)) for x in f], dtype=int)
        s21 = np.full(f.size, -floor_db)
        if bands_db:
            for lo, hi in (sd.BAND_L_MHZ, sd.BAND_U_MHZ):
                s21[(mhz >= lo) & (mhz <= hi)] = -floor_db - bands_db
        s11 = np.full(f.size, -20.0)
        return dict(
            num_periods=float(num_periods), n_freqs=int(f.size), n_steps=0,
            record_ns=float("nan"), dft_res_GHz=float("nan"), wall_s=0.0,
            freqs_GHz=[float(x) / 1e9 for x in f],
            freqs_MHz=[int(x) for x in mhz],
            s21_db=[float(x) for x in s21], s11_db=[float(x) for x in s11],
            f_min_GHz=float(f[int(np.argmin(s21))] / 1e9),
            depth_min_db=float(np.min(s21)),
            settling_worst_db=float(settling_db),
            settled=bool(settling_db <= sd.SETTLING_MAX_DB),
            reliable_bins=[2 * int(f.size), 2 * int(f.size)],
            passivity_worst=0.0, passivity_projected=False,
            settling_db=[float(settling_db)] * int(f.size),
            reliable=np.ones((2, f.size), dtype=bool).tolist(),
            passivity_correction=[0.0] * int(f.size),
        )
    return run


def _smoke(toy_solve: bool = False) -> int:
    ok = True
    rule = "=" * 100
    print(rule)
    print("PHASE-2 CALIBRATION — G-A (empty-line reference + IL) and "
          "G-B (dx/2 mesh)")
    print(rule)

    # ---- (1) both meshes ---------------------------------------------------
    print("\n-- 1. the two meshes --")
    m_c, m_f = fx.mesh(DX), fx.mesh(DX / 2.0)
    for m in (m_c, m_f):
        print(f"  {m.describe()}")
    for a, b, what in ((m_c.nx_box, m_f.nx_box, "nx_box"),
                       (m_c.ny_side, m_f.ny_side, "ny_side"),
                       (m_c.n_trace_cells, m_f.n_trace_cells, "n_trace"),
                       (m_c.ny_cells, m_f.ny_cells, "ny_total"),
                       (m_c.n_metal_cells, m_f.n_metal_cells, "n_metal")):
        if b != 2 * a:
            print(f"  !! {what}: fine {b} is not 2x coarse {a}")
            ok = False
    print(f"  every coarse cell count doubles exactly -> a coarse mask refines "
          f"by 2x2 replication with no remainder")

    fixt_c = fx.build_sim(np.array([5.25e9]), dx=DX)
    fixt_f = fx.build_sim(np.array([5.25e9]), dx=DX / 2.0)
    g_c, g_f = fixt_c.sim._build_grid(), fixt_f.sim._build_grid()
    for tag, g in (("coarse", g_c), ("fine  ", g_f)):
        print(f"  {tag}  GRID SHAPE = {tuple(g.shape)}  "
              f"({int(np.prod(g.shape)):,d} cells)  dt = "
              f"{float(g.dt)*1e12:.4f} ps  pads = {tuple(g.axis_pads)}")
    print(f"  cells x{np.prod(g_f.shape)/np.prod(g_c.shape):.2f}   "
          f"dt x{float(g_f.dt)/float(g_c.dt):.3f}   "
          f"-> a fixed-duration record costs "
          f"{np.prod(g_f.shape)/np.prod(g_c.shape)*float(g_c.dt)/float(g_f.dt):.1f}x")
    if abs(float(g_c.dt) / float(g_f.dt) - 2.0) > 1e-6:
        print("  !! dt did not halve — the CFL step is not tracking dx")
        ok = False

    # ---- (2) the design box, physically ------------------------------------
    print("\n-- 2. the design box, in metres, at both meshes --")
    box_c = fx.design_box(g_c, m_c)
    box_f = fx.design_box(g_f, m_f)
    print(f"  {'':6s} {'side':5s} {'cells x':>14s} {'cells y':>14s} "
          f"{'x (mm)':>20s} {'y (mm)':>20s} {'z (um)':>18s}")
    for tag, bx in (("coarse", box_c), ("fine", box_f)):
        for name, s in bx.items():
            (xl, xh), (yl, yh), (zl, zh) = s.extent_m
            print(f"  {tag:6s} {name:5s} "
                  f"[{s.ix_lo:5d},{s.ix_hi:5d}) [{s.iy_lo:5d},{s.iy_hi:5d}) "
                  f"[{xl*1e3:8.4f},{xh*1e3:8.4f}] "
                  f"[{yl*1e3:8.4f},{yh*1e3:8.4f}] "
                  f"[{zl*1e6:7.2f},{zh*1e6:7.2f}]")
    try:
        agree = assert_boxes_agree(box_c, box_f)
        worst = max(abs(d) for _, _, d in agree.values())
        print(f"  -> PHYSICAL BOUNDS MATCH on all {len(agree)} realized box "
              f"planes; worst disagreement {worst*1e9:.3f} nm")
        print(f"     (and every mesh-level plane too: "
              f"{sorted(m_c.physical_bounds())})")
    except AssertionError as e:
        print(f"  !! {e}")
        ok = False

    print(f"  design variables: coarse {box_c.n_vars}  fine {box_f.n_vars} "
          f"(x{box_f.n_vars/box_c.n_vars:.0f})")
    for tag, g, bx, m in (("coarse", g_c, box_c, m_c), ("fine", g_f, box_f, m_f)):
        j0, j1 = fx._trace_cell_span(g, m)
        contig = (bx.lo.iy_hi == j0) and (bx.hi.iy_lo == j1)
        yl, yh = (j0 - g.axis_pads[1]) * m.dx, (j1 - g.axis_pads[1]) * m.dx
        print(f"  {tag:6s} rasterized trace cells [{j0},{j1}) = {j1-j0} cells "
              f"= y [{yl*1e3:.4f}, {yh*1e3:.4f}] mm "
              f"({(yh-yl)*1e6:.1f} um wide)  design box contiguous with it: "
              f"{'YES' if contig else 'NO'}")
        if not contig:
            print("  !! design metal would be DETACHED from the feed line")
            ok = False

    # ---- (2b) what the refined mesh does to preflight ----------------------
    print("\n-- 2b. preflight at both meshes (empty fixtures) --")
    pre_c = [(getattr(x, "code", "uncoded"), str(x))
             for x in fixt_c.sim.preflight()]
    pre_f = [(getattr(x, "code", "uncoded"), str(x))
             for x in fixt_f.sim.preflight()]
    for tag, pre in (("coarse", pre_c), ("fine  ", pre_f)):
        print(f"  {tag}: {len(pre)} message(s), codes "
              f"{sorted({c for c, _ in pre})}")
    gone = sorted({c for c, _ in pre_c} - {c for c, _ in pre_f})
    new = sorted({c for c, _ in pre_f} - {c for c, _ in pre_c})
    print(f"  codes the refinement REMOVES: {gone}")
    print(f"  codes the refinement ADDS   : {new}")
    z0_c = [m for c, m in pre_c if "substrate cell" in m]
    z0_f = [m for c, m in pre_f if "substrate cell" in m]
    print(f"  'only N substrate cells in z' warnings: coarse {len(z0_c)}, "
          f"fine {len(z0_f)}  -- the coarse fixture's own preflight asks for "
          f"dx <= 64 um and 4+ aligned substrate cells; the fine mesh IS that "
          f"mesh (h_sub/dx = {fx.H_SUB/(DX/2):.0f}, integer), so Z0 is expected "
          f"to MOVE between the meshes by roughly the bias preflight quotes. "
          f"That is mesh convergence, not an etch effect, and it must not be "
          f"read as one.")
    if new:
        print("  !! the refined mesh introduces preflight codes the coarse "
              "mesh does not have -- read them before quoting a fine number")

    # ---- (3) mask refinement ----------------------------------------------
    print("\n-- 3. mask refinement (coarse design -> fine mesh) --")
    x_mid = fx.LX / 2.0
    stubs = [("lo", x_mid - 4.0e-3, fx.W_TRACE, fx.quarter_wave(5.25e9)),
             ("hi", x_mid + 4.0e-3, fx.W_TRACE, fx.quarter_wave(5.775e9))]
    mk_c = fx.mask_from_stubs(stubs, box_c)
    mk_f = refine_mask_to_box(mk_c, box_c, box_f)
    fill_c = sum(int(mk_c[n].sum()) for n in SIDES)
    fill_f = sum(int(mk_f[n].sum()) for n in SIDES)
    print(f"  classical two-stub mask: coarse fill {fill_c} cells, fine fill "
          f"{fill_f} cells (x{fill_f/fill_c:.0f})")
    print(f"  metal area: coarse {fill_c*DX*DX*1e6:.6f} mm^2   fine "
          f"{fill_f*(DX/2)**2*1e6:.6f} mm^2   "
          f"delta {abs(fill_c*DX*DX - fill_f*(DX/2)**2)*1e12:.3e} um^2")
    if fill_f != 4 * fill_c:
        print("  !! refinement did not quadruple the cell count")
        ok = False

    back = coarsen_mask(mk_f, 2, "all")
    rt = all(np.array_equal(np.asarray(mk_c[n], dtype=bool),
                            np.asarray(back[n], dtype=bool)) for n in SIDES)
    rt_any = all(np.array_equal(
        np.asarray(mk_c[n], dtype=bool),
        np.asarray(coarsen_mask(mk_f, 2, "any")[n], dtype=bool)) for n in SIDES)
    print(f"  round trip coarse -> fine -> coarse: "
          f"rule='all' {'OK' if rt else 'MISMATCH'}, "
          f"rule='any' {'OK' if rt_any else 'MISMATCH'} "
          f"(they must agree: every fine 2x2 block is uniform by construction)")
    if not (rt and rt_any):
        ok = False

    # the strong check: the COARSE boxes, re-rasterized on the FINE grid by
    # rfx's own cell-centre containment rule, must reproduce the refined mask.
    boxes_c = fx.boxes_from_mask(mk_c, box_c)
    re_ras = fx.mask_from_boxes(boxes_c, box_f)
    same = all(np.array_equal(np.asarray(re_ras[n], dtype=bool),
                              np.asarray(mk_f[n], dtype=bool)) for n in SIDES)
    print(f"  coarse PEC boxes re-rasterized on the FINE box == refined mask: "
          f"{'OK' if same else 'MISMATCH'}  ({len(boxes_c)} coarse boxes, "
          f"{len(fx.boxes_from_mask(mk_f, box_f))} fine boxes)")
    if not same:
        ok = False

    # a random free-form design too -- stubs are the easy case
    rng = np.random.default_rng(20260827)
    rnd = {n: (rng.random(s.shape) < 0.25).astype(np.uint8)
           for n, s in box_c.items()}
    rnd_f = refine_mask_to_box(rnd, box_c, box_f)
    rnd_back = coarsen_mask(rnd_f, 2, "majority")
    same_r = all(np.array_equal(np.asarray(rnd[n], dtype=bool),
                                np.asarray(rnd_back[n], dtype=bool))
                 for n in SIDES)
    same_b = all(np.array_equal(
        np.asarray(fx.mask_from_boxes(fx.boxes_from_mask(rnd, box_c),
                                      box_f)[n], dtype=bool),
        np.asarray(rnd_f[n], dtype=bool)) for n in SIDES)
    print(f"  free-form p=0.25 ({sum(int(rnd[n].sum()) for n in SIDES)} cells): "
          f"round trip {'OK' if same_r else 'MISMATCH'}, "
          f"box re-rasterization {'OK' if same_b else 'MISMATCH'}")
    if not (same_r and same_b):
        ok = False

    # ---- (4) insertion loss: the sign, stated and tested -------------------
    print("\n-- 4. insertion loss --")
    g = sd.scoring_grid_mhz()
    empty_trace = np.full(g.size, -0.011)          # a nearly ideal bare line
    dut_trace = empty_trace.copy()
    inband = ((g >= sd.BAND_L_MHZ[0]) & (g <= sd.BAND_L_MHZ[1])) | \
             ((g >= sd.BAND_U_MHZ[0]) & (g <= sd.BAND_U_MHZ[1]))
    dut_trace[inband] -= 25.0                      # 25 dB of rejection
    il, cal = insertion_loss(dut_trace, empty_trace)
    print(f"  empty |S21| = {empty_trace[0]:+.3f} dB flat, design |S21| = "
          f"{dut_trace[inband][0]:+.3f} dB in-band")
    print(f"  IL in-band = {il[inband].min():+.2f} dB (POSITIVE = attenuation), "
          f"IL out-of-band = {il[~inband].max():+.3f} dB")
    print(f"  empty_cal_max_db = {cal:.4f} dB  (gate "
          f"{sd.EMPTY_CAL_MAX_DB:.2f} dB -> "
          f"{'PASS' if cal <= sd.EMPTY_CAL_MAX_DB else 'FAIL'})")
    if not (il[inband].min() > 24.0 and abs(il[~inband]).max() < 1e-9):
        print("  !! IL sign or magnitude is wrong")
        ok = False
    r_right = sd.score(g, il)
    r_wrong = sd.score(g, -il)
    print(f"  scored with the RIGHT sign: M = {r_right.M:.2f} "
          f"(S_L={r_right.S_L:.2f} S_U={r_right.S_U:.2f})")
    print(f"  scored with the FLIPPED sign: M = {r_wrong.M:.2f} "
          f"(S_L={r_wrong.S_L:.2f} S_U={r_wrong.S_U:.2f}) "
          f"-- a 25 dB filter reads as a total failure. This is why the "
          f"subtraction lives in one place.")
    if not (r_right.M < 1.0 < r_wrong.M):
        print("  !! the sign test did not separate")
        ok = False
    for bad, why in (((np.zeros(5), np.zeros(6)), "length mismatch"),
                     ((np.array([np.nan, 0.0]), np.zeros(2)), "non-finite")):
        try:
            insertion_loss(*bad)
        except ValueError as e:
            print(f"  guard ok ({why}): {str(e)[:70]}")
        else:
            print(f"  !! no guard for {why}")
            ok = False

    # ---- (5) score_design end to end, synthetic solver ---------------------
    print("\n-- 5. score_design assembly (synthetic solver, no FDTD) --")
    freqs_hz = g.astype(float) * 1e6
    syn_dir = CACHE_DIR / "_smoke_synthetic"
    fake_empty = _synthetic_solver(bands_db=0.0, floor_db=0.011)
    fake_dut = _synthetic_solver(bands_db=25.0, floor_db=0.011)
    emp = empty_reference(freqs_hz, 90.0, dx=DX, cache_dir=syn_dir,
                          force=True, solver=fake_empty, verbose=False)
    print(f"  {emp.summary()}")
    emp2 = empty_reference(freqs_hz, 90.0, dx=DX, cache_dir=syn_dir,
                           solver=fake_empty, verbose=False)
    print(f"  second call -> cached={emp2.cached}  same key={emp2.key == emp.key}"
          f"  identical trace="
          f"{np.array_equal(emp.s21_db, emp2.s21_db)}")
    if not (emp2.cached and emp2.key == emp.key
            and np.array_equal(emp.s21_db, emp2.s21_db)):
        print("  !! the cache did not round-trip")
        ok = False
    key_other_dx = reference_key(freqs_hz, 90.0, DX / 2.0)
    key_other_win = reference_key(freqs_hz, 45.0, DX)
    key_other_grid = reference_key(freqs_hz[:-1], 90.0, DX)
    print(f"  key(dx)={emp.key}  key(dx/2)={key_other_dx}  "
          f"key(45 periods)={key_other_win}  key(shorter grid)={key_other_grid}")
    if len({emp.key, key_other_dx, key_other_win, key_other_grid}) != 4:
        print("  !! the cache key does not separate mesh / window / grid")
        ok = False

    scored = score_design(fx.mask_from_stubs(stubs, box_c), freqs_hz=freqs_hz,
                          num_periods=90.0, dx=DX, empty=emp,
                          cache_dir=syn_dir, solver=fake_dut,
                          label="synthetic 25 dB dual notch", verbose=False)
    print(f"  {scored.summary()}")
    print(f"  validity: settled={scored.validity.settled} "
          f"passivity_worst={scored.validity.passivity_worst:.3f} "
          f"empty_cal_max_db={scored.validity.empty_cal_max_db:.4f} "
          f"-> ok={scored.validity.ok}")
    if not (scored.validity.ok and scored.result.M < 1.0):
        print("  !! the assembled score is not what the synthetic design means")
        ok = False
    if scored.result.validity is None or \
            scored.result.validity.get("empty_cal_max_db") is None:
        print("  !! empty_cal_max_db did not reach the frozen Result")
        ok = False

    # an UNSETTLED empty line must be refused, not silently subtracted
    unsettled = empty_reference(
        freqs_hz, 10.0, dx=DX, cache_dir=syn_dir, force=True, verbose=False,
        solver=_synthetic_solver(0.0, 0.011, settling_db=-18.8))
    print(f"  unsettled reference (10 periods, -18.8 dB): settled="
          f"{unsettled.settled}  quotable={unsettled.quotable}")
    try:
        score_design(fx.mask_from_stubs(stubs, box_c), freqs_hz=freqs_hz,
                     num_periods=10.0, dx=DX, empty=unsettled,
                     cache_dir=syn_dir, solver=fake_dut, verbose=False)
    except ValueError as e:
        print(f"  guard ok (unsettled empty refused): {str(e)[:90]}...")
    else:
        print("  !! an unsettled empty-line reference was accepted")
        ok = False

    # composition with robust_eval (defect D4 was that nothing composed)
    try:
        import robust_eval as re_
        resp = {k: scored.as_response() for k in re_.FIELD_ORDER}
        rr = re_.robust_score(resp, etch=re_.calibrate_etch(
            cell_um=re_.CELL_FINE_UM))
        print(f"  as_response() feeds robust_eval.robust_score: "
              f"M_worst={rr.M_worst:.2f} all_valid={rr.all_valid} "
              f"(three identical fields, so spread must be 0: "
              f"{rr.M_spread:.2f})")
        if rr.M_spread != 0.0 or rr.all_valid is not True:
            print("  !! composition with robust_eval misbehaved")
            ok = False
    except Exception as e:                                  # pragma: no cover
        print(f"  !! robust_eval composition failed: {type(e).__name__}: {e}")
        ok = False

    # ---- (6) cost ----------------------------------------------------------
    print("\n-- 6. cost of a REAL empty-line calibration --")
    print(f"  anchor: Stage-0 measured {_ANCHOR_WALL_S:.0f} s for "
          f"{_ANCHOR_STEPS:,d} steps on {_ANCHOR_CELLS_MEASURED:,d} cells "
          f"(one-sided fixture, 2 port drives, one 4090)")
    costs = {}
    for tag, dxv, gg in (("coarse", DX, g_c), ("fine", DX / 2.0, g_f)):
        for per in (45.0, 90.0):
            c = cost_estimate(dxv, per, n_freqs=int(g.size), grid=gg)
            costs[(tag, per)] = c
            print(f"  {tag:6s} {per:5.0f} periods  {_fmt_cost(c)}")
    r90 = costs[("fine", 90.0)]["wall_est_s"] / costs[("coarse", 90.0)]["wall_est_s"]
    print(f"  -> the fine mesh costs {r90:.1f}x the coarse mesh at the same "
          f"window. The empty line is ONE solve per (mesh, window, grid) and "
          f"is then cached, so it is a fixed overhead of "
          f"~{costs[('fine', 90.0)]['wall_est_s']/60:.0f} min for the whole "
          f"fine-mesh campaign, not a per-design cost.")
    print(f"  the three-field etch bracket needs 3 design solves + the 1 shared "
          f"empty line = "
          f"{4*costs[('fine', 90.0)]['wall_est_s']/60:.0f} min per design at "
          f"dx/2, 90 periods.")

    # ---- (7) optional real toy solve --------------------------------------
    if toy_solve:
        print("\n-- 7. toy 2-period empty solve (CPU) — proves the solve path --")
        print("     NOT a calibration: 2 periods does not settle and nothing "
              "from it is quotable.")
        toy_f = np.array([5.15e9, 5.25e9, 5.775e9])
        t0 = time.time()
        toy = empty_reference(toy_f, 2.0, dx=DX,
                              cache_dir=CACHE_DIR / "_smoke_toy", force=True)
        print(f"  {toy.summary()}")
        print(f"  s21_db = {np.round(toy.s21_db, 3).tolist()}   "
              f"s11_db = {np.round(toy.s11_db, 2).tolist()}")
        print(f"  settling {toy.record['settling_worst_db']:.1f} dB -> settled="
              f"{toy.settled} (expected NOT settled at 2 periods); "
              f"passivity_worst={toy.record['passivity_worst']:.4f}, "
              f"projected={toy.record['passivity_projected']}")
        print(f"  solver warnings: {len(toy.warnings_text)} "
              f"({len(toy.z0_warnings)} mentioning Z0)")
        for w in toy.z0_warnings:
            print(f"    Z0: {w[:200]}")
        print(f"  wall {time.time()-t0:.0f} s on CPU")
        again = empty_reference(toy_f, 2.0, dx=DX,
                                cache_dir=CACHE_DIR / "_smoke_toy")
        print(f"  re-read from cache: cached={again.cached}, trace identical="
              f"{np.array_equal(toy.s21_db, again.s21_db)}")
        if not again.cached:
            print("  !! the real-solve cache did not hit")
            ok = False
    else:
        print("\n-- 7. toy solve skipped (pass --toy-solve to run it) --")

    print("\n" + rule)
    print(f"SMOKE: {'PASS' if ok else 'FAIL'}")
    print(rule)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toy-solve", action="store_true",
                    help="also run one real 2-period 3-frequency empty solve "
                         "at the coarse mesh (CPU-feasible, not quotable)")
    args = ap.parse_args()
    return _smoke(toy_solve=args.toy_solve)


if __name__ == "__main__":
    raise SystemExit(main())
