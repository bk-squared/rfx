#!/usr/bin/env python
"""Does the domain wall reach the Sheen filter's stopband?  The openEMS side.

Two 2.413 mm wide 50 ohm microstrip feeds on 0.794 mm of lossless RT/Duroid
(eps_r 2.2) are joined by one wide 20.320 x 2.540 mm low-impedance section.  The
section is a shunt capacitance, so |S21| is flat to about 5.5 GHz and then falls
into a stopband whose transmission zeros near 7 and 8 GHz come from the
transverse resonance of that wide section -- they are set by the field at its
two transverse (y) edges, and the reference's domain wall sits 3.0 mm, 3.8
substrate thicknesses, beyond each of those edges.  What was seen: the frozen
openEMS record puts the deepest 5-10 GHz minimum at 7.9939 GHz on its coarse
rung and 8.0379 GHz on its fine rung, still rising 0.2 % per refinement, while
its box energy sits flat at about -31 dB from 10 to 20 ns and meets no end
criterion -- the record is a declared 10 ns length for that reason.

The two transverse faces of that box are MUR -- the maker's own boundary list
is ``['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR']``, PML on the two x faces
only.  A first-order Mur wall is written for a wave arriving along the wall
normal; the substrate carries its field along the board, across that wall at a
grazing angle.

ASSUMPTION this probe measures, stated as an assumption and not as a finding:
that the transverse wall enters the record's stopband features at all.  Nothing
here decides whether it does, which way it moves the null, or what the record's
-31 dB energy plateau is made of.  This script produces numbers; the leader
reads them.

WHAT IT VARIES
--------------
One mesh rung -- the record's coarse one, ``resolution_factor`` 1.0 = 198.5 um,
417,000 cells, 232 s per 60,000 steps in the reference job -- and four boxes:

    id   Y_CLEAR   y faces    why
    B0    3 mm     MUR        control = the record's ``stage_b_coarse``
    B1   12 mm     MUR        wall moved far out
    B2    3 mm     PML_8      wall made absorbing
    B3   12 mm     PML_8      both

Read per config: the realized mesh (y line count, substrate cells, total
cells), the box energy openEMS reported at the end of the record and its whole
progress trace, the deepest 5-10 GHz minimum, the passband mean and the -3 dB
corner through the maker's own ``_stage_b_features``, the wall time; then the
residual of each null against B0's and the max |S21| difference in dB against
B0 over 2-12 GHz where both curves are above -20 dB.

WHAT THIS IS NOT
----------------
A probe.  STAGE A IS NOT RUN.  The reproduce gate -- openEMS's own
``MSL_NotchFilter`` tutorial, verbatim, judged against the analytic quarter-wave
frequency -- passed in the three jobs that produced the frozen record: VESSL
369367263406, 369367263407 and 369367263408.  This script solves the Sheen
board only, writes nothing under ``tests/crossval/sheen_lpf/reference/``, gates
nothing of its own and commits no record.  Every per-run sanity gate the shared
runner applies is kept: the stdout scan for unused primitives and off-mesh
ports, nonzero excitation energy and non-empty port traces, |S| finite and
<= 2, and passivity over 2-12 GHz.

MECHANICS, AND WHAT WAS CHECKED RATHER THAN ASSUMED
---------------------------------------------------
``tests/crossval/sheen_lpf/reference/make_openems_reference.py`` is imported by
path and left unedited.  Its y-derived module globals are re-derived per config
exactly as it derives them at its lines 428-451:

    Y_CLEAR, Y_SHIFT, IN_FEED_YC, OUT_FEED_YC, PATCH_Y_LO, PATCH_Y_HI, LY

Checked by reading ``_build_sheen_board_at_rung`` (lines 816-875), ``_plan``
and ``_run_stage_b``: all three read those names out of the module at CALL time
and none carries one in a default argument, and ``LX``, ``PATCH_X0``,
``PATCH_X1`` and ``PORT_MARGIN`` do not depend on Y_CLEAR.
``_audit_definition_time_capture`` re-checks both properties at run time and
prints which names each function reads.

The boundary list is a LITERAL inside the builder, so it cannot be moved by a
global.  For B2 and B3 the module-level ``_build_sheen_board_at_rung`` is
replaced by a wrapper that hands the original builder a SUBCLASS of the
``openEMS`` class the runner imported, whose ``SetBoundaryCond`` substitutes
this config's list for whatever the builder passes.  ``_run_stage_b``'s own
inner ``build`` closure looks the builder up as a module global at call time,
so the wrapper reaches it.  If the class cannot be subclassed on the image
(``openEMS`` is a Cython extension type and a final one cannot be subclassed),
the wrapper falls back to calling ``SetBoundaryCond`` again on the built
object, after the builder returns and before the solver runs, which sets the
same six faces; the mechanism that was used is printed and recorded.  B0 and B1
run the untouched builder -- their list is already the maker's own.

``--dry-run`` proves the substitution without openEMS: it calls the real
builder with stub ``openEMS`` / ``ContinuousStructure`` / ``MSLPort`` classes
that record every ``SetBoundaryCond`` call and every mesh line, and prints what
each config would set.  ``--mutate-skip-boundary-override`` removes the wrapper
so B2 prints the builder's own MUR list -- the proof that the printed boundary
comes from the wrapper and not from this script's table.

The reference ran with ``--real-nrts 60000 --real-end-criteria 1e-4
--accept-truncation`` (the record's own ``stage_b_coarse`` meta says so), so
those are the defaults here.

MESH LADDER (``--resolution-factors``)
-------------------------------------
openEMS's own smoothed mesh reads the 8 GHz zero at 7.994 / 8.020 / 8.038 GHz
on the frozen record's three rungs, still rising; openEMS on rfx's node-snapped
uniform lattices reads 8.195 / 8.224 / 8.186 / 8.206 GHz at h/3 .. h/12 (VESSL
369367263706, 369367263704).  ``--resolution-factors`` solves every config at
every factor given, through the maker's own builder, which already takes the
factor (``_build_sheen_board_at_rung``: in-plane smoothing target 198.5 um x
factor, round(4 / factor) substrate cells) -- no geometry changes.  Per
(config, factor) it prints the realized mesh, the box energy and step count
the pass ended at, both zeros (the |S21| minimum in 6.5-7.6 GHz and in
7.6-9.5 GHz, ``refined_extremum`` in log), the -3 dB corner and the passband
mean; then a ladder table per config with the step-to-step change in percent.
CONTROL: B2 at factor 1.0 must reproduce its 8.01582 GHz of VESSL 369367263565
within 0.01 %; a job without factor 1.0 prints that it does not carry the
control.  The default, factor 1.0 alone, is the probe as it was: same passes,
same output, byte for byte.  ``--refuse-truncation`` turns the default
``accept_truncation`` off, so a pass that reaches ``--real-nrts`` fires the
shared runner's gate instead of being recorded.  ``--dry-run`` with a ladder
prints the plan and a cost estimate per factor.

CLI: ``--configs``, ``--out DIR``, ``--sim-root``, ``--threads``,
``--real-nrts``, ``--real-end-criteria``, ``--resolution-factors``,
``--refuse-truncation``, ``--dry-run``, ``--mutate-skip-boundary-override``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------- the configs
MUR_Y = ["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"]
PML_Y = ["PML_8", "PML_8", "PML_8", "PML_8", "PEC", "MUR"]

CONFIGS = (
    {"id": "B0", "y_clear": 3.0e-3, "boundary": MUR_Y,
     "why": "control = the record's stage_b_coarse"},
    {"id": "B1", "y_clear": 12.0e-3, "boundary": MUR_Y,
     "why": "wall moved far out"},
    {"id": "B2", "y_clear": 3.0e-3, "boundary": PML_Y,
     "why": "wall made absorbing"},
    {"id": "B3", "y_clear": 12.0e-3, "boundary": PML_Y,
     "why": "both"},
)
CONTROL_ID = "B0"

PATCHED = ("Y_CLEAR", "Y_SHIFT", "IN_FEED_YC", "OUT_FEED_YC", "PATCH_Y_LO",
           "PATCH_Y_HI", "LY")
AUDITED = ("_build_sheen_board_at_rung", "_plan", "_run_stage_b",
           "_feed_edges_against_lines", "resolution_m")

RESOLUTION_FACTOR = 1.0          # the record's coarse rung, 198.5 um
DEFAULT_NRTS = 60000
DEFAULT_END_CRITERIA = 1.0e-4
BAND_GHZ = (2.0, 12.0)           # the record's own witness band
DEEP_NULL_DB = -20.0             # the case's two-sided deep-null exclusion
STAGE_A_REFERENCE_JOBS = ("369367263406", "369367263407", "369367263408")

# ---- the mesh ladder -------------------------------------------------------
ZERO_BANDS_GHZ = {"zero_7": (6.5, 7.6), "zero_8": (7.6, 9.5)}
# The ladder's control: B2 (PML on y at 3 mm) at factor 1.0 in VESSL
# 369367263565, commit 6635cafa -- the deepest 5-10 GHz minimum it recorded,
# which is its 8 GHz zero.  It ended on its 1e-4 criterion at step 8251.
LADDER_CONTROL = {"id": "B2", "factor": 1.0, "zero_8_ghz": 8.015824224944472,
                  "vessl": "369367263565", "tol": 1.0e-4}
# The cost model's measured inputs.  B2 at factor 1: 417000 cells, 8251 steps
# (1.38396e-9 s of record) in 18.6 s on 8 threads (VESSL 369367263565).  The
# frozen record's rungs (tests/crossval/sheen_lpf/reference/openems_sheen.json
# meta.stages) give the realized cells and dt at factors 1, 1/sqrt2, 1/2 and a
# second throughput, 3.62e8 cell-steps/s on the fine rung.
COST_B2_F1 = {"cells": 417000, "steps": 8251, "record_s": 1.3839572858894961e-09,
              "wall_s": 18.6}
COST_FINE_RATE = 3.62e8
COST_DECAY_MULTIPLES = (1.0, 3.0)


# ---------------------------------------------------------------- the module
def repo_root() -> Path:
    env = os.environ.get("RFX_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def load_maker():
    """``tests/crossval/sheen_lpf/reference/make_openems_reference.py``.

    Loaded by path, as the maker itself loads the shared tutorial gate: this
    file runs as a bare script on the cluster, where ``tests`` is not an
    importable package.
    """
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = (root / "tests" / "crossval" / "sheen_lpf" / "reference"
            / "make_openems_reference.py")
    if not path.is_file():
        raise SystemExit(f"the reference maker is not at {path}")
    spec = importlib.util.spec_from_file_location("_sheen_box_probe_maker", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_sheen_box_probe_maker"] = module
    spec.loader.exec_module(module)
    return module


def config_globals(maker, cfg: dict) -> dict:
    """The seven globals this config implies, derived as the maker derives them."""
    y_clear = float(cfg["y_clear"])
    y_shift = y_clear - maker.PATCH_XS_LO
    patch_y_lo = maker.PATCH_XS_LO + y_shift
    patch_y_hi = maker.PATCH_XS_HI + y_shift
    return {
        "Y_CLEAR": y_clear,
        "Y_SHIFT": y_shift,
        "IN_FEED_YC": maker.IN_FEED_XS_C + y_shift,
        "OUT_FEED_YC": maker.OUT_FEED_XS_C + y_shift,
        "PATCH_Y_LO": patch_y_lo,
        "PATCH_Y_HI": patch_y_hi,
        "LY": patch_y_hi + y_clear,
    }


def snapshot(maker) -> dict:
    return {name: getattr(maker, name) for name in PATCHED}


def apply_globals(maker, values: dict) -> None:
    for name, value in values.items():
        setattr(maker, name, value)


def _audit_definition_time_capture(maker) -> dict:
    control = {name: getattr(maker, name) for name in PATCHED}
    report: dict = {"defaults_carrying_a_patched_value": [], "reads": {}}
    for fname in AUDITED:
        func = getattr(maker, fname)
        defaults = list(func.__defaults__ or ()) + list(
            (func.__kwdefaults__ or {}).values())
        for d in defaults:
            for name, value in control.items():
                if isinstance(d, (int, float)) and not isinstance(d, bool) and d == value:
                    report["defaults_carrying_a_patched_value"].append(
                        f"{fname}: default {d!r} equals {name}")
        report["reads"][fname] = [n for n in PATCHED if n in func.__code__.co_names]
    return report


# --------------------------------------------------- the boundary substitution
def _boundary_class(base_openems, boundary):
    """An ``openEMS`` class whose ``SetBoundaryCond`` sets ``boundary``.

    Returns ``(class, mechanism)``.  ``mechanism`` is ``"subclass"`` when the
    substitution happens inside the builder's own call, and ``"post-call"``
    when the base class refuses to be subclassed and the caller has to set the
    six faces again on the built object instead.
    """
    want = list(boundary)
    try:
        class _OpenEMSBoxProbe(base_openems):
            probe_boundary = want

            def SetBoundaryCond(self, bc):
                return base_openems.SetBoundaryCond(self, list(want))

        _OpenEMSBoxProbe.__name__ = "openEMS_box_probe"
        return _OpenEMSBoxProbe, "subclass"
    except TypeError:
        return base_openems, "post-call"


def make_builder(maker, boundary, *, mutate_skip: bool = False):
    """The module-global builder, wrapped so this config's boundary reaches it.

    ``mutate_skip=True`` returns the original builder untouched: the mutation
    check, whose point is that the printed boundary then falls back to the
    builder's own literal.
    """
    original = maker._build_sheen_board_at_rung
    if mutate_skip:
        return original, ("REMOVED by --mutate-skip-boundary-override; the "
                          f"builder's own literal {maker.B_BOUNDARY} stands")
    if list(boundary) == list(maker.B_BOUNDARY):
        return original, ("none -- the builder's own literal "
                          f"{maker.B_BOUNDARY} is already this config's")

    used: list = []

    def build(ContinuousStructure, openEMS, MSLPort, **kw):
        cls, mechanism = _boundary_class(openEMS, boundary)
        fdtd, p0, p1 = original(ContinuousStructure, cls, MSLPort, **kw)
        if mechanism == "post-call":
            fdtd.SetBoundaryCond(list(boundary))
        used.append(mechanism)
        return fdtd, p0, p1

    build.mechanism_used = used
    return build, "subclass of the runner's openEMS, post-call re-set as fallback"


# ------------------------------------------------------------------ the stubs
class _StubPrimitive:
    def __init__(self, rec, name):
        self._rec, self._name = rec, name

    def AddBox(self, start, stop, **kw):
        self._rec["boxes"].append({"property": self._name,
                                   "start_mm": [v * 1e3 for v in start],
                                   "stop_mm": [v * 1e3 for v in stop],
                                   "kw": {k: v for k, v in kw.items()}})


class _StubMesh:
    """Records what the builder puts on the grid.

    ``SmoothMeshLines`` is answered with ``_openems_tutorial_gate._smooth_estimate``
    -- the maker's own pure-numpy stand-in for the CSXCAD call, whose line count
    is a LOWER bound.  Nothing here is a realized mesh; the real run records
    that from the solver.
    """

    def __init__(self, rec, smooth_estimate):
        self._rec, self._smooth = rec, smooth_estimate

    def SetDeltaUnit(self, unit):
        self._rec["delta_unit"] = unit

    def AddLine(self, axis, values):
        self._rec["added"][axis].extend(
            [float(v) for v in np.atleast_1d(np.asarray(values, dtype=float))])

    def SmoothMeshLines(self, axis, res, **kw):
        self._rec["smooth_res"][axis] = float(res)
        self._rec["lines"][axis] = np.asarray(
            self._smooth(self._rec["added"][axis], res), dtype=float)

    def GetLines(self, axis):
        return self._rec["lines"][axis]


class _StubCSX:
    def __init__(self, rec, smooth_estimate):
        self._rec = rec
        self._mesh = _StubMesh(rec, smooth_estimate)

    def GetGrid(self):
        return self._mesh

    def AddMaterial(self, name, **kw):
        self._rec["materials"].append({"name": name, "kw": kw})
        return _StubPrimitive(self._rec, name)

    def AddMetal(self, name):
        self._rec["metals"].append(name)
        return _StubPrimitive(self._rec, name)


class _StubFDTD:
    def __init__(self, rec, **kw):
        self._rec = rec
        rec["fdtd_kwargs"] = {k: v for k, v in kw.items()}

    def SetGaussExcite(self, f0, fc):
        self._rec["gauss_excite_hz"] = [float(f0), float(fc)]

    def SetCSX(self, csx):
        self._rec["csx_set"] = True
        self._csx = csx

    def GetCSX(self):
        return self._csx

    def SetBoundaryCond(self, bc):
        self._rec["boundary_calls"].append(list(bc))

    def AddMSLPort(self, number, metal, start, stop, prop_dir, exc_dir, **kw):
        self._rec["ports"].append({
            "number": number, "start_mm": [v * 1e3 for v in start],
            "stop_mm": [v * 1e3 for v in stop], "prop": prop_dir,
            "exc": exc_dir, "kw": {k: (float(v) if isinstance(v, (int, float))
                                       else v) for k, v in kw.items()}})
        return object()


def _stub_record(smooth_estimate):
    """One recording stub set.

    Both stand-ins are CLASSES, not factory functions, so ``_boundary_class``
    subclasses the stub exactly as it subclasses the real ``openEMS`` -- the
    dry run then exercises the same substitution path the solver run takes,
    instead of silently falling back.
    """
    rec = {"boundary_calls": [], "added": {"x": [], "y": [], "z": []},
           "lines": {"x": None, "y": None, "z": None},
           "smooth_res": {}, "boxes": [], "materials": [], "metals": [],
           "ports": [], "fdtd_kwargs": {}, "csx_set": False,
           "gauss_excite_hz": None, "delta_unit": None}

    class _CSXStub(_StubCSX):
        def __init__(self):
            super().__init__(rec, smooth_estimate)

    class _FDTDStub(_StubFDTD):
        def __init__(self, **kw):
            super().__init__(rec, **kw)

    return rec, _CSXStub, _FDTDStub


# ---------------------------------------------------------------- the dry run
def dry_run(maker, configs, *, mutate_skip_for: str | None,
            real_nrts: int, real_end_criteria: float,
            resolution_factor: float = RESOLUTION_FACTOR,
            stop_note: str | None = None, common: bool = True,
            collect: list | None = None) -> int:
    """The per-config stub build and its printout.

    With the defaults this is the dry run as it was, byte for byte.  The
    ladder calls it once per factor: ``common=False`` drops the lines that do
    not depend on the factor after the first, ``collect`` receives each
    config's mesh estimate for the cost table.
    """
    if common:
        print("=" * 78)
        print("The Sheen low-pass filter -- openEMS box probe, DRY RUN (no solver)")
        print("=" * 78)
    if resolution_factor == RESOLUTION_FACTOR and stop_note is None:
        print(f"  rung             resolution_factor {RESOLUTION_FACTOR:g} = "
              f"{maker.resolution_m(RESOLUTION_FACTOR) * 1e6:.2f} um, the record's "
              f"coarse rung")
    else:
        print(f"  rung             resolution_factor {resolution_factor:.6g} = "
              f"{maker.resolution_m(resolution_factor) * 1e6:.2f} um in-plane "
              f"smoothing target, {maker.substrate_z_cells(resolution_factor)} "
              f"substrate z cells")
    if stop_note is None:
        print(f"  stop criteria    --real-nrts {real_nrts} --real-end-criteria "
              f"{real_end_criteria:g} --accept-truncation, the way the frozen record "
              f"was made")
    elif common:
        print(f"  stop criteria    {stop_note}")
    if common:
        print(f"  STAGE A          NOT RUN here. The reproduce gate (openEMS's own "
              f"MSL_NotchFilter tutorial, verbatim) passed in the three jobs that "
              f"made the record: VESSL {', '.join(STAGE_A_REFERENCE_JOBS)}")
        print(f"  builder literal  {maker.B_BOUNDARY}  "
              f"(make_openems_reference.py, inside _build_sheen_board_at_rung)")
        print(f"  x layout         untouched by every config: LX "
              f"{maker.LX * 1e3:.3f} mm, section x {maker.PATCH_X0 * 1e3:.3f}-"
              f"{maker.PATCH_X1 * 1e3:.3f} mm")
        audit = _audit_definition_time_capture(maker)
        print("\n  definition-time capture audit "
              "(does anything freeze a patched constant at def time?):")
        print(f"    defaults carrying a patched value: "
              f"{audit['defaults_carrying_a_patched_value'] or 'none'}")
        for fname, reads in audit["reads"].items():
            print(f"    {fname:28s} reads {reads if reads else '(none directly)'}")
    if mutate_skip_for:
        print(f"\n  MUTATION: the boundary wrapper is REMOVED for config "
              f"{mutate_skip_for}. Its SetBoundaryCond call below should then be "
              f"the builder's own {maker.B_BOUNDARY}, which is what shows the "
              f"printed boundary is produced by the wrapper.")
    print()

    control = snapshot(maker)
    original_builder = maker._build_sheen_board_at_rung
    rows = []
    try:
        for cfg in configs:
            values = config_globals(maker, cfg)
            apply_globals(maker, values)
            skip = mutate_skip_for is not None and cfg["id"] == mutate_skip_for
            builder, how = make_builder(maker, cfg["boundary"], mutate_skip=skip)
            rec, csx_stub, fdtd_stub = _stub_record(maker._smooth_estimate)
            builder(csx_stub, fdtd_stub, None,
                    nrts=real_nrts, end_criteria=real_end_criteria,
                    resolution_factor=resolution_factor)
            used = list(getattr(builder, "mechanism_used", []) or []) or ["none"]
            y = rec["lines"]["y"]
            effective = rec["boundary_calls"][-1] if rec["boundary_calls"] else None
            declared_lo = values["PATCH_Y_LO"]
            declared_hi = values["PATCH_Y_HI"]
            pml_cells = 8 if effective and effective[2].startswith("PML_") else 0
            inner_lo = float(y[pml_cells]) if y is not None and y.size > pml_cells else None
            inner_hi = (float(y[-1 - pml_cells])
                        if y is not None and y.size > pml_cells else None)
            rows.append({"cfg": cfg, "values": values, "rec": rec,
                         "effective": effective, "how": how, "skipped": skip,
                         "y_lines": None if y is None else int(y.size),
                         "inner": (inner_lo, inner_hi)})
            print(f"  {cfg['id']}  {cfg['why']}{'   <-- WRAPPER REMOVED (mutation)' if skip else ''}")
            print(f"      Y_CLEAR {values['Y_CLEAR'] * 1e3:6.3f} mm; box "
                  f"{maker.LX * 1e3:.3f} x {values['LY'] * 1e3:.3f} x "
                  f"{maker.LZ * 1e3:.3f} mm")
            print(f"      feeds y {values['IN_FEED_YC'] * 1e3:.4f} / "
                  f"{values['OUT_FEED_YC'] * 1e3:.4f} mm; section y "
                  f"{declared_lo * 1e3:.4f}-{declared_hi * 1e3:.4f} mm")
            print(f"      override wrapper: {how}")
            print(f"      substitution path taken: {used}")
            print(f"      SetBoundaryCond call(s) the solver would see: "
                  f"{rec['boundary_calls']}")
            print(f"      effective y faces: {effective[2]} / {effective[3]}"
                  if effective else "      effective y faces: (none recorded)")
            if y is not None:
                print(f"      mesh estimate: x {rec['lines']['x'].size} lines, "
                      f"y {y.size} lines, z {rec['lines']['z'].size} lines; "
                      f"y step {np.min(np.diff(y)) * 1e6:.2f}-"
                      f"{np.max(np.diff(y)) * 1e6:.2f} um")
                if collect is not None:
                    est_cells = ((rec['lines']['x'].size - 1) * (y.size - 1)
                                 * (rec['lines']['z'].size - 1))
                    collect.append({"id": cfg["id"], "factor": resolution_factor,
                                    "cells_lower_bound": int(est_cells),
                                    "lines": [int(rec['lines']['x'].size), int(y.size),
                                              int(rec['lines']['z'].size)]})
                    print(f"      cells estimate {est_cells} (numpy lower bound)")
                if pml_cells:
                    print(f"      PML_8 on y eats the outer {pml_cells} cells of "
                          f"each y face: its inner faces sit at "
                          f"{inner_lo * 1e3:.4f} and {inner_hi * 1e3:.4f} mm, "
                          f"{(declared_lo - inner_lo) * 1e3:.4f} / "
                          f"{(inner_hi - declared_hi) * 1e3:.4f} mm from the "
                          f"section's declared edges")
                else:
                    print(f"      MUR on y is a wall on the mesh line itself: it "
                          f"eats no cell, so the section's declared edges sit "
                          f"{declared_lo * 1e3:.4f} / "
                          f"{(values['LY'] - declared_hi) * 1e3:.4f} mm from it")
            print(f"      ports: {[(p['number'], round(p['start_mm'][1], 4)) for p in rec['ports']]}"
                  f" (number, y start mm)")
            # Why no cell snap here, checked rather than assumed: this mesh is
            # not a uniform lattice. The builder puts an explicit line on each
            # declared feed edge (plus the thirds-rule pair), so those anchors
            # move rigidly with Y_CLEAR and SmoothMeshLines only subdivides the
            # gaps between them. The realized feed width below is what shows it
            # -- if it moved with the clearance, the boxes would not be the same
            # board.
            if y is not None:
                edges = maker._feed_edges_against_lines(y * 1e6)
                for name, e in edges.items():
                    print(f"      {name} realized width between nearest lines: "
                          f"{e['width_between_nearest_lines_um']:.2f} um "
                          f"(declared {e['declared_width_um']:.1f}, "
                          f"{e['width_error_pct']:+.3f} %)")
            print()
    finally:
        apply_globals(maker, control)
        maker._build_sheen_board_at_rung = original_builder

    print("  y faces side by side (what the mutation check reads):")
    for row in rows:
        eff = row["effective"]
        print(f"    {row['cfg']['id']}  Y_CLEAR "
              f"{row['values']['Y_CLEAR'] * 1e3:6.3f} mm  y faces "
              f"{eff[2]} / {eff[3]}  y lines {row['y_lines']}")
    if common:
        print("\n  WHAT THIS DRY RUN CANNOT TELL YOU: the mesh line counts are the "
              "maker's own pure-numpy lower bound; CSXCAD's SmoothMeshLines grades "
              "fine-to-coarse transitions and adds lines this estimate does not "
              "model. The realized mesh, the port snaps and the energy trace come "
              "from the run itself.")
    return 0


# ------------------------------------------------------------ the mesh ladder
def _ftag(factor: float) -> str:
    """A path-safe name for a factor: 1 -> rf1, 0.5 -> rf0p5."""
    return "rf" + f"{factor:.6g}".replace(".", "p")


def _stop_note(real_nrts: int, real_end_criteria: float, refuse: bool) -> str:
    return (f"--real-nrts {real_nrts} --real-end-criteria {real_end_criteria:g}, "
            + ("truncation REFUSED: a pass that reaches the step ceiling fires "
               "the shared runner's gate and leaves no number" if refuse else
               "--accept-truncation (a pass that reaches the ceiling is recorded "
               "as truncated)"))


def _frozen_rungs() -> dict:
    """factor -> (realized cells, dt) from the frozen record's own meta."""
    path = (repo_root() / "tests" / "crossval" / "sheen_lpf" / "reference"
            / "openems_sheen.json")
    out = {}
    if not path.is_file():
        return out
    with path.open() as fh:
        meta = json.load(fh).get("meta", {}).get("stages", {})
    for name in ("stage_b_coarse", "stage_b_mid", "stage_b_fine"):
        st = meta.get(name) or {}
        f = st.get("resolution_factor")
        cells = (st.get("mesh_realized") or {}).get("n_cells")
        if f is not None and cells is not None and st.get("dt_s") is not None:
            out[float(f)] = {"stage": name, "cells": int(cells), "dt_s": float(st["dt_s"])}
    return out


def cost_rows(maker, factors, lower_bounds: dict) -> list:
    """Per factor: cells, dt, steps to ring down, wall time on 8 threads.

    Cells and dt are the frozen record's REALIZED values where it has that
    factor.  Otherwise cells are this plan's numpy mesh estimate (a lower
    bound: CSXCAD's smoothing adds lines) times the realized-over-estimate
    ratio the frozen record's finest rung shows on the same box, and dt scales
    as the factor from B2's own dt at factor 1.  Steps: the record time B2
    took to reach its 1e-4 criterion at factor 1 (1.384 ns), and three times
    that.  Rates: B2's own 417000 x 8251 / 18.6 s, and the frozen fine rung's
    3.62e8 cell-steps/s.
    """
    frozen = _frozen_rungs()
    rate_b2 = COST_B2_F1["cells"] * COST_B2_F1["steps"] / COST_B2_F1["wall_s"]
    dt_f1 = COST_B2_F1["record_s"] / COST_B2_F1["steps"]
    ratio, ratio_note = None, ""
    if frozen:
        finest = min(frozen)
        lb_finest = int(maker._plan("cost", finest)["cells_estimate"])
        ratio = frozen[finest]["cells"] / lb_finest
        ratio_note = (f"x {ratio:.4f} = realized/estimate at factor {finest:.6g} "
                      f"({frozen[finest]['cells']}/{lb_finest})")
    rows = []
    for f in factors:
        hit = next((v for k, v in frozen.items() if abs(k - f) < 1e-9), None)
        lb = lower_bounds.get(f)
        if hit is not None:
            cells, dt, src = hit["cells"], hit["dt_s"], f"realized, {hit['stage']}"
        elif lb is not None and ratio is not None:
            cells = int(round(lb * ratio))
            dt = dt_f1 * f
            src = f"estimate {ratio_note}; dt = B2 dt x factor"
        else:
            cells = int(round(COST_B2_F1["cells"] / f ** 3))
            dt = dt_f1 * f
            src = "B2 cells x factor^-3; dt = B2 dt x factor"
        for m in COST_DECAY_MULTIPLES:
            steps = int(np.ceil(m * COST_B2_F1["record_s"] / dt))
            for rname, rate in (("B2 f=1", rate_b2), ("fine record", COST_FINE_RATE)):
                rows.append({"factor": f, "cells": cells, "cells_source": src,
                             "cells_lower_bound": lb, "dt_s": dt,
                             "decay_multiple": m, "steps": steps, "rate_name": rname,
                             "rate": rate, "wall_s": cells * steps / rate})
    return rows


def dry_run_ladder(maker, configs, factors, *, mutate_skip_for, real_nrts,
                   real_end_criteria, refuse_truncation) -> int:
    note = _stop_note(real_nrts, real_end_criteria, refuse_truncation)
    collect: list = []
    print("=" * 78)
    print("The Sheen low-pass filter -- openEMS box probe, MESH LADDER, DRY RUN "
          "(no solver)")
    print("=" * 78)
    print(f"  plan             configs {[c['id'] for c in configs]} x factors "
          f"{[float(f) for f in factors]} = {len(configs) * len(factors)} passes, "
          f"config by config, factors in the order given")
    for f in factors:
        print(f"    factor {f:.10g}: in-plane smoothing target "
              f"{maker.resolution_m(f) * 1e6:.2f} um, "
              f"{maker.substrate_z_cells(f)} substrate z cells, pass label "
              f"box_probe_<id>_{_ftag(f)}")
    ctrl = LADDER_CONTROL
    carried = (any(c["id"] == ctrl["id"] for c in configs)
               and any(abs(f - ctrl["factor"]) < 1e-12 for f in factors))
    print(f"  control          {ctrl['id']} at factor {ctrl['factor']:g} must reproduce "
          f"{ctrl['zero_8_ghz']:.5f} GHz (VESSL {ctrl['vessl']}) within "
          f"{100 * ctrl['tol']:.2f} %: "
          + ("CARRIED by this plan" if carried else "NOT carried by this plan "
             "(no factor 1.0 for that config); this job's numbers are read "
             "against a job that carries it"))
    for i, f in enumerate(factors):
        print(f"\n--- factor {f:.10g} " + "-" * 50)
        dry_run(maker, configs, mutate_skip_for=mutate_skip_for,
                real_nrts=real_nrts, real_end_criteria=real_end_criteria,
                resolution_factor=f, stop_note=note, common=(i == 0),
                collect=collect)
    lower = {}
    for c in collect:
        lower[c["factor"]] = max(lower.get(c["factor"], 0), c["cells_lower_bound"])
    print("\n" + "=" * 78)
    print("COST ESTIMATE per factor (8 threads; every config in the plan costs this "
          "per pass)")
    print("=" * 78)
    rows = cost_rows(maker, factors, lower)
    seen = []
    for r in rows:
        if r["factor"] not in seen:
            seen.append(r["factor"])
            print(f"  factor {r['factor']:.6g}: {r['cells']} cells ({r['cells_source']}); "
                  f"numpy lower bound for this plan {r['cells_lower_bound']}; "
                  f"dt {r['dt_s']:.4e} s")
    print("  factor      cells        decay   steps    rate                     wall")
    for r in rows:
        print(f"  {r['factor']:<10.6g}  {r['cells']:>10d}  {r['decay_multiple']:3.0f}x  "
              f"{r['steps']:7d}  {r['rate_name']:11s} {r['rate']:.2e}  "
              f"{r['wall_s']:8.0f} s ({r['wall_s'] / 3600:.2f} h)")
    worst = max(r["wall_s"] for r in rows) if rows else 0.0
    per_cfg = {}
    for r in rows:
        if r["decay_multiple"] == max(COST_DECAY_MULTIPLES) and r["rate_name"] == "B2 f=1":
            per_cfg[r["factor"]] = r["wall_s"]
    total = len(configs) * sum(per_cfg.values())
    print(f"  plan total at {max(COST_DECAY_MULTIPLES):g}x decay and the B2 rate: "
          f"{total:.0f} s ({total / 3600:.2f} h) for {len(configs)} config(s); "
          f"worst single pass in the table {worst:.0f} s ({worst / 3600:.2f} h)")
    print("  WHAT THIS CANNOT TELL YOU: the ring-down time at each factor (the "
          "rows assume B2's 1.384 ns and three times it), the realized cells "
          "beyond the frozen record's finest rung (estimated, not realized), "
          "the throughput at those sizes, and peak memory.")
    return 0


# ------------------------------------------------------------------- the solve
def _band_mask(freqs_ghz) -> np.ndarray:
    f = np.asarray(freqs_ghz, dtype=float)
    return (f >= BAND_GHZ[0]) & (f <= BAND_GHZ[1])


def _db(mag) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(mag, dtype=float), 1e-300))


def delta_against(record: dict, ref: dict) -> dict:
    """Max |S21| difference in dB over the band where both curves are above the
    deep-null level -- the case's own two-sided exclusion, on the record's own
    shared frequency grid (both configs use ``linspace(0.5, 20 GHz, 801)``)."""
    f = np.asarray(record["freqs_ghz"], dtype=float)
    ours = _db(record["s21_mag"])
    theirs = _db(ref["s21_mag"])
    keep = _band_mask(f) & (ours >= DEEP_NULL_DB) & (theirs >= DEEP_NULL_DB)
    delta = ours - theirs
    if not keep.any():
        return {"max_abs_delta_db": float("nan"), "worst_f_ghz": None,
                "n_compared": 0, "n_in_band": int(_band_mask(f).sum())}
    worst = int(np.argmax(np.where(keep, np.abs(delta), -np.inf)))
    return {"max_abs_delta_db": float(np.abs(delta[keep]).max()),
            "worst_f_ghz": float(f[worst]), "n_compared": int(keep.sum()),
            "n_in_band": int(_band_mask(f).sum())}


def zeros_of(sf, freqs_ghz, s21_mag) -> dict:
    """Both zeros with the repository's estimator, each in its own window."""
    f = np.asarray(freqs_ghz, dtype=float)
    s = np.abs(np.asarray(s21_mag, dtype=float))
    out = {}
    for key, (lo, hi) in ZERO_BANDS_GHZ.items():
        try:
            e = sf.refined_extremum(f, s, lo, hi, transform="log")
            band = np.flatnonzero((f >= lo) & (f <= hi))
            out[key] = {"f_ghz": float(e["refined_f"]), "bin_f_ghz": float(e["bin_f"]),
                        "depth_db": float(e["depth_db"]),
                        "at_window_edge": bool(band.size and int(e["index"])
                                               in (int(band[0]), int(band[-1])))}
        except Exception as exc:
            out[key] = {"error": repr(exc)}
    return out


def run_config(maker, cfg: dict, *, sim_root: str, threads: int,
               real_nrts: int, real_end_criteria: float, sf,
               resolution_factor: float = RESOLUTION_FACTOR,
               ladder: bool = False, accept_truncation: bool = True) -> dict:
    values = config_globals(maker, cfg)
    apply_globals(maker, values)
    original_builder = maker._build_sheen_board_at_rung
    builder, how = make_builder(maker, cfg["boundary"])
    maker._build_sheen_board_at_rung = builder

    print(f"\n{'=' * 78}\n=== config {cfg['id']}: Y_CLEAR "
          f"{cfg['y_clear'] * 1e3:g} mm, y faces {cfg['boundary'][2]} / "
          f"{cfg['boundary'][3]} -- {cfg['why']}\n{'=' * 78}")
    if ladder:
        print(f"  resolution factor {resolution_factor:.10g}: in-plane smoothing "
              f"target {maker.resolution_m(resolution_factor) * 1e6:.2f} um, "
              f"{maker.substrate_z_cells(resolution_factor)} substrate z cells; "
              f"truncation {'accepted' if accept_truncation else 'REFUSED'}")
    print(f"  declared box {maker.LX * 1e3:.3f} x {values['LY'] * 1e3:.3f} x "
          f"{maker.LZ * 1e3:.3f} mm; section y {values['PATCH_Y_LO'] * 1e3:.4f}-"
          f"{values['PATCH_Y_HI'] * 1e3:.4f} mm; feeds y "
          f"{values['IN_FEED_YC'] * 1e3:.4f} / {values['OUT_FEED_YC'] * 1e3:.4f} mm")
    print(f"  boundary this config asks for: {cfg['boundary']}")
    print(f"  override mechanism: {how}", flush=True)

    t0 = time.time()
    try:
        record, meta = maker._run_stage_b(
            label=(f"box_probe_{cfg['id']}_{_ftag(resolution_factor)}" if ladder
                   else f"box_probe_{cfg['id']}"),
            sim_root=sim_root, threads=threads,
            resolution_factor=resolution_factor, sf=sf,
            real_nrts=real_nrts, real_end_criteria=real_end_criteria,
            accept_truncation=accept_truncation)
    finally:
        maker._build_sheen_board_at_rung = original_builder
    wall = time.time() - t0

    mechanism_used = list(getattr(builder, "mechanism_used", []) or [])
    mesh = meta.get("mesh_realized") or {}
    for name, e in (meta.get("feed_edges_realized") or {}).items():
        print(f"  {name} realized width between the nearest mesh lines: "
              f"{e.get('width_between_nearest_lines_um')} um (declared "
              f"{e.get('declared_width_um')}, {e.get('width_error_pct')} %)")
    print(f"  realized mesh: x {mesh.get('x', {}).get('n_lines')} lines, "
          f"y {mesh.get('y', {}).get('n_lines')} lines, "
          f"z {mesh.get('z', {}).get('n_lines')} lines; "
          f"{mesh.get('substrate_z_cells_realized')} substrate cells, "
          f"{mesh.get('n_cells')} cells")
    print(f"  box energy at the end of the record: "
          f"{meta.get('final_energy_db')} dB at timestep "
          f"{meta.get('final_timestep')}; record length "
          f"{meta.get('record_length_steps')} steps = "
          f"{meta.get('record_length_s')} s")
    null = record.get("null", {})
    cut = record.get("cutoff_3db", {})
    pb = record.get("passband", {})
    print(f"  stopband null {null.get('refined_f_ghz')} GHz "
          f"(bin {null.get('bin_f_ghz')}), depth {null.get('depth_db')} dB")
    print(f"  passband mean {pb.get('mean_db')} dB; -3 dB corner "
          f"{cut.get('f_ghz')} GHz")
    print(f"  energy sum over {BAND_GHZ[0]:.0f}-{BAND_GHZ[1]:.0f} GHz: "
          f"{record.get('min_energy_sum_band')} - "
          f"{record.get('max_energy_sum_band')}")
    print(f"  wall time {wall:.1f} s (solver's own {meta.get('wall_time_s')} s); "
          f"mechanism used: {mechanism_used or 'none'}", flush=True)

    out = {"id": cfg["id"], "why": cfg["why"],
           "y_clear_mm": cfg["y_clear"] * 1e3, "boundary": list(cfg["boundary"]),
           "override_mechanism": how, "mechanism_used": mechanism_used,
           "wall_s": wall, "record": record, "meta": meta}
    if ladder:
        zeros = zeros_of(sf, record["freqs_ghz"], record["s21_mag"])
        z7, z8 = zeros.get("zero_7", {}), zeros.get("zero_8", {})
        print(f"  zero7 {z7.get('f_ghz')} GHz (depth {z7.get('depth_db')} dB"
              f"{', AT THE WINDOW EDGE' if z7.get('at_window_edge') else ''}); "
              f"zero8 {z8.get('f_ghz')} GHz (depth {z8.get('depth_db')} dB"
              f"{', AT THE WINDOW EDGE' if z8.get('at_window_edge') else ''}); "
              f"end criterion reached: {meta.get('end_criteria_reached')}",
              flush=True)
        out.update({"resolution_factor": float(resolution_factor),
                    "resolution_um": float(maker.resolution_m(resolution_factor) * 1e6),
                    "zeros": zeros})
    return out


def write_figure(results, frozen, directory: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sheen_box_probe_openems.png"
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for res in results:
        rec = res["record"]
        ax.plot(rec["freqs_ghz"], _db(rec["s21_mag"]), lw=1.4,
                label=f"{res['id']}  y_clear {res['y_clear_mm']:g} mm, "
                      f"y faces {res['boundary'][2]}")
    if frozen is not None:
        ax.plot(frozen["freqs_ghz"], _db(frozen["s21_mag"]), "k-", lw=1.0,
                label="frozen record stage_b_coarse")
    ax.set_xlim(*BAND_GHZ)
    ax.set_ylim(-70, 5)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("|S21| (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------- the ladder's run
def _pct(a, b):
    return None if a is None or b is None else 100.0 * (a - b) / b


def _fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def run_ladder(maker, configs, factors, args, sf, out_dir: Path) -> int:
    print("=" * 78)
    print("The Sheen low-pass filter -- openEMS box probe, MESH LADDER")
    print("=" * 78)
    print(f"  STAGE A IS NOT RUN. The reproduce gate passed in VESSL "
          f"{', '.join(STAGE_A_REFERENCE_JOBS)}, which produced the frozen record.")
    print(f"  openEMS build: {os.environ.get('RFX_OPENEMS_COMMIT', '(unstamped)')}; "
          f"image {os.environ.get('RFX_OPENEMS_IMAGE', '(unset)')}")
    print(f"  configs {[c['id'] for c in configs]} x factors "
          f"{[float(f) for f in factors]}; "
          f"{_stop_note(args.real_nrts, args.real_end_criteria, args.refuse_truncation)}")
    audit = _audit_definition_time_capture(maker)
    print(f"  definition-time capture audit: defaults carrying a patched value = "
          f"{audit['defaults_carrying_a_patched_value'] or 'none'}")
    if audit["defaults_carrying_a_patched_value"]:
        print("ERROR: a patched constant is frozen in a default argument.",
              file=sys.stderr)
        return 3

    control = snapshot(maker)
    results, failures = [], {}
    rc = 0
    for cfg in configs:
        for f in factors:
            try:
                results.append(run_config(
                    maker, cfg, sim_root=args.sim_root, threads=args.threads,
                    real_nrts=args.real_nrts,
                    real_end_criteria=args.real_end_criteria, sf=sf,
                    resolution_factor=f, ladder=True,
                    accept_truncation=not args.refuse_truncation))
            except Exception as exc:
                print(f"CONFIG {cfg['id']} FACTOR {f:.10g} FAILED: {exc}",
                      file=sys.stderr)
                failures[f"{cfg['id']}@{f:.10g}"] = str(exc)
                rc = 1
            finally:
                apply_globals(maker, control)

    def z(r, key):
        return (r.get("zeros") or {}).get(key, {}).get("f_ghz")

    print("\n" + "=" * 78)
    print("READING 1 -- per (config, factor): the mesh, the end of the record, the cost")
    print("=" * 78)
    print("  id  factor      res (um)  x/y/z lines      sub   cells       end (dB)  "
          "steps    end crit  wall (s)")
    for r in results:
        mesh = r["meta"].get("mesh_realized") or {}
        lines = "/".join(str((mesh.get(a) or {}).get("n_lines")) for a in "xyz")
        print(f"  {r['id']}  {r['resolution_factor']:<10.6g}  {r['resolution_um']:8.2f}  "
              f"{lines:15s}  {str(mesh.get('substrate_z_cells_realized')):>3s}  "
              f"{str(mesh.get('n_cells')):>10s}  "
              f"{str(r['meta'].get('final_energy_db')):>8s}  "
              f"{str(r['meta'].get('final_timestep')):>6s}  "
              f"{str(r['meta'].get('end_criteria_reached')):>8s}  {r['wall_s']:8.1f}")

    print("\n" + "=" * 78)
    print("READING 2 -- per (config, factor): the features")
    print("=" * 78)
    print("  id  factor      zero7 (GHz)  zero8 (GHz)  deepest 5-10  -3 dB (GHz)  "
          "passband (dB)")
    for r in results:
        rec = r["record"]
        cut, pb = rec.get("cutoff_3db", {}), rec.get("passband", {})
        print(f"  {r['id']}  {r['resolution_factor']:<10.6g}  {_fmt(z(r, 'zero_7'), '11.5f')}  "
              f"{_fmt(z(r, 'zero_8'), '11.5f')}  "
              f"{_fmt((rec.get('null') or {}).get('refined_f_ghz'), '12.5f')}  "
              f"{_fmt(cut.get('f_ghz'), '11.5f')}  {_fmt(pb.get('mean_db'), '13.3f')}")

    print("\n" + "=" * 78)
    ctrl = LADDER_CONTROL
    print(f"CONTROL -- {ctrl['id']} at factor {ctrl['factor']:g} against VESSL "
          f"{ctrl['vessl']}")
    print("=" * 78)
    hit = next((r for r in results if r["id"] == ctrl["id"]
                and abs(r["resolution_factor"] - ctrl["factor"]) < 1e-12), None)
    control_out = {"carried": hit is not None}
    if hit is None:
        print(f"  NOT CARRIED by this job: {ctrl['id']} at factor {ctrl['factor']:g} "
              f"was not solved here.")
    else:
        got = z(hit, "zero_8")
        rel = None if got is None else abs(got - ctrl["zero_8_ghz"]) / ctrl["zero_8_ghz"]
        ok = rel is not None and rel <= ctrl["tol"]
        control_out.update({"zero_8_ghz": got, "reference_ghz": ctrl["zero_8_ghz"],
                            "rel": rel, "pass": ok})
        print(f"  zero8 {_fmt(got, '.6f')} GHz vs {ctrl['zero_8_ghz']:.6f} GHz: "
              f"{_fmt(None if rel is None else 100 * rel, '.5f')} % -> "
              f"{'PASS' if ok else 'FAIL'} (tolerance {100 * ctrl['tol']:.2f} %)")
        if not ok:
            rc = rc or 1

    print("\n" + "=" * 78)
    print("LADDER -- per config, coarse to fine, change from the previous factor")
    print("=" * 78)
    ladder_table = {}
    for cfg in configs:
        rows = sorted((r for r in results if r["id"] == cfg["id"]),
                      key=lambda r: -r["resolution_factor"])
        if not rows:
            continue
        print(f"  {cfg['id']} ({cfg['why']}):")
        print("    factor      res (um)  cells       zero7      d (%)     zero8      "
              "d (%)     corner     d (%)")
        prev, table = None, []
        for r in rows:
            c = (r["record"].get("cutoff_3db") or {}).get("f_ghz")
            row = {"factor": r["resolution_factor"], "resolution_um": r["resolution_um"],
                   "cells": (r["meta"].get("mesh_realized") or {}).get("n_cells"),
                   "zero_7": z(r, "zero_7"), "zero_8": z(r, "zero_8"), "corner": c,
                   "d_zero_7_pct": None if prev is None else _pct(z(r, "zero_7"), prev["zero_7"]),
                   "d_zero_8_pct": None if prev is None else _pct(z(r, "zero_8"), prev["zero_8"]),
                   "d_corner_pct": None if prev is None else _pct(c, prev["corner"])}
            table.append(row)
            print(f"    {row['factor']:<10.6g}  {row['resolution_um']:8.2f}  "
                  f"{str(row['cells']):>10s}  {_fmt(row['zero_7'], '9.5f')}  "
                  f"{_fmt(row['d_zero_7_pct'], '+8.3f'):>8s}  {_fmt(row['zero_8'], '9.5f')}  "
                  f"{_fmt(row['d_zero_8_pct'], '+8.3f'):>8s}  {_fmt(row['corner'], '9.5f')}  "
                  f"{_fmt(row['d_corner_pct'], '+8.3f'):>8s}")
            prev = row
        ladder_table[cfg["id"]] = table

    frozen_all = None
    ref_path = (repo_root() / "tests" / "crossval" / "sheen_lpf" / "reference"
                / "openems_sheen.json")
    if ref_path.is_file():
        with ref_path.open() as fh:
            frozen_all = json.load(fh)
    frozen_ladder = []
    if frozen_all is not None:
        print("\n  for reference, the frozen record's own ladder (the B0 box: MUR y "
              "faces, a declared 10 ns record), same estimator:")
        prev = None
        for name in ("stage_b_coarse", "stage_b_mid", "stage_b_fine"):
            st = frozen_all.get(name)
            if not st:
                continue
            zz = zeros_of(sf, st["freqs_ghz"], st["s21_mag"])
            row = {"stage": name,
                   "factor": frozen_all["meta"]["stages"][name].get("resolution_factor"),
                   "zero_7": zz.get("zero_7", {}).get("f_ghz"),
                   "zero_8": zz.get("zero_8", {}).get("f_ghz"),
                   "corner": st["cutoff_3db"]["f_ghz"]}
            d8 = None if prev is None else _pct(row["zero_8"], prev["zero_8"])
            dc = None if prev is None else _pct(row["corner"], prev["corner"])
            print(f"    {name:15s} factor {row['factor']:<8.6g} zero7 "
                  f"{_fmt(row['zero_7'], '.5f')}  zero8 {_fmt(row['zero_8'], '.5f')} "
                  f"({_fmt(d8, '+.3f')} %)  corner {_fmt(row['corner'], '.5f')} "
                  f"({_fmt(dc, '+.3f')} %)")
            frozen_ladder.append(row)
            prev = row

    fig_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / "sheen_box_probe_openems_ladder.png"
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for r in results:
            ax.plot(r["record"]["freqs_ghz"], _db(r["record"]["s21_mag"]), lw=1.3,
                    label=f"{r['id']} factor {r['resolution_factor']:.4g} "
                          f"({r['resolution_um']:.1f} um)")
        if frozen_all is not None and "stage_b_fine" in frozen_all:
            st = frozen_all["stage_b_fine"]
            ax.plot(st["freqs_ghz"], _db(st["s21_mag"]), "k-", lw=1.0,
                    label="frozen record stage_b_fine (B0, 99.25 um)")
        ax.set_xlim(*BAND_GHZ)
        ax.set_ylim(-70, 5)
        ax.set_xlabel("frequency (GHz)")
        ax.set_ylabel("|S21| (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=130)
        plt.close(fig)
        print(f"\n  figure: {fig_path}")
    except Exception as exc:
        fig_path = None
        print(f"\n  figure NOT written: {exc!r}")

    payload = {
        "what": "openEMS box probe, mesh ladder, for the Sheen low-pass filter -- "
                "a diagnostic, not a reference record. Stage A is not run; it "
                "passed in VESSL " + ", ".join(STAGE_A_REFERENCE_JOBS),
        "resolution_factors": [float(f) for f in factors],
        "real_nrts": args.real_nrts, "real_end_criteria": args.real_end_criteria,
        "accept_truncation": not args.refuse_truncation,
        "definition_time_capture_audit": audit,
        "openems_commit": os.environ.get("RFX_OPENEMS_COMMIT"),
        "openems_image": os.environ.get("RFX_OPENEMS_IMAGE"),
        "control": control_out, "ladder": ladder_table,
        "frozen_record_ladder": frozen_ladder, "failures": failures,
        "configs": results,
        "figure": None if fig_path is None else str(fig_path),
    }
    json_path = out_dir / "sheen_box_probe_openems_ladder.json"
    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print(f"  json:   {json_path}")
    return rc


# ------------------------------------------------------------------- the main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--configs", default=",".join(c["id"] for c in CONFIGS),
                   help="comma separated subset of "
                        f"{','.join(c['id'] for c in CONFIGS)}")
    p.add_argument("--out", default=None, help="directory for the JSON and figure")
    p.add_argument("--sim-root", default="/tmp/sheen_box_probe_openems")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--real-nrts", type=int, default=DEFAULT_NRTS,
                   help="the record's own declared length, 60000 steps")
    p.add_argument("--real-end-criteria", type=float, default=DEFAULT_END_CRITERIA,
                   help="the record's own 1e-4")
    p.add_argument("--resolution-factors", default="1.0", metavar="LIST",
                   help="comma separated mesh factors on the maker's own "
                        "resolution (1.0 = 198.5 um, the record's coarse rung); "
                        "every config is solved at every factor. Default 1.0, "
                        "the probe as it was")
    p.add_argument("--refuse-truncation", action="store_true",
                   help="do not accept a pass that reaches --real-nrts: the "
                        "shared runner's gate fires instead. Default off, as "
                        "the box probe ran")
    p.add_argument("--dry-run", action="store_true",
                   help="build every config against stub openEMS/CSXCAD classes "
                        "and print the boundary and mesh it would set; runs "
                        "without openEMS installed")
    p.add_argument("--mutate-skip-boundary-override", default=None, metavar="ID",
                   help="DRY RUN ONLY. Remove the boundary wrapper for this "
                        "config so its SetBoundaryCond falls back to the "
                        "builder's own literal. The proof that the printed "
                        "boundary comes from the wrapper.")
    args = p.parse_args(argv)

    wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
    known = {c["id"]: c for c in CONFIGS}
    bad = [c for c in wanted if c not in known]
    if bad:
        print(f"ERROR: unknown config(s) {bad}; known {list(known)}", file=sys.stderr)
        return 3
    configs = [known[c] for c in wanted]

    if args.mutate_skip_boundary_override and not args.dry_run:
        print("ERROR: --mutate-skip-boundary-override is a dry-run check",
              file=sys.stderr)
        return 3

    try:
        factors = [float(v) for v in args.resolution_factors.split(",") if v.strip()]
    except ValueError:
        print(f"ERROR: --resolution-factors {args.resolution_factors!r} is not a "
              f"list of numbers", file=sys.stderr)
        return 3
    if not factors or any(not (0.0 < f <= 1.0) for f in factors):
        print(f"ERROR: every resolution factor must be in (0, 1]; got {factors}",
              file=sys.stderr)
        return 3
    # The probe as it was -- factor 1.0 alone, truncation accepted -- keeps its
    # own code path and its own output byte for byte; anything else is a ladder.
    ladder = factors != [RESOLUTION_FACTOR] or args.refuse_truncation

    maker = load_maker()

    if args.dry_run and ladder:
        return dry_run_ladder(maker, configs, factors,
                              mutate_skip_for=args.mutate_skip_boundary_override,
                              real_nrts=args.real_nrts,
                              real_end_criteria=args.real_end_criteria,
                              refuse_truncation=args.refuse_truncation)
    if args.dry_run:
        return dry_run(maker, configs,
                       mutate_skip_for=args.mutate_skip_boundary_override,
                       real_nrts=args.real_nrts,
                       real_end_criteria=args.real_end_criteria)

    out_dir = Path(args.out) if args.out else Path("sheen_box_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        sf = maker._gate.load_spectral_features()
    except Exception as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 3
    try:
        maker._gate._import_openems()
    except Exception as exc:
        print(f"openEMS IS NOT IMPORTABLE: {exc!r}", file=sys.stderr)
        return 2

    if ladder:
        return run_ladder(maker, configs, factors, args, sf, out_dir)

    print("=" * 78)
    print("The Sheen low-pass filter -- openEMS box probe")
    print("=" * 78)
    print(f"  STAGE A IS NOT RUN. The reproduce gate passed in VESSL "
          f"{', '.join(STAGE_A_REFERENCE_JOBS)}, which produced the frozen record.")
    print(f"  openEMS build: {os.environ.get('RFX_OPENEMS_COMMIT', '(unstamped)')}; "
          f"image {os.environ.get('RFX_OPENEMS_IMAGE', '(unset)')}")
    print(f"  rung {maker.resolution_m(RESOLUTION_FACTOR) * 1e6:.2f} um; "
          f"--real-nrts {args.real_nrts} --real-end-criteria "
          f"{args.real_end_criteria:g} --accept-truncation")
    audit = _audit_definition_time_capture(maker)
    print(f"  definition-time capture audit: defaults carrying a patched value = "
          f"{audit['defaults_carrying_a_patched_value'] or 'none'}")
    for fname, reads in audit["reads"].items():
        print(f"    {fname:28s} reads {reads if reads else '(none directly)'}")
    if audit["defaults_carrying_a_patched_value"]:
        print("ERROR: a patched constant is frozen in a default argument.",
              file=sys.stderr)
        return 3

    control = snapshot(maker)
    results = []
    rc = 0
    for cfg in configs:
        try:
            results.append(run_config(
                maker, cfg, sim_root=args.sim_root, threads=args.threads,
                real_nrts=args.real_nrts, real_end_criteria=args.real_end_criteria,
                sf=sf))
        except Exception as exc:
            print(f"CONFIG {cfg['id']} FAILED: {exc}", file=sys.stderr)
            rc = 1
        finally:
            apply_globals(maker, control)

    frozen = None
    ref_path = (repo_root() / "tests" / "crossval" / "sheen_lpf" / "reference"
                / "openems_sheen.json")
    if ref_path.is_file():
        with ref_path.open() as fh:
            frozen = json.load(fh).get("stage_b_coarse")

    by_id = {r["id"]: r for r in results}
    base = by_id.get(CONTROL_ID)

    print("\n" + "=" * 78)
    print("READING 1 -- the mesh, the record and the cost")
    print("=" * 78)
    print("  id   y_clear  y faces  y lines  sub cells     cells   final energy "
          "(dB)  steps    wall (s)")
    for r in results:
        mesh = r["meta"].get("mesh_realized") or {}
        print(f"  {r['id']}  {r['y_clear_mm']:7.2f}  {r['boundary'][2]:>7s}  "
              f"{str((mesh.get('y') or {}).get('n_lines')):>7s}  "
              f"{str(mesh.get('substrate_z_cells_realized')):>9s}  "
              f"{str(mesh.get('n_cells')):>9s}  "
              f"{str(r['meta'].get('final_energy_db')):>16s}  "
              f"{str(r['meta'].get('record_length_steps')):>6s}  {r['wall_s']:9.1f}")

    print("\n" + "=" * 78)
    print("READING 2 -- the features")
    print("=" * 78)
    print("  id   null (GHz)   depth (dB)  -3 dB (GHz)  passband (dB)  "
          "energy sum band")
    for r in results:
        rec = r["record"]
        null, cut, pb = rec.get("null", {}), rec.get("cutoff_3db", {}), rec.get("passband", {})
        corner = "none" if cut.get("f_ghz") is None else f"{cut['f_ghz']:.5f}"
        print(f"  {r['id']}  {null.get('refined_f_ghz', float('nan')):10.5f}  "
              f"{null.get('depth_db', float('nan')):9.2f}  "
              f"{corner:>11s}  "
              f"{pb.get('mean_db', float('nan')):12.3f}  "
              f"{rec.get('min_energy_sum_band', float('nan')):.4f}-"
              f"{rec.get('max_energy_sum_band', float('nan')):.4f}")

    print("\n" + "=" * 78)
    print(f"READING 3 -- residual of the stopband null against {CONTROL_ID}, and "
          f"|S21| in dB against it")
    print("=" * 78)
    residuals, deltas = {}, {}
    if base is None:
        print(f"  {CONTROL_ID} was not run in this invocation; no residual table.")
    else:
        f0 = base["record"]["null"]["refined_f_ghz"]
        print(f"  {CONTROL_ID} null = {f0:.5f} GHz")
        print("  id   null (GHz)    delta (MHz)    residual (%)   max |dS21| (dB)  "
              "at (GHz)")
        for r in results:
            f = r["record"]["null"]["refined_f_ghz"]
            res = abs(f - f0) / f0
            d = delta_against(r["record"], base["record"])
            residuals[r["id"]] = res
            deltas[r["id"]] = d
            print(f"  {r['id']}  {f:10.5f}  {(f - f0) * 1e3:+12.3f}  "
                  f"{100.0 * res:13.4f}   {d['max_abs_delta_db']:14.3f}  "
                  f"{d['worst_f_ghz']}")

    if frozen is not None and base is not None:
        f_frozen = frozen["null"]["refined_f_ghz"]
        f0 = base["record"]["null"]["refined_f_ghz"]
        d = delta_against(base["record"], frozen)
        print(f"\n  the control against the frozen record's stage_b_coarse: "
              f"{f0:.5f} vs {f_frozen:.5f} GHz "
              f"({100.0 * abs(f0 - f_frozen) / f_frozen:.4f} %); max |dS21| "
              f"{d['max_abs_delta_db']:.4f} dB at {d['worst_f_ghz']} GHz")

    fig_path = None
    try:
        fig_path = write_figure(results, frozen, out_dir)
        print(f"\n  figure: {fig_path}")
    except Exception as exc:
        print(f"\n  figure NOT written: {exc!r}")

    payload = {
        "what": "openEMS box probe for the Sheen low-pass filter -- a diagnostic, "
                "not a reference record. Stage A is not run; it passed in VESSL "
                + ", ".join(STAGE_A_REFERENCE_JOBS),
        "resolution_factor": RESOLUTION_FACTOR,
        "resolution_um": maker.resolution_m(RESOLUTION_FACTOR) * 1e6,
        "real_nrts": args.real_nrts, "real_end_criteria": args.real_end_criteria,
        "control": CONTROL_ID,
        "definition_time_capture_audit": audit,
        "openems_commit": os.environ.get("RFX_OPENEMS_COMMIT"),
        "openems_image": os.environ.get("RFX_OPENEMS_IMAGE"),
        "configs": results,
        "residual_vs_control": residuals,
        "delta_vs_control": deltas,
        "figure": None if fig_path is None else str(fig_path),
    }
    json_path = out_dir / "sheen_box_probe_openems.json"
    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print(f"  json:   {json_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
