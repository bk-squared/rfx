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

CLI: ``--configs``, ``--out DIR``, ``--sim-root``, ``--threads``,
``--real-nrts``, ``--real-end-criteria``, ``--dry-run``,
``--mutate-skip-boundary-override``.
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
            real_nrts: int, real_end_criteria: float) -> int:
    print("=" * 78)
    print("The Sheen low-pass filter -- openEMS box probe, DRY RUN (no solver)")
    print("=" * 78)
    print(f"  rung             resolution_factor {RESOLUTION_FACTOR:g} = "
          f"{maker.resolution_m(RESOLUTION_FACTOR) * 1e6:.2f} um, the record's "
          f"coarse rung")
    print(f"  stop criteria    --real-nrts {real_nrts} --real-end-criteria "
          f"{real_end_criteria:g} --accept-truncation, the way the frozen record "
          f"was made")
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
                    resolution_factor=RESOLUTION_FACTOR)
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
    print("\n  WHAT THIS DRY RUN CANNOT TELL YOU: the mesh line counts are the "
          "maker's own pure-numpy lower bound; CSXCAD's SmoothMeshLines grades "
          "fine-to-coarse transitions and adds lines this estimate does not "
          "model. The realized mesh, the port snaps and the energy trace come "
          "from the run itself.")
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


def run_config(maker, cfg: dict, *, sim_root: str, threads: int,
               real_nrts: int, real_end_criteria: float, sf) -> dict:
    values = config_globals(maker, cfg)
    apply_globals(maker, values)
    original_builder = maker._build_sheen_board_at_rung
    builder, how = make_builder(maker, cfg["boundary"])
    maker._build_sheen_board_at_rung = builder

    print(f"\n{'=' * 78}\n=== config {cfg['id']}: Y_CLEAR "
          f"{cfg['y_clear'] * 1e3:g} mm, y faces {cfg['boundary'][2]} / "
          f"{cfg['boundary'][3]} -- {cfg['why']}\n{'=' * 78}")
    print(f"  declared box {maker.LX * 1e3:.3f} x {values['LY'] * 1e3:.3f} x "
          f"{maker.LZ * 1e3:.3f} mm; section y {values['PATCH_Y_LO'] * 1e3:.4f}-"
          f"{values['PATCH_Y_HI'] * 1e3:.4f} mm; feeds y "
          f"{values['IN_FEED_YC'] * 1e3:.4f} / {values['OUT_FEED_YC'] * 1e3:.4f} mm")
    print(f"  boundary this config asks for: {cfg['boundary']}")
    print(f"  override mechanism: {how}", flush=True)

    t0 = time.time()
    try:
        record, meta = maker._run_stage_b(
            label=f"box_probe_{cfg['id']}", sim_root=sim_root, threads=threads,
            resolution_factor=RESOLUTION_FACTOR, sf=sf,
            real_nrts=real_nrts, real_end_criteria=real_end_criteria,
            accept_truncation=True)
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

    return {"id": cfg["id"], "why": cfg["why"],
            "y_clear_mm": cfg["y_clear"] * 1e3, "boundary": list(cfg["boundary"]),
            "override_mechanism": how, "mechanism_used": mechanism_used,
            "wall_s": wall, "record": record, "meta": meta}


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

    maker = load_maker()

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
