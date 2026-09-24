#!/usr/bin/env python
"""Make the MSL notch filter's openEMS reference from openEMS's OWN tutorial.

NEVER RUN BY CI. The external solver runs by hand, on the cluster, when the case
is created or its geometry changes; this script is the thing that is run.

WHAT IS SIMULATED
-----------------
A 600 um wide microstrip line on 254 um of lossless RO4350B (eps_r = 3.66) over a
PEC ground, with a 12 mm open-circuit stub of the same width branching off the
middle of the line. The stub is a quarter wavelength near 3.7 GHz, where its open
end transforms into a short across the line and |S21| collapses into a deep
transmission notch. That is the structure openEMS itself ships as a tutorial, and
it is the structure this repository's cross-validation case uses.

WHAT LIVES IN THE SHARED MODULE, AND WHY
----------------------------------------
``tests/crossval/_openems_tutorial_gate.py`` owns everything that is not this
case's: the tutorial's own constants and attribution, the recorded reproduction
that is its audit trail, the verbatim Stage A builder, the analytic notch
frequency and the Stage A gate, the thirteen sanity helpers copied
byte-identically from ``validation/crossval/20_msl_phase_referee.py``, the
generalized stage runner, the energy summary, the version probe and the record /
evidence writers. Read that module's docstring for the tutorial's citation, the
recorded reproduction, the list of sanity gates and the reason the Stage A
witness band is 2-7 GHz rather than the whole CalcPort grid. It is imported by
path, not by package name, because this script runs as a bare file on the
cluster.

What stays here is this case: its Stage B (the tutorial's own geometry on three
meshes), its delta list, its dry run, its self-check and its main.

STAGE A -- the reproduce gate
-----------------------------
The shared module's, unchanged: the tutorial verbatim, a 200-step smoke pass
first so a geometry or port defect costs seconds, then the real pass at the
library's own defaults, then ``CalcPort`` on the tutorial's own 1601-point grid,
then the gate -- the measured notch inside 0.80-1.05 x the analytic frequency
AND at least 20 dB deep. If the gate fails the script exits 1 and writes no
Stage B record.

STAGE B -- the record this case needs: the tutorial's geometry, three mesh rungs
--------------------------------------------------------------------------------
    DELTA 1 (geometry): NONE. MSL_length stays at the tutorial's 50 000 um each
        side of centre, and so does everything the tutorial derives from it --
        the x mesh extent, the substrate's x extent, ``FeedShift =
        10*resolution`` and ``MeasPlaneShift = MSL_length/3``.

    DELTA 2 (mesh rung): the tutorial's own ``resolution`` is multiplied by a
        factor -- 1.0 (``stage_b_coarse``), 1/sqrt(2) (``stage_b_mid``) and 0.5
        (``stage_b_fine``). Three meshes, two refinements, which is the least a
        mesh statement can be made of.

        The factor scales the x lines, the y lines, the air above the board AND
        the substrate's own z lines: ``linspace(0, h_sub, 5)`` becomes
        ``linspace(0, h_sub, round(4/factor)+1)``, so the 254 um board carries
        4, 6 and 8 cells. Refining x and y alone would leave the thickness
        direction at 4 cells on every rung -- and that is the direction that
        sets the effective permittivity, hence the notch frequency. A mesh
        statement made with it frozen would say nothing about it.

        TWO CONSEQUENCES follow, by the tutorial's own rules rather than by any
        new choice, and the dry run lists both per rung:
          * ``FeedShift = 10*resolution`` shrinks with the factor, so the
            excitation plane moves towards the port face: 4.477, 3.166,
            2.239 mm in from it.
          * ``PML_8`` is eight CELLS, not a length, so the absorber thins with
            the factor: about 3.58, 2.53, 1.79 mm.
        ``MeasPlaneShift = MSL_length/3`` is a length and does not move.

An earlier draft shortened the line arms to 10 000 um each side. It was dropped
(leader decision, 2026-09-22) for a reason that is geometry, not taste: at
10 000 um the tutorial's own ``MeasPlaneShift = MSL_length/3`` is 3.333 mm while
the x PML_8 is about 3.500 mm deep at the tutorial's resolution, so the port's
measurement plane falls INSIDE the absorber and upstream of the feed -- that is
no longer the tutorial's port model, whatever else it is. The shorter-arm
question moves to rfx's own box. The PML / feed / measurement-plane positions
are still computed and printed for every stage, which is how the case stays
readable when a future rung or box change moves them again.

Because DELTA 1 is empty, ``stage_b_coarse`` IS Stage A -- the same geometry on
the same mesh. The script does not solve it twice: when Stage A has run in the
same invocation, ``stage_b_coarse`` carries Stage A's arrays and says so in its
own ``source`` field. Run with ``--stage B`` alone, there is nothing to reuse and
the rung is solved.

Nothing else changes: boundary, excitation, the thirds-rule mesh recipe, the port
class and its arguments, NrTS and EndCriteria are all the tutorial's. That claim
is not prose -- ``--self-check`` regenerates the Stage B builder's body from the
shared module's Stage A builder body by three declared textual substitutions,
compares them character for character, and asserts that the ``msl_length_um``
parameter's DEFAULT is the tutorial's own 50 000 um.

DO-NOT-REPEAT, TICKED (the task recipe's 2026-08-03 addendum: quote the
precedent's header IN FULL and tick each recorded failure BEFORE writing code)
------------------------------------------------------------------------------
From ``validation/crossval/20_msl_phase_referee.py``'s own DO-NOT-REPEAT block,
which quotes ``build_msl_notch_openems_comparison.py``:

    "at dx=80 um the substrate is only 3.175 cells (the 'mixed-cell danger zone'
    rfx preflight warns about), where the openEMS MSL-port extraction is
    NON-PHYSICAL (|S11|^2+|S21|^2 up to 8.9, passivity grossly violated). dx=50
    um gives 5.08 substrate cells where BOTH solvers are passive, so it is the
    only valid matched-mesh comparison."

Facts about this script against it, no verdict:
  * Neither stage lays a uniform dx across the substrate. The tutorial's z
    recipe is an explicit ``linspace(0, 254, N+1)``, so the substrate top is
    always ON a mesh line and the cell count is an integer by construction, not
    a ratio that can land at 3.175.
  * Substrate cells per rung: 4 (coarse, 63.5 um each), 6 (mid, 42.33 um),
    8 (fine, 31.75 um). The recorded non-physical case had 3.175; the recorded
    passive case had 5.08.
  * The precedent's own header says the same about its Stage A: "Stage A's own
    mesh is the tutorial's own dx, ~lambda/50, unrelated to this trap."
  * The passivity witness is recorded and gated on every real pass here
    (2-7 GHz, 1.05), so if a rung does land somewhere non-physical, the record
    says so rather than the reader having to infer it.

Other ticks from the same precedent header: MUR sits only on the y faces, where
the tutorial's substrate spans the full y extent, so the y-face MUR sees a
uniform dielectric cross-section and not the mixed air/substrate step that blew
the coax lane up; ``ref_impedance`` is never passed to ``CalcPort`` (this case
passes no ``calcport_ref_impedance`` to the shared runner, so the single
unreferenced pass is what runs); the excitation guard has no absolute floor; no
complex value reaches ``json.dump``.

A failed gate exits non-zero and names itself. It also leaves its evidence: the
record built so far -- every array measured, the energy-sum numbers, and a
top-level ``failed_gate`` carrying the gate's own message -- is written to
``<output stem>_FAILED.json`` beside the requested output BEFORE the non-zero
exit. A gate that fires and leaves nothing to look at costs a whole cluster
cycle per iteration.

EXIT CODES
----------
0 every requested stage ran and every gate passed; 1 a gate failed; 2 openEMS is
not importable; 3 a layout/config bug in this script.

USAGE
-----
    python make_openems_reference.py --self-check
    python make_openems_reference.py --dry-run --stage both
    python make_openems_reference.py --stage both --output <path>.json \\
        --sim-root /tmp/msl_notch_openems --threads 8

The job file is ``scripts/vessl_msl_notch_openems_reference.yaml``. ``run_id`` in
the record is always ``null``: VESSL does not export the run id into the pod, so
the submitter fills it (``~/.claude/rules/vessl-jobs.md``).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# The shared tutorial gate, loaded by PATH. This script runs as a bare file on
# the cluster (``python tests/crossval/.../make_openems_reference.py``), where
# the repository root is not on sys.path and ``tests`` is not an importable
# package, so a package import would work under pytest and fail on the box that
# actually produces the record.
# ---------------------------------------------------------------------------
_GATE_MODULE_NAME = "_rfx_openems_tutorial_gate"


def _load_tutorial_gate():
    if _GATE_MODULE_NAME in sys.modules:
        return sys.modules[_GATE_MODULE_NAME]
    path = Path(__file__).resolve().parents[2] / "_openems_tutorial_gate.py"
    if not path.is_file():
        raise RuntimeError(
            f"the shared tutorial gate is missing at {path} -- this maker cannot "
            f"run without it"
        )
    spec = importlib.util.spec_from_file_location(_GATE_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_GATE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_gate = _load_tutorial_gate()

# The tutorial's own constants and the Stage A gate: ONE copy, in the shared
# module. Named here so the code below reads as it did when they were local.
_C0 = _gate._C0
A_UNIT = _gate.A_UNIT
A_MSL_LENGTH_UM = _gate.A_MSL_LENGTH_UM
A_MSL_WIDTH_UM = _gate.A_MSL_WIDTH_UM
A_SUBSTRATE_THICKNESS_UM = _gate.A_SUBSTRATE_THICKNESS_UM
A_SUBSTRATE_EPR = _gate.A_SUBSTRATE_EPR
A_STUB_LENGTH_UM = _gate.A_STUB_LENGTH_UM
A_F_MAX_HZ = _gate.A_F_MAX_HZ
A_N_FREQS = _gate.A_N_FREQS
A_BOUNDARY = _gate.A_BOUNDARY
A_PML_CELLS = _gate.A_PML_CELLS
_A_EPS_EFF = _gate._A_EPS_EFF
F_NOTCH_AN_HZ = _gate.F_NOTCH_AN_HZ
REPRODUCE_GATE_RECORD = _gate.REPRODUCE_GATE_RECORD
STAGE_A_GATE = _gate.STAGE_A_GATE
STAGE_A_MIN_DEPTH_DB = _gate.STAGE_A_MIN_DEPTH_DB
PASSIVITY_TOL = _gate.PASSIVITY_TOL
A_SUBSTRATE_Z_CELLS = _gate.A_SUBSTRATE_Z_CELLS
substrate_z_cells = _gate.substrate_z_cells
_build_stage_a_notch_tutorial = _gate._build_stage_a_notch_tutorial
_builder_body_after_kw = _gate._builder_body_after_kw
_smooth_estimate = _gate._smooth_estimate
_stage_a_gate_verdict = _gate.stage_a_gate_verdict
_failed_output_path = _gate.failed_output_path
_load_refined_extremum = _gate.load_refined_extremum
_write = _gate.write_record
_stages_for = _gate.stages_for
StageFailure = _gate.StageFailure
_import_openems = _gate._import_openems

# The band this record serves: 2-7 GHz, what the case's own test reads its
# reference records over. It is the tutorial gate's own band -- this case's
# Stage B IS the tutorial, so there is nothing to choose -- and one constant, so
# the number that is gated and the number that is reported cannot come from
# different bins. It is not a window derived from any run.
WITNESS_BAND_HZ = _gate.STAGE_A_WITNESS_BAND_HZ
NOTCH_BAND_GHZ = _gate.STAGE_A_NOTCH_BAND_GHZ


def _energy_summary(freqs_hz, s11, s21) -> dict:
    """This case's band and tolerance, on the shared energy summary."""
    return _gate.energy_summary(freqs_hz, s11, s21,
                                witness_band_hz=WITNESS_BAND_HZ,
                                passivity_tol=PASSIVITY_TOL)


# ---------------------------------------------------------------------------
# Stage B -- the tutorial's geometry on three meshes. No geometry delta.
#
# Three rungs, two refinements: 1.0, 1/sqrt(2), 0.5. The middle rung exists
# because a mesh statement needs at least two refinements to have a trend.
# ---------------------------------------------------------------------------
B_MSL_LENGTH_UM = A_MSL_LENGTH_UM  # unchanged from the tutorial, by decision
B_COARSE_RESOLUTION_FACTOR = 1.0
B_MID_RESOLUTION_FACTOR = 1.0 / np.sqrt(2.0)
B_FINE_RESOLUTION_FACTOR = 0.5
DELTA_LIST = [
    "DELTA 1 (geometry): NONE. MSL_length stays at the tutorial's 50000 um each "
    "side of centre, and so does everything the tutorial derives from it -- the x "
    "mesh extent, the substrate's x extent, FeedShift = 10*resolution and "
    "MeasPlaneShift = MSL_length/3. An earlier draft shortened the arms; it was "
    "dropped (leader decision, 2026-09-22) because at 50000/5 the tutorial's own "
    "MeasPlaneShift lands inside the x PML at the tutorial's resolution, which is "
    "no longer the tutorial's port model. The shorter-arm question moves to rfx's "
    "own box.",
    "DELTA 2 (mesh rung): the tutorial's own resolution is multiplied by a factor "
    "-- 1.0 (stage_b_coarse, the tutorial's own mesh, so it IS Stage A's model), "
    "1/sqrt(2) = 0.70711 (stage_b_mid) and 0.5 (stage_b_fine). The factor scales "
    "the x lines, the y lines, the air above the board AND the substrate's own z "
    "lines: linspace(0, h_sub, 5) becomes linspace(0, h_sub, round(4/factor)+1), "
    "so the substrate carries 4, 6 and 8 cells across its 254 um. Scaling x and y "
    "alone would leave the direction that sets the effective permittivity "
    "unrefined, and the mesh statement would say nothing about it. TWO "
    "CONSEQUENCES, by the tutorial's own rules and not by a new choice, listed "
    "per rung in the dry run: FeedShift = 10*resolution shrinks with the factor, "
    "so the excitation plane MOVES towards the port face (4.477, 3.166, 2.239 mm "
    "from it); and PML_8 is eight CELLS, so the absorber thins with the factor "
    "(about 3.58, 2.53, 1.79 mm). MeasPlaneShift = MSL_length/3 does not move.",
    "NOTHING ELSE: boundary ['PML_8','PML_8','MUR','MUR','PEC','MUR'], "
    "SetGaussExcite(f_max/2, f_max/2), the thirds-rule third_mesh refinement, the "
    "substrate/stub/trace dimensions, MSLPort with port0 excite=-1, the 1601-point "
    "CalcPort grid, and openEMS's own NrTS/EndCriteria defaults are all unchanged "
    "from Stage A.",
]

# The THREE textual substitutions that turn the Stage A builder's body into the
# Stage B builder's body. --self-check applies them and compares, character for
# character, and counts each one's occurrences in the Stage A body.
BUILDER_SUBSTITUTIONS = [
    ("A_MSL_LENGTH_UM", "msl_length_um"),
    ("/ A_UNIT / 50.0", "/ A_UNIT / 50.0 * resolution_factor"),
    ("np.linspace(0.0, A_SUBSTRATE_THICKNESS_UM, 5)",
     "np.linspace(0.0, A_SUBSTRATE_THICKNESS_UM, substrate_z_cells(resolution_factor) + 1)"),
]
BUILDER_SUBSTITUTION_COUNTS = {"A_MSL_LENGTH_UM": 8, "/ A_UNIT / 50.0": 1,
                               "np.linspace(0.0, A_SUBSTRATE_THICKNESS_UM, 5)": 1}

# ---------------------------------------------------------------------------
# STAGE B: the SAME builder with the mesh rung passed in. Its body is
# _build_stage_a_notch_tutorial's body with BUILDER_SUBSTITUTIONS applied --
# checked character for character by --self-check, which also asserts that
# msl_length_um DEFAULTS to the tutorial's own 50000 um, so the length is a
# parameter in form only and the geometry cannot drift from the tutorial's
# without someone passing it a different value on purpose.
# ---------------------------------------------------------------------------
def _build_notch_tutorial_at_rung(ContinuousStructure, openEMS, MSLPort, *,
                                  nrts: int | None, end_criteria: float | None,
                                  resolution_factor: float,
                                  msl_length_um: float = A_MSL_LENGTH_UM):
    """The tutorial's geometry at a given mesh rung."""
    kw = {}
    if nrts is not None:
        kw["NrTS"] = nrts
    if end_criteria is not None:
        kw["EndCriteria"] = end_criteria
    fdtd = openEMS(**kw)
    fdtd.SetGaussExcite(A_F_MAX_HZ / 2.0, A_F_MAX_HZ / 2.0)
    fdtd.SetBoundaryCond(["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"])

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(A_UNIT)

    resolution = _C0 / (A_F_MAX_HZ * np.sqrt(A_SUBSTRATE_EPR)) / A_UNIT / 50.0 * resolution_factor
    third_mesh = np.array([2.0 * resolution / 3.0, -resolution / 3.0]) / 4.0

    mesh.AddLine("x", [0.0])
    mesh.AddLine("x", A_MSL_WIDTH_UM / 2.0 + third_mesh)
    mesh.AddLine("x", -A_MSL_WIDTH_UM / 2.0 - third_mesh)
    mesh.SmoothMeshLines("x", resolution / 4.0)
    mesh.AddLine("x", [-msl_length_um, msl_length_um])
    mesh.SmoothMeshLines("x", resolution)

    mesh.AddLine("y", [0.0])
    mesh.AddLine("y", A_MSL_WIDTH_UM / 2.0 + third_mesh)
    mesh.AddLine("y", -A_MSL_WIDTH_UM / 2.0 - third_mesh)
    mesh.SmoothMeshLines("y", resolution / 4.0)
    mesh.AddLine("y", [-15.0 * A_MSL_WIDTH_UM, 15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM])
    mesh.AddLine("y", (A_MSL_WIDTH_UM / 2.0 + A_STUB_LENGTH_UM) + third_mesh)
    mesh.SmoothMeshLines("y", resolution)

    mesh.AddLine("z", np.linspace(0.0, A_SUBSTRATE_THICKNESS_UM, substrate_z_cells(resolution_factor) + 1))
    mesh.AddLine("z", [3000.0])
    mesh.SmoothMeshLines("z", resolution)

    substrate = csx.AddMaterial("RO4350B", epsilon=A_SUBSTRATE_EPR)
    substrate.AddBox(
        [-msl_length_um, -15.0 * A_MSL_WIDTH_UM, 0.0],
        [msl_length_um, 15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM, A_SUBSTRATE_THICKNESS_UM],
    )

    pec = csx.AddMetal("PEC")
    port0 = MSLPort(
        csx, port_nr=1, metal_prop=pec,
        start=[-msl_length_um, -A_MSL_WIDTH_UM / 2.0, A_SUBSTRATE_THICKNESS_UM],
        stop=[0.0, A_MSL_WIDTH_UM / 2.0, 0.0],
        prop_dir="x", exc_dir="z", excite=-1.0,
        FeedShift=10.0 * resolution, MeasPlaneShift=msl_length_um / 3.0,
        priority=10,
    )
    port1 = MSLPort(
        csx, port_nr=2, metal_prop=pec,
        start=[msl_length_um, -A_MSL_WIDTH_UM / 2.0, A_SUBSTRATE_THICKNESS_UM],
        stop=[0.0, A_MSL_WIDTH_UM / 2.0, 0.0],
        prop_dir="x", exc_dir="z",
        MeasPlaneShift=msl_length_um / 3.0,
        priority=10,
    )

    pec.AddBox(
        [-A_MSL_WIDTH_UM / 2.0, A_MSL_WIDTH_UM / 2.0, A_SUBSTRATE_THICKNESS_UM],
        [A_MSL_WIDTH_UM / 2.0, A_MSL_WIDTH_UM / 2.0 + A_STUB_LENGTH_UM, A_SUBSTRATE_THICKNESS_UM],
        priority=10,
    )
    return fdtd, port0, port1






# The tutorial's own plan, in the shared module: one copy, read by the dry-run
# table, by the Stage A record and by this case's own rungs (its Stage B IS the
# tutorial at a mesh rung).
_plan = _gate.tutorial_plan
_print_plan = _gate.print_tutorial_plan


def _stage_plans(fine_factor: float) -> dict:
    return {
        "stage_a": _plan("stage_a", A_MSL_LENGTH_UM, 1.0),
        "stage_b_coarse": _plan("stage_b_coarse", B_MSL_LENGTH_UM, B_COARSE_RESOLUTION_FACTOR),
        "stage_b_mid": _plan("stage_b_mid", B_MSL_LENGTH_UM, B_MID_RESOLUTION_FACTOR),
        "stage_b_fine": _plan("stage_b_fine", B_MSL_LENGTH_UM, fine_factor),
    }


def _coarse_is_stage_a() -> bool:
    """True when stage_b_coarse is Stage A's model, geometry and mesh alike."""
    return bool(B_MSL_LENGTH_UM == A_MSL_LENGTH_UM
                and B_COARSE_RESOLUTION_FACTOR == 1.0)



# ---------------------------------------------------------------------------
# --dry-run and --self-check
# ---------------------------------------------------------------------------
def _print_delta_list() -> None:
    print("DELTA LIST -- how Stage B differs from the tutorial:")
    for i, line in enumerate(DELTA_LIST, start=1):
        print(f"  [{i}] {line}")


def _dry_run(stage: str, fine_factor: float) -> int:
    print("=" * 78)
    print("The MSL notch filter -- openEMS reference maker, DRY RUN (no solver)")
    print("=" * 78)
    print(f"tutorial          {REPRODUCE_GATE_RECORD['tutorial']['repo']} "
          f"{REPRODUCE_GATE_RECORD['tutorial']['path']}")
    print(f"attribution       {REPRODUCE_GATE_RECORD['tutorial']['attribution']}")
    print(f"port used here    {REPRODUCE_GATE_RECORD['port_in_this_repository']} "
          f"(copied, never imported)")
    print(f"recorded notch    {REPRODUCE_GATE_RECORD['reproduced_f_notch_hz']/1e9:.4f} GHz "
          f"measured vs {REPRODUCE_GATE_RECORD['analytic_f_notch_hz']/1e9:.4f} GHz analytic, "
          f"{REPRODUCE_GATE_RECORD['reproduced_f_notch_dev_pct']:.2f} % "
          f"(VESSL {REPRODUCE_GATE_RECORD['vessl_run_id']}, "
          f"{REPRODUCE_GATE_RECORD['real_pass_wall_time_s']} s real pass)")
    print(f"boundary          {A_BOUNDARY}")
    print(f"excitation        SetGaussExcite({A_F_MAX_HZ/2:.4g}, {A_F_MAX_HZ/2:.4g}) Hz")
    print("NrTS / EndCriteria  not passed on a real pass: NrTS ~1e9 (the python "
          "binding's default) and EndCriteria 1e-6 (the pinned build's C++ default, "
          "openems.cpp:117; the binding's docstring says 1e-5, which is not what "
          "runs); 200 / 0.0 on the smoke pass")
    print(f"CalcPort grid     linspace(1e6, {A_F_MAX_HZ:.4g}, {A_N_FREQS})")
    print(f"analytic notch    F_NOTCH_AN = {F_NOTCH_AN_HZ/1e9:.4f} GHz "
          f"(eps_eff = {_A_EPS_EFF:.5f}, Hammerstad-Jensen)")
    print(f"Stage A gate      {STAGE_A_GATE['f_notch_lo_hz']/1e9:.4f} .. "
          f"{STAGE_A_GATE['f_notch_hi_hz']/1e9:.4f} GHz "
          f"(0.80-1.05 x F_NOTCH_AN, copied from the port)")
    print(f"notch estimator   refined_extremum(log) over "
          f"{NOTCH_BAND_GHZ[0]:.1f}-{NOTCH_BAND_GHZ[1]:.1f} GHz, "
          f"validation/crossval/comparators/spectral_features.py")
    print(f"passivity witness max(|S11|^2+|S21|^2) <= {1.0 + PASSIVITY_TOL:.2f} over the "
          f"SAME {NOTCH_BAND_GHZ[0]:.1f}-{NOTCH_BAND_GHZ[1]:.1f} GHz band "
          f"(WITNESS_BAND_HZ), not over the whole 1 MHz-7 GHz grid; the energy sum "
          f"is recorded for every bin either way")
    print(f"notch depth gate  the minimum must also be at least "
          f"{abs(STAGE_A_MIN_DEPTH_DB):.0f} dB deep (STAGE_A_MIN_DEPTH_DB), so a thru "
          f"line cannot pass the reproduce gate on frequency alone")
    print("on a failed gate   the record built so far, the four energy-sum numbers "
          "and failed_gate go to <output stem>_FAILED.json before the non-zero exit")
    print()
    _print_delta_list()
    print()
    plans = _stage_plans(fine_factor)
    order = _stages_for(stage)
    print("STAGE PLAN -- the geometry each stage builds:")
    for name in order:
        _print_plan(plans[name])
        print()
    if _coarse_is_stage_a():
        print("NOTE: DELTA 1 is empty, so stage_b_coarse is Stage A's model -- same "
              "geometry, same mesh. With --stage both it is not solved twice: it "
              "carries Stage A's arrays and says so in its own 'source' field. With "
              "--stage B there is nothing to reuse and it is solved.")
        print()
    print("WHAT THIS DRY RUN CANNOT TELL YOU: every mesh-line count and cell size "
          "above is a lower/upper bound from a pure-numpy subdivision. CSXCAD's own "
          "SmoothMeshLines grades fine-to-coarse transitions and adds lines this "
          "estimate does not model, and whether FeedShift/MeasPlaneShift land on a "
          "mesh line is decided by that same call. The real values are written into "
          "the record's meta block by the run itself.")
    return 0


def _self_check(fine_factor: float) -> int:
    failures = []
    notes = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        print(f"  [{'ok ' if ok else 'FAIL'}] {what}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(what)

    print("=" * 78)
    print("The MSL notch filter -- openEMS reference maker, SELF-CHECK (pure numpy)")
    print("=" * 78)

    print("constants (the tutorial's own):")
    check(A_MSL_LENGTH_UM == 50000.0, "MSL_length 50000 um")
    check(A_MSL_WIDTH_UM == 600.0, "MSL_width 600 um")
    check(A_SUBSTRATE_THICKNESS_UM == 254.0, "substrate 254 um")
    check(A_SUBSTRATE_EPR == 3.66, "eps_r 3.66")
    check(A_STUB_LENGTH_UM == 12000.0, "stub 12000 um")
    check(A_F_MAX_HZ == 7.0e9 and A_N_FREQS == 1601, "f_max 7 GHz, 1601 frequencies")
    check(A_BOUNDARY == ["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"], "boundary")
    check(abs(F_NOTCH_AN_HZ - REPRODUCE_GATE_RECORD["analytic_f_notch_hz"]) < 1.0,
          "F_NOTCH_AN recomputes to the recorded analytic frequency",
          f"{F_NOTCH_AN_HZ:.4f} Hz vs {REPRODUCE_GATE_RECORD['analytic_f_notch_hz']:.4f} Hz")

    print("the declared deltas (geometry: none; mesh rungs 1.0, 1/sqrt2, 0.5):")
    check(B_MSL_LENGTH_UM == A_MSL_LENGTH_UM,
          "Stage B's MSL_length IS the tutorial's",
          f"{B_MSL_LENGTH_UM:.0f} um")
    check(len(DELTA_LIST) == 3, "delta list has its three declared entries")
    check("DELTA 1 (geometry): NONE" in DELTA_LIST[0],
          "delta 1 declares no geometry change")
    check(B_COARSE_RESOLUTION_FACTOR == 1.0
          and abs(B_MID_RESOLUTION_FACTOR - 2.0 ** -0.5) < 1e-12
          and B_FINE_RESOLUTION_FACTOR == 0.5,
          "the three mesh rungs are 1.0, 1/sqrt(2) and 0.5",
          f"{B_COARSE_RESOLUTION_FACTOR:g}, {B_MID_RESOLUTION_FACTOR:.5f}, "
          f"{B_FINE_RESOLUTION_FACTOR:g}")
    check(0.0 < fine_factor < 1.0,
          "--resolution-factor is in (0, 1)", f"{fine_factor}")
    check((substrate_z_cells(1.0), substrate_z_cells(B_MID_RESOLUTION_FACTOR),
           substrate_z_cells(0.5)) == (4, 6, 8),
          "the rung factor gives 4, 6, 8 substrate cells",
          f"{substrate_z_cells(1.0)}, {substrate_z_cells(B_MID_RESOLUTION_FACTOR)}, "
          f"{substrate_z_cells(0.5)}")

    import inspect
    default = inspect.signature(_build_notch_tutorial_at_rung).parameters["msl_length_um"].default
    check(default == A_MSL_LENGTH_UM,
          "the Stage B builder's msl_length_um DEFAULTS to the tutorial's 50000 um",
          f"{default}")

    a_body = _builder_body_after_kw(_build_stage_a_notch_tutorial)
    b_body = _builder_body_after_kw(_build_notch_tutorial_at_rung)
    derived = a_body
    for old, new in BUILDER_SUBSTITUTIONS:
        derived = derived.replace(old, new)
    check(len(BUILDER_SUBSTITUTIONS) == 3, "three declared substitutions")
    for old, expected in BUILDER_SUBSTITUTION_COUNTS.items():
        check(a_body.count(old) == expected,
              f"'{old}' appears {expected}x in the tutorial builder's body",
              f"{a_body.count(old)}")
    check(derived == b_body,
          "the Stage B builder's body IS the tutorial builder's body with exactly "
          "the three declared substitutions applied")
    if derived != b_body:
        import difflib
        notes.append("\n".join(difflib.unified_diff(
            derived.splitlines(), b_body.splitlines(),
            "derived-from-stage-A", "stage-B-as-written", lineterm="")))

    # P1-2: these read the BUILDER'S OWN SOURCE. Comparing a plan value with the
    # constant the plan computed it from is a tautology -- both mutations of the
    # builder stayed green under the old form. Mutating the builder now reddens
    # these, because the statement they look for is no longer there.
    print("the builder's own source says where the ports and the board are:")
    check('mesh.AddLine("x", [-A_MSL_LENGTH_UM, A_MSL_LENGTH_UM])' in a_body,
          "the tutorial builder adds +-MSL_length as EXPLICIT x lines, so both "
          "port start planes are on a line by construction")
    check('start=[-A_MSL_LENGTH_UM, -A_MSL_WIDTH_UM / 2.0, A_SUBSTRATE_THICKNESS_UM]' in a_body
          and 'start=[A_MSL_LENGTH_UM, -A_MSL_WIDTH_UM / 2.0, A_SUBSTRATE_THICKNESS_UM]' in a_body,
          "both MSLPort start planes in the source ARE +-MSL_length")
    check('mesh.AddLine("x", [0.0])' in a_body and 'mesh.AddLine("y", [0.0])' in a_body,
          "the trace centre lines x=0 and y=0 are explicit in the source")
    check('np.linspace(0.0, A_SUBSTRATE_THICKNESS_UM, 5)' in a_body,
          "the tutorial builder's substrate z recipe is linspace(0, h_sub, 5): the "
          "substrate top is an explicit line and the board carries 4 cells")
    check('np.linspace(0.0, A_SUBSTRATE_THICKNESS_UM, substrate_z_cells(resolution_factor) + 1)'
          in b_body,
          "the Stage B builder's substrate z recipe scales with the rung")
    for name, plan in _stage_plans(fine_factor).items():
        check(plan["stub_tip_to_substrate_edge_mm"] > 0.0,
              f"{name}: the stub tip stops short of the substrate's y edge",
              f"{plan['stub_tip_to_substrate_edge_mm']:.3f} mm")

    print("the reproduce gate rejects a curve with no notch:")
    check(STAGE_A_MIN_DEPTH_DB == -20.0,
          "the gate's minimum depth is the case's own deep-null level, -20 dB")
    good = _stage_a_gate_verdict(REPRODUCE_GATE_RECORD["reproduced_f_notch_hz"], -53.16)
    check(good["passed"],
          "the recorded tutorial reproduction (3.6711 GHz, -53.16 dB) PASSES",
          f"f_ok={good['f_notch_ok']} depth_ok={good['depth_ok']}")
    # A thru line: |S21| flat at 1.0, so its shallowest-bin "notch" is 0 dB and
    # lands wherever numerical ripple puts it -- inside the frequency band.
    thru = _stage_a_gate_verdict(F_NOTCH_AN_HZ, -0.02)
    check(not thru["passed"] and thru["f_notch_ok"] and not thru["depth_ok"],
          "a THRU line (flat |S21|, minimum -0.02 dB, inside the frequency band) "
          "is REJECTED on depth -- the band alone would have passed it",
          f"f_ok={thru['f_notch_ok']} depth_ok={thru['depth_ok']} "
          f"passed={thru['passed']}")
    shallow = _stage_a_gate_verdict(F_NOTCH_AN_HZ, -19.9)
    check(not shallow["passed"],
          "a 19.9 dB dip is REJECTED too -- the threshold is exercised, not just "
          "the extremes")
    off_band = _stage_a_gate_verdict(1.5e9, -53.0)
    check(not off_band["passed"] and not off_band["f_notch_ok"],
          "a deep notch at 1.5 GHz is REJECTED on frequency")

    print("reported, not gated -- the port planes against the x PML_8 (estimate):")
    for name, plan in _stage_plans(fine_factor).items():
        print(f"  [--] {name}: PML depth {plan['pml_depth_mm_estimate']:.3f} mm, "
              f"FeedShift {plan['feed_shift_mm']:.3f} mm "
              f"(inside PML: {plan['feed_inside_pml_estimate']}), "
              f"MeasPlaneShift {plan['measplane_shift_mm']:.3f} mm "
              f"(inside PML: {plan['measplane_inside_pml_estimate']}, "
              f"downstream of the feed: {plan['measplane_downstream_of_feed']})")

    print("the witness band, and the energy-sum bookkeeping:")
    check(WITNESS_BAND_HZ == (2.0e9, 7.0e9), "witness band is 2-7 GHz")
    check(tuple(NOTCH_BAND_GHZ) == (WITNESS_BAND_HZ[0] / 1e9, WITNESS_BAND_HZ[1] / 1e9),
          "the notch estimator and the passivity witness read the SAME band")
    check(PASSIVITY_TOL == 0.05, "the passivity tolerance is unchanged at 1.05")
    f_hz = np.linspace(1.0e6, A_F_MAX_HZ, A_N_FREQS)
    # A planted spectrum: physical in the band, over unity below it -- what a
    # ratio of two near-floor port voltages looks like. The band witness must
    # not see the excess; the full-grid number must report it.
    s11_t = np.full(f_hz.shape, 0.30)
    s21_t = np.where(f_hz < WITNESS_BAND_HZ[0], 1.20, 0.90)
    summary = _energy_summary(f_hz, s11_t, s21_t)
    check(abs(summary["max_energy_sum_band"] - (0.09 + 0.81)) < 1e-9,
          "the band witness reads only in-band bins",
          f"{summary['max_energy_sum_band']:.4f}")
    check(abs(summary["max_energy_sum_full"] - (0.09 + 1.44)) < 1e-9,
          "the full-grid number still reports the out-of-band excess",
          f"{summary['max_energy_sum_full']:.4f}")
    check(abs(summary["max_energy_sum_below_band"]["0_1_ghz"] - 1.53) < 1e-9
          and abs(summary["max_energy_sum_below_band"]["1_2_ghz"] - 1.53) < 1e-9,
          "0-1 GHz and 1-2 GHz are reported separately")
    check(summary["energy_sum"].size == A_N_FREQS,
          "the energy sum is recorded for every bin", f"{summary['energy_sum'].size}")
    check(_failed_output_path(Path("/tmp/openems_tutorial.json")).name
          == "openems_tutorial_FAILED.json",
          "a failed gate's evidence file is named from the output stem")

    print("the shared notch estimator:")
    try:
        refined = _load_refined_extremum()
        f = np.linspace(2.0, 7.0, 201)
        mag = np.abs(f - 3.6) + 1e-3
        r = refined(f, mag, NOTCH_BAND_GHZ[0], NOTCH_BAND_GHZ[1], transform="log")
        check(abs(r["refined_f"] - 3.6) < 0.05,
              "refined_extremum loads by path and finds a planted minimum",
              f"{r['refined_f']:.4f} GHz")
    except Exception as exc:  # pragma: no cover - environment problem, reported
        check(False, "refined_extremum loads by path", repr(exc))

    print()
    print("WHAT THIS SELF-CHECK CANNOT DO: every mesh line between the explicitly "
          "added ones is placed by CSXCAD's SmoothMeshLines, and MSLPort snaps "
          "FeedShift and MeasPlaneShift to the nearest line itself. Both are CSXCAD "
          "calls. What is checked above is the BUILDER'S SOURCE -- which lines it "
          "adds explicitly, and with what arguments -- plus arithmetic on the "
          "declared numbers. It does NOT establish any interior line position, any "
          "exact line count, or that the feed and measurement planes land on lines. "
          "The run reads the realized x lines back from CSXCAD and records, per "
          "port, the nearest line to each declared plane and the distance to it "
          "(feed_plane_nearest_line_mm / meas_plane_nearest_line_mm and their "
          "*_snap_um); that is the only place those questions are answered.")
    for note in notes:
        print(note)
    print()
    print(f"SELF-CHECK {'PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0

# ---------------------------------------------------------------------------
# The run -- this case's stage, on the shared runner
# ---------------------------------------------------------------------------
def _run_stage(*, label: str, sim_root: str, threads: int,
               msl_length_um: float, resolution_factor: float,
               refined_extremum) -> tuple[dict, dict]:
    """This case's builder, grid, band and meta, handed to ``_gate.run_stage``.

    The sequence, every gate call and the order in which they run are the shared
    module's (see its ``run_stage`` docstring for what differs from the
    precedent's ``_run_stage_a_reproduce_gate``). What is chosen here: which
    builder a stage uses, the tutorial's own 1601-point CalcPort grid, the
    2-7 GHz witness band, openEMS's own NrTS/EndCriteria defaults on the real
    pass, and no ``ref_impedance`` on ``CalcPort``.
    """
    resolution_um = (_C0 / (A_F_MAX_HZ * np.sqrt(A_SUBSTRATE_EPR)) / A_UNIT / 50.0
                     * resolution_factor)

    def build(ContinuousStructure, openEMS, MSLPort, *, nrts, end_criteria):
        if label == "stage_a":
            return _gate.build_stage_a(ContinuousStructure, openEMS, MSLPort,
                                       nrts=nrts, end_criteria=end_criteria)
        return _build_notch_tutorial_at_rung(
            ContinuousStructure, openEMS, MSLPort,
            nrts=nrts, end_criteria=end_criteria,
            msl_length_um=msl_length_um, resolution_factor=resolution_factor)

    def mesh_realized_fn(lines):
        return _gate._mesh_realized(
            _gate.lines_in_um(lines, A_UNIT),
            substrate_thickness_um=A_SUBSTRATE_THICKNESS_UM)

    def meta_extra_fn(*, lines, port0, port1) -> dict:
        lines_um = _gate.lines_in_um(lines, A_UNIT)
        x = None if lines_um is None else lines_um["x"]
        return {
            "resolution_um": resolution_um,
            "resolution_factor": float(resolution_factor),
            "msl_length_um": float(msl_length_um),
            "box_mm": {
                "x": [-msl_length_um / 1e3, msl_length_um / 1e3],
                "y": [-15.0 * A_MSL_WIDTH_UM / 1e3,
                      (15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM) / 1e3],
                "z": [0.0, 3.0],
            },
            "port0": _gate._port_declared_and_snap(
                x, start_x_um=-msl_length_um, direction=+1.0,
                feed_shift_um=10.0 * resolution_um,
                measplane_shift_um=msl_length_um / 3.0, port_obj=port0,
                csx_unit_m=A_UNIT),
            "port1": _gate._port_declared_and_snap(
                x, start_x_um=+msl_length_um, direction=-1.0,
                feed_shift_um=10.0 * resolution_um,
                measplane_shift_um=msl_length_um / 3.0, port_obj=port1,
                csx_unit_m=A_UNIT),
            "substrate_z_cells_declared": substrate_z_cells(resolution_factor),
            "plan_estimate": _plan(label, msl_length_um, resolution_factor),
        }

    return _gate.run_stage(
        label=label, sim_root=sim_root, threads=threads, build=build,
        freqs_hz=np.linspace(1.0e6, A_F_MAX_HZ, A_N_FREQS),
        witness_band_hz=WITNESS_BAND_HZ, passivity_tol=PASSIVITY_TOL,
        real_nrts=None, real_end_criteria=None,
        mesh_realized_fn=mesh_realized_fn, meta_extra_fn=meta_extra_fn,
        features_fn=_gate.stage_a_notch_features(refined_extremum))


def _build_artifact(records: dict, stage_meta: dict, stage_a_gate: dict,
                    stages: list, *, failed_gate: str | None = None) -> dict:
    """This case's meta block, on the shared record writer."""
    return _gate.build_artifact(
        records, stage_meta, stage_a_gate, stages,
        stage_names=("stage_a", "stage_b_coarse", "stage_b_mid", "stage_b_fine"),
        produced_by="tests/crossval/msl_notch_filter/reference/make_openems_reference.py",
        failed_gate=failed_gate,
        meta_common={
            "comparability": (
                "The S11/S21 PHASES are referenced at the tutorial's own measurement "
                "planes, MeasPlaneShift = MSL_length/3 = 16.667 mm in from each port "
                "face on the tutorial's 50 mm arms, and are NOT comparable with a "
                "record taken on a different arm length; the MAGNITUDES are."
            ),
            "tutorial_source": (
                f"{REPRODUCE_GATE_RECORD['tutorial']['repo']}/"
                f"{REPRODUCE_GATE_RECORD['tutorial']['path']} -- "
                f"{REPRODUCE_GATE_RECORD['tutorial']['attribution']}, fetched verbatim "
                f"{REPRODUCE_GATE_RECORD['tutorial']['fetched_verbatim_on']} via "
                f"{REPRODUCE_GATE_RECORD['tutorial']['fetched_via']}"
            ),
            "delta_list": DELTA_LIST,
            "boundary": A_BOUNDARY,
            "excitation": f"SetGaussExcite({A_F_MAX_HZ/2.0}, {A_F_MAX_HZ/2.0}) Hz",
            "nrts": "openEMS library default (~1e9) on every real pass; 200 on the smoke pass",
            "end_criteria": ("not passed on any real pass, so the pinned openEMS build's "
                             "C++ default 1e-6 ran (openems.cpp:117; the python binding's "
                             "docstring says 1e-5, which is not what runs); 0.0 on the "
                             "smoke pass"),
            "calcport_grid": f"linspace(1e6, {A_F_MAX_HZ}, {A_N_FREQS})",
            "calcport_passes": (
                "one pass, no ref_impedance -- the precedent's own tick, kept"
            ),
            "notch_estimator": "validation/crossval/comparators/spectral_features.py::"
                               "refined_extremum, transform='log', band "
                               f"{NOTCH_BAND_GHZ[0]:g}-{NOTCH_BAND_GHZ[1]:g} GHz",
            "witness_band_ghz": list(NOTCH_BAND_GHZ),
            "passivity_tol": 1.0 + PASSIVITY_TOL,
            "passivity_witness": (
                "max(|S11|^2+|S21|^2) over the witness band only. The CalcPort grid "
                "starts at 1 MHz, where both port voltages are at the numerical floor "
                "and their ratio is not an S-parameter the structure produced; the "
                "band this record serves is 2-7 GHz. Every bin's energy sum is still "
                "recorded, with its maximum over the band, over the whole grid, and "
                "over 0-1 GHz and 1-2 GHz separately."
            ),
        })


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=["A", "B", "both"], default="both")
    p.add_argument("--output", default=None, help="where the JSON record is written")
    p.add_argument("--sim-root", default="/tmp/msl_notch_openems")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--resolution-factor", type=float, default=B_FINE_RESOLUTION_FACTOR,
                   help="Stage B only: the FINEST rung's factor on the tutorial's own "
                        "resolution. The coarse rung is always 1.0 and the middle rung "
                        "1/sqrt(2), so a Stage B run always produces three rungs. "
                        "Default 0.5.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args(argv)

    # The range check runs on EVERY mode, not only a real run: a dry run or a
    # self-check on an out-of-range factor would print a plan nothing can build.
    if args.resolution_factor <= 0.0 or args.resolution_factor >= 1.0:
        print(f"ERROR: --resolution-factor must be in (0, 1); got "
              f"{args.resolution_factor}", file=sys.stderr)
        return 3

    if args.self_check:
        return _self_check(args.resolution_factor)
    if args.dry_run:
        return _dry_run(args.stage, args.resolution_factor)

    if not args.output:
        print("ERROR: --output is required for a real run", file=sys.stderr)
        return 3
    # P2-7: the image stamps the commit it built openEMS from. A record that
    # cannot say which solver build produced it is not a reference.
    if not os.environ.get("RFX_OPENEMS_COMMIT"):
        print("ERROR: RFX_OPENEMS_COMMIT is not set. The record must name the "
              "openEMS build it came from; the job file exports it from the image. "
              "Refusing to produce a reference with no solver provenance.",
              file=sys.stderr)
        return 3

    try:
        refined_extremum = _load_refined_extremum()
    except Exception as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 3

    try:
        _import_openems()
    except Exception as exc:
        print(f"openEMS IS NOT IMPORTABLE: {exc!r}", file=sys.stderr)
        return 2

    print("=" * 78)
    print("The MSL notch filter -- openEMS reference from openEMS's own tutorial")
    print("=" * 78)
    _print_delta_list()
    print()

    stages = _stages_for(args.stage)
    if "stage_a" not in stages:
        print("WARNING: --stage B does not run the reproduce gate. The record this "
              "invocation writes carries stage_a: null and reproduce_gate_ran: false, "
              "and no number in it has been checked against openEMS's own tutorial "
              "result in this run. The job file submits --stage both.", flush=True)
    factors = {"stage_a": (A_MSL_LENGTH_UM, 1.0),
               "stage_b_coarse": (B_MSL_LENGTH_UM, B_COARSE_RESOLUTION_FACTOR),
               "stage_b_mid": (B_MSL_LENGTH_UM, B_MID_RESOLUTION_FACTOR),
               "stage_b_fine": (B_MSL_LENGTH_UM, args.resolution_factor)}

    records: dict = {"stage_a": None, "stage_b_coarse": None,
                     "stage_b_mid": None, "stage_b_fine": None}
    stage_meta: dict = {}
    stage_a_gate: dict = {}

    for name in stages:
        msl_len, factor = factors[name]
        print(f"--- {name} ---", flush=True)

        # DELTA 1 is empty, so stage_b_coarse is Stage A's model on Stage A's
        # mesh. Solving it again would put two arrays in the record that a
        # reader could mistake for two measurements. Carry Stage A's and say so.
        if (name == "stage_b_coarse" and _coarse_is_stage_a()
                and records["stage_a"] is not None):
            record = dict(records["stage_a"])
            record.pop("gate", None)
            record["source"] = (
                "stage_a's arrays, not a second run: DELTA 1 is empty, so this rung "
                "is the same geometry on the same mesh as Stage A. Solving it twice "
                "would put two identical arrays in the record."
            )
            records[name] = record
            stage_meta[name] = dict(stage_meta["stage_a"], reused_from="stage_a")
            print(f"  reusing stage_a's arrays (same model, same mesh); "
                  f"notch {record['notch']['refined_f_ghz']:.4f} GHz", flush=True)
            continue

        try:
            record, meta = _run_stage(
                label=name, sim_root=args.sim_root, threads=args.threads,
                msl_length_um=msl_len, resolution_factor=factor,
                refined_extremum=refined_extremum)
        except StageFailure as exc:
            print(f"SANITY GATE FAILED [{name}]: {exc}", file=sys.stderr)
            records[name] = exc.partial or None
            stage_meta[name] = exc.meta
            # The gate messages from the copied helpers already open with
            # "[<stage>]", so the message is carried as it was raised.
            failed = _build_artifact(records, stage_meta, stage_a_gate, stages,
                                     failed_gate=str(exc))
            path = _failed_output_path(Path(args.output))
            _write(failed, path)
            print(f"evidence written to {path} -- the arrays measured before the "
                  f"gate fired, the energy-sum numbers and failed_gate.",
                  file=sys.stderr)
            return 1
        records[name] = record
        stage_meta[name] = meta
        notch_hz = record["notch"]["refined_f_ghz"] * 1e9
        print(f"  notch {record['notch']['refined_f_ghz']:.4f} GHz "
              f"(bin {record['notch']['bin_f_ghz']:.4f} GHz, "
              f"depth {record['notch']['depth_db']:.2f} dB), "
              f"{meta['wall_time_s']} s", flush=True)

        if name == "stage_a":
            stage_a_gate = _stage_a_gate_verdict(notch_hz, record["notch"]["depth_db"])
            record["gate"] = stage_a_gate
            print(f"  reproduce gate: measured {notch_hz/1e9:.4f} GHz "
                  f"({stage_a_gate['measured_depth_db']:.2f} dB deep) vs analytic "
                  f"{F_NOTCH_AN_HZ/1e9:.4f} GHz ({stage_a_gate['deviation_pct']:.2f} %); "
                  f"band {STAGE_A_GATE['f_notch_lo_hz']/1e9:.4f}-"
                  f"{STAGE_A_GATE['f_notch_hi_hz']/1e9:.4f} GHz -> "
                  f"{'ok' if stage_a_gate['f_notch_ok'] else 'RED'}, "
                  f"depth <= {STAGE_A_MIN_DEPTH_DB:.0f} dB -> "
                  f"{'ok' if stage_a_gate['depth_ok'] else 'RED'} => "
                  f"{'PASSED' if stage_a_gate['passed'] else 'FAILED'}", flush=True)
            if not stage_a_gate["passed"]:
                why = []
                if not stage_a_gate["f_notch_ok"]:
                    why.append(f"frequency {notch_hz/1e9:.4f} GHz outside "
                               f"{STAGE_A_GATE['f_notch_lo_hz']/1e9:.4f}-"
                               f"{STAGE_A_GATE['f_notch_hi_hz']/1e9:.4f} GHz")
                if not stage_a_gate["depth_ok"]:
                    why.append(f"minimum only {stage_a_gate['measured_depth_db']:.2f} dB "
                               f"deep, not a notch at the {STAGE_A_MIN_DEPTH_DB:.0f} dB level")
                print("REPRODUCE GATE FAILED: no Stage B record is written.", file=sys.stderr)
                failed = _build_artifact(
                    records, stage_meta, stage_a_gate, stages,
                    failed_gate="[stage_a] reproduce gate FAILED: " + "; ".join(why))
                path = _failed_output_path(Path(args.output))
                _write(failed, path)
                print(f"evidence written to {path}", file=sys.stderr)
                return 1

    artifact = _build_artifact(records, stage_meta, stage_a_gate, stages)
    out = Path(args.output)
    _write(artifact, out)
    print(f"\n=== written to {out} ===")
    print("run_id is null by design: VESSL does not export the run id into the pod, "
          "so the submitter records it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
