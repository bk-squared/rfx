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

THE TUTORIAL, ITS SOURCE AND ITS ATTRIBUTION
--------------------------------------------
Copied from the docstring of ``validation/crossval/20_msl_phase_referee.py``
(that script's STAGE A section), which is where this repository's faithful port
of the tutorial already lives:

    ``python/Tutorials/MSL_NotchFilter.py`` (fetched verbatim 2026-08-04 via
    ``gh api repos/thliebig/openEMS/contents/python/Tutorials/MSL_NotchFilter.py``;
    "(c) 2016-2023 Thorsten Liebig", "Tested with python 3.10, openEMS v0.0.35+")
    builds an open-circuit-stub microstrip notch filter on RO4350B (eps_r=3.66,
    h_sub=254um), MSL_width=600um, MSL_length=50mm (each side), stub_length=12mm,
    f_max=7GHz, boundary ``['PML_8','PML_8','MUR','MUR','PEC','MUR']``,
    ``SetGaussExcite(f_max/2, f_max/2)`` (both center AND corner set to the SAME
    3.5GHz -- an unusual-looking but literal reading, reproduced verbatim rather
    than "corrected"). ``AddMSLPort`` (thirds-rule mesh already built into the
    tutorial's own ``third_mesh`` refinement at the trace's Y-edges) with port[0]
    ``excite=-1`` (verbatim, not "corrected" to +1) and ``FeedShift=10*resolution``,
    ``MeasPlaneShift=MSL_length/3`` on BOTH ports. openEMS(NrTS, EndCriteria) are
    left at the library's own defaults (NrTS~=1e9, EndCriteria=1e-5 --
    ``python/openEMS/openEMS.pyx`` docstring, fetched 2026-08-04) because the
    tutorial itself never overrides them.

THE RECORDED REPRODUCTION (task recipe ``external_solver_comparator.md``, step 2)
---------------------------------------------------------------------------------
The port above has already reproduced the tutorial's own notch, and that is the
number this script's Stage A gate is built on:

    measured f_notch  3.6711 GHz
    analytic f_notch  3.6872 GHz  (quarter-wave open stub, Hammerstad-Jensen
                                   eps_eff; recomputed below as F_NOTCH_AN_HZ)
    deviation         0.44 %
    VESSL run         369367251705 (2026-08-04)
    log               validation/crossval/_20_msl_phase_referee_logs/
                      20260804T070702Z_run.log  (its lines 10-17)
    wall time         41.6 s for the real pass, 8 threads
                      (validation/crossval/_20_msl_phase_referee_logs/
                       20260804T055009Z_result.json, ``stage_a.elapsed_s``;
                       it is NOT in the log text)

STAGE A -- the reproduce gate
-----------------------------
The tutorial port, verbatim. ``_build_stage_a_notch_tutorial`` below is a
byte-identical copy of the same function in
``validation/crossval/20_msl_phase_referee.py`` (that file is another case's and
is never imported from here -- it is copied, never edited). A 200-step smoke pass
(``end_criteria=0.0``) runs first so a geometry or port defect costs seconds, then
the real pass at the library's own defaults, then ``CalcPort`` on the tutorial's
own 1601-point grid.

The gate is the SAME band the port's own ``STAGE_A_GATE`` used, copied with its
justification (see ``STAGE_A_GATE`` below). What is NOT copied: the port picked
the notch with a bare ``argmin`` over 0.5-1.5 x the analytic frequency. This
script uses the repository's shared estimator instead --
``validation/crossval/comparators/spectral_features.py::refined_extremum``, log
transform, over 2-7 GHz -- the same estimator the case's own test reads its
reference records with, so the record and the case cannot drift apart. If the
gate fails the script exits 1 and writes no Stage B record.

STAGE B -- the record this case needs: the tutorial's geometry, two mesh rungs
------------------------------------------------------------------------------
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
Stage A builder's body by two declared textual substitutions, compares them
character for character, and asserts that the ``msl_length_um`` parameter's
DEFAULT is the tutorial's own 50 000 um (see ``_builder_body_after_kw``).

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
the coax lane up; ``ref_impedance`` is never passed to ``CalcPort``; the
excitation guard has no absolute floor; no complex value reaches ``json.dump``.

SANITY GATES PORTED, AND FROM WHERE
-----------------------------------
All from ``validation/crossval/20_msl_phase_referee.py`` (byte-identical copies;
``--self-check`` does not verify that, a reviewer does -- see the REPORT's diff
proof), applied unchanged to EVERY real pass:

  * ``_scan_stdout_for_bad_patterns`` -- stdout AND stderr scan for "Unused
    primitive" / "not on the mesh" / "unused excitation" before the run is
    trusted, plus the excitation-clipping patterns on real passes only;
  * ``_run_openems_capturing_stdout`` -- captures fd 1 AND fd 2 (CSXCAD writes
    some port/mesh warnings to stderr; an fd-1-only capture saw none of them);
  * ``_check_excitation_and_trace`` -- nonzero excitation energy and non-empty,
    non-zero port voltage traces, with no absolute floor;
  * ``_log_indicates_truncation`` -- openEMS's own "reached before the
    end-criteria of" text. A real pass that trips it FAILS here;
  * ``_non_physical_guard`` -- |S| finite and <= 2;
  * ``_passivity_witness`` -- max(|S11|^2+|S21|^2) <= 1.05 OVER 2-7 GHz, else fail.

THE WITNESS BAND, AND WHY IT IS NOT THE WHOLE GRID (leader decision, 2026-09-22)
--------------------------------------------------------------------------------
The tutorial's ``CalcPort`` grid is ``linspace(1e6, 7e9, 1601)``, so its first
bins sit near 1 MHz -- three and a half decades below the Gaussian excitation's
3.5 GHz centre. Down there both port voltages, incident and reflected, are at the
numerical floor, and ``uf_ref/uf_inc`` is a ratio of two such numbers rather than
an S-parameter the structure produced. The band this record serves is 2-7 GHz,
and that is where the witness is evaluated, via ``_passivity_witness``'s own
``idx`` argument (the function itself is untouched). The energy sum is still
recorded for EVERY bin, together with its maximum over the band, over the whole
grid, and separately over 0-1 GHz and 1-2 GHz, so a reader sees where any excess
lives instead of being told a single number.

On VESSL 369367263115 (commit 09487db4) the whole-grid witness read 1.2717 and
stopped Stage A after a ~39 s real pass. WHERE on the grid that 1.2717 sat is not
known: no record was written. That is the second thing this script now fixes --
see below.

A failed gate exits non-zero and names itself. It also leaves its evidence: the
record built so far -- every array measured, the four energy-sum numbers, and a
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
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants -- the tutorial's own, copied from validation/crossval/
# 20_msl_phase_referee.py's Stage A block.
# ---------------------------------------------------------------------------
_C0 = 2.998e8  # m/s -- matches the coax/notch precedent's own constant

A_UNIT = 1.0e-6  # CSXCAD length unit: um (tutorial's own choice)
A_MSL_LENGTH_UM = 50000.0
A_MSL_WIDTH_UM = 600.0
A_SUBSTRATE_THICKNESS_UM = 254.0
A_SUBSTRATE_EPR = 3.66
A_STUB_LENGTH_UM = 12.0e3
A_F_MAX_HZ = 7.0e9
A_N_FREQS = 1601  # tutorial's own np.linspace(1e6, f_max, 1601)

A_BOUNDARY = ["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"]
A_PML_CELLS = 8  # "PML_8" -- the x faces only

# Quarter-wave open-stub notch, Hammerstad-Jensen eps_eff. Recomputed here, not
# copy-pasted, exactly as the precedent does (it cross-checks to 5 significant
# figures against this repo's own recorded "fringing-free analytic 3.69 GHz").
_A_STUB_LEN_M = 12.0e-3
_A_W_TRACE_M = 600e-6
_A_H_SUB_M = 254e-6
_A_EPS_R = 3.66
_A_U = _A_W_TRACE_M / _A_H_SUB_M
_A_EPS_EFF = (_A_EPS_R + 1.0) / 2.0 + (_A_EPS_R - 1.0) / 2.0 * (1.0 + 12.0 / _A_U) ** -0.5
F_NOTCH_AN_HZ = _C0 / (4.0 * _A_STUB_LEN_M * np.sqrt(_A_EPS_EFF))

# The recorded reproduction, as an audit artifact carried into every record this
# script writes (the precedent's REPRODUCE_GATE_RECORD, narrowed to Stage A).
REPRODUCE_GATE_RECORD = {
    "tutorial": {
        "repo": "thliebig/openEMS",
        "path": "python/Tutorials/MSL_NotchFilter.py",
        "attribution": "(c) 2016-2023 Thorsten Liebig",
        "fetched_verbatim_on": "2026-08-04",
        "fetched_via": "gh api repos/thliebig/openEMS/contents/python/Tutorials/MSL_NotchFilter.py",
    },
    "port_in_this_repository": "validation/crossval/20_msl_phase_referee.py",
    "reproduced_f_notch_hz": 3671100625.0,
    "analytic_f_notch_hz": 3687193135.4851503,
    "reproduced_f_notch_dev_pct": 0.4364433837294213,
    "vessl_run_id": "369367251705",
    "log_path": "validation/crossval/_20_msl_phase_referee_logs/20260804T070702Z_run.log",
    "log_lines": "10-17",
    "real_pass_wall_time_s": 41.6,
    "real_pass_wall_time_source": (
        "validation/crossval/_20_msl_phase_referee_logs/20260804T055009Z_result.json, "
        "stage_a.elapsed_s -- the wall time is not in the run log's text"
    ),
    "verified_on": "2026-08-04",
}

# STAGE_A_GATE -- copied, with its justification, from the port's own
# REPRODUCE_GATE_RECORD["gate"] (validation/crossval/20_msl_phase_referee.py):
#
#   One-sided-low-biased band: docs/agent-memory/rfx-known-issues.md records
#   openEMS reading ~7% LOW vs the fringing-free analytic on a DIFFERENT (not
#   identical) line-length/domain combination of this SAME substrate/trace/stub
#   -- PREDICTED, not yet measured for THIS exact tutorial geometry, hence the
#   generosity.
#
# The band is unchanged here. What the band is applied TO changed: the shared
# estimator's refined frequency over 2-7 GHz, not a bare argmin over
# 0.5-1.5 x F_NOTCH_AN_HZ.
STAGE_A_GATE = {
    "f_notch_lo_hz": 0.80 * float(F_NOTCH_AN_HZ),
    "f_notch_hi_hz": 1.05 * float(F_NOTCH_AN_HZ),
}

# A frequency band alone does not say a notch is there. A through line with no
# stub has a |S21| minimum somewhere in 2-7 GHz too, and it would sit inside the
# band above. The gate therefore also requires the minimum to be a NOTCH: at
# least 20 dB below unity -- the same deep-null level the case itself uses
# (PI, 2026-09-22). A thru line fails it, which is what the self-check plants.
STAGE_A_MIN_DEPTH_DB = -20.0

# The band this record serves: 2-7 GHz, what the case's own test reads its
# reference records over. ONE constant, used by both the notch estimator and the
# passivity witness, so the number that is gated and the number that is reported
# cannot come from different bins. It is not a window derived from any run.
WITNESS_BAND_HZ = (2.0e9, 7.0e9)
NOTCH_BAND_GHZ = (WITNESS_BAND_HZ[0] / 1e9, WITNESS_BAND_HZ[1] / 1e9)

# The passivity tolerance, unchanged: max(|S11|^2+|S21|^2) <= 1.05.
PASSIVITY_TOL = 0.05

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

# The tutorial's own substrate z recipe is linspace(0, h_sub, 5) -- 4 cells.
# The rung factor scales it too (see DELTA 2).
A_SUBSTRATE_Z_CELLS = 4


def substrate_z_cells(resolution_factor: float) -> int:
    """Substrate cells at a rung: round(4 / factor). 4, 6, 8 at 1.0, 1/sqrt2, 0.5."""
    return max(1, int(round(A_SUBSTRATE_Z_CELLS / resolution_factor)))


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
# openEMS import plumbing (deferred, matches the coax/thru precedent so
# this module stays importable -- and testable -- without openEMS).
# ---------------------------------------------------------------------------
def _ensure_openems_numpy_compat() -> None:
    for name, value in {"float": float, "int": int, "complex": complex, "mat": np.matrix}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _import_openems():
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS
    from openEMS.ports import MSLPort
    return ContinuousStructure, openEMS, MSLPort


# ---------------------------------------------------------------------------
# Shared sanity-check helpers (ticked against the coax/thru precedent above)
# ---------------------------------------------------------------------------
_BAD_STDOUT_PATTERNS = ("Unused primitive", "not on the mesh", "unused excitation")

# Truncation patterns (D3/TRUNCATION PATTERNS fix, run-1 forensics, review
# 2026-08-04): openEMS's own excitation-pulse-clipping warnings. Checked ONLY
# via ``check_truncation=True`` (real-run scans), NEVER for smoke scans --
# the smoke runs in THIS script are deliberately tiny (NrTS=200,
# EndCriteria=0.0/"-infdB", see ``_build_stage_a_notch_tutorial``/
# ``_build_stage_b``'s own smoke calls), so their excitation pulse (tens of
# thousands of timesteps at this script's own f0/fc) is ALWAYS clipped by
# construction -- run-1's own committed log (``validation/crossval/
# _20_msl_phase_referee_logs/20260804T070702Z_run.log``) shows both strings
# firing for the smoke portion of BOTH stages while the REAL portion (which
# reached its own EndCriteria, per REPRODUCE_GATE_RECORD["settling_evidence"])
# shows neither -- the exact positive control this scoping relies on: same
# binary, same stream, smoke trips it and real does not.
_TRUNCATION_STDOUT_PATTERNS = (
    "Cutting to max number of timesteps",
    "max. number of timesteps is smaller than three times the excitation",
)

# Benign, narrowly-scoped exception (GUARD CHANNEL-GAP FIX, run-1 forensics,
# 2026-08-04): CSXCAD's "Unused primitive (type: Box) detected in property:
# ..." warning fires for port0_metal/port1_metal on EVERY real Stage B run,
# for a structural reason, not a defect -- Stage B's M3 topology fix (module
# docstring "Port topology") gives the thru line's own trace box (property
# "copper") a span EXACTLY feed_x0..feed_x1, which fully covers each port's
# own SHORTER metal box ([start, start+port_w]); CSXCAD's own priority-based
# rasterizer resolves the overlap to the higher-priority primitive (both are
# priority=10 in this script, and the trace is added to the CSX tree first),
# so the port's own box never contributes a rasterized cell of its own and
# CSXCAD reports it "unused". This is a REPORTING artifact of two PEC
# primitives sharing the same physical footprint, not a missing-conductor
# defect -- the conduction proof is |S21| ~= 1.0 band-wide (run-1: 0.9985 to
# 1.0087, self_consistency_openems/passivity BOTH passed), which would be
# impossible if the line were actually open-circuited at either port.
# Scoped to the EXACT two property names below -- any OTHER "Unused
# primitive" (e.g. property: substrate!, a genuinely dropped conductor
# elsewhere) still trips the gate.
_ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES = ("port0_metal!", "port1_metal!")


def _scan_stdout_for_bad_patterns(log_text: str, label: str, *, check_truncation: bool = False) -> None:
    """Pre-solve fail-fast gate. ``check_truncation=True`` ADDS
    ``_TRUNCATION_STDOUT_PATTERNS`` to the scan -- callers must pass this
    ONLY for a REAL run's own captured log, never a smoke run's (see
    ``_TRUNCATION_STDOUT_PATTERNS``'s own docstring for why: smoke's tiny
    NrTS/EndCriteria budget makes those patterns fire by design there).
    """
    patterns = _BAD_STDOUT_PATTERNS + (_TRUNCATION_STDOUT_PATTERNS if check_truncation else ())
    hits = []
    for line in log_text.splitlines():
        line_lower = line.lower()
        if not any(p.lower() in line_lower for p in patterns):
            continue
        if "unused primitive" in line_lower and any(
            f"property: {prop}".lower() in line_lower
            for prop in _ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES
        ):
            continue  # allowlisted -- see _ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES docstring
        hits.append(line.strip())
    if hits:
        raise RuntimeError(
            f"[{label}] pre-solve mesh/port fail-fast gate tripped: openEMS "
            f"stdout/stderr contains {hits!r}. Aborting BEFORE the full NrTS "
            f"budget is spent."
        )


def _run_openems_capturing_stdout(fdtd, sim_path: str, *, threads: int) -> str:
    """Run openEMS while capturing its OS-level stdout AND stderr to a file
    we can grep.

    ``fdtd.Run()`` invokes the openEMS C++ binary; its stdout/stderr go to
    the process's OS-level fd 1 / fd 2, not Python's ``sys.stdout``/
    ``sys.stderr`` -- redirect BOTH fds, restoring both afterward even on
    error.

    GUARD CHANNEL-GAP FIX (run-1 forensics, 2026-08-04): the pre-fix version
    of this function redirected fd 1 ONLY. CSXCAD emits some port/mesh
    warnings on STDERR, not stdout -- e.g. "Unused primitive (type: Box)
    detected in property: port0_metal!/port1_metal!", which appears 4x in
    run-1's own VESSL-captured ``run.log`` (a terminal capture that sees
    BOTH fds; see ``validation/crossval/_20_msl_phase_referee_logs/
    20260804T070702Z_run.log`` lines 24-25/29-30) but was ABSENT from this
    function's own fd-1-only capture file -- ``_scan_stdout_for_bad_
    patterns`` never saw it and the pre-solve fail-fast gate below silently
    passed (rc=0) despite the warning having actually occurred. Both fds are
    now dup2'd into the SAME capture file so the scan below sees everything
    the terminal would have.
    """
    os.makedirs(sim_path, exist_ok=True)
    log_path = os.path.join(sim_path, "_openems_stdout.log")
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    with open(log_path, "w") as logf:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(logf.fileno(), stdout_fd)
        os.dup2(logf.fileno(), stderr_fd)
        try:
            fdtd.Run(sim_path, cleanup=True, verbose=1, numThreads=threads)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
    with open(log_path) as logf:
        return logf.read()


_END_CRITERIA_NOT_REACHED_TEXT = "reached before the end-criteria of"


def _log_indicates_truncation(real_log: str) -> bool:
    """D3 fix (run-1 forensics, review 2026-08-04): the single, pure,
    directly-testable source of truth for "did this REAL run hit its NrTS
    cap without reaching its own EndCriteria decay target" -- openEMS's own
    ``RunFDTD: Warning: Max. number of timesteps was reached before the
    end-criteria of ... was reached`` text (visible only after the fd-2
    capture fix; this exact phrase appears verbatim in run-1's committed
    ``run.log``, in the SMOKE portion of both stages, never the real
    portion). Both ``_run_stage_a_reproduce_gate``'s ``truncated_suspected``
    and ``_run_stage_b``'s ``end_criteria_not_reached``/``truncated`` call
    this SAME function on their own real (never smoke) captured log, rather
    than each maintaining its own copy of the substring or (the pre-fix
    Stage B behavior) an unrelated, structurally-unreachable probe-row-count
    comparison.
    """
    return _END_CRITERIA_NOT_REACHED_TEXT in real_log


def _check_excitation_and_trace(port, sim_path: str, label: str, *,
                                channel: str = "uf_inc") -> tuple[float, int]:
    """Scale-free excitation/trace guard (issue #465/PR #473, reaffirmed
    PR #547 -- see the module docstring PRECEDENT TICK-LIST item 8). No
    absolute floor on the launched-channel peak: only exact-zero/non-finite
    is a defect signature, because the healthy and broken populations
    overlap in openEMS's raw units (verified-good runs as small as
    ~3e-14 are recorded in ``rfx/interop/emitters/openems.py``'s own [D5]
    guard).
    """
    launched = np.asarray(getattr(port, channel), dtype=np.complex128)
    peak = float(np.max(np.abs(launched))) if launched.size else 0.0
    if not np.isfinite(peak) or peak == 0.0:
        raise RuntimeError(
            f"[{label}] openEMS port injected/received NO wave energy on "
            f"its own launched channel ({channel}={peak!r}): excitation "
            f"did not couple or the port never saw the wave."
        )
    n_samples = 0
    for name in list(getattr(port, "U_filenames", []) or []):
        trace_path = os.path.join(sim_path, name)
        if not os.path.exists(trace_path):
            continue
        raw = np.loadtxt(trace_path, comments="%")
        if not raw.size:
            continue
        raw2 = np.atleast_2d(raw)
        n_samples = max(n_samples, raw2.shape[0])
        peak_here = float(np.max(np.abs(raw2[:, 1])))
        if peak_here == 0.0:
            raise RuntimeError(
                f"[{label}] openEMS port voltage time trace is identically "
                f"zero: the excitation never entered the grid."
            )
    return peak, n_samples


def _non_physical_guard(s_mag: np.ndarray, label: str) -> None:
    peak = float(np.max(s_mag)) if s_mag.size else float("nan")
    if not np.all(np.isfinite(s_mag)) or peak > 2.0:
        raise RuntimeError(
            f"[{label}] non-physical/unstable |S| max={peak!r}: field blew "
            f"up or diverged."
        )


def _passivity_witness(s11: np.ndarray, s21: np.ndarray, label: str, *,
                       tol: float = 0.05, idx=None) -> dict:
    balance_full = np.abs(s11) ** 2 + np.abs(s21) ** 2
    balance = balance_full[idx] if idx is not None else balance_full
    max_balance = float(np.max(balance))
    passed = bool(max_balance <= 1.0 + tol)
    if not passed:
        raise RuntimeError(
            f"[{label}] passivity witness failed: max(|S11|^2+|S21|^2)="
            f"{max_balance:.4f} > {1.0 + tol} -- non-physical energy gain."
        )
    return {"balance": balance.tolist(), "max_balance": max_balance,
            "tol": tol, "passed": passed}


# ---------------------------------------------------------------------------
# STAGE A: faithful port of MSL_NotchFilter.py.
# ---------------------------------------------------------------------------
def _build_stage_a_notch_tutorial(ContinuousStructure, openEMS, MSLPort, *,
                                  nrts: int | None, end_criteria: float | None):
    """Build MSL_NotchFilter.py's geometry fresh (smoke + real run, matching
    the coax/thru precedent's "separate openEMS instances" pattern)."""
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

    resolution = _C0 / (A_F_MAX_HZ * np.sqrt(A_SUBSTRATE_EPR)) / A_UNIT / 50.0
    third_mesh = np.array([2.0 * resolution / 3.0, -resolution / 3.0]) / 4.0

    mesh.AddLine("x", [0.0])
    mesh.AddLine("x", A_MSL_WIDTH_UM / 2.0 + third_mesh)
    mesh.AddLine("x", -A_MSL_WIDTH_UM / 2.0 - third_mesh)
    mesh.SmoothMeshLines("x", resolution / 4.0)
    mesh.AddLine("x", [-A_MSL_LENGTH_UM, A_MSL_LENGTH_UM])
    mesh.SmoothMeshLines("x", resolution)

    mesh.AddLine("y", [0.0])
    mesh.AddLine("y", A_MSL_WIDTH_UM / 2.0 + third_mesh)
    mesh.AddLine("y", -A_MSL_WIDTH_UM / 2.0 - third_mesh)
    mesh.SmoothMeshLines("y", resolution / 4.0)
    mesh.AddLine("y", [-15.0 * A_MSL_WIDTH_UM, 15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM])
    mesh.AddLine("y", (A_MSL_WIDTH_UM / 2.0 + A_STUB_LENGTH_UM) + third_mesh)
    mesh.SmoothMeshLines("y", resolution)

    mesh.AddLine("z", np.linspace(0.0, A_SUBSTRATE_THICKNESS_UM, 5))
    mesh.AddLine("z", [3000.0])
    mesh.SmoothMeshLines("z", resolution)

    substrate = csx.AddMaterial("RO4350B", epsilon=A_SUBSTRATE_EPR)
    substrate.AddBox(
        [-A_MSL_LENGTH_UM, -15.0 * A_MSL_WIDTH_UM, 0.0],
        [A_MSL_LENGTH_UM, 15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM, A_SUBSTRATE_THICKNESS_UM],
    )

    pec = csx.AddMetal("PEC")
    port0 = MSLPort(
        csx, port_nr=1, metal_prop=pec,
        start=[-A_MSL_LENGTH_UM, -A_MSL_WIDTH_UM / 2.0, A_SUBSTRATE_THICKNESS_UM],
        stop=[0.0, A_MSL_WIDTH_UM / 2.0, 0.0],
        prop_dir="x", exc_dir="z", excite=-1.0,
        FeedShift=10.0 * resolution, MeasPlaneShift=A_MSL_LENGTH_UM / 3.0,
        priority=10,
    )
    port1 = MSLPort(
        csx, port_nr=2, metal_prop=pec,
        start=[A_MSL_LENGTH_UM, -A_MSL_WIDTH_UM / 2.0, A_SUBSTRATE_THICKNESS_UM],
        stop=[0.0, A_MSL_WIDTH_UM / 2.0, 0.0],
        prop_dir="x", exc_dir="z",
        MeasPlaneShift=A_MSL_LENGTH_UM / 3.0,
        priority=10,
    )

    pec.AddBox(
        [-A_MSL_WIDTH_UM / 2.0, A_MSL_WIDTH_UM / 2.0, A_SUBSTRATE_THICKNESS_UM],
        [A_MSL_WIDTH_UM / 2.0, A_MSL_WIDTH_UM / 2.0 + A_STUB_LENGTH_UM, A_SUBSTRATE_THICKNESS_UM],
        priority=10,
    )
    return fdtd, port0, port1


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


def _builder_body_after_kw(func) -> str:
    """The builder's source from its first statement (``kw = {}``) onward.

    Signature and docstring are excluded on purpose: they are where the
    parameters are named, and naming them is not a change to what is built.
    """
    import inspect

    src = inspect.getsource(func)
    marker = "\n    kw = {}\n"
    if marker not in src:
        raise RuntimeError(f"{func.__name__}: no '    kw = {{}}' first statement to anchor on")
    return src[src.index(marker) + 1:]


# ---------------------------------------------------------------------------
# The repository's one notch estimator, loaded by path (validation/ is not a
# package; it imports numpy only). Same loader the case's own test uses.
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    env = os.environ.get("RFX_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[4]


def _load_refined_extremum():
    path = _repo_root() / "validation" / "crossval" / "comparators" / "spectral_features.py"
    if not path.is_file():
        raise RuntimeError(
            f"shared notch estimator not found at {path} -- this script must run "
            f"from inside the repository (or set RFX_REPO_ROOT)"
        )
    spec = importlib.util.spec_from_file_location("_msl_notch_spectral_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.refined_extremum


# ---------------------------------------------------------------------------
# Pure-numpy geometry / mesh plan. No openEMS, no CSXCAD.
#
# WHAT IT CANNOT DO: CSXCAD's own SmoothMeshLines grades the transition between
# a fine and a coarse region, and it is a CSXCAD call. The estimate below only
# subdivides each gap into equal parts no wider than the target, so every line
# count it reports is a LOWER BOUND and every cell width it reports near a
# fine/coarse transition is an UPPER bound. Away from such a transition -- which
# is where the x boundaries and therefore the PML sit -- the two agree.
# ---------------------------------------------------------------------------
def _smooth_estimate(lines, max_res: float) -> np.ndarray:
    lines = np.unique(np.asarray(lines, dtype=float))
    out = [lines[0]]
    for a, b in zip(lines[:-1], lines[1:]):
        n = max(1, int(np.ceil((b - a) / max_res - 1e-9)))
        out.extend(np.linspace(a, b, n + 1)[1:])
    return np.unique(np.asarray(out, dtype=float))


def _plan(label: str, msl_length_um: float, resolution_factor: float) -> dict:
    """The geometry and mesh the named stage builds, as numbers."""
    resolution = _C0 / (A_F_MAX_HZ * np.sqrt(A_SUBSTRATE_EPR)) / A_UNIT / 50.0 * resolution_factor
    third_mesh = np.array([2.0 * resolution / 3.0, -resolution / 3.0]) / 4.0

    x = [0.0]
    x += list(A_MSL_WIDTH_UM / 2.0 + third_mesh)
    x += list(-A_MSL_WIDTH_UM / 2.0 - third_mesh)
    x = _smooth_estimate(x, resolution / 4.0)
    x = _smooth_estimate(np.concatenate([x, [-msl_length_um, msl_length_um]]), resolution)

    y = [0.0]
    y += list(A_MSL_WIDTH_UM / 2.0 + third_mesh)
    y += list(-A_MSL_WIDTH_UM / 2.0 - third_mesh)
    y = _smooth_estimate(y, resolution / 4.0)
    y = _smooth_estimate(
        np.concatenate([
            y,
            [-15.0 * A_MSL_WIDTH_UM, 15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM],
            (A_MSL_WIDTH_UM / 2.0 + A_STUB_LENGTH_UM) + third_mesh,
        ]),
        resolution,
    )

    n_sub = substrate_z_cells(resolution_factor)
    z = _smooth_estimate(
        np.concatenate([np.linspace(0.0, A_SUBSTRATE_THICKNESS_UM, n_sub + 1), [3000.0]]),
        resolution,
    )

    feed_shift = 10.0 * resolution
    measplane_shift = msl_length_um / 3.0
    # PML_8 on the x faces only: it eats the outermost 8 cells at each end.
    pml_lo = float(x[A_PML_CELLS]) if x.size > A_PML_CELLS else float(x[-1])
    pml_hi = float(x[-1 - A_PML_CELLS]) if x.size > A_PML_CELLS else float(x[0])
    feed_x = -msl_length_um + feed_shift
    measplane_x = -msl_length_um + measplane_shift
    stub_tip_y = A_MSL_WIDTH_UM / 2.0 + A_STUB_LENGTH_UM
    substrate_y_hi = 15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM

    return {
        "label": label,
        "msl_length_um": float(msl_length_um),
        "resolution_factor": float(resolution_factor),
        "resolution_um": float(resolution),
        "third_mesh_um": [float(v) for v in third_mesh],
        "box_mm": {
            "x": [-msl_length_um / 1e3, msl_length_um / 1e3],
            "y": [-15.0 * A_MSL_WIDTH_UM / 1e3, substrate_y_hi / 1e3],
            "z": [0.0, 3000.0 / 1e3],
        },
        "substrate_box_mm": {
            "x": [-msl_length_um / 1e3, msl_length_um / 1e3],
            "y": [-15.0 * A_MSL_WIDTH_UM / 1e3, substrate_y_hi / 1e3],
            "z": [0.0, A_SUBSTRATE_THICKNESS_UM / 1e3],
        },
        "stub_tip_y_mm": stub_tip_y / 1e3,
        "stub_tip_to_substrate_edge_mm": (substrate_y_hi - stub_tip_y) / 1e3,
        "mesh_lines_estimate": {"x": int(x.size), "y": int(y.size), "z": int(z.size)},
        "cells_estimate": int((x.size - 1) * (y.size - 1) * (z.size - 1)),
        "mesh_step_estimate_um": {
            "x_min": float(np.min(np.diff(x))), "x_max": float(np.max(np.diff(x))),
            "y_min": float(np.min(np.diff(y))), "y_max": float(np.max(np.diff(y))),
            "z_min": float(np.min(np.diff(z))), "z_max": float(np.max(np.diff(z))),
        },
        "substrate_z_cells_estimate": int(np.sum(
            (z >= -1e-9) & (z <= A_SUBSTRATE_THICKNESS_UM + 1e-9)) - 1),
        "substrate_z_step_estimate_um": float(A_SUBSTRATE_THICKNESS_UM / n_sub),
        "port0_start_x_mm": -msl_length_um / 1e3,
        "port1_start_x_mm": msl_length_um / 1e3,
        "feed_shift_mm": feed_shift / 1e3,
        "measplane_shift_mm": measplane_shift / 1e3,
        "feed_x_mm": feed_x / 1e3,
        "measplane_x_mm": measplane_x / 1e3,
        "pml_inner_face_x_mm_estimate": [pml_lo / 1e3, pml_hi / 1e3],
        "pml_depth_mm_estimate": (pml_lo + msl_length_um) / 1e3,
        "feed_inside_pml_estimate": bool(feed_x < pml_lo),
        "measplane_inside_pml_estimate": bool(measplane_x < pml_lo),
        "measplane_downstream_of_feed": bool(measplane_shift > feed_shift),
    }


STAGE_RUNGS = {
    "stage_a": ("stage_a", 1.0),
    "stage_b_coarse": ("stage_b_coarse", B_COARSE_RESOLUTION_FACTOR),
    "stage_b_mid": ("stage_b_mid", B_MID_RESOLUTION_FACTOR),
    "stage_b_fine": ("stage_b_fine", B_FINE_RESOLUTION_FACTOR),
}


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


def _print_plan(plan: dict) -> None:
    b, s = plan["box_mm"], plan["substrate_box_mm"]
    m, st = plan["mesh_lines_estimate"], plan["mesh_step_estimate_um"]
    print(f"  {plan['label']}  (MSL_length = {plan['msl_length_um']:.0f} um, "
          f"resolution factor {plan['resolution_factor']:g})")
    print(f"    resolution              {plan['resolution_um']:.3f} um   "
          f"third_mesh {plan['third_mesh_um'][0]:+.3f} / {plan['third_mesh_um'][1]:+.3f} um")
    print(f"    box (mm)                x [{b['x'][0]:+.3f}, {b['x'][1]:+.3f}]  "
          f"y [{b['y'][0]:+.3f}, {b['y'][1]:+.3f}]  z [{b['z'][0]:.3f}, {b['z'][1]:.3f}]")
    print(f"    substrate box (mm)      x [{s['x'][0]:+.3f}, {s['x'][1]:+.3f}]  "
          f"y [{s['y'][0]:+.3f}, {s['y'][1]:+.3f}]  z [{s['z'][0]:.3f}, {s['z'][1]:.3f}]")
    print(f"    stub tip y              {plan['stub_tip_y_mm']:.3f} mm  "
          f"({plan['stub_tip_to_substrate_edge_mm']:.3f} mm short of the substrate's y edge)")
    print(f"    mesh lines (estimate)   x {m['x']}  y {m['y']}  z {m['z']}   "
          f"-> cells >= {plan['cells_estimate']:,}")
    print(f"    cell size (estimate)    x {st['x_min']:.2f}-{st['x_max']:.2f}  "
          f"y {st['y_min']:.2f}-{st['y_max']:.2f}  z {st['z_min']:.2f}-{st['z_max']:.2f} um")
    print(f"    substrate z cells       {plan['substrate_z_cells_estimate']} of "
          f"{plan['substrate_z_step_estimate_um']:.2f} um across the 254 um board")
    print(f"    ports                   port0 start x {plan['port0_start_x_mm']:+.3f} mm, "
          f"port1 start x {plan['port1_start_x_mm']:+.3f} mm")
    print(f"    FeedShift               {plan['feed_shift_mm']:.3f} mm "
          f"-> feed at x {plan['feed_x_mm']:+.3f} mm")
    print(f"    MeasPlaneShift          {plan['measplane_shift_mm']:.3f} mm "
          f"-> measurement plane at x {plan['measplane_x_mm']:+.3f} mm")
    print(f"    x PML_8 inner face      {plan['pml_inner_face_x_mm_estimate'][0]:+.3f} mm "
          f"(depth {plan['pml_depth_mm_estimate']:.3f} mm, estimate)")
    print(f"    feed inside the PML     {plan['feed_inside_pml_estimate']}")
    print(f"    meas plane inside PML   {plan['measplane_inside_pml_estimate']}")
    print(f"    meas plane downstream of the feed  {plan['measplane_downstream_of_feed']}")


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
    print("NrTS / EndCriteria  openEMS library defaults (~1e9 / 1e-5) on every real "
          "pass; 200 / 0.0 on the smoke pass")
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
# The run
# ---------------------------------------------------------------------------
_VERSION_RE = re.compile(r"openEMS[^\n]*?(v\d[\w.\-+]*)", re.IGNORECASE)
_TIMESTEP_RE = re.compile(r"Timestep:\s*(\d+)")


def _openems_version(log_text: str) -> dict:
    """The solver's version, or a loud failure.

    Four sources, in this order, and the record names which one answered. A
    record that cannot say which openEMS produced it is not a reference, so
    "none of the four" raises instead of writing ``null``.
    """
    tried = []
    try:
        import openEMS as _pkg
        v = getattr(_pkg, "__version__", None)
        if v:
            return {"version": str(v), "source": "openEMS.__version__"}
        tried.append("openEMS.__version__ (attribute absent)")
    except Exception as exc:
        tried.append(f"openEMS.__version__ ({exc!r})")
    try:
        from openEMS.openEMS import openEMS as _cls
        v = getattr(_cls, "__version__", None)
        if v:
            return {"version": str(v), "source": "openEMS.openEMS.openEMS.__version__"}
        tried.append("openEMS.openEMS.openEMS.__version__ (attribute absent)")
    except Exception as exc:
        tried.append(f"openEMS.openEMS.openEMS.__version__ ({exc!r})")
    try:
        import importlib.metadata as md
        return {"version": md.version("openEMS"), "source": "importlib.metadata"}
    except Exception as exc:
        tried.append(f"importlib.metadata.version('openEMS') ({exc!r})")
    m = _VERSION_RE.search(log_text)
    if m:
        return {"version": m.group(1),
                "source": "openEMS's own startup banner in the captured log"}
    tried.append("the solver's startup banner in the captured log (no match)")
    raise RuntimeError(
        "the container reports NO openEMS version. Tried, in order: "
        + "; ".join(tried)
        + ". A reference record that cannot name the solver that produced it is "
          "not a reference -- refusing to write one."
    )


def _timesteps_executed(log_text: str):
    hits = [int(m.group(1)) for m in _TIMESTEP_RE.finditer(log_text)]
    return max(hits) if hits else None


def _mesh_lines(fdtd):
    """The x/y/z lines CSXCAD actually built, in the CSX unit (um)."""
    try:
        grid = fdtd.GetCSX().GetGrid()
        return {ax: np.asarray(grid.GetLines(ax), dtype=float) for ax in ("x", "y", "z")}
    except Exception:
        return None


def _mesh_realized(lines) -> dict:
    """Summary of the realized mesh. Every number here was read back, not declared."""
    if lines is None:
        return {"error": "CSXCAD did not return its grid lines"}
    out: dict = {}
    cells = 1
    for ax in ("x", "y", "z"):
        ln = lines[ax]
        steps = np.diff(ln)
        out[ax] = {
            "n_lines": int(ln.size),
            "min_um": float(ln.min()), "max_um": float(ln.max()),
            "step_min_um": float(steps.min()) if steps.size else None,
            "step_max_um": float(steps.max()) if steps.size else None,
        }
        cells *= max(int(ln.size) - 1, 0)
    out["n_cells"] = int(cells)
    z = lines["z"]
    in_sub = z[(z >= -1e-9) & (z <= A_SUBSTRATE_THICKNESS_UM + 1e-9)]
    out["substrate_z_cells_realized"] = int(in_sub.size - 1) if in_sub.size else 0
    out["substrate_top_on_a_line"] = bool(
        in_sub.size and abs(in_sub.max() - A_SUBSTRATE_THICKNESS_UM) < 1e-6)
    return out


def _nearest_line(lines_x, target_um: float) -> tuple:
    """(nearest realized x line, |target - line|) in um."""
    i = int(np.argmin(np.abs(lines_x - target_um)))
    return float(lines_x[i]), float(abs(lines_x[i] - target_um))


def _port_declared_and_snap(lines_x, *, start_x_um: float, direction: float,
                            feed_shift_um: float, measplane_shift_um: float,
                            port_obj=None) -> dict:
    """What the script declared for this port, and where the grid put it.

    The ``*_declared_mm`` fields are the declared numbers. The ``*_nearest_line_mm``
    / ``*_snap_um`` fields are measured against the x lines CSXCAD built: the
    distance is how far the plane had to move to reach a line.
    """
    feed_x = start_x_um + direction * feed_shift_um
    meas_x = start_x_um + direction * measplane_shift_um
    out = {
        "start_x_declared_mm": start_x_um / 1e3,
        "prop_direction": "+x" if direction > 0 else "-x",
        "feed_shift_declared_mm": feed_shift_um / 1e3,
        "meas_plane_shift_declared_mm": measplane_shift_um / 1e3,
        "feed_plane_x_declared_mm": feed_x / 1e3,
        "meas_plane_x_declared_mm": meas_x / 1e3,
    }
    if lines_x is not None and np.size(lines_x):
        f_line, f_snap = _nearest_line(lines_x, feed_x)
        m_line, m_snap = _nearest_line(lines_x, meas_x)
        out.update({
            "feed_plane_nearest_line_mm": f_line / 1e3,
            "feed_plane_snap_um": f_snap,
            "meas_plane_nearest_line_mm": m_line / 1e3,
            "meas_plane_snap_um": m_snap,
        })
    else:
        out.update({"feed_plane_nearest_line_mm": None, "feed_plane_snap_um": None,
                    "meas_plane_nearest_line_mm": None, "meas_plane_snap_um": None})
    # What MSLPort itself reports after its own snap, kept separate from both the
    # declared numbers and the line measurement above.
    for attr, key in (("feed_shift", "port_object_feed_shift_mm"),
                      ("measplane_shift", "port_object_meas_plane_shift_mm")):
        try:
            out[key] = float(getattr(port_obj, attr)) * A_UNIT / 1e-3
        except Exception:
            out[key] = None
    return out


class StageFailure(RuntimeError):
    """A sanity gate fired. Carries what had been measured when it did.

    The precedent's own convention (its PRECEDENT TICK-LIST item 11: "partial
    per-bin data attached to a raised RuntimeError so a failing run still leaves
    numbers to inspect"). ``main`` writes it to ``<output stem>_FAILED.json``.
    """

    def __init__(self, message: str, *, stage: str, partial: dict, meta: dict):
        super().__init__(message)
        self.stage = stage
        self.partial = partial
        self.meta = meta


def _energy_summary(freqs_hz: np.ndarray, s11: np.ndarray, s21: np.ndarray) -> dict:
    """|S11|^2+|S21|^2 per bin, and where its maxima sit.

    Reported for the whole grid AND split at the band edge, so "the witness read
    X" is never a single number a reader cannot place.
    """
    energy = np.abs(s11) ** 2 + np.abs(s21) ** 2
    band = (freqs_hz >= WITNESS_BAND_HZ[0]) & (freqs_hz <= WITNESS_BAND_HZ[1])
    below_1 = freqs_hz < 1.0e9
    one_to_two = (freqs_hz >= 1.0e9) & (freqs_hz < WITNESS_BAND_HZ[0])

    def _max(mask) -> float:
        return float(np.max(energy[mask])) if np.any(mask) else float("nan")

    return {
        "energy_sum": energy,
        "band_mask": band,
        "max_energy_sum_band": _max(band),
        "max_energy_sum_full": float(np.max(energy)) if energy.size else float("nan"),
        "max_energy_sum_below_band": {
            "0_1_ghz": _max(below_1),
            "1_2_ghz": _max(one_to_two),
        },
        "witness_band_ghz": list(NOTCH_BAND_GHZ),
        "passivity_tol": 1.0 + PASSIVITY_TOL,
    }


def _stage_a_gate_verdict(f_notch_hz: float, depth_db: float) -> dict:
    """The reproduce gate: the right frequency AND an actual notch.

    Pure arithmetic so the self-check can plant curves at it.
    """
    f_ok = bool(STAGE_A_GATE["f_notch_lo_hz"] <= f_notch_hz <= STAGE_A_GATE["f_notch_hi_hz"])
    depth_ok = bool(depth_db <= STAGE_A_MIN_DEPTH_DB)
    return {
        "measured_f_notch_hz": float(f_notch_hz),
        "measured_depth_db": float(depth_db),
        "analytic_f_notch_hz": float(F_NOTCH_AN_HZ),
        "deviation_pct": abs(f_notch_hz - F_NOTCH_AN_HZ) / F_NOTCH_AN_HZ * 100.0,
        "gate_band_hz": [STAGE_A_GATE["f_notch_lo_hz"], STAGE_A_GATE["f_notch_hi_hz"]],
        "gate_min_depth_db": STAGE_A_MIN_DEPTH_DB,
        "f_notch_ok": f_ok,
        "depth_ok": depth_ok,
        "passed": bool(f_ok and depth_ok),
    }


def _run_stage(*, label: str, sim_root: str, threads: int,
               msl_length_um: float, resolution_factor: float,
               refined_extremum) -> tuple[dict, dict]:
    """Smoke pass, real pass, CalcPort, the ported sanity gates, the notch.

    The sequence and every gate call is the precedent's
    ``_run_stage_a_reproduce_gate`` (validation/crossval/20_msl_phase_referee.py).
    Five things differ, all declared:

      1. the builder is selected by stage;
      2. a truncated real pass RAISES here instead of being reported as a flag;
      3. the notch comes from the repository's shared estimator over the witness
         band instead of a bare argmin over 0.5-1.5 x the analytic frequency;
      4. ``_passivity_witness`` runs on every stage (the precedent's Stage A
         runner does not call it at all) and is given the 2-7 GHz ``idx`` mask
         through the function's own existing argument -- the function is
         untouched;
      5. the arrays, the notch and the four energy-sum numbers are computed and
         stashed BEFORE ``_non_physical_guard`` and ``_passivity_witness`` run,
         so a gate that fires still leaves them behind. Only the order of pure
         post-processing moved; both guards still block the record.
    """
    ContinuousStructure, openEMS, MSLPort = _import_openems()

    sim_dir = os.path.join(sim_root, label)
    smoke_dir = os.path.join(sim_root, label + "_smoke")

    def build(*, nrts, end_criteria):
        if label == "stage_a":
            return _build_stage_a_notch_tutorial(
                ContinuousStructure, openEMS, MSLPort,
                nrts=nrts, end_criteria=end_criteria)
        return _build_notch_tutorial_at_rung(
            ContinuousStructure, openEMS, MSLPort,
            nrts=nrts, end_criteria=end_criteria,
            msl_length_um=msl_length_um, resolution_factor=resolution_factor)

    record: dict = {}
    meta: dict = {"stage": label}

    def fail(exc: Exception):
        return StageFailure(str(exc), stage=label, partial=record, meta=meta)

    try:
        smoke_fdtd, _p0, _p1 = build(nrts=200, end_criteria=0.0)
        smoke_log = _run_openems_capturing_stdout(smoke_fdtd, smoke_dir, threads=threads)
        _scan_stdout_for_bad_patterns(smoke_log, label + "_smoke")

        fdtd, port0, port1 = build(nrts=None, end_criteria=None)
        lines = _mesh_lines(fdtd)
        mesh = _mesh_realized(lines)
        meta["mesh_realized"] = mesh

        t0 = time.time()
        real_log = _run_openems_capturing_stdout(fdtd, sim_dir, threads=threads)
        _scan_stdout_for_bad_patterns(real_log, label, check_truncation=True)
        elapsed = time.time() - t0

        freqs = np.linspace(1.0e6, A_F_MAX_HZ, A_N_FREQS)
        port0.CalcPort(sim_dir, freqs)
        port1.CalcPort(sim_dir, freqs)

        inc_peak, n_samples = _check_excitation_and_trace(port0, sim_dir, label)

        if _log_indicates_truncation(real_log):
            raise RuntimeError(
                f"[{label}] SANITY GATE 'end criteria reached' FAILED: openEMS's own "
                f"'reached before the end-criteria of' warning is in this real pass's "
                f"captured log -- the run hit its NrTS cap before the field decayed, so "
                f"the spectrum is truncated and no record is written."
            )

        s11 = np.asarray(port0.uf_ref, dtype=np.complex128) / np.asarray(port0.uf_inc, dtype=np.complex128)
        s21 = np.asarray(port1.uf_ref, dtype=np.complex128) / np.asarray(port0.uf_inc, dtype=np.complex128)

        freqs_ghz = freqs / 1e9
        s21_mag = np.abs(s21)
        summary = _energy_summary(freqs, s11, s21)

        record.update({
            "freqs_ghz": freqs_ghz.tolist(),
            "s11_mag": np.abs(s11).tolist(),
            "s11_deg": np.degrees(np.angle(s11)).tolist(),
            "s21_mag": s21_mag.tolist(),
            "s21_deg": np.degrees(np.angle(s21)).tolist(),
            "energy_sum": summary["energy_sum"].tolist(),
            "max_energy_sum_band": summary["max_energy_sum_band"],
            "max_energy_sum_full": summary["max_energy_sum_full"],
            "max_energy_sum_below_band": summary["max_energy_sum_below_band"],
            "witness_band_ghz": summary["witness_band_ghz"],
            "passivity_tol": summary["passivity_tol"],
        })

        # Printed before the guards run, so the four numbers reach the log even
        # when the next line raises.
        print(f"  energy sum max(|S11|^2+|S21|^2): "
              f"band {NOTCH_BAND_GHZ[0]:.1f}-{NOTCH_BAND_GHZ[1]:.1f} GHz "
              f"{summary['max_energy_sum_band']:.4f} | "
              f"full grid {summary['max_energy_sum_full']:.4f} | "
              f"0-1 GHz {summary['max_energy_sum_below_band']['0_1_ghz']:.4f} | "
              f"1-2 GHz {summary['max_energy_sum_below_band']['1_2_ghz']:.4f} "
              f"(tol {summary['passivity_tol']:.2f}, judged on the band)", flush=True)

        # Inside the guarded block on purpose: a container that reports no
        # openEMS version refuses the record, but the arrays measured above
        # still reach the evidence file instead of a bare traceback (review,
        # 2026-09-22).
        openems_info = _openems_version(real_log)

        try:
            notch = refined_extremum(freqs_ghz, s21_mag,
                                     NOTCH_BAND_GHZ[0], NOTCH_BAND_GHZ[1], transform="log")
            record["notch"] = {
                "bin_f_ghz": float(notch["bin_f"]),
                "refined_f_ghz": float(notch["refined_f"]),
                "depth_db": float(notch["depth_db"]),
                "sub_bin_shift_bins": float(notch["sub_bin_shift"]),
                "bin_width_ghz": float(notch["bin_width"]),
                "band_ghz": list(NOTCH_BAND_GHZ),
                "estimator": "validation/crossval/comparators/spectral_features.py::"
                             "refined_extremum, transform='log'",
            }
        except Exception as exc:
            record["notch"] = {"error": repr(exc)}

        _non_physical_guard(np.abs(s11), label + "_s11")
        _non_physical_guard(np.abs(s21), label + "_s21")
        # The witness function is byte-identical to the precedent's; the band is
        # supplied through its own idx argument.
        _passivity_witness(s11, s21, label,
                           tol=PASSIVITY_TOL, idx=summary["band_mask"])
    except RuntimeError as exc:
        raise fail(exc) from exc

    resolution_um = _C0 / (A_F_MAX_HZ * np.sqrt(A_SUBSTRATE_EPR)) / A_UNIT / 50.0 * resolution_factor
    meta.update({
        "resolution_um": resolution_um,
        "resolution_factor": float(resolution_factor),
        "msl_length_um": float(msl_length_um),
        "mesh_realized": mesh,
        "box_mm": {
            "x": [-msl_length_um / 1e3, msl_length_um / 1e3],
            "y": [-15.0 * A_MSL_WIDTH_UM / 1e3,
                  (15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM) / 1e3],
            "z": [0.0, 3.0],
        },
        "port0": _port_declared_and_snap(
            None if lines is None else lines["x"],
            start_x_um=-msl_length_um, direction=+1.0,
            feed_shift_um=10.0 * resolution_um,
            measplane_shift_um=msl_length_um / 3.0, port_obj=port0),
        "port1": _port_declared_and_snap(
            None if lines is None else lines["x"],
            start_x_um=+msl_length_um, direction=-1.0,
            feed_shift_um=10.0 * resolution_um,
            measplane_shift_um=msl_length_um / 3.0, port_obj=port1),
        "substrate_z_cells_declared": substrate_z_cells(resolution_factor),
        "timesteps_executed": _timesteps_executed(real_log),
        "end_criteria_reached": True,
        "excitation_energy_peak": inc_peak,
        "port_trace_samples": n_samples,
        "wall_time_s": round(elapsed, 1),
        "stdout_log_path": os.path.join(sim_dir, "_openems_stdout.log"),
        "smoke_stdout_log_path": os.path.join(smoke_dir, "_openems_stdout.log"),
        "openems": openems_info,
        "plan_estimate": _plan(label, msl_length_um, resolution_factor),
    })
    return record, meta


def _build_artifact(records: dict, stage_meta: dict, stage_a_gate: dict,
                    stages: list, *, failed_gate: str | None = None) -> dict:
    """The record, whole or partial. ``failed_gate`` marks the partial one."""
    first_meta = next((m for m in stage_meta.values() if m.get("openems")), None)
    artifact = {
        "meta": {
            "tool": "openEMS",
            "openems": (first_meta["openems"] if first_meta
                        else {"version": None, "source": "no stage reached the solver"}),
            "rfx_openems_commit": os.environ.get("RFX_OPENEMS_COMMIT"),
            "rfx_openems_image": os.environ.get("RFX_OPENEMS_IMAGE"),
            "comparability": (
                "The S11/S21 PHASES are referenced at the tutorial's own measurement "
                "planes, MeasPlaneShift = MSL_length/3 = 16.667 mm in from each port "
                "face on the tutorial's 50 mm arms, and are NOT comparable with a "
                "record taken on a different arm length; the MAGNITUDES are."
            ),
            "stage_a_min_depth_db": STAGE_A_MIN_DEPTH_DB,
            "tutorial_source": (
                f"{REPRODUCE_GATE_RECORD['tutorial']['repo']}/"
                f"{REPRODUCE_GATE_RECORD['tutorial']['path']} -- "
                f"{REPRODUCE_GATE_RECORD['tutorial']['attribution']}, fetched verbatim "
                f"{REPRODUCE_GATE_RECORD['tutorial']['fetched_verbatim_on']} via "
                f"{REPRODUCE_GATE_RECORD['tutorial']['fetched_via']}"
            ),
            "reproduce_gate_record": REPRODUCE_GATE_RECORD,
            "delta_list": DELTA_LIST,
            "boundary": A_BOUNDARY,
            "excitation": f"SetGaussExcite({A_F_MAX_HZ/2.0}, {A_F_MAX_HZ/2.0}) Hz",
            "nrts": "openEMS library default (~1e9) on every real pass; 200 on the smoke pass",
            "end_criteria": "openEMS library default (1e-5) on every real pass; 0.0 on the smoke pass",
            "calcport_grid": f"linspace(1e6, {A_F_MAX_HZ}, {A_N_FREQS})",
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
            "stage_a_gate": stage_a_gate,
            "reproduce_gate_ran": "stage_a" in stages,
            "stages_requested": stages,
            "stages": stage_meta,
            "produced_by": "tests/crossval/msl_notch_filter/reference/make_openems_reference.py",
            "ci_runs_this": False,
        },
        "stage_a": records.get("stage_a"),
        "stage_b_coarse": records.get("stage_b_coarse"),
        "stage_b_mid": records.get("stage_b_mid"),
        "stage_b_fine": records.get("stage_b_fine"),
        "run_id": None,
        "run_id_note": (
            "null by design. VESSL does not export VESSL_RUN_ID into the pod, so the "
            "job cannot write its own id; the submitter fills this field "
            "(scripts/vessl_submit.sh drops run_id.txt beside the record)."
        ),
    }
    if failed_gate is not None:
        artifact["failed_gate"] = failed_gate
        artifact["meta"]["record_is_partial"] = True
    return artifact


def _failed_output_path(out: Path) -> Path:
    return out.with_name(out.stem + "_FAILED" + out.suffix)


def _write(artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(artifact, fh, indent=1)


def _stages_for(stage: str) -> list:
    if stage == "A":
        return ["stage_a"]
    if stage == "B":
        return ["stage_b_coarse", "stage_b_mid", "stage_b_fine"]
    return ["stage_a", "stage_b_coarse", "stage_b_mid", "stage_b_fine"]


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
