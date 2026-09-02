#!/usr/bin/env python3
"""openEMS external referee for issue #498 -- the probe-fed microstrip
transition (lumped/wire feed -> MSL port) that ``tests/
test_mixed_port_sparam.py``'s #488 fixture measures in rfx.

SCOPE FENCE (same fence as ``validation/crossval/20_msl_phase_referee.py``
and ``scripts/diagnostics/openems_thru_referee/thru_openems.py``): this is a
COMPARATOR LEG. It builds and runs an INDEPENDENT openEMS model and reports
its own S-parameters. It does NOT import rfx and it does NOT run any rfx
simulation. rfx's side enters -- optionally -- as ONE committed JSON data
file (``--rfx-json``, the artifact of the #498/#517 refplane-instrumented
measurement), exactly as Stage B of the #490 referee reads its rfx fixture.
It brackets; it does not judge. It cannot say WHICH of rfx's own diagonals
is lying -- that is what F1-F3 of the predeclaration are for, which is why
this referee runs after them.

============================================================================
STAGE CONTRACT (the repo's external-comparator law, enforced in code)
============================================================================
STAGE 1  reproduce-gate.  TWO legs, because the DUT uses TWO port classes:
         A1  openEMS ``python/Tutorials/MSL_NotchFilter.py`` (the MSL-port
             leg), delegated to the ALREADY-VERIFIED faithful port in
             ``validation/crossval/20_msl_phase_referee.py``
             (``_run_stage_a_reproduce_gate``) so this script cannot drift
             from the run that produced the recorded number.
         A2  openEMS ``python/Tutorials/Simple_Patch_Antenna.py`` (the
             lumped-probe leg -- the DUT's feed IS a vertical lumped probe
             from the ground plane to the trace, i.e. the patch tutorial's
             own feed model).  Faithful port, precedent
             ``scripts/diagnostics/patch_tutorial_openems.py``, with the
             precedent's harminv readout REPLACED by the tutorial's own
             S11-dip readout (harminv would require importing rfx and
             break the scope fence -- see A2's record).
STAGE 2  the #488 probe-fed microstrip transition.  It REFUSES TO RUN
         unless Stage 1 passed: ``assert_stage1_gate_passed`` is called
         BEFORE any Stage-2 geometry is constructed, and ``--stage 2``
         without a Stage-1 PASS record (``--stage1-json``) exits 4 with no
         geometry built and no number written.  No rfx-vs-openEMS number
         may exist in an artifact whose ``REPRODUCE_GATE_RECORD`` is not
         ``status="RUN"`` and inside its gate.

``REPRODUCE_GATE_RECORD`` (below) is serialized into EVERY artifact this
script writes -- the ``20_msl_phase_referee.py`` pattern, chosen because
prose that lives only in a docstring never reaches the artifact.

============================================================================
STAGE 1 -- example name / documented number / reproduced value / log path
============================================================================
A1  example  : openEMS python/Tutorials/MSL_NotchFilter.py
    documented known-good number: quarter-wave open-stub notch
      F_NOTCH_AN = c0 / (4 * 12 mm * sqrt(eps_eff_HJ)) = 3.6871 GHz
      (RECOMPUTED in code, never copy-pasted -- see
      ``validation/crossval/20_msl_phase_referee.py::F_NOTCH_AN_HZ``)
    this repo's reproduced value: 3.6711 GHz, deviation 0.4364 %
    log path: validation/crossval/_20_msl_phase_referee_logs/
              20260804T070702Z_run.log   (git-tracked, PRESENT)
    VESSL run: 369367251705
    gate: 0.80*F_NOTCH_AN <= f_notch <= 1.05*F_NOTCH_AN AND not
          truncation-suspect

A2  example  : openEMS python/Tutorials/Simple_Patch_Antenna.py
    documented UPSTREAM number: "~7 dBi broadside" -- that is the ONLY
      upstream-published figure this repo records for the tutorial.
    this repo's reproduced value (VESSL run 369367246713, 2026-07-11,
      via scripts/diagnostics/patch_tutorial_openems.py):
      f_res 2.4221 GHz (harminv on port V, Q 20.1), S11 dip 2.4300 GHz at
      -27.8 dB, broadside D 6.79 dBi, stopped on EndCriteria=1e-4 at step
      8671 with energy -41.09 dB.
    log path: docs/research_notes/vessl_logs/
              patch_tutorial_openems_GOOD_369367246713.log -- LOCAL-ONLY.
      Verified 2026-09-01 from this worktree: ``ls docs/research_notes/
      vessl_logs/`` -> "No such file or directory". The number above is
      carried from the precedent script's own header, which marks the log
      "(local-only)". This referee's A2 record therefore states the log as
      MISSING-FROM-TREE rather than citing it as if it were tracked, and
      the A2 run writes its OWN log into the lane's artifact directory so
      the next reader has a tracked one.
    gate, deliberately split by provenance because the two halves are not
      the same kind of claim:
        (i)  UPSTREAM-ANCHORED   : broadside D >= 6.0 dBi   ("~7 dBi")
        (ii) REPO-INTERNAL LOCK  : |f_dip - 2.4300 GHz| / 2.4300 GHz
                                   <= 0.01  -- a regression lock on THIS
                                   repo's own earlier source-built run,
                                   NOT a reproduction of an upstream
                                   number.  Labelled as such in the
                                   record; both must hold to pass.
      (The precedent's 2.4221 GHz f_res is a harminv-on-port-V readout
      that needs ``rfx.harminv``; importing rfx here would break the scope
      fence, so the gate uses the S11 dip from the SAME run instead.)

============================================================================
STAGE 2 -- the DUT, and the REALIZED-vs-DECLARED board (the #723 lesson)
============================================================================
The rfx fixture DECLARES eps_r=3.66, h_sub=254 um, W_trace=600 um on a
dx=80 um uniform grid.  What it SOLVES is a different board, and the #490
referee's own run-1 was invalidated by exactly this class of mismatch
(#723: "run-1 was measured with Stage B on the DECLARED 254um board while
rfx solved its realized 300um one").  Measured here, on the real grid,
before writing a line of geometry (command and verbatim output in
``RFX_REALIZED_RECORD``):

    grid shape (117, 55, 19); pads x 8/8, y 8/8, z 0/8
    interior slices x[8:109] y[8:47] z[0:11]
    conductor_mask() non-zero at k = 4 ONLY, y nodes 24..30, x nodes 8..108
    eps_r > 1 at k = 0..3 (substrate), 1.0 above

  => REALIZED h_sub   = 4 cells = 320 um   (declared 254 um)
  => REALIZED trace   = node span 24..30 about the feed node 27
                      = 6 cell widths = 480 um (declared 600 um);
                        the cell-span reading of the same footprint is
                        7 cells = 560 um -- a +-1-cell ambiguity carried
                        as a DECLARED systematic, both numbers recorded.
  => REALIZED y_c     = node 27 = 1.52 mm (the declared 1.50 mm snaps)
  => REALIZED domain  = 8.00 x 3.04 x 0.80 mm
  => trace OPEN at x = 0 and x = 8.00 mm, with 8 absorber cells beyond
     each declared face and NO metal in the pad; the DIELECTRIC ***is***
     edge-replicated through the pad (measured: eps_r = 3.66 at x nodes
     0..7 and 109..116).

Consequence that the referee must state up front, because it sets the
anchor term in the predeclaration's budget B: rfx anchors its MSL port to
the Hammerstad-Jensen Z0 of the DECLARED board, 47.89479996289313 ohm
(W=600, h=254).  The HJ Z0 of the board it actually solves is
    W=480, h=320 -> 62.652 ohm      (eps_eff 2.77333)
    W=560, h=320 -> 57.463 ohm      (eps_eff 2.80448)
i.e. the realized line is 57.5-62.7 ohm against a 47.89 ohm anchor.
REPORTED, NEVER GATED, and it must not replace the analytic HJ anchor
anywhere in shipped code (predeclaration section 10).  It is stated here
because a |S22| comparison inside budget B inherits it.

REVIEWER BLOCKER B4, APPLIED.  The predeclaration's section 7.2 said
"trace from x = 0 to 8 mm ... PML on x/y and z_hi", which in openEMS runs
the conductor INTO the PML -- a MATCHED termination, not rfx's open end.
The load seen from the line on the -x side is {feed + 2.0 mm of line +
OPEN END at x=0}, and there is a 2.48 mm OPEN STUB beyond the MSL feed
plane at 5.52 mm; |S22| and the predeclaration's M2 both contain those
reflections.  This script therefore reproduces BOTH open ends: the metal
box spans EXACTLY x = 0 .. 8.00 mm and the mesh is extended by
``PML_CELLS`` cells beyond each declared face, so the conductor ends at
the absorber's inner face with nothing in the pad -- rfx's own topology,
cell-count for cell-count.  The PHYSICAL pad thickness differs
(rfx 8*80 um = 0.64 mm; openEMS 8*dx_oe = 0.40 mm at the dx=50 um
comparator mesh) and is recorded as a declared systematic.

MESH.  dx = 50 um is the COMPARATOR mesh, never 80 um.  Carried
``do_not_repeat`` (``scripts/diagnostics/build_msl_notch_openems_
comparison.py`` via the #490 record): "at dx=80 um the substrate is only
3.175 cells ... the openEMS MSL-port extraction is NON-PHYSICAL
(|S11|^2+|S21|^2 up to 8.9)"; "dx=50 um gives 5.08 substrate cells where
BOTH solvers are passive".  On the REALIZED 320 um board the substrate is
6 cells at dx=50 um and 4 cells at dx=80 um.  A dx=80 um leg is ALSO run
and REPORTED ONLY, with its passivity sum quoted verbatim, to quantify the
mesh's own contribution; it is never the comparator.

PORTS.  (i) an openEMS LUMPED port, ground -> trace, at x = 2.00 mm,
50 ohm, spanning the substrate height (the rfx wire feed's ``extent``);
(ii) an MSLPort whose ``start`` is the x = 5.52 mm feed plane with
``prop_dir`` pointing INTO the line (-x, the upstream MSL_NotchFilter.py
convention and rfx's own ``direction="-x"``), 50 ohm.

DE-EMBEDDING.  The MSL measurement stencil (``MeasPlaneShift``) is placed
at rfx's OWN probe-0 coordinate x = 4.72 mm -- 0.80 mm from the port's
start plane, 16 cells at dx=50 um, exactly on-grid.  ``CalcPort(
ref_plane_shift=...)`` is called on every run and its EFFECTIVE shift
(``ref_plane_shift - measplane_shift``) is RECORDED; on this on-grid
placement it is expected to measure exactly 0.0 -- a measured no-op,
reported, not skipped.  The lumped port's own reference plane is its gap
at x = 2.00 mm, which is rfx's own lw port-cell x; that referral is
therefore 0.0 by construction and is likewise recorded, not assumed.

============================================================================
WHAT IS COMPARED (always against rfx's S_raw, NEVER result.S)
============================================================================
  |S21| lumped -> MSL, de-embedded to x = 4.72 mm : PRIMARY absolute-
        magnitude comparison -- with the frame caveat in CANNOT_COMPARE[1].
  |S22| MSL-driven, de-embedded to x = 4.72 mm    : predeclared agreement
        within the predeclaration's budget B (computed FROM the rfx run,
        not fixed here).
  |S11| lumped-driven                             : REPORTED only.
  phase: each solver's arg(S21) against ITS OWN measured beta (self-
        consistency, the #490 lane's 3 deg budget from a +-4-cell plane-
        position allowance).  The RAW cross-solver phase difference and
        the implied plane error Delta_d = Delta_phi / beta are REPORTED
        and never gated.
rfx's own 3-substrate-cell staircase advisory is quoted as the expected
envelope; the cross-mesh gap (50 vs 80 um) is a declared systematic and a
disagreement inside it convicts nothing.  UNTIL THIS RUNS, THE MIXED
LANE'S ABSOLUTE |S| STAYS UNVALIDATED, and that sentence stays in the
lane's documentation unchanged.

WHAT CANNOT BE COMPARED: see ``CANNOT_COMPARE`` below -- it is printed by
``--dry-run``, printed at the end of every real run, and serialized into
every artifact.

============================================================================
USAGE
============================================================================
    # no openEMS needed -- prints the full stage plan + the geometry it
    # WOULD build, and the pure-numpy mesh/geometry self-check:
    python scripts/diagnostics/probe_fed_msl_openems_referee.py --dry-run
    python scripts/diagnostics/probe_fed_msl_openems_referee.py --self-check

    # VESSL only (scripts/vessl_probe_fed_msl_referee.yaml), openEMS present:
    ... --stage 1    --output .omx/probe-fed-msl-referee/stage1.json
    ... --stage 2    --stage1-json .omx/probe-fed-msl-referee/stage1.json   # refuses without it
    ... --stage both --output .omx/probe-fed-msl-referee/referee.json
    # STAGE=1|2|both is also read from the environment when --stage is absent.

EXIT CODES
    0 stage(s) ran and every self-check/gate passed
    1 a gate or a self-check FAILED (physics/instrument finding)
    2 openEMS python bindings not importable (this pod; VESSL-only script)
    3 config/layout error -- a script bug, not a physics finding
    4 STAGE 2 REFUSED: no Stage-1 PASS record (the stage contract firing)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

C0_M_S = 2.99792458e8
# CSXCAD drawing unit for Stage 2 (mm). ref_plane_shift and
# MeasPlaneShift are both expressed in it, never in metres.
_CSX_UNIT_M = 1.0e-3

# ---------------------------------------------------------------------------
# What cannot be compared -- stated BEFORE the run, serialized into every
# artifact, printed by --dry-run and at the end of every real run.
# ---------------------------------------------------------------------------
CANNOT_COMPARE: tuple[str, ...] = (
    "(1) THE LUMPED/WIRE DIAGONAL, as a port-model comparison. rfx's wire "
    "diagonal is a per-cell, PRE-INJECTION port-cell quantity with "
    "Z0c = Z0/n_live = 50/4 = 12.5 ohm; openEMS's lumped port is a single "
    "lumped resistor across a one-cell gap referenced to the full 50 ohm. "
    "Agreement or disagreement there is evidence about the FRAME, not about "
    "the port, and cannot settle #683/#764/#776/#778. This referee reports "
    "|S11| and never adjudicates it.",
    "(2) THE OFF-DIAGONAL'S ABSOLUTE NORMALIZATION, to the extent the frame "
    "question touches it. rfx's mixed lane takes the off-diagonal MAGNITUDE "
    "from the Poynting-flux channel and its PHASE from the wave channel, "
    "while openEMS reports one port-based S. Only the composed |S_ij| is "
    "comparable, never 'rfx's wave channel vs openEMS'. In particular, a "
    "|S21| disagreement close to sqrt(n_live) = 2.0 is the same FRAME "
    "question as (1) and is NOT a physics disagreement this referee can "
    "settle.",
    "(3) ANYTHING AFTER enforce_passivity. rfx's result.S is a joint SVD "
    "projection (~4.3x on this fixture); only S_raw is comparable. This "
    "script refuses to read an 'S' field from the rfx artifact.",
    "(4) CROSS-FAMILY ABSOLUTE PHASE. rfx's mixed lane mixes two reference-"
    "plane conventions across families (port cell vs de-embedded MSL probe "
    "plane) plus a component-mixing +-1 -- which is why its own reciprocity "
    "witness is magnitude-only. Only each solver's own arg(S21) referred to "
    "its OWN measured beta is compared; the raw cross-solver phase "
    "difference is reported, never gated.",
    "(5) WHICH OF rfx's OWN DIAGONALS IS LYING. An external referee cannot "
    "answer that -- F1-F3 of the predeclaration can, which is why this runs "
    "after them.",
    "(6) MESH CONVERGENCE. One mesh pair (50 um comparator + 80 um "
    "reported-only) is not a convergence study, and this run does not "
    "create one.",
    "(7) THE ABSOLUTE ANCHOR IMPEDANCE. rfx normalizes its MSL port to the "
    "Hammerstad-Jensen Z0 of the DECLARED board (47.89479996289313 ohm, "
    "W=600 um / h=254 um) while the board it actually solves is W=480 um "
    "(node span) / h=320 um, whose HJ Z0 is 62.652 ohm (57.463 ohm on the "
    "560 um cell-span reading). openEMS's MSLPort measures its own Z0 from "
    "the fields. The gap between those anchors is REPORTED, never gated, "
    "and it is part of the predeclaration's budget B -- not a defect this "
    "referee attributes to either solver.",
    "(8) THE ABSORBER PAD THICKNESS. rfx pads 8 cells x 80 um = 0.64 mm "
    "beyond each declared face; this script pads 8 cells x dx_oe = 0.40 mm "
    "at the comparator mesh. Cell COUNT matches, physical thickness does "
    "not. Declared systematic on the open-end reflections at x = 0 and "
    "x = 8.00 mm.",
)

# ---------------------------------------------------------------------------
# rfx-side geometry of record -- MEASURED, not assumed. The commands and
# their verbatim output are carried here so a reader never has to trust a
# remembered number, and so this script (which never imports rfx) can be
# audited against the tree that produced it.
# ---------------------------------------------------------------------------
RFX_REALIZED_RECORD: dict = {
    "fixture": (
        "tests/test_mixed_port_sparam.py::_base_sim/_add_feed/_add_msl "
        "(the committed #488 fixture), read-only -- this script does not "
        "touch it."
    ),
    "declared": {
        "eps_r": 3.66,
        "h_sub_m": 254e-6,
        "w_trace_m": 600e-6,
        "dx_m": 80e-6,
        "domain_m": [8e-3, 3e-3, 754e-6],
        "cpml_layers": 8,
        "boundary": "x=cpml, y=cpml, z=(lo pec, hi cpml)",
        "feed_x_m": 2.0e-3,
        "msl_feed_x_m": 5.5e-3,
        "y_c_m": 1.5e-3,
    },
    "measured_command": (
        "cd <local-worktree> && PYTHONPATH=<local-worktree> python3 -c "
        "\"<build _base_sim geometry verbatim>; g=sim._build_grid(); "
        "mats,_,_=sim._build_materials(g); cm=sim.conductor_mask(g); print(...)\" "
        "-- run 2026-09-01 on branch meas/498-517-mixed-referee @ 3038f845"
    ),
    "measured_output_verbatim": [
        "shape (117, 55, 19) pad_x 8 8 pad_y 8 8 pad_z 0 8",
        "interior (slice(8, 109, None), slice(8, 47, None), slice(0, 11, None))",
        "idx x=0: (8, 27, 0)",
        "idx x=8mm-eps: (108, 27, 0)",
        "idx feed 2mm: (33, 27, 0)",
        "idx 1.44mm: (26, 27, 0)",
        "idx 4.72mm: (67, 27, 0)",
        "idx 5.5mm: (77, 27, 0)",
        "metal k [4] y [24 25 26 27 28 29 30] x 8 108",
        "eps sub k idx where eps>1 at x=50,y=27: [0 1 2 3]",
        "eps[:,27,1] x profile at nodes [0,4,7,8,50,108,109,116]: "
        "[3.66 3.66 3.66 3.66 3.66 3.66 3.66 3.66]  (dielectric IS "
        "edge-replicated through the CPML pad; the conductor is NOT)",
    ],
    # Realized board -- the board rfx actually solves (#723 class).
    "realized": {
        "h_sub_m": 320e-6,                 # metal node k=4 over the PEC z_lo
        "n_z_sub_cells_rfx": 4,            # eps_r > 1 at k = 0..3
        "w_trace_node_span_m": 480e-6,     # nodes 24..30 = 6 cell widths
        "w_trace_cell_span_m": 560e-6,     # the +-1-cell alternative reading
        "y_c_m": 1.52e-3,                  # node 27 (declared 1.50 mm snaps)
        "trace_y_lo_m": 1.28e-3,           # node 24
        "trace_y_hi_m": 1.76e-3,           # node 30
        "trace_x_lo_m": 0.0,               # node 8   -- OPEN END
        "trace_x_hi_m": 8.0e-3,            # node 108 -- OPEN END
        "trace_thickness_cells": 1,
        "domain_m": [8.0e-3, 3.04e-3, 0.80e-3],
        "pad_cells": {"x_lo": 8, "x_hi": 8, "y_lo": 8, "y_hi": 8,
                      "z_lo": 0, "z_hi": 8},
        "pad_thickness_m": 0.64e-3,
        "dielectric_extends_into_pad": True,
        "conductor_extends_into_pad": False,
    },
    "planes_of_record_m": {
        "trace_open_end_lo": 0.0,
        "predeclared_minus_x_flux_plane": 1.44e-3,
        "box_minus_x_face": 1.76e-3,
        "lw_feed": 2.00e-3,
        "box_plus_x_face": 2.24e-3,
        "refplane_N": 2.80e-3,
        "refplane_2N": 3.60e-3,
        "msl_probe_2": 4.08e-3,
        "msl_probe_1": 4.40e-3,
        "msl_probe_0": 4.72e-3,
        "msl_feed_plane": 5.52e-3,
        "trace_open_end_hi": 8.00e-3,
    },
    "anchor": {
        "rfx_z0_hj_msl_ohm": 47.89479996289313,
        "rfx_anchor_board": "DECLARED W=600 um / h=254 um",
        "hj_z0_realized_node_span_ohm": 62.652,
        "hj_z0_realized_cell_span_ohm": 57.463,
        "note": (
            "REPORTED, NEVER GATED. Must not replace the analytic "
            "Hammerstad-Jensen anchor anywhere in shipped code "
            "(predeclaration section 10)."
        ),
    },
    "n_live_lw": 4,
    "rfx_freqs_hz": [1.00e9, 1.75e9, 2.50e9, 3.25e9, 4.00e9],
}

# ---------------------------------------------------------------------------
# REPRODUCE_GATE_RECORD -- serialized into EVERY artifact this script writes.
# ---------------------------------------------------------------------------
REPRODUCE_GATE_RECORD: dict = {
    "law": (
        "Before any rfx-vs-openEMS number exists, this script must imitate "
        "openEMS's own canonical example and reproduce its documented "
        "known-good number. A lumped/wire-to-MSL transition has no closed "
        "form, so there is no analytic escape hatch. Two legs are required "
        "because the DUT uses two port classes."
    ),
    "a1": {
        "leg": "MSL-port leg",
        "example": "openEMS python/Tutorials/MSL_NotchFilter.py",
        "upstream": {
            "repo": "thliebig/openEMS",
            "path": "python/Tutorials/MSL_NotchFilter.py",
            "verified_via": (
                "gh api repos/thliebig/openEMS/contents/python/Tutorials/"
                "MSL_NotchFilter.py (2026-08-04, recorded by the #490 lane)"
            ),
        },
        "implementation": (
            "DELEGATED to validation/crossval/20_msl_phase_referee.py::"
            "_run_stage_a_reproduce_gate -- the already-verified faithful "
            "port that produced the recorded number, loaded by path (the "
            "module name starts with a digit). Delegation, not a second "
            "hand-copy, so this leg cannot drift from the run it cites."
        ),
        "documented_check": (
            "Quarter-wave open-stub notch F_NOTCH_AN = c0/(4*stub_len*"
            "sqrt(eps_eff_HJ)) = 3.6871 GHz on the tutorial's own RO4350B "
            "board -- RECOMPUTED in the delegated module, never copy-pasted."
        ),
        "gate": "0.80*F_NOTCH_AN <= f_notch <= 1.05*F_NOTCH_AN AND not truncation-suspect",
        "recorded_reproduction": {
            "f_notch_hz": 3671100625.0,
            "f_notch_expected_hz": 3687100377.611141,
            "dev_pct": 0.4364433837294213,
            "vessl_run_id": "369367251705",
            "log_path": (
                "validation/crossval/_20_msl_phase_referee_logs/"
                "20260804T070702Z_run.log"
            ),
            "log_present_in_tree": True,
            "verified_on": "2026-08-04",
        },
        "status": "RECORDED",   # -> "RUN" only after this script runs the leg
    },
    "a2": {
        "leg": "lumped-probe leg (the DUT's feed IS this tutorial's feed model)",
        "example": "openEMS python/Tutorials/Simple_Patch_Antenna.py",
        "upstream": {
            "repo": "thliebig/openEMS",
            "path": "python/Tutorials/Simple_Patch_Antenna.py",
            "documented_number": "'~7 dBi broadside' -- the ONLY upstream-published figure this repo records",
        },
        "implementation": (
            "Faithful port of scripts/diagnostics/patch_tutorial_openems.py "
            "(the repo precedent that produced the recorded numbers), with "
            "ONE declared delta: the precedent's resonance readout is "
            "harminv-on-port-V from rfx.harminv, which this script cannot "
            "use without breaking its scope fence, so the gate uses the "
            "tutorial's OWN S11-dip readout from the same run."
        ),
        "gate_upstream_anchored": "broadside directivity D >= 6.0 dBi ('~7 dBi')",
        "gate_repo_internal_lock": (
            "|f_s11_dip - 2.4300 GHz| / 2.4300 GHz <= 0.01 -- a REGRESSION "
            "LOCK on this repo's own earlier source-built run, explicitly "
            "NOT a reproduction of an upstream number. Both halves must "
            "hold."
        ),
        "recorded_reproduction": {
            "f_res_harminv_ghz": 2.4221,
            "q": 20.1,
            "f_s11_dip_ghz": 2.4300,
            "s11_min_db": -27.8,
            "broadside_d_dbi": 6.79,
            "stop": "EndCriteria=1e-4 at step 8671, energy -41.09 dB",
            "vessl_run_id": "369367246713",
            "log_path": (
                "docs/research_notes/vessl_logs/"
                "patch_tutorial_openems_GOOD_369367246713.log"
            ),
            "log_present_in_tree": False,
            "log_caveat": (
                "LOCAL-ONLY. Verified 2026-09-01 from this worktree: 'ls "
                "docs/research_notes/vessl_logs/' -> 'No such file or "
                "directory'. The precedent script's header marks it "
                "'(local-only)'; the predeclaration dropped that caveat and "
                "it is restored here. This lane writes its own A2 log into "
                "the artifact directory so the next reader has one."
            ),
            "verified_on": "2026-07-11",
        },
        "status": "RECORDED",
    },
    "do_not_repeat": (
        "scripts/diagnostics/build_msl_notch_openems_comparison.py header, "
        "carried in the #490 referee's own record: 'at dx=80 um the "
        "substrate is only 3.175 cells ... the openEMS MSL-port extraction "
        "is NON-PHYSICAL (|S11|^2+|S21|^2 up to 8.9)'; 'dx=50 um gives 5.08 "
        "substrate cells where BOTH solvers are passive'. Stage 2's "
        "COMPARATOR mesh is therefore dx=50 um; the dx=80 um leg is run and "
        "REPORTED ONLY, with its passivity sum quoted verbatim."
    ),
    "settling_evidence_protocol": (
        "External scripts get no rfx preflight, so the -40 dB settling "
        "witness is hand-ported the way the precedents do it: EndCriteria = "
        "1e-4 (= -40 dB) plus a POSITIVE CONTROL -- each stage runs a "
        "deliberately short SMOKE pass (NrTS=200, EndCriteria=0.0) whose "
        "max-timesteps warning MUST fire, and a real pass in which it MUST "
        "be ABSENT. Both appear in the committed run log. openEMS writes "
        "that warning to STDERR, so both fds are captured (the #490 lane's "
        "own GUARD CHANNEL-GAP fix)."
    ),
    "stage2_contract": (
        "Stage 2 constructs NO geometry and writes NO rfx-vs-openEMS number "
        "unless assert_stage1_gate_passed() accepts a Stage-1 record whose "
        "a1 and a2 legs both have status='RUN' and passed=True."
    ),
    "status": "NOT_RUN",
}

# ---------------------------------------------------------------------------
# Stage 2 constants -- the REALIZED board (see the module docstring).
# ---------------------------------------------------------------------------
B_EPS_R = 3.66
B_H_SUB_M = RFX_REALIZED_RECORD["realized"]["h_sub_m"]            # 320 um
B_W_TRACE_M = RFX_REALIZED_RECORD["realized"]["w_trace_node_span_m"]  # 480 um
B_TRACE_X_LO_M = 0.0
B_TRACE_X_HI_M = 8.0e-3
B_LY_M = RFX_REALIZED_RECORD["realized"]["domain_m"][1]           # 3.04 mm
B_LZ_M = RFX_REALIZED_RECORD["realized"]["domain_m"][2]           # 0.80 mm
B_Y_C_M = RFX_REALIZED_RECORD["realized"]["y_c_m"]                # 1.52 mm
B_FEED_X_M = 2.0e-3                    # lumped probe, rfx's lw port cell x
B_MSL_FEED_X_M = 5.52e-3               # MSL port start plane (rfx node 77)
B_MSL_MEAS_X_M = 4.72e-3               # rfx probe-0 plane -> MeasPlaneShift
B_PML_CELLS = 8                        # rfx cpml_layers=8, cell-count match
B_PORT_W_CELLS = 6                     # thru_openems.py / #490 lane pattern
B_DX_COMPARATOR_M = 50e-6              # the do_not_repeat mesh
B_DX_REPORTED_ONLY_M = 80e-6           # rfx's own dx -- reported, never compared
B_FEED_R_OHM = 50.0
B_F0_HZ = 2.5e9                        # rfx GaussianPulse(f0=2.5 GHz)
B_FC_HZ = 2.5e9                        # covers DC..5 GHz, rfx freq_max=5 GHz
B_NRTS_DEFAULT = 500000
B_END_CRITERIA = 1e-4                  # -40 dB, the settling witness
B_PHASE_SELF_CONSISTENCY_TOL_DEG = 3.0  # #490 lane's +-4-cell allowance
B_PASSIVITY_TOL = 0.05
B_SMOKE_NRTS = 200
B_SMOKE_END_CRITERIA = 0.0

# rfx's own 5 bins, always evaluated exactly; a dense grid is added for
# beta / phase context.
B_RFX_FREQS_HZ = np.asarray(RFX_REALIZED_RECORD["rfx_freqs_hz"], dtype=float)
B_DENSE_FREQS_HZ = np.linspace(0.5e9, 5.0e9, 46)


# ===========================================================================
# PURE ARITHMETIC -- no openEMS, no rfx. Everything below this line is unit-
# testable on this pod and is exercised by --dry-run / --self-check.
# ===========================================================================
class Stage1GateError(RuntimeError):
    """Stage 2 refused: Stage 1 did not run, or did not pass its gate."""


class ConfigError(RuntimeError):
    """A script/layout bug -- never a physics finding."""


def rfx_node_index(x_m: float, *, dx_m: float = 80e-6, pad: int = 8) -> int:
    """rfx's own x node index for a coordinate: ``pad + round(x/dx)``.

    Verified against the real grid (RFX_REALIZED_RECORD['measured_output_
    verbatim']): x=0 -> 8, 1.44 mm -> 26, 2.00 mm -> 33, 4.72 mm -> 67,
    5.50 mm -> 77 (i.e. 5.52 mm, the snap), 8.00 mm - eps -> 108.
    """
    return int(pad + round(x_m / dx_m))


def snap_shift_to_mesh(target_shift_m: float, dx_m: float) -> float:
    """Predict ``MSLPort.__init__``'s own mesh-line snap on a uniform,
    start-aligned x mesh.

    openEMS (``python/openEMS/ports.py``): ``meas_pos_idx = argmin(abs(
    prop_lines - measplane_pos))`` then ``measplane_shift = abs(start -
    prop_lines[meas_pos_idx])`` -- on a uniform grid whose ``start`` already
    sits on a mesh line this is exactly rounding to the nearest multiple of
    ``dx``. The real, openEMS-measured value is cross-checked against this
    at build time so a future mesh change fails loud, not silently.
    """
    return round(target_shift_m / dx_m) * dx_m


def refer_plane(s: np.ndarray, beta: np.ndarray, distance_m: float,
                n_planes: int = 1) -> np.ndarray:
    """Move a reference plane ``distance_m`` further ALONG the line, toward
    the DUT, across ``n_planes`` plane traversals.

    Convention, fixed here and unit-tested on planted data: a forward wave
    travels as ``exp(-j*beta*x)``, so a load at distance ``d`` behind the
    measurement plane reads ``s_meas = s_true * exp(-j*beta*d*n_planes)``
    (n_planes = 2 for a reflection, which traverses the length twice;
    n_planes = 1 per port for a transmission). De-embedding therefore
    MULTIPLIES by ``exp(+j*beta*d*n_planes)``.
    """
    s = np.asarray(s, dtype=np.complex128)
    beta = np.asarray(beta, dtype=np.complex128)
    return s * np.exp(1j * beta * float(distance_m) * int(n_planes))


def deembed_two_port(s11: np.ndarray, s21: np.ndarray, s22: np.ndarray,
                     s12: np.ndarray, *, beta1: np.ndarray, d1_m: float,
                     beta2: np.ndarray, d2_m: float) -> dict:
    """De-embed a full 2x2 by moving port 1's plane ``d1`` and port 2's
    plane ``d2`` toward the DUT. Reflections rotate twice at their own
    port; transmissions rotate once at each."""
    return {
        "s11": refer_plane(s11, beta1, d1_m, 2),
        "s22": refer_plane(s22, beta2, d2_m, 2),
        "s21": refer_plane(refer_plane(s21, beta1, d1_m, 1), beta2, d2_m, 1),
        "s12": refer_plane(refer_plane(s12, beta1, d1_m, 1), beta2, d2_m, 1),
    }


def implied_plane_error_m(dphi_rad: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """The plane offset a phase difference alone would map to,
    ``Delta_d = Delta_phi / beta`` -- the #490 lane's own reported
    diagnostic. A genuine fixed-position referral defect maps to a
    CONSTANT-SIGN Delta_d at every frequency; a sign flip across the band
    does not."""
    beta = np.asarray(beta, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(dphi_rad, dtype=np.float64) / beta


def phase_self_consistency_deg(s21: np.ndarray, beta: np.ndarray,
                               length_m: float) -> np.ndarray:
    """Deviation, in degrees, of ``arg(S21)`` from ``-beta*L`` -- each
    solver against ITS OWN measured beta (never against the other's)."""
    predicted = -np.asarray(beta, dtype=np.float64) * float(length_m)
    measured = np.unwrap(np.angle(np.asarray(s21, dtype=np.complex128)))
    predicted = np.unwrap(predicted)
    dev = measured - predicted
    dev = dev - np.round(dev / (2.0 * np.pi)) * 2.0 * np.pi
    return np.degrees(dev)


# A mesh line pair closer together than this fraction of dx is NOT a mesh
# feature -- it is one line that floating-point round-off split in two.
# ``np.arange`` reproduces a coordinate that lies exactly on its own grid
# only to within a ULP (~1e-19 m here), while the same coordinate written
# as a literal is exact, so ``np.unique`` on their union keeps BOTH and
# leaves a ZERO-WIDTH cell between them. openEMS then builds an operator
# with a singular cell and every field it writes is NaN -- VESSL run
# 369367257610 died exactly there, with the duplicate sitting ON the
# lumped probe's own feed plane (x = 2.00 mm). The smallest DELIBERATE
# spacing in this plan is the thirds-rule y line at dx/12, seven orders of
# magnitude above this floor, so nothing real can be merged by it.
_MESH_LINE_MERGE_FRAC = 1e-6


def merge_coincident_mesh_lines(lines, dx_m: float,
                                exact: np.ndarray | None = None) -> np.ndarray:
    """Sort ``lines`` and collapse ULP-separated duplicates into one line.

    ``exact`` names the coordinates whose EXACT value must survive a merge
    (the planes of record: the port feed planes, the measurement plane and
    the two open ends). Without that preference the surviving copy would be
    whichever of the two ``np.unique`` happened to sort first, which can be
    the ``arange`` round-off rather than the declared coordinate -- the
    plane would then sit ~1e-19 m off where the artifact says it does.

    Pure numpy; no openEMS, no rfx. Wired into ``openems_mesh_plan`` so the
    self-check, ``--dry-run`` and the builder all see the SAME mesh.
    """
    lines = np.unique(np.asarray(lines, dtype=float))
    if lines.size < 2:
        return lines
    tol = _MESH_LINE_MERGE_FRAC * float(dx_m)
    exact_arr = (np.unique(np.asarray(exact, dtype=float))
                 if exact is not None and len(np.atleast_1d(exact))
                 else np.empty(0, dtype=float))

    kept = [lines[0]]
    for value in lines[1:]:
        if value - kept[-1] < tol:
            # same line, twice. Prefer the declared coordinate if either
            # copy is one; otherwise keep the one already accepted.
            for candidate in (kept[-1], value):
                if exact_arr.size and np.any(np.abs(exact_arr - candidate) == 0.0):
                    kept[-1] = candidate
                    break
            continue
        kept.append(value)
    return np.asarray(kept, dtype=float)


def min_mesh_cell_m(lines) -> float:
    """Smallest cell in a line array -- 0.0 for a single line."""
    lines = np.asarray(lines, dtype=float)
    return float(np.min(np.diff(lines))) if lines.size > 1 else 0.0


# ---------------------------------------------------------------------------
# CSXCAD's OWN mesh smoother, ported verbatim so the PLAN can state the mesh
# the BUILDER will hand openEMS instead of a bound on it.
#
# Provenance: thliebig/CSXCAD, python/CSXCAD/SmoothMeshLines.py (Unique,
# CheckSymmetry, SnapToLines, SmoothRange, SmoothMeshLines). CSXCAD is not
# importable off the solver image, so the plan -- which must produce the same
# numbers on this pod and on VESSL -- uses this port. It is NOT trusted on its
# own: ``_build_stage2`` reads the REAL grid back after CSXCAD has smoothed it
# and raises if the two disagree, so a drift between this port and the
# installed CSXCAD fails LOUD at build time (seconds), never silently in a
# recorded number. Same predicted-vs-real discipline the MeasPlaneShift
# cross-check in this file already uses.
# ---------------------------------------------------------------------------
def _csx_unique(l, tol=1e-7):
    l = np.unique(l)
    dl = np.diff(l)
    idx = np.where(dl < np.mean(dl) * tol)[0]
    if len(idx) > 0:
        l = np.delete(l, idx)
    return l


def _csx_check_symmetry(lines):
    tolerance = 1e-10
    NP = len(lines)
    if NP <= 2:
        return 0
    line_range = lines[-1] - lines[0]
    center = 0.5 * (lines[-1] + lines[0])
    for n in range(int(NP / 2)):
        if abs((center - lines[n]) - (lines[-n - 1] - center)) > line_range * tolerance:
            return 0
    if NP % 2 == 1:
        if abs(lines[int(NP / 2)] - center) > line_range * tolerance:
            return 0
    return 2 if NP % 2 == 0 else 1


def _csx_snap_to_lines(lines, ref, tol=1e-10):
    lines = np.array(lines, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if len(ref) < 2:
        return lines
    abs_tol = (ref[-1] - ref[0]) * tol
    idx = np.clip(np.searchsorted(ref, lines), 1, len(ref) - 1)
    left, right = ref[idx - 1], ref[idx]
    nearest = np.where(lines - left <= right - lines, left, right)
    snap = np.abs(lines - nearest) <= abs_tol
    lines[snap] = nearest[snap]
    return lines


def _csx_smooth_range(start, stop, start_res, stop_res, max_res, ratio):
    assert ratio > 1
    rng = stop - start
    if rng < max_res and rng < start_res * ratio and rng < stop_res * ratio:
        return _csx_unique([start, stop])
    if start_res >= (max_res / ratio) and stop_res >= (max_res / ratio):
        N = np.ceil(rng / max_res).astype("int")
        tmp = np.linspace(start, stop, N + 1)
        return np.append(np.append(start, tmp[1:-1]), stop)

    def one_side_taper(start_res, ratio, max_res):
        res, pos, N = start_res, 0, 0
        while res < max_res and pos < rng:
            res *= ratio
            pos += res
            N += 1
        if pos > rng:
            l = np.zeros(N + 1)
            for n in range(N + 1):
                l[n] = np.sum(start_res * ratio ** np.arange(1, n + 1))
            return l * rng / pos
        _ratio = np.e ** ((np.log(max_res) - np.log(start_res)) / (N))
        l, pos, res = [0], 0, start_res
        for n in range(N):
            res *= _ratio
            pos += res
            l.append(pos)
        while pos < rng:
            pos += max_res
            l.append(pos)
        return np.array(l) * rng / l[-1]

    if start_res < (max_res / ratio) and stop_res >= (max_res / ratio):
        tmp = start + one_side_taper(start_res, ratio, max_res)
        return np.append(np.append(start, tmp[1:-1]), stop)
    if start_res >= (max_res / ratio) and stop_res < (max_res / ratio):
        tmp = np.sort(stop - one_side_taper(stop_res, ratio, max_res))
        return np.append(np.append(start, tmp[1:-1]), stop)

    pos1, N1, res = 0, 0, start_res
    while res < max_res:
        res *= ratio
        pos1 += res
        N1 += 1
    ratio1 = np.e ** ((np.log(max_res) - np.log(start_res)) / N1)
    pos1 = np.sum(start_res * ratio1 ** np.arange(1, N1 + 1))
    pos2, N2, res = 0, 0, stop_res
    while res < max_res:
        res *= ratio
        pos2 += res
        N2 += 1
    ratio2 = np.e ** ((np.log(max_res) - np.log(stop_res)) / N2)
    pos2 = np.sum(stop_res * ratio2 ** np.arange(1, N2 + 1))

    if (pos1 + pos2) < rng:
        l = [0]
        for n in range(1, N1 + 1):
            l.append(l[-1] + start_res * ratio1 ** n)
        r = [0]
        for n in range(1, N2 + 1):
            r.append(r[-1] + stop_res * ratio2 ** n)
        left = rng - pos1 - pos2
        N = int(np.ceil(left / max_res))
        for n in range(N):
            l.append(l[-1] + max_res)
        length = l[-1] + r[-1]
        c = _csx_unique(np.r_[np.array(l), length - np.array(r)])
        tmp = start + c * rng / length
        return np.append(np.append(start, tmp[1:-1]), stop)

    l, r = [0], [0]
    while l[-1] + r[-1] < rng:
        if start_res == stop_res:
            start_res *= ratio
            l.append(l[-1] + start_res)
            stop_res *= ratio
            r.append(r[-1] + start_res)
        elif start_res < stop_res:
            start_res *= ratio
            l.append(l[-1] + start_res)
        else:
            stop_res *= ratio
            r.append(r[-1] + start_res)
    length = l[-1] + r[-1]
    c = _csx_unique(np.r_[np.array(l), length - np.array(r)])
    tmp = start + c * rng / length
    return np.append(np.append(start, tmp[1:-1]), stop)


def _csxcad_smooth_mesh_lines(lines, max_res: float, ratio: float = 1.5):
    """Verbatim port of CSXCAD's ``SmoothMeshLines``. See the block comment."""
    out_l = _csx_unique(lines)
    orig_l = out_l
    sym = _csx_check_symmetry(out_l)
    if sym == 1:
        center = 0.5 * (out_l[-1] + out_l[0])
        out_l = out_l[:int(len(out_l) / 2) + 1]
    elif sym == 2:
        center = 0.5 * (out_l[-1] + out_l[0])
        out_l = out_l[:int(len(out_l) / 2)]
    dl = np.diff(out_l)
    while len(np.where(dl > max_res * (1 + 1e-10))[0]) > 0:
        N = len(out_l)
        dl[dl <= max_res] = np.max(dl) * 2
        idx = np.argmin(dl)
        dl = np.diff(out_l)
        start_res = dl[idx - 1] if idx > 0 else max_res
        stop_res = dl[idx + 1] if idx < len(dl) - 1 else max_res
        l = _csx_smooth_range(out_l[idx], out_l[idx + 1], start_res, stop_res,
                              max_res, ratio)
        out_l = _csx_unique(np.r_[out_l, l])
        dl = np.diff(out_l)
        if len(out_l) == N:
            break
    if sym == 1:
        return _csx_snap_to_lines(_csx_unique(np.r_[out_l, 2 * center - out_l[:-1]]), orig_l)
    elif sym == 2:
        l = _csx_smooth_range(out_l[-1], 2 * center - out_l[-1], dl[-1], dl[-1],
                              max_res, ratio)
        return _csx_snap_to_lines(_csx_unique(np.r_[out_l, l, 2 * center - out_l]), orig_l)
    return _csx_unique(out_l)


def _pml_depths_m(x, y, z) -> dict:
    """Physical depth of the B_PML_CELLS-cell absorber on each ABSORBING face.

    The PML is a CELL COUNT, so its depth follows whatever cells end up at the
    face -- it is NOT the pad the domain was sized with. z_lo is PEC and has
    no entry.
    """
    def lo(l):
        return float(l[B_PML_CELLS] - l[0]) if l.size > B_PML_CELLS else float("nan")

    def hi(l):
        return float(l[-1] - l[-1 - B_PML_CELLS]) if l.size > B_PML_CELLS else float("nan")

    return {"x_lo": lo(x), "x_hi": hi(x), "y_lo": lo(y), "y_hi": hi(y),
            "z_hi": hi(z)}


def _x_uniformity_report(lines, dx_m: float) -> dict:
    """Measured (never asserted) uniformity of a mesh axis.

    Replaces a hardcoded ``x_mesh_is_uniform_and_start_aligned: True``. The
    planes of record 4.72 mm and 5.52 mm are off the dx=50um lattice, so the
    x mesh is NOT uniform and never was; making it uniform would mean moving
    a plane of record, i.e. changing the DUT. Declared as a systematic.
    """
    d = np.diff(np.asarray(lines, dtype=float))
    if d.size == 0:
        return {"uniform": False, "reason": "fewer than two lines"}
    tol = 1e-9 * dx_m
    off = np.abs(d - dx_m) > tol
    return {
        "uniform": bool(not off.any()),
        "n_cells": int(d.size),
        "n_cells_off_nominal": int(off.sum()),
        "nominal_dx_m": float(dx_m),
        "min_cell_m": float(d.min()),
        "max_cell_m": float(d.max()),
        "off_nominal_cells_m": sorted({round(float(v), 15) for v in d[off]}),
        "cause": (
            "planes of record that are off the nominal lattice (the MSL "
            "measurement plane and the MSL port start); making x uniform "
            "would move a plane of record and change the DUT"),
    }


def openems_mesh_plan(dx_m: float) -> dict:
    """The Stage-2 mesh this script WOULD build at ``dx_m`` -- pure numpy,
    no openEMS. Used by --dry-run, by --self-check, and by the builder
    itself (wired, not a second hand-copy)."""
    pad = B_PML_CELLS * dx_m
    x0, x1 = B_TRACE_X_LO_M - pad, B_TRACE_X_HI_M + pad
    y0, y1 = 0.0 - pad, B_LY_M + pad
    z1 = B_LZ_M + pad

    x_lines = np.arange(x0, x1 + 0.5 * dx_m, dx_m)
    # Lines that MUST exist: the lumped feed, the MSL start plane, the
    # measurement plane, and both open ends (a port off the mesh is
    # CSXCAD's "Unused primitive" -> zero energy; the patch precedent's
    # own CRITICAL note).
    required_x = np.asarray([B_TRACE_X_LO_M, B_FEED_X_M, B_MSL_MEAS_X_M,
                             B_MSL_FEED_X_M, B_TRACE_X_HI_M])
    x_lines = merge_coincident_mesh_lines(
        np.concatenate([x_lines, required_x]), dx_m, exact=required_x)

    y_lines = np.arange(y0, y1 + 0.5 * dx_m, dx_m)
    trace_y_lo = B_Y_C_M - 0.5 * B_W_TRACE_M
    trace_y_hi = B_Y_C_M + 0.5 * B_W_TRACE_M
    third = np.asarray([2.0 * dx_m / 3.0, -dx_m / 3.0]) / 4.0
    required_y = np.asarray([B_Y_C_M, trace_y_lo, trace_y_hi])
    y_lines = merge_coincident_mesh_lines(
        np.concatenate([y_lines, required_y,
                        trace_y_lo + third, trace_y_hi + third]),
        dx_m, exact=required_y)

    # Substrate z lines: an explicit linspace across the REALIZED h_sub so
    # the substrate cell count is chosen, never left to an unguided arange.
    n_sub = max(int(round(B_H_SUB_M / dx_m)), 1)
    z_sub = np.linspace(0.0, B_H_SUB_M, n_sub + 1)
    z_air = np.arange(B_H_SUB_M, z1 + 0.5 * dx_m, dx_m)
    z_lines = merge_coincident_mesh_lines(
        np.concatenate([z_sub, z_air]), dx_m, exact=z_sub)

    # The builder smooths ONLY y (see _build_stage2); x and z go over as-is.
    # IN THE CSX DRAWING UNIT (mm), because that is the scale CSXCAD actually
    # smooths at -- _build_stage2 hands it mm and a mm max_res. The smoother is
    # NOT scale-invariant (np.ceil(rng/max_res) and Unique's relative tolerance
    # both bite differently at 1e-3 vs 1e0), and smoothing in metres here gives
    # 386 y lines where the builder gets 401.
    _u = 1.0 / _CSX_UNIT_M
    y_built = _csxcad_smooth_mesh_lines(
        y_lines * _u, (dx_m / 4.0) * _u, 1.4) / _u

    meas_shift_target = B_MSL_FEED_X_M - B_MSL_MEAS_X_M
    meas_shift_snapped = snap_shift_to_mesh(meas_shift_target, dx_m)

    lam_min_m = C0_M_S / (B_F0_HZ + B_FC_HZ) / np.sqrt(B_EPS_R)
    return {
        "dx_m": dx_m,
        "pml_cells": B_PML_CELLS,
        "pml_thickness_m": pad,
        "rfx_pad_thickness_m": RFX_REALIZED_RECORD["realized"]["pad_thickness_m"],
        "domain_with_pad_m": {"x": [x0, x1], "y": [y0, y1], "z": [0.0, z1]},
        "n_lines": {"x": int(x_lines.size), "y": int(y_lines.size),
                    "z": int(z_lines.size)},
        "n_cells_total": int((x_lines.size - 1) * (y_lines.size - 1)
                             * (z_lines.size - 1)),
        "n_substrate_cells": int(n_sub),
        "substrate_cell_dz_m": float(B_H_SUB_M / n_sub),
        "trace_y_lo_m": float(trace_y_lo),
        "trace_y_hi_m": float(trace_y_hi),
        "trace_x_span_m": [B_TRACE_X_LO_M, B_TRACE_X_HI_M],
        "open_end_to_pml_cells": B_PML_CELLS,
        "lumped_port_x_m": B_FEED_X_M,
        "msl_port_start_x_m": B_MSL_FEED_X_M,
        "msl_port_len_m": B_PORT_W_CELLS * dx_m,
        "msl_meas_plane_x_m": B_MSL_MEAS_X_M,
        "measplane_shift_target_m": meas_shift_target,
        "measplane_shift_predicted_m": meas_shift_snapped,
        "measplane_shift_residual_m": meas_shift_target - meas_shift_snapped,
        "ref_plane_shift_msl_m": meas_shift_target,
        "effective_calcport_shift_predicted_m": meas_shift_target - meas_shift_snapped,
        "ref_plane_shift_lumped_m": 0.0,
        "open_stub_beyond_msl_feed_m": B_TRACE_X_HI_M - B_MSL_FEED_X_M,
        "line_feed_to_meas_plane_m": B_MSL_MEAS_X_M - B_FEED_X_M,
        "lambda_min_in_substrate_m": float(lam_min_m),
        "cells_per_lambda_min": float(lam_min_m / dx_m),
        "y_lines_are_pre_smoothing": True,
        # THE MESH THE BUILDER ACTUALLY HANDS openEMS. The y lines above are
        # pre-smoothing; _build_stage2 then calls SmoothMeshLines("y", dx/4,
        # 1.4). Computed here with a verbatim port of CSXCAD's own smoother
        # (see _csxcad_smooth_mesh_lines) so --dry-run and --self-check state
        # the built mesh off the solver image too, and cross-checked against
        # the REAL grid at build time. VESSL 369367257610's record carried the
        # PLANNED cell count and the PLANNED y-PML depth; the solver saw
        # neither. Reported, never gated.
        "mesh_as_planned_built": {
            "smoother": "SmoothMeshLines('y', dx/4, 1.4) in _build_stage2",
            "smoother_source": "verbatim port of CSXCAD SmoothMeshLines.py",
            "max_cell_target_m": dx_m / 4.0,
            "n_lines": {"x": int(x_lines.size), "y": int(y_built.size),
                        "z": int(z_lines.size)},
            "n_cells_total": int((x_lines.size - 1) * (y_built.size - 1)
                                 * (z_lines.size - 1)),
            "min_cell_m": {"x": min_mesh_cell_m(x_lines),
                           "y": min_mesh_cell_m(y_built),
                           "z": min_mesh_cell_m(z_lines)},
            "max_cell_m": {"x": float(np.max(np.diff(x_lines))),
                           "y": float(np.max(np.diff(y_built))),
                           "z": float(np.max(np.diff(z_lines)))},
            "pml_depth_m": _pml_depths_m(x_lines, y_built, z_lines),
            "pml_cells": B_PML_CELLS,
            "note": (
                "The PML is a CELL COUNT, so smoothing y to dx/4 shrinks the y "
                "absorber DEPTH well below the pad the domain was sized with, "
                "while x and z keep the full pad. Reported, not gated -- "
                "changing the smoothing would change the mesh the #490 lane "
                "validated."),
        },
        # COMPUTED, never asserted: the required planes 4.72 mm and 5.52 mm
        # are off the dx=50um grid, so inserting them leaves short cell
        # pairs. A self-check that returns a literal is not a check.
        "x_mesh_uniformity": _x_uniformity_report(x_lines, dx_m),
        # Smallest cell on each axis. A value at round-off scale means a
        # plane of record got carried by TWO ULP-separated mesh lines and
        # openEMS's operator has a zero-width cell there (all-NaN fields);
        # ``merge_coincident_mesh_lines`` is what keeps it physical, and
        # this row is how a reader SEES that it did.
        "min_cell_m": {"x": min_mesh_cell_m(x_lines),
                       "y": min_mesh_cell_m(y_lines),
                       "z": min_mesh_cell_m(z_lines)},
        "x_lines_m": x_lines,
        "y_lines_m": y_lines,
        "z_lines_m": z_lines,
    }


def geometry_self_check(*, verbose: bool = True) -> dict:
    """Pure-numpy mesh/geometry self-check, printed before anything runs.

    It re-derives rfx's index map arithmetically and checks the RECORDED
    realized footprint for internal consistency; it deliberately does NOT
    reimplement rfx's rasterizer (that footprint is MEASURED -- see
    RFX_REALIZED_RECORD). Then it checks the openEMS mesh this script would
    build at both meshes.
    """
    out: dict = {"checks": [], "failures": []}

    def check(name: str, cond: bool, detail: str) -> None:
        out["checks"].append({"name": name, "ok": bool(cond), "detail": detail})
        if not cond:
            out["failures"].append(f"{name}: {detail}")

    dx_rfx = RFX_REALIZED_RECORD["declared"]["dx_m"]
    planes = RFX_REALIZED_RECORD["planes_of_record_m"]
    real = RFX_REALIZED_RECORD["realized"]

    # (a) every plane of record is an exact multiple of rfx's own dx, and
    #     maps to the measured node index.
    expect_idx = {"trace_open_end_lo": 8, "predeclared_minus_x_flux_plane": 26,
                  "lw_feed": 33, "refplane_N": 43, "refplane_2N": 53,
                  "msl_probe_2": 59, "msl_probe_1": 63, "msl_probe_0": 67,
                  "msl_feed_plane": 77, "trace_open_end_hi": 108}
    for name, x in planes.items():
        n = x / dx_rfx
        check(f"plane_on_grid[{name}]", abs(n - round(n)) < 1e-9,
              f"x = {x*1e3:.3f} mm = {n:.6f} cells of {dx_rfx*1e6:.0f} um")
        if name in expect_idx:
            got = rfx_node_index(x, dx_m=dx_rfx)
            check(f"plane_index[{name}]", got == expect_idx[name],
                  f"rfx node index {got} (measured {expect_idx[name]})")

    # (b) the realized board's internal consistency.
    check("realized_h_sub_is_integer_cells",
          abs(real["h_sub_m"] / dx_rfx - real["n_z_sub_cells_rfx"]) < 1e-9,
          f"h_sub {real['h_sub_m']*1e6:.0f} um = "
          f"{real['h_sub_m']/dx_rfx:.3f} cells (measured {real['n_z_sub_cells_rfx']})")
    check("realized_trace_is_symmetric_about_feed_node",
          abs((real["trace_y_lo_m"] + real["trace_y_hi_m"]) / 2.0
              - real["y_c_m"]) < 1e-12,
          f"trace y {real['trace_y_lo_m']*1e3:.2f}..{real['trace_y_hi_m']*1e3:.2f} mm "
          f"centred at {(real['trace_y_lo_m']+real['trace_y_hi_m'])/2*1e3:.2f} mm, "
          f"feed node y = {real['y_c_m']*1e3:.2f} mm")
    check("realized_w_is_node_span",
          abs((real["trace_y_hi_m"] - real["trace_y_lo_m"])
              - real["w_trace_node_span_m"]) < 1e-12,
          f"node span {(real['trace_y_hi_m']-real['trace_y_lo_m'])*1e6:.0f} um "
          f"(cell-span alternative {real['w_trace_cell_span_m']*1e6:.0f} um, "
          f"carried as a declared +-1-cell systematic)")
    check("realized_board_differs_from_declared",
          real["h_sub_m"] != RFX_REALIZED_RECORD["declared"]["h_sub_m"],
          f"realized h_sub {real['h_sub_m']*1e6:.0f} um vs declared "
          f"{RFX_REALIZED_RECORD['declared']['h_sub_m']*1e6:.0f} um -- the "
          f"#723 class; Stage 2 models the REALIZED board")

    # (c) the open ends B4 is about.
    check("trace_has_two_open_ends",
          real["trace_x_lo_m"] == 0.0 and real["trace_x_hi_m"] == 8.0e-3
          and not real["conductor_extends_into_pad"],
          "conductor spans exactly 0..8.00 mm with NO metal in the absorber "
          "pad; the openEMS model reproduces both open ends (blocker B4)")
    check("open_stub_beyond_msl_feed",
          abs((real["trace_x_hi_m"] - planes["msl_feed_plane"]) - 2.48e-3) < 1e-9,
          f"{(real['trace_x_hi_m']-planes['msl_feed_plane'])*1e3:.2f} mm of "
          f"open stub hangs beyond the MSL feed plane -- it is part of the "
          f"DUT and is in |S22|")

    # (d) the openEMS mesh, both legs.
    out["mesh"] = {}
    for label, dx in (("comparator_dx50um", B_DX_COMPARATOR_M),
                      ("reported_only_dx80um", B_DX_REPORTED_ONLY_M)):
        plan = openems_mesh_plan(dx)
        out["mesh"][label] = {k: v for k, v in plan.items()
                              if not k.endswith("_lines_m")}
        for nm, xr in (("lumped_feed", B_FEED_X_M),
                       ("msl_start", B_MSL_FEED_X_M),
                       ("meas_plane", B_MSL_MEAS_X_M),
                       ("open_end_lo", B_TRACE_X_LO_M),
                       ("open_end_hi", B_TRACE_X_HI_M)):
            on = float(np.min(np.abs(plan["x_lines_m"] - xr)))
            check(f"{label}:x_line_present[{nm}]", on < 1e-12,
                  f"nearest mesh line {on*1e9:.3f} nm away")
        check(f"{label}:measplane_snap_exact",
              abs(plan["measplane_shift_residual_m"]) < 1e-12,
              f"target {plan['measplane_shift_target_m']*1e3:.3f} mm, snapped "
              f"{plan['measplane_shift_predicted_m']*1e3:.3f} mm, effective "
              f"CalcPort shift {plan['effective_calcport_shift_predicted_m']*1e12:.1f} pm "
              f"(a measured no-op is expected here, reported not skipped)")
        if label.startswith("comparator"):
            check(f"{label}:substrate_cells_ge_5",
                  plan["n_substrate_cells"] >= 5,
                  f"{plan['n_substrate_cells']} substrate cells -- the "
                  f"do_not_repeat rule needs the comparator mesh above the "
                  f"3.175-cell non-physical regime")
        for ax in ("x", "y", "z"):
            mc = plan["min_cell_m"][ax]
            check(f"{label}:no_degenerate_cell[{ax}]",
                  mc >= _MESH_LINE_MERGE_FRAC * dx,
                  f"smallest {ax} cell {mc*1e9:.3f} nm "
                  f"({mc/dx:.4f} of dx) -- a round-off-scale cell means a "
                  f"plane of record is carried by two coincident mesh "
                  f"lines and openEMS's operator is singular there")
        check(f"{label}:cells_per_lambda_min_ge_10",
              plan["cells_per_lambda_min"] >= 10.0,
              f"{plan['cells_per_lambda_min']:.1f} cells per shortest "
              f"substrate wavelength")

    out["passed"] = not out["failures"]
    if verbose:
        print("=" * 78)
        print("MESH / GEOMETRY SELF-CHECK (pure numpy -- no openEMS, no rfx)")
        print("=" * 78)
        for c in out["checks"]:
            print(f"  [{'OK  ' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
        for label, m in out["mesh"].items():
            print(f"\n  -- {label} --")
            print(f"     PLANNED (y pre-smoothing) lines {m['n_lines']}  "
                  f"cells {m['n_cells_total']:,}  substrate "
                  f"{m['n_substrate_cells']} cells of "
                  f"{m['substrate_cell_dz_m']*1e6:.2f} um")
            b = m["mesh_as_planned_built"]
            print(f"     AS BUILT  (after SmoothMeshLines y) lines "
                  f"{b['n_lines']}  cells {b['n_cells_total']:,}  "
                  f"= {b['n_cells_total']/m['n_cells_total']:.2f}x the plan")
            pml = b["pml_depth_m"]
            print(f"     PML_8 depth per face: x {pml['x_lo']*1e6:.0f}/"
                  f"{pml['x_hi']*1e6:.0f} um, y {pml['y_lo']*1e6:.1f}/"
                  f"{pml['y_hi']*1e6:.1f} um (SHRUNK by the y smoothing), "
                  f"z_hi {pml['z_hi']*1e6:.0f} um; z_lo is PEC")
            print(f"     pad {m['pml_thickness_m']*1e3:.2f} mm "
                  f"({m['pml_cells']} cells) vs rfx's "
                  f"{m['rfx_pad_thickness_m']*1e3:.2f} mm "
                  f"({m['pml_cells']} cells) -- declared systematic")
            print(f"     MSL meas plane {m['msl_meas_plane_x_m']*1e3:.2f} mm, "
                  f"shift {m['measplane_shift_target_m']*1e3:.2f} mm, "
                  f"effective CalcPort shift "
                  f"{m['effective_calcport_shift_predicted_m']:.3e} m")
        print(f"\n  self-check passed: {out['passed']}")
        if out["failures"]:
            for f in out["failures"]:
                print(f"  FAILURE: {f}")
    return out


# ---------------------------------------------------------------------------
# THE STAGE CONTRACT
# ---------------------------------------------------------------------------
def stage1_leg_passed(leg: dict | None) -> bool:
    """One Stage-1 leg passes only when it actually RAN and cleared its
    gate. A recorded-but-not-run leg (status 'RECORDED') is NOT a pass."""
    if not isinstance(leg, dict):
        return False
    return bool(leg.get("status") == "RUN" and leg.get("passed") is True)


def assert_stage1_gate_passed(stage1: dict | None) -> None:
    """Refuse Stage 2 unless BOTH reproduce legs ran and passed.

    Called BEFORE any Stage-2 geometry is constructed. This is the
    external-comparator law in code: no rfx-vs-openEMS number may exist in
    an artifact whose reproduce gate is not RUN and inside its gate.
    """
    if stage1 is None:
        raise Stage1GateError(
            "Stage 2 REFUSED: no Stage 1 reproduce-gate record. Run "
            "--stage 1 (or --stage both) first, or pass its artifact with "
            "--stage1-json. openEMS's own canonical examples must be "
            "reproduced BEFORE the transition geometry is built."
        )
    if not isinstance(stage1, dict):
        raise Stage1GateError(
            f"Stage 2 REFUSED: Stage 1 record is not a dict "
            f"({type(stage1).__name__})."
        )
    missing = [k for k in ("a1", "a2") if not stage1_leg_passed(stage1.get(k))]
    if missing:
        detail = {k: {"status": (stage1.get(k) or {}).get("status")
                      if isinstance(stage1.get(k), dict) else None,
                      "passed": (stage1.get(k) or {}).get("passed")
                      if isinstance(stage1.get(k), dict) else None}
                  for k in ("a1", "a2")}
        raise Stage1GateError(
            f"Stage 2 REFUSED: Stage 1 reproduce-gate leg(s) {missing} did "
            f"not run-and-pass (need status='RUN' and passed=True on BOTH "
            f"legs; got {detail}). Stage A FAILED its reproduce-gate -- "
            f"skipping Stage B."
        )
    if stage1.get("passed") is not True:
        raise Stage1GateError(
            "Stage 2 REFUSED: Stage 1 record's own overall 'passed' is not "
            f"True (got {stage1.get('passed')!r})."
        )


# ---------------------------------------------------------------------------
# rfx artifact contract (data dependency only -- never a python import)
# ---------------------------------------------------------------------------
RFX_ARTIFACT_CONTRACT = (
    "The --rfx-json file is the #498/#517 measurement artifact. Required "
    "keys: 'freqs_hz' (list of bin frequencies) and 's_raw' (the UNPROJECTED "
    "S, shape [2][2][n_freqs] of [re, im] pairs, index 0 = lumped/wire "
    "family, index 1 = MSL family). An 'S' / post-passivity field is "
    "REFUSED: only S_raw is comparable (CANNOT_COMPARE item 3). Optional: "
    "'z0_hj_msl'. Absent --rfx-json, Stage 2 still runs and reports its own "
    "openEMS numbers, with comparison = null and status "
    "'openems-only (rfx artifact not supplied)'."
)


def load_rfx_artifact(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"rfx artifact not found: {path}. {RFX_ARTIFACT_CONTRACT}")
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"rfx artifact unreadable: {path} ({exc})") from exc
    for key in ("freqs_hz", "s_raw"):
        if key not in data:
            raise ConfigError(
                f"rfx artifact {path} is missing required key '{key}'. "
                f"{RFX_ARTIFACT_CONTRACT}"
            )
    s_raw = np.asarray(data["s_raw"], dtype=float)
    if s_raw.ndim != 4 or s_raw.shape[:2] != (2, 2) or s_raw.shape[3] != 2:
        raise ConfigError(
            f"rfx artifact {path} 's_raw' has shape {s_raw.shape}; expected "
            f"(2, 2, n_freqs, 2). {RFX_ARTIFACT_CONTRACT}"
        )
    freqs = np.asarray(data["freqs_hz"], dtype=float)
    if freqs.size != s_raw.shape[2]:
        raise ConfigError(
            f"rfx artifact {path}: len(freqs_hz)={freqs.size} but s_raw has "
            f"{s_raw.shape[2]} bins."
        )
    return {"freqs_hz": freqs,
            "s_raw": s_raw[..., 0] + 1j * s_raw[..., 1],
            "z0_hj_msl": data.get("z0_hj_msl"),
            "path": str(p)}


def compare_against_rfx(openems: dict, rfx: dict) -> dict:
    """Magnitude comparison at rfx's own bins. REPORTS deviations; it draws
    no verdict and it never gates -- the budget B is computed from the rfx
    run, not here."""
    f_oe = np.asarray(openems["freqs_hz"], dtype=float)
    rows = []
    for i, f in enumerate(np.asarray(rfx["freqs_hz"], dtype=float)):
        j = int(np.argmin(np.abs(f_oe - f)))
        if abs(f_oe[j] - f) > 1e-3:
            raise ConfigError(
                f"openEMS grid has no bin at rfx's {f/1e9:.4f} GHz "
                f"(nearest {f_oe[j]/1e9:.4f} GHz)."
            )
        s21_oe = abs(complex(openems["s21"][j]))
        s22_oe = abs(complex(openems["s22"][j]))
        s11_oe = abs(complex(openems["s11"][j]))
        s21_rfx = float(abs(rfx["s_raw"][1, 0, i]))
        s22_rfx = float(abs(rfx["s_raw"][1, 1, i]))
        s11_rfx = float(abs(rfx["s_raw"][0, 0, i]))
        rows.append({
            "f_hz": float(f),
            "abs_s21_openems": s21_oe, "abs_s21_rfx_raw": s21_rfx,
            "abs_s21_ratio_rfx_over_openems":
                (s21_rfx / s21_oe) if s21_oe else float("nan"),
            "abs_s21_diff": s21_rfx - s21_oe,
            "abs_s22_openems": s22_oe, "abs_s22_rfx_raw": s22_rfx,
            "abs_s22_diff": s22_rfx - s22_oe,
            "abs_s11_openems_reported_only": s11_oe,
            "abs_s11_rfx_raw_reported_only": s11_rfx,
        })
    ratios = np.asarray([r["abs_s21_ratio_rfx_over_openems"] for r in rows])
    return {
        "rows": rows,
        "abs_s22_max_diff": float(np.max([abs(r["abs_s22_diff"]) for r in rows])),
        "abs_s21_ratio_band": [float(np.nanmin(ratios)), float(np.nanmax(ratios))],
        "frame_note": (
            "A |S21| ratio near sqrt(n_live) = 2.0 is the FRAME question "
            "(#683/#764/#776/#778), not a physics disagreement -- see "
            "CANNOT_COMPARE item 2. No verdict is drawn here."
        ),
        "budget_note": (
            "The |S22| agreement budget B = |(Zc_meas - 47.8948)/(Zc_meas + "
            "47.8948)| + 0.01 is computed FROM THE rfx RUN (predeclaration "
            "section 6.2), not by this script. This row set is the input to "
            "that comparison, not the comparison itself."
        ),
        "not_compared": CANNOT_COMPARE,
    }


# ===========================================================================
# openEMS-dependent code -- deferred import, so this module stays importable
# (and testable) on a pod with no openEMS.
# ===========================================================================
def _ensure_openems_numpy_compat() -> None:
    for name, value in {"float": float, "int": int, "complex": complex,
                        "mat": np.matrix}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _import_openems():
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS
    from openEMS.ports import MSLPort
    return ContinuousStructure, openEMS, MSLPort


_BAD_STDOUT_PATTERNS = ("Unused primitive", "not on the mesh", "unused excitation")
_TRUNCATION_STDOUT_PATTERNS = (
    "Cutting to max number of timesteps",
    "max. number of timesteps is smaller than three times the excitation",
)
_END_CRITERIA_NOT_REACHED_TEXT = "reached before the end-criteria of"
# Same structural allowlist as the #490 lane: the trace box fully covers the
# MSL port's own shorter metal box, so CSXCAD reports the port metal
# "unused". A reporting artifact of two PEC primitives sharing a footprint,
# not a missing conductor -- any OTHER unused primitive still trips the gate.
_ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES = ("msl_port_metal!",)


def _log_indicates_truncation(real_log: str) -> bool:
    return _END_CRITERIA_NOT_REACHED_TEXT in real_log


def _scan_log_for_bad_patterns(log_text: str, label: str, *,
                               check_truncation: bool = False) -> None:
    patterns = _BAD_STDOUT_PATTERNS + (
        _TRUNCATION_STDOUT_PATTERNS if check_truncation else ())
    hits = []
    for line in log_text.splitlines():
        low = line.lower()
        if not any(p.lower() in low for p in patterns):
            continue
        if "unused primitive" in low and any(
                f"property: {prop}".lower() in low
                for prop in _ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES):
            continue
        hits.append(line.strip())
    if hits:
        raise RuntimeError(
            f"[{label}] pre-solve mesh/port fail-fast gate tripped: openEMS "
            f"stdout/stderr contains {hits!r}. Aborting BEFORE the full NrTS "
            f"budget is spent."
        )


def _run_openems_capturing_stdout(fdtd, sim_path: str, *, threads: int) -> str:
    """Run openEMS capturing BOTH fd 1 and fd 2 -- openEMS writes the
    max-timesteps warning (the settling positive control) to STDERR; the
    #490 lane's own GUARD CHANNEL-GAP fix."""
    os.makedirs(sim_path, exist_ok=True)
    log_path = os.path.join(sim_path, "_openems_stdout.log")
    stdout_fd, stderr_fd = sys.stdout.fileno(), sys.stderr.fileno()
    saved_out, saved_err = os.dup(stdout_fd), os.dup(stderr_fd)
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
            os.dup2(saved_out, stdout_fd)
            os.dup2(saved_err, stderr_fd)
            os.close(saved_out)
            os.close(saved_err)
    with open(log_path) as logf:
        return logf.read()


def _check_excitation(port, label: str, *, channel: str = "uf_inc") -> float:
    launched = np.asarray(getattr(port, channel), dtype=np.complex128)
    peak = float(np.max(np.abs(launched))) if launched.size else 0.0
    # nan and 0.0 are DIFFERENT causes and must not share a sentence.
    # LumpedPort.CalcPort pins ``Z_ref = self.R`` (a scalar) and base
    # Port.CalcPort then computes ``uf_inc = 0.5*(uf_tot + if_tot*Z_ref)``
    # -- a product with NO division -- so a port that genuinely integrated
    # no field yields exactly 0.0 and can never yield nan. A nan therefore
    # says the FIELD was already nan in the raw probe files (the solve
    # diverged or the operator was singular), and the port is not the
    # suspect. VESSL 369367257610 printed nan and sent the next reader
    # after the port; it was a zero-width mesh cell at the feed plane.
    if not np.isfinite(peak):
        raise RuntimeError(
            f"[{label}] openEMS port channel {channel} is {peak!r}: the FIELD "
            "was already non-finite in the raw probe files, so the solve "
            "diverged or the operator was singular. The port is NOT the "
            "suspect -- a lumped port cannot produce nan from a zero field "
            "(Z_ref is a scalar and uf_inc is a product, no division). Check "
            "the mesh for a degenerate cell and read _openems_stdout.log.")
    if peak == 0.0:
        raise RuntimeError(
            f"[{label}] openEMS port saw NO wave energy on its launched "
            f"channel ({channel}=0.0): excitation did not couple.")
    return peak


def _non_physical_guard(s_mag: np.ndarray, label: str) -> None:
    peak = float(np.max(s_mag)) if np.size(s_mag) else float("nan")
    if not np.all(np.isfinite(s_mag)) or peak > 2.0:
        raise RuntimeError(
            f"[{label}] non-physical/unstable |S| max={peak!r}.")


# ---------------------------------------------------------------------------
# STAGE 1
# ---------------------------------------------------------------------------
def _load_msl_phase_referee_module(repo_root: Path):
    """Load validation/crossval/20_msl_phase_referee.py by path (the module
    name starts with a digit, so it cannot be imported by name)."""
    import importlib.util
    path = repo_root / "validation" / "crossval" / "20_msl_phase_referee.py"
    if not path.exists():
        raise ConfigError(
            f"Stage A1 delegate missing: {path}. A1 is DELEGATED to the "
            f"already-verified faithful MSL_NotchFilter.py port rather than "
            f"hand-copied; without it this script cannot run its own "
            f"reproduce gate."
        )
    spec = importlib.util.spec_from_file_location("_msl_phase_referee_delegate", path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_stage1_a1(*, repo_root: Path, sim_root: str, threads: int) -> dict:
    mod = _load_msl_phase_referee_module(repo_root)
    t0 = time.time()
    res = mod._run_stage_a_reproduce_gate(
        sim_root=os.path.join(sim_root, "a1_msl_notch"), threads=threads)
    rec = dict(REPRODUCE_GATE_RECORD["a1"])
    rec.update({
        "status": "RUN",
        "passed": bool(res["passed"]),
        "f_notch_hz": res["f_notch_hz"],
        "f_notch_expected_hz": res["f_notch_expected_hz"],
        "f_notch_dev_pct": res["f_notch_dev_pct"],
        "notch_depth_db": res["notch_depth_db"],
        "truncated_suspected": res["truncated_suspected"],
        "elapsed_s": round(time.time() - t0, 1),
        "gate_band_hz": [mod.REPRODUCE_GATE_RECORD["gate"]["f_notch_lo_hz"],
                         mod.REPRODUCE_GATE_RECORD["gate"]["f_notch_hi_hz"]],
        "delegated_to": str(
            (repo_root / "validation/crossval/20_msl_phase_referee.py")),
    })
    return rec


def _build_stage1_a2(ContinuousStructure, openEMS, *, nrts: int,
                     end_criteria: float):
    """Faithful port of the Simple_Patch_Antenna tutorial, following
    scripts/diagnostics/patch_tutorial_openems.py (the repo precedent that
    produced the recorded numbers) -- including its two CRITICAL recipe
    points: AddEdges2Grid thirds-rule metal-edge mesh, and explicit mesh
    lines AT the feed so the lumped port is not an 'Unused primitive'."""
    import math
    eps0 = 8.8541878e-12
    unit = 1e-3
    patch_w, patch_l = 32.0, 40.0
    sub_eps_r, sub_thick = 3.38, 1.524
    sub_w = sub_l = 60.0
    feed_pos, feed_r = -6.0, 50.0
    sub_kappa = 1e-3 * 2 * math.pi * 2.45e9 * eps0 * sub_eps_r
    f0, fc = 2.0e9, 1.0e9
    sim_box = np.array([200.0, 200.0, 150.0])
    mesh_res = C0_M_S / (f0 + fc) / 1e-3 / 20.0
    sub_cells = 3

    fdtd = openEMS(NrTS=nrts, EndCriteria=end_criteria)
    fdtd.SetGaussExcite(f0, fc)
    fdtd.SetBoundaryCond(["MUR"] * 6)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)

    patch = csx.AddMetal("patch")
    patch.AddBox(priority=10, start=[-patch_w / 2, -patch_l / 2, sub_thick],
                 stop=[patch_w / 2, patch_l / 2, sub_thick])
    fdtd.AddEdges2Grid(dirs="xy", properties=patch, metal_edge_res=mesh_res / 2)
    sub = csx.AddMaterial("substrate", epsilon=sub_eps_r, kappa=sub_kappa)
    sub.AddBox(priority=0, start=[-sub_w / 2, -sub_l / 2, 0],
               stop=[sub_w / 2, sub_l / 2, sub_thick])
    gnd = csx.AddMetal("gnd")
    gnd.AddBox(priority=10, start=[-sub_w / 2, -sub_l / 2, 0],
               stop=[sub_w / 2, sub_l / 2, 0])
    fdtd.AddEdges2Grid(dirs="xy", properties=gnd)
    port = fdtd.AddLumpedPort(1, feed_r, [feed_pos, 0, 0],
                              [feed_pos, 0, sub_thick], "z", 1.0, priority=5)
    mesh.AddLine("x", [-sim_box[0] / 2, feed_pos, sim_box[0] / 2])
    mesh.AddLine("y", [-sim_box[1] / 2, 0.0, sim_box[1] / 2])
    mesh.AddLine("z", [-sim_box[2] / 3, sim_box[2] * 2 / 3])
    mesh.AddLine("z", np.linspace(0, sub_thick, sub_cells + 1))
    mesh.SmoothMeshLines("all", mesh_res, 1.4)
    nf2ff = fdtd.CreateNF2FFBox()
    return fdtd, port, nf2ff, (f0, fc, sub_thick, unit)


def run_stage1_a2(*, sim_root: str, threads: int) -> dict:
    ContinuousStructure, openEMS, _MSLPort = _import_openems()
    sim_dir = os.path.join(sim_root, "a2_patch_tutorial")
    smoke_dir = os.path.join(sim_root, "a2_patch_tutorial_smoke")

    smoke_fdtd, _p, _nf, _c = _build_stage1_a2(
        ContinuousStructure, openEMS, nrts=B_SMOKE_NRTS,
        end_criteria=B_SMOKE_END_CRITERIA)
    smoke_log = _run_openems_capturing_stdout(smoke_fdtd, smoke_dir, threads=threads)
    _scan_log_for_bad_patterns(smoke_log, "a2_smoke")
    smoke_warned = _log_indicates_truncation(smoke_log)

    fdtd, port, nf2ff, (f0, fc, sub_thick, unit) = _build_stage1_a2(
        ContinuousStructure, openEMS, nrts=30000, end_criteria=1e-4)
    t0 = time.time()
    real_log = _run_openems_capturing_stdout(fdtd, sim_dir, threads=threads)
    _scan_log_for_bad_patterns(real_log, "a2", check_truncation=True)
    elapsed = time.time() - t0
    truncated = _log_indicates_truncation(real_log)

    freqs = np.linspace(f0 - fc, f0 + fc, 401)
    port.CalcPort(sim_dir, freqs)
    _check_excitation(port, "a2")
    s11 = np.asarray(port.uf_ref, dtype=np.complex128) / np.asarray(
        port.uf_inc, dtype=np.complex128)
    _non_physical_guard(np.abs(s11), "a2_s11")
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-12))
    i_dip = int(np.argmin(s11_db))
    f_dip_ghz = float(freqs[i_dip]) / 1e9
    s11_min_db = float(s11_db[i_dip])

    theta = np.arange(0, 180.1, 2.0)
    phi = np.array([0.0, 90.0, 180.0, 270.0])
    nf = nf2ff.CalcNF2FF(sim_dir, [freqs[i_dip]], theta, phi,
                         center=[0, 0, sub_thick / 2 * unit])
    d_dbi = float(10.0 * np.log10(float(np.atleast_1d(nf.Dmax)[0])))

    lock_dev = abs(f_dip_ghz - 2.4300) / 2.4300
    upstream_ok = bool(d_dbi >= 6.0)
    lock_ok = bool(lock_dev <= 0.01)
    rec = dict(REPRODUCE_GATE_RECORD["a2"])
    rec.update({
        "status": "RUN",
        "passed": bool(upstream_ok and lock_ok and not truncated),
        "f_s11_dip_ghz": f_dip_ghz,
        "s11_min_db": s11_min_db,
        "broadside_d_dbi": d_dbi,
        "upstream_anchored_gate_ok": upstream_ok,
        "repo_internal_lock_dev": float(lock_dev),
        "repo_internal_lock_ok": lock_ok,
        "truncated_suspected": bool(truncated),
        "settling_positive_control_smoke_warned": bool(smoke_warned),
        "settling_positive_control_real_warned": bool(truncated),
        "elapsed_s": round(elapsed, 1),
        "log_written_by_this_run": os.path.join(sim_dir, "_openems_stdout.log"),
    })
    return rec


def run_stage1(*, repo_root: Path, sim_root: str, threads: int) -> dict:
    a1 = run_stage1_a1(repo_root=repo_root, sim_root=sim_root, threads=threads)
    print(f"  A1 (MSL_NotchFilter): f_notch = {a1['f_notch_hz']/1e9:.4f} GHz "
          f"vs {a1['f_notch_expected_hz']/1e9:.4f} GHz analytic, dev "
          f"{a1['f_notch_dev_pct']:.4f}% -> passed={a1['passed']}")
    a2 = run_stage1_a2(sim_root=sim_root, threads=threads)
    print(f"  A2 (Simple_Patch_Antenna): S11 dip {a2['f_s11_dip_ghz']:.4f} GHz "
          f"({a2['s11_min_db']:.1f} dB), broadside D {a2['broadside_d_dbi']:.2f} dBi "
          f"-> passed={a2['passed']}")
    return {"a1": a1, "a2": a2,
            "passed": bool(a1["passed"] and a2["passed"]),
            "law": REPRODUCE_GATE_RECORD["law"],
            "settling_evidence_protocol":
                REPRODUCE_GATE_RECORD["settling_evidence_protocol"]}


# ---------------------------------------------------------------------------
# STAGE 2
# ---------------------------------------------------------------------------
def _build_stage2(ContinuousStructure, openEMS, MSLPort, *, dx_m: float,
                  drive: str, nrts: int, end_criteria: float):
    """Build the #488 probe-fed microstrip transition on the REALIZED board.

    ``drive`` is 'lumped' or 'msl' -- openEMS needs one run per drive, the
    same two-drive structure the rfx mixed lane uses.
    """
    if drive not in ("lumped", "msl"):
        raise ConfigError(f"drive must be 'lumped' or 'msl', got {drive!r}")
    plan = openems_mesh_plan(dx_m)
    unit = _CSX_UNIT_M   # mm
    to_u = 1.0 / unit    # metres -> mm

    fdtd = openEMS(NrTS=nrts, EndCriteria=end_criteria)
    fdtd.SetGaussExcite(B_F0_HZ, B_FC_HZ)
    # x lo/hi, y lo/hi, z lo/hi -- PEC on z_lo (the ground plane), PML_8
    # everywhere else: rfx's BoundarySpec(x='cpml', y='cpml',
    # z=Boundary(lo='pec', hi='cpml')) with cpml_layers=8, cell for cell.
    fdtd.SetBoundaryCond(["PML_8", "PML_8", "PML_8", "PML_8", "PEC", "PML_8"])

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)
    dx_u = dx_m * to_u
    # NO ZERO-WIDTH CELL may reach openEMS. A plane of record carried by
    # two ULP-separated mesh lines makes the FDTD operator singular there
    # and every field openEMS writes is NaN -- which is how VESSL run
    # 369367257610 died (uf_inc=nan on the lumped probe's OWN feed plane,
    # x = 2.00 mm). merge_coincident_mesh_lines is the fix; this is the
    # build-time proof that it held, checked BEFORE a line is handed over.
    for _ax in ("x", "y", "z"):
        _mc = plan["min_cell_m"][_ax]
        if _mc < _MESH_LINE_MERGE_FRAC * dx_m:
            raise ConfigError(
                f"Stage-2 {_ax} mesh has a degenerate cell of {_mc:.6e} m "
                f"(dx = {dx_m:.6e} m): two coincident mesh lines. openEMS "
                f"builds a singular operator there and every field it "
                f"writes is NaN -- the uf_inc=nan failure of VESSL run "
                f"369367257610. Refusing to build the geometry.")
    mesh.AddLine("x", (plan["x_lines_m"] * to_u).tolist())
    mesh.AddLine("y", (plan["y_lines_m"] * to_u).tolist())
    mesh.AddLine("z", (plan["z_lines_m"] * to_u).tolist())
    # Only y is smoothed (the thirds-rule trace-edge lines would otherwise
    # leave a ~3x cell-ratio jump), exactly as the #490 lane does; the x
    # mesh is handed over as planned, and the MeasPlaneShift cross-check
    # below fails loud if openEMS's own snap disagrees with the plan.
    mesh.SmoothMeshLines("y", dx_u / 4.0, 1.4)

    # The mesh AS BUILT, read back from the object openEMS is about to solve.
    # The plan above is pre-smoothing on y and says so; this is the truth, and
    # it is what the artifact must carry. VESSL 369367257610's record claimed
    # the planned cell count and the planned y-PML depth, neither of which the
    # solver ever saw. Reported, never gated.
    _as_built = {}
    for _ax in ("x", "y", "z"):
        _l = np.asarray(mesh.GetLines(_ax), dtype=float) / to_u
        _d = np.diff(_l)
        _as_built[_ax] = {
            "n_lines": int(_l.size),
            "min_cell_m": float(_d.min()) if _d.size else None,
            "max_cell_m": float(_d.max()) if _d.size else None,
            "span_m": [float(_l[0]), float(_l[-1])] if _l.size else None,
            # The PML is a CELL COUNT, so its physical depth follows the cells
            # that end up at each face -- not the pad the domain was sized with.
            "pml_depth_lo_m": (float(_l[B_PML_CELLS] - _l[0])
                               if _l.size > B_PML_CELLS else None),
            "pml_depth_hi_m": (float(_l[-1] - _l[-1 - B_PML_CELLS])
                               if _l.size > B_PML_CELLS else None),
        }
    _as_built["total_cells"] = int(
        max(_as_built["x"]["n_lines"] - 1, 0)
        * max(_as_built["y"]["n_lines"] - 1, 0)
        * max(_as_built["z"]["n_lines"] - 1, 0))
    # plan["n_cells_total"] -- NOT "total_cells", a key the plan never
    # defined, which recorded a planned cell count of 0 in every artifact.
    _as_built["planned_total_cells"] = int(plan.get("n_cells_total") or 0)
    # The prediction in the plan is a PORT of CSXCAD's smoother; this is where
    # it gets checked against the real thing. A drift fails here, in seconds,
    # before the solve -- never silently inside a recorded number.
    _pred = plan["mesh_as_planned_built"]
    for _ax in ("x", "y", "z"):
        _got, _want = _as_built[_ax]["n_lines"], _pred["n_lines"][_ax]
        if _got != _want:
            raise ConfigError(
                f"Stage-2 {_ax} mesh as built has {_got} lines but the plan "
                f"predicted {_want}. The vendored CSXCAD smoother port "
                f"(_csxcad_smooth_mesh_lines) has drifted from the installed "
                f"CSXCAD, so the artifact's own mesh record would be wrong. "
                f"Refusing to run.")
    _as_built["prediction_matched"] = True
    _as_built["note"] = (
        "Read back from the built mesh after SmoothMeshLines. Compare with the "
        "plan's pre-smoothing y rows; a large ratio here is the #490-lane "
        "smoothing, not a defect, but the PML depths are the honest ones.")
    plan["mesh_as_built"] = _as_built

    x0 = plan["domain_with_pad_m"]["x"][0] * to_u
    x1 = plan["domain_with_pad_m"]["x"][1] * to_u
    y0 = plan["domain_with_pad_m"]["y"][0] * to_u
    y1 = plan["domain_with_pad_m"]["y"][1] * to_u
    h_sub = B_H_SUB_M * to_u

    # Dielectric extended THROUGH the pad -- rfx edge-replicates its
    # materials into the CPML pad (measured: eps_r = 3.66 at x nodes 0..7
    # and 109..116). The CONDUCTOR is not (next block).
    substrate = csx.AddMaterial("ro4350b", epsilon=B_EPS_R)
    substrate.AddBox([x0, y0, 0.0], [x1, y1, h_sub], priority=0)

    # BLOCKER B4: the trace spans EXACTLY 0..8.00 mm and stops at the
    # absorber's inner face -- BOTH open ends reproduced, nothing in the
    # pad. Running it into the PML would make a matched termination and
    # delete the two reflections |S22| and M2 are made of.
    trace_y_lo = plan["trace_y_lo_m"] * to_u
    trace_y_hi = plan["trace_y_hi_m"] * to_u
    trace = csx.AddMetal("trace")
    trace.AddBox([B_TRACE_X_LO_M * to_u, trace_y_lo, h_sub],
                 [B_TRACE_X_HI_M * to_u, trace_y_hi, h_sub + dx_u],
                 priority=10)

    # (i) the lumped probe: ground -> trace at x = 2.00 mm, spanning the
    # substrate height (the rfx wire feed's extent). Its own reference
    # plane IS rfx's lw port-cell x, so its referral is 0.0 by
    # construction -- recorded, not assumed.
    lumped = fdtd.AddLumpedPort(
        1, B_FEED_R_OHM,
        [B_FEED_X_M * to_u, B_Y_C_M * to_u, 0.0],
        [B_FEED_X_M * to_u, B_Y_C_M * to_u, h_sub],
        "z", 1.0 if drive == "lumped" else 0.0, priority=5)

    # (ii) the MSL port: start at the 5.52 mm feed plane, prop_dir INTO the
    # line (-x), MeasPlaneShift placing the stencil at rfx's own probe-0
    # coordinate 4.72 mm.
    port_w = plan["msl_port_len_m"] * to_u
    msl_metal = csx.AddMetal("msl_port_metal")
    msl = MSLPort(
        csx, port_nr=2, metal_prop=msl_metal,
        start=[B_MSL_FEED_X_M * to_u, trace_y_lo, h_sub],
        stop=[B_MSL_FEED_X_M * to_u - port_w, trace_y_hi, 0.0],
        prop_dir="x", exc_dir="z",
        excite=1.0 if drive == "msl" else 0.0,
        Feed_R=B_FEED_R_OHM,
        MeasPlaneShift=plan["measplane_shift_target_m"] * to_u,
        priority=10,
    )
    if abs(float(msl.start[0]) - B_MSL_FEED_X_M * to_u) > 1e-6:
        raise ConfigError(
            "MSL port start drifted from the MeasPlaneShift target "
            "(silent-coupling regression).")
    meas_real_m = float(msl.measplane_shift) * unit
    if abs(meas_real_m - plan["measplane_shift_predicted_m"]) > 1e-9:
        raise ConfigError(
            f"MSL port's REAL measplane_shift ({meas_real_m} m) does not "
            f"match the pure-arithmetic prediction "
            f"({plan['measplane_shift_predicted_m']} m) -- the uniform, "
            f"start-aligned x-mesh assumption snap_shift_to_mesh relies on "
            f"broke.")
    port_info = {
        "measplane_shift_target_m": plan["measplane_shift_target_m"],
        "measplane_shift_predicted_m": plan["measplane_shift_predicted_m"],
        "measplane_shift_real_m": meas_real_m,
        "ref_plane_shift_msl_m": plan["ref_plane_shift_msl_m"],
        "effective_calcport_shift_real_m":
            plan["ref_plane_shift_msl_m"] - meas_real_m,
        "ref_plane_shift_lumped_m": 0.0,
        "msl_meas_plane_x_m": B_MSL_MEAS_X_M,
        "lumped_plane_x_m": B_FEED_X_M,
    }
    return fdtd, lumped, msl, plan, port_info


def _run_stage2_leg(*, dx_m: float, sim_root: str, threads: int, nrts: int,
                    end_criteria: float, label: str) -> dict:
    ContinuousStructure, openEMS, MSLPort = _import_openems()
    freqs = np.unique(np.concatenate([B_DENSE_FREQS_HZ, B_RFX_FREQS_HZ]))

    # Settling positive control: the smoke pass MUST warn, the real pass
    # MUST NOT (run once per leg, on the lumped drive).
    smoke_fdtd, _l, _m, _p, _pi = _build_stage2(
        ContinuousStructure, openEMS, MSLPort, dx_m=dx_m, drive="lumped",
        nrts=B_SMOKE_NRTS, end_criteria=B_SMOKE_END_CRITERIA)
    smoke_log = _run_openems_capturing_stdout(
        smoke_fdtd, os.path.join(sim_root, f"{label}_smoke"), threads=threads)
    _scan_log_for_bad_patterns(smoke_log, f"{label}_smoke")
    smoke_warned = _log_indicates_truncation(smoke_log)

    results = {}
    port_info = None
    plan = None
    truncated_any = False
    t0 = time.time()
    for drive in ("lumped", "msl"):
        sim_dir = os.path.join(sim_root, f"{label}_{drive}")
        fdtd, lumped, msl, plan, port_info = _build_stage2(
            ContinuousStructure, openEMS, MSLPort, dx_m=dx_m, drive=drive,
            nrts=nrts, end_criteria=end_criteria)
        real_log = _run_openems_capturing_stdout(fdtd, sim_dir, threads=threads)
        _scan_log_for_bad_patterns(real_log, f"{label}_{drive}",
                                   check_truncation=True)
        truncated_any = truncated_any or _log_indicates_truncation(real_log)
        # ref_plane_shift is in the CSX DRAWING UNIT (mm here), the #490
        # lane's own convention -- and it is passed on EVERY run, never
        # skipped, so its effective value is measured rather than assumed.
        # ref_impedance is deliberately NOT passed: thru_openems.py's own
        # recorded trap ("do NOT pass ref_impedance as a scalar float --
        # that triggers a bug in the base Port.CalcPort when Z_ref is
        # array-valued").
        lumped.CalcPort(sim_dir, freqs)
        msl.CalcPort(sim_dir, freqs,
                     ref_plane_shift=port_info["ref_plane_shift_msl_m"] / _CSX_UNIT_M)
        driven = lumped if drive == "lumped" else msl
        _check_excitation(driven, f"{label}_{drive}")
        inc = np.asarray(driven.uf_inc, dtype=np.complex128)
        _nan = np.full(freqs.shape, np.nan, dtype=np.complex128)
        results[drive] = {
            "lumped_ref": np.asarray(lumped.uf_ref, dtype=np.complex128),
            "msl_ref": np.asarray(msl.uf_ref, dtype=np.complex128),
            "inc": inc,
            "beta": np.asarray(getattr(msl, "beta", _nan), dtype=np.complex128),
            # openEMS's OWN measured line impedance -- recorded next to
            # rfx's analytic HJ anchor, REPORTED and never gated (it must
            # not replace the analytic anchor anywhere in shipped code).
            "z0_msl": np.asarray(getattr(msl, "ZL", getattr(msl, "Z_ref", _nan)),
                                 dtype=np.complex128),
        }
    elapsed = time.time() - t0

    s11 = results["lumped"]["lumped_ref"] / results["lumped"]["inc"]
    s21 = results["lumped"]["msl_ref"] / results["lumped"]["inc"]
    s22 = results["msl"]["msl_ref"] / results["msl"]["inc"]
    s12 = results["msl"]["lumped_ref"] / results["msl"]["inc"]
    for nm, arr in (("s11", s11), ("s21", s21), ("s22", s22), ("s12", s12)):
        _non_physical_guard(np.abs(arr), f"{label}_{nm}")

    beta = np.real(results["lumped"]["beta"])
    line_len = plan["line_feed_to_meas_plane_m"]
    phase_dev = phase_self_consistency_deg(s21, beta, line_len)
    balance = np.abs(s11) ** 2 + np.abs(s21) ** 2
    passivity_max = float(np.max(balance))

    return {
        "label": label,
        "dx_m": dx_m,
        "role": ("COMPARATOR" if abs(dx_m - B_DX_COMPARATOR_M) < 1e-12
                 else "REPORTED ONLY -- never the comparator (do_not_repeat)"),
        "freqs_hz": freqs.tolist(),
        "s11": [[float(c.real), float(c.imag)] for c in s11],
        "s21": [[float(c.real), float(c.imag)] for c in s21],
        "s22": [[float(c.real), float(c.imag)] for c in s22],
        "s12": [[float(c.real), float(c.imag)] for c in s12],
        "abs_s11_reported_only": np.abs(s11).tolist(),
        "abs_s21": np.abs(s21).tolist(),
        "abs_s22": np.abs(s22).tolist(),
        "arg_s21_deg_unwrapped": np.degrees(np.unwrap(np.angle(s21))).tolist(),
        "beta_rad_per_m": beta.tolist(),
        "z0_msl_measured_ohm": [[float(c.real), float(c.imag)]
                                for c in results["lumped"]["z0_msl"]],
        "phase_self_consistency_dev_deg": phase_dev.tolist(),
        "phase_self_consistency_max_deg": float(np.max(np.abs(phase_dev))),
        "phase_self_consistency_tol_deg": B_PHASE_SELF_CONSISTENCY_TOL_DEG,
        "phase_self_consistency_passed":
            bool(np.max(np.abs(phase_dev)) <= B_PHASE_SELF_CONSISTENCY_TOL_DEG),
        "passivity_balance_verbatim": balance.tolist(),
        "passivity_balance_max": passivity_max,
        "passivity_passed": bool(passivity_max <= 1.0 + B_PASSIVITY_TOL),
        "settling_positive_control_smoke_warned": bool(smoke_warned),
        "settling_real_run_truncated": bool(truncated_any),
        "port_info": port_info,
        "mesh": {k: v for k, v in plan.items() if not k.endswith("_lines_m")},
        "elapsed_s": round(elapsed, 1),
    }


def run_stage2(*, stage1: dict, sim_root: str, threads: int, nrts: int,
               end_criteria: float, rfx_artifact: dict | None,
               also_dx80: bool) -> dict:
    # THE CONTRACT: before any geometry exists.
    assert_stage1_gate_passed(stage1)

    comparator = _run_stage2_leg(
        dx_m=B_DX_COMPARATOR_M, sim_root=sim_root, threads=threads, nrts=nrts,
        end_criteria=end_criteria, label="dx50um")
    legs = {"comparator_dx50um": comparator}
    if also_dx80:
        legs["reported_only_dx80um"] = _run_stage2_leg(
            dx_m=B_DX_REPORTED_ONLY_M, sim_root=sim_root, threads=threads,
            nrts=nrts, end_criteria=end_criteria, label="dx80um")

    comparison = None
    comparison_status = "openems-only (rfx artifact not supplied)"
    if rfx_artifact is not None:
        oe = {"freqs_hz": np.asarray(comparator["freqs_hz"]),
              "s11": [complex(r, i) for r, i in comparator["s11"]],
              "s21": [complex(r, i) for r, i in comparator["s21"]],
              "s22": [complex(r, i) for r, i in comparator["s22"]]}
        comparison = compare_against_rfx(oe, rfx_artifact)
        comparison_status = f"compared against S_raw from {rfx_artifact['path']}"

    sanity = bool(comparator["passivity_passed"]
                  and comparator["phase_self_consistency_passed"]
                  and not comparator["settling_real_run_truncated"])
    return {
        "legs": legs,
        "comparison": comparison,
        "comparison_status": comparison_status,
        "sanity_passed": sanity,
        "cannot_compare": CANNOT_COMPARE,
        "unvalidated_sentence": (
            "Until an rfx-vs-openEMS comparison has actually been run and "
            "reviewed, the mixed lane's absolute |S| stays UNVALIDATED and "
            "that sentence stays in the lane's documentation unchanged."
        ),
    }


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def print_stage_plan(*, stage: str, rfx_json: str | None,
                     stage1_json: str | None) -> dict:
    print("=" * 78)
    print("openEMS referee for issue #498 -- probe-fed microstrip transition")
    print("COMPARATOR LEG ONLY. It brackets; it does not judge.")
    print("=" * 78)
    print(f"\nRequested stage: {stage}")
    print("\n--- STAGE 1: reproduce-gate (runs BEFORE any DUT geometry) ---")
    for key in ("a1", "a2"):
        rec = REPRODUCE_GATE_RECORD[key]
        print(f"  [{key.upper()}] {rec['leg']}")
        print(f"       example : {rec['example']}")
        if key == "a1":
            print(f"       check   : {rec['documented_check']}")
            print(f"       gate    : {rec['gate']}")
        else:
            print(f"       upstream: {rec['upstream']['documented_number']}")
            print(f"       gate (i)  UPSTREAM-ANCHORED : {rec['gate_upstream_anchored']}")
            print(f"       gate (ii) REPO-INTERNAL LOCK: {rec['gate_repo_internal_lock']}")
        r = rec["recorded_reproduction"]
        print(f"       recorded: {json.dumps({k: v for k, v in r.items() if k != 'log_caveat'})}")
        print(f"       log     : {r['log_path']} "
              f"(present in tree: {r['log_present_in_tree']})")
        if "log_caveat" in r:
            print(f"       CAVEAT  : {r['log_caveat']}")
    print(f"\n  do_not_repeat: {REPRODUCE_GATE_RECORD['do_not_repeat']}")
    print(f"  settling     : {REPRODUCE_GATE_RECORD['settling_evidence_protocol']}")

    print("\n--- STAGE 2: the DUT (REFUSED unless Stage 1 ran and passed) ---")
    print(f"  contract: {REPRODUCE_GATE_RECORD['stage2_contract']}")
    print(f"  stage1 record supplied: {stage1_json or '(none -- --stage 2 would exit 4)'}")
    print(f"  rfx artifact: {rfx_json or '(none -- openEMS-only report)'}")
    print(f"  {RFX_ARTIFACT_CONTRACT}")

    print("\n  GEOMETRY IT WOULD BUILD (the REALIZED rfx board, #723 class):")
    real = RFX_REALIZED_RECORD["realized"]
    decl = RFX_REALIZED_RECORD["declared"]
    print(f"    eps_r            {B_EPS_R}")
    print(f"    h_sub            {B_H_SUB_M*1e6:.0f} um  REALIZED "
          f"(declared {decl['h_sub_m']*1e6:.0f} um)")
    print(f"    trace width      {B_W_TRACE_M*1e6:.0f} um  REALIZED node span "
          f"(declared {decl['w_trace_m']*1e6:.0f} um; cell-span reading "
          f"{real['w_trace_cell_span_m']*1e6:.0f} um)")
    print(f"    trace            x = {B_TRACE_X_LO_M*1e3:.2f} .. "
          f"{B_TRACE_X_HI_M*1e3:.2f} mm, 1 cell thick, at z = {B_H_SUB_M*1e6:.0f} um")
    print(f"                     BOTH ENDS OPEN (blocker B4) -- the metal "
          f"stops at the absorber's inner face, nothing in the pad")
    print(f"    ground           PEC at z_lo; PML_8 on x/y and z_hi")
    print(f"    y_c              {B_Y_C_M*1e3:.2f} mm (rfx node 27; declared "
          f"{decl['y_c_m']*1e3:.2f} mm snaps)")
    print(f"    domain (clear)   {B_TRACE_X_HI_M*1e3:.2f} x {B_LY_M*1e3:.2f} "
          f"x {B_LZ_M*1e3:.2f} mm")
    print(f"    lumped port      x = {B_FEED_X_M*1e3:.2f} mm, ground -> trace, "
          f"{B_FEED_R_OHM:.0f} ohm, spanning the substrate height")
    print(f"    MSL port         start x = {B_MSL_FEED_X_M*1e3:.2f} mm, "
          f"prop_dir -x (INTO the line), {B_FEED_R_OHM:.0f} ohm")
    print(f"    MSL meas plane   x = {B_MSL_MEAS_X_M*1e3:.2f} mm = rfx's own "
          f"probe-0 coordinate (MeasPlaneShift)")
    print(f"    open stub        {(B_TRACE_X_HI_M-B_MSL_FEED_X_M)*1e3:.2f} mm "
          f"beyond the MSL feed plane -- part of the DUT, and in |S22|")
    print(f"    anchor           rfx normalizes to HJ Z0 = "
          f"{RFX_REALIZED_RECORD['anchor']['rfx_z0_hj_msl_ohm']:.5f} ohm of the "
          f"DECLARED board; the REALIZED board's HJ Z0 is "
          f"{RFX_REALIZED_RECORD['anchor']['hj_z0_realized_node_span_ohm']:.3f} ohm "
          f"({RFX_REALIZED_RECORD['anchor']['hj_z0_realized_cell_span_ohm']:.3f} "
          f"ohm on the cell-span reading). REPORTED, NEVER GATED.")
    print(f"    excitation       Gauss f0 = {B_F0_HZ/1e9:.2f} GHz, fc = "
          f"{B_FC_HZ/1e9:.2f} GHz; EndCriteria = {B_END_CRITERIA} (-40 dB)")
    print(f"    drives           two runs per mesh leg (lumped-driven and "
          f"MSL-driven), as the rfx mixed lane does")
    print(f"    rfx bins         {[f'{f/1e9:.2f}' for f in B_RFX_FREQS_HZ]} GHz, "
          f"evaluated exactly")

    self_check = geometry_self_check(verbose=True)

    print("\n--- WHAT IS COMPARED (against S_raw, never result.S) ---")
    print("    |S21| lumped -> MSL, de-embedded to 4.72 mm : PRIMARY magnitude")
    print("    |S22| MSL-driven,   de-embedded to 4.72 mm : within budget B")
    print("    |S11| lumped-driven                        : REPORTED only")
    print("    phase: each solver's arg(S21) vs ITS OWN measured beta "
          f"({B_PHASE_SELF_CONSISTENCY_TOL_DEG:.0f} deg); the raw cross-solver "
          "difference and the implied plane error are REPORTED, never gated")
    print("\n--- WHAT CANNOT BE COMPARED (stated before the run) ---")
    for item in CANNOT_COMPARE:
        print(f"    * {item}")
    return self_check


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default=os.environ.get("STAGE", "both"),
                   choices=["1", "2", "both"],
                   help="which stage to run (env STAGE is the fallback)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the full stage plan and the geometry that "
                        "WOULD be built; needs no openEMS")
    p.add_argument("--self-check", action="store_true",
                   help="print the pure-numpy mesh/geometry self-check only")
    p.add_argument("--output", default=".omx/probe-fed-msl-referee/referee.json")
    p.add_argument("--sim-root", default="/tmp/probe_fed_msl_openems_referee")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--nrts", type=int, default=B_NRTS_DEFAULT)
    p.add_argument("--end-criteria", type=float, default=B_END_CRITERIA)
    p.add_argument("--stage1-json", default=None,
                   help="Stage-1 artifact from a previous run; REQUIRED by "
                        "--stage 2 (the stage contract)")
    p.add_argument("--rfx-json", default=None,
                   help="the #498/#517 rfx measurement artifact (data "
                        "dependency; this script never imports rfx)")
    p.add_argument("--repo-root", default=None,
                   help="repo root for the Stage A1 delegate (default: "
                        "inferred from this file's location)")
    p.add_argument("--no-dx80-leg", action="store_true",
                   help="skip the reported-only dx=80 um leg")
    args = p.parse_args(argv)

    repo_root = Path(args.repo_root) if args.repo_root else Path(
        __file__).resolve().parents[2]

    if args.self_check:
        res = geometry_self_check(verbose=True)
        return 0 if res["passed"] else 1

    if args.dry_run:
        res = print_stage_plan(stage=args.stage, rfx_json=args.rfx_json,
                               stage1_json=args.stage1_json)
        print("\nDRY RUN: no openEMS was imported, no geometry was built, "
              "no number was produced.")
        return 0 if res["passed"] else 1

    self_check = geometry_self_check(verbose=True)
    if not self_check["passed"]:
        print("CONFIG ERROR: mesh/geometry self-check failed -- refusing to "
              "run.", file=sys.stderr)
        return 3

    rfx_artifact = None
    try:
        if args.rfx_json:
            rfx_artifact = load_rfx_artifact(args.rfx_json)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 3

    stage1_record = None
    if args.stage1_json:
        try:
            loaded = json.loads(Path(args.stage1_json).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"CONFIG ERROR: --stage1-json unreadable ({exc})", file=sys.stderr)
            return 3
        stage1_record = loaded.get("stage1", loaded)

    artifact: dict = {
        "issue": 498,
        "scope": "comparator leg only -- brackets, does not judge rfx",
        "reproduce_gate_record": REPRODUCE_GATE_RECORD,
        "rfx_realized_record": RFX_REALIZED_RECORD,
        "cannot_compare": CANNOT_COMPARE,
        "rfx_artifact_contract": RFX_ARTIFACT_CONTRACT,
        "self_check": self_check,
        "stage_requested": args.stage,
    }
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def _archive_openems_stdout() -> list:
        """Copy every openEMS stdout log out of sim_root, beside the artifact.

        _run_openems_capturing_stdout writes ``_openems_stdout.log`` INSIDE
        sim_root, which is not archived, so a job that dies leaves nothing to
        read. That absence is the single reason VESSL 369367257610's failure
        could not be separated into 'the solve diverged' vs 'the port was
        starved' from its artifacts, and it costs a whole re-run each time.
        Runs on every exit path, including the failing ones, and never raises.
        """
        archived = []
        try:
            root = os.path.abspath(args.sim_root)
            dest_root = os.path.join(os.path.dirname(out_path) or ".",
                                     "openems_logs")
            for dirpath, _dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if fn != "_openems_stdout.log":
                        continue
                    src = os.path.join(dirpath, fn)
                    rel = os.path.relpath(dirpath, root).replace(os.sep, "_")
                    dst = os.path.join(dest_root, f"{rel}_{fn}")
                    os.makedirs(dest_root, exist_ok=True)
                    shutil.copyfile(src, dst)
                    archived.append(dst)
        except Exception as exc:  # never let archiving mask the real outcome
            archived.append(f"<archiving failed: {type(exc).__name__}: {exc}>")
        return archived

    def _write(rc_note: str) -> None:
        artifact["note"] = rc_note
        artifact["openems_stdout_logs"] = _archive_openems_stdout()
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2, default=str)
        print(f"\n=== Written to {out_path} ===")
        if artifact["openems_stdout_logs"]:
            print(f"    openEMS stdout archived: "
                  f"{len(artifact['openems_stdout_logs'])} log(s)")

    t0 = time.time()
    try:
        if args.stage in ("1", "both"):
            print("\n--- STAGE 1: reproduce-gate (openEMS's own canonical "
                  "examples) ---")
            stage1_record = run_stage1(repo_root=repo_root,
                                       sim_root=os.path.abspath(args.sim_root),
                                       threads=args.threads)
            artifact["stage1"] = stage1_record
            if not stage1_record["passed"]:
                print("\nStage A FAILED its reproduce-gate -- skipping "
                      "Stage B (external_solver_comparator.md: 'only then "
                      "swap in the target geometry').", file=sys.stderr)
                artifact["stage2"] = None
                artifact["elapsed_s"] = round(time.time() - t0, 1)
                _write("Stage 1 failed its gate; Stage 2 not attempted.")
                return 1

        if args.stage in ("2", "both"):
            try:
                assert_stage1_gate_passed(stage1_record)
            except Stage1GateError as exc:
                print(f"\nSTAGE 2 REFUSED: {exc}", file=sys.stderr)
                artifact["stage2"] = None
                artifact["stage2_refusal"] = str(exc)
                artifact["elapsed_s"] = round(time.time() - t0, 1)
                _write("Stage 2 refused by the stage contract.")
                return 4
            print("\n--- STAGE 2: the #488 probe-fed microstrip transition ---")
            artifact["stage2"] = run_stage2(
                stage1=stage1_record, sim_root=os.path.abspath(args.sim_root),
                threads=args.threads, nrts=args.nrts,
                end_criteria=args.end_criteria, rfx_artifact=rfx_artifact,
                also_dx80=not args.no_dx80_leg)
    except ImportError as exc:
        print(f"SKIP: openEMS Python bindings not importable ({exc}). This "
              "script is VESSL-only; see "
              "scripts/vessl_probe_fed_msl_referee.yaml.", file=sys.stderr)
        return 2
    except ConfigError as exc:
        print(f"CONFIG ERROR (script bug, not a physics finding): {exc}",
              file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(f"SELF-CHECK / GATE FAILED: {exc}", file=sys.stderr)
        artifact["error"] = str(exc)
        artifact["elapsed_s"] = round(time.time() - t0, 1)
        _write("A gate or self-check failed.")
        return 1

    artifact["elapsed_s"] = round(time.time() - t0, 1)
    ok = True
    if artifact.get("stage1") is not None:
        ok = ok and bool(artifact["stage1"]["passed"])
    if artifact.get("stage2") is not None:
        ok = ok and bool(artifact["stage2"]["sanity_passed"])
    _write("ok" if ok else "a self-check failed")

    print("\n" + "=" * 78)
    print("WHAT CANNOT BE COMPARED (repeated after the run, deliberately):")
    for item in CANNOT_COMPARE:
        print(f"  * {item}")
    print("=" * 78)
    print(f"overall_passed={ok} (self-checks only -- NOT a verdict on rfx "
          f"vs openEMS agreement)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
