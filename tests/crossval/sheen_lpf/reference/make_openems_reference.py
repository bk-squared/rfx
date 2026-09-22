#!/usr/bin/env python
"""Make the Sheen low-pass filter's openEMS reference, with provenance.

NEVER RUN BY CI. The external solver runs by hand, on the cluster, when the case
is created or its geometry changes; this script is the thing that is run.

WHAT IS SIMULATED
-----------------
The stepped-impedance microstrip low-pass filter of D. M. Sheen, S. M. Ali,
M. D. Abouzahra and J. A. Kong, "Application of the three-dimensional
finite-difference time-domain method to the analysis of planar microwave
circuits", IEEE Trans. MTT 38(7):849-857, July 1990. On RT/Duroid (eps_r = 2.2,
h = 0.794 mm) two 2.413 mm wide 50 ohm feeds are joined by one wide
20.320 x 2.540 mm low-impedance section. The wide section is a shunt
capacitance: it passes the band up to about 5 GHz, and above it the mismatch at
its two steps reflects, so |S21| falls away into a deep stopband. The exact
trace coordinates are the Elsherbeni-Demir reproduction's ("The FDTD Method for
Electromagnetics with MATLAB Simulations", Sec. 6.2), as transcribed in
roseengineering/rffdtd's ``examples/lowpass.py`` -- the same coordinates
``validation/crossval/07_sheen_lpf.py`` declares, copied from it, never edited
there.

WHAT THE STOPBAND LOOKS LIKE, AS A FACT ABOUT THE RECORD AND NOT A VERDICT
--------------------------------------------------------------------------
``07_sheen_lpf.py``'s own header records, measured, that the stopband of this
board is a DOUBLE transmission zero near 7.0 and 8.0 GHz rather than a single
null, and that a bare ``argmin`` over a window picks different members of that
pair on different solvers and on different meshes. This script therefore records
the whole 801-bin curve, reports the deepest minimum in 5-10 GHz with the
repository's sub-bin estimator, and GATES NOTHING on it. A reader who wants to
know how many zeros a rung resolved reads the curve, which is in the record.

WHY THE REFERENCE IS BEING REMADE
---------------------------------
The committed record ``validation/crossval/_07_sheen_results/openems.json``
carries eight fields: solver, res_um, runtime_s, and the five 801-point arrays.
It names no openEMS version, no build, no run, no mesh beyond the single number
198.5, and its ``reproduce_gate_record`` is absent -- while the script's own
gate record says ``status: "UNRUN"``. Nothing in it can be traced to a solver
or a commit. This script produces the same measurement with the provenance, the
mesh statement and the witnesses a reference needs: the openEMS tutorial
reproduced first as Stage A, the Sheen board as Stage B on three meshes, and
every sanity gate recorded per bin.

WHAT LIVES IN THE SHARED MODULE
-------------------------------
``tests/crossval/_openems_tutorial_gate.py`` owns the reproduce gate and the
machinery: the openEMS MSL notch filter tutorial verbatim, its attribution, its
recorded reproduction, the analytic notch frequency and the Stage A gate, the
thirteen sanity helpers copied byte-identically from
``validation/crossval/20_msl_phase_referee.py``, the generalized stage runner,
the energy summary, the version probe and the record / evidence writers. Read
that module's docstring for the tutorial's citation and the list of gates.

STAGE A -- the reproduce gate (comparator first)
------------------------------------------------
``_openems_tutorial_gate.run_stage_a``: the tutorial, verbatim, on its own mesh
and its own 1601-point grid, solved in THIS invocation. The gate is the measured
notch inside 0.80-1.05 x the analytic quarter-wave frequency AND at least 20 dB
deep. If it fails, the script exits 1 and writes no Stage B record.

THE do_not_repeat THIS ANSWERS, QUOTED IN FULL
----------------------------------------------
``07_sheen_lpf.py``'s own ``REPRODUCE_GATE_RECORD["do_not_repeat"]``:

    "07_sheen_lpf.py's own run_openems_tutorial(), pre-#971: the docstring and
    its print both claimed 'known-good ... S21 notch ~3.43 GHz (repo fixture:
    openEMS 3.4286 GHz)' -- that number was never this function's own
    measurement, it was cv06b's realized-board reading, later shown to be an
    outlier by an independent Palace-FEM referee
    (scripts/diagnostics/palace_notch_referee/). Do not cite a sibling case's
    number as this function's own known-good without running THIS function and
    checking the result against F_NOTCH_TUTORIAL_DECLARED_HZ."

Ticked: Stage A runs here, in the same invocation as Stage B, and is judged
against the analytic frequency the shared module recomputes from the tutorial's
own declared 600 um / 254 um / 12 mm geometry. The MSL notch filter's recorded
reproduction travels in the record as the tutorial's audit trail and is never
what this run is judged by. ``--self-check`` recomputes
``07_sheen_lpf.py``'s own ``F_NOTCH_TUTORIAL_DECLARED_HZ`` from its own
constants and asserts it equals the shared module's ``F_NOTCH_AN_HZ``, so the
two paths cannot drift.

STAGE B -- the Sheen board, three mesh rungs. THE FULL DELTA LIST
-----------------------------------------------------------------
Every one of these is a difference from the tutorial Stage A runs. The list is
``DELTA_LIST`` below, it goes into the record's ``meta``, and the dry run prints
it.

    DELTA 1  substrate: RT/Duroid eps_r 2.2, h 0.794 mm, in place of the
             tutorial's RO4350B eps_r 3.66, h 0.254 mm.
    DELTA 2  trace: two 2.413 mm wide 50 ohm feeds and ONE wide
             20.320 x 2.540 mm low-impedance section, in place of the
             tutorial's single 600 um through line with a 12 mm open stub.
             There is no open stub on this board and no through line across it.
    DELTA 3  the feeds are OFFSET and not collinear: the input feed's centre
             sits at y 9.8565 mm and the output feed's at y 16.4635 mm, 6.607 mm
             apart transversely, both joined to the wide section. The tutorial's
             one line runs along y = 0.
    DELTA 4  arms and box: each feed is extended by EXTEND_FEED = 4.0 mm of
             matched 50 ohm line, so the domain is 27.472 x 26.320 x 3.794 mm
             (x propagation, y transverse, z stack) against the tutorial's
             100 x 30 x 3 mm. Y_CLEAR = 3.0 mm of substrate clears the wide
             section transversely; Z_AIR = 3.0 mm of air sits above the trace.
    DELTA 5  axis map: rfx/openEMS x is the Sheen paper's y (propagation) and
             rfx/openEMS y is its x (transverse). The board is laid out in
             absolute metres from x = 0 at the input edge.
    DELTA 6  ports: both MSLPorts sit ON the x faces of the domain, as the
             tutorial's do, but their shifts are this board's:
             FeedShift = PORT_MARGIN = 2.5 mm on the driven port (a LENGTH, not
             the tutorial's 10*resolution, so it does not move with the rung),
             MeasPlaneShift = 0.45 x PATCH_X0 = 5.610 mm on port 1 and
             0.45 x (LX - PATCH_X1) = 5.610 mm on port 2. Port 2 declares no
             FeedShift at all -- it is the passive port.
    DELTA 7  frequency grid: linspace(0.5e9, 20e9, 801) in place of the
             tutorial's linspace(1e6, 7e9, 1601). The Gaussian excitation is
             SetGaussExcite(F_MAX/2, F_MAX/2) = 10 GHz centre and corner, which
             is the same literal reading of openEMS's call the tutorial uses, at
             this board's own F_MAX.
    DELTA 8  TWO CalcPort passes: pass 1 with no ref_impedance, so
             ``port.Z_ref`` is the line impedance openEMS measured, recorded as
             ``re_z0``; pass 2 with ref_impedance = 50, and S is taken from that
             pass. This is the Sheen script's own extraction and is kept
             (leader decision, this PR). It is a DEPARTURE from the MSL notch
             filter maker's own tick "ref_impedance is never passed to
             CalcPort", and it is declared here rather than inherited silently.
    DELTA 9  NrTS / EndCriteria: openEMS's own library defaults (~1e9 / 1e-5) on
             every real pass, exactly as Stage A and the tutorial run. This is
             NOT a delta from the tutorial -- it is the one place where a
             DEPARTURE FROM THE RETIRED SCRIPT is declared instead: that script's
             ``_openems_common_setup`` capped the run at NrTS = 30000 with
             EndCriteria = 1e-4, and the cap is NOT carried here (leader
             decision, this PR). The reason is the mesh rungs. The cap was a cost
             choice on one mesh; halving the cell halves the timestep, so the
             same physical decay needs about twice the steps, and a 30000-step
             cap that just about reached 1e-4 on the coarse board would stop the
             fine board mid-ring-down. The shared runner treats openEMS's own
             "reached before the end-criteria of" warning on a real pass as a
             FAILED gate, so that would cost a cluster hour and produce nothing.
             The smoke pass keeps its 200 steps. WHAT THIS COSTS INSTEAD: a real
             pass now runs until the energy has decayed 50 dB rather than until
             a step budget runs out, so it can take considerably longer than the
             172 s the retired script recorded at the coarse mesh. The job file's
             21600 s timeout is the only bound on that.
    DELTA 10 mesh rule: res = min(c0 / (F_MAX sqrt(eps_r)) / 50, h_sub / 4)
             = 198.5 um at rung 1.0, so the substrate-thickness term decides it,
             not the wavelength term (202.1 um). The tutorial's rule has no
             h_sub/4 term and no min.
    DELTA 11 mesh rung: that resolution is multiplied by a factor -- 1.0
             (``stage_b_coarse``), 1/sqrt(2) (``stage_b_mid``) and 0.5
             (``stage_b_fine``). Three meshes, two refinements, which is the
             least a mesh statement can be made of. The factor scales the x
             lines, the y lines, the air above the board AND the substrate's own
             z lines: ``linspace(0, H_SUB, 5)`` becomes
             ``linspace(0, H_SUB, round(4/factor)+1)``, so the 794 um board
             carries 4, 6 and 8 cells (198.5, 132.3, 99.25 um). Refining x and y
             alone would leave the thickness direction at 4 cells on every rung,
             and that is the direction that sets the effective permittivity and
             therefore where the stopband sits. Exactly as the MSL notch filter
             maker does it.
    DELTA 12 CSX length unit: SetDeltaUnit(1.0), metres, in place of the
             tutorial's 1e-6. Copied from the script; every reported length is
             converted to um or mm at the reporting boundary, once.

    DELTA 14 Mesh refinement around the metal: the tutorial smooths x and y
             twice (resolution/4 around its trace edges, then resolution); this
             builder smooths once per axis at res and lays thirds-rule lines at
             the two feed edges only, not at the wide section's four edges.
             Inherited from the script and proved so; the run records what the
             grid made of each feed edge per rung (DELTA_LIST entry 14 has the
             full text; there is no DELTA 13).

Nothing else changes: the boundary list
``['PML_8','PML_8','MUR','MUR','PEC','MUR']``, the thirds-rule OFFSET recipe,
MSLPort with port 1 ``excite=-1``, and the zero-thickness PEC sheets on
z = H_SUB are the script's and the tutorial's alike (where the offsets are
applied, and how often each axis is smoothed, is delta 14).

That the Stage B builder IS the script's builder is not prose. ``--self-check``
reads ``validation/crossval/07_sheen_lpf.py``'s own ``run_openems`` source,
applies the three declared renames in ``COPY_SUBSTITUTIONS``, and compares the
result with ``_build_sheen_board``'s corresponding slice character for
character; then it applies the two declared rung substitutions in
``RUNG_SUBSTITUTIONS`` and compares with ``_build_sheen_board_at_rung``'s.

THE WITNESS BAND: 2-12 GHz, AND WHY NOT THE WHOLE GRID
------------------------------------------------------
``WITNESS_BAND_HZ = (2.0e9, 12.0e9)`` is the band this record is meant to serve
and the band the passivity witness is evaluated over. The reason is in the
committed record: its energy sum |S11|^2+|S21|^2 runs 0.78 to 1.00 across
2-12 GHz and falls to 0.41 by 20 GHz. A sum that low is not a passivity
question -- the witness bounds the EXCESS above unity only -- so the deficit is
RECORDED (per bin, plus the band and whole-grid minima and the maxima above the
band) and not judged here. What it means physically is the case's to argue with
the curve in front of it, not this script's.

DO-NOT-REPEAT, TICKED (task recipe ``external_solver_comparator.md``, the
2026-08-03 addendum: quote the precedent's header IN FULL and tick each
recorded failure BEFORE writing code)
------------------------------------------------------------------------------
From ``validation/crossval/20_msl_phase_referee.py``'s DO-NOT-REPEAT block, as
quoted in the MSL notch filter maker:

    "at dx=80 um the substrate is only 3.175 cells (the 'mixed-cell danger zone'
    rfx preflight warns about), where the openEMS MSL-port extraction is
    NON-PHYSICAL (|S11|^2+|S21|^2 up to 8.9, passivity grossly violated). dx=50
    um gives 5.08 substrate cells where BOTH solvers are passive, so it is the
    only valid matched-mesh comparison."

Facts about this script against it, no verdict:
  * No stage lays a uniform dx across the substrate. The z recipe is an explicit
    ``linspace(0, H_SUB, N+1)``, so the substrate top is always ON a mesh line
    and the cell count is an integer by construction, never a ratio that can
    land at 3.175.
  * Substrate cells per rung: 4 (198.5 um each), 6 (132.33 um), 8 (99.25 um).
    The recorded non-physical case had 3.175 cells; the recorded passive case
    had 5.08.
  * The passivity witness is recorded and gated on every real pass, over
    2-12 GHz, so a rung that does land somewhere non-physical says so.

Also from that block, and from the coax lane it quotes:
  * "MUR-on-dielectric is unstable (exponential energy blow-up 5e-16 ->
    2.8e13). Use PML on +-z instead" and "conductors-through-PML is the
    validated reflectionless pattern". TICKED: the substrate box here is
    [0,0,0]..[LX,LY,H_SUB], the FULL domain footprint, so each y-face MUR sees a
    uniform dielectric cross-section rather than an air/substrate step; the
    z-max MUR sees air only; z-min is PEC (the ground); and the two x faces that
    the feed conductors run into are PML_8.
  * "a radial AddLumpedPort cannot excite the coax TEM mode (3 runs failed)".
    N/A -- both ports are MSLPorts.
  * "every lumped-port/feed position needs an explicit mesh line". The x lines
    0, LX, PATCH_X0 and PATCH_X1 are explicit, so both port start planes are on
    a line by construction. Where the FEED and MEASUREMENT planes landed is not
    knowable without CSXCAD: the run reads the realized x lines back and records
    each plane's nearest line and the snap distance.
  * The excitation guard has no absolute floor, and no complex value reaches
    ``json.dump`` (magnitudes and degrees; ``re_z0`` via ``np.real``).

One more, from ``07_sheen_lpf.py``'s own header, kept as a measurement rather
than a fix:

    "openEMS's own thirds-rule mesh lines (tm = [+33.08, -16.54] um at
    res=198.5um) deliberately straddle each feed edge rather than sit on it,
    covering only 2379.92 um (-1.371%)."

TICKED as a recorded quantity: per rung, and again from the realized lines after
the run, this script records the nearest mesh line to each declared feed edge,
the snap distance, and the width those lines bracket. It changes nothing about
the mesh -- the thirds-rule offsets are the script's and the tutorial's -- it
makes the number readable instead of re-derivable.

TWO BLOCKS OF THAT HEADER THAT DO NOT TRANSFER, QUOTED IN FULL SO THE JUDGEMENT
IS CHECKABLE (the addendum's point is that a summary is how the decisive
sentence gets dropped, so they are pasted whole and then marked)
-------------------------------------------------------------------------------
The first, ``07_sheen_lpf.py`` lines 135-153:

    "THE MESH STAYS dx = 200 um, and the on-lattice redraw the design note
    prescribes (S1.3, "dx = h_sub/n") is REFUSED HERE WITH THE MEASUREMENT
    THAT REFUSES IT. dx = H_SUB/4 = 198.5 um does put the sheet exactly on
    the DECLARED interface and realize h_sub = 794.0 um exactly - but it
    realizes the two nominally identical 50-ohm feeds 12 and 13 node rows
    wide (2183.5 vs 2382.0 um; HJ Z0 54.22 vs 51.19 ohm), because their
    centres sit at different sub-cell offsets. A mesh that makes a
    symmetric board asymmetric is a worse board than one that realizes
    h_sub 0.76% thick, and `assert_realized_metal` now refuses it. Finer
    aligned meshes that restore the symmetry (H_SUB/5 = 158.8 um, H_SUB/6 =
    132.3 um; both feeds 15 and 18 rows) also change `n_probe_offset`'s
    physical meaning - it is 30 CELLS, derived at :459-464 as 6 mm >=
    5*h_sub upstream with a 3.2 mm >= lambda_g/4 downstream clearance - and
    a mesh change plus a recipe re-derivation must not ride along with a
    realization change. The S1.3 pathology does not arise at dx = 200 um:
    the sheet lands on the realized substrate TOP (the dielectric occupies
    cells 0..3), not buried inside the laminate, so there is no vacuum slot
    and nothing is absorbed by a tie rule."

The second, lines 154-175:

    "THE PORT MISMATCH IS RE-MEASURED, NOT CLOSED. :73-86 records that
    `add_msl_port` is handed the DECLARED width=W_FEED (2413 um) and
    height=H_SUB (794 um) while the metal realizes something else. Under
    the contract that gap is now visible per port and printed by
    `assert_realized_metal`: each port's cross-section spans 13 node rows
    (43..55 and 84..96) while its feed metal realizes 12 (44..55, 85..96) -
    the port rounds each face to the NEAREST node while a sheet footprint
    is closed [lo, hi], so the port integrates one row that carries no
    metal. Handing the port the contract's realized extent (2200 um) does
    fix the rows - measured, both ports then span exactly the metal's 12 -
    but `width` also sets the Hammerstad-Jensen reference impedance the
    wave split uses (a = (V + Z0*I)/2, b = (V - Z0*I)/2, rfx/api/_sparams.py),
    and HJ(2200, 800) = 54.22 ohm on a line whose own passband median
    Re(Z0) the current leg measures at 51.91 ohm (pre-#931: 50.30 ohm).
    That is a 2.31-ohm mismatch, so the earlier 4-ohm rationale is historical.
    Changing the reference to fix a one-row aperture is still not a measured
    remedy for this port; the committed leg does not settle that choice.
    The real remedy is S1.9's: the MSL port takes its
    cross-section from the realized sheet footprint and its reference
    impedance from the line, which is core work (#729 class), not a script
    constant. Left open, measured, and named."

**Both are N/A here, for one reason, and it is the same reason for both: they
are about rfx's single-scalar cubic lattice and rfx's own port reference
impedance, neither of which exists on the openEMS side.** rfx's
``Simulation(dx=)`` is one scalar for all three axes, so every declared
dimension lands where that one number puts it, and the first block is the
measurement of what happens when the scalar is set to h_sub/4: the two
nominally identical feeds fall on different sub-cell offsets and come out 12
and 13 rows wide. openEMS's mesh is non-uniform and line-based -- this builder
adds explicit lines at 0, LX and both wide-section faces and thirds-rule pairs
at both feed edges, then smooths each axis separately -- so "198.5 um" here is
a smoothing target per axis, not a lattice pitch that every face must land on.
It is also, and this is the part that reads as a coincidence and is not one,
the committed openEMS leg's own ``res_um``: 198.5 is what
``min(c0/(F_MAX*sqrt(eps_r))/50, H_SUB/4)`` returns, which is where delta 10
gets it. The asymmetry the first block refuses is an rfx-lattice
outcome and does not follow from that number being the same.
The second block is about ``add_msl_port`` being handed a declared width while
rfx's sheet realizes another, and about the Hammerstad-Jensen reference
impedance rfx's wave split uses. openEMS's ``AddMSLPort`` takes the port's
cross-section as a box in metres and, on the second CalcPort pass, the system
impedance as an explicit 50 ohm (delta 8); no Hammerstad-Jensen reference is
computed anywhere on this side. What IS comparable -- how far the grid's lines
sit from each declared feed edge, and what width they bracket -- is measured
and recorded per rung, which is the tick above.

A failed gate exits non-zero and names itself. It also leaves its evidence: the
record built so far -- every array measured, the energy-sum numbers, and a
top-level ``failed_gate`` carrying the gate's own message -- is written to
``<output stem>_FAILED.json`` beside the requested output BEFORE the non-zero
exit.

EXIT CODES
----------
0 every requested stage ran and every gate passed; 1 a gate failed; 2 openEMS is
not importable; 3 a layout/config bug in this script.

USAGE
-----
    python make_openems_reference.py --self-check
    python make_openems_reference.py --dry-run --stage both
    python make_openems_reference.py --stage both --output <path>.json \\
        --sim-root /tmp/sheen_lpf_openems --threads 8

The job file is ``scripts/vessl_sheen_lpf_openems_reference.yaml``. ``run_id`` in
the record is always ``null``: VESSL does not export the run id into the pod, so
the submitter fills it (``~/.claude/rules/vessl-jobs.md``).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# The shared tutorial gate, loaded by PATH. This script runs as a bare file on
# the cluster, where the repository root is not on sys.path and ``tests`` is not
# an importable package, so a package import would work under pytest and fail on
# the box that actually produces the record.
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

F_NOTCH_AN_HZ = _gate.F_NOTCH_AN_HZ
REPRODUCE_GATE_RECORD = _gate.REPRODUCE_GATE_RECORD
STAGE_A_GATE = _gate.STAGE_A_GATE
STAGE_A_MIN_DEPTH_DB = _gate.STAGE_A_MIN_DEPTH_DB
STAGE_A_NOTCH_BAND_GHZ = _gate.STAGE_A_NOTCH_BAND_GHZ
substrate_z_cells = _gate.substrate_z_cells
_smooth_estimate = _gate._smooth_estimate
_builder_body_after_kw = _gate._builder_body_after_kw


# ---------------------------------------------------------------------------
# SHEEN geometry (metres). Copied from validation/crossval/07_sheen_lpf.py's own
# constants block, which is never edited there. Native Sheen frame: x_S
# (transverse, patch length), y_S (propagation, feeds), z (stack). Values from
# the rffdtd examples/lowpass.py transcription of Elsherbeni-Demir Sec. 6.2.
#
# C0 here is the SHEEN SCRIPT'S own constant, 2.99792458e8, not the tutorial
# gate's 2.998e8. They differ in the 7th digit and the mesh resolution is
# computed from this one, exactly as the script computes it.
# ---------------------------------------------------------------------------
C0 = 2.99792458e8

EPS_R = 2.2
H_SUB = 0.794e-3
BOARD_XS = 22.320e-3        # Sheen x extent
BOARD_YS = 19.472e-3        # Sheen y extent (propagation)
W_FEED = 2.413e-3           # 50-ohm feed width
# input feed:  Sheen x 6.650-9.063, y 0-8.466   -> centre_xS = 7.8565, len 8.466
# output feed: Sheen x 13.257-15.670, y 11.006-19.472
# wide patch:  Sheen x 1.000-21.320 (20.320), y 8.466-11.006 (2.540)
IN_FEED_XS_C = 0.5 * (6.650e-3 + 9.063e-3)      # 7.8565 mm
OUT_FEED_XS_C = 0.5 * (13.257e-3 + 15.670e-3)   # 14.4635 mm
PATCH_XS_LO, PATCH_XS_HI = 1.000e-3, 21.320e-3  # transverse span of patch
PATCH_YS_LO, PATCH_YS_HI = 8.466e-3, 11.006e-3  # propagation span of patch
IN_FEED_LEN = PATCH_YS_LO                        # 8.466 mm (edge -> patch)
OUT_FEED_LEN = BOARD_YS - PATCH_YS_HI            # 8.466 mm (patch -> edge)

EXTEND_FEED = 4.0e-3        # extra matched 50-ohm feed per side (de-embed room)
Y_CLEAR = 3.0e-3            # transverse clearance from patch edge to boundary
Z_AIR = 3.0e-3              # air above the trace
F_MAX = 20.0e9              # Sheen analysis band top
F_LO = 0.5e9

PATCH_X0 = EXTEND_FEED + IN_FEED_LEN
PATCH_LEN_PROP = PATCH_YS_HI - PATCH_YS_LO        # 2.540 mm
PATCH_X1 = PATCH_X0 + PATCH_LEN_PROP
LX = PATCH_X1 + OUT_FEED_LEN + EXTEND_FEED

PATCH_TRV_LEN = PATCH_XS_HI - PATCH_XS_LO          # 20.320 mm
Y_SHIFT = Y_CLEAR - PATCH_XS_LO                    # patch_lo -> Y_CLEAR


def yS(x_sheen):                                   # Sheen-x -> domain-y
    return x_sheen + Y_SHIFT


IN_FEED_YC = yS(IN_FEED_XS_C)
OUT_FEED_YC = yS(OUT_FEED_XS_C)
PATCH_Y_LO, PATCH_Y_HI = yS(PATCH_XS_LO), yS(PATCH_XS_HI)
LY = PATCH_Y_HI + Y_CLEAR
LZ = H_SUB + Z_AIR

PORT_MARGIN = 2.5e-3       # port plane distance from x-boundary (>2*h_sub)

# ---------------------------------------------------------------------------
# Stage B's run parameters. The grid, the boundary, the ports and the two-pass
# extraction are the script's own (deltas 7, 8, 12). The stop criteria are NOT
# (delta 9).
# ---------------------------------------------------------------------------
B_N_FREQS = 801
B_BOUNDARY = ["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"]
B_PML_CELLS = 8            # "PML_8" -- the x faces only
B_CALCPORT_REF_IMPEDANCE = 50

# The real pass runs at openEMS's OWN defaults, like Stage A and the tutorial:
# None here means "do not pass NrTS/EndCriteria at all", so the library's ~1e9
# and 1e-5 apply. The retired script's cap is recorded below and deliberately
# NOT used -- see delta 9.
B_REAL_NRTS = None
B_REAL_END_CRITERIA = None

# What validation/crossval/07_sheen_lpf.py's _openems_common_setup sets, recorded
# so the departure is a stated number rather than a memory. --self-check reads
# that function off disk and asserts these two are still what it says.
SCRIPT_NRTS_CAP = 30000
SCRIPT_END_CRITERIA_CAP = 1.0e-4
SCRIPT_SETUP_FUNCTION = "_openems_common_setup"
CSX_UNIT_M = 1.0           # SetDeltaUnit(1.0): this builder works in metres

B_COARSE_RESOLUTION_FACTOR = 1.0
B_MID_RESOLUTION_FACTOR = 1.0 / np.sqrt(2.0)
B_FINE_RESOLUTION_FACTOR = 0.5

# The band this record serves, and the band the passivity witness reads. See the
# module docstring: the committed record's energy sum runs 0.78-1.00 over
# 2-12 GHz and falls to 0.41 by 20 GHz, so 12 GHz is where the record stops
# being one this case can hold a magnitude against. It is not a window derived
# from any run: it is a band edge read off the committed record's own deficit.
WITNESS_BAND_HZ = (2.0e9, 12.0e9)
WITNESS_BAND_GHZ = (WITNESS_BAND_HZ[0] / 1e9, WITNESS_BAND_HZ[1] / 1e9)

# The passivity tolerance, the shared module's: max(|S11|^2+|S21|^2) <= 1.05.
PASSIVITY_TOL = _gate.PASSIVITY_TOL

# Reported, never gated. The stopband of this board is a double transmission
# zero near 7 and 8 GHz (07_sheen_lpf.py's header, measured), so "the deepest
# minimum in 5-10 GHz" names ONE member of a pair whose members swap places
# between solvers and meshes. The whole curve is in the record.
NULL_BAND_GHZ = (5.0, 10.0)
PASSBAND_GHZ = (1.0, 4.0)
CUTOFF_SEARCH_FROM_GHZ = 2.0
# 20*log10(1/sqrt(2)); the -3 dB corner is this far below the passband mean.
CUTOFF_LEVEL_DB = float(20.0 * np.log10(1.0 / np.sqrt(2.0)))


DELTA_LIST = [
    "DELTA 1 (substrate): RT/Duroid eps_r 2.2, h 0.794 mm, in place of the "
    "tutorial's RO4350B eps_r 3.66, h 0.254 mm.",
    "DELTA 2 (trace): two 2.413 mm wide 50 ohm feeds joined by ONE wide "
    "20.320 x 2.540 mm low-impedance section, in place of the tutorial's single "
    "600 um through line with a 12 mm open-circuit stub. There is no open stub "
    "on this board and no through line across it.",
    "DELTA 3 (offset feeds): the feeds are not collinear. The input feed's "
    "centre sits at y 9.8565 mm and the output feed's at y 16.4635 mm, 6.607 mm "
    "apart transversely, both joined to the wide section; the tutorial's one "
    "line runs along y = 0.",
    "DELTA 4 (arms and box): each feed is extended by EXTEND_FEED = 4.0 mm of "
    "matched 50 ohm line, so the domain is 27.472 x 26.320 x 3.794 mm (x "
    "propagation, y transverse, z stack) against the tutorial's 100 x 30 x 3 mm. "
    "Y_CLEAR = 3.0 mm of substrate clears the wide section transversely and "
    "Z_AIR = 3.0 mm of air sits above the trace.",
    "DELTA 5 (axis map): rfx/openEMS x is the Sheen paper's y (propagation) and "
    "rfx/openEMS y is its x (transverse). The board is laid out in absolute "
    "metres from x = 0 at the input edge.",
    "DELTA 6 (ports): both MSLPorts sit ON the x faces of the domain, as the "
    "tutorial's do, with this board's own shifts: FeedShift = PORT_MARGIN = "
    "2.5 mm on the driven port -- a LENGTH, not the tutorial's 10*resolution, so "
    "it does NOT move with the rung -- MeasPlaneShift = 0.45 x PATCH_X0 = "
    "5.610 mm on port 1 and 0.45 x (LX - PATCH_X1) = 5.610 mm on port 2. Port 2 "
    "declares no FeedShift at all; it is the passive port.",
    "DELTA 7 (frequency grid): linspace(0.5e9, 20e9, 801) in place of the "
    "tutorial's linspace(1e6, 7e9, 1601). The excitation is "
    "SetGaussExcite(F_MAX/2, F_MAX/2) = 10 GHz centre and corner -- the same "
    "literal reading of openEMS's call the tutorial uses, at this board's F_MAX.",
    "DELTA 8 (two CalcPort passes): pass 1 with no ref_impedance, so port.Z_ref "
    "is the line impedance openEMS measured and is recorded as re_z0; pass 2 "
    "with ref_impedance = 50, and S is taken from that pass. This is the Sheen "
    "script's own extraction and is kept (leader decision, this PR). It is a "
    "DEPARTURE from the MSL notch filter maker's tick 'ref_impedance is never "
    "passed to CalcPort', declared here rather than inherited silently.",
    "DELTA 9 (NrTS / EndCriteria): the library defaults, as the tutorial -- "
    "openEMS's own ~1e9 and 1e-5 on every real pass, exactly what Stage A runs; "
    "the retired script's 30000 / 1e-4 cap is NOT carried. This entry is a "
    "declared departure from the SCRIPT, not from the tutorial (leader "
    "decision, this PR). Why: the cap was a cost choice on one mesh. Halving "
    "the cell halves the timestep, so the same physical decay needs about twice "
    "the steps, and a 30000-step cap that just about reached 1e-4 on the coarse "
    "board would stop the fine board mid-ring-down -- which the shared runner "
    "correctly reports as a FAILED gate, for a cluster hour and no record. What "
    "it costs instead: a real pass now runs until the energy has decayed 50 dB "
    "rather than until a step budget runs out, so it can take considerably "
    "longer than the 172 s the retired script recorded at the coarse mesh; the "
    "job file's 21600 s timeout is the only bound on that -- measured since: the "
    "coarse rung alone ran 107 minutes on 8 threads (VESSL 369367263243, "
    "369367263269), so --real-end-criteria and --real-nrts exist to loosen the "
    "stop criteria for a run that cannot afford them, Stage A is never "
    "overridden, and any record made with an override says so in every Stage B "
    "block and in meta.stop_criteria_note. The smoke pass keeps its 200 steps at "
    "EndCriteria 0.0.",
    "DELTA 10 (mesh rule): res = min(c0 / (F_MAX sqrt(eps_r)) / 50, h_sub / 4) "
    "= 198.5 um at rung 1.0, so the substrate-thickness term decides it and not "
    "the wavelength term (202.1 um). The tutorial's rule has no h_sub/4 term and "
    "no min.",
    "DELTA 11 (mesh rung): that resolution is multiplied by a factor -- 1.0 "
    "(stage_b_coarse), 1/sqrt(2) = 0.70711 (stage_b_mid) and 0.5 "
    "(stage_b_fine). The factor scales the x lines, the y lines, the air above "
    "the board AND the substrate's own z lines: linspace(0, H_SUB, 5) becomes "
    "linspace(0, H_SUB, round(4/factor)+1), so the 794 um board carries 4, 6 "
    "and 8 cells (198.5, 132.3, 99.25 um). Scaling x and y alone would leave "
    "the direction that sets the effective permittivity unrefined, and the mesh "
    "statement would say nothing about where the stopband sits. Exactly as the "
    "MSL notch filter maker does it.",
    "DELTA 12 (CSX length unit): SetDeltaUnit(1.0), metres, in place of the "
    "tutorial's 1e-6. Copied from the script; every reported length is "
    "converted to um or mm at the reporting boundary, once.",
    "DELTA 14 (mesh refinement around the metal): the tutorial smooths each of "
    "x and y TWICE -- once at resolution/4 right after adding the thirds-rule "
    "lines at its trace edges, and again at resolution after adding the arm "
    "ends -- so its cells are up to four times finer across the trace than "
    "across the board. This builder smooths ONCE per axis, at res, and lays "
    "thirds-rule lines at the two 50 ohm feed edges only: the wide section's "
    "four edges (x = PATCH_X0 and PATCH_X1, the two impedance steps, and "
    "y = PATCH_Y_LO and PATCH_Y_HI) get an explicit mesh line each and no "
    "straddling pair. That is the script's builder, inherited verbatim and "
    "proved so, not a choice made here, and it is recorded rather than judged: "
    "the run reports what the grid made of each declared feed edge, per rung, "
    "in meta.stages.<stage>.feed_edges_realized (nearest line to each edge, the snap, and the "
    "width the two lines bracket), and the realized cell sizes per axis in "
    "meta.stages.<stage>.mesh_realized. There is no DELTA 13: this entry was written after the "
    "first twelve were numbered and the numbers are not reassigned, so that a "
    "citation of 'delta 9' keeps meaning what it meant.",
    "NOTHING ELSE: the boundary list ['PML_8','PML_8','MUR','MUR','PEC','MUR'], "
    "MSLPort with port 1 excite=-1, the zero-thickness PEC sheets on z = H_SUB, "
    "and SetGaussExcite's own centre-equals-corner form are the script's and "
    "the tutorial's alike. The thirds-rule OFFSETS are the same recipe too -- "
    "but where they are applied, and how often each axis is smoothed, is not: "
    "see delta 14.",
]

# ---------------------------------------------------------------------------
# The proof that _build_sheen_board IS validation/crossval/07_sheen_lpf.py's own
# run_openems builder. --self-check reads that file, slices the builder out
# between the two anchors below, applies COPY_SUBSTITUTIONS, and compares the
# result with this file's own slice character for character.
# ---------------------------------------------------------------------------
SCRIPT_REL_PATH = "validation/crossval/07_sheen_lpf.py"
SCRIPT_FUNCTION = "run_openems"
BUILDER_SLICE_FIRST_LINE = "    unit = 1.0  # work in metres"
BUILDER_SLICE_LAST_LINE = "               priority=10)"

# Three renames, and nothing else. FDTD/CSX are the script's local names for the
# solver and the geometry tree; this maker receives them as constructor
# arguments, so they are lower case. f_max is a parameter of run_openems; here
# the board's own F_MAX constant is used, and the script's only caller passes
# F_MAX.
COPY_SUBSTITUTIONS = [
    ("FDTD.", "fdtd."),
    ("CSX.", "csx."),
    ("f_max", "F_MAX"),
]
COPY_SUBSTITUTION_COUNTS = {"FDTD.": 3, "CSX.": 3, "f_max": 1}

# The excitation is NOT in the build block: the script sets it inside
# _openems_common_setup, which the build block deliberately excludes (delta 9
# declines that function's stop criteria). It is therefore derived separately,
# by the SAME declared renames, from that function's own SetGaussExcite
# statement. Without this the one line that decides what frequencies are
# launched would be the only line in either builder that nothing compares
# against -- changing it to F_MAX / 4 used to leave every check green.
EXCITATION_CALL = "SetGaussExcite"

# The TWO substitutions that turn the copied builder's body into the rung
# builder's body: the resolution factor, and the substrate's own z lines.
RUNG_SUBSTITUTIONS = [
    ("    res = min(res, H_SUB / 4.0)                          # >=4 substrate cells",
     "    res = min(res, H_SUB / 4.0) * resolution_factor      # 4 / 6 / 8 substrate cells"),
    ("mesh.AddLine('z', np.linspace(0, H_SUB, 5))",
     "mesh.AddLine('z', np.linspace(0, H_SUB, substrate_z_cells(resolution_factor) + 1))"),
]
RUNG_SUBSTITUTION_COUNTS = {
    "    res = min(res, H_SUB / 4.0)                          # >=4 substrate cells": 1,
    "mesh.AddLine('z', np.linspace(0, H_SUB, 5))": 1,
}


# ---------------------------------------------------------------------------
# STAGE B: the Sheen board. Copied from validation/crossval/07_sheen_lpf.py's
# run_openems, never edited there, with the three renames declared above.
# ---------------------------------------------------------------------------
def _build_sheen_board(ContinuousStructure, openEMS, MSLPort, *,
                       nrts: int | None, end_criteria: float | None):
    """The Sheen board at the script's own resolution (rung 1.0).

    The first four lines stand in for the script's
    ``_openems_common_setup(f_max)``: the solver is built from the NrTS and
    EndCriteria the CALLER hands in -- ``None`` meaning openEMS's own defaults,
    which is what every real pass here uses (delta 9) -- then
    ``SetGaussExcite(F_MAX/2, F_MAX/2)`` and an empty ContinuousStructure, both
    the script's. The rest of the body, from ``unit = 1.0`` on, is the copied
    builder, and that is the only part ``--self-check`` compares with the script.
    """
    kw = {}
    if nrts is not None:
        kw["NrTS"] = nrts
    if end_criteria is not None:
        kw["EndCriteria"] = end_criteria
    fdtd = openEMS(**kw)
    fdtd.SetGaussExcite(F_MAX / 2, F_MAX / 2)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)

    unit = 1.0  # work in metres
    fdtd.SetBoundaryCond(['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'])
    mesh = csx.GetGrid(); mesh.SetDeltaUnit(unit)

    res = C0 / (F_MAX * np.sqrt(EPS_R)) / 50.0          # ~lambda/50 transverse
    res = min(res, H_SUB / 4.0)                          # >=4 substrate cells
    tm = np.array([2 * res / 3, -res / 3]) / 4
    # x (propagation)
    mesh.AddLine('x', [0.0, LX, PATCH_X0, PATCH_X1])
    mesh.SmoothMeshLines('x', res)
    # y (transverse) - refine both feed edges + patch edges
    for yc in (IN_FEED_YC, OUT_FEED_YC):
        mesh.AddLine('y', yc + W_FEED / 2 + tm)
        mesh.AddLine('y', yc - W_FEED / 2 - tm)
    mesh.AddLine('y', [0.0, LY, PATCH_Y_LO, PATCH_Y_HI])
    mesh.SmoothMeshLines('y', res)
    # z: >=4 substrate cells + air
    mesh.AddLine('z', np.linspace(0, H_SUB, 5))
    mesh.AddLine('z', LZ)
    mesh.SmoothMeshLines('z', res)

    sub = csx.AddMaterial('duroid', epsilon=EPS_R)
    sub.AddBox([0.0, 0.0, 0.0], [LX, LY, H_SUB])
    pec = csx.AddMetal('PEC')

    port = [None, None]
    # port 1: excited, propagation +x, feed spans x 0 -> PATCH_X0 at IN_FEED_YC
    port[0] = fdtd.AddMSLPort(
        1, pec, [0.0, IN_FEED_YC - W_FEED / 2, H_SUB],
        [PATCH_X0, IN_FEED_YC + W_FEED / 2, 0.0], 'x', 'z', excite=-1,
        FeedShift=PORT_MARGIN, MeasPlaneShift=0.45 * PATCH_X0, priority=10)
    # port 2: passive, propagation -x, feed spans x LX -> PATCH_X1 at OUT_FEED_YC
    out_len = LX - PATCH_X1
    port[1] = fdtd.AddMSLPort(
        2, pec, [LX, OUT_FEED_YC - W_FEED / 2, H_SUB],
        [PATCH_X1, OUT_FEED_YC + W_FEED / 2, 0.0], 'x', 'z',
        MeasPlaneShift=0.45 * out_len, priority=10)
    # wide low-impedance patch (top surface)
    pec.AddBox([PATCH_X0, PATCH_Y_LO, H_SUB], [PATCH_X1, PATCH_Y_HI, H_SUB],
               priority=10)
    return fdtd, port[0], port[1]


# ---------------------------------------------------------------------------
# The SAME builder with the mesh rung passed in. Its body is
# _build_sheen_board's body with RUNG_SUBSTITUTIONS applied -- checked character
# for character by --self-check.
# ---------------------------------------------------------------------------
def _build_sheen_board_at_rung(ContinuousStructure, openEMS, MSLPort, *,
                               nrts: int | None, end_criteria: float | None,
                               resolution_factor: float):
    """The Sheen board at a given mesh rung."""
    kw = {}
    if nrts is not None:
        kw["NrTS"] = nrts
    if end_criteria is not None:
        kw["EndCriteria"] = end_criteria
    fdtd = openEMS(**kw)
    fdtd.SetGaussExcite(F_MAX / 2, F_MAX / 2)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)

    unit = 1.0  # work in metres
    fdtd.SetBoundaryCond(['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'])
    mesh = csx.GetGrid(); mesh.SetDeltaUnit(unit)

    res = C0 / (F_MAX * np.sqrt(EPS_R)) / 50.0          # ~lambda/50 transverse
    res = min(res, H_SUB / 4.0) * resolution_factor      # 4 / 6 / 8 substrate cells
    tm = np.array([2 * res / 3, -res / 3]) / 4
    # x (propagation)
    mesh.AddLine('x', [0.0, LX, PATCH_X0, PATCH_X1])
    mesh.SmoothMeshLines('x', res)
    # y (transverse) - refine both feed edges + patch edges
    for yc in (IN_FEED_YC, OUT_FEED_YC):
        mesh.AddLine('y', yc + W_FEED / 2 + tm)
        mesh.AddLine('y', yc - W_FEED / 2 - tm)
    mesh.AddLine('y', [0.0, LY, PATCH_Y_LO, PATCH_Y_HI])
    mesh.SmoothMeshLines('y', res)
    # z: >=4 substrate cells + air
    mesh.AddLine('z', np.linspace(0, H_SUB, substrate_z_cells(resolution_factor) + 1))
    mesh.AddLine('z', LZ)
    mesh.SmoothMeshLines('z', res)

    sub = csx.AddMaterial('duroid', epsilon=EPS_R)
    sub.AddBox([0.0, 0.0, 0.0], [LX, LY, H_SUB])
    pec = csx.AddMetal('PEC')

    port = [None, None]
    # port 1: excited, propagation +x, feed spans x 0 -> PATCH_X0 at IN_FEED_YC
    port[0] = fdtd.AddMSLPort(
        1, pec, [0.0, IN_FEED_YC - W_FEED / 2, H_SUB],
        [PATCH_X0, IN_FEED_YC + W_FEED / 2, 0.0], 'x', 'z', excite=-1,
        FeedShift=PORT_MARGIN, MeasPlaneShift=0.45 * PATCH_X0, priority=10)
    # port 2: passive, propagation -x, feed spans x LX -> PATCH_X1 at OUT_FEED_YC
    out_len = LX - PATCH_X1
    port[1] = fdtd.AddMSLPort(
        2, pec, [LX, OUT_FEED_YC - W_FEED / 2, H_SUB],
        [PATCH_X1, OUT_FEED_YC + W_FEED / 2, 0.0], 'x', 'z',
        MeasPlaneShift=0.45 * out_len, priority=10)
    # wide low-impedance patch (top surface)
    pec.AddBox([PATCH_X0, PATCH_Y_LO, H_SUB], [PATCH_X1, PATCH_Y_HI, H_SUB],
               priority=10)
    return fdtd, port[0], port[1]


# ---------------------------------------------------------------------------
# The copy proof's own plumbing: slice a builder's body between two anchors.
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    env = os.environ.get("RFX_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[4]


def _slice_builder(text: str, what: str) -> str:
    """The build block of ``text``, from the first anchor line to the last."""
    first = text.find(BUILDER_SLICE_FIRST_LINE)
    if first < 0:
        raise RuntimeError(f"{what}: the anchor {BUILDER_SLICE_FIRST_LINE!r} is not there")
    last = text.rfind(BUILDER_SLICE_LAST_LINE)
    if last < 0 or last < first:
        raise RuntimeError(f"{what}: the anchor {BUILDER_SLICE_LAST_LINE!r} is not there")
    return text[first:last + len(BUILDER_SLICE_LAST_LINE)]


def _script_function_source(name: str) -> str:
    """A named top-level function of ``07_sheen_lpf.py``, straight off disk."""
    import ast

    path = _repo_root() / SCRIPT_REL_PATH
    if not path.is_file():
        raise RuntimeError(
            f"the script this builder is copied from is not at {path} -- the copy "
            f"proof cannot run (set RFX_REPO_ROOT, or run from the repository)"
        )
    src = path.read_text()
    lines = src.split("\n")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    raise RuntimeError(f"{SCRIPT_REL_PATH} has no top-level {name}")


def _script_builder_slice() -> str:
    """``07_sheen_lpf.py``'s ``run_openems`` build block, straight off disk.

    The slice starts at that function's ``unit = 1.0`` line, which is AFTER its
    ``FDTD, CSX = _openems_common_setup(f_max)`` call and after its banner
    prints.  That exclusion is deliberate and is what keeps the copy proof
    honest now that delta 9 refuses the script's stop criteria: the solver
    object's NrTS/EndCriteria are created inside ``_openems_common_setup``, so
    if that call were inside the compared block the proof would be asserting
    that this maker adopts the 30000 / 1e-4 cap -- which it does not.  What the
    proof compares is the geometry, the mesh and the ports.  ``--self-check``
    asserts the exclusion rather than trusting the anchor.
    """
    body = _script_function_source(SCRIPT_FUNCTION)
    return _slice_builder(body, f"{SCRIPT_REL_PATH}::{SCRIPT_FUNCTION}")


def _statement_containing(text: str, needle: str, what: str) -> str:
    """The one line of ``text`` that contains ``needle``, verbatim."""
    hits = [ln for ln in text.split("\n") if needle in ln]
    if len(hits) != 1:
        raise RuntimeError(
            f"{what}: expected exactly one line containing {needle!r}, found "
            f"{len(hits)}"
        )
    return hits[0]


def _script_excitation_line() -> str:
    """The script's own ``SetGaussExcite`` statement, with the declared renames.

    Read off disk from ``_openems_common_setup``, not from the build block:
    that is where the script sets it, and the build block excludes that
    function on purpose (delta 9).
    """
    setup = _script_function_source(SCRIPT_SETUP_FUNCTION)
    line = _statement_containing(
        setup, EXCITATION_CALL, f"{SCRIPT_REL_PATH}::{SCRIPT_SETUP_FUNCTION}")
    for old_, new_ in COPY_SUBSTITUTIONS:
        line = line.replace(old_, new_)
    return line


def _builder_excitation_line(func) -> str:
    """A builder's own ``SetGaussExcite`` statement, from its source.

    Read from the body only (``_builder_body_after_kw`` starts at ``kw = {}``),
    so the docstring's mention of the same call is not what gets compared.
    """
    return _statement_containing(
        _builder_body_after_kw(func), EXCITATION_CALL, func.__name__)


def _copy_proof() -> dict:
    """Derive both builders from the script's, and report what matched.

    Two steps, each a character-for-character comparison:
      1. the script's build block + COPY_SUBSTITUTIONS  ==  _build_sheen_board's;
      2. _build_sheen_board's + RUNG_SUBSTITUTIONS      ==  _build_sheen_board_at_rung's.
    """
    import inspect

    script = _script_builder_slice()
    mine = _slice_builder(inspect.getsource(_build_sheen_board), "_build_sheen_board")
    rung = _slice_builder(inspect.getsource(_build_sheen_board_at_rung),
                          "_build_sheen_board_at_rung")

    derived = script
    for old, new in COPY_SUBSTITUTIONS:
        derived = derived.replace(old, new)
    derived_rung = mine
    for old, new in RUNG_SUBSTITUTIONS:
        derived_rung = derived_rung.replace(old, new)

    exc_script = _script_excitation_line()
    exc_mine = _builder_excitation_line(_build_sheen_board)
    exc_rung = _builder_excitation_line(_build_sheen_board_at_rung)

    return {
        "script_slice": script,
        "mine": mine,
        "rung": rung,
        "derived": derived,
        "derived_rung": derived_rung,
        "copy_matches": derived == mine,
        "rung_matches": derived_rung == rung,
        "copy_counts": {old: script.count(old) for old, _ in COPY_SUBSTITUTIONS},
        "rung_counts": {old: mine.count(old) for old, _ in RUNG_SUBSTITUTIONS},
        "excitation_script": exc_script,
        "excitation_mine": exc_mine,
        "excitation_rung": exc_rung,
        "excitation_matches": exc_script == exc_mine,
        "excitation_rung_matches": exc_script == exc_rung,
    }


# ---------------------------------------------------------------------------
# Pure-numpy geometry / mesh plan. No openEMS, no CSXCAD.
#
# WHAT IT CANNOT DO: CSXCAD's own SmoothMeshLines grades the transition between
# a fine and a coarse region, and it is a CSXCAD call. The estimate below only
# subdivides each gap into equal parts no wider than the target, so every line
# count it reports is a LOWER BOUND and every cell width it reports near a
# fine/coarse transition is an UPPER bound. MSLPort snaps FeedShift and
# MeasPlaneShift to a line itself, which is also a CSXCAD call. The run reads
# the realized lines back; that is the only place these questions are answered.
# ---------------------------------------------------------------------------
def resolution_m(resolution_factor: float) -> float:
    """The builder's own resolution at a rung, in metres."""
    res = C0 / (F_MAX * np.sqrt(EPS_R)) / 50.0
    return min(res, H_SUB / 4.0) * resolution_factor


def _feed_edges(res: float) -> dict:
    """The declared feed edges, and the thirds-rule lines added around them."""
    tm = np.array([2 * res / 3, -res / 3]) / 4
    out = {}
    for name, yc in (("in_feed", IN_FEED_YC), ("out_feed", OUT_FEED_YC)):
        out[name] = {
            "centre_mm": yc * 1e3,
            "declared_lo_mm": (yc - W_FEED / 2) * 1e3,
            "declared_hi_mm": (yc + W_FEED / 2) * 1e3,
            "declared_width_um": W_FEED * 1e6,
            "thirds_rule_offsets_um": [float(v * 1e6) for v in tm],
        }
    return out


def _feed_edges_against_lines(lines_y_um) -> dict:
    """Where the realized y lines put each declared feed edge.

    ``07_sheen_lpf.py``'s header records this quantity as a measurement: the
    thirds-rule lines straddle each edge instead of sitting on it, so the metal
    the grid can carry is narrower than the declared 2.413 mm. Recorded, not
    changed.
    """
    out = {}
    for name, yc in (("in_feed", IN_FEED_YC), ("out_feed", OUT_FEED_YC)):
        lo_declared = (yc - W_FEED / 2) * 1e6
        hi_declared = (yc + W_FEED / 2) * 1e6
        entry = {
            "declared_lo_um": lo_declared,
            "declared_hi_um": hi_declared,
            "declared_width_um": W_FEED * 1e6,
        }
        if lines_y_um is not None and np.size(lines_y_um):
            lo_line, lo_snap = _gate.nearest_line(lines_y_um, lo_declared)
            hi_line, hi_snap = _gate.nearest_line(lines_y_um, hi_declared)
            realized = hi_line - lo_line
            entry.update({
                "nearest_line_lo_um": lo_line, "snap_lo_um": lo_snap,
                "nearest_line_hi_um": hi_line, "snap_hi_um": hi_snap,
                "width_between_nearest_lines_um": realized,
                "width_error_pct": (realized - W_FEED * 1e6) / (W_FEED * 1e6) * 100.0,
            })
        else:
            entry.update({"nearest_line_lo_um": None, "snap_lo_um": None,
                          "nearest_line_hi_um": None, "snap_hi_um": None,
                          "width_between_nearest_lines_um": None,
                          "width_error_pct": None})
        out[name] = entry
    return out


def _plan(label: str, resolution_factor: float) -> dict:
    """The geometry and mesh the named Stage B rung builds, as numbers."""
    res = resolution_m(resolution_factor)
    tm = np.array([2 * res / 3, -res / 3]) / 4

    x = _smooth_estimate([0.0, LX, PATCH_X0, PATCH_X1], res)

    y = []
    for yc in (IN_FEED_YC, OUT_FEED_YC):
        y += list(yc + W_FEED / 2 + tm)
        y += list(yc - W_FEED / 2 - tm)
    y += [0.0, LY, PATCH_Y_LO, PATCH_Y_HI]
    y = _smooth_estimate(y, res)

    n_sub = substrate_z_cells(resolution_factor)
    z = _smooth_estimate(
        np.concatenate([np.linspace(0.0, H_SUB, n_sub + 1), [LZ]]), res)

    # PML_8 on the x faces only: it eats the outermost 8 cells at each end.
    pml_lo = float(x[B_PML_CELLS]) if x.size > B_PML_CELLS else float(x[-1])
    pml_hi = float(x[-1 - B_PML_CELLS]) if x.size > B_PML_CELLS else float(x[0])
    out_len = LX - PATCH_X1
    feed_x = PORT_MARGIN
    meas_x_p1 = 0.45 * PATCH_X0
    meas_x_p2 = LX - 0.45 * out_len

    return {
        "label": label,
        "resolution_factor": float(resolution_factor),
        "resolution_um": float(res * 1e6),
        "resolution_wavelength_term_um": float(C0 / (F_MAX * np.sqrt(EPS_R)) / 50.0
                                               * resolution_factor * 1e6),
        "resolution_thickness_term_um": float(H_SUB / 4.0 * resolution_factor * 1e6),
        "third_mesh_um": [float(v * 1e6) for v in tm],
        "box_mm": {"x": [0.0, LX * 1e3], "y": [0.0, LY * 1e3], "z": [0.0, LZ * 1e3]},
        "substrate_box_mm": {"x": [0.0, LX * 1e3], "y": [0.0, LY * 1e3],
                             "z": [0.0, H_SUB * 1e3]},
        "patch_box_mm": {"x": [PATCH_X0 * 1e3, PATCH_X1 * 1e3],
                         "y": [PATCH_Y_LO * 1e3, PATCH_Y_HI * 1e3],
                         "z": [H_SUB * 1e3, H_SUB * 1e3]},
        "patch_to_y_edge_mm": [PATCH_Y_LO * 1e3, (LY - PATCH_Y_HI) * 1e3],
        "mesh_lines_estimate": {"x": int(x.size), "y": int(y.size), "z": int(z.size)},
        "cells_estimate": int((x.size - 1) * (y.size - 1) * (z.size - 1)),
        "mesh_step_estimate_um": {
            "x_min": float(np.min(np.diff(x)) * 1e6), "x_max": float(np.max(np.diff(x)) * 1e6),
            "y_min": float(np.min(np.diff(y)) * 1e6), "y_max": float(np.max(np.diff(y)) * 1e6),
            "z_min": float(np.min(np.diff(z)) * 1e6), "z_max": float(np.max(np.diff(z)) * 1e6),
        },
        "substrate_z_cells_estimate": int(np.sum(
            (z >= -1e-12) & (z <= H_SUB + 1e-12)) - 1),
        "substrate_z_step_estimate_um": float(H_SUB / n_sub * 1e6),
        "port0_start_x_mm": 0.0,
        "port1_start_x_mm": LX * 1e3,
        "feed_shift_mm": PORT_MARGIN * 1e3,
        "measplane_shift_mm": [meas_x_p1 * 1e3, 0.45 * out_len * 1e3],
        "feed_x_mm": feed_x * 1e3,
        "measplane_x_mm": [meas_x_p1 * 1e3, meas_x_p2 * 1e3],
        "pml_inner_face_x_mm_estimate": [pml_lo * 1e3, pml_hi * 1e3],
        "pml_depth_mm_estimate": pml_lo * 1e3,
        "feed_inside_pml_estimate": bool(feed_x < pml_lo),
        "measplane_inside_pml_estimate": bool(meas_x_p1 < pml_lo or meas_x_p2 > pml_hi),
        "measplane_downstream_of_feed": bool(meas_x_p1 > feed_x),
        "feed_edges": _feed_edges(res),
    }


# The three rungs, by the short name the CLI and the job file use. One rung per
# cluster job is the normal way to run this now: with openEMS's own EndCriteria
# the coarse rung alone ran 107 minutes on 8 threads (VESSL 369367263243,
# 369367263269), and the VESSL rule prefers parallel jobs over one long serial
# one. `--merge` puts the parts back together.
RUNG_KEYS = {"coarse": "stage_b_coarse", "mid": "stage_b_mid", "fine": "stage_b_fine"}
RUNG_ORDER = ("coarse", "mid", "fine")
DEFAULT_RUNGS = ",".join(RUNG_ORDER)


def parse_rungs(spec: str) -> list:
    """``"coarse,fine"`` -> ``["stage_b_coarse", "stage_b_fine"]``, in rung order."""
    names = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not names:
        raise ValueError("--rungs is empty; give at least one of "
                         + ", ".join(RUNG_ORDER))
    unknown = [n for n in names if n not in RUNG_KEYS]
    if unknown:
        raise ValueError(f"--rungs names {unknown!r}; known rungs are "
                         + ", ".join(RUNG_ORDER))
    seen = []
    for n in RUNG_ORDER:
        if n in names and n not in seen:
            seen.append(n)
    return [RUNG_KEYS[n] for n in seen]


def rung_short_names(stage_names) -> list:
    """The short names of the rungs in a list of stage names, in rung order."""
    back = {v: k for k, v in RUNG_KEYS.items()}
    return [back[s] for s in RUNG_ORDER_STAGES if s in stage_names]


ACCEPT_TRUNCATION_DEFAULT = False


def stop_criteria_note(real_end_criteria, real_nrts,
                       accept_truncation: bool = ACCEPT_TRUNCATION_DEFAULT) -> str:
    """What stopped the real passes in this record, and why, in one string.

    With no override this says the library defaults ran. With one it says which
    override, and carries the measurement that motivated it -- so a reader of a
    record made with a looser criterion does not have to go and find out.
    """
    accepted = (
        " --accept-truncation was given, so a real pass that reached its NrTS cap "
        "was RECORDED (truncated: true, with the box energy it had reached at the "
        "cap) instead of failing the end-criteria gate. A record made this way is "
        "as long as the cap, not as long as a decay level: what that costs the "
        "numbers in it is not decided here." if accept_truncation else "")
    if real_end_criteria is None and real_nrts is None:
        return ("openEMS's own library defaults (~1e9 / 1e-5) on every real pass, "
                "Stage A and Stage B alike. No override was given. This is the "
                "declared design (delta 9); see B_REAL_NRTS / B_REAL_END_CRITERIA."
                + accepted)
    bits = []
    if real_end_criteria is not None:
        bits.append(f"--real-end-criteria {real_end_criteria!r}")
    if real_nrts is not None:
        bits.append(f"--real-nrts {real_nrts!r}")
    return ("made with " + " ".join(bits) + ": the tutorial's default 1e-5 took "
            "over 107 minutes on the coarsest rung on 8 threads (runs "
            "369367263243, 369367263269); 1e-4 is the repository's own ring-down "
            "level (rfx CLAUDE.md: end-of-run energy below -40 dB of the "
            "post-source peak)" + accepted)


STAGE_B_FACTORS = {
    "stage_b_coarse": B_COARSE_RESOLUTION_FACTOR,
    "stage_b_mid": B_MID_RESOLUTION_FACTOR,
    "stage_b_fine": B_FINE_RESOLUTION_FACTOR,
}
STAGE_NAMES = ("stage_a", "stage_b_coarse", "stage_b_mid", "stage_b_fine")
RUNG_ORDER_STAGES = tuple(RUNG_KEYS[n] for n in RUNG_ORDER)


def _stage_plans(fine_factor: float) -> dict:
    return {
        "stage_b_coarse": _plan("stage_b_coarse", B_COARSE_RESOLUTION_FACTOR),
        "stage_b_mid": _plan("stage_b_mid", B_MID_RESOLUTION_FACTOR),
        "stage_b_fine": _plan("stage_b_fine", fine_factor),
    }


def _print_plan(plan: dict) -> None:
    b, s, p = plan["box_mm"], plan["substrate_box_mm"], plan["patch_box_mm"]
    m, st = plan["mesh_lines_estimate"], plan["mesh_step_estimate_um"]
    print(f"  {plan['label']}  (resolution factor {plan['resolution_factor']:g})")
    print(f"    resolution              {plan['resolution_um']:.3f} um  "
          f"= min(lambda/50 {plan['resolution_wavelength_term_um']:.3f}, "
          f"h_sub/4 {plan['resolution_thickness_term_um']:.3f})   "
          f"third_mesh {plan['third_mesh_um'][0]:+.3f} / {plan['third_mesh_um'][1]:+.3f} um")
    print(f"    box (mm)                x [{b['x'][0]:.3f}, {b['x'][1]:.3f}]  "
          f"y [{b['y'][0]:.3f}, {b['y'][1]:.3f}]  z [{b['z'][0]:.3f}, {b['z'][1]:.3f}]")
    print(f"    substrate box (mm)      x [{s['x'][0]:.3f}, {s['x'][1]:.3f}]  "
          f"y [{s['y'][0]:.3f}, {s['y'][1]:.3f}]  z [{s['z'][0]:.3f}, {s['z'][1]:.3f}]")
    print(f"    wide section (mm)       x [{p['x'][0]:.3f}, {p['x'][1]:.3f}]  "
          f"y [{p['y'][0]:.3f}, {p['y'][1]:.3f}]  on z {p['z'][0]:.3f} "
          f"(clears the y walls by {plan['patch_to_y_edge_mm'][0]:.3f} / "
          f"{plan['patch_to_y_edge_mm'][1]:.3f} mm)")
    print(f"    mesh lines (estimate)   x {m['x']}  y {m['y']}  z {m['z']}   "
          f"-> cells >= {plan['cells_estimate']:,}")
    print(f"    cell size (estimate)    x {st['x_min']:.2f}-{st['x_max']:.2f}  "
          f"y {st['y_min']:.2f}-{st['y_max']:.2f}  z {st['z_min']:.2f}-{st['z_max']:.2f} um")
    print(f"    substrate z cells       {plan['substrate_z_cells_estimate']} of "
          f"{plan['substrate_z_step_estimate_um']:.2f} um across the 794 um board")
    print(f"    ports                   port0 start x {plan['port0_start_x_mm']:.3f} mm, "
          f"port1 start x {plan['port1_start_x_mm']:.3f} mm")
    print(f"    FeedShift               {plan['feed_shift_mm']:.3f} mm "
          f"-> feed at x {plan['feed_x_mm']:.3f} mm (port 2 declares none)")
    print(f"    MeasPlaneShift          {plan['measplane_shift_mm'][0]:.3f} / "
          f"{plan['measplane_shift_mm'][1]:.3f} mm -> measurement planes at x "
          f"{plan['measplane_x_mm'][0]:.3f} / {plan['measplane_x_mm'][1]:.3f} mm")
    print(f"    x PML_8 inner faces     {plan['pml_inner_face_x_mm_estimate'][0]:.3f} / "
          f"{plan['pml_inner_face_x_mm_estimate'][1]:.3f} mm "
          f"(depth {plan['pml_depth_mm_estimate']:.3f} mm, estimate)")
    print(f"    feed inside the PML     {plan['feed_inside_pml_estimate']}")
    print(f"    meas planes inside PML  {plan['measplane_inside_pml_estimate']}")
    print(f"    meas plane downstream of the feed  {plan['measplane_downstream_of_feed']}")
    for name, fe in plan["feed_edges"].items():
        print(f"    {name:9s} declared y {fe['declared_lo_mm']:.4f}-"
              f"{fe['declared_hi_mm']:.4f} mm ({fe['declared_width_um']:.1f} um wide); "
              f"thirds-rule lines at the edges {fe['thirds_rule_offsets_um'][0]:+.3f} / "
              f"{fe['thirds_rule_offsets_um'][1]:+.3f} um")


# ---------------------------------------------------------------------------
# The features this record reports. NONE of them is gated.
# ---------------------------------------------------------------------------
def _stage_b_features(sf):
    """The stopband minimum, the passband mean and the -3 dB corner.

    Reported, not judged. ``sf`` is
    ``validation/crossval/comparators/spectral_features.py`` -- the repository's
    one estimator module, so the number written here is the number a test reads
    back.
    """
    def features(freqs_ghz, s11, s21) -> dict:
        mag = np.abs(s21)
        out: dict = {}
        try:
            null = sf.refined_extremum(freqs_ghz, mag, NULL_BAND_GHZ[0],
                                       NULL_BAND_GHZ[1], transform="log")
            out["null"] = {
                "bin_f_ghz": float(null["bin_f"]),
                "refined_f_ghz": float(null["refined_f"]),
                "depth_db": float(null["depth_db"]),
                "sub_bin_shift_bins": float(null["sub_bin_shift"]),
                "bin_width_ghz": float(null["bin_width"]),
                "band_ghz": list(NULL_BAND_GHZ),
                "estimator": "validation/crossval/comparators/spectral_features.py::"
                             "refined_extremum, transform='log'",
                "what_it_is": (
                    "the DEEPEST minimum in 5-10 GHz. This board's stopband is a "
                    "double transmission zero near 7 and 8 GHz, so this names one "
                    "member of a pair; read the curve for both. Not gated."
                ),
            }
        except Exception as exc:
            out["null"] = {"error": repr(exc)}
        try:
            pb = (freqs_ghz >= PASSBAND_GHZ[0]) & (freqs_ghz <= PASSBAND_GHZ[1])
            mean_lin = float(np.mean(mag[pb]))
            out["passband"] = {
                "band_ghz": list(PASSBAND_GHZ),
                "n_bins": int(np.count_nonzero(pb)),
                "mean_s21_linear": mean_lin,
                "mean_db": float(20.0 * np.log10(mean_lin + 1e-30)),
            }
            level = mean_lin / np.sqrt(2.0)
            fc = sf.level_crossing(freqs_ghz, mag, level,
                                   f_min=CUTOFF_SEARCH_FROM_GHZ)
            out["cutoff_3db"] = {
                "f_ghz": None if fc is None else float(fc),
                "level_linear": level,
                "level_db_below_passband_mean": CUTOFF_LEVEL_DB,
                "search_from_ghz": CUTOFF_SEARCH_FROM_GHZ,
                "estimator": "validation/crossval/comparators/spectral_features.py::"
                             "level_crossing, falling, linear interpolation between "
                             "the two bracketing bins",
                "what_it_is": (
                    "the first falling crossing above 2 GHz of the passband mean "
                    "|S21| divided by sqrt(2). Not gated."
                ),
            }
        except Exception as exc:
            out["passband_error"] = repr(exc)
        return out
    return features


# ---------------------------------------------------------------------------
# The run -- this case's Stage B, on the shared runner
# ---------------------------------------------------------------------------
def _run_stage_b(*, label: str, sim_root: str, threads: int,
                 resolution_factor: float, sf,
                 real_nrts=None, real_end_criteria=None,
                 accept_truncation: bool = ACCEPT_TRUNCATION_DEFAULT
                 ) -> tuple[dict, dict]:
    """The Sheen board at a rung, with the shared module's gates around it.

    ``accept_truncation`` defaults to off: a real pass that hits its NrTS cap
    is a fired gate, no record. Turned on (only by ``--accept-truncation``), such
    a pass is recorded with ``truncated: true``, the box energy it had reached,
    and a note. Stage A never sees it.

    ``real_nrts`` / ``real_end_criteria`` default to ``None``, which means the
    declared design: pass nothing, so openEMS's own ~1e9 / 1e-5 apply (delta 9).
    A caller that overrides them is recorded doing so, per stage block and in
    ``meta.stop_criteria_note``. STAGE A IS NEVER OVERRIDDEN -- it is the gate,
    it runs the tutorial's own model, and it takes 44 s.
    """
    res = resolution_m(resolution_factor)
    out_len = LX - PATCH_X1

    def build(ContinuousStructure, openEMS, MSLPort, *, nrts, end_criteria):
        return _build_sheen_board_at_rung(
            ContinuousStructure, openEMS, MSLPort,
            nrts=nrts, end_criteria=end_criteria,
            resolution_factor=resolution_factor)

    def mesh_realized_fn(lines):
        return _gate._mesh_realized(
            _gate.lines_in_um(lines, CSX_UNIT_M),
            substrate_thickness_um=H_SUB * 1e6)

    def meta_extra_fn(*, lines, port0, port1) -> dict:
        lines_um = _gate.lines_in_um(lines, CSX_UNIT_M)
        x = None if lines_um is None else lines_um["x"]
        y = None if lines_um is None else lines_um["y"]
        pml = {"cells": B_PML_CELLS, "faces": "x min and x max only"}
        if x is not None and x.size > B_PML_CELLS:
            pml.update({
                "inner_face_lo_mm": float(x[B_PML_CELLS]) / 1e3,
                "inner_face_hi_mm": float(x[-1 - B_PML_CELLS]) / 1e3,
                "depth_lo_mm": float(x[B_PML_CELLS] - x[0]) / 1e3,
                "depth_hi_mm": float(x[-1] - x[-1 - B_PML_CELLS]) / 1e3,
            })
        return {
            "model": "the Sheen 1990 stepped-impedance low-pass filter",
            "resolution_um": float(res * 1e6),
            "resolution_factor": float(resolution_factor),
            "csx_unit_m": CSX_UNIT_M,
            "box_mm": {"x": [0.0, LX * 1e3], "y": [0.0, LY * 1e3], "z": [0.0, LZ * 1e3]},
            "substrate_box_mm": {"x": [0.0, LX * 1e3], "y": [0.0, LY * 1e3],
                                 "z": [0.0, H_SUB * 1e3]},
            "port0": _gate._port_declared_and_snap(
                x, start_x_um=0.0, direction=+1.0,
                feed_shift_um=PORT_MARGIN * 1e6,
                measplane_shift_um=0.45 * PATCH_X0 * 1e6, port_obj=port0,
                csx_unit_m=CSX_UNIT_M),
            "port1": _gate._port_declared_and_snap(
                x, start_x_um=LX * 1e6, direction=-1.0,
                feed_shift_um=0.0,
                measplane_shift_um=0.45 * out_len * 1e6, port_obj=port1,
                csx_unit_m=CSX_UNIT_M),
            "excitation_energy_peak_note": (
                "read from port 1's uf_inc AFTER both CalcPort passes, so on "
                "Stage B it is the ref_impedance = 50 pass's incident channel. "
                "Stage A runs only the unreferenced pass, so its own value is "
                "that pass's. The guard the number feeds is scale-free -- it "
                "rejects exact zero and non-finite only -- so the two are "
                "comparable as a pass/fail, not as a magnitude."
            ),
            "port1_feed_shift_note": (
                "port 2 declares no FeedShift; it is the passive port, so the "
                "0.0 above is what the script declares, not a measurement"
            ),
            "substrate_z_cells_declared": substrate_z_cells(resolution_factor),
            "feed_edges_realized": _feed_edges_against_lines(y),
            "pml_realized": pml,
            "nrts_declared": (
                f"{real_nrts!r} -- given on the command line, NOT the declared "
                f"design; see stop_criteria_note" if real_nrts is not None
                else ("openEMS library default (~1e9) -- the retired script's cap "
                      f"of {SCRIPT_NRTS_CAP} is not carried")),
            "end_criteria_declared": (
                f"{real_end_criteria!r} -- given on the command line, NOT the "
                f"declared design; see stop_criteria_note"
                if real_end_criteria is not None
                else ("openEMS library default (1e-5) -- the retired script's cap "
                      f"of {SCRIPT_END_CRITERIA_CAP} is not carried")),
            "stop_criteria_note": stop_criteria_note(
                real_end_criteria, real_nrts, accept_truncation),
            "calcport_grid": f"linspace({F_LO}, {F_MAX}, {B_N_FREQS})",
            "calcport_passes": (
                "two: pass 1 with no ref_impedance (re_z0 = Re(port.Z_ref)), then "
                f"pass 2 with ref_impedance = {B_CALCPORT_REF_IMPEDANCE}; S comes "
                "from pass 2"
            ),
            "plan_estimate": _plan(label, resolution_factor),
        }

    return _gate.run_stage(
        label=label, sim_root=sim_root, threads=threads, build=build,
        freqs_hz=np.linspace(F_LO, F_MAX, B_N_FREQS),
        witness_band_hz=WITNESS_BAND_HZ, passivity_tol=PASSIVITY_TOL,
        real_nrts=real_nrts, real_end_criteria=real_end_criteria,
        mesh_realized_fn=mesh_realized_fn, meta_extra_fn=meta_extra_fn,
        features_fn=_stage_b_features(sf),
        calcport_ref_impedance=B_CALCPORT_REF_IMPEDANCE,
        record_deficit=True,
        accept_truncation=accept_truncation)


def _build_artifact(records: dict, stage_meta: dict, stage_a_gate: dict,
                    stages: list, *, failed_gate: str | None = None,
                    real_nrts=None, real_end_criteria=None,
                    accept_truncation: bool = ACCEPT_TRUNCATION_DEFAULT,
                    stage_names=None) -> dict:
    """This case's meta block, on the shared record writer.

    ``stage_names`` is which stage blocks the record carries. A job that solved
    one rung passes that rung (plus Stage A), so the record has no null blocks
    standing in for rungs nobody ran.
    """
    names = tuple(stage_names) if stage_names is not None else STAGE_NAMES
    return _gate.build_artifact(
        records, stage_meta, stage_a_gate, stages,
        stage_names=names,
        produced_by="tests/crossval/sheen_lpf/reference/make_openems_reference.py",
        failed_gate=failed_gate,
        meta_common={
            "rfx_commit": os.environ.get("RFX_COMMIT"),
            "rungs_in_this_record": [n for n in RUNG_ORDER
                                     if RUNG_KEYS[n] in names],
            "stop_criteria_note": stop_criteria_note(
                real_end_criteria, real_nrts, accept_truncation),
            "structure": (
                "the Sheen 1990 stepped-impedance microstrip low-pass filter: "
                "RT/Duroid eps_r 2.2, h 0.794 mm; two 2.413 mm wide 50 ohm feeds "
                "joined by one 20.320 x 2.540 mm low-impedance section. IEEE Trans. "
                "MTT 38(7):849-857, July 1990; coordinates via Elsherbeni-Demir "
                "Sec. 6.2 as transcribed in roseengineering/rffdtd examples/"
                "lowpass.py, copied from validation/crossval/07_sheen_lpf.py."
            ),
            "stage_a_is": (
                "openEMS python/Tutorials/MSL_NotchFilter.py, verbatim -- the "
                "reproduce gate, a DIFFERENT structure on a different substrate. "
                "It is not a measurement of the Sheen board and is not comparable "
                "with one."
            ),
            "comparability": (
                "The S11/S21 PHASES are referenced at this board's own measurement "
                "planes, 0.45 x 12.466 mm = 5.610 mm in from each port face, and are "
                "NOT comparable with a record taken on a different box or a different "
                "MeasPlaneShift; the MAGNITUDES are."
            ),
            "delta_list": DELTA_LIST,
            "boundary": B_BOUNDARY,
            "excitation": (
                f"{_builder_excitation_line(_build_sheen_board).strip()}"
                f"  -- the builder's own source line, derived from "
                f"{SCRIPT_REL_PATH}::{SCRIPT_SETUP_FUNCTION} by the renames in "
                f"COPY_SUBSTITUTIONS and compared with it character for character "
                f"by --self-check. F_MAX evaluates to {F_MAX!r} Hz."
            ),
            "nrts": ("openEMS's library default (~1e9) on EVERY real pass, Stage A "
                     "and Stage B alike; 200 on every smoke pass. The retired "
                     f"script's cap of {SCRIPT_NRTS_CAP} is NOT carried -- see "
                     "delta 9."),
            "end_criteria": ("openEMS's library default (1e-5) on EVERY real pass, "
                             "Stage A and Stage B alike; 0.0 on every smoke pass. The "
                             f"retired script's {SCRIPT_END_CRITERIA_CAP} is NOT "
                             "carried -- see delta 9."),
            "calcport_grid": f"Stage B linspace({F_LO}, {F_MAX}, {B_N_FREQS})",
            "calcport_passes": (
                "Stage B runs TWO passes: pass 1 with no ref_impedance, whose "
                "Re(port.Z_ref) is recorded as re_z0, then pass 2 with "
                f"ref_impedance = {B_CALCPORT_REF_IMPEDANCE}, from which S is taken. "
                "Stage A runs the tutorial's single unreferenced pass."
            ),
            "null_estimator": ("validation/crossval/comparators/spectral_features.py::"
                               "refined_extremum, transform='log', band "
                               f"{NULL_BAND_GHZ[0]:g}-{NULL_BAND_GHZ[1]:g} GHz; "
                               "reported, not gated"),
            "cutoff_estimator": ("validation/crossval/comparators/spectral_features.py::"
                                 "level_crossing, falling, from "
                                 f"{CUTOFF_SEARCH_FROM_GHZ:g} GHz, at the passband "
                                 f"({PASSBAND_GHZ[0]:g}-{PASSBAND_GHZ[1]:g} GHz) mean "
                                 "|S21| / sqrt(2); reported, not gated"),
            "witness_band_ghz": list(WITNESS_BAND_GHZ),
            "passivity_tol": 1.0 + PASSIVITY_TOL,
            "passivity_witness": (
                "max(|S11|^2+|S21|^2) over the witness band only, 2-12 GHz. The "
                "witness bounds the EXCESS above unity; it says nothing about a sum "
                "BELOW unity, and the committed record this replaces runs 0.78-1.00 "
                "across this band and falls to 0.41 by 20 GHz. Every bin's energy sum "
                "is recorded, with the band and whole-grid maxima AND minima and the "
                "maxima over 0-1 GHz, 1-2 GHz and above the band, so the deficit is "
                "readable. What it means is not decided here."
            ),
            "what_is_not_here": (
                "no comparison with rfx, no comparison with the Palace FEM record, no "
                "verdict on either. This script writes one solver's result with its "
                "provenance."
            ),
        })


# ---------------------------------------------------------------------------
# --dry-run and --self-check
# ---------------------------------------------------------------------------
def _print_delta_list() -> None:
    print("DELTA LIST -- how Stage B differs from the tutorial Stage A runs:")
    for i, line in enumerate(DELTA_LIST, start=1):
        print(f"  [{i}] {line}")


def _stages_for(stage: str, rung_stages=None) -> list:
    """Which stages this invocation runs.

    ``rung_stages`` defaults to all three. Stage A is in every job that asks for
    it: it is the reproduce gate, it runs the tutorial's own model, and it costs
    44 s, so splitting the rungs across jobs does not make it optional.
    """
    rungs = list(RUNG_ORDER_STAGES) if rung_stages is None else list(rung_stages)
    if stage == "A":
        return ["stage_a"]
    if stage == "B":
        return rungs
    return ["stage_a"] + rungs


def _dry_run(stage: str, fine_factor: float, rung_stages=None,
             real_nrts=None, real_end_criteria=None,
             accept_truncation: bool = ACCEPT_TRUNCATION_DEFAULT) -> int:
    order = _stages_for(stage, rung_stages)
    order_rungs = [s for s in order if s in RUNG_ORDER_STAGES]
    print("=" * 78)
    print("The Sheen low-pass filter -- openEMS reference maker, DRY RUN (no solver)")
    print("=" * 78)
    print("STAGE A -- the reproduce gate, the shared tutorial gate, unchanged")
    print(f"  tutorial        {REPRODUCE_GATE_RECORD['tutorial']['repo']} "
          f"{REPRODUCE_GATE_RECORD['tutorial']['path']}")
    print(f"  attribution     {REPRODUCE_GATE_RECORD['tutorial']['attribution']}")
    print(f"  recorded notch  {REPRODUCE_GATE_RECORD['reproduced_f_notch_hz']/1e9:.4f} GHz "
          f"measured vs {REPRODUCE_GATE_RECORD['analytic_f_notch_hz']/1e9:.4f} GHz analytic, "
          f"{REPRODUCE_GATE_RECORD['reproduced_f_notch_dev_pct']:.2f} % "
          f"(VESSL {REPRODUCE_GATE_RECORD['vessl_run_id']}) -- an AUDIT TRAIL for the "
          f"tutorial, never this run's gate")
    print(f"  gate            {STAGE_A_GATE['f_notch_lo_hz']/1e9:.4f} .. "
          f"{STAGE_A_GATE['f_notch_hi_hz']/1e9:.4f} GHz (0.80-1.05 x the analytic "
          f"{F_NOTCH_AN_HZ/1e9:.4f} GHz) AND at least "
          f"{abs(STAGE_A_MIN_DEPTH_DB):.0f} dB deep")
    print(f"  measured in     Stage A, run in THIS invocation, on the tutorial's own "
          f"1601-point grid, witness band "
          f"{STAGE_A_NOTCH_BAND_GHZ[0]:.1f}-{STAGE_A_NOTCH_BAND_GHZ[1]:.1f} GHz")
    print()
    print("STAGE B -- the Sheen board")
    print(f"  substrate       eps_r {EPS_R}, h {H_SUB*1e3:.3f} mm")
    print(f"  50-ohm feed     {W_FEED*1e3:.3f} mm wide, centres y "
          f"{IN_FEED_YC*1e3:.4f} and {OUT_FEED_YC*1e3:.4f} mm")
    print(f"  wide section    {PATCH_TRV_LEN*1e3:.3f} x {PATCH_LEN_PROP*1e3:.3f} mm")
    print(f"  feed extension  {EXTEND_FEED*1e3:.1f} mm per side")
    print(f"  domain          {LX*1e3:.3f} x {LY*1e3:.3f} x {LZ*1e3:.3f} mm "
          f"(x prop, y trv, z)")
    print(f"  boundary        {B_BOUNDARY}")
    print(f"  excitation      {_builder_excitation_line(_build_sheen_board).strip()}"
          f"   (the builder's own source line; F_MAX = {F_MAX:.4g} Hz, so "
          f"{F_MAX/2:.4g} Hz centre and corner)")
    print(f"  NrTS / EndCrit  openEMS library defaults (~1e9 / 1e-5) on every real "
          f"pass, as Stage A; 200 / 0.0 on the smoke pass. The retired script's "
          f"{SCRIPT_NRTS_CAP} / {SCRIPT_END_CRITERIA_CAP} cap is NOT carried "
          f"(delta 9)")
    print(f"  stop criteria   "
          f"{stop_criteria_note(real_end_criteria, real_nrts, accept_truncation)}")
    print(f"  rungs requested {', '.join(rung_short_names(order_rungs)) or '(none)'}"
          f"   -- Stage A runs in every job that asks for it")
    print(f"  CalcPort grid   linspace({F_LO:.4g}, {F_MAX:.4g}, {B_N_FREQS}), TWO "
          f"passes (Z_ref, then ref_impedance={B_CALCPORT_REF_IMPEDANCE})")
    print(f"  witness band    max(|S11|^2+|S21|^2) <= {1.0 + PASSIVITY_TOL:.2f} over "
          f"{WITNESS_BAND_GHZ[0]:.1f}-{WITNESS_BAND_GHZ[1]:.1f} GHz (WITNESS_BAND_HZ). "
          f"The excess only: the committed record's sum runs 0.78-1.00 there and "
          f"falls to 0.41 by 20 GHz, and that deficit is recorded, not judged")
    print(f"  reported only   the deepest minimum in "
          f"{NULL_BAND_GHZ[0]:.1f}-{NULL_BAND_GHZ[1]:.1f} GHz, the passband mean over "
          f"{PASSBAND_GHZ[0]:.1f}-{PASSBAND_GHZ[1]:.1f} GHz, and the "
          f"{CUTOFF_LEVEL_DB:.2f} dB corner above {CUTOFF_SEARCH_FROM_GHZ:.1f} GHz. "
          f"NO gate on any of the three")
    print("on a failed gate  the record built so far, the energy-sum numbers and "
          "failed_gate go to <output stem>_FAILED.json before the non-zero exit")
    print()
    _print_delta_list()
    print()
    plans = _stage_plans(fine_factor)
    print("STAGE PLAN -- the geometry each stage builds:")
    if "stage_a" in order:
        _gate.print_tutorial_plan(_gate.tutorial_plan("stage_a", _gate.A_MSL_LENGTH_UM, 1.0))
        print()
    for name in order:
        if name in plans:
            _print_plan(plans[name])
            print()
    print("WHAT THIS DRY RUN CANNOT TELL YOU: every mesh-line count and cell size "
          "above is a lower/upper bound from a pure-numpy subdivision. CSXCAD's own "
          "SmoothMeshLines grades fine-to-coarse transitions and adds lines this "
          "estimate does not model, and whether FeedShift/MeasPlaneShift and the "
          "feed edges land on mesh lines is decided by that same call. The real "
          "values are written into the record's meta block by the run itself.")
    return 0


def _self_check(fine_factor: float) -> int:
    failures = []
    notes = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        print(f"  [{'ok ' if ok else 'FAIL'}] {what}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(what)

    print("=" * 78)
    print("The Sheen low-pass filter -- openEMS reference maker, SELF-CHECK (pure numpy)")
    print("=" * 78)

    print("constants -- equal to validation/crossval/07_sheen_lpf.py's own:")
    check(EPS_R == 2.2 and H_SUB == 0.794e-3, "substrate eps_r 2.2, h 0.794 mm")
    check(W_FEED == 2.413e-3, "50-ohm feed width 2.413 mm")
    check(abs(PATCH_TRV_LEN - 20.320e-3) < 1e-12
          and abs(PATCH_LEN_PROP - 2.540e-3) < 1e-12,
          "wide section 20.320 x 2.540 mm",
          f"{PATCH_TRV_LEN*1e3:.3f} x {PATCH_LEN_PROP*1e3:.3f} mm")
    check(abs(IN_FEED_XS_C - 7.8565e-3) < 1e-12 and abs(OUT_FEED_XS_C - 14.4635e-3) < 1e-12,
          "feed centres in the Sheen frame 7.8565 / 14.4635 mm")
    check(abs(IN_FEED_LEN - 8.466e-3) < 1e-12 and abs(OUT_FEED_LEN - 8.466e-3) < 1e-12,
          "both feed runs 8.466 mm", f"{IN_FEED_LEN*1e3:.3f} / {OUT_FEED_LEN*1e3:.3f} mm")
    check(EXTEND_FEED == 4.0e-3 and Y_CLEAR == 3.0e-3 and Z_AIR == 3.0e-3,
          "EXTEND_FEED 4.0, Y_CLEAR 3.0, Z_AIR 3.0 mm")
    check(F_MAX == 20.0e9 and F_LO == 0.5e9 and B_N_FREQS == 801,
          "0.5-20 GHz on 801 points")
    check(abs(PATCH_X0 - 12.466e-3) < 1e-12 and abs(PATCH_X1 - 15.006e-3) < 1e-12,
          "PATCH_X0 / PATCH_X1 = 12.466 / 15.006 mm",
          f"{PATCH_X0*1e3:.3f} / {PATCH_X1*1e3:.3f} mm")
    check(abs(LX - 27.472e-3) < 1e-12 and abs(LY - 26.320e-3) < 1e-12
          and abs(LZ - 3.794e-3) < 1e-12,
          "domain 27.472 x 26.320 x 3.794 mm",
          f"{LX*1e3:.3f} x {LY*1e3:.3f} x {LZ*1e3:.3f} mm")
    check(abs(Y_SHIFT - 2.0e-3) < 1e-12
          and abs(IN_FEED_YC - 9.8565e-3) < 1e-12
          and abs(OUT_FEED_YC - 16.4635e-3) < 1e-12
          and abs(PATCH_Y_LO - 3.0e-3) < 1e-12
          and abs(PATCH_Y_HI - 23.320e-3) < 1e-12,
          "Y_SHIFT 2.0 mm; feed centres 9.8565 / 16.4635; patch y 3.000-23.320 mm")
    check(PORT_MARGIN == 2.5e-3, "PORT_MARGIN 2.5 mm")
    check(B_REAL_NRTS is None and B_REAL_END_CRITERIA is None,
          "the real pass passes NO NrTS/EndCriteria, so openEMS's own defaults "
          "(~1e9 / 1e-5) apply, exactly as Stage A runs -- the DECLARED design, "
          "which --real-end-criteria / --real-nrts override without changing",
          f"{B_REAL_NRTS!r} / {B_REAL_END_CRITERIA!r}")

    print("the rung selector and the stop-criteria override:")
    check(parse_rungs(DEFAULT_RUNGS) == list(RUNG_ORDER_STAGES),
          "the default --rungs is all three, in rung order",
          f"{parse_rungs(DEFAULT_RUNGS)}")
    check(parse_rungs("fine,coarse") == ["stage_b_coarse", "stage_b_fine"],
          "a subset comes back in rung order, whatever order it was given in")
    bad = []
    for spec in ("", "  ", "middle", "coarse,middle"):
        try:
            parse_rungs(spec)
        except ValueError:
            bad.append(spec)
    check(len(bad) == 4, "an empty or unknown --rungs is refused, not silently "
                         "dropped", f"refused {bad!r}")
    check(_stages_for("both", parse_rungs("mid")) == ["stage_a", "stage_b_mid"],
          "a one-rung job still runs Stage A -- it is the gate, and it costs 44 s")
    check(_stages_for("B", parse_rungs("mid")) == ["stage_b_mid"],
          "--stage B still skips the gate, whatever the rungs are")
    default_note = stop_criteria_note(None, None)
    check("No override was given" in default_note
          and "library defaults" in default_note,
          "with no override the note says the library defaults ran")
    over = stop_criteria_note(1e-4, None)
    check(over.startswith("made with --real-end-criteria "),
          "an overridden record's note opens with what was given", over[:46])
    check("107 minutes on the coarsest rung on 8 threads" in over
          and "369367263243" in over and "369367263269" in over
          and "-40 dB of the post-source peak" in over,
          "and carries the measurement that motivated it, with its run ids")
    both_over = stop_criteria_note(1e-4, 60000)
    check("--real-end-criteria" in both_over and "--real-nrts 60000" in both_over,
          "both overrides appear when both are given")
    check(ACCEPT_TRUNCATION_DEFAULT is False,
          "--accept-truncation is OFF by default: a real pass that hits its NrTS "
          "cap is a FIRED GATE and no record, unless someone asks for one",
          f"{ACCEPT_TRUNCATION_DEFAULT!r}")
    check("--accept-truncation" not in stop_criteria_note(1e-4, 60000),
          "a record made without the flag does not mention it")
    accepted = stop_criteria_note(1e-4, 60000, True)
    check("--accept-truncation was given" in accepted
          and "truncated: true" in accepted,
          "and a record made WITH it says so, and says what it means for the "
          "record's length")
    import inspect as _insp
    check(_insp.signature(_run_stage_b).parameters["accept_truncation"].default
          is ACCEPT_TRUNCATION_DEFAULT,
          "the Stage B runner's own default is the same one")
    check("accept_truncation" not in _insp.signature(_gate.run_stage_a).parameters,
          "Stage A's runner takes no such parameter at all -- the gate cannot be "
          "turned off for the reproduce stage even by mistake")
    del _insp
    check(B_BOUNDARY == ["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"], "boundary")
    check(C0 == 2.99792458e8,
          "C0 is the Sheen script's own, not the tutorial gate's 2.998e8")
    check(abs(resolution_m(1.0) * 1e6 - 198.5) < 1e-9,
          "res at rung 1.0 is h_sub/4 = 198.5 um, not the 202.1 um wavelength term",
          f"{resolution_m(1.0)*1e6:.4f} um (wavelength term "
          f"{C0/(F_MAX*np.sqrt(EPS_R))/50.0*1e6:.4f} um)")

    print("the reproduce gate is the TUTORIAL's, and its own analytic frequency:")
    # 07_sheen_lpf.py's own F_NOTCH_TUTORIAL_DECLARED_HZ, recomputed here from
    # its own constants exactly as that script computes it. Its do_not_repeat
    # says the gate must be checked against THIS number; the shared module's
    # F_NOTCH_AN_HZ must therefore be the same number, or the two paths gate
    # different things.
    _tut_c0 = 2.998e8
    _tut_u = 600e-6 / 254e-6
    _tut_eps_eff = (3.66 + 1.0) / 2.0 + (3.66 - 1.0) / 2.0 * (1.0 + 12.0 / _tut_u) ** -0.5
    f_tut = _tut_c0 / (4.0 * 12e-3 * np.sqrt(_tut_eps_eff))
    check(abs(f_tut - F_NOTCH_AN_HZ) < 1.0,
          "07_sheen_lpf.py's own F_NOTCH_TUTORIAL_DECLARED_HZ recomputes to the "
          "shared module's F_NOTCH_AN_HZ",
          f"{f_tut:.4f} Hz vs {F_NOTCH_AN_HZ:.4f} Hz")
    check(abs(F_NOTCH_AN_HZ - REPRODUCE_GATE_RECORD["analytic_f_notch_hz"]) < 1.0,
          "and to the analytic frequency the recorded reproduction was measured "
          "against")
    check(REPRODUCE_GATE_RECORD.get("reproduced_f_notch_hz") is not None,
          "the recorded reproduction carries a measured number, not None -- the "
          "status the Sheen script's own gate record still sits at is UNRUN")

    print("the declared deltas:")
    check(len(DELTA_LIST) == 14, "delta list has its fourteen declared entries",
          f"{len(DELTA_LIST)}")
    for i, key in enumerate([
            "DELTA 1 (substrate)", "DELTA 2 (trace)", "DELTA 3 (offset feeds)",
            "DELTA 4 (arms and box)", "DELTA 5 (axis map)", "DELTA 6 (ports)",
            "DELTA 7 (frequency grid)", "DELTA 8 (two CalcPort passes)",
            "DELTA 9 (NrTS / EndCriteria)", "DELTA 10 (mesh rule)",
            "DELTA 11 (mesh rung)", "DELTA 12 (CSX length unit)",
            "DELTA 14 (mesh refinement around the metal)", "NOTHING ELSE"]):
        check(DELTA_LIST[i].startswith(key), f"entry {i+1} is {key!r}")
    check("2.413" in DELTA_LIST[1] and "20.320 x 2.540" in DELTA_LIST[1],
          "delta 2 names the feed width and the wide section")
    check("9.8565" in DELTA_LIST[2] and "16.4635" in DELTA_LIST[2],
          "delta 3 names both feed centres")
    check("27.472 x 26.320 x 3.794" in DELTA_LIST[3], "delta 4 names the box")
    check("2.5 mm" in DELTA_LIST[5] and "5.610 mm" in DELTA_LIST[5],
          "delta 6 names FeedShift and both MeasPlaneShifts")
    check("801" in DELTA_LIST[6], "delta 7 names the frequency grid")
    check("ref_impedance" in DELTA_LIST[7], "delta 8 names the two CalcPort passes")
    check("library defaults" in DELTA_LIST[8] and "NOT carried" in DELTA_LIST[8]
          and str(SCRIPT_NRTS_CAP) in DELTA_LIST[8],
          "delta 9 says the library defaults run and names the script cap it "
          "refuses")
    check("h_sub / 4" in DELTA_LIST[9] and "198.5" in DELTA_LIST[9],
          "delta 10 names the min(lambda/50, h_sub/4) rule and its value")
    check("1.0" in DELTA_LIST[10] and "0.70711" in DELTA_LIST[10]
          and "0.5" in DELTA_LIST[10],
          "delta 11 names all three rung factors")
    check("z lines" in DELTA_LIST[10] and "4, 6 and 8 cells" in DELTA_LIST[10],
          "delta 11 says the substrate z lines scale, and by how much")
    check("resolution/4" in DELTA_LIST[12] and "smooths ONCE per axis" in DELTA_LIST[12],
          "delta 14 names the tutorial's second smoothing and this builder's one")
    check("PATCH_X0 and PATCH_X1" in DELTA_LIST[12]
          and "feed_edges_realized" in DELTA_LIST[12],
          "delta 14 names the wide section's edges and where the realized widths "
          "are recorded")
    check("['PML_8','PML_8','MUR','MUR','PEC','MUR']" in DELTA_LIST[13]
          and "see delta 14" in DELTA_LIST[13],
          "the last entry names the boundary list that does NOT change, and hands "
          "the thirds-rule question to delta 14")

    print("the mesh rungs:")
    check(B_COARSE_RESOLUTION_FACTOR == 1.0
          and abs(B_MID_RESOLUTION_FACTOR - 2.0 ** -0.5) < 1e-12
          and B_FINE_RESOLUTION_FACTOR == 0.5,
          "the three rungs are 1.0, 1/sqrt(2) and 0.5",
          f"{B_COARSE_RESOLUTION_FACTOR:g}, {B_MID_RESOLUTION_FACTOR:.5f}, "
          f"{B_FINE_RESOLUTION_FACTOR:g}")
    check(0.0 < fine_factor < 1.0, "--resolution-factor is in (0, 1)", f"{fine_factor}")
    check((substrate_z_cells(1.0), substrate_z_cells(B_MID_RESOLUTION_FACTOR),
           substrate_z_cells(0.5)) == (4, 6, 8),
          "the rung factor gives 4, 6, 8 substrate cells",
          f"{substrate_z_cells(1.0)}, {substrate_z_cells(B_MID_RESOLUTION_FACTOR)}, "
          f"{substrate_z_cells(0.5)}")
    check(all(abs(H_SUB * 1e6 / substrate_z_cells(f) - want) < 1e-9
              for f, want in ((1.0, 198.5), (B_MID_RESOLUTION_FACTOR, 794.0 / 6),
                              (0.5, 99.25))),
          "the substrate cells are 198.5 / 132.33 / 99.25 um thick")

    print("the Stage B builder IS 07_sheen_lpf.py's run_openems builder:")
    try:
        proof = _copy_proof()
        for old, want in COPY_SUBSTITUTION_COUNTS.items():
            check(proof["copy_counts"][old] == want,
                  f"'{old}' appears {want}x in the script's builder block",
                  f"{proof['copy_counts'][old]}")
        check(proof["copy_matches"],
              "_build_sheen_board's build block IS the script's build block with "
              "exactly the three declared renames applied")
        if not proof["copy_matches"]:
            import difflib
            notes.append("\n".join(difflib.unified_diff(
                proof["derived"].splitlines(), proof["mine"].splitlines(),
                "derived-from-07_sheen_lpf.py", "_build_sheen_board-as-written",
                lineterm="")))
        for old, want in RUNG_SUBSTITUTION_COUNTS.items():
            check(proof["rung_counts"][old] == want,
                  f"the rung substitution target appears {want}x "
                  f"({old.strip()[:48]}...)",
                  f"{proof['rung_counts'][old]}")
        check(proof["rung_matches"],
              "_build_sheen_board_at_rung's build block IS _build_sheen_board's with "
              "exactly the two declared rung substitutions applied")
        # The excitation sits outside the build block (the script sets it in
        # _openems_common_setup), so it is derived on its own, by the same
        # renames. Without this, F_MAX/2 could become F_MAX/4 and everything
        # above would stay green.
        check(proof["excitation_matches"] and proof["excitation_rung_matches"],
              "both builders' SetGaussExcite statement IS the script's, with the "
              "declared renames applied",
              f"script {proof['excitation_script'].strip()!r} | "
              f"builder {proof['excitation_mine'].strip()!r} | "
              f"rung {proof['excitation_rung'].strip()!r}")
        # Delta 9 refuses the script's stop criteria, so the compared block must
        # NOT contain the call that sets them. Asserted, not left to the anchor.
        check(SCRIPT_SETUP_FUNCTION not in proof["script_slice"],
              f"the compared block EXCLUDES the script's {SCRIPT_SETUP_FUNCTION}() "
              f"call, where its NrTS/EndCriteria cap is set -- the proof compares "
              f"geometry, mesh and ports, not stop criteria")
        check(SCRIPT_SETUP_FUNCTION not in proof["mine"]
              and SCRIPT_SETUP_FUNCTION not in proof["rung"],
              "and neither builder's compared block mentions it either")
        # The cap is READ from the script, not remembered: the numbers are
        # parsed out of its own openEMS(...) call and compared as numbers, so a
        # different literal spelling of the same value still matches and a
        # different value does not.
        import re as _re
        setup = _script_function_source(SCRIPT_SETUP_FUNCTION)
        m_n = _re.search(r"NrTS\s*=\s*([0-9_eE.+-]+)", setup)
        m_e = _re.search(r"EndCriteria\s*=\s*([0-9_eE.+-]+)", setup)
        read_n = float(m_n.group(1)) if m_n else None
        read_e = float(m_e.group(1)) if m_e else None
        check(read_n == SCRIPT_NRTS_CAP and read_e == SCRIPT_END_CRITERIA_CAP,
              f"the retired script's {SCRIPT_SETUP_FUNCTION}() still sets the cap "
              f"delta 9 declines, and it is read from that file rather than "
              f"remembered",
              f"read NrTS={read_n!r} EndCriteria={read_e!r}; declining "
              f"{SCRIPT_NRTS_CAP} / {SCRIPT_END_CRITERIA_CAP}")
        check("fdtd = openEMS(**kw)" in proof["mine"] or "openEMS(**kw)" in
              _builder_body_after_kw(_build_sheen_board),
              "this maker creates the solver itself, from the nrts/end_criteria it "
              "is handed, so the stop criteria are the caller's choice and not the "
              "script's")
        if not proof["rung_matches"]:
            import difflib
            notes.append("\n".join(difflib.unified_diff(
                proof["derived_rung"].splitlines(), proof["rung"].splitlines(),
                "derived-from-_build_sheen_board", "_build_sheen_board_at_rung",
                lineterm="")))
    except Exception as exc:  # pragma: no cover - reported, not hidden
        check(False, "the copy proof runs", repr(exc))

    print("the builders' own source says where the board and the ports are:")
    try:
        import inspect
        body = _builder_body_after_kw(_build_sheen_board)
        rung_body = _builder_body_after_kw(_build_sheen_board_at_rung)
        check("mesh.AddLine('x', [0.0, LX, PATCH_X0, PATCH_X1])" in body,
              "the x lines 0, LX and both wide-section faces are EXPLICIT, so both "
              "port start planes and both steps are on a line by construction")
        check("sub.AddBox([0.0, 0.0, 0.0], [LX, LY, H_SUB])" in body,
              "the substrate spans the FULL domain footprint, so each y-face MUR "
              "sees a uniform dielectric cross-section")
        check("[PATCH_X1, PATCH_Y_HI, H_SUB]" in body
              and "[PATCH_X0, PATCH_Y_LO, H_SUB]" in body,
              "the wide section is a ZERO-thickness sheet: both z corners are H_SUB")
        check("FeedShift=PORT_MARGIN" in body and "MeasPlaneShift=0.45 * PATCH_X0" in body,
              "port 1's FeedShift and MeasPlaneShift are the script's own")
        check("MeasPlaneShift=0.45 * out_len" in body and "FeedShift" not in
              body.split("port[1] = fdtd.AddMSLPort(")[1].split("priority=10)")[0],
              "port 2 declares MeasPlaneShift and no FeedShift")
        check("excite=-1" in body, "port 1 is the excited port, excite=-1 verbatim")
        check("np.linspace(0, H_SUB, 5)" in body,
              "the base builder's substrate z recipe is linspace(0, H_SUB, 5): the "
              "substrate top is an explicit line and the board carries 4 cells")
        check("np.linspace(0, H_SUB, substrate_z_cells(resolution_factor) + 1)"
              in rung_body,
              "the rung builder's substrate z recipe scales with the rung")
        check("* resolution_factor" in rung_body,
              "the rung builder scales the resolution")
        check("resolution_factor" not in body,
              "the base builder does NOT take a rung -- it is the script's own mesh")
        del inspect
    except Exception as exc:  # pragma: no cover
        check(False, "the builders' source reads", repr(exc))

    print("the witness band, and the energy-sum bookkeeping:")
    check(WITNESS_BAND_HZ == (2.0e9, 12.0e9), "witness band is 2-12 GHz")
    check(tuple(WITNESS_BAND_GHZ) == (2.0, 12.0),
          "the GHz form is the same band")
    check(PASSIVITY_TOL == 0.05, "the passivity tolerance is the shared module's, 1.05")
    f_hz = np.linspace(F_LO, F_MAX, B_N_FREQS)
    # A planted spectrum: physical in the band, over unity below it, and in
    # DEFICIT above it -- the shape the committed record has (0.78-1.00 in band,
    # 0.41 at 20 GHz). The band witness must not see the below-band excess; the
    # full-grid number must report it; and the deficit must be recorded.
    s11_t = np.full(f_hz.shape, 0.30)
    s21_t = np.where(f_hz < WITNESS_BAND_HZ[0], 1.20,
                     np.where(f_hz > WITNESS_BAND_HZ[1], 0.50, 0.90))
    summary = _gate.energy_summary(f_hz, s11_t, s21_t,
                                   witness_band_hz=WITNESS_BAND_HZ,
                                   passivity_tol=PASSIVITY_TOL)
    check(abs(summary["max_energy_sum_band"] - (0.09 + 0.81)) < 1e-9,
          "the band witness reads only in-band bins",
          f"{summary['max_energy_sum_band']:.4f}")
    check(abs(summary["max_energy_sum_full"] - (0.09 + 1.44)) < 1e-9,
          "the full-grid number still reports the out-of-band excess",
          f"{summary['max_energy_sum_full']:.4f}")
    check(abs(summary["max_energy_sum_above_band"] - (0.09 + 0.25)) < 1e-9
          and abs(summary["min_energy_sum_full"] - (0.09 + 0.25)) < 1e-9,
          "the deficit above the band is recorded, as a maximum and as the "
          "whole-grid minimum",
          f"above band {summary['max_energy_sum_above_band']:.4f}, grid min "
          f"{summary['min_energy_sum_full']:.4f}")
    check(abs(summary["min_energy_sum_band"] - (0.09 + 0.81)) < 1e-9,
          "the band minimum is recorded too")
    check(summary["energy_sum"].size == B_N_FREQS,
          "the energy sum is recorded for every bin", f"{summary['energy_sum'].size}")
    check(_gate.failed_output_path(Path("/tmp/openems_sheen.json")).name
          == "openems_sheen_FAILED.json",
          "a failed gate's evidence file is named from the output stem")

    print("the reproduce gate still rejects a curve with no notch:")
    good = _gate.stage_a_gate_verdict(REPRODUCE_GATE_RECORD["reproduced_f_notch_hz"], -53.16)
    check(good["passed"],
          "the recorded tutorial reproduction (3.6711 GHz, -53.16 dB) PASSES",
          f"f_ok={good['f_notch_ok']} depth_ok={good['depth_ok']}")
    thru = _gate.stage_a_gate_verdict(F_NOTCH_AN_HZ, -0.02)
    check(not thru["passed"] and thru["f_notch_ok"] and not thru["depth_ok"],
          "a THRU line (flat |S21|, minimum -0.02 dB, inside the frequency band) "
          "is REJECTED on depth")
    off_band = _gate.stage_a_gate_verdict(1.5e9, -53.0)
    check(not off_band["passed"] and not off_band["f_notch_ok"],
          "a deep notch at 1.5 GHz is REJECTED on frequency")

    print("the reported features, on planted curves (none of them is gated):")
    try:
        sf = _gate.load_spectral_features()
        f_ghz = np.linspace(F_LO / 1e9, F_MAX / 1e9, B_N_FREQS)
        # A low-pass shape: flat 0.95 to 5 GHz, then a deep dip planted at 7.6 GHz.
        mag = np.where(f_ghz <= 5.0, 0.95, 0.95 / (1.0 + ((f_ghz - 5.0) / 0.8) ** 2))
        mag = mag * (1.0 - 0.999 * np.exp(-((f_ghz - 7.6) / 0.06) ** 2))
        feats = _stage_b_features(sf)(f_ghz, np.full_like(f_ghz, 0.2), mag)
        check(abs(feats["null"]["refined_f_ghz"] - 7.6) < 0.05,
              "the planted stopband minimum is found in 5-10 GHz",
              f"{feats['null']['refined_f_ghz']:.4f} GHz at "
              f"{feats['null']['depth_db']:.2f} dB")
        check(abs(feats["passband"]["mean_s21_linear"] - 0.95) < 0.01,
              "the passband mean over 1-4 GHz reads the planted 0.95",
              f"{feats['passband']['mean_s21_linear']:.4f} "
              f"({feats['passband']['mean_db']:.3f} dB)")
        check(feats["cutoff_3db"]["f_ghz"] is not None
              and 5.0 < feats["cutoff_3db"]["f_ghz"] < 6.5,
              "the -3 dB corner is found above 2 GHz on the planted roll-off",
              f"{feats['cutoff_3db']['f_ghz']:.4f} GHz at level "
              f"{feats['cutoff_3db']['level_linear']:.4f}")
        # A flat curve has no crossing at all: the estimator must say None
        # rather than invent a corner.
        flat = _stage_b_features(sf)(f_ghz, np.full_like(f_ghz, 0.2),
                                     np.full_like(f_ghz, 0.95))
        check(flat["cutoff_3db"]["f_ghz"] is None,
              "a FLAT |S21| yields no corner, not a fabricated one")
        check(abs(CUTOFF_LEVEL_DB + 3.0102999566398116) < 1e-9,
              "the corner level is 20*log10(1/sqrt(2)) dB below the passband mean",
              f"{CUTOFF_LEVEL_DB:.4f} dB")
    except Exception as exc:  # pragma: no cover - environment problem, reported
        check(False, "the shared estimators load by path", repr(exc))

    print("reported, not gated -- the port planes and the feed edges against the "
          "x PML_8 (estimate):")
    for name, plan in _stage_plans(fine_factor).items():
        print(f"  [--] {name}: PML depth {plan['pml_depth_mm_estimate']:.3f} mm, "
              f"FeedShift {plan['feed_shift_mm']:.3f} mm "
              f"(inside PML: {plan['feed_inside_pml_estimate']}), "
              f"MeasPlaneShift {plan['measplane_shift_mm'][0]:.3f} mm "
              f"(inside PML: {plan['measplane_inside_pml_estimate']}, "
              f"downstream of the feed: {plan['measplane_downstream_of_feed']})")

    print()
    print("WHAT THIS SELF-CHECK CANNOT DO: every mesh line between the explicitly "
          "added ones is placed by CSXCAD's SmoothMeshLines, and MSLPort snaps "
          "FeedShift and MeasPlaneShift to the nearest line itself. Both are CSXCAD "
          "calls. What is checked above is the BUILDERS' SOURCE -- that it IS the "
          "script's, character for character, and which lines it adds explicitly -- "
          "plus arithmetic on the declared numbers. It does NOT establish any "
          "interior line position, any exact line count, that the feed and "
          "measurement planes land on lines, or what the grid makes of the 2.413 mm "
          "feed width. The run reads the realized lines back from CSXCAD and records "
          "the ports' nearest lines and snaps and, per feed, the lines nearest each "
          "declared edge and the width between them; that is the only place those "
          "questions are answered.")
    for note in notes:
        print(note)
    print()
    print(f"SELF-CHECK {'PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# --merge: one record out of per-rung parts
# ---------------------------------------------------------------------------
_STAGE_A_ARRAYS = ("freqs_ghz", "s11_mag", "s11_deg", "s21_mag", "s21_deg",
                   "energy_sum")
_MERGE_META_MUST_AGREE = ("tutorial_source", "rfx_openems_image",
                          "rfx_openems_commit")


def _merge_refuse(msg: str) -> int:
    print(f"MERGE REFUSED: {msg}", file=sys.stderr)
    return 3


def _merge(part_paths: list, output: str) -> int:
    """Combine per-rung records into one, or refuse and say why.

    The parts were solved in separate cluster jobs, so nothing guarantees they
    are the same measurement -- a different image, a different openEMS build or a
    different tutorial port would each make the merged record a mixture. Stage A
    is the check that costs nothing: every job runs it, on the same tutorial, on
    the same mesh and grid, so if two parts disagree bin for bin on Stage A they
    did not come from the same solver and the merge refuses. Nothing is averaged
    or reconciled here; each rung block is carried across whole, from the one
    part that has it.
    """
    import copy

    if not part_paths:
        return _merge_refuse("no parts given")
    parts = []
    for raw in part_paths:
        path = Path(raw)
        if not path.is_file():
            return _merge_refuse(f"{path} is not a file")
        try:
            with open(path) as fh:
                parts.append((path, json.load(fh)))
        except Exception as exc:
            return _merge_refuse(f"{path} is not readable JSON: {exc!r}")

    # -- every part must carry a Stage A, and they must be the same one --------
    for path, a in parts:
        if not isinstance(a.get("meta"), dict):
            return _merge_refuse(f"{path} has no meta block")
        if a.get("failed_gate") is not None:
            return _merge_refuse(f"{path} is a FAILED-gate evidence file "
                                 f"({a['failed_gate']!r}), not a record")
        if not isinstance(a.get("stage_a"), dict):
            return _merge_refuse(f"{path} carries no stage_a: every part runs the "
                                 f"reproduce gate, so a part without it cannot be "
                                 f"shown to be the same measurement")
        missing = [k for k in _STAGE_A_ARRAYS if k not in a["stage_a"]]
        if missing:
            return _merge_refuse(f"{path}'s stage_a is missing {missing}")

    first_path, first = parts[0]
    for path, a in parts[1:]:
        for key in _STAGE_A_ARRAYS:
            if a["stage_a"][key] != first["stage_a"][key]:
                lhs, rhs = first["stage_a"][key], a["stage_a"][key]
                where = "length" if len(lhs) != len(rhs) else next(
                    (f"bin {i}" for i, (x, y) in enumerate(zip(lhs, rhs)) if x != y),
                    "?")
                return _merge_refuse(
                    f"{path}'s stage_a.{key} differs from {first_path}'s at "
                    f"{where} -- the two parts did not measure the same tutorial, "
                    f"so their rungs are not one record")
        for key in _MERGE_META_MUST_AGREE:
            if a["meta"].get(key) != first["meta"].get(key):
                return _merge_refuse(
                    f"{path}'s meta.{key} is {a['meta'].get(key)!r} but "
                    f"{first_path}'s is {first['meta'].get(key)!r}")

    # -- each rung comes from exactly one part --------------------------------
    owner: dict = {}
    for path, a in parts:
        for short in RUNG_ORDER:
            stage = RUNG_KEYS[short]
            if isinstance(a.get(stage), dict):
                if stage in owner:
                    return _merge_refuse(
                        f"{stage} is in both {owner[stage][0]} and {path}; a merge "
                        f"does not choose between two measurements of one rung")
                owner[stage] = (path, a)
    absent = [s for s in RUNG_ORDER_STAGES if s not in owner]
    if absent:
        return _merge_refuse(f"no part carries {absent}; the merged record would "
                             f"have fewer than the three rungs the mesh statement "
                             f"is made of")

    merged = copy.deepcopy(first)
    stages_meta = dict(merged["meta"].get("stages") or {})
    for stage, (path, a) in owner.items():
        merged[stage] = copy.deepcopy(a[stage])
        src_meta = (a["meta"].get("stages") or {}).get(stage)
        if src_meta is not None:
            stages_meta[stage] = copy.deepcopy(src_meta)
    merged["meta"]["stages"] = stages_meta
    merged["meta"]["rungs_in_this_record"] = list(RUNG_ORDER)
    merged["meta"]["stages_requested"] = ["stage_a"] + list(RUNG_ORDER_STAGES)
    merged["meta"]["stop_criteria_note"] = (
        "per part -- see meta.merged_from; the parts need not share one, and this "
        "record does not claim they do")
    maker_commits = [a["meta"].get("rfx_commit") for _, a in parts]
    merged["meta"]["maker_commits_agree"] = bool(len(set(map(str, maker_commits))) == 1)
    merged["meta"]["merged_from"] = [
        {
            "path": str(path),
            "run_id": a.get("run_id"),
            "maker_commit": a["meta"].get("rfx_commit"),
            "rungs": [s for s in RUNG_ORDER
                      if owner.get(RUNG_KEYS[s], (None, None))[0] == path],
            "stop_criteria_note": a["meta"].get("stop_criteria_note"),
        }
        for path, a in parts
    ]
    merged["run_id"] = None
    merged["run_id_note"] = (
        "null by design on a merged record: there is no single run. Each part's "
        "own run id is in meta.merged_from."
    )

    out = Path(output)
    _gate.write_record(merged, out)
    print(f"merged {len(parts)} part(s) into {out}")
    for entry in merged["meta"]["merged_from"]:
        print(f"  {entry['path']}  rungs={','.join(entry['rungs']) or '-'}  "
              f"run_id={entry['run_id']}  maker_commit={entry['maker_commit']}")
    if not merged["meta"]["maker_commits_agree"]:
        print("NOTE: the parts do not all name the same maker commit "
              f"({maker_commits!r}). Recorded in meta, not judged here.")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=["A", "B", "both"], default="both")
    p.add_argument("--output", default=None, help="where the JSON record is written")
    p.add_argument("--sim-root", default="/tmp/sheen_lpf_openems")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--resolution-factor", type=float, default=B_FINE_RESOLUTION_FACTOR,
                   help="Stage B only: the FINEST rung's factor on this board's own "
                        "resolution. The coarse rung is always 1.0 and the middle "
                        "rung 1/sqrt(2). Default 0.5. Which of the three a given "
                        "invocation solves is --rungs; each rung block records the "
                        "factor it was built with, so a part made with a "
                        "non-default one says so.")
    p.add_argument("--rungs", default=DEFAULT_RUNGS, metavar="LIST",
                   help="Stage B only: which mesh rungs this invocation solves, "
                        "comma separated, from coarse,mid,fine. Default all three. "
                        "One rung per cluster job is the normal way to run this: "
                        "with openEMS's own EndCriteria the coarse rung alone ran "
                        "107 minutes on 8 threads. The record then carries only the "
                        "rungs it solved, and meta.rungs_in_this_record lists them; "
                        "--merge puts the parts back together. Stage A runs in every "
                        "job.")
    p.add_argument("--real-end-criteria", type=float, default=None, metavar="FLOAT",
                   help="Stage B only: openEMS EndCriteria for the REAL pass. "
                        "Default: pass nothing, so openEMS's own 1e-5 applies -- "
                        "that is the declared design (delta 9). Giving a value is "
                        "recorded in every Stage B block and in "
                        "meta.stop_criteria_note. Stage A is never overridden.")
    p.add_argument("--real-nrts", type=int, default=None, metavar="INT",
                   help="Stage B only: openEMS NrTS for the REAL pass. Default: "
                        "pass nothing, so openEMS's own ~1e9 applies. Recorded the "
                        "same way as --real-end-criteria. Stage A is never "
                        "overridden.")
    p.add_argument("--accept-truncation", action="store_true",
                   help="Stage B only: record a real pass that reaches its NrTS "
                        "cap instead of failing the end-criteria gate. OFF by "
                        "default, and off is the declared design: a truncated "
                        "spectrum is normally not a reference. With it, the stage "
                        "records truncated: true and the box energy openEMS "
                        "reported at the cap, and meta.stop_criteria_note names "
                        "the flag. It exists so a record of DECLARED length can "
                        "be made on purpose -- what such a record is good for is "
                        "the leader's to decide from the data, and this script "
                        "decides nothing about it. Stage A is never affected.")
    p.add_argument("--merge", nargs="+", default=None, metavar="PART.json",
                   help="Combine per-rung records into one, written to --output. "
                        "Refuses unless every part's Stage A agrees bin for bin and "
                        "their tutorial source, image and openEMS build commit "
                        "match; refuses on a duplicated or a missing rung. Runs no "
                        "solver.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args(argv)

    try:
        rung_stages = parse_rungs(args.rungs)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    # The range check runs on EVERY mode, not only a real run: a dry run or a
    # self-check on an out-of-range factor would print a plan nothing can build.
    if args.resolution_factor <= 0.0 or args.resolution_factor >= 1.0:
        print(f"ERROR: --resolution-factor must be in (0, 1); got "
              f"{args.resolution_factor}", file=sys.stderr)
        return 3

    if args.self_check:
        return _self_check(args.resolution_factor)
    if args.dry_run:
        return _dry_run(args.stage, args.resolution_factor, rung_stages,
                        real_nrts=args.real_nrts,
                        real_end_criteria=args.real_end_criteria,
                        accept_truncation=args.accept_truncation)

    # --merge runs no solver and needs no image stamp: it reads records that
    # already carry their own.
    if args.merge:
        if not args.output:
            print("ERROR: --merge needs --output", file=sys.stderr)
            return 3
        return _merge(args.merge, args.output)

    if not args.output:
        print("ERROR: --output is required for a real run", file=sys.stderr)
        return 3
    # The image stamps the commit it built openEMS from. A record that cannot say
    # which solver build produced it is not a reference.
    if not os.environ.get("RFX_OPENEMS_COMMIT"):
        print("ERROR: RFX_OPENEMS_COMMIT is not set. The record must name the "
              "openEMS build it came from; the job file exports it from the image. "
              "Refusing to produce a reference with no solver provenance.",
              file=sys.stderr)
        return 3

    try:
        sf = _gate.load_spectral_features()
    except Exception as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 3
    try:
        _script_builder_slice()
    except Exception as exc:
        print(f"CONFIG ERROR: the copy proof cannot read the script this builder is "
              f"copied from: {exc}", file=sys.stderr)
        return 3

    try:
        _gate._import_openems()
    except Exception as exc:
        print(f"openEMS IS NOT IMPORTABLE: {exc!r}", file=sys.stderr)
        return 2

    print("=" * 78)
    print("The Sheen low-pass filter -- openEMS reference, with provenance")
    print("=" * 78)
    _print_delta_list()
    print()

    stages = _stages_for(args.stage, rung_stages)
    stage_names = ["stage_a"] + [s for s in RUNG_ORDER_STAGES if s in stages]
    print(f"stop criteria: {stop_criteria_note(args.real_end_criteria, args.real_nrts, args.accept_truncation)}")
    print(f"rungs this job solves: "
          f"{', '.join(rung_short_names(stages)) or '(none)'}", flush=True)
    print()
    if "stage_a" not in stages:
        print("WARNING: --stage B does not run the reproduce gate. The record this "
              "invocation writes carries stage_a: null and reproduce_gate_ran: false, "
              "and no number in it has been checked against openEMS's own tutorial "
              "result in this run. The job file submits --stage both.", flush=True)
    factors = dict(STAGE_B_FACTORS)
    factors["stage_b_fine"] = args.resolution_factor

    records: dict = {name: None for name in STAGE_NAMES}
    stage_meta: dict = {}
    stage_a_gate: dict = {}

    def artifact(failed_gate=None) -> dict:
        return _build_artifact(records, stage_meta, stage_a_gate, stages,
                               failed_gate=failed_gate,
                               real_nrts=args.real_nrts,
                               real_end_criteria=args.real_end_criteria,
                               accept_truncation=args.accept_truncation,
                               stage_names=stage_names)

    def bail(name: str, exc) -> int:
        print(f"SANITY GATE FAILED [{name}]: {exc}", file=sys.stderr)
        records[name] = exc.partial or None
        stage_meta[name] = exc.meta
        failed = artifact(failed_gate=str(exc))
        path = _gate.failed_output_path(Path(args.output))
        _gate.write_record(failed, path)
        print(f"evidence written to {path} -- the arrays measured before the gate "
              f"fired, the energy-sum numbers and failed_gate.", file=sys.stderr)
        return 1

    for name in stages:
        print(f"--- {name} ---", flush=True)
        try:
            if name == "stage_a":
                record, meta = _gate.run_stage_a(
                    sim_root=args.sim_root, threads=args.threads,
                    refined_extremum=sf.refined_extremum)
            else:
                record, meta = _run_stage_b(
                    label=name, sim_root=args.sim_root, threads=args.threads,
                    resolution_factor=factors[name], sf=sf,
                    real_nrts=args.real_nrts,
                    real_end_criteria=args.real_end_criteria,
                    accept_truncation=args.accept_truncation)
        except _gate.StageFailure as exc:
            return bail(name, exc)
        records[name] = record
        stage_meta[name] = meta

        if name == "stage_a":
            notch = record.get("notch") or {}
            if "refined_f_ghz" not in notch:
                print(f"REPRODUCE GATE CANNOT BE READ: the Stage A notch estimate is "
                      f"{notch!r}. No Stage B record is written.", file=sys.stderr)
                failed = artifact(
                    failed_gate="[stage_a] the notch estimator produced no "
                                f"frequency: {notch!r}")
                path = _gate.failed_output_path(Path(args.output))
                _gate.write_record(failed, path)
                print(f"evidence written to {path}", file=sys.stderr)
                return 1
            notch_hz = notch["refined_f_ghz"] * 1e9
            stage_a_gate = _gate.stage_a_gate_verdict(notch_hz, notch["depth_db"])
            record["gate"] = stage_a_gate
            print(f"  tutorial notch {notch['refined_f_ghz']:.4f} GHz "
                  f"(bin {notch['bin_f_ghz']:.4f} GHz, depth "
                  f"{notch['depth_db']:.2f} dB), {meta['wall_time_s']} s", flush=True)
            print(f"  reproduce gate: measured {notch_hz/1e9:.4f} GHz vs analytic "
                  f"{F_NOTCH_AN_HZ/1e9:.4f} GHz "
                  f"({stage_a_gate['deviation_pct']:.2f} %); band "
                  f"{STAGE_A_GATE['f_notch_lo_hz']/1e9:.4f}-"
                  f"{STAGE_A_GATE['f_notch_hi_hz']/1e9:.4f} GHz -> "
                  f"{'ok' if stage_a_gate['f_notch_ok'] else 'RED'}, depth <= "
                  f"{STAGE_A_MIN_DEPTH_DB:.0f} dB -> "
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
                               f"deep, not a notch at the "
                               f"{STAGE_A_MIN_DEPTH_DB:.0f} dB level")
                print("REPRODUCE GATE FAILED: no Stage B record is written.",
                      file=sys.stderr)
                failed = artifact(
                    failed_gate="[stage_a] reproduce gate FAILED: " + "; ".join(why))
                path = _gate.failed_output_path(Path(args.output))
                _gate.write_record(failed, path)
                print(f"evidence written to {path}", file=sys.stderr)
                return 1
            continue

        null = record.get("null") or {}
        pb = record.get("passband") or {}
        fc = (record.get("cutoff_3db") or {}).get("f_ghz")
        print(f"  deepest minimum in {NULL_BAND_GHZ[0]:.0f}-{NULL_BAND_GHZ[1]:.0f} GHz "
              f"{null.get('refined_f_ghz', float('nan')):.4f} GHz at "
              f"{null.get('depth_db', float('nan')):.2f} dB | passband mean "
              f"{pb.get('mean_db', float('nan')):+.3f} dB | -3 dB corner "
              f"{'n/a' if fc is None else f'{fc:.4f} GHz'} | "
              f"Re(Z0) median {record.get('re_z0_median_ohm', float('nan')):.2f} ohm "
              f"| {meta['wall_time_s']} s"
              + (f" | TRUNCATED at timestep {record.get('final_timestep')} with the "
                 f"box energy at {record.get('final_energy_db')} dB"
                 if record.get("truncated") else ""), flush=True)

    out = Path(args.output)
    _gate.write_record(artifact(), out)
    print(f"\n=== written to {out} ===")
    print("run_id is null by design: VESSL does not export the run id into the pod, "
          "so the submitter records it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
