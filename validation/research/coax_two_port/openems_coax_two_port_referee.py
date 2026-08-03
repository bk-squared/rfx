#!/usr/bin/env python3
"""External openEMS referee for rfx issue #489 stage 3 (coax two-port).

SCOPE FENCE (read before extending this file): this is a COMPARATOR-LEG
referee. It builds and runs an INDEPENDENT openEMS coaxial two-port model
and reports its own S-parameters; it does NOT run any rfx simulation, does
NOT import rfx, and makes NO pass/fail verdict about rfx's own numbers. It
brackets, it does not judge -- same posture as
``scripts/diagnostics/openems_thru_referee/thru_openems.py`` (issue #313)
and ``validation/research/floquet/rcwa_referee.py`` (issue #491). A human
reviewer compares this script's JSON output against the rfx numbers
recorded in ``docs/agent-memory/rfx-known-issues.md`` ("#489 stage 2
SHIPPED" entry, 2026-08-02) by hand.

Target (rfx side, NOT reproduced or re-derived by this script -- quoted
for the reviewer's convenience only):
    rfx PR #534 (657451b) ``Simulation.compute_coaxial_two_port``, measured
    4-12 GHz: |S21| 0.960 -> 0.737 (raw); compensated
    |S21|*exp(+Re(gamma_bar)*L12) = 0.979-0.992. The compensated gate is
    EXACTLY invariant to a reference-plane referral error (5-decimal
    cancellation measured at +30 cells) -- this referee is the only
    instrument in the #489 record that can see a referral error, because
    it recovers the port-to-port span with an independently-built solver
    and independently-placed feed planes instead of rfx's own
    matrix-pencil extrapolation.

DECLARED QUESTION (state up front, per this repo's comparator discipline):
    Does rfx's |S21| attenuation profile (0.96 -> 0.74, decay-rate-
    consistent per the compensated-gate check) and its reference-plane
    referral survive an INDEPENDENT solver? rfx's own compensated gate is
    structurally blind to referral errors (the exponential compensation
    factor is built from the SAME extrapolation whose correctness is in
    question); an external solver with its own independently-placed ports
    is not.

============================================================================
STAGE A -- REPRODUCE-GATE
============================================================================
Per ``docs/agent-memory/task_recipes/external_solver_comparator.md`` step
1-2: replicate the solver's own canonical tutorial for the structure class
VERBATIM and reproduce its documented known-good result before any
comparator use.

NO CANONICAL OPENEMS COAX TUTORIAL EXISTS. Verified 2026-08-03 via the
GitHub API against the two places openEMS ships worked examples:

    thliebig/openEMS python/Tutorials/ (10 files):
        Bent_Patch_Antenna.py, CRLH_Extraction.py, Dipole_SAR.py,
        Helical_Antenna.py, Horn_Antenna.py, MSL_NotchFilter.py,
        RCS_Sphere.py, Rect_Waveguide.py, Simple_Patch_Antenna.py,
        StripLine2MSL.py

    thliebig/openEMS matlab/examples/transmission_lines/ (6 files):
        CPW_Line.m, Finite_Stripline.m, MSL.m, MSL_Losses.m, Stripline.m,
        directional_coupler.m

Neither directory (nor any other example directory checked: antennas,
waveguide, other, __deprecated__) contains a coax/coaxial example. This
triggers the documented fallback in ``research/CLAUDE.md`` (comparator-
construction bullet): "If no canonical example exists for the structure
class: start from the nearest one, list every delta in the script header,
and satisfy the reproduce-gate via an analytic/independent oracle instead
(closed form, limiting case) -- the gate binds either way."

Nearest starting point: this repo's own
``scripts/diagnostics/build_coaxial_line_openems_broad_comparison.py``,
which is NOT an upstream openEMS tutorial but the only openEMS coax
CONSTRUCTION METHOD this repo has already worked out and documented through
real (if ultimately superseded-for-other-reasons) VESSL debugging: it fixed
three real, verified defects through actual failed runs --
    (1) MUR-on-dielectric instability on the z ends (energy blew up
        5e-16 -> 2.8e13) -- fix: PML_8 instead of MUR;
    (2) Z0 mismatch from an unsealed outer shell -- fix: PTFE fills the
        annulus out to b=SMA_OUTER_RADIUS, then a >=2-cell PEC tube OUTSIDE
        b seals the shield;
    (3) a "silent zero-energy" dropped excitation -- fix: pin every
        conductor radius and port z-plane to a fixed mesh line before
        ``SmoothMeshLines`` fills the interior (the coax analog of the
        AddEdges2Grid thirds-rule case in ``rfx-known-issues.md`` -- same
        general class, "a conductor/port edge must sit on a declared mesh
        line", different structure family: cylindrical, not planar).
This script inherits fixes (1)-(3) verbatim; see ``_build_coax_stub`` and
``_snap_mesh_lines`` below, both docstring-tagged with the fix they encode.

Analytic oracle (in place of a tutorial's documented number): the
closed-form lossless-TEM-line reflection coefficient. Two termination
cases, run on the SAME cross-section, mesh, and port construction Stage B
uses (so the reproduce-gate actually exercises the machinery Stage B
depends on, not a disconnected toy):
    short   -> |Gamma| = 1.0 EXACTLY (unitarity on a lossless network;
               no assumption about mesh, Z0 calibration, or discretization
               enters this one -- the tightest, most defensible gate
               available without a real tutorial to anchor to).
    matched -> |Gamma| ~= 0.0 in the ideal limit. Gated at < 0.10 here.
               This tolerance is an ENGINEERING JUDGMENT, not a number
               measured on this exact geometry (openEMS is not installed
               in the authoring environment -- see the worktree note in
               the accompanying VESSL YAML). It is informed by, but not
               copied from, the comparable envelopes this repo's other
               coax/openEMS calibration lanes measured on adjacent
               fixtures (0.02-0.08 typical match residual,
               ``scripts/diagnostics/build_coaxial_openems_calibration_
               fixture.py`` DEFAULT_CASES tight band; ``tests/
               test_coaxial_line_calibration.py`` rfx-side envelope
               0.02-0.08). If the first VESSL run lands the matched |S11|
               close to but above 0.10, the fix is mesh-line-snap /
               resolution (the case-8 general class), NOT loosening this
               number without a written root cause
               (rfx/CLAUDE.md "No silent gate loosening").
Both cases are SINGLE-PORT (only port 1's own uf_inc/uf_ref are read) --
deliberately, to keep Stage A free of the Stage-B-specific transmission-
channel question flagged below.

The REPRODUCE_GATE_RECORD dict below is the audit artifact
(``external_solver_comparator.md`` step 2: "record the reproduced number +
the run/log path that produced it"). It is committed in its UNRUN
placeholder state -- honesty over completeness: this script has never run
against a real openEMS build (not installed locally, VESSL-only). The
accompanying test (``tests/test_coax_two_port_referee_header.py``) asserts
the placeholder shape is present and self-consistent (UNRUN implies no
numbers, no log path) so the record cannot silently claim numbers without
a log path pointing at the run that produced them.

============================================================================
STAGE B -- TARGET GEOMETRY (rfx's through-line fixture, matched as closely
as openEMS's Cartesian mesh allows)
============================================================================
Geometry constants below were NOT eyeballed -- they were read directly off
the live rfx grid-construction code path by running, in this environment
(rfx IS locally importable; only openEMS is VESSL-only):

    from rfx.api import Simulation
    from rfx.sources.sources import GaussianPulse
    sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9, boundary='cpml')
    sim.add_coaxial_port((0.004, 0.004, 0.020), face='top', pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    grid = sim._build_grid()
    # -> grid.shape=(55,55,194), grid.dx=3.7474057249999997e-4,
    #    pad_x/y/z_lo=pad_x/y/z_hi=16, sim._cpml_layers=16
    # then the SAME z_hi_coax_top/z_feed_top/z_lo_coax_bot/z_feed_bot
    # arithmetic compute_coaxial_two_port() itself uses (rfx/api/_sparams.py)

This is the EXACT fixture ``tests/test_coax_two_port_fdtd.py::
test_matched_through_line_transmits_reciprocally`` (``slow_physics``) runs
and that PR #534's numbers came from.

DELTA LIST (every way this openEMS model differs from the rfx fixture --
declared per the comparator-construction discipline, not glossed over):

  1. BOUNDARY TOPOLOGY: rfx's ``compute_coaxial_two_port`` requires CPML on
     ALL SIX faces (verified: it raises if any boundary-spec face token is
     not "cpml" -- ``rfx/api/_sparams.py``, "requires CPML tokens on all
     six boundary faces"). This is DIFFERENT from this repo's existing
     1-port openEMS coax precedent (PEC on the 4 transverse faces, PML only
     on +-z, a shielded-box model). This script uses PML on all six faces
     (``PML_16`` matching rfx's cpml_layers=16), per the target fixture, not
     the older precedent's box topology.
  2. SHELL PLACEMENT: rfx's ``stamp_coaxial_line`` puts its one-cell PEC
     shell INSIDE the nominal outer radius b (shell spans
     ``[b - dz, b]``), thinning the PTFE annulus by one cell versus the
     textbook a-to-b span; this is a sub-cell rasterization choice made at
     the Yee materials-array level with no CSXCAD-primitive equivalent.
     This script instead uses the PROVEN openEMS recipe (fix 2 above):
     PTFE fills the full a-to-b annulus, and the PEC shield is a >=2-cell
     tube OUTSIDE b (``b`` to ``b + 2*dx``). Both realize "an outer
     conductor at approximately radius b"; they differ at the sub-cell
     scale in which side of b the metal sits. Z0 is essentially unaffected
     (the shell is a boundary condition, not a propagating medium).
  3. FEED / SOURCE MODEL (the single largest, most consequential delta):
     rfx drives each end with a distributed TEM plane-wave TFSF source
     spanning the FULL annular cross-section (``build_coaxial_tem_plane_
     source_specs``) behind a separate annular-resistor feed
     (``stamp_coaxial_annular_resistor``) -- an azimuthally-symmetric,
     Yee-native construction with no CSXCAD/openEMS equivalent. openEMS
     has no azimuthally-distributed coax launch primitive; this script
     uses ``AddLumpedPort`` -- a LOCALIZED radial probe/feed at ONE
     azimuthal angle (+x), the same primitive and orientation this repo's
     existing 1-port coax precedent already validated. A single-angle
     radial lumped port couples imperfectly to the pure TM0n (azimuthally
     uniform) TEM mode at launch and settles onto it a short distance down
     the line -- exactly the kind of feed-model mismatch
     ``external_solver_comparator.md`` step 5 says to declare rather than
     paper over. This is inherent to what openEMS's Python bindings offer,
     not a shortcut taken here.
  4. MESH: rfx auto-derives ``dx=dy=dz=3.7474057249999997e-4`` m from
     ``freq_max=40e9``. This script uses the SAME cell size on all three
     axes (apples-to-apples numerical dispersion at the two solvers'
     shared resolution) rather than an independently-chosen openEMS mesh.
  5. CPML DEPTH: rfx pads 16 cells BEYOND the declared ``domain=`` argument
     (additive: ``domain_z=60mm`` -> ``grid.shape[2]=194`` cells, ~34 more
     than the bare 60mm/dz~=160-cell interior). openEMS's ``PML_16``
     instead consumes 16 cells FROM WHATEVER MESH IS DECLARED (subtractive
     -- the same convention ``thru_openems.py`` and the 1-port coax
     precedent both use). This script declares a total mesh 2*16 cells
     LARGER than the rfx clear span on every padded axis so the CLEAR
     (non-PML) interior matches rfx's ``domain=`` argument exactly, with
     matching PML depth (16 cells) on top -- not a cell-for-cell replica of
     rfx's own padding bookkeeping, but the same physical clear-line length
     and the same absorber depth.
  6. PORT-TO-PORT SPAN: rfx's own two feed planes sit at z=1.1242217175mm
     and z=59.58375102749999mm in its own pad-relative frame (L12 =
     58.4595293mm exactly). This script places its two ports at the SAME
     physical separation (L12_MM below), inside a domain padded with
     openEMS's PML-eats-mesh convention (delta 5) instead of rfx's
     pad-beyond-domain convention.
  7. EXCITATION WAVEFORM: rfx uses a differentiated Gaussian pulse
     (``GaussianPulse``, FRACTIONAL bandwidth 1.2, f0=8 GHz -- see
     ``rfx/sources/sources.py`` docstring: bandwidth is a fraction of f0,
     NOT an absolute Hz value, issue #386). openEMS's ``SetGaussExcite``
     takes an absolute corner frequency; this script follows the SAME
     f0=midband / fc=0.85*f_stop pattern ``thru_openems.py`` already uses
     rather than attempting to bit-match rfx's specific pulse SHAPE (a
     different waveform family by construction; only the illuminated band
     needs to cover the comparison window).
  8. TRANSMISSION-CHANNEL CONVENTION (Stage B specific, flagged as an OPEN
     ITEM for the reviewer -- see ``_extract_two_port_s`` docstring below
     for the full derivation): ``AddLumpedPort`` measures LOCAL (V, I) at a
     single z-plane with no z-extent, so unlike ``MSLPort`` (used in
     ``thru_openems.py``, which has an explicit ``prop_dir``) it carries no
     built-in "which end of the line is this" orientation. The physically-
     derived convention this script uses is: for a wave launched at port 1
     (bottom) and travelling toward port 2 (top), the transmitted wave
     shows up in port 2's ``uf_inc`` channel (NOT ``uf_ref`` -- see the
     derivation), because "uf_inc" is a FIXED V+Z_ref*I projection that
     happens to coincide with the physically forward-travelling branch at
     BOTH ports here (neither port geometrically encodes a z-direction).
     This is REASONED, not empirically verified against a running openEMS
     -- flagged explicitly rather than silently assumed. The script records
     BOTH channels for both ports at both drives in the JSON output, and
     self-checks reciprocity (|S21| vs |S12|) as a genuine gate: a wrong
     channel choice would show up there.

MUST-MOVE-WHEN-VALIDATED CONDITION (mirrors the governance note in
``validation/research/floquet/rcwa_referee.py``): this script lives at
``validation/research/coax_two_port/`` and is deliberately OUTSIDE
``validation/crossval/`` and its ``manifest.json`` -- a script outside that
directory is exempt from crossval governance by construction (per
``.claude/rules/rfx-feature-discovery.md`` + ``feedback_crossval_
governance_glob_bypass.md``), so its presence here must NOT be read as a
registered crossval pass. It must be MOVED into ``validation/crossval/``
(and added to ``manifest.json``) only once: (a) it has actually run on
VESSL and the reproduce-gate in ``REPRODUCE_GATE_RECORD`` is filled with a
real number + log path, AND (b) issue #489 stage 3's own reviewer has
judged the resulting bracket against the rfx PR #534 numbers -- not before.

============================================================================
Hand-ported sanity checks (external scripts get no rfx preflight --
``external_solver_comparator.md`` step 3; each one is implemented, not just
listed):
============================================================================
  (a) nonzero excitation energy: max|uf_inc| finite and > 0 for the driven
      port at every drive (mirrors the D5 guard in
      ``build_coaxial_line_openems_broad_comparison.py``).
  (b) port time-domain trace nonzero (independent witness alongside (a),
      same precedent).
  (c) settling / ring-down witness: this repo's ring-down rule
      (rfx/CLAUDE.md) is energy-based; openEMS's own analogous instrument
      is its ``EndCriteria`` energy-decay stop. This script reports whether
      the run's actual timestep count (measured from the saved port-voltage
      trace length, since openEMS's Python bindings do not directly return
      the final timestep count) reached the ``NrTS`` cap (truncation
      suspected, EndCriteria never satisfied) or stopped early (decayed).
      A capped run is flagged, not silently reported as settled.
  (d) port/mesh clearance: both feed z-planes and (Stage A) the DUT plane
      are asserted, in code, to sit at least ``CPML_CELLS`` cells inside
      the declared PML depth -- not just described in prose.
  (e) mesh-line snap at every conductor radius (a, b, b+2dx) and at every
      port/DUT z-plane -- the coax analog of the AddEdges2Grid thirds-rule
      case (``rfx-known-issues.md`` comparator-bug case 8), implemented in
      ``_snap_mesh_lines`` / ``_build_stage_a_case`` / ``_build_stage_b_drive``.
  (f) non-physical field guard: |S| > 2.0 anywhere raises (mirrors the
      precedent's blown-up-field guard) rather than being silently
      reported as a physics result.
  (g) PRE-solve fail-fast gate (``_configs/.claude/rules/vessl-jobs.md``
      "Submission discipline"): a short smoke run (small NrTS, no decay
      criterion) BEFORE the real run, with openEMS's OS-level stdout
      captured and scanned for "Unused primitive" / off-mesh warning
      classes -- aborts before the full NrTS budget is spent on a doomed
      config, not just after (``_run_openems_capturing_stdout`` /
      ``_scan_stdout_for_bad_patterns``).

Exit codes (lane convention, ``docs/agent-memory/task_recipes/
vessl_external_referee_lane.md``): 0 = both stages' self-checks passed
(reproduce-gate AND Stage B sanity checks; this does NOT mean rfx and
openEMS numerically agree -- this referee brackets, it does not judge);
1 = a self-check failed (reproduce-gate OR a Stage B sanity/physical
guard); 2 = openEMS Python bindings not importable (declared skip, not a
failure of this script's own logic).

Usage (VESSL-only; openEMS is not expected to be importable outside the
lane in ``scripts/vessl_coax_two_port_referee.yaml``)::

    python validation/research/coax_two_port/openems_coax_two_port_referee.py \\
        --output .omx/coax_two_port_referee/openems_coax_two_port.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Stage A reproduce-gate record -- the audit artifact
# (external_solver_comparator.md step 2). Committed in its UNRUN placeholder
# state; a VESSL run fills these fields AND must supply a log path that
# actually exists on disk (see tests/test_coax_two_port_referee_header.py).
# ---------------------------------------------------------------------------
REPRODUCE_GATE_RECORD: dict = {
    "stage": "A",
    "no_canonical_tutorial_found": True,
    "tutorials_checked": {
        "thliebig/openEMS python/Tutorials": [
            "Bent_Patch_Antenna.py", "CRLH_Extraction.py", "Dipole_SAR.py",
            "Helical_Antenna.py", "Horn_Antenna.py", "MSL_NotchFilter.py",
            "RCS_Sphere.py", "Rect_Waveguide.py", "Simple_Patch_Antenna.py",
            "StripLine2MSL.py",
        ],
        "thliebig/openEMS matlab/examples/transmission_lines": [
            "CPW_Line.m", "Finite_Stripline.m", "MSL.m", "MSL_Losses.m",
            "Stripline.m", "directional_coupler.m",
        ],
    },
    "checked_on": "2026-08-03",
    "fallback": (
        "research/CLAUDE.md comparator-construction bullet: no canonical "
        "example -> nearest starting point + analytic oracle reproduce-gate"
    ),
    "nearest_starting_point": (
        "scripts/diagnostics/build_coaxial_line_openems_broad_comparison.py "
        "(this repo's own validated openEMS coax construction fixes: "
        "PML_8-not-MUR, PTFE-to-b + outer PEC tube, mesh-line snap)"
    ),
    "oracle": "closed-form lossless TEM-line reflection: |Gamma_short|=1.0 exactly, |Gamma_matched|~0.0",
    "gate": {"short_tol_abs": 0.02, "matched_tol_abs": 0.10},
    "status": "UNRUN",
    "reproduced_short_mag": None,
    "reproduced_matched_mag": None,
    "log_path": None,
    "vessl_run_id": None,
}

DECLARED_QUESTION = (
    "Does rfx's |S21| attenuation profile (0.96 -> 0.74 over 4-12 GHz, "
    "decay-rate-consistent per the compensated-gate check in PR #534) and "
    "its reference-plane referral survive an independent solver? rfx's own "
    "compensated gate is EXACTLY invariant to a reference-plane referral "
    "error (5-decimal cancellation at +30 cells, per docs/agent-memory/"
    "rfx-known-issues.md '#489 stage 2 SHIPPED' entry) -- this referee, "
    "with its own independently-built ports and reference planes, is not."
)

MUST_MOVE_WHEN_VALIDATED = (
    "Move into validation/crossval/ (+ manifest.json) only after: (a) a "
    "real VESSL run has filled REPRODUCE_GATE_RECORD with a number + log "
    "path, AND (b) issue #489 stage 3's reviewer has judged the resulting "
    "bracket against rfx PR #534's numbers. Not before -- this file's "
    "presence under validation/research/ must not be read as a registered "
    "crossval pass (validation/research/floquet/rcwa_referee.py precedent)."
)

# ---------------------------------------------------------------------------
# Geometry constants -- see module docstring "STAGE B" for how these were
# obtained (read directly off the live rfx grid-construction code path).
# ---------------------------------------------------------------------------
UNIT = 1.0e-3  # CSXCAD length unit: mm

A_MM = 0.635                       # SMA_PIN_RADIUS (rfx/sources/coaxial_port.py)
B_MM = 2.055                       # SMA_OUTER_RADIUS
PTFE_EPS_R = 2.1
DX_MM = 0.37474057249999997        # rfx's own Yee dz at freq_max=40 GHz
CPML_CELLS = 16                    # rfx's own resolved cpml_layers at this config
PML_DEPTH_MM = CPML_CELLS * DX_MM  # 5.99584916 mm

DOMAIN_CLEAR_X_MM = 8.0            # rfx domain=(0.008, 0.008, 0.060) transverse arg
DOMAIN_CLEAR_Y_MM = 8.0
DOMAIN_CLEAR_Z_MM = 60.0

Z_FEED_BOT_RFX_MM = 1.1242217174999998   # rfx z_feed_bot, pad-relative frame
Z_FEED_TOP_RFX_MM = 59.58375102749999    # rfx z_feed_top, pad-relative frame
L12_MM = Z_FEED_TOP_RFX_MM - Z_FEED_BOT_RFX_MM  # 58.4595293 mm, port-to-port

# Z0 = sqrt(L'/C') closed form, matches
# rfx.sources.coaxial_port.coaxial_tem_characteristic_impedance(A_MM*1e-3,
# B_MM*1e-3, PTFE_EPS_R) exactly (verified locally against the live rfx
# function; this script does not import rfx, so the value is reproduced
# here from the standard TEM formula Z0 = (eta0 / (2*pi*sqrt(eps_r))) *
# ln(b/a), eta0 = sqrt(mu0/eps0)).
_ETA0 = 376.73031346177066
Z0_OHM = (_ETA0 / (2.0 * np.pi * np.sqrt(PTFE_EPS_R))) * np.log(B_MM / A_MM)

# Frequency grid: 9 points 4-12 GHz (1 GHz step) -- overlaps the rfx BAND
# ([4,6,8,10,12] GHz, tests/test_coax_two_port_fdtd.py) at every other
# point for direct comparison, with finer resolution in between for a
# cleaner phase/group-delay curve (thru_openems.py precedent).
F_START_GHZ = 4.0
F_STOP_GHZ = 12.0
N_FREQS = 9
FREQS_GHZ = np.linspace(F_START_GHZ, F_STOP_GHZ, N_FREQS)
FREQS_HZ = FREQS_GHZ * 1e9

F0_GHZ = 0.5 * (F_START_GHZ + F_STOP_GHZ)  # 8.0 GHz -- matches rfx's f0=8e9
FC_GHZ = F_STOP_GHZ * 0.85                  # 10.2 GHz -- thru_openems.py pattern

# rfx PR #534 numbers, quoted ONLY for the reviewer's convenience (this
# script never compares against them programmatically).
RFX_REFERENCE_QUOTE = (
    "rfx PR #534 (657451b), measured 4-12 GHz: |S21| 0.960 -> 0.737 (raw); "
    "compensated |S21|*exp(+Re(gamma_bar)*L12) = 0.979-0.992. See "
    "docs/agent-memory/rfx-known-issues.md '#489 stage 2 SHIPPED' entry."
)


def _ensure_openems_numpy_compat() -> None:
    """openEMS Python bindings still expect deprecated NumPy aliases."""
    for name, value in {"float": float, "int": int, "complex": complex, "mat": np.matrix}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _import_openems():
    """Deferred openEMS import (kept OUT of module scope on purpose).

    Unlike ``scripts/diagnostics/openems_thru_referee/thru_openems.py``
    (which sys.exit(2)s at IMPORT time), this module must stay importable
    without openEMS installed so
    ``tests/test_coax_two_port_referee_header.py`` can load it locally and
    check the header/record fields (per the task's fail-loud-honest test
    design). The ImportError-to-exit-2 conversion happens in ``main()``
    instead.
    """
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS
    return ContinuousStructure, openEMS


# ---------------------------------------------------------------------------
# Shared geometry helpers (fix 2 + fix 3 from the module docstring, both
# ported verbatim from build_coaxial_line_openems_broad_comparison.py).
# ---------------------------------------------------------------------------
def _snap_mesh_lines(mesh, *, cx_mm: float, cy_mm: float, z_fixed_mm: list[float],
                     lx_mm: float, ly_mm: float, dx_mm: float) -> None:
    """Pin every conductor radius + port z-plane to a mesh line (fix 3).

    Without this, the SMA radii (a=0.635, b=2.055 mm) are not round
    multiples of dx_mm and a plain uniform grid leaves conductor/port edges
    BETWEEN lines -> openEMS silently drops the excitation ("Unused
    primitive", uf_inc=0, S11=NaN) -- the coax analog of the AddEdges2Grid
    thirds-rule case (rfx-known-issues.md comparator-bug case 8).
    """
    shell_outer_mm = B_MM + 2.0 * dx_mm
    fixed = {
        "x": sorted({0.0, lx_mm, cx_mm, cx_mm - A_MM, cx_mm + A_MM,
                    cx_mm - B_MM, cx_mm + B_MM,
                    cx_mm - shell_outer_mm, cx_mm + shell_outer_mm}),
        "y": sorted({0.0, ly_mm, cy_mm, cy_mm - A_MM, cy_mm + A_MM,
                    cy_mm - B_MM, cy_mm + B_MM,
                    cy_mm - shell_outer_mm, cy_mm + shell_outer_mm}),
        "z": sorted(set(z_fixed_mm)),
    }
    for axis in "xy":
        mesh.AddLine(axis, fixed[axis])
    mesh.AddLine("z", fixed["z"])
    mesh.SmoothMeshLines("all", dx_mm, 1.4)


def _build_coax_cross_section(csx, *, cx_mm: float, cy_mm: float,
                              z_lo_mm: float, z_hi_mm: float, dx_mm: float,
                              priority_base: int = 0):
    """PEC pin / PTFE annulus / outer PEC tube, spanning z_lo_mm..z_hi_mm.

    Fix 2 from the module docstring: PTFE fills the FULL a-to-b annulus
    (not thinned, unlike rfx's own Yee-level shell placement -- delta 2 in
    the header); the PEC shield is a tube OUTSIDE b, >=2 cells thick so the
    staircased shield is sealed (Z0 fix, verified working in
    build_coaxial_line_openems_broad_comparison.py).
    """
    shell_outer_mm = B_MM + 2.0 * dx_mm
    outer = csx.AddMetal(f"shell_{priority_base}")
    outer.AddCylinder([cx_mm, cy_mm, z_lo_mm], [cx_mm, cy_mm, z_hi_mm],
                      radius=shell_outer_mm, priority=priority_base + 1)
    ptfe = csx.AddMaterial(f"ptfe_{priority_base}", epsilon=PTFE_EPS_R)
    ptfe.AddCylinder([cx_mm, cy_mm, z_lo_mm], [cx_mm, cy_mm, z_hi_mm],
                     radius=B_MM, priority=priority_base + 5)
    pin = csx.AddMetal(f"pin_{priority_base}")
    pin.AddCylinder([cx_mm, cy_mm, z_lo_mm], [cx_mm, cy_mm, z_hi_mm],
                    radius=A_MM, priority=priority_base + 10)
    return outer, ptfe, pin


def _check_excitation_and_trace(port, sim_path: str, label: str) -> tuple[float, int]:
    """Hand-ported sanity checks (a) + (b): nonzero energy + nonzero trace.

    Returns (max|uf_inc|, n_trace_samples). Raises RuntimeError (self-check
    failure, exit 1) if either witness is exactly zero / non-finite --
    mirrors the D5 guard in build_coaxial_line_openems_broad_comparison.py.
    """
    uf_inc = np.asarray(port.uf_inc, dtype=np.complex128)
    inc_peak = float(np.max(np.abs(uf_inc))) if uf_inc.size else 0.0
    if not np.isfinite(inc_peak) or inc_peak == 0.0:
        raise RuntimeError(
            f"[{label}] openEMS port injected/received NO wave energy "
            f"(max|uf_inc|={inc_peak!r}): excitation did not couple or the "
            f"port never saw the wave. Check mesh-line snap at this port's "
            f"z-plane and conductor radii."
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
    return inc_peak, n_samples


def _non_physical_guard(s_mag: np.ndarray, label: str) -> None:
    """Hand-ported sanity check (f): a passive/lossless line has |S|<=1."""
    peak = float(np.max(s_mag)) if s_mag.size else float("nan")
    if not np.all(np.isfinite(s_mag)) or peak > 2.0:
        raise RuntimeError(
            f"[{label}] non-physical/unstable |S| max={peak!r}: field blew "
            f"up or diverged."
        )


# ---------------------------------------------------------------------------
# Pre-solve fail-fast gate (hand-ported sanity check (g);
# _configs/.claude/rules/vessl-jobs.md "Submission discipline": a solver
# that cannot be smoke-tested locally MUST carry an in-script pre-solve
# stdout scan that aborts BEFORE the full NrTS budget is spent -- a full
# build+run that returns all-NaN silently wastes a full VESSL cycle. This
# repo's own case 8 (rfx-known-issues.md, AddEdges2Grid) is exactly the
# defect class this scan targets: a conductor/port sitting off the mesh
# prints an "Unused primitive" warning during openEMS's geometry/mesh
# parse, well before timestepping starts.
# ---------------------------------------------------------------------------
_BAD_STDOUT_PATTERNS = ("Unused primitive", "not on the mesh", "unused excitation")


def _scan_stdout_for_bad_patterns(log_text: str, label: str) -> None:
    hits = [p for p in _BAD_STDOUT_PATTERNS if p.lower() in log_text.lower()]
    if hits:
        raise RuntimeError(
            f"[{label}] pre-solve mesh/port fail-fast gate tripped: openEMS "
            f"stdout contains {hits!r} -- a conductor or port sits off the "
            f"mesh (the case-8 AddEdges2Grid general class: 'a conductor/"
            f"port edge must sit on a declared mesh line'). Aborting BEFORE "
            f"the full NrTS budget is spent."
        )


def _run_openems_capturing_stdout(fdtd, sim_path: str, *, threads: int) -> str:
    """Run openEMS while capturing its OS-level stdout to a file we can grep.

    ``fdtd.Run()`` invokes the openEMS C++ binary; its stdout goes to the
    process's OS-level fd 1, not Python's ``sys.stdout``, so
    ``contextlib.redirect_stdout`` cannot see it -- redirect fd 1 itself,
    restoring it afterward even on error.
    """
    os.makedirs(sim_path, exist_ok=True)
    log_path = os.path.join(sim_path, "_openems_stdout.log")
    stdout_fd = sys.stdout.fileno()
    saved_fd = os.dup(stdout_fd)
    with open(log_path, "w") as logf:
        sys.stdout.flush()
        os.dup2(logf.fileno(), stdout_fd)
        try:
            fdtd.Run(sim_path, cleanup=True, verbose=1, numThreads=threads)
        finally:
            sys.stdout.flush()
            os.dup2(saved_fd, stdout_fd)
            os.close(saved_fd)
    with open(log_path) as logf:
        return logf.read()


# ---------------------------------------------------------------------------
# STAGE A: reproduce-gate (short + matched, single-port, on the SAME
# construction Stage B uses -- see module docstring).
# ---------------------------------------------------------------------------
def _build_stage_a_case(openEMS, ContinuousStructure, case: str, *,
                        nrts: int, end_criteria: float):
    """Build one Stage A case's CSX/fdtd/port fresh (called for BOTH the
    pre-solve smoke run and the real run -- see ``_run_openems_capturing_
    stdout``'s docstring for why these are two separate ``openEMS``
    instances rather than one object re-run with a different NrTS)."""
    dx_mm = DX_MM
    clear_z_mm = 30.0  # short stub -- only needs to be long enough for a
                        # clean feed-to-DUT run, not the full L12 span
    pml_mm = CPML_CELLS * dx_mm
    lz_mm = clear_z_mm + 2.0 * pml_mm
    lx_mm = ly_mm = DOMAIN_CLEAR_X_MM + 2.0 * pml_mm
    cx_mm = lx_mm / 2.0
    cy_mm = ly_mm / 2.0
    z_feed_mm = lz_mm - pml_mm - 2.0 * dx_mm  # 2 cells clear of the PML inner edge
    z_dut_mm = pml_mm + 2.0 * dx_mm

    fdtd = openEMS(NrTS=nrts, EndCriteria=end_criteria)
    fdtd.SetGaussExcite(F0_GHZ * 1e9, FC_GHZ * 1e9)
    fdtd.SetBoundaryCond(["PML_16"] * 6)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    z_fixed = [0.0, lz_mm, z_feed_mm, z_dut_mm, z_dut_mm + dx_mm]
    _snap_mesh_lines(mesh, cx_mm=cx_mm, cy_mm=cy_mm, z_fixed_mm=z_fixed,
                     lx_mm=lx_mm, ly_mm=ly_mm, dx_mm=dx_mm)
    # clearance check (d): port and DUT planes must sit well inside the
    # PML depth, not merely "somewhere in the domain".
    assert z_feed_mm < lz_mm - pml_mm, "Stage A feed plane inside PML"
    assert z_dut_mm > pml_mm, "Stage A DUT plane inside PML"

    _build_coax_cross_section(csx, cx_mm=cx_mm, cy_mm=cy_mm,
                              z_lo_mm=z_dut_mm, z_hi_mm=z_feed_mm + dx_mm,
                              dx_mm=dx_mm)

    if case == "short":
        cap = csx.AddMetal("short_cap")
        cap.AddCylinder([cx_mm, cy_mm, z_dut_mm], [cx_mm, cy_mm, z_dut_mm + dx_mm],
                        radius=B_MM + 2.0 * dx_mm, priority=20)

    feed = fdtd.AddLumpedPort(1, float(Z0_OHM), [cx_mm + A_MM, cy_mm, z_feed_mm],
                              [cx_mm + B_MM, cy_mm, z_feed_mm], "x", excite=1.0)
    if case == "matched":
        fdtd.AddLumpedPort(2, float(Z0_OHM), [cx_mm + A_MM, cy_mm, z_dut_mm],
                           [cx_mm + B_MM, cy_mm, z_dut_mm], "x", excite=0.0)
    return fdtd, feed


def _run_stage_a_reproduce_gate(*, sim_root: str, threads: int, nrts: int,
                                end_criteria: float) -> dict:
    _, openEMS = _import_openems()
    from CSXCAD.CSXCAD import ContinuousStructure

    results = {}
    for case in ("short", "matched"):
        sim_dir = os.path.join(sim_root, f"stage_a_{case}")

        # Pre-solve fail-fast gate: a short smoke run (small NrTS, no
        # energy-decay criterion) purely to trigger openEMS's own mesh/
        # port-parse warnings, scanned BEFORE the full NrTS budget runs.
        smoke_dir = os.path.join(sim_root, f"stage_a_{case}_smoke")
        smoke_fdtd, _smoke_feed = _build_stage_a_case(
            openEMS, ContinuousStructure, case, nrts=min(200, nrts), end_criteria=0.0)
        smoke_log = _run_openems_capturing_stdout(smoke_fdtd, smoke_dir, threads=threads)
        _scan_stdout_for_bad_patterns(smoke_log, f"stage_a_{case}_smoke")

        fdtd, feed = _build_stage_a_case(
            openEMS, ContinuousStructure, case, nrts=nrts, end_criteria=end_criteria)

        t0 = time.time()
        real_log = _run_openems_capturing_stdout(fdtd, sim_dir, threads=threads)
        _scan_stdout_for_bad_patterns(real_log, f"stage_a_{case}")  # belt-and-suspenders
        elapsed = time.time() - t0

        feed.CalcPort(sim_dir, FREQS_HZ)
        inc_peak, n_samples = _check_excitation_and_trace(feed, sim_dir, f"stage_a_{case}")
        truncated = bool(nrts > 0 and n_samples >= nrts)

        s11 = np.asarray(feed.uf_ref, dtype=np.complex128) / np.asarray(feed.uf_inc, dtype=np.complex128)
        s11_mag = np.abs(s11)
        _non_physical_guard(s11_mag, f"stage_a_{case}")

        tol = REPRODUCE_GATE_RECORD["gate"]["short_tol_abs"] if case == "short" else \
            REPRODUCE_GATE_RECORD["gate"]["matched_tol_abs"]
        target = 1.0 if case == "short" else 0.0
        dev = float(np.max(np.abs(s11_mag - target)))
        results[case] = {
            "s11_mag": s11_mag.tolist(),
            "mean_s11_mag": float(np.mean(s11_mag)),
            "max_dev_from_analytic": dev,
            "tol": tol,
            "passed": bool(dev < tol),
            "max_uf_inc": inc_peak,
            "n_trace_samples": n_samples,
            "nrts_cap": nrts,
            "truncated_suspected": truncated,
            "elapsed_s": round(elapsed, 1),
        }

    passed = bool(results["short"]["passed"] and results["matched"]["passed"])
    return {"cases": results, "passed": passed}


# ---------------------------------------------------------------------------
# STAGE B: target through-line geometry.
# ---------------------------------------------------------------------------
def _extract_two_port_s(port1, port2, freqs_hz: np.ndarray) -> dict:
    """Recover S11/S21 (or S22/S12) from one drive's CalcPort results.

    TRANSMISSION-CHANNEL DERIVATION (delta 8 in the module docstring --
    read that section first). ``AddLumpedPort`` measures purely LOCAL
    (V, I) at a single z-plane with no z-extent and no declared prop_dir,
    unlike ``MSLPort``. Standard transmission-line algebra:
    V/I = +Z0 for a wave travelling in whichever direction the port's
    current-sign convention calls "positive", V/I = -Z0 for the opposite
    direction. Both ports here share the SAME radial 'x' box ordering
    ([cx+a,...] -> [cx+b,...]), so that sign convention is IDENTICAL at
    both z-planes -- neither port geometrically encodes "which end of the
    line it is". A wave launched at the DRIVEN port necessarily satisfies
    V/I=+Z0 there BY CONSTRUCTION of the excitation (that is what
    "uf_inc" means for the port doing the exciting); since the convention
    is shared, the SAME wave arriving at the OTHER port also satisfies
    V/I=+Z0 there, i.e. it appears in that port's ``uf_inc`` channel too
    (not ``uf_ref``). This is REASONED from openEMS's documented Port
    base-class split formula (uf_inc=0.5*(uf_tot+if_tot*Z_ref),
    uf_ref=0.5*(uf_tot-if_tot*Z_ref)), not empirically verified against a
    running openEMS instance (not installed in the authoring environment).

    Returns both channels for the non-driven port so a reviewer can
    recover the alternate reading without rerunning.
    """
    uf_inc1 = np.asarray(port1.uf_inc, dtype=np.complex128)
    uf_ref1 = np.asarray(port1.uf_ref, dtype=np.complex128)
    uf_inc2 = np.asarray(port2.uf_inc, dtype=np.complex128)
    uf_ref2 = np.asarray(port2.uf_ref, dtype=np.complex128)

    s_self = uf_ref1 / uf_inc1
    s_thru_primary = uf_inc2 / uf_inc1   # primary convention -- see docstring
    s_thru_alternate = uf_ref2 / uf_inc1  # recorded for the reviewer, not used by the gate

    return {
        "s_self": s_self,
        "s_thru": s_thru_primary,
        "s_thru_alternate_channel": s_thru_alternate,
        "uf_inc1": uf_inc1, "uf_ref1": uf_ref1,
        "uf_inc2": uf_inc2, "uf_ref2": uf_ref2,
    }


def _build_stage_b_drive(openEMS, ContinuousStructure, drive: str, *,
                         nrts: int, end_criteria: float):
    """Build one Stage B drive's CSX/fdtd/ports fresh (smoke + real, same
    rationale as ``_build_stage_a_case``)."""
    dx_mm = DX_MM
    pml_mm = CPML_CELLS * dx_mm
    lz_mm = DOMAIN_CLEAR_Z_MM + 2.0 * pml_mm
    lx_mm = ly_mm = DOMAIN_CLEAR_X_MM + 2.0 * pml_mm
    cx_mm = lx_mm / 2.0
    cy_mm = ly_mm / 2.0
    z_port1_mm = pml_mm + Z_FEED_BOT_RFX_MM  # bottom port (drives +z when excited)
    z_port2_mm = pml_mm + Z_FEED_TOP_RFX_MM  # top port

    fdtd = openEMS(NrTS=nrts, EndCriteria=end_criteria)
    fdtd.SetGaussExcite(F0_GHZ * 1e9, FC_GHZ * 1e9)
    fdtd.SetBoundaryCond(["PML_16"] * 6)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    z_fixed = [0.0, lz_mm, z_port1_mm, z_port2_mm]
    _snap_mesh_lines(mesh, cx_mm=cx_mm, cy_mm=cy_mm, z_fixed_mm=z_fixed,
                     lx_mm=lx_mm, ly_mm=ly_mm, dx_mm=dx_mm)
    assert z_port1_mm > pml_mm + 2 * dx_mm, "port1 too close to PML"
    assert z_port2_mm < lz_mm - pml_mm - 2 * dx_mm, "port2 too close to PML"
    assert abs((z_port2_mm - z_port1_mm) - L12_MM) < 1e-6, "L12 mismatch vs rfx fixture"

    _build_coax_cross_section(csx, cx_mm=cx_mm, cy_mm=cy_mm,
                              z_lo_mm=z_port1_mm, z_hi_mm=z_port2_mm, dx_mm=dx_mm)

    excite1 = 1.0 if drive == "port1" else 0.0
    excite2 = 1.0 if drive == "port2" else 0.0
    port1 = fdtd.AddLumpedPort(1, float(Z0_OHM), [cx_mm + A_MM, cy_mm, z_port1_mm],
                               [cx_mm + B_MM, cy_mm, z_port1_mm], "x", excite=excite1)
    port2 = fdtd.AddLumpedPort(2, float(Z0_OHM), [cx_mm + A_MM, cy_mm, z_port2_mm],
                               [cx_mm + B_MM, cy_mm, z_port2_mm], "x", excite=excite2)
    return fdtd, port1, port2


def _run_one_drive(openEMS, ContinuousStructure, *, drive: str, sim_root: str,
                   threads: int, nrts: int, end_criteria: float) -> dict:
    sim_dir = os.path.join(sim_root, f"stage_b_drive_{drive}")

    # Pre-solve fail-fast gate (same rationale as Stage A): smoke run first.
    smoke_dir = os.path.join(sim_root, f"stage_b_drive_{drive}_smoke")
    smoke_fdtd, _sp1, _sp2 = _build_stage_b_drive(
        openEMS, ContinuousStructure, drive, nrts=min(200, nrts), end_criteria=0.0)
    smoke_log = _run_openems_capturing_stdout(smoke_fdtd, smoke_dir, threads=threads)
    _scan_stdout_for_bad_patterns(smoke_log, f"stage_b_drive_{drive}_smoke")

    fdtd, port1, port2 = _build_stage_b_drive(
        openEMS, ContinuousStructure, drive, nrts=nrts, end_criteria=end_criteria)

    t0 = time.time()
    real_log = _run_openems_capturing_stdout(fdtd, sim_dir, threads=threads)
    _scan_stdout_for_bad_patterns(real_log, f"stage_b_drive_{drive}")  # belt-and-suspenders
    elapsed = time.time() - t0

    port1.CalcPort(sim_dir, FREQS_HZ)
    port2.CalcPort(sim_dir, FREQS_HZ)

    driven_port = port1 if drive == "port1" else port2
    inc_peak, n_samples = _check_excitation_and_trace(driven_port, sim_dir, f"stage_b_drive_{drive}")
    truncated = bool(nrts > 0 and n_samples >= nrts)

    if drive == "port1":
        extracted = _extract_two_port_s(port1, port2, FREQS_HZ)
    else:
        extracted = _extract_two_port_s(port2, port1, FREQS_HZ)

    return {
        "extracted": extracted,
        "z0_port1": np.asarray(port1.Z_ref, dtype=np.complex128),
        "z0_port2": np.asarray(port2.Z_ref, dtype=np.complex128),
        "max_uf_inc": inc_peak,
        "n_trace_samples": n_samples,
        "nrts_cap": nrts,
        "truncated_suspected": truncated,
        "elapsed_s": round(elapsed, 1),
    }


def _group_delay(freqs_hz: np.ndarray, s21: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phase = np.unwrap(np.angle(s21))
    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64)
    gd_s = -np.gradient(phase, omega)
    return phase, gd_s


def _run_stage_b(*, sim_root: str, threads: int, nrts: int, end_criteria: float) -> dict:
    _, openEMS = _import_openems()
    from CSXCAD.CSXCAD import ContinuousStructure

    drive1 = _run_one_drive(openEMS, ContinuousStructure, drive="port1", sim_root=sim_root,
                            threads=threads, nrts=nrts, end_criteria=end_criteria)
    drive2 = _run_one_drive(openEMS, ContinuousStructure, drive="port2", sim_root=sim_root,
                            threads=threads, nrts=nrts, end_criteria=end_criteria)

    s11 = drive1["extracted"]["s_self"]
    s21 = drive1["extracted"]["s_thru"]
    s22 = drive2["extracted"]["s_self"]
    s12 = drive2["extracted"]["s_thru"]

    for label, arr in (("s11", s11), ("s21", s21), ("s12", s12), ("s22", s22)):
        _non_physical_guard(np.abs(arr), label)

    recip_mag_dev = float(np.max(np.abs(np.abs(s21) - np.abs(s12))))
    recip_phase_dev_deg = float(np.max(np.abs(np.degrees(np.angle(s21) - np.angle(s12)))))

    phase21, gd21_s = _group_delay(FREQS_HZ, s21)

    sanity_passed = bool(
        not drive1["truncated_suspected"] and not drive2["truncated_suspected"]
    )

    return {
        "freqs_ghz": FREQS_GHZ.tolist(),
        "s11": [[float(c.real), float(c.imag)] for c in s11],
        "s21": [[float(c.real), float(c.imag)] for c in s21],
        "s12": [[float(c.real), float(c.imag)] for c in s12],
        "s22": [[float(c.real), float(c.imag)] for c in s22],
        "s11_mag": np.abs(s11).tolist(), "s21_mag": np.abs(s21).tolist(),
        "s12_mag": np.abs(s12).tolist(), "s22_mag": np.abs(s22).tolist(),
        "s21_phase_rad_unwrapped": phase21.tolist(),
        "group_delay_ps": (gd21_s * 1e12).tolist(),
        "reciprocity_max_mag_dev": recip_mag_dev,
        "reciprocity_max_phase_dev_deg": recip_phase_dev_deg,
        "z0_port1_drive1": [[float(c.real), float(c.imag)] for c in drive1["z0_port1"]],
        "alternate_channel_s21": [[float(c.real), float(c.imag)]
                                  for c in drive1["extracted"]["s_thru_alternate_channel"]],
        "drive1_diagnostics": {k: v for k, v in drive1.items() if k != "extracted"},
        "drive2_diagnostics": {k: v for k, v in drive2.items() if k != "extracted"},
        "sanity_passed": sanity_passed,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=".omx/coax_two_port_referee/openems_coax_two_port.json")
    p.add_argument("--sim-root", default="/tmp/openems_coax_two_port_referee")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--nrts", type=int, default=200000)
    p.add_argument("--end-criteria", type=float, default=1e-4)
    p.add_argument("--skip-stage-b", action="store_true",
                   help="run only the Stage A reproduce-gate (fast smoke check)")
    args = p.parse_args(argv)

    try:
        _import_openems()
    except ImportError as exc:
        print(f"SKIP: openEMS Python bindings not importable ({exc}).\n"
              "  This script is VESSL-only (source-built openEMS); see "
              "scripts/vessl_coax_two_port_referee.yaml.", file=sys.stderr)
        return 2

    print("=" * 78)
    print("openEMS external referee -- rfx issue #489 stage 3 (coax two-port)")
    print("COMPARATOR LEG ONLY. See module docstring for the full scope fence,")
    print("delta list, and the Stage-B transmission-channel open item.")
    print("=" * 78)
    print(f"\nDeclared question:\n  {DECLARED_QUESTION}")
    print(f"\n{RFX_REFERENCE_QUOTE}")

    t_start = time.time()
    sim_root = os.path.abspath(args.sim_root)

    print("\n--- Stage A: reproduce-gate (short + matched, single-port) ---")
    stage_a = _run_stage_a_reproduce_gate(
        sim_root=sim_root, threads=args.threads, nrts=args.nrts,
        end_criteria=args.end_criteria,
    )
    for case, res in stage_a["cases"].items():
        print(f"  {case}: mean|S11|={res['mean_s11_mag']:.4f} "
              f"max_dev={res['max_dev_from_analytic']:.4f} (tol {res['tol']}) "
              f"passed={res['passed']} truncated={res['truncated_suspected']}")
    print(f"  Stage A passed: {stage_a['passed']}")

    stage_b = None
    if stage_a["passed"] and not args.skip_stage_b:
        print("\n--- Stage B: target through-line geometry ---")
        stage_b = _run_stage_b(
            sim_root=sim_root, threads=args.threads, nrts=args.nrts,
            end_criteria=args.end_criteria,
        )
        print(f"  |S21| band: {min(stage_b['s21_mag']):.4f} - {max(stage_b['s21_mag']):.4f}")
        print(f"  |S11| band: {min(stage_b['s11_mag']):.4f} - {max(stage_b['s11_mag']):.4f}")
        print(f"  reciprocity max mag dev: {stage_b['reciprocity_max_mag_dev']:.4f}, "
              f"max phase dev: {stage_b['reciprocity_max_phase_dev_deg']:.2f} deg")
        print(f"  sanity_passed: {stage_b['sanity_passed']}")
    elif not stage_a["passed"]:
        print("\nStage A FAILED its reproduce-gate -- skipping Stage B "
              "(external_solver_comparator.md: 'only then swap in the "
              "target geometry').")

    overall_passed = bool(
        stage_a["passed"] and (stage_b is None or stage_b["sanity_passed"])
    )

    elapsed_total = time.time() - t_start
    artifact = {
        "issue": 489,
        "stage": "3 (external openEMS referee)",
        "scope": "comparator leg only -- brackets, does not judge rfx's own numbers",
        "declared_question": DECLARED_QUESTION,
        "rfx_reference_quote": RFX_REFERENCE_QUOTE,
        "must_move_when_validated": MUST_MOVE_WHEN_VALIDATED,
        "reproduce_gate_record": REPRODUCE_GATE_RECORD,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "overall_passed": overall_passed,
        "elapsed_s": round(elapsed_total, 1),
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n=== Written to {out_path} ===")

    print("\n" + "=" * 78)
    print(f"overall_passed={overall_passed} (self-checks only -- NOT a "
          f"verdict on rfx vs openEMS agreement)")
    print("=" * 78)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
