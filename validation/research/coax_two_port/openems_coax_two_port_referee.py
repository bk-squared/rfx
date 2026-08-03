#!/usr/bin/env python3
"""External openEMS referee for rfx issue #489 stage 3 (coax two-port).

REBUILT (2026-08-03) after adversarial review of PR #540 -- REQUEST-CHANGES
at the premise level. The first version wrongly asserted no canonical
openEMS coax tutorial exists and no coax-native port primitive exists.
Both were false, verified independently before this rewrite:

  1. ``matlab/examples/waveguide/Coax.m`` (thliebig/openEMS) IS a canonical
     two-port coax through-line tutorial with a built-in analytic Z0 check
     -- the first version checked ``matlab/examples/transmission_lines/``
     and claimed ``waveguide/`` was checked too without actually opening
     it (it lists ``Coax.m``, ``Coax_CylinderCoords.m``,
     ``Coax_Cylindrical_MG.m``). Verified 2026-08-03: ``gh api
     repos/thliebig/openEMS/contents/matlab/examples/waveguide``.
  2. openEMS HAS a coax-native port class: ``CoaxialPort``
     (``python/openEMS/ports.py:731``) -- azimuthally-distributed
     thin-shell radial-E excitation, computes its own ``beta`` from
     measured fields, and ``CalcPort(ref_plane_shift=...)`` can referral
     the reported S-parameters to an arbitrary reference plane. The
     submodule pin this lane's YAML clones (``2000574e``) includes it
     (added 2026-05-13, fixed 2026-07-29).

SCOPE FENCE (unchanged): this is a COMPARATOR-LEG referee. It builds and
runs an INDEPENDENT openEMS model and reports its own S-parameters; it
does NOT run any rfx simulation, does NOT import rfx, and makes NO
pass/fail verdict about rfx's own numbers -- it brackets, it does not
judge (``scripts/diagnostics/openems_thru_referee/thru_openems.py``,
``validation/research/floquet/rcwa_referee.py`` precedent).

DO-NOT-REPEAT (R1/R2 class -- read before touching the port model): the
FIRST attempt at an openEMS coax comparator in this repo,
``scripts/diagnostics/build_coaxial_line_openems_broad_comparison.py``,
records in its own header: *"cluster openEMS v0.0.35 has no
AddCoaxialPort, and a radial AddLumpedPort cannot excite the coax TEM mode
(3 runs failed: mesh-misalignment, MUR-on-PTFE instability, then weak/zero
coupling with uf_inc~6e-14)."* That finding was true FOR THAT openEMS
version/pin; ``CoaxialPort`` was added 2026-05-13, after that script was
written. This script uses ``CoaxialPort`` throughout (Stage A AND Stage
B) specifically to avoid repeating that recorded, three-times-failed
mechanism with the same underlying primitive (``AddLumpedPort``) this
rebuild replaces.

Target (rfx side, NOT reproduced or re-derived by this script -- quoted
for the reviewer's convenience only):
    rfx PR #534 (657451b) ``Simulation.compute_coaxial_two_port``, measured
    4-12 GHz: |S21| 0.960 -> 0.737 (raw); compensated
    |S21|*exp(+Re(gamma_bar)*L12) = 0.979-0.992. See
    docs/agent-memory/rfx-known-issues.md '#489 stage 2 SHIPPED' entry.

DECLARED QUESTION:
    Does rfx's |S21| attenuation profile (0.96 -> 0.74 over 4-12 GHz,
    decay-rate-consistent per the compensated-gate check in PR #534) and
    its reference-plane referral survive an independent solver? rfx's own
    compensated gate is EXACTLY invariant to a reference-plane referral
    error (5-decimal cancellation at +30 cells, per docs/agent-memory/
    rfx-known-issues.md '#489 stage 2 SHIPPED' entry) -- this referee,
    with its own independently-built ports, its own independently-measured
    ``beta``, and ``CalcPort(ref_plane_shift=...)`` to place its reference
    planes exactly where rfx places its own, is not.

============================================================================
STAGE A -- REPRODUCE-GATE: a faithful Python-bindings port of Coax.m
============================================================================
Per ``docs/agent-memory/task_recipes/external_solver_comparator.md`` step
1-2: replicate the solver's own canonical tutorial for the structure class
VERBATIM and reproduce its documented known-good result before any
comparator use.

Coax.m (``matlab/examples/waveguide/Coax.m``, (C) 2010 Thorsten Liebig,
tested with openEMS v0.0.17) sets up:
  - air-filled (epsR=1) coax, length=1000mm, inner radius r_i=100mm,
    outer-conductor inner radius r_o=230mm, outer-conductor outer radius
    r_os=240mm, uniform mesh_res=[5,5,5]mm.
  - Boundary: PEC on the 4 transverse faces (a snug shielded box, mesh
    sized exactly to r_os), MUR on both z ends (``use_pml=0``, the
    tutorial's own default switch value; a PML_8 variant exists behind
    the SAME switch, exposed here via ``--use-pml``).
  - TWO ``AddCoaxialPort`` calls sharing ONE run: port 1 spans
    z=[0, length/2] with ``ExciteAmp=1`` and ``FeedShift=10*mesh_res(1)``
    (50mm); port 2 spans z=[length/2, length], passive (``ExciteAmp``
    defaults 0). Neither sets ``Feed_R`` -- both use the class default
    (``inf``, open) at their own ``start``, so NEITHER end of the device
    gets an explicit termination cap: the coax conductors simply run to
    z=0 and z=length, i.e. straight into the MUR boundary. THIS is the
    line-termination topology Stage B's rebuild imitates (H2 below) --
    Coax.m does NOT retract the conductors into an open stub short of the
    absorbing boundary.
  - FDTD: numTS=5000, EndCriteria=1e-5, ``SetGaussExcite(f0, f0)`` with
    f0=fc=0.5 GHz (both center AND corner set to the SAME 0.5 GHz -- an
    unusual-looking but literal reading of the tutorial, reproduced
    verbatim rather than "corrected").
  - Post-processing: ``freq = linspace(0, 2*f0, 201)``; ``s11 =
    port{1}.uf.ref / port{1}.uf.inc``; ``s21 = port{2}.uf.inc /
    port{1}.uf.inc`` -- CONFIRMS the transmission-channel convention this
    referee's Stage B also uses (port 2's ``uf_inc``, not ``uf_ref``,
    carries the transmitted wave for this port class) is the tutorial's
    own, not a guess. The frequency SWEEP (201 points, post-processing
    only, does not affect the FDTD run itself) is reproduced verbatim too
    -- DFT post-processing over already-recorded time-domain files is
    cheap regardless of point count, so there is no reason to reduce it.
  - The tutorial's OWN documented validation content is its "plot
    line-impedance comparison" figure: measured ``port{1}.ZL`` (real +
    imag) plotted against the closed-form
    ``ZL_a = Z0/(2*pi*sqrt(epsR))*log(r_o/r_i)`` it computes itself. This
    IS the "documented known-good result" the reproduce-gate targets --
    not an external number invented for this script. Python's
    ``CoaxialPort`` exposes the equivalent quantity as ``self.Z_ref``
    (set in ``ReadUIData`` from the SAME ``Et*dEt/(Ht*dHt)`` differential-
    line-probe formula the MATLAB class uses for ``ZL``).

The REPRODUCE_GATE_RECORD dict below is the audit artifact
(``external_solver_comparator.md`` step 2). It is committed in its UNRUN
placeholder state -- this script has never run against a real openEMS
build (not installed locally, VESSL-only).

============================================================================
STAGE B -- TARGET GEOMETRY (rfx's through-line fixture, matched as closely
as openEMS's Cartesian mesh + CoaxialPort allow)
============================================================================
Geometry constants below were read directly off the live rfx grid-
construction code path (rfx IS locally importable; only openEMS is
VESSL-only) -- and, per review fix B3, off the RASTERIZED interior cell
counts, not the nominal ``domain=`` argument:

    from rfx.api import Simulation
    from rfx.sources.sources import GaussianPulse
    sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9, boundary='cpml')
    sim.add_coaxial_port((0.004, 0.004, 0.020), face='top', pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    grid = sim._build_grid()
    # -> grid.shape=(55,55,194), grid.dx=3.7474057249999997e-4,
    #    pad_x/y/z_lo=pad_x/y/z_hi=16 (cpml_layers=16)
    # interior (rasterized) cells: x=y=55-32=23, z=194-32=162
    # -> RASTERIZED clear transverse span = 23*dz = 8.6190331675 mm
    #    (NOT the nominal domain= argument, 8.0 mm -- #325 class: the
    #    nominal size and the rasterized cell count disagree once padding
    #    and rounding are accounted for)
    # -> RASTERIZED clear z span = 162*dz = 60.707972745 mm (NOT 60.0 mm)
    # then the SAME z_hi_coax_top/z_feed_top/z_lo_coax_bot/z_feed_bot
    # arithmetic compute_coaxial_two_port() itself uses (rfx/api/_sparams.py)

This is the EXACT fixture ``tests/test_coax_two_port_fdtd.py::
test_matched_through_line_transmits_reciprocally`` (``slow_physics``) runs
and that PR #534's numbers came from.

PORT CLASS: ``CoaxialPort`` (not ``AddLumpedPort`` -- do-not-repeat above).
Each drive builds TWO ``CoaxialPort`` instances that together span the
rasterized clear line, split at an arbitrary shared midpoint
(``Z_SPLIT``) -- mirroring the mirror-symmetric "each port looks INTO the
line from its own end" convention rfx itself uses (port 1 = the "top"/+z
end, port 2 = the "bottom"/-z end), NOT Coax.m's own same-direction
cascaded-segments layout (Coax.m's two ports both point +z, because
they're sequential SEGMENTS of one propagating wave, not two opposing
feeds of a mirror-symmetric 2-port device the way rfx's fixture is).

LINE TERMINATION TOPOLOGY (H2 fix): the previous version retracted the
coax conductors 2 cells short of the CPML AND left that retraction as an
open, disconnected gap with a separate lumped port sitting in it -- an
open-stub artifact the tutorial's own topology does not have. This
version's conductors run to EXACTLY rfx's own z_lo_coax_bot / z_hi_coax_top
(the ACTUAL extent rfx's ``stamp_coaxial_line`` itself stamps, 2 cells
inside the CPML per rfx's own documented "PEC into CPML is
numerically-unstable, verified" rule in ``stamp_coaxial_line``'s
docstring) -- no further retraction, no gap, and (matching Coax.m's own
default ``Feed_R=inf``) no extra termination cap is added at either end:
each port's own natural "open end" behavior at its own ``start`` is all
that's there, exactly matching how Coax.m's ports work.

REFERENCE-PLANE REFERRAL (the capability this referee exists to exercise):
each port's own default field-probe geometry (mirroring Coax.m's
``FeedShift``/default ``MeasPlaneShift``) sits at a modest, well-
conditioned interior offset from that port's own ``start`` -- NOT
necessarily at rfx's own feed plane. ``CalcPort(ref_plane_shift=...)``
then does the referral: both ports need only ``ref_plane_shift =
DX_MM`` (ONE cell, computed programmatically as the distance from each
port's own ``start`` to rfx's own z_feed_bot / z_feed_top respectively --
see ``_stage_b_layout()``), because rfx's OWN z_feed_bot/z_feed_top sit
exactly 1 cell inside its own z_lo_coax_bot/z_hi_coax_top. This referral
distance is SMALL (1 cell) and well-conditioned precisely because each
port's own natural geometric end (``start``) already sits close to rfx's
target reference plane -- a large, poorly-conditioned referral was
deliberately avoided by construction, not by accident.

DELTA LIST (every remaining way this openEMS model differs from the rfx
fixture):
  1. BOUNDARY TOPOLOGY: rfx's ``compute_coaxial_two_port`` requires CPML
     on all six faces (``rfx/api/_sparams.py``, "requires CPML tokens on
     all six boundary faces"). Stage B uses ``PML_16`` (matching rfx's
     cpml_layers=16) on all six faces -- NOT Coax.m's own PEC-box+MUR
     topology, which is reserved for Stage A's literal tutorial
     reproduction only.
  2. SHELL PLACEMENT: rfx's ``stamp_coaxial_line`` puts its one-cell PEC
     shell INSIDE the nominal outer radius b (shell spans [b-dz, b]),
     thinning the PTFE annulus by one cell versus the textbook a-to-b
     span -- a sub-cell Yee-materials-array choice with no CSXCAD-
     primitive equivalent. ``CoaxialPort``'s own constructor instead
     fills the dielectric out to r_o and puts the outer conductor as a
     cylindrical SHELL from r_o to r_os (docstring: "r_o: Outer conductor
     inner radius", "r_os: Outer conductor outer radius") -- the
     textbook a-to-b convention, not rfx's thinned one. r_os is set to
     ``B_MM + 2*DX_MM`` (>=2 cells, sealed shield, matching the Z0 fix
     this repo's earlier coax precedent already validated).
  3. FEED / EXCITATION MODEL: rfx drives each end with a distributed TEM
     plane-wave TFSF source behind a separate annular-resistor feed
     (Yee-native, no CSXCAD equivalent). ``CoaxialPort``'s own excitation
     (a thin cylindrical shell with a 1/r-weighted radial-E profile,
     azimuthally uniform, NOT a TFSF one-way launch) is the closest
     coax-native primitive openEMS actually offers -- an azimuthally-
     distributed excitation, unlike the do-not-repeat's single-angle
     ``AddLumpedPort``, but still launches BIDIRECTIONALLY along the
     line's axis (toward BOTH ends) rather than rfx's directional TFSF
     split. The short backward-going lobe (toward each port's own nearby
     line-edge / CPML) is expected to be absorbed quickly and not
     contaminate the forward-going measurement, mirroring how Coax.m's
     OWN excitation (identical mechanism) works with its own nearby MUR
     boundary -- not verified against a running openEMS instance.
  4. MESH: rfx auto-derives dx=dy=dz=3.7474057249999997e-4 m from
     freq_max=40e9. Stage B uses the SAME cell size on all three axes.
  5. CPML DEPTH / DOMAIN CONVENTION: rfx pads 16 cells BEYOND its
     rasterized interior (additive). openEMS's ``PML_16`` instead
     consumes 16 cells FROM the declared mesh (subtractive, same
     convention ``thru_openems.py`` uses). Stage B declares a total mesh
     2*16 cells LARGER than the rasterized clear span on every padded
     axis so the CLEAR (non-PML) interior matches rfx's RASTERIZED
     interior exactly (B3 fix), with matching PML depth on top.
  6. PORT-TO-PORT SPAN: both ports' referred reference planes sit at
     rfx's own z=1.1242217175mm and z=59.58375102749999mm (pad-relative
     frame), L12=58.4595293mm exactly -- via ``ref_plane_shift``, not by
     physically building the ports at those exact locations (delta from
     v1, which physically built ports there; here the ports physically
     span the whole rasterized line and the referral does the rest).
  7. EXCITATION WAVEFORM: rfx uses a differentiated Gaussian pulse
     (fractional bandwidth 1.2, f0=8 GHz -- issue #386). Stage B's
     openEMS ``SetGaussExcite`` uses f0=8 GHz (matching rfx's own f0) and
     fc=0.85*12=10.2 GHz (the ``thru_openems.py`` f0-midband/fc-0.85x
     pattern) -- a different waveform family by construction; only the
     illuminated band needs to cover the comparison window.

Hand-ported sanity checks (external scripts get no rfx preflight --
``external_solver_comparator.md`` step 3):
  (a) nonzero excitation energy + (b) nonzero port time-domain trace:
      ``_check_excitation_and_trace``.
  (c) settling / truncation witness: actual timestep count (from the
      saved port-voltage trace length) vs the ``NrTS`` cap -- NOW GATED
      into ``passed`` for BOTH stages (M2 fix; previously recorded but
      not gated for Stage A).
  (d) port/mesh clearance: asserted in code, not just described in prose
      -- via ``_stage_b_layout()``, a PURE function testable without
      openEMS (see ``tests/test_coax_two_port_referee_header.py``).
  (e) mesh-line snap at every conductor radius, for rasterization
      QUALITY (``CoaxialPort`` does not have the ``AddLumpedPort``
      off-mesh SILENT-DROP failure mode -- its radii are just rasterized
      onto whatever transverse mesh exists -- so this is a resolution
      concern now, not a silent-failure-avoidance one).
  (f) non-physical field guard: |S|>2.0 anywhere raises.
  (g) PRE-solve fail-fast gate (``_configs/.claude/rules/vessl-jobs.md``
      "Submission discipline"): a short smoke run before every real run,
      openEMS's OS-level stdout captured and scanned for mesh/port-parse
      warning classes.
  (h) B5 witnesses, WIRED INTO ``passed`` (not just recorded):
      - per-bin passivity: |S11|^2+|S21|^2 <= 1+tol (hard upper bound for
        any passive structure; a lower bound is NOT gated since the true
        loss is not known a priori -- only recorded).
      - Stage-A matched-through expectation: |S21|~=1 with phase~=-beta*L
        and group delay~=L/c (Coax.m's line is air-filled, lossless,
        non-dispersive TEM) over a central frequency sub-band, away from
        f=0 and the sweep's own edges. ONE-SIDED per the review: this
        compares against an INDEPENDENTLY-known expected value (~1), not
        against another internally-derived quantity like S12 -- a
        systematic channel-convention error (e.g. reading ``uf_ref``
        instead of ``uf_inc`` for port 2) would read close to 0 here,
        which a symmetric reciprocity check could fail to catch if the
        SAME error affected both drives identically.
      - Stage A's truncation witness (c) is now included in ``passed``.

Exit codes (lane convention, ``docs/agent-memory/task_recipes/
vessl_external_referee_lane.md``, extended per review): 0 = both stages'
self-checks passed (does NOT mean rfx and openEMS numerically agree --
this referee brackets, it does not judge); 1 = a PHYSICS/self-check gate
failed (reproduce-gate, a Stage B sanity/physical guard, or a B5
witness) -- ``RuntimeError``; 2 = openEMS Python bindings not importable
(declared skip); 3 = an INTERNAL CONFIGURATION error in this script's own
geometry/layout math (``AssertionError``, e.g. a clearance computation
that doesn't hold) -- distinct from 1 so a reviewer can immediately tell
"the physics disagreed" from "this script has a bug", per review fix M2.

MUST-MOVE-WHEN-VALIDATED CONDITION (mirrors ``validation/research/
floquet/rcwa_referee.py``'s governance note): this script lives at
``validation/research/coax_two_port/`` and is deliberately OUTSIDE
``validation/crossval/`` and its ``manifest.json`` -- exempt from
crossval governance by construction (``.claude/rules/
rfx-feature-discovery.md`` + ``feedback_crossval_governance_glob_
bypass.md``). Move into ``validation/crossval/`` (+ ``manifest.json``)
only after: (a) a real VESSL run has filled ``REPRODUCE_GATE_RECORD``
with numbers + a log path, AND (b) issue #489 stage 3's reviewer has
judged the resulting bracket against rfx PR #534's numbers.

Usage (VESSL-only; openEMS is not expected to be importable outside the
lane in ``scripts/vessl_coax_two_port_referee.yaml``, and that lane runs
the PRIMARY checkout -- this script must be merged to main before
submitting)::

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
# actually exists on disk under .omx/ or docs/research_notes/vessl_logs/
# (M3 fix -- see tests/test_coax_two_port_referee_header.py).
# ---------------------------------------------------------------------------
_ETA0 = 376.73031346177066  # sqrt(mu0/eps0), vacuum wave impedance (ohm)
ZL_A_ANALYTIC_OHM = (_ETA0 / (2.0 * np.pi)) * np.log(230.0 / 100.0)  # Coax.m's own ZL_a

REPRODUCE_GATE_RECORD: dict = {
    "stage": "A",
    "tutorial": {
        "repo": "thliebig/openEMS",
        "path": "matlab/examples/waveguide/Coax.m",
        "verified_present_on": "2026-08-03",
        "verified_via": "gh api repos/thliebig/openEMS/contents/matlab/examples/waveguide",
        "submodule_pin_note": (
            "openEMS-Project submodule pin 2000574e (scripts/"
            "vessl_coax_two_port_referee.yaml's clone) includes CoaxialPort "
            "(added 2026-05-13, fixed 2026-07-29)."
        ),
    },
    "do_not_repeat": (
        "scripts/diagnostics/build_coaxial_line_openems_broad_comparison.py "
        "header: 'a radial AddLumpedPort cannot excite the coax TEM mode "
        "(3 runs failed: mesh-misalignment, MUR-on-PTFE instability, then "
        "weak/zero coupling with uf_inc~6e-14)' -- this script uses "
        "CoaxialPort (python/openEMS/ports.py:731) instead, throughout "
        "both stages, precisely to avoid repeating that recorded failure."
    ),
    "geometry": (
        "two-port coax through-line, air-filled (epsR=1), length=1000mm, "
        "r_i=100mm, r_o=230mm, r_os=240mm, mesh_res=5mm uniform, PEC "
        "transverse + MUR z (matches the tutorial's own use_pml=0 default)"
    ),
    "documented_check": (
        "Coax.m's own 'plot line-impedance comparison' figure: measured "
        "port{1}.ZL (Python: CoaxialPort.Z_ref, real+imag) vs the "
        "closed-form ZL_a = Z0/(2*pi*sqrt(epsR))*log(r_o/r_i) the "
        "tutorial itself computes and plots alongside -- the tutorial's "
        "own documented validation content, not an external number "
        "invented for this script."
    ),
    "expected_zl_a_ohm": float(ZL_A_ANALYTIC_OHM),
    "gate": {"zl_re_tol_ohm": 5.0, "zl_im_tol_ohm": 5.0, "s21_thru_band": [0.5, 1.3]},
    "status": "UNRUN",
    "reproduced_zl_mean_ohm": None,
    "reproduced_zl_max_dev_ohm": None,
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
    "with its own independently-built ports, independently-measured beta, "
    "and CalcPort(ref_plane_shift=...) referral, is not."
)

MUST_MOVE_WHEN_VALIDATED = (
    "Move into validation/crossval/ (+ manifest.json) only after: (a) a "
    "real VESSL run has filled REPRODUCE_GATE_RECORD with a number + log "
    "path, AND (b) issue #489 stage 3's reviewer has judged the resulting "
    "bracket against rfx PR #534's numbers. Not before -- this file's "
    "presence under validation/research/ must not be read as a registered "
    "crossval pass (validation/research/floquet/rcwa_referee.py precedent)."
)

RFX_REFERENCE_QUOTE = (
    "rfx PR #534 (657451b), measured 4-12 GHz: |S21| 0.960 -> 0.737 (raw); "
    "compensated |S21|*exp(+Re(gamma_bar)*L12) = 0.979-0.992. See "
    "docs/agent-memory/rfx-known-issues.md '#489 stage 2 SHIPPED' entry."
)

# ---------------------------------------------------------------------------
# Stage A geometry constants -- Coax.m, verbatim.
# ---------------------------------------------------------------------------
UNIT = 1.0e-3  # CSXCAD length unit: mm

A_NUM_TS = 5000
A_END_CRITERIA = 1.0e-5
A_LENGTH_MM = 1000.0
A_R_I_MM = 100.0
A_R_O_MM = 230.0
A_R_OS_MM = 240.0
A_MESH_RES_MM = 5.0
A_F0_HZ = 0.5e9
A_FC_HZ = 0.5e9  # Coax.m: SetGaussExcite(FDTD, f0, f0) -- verbatim, not a typo
A_N_FREQS = 201
A_FREQS_HZ = np.linspace(0.0, 2.0 * A_F0_HZ, A_N_FREQS)
A_FEEDSHIFT_MM = 10.0 * A_MESH_RES_MM  # Coax.m: 'FeedShift', 10*mesh_res(1)
A_MEASPLANE_1_MM = 0.25 * A_LENGTH_MM  # default midpoint of port1's [0, length/2] span
A_MEASPLANE_2_MM = 0.75 * A_LENGTH_MM  # default midpoint of port2's [length/2, length] span
A_L_THRU_MM = A_MEASPLANE_2_MM - A_MEASPLANE_1_MM  # 500mm, for the phase/group-delay witness
_C0 = 2.998e8  # m/s

# ---------------------------------------------------------------------------
# Stage B geometry constants -- read off the live rfx grid-construction
# code path (see module docstring "STAGE B" for the exact snippet run).
# ---------------------------------------------------------------------------
B_A_MM = 0.635                       # SMA_PIN_RADIUS (rfx/sources/coaxial_port.py)
B_B_MM = 2.055                       # SMA_OUTER_RADIUS
B_PTFE_EPS_R = 2.1
B_DX_MM = 0.37474057249999997         # rfx's own Yee dz at freq_max=40 GHz
B_CPML_CELLS = 16                     # rfx's own resolved cpml_layers at this config
B_PML_DEPTH_MM = B_CPML_CELLS * B_DX_MM  # 5.99584916 mm

# RASTERIZED interior cell counts (B3 fix -- NOT the nominal domain= arg):
# grid.shape=(55,55,194), pad=16 all round -> interior x=y=23, z=162.
B_INTERIOR_X_CELLS = 23
B_INTERIOR_Y_CELLS = 23
B_INTERIOR_Z_CELLS = 162
B_CLEAR_X_MM = B_INTERIOR_X_CELLS * B_DX_MM   # 8.6190331675 mm (NOT 8.0)
B_CLEAR_Y_MM = B_INTERIOR_Y_CELLS * B_DX_MM
B_CLEAR_Z_MM = B_INTERIOR_Z_CELLS * B_DX_MM   # 60.707972745 mm (NOT 60.0)

# rfx's own pad-relative z positions (from the live grid-construction code
# path; z_lo_coax_bot/z_hi_coax_top are the ACTUAL stamped-line extent,
# z_feed_bot/z_feed_top the feed/reference planes 1 cell further in).
B_Z_LO_COAX_BOT_REL_MM = 0.749481145
B_Z_HI_COAX_TOP_REL_MM = 59.958491599999995
B_Z_FEED_BOT_REL_MM = 1.1242217174999998
B_Z_FEED_TOP_REL_MM = 59.58375102749999
B_L12_MM = B_Z_FEED_TOP_REL_MM - B_Z_FEED_BOT_REL_MM  # 58.4595293 mm, port-to-port

# Both ports' referral distance from their own `start` to rfx's own feed
# plane is exactly 1 cell (verified: both differences below equal
# B_DX_MM to float precision) -- see module docstring "REFERENCE-PLANE
# REFERRAL".
B_REF_PLANE_SHIFT_MM = B_DX_MM

# Z0 = sqrt(L'/C') closed form for the rfx PTFE-filled cross-section,
# matches rfx.sources.coaxial_port.coaxial_tem_characteristic_impedance
# to 8 significant digits (verified locally against the live rfx
# function; this script does not import rfx, so the value is reproduced
# here from the standard TEM formula Z0 = (eta0/(2*pi*sqrt(eps_r))) *
# ln(b/a)).
B_Z0_OHM = (_ETA0 / (2.0 * np.pi * np.sqrt(B_PTFE_EPS_R))) * np.log(B_B_MM / B_A_MM)

# Frequency grid: 9 points 4-12 GHz (1 GHz step) -- overlaps the rfx BAND
# ([4,6,8,10,12] GHz, tests/test_coax_two_port_fdtd.py) at every other
# point, with finer resolution in between for phase/group-delay.
B_F_START_GHZ = 4.0
B_F_STOP_GHZ = 12.0
B_N_FREQS = 9
B_FREQS_GHZ = np.linspace(B_F_START_GHZ, B_F_STOP_GHZ, B_N_FREQS)
B_FREQS_HZ = B_FREQS_GHZ * 1e9
B_F0_GHZ = 0.5 * (B_F_START_GHZ + B_F_STOP_GHZ)  # 8.0 GHz -- matches rfx's f0=8e9
B_FC_GHZ = B_F_STOP_GHZ * 0.85                    # 10.2 GHz -- thru_openems.py pattern

# MeasPlaneShift / FeedShift for Stage B's own ports, in cells from each
# port's own `start` -- mirrors the SPATIAL ORDER rfx itself uses (its
# reference/feed plane sits closer to the line's own edge than its
# source): a small, well-conditioned measurement offset (3 cells) with
# the excitation further in (8 cells), clearly separated from it.
B_MEASPLANE_SHIFT_CELLS = 3
B_FEEDSHIFT_CELLS = 8


def _stage_b_layout() -> dict:
    """Pure arithmetic: every Stage B z-position/clearance, in mm.

    Deliberately openEMS-FREE so it is testable locally without the
    solver installed (review fix: distinguish a config/layout bug,
    AssertionError -> exit 3, from a physics/self-check failure,
    RuntimeError -> exit 1 -- this function is where the former would
    originate, and its assertions are checked by
    tests/test_coax_two_port_referee_header.py without needing openEMS).
    """
    pml_mm = B_PML_DEPTH_MM
    lx_mm = ly_mm = B_CLEAR_X_MM + 2.0 * pml_mm
    lz_mm = B_CLEAR_Z_MM + 2.0 * pml_mm
    cx_mm = lx_mm / 2.0
    cy_mm = ly_mm / 2.0
    z_lo_coax_bot_mm = pml_mm + B_Z_LO_COAX_BOT_REL_MM
    z_hi_coax_top_mm = pml_mm + B_Z_HI_COAX_TOP_REL_MM
    z_feed_bot_mm = pml_mm + B_Z_FEED_BOT_REL_MM
    z_feed_top_mm = pml_mm + B_Z_FEED_TOP_REL_MM
    z_split_mm = 0.5 * (z_lo_coax_bot_mm + z_hi_coax_top_mm)

    layout = {
        "lx_mm": lx_mm, "ly_mm": ly_mm, "lz_mm": lz_mm,
        "cx_mm": cx_mm, "cy_mm": cy_mm,
        "z_lo_coax_bot_mm": z_lo_coax_bot_mm, "z_hi_coax_top_mm": z_hi_coax_top_mm,
        "z_feed_bot_mm": z_feed_bot_mm, "z_feed_top_mm": z_feed_top_mm,
        "z_split_mm": z_split_mm,
        "ref_plane_shift_port1_mm": z_feed_bot_mm - z_lo_coax_bot_mm,
        "ref_plane_shift_port2_mm": z_hi_coax_top_mm - z_feed_top_mm,
    }

    # Clearance / config assertions -- an AssertionError here is a BUG in
    # this script's own geometry math (exit 3), not a physics finding.
    assert z_lo_coax_bot_mm > pml_mm, "Stage B port1 line-edge inside PML"
    assert z_hi_coax_top_mm < lz_mm - pml_mm, "Stage B port2 line-edge inside PML"
    assert z_split_mm > z_lo_coax_bot_mm, "Stage B z_split not above port1's line edge"
    assert z_split_mm < z_hi_coax_top_mm, "Stage B z_split not below port2's line edge"
    assert abs(layout["ref_plane_shift_port1_mm"] - B_REF_PLANE_SHIFT_MM) < 1e-6, (
        "Stage B port1 referral distance != B_REF_PLANE_SHIFT_MM (rfx geometry drift?)"
    )
    assert abs(layout["ref_plane_shift_port2_mm"] - B_REF_PLANE_SHIFT_MM) < 1e-6, (
        "Stage B port2 referral distance != B_REF_PLANE_SHIFT_MM (rfx geometry drift?)"
    )
    assert abs((z_feed_top_mm - z_feed_bot_mm) - B_L12_MM) < 1e-6, "L12 mismatch vs rfx fixture"
    return layout


def _ensure_openems_numpy_compat() -> None:
    """openEMS Python bindings still expect deprecated NumPy aliases."""
    for name, value in {"float": float, "int": int, "complex": complex, "mat": np.matrix}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _import_openems():
    """Deferred openEMS import (kept OUT of module scope on purpose).

    This module must stay importable without openEMS installed so
    ``tests/test_coax_two_port_referee_header.py`` can load it locally
    and check the header/record/layout fields (fail-loud-honest test
    design). The ImportError-to-exit-2 conversion happens in ``main()``.
    """
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS
    from openEMS.ports import CoaxialPort
    return ContinuousStructure, openEMS, CoaxialPort


# ---------------------------------------------------------------------------
# Pre-solve fail-fast gate (hand-ported sanity check (g);
# _configs/.claude/rules/vessl-jobs.md "Submission discipline").
# ---------------------------------------------------------------------------
_BAD_STDOUT_PATTERNS = ("Unused primitive", "not on the mesh", "unused excitation")


def _scan_stdout_for_bad_patterns(log_text: str, label: str) -> None:
    hits = [p for p in _BAD_STDOUT_PATTERNS if p.lower() in log_text.lower()]
    if hits:
        raise RuntimeError(
            f"[{label}] pre-solve mesh/port fail-fast gate tripped: openEMS "
            f"stdout contains {hits!r}. Aborting BEFORE the full NrTS "
            f"budget is spent."
        )


def _run_openems_capturing_stdout(fdtd, sim_path: str, *, threads: int) -> str:
    """Run openEMS while capturing its OS-level stdout to a file we can grep.

    ``fdtd.Run()`` invokes the openEMS C++ binary; its stdout goes to the
    process's OS-level fd 1, not Python's ``sys.stdout`` -- redirect fd 1
    itself, restoring it afterward even on error.
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


def _check_excitation_and_trace(port, sim_path: str, label: str) -> tuple[float, int]:
    """Hand-ported sanity checks (a) + (b): nonzero energy + nonzero trace.

    Works generically across port classes: ``CoaxialPort.U_filenames``
    holds 3 entries (its 3-plane differential probe), ``AddLumpedPort``
    held 1 -- this function just iterates whatever list is there.
    """
    uf_inc = np.asarray(port.uf_inc, dtype=np.complex128)
    inc_peak = float(np.max(np.abs(uf_inc))) if uf_inc.size else 0.0
    if not np.isfinite(inc_peak) or inc_peak == 0.0:
        raise RuntimeError(
            f"[{label}] openEMS port injected/received NO wave energy "
            f"(max|uf_inc|={inc_peak!r}): excitation did not couple or the "
            f"port never saw the wave."
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


def _passivity_witness(s11: np.ndarray, s21: np.ndarray, label: str, *, tol: float = 0.05) -> dict:
    """B5(a): per-bin energy balance |S11|^2+|S21|^2 <= 1+tol, HARD gate.

    Only the upper bound is enforced (any passive structure must satisfy
    it); the lower bound (how far below 1, i.e. how lossy) is recorded
    but NOT gated -- the true loss of either fixture is not known a
    priori, so gating a lower bound would be an unverified assumption.
    """
    balance = np.abs(s11) ** 2 + np.abs(s21) ** 2
    max_balance = float(np.max(balance))
    passed = bool(max_balance <= 1.0 + tol)
    if not passed:
        raise RuntimeError(
            f"[{label}] passivity witness failed: max(|S11|^2+|S21|^2)="
            f"{max_balance:.4f} > {1.0 + tol} -- non-physical energy gain."
        )
    return {"balance": balance.tolist(), "max_balance": max_balance, "tol": tol, "passed": passed}


def _matched_through_witness(freqs_hz: np.ndarray, s21: np.ndarray, *, L_m: float,
                             eps_r: float, label: str) -> dict:
    """B5(b): Stage-A-only, one-sided matched-through expectation.

    |S21|~=1, phase~=-beta*L, group delay~=L*sqrt(eps_r)/c over a central
    frequency sub-band (away from f=0 and the sweep's own edges, where a
    Gaussian-pulse spectrum is weak and DFT noise dominates). ONE-SIDED
    per the review: compared against an INDEPENDENTLY-known expected
    value, not against another internally-derived quantity (e.g. S12) --
    catches a systematic channel-convention error that a symmetric
    reciprocity check could miss if it affected both drives identically.
    """
    n = freqs_hz.size
    lo, hi = int(0.25 * n), int(0.75 * n)
    idx = slice(max(lo, 1), max(hi, lo + 1))  # skip f=0 and the extreme edges

    s21_mag = np.abs(s21)
    s21_band = s21_mag[idx]
    mag_lo, mag_hi = REPRODUCE_GATE_RECORD["gate"]["s21_thru_band"]
    mag_ok = bool(np.all((s21_band >= mag_lo) & (s21_band <= mag_hi)))

    beta = 2.0 * np.pi * freqs_hz * np.sqrt(eps_r) / _C0
    expected_phase = -beta * L_m
    measured_phase = np.unwrap(np.angle(s21))
    # Compare modulo 2*pi (absolute phase reference is convention-
    # dependent; the SLOPE / relative behavior is what beta predicts).
    phase_dev_rad = np.angle(np.exp(1j * (measured_phase - expected_phase)))
    max_phase_dev_deg = float(np.degrees(np.max(np.abs(phase_dev_rad[idx]))))

    omega = 2.0 * np.pi * freqs_hz
    gd_measured_s = -np.gradient(measured_phase, omega)
    gd_expected_s = L_m * np.sqrt(eps_r) / _C0
    gd_dev_ps = float(np.max(np.abs(gd_measured_s[idx] - gd_expected_s)) * 1e12)

    # Generous, explicitly-untested-a-priori tolerances (documented
    # engineering judgment, per external_solver_comparator.md's own
    # "no silent gate loosening" discipline -- these are the FIRST
    # numbers, not re-derived from a real run).
    phase_tol_deg = 30.0
    gd_tol_ps = 200.0
    phase_ok = bool(max_phase_dev_deg < phase_tol_deg)
    gd_ok = bool(gd_dev_ps < gd_tol_ps)
    passed = bool(mag_ok and phase_ok and gd_ok)

    result = {
        "band_s21_mag": s21_band.tolist(),
        "mag_band_expected": [mag_lo, mag_hi], "mag_ok": mag_ok,
        "max_phase_dev_deg": max_phase_dev_deg, "phase_tol_deg": phase_tol_deg, "phase_ok": phase_ok,
        "group_delay_dev_ps": gd_dev_ps, "gd_tol_ps": gd_tol_ps, "gd_ok": gd_ok,
        "expected_group_delay_ps": gd_expected_s * 1e12,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"[{label}] matched-through witness failed: {result}")
    return result


# ---------------------------------------------------------------------------
# STAGE A: faithful port of Coax.m.
# ---------------------------------------------------------------------------
def _build_stage_a_coax_tutorial(ContinuousStructure, openEMS, CoaxialPort, *,
                                 nrts: int, end_criteria: float, use_pml: bool):
    """Build Coax.m's geometry fresh (called for BOTH the pre-solve smoke
    run and the real run -- see ``_run_openems_capturing_stdout``'s
    docstring for why these are separate ``openEMS`` instances)."""
    fdtd = openEMS(NrTS=nrts, EndCriteria=end_criteria)
    fdtd.SetGaussExcite(A_F0_HZ, A_FC_HZ)
    bc = ["PEC", "PEC", "PEC", "PEC"] + (["PML_8", "PML_8"] if use_pml else ["MUR", "MUR"])
    fdtd.SetBoundaryCond(bc)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)
    x_lines = np.arange(-A_R_OS_MM, A_R_OS_MM + 0.5 * A_MESH_RES_MM, A_MESH_RES_MM)
    z_lines = np.arange(0.0, A_LENGTH_MM + 0.5 * A_MESH_RES_MM, A_MESH_RES_MM)
    mesh.SetLines("x", x_lines)
    mesh.SetLines("y", x_lines)
    mesh.SetLines("z", z_lines)

    copper = csx.AddMetal("copper")
    port1 = CoaxialPort(
        csx, port_nr=1, pec_prop=copper, mat_prop=None,
        start=[0.0, 0.0, 0.0], stop=[0.0, 0.0, A_LENGTH_MM / 2.0],
        prop_dir="z", r_i=A_R_I_MM, r_o=A_R_O_MM, r_os=A_R_OS_MM,
        excite_amp=1.0, FeedShift=A_FEEDSHIFT_MM, priority=10,
    )
    port2 = CoaxialPort(
        csx, port_nr=2, pec_prop=copper, mat_prop=None,
        start=[0.0, 0.0, A_LENGTH_MM / 2.0], stop=[0.0, 0.0, A_LENGTH_MM],
        prop_dir="z", r_i=A_R_I_MM, r_o=A_R_O_MM, r_os=A_R_OS_MM,
        priority=10,
    )
    return fdtd, port1, port2


def _run_stage_a_reproduce_gate(*, sim_root: str, threads: int, nrts: int,
                                end_criteria: float, use_pml: bool) -> dict:
    ContinuousStructure, openEMS, CoaxialPort = _import_openems()

    sim_dir = os.path.join(sim_root, "stage_a_coax_tutorial")
    smoke_dir = os.path.join(sim_root, "stage_a_coax_tutorial_smoke")

    smoke_fdtd, _sp1, _sp2 = _build_stage_a_coax_tutorial(
        ContinuousStructure, openEMS, CoaxialPort,
        nrts=min(200, nrts), end_criteria=0.0, use_pml=use_pml)
    smoke_log = _run_openems_capturing_stdout(smoke_fdtd, smoke_dir, threads=threads)
    _scan_stdout_for_bad_patterns(smoke_log, "stage_a_smoke")

    fdtd, port1, port2 = _build_stage_a_coax_tutorial(
        ContinuousStructure, openEMS, CoaxialPort,
        nrts=nrts, end_criteria=end_criteria, use_pml=use_pml)

    t0 = time.time()
    real_log = _run_openems_capturing_stdout(fdtd, sim_dir, threads=threads)
    _scan_stdout_for_bad_patterns(real_log, "stage_a")  # belt-and-suspenders
    elapsed = time.time() - t0

    port1.CalcPort(sim_dir, A_FREQS_HZ)
    port2.CalcPort(sim_dir, A_FREQS_HZ)

    inc_peak, n_samples = _check_excitation_and_trace(port1, sim_dir, "stage_a")
    truncated = bool(nrts > 0 and n_samples >= nrts)

    zl = np.asarray(port1.Z_ref, dtype=np.complex128)
    zl_dev = np.abs(zl.real - ZL_A_ANALYTIC_OHM)
    zl_re_ok = bool(np.max(zl_dev) < REPRODUCE_GATE_RECORD["gate"]["zl_re_tol_ohm"])
    zl_im_ok = bool(np.max(np.abs(zl.imag)) < REPRODUCE_GATE_RECORD["gate"]["zl_im_tol_ohm"])

    s11 = np.asarray(port1.uf_ref, dtype=np.complex128) / np.asarray(port1.uf_inc, dtype=np.complex128)
    s21 = np.asarray(port2.uf_inc, dtype=np.complex128) / np.asarray(port1.uf_inc, dtype=np.complex128)
    _non_physical_guard(np.abs(s11), "stage_a_s11")
    _non_physical_guard(np.abs(s21), "stage_a_s21")

    passivity = _passivity_witness(s11, s21, "stage_a")
    matched_thru = _matched_through_witness(
        A_FREQS_HZ, s21, L_m=A_L_THRU_MM * 1e-3, eps_r=1.0, label="stage_a")

    passed = bool(zl_re_ok and zl_im_ok and not truncated
                 and passivity["passed"] and matched_thru["passed"])

    return {
        "zl_real_ohm": zl.real.tolist(), "zl_imag_ohm": zl.imag.tolist(),
        "zl_mean_ohm": float(np.mean(zl.real)), "zl_max_dev_ohm": float(np.max(zl_dev)),
        "zl_re_ok": zl_re_ok, "zl_im_ok": zl_im_ok,
        "s11_mag": np.abs(s11).tolist(), "s21_mag": np.abs(s21).tolist(),
        "passivity": passivity, "matched_through_witness": matched_thru,
        "max_uf_inc": inc_peak, "n_trace_samples": n_samples, "nrts_cap": nrts,
        "truncated_suspected": truncated, "elapsed_s": round(elapsed, 1),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# STAGE B: rfx-matched through-line, via CoaxialPort.
# ---------------------------------------------------------------------------
def _build_stage_b_drive(ContinuousStructure, openEMS, CoaxialPort, drive: str, *,
                         nrts: int, end_criteria: float):
    """Build one Stage B drive's CSX/fdtd/ports fresh (smoke + real, same
    rationale as Stage A)."""
    layout = _stage_b_layout()
    dx_mm = B_DX_MM

    fdtd = openEMS(NrTS=nrts, EndCriteria=end_criteria)
    fdtd.SetGaussExcite(B_F0_GHZ * 1e9, B_FC_GHZ * 1e9)
    fdtd.SetBoundaryCond(["PML_16"] * 6)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    cx_mm, cy_mm = layout["cx_mm"], layout["cy_mm"]
    r_os_mm = B_B_MM + 2.0 * dx_mm
    fixed_xy = sorted({
        0.0, layout["lx_mm"], cx_mm, cx_mm - B_A_MM, cx_mm + B_A_MM,
        cx_mm - B_B_MM, cx_mm + B_B_MM, cx_mm - r_os_mm, cx_mm + r_os_mm,
    })
    fixed_z = sorted({
        0.0, layout["lz_mm"], layout["z_lo_coax_bot_mm"], layout["z_hi_coax_top_mm"],
        layout["z_split_mm"],
    })
    mesh.AddLine("x", fixed_xy)
    mesh.AddLine("y", fixed_xy)
    mesh.AddLine("z", fixed_z)
    mesh.SmoothMeshLines("all", dx_mm, 1.4)

    copper = csx.AddMetal("copper")
    ptfe = csx.AddMaterial("ptfe", epsilon=B_PTFE_EPS_R)

    excite1 = 1.0 if drive == "port1" else 0.0
    excite2 = 1.0 if drive == "port2" else 0.0
    feedshift_mm = B_FEEDSHIFT_CELLS * dx_mm
    measplane_mm = B_MEASPLANE_SHIFT_CELLS * dx_mm

    # port1: bottom, start at the line's own low edge, pointing +z (into
    # the line, toward z_split). port2: top, start at the line's own high
    # edge, pointing -z (into the line, toward z_split) -- mirror-
    # symmetric, matching rfx's own "each port looks into the line from
    # its own end" convention (NOT Coax.m's same-direction layout).
    port1 = CoaxialPort(
        csx, port_nr=1, pec_prop=copper, mat_prop=ptfe,
        start=[cx_mm, cy_mm, layout["z_lo_coax_bot_mm"]],
        stop=[cx_mm, cy_mm, layout["z_split_mm"]],
        prop_dir="z", r_i=B_A_MM, r_o=B_B_MM, r_os=r_os_mm,
        excite_amp=excite1, FeedShift=feedshift_mm, MeasPlaneShift=measplane_mm,
        priority=10,
    )
    port2 = CoaxialPort(
        csx, port_nr=2, pec_prop=copper, mat_prop=ptfe,
        start=[cx_mm, cy_mm, layout["z_hi_coax_top_mm"]],
        stop=[cx_mm, cy_mm, layout["z_split_mm"]],
        prop_dir="z", r_i=B_A_MM, r_o=B_B_MM, r_os=r_os_mm,
        excite_amp=excite2, FeedShift=feedshift_mm, MeasPlaneShift=measplane_mm,
        priority=10,
    )
    return fdtd, port1, port2, layout


def _extract_two_port_s(port_self, port_thru, freqs_hz: np.ndarray) -> dict:
    """S_self/S_thru for one drive, per Coax.m's OWN convention: s21 =
    port{2}.uf.inc ./ port{1}.uf.inc (confirmed by the tutorial's source,
    not derived from first principles as v1 had to -- see the module
    docstring's Stage A section)."""
    uf_inc_self = np.asarray(port_self.uf_inc, dtype=np.complex128)
    uf_ref_self = np.asarray(port_self.uf_ref, dtype=np.complex128)
    uf_inc_thru = np.asarray(port_thru.uf_inc, dtype=np.complex128)
    uf_ref_thru = np.asarray(port_thru.uf_ref, dtype=np.complex128)

    return {
        "s_self": uf_ref_self / uf_inc_self,
        "s_thru": uf_inc_thru / uf_inc_self,  # Coax.m convention, confirmed
        "s_thru_alternate_channel": uf_ref_thru / uf_inc_self,  # recorded, not used
        "uf_inc_self": uf_inc_self, "uf_ref_self": uf_ref_self,
        "uf_inc_thru": uf_inc_thru, "uf_ref_thru": uf_ref_thru,
    }


def _run_one_drive(ContinuousStructure, openEMS, CoaxialPort, *, drive: str, sim_root: str,
                   threads: int, nrts: int, end_criteria: float) -> dict:
    sim_dir = os.path.join(sim_root, f"stage_b_drive_{drive}")
    smoke_dir = os.path.join(sim_root, f"stage_b_drive_{drive}_smoke")

    smoke_fdtd, _sp1, _sp2, _layout = _build_stage_b_drive(
        ContinuousStructure, openEMS, CoaxialPort, drive,
        nrts=min(200, nrts), end_criteria=0.0)
    smoke_log = _run_openems_capturing_stdout(smoke_fdtd, smoke_dir, threads=threads)
    _scan_stdout_for_bad_patterns(smoke_log, f"stage_b_drive_{drive}_smoke")

    fdtd, port1, port2, layout = _build_stage_b_drive(
        ContinuousStructure, openEMS, CoaxialPort, drive,
        nrts=nrts, end_criteria=end_criteria)

    t0 = time.time()
    real_log = _run_openems_capturing_stdout(fdtd, sim_dir, threads=threads)
    _scan_stdout_for_bad_patterns(real_log, f"stage_b_drive_{drive}")
    elapsed = time.time() - t0

    port1.CalcPort(sim_dir, B_FREQS_HZ, ref_plane_shift=layout["ref_plane_shift_port1_mm"])
    port2.CalcPort(sim_dir, B_FREQS_HZ, ref_plane_shift=layout["ref_plane_shift_port2_mm"])

    driven_port = port1 if drive == "port1" else port2
    inc_peak, n_samples = _check_excitation_and_trace(driven_port, sim_dir, f"stage_b_drive_{drive}")
    truncated = bool(nrts > 0 and n_samples >= nrts)

    if drive == "port1":
        extracted = _extract_two_port_s(port1, port2, B_FREQS_HZ)
    else:
        extracted = _extract_two_port_s(port2, port1, B_FREQS_HZ)

    return {
        "extracted": extracted,
        "z0_port1": np.asarray(getattr(port1, "Z_ref", np.nan), dtype=np.complex128),
        "z0_port2": np.asarray(getattr(port2, "Z_ref", np.nan), dtype=np.complex128),
        "beta_port1": np.asarray(getattr(port1, "beta", np.nan), dtype=np.complex128),
        "beta_port2": np.asarray(getattr(port2, "beta", np.nan), dtype=np.complex128),
        "max_uf_inc": inc_peak, "n_trace_samples": n_samples, "nrts_cap": nrts,
        "truncated_suspected": truncated, "elapsed_s": round(elapsed, 1),
    }


def _group_delay(freqs_hz: np.ndarray, s21: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phase = np.unwrap(np.angle(s21))
    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64)
    gd_s = -np.gradient(phase, omega)
    return phase, gd_s


def _run_stage_b(*, sim_root: str, threads: int, nrts: int, end_criteria: float) -> dict:
    ContinuousStructure, openEMS, CoaxialPort = _import_openems()

    drive1 = _run_one_drive(ContinuousStructure, openEMS, CoaxialPort, drive="port1",
                            sim_root=sim_root, threads=threads, nrts=nrts, end_criteria=end_criteria)
    drive2 = _run_one_drive(ContinuousStructure, openEMS, CoaxialPort, drive="port2",
                            sim_root=sim_root, threads=threads, nrts=nrts, end_criteria=end_criteria)

    s11 = drive1["extracted"]["s_self"]
    s21 = drive1["extracted"]["s_thru"]
    s22 = drive2["extracted"]["s_self"]
    s12 = drive2["extracted"]["s_thru"]

    for label, arr in (("s11", s11), ("s21", s21), ("s12", s12), ("s22", s22)):
        _non_physical_guard(np.abs(arr), label)

    passivity_drive1 = _passivity_witness(s11, s21, "stage_b_drive1")
    passivity_drive2 = _passivity_witness(s22, s12, "stage_b_drive2")

    recip_mag_dev = float(np.max(np.abs(np.abs(s21) - np.abs(s12))))
    recip_phase_dev_deg = float(np.max(np.abs(np.degrees(np.angle(s21) - np.angle(s12)))))

    phase21, gd21_s = _group_delay(B_FREQS_HZ, s21)

    sanity_passed = bool(
        not drive1["truncated_suspected"] and not drive2["truncated_suspected"]
        and passivity_drive1["passed"] and passivity_drive2["passed"]
    )

    return {
        "freqs_ghz": B_FREQS_GHZ.tolist(),
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
        "passivity_drive1": passivity_drive1, "passivity_drive2": passivity_drive2,
        "z0_port1_drive1": [[float(c.real), float(c.imag)] for c in drive1["z0_port1"]],
        "beta_port1_drive1": [[float(c.real), float(c.imag)] for c in drive1["beta_port1"]],
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
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--nrts", type=int, default=200000)
    p.add_argument("--end-criteria", type=float, default=1e-4)
    p.add_argument("--use-pml", action="store_true",
                   help="Stage A: use PML_8 instead of Coax.m's default MUR z boundary")
    p.add_argument("--skip-stage-b", action="store_true",
                   help="run only the Stage A reproduce-gate (fast smoke check)")
    args = p.parse_args(argv)

    # Pure-arithmetic layout check FIRST, before touching openEMS at all --
    # an AssertionError here is an internal config bug (exit 3), distinct
    # from a physics/self-check failure (exit 1).
    try:
        _stage_b_layout()
    except AssertionError as exc:
        print(f"CONFIG ERROR: Stage B layout assertion failed: {exc}", file=sys.stderr)
        return 3

    try:
        _import_openems()
    except ImportError as exc:
        print(f"SKIP: openEMS Python bindings not importable ({exc}).\n"
              "  This script is VESSL-only (source-built openEMS); see "
              "scripts/vessl_coax_two_port_referee.yaml.", file=sys.stderr)
        return 2

    print("=" * 78)
    print("openEMS external referee -- rfx issue #489 stage 3 (coax two-port)")
    print("COMPARATOR LEG ONLY. Stage A = faithful Coax.m port (reproduce-gate).")
    print("See module docstring for the full scope fence, delta list, do-not-repeat.")
    print("=" * 78)
    print(f"\nDeclared question:\n  {DECLARED_QUESTION}")
    print(f"\n{RFX_REFERENCE_QUOTE}")

    t_start = time.time()
    sim_root = os.path.abspath(args.sim_root)

    try:
        print("\n--- Stage A: reproduce-gate (Coax.m, faithful port) ---")
        stage_a = _run_stage_a_reproduce_gate(
            sim_root=sim_root, threads=args.threads, nrts=args.nrts,
            end_criteria=args.end_criteria, use_pml=args.use_pml,
        )
        print(f"  ZL: mean={stage_a['zl_mean_ohm']:.3f} ohm "
              f"(analytic {ZL_A_ANALYTIC_OHM:.3f}) max_dev={stage_a['zl_max_dev_ohm']:.3f} ohm")
        print(f"  passivity passed: {stage_a['passivity']['passed']}, "
              f"matched-through witness passed: {stage_a['matched_through_witness']['passed']}")
        print(f"  Stage A passed: {stage_a['passed']}")

        stage_b = None
        if stage_a["passed"] and not args.skip_stage_b:
            print("\n--- Stage B: rfx-matched through-line (CoaxialPort) ---")
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
    except AssertionError as exc:
        print(f"\nCONFIG ERROR (internal script bug, not a physics finding): {exc}", file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(f"\nSELF-CHECK / PHYSICS-GATE FAILED: {exc}", file=sys.stderr)
        elapsed_total = time.time() - t_start
        artifact = {
            "issue": 489, "stage": "3 (external openEMS referee)",
            "scope": "comparator leg only",
            "error": str(exc), "elapsed_s": round(elapsed_total, 1),
        }
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2)
        return 1

    overall_passed = bool(
        stage_a["passed"] and (stage_b is None or stage_b["sanity_passed"])
    )

    elapsed_total = time.time() - t_start
    artifact = {
        "issue": 489, "stage": "3 (external openEMS referee)",
        "scope": "comparator leg only -- brackets, does not judge rfx's own numbers",
        "declared_question": DECLARED_QUESTION,
        "rfx_reference_quote": RFX_REFERENCE_QUOTE,
        "must_move_when_validated": MUST_MOVE_WHEN_VALIDATED,
        "reproduce_gate_record": REPRODUCE_GATE_RECORD,
        "stage_a": stage_a, "stage_b": stage_b,
        "overall_passed": overall_passed, "elapsed_s": round(elapsed_total, 1),
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
