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
FULL domain in z (H2' fix below), split at the domain's own geometric
midpoint (``z_split``). BOTH ports have ``direction=+1`` (port 1:
start=z=0, stop=z_split; port 2: start=z_split, stop=lz) -- Coax.m's OWN
same-direction cascaded-segments layout (Coax.m's two ports both point
+z, sequential SEGMENTS of one propagating wave), NOT the mirror-
symmetric "each port looks into the line from its own end" convention
rfx itself uses internally.

B4' FIX (round-2 review, BLOCKING): v2 built port 2 with stop<start
(direction=-1), mirroring rfx's own port-orientation convention. This
was WRONG for ``CoaxialPort`` specifically: its current probe is
direction-SIGNED (``CSX.AddProbe(..., weight=self.direction, ...)`` in
``python/openEMS/ports.py``'s ``CoaxialPort.__init__``), so a port's own
``uf_inc``/``uf_ref`` split is tied to ITS OWN geometric direction, not
to a convention shared across ports regardless of orientation. The
reviewer ran openEMS's own ``CalcPort`` formulas on a synthetic forward
TEM wave: Coax.m's own same-direction layout gives
``|uf_inc2/uf_inc1|=1.0`` (correct); the mirror-symmetric (direction=-1
port 2) layout this script previously used gives ``|uf_inc2/uf_inc1|=0.0``
-- Coax.m's validated convention does NOT transfer across an orientation
flip. Fix: both ports now share direction=+1, exactly matching Coax.m,
so Stage A's validated ``s21 = uf_inc[thru]/uf_inc[self]`` convention
transfers unchanged for the port1-driven case.

This same-direction (not mirror-symmetric) topology has a DERIVED
consequence for the port2-driven (REVERSE) case that Coax.m itself never
exercises (Coax.m only ever drives port 1): BOTH the thru numerator AND
the self denominator depend on which physical direction is "into the
device" from the driven port's own location, not just the numerator --
  - FORWARD drive (port 1, bottom): +z IS into the device, so port 1's
    own launched wave is ``uf_inc_self``. s_self (S11) = ``uf_ref_self/
    uf_inc_self`` (Coax.m's own formula, unchanged). s_thru (S21) =
    ``uf_inc_thru/uf_inc_self`` -- the wave arrives at port 2 travelling
    +z, ALIGNED with port 2's own convention.
  - REVERSE drive (port 2, top): +z is AWAY from the device, so port 2's
    own launched wave is ``uf_ref_self`` (NOT ``uf_inc_self``). s_self
    (S22) = ``uf_inc_self/uf_ref_self``. s_thru (S12) = ``uf_ref_thru/
    uf_ref_self`` -- the wave arrives at port 1 travelling -z, OPPOSED to
    port 1's own convention, AND the denominator is port 2's own launched
    (-z) wave, not its +z one.

ROUND-3 FIX (2026-08-03): an earlier version of this rebuild flipped
ONLY the thru numerator for the reverse drive and left s_self as
``uf_ref_self/uf_inc_self`` unconditionally -- WRONG, since
``uf_inc_self`` is not the reverse drive's launched wave. The reviewer's
synthetic check (plain complex arithmetic on fake ``uf_inc``/``uf_ref``
values, no openEMS needed -- see ``_extract_two_port_s``'s docstring and
``tests/test_coax_two_port_referee_header.py::
test_extract_two_port_s_synthetic_forward_and_reverse_drive``) caught it:
the numerator-only fix reported S22 as 1/S22_true (0.051 -> 19.61) and
S12 scaled by the same inverted factor (0.9 -> 17.65). Both directions
are now confirmed by that synthetic check (the forward direction was
already confirmed by Coax.m's own source; the reverse direction, which
Coax.m never exercises, by the reviewer's CalcPort-formula arithmetic).

Both ports' computed directions are pinned by
``tests/test_coax_two_port_referee_header.py::
test_stage_b_port_directions_are_both_positive`` (closes the round-1
hole: a mutated orientation previously left all 13 tests green).

LINE TERMINATION TOPOLOGY (H2, RESTORED 2026-08-03 after a diagnosis
reversal -- read this section before touching the boundary condition
again): the coax conductors run straight to the domain edges, INTO
openEMS's PML_16 at both z ends. ``CoaxialPort`` has no matched-
impedance ``Feed_R`` (only 0=short or inf=open are implemented -- finite
Feed_R>0 raises ``NotImplementedError``), so a line simply CONTINUING
into the absorber is the only matched-like termination openEMS's own
primitives offer -- and this repo's OWN do-not-repeat record
(``build_coaxial_line_openems_broad_comparison.py``, lines ~100-116)
independently validates exactly this pattern for a PTFE-filled coax:
"the full coax cross-section (pin+PTFE+shell) extends uniformly THROUGH
both PML regions so PML is reflectionless (constant cross-section into
the PML)."

RUN-1 DIAGNOSIS AND ITS REVERSAL (both recorded here so a future reader
does not repeat either mistake): the first live run (VESSL 369367251366)
failed its own non-physical-field guard -- [s11] |S| max=280.7 -- with
both drives' real FDTD solves separately reporting "Max. number of
timesteps was reached before the end-criteria of -40dB was reached"
after the full 200000-timestep budget. The FIRST diagnosis attributed
this to the PML-into-conductor topology above and replaced the z-
boundary with MUR, reasoning that Stage A's clean MUR pass (air-filled,
epsR=1) showed the conductor-to-MUR-wall combination was safe. That fix
was WRONG AT THE PREMISE LEVEL, and the error was avoidable: this
script's OWN do-not-repeat quote, a few paragraphs above (DO-NOT-REPEAT
section), already names "MUR-on-PTFE instability" as one of three
recorded failure modes, and the do-not-repeat file itself records the
MEASURED mechanism: "MUR on the +-z faces assumes phase velocity == c,
but the PTFE (eps_r=2.1) TEM wave travels at c/sqrt(2.1) -> MUR-on-
dielectric is unstable (exponential energy blow-up 5e-16 -> 2.8e13)."
Permittivity is exactly the parameter MUR's stability depends on --
Stage A is air-filled (v=c, MUR matched by construction) and Stage B is
PTFE-filled (eps_r=2.1), so Stage A's clean pass does NOT transfer, and
"MUR is what Coax.m uses, and Coax.m worked" was never a valid argument
for Stage B specifically. The MUR change has been REVERTED; PML_16 (with
conductors running through it, as above) is restored.

The sharper, now-LEAD hypothesis for what actually caused run-1's
failure is NOT the boundary topology -- it is a measurement-plane
ordering bug in ``_stage_b_layout()`` (present in the ORIGINAL PML_16
code too, unrelated to the MUR detour) -- see "REFERENCE-PLANE REFERRAL"
below for the mechanism and the M1 fix. Because run-1's own JSON
artifact was never written (a separate bug, B2 below), there is no
per-bin S-parameter or energy trace surviving from that run to
DISCRIMINATE between "topology-induced field instability" and "a
measurement-plane artifact" as the actual cause -- run.log's printed
warnings are consistent with either. Both are addressed here (PML_16
restored; the ordering fixed), but which one (or both) actually explains
run-1 remains UNSETTLED pending a rerun with readable artifacts (see the
updated falsifier at the end of this section).

REFERENCE-PLANE REFERRAL (the capability this referee exists to exercise):
each port's own field-probe geometry (``FeedShift``/``MeasPlaneShift``,
both computed in ``_stage_b_layout()``) sits near, but not exactly at,
rfx's own feed plane, and ``CalcPort(ref_plane_shift=...)`` does the
referral. ``ref_plane_shift`` and ``MeasPlaneShift`` share ONE convention
inside openEMS's own ``CalcPort`` (``python/openEMS/ports.py``, base
``Port.CalcPort``): ``shift = ref_plane_shift - self.measplane_shift``,
both "distance from this port's own ``start``," before the TL
phase-rotation transform is applied -- confirmed by reading the fetched
``ports.py`` source directly (not assumed), since an earlier round of
this diagnosis got exactly this kind of api trust wrong once already.

M1 FIX (PR #546 review, the sharper hypothesis for run-1): the version
before this one placed ``MeasPlaneShift`` only 2 cells short of the
target and ``FeedShift`` 5-7 cells FURTHER OUT than that -- i.e. the
MEASUREMENT plane sat BETWEEN the absorbing z-boundary and the
excitation, the reverse of Coax.m's own ordering (FeedShift near the
port's own outer end, MeasPlaneShift at the port's own midpoint, far
inside). ``CoaxialPort.__init__`` (``ports.py``, "Snap measurement plane
to mesh and extract 3 adjacent lines") builds ``uf_tot``/``if_tot`` --
and, via a raw finite-difference derivative across that SAME 3-line
stencil, ``beta`` and ``Z_ref`` too (``ReadUIData``: ``dEt = (ui_f_val[2]
- ui_f_val[0]) / (...)``) -- from whatever that stencil samples. A
stencil sitting between the wall and the feed samples mostly boundary-
reflected energy, not the genuine device-bound wave; as the absorber
does its job better, that reading shrinks toward zero. This is exactly
the script's own documented CHANNEL-INVERSION SIGNATURE (see
``_extract_two_port_s``'s docstring: "the SELF ratio ... blows up to
roughly 1/(true value)"), and it is quantitatively the same SHAPE of
failure the do-not-repeat file records for a different port class: "weak/
zero coupling with uf_inc~6e-14." Fix: restore Coax.m's own ordering --
FeedShift a few cells past the PML's own inner edge (near the wall, so
the excitation's backward-going lobe is promptly absorbed -- this repo's
own do-not-repeat recipe: "place the feed a few cells below the PML
inner edge, in the clean line"), MeasPlaneShift well inside on the
DEVICE side, comfortably clear of both the PML and FeedShift (see the
``B_FEED_FROM_PML_EDGE_CELLS``/``B_MEAS_FROM_PML_EDGE_CELLS`` constants
and ``_stage_b_layout()`` for the exact cell counts and the M2/M3
assertions that pin this ordering so a regression fails loud, at exit 3,
not silently at run time).

Because ``ref_plane_shift`` is unaffected by WHERE a port measures (it
is purely "target distance from ``start``"), this reorder does not
change ``ref_plane_shift_port{1,2}_mm``'s own formula at all -- only
``self.measplane_shift`` (openEMS's own re-derivation of MeasPlaneShift
from its mesh-snapped stencil) moves, which can now put MeasPlaneShift
on the FAR side of the target from ``start`` for port 1 specifically
(target at ~19 cells from its wall, MeasPlaneShift now at ~36) --
``shift = ref_plane_shift - measplane_shift`` goes NEGATIVE in that case
(~-17 cells). ``CalcPort`` handles this correctly: the phase-rotation
formula (``phase = real(beta)*shift``; ``uf_tot' = uf_tot*cos(-phase) +
j*if_tot*Z_ref*sin(-phase)``, etc.) is a plain trigonometric transform,
sign-symmetric in ``shift`` -- a negative shift is a backward TL
transform, not a special or ill-conditioned case.

UPDATED FALSIFIER (supersedes any earlier one; this is the discriminating
experiment run-1 could not provide): rerun with this restored/corrected
layout and B2's serialization fix (so the artifact is actually readable
regardless of outcome). If both drives' energy settles below -40 dB and
all four |S| values stay <~1 (passivity), the measurement-plane ordering
(M1) was run-1's cause -- the boundary topology (H2) is exonerated. If
energy genuinely fails to settle WITH readable per-drive energy/S data
this time, the topology hypothesis (H2/H3a) reopens on REAL forensics,
not a re-guess. Either way, no further topology change without that
data.

DELTA LIST (every remaining way this openEMS model differs from the rfx
fixture):
  1. BOUNDARY TOPOLOGY: rfx's ``compute_coaxial_two_port`` requires CPML
     on all six faces (``rfx/api/_sparams.py``, "requires CPML tokens on
     all six boundary faces"). Stage B uses ``PML_16`` (matching rfx's
     cpml_layers=16) on ALL six faces -- NOT Coax.m's own PEC-box+MUR
     topology, which is reserved for Stage A's literal tutorial
     reproduction only (module docstring "LINE TERMINATION TOPOLOGY" has
     the run-1 diagnosis reversal: a MUR z-boundary was tried and
     reverted -- MUR-on-PTFE is a RECORDED-unstable combination in this
     repo's own do-not-repeat file, not a fix).
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
     split. The backward-going lobe (toward each port's own domain edge,
     running into the PML per the restored H2 topology) is expected to
     be absorbed -- this repo's own do-not-repeat file validates exactly
     this "conductor extends through the PML, PML absorbs the backward
     lobe" pattern for a PTFE-filled coax (module docstring "LINE
     TERMINATION TOPOLOGY"). M1 fix (PR #546 review): FeedShift is now
     placed close enough to that same boundary that the backward lobe is
     absorbed promptly, without contaminating MeasPlaneShift, which sits
     well inside on the device side instead (previously the reverse
     order, which is the sharper lead hypothesis for run-1 -- see
     "REFERENCE-PLANE REFERRAL").
  4. MESH: rfx auto-derives dx=dy=dz=3.7474057249999997e-4 m from
     freq_max=40e9. Stage B uses the SAME cell size on all three axes.
  5. CPML DEPTH / DOMAIN CONVENTION: rfx pads 16 cells BEYOND its
     rasterized interior (additive). openEMS's ``PML_16`` instead
     consumes 16 cells FROM the declared mesh (subtractive, same
     convention ``thru_openems.py`` uses). Stage B declares a total mesh
     2*16 cells LARGER than the rasterized clear span on every padded
     axis (x, y, AND z -- restored) so the CLEAR (non-PML) interior
     matches rfx's RASTERIZED interior exactly (B3 fix), with matching
     PML depth on top.
  6. PORT-TO-PORT SPAN: both ports' referred reference planes sit at
     rfx's own z=1.1242217175mm and z=59.58375102749999mm (rasterized-
     interior-relative frame), L12=58.4595293mm exactly -- via
     ``ref_plane_shift``, not by physically building the ports at those
     exact locations. The ports THEMSELVES physically span the FULL
     domain edge-to-edge (z=0 to z_split, z_split to lz) -- much longer
     than L12 for port 2 -- with the referral doing all the work of
     landing the REPORTED S-parameters at rfx's own, much shorter,
     feed-to-feed span.
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
  (h) B5/B5' witnesses, WIRED INTO ``passed``/``sanity_passed`` (not just
      recorded):
      - per-bin passivity: |S11|^2+|S21|^2 <= 1+tol (hard upper bound for
        any passive structure; a lower bound is NOT gated since the true
        loss is not known a priori -- only recorded). Stage A's own
        passivity gate is computed over the SAME central frequency
        sub-band as the matched-through witness below (M1' fix -- Z_ref/
        beta are ill-conditioned at DC and the sweep's own edges, and
        Coax.m's own ``linspace(0, 2*f0, 201)`` starts AT f=0).
      - Stage-A matched-through expectation: |S21|~=1 with phase~=-beta*L
        and group delay~=L/c (Coax.m's line is air-filled, lossless,
        non-dispersive TEM) over a central frequency sub-band, away from
        f=0 and the sweep's own edges. ONE-SIDED per the review: this
        compares against an INDEPENDENTLY-known expected value (~1), not
        against another internally-derived quantity like S12 -- a
        systematic channel-convention error (e.g. reading the wrong
        channel for port 2) would read close to 0 here, which a
        symmetric reciprocity check could fail to catch if the SAME error
        affected both drives identically.
      - Stage-B matched-through expectation (B5' fix, round-2 review):
        the SAME one-sided witness, re-parameterized for Stage B's own
        fixture -- |S21| in [0.5, 1.1] (rfx's own raw profile is
        0.960->0.737, PR #534) and group delay ~= L12*sqrt(eps_r)/c ~=
        282 ps (L12=58.4595mm, eps_r=2.1) -- wired into Stage B's own
        ``sanity_passed``, not just Stage A's ``passed``.
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
# path). z_lo_coax_bot/z_hi_coax_top are the extent rfx ITSELF stamps
# (2 cells inside its own CPML, per its "PEC into CPML is numerically-
# unstable" rule) -- kept here as DOCUMENTATION for how close rfx's own
# feed planes sit to its own line edge; they no longer drive Stage B's
# own port span directly (H2 fix: Stage B's ports run to the ACTUAL
# domain edges, INTO the PML, not rfx's retracted extent -- see module
# docstring "LINE TERMINATION TOPOLOGY"). z_feed_bot/z_feed_top ARE still
# the exact rfx target reference planes ref_plane_shift referral lands
# on.
B_Z_LO_COAX_BOT_REL_MM = 0.749481145
B_Z_HI_COAX_TOP_REL_MM = 59.958491599999995
B_Z_FEED_BOT_REL_MM = 1.1242217174999998
B_Z_FEED_TOP_REL_MM = 59.58375102749999
B_L12_MM = B_Z_FEED_TOP_REL_MM - B_Z_FEED_BOT_REL_MM  # 58.4595293 mm, port-to-port

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

# B5' fix (round-2 review): Stage B's own one-sided matched-through
# expectation band. Rfx's own RAW |S21| profile is 0.960->0.737 over
# 4-12 GHz (PR #534) -- a real, lossy line, unlike Stage A's ideal
# lossless air-filled Coax.m tutorial (whose band is [0.5, 1.3]). [0.5,
# 1.1] brackets rfx's own measured range with margin on both sides while
# still catching a channel-inversion (~0) one-sidedly.
B_S21_THRU_BAND = (0.5, 1.1)

# FEED/MEASURE PLANE PLACEMENT (M1 fix, PR #546 review, 2026-08-03 --
# supersedes the pre-review "MeasPlaneShift 2 cells short of the target,
# FeedShift 5-7 cells further out" scheme entirely; see module docstring
# "REFERENCE-PLANE REFERRAL" for the full mechanism). Both FeedShift and
# MeasPlaneShift are now placed as FIXED cell offsets from each port's
# own OUTER (PML-adjacent) end, mirroring Coax.m's own ordering: FEED
# near the wall (a few cells past the PML's own inner edge -- close
# enough that the excitation's backward-going lobe is absorbed promptly,
# matching this repo's own do-not-repeat recipe,
# build_coaxial_line_openems_broad_comparison.py's "place the feed a few
# cells below the PML inner edge, in the clean line"), MEASURE well
# inside on the DEVICE side, comfortably clear of both the PML and of
# FeedShift. The prior scheme put MEASURE nearer the wall than FEED --
# backwards -- which is the sharper lead hypothesis for run-1's |S11|=280
# (module docstring "REFERENCE-PLANE REFERRAL" has the full derivation).
B_FEED_FROM_PML_EDGE_CELLS = 2   # feed: a few cells past the PML's inner edge
B_MEAS_FROM_PML_EDGE_CELLS = 20  # measure: well inside, clear of PML and feed
B_MEAS_STENCIL_MARGIN_CELLS = 5  # M3 fix: extra clearance so CoaxialPort's
                                  # own 3-line probe stencil (ports.py,
                                  # CoaxialPort.__init__ "Snap measurement
                                  # plane to mesh and extract 3 adjacent
                                  # lines") cannot land back on a
                                  # PML-adjacent mesh line


def _stage_b_layout() -> dict:
    """Pure arithmetic: every Stage B z-position/clearance, in mm.

    Deliberately openEMS-FREE so it is testable locally without the
    solver installed (review fix: distinguish a config/layout bug,
    AssertionError -> exit 3, from a physics/self-check failure,
    RuntimeError -> exit 1 -- this function is where the former would
    originate, and its assertions are checked by
    tests/test_coax_two_port_referee_header.py without needing openEMS).

    H2 (RESTORED 2026-08-03, PR #546 review): both ports' conductors run
    straight to the domain's own z edges (z=0, z=lz_mm), INTO PML_16 --
    Coax.m's own topology, and this repo's own do-not-repeat record's own
    validated pattern for a PTFE-filled coax specifically (module
    docstring "LINE TERMINATION TOPOLOGY" has the full run-1 diagnosis
    reversal: a MUR z-boundary was tried in between and reverted -- MUR
    is a RECORDED-unstable choice for a PTFE-filled line in this repo's
    own history, not a fix).
    M1 fix (PR #546 review): FeedShift/MeasPlaneShift are ordered like
    Coax.m's own convention -- FEED near the wall, MEASURE well inside on
    the device side -- not the reverse (see the ``B_FEED_FROM_PML_EDGE_
    CELLS`` comment above for the full mechanism).
    B4' fix: port 2 spans [z_split, lz_mm] (start < stop, direction=+1),
    NOT [lz_mm, z_split] -- both ports now share direction=+1, matching
    Coax.m's own same-direction layout (module docstring "PORT CLASS").
    """
    pml_mm = B_PML_DEPTH_MM
    lx_mm = ly_mm = B_CLEAR_X_MM + 2.0 * pml_mm
    lz_mm = B_CLEAR_Z_MM + 2.0 * pml_mm
    cx_mm = lx_mm / 2.0
    cy_mm = ly_mm / 2.0

    z_port1_start_mm = 0.0     # domain floor -- INTO the PML (H2, restored)
    z_port2_stop_mm = lz_mm    # domain ceiling -- INTO the PML (H2, restored)
    z_split_mm = 0.5 * lz_mm   # arbitrary shared midpoint, both ports meet here
    z_port1_stop_mm = z_split_mm
    z_port2_start_mm = z_split_mm

    # rfx's own reference/feed planes -- unaffected by any of this, still
    # safely inside the clear (non-PML) region.
    z_feed_bot_mm = pml_mm + B_Z_FEED_BOT_REL_MM
    z_feed_top_mm = pml_mm + B_Z_FEED_TOP_REL_MM

    # ref_plane_shift is "distance from this port's own `start` to the
    # target" -- the SAME convention CoaxialPort's own MeasPlaneShift
    # kwarg uses (openEMS ports.py, base Port.CalcPort: ``shift =
    # ref_plane_shift - self.measplane_shift``, both in that convention
    # before the length-unit/TL-phase-rotation transform). This formula
    # is UNCHANGED by the M1 reorder below, which only moves WHERE each
    # port physically measures/excites, not where the TARGET itself is.
    # Port 1's distance is small (~19 cells, ~7.12mm) since its `start`
    # sits at the near z edge; port 2's is large (~78 cells, ~29.23mm)
    # because its `start` is anchored at z_split (B4'), far from the
    # target near the FAR z edge -- an asymmetry from B4', not from M1.
    ref_plane_shift_port1_mm = z_feed_bot_mm - z_port1_start_mm
    ref_plane_shift_port2_mm = z_feed_top_mm - z_port2_start_mm

    cell = B_DX_MM
    feed_from_wall_mm = (B_CPML_CELLS + B_FEED_FROM_PML_EDGE_CELLS) * cell
    meas_from_wall_mm = (B_CPML_CELLS + B_MEAS_FROM_PML_EDGE_CELLS) * cell

    port1_span_mm = z_port1_stop_mm - z_port1_start_mm
    port2_span_mm = z_port2_stop_mm - z_port2_start_mm

    # Port 1: its own `start` (z=0) IS the near wall -- FeedShift/
    # MeasPlaneShift are plain offsets from it, matching Coax.m's own
    # FeedShift convention directly.
    feedshift_port1_mm = feed_from_wall_mm
    measplane_port1_mm = meas_from_wall_mm

    # Port 2: its own `start` is z_split (the DEVICE-facing end, B4'),
    # NOT the wall -- its wall is at `stop`. "Near the wall"/"well inside
    # on the device side" therefore both mirror across the port's own
    # span before being expressed as offsets from `start`.
    feedshift_port2_mm = port2_span_mm - feed_from_wall_mm
    measplane_port2_mm = port2_span_mm - meas_from_wall_mm

    layout = {
        "lx_mm": lx_mm, "ly_mm": ly_mm, "lz_mm": lz_mm,
        "cx_mm": cx_mm, "cy_mm": cy_mm,
        "z_port1_start_mm": z_port1_start_mm, "z_port1_stop_mm": z_port1_stop_mm,
        "z_port2_start_mm": z_port2_start_mm, "z_port2_stop_mm": z_port2_stop_mm,
        "z_split_mm": z_split_mm,
        "z_feed_bot_mm": z_feed_bot_mm, "z_feed_top_mm": z_feed_top_mm,
        "ref_plane_shift_port1_mm": ref_plane_shift_port1_mm,
        "ref_plane_shift_port2_mm": ref_plane_shift_port2_mm,
        "measplane_port1_mm": measplane_port1_mm, "feedshift_port1_mm": feedshift_port1_mm,
        "measplane_port2_mm": measplane_port2_mm, "feedshift_port2_mm": feedshift_port2_mm,
    }

    # Clearance / config assertions -- an AssertionError here is a BUG in
    # this script's own geometry math (exit 3), not a physics finding.
    #
    # B4' hole-closer: both ports' computed DIRECTION must be +1 (matching
    # Coax.m -- the whole point of the B4' fix). A silent reintroduction
    # of the mirror-symmetric (direction=-1 port 2) layout must fail HERE,
    # not just in the test that reads these same fields.
    direction_port1 = 1.0 if (z_port1_stop_mm - z_port1_start_mm) > 0 else -1.0
    direction_port2 = 1.0 if (z_port2_stop_mm - z_port2_start_mm) > 0 else -1.0
    assert direction_port1 == 1.0, "Stage B port1 direction != +1 (B4' regression)"
    assert direction_port2 == 1.0, "Stage B port2 direction != +1 (B4' regression)"

    # Reference-plane clearance: the TARGET planes (not the bare
    # conductor extent, which legitimately reaches the domain edges, into
    # the PML, per the restored H2) must sit safely inside the clear,
    # non-PML region.
    assert z_feed_bot_mm > pml_mm, "Stage B port1 target reference plane inside PML"
    assert z_feed_top_mm < lz_mm - pml_mm, "Stage B port2 target reference plane inside PML"
    assert z_port1_start_mm < z_split_mm < z_port2_stop_mm, "Stage B port ordering broken"
    assert abs((z_feed_top_mm - z_feed_bot_mm) - B_L12_MM) < 1e-6, "L12 mismatch vs rfx fixture"

    # M1/M2/M3 fixes (PR #546 review): FEED must sit strictly past the
    # PML's own inner edge (never inside the absorber -- a source inside
    # PML is not physically meaningful), MEASURE must sit strictly
    # FURTHER from the wall than FEED (the M1 ordering fix itself, pinned
    # here so a future regression back to the old, backwards order fails
    # LOUD at exit 3, not silently at run time), and MEASURE additionally
    # needs a stencil-safety margin beyond the PML edge so CoaxialPort's
    # own 3-line probe snap cannot land back on a PML-adjacent line
    # (which would corrupt not just uf_tot/if_tot but the finite-
    # difference-derived beta/Z_ref too -- ports.py ReadUIData).
    assert feed_from_wall_mm > pml_mm, (
        "Stage B FeedShift does not clear the PML (M2 regression)"
    )
    assert meas_from_wall_mm > feed_from_wall_mm, (
        "Stage B MeasPlaneShift is not device-side of FeedShift (M1 regression)"
    )
    assert meas_from_wall_mm > pml_mm + B_MEAS_STENCIL_MARGIN_CELLS * cell, (
        "Stage B MeasPlaneShift stencil margin too thin (M3 regression)"
    )

    # FeedShift/MeasPlaneShift must land strictly inside each port's own
    # span (not past z_split, not past the domain edge) and stay
    # positively separated from each other (no near-field overlap).
    for name, meas, feed, span in (
        ("port1", measplane_port1_mm, feedshift_port1_mm, port1_span_mm),
        ("port2", measplane_port2_mm, feedshift_port2_mm, port2_span_mm),
    ):
        assert 0.0 < meas < span, f"Stage B {name} MeasPlaneShift outside its own span"
        assert 0.0 < feed < span, f"Stage B {name} FeedShift outside its own span"
        assert abs(meas - feed) > cell, f"Stage B {name} FeedShift/MeasPlaneShift too close"

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


_EXCITATION_ENERGY_FLOOR = 1e-6
"""M1 fix (PR #546 review): a FLOOR, not an exact-zero check -- the
do-not-repeat file's own recorded failure mode for a different port
class reads "weak/zero coupling with uf_inc~6e-14"; a floor of 1e-6 sits
many orders of magnitude above that recorded noise floor while staying
far below any genuinely-coupled signal for excite_amp=1.0. Documented,
not measured from a real run (no local openEMS to calibrate against --
external scripts get no rfx preflight, external_solver_comparator.md)."""


def _check_excitation_and_trace(port, sim_path: str, label: str, *,
                                channel: str = "uf_inc") -> tuple[float, int]:
    """Hand-ported sanity checks (a) + (b): nonzero energy + nonzero trace.

    ``channel`` selects WHICH of the port's own two wave components is
    the one THIS DRIVE actually launched (M1 fix, PR #546 review): for
    the forward drive (port1) that is ``uf_inc`` (+z, Coax.m's own
    convention, the default here); for the reverse drive (port2) it is
    ``uf_ref`` (-z -- see ``_extract_two_port_s``'s docstring: "port 2's
    own launched wave into the device is its -z component, uf_ref_self
    (NOT uf_inc_self)"). Checking ``uf_inc`` unconditionally, as before,
    silently verified the WRONG channel for the reverse drive.

    Works generically across port classes: ``CoaxialPort.U_filenames``
    holds 3 entries (its 3-plane differential probe), ``AddLumpedPort``
    held 1 -- this function just iterates whatever list is there.
    """
    launched = np.asarray(getattr(port, channel), dtype=np.complex128)
    inc_peak = float(np.max(np.abs(launched))) if launched.size else 0.0
    if not np.isfinite(inc_peak) or inc_peak < _EXCITATION_ENERGY_FLOOR:
        raise RuntimeError(
            f"[{label}] openEMS port injected/received NO (or near-zero) "
            f"wave energy on its own launched channel ({channel}="
            f"{inc_peak!r} < floor {_EXCITATION_ENERGY_FLOOR!r}): "
            f"excitation did not couple or the port never saw the wave."
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


def _central_band_idx(n: int) -> slice:
    """The central 50% of an n-point frequency sweep, skipping bin 0.

    M1' fix (round-2 review): Z_ref/beta (and anything derived from them,
    including the passivity witness) are ill-conditioned at DC and at a
    sweep's own edges -- Coax.m's own ``linspace(0, 2*f0, 201)`` STARTS
    at f=0 exactly. This slice is now shared by the ZL gate, Stage A's
    passivity witness, AND the matched-through witness (previously only
    the latter excluded these bins).
    """
    lo, hi = int(0.25 * n), int(0.75 * n)
    return slice(max(lo, 1), max(hi, lo + 1))


def _passivity_witness(s11: np.ndarray, s21: np.ndarray, label: str, *,
                       tol: float = 0.05, idx: slice | None = None) -> dict:
    """B5(a): per-bin energy balance |S11|^2+|S21|^2 <= 1+tol, HARD gate.

    Only the upper bound is enforced (any passive structure must satisfy
    it); the lower bound (how far below 1, i.e. how lossy) is recorded
    but NOT gated -- the true loss of either fixture is not known a
    priori, so gating a lower bound would be an unverified assumption.

    ``idx`` (M1' fix): restrict the gate to a caller-supplied frequency
    sub-band (typically ``_central_band_idx(n)``) -- Stage A's own sweep
    includes f=0 and its own edges, where Z_ref/beta (and anything that
    depends on a well-conditioned wave decomposition) are noisy; Stage
    B's sweep has no DC point and passes ``idx=None`` (full band).
    """
    balance_full = np.abs(s11) ** 2 + np.abs(s21) ** 2
    balance = balance_full[idx] if idx is not None else balance_full
    max_balance = float(np.max(balance))
    passed = bool(max_balance <= 1.0 + tol)
    if not passed:
        raise RuntimeError(
            f"[{label}] passivity witness failed: max(|S11|^2+|S21|^2)="
            f"{max_balance:.4f} > {1.0 + tol} -- non-physical energy gain."
        )
    return {
        "balance": balance.tolist(), "max_balance": max_balance, "tol": tol,
        "band_restricted": idx is not None, "passed": passed,
    }


def _matched_through_witness(freqs_hz: np.ndarray, s21: np.ndarray, *, L_m: float,
                             eps_r: float, mag_band: tuple[float, float], label: str) -> dict:
    """B5/B5': one-sided matched-through expectation (Stage A AND Stage B).

    |S21|~=1 (or, for Stage B's lossy fixture, within ``mag_band``),
    phase~=-beta*L, group delay~=L*sqrt(eps_r)/c over the SAME central
    frequency sub-band ``_passivity_witness``/the ZL gate now use (M1'
    fix -- away from f=0 and the sweep's own edges, where a Gaussian-
    pulse spectrum is weak and DFT noise dominates). ONE-SIDED per the
    review: compared against an INDEPENDENTLY-known expected value, not
    against another internally-derived quantity (e.g. S12) -- catches a
    systematic channel-convention error that a symmetric reciprocity
    check could miss if it affected both drives identically.

    ``mag_band`` is caller-supplied (not read from ``REPRODUCE_GATE_
    RECORD`` unconditionally) so Stage A (ideal lossless Coax.m line,
    [0.5, 1.3]) and Stage B (rfx's own lossy fixture, [0.5, 1.1] --
    B5' fix) can each use their own physically-appropriate band.
    """
    idx = _central_band_idx(freqs_hz.size)

    s21_mag = np.abs(s21)
    s21_band = s21_mag[idx]
    mag_lo, mag_hi = mag_band
    mag_ok = bool(np.all((s21_band >= mag_lo) & (s21_band <= mag_hi)))

    beta = 2.0 * np.pi * freqs_hz * np.sqrt(eps_r) / _C0
    expected_phase = -beta * L_m
    measured_phase = np.unwrap(np.angle(s21))
    # Wrap the DIFFERENCE to (-pi, pi] purely to avoid a spurious 2*pi*n
    # branch-cut artefact from np.unwrap's own arbitrary starting branch.
    # This does NOT make the check tolerant of an arbitrary constant
    # phase offset (LOW fix -- the previous comment overclaimed
    # "slope/relative behavior only"): it compares measured vs -beta*L at
    # EVERY frequency point, so it pins the ABSOLUTE phase reference too,
    # not just its rate of change with frequency -- CalcPort's own
    # uf_inc/uf_ref split ties phase directly to the physical excitation,
    # with no free additive constant to calibrate away.
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


def _run_stage_a_reproduce_gate(*, sim_root: str, threads: int, use_pml: bool) -> dict:
    """H1' fix (round-2 review, BLOCKING): Stage A's NrTS/EndCriteria are
    ALWAYS Coax.m's own A_NUM_TS/A_END_CRITERIA -- they no longer accept
    caller-supplied ``nrts``/``end_criteria`` at all (the round-1 bug was
    that this function silently took the Stage-B CLI values, 200000/1e-4,
    instead of the tutorial's own 5000/1e-5, against MUR boundaries where
    run length controls reflected-energy accumulation). ``--nrts``/
    ``--end-criteria`` on the command line now govern Stage B only.
    """
    ContinuousStructure, openEMS, CoaxialPort = _import_openems()

    sim_dir = os.path.join(sim_root, "stage_a_coax_tutorial")
    smoke_dir = os.path.join(sim_root, "stage_a_coax_tutorial_smoke")

    smoke_fdtd, _sp1, _sp2 = _build_stage_a_coax_tutorial(
        ContinuousStructure, openEMS, CoaxialPort,
        nrts=min(200, A_NUM_TS), end_criteria=0.0, use_pml=use_pml)
    smoke_log = _run_openems_capturing_stdout(smoke_fdtd, smoke_dir, threads=threads)
    _scan_stdout_for_bad_patterns(smoke_log, "stage_a_smoke")

    fdtd, port1, port2 = _build_stage_a_coax_tutorial(
        ContinuousStructure, openEMS, CoaxialPort,
        nrts=A_NUM_TS, end_criteria=A_END_CRITERIA, use_pml=use_pml)

    t0 = time.time()
    real_log = _run_openems_capturing_stdout(fdtd, sim_dir, threads=threads)
    _scan_stdout_for_bad_patterns(real_log, "stage_a")  # belt-and-suspenders
    elapsed = time.time() - t0

    port1.CalcPort(sim_dir, A_FREQS_HZ)
    port2.CalcPort(sim_dir, A_FREQS_HZ)

    inc_peak, n_samples = _check_excitation_and_trace(port1, sim_dir, "stage_a")
    truncated = bool(A_NUM_TS > 0 and n_samples >= A_NUM_TS)

    # M1' fix: ZL gate + passivity witness restricted to the SAME central
    # band the matched-through witness already used -- Coax.m's own sweep
    # starts at f=0 exactly, where Z_ref/beta are ill-conditioned.
    idx = _central_band_idx(A_FREQS_HZ.size)

    zl = np.asarray(port1.Z_ref, dtype=np.complex128)
    zl_dev_full = np.abs(zl.real - ZL_A_ANALYTIC_OHM)
    zl_dev = zl_dev_full[idx]
    zl_re_ok = bool(np.max(zl_dev) < REPRODUCE_GATE_RECORD["gate"]["zl_re_tol_ohm"])
    zl_im_ok = bool(np.max(np.abs(zl.imag[idx])) < REPRODUCE_GATE_RECORD["gate"]["zl_im_tol_ohm"])

    s11 = np.asarray(port1.uf_ref, dtype=np.complex128) / np.asarray(port1.uf_inc, dtype=np.complex128)
    s21 = np.asarray(port2.uf_inc, dtype=np.complex128) / np.asarray(port1.uf_inc, dtype=np.complex128)
    _non_physical_guard(np.abs(s11), "stage_a_s11")
    _non_physical_guard(np.abs(s21), "stage_a_s21")

    passivity = _passivity_witness(s11, s21, "stage_a", idx=idx)
    mag_lo, mag_hi = REPRODUCE_GATE_RECORD["gate"]["s21_thru_band"]
    matched_thru = _matched_through_witness(
        A_FREQS_HZ, s21, L_m=A_L_THRU_MM * 1e-3, eps_r=1.0,
        mag_band=(mag_lo, mag_hi), label="stage_a")

    passed = bool(zl_re_ok and zl_im_ok and not truncated
                 and passivity["passed"] and matched_thru["passed"])

    return {
        "zl_real_ohm": zl.real.tolist(), "zl_imag_ohm": zl.imag.tolist(),
        "zl_mean_ohm": float(np.mean(zl.real[idx])), "zl_max_dev_ohm": float(np.max(zl_dev)),
        "zl_re_ok": zl_re_ok, "zl_im_ok": zl_im_ok,
        "s11_mag": np.abs(s11).tolist(), "s21_mag": np.abs(s21).tolist(),
        "passivity": passivity, "matched_through_witness": matched_thru,
        "max_uf_inc": inc_peak, "n_trace_samples": n_samples, "nrts_cap": A_NUM_TS,
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
    fdtd.SetBoundaryCond(["PML_16"] * 6)  # H2 restored: PML_16 on all six faces
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
    # z lines: domain edges, the shared split, and every feed/measure/
    # target plane -- so CoaxialPort's own measurement-plane mesh-line
    # snap (mesh.GetLines(prop_ny) inside its constructor) lands cleanly.
    fixed_z = sorted({
        layout["z_port1_start_mm"], layout["z_port2_stop_mm"], layout["z_split_mm"],
        layout["z_feed_bot_mm"], layout["z_feed_top_mm"],
        layout["z_port1_start_mm"] + layout["measplane_port1_mm"],
        layout["z_port1_start_mm"] + layout["feedshift_port1_mm"],
        layout["z_port2_start_mm"] + layout["measplane_port2_mm"],
        layout["z_port2_start_mm"] + layout["feedshift_port2_mm"],
    })
    mesh.AddLine("x", fixed_xy)
    mesh.AddLine("y", fixed_xy)
    mesh.AddLine("z", fixed_z)
    mesh.SmoothMeshLines("all", dx_mm, 1.4)

    copper = csx.AddMetal("copper")
    ptfe = csx.AddMaterial("ptfe", epsilon=B_PTFE_EPS_R)

    excite1 = 1.0 if drive == "port1" else 0.0
    excite2 = 1.0 if drive == "port2" else 0.0

    # B4' fix: port 2 now start=z_split, stop=z_port2_stop_mm (domain
    # ceiling) -- direction=+1, SAME as port 1, matching Coax.m's own
    # cascaded-segments layout (module docstring "PORT CLASS" / B4' has
    # the full derivation of why the previous mirror-symmetric,
    # direction=-1 port 2 broke CoaxialPort's direction-signed current
    # probe). H2 fix: both ports' outer ends reach the domain's own edges,
    # not rfx's retracted extent, INTO the PML -- restored 2026-08-03
    # after a diagnosis reversal (module docstring "LINE TERMINATION
    # TOPOLOGY": a MUR z-boundary was tried and reverted -- MUR-on-PTFE
    # is a RECORDED-unstable combination in this repo's own history).
    port1 = CoaxialPort(
        csx, port_nr=1, pec_prop=copper, mat_prop=ptfe,
        start=[cx_mm, cy_mm, layout["z_port1_start_mm"]],
        stop=[cx_mm, cy_mm, layout["z_port1_stop_mm"]],
        prop_dir="z", r_i=B_A_MM, r_o=B_B_MM, r_os=r_os_mm,
        excite_amp=excite1, FeedShift=layout["feedshift_port1_mm"],
        MeasPlaneShift=layout["measplane_port1_mm"], priority=10,
    )
    port2 = CoaxialPort(
        csx, port_nr=2, pec_prop=copper, mat_prop=ptfe,
        start=[cx_mm, cy_mm, layout["z_port2_start_mm"]],
        stop=[cx_mm, cy_mm, layout["z_port2_stop_mm"]],
        prop_dir="z", r_i=B_A_MM, r_o=B_B_MM, r_os=r_os_mm,
        excite_amp=excite2, FeedShift=layout["feedshift_port2_mm"],
        MeasPlaneShift=layout["measplane_port2_mm"], priority=10,
    )
    return fdtd, port1, port2, layout


def _extract_two_port_s(port_self, port_thru, *, reverse_drive: bool) -> dict:
    """S_self/S_thru for one drive.

    Both Stage B ports share direction=+1 (B4' fix, module docstring
    "PORT CLASS"): ``uf_inc_k`` ALWAYS tracks the +z-travelling component
    AT PORT k, ``uf_ref_k`` the -z component -- a purely local, geometry-
    based split that does not by itself know which port is driven. What
    flips between drives is which physical direction is "into the
    device" FROM the driven port's own location -- and BOTH the
    numerator (thru) and the denominator (self) depend on that, not just
    the numerator:

      - FORWARD drive (port 1, bottom): +z IS into the device (toward
        port 2), so port 1's own launched wave is its +z component,
        ``uf_inc_self``. ``s_self`` (S11) = ``uf_ref_self/uf_inc_self``
        (Coax.m's own formula: what bounces back / what was launched).
        ``s_thru`` (S21) = ``uf_inc_thru/uf_inc_self`` -- the wave
        arrives at port 2 travelling +z, ALIGNED with port 2's own
        convention.
      - REVERSE drive (port 2, top): +z is AWAY from the device (toward
        the top PML) -- port 2's own launched wave into the device is
        its -z component, ``uf_ref_self`` (NOT ``uf_inc_self``).
        ``s_self`` (S22) = ``uf_inc_self/uf_ref_self`` (what bounces
        back -- +z at port 2 -- over what was actually launched -- -z at
        port 2). ``s_thru`` (S12) = ``uf_ref_thru/uf_ref_self`` -- the
        wave arrives at port 1 travelling -z, OPPOSED to port 1's own
        convention (``uf_ref_thru``), and the DENOMINATOR is port 2's
        own launched wave, ``uf_ref_self``, not ``uf_inc_self``.

    Fixed 2026-08-03 (round-3 review of PR #540): the previous version
    flipped ONLY the ``s_thru`` numerator for the reverse drive
    (``thru_uses_ref``) and left ``s_self`` as ``uf_ref_self/
    uf_inc_self`` unconditionally -- WRONG for the reverse drive, where
    ``uf_inc_self`` is not the launched wave. The reviewer's synthetic
    check (plain complex arithmetic on fake ``uf_inc``/``uf_ref`` values
    -- no openEMS needed, see ``tests/test_coax_two_port_referee_header.
    py::test_extract_two_port_s_synthetic_forward_and_reverse_drive``)
    caught it: the old code reported S22 as 1/S22_true (0.051 -> 19.61)
    and S12 scaled by the SAME inverted factor (0.9 -> 17.65) -- both
    errors traced to the same wrong denominator, not two separate bugs.

    CHANNEL-INVERSION SIGNATURE (for a future reviewer reading a failed
    run): if this split is ever wrong again, the SELF ratio (S11 or S22)
    blows up to roughly 1/(true value) -- a passivity violation or
    non-physical-field-guard failure on the SELF ratio, at
    ~1/S-magnitude scale, IS the tell. Check this split first, before
    doubting the fixture geometry or the FDTD run itself.

    Both channels are always recorded (``s_thru`` / ``s_thru_alternate_
    channel``) so a reviewer can recover the other reading without
    rerunning, regardless of which one this function selected.
    """
    uf_inc_self = np.asarray(port_self.uf_inc, dtype=np.complex128)
    uf_ref_self = np.asarray(port_self.uf_ref, dtype=np.complex128)
    uf_inc_thru = np.asarray(port_thru.uf_inc, dtype=np.complex128)
    uf_ref_thru = np.asarray(port_thru.uf_ref, dtype=np.complex128)

    if reverse_drive:
        denom_self = uf_ref_self          # port 2's own launched wave (-z)
        numer_self = uf_inc_self          # what bounces back (+z at port 2)
        numer_thru = uf_ref_thru          # arrives at port 1 travelling -z
        alternate_thru = uf_inc_thru
    else:
        denom_self = uf_inc_self          # port 1's own launched wave (+z)
        numer_self = uf_ref_self          # what bounces back (-z at port 1)
        numer_thru = uf_inc_thru          # arrives at port 2 travelling +z
        alternate_thru = uf_ref_thru

    return {
        "s_self": numer_self / denom_self,
        "s_thru": numer_thru / denom_self,
        "s_thru_alternate_channel": alternate_thru / denom_self,
        "reverse_drive": reverse_drive,
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

    # M1 fix (PR #546 review): check the DRIVE-CORRECT launched channel --
    # port1 (forward drive) launches on uf_inc, port2 (reverse drive) on
    # uf_ref (_extract_two_port_s's own documented convention).
    driven_port = port1 if drive == "port1" else port2
    driven_channel = "uf_inc" if drive == "port1" else "uf_ref"
    inc_peak, n_samples = _check_excitation_and_trace(
        driven_port, sim_dir, f"stage_b_drive_{drive}", channel=driven_channel)
    truncated = bool(nrts > 0 and n_samples >= nrts)

    # Minor fix (PR #546 review): make openEMS's own "end-criteria not
    # reached" warning a machine-readable, per-drive fact -- recorded
    # data feeding truncated_suspected, reaching disk now that B2's
    # serialization fix makes drive{1,2}_diagnostics safely JSON-able.
    end_criteria_not_reached = "reached before the end-criteria of" in real_log

    # reverse_drive: see _extract_two_port_s's docstring. Driving port1
    # (bottom) is the FORWARD drive (both numerator and denominator use
    # the +z/uf_inc convention); driving port2 (top) is the REVERSE drive
    # (both numerator and denominator flip to the -z/uf_ref convention,
    # since +z at port2 points AWAY from the device).
    if drive == "port1":
        extracted = _extract_two_port_s(port1, port2, reverse_drive=False)
    else:
        extracted = _extract_two_port_s(port2, port1, reverse_drive=True)

    return {
        "extracted": extracted,
        "z0_port1": np.asarray(getattr(port1, "Z_ref", np.nan), dtype=np.complex128),
        "z0_port2": np.asarray(getattr(port2, "Z_ref", np.nan), dtype=np.complex128),
        "beta_port1": np.asarray(getattr(port1, "beta", np.nan), dtype=np.complex128),
        "beta_port2": np.asarray(getattr(port2, "beta", np.nan), dtype=np.complex128),
        "max_uf_inc": inc_peak, "n_trace_samples": n_samples, "nrts_cap": nrts,
        "truncated_suspected": truncated, "elapsed_s": round(elapsed, 1),
        "end_criteria_not_reached": end_criteria_not_reached,
    }


def _group_delay(freqs_hz: np.ndarray, s21: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phase = np.unwrap(np.angle(s21))
    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64)
    gd_s = -np.gradient(phase, omega)
    return phase, gd_s


def _json_safe_diagnostics(drive: dict) -> dict:
    """JSON-safe copy of a per-drive diagnostics dict (everything
    ``_run_one_drive`` returns except ``extracted``).

    B2 fix (PR #546 review, BLOCKING): ``z0_port{1,2}``/``beta_port{1,2}``
    are complex128 ndarrays -- ``json.dump`` cannot serialize a complex
    number or a raw ndarray, and raises ``TypeError`` partway through
    writing the file. This crashed the NEW failure-path writer
    (``partial_data``, below) outright (reviewer-reproduced), and was
    LATENT on the pre-existing SUCCESS path's own ``drive{1,2}_
    diagnostics`` fields too -- never yet exercised, since Stage B has
    never completed a real run either way. Convert complex arrays to
    [re, im] pair lists, matching the convention already used for
    s11/s21/s12/s22 elsewhere in this module.
    """
    out = {}
    for key, value in drive.items():
        if key == "extracted":
            continue
        if isinstance(value, np.ndarray) and np.iscomplexobj(value):
            out[key] = [[float(c.real), float(c.imag)] for c in value]
        else:
            out[key] = value
    return out


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

    # Forensics fix (2026-08-03, run-1 diagnosis): the FIRST live run
    # (VESSL 369367251366) raised out of the guard loop below with only a
    # single scalar (max|S|) ever reaching disk -- main()'s RuntimeError
    # handler writes a JSON artifact, but it only had ``str(exc)`` to put
    # in it, so the per-bin S-matrices, per-drive diagnostics (max_uf_inc,
    # truncation flags, elapsed time) were never recorded anywhere,
    # forcing this diagnosis to work from run.log's printed warnings
    # alone (R5: a scalar verdict is not enough to inspect a result).
    # Snapshot everything computed so far BEFORE any guard can raise, and
    # attach it to the exception so a failing run's JSON still carries
    # the full per-bin picture for the next reviewer.
    partial_data = {
        "freqs_ghz": B_FREQS_GHZ.tolist(),
        "s11": [[float(c.real), float(c.imag)] for c in s11],
        "s21": [[float(c.real), float(c.imag)] for c in s21],
        "s12": [[float(c.real), float(c.imag)] for c in s12],
        "s22": [[float(c.real), float(c.imag)] for c in s22],
        "s11_mag": np.abs(s11).tolist(), "s21_mag": np.abs(s21).tolist(),
        "s12_mag": np.abs(s12).tolist(), "s22_mag": np.abs(s22).tolist(),
        "drive1_diagnostics": _json_safe_diagnostics(drive1),
        "drive2_diagnostics": _json_safe_diagnostics(drive2),
    }

    try:
        for label, arr in (("s11", s11), ("s21", s21), ("s12", s12), ("s22", s22)):
            _non_physical_guard(np.abs(arr), label)

        passivity_drive1 = _passivity_witness(s11, s21, "stage_b_drive1")
        passivity_drive2 = _passivity_witness(s22, s12, "stage_b_drive2")

        # B5' fix (round-2 review, BLOCKING): the SAME one-sided matched-
        # through witness Stage A uses, re-parameterized for Stage B's own
        # (lossy, rfx-matched) fixture -- wired into sanity_passed, not just
        # recorded. A channel-inversion bug in _extract_two_port_s reads
        # |S21|~=0 here against an expected ~[0.5,1.1], failing one-sidedly
        # rather than agreeing with an equally-wrong S12.
        #
        # Round-3 fix: BOTH drives now get this witness, not just the
        # forward one -- s21 alone left the REVERSE drive's own channel
        # selection (denom_self=uf_ref_self, round-3 fix above) covered only
        # by the passivity ceiling, which a 1/S-scale channel-inversion can
        # still satisfy in some regimes. s12 must land in the SAME
        # [0.5, 1.1] band with the SAME ~282.57 ps group delay BY
        # RECIPROCITY (a reciprocal 2-port has S12=S21) -- this is the
        # one-sided witness for the reverse leg specifically.
        matched_thru_s21 = _matched_through_witness(
            B_FREQS_HZ, s21, L_m=B_L12_MM * 1e-3, eps_r=B_PTFE_EPS_R,
            mag_band=B_S21_THRU_BAND, label="stage_b_s21")
        matched_thru_s12 = _matched_through_witness(
            B_FREQS_HZ, s12, L_m=B_L12_MM * 1e-3, eps_r=B_PTFE_EPS_R,
            mag_band=B_S21_THRU_BAND, label="stage_b_s12")
    except RuntimeError as exc:
        # Forensics fix (see partial_data comment above): a guard/witness
        # failure here must not lose the numbers that led to it.
        exc.partial_stage_b_data = partial_data
        raise

    recip_mag_dev = float(np.max(np.abs(np.abs(s21) - np.abs(s12))))
    recip_phase_dev_deg = float(np.max(np.abs(np.degrees(np.angle(s21) - np.angle(s12)))))

    phase21, gd21_s = _group_delay(B_FREQS_HZ, s21)

    sanity_passed = bool(
        not drive1["truncated_suspected"] and not drive2["truncated_suspected"]
        and passivity_drive1["passed"] and passivity_drive2["passed"]
        and matched_thru_s21["passed"] and matched_thru_s12["passed"]
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
        "matched_through_witness": matched_thru_s21,
        "matched_through_witness_s12": matched_thru_s12,
        "z0_port1_drive1": [[float(c.real), float(c.imag)] for c in drive1["z0_port1"]],
        "beta_port1_drive1": [[float(c.real), float(c.imag)] for c in drive1["beta_port1"]],
        "alternate_channel_s21": [[float(c.real), float(c.imag)]
                                  for c in drive1["extracted"]["s_thru_alternate_channel"]],
        "drive1_diagnostics": _json_safe_diagnostics(drive1),
        "drive2_diagnostics": _json_safe_diagnostics(drive2),
        "sanity_passed": sanity_passed,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=".omx/coax_two_port_referee/openems_coax_two_port.json")
    p.add_argument("--sim-root", default="/tmp/openems_coax_two_port_referee")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--nrts", type=int, default=200000,
                   help="Stage B only (H1' fix) -- Stage A always uses its own "
                        "Coax.m-derived A_NUM_TS, never this value")
    p.add_argument("--end-criteria", type=float, default=1e-4,
                   help="Stage B only (H1' fix) -- Stage A always uses its own "
                        "Coax.m-derived A_END_CRITERIA, never this value")
    p.add_argument("--use-pml", action="store_true",
                   help="Stage A: use PML_8 instead of Coax.m's default MUR z boundary")
    p.add_argument("--skip-stage-b", action="store_true",
                   help="run only the Stage A reproduce-gate (fast smoke check)")
    args = p.parse_args(argv)

    # Forensics fix (2026-08-03, run-1 diagnosis): resolve the output path
    # ONCE, here, before any openEMS call. The first live run's JSON never
    # reached .omx/coax-two-port-referee/20260803T071116Z/ despite exiting
    # via the RuntimeError handler below, which calls
    # os.path.abspath(args.output) -- a RELATIVE path, resolved against
    # whatever the process's cwd happens to be AT THAT LATER POINT, after
    # several openEMS ``fdtd.Run(sim_path, cleanup=True, ...)`` calls
    # (openEMS's own Python bindings are not inspectable here -- VESSL-
    # only, not installed locally -- so whether/how ``Run()`` touches cwd
    # is UNCONFIRMED, not a verified root cause). Resolving the absolute
    # path up front removes the dependency on cwd staying stable across
    # the whole run either way, at zero cost.
    out_path = os.path.abspath(args.output)

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
            sim_root=sim_root, threads=args.threads, use_pml=args.use_pml,
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
            print(f"  matched-through witness (s21) passed: "
                  f"{stage_b['matched_through_witness']['passed']}")
            print(f"  matched-through witness (s12) passed: "
                  f"{stage_b['matched_through_witness_s12']['passed']}")
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
        # Forensics fix: carry whatever Stage B had already computed
        # (per-bin S-matrices, per-drive diagnostics) into the failure
        # artifact, if _run_stage_b attached any (see its partial_data
        # comment) -- a failing run must still leave the numbers a
        # reviewer needs, not just the one scalar that tripped the gate.
        partial = getattr(exc, "partial_stage_b_data", None)
        if partial is not None:
            artifact["stage_b_partial"] = partial
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
