"""Cross-validation 06b: MSL Notch Filter — uniform mesh + add_msl_port.

This began as a sibling to the RETIRED ``06_msl_notch_filter.py`` (non-uniform
mesh + wire ``add_port`` + graded-σ absorber + Z_probe=1kΩ workaround; removed
as artifact-anchored — issue #339, content in git history). 06b
shows the same notch-filter physics using the **distributed MSL port**
(``add_msl_port``) on a uniform mesh, with no graded-σ absorber. Under the
physics-first evidence taxonomy this is E2-promising: it uses an analytic
quarter-wave notch and internal MSL gates, but broad E5 claims still require
raw V/I replay and an external cross-solver envelope.

Rationale:
  - Wire ``add_port(extent=...)`` covers ONE cell transverse to the trace,
    missing ~3/4 of the quasi-TEM mode's lateral extent. Partial reflection
    at both ports sets up a Fabry-Perot comb that masks the stub notch.
  - ``add_msl_port`` covers the FULL trace cross-section with a Laplace-Ez
    source distribution + distributed-σ matched termination + 3-probe
    de-embedding. F-P ripple is reduced to the current narrow-envelope
    floor (|S11|≈0.10 = -20dB).

External cross-check (openEMS, 2026-07-05):
  A matched-geometry rfx-vs-openEMS comparison is committed under
  ``tests/fixtures/msl_notch_e4/`` (gate:
  ``tests/crossval/test_msl_notch_e4_comparison_gates.py``). At a CONVERGED dx=50µm mesh
  (5.08 substrate cells) where both solvers are passive, the off-notch |S21|
  transmission agrees to ~0.1, while the notch frequency agrees to ~6% (rfx
  3.63 GHz vs openEMS 3.43 GHz; fringing-free analytic 3.69 GHz).
  UPDATE (Palace FEM referee, 2026-07-07): an independent
  conformal-tet FEM run on the matched geometry lands at ~3.631 GHz at two mesh
  densities — closest to rfx (+0.1%) — see
  ``tests/fixtures/msl_notch_e4/msl_stub_notch_palace_referee.json``. Our earlier
  working interpretation (open-end fringing as the driver of the split) is
  revised: the FEM value indicates the fringing correction is ~1-2%. The
  "err<15% vs analytic" gate below is rfx-vs-ANALYTIC and is NOT an
  OpenEMS-class number.

  The paragraph above was written when this script ran at dx=80µm, where
  its own preflight called the mesh UNDER-RESOLVED (h_sub/dx=3.175, a
  mixed-cell substrate) and no external comparison at that mesh was
  valid. It now ships at dx=63.5µm — see "Mesh convention" below.

  CAVEAT this fix does NOT resolve: the E4 fixture
  (``tests/fixtures/msl_notch_e4/``) was produced by
  ``scripts/diagnostics/build_msl_notch_rfx_dx50.py`` at DX=50µm, where
  h_sub/dx=5.08 is itself off-lattice and realizes h_sub=300µm (not
  254µm) — that producer carries the SAME #722/#723 defect this script
  just fixed and is EXPLICITLY DEFERRED (see its own docstring), not
  fixed here. So as of this change the E4/Palace comparison above and
  this script solve DIFFERENT boards: every figure in this paragraph
  ("rfx 3.63 GHz", "openEMS 3.43 GHz", the Palace 3.631 GHz referee)
  belongs to a 300µm board, while this script now solves 254µm. Do not
  read them as a cross-check on this script's own output until that
  producer is fixed.

Mesh convention (issue #723, 2026-08-27):
  DX = H_SUB / 4 = 63.5µm (was 80µm, h_sub/dx=3.175 — the "External
  cross-check" section above already called that mesh UNDER-RESOLVED for
  external-class work). gcd(254, 600) = 2µm, so no single cubic cell size
  makes both H_SUB and W_TRACE land exactly on the lattice at any sane
  cost (63.5µm: 600/63.5=9.449 cells). This script resolves that by
  REALIZING H_SUB exactly (the dimension the port-resolution preflight
  checks and the Z0-bias sweep below both key off) and QUOTING THE
  REALIZED W_TRACE in the analytic reference, rather than re-declaring
  W_TRACE on a lattice multiple. ``_realized_trace_width()`` reads that
  value live from ``sim.fidelity_report()`` after ``_build_sim()`` — NOT
  a ``round(W_TRACE / DX) * DX`` re-derivation, which gives the WRONG
  answer here (571.5µm / 9 cells) because the half-open ``[lo, hi)`` node
  rasterization (``rfx.geometry.csg.Box``) counts the OVERLAPPED node
  span, not the declared extent rounded to the nearest cell.

  Verified via ``sim.fidelity_report()`` at DX=H_SUB/4 (this run, quoted
  verbatim):
    "geometry[0] 'ro4350b' ... z: declared [0.0, 254.0] um -> realized
    [0.0, 254.0] um | face residuals (0.0, 0.0) um | extent 254.0 ->
    254.0 um" — substrate thickness now EXACT (was +25.98% at dx=80µm).
    "geometry[1] 'pec' ... y: declared [1016.0, 1616.0] um -> realized
    [1016.0, 1651.0] um | face residuals (0.0, 35.0) um | extent 600.0 ->
    635.0 um" and "geometry[2] 'pec' ... x: ... extent 600.0 -> 635.0 um"
    — main trace and stub realize the SAME 635.0µm width (10 cells), so
    ``u = W_realized / H_SUB`` = 2.500 describes one consistent board
    (this was NOT true at dx=80µm: 560µm on the trace vs 640µm on the
    stub). Total: 5 entities, 9 findings (was 12 at dx=80µm) — the two
    retired findings are BOTH MSL-port substrate-resolution warnings
    (below); the off-lattice conductor-edge warning is NOT retired (see
    "Preflight honesty" below).

  Reference-formula effect: u 2.362 (declared) -> 2.500 (realized) moves
  ε_eff_HJ 2.869 -> 2.882 and the analytic notch 3.6872 -> 3.6790 GHz
  (-0.22%) — small next to the current 15% gate.

  Z0 anchor: ``scripts/diagnostics/msl_z0_bias_floor_sweep.py`` runs the
  SAME board cross-section (W=600/h=254 RO4350B) on a DIFFERENT line — a
  10mm thru with no stub — through a predeclared dx grid that includes
  this exact mesh, which is what makes it an independent cross-check
  rather than a re-run. Committed row (``msl_z0_bias_floor_sweep/
  msl_z0_bias_floor_sweep.json``, label "aligned h_sub/4"): dx_um=63.5,
  z0_measured_ohm=46.098, z0_hj_ohm=47.895 (HJ on the DECLARED 600/254
  board). ``rfx.sources.msl_eigenmode.hammerstad_jensen_z0_eps_eff``:
  HJ(635µm, 254µm, 3.66) = 46.18 Ω, 0.18% from the measured 46.10 Ω, vs
  HJ(600µm, 254µm, 3.66) = 47.90 Ω, 3.9% away — the realized width, not
  the declared one, is the analytic anchor that matches what this mesh
  measures. READ THAT NARROWLY: it says the extractor tracks HJ on the
  board it actually solves. It does NOT say this mesh extracts Z0 better
  than the old one — the same recheck run on all six sweep points puts
  every one of them within 0.38% of HJ on ITS realized board, dx=80µm
  included (filed as #752). Predicted post-fix median Re(Z0) ≈ 46.1 Ω,
  15.2% above the (40, 65) Ω gate floor below (window unchanged; NOT
  re-pinned by this change — that needs a fresh solve, see "Runtime"
  below).

  Runtime and the re-pinned envelope: measured grid shape
  (``sim._build_grid().shape``) dx=80µm -> (442, 232, 31) = 3,178,864
  cells; dx=63.5µm -> (553, 280, 37) = 5,729,080 cells (1.802x). Combined
  with the ~1.260x more timesteps (dt ∝ dx), the cell x timestep product
  scales 2.271x. TREAT THAT AS A LOWER BOUND, NOT A FORECAST: the
  dx=63.5µm CPU attempt was abandoned unfinished at 2h52m on a 32-core
  pod, i.e. past 2.271x ANY of the dx=80µm CPU baselines measured here
  (2599.6s on 2026-08-09; 1621.2s on 2026-08-28, different machines), so
  the linear model underpredicts on this lane. Use the GPU.

  MEASURED. Post-fix column: 2026-08-27, VESSL 369367256574, remilab-c0
  single RTX4090, log ``_06b_notch_uniform_logs/20260827T131217Z_run.log``,
  solve 329.2s, exit 0. Pre-fix column: RE-MEASURED 2026-08-28 on
  origin/main (cdc38bc8) rather than quoted from the committed 2026-08-09
  log, because that log predates #682, #698 and #699 — all three touch the
  MSL port or its extractor — so using it would have compared two code
  versions as well as two meshes. CPU, solve 1621.2s, exit 0, log
  ``_06b_notch_uniform_logs/20260828T054132Z_dx80_origin_main_cdc38bc8_
  run.log``. It reproduces the 2026-08-09 numbers to the printed digits,
  so the three MSL merges did not move this case:

                              dx=80µm      dx=63.5µm    gate
    Notch frequency (rfx)     3.627 GHz    3.627 GHz    --
    Notch frequency (analytic) 3.687 GHz   3.679 GHz    --
    Notch frequency error     1.63 %       1.40 %       < 15 %
    Notch depth |S21|         -34.2 dB     -43.3 dB     < -10 dB
    Re(Z0) median             57.9 Ω       46.5 Ω       (40, 65) Ω

  WHAT THE Z0 ROW MEANS. The mesh now realizes the board that was
  declared, so the reported Re(Z0) can be read against the DESIGN for the
  first time: 46.5 Ω vs HJ(600µm, 254µm) = 47.90 Ω, -2.9%. At dx=80µm no
  such reading existed — that mesh solved a 560µm/320µm board whose own
  HJ impedance is 57.46 Ω, i.e. the BOARD was 20% off the design, and
  57.9 Ω was a faithful measurement of the wrong board. The post-fix
  value also lands within 0.9% of ``msl_z0_bias_floor_sweep``'s committed
  "aligned h_sub/4" row (z0_measured_ohm = 46.098) — a different line
  (10mm thru, no stub), so that is an independent cross-check.

  RETRACTED (2026-08-28, #723 review BLOCKING 1). An earlier version of
  this block said: "Against Hammerstad-Jensen on the board each mesh
  actually solves, Re(Z0) goes from +20.9% (57.9 Ω vs HJ(600,254) =
  47.90 Ω) to +0.7% (46.5 Ω vs HJ(635,254) = 46.18 Ω) — a 30x reduction
  in the port-impedance bias." That applied its own stated rule to the
  post-fix column only: HJ(600,254) is NOT the board dx=80µm solves.
  Measured (``sim.fidelity_report()`` on this script's own ``_build_sim``,
  metadata only): dx=80µm realizes substrate 320.0µm and trace 560.0µm,
  and HJ(560µm, 320µm, 3.66) = 57.46 Ω — so the pre-fix measurement is
  +0.77%, against +0.69% post-fix. On the rule as written the two meshes
  are the same, and THERE IS NO PORT-ACCURACY IMPROVEMENT TO CLAIM here.
  The improvement is in BOARD FIDELITY, which is what #723 is about.
  The same recheck across all six committed ``msl_z0_bias_floor_sweep``
  points reads -0.38 / -0.18 / -0.24 / -0.13 / +0.20 / +0.13 % against HJ
  on each point's own realized board — filed as #752, not claimed here.
  Also retracted with it: the sentence crediting this script's own
  preflight prediction ("+20.2% vs -7.9% at ~3 cells ..."), which is the
  same declared-board comparison and cannot vindicate anything.
  ``tests/crossval/test_msl_notch_public_carriers.py::
  test_z0_anchor_is_the_design_board_not_a_realized_one`` pins the
  retraction so the 30x framing cannot return silently.

  THE NOTCH-FREQUENCY ROW IS BIN-LIMITED and must not be read as
  "unchanged". ``compute_msl_s_matrix(n_freqs=100)`` over the 7 GHz band
  gives 70.7 MHz bins = 1.95% at 3.627 GHz, so one bin is WIDER than the
  1.40% error being reported and both meshes' notches land in the same
  bin by construction. The improvement from 1.63% to 1.40% is the
  ANALYTIC reference moving (3.6872 -> 3.6790 GHz as u goes 2.362 ->
  2.500), not a measured shift in rfx's notch. Any future claim about
  this script's notch-frequency accuracy needs a finer sweep first.

  THE NOTCH DEPTH deepens ~9 dB (-34.2 -> -43.3). Board and mesh both
  changed in one step and no falsifier separates them, so that is
  recorded, not attributed.

  KNOWN LIMITATION, filed as #729 (NOT folded into #723): at every
  ALIGNED dx, ``add_msl_port``'s own cross-section audit
  (``rfx/sources/msl_port.py::msl_cross_section_span``) independently
  rasterizes the substrate height and overshoots it by one cell — the
  z_hi = h_sub face lands exactly on a node and ``Grid.position_to_index``
  (round-to-nearest) resolves to the cell above. Measured during the
  #723 review: n_z_sub=5 rows / 317.5µm at DX=H_SUB/4 (was n_z_sub=4 /
  320µm at the old dx=80µm, where it coincidentally matched the then
  26%-too-thick FDTD board), so the port's quasi-static Laplace mode
  model solves a ~317.5µm substrate with the trace strip at z=317.5µm
  while the FDTD board is exactly 254µm with the PEC wall at z=254µm
  (z0_static 53.29 -> 56.88 Ω across that same measurement). This is an
  rfx API rasterization behaviour, not a script-convention choice, and
  the extraction reads real FDTD V/I fields rather than the port's
  z0_static, so its effect is bounded by the committed
  msl_z0_bias_floor_sweep row above (gamma_implied=-0.019,
  mean_s11_raw=0.0223) — it does not block this fix, but needs its own
  issue against ``msl_cross_section_span``.

  Preflight honesty: this mesh retires BOTH MSL-port substrate-resolution
  warnings ("only 3 substrate cell(s) in z ... Refine to dx ≤ 64µm" and
  "h_sub/dx = 3.175 ... mixed-cell danger zone", both present at dx=80µm,
  both ABSENT at dx=63.5µm). It does NOT retire, and marginally WORSENS,
  the off-lattice conductor-edge warning — this run's own preflight,
  quoted verbatim: dx=80µm "geometry[1] 'pec' y: extent 600µm, worst face
  residual 28µm (4.67% of the extent)" -> dx=63.5µm "geometry[1] 'pec' y:
  extent 600µm, worst face residual 28.5µm (4.75% of the extent)" — the
  expected price of quoting the realized width instead of re-declaring
  W_TRACE on a lattice multiple.

  R5 DISCLOSURE — the rest of the 2026-08-27 run's own warnings, which
  qualify every number in the table above and are quoted verbatim rather
  than summarized (``_06b_notch_uniform_logs/20260827T131217Z_run.log``):

    "standing-wave null at the port plane: 9 bins in [3.6273, 7.0000]
    GHz have |V|,|I| below 10% of band median — wave-split S-parameters
    are unreliable there"

  That band STARTS at the reported notch (3.627 GHz), so the -43.3 dB
  depth is read at the edge of the flagged region. A deep notch IS a
  standing-wave null at the port plane, so this is expected rather than
  anomalous — but it means the depth is a demonstration that the notch
  resolves, not a calibrated magnitude.

    "compute_msl_s_matrix: reported Z0 for MSL port 'msl_0' = 61.02 ohm
    deviates 34.2% from analytic Hammerstad-Jensen 47.89 ohm at f =
    3.8818 GHz" / "... 'msl_1' = 39.90 ohm deviates 17.4% ..."

  Those are the argmax-over-bins deviations (+32% / -14% against
  HJ(635,254) = 46.18 Ω), not the median the gate reads. The gate reads
  ``np.median(res.Z0[0, :].real)`` — PORT 0 ONLY, median over all 100
  bins — so the 46.5 Ω headline neither covers port 1 nor bounds the
  per-bin spread.

    "S-matrix projected onto the passive set (singular values clipped to
    1): 63 of 100 frequency bins were non-passive as extracted, worst
    sigma_max = 1.006 at 3.627 GHz"

  The worst bin is the notch bin. 1.006 is inside the documented
  single-run Yee envelope, and the projection is recorded rather than
  silent, but the |S| values in the table are post-projection.

ESTIMATOR RESOLUTION (#812 mechanism P3, 2026-09-01) — APPENDED. Nothing
above is withdrawn except the one arithmetic slip corrected below.

  CORRECTION to the "THE NOTCH-FREQUENCY ROW IS BIN-LIMITED" paragraph above.
  It says ``compute_msl_s_matrix(n_freqs=100)`` "over the 7 GHz band gives
  70.7 MHz bins = 1.95% at 3.627 GHz". **That is wrong.** 70.7 MHz is
  7.0 GHz / 99, i.e. it assumes the sweep starts at DC. It does not: that
  entry point sweeps ``jnp.linspace(freq_max / 10, freq_max, n_freqs)``
  (``rfx/api/_sparams.py``, the ``freqs_arr`` line inside
  ``compute_msl_s_matrix``), so the sweep is 0.7 – 7.0 GHz and the bin is
  6.3 GHz / 99 = **63.6364 MHz = 1.754%** at 3.627 GHz. Confirmed against
  committed data: ``tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json``
  has ``freqs_ghz[0] = 0.7`` and a 0.0636364 GHz step. The paragraph's
  CONCLUSION is unaffected and still holds — one bin is still wider than the
  1.40% error being reported — but the width was overstated by 11%.

  THE DEPTH GATE COULD NOT FAIL. ``pass_notch_depth = s21_notch_db < -10``
  reads the *sampled* minimum of a true transmission zero, so it measures how
  close a bin happened to land, not the notch's quality. For an ideal shunt
  open stub, S21 = 2/(2 + j·r·tan(θ)) with θ = (π/2)(f/f0) and
  r = Z0_line/Z_stub; this board realizes the SAME 635.0µm width for stub and
  main line (see "Mesh convention" above), so r = 1 exactly by construction.
  The worst case is a bin half a bin off f0: θ = (π/2)(1 + h/(2f0)) with
  h = 63.6364 MHz and f0 = 3.6424 GHz gives |S21| = 2/√(4 + tan²θ) =
  **-31.23 dB**, i.e. 21.2 dB INSIDE the -10 dB gate. (#812 published -30.7 dB
  for this quantity; the derivation here is independent and lands at -31.2 dB
  — same conclusion, the 0.5 dB difference is bin-centre vs refined f0.)

  WHAT CHANGED (gates below; nothing is widened, one gate is tightened):
    * The notch frequency is now located by LOG-PARABOLIC SUB-BIN VERTEX
      REFINEMENT (``validation/crossval/comparators/spectral_features.py``,
      the same method already committed in
      ``scripts/diagnostics/build_msl_notch_palace_referee.py``), so the
      reported error is no longer quantised to 1.754% steps.
    * G1's window is TIGHTENED 15% -> 4.0%, derived from the three physical
      corrections the fringing-free quarter-wave oracle omits, evaluated on
      the realized board: open-end fringing (Hammerstad-Bekkadal) 0.886%,
      shunt-T reference plane bounded by 0.5·W_realized 2.646%, half-cell
      stub rasterisation 0.265% — worst-case sum 3.796%.
    * G2 gates the **-10 dB stopband fractional bandwidth** against the ideal
      shunt-open-stub closed form (4/π)·atan(r/6) = 0.210274 at r = 1, window
      ±20% (a stub whose coupling is 25% degraded, r ≤ 0.75, reads -24.7%).
      A shallow notch narrows this band whatever the sampling does, which is
      exactly what the depth gate could not see. The old depth gate is KEPT
      as a reported witness, not removed.
    * G4 is an in-run PROOF that the estimate is not bin-quantised: the two
      interleaved half-density sub-grids are disjoint in frequency, so a bare
      argmin's two answers are ALWAYS ≥ 1 full-grid bin apart, while the
      refined pair must agree to < 1 bin.
  Derivations, falsifiers and evidence:
  ``docs/design_notes/estimator_resolution_regate.md``.

Scope:
  - Uniform mesh dx=63.5µm = H_SUB/4 (issue #723; was dx=80µm, h_sub/dx=
    3.175, an UNDER-RESOLVED mixed-cell substrate per the "External
    cross-check" paragraph above and this script's own MSL-port
    preflight). The retired cv06 used non-uniform; ``add_msl_port``
    promotion remains uniform-lane only until a separate non-uniform
    evidence ladder exists.
  - Smaller domain than cv06 (line length 30mm vs 100mm) to keep
    runtime modest.
  - Stub length 12mm (same as cv06) → analytic notch ~3.68 GHz (realized
    width; see "Mesh convention").

Authoritative MSL port correctness gates: the unit + integration tests
under ``tests/unit/ports/test_msl_port*.py``. This crossval is a **physics-level
demo** that the new port API can resolve a stub-notch resonance without
the wire-port + absorber workaround.

Run: ``python validation/crossval/06b_msl_notch_filter_uniform.py``
(GPU-measured 329.2s solve on a single RTX4090; a CPU run of this mesh
was abandoned at 2 h 52 m unfinished — this script is GPU-lane, see the
manifest's cpu_runner note and "Runtime" above).
"""

import importlib.util
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# Sub-bin spectral-feature estimators shared with the Palace referee producers
# (#812 P3). Loaded by path so a bare run picks up THIS checkout's copy.
_SPEC = importlib.util.spec_from_file_location(
    "_cv06b_spectral_features",
    os.path.join(SCRIPT_DIR, "comparators", "spectral_features.py"))
sf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sf)

# --- #812 P3 gate windows, pre-declared in the design note before any
# --- measurement that judges them. See the "ESTIMATOR RESOLUTION" docstring
# --- section and docs/design_notes/estimator_resolution_regate.md.
NOTCH_FREQ_TOL_PCT = 4.0        # 0.886 (open end) + 2.646 (shunt-T plane)
                                # + 0.265 (half-cell stub) = 3.796, rounded up
STOPBAND_LEVEL_DB = -10.0
STOPBAND_BW_FRAC_IDEAL = 0.210274   # (4/pi)*atan(r/6) at r = Z0_line/Z_stub = 1
STOPBAND_BW_RATIO_WINDOW = (0.80, 1.20)   # fires at r <= 0.79 / r >= 1.28
HALF_GRID_WITNESS_BINS = 1.0    # structural: a quantised estimator scores 1.0


# Geometry — same as cv06, smaller line length
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
STUB_LEN = 12e-3
# The stub is the SAME width as the main line, so the ideal-shunt-stub
# coupling ratio r = Z0_line / Z_stub is exactly 1 -- which is what makes G2's
# closed-form -10 dB bandwidth (4/pi)*atan(r/6) a first-principles constant
# rather than a fit. Named separately (== W_TRACE by default, no behaviour
# change) only so scripts/diagnostics/cv06b_build_falsifiers.py can build the
# degraded-r variant G2 is supposed to catch.
W_STUB = W_TRACE
L_LINE = 30e-3        # vs cv06's 100mm
PORT_MARGIN = 2e-3
F_MAX = 7e9
DX = H_SUB / 4         # 63.5um — REALIZE-DECLARED on z (issue #723); see
                        # "Mesh convention" below.


def _build_sim() -> Simulation:
    """Build the notch-filter simulation with msl_port at both ends."""
    LX = L_LINE + 2 * PORT_MARGIN
    # Lateral box: W + 2·(2·h_sub + 8·dx) on the MSL side, plus stub_length
    # on the +y side to fit the open-circuit stub.
    msl_clearance = 2 * (2 * H_SUB + 8 * DX)
    LY = W_TRACE + msl_clearance + STUB_LEN + 2 * (2 * H_SUB + 8 * DX)
    LZ = H_SUB + 1.5e-3

    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")

    # Place trace at y where there's clearance below + stub above
    y_trace = (2 * H_SUB + 8 * DX) + W_TRACE / 2.0
    trace_y_lo = y_trace - W_TRACE / 2.0
    trace_y_hi = y_trace + W_TRACE / 2.0

    # Main microstrip line (full LX so it goes through CPML — required for
    # MSL port termination, see commit 8882ef1 on msl_port_integration test).
    sim.add(
        Box((0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + DX)),
        material="pec",
    )

    # Open-circuit stub branching off the main line at x = LX/2
    stub_x_centre = LX / 2.0
    stub_x_lo = stub_x_centre - W_STUB / 2.0
    stub_x_hi = stub_x_centre + W_STUB / 2.0
    sim.add(
        Box((stub_x_lo, trace_y_hi, H_SUB),
            (stub_x_hi, trace_y_hi + STUB_LEN, H_SUB + DX)),
        material="pec",
    )

    sim.add_msl_port(
        position=(PORT_MARGIN, y_trace, 0.0),
        width=W_TRACE, height=H_SUB,
        direction="+x", impedance=50.0,
    )
    sim.add_msl_port(
        position=(PORT_MARGIN + L_LINE, y_trace, 0.0),
        width=W_TRACE, height=H_SUB,
        direction="-x", impedance=50.0,
    )
    return sim


def _realized_trace_width(sim: Simulation) -> float:
    """Main-trace width as the RASTERIZER actually realizes it (metres).

    Read live from ``sim.fidelity_report()`` rather than re-derived with a
    ``round(W_TRACE / DX) * DX`` formula: the half-open ``[lo, hi)`` node
    convention (``rfx.geometry.csg.Box``) counts the OVERLAPPED node span,
    not the rounded declared extent, and the two disagree by a cell at this
    mesh (571.5um / 9 cells from the round() formula vs the true 635.0um /
    10 cells — see the "Mesh convention" docstring section, issue #723
    BLOCKER 1). ``geometry[1]`` is the main trace (added first of the two
    PEC bodies in ``_build_sim``); its y-axis is transverse to propagation.
    """
    report = sim.fidelity_report(print_report=False)
    for item in report:
        if item["entity"] == "geometry[1] 'pec'":
            for ax in item["axes"]:
                if ax["axis"] == "y":
                    return float(ax["realized_extent_um"]) * 1e-6
    raise RuntimeError(
        "_realized_trace_width: could not find geometry[1] 'pec' y-axis in "
        "sim.fidelity_report() — did _build_sim()'s geometry order change?"
    )


def evaluate(freqs, s21_mag, z0_real, f_notch_analytic):
    """Every gated quantity, as a pure function of the sweep.

    Factored out of ``main()`` deliberately (#812): the judgement of this case
    must be replayable on a saved or synthesised |S21| sweep without a
    5,729,080-cell solve, so a falsifier can show each gate failing on the
    defect it was added for. ``main()`` calls this; so does
    ``scripts/diagnostics/cv06b_estimator_falsifiers.py``.

    ``freqs`` in Hz, ``s21_mag`` linear magnitude, ``z0_real`` in ohm.
    """
    f = np.asarray(freqs, dtype=float)
    s21_mag = np.asarray(s21_mag, dtype=float)
    s21_db = 20 * np.log10(s21_mag + 1e-30)
    i_notch = int(np.argmin(s21_db))

    # The BIN argmin is kept and reported for continuity with every committed
    # log; the GATED number is the sub-bin log-parabolic vertex (#812 P3 — a
    # bin here is 63.6364 MHz = 1.754%, wider than the error being reported).
    est = sf.refined_extremum(f, s21_mag)

    # -10 dB stopband width — the quantity that replaces the unfailable depth
    # gate. Both edges are interpolated between bracketing bins (sub-bin).
    band = sf.band_at_level(f, s21_mag, STOPBAND_LEVEL_DB, i_notch)
    if band is None:
        bw_lo = bw_hi = bw_frac = bw_ratio = 0.0
        bw_bins = 0
    else:
        bw_lo, bw_hi, bw_bins = band
        bw_frac = (bw_hi - bw_lo) / est["refined_f"]
        bw_ratio = bw_frac / STOPBAND_BW_FRAC_IDEAL

    wit = sf.half_grid_witness(f, s21_mag)          # in-run resolution proof
    z0_median = float(np.median(np.asarray(z0_real, dtype=float)))
    lo_r, hi_r = STOPBAND_BW_RATIO_WINDOW

    m = {
        "f_notch_analytic": float(f_notch_analytic),
        "f_notch_bin": float(f[i_notch]),
        "f_notch_refined": float(est["refined_f"]),
        "sub_bin_shift": float(est["sub_bin_shift"]),
        "bin_hz": float(est["bin_width"]),
        "notch_depth_db": float(s21_db[i_notch]),
        "err_pct": abs(est["refined_f"] - f_notch_analytic) / f_notch_analytic * 100.0,
        "err_pct_bin": abs(float(f[i_notch]) - f_notch_analytic) / f_notch_analytic * 100.0,
        "bw_lo": bw_lo, "bw_hi": bw_hi, "bw_bins": int(bw_bins),
        "bw_frac": bw_frac, "bw_ratio": bw_ratio,
        "witness_bins": float(wit["spread_bins"]),
        "witness_argmin_bins": float(wit["argmin_spread_bins"]),
        "z0_median": z0_median,
    }
    m["gates"] = {
        "G1 notch freq vs analytic": m["err_pct"] < NOTCH_FREQ_TOL_PCT,
        "G2 -10 dB stopband width": lo_r < bw_ratio < hi_r,
        "G3 half-grid resolution witness": m["witness_bins"] < HALF_GRID_WITNESS_BINS,
        "G4 Z0 median": 40 < z0_median < 65,
        # RETAINED, NOT REMOVED, NOT WIDENED — and it cannot fail while a notch
        # exists at all (worst sampled minimum on this grid for an ideal r=1
        # stub is -31.23 dB). It stays as a witness; G2 carries the real depth
        # requirement.
        "notch depth (witness only)": m["notch_depth_db"] < -10,
    }
    return m


def report(m) -> bool:
    """Print the Result / Estimator / Gates blocks. Returns the verdict.

    The three ``Result:`` lines keep their exact historical labels so every
    committed log and ``scripts/diagnostics/report_msl_envelope.py`` keep
    parsing.
    """
    lo_r, hi_r = STOPBAND_BW_RATIO_WINDOW
    print()
    print("Result:")
    print(f"  Notch frequency (rfx)      = {m['f_notch_refined']/1e9:.3f} GHz")
    print(f"  Notch frequency (analytic) = {m['f_notch_analytic']/1e9:.3f} GHz")
    print(f"  Notch frequency error      = {m['err_pct']:.2f} %")
    print(f"  Notch depth |S21|          = {m['notch_depth_db']:.1f} dB")
    print(f"  Re(Z0) median              = {m['z0_median']:.1f} Ω")
    print()
    print("Estimator resolution (#812 P3):")
    print(f"  sweep bin                  = {m['bin_hz']/1e6:.4f} MHz "
          f"= {m['bin_hz']/m['f_notch_refined']*100:.3f} % at the notch")
    print(f"  bin argmin                 = {m['f_notch_bin']/1e9:.4f} GHz "
          f"(would report {m['err_pct_bin']:.2f} % vs analytic)")
    print(f"  sub-bin refined vertex     = {m['f_notch_refined']/1e9:.4f} GHz "
          f"({m['sub_bin_shift']:+.3f} bin)")
    print(f"  half-grid witness spread   = {m['witness_bins']:.4f} bin "
          f"(bare argmin on the same two sub-grids: "
          f"{m['witness_argmin_bins']:.4f} bin)")
    print(f"  -10 dB stopband            = {m['bw_lo']/1e9:.4f} – "
          f"{m['bw_hi']/1e9:.4f} GHz, fractional {m['bw_frac']:.5f} "
          f"({m['bw_bins']} bins), ratio to ideal r=1 stub "
          f"{m['bw_ratio']:.4f}")
    g = m["gates"]
    print()
    print("Gates:")
    print(f"  G1 Notch freq vs analytic (< {NOTCH_FREQ_TOL_PCT:.1f} %): "
          f"{'PASS' if g['G1 notch freq vs analytic'] else 'FAIL'}  "
          f"({m['err_pct']:.2f} %, sub-bin refined)")
    print(f"  G2 -10 dB stopband width / ideal r=1 stub ∈ "
          f"({lo_r:.2f}, {hi_r:.2f}): "
          f"{'PASS' if g['G2 -10 dB stopband width'] else 'FAIL'}  "
          f"({m['bw_ratio']:.4f}; measured fractional BW {m['bw_frac']:.5f} "
          f"vs closed form {STOPBAND_BW_FRAC_IDEAL:.6f})")
    print(f"  G3 half-grid resolution witness (< "
          f"{HALF_GRID_WITNESS_BINS:.1f} bin): "
          f"{'PASS' if g['G3 half-grid resolution witness'] else 'FAIL'}  "
          f"({m['witness_bins']:.4f} bin; a bin-quantised estimator scores "
          f"{m['witness_argmin_bins']:.4f} and cannot pass)")
    print(f"  G4 Z0 median ∈ (40, 65) Ω:       "
          f"{'PASS' if g['G4 Z0 median'] else 'FAIL'}  ({m['z0_median']:.1f} Ω)")
    print(f"  Notch depth (< -10 dB):          "
          f"{'PASS' if g['notch depth (witness only)'] else 'FAIL'}  "
          f"({m['notch_depth_db']:.1f} dB) — WITNESS ONLY: this gate is "
          f"21.2 dB from its own worst case and cannot fail while a notch "
          f"exists (#812; see the docstring)")
    return all(g.values())


def main() -> int:
    print("=" * 70)
    print("Crossval 06b: MSL Notch Filter (uniform mesh + add_msl_port)")
    print("=" * 70)
    print(f"εr={EPS_R}, h_sub={H_SUB*1e6:.0f}µm, W_declared={W_TRACE*1e6:.0f}µm")
    print(f"line length L={L_LINE*1e3:.0f}mm, stub L_stub={STUB_LEN*1e3:.1f}mm")
    print(f"mesh: dx={DX*1e6:.1f}µm, n_z_sub={int(round(H_SUB/DX))}")

    sim = _build_sim()

    # Hammerstad-Jensen ε_eff for the analytic notch — from the REALIZED
    # trace width, not the declared one (issue #723; see "Mesh convention").
    w_realized = _realized_trace_width(sim)
    u = w_realized / H_SUB
    EPS_EFF = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 * (1 + 12 / u) ** -0.5
    F_NOTCH_AN = C0 / (4 * STUB_LEN * np.sqrt(EPS_EFF))
    print(f"W_realized={w_realized*1e6:.1f}µm, u={u:.3f}, ε_eff_HJ={EPS_EFF:.3f}, "
          f"analytic notch f={F_NOTCH_AN/1e9:.3f} GHz")
    print()

    print("Preflight:")
    sim.preflight(strict=False)
    print()

    print("Running rfx 2-port S-matrix sweep...")
    t0 = time.time()
    res = sim.compute_msl_s_matrix(n_freqs=100, num_periods=20.0)
    dt = time.time() - t0
    print(f"  ... done in {dt:.1f}s")

    f = np.asarray(res.freqs)
    s11 = np.asarray(res.S[0, 0, :])
    s21 = np.asarray(res.S[1, 0, :])
    z0 = np.asarray(res.Z0[0, :])

    m = evaluate(f, np.abs(s21), z0.real, F_NOTCH_AN)
    all_ok = report(m)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(f / 1e9, 20 * np.log10(np.abs(s21) + 1e-30),
                 label="|S21| rfx (msl_port)", color="C0")
    axes[0].plot(f / 1e9, 20 * np.log10(np.abs(s11) + 1e-30),
                 label="|S11| rfx (msl_port)", color="C1")
    axes[0].axvline(F_NOTCH_AN / 1e9, color="k", ls="--", lw=0.8,
                    label=f"analytic notch ({F_NOTCH_AN/1e9:.3f} GHz)")
    axes[0].axvline(m['f_notch_refined'] / 1e9, color="C3", ls="-", lw=0.8,
                    label=f"rfx notch, sub-bin "
                          f"({m['f_notch_refined']/1e9:.4f} GHz)")
    axes[0].axhline(STOPBAND_LEVEL_DB, color="0.6", ls=":", lw=0.8,
                    label=f"{STOPBAND_LEVEL_DB:.0f} dB stopband level")
    axes[0].set_ylabel("|S| [dB]")
    axes[0].set_ylim(-50, 5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)
    axes[0].set_title("MSL notch filter — uniform mesh + add_msl_port")

    axes[1].plot(f / 1e9, np.abs(z0), label="|Z0|")
    axes[1].axhline(50, color="k", ls="--", lw=0.8, label="50 Ω")
    axes[1].set_xlabel("Frequency [GHz]")
    axes[1].set_ylabel("Z0 [Ω]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, "06b_msl_notch_filter_uniform.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out_png}")
    print(f"\n{'PASS' if all_ok else 'FAIL'}: cv06b — "
          f"{'MSL port resolves stub notch' if all_ok else 'gates failed'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
