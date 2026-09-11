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
  **-31.23 dB**, i.e. 21.2 dB INSIDE the -10 dB gate. (#812 published
  -30.7 dB for this quantity. The derivation here is independent and lands at
  -31.20 dB at f0 = 3.627 GHz, -31.23 at 3.6424, -31.32 at 3.679 — so the
  choice of f0 moves it 0.12 dB and does NOT account for the 0.5 dB gap;
  reaching -30.7 dB from this model needs a 67.4 MHz bin at f0 = 3.627 GHz
  (67.7 MHz at 3.6424 GHz). The origin of the
  difference is NOT established. Both give the same verdict, and this script
  quotes its own -31.23 dB rather than the audit's.)

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
    * G3 is an in-run PROOF that the estimate is not bin-quantised: the two
      interleaved half-density sub-grids are disjoint in frequency, so a bare
      argmin's two answers are ALWAYS ≥ 1 full-grid bin apart, while the
      refined pair must agree to < 1 bin.
  Derivations, falsifiers and evidence:
  ``docs/design_notes/estimator_resolution_regate.md``.

  ROUND-2 APPEND (2026-09-01; PRE-#931 HISTORY, superseded by the
  committed sheet-board measurements below). Two things a reader must not infer.
  (1) CRITERION (A) -- ANSWERED 2026-09-02 (was "not yet demonstrated on
      this board"): ``scripts/vessl_cv06b_estimator_falsifiers.yaml`` ran
      (VESSL 369367257702, exit 0) and every gate passed on THIS mesh --
      ``validation/crossval/_06b_msl_notch_results/
      cv06b_build_falsifiers_summary.json`` BEFORE its #931 regeneration
      (historical criterion_A_baseline: err 1.453 %,
      BW ratio 0.9684, witness 0.3175 bin, Z0 46.48 ohm). Same run, build-level
      (B): a 5-cell stub fires G2 (BW ratio 0.648) while the depth witness
      still passes; and ONE PRE-DECLARED FALSIFIER FIRED -- a one-cell
      stub-length error moved the refined estimate 0.145 % against a
      predicted 0.532 % (bin argmin: 0.000 %). Recorded, not softened, and
      not attributed (see the design note, section 7.6). G1's verdict does
      not rest on sub-bin resolution (1.45 + 1.75 < 4.0), but sub-bin
      resolution of the notch on this board is NOT demonstrated.
  (2) The round-1 shallow-notch falsifier was WITHDRAWN as near-circular (it
      rescaled the measured sweep by the closed form G2's window comes from).
      Its replacement builds the defect from geometry alone
      (``scripts/diagnostics/cv06b_shallow_stub_model.py``); the ladder and
      its independence check are
      ``tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json
      ::case_C_shallow_notch_from_geometry``. No window moved in either round.

#931 LATTICE OWNERSHIP — APPENDED 2026-09-07. Nothing above is withdrawn;
the paragraphs it supersedes are marked below and stay as a pre-#931 record.

  WHAT CHANGED IN THE DECLARATION. The main line and the stub used to be
  ``Box((.., H_SUB), (.., H_SUB + DX))`` — a Box one cell thick, which the
  lattice ownership contract realizes as a VOLUME: a 63.5µm filled metal
  slab with electric walls at BOTH z=254 and z=317.5µm and Ez shorted
  between them, a conductor a quarter of the substrate thick on a board
  whose whole point is a zero-thickness microstrip. They are now declared
  as SHEETS — a Box with EQUAL z corners at z=H_SUB, which is how the
  contract spells "a conductor on this node plane, zero thickness" (§1.5).
  The realization is the tangential Ex/Ey edges between neighbouring
  footprint nodes on the z=254µm plane, with the normal Ez left live.
  ``add_thin_conductor(...)`` would produce exactly the same sheet; the
  zero-thickness Box keeps the declaration inside ``sim.add``.

  WHAT CHANGED IN THE REALIZED BOARD — counted edge by edge, not assumed.
  The pre-#931 rule zeroed a tangential edge at EVERY masked node with a
  masked neighbour, which includes one edge REACHING PAST the hi rim of the
  footprint. The contract's sheet rule zeroes the edge between two
  footprint nodes and no other, so n footprint nodes carry n-1 edges. What
  that does and does NOT move here:

    * the LONGITUDINAL-current rows are UNCHANGED. The main line still has
      10 Ex rows (y-nodes 16..25) and the stub still has 10 Ey columns
      (x-nodes 263..272). The contract removed rim edges, not filaments, so
      the two lines' current distributions are the same as before;
    * the GEOMETRIC realized extent, which is the node span the contract
      owns and ``fidelity_report`` prints, drops one cell on every in-plane
      dimension: 635.0 -> 571.5 µm on both widths;
    * the stub's OPEN END moves down one node, 13652.5 -> 13589.0 µm. From
      the main line's centre row (1301.75 µm) the quarter-wave length goes
      12350.75 -> 12287.25 µm, -0.514%. Measured from the line's far EDGE
      it does not move at all (12001.5 µm both eras) — both ends lost the
      same edge — which is why the length must be quoted from the junction
      reference plane, not from a rim.

  r = Z0_line/Z_stub stays exactly 1 (both lines realize the same width and
  the same row count — now ASSERTED at build time rather than argued from
  ``W_STUB == W_TRACE``), so G2's closed form is untouched.

  THE ANALYTIC REFERENCE DOES NOT MOVE, and that is a measured claim, not a
  convenience. ``_realized_trace_width`` feeds Hammerstad-Jensen, which
  wants the ELECTRICAL width of the strip, and the strip's row count did
  not change: n_rows * DX = 635.0 µm, exactly the number this case has
  always used. The evidence that n*DX (not (n-1)*DX) is the electrical
  width was the PRE-#931 measurement — median Re(Z0) 46.48 Ω
  against HJ(635.0, 254) = 46.18 Ω (+0.65%) and HJ(571.5, 254) = 49.39 Ω
  (-5.9%) — and cv07 repeating it on a different board (12 rows at
  dx=200 µm, measured 50.30 Ω, HJ(2400, 800) = 51.19 Ω vs HJ(2200, 800) =
  54.22 Ω). Those measurements are historical: the committed sheet-board
  medians are now 48.19 Ω for cv06b and 51.91 Ω for cv07 (passband). The
  cv06b width convention is not settled by the new result; see below. The
  retained reference uses u = 2.500, ε_eff = 2.882252, F_NOTCH_AN = 3.678954 GHz and
  G1's three window terms (0.886 / 2.646 / 0.265 = 3.796%) all stand.

  SUPERSEDED BY THIS SECTION (kept above as the pre-#931 record):
    * "Mesh convention"'s verbatim ``fidelity_report()`` quote "geometry[1]
      'pec' ... realized [1016.0, 1651.0] µm | extent 600.0 -> 635.0 µm".
      The report now reads the SHEET row: realized [1016.0, 1587.5] µm,
      extent 571.5 µm. The 635.0 µm that survives above is the electrical
      width, which is a different quantity that happened to coincide with
      the old geometric one because the old rule zeroed one edge past the
      rim;
    * its argument that ``round(W_TRACE/DX)*DX`` = 571.5 µm is the WRONG
      answer: 571.5 µm is the geometric extent and the formula agrees with
      it. What the formula cannot give is which of the two quantities an
      electrical formula wants;
    * the Z0 cross-check against ``msl_z0_bias_floor_sweep``'s committed
      "aligned h_sub/4" row (46.098 Ω): that artifact was produced under
      the pre-#931 realization by a producer outside this case, so it is
      history until it is re-run — not a live cross-check.

  PRE-DECLARED BEFORE THE POST-CONTRACT SOLVE (design note §5 discipline;
  measured values are reported against this list verbatim, pass or fail):
    * G4 Z0 median stays 46.48 ± 1.0 Ω. THIS IS THE FALSIFIER FOR THE WIDTH
      CONVENTION: ~46.5 Ω says the electrical width of an n-row strip is
      n*DX and the analytic reference was right to stay; ~49.4 Ω says the
      geometric span is the electrical width, and then
      ``_realized_trace_width``, EPS_EFF, F_NOTCH_AN (3.6790 -> 3.6942 GHz)
      and G1's shunt-T term (2.646 -> 2.381%) all re-derive on 571.5 µm;
    * the measured notch RISES ≈0.51%, the stub's one-cell shortening:
      3.6255 -> 3.644 GHz (refined vertex), predicted ±0.3 pp;
    * err_pct therefore FALLS 1.4530 -> ≈0.95% (the measured notch moves
      toward a reference that stays put). Still G1 PASS by a wider margin;
      a rise above 1.45% falsifies the reading above;
    * G2 bw_ratio stays 0.968 ± 0.05 (r = 1 preserved by construction);
    * the half-grid witness and the notch depth are not predicted to move.

  COMMITTED POST-#931 RESULT (VESSL 369367259191, 2026-09-07):
  ``validation/crossval/_06b_msl_notch_results/
  cv06b_build_falsifiers_summary.json`` reports baseline notch error
  2.1649 %, BW ratio 0.9991, witness 0.4469 bin, and median Re(Z0) 48.19 Ω.
  The refined notch is 3.7586 GHz and the depth is -39.44 dB. All four
  existing gates pass, but the width-convention pre-declaration above is
  FALSIFIED: 48.19 Ω is outside 46.48 ± 1.0 Ω, and the notch error rose
  instead of falling. The electrical-width attribution remains unresolved;
  no reference or tolerance is changed on the strength of this result.
  The one-cell stub arm now moves the refined estimate 0.8228 % against
  the predicted 0.5320 % (bare argmin 1.6949 %), so the declared visibility
  criterion passes, with a 1.55x over-response that is not attributed.
  The narrow-stub arm has BW ratio 0.6553 and notch error 6.4388 %: G2 and
  G1 both fire while the retained depth witness still passes.

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
                                # + 0.265 (half-cell stub) = 3.796, rounded
                                # up. UNCHANGED by #931: all three terms are
                                # functions of the ELECTRICAL trace width,
                                # and the strip's row count did not change
                                # (see the "#931 LATTICE OWNERSHIP"
                                # docstring section). On the GEOMETRIC
                                # extent (571.5um) the same terms would sum
                                # to 3.519%; G4's measured Z0 in the
                                # post-contract run decides which reading is
                                # the electrical one.
STOPBAND_LEVEL_DB = -10.0
STOPBAND_BW_FRAC_IDEAL = 0.210274   # (4/pi)*atan(r/6) at r = Z0_line/Z_stub = 1
STOPBAND_BW_RATIO_WINDOW = (0.80, 1.20)   # fires at r <= 0.797 / r >= 1.205
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


def _build_sim(*, stub_x_bounds: tuple[float, float] | None = None) -> Simulation:
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
    #
    # SHEET, not a one-cell volume (#931 §1.5). The z corners are EQUAL, which
    # is how the lattice-ownership contract spells "a conductor on this node
    # plane, zero thickness": the realization is the tangential Ex/Ey edges of
    # the footprint on the z = H_SUB node plane, and Ez through the metal stays
    # live. Drawn as `H_SUB -> H_SUB + DX` (what this script shipped before
    # #931) the same Box is a VOLUME and realizes a 63.5 um filled slab with
    # walls at BOTH 254 and 317.5 um — a conductor a quarter of the substrate
    # thick on a board whose whole point is a zero-thickness MSL trace.
    sim.add(
        Box((0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB)),
        material="pec",
    )

    # Open-circuit stub branching off the main line at x = LX/2 (same sheet
    # plane, so trace and stub are one connected conductor).
    stub_x_centre = LX / 2.0
    stub_x_lo = stub_x_centre - W_STUB / 2.0
    stub_x_hi = stub_x_centre + W_STUB / 2.0
    if stub_x_bounds is not None:
        # The build falsifier supplies physical coordinates from a completed
        # baseline grid. A width in cells alone does not align both faces.
        stub_x_lo, stub_x_hi = map(float, stub_x_bounds)
        if (not np.isfinite([stub_x_lo, stub_x_hi]).all()
                or stub_x_hi <= stub_x_lo
                or abs(stub_x_hi - stub_x_lo - W_STUB) > 1e-12):
            raise ValueError("stub_x_bounds must span the declared W_STUB")
    sim.add(
        Box((stub_x_lo, trace_y_hi, H_SUB),
            (stub_x_hi, trace_y_hi + STUB_LEN, H_SUB)),
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


def realized_metal(sim: Simulation) -> dict:
    """The metal this build actually realizes, read from the ONE realization
    function (#931 §1.7). No solve, no re-derived rule.

    ``rfx.boundaries.pec.realized_pec_edge_masks`` is the single owner that
    turns declared geometry into PEC E edges, so every number this case
    quotes about its own conductor is measured from ITS output and the
    script cannot drift from the solver. Both metal entries here are SHEETS
    (§1.3): a zero-thickness Box on the substrate-top node plane, realized
    as the tangential Ex/Ey edges BETWEEN neighbouring footprint nodes, with
    the normal Ez through the metal left live.

    Returned (metres unless the name says otherwise):

    ``plane_k`` / ``plane_z``   the node plane both sheets land on;
    ``trace_w`` / ``stub_w``    GEOMETRIC realized width = the span between
                                the outermost node rows carrying the
                                conductor's longitudinal edges, i.e.
                                ``(n_rows - 1) * dx``. This is what the
                                contract owns and what ``fidelity_report``
                                prints for a sheet;
    ``trace_w_elec`` / ``stub_w_elec``
                                ELECTRICAL strip width = ``n_rows * dx``.
                                See :func:`_realized_trace_width` for why
                                a quasi-TEM formula takes this one and the
                                measurement that decides between them;
    ``stub_len``                stub open end minus the main line's far
                                edge (both moved together, so this is the
                                one length the contract did NOT change);
    ``stub_len_centreline``     open end minus the main line's CENTRE row —
                                the quarter-wave length, which DID move.

    A sheet IS its set of edges, and n footprint nodes carry n-1 edges. The
    pre-#931 neighbour rule zeroed one EXTRA edge past the hi rim of every
    footprint, which is why this case used to read 635.0 um (10 * dx) for a
    600 um trace as a GEOMETRIC extent — a CELL count of a NODE mask, the
    #929 class. The row count itself did not change.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes

    grid = sim._build_grid()
    sheets: list = []
    wires: list = []
    assembled = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
    pec_mask = assembled[3]
    if pec_mask is None and not sheets and not wires:
        raise RuntimeError("realized_metal: this build has no conductor at all")
    edges = realized_pec_edge_masks(pec_mask, sheets=tuple(sheets),
                                    wires=tuple(wires),
                                    periodic=sim._periodic_flags())
    mx, my, mz = (np.asarray(e) for e in edges)
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    gc = coords_from_uniform_grid(grid)
    nodes = (np.asarray(gc.x), np.asarray(gc.y), np.asarray(gc.z))

    planes = realized_wall_planes(edges, 2)
    if len(planes) != 1:
        raise RuntimeError(
            "realized_metal: this board declares two zero-thickness sheets on "
            f"ONE node plane, but the realization has tangential walls at z "
            f"planes {planes} — a sheet is one plane (#931 §1.3); two planes "
            "is what a one-cell VOLUME trace would realize")
    k = planes[0]
    # Main line: the rows whose Ex edges span the propagation axis. The
    # stub's rows carry only its own W_STUB-wide handful of Ex edges, so the
    # longest rows are the main line's, and they must all be equally long
    # (the line is a rectangle) — a row count that is not a plateau means
    # the footprint is not the rectangle this case declares.
    ex_per_row = mx[:, :, k].sum(axis=0)
    if not ex_per_row.any():
        raise RuntimeError(
            "realized_metal: no Ex edge on the sheet plane — no metal runs "
            "along the propagation axis")
    trace_rows = np.flatnonzero(ex_per_row == ex_per_row.max())
    if trace_rows.size < 2 or trace_rows.max() - trace_rows.min() + 1 != trace_rows.size:
        raise RuntimeError(
            f"realized_metal: the longest Ex rows are {trace_rows.tolist()} — "
            "the main line does not realize as one contiguous block of rows")
    # ``Ex[i, j, k]`` sits ON node row j (it is the edge from node i to i+1),
    # so the rows carrying Ex are the conductor's NODE rows and the metal
    # spans from the first to the last of them.
    j0, j1 = int(trace_rows.min()), int(trace_rows.max())
    trace_w = float(nodes[1][j1] - nodes[1][j0])
    # Stub: the transverse (Ey) edges at or beyond the main line's far row.
    # ``Ey[i, j, k]`` sits on node column i and spans node j -> j+1, so the
    # columns are node indices and the edge indices are the y intervals.
    stub_cols = np.flatnonzero(my[:, j1:, k].any(axis=1))
    if stub_cols.size == 0:
        raise RuntimeError(
            "realized_metal: no Ey edge above the main line — the stub is not "
            "connected to the trace (a slit at the junction)")
    i0, i1 = int(stub_cols.min()), int(stub_cols.max())
    stub_w = float(nodes[0][i1] - nodes[0][i0])
    stub_edges = np.flatnonzero(my[i0:i1 + 1, :, k].any(axis=0))
    stub_open = int(stub_edges.max()) + 1
    stub_len = float(nodes[1][stub_open] - nodes[1][j1])
    n_rows = j1 - j0 + 1
    n_cols = i1 - i0 + 1
    y_centre = 0.5 * (float(nodes[1][j0]) + float(nodes[1][j1]))
    return dict(
        plane_k=int(k), plane_z=float(nodes[2][k]),
        n_sheets=len(sheets),
        n_volume_cells=0 if pec_mask is None else int(np.asarray(pec_mask).sum()),
        trace_j=(j0, j1), trace_w=trace_w, n_rows=n_rows,
        trace_y=(float(nodes[1][j0]), float(nodes[1][j1])),
        stub_i=(i0, i1), stub_w=stub_w, n_cols=n_cols,
        stub_x=(float(nodes[0][i0]), float(nodes[0][i1])),
        trace_w_elec=n_rows * DX, stub_w_elec=n_cols * DX,
        stub_len=stub_len, stub_open_j=stub_open,
        stub_len_centreline=float(nodes[1][stub_open]) - y_centre,
        n_ex=int(mx.sum()), n_ey=int(my.sum()), n_ez=int(mz.sum()),
    )


def assert_realized_metal(sim: Simulation) -> dict:
    """MANDATORY build-time geometry check (#931): refuse to solve unless the
    realized metal IS the declared metal. No FDTD step runs here.

    cv15's ``assert_realized_stack`` is the model — a case that quotes a
    physical number must first show the lattice built the object its
    docstring describes. What this asserts, and what each item catches:

    1. both metal entries are SHEETS on ONE node plane, and that plane is
       the substrate top ``z = H_SUB``. Catches a foil re-declared as a
       volume (which realizes a 63.5 um filled slab with walls at 254 AND
       317.5 um and Ez shorted between them), a sheet snapped to the wrong
       node, and a mesh on which ``H_SUB`` stops landing on a node line;
    2. no PEC VOLUME cell exists at all — the same defect from the other
       side, and the cheapest possible statement of "this board's metal has
       no thickness";
    3. trace and stub realize the SAME width, geometrically AND in row
       count. G2's closed form ``(4/pi)*atan(r/6)`` is evaluated at
       ``r = Z0_line/Z_stub = 1`` "exactly by construction", and the
       construction is this equality of REALIZED widths, not the
       declaration ``W_STUB == W_TRACE``;
    4. realized width and realized stub length are each within one cell of
       their declared values — catches a footprint that lost or gained a row.

    Returns the measured dict for the caller to print and record.
    """
    m = realized_metal(sim)
    problems = []
    if m["n_sheets"] != 2:
        problems.append(f"expected 2 PEC sheets, classified {m['n_sheets']}")
    if m["n_volume_cells"]:
        problems.append(
            f"{m['n_volume_cells']} PEC VOLUME cell(s) realized; both metal "
            "entries must be zero-thickness sheets")
    if abs(m["plane_z"] - H_SUB) > 1e-12:
        problems.append(
            f"sheet plane at z={m['plane_z']*1e6:.3f} um, declared substrate "
            f"top z={H_SUB*1e6:.3f} um (k={m['plane_k']})")
    if abs(m["trace_w"] - m["stub_w"]) > 1e-12 or m["n_rows"] != m["n_cols"]:
        problems.append(
            f"realized trace width {m['trace_w']*1e6:.1f} um "
            f"({m['n_rows']} rows) != realized stub width "
            f"{m['stub_w']*1e6:.1f} um ({m['n_cols']} rows), so G2's r = 1 "
            "is not by construction")
    if abs(m["trace_w"] - W_TRACE) > DX:
        problems.append(
            f"realized trace width {m['trace_w']*1e6:.1f} um is more than one "
            f"cell ({DX*1e6:.1f} um) from the declared {W_TRACE*1e6:.1f} um")
    if abs(m["stub_len"] - STUB_LEN) > DX:
        problems.append(
            f"realized stub length {m['stub_len']*1e6:.1f} um is more than one "
            f"cell from the declared {STUB_LEN*1e6:.1f} um")
    if problems:
        raise RuntimeError(
            "assert_realized_metal: the realized conductor is not the declared "
            "one — refusing to quote a notch frequency. "
            + "; ".join(problems) + f" [measured: {m}]")
    return m


def _realized_trace_width(sim: Simulation) -> float:
    """ELECTRICAL width of the realized main line (metres), for the
    quasi-TEM analytic reference.

    Measured from the realized PEC EDGE set (:func:`realized_metal`) — what
    the solver zeroes — not from a cell-mask-derived report. It is
    ``n_rows * DX``: the strip carries its longitudinal current on n node
    rows spaced DX apart, and each row stands for a DX-wide filament, so
    the metal reaches half a cell beyond the outermost rows. openEMS
    encodes the same convention geometrically — its thirds rule puts mesh
    lines INSIDE each metal edge rather than on it (see ``run_openems`` in
    cv07), i.e. the conductor extends past its outermost line.

    NOT the same quantity as the contract's GEOMETRIC realized extent, and
    the difference is exactly one cell:

      geometric (node span, what ``fidelity_report`` prints for a sheet)
          (n_rows - 1) * DX = 571.5 um
      electrical (this function)
          n_rows * DX       = 635.0 um

    WHICH ONE BELONGS IN HAMMERSTAD-JENSEN IS A MEASURED QUESTION. The
    PRE-#931 medians were 46.48 ohm here and 50.30 ohm in cv07's passband.
    Those values motivated n_rows*DX: HJ(635.0, 254) = 46.18 ohm versus
    HJ(571.5, 254) = 49.39 ohm; cv07's HJ values were 51.19 and 54.22 ohm.
    The committed post-contract runs instead read 48.19 ohm here and
    51.91 ohm in cv07's passband. The cv06b result is outside the declared
    46.48 +/- 1.0 ohm falsifier window and between the two width predictions.
    The width convention is therefore unresolved, not verified by the old
    measurement. This function retains n_rows*DX and its analytic reference;
    changing them requires a separate physics adjudication. See the module
    docstring's COMMITTED POST-#931 RESULT and the run's RECOMPUTE.md.

    PRE-#931 this read ``fidelity_report()``'s ``realized_extent_um`` for
    ``geometry[1] 'pec'`` and got the same 635.0 um, but as a GEOMETRIC
    extent — the old rule zeroed one edge past the footprint's hi rim, so
    its geometric and electrical answers coincided by accident. The old
    docstring's claim that ``round(W_TRACE/DX)*DX`` = 571.5 um is "the
    WRONG answer" is retired: 571.5 um is the geometric extent, and the
    formula agrees with it.
    """
    return float(realized_metal(sim)["trace_w_elec"])


def worst_sampled_notch_db(bin_hz, f0, r=1.0):
    """dB of the WORST sampled minimum of an ideal shunt-open-stub zero.

    The retained depth gate reads the SAMPLED minimum of a true transmission
    zero, so its blindness is set by the grid, not by the notch. The worst a
    grid of ``bin_hz`` can do is land half a bin off ``f0``:
    ``|S21| = 2/sqrt(4 + tan^2 theta)``, ``theta = (pi/2)(1 + h/(2 f0))``.
    Computed live so the margin printed beside the gate cannot go stale if the
    sweep length ever changes (#812 numeric-provenance discipline).
    """
    theta = 0.5 * np.pi * (1.0 + 0.5 * float(bin_hz) / float(f0))
    return float(20.0 * np.log10(2.0 / np.sqrt(4.0 + (r * np.tan(theta)) ** 2)))


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
    m["worst_sampled_depth_db"] = worst_sampled_notch_db(
        m["bin_hz"], m["f_notch_refined"])
    m["depth_gate_blind_margin_db"] = m["worst_sampled_depth_db"] - (-10.0)
    m["gates"] = {
        "G1 notch freq vs analytic": m["err_pct"] < NOTCH_FREQ_TOL_PCT,
        "G2 -10 dB stopband width": lo_r < bw_ratio < hi_r,
        "G3 half-grid resolution witness": m["witness_bins"] < HALF_GRID_WITNESS_BINS,
        "G4 Z0 median": 40 < z0_median < 65,
        # RETAINED, NOT REMOVED, NOT WIDENED — and it cannot fail while a notch
        # exists at all: the worst sampled minimum on this grid for an ideal
        # r=1 stub is computed live above as m["worst_sampled_depth_db"] and
        # printed beside the verdict. It stays as a witness; G2 carries the
        # real depth requirement.
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
          f"({m['notch_depth_db']:.1f} dB) — WITNESS ONLY: an ideal r=1 stub's "
          f"WORST sampled minimum on this grid is "
          f"{m['worst_sampled_depth_db']:.2f} dB, i.e. "
          f"{abs(m['depth_gate_blind_margin_db']):.1f} dB inside the gate, so "
          f"it cannot fail while a notch exists (#812; see the docstring)")
    return all(g.values())


def main() -> int:
    print("=" * 70)
    print("Crossval 06b: MSL Notch Filter (uniform mesh + add_msl_port)")
    print("=" * 70)
    print(f"εr={EPS_R}, h_sub={H_SUB*1e6:.0f}µm, W_declared={W_TRACE*1e6:.0f}µm")
    print(f"line length L={L_LINE*1e3:.0f}mm, stub L_stub={STUB_LEN*1e3:.1f}mm")
    print(f"mesh: dx={DX*1e6:.1f}µm, n_z_sub={int(round(H_SUB/DX))}")

    sim = _build_sim()

    # Build-time geometry check (#931), BEFORE the solve and before any
    # number is quoted: the realized metal must be the declared metal.
    rm = assert_realized_metal(sim)
    print()
    print("Realized metal (#931, from realized_pec_edge_masks — no solve):")
    print(f"  PEC sheets                 = {rm['n_sheets']} "
          f"(volume cells: {rm['n_volume_cells']})")
    print(f"  sheet plane                = k={rm['plane_k']}, "
          f"z={rm['plane_z']*1e6:.1f} um (declared substrate top "
          f"{H_SUB*1e6:.1f} um)")
    print(f"  trace width  declared/real = {W_TRACE*1e6:.1f} / "
          f"{rm['trace_w']*1e6:.1f} um geometric, "
          f"{rm['trace_w_elec']*1e6:.1f} um electrical "
          f"({rm['n_rows']} node rows)")
    print(f"  stub  width  declared/real = {W_STUB*1e6:.1f} / "
          f"{rm['stub_w']*1e6:.1f} um geometric ({rm['n_cols']} node rows) "
          "— r = 1 needs these two equal")
    print(f"  stub length  declared/real = {STUB_LEN*1e6:.1f} / "
          f"{rm['stub_len']*1e6:.1f} um (edge-to-edge), "
          f"{rm['stub_len_centreline']*1e6:.1f} um from the line centre")
    print(f"  PEC edges (Ex, Ey, Ez)     = "
          f"({rm['n_ex']}, {rm['n_ey']}, {rm['n_ez']})")

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
