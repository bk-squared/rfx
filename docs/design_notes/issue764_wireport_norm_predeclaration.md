# Issue #764 — wire-port driven-diagonal normalization: pre-declared falsifiers and fixtures

Committed BEFORE any implementation or measurement, per the adjudicated design
for issue #764 (whole-port V against the whole-port Z0 on the DRIVEN diagonal;
passive/legacy path byte-frozen). The falsifier text below is the adjudication's
text VERBATIM; the fixture section then binds every symbolic quantity in it
(bins, in-band set, L_short, geometries) to concrete numbers. Nothing in this
file may be widened after a measurement; a miss is reported as a miss with the
residual mechanism named.

## 1. Falsifiers (verbatim from the adjudicated design)

Committed BEFORE any measurement; never widened; a miss is reported as a miss
with the residual mechanism named. NU lane, POST ordering, driven-port diagonal
S_kk = (V_port − Z0·I)/(V_port + Z0·I). In-band only: top/bottom band-edge
decade excluded (float32 E4 ceiling; complex128 refuted at premise — do not
re-attempt). Fixtures: cubic dx = 0.5 mm class, wire extent ≤ 2 mm (validity
limit k·h_port < 0.3), loads attached directly at the gap (no feed posts)
except the thru prediction. No gate references +0.35426, +0.26780, 28.511,
(1−n_live)/(1+n_live), or any self-consistency value.

F0 (run FIRST — sense discriminator): a(ω) = (V_port + Z0·I)/(2√Z0) recorded
across {matched 50 Ω, PEC short, open} with identical drive/n_steps/dt: max
relative spread > 5% at any in-band bin falsifies the source-sense split.

F1 (matched): |S11| < 0.05 at every in-band bin; < 0.02 at bins ≤ 1 GHz. FAIL
kills the derivation.

F2 (PEC short): |S11| ∈ [0.95, 1.02] at every in-band bin; |arg(S11) − 180°| ≤
10° at the lowest in-band bin, ≤ 25° at band top (covers L_short ≤ 0.25 nH via
2·atan(ωL/Z0)); Re(S11) < 0 at every in-band bin.

F3 (open): |S11| ∈ [0.95, 1.02] at the lowest in-band bin, ≥ 0.93 at every
in-band bin; arg(S11) ∈ [−10°, +2°] at the lowest bin.

F4 (passivity order-of-operations): BEFORE interpreting any |S11| > 1, compute
Re(Z) with the SAME sense the formula uses (driven: Z = +V_port/I): Re(Z) ≥
−1e-3·|Z| at every in-band bin on every passive load; any in-band |S11| > 1.02
with Re(Z) ≥ 0 falsifies the extraction algebra, not the reference.

F5 (#683 circuit law + load law): quasi-static bins, Z_L ∈ {12.5, 25, 50, 100,
200} Ω: |I − V_src/(Z0+Z_L)| / |V_src/(Z0+Z_L)| < 5% at every load; regression
of measured S11 against analytic (Z_L−50)/(Z_L+50): slope ∈ [0.9, 1.1],
|intercept| < 0.05. (Slope −1-class kills the sense; the reciprocal form kills
magnitude.)

F6 (n_live invariance): same physical wire, dx halved (n_live 2 → 4): driven
matched-load |S11| moves < 0.05 per bin (the defect frame predicts a ~0.267
DC-class jump — 5× separation).

F7 (power bookkeeping, sense-matched): max in-band | |a|²−|b|² − Re(V_port·I*)
| ≤ 1e-5·max|a|² (wiring identity); matched-line fixture: |(1−|S11|²) −
P_flux/|a|²| ≤ 0.10 in-band (net-flux referee at a non-reflective plane only,
per the #313 do-not-repeat); short: 1−|S11|² ≤ 0.10 in-band.

F8 (KVL witness, falsifies the V definition independently of S): PEC-short
fixture, quasi-static bins: |V_port|/|V_mid| < 0.1.

F9 (current-uniformity premise): first/mid/last live-cell Ampère-loop |I|
spread ≤ 5% in-band on the matched fixture; failure falsifies the
single-reference-current definition itself — stop and report, do not average
silently.

F10 (extraction-only guard): field state after n steps bit-identical pre/post
change, same seed/fixture.

F11 (frozen-channel guard): lane-parity closed-form locks, sigma ORACLE 1/2,
and all off-diagonal locks remain green bit-for-bit — any red there means the
scoping was violated, not that a lock "moved".

Reported prediction, NOT a kill gate (feed-post reactance is DUT physics per
#318): canonical thru with 1 mm posts, |S11| in the measured 0.033–0.086
V-shape class, envelope ≤ 0.15 in-band; a value in 0.15–0.19 with
post-reactance shape is reported as fixture physics with the recomputed post
model, never folded into the gates.

Uniform lane: the same gates re-run only after the #683 flip lands; no
uniform-lane measurement before that flip can falsify or validate this design
(pre-injection sampling contaminates driven V at order 1: sigma·dt/eps ≈ 0.96
on the canonical cell).

## 2. Concrete bindings (committed with the falsifiers, before measurement)

**Frequency bins.** `FREQS = [0.2, 0.5, 1.0, 2.0] GHz`. In-band set = all four
bins (the drive pulse `GaussianPulse(f0=2e9, bandwidth=0.9)` carries healthy
spectral content across 0.2–3.8 GHz; 0.2 GHz is more than a decade above DC and
2 GHz more than a decade below the 10 GHz `freq_max` band edge, satisfying the
"top/bottom band-edge decade excluded" rule against the float32 E4 ceiling).
"Lowest in-band bin" = 0.2 GHz. "Band top" = 2.0 GHz. "Bins ≤ 1 GHz" =
{0.2, 0.5, 1.0} GHz. Quasi-static bins (F5, F8) = {0.2, 0.5} GHz.
k·h_port at 2 GHz with h_port = 1.5 mm (gap 1.0 mm + one dual half-cell each
end): 2π·2e9/3e8 · 1.5e-3 = 0.063 < 0.3. ✓

**FIX-A (clean gap-attached load; F0–F5, F7–F9).** NU lane (uniform-valued
`dz_profile`, POST ordering). `Simulation(freq_max=10e9, domain=(10e-3, 10e-3,
8e-3), dx=0.5e-3, dz_profile=np.full(16, 0.5e-3), boundary="pec")`, preflight
ON (no suppression), `n_steps=3000`.

- Driven port: `ez`, position (5.0e-3, 5.0e-3, 3.5e-3), extent 0.5e-3, Z0=50,
  excite=True, `GaussianPulse(f0=2e9, bandwidth=0.9)`. Expected 2 live cells
  spanning z ∈ [3.5, 4.5] mm (gap height 1.0 mm).
- Electrode plates (the "gap" terminals): PEC boxes
  `Box((3.5e-3, 3.5e-3, 3.0e-3), (6.5e-3, 6.5e-3, 3.5e-3))` (bottom) and
  `Box((3.5e-3, 3.5e-3, 4.5e-3), (6.5e-3, 6.5e-3, 5.0e-3))` (top): 3×3 mm,
  one cell thick, directly abutting the gap ends. No feed posts, no line.
- Load = FOUR passive wire ports (`excite=False`), `ez`, extent 0.5e-3, same z,
  at (4.5, 5.0), (5.5, 5.0), (5.0, 4.5), (5.0, 5.5) mm — each impedance
  4·Z_L so the parallel combination realizes Z_L attached one cell from the
  driven column, symmetric (loop inductance L_loop/4-class; geometric estimate
  L_eff ≈ 0.1 nH → ωL/(2Z0) ≈ 0.006 at 1 GHz, 0.013 at 2 GHz — comfortably
  inside F1's envelopes, so a F1 miss indicts the extraction, not the fixture).
  - Matched: Z_L = 50 (per-column 200).
  - F5 sweep: Z_L ∈ {12.5, 25, 50, 100, 200} (per-column {50, 100, 200, 400, 800}).
  - PEC short: replace the four load columns by four PEC boxes filling the same
    columns plate-to-plate. **L_short commitment (open question 1, committed
    BEFORE measuring):** four parallel one-cell loops, geometric estimate
    L_short ≈ 0.1 nH, comfortably ≤ 0.25 nH, so the verbatim F2 phase envelope
    (±10° lowest bin, ±25° band top via 2·atan(ωL/Z0)) stands un-widened.
  - Open: no load columns at all (plates present; DUT = plate capacitance
    ~35 fF: |Γ| = 1.000, arg ≈ −2·atan(Z0ωC) = −0.4° at 0.2 GHz — inside F3).
- Fixture sanity gate (G0-class, evaluated before the falsifiers; a failure
  here is FIXTURE INVALID, not a falsifier verdict): realized n_live = 2 on the
  driven port and every load column (asserted via `_wire_port_live_cells`
  against the assembled pec mask); the plates rasterize PEC at the intended
  k-planes; preflight ran (no `skip_preflight`).
- Raw quantities are read from the NU runner's raw DFT accumulators
  (`v_port_dft`, `v_dft`, `i_dft` — surfaced as `wire_sparams_raw` on the
  result), NOT from any wave decomposition, for F0/F4/F5a/F7a/F8/F9.
- V_src for F5a: the per-cell source table is captured from the runner
  (683-harness spy pattern); the port waveform is a CURRENT (amperes) per
  `make_current_source`, each live cell injecting waveform/n_live, so the
  whole-port Thevenin EMF is V_src(ω) = Z0·Ŵ_cell(ω) with Ŵ_cell the rect-DFT
  (×dt) of the captured per-cell table — the same discrete kernel the
  accumulators use.
- F9: per-cell I at first/mid/last live cell of the driven port, captured by
  appending two sampling-only wire-port spec entries (same cells, mid pinned to
  the first/last live cell) via a spy on `run_nonuniform` — sampling-only spec
  entries add no sigma and no source, so the field trajectory is untouched.
- F7b flux referee: six `add_flux_monitor` planes forming a closed box around
  the driven column only (faces at x = 4.75±0 mm…, i.e. between the driven and
  load columns, and above/below the plates), net outward P_box compared to
  (|a|²−|b|²) = |a|²(1−|S11|²): |(1−|S11|²) − P_box/|a|²| ≤ 0.10 at in-band
  bins on the matched fixture; short: 1−|S11|² ≤ 0.10 in-band. If the NU flux
  plumbing cannot express the box, that sub-check is reported NOT-RUN with the
  plumbing gap named — the gate is not re-aimed at another referee.

**FIX-A′ (F6 refinement arm).** Same physical hardware as FIX-A matched:
`dx=0.25e-3`, `dz_profile=np.full(32, 0.25e-3)`, `n_steps=6000`, driven-port
extent 0.75e-3 (the SAME physical plate-to-plate column, z ∈ [3.5, 4.5] mm:
n_live 2 → 4 exactly as the design states, since the physical column spans
extent + dx), load columns and plates at identical physical coordinates,
per-column impedance 200. F6 compares driven matched-load |S11| per bin
against FIX-A: move < 0.05 per bin.

**FIX-C (thru-with-posts, reported prediction ONLY).** Same grid class as
FIX-A with domain (14e-3, 10e-3, 8e-3): plates extended to x ∈ [4.0, 9.5] mm
(y ∈ [4.0, 6.0] mm), driven column at x = 4.5 mm, single matched load column
(impedance 50, extent 0.5e-3) at x = 9.0 mm — a ~4.5 mm parallel-plate thru
whose vertical runs are the ~1 mm "posts". Reported: |S11| per in-band bin vs
the measured 0.033–0.086 V-shape class, envelope ≤ 0.15; 0.15–0.19 with
post-reactance shape = fixture physics, reported with the recomputed post
model, never folded into the gates.

**F10 binding.** FIX-A matched fixture, 400 steps: the full final field state
(all E and H components) from the NU runner must be bit-identical between main
@ b29f9de and this branch (the change adds DFT accumulators and extraction
only; no field write is touched).

**F11 binding.** `tests/test_nu_wire_port_lane_parity.py` (closed-form +
parity, excite=False), `tests/test_nu_port_sigma_dual_spacing.py` (ORACLE 1/2,
excite=False), `tests/test_twoport_wire_port.py` off-diagonal assertions,
`tests/test_nonuniform_source_port_dual_spacing.py` — green without edits,
except the tests named in the lock-move list below.

## 3. Locks EXPECTED to move (re-pin only with written physical provenance)

Pins on the known-wrong driven normalization (single-cell V against whole-port
Z0) or on the dead-mid/all-extent midpoint pick:

- `tests/test_twoport_wire_port.py` — 28.511 envelope and the
  +0.35426 / +0.26780 docstring witnesses (driven column; diagonal moves).
- `tests/test_nu_wire_port_lane_parity.py::test_excited_port_lane_ordering_disagreement_is_open_683`
  — xfail(strict) on the OLD lane disagreement; the NU driven diagonal moves,
  so the measured residuals in its docstring are re-measured. It must remain a
  failing (xfail) marker while the uniform lane awaits the #683 flip.
- Uniform-lane driven-diagonal value pins discovered by the suite run
  (forward/run fast-path S11 values), including the dump-parity schema lock in
  `tests/test_sparam_driver_dump_parity.py` (the replay bundle grows the
  whole-port gap-voltage channel the physical diagonal needs).
- Any dead-mid-cell quenched-current pins surfaced by the suite run (open
  question 2) — each re-pin carries its own provenance note.

NOT moving (frozen witnesses): lane-parity closed forms, sigma ORACLE 1/2, all
off-diagonal #308/#313 locks, the #313 refplane path (byte-frozen legacy
diagonals), the vinc channel, `extract_s11_normalised`.

## 4. Pre-measurement deviation notes

None at commit time. (The F6 "n_live 2 → 4" parenthetical was checked against
the coded rasterization — cells span position…position+extent inclusive, so
the same physical plate-to-plate column at halved dx does give 2 → 4 with
extent 0.5 mm → 0.75 mm — no deviation needed.)
