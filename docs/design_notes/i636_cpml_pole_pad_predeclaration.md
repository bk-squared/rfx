# 2026-08-29 — #636: dispersion poles in the CPML pad, pre-run predeclaration

Written BEFORE any measurement of this work stage, under the repository's
one-clean-predeclared-attempt discipline. Author agent session for backlog
item 636 (the deferred half of #627: extending Debye/Lorentz pole masks into
the CPML pad destabilizes high-Q edge-touching structures).

## R1 — tracked context (premise verification, already done)

Re-measured on `b29f9de` with the lock-test fixture
(`tests/unit/boundaries/test_cpml_pad_material_extension.py::test_pole_extension_stability_lock`,
45x39x12 cells, dx=1mm, eps_inf=4 slab with one Lorentz pole
omega_0 = 2*pi*3e9, delta = omega_0/120 (Q=60), kappa = 3*omega_0^2,
edge-touching in x and y, cpml_layers=8), 20,000 steps:

- shipped (poles NOT extended): last/mid decile ratio **0.2145**, decays.
- poles extended into the pad (statics-style replication incl. #627a
  hi-face fallback): ratio **5.032**, growing, finite, no NaN.
- at 8,000 steps the extended variant measures **0.4499** (< 1): the
  committed 8000-step lock NO LONGER reds on naive re-addition — onset
  moved past 8000 steps since `c9c1864` (#655 boundary-node fix). The
  historical docstring numbers (2.546 at 8k, 649 at 20k) are stale.

Code sites: `rfx/api/_compile.py` (poles deliberately not extended),
`rfx/geometry/rasterize_grid.py::extend_cpml_pad_materials` (`_vacuum`
ignores poles — the pole-only-material hole is intact),
`rfx/boundaries/cpml.py` (`alpha = 0.05 * (1 - rho)` literal, max at the
interior edge, in S/m units — `b = exp(-(sigma/kappa + alpha) dt/eps0)`).

## Hypothesis H1 — mechanism

The shipped step composes the ADE pole recurrence
`P^{n+1} = a P^n + b_p P^{n-1} + c_p E^n` (rfx/materials/lorentz.py) with
the CPML recursive-convolution correction
`psi^{n+1} = b psi^n + c (curl term)`, `E += ce * (psi + (1/kappa - 1) curl)`
(rfx/boundaries/cpml.py) applied AFTER the dispersive E-update
(rfx/simulation.py step body). For a high-Q pole the homogeneous P roots
sit at |root| = sqrt((1-delta*dt)/(1+delta*dt)) ~ 1 - delta*dt, i.e. just
inside the unit circle; H1 says the composed per-cell one-step operator
(E, H, P, P_prev, psi_e, psi_h) has spectral radius > 1 for some spatial
frequency when the pole coexists with CPML sigma > 0, i.e. the discrete
composed update is NOT unconditionally stable even though (pole alone) and
(CPML alone) each are.

## M1 — discrete frozen-coefficient eigenvalue scan (root-cause probe)

Script: `validation/research/cpml_pole_pad/eigen_scan_636.py` (numpy only,
seconds). Builds the EXACT shipped one-step matrices, frozen per CPML
layer, using the shipped coefficient formulas:

- ADE: `a_p = (2 - w0^2 dt^2)/(1 + d dt)`, `b_p = -(1 - d dt)/(1 + d dt)`,
  `c_p = eps0 kappa_p dt^2 / (1 + d dt)`; E-update `Ca, Cb=dt/gamma,
  Cc=1/gamma`, `gamma = eps_inf eps0` (sigma=0 fixture).
- CPML: `(sigma, kappa, alpha, b, c)` from
  `rfx.boundaries.cpml._cpml_profile(n_layers, dt, dx, kappa_max=1.0)`
  (shipped defaults: order=3, R=1e-15, alpha_max literal 0.05).
- Update order exactly as `rfx/simulation.py`: H half-step + psi_h
  correction, then P update from E^n, then E update + psi_e correction.
- 1D face system (6x6: Ez, Hy, P, P_prev, psi_e, psi_h), plane-wave
  `e^{ikx}` on the staggered grid, k dx in (0, pi], 481 points, all 8
  layer coefficient sets (E and H share the same per-layer profile, as
  shipped).
- 2D corner system (9x9: Ez, Hx, Hy, P, P_prev, psi_ez_x, psi_ez_y,
  psi_hx_y, psi_hy_x), (kx dx, ky dx) in (0, pi]^2 on a 61x61 grid, all
  (layer_x, layer_y) pairs.
- Material cases: (i) fixture pole (Q60, eps_inf=4); controls:
  (ii) pole with NO CPML (identity psi, b=1?no — psi absent: pure ADE
  4x4/5x5 system), (iii) CPML with eps_inf=4 and NO pole.
- Also, prediction-only: the same scans with the M2 alpha rule
  `alpha_max = 1.2 * 2*pi * f_top * eps0`, f_top = 7.5e9 (=> 0.5007 S/m,
  same `(1 - rho)` grading) to see whether the rule closes the gap.

Declared thresholds (BEFORE running):

- H1 CONFIRMED if max |lambda| over (layers, k) with pole+CPML active
  exceeds `1 + 1e-6` (observed 3D growth ~ exp(1.35e-4/step) so the
  per-cell rate should be at least of that order), AND both controls stay
  below `1 + 1e-9`.
- **Falsifier F1**: if the pole+CPML scans (face AND corner) stay below
  `1 + 1e-6` everywhere, H1 (per-cell composed-recursion instability) is
  FALSIFIED — the instability then involves grading/interfaces or 3D
  structure not captured per-cell; record that and move on. No matrix
  tweaking, no threshold adjustment after the fact.

M1 is a linear-algebra diagnostic, not the physics attempt; it consumes no
FDTD attempt.

## M2 — the issue's unspent 2026-08-12 pre-declared attempt (ONE attempt)

Script: `validation/research/cpml_pole_pad/factorial_636.py`. Executing
this battery consumes the single attempt; whatever it says is final for
this session (no tuning afterwards).

Fixed knobs (all runs unless stated): dx=1mm, domain 45x39x12 mm,
freq_max=7.5e9, GaussianPulse(f0=3e9, bandwidth=0.8) "ez" field source at
(15mm, 13mm, 5mm), `cpml_layers=12`, 60,000 steps, float32,
`skip_preflight=True` (research script; committed examples never skip),
`subpixel_smoothing=False`, `compute_s_params=False`.

ON variant = pole masks replicated into the pad exactly like the statics
(piggyback on `extend_cpml_pad_materials`, including the #627a hi-face
fallback) AND the CFS alpha rule `alpha_max = 1.2*2*pi*f_top*eps0`
(f_top = freq_max = 7.5e9), same `(1-rho)` grading, monkeypatched into
`_cpml_profile`. OFF control = identical in every knob (alpha rule
included) except poles are NOT extended (shipped mask behaviour).

Four cells (geometry as the lock fixture, slab z 3mm..7mm, unless stated):

- C1 both-face: slab (0,0,3mm)-(45mm,39mm,7mm), eps_inf=4, Lorentz Q60
  (omega_0=2*pi*3e9, delta=omega_0/120, kappa=3*omega_0^2).
- C2 lo-only: slab (0,0,3mm)-(30mm,26mm,7mm) (touches x-lo and y-lo
  only), same material as C1.
- C3 eps_inf=1 Lorentz Q5: eps_r=1, omega_0=2*pi*3e9, delta=omega_0/10,
  kappa=3*omega_0^2, geometry as C1.
- C4 eps_inf=1 Drude: eps_r=1, `drude_pole(omega_p=2*pi*3e9,
  gamma=omega_p/100)`, geometry as C1.

Float64 control: C1-ON repeated with `precision="float64"` (unblocked by
#656 via fd37c62). Prediction under H1: still grows (not a float32
artifact).

Observable (declared): probes at PAD cells, absolute physical
coordinates (pads live at x<0, x>44mm etc.; `position_to_index` maps them
legally): "ez" at face pads x-lo/x-hi/y-lo/y-hi, depths 2, 6, 10 cells
into the 12-layer pad, transverse mid, z=5mm; "ez"+"ex"+"ey" at the four
x-y corner pads, depth (6,6), z=5mm. Envelope = max |value| over each
200-step window across (a) face probes, (b) corner probes, (c) all pad
probes. Growth rate = least-squares slope of ln(envelope) vs step over
the LAST 50% of windows: `g = d ln(max|E|_pad)/dstep`. Plus finiteness
flag. The face-vs-corner split is the free discriminator (which pad
region carries the growing mode); the C1-ON final-state |E| field is kept
as the localization witness.

Vacuum-floor check (alpha-rule cost): pure-vacuum fixture, same domain,
cpml_layers=12, same source, interior probe at (36mm, 19mm, 5mm), 4,000
steps; floor = max|ez| over the last 1,000 steps / peak |ez|, in dB.
Compare shipped alpha (0.05) vs rule alpha. Degradation = floor_rule -
floor_shipped in dB (positive = worse).

**Two-sided falsifier F2 (as declared in the issue, executed verbatim):**

- Fix VIABLE only if: every ON cell (C1..C4, and the float64 control) has
  g <= 0 and stays finite, AND the vacuum-floor degradation <= 3 dB.
  Then deliverable 3 applies (pole extension + alpha rule behind a
  behaviour-compatible default), with the `_vacuum` pole-hole fold and a
  re-baselined stability lock.
- Otherwise: STOP to guards-only — deliverable 4 (preflight advisory on
  dispersive material touching a CPML face on a high-Q-risk config +
  re-baselined stability lock + this note updated with the measured
  envelope). No parameter tuning after the battery has run.

Independent of F2's branch, the stale 8000-step lock is re-baselined to
the step count where the measured separation is decisive (20,000 steps:
0.2145 vs 5.032 on 8 layers), and a minimal committed repro of the
instability itself (both variants, slow-marked) is added — deliverable 1
does not depend on the attempt's outcome.

## M1 outcome (recorded 2026-08-29, before M1b was designed further)

**FALSIFIER F1 FIRED.** Max |lambda| minus 1 across every scanned case
(1D face and 2D corner, shipped and rule alpha, 8 and 12 layers, C1/C3/C4
materials) is at the +1e-12 level — eigen-solver roundoff. The per-cell
composed ADE+CPML update is spectrally stable within the frozen-coefficient
scope. H1 as stated is falsified; the shipped composition is NOT per-cell
unstable. Consistent with the continuous-level Becache-Joly geometric
criterion, which the Lorentz medium satisfies (vp*vg > 0 on both branches).
The instability must involve what the frozen scan cannot see: profile
grading, the interior/pad interface, the pad's outer PEC wall, or modes
with complex transverse wavenumber. No thresholds were changed after the
fact; the 1e-6/1e-9 gates stand as declared.

## M1b / M1c — finite-operator diagnostics (declared BEFORE running them)

M1b: exact FINITE 1D one-step operator, x-line of the fixture: 8-layer lo
pad + 45 interior + 8-layer hi pad, Ez/Hy TM, PEC outer nodes (Ez=0),
shipped graded profile per layer, statics eps_inf=4 everywhere (extended
pads), pole (Q60) mask either interior-only (variant A = shipped) or
extended into both pads (variant B). State [Ez, Hy, P, Pprev, psi_e,
psi_h] per node, dense one-step matrix (~366 dim), exact eigenvalues.
Prediction: B has rho > 1 + 1e-6, A <= 1 + 1e-9.
Falsifier F1b: B staying <= 1 + 1e-6 falsifies the 1D interface/grading
mechanism; then go to M1c.

M1c (only if F1b fires): finite 2D TM corner operator, 61 x 55 grid of
the fixture's x-y cross-section (pads on all four sides, PEC outer ring),
eps_inf=4 + Q60 pole with mask variant A (interior slab only) vs B
(extended into pads, corners included, exactly the array the piggyback
extension produces in 2D), sparse largest-|lambda| eigensolve
(ARPACK) on the ~30k-dim one-step operator implemented in numpy with the
shipped update order/signs. Prediction: B has rho > 1 + 1e-6 and the
eigenvector localizes in the pad; A <= 1 + 1e-9.
Falsifier F1c: if B also stays <= 1 + 1e-6 in 2D, the linear mechanism is
not reachable below 3D (or not by these reduced models); the root-cause
statement then rests on M2's empirical operator measurement (growth rate +
final-state localization) alone, reported as such.

## M1b / M1c outcomes (recorded 2026-08-29, before M2 finished)

**F1b FIRED**: finite 1D operator (grading + interfaces + PEC wall
included) has rho = 1 to machine precision for BOTH variants.
**F1c FIRED**: finite 2D TM x-y corner operator (power iteration, 40k
steps; solver substituted for ARPACK as recorded in the script docstring,
thresholds unchanged) measures rho_est = 0.99998 for both variants —
stable. The linear instability is not reachable per-cell, in 1D, or in
the 2D TM cross-section: it requires the 3D structure (the z-confined
pole slab extended along x/y inside the pad — consistent with the
original #636 observation that a thin-in-z fixture does not reproduce).

## M3 — growing-mode localization addendum (declared BEFORE running)

M2's C1-ON cell (rule alpha, 12 layers) turned out STABLE, so the
declared "C1-ON final state" cannot witness the growing mode. M3 is a
non-gating, characterization-only diagnostic of the KNOWN-unstable
configuration (no verdict, no fix decision, no tuning depends on it):
the scout/premise configuration — 8 layers, SHIPPED alpha (0.05), poles
extended, C1 material/geometry, 20,000 steps — re-run once recording
(a) final-state |E| mass split interior / x-y face pads / corner pads
and argmax, (b) FFT of the late-time pad probe.
Prediction (informational): the growing mode concentrates in the x/y
pads at slab z-levels and its spectrum peaks near the polariton band
edge of eps(omega) = 0 (3.97 GHz for this material; the stable C1-ON
residual already peaked at 3.88 GHz).

## M2 outcome (recorded 2026-08-29) — FALSIFIER F2 FIRED: GUARDS-ONLY

Full numbers in `validation/research/cpml_pole_pad/factorial_636_result.json`
(+ `_f64.json`). Growth rate g = d ln(max|E| over pad probes)/dstep over
the last 50% of 200-step envelopes at 60,000 steps; all runs finite.

| cell | ON (poles extended, rule alpha, L12) | OFF control |
|------|--------------------------------------|-------------|
| C1 both-face Lorentz Q60 eps4 | g = -1.30e-4 (decays) | -4.2e-5 |
| C1 float64 control            | g = -1.30e-4 (matches f32) | — |
| C2 lo-only Lorentz Q60 eps4   | g = -1.29e-4 (decays) | -5.7e-5 |
| C3 Lorentz Q5 eps_inf=1       | g = -2.6e-7 (floor; corner +6.5e-8 also in OFF -> noise) | -1.0e-7 |
| C4 Drude eps_inf=1            | **g = +5.23e-4 DIVERGES** (peak 1.6e11) | -1.9e-4 |

Vacuum floor: shipped alpha -70.1 dB, rule alpha -68.0 dB; degradation
+2.1 dB (within the 3 dB gate — the alpha rule itself is affordable).

C4-ON free discriminators: 59% of final |E| mass in the x/y pads
(face 0.46, corner 0.13), argmax inside the pad, spectrum peak
2.64 GHz — INSIDE the Drude eps(omega)<0 band (omega < omega_p =
2*pi*3e9). The two-sided falsifier fired -> guards-only, no tuning, as
declared. The R2 attempt is spent.

## M3 outcome — growing-mode localization (prediction confirmed)

Known-unstable config (shipped alpha 0.05, 8 layers, poles extended,
C1), 20,000 steps: g = +2.6e-4/step; 58% of final |E| mass in the x/y
pads (face 0.50, corner 0.09), argmax in the x-lo/y-lo pad corner
region; pad |E| z-profile peaks AT the slab's z-interface (k=15,
slab k=11..14); spectrum peak 3.83 GHz — inside the polariton gap
(eps<0 between omega_0 = 3 GHz and the eps=0 edge at 3.97 GHz),
near the predicted band edge. Full numbers in
`validation/research/cpml_pole_pad/localize_636_result.json`.

## Root-cause statement (deliverable 2)

The composed discrete update — ADE pole recurrence + CPML recursive
convolution as shipped — is NOT itself unstable:

1. Per-cell (frozen-coefficient von Neumann, exact shipped coefficient
   formulas and update order): spectral radius <= 1 + 1e-12 for every
   layer, wavenumber, material class (Q60/Q5 Lorentz, Drude), both
   alphas, 8 and 12 layers, 1D face and 2D corner (M1).
2. Finite 1D operator with the graded profile, both interfaces and the
   outer PEC wall: rho = 1 to machine precision, poles extended or not
   (M1b).
3. Finite 2D TM x-y operator (corners, grading, PEC ring): rho < 1 for
   both variants (M1c). Note the extended variant is pole-UNIFORM in
   this plane — no material interface — which is exactly why it is
   stable and why the mechanism was invisible below 3D.

What diverges is an INTERFACE mode: extending the pole into the pad
extends the slab's eps(omega) < 0 band (Lorentz polariton gap
3.00-3.97 GHz; Drude omega < omega_p) into the absorber, and the
structure's boundaries inside the pad (the slab's z-interfaces, pad
corner wedges) then support surface-polariton waves at eps < 0. Both
measured growing modes sit in that band (3.83 GHz Lorentz, 2.64 GHz
Drude), concentrate in the pads, and peak at the slab's z-interface —
and backward surface waves on negative-eps interfaces are the classic
regime where stretched-coordinate PMLs violate the geometric stability
condition (phase and group velocity anti-parallel along the absorber
normal; Becache-Joly-type instability for plasmonic/NIM media). The
z-confined interface is essential (thin-in-z fixtures never reproduced;
1D/2D models without the interface are provably stable), which is why
this evaded plane-wave and low-dimensional analysis.

Consistent corollaries: raising the CFS alpha to the corner rule damps
the NARROW Lorentz gap (C1/C2/C3 all stabilize — the gap-edge mode at
3.8-3.9 GHz appears in the STABLE C1-ON residual too, decaying), but
cannot rescue Drude, whose eps<0 band spans (0, omega_p); and the
shipped statics-only pad is stable simply because a pad without poles
has no eps<0 medium and therefore no surface-polariton branch.

Implication honored by the guards-only outcome: no CPML parameter choice
covered the declared factorial, so pole extension stays out (a stable
pad for eps<0 media needs a reformulated dispersive PML, out of scope);
the shipped behaviour keeps the band-limited eps_inf-matched pad, now
surfaced by the preflight advisory `dispersive_pole_at_absorber_face`.

## Guards landed (deliverable 1 + 4)

- `tests/unit/boundaries/test_cpml_pad_material_extension.py::test_pole_extension_divergence_repro_636`
  (slow lane): minimal committed repro + physics lock, both variants at
  20,000 steps (shipped 0.2145 decays / extended 5.032 grows, measured
  margins in the test); the 8,000-step lock re-scoped as the fast-lane
  shipped-decay canary (its historical numbers were stale post-#655).
- `rfx/api/_preflight.py::_validate_cfg_dispersive_pole_at_absorber_face`
  advisory (code `dispersive_pole_at_absorber_face`), 8 targeted tests in
  `tests/unit/preflight/test_preflight_absorber.py`.
- NOT landed (measured unsupportable): pole-mask extension in any form,
  including behind the CFS alpha rule — C4 red. The `_vacuum`
  pole-fold and alpha_max plumbing drafted during the session were
  reverted with the fix branch; the pole-only-material fallback hole
  (rasterize_grid.py `_vacuum`) remains open and documented in the
  ledger.

## Runtime budget

M1: seconds. M2: ~10 runs x ~2-5 min (156k cells x 60k steps, CPU JAX)
— each under the 20-minute single-measurement limit; no VESSL needed.
