# 2026-08-29 — #636: dispersion poles in the CPML pad, pre-run predeclaration

Written BEFORE any measurement of this work stage, under the repository's
one-clean-predeclared-attempt discipline. Author agent session for backlog
item 636 (the deferred half of #627: extending Debye/Lorentz pole masks into
the CPML pad destabilizes high-Q edge-touching structures).

## R1 — tracked context (premise verification, already done)

Re-measured on `b29f9de` with the lock-test fixture
(`tests/test_cpml_pad_material_extension.py::test_pole_extension_stability_lock`,
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

## Runtime budget

M1: seconds. M2: ~10 runs x ~2-5 min (156k cells x 60k steps, CPU JAX)
— each under the 20-minute single-measurement limit; no VESSL needed.
