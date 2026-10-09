# rfx Known Limitations

Defects and limits that a user can hit today, on the current `main`. This page
is the companion to the [support matrix](support_matrix.md): that page says what
is supported and within what limits, this one says what is known to be wrong or
unevidenced inside those limits.

Each subject has its own heading and a `Tracker:` line: issue/PR numbers, or
`none — accepted limitation (<reason>)` for a deliberate untracked limit.

Each entry gives the symptom you would see, the mechanism in one line, what to do
about it now, and the tracking issue. **The issue is the evidence**: measured
numbers, the fixture they came from, and the commit they were measured on live
there, not here. An entry disappears from this page when its defect is fixed and
a committed test pins the fix, not when someone believes it is better.

Nothing here is a substitute for the repo's standing rule: a warning's absence is
not an accuracy guarantee, and a preflight pass is not a convergence study.

## General constraints

### Placing a solve on CPU in a process whose default backend is GPU

Tracker: #952

The absorber write-back is selected from the process's default backend at trace
time. A process whose default backend is GPU but which places a solve on its CPU
device therefore uses the GPU form there. A plain call and a `jax.jit` call of
the same solve can then differ by up to 63 float32 ULP at the probe record peak
(5e-6 of the peak): 31–63 ULP measured on a uniform mesh for records of 120 to
480 steps on nine machines, unchanged from 480 to 3840 steps on the one where
longer records were run; a graded-mesh case read 0. Results on the GPU, and in a
CPU-only process, are not affected. Use the GPU device or a CPU-only process to
avoid this spread.

### Lossy sheets and sheet_effective_size

Tracker: #1138

Lossy sheets (`surface_impedance_f0`, or sigma below the PEC threshold) are not judged by `sheet_effective_size` (`rfx/materials/thin_conductor.py::ThinConductor.is_pec`, `rfx/preflight/realization.py::_CampaignStaticsContext.pec_entries`).

---

### Experimental ADI

Tracker: #1487

`solver="adi"` is outside the 2.0 supported scope.
  `boundary="cpml"` is refused: it is an unmatched conductivity sponge
  (−10.4 dB reflection at 10 GHz). Use Yee with CPML for open structures,
  or ADI with `boundary="pec"` for closed cavities.

### Distributed runs: reduced-frequency ghost exchange (exchange_interval > 1) is refused

Tracker: none — accepted limitation (refused at default: tests/unit/runners/test_distributed.py::TestExchangeInterval::test_interval_refused)

exchange_interval > 1 is refused. With a one-cell ghost layer and the exchange skipped for K-1 steps, each slab updates its seam cells from the neighbour's stale values and injects energy every skipped step.
In a lossless 48x16x16 mm PEC box (dx = 1 mm, float32, 2000 steps) the probe amplitude, relative to the single-device peak, reaches 2.0e2 for K=2 and 2.7e5 for K=4 on two devices, 4.6e7 (K=2) and 5.3e19 (K=4) on four devices, growing exponentially from the first skipped exchange, while K=1 stays within 1.0 of the peak.
This is a scheme instability, not an O(dt*K) boundary error; a K-cell overlap would be required to skip exchanges.
Use `exchange_interval=1`.

## Ports and extraction

### A microstrip strip drawn past an MSL port to the domain edge is an open stub that can short the port.

Tracker: #1512

rfx ends a two-conductor line at its port: the piece of strip between the port plane and an
absorbing face ends at the absorber's entrance, so it is an open-ended stub. Near the frequency
where that stub is a quarter guided wavelength long, its input shorts the port, the line between
the ports becomes a low-loss resonator and S is wrong. On a 24 mm, 50 Ω line (εr 3.66, h 254 µm,
20–40 GHz) a 2 mm overhang behind each port gave |S11| −2.8 dB and |S21| −4.6 dB near 20 GHz; with
the strip starting at the port planes the same line reads |S11| below −36 dB and |S21| within
±0.25 dB. The bad band moves with the overhang length, not with the mesh, the absorber or the
record length. The refusal judges the stub's effective length: the realized node-to-node
overhang plus the open-end extension (Hammerstad's closed form for a microstrip; none is added
for a coax pin). It reports the realized and declared lengths, the extension and the effective
length, and prints the port's realized grid-node coordinate along its axis.
Start the signal strip at that coordinate (the port's grid node), so it covers the port node
and nothing behind it. The absorber pads lie outside
the declared domain, so drawing the strip "into the absorber" is not possible today.
→ [#1512](https://github.com/bk-squared/rfx/issues/1512)

### The default wire port is a mesh-sized probe.

Tracker: #1138

`add_port(..., extent=...)`
with `radius=None` acts as a probe of radius approximately `0.20 * dx` on a
square transverse mesh. Refinement therefore changes its series inductance;
it is not a fixed-radius physical pin. Opt in to `radius=` in metres to use
the local thin-probe model, within `radius <= 0.20 * dx` and a locally uniform
square transverse stencil and uniform spacing along the pin. Axial grading
across the port refuses this model. It is carried by single-device,
nondispersive 3-D second-order Yee `run()` and `forward()`; other solvers,
Debye/Lorentz, tensor/design-box updates and traced mesh metrics refuse it.
Resolve larger wire ports geometrically with a coax feed or a volume wire.

### Positive subcell PEC filament radii are unsupported.

Tracker: none — accepted limitation (refused at default: tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan)

A PEC `PolylineWire`
with `0 < a < 0.50 * d_min` refuses before stepping. Resolve the wire as a
volume: refine the mesh until `a >= 0.50 * d_min`, using the smallest local
cell at its vertices. The existing volume-wire rule and legacy `radius=0`
filament ownership remain; zero does not declare a physical wire radius.
A positive-radius PEC `PolylineWire` also refuses when the mesh cell sizes
are traced; its geometric radius check requires concrete cell metrics.
The attempted filament correction failed its independent uncorrected-main
reference at `a/d=0.200`: for a=0.0375 mm and branch d=0.75 mm, the maximum
8–14 GHz impedance error was 4.93%, and the 0.75→0.375 mm X change at
11.3 GHz was 2.58 Ω (limits 2% and 1 Ω). The correction is removed.
That one-edge feed has nonuniform longitudinal current, so the full-height
wire-port Hankel oracle does not establish its accuracy. The port radius
model and its full-height oracle remain separate.

### Automatic preflight for compute_coax_msl_transition

Tracker: #1138

`compute_coax_msl_transition` checks sheet size through the shared preflight
verdict by default, unless `skip_preflight=True`. The other preflight families
do not run on this path.

### The coax→microstrip transition over-reads power by about a factor of three.

Tracker: #1479

Measured twice independently on the MSL port's power-wave normalization: the
returned matrix's own MSL-driven column power runs about 3x the incident power
(the coax-driven column reads 0.379 / 0.379 / 0.367 on the same run), and a
six-face Poynting flux box on the same run reads the same factor. The matrix is
therefore not passive — `compute_coax_msl_transition(...)` refuses it by default
(`strict_passivity=True`, raising `ValueError`); pass `strict_passivity=False` to
get the diagnostic matrix back with a `UserWarning` — and transition
S-parameters from that path are not usable as an absolute power reference.
Scope: cross-family transition lanes are not pursued further before
2.0, so this over-read will not be attributed or corrected; use the
single-family extractors separately (`compute_msl_s_matrix(...)`, the coaxial
two-port and line-reflection lanes) for a result you can cite.

The single-family microstrip lane does not carry this ~3x: the V·I-over-Poynting-flux
oracle (`scripts/diagnostics/msl_vi_flux_oracle/msl_vi_flux_oracle.json`) bounds its power
scale against the true flux to about 1 % on both committed meshes, and a 3x over-read cannot
hide inside 1 %. That lane has its own limits — see the
[S-parameter support matrix](sparameter_support_matrix.md) — including a raw passivity excess
above 17 GHz recorded by the Sheen low-pass filter case.
#838 was closed as not planned before 2.0 (PI decision, 2026-09-20); this is a standing limitation.

### The microstrip S-matrix can come back non-passive, and says so.

Tracker: none — accepted limitation (warns at default: tests/unit/autodiff/test_msl_forward_identity.py::test_a_nonpassive_extraction_is_named_bin_by_bin_in_one_warning)

`compute_msl_s_matrix(...)` returns the S it extracted; it no longer projects it
onto the passive set by default, so that the S a user reads and the S a gradient
differentiates are one function. When a bin's largest singular value exceeds 1
(beyond float32 rounding) the call warns with the bin count and the worst value,
and `result.sigma_max_excess` carries the per-bin amount. A passive structure
cannot do that: read it as a record that ended before ring-down or a mesh too
coarse for the geometry (`settling_db`, `reliable`), not as gain.
`enforce_passivity=True` returns the projected matrix instead. On a reflecting fixture
the raw S is gated: the microstrip chain battery holds its maximum column power at
or below 1.02 on the open-stub notch at its 25 µm claims rung, in
`tests/oracle/test_msl_chain_battery.py`. Read the current fixture for the
measured notch and thru values. Coarser meshes are
reported, not gated.

### The fitted microstrip propagation constant sits 1.0 to 1.3 % above the Hammerstad–Jensen closed form on every in-band bin.

Tracker: none — accepted limitation (diagnostic, does not enter the result: tests/unit/sparams/test_msl_fitted_beta_does_not_enter_s.py::test_s_is_bit_identical_when_the_fitted_beta_is_wrong)

On a 600 µm trace over
250 µm of RO4350B a float64 refit of the probe phasors reads 1.32 … 1.33 % with
five cells under the strip and 0.97 … 1.04 % with ten, over 3.0–4.5 GHz; the
offset is not attributed and no trend is claimed, because the strip is one cell
thick in each run. It is the FITTED `beta`, a diagnostic that does not enter S
(pinned by a test), so it is not a bias on the phase of an S-parameter. Stated
in the Microstrip-line row of the [support matrix](support_matrix.md); #830 is
closed as characterized.

### Microstrip `Z0` and `beta` are unreadable when the probes sit near a reflector.

Tracker: none — accepted limitation (diagnostic, does not enter the result: tests/unit/sparams/test_msl_power_normalization.py::test_reference_metadata_is_neither_fitted_z0_nor_the_50_ohm_load)

The N-probe fit rides the standing wave instead of measuring the
line: across the three board runs tabulated in #726 the fitted `Z0` reached
2.4x the analytic value and its ripple tracked `|S11|` in dB at r = 0.71-0.77,
while `reliable` was True on 100 % of in-band bins. Those three runs carry no
VESSL id in the record. `S` itself is normalized with the analytic
Hammerstad-Jensen `Z0` and moved far less, by 0.009 dB in one controlled
comparison — one fixture, one bin, an arm-to-arm difference, not a bound on `S`.
Gate on `probe_clearance` and `beta_railed`, not on `reliable`; the
fix is a longer uniform feed or a moved reference plane. Every warning about
this condition now carries that measurement, and `docs/guides/sparameter_support_matrix.md`
has the full reading guidance. Settled in #726 (closed): the guard and preflight
used to contradict each other about this, and the measurement decided it.

### A source on a smoothed or conformal edge is injected with a coefficient that differs from the field update's.

Tracker: #1524

The field increment a source adds on its edge is the update's coefficient times the drive. One place still computes that coefficient apart from the update: a rectangular-waveguide port or plane-wave source whose edges touch `subpixel_smoothing` or conformal-PEC cells. The drive on those edges is scaled by the ratio of the two coefficients; S-parameters normalised to the incident wave are not known to move, absolute field and probe amplitudes are. Until it is fixed, read absolute amplitudes from such a run as uncalibrated, or declare the source on a plain edge.

---

## Ring-down completion and the early stop

### A pair of modes that no window of the record separates is completed as one mode, and every witness agrees.

Tracker: #1514

`run(ringdown=...)` and
`forward(ringdown=...)` identify the ringing's poles on the record's second
half. Two resonances closer than the record can resolve come back as one
blended pole, and the completion from any window of that record makes the same
blend. The error witness WE then agrees while the completed S can be several
percent off near the pair. What to do: record longer. In the measured case, a
pair 0.03 % apart was resolved at about a fifth of its decay time.
#1381 was closed as a stated limit (PI decision, 2026-09-30); this is a standing limitation.

### The early stop can end the run before a weak unresolved high-Q pair is resolved.

Tracker: #1514

The rule is `run(..., until_identified=True)`: stop once WE agrees
twice and the record is half the decay time of the slowest mode that moves S.
A weakly coupled pair of very high-Q modes (Q about 5e4, each feature 0.02–0.03
in |S|, 40–60 MHz apart) beats inside a short record. The record reads the beat
as fast decay, the blended pole's weight falls below the bar, and the floor
does not hold. Measured on a synthetic one-port: the stop fires at 2250 steps
with the completed S up to 3e-2 off at the pair while WE reads about 1e-4.
Stronger or single weak modes do hold the stop. What to do: where weak high-Q
features matter, use a fixed record (`run(n_steps=..., ringdown=...)`) long
enough to resolve them. Pinned as a strict xfail in
`tests/unit/sparams/test_ringdown_early_stop.py`.

### A plain record shorter than a weakly coupled resonance's decay misreads S near it on any port, and the end-of-run witness can miss it.

Tracker: #1514

A lossless 50 × 50 × 25 mm PEC box with a one-cell 50 Ω
port rings on TM110 at 4.148 GHz with Q ≈ 2000 (amplitude e-fold 153 ns); the port is its only loss.
On a plain 12 ns record, |S11| reads 0.975–1.009 around the mode (largest deviation 0.025);
24 / 60 / 120 ns records: deviation 0.042 / 0.032 / 0.026. The end-of-run witness read −64 dB
and passed: its reference peak was the direct response, and the weakly coupled mode rings below it.
On the same 12 ns record, one-cell wire-port completion restores |S11| to 1 within 1e-3 (measured about 1e-6) with passing witnesses
(`tests/unit/sparams/test_sparam.py::test_wire_port_pec_cavity_s11_around_the_first_mode`).
`run(ringdown=...)` currently refuses one-cell lumped ports.
What to do: record well past the slowest mode's decay, or use a one-cell wire port
(`add_port(..., extent=dx)`) with `run(ringdown=...)`.
#1255 was closed as a stated limit (PI decision, 2026-10-01); this is a standing limitation.

## Solver lanes

### ADI (`solver='adi'`) refuses a dielectric or conductivity interface until 2.1.

Tracker: #1373

Its update reads each cell's own eps_r and sigma where the Yee lanes take the four-cell edge mean, so a material face sits half a cell off (+2.2 % on a slab-loaded PEC cavity's first resonance at 12 cells per loaded wavelength). A homogeneous fill runs (as a declared fill or a scalar override); a traced array override is refused because it cannot be inspected. Use `solver='yee'` otherwise.
→ [#1373](https://github.com/bk-squared/rfx/issues/1373)

## Absorbing boundaries

### Magnetic faces require the single-device second-order Yee image.

Tracker: #1221

Uniform and graded run/forward place the wall on the declared E-node face.
Distributed kernels, subgridded and ADI lanes refuse magnetic faces.
Waveguide-port execution also refuses magnetic faces: its aperture mode solver
implements PEC walls only. The fourth-order stencil refuses them because
its far neighbors have no magnetic image. Subpixel smoothing refuses a material surface within half
an adjacent cell of a magnetic face when the smoother would need the
material's even extension; extending the material at least a cell beyond
the face is supported. These refusals remain until the corresponding
kernels and smoothing extension are implemented.
→ [#1221](https://github.com/bk-squared/rfx/issues/1221)

### With the mesh as a design variable, a ground plane still ends at the absorber.

Tracker: #1230

A conductor drawn to an absorbing boundary is continued through the absorber, so
a grounded board stays grounded inside it. When any mesh axis is traced
(`dz_profile` and its siblings under `jax.grad`), nothing is continued and a
`UserWarning` says so: the run then has the older behaviour, in which a lossless
patch on a grounded substrate gained energy with six absorber layers or fewer
(0 dB settling, +5.8e-4 per step on the six-layer rig, against −44.8 dB with the
ground continued). Use eight or more absorber layers for such a run, or draw the
ground past the domain face by the absorber thickness, which fills every absorber
cell except the outermost row on the +x and +y faces.
→ [#1230](https://github.com/bk-squared/rfx/issues/1230)

### A plane wave requires compatible transverse faces and an invariant absorber replacement.

Tracker: #1221

On the uniform mesh at normal incidence, dispatch refuses absorbing transverse faces for a TF/SF plane wave when
realized materials, conductors or localized field updates vary along that axis,
including in the pads. The refusal also applies with `skip_preflight=True`.
For an array, declare its axes periodic; for a finite scatterer use
`closed_box=True` or `rfx.rcs.compute_rcs`. For propagation along x with E along z, the admitted wall
pairs are PMC on y and PEC on z; either pair may instead be periodic. A wall
pair keeps the legacy wrap only for an invariant model; finite transverse
structure with those walls is refused. UPML remains unsupported for TF/SF.
The uniform plane-wave lane retains its existing padded periodic wrap for an
invariant absorber pair and reports the affected faces in preflight. Its grid
and period do not change. Traced material overrides retain the legacy operator
with a warning when invariance cannot be judged. At oblique incidence the tilt
axis keeps its operator and is reported as not judged for invariance.
The graded mesh keeps its transverse absorbers under a plane wave: no wrap and
no refusal from this rule (the experimental operator remains a PR3b item).
→ [#1221](https://github.com/bk-squared/rfx/issues/1221)

### Full-aperture guides on a graded mesh require explicit transverse boundaries.

Tracker: #1221

The non-uniform and distributed non-uniform paths refuse a full-aperture waveguide
when `boundary` was omitted, including with preflight skipped. Their default keeps
transverse absorbers, so the port would not be in a guide. For an x-directed guide,
write `boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}`; substitute the port axis for
other directions. Explicit absorbers retain the open cross-section. Uniform and
graded builders can still differ by one transverse node when the width is not a
whole number of cells; matching their boundary declarations does not remove that
difference. → [#1221](https://github.com/bk-squared/rfx/issues/1221)

## Gradients and optimization

### A gradient with respect to conductivity does not settle with record length on lossless or nearly lossless design cells

Tracker: #1424

`jax.grad` through `forward()` admits a conductivity design. Where T·σ/ε on
those design cells is not large, the gradient depends on the record length's
phase at each frequency bin: a static sensitivity remains although the field
decays. Measured at σ = 0, the oscillation is 8–11 times the settled value;
at T·σ/ε = 0.34 / 3.4 / 68 it is 0.70 / 0.12 / 9e-8 of that oscillation.
A longer record does not help. `gradient_record_length_witness` detects it;
`gradient="adjoint"` refuses such a design and returns settled-spectrum
gradients only.

### Port-only ring-down completion can miss the gradient of a weakly coupled high-Q resonance.

Tracker: #1514

A pole and the completed S-parameter can be accurate while
its material derivative is tens of percent wrong at the resonance bin. Add
interior field channels with
`RingdownSpec(identification_probes=(((x, y, z), "ez"),))` on `run()` or
`forward()`. They inform pole identification; S still uses the port V/I
residues. The report lists snapped identification channels and per-pole
residue/RMS shares as information, without an observability threshold or a
W0-style accumulator witness for these channels.

Judge completed gradients with `gradient_witness(g, g_early, ringdown=result.ringdown,
bin_axis=k)` when axis `k` carries frequency: it normalizes each bin of each
parameter leaf independently and returns the worst bin, so pass only the bins
your objective reads (a near-zero gradient at an unused band edge reads large). Exactly zero reference
bins are skipped and counted. Without `bin_axis`, normalization is per leaf;
a scalar objective's gradient is already per objective. A passing value witness
does not establish a passing gradient witness. Where the early-start witness
cannot be formed, compare against a record 1.5–2 times longer.
The weak-port cavity contract is in
`tests/unit/sparams/test_ringdown_identification.py` (#1419, fixed by the
identification-probe option; port-only identification remains the default).

### A gradient can be wrong by tens of percent on a record whose value is converged.

Tracker: #1514

Every frequency-domain quantity rfx differentiates is a DFT of a
finite time record. A structure still ringing when the record ends leaves a term
in every bin whose phase is `(ω − ω_r)·T`; a parameter that moves the resonance
spins that phase, and because the phase grows with the record length the spin
lands in the derivative magnified while staying nearly invisible in the value.
Measured on a grounded-slab patch (graded z, float32) fed by a 50 Ω wire port: a
2200-step record settling to −39.7 dB — a pass on the −40 dB rule — gave |S11|²
within 0.99 % of a converged record, while `d ln|S11|² / d ln εr` was out by
9.7 % at 5.75 GHz and 23 % at 6.5 GHz, and a local-permittivity parameter by
8.7 % and 38 %. At 4400 steps (−75 dB) the worst was 2.1 %. The same board
driven by a soft source instead of a port settles far more slowly (−40.9 dB in
6000 steps against −101 dB for the port); there `d ln U(0) / d ln εr` read
5.79 against a converged 6.80 and a pattern-ratio gradient 0.331 against 0.070.
In every measured case the sign of the slope was right and its size was not — an
optimizer reading them takes steps of the wrong length, and a gate on a gradient
value passes or fails on the record length rather than on the physics.

The committed reproduction is a 30×20×20 mm lossy-filled PEC cavity
(`tests/unit/autodiff/test_gradient_record_length_witness.py`): at 1600 steps it
has decayed 46.7 dB, the DFT power a user reads is converged to 0.399 %, and the
gradient with respect to the fill permittivity is 11.0 % from its converged
value.

Neither existing check sees it. `settling_verdict` scores the end of the record
against its peak, which is a statement about the value. AD against a finite
difference agrees to the precision floor at every record length, because both
sides differentiate the same truncated record. Score a gradient you intend to
report or optimize with `gradient_record_length_witness(objective, params,
n_steps, tol=...)`: it takes the same gradient from a record `factor` times
longer (default 2) and reports, per frequency bin,
`‖g_long − g_short‖ / ‖g_long‖` over the whole parameter vector together with
the cosine between the two — so a change of step length reads differently from a
turn of the descent direction. Complex observables (an S-parameter, a DFT
phasor) are compared as complex sensitivities, since a parameter that rotates a
phasor at constant magnitude moves only its imaginary part. A per-element table
comes with the result to locate where a failure sits; it is not the verdict,
because an element carrying a millionth of the dominant sensitivity moves by
100 % on its own rounding. The witness does not replace the −40 dB rule — a
record too short by more than `factor` can still pass it, because both arms then
carry a similar leftover — and it has no default tolerance, because the right bar
depends on the structure's Q and on what the gradient is for.
Measured in #1181; the witness above landed with #1186.

### An optimization on a fixed record can walk into a design that the record does not settle.

Tracker: #1514

The record length is usually chosen so that the starting design
settles. During the descent the optimizer can build a weakly coupled, high-Q mode:
for example a mode that the structure's symmetry keeps orthogonal to the port mode,
coupled only through a small asymmetry the descent itself introduces. Its ringing
outlasts the record, the truncated DFT reads a response the settled structure does
not have, and the objective can keep improving on the truncated record while the
settled response does not. A synthetic case: a dielectric insert in a rectangular
waveguide whose permittivity varies only along the guide keeps every higher mode
across the broad wall orthogonal to the dominant drive; once the permittivity may
also vary across the wall, a higher mode held inside a high-permittivity region
can couple to the drive very weakly and ring far longer than the record.

What to do: judge the per-bin settling witness (`settling_witness` on the
S-parameter result) on the final design, and periodically during the descent, on
a concrete solve rather than inside the traced objective. When it fails, lengthen
the record if the decay is within reach; when the decay time is far longer than any
affordable record, lengthening does not help, and constraining the design space so
that the mode cannot be excited (for instance tying parameters so the design keeps
the symmetry that makes the mode orthogonal to the drive) removes the cause. This
is a property of optimizing on a truncated record, not a solver defect. At default
settings nothing flags it during the descent, so it is tracked with the same cause
as the witness blind spot near weakly coupled resonances.

## Examples and validation coverage

### A shipped paper example's headline numbers are not WR-90 numbers.

Tracker: #1138

`validation/tmtt_paper/waveguide_dielectric_taper.py` declares 22.86 × 10.16 mm.
Its default SMOKE lane meshes at a commensurate dx = 1.27 mm and realizes WR-90
exactly. Its paper lane does not: dx = 0.5 mm realizes 23.00 × 10.50 mm
(46 × 21 cells, +0.61 % / +3.35 %) and the dx = 0.25 mm production mesh realizes
23.00 × 10.25 mm (92 × 41 cells, +0.61 % / +0.89 %). Both have a TE10 cutoff of
6.517 GHz against the declared 6.557 GHz. The −26.7 dB and −38.0 dB figures the
docstring quotes were measured on those guides.

Decided 2026-09-19 (#1122, closed): disclosed, not corrected — no erratum and no
re-mesh. The file states the realized guide in its docstring, in its README entry
and in the result block the run prints, and derives every analytic reference from
the realized walls, so a reader meets the right structure beside every number.
The SMOKE half was #1100.

### Committed examples are verified at build time, not by their results.

Tracker: #737

The
example-fidelity gate builds every reachable example and pins the preflight rows
it emits; it does not step time, and it does not check that any example's printed
result is right. Six tutorials plus `hello_world` are additionally run end to end
with physics assertions. Everything else in `examples/` and `validation/` is
covered only as far as its build.
→ [#737](https://github.com/bk-squared/rfx/issues/737)

### The RT5880 patch-antenna accuracy figure is not claimed for 2.0.

Tracker: #1471

The patch-antenna accuracy figure of 0.097 % is not claimed for 2.0 because
its record length predates the current settling witness. It is being
re-measured in [#1471](https://github.com/bk-squared/rfx/issues/1471).

### The weekly scientific-validation lane is red.

Tracker: #1501

The latest weekly run, [37322809911](https://github.com/bk-squared/rfx/actions/runs/37322809911)
on 2026-10-05, failed `slow-tests (3)`, `slow-tests (4)` and `weekly-a6000-lane`.
That run does not support a claim that the weekly suite passes.
→ [#1501](https://github.com/bk-squared/rfx/issues/1501)

---

## What this page is not

It is not the issue tracker: an entry here exists because a *user* can hit the
thing, not because work is scheduled on it. Internal bookkeeping, campaign
tracking, decisions pending, and research narrative are deliberately absent — for
open work see the
[issue tracker](https://github.com/bk-squared/rfx/issues), and for what is
supported at all see the [support matrix](support_matrix.md).
