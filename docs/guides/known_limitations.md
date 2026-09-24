# rfx Known Limitations

Defects and limits that a user can hit today, on the current `main`. This page
is the companion to the [support matrix](support_matrix.md): that page says what
is supported and within what limits, this one says what is known to be wrong or
unevidenced inside those limits.

Each entry gives the symptom you would see, the mechanism in one line, what to do
about it now, and the tracking issue. **The issue is the evidence**: measured
numbers, the fixture they came from, and the commit they were measured on live
there, not here. An entry disappears from this page when its defect is fixed and
a committed test pins the fix, not when someone believes it is better.

Nothing here is a substitute for the repo's standing rule: a warning's absence is
not an accuracy guarantee, and a preflight pass is not a convergence study.

---

## Distributed runs: reduced-frequency ghost exchange (exchange_interval > 1) is refused

exchange_interval > 1 is refused. With a one-cell ghost layer and the exchange skipped for K-1 steps, each slab updates its seam cells from the neighbour's stale values and injects energy every skipped step.
In a lossless 48x16x16 mm PEC box (dx = 1 mm, float32, 2000 steps) the probe amplitude, relative to the single-device peak, reaches 2.0e2 for K=2 and 2.7e5 for K=4 on two devices, 4.6e7 (K=2) and 5.3e19 (K=4) on four devices, growing exponentially from the first skipped exchange, while K=1 stays within 1.0 of the peak.
This is a scheme instability, not an O(dt*K) boundary error; a K-cell overlap would be required to skip exchanges.
Use `exchange_interval=1`.

## Ports and extraction

**The coax→microstrip transition over-reads power by about a factor of three.**
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
above 17 GHz recorded in `validation/crossval/07_sheen_lpf.py`.
#838 was closed as not planned before 2.0 (PI decision, 2026-09-20); this is a standing limitation.

**The microstrip S-matrix can come back non-passive, and says so.**
`compute_msl_s_matrix(...)` returns the S it extracted; it no longer projects it
onto the passive set by default, so that the S a user reads and the S a gradient
differentiates are one function. When a bin's largest singular value exceeds 1
(beyond float32 rounding) the call warns with the bin count and the worst value,
and `result.sigma_max_excess` carries the per-bin amount. A passive structure
cannot do that: read it as a record that ended before ring-down or a mesh too
coarse for the geometry (`settling_db`, `reliable`), not as gain.
`enforce_passivity=True` returns the projected matrix instead. On a reflecting fixture
the raw S is gated: the microstrip chain battery holds its maximum column power at
or below 1.02 on the open-stub notch at its 25 µm claims rung (1.0103 measured; the
thru line 1.0030), in `tests/oracle/test_msl_chain_battery.py`. Coarser meshes are
reported, not gated.

**The fitted microstrip propagation constant sits 1.0 to 1.3 % above the
Hammerstad–Jensen closed form on every in-band bin.** On a 600 µm trace over
250 µm of RO4350B a float64 refit of the probe phasors reads 1.32 … 1.33 % with
five cells under the strip and 0.97 … 1.04 % with ten, over 3.0–4.5 GHz; the
offset is not attributed and no trend is claimed, because the strip is one cell
thick in each run. It is the FITTED `beta`, a diagnostic that does not enter S
(pinned by a test), so it is not a bias on the phase of an S-parameter. Stated
in the Microstrip-line row of the [support matrix](support_matrix.md); #830 is
closed as characterized.

**Microstrip `Z0` and `beta` are unreadable when the probes sit near a
reflector.** The N-probe fit rides the standing wave instead of measuring the
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

## Absorbing boundaries

**On the distributed lanes and with `solver='adi'`, a magnetic face is not a magnetic wall.**
A face declared `pmc` is solved as a magnetic wall only on a single-device Yee run. On `run(devices=...)` and
`forward(distributed=True)` with no absorbing face the plane is shorted (tangential E held at zero); with absorbing
faces the cells next to it absorb, and a source one cell off the plane reaches the volume 65–75 dB below the
single-device run. `solver='adi'` solves the face as an electric wall. A half-model on these lanes is a different
structure from the one declared. Use a single-device Yee run for a symmetry plane; there the wall sits half a cell
inside the declared face, which #1221 also fixes.
→ [#1221](https://github.com/bk-squared/rfx/issues/1221)

**With the mesh as a design variable, a ground plane still ends at the absorber.**
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

## Gradients and optimization

**A gradient can be wrong by tens of percent on a record whose value is
converged.** Every frequency-domain quantity rfx differentiates is a DFT of a
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

## Examples and validation coverage

**A shipped paper example's headline numbers are not WR-90 numbers.**
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

**Committed examples are verified at build time, not by their results.** The
example-fidelity gate builds every reachable example and pins the preflight rows
it emits; it does not step time, and it does not check that any example's printed
result is right. Six tutorials plus `hello_world` are additionally run end to end
with physics assertions. Everything else in `examples/` and `validation/` is
covered only as far as its build.
→ [#737](https://github.com/bk-squared/rfx/issues/737)

**Patch-antenna cross-validation has no trustworthy quantitative accuracy
baseline.** The integration case is a coarse same-geometry envelope and says so;
the case that was supposed to carry the quantitative claim does not currently
close it.
→ [#715](https://github.com/bk-squared/rfx/issues/715)

**The weekly scientific-validation lane is red.** Two shards fail on the current
schedule. Until it is green, a claim of "the weekly suite passes" is not
available.
→ [#1022](https://github.com/bk-squared/rfx/issues/1022)

---

## What this page is not

It is not the issue tracker: an entry here exists because a *user* can hit the
thing, not because work is scheduled on it. Internal bookkeeping, campaign
tracking, decisions pending, and research narrative are deliberately absent — for
open work see the
[issue tracker](https://github.com/bk-squared/rfx/issues), and for what is
supported at all see the [support matrix](support_matrix.md).
