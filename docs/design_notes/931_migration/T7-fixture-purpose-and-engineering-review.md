# Decide what the two numerical locks should protect

Date: 2026-09-08. Reviewed branch: `feat/931-slow-regressions`,
`4f0b3b1e2c92d63ad12046fb7d8b53e98fc62725`. **Proposal for PI decision.**
Only this note changes. No code, fixture, tolerance, or baseline is changed.

My recommendation is to **keep the corrected 254 µm microstrip as a legitimate
idealized engineering sample, qualify its physical checks, and then re-pin its
end-to-end drift lock**. Keep the historical replay dataset on its historical
geometry. **Do not simply re-pin V173A as an antenna-physics test.** Its foil
model and measurement need redesign; retain its old record as migration history
or a clearly labelled software continuity sentinel if that coverage is wanted.

Neither lock can decide whether a changed answer is an improvement. That is a
reasonable limitation for a change detector, but an inadequate contract for a
physics validator. Neither executable assertion is actually bit identity.

## Evidence boundary

Three kinds of evidence are distinguished throughout:

* **User-supplied measurements:** VESSL `369367259538`
  (`rfx-931-v173a-lock-evidence`, rc 0) and `369367259539`
  (`rfx-931-msl-lock-evidence`, rc 1). I did not retrieve their full remote logs
  or run either simulation again. The quoted V173A preflight excerpt is:
  `geometry[1] 'metal' (Box) drawn 1.5 mm -> 2.5 mm on a 1 mm mesh`,
  realized walls `1 mm and 2 mm`, worst residual `500 um = 50.00% of the
  extent`, `df/f ~ 50.00%`. This is the supplied excerpt, not a claim to have
  inspected every warning from that job.
* **Repository evidence:** source at the SHA above, the normative
  [ownership contract](../20260906_plan_realign_lattice_ownership.md), and
  [prior evidence note](T7-slow-interaction-stop.md), including its archived
  [full ten-bin complex trace](slow-interaction-evidence/baseline-summary.json)
  and [structural witness](slow-interaction-evidence/structural-witnesses.json).
  These are historical measurements, not newly executed physics results.
* **New lightweight checks:** read all 40 archived complex entries and the
  committed golden; calculated the table below; assembled the current MSL
  geometry without stepping fields. Used `PYTHONPATH=$PWD JAX_PLATFORMS=cpu`,
  one CPU affinity, one BLAS/OpenMP thread, and a 40-second timeout. Assembly
  returned shape `(183,53,30)`, sheet z = 254 µm, y bounds
  1185.333333… and 1778 µm, width 592.666667 µm. An initial helper import typo
  stopped before assembly; the corrected invocation completed. No pytest,
  FDTD solve, or external-solver experiment was run in this discussion.

The primary checkout's private `docs/agent-memory/development_methodology.md`
§2.2 says pure code-motion refactors require a before/after bit-identical
comparison. Its `rfx-known-issues.md` entry dated 2026-07-13 says V173A's earlier
red “was NOT solver drift”: Harminv's decimation default had changed. This
proposal is consistent with both: preserve strict same-problem refactor checks,
but distinguish a deliberate problem change from a numerically neutral refactor.
These historical lessons were checked against the current harness's explicit
`decimate=False`, rather than used as current physics evidence.

## What each test is for, and what would be lost

| Test | Property it actually protects | What it does not establish |
|---|---|---|
| V173A slow lock | Continuity of the composed geometry/material/PEC/CPML/time-step/probe pipeline on a small resonant problem, through a pinned Harminv instrument and FFT statistic. Without it, changes affecting this particular composition can escape smaller tests. | Accuracy of an antenna resonance, identification of its physical mode, calibrated return loss, radiation efficiency/pattern, or convergence. |
| MSL historical E2E lock | Cross-version continuity of the entire live source → FDTD → DFT planes → V/I extraction → wave assembly/de-embedding → default passivity projection pipeline. It can detect phase, sign, port ordering, and normalization drift that broad magnitude bounds miss. | Physical accuracy, correct gradients, or a verdict that any observed change is unintended. |
| MSL fixed-field replay tests | Numerical equivalence of JAX assembly to the NumPy golden on identical captured inputs, plus a bound on f32/f64 assembly differences. | Live field generation or the physical validity of the captured board. |
| MSL integration / length invariance | Broad near-through transmission, plausible impedance and reflection; stability of extracted impedance across line lengths. | Pointwise complex agreement over the whole spectrum or independent accuracy of every extraction stage. |

The first two share the general purpose of detecting end-to-end drift, but
exercise different compositions: V173A has no MSL port or S-matrix assembly;
MSL has no antenna-mode identification. They are not interchangeable. Within
MSL, replay and E2E overlap in assembly coverage; only E2E steps the fields.
The synthetic AD test guards a responsive gradient path, a separate property.

Source anchors at the reviewed SHA: V173A harness
`scripts/harnesses/v173a_physics_equivalence.py:59-123`; MSL replay
`tests/unit/autodiff/test_msl_sparam_ad.py:125-278`, synthetic AD
`:499-536`, and E2E contract `:544-617`.

Existing tests already provide parts of the desired physical coverage:

* `tests/contracts/test_lattice_ownership_contract.py:106-272` pins slab faces,
  thickness, sheets and footprints directly. The cavity oracle
  `tests/contracts/test_lattice_ownership_cavity_oracle.py:112-149` tests drawn
  cavity frequency and equivalent sheet/volume walls. A synthetic off-grid
  object can be entirely sensible for this geometrical purpose.
* `tests/locks/test_patch_edgefed_resonance_harminv.py:47-70` explicitly uses
  spatial mode identity instead of amplitude rank; `:87-108` requires settling
  and a separate geometry witness. Its signed frequency offsets remain
  characterized locks, not exact antenna oracles.
* `tests/locks/test_patch_edgefed_s11_passivity.py:85-106` separates port-plane
  matching from modal resonance. The dielectric cavity control in
  `tests/oracle/test_patch_cavity_eps_oracle.py:9-23` provides an analytic
  bulk-medium witness, but cannot replace an open-antenna/CPML check.
* `tests/locks/test_refplane_port_waves.py:939-1045` contains transmission,
  reciprocity, line-constant, de-embedding and energy checks for another port
  route. These are useful patterns, not proof that the legacy MSL route here
  passes them. The prior evidence note records setup errors in those live
  tests; their presence is not a green validation result.

Thus I would reuse and strengthen this coverage before adding another expensive
patch suite. Removing V173A does not remove all patch testing; removing MSL E2E
does not remove all MSL physics testing. The residual justification for each
extra slow lock must be its particular composition or numerical sensitivity.

## Is the microstrip a board someone would build?

**Yes.** A 254 µm substrate is nominal 10 mil material. Rogers lists RO4350B
at 0.010 inch nominal thickness, with standard 18 and 35 µm copper claddings;
its design dielectric constant is 3.66. This directly supports the stack's
buildability, not the accuracy of a lossless model over this test's whole band.
The data sheet lists thickness tolerance too: fabrication is not exact to the
micrometre. [Rogers RO4000 data sheet, pp. 2 and 4](https://rogerscorp.com/-/media/project/rogerscorp/documents/advanced-electronics-solutions/english/data-sheets/ro4000-laminates-ro4003c-and-ro4350b---data-sheet.pdf).

An etched 600 µm trace over that laminate, with 10 mm between electrical
reference planes, is an ordinary laboratory coupon. A sheet at z = 254 µm,
with the ground at z = 0, is a sensible zero-thickness conductor approximation
at the laminate interface. It represents finite copper by an ideal boundary;
it does not claim the physical copper has zero thickness. Finite copper
thickness, loss, roughness, dielectric dispersion, board edges and connectors
need separate treatment for fabricated-board accuracy. “50 Ω design” is a
nominal target; use the actual stack's impedance, not an assumption of exactly
50 Ω or identically zero reflection.

The present E2E builder uses `add_thin_conductor` with a box from h to h+dx
and copper metadata. The half-cell tie puts the PEC sheet at h; it does not
resolve 35 µm of copper. The sibling integration builder uses the clearer
zero-thickness declaration. This spelling difference deserves cleanup if
implementation is approved, but does not make the current realized sheet
physically wrong (`test_msl_sparam_ad.py:71-102`; integration `:212-227`).

“On-lattice” currently means **the vertical stack is aligned**, not every
dimension: h/dx = 3, but 600/dx is about 7.087. The new static check reproduces
the 592.667 µm width already asserted in
`tests/unit/sparams/test_msl_port_integration.py:706-714`. Three cells through
the substrate is a coarse integration fixture, not evidence of converged
impedance. LY also depends on dx (`:114-123`); refinement must hold the
physical exterior dimensions fixed rather than shrink the boundary clearance.

The former trace plane at 320 µm described a different discretized stack.
A deliberately suspended line is possible, but it requires a specified gap
and support; that was not this fixture's intended laminate coupon. I would
not treat the old comments' “66 µm air gap” as a demonstrated literal layer
in every Yee material component without inspecting those components. The
well-supported conclusion is that old and new realizations differ, not that
the old raster is a faithfully modelled manufactured suspended board.

The user-supplied 36/40 failures and max difference 0.17028876 are substantial.
But the example entries are reflections; the **whole S matrix has not changed
scale by an order of magnitude**. Recalculated from the archived local trace
and the committed golden (not the new VESSL arrays):

| GHz | old abs(S11) | current abs(S11) | old abs(S21) | current abs(S21) | Δarg(S21), degrees |
|---:|---:|---:|---:|---:|---:|
| 0.5 | .016593 | .003013 | .999851 | .999988 | +.175 |
| 1.0 | .033036 | .006011 | .999439 | .999958 | +.348 |
| 1.5 | .049150 | .008988 | .998736 | .999912 | +.514 |
| 2.0 | .064957 | .011814 | .997785 | .999838 | +.674 |
| 2.5 | .080370 | .014450 | .996627 | .999772 | +.827 |
| 3.0 | .095220 | .016909 | .995300 | .999724 | +.970 |
| 3.5 | .109329 | .019204 | .993846 | .999686 | +1.100 |
| 4.0 | .122539 | .021352 | .992310 | .999652 | +1.216 |
| 4.5 | .134715 | .023353 | .990731 | .999615 | +1.316 |
| 5.0 | .145758 | .025211 | .989161 | .999569 | +1.400 |

Here S11 is array `[0,0]`, S21 is `[1,0]`, not the trace's zero-based S11
label. Both reflection entries fail in all ten bins; each transmission entry
fails in eight. The current maximum |S12−S21| is 6.59e-5 on that archived
output. Transmission persists near unity; reflection falls approximately
5–6 times and changes complex direction. This is consistent with an impedance
change, not evidence that the line stopped transmitting. Small reciprocity
error is a useful witness but not sufficient validation, especially on projected
output. The sibling integration file's recorded run `369367259284` reports
mean Z0 57.58 → 46.16 Ω and mean |S11| .1160 → .0203 with unchanged broad
bounds (`:55-71`); it supports the interpretation without isolating every
contribution to this E2E difference.

## Is V173A a sensible antenna sample?

The drawn 28 × 36 mm rectangle and 1.5 mm, εr = 4.3 laminate are plausible
patch dimensions. A 1 mm metal plate can also be manufactured. The problem
is that the harness calls this a downsized PCB patch while defining its
metal thickness as `_DX`, not as an independently specified physical thickness
(`scripts/harnesses/v173a_physics_equivalence.py:48-81`). Naively refining dx
changes the object. That is strong evidence of a harness convenience, not a
considered foil or thick-plate model; the author's original intent is unknown.

Under the current contract its positive-thickness, high-sigma API Box is a
PEC volume, not the separately fenced raw sigma-fill model
(`rfx/api/_compile.py:149,276`; normative §§1.1,1.8). The archived witness
shows 1008 occupied cells, no sheets and walls at z = 1 and 2 mm. The drawing
was z = 1.5 to 2.5 mm. It therefore fails to describe its intended laminate
interface on this mesh. An infinite ground boundary at z_lo and a substrate
continuing to the absorber are also idealizations; they do not represent a
finite 50 × 50 mm manufactured ground plane. These choices can still support
a useful numerical control if explicitly declared and independently checked.

The baseline note itself is unreliable about geometry:
`tests/data/v173a_pre_t7_phase2_baseline.json:6` says every dimension is an
integer millimetre and the fixture is fully aligned. The live constants plainly
contradict that. It also attributes a previous frequency change to lateral
27 × 35 versus 28 × 36 occupancy. That past attribution is not a fresh A/B
measurement, but it identifies another change that z alignment alone does not
undo. The ownership operator and the in-plane boundary realization changed too.

**The “S11 dip” is not S11 at all.** The builder adds a soft Ez source and
a field probe, no lumped port. The statistic is the minimum of
`20 log10(abs(FFT(Hann * E)) + epsilon)` over 1.5–3.5 GHz, with epsilon
proportional to the band's maximum (`harness:103-123`). There is no incident
wave, reflected wave, impedance reference or source normalization. Algebraically,
doubling the trace amplitude raises this “dip” by 6.0206 dB even when physical
reflection is unchanged. A deep minimum may be a spectral node, weak source
content or window effect; it is not evidence of matching. This is stronger
than the harness's own admission that it is an uncalibrated surrogate. Its
header's “via lumped port” claim and the baseline's matching interpretation
should be retired if implementation proceeds.

The supplied run reports 1,914,163,251.7904541 Hz versus
1,994,994,938.1663296 Hz (−4.051724%), and the FFT statistic
−72.23665910385223 dB at 3,496,671,844.996355 Hz. Relative to the committed
−68.32136968566955, the statistic changes −3.915289 dB. The dip is near the
upper search limit. These are observations, not identified antenna performance.
The harness selects the strongest Harminv amplitude with Q ≥ 5, rather than
tracking a mode's spatial identity (`:85-100`); it can switch modes. Fixed
3000-step runs provide no asserted settling/convergence witness here. Standalone
rc 0 only means the reporting script completed: `:135-147` contains no baseline
assertion. The pytest lock failed at resonance before its dip assertion.

**I do not predict recovery of the old number from alignment.** Preflight
computes max face-to-node distance divided by the corresponding axis extent
(`rfx/api/_preflight.py:9277-9358`). It does not solve an EM sensitivity
problem. Both z faces move down by 0.5 mm, so metal thickness remains 1 mm;
the printed 50% concerns face placement relative to thickness. The familiar
δf/f ≈ −δL/L estimate applies to a relevant resonant length under additional
assumptions, not automatically to this displaced metal thickness. It is also
not a controlled small perturbation here.

An aligned redraw is worth considering to isolate geometry, but “see whether
1.995 GHz returns” should be a recorded observation, never its acceptance
criterion. Preserve the 1.5 mm substrate and an independently fixed 1 mm plate
on dx = 0.5 mm if the question is a thick plate; choose a sheet on the same
interface if the question is a PCB patch. These are different experiments.
Changing substrate height to 1 or 2 mm solely to fit the old 1 mm mesh changes
the board again. A single coincident scalar would not prove recovery of the
old physical problem or validate the old instrument.

## Is a full complex historical lock the right instrument?

The user's brittleness concern is **right about role, overstated about the
present assertions**. V173A uses frequency rtol 1e-6 and FFT-statistic atol
0.1 dB (`test_v173a_physics_equivalence_slow.py:62-81`), despite stale exactness
docstrings. MSL uses complex rtol .005 and atol .002 (`test_msl_sparam_ad.py:617`):
each entry gets `abs(actual−golden) <= .002 + .005*abs(golden)`. That is not
bit identity, and the absolute floor avoids singular relative demands near
a reflection null. The current red is not last-bit noise.

For a pure refactor on fixed inputs, tight comparison of full complex data is
an appropriate instrument. For continuing development, a tolerance-based full
S snapshot is also useful: magnitudes alone miss reference-plane phase drift,
port ordering, current-sign and de-embedding defects. Such a snapshot detects
change; it cannot know intent. Human adjudication is part of its job. A small
set of physical properties likewise cannot automatically bless every intended
change. Keep both instruments when their incremental coverage justifies cost.

The historical-board doubt is **confirmed**. Comparing a corrected live board
against an old-board output no longer tests same-problem equivalence. MSL's
own contract already narrowed the test to cross-version drift after an earlier
algorithm rebaseline (`:548-593`); its old “genuine pre-change” heading and
“precision-only” trailing comments are stale. Repeatedly restoring a past
answer after changing the problem would protect a number rather than its
physical premise. Conversely, frozen old fields with frozen old geometry remain
valid inputs to an assembly regression test even if that board is unsuitable
as a physical example. Do not regenerate replay binaries just to modernize
the live board.

There is another limitation: the live MSL lock uses default passivity enforcement.
`rfx/api/_sparams.py:4781-4807` can project S and exposes `S_raw`, correction,
reliability, settling and conditioning metadata. Production also audits raw
extraction, but this test only asserts the final S. A projection can hide some
raw defects. The sibling integration test also reads default projected S and
gates means (`test_msl_port_integration.py:248-293`); it does not independently
prove pointwise raw full-matrix passivity. This argues for better physics gates,
not discarding useful complex phase information.

## What should exist, and what a re-pin requires

**MSL: keep the board; permit a conditional E2E re-pin after qualification.**
The two supplied jobs explain why the old numerical locks need a decision;
they are not, alone, enough evidence to certify a replacement physical answer.

1. Freeze a physical drawing and a manifest: material model, sheet plane,
   realized footprint, fixed domain, port/reference planes, mesh, frequency
   vector, source, integration horizon, extraction settings, dtype/backend,
   JAX version and source SHA. Check realized dielectric cells and every
   voltage/current contour against that drawing. Report intentional residuals.
2. Inspect all complex bins with raw S and projected S distinguished. Gate
   finite, sufficiently excited and well-conditioned extraction; require
   settling evidence for each drive. Select a qualified frequency band in
   advance. The old E2E spans 0.5–5 GHz; the broad integration physics gate
   only claims 3–4.5 GHz. Extending that physical claim needs evidence, not
   merely ten finite outputs.
3. Use a uniform-line reference for that same cross-section and 50 Ω reference
   impedance. Check characteristic impedance and propagation constant against
   an independently qualified cross-section/line calculation; compare the
   complete finite-line response using its ABCD matrix and the actual plane
   separation. A quasi-static formula is a plausibility anchor with a model
   error budget, not an exact 0.2% oracle. Require bounded reciprocity,
   transmission phase/group delay, and raw singular-value passivity. Lossless
   guided power balance is approximate when radiation/absorber loss exists;
   account for it instead of forcing exactly |S11|²+|S21|² = 1.
4. Establish numerical uncertainty with fixed-geometry nested meshes and a
   longer record, plus a port-plane/line-length witness where extraction is
   suspect. Two meshes detect gross nonconvergence; they do not by themselves
   establish an asymptotic convergence order. Set physical tolerances from
   the oracle/model and numerical error budgets, before a candidate gate is
   judged. Do not derive them merely by surrounding today's answer.
5. Demonstrate that targeted wrong cases fail: e.g. missing trace/wrong interface
   for the geometry check; swapped current sign, phase/de-embedding error, or
   zero transmission for the observable checks. Keep these checks distinct
   enough to say which property broke. Recapture the qualified live E2E golden
   and manifest in a dedicated implementation change, then reproduce it in an
   independent confirmation run. Report why the existing numerical tolerance
   remains appropriate; do not widen it to accommodate a different board.

These are proposed future checks, not work launched here. Reuse the existing
integration, replay and reference-plane machinery where it covers the same
route. Initially keep one explicit full complex E2E snapshot as a secondary
drift sentinel. If its only marginal benefit is already covered by the
qualified complex-line comparison, retire the redundant slow solve. Test cost
and coverage, not attachment to a historic filename, should decide that.

**V173A: replace the physical claim, rather than refreshing its two numbers.**
Use explicit sheets for a PCB antenna, independent physical dimensions, a
realized stack/footprint assertion, and a modal observable identified by spatial
symmetry or field pattern. Reuse the existing mode-labelled patch approach.
Require post-source settling and fixed-board mesh/boundary convergence. Compare
to a qualified external solver or a stated approximate patch model with its
model discrepancy separated from discretization. If matching is a desired
property, use calibrated incident/reflected waves or voltage/current at a
specified 50 Ω port plane. If radiation performance is desired, add the
appropriate finite-ground and far-field/power witness; resonance alone cannot
guard it. A source-amplitude invariance check would immediately reject the
current FFT statistic as S11.

If the PI only wants a cheap composition sentinel, keep an explicitly synthetic
V173A descendant with a documented fixed geometry and fixed instrument; call
the FFT quantity a probe-spectrum statistic. It may have a tight output pin,
but no antenna matching claim. Off-lattice tie behavior belongs primarily in
fast geometry contract tests. Once equivalent composition coverage exists,
there is little value in paying for two identical V173A solves merely to read
two statistics from the same deterministic trace.

## Concrete samples for the PI to choose from

These are **unmeasured engineering proposals**, not validated replacements or
requests to launch a mesh campaign now. A mesh making dimensions exact does
not establish sufficient EM resolution.

| Purpose | Physical dimensions and idealization | Mesh that preserves the drawing |
|---|---|---|
| Keep today's microstrip | RO4350B h = 0.254 mm, drawn W = 0.600 mm, 10 mm between ports, sheet on h, ground at 0. Buildable nominal coupon; explicitly retain the known lateral raster error for the existing coarse sentinel. | Current uniform dx = 0.254/3 mm aligns the stack, but realizes W = 0.592667 mm. Do not call every face aligned. |
| Fully aligned microstrip alternative | Same laminate, nominal 18 or 35 µm copper represented by a PEC sheet; W = 0.635 mm (25 mil); plane separation 10.160 mm; margins 2.032 mm; domain 14.224 × 2.667 × 1.778 mm. Trace centered at y = 1.3335 mm, edges y = 1.016 and 1.651 mm; ports x = 2.032 and 12.192 mm; ground z = 0, sheet z = 0.254 mm, air above = 1.524 mm. Ordinary etched dimensions; impedance is near the intended class, not asserted exactly 50 Ω. The simulated uniform continuation/absorber represents a line segment, not a fabricated connector. | Uniform dx = 0.0635 mm: h = 4 cells, W = 10, plane separation = 160, margins = 32, domain = 224 × 42 × 28 cells before padding. Refinement 0.03175 mm preserves all these faces and planes. |
| Antenna-shaped replacement | Specify a nominal 1.5 mm FR4-like laminate with εr = 4.3 as a lossless test material; ordinary 35 µm foil represented by sheets. Finite 50 × 50 mm substrate/ground at x,y = 15…65 mm, ground z = 5 mm, substrate top z = 6.5 mm. Patch 28 × 36 mm at x = 26…54, y = 22…58 mm on z = 6.5 mm. Domain 80 × 80 × 30 mm with CPML outside. These are practical etched dimensions; actual FR4 material/loss/tolerance must be specified before comparison to hardware. A point excitation is adequate for a modal test; a matching test must additionally specify its physical feed and reference plane. | Uniform dx = 0.25 mm gives six substrate cells and exact board/patch faces; dx = 0.125 mm preserves the geometry. Keep dimensions fixed. The proposed clearances and resolution are starting choices to qualify, not demonstrated convergence. |

The antenna proposal changes the ground and substrate extent as well as the
foil, so it is a new physical fixture, not an attribution experiment for the
old 1.995 GHz pin. If scope is solely conductor-wall ownership, the existing
analytic cavity battery is a more direct instrument than either antenna. It
should not be presented as a substitute for open-region antenna physics.

## Confidence and decision requested

**High confidence:** the executable tolerance facts, the non-S11 FFT diagnosis,
the current volume/sheet distinction, ordinary buildability of the 10 mil
microstrip stack, the archived matrix's reflection/transmission distinction,
and the lack of a like-for-like premise for the old live MSL golden.

**Moderate confidence:** retaining a small complex E2E snapshot alongside
qualified physical checks is worth its maintenance cost; that is a coverage
judgment and can change when the replacement route demonstrably covers it.
The impedance change explains the observed direction, but does not isolate
the effects of every geometry and operator change.

**Unestablished:** which fraction of the V173A shift comes from vertical
placement, lateral footprint, volume ownership, mode selection or convergence;
whether any aligned drawing recovers the old frequency; whether the candidate
meshes converge; and whether either newly reported result merits a physics
accuracy claim. I am not inferring those answers from preflight or rc alone.

The PI's decision is therefore between a **qualified engineering fixture plus
an honestly scoped drift sentinel**, and retaining a **synthetic composition
sentinel only** where its cost is justified. My preference is the former for
MSL and reuse/redesign of existing physical patch coverage for V173A. No
baseline refresh is recommended before that choice and its evidence are made
explicit.
