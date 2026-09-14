# cv01 straight-guide flux self-check under `RFX_BOUNDARY=cpml` — pre-declaration

Issue #813. Written 2026-09-13, **before the run**, on branch
`cv01-disposition-review` at base main `d56f68eb`.

## What is measured

`mean_self`, exactly as `validation/crossval/01_waveguide_bend.py:293` defines
it: the band mean of `uniform_filter1d(flux_out_s / safe_in_s, size=20)` over
the `above` mask (`01:278`), on **Run 1 only** — the straight-guide
self-calibration arm (`01:230-249`). The bend arm plays no part; `mean_self` is
a self-invariant, not a cross-solver comparison.

The driver is `scripts/diagnostics/cv01_cpml_flux_selfcheck.py`. It rebuilds
Run 1's rig from the same constants and asserts, before solving, that each rig
line it copied is still present verbatim in the committed cv01 script, so a
future edit to cv01 fails this driver instead of silently diverging from it.

## Arms

Five, all CPU, all the straight guide only. `full` = monitors as cv01 registers
them today (no `size=`, the legacy full padded plane). `interior` = the same
monitors with `size=(sy, dx)`, which post-#990 clamps to interior cells and
excludes CPML and bounding-node slots. `aperture` = `size=(2*w_wg, dx)`,
the Meep tutorial's own aperture-sized plane (#973).

| arm | boundary | monitor window |
|---|---|---|
| `upml_full` | upml | full padded plane (cv01 as committed) |
| `cpml_full` | cpml | full padded plane (cv01 as committed) — **this is #813's number** |
| `upml_interior` | upml | interior only |
| `cpml_interior` | cpml | interior only |
| `cpml_aperture` | cpml | 2·w_wg about the guide centre |

## Gate

**Pass = `cpml_full` gives `mean_self` inside [0.95, 1.05]** — cv01's own G2
window (`01:356`), unchanged. Nothing else counts as closing #813.

If it passes, #813 was the monitor-region defect that PR #990 closed and the
issue closes on the new record. If it fails, the number is physics or a second
defect, and no physics is touched here: the finding goes to the issue with
file:line evidence and the next pre-declared measurement.

## Falsifier for the "monitor region" attribution

`cpml_interior` reading the same `mean_self` as `cpml_full` — call it the same
within 1% relative — **refutes** the monitor-region attribution. The interior
window is the one thing that changes between those two arms; if removing the
absorber cells from the integration plane does not move the number, the
absorber cells are not where the missing quarter of the power went.

`upml_full` is the control: it must reproduce the committed artifact's
`mean_self_smoothed_over_band = 0.9891610388008335`
(`validation/crossval/_01_waveguide_bend_results/crossval.json`) to within
float noise, or the driver's rig differs from cv01's and no other arm is
readable.

## Prediction on record, from reading the code before running it

`cpml_full` **fails**, i.e. #990 did not change #813's number.

Reason, in the code rather than in memory: cv01 passes no `size=` to
`add_flux_monitor` (`01:242-244`, `01:265-267`), and both the helper docstring
(`rfx/runners/uniform.py:74`, "``size=None`` keeps the legacy full plane") and
PR #990's own body ("the legacy `size=None` full-plane behavior is preserved")
say that path is untouched. #910 was about `size=` requests overflowing into
the absorber; cv01 never makes one. So the hypothesis "#813 is the same defect
#990 fixed" should be refuted at the level of which code path runs, and the
run is here to confirm that rather than to discover it.

A prediction that turns out wrong is the interesting outcome and is reported as
such.

## What is recorded

For every arm: the full per-bin `T_self` trace over the band (not only the
mean), both raw flux spectra, the realized monitor cell slices and bounds in
metres, every preflight warning verbatim, the boundary, the grid shape and pad
counts, the commit sha, the wall time, and the rig-fidelity check's result.
Artifact: `scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck.json`.

Workspace rule R5: the headline `mean_self` is not the result. The per-bin
trace plus a second, independent witness — the `interior`/`aperture` arms, which
change the integration plane and nothing else — is.

## Not in scope

No gate, tolerance, window or committed artifact is modified. The committed
UPML run of 2026-09-06 stays exactly as it is; any new record is appended
beside it, never over it.

---

## Addendum 2026-09-14 — the next measurement, pre-declared (not run)

Everything above is the pre-declaration for the six-arm run that has now
happened. Nothing above is edited; this section is appended, and it declares
the next measurement before it runs, as R2 requires.

What the six arms left open: the guided channel conserves under both
boundaries (aperture `out/in` 0.98678 cpml, 0.99120 upml), so the 25-point
`mean_self` deficit is entirely net backward power through the **off-guide**
part of a full-cross-section plane — 26.04 points at the output plane, negative
in 88 of 88 gated bins, under CPML only. **Why** that region returns power is
not established. #813's own candidate (a), a far-x CPML return, is not refuted
by anything measured so far.

### Arm 1 (first) — `cpml_layers` sweep on the straight guide

This is the discriminator #813 itself named ("sweep CPML layer count and pad
clearance on the straight-guide arm alone") and the one the six arms did not
run. It goes first because it varies the absorber directly, where the x-ladder
varies only where the plane sits.

cv01 pins `cpml_n = 10`, below rfx's own `Simulation(cpml_layers=16)` default —
and that default is 16 *because* CPML guided-mode reflection was measured here
(`docs/agent-memory/rfx-known-issues.md`, "CPML guided-mode reflection ~12% at
default 8-10 layers": 10 layers → 11.7 %, 20 → 4.2 %, 40 → 1.8 %). So the rig
runs at the layer count that entry says is the worst of the three.

**Arms**: `cpml_full` at `cpml_layers` = 10 / 16 / 20 / 40. Four runs, CPU,
~2.5 min total at cv01's `n_steps = 25000`.

**Vary one thing.** cv01 writes `pml = cpml_n * dx` and then places the source
at `src_x = pml + dx` and the output plane at `sx - pml - 5*dx`. Letting `pml`
track the swept layer count would move the source and both monitors as well as
the absorber, confounding absorber thickness with plane position — that is the
naive sweep and it is **not** the measurement. The arm holds `src_x` and both
monitor coordinates at cv01's values (`src_x = 1.1 µm`, input plane 4.0 µm,
output plane 14.5 µm) and varies **only** `Simulation(cpml_layers=…)`.

**Anti-confound witness, recorded per arm**: rfx pads the grid by
`cpml_layers` on each face outside the declared 16 µm domain, so the interior
must stay at 161 cells per axis while `grid.shape` goes 181 → 193 → 201 → 241
and `grid.pad_*` follows the swept value; and preflight's realized monitor
coordinate must read 1.45e-05 m on every arm (the `normal_index` shifts with
the pad, the physical plane does not move). An arm that fails either check is
not readable and is reported as such rather than averaged in.

**Observable**: the outside-aperture band-summed backward power fraction at the
output plane — today −26.04 % at 10 layers — reported alongside `mean_self`.

**Gate (the absorber is the source of the backward power)**: the fraction is
monotone non-increasing across 10 → 16 → 20 → 40 **and** its magnitude at 40
layers is ≤ 1/3 of its magnitude at 10 layers. Equivalently, `mean_self` at 40
layers must have closed ≥ 2/3 of the gap from today's 0.748852 to the
`upml_full` control 0.989162 — i.e. ≥ 0.90906.

> **SUPERSEDED 2026-09-14** — the two halves joined by "Equivalently" above are
> not equivalent, and an arm landing in `mean_self` ∈ [0.894243, 0.909059] would
> pass one and fail the other. The binding gate is now `mean_self` ≥ 0.90906
> alone. See "Correction 2026-09-14" at the end of this file for the measured
> table and the reason.

**Falsifier**: the fraction is flat within 3 points across all four layer
counts and `mean_self` stays within 0.03 of 0.748852. That refutes "the CPML
backplane returns the off-guide power" — #813's candidate (a) — and leaves the
corner regions and a domain-wide standing pattern as the live candidates.

**Neither** (non-monotone, or a shrink smaller than 3× but larger than 3
points) is **not a verdict**. It is reported as a non-closing attempt, and R2
forbids a fifth layer count without a new pre-declaration naming a new
mechanism or an identified defect in this one.

### Arm 2 (second) — output-plane x-ladder

`cpml_full`, full plane, same rig, output plane at x = 12.0 / 13.0 / 14.0 /
14.5 µm. **Gate**: the outside-aperture backward fraction growing monotonically
toward the boundary is boundary return. **Falsifier**: flat within 3 points
across the ladder ⇒ a domain-wide pattern, boundary exonerated. Four arms,
~2.5 min CPU.

This was the previously announced "next measurement". It is demoted to second
because a flat ladder is consistent with both a domain-wide pattern *and* a
return that has already decayed to a plateau by 12 µm, while the layer sweep
changes the absorber itself.

### Arm 3 (independent route) — domain grown in x

Re-run `cpml_full` with `sx` grown in x only. A boundary-return artefact moves
with the boundary; a physical pattern does not. Independent of arms 1 and 2 in
that it shares neither the swept quantity nor the plane position.

### Still not in scope

No gate, tolerance, window or committed crossval artifact is modified by any of
these arms, and none of them is run in this PR.

---

## Correction 2026-09-14 — Arm 1's gate joined two criteria that are not equivalent

Independent verification review of PR #1006 found a defect in the Arm 1 gate
written above. This section **supersedes** the gate paragraph in "Arm 1 (first)
— `cpml_layers` sweep on the straight guide". Nothing above is rewritten; the
original wording stays visible so the correction is reviewable.

### What was wrong

The gate read: *"the fraction is monotone non-increasing across 10 → 16 → 20 →
40 **and** its magnitude at 40 layers is ≤ 1/3 of its magnitude at 10 layers.
**Equivalently**, `mean_self` at 40 layers must have closed ≥ 2/3 of the gap
from today's 0.748852 to the `upml_full` control 0.989162 — i.e. ≥ 0.90906."*

The two halves are **not** equivalent, so "Equivalently" was false and the gate
was undecidable in a band of outcomes.

Measured, not argued. Taking the committed artifact and scaling the
outside-aperture per-bin term at **both** flux planes by a factor `s` — the
right construction for a layer sweep, since the absorber acts on the whole
domain and not only on the output plane — then rebuilding `mean_self` through
cv01's own arithmetic (`flux_out / safe_in`, `uniform_filter1d(size=20)`, band
mean over `above`):

| `s` | `mean_self` | outside-aperture fraction | note |
|---|---|---|---|
| 1.000000 | 0.748852 | −26.0418 % | today, at `cpml_n = 10` |
| 0.386557 | **0.894243** | **−8.6800 %** | the fraction half's threshold (−26.04/3) |
| 0.333333 | 0.907331 | −7.3965 % | outside power literally ×1/3 |
| 0.326332 | **0.909059** | **−7.2299 %** | the `mean_self` half's threshold |
| 0.000000 | 0.991232 | 0 % | outside term removed entirely |

The `s = 1` row reproduces the committed `cpml_full` `mean_self`
0.7488520140093946 bit-identically, and the `s = 0` row reproduces the committed
`cpml_aperture` 0.991232, so the reconstruction is cv01's arithmetic and not a
second implementation of it.

**A 40-layer arm landing anywhere in `mean_self` ∈ [0.894243, 0.909059] — a
window 0.0148 wide — passes one half of the gate and fails the other.**

Why they diverge: the fraction is normalized by the full-plane band sum, which
itself moves as the outside term shrinks, while `mean_self` is a band mean of
*smoothed per-bin ratios*, not a ratio of band sums. (At `s = 1` the
ratio-of-band-sums is 0.758845 against `mean_self` 0.748852 — different
quantities, and the gap between them is not constant in `s`.)

### The corrected gate

**Gate (the absorber is the source of the backward power) — one binding
criterion**: `mean_self` on the 40-layer arm **≥ 0.90906**, i.e. it has closed
≥ 2/3 of the gap from 0.748852 to the `upml_full` control 0.989162.

`mean_self` is the binding quantity because it is what cv01's G2 actually gates
and what the control arm anchors; the fraction is a derived diagnostic with no
gate of its own anywhere in the repo.

**Reported alongside, and NOT gating**: the outside-aperture backward power
fraction at each layer count, and whether it is monotone non-increasing across
10 → 16 → 20 → 40. These are observables. If the fraction's behaviour and the
`mean_self` verdict disagree, **that disagreement is itself reported** — it is
not resolved silently, and `mean_self` is the one that decides pass or fail.

**Falsifier, unchanged in substance, with its logic made explicit**: the two
conditions are a **conjunction** — both must hold — and were never claimed to be
equivalent, so the defect above does not reach them. `mean_self` at 40 layers
stays within 0.03 of 0.748852 **and** the fraction is flat within 3 points
across all four layer counts. If only one holds, the attempt is non-closing and
is reported as such. As with the gate, `mean_self` is the binding half.

### Scope

This corrects a pre-declared criterion **before** the measurement runs, which is
what a pre-declaration is for. No gate, tolerance, committed crossval artifact or
measured number changes: the sweep has not run, and every number in the table
above is a recomputation from the already-committed artifact, not a new
simulation.

---

## Result 2026-09-14 — Arm 1 ran, and its single binding gate passes

Arm 1 of the addendum above ran on CPU at base main `883615c6`: eight solves,
`n_steps = 25000` (cv01's own value). New artifact
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json`, log
`layer_sweep_run.log`, both written beside the six-arm `selfcheck.json`, which
is not touched. Driver: the same
`scripts/diagnostics/cv01_cpml_flux_selfcheck.py`, which gained a second mode
(`--mode layer-sweep`); the six-arm mode is unchanged, and a re-run of its
`upml_full` control is identical to the committed artifact on all 22 arm fields
— including all six 200-point per-bin arrays, 1200 elements, element for
element — with only `wall_s` differing (35.1 s → 37.7 s).

Nothing here closes #813. It attributes the mechanism; the issue stays open.

### What varied, and what did not

Only `Simulation(cpml_layers=...)`. `pml = cpml_n * dx` stayed at cv01's
`10 * dx`, so the source (1.1 µm) and both monitor planes (4.0 µm, 14.5 µm) did
not move with the absorber — the anti-confound clause above. rfx pads the grid
outside the declared 16 µm domain, so the interior stayed 161 cells per axis on
every arm while `grid.shape` went 181 → 193 → 201 → 241, exactly the witness
written before the run. Preflight's realized output plane read 1.45e-05 m at
all four layer counts, its normal index moving with the pad (155 → 161 → 165 →
185) while the physical plane stood still. All four arms are readable; none was
excluded.

### The table

`mean_self` is cv01's Run-1 band mean. The `aperture` column is the companion
arm with `size=(2*w_wg, dx)`; the outside-aperture column is `full` −
`aperture`, band-summed at the output plane over the full-plane band sum — the
construction that gave −26.04 % on the six-arm artifact. Every number is
resolved against the artifact key by key in "Numeric provenance" below.

| `cpml_layers` | `grid.shape` | interior | `mean_self` (full) | `mean_self` (aperture) | outside-aperture (%) | negative bins | wall full/aperture (s) |
|---|---|---|---|---|---|---|---|
| 10 — cv01 today | 181×181×1 | 161 | 0.748852 | 0.991232 | −26.04182 | 88/88 | 45.3 / 35.2 |
| 16 — rfx default | 193×193×1 | 161 | 0.884100 | 0.996183 | −8.56374 | 88/88 | 45.3 / 37.9 |
| 20 | 201×201×1 | 161 | 0.919530 | 0.996269 | −4.90301 | 88/88 | 47.4 / 40.4 |
| 40 | 241×241×1 | 161 | 0.947345 | 0.990150 | −1.93523 | 88/88 | 67.8 / 60.4 |

### Control

The 10-layer arm **is** cv01's rig, so it is the control, and it reproduces the
committed six-arm `cpml_full` exactly: `mean_self` 0.7488520140093946 against
0.7488520140093946, and the outside-aperture fraction −26.041819705557895
against −26.041819705557895. Not "within float noise" — the same doubles.

### Verdict against the gate written before the run

**PASS.** The single binding gate is `mean_self` ≥ 0.90906 on the 40-layer arm;
it reads 0.947345, closing 82.6 % of the gap from 0.748852 to the `upml_full`
control 0.989162. The reported, non-gating observable moved with it: the
outside-aperture fraction is monotone non-increasing in magnitude across
10 → 16 → 20 → 40 and shrinks to 0.07431 of its 10-layer value, about a
thirteenth. Gate and observable agree, so there is no disagreement to report,
and the falsifier holds on neither half.

The 2026-09-14 correction to this gate did not change the outcome: at 40 layers
the retired fraction half (|−1.93523| ≤ |−26.04182| / 3 = 8.68) and the binding
`mean_self` half both pass. The correction would have decided only an arm
landing in `mean_self` ∈ [0.894243, 0.909059], and none did. That the two
halves genuinely decouple is visible at 16 layers anyway: there the
fraction's magnitude, 8.56, is already **below** the retired threshold of 8.68
— that half would read PASS — while `mean_self`, 0.884100, is still **below**
0.90906, so the binding half reads FAIL. Same arm, opposite verdicts.

### What this says, and what it does not

**Says**: the missing power is the CPML absorber's own reflection. Deepening
the absorber and changing nothing else recovers it monotonically, and what it
recovers is the off-guide part — the aperture arms conserve at ≈0.99 at every
layer count, so the guided channel was never where the deficit lived. This is
*consistent with* `docs/agent-memory/rfx-known-issues.md:4158` ("CPML
guided-mode reflection ~12% at default 8-10 layers ... 10 layers → 11.7 %, 20 →
4.2 %, 40 → 1.8 %"), the entry that raised rfx's own `cpml_layers` default from
8 to 16. A different observable (|b1/a1| on WR-90) and a different structure,
but the same monotone-in-thickness shape and the same verdict on 10 layers.

**Does not say** that cv01's CPML variant now passes cv01's own G2. It does
not. G2 is `0.95 <= mean_self <= 1.05` and the 40-layer arm reads 0.947345, so
G2 is false at every swept count. A deeper absorber takes the deficit from 25
points to 5; it does not remove it. What is left at 40 layers is unattributed
and is not attributed here.

**Does not say** that the residual at 40 layers is all off-guide. The
aperture column is not monotone — 0.991232, 0.996183, 0.996269, 0.990150 — so
the 40-layer arm's guided channel gives back about 0.6 points against the
20-layer arm's. The aperture windows conserve to within a point of each other
at every depth, which is what carries the off-guide reading at 10 layers; it
does not license attributing all 4.2 remaining points at 40 layers to the
off-guide region, and that split is not measured in this campaign. (It is
measured in the next one: see "Result 2026-09-14 — the arm ran, and the
prediction holds" below, which splits the 40-layer residual.)

**Does not say** anything about the bend arm, the Meep leg, `mean_T`, or the
committed UPML run of 2026-09-06. None of them is touched by this measurement.

### R2

One attempt, pre-declared, closing: it reached the gate declared before it ran.
No second attempt on this mechanism. Arms 2 (output-plane x-ladder) and 3
(domain grown in x) stay written down and unrun — with the absorber identified
they are no longer the next thing to run, and the residual 5 points at 40 layers
would need its own pre-declaration naming its own mechanism.

### Preflight, verbatim

Identical on all four `full` arms; the `aperture` companions add two
`[FLUX REGION]` lines and nothing else.

    [PREFLIGHT] dielectric 'wg' on x: 11.5 cells per λ_eff (eps_r=12.00, freq_max=74.95THz, dx=100nm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    [PREFLIGHT] dielectric 'wg' on y: 11.5 cells per λ_eff (eps_r=12.00, freq_max=74.95THz, dx=100nm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    [PREFLIGHT] dielectric 'wg' on z: 11.5 cells per λ_eff (eps_r=12.00, freq_max=74.95THz, dx=100nm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    [PREFLIGHT] all dielectric(s) ['wg'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)

Every arm also carries one `warnings.warn` from the build — the
`amplitude_kind` deprecation (issue #571), recorded per arm under
`preflight_warnings` — and the run's advisory echo of the four lines above
under `run_warnings`. None of the four lines is boundary-count dependent: three
are resolution, the fourth is the lossless-dielectric-in-an-open-domain
advisory, and all four read the same on the 10-layer and the 40-layer arm. None
can explain a difference between arms, and none is suppressed.

### The rig change this licenses (separate commit, diagnostic variant only)

cv01 pins `cpml_n = 10`, below rfx's own `Simulation(cpml_layers=16)` default,
and the sweep says what that costs on the `RFX_BOUNDARY=cpml` path. The
smallest swept count that reaches the pre-declared bar is **20** — 16 gives
0.884100, short of 0.90906. So the CPML variant moves to 20 layers while
`pml`, and with it every plane position, stays where it was.

20 has precedent here for the same reason — a guided mode running into the
absorber needs more depth than a radiating one. The WR-90 slab measurement
that fixed rfx's phase-alignment constants ran at it
(`scripts/verify_phase_alignment.py:40`, "Measured on 2026-04-22 WR-90 slab,
cpml_layers=20 on both sides"), and it is the middle rung of the layer ladder
in `docs/agent-memory/rfx-known-issues.md:4158` ("20 → 4.2 %").

**Correction, round-1 review.** The first version of this paragraph, of the
comment in cv01 and of PR #1027's body said 20 was "what
`examples/crossval/11_waveguide_port_wr90.py` uses". That was wrong twice
over. The path does not exist — `examples/crossval/` holds only a README, and
the case is `validation/crossval/11_waveguide_port_wr90.py` — and that file
does not use 20: it derives its depth from the guide wavelength,
`CPML_LAYERS = int(np.ceil(0.75 * _LAMBDA_G_LOW_M / DX_M))` = **43** (its own
comment weighs 46 against the 43 shipped; 20 appears nowhere in it). The claim
came from `rfx-known-issues.md:4158`, whose 2026-04-22 sentence "`examples/
crossval/11_waveguide_port_wr90.py` uses 20 layers" was true when written and
has since rotted on both halves — the file moved and its depth was re-derived.
Quoting a memory entry is not the same as checking it, and this one was not
checked. The choice of 20 does not depend on the mistake: it is the smallest
swept depth that cleared the gate, which is measured above.

That is a rig change to an env-var diagnostic variant: not physics, not a gate.
The UPML path — the committed run, its artifact and every number in it — is
untouched, and no gate, tolerance or committed crossval artifact moves. cv01
was not re-run for it, because running it would overwrite the committed UPML
results; the evidence is the straight-guide arm above, whose driver copies
cv01's Run 1 line for line under a rig-fidelity check and whose 10-layer arm
reproduces the committed number to the last bit.

### Numeric provenance

Every number in the table above, by artifact key, so a regenerated artifact
that moved one of them reds the numeric-provenance gate instead of leaving a
stale table.

`mean_self`, full plane:
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.10.mean_self_full = 0.748852`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.16.mean_self_full = 0.884100`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.20.mean_self_full = 0.919530`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.40.mean_self_full = 0.947345`.

`mean_self`, aperture companion:
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.10.mean_self_aperture = 0.991232`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.16.mean_self_aperture = 0.996183`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.20.mean_self_aperture = 0.996269`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.40.mean_self_aperture = 0.990150`.

Outside-aperture fraction, in percent (cited unitless — the gate's `%` unit
would rescale the literal by 1e-2, and the artifact stores percent):
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.10.outside_aperture.fraction_percent = -26.04182`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.16.outside_aperture.fraction_percent = -8.56374`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.20.outside_aperture.fraction_percent = -4.90301`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.40.outside_aperture.fraction_percent = -1.93523`.

Verdict and gate:
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::verdict.gate_threshold = 0.90906`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::verdict.mean_self_at_40 = 0.947345`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::verdict.fraction_shrink_ratio_40_over_10 = 0.07431`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::verdict.gate_pass`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::verdict.fraction_monotone_non_increasing_in_magnitude`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::verdict.falsifier_holds`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::verdict.gate_and_observable_disagree`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.40.gate_G2_0p95_1p05_full`.

Control and anti-confound witness:
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::control_reproduces_committed_cpml_full.mean_self_rel_diff = 0.0`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::all_arms_readable`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.10.witness.grid_shape[0] = 181`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.16.witness.grid_shape[0] = 193`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.20.witness.grid_shape[0] = 201`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.40.witness.grid_shape[0] = 241`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.40.witness.interior_cells.x = 161`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.40.witness.realized_output_coordinate_m = 0.0000145 m`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.40.witness.readable`.

Wall time and run identity:
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.10.full.wall_s = 45.3`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.40.full.wall_s = 67.8`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::n_steps = 25000`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::n_steps_is_cv01_value`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::rig_fidelity_check.checked_lines = 34`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::rfx_provenance_check.under_repo_root`.

`checked_lines` is **34** in the artifact and **35** at HEAD: the rig change
added cv01's new `cpml_layers` line to the driver's `RIG_LINES` after this
artifact was written. So a regeneration at HEAD is expected to differ in
exactly that field and in `wall_s`, and in nothing else. The citation above
resolves against the record as written, which is what a record is for.

---

## Next measurement, pre-declared 2026-09-14 — splitting the 40-layer residual

Written and committed **before it runs**, as R2 requires. Appended; nothing
above is edited.

### What the sweep left open

At 40 layers `mean_self` is 0.947345 against the UPML control's 0.989162 — 4.2
points still missing, and the sweep does not say where they are. Two things
above make that gap worth one more arm rather than a shrug:

1. The aperture column is not monotone (0.991232 / 0.996183 / 0.996269 /
   0.990150), so the 40-layer guided channel gives back ~0.6 points against the
   20-layer one, and "all of the residual is off-guide" is not established.
2. **The `full` window is not held fixed across the sweep.** Every plane
   *position* is (1.1 / 4.0 / 14.5 µm, witnessed per arm), but `size=None`
   integrates the whole padded plane, whose tangential cell count grows with
   the absorber: 181 cells at 10 layers, 241 at 40. The physical interior
   (161) and the aperture (21) do not grow. So a deeper absorber changes both
   the absorber and how many absorber-cell slots the full-plane integral
   contains, and the sweep cannot separate the two on its own.

### The arm

One run, ~60 s CPU: the `interior` window, `size=(sy, dx)`, at **40 layers** —
the one window the sweep does not run. With the sweep's committed `full` and
`aperture` arms at the same depth it splits the residual exactly as the
six-arm campaign split the 10-layer case:

| part | 10 layers (six-arm artifact) | 40 layers |
|---|---|---|
| absorber-cell slots, `full` − `interior` | −1.25 points | to be measured |
| interior off-guide, `interior` − `aperture` | −24.79 points | to be measured |
| total outside-aperture, `full` − `aperture` | −26.04 points | −1.94 points |

Both parts band-summed at the output plane over cv01's own `above` mask, as
percentage of the full-plane band sum — the construction the six-arm artifact
used, run through one code path for both depths so the two are comparable.

The `full` and `aperture` arms are read out of the committed sweep artifact
rather than re-run. That is only legitimate if they are the same rig, so the
driver asserts it before subtracting: grid, `above` mask, frequency axis,
`n_steps` and boundary must all match the new arm, and a mismatch fails the
run instead of producing a subtraction across two rigs.

### Prediction, on record before the run

**Interior off-guide still dominates**: |`interior` − `aperture`| stays larger
than |`full` − `interior`|, as it is at 10 layers (24.79 against 1.25). That
would say the residual is still off-guide power inside the physical interior,
and the padded-plane window's growth is not what is being measured.

**The interesting alternative**: `full` − `interior` has **grown** in
magnitude past its 10-layer −1.25 points. Four times as many absorber-cell
slots sit inside the full plane at 40 layers as at 10, so if those slots now
carry the bulk of the residual, the integration window — not the absorber — is
the next suspect, and the next measurement is a window question rather than a
boundary one.

### No gate, one attempt

This arm gates nothing and closes nothing. It is a split of a residual the
sweep left unattributed, and its outcome names the next suspect. One attempt;
a second needs its own pre-declaration naming a new mechanism. Artifact:
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json`, appended
beside the other two, neither of which is touched. No gate, tolerance, window
or committed crossval artifact is modified by it.

### Result 2026-09-14 — the arm ran, and the prediction holds

One run at commit `8ed6278e`, the commit that carries the pre-declaration
above: the `interior` window at 40 layers, 78.4 s CPU, `n_steps = 25000`.
Artifact `scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json`,
appended beside the other two; neither is touched.

`mean_self` on that window reads 0.948519, against the same depth's full plane
0.947345 and aperture 0.990150.

| part | 10 layers | 40 layers | negative bins at 40 |
|---|---|---|---|
| absorber-cell slots, `full` − `interior` | −1.24765 | **−0.10218** | 58/88 |
| interior off-guide, `interior` − `aperture` | −24.79417 | **−1.83304** | 88/88 |
| total outside-aperture, `full` − `aperture` | −26.04182 | −1.93523 | 88/88 |

Points of the full-plane band sum, both depths computed through the same code
path — the 10-layer column is recomputed from the six-arm artifact, not
retyped, and reproduces the −1.25 / −24.79 / −26.04 that campaign reported.

**The pre-declared prediction holds.** Interior off-guide still dominates:
−1.83304 against −0.10218, a ratio of 18 to 1 (it was 20 to 1 at 10 layers).
The alternative — `full` − `interior` having grown — did not happen: that term
**shrank by 12.2×** while `interior` − `aperture` shrank by 13.5×, and the two
sum to the −1.93523 the sweep already recorded.

**So the window caveat is retired, measured rather than argued.** The full
plane holds four times as many absorber-cell slots at 40 layers as at 10 (241
tangential cells against 181, with the interior window fixed at 161 and the
aperture at 21), and those slots nevertheless contribute twelve times *less*.
The sweep's improvement is therefore not an artefact of the integration window
growing with the absorber; it is the absorber. At 40 layers the slots term is
no longer even consistently signed — 58 of 88 bins negative, against 88 of 88
at 10 layers — which is what a term decaying into noise looks like.

**What it still does not say.** Where the last points live. Against the
UPML controls the 40-layer arm is 4.18 points short on the full plane
(0.947345 vs 0.989162) and 0.26 points short on the aperture (0.990150 vs the
six-arm `upml_aperture` 0.992737), and −1.94 points of band-summed backward
power is outside the aperture. Those are three different normalizations of the
same run — a band mean of smoothed per-bin ratios in the first two, a ratio of
band sums in the third — and this note has already measured that they do not
convert into one another. Attributing the remaining points across them needs
its own pre-declaration and did not happen here.

**Comparability**, since two of the three arms were read out of a committed
artifact rather than re-run: grid, `above` mask, frequency axis, `n_steps` and
boundary all match between the new arm and both stored arms, asserted by the
driver before it subtracted. Witnesses on the new arm: `grid.shape`
241×241×1, interior 161 cells, realized planes 4e-06 m and 1.45e-05 m with
normal indices 80 and 185, interior window cell slice [40, 200]. Preflight is
identical to the 40-layer full arm's, verbatim, plus the same `amplitude_kind`
deprecation.

**R2**: one attempt, pre-declared, and it reached the prediction it declared.
No gate was involved and none moved. A further split of the last points needs
a new pre-declaration naming its own mechanism.

### Numeric provenance, residual split

`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::interior_arm.mean_self = 0.948519`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::split_at_swept_layers.absorber_cell_slots_points = -0.10218`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::split_at_swept_layers.interior_off_guide_points = -1.83304`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::split_at_swept_layers.total_outside_aperture_points = -1.93523`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::split_at_cv01_layers_reference.absorber_cell_slots_points = -1.24765`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::split_at_cv01_layers_reference.interior_off_guide_points = -24.79417`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::split_at_cv01_layers_reference.total_outside_aperture_points = -26.04182`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::split_at_swept_layers.n_bins_slots_negative = 58`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::split_at_swept_layers.n_bins_interior_off_guide_negative = 88`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::split_at_swept_layers.n_bins_in_band = 88`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::full_plane_tangential_cells = 241`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::interior_window_cells = 161`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::interior_arm.wall_s = 78.4`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::cpml_layers = 40`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::stored_arms_source.full_mean_self = 0.947345`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::stored_arms_source.aperture_mean_self = 0.990150`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::comparison.interior_off_guide_still_dominates`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::comparison.absorber_cell_slots_grew`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::comparability_checks.vs_stored_full.grid_matches`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::comparability_checks.vs_stored_aperture.above_mask_matches`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::rig_fidelity_check.checked_lines = 35`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/residual_split.json::n_steps = 25000`.

The `upml_aperture` control quoted above is
`scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck.json::arms.upml_aperture.mean_self = 0.992737`.
