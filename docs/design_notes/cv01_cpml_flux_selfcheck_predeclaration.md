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
`upml_full` control 0.989162 — i.e. ≥ 0.9092.

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
