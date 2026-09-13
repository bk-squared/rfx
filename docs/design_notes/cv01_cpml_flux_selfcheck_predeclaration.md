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
