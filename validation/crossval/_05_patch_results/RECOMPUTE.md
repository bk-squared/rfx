# cv05 + dielectric controls — recompute record (#931, group X-A)

Branch `feat/931-crossval-a`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XA-crossval-a`.

Everything here re-solves; nothing is translated from an old number. Each job
reads THIS worktree in place over the NFS mount (not a staged copy), so the
ring-down fixture the cv05 producer writes lands on the branch and the cv04
control can be a `git diff` against the committed artifacts.

Submitted 2026-09-07, cluster `remilab-c0`, preset `gpu-rtx4090`,
`JAX_PLATFORMS=cpu` inside the job (these are CPU-produced fixtures; the preset
is the machine allocation, not a device choice).

| case | VESSL run | yaml | what it does | expected wall clock |
|---|---|---|---|---|
| cv05 | **369367259142** | `scripts/vessl_931/cv05.yaml` | realization assert (no solve) → all three parts with the image's openEMS → geometry-only census → ring-down fixture rebuild | 1.5–3 h (baseline PART 1 alone was 158.1 s; the fixture rebuild is 5 lengths × 2 solves) |
| cv04 | **369367259144** | `scripts/vessl_931/cv04.yaml` | re-run + `git diff --exit-code` on the two committed JSONs | 10–30 min (8000 steps, hand-written loop; it exceeded 2 min on the shared pod, which is why it is here) |
| cv01 | **369367259145** | `scripts/vessl_931/cv01.yaml` | two rfx runs (straight + bend), Meep absent → exit 2 | 20–40 min |
| cv03 | **369367259146** | `scripts/vessl_931/cv03.yaml` | rfx leg + slab TE0 oracle gate, Meep absent → exit 2 | 20–40 min |

Submit command (run it from OUTSIDE a git worktree — the vessl CLI opens
`.git/HEAD` as a path and a linked worktree's `.git` is a FILE, so
`vessl run create` raises `NotADirectoryError` when cwd is inside one):

    cp scripts/vessl_931/cv05.yaml /tmp/x/ && cd /tmp/x && vessl run create -f cv05.yaml

Outputs land in `/root/workspace/claude-workspace/rfx/runs/issue931-post-<case>-<UTC>/`,
with the newest path echoed to `issue931-post-<case>.latest`.

## Pre-declared, before any of the four finished

Written down now so the numbers can be adjudicated instead of merely reported.

**cv05 will move away from its committed 6.48 % rfx-vs-openEMS agreement, and
that is not a regression.** Two records say so independently:

* `rfx-known-issues.md`, 2026-08-28 A/B verdict: "two_plane is the ONLY
  realization that gives the physically correct cavity ... and it is the
  realization that agrees WORST with the external reference ... Exactness of
  the cavity and agreement with the reference point in opposite directions."
* research note 20260711: a six-cell substrate next to a fine↔coarse grading
  transition split the mode and took the openEMS agreement from 2.65 % to
  6.45 %. The transition still sits immediately outside the fine block here
  (the z-profile builder grades the air, not the substrate), so that risk is
  live and unmitigated. It is reported, not designed around.

Direction: the realized cavity goes from 1.8161 mm (of which the top 455 µm was
air) to the declared 1.5 mm of FR4. A thinner, fully-loaded cavity raises the
effective permittivity seen by the patch and lowers the resonance; the
committed `rfx_harminv_hz` 2.331854551 GHz sat 3.78 % BELOW the analytic
2.423509825 GHz, so the expectation is that `rfx_vs_analytic_pct` grows in
magnitude on the same (negative) side. **A sign flip would mean something other
than the cavity moved and must be investigated before the number is quoted.**

The window it is judged against is measured, not asserted: the
`RFX_CV05_SHEET_PLANE_DELTA` arm (ground down one node plane, patch up one)
realizes a 2.1500 mm cavity against the declared 1.5000 mm, i.e. one node plane
per wall is worth +43 % of cavity height. The pre-contract error was +21.1 %,
about half of that, so a resonance shift of the order of the half-arm is
expected and anything far outside it is the thing to explain.

**cv01 / cv03 / cv04 must not move at all.** They are dielectric-only. This is
already established at build time — `scripts/diagnostics/cv0104_dielectric_control_witness.py`
digests the assembled `eps_r` / `sigma` arrays as raw bytes and compares this
branch against the pre-#931 checkout `rfx-baseline-d990e18c`:

    01_waveguide_bend    [181,181,1]  eps a693e67131469a9d  sigma 81205f6f74a3487c  IDENTICAL
    02_ring_resonator    [162,162,1]  eps 40dbec5bbf922728  sigma e006234d0697ae9c  IDENTICAL
    03_straight_wg_flux  [201,131,1]  eps f12b4af1ea61604e  sigma 309d6171b688ed09  IDENTICAL

all three with 0 PEC cells, 0 sheets, 0 wires on both sides — conductor-free by
measurement, not by reading the source. The four jobs above are the end-to-end
confirmation of the same claim. **cv04's job FAILS if its two committed JSONs
change by a byte.**

cv02 has no post job: no pre-change baseline job exists for it and no artifact
of it is committed, so a post-only run would have nothing to be compared
against. Its control is the digest row above, which compares bit patterns and
is stronger evidence than a solve.

## Fixture keys that will change

`tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json`, rebuilt
by `scripts/diagnostics/build_cv05_ringdown_spectra.py` inside the cv05 job:

* `runs.*.modes[*].freq` / `.Q` / `.amplitude` — every one; the cavity changed.
* `runs.*.realized_stack` — NEW. The planes, the cavity and the footprints of
  the build that produced each row, from `realized_pec_edge_masks`. A fixture
  whose provenance is a resonance and nothing else cannot say which cavity
  produced it; that is how the +21.1 % survived several re-pins.
* `_realized_x_cell_census` — re-measured, not carried forward, and its UNIT
  changes with the measurement. The old block counted masked NODES; the
  realized conductor is the tangential E edges its footprint stands on, and
  28 mm of metal is what sets the resonance, not 29 nodes. Measured locally,
  geometry only: 29.5 mm declared → **28** edges (was 29 nodes); 22.0 mm → 22
  either way, because both its faces are on the lattice. Expect every
  off-lattice row to drop by one and every on-lattice row to hold.
* `_provenance` — new commit, new timestamp.
* `_declared_TM100_hz`, `_what`, `_settling` — unchanged; `_declared_TM100_hz`
  is a closed form in εr / h / L / W and is mesh-independent.

`validation/crossval/_05_patch_results/cv05_run_openems_369367257743.json`:
every `rfx_*` key is a measurement of the old realization and is regenerated,
not adjusted. `openems_*` and `declared_modes_hz` are unaffected by the rfx
contract, which makes the old file the natural before/after witness — keep it
beside the new one.

## What ingest still has to do

1. Place the new cv05 run record beside the old one — copy
   `/root/workspace/claude-workspace/rfx/runs/issue931-post-cv05-20260907T095630Z/cv05_run.json`
   in as `cv05_run_openems_369367259142.json`, and do NOT delete
   `cv05_run_openems_369367257743.json`; it is the "before" half of the
   evidence. Then update `manifest.json` from
   `docs/design_notes/931_migration/XA-manifest.json.md`.
2. Commit the rebuilt `cv05_ringdown_spectra.json` — this phase deliberately
   does not.
3. Re-derive the farfield envelope band from the new run per the rule stated in
   the manifest note. It is fitted to a realization, so it cannot be translated.

---

# MEASURED (2026-09-07, all four jobs)

Written against the pre-declaration above, verbatim, pass or fail.

## Controls — cv01, cv03, cv04: unchanged, as declared

* **cv04 (369367259144): CONTROL PASS.** `git diff --exit-code` on
  `validation/crossval/_04_fresnel_results/` is EMPTY. Both committed JSONs
  (`fringe_gate_geometry.json`, `lattice_witness.json`) reproduce byte-for-byte
  from a re-run of the migrated checkout. "rfx accuracy: PASS" in the log.
* **cv01 (369367259145) and cv03 (369367259146):** the post-change logs are
  IDENTICAL to the pre-change baseline logs (369367258*, run 05:16 / 05:22) line
  for line, apart from one wall-clock line in cv03 ("Done in 2.0s" → "2.6s").
  Every printed digit holds: cv01 straight self-T 0.9892, smoothed T 0.6374;
  cv03 two-wave residual 0.0077, n_eff band mean rfx 2.84634 / analytic 2.84203,
  max |n_eff deviation| 0.262 % at f = 0.1592 (rfx 2.89607 vs analytic 2.88849),
  band-mean T 0.9657. cv03's 2 % n_eff gate against a closed-form slab TE0
  oracle is the sharpest detector of a dielectric Box occupancy change in this
  group, and it did not move by a digit.
* cv02 has no solve, by the reasoning above; its build-time material digest is
  bit-identical to the pre-#931 checkout.

The claim "dielectric sampling is untouched" holds on every control, at both the
array level and end to end.

## cv05 (369367259142) — migrated, and the pre-declaration was WRONG on sign

| quantity | committed (VESSL 369367257743) | post-contract | |
|---|---|---|---|
| realized cavity | 1.8161 mm node-to-node, top 455 µm air | **1.5000 mm, six FR4 cells** | as declared |
| realized wall planes | 12.0000 / 13.8161 mm (argmin) | **12.0000 / 13.5000 mm (k = 22 / 28)** | declared |
| ground footprint | 58 × 54 mm (drawn 60 × 55) | **60.0 × 55.0 mm** | drawn |
| patch footprint | 28 × 37 mm (drawn 29.5 × 38) | 28.0 × 37.0 mm | unchanged; half-cell debt |
| `rfx_harminv_hz` | 2 331 854 551 | **2 446 496 652** | +4.92 % |
| `rfx_vs_analytic_pct` | −3.7819 | **+0.9485** | **SIGN FLIPPED** |
| `rfx_vs_openems_harminv_pct` | 6.4774 | **11.7122** | worse, as predicted |
| `rfx_internal_pct` | 0.5084 | 1.3694 | still inside the 5 % self-consistency gate |
| `rfx_s11_dip_hz` / depth | 2.32 GHz / −1.61 dB | **2.48 GHz / −9.62 dB** | the feed now couples |
| settling witness | −21.6 dB (UNDERSETTLED) | −20.4 dB (UNDERSETTLED) | unchanged caveat |
| `status` | failed | passed | |

**The prediction that failed.** RECOMPUTE.md said, before the run: "the
expectation is that `rfx_vs_analytic_pct` grows in magnitude on the same
(negative) side. A sign flip would mean something other than the cavity moved
and must be investigated before the number is quoted." It flipped, and the
magnitude shrank. The prediction was wrong, and it was wrong for a reason that
was already written down in the memory this note cites: it modelled only the
cavity HEIGHT and ignored the two other mechanisms the same change removes,
both of which push the resonance UP —

* the #702 own-cell resample, which rewrote the ground sheet's own cell layer
  from vacuum to dielectric INSIDE the cavity (`rfx-known-issues.md`: "18590
  cells, ALL on z-plane k=29, all 1.000 → 3.380 ... the whole ground-plane cell
  layer, which sits INSIDE the patch cavity", worth ~13.5 pp on the canonical
  patch). Deleting it lowers the cavity's effective permittivity and raises f;
* the 455 µm vacuum layer that sat directly under the old patch plane, which is
  now FR4.

The same memory entry says outright that "the historical 9.32 was not correct —
it was two errors of opposite sign cancelling". A one-mechanism prediction
should not have been written after quoting that sentence. Recorded, not
repaired: no second cv05 solve was run to chase the prediction.

**What was predicted correctly.** The openEMS agreement got worse, 6.48 % →
11.71 %, exactly the direction the 2026-08-28 A/B verdict gives ("exactness of
the cavity and agreement with the external reference point in opposite
directions"). It stays inside the case's 20 % smoke bound. Note that the
openEMS number it is measured against, 2.19 GHz, is one the case itself refuses
to gate: `openems_mode_id_ok` is false on both runs because openEMS's
port-voltage ring-down carries poles that claim several declared members at
once.

**R5 — the metric is not quoted alone.** The full ring-down spectrum, not the
headline, and it reorganized in the direction a corrected cavity predicts:

| declared member | committed run | post-contract run |
|---|---|---|
| TM010 (1.914913 GHz) | **not found** | 1.900347 GHz, **−0.76 %** |
| TM100 (2.423510 GHz) | 2.331855 GHz, −3.78 % | 2.446497 GHz, **+0.95 %** |
| TM110 (3.088737 GHz) | 3.027300 GHz, −1.99 % | 3.186911 GHz, **+3.18 %** |
| unidentified poles | 3.605107 GHz | 3.758955 GHz |

Three measured poles now identify against the declared TM_mn0 spectrum instead
of two, and the member that was missing entirely is found within 0.8 %. That is
a structural agreement with the declared cavity, not a single number moving.
The second independent witness is the port: the S11 dip deepened from −1.61 dB
to −9.62 dB, which is the port-extent fix (it now spans realized wall to
realized wall rather than stopping at a drawn coordinate a node plane below the
patch) showing up as coupling. The two witnesses do not share a suspect
quantity.

**Caveat that did NOT improve:** the ring-down is still UNDERSETTLED (−20.4 dB
against a −40 dB bar), as it was before (−21.6 dB). Harminv frequency and Q
carry truncation error on both runs equally. Raising `num_periods` is a
separate change and was not made here; it must not be folded into this
migration.

## Census, re-measured (geometry only, no FDTD)

| declared L (mm) | committed (nodes) | post-contract (edges) |
|---|---|---|
| 29.5 | 29 | 28 |
| 23.0 | 23 | 22 |
| 22.5 | 23 | 22 |
| 22.0 | 22 | 22 |
| 21.65 | 21 | 20 |
| 21.5 | 21 | 20 |
| 21.0 | 21 | 20 |
| 20.5 | 21 | 20 |
| 38.0 | 38 | 38 |

Every row is exactly one lower except 22.0 and 38.0, whose faces are on the
lattice. This is the unit change (masked nodes → realized metal edges), not a
geometry change: N masked nodes have always carried N−1 tangential edges, and
the realized conductor is the edges. The rows that hold are the ones where the
old node count already equalled the edge count.

## Still open at the time of writing (closed in phase 2b — see below)

The ring-down fixture rebuild (`cv05_ringdown_spectra.json`, 5 lengths × 2
solves) was still running inside run 369367259142 when this section was
written; it writes into this worktree. Ingest takes it from
`git status` / the run's `produced/` directory. Its `runs.*.modes` will follow
the spectrum above, and the manifest's three cited fixture values must be
re-read from it per `docs/design_notes/931_migration/XA-manifest.json.md` §3 —
including re-checking that the 22.0 mm build is still assigned TM110, which is
what makes criterion B a demonstration rather than a claim.

# Phase 2b — ingest (2026-09-07, after the phase-2a merge)

The branch was merged onto `feat/931-lattice-ownership` at `770c4e6c`, so
everything below is measured on the tree that carries every group's changes,
not on X-A alone.

## Committed from run 369367259142

| file | source |
|---|---|
| `cv05_run_openems_369367259142.json` | `$OUT/cv05_run.json`, rc 0 |
| `cv05_run_openems_369367259142.log` | `$OUT/cv05_run.log` |
| `cv05_ringdown_fixture_rebuild_369367259142.log` | `$OUT/cv05_fixture_build.log`, rc 0 |
| `tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json` | written into this worktree by the producer during the run; md5 `b9ff9a20520ffcaba19a38e741878abe`, identical to the harvested `$OUT/cv05_ringdown_spectra.json` |

`$OUT = /root/workspace/claude-workspace/rfx/runs/issue931-post-cv05-20260907T095630Z`.
The 369367257743 record stays beside them as the before half.

Three tests in `tests/crossval/test_patch_mode_identification.py` were pinned to
the old fixture and were repinned from the new one (commit `efc9d2b1`); the
verdicts they assert are unchanged, only which pole carries them. The sheet
board resolves TM010 in every mis-realized leg, so `modes[0]` is now a correctly
identified TM010 and the drifted a-axis mode is `modes[1]`.

## The two open questions the manifest note demanded, answered from the rebuild

* **22.0 mm still assigned TM110.** `identify_patch_modes` returns
  `[1.9216 GHz → TM010, 3.0432 GHz → TM110]`, refuses to name a resonance and
  reports TM100 missing. Criterion (B) is demonstrated by the same length; no
  length was swapped, no tolerance moved.
* **The 38.0 mm falsifier still fires**, same mechanism: a weak third pole
  (amplitude 3.10e4 against 3.67e5) inside the identification window is named
  TM100 and that length passes there. No amplitude floor was added.

One new fact the census forces: 22.5 mm and 22.0 mm now realize the SAME 22
edges, so those two legs are one board with one ring-down and the
parametrization carries two identical rows.
`test_cv05_22p5_and_22p0_are_one_realization_since_931` asserts that identity
from the fixture's own `realized_stack` and fails if a future change separates
them again.

## Controls, re-measured on the MERGED tree

`scripts/diagnostics/cv0104_dielectric_control_witness.py` run twice —
`rfx-931-XA-crossval-a` at `b4961b56` against the pre-#931 checkout
`rfx-baseline-d990e18c` at `d990e18c` — build-time only, no solve. Digests are
sha256 of the assembled arrays as raw bytes. Both records are committed under
`control_witness_2b/` and each carries its own `repo_commit` / `repo_dirty`, so
the filename is not the provenance claim; both sides ran with a clean worktree.

| case | grid | eps_r sha256[:16] | sigma sha256[:16] | pec cells / sheets / wires | verdict |
|---|---|---|---|---|---|
| `01_waveguide_bend` | 181×181×1 | `a693e67131469a9d` | `81205f6f74a3487c` | 0 / 0 / 0 | IDENTICAL |
| `02_ring_resonator` | 162×162×1 | `40dbec5bbf922728` | `e006234d0697ae9c` | 0 / 0 / 0 | IDENTICAL |
| `03_straight_waveguide_flux` | 201×131×1 | `f12b4af1ea61604e` | `309d6171b688ed09` | 0 / 0 / 0 | IDENTICAL |

Both sides report the same digest for every array, and both report the case as
conductor-free — which is the second half of the control: a dielectric-only case
that grew a sheet or a wire would be a defect whichever side it appeared on.
This is the design note §5 pre-declaration "Dielectric-only cases: bit-identical
results before/after — the change must not touch them", checked against the
merged tree rather than against one branch.

cv02 has no committed artifact and no baseline job, so the digest IS its whole
control; that was true in phase 2a and stays true.

cv04 has no geometry layer to digest — it is a hand-written Fresnel loop — so
its control is end-to-end: re-run and `git diff --exit-code` on the two
committed JSONs. On the X-A branch (run 369367259144) that diff came back rc 0
and EMPTY. **Re-run on the merged tree at `b798d41f` as run 369367259290: same
verdict, `cv04_artifact_diff.rc = 0` with both diff files zero bytes**, and the
gate line reproduces digit for digit (T mean error 0.0110, R 0.0066, R+T energy
deviation 0.0091, all against a 0.0500 limit). `cv04.rc = 2` on both runs is the
script's "Meep absent" exit, not a failure. The contract's dielectric-only
pre-declaration holds end-to-end as well as at build time.

## Submitted in phase 2b, not yet returned

| run | yaml | what it decides |
|---|---|---|
| **369367259288** | `scripts/vessl_931/cv05_fixture_check.yaml` | **RETURNED rc 0: "OK: committed cv05_ringdown_spectra.json reproduces".** See below. |
| **369367259302** | `scripts/vessl_931/cv05_farfield_envelope.yaml` | **RETURNED rc 0 — see the section below.** Re-derives `D_ABS_TOL_DB`, `F_RES_REL_LO`/`HI` and the mode-pair band for `tests/crossval/test_patch_canonical_farfield_e4.py` by calling that file's own `rfx_run` fixture function on the sheet-declared canonical patch, and writes the arithmetic beside the measurement. Until it returns, `_ENVELOPES_REDERIVED_FOR_931` stays `False` and the three slow gates stay skipped. ~15 min. Replaces run **369367259289**, which died in 40 s on `No module named pytest`: its refusal gate shelled out to pytest and the solver image does not carry it, so the job refused on a missing test runner rather than on the board. The four fast gates now run as plain function calls inside the measuring script and pytest is in the pip line because the gate file imports it at module scope. |

Nothing in those three is a re-solve of a question already answered: the first
is a reproducibility check on a file just committed, the second is the first
measurement of a quantity that has never existed on this board, the third is the
contract's own dielectric control moved from one branch to the merged tree.

## The fixture reproduces — run 369367259288, rc 0

`build_cv05_ringdown_spectra.py --check` rebuilt all five lengths (ten solves)
on the merged tree at `b798d41f` and compared every mode frequency and Q
against the committed file at 1e-6 relative:

    tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json b9ff9a20520ffcaba19a38e741878abe
    OK: committed cv05_ringdown_spectra.json reproduces

The md5 in the run's own record is the file `efc9d2b1` committed, so the check
is against that file and not a copy of it. The worktree was untouched
afterwards — `git status --porcelain` empty, the fixture diff empty, zero files
harvested — which is the second half of the check: `--check` must compare
without writing, or a "reproduces" verdict could be a file overwriting itself.

Two things follow. The fixture is reproducible from the repo, so the manifest
citations in `docs/design_notes/931_migration/XA-manifest-2b.md` §A rest on
something a reader can regenerate; and the sheet-declared board's ring-down is
deterministic across runs on this image, which is worth knowing separately —
the round-1 macOS build differed from the cluster build by up to 1.3e-3
relative at 22.0 mm, and that scatter is what made the 38.0 mm falsifier
build-dependent in the first place.
## The canonical-patch envelope, re-derived — run 369367259302, rc 0

`measure_patch_canonical_farfield_e4.py` called
`tests/crossval/test_patch_canonical_farfield_e4.py`'s own `rfx_run` fixture
function on the merged tree at `09aab9d9`, clean worktree. All four of that
file's fast gates passed before the solve, then one FDTD run of the lean frame
(dx = 2 mm, `num_periods` = 110).

**Settling first, because nothing else is quotable without it: −42.9 dB against
a −40 dB bar.** The measurement is admissible; a run under the bar would not
have been, and the producer records the flag rather than leaving it to be
noticed.

| | pre-#931 one-cell ground | #931 sheet board |
|---|---|---|
| broadside D at the radiating bin | 7.39 dBi, \|Δ\| **0.60 dB** | 6.7241 dBi, \|Δ\| **0.0659 dB** |
| design-mode f_res vs openEMS 2.4221 GHz | 2.6954 GHz, **+11.3 %** | 2.5070 GHz, **+3.51 %** |
| mode-pair ratio (TL model 1.232) | 2.6954/2.2147 = 1.217 | 2.5070/2.0346 = **1.2322** |
| settling | — | −42.9 dB |

The #693 mechanism is measurable in the difference. The one-cell PEC ground had
its own cell inside the modelled cavity as vacuum; that diluted eps_eff and was
worth about +15 percentage points of the +11.3 %. A sheet owns no cell, the two
reserved fine cells are out of the mesh, and the offset drops to +3.51 %. The
directivity agreement improves by a factor of nine and the mode-pair ratio lands
on the transmission-line model's 1.232 to four digits.

### The prediction, and what it got right

The #740 `two_plane` arm measured −4.7 % at dx = 2 and was written into the gate
file as "a PREDICTION to check against — not a number to type in here". It gave
the direction (down from +11.3 %) and not the crossing: measured +3.51 %, still
high. Recorded, not repaired. `two_plane` put a second wall a cell away from the
laminate face; a sheet puts one wall on it, so the two are not the same board and
the prediction was never going to be more than directional.

### What moved, with the arithmetic

| constant | old | new | rule |
|---|---|---|---|
| `_ENVELOPES_REDERIVED_FOR_931` | `False` | `True` | — |
| `D_ABS_TOL_DB` | 1.0 | **1.0 (HELD)** | round-UP(0.0659 × 1.5, .1) = 0.1; `max(0.1, 1.0) = 1.0` — a rerun may not narrow a gate |
| `F_RES_REL_LO` | +0.06 | **−0.02** | `floor(+3.51 − 5) = −2 %` |
| `F_RES_REL_HI` | +0.16 | **+0.09** | `ceil(+3.51 + 5) = +9 %` |
| mode-pair band | [1.15, 1.30] | **unchanged** | 1.2322 sits inside with margin; the producer's rule would have moved it ±0.01 in both directions, widening the high side of a gate for no measured reason |

### The change of kind, said rather than absorbed

`[+6%, +16%]` excluded zero, so passing it also asserted "rfx reads HIGH here" —
the deleted mechanism is what put the whole band on one side. `[-2%, +9%]`
contains zero, so **the sign is no longer part of the lock**. What survives is
the magnitude envelope and its asymmetry: the high side is nine times the low,
so a drift back toward +11.3 % still fails and so does a 2 % swing low.
`XA-manifest.json.md` §5 pre-declared that this outcome would have to be stated
instead of re-tuned into looking like the old band, and the gate's own docstring
now states it.

### Verification without a second solve

The five gates the flag was holding are functions of the values in the committed
record, so they were checked against it directly rather than by re-solving on
the shared pod:

    PASS settling_witness                          -42.90 < -40.0
    PASS radiating_mode_is_broadside               peak 4.0, E -1.0, H -4.0 <= 15.0
    PASS directivity_within_committed_envelope     0.0659 <= 1.0
    PASS mode_pair_present_with_aspect_ratio       ratio 1.2322 in [1.15, 1.30]
    PASS f_res_inside_documented_coarse_dx_envelope +3.51% in [-2%, +9%]

The file's four fast gates run green in the ordinary lane. The slow lane's own
`pytest -m slow` re-solve is a duplicate of run 369367259302 and was not
submitted; if it is ever wanted, it is `pytest -q -m slow
tests/crossval/test_patch_canonical_farfield_e4.py` on a machine that can spend
~12 min of CPU.

### Pointers that moved with the edit

The constant block grew, so `D_ABS_TOL_DB` / `F_RES_REL_LO` / `F_RES_REL_HI` are
now at lines **157 / 163 / 164**, not 134 / 140 / 141.
`validation/README.md`'s pointer is refreshed here;
`validation/crossval/manifest.json` carries the same pointer twice and is the
merge agent's file — the replacement is in
`docs/design_notes/931_migration/XA-manifest-2b.md` §E.
