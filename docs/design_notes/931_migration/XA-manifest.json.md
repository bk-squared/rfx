# manifest.json — replacement text for case `05_patch_antenna` (group X-A, #931)

Owner of the file: the merge/ingest agent. This note is the text to apply, plus
the rule for the two numeric slots that cannot be filled until VESSL run
**369367259142** finishes. Nothing here is hand-computed from an old number.

Nothing outside `cases[] id="05_patch_antenna"` changes. cv01, cv02, cv03 and
cv04 entries are untouched — they are dielectric-only and their assembled
material arrays are bit-identical to the pre-#931 checkout (digests in
`validation/crossval/_05_patch_results/RECOMPUTE.md`).

## 1. `claim_scope` — the sentences that became false

Three statements in the current text describe the OLD realization and are wrong
in kind, not only in number. They must go, not be re-tuned:

1. **"the 20% openEMS gate provably passes over two physically different
   geometries (broken 2-cell and split 6-cell substrate, #325)"** — the case no
   longer has two substrate geometries. `smooth_grading` over one concatenated
   array with no `preserve_regions` used to grade the fine block itself away
   (realized node planes 12.0000 / 12.7692 / 13.3609 mm, three planes of
   0.77/0.59/0.46 mm). The z profile is now built block by block, so the
   substrate is six exact 250 µm cells between exact nodes and the build
   asserts it before any solve. The #325 caveat is superseded HERE; it stays
   true wherever else it is stated about other cases.
2. **"the ~2.65% agreement is a documented z-under-res/staircase error
   cancellation"** — that cancellation was between a 1.8161 mm realized cavity
   (+21.1 %, its top 455 µm vacuum) and the in-plane staircase. One half of it
   is gone. The replacement sentence must not claim a new cancellation; it
   states what is now realized and leaves the agreement to be reported.
3. **"the ~2.65% / 6.48% agreement"** as a thing to preserve — the openEMS leg
   has always modelled both conductors as 2-D PEC at z = 0 and z = h_sub, i.e.
   a 1.5 mm cavity. The two tools have never solved the same cavity. They do
   now, for the first time. Per `rfx-known-issues.md` (2026-08-28 A/B verdict)
   exactness of the cavity and agreement with the external reference point in
   opposite directions, so the number is expected to move and moving is not a
   regression.

Replace the opening of `claim_scope` (everything from "INTEGRATION/SMOKE" up to
and including "...omits the thirds-rule + reproduce-gate.") with:

> INTEGRATION/SMOKE diagnostic-reporter only (demoted from claims-bearing
> 2026-07-15, see docs/research_notes/20260715_cv05_first_principles_review.md):
> exercises probe-fed patch + finite-GP + CPML + NU-graded-z + openEMS lumped
> port end-to-end and REPORTS a coarse-mesh (dx=1mm) resonance. It does NOT
> establish accuracy. Since the lattice ownership contract (#931) both
> conductors are DECLARED PEC SHEETS on the substrate floor and top node planes
> (`sim.add_thin_conductor`), the z profile is built block by block so those two
> planes and the six 250 µm substrate cells between them are exact nodes, and
> the build asserts the realized wall planes against the declaration before any
> solve (`assert_realized_sheets`, reading `realized_pec_edge_masks` /
> `realized_wall_planes`; gated by
> tests/crossval/test_cv05_realized_sheet_planes.py). Before that migration the
> two conductors were 250 µm PEC Boxes — sub-cell on this mesh, so neither a
> volume nor a sheet — placed by a nearest-node argmin: walls at 12.0000 and
> 13.8161 mm, a 1.8161 mm cavity against a 1.5 mm laminate (+21.1 %) whose top
> 455 µm was air. The openEMS leg has always modelled both conductors as 2-D PEC
> at z = 0 and z = h_sub, so the two tools now solve the same 1.5 mm cavity for
> the first time; the pre-#931 rfx-vs-openEMS figures are therefore a comparison
> between two different cavities and are NOT a baseline this case preserves
> (rfx-known-issues.md, 2026-08-28 A/B verdict: exactness of the cavity and
> agreement with the external reference point in opposite directions). The
> realized in-plane patch is 28 x 37 mm from a 29.5 x 38.0 mm declaration — the
> faces sit half a 1 mm cell off the lattice, which is mesh resolution, reported
> in the result JSON, not absorbed. Authoritative patch-accuracy evidence
> remains DELEGATED to the committed tests in gate_paths, including the
> canonical thirds-rule far-field ENVELOPE LOCK
> test_patch_canonical_farfield_e4.py (committed openEMS reference 2.4221 GHz /
> 6.79 dBi; D within D_ABS_TOL_DB, f_res inside the [F_RES_REL_LO,
> F_RES_REL_HI] coarse-dx bias band — a regression lock, not an accuracy claim.

Keep the rest of the paragraph (the 2026-08-31 CORRECTION block and the MODE
IDENTIFICATION block) with the edits in §2 and §3 below. Both carry pointer
citations that `tests/contracts/test_evidence_citation_pointers.py` gates, so
they must keep NAMING their constants — cite
`tests/crossval/test_patch_canonical_farfield_e4.py:<lines> (D_ABS_TOL_DB,
F_RES_REL_LO, F_RES_REL_HI)` with the names present in the prose and the line
numbers refreshed against the file as it lands.

## 2. The `[+6%, +16%]` envelope — derivation, not translation

`F_RES_REL_LO = +0.06` / `F_RES_REL_HI = +0.16` / `D_ABS_TOL_DB = 1.0` live in
`tests/crossval/test_patch_canonical_farfield_e4.py`, which builds its OWN
canonical thirds-rule patch (its own 1-cell PEC Boxes at lines ~243-245) — it is
not cv05's script and it is inventoried under the **tests-crossval** group, not
here. That group re-declares and re-measures those three constants when it
migrates that build.

Two things follow, and they are the reason this is written down rather than
guessed:

* **The manifest must not carry the numbers.** The current text quotes
  `[+6%, +16%]` inline, which is how it ended up quoting the retired pre-#693
  band with the sign flipped for four days in August. The replacement text above
  cites the constants BY NAME only, so whichever value the tests-crossval group
  measures flows through without a second hand-edit anywhere. Apply it that way
  even if the constants turn out unchanged.
* **The band cannot be translated.** Its comment says it is "measured +11.3 %"
  and attributes that to the one-plane sheet realization: "the ground sheet's
  own cell sits inside the cavity as vacuum (+15%p, series-capacitance dilution
  of eps_eff), plus the one-plane length cost, minus ~4%p in-plane staircase",
  with a corrected (`two_plane`) figure of −4.7 % at dx = 2. The +15%p term is
  the mechanism #931 removes and the −4.7 % figure was measured on `two_plane`,
  which is not the volume rule that shipped. Neither number survives as an
  input. The band is re-measured from the migrated build or the lock is retired.

## 3. Fixture citations inside `claim_scope`

The MODE IDENTIFICATION block cites four values into
`tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json`. Three of
them are measurements of the old realization and move with the re-solve
(VESSL 369367259142 rebuilds the fixture into the branch worktree):

| citation | fate |
|---|---|
| `runs.baseline.modes[0].freq = 2.331855 GHz` (and its "−3.76 %") | re-read from the rebuilt fixture |
| `runs.patch_len_22p0mm.modes[0].freq = 2.993459 GHz` (criterion B) | re-read; **and re-check that 22.0 mm is still assigned TM110** — if the assignment changes, criterion B is no longer demonstrated by that length and a different length must be named, not a tolerance adjusted |
| `runs.patch_len_38p0mm.modes[1].freq = 2.609612 GHz` (the fired falsifier) | re-read; **the falsifier may or may not fire again.** It fired on the cluster build and not on the round-1 macOS build, so it was already build-dependent. If it does not fire, say so — do not delete the paragraph, and do not add an amplitude floor after the fact (the current text explicitly refuses that) |
| `_declared_TM100_hz = 2.423510 GHz` | **unchanged.** Closed form in εr / h / L / W, mesh-independent |

Also add one sentence to that block, because the fixture gains a key:

> Each run record in the fixture now carries `realized_stack` — the sheet
> planes, the node-to-node cavity and the realized footprints of the build that
> produced it, from `realized_pec_edge_masks`. The `_realized_x_cell_census` is
> re-measured on every rebuild rather than copied forward, and its unit changed
> with the measurement: it counts the tangential E edges the patch footprint
> stands on (the realized conductor length) instead of the masked nodes, so
> off-lattice rows read one lower than the pre-#931 block (29.5 mm → 28 edges,
> was 29 nodes) while on-lattice rows hold (22.0 mm → 22 either way).

## 4. `artifact_paths`

Add the new cv05 run record beside the old one; do not remove
`cv05_run_openems_369367257743.json`. Its `openems_*` keys and
`declared_modes_hz` are unaffected by the rfx contract, which makes it the
before half of a paired before/after witness for the resonance shift. The new
record's name follows the same `cv05_run_openems_<vessl run id>.json` pattern
with **369367259142**.

## 5. What this note deliberately does not settle

* The **exact replacement numbers** for §3. They are measurements; they arrive
  with the run. Writing plausible values here would be the hand-edited number
  the migration exists to remove.
* Whether the far-field envelope lock **survives at all**. If the migrated
  canonical build lands inside the patch's own bandwidth, a ±band around zero is
  a different kind of lock from a sign-locked bias band, and that is a decision
  for whoever re-measures it, not a re-tune.
* `docs/public/guide/benchmarks.mdx:108-114` quotes cv05's `two_plane` numbers.
  It is owned by the docs group, not by X-A; the critic flagged that the same
  artifact chain is marked `gpu` there and `external-solver` here. It is
  external-solver: the number comes from VESSL 369367259142, which runs openEMS
  from the image on CPU.
