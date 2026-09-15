# Replacement text for `validation/crossval/manifest.json` — group X-E

Group X-E does not own `manifest.json`. This file is the text X-E wants
applied; the merge/ingest agent applies it. Every edit below is confined to
the four cases X-E migrated (20, 21, 22, 23) plus one coupling note. Cases 15
and 19 are group C's and group D's respectively and are NOT written here —
the recompute-inconsistency between crossval-C ("case 15 claim_scope needs no
recompute") and crossval-E ("cpu-hour") is resolved in C's favour by
ownership: C re-runs cv15, so C's claim_scope text is the one that lands.

Apply AFTER the post-#931 runs land (run ids in
`scripts/vessl_issue931_post_XE/RECOMPUTE.md`) — every sentence below that
quotes a number is written to be true of the post-contract artifacts, and
applying it before them would put a claim in the manifest that no artifact
supports.

---

## case `20_msl_phase_referee` — APPEND to `claim_scope`

> LATTICE OWNERSHIP CONTRACT (#931, 2026-09-07). The rfx-side trace is a
> VOLUME: the 1-cell PEC Box in the fixture producer is drawn unchanged and
> now realizes tangential walls at BOTH of its faces (z = 250 and 300 um at
> dx = 50 um), a one-cell metal thickness, which is what openEMS Stage B
> already built. Before the contract rfx realized a single wall plane with Ez
> live through the metal, so this case's `conductor_thickness_one_cell`
> tolerance term (0.0121, 68% of B_BETA_ANALYTIC_TOL_FRAC) bounded a
> difference rfx did not have. The term's VALUE is unchanged and its
> DERIVATION was re-declared before the re-solve; had the contract made rfx's
> realized thickness zero the term would have been deleted, not re-tuned. The
> sheet alternative was measured and rejected: 254 um is off-lattice at
> dx = 50 um, a sheet snaps to the 250 um node and leaves 50 um of realized
> laminate above the strip (a buried strip, not this board), and its realized
> width is 550 um against the drawn 600 um. Redrawing on-lattice
> (dx = h_sub/5 = 50.8 um) stays rejected for this lane's own recorded
> reason: it destroys the on-lattice `ref_plane_shift`.
>
> Both the rfx fixture and the openEMS Stage B artifact are superseded, not
> patched. `meta['trace_y_lo/hi_realized_m']` move one cell (950/1550 ->
> 900/1500 um) and `meta` gains `trace_wall_planes_realized`,
> `trace_wall_planes_realized_z_m`, `trace_realization_kind` and
> `t_metal_realized_m`; `h_sub_realized_m` (300 um), `n_z_sub_realized` (6)
> and `w_trace_realized_m` (600 um) do not move. Stage B asserts its own
> metal thickness against `t_metal_realized_m` rather than stating the
> equality in prose.

## case `20_msl_phase_referee` — REPLACE in `references[0].name`

Replace the parenthetical

> ", NOT cv06b's own runtime u any more: cv06b's #723 fix computes u from ITS
> realized 635.0um trace width, not the declared 600um"

with

> ", NOT cv06b's own runtime u any more: cv06b's #723 fix computes u from ITS
> OWN realized trace width. Under #931 cv06b's realized width is re-derived
> in its own group (X-B) and this entry must be re-read against that number
> once it lands; the 635.0 um quoted here was measured on the pre-contract
> node sampler."

## case `21_coax_two_port_referee` — APPEND to `claim_scope`

> LATTICE OWNERSHIP CONTRACT (#931): OUT OF SCOPE and unchanged. rfx's coax
> pin and shell are a sigma stamp (`stamp_coaxial_line` writes PEC_SIGMA into
> `materials.sigma`), which design note section 1.8 fences out of the contract
> as a lossy-volume model; they never enter `pec_mask` or the realized edge
> set. No geometry moves and no committed artifact is invalidated. What the
> contract does NOT resolve stays recorded rather than half-migrated: the
> shell spanning [b - dz, b], `r_os = b + 2*dx`, and `B_Z0_OHM` computed on the
> nominal a-to-b annulus are all compensations for a conductor model that has
> no declared realization at all. That is the follow-up issue section 1.8
> names.

## cases `22_dispersive_slab_fresnel` and `23_lossy_slab_fresnel` — APPEND to `claim_scope`

> LATTICE OWNERSHIP CONTRACT (#931): CONTROL. This case contains no conductor
> — cv22's slab is written straight into MaterialArrays, cv23's api arm adds a
> dielectric Box — and dielectric sampling (node, half-open) is untouched by
> the contract (section 1.8). The committed artifacts were re-run under the
> contract and are bit-identical [ingest: state the run id and the measured
> verdict here]. cv23's api arm now asserts the premise instead of testing it
> with a `pec_mask` query: a sheet owns no cell, so a `pec_mask`-only test
> would report "no conductor" for a case that had declared one; the arm passes
> the `pec_sheets`/`pec_wires` collectors and calls `assert_no_conductor`.

## cases 22 and 23 — `references` (both already list `docs/public/guide/materials-geometry.md`)

No text change here, but the reference is now load-bearing for a different
reason and the docs group must land its rewrite first: that page states the
Box/material semantics, and under #931 it has to state the volume/sheet/wire
split. Flagged so the two are not merged out of order.
