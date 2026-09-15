# Replacement text for `validation/crossval/manifest.json`, cases 18 and 19

Owner of this file: the X-D migration agent (cv18, cv19). Owner of
`manifest.json`: the merge/ingest agent. Apply these edits there.

**Sequencing.** Both `claim_scope` strings quote measured digits that the
regeneration moves. Apply the STRUCTURAL rewrites below at ingest, and take
the digits from the regenerated fixtures
(`tests/fixtures/wr90_iris_modematch/fixture.json`,
`tests/fixtures/wr90_iris_filter/fixture.json`) rather than from this file.
`tests/contracts/test_evidence_numeric_provenance.py` requires 4 artifact-key
citations for case 18 and 4 for case 19; keep the citation style intact.
`artifact_paths` and `gate_paths` stay valid for both cases and need no edit.

---

## Case `18_wr90_iris_modematch`

### Sentences to DELETE (they describe a realization that no longer exists)

* "half-ulp-fragile node-plane box corners" as a standing property — the
  fragility was a property of the pre-#931 half-open NODE mask.
* "a fin footprint that made the electrical aperture d + 2*dx" stays (it is
  real history), but any sentence deriving the aperture from an OPEN-NODE
  count must go.
* "the fine rung's own effective aperture is half a fine cell wider than
  nominal" — this was measured against the old realization and is re-measured
  by the regeneration. Do not carry the digit across.

### Sentences to ADD

> Under the #931 lattice ownership contract a PEC volume realizes tangential
> walls at BOTH of its faces, so the iris this case draws t_c cells thick is
> realized t_c cells thick. That closes a defect this case carried from the
> start and never measured: it fed its mode-matching oracle the drawn
> t = 1.524 mm while the lattice realized (t_c - 1)*dx — 0.762 mm at a/30 and
> 1.143 mm at a/60 — because a body's far face was never a wall, and every
> assert in the case counted MASKED PLANES, a quantity that agreed with the
> drawing by construction. The whole record is regenerated on the corrected
> geometry; no envelope, gate or trace is carried forward. The case also gains
> the contract's ONE-CELL VOLUME WITNESS (fixture key
> `one_cell_volume_witness`): an iris-thickness sweep t = 1..8 cells at a/30
> against the lattice-blind mode-matching oracle, gated on the t = 1 residual
> lying inside the range the t = 2..8 rungs span. Until #931 nothing
> independent said the two-face rule was right AT ONE CELL — the thin-limit
> anchor is a t -> 0 statement, not a t = dx one.

Also note the corner-recipe inversion in one sentence:

> Every corner now sits ON a node plane. The pre-#931 "midpoint recipe" put
> them half a cell off because the volume mask was half-open over NODE
> coordinates; under cell-centre sampling a node-plane corner selects whole
> cells and the half-cell offset is the tie, so the recipe inverts.

### Fields whose meaning changed

Rows now carry `realized_aperture_cells` (= d_c, was `aperture_cells` = d_c-1),
`realized_thickness_cells`, `iris_wall_nodes`, `aperture_wall_nodes` and
`t_mm`. `schema_version` is 2.

---

## Case `19_wr90_iris_filter_aghanim`

### Sentences to DELETE

* "the cavity leg (L_c + 1)*dx — the distance between the bounding zeroed node
  planes — is confirmed to 0.04-0.17 cell and carries about 105 of the
  107.5 MHz" — keep as HISTORY only, explicitly past tense.
* "the IRIS-THICKNESS leg is NOT (t_c - 1)*dx … an irreducible
  comparator-input uncertainty of order half a cell" — the contract removes
  the ambiguity; keep the measurement as history, drop "irreducible".
* "Drawn counts are COMPENSATED (t_c = round(t/dx) + 1, L_c = round(L/dx) - 1)"
  — the compensation is deleted.
* "total = span - 1 + 2*sigma" and the face-continuity uniqueness discussion —
  total realized length is now plain addition.
* the "-101.4 MHz uncompensated" figure — it describes a realization that no
  longer exists.

### Sentences to ADD

> Under the #931 lattice ownership contract realized == drawn on all three
> legs (iris thickness, cavity, aperture), and this case reads all three off
> `rfx.boundaries.pec.realized_pec_edge_masks` at build time rather than
> asserting a locally written rule. The `+1` on the iris and the `-1` on the
> cavity are deleted together with the realization they cancelled; the drawn
> counts are the plain roundings and the BUILT FILTER IS UNCHANGED. That was
> verified, not assumed: the realized x wall planes come back at the committed
> pre-change `iris_x_nodes`
> `[[150,158],[214,222],[284,292],[354,362],[418,426]]`, with cavities
> 56/62/62/56 and apertures 40/26/24/26/40, and the FDFD comparator's
> `self_test` on the new inputs reproduces the committed block digit for digit
> (`empty_s11` 4.998689747642886e-14, `unitarity` 1.4655321400880439e-09,
> `metal_nodes_z` 9). One thing did move and is not compensated back: the
> metal span is now 276 cells rather than the compensated 277, so the trailing
> feed loses the extra cell it carried and the domain is symmetric about the
> iris stack; the P2 reference plane moves one cell into uniform guide.
>
> The `-0.68`-cell iris-thickness fit that this case recorded as an
> irreducible half-cell comparator-input uncertainty is superseded: it is best
> read as the signature of a missing far face plus a corner recipe that placed
> every face half a cell off the node planes, not as a residual physical
> property. It was never adopted as a fitted parameter, which was the right
> call. The `iris_thickness_zero_count_sweep` is re-centred accordingly — one
> full cell centred on the realized thickness instead of the one-sided
> 8.00-8.50 window that spanned the disagreement — and now measures
> sensitivity rather than ambiguity.

### The gating posture — a decision the ingest agent must record

Band edges and bandwidth were REPORTED, never gated, because a half-cell input
uncertainty is worth 22-40 MHz on them and cannot sit under a 15 MHz gate.
That uncertainty is gone. This migration deliberately left them REPORTED,
because a gate is set from a measured envelope and the envelope only exists
after the regeneration. With the post-change run in hand, either gate them at
`round-UP(measured envelope x 1.5)` over the nine-configuration population, or
state in `claim_scope` why not. Do not leave the old reason standing.

### Fields whose meaning changed

`gated_rfx.aperture_nodes` -> `aperture_wall_nodes` (realized WALL planes;
`[26,64]` -> `[25,65]` at a/90), `iris_x_nodes` -> `iris_wall_nodes` (same
values at a/90). `electrical_geometry.compensation` now begins "none".
`drawn_iris_thickness_cells` 9 -> 8, `drawn_cavity_cells`
[55,61,61,55] -> [56,62,62,56]; `iris_thickness_cells` and `cavity_cells` are
UNCHANGED, which is the preserving redraw's signature. `schema_version` is 2.
