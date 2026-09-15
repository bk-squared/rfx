# T6 — the places the design note left something open, and what T6 chose

Branch `feat/931-t6-oracle-contracts-locks-studio`. Each item states the
question, the choice, and what would change the choice. Nothing here is a
recommendation to a future reader; these are decisions already taken in the
branch, written down so they can be reversed knowingly.

## 1. `add_thin_conductor` handed a ONE-CELL Box vs a zero-extent Box

Open in the note: three oracle fixtures spelled the same operator two ways and
both passed, with nothing saying which was normative.

**Chosen: both are legal and they land on the same plane, by the tie rule
already in §1.3 — no new rule.** A Box from node `k` to node `k+1` has its
mid-plane exactly half a cell above `k`, which is the tie, and a tie resolves to
the LOWER plane. One cell stays the ceiling: `refuse_thick` rejects anything
thicker ("not a sheet; use add() for a volume"). Pinned in
`tests/contracts/test_lattice_ownership_contract.py::test_add_thin_conductor_takes_both_sheet_spellings_and_lands_on_one_plane`
and measured on the real fixture
(`tests/oracle/test_sheet_film_rta_analytic.py`, x-node 110 either way).

Reverse this only by making one spelling an error, which would break
`test_sheet_film_rta_analytic` for no physical reason.

## 2. `_realized_extent` in the patch-board locks: node census or wall planes?

Open: the critic's "one thing I would fix first" was to re-point
`_realized_extent` at `realized_wall_planes`.

**Chosen: leave it on the node census, and assert the exact one-cell relation
instead of hiding it.** The reasoning is in the commit and in the code comment:
the electrical patch is the edges BETWEEN the census nodes, one cell shorter,
and it always was — before #931 `fidelity_report` shared the node reading, so
the disagreement was invisible rather than absent. Closing it would raise Board
H's Balanis anchor by about 43/42 and move Leg A's centre from -6.17 to roughly
-8.3 pp with NO field re-run, and that number must come from a fresh
configuration sweep. Design note §1.8 fences the inclusive-+1 (#729 class) out
of the ownership contract; T6 keeps it fenced and writes the cost down.

`test_realized_raster_agrees_with_the_public_fidelity_report` now asserts
`fidelity == census - one cell` exactly, so the gate still fails if EITHER
reading drifts.

Reverse this when #729 is worked, and re-derive Leg A from a sweep in the same
change — never by re-centring the window.

## 3. cv15's negative control after `two_plane` is gone

Open in the inventory: with the flag deleted, the #740 one-plane-ground build is
unconstructible, so the mode-pair ratio band's rejection endpoint rests on a
build that cannot be rebuilt. Freeze as history, or retire the derivation?

**T6 does not own cv15** (crossval group C owns the script, the fixtures and
`manifest.json`). T6's position, for the record: freeze the defect leg as
history with a note that it is no longer buildable, and keep the STOP. The
replacement negative control the phase-2 brief already names — a
`ground_plane_z` that selects a wrong plane — is buildable under the contract
and reproduces the same class of defect (a wall in the wrong place), so the band
derivation survives with a constructible rejection endpoint.

## 4. Design IR: v1 relaxed, or v2 with a migration?

**Settled by stage B before T6 touched it: v2, with a named refusal.**
`DESIGN_SCHEMA_VERSION = "rfx-design-ir/v2"`, the schema file is
`rfx-design-ir-v2.schema.json`, and a v1 document is refused with a message that
names `two_plane`, says what the realization change was, and tells the author to
re-export. T6 added the coverage that was missing: a PEC-sheet round-trip
fixture, a v1-refusal test, and a stray-`two_plane`-key refusal test in
`tests/studio/test_interop_design_document.py`.

## 5. `tests/fixtures/experiments/patch_antenna_cpu_v1.json`

Open in the inventory: leave the 2 mm metallization (now a filled slab) or
redraw it as sheets, as the v2 golden already was.

**Chosen: leave it.** That fixture's job is to prove the v1 -> v2 MIGRATOR
works; its `model.ground.thickness_m = 0.002` is the input the migrator
consumes, and zeroing it would test a different migration. A spec that declares
a 2 mm thickness SHOULD migrate to a 2 mm volume — that is the migrator being
correct, not the fixture being wrong. The v2 golden
(`patch_antenna_v2.json`) already carries zero-thickness ground and patch, which
is where the physics statement belongs.

## 6. `tests/_realized_geometry.realized()` cannot read an f0-only fixture

Not a design-note question, but a shared-helper gap T6 hit and worked around:
`realized(sim)` calls `realized_pec_edge_masks`, which refuses a run with no
cell mask, no PEC sheet and no wire — correct for the contract, wrong for a
fixture whose only conductors are surface-impedance sheets
(`tests/oracle/test_leontovich_alpha_oracle.py`). T6 read `sheet_specs` directly
there and said so in the test's docstring. If the helper grows an
`allow_empty=` or returns zero masks for a shaped grid, that test can use it.

## 7. Sigma-promoted conductors reach the contract by TWO doors

`sim.add(Box, material='pec_like'|'copper')` crosses
`Simulation._PEC_SIGMA_THRESHOLD` in `rfx/api/_compile.py` and becomes a PEC
BODY; a raw `rasterize(grid, [(shape, eps, 1e10)])` stays a sigma FILL and §1.8
fences it out. Both are spelled "PEC" by their fixtures. T6 pinned that the two
are not equated
(`test_a_sigma_fill_conductor_is_not_a_pec_body`: 910 vs 912 cells on the same
sphere, and no realized edge on the sigma path) and measured the consequence for
the chain battery (`T6-waveguide-chain-battery.md`). The design note's §1.8
would be clearer for one sentence saying which door a conductivity came through
decides which model it gets.

## 8. A foil board with a RESERVED cell: extend the dielectric, or move the foil?

Not open in the note's text, but forced by it. §3's `sheet_slot_vacuum` finding
names the situation ("a stack-up drawn with a slot for the foil") and gives the
remedy as "extend the dielectric boxes to the sheet plane". §6's stack-up
amendment gives a different one: "a foil sheet goes on the dielectric INTERFACE
it bounds". On the three patch boards the two disagree, and the disagreement is
worth 20 % of the cavity:

* **Extend the dielectric.** The laminate is drawn from the foil's plane, so it
  reads 983.75 µm where the board says 787. The pinned numbers come back
  because this reproduces the pre-#931 electrical board exactly — which is the
  reason to distrust it. It restores a compensation in the drawing after
  deleting it from the code, and the fixture then declares a laminate thickness
  the datasheet does not have.
* **Move the foil to the interface** (chosen). The laminate stays 787 µm, the
  cavity is four cells of it, node-to-node equals face-to-face, and the pins are
  re-derived from a fresh run.

**Chosen: move the foil**, on three grounds that are checkable rather than
aesthetic. (1) The board's own MSL port already declares this stack — its foot
at the laminate bottom, its height `H_SUB` — so before the redraw the port's
ground reference and the realized ground wall stood one cell apart. (2) Migration
rule 3 says the compensation is deleted, not re-tuned, and a stretched dielectric
is the same compensation wearing the drawing's clothes. (3) It is the only one of
the two that makes preflight's own #703 cavity check go silent; the other leaves
it printing +84.5 % on a board whose gate is green, which is how #702 stayed
hidden for a year.

The cost is on the record: three lock modules re-pin from VESSL 369367259225 /
369367259226 / 369367259227, and their measured half-widths are inherited from a
mesh/domain ladder that has NOT been re-run on the redrawn board. Those widths
are therefore an upper bound carried forward, not a fresh measurement — a
re-measured ladder is the follow-up, and it can only narrow them.

Reverse this only with a board whose datasheet thickness really is the
node-to-node distance. Then the drawing says so and the two remedies agree.

## 9. The two sheet-cavity modules moved the wrong way — instrument, don't re-pin

`test_sheet_resonance_position_ab` and `test_sheet_perturbation_q` declare
zero-thickness sheets on exact node planes, and the contract's closed footprint
gives their patch the 5.500 mm it draws instead of the 5.250 mm the old
half-open sampling gave it (measured at build time at HEAD: Ex rows 13..34 on
both patch planes). A cavity mode set by that length must FALL. Both modules'
modes rose (30.2153 GHz against a length-scaled 26.85; 25.3992 against 23.63),
and the A/B module reproduces its number at HEAD on this pod, so it is not
staleness.

**RESOLVED 2026-09-07 by the census run (VESSL 369367259230).** The peaks the
old pin named are still there; the SELECTION moved. `base` takes the two
loudest, and the census shows the old second pin (28.1318) sitting at 27.9141
GHz — within one `df` of where it was — at 0.117 of the loudest, while a
0.418-amplitude line at 30.2153 took its place in the top two. So neither
"the modes moved by the length ratio" nor "nothing changed" is right: the
footprint really did grow to the drawn 5.500 mm, the lines moved by much less
than the length ratio, and the ranking changed which two the pin reads. Both
modules are re-pinned from their own runs (the A/B pair to (25.1741, 30.2153)
GHz, the perturbation-Q sibling's mode-tracking pin to 25.399 GHz), widths and
thresholds untouched, and the census stays printed so the next reader sees the
ranking rather than inferring it.

What is NOT fixed, and is now visible rather than hidden: amplitude rank is not
a mode label. This module has no parity check like the harminv board's ("MODE
IDENTITY — PARITY, NEVER AMPLITUDE RANK"), so a rank swap and a moved mode look
identical to it. That is the follow-up. It does not block the gate, which
compares f0 against PEC at the same frequencies in every arm and so does not
care which two peaks are chosen, only that the choice is common.

**Chosen at the time: add the instrument, leave the pin red.** The modules reported two
headline frequencies with no trace, so a real move and a two-loudest PEAK PICKER
swapping peaks between arms look identical from the outside — an R5 gap. The
census (every peak, every arm, with amplitudes) is committed and the re-run is
VESSL 369367259230 / 369367259231. A provenance pin re-centred on a number whose
mode identity is unknown is worse than a red gate, so nothing is re-centred until
that census says which mode is which.
