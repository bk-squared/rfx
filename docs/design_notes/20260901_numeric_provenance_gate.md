# Numeric provenance gate — numbers in the crossval evidence surface must resolve to a committed artifact key

Issue #812, round 2. Append-only; corrections go in a new section, never by editing text above.

## 1. The measured problem

#812 shipped a wrong number into a durable document four times, and six of round 1's
nine blocking findings were a wrong or falsified claim in prose rather than a defective
gate. PR #814 closed the mechanical half of this class for *pointers* (`file.py:NNN` must
land on a constant assignment whose name the document also mentions). This note is the
same treatment for *values*.

## 2. The contract

A document cites a quantity by writing, inside a single-backtick code span,

    `<repo-relative .json path>::<key path>`
    `<repo-relative .json path>::<key path> = <literal>[ <unit>]`

`tests/contracts/test_evidence_numeric_provenance.py` resolves every such span in the opted-in
documents and asserts the artifact holds that value **to the precision written** —
tolerance is half a unit in the last decimal place of the literal as typed, times an exact
tabulated unit multiplier. More digits is a tighter assertion; a written sign is a signed
assertion, which is the round-1 cv17/cv18 failure shape. Any backtick span carrying the double-colon separator
that does not parse is an error, so a malformed reference cannot be silently skipped.

The unit table is a convenience for readable prose and **fails safe**: applying the wrong
multiplier turns the gate red, never green. Measured while writing this note — cv19's
`gates.f0_gate_mhz` already stores MHz, so citing it as `= 19.0 MHz` double-scales and the
gate rejects it by a factor of 1e6; it is cited unit-less instead.

## 3. Coverage and anti-vacuity

`REQUIRED_SITES` in the gate enumerates every opted-in document *and site* (a manifest case
id, a README case row, a design-note heading) with a floor on how many **value-checked**
references it must carry, plus a whole-surface census of references, value-checked
references, and distinct artifacts reached. Coverage therefore cannot shrink silently:
deleting a reference fails the floor, and lowering a floor is an explicit edit that belongs
in the same commit as its reason.

## 4. Falsifier, pre-declared

- **(A)** the opted-in documents pass on the committed tree, with every reference resolving
  and every value inside its stated precision.
- **(B)** the gate fails on a measured round-1 defect, for the right reason. The instance is
  cv15: the round-1 lane regenerated the committed rfx leg so that its dip depth moved from
  −4.4298 dB to −0.3448 dB, while the prose describing it did not move.
  `test_the_gate_fires_on_the_measured_cv15_regression` reproduces that artifact mutation
  (−0.3448 dB, the constant `CV15_REGENERATED_VALUE`) in a scratch tree and asserts the gate
  reports the document, the reference and both values. A second arm asserts the same for a
  sign inversion of the cited literal. The arm is anchored on whatever the live artifact
  holds, which is what makes it a falsifier rather than a frozen quotation.

  **Re-anchored 2026-09-06 (issue #920).** cv15's leg was regenerated again, this time
  because the fixture itself was wrong: the rfx feed post was floating between the two
  conductors instead of galvanically bridging them, so its dip was a correct extraction of
  the wrong circuit. On the pre-#931 conductor declarations (still one-cell PEC Boxes) the
  committed leg read `s11_dip_db = -21.9242`, and the two values above (−4.4298 and −0.3448)
  became pre-#920 history, no longer resolvable against that artifact.

  **Re-anchored again 2026-09-10 (issue #931, merge of #920 and #931).** #931 changed the
  conductor declarations too -- both ground and patch became zero-thickness SHEETS -- and,
  measured (not assumed), #920's galvanic-feed derivation and #931's independently-chosen
  full-substrate feed are algebraically the SAME span once both conductors are sheets, so
  the two fixes converge on one geometry rather than compounding into a third. The committed
  leg now reads
  `validation/crossval/_15_patch_results/rfx.json::s11_dip_db = -19.0480` (VESSL
  369367260032), and the #920-only value above (−21.9242, measured on the
  pre-#931 volume conductors) is itself now history, no longer resolvable against the current
  artifact. Criterion (B) is unaffected throughout every re-anchor: it fires whenever the
  artifact holds something other than what this document cites, and the injected −0.3448 dB
  is still a value no version of the leg holds. Every pre-fix leg is kept as its own
  resolvable artifact: `_15_patch_results/rfx_floating_post_1f005d0d.json::s11_dip_db =
  -4.4298` (main's archival name for the pre-#920 leg) and
  `_15_patch_results/rfx_pre931_two_plane_ground_1f005d0d.json` (this branch's archival name
  for the SAME bytes -- both kept, see the branch-merge resolution note), so the round-1
  narrative above keeps a resolvable artifact of its own.

## 5. What this gate does NOT catch

1. **Prose claims with no number** — "cannot fire at any width", "at the arithmetic floor".
   Round 1's cv05 false absolute is exactly this shape and is invisible here.
2. **Wrong-but-self-consistent artifacts.** If the harness writes a wrong value and the
   prose quotes it faithfully, both agree and the gate is green. This checks provenance,
   never physics.
3. **A number that is correct today but describes the wrong quantity** — citing
   `pairs[3].mean_mag_abs_diff` in a sentence about the max resolves fine and the sentence
   is still false.
4. **Numbers nobody chose to reference.** Bare numerics inside the opted-in sections are
   deliberately not flagged: these `claim_scope` strings carry dozens of legitimate bare
   numerics (issue and PR numbers, dates, drawn dimensions, cell counts, derived
   percentages), so a bare-numeric detector here would be a false-positive storm rather
   than a gate. The census in §3 bounds shrinkage, not incompleteness.

## 6. Scope

Opted in this round: `validation/crossval/manifest.json` (cases 11, 15, 17, 18, 19),
`validation/README.md` (the same five rows), the cv11 provenance note, and this note.
Repo-wide coverage is deliberately not attempted in one step.


## Re-basing two floors at merge time (2026-09-03)

This gate was drafted on 2026-09-01 against the README rows of that day. Before
it merged, #846 (cv17/cv18) and #847 (cv05/cv15) rewrote the cv15 and cv18 rows
with their own value-checked citations (2 each, resolved by the same
value-checked citation rule this gate enforces; #843, #842 and the cv22/cv23/cv24
lanes cited in that form from the start). The rebase keeps those newer rows —
they carry the reviewed physics (cv15's honest STOP, cv18's per-configuration
gate) — and re-registers their floors from 4 → 2 (cv15) and 3 → 2 (cv18) in
`REQUIRED_SITES`, in this same commit, which is exactly the deliberate act the
floor comment demands. The cv11 and cv19 manifest `claim_scope` texts, which no
later PR touched, keep this branch's citation-form rewrite (4 references each).
The gate test itself moved to `tests/contracts/` with the test reorg.
