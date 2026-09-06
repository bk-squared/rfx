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
  cv15: the round-1 lane regenerated the committed rfx leg so that its `s11_dip_db`
  moved from −4.4298 dB to −0.3448 dB, while the prose describing it did not move.
  (The live value that key holds today is cited in the appended 2026-09-06 section;
  see it for why the reference moved out of this paragraph. The two numbers here are
  round-1 history and are unchanged.) `test_the_gate_fires_on_the_measured_cv15_regression`
  reproduces that artifact mutation in a scratch tree and asserts the gate reports the
  document, the reference and both values. A second arm asserts the same for a sign
  inversion of the cited literal.

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

## The cv15 anchor moves with the artifact (appended 2026-09-06, issue #912)

Appended, not edited above. Issue #912 regenerated the wire-port crossval records
that predate PR #897 (the Yee half-step current-DFT phase correction). One of them
is this note's criterion-(B) anchor, `validation/crossval/_15_patch_results/rfx.json`,
whose producer is `validation/crossval/15_patch_antenna_rt5880.py rfx`. The
regenerated leg (VESSL run 369367258715 on remilab-c0, repo SHA
`f5ee3b59c3e83832f7df1682f0ad30abb260a42e`, SHA-guarded in the job) holds

  `validation/crossval/_15_patch_results/rfx.json::s11_dip_db = -0.3182` dB

so the reference in §4 — which asserted −4.4298 on this same key — no longer
resolves. Written root cause, measured, before anything was changed:

1. The regeneration is legitimate and the artifact was the stale side. In the same
   job an A/B control leg ran the identical script at `aa888b7a` (PR #897's
   immediate parent). It reproduced `s11_dip_db = -0.3448075` — i.e. the round-1
   value this note records in §4, to 6e-7 relative. The committed −4.4298 does not
   reproduce at either SHA.
2. The −4.4298 → −0.3448 move is **not** PR #897. It is already fully present at
   #897's parent. The committed leg dates from `1f005d0d` (PR #768, 2026-08-29) and
   PRs #776 (`7c80714c`, whole-port driven diagonal) and #777 (`ad13b4c7`,
   uniform-lane POST flip + decomposer recalibration) rewrote the wire-port
   one-port diagonal on 2026-08-30/31, after it. Round 1 regenerated the leg
   correctly and the #812 lane reverted that regeneration; the revert restored an
   artifact the committed solver no longer produces.
3. PR #897's own contribution, isolated on the A/B pair, is the advertised
   rotation and nothing else: `s11_dip_db` −0.3448075 → −0.3181958 (+0.0266 dB),
   `f_dip_hz` unchanged, `max |dS11|` 0.007284 over the band, and the measured
   per-bin current rotation is `pi f dt` to 6.4e-6 rad with the run's own
   `dt = 1.5134e-12` s recovered to 1 part in 1e4 from the fit.

What changed here, and what did not. The §4 sentence keeps both of its historical
numbers verbatim; only its *citation form* was dropped, because a live artifact
reference cannot also be a historical one. The live reference now sits in this
section, so `_cv15_dip_references()` still finds an anchor,
`test_the_gate_fires_on_the_measured_cv15_regression` still fires on the
−0.3448069 mutation (which is no longer the committed value either), and the
reference census is unchanged (one reference moved, none added or removed). **No
gate, tolerance, window, floor or `REQUIRED_SITES` entry was changed by #912.**

What this episode says about the gate itself, stated plainly because it is the
uncomfortable half: this gate is a provenance instrument, and §5.2 already says it
cannot tell a wrong-but-self-consistent artifact from a right one. It held a stale
cv15 leg green for six days (2026-08-31 → 2026-09-06) because the prose and the
artifact agreed with each other. What caught it was re-running the producer, not
this gate. A freshness instrument — "the committed leg reproduces on today's
solver" — is a different check and this note does not claim to be one.
