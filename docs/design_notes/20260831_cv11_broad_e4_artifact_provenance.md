# cv11's broad-E4 artifact has no usable provenance and does not reproduce from its own script

**Status:** OPEN — PROVENANCE-DISPUTED. Cannot be closed in issue #812 Phase 0
(documentation and provenance only; no gate, no threshold, no physics, no FDTD run).
**Opened:** 2026-08-31 · **Issue:** #812 Phase 0, item 2 · **Author:** implementation agent
**Artifact under dispute:** `tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json`

This note is append-only. A later correction is a new dated section, never a
silent rewrite of what is written here.

---

## 1. Why this artifact matters

It is `validation/crossval/manifest.json`'s cv11 entry's **only**
`artifact_paths` element, and it is the **E4-external leg** of the
`rectangular_waveguide_port` broad-E5 chain. Four places lean on it:

| Site | How it leans |
|---|---|
| `tests/test_waveguide_broad_e5_envelope_gates.py:145,164` | two gates read it: pairs pass their tolerance, and it qualifies under the auditor's blocking-token rule |
| `scripts/diagnostics/port_external_reference_requirements.json` | listed in `external_comparison_artifacts` for the `add_waveguide_port` family, whose `current_status` is `broad_e5_passed` |
| `docs/guides/sparameter_support_matrix.json` / `.md:334` | quotes its `0.0707` / `0.00943` as the family's uniform Palace WR-90 numbers |
| `docs/guides/physics_validation_evidence_rule.md` | the `add_waveguide_port` row cites "external-solver ... artifacts exist" |

## 2. What was verified on 2026-08-31 (no FDTD; all from committed data)

The artifact's producing script,
`scripts/diagnostics/build_waveguide_wr90_rectangular_broad_e4_comparison.py`,
**runs no FDTD**. It parses the 4-way tables out of a cv11 stdout capture and
computes `||S_rfx| - |S_ref||`. The artifact is therefore fully determined by
the stdout it was fed — which makes it exactly reproducible, and makes a failure
to reproduce meaningful rather than noise.

Method: ran that builder with `--reference-column Palace_r_h2` (the column the
artifact declares) against each of the three cv11 stdouts committed in the same
fixture directory.

| quantity | this artifact | `cv11_wr90_main_baseline_stdout.txt` | `cv11_wr90_fresh_stdout.txt` | `cv11_wr90_witness_np400_stdout.txt` |
|---|---|---|---|---|
| slab `S11` `max_mag_abs_diff` | **0.0707** | 0.0186 | 0.0194 | 0.0193 |
| slab `S11` `mean_mag_abs_diff` | **0.043976** | 0.007705 | 0.007771 | 0.007767 |
| summary `max_mag_abs_diff` | **0.0707** | 0.0186 | 0.0194 | 0.0193 |
| summary `mean_mag_abs_diff` | **0.009434** | 0.001782 | 0.001955 | 0.001954 |
| slab `S11` rfx `\|S\|` range | **[0.0397, 0.5924]** | [0.0018, 0.5251] | [0.0018, 0.5243] | [0.0018, 0.5244] |
| slab `S11` **ref** `\|S\|` range | [0.0014, 0.5437] | [0.0014, 0.5437] | [0.0014, 0.5437] | [0.0014, 0.5437] |
| slab `S21` rfx `\|S\|` range | **[0.8443, 1.0014]** | [0.8438, 1.0000] | [0.8438, 1.0000] | [0.8438, 1.0000] |

Three things this table says, in order of force:

1. **The reference leg reproduces exactly; only the rfx leg does not.** The
   Palace_r_h2 column is bit-for-bit the same in the artifact and in all three
   committed runs. So this is not a "different reference file" story.
2. **The rfx column is a different computation, not a noisier sample of the same
   one.** The `|S11|` peak differs by ~0.068 and the null floor by ~0.038.
   Scatter across the three committed runs is 0.0008 in the peak; the gap to the
   artifact is eighty times that.
3. **The artifact is 3.6x--3.8x worse than every committed run** (5.7x on the
   mean). Also, its rfx `|S21|` reaches 1.0014 — non-passive by 0.0014 — where
   no committed run exceeds 1.0000.

Note on the last digit: the stdouts' own printed
`[summary slab S11 vs Palace_r_h2]` lines read 0.0186 / 0.0193 / 0.0193. The
builder recomputes from the 4-decimal printed columns, which is why the "fresh"
rebuild reads 0.0194 rather than 0.0193. The #812 audit comment reported the
range as 0.0186--0.0193 (the printed summaries); both readings are correct and
neither changes the conclusion. The audit's "3.7x" is confirmed.

## 3. What is missing

- `source_cv11_stdout` = `/tmp/cv11_fresh.stdout`. A `/tmp` path cannot be
  inspected by any reader, cannot be diffed, and does not survive the machine it
  was written on. There is no in-tree copy.
- No `setup` block: no commit, no `dx`, no `NUM_PERIODS`, no `CPML_LAYERS`, no
  run id for the rfx leg. (Contrast the sibling NU fixture
  `tests/fixtures/waveguide_nu_broad_e4/waveguide_wr90_nu_flux_broad_e4_comparison.json`,
  which *does* carry `setup.commit = 6fd6ea0`, `dx_m`, `num_periods`,
  `cpml_layers` — so the schema for doing this right already exists in-tree.)

## 4. What this does NOT establish

**The discrepancy is unexplained, not proven to be an error.** Verified dates and
intervening changes:

- The artifact was written once, at `b0322c1`, 2026-06-16 (PR #181), and has
  never been regenerated (`git log` on the file returns exactly one commit).
- The three committed stdouts are dated 2026-08-28 (`20e5533`, #724/#730).
- At least two physics-relevant changes to cv11 land in between: `2dcafdb`
  (#595, 2026-08-09) replaced cv11's `CPML_LAYERS = 20` with a derived value —
  on main it reads `int(ceil(0.75 * lambda_g_low / dx))` — and `20e5533`
  (#724, 2026-08-28) changed the guide the comparison is run against ("compare
  WR-90 against the guide the solve actually has"). The committed baseline
  stdout's own header says as much: the pre-#724 fixture "is not a baseline".

So a legitimate configuration difference is a live hypothesis. What *is*
established is only that **the artifact cannot be checked**: nothing in the tree
reproduces it, and nothing in the tree records what produced it.

Equally: **no physics verdict is challenged here.** Every committed run is
*better* than the artifact. This is a traceability dispute.

## 5. What was done in #812 Phase 0

Annotation only — **no gate value, tolerance, status field, pair value or summary
value was changed anywhere**, and no FDTD was run.

- A `provenance` block (and a `provenance_status` key) was added to the artifact
  itself, recording everything in sections 2--4 plus section 6.
- The cv11 entry in `validation/crossval/manifest.json` and the cv11 row in
  `validation/README.md` were marked PROVENANCE-DISPUTED.
- `docs/guides/sparameter_support_matrix.md`, `.json`,
  `scripts/diagnostics/port_external_reference_requirements.json`, and the
  `add_waveguide_port` row of `docs/guides/physics_validation_evidence_rule.md`
  were marked at the exact place each quotes or lists the leg.

The dispute was deliberately **not** written into the artifact's `claim_scope`
or `evidence_level`. `scripts/diagnostics/check_port_external_references.py`
(lines ~307--320) and `tests/test_waveguide_broad_e5_envelope_gates.py:53-56,170`
scan those two strings for blocking tokens (`narrow`, `partial`, `limited`,
`only`, ...); writing the dispute there would silently flip the family's audited
status, which is a gate change and out of scope for Phase 0. Verified: the
auditor's JSON output is byte-identical before and after these edits apart from
its own timestamp.

## 6. What would settle it, and what must NOT be done

### Would settle it

A cv11 run at **this artifact's own declared configuration** — which first
requires recovering that configuration, since the artifact does not record it;
`b0322c1` is the commit at which it was written and is the starting point — with:

1. the run's **stdout committed in-tree**, next to the existing three, with a
   header stating branch/commit/date/`dx`/`NUM_PERIODS`/`CPML_LAYERS`, matching
   the convention the three committed stdouts already use;
2. a `setup` block added to the artifact recording commit, `dx`, `num_periods`,
   `cpml_layers`, and run id — copy the shape from the NU sibling fixture;
3. the rebuild rerun from that committed stdout.

Then exactly one of:

- **(a)** the rebuild reproduces `0.0707`, and the artifact is vindicated with
  its configuration finally on the record — and the 3.7x gap to today's main is
  attributed (to #595's absorber change, to #724's realized guide, or to
  something else named);
- **(b)** it does not, and the artifact is **withdrawn** from the broad-E5 chain
  with the reason written down — at which point the family's `broad_e5_passed`
  status must be re-derived honestly rather than assumed.

This needs an FDTD run, which #812 Phase 0 forbids. It belongs to a future lane.

### Must NOT be done

**Do not silently re-pin the artifact to whatever a fresh cv11 run produces.**

Regenerating on current main and overwriting `0.0707` with, say, `0.0193` would
look like a fix and would be the opposite of one:

- it **erases the discrepancy instead of explaining it**. The question is not
  "what does cv11 print today" — the three committed stdouts already answer
  that. The question is "what produced the number that has been carried as
  passed E4 evidence since 2026-06-16", and a fresh run cannot answer it;
- it destroys the only surviving evidence that a 3.7x-divergent rfx leg was once
  committed as passed evidence, leaving the next reader no way to tell the
  number ever moved;
- it would be a lock-value move justified by "the code now produces this value",
  which SPEC-00 §0.2.4 forbids without physics provenance.

If the artifact is regenerated for any reason, the old `summary` and `pairs`
values, this note, and the artifact's `provenance` block must be preserved as an
append-only correction record stating what the number was, what it is now, and
why it moved.

Two further don'ts:

- **Do not** encode the dispute in `claim_scope` / `evidence_level` (see §5).
- **Do not** loosen or tighten `max_mag_abs_tol` (0.1) or `mean_mag_abs_tol`
  (0.07) as part of settling this. The tolerances are not what is in question.

## 7. Reproduce section 2 yourself (no FDTD, seconds)

```
cd scripts/diagnostics
for f in main_baseline fresh witness_np400; do
  PYTHONPATH=<repo>:<repo>/scripts/diagnostics python \
    build_waveguide_wr90_rectangular_broad_e4_comparison.py \
    --cv11-stdout ../../tests/fixtures/waveguide_broad_e5/cv11_wr90_${f}_stdout.txt \
    --reference-column Palace_r_h2 --output-dir /tmp/repro_$f
done
```

Then compare each `/tmp/repro_*/wr90_rectangular_broad_e4_comparison.json`
against the committed
`tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json`.
Write the outputs somewhere scratch — do not let a rebuild land on the committed
artifact (see §6).
