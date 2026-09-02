# cv11's broad-E4 artifact is STALE, not unprovenanced — it reproduces exactly from a committed run of its own script

**Status:** OPEN (narrow) — **STALE reference artifact on a live chain.** The
provenance question is **SETTLED**; what remains is a refresh decision.
**Opened:** 2026-08-31 · **Issue:** #812 Phase 0, item 2 · **Author:** implementation agent
**Artifact:** `tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json`

This note is append-only from here on. Section 0 is a **correction of this
note's own first revision**, published the same day; it states what the
withdrawn claim was, verbatim, and why it was wrong.

---

## 0. CORRECTION (2026-08-31) — the first revision of this note was wrong

**Withdrawn claim.** The first revision of this note, and commit `2d05212`
that carried it, were titled and argued around:

> "cv11's broad-E4 artifact has no usable provenance and does not reproduce
> from its own script" — "This artifact's rfx leg does not reproduce from ANY
> committed run of its own producing script, and the stdout it names as its
> source is not in the repository."

**That is false.** It is withdrawn in full, together with the
`PROVENANCE-DISPUTED` label derived from it, everywhere it was written.

**Why it was wrong — the mechanism of the error.** The measurement in §2 of
the first revision rebuilt from the three cv11 stdouts **as they stand in the
working tree**. It never asked what those paths contained *at the artifact's
own commit*. One of them — `cv11_wr90_fresh_stdout.txt`, exactly the basename
that `source_cv11_stdout` names — was **added at `b0322c1` (2026-06-16, PR
#181), the same commit as the artifact, and later overwritten at `20e5533`
(2026-08-28, #724/#730)**:

```
$ git log --format="%h %ad %s" --date=short -- \
      tests/fixtures/waveguide_broad_e5/cv11_wr90_fresh_stdout.txt
20e5533 2026-08-28 fix(crossval): compare WR-90 against the guide the solve actually has (#724) (#730)
b0322c1 2026-06-16 feat(waveguide): rectangular_waveguide_port broad-E5 close (...) (#181)
```

So the contemporaneous stdout was in the repository the whole time, one
`git show` away, under the name the artifact records. Searching the current
revision of a path is not searching the repository. A **negative existence
claim** ("matches no committed run") requires the search that would refute it,
not the search that failed to confirm it; that search was not run before the
claim was published. The `/tmp/...` prefix on `source_cv11_stdout` is what made
the file look absent, and it is a real wart (§3) — but a wart on a recorded
*path string*, not a missing file.

**What is true instead** is §1 below: the artifact is **stale**, carrying a
number ~3.7x worse than today's code produces on a live broad-E5 leg. That is a
refresh question with a delta to explain, not a break in the evidence chain.
Severity accordingly drops from *evidence-chain break* to *stale reference
artifact*, matching the correction published on issue #812.

---

## 1. The settled finding

### 1.1 The artifact reproduces, bit-for-bit modulo float association

`scripts/diagnostics/build_waveguide_wr90_rectangular_broad_e4_comparison.py`
**runs no FDTD.** It parses the 4-way tables out of a cv11 stdout capture and
computes `||S_rfx| - |S_ref||`. The artifact is therefore fully determined by
the stdout it was fed, and reproducing it needs `git show` and a text
post-processor — nothing more. Measured 2026-08-31:

```
git show b0322c1:tests/fixtures/waveguide_broad_e5/cv11_wr90_fresh_stdout.txt > /tmp/s.out
PYTHONPATH=<repo>:<repo>/scripts/diagnostics python \
  scripts/diagnostics/build_waveguide_wr90_rectangular_broad_e4_comparison.py \
  --reference-column Palace_r_h2 --cv11-stdout /tmp/s.out --output-dir <scratch>
```

Rebuild vs `git show b0322c1:<artifact>`, compared field by field over the
flattened JSON:

| | |
|---|---|
| leaf fields, both sides | **80 / 80**, same key set |
| numeric fields | **54** |
| bit-identical | **52** |
| differing | **2**, at `5.3e-18` (`pairs[2].mean_mag_abs_diff`) and `1.7e-18` (`summary.mean_mag_abs_diff`) — float summation association, ~1 ulp |
| string fields differing | **1**: `source_cv11_stdout`, which records the input path given to the builder |

Headline reproduction: `status=passed geometries=3 pairs=5/5
max_mag_abs_diff=0.0707 mean=0.00943429`, and slab `S11` max `0.0707`, mean
`0.043976`, rfx `|S|` range `[0.0397, 0.5924]`, ref range `[0.0014, 0.5437]`.

**The chain is intact.** The artifact, its stdout, and its builder are one
commit and one command apart.

### 1.2 The artifact is stale by ~3.7x

The other cv11 stdouts in that directory — and the current content of
`cv11_wr90_fresh_stdout.txt` itself — are from **2026-08-28 (`20e5533`,
#724/#730)**. Rebuilding from each of them with the same command:

| quantity | artifact (2026-06-16) | `main_baseline` | `fresh` | `witness_np400` |
|---|---|---|---|---|
| summary `max_mag_abs_diff` | **0.0707** | 0.0186 | 0.0194 | 0.0193 |
| summary `mean_mag_abs_diff` | **0.009434** | 0.001782 | 0.001955 | 0.001954 |
| slab `S11` `max_mag_abs_diff` | **0.0707** | 0.0186 | 0.0194 | 0.0193 |
| slab `S11` `mean_mag_abs_diff` | **0.043976** | 0.007705 | 0.007771 | 0.007767 |
| slab `S11` rfx `\|S\|` range | **[0.0397, 0.5924]** | [0.0018, 0.5251] | [0.0018, 0.5243] | [0.0018, 0.5244] |
| slab `S11` **ref** `\|S\|` range | [0.0014, 0.5437] | [0.0014, 0.5437] | [0.0014, 0.5437] | [0.0014, 0.5437] |

So the E4-external leg of the `rectangular_waveguide_port` broad-E5 chain
publishes **0.0707** while the current code produces **0.0186--0.0194** on the
same comparison: the committed evidence is **3.6x--3.8x worse than the truth**,
i.e. it understates the family. Every committed run is *better* than the
artifact, so **no physics verdict is challenged** and no gate is at risk — the
artifact's own tolerances are `max_mag_abs_tol` 0.1 / `mean_mag_abs_tol` 0.07,
and 0.0707 sits inside them.

Last-digit note, so the two published readings reconcile: the stdouts' own
printed `[summary slab S11 vs Palace_r_h2]` lines read 0.0186 / 0.0193 / 0.0193;
the builder recomputes from the 4-decimal printed columns, which is why the
"fresh" rebuild reads 0.0194. The #812 audit reported 0.0186--0.0193 (the
printed summaries) and this note reports 0.0186--0.0194 (the rebuilds). Both
are correct; ~3.7x either way.

### 1.3 Candidate causes of the delta — named, not established

Three commits touched `validation/crossval/11_waveguide_port_wr90.py` between
the artifact and the current stdouts:

- `60ea8bf` (#340/#363, 2026-07-13) — cv11 implements its advertised per-freq
  `|S11|` gate;
- `2dcafdb` (#496/#574/#595, 2026-08-09) — cv11's `CPML_LAYERS = 20` replaced
  by a derived `int(ceil(0.75 * lambda_g_low / dx))`;
- `20e5533` (#724/#730, 2026-08-28) — the port aperture is trimmed explicitly
  so the extractor cutoff moves 6.241 -> 6.512 GHz against the 6.517 GHz
  closed form (a 4.82% term, per that commit's own error budget), and the
  analytic comparators move onto `A_WG_REALIZED`.

The last of these is the most likely dominant term and is *the direction that
would improve slab agreement*. **This is a hypothesis, not an attribution** —
no per-commit rebuild was run here, and running one is cheap (it is stdout
post-processing only if the intermediate stdouts exist; otherwise it is an
FDTD run per commit).

## 2. Why it matters where it is cited

The artifact is `validation/crossval/manifest.json`'s cv11 entry's **only**
`artifact_paths` element and the E4-external leg of the
`rectangular_waveguide_port` broad-E5 chain. Four places lean on it:

| Site | How it leans |
|---|---|
| `tests/crossval/test_waveguide_broad_e5.py:145,164` | two gates read it: pairs pass their tolerance, and it qualifies under the auditor's blocking-token rule |
| `scripts/diagnostics/port_external_reference_requirements.json` | listed in `external_comparison_artifacts` for the `add_waveguide_port` family, whose `current_status` is `broad_e5_passed` |
| `docs/guides/sparameter_support_matrix.json` / `.md` | quotes its `0.0707` / `0.00943` as the family's uniform Palace WR-90 numbers |
| `docs/guides/physics_validation_evidence_rule.md` | the `add_waveguide_port` row cites "external-solver ... artifacts exist" |

Each of those now says **stale, quote with the date**, not *disputed*.

## 3. The two real warts (both minor, both left standing)

1. **`source_cv11_stdout` records `/tmp/cv11_fresh.stdout`** even though a file
   of that basename is committed beside the artifact. The path string is what
   the builder was handed; it is not wrong about what happened, but it points
   a reader off-tree and is exactly what made the first revision of this note
   conclude the source was missing. The fix is to record the in-tree path plus
   the commit, which §5 asks a refresh to do.
2. **No `provenance` key existed at all** before #812 Phase 0. Contrast the
   sibling NU fixture
   `tests/fixtures/waveguide_nu_broad_e4/waveguide_wr90_nu_flux_broad_e4_comparison.json`,
   which carries `setup.commit = 6fd6ea0`, `dx_m`, `num_periods`,
   `cpml_layers` — the schema for doing this right already exists in-tree.

Neither wart is an evidence-chain break, and neither needs an FDTD run to fix.

## 4. What #812 Phase 0 did

Documentation and provenance only — **no gate value, tolerance, status field,
pair value or summary value was changed anywhere**, and no FDTD was run.

- A `provenance` block was written into the artifact recording the settled
  chain (`b0322c1` -> `cv11_wr90_fresh_stdout.txt` at that commit -> builder ->
  this file), the field-by-field rebuild result of §1.1, and the staleness
  measurement of §1.2, with the exact command a reader can rerun in seconds.
- The cv11 entry in `validation/crossval/manifest.json`, the cv11 row in
  `validation/README.md`, `docs/guides/sparameter_support_matrix.md` / `.json`,
  `scripts/diagnostics/port_external_reference_requirements.json`, and the
  `add_waveguide_port` row of `docs/guides/physics_validation_evidence_rule.md`
  now say **STALE (2026-06-16 number, current code gives 0.0186--0.0194)** at
  the exact place each quotes or lists the leg. The `PROVENANCE-DISPUTED`
  wording written by `2d05212` is withdrawn from all of them.
- The dispute was never written into the artifact's `claim_scope` or
  `evidence_level`, and still is not.
  `scripts/diagnostics/check_port_external_references.py` and
  `tests/crossval/test_waveguide_broad_e5.py:53-56,170` scan those two
  strings for blocking tokens (`narrow`, `partial`, `limited`, `only`, ...);
  writing a status word there would silently flip the family's audited status,
  which is a gate change and out of scope.

## 5. What remains, for a future lane

The provenance question is closed. **One decision remains: whether to refresh
the artifact to a current committed stdout.** The refresh is cheap — it is
`git show` plus the builder, still no FDTD — and the argument for it is that a
live chain should not publish a number 3.7x worse than the code produces.

If a refresh is done, it carries **one binding requirement**:

> **The 3.7x delta must be explained, not silently re-pinned.** Overwriting
> `0.0707` with `0.0186` and moving on would be a lock-value move justified by
> "the code now produces this value", which SPEC-00 §0.2.4 forbids without
> physics provenance. The refresh commit must attribute the improvement to a
> named change — §1.3 lists the three candidates and points at `20e5533` as the
> most likely — by measuring, not by asserting.

And it must, in the same commit:

1. record the in-tree stdout path and the producing commit in
   `source_cv11_stdout` / a `setup` block (wart 1, wart 2), copying the shape
   from the NU sibling fixture;
2. preserve the old `summary` and `pairs` values and this note as an
   append-only correction record stating what the number was, what it is now,
   and why it moved;
3. leave `max_mag_abs_tol` (0.1) and `mean_mag_abs_tol` (0.07) alone. The
   tolerances are not in question, and a refresh is not an occasion to move
   them. Note that `max_mag_abs_tol` = 0.1 is only **1.41x** the stale
   number it gates (0.0707) and **5.15x** the current one (0.0194); whether
   that gate can discriminate anything is a **separate** question, belongs to
   the #812 re-gate phases, and must not be folded into a refresh.

## 6. Reproduce every number above yourself (no FDTD, seconds)

```
R=<repo>
# 1.1 -- the artifact reproduces from its own contemporaneous stdout
git show b0322c1:tests/fixtures/waveguide_broad_e5/cv11_wr90_fresh_stdout.txt > /tmp/s_june.out
PYTHONPATH=$R:$R/scripts/diagnostics python \
  $R/scripts/diagnostics/build_waveguide_wr90_rectangular_broad_e4_comparison.py \
  --reference-column Palace_r_h2 --cv11-stdout /tmp/s_june.out --output-dir /tmp/rb_june
git show b0322c1:tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json \
  > /tmp/artifact_b0322c1.json
# then diff /tmp/rb_june/wr90_rectangular_broad_e4_comparison.json against it

# 1.2 -- staleness, from the three current stdouts
for f in main_baseline fresh witness_np400; do
  PYTHONPATH=$R:$R/scripts/diagnostics python \
    $R/scripts/diagnostics/build_waveguide_wr90_rectangular_broad_e4_comparison.py \
    --reference-column Palace_r_h2 \
    --cv11-stdout $R/tests/fixtures/waveguide_broad_e5/cv11_wr90_${f}_stdout.txt \
    --output-dir /tmp/rb_$f
done
```

Write rebuilds somewhere scratch — do not let one land on the committed
artifact (§5).
