# cv18 regeneration under the #931 lattice ownership contract

Branch `feat/931-wr90-iris`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XD-wr90iris`.
Pre-change baseline: VESSL run **369367259008** (`rfx-931-base-cv18`), which
reproduced the committed fixture BIT-EXACTLY (0 of 2075 numeric keys differ),
so any post-change delta is attributable to the contract and not to the
environment.

## Why this case must be re-solved (it is not a preserving redraw)

cv18 has always handed its mode-matching oracle `t = 1.524 mm` while the
lattice realized `(t_c - 1)*dx` — 0.762 mm at a/30, 1.143 mm at a/60. Under
the contract a PEC volume realizes walls at both faces, so the drawn box now
realizes `t_c*dx` = 1.524 mm and the oracle input is correct for the first
time. The aperture is unchanged (walls at y-nodes `fin_c` and `cells - fin_c`
under both rules), so the iris thickness is the single variable that moved.
Every committed trace, envelope and gate is therefore void and is regenerated
— none is carried forward or re-tuned.

## MEASURED so far — pass 1, first two gated rows

Run 369367259159 was in its gated fine rung when this was written. The two
completed rows already show the direction the correction predicts: the oracle
was being fed an iris 25% thicker (fine rung) than the one the lattice built,
and closing that gap shrinks the disagreement.

| config | committed max abs S11 gap | post-change | per-config gate (pre-change) |
|---|---|---|---|
| d = 18.288, glen 0.20, frac 0.50 | 0.0122 | 0.0079 | 0.019 |
| d = 12.192, glen 0.20, frac 0.50 | 0.0223 | 0.0101 | 0.034 |

Both PASS their pre-change per-configuration gates, which is a weak statement
(the gates were set from the worse numbers). The number that matters is the
new envelope, and the new gates derived from it — see below. Do not quote
these two rows as the result; quote the regenerated record.

## Commands

Two passes, and the second is not optional.

The `--write-fixture` self-check demands EXACT equality between each gate
constant and `round-UP(measured envelope x 1.5)`. Pass 1 runs with the
pre-change constants still in source, so it prints `ENVELOPE/GATE MISMATCH`
and exits 1 **by design**; its printed envelopes are what the constants are
re-derived from. Pass 2 runs the identical command with the updated constants
and produces the committable record. The traces are deterministic, so pass 2
reproduces pass 1's traces and changes only the `gates` block and the
per-row `fine_gate_abs` fields.

```
# pass 1 and pass 2 use the SAME file
vessl run create -f validation/crossval/_18_wr90_iris_results/vessl_931_post_cv18.yaml
```

Local equivalent (do not run on the shared pod; ~2.9 h CPU):

```
JAX_PLATFORMS=cpu python validation/crossval/18_wr90_iris_modematch.py --write-fixture
```

Expected runtime: ~2.9 h CPU per pass (8 fine rows at ~800 s, 8 coarse at
~66 s, 4 modal, 3 raw, 8 truncation, plus the new 7-row thickness sweep at
~8 min). Job timeout is set to 21600 s.

| pass | VESSL run id | rc | what it is for |
|---|---|---|---|
| 1 | 369367259159 | 1 (gate/envelope mismatch, by design) | measure the new envelopes |
| 2 | 369367259291 (`issue931-post-cv18-20260907T195222Z`) | 0 | the committable record — INGESTED, log ends `RESULT: ALL CHECKS PASSED` |

The two downstream producers below ran as a third job,
`issue931-post-cv18-followups-20260907T230853Z` (rc 0 and rc 0, source
`aa66bed2`, probe log ends `CRITERION (B) LIVE: CONFIRMED`). All four produced
files are committed and byte-identical to that run's `produced/` copies.

Outputs land in `/root/workspace/claude-workspace/rfx/runs/issue931-post-cv18-<ts>/`
(`cv18.log`, `cv18.rc`, `produced/` and `produced_files.txt`). The job runs IN
the worktree, so `--write-fixture` writes
`validation/crossval/_18_wr90_iris_results/rfx.json` and
`tests/fixtures/wr90_iris_modematch/fixture.json` directly onto the branch.
**This phase does not commit fixtures**; the ingest phase does.

## Between pass 1 and pass 2 — what to update, and from where

All of these are pure functions of pass 1's printed envelopes and rows. None
may be hand-tuned; each is `round-UP(envelope x 1.5)` at the quantum the
existing code already uses.

1. `GATE_FINE_ABS` (script line ~161) = `gate_from_envelope(env_fine, quantum=100)`
   where `env_fine` is pass 1's printed `envelopes: fine ...`.
2. `GATE_RICH_ABS` = `gate_from_envelope(env_rich, quantum=100)`.
3. All eight entries of `GATE_FINE_ABS_PER_CONFIG` =
   `gate_from_envelope(row max_gap_abs, quantum=1000)` per configuration, and
   the eight-row comment table above it.
4. The `APERTURE RESOLUTION` paragraph of `claim_scope` quotes the eight
   per-config gates inline — same eight numbers.
5. The claim_scope's `0.04`, `0.0232`, `0.01`, `0.0051`, `0.018-0.043`,
   `0.021-0.054`, `0.0077`, `0.0158`, `1.0207`/`1.0013` and the `0.527-0.604`
   first-order ratios: refresh from pass 1's rows. Prose is byte-pinned to the
   fixture by `test_script_prose_literals_match_fixture`, so the source edit
   and the regeneration must land together.

## The job, for the record

`**/vessl*.yaml` is gitignored repo-wide, so the submitted file
`vessl_931_post_cv18.yaml` sits UNTRACKED beside this note (and a copy is in
the run's output directory). Everything needed to rebuild it:

| field | value |
|---|---|
| name | `rfx-931-post-cv18` |
| cluster / preset | `remilab-c0` / `gpu-rtx4090` |
| image | `ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8` |
| env | `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=4`, `MPLBACKEND=Agg` |
| mount | `/root/workspace/` <- `volume://remilab-fs/personal-workspaces/` |
| working dir | the worktree itself, `/root/workspace/byungkwan-workspace/research/rfx-931-XD-wr90iris` (NOT a staging copy — `--write-fixture` must land on the branch) |
| pip | `jax[cpu]==0.6.2`, `numpy<2`, `scipy>=1.11`, `h5py>=3.8`, `matplotlib>=3.7`, after uninstalling the CUDA plugins |
| command | `timeout 21600 python -u validation/crossval/18_wr90_iris_modematch.py --write-fixture` |
| artifacts | `$RUNS/issue931-post-cv18-<ts>/` with `cv18.log`, `cv18.rc`, `produced/`, `produced_files.txt`, `status_before.txt`, `status_after.txt` |

The job exits 0 regardless of the script's rc so the harvest always runs; the
script's own rc is in `cv18.rc`. Modelled on the baseline
`rfx-931-base-cv18` yaml, which reproduced the pre-change fixture bit-exactly.

## Which fixture keys change, and how

* `schema_version` 1 -> 2.
* every row: `aperture_cells` -> `realized_aperture_cells` (value changes,
  `d_c - 1` -> `d_c`), `thickness_cells` -> `realized_thickness_cells` (same
  value, different meaning), plus new `iris_wall_nodes`, `aperture_wall_nodes`
  and `t_mm`.
* every `s11` / `s21` / `oracle_s11` / `max_gap_abs` / `richardson_dev_abs` /
  `max_colpow` / `wall_s`: re-measured (the iris is one cell thicker at a/30
  and one at a/60 than what was previously realized).
* `gates.*`: all re-derived.
* NEW `one_cell_volume_witness` — the design note §5 one-cell witness, gated
  inside the case on "the t = 1 residual lies inside the range t = 2..8 spans".
* `one_cell_aperture_detection_witness`: recomputed from the new rows (no FDTD).
* PRE-EXISTING DEBT this run closes: the committed record is missing the
  `one_cell_aperture_detection_witness` key that the script's payload emits
  (12 keys, not 13), i.e. it was last updated without a full regeneration.
  The regenerated record will have 14 keys with the new witness.

## Downstream producers to re-run after pass 2 (no FDTD, seconds each)

* `python scripts/diagnostics/build_cv18_aperture_resolution.py` — rebuilds
  `aperture_resolution.json` from the new `rfx.json`. Its `--check` mode diffs
  without writing. No code change is needed in it; it reads the record and
  evaluates the oracle.
* `scripts/diagnostics/probe_cv18_one_cell_aperture_defect.py` — MIGRATED in
  this branch (corners on node planes tracking `run_point`; asserts read the
  realized edge set instead of counting open nodes in a `rasterize()` sigma
  mask; emitted keys renamed `nominal_aperture_nodes` ->
  `nominal_aperture_cells`, `realized_aperture_nodes` ->
  `realized_aperture_cells`, plus `realized_thickness_cells`,
  `iris_wall_nodes`, `aperture_wall_nodes`). Verified with `--geometry-only`:
  the defect still realizes one cell too wide (aperture 11 cells against a
  nominal 10 at a/30).

  It MUST be re-run, and only AFTER cv18 pass 2, because it reads
  `GATE_FINE_ABS_PER_CONFIG` from the script source and the pooled/Richardson
  gates from the regenerated `rfx.json`:

  ```
  JAX_PLATFORMS=cpu python scripts/diagnostics/probe_cv18_one_cell_aperture_defect.py \
      --output validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json
  ```

  Two FDTD runs (a/60 ~800 s, a/30 ~66 s) plus the oracle: ~15 min. Submitted
  on VESSL as a third run rather than on the shared pod. DONE — its output feeds
  `test_live_one_cell_defect_is_caught_by_the_per_config_gate_and_not_the_old_ones`,
  whose six pinned digits moved and were refreshed from the new artifact and the
  rebuilt `aperture_resolution.json` in one commit:

  | pin | pre-#931 | post-#931 | where it comes from |
  |---|---|---|---|
  | `measured.fine_gap_abs` | 0.02842 | 0.01246 | measured, probe run |
  | `measured.richardson_dev_abs` | 0.00588 | 7e-05 | measured, probe run |
  | `config.fine_gate_abs_per_config` | 0.015 | 0.006 | ceil(0.0034 x 1.5) at 1/1000 |
  | `config.pooled_fine_gate_abs` | 0.04 | 0.02 | ceil(0.0106 x 1.5) at 1/100 |
  | `config.richardson_gate_abs` | 0.01 | 0.01 | ceil(0.0046 x 1.5) at 1/100 |
  | model `pairs[2].one_cell_defect.over.fine_gap_abs` | 0.0265 | 0.0134 | rebuilt artifact |

  Both fine gates moved DOWN and the defect is caught with more margin than
  before, 1.895x -> 2.077x.

## Not owned by this group (replacement text is in docs/design_notes/931_migration/)

`validation/crossval/manifest.json` case 18 `claim_scope`.
