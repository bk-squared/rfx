# T6 chain fixture ingest — 2026-09-08

Ingest of the PI-adjudicated artifact, on `feat/931-chain-ingest` based on
`b4ae761e`, following [T6's prescription](T6-waveguide-chain-battery.md).
No measurement producer, operator, gate or historical fixture is edited.

## Source and consumers

- Source: `/root/workspace/claude-workspace/rfx/runs/issue931-chain-repin-20260908T073139Z/fixture.json`.
- Run: **369367259427**, supplied by the adjacent `run_id.txt`.
- Producer commit: `6df7ccaf20d572e0c58c194c12bc63467d4ef360`.
- Predeclaration: `docs/design_notes/waveguide_chain_battery_predeclaration.md`,
  amended at `bcce73c9` before measurement; schema 4.
- New file: `tests/fixtures/waveguide_chain_battery/fixture_931_realized_pec_forward2_run369367259427.json`.
  The name carries the realized-PEC operator, forward2 eps reference, and run id.
- SHA-256: `fcfef5cf89b0a736f187e358c6c73f4613521d6cbcee2bb0d556487400c0bf61`.
  The copy is byte-identical to the supplied artifact.

The JSON itself records `run_id="UNSET-see-log-filename"` and backend `cpu`;
this metadata is preserved. The named VESSL run is source provenance, not proof
that the GPU-marked live test ran. No VESSL launch or fixture regeneration occurs.

The CPU coarse/mid and GPU fine cell tests in
`tests/oracle/test_waveguide_chain_battery_v18_close.py` now use `live_fx`.
Its plane-shift test has no artifact dependency and retains its physical checks.
The guide-cell-aperture module's live-target comments and the fixture README
point to this run. Run-3 `fx` remains historical: `fixture_v18_close.json` was
measured with sigma=1e10 device cell fill and is superseded by realized PEC edges,
without editing its provenance or its replay assertions.

## Stored evidence inspected

All 178 verdicts recompute exactly: **102 pass, 76 report_only, 0 fail,
0 not_interpretable**, over 18 cells and 14 AD/FD legs. The six thru rows differ
from the old fixture by 1.4404e-6 to 2.8267e-6, inside the old 5e-6 envelope.
The minimum PEC-short |S11| over all six rows is 0.99702. The four PEC-short eps
Re/Im legs use forward2 with finite f_2h and relative errors 0.010258–0.018292
against the unchanged 0.05 gate. These are stored-artifact checks, not new solves.

## Open item: unexplained slab coarse/flux drift

The PI did not pre-declare or explain this row. The new-minus-old complex
S-matrix maximum is **1.0422476162860573e-5**, at **S11, 8.4 GHz**. It is
2.08 times the **5e-6 envelope**, though 9.59 times below the **1e-4 gate**.
Its cause remains open. Neither a passing live test nor the changed PEC operator
explains a slab-only row. Do not widen the envelope or gate on its account.

Full per-bin absolute complex differences, read from both committed artifacts:

| f (GHz) | abs ΔS11 | abs ΔS21 | abs ΔS12 | abs ΔS22 |
|---|---|---|---|---|
| 8.4000 | 1.042247616e-05 | 3.625608999e-07 | 1.183118417e-06 | 2.884866347e-06 |
| 8.6000 | 3.182715017e-06 | 6.802506572e-07 | 5.470979139e-07 | 1.944601536e-06 |
| 8.8000 | 5.786564302e-07 | 5.631475890e-07 | 3.583256798e-07 | 2.976094294e-06 |
| 9.0000 | 8.301966356e-07 | 2.980232239e-07 | 2.546311541e-07 | 2.932918000e-06 |
| 9.2000 | 1.887218976e-06 | 8.580791979e-07 | 6.258487701e-07 | 2.945724867e-06 |
| 9.4000 | 1.985436444e-06 | 6.444213857e-07 | 1.194418933e-07 | 2.668431448e-06 |
| 9.6000 | 2.328029258e-06 | 3.039252468e-07 | 2.384185791e-07 | 2.165749347e-06 |
| 9.8000 | 2.727685946e-06 | 1.884864366e-07 | 1.884864366e-07 | 1.849560676e-06 |
| 10.0000 | 2.999837287e-06 | 1.709173582e-07 | 1.796752734e-07 | 1.519626234e-06 |
| 10.2000 | 2.391136775e-06 | 1.884864366e-07 | 1.490116119e-07 | 1.013278961e-06 |
| 10.4000 | 2.225417459e-06 | 2.457562462e-07 | 5.960464478e-08 | 3.769728732e-07 |
| 10.6000 | 1.851585456e-06 | 2.402740046e-07 | 7.066670478e-07 | 8.597595535e-07 |
| 10.8000 | 1.313417122e-06 | 1.843171846e-07 | 3.819467311e-07 | 1.648583017e-06 |
| 11.0000 | 4.752045488e-07 | 2.935188354e-07 | 2.546311541e-07 | 3.013573351e-06 |
| 11.2000 | 3.998401125e-07 | 4.430433734e-07 | 4.214684851e-07 | 3.874481007e-06 |
| 11.4000 | 1.500806607e-06 | 6.548913069e-07 | 6.733565265e-07 | 3.216028249e-06 |
| 11.6000 | 3.760279683e-06 | 1.609325409e-06 | 1.700822481e-06 | 3.059205127e-06 |

## Local validation

Executed in this worktree on `feat/931-chain-ingest` with the local import
confirmed at `rfx/__init__.py`, `PYTHONPATH=$PWD`, and `JAX_PLATFORMS=cpu`:

```sh
PYTHONPATH=$PWD JAX_PLATFORMS=cpu timeout 1200 python -m pytest \
  tests/oracle/test_waveguide_chain_battery_v18_close.py \
  -m 'not gpu' -k 'test_live or test_the_live_pin' -n 4 -rA \
  --junitxml=.pytest_cache/chain_ingest/live.xml
```

**6 passed, 0 failed, 0 skipped in 69.96 s**: four live instances (CPU cells
coarse/mid; coarse plane shift false/flux) plus the schema-4 replay/identity guard
and unchanged live-pin guard. The single GPU-marked fine live instance was
excluded with `-m 'not gpu'`: this CPU-only task cannot validate that lane.
The 1200-second timeout is enforced by the shell. An initial command including
`--timeout=1200` exited before collection because pytest-timeout is not installed;
no tests or solves ran in that invocation. There was only one pytest at a time.

Captured comparison and physical-witness output (full local log:
`.pytest_cache/chain_ingest/live.log`):

```text
[live plane-shift coarse false] |S| max diff=1.56e-07 resid_yee=0.512° resid_cont=0.669° wrong_sign_min=64.1°
[live plane-shift coarse flux] |S| max diff=7.11e-08 resid_yee=0.512° resid_cont=0.669° wrong_sign_min=64.1°
[live thru-coarse-false] max|S_live-S_fixture|=5.558e-07 settling=[-84.89036631 -85.32600159] colpow=1.00613 recip_c=1.65e-03
[live thru-coarse-flux] max|S_live-S_fixture|=4.805e-07 settling=[-84.89036631 -85.32600159] colpow=1.00002 recip_c=1.44e-03
[live pec_short-coarse-false] max|S_live-S_fixture|=1.099e-06 settling=[-85.55640639 -84.30821044] colpow=1.00472 recip_c=0.00e+00
[live pec_short-coarse-flux] max|S_live-S_fixture|=5.331e-07 settling=[-84.89036631 -84.30821044] colpow=1.00006 recip_c=0.00e+00
[live slab-coarse-false] max|S_live-S_fixture|=5.373e-07 settling=[-81.76901013 -79.58505539] colpow=1.00711 recip_c=3.09e-02
[live slab-coarse-flux] max|S_live-S_fixture|=3.734e-07 settling=[-81.76901013 -79.58505539] colpow=1.00010 recip_c=1.44e-03
[live thru-mid-false] max|S_live-S_fixture|=5.397e-07 settling=[-97.69759758 -98.29904464] colpow=1.00116 recip_c=2.12e-04
[live thru-mid-flux] max|S_live-S_fixture|=5.960e-07 settling=[-97.69759758 -98.29904464] colpow=1.00000 recip_c=2.11e-04
[live pec_short-mid-false] max|S_live-S_fixture|=1.120e-06 settling=[-96.79915441 -98.58511057] colpow=1.00116 recip_c=0.00e+00
[live pec_short-mid-flux] max|S_live-S_fixture|=5.093e-07 settling=[-96.79915441 -98.29904464] colpow=1.00002 recip_c=0.00e+00
[live slab-mid-false] max|S_live-S_fixture|=8.607e-07 settling=[-94.36686927 -93.97238433] colpow=1.00154 recip_c=1.09e-02
[live slab-mid-flux] max|S_live-S_fixture|=9.197e-07 settling=[-94.36686927 -93.97238433] colpow=1.00004 recip_c=2.11e-04
```

The maximum live difference is 1.120e-6, below the unchanged 1e-4 tolerance;
the producer-versus-live-lane falsifier did not fire on either CPU rung.
The old-fixture slab discrepancy above remains open.

Historical replay check, after the live invocation completed:

```sh
PYTHONPATH=$PWD JAX_PLATFORMS=cpu timeout 1200 python -m pytest \
  tests/oracle/test_waveguide_chain_battery_v18_close.py \
  tests/oracle/test_waveguide_chain_battery_guide_cell_aperture.py \
  -n 4 -rA --junitxml=.pytest_cache/chain_ingest/replay.xml
```

**449 passed, 0 failed, 0 skipped in 4.35 s**, using the default marker filter
that excludes slow/GPU tests. This includes the two fast live-reference guards
also counted in the first invocation. Historical schema-2/3 replay assertions
remain intact. No fixture, live envelope, gate or xfail was changed. The old
artifact is byte-identical to `b4ae761e`, and the new artifact matches its source.

R3: memory=none-after-grep (no docs/agent-memory entries in this worktree; consistent with T6 and the amended predeclaration) | R2-attempts=0 | falsifier=source-copy and old-fixture byte checks, unchanged 1e-4 live gate, and stored-verdict replay all passed; GPU live falsifier not run
