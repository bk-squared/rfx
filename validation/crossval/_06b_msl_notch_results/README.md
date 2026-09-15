# cv06b result legs — `scripts/diagnostics/cv06b_build_falsifiers.py`

What each committed JSON in this directory is, and which is CURRENT versus
HISTORICAL. A historical leg is kept because a committed document still
reasons about it; it is never the leg the diagnostic script re-derives from.

| file | status | board | headline numbers | why |
|---|---|---|---|---|
| `cv06b_build_falsifiers_summary.json` | **CURRENT** | post-#931 sheet board (ground/trace declared as sheets) | `criterion_A_baseline.err_pct` 2.1649 %; `stub_1cell.visible` **true**, `verdict.all_ok` **true**; `stub_narrow.err_pct` 6.4388 % (G1 now fires too, alongside G2) | the artifact `docs/design_notes/estimator_resolution_regate.md` section 8 reasons about. Produced by `fae08d10` (2026-09-07, VESSL run 369367259191, #931 X-B group). |
| `cv06b_build_falsifiers_summary_pre931_fae08d10.json` | HISTORICAL | pre-#931 board (ground/trace as one-cell PEC `Box`es) | `criterion_A_baseline.err_pct` 1.453 %; `stub_1cell.visible` **false**, `verdict.all_ok` **false**; `stub_narrow.err_pct` 0.208 % (only G2 fires) | byte-identical to the file committed at `fae08d10^`, the parent of the commit that rebuilt this fixture for the lattice-ownership contract. Kept because `docs/design_notes/estimator_resolution_regate.md` section 7.6 reasons about this exact board and its own numbers are frozen to it; without this file section 7.6's citations would have to stop being machine-checked. |

## Reading the two together

Same board rebuild as cv05 and cv15 in this same #931 merge: the geometry
realization changed (one-cell PEC boxes -> declared sheets), not the
falsifier script or its gates. Section 8 of the design note re-anchors
section 7.6's citations to the current file and states which verdicts moved
and which held. `verdict.all_ok` is **not** "all gates in this file passed"
— see `scripts/diagnostics/cv06b_build_falsifiers.py`'s `good = ok and
visible and (not g2) and dep`: it never reads `stub_narrow`'s own G1 gate,
which is why `stub_narrow.gates["G1 notch freq vs analytic"]` can flip to
`false` in the CURRENT file while `verdict.all_ok` flips to `true` in the
same file — they are reading different keys, not disagreeing about one.
