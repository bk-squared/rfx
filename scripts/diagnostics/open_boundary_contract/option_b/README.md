# Option B of the open-boundary contract: can the absorber end a line? (measured, not adopted)

The contract (`docs/design_notes/20260921_open_boundary_contract.md`, #1167) kept lines ended by
their ports (C5) and left as "option B" the question whether the CPML's complex-frequency-shift term
`alpha = 0.05·(1−rho)` S/m, a literal with no measured origin, was what kept the absorber from ending a
two-conductor line, so that deriving it from the lowest frequency of interest would make the port
exemption unnecessary. This directory is the sweep that answered it, run 2026-09-22 on main at
`e7f7e027` (`PROVENANCE.txt`), float32, remilab-c0 / rtx4090. Design: `DESIGN.md`; brief and its two
addenda: `BRIEF_sweep_driver.md`; reports: `REPORT.md` (first launch, no solves: git ownership in the
image), `REPORT_2.md` (the 112-arm sweep), `REPORT_3.md` (rig 4 re-run on the committed oracle's
method, the source-centre sweep, and the end-of-record field diagnostics).

## What was measured

alpha scale s ∈ {0, 0.01, 0.03, 0.1, 0.3, 1, 3} on the shipped profile × absorber depth N ∈ {4, 8, 16}
layers, on four structures, one number each: the MSL phase-referee line with its strip continued through
both x absorbers and `terminates=()` on both ports, one-port |S11| of the absorber as a termination
(3–4.5 GHz, and 0.5–4.5 GHz); a WR-90 guide ended by the absorber, |S11| over 7–12 GHz; the patch
ring-down rig (six layers added), settling; and a point-source reflectivity rig. Tables: `TABLE_*.md`;
per-arm records with realized grid, pads, preflight text and energy witnesses: `results/`, `results_3/`,
`WITNESSES_2.csv`; run ids: `run_ids*.json`, `jobs_2/`, `jobs_3/`.

## The leader's reading (from the tables and the per-arm records)

1. **alpha is not what limits the absorber as a line termination.** From s = 0 (plain CPML) to s = 1
   (shipped) every structure moves by less than 0.3 dB. At s = 3 the low-band line collapses (4 layers,
   0.5–4.5 GHz: −8.1 → −1.6 dB), the term's transparency corner alpha/(2π ε0) having risen to 2.7 GHz.
   On the point-source rig, with the reference sized as `tests/unit/boundaries/test_cpml.py` prescribes,
   alpha HELPS: 8 layers, s = 0 / 1 / 3, −64.5 / −68.6 / −71.3 dB at 1 GHz, −66.2 / −68.3 / −70.4 dB at
   2 GHz, −61.4 / −62.5 / −63.7 dB at 4 GHz. The shipped value sits where these numbers say it should;
   there is no case for deriving it from a frequency in this range. (The 0.5 GHz source-centre arm did
   not run: its clean reference needs a 923³ grid, past the GPU's memory; `reference_sizing_3.json`.)
2. **Depth sets how well the absorber ends a line, and it stays far from a port's termination.**
   Absorber-terminated MSL, 3–4.5 GHz, max |S11|: −9.4 / −15.0 / −22.3 dB at 4 / 8 / 16 layers; WR-90:
   −13.9 / −22.3 dB at 8 / 16 (the 4-layer WR-90 arm is an extractor failure, +24 dB, column power 267,
   `passivity_warning.txt`, and is not a reading). Against the port's resistive termination (ring-down
   −98 dB on the same line, #1167 §5) the absorber is not a substitute. C5 stands; `terminates=` (#1178)
   is a necessity, not a statement.
3. **A strip continued to the outer wall leaves a magnetostatic residual.** At the end of the record
   the interior energy sits at −24 dB of the post-source peak while the field probes read −81 dB: 97 %
   of it is magnetic (Hx and Hy), spread over the whole line (`TABLE_fields_3.md`,
   `FIELD_SPATIAL_DETAIL_3.json`). A CPML removes propagating fields and does not touch a static one.
   Which current path carries it is not determined here, and whether a port-ended line carries the same
   residual was not measured. The band-limited S-parameters are not affected by it.
4. **The patch does not care:** −43.6 … −46.9 dB on every arm (`TABLE_patch.md`); no alpha undoes #1178.
5. **Side finding, closed in the same PR:** `tests/oracle/test_pml_reflectivity.py`'s 200 mm reference
   puts its own wall echo inside the window (nearest round trip 0.63 ns in a 2.28 ns record), so it read
   −47.3 dB for all nine (N, s) arms alike (`REPORT_3.md` item 1) — the number `test_cpml.py`'s
   docstring already names as the contaminated floor. It measured the reference, not the absorber, and
   is removed; `test_cpml.py` with the clean reference is the CPML reflectivity gate.

Excluded from this directory: the first-launch variant of rig 4 (uniform source, periodic transverse
boundaries), whose numbers were 20–48 dB off the committed method for a reason not found
(`REPORT_3.md` item 3) and are not cited; raw field dumps and job logs (on the lab share under
`.801-measure/B/`).
