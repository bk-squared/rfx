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
   (shipped) the headline numbers move by at most 0.55 dB: MSL 3–4.5 GHz 0.19 / 0.09 / 0.09 dB at
   4 / 8 / 16 layers, WR-90 0.04 / 0.07 dB at 8 / 16, patch 0.55 / 0.03 / 0.04 dB, and the 0.5–4.5 GHz
   line 0.55 / 0.36 / 0.27 dB (`TABLE_*.md`). The one place alpha does what option B predicted is
   that low band, which reaches below the shipped transparency corner alpha/(2π ε0) = 0.90 GHz: at
   4 layers alpha × 1 is 0.55 dB WORSE than plain CPML and the best arm is s = 0.3 (corner 0.27 GHz,
   just under the band's f_min), −8.40 dB against the shipped −7.51 dB, 0.89 dB better; at 8 and 16
   layers the shipped value is the better one by 0.35 / 0.24 dB (`TABLE_msl_low.md`). At s = 3 the low
   band collapses (4 layers −8.1 → −1.6 dB; corner 2.7 GHz). On the point-source rig, with the reference
   sized as `tests/unit/boundaries/test_cpml.py` prescribes, alpha helps: 8 layers, s = 0 / 1 / 3,
   −64.5 / −68.6 / −71.3 dB at 1 GHz, −66.2 / −68.3 / −70.4 dB at 2 GHz, −61.4 / −62.5 / −63.7 dB at
   4 GHz (s = 0 → 1: 4.1 / 2.0 / 1.0 dB; s = 0 → 3: 6.8 / 4.1 / 2.3 dB); every centre frequency measured
   on that rig lies at or above the shipped corner, so it does not test the low-band hypothesis, and the
   arm that would (0.5 GHz) did not run: its clean reference needs a 923³ grid, past the GPU's memory
   (`reference_sizing_3.json`). Reading: a 0.89 dB gain at four layers, lost at eight, against the
   13 dB that depth buys between 4 and 16 layers; the shipped value is kept and no rule in f_min is
   written. The coefficient c of DESIGN.md's rule was therefore not fitted.
2. **Depth sets how well the absorber ends a line, and it stays far from a port's termination.**
   Absorber-terminated MSL, 3–4.5 GHz, max |S11| at s = 0 (s = 1 within 0.2 dB): −9.4 / −15.0 / −22.3 dB
   at 4 / 8 / 16 layers, a clean one-port extraction (REPORT_2 B2). WR-90: −13.9 / −22.3 dB at 8 / 16,
   with a caveat: every WR-90 record ends while the source is still running (no post-source window,
   `WITNESSES_2.csv`), the extraction is `extract_waveguide_s11` with no clean-guide subtraction, and
   the 4-layer arm of the same extractor returned +24 dB with column power 267
   (`raw_3/diagnostic_waveguide/passivity_warning.txt`); the 8- and 16-layer arms pass the passivity guard (max |S| 0.20 / 0.08),
   which is the only check they have. The MSL numbers carry the conclusion; the WR-90 pair is
   corroboration with that caveat. Against the port's resistive termination (ring-down −98 dB on the
   same line, #1167 §5) the absorber is not a substitute. The criterion DESIGN.md asked for — the |S11|
   bar derived from the v2 2 dB |S21| bar, and the smallest depth meeting it — was NOT derived here; the
   verdict is taken against the port's number, and a derived bar is the PI's to set. C5 stands;
   `terminates=` (#1178) is a necessity, not a statement.
3. **A strip continued to the outer wall leaves a magnetostatic residual.** At the end of the record
   the interior energy sits at −24 dB of the post-source peak while the field probes read −81 dB: 97 %
   of it is magnetic (Hx and Hy) on the two-port drives, 92 % on the one-port drive, spread over the whole line (`TABLE_fields_3.md`,
   `FIELD_SPATIAL_DETAIL_3.json`). A CPML removes propagating fields and does not touch a static one.
   Which current path carries it is not determined here, and whether a port-ended line carries the same
   residual was not measured. The band-limited S-parameters are not affected by it.
4. **The patch does not care:** −43.6 … −46.9 dB on every arm (`TABLE_patch.md`); no alpha undoes #1178.
5. **Side finding, closed in the same PR:** `tests/oracle/test_pml_reflectivity.py`'s 200 mm reference
   puts its own wall echo inside the window (nearest round trip 0.63 ns in a 2.28 ns record), so it read
   −47.30 … −47.56 dB for all nine (N, s) arms (`REPORT_3.md` item 1) — the number `test_cpml.py`'s
   docstring already names as the contaminated floor. It measured the reference, not the absorber, and
   is removed; `test_cpml.py` with the clean reference is the CPML reflectivity gate.

On REPORT_2.md's rig-4 table: it prints the first-launch VARIANT of rig 4 (uniform 5×5 source, periodic
transverse boundaries, two-ended-box residual), which read 21–59 dB off the committed oracle's method
arm by arm for a reason not found (`REPORT_3.md` item 3); those numbers are superseded by REPORT_3
item 1 and are kept only under `failed_instrument/` as the record of a rejected instrument. Not in this
directory: raw field dumps (`raw*/…/*.npz`, the `.png` maps) and the VESSL provider logs
(`jobs*/…/provider.log`), on the lab share under `bk-workspace/.801-measure/B/`; every run id is in
`run_ids*.json` and `jobs*/…/run_id.txt`.
