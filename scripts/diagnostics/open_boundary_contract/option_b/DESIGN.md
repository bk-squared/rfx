# Option B — can the absorber end a line? Sweep design (leader, 2026-09-22)

## The question in physics terms
rfx's CPML has one term nobody chose from a measurement: `alpha = 0.05·(1−rho)` S/m
(`rfx/boundaries/cpml.py:158`), the complex-frequency-shift term. It makes the absorber transparent
below f_alpha = alpha/(2π ε0) ≈ 0.9 GHz and, measured on 2026-09-21, sets how badly a two-conductor
line that runs INTO the absorber is terminated: the MSL phase-referee line with its strip continued,
ring-down −82.7 / −74.7 / −63.4 / −40.6 dB and reciprocity error 6.4e-4 / 4.7e-3 / 5.1e-2 / 0.46 at
alpha × 0.01 / 0.25 / 1 / 4 (records `.801-measure/alpha/`). At alpha × 0.01 the absorber ends the
line about as well as the port's resistive termination does (−98 dB, 6.5e-5). The patch (a radiating
structure) does not care: growth rates 2.77e-3 … 1.96e-3 over the same factors.

So the question is not "what alpha" but: **for a given lowest frequency of interest f_min and absorber
depth, is there an alpha profile under which the absorber terminates a TEM line, a guided mode near
cutoff and a radiating field all within the v2 bar, so that C5's exemption (`terminates=`) becomes a
statement rather than a necessity?** If yes, alpha is derived from `freq_min` (a parameter rfx does
not have yet; `freq_max` exists) and the port exemption stays only as a declaration.

## What is already known (records, not to be re-measured)
- kappa_max > 1 degrades guided-mode absorption (battery header); kappa = 1 stays fixed.
- sigma follows the Meep formula with R = 1e-15, order 3; not varied here (one variable at a time).
- The absorber's lo/hi asymmetry (PR #1012, unmerged) moves where energy sits on the patch rig. The
  sweep runs on main WITHOUT #1012 first; the TEM and cutoff rigs are one-dimensional along the
  absorber normal and are not expected to care. Recorded as an assumption to check on one arm.
- Layer count alone did not fix the patch; ground continuation did (#1178). The patch enters the
  sweep as the "radiating" structure only to confirm alpha does not undo #1178.

## Axes
- alpha profile SCALE: factor s ∈ {0, 0.01, 0.03, 0.1, 0.3, 1, 3} on the shipped 0.05 (s = 0 is
  plain CPML, no CFS). Shape (linear in 1−rho) fixed in this sweep; shape is a later axis.
- Layers N ∈ {4, 8, 16} (the shipped range and one beyond).
- Structure and the number each yields:
  1. TEM line: the MSL phase-referee line (`validation/crossval/20_msl_phase_referee.py`), strip
     continued through both x absorbers with `terminates=()` on both MSL ports (the ports stay for
     excitation and probing), 3–4.5 GHz. Numbers: ring-down settling (dB), max |S12−S21|
     (reciprocity), and the absorber's own return as a termination: |S11| in dB across the band from
     a one-port drive with the far port removed. This is the number a user would receive.
  2. Guided with cutoff: the WR-90 waveguide port battery rig with a matched absorber end
     (`tests/oracle/test_waveguide_port_validation_battery.py`'s builder), 7–12 GHz, cutoff 6.56 GHz:
     |S11| of the absorber termination across the band, worst near cutoff where the guide wavelength
     is longest.
  3. Radiating: the patch ring-down rig, six layers, 150 periods: settling and worst rate (the #1178
     oracle predicate), to see that no alpha re-opens the growth.
  4. Plane wave, the cleanest absorber number: the CPML reflectivity oracle rig
     (`tests/oracle/test_pml_reflectivity.py`, clean-reference method), normal incidence, reflection
     vs frequency from 0.2 to 6 GHz, to read f_alpha directly and its dependence on s and N.
- Low-frequency arm: rig 1 excited with a pulse whose band reaches down to 0.5 GHz, to see where the
  termination collapses as f approaches f_alpha(s).

7 × 3 × 4 = 84 arms plus the low-frequency arm; the patch is 26 s on rtx4090, the lines are minutes;
one VESSL job per structure. float32, provenance printed by each job.

## What the result must contain before a rule is written
- For each structure, a table |S11| or settling vs (s, N) with the record witnesses (end-of-run energy
  below −40 dB of the post-source peak, or marked truncation-suspect).
- The f_alpha reading from rig 4 against the formula alpha_max/(2π ε0), so that the rule can be
  written as alpha_max = c · 2π ε0 · f_min with a MEASURED c, not a quoted one.
- The smallest N at which rig 1's absorber termination is within the v2 bar (2 dB on |S21|, which
  bounds the termination's |S11| — to be derived from the bar, not chosen) at s → the rule.
- A statement per structure whether alpha = 0 (plain CPML) is simply best; if it is, option B is "drop
  CFS" and the sweep is the evidence.

## Implementation (Codex, after the PI has seen this design)
A driver that monkeypatches `_cpml_profile`'s alpha scale (as `.801-measure/alpha/` did), the four
rigs, one YAML per structure, results as JSON with the same layout as `.801-measure/alpha/TABLE.md`.
No product code changes in this step; `freq_min` and the derived alpha come after the table.

Conclusion: leader fills after the sweep.
