# pending recompute — cv04, #888 auxiliary absorber

**Run: 2026-09-16. Adopted: no.** Revision `r2` of `envelope.json` is the same
measurement on the shipped absorber, appended beside r1. Nothing derives a
window from it, `calibration_revision` still says r1, and cv04 is still on
`_ABSORBER_RECOMPUTE_PENDING`. What closes those is an adoption change in the
consumers, which is a separate reviewed PR by construction — the same-commit
guard in `tests/contracts/test_calibration_envelope_ownership.py` refuses a diff
that touches this artifact and an adoption record together.

The committed records at the top of this directory were produced with the
20-cell TF/SF auxiliary absorber. #888 replaced it with a 200-cell absorber
whose sigma is derived from a reflection target
(`rfx/sources/tfsf.py::AUX_N_CPML_1D`, note
`docs/design_notes/20260904_aux_absorber_depth_derivation.md`), so the injected
incident field is no longer the one those numbers were measured under.

## What r2 is

`validation/crossval/_04_fresnel_results/r2/` — this case run with
`--lattice-witness` on the shipped absorber (`aux_n_cpml` 200, reflector depth
0 cells), settling at `nx_interior` 800 / 990 steps after one grow, local CPU,
43 s, exit 2 (rfx self-check PASS, Meep unavailable — the same exit code r1's
committed run returned). Logs at `../_04_fresnel_logs/r2/`.

| | r1 (`envelope.json`) | r2 |
|---|---|---|
| `mean_dR` | 0.0066 | 0.007255 |
| `mean_dT` | 0.011 | 0.007266 |
| `mean_closure` | 0.0091 | 4.59e-05 |
| `per_bin_max_RT_closure` | 0.0487 | 0.0003695 |
| record | 719 steps, nx 600, aux 20 cells | 990 steps, nx 800, aux 200 cells |
| evaluated band | 3.0321–11.8666 GHz, 170 bins | 3.0321–11.8146 GHz, 169 bins |

`r2/sibling_run_length/` is the same rig on a larger declared box (1200 cells,
1533 steps). It is what makes r2 `run-length-witnessed` where r1 is
`carried-unwitnessed`, and it splits the four values in two: the band-mean
errors do not move with record length (0.02 % in R, 0.18 % in T) while both
closure values fall by about 2.5x. The closure numbers are a record-length
reading, not a converged envelope.

## What is affected, and what is not

* `lattice_witness.json` (top level) and `fringe_gate_geometry.json` are still
  r1's, unchanged, and still what every test here reads. The r2 witness sits at
  `r2/lattice_witness.json` precisely so that stays true:
  `tests/crossval/test_aux_echo_record_invariant.py` and
  `test_lattice_witness_gates.py` open `_04_fresnel_results/lattice_witness.json`
  by that exact name.
* `lattice_witness.json::rungs.*.aux_echo` declares the auxiliary layout it was
  written under (`aux_n_cpml = 20`). It is replayed against THAT layout by
  `tests/crossval/test_aux_echo_record_invariant.py`, which also asserts the
  record stays admissible under the shipped absorber's stricter (19-20 steps
  earlier) arrival. Ratios move 0.610 -> 0.617 at cv04's rung, limit 1.0 — and
  the r2 witness measures exactly 0.617, so the bound was right.
* `fringe_gate_geometry.json` is derived from the committed
  `lattice_witness.json`'s `dt_s` and band. r2's band is one bin narrower at the
  top (the mask thresholds the incident spectrum, and the absorber that supplies
  it changed), so this artifact must be regenerated with
  `comparators/emit_cv04_fringe_gate_evidence.py` when r2 becomes the committed
  record. Not now: regenerating it against an unadopted revision would leave the
  gate geometry and the record it cites describing different runs.

## What closing this looks like

An adoption change, in the consumers:

1. `CV04_ADOPTION` in `comparators/cv22_dispersive_gates.py` and
   `cv23_lossy_gates.py` names `r2` and its hash.
2. `r2/lattice_witness.json` becomes `lattice_witness.json`,
   `fringe_gate_geometry.json` is regenerated, and `"cv04"` comes off
   `_ABSORBER_RECOMPUTE_PENDING` in
   `tests/crossval/test_aux_echo_record_invariant.py` — that list is gated in
   both directions, so the waiver cannot outlive the re-run and cannot be lifted
   before the record catches up either.

**That change cannot be made from these numbers alone**, and this is the reason
it is not stacked here. `W_BIN` is
`gate_from_envelope(per_bin_max_RT_closure)` in both consumers — live gate
constants. From r1 that is 0.074; from r2 it is 0.001. Re-judging every
committed cv22/cv23 record through the consumers' own evaluators under both
windows (the r1 pass re-reproduces all 29 committed verdicts exactly) flips
**19 records**, including all six declared arms and every dx-ladder rung but
one: the widest per-bin residual each arm has to fit, `max` over the gated bins
of `|dR| - w_ade`, runs 0.00129 (cv22 drude, R) to 0.01439 (cv23 tand0p1, T),
all of them above 0.001. The two band-mean windows do not even move the same
way — `W_MEAN_R` 0.010 → 0.011, `W_MEAN_T` 0.017 → 0.011.

The recipe is the defect, not the measurement: `|R+T-1|` is an energy-leak
quantity and the residual it has to cover is lattice dispersion, which conserves
energy — and the sibling above measures that directly, by showing the closure
values keep falling with record length while the errors they are meant to bound
do not move. Re-deriving `W_BIN` is issue #928's item 2 and a PI decision.

Envelope revisions are append-only and adoption is an explicit edit in the
consumer (#928): a re-run appends a revision, it does not move any window.

**What this directory's envelope feeds, and why appending r2 is not a gate
change.** `per_bin_max_RT_closure` here is what `W_BIN` is derived from in
`validation/crossval/comparators/cv22_dispersive_gates.py` and
`cv23_lossy_gates.py`. Both derive it from the revision their own
`CV04_ADOPTION` names, which is r1, so appending r2 moves nothing —
`tests/crossval/test_cv22_dispersive_slab_gates.py::test_unadopted_new_evidence_leaves_the_windows_alone`
is that property as a test. The direction of the error in the meantime is
unchanged from before this run: `W_BIN` is wider than the current rig warrants,
never tighter, so no gate widens by leaving it alone. The magnitude is no longer
unestablished — it is 0.0487 against 0.0003695, a factor of 132 — and the
absorber is not all of it. `validation/crossval/_04_fresnel_logs/cv04_settled_summary.json::max_RT_minus_1_per_bin`
= 0.001 is the #974 settled record, 990 steps on the OLD 20-cell absorber, which
splits the fall: **record length carries 0.0487 → 0.001, about 49×, and the
absorber carries 0.001 → 0.0003695, about 2.7×** (48.7 × 2.71 = 132, against
0.0487 / 0.0003695 = 131.8; the 49× is one digit because the committed summary
rounds that value to one).

This is also the reconciliation for the 0.0043 the recompute was predicted to
land on (main's version of this file, and
`docs/agent-memory/rfx-known-issues.md`'s #928 entry). That reading is the #888
branch's, on the 600-cell / 719-step rig with the TF/SF bandwidth also moved
0.5 → 0.8 — it isolates neither variable and is not the same quantity as either
column above.
