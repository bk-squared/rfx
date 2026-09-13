# pending recompute — cv04, #888 auxiliary absorber

**Not yet run.** The committed records in this directory were produced with the
20-cell TF/SF auxiliary absorber. #888 replaced it with a 200-cell absorber
whose sigma is derived from a reflection target
(`rfx/sources/tfsf.py::AUX_N_CPML_1D`, note
`docs/design_notes/20260904_aux_absorber_depth_derivation.md`), so the injected
incident field is no longer the one these numbers were measured under.

What is affected, and what is not:

* `lattice_witness.json::rungs.*.aux_echo` declares the auxiliary layout it was
  written under (`aux_n_cpml = 20`). It is replayed against THAT layout by
  `tests/crossval/test_aux_echo_record_invariant.py`, which also asserts the
  record stays admissible under the shipped absorber's stricter (19-20 steps
  earlier) arrival. Ratios move 0.610 -> 0.617 at cv04's rung, limit 1.0.
* Every measured R/T number in `lattice_witness.json` and `envelope.json` is a
  measurement of the superseded rig. On the 600-cell/719-step rig the branch
  that derived the absorber measured `max|R+T-1|` 0.0487 -> 0.0043 and
  `mean|dR - lattice|` (gated) 1.68e-03 -> 1.98e-04; those are the branch's
  numbers, on the branch's bandwidth, and are NOT what this directory carries.
  Main's committed record is the 990-step settled one from #974 (VESSL run
  369367260232), still on the 20-cell absorber.

Closing this: re-run cv04 (`--lattice-witness` included) on the derived
absorber, then take `"cv04"` out of `_ABSORBER_RECOMPUTE_PENDING` in
`tests/crossval/test_aux_echo_record_invariant.py` — that list is gated in both
directions, so the waiver cannot outlive the re-run.

Envelope revisions are append-only and adoption is an explicit edit in the
consumer (#928): a re-run appends a revision, it does not move any window.
