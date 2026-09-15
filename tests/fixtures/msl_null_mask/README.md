# `msl_null_mask` — realization provenance (#931)

`vi_dump_dx197um.npz` and `vi_dump_dx98um.npz` are V/I dumps from the MSL
null-mask diagnostic, captured on **Board S before the lattice ownership
contract**: the trace and ground were one-cell PEC Boxes realized as ONE wall
each, and the board reserved a vacuum cell for each foil, which rfx's #702
own-cell re-sample silently filled with the laminate.

Under the contract that board is drawn differently — each foil is a
zero-thickness sheet ON the laminate face it bounds, and the cavity is four
cells of laminate with nothing else in it (see
`tests/locks/test_patch_edgefed_s11_passivity.py` and the redraw's measurement
in `docs/design_notes/931_migration/T6-RECOMPUTE.md`). The dumps are therefore
a **frozen pre-contract record**. They are not regenerated here: this phase
does no numeric regeneration, and a dump regenerated without renaming would
lose the provenance that makes it readable.

Two consequences for anyone reading them:

* They are the INPUT to the null-mask diagnostic, so running that diagnostic
  against a post-contract build compares two different boards. Regenerate the
  dumps in the same pass that re-pins Board S's band (VESSL 369367259226), or
  state which realization each side came from.
* Their electrical stack is one cell thicker than the board now declares:
  preflight's #703 cavity check reads `sum(d/eps)` 627.1 µm mesh vs 429.8 µm
  physical (+45.9 %) on the geometry these dumps were taken from.
