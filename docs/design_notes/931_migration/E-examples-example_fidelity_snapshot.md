# E — handoff: `tests/data/example_fidelity_snapshot.json` (+ its contract test)

Owner of the file: the ingest phase. This group (E — examples) did NOT touch it.
Everything below is the replacement text / the actions to apply there.

## Sequencing — why this file is last

The snapshot's 50 variants span `examples/`, `validation/crossval/` and
`validation/research/`. Recapturing it before those groups land freezes the old
physics into a fresh file that LOOKS reviewed. Recapture only when:

1. the preflight group's findings are in (the snapshot pins preflight text
   verbatim — 15 rows quote `two_plane=True on this entry (walls on both
   faces)` and the one-cell-sheet rule, and two `sheet_cavity_electrical_
   thickness` findings carry numbers (+84.5 %, +55.0 %) that are direct
   consequences of the live normal-E edge inside a one-cell sheet);
2. every crossval and validation script in the corpus is migrated. Three
   producers had no inventory row at all — `validation/research/issue764_
   wireport_norm_falsifiers.py`, `issue770_offdiag_adjudication.py`,
   `thru_feedpost_deembed.py` — plus `convergence_floor/fixture.py` and
   `multiband_nu/w4r_port_supraconvergence.py`. The snapshot cannot be
   recaptured consistently until they are.

Command (unchanged, it is in the file's own `_comment`):

```
JAX_ENABLE_X64=0 python scripts/capture_example_fidelity_snapshot.py
```

Review the diff ROW BY ROW. The diff IS the audit of what the contract changed.

## `_comment` header — append

> Re-captured 2026-09-XX for issue #931 (the lattice ownership contract). A
> conductor is now declared as exactly one of three things and the declaration
> decides the realization: a SHEET (`add_thin_conductor`, or a zero-thickness
> Box through `add()`) is a footprint on ONE node plane, owns no cell and is
> absent from `pec_mask`; a VOLUME (`add(shape, material="pec")`) is the primal
> cells whose CENTRES lie inside the shape, with every incident E edge zeroed,
> so a drawn body realizes walls on BOTH of its faces. The `two_plane` kwarg,
> the `one-plane sheet` / `two-plane sheet` realization labels and the #702
> sheet-own-cell resample are gone. Rows whose `kind` prefix moved from
> `geometry|` to `thin_conductor|` are foils migrated from `add()` to
> `add_thin_conductor`; that key move is intended and reviewable, not a key
> regression (see the comment in `tests/_example_fidelity_lib.py::_entity_key`).

## Rows this group's migration will move

From `examples/`:

* `patch_antenna_demo` ground and patch — were `geometry|pec|...` with
  `one-plane sheet (normal z)` and a `sheet-own-cell-live` finding; become
  `thin_conductor|...` sheet rows. The fine z band lost its two reserved metal
  cells, so the domain and substrate rows move too.
* `ports_and_sparams_101` ground and trace — were
  `volumetric PEC (>= 2 cells on every axis)`; become sheet rows on the
  substrate faces.
* `rcs_scattering` sphere (line ~2101) — stays a volume; its realization label
  changes wording only if `fidelity.py` restates it, and its cell count moves
  from node sampling to centre sampling (measured: 1045 -> 1067 cells).
* `slab_rt_flux_monitor` slab — unchanged. Verified bit-identical after the
  `-dx/2` nudge was deleted (290 cells both ways); if this row moves, the
  dielectric sampling was touched and #931 said it must not be.
* `nonuniform_patch_demo` and `cad_mesh_import_demo` have NO snapshot variant
  (classified `module_level_solve` / `builder_fused_with_solve`). See below.

## Two additions asked for by the examples-B inventory

`tests/contracts/test_example_fidelity_contract.py` — add, once the snapshot is
regenerated (adding them BEFORE would fail against the stale file, which is why
this group did not):

1. at least one audited variant per realization class, asserted by class name
   (today no variant declares a conductor as an explicit SHEET, and after
   migration nothing guarantees both classes stay covered);
2. a stale-text guard: no snapshot row may contain `two_plane`, `one-plane
   sheet` or `two-plane sheet`;
3. every conductor entity's `realization` is one of the contract's classes —
   no third string.

## A refactor worth doing while the file is open

`examples/tutorials/nonuniform_patch_demo.py` and `cad_mesh_import_demo.py`
have PEC geometry that no committed artifact witnesses, because both fuse the
builder with the solve. Expose a build-only function (the audited pattern) and
add snapshot variants, so an ownership change leaves a diff instead of nothing.
This group did not do it: it changes the tutorials' top-level structure, which
is a bigger edit than the migration and belongs with whoever recaptures.
