# T6 handover — text for files T6 does not own

T6 owns `tests/oracle`, `tests/contracts`, `tests/locks`, `tests/studio` and
schema-level edits under `tests/fixtures`. Everything below belongs to another
owner. The text is written to be pasted, not paraphrased; the numbers in it are
measured on this branch and are cited with where they came from.

## 1. `CHANGELOG.md` (docs group) — rows T6 contributes to `[Unreleased — 2.0.0]`

Under **BREAKING**:

> * A conductor's realization now follows its DECLARATION. `sim.add(Box, …)`
>   with a PEC material is a volume: it realizes tangential walls on BOTH
>   bounding node planes and shorts the normal edges between them. A foil is
>   declared with `add_thin_conductor` (or a zero-thickness Box) and realizes
>   ONE node plane with its normal edge live. Fixtures that drew a foil as a
>   one-cell PEC Box get two walls where they had one unless they are migrated.
>   Measured on the workspace's own patch board (`dx = 196.75 µm`): as one-cell
>   Boxes the ground and patch realize z walls `{28, 29, 33, 34}` and a patch 44
>   × 52 edges wide; declared as sheets, `{29, 34}` and 42 × 50 — the latter is
>   what the pre-2.0 rule gave the same declaration.

Under **CHANGED** (not breaking, but it moves numbers):

> * A sheet's footprint is sampled CLOSED on its two in-plane axes, so a drawn
>   rectangle realizes exactly, hi row included. A 5.0 mm strip on a 0.5 mm cell
>   realizes 5.0 mm where it realized 4.5 mm before. Boards whose in-plane faces
>   are OFF-lattice are unaffected (closed and half-open pick the same nodes).
> * The waveguide S-parameter lane applies the realized PEC edges instead of
>   folding the PEC cell mask back into a `sigma = 1e10` fill. A hard electric
>   wall and a 1e10 S/m lossy volume both reflect with magnitude ~1 and not with
>   the same phase; measured on the chain battery's `pec_short` DUT,
>   `max|ΔS| = 0.938` (coarse rung) and `0.498` (mid rung), while the empty
>   guide reproduces to 2.5e-6.

Under **ADDED**:

> * `tests/locks/test_volume_sheet_cavity_ladder.py` — the eigenmode witness for
>   the two declarations. One parallel-plate cavity, metal starting at the same
>   coordinates, declared twice: volume 52.3341 GHz on the 16-cell ladder, sheet
>   49.8924 GHz on the 17-cell ladder, each within 0.07 % of its own prediction
>   and more than 4.7 % from the other.

## 2. `docs/guides/sparameter_support_matrix.{md,json}` (docs group)

The waveguide lane's "Setup restrictions" subsection carries iris-rasterization
numbers derived from the old realization. T6 does not own the iris cases (that
is crossval group D), but two sentences in that subsection are about the RULE
rather than about case 18/19, and they are now wrong:

* anything saying a PEC septum drawn `t_c` cells presents `t_c − 1` electrical
  cells — under the contract a drawn volume realizes walls at BOTH faces, so
  drawn == realized and the compensation is deleted;
* anything telling a user to draw one cell extra to land on a nominal
  dimension — that is the compensation, not the rule.

Update the `.json` first from group D's re-measured numbers, then mirror into
the `.md`, then run
`tests/contracts/test_support_matrix_parity.py`. That gate is one-directional by
design and will not catch a stale `.md` sentence for you.

## 3. `validation/crossval/manifest.json` (crossval group C)

T6 does not own the cv15 `claim_scope` rewrite and is not proposing wording for
it. One point from T6's side, for whoever does: the narrative currently explains
the ground wall by naming `two_plane`, and that flag no longer exists in any
form — the reader has nothing to look up. The replacement should say the ground
wall is at the substrate floor because a volume realizes both faces (or because
the ground is declared as a sheet ON that plane, whichever cv15 lands on), and
should state plainly that the #740 one-plane build is no longer constructible so
the mode-pair band's rejection endpoint is historical. T6's position on freezing
vs retiring that endpoint is in `T6-open-questions.md` §3.

`tests/contracts/test_evidence_numeric_provenance.py` pins per-case citation
counts (`15_patch_antenna_rt5880`: 3; `18_wr90_iris_modematch`: 4;
`19_wr90_iris_filter_aghanim`: 4; README rows 2/2/6). Update those counts in the
SAME commit as the narrative, or the gate goes red for a reason that has nothing
to do with the narrative being wrong.

## 4. `docs/design_notes/20260906_plan_realign_lattice_ownership.md` (core)

Two sentences T6 would add, both from measurements on this branch:

* **§1.8, after the sigma-fill paragraph:** which DOOR a conductivity comes
  through decides which model it gets. A material on a geometry entry crosses
  `Simulation._PEC_SIGMA_THRESHOLD` in `rfx/api/_compile.py` and becomes a PEC
  BODY (so `material='pec_like'`, `'copper'`, `'metal'` are bodies); a raw
  `rasterize(grid, [(shape, eps, 1e10)])` stays a sigma FILL and is fenced out.
  Two fixtures that both say "PEC" therefore mean different operators, and
  nothing states this where a fixture author would read it. Pinned in
  `tests/contracts/test_lattice_ownership_contract.py::test_a_sigma_fill_conductor_is_not_a_pec_body`.
* **§6, with the conformal amendment:** the waveguide S-matrix lane's change
  from the sigma fold to the realized edges is not free — measured size on the
  chain battery's `pec_short` is `max|ΔS|` 0.938 / 0.498 across two rungs. The
  amendment already says the fold "was the thing §1.7 replaces"; it should also
  say that replacing it moves every committed device-lane S-parameter for a
  conductor DUT, and name the chain battery as the case that has to be
  re-measured. Details: `T6-waveguide-chain-battery.md`.

## 5. `scripts/diagnostics/coaxial_tem_signal_path_audit.py` (scripts group)

`tests/contracts/test_physics_gate_reporting.py` asserts
`checks['outer_conductor_shell_has_pec_cells']['status'] == 'passed'` and
`shell_pec_cell_count > 0`. That check passes on this branch, because the coax
shell is sigma-stamped by `stamp_coaxial_line` and §1.8 leaves it alone. It
still reasons from PEC CELLS, which is the #929 pattern, and its NAME says "pec
cells" while what it is really auditing is "is there metal where the shell
should be". If that audit is ever re-pointed at `realized_wall_planes`, rename
the check in the same change so the test's assertion keeps meaning what it says.
No action is required for #931.

## Added 2026-09-07 after the runs came back — three items for other owners

The first post-contract runs falsified the "the boards realize identically, so
the pins hold" claim. The cause was the #702 slot geometry, not the sheet
declaration (details in `T6-RECOMPUTE.md`); the three boards are redrawn with
each foil ON the laminate face it bounds. Three files T6 does not own follow
from that.

### A. `scripts/diagnostics/patch_edgefed_s11_band_repin.py` (scripts owner)

It imports `_build_patch_sim()` from the gate module, so it follows the redraw
for free — but its `retired` arm replaces
`rfx.api._compile.resample_sheet_node_materials` with the identity, and this
branch DELETES that function (design note §2). The script cannot run as
written.

The A/B it measured is also moot now, and that is the point worth recording
rather than deleting: it asked "what does the #702 own-cell re-sample change on
this board". With the board drawn so the laminate owns every cell of the
cavity, there is no own cell to re-sample and both arms are the same build. The
honest replacement is a DRAWING A/B — the board as drawn now against the board
with a reserved vacuum cell — which is the same physics question asked in the
declaration instead of in a monkeypatch. Its expected size is on the record:
preflight's #703 check reads +84.5 % on `sum(d/eps)` for the reserved-cell
board (Board H) and +45.9 % (Board S), and Leg A measured +10.365 % against a
window centred on -6.17.

Same for `patch_edgefed_s11_band_repin_replay.py` and
`two_plane_patch_radiation_ab.py` (the latter also names a deleted kwarg).

### B. `scripts/patch_edgefed_s11_validation.py` (scripts owner)

The gate module's docstring says its geometry "mirrors" this script. After the
redraw it no longer does: the script still reserves a cell for each foil and
starts the stack at a bare 4 mm. Either re-point it at the test module's
builder (preferred — the mirror is what drifted) or apply the same redraw:
`Z_GND = round(4e-3 / DX) * DX`, foils as zero-thickness Boxes at `Z_GND` and
`Z_GND + H_SUB`, laminate between them.

### C. `docs/design_notes/issue782_retired_resonance_predeclaration.md` (docs owner)

Section 4's two arms are `main` vs "#702 re-sample replaced by the identity".
Under the ownership contract the second arm is unbuildable. The document should
record that the pre-declaration was DISCHARGED and how — the re-sample is gone,
the geometry it compensated for is drawn away, and the board's band is re-pinned
from VESSL 369367259226 (Board S) / 369367259225 (Board H) — rather than be left
naming a function that no longer exists.
