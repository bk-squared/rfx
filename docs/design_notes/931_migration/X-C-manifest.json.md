# X-C → `validation/crossval/manifest.json` (cases 14, 15, 16, 17)

Owner of this text: the X-C crossval migration agent (issue #931, branch
`feat/931-crossval-C`). `manifest.json` itself is owned by the merge/ingest
agents, so the replacement text lives here.

Two of the four edits are ready to apply now. The cv15 `claim_scope` carries
NUMBERS that come from a re-solve; the rules for filling them are written out
below, and nothing here may be hand-translated from the old figures.

---

## 1. `id: 14_rect_cavity_pozar` — `claim_scope`

One clause changes. Find:

> Gated modes (re-declared 2026-08-31, issue #812): Gate 0 wall registration
> `|(n-1)*dx - (a,b,d)| <= 1e-9 m` on every axis;

replace with:

> Gated modes (re-declared 2026-08-31, issue #812; Gate 0 re-based 2026-09-07,
> issue #931): Gate 0 wall registration `|realized wall separation - (a,b,d)|
> <= 1e-9 m` on every axis, where the separation is MEASURED — the shipped
> `apply_pec` / `apply_pec_faces` are applied to an all-ones state and the
> planes are read back through `rfx.boundaries.pec.realized_wall_planes`, which
> must be exactly `{0, n-1}` per axis, and the same measured planes feed the
> Yee oracle. Before #931 this gate computed `(n-1)*dx` from the grid SHAPE and
> would have kept passing under any change to how a PEC wall is realized. The
> 1e-9 m tolerance is unchanged, so this is a strict tightening;

Rationale for the reviewer: domain-boundary PEC is fenced out of the ownership
contract (design note §1.8) and keeps its own convention. cv14 is the case that
proves the fence holds, and until #931 nothing measured it.

## 2. `id: 16_pec_sphere_mie_ka_sweep` — `claim_scope`

Append one sentence to the end of the existing `claim_scope` (nothing else in
it moves — no cv16 number changes under #931):

> CONDUCTOR MODEL (issue #931): this case's sphere is a high-sigma MATERIAL
> FILL (`rasterize(Sphere, eps_r 1.0, sigma 1e7)` straight onto
> `MaterialArrays`), never a `Simulation.add(..., material='pec')` entry, and
> the lattice ownership contract FENCES that model out (§1.8): a sigma fill is
> a lossy volume model on NODE samples and is unchanged here, while a declared
> PEC volume is sampled at cell CENTRES (§1.1). The two are checked against each
> other at build time instead of being assumed equal —
> `assert_conductor_model()` refuses to quote an RCS unless the fill still
> equals `Sphere.mask(grid)`, and reports the centre-sampled cell set beside
> it. Measured at this case's own gated points: ka = 0.50 coarse N 1082 → 1123
> (a_eff/a 0.988032 → 1.000357, 259 cells differing), fine ka = 2.00 N 9264 →
> 9339 (0.998319 → 1.001006, 957 cells differing). Bringing this case under the
> contract would therefore move a_eff by ~1.2 % at ka = 0.5, move the Mie
> reference leg with it, and require the fixture and both gate constants to be
> regenerated from a re-solve; that is not done here.

## 3. `id: 17_dielectric_sphere_mie` — `claim_scope`

Append one sentence (no cv17 number changes; this case is the #931 control):

> DIELECTRIC CONTROL (issue #931): the lattice ownership contract changes PEC
> VOLUME sampling from node to cell-centre and leaves dielectric sampling
> untouched (§1.1), so every number in this case must be bit-identical across
> that change. `tests/crossval/test_rcs_dielectric_sphere_mie_gates.py` asserts
> it at every gated bin: exactly two distinct eps values with the
> non-background one at the declared 2.56, and an occupied-cell count equal to
> the NODE-sampled shape mask (1082 at ka = 0.5, the same sphere on the same
> mesh as case 16). If this count ever picks up centre sampling, every number
> in this case is a different sphere.

## 4. `id: 15_patch_antenna_rt5880` — `claim_scope` (numbers MEASURED, apply as written)

The re-solve is done (VESSL 369367259156) and the decomposition run that
attributes it is done (VESSL 369367259164), so nothing here is a placeholder.

### 4a. The mechanism clause

Find, inside `claim_scope`:

> Post-#740 (ground wall realized at the substrate floor via two_plane; the
> pre-fix one-plane ground left a vacuum cell in the cavity and read +6.09% vs
> openEMS) rfx reads 0.69% LOW vs openEMS (2.3139 vs 2.330 GHz) and -4.21% vs
> the analytic anchor, openEMS -3.54%: both solvers sit on the same side of the
> closed form by a similar margin; that direction is discretisation, reported
> not hidden.

replace with:

> Post-#931 (lattice ownership contract) BOTH conductors are declared SHEETS —
> zero-thickness PEC `Box`es on the two substrate faces — which is the same
> structure both openEMS legs build (`AddBox` with start z == stop z at 0 and
> at h). The feed spans the full substrate, standing ON the ground sheet plane,
> matching the openEMS `AddLumpedPort` span; a port on a conductor's node plane
> is galvanic, not "inside PEC" (#929). This replaces three repairs of one
> undeclared sheet rule: the `two_plane=True` ground drawn a cell below the
> floor (#740, itself the remedy for the #693 vacuum ground cell that read
> +55.0 % electrical thickness and +6.09 % vs openEMS, preserved as
> `validation/crossval/_15_patch_results/rfx_one_plane_ground_b29f9de7.json`),
> the one-cell patch `Box` whose far wall at 11.9062 mm was suppressed only by
> a realization default, and a feed held one cell short of the patch so
> coupling was capacitive only (#556). `assert_realized_stack()` now reads the
> contract's one edge set (`realized_pec_edge_masks` / `realized_wall_planes`)
> and refuses to quote f0 unless the realized wall planes over the patch
> footprint are EXACTLY {z_sub_lo, z_sub_hi} — no wall at k_patch+1 — and the
> assembled eps_r still holds exactly the two declared values, a sheet owning
> no cell and writing no material. MEASURED on the regenerated leg: rfx reads
> 4.58 % HIGH vs openEMS (2.4366 vs 2.330 GHz) and +0.87 % vs the analytic
> anchor, openEMS -3.54 %. The two solvers no longer sit on the same side of
> the closed form — rfx reads high, openEMS low — and the rfx-vs-openEMS
> distance grew from 0.69 % to 4.58 % while the rfx-vs-analytic distance shrank
> from -4.21 % to +0.87 %; the openEMS leg did not move. The S11 dip deepened
> from -4.4 dB to -19.7 dB against openEMS's -20.1 dB, and the ring-down Q fell
> from 18.9 to 10.2: the feed now reaches the conductor instead of coupling
> capacitively. A decomposition run (production sheets with the PRE-#931 feed,
> `_15_patch_results/rfx_decomposition_feed_pre931.json`) splits the +5.30 %
> almost evenly — the conductor declarations account for +2.48 % (2.3139 →
> 2.3713 GHz, Q 18.9 → 18.1) and the feed change for a further +2.75 %
> (2.3713 → 2.4366 GHz, Q 18.1 → 10.2, and essentially all of the Q collapse).
> The old 0.69 % agreement with openEMS was therefore not a better model: it
> was a cavity electrically 55 % too thick, read through a feed that never
> touched the patch. The #768 leg is preserved verbatim as
> `validation/crossval/_15_patch_results/rfx_pre931_two_plane_ground_1f005d0d.json`.
> All six gates pass on the new leg, including the stack gate.

The sentence "both solvers sit on the same side of the closed form by a
similar margin" is DELETED, not reworded: rfx is +0.87 % (high) and openEMS
-3.54 % (low), which is opposite sides.

## 5. What must NOT change

* No gate threshold in any of the four cases. cv15's f0 envelope stays 8 %,
  the settling bar -40 dB, passivity 1.05, the directivity envelope 3 dB;
  cv14's Gate 0 tolerance stays 1e-9 m and Gates 1-3 stay 1 % / 2 % / 0.1/T;
  cv16 stays 3.3 / 4.0 dB; cv17 stays 6.3 dB and 0.5 % relative.
* `role`, `evidence_levels`, `execution_tiers`, `expected_exit_codes` and
  `failure_sentinel` on all four.
* cv15's delegation of patch ACCURACY to case 05.
