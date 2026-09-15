# #1043 stage B — the pad continuation on the smoothed lane, and what it moved

Stage A (PR #1047) made the two halves of a CPML timestep read the same
permittivity. It said, in its own memory entry, that cv01's and cv03's numbers
stayed **facet-dominated until stage B**. This note is stage B: the pad
material extension now reaches the array the solver actually uses, and this is
what that moved.

Predecessors, and the order matters: `issue831_far_end_return_predeclaration.md`
(section 8 is the landing plan, 8.4 the falsifier this change is judged by),
`issue831_far_end_return_results.md`, `issue1043_cpml_subpixel_coefficient_results.md`.

## 1. The defect, in one paragraph

`_assemble_materials` continues a boundary-touching structure into the absorber
pad by replicating the interior-edge slice of the `eps_r` / `sigma` / `mu_r`
arrays outward — the feature exists, in its own words, "so that guided modes in
dielectric waveguides see an impedance-matched absorber". Under
`subpixel_smoothing` the update permittivity is rebuilt from `sim._geometry`,
and no pad step was applied to the rebuilt array. Every structure touching a
domain face was then solved with `eps_r = 1` in its own absorber, terminated by
an end facet at the interior/pad seam.

## 2. Why the naive fix is wrong, measured before it was written

`extend_cpml_pad_materials` replicates the **interior-edge column**. On the
staircase array that column is the material; on the smoothed array it is the
Kottke half-cell. cv01's committed build, guide centre row, x-lo seam, no FDTD:

| variant | pad | row across the seam | cells at `eps_r = 12` |
|---|---:|---|---:|
| committed (no continuation) | 1.000 | 1.0, 1.0, 6.5, 12.0, 12.0 | 159 |
| naive `extend_cpml_pad_materials` on the smoothed output | 6.500 | 6.5, 6.5, 6.5, 12.0, 12.0 | 159 |
| sourced one column inward | 12.000 | 12.0, 12.0, 6.5, 12.0, 12.0 | 199 |
| reference: `_assemble_materials` | 12.000 | 12.0, 12.0, 12.0, 12.0, 12.0 | 201 |

Sourcing one column inward gets the pad right and strands the half-cell *inside*
the absorber — the one-cell film #655 already had to repair on the staircase
lane. Only continuing the **geometry** and smoothing that reproduces the
reference row, which is what `extend_shapes_into_cpml_pad` does: the reached
face is moved out past the array, so the smoothing sees no interface there and
the pad and seam cells come out at bulk `eps` through the ordinary `f >= 1`
branch. Measured on this tree: 201 of 201, seam row `12.0, 12.0, 12.0, 12.0,
12.0`, cladding untouched at 1.0.

## 3. cv03 — the pre-declared falsifier

Section 8.4, frozen before any fix existed: *the fix is right when the
unmodified case reaches `|B/A| <= 0.03` at 20 layers, the 60-layer value sits
below the 20-layer one, and `settling_db` clears −100 dB. Report band-mean `T`
alongside and do not expect it to move.*

`seam_facet.py --stage sweep`, committed cv03 geometry, before / after:

| arm | `\|B/A\|` before | after | band-mean `T` before | after | settling before | after | exit before | after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| upml 20 | 0.5311 | **0.0296** | 0.9657 | 0.9682 | −37.3 | −138.3 | 2 | 2 |
| upml 40 | 0.5918 | 0.0123 | 0.9470 | 0.9596 | −32.9 | −141.3 | 1 | 2 |
| upml 60 | 0.6221 | 0.0020 | 0.9357 | 0.9558 | −30.3 | −144.4 | 1 | 2 |
| cpml 20 | 0.5240 | 0.0515 | 0.9509 | 0.9559 | −38.8 | −134.7 | 2 | 2 |
| cpml 40 | 0.6006 | 0.0055 | 0.9330 | 0.9571 | −32.7 | −146.9 | 1 | 2 |
| cpml 60 | 0.6309 | 0.0006 | 0.9315 | 0.9565 | −29.9 | −141.2 | 1 | 2 |

The declared arm is `upml 20` — cv03's own absorber and depth. All four clauses
hold, and the numbers land on the ones the hand-widened-`Box` arm B1 produced
(0.0296 / 0.0123 / 0.0020, −138.3 / −141.3 / −144.4): the committed geometry now
solves what widening the `Box` by hand emulated.

`T` is 0.9682 and did not reach 0.99, exactly as 8.4 said it would not. The
second effect the diagnosis lane never isolated is still in there. What did
change is `T`'s **depth trend**: it degraded with depth before and no longer
does.

**The `cpml 20` arm reads 0.0515, above the bar, and that is the absorber and
not a seam.** It falls ~9× per depth doubling (0.0055 at 40, 0.0006 at 60), the
direction an absorber's residual reflection moves and the opposite of a facet's.
The same ladder measured independently on the standalone unit rig reads 0.0509 /
0.0061 / 0.0002 and does not move with transverse clearance (sy 8a vs 16a:
0.0510 vs 0.0509), source position, fit window, or record length (2× steps:
0.0509). Two rigs, one mechanism.

The committed case run unmodified: exit 2 (Meep unavailable in this pod,
unchanged), G1 PASS at max `|n_eff|` deviation 0.226 % against its 2.0 % gate
with a two-wave residual of 0.0049 against 0.05, G2 PASS at band-mean `T` 0.9682
against 1.0 ± 0.05. The case prints its own standing-wave ratio over the band as
**0.017 .. 0.048**, against the **0.393–0.585** the manifest's `claim_scope`
records for it.

Record: `issue1043_cv03_pad_continuation_record.json`, revision 1, adopted by
nothing. `issue812_cv03_dispersion_matched_frequency.json` is untouched.

## 4. cv01 / #813 — the ladder goes flat

`scripts/diagnostics/cv01_cpml_flux_selfcheck.py --mode layer-sweep`, cv01's rig
at four absorber depths, 25000 steps, physical planes held fixed:

| `cpml_layers` | `mean_self` before | after | G2 before | after |
|---:|---:|---:|---|---|
| 10 — cv01 today | 0.748852 | **0.987618** | FAIL | **PASS** |
| 16 — rfx default | 0.884100 | 0.989020 | FAIL | PASS |
| 20 | 0.919530 | 0.989431 | FAIL | PASS |
| 40 | 0.947345 | 0.988931 | FAIL | PASS |

Stage A alone moved the 20-layer arm 0.919530 → 0.916651; stage B moves it to
0.989431.

**This is the reading, and it is not the headline number.** Before, the ladder
climbed 0.749 → 0.947 with depth, and #1027 built a mechanism on that climb: the
absorber's own reflection. After, the ladder is **flat** — spread 0.0018 across
10 → 40 layers, an eighth of a percent, against G2's own half-width of 0.05. The
depth dependence was the seam facet, not the absorber. The outside-aperture
fraction says the same thing from the other side: −26.04 / −8.56 / −4.90 / −1.94 %
before, +0.94 / +0.92 / +0.92 / +0.98 % after — flat, and no longer negative.

#1043's body said #1027's monotone trend "is NOT explained by the facet — that
observable is blind to a standing wave by construction — so it must be
re-measured after a fix, not reasoned about." It was re-measured. The trend is
gone, so the facet did explain it, and the reasoning that said otherwise was
wrong about which observable sees what.

Six-arm mode, 10 layers, cv01's own rig:

| arm | `mean_self` before | after | G2 before | after |
|---|---:|---:|---|---|
| `upml_full` | 0.9891615 | 0.9910753 | PASS | PASS |
| `cpml_full` | 0.7488520 | **0.9876177** | **FAIL** | **PASS** |
| `upml_interior` | 0.9891742 | 0.9912695 | PASS | PASS |
| `cpml_interior` | 0.7593909 | 0.9879249 | **FAIL** | **PASS** |
| `cpml_aperture` | 0.9912322 | 1.0010475 | PASS | PASS |
| `upml_aperture` | 0.9927370 | 1.0000066 | PASS | PASS |

**#813's headline — "cv01 self-check FAILS under `RFX_BOUNDARY=cpml` (flux
0.7489 vs 0.95–1.05)" — now reads 0.9876 and passes G2.** The UPML control moves
too, 0.9891610388008335 → 0.9910752993577158 (+0.194 %): cv01's guide touches
both x faces under UPML as well, so UPML carried a facet too — a small one,
because a UPML pad reflects a facet less than a CPML pad does.

What is NOT done here: cv01's committed crossval record
(`validation/crossval/_01_waveguide_bend_results/crossval.json`,
`mean_self_smoothed_over_band = 0.9891610388008335`) is a UPML record and is
**not** re-run and **not** re-pointed by this change. The driver constant that
names it keeps the committed value; the measured value rides beside it.

### 4.1 What was re-pointed, and how

The sweep's control constants described the pre-#1043 solver and no longer
describe what the driver measures. They are re-pointed **explicitly**, as a
named revision rather than an edited value: `SWEEP_BASELINE_REVISION = "r2"`,
with `..._R1` and `..._R2` both kept and both written into every artifact the
driver produces, so which revision a run reproduces is readable from the record.

Deliberately NOT re-pointed with it: the pre-declared falsifier
`near_baseline`, which asks whether `mean_self` stayed within 0.03 of
**0.748852**. That number is what the pre-declaration froze, not "the current
baseline". Letting it follow the re-pointed constant would have redefined a
frozen falsifier into "within 0.03 of whatever we just measured", which always
holds and never fires.

Also narrowed: the sweep's pass sentence used to end *"the absorber's own
reflection is the source of the deficit"* unconditionally. That attribution came
off a ladder measured on the pre-#1043 solver. It is now conditional on a spread
the run measures, against a threshold equal to G2's half-width — a ladder whose
entire spread is smaller than the gate's half-width cannot move an arm across
the band it is judged by. The gate arithmetic did not change.

The 26 value-checked citations the numeric-provenance contract resolves out of
`layer_sweep.json`, and the 19 of the residual-split section, all still resolve:
that artifact is not edited. The new measurement is `layer_sweep_r2.json`, a new
file beside it.

## 5. Bit identity for everything else

Eight configurations that must not move, A/B against `git archive origin/main`
on one host, SHA-256 over the six final field arrays plus the probe trace, in
both precision lanes: vacuum pads (2-D), interior dielectric under CPML, the
same under UPML, the staircase lane on the very geometry this changes
(`subpixel_smoothing=False`), a touching dielectric behind PEC walls (no pad to
continue into), 3-D interior dielectric, the Stage-2 `kottke_pec` tensor, and
the non-uniform mirror. **All sixteen digests identical.** Harness:
`scripts/diagnostics/pad_continuation_bit_identity.py`.

## 6. What is deliberately not continued

* **PEC volumes.** `pec_mask` is not in `extend_cpml_pad_materials`' signature
  either, so the staircase lane ends a PEC structure at the seam too. The two
  lanes have to agree about what stands in a pad.
* **Dispersive materials.** #627b measured a high-Q Lorentz pole carried into a
  pad turning a stable edge-touching run divergent, with no NaN and no exception
  to catch it; #808 measured the other half — promoting such a column's statics
  alone puts `eps_inf` without its poles into the pad, a material no declared
  model has, and moved a committed Debye recovery from 11 % to 32 % error past
  its 20 % gate. The smoothed lane inherits both rules rather than inventing a
  third answer.
* **Shapes with no continuation across the reached face** — a sphere's tangency,
  a cylinder reached across its axis, an imported mesh. These were silent, and
  are now the new preflight advisory `dielectric_at_absorber_seam`.

## 7. Numeric provenance

cv03, after:
`scripts/diagnostics/_artifacts/cv03_seam_facet/sweep_r2.json::stages.sweep.sweep_upml_20.b_over_a_carrier_bin = 0.0296`,
`scripts/diagnostics/_artifacts/cv03_seam_facet/sweep_r2.json::stages.sweep.sweep_upml_40.b_over_a_carrier_bin = 0.0123`,
`scripts/diagnostics/_artifacts/cv03_seam_facet/sweep_r2.json::stages.sweep.sweep_upml_60.b_over_a_carrier_bin = 0.0020`,
`scripts/diagnostics/_artifacts/cv03_seam_facet/sweep_r2.json::stages.sweep.sweep_cpml_20.b_over_a_carrier_bin = 0.0515`,
`scripts/diagnostics/_artifacts/cv03_seam_facet/sweep_r2.json::stages.sweep.sweep_cpml_40.b_over_a_carrier_bin = 0.0055`,
`scripts/diagnostics/_artifacts/cv03_seam_facet/sweep_r2.json::stages.sweep.sweep_cpml_60.b_over_a_carrier_bin = 0.0006`,
`scripts/diagnostics/_artifacts/cv03_seam_facet/sweep_r2.json::stages.sweep.sweep_upml_20.T_rfx_band_mean = 0.9682`,
`scripts/diagnostics/_artifacts/cv03_seam_facet/sweep_r2.json::stages.sweep.sweep_upml_20.settling_db = -138.34`.

cv03, before (left byte-for-byte as the diagnosis lane recorded it):
`scripts/diagnostics/_artifacts/cv03_seam_facet/sweep.json::stages.sweep.sweep_upml_20.b_over_a_carrier_bin = 0.5311`,
`scripts/diagnostics/_artifacts/cv03_seam_facet/sweep.json::stages.sweep.sweep_upml_60.b_over_a_carrier_bin = 0.6221`,
`scripts/diagnostics/_artifacts/cv03_seam_facet/sweep.json::stages.sweep.sweep_upml_20.settling_db = -37.33`.

cv03 case record:
`docs/design_notes/issue1043_cv03_pad_continuation_record.json::committed_case_run.exit_code = 2`,
`docs/design_notes/issue1043_cv03_pad_continuation_record.json::falsifier_as_predeclared.passed`,
`docs/design_notes/issue1043_cv03_pad_continuation_record.json::falsifier_as_predeclared.b_over_a_at_20 = 0.0296`,
`docs/design_notes/issue1043_cv03_pad_continuation_record.json::falsifier_as_predeclared.b_over_a_at_60 = 0.0020`.

cv01, after:
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep_r2.json::layers.10.mean_self_full = 0.987618`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep_r2.json::layers.16.mean_self_full = 0.989020`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep_r2.json::layers.20.mean_self_full = 0.989431`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep_r2.json::layers.40.mean_self_full = 0.988931`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck_r2.json::arms.cpml_full.mean_self = 0.987618`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck_r2.json::arms.upml_full.mean_self = 0.991075`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck_r2.json::arms.cpml_interior.mean_self = 0.987925`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck_r2.json::arms.upml_interior.mean_self = 0.991270`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck_r2.json::arms.cpml_aperture.mean_self = 1.001047`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck_r2.json::arms.upml_aperture.mean_self = 1.000007`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck_r2.json::gate_verdict_cpml_full`.

cv01, before (unchanged artifacts):
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.10.mean_self_full = 0.748852`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json::layers.40.mean_self_full = 0.947345`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck.json::arms.cpml_full.mean_self = 0.748852`,
`scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck.json::arms.upml_full.mean_self = 0.989162`.

## 8. R2 and R3

**R2**: one attempt. The fix's shape was settled with no FDTD
(`pad_replication_shape.py`, section 2 above) before any solver ran, and the
falsifier was pre-declared in section 8.4 before the implementation existed.

**R3**: `memory=rfx-known-issues.md "Added 2026-09-15"` — the #831 root-cause
entry names the three sites and the missing step, and the #1043 stage-A entry
says both cv01 numbers stay facet-dominated until stage B. Consistent with both;
contradicts neither. `R2-attempts=1`. `falsifier=section 8.4 re-run on the
committed cv03 geometry` — 0.0296 at 20 layers, 0.0020 at 60, −138.3 dB.

## 8a. A FOURTH site, found here, counted, and deliberately not changed

`#1043`'s table names three sites. There is a fourth:
**`rfx/sparams/waveguide.py`**, the waveguide S-parameter lane, which builds
its own `shape_eps_pairs` from `self._geometry` for both the Stage-2 and the
Stage-1 branch under a comment saying "Mirrors rfx/runners/uniform.py". Same
defect class, and after this change it is the one place that mirror no longer
holds.

**Nothing in the tree reaches it**, counted rather than assumed (round-1
review; the first draft of this section said "unmeasured", which was true when
written and is no longer):

* the parameter defaults to `False` (`waveguide.py:76`), so a caller opts in;
* exactly **one** in-tree caller passes a truthy value —
  `tests/unit/sparams/test_waveguide_nu_sparam.py:396` — and it is a **fence**:
  `pytest.raises(NotImplementedError, match="subpixel_smoothing")` on the NU
  dispatch. It stops before any solve;
* every other caller of `compute_waveguide_s_matrix` takes the default,
  including the v1.8 chain-closure battery and this issue's own F1 PEC-short
  gate driver (`scripts/diagnostics/cpml_subpixel_stability/f1_pec_short_gate.py:143`,
  which passes no `subpixel_smoothing` and declares only PEC);
* and PEC volumes are continued by neither lane, so a PEC-only fixture could
  not move even if it did enter.

So it is a **latent gap, not a live wrong number**. Left alone because the lane
is v1.8 chain-closed — 185 verdicts replay against a frozen artifact, and
moving its numbers is a measurement change wanting its own pre-declaration
("refactoring and measurement changes do not travel together", #928). Folding
it into `smoothed_shape_pairs` when the waveguide lane next opens for a
measurement change is tracked as **#1066**. For whoever takes it: the
reference run passes `dielectric_shapes=[]` and cannot carry a facet, so only
the device run can.

## 9. What this does not close

* **cv03's band-mean `T` at 0.968.** The second effect is unisolated and this
  change does not claim it. Section 8.4 pre-declared that, and it holds.
* **The `cpml 20` arm at 0.0515.** Attributed to the absorber by a depth ladder
  measured on two independent rigs; not gated at 0.03 anywhere, and the unit
  test reads its bar at 40 cells for that reason, with the depth trend asserted
  separately as the discriminator.
* **cv01's committed crossval record.** UPML, not re-run here. Its value moves
  by +0.194 % when re-measured, which is recorded and not adopted.
* **The comparator-bug case ledger.** If this lands with #831 and #813 closing
  on it, it is the first entry in that ledger to close as an rfx
  **solver-assembly** defect rather than a comparator or extractor one. The
  ledger and `research/CLAUDE.md`'s invariant sentence live in gitignored
  memory; updating them is a separate act at the landing, not part of this
  change.
