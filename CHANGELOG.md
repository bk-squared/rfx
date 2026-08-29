# Changelog

All notable changes to `rfx-fdtd` that affect user-visible behaviour are
recorded here. Dates follow local (KST) convention. Version bumps follow
SemVer — **BREAKING** entries are flagged in upper-case.

## [Unreleased]

### Fixed — MSL plane-primitive V/I now call the production extractor (issue #514)

`rfx/probes/msl_wave_decomp.py`'s `register_msl_plane_probes` /
`_v_from_plane` / `_i_from_plane` — the plane-DFT geometry primitive
`validation/tmtt_paper/msl_stub_notch_tuning.py` drives its inverse-design
cost through — duplicated `compute_msl_s_matrix`'s V/I integration instead
of calling it, and had drifted in three places: an inclusive `k_lo..k_hi`
Ez span (~12% low V, the same defect issue #511/PR #516 fixed in
production), a single pre-issue-#80 Hy-slab current (~1.5x undercount vs.
the closed Ampere loop), and no leapfrog E/H half-step phase correction
(issue #240). `_v_from_plane` / `_i_from_plane` now call
`rfx.api._sparams.msl_modal_voltage` / `rfx.sources.msl_port.msl_loop_current`
directly, with the trace-conductor span found by the SAME PEC-mask walk
`compute_msl_s_matrix` uses (not the `round(h_sub/dx)` proxy), so this path
cannot re-drift from production.

`register_msl_plane_probes` now registers an Hz plane alongside the
existing 3 Ez + 1 Hy planes (needed for the closed loop) and raises
`NotImplementedError` on a non-uniform mesh (`sim._dx_profile` /
`_dy_profile` / `_dz_profile` set) — `sim._build_grid()` never threads
those profiles into the `Grid` it returns, so this path had no way to
build the same `NonUniformGrid` production's non-uniform branch uses for
its trace-PEC search; anchoring on the uniform-only lookup would have
silently mis-anchored V on a graded mesh.

No production code changed (`rfx/api/_sparams.py`, `rfx/sources/msl_port.py`
untouched). The `validation/tmtt_paper/msl_stub_notch_tuning.py` -46.1 dB
single-variable-descent objective is produced by this lane and is pending
re-derivation after this fix — see the footnote in
`validation/tmtt_paper/README.md`; the -55.7 dB validated optimized null
comes from the (unchanged) production S-matrix path. The
`dx = 254 µm` (1-substrate-cell) MSL case in
`scripts/diagnostics/optimizer_bakeoff/` is not measured post-#514.

Follow-up (not this change): `scripts/diagnostics/patch_edgefed_stage6_voltage_deembed.py`
has its own hand-rolled inclusive `dz[k_lo:k_hi + 1]` voltage span
(`_V_from_plane`) — the same drift class as #514, outside this module;
needs its own issue.

### Changed — PMC-plane convention decided: REALIZE-DECLARED (issue #722 ninth surface)

`apply_pmc_faces` zeros H_tan a half-cell (0.5*dx) INSIDE the declared wall
on every PMC face (`rfx/boundaries/pmc.py`, measured 2026-04, pinned by
`tests/test_boundary_pmc_hi_faces.py` — untouched here). A PMC-mirrored
model's declared mirror plane and its REALIZED H_tan wall have therefore
always differed by half a cell; this closes the campaign-standing question
of which side owns that gap.

**Decision: REALIZE-DECLARED (odd-cell).** A PMC mirror plane must be
declared at `plane + dx/2`, not `plane`, so the H_tan zero lands ON the
intended plane instead of a half-cell short of it. This requires an ODD
cell count on the mirrored axis (so `plane` itself is an H-node); the
alternative, quote-realized (declare `plane`, always compare against
`a_eff = a - dx`), was rejected because it carries the half-cell bias
forever instead of removing it once.

- `rfx.fidelity.fidelity_report`'s domain row is now self-diagnosing on a
  PMC-faced axis (#729 spirit): `realized_um` / `realized_extent_um` /
  `face_residual_um` report the REALIZED H_tan wall (shrinks by half the
  boundary cell per PMC face), not the raw mesh line. The node-to-node mesh
  span is preserved separately as the new `mesh_extent_um` key (unaffected
  by any PMC face; the pre-existing `domain-extent-quantized` finding stays
  keyed on it). A new finding kind, `pmc-wall-half-cell-inside`, names the
  convention explicitly wherever it applies, so a PMC-mirror script cannot
  ship the offset silently.
- `validation/crossval/09_half_symmetric_waveguide.py` applies the
  convention: mesh `dx` 0.635 -> 0.508 mm (0.025 in -> 0.020 in; WR-90's
  broad wall is then 45 cells, ODD, so a/2 is an exact H-node plane), half
  domain declared `a/2` -> `a/2 + dx/2`. Measured: full 8.1958 GHz (Q
  5.11e4), half 8.1959 GHz (Q 6.47e4), gates 0.007% / 0.007% / 0.001% (was
  0.009% / 1.825% / 1.835% pre-change) — gate 3, the PMC-mirror
  self-invariant, drops from a fixed ~1.8% geometry bias to a genuine
  mesh-only residual.
- `validation/crossval/10_pmc_cpml_half_symmetric.py` (a regression lock
  with no closed-form reference, so the offset biases no gate here) and
  `rfx/interop/emitters/openems.py` (D17: the emitter maps `pmc` straight
  to openEMS's `'PMC'` on the mesh line, so an rfx-vs-openEMS comparison on
  a PMC-faced structure needs the rfx side declared this way) now state the
  convention in their own docstrings. `rfx/convergence.py`'s `sim_factory`
  gets a note: a dx sweep that clones a PMC-faced `domain` unchanged moves
  the realized mirror plane by dx/2 per refinement step.

### Fixed — cv15 patch cavity was one vacuum cell taller than its declared substrate (issue #740)

`validation/crossval/15_patch_antenna_rt5880.py`'s mandatory geometry self-check
(the `#325 AVOIDANCE` z-rasterization assert) covered the substrate's z EXTENT
only, not which node plane the bounding PEC walls actually land on. The
one-cell one-plane ground `Box` (the `add()` default, #677-validated) realizes
its electric wall on its LOWER node plane only — one cell BELOW the declared
substrate floor `z_sub_lo`, leaving a live vacuum cell in the cavity (measured
against the mask-derived realized planes: +55.0% electrical thickness versus
the declared 4-cell gap; this is the #693 "vacuum ground cell" trap, closed on
the canonical patch lane by PRs #716/#718). The patch wall was NOT displaced —
its default one-plane wall already sits on its lower face, exactly at
`z_sub_hi`.

Fixed with `two_plane=True` (issue #706) on the GROUND `Box` only (0.0%
electrical-thickness error measured after the fix); NOT on the patch, which
would add an unreferenced wall one cell above `z_sub_hi` that the openEMS
zero-thickness-patch reference has no counterpart for. A new mandatory
self-check, `assert_realized_stack()`, asserts the REALIZED wall planes across
the whole patch footprint (ground at `z_sub_lo`, patch at `z_sub_hi`), not the
declared Box extents, and `compare()`'s new `stack geometry fidelity` gate
re-verifies a leg's recorded `stack_check` against this module's own constants
— a missing `stack_check` (a leg from before this fix) is a FAIL, not a skip.

The pre-fix one-plane-ground result leg is preserved as
`validation/crossval/_15_patch_results/rfx_one_plane_ground_b29f9de7.json`
(committed at b29f9de7, `--num-periods 45 --gain`, 208 s CPU): `f_primary`
2.4719 GHz (Harminv Q 17.04), analytic anchor 2.4156 GHz, openEMS S11 dip
2.330 GHz (+6.09% rfx-vs-openEMS on the pre-fix realization), settle -59.7 dB,
gain 7.357 dBi vs openEMS 7.335 dBi.

Post-fix leg MEASURED and committed as `rfx.json` (same command, CPU, 206.5 s,
`JAX_ENABLE_X64=0`): `f_primary` 2.3139 GHz (Harminv Q 18.9) vs openEMS
2.330 GHz — rfx-vs-openEMS **0.69%** (was +6.09%), rfx-vs-analytic 4.21%
(openEMS-vs-analytic is 3.54%, so the two solvers now sit on the same side of
the closed form by a similar margin); settle −54.0 dB; max|S11| 0.787. The
vacuum ground cell was the dominant term in cv15's cross-solver gap. Every
compare() gate PASSES including the new stack gate (ground wall 7.9375 mm,
patch wall 11.1125 mm, 4 cells of eps 2.2, provenance `two_plane`). `#715`
(cv15's patch-length accuracy, `L_PATCH`/`W_PATCH`, the `--f0-env-pct` gate)
is untouched by this fix; the f0 envelope did not need to move.

Known checker gap, filed as #767: preflight's #703 sheet-cavity check does not
model `two_plane` walls and still prints the pre-fix +55% on this geometry —
the realized `conductor_mask` (walls at k = 17, 18, 22) is the record, not
that advisory.

### Changed — surface-impedance sheets accept patterned shapes, not just boxes (issue #674)

`add_thin_conductor(..., surface_impedance_f0=...)` — the opt-in band-centre
Leontovich conductor loss — used to refuse any shape that was not a `Box`. The
sheets that dominate conductor loss on a printed board are patterned: ground
planes with clearance holes, meandered arms, outlines that arrive from CAD as
meshes. Those all stayed lossless PEC, which is the one case the feature exists
for.

The restriction was scoping, not physics: `sigma_eff = 1/(Rs0 * d_norm)` is a
per-occupied-cell fold, and thin conductors have rasterized through
`mask_on_coords` on both lanes since issue #369. Any shape implementing
`mask_on_coords` and `bounding_box` — `Cylinder`, `Sphere`, `MeshShape`, your
own — may now carry a surface-impedance sheet, on the uniform and the graded
(`dz_profile`) lane alike. A hole is simply a cell the fold never touches.

Two structural requirements replace the shape check, and both fail loud rather
than degrade quietly:

- the shape must rasterize to **exactly one cell layer along its normal** — a
  body with height (an imported solid, a bent sheet, a slab thicker than a
  cell) is not a sheet, and folding it per cell would multiply the sheet
  conductance by the layer count while still reporting `Rs0`;
- it must rasterize to **at least one cell** — a sub-cell mesh slab registers
  only where a grid node falls inside it, and silently vaporized metal is the
  issue #369 class.

Existing models do not move: a `Box` sheet takes the same bounds and the same
arithmetic it always did, and a Box expressed as an equivalent mask shape folds
a bit-identical `sigma` on both lanes. The issue #669 transition-node oracle
re-run with non-Box plates reproduces its numbers exactly (alpha ratios 0.9984
and 1.0005 against their locally-uniform controls).

### Added — `report_every=N` makes a long solve supervisable (issue #667)

A solve printed nothing between the call and its return, so a slow run was
indistinguishable from a hang. The case that prompted it: a 42.15 M-cell /
225,000-step `compute_msl_s_matrix` whose last log line was written at
second 0 and was still the last line 4 h 10 min later, with the job still
`running`. A rate extrapolated from the same campaign's 12.6 M and 24.5 M-cell
runs predicted ~2 h and was wrong, because per-cell throughput falls at
larger grids — so there was no way to tell a bad estimate from a stuck job
except by waiting.

`Simulation.run(report_every=N)`, `compute_msl_s_matrix(report_every=N)` and
the low-level `rfx.simulation.run` / `run_until_decay` now emit one line per
N steps on preflight's stdout channel:

```
  [PROGRESS] MSL drive p1/2: 500/787 steps (63.5%) | elapsed 0:00:02 | 185.9 steps/s | ETA 0:00:01
```

**The default is `None` — off — and nothing moves for existing callers.**
With `report_every=None` the solve executes the unchanged single-`lax.scan`
code path, so identity there is structural rather than tested-into.

**Turning it on does not move a number either, and that is gated rather
than asserted.** `run()` is one `jax.lax.scan`, and printing from inside the
scan body would mean `jax.debug.print` / `io_callback` on the hot path.
Instead the same compiled scan is driven from the host in `report_every`-step
chunks with `carry` threaded through. The carry already holds the DFT
accumulators and every port / flux / monitor state, so this is a
continuation, not a re-solve. `tests/test_run_progress_reporting.py` locks
that with SHA-256 digests over raw field bytes, the probe time series and the
extracted `S` / `Z0` / `beta` — equal, not merely close — including chunk
sizes that leave a ragged final chunk, and with a negative control proving
the digests move when the physics does. The same architecture (host loop over
constant-length jitted chunks) already shipped on the non-uniform
`until_decay` lane in #383.

**The per-chunk outputs are joined with a bounded-arity grouped
concatenate, because the obvious flat one does not scale in the direction
this feature is used.** `jnp.concatenate(parts)` takes one *argument* per
chunk, and XLA's trace-and-compile cost grows superlinearly in argument
count — so the cost lands hardest on exactly the long runs `report_every`
exists for. Measured on an isolated `(225000, 20)` float32 join, first
(compiling) call:

| chunks | flat `jnp.concatenate` | grouped |
|---|---|---|
| 2,250 (225k steps @ `report_every=100`) | 2.567 s | **0.133 s** |
| 22,500 (225k steps @ `report_every=10`) | 232.2 s | **0.994 s** |

In a real solve at 200 chunks, the concatenate compilations drop from one
per chunk to **3**. Grouping is exact — concatenation is associative, so the
joined bytes are unchanged, which is what makes the optimisation admissible
at all and is locked by `test_grouped_concat_matches_flat_concat`. The join
stays on device; a host `numpy` round-trip was faster still in isolation but
buys nothing at realistic chunk counts and would add a dtype-support
question for no gain.

**Runtime cost: below what this measurement can resolve, and the earlier
table quoting it was withdrawn.** An initial version of this entry published
−1.4 % / +1.0 % / −0.2 % / +10.7 % from a 3-run median with no stated noise
floor. Those figures do not survive scrutiny and are retracted. What five
measurement attempts on a 118k-cell / 2000-step CPU fixture actually found:

- Round-robin over 8 configurations, 21 reps: two configurations
  **byte-identical to the baseline** read **+5.6 %** and **+7.6 %**, and every
  reporting cadence sat flat at +15.7…+18.5 % — including `report_every ==
  n_steps`, which is a *single chunk* and cannot cost 17 %. The harness was
  measuring position in the sequence, not the feature.
- An order-flip A/B confirmed it: the same configuration ran **13 % faster**
  in the second block than the first, while *adjacent* pairs reproduced
  **+0.7 % / +0.8 %** for the single-chunk case in **both** orders.
- A paired estimator (adjacent off/on pairs, bootstrap CI) put the
  **off-vs-off self-control at −4.8 % [−9.3, −0.9]**, so the floor is ≈ ±9 %.
  Against it: `report_every` 2000 → −0.25 %, 500 → +1.06 %, 250 → +0.12 %,
  100 → +1.71 %, 25 → +5.02 %. Every one is inside the floor.

So: **no cadence-dependent cost is resolvable above the noise on this
fixture.** The one defensible number is the order-flip A/B's +0.7–0.8 % for a
single-chunk run, which bounds the fixed per-report cost — one device
synchronisation plus one print — at well under 1 %. There is weak,
non-conclusive evidence of a real cost at the finest cadence tested
(`report_every=25`, a chunk of ~70 ms), where several attempts land positive;
its magnitude is not pinned. Direction only: finer cadence costs more.

**What this measurement does not cover.** It is a small CPU, float32 fixture.
It says nothing about the 42.15 M-cell / 225,000-step GPU regime that
motivated the issue, where a 1000-step chunk is minutes of work and the
per-report synchronisation should be proportionally invisible — that
expectation is untested here. Size `report_every` for a useful reporting
interval rather than to minimise an overhead this benchmark could not detect.

XLA compilations, by contrast, were counted directly and are exact: full
chunks share one executable and a ragged final chunk compiles once more
(200 steps → 1 scan compile unchunked, 1 at `report_every=50`, 2 at
`report_every=60`).

Scope and fences, stated rather than discovered later:

- **Uniform lane only.** The distributed, non-uniform, ADI and subgridded
  lanes emit a `UserWarning` naming the reason instead of running silently
  to completion with the request dropped.
- **Forward-only.** It reads the host wall clock, so under
  `jax.jit`/`grad`/`vmap` it raises rather than printing one fabricated line
  at trace time. `compute_msl_s_matrix(eps_override=...)` routes through the
  traced `forward()` and warns that it is ignored there.

  The guard checks the pytrees it is handed — the carry, the per-step inputs
  **and** the material / geometry arrays — rather than asking JAX whether a
  trace is active, which its current API does not expose
  (`jax.core.trace_state_clean` does not exist in JAX 0.10, and a freshly
  built constant is a tracer only under `jit`, not under bare `grad`/`vmap`).
  The material arrays are the route `forward()` / `optimize()` actually trace
  through; a first version checked only the carry and the per-step inputs,
  which stay concrete under bare `grad`/`vmap`, and so printed trace-time
  lines instead of raising. Both are now rejected. A tracer arriving solely
  through some other closure would still slip past and print a meaningless
  elapsed and rate; the computed values are byte-identical in that case, so
  the residue is cosmetic, and the docstrings say so rather than promising a
  guarantee the check cannot make.
- Rejected together with `checkpoint_segments`, whose own scan-of-scans
  segmentation cuts the same axis.
- Nothing in the path branches on a traced value; every loop bound is
  Python-int arithmetic.
- `report_every` must be a whole number ≥ 1. `inf` (which raises
  `OverflowError` from `int()`, not `ValueError`) and `bool` (Python bools
  are ints, so `True` would silently mean "every step") are rejected
  explicitly.

### Added — `add_msl_port(direction=...)` accepts the in-board axes (issue #661)

`direction` now takes `"+x"`, `"-x"`, `"+y"` and `"-y"`. A microstrip feed
entering a board along y no longer forces the whole model to be rotated —
which mattered because for a CAD-imported `MeshShape` that workaround does
not exist at all (the mesh import path takes no rotation argument).

`"+z"` / `"-z"` raise `ValueError` naming the reason. This is a documented
rejection, not an omission: the port's geometry contract is `position =
(x, y, z_lo)` plus a scalar `height`, which fixes the substrate normal to z.
A z-propagating microstrip needs its normal along x or y, and the normal
axis is what the static-Laplace cross-section solve, the `"ez"` source
component, the modal voltage `V = Σ Ez·dz` and the trace-conductor PEC scan
all reference. Accepting `"+z"` would have returned a z-normal answer for a
board lying in a different plane. `mode="eigenmode"` stays `"+x"`/`"-x"`
only (its Schelkunoff J+M launch hard-codes the x-axis TFSF correction pair
and rides on a solver that is a fenced dead-end), and the EXPERIMENTAL
`compute_mixed_s_matrix` lane (#488) is fenced to x rather than extended
untested.

**The axis inventory was measured, not read.** Instrumenting the live
extractor split it into stages bound to the *propagation* axis (probe
ladder, DFT plane normal, the `dx_feed` in the port conductance, the
port-to-CPML and probe-span preflight checks — all relabelings), stages
welded to the *substrate normal* (the four listed above), and stages that
are genuinely axis-free (the N-probe fit, which consumes probe coordinates
and only their differences; the Hammerstad-Jensen anchor; the multi-drive
`S = B A⁻¹` solve).

**The dangerous part was the Ampère loop, and it fails silently.** The
closed-contour current needs the right-handed transverse pair `(a, b)` with
`â × b̂ = p̂`. Deriving it as a plain `x ↔ y` rename is a reflection, not a
rotation. Measured on the committed thru fixture's own recorded H planes:
the cyclic pair reproduces the x-port current to `9.2e-08` relative, the
naive swap returns exactly `−I` (ratio `−1.00000009`). That flip exchanges
`a` and `b`, mapping `S = B A⁻¹` to `A B⁻¹ = S⁻¹`; for the low-loss matched
line every MSL fixture uses, `S` is nearly unitary, so `S⁻¹ ≈ S†` — `|S11|`
moves only `0.17905 → 0.17875`, `max ||S| − |S_swapped||` is `1.3e-03`,
column power stays under 1, and `cond(A)` is `1.32`. **No guard in the lane
fires.** What actually changes is `arg(S21)`, which is exactly negated (the
two angles sum to ≤ 0.02° across the band — a negative group delay); the
complex error is `max |S − S_swapped| = 1.912`. The equivalence test
therefore compares complex `S`; a magnitude-only comparison would have
passed on this exact bug.

**Falsifier — a y-directed port on the x↔y mirrored fixture reproduces the
x-directed result** (12 bins, `num_periods=12`, dx=80 µm, CPU float32):

| quantity | agreement |
|---|---|
| `max \|S_x − S_y\|` | `3.86e-06` |
| rel `max \|Z0_x − Z0_y\|` | `2.14e-04` |
| rel `max \|β_x − β_y\|` | `3.09e-05` |
| `settling_db` | x `[−98.47, −102.96]`, y `[−98.49, −102.91]` |

Both lanes emit the same advisory set. The three tolerances differ because
the quantities differ in conditioning, not by choice: `S` is a well-
conditioned V·I split plus a 2×2 solve and sits at the float32 floor, while
`β` and `Z0` ride on the N-probe least-squares fit that
`rfx/probes/msl_wave_decomp.py` runs in float32 by explicit cast, with
`Z0 = (α − γ)/I` compounding that fit's residual through a difference.

**Existing x-directed behaviour is byte-identical.** SHA-256 over the
extracted `S`, `Z0` and `β` on the committed thru fixture
(`test_msl_thru_line_passive_gate` geometry, `n_freqs=30`,
`num_periods=12`), base `fd37c62` vs branch:

```
S     eb69f37dcf72a8fcd88a532a9607ff4e20a72199aecfadc2538b37a30f0592b6
Z0    7873d42cc7c2668d32e7cfc05afce28ddcb8b48c91465faad4e1c215dc49b64c
beta  529b08fa31f1255b6e63ad4599e7f972444c0167eca43bebde4b73ada5761277
```

identical on both trees, reproducing the recorded calibration
(mean |S11| 0.115946, mean |S21| 0.993048, mean Re(Z0) 57.5710 Ω). A
negative control — the trace width nudged by one part in 10⁴ — moves all
three digests, so the lock is sensitive rather than vacuous. No committed
gate or tolerance was changed; the one contract assertion that moved is
`tests/test_msl_port.py`, which pinned `"+y"` as invalid — the limitation
this issue removes.

Note for anisotropic meshes: the port conductance scales as `1/d_prop`, so
on a cubic grid an x-flavoured σ applied to a y port is bit-identical to
the correct one. The end-to-end rotation-equivalence test runs on a cubic
mesh and is therefore blind to that stage;
`test_port_conductance_uses_the_propagation_axis_cell_size` covers it on a
deliberately anisotropic grid.

**Sign-convention prose corrected along the way (see #524).** Generalising
the `+x`/`-x` current convention to four directions meant resolving whether
the contradiction #524 records is real in the code or only in the prose.
Measured: it is prose only, and the `msl_loop_current` docstring is the
wrong statement. On the committed thru fixture, each port on its own drive,
`Re((α − γ)/I1)` reads **+57.52 Ω** at the `"+x"` port and **−57.56 Ω** at
the `"-x"` port — same magnitude to 0.08 %, opposite sign. So the `#140`
`dir_sign` comment described the code accurately, while the docstring's
unqualified "the returned `I` is positive for a forward quasi-TEM wave"
holds only for a positive-going port. The lane is self-consistent about it:
`dir_sign` restores both reported `Z0` to positive, the wave split consumes
the un-normalised current at every port, and the shipped `S` is physical
(`|S11| = |S22|` to 5 decimals, reciprocity `max ||S21| − |S12|| = 1.27e-05`,
column power ≤ 0.99998). The docstring is corrected and
`test_loop_current_negates_on_the_direction_sign_only` pins what it now
says. No code changed for this. #524 stays open for its other two items —
the passive port's ~30 Ω termination reading and the 0.194-vs-0.073 drive
asymmetry — which this work does not touch.
### Fixed — two S-parameter lanes computed the ring-down settling witness and never compared it to its own -40 dB bar (issue #662)

`settling_db` is the repo's mechanical form of the -40 dB ring-down settling
rule: above that line the fixed-length record ended while the structure was
still ringing, so the single-bin DFTs behind every S value of that run
integrate a cut transient. `compute_coaxial_two_port` and
`compute_coax_msl_transition` computed the witness, documented the -40 dB bar
in their own result docstrings, and never made the comparison — a caller
reading `.s_params` got a plausible-looking truncation artifact in silence.
Both now route the witness through the same `_warn_if_ringdown_truncated`
warner the waveguide/MSL/mixed lanes already used, so all five lanes share one
threshold constant and one warning shape.

Measured on the committed coax through-line fixture (domain 8x8x60 mm,
`freq_max` 40 GHz), before the fix:

| `n_steps` | `settling_db` (dB) | warnings emitted |
|---|---|---|
| 400 | -6.84 / -6.93 | 0 |
| 700 | -28.15 / -29.46 | 0 |
| 1500 | -43.97 / -44.53 | 0 |
| 3000 | -67.26 / -68.09 | 0 |

The first two rows violate the bar by 33 and 12 dB. They now warn, naming
every violating drive, its measured value, and the record-length knob that
lane is actually driven by (`n_steps` for the coax lanes, `num_periods` for
the others).

No extracted number changes: SHA-256 over `S`/`Z0`/`beta` (MSL fixture) and
over `s_params`/`gamma`/`cond_a`/`recurrence_residual`/`fit_residual` (coax
fixture) are identical before and after, including on the run where the new
warning fires. The differentiable (`eps_scale`) channel leaves `settling_db`
`nan` by design — the witness needs a concrete time series — and stays
silent; the warner's finite mask, not luck, is what keeps it that way.

### Fixed — a material flush with the domain face left a one-cell vacuum gap before its CPML pad (issue #655)

A `Box` (or any shape) whose hi face lands on the domain's last interior node
lost exactly that node from its rasterized mask. #627a taught the hi-face CPML
pad to source its material from one column further in when that happens, but
wrote the material only to the *pad* — so the dropped node stayed vacuum and
became a spurious one-cell film sandwiched between the structure and its own
impedance-matched absorber:

```
[pad = material][... interior = material ...][ONE VACUUM CELL][pad = material]
```

This affected **non-dispersive materials** on the shipped path, every hi face
(x/y/z), the non-uniform lane, and `Sphere`/`Cylinder` as well as `Box` — it hit
any geometry drawn out to an absorbing face, which is the ordinary way to build
a substrate, a waveguide fill or a half-space.

**Measured reflection off the phantom film** (1-D plane-wave FDTD, `eps_r=4`
filling the domain, periodic transverse, CPML on z; isolated by field-level DFT
subtraction against the same fixture with the box drawn half a cell past the
face, so the pad is identical and only the one node differs):

| dx | cells/λ₀ @10 GHz | \|r\| (flux DFT) | \|r\| (probe FFT) | thin-film theory |
|---|---|---|---|---|
| 0.25 mm | 120 | 0.0321 | 0.0385 | 0.0393 |
| 0.50 mm | 60 | 0.0830 | 0.0792 | 0.0786 |
| 1.00 mm | 30 | 0.1914 | 0.1570 | 0.1572 |
| **1.50 mm** | **20 (rfx's default `c0/freq_max/20`)** | **0.2377** | 0.2644 | 0.2358 |

At the default mesh that is **−12.5 dB, 5.6 % of incident power**, off a film
that should not exist. The error grows as the mesh gets *coarser*, the opposite
of the direction a convergence check looks in — a fine-mesh-only test would
understate it ~6x. Theory is
`|r| = 2π(dx/λ₀)(ε_m−1)/(2√ε_m)` for a one-cell vacuum layer.

**Fix**, in the shared `rfx.geometry.rasterize_grid.extend_cpml_pad_materials`
(so the uniform, non-uniform and batched assemblers all get it at once): where
the #627a fallback fires, the replicated value is written to the outermost
interior column as well as to the pad. That is precisely what makes the hi face
behave like the lo face, where `_extend_lo` replicates the boundary node itself
and pad and boundary node cannot disagree. **The rasterizer is untouched** — the
half-open `[lo, hi)` convention is deliberate and load-bearing (see the `Box`
docstring), and the defect is not Box-specific anyway.

The one-column bound from #627a is inherited unchanged, so a genuine multi-cell
air gap before the absorber is still never bridged. `run()` is **byte-identical**
for geometry that does not touch a face (SHA-256 over raw field bytes across
PEC / CPML / lossy+CPML / mu_r+CPML / non-uniform+CPML / periodic-mixed).
After the fix, drawing a box flush with the face and drawing it half a cell past
the face produce bit-identical runs.

Regression locks: `tests/test_cpml_pad_face_notch.py` (11 tests, pinned at the
default mesh; 9 of them verified red on the unfixed tree, the other 2 being the
paired must-still-be-vacuum over-fire guards).

### Fixed — thin conductors leaked into the batched sweep's CPML pad (issue #642)

`Simulation._assemble_materials` extends material values into the CPML pad
and only THEN applies thin conductors, so `Simulation.run()`'s padding
carries the background material and never a conductor. `vmap_material_sweep`
was handed `base_materials` — the finished, post-conductor arrays — and
re-extended those per swept value, replicating a non-PEC conductor's own
`eps_r`/`sigma` outward into a pad `run()` never builds.

#643 could not close this by making the extension rule more faithful,
because the rule was never what was wrong: the batched path was given the
wrong INPUT, not running the wrong algorithm.

**Fix: the batched path is handed pre-conductor materials, and re-applies
the conductors itself.** `Simulation._assemble_materials` gained a
keyword-only `include_thin_conductors` (default `True`, and no caller other
than `rfx/vmap_sweep.py` passes it, so every existing path is unaffected by
construction). `_build_batched_materials`' material-named branch now
re-derives the pre-conductor arrays through it, builds the sweep and the pad
from those, and then re-applies the same shared
`rfx.materials.thin_conductor.apply_thin_conductor` under `jax.vmap` — the
package's single conductor rule, in the same order `run()` runs it, rather
than a second copy of it. Production assembly ORDER is untouched; only what
the batched path is given changed. The extra assembly is paid only when the
simulation declares a thin conductor.

- **0 mismatched cells, both material lanes, all three swept fields.**
  Batched element *b* vs `_assemble_materials` for a simulation carrying
  that swept value, on a `sigma_bulk=1e4` sheet spanning the domain's full
  x extent (12167 cells per array, 36501 across the three). Per swept
  element, main → this branch: `slab.eps_r` sweep 88 (40 `eps_r` + 48
  `sigma`) → **0**; `slab.sigma` sweep 88 (48 + 40) → **0**; `slab.mu_r`
  sweep 96 (48 + 48) → **0**. The `sigma` and `mu_r` sweeps were not in the
  issue's own measurement and leak the same way — the leak is the conductor
  being replicated outward, so it does not depend on which field is swept.
  A PEC thin conductor measures 0 on both trees for all three (it routes to
  `pec_mask`, not to the material arrays) and is committed as the must-pass
  companion row, not as filler.
- **This was not only an accuracy leak — it destabilised the run.** R5
  decile dump of the probe envelope at `n_steps=400` with the pre-#642 pad
  order restored: the batched path grows monotonically from decile 5
  (`3.6e-9, 3.0e-8, 4.4e-8, 9.7e-8, 8.8e-7, 2.9e-4, 9.8e-2, 3.3e+01,
  1.1e+04, 3.8e+06`, last/mid ratio 3.9e13) while the sequential reference
  decays (last/mid 9.1e-3). Fixed, the batched path's deciles match
  `run()`'s to every printed digit. Mechanism: a 175 S/m sheet
  (`sigma_eff = sigma_bulk * thickness / dx`) replicated into the CPML pad,
  on top of the absorber's own loss.
- **`Simulation.run()` does not move.** SHA-256 over the raw bytes of the
  final Yee field state plus the probe time series, base (`ef8a008`) vs
  branch, on four lanes × with/without a non-PEC thin conductor — pec
  (`6615242190bedfc9` / `d5a1a4614c419d12`), cpml (`a3ebcc1203fb43b1` /
  `6cc2b67523278d4b`), lossy+CPML (`a52029cec52bca18` /
  `97fd6ff74795c881`), non-uniform `dz_profile`+CPML (`9a79f62121917b85` /
  `5a463547b098b69c`) — all 16 digests (fields + time series per row)
  identical. Two negative controls: nudging `slab` `eps_r` 4.0 → 4.001
  moves the digest, and forcing `include_thin_conductors=False` through
  `run()`'s own assembly moves exactly the thin-conductor rows and leaves
  the conductor-free rows untouched, so the harness is sensitive to the
  edited line specifically and not merely to something.
- **The `xfail(strict=True)` witness is now a normal passing test.**
  `test_thin_conductor_pad_leak_is_issue_642_not_643` was verified XFAIL on
  base, then XPASS(strict) — i.e. red — with the fix and the marker still
  in place, and only then flipped. It is now
  `test_thin_conductor_pad_matches_run`, parametrized over PEC/non-PEC
  conductor × three swept fields, plus
  `test_thin_conductor_fixture_is_live`: a non-vacuity control asserting
  that the conductor really does sit on the column the pad replicates FROM
  (so the old code had something to replicate) and that the pad carries the
  background material anyway (so the new code does not replicate it).
- **Not reached by this defect:** the global (unnamed-material) sweep,
  which does no pad extension; and the non-uniform mesh path, where a
  `dz_profile` makes the simulation ineligible for the vmap fast path
  entirely (`_build_full_scan_fn` returns `None`) so the sequential
  fallback — `run()` per value — is exact by construction.

### Fixed — a sweep-parity test that could not see the defect it covers (issue #642b)

`TestVmapSweepCPML::test_cpml_vmap_matches_sequential` ran at `n_steps=30`
with `atol=1e-5, rtol=1e-4` and stayed green through the entire pre-#637
era for a reason unrelated to the physics it names: the pulse had not
meaningfully reached the absorber, so no pad defect could show. Both knobs
are now measured rather than chosen, by injecting each defect and sweeping
`n_steps`, reporting `max|diff| / (atol + rtol*|reference|)` — the quantity
the assertion actually tests, so >1 means it fails.

- **Run length.** With `_extend_batched_cpml_pad` disabled (the pre-#637
  defect), under the OLD gate: `0.098` at 30 steps, `0.337` at 45, `1.010`
  at 60, `7.218` at 90, `26.77` at 120, `60.77` at 200. It first crosses at
  60 — by 1%, which cross-machine float jitter can erase, so 60 is not a
  safe pick even though it nominally "sees it".
- **Tolerance is an independent axis, not cosmetic.** The tighter gate
  catches that same defect at `97x` at the ORIGINAL `n_steps=30`, where the
  old gate saw `0.098x`. Both are moved because run length is what lets the
  physics reach the absorber, while the tolerance is what a later
  flake-chase is most likely to loosen; either alone leaves the other blind.
- **`n_steps=200`, `atol=1e-8`, `rtol=1e-6`** is the only setting at which
  BOTH parametrized rows are sensitive: `1.47e4x` over the gate for the
  plain row's defect and `5.10e2x` for the thin-conductor row's (which
  measures `0.937` — blind — at 120 AND at 150, then `5.10e2` at 200 and
  `1.07e9` at 300; the jump is the divergence onset above). The clean tree
  measures exactly `0.0` on both rows at every step count tried, so neither
  knob costs anything.
- **A third blindness shape, found while fixing the second.** The
  thin-conductor fixture must span the full x extent: #642 is a
  pad-EXTENSION leak, so the conductor has to sit on the column the x pads
  replicate from. The first draft placed it interior (x 0.005–0.015) and
  measured `7.5e-3` even at `n_steps=200` — blind by GEOMETRY, on top of
  the "blind by run length" and "compared quantity cannot differ" shapes
  the issue names. Recorded in the fixture so it is not undone.

### Fixed — two independent implementations of one CPML pad-extension rule, reconciled (issue #643)

#627 (PR #638) and #637 (PR #641) each needed material values extended
from the interior edge into the CPML pad, and each landed its own
implementation. #627 then changed the rule — adding a per-transverse-cell
hi-face fallback (if the naive interior-edge column is vacuum but the
column one further in is not, replicate from that inner column instead,
recovering the node a `Box` flush with the domain's hi face loses to
`Box`'s half-open `[lo, hi)` rasterization) — while `_extend_batched_cpml_pad`
in `rfx/vmap_sweep.py` went on reproducing the PRE-#627 rule. For a slab
sitting exactly on a domain face, `Simulation.run()` read the material in
its hi pads and `vmap_material_sweep` read vacuum.

**Fix: the second copy is gone, not resynced.** `_extend_batched_cpml_pad`
now calls the package's single shared rule,
`rfx.geometry.rasterize_grid.extend_cpml_pad_materials`, under
`jax.vmap` over the sweep axis. That is what makes reuse possible at all:
the shared helper slices axes 0/1/2 and evaluates its vacuum predicate on
whole transverse planes, while the batched arrays carry a leading sweep
axis whose elements can answer that predicate differently; mapping over
that axis hides it from the helper, so the helper sees exactly the
`(Nx, Ny, Nz)` arrays it was written for and its fallback is evaluated
against each element's own materials. No axis-aware rewrite, no third
implementation. All three material arrays are now extended in ONE call
(the helper takes a single `use_inner` decision from the joint vacuum
predicate and applies it to all three, so a swept value that empties a
column changes that element's `sigma`/`mu_r` pad too — a coupling the
per-field version could not express). The no-worse guard #641 added is
deleted along with the copy it was guarding: it existed only because that
copy could discard a value #627 had already got right, and there is no
longer a second copy to do so.

- **Acceptance criterion met.** Byte-identity between what the batched
  path builds for element *b* and what `Simulation._assemble_materials`
  builds for a simulation carrying that swept value, across #637's
  geometry/boundary matrix plus the exact-hi-face case: single face, all
  six faces (exact-hi AND past-hi), corner, inset, transverse span,
  `cpml_layers=0`, pec, upml, periodic-x with CPML y/z, two materials
  (the second lossy AND magnetic, the only shape that can distinguish
  the joint vacuum predicate from a per-field one — #637 noted none of
  its fixtures could), and asymmetric per-face pads. 12 rows x 3 swept
  fields (`eps_r`, `sigma`, `mu_r`) = 36 comparisons, all three arrays
  each, 0 mismatched cells. On main, 18 of those 36 fail. Committed as
  `TestVmapBatchedPadByteIdentity`.
- **The `xfail(strict=True)` witness is now a normal passing test.**
  `test_exact_hi_face_touch_matches_run_via_shared_fallback` was verified
  XFAIL on base and XPASS(strict) — i.e. red — with the fix, before the
  marker was removed. Its DFT-plane comparison against `run()` is exactly
  `0.0` at every bin for both swept elements, versus the 1.66e-2 /
  4.25e-2 #637 measured. It gained an assert-first non-vacuity
  precondition (run()'s x-hi pad must read the slab, the naive source
  column must read vacuum) so it cannot later pass by no longer
  exercising the fallback — the failure mode this same arc already hit
  once, when the alternate-geometry test's first draft used a fixture
  that never reached the pad.
- **#637's 7-configuration representativeness sweep, re-run per swept
  element (14 elements), three states.** #637's harness is not committed
  anywhere in the repo, so these fixtures are RECONSTRUCTED from the #637
  issue body (30x20x20 mm, dx=1 mm, `cpml_layers=8`, 200 steps, base 2.0
  / 10.0, sweep {2.0, 10.0}) and PR #641's table; the absolute magnitudes
  below are one to two orders below #641's, so this is a new sweep on the
  same classes, not a reproduction of those numbers. Worst per-bin
  DFT-plane relative error against `run()`:

  | configuration | swept | pre-#637 | main | this branch |
  |---|---|---|---|---|
  | issue-table base=2.0 | 2.0 (=base) | 0.0 | 0.0 | **0.0** |
  | issue-table base=2.0 | 10.0 | 7.7037e-04 | 7.5970e-04 | **0.0** |
  | issue-table base=10.0 | 2.0 | 3.1653e-04 | 2.9872e-04 | **0.0** |
  | issue-table base=10.0 | 10.0 (=base) | 0.0 | 0.0 | **0.0** |
  | single-axis touch (y), base=3 | 2.0 | 3.1527e-04 | 2.8198e-04 | **0.0** |
  | single-axis touch (y), base=3 | 6.0 | 7.2871e-04 | 7.3497e-04 | **0.0** |
  | single-face touch (x_lo), base=5 | 2.5 | 2.2078e-06 | 0.0 | **0.0** |
  | single-face touch (x_lo), base=5 | 9.0 | 3.8385e-06 | 0.0 | **0.0** |
  | large delta, cpml=4, base=1.5 | 1.5 (=base) | 0.0 | 0.0 | **0.0** |
  | large delta, cpml=4, base=1.5 | 12.0 | 1.4852e-02 | 1.1973e-02 | **0.0** |
  | large delta, cpml=12, base=8 | 2.0 | 2.5857e-03 | 2.3729e-03 | **0.0** |
  | large delta, cpml=12, base=8 | 14.0 | 5.3260e-03 | 4.1713e-03 | **0.0** |
  | committed fixture, base=4, cpml=6 | 2.0 | 1.3487e-03 | 1.2792e-03 | **0.0** |
  | committed fixture, base=4, cpml=6 | 6.0 | 1.9302e-03 | 1.5955e-03 | **0.0** |

  Every element is exactly `0.0` with 0 mismatched material cells — not
  "at or below a floor", bit-identical. No element is worse than either
  prior state.

- **#637's two disclosed residuals, explained and eliminated.** #641
  shipped its guard with two elements measurably WORSE than the pre-#637
  state (6.421e-2 -> 6.527e-2, +1.6%; 1.193e-2 -> 1.199e-2, +0.5%),
  reproducible but with "their specific mechanism not established". They
  are not a separate effect. Running one fixture through four batched-pad
  states in the same tree (so no cross-tree float path is involved),
  reporting BOTH the wrong-cell count against `_assemble_materials` and
  the DFT-plane error. Comparator check first, since two of those states
  are re-implementations: the `pre-#637` and `main (guard)` rows below
  reproduce the real base-tree runs bit-for-bit (1920 cells /
  1.3487e-03 / 1.9302e-03 and 1140 cells / 1.2792e-03 / 1.5955e-03
  respectively, identical to the same fixture measured on an actual
  checkout of each state), so the emulation is faithful and not an
  artefact of the harness:

  | fixture / element | state | wrong cells | worst rel err |
  |---|---|---|---|
  | committed fixture, swept 2.0 | pre-#637 | 1920 / 12167 | 1.3487e-03 |
  | | main (guard) | 1140 | 1.2792e-03 |
  | | #641 draft, no guard | 1140 | 1.8661e-03 |
  | | this branch | **0** | **0.0** |
  | committed fixture, swept 6.0 | pre-#637 | 1920 | 1.9302e-03 |
  | | main (guard) | 1140 | 1.5955e-03 |
  | | #641 draft, no guard | 1140 | 8.4886e-03 |
  | | this branch | **0** | **0.0** |
  | issue-table base=2.0, swept 2.0 (=base) | pre-#637 | 0 | 0.0 |
  | | main (guard) | 0 | 0.0 |
  | | #641 draft, no guard | 5120 | 2.0642e-04 |
  | | this branch | **0** | **0.0** |

  Two things fall out. The guard did what it was designed to (the
  `swept == base` element: 5120 wrong cells and 2.06e-4 without it, 0 and
  0.0 with it). And the metric is not monotone in the wrong-cell count:
  main and the no-guard draft have the SAME 1140 wrong cells and differ
  by 5.3x in error, while pre-#637 has 68% MORE wrong cells than main
  and a smaller error on one element of this class. Mechanism: pre-#637
  the whole pad was uniformly wrong (base's value); #637 corrects the
  normal pad cells but cannot correct the hi-face-fallback cells, so it
  leaves a permittivity STEP inside the absorber that the uniformly-wrong
  pad did not have, and a step inside a CPML reflects. Whether that trade
  reads better or worse than uniformly-wrong is fixture-dependent and has
  no reason to be monotone — this branch's own reconstruction shows both
  signs (of 14 elements, pre-#637 -> main: 10 improve, 3 unchanged at
  exactly 0.0 — all three are `swept == base` — and 1 regresses,
  single-axis-y swept 6.0 at 7.2871e-04 -> 7.3497e-04, +0.86%, the same
  signature as #641's +0.5% / +1.6%). No third guard
  variant was needed: byte-identity removes the choice, and both residual
  elements' *class* is covered directly — on the committed fixture at
  n_steps 300 / 400 / 600, where main measures 1.45e-2/1.61e-2,
  3.52e-2/7.94e-2 and 1.69e-2/2.21e-2 (bracketing both disclosed
  residuals), this branch measures exactly `0.0` at every bin.
- **`Simulation.run()` does not move.** SHA-256 over the raw bytes of all
  six final Yee field components plus the probe time series, base vs
  branch, on four production lanes — pec (`e4dbb5fa327093d4`), cpml
  (`a3a6bc1d733b36e7`), lossy+magnetic+CPML (`02bdd2db2572df1b`),
  non-uniform `dz_profile`+CPML (`ef96cbf44e9aa14f`) — all 8 digests
  (fields + time series per lane) identical. Negative control: bumping
  `n_steps` 60 -> 61 changes all 8, so the harness can report
  "different".
- **Known gap this INHERITS and slightly widens, disclosed (issue #642).**
  `run()` applies thin conductors AFTER pad extension; the batched path
  only ever sees `base_materials`, which is already post-conductor, so
  extending it replicates a non-PEC thin conductor's own `sigma`/`eps_r`
  into the pad. Routing through the shared helper does not change that
  ordering. Measured on a `sigma_bulk=1e4` thin conductor: this branch
  leaves 88 mismatched cells of 36501 (40 `eps_r` + 48 `sigma`, all
  inside the x pads) where main leaves 6592 (all `eps_r`; zero `sigma`,
  because main only ever extended the ONE swept field) — a 75x reduction
  overall AND a new sigma-side mismatch on this fixture. A PEC thin
  conductor is unaffected (it routes to `pec_mask`, not to the material
  arrays: 0 mismatched cells on both). Not fixed here: #643's scope is
  the pad-extension RULE, #642's is the pipeline ORDER, and closing it
  needs the batched path to see pre-conductor materials, which is a
  production-side change this issue's own byte-identity requirement
  argues against bundling. Pinned by
  `test_thin_conductor_pad_leak_is_issue_642_not_643`, `xfail(strict=True)`.
  **CLOSED since** — see the #642 entry above; the witness went XPASS(strict)
  and is now a normal test under a new name.
- Memory footprint note: the material-named branch now materialises all
  three `(n_batch, Nx, Ny, Nz)` arrays instead of one plus two broadcast
  views, because the joint pad decision needs all three. Materials go from
  ~1 to ~3 batched arrays against the ~6 the batched field state already
  carries. Unchanged for the global sweep path and for any simulation with
  no CPML pad on any face (early return, inputs untouched).

### Fixed — `precision="mixed"` never worked with CPML boundaries (issue #644)

`Simulation(boundary="cpml", precision="mixed").run(...)` raised a raw JAX
`TypeError` ("scan body function carry input and carry output must have equal
types ... `psi_hz_yhi` has type float16[...] but the corresponding output
component has type float32[...]"). Pre-existing on every released version, and
measured identical on the trees before and after the issue #630 compute-dtype
work, so #630 neither caused nor fixed it.

- **Mechanism.** `rfx/boundaries/cpml.py`'s `psi_*` auxiliary arrays were
  allocated at the field storage dtype (float16 under `precision="mixed"`),
  while the CPML profile coefficients (`b`/`c`/`kappa`) are hard-pinned
  float32. `psi = b*psi + c*curl` therefore evaluated to float32 while the
  `lax.scan` carry had been declared float16, and the carry signature stopped
  matching its own input. Loud failure, never a silent wrong number.
- **Fix.** The `psi_*` arrays are *accumulation* state — a recursive
  convolution integrated over every timestep — so they are now allocated at
  `jnp.promote_types(field_dtype, jnp.float32)` and never sit below float32.
  This is deliberately NOT a flat float32 pin: `psi` followed `field_dtype`
  for a reason (the issue #404 oblique-Bloch path needs a *complex* carry),
  and a flat pin would have fixed float16 by silently breaking complex64. The
  promotion satisfies every caller at once — float16 -> float32 (the fix),
  float32 -> float32 (unchanged), float64 -> float64, complex64 -> complex64.
  It is the same idiom already used by `rlc_carry_dtype` in `rfx/lumped.py`
  and by #630's `_cdtype`.
- **Second, latent defect fixed in the same path.** With float32 corrections
  scattering into float16 fields, `apply_cpml_e`/`apply_cpml_h` tripped JAX's
  "cannot safely cast value from dtype=float32 to dtype=float16" *FutureWarning*
  — which JAX states "will result in an error" in a future release. Both
  functions now accumulate at the psi dtype and round back to storage dtype
  once at the end, instead of rounding at each of the four `.at[].add()` per
  component. Measured no-op for float32/float64/complex64 (`promote_types` is
  idempotent there). **This is deprecation-driven, not an accuracy
  improvement, and is not claimed as one**: it moves the mixed+CPML absorption
  floor from -59.5 dB to -59.7 dB, which is noise — the float16 *storage*
  quantization dominates, not the per-add rounding. The justification stands
  on its own: without it the #644 fix expires the moment JAX turns that
  warning into an error.
- **The `forward()` guard from #630 is removed.** #630 shipped a temporary
  `NotImplementedError` in `_forward_from_materials` for `mixed` + CPML rather
  than leak the raw JAX carry-dtype `TypeError` through the newly-`precision`-
  aware `forward()`. The root cause is now fixed, so the guard is gone —
  leaving it would have kept `mixed` + CPML blocked through `forward()`
  forever. Its two regression pins in `tests/test_precision_lane_guard.py`
  are **kept and inverted to assert success** rather than deleted: they encode
  two genuinely distinct paths (a scalar `boundary="cpml"`, and a per-face
  `BoundarySpec` carrying only one cpml face), and both must keep working.
  Measured on the rebased tree: `run()` and `forward()` are now both green
  across all of pec/cpml/upml x float32/mixed.
- **Separate pre-existing defect uncovered by that guard removal — reported,
  NOT fixed here.** The per-face guard test was passing for the wrong reason:
  the guard raised before any CPML compute ran, so it never exercised the
  per-face CPML path. With the guard gone, that fixture (per-face spec, PEC on
  x/y, `cpml_layers=16` against unpadded 8-cell x/y axes) fails in
  `apply_cpml_e` with a SHAPE error — `mul got incompatible shapes for
  broadcasting: (16,1,1) vs (8,8,40)` — because the x/y face slices `[:n]`
  still run on axes that were never padded. MEASURED on the unpatched parent
  commit: identical failure at `precision="float32"` through both `run()` and
  `forward()`, so it is a per-face CPML geometry defect, precision-independent
  and unrelated to this issue. The condition is `cpml_layers` > the unpadded
  width of a PEC axis; the test now pins `cpml_layers=4`, which reaches the
  CPML compute and is red-then-green for the #644 dtype fix as intended.
- **The absorption caveat is documented where users will see it**: the
  `precision=` parameter docstring in `rfx/api/__init__.py` (which feeds the
  API reference) now states the measured -76 dB -> -59.5 dB floor shift, notes
  the numbers come from a specific fixture rather than being a universal
  bound, and says plainly that `"mixed"` is not a drop-in `"float32"`
  substitute for reflection coefficients, S-parameters, or anything near the
  absorber floor.
- **Production paths are bit-identical — verified as an identity claim, not a
  tolerance.** SHA-256 of the raw field bytes (all six components) matches
  between a `git archive` of the base commit and this branch across four
  fixtures: PEC, CPML, lossy-material+CPML, and non-uniform-mesh+CPML, plus
  `precision="mixed"`+PEC. The harness's sensitivity was demonstrated rather
  than assumed: a deliberate 2 ppm perturbation of the CPML `alpha` profile
  changes all three CPML fixture hashes and leaves both PEC hashes untouched
  (PEC never constructs CPML state at all).
- **Accuracy of `mixed` + CPML, measured — with a real cost worth knowing.**
  On a 40^3 domain (dx = 3.0 mm, dt = 5.72 ps), mixed-vs-float32 relative L2
  error on Ez is 0.20% at 50 steps and 0.23% at 100 steps. But float16 fields
  raise the **absorber's residual floor**: total field energy 400 steps after
  the pulse settles reaches -76.3 dB below peak in float32 and only -59.5 dB
  in mixed — roughly **17 dB worse absorption**, consistent with float16's
  ~9.8e-4 machine epsilon quantizing the field storage. `mixed` + CPML is
  correct and stable (no NaN/Inf out to 800 steps), but it is not a drop-in
  substitute for float32 when the quantity of interest is a low-level residual
  or a reflection coefficient near the absorber floor.
- **Test coverage — the reason this stayed invisible.** All 13 tests in
  `tests/test_mixed_precision.py` used `boundary="pec"`, and PEC never builds
  CPML state, so the CPML dtype path had zero coverage. The file was also
  entirely `pytestmark = pytest.mark.gpu`, which `pyproject.toml`'s
  `addopts = "-m 'not gpu and not slow and not slow_physics'"` deselects from
  every default run — two independent reasons the gap could not be seen. The
  file-level `gpu` mark is **removed** (measured: the original 13 tests run in
  6.5 s wall on CPU, slowest 1.27 s — these are dtype/dispatch assertions, not
  the "needs GPU / too slow on CPU" the marker is defined for), and CPML rows
  are added: psi dtype policy per field dtype, the full boundary x precision
  2x2 from the issue, mixed-vs-float32 agreement, an absorption witness in dB,
  and a regression pin for the unsafe-cast warning. 24 tests, 17.3 s on CPU.

### Fixed — Yee update arithmetic was hard-pinned to float32 regardless of field storage dtype (issue #630)

- `rfx/core/yee.py`'s `update_h`/`update_e` computed `_cdtype = jnp.complex64
  if jnp.iscomplexobj(state.ex) else jnp.float32` — the real (non-Bloch)
  path's Yee curl/update ARITHMETIC was hard-pinned to float32 for ANY
  field storage dtype. `_fdtype = state.ex.dtype` was used only to cast
  the *result* back, so a float64 field carry was silently re-quantized to
  float32 at the top of every timestep. The public API had no way to
  reach float64 storage anyway (`precision=` only accepted `"float32"` /
  `"mixed"`), so this was previously unobservable from outside — it
  surfaced only once #630's investigation forced `field_dtype=jnp.float64`
  directly at the `rfx/simulation.py` resolution site and found the FD
  side of FD-vs-AD gates still capped at a ~1e-7 relative noise floor with
  a non-monotone, sign-flipping sub-signal response (the AD side was
  unaffected — the floor only ever inflated the FD reference).
- Fix: `_cdtype` is now `jnp.promote_types(state.ex.dtype, jnp.float32)`
  at both `update_h`/`update_e` call sites, plus the same substitution at
  all 51 literal `.astype(jnp.float32)` field-arithmetic sites across the
  NU (`update_h_nu`/`update_e_nu`), GPU-fast (`update_h_fast`/
  `update_e_fast`/`update_he_fast`), and anisotropic
  (`update_e_nu_aniso`/`update_e_aniso_inv`/`update_e_aniso`) lanes.
  `jnp.promote_types(float16, float32) == float32` (preserves the
  existing mixed-precision upcast intent unchanged) and
  `jnp.promote_types(float32, float32) == float32` (byte-identical on the
  production path — measured across `g_ad` and every FD-ladder rung).
  Material coefficients (`eps_r`/`sigma`/`mu_r`) and the CPML profile
  arrays stay float32 either way — those are fixed setup-time constants
  that measurably bias the primal but do not create the per-timestep
  rounding-lattice noise floor the compute-dtype pin did.
- **New public knob**: `Simulation(..., precision="float64")` (alongside
  the existing `"float32"` / `"mixed"`) routes float64 field storage (and
  now, with the above fix, float64 Yee arithmetic) through both `run()`
  and `forward()`. Requires the caller to have already enabled JAX's x64
  mode (`jax.config.update("jax_enable_x64", True)` or
  `jax.experimental.enable_x64()`); `preflight()` now warns
  (`precision_float64_without_x64`) if `precision="float64"` is set
  without x64 enabled, since JAX otherwise silently downcasts back to
  float32 with no other signal. `forward()` previously never threaded
  `field_dtype` at all (always float32 regardless of `precision=`), so
  this also makes `precision="mixed"` reachable from `forward()` for the
  first time.
- **Follow-up (review), lane scope**: `field_dtype` is threaded only by
  `rfx/runners/uniform.py` — the non-uniform-mesh, distributed,
  distributed-NU, and subgridded runners do not thread it, so
  `precision="mixed"`/`"float64"` would silently run float32 fields there
  with no error. `_dispatch_plan` (the single lane-decision point for
  `run()`/`forward()`) now raises `NotImplementedError` for
  `precision != "float32"` on the `run_nonuniform`, `run_distributed`,
  `run_subgridded`, `fwd_nonuniform`, and `fwd_distributed_nu` lanes;
  `preflight()`'s `_validate_cfg_precision_x64` also warns in advance for
  the non-uniform-mesh case (the distributed case is a call-time
  `run()`/`forward()` kwarg invisible to preflight, so the dispatch-time
  raise is its only enforcement point). The `precision` docstring now
  says the knob is uniform-single-device-lane-only today. New committed
  regression coverage: `tests/test_precision_lane_guard.py`.
- **Follow-up (review), schema/spec/canonical consistency**: the
  `precision` enum in `docs/design_notes/schemas/rfx-experiment-v2.schema.json`
  (widened to accept `"float64"` alongside the compute-dtype fix) was out
  of sync with `rfx/experiments/spec.py`'s v1 validator and
  `rfx/experiments/canonical.py`'s v2 validator, both of which still
  rejected anything outside `{"float32", "mixed"}` — a config with
  `precision: float64` would pass schema validation and then die at
  runtime with a message that never mentioned `float64`. Both Python
  validators now accept `"float32"`/`"mixed"`/`"float64"` consistently
  with the schema; `tests/test_experiment_spec.py`'s pinned error-message
  regex updated to match.
- **Follow-up (review), issue #644 pre-existing defect newly reachable**:
  `precision="mixed"` (float16 fields) with a CPML boundary face has
  always crashed with a raw JAX `TypeError` (`lax.scan` carry dtype
  mismatch: the CPML `psi_*` carry follows `field_dtype`, but the CPML
  coefficients are hard-pinned to float32) — pre-existing on `run()`,
  measured identical on the trees before and after the compute-dtype fix
  above, and NOT caused by it. Because that fix makes `forward()` honour
  `precision=` for the first time (see above), `forward()` newly reaches
  this same crash, where it used to silently no-op to float32 instead.
  `forward()` now raises a clear `NotImplementedError` pointing at #644
  for this specific combination instead of leaking the raw JAX error;
  `precision="float32"`/`"float64"` and `boundary="upml"` are unaffected
  (measured: UPML has no analogous hard-float32 psi carry, and
  `jnp.promote_types(float64, float32) == float64` keeps the CPML carry
  dtype-consistent for `"float64"`). Fixing #644 itself (giving `psi_*`
  and the CPML coefficients one consistent dtype policy) is out of scope
  here — tracked separately as issue #644.
- **BEHAVIOUR CHANGE, opt-in only**: nothing changes for the default
  `precision="float32"` or `precision="mixed"` paths (verified
  byte-identical: primal, AD gradient, and every FD-ladder rung
  unchanged). Only simulations that explicitly opt into
  `precision="float64"` with x64 enabled see different (more accurate)
  numbers.
- Falsifier: reverting only the two `_cdtype` sites while keeping all 51
  literal-site changes reproduces the stock (unpatched) float64-fields
  result exactly — confirming `_cdtype` is the entire load-bearing change
  and the literal-site changes alone are inert on the executed uniform CPU
  lane (`update_he_fast` is GPU-only; the literals there and in the NU/
  aniso lanes are unreachable from the CPU test suite either way).
- No existing gate/tolerance changed; this PR does not tighten anything.

### Fixed — `vmap_material_sweep` didn't sweep the CPML absorber for material-named sweeps (issue #637)

Found by an independent audit of the e4b565c..ce44661 arc: for a material-
**named** sweep (`"substrate.eps_r"`, as opposed to a global `"eps_r"`
sweep), `rfx/vmap_sweep.py`'s `_build_batched_materials` built its sweep
mask from `Shape.mask(grid)` — physical-domain geometry cells only — and
kept `base_materials` (the BASE simulation's own CPML-padded arrays)
everywhere else, including the padding. `Simulation._assemble_materials`
replicates each material's value into the CPML padding it touches
(so the absorber stays impedance-matched); a material-named sweep never
reached that replicated padding, so every swept batch element ran with
an absorber matched to the *base* material instead of its own. Any
substrate/dielectric running out to a CPML face — ordinary microwave
geometry — was affected. Measured on the committed test fixture
(box spanning the full transverse domain extent, `cpml_layers=6`, base
`eps_r=4.0`, sweep `{2.0, 6.0}`): originally 780 of 12167 cells held the
wrong `eps_r` (re-measured at 2040/12167 after this branch's rebase onto
issue #627, which independently fixed a related hi-face pad gap and, in
doing so, corrected the reference `run()` used for this count too — see
"Overlap with issue #627" below); worst per-bin DFT-plane relative error
against `sim.run()` was 5.281e-4 (8.83e-4 on the rebased fixture), which
the previous `rtol=2e-3` gate absorbed and two test
docstrings misattributed to "float32 accumulation roundoff" — the
decisive discriminator (moving the same slab off the CPML faces so no
material lands in the padding) dropped the identical comparison to
exactly `0.0`, not merely smaller, which a genuine roundoff floor would
not do. The error scaled with `|swept - base|`: up to ~6% relative on a
harder edge-touching fixture, six orders above the old gate's nominal
floor.

Fix: a new `_extend_batched_cpml_pad` helper re-runs
`_assemble_materials`'s own per-face edge-slice-copy padding rule on the
already batch-correct interior (after the sweep mask is applied), so each
swept batch element's padding matches what `Simulation.run()` would build
for that value — a no-op on any face with zero CPML depth, so it is safe
unconditionally. The GLOBAL (`"eps_r"`, no material name) sweep path was
already accidentally correct (`non_vac = eps_r != 1.0` selects the
replicated padding too, since it is evaluated on the already-padded base
array) and is untouched; a new regression test pins that it stays
correct. Verified post-fix on the original fixture: the edge-touching
DFT-plane comparison against `run()` is exactly bit-identical (`0.0`),
matching an inset (non-edge-touching) control that was already exact —
stronger than the falsifier's own bar of "toward the inset floor" and,
more generally, evidence that the vmap fast-path scan body reproduces
`Simulation.run()` bit-for-bit whenever the two are actually handed the
same materials (the padding mismatch, not independent scan-body
roundoff, was the entire pre-fix gap). Re-measured across a
representativeness sweep varying `|swept-base|`, which CPML face(s) the
material touches, and `cpml_layers` (7 structurally distinct
configurations, including two directly from the issue's own harder
table): every configuration lands at or below ~4e-7 post-fix, bit-identical
(`0.0`) on several including the committed fixture — versus 4.2e-3 to
5.8e-2 pre-fix — four to five orders of magnitude tighter, none of it
fitted to any one fixture.
`tests/test_vmap_sweep_dft_planes.py` gates tightened from `rtol=2e-3` to
`rtol=1e-6` (anchored near the measured float32/x64 floor with margin for
cross-machine floating-point jitter) and gained three new tests: a
mechanism-level pin comparing the batched material arrays directly
against `Simulation._assemble_materials`, a second DFT-plane equivalence
test on a structurally different geometry (touches the x_lo face instead
of y/z, different domain/dx/cpml_layers/delta — its first draft touched
the x_hi face at the exact domain-edge coordinate instead and turned out
to be vacuous, since `Box.mask` is inclusive on a shape's `lo` corner but
not its `hi` corner there; documented in the test's own docstring as a
falsifier-of-the-test finding), and the global-sweep regression pin
above. Two docstrings that attributed the defect's symptom to float32
roundoff are corrected (module docstring and the x64 class docstring); a
third (`TestVmapAmplitudeKindCurrent`) is partially corrected — its own
floor also included this defect (6.88e-5 pre-fix, 1.74e-7 post-fix, a
~396x drop, numbers re-measured after the rebase; conclusion unchanged
from an earlier 4.89e-5/2.60e-7 pair) alongside genuine dynamic-Cb
float32 noise it was originally trying to describe.

`tests/test_vmap_cpml_dielectric.py::test_vmap_cpml_dielectric_is_finite_and_matches_run`
(a #205 regression pin, unrelated in origin) shared the same fixture
shape as this defect but asserted only the one sweep element
(`eps_r=10.0`) that happened to equal its base simulation's own
material — the one element where #637 can never manifest by
construction. Now asserts every swept element; the previously-unchecked
`eps_r=4.0` element measured rel=9.65e-3 against `run()` pre-fix (the
#637 signature) and is exactly bit-identical post-fix.

**Overlap with issue #627 (landed separately as fce1091, this same
Unreleased block, below) — confirmed divergence, deliberately NOT fixed
here.** #627 moved `Simulation._assemble_materials`'s pad-extension rule
into a new shared `rfx.geometry.rasterize_grid.extend_cpml_pad_materials`,
and changed the rule itself — a per-transverse-cell hi-face fallback (if
the naive interior-edge column is vacuum but the column one further in is
not, replicate from that inner column instead), fixing a case where a
`Box` touching the domain's hi face loses its edge node to `Box`'s
half-open rasterization. `_extend_batched_cpml_pad` (this entry) still
reproduces the PRE-#627 rule — a straight edge-slice copy, no hi-face
fallback. Measured directly post-rebase on a slab spanning the full
domain (`Box((0,0,0), domain)`, touching all six CPML faces including
x-hi exactly at the domain's last interior node, `cpml_layers=6`, base
`eps_r=4.0`): `Simulation.run()`'s assembled x-hi pad now reads the
correct `4.0` (the #627a fix); `vmap_material_sweep`'s batched x-hi pad
for the SAME geometry stays at vacuum `1.0` regardless of the swept
value (`2.0` and `6.0` both read `1.0` there) — the pre-#627 defect,
unaffected by this entry's fix, which only ever reproduced the OLD rule.
DFT-plane impact on this geometry: worst rel err 1.66e-2 (sweep to 2.0)
/ 4.25e-2 (sweep to 6.0) against `run()` — the same order of magnitude
as issue #637's own pre-fix numbers, on a case this entry does not
correct. Both PR bodies already flag reconciling the two helpers as
follow-up work rather than something either PR should absorb now; this
paragraph is that disclosure, not a promise of a later commit in this
PR. Tracked as **issue #643**, which also covers why the fix is not a
drop-in (the batched arrays carry a leading sweep axis the shared
helper's fallback test has to be evaluated per element against, not
once) and its acceptance criterion (byte-identity between the batched
path and `_assemble_materials` across #637's full geometry matrix plus
this exact-hi-face case). Pinned by
`tests/test_vmap_sweep_dft_planes.py::TestVmapMaterialSweepCpmlPad::test_exact_hi_face_touch_matches_run_via_shared_fallback`,
marked `xfail(strict=True)` so it turns into a hard failure — not a
silent pass — the day #643 is closed.

**No-worse guard added to `_extend_batched_cpml_pad`, found in review.**
Re-running #637's own 7-configuration representativeness sweep on the
rebased tree (unmodified geometries, not the margin-nudged variants
elsewhere in this entry) surfaced that #627 didn't just leave the
exact-hi-face case unfixed — for 4 of 7 configurations it made the
vmap/`run()` divergence WORSE than #637's own pre-fix numbers, wherever
a swept value happened to equal the base value. Mechanism: pre-#637-fix,
the unconditional pad copy inherited base's OWN (post-#627, correct)
pad value everywhere, so the swept-equals-base element was accidentally
right; post-#637-fix, that same cell falls back to the naive vacuum
column instead, since `_extend_batched_cpml_pad` doesn't reproduce
#627's fallback — actively discarding a value that was already correct.
Fix: skip the pad-cell overwrite when the source column is vacuum,
using the SAME joint test `rasterize_grid._vacuum` uses
(`eps_r==1.0 & sigma==0.0 & mu_r==1.0`, not a per-field approximation —
an earlier draft tested each field against its own default independently
and, measured against this file's suite, left the same two residuals
this joint version does, since none of these fixtures carries a second
material with nonzero sigma or non-unity `mu_r` to distinguish the two
predicates; the joint version is kept because it is the one that
matches the rest of the package and generalizes to fixtures the current
suite doesn't happen to exercise).

Effect, measured per swept element (not aggregated) across all 14
elements in the 7-configuration sweep: every element where the swept
value equals the base value is now exact or unchanged (previously
already-correct by the coincidence described above; confirmed still
correct, not merely no-worse). Of the 10 elements where the swept value
differs from base, 8 improved (by 0.4% to 16.7%) or were unchanged, and
2 remain measurably worse than #637's pre-fix numbers: the "issue-table
base=2.0" configuration's `eps_r=10.0` element (6.421e-2 -> 6.527e-2,
+1.6%) and the "committed fixture" configuration's `eps_r=2.0` element
(1.193e-2 -> 1.199e-2, +0.5%). Both residuals are small, reproduced
identically across repeated measurement (not floating-point noise), and
their specific mechanism is not established — the joint-vacuum fix did
not change either number, so whatever produces them is a different,
smaller effect than the predicate mismatch the guard targets. Both sit
on the same exact-domain-edge geometry class already disclosed above
and covered by the `xfail(strict=True)` witness; shipped with the
residual disclosed rather than pursuing a third guard variant, since
the class itself is already known-broken and tracked by #643, and #637's
actual target class (material reaching a pad without sitting exactly on
the domain edge) is unaffected — the "single-face touch (x_lo only)"
configuration, which never touches a domain edge exactly, goes
4.191e-3/8.729e-4 -> 2.623e-7/2.258e-7 pre- to post-fix, unchanged by
the guard either way.

### Fixed — import-time binding pollution in the uniform-grid runner (issue #628)

- `rfx/runners/uniform.py` used to do `from rfx.simulation import run as
  _run, run_until_decay as _run_until_decay` at MODULE level. That module is
  imported lazily, the first time some `Simulation` call path actually needs
  it, so if that first import happened while a test had
  `rfx.simulation.run` monkeypatched, `rfx.runners.uniform._run`
  permanently captured the patched stub — `monkeypatch`'s teardown restores
  the attribute on the SOURCE module (`rfx.simulation`) but has no way to
  reach the copy already bound into `rfx.runners.uniform`'s own namespace.
  Every later uniform-lane run in that process then silently called the
  stale stub. Order-dependent symptom: `pytest
  tests/test_coax_two_port_fdtd.py tests/test_refplane_port_waves.py` gave
  `1 failed, 63 passed` (a `TypeError` from the leaked stub in
  `test_run_short_diagonals_byte_frozen_offdiagonals_move`); either file
  alone, or the suite's usual collection order, passed clean.
- Fix: late-bind via `from rfx import simulation as _simulation`, calling
  `_simulation.run(...)` / `_simulation.run_until_decay(...)` — a module
  reference resolved at CALL time, not import time, so it always reflects
  whatever `rfx.simulation.run` currently is. `rfx/runners/uniform.py`'s
  other five `rfx.simulation` names (`make_source`, `make_j_source`,
  `make_probe`, `make_port_source`, `make_wire_port_sources`) are late-bound
  the same way, even though none is currently monkeypatched anywhere in the
  suite: closing the whole block removes the vulnerable SHAPE from the file
  entirely, rather than leaving it one future `monkeypatch.setattr` away
  from recurring in the exact file that already produced a multi-hour false
  attribution during the #582 review. Not a hot path: all seven calls
  happen at most once per `run_uniform()` invocation (setup-time
  port/source/probe registration, or the single entry into the compiled
  `jax.lax.scan` FDTD loop for `run`/`run_until_decay`), so the extra
  attribute lookup is negligible against that call's own runtime.
- Same-shape audit (module-level `from rfx.X import Y` where `X` is
  monkeypatched by a test somewhere in the suite) across `rfx/runners/`,
  `rfx/api/`, `rfx/probes/`, and the package's other eager/lazy import
  sites: no other instance is actually exposed. `rfx/__init__.py`,
  `rfx/gpu.py`, `rfx/rcs.py`, and `rfx/sources/__init__.py` bind the same
  `rfx.simulation`/`rfx.sources.coaxial_port` names, but all three import
  eagerly, as part of `import rfx` itself, which necessarily completes
  before any test can obtain a handle on the source module to patch it —
  structurally impossible to race. `rfx/runners/__init__.py` re-exports
  `run_uniform` from a module (`rfx.runners.uniform`) whose `run_uniform`
  IS monkeypatched by `tests/test_passivity_guard_wiring.py`, but no
  production call site consumes that package-level alias (every caller
  imports `run_uniform` directly, locally, inside the calling function),
  so a stale copy there is inert. `rfx/runners/distributed.py` and
  `distributed_v2.py` import `make_source`/`make_j_source`/`make_probe`/
  `make_port_source` from `rfx.simulation` the same lazily-bound way
  `uniform.py` did, but none of those four names is patched by any current
  test — left as module-level imports, flagged as the same latent shape
  should a distributed-lane test ever patch one of them.
- Regression tests: `tests/test_runner_import_binding.py`. The primary
  test simulates the import-inside-patch-window sequence directly (forces
  `rfx.runners.uniform` out of `sys.modules`, imports it inside an open
  `monkeypatch.context()` on `rfx.simulation.run`, then asserts identity
  after the window closes) — deterministic, independent of collection
  order. A second, `@pytest.mark.slow` test locks the original two-file
  subprocess repro as an ordering regression lock.
### Fixed — CPML hi-face pad was vacuum for domain-face-touching boxes (issue #627a; #627b attempted, found unsafe, reverted — deferred to issue #636)

- Follow-up to #582's review, which found two gaps its uniform-vs-NU pad
  replication mirror faithfully inherited from the (pre-existing) uniform
  path. Only the first is fixed here, in a new shared
  `rfx.geometry.rasterize_grid.extend_cpml_pad_materials`, used by both
  `rfx/api/_compile.py`'s `_assemble_materials` and
  `rfx/runners/nonuniform.py`'s `assemble_materials_nu` — replacing the two
  hand-duplicated pad-extension blocks that #582 had verified byte-identical
  (a duplication that is itself the historical defect class here: two
  hand-maintained copies of one piece of logic drift).
- **(a) Hi-face vacuum column for a domain-face-touching `Box` — FIXED.**
  `rfx.geometry.csg.Box`'s volume rasterization is deliberately half-open,
  `[lo, hi)`, over node coordinates (see that class's docstring — the
  convention is load-bearing across the package, e.g. every WR-90
  aperture/iris measurement, and is explicitly OUT OF SCOPE to change here:
  doing so would move geometry everywhere in the package). Its documented
  consequence is that a box's hi face "contributes no node": a structure
  whose hi face lands on the domain's last interior node loses exactly that
  node from its own rasterized mask, so the naive interior-edge source for
  a hi-face CPML pad read vacuum even though the structure's real material
  sits one column further in. Measured pre-fix on the #582 fixture: x-lo
  pad `eps_r=4.0`, x-hi pad `eps_r=1.0`, for a slab spanning the full x
  extent. Fix: per transverse cell, if the naive interior-edge column is
  vacuum (`eps_r==1 & sigma==0 & mu_r==1`) but the column immediately
  inside it is not, replicate from that inner column instead — bounded to
  exactly one column inward, matching the rasterizer's deterministic
  one-node-per-box shortfall, so a genuine multi-cell vacuum buffer between
  an interior structure and the CPML pad (the overwhelmingly common
  simulation layout) is untouched and still replicates plain vacuum, as
  before. Locked by
  `tests/test_cpml_pad_material_extension.py::test_genuine_vacuum_buffer_before_cpml_is_not_bridged`.
- **(b) Dispersion poles were never extended into the pad — attempted, then REVERTED; NOT shipped, at all.**
  Debye/Lorentz pole masks are built straight from the geometry loop with
  no pad-extension step (on either face) — a dispersive edge-touching
  material has its static `eps_r` matched into the pad but not its poles,
  so the pad medium is non-dispersive: matched at DC, mismatched across
  the band. That gap is **still open**. An earlier revision of this change
  extended the pole masks the same way as the static arrays, using the
  same shared function; review's controlled four-way discriminator (one
  fixture, one harness, only the pad-fill contents varied, printed per
  variant to confirm each matched its label) found: **extending
  dispersion poles into the pad turns a stable high-Q (Q≈60) edge-touching
  Lorentz-slab simulation into a divergent one; the static extension (a)
  alone, with the same high-Q pole left un-extended in the interior,
  decays cleanly** — 20,000-step last-decile/mid-decile energy ratio 649
  (poles extended) vs 0.1557 (statics-only, poles genuinely off in both
  pads, decaying at the same order as the unpatched tree's 0.12). The
  divergence has no NaN and no exception — values stay finite and simply
  grow — so nothing downstream flags it. Because of this, **pole-mask
  extension is not included in this change at all**:
  `extend_cpml_pad_materials` only extends `eps_r`/`sigma`/`mu_r`, with no
  parameter or code path for pole masks. It is not gated behind a flag
  either — an opt-in that silently turns a stable simulation into a
  divergent one is worse than no feature. Tracked in full — the factorial,
  the mechanism hypothesis, and a separate coverage hole the attempted
  design also had (a *pole-only* material, background `eps_r` left at 1.0,
  was invisible to the hi-face-vacuum test) — in **issue #636**; that
  detail is deferred work now, not a caveat on what ships here. Guarded
  against silent reintroduction by a physics-level regression lock (see
  below).
- **Absorber-quality witness, (a) only** (4000-step run, `eps_r=9.8,
  sigma=0.5, mu_r=2.5` slab spanning the full x/y extent, probe near the
  x-hi pad interface, `cpml_layers=8`, no subpixel smoothing, no
  dispersion pole): tail/peak energy improves −85.15 dB → −90.62 dB —
  consistent in direction and rough scale with #582's own −63.2 dB →
  −76.7 dB precedent.
- **Byte-identity preserved.** `eps_r`/`sigma`/`mu_r` are bit-identical
  between the uniform and NU assemblers post-fix (0 mismatched cells of
  72,912, checked for non-dispersive, Debye, and Lorentz materials,
  including a lossy+magnetic case) — extending the property #582's review
  established. `tests/test_nonuniform_uniform_end_to_end_reduction.py`
  (the anchor that originally caught #582) stays green:
  `staircase-slab-cpml` residual 8.83e-5, `subpixel-cpml` residual
  1.40e-4 (both well under the 3e-4 gate; not expected to move much, since
  the fix is symmetric across both assemblers — the reduction anchor
  compares the two paths to each other, not either to ground truth).
- **Regression lock against silently reintroducing (b).**
  `tests/test_cpml_pad_material_extension.py::test_pole_extension_stability_lock`
  runs the (a)-only shipped code on the same edge-touching, high-Q-Lorentz
  fixture the #636 discriminator used (~8000 steps) and asserts the run
  decays (last decile below the mid-run decile). It passes on the shipped
  code and is designed to red the moment pole extension is naively
  reintroduced — that is its entire purpose, so it stays even though (b)
  itself is gone.
- **Scope.** Rasterizer semantics (`Box.mask`/`mask_on_coords`) are
  unchanged — the fix lives entirely in the post-rasterization pad-fill
  step. PEC pad replication remains out of scope (PEC has never been
  extended into the pad on either path; a PEC surface intersecting a CPML
  boundary is a pre-existing, separate situation not addressed by #582 or
  #627). Of the two gaps #582's Scope clause deferred: (a) is closed in
  full; **(b) is NOT closed** — see above.
  **Also out of scope for both #582 and #627, found in review**:
  `subpixel_smoothing=True` builds its anisotropic eps tensor (`aniso_eps`)
  directly from `sim._geometry` shapes with a hardcoded
  `background_eps=1.0` (`rfx/geometry/smoothing.py`'s
  `compute_smoothed_eps` / `compute_smoothed_eps_nonuniform`, called from
  `rfx/runners/uniform.py` and `rfx/runners/nonuniform.py`), independently
  of — and without reading — the CPML-pad-extended `eps_r`/`sigma`/`mu_r`
  arrays this entry and #582 fix. Verified directly: with
  `subpixel_smoothing=True` the field update never sees the pad-replicated
  material at all, on either path. In short, "impedance-matched absorber"
  in this file and #582 has only ever meant the non-smoothed (staircase)
  rasterization; the subpixel-smoothed path's CPML pad has always been
  plain vacuum regardless of the adjacent structure, unaffected by either
  fix.

### Fixed — arc-audit follow-up: flaky benchmark gate, an over-general `n_warmup` claim, an unsafe `field_softmax` default, and asymmetric `jacobian_fwd` safety

An independent adversarial audit of the e4b565c..ce44661 merge arc (10 PRs,
issues #571/#577/#578/#579/#582/#620/#622/#623/#625/#626/#632) found five
of our own mistakes after merge. This entry documents the fixes; it is a
correction of the arc's own errors, not new feature work. A second,
independent verification pass over THIS entry's own fixes then found
several more defects in them (an overflow guard that clipped the value
but not the gradient, a NaN misdiagnosed as a tangent-dependence
violation, a guard stricter than the code it was meant to match, an
under-sampled falsifier sweep that itself overstated its claim, and the
`_ExecuteMixin` leak turning out to be ten methods, not one) — folded
into the same bullets below rather than filed separately, since they are
corrections to fixes that had not yet shipped.

- **Benchmark flakiness (issue #632 follow-up)**:
  `tests/test_benchmark_jacobian_fwd.py` asserted `intercept_vs_plain_ratio`
  landed in `[0.5x, 2.0x]` for EVERY column including `wall_s` — confirmed
  red on main (commit 55fa85b, CI run 31464443445, fast-suite shard 1:
  `wall_s = 0.474`, just outside the band) and green again on the very
  next commit with no code change, exactly the machine-dependence #632
  itself had already diagnosed for this quantity (0.98-1.10 CPU vs
  1.41-1.87 on an RTX 4090). The assertion now gates only the
  deterministic compiler-derived columns (`flops`, `temp_bytes`); `wall_s`
  is still computed and left in the table for a human/CI-log reader, never
  gated. The batched-vs-sequential wall-time comparison (same file) is the
  same class of check and gets the same treatment: reported, not asserted.
  `tests/test_jacobian_fwd.py`'s G3 (jaxpr-structural, backend-independent)
  remains the authoritative primal-sharing check.
- **`n_warmup` truncation error is placement-dependent, not universal
  (issue #626 addendum)**: the shipped curve (near-source design cell,
  ~3 cells from the source) is the WORST case, not the general case. A
  far-from-source counter-fixture (design cell 62 cells from the source,
  `K_safe=108`) measures the AD gradient staying within this repo's own
  established ~1.5% AD-vs-FD noise floor for every `n_warmup <= K_safe`:
  <0.01% through `K_safe`-20 (deep sub-wavefront, near-exact), growing
  SMOOTHLY (not a sharp cliff — a finer sweep bracketing `K_safe`, added
  in the same verification round that found this, shows 0.26% at
  `K_safe`-4, 0.60% at `K_safe`-2, 1.19% AT `K_safe` itself), only
  exceeding that noise floor past `K_safe` — new committed script
  `scripts/diagnostics/i626_n_warmup_wavefront_locality.py`. Corrected
  guidance ships in `rfx/nonuniform.py`'s `n_warmup split` comment,
  `Simulation.forward()`'s `n_warmup` docstring, `rfx.observables.
  jacobian_fwd`'s fence note, and `rfx.runners.distributed_nu.
  run_nonuniform_distributed_pec`'s docstring: compute
  `K_safe ~= floor(min_distance(source, design_region) / (C0 * dt))`
  (`min_distance` over every active source AND each source's own spatial
  extent — a TFSF plane-wave source illuminates from an entire box face,
  not a point);
  `n_warmup <= K_safe` stays within the noise floor above (near-exact
  deep below `K_safe`, merely noise-floor-comfortable AT it). No runtime
  warning ships —
  `forward()` does not know which cells are the "design region" (that
  concept was deliberately removed with `design_mask`, issue #625), so a
  warning built on a guess would be exactly the kind of unverifiable claim
  this repo's own discipline forbids; the formula is documentation, not
  an automated check.
- **`rfx.observables.field_softmax`'s `beta` semantics changed — see the
  dedicated BREAKING entry below** (issue #619). Summary only here: the
  old default was unsafe at realistic field magnitudes; `beta` is now
  auto-scaled and DIMENSIONLESS. Filed as its own top-level BREAKING
  entry, not folded into this list, because it silently changes the
  numeric meaning of an existing top-level public-export parameter for
  every caller who passed a non-default `beta` — unlike this list's other
  four items, there is no raise or new error to notice it by.
- **`jacobian_fwd`'s two tangent-batching paths had asymmetric safety and
  one false docstring claim**: the docstring stated `batch_tangents=False`
  runs its sequential `jax.jvp` calls "inside one `jit`" — there is no
  `jax.jit` anywhere in `rfx/observables.py`; the claim was never true
  (corrected). More substantively: the batched path (`jax.vmap(...,
  out_axes=(None, 0))`) raises loudly if `sim_fn`'s primal genuinely
  depends on the tangent direction (an API-purity invariant); the
  sequential path had no equivalent check and would have silently
  returned one arbitrary tangent row's primal. It now carries two guards:
  (1) a `jax.eval_shape` abstract trace of the same `out_axes=(None, 0)`
  vmap — zero FLOPs, zero extra memory, works whether or not the caller
  wraps the call in an outer `jax.jit`; (2) an eager exact pairwise
  comparison of the actually-computed primal values (skipped only under
  an enclosing `jax.jit`, where values are not yet concrete). New tests
  `tests/test_jacobian_fwd.py::test_g8_*` cover both guards, including a
  deliberately broken `custom_jvp` rule that violates the invariant, a
  cross-call-impurity case guard 1 structurally cannot see (only guard 2
  catches it), and confirmation neither guard false-positives on a normal
  `sim_fn` or breaks under an outer `jax.jit` (the benchmark script's own
  usage pattern).
- **Small fixes**: `tests/test_n_warmup.py`'s "three uniform-lane
  siblings" comment corrected to two (`design_mask` was removed outright,
  issue #625, one commit before this comment was added — not fenced, so
  it is not part of this taxonomy). `Simulation.forward()`'s removed-kwarg
  error used to read `TypeError: _ExecuteMixin.forward() got an
  unexpected keyword argument 'design_mask'`, leaking the internal mixin
  class name (the public surface is `Simulation.forward`) with no reason
  or replacement; `forward()` now accepts `**_removed_kwargs` and raises a
  `TypeError` naming the removal reason and the one-line migration path
  instead (`rfx/api/_execute.py::_reject_removed_forward_kwargs`) — still
  a plain `TypeError`, still matched by `tests/test_design_mask_removed.py`'s
  existing `pytest.raises(TypeError, match="design_mask")` assertions, plus
  a new test pinning the improved message. `docs/public/guide/
  memory-reduction.mdx` gained a "Restricting which cells carry a
  derivative" section (the `design_mask` migration path previously existed
  only in this CHANGELOG, not in public docs).
- **The `_ExecuteMixin` leak above was not one-off — verification round
  fix, ten methods.** Every method `Simulation` inherits from its five
  mixins (`_PreflightMixin`, `_SparamMixin`, `_CompileMixin`,
  `_ExecuteMixin`, `_ArtifactsMixin`) kept that mixin's name in its
  `__qualname__` (Python sets it from the class body where a function is
  DEFINED, not where it is bound via inheritance), so ANY unrecognised
  keyword argument on ANY public method leaked the internal mixin —
  measured: `sim.run(n_stepss=2)` read `_ExecuteMixin.run() got an
  unexpected keyword argument`. `rfx/api/__init__.py` now rebinds every
  inherited method's `__qualname__` to `Simulation.<method>` once, at
  class composition time (structural only — does not change behaviour,
  identity, or MRO). New test:
  `tests/test_design_mask_removed.py::test_no_public_simulation_method_leaks_a_mixin_class_name`.
- `docs/guides/api_symbol_inventory.json` regenerated
  (`python scripts/check_api_reference.py --write`) for `forward()`'s new
  `_removed_kwargs` parameter.

### Changed — `rfx.observables.field_softmax`'s `beta` is now auto-scaled and DIMENSIONLESS (**BREAKING**) (issue #619)

- **The numeric meaning of `beta` changed for every existing caller who
  passed a non-default value — silently, with no raise.** Before this
  change, `field_softmax(names, beta=B)` computed
  `logsumexp(B * vals) / B` directly against the raw `|field|**2` values,
  so `B` had to be hand-matched to whatever physical units/magnitude the
  DFT-plane accumulator happened to carry (commonly `|field|**2` ~1e-10
  to 1e-22 in this repo). At the old default `beta=1.0`, a realistic
  field magnitude made the objective sit at ~100.000002% of the
  design-independent constant `log(count)/beta` — both the value and
  gradient measured rounding noise, not physics (measured, `#619`).
- **The fix**: `field_softmax` now computes
  `beta_eff = beta / stop_gradient(max(vals))` and uses `beta_eff` in
  place of a raw `beta` throughout, so `beta_eff * max(vals) == beta`
  identically regardless of the field's physical units. `beta` is now
  DIMENSIONLESS (a target sharpness — see the docstring's
  "What beta now controls" section) rather than a raw multiplier on
  `|field|**2`. The old rounding-noise swallow cannot recur at any
  representable field magnitude, at any beta, including the default
  (now merely LOOSE at low beta, never rounding-noise).
- **MIGRATION — rescale any beta you already tuned.** A `beta` value
  chosen under the OLD semantics (e.g. `beta=1e11`, tuned so
  `beta * max(vals) ≈ 50` for a specific field magnitude) means something
  completely different under the new one (`beta_eff * max(vals) == beta`
  identically, so passing `beta=1e11` now targets a wildly over-sharp
  softmax relative to the old intent). Do not carry an old numeric `beta`
  forward unexamined: pick the new dimensionless value directly from the
  "What beta now controls" guidance in the docstring (5-50 is a
  reasonable range for most plane sizes), or recompute it as
  `new_beta ≈ old_beta * old_max_vals` if you need to reproduce a
  specific old operating point. This repo's own callers were re-tuned by
  measurement, not by that formula: `tests/test_observables_dft_field.py`'s
  `_SOFTMAX_BETA` (`1e11` raw → `200` dimensionless) and
  `examples/inverse_design/field_observable_shielding.py`'s `BETA`
  (`2.0e24` raw → `20` dimensionless).
- **Follow-up fix, same verification round: the overflow footgun the
  auto-scaling was meant to close RELOCATED rather than closed.**
  `beta_eff = beta / max(vals)` can itself overflow for a large enough
  `beta` against a tiny field — measured: at `max(vals)` ~2.24e-22,
  `beta=2.0e24` (the literal value the shielding example carried before
  its own re-calibration above) returned `value=nan, grad=nan` silently.
  `field_softmax` now clips `beta_eff` to the working dtype's largest
  finite value when it would overflow, keeping the VALUE finite and
  accurate. The GRADIENT is a separate, tighter limit the clip does NOT
  cover: it silently collapses to exactly `0.0` somewhere in the
  `beta~1e15-1e17` range (fixture-dependent; a `1/beta_eff`
  floating-point precision limit in the VJP, not fixed here) — both
  thresholds sit many orders of magnitude above the recommended 5-50
  range, so this is a documented edge, not a practical concern for any
  beta this repo's own guidance recommends. New regression tests:
  `test_field_softmax_beta_eff_overflow_returns_finite_not_nan` and
  `test_field_softmax_gradient_can_silently_collapse_past_the_recommended_beta_range`.
- New regression tests (scale-safety, distinct from the overflow tests
  above): `test_field_softmax_default_beta_is_scale_invariant` (verifies
  `field_softmax(c*vals) == c*field_softmax(vals)` at the default beta
  across 30 orders of magnitude) and
  `test_field_softmax_default_beta_gradient_not_rounding_noise` (direct
  FD-vs-AD check at a physically tiny field magnitude).

### Removed — `design_mask` deleted from every public surface (issue #625, BREAKING)

- `design_mask` is deleted, not deprecated, from every entry point that
  accepted it: `Simulation.forward()`, `rfx.optimize.optimize()`,
  `Simulation.estimate_ad_memory()`, `Simulation.explain_ad_memory()`,
  `Simulation.plan_ad_memory()`, `Simulation.ad_memory_preflight()`,
  `Simulation.ad_memory_compiled_certificate()`,
  `Simulation.mesh_intelligence_report()`, the single-device NU runner
  (`rfx.nonuniform.run_nonuniform`), and the distributed NU runner
  (`rfx.runners.distributed_nu.run_nonuniform_distributed_pec` and the
  `shard_design_mask_x_slab` helper, also deleted). The
  `AD_MemoryEstimate.ad_active_design_fraction` reporting field — which
  existed only to report the mask's active-cell fraction — is deleted
  with it, from both the Python object AND its `to_dict()`/`to_json()`
  serialization: the key is absent, not `null`, so a consumer parsing
  the dict/JSON that indexes `["ad_active_design_fraction"]` now fails
  at READ time with `KeyError`, not at call time — check for the key's
  presence, or catch the read, before assuming it is there. Passing
  `design_mask=` anywhere now raises `TypeError` for an unrecognised
  keyword argument (pinned by `tests/test_design_mask_removed.py`), so a
  future reintroduction is a deliberate act, not a silent slip.
- **Why (measured on the production `forward()`/`nonuniform.py` path)**:
  the mechanism saved **zero** reverse-mode AD memory — the tape is
  byte-identical with and without the mask at `n_steps`=40/80/160
  (127,770,748 bytes both ways at n=160), and **+496,584 bytes WORSE**
  at `checkpoint=False` plus **+4.7% backward wall time**. This is
  structural, not an implementation bug:
  `jnp.where(mask, x, jax.lax.stop_gradient(x))` lowers to `select_n`,
  whose JVP/transpose need only the loop-invariant boolean predicate, so
  it deletes cotangents rather than shrinking storage — JAX's
  partial-eval residuals have whole-array granularity, so confining the
  design variable to 0.13% of cells shrank the tape by 0.06%. Worse, it
  corrupted the gradient in every configuration tested: an observable
  positioned outside the design region read back exactly 0.0, and one
  co-located with the mask read back **69.9%** (3^3 region) / **52.8%**
  (7^3 region) relative error against a reference matching central
  finite differences to 1e-5 — the masked gradient was a function of
  mask geometry alone, not the derivative of anything with respect to
  the design variable.
- **The 1.6.2 entry below (`#41`) and the deleted docstring described
  incompatible things, and the docstring was the wrong one.** The 1.6.2
  entry's stated intent was material-scoped — confine gradient credit to
  a design region so "gradients cannot escape the design volume". What
  was actually implemented was field-scoped: `stop_gradient` applied to
  the E/H field carry inside the scan body every step, framed in the
  removed docstring as a memory optimisation ("backward memory +
  wall-time scale with mask occupancy instead of grid volume"). That
  memory claim was simply false (see the zero-savings measurement
  above), and the field-scoped mechanism it actually shipped is not the
  material-scoped restriction the original issue asked for.
- **The stated goal was already served, orthogonally and fully, by
  segmented remat**, which this repo already ships:
  `checkpoint_every`/`checkpoint_segments` measured
  127,770,748 -> 10,687,688 (`checkpoint_every=13`) -> 3,519,260 bytes
  (n=40), identically with or without a design_mask, two days before
  `design_mask` (`#41`) shipped. No comparable package masks field state
  for this purpose either — Meep/Lumerical/Tidy3D restrict the
  parametrisation and recording, fdtdx/ceviche restrict storage via
  custom VJPs — because the adjoint field must traverse the non-design
  volume regardless of which cells a design variable touches.
- **Migration**: for memory, use `checkpoint_every` (non-uniform
  scan-of-scan) or `checkpoint_segments` (uniform segmented scan) — see
  `docs/public/guide/memory-reduction.mdx`. To restrict which material
  cells carry a derivative, construct the restriction yourself at the
  `eps_override` call site: `eps = jnp.where(region, eps,
  jax.lax.stop_gradient(eps))`. Verified independently for this removal
  (CPU, float32, an absorbing-boundary fixture with a 3x3x3 region):
  forward output is bit-identical with and without the wrapper
  (`stop_gradient` is forward-identity); the resulting per-cell gradient
  is EXACTLY zero outside the region and EXACTLY equal — bit-identical,
  not merely close — to the unmasked reference gradient inside it; that
  unmasked reference itself checks out against independent central
  finite differences to 1.9% on a single-cell spot check, within this
  repo's own established float32 single-cell noise floor (see
  `tests/test_jacobian_fwd.py`'s geometry-leg discussion).
- Tests: `tests/test_design_mask.py` deleted outright (every assertion
  survived the identically-zero/corrupted gradient the feature actually
  produced). The distributed parity tests that pinned the
  outside-mask-gradient-is-zero behaviour as correct
  (`tests/test_distributed_nu_kernel.py::test_distributed_design_mask_stop_grad_matches_single_device`,
  `tests/test_distributed_nu_composition.py::test_forward_distributed_design_mask_stop_grad_matches_single_device`)
  are deleted rather than converted, since every one of their assertions
  was mask-specific. `tests/test_design_mask_removed.py` (new) pins that
  every entry point above now raises `TypeError`.

### Fixed — `forward(n_warmup=...)` on the uniform lane now raises instead of silently doing nothing (**BEHAVIOUR CHANGE**) (issue #626)

- `Simulation.forward()` declared and documented `n_warmup` but only
  forwarded it to the non-uniform / distributed-non-uniform lanes;
  `_forward_from_materials` (the uniform single-device lane) has no
  warmup-split parameter at all, so a nonzero `n_warmup` on a uniform
  mesh was silently accepted and ignored. Measured: gradient bit-identical
  at `n_warmup=0` vs `n_warmup=60` on the uniform lane. A nonzero
  `n_warmup` on the uniform lane now raises `NotImplementedError` in
  `_dispatch_plan`, matching its two uniform-lane siblings
  (`emit_time_series=False`, `checkpoint_every`; `design_mask`, the third
  historical sibling, was removed entirely rather than fenced — see the
  entry above).
- **Decision: fail-loud, not implement.** `checkpoint_segments` already
  gives the uniform lane an EXACT reverse-mode memory reduction
  (~`sqrt(n_steps)` scaling, no gradient approximation). `n_warmup`, where
  it IS implemented (the non-uniform lane), turns out to be an
  APPROXIMATION rather than a free memory lever — see the next entry.
  Porting that approximation to the uniform lane would add a new,
  accuracy-lossy feature with no memory benefit `checkpoint_segments`
  doesn't already cover exactly, and would require threading the
  warmup/optimize scan split through the uniform-lane-only accumulators
  the non-uniform lane doesn't have (S11 DFT, Kerr χ³, `mu_r_override` —
  flux monitors, NTFF and lumped-RLC ADE state are already real
  non-uniform-lane carries, so those three do NOT add to this cost).
  Silently accepting-and-ignoring a kwarg is the SILENT_WRONG shape this
  repo systematically eliminates; a clean raise is the responsible fix.
- **`rfx.optimize.optimize()` is a second entry point for this behaviour
  change**: it accepts `n_warmup` (forwarded straight to `sim.forward()`
  at both the plain and progressive-resolution call sites), so
  `optimize(..., n_warmup=5)` on a uniform-mesh `Simulation` now raises
  through the optimizer too — the right outcome, and the surface where a
  truncated gradient would otherwise actually drive a design loop.
- **What n_warmup actually does, measured** (non-uniform lane, where it IS
  implemented): it is NOT an exact truncation. Severing the scan carry at
  the warmup boundary also severs every gradient path from a design
  variable's influence during the warmup window back into the loss.
  Swept `n_warmup=K` against an independent central-FD oracle (loss
  window held FIXED across the sweep, K varied independently — the
  existing `test_warmup_grad_finite_and_same_sign` test lets the loss
  window track `warmup`, which conflates the two and cannot isolate the
  effect) at two independent design-cell placements: forward output is
  exactly `n_warmup`-invariant (bit-identical) in both; the gradient
  error vs. the FD oracle stays near this repo's established AD-vs-FD
  noise floor (≲1.5%) while K is up to roughly a quarter of the
  pre-loss-window step count, then grows smoothly and monotonically —
  fixture 1 (`n_steps=100`, loss window `[80, 100)`): K=0 → 0.25%, K=10 →
  1.1%, K=30 → 3.3%, K=40 (half the pre-loss-window) → 6.6%, K=50 →
  12.1%, K=70 → 35.2%, K=80 (the loss-window boundary itself) → 58.4%,
  K=95 → exactly 0 (gradient fully vanished, not merely small). Two
  further placements at weaker design/loss coupling (smaller true
  gradient, so the FD oracle itself sits closer to the float32
  single-cell noise floor) reproduce the SAME large-K collapse toward
  the loss-window boundary, but do NOT cleanly reproduce monotonicity at
  small/medium K — the low-K ordering there is noise-floor-dependent on
  how well-conditioned the oracle is at that specific cell, which is why
  the regression test below only asserts strict monotonicity from K=30
  upward rather than across the whole sweep. Docstrings (`forward()`,
  `run_nonuniform`, `run_nonuniform_distributed_pec`, `jacobian_fwd`) and
  `docs/public/guide/memory-reduction.mdx` are updated to state this
  curve plainly instead of framing `n_warmup` as a free memory lever.
- **Caveat on the sweep's own monotonicity gate**: an h-sweep of the FD
  oracle shows K=0 and K=10's errors sit inside the oracle's own noise
  band (at `h=0.05` the K=10 point reads slightly BELOW K=0) — the
  published `h=0.02` figures above are correct as measured, but a strict
  monotonicity assertion across the *whole* sweep would bind that
  noise-band coincidence. `tests/test_n_warmup.py`'s regression test
  floors the low-K comparisons and only asserts strict monotonicity from
  K=30 upward, where the trend is unambiguous.
- Tests: `tests/test_n_warmup.py::test_uniform_forward_rejects_n_warmup`
  (the fail-loud regression witness — pins raise vs. no-raise, not merely
  "both run") and `::test_warmup_truncation_error_grows_with_k` (the
  measured error-curve witness). `tests/test_jacobian_fwd.py`'s G6 fence
  taxonomy moves `n_warmup` from a documented trap to an inherited raise
  (`test_g6b_n_warmup_fence_raises` replaces
  `test_g6b_n_warmup_is_a_documented_trap_not_a_raise`; the taxonomy is
  now three configurations / two raises / one docs-only trap, since
  `design_mask` — one of the prior four — was removed outright rather
  than fenced).
- **Addendum (arc-audit follow-up, ABOVE): the curve above is a
  NEAR-SOURCE worst case, not a universal property of `n_warmup`.** See
  the "Fixed — arc-audit follow-up" entry near the top of this file for
  the corrected, distance-dependent statement and the `K_safe` formula —
  read this entry's measured numbers as "what happens once the wavefront
  has already reached the design region," not as "what `n_warmup` always
  does."

### Changed — `benchmark_jacobian_fwd.py`'s intercept/plain witness text now flags itself as CPU-calibrated (issue #632)

- The printed `intercept/plain ratio (... -- expect close to 1.0)` line
  read as a regression on an accelerator: it is 0.98-1.10 on CPU but
  1.41-1.87 on an RTX 4090 (measured, VESSL runs 369367252824 /
  369367252825, commit 27c4fad), because the two-point endpoint fit
  pushes fixed per-call overhead (kernel launch, tiling) that does not
  scale with `n_t` into the intercept, and that effect is largest exactly
  where the marginal cost per tangent is smallest. Sharing is not broken
  there. Reworded the printed line and the module docstring to state the
  CPU calibration and point at G3 in `tests/test_jacobian_fwd.py` — a
  jaxpr-structural, backend-independent check — as the authoritative
  witness. No behaviour change.

### Fixed — uniform distributed lane's CPML/PEC x-hi wall displacement in the pad_x>0 lane (issue #623)

- Same class as #622, in the sibling UNIFORM-grid runner:
  `rfx.runners.distributed._apply_cpml_e_distributed` and
  `_apply_cpml_h_distributed` (called from `distributed_v2.py`'s
  `_apply_cpml_e_shmap` / `_apply_cpml_h_shmap`, which is what production
  `sim.run(distributed=True)` / `devices=...` dispatches to) left the x-hi
  CPML absorber window at the padded slab end instead of shifting it left
  by `pad_x`, whenever `nx % n_devices != 0` (the common case post-#564,
  since an even cell request now realizes an odd node count). Measured on
  a 2-device CPML fixture with `nx=59` (`pad_x=1`), probe near the x-hi
  face: `rel_err` **3.14e-01 pre-fix vs 8.13e-06 post-fix** (~38,600x) —
  a clean physics witness, unlike #622's PEC-face site which needed a
  structural array assertion because the x-hi terminal node there is an
  identically-zero fixed point under PEC.
- The legacy `jax.pmap` runner's `_apply_pec_local` / `_apply_pmc_local`
  x-hi face appliers had the matching off-by-`pad_x` bug, but this site
  is currently UNREACHABLE through the public `run_distributed()` entry
  point — it hard-raises `ValueError` when `nx % n_devices != 0` (no
  padding support there), so the fix is defensive/consistency-only,
  witnessed by a direct structural unit test (mirrors #622's F4: no
  simulation-level fixture can trigger this site).
- Fix: threads a static `pad_x` (Python int, default 0) into both CPML
  appliers and the legacy PEC/PMC appliers; the x-hi window/face index
  shifts by `pad_x` so it covers the real physical face (global node
  `nx - 1`), matching single-device `cpml.py`'s `[nx-n, nx)`. The
  `pad_x == 0` lane is unchanged (no-op shift). Four pre-existing
  2-device tests in `tests/test_distributed.py` already exercised
  `pad_x=1` (3 Debye + 1 Lorentz fixtures) — measured **0.0 rel_err
  before AND after** this fix. That is a structural, not a
  timing/reach, result: those fixtures use `boundary="pec"`, which
  routes through `distributed_v2.py`'s `_apply_pec_shmap` /
  `_apply_pmc_shmap` (already made `pad_x`-aware by #622) — the CPML
  appliers and legacy pmap appliers this PR touches are never invoked
  on a PEC-boundary run at all, so this PR's fix has no code path to
  move on that fixture regardless of probe placement or run length.
- Lane visibility: `tests/test_distributed.py` carried a module-wide
  `pytest.mark.gpu` (plus a since-superseded `os.environ["XLA_FLAGS"] =
  ...` override that the root conftest's `setdefault`-based ordering
  already made a no-op) that put its entire 2-/4-device family in the
  same runs-nowhere state #622 found and fixed for
  `test_distributed_nu_kernel.py`. The marker is removed; measured
  (2-device conftest default): 33/33 passed, 65-68 s wall, 1.74 GiB peak
  RSS — comparable to (and below) the NU kernel file's precedent (62-66
  s, 1.93 GiB) and well under the `highmem` marker's 3-24 GB band. The
  file now runs in the same fast-suite/weekly CPU shards as its NU
  siblings. `scripts/vessl_gpu_suite.yaml`'s comment is updated to match.

### Added — `rfx.jacobian_fwd`, a batched forward-mode Jacobian wrapper (issue #577)

- `rfx.observables.jacobian_fwd(sim_fn, params, *, tangents="identity",
  batch_tangents=True)` (flat-exported as `rfx.jacobian_fwd`) computes
  `(value, jacobian)` for any JAX-differentiable `sim_fn(params) -> value`
  (typically `lambda p: dft_field(name)(sim.forward(eps_override=eps(p),
  ...))`) via `jax.vmap(jax.jvp(...))` over a tangent basis, NOT
  `jax.linearize`. This is PACKAGING, not new solver capability: forward
  mode already ran end-to-end through `sim.forward()` ->
  `rfx.observables` with zero solver changes before this PR — the new
  surface is tangent-basis construction, primal de-duplication, and four
  documented fences.
- **The headline finding is a negative one, and it is the honest
  deliverable**: the issue's premise that batched forward-mode AD "shares
  the expensive primal sweep" is TRUE (verified both structurally, via
  the compiled jaxpr's scan carry, and via a cost-model fit whose
  intercept lands within a few percent of one plain solve — see
  `tests/test_jacobian_fwd.py::test_g3_primal_carry_is_independent_of_n_t`
  and `scripts/benchmark_jacobian_fwd.py`), but the issue's hoped-for
  economics ("a few times one solve, not N times") is NOT what the
  current kernel delivers. MEASURED, CPU (`jax.default_backend()=="cpu"`),
  jax 0.10.2, float32, `n_steps=120`:
  - grid 35x31x31 = 33,635 cells: wall time at `n_t=10` is **5.9x** one
    plain solve (batched) / **21.3x** (sequential, `batch_tangents=False`);
    flops **17.5x**; XLA-reported `temp_bytes` (a *compiler estimate*, not
    a measured/observed/certified runtime peak — see the honesty-label
    discipline in `tests/test_estimate_ad_memory.py`) **10.4x**.
  - grid 59x55x55 = 178,475 cells (5.3x more cells), same `n_steps`: the
    SAME `n_t=10` batched configuration moves to **7.3x** wall time /
    **18.6x** flops / **10.4x** `temp_bytes` — the flops ratio itself moved
    6.4% from the small grid's 17.5x (it is NOT a fixed constant across
    problem sizes, which is the whole point of quoting two grid sizes).
    Separately, flops and `temp_bytes` (both deterministic compiler
    outputs, unaffected by machine load) reproduce the *evidence base's own
    independent probe measurement* on the same two fixtures to within ~1%
    — a cross-check that this implementation's cost characterization is not an
    artifact of this particular run. Wall time (subject to machine load)
    moved from 5.9x to 7.3x between the two grids. Re-run
    `scripts/benchmark_jacobian_fwd.py --grid-scale 24` for the current
    number on your hardware rather than trusting either one quoted here.
  - forward-mode `temp_bytes` is measured INDEPENDENT of `n_steps`
    (`n_steps=120` vs `240` differ by <0.02% — noise, not a real
    dependency) — this is the actual argument FOR this mode: reverse mode
    on the same fixture costs `temp_bytes` tens of times one plain solve
    for a SINGLE scalar output at `n_steps=120` and cannot produce a
    many-output Jacobian in one pass at all (`jax.jacrev` raises
    `TypeError` on a complex-dtype output without `holomorphic=True`).
  - batching beats running the `n_t` tangent directions one at a time:
    measured wall time ~3x faster and flops ~1.3-1.4x fewer at `n_t=10`
    than the `batch_tangents=False` sequential control on the same
    fixture.
  - Numbers are never pinned as constants in a test or docstring (repo
    anti-rot rule) — `tests/test_benchmark_jacobian_fwd.py` gates
    RELATIONS (intercept-vs-plain-solve ratio band, batched-faster-than-
    sequential, memory-independent-of-n_steps), and
    `scripts/benchmark_jacobian_fwd.py` is the regenerable source for any
    number quoted above or in a PR body.
- Complex-Jacobian convention: for a complex-valued `value` (e.g.
  `dft_field`'s output), the returned Jacobian is `dy/dx` UNCONJUGATED —
  it differs by a conjugate from anything derived through `jax.vjp`/
  `jax.grad` on the same computation. Documented in the function's own
  docstring; gated by `tests/test_jacobian_fwd.py::test_g2_jacobian_is_unconjugated_dy_dx`.
- `tangents="identity"` is defined as per-element identity over SCALAR
  leaves of `params` ONLY, and fails loud (`ValueError`) on any non-scalar
  or non-floating leaf — a naive per-leaf identity on a multi-element leaf
  silently sums that leaf's Jacobian entries instead of isolating one
  column of them. An explicit tangent pytree/matrix (params' structure
  with a leading `n_t` axis on every leaf) is the escape hatch for
  non-all-scalar `params`, e.g. a flat `(n_p,)` design vector with
  `tangents=jnp.eye(n_p)`.
- **Scope limits, stated plainly**: uniform single-device lane only;
  "geometry parameter" means the topology-density pixel / `dz_profile` /
  `pec_occupancy_override` design channels rfx actually has, NOT a
  parametric dimension (a traced `Box` corner still raises
  `ConcretizationTypeError`) — no new geometry differentiability shipped
  here; nonlinear (Kerr) tangents are unverified at the tested chi3.
- Fail-loud fences, one INHERITED raise and two DOCUMENTED traps (see
  `jacobian_fwd`'s own docstring "Fail-loud fences" section for the full
  reasoning): non-uniform+`distributed=True` with a registered DFT-plane
  probe raises `NotImplementedError` via `forward()`'s own pre-existing
  checks, propagated unchanged through `jax.jvp`; `n_warmup` (measured
  SILENT NO-OP on the uniform lane, issue #626) and `checkpoint`/
  `checkpoint_segments` (measured EXACTLY NEUTRAL under forward mode —
  remat only pays off under reverse-mode transposition) have no raise to
  inherit and are documented traps instead, since `jacobian_fwd` is
  generic over `sim_fn` and never sees `forward()`'s own keyword
  arguments. (A fourth candidate fence, `design_mask` on the uniform
  lane, existed when this entry was first drafted; it was deleted from
  every public surface days later in issue #625 — see that entry above
  — so it is not a `jacobian_fwd` fence at all any more.)
- Provenance: the issue's own author downgraded #577 to nice-to-have on
  2026-08-06 in favour of #579 (a reverse-mode scalar-objective need);
  #579 shipped (PR #619, 2026-08-10). The PI directed this session, in
  the same window after #579 shipped, to proceed to #577 following
  #622/#582. `docs/agent-memory/` carries no forward-mode/jvp/jacfwd STOP
  or do-not-repeat entry.
- New docs: `docs/agent/recipe-design-loop.mdx` gains a `jacobian_fwd`
  section; `docs/guides/api_symbol_inventory.json` regenerated
  (`rfx.jacobian_fwd`'s parameter names `sim_fn, params, tangents,
  batch_tangents` are now pinned by `scripts/check_api_reference.py`).

### Fixed — non-uniform-mesh CPML absorber was not impedance-matched (issue #582)

- The uniform-grid material assembler extends the interior-edge
  `eps_r`/`sigma`/`mu_r` slice outward into the CPML padding so guided
  modes see an impedance-matched absorber (`rfx/api/_compile.py`,
  "equivalent to UPML"). The non-uniform-mesh assembler
  (`assemble_materials_nu`, `rfx/runners/nonuniform.py`) never had this
  step, so a structure touching the domain's interior edge (e.g. a
  dielectric slab spanning the transverse extent) saw a **different
  absorber medium per path**: on the issue's fixture, 736 CPML pad cells
  carried `eps_r=4.0` on the uniform path and `eps_r=1.0` on the NU path.
  Found via the uniform-mesh reduction anchor (#562/#568/#570): the
  `boundary="cpml"` + `subpixel_smoothing=True` combination was the only
  one of four boundary x smoothing combinations that did not reduce to the
  uniform-path solve (amplitude off 0.18%, waveform residual 1.98e-2,
  record-length-independent). Root cause was NOT the smoother — a new
  staircase-only (no smoothing) discriminator case already carried ~90% of
  the divergence (residual 7.7261e-3, measured on the pre-fix tree) — it
  was the missing pad replication, confirmed by three independent
  witnesses (input-array diff, field dump, causal A/B relocating the slab
  away from the domain edge).
  Fix: `assemble_materials_nu` now performs the same interior-edge
  replication into the CPML pads, using the NU grid's existing per-face
  `pad_{x,y,z}_{lo,hi}` bookkeeping. Guided-mode / dielectric-waveguide
  structures on a non-uniform mesh now see the same impedance-matched
  absorber the uniform path always has. PEC-bounded simulations are
  unaffected (the new step is gated on `cpml_layers > 0`, which
  `Simulation.__init__` forces to 0 for `boundary="pec"`).
  **Scope**: the fix mirrors the uniform path's replication exactly,
  including two gaps it inherits from that path — (a) for a `Box` whose
  upper face coincides with the domain face, the hi-face replication
  copies a column the rasterizer leaves at vacuum, so only the lo-side pad
  ends up matched; (b) Debye/Lorentz dispersive poles are rasterized with
  no pad-extension step, so a dispersive edge-touching material gets its
  static `eps_r` matched into the pad but not its poles. In short: this
  closes the gap for non-dispersive media, and — for boxes ending exactly
  on the domain face — for the lo-side faces. **Both gaps are now closed,
  by issue #627** (see that entry above) — this clause is kept for
  historical accuracy about what #582 itself did and did not cover.
  `tests/test_nonuniform_uniform_end_to_end_reduction.py`'s
  `subpixel-cpml` case converts from `xfail(strict=True)` to a normal
  assertion (residual 1.9890e-2 pre-fix -> 1.14e-4 post-fix); a new
  `staircase-slab-cpml` case closes the blind spot that hid most of the
  effect (residual 7.7261e-3 pre-fix -> 8.7e-5 post-fix). Pre-fix numbers
  measured on the pre-fix tree (commit 31395e0); post-fix numbers
  reproduce exactly on both trees.

### Fixed — distributed-NU pad-lane hi-x wall displacement (issue #622)

- `rfx.runners.distributed_nu.run_nonuniform_distributed_pec` and the NU
  branch of `rfx.runners.distributed_v2.run_distributed` (`distributed=True`
  with a non-uniform `dx`/`dy`/`dz` profile) produced a distributed cavity
  effectively **one cell wider** than the equivalent single-device run
  whenever the sharded x-extent needed alignment padding to divide evenly
  across devices (`sharded_grid.pad_x > 0` — e.g. any even physical cell
  count on 2 devices, since #564 made the realized node count `N+1`).
  `_build_sharded_inv_dx_arrays` re-derived the H-update inverse-spacing
  array from a padded cell-size profile instead of reading the grid's own
  `inv_dx_h`, which moved the trailing zero coefficient (the one that
  freezes the real boundary node's H term) from the real face (global
  node `nx - 1`) to the padded slab end — so the real boundary cell's H
  term stayed live. The PEC/PMC face appliers and the CPML x-hi window
  had the matching off-by-`pad_x` bug, acting on the padded slab end
  instead of the real face. Net effect on the 13 affected 2-device
  equivalence tests: final-step probe `rel_err` up to 2.0 (gate 5e-5);
  the Class F analytic-resonance test stayed green-but-degraded (2.0%
  measured error under its 5% discretisation gate, now 0.02%).
  Fix: pad cells are now structurally inert (all E/H inverse-spacing
  coefficients zero in the pad region, preserving the grid's own
  trailing zero at the real boundary node), and the PEC/PMC/CPML face
  machinery is shifted by `pad_x` to act on the real face on the last
  rank. The `pad_x == 0` lane is unchanged in behaviour class;
  coefficients now come bit-identically from the grid's own arrays. No
  gate was loosened; the Class F resonance gate (5% discretisation
  bound) is untouched even though the measured error dropped well
  under it.
- Lane visibility: `tests/test_distributed_nu_kernel.py` carried a
  module-wide `pytest.mark.gpu` from file creation (2026-04-16, for a
  test-pollution reason superseded the same day by the root conftest's
  ordered XLA_FLAGS/jax-import sequence) that put its entire 22-test
  2-device equivalence family in a lane that never ran it: every CPU
  lane deselects `gpu`, and the 1-GPU-device VESSL pod's `-m gpu` step
  only creates *virtual* extra devices on the CPU backend, so it always
  skipped there too. That is why this regression (introduced pre-#564,
  unmasked by it) shipped undetected. The marker is removed; the file
  now runs in the same fast-suite/weekly CPU shards as its sibling
  `test_distributed_nu_composition.py`, which never carried the marker.
  `scripts/vessl_gpu_suite.yaml`'s stale comment (claiming distributed
  coverage already ran in the fast suite) is corrected to match.
  `tests/test_distributed.py` (the uniform-grid distributed family) is
  in the same runs-nowhere state and is tracked separately.

### Added — `vmap_material_sweep()` batches DFT plane accumulators on the fast path (#578)

- `add_dft_plane_probe` planes now accumulate inside the `jax.vmap`-batched
  scan carry, inlining `Simulation.run()`'s rect-window DFT kernel exactly
  (same `init_dft_plane_probe` call, same `t = state.step * dt` phase
  reference — using the scan's own step-index instead would be an
  off-by-one, per-bin phase-error class bug). `VmapSweepResult` gains a
  new field, `dft_planes: dict[str, ndarray] | None`, with a leading
  batch axis (`n_batch, n_freqs, n1, n2` complex) — the SAME accessor
  key each plane's `add_dft_plane_probe(name=...)` uses for a single
  `Result.dft_planes[name]`. The sequential fallback also populates this
  field now (by stacking each swept value's `Result.dft_planes`
  accumulator), so a registered DFT plane returns data through
  `vmap_material_sweep()` regardless of which internal path a given
  `Simulation` takes — previously flux monitors and DFT planes both
  forced the sequential fallback and neither path's output carried any
  frequency-domain data at all (`VmapSweepResult` had no such field).
  Verified against `Simulation.run()` per swept element with a predeclared
  tolerance and a one-sided falsifier (a deliberate DFT-kernel sign flip
  turns the equivalence gate red on every frequency bin); see
  `tests/test_vmap_sweep_dft_planes.py`.
- Measured speedup (batched vmap fast path vs. the sequential fallback,
  same machine, same config: CPML dielectric slab + one DFT plane,
  `n_steps=300`, CPU backend, JAX 0.10.2, AMD EPYC 9654 96-core, no
  GPU available in this environment): **1.2x at `n_batch=8`, 2.0x at
  `n_batch=16`, 4.7x at `n_batch=32`** — the fast path pays one XLA
  compile amortized over the whole batch while the fallback recompiles
  per swept value, so the speedup grows with batch size in the
  documented "moderate batch sizes (5-50 values)" regime; a GPU is
  expected to widen this further via genuine parallel kernel execution
  (untested here — no GPU device in this environment).
- **BREAKING (fail-loud, not a numerics change)**: `vmap_material_sweep(...,
  return_fields=True)` now raises `ValueError` instead of silently
  returning `VmapSweepResult.final_fields=None`. `return_fields` was
  documented since this function's introduction but never implemented on
  either the fast path or the sequential fallback — `final_fields` was
  always `None` regardless of the flag. Use `sim.run()`/`sim.forward()`
  for a final-field snapshot, or `parametric_sweep()` for full per-value
  `Result` objects.
- New eligibility guards route simulations carrying MSL (`_msl_ports`),
  Floquet (`_floquet_ports`), or coaxial (`_coaxial_ports`) ports to the
  sequential fallback instead of the vmap fast path. MSL/Floquet ports
  were a genuine silent-drop gap (the fast-path scan bodies never
  launched or recorded them, so a swept sim carrying one previously ran
  the fast path missing that physics with no warning). Coaxial ports are
  not consumed by plain `Simulation.run()` at all today; this guard makes
  that failure loud (the fallback now surfaces `run()`'s existing
  `NotImplementedError` for `add_coaxial_port`) instead of silently
  taking the fast path and ignoring the port.
- Fixed a drifted module-docstring inversion (`rfx/vmap_sweep.py`
  Limitations section used to claim ports/TFSF/dispersion/waveguide
  ports/NTFF/RLC elements were "fully supported" — the opposite
  of the function docstring and the code guards, which always correctly
  listed them as fallback-triggering).
- Angle-batched TFSF sweeps remain out of scope for `vmap_material_sweep`
  — documented as a structural limitation, not a missing feature (the
  TFSF incident field is itself an in-scan auxiliary FDTD solution,
  Method B's auxiliary grid size is angle-dependent, and rfx cannot
  express a general incidence triple today). `parametric_sweep()` is the
  documented route for illumination/angle sweeps, with the pre-existing
  oblique-Bloch-TFSF-plus-DFT-planes `NotImplementedError` noted as a
  caveat there.

### Added — `rfx.observables`: differentiable DFT-plane accessor + objectives (#579)

- New `rfx.observables` module (flat-exported at top level and in the
  curated `rfx.__all__`, re-specced 210 -> 213 for this addition):
  `dft_field(names)` is a result accessor over `result.dft_planes` (a
  single name returns the raw `(n_freqs, n1, n2)` complex accumulator; a
  list of names stacks into `(n_names, n_freqs, n1, n2)` when shapes
  match, else raises `ValueError` pointing at `stack=False`, which opts
  into a `dict[name] -> array` return instead). `field_energy(names)` and
  `field_softmax(names, beta=)` are objective factories in the
  `rfx.optimize_objectives` factory style (`callable(Result) -> scalar`,
  JAX-differentiable), for `sum(|field|**2)` and a temperature-controlled
  soft-max over space + frequency respectively. All three are lane-
  agnostic: they work on `run()`/`forward()` results from both the
  uniform and non-uniform meshes, since `result.dft_planes` is a
  name-keyed dict on every one of those four combinations. See the
  module docstring for the AD-tape contents and an E/H-mixing hazard note
  (no `h_phase_correction` kwarg — apply `exp(+j*omega*dt/2)` to H-component
  planes yourself for any E x H* cross-term objective).
- New example `examples/inverse_design/field_observable_shielding.py`:
  a normal-incidence TFSF illuminates a multilayer dielectric stack;
  `field_softmax` pools two internal DFT-plane leakage monitors (the
  register-N-planes-then-pool pattern, since each plane probe is single-
  component) into one worst-case-leakage scalar minimized via
  `jax.grad` descent.

### Changed — two new fail-loud fences for registered DFT plane probes on distributed lanes (**BEHAVIOUR CHANGE**) (#579)

- `forward(distributed=True)` and `run(devices=[...])` (2+ devices) now
  raise `NotImplementedError` when any `add_dft_plane_probe(...)` plane is
  registered, instead of silently dropping it. Neither
  `rfx.runners.distributed_nu` (the only lane `forward(distributed=True)`
  currently reaches) nor `rfx.runners.distributed_v2`/`rfx.runners.distributed`
  accumulates DFT-plane fields — a registered plane's data was previously
  discarded with no warning. Remediation: drop the DFT plane probe(s), or
  use the single-device lane (`run()` without `devices=`, or a
  non-distributed `forward()`).

### Added — explicit soft-source amplitude semantics: `add_source(..., amplitude_kind=)` (#565/#571)

- `add_source` (and `add_polarized_source`) gain `amplitude_kind='field'|'current'|None`.
  Explicit kinds are boundary- and mesh-independent: `'current'` = amperes,
  `E += Cb·I/dV` on every path (the non-uniform path's native Meep-style
  convention, resolution-independent injected power, and the future default);
  `'field'` = raw E-field increment `E += w(t)` on every path. With
  `amplitude_kind='current'` the uniform and non-uniform builders produce
  EQUAL traces — the boundary-dependent cross-path factor table disappears.
- `None` (the default) keeps the legacy per-path meaning bit-identically for
  the deprecation window and emits one `DeprecationWarning` per `Simulation`
  naming this simulation's concrete legacy meaning (uniform+PEC raw add /
  uniform+CPML/UPML `Cb`-normalized add / non-uniform current). The
  open-uniform legacy contract is named by NEITHER kind; its exact migration
  is waveform amplitude ×`dV` with `amplitude_kind='current'`. Plan:
  `amplitude_kind` required in 1.8, default `'current'` in 1.9.
- Conversion lives in one module (`rfx/api/_source_semantics.py`); the
  source-building helpers declare their native coefficient (`make_source`
  'raw', `make_j_source` 'cb', `make_current_source` 'cb_over_dv') and
  dispatch on the Python-level kind — never on a possibly-traced scale value,
  so `jax.grad` paths are unaffected. 2D grids are treated as one cell deep
  (`dV = dx·dy·dz_one_cell`).
- The field is design state, so every serialization surface carries it: the
  design IR records it on `soft_sources` (schema + registry + round-trip
  test pinned on the non-default value; a lumped entry carrying a non-None
  `amplitude_kind` is refused at export since `add_port` cannot set it), and
  the config CLI accepts an optional `amplitude_kind:` key on `type: source`
  entries.

### Fixed — reproducibility-audit corrections (independent clean-room walk-through, 2026-08-09)

- `validation/tmtt_paper/README.md` no longer claims a blanket `SMOKE=1`
  1-3-min CPU path: only the taper and beam-steering scripts honor `SMOKE`;
  the notch script always runs its full ~12-15-min (fast workstation)
  multi-start workload and the gradient check ignores the variable. The
  README now carries a per-script table, the exact editable-install
  commands, and the notch example's four expected preflight advisories.
- `validation/crossval/` gains a README (case-to-config map; the "20+
  studies" claim corrected to the 18 numbered studies present) and a
  `palace/` directory carrying the Palace mesh/config setups of the
  four-solver patch study (copied from the campaign branch), so the
  paper's validation tree now includes every solver's setup.

### Changed — validation/tap_paper/ renamed to validation/tmtt_paper/

- Finishes the TAP -> T-MTT venue rename (#598 relabeled prose only): the
  paper-support directory and every live reference (bakeoff runner paths,
  multistart test path, source/test docstrings, validation/README) now say
  `tmtt_paper`. Historical CHANGELOG entries and dated inventory snapshots
  keep the old name. The directory README now opens with a five-row map of
  all paper materials (examples, crossval suite, patch record, raw campaign
  data, release tag).

### Changed — MSL AD gate objective replaced: band-mean |S21|^2, not sum_ij|S_ij|^2 (issues #530, #515)

- `test_msl_ad_fd_converged_tight`'s differentiated objective was
  `sum_ij|S_ij|^2` summed over the gate's frequency bins. For a passive
  network `S^dagger S <= I`, so that sum is bounded by `2` per frequency —
  `16` over the gate's 8 bins — and the measured loss, `16.00599`, was
  99.96% that passivity-pinned structural constant: the gradient
  differentiated only the remaining 0.037% residue, which is why the gate
  went blind (issue #527) the first time an extractor fix (PR #516) moved
  `|S|` closer to unitary, and would have gone blind again the next time.
  Replaced with band-mean `|S21|^2` (`tests/_msl_ad_objective.py`, shared
  with the AD smoke below so the two tests cannot drift apart), a
  transmission-power quantity with real dynamic range that is not
  passivity-pinned. New envelope measured on the owner platform
  (gpu-rtx4090, VESSL): rel_err `0.0026` at the gate's `h=1e-3` (worst point
  over a 5-point h-sweep: `0.0146`), new gate threshold `0.03` (derived via
  `tests._gate_policy.gate_from_envelope`, down from the prior objective's
  `0.10`). A planted defect (an issue-#483-class bug: `eps_override` frozen
  before tracing) reds this gate at rel_err `1.0000`, 33x over the new
  threshold, confirming it still discriminates a real defect.
- `test_compute_msl_s_matrix_ad_smoke_has_finite_gradient` (issue #515)
  asserted only `isfinite`/`not isnan`, which passed on a gradient of
  exactly `0.0`. ONE root cause: the synthetic Hy/Hz test fixture was
  degenerate — a spatially uniform H-field makes the Ampere-loop current
  identity cancel exactly, collapsing the multi-drive `S = B*A^-1` solve to
  `~Identity` regardless of the differentiated parameter OR which objective
  read S21. (An earlier draft of this entry claimed a second, independent
  defect — the old `Re(S21)` objective being "structurally flat" — that is
  FALSIFIED: measured with the fixture fixed and `Re(S21)` unchanged,
  `grad = -2.973442e-02`, nonzero and 6.4x LARGER than the new objective's
  `-4.681417e-03`. `Re(S21)`'s `grad = 0.0` on main was a consequence of the
  one fixture defect, not evidence of a second one.) Fixed by giving the
  Hy/Hz planes a non-uniform (linear-ramp) shape. The objective was
  ADDITIONALLY switched from `Re(S21)` to the gate's shared
  `msl_band_mean_s21_sq` — not required to fix the zero gradient, done so
  the smoke and the #530 tight gate cannot drift onto two hand-written
  reductions — and the test now asserts a measured, non-zero gradient floor
  instead of finiteness alone.

### Fixed — wire-port dead-cell preflight advisory now shares the assembler's ground-truth PEC mask (issue #544)

- The `wire_port_dead_extent_cells`/`wire_port_midpoint_in_pec` preflight
  advisory classified a rasterized cell as PEC-dead by comparing a
  computed cell-CENTER (node + half-cell Yee offset) against the PEC
  bounding box, closed interval — a different reference point than the
  real rasterization (node coordinates, half-open interval, or the
  thin-sheet nearest-node rule for a sub-cell-thick box). On the #488
  lane's committed lumped/wire↔MSL fixture this made the advisory report
  `n_live/n = 3/4` while the assembler's actual `n_live_lw` was `4`
  (measured passive-port `Z_in = Z0/4 = 12.5 ohm`, matching `n_live=4`).
  The advisory now calls `_wire_port_live_cells` against the SAME
  assembled `pec_mask` the assembler uses, so the two paths cannot drift
  apart again. Uniform meshes only — on a non-uniform mesh
  (`dz_profile`/`dx_profile`/`dy_profile`) the advisory now emits a
  `wire_port_dead_cell_classification_unavailable` note instead of either
  silently skipping the check or checking a mismatched uniform substitute
  (pre-existing NU-blindness, previously undisclosed). **Behaviour change
  for `preflight(strict=True)`**: an otherwise-clean non-uniform-mesh sim
  with a wire port now RAISES `ValueError` where it previously did not —
  the new note is a warning-severity issue like any other, and `strict`
  escalates every issue (contract-consistent, not a new exception to that
  contract). Callers that want errors-only escalation and this warning
  passed through should call `preflight(strict=False).raise_for_failure()`
  instead of `preflight(strict=True)`.

### Fixed — the #494 advisory's test coverage was bound at one point in a five-dimensional option space

- An independent mutation battery against the merged #495 suite found **40 of 67
  meaningful mutations survived** (60%), where the author's own three-mutation
  battery had found 3/3 caught — all three had landed in the one well-covered
  region. The advisory half was the weak half: 33 of 53 survived.
- The worst survivor: gating the advisory's call site on `if not self._geometry:`
  or `if normalize != "flux":` left all 40 tests green, and **both are the actual
  settings of `validation/crossval/18_wr90_iris_modematch.py`** — the script that
  motivated #494. A one-line regression restoring precisely the #494 blind spot on
  precisely the motivating script was invisible, because the only end-to-end test
  used `normalize=False` with an empty domain.
- Coverage added for each unbound branch point: `normalize` in `(False, "flux")`
  with a PEC obstacle registered; a z-propagating port pair (the axis was
  previously hardcodable to `"x"` with no test noticing); a multimode port binding
  the lowest-cutoff mode (fence (c) had zero coverage); a two-axis sim asserting
  two warnings and a shared-cutoff cube sim plus a two-mode single-axis sim, which
  together bind both components of the `(axis, cutoff)` dedupe key; a mirrored
  per-face test with the thin face on `hi`; and the `port_reference_sims` junction
  path, which must not lose the band-edge check to its band-centre sibling.
- Every number the advisory prints is now recomputed independently from the port
  geometry and asserted, including the ripple ladder it quotes as evidence. Six of
  seven were previously unasserted — notably its only actionable output, where a
  mutant advising `cpml_layers` of 1 instead of 13 passed. The warning category is
  pinned via `pytest.warns(UserWarning)`, since `DeprecationWarning` would be
  suppressed under Python's default filters and previously survived.
- `0.5` is now the module constant `_FAR_PORT_LAMBDA_G_FRACTION`, used both to
  compute the requirement and to render the message. It was two independent
  literals, so a factor change produced a message stating 0.5 while enforcing
  something else; the factor was also only pinned to a 1.6x window (0.394-0.630).
- The `#493` characterization table gains an independent oracle. A coherent author
  could previously mutate the rasterization rule AND re-pin `_NOMINAL_EXCESS` in
  one commit and stay green — which silently falsified the ambiguity docstring and
  left the asymmetric branch of the symmetry test dead in every parametrization.
  The expected excess is now derived from the retreat mechanism and cross-checked
  against the table, and a guard asserts both the one- and two-cell cases occur.
- Re-verified: all 18 previously-surviving mutations are now caught, including the
  two that survived the first round of fixes.

### Changed — the #493 electrical-dimension identity is scoped to the transverse direction

- The `Box` / support-matrix text from #493 stated the electrical dimension as the
  span between innermost zeroed planes, `(n_open + 1) * dx`, in a way that could be
  read as holding on any axis. It is now explicitly scoped to **transverse** to the
  propagation direction — aperture and guide widths, anything setting a cutoff —
  where an independent refit of 16 committed single-iris configurations across two
  meshes pins the realized aperture to within 1/20 of a cell of it.
- The docs now warn **not** to carry that identity into the propagation direction.
  An obstacle's electrical *thickness* is set by field interaction with the
  discontinuity rather than by a cutoff condition, and is measured to fall between
  `t_cells * dx` and `(t_cells - 1) * dx` — so neither integer rule holds, and the
  residual is a fixed per-face offset rather than a discretization error that
  shrinks with `dx`. The longitudinal convention is recorded as **not settled**:
  treat it as an unknown of order half a cell and fold that sensitivity into the
  reported envelope instead of adopting a rule. (Measured during the stage-S3 /
  #499 review; no committed record carries it yet, so it is documented as a caution
  rather than as a number to build on.)
- Odd `(cells - d_cells)` parity is now presented as a **fork rather than a dead
  end**: change `dx`/the aperture so the parity works, or place fins asymmetrically
  on purpose and accept a recorded half-cell offset instead of rounding the aperture
  — the quantity that sets the cutoff — to the wrong parity. Neither option is
  recommended, because the cost of the offset has not been measured. What is
  required is that the offset be recorded and **representable by the comparator**,
  since an off-centre aperture compared against a centred oracle silently becomes
  comparator error. Also fixes a typo in the merged #493 text, which said the
  odd-parity opening is "one cell wide" where it is one cell **wider**.

### Added — thin-absorber advisory on every uniform waveguide S-matrix path (#494)

- `compute_waveguide_s_matrix` documents a "Far-port discipline" requiring an
  absorber `>= ~0.5 * lambda_g`, but nothing checked it on the plain two-port
  path: the sibling `_warn_junction_cpml_thickness` advisory runs only on the
  `port_reference_sims` junction path, and the functional entry points run no
  `sim.preflight()`. A gated revision of crossval case 18 therefore shipped a
  0.30-`lambda_g` stack in silence and the absorber, not discretization, set
  the reported accuracy envelope. A new in-method ADVISORY (warning, never
  raises) now fires whenever the absorber on a port's propagation axis is
  thinner than `0.5 * lambda_g`, quoting both the configured thickness and the
  requirement.
- Evaluated at the **lowest** measured frequency, where `lambda_g` is longest
  and the `cpml_layers=16` default weakest, because `lambda_g` diverges toward
  cutoff — the existing junction advisory uses band centre, which is what let
  the 0.30-`lambda_g` stack pass. The message reports the measured ripple
  ladder (0.0706 at 0.30 `lambda_g`, 0.0366 at 0.50, 0.0093 at 0.75) so
  `0.5 * lambda_g` reads as a floor, not a target, and names the `cpml_layers`
  values for both 0.5 and 0.75.
- Deliberately a LOWER bound, with three fences: silent when the band starts at
  or below cutoff (`lambda_g` undefined; the `port_freqs_below_cutoff` preflight
  owns that), silent when the propagation axis carries no absorbing face, and
  evaluated on the port's lowest-cutoff mode (shortest `lambda_g`, least
  demanding). Silent on a non-uniform mesh, where `cpml_layers * dx` is
  ambiguous under a graded profile. Respects per-face
  `Boundary.lo_thickness` / `.hi_thickness` overrides. No physics changed and no
  gate or tolerance was touched. Firing and non-firing gates in
  `tests/test_waveguide_geometry_hygiene.py`.

### Changed — node-rasterization convention documented where obstacles get drawn (#493)

- `Box`'s volume branch is half-open `[lo, hi)` over NODE coordinates, so a box
  whose corners land on node planes occupies nodes `i..k-1`: the realized extent
  is one cell short of the drawn extent, entirely at the `hi` face, which also
  displaces a SINGLE box by `dx/2`. For a PEC obstacle those nodes are where
  tangential `E` is zeroed, so this is an ELECTRICAL dimension error, measured
  between the innermost zeroed planes as `(n_open + 1) * dx` — the measure that
  reproduces the guide's own `a = cells * dx` exactly.
- A facing pair does NOT simply inherit that per-box displacement, because its
  two interior faces are different corner types: the lo fin's is a `hi` corner,
  which half-openness always drops (so it always retreats one cell), while the
  hi fin's is a `lo` corner, which is kept unless float32 rounding puts the node
  just below it. One retreat gives `d + dx` with the opening **asymmetric**
  (centre `dx/2` low); two retreats cancel and give `d + 2*dx` **centred**.
  Which one occurs is **not predictable from the nominal dimensions**: measured
  on WR-90 at both a/30 and a/60, 7.620 mm and 18.288 mm give `d + dx`
  off-centre while 12.192 mm gives `d + 2*dx` centred. A half-cell offset toward
  the metal retreats both faces by construction rather than by luck, giving
  `d + 2*dx` deterministically at every aperture — the drawing case 18's blocked
  revision used, and why re-comparing it against `oracle(d + 2*dx)` collapsed
  every row. This inflated PR #480's `|S11|` error against an analytic
  mode-matching oracle by 4-6x. Because it scales with `dx` it mimics
  first-order convergence, and on a resonant structure it shifts the passband
  instead of widening a magnitude tolerance.
- The float32 effect is sharper than a single ULP of the corner value: node
  coordinates are themselves double-rounded, `f32(f32(i) * f32(dx))` in
  `_grid_coords`, while a caller's corner is computed in float64 and cast once.
  An f64 reconstruction of the nodes disagrees with production on 30 of 31 WR-90
  nodes by up to 1.1e-9 m (1e-6 of a cell) — enough to move the footprint by a
  whole cell, which is precisely what separates 12.192 mm from the other two
  apertures.
- The recipe therefore has two conditions: interior corners on **cell
  midpoints** (rounding-independent) AND the metal depth an exact number of
  cells, i.e. `(cells - d_cells)` even. Under both, the realized opening equals
  the nominal one exactly (100% of ~50k even-parity combinations measured). At
  odd parity a symmetric opening of that width is not representable on the grid
  and costs one cell however it is drawn — a representability limit, not a
  rasterization defect.
- All of the above is now stated in the `Box` / `Shape.mask_on_coords` /
  `rasterize` docstrings and in the waveguide setup restrictions of
  `docs/guides/sparameter_support_matrix.md`. Characterization tests pin the
  arithmetic in `tests/test_waveguide_geometry_hygiene.py`, deriving node
  coordinates from a real `Grid` rather than an f64 `arange` — an earlier
  revision of those tests used f64 coordinates and consequently pinned
  `d + dx` at every aperture, which is wrong at 12.192 mm.
- **The rasterization rule itself is unchanged** — it is deliberate and other
  paths depend on it. No advisory was added: the predicate floated in #493
  (fire when the rasterized opening differs from the drawn opening by >= 1 cell)
  gives an AMBIGUOUS reading. At `d = 7.620 mm` the defective nominal drawing
  and the correct midpoint recipe both read +1 cell, so +1 cell cannot support a
  defect conclusion — and +1 cell is the common case. The defect is
  `realized != INTENDED`, and the intended dimension is never communicated to
  the simulator, so it is not recoverable from geometry. Pinned by
  `test_drawn_vs_realized_gap_is_ambiguous_between_correct_and_defective`.

### Fixed — MSL `eps_override` gradient: 13.7% converged deficit attributed and removed

- The auto-`eps_r_sub` launch fixture (mode profile / sigma loading / source
  amplitude) sampled the possibly-overridden materials through
  `stop_gradient`, so finite differences and `jax.grad` differentiated
  DIFFERENT functions. The fixture now derives from the registered
  materials on both sides; the converged f64 AD-vs-FD referee moves from
  13.7% to 0.011% (rel_err 0.000110; num_periods=20, full extraction). Regression-locked
  by a committed mini-referee. (#483, #486)

### Changed — DFT accumulator dtype follows the x64 state

- Point/plane DFT accumulators were hardcoded complex64, breaking the scan
  carry the moment x64 was enabled — no f64 gradient referee could run.
  Both constructors now follow `jax.config.x64_enabled`; the f32 path is
  bit-identical. (#484, #477)

## [1.6.7] - 2026-07-28

### Added — open-domain oblique RCS (Method-B TFSF) with calibrated absolute sigma

- `compute_rcs(theta_inc != 0)` now routes a compact scatterer through the
  open-domain Method-B oblique TFSF (`ez` polarization, uniform grid;
  other combos fail loud). The specular direction is validated (reflection
  law tracked across 0/20/40 deg); absolute sigma is normalized by the
  MEASURED 1D-aux incident spectrum and validated against a PO
  uniform-aperture oracle to +0.9 dB at the gate grid — with a measured
  2.4 dB resolution sensitivity, so treat absolute oblique sigma as a
  +/-2 dB-class number (docstrings carry the full envelope). A 4-plane
  vacuum guard now protects all Method-B TFSF boundaries in both the
  runner validator and `compute_rcs`. (#461, #474, #471)

### Added — differentiable plane-wave design lane extensions

- Oblique plane waves in `run()` via opt-in complex-Bloch TFSF (#414) and
  the differentiable `forward()`+TFSF uniform lane (#415); complex Fresnel
  reflection helper `rfx.probes.fresnel_reflection_coefficient` (#418,
  #419); `compute_rcs_jax` differentiable far-field post-processor (#421);
  `mu_r_override` as a differentiable DoF on `forward()` (#449); analytic
  TMM-gated multilayer/magnetic RAM inverse-design batteries (#445, #449).

### Changed — MSL auto `n_probe_offset` solves BOTH clearances (**BEHAVIOUR CHANGE**)

- The auto default previously sat at the upstream (fringing) lower edge and
  the reflector advisory pushed probes the wrong way on short feeds. At
  `compute_msl_s_matrix` time the auto offset now solves the compliant
  interval: unchanged when no downstream reflector exists, midpoint when
  one bounds it, and a loud "mutually unsatisfiable" warning (upstream-
  priority fallback) when the feed is too short. Explicit offsets are never
  adjusted. Library-internal settling-witness probes no longer pollute the
  preflight record (measured 14 -> 4 advisories/run) and no longer
  double-fire the #332 tail advisory next to `settling_db`. (#478,
  #469, #470)

### Added — CAD import (`rfx.MeshShape`) — STL/OBJ/PLY + STEP

- Voxelized mesh import with occupancy caching; STEP via the `cad` extra.
  Top-level export + docs; NOT differentiable (host-side rasterization —
  traced coordinates raise). (#453, #456, #473, #358, #467)

### Added — Kerr nonlinearity validated at absolute magnitude

- D-based self-consistent Kerr update (fixes the reactive-operator defect)
  and a TRUE-CW TFSF waveform enabling an absolute-magnitude SPM oracle:
  measured phase-shift ratio 0.955 +/- 0.03 vs closed form, independently
  confirmed against Meep. (#440, #441, #448, #450, #452, #446, #437)

### Added — design-document interop (`rfx.interop`) with an openEMS emitter

- `rfx-design-ir/v1`: dump a Simulation to a validated design document and
  emit a runnable openEMS script (mesh/boundary/port semantics carried
  explicitly; refuses non-representable constructs rather than
  approximating). Auto MSL offsets are recorded as their RESOLVED frozen
  values. (#463, #472, #478)

### Added — `until_decay` radiated-flux stop criterion

- `run(until_decay=..., radiated_flux_box=...)` stops on radiated energy
  flux instead of the domain-energy floor (which a static charge pins high)
  — both uniform and non-uniform lanes; static-floor advisories route to
  the flux-stop remedy. (#442, #443, #454, #388)

### Fixed — 7 RF-core extractor defects (dual-audit sweep)

- Includes the multimode-flux |S| half-step co-location in
  `_modal_net_power` (reactive->real leak), a sub-cutoff beta scale factor,
  reference-plane metadata, and Floquet fail-loud API validation
  (n_modes>1 / TM rejected explicitly). (#459, #404)

### Added — real-structure cross-validation campaign (governed)

- PEC-sphere and dielectric-sphere Mie ka-sweeps, rectangular PEC cavity
  vs exact Pozar, Sheen-1990 LPF vs openEMS with a Palace-FEM referee
  (three-solver doublet finding), and an RT5880 patch — all registered in
  the crossval manifest with honest roles and fail-closed evidence chains.
  (#462, #475, #476)

### Changed — MSL-FD-TIGHT gradient gate: ownership, comparator envelope, and an f64-determined AD finding

- The converged AD-vs-FD gate is GPU-lane-owned; its FD comparator carries
  an f32 evaluation-noise envelope (+/-3-5%) that straddles the 0.10 gate
  across platforms while AD is platform-stable. Docstring records the
  measured numbers; gate value unchanged. (#479, #477)
- An f64 referee (enabled by making the DFT accumulators follow the x64
  state) subsequently DETERMINED the question the noise left open: the MSL
  `eps_override` AD gradient reads ~13.7% below the clean f64 finite
  difference on this objective (true derivative ~ -0.244 by three agreeing
  estimates). Until issue #483 attributes the mechanism, treat
  `compute_msl_s_matrix(eps_override=...)` gradients as a +/-14%-class
  quantity. Forward S-parameters are unaffected. (#483)


### Changed — `compute_msl_s_matrix` returns a passivity-enforced S by default (**BEHAVIOUR CHANGE**)

- The returned `MSLSMatrixResult.S` now satisfies `||S(f)||_2 <= 1` at every
  frequency: singular values are clipped to the passive bound per bin
  (nearest passive matrix in spectral norm). Nothing is discarded — the raw
  extraction is preserved in the new `S_raw` field, the per-bin clip amount
  in `passivity_correction`, and a warning names the touched bins. Bins with
  a large correction are measurement artifacts (check `reliable` /
  `settling_db`); the projection bounds them, it does not certify them. Pass
  `enforce_passivity=False` for the previous raw behaviour. EXEMPTION: on
  the `eps_override` channel (traced or concrete) the projection is skipped
  so finite-difference and `jax.grad` objectives see the same raw function.
- New ring-down settling witness: `MSLSMatrixResult.settling_db` records the
  worst end/peak `Ez^2` ratio (dB) over all port probe planes per driven
  run, and a run above −40 dB warns loudly — a fixed `num_periods` record
  that ends while the structure is still ringing produces truncation-artifact
  S-parameters (measured: column-power poles up to ~1.8e3 on the Sheen-1990
  LPF at `num_periods=20`).

### Changed — cross-validation scripts moved to `validation/crossval/` (repo layout only; no behaviour change)

- `examples/crossval/` moved to `validation/crossval/` so that `examples/`
  holds only the user-facing learning path (see `examples/README.md`) while
  internal verification fixtures live under `validation/` next to
  `validation/tap_paper/` and `validation/research/`. The manifest
  (`validation/crossval/manifest.json`), CPU runner, CI workflows, VESSL
  YAMLs, and all docs references were re-pointed in the same commit. A
  tombstone README remains at `examples/crossval/README.md`. Script contents
  are unchanged except their own path self-references.

### Fixed — wire-port termination over LIVE cells only (issue #318; field-changing for ports with extent cells inside PEC)

- A wire port whose rasterized extent includes cells inside a PEC conductor
  ("dead" cells — e.g. the top cell of a vertical feed ending inside the
  trace) previously counted those cells in its per-cell sigma distribution,
  drive injection, and `Z0/n_cells` wave normalization. Dead cells carry no
  port current (measured on the issue-#313 thru: `|I_dead|/|I_mid|` =
  0.003–0.03), so the physical series termination came out `Z0*(n_live/n)`
  — 33.3 ohm instead of 50 on the canonical thru. Dead cells are now
  excluded everywhere: sigma is `n_live*d_par/(Z0*dA)` at live cells only
  (series termination exactly `Z0` by closed form), the drive is
  `V/n_live` over live cells, and the wave normalization is
  `Z0c = Z0/n_live`. A port whose extent lies entirely inside PEC now
  raises `ValueError`.
- The port-cell PEC-mask/occupancy clearing is likewise scoped to live
  cells: the old all-cells clearing punched a one-cell in-plane
  conductivity hole in the DUT conductor at each dead cell (witnessed:
  in-plane `|Ex|` at the dead cell 1.5 -> 0.0 after the fix; the port's
  Ez chain is unaffected under the thin-sheet rule).
- **No change for ports with all extent cells live** (`n_live == n`
  degenerates every formula to the previous one): clean fixtures,
  goldens, and the `run()`/`forward()` contract were verified bitwise
  identical. Canonical-thru movement (measured): max in-band |S11|
  0.130 -> 0.086 (mean 0.076 -> 0.051); |S21| 0.524–0.668 ->
  0.546–0.610; reciprocity `|S21−S12|` 1.04e-2 -> 0.75e-2. The
  post-fix |S11| is a V-shaped curve (0.033 at 4.5 GHz rising to 0.086
  at 7 GHz and 0.056 at 3 GHz) — this is rfx's *measured* feed-post
  reflection, not a residual termination error: the two wire ports are
  1 mm vertical feed posts (~0.26 nH each) whose reactances interfere
  across the 16 mm line, giving a reflection null near 4.5 GHz
  (H_FIXTURE re-diagnosis, decisive on three independent witnesses).
  The pre-registered `max|S11| < 0.06` falsifier missed only because it
  neglected feed-post reactance (a modelling omission in the
  prediction, not a fix failure). The lumped/wire V·I battery and the
  reference-plane battery gates are re-baselined onto this measured
  physics in the same change.
- The `wire_port_dead_extent_cells` preflight advisory now states the
  post-fix semantics (dead cells excluded; the `Z0*(n_live/n)`
  termination is cited as the historical pre-fix behaviour). **Update
  (issue #544, see the Unreleased section below)**: the advisory's own
  live/dead cell *counting method* — separate from this wording fix —
  was itself found to disagree with the assembler's actual `n_live_lw`
  on some fixtures (a bounding-box-vs-cell-center approximation drifting
  from the real node-based rasterization); it now shares
  `_wire_port_live_cells` with the assembler directly.

### Added — opt-in reference-plane port waves for the wire S-matrix off-diagonals (`add_port(reference_plane_cells=)`, issue #313)

- `add_port(..., extent=..., direction=..., reference_plane_cells=N)` registers
  two line-V/I reference planes at N and 2N cells outboard (into the DUT) and
  computes the off-diagonal `S[i, j]` (both ports opted in) from the plane
  waves: forward/backward split with the **measured** two-plane line impedance
  and phase-only de-embedding with the **measured** per-bin beta. The Phase-0
  closed-box flux referee showed the port-cell wave pair does not conserve
  power (the port plane is near-field dominated) while the plane waves close
  the power budget; the plane path removes the drive-side |S21| deflation
  kappa(f) = 1.49–1.86 of issue #313.
- Default (`reference_plane_cells=None`) is **byte-identical** to the shipped
  behaviour, and the diagonal `S_jj` always stays on the byte-frozen legacy
  path either way — `forward()` / 1-port S11 results are unaffected.
- Uniform-`run()`-lane only: the non-uniform and subgridded lanes raise
  `NotImplementedError` with the opt-in set (the distributed lane does not
  support `compute_s_params` at all, warned-unsupported).
- Placement guidance: put BOTH planes (N and 2N cells) >= 10 cells from every
  port (Phase-0 pre-registration rule). N=3 planes measured near-field
  contamination on the canonical thru (Zc Im/Re to 8.2%, beta/(w/c)
  1.16–1.20, closed-box-referee |S21| residual -3.1%); a preflight advisory
  fires below N=10, a UserWarning fires at extraction when the measured Zc
  shows the contamination signature, and a wrapped/non-positive measured
  beta fails loudly instead of de-embedding with it.
- Honesty framing (matching the docstring): the supporting numbers are from
  the canonical 16 mm thru battery (dx = 0.5 mm, 2026-07-10) — a single
  geometry class, not an external cross-solver validation; the diagnostics
  (measured `zc`, `beta`, at-plane wave pairs) are exposed for inspection
  rather than asserted as validated beyond that envelope.

### Fixed — `forward()` / `optimize()` are now wrappable in an outer `jax.jit` (blind-docs finding)

- `_assemble_materials` decided whether to return the PEC mask via
  `bool(jnp.any(pec_mask))` — a host-side boolean conversion on a
  geometry-derived device array. Fine eagerly and under a bare `jax.grad`, but
  when the whole `forward()` was wrapped in an **outer `jax.jit`** the
  geometry-derived `pec_mask` became a tracer and the conversion raised
  `TracerBoolConversionError` deep in material assembly — so a user JITing their
  optimization step (a natural performance move) crashed with an opaque error.
- The eager path keeps the exact `jnp.any` test (**bit-identical** results,
  verified across a PEC-cavity / interior-PEC / dielectric-CPML / thin-PEC /
  forward-override / degenerate-empty-mask digest gate); only under a trace,
  where a host bool is impossible, does it fall back to a static Python
  `has_pec` predicate. Now `jax.jit(loss)` and `jax.jit(jax.grad(loss))` over a
  `forward(eps_override=...)` objective both run and match the un-jitted values.
- Locked by `tests/test_forward_outer_jit_traceable.py`.

### Changed — empty-window gradient + `add_lumped_rlc`-is-not-a-port pinned in docs (blind docs-only audit, doc-pin R2-STOP)

- A blind, docs-only agent (validated across two model families) trusted a
  finite-difference-verified gradient whose loss was `~1e-7` — orders of
  magnitude too small — because the reflection it minimized never landed in
  `minimize_reflected_energy`'s late-time split window. AD and FD agreed because
  both differentiate the **same empty window**; the docs prescribed the FD check
  as the trust ritual but never warned it cannot detect an empty observable.
- Documented the round-trip **precondition** for `minimize_reflected_energy`'s
  split window (docstring + [Inverse Design](/rfx/guide/inverse-design/)) and the
  "a passing finite-difference check is necessary, not sufficient — check the loss
  **magnitude** against a physical expectation" caveat in
  [Autodiff & Adjoint](/rfx/guide/autodiff-adjoint/) and
  [Gradient Behavior](/rfx/guide/gradient-behavior/).
- Clarified in [Sources & Ports](/rfx/api/sources-ports/) that
  `add_lumped_rlc(...)` is a circuit element, **not** a reflection-referenced
  port: a single-cell lumped element reports its own self-interaction, not a
  `Z0`-referenced `S11`; drive with `add_port(..., impedance=Z0)` to measure a
  load reflection coefficient.
- R2-STOP on a preflight advisory: the split-window premise depends on the
  geometry's round-trip time and the run length, which `preflight()` cannot know,
  so the guidance lives in the docs, not a call-time gate. Locked by
  `tests/test_empty_window_gradient_caveat_docpin.py`. No physics or preflight
  behaviour changed.

### Changed — lossless-cavity infinite-Q pinned in the harminv docstrings (LLM-naive-usage audit item #4, doc-pin R2-STOP)

- Measuring a resonator Q via `harminv` / `harminv_from_probe` on a **lossless
  closed (PEC) cavity** gives a window-length artefact (infinite physical Q),
  not physics — the frequency is fine, the Q is not. Documented with a
  `.. warning::` in both docstrings (add a realistic `sigma`/`tan_delta` loss
  or use an open CPML boundary before trusting a Q).
- R2-STOP on a preflight advisory: a "closed + lossless + soft source" trigger
  has NO clean call-time discriminator — legitimate lossless-cavity resonance
  sims (e.g. the NU stage-1 cavity physics gate, memory-planning cavities) use
  the same config and would be false-alarmed. Per "a false-alarming preflight
  erodes trust worse than the silent gap," the guidance lives in the API docs,
  not in `preflight()`. No physics or preflight behaviour changed.


### Changed — RCS bistatic-pattern validation scope pinned in docs (LLM-naive-usage audit item #2, doc-pin R2-STOP)

- `compute_rcs` returns a full bistatic `rcs_dbsm` / `rcs_linear` pattern, but
  only the `monostatic_rcs` (backscatter) bin is cross-validated against the
  exact Mie series (~0.06 dB at ka~1). At the auto-placed default NTFF box
  (`ntff_offset=1`, ~1 cell off the TFSF boundary, deep in the reactive near
  field) the off-backscatter bins can be several dB to ~20 dB off — the
  committed ka~1 PEC sphere shows a spurious forward-oblique lobe near 25-55 deg
  scattering angle measured ~10 dB high vs Mie. This was already recorded
  non-gated in `tests/fixtures/rcs_sphere_mie/`; the audit asked whether it
  should surface at call time.
- R2-STOPPED to a doc-pin (no runtime advisory added). `monostatic_rcs` is
  computed at the exact backscatter direction INDEPENDENT of the
  `theta_obs`/`phi_obs` grid, and every validated monostatic test passes a full
  observation grid, so there is no call-time signal that separates
  "monostatic-only" from "bistatic" intent to key an advisory on without
  false-alarming those tests (the same trap as item #3). Wiring the
  `Simulation` NTFF near-field guard into `compute_rcs` would likewise fire on
  every validated monostatic RCS test.
- Pinned the caveat in the `compute_rcs` / `RCSResult` docstrings, the module
  docstring, and `docs/public/guide/farfield-rcs.md`. Verify-first finding also
  recorded: enlarging `ntff_offset` alone does NOT close the oblique gap at test
  scale (offset 1->2 leaves the 25-55 deg error ~10 dB and worsens backscatter;
  a larger domain did not help), so the docs steer users away from that dead
  end. No physics changed; no gate touched. Doc-contract witness in
  `tests/test_rcs_bistatic_caveat_docpin.py`.

### Added — waveguide S-matrix soft over-unity advisory (LLM-naive-usage audit item #5)

- `compute_waveguide_s_matrix(normalize=False)` now emits a SEPARATE, humble
  ADVISORY (warning, never raise) when a passive extraction's max column power
  lands in `(2.25, hard_limit]` — above the documented single-run overshoot
  envelope but below the `passivity_tol` extractor-broken hard limit. On the
  loose `normalize=False` tol (2.0 → column-power limit 3.0) the window
  `(~2.0, 3.0]` was previously unguarded: a validated PEC short sits at column
  power ~2.0 there (a documented Yee/near-cutoff artifact, `|S21|≈1` not
  cancelled by the single-run decomposition), so the existing self-check
  (`_warn_if_nonpassive_smatrix` in `rfx/api/_sparams.py`) stayed silent for a
  passive result materially above that envelope. A real coarse-mesh (`dx=2 mm`)
  WR-90 PEC short landing at column power ~2.51 reproduced the silent gap.
- The floor is column power 2.25 (`|S| ~ 1.5` for a 1-port): above the ~2.0
  documented envelope AND the committed `normalize=False` PEC short (column
  power ~2.00, `test_pec_short_s11_magnitude` /
  `test_normalize_aware_tol_tolerates_documented_overshoot`) with margin. The
  window is EMPTY on the tight-tol path (`normalize='flux'`/`True` → tol 0.10 →
  hard limit 1.10 < 2.25), so the advisory only fires for `normalize=False`. No
  physics changed; the `tol=2.0` hard threshold and every committed gate are
  untouched. Message wording is distinct from the hard `UNRELIABLE` error.
  Witness + false-positive gates in `tests/test_sparam_passivity_guard.py`.

### Changed — pec-open-radiator advisory kept NTFF-gated (LLM-naive-usage audit item #3, R2-STOP)

- The audit asked whether `_validate_cfg_pec_boundary_open_structure`
  (`rfx/api/_preflight.py`) should be ungated from its `self._ntff is not None`
  condition so an open radiator read via a near-field probe / S11 alone also
  warns. R2-STOPPED after investigation: NTFF is the SOLE radiation-intent
  signal on the `Simulation` config, so a source (and/or a finite PEC object)
  inside a `boundary="pec"` domain is config-IDENTICAL between an open radiator
  that mistakenly used PEC and a legitimate closed cavity / internal-PEC
  numerics test (e.g. `test_adi.py::test_simulation_adi_internal_pec_geometry_masks_ez`,
  `test_conformal.py::test_api_conformal_flag`,
  `test_extract_s_matrix_pec_mask.py`). Any broadening that caught the footgun
  would false-alarm that legion of valid closed-structure sims, and a
  false-alarming preflight erodes trust worse than the silent gap. The advisory
  is unchanged; the decision is locked by regression tests in
  `tests/test_preflight_false_positives.py` (NTFF radiator still warns; valid
  closed structures stay silent).

### Added — lumped R/L/C component values as a differentiable design variable (WP 4-E)

- `Simulation.forward()` now processes `add_lumped_rlc(...)` elements on the
  uniform single-device lane and gained a keyword-only `rlc_values_override`
  injection surface, so a lumped element's R/L/C can be a `jax.grad` design
  variable. `LumpedRLCSpec` stores plain floats, so a component value enters
  the AD tape AS a tracer via
  `forward(rlc_values_override={element_index: {"R": R, "C": C, "L": L}}, ...)`;
  a missing index or key falls back to the registered float. A traced meta
  builder (`rfx.lumped.build_rlc_meta_traced` / `setup_rlc_materials_traced`)
  drops the `float()` coercions and keeps element topology decisions static, so
  `jax.grad(|S11|^2)` w.r.t. R and C is finite, nonzero and FD-consistent
  (rel < 5%, scoped x64) — gated in `tests/test_lumped_rlc_ad.py`.
- **Fixed a silent no-op**: previously a registered `add_lumped_rlc` element was
  IGNORED by `forward()` (the differentiable lane never iterated
  `self._lumped_rlc`), so a sim's `forward()` |S11| was byte-identical with or
  without the element. `forward()` now correctly reflects the element even
  without an override. Because `run(compute_s_params=True)` extracts its
  S-matrix through the same `_forward_from_materials` path, that lane was
  IGNORING a co-located RLC too — it now reflects it as well (witness-locked in
  the same test file). The main field solve `run()` was always correct and is
  BYTE-IDENTICAL (the concrete `build_rlc_meta` / `run(lumped_rlc=...)` path is
  untouched; golden regression-locked).
- The lumped/wire port S11 DFT accumulators
  (`forward(port_s11_freqs=...)` / `run(compute_s_params=True)`) now derive
  their complex dtype so they promote under a scoped `jax_enable_x64` instead of
  tripping the `lax.scan` carry-dtype contract — this makes the wave-decomp S11
  AD lane (WP 4-E and the pre-existing `eps_override` channel) usable under
  scoped x64. Byte-identical with x64 off (accumulators stay complex64).
- Scope: uniform single-device forward lane only. `rlc_values_override` on the
  non-uniform / distributed forward lanes raises `NotImplementedError` (those
  lanes do not process `add_lumped_rlc` in `forward()`).

### Added — opt-in multi-start / best-iterate / step-clamp knobs on `optimize()` (WP 4-C)

- `optimize()` gained four keyword-only, DEFAULT-OFF parameters promoted
  from the MSL open-stub inverse-design example (`_multistart_adam`, issue
  #171): `n_starts` (best-of random restarts), `best_iterate` (return the
  lowest-loss visited iterate instead of the last), `step_clamp` (bound the
  L2 norm of each Adam latent update), and `seed` (reproducible restart
  inits).
- Defaults (`n_starts=1`, `best_iterate=False`, `step_clamp=None`) reproduce
  the previous single-run Adam loop BYTE-for-BYTE — start 0 always uses the
  caller's `init_latent`, extra restarts draw i.i.d. standard-normal latents
  from `seed`, and the legacy NaN/Inf gradient guard (warn + return the last
  finite design) is unchanged. A bit-identity gate against a verbatim copy of
  the legacy loop is committed in `tests/test_optimize_multistart.py`.
- The step-clamp is GENERIC (an exact whole-step L2-norm rescale, direction
  preserved) rather than the example's physical-length bisection, which stays
  specialized in the example. Multi-start fails closed (raises) only when
  every restart is non-finite; a single start keeps the fail-soft contract.
- Scope is optimizer-loop plumbing only — no objective or numerics change.
  Extra restarts help only on a genuinely multimodal loss surface.

### Fixed — `compute_rcs` monostatic extraction pointed at −z broadside, not backscatter (issue #276)

- `RCSResult.monostatic_rcs` was argmin-extracted at (θ=π, φ=0), which under
  the far-field convention (`r_hat = [sinθcosφ, sinθsinφ, cosθ]`) is the −z
  BROADSIDE direction — not the −x backscatter of the +x TFSF incidence
  (θ=π/2, φ=π). Measured on a validated exact-Mie PEC-sphere falsifier
  (ka≈1.0, dx=λ/40): the shipped number was 10.06 dB off Mie; the same run
  re-extracted at the true backscatter direction is 0.06 dB off.
- The backscatter direction is now derived from the incident propagation
  unit vector (b_hat = −k_hat) and the far field is evaluated EXACTLY at
  that direction on the already-accumulated NTFF data — no observation-grid
  snapping (the default φ grid `[0, π/2]` does not even contain φ=π).
  `monostatic_rcs` is therefore independent of `theta_obs`/`phi_obs` and is
  now always computed (previously `None` for empty observation grids).
- New committed evidence + gate: `tests/fixtures/rcs_sphere_mie/`
  (exact-Mie oracle self-validated by four rfx-independent witnesses —
  Rayleigh 9(ka)^4, GO→1, term-doubling convergence, bistatic-bridge — plus
  fixture JSON with the full H-plane trace) and
  `tests/test_rcs_mie_fixture.py` (live recompute at the committed
  resolution, gate |Δ| ≤ 1.0 dB vs Mie; measured 0.06 dB). Claim scope is
  MONOSTATIC magnitude at the committed resolution only — the same run
  shows a NON-GATED spurious forward-oblique lobe (25–55°, ~10 dB high vs
  Mie; TFSF/NTFF forward-face contamination suspected) and a ~1.6 dB
  forward-scatter delta, documented per-angle in the fixture rather than
  hidden.

### Fixed — `estimate_ad_memory` counts the live-segment rematerialization tape (#277)

- Both segmented paths (`checkpoint_segments` and `checkpoint_every`) now model
  peak reverse-mode AD memory as
  `(2 x active_segments + live_tape_steps) x field_bytes + forward + ntff`.
  The previous formula (issue #39) counted only the segment-boundary
  carry + cotangent term; during the backward pass each segment is
  rematerialized as a unit, so one segment's per-step field tape
  (`n_steps // checkpoint_segments`, or `checkpoint_every` on the padded
  non-uniform path, capped at the active step count) is resident on top of it.
  At segment counts far from `sqrt(n_steps)` the old estimate under-counted
  peak by up to `~(segment_len / (2 x segments))x` (a hypothetical
  far-from-sqrt(N) knob: ~272x at `n_steps=6999`, `checkpoint_segments=3`;
  the committed desk ladder's sqrt(N) path shifts only ~1.5x); the VESSL 369367233509
  inverse-design gradient peaks (5.84 GB at chunk=100/n_steps=10000 vs the old
  ~3.1 GB estimate) confirm the missing term.
- Planning numbers shift accordingly: `ad_segmented_gb` grows, is minimized
  near `sqrt(2 x n_steps)` steps per segment (no longer monotone in the knob),
  and `preflight`/`estimate_ad_memory` VRAM warnings fire more often — that is
  the estimate becoming honest, not a regression. Warnings now point at the
  dominant term (increase or reduce the knob toward `sqrt(n_steps)`).
- `explain_ad_memory` decomposes the new term as a
  `segmented_live_segment_tape` component; `plan_ad_memory`'s does-not-fit
  diagnostics now report the true least-memory candidate (balanced segment
  size) instead of `checkpoint_segments=1` / `checkpoint_every=n_steps`, which
  are ~full-AD-sized under the corrected model.

### Added — waveguide multi-port junction references (`port_reference_sims`)

- `compute_waveguide_s_matrix(..., port_reference_sims=[...])` exposes public
  plumbing for interior-PEC multi-port structures (T-junctions / branches /
  septa). Each `port_reference_sims[i]` is a `Simulation` describing the matched
  STRAIGHT continuation of driven port `i`'s guide (same domain / `dx` /
  boundary, no junction); its PEC-folded materials feed the flux extractor's
  per-port incident-power reference so the guided `P_inc` is correct. The
  default shared VACUUM reference strips the interior PEC and mis-normalizes
  `P_inc`, inflating every `|S|` (measured on the compact 3-port T-junction:
  `normalize=True` max|S|~230; `normalize='flux'` max|S|~9.8, |S11|~1.9).
- Only valid with `normalize='flux'` (raises otherwise); single-mode ports,
  uniform mesh only; not combinable with `eps_override` / `sigma_override`;
  each reference grid must match the device grid (shape + `dx`); one reference
  per waveguide port.
- Two in-method advisories (pure NumPy, no FDTD) warn when the **far-port
  discipline** is not met: probe clearance < 5 evanescent decay lengths of the
  next higher mode, and CPML thickness < ~0.5 guide wavelengths at band centre.
- **Far-port numbers (verified 2026-07-06).** On a far-port geometry (arms
  90/90/70 mm, 48 mm CPML, dx 1.0/0.667 mm) the matched-reference flux path
  reaches passivity 1.006/1.002, reciprocity 0.001, mesh-convergence 0.0297 and
  0.087 vs MEEP (r2000, cross-device). **Necessary-but-not-sufficient caveat:**
  on COMPACT geometry it fixes |S11| (1.86 → 0.49) but the overall matrix stays
  non-physical (residual max|S|~3.9); this enables junction measurements only
  under the documented discipline, not for arbitrary compact junctions.
- Companion committed evidence: `tests/fixtures/waveguide_tjunction_e4/` +
  gate test `tests/test_waveguide_tjunction_e4e5_gates.py`; guard / advisory /
  A/B-witness coverage in `tests/test_waveguide_port_reference_sims.py`.

### Added — coaxial S-parameters: AD-traceable + end-to-end differentiable + `broad_e5_passed` (PRs #260, #261, #262)

- `compute_coaxial_line_reflection(...)` is now **end-to-end differentiable** via a new
  `eps_scale` design channel: `grad(|S11|**2)` w.r.t. the dielectric flows through
  FDTD -> DFT plane accumulators -> modal voltage -> matrix-pencil reflection -> Gamma.
  Pass a scalar or `(nx, ny, nz)` `eps_scale` to optimize a coaxial reflection under
  `jax.grad`; `eps_scale=None` is byte-identical to the validated numpy path.
- The reflection extractor (`coaxial_line_reflection_from_plane_voltages`) is now
  `jax.numpy`-traceable (dual-path: concrete -> numpy float64, traced -> jnp), and a
  differentiable voltage line-integral `coaxial_line_plane_voltage_jnp` was added.
- **`coaxial_port` promoted to `broad_e5_passed`**: with the committed broad-E5 analytic
  envelope + broad-E4 MEEP comparison (PRs #256/#259) and the passing composition
  AD-vs-FD gate (`tests/test_coax_end_to_end_ad.py`, 2.6%), the clean-checkout auditor
  (`check_port_external_references.py`) returns `coaxial_port` PASSED. Tolerances unchanged.

### Fixed — multi-port wire `run(compute_s_params=True)` S-matrix (item-5 Stage 4)

- Multi-port wire ports now return the FULL S-matrix from `run(compute_s_params=True)`
  on a uniform mesh. Previously the uniform fast-path filled only the diagonal
  `S[j,j]` and silently dropped the off-diagonal (a 2-port wire `S21` came out
  identically `0`); multi-port wire now routes through the production-scan driver
  (`compute_lumped_wire_s_matrix_via_scan`), which fills the full matrix. The
  well-conditioned single-port wire fast-path is unchanged.
- **Behaviour change**: a MIXED lumped + wire port set with `compute_s_params=True`
  now raises `NotImplementedError` (their wave-decomposition conventions differ)
  instead of silently returning a wire-only diagonal matrix that dropped the
  lumped ports. Use a homogeneous all-lumped or all-wire port set.
- The eager `extract_s_matrix_wire` is no longer on the `run()` hot path (kept for
  diagnostics / openEMS-crossval tooling), removing the last hand-maintained
  second-FDTD-loop drift-class root (siblings of #203/#205/#206).

### Added — MSL S-parameters on non-uniform meshes (PRs #238, #239)

- `compute_msl_s_matrix()` now runs on non-uniform (`dz_profile`/`dx_profile`/
  `dy_profile`) meshes with the default `mode="laplace"` (and `"uniform"`) feed,
  routed through the non-uniform runner. `mode="eigenmode"` on a non-uniform
  mesh raises `NotImplementedError` (previously ALL non-uniform meshes were
  rejected). The extractor math (probe abscissae, transverse integrals) is
  NonUniformGrid-aware (#238); uniform-mesh results are byte-identical.
  Scope: the NU lane is internally gated (settled-S11 GPU physics gate,
  `tests/test_msl_nu_sparam_gate.py`); external cross-solver validation of the
  NU lane is still outstanding — see `docs/guides/sparameter_support_matrix.md`.

### Fixed — MSL V·I current DFT leapfrog half-step (PR #240)

- The MSL port's Hy/Hz current DFT now applies the Yee leapfrog `exp(+jω·dt/2)`
  half-step correction (H fields live at half-integer time steps). Reported MSL
  S-parameters change slightly (~0.3–0.7° phase-scale over the validated bands).

### Deprecated — removal slated for rfx v2.0

Both functions below already carried `DeprecationWarning`s; this release only
pins the removal version (v2.0) in the warning text and here.

- `compute_coaxial_s_matrix()` (single-plane V/I in a closed PEC box; reports
  non-physical `|S11| > 1` for a lossless short) — use
  `compute_coaxial_line_reflection()` (validated coax-line method).
- `minimize_s11_at_freq()` (time-gating heuristic biased for short-round-trip
  antennas, issue #72) — use `minimize_s11_at_freq_wave_decomp` +
  `Simulation.forward(port_s11_freqs=...)`.
- (Matching the existing v2.0 removal notices on `pec_faces=` and
  `set_periodic_axes()`.)

### Added — AD-memory planning explainability (PR #231; diagnostic/planning-only, no numerics-path change)

- `Simulation.explain_ad_memory(...)` decomposes the selected AD-memory estimate
  into named contributors (field tape, segment-boundary carries, CPML/material,
  NTFF) with a component-sum invariant. New public types `ADMemoryComponent` and
  `ADMemoryExplainabilityReport` (exported from `rfx`).
- Evidence-class labels on AD-memory artifacts (`static_estimate` /
  `calibrated_conservative_plan` / `static_ad_explainability`) so a planning
  estimate is never confused with profiler evidence or a bounded certificate.

### Changed — AD-memory planning (PR #231)

- Checkpoint knobs separated: `checkpoint_every` (non-uniform scan-of-scan chunk
  length) vs `checkpoint_segments` (uniform segmented-scan count; must divide
  `n_steps`; mutually exclusive).
- Conservative `AD_MEMORY_FIT_SAFETY_FACTOR` (1.30) applied before the
  full-AD / segmented fit flags; `design_mask` is recorded but does NOT reduce
  the estimate until masked-state memory has observed calibration. Strict input
  validation, strict-JSON (`allow_nan=False`) serialization, and MB-aware
  formatting (sub-10 MB no longer rendered as `0.00 GB`).

### Added — AD compiled-memory certificate, saved-residual diagnostics, Pareto + checkify tooling (diagnostic/planning-only, no numerics-path change)

- `Simulation.ad_memory_preflight(...)` composes the static AD-memory planner,
  explainability, and mesh-intelligence reports (plus an optional saved-residual
  diagnostic) into one `ADMemoryPreflightReport` with actionable
  `ADMemoryActionHint`s. Does not run FDTD.
- `Simulation.ad_memory_compiled_certificate(...)` reads a caller-supplied
  compiled executable's `Compiled.memory_analysis()` once and fails closed into
  an `ADCompiledMemoryCertificate` bounded to one exact scope. The verdict is
  estimate-framed (`compiler_estimate_within_budget` /
  `compiler_estimate_exceeds_budget`, boolean `estimate_within_budget`): it is a
  JAX compiler **estimate**, not a runtime peak-memory guarantee — it excludes
  allocator fragmentation/scratch, and the fit recommendation reports the
  estimate's utilization of the target budget.
- AD saved-residual introspection: `inspect_ad_saved_residuals`,
  `diagnose_ad_saved_residuals`, `parse_saved_residual_line` parse JAX's
  saved-residual output into JSON artifacts (`ADResidualInspection`,
  `ADSavedResidualDiagnosticReport`, …). Read-only; not a runtime profile.
- Multi-objective sweep tooling: `pareto_front`, `pareto_mask`,
  `weighted_scalarization`, `epsilon_constraint_mask`,
  `select_epsilon_constrained` (+ `SweepResult.pareto_front`) — optimizer-agnostic
  numpy utilities returning JSON-serializable Pareto artifacts. `atol` is a
  strict-improvement tolerance (a strict partial order: a larger `atol` retains
  more points and never empties a non-empty front).
- Opt-in JAX invariant checks: `checkify_invariants` with `check_finite`,
  `check_positive`, `check_bounds`, `check_courant_number` wrap
  `jax.experimental.checkify` so RF/design invariants survive jit/grad/vmap/scan.

### Changed — `maximize_directivity` objective defaults (behaviour change)

- `maximize_directivity(...)` now defaults to **`log_ratio=True`** — the full,
  sign-correct quotient gradient `U'/U - P'/P` — instead of the legacy
  `-U/stop_gradient(P)` mode, which is wrong-sign for any degree of freedom that
  changes total radiated power (PEC/conductor topology, lossy/σ, magnitude-only
  dielectric reshape; GitHub #129). Pass `log_ratio=False` for the old behaviour.
  The loss is now `-(log U - log P)` (positive below ~11 dBi), not the old
  fixed-negative ratio.
- The directivity denominator `P_rad` is now integrated over the **full sphere**
  (matching `farfield.directivity()` / `antenna._total_radiated_power`) rather
  than the upper hemisphere, so the optimized quantity is the true directivity;
  the hemisphere-only integral inflated it (~+3 dB) for any radiator with back
  radiation. The shipped T-MTT-paper beam-steering example is unaffected (it builds
  its own full-sphere `4π U/P_rad` objective and never called this function).

## [1.6.6] - 2026-06-24

Maintenance release: a behaviour-preserving internal refactor (extract the
waveguide mode solver into its own module — byte-identical, GPU-suite-confirmed)
plus repo hygiene (remove dangling internal-doc references from shipped code and
add a CI guard). **No user-visible behaviour or public-API change.** Per-release
GPU gate green: `pytest -m gpu` on the release commit = 187 passed / 62 skipped /
2 xfailed (VESSL gpu-rtx4090).

### Internal — refactor + repo hygiene (no user-visible behaviour change; PRs #218, #219)

- Extracted the rectangular-waveguide transverse mode solver and mode-profile
  linear algebra (10 pure-NumPy helpers) out of the 3364-line
  `rfx/sources/waveguide_port.py` into a new sibling module
  `rfx/sources/_waveguide_modes.py`, re-imported so every existing import path
  (`rfx.sources.waveguide_port.*`, the `rfx/eigenmode.py` private imports, the
  public re-exports) is unchanged (`waveguide_port.py` 3364 → 3090 LOC). A
  verbatim, behaviour-preserving move: the helpers' outputs are byte-identical
  pre/post (verified by output digest on both `main` and the branch), the
  `jax.grad` S-parameter tape is unaffected, and the GPU suite is unchanged
  (187 passed / 62 skipped / 2 xfailed). A new contract test
  (`tests/test_waveguide_modes_extraction_contract.py`) pins the re-export
  surface and the pure-NumPy (no-import-cycle) invariant. (PR #219)
- Removed 23 dangling references to gitignored internal docs
  (`docs/agent-memory/…`, `docs/research_notes/…`, `.claude/…`, `CLAUDE.md`) from
  shipped code and examples — public clones do not contain those paths — and
  repointed the two with public equivalents to `docs/agent/recipe-*.mdx`. Added a
  CI guard (`.github/workflows/lint.yml` `agent-docs-hygiene`) so the class cannot
  regress. Docstring / comment / warning-message text only; no behaviour change.
  (PR #218)

## [1.6.5] - 2026-06-19

Highlights: a validation-framework **reframe** — the `broad_e5_passed`
port-external-reference verdict now means magnitude envelope + an (approximately)
convention-free phase witness + an AD-vs-FD differentiability moat + a live-physics
anchor + committed-artifact enforcement (it was a magnitude-only checkmark) — plus
machine-readable per-port physics-set validation ceilings, several
documentation-honesty corrections, a public-docs surface narrowing, and MSL /
Floquet / non-uniform / distributed test hardening. This is mostly validation rigor
and honesty; the only user-visible runtime change is a per-port reported-Z0 sign
normalization (S-parameter-invariant).

### Changed — port-external-reference validation reframe (T0–T2.5; PRs #184, #190, #186, #187, #188, #189)

- `scripts/diagnostics/check_port_external_references.py`'s `broad_e5_passed`
  verdict now requires, beyond the magnitude envelope: a documented numeric
  breadth floor (≥4 cases, ≥2 mesh, ≥2 geometry/eps, a freq-span ratio, all cases
  pass) enforced for **every** family (was waveguide-only); an approximately
  convention-free TE10 propagation-phase witness (`tests/test_waveguide_phase_gate.py`);
  an **AD-vs-FD differentiability gate wired into the verdict** (a family whose
  extractor is numpy / not-traceable — e.g. coaxial — cannot pass); a live-physics
  anchor that runs a real `compute_waveguide_s_matrix` PEC-short / empty-guide check
  rather than only replaying a frozen JSON; physics-derived (measured-envelope)
  tolerances; and git-committed-artifact enforcement (`--require-committed`) so
  evidence living only in gitignored `.omx/` no longer counts. `missing_evidence`
  now gates the verdict. (T2.4 note: the planned `C·(k·dx)²` dispersion-tolerance
  model was falsified by the committed data and replaced with a measured envelope —
  a stop-and-redesign, not a tweak.)

### Added — per-port physics-set target ceilings + usage rules (PR #191)

- The port-external manifest and auditor now declare, per family, a
  machine-readable `target_ceiling` (the validation ceiling the port can physically
  reach) and `usage_rule`, from a controlled vocabulary. broad-E5 is **not** a
  universal goal: `rectangular_waveguide_port` = broad-E5 (achieved);
  `microstrip_line_port` = broad-E5 matched-regime only; `coaxial_port` = broad-E5
  pending a differentiable API; lumped/wire = E4 natural ceiling (single-cell feeds
  have no transmission-line oracle — "validated to ceiling", not a failure);
  `floquet_port` = broadside structural-partial; generalized-planar = unimplemented.
  Descriptive context emitted alongside the verdict — it does **not** change the
  pass/block gate logic.

### Changed — public documentation surface narrowed (PR #198)

- Public `/rfx/` docs were narrowed to maintained workflows + bounded support
  envelopes: the route inventory was trimmed, generated-API and agent-deploy-sync
  surfaces removed, and public wording rewritten around documented evidence
  envelopes (temporary surfaces such as SBP-SAT subgridding stay out of user docs).
  No library behaviour change.

### Fixed — documentation honesty (T0; PRs #192, #193)

- Corrected a stale claim that the coaxial family was *"the current validated"*
  method and *"the only family currently passing the clean-checkout port
  external-reference audit"* — true only before the rectangular-waveguide broad-E5
  evidence was committed (PR #181, v1.6.4). Coaxial evidence still lives in
  gitignored `.omx/`, so the auditor reports `coaxial_port` BLOCKED and
  `rectangular_waveguide_port` is the single passing family
  (`coaxial_port` → `broad_e5_demonstrated_evidence_uncommitted` across README,
  support matrix, port-selection guide, evidence-rule doc, manifest, CHANGELOG, and
  the reference-lane doc).
- The MSL thru-line openEMS smoke comparator
  (`compare_msl_thru_openems_reference.py`) no longer reports a bare `passed` when
  its |S11| channel is non-discriminating (on a matched line the reference |S11| ≈
  the tolerance, so a degenerate output would have "passed"); it now flags the S11
  channel informational and rests on transmission. Added a committed
  estimator-level test for `rfx.harminv` (synthetic known-frequency recovery,
  float32 robustness), and corrected the cv05 metric label (a Harminv-vs-Harminv
  resonance-frequency agreement, not an S11-vs-S11 match).

### Fixed — microstrip-line reported characteristic impedance sign (issue #140, PR #194)

- `compute_msl_s_matrix` now reports a positive `Re(Z0)` on **both** ports. A `-x`
  port previously reported a negative Z0 (it inherited the sign of the
  direction-aware closed-Ampère loop current), which also false-fired the |Z0|
  honesty guard at ~228% deviation. This is **S-parameter-invariant** — the reported
  Z0 never enters S11/S21 (those use the static analytic Hammerstad-Jensen Z0); the
  genuine ~20–27% Yee-staircase Z0 warning correctly remains. A new `@slow`
  thru-line test locks |Z0| length-invariance + the positive-sign behaviour. The
  earlier PR-#134 alarm (non-physical Z0 corrupting S-params, runaway passivity) was
  verified-and-refuted; **issue #140 closed**.

### Added — validation test coverage (PRs #195, #196)

- Floquet: an extractor-level AD-vs-FD **agreement** test for
  `compute_floquet_s_params` (the differentiability moat — `jax.grad` agrees with
  central finite-difference to <1e-2 at a fixed step; previously only a finiteness
  smoke existed).
- Non-uniform mesh: an analytic-gated NonUniformGrid accuracy test against a
  **graded-axis-dependent** mode — an air PEC cavity TM111 resonance (p=1, whose
  closed-form frequency moves with the graded *z* extent, unlike the existing
  `test_stage1_nu_physics_gate` TM110 p=0 gate, which is z-independent) reproduced to
  ~2.7% on a genuinely graded mesh. This gates the graded axis against a number it
  actually changes.

### Fixed — distributed tests on a single-GPU pod (issue #162, PR #197)

- The multi-GPU `tests/test_distributed.py` tests are now device-count-adaptive
  (shard across `min(4, jax.device_count())`) and skip cleanly when <2 devices are
  present, instead of failing a hardcoded `len(devices)==4` assert on a single-GPU
  pod (where the host-device-count sentinel does not add virtual devices). The
  distributed runner is verified equivalent to single-device by the committed tests
  (rel-err ≤1e-3) — this was test brittleness, not a runner defect. The GPU suite
  goes green because the multi-device tests now SKIP cleanly on the single-GPU pod;
  reliable multi-device CI coverage is environment-gated (a known follow-up — needs
  an isolated pytest lane). **issue #162 closed**.

## [1.6.4] - 2026-06-16

Highlights: the rectangular-waveguide-port **broad-E5 close** (committed
analytic-Airy band envelopes + an rfx-vs-Palace-FEM external comparison, with the
port-external-reference audit GREEN for `rectangular_waveguide_port` on a clean
checkout) and removal of the orphaned legacy 3-probe MSL extractor — on top of the
accumulated correctness, preflight, AD-tape, and validation-lane work since 1.6.3.

### Added — rectangular waveguide port broad-E5 evidence, committed (PR #181, 2026-06-16)

- `compute_waveguide_s_matrix(normalize='flux')` broad-E5 evidence now survives a
  clean checkout: five WR-band (WR-28/62/15/340/10, eps_r 2 & 4) analytic-Airy flux
  envelopes (20/20 cases, max |S| diff ≤ 0.0414) committed under
  `tests/fixtures/waveguide_broad_e5/`, plus an rfx-vs-Palace-FEM broad-E4 external
  comparison across the empty / PEC-short / dielectric-slab geometry axis (max |S|
  diff 0.0707, gate 0.10). Previously this evidence lived only in gitignored
  `.omx/` outputs, so `scripts/diagnostics/check_port_external_references.py`
  reported the family `blocked` on a clean checkout while the manifest claimed
  `broad_e5_passed`; the auditor now reports `rectangular_waveguide_port` passed.
  New gate `tests/test_waveguide_broad_e5_envelope_gates.py` re-derives both
  verdicts from the committed fixtures and mirrors the auditor's broad-E5/E4
  acceptance. R5 note: coarse Meep (res 3/4) gives a non-physical PEC-short
  |S11|>1, so the converged Palace high-order FEM reference is used for that
  geometry; rfx itself is exact (|S11|=1.0000).

### Removed — orphaned legacy 3-probe MSL extractor (2026-06-15)

- Deleted the pre-issue-#80 closed-form 3-probe MSL de-embedding helpers
  from `rfx/sources/msl_port.py`: `_solve_3probe`, `msl_forward_amplitude`,
  `compute_s21`, and the unused `_integrate_v` / `_integrate_i` line
  integrals.  They had **zero callers in `rfx/`** — the production MSL
  S-matrix path uses the closed Ampère-loop current (`msl_loop_current`,
  retained) plus the SVD N-probe wave decomposition (`extract_msl_nprobe`)
  — and were only kept alive by one unit test
  (`tests/test_msl_port.py::test_compute_s21_round_trip`), removed with its
  imports.  None were exported (no `__all__`; absent from top-level `rfx`),
  so the public surface and the api-reference inventory are unchanged.
  Closes architect-review item NEW-1.

### Fixed — normalize='flux' waveguide S-matrix joins the AD tape (issue #148, 2026-06-12)

- `extract_waveguide_s_matrix_flux` is now jnp-native end to end: the
  `np.array(flux_spectrum(...))` concretizations and the in-place numpy
  S-matrix assembly are gone, so
  `compute_waveguide_s_matrix(normalize='flux', eps_override=<traced>)`
  works under `jax.grad` (previously: `TracerArrayConversionError`) —
  design loops can optimize directly through the production-recommended
  power-flux extraction instead of the normalize=False-then-validate
  workaround.
- The rewrite adds double-where guards at the two genuine gradient
  singularities (sqrt of a zero power ratio at a perfect match/null,
  angle of a zero modal ratio); primal values are preserved exactly.
- Forward regression: S-matrix unchanged vs the numpy path within the
  float-reassociation envelope (measured max|diff| 1.1e-7 on the WR-90
  fixture). New CI gates in `tests/test_waveguide_flux_ad.py`
  (composition-level grad finite + central-FD agreement ≤5% + forward
  no-op-override equivalence); support matrix `ad_evidence` updated.

### Fixed — MSL N-probe extractor NaN gradient at tiny field scales (2026-06-12)

- `extract_msl_nprobe`'s β-refinement (`_estimate_beta`) produced `nan`
  gradients when the plane-integrated probe voltages were very small
  (|V| ~ 1e-14, measured on the density-PEC/Kottke forward path): the
  float32 residual curve over the β scan went numerically flat, the
  parabolic second-difference collapsed below its 1e-20 guard, and the
  **single-where** division guard leaked `0 * nan = nan` through the
  backward pass — the exact trap class the module's `_solve_q`
  custom-JVP comment documents, reintroduced by the lstsq rewrite.
  Forward values were always finite (the failure was invisible to
  value-level checks and to unit-scale AD tests — composition-level
  only). Fixed with the double-where idiom plus scale-normalizing the
  β-estimate input (`v/max|v|`; α/γ/Z0 keep absolute scale via the raw
  final lstsq). Found by the msl_stub G2 re-run (VESSL 369367242390:
  Adam grad=nan from iter 0 while the 17-point brute scan stayed
  finite). Regression-locked by
  `test_nprobe_grad_finite_and_scale_invariant_at_tiny_v` (fails on the
  old code; locks finiteness at scale 1e-14 + scale-invariance + FD
  match).

### Fixed — cv03 flux-region congruence (issue #160, 2026-06-12)

- `examples/crossval/03_straight_waveguide_flux.py`: the rfx flux monitors
  are now bounded to the same `2*wg_width` region the Meep `FluxRegion`
  measures, instead of the full y-plane (UPML padding included). The
  full-plane `flux_in` additionally integrated the line source's radiation
  cone — power that physically exits through the transverse absorber before
  `flux_out` — so the self-transmission read 0.913 against the [0.95, 1.05]
  gate with **no flux-normalization bug present**. Measured matrix
  (resolution 10/15/20): full-plane 0.913 / 0.986 / 0.958 (non-monotonic,
  not a convergence curve); bounded 0.974 / 1.011 / 0.997 — passing at every
  resolution including the recipe mesh. Truncation witness: bounded
  resolution-10 T(f_peak) = 0.977 at 3x run length. Gate unchanged.
  Falsifier matrix: `scripts/diagnostics/cv03_flux/sweep_t_deficit.py`.
- Second comparator defect (same script, surfaced by the lane's first real
  Meep execution): the rfx integration time was slaved to Meep's
  `stop_when_fields_decayed` wall clock — when Meep stopped at t=200 the
  rfx flux DFT was truncated mid-tail and read T=1.155. rfx now runs a
  fixed 400 a/c0 units (measured band: 0.9736 at 1x, 0.9772 at 3x).
  `until_decay=1e-5` was tried and rejected for this geometry: the point
  stopper triggers at ~2200 steps while the eps=12 guide's slow tail is
  still carrying flux (T=0.745) — point-field decay is not a
  flux-convergence witness here (filed as issue #169).
- Gate statistic re-specified to the **central-band mean** T (fcen ±
  0.15·df), tolerances unchanged at 1.0 ± 0.05 and cross-diff < 0.05.
  Measured first (sweep matrix + lane runs 27393931821/27394439174): at
  the recipe mesh — 11.5 cells/λ_eff at freq_max, below the preflight's
  own ≥20 floor for flux extraction — rfx's per-bin T(f) carries the
  preflight-documented ±5-10% coarse-mesh ripple while Meep's curve is
  smooth, so the old single-bin gate sampled at Meep's peak bin landed
  in ripple valleys (0.902 at f=0.1510) even at resolutions where the
  band-energy transmission is clean (band-mean 0.966 / 1.005 / 0.989 at
  resolution 10/15/20). The band mean is the physically meaningful
  energy-transmission estimator; peak-bin values remain printed for
  information.

### Fixed — preflight 2D false positive + unit-adaptive warning text (issue #166, 2026-06-12)

- **`absorber_overlap` no longer false-trips on the collapsed z axis in 2D
  modes.** The preflight thickness mirror assumed an absorber on every
  non-PEC/PMC/periodic axis, but 2D grids collapse z to a single cell with
  no absorber at all (`Grid` sets `pad_z = 0` and strips z from
  `cpml_axes`) — so every 2D source/probe, necessarily at z=0, warned
  "near/inside UPML region" (cv03: one line per line-source point, 20 lines
  of spam per preflight). Real x/y overlap in 2D and all 3D behaviour are
  regression-locked unchanged.
- **Scale-sensitive preflight messages now pick units adaptively**
  (`_fmt_len` / `_fmt_freq`): the mesh-resolution and absorber-placement
  warnings printed fixed mm/GHz, rendering optical-scale setups as
  `dx=0.000mm`, `lo=0.0mm`, and `freq_max=74950.00GHz` — values that read
  like bugs while being correct (cv03: 100nm, 2µm, 74.95THz). Remaining
  RF-lane messages (NTFF, MSL, ports) keep mm/GHz, which is correct at
  their scale, and can migrate incrementally.
- New gates in `tests/test_preflight_structured_and_guards.py` (2D-z no
  false positive, 2D-x/y still fires, formatter units, optical-scale
  mesh-warning text).

### Fixed — waveguide-port default source spectrum (issue #150, 2026-06-12)

- **`f0=None` now defaults to the center of the requested DFT band** instead
  of the unrelated `freq_max / 2`. The old fallback could land at or below
  the port mode's cutoff (canonical WR-90 toy: 6 GHz < fc_TE10 = 6.56 GHz),
  launching an evanescent near-cutoff crawl whose extracted S-parameters were
  physically meaningless and **grew with `n_steps`** (max column power
  20 → 57 → 114 over 600 → 2400 steps on the recorded #150 toy; post-fix
  1.107, identical at 4800 and 9600 steps). Explicit-`f0` setups unchanged.
- **New preflight guards** `port_source_below_cutoff` and
  `port_freqs_below_cutoff`: the resolved source center or any requested
  measurement bin at/below the excited mode's cutoff is now flagged loudly
  (below-cutoff bins also NaN gradients under `jax.grad`).
- `examples/inverse_design/differentiable_s11_design.py` setup corrected
  (issue #149): ports moved clear of the CPML, measurement band kept inside
  the TE20 contamination bound, preflight now runs visibly and aborts on
  issues. AD↔FD relative error after the cleanup: 2.8e-4.
- New gates: `tests/test_waveguide_port_spectrum_guard.py` (preflight codes,
  f0-omitted empty-guide transmission |S21|≈1, and the #150 toy's
  column-power growth signature locked at the measured envelope).

### Current main status (2026-06-10)

- **Recommended public lane remains uniform Cartesian Yee RF/FDTD.**
  Non-uniform mesh, distributed execution, Floquet/Bloch, guarded
  subgridding, and broad inverse-design workflows remain lane-scoped and must
  be described through their support/evidence envelopes.
- **All-port broad-E5 is still incomplete.**  The external-reference audit is
  intentionally blocked until lumped, wire, MSL, Floquet, generalized planar,
  and clean-checkout waveguide artifact tracking satisfy the manifest.  Do not
  turn one port-family promotion into a blanket S-parameter claim.

### Removed — dead multimode waveguide extractor (2026-06-11)

- Deleted the internal helper
  `rfx.sources.waveguide_port.extract_multimode_s_params_normalized`.  It
  was the would-be `normalize=True` multi-mode waveguide S-matrix extractor
  but was never wired into the public API: `compute_waveguide_s_matrix`
  raises for `normalize=True` with `n_modes > 1` (cross-mode channels hit a
  0/0 in the two-run normalization) and routes multi-mode work to
  `extract_multimode_s_matrix` or `extract_multimode_s_matrix_flux` instead.
  Verified zero callers across `rfx/ tests/ examples/ scripts/ docs/`
  (only its own definition plus two docstring cross-references, now
  repointed at `extract_waveguide_s_params_normalized`).  The function was
  not exported (no `__all__` entry, absent from `rfx/api/_sparams.py`
  imports), so the public surface is unchanged.

### Added — preflight, finite-result, and automation guards (2026-06-10)

- `Simulation.preflight()` and `preflight_sparameters()` now return coded
  `PreflightReport` / `PreflightIssue` records while preserving legacy
  list-of-string behaviour.  Automation can gate on `.errors`,
  `.warnings`, `.by_code(...)`, `.raise_for_failure()`, `.to_dict()`, and
  `.to_json()` instead of scraping warning text.
- `run()` now uses the consolidated preflight path with `skip_preflight=...`,
  while preserving hard failures for structurally impossible configurations
  such as UPML + refinement and Floquet + non-uniform z.
- `Result.assert_finite()`, run/forward non-finite warnings, S-matrix
  passivity guards, and optimizer NaN-gradient recovery now surface bad
  states before they silently contaminate inverse-design loops.  Sweeps run
  preflight once and avoid repeated per-case warning floods.
- Added PR/CI guard coverage for the preflight/guard suites and re-enabled the
  tree-wide ruff lint gate.

### Added — coaxial line reflection evidence envelope (2026-06-08)

- `Simulation.compute_coaxial_line_reflection(...)` is the coaxial
  transmission-line reflection path.  Its broad-E5 physics was *demonstrated*
  (analytic Γ envelope over short/open/matched plus resistive 25/100 Ω loads,
  two characteristic impedances and mesh-resolution cases, max |Γ| dev 0.037)
  plus an independent broad-E4 MEEP power-flux short/open comparison over
  4–12 GHz (max |S11| diff 0.063).  **The evidence artifacts live in gitignored
  `.omx/` and are not committed to the repo**, so
  `check_port_external_references.py` reports `coaxial_port` BLOCKED on a clean
  checkout; do not cite this path as `broad_e5_passed` until the artifacts are
  committed and the auditor returns PASSED.  (See the 2026-06-17 documentation
  honesty correction under [Unreleased].)
- `Simulation.compute_coaxial_s_matrix(...)` remains available for backward
  compatibility but is deprecated as the older single-plane V/I path; it is
  not the promoted coaxial claims surface.

### Added — waveguide S-matrix memory control (2026-06-09)

- `Simulation.compute_waveguide_s_matrix(checkpoint_segments=...)` now
  threads segmented checkpointing through the uniform waveguide extractors.
  Regression tests pin bit-identical forward S-matrices for
  `normalize=False`, `normalize=True`, and `normalize="flux"`, finite
  gradients through `eps_override`, rejection for non-divisor segment counts,
  and a loud `NotImplementedError` on non-uniform meshes.

### Fixed — public analysis and objective correctness (2026-06-08 to 2026-06-10)

- Finite-size `FluxMonitor` bookkeeping is regression-locked as
  machine-precision equivalent to summing the same full-plane integrand window
  in standing-wave-heavy fields.  The older "finite-size flux is less stable"
  caveat is superseded by a coverage distinction: finite monitors intentionally
  exclude cells outside their requested window.
- `maximize_directivity(..., log_ratio=True)` / the log-ratio directivity
  objective fixes the wrong-sign gradient for power-changing design variables
  by differentiating the full `log(U_target) - log(P_rad)` ratio instead of a
  partially stopped absolute-power proxy.
- The MSL S-parameter AD tests are restored to CPU CI, with checkpointed tape
  usage so the lane remains covered without requiring GPU-only memory budgets.

### Fixed — waveguide port extractor correctness (2026-04-22)

- **`_co_located_current_spectrum` sign flip.**  The H-derived DFT
  correction was `exp(-jω·dt/2)`; the correct sign from the leapfrog
  timing derivation is `exp(+jω·dt/2)`.  On a lossless empty WR-90 the
  mean `∠(Z_formula / Z_actual)` on a pure forward wave drops from
  −8° to −1°.  No public API change.
- **`_shift_modal_waves` direction-awareness.**  Added a
  `step_sign: int = 1` parameter (+1 for `+x/y/z` ports, −1 for
  `-x/y/z` ports).  Previously the shift formula silently applied the
  `+x` convention regardless of port direction, producing the wrong
  sign for negative-direction ports.  Two-run normalized S-matrices
  are unaffected (the shift cancels in the device/reference ratio),
  but any external caller of `extract_waveguide_port_waves` that
  captured the single-amplitude output for a negative-direction port
  now sees the physically correct sign.
- **`_compute_mode_impedance` below-cutoff sentinel.**  Returns
  `1e30` for TE / `0.0` for TM below cutoff, replacing `jnp.inf`.
  `inf × complex(r, 0)` generated a NaN in the imaginary component
  on most NumPy/JAX implementations and cascaded into NaN
  S-parameters on any multi-mode frequency sweep that straddled a
  higher-mode cutoff.  Regression-locked by
  `tests/test_waveguide_port_validation_battery.py::test_below_cutoff_z_mode_no_nan`.
- **`_compute_beta` Yee-discrete branch.**  Optional `dt, dx` kwargs;
  when both are positive the Yee 3-D dispersion relation is used
  instead of the analytic continuous form.  Now threaded through
  `_compute_mode_impedance`, `extract_waveguide_sparams`,
  `extract_waveguide_port_waves`, and `extract_waveguide_sparams_overlap`
  so Z and β stay internally consistent.

### Changed — defaults (2026-04-22)

- **BREAKING:** `Simulation.compute_waveguide_s_matrix()` now returns
  S-parameters referenced to each port's user-facing `x_position`
  (the plane passed to `add_waveguide_port`).  Previously the default
  was the internal `reference_x_m = x_position + ref_offset·dx`
  (three cells inward), which silently phase-shifted the returned
  complex S-matrix by `exp(-jβ·3·dx)` relative to the user's port
  plane.  Users who explicitly passed `reference_plane=` on the
  port entry see no change.  Magnitude (|S_ij|) is unaffected; the
  phase convention now matches what users get from Meep's
  `get_eigenmode_coefficients` at a monitor placed at the same
  absolute position, and what any analytic formula written against
  the port plane produces.
- **`Simulation(cpml_layers=8)` → `cpml_layers=16`.**  Waveguide-mode
  CPML back-reflection measured on an empty WR-90 scales as 11.7%
  (10 layers) → 4.2% (20) → 1.8% (40).  The previous default was
  tuned for free-space simulations and was inadequate for guided-
  mode absorption.  A parallel `kappa_max` sweep confirmed
  `kappa_max > 1` degrades guided absorption in the current CFS
  formulation, so `kappa_max = 1` is preserved.  Free-space
  simulations see a modest (but monotonic) improvement in peak
  CPML absorption from the thicker default.
- **CPML polynomial order 2 → 3** (`_cpml_profile(order=3)` default
  in `rfx/boundaries/cpml.py`).  Matches the Taflove & Hagness 3rd
  ed. §7.9 recommendation for guided-mode absorbers.

### Added

- **Preflight P2.8 — waveguide-port reference-plane sanity.**
  `Simulation.preflight()` now verifies each waveguide port's
  effective reference plane lies inside the domain, outside the
  CPML absorbing region, and does not intersect a geometry box.
  Raises `ValueError` for out-of-domain; emits `UserWarning` for
  CPML or device overlap with an actionable remediation message.
- **Validation battery** (`tests/test_waveguide_port_validation_battery.py`):
  nine tests locking physical-correctness invariants with Meep-class
  gates where achievable and explicitly-ratcheted gates where the
  extractor still has a known residual.  Supersedes the older loose
  `test_passivity_*` / `test_unitarity_*` gates in
  `tests/test_conservation_laws.py` (those remain as broad
  regression detectors with a loosened lower bound to reflect the
  real extractor accuracy ceiling, documented inline).
- **Two-port contract lock** (`tests/test_waveguide_twoport_contract_v1.py`):
  three tests fixing the v1 normalized two-port invariants (empty
  preservation, PEC-short strong reflection, reference-plane
  invariance on empty guide).
- **WR-90 crossval skeleton** (`examples/crossval/11_waveguide_port_wr90.py`):
  a diagnostic reporter against analytic Airy and (when present) a
  Meep reference JSON.  Not a regression gate — gates live in the
  battery above.
- **Diagnostic scripts** under `scripts/`:
  `waveguide_port_canonical_diagnostics.py` (before/after snapshot
  harness with `--json`), `isolate_extractor_vs_engine.py`
  (empty-guide V/I spillover diagnosis, scriptable CPML and
  kappa_max sweeps), and `slab_physical_diagnostics.py` (per-freq
  rfx vs analytic vs Meep with magnitude and phase breakdown).

### Known issues (carried from earlier sessions)

- `|S11|` at resonance nulls (e.g. dielectric-slab quarter-wave
  minimum) remains ~0.05–0.10 on the default grid; halving `dx`
  locally via `dx_profile=` cuts this by ~30%.  Meep at equivalent
  resolution shows a comparable ~0.05 floor, so this is a
  shared-FDTD discretization limit, not an rfx-specific bug.
- rfx vs Meep `∠S21` residual on the WR-90 crossval slab case
  (`examples/crossval/11_waveguide_port_wr90.py`) fits a linear
  `Δφ(rad) = slope·β + intercept` model to RMS 2.3° with
  slope ≈ −5.9 mm and intercept ≈ −57°
  (`scripts/phase_offset_beta_sweep.py`).  Applying this correction
  back to the slab rfx S21 reduces the RMS phase diff from 113° to
  2.3° — VERIFIED (`scripts/verify_phase_alignment.py`).

  Decomposition against physics:
  - `exp(−j(β_slab − β_empty)·L_slab)` (material-contrast phase that
    appears in two-run normalization through a dielectric slab):
    linear-in-β fit gives slope ≈ −1.95 mm, intercept ≈ −44°
    (range −56° to −69° over the band).
  - Measured: slope ≈ −5.87 mm, intercept ≈ −57°.
  - Physics explains ~1/3 of the slope and most of the intercept;
    a residual of **−3.9 mm slope + −13° intercept** remains
    unexplained.

  Experiment 1 (2026-04-22): rfx `dx` 1.0 mm → 0.5 mm, Meep unchanged.
  Result: slope −5.87 mm → −6.0 mm, intercept −57.3° → −58.2°
  (essentially identical). **Cell-snapping / Yee-discretization
  hypothesis FALSIFIED** — the residual does not scale with rfx mesh
  size.  Remaining candidates: (a) Meep's `get_eigenmode_coefficients`
  α⁺ is referenced to a different plane than rfx assumes (cell
  centre vs monitor plane); (b) implementation difference in how
  either code handles the E/H overlap at a material-discontinuity
  edge during the two-run device/reference pair.

  The same (slope, intercept) also does NOT transfer to the
  PEC-short case (`|S11|` RMS stays 104° → 103°), so the offset is
  slab-geometry-specific rather than a universal convention shift.
  Magnitude agreement (|S21|) remains within 3–5% across the band.
  Practical guidance: compare rfx per-geometry against analytic
  Airy (where rfx matches |S21| within ≈ 5% on the slab); do not
  expect bit-level phase agreement with an external Meep script
  that has its own monitor / source-pulse conventions.

### Do-not-repeat log (carry-over from diagnosis)

- Do **not** retune `kappa_max` above 1 in pursuit of better guided-
  mode absorption under the current CFS-CPML formulation — the
  effect is **negative** (sweep evidence 2026-04-22).  Use thicker
  CPML instead.
- Do **not** treat `max|Ez|` over the full simulation window as a
  source-directionality metric — it is dominated by CPML round-trip
  reflection and has led multiple sessions in a circle.  Use an
  early-time-windowed envelope; the regression-locked version lives
  in `test_source_directionality_early_time`.

---

## [1.6.3] - 2026-04-17

(reconstructed from commit log)

### Fixed

- **Periodic boundary + CPML allocation** (`#68`): `set_periodic_axes` was
  not honoured during CPML layer allocation, causing CPML to be placed on
  periodic faces.  Preflight now detects and rejects this configuration.
  (`fix(boundary): #68 honor set_periodic_axes in CPML allocation + preflight`)

### Added

- **`distributed=True` threaded through `optimize()` and
  `progressive_optimize()`** (`#69`): the `distributed` keyword introduced
  in v1.6.2 for `forward()` is now propagated to the higher-level
  optimisation entry points.
  (`feat(optimize): #69 thread distributed=True through optimize + progressive_optimize`)

---

## [1.6.2] - 2026-04-17

(reconstructed from commit log)

### Added

- **`Simulation.forward(distributed=True)` public API** (`#44`, Phase 3):
  opt-in multi-device execution via `distributed=True`.  Covers the full
  non-uniform runner: sharded grid metadata, ghost exchange, CPML on
  x-slabs, Debye/Lorentz ADE ordering contract, soft-PEC occupancy
  sharding, segmented remat + warmup + `design_mask` + `emit_ts`.
- **`progressive_optimize` multi-resolution orchestrator** (`#42`):
  `Simulation.progressive_optimize(...)` chains resolution levels with
  geometry transfer.  API demo added as crossval 08.
- **`design_mask` stop-gradient on non-design cells** (`#41`): cells
  outside the design region are hard-stopped so gradients cannot escape
  the design volume.
- **Non-uniform sentinel hardening** (`#45`): tracer-safe `dz_profile`,
  soft-PEC occupancy, and bit-identical CPML path verified against the
  uniform runner.

### Fixed

- **Multi-device grad NaN at rank-0 corners** (`#44` Phase 4): corner
  cells at rank-0 were not receiving a ghost exchange contribution,
  producing NaN gradients in distributed training runs.
- **`Simulation.__init__` host-coercion** (`#44`): closes NU sentinel #2 —
  non-uniform grid parameters are coerced to host arrays at construction
  time, preventing JIT-time shape errors.

---

## [1.6.1] - 2026-04-16

(reconstructed from commit log)

### Added

- **Preflight auto-run before `forward()` / `optimize()` /
  `topology_optimize()`** (`#66`): preflight is now invoked automatically
  by all three execution entry points; pass `skip_preflight=True` to
  suppress for benchmarking.
- **`optimize()` routed through `sim.forward()`** (`#64`): optimizer now
  uses the single differentiable forward path, removing a separate
  code branch and unifying the differentiable-path surface.
- **Simulation-time breakdown utility** (`#58`): `Simulation.profile()`
  returns a per-phase timing breakdown (CPML init, Yee loop, probe
  accumulation, post-processing).
- **Patch antenna ground-plane size sweep** (`#59`): parametric test
  verifying that ground-plane size does not affect resonance frequency
  within the validated range.

### Fixed

- **Preflight: PEC inside CPML region** (`#61`): raises `ValueError` when
  any PEC geometry box overlaps the CPML absorbing region.
- **Preflight: Taflove dispersion check** replaces the earlier ratio
  heuristic with the exact Yee dispersion criterion from Taflove &
  Hagness Ch. 4.
- **Preflight: probe-in-PEC check**; aspect-ratio threshold tightened to
  2.0:1.
- **Preflight: Courant asymmetry warning** for non-uniform grids with
  per-axis `ν` values.
- **Preflight: NU cell aspect-ratio warning** at > 2.5:1.
- **Decimated Harminv** fixes F3 post-processing OOM on fine-mesh
  convergence runs.

---

## [1.6.0] - 2026-04-16

(reconstructed from commit log)

### Added

- **Memory-efficient inverse design on non-uniform mesh** (`#35`, `#36`):
  segmented scan + remat path; `estimate_ad_memory()` gains
  `ad_segmented_gb` field (`#39`).
- **Non-uniform runner feature parity + inverse-design unblock** (`#34`):
  NU runner now supports all source/boundary types available in the
  uniform runner relevant to inverse design.
- **Per-cell `dx`/`dy` profile support** for non-uniform grids.
- **`n_warmup` stop-gradient split** (`#40`, `#56`): warmup steps are
  detached from the AD tape, preventing spurious gradients from the
  initial transient.
- **Streaming multi-frequency NTFF sweep** (`#43`, `#55`): a single
  forward pass accumulates DFT data at multiple frequencies without
  storing full time-series.
- **3D structure + far-field visualisation API** (`#38`, `#54`):
  `Simulation.visualize_structure()` and `visualize_far_field()`.
- **`minimize_s11_at_freq` objective** (`#50`, `#52`): single-frequency
  S11 proxy usable directly inside `forward()`.
- **2-port wire-port S-matrix with passive loads and direction** (`#34`
  area): `add_wire_port` supports two-port extraction with explicit
  termination impedance and `+`/`-` direction.

### Fixed

- **Physics-based resolution thresholds in preflight** (`#37`, `#53`):
  replaces heuristic cell-count thresholds with wavelength/skin-depth
  criteria.
- **Thin-PEC on non-uniform mesh** (`#48`, `#51`): rasterisation fix +
  preflight warning + mesh-aligned patch visualisation.
- **`excite=False` guards + `forward()` profile check + preflight**:
  sources with `excite=False` no longer contribute to the excitation
  sum used by normalisation.
- **`NameError` `base_materials` in differentiable forward path**.

---

Earlier releases (v1.0.0–v1.3.0) predate this changelog's version sections; see git tags.
