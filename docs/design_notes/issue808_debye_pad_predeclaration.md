# #808 pre-declaration — Debye recovery regression from the CPML pad's eps_inf-without-pole material

Date: 2026-09-01. Base: origin/main 635ab2e3. Author lane: implementer worktree
(`fix/issue808-debye-pad`). Written and committed BEFORE any verification arm
of the fix runs, per the one-attempt discipline.

## What is established (inputs to this note, not re-run here)

- `tests/unit/autodiff/test_differentiable_material_fit.py::test_recover_debye_reference_mode_public_entry`
  fails on main: de recovered 3.969 (CPU) vs true 3.0, 32.3% against the 20%
  gate pinned 2026-08-07 at de 3.330 (11.0%) / tau 5.29e-11 (5.8%),
  loss 1.44e-4 -> 8.40e-6. Bisect first-bad: fce10916 (PR #638).
- Discriminator session (logs in the session scratchpad,
  `run1_main_baseline.log` / `run2_discriminator.log` / `run3_padrules.log`):
  - Reading (b) CONFIRMED. Holding geometry, observation pipeline and
    optimizer fixed, swapping ONLY the pad rule between the shipped #638+#655
    rule and the pre-#638 legacy rule toggles the result between
    3.969/4.66e-11 (fail) and 3.330/5.29e-11 (pass, digit-for-digit the
    2026-08-07 pin).
  - The issue's one-cell-off geometry arm is confounded at this dx (slab
    collapses to 1x1 columns; tau unidentifiable at 118.7% err) and is NOT
    reused here.
  - IDENTITY arm (all pads vacuum): de 2.651 (11.6%, passes) but
    tau 8.21e-11 (64.3%, FAILS the 15% tau gate).
- Mechanism: the fixture's Debye slab (`Box((0.010,0,0),(0.016,0.009,0.009))`,
  eps_inf=2, touches y/z lo AND hi faces) gets, from #638's hi-face fallback
  plus #655's boundary-node write, a surround of eps_inf=2 WITHOUT the Debye
  pole — 2688 pad-ring cells + 18 repaired hi-face boundary nodes, against a
  32-cell pole mask. That material exists in no declared model.
- #636 (PR #773): extending the POLE into the pad is measured divergent
  (surface-polariton modes of the Re(eps(w))<0 band inside the absorber;
  the one pre-declared stabilization attempt — CFS corner alpha — fired its
  falsifier and was not landed; a stable dispersive pad needs a reformulated
  dispersive PML, out of scope).

## Chosen fix

**Pole-aware hi-face fallback gate** (the archaeology's (ii-restore)):
in the single shared rule `rfx.geometry.rasterize_grid.extend_cpml_pad_materials`,
the #627a hi-face fallback promotion and its #655 boundary-node write are
suppressed for a transverse cell whose SOURCE (inner) column carries any
dispersion pole (Debye or Lorentz/Drude). For such a cell the hi pad takes the
naive outer-column copy (background) and the dropped node stays as rasterized —
exactly the pre-#638 hi-face state the 2026-08-07 pin was measured with.
Lo-face behaviour and all static-material behaviour are unchanged.

Threading: a combined boolean `dispersion_pole_mask` (OR over all
Debye+Lorentz per-pole masks) is passed from all three call sites — uniform
(`rfx/api/_compile.py`), non-uniform (`rfx/runners/nonuniform.py`), batched
sweep (`rfx/vmap_sweep.py`, closure constant under `jax.vmap`). Default
`None` keeps the helper's old behaviour for other callers (the
`_PoleExtendedSim` test harness stays valid).

### Why not the alternatives

- **Completing the pole extension (candidate i)**: allowed only if the prior
  destabilisation has an identified AND FIXED cause. The cause is identified
  (#636 root cause: interface polariton modes) but NOT fixed — the one
  pre-declared stabilization attempt was spent and refuted, and PR #773's own
  disposition is guards-only. Rejected without a run.
- **(ii-strict), pole columns extend nothing on either face**: for this
  fixture (only one material) it is realized-state-identical to the
  discriminator's IDENTITY arm, which already measured tau 64.3% err — a
  committed-gate FAIL. Running it as a fresh arm would be a
  mechanism-equivalent repeat of a measured-failing configuration (R2).
  Rejected on that recorded evidence, not re-run.
- **Widening the 20% gate / moving the committed fixture**: forbidden by the
  task and by the no-silent-gate-loosening rule. Not considered.

### The guard (class guard, not fixture guard)

1. Contract tests in `tests/unit/boundaries/test_cpml_pad_material_extension.py` (uniform +
   NU): for the existing dispersive face-touching fixture, the hi-face pad
   must stay background (eps 1.0) and the dropped boundary node must stay
   unrepaired, while the same fixture with the pole removed keeps the full
   #638 promotion (eps 4.0). Reds on revert of the fix.
2. Preflight advisory: `dispersive_pole_at_absorber_face` broadened from
   {high-Q in-band Lorentz, Drude} to ALL pole families touching an absorbing
   face (closing the #773 Debye blind spot #808 exposed). Message states the
   realized input-side pad contents per the shipped rule (lo pad: statics
   without the pole; hi face: no promotion, unrepaired node) and keeps the
   #636 do-not-extend wording for resonance-risk families. Same emission
   site, same code — the #737 frozen counts (87 sites / 58 codes) do not
   move. Three quiet-tests are inverted BY DESIGN with #808 as the written
   rationale: `test_debye_touching_face_stays_quiet`,
   `test_low_q_lorentz_stays_quiet`, `test_out_of_band_high_q_lorentz_stays_quiet`.

### Enumerated collateral (the ONLY committed values allowed to move)

- `test_pole_extension_stability_lock` (8k canary) and
  `test_pole_extension_divergence_repro_636` (20k, slow): the Lorentz
  stability fixture touches x/y faces on all four sides, so its hi pads for
  pole columns go from eps 4.0 to vacuum under the gate — the measured decay
  ratios move. The asserts (`shipped ratio < 1`, `extended ratio > 1`,
  finiteness) must hold UNCHANGED; docstring numbers are re-measured in this
  change (the #655 precedent). The `_PoleExtendedSim` harness must re-extend
  the STATICS ungated itself, so the "extended" variant keeps measuring the
  documented #636 factorial row (statics+poles both extended) instead of
  drifting to the poles-over-vacuum NaN row.
- The three advisory quiet-tests above (inverted).
- Nothing else. In particular the V173-A FR4 lock (static material — no pole,
  no gate taken) and every static pad test must be bit-unchanged.

## Falsifiers — declared now, run after implementation

- **F1 (recovery returns to the pinned pipeline)**: the #808 test passes on
  CPU with the committed gates untouched AND the printed recovery matches the
  2026-08-07 pin: de in [3.30, 3.36] (pin 3.330), tau in
  [5.19e-11, 5.39e-11] (pin 5.29e-11), loss ~1.44e-4 -> ~8.40e-6 (same
  leading digits). Expected: exact digit match, because the realized material
  arrays under the gate are value-identical to the discriminator's verified
  legacy arm. Gates passing with the pin digits NOT matching = F1 fired;
  investigate, do not claim success.
- **F2 (#638 static benefit survives)**: `scripts/harnesses/v173a_physics_equivalence.py`
  prints f_res_hz == 1994994938.1663296 exactly (FR4 is static; the gate is
  never taken; bit-identity expected). Any drift = F2 fired.
- **F3 (no unenumerated mover)**: the targeted suites below are green with no
  gate/tolerance edit anywhere, and the only value changes are the enumerated
  collateral above. A red anywhere else = F3 fired.
- **F4 (guard reds/fires on the inconsistent-pad configuration)**:
  (a) revert probe — with the pad-rule source changes reverted (advisory left
  in place), the new contract tests must FAIL showing the eps_inf-without-pole
  promotion back in the hi pad; (b) `preflight()` on the #808 factory fixture
  must emit `dispersive_pole_at_absorber_face` (quoted verbatim in the
  results). Guard staying green under (a) or silent under (b) = F4 fired.

Stability pre-declaration: the gate only REMOVES material from pads (pole
columns' hi pads become background); no new medium enters the absorber, so no
new instability class is expected — the canary/repro decay asserts must stay
green. If either goes red, that is a STOP, not a gate edit.

## Planned verification runs

1. F1: `pytest tests/unit/autodiff/test_differentiable_material_fit.py::test_recover_debye_reference_mode_public_entry -m gpu -s`
   (CPU, editable-install finder stripped in-process, `rfx.__file__` printed).
2. F2: v173a harness (~1 min).
3. F3: `tests/unit/boundaries/test_cpml_pad_material_extension.py` (fast lane + the slow 20k
   repro explicitly), `tests/unit/boundaries/test_cpml_pad_face_notch.py`,
   `tests/unit/runners/test_vmap_sweep_dft_planes.py::TestVmapBatchedPadByteIdentity`,
   `tests/unit/nonuniform/test_nonuniform_uniform_end_to_end_reduction.py`,
   `tests/unit/preflight/test_preflight_dispersive_pole_at_absorber.py`,
   `tests/unit/preflight/test_preflight_advisory_emission_contract.py`, full
   `tests/unit/autodiff/test_differentiable_material_fit.py -m gpu`, ruff (repo profile).
4. F4: revert probe + advisory quote.

Declared gap: GPU acceptance (the pinned VESSL GPU suite — #808's failing
lane) is the orchestrator's job post-review; not attempted from this worktree.

R2 ledger for this session: 0 verification arms run at the time of this
commit; the fix gets ONE pre-declared arm (F1-F4 above). The two discriminator
arms already spent belong to the discriminator session and are cited, not
repeated.

## Results (2026-09-01, appended after the pre-declared arm ran; CPU, base 635ab2e3 + this change)

Every falsifier came back on the pass side; none fired.

- **F1 PASS, digit-for-digit**: printed by the committed test (275.3 s):
  `#580 recovery: de 6.00->3.330 (true 3.0), tau 1.00e-10->5.29e-11
  (true 5.0e-11), loss 1.436e-04->8.399e-06` — identical to the
  2026-08-07 pin and to the discriminator's legacy arm. Inside the
  declared bands [3.30, 3.36] / [5.19e-11, 5.39e-11].
- **F2 PASS, bit-identical**: harness JSON
  `{"sha": "3428981502de", "f_res_hz": 1994994938.1663296,
  "s11_dip_db": -68.32136968566955, "s11_dip_f_hz": 3321838252.746537}` —
  f_res equals the declared baseline exactly.
- **F3 PASS**: enumerated movers only. 8k canary last/mid 0.4499-era
  baseline -> 0.3979 measured (still decays; witness `x-hi pad eps=1.00`,
  0 pole cells in pads); 20k repro shipped 0.2145 -> 0.1204 (decays),
  extended variant 5.032 — value-identical to the documented b29f9de
  number, witnessing that the harness statics re-extension reproduces the
  same factorial row. Green with no gate edits: pad file 9+1 (fast+slow),
  face-notch 9 (+2 pre-existing distributed skips), vmap byte-identity 44,
  NU end-to-end reduction 6, preflight family (all 11 files) 124, advisory
  file 8 (3 inverted by design), emission contract 3 (frozen 87/58
  untouched), full material-fit file (see run log). Ruff clean.
- **F4 PASS both legs**: (a) with only the four pad-rule sources reverted
  to the pre-fix state, both new guard tests red with `4.0 == 1.0`
  (the chimera promotion back in the hi pad); restored cleanly.
  (b) `sim.preflight()` on the exact #808 factory fixture emits
  `dispersive_pole_at_absorber_face` (severity warning,
  loc `geometry[#0 y-lo+y-hi+z-lo+z-hi Debye tau=5e-11s]`), stating the
  lo-pad statics-without-pole and hi-face no-promotion facts. The other
  six advisories on this fixture (3x mesh_resolution 14.1 cells/lambda_eff,
  3x absorber_proximity for the probes/port) are unchanged from main.
  Preflight-context note: the committed fit test itself surfaces no
  preflight output — `differentiable_material_fit` calls the factory
  sim's `_assemble_materials` directly (its module never routes through
  `run()`'s auto-preflight) — so the advisory quote comes from
  `sim.preflight()` on the identical fixture.

Stability pre-declaration held: both stability asserts stayed green; the
gate only removed material from pads.

Declared gap (unchanged): the pinned VESSL GPU suite — #808's failing
lane — is the orchestrator's job post-review.
