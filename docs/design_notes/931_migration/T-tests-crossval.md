# T (tests/crossval) — findings for other owners, and the text to apply

Owner: the tests-crossval migration group, #931 phase 2 section T.
Branch `feat/931-t5-crossval-tests`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-T5-crossval-tests`.

This file is for the merge / ingest agents. Everything below concerns files
section T does **not** own; the measurements were taken on the migrated branch
and are stated so the owner can apply them without re-deriving.

## 1. Measured findings other groups depend on

### 1.1 cv06b's realized trace width moves 635.0 µm → 571.5 µm (crossval-B, docs)

Measured on a stand-in build of cv06b's own board (W_TRACE = 600 µm,
H_SUB = 254 µm, DX = H_SUB/4 = 63.5 µm), through
`rfx.boundaries.pec.realized_pec_edge_masks`:

| declaration | realized z walls | realized in-plane width |
|---|---|---|
| 1-cell PEC Box (volume) | 254.0 **and** 317.5 µm | 635.0 µm |
| zero-thickness sheet | 254.0 µm | **571.5 µm** |

The sheet keeps the electrical plane the board has today and moves the width;
the volume keeps the width and turns 35 µm of foil into a 63.5 µm slab with a
second wall. Foil is a sheet (design note §1.3), so **571.5 µm is the number**.

Hammerstad–Jensen on RO4350B (ε_r = 3.66), for whoever re-derives the anchors:

| board | Z0 (Ω) | ε_eff |
|---|---|---|
| design 600/254 µm | 47.895 | 2.8694 |
| old realized 635/254 µm | 46.183 | 2.8823 |
| **contract sheet 571.5/254 µm** | **49.391** | **2.8585** |
| dx = 80 µm board 560/320 µm | 57.463 | 2.8045 |

Consequences the owners must apply:

* `validation/crossval/06b_msl_notch_filter_uniform.py` — `_realized_trace_width`
  keeps reading the realization; `F_NOTCH_AN` moves a second time (it already
  moved 3.6872 → 3.6790 GHz under #723).
* Five public carriers quote the 635.0 µm board and its 46.5 Ω median:
  `validation/README.md`, `docs/guides/sparameter_support_matrix.md`,
  `docs/guides/sparameter_support_matrix.json`, `docs/agent/port-selection.mdx`,
  `docs/public/guide/benchmarks.mdx`. All five must be refreshed in the same
  commit as the re-solved run log, from the run, not by translation.
* **Requested wording** wherever a carrier explains the width (replaces the
  #723 sentence "`round(W_TRACE/DX)*DX` gives 571.5 µm here and is wrong"):

  > The realized trace width is read off the build, never computed from the
  > declared width. Under the lattice ownership contract the trace is foil, so
  > it is declared a sheet and its footprint is the closed node rectangle
  > (#931 §1.3): 571.5 µm on this board. The pre-#931 rule realized 635.0 µm
  > for the same drawing.

* `tests/crossval/test_msl_notch_public_carriers.py` (section T, already done)
  no longer pins 635.0; it compares the run log's `W_realized` against the live
  reading, so a stale log reds with a message naming the re-run.

### 1.2 cv20's Stage-B referee (crossval-E)

`n_z_sub_realized` is a DIELECTRIC reading and does **not** move (§1.1 leaves
dielectric sampling alone), so the 5-vs-6 declared-vs-realized
distinguishability argument survives. `w_trace_realized_m` **does** move by
§1.1 above, so the openEMS Stage-B board is re-meshed and re-run.

### 1.3 Cases that do NOT need a re-run (measured, contradicts the inventory)

The tests-crossval inventory marked these `needs_recompute=true/gpu`. Traced to
their producers, every one of them builds its metal with the low-level
`rasterize(grid, [(shape, 1.0, sigma)])` **cell fill**, which design note §1.8
fences out of the contract:

* cv16 / `tests/fixtures/rcs_mie_ka_sweep` (`test_rcs_mie_ka_sweep_gates.py`)
* `tests/fixtures/rcs_mie_e4` (`test_rcs_mie_reference_gates.py`)
* `tests/fixtures/rcs_sphere_three_way` (`test_rcs_sphere_three_way_gates.py`)
* `tests/fixtures/rcs_cube_bem` (`test_rcs_cube_bem_gates.py`,
  `generate.py:58`)
* the WR-90 T-junction E4/E5 lane
  (`scripts/diagnostics/build_waveguide_tjunction_broad_e5_envelope.py:41-42, 64-65`
  stamps `sigma = 1e10` onto the material array)

**Five GPU re-runs dropped.** crossval-C's reading was right and the
tests-crossval rows were wrong. §1.8 also asks for a test that the two models
are not silently equated; it exists now at
`tests/crossval/test_rcs_cube_bem_gates.py::test_the_sigma_fill_and_the_pec_contract_are_not_the_same_object`,
and it records the measurement: the cube's node- and centre-sampled cell sets
are both 18³ but are **different cells**, and the sphere's are 3023 vs 3082.

### 1.4 The chain battery's absorber does not move (measured)

`tests/_waveguide_chain_battery_fixture.py::transverse_spans` reports
`guide_source = ("domain_faces", "domain_faces")` at every rung, so
`a_guide = 22.86 mm` exactly, `fc_TE10 = 6.55714 GHz` and
`cpml_layers = 17 / 34 / 68` are unchanged by #931. This was flagged as the
least obvious blast-radius edge in the group; it is closed. The reader now
raises if the guide walls ever stop being domain faces.

## 2. Text to apply to files section T does not own

### 2.1 `tests/fixtures/wr90_iris_filter/fixture.json` — `claim_scope` (crossval-D)

The claim scope currently contains the phrase **"bounding zeroed node planes"**,
which `tests/crossval/test_wr90_iris_filter_gates.py` used to require and now
requires to be ABSENT. Replace that clause with:

> The oracle's inputs are read back off the realized metal through
> `rfx.boundaries.pec.realized_wall_planes`. Under the lattice ownership
> contract (#931 §1.2) a region realizes tangential walls on both of its drawn
> faces, so the electrical thickness of an iris drawn `t_c` cells is `t_c·dx`
> and the clear cavity drawn `L_c` cells is `L_c·dx` — drawn equals realized,
> and the total length of the cascade closes on the span exactly. The
> `t_c = round(t/dx) + 1` / `L_c = round(L/dx) - 1` compensations that the
> pre-#931 realization required are deleted, not re-tuned.

The same fixture's `electrical_geometry.rule`, `.compensation`, `.cost_note`
and `cost_of_using_intended_counts_mhz` describe the deleted mechanism. Keep
them only if they are relabelled as history; the live identity is
`iris_thickness_cells == round(t/dx)` and `cavity_cells == [round(L/dx)]`.

`validation/crossval/19_wr90_iris_filter_aghanim.py` must name
`realized_wall_planes` and `#931` (both are now pinned present by the test) and
must NOT contain `round(t/dx) + 1`, `round(L/dx) - 1` or
`bounding zeroed node planes` (pinned absent).

The `iris_thickness_zero_count_sweep` must be re-centred: the test now requires
`rows[0].t_elec_cells == iris_thickness_cells - 0.5` and
`rows[-1] == + 0.5`, at least 11 rows, with the zero count invariant across it.

### 2.2 `validation/crossval/manifest.json` — cases 18 and 19 (crossval-D)

Both cases' `claim_scope` entries quote the compensated geometry. Requested
replacement clause for each:

> Geometry is drawn on node planes and realized as drawn (#931 §1.2): the
> electrical dimensions of every iris, fin and cavity equal their drawn cell
> counts, with no per-case correction. cv18 and cv19 now describe the same
> WR-90 inductive iris with the same convention; before #931 they did not.

### 2.3 `validation/README.md` — the cv15 row (crossval-C)

Line 47 quotes `rfx_one_plane_ground_b29f9de7.json`'s **+6.09 % one-plane**
number. That artifact records a realization the contract cannot produce and
cannot be rebuilt. Requested replacement:

> cv15 (RT/duroid 5880 patch): the ground and patch are declared as sheets on
> the laminate faces (#931 §1.3); the realized cavity is the declared 3.175 mm
> with no vacuum cell. The historical +6.09 % figure was measured on the
> pre-#931 one-plane realization and is retained only in
> `docs/design_notes/` as history.

`tests/contracts/test_evidence_numeric_provenance.py` pins that citation count,
so the two move together.

### 2.4 `CHANGELOG.md` (docs group)

One line section T needs present, for the carriers above:

> **BREAKING** — a microstrip trace declared as foil is now a sheet, and a
> sheet's footprint is the closed node rectangle. On the canonical
> dx = 63.5 µm / 254 µm board the realized trace width moves 635.0 µm →
> 571.5 µm and every quoted Z0 and notch frequency for that board moves with
> it. Boards whose numbers are quoted publicly must be re-solved, not
> translated.

## 3. Open items section T could not settle

1. **cv15's builder API.** `tests/crossval/test_crossval_cv15_wall_planes.py` is
   written against `build_rfx_sim(*, do_gain, ground_plane_z=None)` and skips
   with a named reason on anything else. If crossval-C picks a different knob
   for the negative control, the skip says so; adjust the two `inspect`
   guards in that file.
2. **cv19 / cv18 fixture keys.** The tests read `drawn_*` keys tolerantly
   (`eg.get("drawn_iris_thickness_cells", eg["iris_thickness_cells"])`), so a
   regeneration may drop them. If they are kept, they must EQUAL the electrical
   counts.
3. **Hard numeric pins.** `_PINS_REPINNED_FOR_931` (cv19) and
   `_ENVELOPES_REDERIVED_FOR_931` (cv05) are `False` and gate the pins that
   cannot be recomputed without the re-solve. Flip each in the same commit that
   ingests its artifact, and re-pin from the artifact.
4. **`tests/fixtures/waveguide_chain_battery/` (tests-oracle group).** The run
   is DONE — VESSL 369367259196, artifact
   `/root/workspace/claude-workspace/rfx/runs/issue931-post-chain-battery-20260907T105351Z/fixture.json`,
   full measured-vs-predeclared table in `scripts/vessl_931/T/RECOMPUTE.md`. Three
   things the ingest owner needs from it:

   * the `("pec_short", "sigma")` AD leg did NOT move (g_AD −6.4283 vs
     −6.4214, rel 4.9e-4 against a 0.05 gate). The prediction that it would
     was wrong and is recorded as wrong;
   * `S22` on the pec_short DUT moves by exactly one cell at all three rungs
     (−2.5263 / −1.2646 / −0.6323 mm at dx = 2.54 / 1.27 / 0.635 mm) while
     `S11` does not move at all. That is the far face arriving where it is
     drawn — the design note §5 one-cell witness, on a case that was not asked
     to provide one. Adopting the artifact re-pins every `S22`-derived number
     for that DUT, and only those;
   * thru and slab are unchanged to 1.1e-5, and the nine
     `forward_identity|*|flux|*` failures the run-2 fixture carries are gone.

5. **The eps FD leg's step is outside the Courant limit — a PI decision, not a
   re-pin.** All six `ad_vs_fd|pec_short|*|eps|*` legs come back with
   `g_fd = nan`. Cause, traced and measured (`RECOMPUTE.md`): the minus arm
   evaluates the vacuum θ window at `eps_r = 0.95` while dt is picked at 0.99
   of the `eps_r = 1` limit — `0.99/sqrt(0.95) = 1.0157`, on every rung, since
   the fixture was written. A thru guide with no conductor anywhere NaNs the
   same way at the mid rung, so this is not the realization change; the change
   only removed the lossy `sigma = 1e10` block that used to damp it one cell
   from the window.

   Before the 2026-09-08 amendment, `tests/_waveguide_chain_battery_fixture.py`
   carried `eps_fd_step_courant_ratio` / `assert_eps_fd_step_is_courant_admissible`
   (build-time, no solve; WINDOW-interior ratios 1.015719 for pec_short,
   0.498123 for slab). Nothing called that assert: section T did not own the
   battery tests in `tests/oracle/`. These historical helpers have now been
   replaced by the full-array per-arm certificate; the old window calculation
   is explicitly named `eps_fd_minus_window_interior_courant_ratio`.
   Section T did NOT touch `THETA0_EPS` (0.0) or `FD_STEP_EPS` (0.05): moving
   either re-declares a measurement the battery's predeclaration fixes (a
   CENTRAL difference AT the shipped fixture). The three ways out, for whoever
   re-declares:

   a. `θ0 = +h` for the eps leg on pec_short — the minus arm lands on
      `eps_r = 1.0` exactly. Cheapest; costs the "AD is evaluated at the
      shipped fixture" property, which is the property the predeclaration
      names.
   b. a second-order one-sided difference `(-3 f0 + 4 f_h - f_2h)/(2h)` —
      keeps θ0 and O(h²), with two perturbed evaluations and the existing f0.
      The 0.05 relative gate is retained without re-derivation. Calling all
      one-sided differences first order was incorrect (corrected 2026-09-08).
   c. lower the fixture's Courant factor so `eps_r = 0.95` is admissible —
      correct and unaffordable: dt changes, and every recorded number in the
      battery changes with it.

   PI decision 2026-09-08: (b), for all eps legs including the slab; see the
   dated amendment in `waveguide_chain_battery_predeclaration.md`. Theta0 and
   dt stay fixed. The old 0.498123 slab figure above was WINDOW-interior only,
   not a global certificate: vacuum elsewhere sets the global ratio to 0.99
   for the newly declared arms. The full-array certificate now gates every arm
   and blocks invalid configurations before solving. Material eps_r >= 1 is
   a declared test scope, not implied by passivity (dispersive plasma is a
   counterexample); numerical stability separately requires theta > -0.0199
   in the vacuum window at this dt.
