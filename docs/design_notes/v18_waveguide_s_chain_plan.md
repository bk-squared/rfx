# v1.8 work plan — rectangular waveguide chain closure, then the lumped/wire tail

Base: main 1c38b0d7 (2026-09-02). Every file:line below was re-checked against that commit.

**Rules binding every work package.** No gate loosening without a written root cause. No
re-implementation of the in-graph S assembly or the flux-path AD — that work is DONE and
recorded at `docs/agent-memory/rfx-known-issues.md:3093-3099` (RESOLVED 2026-05-25,
FD↔AD rel_err 2.0e-4). The earlier note `docs/agent-memory/port_sparam_review_2026-05-19.md:11`
says the opposite ("WI-1/WI-2 jnp-ified only the OUTER assembly — end-to-end AD still does
NOT flow"); it is **superseded** and must be cited only as history. `normalize=True` stays
out of every reflection gate (`rfx-known-issues.md:3384-3395`). R2 at the rfx threshold:
one pre-declared attempt per mechanism hypothesis; a second needs a named new falsifier in
writing. Every WP lands with an R3 line.

**Out of scope for the whole of v1.8** (see the contract's exclusions): the #812 artifact
lane. WP6 does not refresh, re-pin or re-label
`tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json`, and no WP
edits the crossval gates cv02/03/04/09/10/14/20/21 or cv06b. That lane is owned by the
Mac-side session (`rfx-known-issues.md:107-108`).

---

## Decisions (PI, 2026-09-02)

The seven questions this plan raised in Appendix B were decided on 2026-09-02. Deferred work is
collected in issue **#854**; the reasoning for each is in Appendix B.

1. **Flux-lane `complex64` cast — remove it.** `normalize="flux"` follows `JAX_ENABLE_X64` like
   `normalize=False` and the NU lane. Falsifier: the sha-pinned float64 golden at
   `tests/unit/autodiff/test_waveguide_sparam_ad.py:98-141`, unchanged. WP1 step 3.
2. **Plane-shift gradient leg — report-then-pin.** First run reports against a pre-declared 1e-2;
   the same PR pins `gate_from_envelope(measured, quantum=1000)`. Magnitude objectives are
   gradient-invariant, complex objectives rotation-covariant. WP2(b), contract 3(b).
3. **No NU graded-region fixture now.** WP4 keeps the derived table, the grading-zone assertion and
   the support-matrix envelope. The fixture and β integration over the shift span → #854 item 1.
4. **Claims-bearing set.** The WP2 battery carries criteria 1, 2 and 3(a)–(c); the 3(d) referee set
   is the battery's Airy slab and PEC-short plus the five broad-E5 replay bands. cv18, cv19 and the
   Meep T-junction are 3(d) support only. Their phase legs → #854 item 2.
5. **Phase referee — Airy suffices for v1.8.** A gated conjugation-corrected Meep phase leg →
   #854 item 3.
6. **dx ladder — non-increase gate, plus witnesses and structural guards.** Monotonicity and
   Richardson are reported first, gated later; rungs are `dx = a/N` at integer N. WP2(c).
7. **Runtime reciprocity warning — off for now.** Warn-only for the waveguide family is sequenced
   after WP2 measures the complex envelope → #854 item 4. WP7 carries it as a follow-on.

---

## WP0 — contract document and ledger hygiene (CPU, no physics)

**Scope.** New `docs/design_notes/chain_closure_contract.md`. Three stale statements:
- `docs/agent-memory/index.md:249` and `:280` still present the cv11 ∠S21 offset as an OPEN
  residual pointing at roadmap W3.4; `rfx-known-issues.md:4093-4112` closed it on 2026-07-02
  as a Meep time-convention conjugation.
- `rfx-known-issues.md:3394` still reads "NU mesh not yet supported (raises)" for
  `normalize="flux"`, which is now the gated NU lane.
- The open-issue list at `rfx-known-issues.md:27-28` still carries #822 and #823. GitHub
  reports both CLOSED, and `:120` records them closing with #837.

Plus one new per-lane status row for the waveguide family.

**Gitignore step (mandatory).** `docs/agent-memory/` is gitignored in this public repo, so
these edits leave no history here. After editing, run the workspace's
`scripts/sync_research_archive.sh` from the primary checkout and commit the refreshed mirror
under `docs/research-archive/rfx/`. An unmirrored ledger edit is unreviewable.

**Falsifier.** `grep -n "W3.4" docs/agent-memory/index.md` returns only rows that say
RESOLVED 2026-07-02 with a pointer to `rfx-known-issues.md:4093-4112`; `grep -n "#822\|#823"`
returns rows with one consistent state; `scripts/sync_research_archive.sh --check` is clean.

**Memory.** Consistent with `rfx-known-issues.md:4093-4112` ("Do not de-embed by the fitted
slope/intercept; conjugate the Meep data") and `:196-197` (the contract doc is the first v1.8
deliverable).

**Must not.** Touch any test or extractor. Delete the cv11 row's half-step lesson at
`index.md:249` (the "PEC-short per-freq spread 0.13 was a dump-comparator artefact" sentence
stays).

---

## WP1 — lane honesty: red test first, then the guard (CPU)

**Step 1, before any source edit.** Commit a red test that records what the uniform
`normalize=True` lane actually does under a traced `eps_override`. The lane concretises at
three sites — `rfx/sources/waveguide_port.py:2383` (`np.array(a_inc_ref)`), `:2397`
(`np.array(b_ref_i)`) and `:2434` (`np.array(b_recv_dev)`) — and **which one raises has not
been measured**. The test must capture the exception type and the traceback frame. The draft
of this plan carried unrecorded probe numbers (`7.58e-02`, `3.09`) with no script path; they
are dropped.

**Step 2.** Add the fail-fast guard to the uniform dispatch in `rfx/api/_sparams.py`, mirroring
the non-uniform guard at `:2332-2341`, so a traced `eps_override` or `sigma_override` with
`normalize=True` raises `NotImplementedError` naming the lane. The step-1 test flips from
recording the tracer error to asserting the `NotImplementedError`.

**Step 3, dtype — remove the cast (decision 1), as its own commit.** The flux lane hard-casts to
`complex64` at `waveguide_port.py:2217`, while `normalize=False` follows the `freqs` precision
through `_rect_dft` (the dtype rule is stated at `waveguide_port.py:1583-1585`) and the NU flux lane
carries no such cast. Delete the cast, so all three waveguide lanes follow `JAX_ENABLE_X64`.
Falsifier: the sha-pinned float64 golden `tests/unit/autodiff/test_waveguide_sparam_ad.py:98-141` at
`diff.max() < 1e-6` (`:141`). complex64 rounding at |S| ≤ 1 is about 6e-8, so the golden is expected
to hold; if the delta exceeds 1e-6 the cast is restored and the reason recorded in the PR body. The
golden is never moved.

**Step 4, docs.** `_sparams.py:2106-2130` gains the normalization statement from gap-table row
1c. `docs/guides/support_matrix.md:36` and `:94` gain "multimode (`n_modes>1`) is host-assembled
and outside the differentiable chain".

**Falsifier.** After step 2, `jax.grad` on a tiny WR-90 probe with a traced `eps_override` and
`normalize=True` raises `NotImplementedError` at the public entry. The three sha-pinned golden
equivalence tests (`tests/unit/autodiff/test_waveguide_sparam_ad.py:98-141`, float64, `diff.max() < 1e-6` at
`:141`) stay green. Expected: the `False` and `True` goldens bit-identical; the flux golden moves
with the step-3 cast removal, and then by less than 1e-6. If the flux delta exceeds 1e-6, the cast
removal is reverted — the gate is not.

**Cheap refute.** `pytest tests/unit/autodiff/test_waveguide_sparam_ad.py tests/unit/autodiff/test_waveguide_flux_ad.py tests/unit/autodiff/test_sparam_ad_end_to_end.py`
on CPU.

**Must not.** jnp-ify `extract_waveguide_s_params_normalized` (`waveguide_port.py:2229-2454`) —
that creates new differentiable surface on a lane memory excludes. Re-pin the goldens.

---

## WP2 — the falsifier battery on one common fixture set (CPU; the template)

**Deliverables.** `tests/oracle/test_waveguide_chain_battery.py`,
`tests/fixtures/waveguide_chain_battery/fixture.json`, and
`docs/design_notes/waveguide_chain_battery_predeclaration.md` — **written and committed before
the first run**.

**What this battery is claims-bearing for (decision 4).** Criteria 1, 2 and 3(a)–(c) rest on this
battery alone. Criterion 3(d)'s referee set is the battery's Airy slab and PEC-short **plus** the
five broad-E5 replay bands — WR-340, WR-62, WR-28, WR-15, WR-10
(`tests/fixtures/waveguide_broad_e5/*_broad_e5_envelope.json`, gated in the fast lane by
`tests/crossval/test_waveguide_broad_e5_envelope_gates.py`, zero run cost because they are replay). cv18,
cv19 and the Meep T-junction are cited as 3(d) support only: they are magnitude-only flux gates with
no AD leg, so they can never carry criterion 1 or 3(a).

**Lane placement.** Whether the battery lands in the fast lane or the slow lane is decided by its
measured wall time against the contract's 30 s fast-lane limit, not assumed. Record the measurement
in the PR body, and name the shard if it goes slow.

**Fixture predeclaration (all of this goes in the note before anything is run).**
- Guide: WR-90, a = 22.86 mm, b = 10.16 mm. Domain and dx as in the ladder below.
- DUTs, three: (i) thru, empty guide, used only as the non-vacuity control, never as a gate
  fixture (#395); (ii) PEC-short — a `pec_like` box (`eps_r=1.0, sigma=1e10`), the
  construction at `tests/unit/sparams/test_waveguide_twoport_contract_v1.py:59-60`, which spans the full
  cross-section and is 5 mm thick at x ∈ [0.050, 0.055] m in **that** file's 40 × 20 mm guide;
  the WR-90 fixture must restate thickness and x-position as its own absolute coordinates,
  not inherit those numbers; (iii) an εr = 4 slab, full cross-section, stated the same way.
- Port planes, probe planes and reference planes: absolute physical coordinates for each,
  with the reference-plane offset from each port stated explicitly.
- CPML: derived, not chosen, by the **rule** at
  `tests/unit/sparams/test_waveguide_twoport_contract_v1.py:35-48` —
  `CPML_LAYERS = ceil(0.75 * lambda_g_low / dx)` using the port's **numerical** TE10 cutoff
  from the far-port advisory, not the analytic `c/2a`. That file's `FC_TE10_NUMERICAL = 3.476e9`
  (`:39`) belongs to its own 40 × 20 mm guide; the WR-90 fixture records its own advisory
  cutoff and its own resulting cell count. Only the rule transfers.
- Drive: `num_periods = 40` per drive, with the `settling_db` (or the named substitute)
  quoted per drive in the fixture JSON.
- Lanes: `normalize=False` and `normalize="flux"` only.

**Pre-declared falsifiers.** Each value below comes from an existing gate; none is invented.
- **(a) AD vs FD.** FD legs under a per-test x64 context with a ULP-span assert following
  `_MIN_FD_ULP_SPAN = 1.0e4` at `tests/unit/autodiff/test_msl_ad_fd_converged.py:136` (the gate assert is
  `:556`, the bidirectional falsifier `:629-634`). Objectives: |S11|² (PEC-short and slab),
  |S21|² (slab), and Re/Im S21 (slab), at the band-centre bin. Pass `rel ≤ 0.05`
  (`tests/unit/autodiff/test_sparam_ad_end_to_end.py:298`, `tests/unit/autodiff/test_waveguide_flux_ad.py:84`). Expected
  order 1e-3 (ledger 2.0e-4 at `:3093-3099`). An FD leg below the span floor skips with the
  span printed; it never passes.
- **(b) Reference plane.** Shift left 0.02 m and right 0.08 m, the asymmetric pair actually
  used at `tests/unit/sparams/test_waveguide_twoport_contract_v1.py:257` (an earlier draft of this plan
  wrote ±0.02 m; that pair does not appear in the repo). |S| allclose `rtol=1e-3, atol=1e-4`
  (`:270`); complex S21 invariant (`:276`); ∠S11 rotation equals 2β·Δ against `_compute_beta`
  within 3° (`tests/unit/sparams/test_waveguide_phase_gate.py:259`) and against a continuous analytic β
  within 6° (`PHASE_TOL_DEG = 6.0`, `:63`). Gradient invariance d|S21|²/dθ across the two
  plane sets: **report-only on the first run** against a pre-declared 1e-2, then pinned by
  `gate_from_envelope(measured, quantum=1000)` in the same PR.

  **What the gradient leg actually tests (decision 2).** The plane shift is post-processing: the
  extracted waves are multiplied by the unit-modulus factor `exp(∓jβΔ)` (`waveguide_port.py:1681-1682`)
  whose β comes from the port cross-section and does not depend on θ. So for a **magnitude**
  objective — |S11|², |S21|² — the gradient is invariant up to rounding, and the expected value is
  about 1e-6, three to four orders below the 1e-2 report bar. For a **complex** objective the
  invariance statement is wrong and the correct one is rotation covariance:
  `d(S21·e^{jφ})/dθ = e^{jφ}·dS21/dθ`, so the leg compares the rotated gradient, not the raw one.
  Two defects this can catch: a β that reaches the tape (making the shift factor θ-dependent), and a
  shift factor that is not unit-modulus. A first measurement near 1e-2 rather than 1e-6 is therefore
  a finding, not a tolerance question.

- **(c) dx ladder (decision 6).** **Rungs: {2.54, 1.27, 0.635} mm.** Every rung is `dx = a/N` at
  integer N, so all three realize the same guide. WR-90 has a = 22.86 mm and b = 10.16 mm = a·4/9,
  so N must be a multiple of 9: the three rungs are a/9, a/18, a/36 and give b = 4, 8, 16 cells.
  Slab thickness and every x-position are multiples of 2.54 mm. This **replaces** the earlier
  {2, 1.5, 1} mm, which realizes a as 11.43 / 15.24 / 22.86 cells — three different guides under one
  name, the #703 class. The existing battery ladder has the same defect and is a warning, not a
  model: dx ∈ {3, 2, 1.5} mm in a 40 × 20 mm guide
  (`tests/oracle/test_waveguide_port_validation_battery.py:42`, rungs at `:454`) realizes a as
  13.33 / 20 / 26.67 cells. This WP does not change that test; the new battery does not copy it.
  CPML is scaled to constant physical thickness (the `:449-457` pattern).

  **Gate (unchanged in shape).** `fine_delta ≤ coarse_delta + floor` on |S21| (`:474`), extended to
  |S11| and ∠S21. Floors, pre-declared: 0.005 for magnitude (the existing `:474` value, consistent
  with the production PEC-short spread 0.0004 recorded at `rfx-known-issues.md:3512`), 1° for phase.

  **Two report-first witnesses**, written into the fixture JSON on the first run and pinned by
  `gate_from_envelope` only in a second step: (i) monotonicity of the three-point sequence together
  with the successive-delta ratio; (ii) a Richardson estimate `2*S_fine - S_coarse` compared against
  the oracle on adjacent rung pairs. The cv18 precedent is exactly this — a measured Richardson
  envelope of 0.0051 turned into a 0.01 gate
  (`validation/crossval/18_wr90_iris_modematch.py:162`). PEC-short **magnitude** is excluded from
  the Richardson witness because it is trivially 1; PEC-short **phase** against 2βL is included.

  **Interpretability guard.** Every frequency bin is evaluated and the worst bin is reported, not
  the band-centre bin alone. The rungs are ratio-2, so the successive-delta ratio should approach
  0.5 for a first-order error and 0.25 for second order. A measured ratio far from both means the
  coarse rung is still pre-asymptotic; the ladder is then reported **"not interpretable"** rather
  than passed or failed, and the coarse rung is dropped or a finer one added before any claim.

  **Rasterization guard.** Each rung asserts that the rasterized slab and PEC-short cell counts
  scale exactly with 1/dx. Without it a rung can silently realize a different DUT — the #703 class
  again, one level down from the guide itself.

- **(d) Referee.** PEC-short `0.99 ≤ |S11| < 1.03` (`battery:541`, `:550`); slab against the
  analytic Airy with `MAX_TOL = 0.05` and `MAX_PHASE_TOL_DEG = 15.0`
  (`scripts/diagnostics/build_waveguide_band_broad_e5_envelope.py:33`,
  `scripts/diagnostics/build_waveguide_band_broad_e5_phase_envelope.py:99`), conventions cited.

**Physics gates on the same fixture.** Column power < 1.02 (`battery:307`); magnitude
reciprocity < 0.01 (`battery:340`); complex reciprocity pre-declared at ≤ 0.01, with a first
measurement above that reported rather than absorbed.

**Cheap refute.** Run the battery with the reference-plane shift sign flipped in a local copy:
(b) must go red. `phase_gate:264` describes the wrong-sign error as ~50° and asserts
`> 10°` at `:266`, so a flipped sign that stays inside 3° means the gate does not bind.

**Memory.** Consistent with `rfx-known-issues.md:3093-3099` (AD is done; the battery adds
objectives, not plumbing), `:3384-3395` (no `normalize=True`), `:4093-4112` and `:4315`
(conjugate before any phase comparison), `project_issue527_f32_comparator` (FD validity before
the accuracy gate), and `:3705-3716` (the aperture DROP weight — the ladder declares the
staircase envelope, it does not reopen the weight).

**Must not.** Loosen any quoted tolerance. Introduce a new source construction (reuse
`cfg.e_inc_table` / `h_inc_table`). Import `rfx/probes/refplane.py` code into the waveguide
path — it would add the numpy round-trip at `:539-542`.

---

## WP3 — power-closure witnesses (CPU)

**Scope.**
1. `tests/crossval/test_waveguide_broad_e5_envelope_gates.py`: assert each fixture's `unitarity_min` /
   `unitarity_max`. All five committed uniform envelopes carry those fields and no test reads
   them (`grep unitarity` on that file returns nothing).
2. A new test placing `add_flux_monitor` planes inside the guide on the WP2 slab fixture,
   comparing `1 − |S11|² − |S21|²` from the flux lane against the monitor's net flux ratio.
   This is the only independent witness in the plan: the port column power reuses the port's
   own Poynting integrals, so it and the S-matrix are one witness, not two.
3. Not in this WP (decision 7): wiring `check_reciprocity` into the runtime guard. It exists at
   `rfx/validation.py:468-486` but defaults to `False` (`:342`) and `_sparams.py:816-824`
   never enables it. The warn-only version is sequenced after this WP measures the complex
   reciprocity envelope — #854 item 4, carried in WP7.

**Falsifier.** The unitarity assert passes on the five committed JSONs with the tolerance
derived by `gate_from_envelope` (`tests/_gate_policy.py:89`, quantum 1000). Interior-monitor
closure and port closure agree within 0.02 (the column-power tolerance); a larger disagreement
means one of the two routes is wrong and is reported as such.

**Cheap refute.** Perturb one `unitarity_max` in a copied fixture by +0.01; the new assert must
go red.

**Must not.** Change the `0.6 < mean_power < 1.40` gate at `tests/oracle/test_conservation_laws.py:161`
without a written root cause for the `normalize=True` deficit it documents.

---

## WP4 — NU lane: the β cell-size question is arithmetic, not a measurement (CPU)

**The finding that reshapes this WP.** The expected effect was derived, not guessed. The Yee
correction to β is second order: β(dx) ≈ s_x·(1 + (s_x·dx)²/24), so switching the cell size
from the boundary 1.5 mm to the local 0.75 mm changes β by (s_x·dx)²/24 evaluated at the two
sizes. For WR-90 with a numerical cutoff at 6.557 GHz and a dt set by the fine cell:

| f (GHz) | β (rad/m) | β(dx=1.5 mm) − β(dx=0.75 mm) | Δφ over a 20 mm plane offset |
|---|---|---|---|
| 8 | 96.05 | 0.062 rad/m | 0.07° |
| 10 | 158.24 | 0.280 rad/m | 0.32° |
| 12 | 210.63 | 0.665 rad/m | 0.76° |

A 3° threshold therefore cannot be reached by this mechanism at the plane offsets this lane
uses; a falsifier written against 3° is pre-decided in favour of "leave the boundary cell".

**Second finding: the existing NU fixture cannot exercise the defect at all.**
`tests/unit/autodiff/test_waveguide_nu_flux_ad.py:50-95` builds 30 mm of 1.5 mm cells, then 40 mm of 0.75 mm
cells, then 30 mm of 1.5 mm, smoothed by `smooth_grading(max_ratio=1.3)`, with both reference
planes 0.020 m from the ends (`:90`, `:95`). Running that construction shows the first
non-coarse cell begins at x = 0.030 m, so both shift spans are uniform 1.5 mm cells and the
boundary cell **is** the local cell there. `rfx/nonuniform.py:37-55` documents `dx`/`dy` as
the boundary sizes (`:54`), and `waveguide_port.py:603` reads exactly that.

**Reframed WP (decision 3: no new fixture now).** This is a documented-envelope item, not a code
change:
1. Compute `_compute_beta` (`waveguide_port.py:1419-1469`) and `_compute_mode_impedance`
   (`:1471`) at both cell sizes over the band. This is pure arithmetic — no FDTD, no noise —
   so it discriminates at any precision. Record the table above with the fixture's own numbers.
2. Add a **grading-zone assertion** to the NU reference-plane path: the span from the port plane to
   the reference plane must lie inside one uniform grading zone. Without it, a single β is correct
   only by accident. The assertion is the whole of this step — integrating β over the span is
   **deferred to #854 item 1**, together with a fixture whose plane sits inside the graded region.
   Deferred, not dropped: the assertion makes the untested case fail loudly instead of silently.
3. Write the resulting phase error into the NU phase envelope in `docs/guides/support_matrix.md:94`,
   which today states only that phase is not validated on this lane.
4. Write a NU sibling of WP2(b) and WP2(c). The (b) sibling shifts planes that stay inside a uniform
   zone, which is what the existing fixture offers. The (c) sibling varies base dx at a held grading
   ratio under the same integer-N rule as WP2(c), applied to that fixture's own cross-section; the
   NU broad-E5 envelope varies the ratio, never dx.

**Why no fixture now (decision 3).** The derived effect is 0.07° / 0.32° / 0.76° at 8 / 10 / 12 GHz
over a 20 mm offset, so a fixture built to measure it would measure under 1° and cost a new NU
fixture plus its settling witness. The assertion in step 2 closes the silent-failure path, which is
what the risk actually was.

**Cheap refute.** `pytest tests/unit/autodiff/test_waveguide_nu_flux_ad.py tests/unit/sparams/test_waveguide_nu_nontrivial.py -m slow`
after any change: the gates at `nu_flux_ad:145-161` and `nu_nontrivial:491-500` must stay inside
their own tolerances.

**Memory.** Consistent with `rfx-known-issues.md:198-200` (the NU waveguide lane carries
`settling_db` since PR #841; #827 stays open for the general lane, so quote `settling_db` with
every NU number) and with `project_msl_nu_extractor` (a `NonUniformGrid` has no `*_profile`
attributes; read cell sizes from the grid arrays).

**Must not.** Touch dz-graded accuracy (#810 is a separate track). Loosen the NU 0.02 gate.

---

## WP5 — CI visibility of the slow-lane gates (CPU, workflow only)

**The problem.** `.github/workflows/validation.yml` has concluded `failure` on every run since
2026-07-06; the last `success` was 2026-06-29 and 2026-08-24 was cancelled. Two NU gates that
the contract leans on run only there.

**Wall time, from `.test_durations`.** Two options, and the WP must pick one in writing:

| Move | Tests | CI seconds |
|---|---|---|
| Single tests | `nu_flux_ad::..._grad_finite_and_fd_consistent` 17.354 + `nu_nontrivial::test_nu_nontrivial_matches_uniform` 8.113 | 25.5 |
| Whole modules | the two above plus `nu_flux_ad` forward 6.546 and null 7.902, `nu_nontrivial` slab_verdict 7.965 and shape 0.002 | 47.9 |

25.5 s is close to the contract's 30 s fast-lane limit, and the module fixtures share builds,
so moving single tests does not save the whole difference. Recommendation: move the two single
tests, and state in the PR that the sibling tests stay in the weekly lane.

**Shard balance is load-bearing.** `pyproject.toml:53-57` pins `pytest-split<0.12` precisely
because the weekly shard balance depends on `--store-durations` merge semantics and the
`duration_based_chunks` grouping. Any change here regenerates `.test_durations` and must say so.

**Falsifier.** After the change, `pytest --collect-only -q | grep nu_nontrivial` lists the test
in the PR lane, or the weekly workflow pins the two tests to a named shard and that shard is
green on the next run. Fast-suite wall time grows by ≤ 30 s.

**Must not.** Fix the shard-3 OOM or #797 here; separate issues.

---

## WP6 — artifacts: re-capture pointers, settling witnesses, snapshot definition (CPU)

**Scope, narrowed.** The #812 refresh decision is removed from v1.8 (see the top of this plan
and the contract's exclusions). What remains:
1. **Dangling re-capture pointers.** All five uniform E5 fixtures set `rfx_manifest_path` into
   the gitignored `.omx/physics-gate/...`; the NU E5 fixture points into a different checkout
   (a different checkout's gitignored `.omx/...`); and
   `tests/fixtures/waveguide_broad_e5/waveguide_wr28_kaband_broad_e5_envelope.json:180` cites
   `scripts/vessl_i496_band_absorber_probe.yaml`, which `git ls-files` does not return. (The
   earlier draft attributed that citation to WR-15; it is WR-28.) Every fixture pointer must
   resolve to a tracked path or a tracked VESSL YAML — `scripts/vessl_waveguide_broad_e5.yaml`
   is tracked and is the lane's re-capture entry point.
2. **Settling witnesses.** No committed waveguide fixture records an energy-based `settling_db`.
   Record the substitute explicitly in the ledger row, naming the record-length form at
   `tests/crossval/test_waveguide_nu_broad_e5_envelope_gates.py:170-199` and its stated reason (`:175-176`).
3. **Snapshot definition.** Decide what `tests/data/example_fidelity_snapshot.json` pins for a
   port entry, or state that the host-geometry entry is the whole deliverable. Note that cv18
   is not missing from the snapshot — `tests/_example_fidelity_lib.py:468-471` classifies it as
   `builder_fused_with_solve`, which is why it yields no separable builder entry.

**Falsifier.**
`for p in $(jq -r '..|strings|select(startswith("scripts/"))' tests/fixtures/waveguide_*/*.json); do git ls-files --error-unmatch $p; done`
exits 0 for every path.

**Must not.** Re-pin any tolerance. Remove the `absorber_discipline.status =
"below_floor_accepted"` notes, which all five uniform magnitude envelopes carry. Touch the
#812 fixture.

---

## WP7 — closing PR for the waveguide family (CPU)

**Scope.** One PR linking the WP2/WP3/WP4 artifacts, the per-lane ledger row (status against
criteria 1–3 with file:line), and the support-matrix rows (`docs/guides/support_matrix.md:36`
and `:94`; `sparameter_support_matrix.{md,json}`). Wording: append "chain-closed (v1.8)";
keep "limited"/"experimental" until `ROADMAP.md:41`.

**Row wording, decided (decision 4).** The `:36` row reads "chain-closed (v1.8)" for **uniform
single-mode** S only, and names its evidence: the WP2 chain battery plus the five broad-E5 replay
bands (WR-340, WR-62, WR-28, WR-15, WR-10). It keeps the two fences it already needs — "phase
referee = Airy only" and "junctions and nonuniform excluded". cv18, cv19 and the Meep T-junction may
appear as 3(d) support, never as the chain-closure evidence.

**Follow-on, not part of this PR (decision 7).** The warn-only runtime reciprocity check for the
waveguide family is sequenced after WP2 measures the complex reciprocity envelope: the tolerance
comes from `gate_from_envelope`, and the preflight emission-site counters in
`tests/unit/preflight/test_preflight_advisory_emission_contract.py` (`_FROZEN_LITERAL_CODE_COUNT` at `:202`, test at
`:218`) are updated in the same PR as the new advisory. Tracked as #854 item 4.

**Verifier lane, named.** A `verifier` agent that did not author any of WP1–WP6 reviews the PR
against a **git archive of the PR head**, not the live worktree (a builder editing the same
worktree moves the review target silently). What the verifier signs, in the PR body: (i) every
claim in the ledger row traces to a green test name on the PR's own CI run; (ii) the
predeclaration note predates the first battery run in the commit history; (iii) no committed
tolerance moved. The verifier's sign-off comment is the artifact.

**Must not.** Use the word "supported". Self-approve.

---

## WP8 — lumped/wire tail (GPU/VESSL likely for #819)

**#819 — dx-refinement study.** Fixture: the 2-port wire THRU behind
`tests/unit/sparams/test_lumped_twoport_vi_validation_battery.py:722`
(`test_thru_passivity_singular_values`). Recorded state: sv_max 1.003227 at 3 GHz falling
monotonically to 0.9874 at 7 GHz (`:727-731`), reproduced under f64 fields, 4× and 2× step
counts and complex128 algebra, mechanism unidentified (`rfx-known-issues.md:137-141`; gh #819
OPEN, "Thru-fixture singular-value excess is systematic, not extraction float noise"). The
gate `_THRU_MAX_SINGULAR_VALUE = 1.01` (`:326`) and every measured number are unchanged by
that work.

Pre-declared falsifier, the issue's own candidate 1: run at dx, dx/2 and dx/4 with CPML
thickness held constant. If `(sv_max − 1)` at 3 GHz falls by ≥ 2× per halving, the excess is
Yee/CPML discretization and the envelope is bounded by a fitted order. If it changes by < 20 %
across the ladder, discretization is refuted and candidate 2 is the next single attempt.
Anything between is non-closing: STOP and write the redesign. **The expected value is not
known — this is a measurement**, and the existing gate pins the envelope until it lands. dx/4
is a long run; GPU/VESSL lane.

**#683 — implementation.** Implement the POST-injection uniform-lane flip exactly as
pre-declared in `docs/design_notes/issue683_decomposer_flip_predeclaration.md` §5 (falsifiers,
committed before implementation, never widened), with the verdict recorded in
`issue683_sampling_order_decision_protocol.md` §9 (results appended 2026-08-29; PRE refuted).
Cheap refute: the G2 identity `V_pre = V_post + d_par·W` on bit-identical geometry.

**Memory.** Consistent with `rfx-known-issues.md:4278-4306` (the #313 reference-plane lane;
diagonals byte-frozen; never de-embed with ω/c or a nominal 50 Ω) and with
`feedback_negated_closing_keyword` (keep "#819" away from fix/close words in PR titles until
it is actually closed).

**Must not.** Fold #819 into the waveguide battery — it is a lumped/wire thru fixture
(`ROADMAP.md:39`). Attempt a third #819 mechanism without a written new falsifier (R2).

---

## Appendix A — gap table against the contract (main 1c38b0d7, 2026-09-02)

# Rectangular-waveguide chain-closure gap table (main 1c38b0d7, 2026-09-02)

Every file:line below was re-checked against `main` at 1c38b0d7 on 2026-09-02.

**Status vocabulary.**
- **VALIDATED (fast)** — a gate exists and ran green in the PR lane.
- **VALIDATED (slow)** — the gate exists and the individual test PASSED, but only
  inside `.github/workflows/validation.yml`, whose workflow-level conclusion has been
  `failure` on every run since 2026-07-06 (last `success` 2026-06-29; 08-24 was
  cancelled). A per-test pass inside a red workflow is evidence, not CI coverage.
- **PARTIAL** — evidence exists but does not cover the criterion as written.
- **DIAGNOSTIC** — frozen replay or reporter script only.
- **ABSENT** — nothing.

**Measure discipline.** Where two numbers exist for one quantity, both are given with
the measure named. A bare percentage with no measure is what the ledger's
"quote the measure with the number" rule exists to prevent.

| Contract sub-item | Uniform lane | Uniform evidence | NU lane | NU evidence | What v1.8 must add |
|---|---|---|---|---|---|
| **1a. In-graph extraction → complex S_ij(f)** | VALIDATED (fast) for `normalize=False` and `normalize="flux"`; ABSENT for `normalize=True` and for `n_modes>1` | On tape: `rfx/sources/waveguide_port.py:1955` (`b_recv / safe_a`), stacks at `:1957`/`:1961`; flux lane `:1967-2223` (magnitude `:2208-2215`, stack `:2221-2223`). `normalize=True` is a numpy shell: `np.zeros(..., complex64)` `:2327`, `np.array(...)` on extracted waves at `:2383`, `:2397`, `:2434`. Multimode host-side: `:2836` (normalize), `:3026`/`:3036` (flux). Gates: `tests/unit/autodiff/test_sparam_ad_end_to_end.py:228-306`, `tests/unit/autodiff/test_waveguide_flux_ad.py:73-86`, `tests/unit/autodiff/test_waveguide_sparam_ad.py:98-141` (sha-pinned float64 goldens, `diff.max() < 1e-6` at `:141`). | VALIDATED (slow) for `normalize="flux"`; `eps_override`/`sigma_override` rejected on every other NU mode | `rfx/api/_sparams.py:7574` (device run threads the override) and `:7592` (vacuum reference run); flux magnitude `:7700-7717`; stack `:7766`. `rfx/runners/nonuniform.py:1400-1414` keeps `flux_spectrum` on the tape. Dispatch guard `_sparams.py:2332-2341`, locked by `tests/unit/sparams/test_waveguide_nu_sparam.py:377-390` (two `NotImplementedError` tests). Gate `tests/unit/autodiff/test_waveguide_nu_flux_ad.py:145-161`. | A fail-fast guard on the **uniform** `normalize=True` lane mirroring `_sparams.py:2332-2341`, so a traced `eps_override` raises `NotImplementedError` at dispatch instead of a `TracerArrayConversionError` deep in the extractor. The raise site must be recorded by a committed red test first — the lane has three `np.array` sites (`:2383`, `:2397`, `:2434`) and which one fires has not been measured. Written scope statement that multimode is outside the v1.8 battery. Removal of the flux lane's `complex64` cast at `waveguide_port.py:2217` (decision 1), so all three lanes follow `JAX_ENABLE_X64` as `normalize=False` already does through `_rect_dft` (`waveguide_port.py:1583-1585`). |
| **1b. Reference-plane shift** | VALIDATED (fast) at value level | `waveguide_port.py:1655-1684` `_shift_modal_waves` (`exp(∓jβ·shift)` at `:1681-1682`, `step_sign` handling in the docstring `:1666-1670`); shift distance `_sparams.py:2799-2805`; β from `_compute_beta` `:1419-1469` (Yee-discrete branch `:1444-1459`), cutoff from the discrete 2D eigenvalue by default (`rfx/api/__init__.py:2283`, `mode_profile="discrete"`). Gates: `tests/unit/sparams/test_waveguide_phase_gate.py:211-266` (rotation vs analytic, `≤3.0°` at `:259`, wrong-sign witness asserted `>10°` at `:266` and described as ~50° at `:264`); `:129-145` (`PHASE_TOL_DEG = 6.0` at `:63`, residual vs an independent β). | PARTIAL | Shift computed at `_sparams.py:7635-7645`. But `waveguide_port.py:603` sets `dx = float(grid_obj.dx)`, and `rfx/nonuniform.py:37-55` documents `dx`/`dy` as the **boundary** cell (`:54`), so β and Z_TE use the boundary cell at a graded port. No NU phase or reference-plane gate; `docs/guides/support_matrix.md:94` states phase is not validated on this lane. | Both lanes: record that `shift_m` is a Python float (`waveguide_port.py:1679`, `if shift_m == 0.0`) — the plane is static geometry, not a design variable. NU: the boundary-vs-local-cell β question is **arithmetic, not a measurement** (see the derivation in the plan, WP4); it also needs the check that the shift span lies in one grading zone. In the existing NU AD fixture it does: `tests/unit/autodiff/test_waveguide_nu_flux_ad.py:50-95` grades 1.5 mm → 0.75 mm with the fine block starting at x = 0.030 m, and both reference planes sit 0.020 m from the ends, so the whole shift span is uniform 1.5 mm cells. **That fixture therefore cannot exercise this defect at all.** |
| **1c. Impedance normalization** | PARTIAL | S is the ratio `b_i/a_j` (`waveguide_port.py:1955`) with per-port discrete Z_TE from `_compute_mode_impedance:1471` via `_extract_global_waves:1540`. No `sqrt(Z_i/Z_j)` renormalization exists anywhere in the port path. Flux magnitude is `sqrt(P/P_inc)` (`:2208-2215`). Documented only as a caveat: `_sparams.py:2106-2130` ("Yee impedance mismatch Z_TE_num/Z_TE_exact ≈ 3 % at dx/λ = 0.07", `:2111-2112`); `docs/public/guide/probes-sparams.mdx` names no reference impedance. | PARTIAL | Same ratio form at `_sparams.py:7731` (diagonal) and `:7749` (off-diagonal); same Z_TE from the boundary `dx` (see 1b). | State in the public docstring that S is a voltage-wave ratio referenced to each port's own discrete TE10 Z_TE, and equals a power-wave S only when all ports share a cross-section. Add either a guard or an explicit same-cross-section scope sentence, plus one dissimilar-port test. No new normalization math unless the PI asks. |
| **1d. No host round-trip θ→S** | PARTIAL | Break-free: `normalize=False`, `normalize="flux"`. Hard breaks: `normalize=True` (`waveguide_port.py:2383`, `:2397`, `:2434`), multimode (`:2836`, `:3026`, `:3036`), traced `freqs` (`rfx/api/__init__.py:2472`, documented by `tests/unit/autodiff/test_waveguide_sparam_ad.py:206-257`), plane distance as a Python float (`waveguide_port.py:1679`). Tracer-safe skips, not breaks: settling witness (`waveguide_port.py:1794`, tracer → NaN at `:1826-1827`), passivity guard (`_sparams.py:766-824`), `rfx/probes/probes.py:745-764` (flux check skipped under jit/grad). | VALIDATED (slow) on flux | `_sparams.py:7574`/`:7592`; settling witness attached `:7758-7763`. | The guard from 1a, plus a written list of what is and is not a traced input (θ = `eps_override` / `sigma_override` only). |
| **2a. Passivity** | VALIDATED (fast) — warn-only runtime guard plus live gates | Guard `_sparams.py:766-824` → `rfx/validation.py:447-466`, wired at `:2414`, `:2752`, `:2945` with `passivity_tol = 2.0` for `normalize=False` and `0.10` otherwise; warn unless `strict_passivity`. `_project_passive` (`:699`) is **not** applied on this family (only MSL `:4157`, mixed `:5101`). Live gates: `tests/oracle/test_waveguide_port_validation_battery.py:278-309` (flux, εr=4 DUT, max column power `<1.02` at `:307`, measured 1.0005); `:500-554` PEC-short on `normalize=False` (min `≥0.99` at `:541`, max `<1.03` at `:550`); `tests/unit/sparams/test_waveguide_twoport_contract_v1.py:129-197` (dielectric, max column power `<1.05` at `:187`, measured 1.0128 at `:148`/`:183`) and `:199-231` (PEC-short, `<1.05` at `:226`, measured mean 0.9998 / max 1.0208 at `:224`); `tests/unit/sparams/test_sparam_passivity_guard.py` runs in the required `guards-and-preflight` job. | VALIDATED (slow) at the tight tolerance | `tests/unit/sparams/test_waveguide_nu_nontrivial.py:430-523` (dx-graded, flux, `|P_col − 1| < 0.02` on both drives at `:491` and `:496`). The fast lane carries only `tests/unit/sparams/test_waveguide_nu_flux.py:100-127` at 1.10 (`:121`). Graded-dy: frozen replay only. Multimode: ABSENT — `tests/unit/ports/test_multimode_waveguide.py:362-401` asserts shape and mode map only. | Nothing on values. Make the NU 0.02 gate visible in the PR lane or pin its shard (see WP5 for the wall-time arithmetic), and add the multimode scope statement. |
| **2b. Reciprocity** | VALIDATED (fast) — **magnitude only** | `battery:320-342` mean `‖S21|−|S12‖/max < 0.01` at `:340` (measured 0.0005, `:337-338`); `:398-...` asymmetric case at `<0.10`; `contract_v1:196` `mean(recip) < 1e-3`; `tests/oracle/test_conservation_laws.py:170-202` `< 0.05` at `:201` (measured 0.0414). `rfx/validation.py:468-486` implements a reciprocity check, but the runtime guard never enables it — `check_reciprocity` defaults to `False` (`rfx/validation.py:342`) and `_sparams.py:816-824` passes only `check_passivity=True`. | VALIDATED (slow) — magnitude only | `tests/unit/sparams/test_waveguide_nu_nontrivial.py:500` (`recip.max() < 0.01`), where `recip = np.abs(s21_nu - s12_nu)` at `:480` is built from magnitudes taken at `:474-475` — so this is a magnitude gate, not a complex one. | A **complex** reciprocity gate `max_f |S21 − S12| / max|S|` on the same fixtures (WP3). Runtime wiring of `check_reciprocity` is optional and not a contract requirement. |
| **2c. Power closure** | VALIDATED (fast) on flux; PARTIAL otherwise | Flux: `battery:278-309` (1.0005 against 1.02). `normalize=True`: `tests/oracle/test_conservation_laws.py:129-163` gates `0.6 < mean_power < 1.40` at `:161` around a measured ~0.73 — that documents a deficit, it does not demonstrate closure. The five committed E5 fixtures carry `unitarity_min`/`unitarity_max` fields, but `grep unitarity tests/crossval/test_waveguide_broad_e5_envelope_gates.py` returns nothing, so no test reads them. Independent witness ABSENT: no `add_flux_monitor` appears in any waveguide test, and column power reuses the port's own Poynting integrals. | VALIDATED (slow) | `nu_nontrivial:491-513`; record-length witness `tests/crossval/test_waveguide_nu_broad_e5_envelope_gates.py:170-199` (max\|S11\| shift under a doubled record window `< MAX_TOL/10`, column power at np=60 within 1e-3 of unity). | Assert `unitarity_min`/`unitarity_max` per case in the frozen E5 gate. Add one interior-flux-monitor witness of `1 − |S11|² − |S21|²` on a lossless DUT — two routes through the same port DFT are one witness, not two. |
| **3a. AD-vs-FD on an S-native objective** | PARTIAL | `tests/unit/autodiff/test_sparam_ad_end_to_end.py:228-306`: objective `jnp.real(jnp.sum(jnp.abs(S[:,:,k0])**2))` at `:257-259`, `normalize=False`, `rel_err < 0.05` at `:298`, FD in float32 at h = 1e-3 (`:288`) with **no** ULP-span validity assert. The repo's ULP-span pattern is `_MIN_FD_ULP_SPAN = 1.0e4` at `tests/unit/autodiff/test_msl_ad_fd_converged.py:136` (gate assert `:556`, bidirectional falsifier `:629-634`), also used by `tests/unit/autodiff/test_coax_two_port_ad.py`. `tests/unit/autodiff/test_waveguide_flux_ad.py:73-86`: \|S21\|² at bin 2, flux, `rel ≤ 0.05` at `:84`, float32 FD at h = 0.05. | PARTIAL (slow) | `tests/unit/autodiff/test_waveguide_nu_flux_ad.py:145-161` (\|S21\|² flux, `rel ≤ 0.05`) and the perfect-null gradient test. No NU `sigma_override` FD test — stated in `docs/guides/sparameter_support_matrix.md:461-462`. | Float64 FD legs with a `_MIN_FD_ULP_SPAN`-style validity assert **before** the accuracy gate (the #527 class); add \|S11\|² and one complex-S objective; add the NU `sigma_override` leg. No new AD plumbing — the tape already carries complex S (`rfx-known-issues.md:3093-3099`, RESOLVED 2026-05-25). |
| **3b. Reference-plane-shift invariance** | PARTIAL | `contract_v1:233-300`: shift is left 0.02 m / right 0.08 m (`:257`), `|S|` allclose `rtol=1e-3, atol=1e-4` at `:270`, complex S21 invariant at `:276`, S11 rotation asserted only as `> 0.1 rad` at `:285` — and the whole test runs on `normalize=True`. `phase_gate:211-266` checks the one-way βΔ on a single-port incident wave to 3°. Gradient invariance under a shift: ABSENT (`grep -l "jax.grad\|value_and_grad" tests/ | xargs grep -l reference_plane` → no hit). | ABSENT | No NU reference-plane test. The NU AD fixture sets non-zero planes (`tests/unit/autodiff/test_waveguide_nu_flux_ad.py:90`, `:95`) but never varies them. | A value gate on the two differentiable lanes asserting S11 rotation = 2βL against the Yee β to a pre-declared tolerance; a gradient-invariance falsifier; a NU sibling. |
| **3c. Mesh-refinement consistency** | PARTIAL | `battery:449-479`: dx ∈ {3, 2, 1.5} mm at 6 GHz, `normalize=True`, `fine_delta ≤ coarse_delta + 0.005` at `:474` and `fine_delta < 0.10` at `:478`. cv18 Richardson replay: `tests/fixtures/wr90_iris_modematch/fixture.json` pins `richardson_measured_envelope_abs = 0.0051` against `richardson_gate_abs = 0.01`. The broad-E5 gates require ≥2 dx values per band (`tests/crossval/test_waveguide_broad_e5_envelope_gates.py:81`) but make no monotonic statement. Conformal-PEC ladder is closed by a strict xfail (`tests/unit/geometry/test_subpixel_pec.py:614-637`). | ABSENT | The NU broad-E5 envelope varies the grading ratio at fixed base dx (`mesh_axis_kind = nonuniform_dy_profile_ratio`), never dx itself. | A ladder on \|S11\| and ∠S21 for PEC-short and slab on the flux/`False` lanes, and a NU base-dx ladder. Note the shape limitation: `fine ≤ coarse + floor` is a **non-increase** test, not a convergence test — decision 6 keeps that gate and adds the monotonicity and Richardson witnesses plus the integer-N rung, worst-bin and rasterization guards. |
| **3d. External / analytic referee** | VALIDATED (fast, magnitude); phase = analytic Airy only | Airy magnitude, 5 bands: `tests/crossval/test_waveguide_broad_e5_envelope_gates.py:73-98`, `MAX_TOL = 0.05` (`scripts/diagnostics/build_waveguide_band_broad_e5_envelope.py:33`), worst case WR-15 at 0.041399. Airy phase: `tests/crossval/test_waveguide_broad_e5_phase_gates.py:67-99`, `MAX_PHASE_TOL_DEG = 15.0` (`build_waveguide_band_broad_e5_phase_envelope.py:99`), worst WR-28 at 11.9897°. Palace E4: `tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json` — `summary.max_mag_abs_diff = 0.0707` against `max_mag_abs_tol = 0.1`, `summary.mean_mag_abs_diff = 0.009434` against `mean_mag_abs_tol = 0.07`, `provenance.status = "STALE (provenance settled; refresh decision open)"`. Meep T-junction: measured `max_mag_abs_diff = 0.09154` against `EXPECTED_XFDTD_TOL = 0.11` (`tests/crossval/test_waveguide_tjunction_e4e5_gates.py:103`). cv18: 0.0232 measured against 0.04. cv19: `f0_measured_envelope_mhz = 12.123` against `f0_gate_mhz = 19.0`. Group delay WR-340: worst 0.0320 ns against 0.042 ns (`tests/oracle/test_waveguide_group_delay_tolerance_envelope.py:36`). | VALIDATED (slow, magnitude, graded-dy only) | Airy: `MEASURED_MAX_ENVELOPE = 0.001081` (`tests/crossval/test_waveguide_nu_broad_e5_envelope_gates.py:66`) under a derived `MAX_TOL` and an outside ceiling of 0.0013 (`:88`). Palace: **two measures, name which** — the per-pair worst max is 0.008529 and the per-pair worst mean is 0.002998 (`tests/crossval/test_waveguide_nu_broad_e4_comparison_gates.py:60-61`, ceilings 0.010 / 0.004 at `:83-84`); the *summary* mean 0.000709 is the number `docs/guides/sparameter_support_matrix.md:456` quotes, and `:57-58` records that deriving a gate from it "failed 1 of 5 pairs on its first run". No dz-graded evidence (#810, OPEN). | The #812 refresh decision on the uniform Palace artifact is **not** a v1.8 item (see the contract's exclusions). The analytic Airy is the v1.8 phase referee (decision 5); a gated conjugation-corrected Meep phase leg is #854 item 3. |
| **4a. Pinned envelopes** | VALIDATED (fast) as replay plus bounded-margin locks; the re-capture chain is partly off-tree | Goldens sha-pinned (`tests/unit/autodiff/test_waveguide_sparam_ad.py:107-120`). Gate derivation is shared: `gate_from_envelope` (`tests/_gate_policy.py:89`) with `ENVELOPE_GATE_MULTIPLIER = 1.5` (`:81`). All five uniform magnitude envelopes carry `absorber_discipline.status = "below_floor_accepted"` (#496). Dangling pointers: every uniform `rfx_manifest_path` points into the gitignored `.omx/physics-gate/...`; the NU E5 one points into a different checkout (a different checkout's gitignored `.omx/...`); the WR-28 fixture's absorber note (`waveguide_wr28_kaband_broad_e5_envelope.json:180`) cites `scripts/vessl_i496_band_absorber_probe.yaml`, which `git ls-files` does not return. No committed waveguide fixture records an energy-based `settling_db`. | same | same | Fix the dangling re-capture pointers. Record the substitute settling witness explicitly in the ledger row, naming the record-length form (`tests/crossval/test_waveguide_nu_broad_e5_envelope_gates.py:170-199`) and the reason it is a substitute: that file's own docstring at `:175-176` states rfx exposes no total-energy monitor. |
| **4b. Fidelity-snapshot entries** | PARTIAL | `tests/data/example_fidelity_snapshot.json` carries cv11, cv19, tmtt taper, tutorial and inverse-design variants, each stamped "NOT AUDITED by this report". cv18 **is** registered — `tests/_example_fidelity_lib.py:468-471` classifies `validation/crossval/18_wr90_iris_modematch.py` as `builder_fused_with_solve` (the builder and the solve live in one function), which is why it produces no separable builder entry. | same | same | Define what a snapshot entry pins for a port (realized aperture, cutoff, evanescent advisory), or state that the host-geometry entry is the whole deliverable. |
| **4c. Ledger record** | PARTIAL | Live and correct: `rfx-known-issues.md:3093-3099` (waveguide end-to-end AD RESOLVED 2026-05-25, FD↔AD rel_err 2.0e-4), `:3384-3395` (`normalize=True` limitation), `:196-197` (the contract doc is the first v1.8 deliverable), `:198-200` (#827 stays open for the general lane), `:256-260` (v2.0 milestones). Stale: `docs/agent-memory/index.md:249` still calls the cv11 ∠S21 offset an OPEN residual and points at roadmap W3.4, and `:280` repeats it, while `rfx-known-issues.md:4093-4112` closed it on 2026-07-02 as a Meep time-convention conjugation. `rfx-known-issues.md:3394` still says "NU mesh not yet supported (raises)" for the flux lane, which is now the gated NU lane. The open-issue list at `:27-28` still carries #822 and #823, which GitHub reports CLOSED and `:120` records as closed with #837. No per-lane in-graph status table exists anywhere in the repo. | same | same | The contract document plus one per-lane row in `rfx-known-issues.md`; the three stale statements above. |

---

## Appendix B — decisions taken (2026-09-02)

All seven were decided by the PI on 2026-09-02. Each entry gives the decision, then why. Deferred
work is collected in **#854**; the one-line summary is the "Decisions" section at the top.

---

## 1. The flux lane's `complex64` cast — REMOVE it

`rfx/sources/waveguide_port.py:2217` hard-casts the assembled flux column to `complex64`.
`normalize=False` instead follows the `freqs` precision through `_rect_dft` (the rule is stated at
`waveguide_port.py:1583-1585`), and the NU flux lane does not carry this cast.

**Decision: remove the cast**, so `normalize="flux"` follows `JAX_ENABLE_X64` like its two siblings.
The alternative — keep it and document that one lane silently ignores the precision knob — costs
nothing to ship and everything to explain, and it would leave the support matrix asserting a
precision promise the lane does not keep. The risk is bounded by a falsifier that already exists:
the sha-pinned float64 golden `tests/unit/autodiff/test_waveguide_sparam_ad.py:98-141` at `diff.max() < 1e-6`
(`:141`). complex64 rounding at |S| ≤ 1 is about 6e-8, so the golden is expected to pass. If it
exceeds 1e-6 the cast is restored and the reason recorded — the golden is never moved. Implemented
as WP1 step 3, in its own commit.

---

## 2. Gradient-invariance tolerance under a reference-plane shift — REPORT-THEN-PIN

**Decision: keep what the contract says** — the first run reports against a pre-declared 1e-2, and
the same PR pins `gate_from_envelope(measured, quantum=1000)`. Pinning 1e-2 up front would be a
tolerance nobody measured; leaving it permanently report-only would give criterion 3(b) no failing
condition. Report-then-pin is how every other envelope gate here is derived
(`tests/_gate_policy.py:89`, multiplier 1.5 at `:81`).

**Added to the contract and to WP2(b): the physics.** The shift is post-processing by a unit-modulus
factor `exp(∓jβΔ)` (`waveguide_port.py:1681-1682`) whose β comes from the port cross-section and does
not depend on θ. For a **magnitude** objective (|S11|², |S21|²) the gradient is therefore invariant
up to rounding — expected around 1e-6, not 1e-2. For a **complex** objective (Re/Im S21) plain
invariance is the wrong statement; the right one is rotation covariance,
`d(S21·e^{jφ})/dθ = e^{jφ}·dS21/dθ`. Stating this changes what the leg is for: it catches a β that
reaches the tape, or a shift factor that is not unit-modulus. Without the split, a measurement at
1e-3 would look like a comfortable pass when it is three orders worse than the physics allows.

---

## 3. A NU fixture with a reference plane inside the graded region — DO NOT BUILD ONE NOW

Measured on main: the existing NU AD fixture cannot exercise the boundary-cell-β question at all.
`tests/unit/autodiff/test_waveguide_nu_flux_ad.py:50-95` puts both reference planes 0.020 m from the ends, and the
first non-coarse cell begins at x = 0.030 m, so both shift spans are uniform 1.5 mm cells. The
derived size of the effect at WR-90 is 0.07° / 0.32° / 0.76° at 8 / 10 / 12 GHz over a 20 mm offset
(WP4's table).

**Decision: option (a).** WP4 keeps its steps 1–3 — the derived table in the fixture's own numbers,
a grading-zone assertion in the NU reference-plane path, and the NU phase envelope written into
`docs/guides/support_matrix.md:94`. A fixture measuring an effect under 1° does not earn a new NU
fixture plus its settling witness, and the assertion removes the failure mode that mattered: the
case arising silently. The fixture **and** β integration over the shift span are **deferred to #854
item 1** — deferred, not dropped.

---

## 4. Which fixtures are claims-bearing for the v1.8 waveguide row — BATTERY PLUS THE FIVE BANDS

| Fixture | Referee | AD leg | Criteria it can support |
|---|---|---|---|
| WP2 chain battery (new) | analytic Airy + PEC-short | yes | 1, 2, 3(a)–(d) |
| Broad-E5 five bands | analytic Airy, magnitude and phase | no | 3(d), 4 |
| cv18 iris mode-match | mode-matching oracle, magnitude | no | 3(c) partially, 3(d) |
| cv19 iris filter | f0 against an oracle | no | 3(d) |
| Meep T-junction | external, magnitude | no | 3(d) |

**Decision: option (b).** The WP2 battery carries criteria 1, 2 and 3(a)–(c) on its own. The
referee set for 3(d) is the battery's Airy slab and PEC-short **plus** the five broad-E5 replay
bands — WR-340, WR-62, WR-28, WR-15, WR-10
(`tests/fixtures/waveguide_broad_e5/*_broad_e5_envelope.json`, gated by
`tests/crossval/test_waveguide_broad_e5_envelope_gates.py`). They are replay gates, so widening 3(d) from one
band to five costs no run time. cv18, cv19 and the Meep T-junction are cited as 3(d) support only:
magnitude-only flux gates with no AD leg can never carry criterion 1 or 3(a).

The support-matrix wording that follows (WP7) is "chain-closed (v1.8)" for **uniform single-mode**
S, naming the battery and the five bands, and keeping "phase referee = Airy only; junctions and
nonuniform excluded".

The cv18/cv19 phase legs are **deferred to #854 item 2**. Both scripts currently say phase is not
claimed — `validation/crossval/18_wr90_iris_modematch.py:54` ("phase: NOT claimed (magnitude-only
lane posture)") and `validation/crossval/19_wr90_iris_filter_aghanim.py:58`. That is a **policy, not
a limit**: cv18's mode-matching oracle is complex-valued, so a phase leg is available whenever the
lane posture changes.

---

## 5. Phase referee beyond the analytic Airy — AIRY SUFFICES FOR v1.8

**Decision: option (a).** rfx's ∠S21 matches the analytic Airy insertion phase to ≤ 0.89°
(`rfx-known-issues.md:4113-4114`, the W3.4 cv11 resolution), and the frozen phase envelopes are
gated at `MAX_PHASE_TOL_DEG = 15.0`
(`scripts/diagnostics/build_waveguide_band_broad_e5_phase_envelope.py:99`), worst measured 11.9897°
on WR-28. That is a real phase referee, and it is enough for a v1.8 chain-closure claim that already
fences phase as Airy-only.

A conjugation-corrected **Meep** phase leg as a *gated* referee is **#854 item 3**, not a v1.8
requirement, for three reasons. The conjugation is understood
(`rfx-known-issues.md:4093-4112`: Meep carries `exp(-iωt)`, rfx reports `exp(+jωt)`), but the
2026-09-01 external lane still leaves a PEC-short residual of about 17.55° mean and 21.68° max
*after* conjugation (VESSL run 369367257496) — a gate cannot be pinned on an unexplained residual of
that size. Those two residual numbers are read from that run's log (the cv11 informational row
"pec-short S11 (rfx vs conj(MEEP), time-convention corrected)"), backed up under the local
`docs/vessl-logs/` directory as `rfx-v170-crossval-external_369367257496_completed.log`; no
committed JSON carries them, so the leg's first step is to commit them as a fixture. Palace and
OpenEMS phase conventions are unverified here (the same log shows Palace slab phase mean |Δ| 90°). And the Meep runtime was lost
in the pod restart, so the leg is not even runnable today without a rebuild.

---

## 6. Monotone-approach clause on the dx ladder — KEEP THE GATE, ADD REPORT-FIRST WITNESSES

**Decision: gate stays option (a)** — `fine_delta ≤ coarse_delta + floor`, the shape the existing
`tests/oracle/test_waveguide_port_validation_battery.py:474` gate has, with its limitation stated. A
monotonicity gate can go red for physical reasons (band-edge behaviour, staircase parity) and would
need a root cause each time, which is a poor trade for a first battery.

**Added: two report-first witnesses** — (b) monotonicity plus the successive-delta ratio, and (c) a
Richardson estimate `2*S_fine - S_coarse` against the oracle on adjacent rung pairs. Both are
recorded in the fixture JSON on the first run and pinned by `gate_from_envelope` afterwards, exactly
the path cv18 took from a measured envelope of 0.0051 to a 0.01 gate
(`validation/crossval/18_wr90_iris_modematch.py:162`). PEC-short magnitude is excluded from the
Richardson witness as trivially 1; PEC-short phase against 2βL is included.

**Added: three structural guards**, without which the ladder measures the wrong thing.

1. **Integer-N rungs.** The ladder is `dx = a/N` with integer N, so every rung realizes the *same*
   guide. WR-90 has a = 22.86 mm and b = 10.16 mm = a·4/9, so N must be a multiple of 9: the rungs
   are **{2.54, 1.27, 0.635} mm** = a/9, a/18, a/36, giving b = 4, 8, 16 cells, with slab thickness
   and all x-positions multiples of 2.54 mm. This replaces the plan's earlier {2, 1.5, 1} mm, which
   realizes a as 11.43 / 15.24 / 22.86 cells — three different guides under one name, the #703
   class. The existing battery ladder has the same property and is a warning rather than a model:
   dx ∈ {3, 2, 1.5} mm in a 40 × 20 mm guide
   (`tests/oracle/test_waveguide_port_validation_battery.py:42`, `:454`) realizes a as 13.33 / 20 / 26.67
   cells.
2. **Worst-bin reporting and an interpretability flag.** Every frequency bin is evaluated and the
   worst reported. At ratio-2 rungs the successive-delta ratio should approach 0.5 for a first-order
   error and 0.25 for second order; far from both means the coarse rung is pre-asymptotic, and the
   ladder is reported "not interpretable" rather than passed or failed.
3. **Rasterization scaling.** Each rung asserts that the rasterized slab and PEC-short cell counts
   scale exactly with 1/dx.

Fast-lane or slow-lane placement for the battery is decided by the wall time WP2 measures, not
assumed here.

---

## 7. Runtime reciprocity warning — OFF FOR NOW, WARN-ONLY LATER

`rfx/validation.py:468-486` implements a reciprocity check. `check_reciprocity` defaults to `False`
(`:342`) and the runtime guard at `rfx/api/_sparams.py:816-824` never enables it, so no `run()` ever
warns on a non-reciprocal S.

**Decision: option (a) now, option (b) sequenced after WP2.** Turning the warning on today means
choosing a tolerance before the complex reciprocity envelope has been measured, and the warning
would fire across the existing suite with no measured basis for the threshold. So: WP2 measures the
envelope first; then a warn-only check for the **waveguide family** takes its tolerance from
`gate_from_envelope`, and the preflight emission-site counters in
`tests/unit/preflight/test_preflight_advisory_emission_contract.py` (`_FROZEN_LITERAL_CODE_COUNT` at `:202`, test
at `:218`) are updated in the same PR. Tracked as **#854 item 4**, carried in WP7 as a follow-on,
and listed in the contract's "explicitly not required for v1.8".
