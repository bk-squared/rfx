# Graded-mesh S-parameter accuracy — the WR-90 control: pre-declaration

**Status:** pre-declaration. This note and the driver
`scripts/diagnostics/graded_mesh_sparameter_accuracy_measure.py` are committed
before the measurement arms run. Every window, profile, expectation and
decision rule below is frozen from this commit; results are appended under
"Results", never edited into the windows. A fired window is a result.

**Tracker:** #810, Tier 1, the WR-90 control case only. The MSL case (cv06b)
and the Tier-2 external-solver cases (cv07 / cv15) are out of scope here.

**Tree:** worktree `agent-a64f3761bf9faad17`, branch off `origin/main`
`6d721a56` ("cv04 re-measured on the shipped absorber", 2026-09-16).
**Import provenance:** `PYTHONPATH=<worktree> python3`; `rfx.__file__` prints
`<worktree>/rfx/__init__.py` (there is no editable install in this pod). The
driver records `rfx.__file__`, the commit, `git_dirty`, `argv`, the JAX
version and backend into every JSON it writes.

**Lane:** CPU, local, no VESSL. Measured bring-up cost: 12–16 s per cell at
the `mid` rung, so all 14 cells fit in about 5 minutes.

---

## 0. Memory citation (R1) and premise verification

| entry | load-bearing sentence | this plan is … |
|---|---|---|
| `docs/guides/physics_validation_evidence_rule.md` line 129 | "Nonuniform S-parameter paths — Shadow unless they pass a co-refined analytic/external field or S-parameter oracle. 'Finite gradient' is E0/E1, not validation." | **consistent** — this lane measures a NU S-parameter path against analytic oracles and proposes NO promotion. Promotion is the PI's call on the evidence. |
| `docs/guides/support_matrix.md`, "Multi-band graded mesh" | "The MESH ITSELF is documented — not any observable computed on it … **Not witnessed:** … in-plane grading." and "in-plane (`dx_profile` / `dy_profile`) grading is UNCOVERED, absorber-adjacent grading is EXCLUDED, and `dt` is unchanged (global min-cell CFL)." | **consistent** — arm B is a z-axis profile with ratio ≤ 1.4, ≤ 3 fine bands, and the z faces are PEC so no grading approaches an absorber. Arms D2 and E grade an in-plane axis on purpose and are labelled out-of-envelope throughout. |
| `docs/agent-memory/rfx-known-issues.md`, waveguide chain-status row | "Still outside this row: NU `flux` sub-lane AD-vs-FD only (no plane test, no ladder; **#810 dz-graded evidence absent**)" | **consistent** — this is the missing dz-graded forward evidence for that row. No AD leg, no plane-shift leg, no ladder is claimed here. |
| `docs/agent-memory/rfx-known-issues.md`, `~~#811~~ RESOLVED 2026-09-01 (PR #828)` | "dz-only now routes to the NU lane; dz-only+normalize=False fails loud; … dz-graded ACCURACY stays open under #810" | **consistent, and it constrains the design**: every graded arm is flux-lane-only. Verified by running it — see §1.3. |
| `docs/agent-memory/rfx-known-issues.md`, `#827` | "NU lane emits NO ring-down settling witness (found by the #811 independent witness; graded runs lose the truncation guard)" | **contradicted in part, and the contradiction is recorded**: `compute_waveguide_s_matrix(normalize="flux")` on a dz-graded mesh DOES return `settling_db` (bring-up: −97.8 / −98.3 dB on arm B's thru), because the witness reads the port's own records, not the run loop. #827's scope is the `run()` / `run_nonuniform` lane. This note takes no position on #827 beyond recording that reading. |
| `docs/design_notes/20260829_spec01_multiband_predeclaration.md` + `validation/research/multiband_nu/results/predeclared_windows.json` | F-S2, r = 1.4 abrupt: `R_model = 1.9982475271387787e-3` (−53.987 dB) at `cells_per_guided_wavelength_fine = 34.59346263584403` | the allowance is derived from **the JSON, read by the driver at run time**, not retyped from the note's prose (the prose has a recorded slip, C6a of that note). |
| `docs/design_notes/20260907_nu_exp1_band_law_sweep_predeclaration.md`, Table C | the single-ramp reflection scales as `(dz/λ)²` to within 4 % over 15–60 cells per wavelength at r ≤ 1.4 | this fixture's fine band sits at 32–57 cells per guided wavelength (§3), i.e. **inside** E1's measured validity domain for the scaling law. The law is used, not assumed. |
| `docs/agent-memory/nu_known_limits.md` §2 | "The caller is responsible for keeping `dx_profile[0] == dx_profile[-1] == dx`" | **consistent** — arms D2 and E pin both end cells to `dx`; the driver asserts it before any solve. |

**Premise verification on `6d721a56`** (checked, not assumed):

| premise | status |
|---|---|
| `tests/_waveguide_chain_battery_fixture.py` builds the WR-90 battery geometry from named constants | confirmed |
| `tests/fixtures/waveguide_chain_battery/fixture_v18_close.json` carries `mid`/`flux` cells and per-rung referee blocks | confirmed (18 cells; `referee.pec_short["mid|flux"]`, `referee.slab_airy["mid|flux"]`) |
| the same directory holds a newer schema-4 artifact | confirmed: `fixture_931_realized_pec_forward2_run369367259427.json` supersedes `fixture_v18_close.json` for the **device lane's realized-PEC operator**. §1.4 says which is used and why |
| `Simulation` accepts `dx_profile` / `dy_profile` / `dz_profile` | confirmed, `rfx/api/__init__.py:400-402` |
| `compute_waveguide_s_matrix` refuses `normalize=False` on a NU mesh | confirmed by running it, §1.3 |
| `res.s21_phase_residual_deg_rms` ships on `WaveguideSMatrixResult` | confirmed, `rfx/api/_spec.py:1319` |

**R2 status:** attempts on this mechanism hypothesis = **0**. This is the
first pre-declared attempt. The RF/EM intensifier (N ≥ 1) applies: a second
attempt needs a named new falsifier or an identified defect in this one, in
writing.

---

## 1. What is measured, and the two constraints that shaped it

### 1.1 One declared geometry, five meshes

The declared geometry is the committed WR-90 chain battery: a = 22.86 mm
(y), b = 10.16 mm (z), 121.92 mm of guide along x, CPML on x, PEC on y and z,
two TE10 waveguide ports at x = 12.70 / 109.22 mm, reference planes at
20.32 / 101.60 mm, 17 bins from 8.4 to 11.6 GHz, `num_periods = 40`, and
three DUTs: `thru` (empty), `pec_short` (σ = 1e10 block, 5.08 mm thick) and
`slab` (εr = 4, 10.16 mm thick). Nothing about the geometry, the drive, the
ports, the planes or the absorber changes between arms — only the mesh.

All arms run at the battery's **`mid` rung** (dx = 1.27 mm), on the
**`normalize="flux"`** lane, in **float32**, with the absorber pinned to the
committed mid-rung count (`cpml_layers = 34`).

| arm | mesh | DUTs | cells on the graded axis | max adjacent ratio | transitions | min cell |
|---|---|---|---|---|---|---|
| **A** | uniform 1.27 mm — the battery's own build | thru, pec_short, slab | 8 (z) | 1.0 | 0 | 1.27 mm |
| **B** | **z multi-band, in envelope**: 3 fine bands of 3 cells (0.635 mm) separated by coarse bands of 2 and 3 cells (0.889 mm) | thru, pec_short, slab | 14 | **1.4** exactly | 4 | 0.635 mm |
| **C** | z uniform at B's finest cell (0.635 mm) — cost / `dt` control | thru, pec_short, slab | 16 | 1.0 | 0 | 0.635 mm |
| **D1** | z multi-band, **out-of-envelope ratio**: fine bands of 2 cells (0.635 mm), coarse bands of 2 and 3 cells (1.27 mm) | thru, pec_short, slab | 11 | **2.0** | 4 | 0.635 mm |
| **D2** | x band of 10 cells at 2.54 mm over [50.8, 76.2] mm, **out-of-envelope ratio on the propagation axis** | thru | 86 | **2.0** | 2 | 1.27 mm |
| **E** | x band of 10 cells at 1.778 mm over [52.07, 69.85] mm, in-envelope ratio on an axis the support matrix does not cover | thru | 92 | 1.4 | 2 | 1.27 mm |

Explicit vectors, no `smooth_grading`, no `_make_dz_profile` — the #785
W1–W5 idiom, so every cell size is checkable by multiplication. The driver
asserts, before any solve, that each vector sums to its declared axis extent
to 1e-12 m, that the adjacent-ratio cap is what the table says, and that
`dx_profile[0] == dx_profile[-1] == dx` on the in-plane arms.

Arms B, C and D1 all have minimum cell 0.635 mm, so they share one `dt`
(1.7121530642382066e-12 s); arms A, D2 and E share the uniform `dt`
(2.421350084304325e-12 s). The `dt` split is the whole reason arm C exists.

### 1.2 Constraint 1 — this fixture's TE10 does not see the z cells

The battery's propagation axis is x; a = 22.86 mm lies along y and
b = 10.16 mm along z. The TE10 mode of that guide has one electric component
(Ez), varying as sin(πy/a) and **constant in z**, with Hx and Hy likewise
z-independent. In the Yee update equations the Ez update reads
∂Hy/∂x − ∂Hx/∂y, the Hx update reads ∂Ez/∂y, and the Hy update reads
−∂Ez/∂x: **no z spacing enters any of them.** So the discrete TE10 is an exact
mode of a z-graded mesh with the same x dispersion, and a z transition
presents it no discontinuity.

Three consequences, all declared here rather than discovered later:

1. Arm B's expected deviation from arm C is **not** transition reflection.
   It is float32 reassociation plus whatever the port extractor's z
   dual/primal weighting does. That makes arm B a sharp test of the NU
   extractor's z weighting, and a weak test of the transition law.
2. The allowance derived from the F-S2 reflection budget (§3) is therefore a
   **conservative upper bound** on this axis, not a tight prediction. It is
   kept as declared anyway, because it is the rule #810 asks for and because
   a zero-tolerance identity is not a usable window.
3. A ratio-2.0 profile on the same axis (arm D1) is expected to be **equally**
   invisible. Arm D1 is therefore not an allowance falsifier; it is the
   witness that says whether the observable can see the graded axis at all.
   The allowance falsifier has to move to the propagation axis (arm D2).

### 1.3 Constraint 2 — the graded arms have exactly one lane

`compute_waveguide_s_matrix(normalize=False)` on a non-uniform mesh raises,
verbatim (run on this tree, arm B's thru):

```
NotImplementedError: compute_waveguide_s_matrix() on a non-uniform mesh
(dx_profile / dy_profile / dz_profile) supports normalize=True or
normalize='flux' and single-mode ports. normalize=True or normalize='flux'
is required. Drop the dx/dy/dz profile to use the uniform lane.
```

`normalize=True` is fenced out of every reflection gate in this repo (the
known-issues entry "normalize=True S11 formula wrong for strong reflectors"),
and the battery's own lane set is `{False, "flux"}`. So **flux is the only
lane on which a graded and a uniform arm can be compared**, and the uniform
arm is read on it too.

The flux lane is a **two-run** extraction (`rfx/sources/waveguide_port.py`,
`extract_waveguide_s_matrix_flux`): a reference run in the empty guide and a
device run on the same mesh, with
`|S11| = sqrt(|F_ref(drive) − F_dev(drive)| / P_inc)` and
`|S21| = sqrt(|F_dev(recv)| / P_inc)`. Two structural facts follow, both
declared before the arms run:

* for the `thru` DUT the device run **is** the reference run, so `S11 ≡ 0`
  identically on the flux lane at every rung (confirmed in the committed
  artifact: `thru|*|flux` has `max|S11| = 0.0` on all three rungs, while
  `thru|mid|false` has 1.64e-2). A mesh reflection in the runway therefore
  cannot be read off a flux-lane thru's `|S11|`;
* in a lossless guide the net flux at the driven probe plane is already
  reduced by whatever turns around, so a runway reflection cancels in the
  `|S21|` ratio as well.

**The flux lane's magnitudes on a thru are blind to mesh-transition
reflection by construction.** What is not blind is the **phase**, which comes
from the modal amplitude ratio, not from the flux. This is why the phase
observables carry the discriminating weight below, and why §3 needs a second,
separately derived allowance term.

### 1.4 Which reference, and why

Every arm is judged against the **analytic references of the committed WR-90
chain battery**, recomputed per arm by the battery's own shared gate
arithmetic (`tests/_waveguide_chain_battery_gates.py`):

* `pec_short`: |S11| and |S22| against 1 (lossless short), gate [0.99, 1.03],
  mean tolerance 0.02 — `G.referee_pec_short`;
* `slab`: S11 and S21 against the analytic **Airy** slab moved to the default
  reference planes, gate 0.05 in magnitude and 15° in phase above the 0.30
  magnitude mask — `G.referee_slab_airy`;
* every DUT: column power (≤ 1.02), magnitude reciprocity (≤ 0.01) and
  complex reciprocity (≤ 0.01) — `G.cell_metrics`;
* `thru`: the shipped S21 phase residual against −βL
  (`res.s21_phase_residual_deg_rms`, #925), which is a port-discretization
  witness for an empty guide and is exactly the quantity a graded propagation
  path perturbs.

These are **analytic** oracles, so they are mesh-independent and all five
arms are judged against the same numbers. That is the property #810 asks for
("judged against the SAME committed reference"), and it is why this lane does
not use cv11's external legs: cv11 is a diagnostic reporter whose Meep legs
are #854-deferred, and its references are not per-arm recomputable.

`fixture_v18_close.json` (schema 3) is the artifact arm A is checked against,
not the newer `fixture_931_realized_pec_forward2_run369367259427.json`
(schema 4). Reason: schema 4 was measured with the realized-PEC device-lane
operator and a different AD stencil; this lane measures no AD leg and wants
the run whose forward cells were measured under the same operator the v1.8
chain closure quotes. Neither artifact is edited, and neither is cv11.

**Provenance check on the instrument, not a gate on the physics:** arm A is
compared entry by entry with `fixture_v18_close.json`'s `mid|flux` cells
under the battery's own live-comparison tolerance, 1e-4
(`tests/oracle/test_waveguide_chain_battery_v18_close.py`). Bring-up on this
tree already reproduced the committed slab referee to 1.7e-6 in magnitude
(0.03616547 here vs 0.03616716 committed) and 3e-5 degrees in phase.

### 1.5 Precision

**float32 on every arm**, which is `provenance.precision` of the committed
artifact and the forward default of the shipped lane. The v1.8 x64
declaration (`X64_DECLARED_LANES = {"flux"}`) covers contract criterion 1
(forward identity) and 3(a) (AD-vs-FD); this lane measures neither. No
`jax.config.update` is called anywhere — not at module level, not per run.

---

## 2. Realized meshes and the rasterization check (#325 class)

For every arm the driver records, before the solve:

* the realized cell vector of all three axes, **read back off the built grid**
  (`NonUniformGrid.dx_arr_f64` / `dy_arr_f64` / `dz_f64` with the CPML pad
  sliced off), not the vector that was passed in;
* the realized guide spans: Σ(y cells) against a = 22.86 mm, Σ(z cells)
  against b = 10.16 mm, Σ(x cells) against 121.92 mm;
* the DUT's realized cell count and its per-axis run, read off the **assembled
  material arrays** the solver uses (`_assemble_materials_nu` on the graded
  arms, `_assemble_materials` on arm A): εr > 2 cells for the slab, σ > 1
  cells for the PEC short.

**Declared expectation** (this is the #325 check, and #369 — "NU silently
dropped thin conductors" — is the failure it guards against): the guide spans
match their declared values to float64 accumulation (≤ 1e-15 m); the slab and
the PEC short span **every** y and z cell of their arm, and their x runs are
identical across all arms, because no arm changes the x mesh inside the DUT
window except D2 and E, which run `thru` only. A realized DUT cell count that
does not fill the cross-section, or an x run that moves, is a fired check and
that arm's S-parameters are not interpretable.

**Declared preflight expectations** (the output is part of the result, so
every finding and warning is stored verbatim per arm):

* arms D1, D2 and E must emit the adjacent-ratio warning
  (`dz_profile`/`dx_profile` "has max adjacent cell ratio …"); D2 and E also
  trip the in-plane threshold of 1.3 even at ratio 1.4. Arms A, B and C must
  not emit it.
* every arm inherits the battery's own `mid`-rung findings: the
  `T/τ_far = 2.125 < 3` record-length warning on both ports, the port index
  mirror audit note, the `pec_like` 4-cell volume notice on `pec_short`, and
  the `10.2 cells per λ_eff` dielectric notices on `slab`. These are
  properties of the committed fixture, not of the grading; a **new** finding
  on a graded arm is a result.

**Settling witness, per drive, per arm** (CLAUDE.md ring-down rule): every
`settling_db` entry must be ≤ −40 dB. Bring-up measured −94 … −98 dB on arm
A and −97.8 / −98.3 dB on arm B's thru. A drive above −40 dB invalidates that
arm's numbers and is reported as such rather than re-run to a better number.

---

## 3. The allowance, derived

### 3.1 Term 1 — the transition-reflection budget, scaled to this fixture

From the committed artifact
`validation/research/multiband_nu/results/predeclared_windows.json`, read by
the driver at run time:

| ratio | `fs2[r].abrupt.R_model` | dB |
|---|---|---|
| 1.4 | 1.9982475271387787e-3 | −53.987 |
| 2.0 | 6.298234160689015e-3 | −44.016 |

measured at `cells_per_guided_wavelength_fine = 34.59346263584403`. E1's
Table C measures that single-ramp reflection scaling as `(d/λ)²` to within
4 % over 15–60 cells per wavelength at r ≤ 1.4. This fixture's guided
wavelength, at its own numerical TE10 cutoff (`G.FC_TE10_HZ` = 6.5571 GHz),
runs 57.10 mm at 8.4 GHz to 31.33 mm at 11.6 GHz, so the 0.635 mm fine band
sits at 89.9 down to 49.3 cells per guided wavelength and the 1.27 mm x cells
at 45.0 down to 24.7 — inside E1's measured domain at the low end and above
it at the high end (extrapolating a law upward in resolution is the safe
direction: it makes the reflection smaller).

Per transition, per bin:

    R₁(r, d_fine, f) = R_model(r) · ( N_ref · d_fine / λ_g(f) )²

and the transitions add coherently in the worst case, so

    A_transition(f) = n_transitions · R₁(r, d_fine, f).

### 3.2 Term 2 — the instrument floor

`A_floor = 1e-4`, the live cell-comparison tolerance of
`tests/oracle/test_waveguide_chain_battery_v18_close.py` (CPU against the
committed GPU artifact). It is the only measured float32 reproducibility
number this battery owns. Bring-up on this tree lands 1.7e-6 inside it.

### 3.3 The amplitude allowance

    A(f) = A_transition(f) + A_floor

| arm | axis | r | n_tr | A(8.4 GHz) | A(10.0 GHz) | A(11.6 GHz) |
|---|---|---|---|---|---|---|
| B | z | 1.4 | 4 | 1.283e-3 | 2.546e-3 | 4.029e-3 |
| D1 | z | 2.0 | 4 | 3.828e-3 | 7.810e-3 | 1.2485e-2 |
| D2 | x | 2.0 | 2 | 7.557e-3 | 1.5521e-2 | 2.4870e-2 |
| E | x | 1.4 | 2 | 2.466e-3 | 4.993e-3 | 7.959e-3 |

(The full 17-bin vectors are written into the artifact by the driver; these
three columns are the band edges and centre.)

### 3.4 The phase allowance, and what the reflection budget does NOT bound

A spurious reflection of amplitude ρ riding on an entry of magnitude |S|
turns its phase by at most `arcsin(ρ/|S|)`. So the phase allowance for the
magnitude-masked gates is

    A_phase(f) = degrees( arcsin( A(f) / |S| ) ),

reported at |S| = 1 and at the slab phase gate's own 0.30 magnitude mask:

| arm | A_phase at |S| = 1 (8.4 / 10.0 / 11.6 GHz) | at \|S\| = 0.30 |
|---|---|---|
| B | 0.0735° / 0.1459° / 0.2309° | 0.245° / 0.486° / 0.770° |
| D1 | 0.219° / 0.448° / 0.715° | 0.731° / 1.492° / 2.385° |

**Stated limitation, declared before any arm runs:** the F-S2 budget bounds
*reflection*, and reflection is not the dominant phase effect of a graded
mesh on a propagation path. A coarse cell changes the phase a wave
accumulates **inside** it. That term is separate, and it is derivable from
the same Yee dispersion relation the port extractor already uses
(`_compute_beta`, `G.beta_yee`):

    Δφ = Σ_cells [ β_Yee(f; d_cell, dt) − β_Yee(f; dx, dt_ref) ] · d_cell

evaluated over the reference-plane separation L = 81.28 mm. Computed on this
fixture, with no FDTD:

| term | 8.4 GHz | 10.0 GHz | 11.6 GHz | rms over 17 bins |
|---|---|---|---|---|
| `dt` change only (arms B / C / D1 vs A; same x cells, smaller `dt`) | +0.4479° | +0.6268° | +0.8978° | **0.6586°** |
| arm D2's 25.4 mm band of 2.54 mm cells | +0.3932° | +1.1835° | +2.4423° | **1.4186°** |
| arm E's 17.78 mm band of 1.778 mm cells | +0.0877° | +0.2626° | +0.5387° | **0.3138°** |

The `dt` row is why the binding comparison is B against **C**, not B against
A: arms B and A do not share a `dt`, and 0.659° rms of that difference is
pure temporal discretization, larger than arm B's entire reflection-derived
phase allowance and nothing to do with grading.

---

## 4. Decision rules (frozen)

Deviation `dev_X(g, f)` means arm X's deviation from the **analytic**
reference of gate g at bin f — `||S11| − 1|` for the PEC short,
`|S − S_Airy|` in magnitude and the wrapped phase difference for the slab,
`|column power − 1|` and the reciprocity residuals for every DUT, and the S21
phase residual against −βL for the thru.

**R-1 (primary, the #810 rule at equal `dt`).** For every gate g and every
bin f:

    dev_B(g, f)  ≤  dev_C(g, f) + A_B(f)

with `A_B` the amplitude allowance of §3.3 on magnitude gates and `A_phase`
of §3.4 on phase gates. Arms B and C share `dt`, the minimum cell, the x and
y meshes, the absorber and the geometry; the multi-band profile is the only
difference. **Fires ⇒ the #785 multi-band envelope does not extend to this
observable, the support-matrix row stays mesh-only, and the firing is the
result.**

**R-2 (secondary, against the case's own uniform mesh).** For every gate and
bin:

    dev_B(g, f)  ≤  dev_A(g, f) + A_B(f) + A_dt(f)

with `A_dt` the `dt` row of §3.4 on phase gates and 0 on magnitude gates.
This is the comparison a user would make; it is reported with the `A_dt`
term named so a reader can see which part is grading and which is `dt`.

**W-BC (the sharp control window).** Arms B and C differ only in the z cell
pattern, and this fixture's TE10 does not read the z cells (§1.2). So:

    max over entries and bins of |S_B − S_C|  ≤  1e-4

(the battery's own live-comparison tolerance, §3.2). **Expectation: ≈ 1e-6,
float32 reassociation only.** Firing means the NU waveguide-port extractor's
z dual/primal weighting is not exact for a z-invariant mode — a defect
report, not an accuracy verdict.

**F-A (the falsifier of the allowance itself).** Arm D2 — ratio 2.0 on the
propagation axis — **must exceed** `A_B` (arm B's r ≤ 1.4 allowance) on at
least one gate and bin. Pre-declared expectation: it exceeds on the thru's
S21 phase, by about +1.42° rms against `A_B,phase ≤ 0.231°`, and it does
**not** exceed on any flux-lane magnitude, because of §1.3. If D2 stays
inside `A_B` on every gate, the allowance is not falsifiable on this fixture
and arm B's pass carries no weight — that outcome is reported as the result,
with no window moved.

**F-Z (the instrument-blindness witness).** Arm D1 — ratio 2.0 on the z axis,
four transitions — is **expected NOT to exceed** `A_B`, because of §1.2. If
D1 does exceed it, the z-invariance argument above is wrong, and arm B's pass
has to be re-read as a measurement rather than an identity. Either outcome is
recorded; neither moves a window.

**Arm E** carries no window. It is the reference point for "in-envelope
ratio, on the axis the mode actually traverses", on an axis the support
matrix does not cover. Its Yee prediction (0.3138° rms) is recorded beside
its measurement. **No promotion, no envelope extension and no support-matrix
edit attaches to arm E in this lane.**

**Cost** (documentation, no claim): total cells, peak RSS and wallclock per
arm, at equal `dt` where the arms share one.

---

## 5. Run plan, hygiene, and what has already been run

Order: commit this note and the driver → run arms A, B, C, D1, D2, E →
assemble the artifact into
`tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json` → write
the results note. One attempt per arm.

**Bring-up already executed on this tree, disclosed in full** (SPEC-00's
rule: bring-up is instrument validation, never falsifier data — no window in
this note is derived from any of it):

* a four-arm thru-only probe (`810_probe.py`, scratch, not committed) that
  established that the graded arms build and solve at all, that the flux lane
  is the only one available (§1.3), that `settling_db` is populated, and the
  per-cell wall time. It returned: arm A thru `s21_phase_residual` 0.03497°
  rms; arm B 0.03376°; arm C 0.03377°; a graded-x r=2.0 thru 1.458°.
* one run of the committed driver's `arms` stage restricted to arm A, which
  reproduced the committed `mid|flux` slab referee to 1.7e-6 (§1.4).

The windows of §3 and §4 come from the F-S2 JSON and from the Yee dispersion
relation, both computed without FDTD. The probe's 1.458° is not a window; the
Yee derivation independently predicts 1.4186° for that mesh, and the two
agreeing is reported in the results, not used to set anything.

Nothing in `docs/agent-memory/` is edited by this lane. No gate, tolerance or
golden anywhere in the repo is touched. No promotion of the NU lane out of
shadow is proposed here; that is the PI's call on this evidence.

**R3:** memory=`physics_validation_evidence_rule.md` line 129 + known-issues
chain-status row ("#810 dz-graded evidence absent") + `~~#811~~` row |
R2-attempts=0 | falsifier=F-A (arm D2 must exceed the r ≤ 1.4 allowance;
about 30 s of CPU).

---

## Results

Measured 2026-09-16, local CPU, 14 cells / 250 s of solve. The numbers, the
per-bin traces and the recomputed verdicts live in
[`graded_mesh_sparameter_accuracy_results.md`](graded_mesh_sparameter_accuracy_results.md)
and in `tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json`.
No window, profile, decision rule or expectation above was changed.

In one line each: **R-1 holds on all 13 gates × 17 bins** (worst margin
+1.28e-3 amplitude, +0.0735° phase); **R-2 fires at one bin** (slab |S11| vs
Airy at 8.4 GHz, −1.27e-4), traced to the z refinement because arm B equals
arm C to 1.05e-6; **W-BC holds** at 1.047e-6 against 1e-4, the declared ≈1e-6;
**F-Z holds** at 1.696e-6, so a ratio-2.0 z profile is as invisible to this
guide's TE10 as §1.2 predicted and the envelope is untested by this fixture;
**F-A fires as declared**, arm D2's thru S21 phase residual exceeding the
allowance at all 17 bins by up to +2.205°, with no flux-lane magnitude
exceeding. The three Yee-dispersion predictions of §3.4 came in at 0.28 %,
0.33 % and 0.09 % of the measurement.

Six corrections to this note's own reasoning are recorded in the results note
and none of them moves a window: **C1** the schema-3/schema-4 artifact choice
of §1.4; **C2** two reader defects in the driver; **C3** §3.1 calls
`G.FC_TE10_HZ` "numerical" when it is the analytic `c/2a` (worth ≤ 0.39 % on
every allowance number, no verdict moves); **C4** §3.2 calls 1e-4 "measured"
when the measured envelope is 5.000e-6 and 1e-4 is
`gate_from_envelope(…, quantum=10000)` (re-running R-2 at 5.000e-6 fires the
same one gate at the same one bin); **C5** §1 of the results note over-read
W-BC as exactness — the NU runner cannot run float64 fields, so 1.05e-6 is a
bound; **C6** `stage_verdicts` landed in the measurement commit `ed55486d`,
not in this one, and all 14 records carry `git_dirty = True` from the C2
repairs (the independent reviewer's bit-identical re-run of seven cells at the
committed sha is what closes that).

**R1 addendum (2026-09-16, after the run — this is not a pre-run citation).**
§0's memory table should have carried `docs/agent-memory/rfx-known-issues.md`,
entry dated 2026-07-16 (cv05 demotion, #325 / PR #378): *"the graded
fine↔coarse transition SPLITS the mode"*, and *"a correct build needs a
uniform-fine substrate band with no adjacent transition (redesign, not
re-pin)"*. It is **consistent with** this lane — that entry is a graded-mesh
transition damaging an observable on a substrate stack, i.e. the mechanism
this fixture's z-invariant TE10 cannot show — and it is the governing prior
art for the MSL follow-up in §8 of the results note. It is recorded here
rather than inserted into §0 because a pre-declaration's citation table is
frozen at its commit; adding a row to it now would back-date a citation that
was not made.
