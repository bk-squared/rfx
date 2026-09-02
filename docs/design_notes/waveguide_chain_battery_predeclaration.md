# WR-90 chain battery — pre-declaration (v1.8 WP2, before the first run)

Status: pre-declaration. Committed **before** any S-parameter of this fixture set is
measured, so that in git history every tolerance, position and drive setting provably
predates the first number. This note carries no measured S value; the first measurement
lands in a later PR as `tests/fixtures/waveguide_chain_battery/fixture.json` (schema:
`tests/fixtures/waveguide_chain_battery/README.md`) together with
`tests/oracle/test_waveguide_chain_battery.py`.

Governing documents: `docs/design_notes/v18_waveguide_s_chain_plan.md` (WP2, decisions 2, 4
and 6) and `docs/design_notes/chain_closure_contract.md` (criteria 1–3). Builder (constructs,
never runs): `tests/_waveguide_chain_battery_fixture.py`. Structural guard, fast lane:
`tests/unit/geometry/test_waveguide_chain_battery_geometry.py`. Every file:line below was read on main
`378a9c95` (2026-09-02).

What this note does not do: run anything, loosen any quoted tolerance, introduce a source
construction, import `rfx/probes/refplane.py` into the waveguide path, or use
`normalize=True`.

---

## 1. Memory (R1)

Line numbers are those of `docs/agent-memory/rfx-known-issues.md` in the primary checkout
on 2026-09-02; the plan's numbers (in parentheses) drifted by a few lines since its base.

- **`:3096-3112` (plan `:3093-3099`)** — "End-to-end AD through `compute_*_s_matrix` —
  RESOLVED 2026-05-25 … public-API `jax.grad` flows, FD↔AD rel_err 2.0e-4." *Consistent*:
  the battery adds objectives and fixtures on top of a working tape; it re-implements no
  assembly. Expected order for leg (a) below is taken from this number.
- **`:3397-3408` (plan `:3384-3395`)** — "The two-run diagonal formula
  `S11 = (b_dev − b_ref)/a_ref` does NOT cancel Yee dispersion for reflection … ±10–20 %
  |S11| error with `normalize=True`." *Consistent*: lanes are `normalize=False` and
  `normalize="flux"` only. Consequence recorded in §5(b): the complex-S21 invariance pinned at
  `tests/unit/sparams/test_waveguide_twoport_contract_v1.py:276` is a two-run cancellation of that lane and
  is **not** the expectation here.
- **`:4106-4125` (plan `:4093-4112`)** and **`:4120`** — "The offset is a time-convention
  conjugation … `S21_meep ≈ conj(S21_rfx)` … Do not de-embed by the fitted slope/intercept;
  conjugate the Meep data." *Consistent*: the only phase referee in this battery is the
  analytic Airy oracle in rfx's own `exp(+jωt)` convention (`airy_slab`,
  `scripts/diagnostics/build_waveguide_band_broad_e5_envelope.py:59-73`), moved to the
  reference planes with `exp(−2jβd)` / `exp(−jβ(d_L+d_R))`
  (`build_waveguide_band_broad_e5_phase_envelope.py:47-50`, `:134-137`). No external phase
  data enters; if one ever does, it is conjugated first.
- **`:4315`** — "Do-not-repeat: never anchor a drive-side wave at the port cell." *Consistent*:
  probe planes sit 25.40 mm (10 / 20 / 40 cells) inward of each port plane at every rung.
- **`:3718-3726` (plan `:3705-3716`)** — "The `aperture_dA` DROP-weight workaround in
  `init_waveguide_port` (zero out the boundary cell) trades a ∑aperture_dA mesh-convergence
  regression for clean PEC-short |S11| / reciprocity." *Consistent*: the dx ladder declares the
  staircase envelope of the current extractor; it does not reopen the weight. A ladder result
  is never a reason to change the weight.
- **`:3525` (plan `:3512`)** — "Production `compute_waveguide_s_matrix` on PEC-short geometry
  returns spread 0.0004 at R=1 — Meep-class." *Consistent*: the magnitude floor 0.005 in the
  ladder gate is 12× that spread.
- **`project_issue527_f32_comparator`** (auto-memory) — "The load-bearing change is not f64 —
  it is a comparator-validity assert placed BEFORE the accuracy gate (`_MIN_FD_ULP_SPAN`), so a
  comparator failure reports as one." *Consistent*: every FD leg asserts its ULP span before its
  accuracy gate, and a leg under the floor skips with the span printed; it never passes.

Nothing found in the ledger contradicts this fixture set (grep terms: `WR-90`, `chain`,
`ladder`, `Richardson`, `reference_plane`, `settling_db`, `#703`).

---

## 2. Fixture pre-declaration

### 2.1 Guide

WR-90: a = 22.86 mm along y, b = 10.16 mm along z. Port-normal axis x. Boundary: CPML on
x, PEC on both y faces and both z faces — the construction of
`tests/unit/autodiff/test_waveguide_flux_ad.py:33-44` (`BoundarySpec(x="cpml", y=pec/pec, z=pec/pec)`).
Preflight measures the guide on the domain faces (`guide_source = "domain_faces"`) and reports
22.86 × 10.16 mm at every rung (verified by the geometry test).

### 2.2 Ladder (decision 6)

| rung | dx | N = a/dx | b/dx | guide nodes y × z | grid incl. CPML (nx, ny, nz) |
|---|---|---|---|---|---|
| coarse | 2.540 mm | 9 | 4 | 10 × 5 | (83, 10, 5) |
| mid | 1.270 mm | 18 | 8 | 19 × 9 | (165, 19, 9) |
| fine | 0.635 mm | 36 | 16 | 37 × 17 | (329, 37, 17) |

Every rung is `dx = a/N` at integer N, so all three realize one guide (nodes = N + 1; the
wall-to-wall extent is N·dx = a exactly). This replaces any ladder whose rungs do not divide
a (the #703 class; the existing battery's {3, 2, 1.5} mm in a 40 mm guide realizes 13.33 /
20 / 26.67 cells, `tests/oracle/test_waveguide_port_validation_battery.py:42`, `:454`, and is a
warning, not a model).

### 2.3 x layout — absolute coordinates, all integer multiples of 2.54 mm

k is the coarse-cell index; metres = k × 0.00254. The integers are the source of truth in the
builder; the reader checks each metre value by multiplication.

| item | k | x (m) | note |
|---|---|---|---|
| domain length | 48 | 0.12192 | guide between the two CPML pads; pads lie outside |
| left port plane (source, +x launch) | 5 | 0.01270 | 5 / 10 / 20 cells from the CPML interface |
| left reference plane, default | 8 | 0.02032 | `ref_offset` = 3 / 6 / 12 cells inward |
| left probe plane | 15 | 0.03810 | `probe_offset` = 10 / 20 / 40 cells inward |
| θ window for the PEC-short AD legs | 19 → 23 | 0.04826 → 0.05842 | 10.16 mm, ends on the short's front face |
| slab, εr = 4 | 22 → 26 | 0.05588 → 0.06604 | 10.16 mm thick, full cross-section |
| PEC-short, `pec_like` | 23 → 25 | 0.05842 → 0.06350 | 5.08 mm thick, full cross-section |
| right probe plane | 33 | 0.08382 | 10 / 20 / 40 cells inward of the right port |
| right reference plane, default | 40 | 0.10160 | 3 / 6 / 12 cells inward |
| right port plane (source, −x launch) | 43 | 0.10922 | symmetric to the left port about x = 0.06096 |
| shifted reference plane, left (§5(b)) | 12 | 0.03048 | Δ_L = +10.16 mm (4 coarse cells inward) |
| shifted reference plane, right (§5(b)) | 35 | 0.08890 | Δ_R = −12.70 mm (5 coarse cells inward) |

Both DUTs are centred on x = 0.06096 m, so the two default reference planes are 38.10 mm from
the PEC-short faces and 35.56 mm from the slab faces on both sides.

The DUTs restate the construction of `tests/unit/sparams/test_waveguide_twoport_contract_v1.py:59-60`
(`pec_like`: `eps_r=1.0, sigma=1e10`, a Box spanning the full cross-section) and `:66-67`
(`diel`: `eps_r=4.0`) in this guide's own coordinates; the 40 × 20 mm file's 5 mm / 20 mm
thicknesses at x ∈ [0.050, 0.055] / [0.050, 0.070] are not inherited. Thicknesses are 2 and
4 coarse cells so the coarse rung stays on the Box volume branch (a 1-cell box takes the
thin-sheet branch, `rfx/geometry/csg.py` Box docstring).

The Box rule is half-open `[lo, hi)` on node coordinates compared in float64 (`csg.py`,
`Box.mask_on_coords`). With every face on a node, the rasterized runs are exactly
(thickness/dx, a/dx, b/dx):

| DUT | coarse (nx, ny, nz) → cells | mid | fine |
|---|---|---|---|
| PEC-short | (2, 9, 4) → 72 | (4, 18, 8) → 576 | (8, 36, 16) → 4608 |
| slab | (4, 9, 4) → 144 | (8, 18, 8) → 1152 | (16, 36, 16) → 9216 |

`fidelity_report` agrees with `Box.mask` on every count and raises only
`declared-lossy-realized-pec` on the `pec_like` row (that finding *is* the construction).
The geometry test pins all of this.

### 2.4 Absorber — derived by the rule, with this guide's own cutoff

Rule (`tests/unit/sparams/test_waveguide_twoport_contract_v1.py:35-48`):
`CPML_LAYERS = ceil(0.75 · λ_g(f_low) / dx)` with λ_g at the port's **numerical** TE10
cutoff. That cutoff is what preflight's `_check_waveguide_port_evanescent` computes
(`rfx/api/_preflight.py:2364`; `_emit_waveguide_port_cutoff_findings` at `:2204`:
`fc = (c/2)·sqrt((m/a)² + (n/b)²)` on the `guide` span of `_port_transverse_spans` (`:1816`), the
wall-to-wall extent measured on the assembled PEC mask). The 40 × 20 mm file's
`FC_TE10_NUMERICAL = 3.476e9` (`:39`) is that file's own guide, not this one's. Computed
with the same reader (`numerical_te10_cutoff_hz` in the builder):

- **fc_TE10, numerical = 6.557140 GHz** at all three rungs — identical to c/2a because dx
  divides a exactly (the discretization cannot move a wall that sits on a node). The twoport
  file's 0.27 GHz gap between numerical and analytic was a non-commensurate domain
  (40 mm / 1.8737 mm); here the gap is zero by construction.
- λ_g(8.4 GHz) = 57.102 mm; 0.75 λ_g = 42.83 mm.
- **CPML layers = 17 / 34 / 68** (0.75 λ_g/dx = 16.86 / 33.72 / 67.44), i.e. 43.18 mm at
  every rung — the constant-physical-thickness pattern of
  `tests/oracle/test_waveguide_port_validation_battery.py:449-457` falls out of the rule without a
  separate rounding step.

### 2.5 Band and drive

- `freqs = linspace(8.4, 11.6, 17) GHz`, 0.2 GHz spacing; **band-centre bin = index 8 =
  10.0 GHz** exactly. f_low/fc = 1.28; f_high = 0.885 × fc_TE20 (13.115 GHz), inside the 0.90
  margin of the `port_evanescent` advisory, so that advisory stays silent. fc_TE01 = 14.75 GHz.
- Source: `f0 = 10.0 GHz`, `bandwidth = 0.5`, `waveform = "modulated_gaussian"` (the
  `add_waveguide_port` default), `mode = (1, 0)` TE, `mode_profile = "discrete"` (default),
  `n_modes = 1`. No new source construction: the port's `cfg.e_inc_table` / `h_inc_table`
  as built by `add_waveguide_port`.
- `num_periods = 40` per drive → `n_steps = ceil(40 / (f_max · dt))` = **713 / 1425 / 2849**
  at dt = 4.8427e-12 / 2.4214e-12 / 1.2107e-12 s (`Grid.num_timesteps`, `rfx/grid.py:185-188`).
- **Settling witness (mandatory).** Each drive's `settling_db` (`WaveguideSMatrixResult`,
  uniform lane `rfx/api/_sparams.py` around `:2929-2939`; NU form at `:7758-7763`) is written
  to the fixture JSON per (dut, dx, lane, port). Pre-declared threshold **≤ −40 dB**
  (`CLAUDE.md` ring-down rule; contract criterion 2). A drive above −40 dB at
  `num_periods = 40` is re-run at `num_periods = 80` (record-length doubling at the same
  absorber, the form of `tests/crossval/test_waveguide_nu_broad_e5_envelope_gates.py:170-199`) and
  **both** numbers are written; the 40-period number of that cell is then not claims-bearing.
  `num_periods` is never tuned per cell silently.
- Lanes: `normalize=False` and `normalize="flux"`. Precision: the `Simulation` default
  (`float32` fields); FD legs of §5(a) run under a per-test x64 context (never a module-level
  flip, `CLAUDE.md`).

### 2.6 Preflight findings each rung will carry (input fidelity, quoted verbatim)

Preflight output is part of the result. Captured on the built fixtures (no run):

- coarse 2.54 mm — thru: none. PEC-short: none. Slab, 3 + 1 warnings:
  `mesh_resolution: dielectric 'diel' on x: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz,
  dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction
  amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17
  cells/λ_eff.` (same on y and z) and `lossless_q: all dielectric(s) ['diel'] are perfectly
  lossless in an open (CPML) domain … (Harmless if you are not measuring Q.)`
- mid 1.27 mm — thru: none. PEC-short: `mesh_resolution: PEC 'pec_like' x-extent 5.08mm = 4.0
  cells — volume under-resolved (PEC volume needs ≥5 cells; thin sheets <3 cells are fine).`
  Slab: the same three `mesh_resolution` warnings at `10.2 cells per λ_eff`, plus `lossless_q`.
- fine 0.635 mm — thru: none. PEC-short: none. Slab: `lossless_q` only (20.3 cells/λ_eff).

Consequence, pre-declared: the **fine rung** is the claims-bearing rung for the referee and
physics gates (§5(d), §6); the coarse and mid rungs exist for the ladder (§5(c)). The coarse
rung is expected to be pre-asymptotic inside the εr = 4 slab (5.1 cells/λ_eff) — that is
exactly the case the interpretability guard of §5(c) is written for. The `lossless_q` advisory
is informational here (no Q is measured).

---

## 3. What the geometry test pins now (guard iii of decision 6)

`tests/unit/geometry/test_waveguide_chain_battery_geometry.py`, fast lane, 15 checks, measured 4.7 s wall on
CPU for nine builds (budget 20 s): guide cells 9/18/36 × 4/8/16 and aperture = guide = 22.86 ×
10.16 mm with source `domain_faces`; numerical fc_TE10 = c/2a to 1e-12 relative; CPML 17/34/68
= 43.18 mm with zero transverse pads; the DUT run lengths and counts of §2.3 with
`fidelity_report` agreement; the source / reference / probe planes and the θ windows on the
declared coordinates; the θ override at θ = 0 equal to the fixture's own assembled material
array. Any later change to the fixture that moves a face off the node lattice goes red here
before any physics is measured.

---

## 4. Non-vacuity control

The thru (empty guide) cell is run and written but gates nothing: on an empty guide the
identities are two-run bookkeeping (`tests/unit/sparams/test_waveguide_twoport_contract_v1.py:131-136`,
#395). Its role is the control that the two reflecting DUTs differ from it: pre-declared
witness `max_f |S11| > 0.20` on both DUTs (the form at `:266`). By the Airy oracle the slab's
|S11| runs 0.19 (8.4 GHz) → 0.69 (11.2 GHz), 0.63 at the centre bin, with its nearest null at
8.07 GHz, below the band; the PEC-short is ≈ 1 everywhere.

---

## 5. Pre-declared falsifiers — every number is an existing gate

### (a) AD vs central FD — contract 3(a)

- Design variable θ enters through `eps_override` / `sigma_override` of
  `compute_waveguide_s_matrix` (both accepted on the uniform `False` and `"flux"` lanes;
  `rfx/api/_sparams.py:2452-2458`, the "G-AD-WIRE-WG2" override channel). The override array is the
  fixture's **own** assembled material array plus θ on the window (builder
  `design_override`; a `jnp.ones` base would delete the slab). Windows: the slab's own cells
  (eps_r = 4 + θ); for the PEC-short the vacuum window 0.04826–0.05842 m (eps_r = 1 + θ, or
  sigma = θ). θ0 = 0 for eps legs, so the AD leg is evaluated on exactly the fixture the gates
  see; loss leg θ0 = 0.05 S/m.
- Objectives at the band-centre bin (10.0 GHz): |S11|² (slab, eps θ; PEC-short, sigma θ),
  |S21|² (slab, eps θ), Re S21 and Im S21 (slab, eps θ), Re S11 and Im S11 (PEC-short, eps θ).
- FD: central, h = 0.05 on eps (`tests/unit/autodiff/test_waveguide_flux_ad.py:80`), h = 0.005 S/m on
  sigma, under a per-test x64 context; **ULP-span assert before the accuracy gate**,
  `_MIN_FD_ULP_SPAN = 1.0e4` (`tests/unit/autodiff/test_msl_ad_fd_converged.py:136`; gate `:556`;
  bidirectional falsifier `:629-634`), the ULP taken in the loss's own dtype. A leg under the
  floor **skips with the span printed; it never passes**.
- Gate: `rel = |g_AD − g_FD| / max(|g_FD|, 1e-12) ≤ 0.05`
  (`tests/unit/autodiff/test_sparam_ad_end_to_end.py:298`, `tests/unit/autodiff/test_waveguide_flux_ad.py:84`).
- Forward identity alongside (contract criterion 1): S under the θ = 0 traced override equals
  the untraced call to `rtol=1e-5, atol=1e-7` (`tests/unit/autodiff/test_waveguide_flux_ad.py:104`).
- **Expected order**: rel ≈ 1e-3 (ledger 2.0e-4 at `:3110`). **Expected skip, declared now**:
  PEC-short |S11|² under a lossless eps θ has a physically zero derivative (|S11| = 1 for any
  lossless window), so that leg is expected to fall under the ULP floor and skip; the
  magnitude leg that carries weight on the PEC-short is the sigma leg, and the eps legs on the
  PEC-short are the complex ones. A measured PEC-short |S11|² eps-gradient that *passes* the
  floor is a finding to explain, not a bonus.

### (b) Reference-plane invariance and rotation — contract 3(b)

Base = default planes (0.02032 / 0.10160 m). Shifted = (0.03048 / 0.08890 m): Δ_L = +10.16
mm, Δ_R = −12.70 mm, both inward, deliberately unequal so a sign error on one port cannot be
cancelled by the other — the reason the source pair at
`tests/unit/sparams/test_waveguide_twoport_contract_v1.py:257` is asymmetric. That file's 0.02 / 0.08 m
belong to its 0.12 m domain with ports at 0.01 / 0.09 m and are not inherited.

The shift is post-processing: `forward · exp(−jβ·s)`, `backward · exp(+jβ·s)` with
`s = shift_m · step_sign` (`_shift_modal_waves`, `rfx/sources/waveguide_port.py:1655`, factors at `:1681-1682`), β from the port
cross-section (`_compute_beta`, `:1419-1469`, Yee-discrete when dt and dx are passed).
Working the factors through `S = b/a` on these lanes gives, pre-declared:

| quantity | expected under the shift | at 8.4 / 10.0 / 11.6 GHz (continuous β) |
|---|---|---|
| \|S\|, whole matrix | invariant, `allclose(rtol=1e-3, atol=1e-4)` (`:270`) | — |
| ∠S11 | +2β·Δ_L | 128.1° / 184.2° / 233.5° |
| ∠S22 | +2β·\|Δ_R\| | 160.1° / 230.3° / 291.9° |
| ∠S21 = ∠S12 | +β·(Δ_L + \|Δ_R\|) = β · 22.86 mm | 144.1° / 207.3° / 262.7° |

Angle gates: against `_compute_beta(dt, dx)` within **3°** (`tests/unit/sparams/test_waveguide_phase_gate.py:259`);
against the continuous analytic β within **6°** (`PHASE_TOL_DEG = 6.0`, `:63`). Yee-vs-continuous
over Δ_L is 0.76° / 0.19° / 0.05° at the three rungs (computed from `_compute_beta`), so both
gates are reachable at every rung. Sign-discrimination witness: the wrong-sign prediction must
sit > 10° from the measurement (`:266`); all rotations above exceed 128°, so a flipped sign
cannot hide.

**Not the expectation here:** the complex-S21 invariance at `:276`. That is the two-run
cancellation of `normalize=True` (device and reference runs carry the same factor); on
`normalize=False` and `"flux"` the S21 phase rotates by β·(Δ_L + |Δ_R|) as tabulated. Writing
`:276` into these lanes would fail on a correct extractor.

**Gradient leg (decision 2): report-then-pin.** With θ from (a), across base vs shifted:
- magnitude objectives d|S11|²/dθ, d|S21|²/dθ: invariant; expected relative change ≈ 1e-6
  (rounding); pre-declared report bar **1e-2**;
- complex objectives: rotation-covariant, compare `e^{jφ} · dS/dθ|base` with `dS/dθ|shifted`,
  φ = 2βΔ_L for S11 and β·22.86 mm for S21; same 1e-2 report bar.
First run reports against 1e-2; the same PR pins `gate_from_envelope(measured, quantum=1000)`
(`tests/_gate_policy.py:89`, multiplier 1.5 at `:81`). A first measurement near 1e-2 rather
than 1e-6 is a finding (a β on the tape, or a non-unit-modulus factor), not a tolerance
question.

### (c) dx ladder — decision 6

Observables, every bin evaluated, worst bin reported: slab |S11|, |S21|, ∠S21; PEC-short
|S11|, ∠S11. Deltas between adjacent rungs: `coarse_delta = |S(2.54) − S(1.27)|`,
`fine_delta = |S(1.27) − S(0.635)|`, phases wrapped.

- **Gate**: `fine_delta ≤ coarse_delta + floor`, floor **0.005** for magnitudes
  (`tests/oracle/test_waveguide_port_validation_battery.py:474`; 12× the 0.0004 PEC-short spread at
  ledger `:3525`) and **1°** for phases. Stated limitation, from the contract: this is a
  non-increase test — a lane stuck at a wrong value passes it.
- **Witness (i), report-first**: monotonicity of the three-point sequence and the
  successive-delta ratio `fine_delta / coarse_delta`, written per observable and bin.
- **Witness (ii), report-first**: Richardson `2·S_fine − S_coarse` on the adjacent pairs
  (2.54, 1.27) and (1.27, 0.635) compared with the oracle — slab against Airy (|S11|, |S21|,
  ∠S21), PEC-short **phase** against π − 2βd (d = 38.10 mm from each default plane to the
  face; Yee-discrete β). PEC-short magnitude is excluded (trivially 1). Precedent: cv18's
  measured Richardson envelope 0.0051 → gate 0.01
  (`validation/crossval/18_wr90_iris_modematch.py:162`). Both witnesses are pinned by
  `gate_from_envelope` in a second step only.
- **Guard 1, one guide**: rungs are a/9, a/18, a/36 (§2.2).
- **Guard 2, interpretability**: the rungs are ratio 2, so the successive-delta ratio should
  approach 0.5 (first order) or 0.25 (second order). Pre-declared window: a ratio in
  **[0.15, 0.70]** is interpretable; outside it the ladder is reported **"not interpretable"**
  — neither passed nor failed — and the coarse rung is dropped or a finer one added before
  any claim. The only admissible fourth rung is **a/72 = 0.3175 mm** (b = 32 cells), keeping
  guard 1. Expected: the slab at the coarse rung (5.1 cells/λ_eff) may well land outside the
  window; the PEC-short (walls on nodes) is expected second-order-like.
- **Guard 3, rasterization**: each rung's DUT cell counts scale exactly with 1/dx — pinned
  now by the geometry test (§3), re-asserted by the battery at run time from the same
  builder.

### (d) Referee — contract 3(d)

- PEC-short, fine rung, every bin: **0.99 ≤ |S11|** (`battery:541`) and **|S11| < 1.03**
  (`:550`), mean within 0.02 of 1 (`:554`). f/fc ≥ 1.28 here; the 1.03 headroom was set for
  a near-cutoff residual at f/fc = 1.33, so a bin above 1.03 at 8.4 GHz is reported as a
  passivity finding with its preflight context, not absorbed.
- Slab vs analytic Airy (`airy_slab`, `build_waveguide_band_broad_e5_envelope.py:59-73`,
  TE-mode impedances `Z = η/√(1−(fc/f)²)`, `fc_d = fc/√εr`, engineering `exp(−jβ_d L)`), at
  the default planes with the oracle shifted by `exp(−2jβ_v d_L)` (S11) and
  `exp(−jβ_v(d_L + d_R))` (S21), d_L = d_R = 35.56 mm, continuous vacuum β
  (`build_waveguide_band_broad_e5_phase_envelope.py:47-50`, `:134-137`): magnitude
  **max |ΔS| ≤ 0.05** (`MAX_TOL`, `envelope.py:33`), phase **≤ 15°**
  (`MAX_PHASE_TOL_DEG`, `phase_envelope.py:99`), phase difference via `angle(S·conj(S_ref))`.
  Yee-vs-continuous β over 35.56 mm contributes 2.7° / 0.7° / 0.2° at the three rungs.
- Also in the 3(d) set (decision 4), zero run cost: the five broad-E5 replay bands
  (`tests/fixtures/waveguide_broad_e5/*_broad_e5_envelope.json`, gated by
  `tests/crossval/test_waveguide_broad_e5_envelope_gates.py`). cv18, cv19 and the Meep T-junction are
  magnitude-only support and carry neither criterion 1 nor 3(a).

---

## 6. Physics gates on the same fixture (fine rung; reported at all rungs)

- Column power `max_f Σ_i |S_ij|² < 1.02` on both reflecting DUTs
  (`tests/oracle/test_waveguide_port_validation_battery.py:307`).
- Magnitude reciprocity `mean_f |S21| − |S12| / max < 0.01` (`:340`).
- **Complex reciprocity** `max_f |S21 − S12| / max|S| ≤ 0.01`, pre-declared here for the
  first time; a first measurement above 0.01 is **reported**, not absorbed — the runtime
  reciprocity warning (#854 item 4) waits for this envelope.
- Power closure `|1 − Σ_i |S_ij|²|` on the lossless slab: measured and written; gated in WP3
  by `gate_from_envelope` together with the interior flux-monitor witness (the port column
  power and the S-matrix share one Poynting integral and are one witness).
- Settling: `settling_db ≤ −40 dB` per drive (§2.5).

---

## 7. Expected-order statements, collected

| leg | expected | source of the expectation |
|---|---|---|
| (a) rel(AD, FD) | ~1e-3 | ledger `:3110`, 2.0e-4 |
| (a) PEC-short \|S11\|² under eps θ | FD span under floor → skip | \|S11\| = 1 for any lossless window |
| (b) \|S\| under the shift | unchanged to rounding | unit-modulus factor |
| (b) ∠S11, ∠S22, ∠S21 | table in §5(b) | shift algebra at `waveguide_port.py:1681-1682` |
| (b) magnitude-gradient change | ~1e-6 against the 1e-2 bar | rounding only |
| (c) ratio fine/coarse delta | 0.25–0.5 where asymptotic; coarse slab rung may be outside | Yee second order; interface/staircase first order |
| (d) PEC-short \|S11\| | 0.99–1.03 with ≈ 0.0004 spread | `battery:541-554`, ledger `:3525` |
| (d) slab vs Airy | ≤ 0.05, ≤ 15°; E5 envelopes measured 0.0114 / 11.99° worst | `envelope.py:28-33`, `phase_envelope.py:95-99` |
| §6 column power, reciprocity | ≈ 1.0005, ≈ 0.0005 | `battery:304-307`, `:338-340` |

---

## 8. What the first run writes, and lane placement

The battery writes `tests/fixtures/waveguide_chain_battery/fixture.json` with every measured
value, `settling_db` per drive, the preflight findings per cell verbatim, the run id, and the
commit — schema in `tests/fixtures/waveguide_chain_battery/README.md`. Fast lane if the
measured wall time is ≤ 30 s, else slow with the shard named; the measurement goes in the PR
body (contract criterion 3, "fast lane when ≤ 30 s").

Cheap refute, to be run in that PR: flip the sign of the reference-plane shift in a local copy —
§5(b) must go red by > 10° (`phase_gate:264-266`); a flipped sign that stays inside 3° means the
gate does not bind.

---

## 9. Must not

- Loosen any tolerance quoted above; a red gate needs a written root cause first.
- Use `normalize=True` anywhere in the battery.
- Import `rfx/probes/refplane.py` into the waveguide path (numpy round-trip at `:539-542`).
- Introduce a new source construction; reuse the port's `cfg.e_inc_table` / `h_inc_table`.
- Add a ladder rung that is not `a/N` at integer N; move a DUT face off the 2.54 mm lattice.
- Copy the `:276` complex-S21 invariance into the `False` / `"flux"` lanes (§5(b)).
- Tune `num_periods` per cell without writing both numbers.
- Touch the #812 lane (`wr90_rectangular_broad_e4_comparison.json`, cv02/03/04/09/10/14/20/21,
  cv06b).

---

R3: memory=rfx-known-issues.md:3096-3112,:3397-3408,:4106-4125,:4315,:3718-3726,:3525 + project_issue527_f32_comparator | R2-attempts=0 (no measurement in this lane) | falsifier=tests/unit/geometry/test_waveguide_chain_battery_geometry.py — 15 checks pass in 4.7 s CPU; a face moved off the 2.54 mm lattice or a rung not dividing a goes red before any physics runs

---

## Erratum (added with the first measurement, WP2 measurement PR)

Two premises of this note were contradicted by the extractor on the first coarse-rung
plumbing run (CPU, not claims-bearing). Neither changes a tolerance, a position, a rung or a
drive setting; both change what the numbers above are *referenced to*, and the fixture set
was corrected in the builder before the one claims-bearing run so that the note's geometry
holds as written.

1. **§2.3, "left reference plane, default = 0.02032".** rfx's default reported reference plane
   is the port (source) plane, not `source + ref_offset·dx`
   (`rfx/api/_sparams.py`, `desired_ref = entry.reference_plane if ... else planes["source"]`,
   RF-audit 2026-07-23): under `reference_planes=(None, None)` the result reported
   `reference_planes = [0.01270, 0.10922]`. The builder now passes the declared planes
   (0.02032 / 0.10160 m) explicitly, which makes them the raw record planes (`ref_shift = 0`,
   no de-embed β on the base S). Every distance and shift of §2.3, §5(b), §5(c) and §5(d) is
   therefore realized exactly as written; the geometry test pins the explicit planes.
2. **§2.4 / §5(b), "the port's numerical TE10 cutoff … 6.557140 GHz" and "against
   `_compute_beta(dt, dx)`".** Two different "numerical cutoffs" exist. Preflight's
   wall-to-wall reader gives 6.557 GHz (= c/2a, the number this note computed). The PORT
   CONFIG's `f_cutoff` — the cutoff `_shift_modal_waves` and `_compute_mode_impedance`
   actually use — is the discrete 2D eigenvalue on the port aperture
   (`mode_profile="discrete"`), which measured 5.877 / 6.205 / 6.378 GHz at the three rungs:
   the discrete cutoff of an aperture ONE CELL WIDER than the guide (effective width
   10.04 / 19.02 / 37.01 cells). The FDTD guide itself propagates with the 9/18/36-cell
   discrete cutoff (thru S21 phase fit, coarse rung: fc = 6.525 GHz, rms 0.08°). The
   rotation gate of §5(b) is therefore evaluated, as declared, against β of the guide's
   cutoff (c/2a; Yee-discrete and continuous), and the fixture additionally records the
   residual against the port config's own β as the mechanism witness. Where that gate is red
   the reason is this cutoff, not the shift algebra (which the port-β residual shows exact).
3. **§5(a), "sigma = θ" on the PEC-short window.** `sigma_override` replaces the sigma array
   AFTER `compute_waveguide_s_matrix` folds `pec_mask` into sigma = 1e10, so a base taken from
   `_assemble_materials` (sigma = 0 inside `pec_like`) silently deleted the short
   (`sigma_override(θ=0)` reproduced the empty guide's |S11| = 0.076). The builder's
   `design_override(kind="sigma")` now carries the fold; the leg then measures what §5(a)
   declares (a lossy window in front of the short).

The measured consequences (which gates are red, with the numbers) are in the fixture and in
the measurement PR; nothing here is absorbed into a tolerance.
