# Lattice-witness standardisation — pre-declaration (slab family: cv04, cv22, cv23)

Date: 2026-09-03 · Lane: `agent/lattice-witness-standard` · Cases:
`validation/crossval/04_multilayer_fresnel.py`,
`validation/crossval/22_dispersive_slab_fresnel.py`,
`validation/crossval/23_lossy_slab_fresnel.py`.

**Append-only.** Corrections are added as new sections; nothing above a
correction is edited. Every number here is ANALYTIC (closed-form ε(f), the
transfer matrix, the exact time-harmonic solution of the Yee lattice, the
z-transform of the updates) or READ from a committed artifact by
`path.json::key = value`. No FDTD was run for this note: every measured number
it quotes was already committed by cv22 round 4 or cv23 round 3, and the new
artifacts (`lattice_witness.json`) are post-processing of those, the same class
as `--meep-ladder-summary` and `--refit-tail-fits`.

**PI decision this note implements (2026-09-03, verbatim):** "격자별로 테스트
하고 다른 유사한 이슈들도 있으니 같은 엄밀성 기준을 쓰자" — test per lattice
(each dx rung), and apply the same rigor standard to the other cases with the
same kind of issue.

## 0. Why this exists

cv23 round 1 (`docs/design_notes/20260902_cv23_lossy_slab_predeclaration.md`
§12.2) found that rfx's residual against the CONTINUUM transfer matrix is not
error: it is the Yee lattice's own second-order term, reproduced to 3e-5 by an
exact 1-D Yee-lattice time-harmonic solution with no fitted parameter. cv23 §14.6
put two options to the PI — carry `W_lat` inside the continuum window, or refine
the one under-resolved arm — and took the second, keeping the lattice solution as
a REPORTED witness (§13.3).

The PI has now decided a third thing, which is neither: **the continuum window
stays exactly as it is, and a SECOND, independent gate is added at every dx rung
every slab-family case runs.** The continuum gate says "rfx reproduces the
transfer matrix inside a window derived from a committed envelope". The lattice
gate says "and the part of that residual which is not zero IS the lattice's own
term, at this mesh, to within an error budget derived from the lattice model
itself". A case passes when both hold. That is what makes a dx-ladder claim
rigorous rung by rung instead of only at the end of the ladder.

**No window is widened anywhere in this lane.** `W_bin = 0.074`,
`W_mean,R = 0.010`, `W_mean,T = 0.017`, `W_bin,A = 0.148`, `W_mean,A = 0.027`,
the −40 dB settling bar `SETTLING_LIMIT = 1e-2`, cv04's `TAIL_PURITY_LIMIT =
1e-3` and `CONS_MAX_LIMIT = 0.06` are untouched, and
`validation/crossval/comparators/slab_rig.py` REFUSES a settling bar looser than
the declared one (`test_settling_bar_can_only_be_tightened`).

## 1. Scope — which cases are slab family, and why

| case | in scope | why |
|---|---|---|
| `04_multilayer_fresnel` | **yes** | the rig every other slab case is a copy of; one slab, ε′ = 4, d = 10 mm, dx = 1 mm, normal incidence, 2-D TMz with periodic y. Its committed band-mean `|ΔR|` is the envelope cv22 and cv23 derive their windows from, so if that number is the lattice term the whole family's window derivation rests on a discretization artefact — which is exactly what §5.3 shows. |
| `22_dispersive_slab_fresnel` | **yes** | the SAME rig with a Debye / Lorentz / Drude pole in the slab (`validation/crossval/comparators/slab_rig.py`, "cv22's `run_rfx_arm` with the slab MATERIAL factored out"). §3 derives the exact lattice for the ADE. |
| `23_lossy_slab_fresnel` | **yes** | the same rig with `materials.sigma`; the case that found the term. |
| everything else in `validation/crossval/manifest.json` | no | not a normal-incidence homogeneous slab on this rig. |
| an oblique-incidence slab case | **out of scope, named** | there is no `26_*` (or any oblique slab) case in the tree at `71c919a` — `git grep cv26` on this branch returns nothing, and `docs/design_notes/20260903_e4_all_solver_classes_plan.md` (branch `origin/agent/e4-all-solvers-plan`, commit `eb78254`) does not list one either. Independently of that, oblique incidence is NOT a 1-D lattice problem: the exact witness there is a 2-D Yee lattice with a Floquet phase per row, a different derivation with its own interface bookkeeping and its own numerical-dispersion anisotropy. It needs its own note; this standard does not silently cover it. |

`docs/design_notes/20260903_e4_all_solver_classes_plan.md` lane **L1** ("Slab
TEM-box rig: cv04 + cv23 × {openEMS, Palace}") reuses this rig. When it lands,
each new reference solver gets its own lattice — openEMS is a Yee FDTD and this
same 1-D solution applies at its dx and dt; Palace is FEM and does not have one,
so its leg is a continuum-only reference. Stated here so L1 does not have to
rediscover it.

## 2. The lattice model — one function, four materials

`validation/crossval/comparators/dispersive_eps.py::yee_lattice_slab_rt_eps` is
the exact time-harmonic solution of the 1-D Yee lattice whose slab nodes realize
a given DISCRETE-TIME permittivity `ε_num(ω)`. With `z = e^{jωdt}`,
`ω̂ = 2 sin(ωdt/2)/dt`:

    H_{i+1/2} − H_{i−1/2} = dx · (j ω̂ ε0 ε_num,i) · E_i
    E_{i+1}   − E_i       = dx · (j ω̂ μ0) · H_{i+1/2}

marched from a unit transmitted lattice plane wave (vacuum lattice wavenumber
`k = (2/dx) asin(ω̂ dx/2c)`) back to the incidence side, where two nodes are
decomposed into incident + reflected. The vacuum nodes are exact solutions of the
same recurrence, so the padding `n_vac` does not enter the answer — it only fixes
where the decomposition is read.

**Why one `ε_num` covers everything this rig runs.** Whatever the E-update is —
ordinary (`rfx/core/yee.py::update_e`), conductive (same, `σ ≠ 0`), Debye
(`rfx/materials/debye.py:229`), Lorentz/Drude (`rfx/materials/lorentz.py:262`) —
it is algebraically

    ε0 ε∞ (E^{n+1} − E^n)/dt + Σ_p (P^{n+1} − P^n)/dt + σ (E^{n+1} + E^n)/2
        = curl H^{n+1/2}

Z-transforming factors out the ordinary Yee temporal factor
`(z−1)/dt · z^{−1/2} = 2j sin(ωdt/2)/dt = jω̂`, common to vacuum, and what is
left multiplying it is exactly `ε0 · ε_num(ω)` — the SAME `ε_num` that
`dispersive_eps.eps_numerical_ade` returns and that cv22 §3 (`W_ADE`) and cv23 §3
(`W_σ`) already carry as window terms. So the node admittance is
`y_i = j ω̂ ε0 ε_num,i` and the whole material zoo enters through one complex
number per bin. For a `P^{n+1} = a P^n + b P^{n−1} + c E^n` recurrence,
`(P^{n+1} − P^n)/(E^{n+1} − E^n) = (z−1)G(z)E / (z−1)E = G(z) = χ_num`, which is
why the Lorentz/Drude half-step centring does not add a term.

Check that the derivation is not new physics: for the conductive case
`ε_num = ε′ − jσ(x/tan x)/(ωε0)`, `x = ωdt/2`, and
`jω̂ ε0 ε_num = jω̂ ε0 ε′ + ω̂ σ (x/tan x)/ω = jω̂ ε0 ε′ + σ cos x`
because `ω̂/ω = sin x / x` — the literal expression cv23 wrote by hand in
`yee_lattice_slab_rt`.

**Numerically verified** (`tests/crossval/test_lattice_witness_gates.py::test_extended_lattice_reproduces_the_cv23_solver_it_generalizes`):
`yee_lattice_slab_rt_model("conductive", …)` equals `yee_lattice_slab_rt`
**bit-for-bit at σ = 0** and to **≤ 1e-15 in R and T** on all three cv23 arms —
cv23's committed lattice numbers are unchanged (the old function is kept as the
literal expression and now shares the marcher, and reproduces its pre-refactor
output exactly).

Per case:

| case | model | parameters | ε_num |
|---|---|---|---|
| cv04 | `conductive` | ε′ = 4, σ = 0 | ε′ (no discrete-time term at all: the update has no material state) |
| cv22 Debye | `debye` | ε∞ = 2, Δε = 4, τ = 1/(2π·5 GHz) | `ε∞ + Δε/(1 + j ω̃ τ)`, `ω̃ = (2/dt) tan(ωdt/2)` (Crank–Nicolson, bilinear) |
| cv22 Lorentz | `lorentz` | ε∞ = 2, Δε = 1.5, f0 = 7 GHz, δ = ω0/6 | `ε∞ + κ/(ω0² − ω̃² + 2jδω̂')`, `ω̃ = (2/dt) sin(ωdt/2)`, `ω̂' = sin(ωdt)/dt` |
| cv22 Drude | `drude` | ε∞ = 3, fp = 7 GHz, γ = 2π·3 GHz | as Lorentz with ω0 = 0, κ = ωp², δ = γ/2 |
| cv23 | `conductive` | ε′ = 4, σ ∈ {0.15577, 1.5577, 4.6731} S/m | `ε′ − jσ(x/tan x)/(ωε0)` |

**No arm is excluded.** The PI's condition ("if a pole model cannot be done
exactly, exclude that arm and say why") does not bite: all three cv22 pole models
are exact through the same identity, and §5.2 shows the lattice model predicts
each arm's measured continuum residual a priori.

The lattice converges to the transfer matrix at second order for every model
(`test_lattice_converges_to_the_transfer_matrix_at_second_order_for_every_model`,
measured order in [1.7, 2.3] on each doubling for all four).

## 3. W_witness — the derivation

The lattice solution is the exact steady state of an INFINITE record on an
INFINITE lattice. Five things separate it from what the rig measures. Three are
zero by construction and are ASSERTED, not modelled; two are non-zero and are
bounded by witnesses the cases already gate for their own reasons.

**(Z1) CPML round trip = 0.** The rig sizes the box so the CPML round trip
exceeds the record (`slab_rig.py`, `t_safe`; `run.record.t_safe_cpml_steps ≥
run.n_steps`). Any CPML echo therefore arrives after the record ends and is
already inside (T1). Asserted as `gates.precond_cpml_gate`.

**(Z2) 2-D rig vs 1-D model = 0.** At normal incidence with periodic y and a
y-uniform TFSF plane wave, ∂/∂y = 0 and the 2-D TMz update IS the 1-D Ez/Hy
lattice. The TFSF auxiliary grid (`rfx/sources/tfsf.py`) runs the same update at
the same dx and dt, so the injected field is an exact lattice plane wave and R, T
are lattice quantities by construction, not approximately.

**(Z3) Probe standoff = 0.** The 30-cell standoff is lossless: the vacuum lattice
wavenumber `k = (2/dx) asin(ω̂ dx/2c)` is REAL over the whole band
(`ω̂ dx/2c = 0.105` at 10 GHz, dx = 1 mm), so propagation contributes only a
phase and drops out of `|·|²`.

**(T1) Record truncation.** The rig records N steps; what is still ringing after
step N is missing from both rFFTs. With `R = |S|²/|I|²` (numpy `rfft`, no dt
factor), a spectral error `e_S` on the scattered transform and `e_I` on the
incident reference give, to first order,

    |ΔR| ≤ 2 √R · (e_S/|I|) + 2 R · (e_I/|I|)

and the missing tail is bounded coherently,

    e_S ≤ Σ_{n≥N} |s_n| ≤ A_tail · inc_peak / (1 − e^{−Γ dt})

where `A_tail` is the case's OWN settling witness
(`tail.scat_refl_rel` / `tail.total_trans_rel`, both defined relative to
`inc_peak` in `slab_rig.py::_witness`) and Γ is the slowest amplitude decay rate.

**(T2) Incident-reference truncation** — "injection leakage" as this rig
witnesses it. The 1-D auxiliary reference that forms the denominator is truncated
at the same step; its level is cv04's tail-purity witness `tail.purity_inc_rel`
(bar 1e-3) and its envelope rate is the differentiated Gaussian's own,
`Γ_inc = 2a/τ` with `2a e^{−a²} = purity`, `a > 1/√2`.

**(T3) float32.** The fields are float32 (`rfx/core/yee.py::init_state`,
`field_dtype=jnp.float32`), `ε32 = 2^−24 = 5.96e-8`. Carried with the
STATISTICAL (√-accumulation) size `N ε32/√2`; the fully coherent worst case
`N² ε32/2` is computed and recorded as
`budget.mean_delta_round_coherent_gated` but is NOT used, and §5 shows it is not
reached (it is 2.4e-3–1.2e-2 in δ, i.e. 3 orders above the statistical term and
2 orders above the measured residual — if it were reached, every rung would fail
by ~100×, and none does).

**The denominator.** The rig's source is `s = −2u e^{−u²}`, `u = (t−t0)/τ`,
`τ = 1/(π f0 bw)` (`rfx/sources/tfsf.py:446-449`). Its continuous transform is
`|S(ω)| = ω τ² √π e^{−ω²τ²/4}` and its time-domain peak is `√2 e^{−1/2}`, so

    |I(f)| = inc_peak · Λ · a(f),      **Λ = √π τ / dt**

with `a(f)` the relative incident amplitude the artifacts already store
(`inc_amp_rel`). `τ = 6.36620e-11 s`, `√π τ = 1.12838e-10 s`, so **Λ = 48.323**
at dx (dt = 2.335067793382187e-12 s), 96.65 at dx/2, 193.29 at dx/4. Checked
against a numerical rFFT of the sampled waveform to 2e-3
(`test_source_spectral_gain_matches_a_numerical_transform_of_the_rig_waveform`);
Λ enters as a divisor, so a 0.2 % error in it is a 0.2 % error in the window.
`inc_peak` cancels: every level in the budget is expressed relative to the same
peak the witnesses use.

**The window, per bin:**

    δ_scat(f)  = A_scat  · κ   / (Λ a(f)),   κ   = 1/(1 − e^{−Γ dt})
    δ_trans(f) = A_trans · κ   / (Λ a(f))
    δ_inc(f)   = A_purity· κ_i / (Λ a(f)),   κ_i = 1/(1 − e^{−Γ_inc dt})
    δ_round(f) = N ε32 / (√2 Λ a(f))

    W_witness,R(f) = 2 √R_lat (δ_scat  + δ_round) + 2 R_lat δ_inc
    W_witness,T(f) = 2 √T_lat (δ_trans + δ_round) + 2 T_lat δ_inc
    W_witness,A(f) = W_witness,R + W_witness,T          (triangle, as cv23 §4 does for A)

**Γ, and the one place this is an assumption.** Γ is taken as
`min(run.record.rate_ring_1_s, tail.fitted_rate_scat_refl_1_s,
tail.fitted_rate_total_trans_1_s)` over the finite positive ones — the more
conservative of the case's a-priori derivation and its own measured fit, because
cv22 §14.1 measured a Debye tail decaying **1.5× slower** than the derivation.
The bound needs SOME decay assumption (a non-decaying continuation has unbounded
`Σ|s_n|`); the physics that supplies it is that the slab is passive and every
mode decays at least at the slowest rate over the incident ring band, and the
fitted rate is the witness on that. This is the single non-rigorous input in the
budget and is named here rather than buried.

**What makes it not tuned.** Two statements, both checkable:

1. every input is either a pre-declared constant (`SETTLING_LIMIT`,
   `TAIL_PURITY_LIMIT`, τ, ε32, Λ) or a witness the case ALREADY gates for its
   own reasons (`tail.*`, `run.record.rate_ring_1_s`, the CPML gate). **None of
   them is the residual being gated**, and no measured `|rfx − lattice|` enters
   the window.
2. `lattice_witness.ceiling_windows` computes the same window with the DECLARED
   BARS (`A_scat = A_trans = 1e-2`, `A_purity = 1e-3`) and the DERIVED rate only.
   That is available before any run and is recorded per rung as
   `mean_W_ceiling_{R,T,A}_gated`. A run whose measured tail decays slower than
   its derivation can carry a window above its ceiling; that is flagged
   (`W_exceeds_ceiling_R`) rather than absorbed — it happens on exactly one rung,
   cv22's Debye (§5.2).

The budget is monotone in every witness it reads
(`test_the_window_is_monotone_in_every_witness_it_reads`): a looser tail, a
looser purity, a slower ring-down or a longer record can only WIDEN
`W_witness`. A run cannot buy a tighter window by being worse.

## 4. The gates

Per arm, per rung, on the SAME gated bins the continuum gate uses
(4.0–10.0 GHz):

- **GL1 (per bin)**: `|R_rfx − R_lattice| ≤ W_witness,R(f)` at every gated bin; T
  and A likewise.
- **GL2 (band mean)**: `mean|ΔR| ≤ mean W_witness,R` over the gated bins; T and A
  likewise.
- **Preconditions** (the terms declared zero): `precond_cpml_gate`
  (`t_safe_cpml_steps ≥ n_steps`) and `precond_tail_witness` (`tail.ok`, the
  case's own settling + purity witnesses).

A case passes when its continuum gates AND its lattice gates hold at every rung.
The lattice gate is evaluated by
`validation/crossval/comparators/lattice_witness.py::evaluate` and written to
`lattice_witness.json` beside the case's other artifacts by
`<case>.py --lattice-witness`; it returns the same gate-record shape as the other
cv gates (per-bin arrays, gated scalars, a boolean `gates` dict, `witness_ok`).

## 5. The rungs, and what they measure

Reuse only: no rung was added to get these numbers. cv23's ladder is the one
round 2 ran and round 3 committed; cv22 runs one rung per arm; cv04 runs one.

### 5.1 cv23 — nine committed entries, eight distinct meshes, all green

`validation/crossval/_23_lossy_results/lattice_witness.json::verdict.all_rungs_ok = true`,
`validation/crossval/_23_lossy_results/lattice_witness.json::verdict.n_rungs = 9`.
(`tand3` and `tand3_dx2` are the same mesh: tand3's declared primary recipe IS
dx/2, so its ladder is {dx/2, dx/4} and it has no committed dx rung — cv23
§13.3. Hence eight distinct meshes.)

| rung | dx | N | tails scat / trans / purity | Γ (source) | W_witness R / T / A | \|rfx − lattice\| R / T / A | worst per-bin ratio R / T / A |
|---|---|---|---|---|---|---|---|
| `tand0p1` | 1.00 mm | 1067 | 8.9e-4 / 1.0e-3 / 1.8e-4 | 1.54e10 (derived) | 1.28e-3 / 3.33e-3 / 4.61e-3 | 3.0e-5 / 3.1e-5 / 3.8e-5 | 0.06 / 0.02 / 0.02 |
| `tand0p1_dx2` | 0.50 | 2134 | 7.9e-4 / 4.7e-4 / 1.8e-4 | 1.54e10 (derived) | 1.13e-3 / 1.57e-3 / 2.70e-3 | 2.6e-5 / 5.3e-5 / 6.3e-5 | 0.06 / 0.07 / 0.05 |
| `tand0p1_dx4` | 0.25 | 4267 | 7.9e-4 / 2.5e-4 / 1.8e-4 | 1.54e10 (derived) | 1.08e-3 / 8.42e-4 / 1.93e-3 | 2.5e-5 / 7.1e-5 / 7.9e-5 | 0.06 / 0.16 / 0.10 |
| `tand1` | 1.00 | 1158 | 9.8e-5 / 2.8e-5 / 1.3e-4 | 3.18e9 (fitted) | 8.56e-4 / 1.24e-4 / 9.81e-4 | 2.1e-5 / 5.3e-6 / 1.9e-5 | 0.06 / 0.08 / 0.05 |
| `tand1_dx2` | 0.50 | 2315 | 9.2e-5 / 2.9e-5 / 1.2e-4 | 3.41e9 (fitted) | 7.50e-4 / 1.23e-4 / 8.73e-4 | 2.0e-5 / 5.5e-6 / 1.8e-5 | 0.07 / 0.09 / 0.05 |
| `tand1_dx4` | 0.25 | 4629 | 9.0e-5 / 2.9e-5 / 1.2e-4 | 4.21e9 (fitted) | 5.75e-4 / 9.76e-5 / 6.73e-4 | 2.0e-5 / 5.4e-6 / 1.7e-5 | 0.10 / 0.12 / 0.07 |
| `tand3` = `tand3_dx2` | 0.50 | 2362 | 8.6e-5 / 6.4e-6 / 1.1e-4 | 7.52e9 (derived) | 4.71e-4 / **1.94e-6** / 4.73e-4 | 3.0e-5 / 2.2e-7 / 3.0e-5 | 0.18 / 0.29 / 0.18 |
| `tand3_dx4` | 0.25 | 4723 | 8.2e-5 / 6.7e-6 / 1.1e-4 | 7.52e9 (derived) | 4.34e-4 / 1.97e-6 / 4.36e-4 | 2.9e-5 / 2.2e-7 / 2.9e-5 | 0.19 / **0.30** / 0.19 |

Every value in that table is a key of
`validation/crossval/_23_lossy_results/lattice_witness.json` (`rungs.<rung>.…`);
e.g.
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand3.mean_W_witness_R_gated = 0.00047`,
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand3.mean_dR_lattice_gated = 3e-05`,
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand3_dx4.worst_ratio_T = 0.3`
(the worst per-bin ratio over ALL nine rungs and all three observables).

Two readings worth stating:

- **The lattice gate un-vacuums tand3's transmission.** cv23 §2 declared the
  tand3 T gate vacuous against the continuum window (T_TMM ≤ 0.003 everywhere,
  `W_mean,T = 0.017`). Against the lattice the same observable carries
  `mean_W_witness_T_gated = 1.9e-06` and a one-cell thickness error fails it at
  **all 229 gated bins** (§7). The observable the continuum gate cannot resolve
  is the sharpest one the lattice gate has.
- **The margin is 3–50×, not 1.05×.** The worst ratio anywhere is 0.30. The gate
  is not passing by luck, and it is 20–200× tighter than the continuum window it
  sits beside.

### 5.2 cv22 — three rungs, all green; the pole lattice predicts the residual a priori

`validation/crossval/_22_dispersive_results/lattice_witness.json::verdict.all_rungs_ok = true`.

| rung | tails scat / purity | Γ (source) | W_witness R / T | \|rfx − lattice\| R / T | worst ratio R / T / A |
|---|---|---|---|---|---|
| `debye` | 7.2e-3 / 1.5e-4 | 8.05e9 (fitted) | 1.71e-2 / 2.26e-2 | 6.5e-4 / 8.5e-4 | 0.11 / 0.10 / 0.07 |
| `lorentz` | 8.0e-4 / 9.8e-5 | 7.07e9 (fitted) | 2.58e-3 / 2.11e-3 | 2.5e-4 / 1.9e-4 | 0.28 / 0.29 / 0.18 |
| `drude` | 7.6e-5 / 1.2e-4 | 9.42e9 (derived) | 8.25e-5 / 2.93e-4 | 7.2e-6 / 3.5e-5 | 0.19 / 0.19 / 0.18 |

**The new physics result of this lane.** The lattice model, evaluated a priori
from the declared pole and the rig's (dx, dt) with no fitted parameter, predicts
cv22's ENTIRE measured residual against the transfer matrix:

| arm | lattice − TMM, gated mean \|ΔR\| / \|ΔT\| (a priori) | measured rfx − TMM (committed) |
|---|---|---|
| Debye | 0.00222 / 0.00301 | `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.mean_dR_gated = 0.0023` / `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.mean_dT_gated = 0.0028` |
| Lorentz | 0.00284 / 0.00153 | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.mean_dR_gated = 0.0028` / `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.mean_dT_gated = 0.0016` |
| Drude | 0.00049 / 0.00168 | `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.mean_dR_gated = 0.00049` / `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.mean_dT_gated = 0.0017` |

So cv23 §12.2's identification generalizes: **rfx's residual against the
continuum transfer matrix is the Yee lattice's own second-order term on the
dispersive arms too, including the ADE**, and the ADE contributes to it only
through the `ε_num` the `W_ADE` term already named. cv22's dispersive claim is
now a statement about the solver, not about an unresolved 3e-3.

**Stated limitation — the Debye arm's lattice gate is NON-DISCRIMINATING.**
That arm's record settles only to
`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.tail.scat_refl_rel = 0.0072`
(the −40 dB bar is 1e-2, so it is inside its own witness but only just), and its
measured tail decays 1.5× slower than derived, so `W_witness,R = 1.71e-2` —
larger than its own lattice term of 2.2e-3, and larger than its a-priori ceiling
(`rungs.debye.W_exceeds_ceiling_R = true`; the ceiling assumes the derived
rate). The gate passes, but F2 and F3 do not fire there (§7): at this record the
lattice gate cannot tell the lattice model from the continuum one on the Debye
arm. That is recorded as a limitation of the RUNG, not of the standard, and the
remedy rung is pre-declared in §8.1 with its cost. **The Debye lattice gate is
therefore reported as passing-but-non-discriminating, and no claim rests on it.**

### 5.3 cv04 — the witness is REPORTED, not gated, and the derivation says why

cv04 runs one rung (dx = 1 mm, nx 600, 719 steps). `--lattice-witness` writes
`validation/crossval/_04_fresnel_results/lattice_witness.json` with
`gated_here = false`. The reason is derived, not measured: cv04's own tail
witness reads **0.036 / 0.051** of the incident peak (`04_multilayer_fresnel.py`,
the issue-#341 comment block, committed config 2026-07-13) against the family's
−40 dB bar of 1e-2 — cv04's record does not settle, by design (`TAIL_LIMIT =
0.10` there bounds "gross non-settling", and the residual is the documented
order-2 etalon echo still in flight). Put those levels through §3 with the
lossless etalon rate `Γ = 1.65e10 s⁻¹` (ρ = |r|² = 1/9 per round trip, t_rt =
2·2·d/c) and Λ = 48.323:

    W_witness,R = 5.2e-2 in the gated mean,  W_witness,T = 1.7e-1

which is looser than cv04's own band-mean windows. The wrong-model falsifier F2
does not fire there (0 of 115 gated bins), nor does F3 (0 of 115). A gate that
cannot reject the continuum model is not a gate; it is reported.

**cv04's material does have a claims-bearing lattice rung today, at zero cost.**
cv23's `tand0p1_sigma_zero` falsifier arm is cv04's slab exactly — ε′ = 4,
σ = 0, d = 10 mm, dx = 1 mm — on the SETTLED version of the same rig
(`validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_zero.json::arms.tand0p1.params_run`
= `{eps_inf: 4.0, sigma: 0.0}`, 1078 steps, tails 3.6e-3 / 1.2e-3). Gated
against its own material:

| quantity | value |
|---|---|
| `W_witness,R` gated mean | 5.51e-3 |
| `\|rfx − lattice\|` gated mean / max, R | **1.69e-4 / 8.10e-4** |
| worst per-bin ratio, R | **0.06** — passes |
| `\|rfx − TMM\|` gated mean, R | 5.24e-3 |
| `lattice − TMM` gated mean, R | 5.28e-3 |
| F2 (continuum as the model) | fires, 91 of 229 bins |

Read plainly: **cv04's committed band-mean `|ΔR|` — the number the whole slab
family's windows are derived from
(`tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[id=mean_reflectance_error].observed_baseline
= 0.0066`) — IS this slab's Yee-lattice second-order term, not solver error.**
The lattice term over cv04's own masked band (3.03–11.81 GHz, 169 bins) is
0.0072 in `|ΔR|`; over 4–10 GHz it is 0.0053. cv23 §12.2 said this in one
sentence; here it is measured, against a committed artifact, with a derived
window. Nothing about cv04's gates changes — the point is that the envelope those
gates and cv22's and cv23's windows are built on is a discretization number, and
anyone tightening it must refine the mesh, not the tolerance.

## 6. What is NOT changed

- No continuum window, per-bin or band-mean, on any case.
- No settling bar, purity bar or passivity ceiling is loosened; the rig raises on
  an attempt to loosen the settling bar.
- No case's exit code depends on the lattice gate in this commit: `--lattice-witness`
  is a separate post-processing invocation with its own exit code (0 = every rung
  passed), exactly like `--meep-ladder-summary`. Wiring the lattice verdict into
  the cases' own `verdict.exit_code` is a follow-up the PI can call for once the
  VESSL run has confirmed the numbers on a fresh tree; it is NOT done here, and
  that is stated rather than implied.
- cv23's `arms.<arm>.lattice` reported block is unchanged and still written; the
  new artifact does not replace it.

## 7. Falsifiers — analytic, no FDTD

Each falsifier replaces the WITNESS MODEL with a defective one and re-runs the
gate on the same committed measurement
(`lattice_witness.evaluate_falsifier`). "Fires" = the gate rejects. Separation
is `|defective − declared|` in the gated mean over `mean W_witness`; bins is the
count of gated bins above the per-bin window (of 229, or 115 for cv04).

### F1 — a one-cell thickness error, at every rung

The lattice built with one E node more (or fewer) in the slab, i.e. `d ± dx` at
that rung. **Fires at every rung of every arm of both cases** — asserted by
`test_f1_one_cell_thickness_fails_the_lattice_gate_at_every_rung`, which also
requires at least 40 bins over the window on R or T at each.

| rung | sep/W (R) | sep/W (T) | bins over window R / T / A |
|---|---|---|---|
| cv22 debye | 1.23 | 1.08 | 110 / 165 / 127 |
| cv22 lorentz | 6.62 | 7.85 | 197 / 227 / 229 |
| cv22 drude | 179.4 | 71.1 | 229 / 229 / 229 |
| cv23 tand0p1 | 47.0 | 13.0 | 229 / 226 / 196 |
| cv23 tand0p1_dx2 | 27.3 | 14.0 | 229 / 225 / 188 |
| cv23 tand0p1_dx4 | 14.3 | 13.1 | 229 / 223 / 159 |
| cv23 tand1 | 11.5 | 97.4 | 216 / 229 / 198 |
| cv23 tand1_dx2 | 7.5 | 53.6 | 218 / 229 / 157 |
| cv23 tand1_dx4 | 5.2 | 35.2 | 214 / 229 / 132 |
| cv23 tand3 (= dx2) | **0.94** | **174.4** | 113 / 229 / 74 |
| cv23 tand3_dx4 | 0.54 | 93.7 | 101 / 229 / 46 |
| cv04 material (settled) | 14.1 | 17.4 | 228 / 227 / 0 |
| cv04 as committed | 1.50 | 0.45 | 90 / 28 / 0 |

(The `thickness_minus_cell` twin is within 15 % of each row and also fires
everywhere; both are recorded in the artifact.)

Two honest readings. **On tand3, R is thickness-blind** — the back face is
invisible at δ_skin ≈ d/3 and the surface impedance does not depend on d, exactly
as cv23 §12.2 established — so `sep/W(R)` is 0.94 and F1 there is carried by T
(174×) and by the 113 per-bin R failures, not by the R band mean. **On a lossless
slab A ≡ 0 identically**, so the A column is vacuous for cv04's material; that is
geometry, not a gate weakness. The margin is "computed a priori" in the sense the
PI asked: none of these numbers uses `|rfx − lattice|`, only the model and the
window.

### F2 — the deliberately wrong model: the continuum instead of the lattice

The separation is exactly `W_lat(f) = |lattice − TMM|`, the second-order term. So
F2 must STOP firing as the mesh is refined; the rung at which it does is a
convergence statement, and
`test_f2_is_the_lattice_term_itself_and_falls_at_second_order` pins that the
separation falls at order 2.

| case / arm | rungs run | sep/W by rung | **fires at** | silent from |
|---|---|---|---|---|
| cv23 tand0p1 | dx, dx/2, dx/4 | 3.07 / 0.85 / 0.22 | **dx** (198 of 229 bins) and dx/2 (59 bins) | **dx/4** (0 bins) |
| cv23 tand1 | dx, dx/2, dx/4 | 5.95 / 1.68 / 0.54 | **dx** (229 bins) | still fires at dx/4 via T (176 bins); R alone is silent there |
| cv23 tand3 | dx/2, dx/4 | 6.61 / 1.79 | **dx/2** (229 bins), the coarsest rung it runs | — |
| cv22 lorentz | dx | 1.10 | **dx** (141 bins) | — |
| cv22 drude | dx | 5.99 | **dx** (214 bins) | — |
| cv22 debye | dx | **0.13** | **never** — see §5.2 | — |
| cv04 material (settled) | dx | 0.96 | **dx** (91 bins) | — |
| cv04 as committed | dx | 0.10 | **never** — see §5.3 | — |

**Answer to "compute which rung F2 fires at":** F2 fires at the COARSEST rung of
every ladder that runs one — dx for cv23's tand0p1 and tand1 and for cv22's
Lorentz and Drude, dx/2 for cv23's tand3 (its coarsest committed rung) — and it
goes silent at **dx/4 on tand0p1** (0 of 229 bins; on tand1 the T observable
still catches it there). It never fires on cv22's Debye arm and on cv04's
committed configuration, in both cases because the truncation term of the window
exceeds the lattice term at that record; both are §5.2/§5.3's declared
non-discriminating rungs, and both are named in the manifest and the public rows
rather than left to be discovered.

### F3 — a 1 % ε′ perturbation

The declared lattice with the DISPERSIONLESS part of the permittivity (ε′ for
cv04/cv23, ε∞ for cv22) 1 % high.

| rung | sep/W R | bins R / T / A | fires |
|---|---|---|---|
| cv22 debye | 0.02 | 0 / 0 / 0 | **no** (§5.2) |
| cv22 lorentz | 0.50 | 12 / 0 / 0 | yes |
| cv22 drude | 14.6 | 222 / 198 / 157 | yes |
| cv23 tand0p1 / _dx2 / _dx4 | 2.67 / 2.93 / 3.03 | 209 / 212 / 212 in R | yes |
| cv23 tand1 / _dx2 / _dx4 | 0.78 / 0.87 / 1.12 | 76 / 110 / 160 in R, 216 / 219 / 229 in T | yes |
| cv23 tand3 / _dx4 | 0.52 / 0.57 | 94 / 95 in R, 229 / 229 in T | yes |
| cv04 material (settled) | 0.79 | 123 / 128 / 0 | yes |
| cv04 as committed | 0.08 | 0 / 0 / 0 | **no** (§5.3) |

F3 fires on every rung except the two declared non-discriminating ones. On
several arms it is carried by the per-bin gate (GL1), not the band mean — the 1 %
perturbation moves the fringe positions more than the fringe-averaged level, and
GL1 is what sees that.

All three falsifiers, at every rung, are asserted in
`tests/crossval/test_lattice_witness_gates.py::test_falsifiers_fire_exactly_where_the_note_says_they_do`
against the table `_F_FIRES`, which is this section in code.

## 8. New rungs — pre-declared, with cost, not run here

The PI's rule is "add a rung only if a claim requires it, and say what it costs".
Two claims require one; neither rung is run in this commit. Both are legs of
`scripts/vessl_lattice_witness.yaml`, which I have NOT submitted.

### 8.1 cv22 Debye at a 3e-4 settling bar (the only rung a claim requires)

The claim that needs it: "F2 fires at the coarsest rung of every arm". It does
not, on Debye, and §5.2 says why. The remedy is resolution in TIME, not
tolerance: run the same arm with `--settling-bar 3e-4 --tag debye_tail3e4`. The
bar can only be tightened (`slab_rig.py` raises otherwise), so this is not a
window move.

Derivation of the cost and the prediction, from the committed artifact:
the tail must fall from
`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.tail.scat_refl_rel = 0.0072`
to 3e-4 at the fitted rate
`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.tail.fitted_rate_scat_refl_1_s = 8.8e9`
— but the budget uses the conservative
`min(fitted_scat, fitted_trans, derived) = 8.05e9 s⁻¹`, giving
`ln(7.17e-3/3e-4)/(8.05e9 · 2.335e-12) = 169` steps, i.e. **2 adaptive
extensions of 100 → n_steps ≈ 1308**. The nx-1000 CPML gate is 1262
(`…::arms.debye.run.record.t_safe_cpml_steps`), so the existing
`NX_GROW_CELLS = 200` rule grows the box to nx 1200 (gate ≈ 1514) — that path is
already in the script and needs no change. **Cost: ≈ 1.7× one cv22 arm, which
is ~1 s on the pod.**

**Prediction, both branches stated before the run.** `W_witness,R` becomes
**7.3e-4** (the truncation term falls by 24×, the other terms are unchanged).
Then:

- if the truncation identification is right, `|rfx − lattice|` falls with the
  tail from 6.5e-4 to ≈ 2.7e-5, the gate passes with ~27× margin and F2 then
  fires on the Debye arm at **3.05×** (229 of 229 bins in R). The Debye lattice
  gate becomes discriminating and §5.2's limitation is closed.
- if instead the residual stays at 6.5e-4, the gate **FIRES** (worst per-bin
  ratio 2.55) and the finding is that the Debye arm's residual against its own
  lattice is NOT truncation-dominated — which would be a real defect to chase,
  in the ADE or in the extraction, and would be reported as such. No window would
  be widened either way.

This is a risky prediction on purpose. The leg's rc is admissible either way in
the YAML, and it runs into its own results directory so it cannot change cv22's
committed witness document.

### 8.2 cv04's slab on the settled rig (a re-run of a committed leg, for provenance)

§5.3's claims-bearing cv04-material rung is cv23's `tand0p1_sigma_zero` artifact.
The YAML re-runs that falsifier so the cv04 row rests on a leg of this lane too;
it must exit 1 (it is a cv23 falsifier). **Cost: one cv23 arm, ~1 s.** No new
recipe.

### 8.3 Rungs deliberately NOT added

- **cv22 dx/2 and dx/4.** cv22 r2 ran them as diagnostics and did not commit
  them. They would make cv22's lattice claim a ladder rather than a single rung,
  and they are cheap (~4× and ~16× of a 1 s arm). They are NOT added, because no
  claim in this note requires them: the single-rung lattice gate is already a
  rigorous statement at that rung, and F2's convergence reading is supplied by
  cv23's ladder on the same rig. If the PI wants cv22's ladder, it is
  `--dx-div 2|4 --tag <arm>_dx<K>` and about 30 s of pod time.
- **A cv04 rung at the settled recipe with cv04's own script.** cv04 has no
  record derivation and no `--nx-interior`; adding one would re-engineer a
  committed legacy configuration for no new physics (cv23's `sigma_zero` arm
  already measures that material at that recipe on that rig). Recorded as a
  deliberate non-change.

## 9. Artifacts and keys

- `validation/crossval/_22_dispersive_results/lattice_witness.json` — schema
  `lattice-witness/v1`; `rungs.<rung>` carries `R/T/A_lattice`,
  `dR/dT/dA_lattice`, `W_witness_{R,T,A}` per bin; `budget.*` (Λ, κ, κ_i, Γ and
  its source, the four δ terms as gated means, the coherent round-off
  alternative); `mean/max_d{R,T,A}_lattice_gated`,
  `mean_W_witness_{R,T,A}_gated`, `mean_W_ceiling_{R,T,A}_gated`,
  `W_exceeds_ceiling_{R,T,A}`, `worst_ratio_{R,T,A}`, `n_bins_*_over_window`;
  `gates.{precond_cpml_gate, precond_tail_witness, GL1_R, GL1_T, GL1_A, GL2_R,
  GL2_T, GL2_A}`; `falsifiers.<kind>` with its own gates, bins and
  separation ratios; top-level `verdict.{all_rungs_ok, n_rungs}`.
- `validation/crossval/_23_lossy_results/lattice_witness.json` — the same, nine
  rungs.
- `validation/crossval/_04_fresnel_results/lattice_witness.json` — one rung,
  written by `python validation/crossval/04_multilayer_fresnel.py
  --lattice-witness`, with `gated_here = false` and `gated_here_reason`. **Not
  committed in this PR**: it needs an FDTD run of cv04, which is a VESSL leg
  (`scripts/vessl_lattice_witness.yaml` section 6), and the manifest contract
  requires every listed artifact path to exist. cv04's `artifact_paths` therefore
  stays empty and §5.3's cv04 numbers are DERIVED here, not read from an
  artifact — stated plainly rather than cited to a file that does not exist.

Both committed witness JSONs are rebuilt from the committed `rfx*.json` rungs by
`test_committed_witness_artifact_rebuilds_from_the_committed_rungs`, so they can
never drift from the measurements they summarize.

## 10. Dead ends and things I could not close

Recorded because the record is the point.

1. **A tighter truncation bound exists and I did not take it.** `Σ_{n≥N}|s_n|` is
   a COHERENT bound: it is attained only at a frequency where the whole remaining
   tail adds in phase, which for an etalon ring-down is near the resonance, not
   at every bin. A per-bin Lorentzian bound (the tail is a decaying mode at the
   etalon frequency, transform `A/(Γ + j(ω−ω_r))`) would be ~10× tighter and
   would make the cv22 Debye and cv04 rungs discriminating with no extra
   compute. It is NOT taken because it assumes a single-mode tail, which is a
   model, not a bound, and this lane's whole point is that the window is derived
   rather than modelled. The looser rigorous bound is what is gated; the tighter
   one is named here so a future lane can pre-declare it with its own witness.
2. **The float32 term is a statistical estimate, not a bound.** §3(T3). The
   coherent worst case is 2 orders above the measured residual; if it were
   attained every rung would fail by ~100× and none does, which is evidence but
   not proof. The clean way to close it is one leg at `field_dtype=float64` under
   `tests/_x64_compat.enable_x64()` and a direct measurement of the difference.
   That is an FDTD leg plus a process-global x64 flip in a shared rig, and this
   repo has explicit rules about the latter (`conftest.py:305-351`); it is out of
   this lane's scope and is left open.
3. **Γ needs a decay assumption.** §3. There is no way to bound the un-recorded
   tail without one. The passive-slab argument plus the fitted-rate witness is
   what stands behind it.
4. **cv22's Debye and cv04's committed rungs do not discriminate.** §5.2, §5.3,
   §7. Both are declared, both have a pre-declared remedy (§8.1 for Debye; a
   settled record for cv04), and no claim in the manifest or the public rows
   rests on either.
5. **`docs/design_notes/20260903_e4_all_solver_classes_plan.md` is not on
   `origin/main`.** It lives on `origin/agent/e4-all-solvers-plan` at `eb78254`
   and was read from there (§1). Its lane L1 is the consumer of this rig; nothing
   in this note depends on it landing.
6. **No `26_*` / oblique case exists to exclude.** §1. The exclusion is stated as
   a scope boundary of the 1-D derivation, not as a reference to a case in the
   tree.
7. **The lattice verdict is not wired into the cases' exit codes.** §6. It is a
   separate invocation with its own rc. Wiring it changes what a red case means
   and should follow a green VESSL run, not precede it.
8. **The committed cv23 `arms.<arm>.lattice.R_lattice` arrays reproduce to 1e-15,
   not bit-for-bit,** when recomputed today. The refactor of
   `yee_lattice_slab_rt` is bit-for-bit identical to its pre-refactor self
   (checked against `git show HEAD:…`), so the 1e-15 is an environment
   difference in the r3 pod's numpy/BLAS, not this lane. Recorded, not chased.

## 11. What would refute this note

- Any committed rung failing GL1 or GL2 → either the lattice identification is
  wrong at that mesh, or a budget term is missing. No window is widened; the term
  is found or the identification is withdrawn.
- F1 not firing at some rung → the gate does not resolve a one-cell geometry
  error there and the rung's lattice claim is not claims-bearing.
- F2 firing at a rung where §7 says it is silent, or silent where §7 says it
  fires → the convergence reading of the ladder is wrong.
- The §8.1 Debye rung's residual NOT falling with its tail → the Debye arm's
  residual is not truncation-dominated and §5.2's explanation is withdrawn.
- `mean_W_witness_*` exceeding `mean_W_ceiling_*` on a rung whose fitted rate is
  NOT slower than its derived rate → a term in §3 is mis-signed.
- A measured float64-vs-float32 difference above `budget.mean_delta_round_gated`
  → §3(T3)'s statistical estimate is the wrong one and the coherent term must be
  carried, which would loosen every window in §5 by ~3×. That is a finding, not a
  window move: it would be a correction to a DERIVED term, appended here.
