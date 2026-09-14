# Lattice-witness standardisation — pre-declaration (slab family: cv04, cv22, cv23)

Date: 2026-09-03 · Lane: `agent/lattice-witness-standard` · Cases:
`validation/crossval/04_multilayer_fresnel.py`,
`validation/crossval/22_dispersive_slab_fresnel.py`,
`validation/crossval/23_lossy_slab_fresnel.py`.

**Append-only from the moment it lands.** Corrections are added as new sections;
nothing above a correction is edited. The one exception is this document's own
pre-merge review: the corrections of the 2026-09-03 independent review, listed
in §12, were applied IN PLACE, because a claim that was measured to be wrong
should not survive into `main` merely to be contradicted three sections later.
Every one of them is itemised in §12 with what it said, what the measurement
says, and where. After the merge the append-only rule is absolute.

Every number here is ANALYTIC (closed-form ε(f), the transfer matrix, the exact
time-harmonic solution of the Yee lattice, the z-transform of the updates) or
READ from a committed artifact by a `path.json` reference of the form
path-double-colon-key-equals-value — and every such reference in this note is
now resolved and value-checked by
`tests/contracts/test_evidence_numeric_provenance.py`, which the note is opted
into (33 references). No FDTD was run FOR this note: every measured number it
quotes was already committed by cv22 round 4 or cv23 round 3, or is read from
the `lattice_witness.json` files the VESSL run of 2026-09-03 wrote (§9.1), which
are post-processing of those, the same class as `--meep-ladder-summary` and
`--refit-tail-fits`.

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
`validation/crossval/comparators/slab_rig.py` REFUSES any `--settling-bar`
outside `(0, SETTLING_LIMIT]` — the comparison is against the FAMILY constant
`SETTLING_LIMIT = 1e-2`, not against whatever bar the active recipe carries
(`test_settling_bar_outside_the_declared_interval_is_refused`,
`test_settling_bar_can_only_be_tightened`).

**Correction (2026-09-03 review, applied here).** As first written the guard
compared the requested bar against the ACTIVE recipe's `tail_limit`, so it was
not the sentence above. `--recipe cv04` is a declared flag of both
`22_dispersive_slab_fresnel.py` and `23_lossy_slab_fresnel.py`, and under it the
bar was checked against cv04's `TAIL_LIMIT = 0.10`: a `--settling-bar 5e-2` —
five times LOOSER than the declared −40 dB witness — was accepted. A
non-positive bar (`0.0`, `-1.0`) also passed the `<=` test under r3 and switched
the witness off entirely. All four probes are now refused and all four are
asserted as tests; a genuine tightening (`3e-4`, §8.1) is still accepted.

## 1. Scope — which cases are slab family, and why

| case | in scope | why |
|---|---|---|
| `04_multilayer_fresnel` | **yes** | the rig every other slab case is a copy of; one slab, ε′ = 4, d = 10 mm, dx = 1 mm, normal incidence, 2-D TMz with periodic y. Its committed band-mean `\|ΔR\|` = 0.0066 and `\|ΔT\|` = 0.011 are the envelopes cv22 and cv23 derive `W_mean,R` and `W_mean,T` from, so if those numbers are the lattice term the family's window derivation rests on a discretization artefact. §5.3 shows that this holds for `\|ΔR\|` and **not** for `\|ΔT\|`: the lattice term is the same 0.00727 in both, but it is 1.10× cv04's R envelope and only 0.66× its T envelope, the rest of T being this record's own truncation. |
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

`validation/crossval/comparators/dispersive_eps.py` `yee_lattice_slab_rt_eps` is
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
ordinary (`rfx/core/yee.py` `update_e`), conductive (same, `σ ≠ 0`), Debye
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

**Numerically verified** (`tests/crossval/test_lattice_witness_gates.py`
`test_extended_lattice_reproduces_the_cv23_solver_it_generalizes`):
`yee_lattice_slab_rt_model("conductive", …)` equals `yee_lattice_slab_rt`
**bit-for-bit at σ = 0** and to a **measured maximum of 1.7e-15 in R and T** on
all three cv23 arms at the test's grid (the note first said "≤ 1e-15", which the
measurement does not support; the test asserts the 1e-14 class, which is the
part that is stable across environments — §10(8)) —
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

> **SCOPED 2026-09-14 by §13 (#1015).** Every word below stands, and no number
> in it moves. What §13 adds is where it is allowed to be applied: this
> derivation bounds three terms and DECLARES three zero by construction, and on
> a rig that cannot make one of those declarations the per-bin gate built on it
> (GL1, §4) has no validity. §13 states that precondition and measures it. §3's
> τ is also the CASE's source pulse width, not the family constant — §13.1.

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
`inc_peak` in `slab_rig.py` `_witness`) and Γ is the slowest amplitude decay rate.

**(T2) Incident-reference truncation** — "injection leakage" as this rig
witnesses it. The 1-D auxiliary reference that forms the denominator is truncated
at the same step; its level is cv04's tail-purity witness `tail.purity_inc_rel`
(bar 1e-3) and its envelope rate is the differentiated Gaussian's own,
`Γ_inc = 2a/τ` with `2a e^{−a²} = purity`, `a > 1/√2`.

**(T3) float32.** The fields are float32 (`rfx/core/yee.py` `init_state`,
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

> **SCOPED 2026-09-14 by §13 (#1015).** GL1 below reads "at every gated bin".
> §13 narrows that to "at every gated bin IN THE VALIDITY DOMAIN", and the
> domain is every gated bin for cv04, cv22 and cv23 — so nothing in this
> section's verdicts, windows or numbers changes. It is not every gated bin on
> a rig that admits an absorber echo inside its record by amplitude.

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
`validation/crossval/comparators/lattice_witness.py` `evaluate` and written to
`lattice_witness.json` beside the case's other artifacts by
`<case>.py --lattice-witness`; it returns the same gate-record shape as the other
cv gates (per-bin arrays, gated scalars, a boolean `gates` dict, `witness_ok`).

## 5. The rungs, and what they measure

Reuse only: no rung was added to get these numbers. cv23's ladder is the one
round 2 ran and round 3 committed; cv22 runs one rung per arm; cv04 runs one.

### 5.1 cv23 — nine committed entries, eight distinct meshes, all green

`validation/crossval/_23_lossy_results/lattice_witness.json::verdict.all_rungs_ok`,
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
  is not passing by luck.
- **How much tighter than the continuum window it sits beside: 5–8800×
  (median ≈ 20×), not "20–200×".** The 2026-09-03 review recomputed the claim
  per observable and it was wrong at both ends. The ratio is
  `mean_window_{R,T,A}` (the arm's own committed continuum band-mean window,
  `rfx.json`) over `mean_W_witness_{R,T,A}_gated`, on the same gated bins:

| rung | band-mean R | T | A | per-bin floor R | T | A |
|---|---|---|---|---|---|---|
| `tand0p1` | 7.8 | **5.2** | 5.9 | 12.0 | **7.1** | 8.9 |
| `tand0p1_dx2` | 8.9 | 10.9 | 10.0 | 13.2 | 14.4 | 13.7 |
| `tand0p1_dx4` | 9.2 | 20.2 | 14.0 | 13.9 | 27.1 | 18.3 |
| `tand1` | 11.8 | 137.5 | 27.6 | 26.0 | 183.8 | 45.4 |
| `tand1_dx2` | 13.4 | 138.1 | 30.9 | 28.6 | 176.8 | 49.2 |
| `tand1_dx4` | 17.4 | 174.3 | 40.1 | 37.7 | 225.3 | 64.6 |
| `tand3` = `tand3_dx2` | 21.3 | **8772** | 57.2 | 47.4 | 13211 | 94.4 |
| `tand3_dx4` | 23.1 | 8649 | 62.0 | 52.1 | 13101 | 103.7 |

  Read plainly: the minimum over the 27 (rung, observable) pairs is **5.2×**
  (`tand0p1`, T), the per-bin floor is **7.1×** (same pair), the maximum is
  **8772×** (`tand3`, T), and the median is **21×**. **Six of the nine rungs are
  below 20× in at least one observable** — every `tand0p1` and every `tand1`
  rung. The lattice gate is uniformly tighter than the continuum gate on cv23,
  but "20–200×" overstated the floor by 4× and understated the ceiling by 44×,
  and the previous wording hid that the tightest observable on the arm that
  matters least (`tand3` T, whose continuum gate is vacuous) is what carried the
  high end.

  On cv22 the same ratio is **not** uniformly above 1: the Debye arm's lattice
  window is 0.6× (R) and 0.8× (T) of its continuum window — i.e. LOOSER — which
  is §5.2's declared non-discriminating rung stated as a number. Lorentz is
  4.0× / 8.3× and Drude 122× / 58×.

### 5.2 cv22 — three rungs, all green; the pole lattice predicts the residual a priori

`validation/crossval/_22_dispersive_results/lattice_witness.json::verdict.all_rungs_ok`, `validation/crossval/_22_dispersive_results/lattice_witness.json::verdict.n_rungs = 3`.

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
remedy rung is pre-declared in §8.1 with its cost. **The Debye lattice gate at
THIS rung is therefore reported as passing-but-non-discriminating, and no claim
rests on it.**

**Closed by the §8.1 remedy rung, which the VESSL run did take.** The note first
said §8.1 was pre-declared and not run; the run of 2026-09-03 ran it
(`--arms debye --settling-bar 3e-4 --tag debye_tail3e4`, into its own results
directory so it cannot touch the committed cv22 document — §9). The risky
prediction's FIRST branch is what happened: the tail fell to
`validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.budget.scat_tail_rel = 1.42e-4` at
`validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.n_steps = 1451` steps, `W_witness,R` fell to
`validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.mean_W_witness_R_gated = 0.000332` (from 1.71e-2, a
52× shrink), `|rfx − lattice|` fell with it to
`validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.mean_dR_lattice_gated = 1.06e-5` (from 6.5e-4), the
gate passes with worst per-bin ratio
`validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.worst_ratio_R = 0.092`, the window no longer exceeds
its a-priori ceiling (`validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.W_exceeds_ceiling_R`, now
false), and **F2 fires on the Debye arm at 6.68× on 229 of 229 gated bins in R**
(F3 too, at 1.07×). §7's tables carry the rung.

Two things worth reading off it. First, the arm's residual against the CONTINUUM
transfer matrix did **not** move — `mean|ΔR|` is 0.00222 at 1451 steps against
0.0023 at 1108 — which is exactly what §5.2 predicted a priori from the lattice
model (0.00222) and is the strongest single confirmation in this note that the
residual is the lattice term and not truncation. Second, the truncation
identification of §5.2 is therefore CONFIRMED, not merely asserted: the part of
`|rfx − lattice|` that fell by 61× with the record was truncation, and the part
that stayed is the lattice.

### 5.3 cv04 — the witness was REPORTED, not gated; the derivation said why, and the settling-extension fix (2026-09-10) closed it

**UPDATE 2026-09-10, read this first.** Section 8.3 below declined to give
cv04 its own settled record ("no new physics"); the PI overrode that call --
a case that cannot settle its own spectra cannot gate its own case, and the
un-extendable legacy record was the defect this campaign exists to fix, not a
constraint to protect. cv04 now runs the SAME settling-extension mechanism
cv22/cv23 already had (grow `nx_interior` by 200 until both tails clear the
family's −40 dB bar), reaching
`validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.n_steps = 990`
at `nx_interior = 800` (one grow attempt from the declared 600), tails
0.0004 / 0.0001 of the incident peak (was 0.036 / 0.051). The witness is now
GATED (`gated_here = true`):

| quantity (cv04 AS COMMITTED, POST-FIX, 990 steps) | value |
|---|---|
| `W_witness,R` gated mean | `validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.mean_W_witness_R_gated = 0.000635` |
| `W_witness,T` gated mean | `validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.mean_W_witness_T_gated = 0.000291` |
| `\|rfx − lattice\|` gated mean, R | `validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.mean_dR_lattice_gated = 2.01e-05` |
| `\|rfx − lattice\|` gated mean, T | `validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.mean_dT_lattice_gated = 9.52e-05` |
| worst per-bin ratio, R / T | `validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.worst_ratio_R = 0.081` / `validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.worst_ratio_T = 0.551` |
| a-priori ceiling, R | `validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.mean_W_ceiling_R_gated = 0.0149` |
| Γ, source (ringdown rate) | `validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.budget.rate_ringdown_1_s = 1.647e10` |

Both pass with margin; T's margin (0.551) is far tighter than R's (0.081) even
though T's absolute residual also shrank ~66×, because T's window shrank
further still (~611×) -- the window, not the residual, is now T's binding
constraint. The ceiling and Γ are geometry-derived, not record-length-derived,
so the fix barely moves them: Γ is bit-identical (16467783920.44 s⁻¹ both
sides) and the ceiling moves only in its 6th significant digit (0.0148608
pre-fix, 0.0148604 post-fix; both round to 0.0149) -- cited live above
against the current artifact rather than repeated as plain numbers below.

The wrong-model and 1 % ε′ falsifiers WERE re-run against this settled rung,
and both now discriminate strongly where they previously did not:
`validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.falsifiers.continuum.separation_over_window_R = 8.35`
(F2; was 0.099 pre-fix, 0 of 115 bins over window) and
`validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.falsifiers.eps_x1p01.separation_over_window_R = 6.87`
(F3; was 0.082 pre-fix, 0 of 115 bins over window); both now clear 111 of 115
gated bins in R (§7's F2/F3 tables carry the added rows). Both separations
above are R-channel (`separation_over_window_R`), so the relevant shrink is
R's own W_witness window, ~84× (N2, PR #974 round 2: mean_W_witness_R_gated
0.0535 -> 0.000635; T's window shrank further still, ~611×, but that number
belongs with T's own separation, not this one). It is that ~84× window
shrink from settling that turns the same physical model-mismatch and 1 %
ε′-perturbation terms from invisible into strongly discriminating -- nothing
about the perturbations themselves changed.

The rest of this section (below) describes the PRE-FIX 719-step record as
committed 2026-07-13 through 2026-09-10 -- kept for the physical reasoning
(the settling-rate derivation via Γ, Λ still applies) and as the historical
record the fix superseded, not as a claim about the artifact today. Most
numbers below are stated plainly, without a live citation, because
`lattice_witness.json` no longer holds them -- `W_witness,R/T` and the
worst-bin ratios have no other committed home; retrieve them at the pinned
pre-fix commit, `git show e079b0b5:validation/crossval/_04_fresnel_results/lattice_witness.json`.
The two `|rfx − lattice|` gated-mean numbers (R, T) are the exception: they
are live-cited below against the immutable r1 revision of `envelope.json`,
which archived them before the fix and does not change when the producer is
re-run.

cv04 ran one rung (dx = 1 mm, nx 600, 719 steps). `--lattice-witness` wrote
`validation/crossval/_04_fresnel_results/lattice_witness.json` with
`gated_here = false`. The reason was derived, not measured: cv04's own tail
witness read **0.036 / 0.051** of the incident peak (`04_multilayer_fresnel.py`,
the issue-#341 comment block, committed config 2026-07-13) against the family's
−40 dB bar of 1e-2 — cv04's record did not settle, by design (`TAIL_LIMIT =
0.10` there bounds "gross non-settling", and the residual was the documented
order-2 etalon echo still in flight). Put those levels through §3 with the
lossless etalon rate `Γ = 1.65e10 s⁻¹` (ρ = |r|² = 1/9 per round trip, t_rt =
2·2·d/c) and Λ = 48.323, and the run confirmed the derivation:

| quantity (cv04 AS COMMITTED PRE-FIX, 719 steps, 115 gated bins — historical, superseded 2026-09-10) | value |
|---|---|
| `W_witness,R` gated mean | 0.0535 (retrieval above: `lattice_witness.json` at `e079b0b5`) |
| `W_witness,T` gated mean | 0.178 (same retrieval) |
| `\|rfx − lattice\|` gated mean, R | `validation/crossval/_04_fresnel_results/envelope.json::revisions.r1.lattice_identification.rfx_vs_lattice_mean_dR_gated.value = 0.00168` |
| `\|rfx − lattice\|` gated mean, T | `validation/crossval/_04_fresnel_results/envelope.json::revisions.r1.lattice_identification.rfx_vs_lattice_mean_dT_gated.value = 0.00625` |
| worst per-bin ratio, R / T | 0.095 / 0.056 (same retrieval) |

(The a-priori ceiling and Γ are omitted from this table on purpose: both are
geometry-derived, not record-length-derived, and are cited live in the
POST-FIX table above at values unchanged to displayed precision -- repeating
them here as plain pre-fix numbers would suggest they moved with the fix,
which they did not.)

which was looser than cv04's own band-mean windows, and 3.6× looser than its own
a-priori ceiling because the measured tails were 3.6× and 5.1× the declared bar.
The wrong-model falsifier F2 did not fire there (separation 0.099 of the
window, 0 of 115 gated bins), nor did F3 (0.082, 0 of 115). A gate that cannot
reject the continuum model is not a gate; it is reported -- which was the
state this section's title described, before the fix above.

**These are the run's numbers, not a reconstruction.** The note's first draft
carried derived values for this table that were 3–15 % off — 5.2e-2 for a
measured 5.35e-2, 1.7e-1 for 1.78e-1, and F1 / F2 / F3 separations of
1.50 / 0.45, 0.10 and 0.08 for a measured 1.45 / 0.43, 0.099 and 0.082. Every
number above WAS a key of the committed artifact that original VESSL run
wrote (§9.1) -- retrievable today at the pinned pre-fix commit, `git show
e079b0b5:validation/crossval/_04_fresnel_results/lattice_witness.json`, not
at the current one, which the 2026-09-10 settling fix superseded.

**cv04's material does have a claims-bearing lattice rung today, at zero cost.**
cv23's `tand0p1_sigma_zero` falsifier arm is cv04's slab exactly — ε′ = 4,
σ = 0, d = 10 mm, dx = 1 mm — on the SETTLED version of the same rig
(`validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_zero.json::arms.tand0p1.params_run.eps_inf = 4.0`,
`validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_zero.json::arms.tand0p1.params_run.sigma = 0`,
`validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_zero.json::arms.tand0p1.run.n_steps = 1078`
steps, tails 3.6e-3 / 1.2e-3). Gated
against its own material:

| quantity | value |
|---|---|
| `W_witness,R` gated mean | 5.51e-3 |
| `\|rfx − lattice\|` gated mean / max, R | **1.69e-4 / 8.10e-4** |
| worst per-bin ratio, R | **0.06** — passes |
| `\|rfx − TMM\|` gated mean, R | 5.24e-3 |
| `lattice − TMM` gated mean, R | 5.28e-3 |
| F2 (continuum as the model) | fires, 0.958 of the window, 91 of 229 bins |
| F3 (ε′ 1 % high) | fires, 0.790, 123 of 229 bins |
| F4 (continuum ε in the lattice) | **0 by construction** — σ = 0, no pole (§7) |

**The settled rung is ten times sharper than cv04's own.** `|rfx − lattice|` is
**1.69e-4** in the gated mean here against **1.68e-3** at cv04's committed
719-step rung — a factor of ten, and the whole of it is record truncation: the
material, the mesh and dt are identical, only the record length differs (1078
steps against 719, tails 3.6e-3 / 1.2e-3 against 0.036 / 0.051). In T the same
comparison is 1.34e-4 against 6.25e-3, a factor of 47.

Read plainly: **cv04's committed band-mean `|ΔR|`
(`tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[0].observed_baseline = 0.0066`),
the number `W_mean,R = 0.010` is derived from, IS this slab's Yee-lattice
second-order term, not solver error.** cv23 §12.2 said this in one sentence;
here it is measured, against a committed artifact, with a derived window.

**Scope of that claim: `|ΔR|` only. It does NOT extend to `|ΔT|`, and the
2026-09-03 review was right to say so.** The lattice term itself is the SAME
number in the two observables — 0.00727 over cv04's own mask (3.032–11.867 GHz,
170 bins), 0.00530 over the 115 gated bins — because the slab is lossless, so
`A ≡ 0` and `|R_lat − R_TMM| = |T_lat − T_TMM|` bin by bin. cv04's committed
envelopes are not the same number:

| observable (PRE-FIX, 719 steps, historical) | committed envelope | lattice term over the same mask | lattice / envelope | `\|rfx − lattice\|`, gated mean |
|---|---|---|---|---|
| `\|ΔR\|` | `tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[0].observed_baseline = 0.0066` | 0.00727 | **1.10** | `validation/crossval/_04_fresnel_results/envelope.json::revisions.r1.lattice_identification.rfx_vs_lattice_mean_dR_gated.value = 0.00168` |
| `\|ΔT\|` | `tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[1].observed_baseline = 0.011` | 0.00727 | **0.66** | `validation/crossval/_04_fresnel_results/envelope.json::revisions.r1.lattice_identification.rfx_vs_lattice_mean_dT_gated.value = 0.00625` |

(The last column is live-cited against the immutable r1 revision of
`envelope.json`, not against `lattice_witness.json` -- the settling fix
regenerated `lattice_witness.json` in place, so it no longer holds the
PRE-FIX 719-step reading, but r1 archived these two numbers before the fix
and does not move when the producer is re-run. See the 2026-09-10 update at
the top of this section for the current, settled values,
`mean_dR_lattice_gated = 2.01e-05` and `mean_dT_lattice_gated = 9.52e-05`.)

In R the lattice accounted for the whole PRE-FIX envelope (1.10×, i.e. the
identification was complete to within the residual the last column measured).
In T it accounted for two thirds, and the residual against the lattice at
cv04's own PRE-FIX rung was **0.00625** — **more than half of the 0.011
envelope was that record's TRUNCATION, not a discretisation term.** The
settled (POST-FIX) record confirms this reading directly rather than by
extrapolation: at 990 steps `\|rfx − lattice\|` in T fell to 9.52e-05, 0.87%
of the same 0.011 envelope. Read as CONSISTENT with the truncation
diagnosis, not as its own new answer to "what fraction of a properly re-derived
T envelope is the lattice term" -- the T envelope itself (0.011) is a PRE-FIX
number too (`envelope.json` r1, §5.3's own update), and re-deriving it (r2,
issue #928) is a separate, not-yet-done step.

**Consequence for the family's windows, stated rather than implied.**
`W_mean,R = 0.010 = gate_from_envelope(0.0066, quantum=1000)` was derived from a
PRE-FIX number this note showed to be a discretisation term (and the 2026-09-10
settled record confirms directly, not just by inference from cv23's sibling
arm). `W_mean,T = 0.017 = gate_from_envelope(0.011, quantum=1000)` was **not**:
it was derived from a truncation-dominated PRE-FIX number, and `W_MEAN_T` must
not be described anywhere as a discretisation number -- that conclusion still
stands, though the T envelope it was derived from is itself now a candidate for
re-derivation (issue #928) rather than the last word. Nothing about cv04's
gates changed with the settling fix and no window moved; what changed is that
cv04 can now confirm this section's claims on its own record instead of
borrowing cv23's. Anyone tightening `W_mean,R` must refine the mesh, not the
tolerance; anyone tightening `W_mean,T` should look at whether the settled
record changes the mesh-vs-record-length balance the old advice ("lengthen
cv04's record") assumed, now that lengthening it is exactly what happened.

### 5.4 The consequence §5.3 states the premise of, and does not draw (added after the 2026-09-03 review)

§5.3 says cv04's committed `|ΔR|` envelope IS a discretisation term. §5.2 and
§5.1 say every arm's residual against the transfer matrix IS that arm's own
lattice term. **Put together, those two say something about the family's
CONTINUUM gate that the note stated the premise of and never the consequence.**
The review asked the question directly — "does anything pass only because its
window carries the lattice term?" — and the answer is yes.

The experiment is a re-derivation, not a re-run. `W_mean,R = 0.010` is
`gate_from_envelope(0.0066, quantum=1000)` — cv04's envelope, which §5.3 shows
is 1.10× this slab's lattice term. Re-derive it from the NON-lattice part of the
same measurement instead: at the settled rung of the same slab (cv23's
`sigma_zero` arm) `|rfx − lattice|` is 1.69e-4, and
`gate_from_envelope(1.69e-4, quantum=1000)` = **0.001**. Judged against that,
of the twelve committed rungs:

| rung | measured `\|rfx − TMM\|`, gated mean R | that rung's OWN lattice term | ≤ 0.001? |
|---|---|---|---|
| cv22 `debye` | 0.0023 | 0.0022 | **no** |
| cv22 `lorentz` | 0.0028 | 0.0028 | **no** |
| cv22 `drude` | 0.00049 | 0.00049 | yes |
| cv23 `tand0p1` | 0.0039 | 0.0039 | **no** |
| cv23 `tand0p1_dx2` | 0.00096 | 0.00096 | yes |
| cv23 `tand0p1_dx4` | 0.00024 | 0.00024 | yes |
| cv23 `tand1` | 0.0051 | 0.0051 | **no** |
| cv23 `tand1_dx2` | 0.0013 | 0.0013 | **no** |
| cv23 `tand1_dx4` | 0.00031 | 0.00031 | yes |
| cv23 `tand3` = `tand3_dx2` | 0.0031 | 0.0031 | **no** (both) |
| cv23 `tand3_dx4` | 0.00078 | 0.00078 | yes |

**Seven of the twelve committed rungs pass their continuum band-mean R gate only
because `W_mean,R = 0.010` carries cv04's lattice term.** And the middle column
equals the right-hand column to two significant figures at every single rung —
which is this lane's whole result, read the other way round: what the continuum
gate measures at these meshes IS the discretisation.

Said in one sentence, and it belongs on the public rows as well as here: **the
slab family's continuum gate is one slab's discretisation error measured inside
a window derived from another slab's discretisation error.** The five rungs that
survive the re-derivation are the refined ones (`dx/2`, `dx/4`) plus Drude,
i.e. exactly the rungs where the lattice term has fallen far enough that the
solver's own residual is what is left.

**Nothing is changed here, deliberately.** No window is widened, none is
tightened, no gate moves, and `W_mean,R` stays 0.010. Two reasons, both stated
so that the non-change is a decision and not an omission:

1. Tightening `W_mean,R` to 0.001 would turn seven green cases red for a reason
   that is not a defect — the mesh, not the solver. The remedy for a
   discretisation-limited gate is resolution, and choosing which rungs to refine
   (and paying for them) is a PI decision, not a review action.
2. The re-derived 0.001 is itself only as good as the settled rung it comes
   from; it inherits that rung's own truncation and float32 terms (§3), and
   §7's F4 shows the record length, not the mesh, is what limits several arms.

**Follow-up lane, proposed here and not taken: `slab-family continuum window
re-derivation`.** Its content: (a) re-derive `W_mean,R` and `W_mean,T` from
`|rfx − lattice|` at a SETTLED rung of each material rather than from cv04's
un-settled envelope, with the lattice term carried as a separate, per-rung,
per-mesh term instead of being folded into a constant; (b) run the refined rungs
each arm then needs — from the table above that is the dx rungs of cv22 Debye
and Lorentz and of cv23 `tand0p1`, `tand1` and `tand3`; (c) re-run cv04 itself
on a settled record so its envelope stops being the family's datum while being
the one measurement in the family that does not settle.
**(c) is SUPERSEDED 2026-09-10** (N7, PR #974 round 2): the settling-extension
fix did this -- cv04 now runs on a settled 990-step record (§5.3's top
UPDATE). Its ENVELOPE (`envelope.json` r1, mean |dR|/|dT| = 0.0066/0.011) is
still the pre-fix, un-settled measurement, though; re-deriving the envelope
itself from the settled record is r2 (issue #928, drafted, held for its own
PR), a narrower and now-separate task from what (c) originally proposed.
(a) and (b) remain open. Cost: five to seven arms, order a minute of pod
time; the expensive part is the decision in (a), not the compute. Nothing in
THIS lane depends on it, and this lane's gates are unaffected either way.

## 6. What is NOT changed

- No continuum window, per-bin or band-mean, on any case.
- No settling bar, purity bar or passivity ceiling is loosened; the rig raises on
  an attempt to loosen the settling bar.
- **SUPERSEDED, 2026-09-10.** This bullet described the 2026-09-03 state only:
  no case's exit code depended on the lattice gate, `--lattice-witness` was a
  separate post-processing invocation with its own exit code, and wiring it
  into a case's own verdict was left as a PI call. That call was made twice
  since: issue #970 wired cv23's `main()` (`GL_witness` folds into `e2_ok`,
  live on this run's own freshly-computed R/T, `23_lossy_slab_fresnel.py`),
  and PR #974 round 2 (B1) wired cv04's `main()` the same way -- the witness
  now EVALUATES on every cv04 run, not only under `--lattice-witness` (the
  flag now controls only whether the artifact is rewritten), and folds into
  `rfx_self_ok`. cv22 is the one case where this bullet is still literally
  true: its own `witness_ok`-type value is computed and never read by
  `main()` (the #970 PR body's bounded pattern grep names it; left untouched
  there, a separate decision).
- No committed rung of any case is added, removed or re-run. §8.1's remedy rung
  is a NEW rung and lives in `_22_dispersive_diag/`, outside every case's
  `rungs_from_results` glob, precisely so that this bullet stays true.
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
| cv22 debye @ 3e-4 (§8.1) | 63.6 | 98.5 | 228 / 229 / 229 |
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
| cv04 as committed, PRE-FIX (719 steps, superseded 2026-09-10) | 1.45 | 0.43 | 89 / 24 / 0 |
| cv04 as committed, POST-FIX (990 steps) | 121.74 | 265.47 | 115 / 115 / 0 |

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
| cv22 debye @ 3e-4 (§8.1) | dx | **6.68** | **dx** (229 of 229 bins in R) | — |
| cv04 material (settled) | dx | 0.96 | **dx** (91 bins) | — |
| cv04 as committed, PRE-FIX (719 steps, superseded 2026-09-10) | dx | 0.099 | **never** (0 of 115 bins) | — |
| cv04 as committed, POST-FIX (990 steps) | dx | 8.35 | **dx** (111 of 115 bins) | — |

**Answer to "compute which rung F2 fires at":** F2 fires at the COARSEST rung of
every ladder that runs one — dx for cv23's tand0p1 and tand1 and for cv22's
Lorentz and Drude, dx/2 for cv23's tand3 (its coarsest committed rung) — and it
goes silent at **dx/4 on tand0p1** (0 of 229 bins; on tand1 the T observable
still catches it there). It never fired on cv22's Debye arm AT ITS COMMITTED
RUNG, nor on cv04's PRE-FIX committed configuration (719 steps, superseded
2026-09-10), in both cases because the truncation term of the window exceeded
the lattice term at that record; both were §5.2/§5.3's declared
non-discriminating rungs, named in the manifest and the public rows rather
than left to be discovered. **Both are now statements about the RUNG, not the
arm or case, and both are closed (§10(4)): at §8.1's 3e-4 rung, which the
VESSL run took, F2 fires at 6.68× on 229 of 229 gated bins in R; on cv04's
POST-FIX (990-step) committed configuration, F2 fires at 8.35× on 111 of 115
gated bins in R** — both rows above.

### F3 — a 1 % ε′ perturbation

The declared lattice with the DISPERSIONLESS part of the permittivity (ε′ for
cv04/cv23, ε∞ for cv22) 1 % high.

| rung | sep/W R | bins R / T / A | fires |
|---|---|---|---|
| cv22 debye | 0.02 | 0 / 0 / 0 | **no** (§5.2) |
| cv22 debye @ 3e-4 (§8.1) | 1.07 | 107 / 178 / 72 | yes |
| cv22 lorentz | 0.50 | 12 / 0 / 0 | yes |
| cv22 drude | 14.6 | 222 / 198 / 157 | yes |
| cv23 tand0p1 / _dx2 / _dx4 | 2.67 / 2.93 / 3.03 | 209 / 212 / 212 in R | yes |
| cv23 tand1 / _dx2 / _dx4 | 0.78 / 0.87 / 1.12 | 76 / 110 / 160 in R, 216 / 219 / 229 in T | yes |
| cv23 tand3 / _dx4 | 0.52 / 0.57 | 94 / 95 in R, 229 / 229 in T | yes |
| cv04 material (settled) | 0.79 | 123 / 128 / 0 | yes |
| cv04 as committed, PRE-FIX (719 steps, superseded 2026-09-10) | 0.082 | 0 / 0 / 0 | **no** (§5.3, historical) |
| cv04 as committed, POST-FIX (990 steps) | 6.87 | 111 / 113 / 0 | **yes** (§10(4), closed 2026-09-10) |

F3 fired on every rung except the two declared non-discriminating ones — and on
the Debye arm it fires as soon as the record is long enough (1.07× at §8.1's
rung), so that pre-fix "no" was the rung's, not the arm's; cv04's own rung now
fires too, by the same logic (§10(4)). On
several arms it is carried by the per-bin gate (GL1), not the band mean — the 1 %
perturbation moves the fringe positions more than the fringe-averaged level, and
GL1 is what sees that.

### F4 — the lattice built on the CONTINUUM ε (the falsifier for this lane's one new ingredient)

**Added 2026-09-03 after the independent review, which found that the lane's one
new ingredient — carrying `ε_num` into the lattice — had no falsifier.** F1
moves the geometry, F2 replaces the whole model, F3 moves `ε′`; none of them
touches the discrete-time correction the ADE and the σ average contribute. F4
does exactly that and nothing else: the same marcher, the same geometry, the
same (dx, dt), with the slab nodes built on `dispersive_eps.eps_analytic`
instead of `dispersive_eps.eps_numerical_ade`
(`FALSIFIER_KINDS` gained `"eps_continuum"`).

| rung | sep/W R | sep/W T | sep/W A | bins over window R / T / A | fires |
|---|---|---|---|---|---|
| cv22 debye | 0.003 | 0.003 | 0.001 | 0 / 0 / 0 | **no** |
| cv22 debye @ 3e-4 (§8.1) | 0.160 | 0.279 | 0.046 | 0 / 0 / 0 | **no** |
| cv22 lorentz | 0.149 | 0.263 | 0.066 | 0 / 0 / 0 | **no** |
| cv22 drude | 0.447 | 0.280 | 0.268 | 33 / 57 / 28 | yes |
| cv23 tand0p1 | 0.017 | 0.054 | 0.043 | 0 / 0 / 0 | no |
| cv23 tand0p1_dx2 | 0.005 | 0.029 | 0.018 | 0 / 0 / 0 | no |
| cv23 tand0p1_dx4 | 0.001 | 0.013 | 0.006 | 0 / 0 / 0 | no |
| cv23 tand1 | 0.147 | 0.905 | 0.027 | 0 / 141 / 0 | yes (T) |
| cv23 tand1_dx2 | 0.041 | 0.234 | 0.007 | 0 / 0 / 0 | no |
| cv23 tand1_dx4 | 0.013 | 0.074 | 0.002 | 0 / 0 / 0 | no |
| cv23 tand3 (= dx2) | 0.102 | 0.593 | 0.100 | 0 / 46 / 0 | yes (T) |
| cv23 tand3_dx4 | 0.027 | 0.149 | 0.027 | 0 / 0 / 0 | no |
| cv04 material (settled) | **0.000** | **0.000** | **0.000** | 0 / 0 / 0 | **no, by construction** |

**Read plainly, because this is the honest reading and it is not the flattering
one.** On cv22 the ADE's discrete-time term is separately testable on the
**Drude arm only** (sep/W = 0.447 in R, 33 gated bins over the per-bin window).
On the **Debye arm the separation is 0.003 of the window and on the Lorentz arm
0.149** — three and one order(s) below the level the gate can resolve at those
records. **So at the committed cv22 rungs the lattice gate does NOT separately
test the ADE term on the Debye and Lorentz arms.** Those arms' lattice gates
still pass, and §5.2's a-priori prediction of their continuum residual still
holds; what does not hold is any claim that the gate would have caught a wrong
`ε_num` there.

On cv04's material the separation is IDENTICALLY zero: with σ = 0 and no pole,
`eps_analytic` IS `eps_num`, so there is no discrete-time correction to remove.
That row is geometry, not a gate weakness, and it is stated rather than left as
a silent pass.

**What would make it testable.** Two independent routes, neither run here:

1. **A longer record on the Debye and Lorentz arms.** F4's separation is fixed
   by the material and (dx, dt) — 5.3e-5 in the gated mean on Debye — while the
   window is not: `W_witness,R` is 1.71e-2 there and truncation-dominated.
   §8.1's `3e-4` settling bar shrank it, MEASURED on the VESSL run, to 3.32e-4
   (52×), which takes Debye's F4 ratio from 0.003 to **0.160**: still short.
   Firing on the band mean needs the tail below **1e-5** — 350 steps past the
   committed 1108,
   i.e. 181 past §8.1's rung, about 1.3× one cv22 arm — where the predicted
   window is 3.7e-5 and sep/W is 1.44. Even a perfectly settled record does not
   make it arbitrarily sharp: with the truncation term driven to zero the
   remaining incident-purity and float32 terms leave `W_witness,R = 1.24e-5`,
   sep/W = 4.3. That is the ceiling of what F4 can ever say on this arm at this
   mesh, and it is 4×, not 400×.
2. **A rung where the correction is larger.** `ε_num − ε` grows like (ω dt)²;
   the same arm at the SAME dx with a larger Courant number, or at a higher
   band, separates faster than the window shrinks. That is a new recipe, not a
   new rung of this ladder, and it is NOT proposed here.

Until one of those runs, the honest statement is the one above: the ADE term is
carried, its effect on the continuum residual is predicted a priori (§5.2), and
it is separately FALSIFIED only on Drude and on cv23's tand1 / tand3.

All four falsifiers, at every rung, are asserted in
`tests/crossval/test_lattice_witness_gates.py` `test_falsifiers_fire_exactly_where_the_note_says_they_do`
against the table `_F_FIRES`, which is this section in code; F4's two structural
properties (identically zero on the lossless slab, strictly positive on every
material that has a correction) are asserted analytically in
`test_f4_eps_continuum_isolates_the_one_ingredient_this_lane_adds`.

## 8. New rungs — pre-declared with cost, then run

The PI's rule is "add a rung only if a claim requires it, and say what it costs".
Two claims require one. **Both were pre-declared below BEFORE they ran, and both
then ran, as legs of `scripts/vessl_lattice_witness.yaml` on the 2026-09-03
VESSL run** (§9.1). The predictions are left exactly as they were written; the
outcome is appended under each. The note's first draft said the YAML had not
been submitted, which was true when the section was written and false by the
time the lane shipped; the correction is recorded rather than the sentence
quietly deleted.

### 8.1 cv22 Debye at a 3e-4 settling bar (the only rung a claim requires)

The claim that needs it: "F2 fires at the coarsest rung of every arm". It does
not, on Debye, and §5.2 says why. The remedy is resolution in TIME, not
tolerance: run the same arm with `--settling-bar 3e-4 --tag debye_tail3e4`. The
bar can only be tightened — `slab_rig.py` raises on anything outside
`(0, SETTLING_LIMIT]`, measured against the family constant and not against the
active recipe (§0's correction) — so this is not a window move.

Derivation of the cost and the prediction, from the committed artifact:
the tail must fall from
`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.tail.scat_refl_rel = 0.0072`
to 3e-4 at the fitted rate
`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.tail.fitted_rate_scat_refl_1_s = 8.8e9`
— but the budget uses the conservative
`min(fitted_scat, fitted_trans, derived) = 8.05e9 s⁻¹`, giving
`ln(7.17e-3/3e-4)/(8.05e9 · 2.335e-12) = 169` steps, i.e. **2 adaptive
extensions of 100 → n_steps ≈ 1308**. The nx-1000 CPML gate is 1262
(`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.run.record.t_safe_cpml_steps = 1262`), so the existing
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

**OUTCOME (VESSL run 2026-09-03, leg `cv22_debye_tail3e4`, rc 0). The first
branch. Every predicted direction was right; two magnitudes were not, both in
the conservative direction.**

| quantity | predicted above | measured |
|---|---|---|
| `n_steps` | ≈ 1308 (2 extensions) | `validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.n_steps = 1451` (2 extensions, and the box grew once as the rule provides) |
| tail `scat_refl_rel` | ≤ 3e-4 (the bar) | `validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.budget.scat_tail_rel = 1.42e-4` |
| `W_witness,R` | 7.3e-4 | `validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.mean_W_witness_R_gated = 0.000332` |
| `\|rfx − lattice\|`, gated mean R | ≈ 2.7e-5 | `validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.mean_dR_lattice_gated = 1.06e-5` |
| gate | passes, ~27× margin | passes; worst per-bin ratio `validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.worst_ratio_R = 0.092`, band-mean margin 31× |
| F2 on Debye | **fires at 3.05×**, 229 of 229 bins in R | **fires at 6.68×**, 229 of 229 bins in R |
| window vs a-priori ceiling | (not predicted) | `validation/crossval/_22_dispersive_diag/lattice_witness.json::rungs.debye_tail3e4.mean_W_ceiling_R_gated = 0.0164`; `W_exceeds_ceiling_R` now false |

The two magnitude misses are the same miss twice: the record ran 143 steps
longer than derived because the adaptive rule stops on the WITNESS, not on the
derivation, so the tail landed at 1.42e-4 rather than at the 3e-4 bar — and the
window and the residual both fell further than predicted in consequence. The
derivation was conservative, which is the direction a bound is allowed to be
wrong in.

**What this closes.** §5.2's stated limitation and §7's two "never" rows for the
Debye arm: at this rung the lattice gate DOES tell the lattice model from the
continuum one, by 6.68× on every gated bin. And the arm's residual against the
continuum transfer matrix is unchanged by the longer record —
`validation/crossval/_22_dispersive_diag/rfx__debye_tail3e4.json::arms.debye.mean_dR_gated = 0.0022` at 1451 steps against 0.0023 at 1108,
against the a-priori lattice prediction of 0.00222 — so the residual is the
lattice term and the truncation identification of §5.2 is confirmed rather than
assumed. **Cost, measured: 0.9 s of FDTD.**

**What it does NOT close.** F4 (§7) still does not fire here: the ADE's own
discrete-time separation is 0.160 of this window, up from 0.003 at the committed
rung but still short of 1. Firing it needs the tail near 1e-5, not 1.4e-4.

**Nothing in cv22's committed gate changes.** The leg wrote to
`validation/crossval/_22_dispersive_diag/`, not to `_22_dispersive_results/`, so
`rungs_from_results` does not see it and cv22's three committed rungs, its
verdict and its windows are exactly as they were. This rung is carried as a
committed DIAGNOSTIC with its own witness document.

### 8.2 cv04's slab on the settled rig (a re-run of a committed leg, for provenance)

§5.3's claims-bearing cv04-material rung is cv23's `tand0p1_sigma_zero` artifact.
The YAML re-runs that falsifier so the cv04 row rests on a leg of this lane too;
it must exit 1 (it is a cv23 falsifier). **Cost: one cv23 arm, ~1 s.** No new
recipe.

**OUTCOME (VESSL run 2026-09-03, leg `cv23_sigma_zero`): rc 1, as pre-declared.**
The leg re-ran the committed falsifier and it exited 1, which is what a cv23
falsifier must do; §5.3's cv04-material numbers therefore rest on a leg of this
lane as well as on the committed artifact.

### 8.3 Rungs deliberately NOT added

- **cv22 dx/2 and dx/4.** cv22 r2 ran them as diagnostics and did not commit
  them. They would make cv22's lattice claim a ladder rather than a single rung,
  and they are cheap (~4× and ~16× of a 1 s arm). They are NOT added, because no
  claim in this note requires them: the single-rung lattice gate is already a
  rigorous statement at that rung, and F2's convergence reading is supplied by
  cv23's ladder on the same rig. If the PI wants cv22's ladder, it is
  `--dx-div 2|4 --tag <arm>_dx<K>` and about 30 s of pod time.
- **A cv04 rung at the settled recipe with cv04's own script.**
  **SUPERSEDED 2026-09-10** -- see §5.3's top UPDATE. The PI overrode this
  call: cv04 now has its own record derivation (the settling-extension loop,
  `slab_family.NX_GROW_CELLS`) and a settled rung, on its own script, not
  borrowed from cv23. Kept below, struck through in spirit, because the
  reasoning explains why this bullet existed before the override, not
  because it still describes the current state.
  ~~cv04 has no record derivation and no `--nx-interior`; adding one would
  re-engineer a committed legacy configuration for no new physics (cv23's
  `sigma_zero` arm already measures that material at that recipe on that
  rig). Recorded as a deliberate non-change.~~

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
- `validation/crossval/_22_dispersive_diag/rfx__debye_tail3e4.json` and
  `validation/crossval/_22_dispersive_diag/lattice_witness.json` — §8.1's remedy
  rung, in its OWN results directory. `rungs_from_results` globs
  `_22_dispersive_results/`, so this rung is not part of cv22's committed ladder,
  its verdict or its windows; it is a committed DIAGNOSTIC with its own witness
  document, and it is what closes §5.2's stated limitation.
- `validation/crossval/_04_fresnel_results/lattice_witness.json` — one rung,
  written by `python validation/crossval/04_multilayer_fresnel.py
  --lattice-witness`. Originally committed 2026-09-03 with `gated_here =
  false` (719-step record, did not settle); regenerated by the 2026-09-10
  settling-extension fix and now carries `gated_here = true` (990-step
  settled record) -- see §9.1's second run entry below. cv04's
  `artifact_paths` lists it and §5.3's cv04 numbers are READ from it rather
  than derived.

All three committed witness JSONs are rebuilt from the committed `rfx*.json`
rungs by `test_committed_witness_artifact_rebuilds_from_the_committed_rungs`
(cv22 and cv23; cv04's single rung has no committed `rfx.json` and is rebuilt by
re-running the case), so they can never drift from the measurements they
summarize.

### 9.1 Provenance of the committed copies (VESSL run, 2026-09-03)

The three committed `lattice_witness.json` files are the ones the cluster run
`lattice-witness-20260903T044345Z` wrote, at branch commit
`9f28d0bd3898810520fae1ed5ce8abdd45135e2b`, on `remilab-c0`. **All 12 committed
rungs passed on the cluster, worst `|rfx − lattice| / W_witness` = 0.30**
(cv23 `tand3_dx4`, T).

- **cv22 / cv23.** The run's copies replace the ones this lane had generated
  locally. Every scalar leaf — every gate, verdict, budget term, gated mean and
  worst ratio — agrees to the `rel = 1e-9, abs = 1e-15` leaf tolerance the
  rebuild test uses. **Eighty-one per-bin array entries in cv23 do not**, out of
  25 071 differing leaves and ~29 000 total: they are `dR_lattice` / `dA_lattice`
  values of order 1e-6 differing by at most **9.1e-15** absolute (3e-9 relative),
  on the finest rungs. That is the same numpy/BLAS environment difference §10(8)
  records, accumulated through the marcher; it is 8 orders below the smallest
  window in §5 and changes no gate. It is recorded here rather than rounded away.
  cv22's copies agree at every leaf, including every array entry.
- **cv04.** No committed copy existed before; this is the case's first. Its
  measured arrays (`dR_lattice`, `dT_lattice`) are float32 FDTD output and differ
  from a local re-run at the 5e-6 level — the platform difference of the run
  itself, not of the lattice model, whose `R_lattice` / `T_lattice` agree to
  1.7e-15 between the pod and a local re-run. The gated means the note quotes
  (§5.3) are the pod's.
- **cv04's leg exited 2, and that is not a lattice-witness failure.** Exit 2 on
  `04_multilayer_fresnel.py` means "Meep secondary reference unavailable"
  (`ModuleNotFoundError: No module named 'meep'` on that image). The case's own
  gates PASS on that run — `T` mean error 0.0110, `R` mean error 0.0066,
  `R+T` mean deviation 0.0091 (all against 0.05), `max|R+T−1|` 0.0487 ≤ 0.06,
  both tail witnesses and the tail purity `ok`, and `rfx accuracy: PASS` — and
  the lattice witness wrote its record and reported `witness_ok = true`.
- **cv22/cv23's committed copies were written before §7's F4 existed** and
  therefore still carry `falsifiers.{thickness_plus_cell, thickness_minus_cell,
  continuum, eps_x1p01}` only. F4 (`eps_continuum`) is analytic post-processing
  of the same committed rungs; its verdict at every rung is tabulated in §7 and
  asserted by `test_falsifiers_fire_exactly_where_the_note_says_they_do`, which
  computes it from the rungs rather than reading it from the artifact. Their
  next `--lattice-witness` invocation will add the block; the addition is
  purely additive and changes no gate. **cv04's re-run (below) already carries
  it**, because F4 existed in the script by the time that run wrote its
  artifact.

**2026-09-10 run: cv04's settling-extension fix (VESSL `369367260232`).**
The settling-extension mechanism (§5.3's 2026-09-10 UPDATE) required a
fresh cv04 run: VESSL run `369367260232`, branch commit `0c66b8c1`
(`0c66b8c1d4713ac791a919876606e9b63f2e6c75`), `cv04.rc = 2` (S5, PR #974
round 2: the case's own docstring is explicit that exit 2 is "inconclusive
crossval, NOT a pass" -- rfx's self-check against the analytic reference
passed, but Meep, the secondary cross-check, was unavailable; not a failure
either), `fringe_evidence.rc = 0`. The
run's raw output is tracked, not only referenced: `commit.txt`, `run_id.txt`,
`cv04.log`, `cv04.rc`, `cv04_settled_summary.json`, `fringe_evidence.log`,
`fringe_evidence.rc` and the VESSL stdout capture are all committed at
`validation/crossval/_04_fresnel_logs/`, byte-identical to the run's own
copies (`sha256sum` on each pair, source vs. tracked copy, matches on all
eight files -- e.g. `cv04.log` both `6a819810bc5ae4b7...`,
`rfx-cv04-settling-fix_369367260232_completed.log` both
`49001ce7a4035412...`; full list in the PR body). This run wrote the current
`_04_fresnel_results/lattice_witness.json` (990 steps, `gated_here = true`)
and `fringe_gate_geometry.json`. `envelope.json` is untouched by it -- still
r1 only; a candidate r2 is drafted and held for its own PR, per the
calibration-envelope-ownership guard in `test_calibration_envelope_ownership.py`.
N6 (PR #974 round 2): r1 itself did not change, but three of its own
`lattice_identification.*.source` pointers (`rfx_vs_lattice_mean_dR_gated`,
`rfx_vs_lattice_mean_dT_gated`, `witness_gated_here`) point INTO this
mutable `lattice_witness.json`, which this run regenerated in place -- so
those three pointers now resolve to different values than when r1 was
written (0.00168 -> 2.01e-05, 0.00625 -> 9.52e-05, false -> true; full list
in the PR body). This is disclosed, not fixed, here and gets a top-level
note in the r2 PR.

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
4. **cv04's committed rung did not discriminate; cv22's Debye rung did not
   either.** CLOSED for both, 2026-09-10, by two different routes. For cv04:
   the settling-extension fix (§5.3's 2026-09-10 UPDATE, §9.1) gave it its own
   settled 990-step record, and the wrong-model falsifier F2 now fires there
   at 8.35× on 111 of 115 gated bins
   (`validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.falsifiers.continuum.separation_over_window_R
   = 8.35`) — cv23's `sigma_zero` arm is no longer load-bearing for this
   claim, though it remains a cheap corroborating cross-check on the same
   material. For Debye, §8.1's remedy rung ran on the VESSL run and F2 fires
   there at 6.68× on every gated bin, so that limitation is a property of the
   committed RUNG and is closed by a committed diagnostic; §5.2 records both.
   No claim in the manifest or the public rows rests on a non-discriminating
   rung any more.
5. **`docs/design_notes/20260903_e4_all_solver_classes_plan.md` is not on
   `origin/main`.** It lives on `origin/agent/e4-all-solvers-plan` at `eb78254`
   and was read from there (§1). Its lane L1 is the consumer of this rig; nothing
   in this note depends on it landing.
6. **No `26_*` / oblique case exists to exclude.** §1. The exclusion is stated as
   a scope boundary of the 1-D derivation, not as a reference to a case in the
   tree.
7. **The lattice verdict is not wired into the cases' exit codes.** §6. It is a
   separate invocation with its own rc. Wiring it changes what a red case means
   and should follow a green VESSL run, not precede it. The run has now happened
   and is green on all 12 rungs (§9.1), so the precondition is met and the wiring
   is a PI call, not a blocker — still NOT done in this lane.
8. **The committed cv23 `arms.<arm>.lattice.R_lattice` arrays reproduce to
   7e-15 (measured maximum 6.9e-15, on `tand3`'s R), not bit-for-bit,** when
   recomputed today. The note first said "1e-15", which the measurement does not
   support. The refactor of
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
  **Resolved: it fell, 6.5e-4 → 1.06e-5, while the arm's residual against the
  continuum transfer matrix stayed at 0.0022. The explanation stands.**
- `mean_W_witness_*` exceeding `mean_W_ceiling_*` on a rung whose fitted rate is
  NOT slower than its derived rate → a term in §3 is mis-signed.
- A measured float64-vs-float32 difference above `budget.mean_delta_round_gated`
  → §3(T3)'s statistical estimate is the wrong one and the coherent term must be
  carried, which would loosen every window in §5 by ~3×. That is a finding, not a
  window move: it would be a correction to a DERIVED term, appended here.

## 12. The 2026-09-03 independent review — every finding, and what was done

The review's verdict was FIX-THEN-SHIP. Its findings are applied above rather
than appended, per the amended append-only rule at the head of this note. Each
one, with the measurement that settled it:

**B1 — the tightness claim was wrong at both ends.** §5.1 said the lattice gate
is "20–200× tighter than the continuum window it sits beside". Recomputed per
observable over cv23's nine rungs: minimum **5.2×** (`tand0p1`, T), per-bin
floor **7.1×**, maximum **8772×** (`tand3`, T), median **21×**, and six of the
nine rungs are below 20× in at least one observable. §5.1 now reads
"5–8800× (median ≈ 20×)" and carries the full R/T/A-per-rung table, plus the
cv22 ratios, where the Debye arm's lattice window is 0.6× the continuum one —
i.e. LOOSER.

**B2 — claim (b) holds for R only.** §5.3 said cv04's committed band-mean
envelope IS this slab's Yee-lattice second-order term. Measured: the lattice
term is 0.00727 over cv04's mask in BOTH R and T (identical: the slab is
lossless), against committed envelopes of 0.0066 in R (ratio 1.10) and 0.011 in
T (0.66); and `|rfx − lattice|` at cv04's own rung is 0.00168 in R but 0.00625
in T, so more than half the T envelope is record truncation. The claim is scoped
to `|ΔR|` in §1, §5.3, the manifest's `04_multilayer_fresnel` claim_scope and
cv04's README and benchmarks rows, and `W_MEAN_T` is no longer described as a
discretisation number anywhere.

**B3 — the settling-bar guard did not do what §0 said.** It compared the
requested bar against the ACTIVE recipe's `tail_limit`, so `--recipe cv04` (a
declared flag of both cv22 and cv23) accepted `5e-2`, five times looser than
`SETTLING_LIMIT`, and `0.0` / `−1.0` passed under r3. The guard now bars the
family constant and the sign, `0 < bar ≤ SETTLING_LIMIT`; the four probes and a
legitimate tightening are asserted in
`tests/crossval/test_lattice_witness_gates.py`. §0 and §8.1 are corrected.

**M1 — the consequence, not just the premise.** §5.4, new. Seven of the twelve
committed rungs pass their continuum band-mean R gate only because
`W_mean,R = 0.010` carries cv04's lattice term, and every measured residual
equals its own lattice term to two significant figures. Recorded, with the
follow-up lane proposed and its cost; no window moved.

**M2 — the one new ingredient had no falsifier.** §7 F4 (`eps_continuum`), new:
the lattice built on the continuum ε. It fires on Drude only among cv22's arms
(sep/W 0.003 debye, 0.149 lorentz, 0.447 drude), so the ADE term is NOT
separately testable on the Debye and Lorentz arms at these rungs — said plainly
there, with what would make it testable and what it would cost.

**M3 — two reproduction tolerances were stated more strongly than measured.**
"≤ 1e-15" (§2) against a measured 1.7e-15, and "reproduce to 1e-15" (§10(8))
against a measured 6.9e-15. Both now carry the measurement. (The review's own
run measured 1.39e-15 for the first; the value here is what this environment
measures on the test's own grid, and the environment-stable statement — the one
the test asserts — is the 1e-14 class.)

**M4 — §5.3's and §7's cv04 rows were reconstructed.** They were 3–15 % off the
real run. Every one is now a key of
`validation/crossval/_04_fresnel_results/lattice_witness.json`, which this PR
commits (§9.1), and the mask is 3.032–11.867 GHz / 170 bins / 115 gated, not
"3.03–11.81 GHz, 169 bins".

**M5 — the float32 term reads as a bound and is not one.** The qualifier is
appended to the cv22 / cv23 manifest claim_scopes and to their README and
benchmarks rows.

**Artifacts.** The three `lattice_witness.json` files are the VESSL run's, cv04's
committed for the first time; §9.1 records the provenance, the one leaf class
that differs between the run and this lane's local copies (81 per-bin entries at
≤ 9.1e-15), and that cv04's exit 2 is "Meep secondary reference unavailable"
with the case's own gates passing.

**A finding the review did not raise, found while applying it.** §8 said its two
remedy rungs were pre-declared and NOT run, and that
`scripts/vessl_lattice_witness.yaml` had not been submitted. Both were false by
the time the lane shipped: the VESSL run took both legs. §8.1's Debye rung at a
3e-4 settling bar came out on the FIRST branch of its own risky prediction —
`W_witness,R` 1.71e-2 → 3.32e-4, `|rfx − lattice|` 6.5e-4 → 1.06e-5, F2 firing
at 6.68× on 229 of 229 gated bins — while the arm's residual against the
CONTINUUM transfer matrix did not move (0.0022 at 1451 steps against 0.0023 at
1108, against an a-priori lattice prediction of 0.00222). §5.2's limitation is
therefore closed and its explanation confirmed rather than assumed. §5.2, §7,
§8, §9, §10(4) and §11 are corrected, and the rung's two artifacts are committed
under `validation/crossval/_22_dispersive_diag/` — deliberately NOT under
`_22_dispersive_results/`, so cv22's committed ladder, verdict and windows are
untouched.

**Not done, and why.** No window is widened or tightened anywhere in this lane —
including the one §5.4 shows is discretisation-limited. That is a PI decision
with a compute cost, and it is proposed in §5.4 rather than taken here.

## 13. Revision 2026-09-14 — GL1's validity domain, and why the budget was not widened (#1015)

**Append-only.** §3 and §4 are unchanged; each carries a pointer to this
section. No window in this note moves and no committed gate value moves. cv22's
and cv23's `lattice_witness.json` rebuild byte-identical from their committed
records under this revision and under the commit before it; cv04 commits no
`rfx.json`, so its artifact cannot be rebuilt at all (§5.3) and is untouched.

### 13.0 What #1015 asked

cv26 is the first oblique consumer of this standard. With the corrected source
tau (§13.1), GL1 breaches on five of its seven primary arms; `tm_60` fails GL2_R
at 244 % of its window. The issue offered three options and investigated none:
carry the absorber term in the budget (A), demote GL1 to reported (B), or
declare GL1's validity domain (C). The rule for choosing was written and
committed before any of the numbers below existed:
`docs/design_notes/20260914_lattice_witness_gl1_predeclaration.md`.

### 13.1 τ is the CASE's source, not the family's

`lattice_witness.TAU_SRC_S` is `1/(π f0 bw)` at the slab family's FIXED
bandwidth 0.5. §3 reads as though that constant were a property of the standard.
It is not: `Λ = √π τ / dt` divides every budget term, the incident tail rate is
`2a/τ`, and `inc_amp_rel` is max-normalised, so a case driving a different
bandwidth gets a wrong window with nothing cancelling. #1011 gave the primitives
an explicit `tau_s` (the family default is unchanged, so cv04, cv22 and cv23 are
untouched) and cv26 passes the tau its own record carries. §3 should be read
with that: **τ is the case's.**

### 13.2 The measurement

Replay only, on records already committed; no FDTD was run for this section.

### T1 — the slab family (cv04, cv22, cv23): every committed rung

| case | rung | mean `W_R` | mean `\|ΔR−lat\|` | GL1_R beyond | GL1_T beyond | GL2_R/T | unmodelled term | #888 echo arrival ÷ record |
|---|---|---|---|---|---|---|---|---|
| cv04 | `slab_eps4` | 6.3463e-04 | 2.012e-05 | 0 | 0 | pass/pass | 0 by construction | 1.64× |
| cv22 | `debye` | 1.7129e-02 | 6.544e-04 | 0 | 0 | pass/pass | 0 by construction | 1.85× |
| cv22 | `lorentz` | 2.5801e-03 | 2.535e-04 | 0 | 0 | pass/pass | 0 by construction | 1.67× |
| cv22 | `drude` | 8.2457e-05 | 7.147e-06 | 0 | 0 | pass/pass | 0 by construction | 1.76× |
| cv23 | `tand0p1` | 1.2800e-03 | 2.957e-05 | 0 | 0 | pass/pass | 0 by construction | 1.92× |
| cv23 | `tand1` | 8.5611e-04 | 2.050e-05 | 0 | 0 | pass/pass | 0 by construction | 1.77× |
| cv23 | `tand3` | 4.7118e-04 | 3.039e-05 | 0 | 0 | pass/pass | 0 by construction | 1.71× |
| cv23 | `tand0p1_dx2` | 1.1293e-03 | 2.566e-05 | 0 | 0 | pass/pass | 0 by construction | 1.89× |
| cv23 | `tand0p1_dx4` | 1.0835e-03 | 2.481e-05 | 0 | 0 | pass/pass | 0 by construction | 1.88× |
| cv23 | `tand1_dx2` | 7.4952e-04 | 2.037e-05 | 0 | 0 | pass/pass | 0 by construction | 1.75× |
| cv23 | `tand1_dx4` | 5.7528e-04 | 1.970e-05 | 0 | 0 | pass/pass | 0 by construction | 1.73× |
| cv23 | `tand3_dx2` | 4.7118e-04 | 3.039e-05 | 0 | 0 | pass/pass | 0 by construction | 1.71× |
| cv23 | `tand3_dx4` | 4.3393e-04 | 2.921e-05 | 0 | 0 | pass/pass | 0 by construction | 1.70× |

### T2 — cv26: all twelve entries

| arm | gated bins | mean `W_R` | mean `\|ΔR−lat\|` | GL1_R beyond | GL1_T beyond | GL2_R | GL2_T | absorber term (R, gated max) | ÷ `W_R` | domain (R) | breaches in / out of domain (R) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| te_00 | 290 | 5.3944e-03 | 3.336e-04 | 0 | 0 | pass | pass | 3.7188e-04 | 0.07× | 100.0 % | 0 / 0 |
| te_30 | 441 | 6.4805e-04 | 1.413e-04 | 18 | 38 | pass | pass | 6.1395e-04 | 0.95× | 100.0 % | 18 / 0 |
| te_45 | 517 | 5.0540e-04 | 1.671e-04 | 75 | 36 | pass | pass | 4.1961e-04 | 0.83× | 66.0 % | 16 / 59 |
| te_60 | 473 | 5.6726e-04 | 4.052e-04 | 98 | 73 | pass | pass | 4.0328e-03 | 7.11× | 45.2 % | 8 / 90 |
| tm_00 | 290 | 5.4672e-03 | 3.410e-04 | 0 | 0 | pass | pass | 2.0796e-04 | 0.04× | 100.0 % | 0 / 0 |
| tm_45 | 517 | 1.0097e-04 | 4.351e-05 | 190 | 45 | pass | pass | 1.5299e-04 | 1.52× | 40.8 % | 13 / 177 |
| tm_60 | 473 | 2.0705e-05 | 5.060e-05 | 357 | 98 | **FAIL** | pass | 5.0478e-04 | 24.38× | 8.9 % | 14 / 343 |
| graze_vac | — | window undefined (no ring-down rate) | — | — | — | — | — | — | — | — | — |
| graze_pec | — | window undefined (no ring-down rate) | — | — | — | — | — | — | — | — | — |
| graze_te | 69 | 6.3482e-04 | 2.035e-04 | 0 | 17 | pass | pass | 1.0953e-01 | 172.54× | 0.0 % | 0 / 0 |
| te_00__settle60 | 290 | 1.0491e-04 | 4.547e-05 | 70 | 50 | pass | pass | 3.7188e-04 | 3.54× | 76.6 % | 4 / 66 |
| tm_00__settle60 | 290 | 1.0604e-04 | 4.168e-05 | 71 | 55 | pass | pass | 2.0796e-04 | 1.96× | 75.5 % | 5 / 66 |

### T3 — option (A) measured: window before and after, per cv26 arm

| arm | `W_R` now | `W_R` with the term | ×  | `W_T` × | GL1_R beyond now | after | GL1_T after | GL2_R now | after |
|---|---|---|---|---|---|---|---|---|---|
| te_00 | 5.3944e-03 | 5.4386e-03 | **1.01×** | 1.01× | 0 | 0 | 0 | pass | pass |
| te_30 | 6.4805e-04 | 7.1859e-04 | **1.11×** | 1.27× | 18 | 0 | 0 | pass | pass |
| te_45 | 5.0540e-04 | 6.7866e-04 | **1.34×** | 1.67× | 75 | 0 | 0 | pass | pass |
| te_60 | 5.6726e-04 | 1.4584e-03 | **2.57×** | 2.38× | 98 | 6 | 6 | pass | pass |
| tm_00 | 5.4672e-03 | 5.5082e-03 | **1.01×** | 1.01× | 0 | 0 | 0 | pass | pass |
| tm_45 | 1.0097e-04 | 1.6162e-04 | **1.60×** | 1.65× | 190 | 2 | 0 | pass | pass |
| tm_60 | 2.0705e-05 | 1.2588e-04 | **6.08×** | 3.10× | 357 | 41 | 6 | **FAIL** | pass |
| graze_te | 6.3482e-04 | 9.6612e-02 | **152.19×** | 787.87× | 0 | 0 | 0 | pass | pass |
| te_00__settle60 | 1.0491e-04 | 1.4916e-04 | **1.42×** | 1.48× | 70 | 0 | 0 | pass | pass |
| tm_00__settle60 | 1.0604e-04 | 1.4707e-04 | **1.39×** | 1.43× | 71 | 0 | 0 | pass | pass |

### 13.3 The three options, measured against the rule

**(A) carry the absorber term in the budget — REJECTED, on three of its five
pre-declared conditions.**

- **A2 fails.** Adding the term per bin flips `tm_60`'s reported `GL2_R`
  failure to a pass (T3). The pre-declaration named that as an outright
  rejection: a budget change that erases a reported failure absorbs a finding
  instead of reporting it.
- **A4 fails.** Even with the term carried, GL1 still breaches — 41 R bins on
  `tm_60`, 6 on `te_60`, 2 on `tm_45`, and 6 T bins on each of `te_60` and
  `tm_60`. The widening does not close the per-bin bound; it only moves it.
- **A5 fails.** The term has no a-priori bar. `SETTLING_BAR` and `PURITY_BAR`
  let `ceiling_windows` bound a passing run's window before the run; the
  absorber term's only available size is the measured one, and §3's claim 1 —
  "none of them is the residual being gated" — does not survive adding a
  quantity whose correlation with the residual is the very observation that
  opened #1015.
- The cost, for the record: the window widens 1.01× to **6.08×** in R on the
  primary arms and **152×** in R / **788×** in T on `graze_te` (T3). That is a
  gate loosening of up to three orders and it is tabled, not hidden.

**(B) demote GL1 to reported — REJECTED, because GL1 catches what GL2 does
not.** Over the 65 falsifier × rung entries of cv04, cv22 and cv23 (195 rows
counting R, T and A separately; cv22's and cv23's replayed from their committed
records, because their committed `lattice_witness.json` predates F4 and carries
no `eps_continuum` block — where it does carry one the two agree on every row):

- GL1 fires while GL2 passes on **38 of 195** rows;
- GL2 never fires while GL1 passes — **0 of 195**;
- **six falsifiers of 65 are caught by GL1 ALONE**, and demoting GL1 turns all
  six silent: cv22 `lorentz` / `eps_x1p01` (F3, the 1 % ε′ defect, 12 of 229 R
  bins), cv23 `tand0p1_dx2` / `continuum` (F2, the deliberately wrong model, 59
  R and 68 T bins), and **every F4 that fires anywhere** — `eps_continuum` on
  cv22 `drude` and on cv23 `tand1`, `tand3` and `tand3_dx2`. F4 is the
  falsifier for the one ingredient this lane adds (carrying `ε_num` into the
  lattice, §7); demoting GL1 retires it completely. All six are pre-declared to
  FIRE in the `_F_FIRES` table of
  `tests/crossval/test_lattice_witness_gates.py`, so six rows of that table
  would have to be flipped to `False`. That is a measurable loss of the
  standard's own detection power.

**(C) declare GL1's validity domain — TAKEN.** §3 bounds three terms and
DECLARES three zero by construction. Two of the three declared-zero terms are
absorber echoes, and both are put outside the record by ARRIVAL —
`precond_cpml_gate` for the 3-D CPML, `precond_aux_echo_record` for the
auxiliary grid (#888). The family's reference, the infinite lattice of
`dispersive_eps`, has no absorber in it at all, so there is nothing for that
declaration to be wrong about: T1 shows both preconditions holding at all
thirteen committed rungs with the echo arriving at 1.64×–1.92× the record.

A rig that admits its echo INSIDE the record by AMPLITUDE instead has no such
declaration to stand on. cv26 is exactly that: `e_absorber` / `absorber_ok`
replaced the arrival cap, and its reference carries the realized absorbers. The
size of what §3 then omits is the reference's OWN dependence on them,

    U(f) = | X_ref(f; the realized absorbers) − X_ref(f; outgoing-wave
             termination) |,        X ∈ {R, T}

and GL1 is defined on the bins where

    U(f) ≤ W_witness(f)

— the omitted term must sit inside the window that omits it. Threshold 1, fixed
in the pre-declaration and not moved afterwards.

### 13.4 What the domain does and does not do — stated with the residue

Across cv26's ten entries with a defined window, **1156 of 1291 breaches
(89.5 %) fall outside the domain** and **135 (1.85 % of the 7300 gated R+T
bins) fall inside it** (T2). The two arms with zero breaches, `te_00` and
`tm_00`, have a total domain; `tm_60`, whose GL2 fails, has 8.9 % of its R bins
inside it.

**The predicate is NECESSARY and not SUFFICIENT, and this note says so rather
than rounding it off.** It states where the window is not a valid bound; it does
not promise GL1 holds where it is. The pre-declaration's condition C2 asked the
domain to put every breaching arm outside itself, and the data does not give
that: `te_30`'s R domain is the WHOLE band (`U/W ≤ 0.875` at every bin), so all
18 of its R breaches sit inside it, and 25 of its 38 T breaches do too; 16 of
`te_45`'s R breaches and 13 of `tm_45`'s are likewise in-domain. That
is the pre-declaration disagreeing with the measurement, reported as such. The
threshold was NOT moved to 0.87 to swallow `te_30`; moving it would be fitting
the predicate to the breach counts, which is what §3's claim 1 forbids.

`graze_te` — the compact box whose record is BUILT to contain the echo, at
`U/W = 172×` with zero breaches — is not a counter-example to (C) and is one to
any reading of the ratio as a prediction. Its domain is empty: the window says
nothing there, and the arm is judged on G7.

### 13.5 What changed, and what did not

- `lattice_witness.witness_domain(W, unmodelled_term=None)` and
  `lattice_witness.domain_report(...)`: the predicate and its bookkeeping.
  **The default is the family's**: `None` means zero by construction.
- `lattice_witness.evaluate(..., unmodelled_term=None)`: GL1 is judged on the
  domain. With the default the domain is every gated bin, the output keys are
  unchanged, and the artifact is byte-identical — verified by rebuilding cv22's
  and cv23's `lattice_witness.json` under this revision and under the commit
  before it, in one process, and comparing the serialized text.
- cv26 reports `domain_R` / `domain_T` per arm and an updated
  `GL1_not_gated_reason`. **`GL1_gated` stays `false`** — the PI decision of
  2026-09-14 (cv26 close note §10.4) is unchanged, and the 135 in-domain
  breaches are why: declaring the domain does not make cv26's GL1 green.
- **Nothing numeric moved anywhere.** Every number the cv26 replay artifact
  already carried is bit-identical after regeneration; the diff is 33 added
  keys and one rewritten reason string.

### 13.6 What would refute this revision

- A slab-family rung whose `precond_cpml_gate` or `precond_aux_echo_record`
  fails: the "zero by construction" that makes the family's domain total would
  then be a measurement, not a construction, and T1's last column would have to
  be replaced by an amplitude.
- A case with `U(f) ≤ W(f)` everywhere whose GL1 still breaches broadly: the
  domain would be necessary, sufficient for nothing, and too weak to be worth
  stating. `te_30` is one arm's worth of that evidence already; a second
  independent one would close the argument against this section.
- A derivation that bounds `U` a priori, from the absorber's declared
  reflectivity and the record length, the way `SETTLING_BAR` bounds the
  truncation term. That would make option (A) available on A5 and this section
  would be superseded by it — but not by a measured term alone.

### 13.7 R2 / R3

R2: attempt 1 on "what does GL1's window omit". The adjacent attempt — carrying
the exact second-order term `(δ_scat + δ_round)²` — is recorded FALSIFIED in the
cv26 close note §10.4 (58 → 54, 110 → 109) and was not retried.

R3: `memory=rfx-known-issues.md comparator-bug case ledger (13/13) + cv26 close
note §10.4–§10.5 | R2-attempts=1 | falsifier=cv22's and cv23's
lattice_witness.json rebuilt byte-identical between origin/main's comparator and
this one (PASS, 260748 and 784777 bytes)`.
