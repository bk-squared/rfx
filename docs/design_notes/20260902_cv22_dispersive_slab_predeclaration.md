# cv22 dispersive-slab Fresnel — pre-declaration (gap lane 1, dispersive materials)

Date: 2026-09-02 · Lane: `agent/gap1-dispersive-slab` · Case:
`validation/crossval/22_dispersive_slab_fresnel.py` (new; id
`22_dispersive_slab_fresnel`; claims-bearing, E2 + E4).

**Append-only.** Corrections are added as new sections; nothing above a
correction is edited. Every number in this note is ANALYTIC (closed-form ε(f),
transfer matrix, discrete-time transfer functions of the ADE recurrences) or
READ from a committed cv04 artifact. No FDTD has been run for this case at the
time of this commit; measured numbers arrive later, by artifact key.

## 0. Why this case exists

Debye (`rfx/materials/debye.py`), Lorentz and Drude (`rfx/materials/lorentz.py`)
are documented, user-reachable (`docs/public/guide/materials-geometry.mdx`,
"Dispersive materials") and have **no crossval entry and no external-solver
evidence**. The only physics evidence is
`tests/test_dispersive_fresnel_validation.py` (slow_physics, E2, band-MEAN
`|R_fdtd - R_analytic| < 0.05` for one Debye and one Lorentz slab; no T, no
Drude, no per-bin gate, no external solver) and the E0/E1 unit tests
`tests/test_debye.py`, `tests/test_lorentz.py`. cv08 (material dispersion) was
removed and never replaced. This case promotes that physics onto the cv04
rig, where the rig's own discretization error is a committed number, and adds
Meep as the E4 leg.

## 1. Rig — identical to cv04 by construction

`validation/crossval/04_multilayer_fresnel.py`, PART 1, verbatim: 2-D TMz
`Grid(freq_max=20e9, domain=(0.600, 0.004, 1e-3), dx=1e-3, cpml_layers=20)`
(shape 641 x 45 x 1), `dt = 2.335067793382187e-12 s` (Courant 0.700 on dx),
TFSF `+x`, `polarization="ez"`, `f0=10e9`, `bandwidth=0.5`, default waveform
`differentiated_gaussian` (τ = 1/(π·f0·bw) = 63.7 ps; amplitude spectrum
∝ f·exp(−(πfτ)²), peak at 3.5 GHz, 8.6 % of peak at 10 GHz, 2 % at 11.9 GHz),
slab `d = 10 mm` (cells [315, 325)) at the domain centre, probes 30 cells
either side (cells 285 / 355), 1-D auxiliary incident reference at the same
x, time-gate `n_steps = 719` (CPML round-trip rule, 0.95 factor), rFFT with
`nfft = 8192` (`df = 52.28 MHz`), cv04's mask (`3 GHz < f < 15 GHz` and
incident amplitude `> 2 %` of peak). The settling-tail witnesses
(`TAIL_WINDOW = 50`, `TAIL_PURITY_LIMIT = 1e-3`, `TAIL_LIMIT = 0.10`,
`04_multilayer_fresnel.py:208-210`) are carried unchanged.

The ONLY change from cv04 is the slab material: `eps_r` in the slab is ε∞ and
the E-update in the loop is `update_e_debye` (Debye arm) or
`update_e_lorentz` (Lorentz, Drude arms) with a slab-only mask, in place of
`update_e`. Everything else (H update, CPML, TFSF, probes, FFT, mask, tail
witnesses) is byte-identical.

Three material arms, one run each (three FDTD runs of ~12 s CPU each — cv04's
committed `runtime_seconds = 12.392` at
`docs/public/gallery/assets/multilayer_fresnel/manifest.json::provenance.runtime_seconds`).

## 2. Material arms — parameters and why

rfx convention throughout: `e^{+jωt}`, `Im ε < 0` for loss (the ADE equations
in `debye.py` / `lorentz.py` fix this; see §3).

| arm | ε(ω) (rfx convention) | parameters | source constructor |
|---|---|---|---|
| (a) Debye | `ε∞ + Δε / (1 + jωτ)` | ε∞ = 2.0, Δε = 4.0, τ = 1/(2π·5 GHz) = 31.83 ps | `DebyePole(delta_eps=4.0, tau=τ)` |
| (b) Lorentz | `ε∞ + Δε ω0² / (ω0² − ω² + 2jδω)` | ε∞ = 2.0, Δε = 1.5, f0 = 7 GHz, δ = ω0/6 (Q = ω0/2δ = 3) | `lorentz_pole(1.5, 2π·7e9, ω0/6)` (`lorentz.py:71`, κ = Δε ω0²) |
| (c) Drude | `ε∞ − ωp² / (ω² − jγω)` | ε∞ = 3.0, fp = 7 GHz (ωp = 2π fp), γ = 2π·3 GHz | `drude_pole(ωp, γ)` (`lorentz.py:56`, δ = γ/2, κ = ωp²) |

Analytic properties across the gated band 4–10 GHz (§5), from the closed
forms above:

| arm | \|ε\| range | \|ε\| variation | tan δ = −Im ε/Re ε | R_TMM range | T_TMM range |
|---|---|---|---|---|---|
| Debye | 3.23 – 4.84 | 33 % | 0.44 – 0.58 | 0.076 – 0.284 | 0.126 – 0.322 |
| Lorentz | 0.96 – 5.57 | 83 % | 0.15 – 31.7 (Re ε crosses ~0 at 8 GHz) | 0.035 – 0.310 | 0.020 – 0.523 |
| Drude | 1.73 – 2.55 | 32 % | 1.37 → 0.05 (falls through 1 → 0.1 across the band) | 0.002 – 0.109 | 0.370 – 0.813 |

All three meet the "strong dispersion" requirement (|ε| varies ≥ 30 %; loss
tangent inside or passing through 0.1–1). They were chosen so that a wrong
pole is unmistakable (§6) while T stays ≥ 0.12 (Debye), ≥ 0.37 (Drude) so the
T gate is not a measurement of noise; the Lorentz arm's T dips to 0.02 at
resonance, so near 7–8 GHz its T gate is weak and its R gate (0.22) carries
the discrimination — stated here, not discovered later.

**Drude docstring discrepancy (found while reading, not fixed here):**
`rfx/materials/lorentz.py:5-6` writes the Drude form as
`ε∞ − ωp²/(ω² + jγω)`, which is the `e^{−iωt}` sign. The ADE it discretizes
(`d²P/dt² + 2δ dP/dt + ω0² P = ε0 κ E`, with `P ∝ e^{+jωt}`) gives
`κ/(ω0² − ω² + 2jδω)` and hence, at ω0 = 0, κ = ωp², δ = γ/2:
`ε∞ − ωp²/(ω² − jγω)`. The comparator uses the ADE-implied form (the one the
update actually realizes), which has `Im ε < 0` consistently with the Debye
and Lorentz forms. Also: `rfx.material_fit.eval_lorentz` sets
`delta_eps = 0.0` when `omega_0 == 0` (`material_fit.py:495`), i.e. it
**silently drops a Drude pole** and returns ε∞ — so it cannot be the Drude
oracle; the comparator's Drude closed form is used and this limitation is
pinned by a strict-xfail test.

### 2.1 Stability and accuracy constraints checked (rfx side)

- Debye, Crank–Nicolson, `debye.py:133-134`:
  `α = (2τ − dt)/(2τ + dt) = 0.92924`, `|α| < 1` for all τ > 0 →
  unconditionally stable. Accuracy set by ω·dt (§3).
- Lorentz/Drude, explicit central difference + CN damping,
  `lorentz.py:154-156`: characteristic polynomial of the P recurrence
  `(1 + δdt) z² − (2 − ω0²dt²) z + (1 − δdt) = 0`; Jury conditions give
  stability iff `ω0·dt < 2` and `δ ≥ 0`. Lorentz: `ω0·dt = 0.1027`,
  `δ·dt = 0.01712`. Drude: `ω0 = 0`, `δ·dt = γdt/2 = 0.02201`. Both far
  inside.
- Coupled E–P stability inherits cv04's Courant 0.700 with ε∞ ≥ 1 in the
  slab (ε∞ = 2, 2, 3 only raises the local limit).

## 3. The new physics term: temporal discretization of ε(ω) by the ADE

This is the term cv04 does not have and must be named. Derived by
z-transforming the recurrences in `debye.py` and `lorentz.py` with
`z = e^{jω dt}`.

**Debye** (`P^{n+1} = αP^n + β(E^{n+1} + E^n)`, `debye.py:237`):
`P/E = β(z+1)/(z−α)`; substituting α, β and `(z−1)/(z+1) = j tan(ωdt/2)`:

    χ_num(ω) = Δε / (1 + jω̃τ),   ω̃ = (2/dt)·tan(ωdt/2)          (bilinear)

The E-update (`debye.py:229`) is algebraically
`ε0ε∞(E^{n+1}−E^n)/dt + Σ(P^{n+1}−P^n)/dt + σ(E^{n+1}+E^n)/2 = curl H^{n+1/2}`
(verified by substituting `γ, Ca, Cb, Cc` from `debye.py:159-169`), so the
D/E ratio the Yee scheme sees is exactly `ε∞ + χ_num`; the leading factor
`(z−1)/dt · z^{−1/2} = 2j sin(ωdt/2)/dt` is the ordinary Yee temporal factor
common to vacuum and is already inside cv04's envelope.
Warp at the band top: `tan(x)/x − 1 = +1.80e-3` at 10 GHz (`x = ωdt/2`).

**Lorentz / Drude** (`P^{n+1} = aP^n + bP^{n−1} + cE^n`, `lorentz.py:249`;
`E^{n+1} = CaE^n + Cb curl − Cc ΣΔP`, `lorentz.py:262`):
`P/E = c z/(z² − az − b) = c/(z − a − b/z)`; substituting a, b, c:

    χ_num(ω) = κ / (ω0² − ω̃² + 2jδ ω̂),
    ω̃ = (2/dt)·sin(ωdt/2),   ω̂ = sin(ωdt)/dt

(Drude: ω0 = 0.) Warp at 10 GHz: `(sin x / x)² − 1 = −1.79e-3`,
`sin(2x)/(2x) − 1 = −3.6e-3`.

The ADE window term is the exact propagation of `ε_num − ε` through the same
transfer matrix, per bin:

    W_ADE,R(f) = |R_TMM(ε_num(f)) − R_TMM(ε(f))|,  W_ADE,T(f) likewise.

Evaluated on the rfx bin grid (nfft 8192, 115 bins in 4–10 GHz):

| arm | max W_ADE,R | max W_ADE,T | mean W_ADE,R | mean W_ADE,T | at 10 GHz (R, T) |
|---|---|---|---|---|---|
| Debye | 1.9e-4 | 2.0e-4 | 5.1e-5 | 6.8e-5 | 1.9e-4, 2.0e-4 |
| Lorentz | 9.9e-4 | 1.8e-3 | 3.8e-4 | 5.6e-4 | 4.8e-4, 1.8e-3 |
| Drude | 7.7e-5 | 1.1e-4 | 3.6e-5 | 8.3e-5 | 5.2e-5, 1.1e-5 |

At cv04's dt the ADE term is 1–2 orders below the rig term (§4). It is named
and carried anyway: the same window formula applied at a coarser dt (or a
higher band) would not be negligible, and the falsifiability of the window
must not depend on that accident. The formula is also witnessed numerically
(the recurrences driven with the live `init_debye` / `init_lorentz`
coefficient arrays reproduce χ_num) in `tests/test_cv22_dispersive_slab_gates.py`.

The identical derivation applies to Meep's Lorentzian/Drude update
(`(1 + γdt/2) z² − (2 − ω0²dt²) z + (1 − γdt/2)`, same polynomial with
γ = 2δ) at Meep's `dt_meep = 0.5·dx/c = 1.668e-12 s`; `W_ADE,meep` is
computed from the `dt` recorded in the Meep JSON with the same function.

## 4. Windows — derived, not chosen

Repo rule (`tests/_gate_policy.py::gate_from_envelope`): gate =
round-up(committed envelope × `ENVELOPE_GATE_MULTIPLIER` = 1.5), quantum
1000 (three decimals).

Committed cv04 envelope on this rig (the same nx = 600 / 719-step
configuration — `docs/public/gallery/assets/multilayer_fresnel/manifest.json::provenance.params.{nx_interior,n_steps} = 600, 719`):

- band-mean `|ΔR|` vs TMM:
  `tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[id=mean_reflectance_error].observed_baseline = 0.0066`
- band-mean `|ΔT|` vs TMM:
  `…::expected_metrics[id=mean_transmittance_error].observed_baseline = 0.011`
- per-bin `max|R+T−1|` = **0.0487** — the only committed PER-BIN number on
  this rig; it lives in a code comment
  (`04_multilayer_fresnel.py:309`, rung C4, job 369367246779), not in an
  artifact, and is stated as such. cv04 attributes it entirely to order-2
  etalon-echo truncation by the 719-step time gate; a lossy slab (all three
  arms, tan δ ≥ 0.05) damps that echo, so adopting it unchanged is
  conservative. It is taken as the per-bin envelope for `|ΔR|` and `|ΔT|`
  individually (the echo perturbs R and T through one mechanism; no
  committed per-bin `|ΔR|`, `|ΔT|` exists to do better).

Derived rig terms:

    W_bin  = gate_from_envelope(0.0487, quantum=1000) = 0.074
    W_mean,R = gate_from_envelope(0.0066, quantum=1000) = 0.010
    W_mean,T = gate_from_envelope(0.011,  quantum=1000) = 0.017

### E2 gates (rfx vs analytic TMM with the continuous complex ε(f)), per arm

- **G1 per-bin**, every gated bin f:
  `|R_rfx(f) − R_TMM(f)| ≤ W_bin + W_ADE,R(f)`, and the same for T.
- **G2 band-mean** over the gated bins:
  `mean|ΔR| ≤ W_mean,R + mean W_ADE,R`, `mean|ΔT| ≤ W_mean,T + mean W_ADE,T`.
  Numerically: Debye 0.01005 / 0.01707; Lorentz 0.01038 / 0.01756;
  Drude 0.01004 / 0.01708.
- **G3 witnesses** (cv04's, unchanged constants): tail purity and tail
  levels (`TAIL_*`), and passivity `R + T ≤ 1 + CONS_MAX_LIMIT` (0.06)
  at every masked bin (a lossy slab must not gain).

### E4 gates (Meep present), per arm

- **G4 reference soundness**, Meep vs TMM: per-bin
  `|R_meep − R_TMM| ≤ W_bin + W_ADE,meep,R(f) + W_map,R(f)` (T likewise);
  band-mean `≤ W_mean + mean(W_ADE,meep) + mean(W_map)`. `W_map` is the
  Debye→overdamped-Lorentz mapping residual (§7; zero for Lorentz, Drude).
  Meep at the same dx has the same spatial-discretization class as rfx and
  no time-gate truncation (flux integrated until decay), so cv04's rig
  envelope bounds it from above; this is stated, not measured.
- **G5 rfx vs Meep** (triangle inequality over G1 and G4): per-bin
  `|R_rfx − R_meep| ≤ 2·W_bin + W_ADE,R + W_ADE,meep,R + W_map,R`
  (= 0.148 + small); band-mean
  `≤ 2·W_mean + mean terms` (R: 0.020 + …, T: 0.034 + …).

Exit contract (cv04's): 0 = G1–G5 pass for all three arms; 1 = any gate
fails; 2 = G1–G3 pass but a Meep JSON is missing for any arm (inconclusive).

## 5. Gated bins and why the edges are excluded

Gated band: rfx rFFT bins with **4.0 GHz ≤ f ≤ 10.0 GHz** (115 bins,
4.025–9.985 GHz). The rest of cv04's masked band (3–4 and 10–11.9 GHz) is
REPORTED, not gated:

- above 10 GHz the differentiated-Gaussian incident amplitude is < 8.6 % of
  peak (power < 0.75 %); ratios there are the mask-amplified class cv04
  documents at its 11.87 GHz worst bin (issue #341) — cv04's per-bin
  envelope was SET by that edge, so gating it would gate the rig, not the
  material;
- below 4 GHz the band abuts cv04's hard 3 GHz mask edge, where the
  719-step record (`1/T = 0.60 GHz`) resolves fewer than two bins per
  record-length and the low-frequency CPML/TFSF behaviour is uncharacterised
  on this rig.

The script asserts at runtime that the incident amplitude is ≥ 5 % of peak
over the whole gated band (rig sanity; a failure here is a rig defect, exit 1,
not a material verdict) and that the Meep flux band covers it.

## 6. Falsifiers — pre-declared, with analytic margins

Each falsifier is a `--falsifier <name>` arm of the case script (rfx side)
or of the Meep leg, run on VESSL, and MUST exit 1. The margins below are the
analytic `|ΔR|`, `|ΔT|` between the defective and the declared ε(f), through
the same TMM, on the gated bins — the smallest effect the gate must resolve.
"bins > W_bin" counts bins where the per-bin G1 must fail on its own.

**F1 — wrong pole position / relaxation time.**

| name | defect | mean\|ΔR\| / W_mean,R | mean\|ΔT\| / W_mean,T | bins > W_bin (R, T) | named bins |
|---|---|---|---|---|---|
| `debye_tau_x2` | τ → 2τ | 0.026 / 0.010 (2.6×) | 0.087 / 0.017 (5.1×) | 0, 72 | T fails per-bin at every bin ≥ ~7 GHz (0.133 at 10 GHz); G2 fails for R and T |
| `lorentz_f0_x1p3` | f0 → 1.3 f0 | 0.091 (9×) | 0.199 (12×) | 55, 98 | R 0.264 at 10 GHz; T 0.418 at 6.3 GHz |
| `drude_fp_x1p3` | fp → 1.3 fp | 0.031 (3×) | 0.167 (10×) | 17, 115 | R 0.143 at 4 GHz; T > W_bin at every gated bin |

Debye τ × 1.3 was evaluated and REJECTED as a falsifier: mean|ΔR| = 0.0105
sits on the 0.010 window (1.05×) — a pass/fail there would be a coin toss,
which is exactly what a falsifier must not be. ×2 is the declared amount.

**F2 — dispersion removed** (`delta_eps = 0`, i.e. a dispersionless ε∞ slab):

| name | mean\|ΔR\| | mean\|ΔT\| | bins > W_bin (R, T) |
|---|---|---|---|
| `debye_deps_zero` | 0.053 | 0.716 | 28, 115 |
| `lorentz_deps_zero` | 0.109 | 0.733 | 77, 115 |
| `drude_wp_zero` | 0.069 | 0.249 | 44, 115 |

**F3 — Meep material mapped with the WRONG convention** (the round-1 failure
class of this repo). Run on the Lorentz arm of the Meep leg
(`scripts/crossval/meep_cv22_dispersive_slab.py --falsifier …`); the leg's
1e-9 pre-run ε check is *recorded as failed* and the run proceeds, so the
E4 gates (G4, G5) are exercised on real Meep output:

| name | defect | mean\|ΔR\| / (2·W_mean,R) | mean\|ΔT\| / (2·W_mean,T) | named bins beyond 2·W_bin = 0.148 |
|---|---|---|---|---|
| `meep_lorentz_no_2pi` | `frequency = ω_n·a/c` instead of `ω_n/(2π)·a/c` | 0.114 (5.7×) | 0.735 (22×) | R 0.232 at 4 GHz; T 0.910 at 7.8 GHz (T > W_bin at all 115 bins) |
| `meep_lorentz_gamma_half` | `gamma` from δ instead of 2δ (rfx's `2jδω` vs Meep's `iωγ`) | 0.090 (4.5×) | 0.056 (1.6×) | R 0.203 at 7.8 GHz |

Unit-level F3 (no FDTD): `tests/test_cv22_dispersive_eps_mapping.py` asserts
the 1e-9 mapping test FAILS when the 2π is dropped, when σ is scaled by ω_n²
(Meep units), when γ is halved, and when the `e^{−iωt}` conjugation is
dropped.

## 7. Meep mapping (summary; the module is the authority)

Meep (`e^{−iωt}`, frequencies in units of c/a, a = 1 cm as in cv04):
`LorentzianSusceptibility(frequency=f_n, gamma=g_n, sigma=σ)` realises
`ε∞ + σ ω_n² / (ω_n² − ω² − iωγ_n)` with `ω_n = 2π f_n`, `γ_n = 2π g_n`;
`DrudeSusceptibility` realises `ε∞ − σ ω_n² / (ω² + iωγ_n)`.

- Lorentz: `f_n = f0·a/c`, `g_n = 2δ/(2π)·a/c = δ/π · a/c`, `σ = Δε`.
- Drude: `f_n = fp·a/c` (so `σ ω_n² = ωp²` with `σ = 1`), `g_n = γ/(2π)·a/c`.
- Debye has NO native Meep susceptibility and is NOT the ω_n → 0 limit of a
  Lorentzian (that limit is Drude). It is the **overdamped** limit: with
  `σ = Δε`, `γ_n = ω_n²τ`, the Lorentzian is `Δε / (1 − ω²/ω_n² − iωτ)`,
  i.e. Debye plus a residual of relative size `(ω/ω_n)²/|1 − iωτ|`. Declared
  `f_n = 100 GHz` (`ω_n·dt_meep = 1.048 < 2`, Meep-stable by the same Jury
  criterion as §2.1; `γ_n·dt_meep = 21`), giving a max relative ε residual of
  **2.5e-3** in 4–10 GHz and, through the TMM,
  `max W_map,R = 3.2e-4`, `max W_map,T = 1.1e-3`. `W_map(f)` is computed per
  bin from the mapped ε and enters G4/G5 for the Debye arm only.
- Comparison is always `conj(ε_meep(ω)) == ε_rfx(ω)`.

The Meep leg evaluates the mapped ε at three band frequencies — through
`Medium.epsilon(f)` when the installed Meep exposes it, else through the
reconstruction formula — against `eps_analytic` (Debye: against the mapped
overdamped-Lorentz target, with the residual vs true Debye recorded) and
aborts if any exceeds 1e-9 relative (unless `--falsifier`).

## 8. Artifacts and keys (to be filled by the VESSL run; prose only until then)

- `validation/crossval/_22_dispersive_results/rfx.json` — schema
  `cv22-dispersive-slab/v1`: per arm `freqs_hz`, `gated`, `R_rfx`, `T_rfx`,
  `R_tmm`, `T_tmm`, `R_tmm_ade`, `T_tmm_ade`, `dR`, `dT`, `window_R`,
  `window_T`, `max_dR_gated`, `max_dT_gated`, `mean_dR_gated`,
  `mean_dT_gated`, `worst_bin_R_hz`, `worst_bin_T_hz`, `gates.{G1_R,G1_T,G2_R,G2_T,G3_tail,G3_passivity}`,
  and `meep.{present, dt_meep, R_meep, T_meep, dR_meep_tmm, …, gates.{G4_*, G5_*}}`;
  top-level `verdict.{rfx_self_ok, meep_present, exit_code}`.
- `…/rfx__falsifier_<name>.json` — same schema, one per F1/F2 arm, each
  expected `verdict.exit_code = 1`.
- `…/meep_<arm>.json` (three) and `…/meep_lorentz__falsifier_<name>.json`
  (two) — Meep leg outputs: `freqs_hz`, `R`, `T`, `dt_meep_s`,
  `resolution`, `material`, `precheck.{passed, max_rel_err, freqs_hz}`.
- Public numbers, once measured, are quoted only as
  `path.json::key = value` (#829 form).

## 9. What the VESSL run owes, and what would refute this note

Owed: the six Meep JSONs, `rfx.json` with exit 0, six falsifier `rfx.json`s
each with exit 1, the gate test green on the committed set.

Refutations this note accepts: (i) the baseline fails G1/G2 on any arm — then
either the ADE has a defect the E0/E1 tests do not see, or the cv04 envelope
does not transfer to a lossy slab as claimed in §4; either is a finding, and
the window is NOT to be widened to fit; (ii) a falsifier exits 0 — the gate
does not resolve the declared defect and the case is not claims-bearing;
(iii) the Meep leg's pre-run ε check fails at 1e-9 on a non-falsifier arm —
the mapping in `dispersive_eps.py` is wrong and nothing downstream counts.

## 10. Addendum (2026-09-02, same day, before any measurement) — two defects caught in review, and the artifact count

Both were found while writing the gate test against this note, before the
first FDTD run; neither changes a window, an arm, a band, or a falsifier.

1. **Oracle for a falsifier run.** The first draft of the case script judged
   a `--falsifier` run against the transfer matrix of the *defective* ε(f) —
   self-consistent by construction, so every F1/F2 arm would have exited 0
   and read as "gate does not resolve the defect". Fixed: the FDTD is built
   with the defect (`params_run`, recorded in the artifact) and judged
   against the DECLARED material (`params`). A smoke run of `debye_tau_x2`
   on the 200-cell grid now fails G1_T, G2_R and G2_T with
   mean|ΔR| = 0.024 / mean|ΔT| = 0.086 against the analytic §6 prediction
   0.026 / 0.087 — the smoke rig is not evidence, but it shows the defect and
   the gate are wired to each other.
2. **Meep-side window from the declared material.** The first draft of
   `evaluate_e4` computed `W_ADE,meep` from the `meep_params` the Meep JSON
   *reports*, so a wrongly mapped material (F3) widened its own window and
   `meep_lorentz_no_2pi` passed E4 in the analytic pre-test. Fixed: the
   Meep-side windows are derived from `to_meep(declared material)`; the
   reported parameters are kept for audit only. The analytic F3 pre-test now
   fails E4 for both Meep falsifiers and passes for the correct mapping.

Artifact count owed by the run (correcting §8/§9's "six"): the case script
writes **eight** falsifier `rfx__falsifier_<name>.json` files — six F1/F2
arms plus the two `meep_lorentz_*` arms, in which the rfx Lorentz arm is
correct and the wrong-convention Meep JSON is read — each with
`verdict.exit_code = 1`; plus `rfx.json` (exit 0) and the five Meep JSONs.

## 11. Round 1 fired (VESSL 369367257804, 2026-09-02) — result recorded, round 2 pre-declared

Source for every number below: the run log
`~/Documents/vessl-run-logs/369367257804_cv22-dispersive-try1.log` (rfx
baseline lines 386–460, Meep legs 318–398). The artifacts under
`runs/cv22-dispersive-20260902T101742Z/_22_dispersive_results/` were written
root-owned 0600 by the YAML's final `cp -a` and are unreadable from the Mac;
the `path.json::key` citations are added as a further append once they can be
read (the YAML is fixed in this same commit series: `--no-preserve` copy plus
`chmod -R a+rX` under an EXIT trap).

### 11.1 What fired

| arm | E2 | numbers (gated 4–10 GHz, 115 bins) | E4 |
|---|---|---|---|
| Debye (rfx) | **PASS** (the gate test's per-arm loop passed `debye` before failing on `lorentz`; the E2 summary line itself was cut by the log's `tail -n 40`) | per-frequency table shows \|ΔR\| ≤ 0.023 (9.98 GHz), \|ΔT\| ≤ 0.019 (8.99 GHz) | not evaluated: **Meep Debye leg blew up** (`meep: simulation fields are NaN or Inf` at 0.35 s wall, first steps) |
| Lorentz (rfx) | **FAIL G2_R** — `mean\|ΔR\| = 0.0122` vs window 0.0104 (1.17×); G2_T pass 0.0140 / 0.0176; G1 pass (max\|ΔR\| 0.0289 at 7.95 GHz, max\|ΔT\| 0.0436 at 9.62 GHz, both < 0.074); G3 pass | — | Meep-vs-TMM mean 0.0211 / 0.0196 (G4_mean FAIL both, per-bin pass: max 0.0463 / 0.0502); rfx-vs-Meep mean R 0.0224 FAIL, T 0.0154 pass |
| Drude (rfx) | **PASS** — max\|ΔR\| 0.0012, max\|ΔT\| 0.0059, mean 0.0005 / 0.0017 | — | Meep-vs-TMM mean\|ΔT\| **0.0377** (max 0.0449) FAIL, R 0.0077 pass; rfx-vs-Meep mean T 0.0360 FAIL |
| falsifiers | all 8 exited 1 as pre-declared; Meep F3 legs ran with `precheck.passed = false` recorded (no_2pi: Meep-vs-TMM mean 0.103 / 0.650; gamma_half: 0.110 / 0.050) | | |

Baseline rc 1, gate test 1 failed / 43 passed (the Lorentz replay), Meep
Debye rc 1. **No window is moved.** Per §9(i) the Lorentz E2 failure is a
finding with two admissible readings — an ADE defect the E0/E1 tests do not
see, or the cv04 envelope not transferring — and round 2 is designed to
separate them. Two further findings not anticipated by §4:

- **F-A (Meep side).** For Drude, rfx agrees with the TMM to 0.006 in T
  while **Meep is 0.04 off in T**; for Lorentz Meep's mean deviation is
  ~2× rfx's. The E4 window borrowed cv04's *rfx* envelope for Meep (§4,
  "stated, not measured"); Meep evidently has its own term that this did
  not cover. Candidate terms, each with a pre-declared discriminator below:
  (i) Meep's own spatial discretization at 10 px/cm, (ii) record
  truncation — `stop_when_fields_decayed(…, 1e-3)` against a source whose
  Gaussian amplitude is only 4.2 % of peak at 4 GHz and 11 % at 5 GHz
  (fcen 10, fwidth 15 GHz), so the low band is a ratio of two small
  numbers truncated at 1e-3 of the run maximum, (iii) an interface
  half-cell term.
- **F-B (Meep Debye instability).** Mechanism identified from Meep's own
  discrete Lorentzian (§3): at the Nyquist frequency `ω̃² = 4/dt²`, `ω̂ = 0`,
  so the numerical susceptibility of the mapped pole is
  `χ(Nyq) = −σ (ω_n dt)² / (4 − (ω_n dt)²)`; with σ = 4, ω_n·dt = 1.048 this
  gives `ε_num(Nyq) = 2 − 1.514 = 0.486 < 1`, i.e. an effective permittivity
  below vacuum at the grid's highest mode, which puts Meep's Courant 0.5
  outside the 2-D limit `0.707·√0.486 = 0.49`. The standalone Jury criterion
  of §2.1 (`ω_n·dt < 2`) is necessary, not sufficient; the coupled criterion
  is `ε_num(Nyq) ≥ (S/S_max)²`, and the note's "ω_n·dt = 1.048 < 2,
  Meep-stable" in §7 was wrong on this point. rfx is unaffected: its Lorentz
  arm has `ω0·dt = 0.103` (ε_num(Nyq) = 1.996) and Drude
  `ε_num(Nyq) = 2.997`.

### 11.2 Round 2 — pre-declared experiments (`scripts/vessl_cv22_dispersive_slab_r2.yaml`)

Every window stays as declared in §4. Nothing below is a gate change; each
experiment has a prediction written before the run and an action attached to
each outcome.

**(a) rfx Lorentz and Debye at dx/2 and dx/4, same rig scaled in cells**
(`--dx-div 2|4`: nx_interior, CPML layers, TFSF margin, probe offsets,
tail window and step cap all ×K so the geometry is identical; dt follows the
Grid's Courant; nfft follows n_steps). Artifacts `rfx__lorentz_dx2.json`,
`rfx__lorentz_dx4.json`, `rfx__debye_dx2.json`, `rfx__debye_dx4.json`.

A-priori candidate missing term — TMM sensitivity of R, T to a slab
thickness error of ±dx/2 (the interface half-cell), gated band, computed
from `dispersive_eps.tmm_slab_rt` before the run:

| arm | ±0.5 mm: mean\|ΔR\| / max | mean\|ΔT\| / max | ±0.25 mm: mean\|ΔR\| | mean\|ΔT\| |
|---|---|---|---|---|
| Debye | 0.0108–0.0111 / 0.0207 (5.1–5.4 GHz) | 0.0125–0.0130 / 0.0162 (7.4–7.8 GHz) | 0.0054 | 0.0063 |
| Lorentz | 0.0088–0.0094 / 0.0314 (5.2–5.3 GHz) | 0.0085–0.0089 / 0.0204 (10 GHz) | 0.0045 | 0.0043 |
| Drude | 0.0073–0.0076 / 0.0173 | 0.0079–0.0094 / 0.0234 | 0.0037 | 0.0043 |
| cv04's ε = 4 slab (control) | 0.039 / 0.072 | 0.039 / 0.072 | — | — |

Reading: a full half-cell thickness error would be worth 0.009 of Lorentz
mean|ΔR| — the same size as the whole measured 0.0122 — but the control row
shows cv04's slab is 4× MORE sensitive to that error and still measured
0.0066, so the rig does not realize a full half-cell error and this term
alone cannot be the Lorentz excess. It is carried as the *first-order*
hypothesis to be tested by the scaling, not asserted.

Predictions (Lorentz mean|ΔR|, gated; baseline 0.0122):

| hypothesis | dx/2 | dx/4 | action if this is what is seen |
|---|---|---|---|
| H1 first-order (interface / staircase) | ≈ 0.0061 (2×) | ≈ 0.0031 (4×) | the cv04 envelope does not transfer to a pole with Re ε → 0 in band; the window gains a derived first-order interface term evaluated per arm (the ±dx/2 table), re-declared in a §12 before any re-run; the ADE is not implicated |
| H2 second-order (bulk numerical dispersion) | ≈ 0.0031 (4×) | ≈ 0.0008 (16×) | same as H1 with a second-order term |
| H3 no fall (≥ 0.010 at dx/4) | ≈ 0.012 | ≈ 0.012 | not discretization: either the 719-step time gate (checked by (a') below) or the material model as realized by `update_e_lorentz`; the latter becomes an ADE-defect investigation with the E0/E1 tests as the first suspect |

A first-order fall reads as `mean|ΔR|(dx/4)/mean|ΔR|(dx) ∈ [0.20, 0.35]`,
second-order `≤ 0.12`, "no fall" `≥ 0.7`; anything between is reported as
unresolved, not fitted.

**(a') rfx Lorentz at dx with the time gate opened** (`--nx-interior 1500`,
cv04's rung C4 geometry, ≈ 1940 steps; artifact `rfx__lorentz_nx1500.json`).
Prediction: if the Lorentz excess is record truncation the mean|ΔR| collapses
below 0.0104 here while (a) shows no fall; if (a) shows a fall and (a') does
not move, truncation is excluded. cv04 measured this collapse for its own
closure witness (0.0487 → 0.0002).

**(b) Meep Lorentz and Drude at 2× and 4× resolution** (20 and 40 px/cm,
Courant 0.5; artifacts `meep_<arm>__res20.json`, `meep_<arm>__res40.json`).
Prediction on Drude mean|ΔT| vs TMM (baseline 0.0377): first-order → ≈ 0.019
then ≈ 0.009; second-order → ≈ 0.009 then ≈ 0.002; no fall → ≈ 0.038. A fall
means the E4 window's derivation lacked Meep's own discretization term (it
borrowed the rfx envelope); that term is then derived from the measured
scaling and declared in §12 before any E4 re-gate. No fall means (i) is
excluded and (b') decides.

**(b') Meep Lorentz and Drude at 10 px/cm with the truncation and source
hypotheses removed one at a time**: decay tolerance 1e-6 instead of 1e-3
(`meep_<arm>__decay1e-6.json`), and the source re-centred on the gated band
(fcen 7 GHz, fwidth 10 GHz → amplitude 17 % of peak at 4 GHz instead of
4.2 %; `meep_<arm>__src7.json`). Prediction: if the Drude 0.0377 is
truncation/source-floor it drops below 0.017 (the T mean window) in at least
one of these while (b) shows no fall, and the deviation is concentrated in
4–6 GHz; if both leave it unchanged, (ii) is excluded.

**(c) Meep Debye — pre-declared fix: 4× resolution (40 px/cm), f_n = 100 GHz
kept.** Then `ω_n·dt_meep = 0.262` (≤ 0.5 as required), `ε_num(Nyq) = 1.930`
(> 1: the mechanism of F-B is removed with margin), and `W_map` is unchanged
from §7 (max 3.2e-4 / 1.05e-3 in R / T), `W_ADE,meep` falls to 1.3e-5.
Chosen over lowering f_n because the alternatives at 10 px/cm cost window:
f_n = 40 GHz gives `ω_n·dt = 0.419`, `ε_num(Nyq) = 1.816` but a mapping
residual of 1.6e-2 and `W_map,T` max 6.5e-3 / mean 5.6e-3, a third of the T
mean window spent on the mapping. The f_n = 40 GHz variant is ALSO run, as a
cross-check only (`meep_debye__fn40.json`), with its larger `W_map` carried
by the same formula. The primary res-40 leg is written as `meep_debye.json`
so the baseline case reads it. Acceptance for (c): the res-40 leg runs finite
and `precheck.passed` is true — that much is gated in the r2 replay test;
its E4 verdict is whatever G4/G5 say with the declared windows.

**(d) No threshold moves.** The r2 YAML re-runs the r1 baseline and the eight
falsifier arms unchanged (so a readable `rfx.json` exists), then the
diagnostics above, then the gate test. The r2 replay test computes the
scaling ratios and prints them; it asserts only the structural acceptances
(finite, precheck, artifact schema) and the (c) acceptance — a test that
fails when nature disagrees with a hypothesis is the wrong instrument.

Cost: every leg is seconds (r1 rfx arm 0.4 s, Meep leg 0.1–0.6 s); dx/4 is
16× cells × 4× steps ≈ 64× → ≈ 30 s; Meep 40 px/cm ≈ 16× → seconds.

## 12. Round 2 read (VESSL 369367257805) — two mechanisms decided; round 3 pre-declared

Artifacts (world-readable): `~/mnt/remilab-fs/personal-workspaces/claude-workspace/rfx/runs/cv22-dispersive-r2-20260902T103036Z/_22_dispersive_results/` (abbreviated `r2/` below); log `~/Documents/vessl-run-logs/369367257805_cv22-dispersive-r2.log`. Every number is by key; the derived Meep-vs-TMM numbers are `evaluate_e4(rfx.json::arms.<arm>, <meep file>)` fields, reproduced verbatim by `meep_ladder_summary.json` (§12.3) once committed.

### 12.1 rfx Lorentz — TRUNCATION, decided by the §11.2(a)/(a') reading rules

| artifact | mean\|ΔR\| (`::arms.lorentz.mean_dR_gated`) | ratio to baseline | mean\|ΔT\| | tail scat / trans (`::arms.lorentz.tail.*`) |
|---|---|---|---|---|
| `r2/rfx.json` (dx, 719 steps) | 0.01221 | 1 | 0.01403 | 0.0308 / 0.0584 (−30 / −25 dB) |
| `r2/rfx__lorentz_dx2.json` (1438 steps) | 0.01135 | ×0.93 | 0.01277 | 0.0350 / 0.0595 |
| `r2/rfx__lorentz_dx4.json` (2876 steps) | 0.01034 | ×0.85 | 0.01150 | 0.0363 / 0.0598 |
| `r2/rfx__lorentz_nx1500.json` (1940 steps, gate opened) | **0.00284** | **×0.23** | 0.00153 | 2.1e-5 / 2.5e-5 (−94 dB) |

The dx/4 ratio 0.85 is in the pre-declared "no fall ≥ 0.7" band (§11.2(a), H3): the excess is **not spatial discretization**. The (a') arm collapses it ×0.23 to 0.0028 (max|ΔR| 0.0049, E2 PASS) with the tail witness falling from −25 dB to −94 dB: the excess is **record truncation** — cv04's 719-step CPML rule cuts the Lorentz slab's ring-down (material pole δ = 7.33e9 s⁻¹, −40 dB in 0.63 ns = 269 steps, starting only after the incident pulse has passed the probe at step ~622; the record ends at 719). The dx ladder kept the *physical* record length constant (steps ×K at dt/K), which is exactly why it could not move the truncation term — consistent, not a coincidence. cv04's own #341 comment found the same mechanism for its fringe (0.0487 → 0.0002 at nx 1500); cv04's envelope was borrowed for a lossless slab with fast etalon decay and does not transfer to a Q = 3 pole. Debye (`r2/rfx.json::arms.debye.mean_dR_gated = 0.00521`, PASS; dx ladder 0.00465 / 0.00427, no-fall) and Drude (0.00051, PASS) carry the same truncated tails (−24 dB / −40 dB) but their windows absorb it. **The ADE is not implicated on any arm.**

### 12.2 Meep — its own FIRST-ORDER spatial discretization is the missing E4 term

Meep-vs-TMM, gated band, `evaluate_e4(rfx.json::arms.<arm>, r2/meep_<arm>[__tag].json)::mean_d{R,T}_meep_tmm_gated`:

| arm | res 10 (`meep_<arm>.json`) | res 20 (`__res20`) | res 40 (`__res40`) | decay 1e-6 (`__decay1e-6`) | source 7 GHz (`__src7`) |
|---|---|---|---|---|---|
| Lorentz R / T | 0.0211 / 0.0196 | 0.0101 / 0.0100 | **0.0049 / 0.0051** | 0.0211 / 0.0196 (×1.00) | 0.0211 / 0.0196 (×1.00) |
| Drude R / T | 0.0077 / 0.0377 | 0.0040 / 0.0189 | **0.0020 / 0.0095** | 0.0077 / 0.0377 (×1.00) | 0.0077 / 0.0377 (×1.00) |

Measured order per doubling (`meep_ladder_summary.json::arms.<arm>.orders`): Lorentz R 1.06 / 1.03, T 0.97 / 0.98; Drude R 0.94 / 0.97, T 0.99 / 1.00 — the §11.2(b) "first-order" prediction (0.0377 → ≈ 0.019 → ≈ 0.009; measured 0.0189, 0.0095). Decay tolerance and source re-centring change nothing (×1.00): truncation and the source floor are excluded (§11.2(b')). So the E4 window's derivation (§4: "Meep at the same dx has the same class … stated, not measured") was wrong for Meep's *interface* handling — Meep converges first-order on this slab where rfx at the same dx sits at 0.0005–0.006. rfx-vs-Meep at res 40: Lorentz 0.0127 / 0.0127, Drude 0.0017 / 0.0078, Debye 0.0065 / 0.0078 — all inside the declared G5 means (0.020 / 0.034).

### 12.3 Meep Debye — the §11.2(c) fix held; the cross-check did not run

`r2/meep_debye.json` (40 px/cm, f_n 100 GHz): `::run.finite = true`, `::precheck.max_rel_err = 1.97e-16`, `::resolution = 40`; ω_n·dt = 0.262, ε_num(Nyq) = 1.930 as predicted. Meep-vs-TMM 0.0046 / 0.0084, rfx-vs-Meep 0.0065 / 0.0078, all G4/G5 pass. The f_n = 40 GHz cross-check (`meep_debye__fn40`, rc 1) **did not run**: the leg's pre-run check applies the declared residual bound 3e-3 (§7) to every Debye mapping and rejected the 1.6e-2 residual of the f_n = 40 GHz pole (`precheck: max_rel_err 1.9e-16 … FAIL`, because `passed` also requires `residual < 3e-3`), before any FDTD. §11.2(c) promised to carry the larger W_map for that leg but did not exempt it from the bound; the bound is the stricter statement and stands. With the primary stable and passing, the cross-check is moot and is dropped. 10 px/cm remains the pre-declared F-B instability witness and is re-run in r3 as the Debye rung 10 (expected NaN, `run.finite = false`).

### 12.4 Round 3 — pre-declared before the run (`scripts/vessl_cv22_dispersive_slab_r3.yaml`)

**(1) Recipe correction, physics-derived** (`cv22_dispersive_gates.derive_record_length`, locked by `tests/test_cv22_dispersive_slab_gates.py::test_r3_record_lengths_are_derived_from_the_slab_ringdown`). This corrects a truncated measurement; it moves NO tolerance. The record length comes from the slab's own ring-down, not from cv04:

    n_steps = n_pulse_end + n_ring + TAIL_WINDOW
    n_pulse_end = ceil( (t0 + a40·τ)/dt + (probe_trans − x_lo)/v_cells )
        τ = 1/(π f0 bw) = 63.66 ps, t0 = 3τ (rfx.sources.tfsf), a40 = 2.5255
        (2a e^{−a²} = 1e-2 of its peak), v_cells = c·dt/dx = 0.700
    n_ring = ceil( ln(100) / (rate_slowest · dt) )        -- amplitude 1 → 1e-2 (−40 dB)
        rate_slowest = min(material pole, slowest etalon round-trip in 4–10 GHz)
        material: Debye 1/τ = 3.14e10, Lorentz δ = 7.33e9, Drude γ/2 = 9.42e9 s⁻¹
        etalon: −ln ρ / t_rt, ρ = |r|² e^{−2 k0 Im(n) d}, t_rt = 2 Re(n) d / c:
                Debye 1.84e10 (4.0 GHz), Lorentz 1.77e10 (4.35 GHz), Drude 2.92e10 (10 GHz)

The rig is widened to `nx_interior = 1000` — the smallest round value whose CPML round-trip gate (cv04's 0.95 rule, 1262 steps) exceeds the longest derived record; everything else (dx, CPML depth, TFSF, probes at ±30 cells, mask, FFT oversampling) is cv04's. Derived, per arm, at dt = 2.335 ps (n_pulse_end = 908 on the nx-1000 rig):

| arm | rate_slowest (which) | n_ring | **n_steps** | CPML gate | nfft |
|---|---|---|---|---|---|
| Debye | 1.84e10 s⁻¹ (etalon, 4.0 GHz) | 108 | **1066** | 1262 | 16384 |
| Lorentz | 7.33e9 s⁻¹ (material δ) | 270 | **1228** | 1262 | 16384 |
| Drude | 9.42e9 s⁻¹ (material γ/2) | 210 | **1168** | 1262 | 16384 |

All three land on nfft 16384 (df 26.1 MHz, ~230 gated bins), so the arms share one bin grid. The r2 datum (nx 1500 / 1940 steps) is not copied: 1228 is what the ring-down requires; 1940 was the CPML rule of a wider box. **Witness gate:** the last-50-step tail of the scattered-reflected and total-transmitted records must be **below 1e-2 (−40 dB) of the incident peak** (`SETTLING_LIMIT`; cv04's −20 dB `TAIL_LIMIT` is retired for this case), with cv04's 1e-3 purity check unchanged; the witness values are recorded per arm (`rfx.json::arms.<arm>.tail.{scat_refl_rel,total_trans_rel}`) and the derivation itself is stored (`::arms.<arm>.run.record`). Prediction: the tails land near −46 dB (the ring-down starts from ≤ 0.5, the derivation assumed 1), so a witness failure would mean a slower mode than the two derived ones, which would itself be a finding. Predicted Lorentz mean|ΔR| after the correction: between the r2 floor 0.0028 (−94 dB) and ≈ 0.005 (scaling the 0.009 truncation excess by 1e-2/0.045) — inside the 0.0104 window with ≥ 2× margin. The cv04-derived windows (§4) stay as declared; a truncation-dominated envelope is a conservative one for a record that is no longer truncated.

**(2) E4 reference = Meep at 40 px/cm for all three arms** (`MEEP_PRIMARY_RESOLUTION`), the converged rung of the measured ladder; its remaining deviation (0.002–0.010) sits inside the declared G4 means and no E4 window is widened to admit 10 px/cm. The two wrong-convention Meep falsifiers run at 40 px/cm too (same reference resolution), and the ladder (Lorentz/Drude at 10/20/40; Debye at 10 [instability witness] / 20 / 40) is re-produced and committed as evidence of Meep's first-order term (`meep_ladder_summary.json`, written from the committed rungs by `--meep-ladder-summary`; the gate test replays it and locks the order to [0.8, 1.3] — that lock is **measured-in-r2**, not pre-declared in §4). No derived Meep term is added to the window: at 40 px/cm none is needed, and adding one now would be fitting.

**(3) Falsifiers.** All eight rfx-side arms re-run at the corrected recipe; each must exit 1 — with a passing baseline they now discriminate on every arm. Analytic margins (§6) are unchanged by the recipe (the defects enter through ε(f), not the record).

**(4) The gate test must be green** on the r3 set: baseline replay (recipe r3, derived n_steps per arm, tail ≤ 1e-2, all E2 gates), Meep primary legs at res 40 with `precheck.passed`, all G4/G5, every falsifier artifact failing on the band-mean, both Meep falsifiers failing E4 with `precheck.passed = false`, and the ladder summary reproducing from its rungs.

Refutations accepted: the Lorentz baseline still failing G2_R at the derived record (then the residual is not truncation and the H3 branch of §11.2 reopens with the ADE as the suspect); a −40 dB witness failing (a slower mode than derived); a Meep res-40 leg failing G4 (Meep's first-order term is larger than the ladder measured, or the mapping is wrong in a way 1e-9 does not see).

## 13. Round 3 read (VESSL 369367257810) — truncation diagnosis confirmed; Debye witness missed; round 4 pre-declared

Artifacts: `~/mnt/remilab-fs/personal-workspaces/claude-workspace/rfx/runs/cv22-dispersive-r3-20260902T104347Z/_22_dispersive_results/` (`r3/`); log `~/Documents/vessl-run-logs/369367257810_cv22-dispersive-r3.log`.

### 13.1 What fired

| arm | E2 | `r3/rfx.json::arms.<arm>.{mean_dR_gated, mean_dT_gated}` | max | tail scat / trans (`::tail.*`, bar 1e-2) | E4 (Meep 40 px/cm) |
|---|---|---|---|---|---|
| Lorentz | **PASS** | 0.0028 / 0.0016 (windows 0.0104 / 0.0176) | 0.0050 / 0.0032 | 8.0e-4 / 6.4e-4 | pass |
| Drude | **PASS** | 0.0005 / 0.0017 | — | 7.6e-5 / 4.1e-5 | pass |
| Debye | accuracy PASS, **G3_tail FAIL** | 0.0024 / 0.0029 | 0.0077 / 0.0073 | **1.63e-2 / 1.69e-2** (−36 dB) | pass |

The §12 prediction for Lorentz (0.0028–0.005 at the derived 1228 steps) landed on its floor: the truncation diagnosis is confirmed and the ADE is clear on every arm. All three Meep primaries at 40 px/cm: `precheck.max_rel_err` 2.0e-16 / 1.8e-15 / 8.2e-16, finite, every G4/G5 gate true. Ladders rc 0 except Debye 10 px/cm rc 1 — the pre-declared F-B instability witness. All eight rfx falsifiers rc 1 and both Meep falsifiers rc 1: with a passing Lorentz/Drude baseline they now discriminate. The only red is the Debye witness, which makes `verdict.exit_code = 1` and fails two gate-test replays.

### 13.2 Why the Debye record was short — the slowest component is the sub-band etalon

§12's derivation searched the etalon rate over the **gated** band only. There the slowest Debye component is the 4 GHz etalon (1.84e10 s⁻¹, n_ring 108). But the incident pulse is a differentiated Gaussian peaking at 3.5 GHz with 80 % amplitude at 2 GHz and 45 % at 1 GHz, and Debye's absorption per pass vanishes as k0 → 0 (`ρ = |r|² e^{−2k0 Im(n) d}` → |r_dc|² = 0.176 with n_dc = √6), so the etalon rate falls monotonically toward DC: 1.29e10 at 2 GHz, 1.12e10 at 1 GHz, 1.06e10 as f → 0. The r3 tail is that sub-band component: the record ends 108 steps after the pulse at a scattered level ≤ 0.53 (√R), and 1.63e-2 after 108 steps bounds the effective rate to **≤ 1.36e10 s⁻¹** (`ln(0.53/0.0163)/(108·dt)`), below the 1.84e10 that was assumed and inside the 1.1–1.3e10 of the 1–2.5 GHz etalon. (The r3 artifact does not carry the time record — a gap of the r1 schema, closed in §13.3 by storing the tail envelope and its fitted rate — so this is a bound from the witness value, not a fitted rate; the fit is owed by r4.) Lorentz and Drude are unaffected: their material poles (7.33e9, 9.42e9) are slower than any etalon in the incident band (Lorentz 1.77e10 min; Drude 1.27e10 min at 1.13 GHz), and their r3 tails (−62 dB, −82 dB) are consistent with that.

### 13.3 Round 4 recipe — physics-derived, then adaptive; no window moves

**Derivation (`derive_record_length`, §12 form with two corrections):**

1. the ring-down search runs over the **incident ring band** — the frequencies where the incident amplitude is ≥ 0.5 of its peak, `[1.13, 15] GHz` (`ring_band_hz()`), not the gated band;
2. each spectral component starts at most at its incident weight w(f), so it needs `ln(100·w(f)) / rate(f)` to reach −40 dB of the incident peak; `n_ring = ceil(max_f ln(100 w)/rate / dt)` with `rate(f) = min(material, etalon(f))`.

Result at dt = 2.335 ps on the nx-1000 rig (`n_pulse_end = 908`, CPML gate 1262; locked by `test_r3_record_lengths_are_derived_from_the_slab_ringdown`):

| arm | governing component | rate | n_ring | **n_steps_min** |
|---|---|---|---|---|
| Debye | etalon at 1.45 GHz, w = 0.62 | 1.18e10 s⁻¹ | 150 | **1108** (r3: 1066) |
| Lorentz | material δ (w = 1 at 3.5 GHz) | 7.33e9 | 270 | **1228** (unchanged) |
| Drude | material γ/2 | 9.42e9 | 210 | **1168** (unchanged) |

**Adaptive witness (never clip):** the arm runs to `n_steps_min`, evaluates the −40 dB witness (and cv04's 1e-3 purity), and while it is not met extends the record by `RECORD_EXTEND_STEPS = 100` up to the CPML gate; if the gate would be crossed, the box grows by `NX_GROW_CELLS = 200` (cv04's rig rule: the gate is a property of the box) and the arm reruns. The artifact records `run.record.{n_steps_min, n_steps, extensions, nx_grows}`, the witness values, the last 300 steps of both tail envelopes (`tail.envelope_*`) and their log-linear fitted decay rates (`tail.fitted_rate_*_1_s`, block maxima over 50-step windows; `fit_tail_rate` reproduces 1.26e10 from a synthetic 1.2e10 decay).

**Predictions.** Debye: from the r3 level 1.69e-2 at step 1016 decaying at ≥ 1.06e10 (the slowest etalon in the whole incident band), the tail at the 1108-step window start (1058) is ≤ 0.0169·e^{−1.06e10·42·dt} = **6.1e-3 (−44 dB)** → the witness is met at `n_steps = 1108` with **0 extensions** expected; if one extension fires, 1208 (tail ≈ 5e-4), still inside the 1262 gate, no box growth. Fitted Debye rate: **1.1–1.4e10 s⁻¹** (between the 1–2.5 GHz etalon rates and the r3 bound), i.e. below the material 3.1e10 and the 4 GHz etalon 1.84e10 — that ordering is the check that the component is the sub-band etalon. Lorentz 1228 / Drude 1168 with 0 extensions (their r3 tails were already 8e-4 and 7.6e-5 at those lengths); fitted rates ≈ 7e9 / 9e9 (the material poles). Debye E2 unchanged at the 1e-3 level (0.0024 / 0.0029 at 1066 steps is already inside the window; the longer record can only lower it). All windows as declared in §4.

**Round 4 set** (`scripts/vessl_cv22_dispersive_slab_r4.yaml`, the r3 lane at the corrected recipe): Meep primaries at 40 px/cm (three arms) + both wrong-convention Meep falsifiers at 40; the 10/20/40 ladders (Debye 10 = instability witness); the rfx baseline; the eight rfx falsifiers (each must exit 1); `--meep-ladder-summary`; the gate test, which must be green: it now also asserts `n_steps = n_steps_min + 100·extensions ≤ gate`, `nx_interior ≥ 1000`, the witness ≤ 1e-2, and that the stored envelope refits to the recorded rate.

Refutations accepted: a Debye witness still above the bar after box growth to 4× (a slower mode than any derived here — reopened as a finding); a fitted Debye rate above 1.84e10 (then the r3 tail was not the sub-band etalon and the explanation above is wrong even if the witness passes); any E2/E4 gate failing at the longer record.


## 14. Round 4 (VESSL 369367257811) — green; the witness is the gate, the derivation is the estimate; chronology by key

Artifacts committed under `validation/crossval/_22_dispersive_results/` (this commit; the branch commit that ran was `e81080a`; the artifact's own `commit` field reads `unknown` because the staged copy on the pod ran `git rev-parse` outside a safe directory — the run log `r4_369367257811_rfx_baseline.log` and `r4_369367257811_commit.txt` sit beside the JSONs). Every gate-test replay is green on the committed set: 52 passed, 1 xfail (the pinned `eval_lorentz` Drude gap), no skips.

### 14.1 What the witness measured vs what the derivation estimated

| arm | n_steps_min → reached (+ext) | tail scat / trans (bar 1e-2) | fitted rate scat / trans | derived ring rate (component) |
|---|---|---|---|---|
| Debye | `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.run.record.n_steps_min = 1108` → `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.run.n_steps = 1108` (`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.run.record.extensions = 0`) | `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.tail.scat_refl_rel = 0.0072` / `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.tail.total_trans_rel = 0.0072` | `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.tail.fitted_rate_scat_refl_1_s = 7.8e9` / `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.tail.fitted_rate_total_trans_1_s = 8.1e9` | `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.run.record.rate_ring_1_s = 1.2e10` (`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.run.record.f_ring_hz = 1.4e9` etalon) |
| Lorentz | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.run.record.n_steps_min = 1228` → `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.run.n_steps = 1228` (`validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.run.record.extensions = 0`) | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.tail.scat_refl_rel = 0.0008` / `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.tail.total_trans_rel = 0.00064` | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.tail.fitted_rate_scat_refl_1_s = 8.4e9` / `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.tail.fitted_rate_total_trans_1_s = 7.9e9` | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.run.record.rate_ring_1_s = 7.3e9` (material δ) |
| Drude | `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.run.record.n_steps_min = 1168` → `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.run.n_steps = 1168` (`validation/crossval/_22_dispersive_results/rfx.json::arms.drude.run.record.extensions = 0`) | `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.tail.scat_refl_rel = 0.000076` / `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.tail.total_trans_rel = 0.000041` | `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.tail.fitted_rate_scat_refl_1_s = 1.6e10` / `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.tail.fitted_rate_total_trans_1_s = 1.8e10` | `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.run.record.rate_ring_1_s = 9.4e9` (material γ/2) |

Read plainly: **the −40 dB witness is the gate; the §13 derivation is the estimate that sizes the first attempt.** The Debye estimate missed by ~1.5×: the fitted tail decays at 7.8–8.1e9 s⁻¹ against the derived 1.18e10 (and against §13's predicted 1.1–1.4e10). The record still cleared the bar with zero extensions because the tail's starting level was below the model's assumed w·1 — i.e. the level assumption, not the rate, carried the margin. Lorentz fitted 8.4e9 / 7.9e9 vs the material 7.33e9 (close, slightly faster). Drude fitted 1.56e10 / 1.79e10 vs the material 9.42e9: a faster component dominates its −82 dB tail there; the material-pole estimate was conservative. None of this is tuned: the rate model stays as written (it is only an estimate), the witness stays at 1e-2, and a future arm whose tail decays slower than its estimate is caught by the witness plus the adaptive extension, not by the estimate.

### 14.2 The claims-bearing result (r4, all by key)

E2, rfx vs the transfer matrix with the continuous complex ε(f), gated 4–10 GHz (`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.n_bins_gated = 229` bins):

| arm | mean\|ΔR\| / window | mean\|ΔT\| / window | max\|ΔR\| | max\|ΔT\| |
|---|---|---|---|---|
| Debye | `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.mean_dR_gated = 0.0023` / `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.mean_window_R = 0.01` | `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.mean_dT_gated = 0.0028` / `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.mean_window_T = 0.017` | `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.max_dR_gated = 0.0056` | `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.max_dT_gated = 0.0055` |
| Lorentz | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.mean_dR_gated = 0.0028` / `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.mean_window_R = 0.01` | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.mean_dT_gated = 0.0016` / `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.mean_window_T = 0.018` | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.max_dR_gated = 0.005` | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.max_dT_gated = 0.0032` |
| Drude | `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.mean_dR_gated = 0.00049` / `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.mean_window_R = 0.01` | `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.mean_dT_gated = 0.0017` / `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.mean_window_T = 0.017` | `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.max_dR_gated = 0.0013` | `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.max_dT_gated = 0.0039` |

E4, Meep at 40 px/cm (pre-run mapping check `validation/crossval/_22_dispersive_results/meep_debye.json::precheck.max_rel_err = 2e-16`, `validation/crossval/_22_dispersive_results/meep_lorentz.json::precheck.max_rel_err = 1.8e-15`, `validation/crossval/_22_dispersive_results/meep_drude.json::precheck.max_rel_err = 8.2e-16`):

| arm | Meep vs TMM mean R / T (window) | rfx vs Meep mean R / T (window) |
|---|---|---|
| Debye | `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.meep.mean_dR_meep_tmm_gated = 0.0046` / `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.meep.mean_dT_meep_tmm_gated = 0.0084` (`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.meep.mean_window4_R = 0.01` / `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.meep.mean_window4_T = 0.018`) | `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.meep.mean_dR_rfx_meep_gated = 0.004` / `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.meep.mean_dT_rfx_meep_gated = 0.0055` (`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.meep.mean_window5_R = 0.02` / `validation/crossval/_22_dispersive_results/rfx.json::arms.debye.meep.mean_window5_T = 0.035`) |
| Lorentz | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.meep.mean_dR_meep_tmm_gated = 0.005` / `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.meep.mean_dT_meep_tmm_gated = 0.0051` (`validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.meep.mean_window4_R = 0.01` / `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.meep.mean_window4_T = 0.017`) | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.meep.mean_dR_rfx_meep_gated = 0.004` / `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.meep.mean_dT_rfx_meep_gated = 0.004` (`validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.meep.mean_window5_R = 0.02` / `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.meep.mean_window5_T = 0.035`) |
| Drude | `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.meep.mean_dR_meep_tmm_gated = 0.002` / `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.meep.mean_dT_meep_tmm_gated = 0.0095` (`validation/crossval/_22_dispersive_results/rfx.json::arms.drude.meep.mean_window4_R = 0.01` / `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.meep.mean_window4_T = 0.017`) | `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.meep.mean_dR_rfx_meep_gated = 0.0017` / `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.meep.mean_dT_rfx_meep_gated = 0.0078` (`validation/crossval/_22_dispersive_results/rfx.json::arms.drude.meep.mean_window5_R = 0.02` / `validation/crossval/_22_dispersive_results/rfx.json::arms.drude.meep.mean_window5_T = 0.034`) |

Verdict `validation/crossval/_22_dispersive_results/rfx.json::verdict.exit_code = 0`. Meep's first-order interface term, committed as evidence (`meep_ladder_summary.json`): Lorentz orders R `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.lorentz.orders.order_dR_10_20 = 1.1` / `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.lorentz.orders.order_dR_20_40 = 1`, T `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.lorentz.orders.order_dT_10_20 = 0.97` / `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.lorentz.orders.order_dT_20_40 = 0.98` (mean|ΔT| `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.lorentz.rungs.10.mean_dT_meep_tmm_gated = 0.02` → `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.lorentz.rungs.20.mean_dT_meep_tmm_gated = 0.01` → `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.lorentz.rungs.40.mean_dT_meep_tmm_gated = 0.0051`); Drude R `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.drude.orders.order_dR_10_20 = 0.94` / `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.drude.orders.order_dR_20_40 = 0.97`, T `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.drude.orders.order_dT_10_20 = 0.99` / `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.drude.orders.order_dT_20_40 = 1` (`validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.drude.rungs.10.mean_dT_meep_tmm_gated = 0.038` → `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.drude.rungs.20.mean_dT_meep_tmm_gated = 0.019` → `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.drude.rungs.40.mean_dT_meep_tmm_gated = 0.0095`); Debye 20→40 R `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.debye.orders.order_dR_20_40 = 0.99`, T `validation/crossval/_22_dispersive_results/meep_ladder_summary.json::arms.debye.orders.order_dT_20_40 = 0.92` (10 px/cm is the instability witness: no JSON, rc 1).

Falsifiers, each exit 1 as pre-declared, now against a passing baseline:

| arm | mean\|ΔR\| / mean\|ΔT\| vs the declared oracle | exit |
|---|---|---|
| `debye_tau_x2` | `validation/crossval/_22_dispersive_results/rfx__falsifier_debye_tau_x2.json::arms.debye.mean_dR_gated = 0.025` / `validation/crossval/_22_dispersive_results/rfx__falsifier_debye_tau_x2.json::arms.debye.mean_dT_gated = 0.085` | `validation/crossval/_22_dispersive_results/rfx__falsifier_debye_tau_x2.json::verdict.exit_code = 1` |
| `debye_deps_zero` | `validation/crossval/_22_dispersive_results/rfx__falsifier_debye_deps_zero.json::arms.debye.mean_dR_gated = 0.052` / `validation/crossval/_22_dispersive_results/rfx__falsifier_debye_deps_zero.json::arms.debye.mean_dT_gated = 0.72` | `validation/crossval/_22_dispersive_results/rfx__falsifier_debye_deps_zero.json::verdict.exit_code = 1` |
| `lorentz_f0_x1p3` | `validation/crossval/_22_dispersive_results/rfx__falsifier_lorentz_f0_x1p3.json::arms.lorentz.mean_dR_gated = 0.093` / `validation/crossval/_22_dispersive_results/rfx__falsifier_lorentz_f0_x1p3.json::arms.lorentz.mean_dT_gated = 0.2` | `validation/crossval/_22_dispersive_results/rfx__falsifier_lorentz_f0_x1p3.json::verdict.exit_code = 1` |
| `lorentz_deps_zero` | `validation/crossval/_22_dispersive_results/rfx__falsifier_lorentz_deps_zero.json::arms.lorentz.mean_dR_gated = 0.11` / `validation/crossval/_22_dispersive_results/rfx__falsifier_lorentz_deps_zero.json::arms.lorentz.mean_dT_gated = 0.73` | `validation/crossval/_22_dispersive_results/rfx__falsifier_lorentz_deps_zero.json::verdict.exit_code = 1` |
| `drude_fp_x1p3` | `validation/crossval/_22_dispersive_results/rfx__falsifier_drude_fp_x1p3.json::arms.drude.mean_dR_gated = 0.031` / `validation/crossval/_22_dispersive_results/rfx__falsifier_drude_fp_x1p3.json::arms.drude.mean_dT_gated = 0.17` | `validation/crossval/_22_dispersive_results/rfx__falsifier_drude_fp_x1p3.json::verdict.exit_code = 1` |
| `drude_wp_zero` | `validation/crossval/_22_dispersive_results/rfx__falsifier_drude_wp_zero.json::arms.drude.mean_dR_gated = 0.07` / `validation/crossval/_22_dispersive_results/rfx__falsifier_drude_wp_zero.json::arms.drude.mean_dT_gated = 0.25` | `validation/crossval/_22_dispersive_results/rfx__falsifier_drude_wp_zero.json::verdict.exit_code = 1` |
| `meep_lorentz_no_2pi` (rfx correct; Meep vs TMM `validation/crossval/_22_dispersive_results/rfx__falsifier_meep_lorentz_no_2pi.json::arms.lorentz.meep.mean_dR_meep_tmm_gated = 0.1` / `validation/crossval/_22_dispersive_results/rfx__falsifier_meep_lorentz_no_2pi.json::arms.lorentz.meep.mean_dT_meep_tmm_gated = 0.66`; pre-check `validation/crossval/_22_dispersive_results/meep_lorentz__falsifier_no_2pi.json::precheck.max_rel_err = 3.15`) | rfx vs Meep `validation/crossval/_22_dispersive_results/rfx__falsifier_meep_lorentz_no_2pi.json::arms.lorentz.meep.mean_dR_rfx_meep_gated = 0.1` / `validation/crossval/_22_dispersive_results/rfx__falsifier_meep_lorentz_no_2pi.json::arms.lorentz.meep.mean_dT_rfx_meep_gated = 0.66` | `validation/crossval/_22_dispersive_results/rfx__falsifier_meep_lorentz_no_2pi.json::verdict.exit_code = 1` |
| `meep_lorentz_gamma_half` (Meep vs TMM `validation/crossval/_22_dispersive_results/rfx__falsifier_meep_lorentz_gamma_half.json::arms.lorentz.meep.mean_dR_meep_tmm_gated = 0.094` / `validation/crossval/_22_dispersive_results/rfx__falsifier_meep_lorentz_gamma_half.json::arms.lorentz.meep.mean_dT_meep_tmm_gated = 0.0535`; pre-check `validation/crossval/_22_dispersive_results/meep_lorentz__falsifier_gamma_half.json::precheck.max_rel_err = 0.91`) | rfx vs Meep `validation/crossval/_22_dispersive_results/rfx__falsifier_meep_lorentz_gamma_half.json::arms.lorentz.meep.mean_dR_rfx_meep_gated = 0.092` / `validation/crossval/_22_dispersive_results/rfx__falsifier_meep_lorentz_gamma_half.json::arms.lorentz.meep.mean_dT_rfx_meep_gated = 0.054` | `validation/crossval/_22_dispersive_results/rfx__falsifier_meep_lorentz_gamma_half.json::verdict.exit_code = 1` |

The §6 analytic margins (Debye τ×2 0.026 / 0.087; Lorentz f0×1.3 0.091 / 0.199; Drude fp×1.3 0.031 / 0.167; Δε = 0 rows 0.053 / 0.716, 0.109 / 0.733, 0.069 / 0.249) are reproduced by the FDTD to two digits.

### 14.3 Chronology, r1 → r4 (Lorentz mean|ΔR| is the thread)

| round | VESSL | recipe | Lorentz mean\|ΔR\| | what it decided |
|---|---|---|---|---|
| r1 | 369367257804 | cv04's 719 steps, Meep 10 px/cm | 0.0122 (log; artifacts unreadable) | Lorentz G2_R fired; Meep Debye NaN (ε_num(Nyq) = 0.486); Meep 0.04 off in Drude T |
| r2 | 369367257805 | + dx/2, dx/4, nx 1500; Meep 20/40, decay, source | 0.0114 / 0.0103 / **0.0028** | no-fall on dx, ×0.23 on the time gate → truncation; Meep first-order (ladder); Debye res-40 stable |
| r3 | 369367257810 | derived record over the gated band (1066/1228/1168), Meep 40 | 0.0028 (PASS) | Lorentz/Drude pass; Debye witness −36 dB → sub-band etalon missed |
| r4 | 369367257811 | incident-weighted derivation (1108/1228/1168) + adaptive witness | `validation/crossval/_22_dispersive_results/rfx.json::arms.lorentz.mean_dR_gated = 0.0028` | all gates green; committed here |

No window moved at any round; every change was a recipe correction to a truncated or unstable *measurement*, each pre-declared with its prediction before the run (§11.2, §12.4, §13.3), and each prediction is recorded above with its miss.
