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
