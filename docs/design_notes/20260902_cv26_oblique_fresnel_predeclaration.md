# cv26 oblique-slab Fresnel — pre-declaration (gap lane 4, oblique incidence and the CPML at grazing)

Date: 2026-09-02 (written), committed 2026-09-03 · Lane: `agent/gap4-oblique-fresnel` ·
Case: `validation/crossval/26_oblique_slab_fresnel.py` (new; id `26_oblique_slab_fresnel`;
claims-bearing, E2 + E4).

**Append-only.** Corrections are added as new sections; nothing above a correction is
edited. Every number in this note is ANALYTIC (Fresnel / transfer matrix at the realized
angle, the 2-D Yee dispersion relation, the exact time-harmonic solution of the discrete
lattice with rfx's own CPML recursion) or READ from a committed cv04 artifact, and every
one of them lives in `validation/crossval/comparators/oblique_fresnel.py`; the note quotes
it. No FDTD from this case is evidence at the time of this commit. Section 12 records the
LOCAL rig checks (≤ 20 s each, cv24 §12's discipline: not evidence) and the dead ends they
closed before the design was frozen — including two of my own design errors caught by the
lattice model before any VESSL arm ran.

## 0. Why this case exists

Oblique incidence is never validated in the campaign: cv04 (the only Fresnel case) is
normal incidence. The solver has two oblique TFSF paths — `rfx/sources/tfsf_2d.py`
(`method="bloch"`, Bloch-periodic transverse axis, the #404 field-transformation) and
`rfx/sources/tfsf_oblique_open.py` (`method="methodB"`, open-domain 2.5-D) — documented in
`docs/public/guide/sources-ports.mdx` ("TFSF plane-wave source"), and neither is in
crossval. The existing physics evidence is `tests/test_oblique_fresnel_magnitude.py` /
`_phase.py` (slow / highmem, |Γ| vs half-space Fresnel R_TE at 30° / 45° within 10 % / 8 %,
measured 7.2 % / 3.9 %; 60° "CPML-contaminated (memory)" and not gated) and the TFSF leakage
tests (`tests/test_tfsf_oblique*.py`, SF/TF energy < 1 % at 30° / 45°). The CPML's
behaviour at grazing incidence is gated nowhere. This case puts both onto cv04's slab, where
the rig's own error is a committed number, with the exact discrete lattice as the a-priori
model of everything the rig does.

## 1. Rig — cv04's slab and probes on the Bloch TFSF path

`validation/crossval/04_multilayer_fresnel.py` PART 1, with three declared changes:

1. **The injector is the 2-D auxiliary grid** (`init_tfsf_2d`, called directly so that the
   θ₀ = 0 control runs on the SAME complex-envelope path; `init_tfsf(angle_deg=0)` would
   dispatch to cv04's 1-D aux). The 3-D grid carries the complex Bloch envelope
   (`init_state(field_dtype=complex64)`, `bloch_phase_tuple` on the y roll,
   `init_cpml(field_dtype=complex64)`), the update order is cv04's
   (update_h → tfsf_h → cpml_h → aux_h; update_e → tfsf_e → cpml_e → aux_e), probes at
   ±30 cells from the slab faces, the aux grid's field at the same x as the incident
   reference, R = |scat/inc|², T = |tot/inc|² per bin, cv04's masks (3–15 GHz, 2 % incident
   amplitude) and witnesses. Spectra use the +j DFT kernel (the envelope carries
   `exp(−j2πf₀t)`, `tfsf_2d.py:385`).
2. **The transverse wavenumber is fixed per run**, `k_y = k₀(f₀) sin θ₀`
   (`tfsf_2d.py:233`), f₀ = 10 GHz. The Bloch boundary condition holds at EVERY frequency
   of the pulse, so a bin at frequency f is a plane wave at the **realized angle**
   θ(f) = asin(k_y c / 2πf); below the cutoff f_c = f₀ sin θ₀ the incident is evanescent.
   Every oracle in this case is evaluated at θ(f), never at θ₀ (§5). An arm therefore
   measures R(θ) over a RANGE of angles, and the arms overlap.
3. **The interior box is 1500 cells** (nx 1541 at dx; cv04's 600) and the record is derived
   per arm (§6): at fixed k_y the group velocity along x is c cos θ, so the slow components
   arrive late while the fast ones echo off the absorber early; cv04's 600-cell box cannot
   hold a 70° component's ring-down inside the 45° component's CPML time gate (the
   condition 1.5 N/cos θ_min ≳ 0.5 N/cos θ_max + pulse + ring has no solution below
   N ≈ 1400 for θ_max = 70°).

Unchanged from cv04: dx = 1 mm, `Grid(freq_max=20e9, mode="2d_tmz")` → dt = 0.99 dx/(c√2)
= 2.3351 ps (c dt/dx = 0.70004; `rfx/grid.py:96`), 20-cell CPML on x (cubic σ, κ = 1,
R_asym = 1e-15: `rfx/boundaries/cpml.py:101-158`), TFSF margin 5, slab ε = 4, d = 10 mm on
the E nodes [765, 775) at dx, y extent 0.004 m (4 cells; the envelope is y-uniform).
Bandwidth is per arm (§2). The oblique arms run at **dx/2** by the rule of §4.6 (interior,
absorber depth, margin, probe offsets, tail window and extension all ×2; the aux grid's own
constants are `tfsf_2d`'s and do not scale).

**Polarizations.** Both rfx oblique paths inject E perpendicular to the plane of incidence
(`ez` tilts in xy, `ey` tilts in xz; `methodB` is `ez` only): Fresnel **TE (s)** in every
case. p-polarization on an ε-slab is not injectable through the TFSF API. TM is gated
through the exact ε ↔ μ duality: for an `ez` wave, a slab with (ε_r = 1, μ_r = 4) has
r₁₂ = (μ k₁ₓ − k₂ₓ)/(μ k₁ₓ + k₂ₓ) and k₂ₓ = √(εμ k₀² − k_y²), i.e. bit-for-bit the TM
r₁₂ = (ε k₁ₓ − k₂ₓ)/(ε k₁ₓ + k₂ₓ) of the (ε_r = 4, μ_r = 1) slab, Brewster included
(`test_te_tm_duality_is_exact`: 0 ulp at 0–85°). The TM arms are therefore a claim about
rfx's μ_r update under oblique Bloch injection (Hy(i+½) and Hx(i) both read `mu_r[i]`,
`rfx/core/yee.py:263-290`; the half-cell stagger is in the lattice model), not about a
p-polarized source. Meep runs the real p-polarization (Hz on the ε-slab).

## 2. Arms

| arm | θ₀ | pol (rfx slab) | bw | f_c (GHz) | 10 % incident band (GHz) | realized θ gated | recipe |
|---|---|---|---|---|---|---|---|
| `te_00` | 0 | TE (ε 4) | 0.25 | — | 6.21–13.79 | 0 | dx |
| `te_30` | 30 | TE | 0.1902 | 5.000 | 7.11–12.89 | 22.8–44.6° | dx/2 |
| `te_45` | 45 | TE | 0.1114 | 7.071 | 8.31–11.69 | 37.2–58.3° | dx/2 |
| `te_60` | 60 | TE | 0.0509 | 8.660 | 9.23–10.77 | 53.5–69.7° | dx/2 |
| `tm_00` | 0 | TM dual (μ 4) | 0.25 | — | 6.21–13.79 | 0 | dx |
| `tm_45` | 45 | TM dual | 0.1114 | 7.071 | 8.31–11.69 | 37.2–58.3° | dx/2 |
| `tm_60` | 60 | TM dual | 0.0509 | 8.660 | 9.23–10.77 | 53.5–69.7° (θ_B = 63.43° inside) | dx/2 |
| `graze_vac` | 82 | vacuum | 0.0037 | 9.903 | 9.94–10.06 | 80.0–84.8° | dx, compact box |
| `graze_pec` | 82 | PEC on the slab nodes | 0.0037 | 9.903 | 9.94–10.06 | 80.0–84.8° | dx, compact box |
| `graze_te` | 82 | TE (ε 4) | 0.0037 | 9.903 | 9.94–10.06 | 80.0–84.8° | dx, compact box |

**Bandwidth is derived, not chosen** (`bandwidth_for`): the aux source is the complex
modulated Gaussian of `tfsf_2d.py:383-386` (τ = 1/(π f₀ bw), t₀ = 3τ; amplitude spectrum
exp(−((f−f₀)/(bw f₀))²)). Spectral content at the cutoff has zero group velocity along x
and never leaves the probes, so it sits in the tail window as "incident" and would fire
cv04's purity witness (1e-3, `04_multilayer_fresnel.py:209`); the cutoff amplitude is set
to that bar: (1 − sin θ₀)/bw ≥ √ln 1000 = 2.628, floored to 4 decimals, capped at 0.25
(`tfsf_2d` docstring's "≲ 0.3"). §12.3 records that 1e-2 at cutoff (my first choice) fired
the purity witness at 60° in the rig check.

**Compact box** (`NX_INTERIOR_GRAZE = 100`, nx 141, x_lo 25, slab nodes [65, 75), probes
35 / 105, hi-CPML [121, 141)): the absorber echo is INSIDE the record by design (§4.5).
`graze_pec` pins E = 0 on the slab nodes after every E update (a hard PEC: R_Fresnel ≡ 1
at every angle and polarization, so the whole measured excess R − 1 is the rig's).

Gated bins on every arm: incident amplitude ≥ 10 % of peak (power ≥ 1 %; cv22's 4–10 GHz
band on the differentiated Gaussian was ≥ 8.6 %), propagating, θ(f) ∈ [θ_lo, θ_hi] with
θ_hi ≤ 70° on the wide rig (§6's CPML gate) and [80°, 85°] on the compact box.

## 3. The Yee lattice at fixed k_y — derived before the run

### 3.1 Numerical dispersion (anisotropy) and the phase error across the slab

With ω̂ = 2 sin(ω dt/2)/dt and the exact transverse Bloch difference
K_y = 2 sin(k_y dx/2)/dx (`(e^{−jk_y dx} − 1)/dx`, `tfsf_2d.py:344`), the 2-D lattice
wavenumber along x in a medium (ε, μ) is

    (2 sin(k_x,num dx/2)/dx)² + K_y² = εμ ω̂²/c²          (`yee_kx`)

— the relation `tfsf_oblique_open._k_numerical` bisects. The named window term
`W_disp(f) = |R_TMM(k_x,num) − R_TMM(k_x)|` (T likewise) puts the lattice k_x of each
medium through the same transfer matrix (`dispersion_term`); the round-trip phase error
across the slab is 2 (k₂ₓ,num − k₂ₓ) d. At dx = 1 mm:

| arm | phase error at f₀ (rad) | max over the gated band (rad) | mean W_disp,R at dx | at dx/2 |
|---|---|---|---|---|
| te_00 | 0.013 (6.2 GHz) … 0.146 (13.8 GHz) | 0.146 | 0.0074 | 0.0018 |
| te_30 | 0.049 | 0.111 | 0.0070 | 0.0017 |
| te_45 | 0.044 | 0.075 | 0.0111 | 0.0028 |
| te_60 | 0.040 (0.030 at 69.5°, 0.049 at 55°) | 0.052 | 0.0160 | 0.0040 |
| tm_00 / tm_45 / tm_60 | as TE | as TE | 0.0086 / 0.0012 / 0.0002 | 0.0021 / 0.0003 / 0.0001 |

The phase error is larger where the band reaches higher frequency, not where the angle is
larger (the anisotropy of the Yee lattice at Courant 0.7 is small next to its dispersion).
W_disp is second order (`test_dispersion_term_is_second_order…`).

### 3.2 The exact time-harmonic solution of the discrete system (`yee_lattice_full`)

cv23 §12.2 wrote the 1-D lattice of the staircase slab. Here the 2-D TMz lattice at fixed
k_y reduces EXACTLY to a 1-D lattice in x: with Ĥx = −D⁺Ê/(jω̂μ) and D⁻D⁺ = −K_y²,

    jω̂ μ_i Hy_{i+½} = S_i⁻¹ (E_{i+1} − E_i)/dx
    jω̂ ε̃_i E_i     = S_i⁻¹ (Hy_{i+½} − Hy_{i−½})/dx,     ε̃_i = ε_i − K_y²/(ω̂² μ_i),

with E_nx = 0 and Hy_{−½} = 0 (the zero-padded differences, `rfx/core/yee.py:121-138`),
the slab's (ε_i, μ_i) on its nodes (Hy(i+½) and Hx(i) both take `mu_r[i]`), and
`S_i⁻¹ = 1/κ_i + c_i/(1 − b_i z⁻¹)` the z-domain transfer function of rfx's CPML recursion
(`psi = b psi + c curl; field += coeff (psi + (1/κ − 1) curl)`, `cpml.py:631-648, 892-901`,
both E and H; z = e^{jωdt}) on the 20 outer nodes of each face (hi face flipped). The TFSF
face corrections (`tfsf_2d.py:664-667, 707-710`, coefficients dt/(ε₀dx), dt/(μ₀dx)) are
forcing terms; the injected incident is **the aux grid's own exact field**
(`aux_lattice_field`: the same lattice with `tfsf_2d`'s CFS-CPML — 30 cells, 4th order,
κ_max 7, σ_max = 0.8·5·7/(η dx) = 74 S/m, α = 0.05(1−ρ) — on both ends and a unit soft
source at its node 33), sampled at the two faces and at the probes, exactly as the run
normalizes. The result is a tridiagonal system per frequency. It reproduces the run's R
and T once settled — the interface nodes, the bulk dispersion, the absorber's reflection
AND the aux grid's own absorber echoes at once. `ideal_absorber=True, aux="plane"`
replaces both absorbers by outgoing-wave terminations and the aux by a unit lattice plane
wave: the absorber-free reference, which equals cv23's 1-D march to 1e-14 at k_y = 0
(`test_full_lattice_with_ideal_absorber_equals…`) and converges to Fresnel at second order
at 45° (`test_lattice_converges…`).

The a-priori residual it predicts for rfx against Fresnel (mean over the gated bins;
`lattice_margin`) — cv23's "W_lat", REPORTED here, never carried in a window:

| arm | dx: mean/max \|lattice − Fresnel\|_R | dx/2 | mean window R at dx / dx/2 (§4) | margin at dx / dx/2 |
|---|---|---|---|---|
| te_00 | 0.0098 / 0.0185 | 0.0024 / 0.0046 | 0.0182 / 0.0126 | 1.86 / 5.28 |
| te_30 | 0.0124 / 0.0199 | 0.0031 / 0.0049 | 0.0179 / 0.0127 | 1.45 / 4.13 |
| te_45 | 0.0176 / 0.0211 | 0.0043 / 0.0052 | 0.0222 / 0.0139 | 1.27 / 3.20 |
| te_60 | 0.0224 / 0.0269 | 0.0056 / 0.0066 | 0.0274 / 0.0154 | 1.22 / 2.75 |
| tm_00 | 0.0098 / 0.0185 | 0.0024 / 0.0046 | 0.0194 / 0.0129 | 1.97 / 5.40 |
| tm_45 | 0.0112 / 0.0198 | 0.0027 / 0.0049 | 0.0117 / 0.0108 | 1.05 / 3.99 |
| tm_60 | 0.0081 / 0.0164 | 0.0019 / 0.0039 | 0.0104 / 0.0102 | 1.28 / 5.31 |

(T identical to R: lossless.) Read plainly: cv04's committed 0.0066 band-mean at normal
incidence is, as cv23 found, the lattice's own term; at 45–60° the same term is 1.8–2.3×
larger at dx = 1 mm, and the μ-slab's half-cell stagger adds an interface term of its own
that the dispersion-only W_disp does not carry (tm_45: 0.0112 against W_disp 0.0012).

### 3.3 The absorber at grazing incidence, a priori

Continuum theory for rfx's profile (σ_max = −ln R_asym (m+1)/(2ηd), so ∫σ dx is
depth-independent): amplitude reflection R_asym^{cos θ} = 1e-15^{cos θ}: 1e-13 at 30°,
3.2e-8 at 60°, 7.4e-6 at 70°, **2.5e-3 at 80°, 8.2e-3 at 82°, 4.9e-2 at 85°**
(`cpml_continuum_reflection`). The exact lattice on the compact PEC box at θ₀ = 82°
(`evaluate_grazing_pec`, band 80.0–84.8°, 69 bins at nfft 262144) splits the a-priori
excess R − 1 into the 3-D absorber's own term (lattice with rfx's CPML, plane-wave aux) and
the injection path's aux-echo term (the difference to the full model):

| θ(f) | f (GHz) | 3-D CPML term R−1 | aux-echo term | total (the prediction) | 2·R_asym^{cos θ} |
|---|---|---|---|---|---|
| 84.77 | 9.944 | +0.109 | +0.127 | +0.236 | 0.086 |
| 83.83 | 9.960 | +0.115 | +0.157 | +0.272 | 0.049 |
| 83.01 | 9.977 | +0.111 | +0.119 | +0.229 | 0.030 |
| 82.29 | 9.993 | +0.098 | +0.036 | +0.134 | 0.019 |
| 81.63 | 10.009 | +0.080 | −0.053 | +0.027 | 0.013 |
| 81.01 | 10.026 | +0.057 | −0.122 | −0.064 | 0.009 |
| 80.44 | 10.042 | +0.033 | −0.162 | −0.129 | 0.007 |

So: (i) the discrete 20-cell CPML at 80–85° reflects |r| ≈ 0.007–0.06 in amplitude
(R−1 ≈ 2|r|cos φ: 0.013–0.115), **2–3× the continuum estimate**, and σ_max·dt/ε₀ = 2.4 at
its outer cell says why: the cubic profile is steep on this lattice; (ii) the aux grid's
absorber (σ dt/ε₀ = 19.6 at its outer cell) echoes a −x component of comparable size back
into the "incident" that the TFSF injects — invisible on the wide rig (its round trip is
2(nx + 55)/v_gx ≥ 5000 steps, outside every record) and inside the record on the compact
box. The depth ladder (3-D term, max over the band): **8 cells 0.65, 16 cells 0.17,
20 cells 0.115, 32 cells 0.051**; halving to 10 cells: 0.41. A weaker σ (R_asym → 10⁻⁷·⁵):
0.33 max but crossing the declared curve (39 of 69 bins outside the window, §8).

## 4. Windows — derived, not chosen

Repo rule (`tests/_gate_policy.py::gate_from_envelope`, multiplier 1.5, quantum 1000).
cv04's committed envelope (the same three sources cv22 §4 and cv23 §4 cite):
`tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[id=mean_reflectance_error].observed_baseline = 0.0066`,
`…[id=mean_transmittance_error].observed_baseline = 0.011`, and the per-bin
max|R+T−1| = 0.0487 pinned at `validation/crossval/04_multilayer_fresnel.py:309`
(rung C4, job 369367246779; taken as the per-bin envelope for |ΔR| and |ΔT| as cv22 did).

    W_bin = 0.074,   W_mean,R = 0.010,   W_mean,T = 0.017

Two named rig terms of THIS rig, per bin:

- `W_disp(f)` (§3.1), the lattice's numerical dispersion at the realized angle;
- `W_inj(f) = 2√X·L + L²` with `L = LEAK_BAR = 1e-3` (−60 dB): a leakage field of relative
  amplitude ≤ L adds coherently to the scattered or transmitted field. The bar is gated by
  the vacuum arm (§4.5); the transformed-frame TFSF is exact on the lattice (the aux and
  the 3-D grid are the same discrete operator) and the rig check measured 3.6e-7 (§12.1),
  so L is a bar with ≥ 10³ headroom, not an estimate.

### 4.1 E2 gates (rfx vs Fresnel at θ(f)), per arm

- **G1 per bin**, every gated bin: `|R_rfx − R_F(θ(f))| ≤ W_bin + W_disp,R(f) + W_inj,R(f)`,
  T likewise.
- **G2 band-mean**: `mean|ΔR| ≤ W_mean,R + mean W_disp,R + mean W_inj,R` (T likewise). At
  the primary recipes (R / T): te_00 0.0182 / 0.0262, te_30 0.0127 / 0.0204, te_45
  0.0139 / 0.0214, te_60 0.0154 / 0.0225, tm_00 0.0194 / 0.0274, tm_45 0.0108 / 0.0192,
  tm_60 0.0102 / 0.0190.
- **G3**: closure `|R+T−1| ≤ W_bin + W_disp,R + W_disp,T + W_inj,R + W_inj,T` per gated bin
  (lossless slab); passivity `R + T ≤ 1.06` (cv04's `CONS_MAX_LIMIT`); the −40 dB settling
  witness on both tails (`SETTLING_LIMIT = 1e-2`) and cv04's 1e-3 purity on the incident.

### 4.2 The lattice witness (reported, never gated)

Per arm and bin: `R_lattice`, `T_lattice` (the full model, §3.2), the absorber-free
reference, `|rfx − lattice|` (prediction ≤ 3e-4 in the mean at every recipe, from the rig
checks of §12) and the split of the absorber term into 3-D-CPML and aux-echo parts. As in
cv23 §14.6, carrying W_lat inside the window is the alternative NOT taken.

### 4.3 E4 gates (Meep present), per Meep arm

Meep 2-D, `k_point = (0, k_y a/2π, 0)` (§7), Ez (TE) or Hz (real TM), 40 px/cm (cv22 /
cv23's converged rung) with the block centred +½ pixel off a node (cv23 §14.3: the nominal
block holds N+1 integer-position nodes; shifted it holds N — the +a/res thickness excess
cv22 carried is removed by construction here, stated, not measured on this case).
- **G4** Meep vs Fresnel at θ(f): per bin `≤ W_bin + W_disp(f)` (Meep's own dispersion at
  a/40 is 16× smaller than rfx's at dx; the rfx term bounds it), band-mean
  `≤ W_mean + mean W_disp`; pre-check passed; the k_point maps back to the declared k_y to
  1e-9; the flux band covers the gated band.
- **G5** rfx vs Meep, triangle: per bin `≤ 2 W_bin + W_disp + W_inj`, band-mean
  `≤ 2 W_mean + mean(W_disp + W_inj)`.
The grazing arms have no Meep leg: Meep's PML at 80–85° is its own uncontrolled term.

### 4.4 The Brewster gate (`tm_60`)

θ_B = atan 2 = 63.435°. At dx/2 (nfft 131072, df 6.5 MHz) the gated bin nearest θ_B is
9.68431 GHz, θ = 63.413°, where R_TM,Fresnel = 1.1e-7 and R_TE = 0.437; the exact lattice
predicts R = 1.05e-3 there (its own minimum 1.02e-3 sits at 63.80°). **Gate:**
`R_rfx(bin) ≤ W_bin + W_disp + W_inj = 0.0740` at that bin. Reported: the angle of the
measured minimum (prediction: within 0.4° of θ_B) and R_TM within ±1° (Fresnel ≤ 2.2e-4).
The floor is the only committed per-bin window; the lattice prediction is 70× under it and
is what the F2 falsifier (§8, R_TE = 0.437 at that bin, 4.7× the floor) discriminates.

### 4.5 The grazing gates (compact box)

- **G_leak** (`graze_vac`): `|scat/inc| ≤ LEAK_BAR` at every gated bin. With no scatterer
  the incident never leaves the total-field region (the TFSF removes it at x_hi), so the
  absorber is NOT hit and this arm is the injection witness alone — §12.2 records that I
  first designed it as the absorber gate and the lattice refuted that (r = 1e-14) before
  any run. A failure is a rig defect (exit 1), never a physics verdict.
- **G6 absorber** (`graze_pec`): `|R_rfx − R_lattice| ≤ PML_REL·|R_lattice,3D − 1| + PML_FLOOR_R`
  with `PML_REL = 0.5`, `PML_FLOOR_R = W_inj(1) = 2.001e-3`, on the gated bins where the
  3-D term `|R_lattice,3D − 1| ≥ 3·PML_FLOOR_R` (all 69 at 80–85°; the window is
  8.7e-3–0.060 there — the relative part scales with the claim, never with the total, whose
  zero crossings would collapse it to the floor; §12.4's rig check sat 5× under its
  smallest value). R_lattice is the full
  model of the DECLARED absorber (20 cells, R_asym 1e-15); falsifier arms are built with
  their defect and judged against it. **The claim:** rfx's CPML at 80–85° reflects
  |r| = 0.007–0.06 (R − 1 up to 0.115 behind a PEC; 2–3× the continuum R_asym^{cos θ}) and
  the run equals the exact discrete model of that absorber. **Refuted by:** a measured
  excess outside 0.5× the prediction (larger: the absorber is worse than its own discrete
  model — a defect in `apply_cpml_e/h` relative to the recursion transcribed in §3.2;
  smaller: the echo is not in the record or the PEC leaks, a rig defect); the depth ladder
  not following its rung's prediction (§8, F5).
- **G7 slab** (`graze_te`): R and T against the full lattice within `W_bin + W_inj`; the
  excess over Fresnel (a priori up to 0.186 in R at 84°, R_F = 0.86–0.95) is REPORTED next
  to the a-priori 3-D and aux terms (0.108 / 0.154 max). The absorber-free lattice differs
  from Fresnel by ≤ 4.8e-3 here, so everything above that is the absorber.

Only ~2 independent spectral degrees of freedom sit in the 0.12 GHz grazing band (record
≈ 51 ns); the 69 bins are the 8× oversampled grid. Stated, not hidden.

### 4.6 The primary-recipe rule

An arm whose a-priori lattice term (§3.2 table) is within `LATTICE_MARGIN_MIN = 1.5×` of
its mean window at dx runs at **dx/2 as its primary recipe** (`primary_dx_div`; cv23
§13.3: resolution, not tolerance — no window moves, W_disp shrinks 4× with it). Result:
dx/2 on te_30 (1.45×), te_45 (1.27×), te_60 (1.22×), tm_45 (1.05×), tm_60 (1.28×); dx on
te_00 (1.86×) and tm_00 (1.97×). Margins at the primary recipes: 2.75–5.4×. The dx rungs
of the five oblique arms are run as evidence (predicted: the lattice term at dx, second
order: 0.0124 / 0.0176 / 0.0224 / 0.0112 / 0.0081, i.e. te_60 and tm_45 would FAIL G2 at
dx by 0.5–5 % — recorded as the prediction, not run as a primary).

## 5. The realized-angle convention

Declared: **gate R(f, θ(f)) against Fresnel at the same θ(f)**. A fixed k_y makes the
angle sweep across the band (60° arm: 53.5–69.7°; the union of the arms 22.8–69.7° plus
80–85°); the alternative — gating only the bins near θ₀ — throws away most of the record
and, at fixed k_y, is not what the solver computes. θ₀ is a label for k_y. F1 (§8) is
exactly the failure this convention must catch: a k_y of 55° judged at the 60° curve.

## 6. Record length — derived per arm; the witness is the gate

cv22 §13.3's rule adapted to the oblique path (`derive_record`):

    n_steps_min = n_pulse_end + n_echo + n_ring + TAIL_WINDOW·K
    n_pulse_end = ceil( (t₀ + a₄₀ τ)/dt + (22 + probe_trans − x_lo)/v_slow ),  a₄₀ = √ln100 = 2.146
                  v_slow = (c dt/dx) cos θ_hi   (the SLOWEST gated component; 22 = aux source → x_lo)
    n_ring      = ceil( max_f ln(100 w(f))/rate(f) / dt ),  rate = −ln|r₁₂|² / t_rt,
                  t_rt = 2 d n/(c cos θ_t)   (the round-trip group delay along x at fixed k_y: v_gx = (c/n) cos θ_t)
    n_echo      = compact box only: the absorber echo's path at v_slow
    CPML gate   = arrival of the FASTEST gated component + 2·dist/v_fast·0.95  (must exceed n_steps_min on the wide rig)

| arm (recipe) | n_pulse_end | n_ring (θ, rate) | n_echo | **n_steps_min** | CPML gate | nfft |
|---|---|---|---|---|---|---|
| te_00 (dx) | 1427 | 120 (0°, 1.65e10 s⁻¹) | — | **1597** | 3141 | 16384 |
| te_30 (dx/2) | 3911 | 285 (32.0°, 1.36e10) | — | **4296** | 6807 | 65536 |
| te_45 (dx/2) | 5559 | 361 (48.9°, 1.03e10) | — | **6020** | 7959 | 65536 |
| te_60 (dx/2) | 9239 | 531 (64.2°, 6.66e9) | — | **9870** | 10867 | 131072 |
| tm_00 (dx) | 1427 | 120 | — | **1597** | 3141 | 16384 |
| tm_45 (dx/2) | 5559 | 180 (43.0°, 2.15e10) | — | **5839** | 7959 | 65536 |
| tm_60 (dx/2) | 9239 | 111 (56.7°, 3.22e10) | — | **9450** | 10867 | 131072 |
| graze_vac | 20528 | 0 | 1738 | **22316** | (compact) | 262144 |
| graze_pec | 20528 | 0 | 923 | **21501** | (compact) | 262144 |
| graze_te | 20528 | 990 (83.5°, 1.69e9) | 1738 | **23306** | (compact) | 262144 |

Adaptive witness as cv22 §13.3: while either tail is ≥ 1e-2 of the incident peak or the
purity ≥ 1e-3, extend by 100·K steps up to the CPML gate (wide rig; then grow the box by
200 cells and rerun) or up to 2× n_steps_min (compact box; then the tail witness fails,
exit 1). Prediction: te_60 (margin 997 steps = 4 extensions) and the compact arms settle
with ≤ 5 extensions (the 82° PEC rig check needed 5, §12.4); the wide arms' tails land at
−50 to −60 dB. Never borrowed: cv04's 719 steps would end before the 60° pulse arrives.

## 7. Meep mapping (the module is the authority: `meep_k_point`, `meep_fwidth_for`)

Meep: "the k_point vector is specified in Cartesian coordinates in units of 2π/distance",
Bloch phase e^{ik·r}. With a = 1 cm per Meep unit: **k_meep = (0, k_y a/2π, 0) =
(0, (f₀a/c) sin θ₀, 0)** — te_30 0.16678, te_45 0.23587, te_60 0.28887, te_00 0. The line
source carries `amp_func = exp(i 2π k_meep·r)` (Meep's documented oblique-planewave
construction). The wrong convention (k in rad/a, ×2π: 1.4820 at 45°) maps back to a k_y
above k₀ at every band frequency — evanescent — and fails the 1e-9 round-trip
(`test_meep_k_point_round_trips_to_1e9…`; F4 in §8). The Gaussian source: Meep's
`fwidth = √2 π bw f₀` equals rfx's spectrum bin for bin (0.3705 / 0.2819 / 0.1651 / 0.0754
Meep units for bw 0.25 / 0.1902 / 0.1114 / 0.0509), fcen 0.33356. The pre-run check maps
the k_point back to k_y and requires θ(f₀) = θ₀ to 1e-9; with `--falsifier k_2pi` it is
recorded as failed and the run proceeds.

## 8. Falsifiers — pre-declared, with analytic margins (`falsifier_prediction`)

Each is `--falsifier <name>` of the case script (rfx side) or of the Meep leg, and MUST
exit 1 — judged against the DECLARED oracle (cv22 §10.1). Margins are at the arm's primary
recipe over its gated bins.

| name | defect | mean\|ΔR\| / window | mean\|ΔT\| / window | bins > W_bin | named bin |
|---|---|---|---|---|---|
| `te_60_angle_m5` (F1) | k_y of 55° in the TFSF, judged at 60° | 0.0455 / 0.0154 (**3.0×**) | 2.0× | 0 / 236 | ΔR 0.056 at 69.4° |
| `te_45_swap_tm` (F2) | TE run judged against TM | 0.235 / 0.0119 (**19.8×**) | 11.6× | 226 / 259 | ΔR 0.318 at 42.7° |
| `tm_60_swap_te` (F2) | TM (dual) run judged against TE | 0.465 / 0.0284 (**16.4×**) | 13.1× | 236 / 236 | Brewster bin: R_TE 0.437 vs floor 0.093 |
| `te_45_eps_x1p2` (F3) | slab ε 4.8, judged at 4 | 0.207 / 0.0231 (**9.0×**) | 6.8× | 213 / 259 | ΔR 0.360 at 54.4° |
| `meep_te_45_k_2pi` (F4) | Meep k_point in rad/a | pre-check fails; E4 on an evanescent k_y | — | — | `precheck.passed = false` |
| `graze_pec_depth_half` (F5) | 10-cell CPML, judged at the 20-cell prediction | \|ΔR\| max 0.325 vs window max 0.060 | — | **69 / 69** | excess up to 90× the declared |
| `graze_pec_sigma_half` (F5b) | R_asym 10⁻⁷·⁵ (σ_max halved) | \|ΔR\| max 0.093 vs window max 0.060 | — | **39 / 69** | the curve crosses the declared one |

F2's control: at θ₀ = 0 the TE and TM oracles are identical (0 ulp), so the swapped
reference must PASS — evaluated on the `te_00` arm and recorded as
`swap_ref_at_normal.e2_ok`. **Evaluated and not declared** (`FALSIFIERS_REJECTED`): the
brief's "+5° on the 45° arm" gives 0.94× its window and "+5° on the 30° arm" 1.07× — coin
tosses by cv22's Debye τ×1.3 rule; +5° on the 60° arm is 4.8× but a k_y of 65° puts the
run's cutoff at 9.06 GHz with 3.4e-2 of incident there, so the purity witness would fire
too and the reading would be ambiguous — the 55° defect (cutoff content 3e-6) is declared.
F5 as proposed in the brief was predicted (before the lattice was built) to be a
continuum-dominated non-failure; the exact lattice says the opposite — the discrete term of
a 10-cell profile at this dt/dx is 3.5× the 20-cell one — and the prediction is committed
here so the run can refute it.

Reading rules: an F1 exit 1 counts only with `G2_R = false`; a purity-only failure does
not. F5 must fail G6 on ≥ 60 of 69 bins; F5b on ≥ 20 (a falsifier failing on fewer bins
than predicted is a finding about the window, recorded). Unit-level (no FDTD,
`tests/test_cv26_oblique_fresnel_comparator.py`): every declared margin above is recomputed
from the module and asserted ≥ 2.9× (E2) / ≥ 60 and ≥ 20 bins (F5, F5b); the rejected ones are
asserted to be coin tosses.

## 9. Artifacts and keys (prose only until the run lands)

`validation/crossval/_26_oblique_results/rfx.json` (schema `cv26-oblique-slab/v1`):
`commit` (from `.staged_commit`), `arms.<arm>.{theta0_deg, pol, bw, ky_rad_m, f_cutoff_hz,
freqs_hz, theta_deg, gated, n_bins_gated, R_rfx, T_rfx, R_an, T_an, dR, dT, w_disp_*,
w_inj_*, window_*, phase_err_rad, max/mean_d{R,T}_gated, mean_window_*, worst_bin_*,
max_closure_gated, theta_gated_deg, tail.*, gates.{G1_R,G1_T,G2_R,G2_T,G3_closure,
G3_passivity,G3_tail}, gates_all, e2_ok, lattice.{R_lattice, T_lattice,
R/T_lattice_ideal_absorber, W_lat_*, mean_W_lat_*_gated, mean/max_d{R,T}_lattice_gated,
absorber_term_*, cpml3d_term_R_gated_max, aux_echo_term_R_gated_max}, run.{dt_s, n_steps,
nfft, nx_interior, n_cpml, n_cpml_run, dx_m, dx_div, record.{n_steps_min, n_steps,
extensions, cap_steps, cap_reached, n_pulse_end, n_ring, n_echo, …}, theta0_run_deg,
ky_run_rad_m, eps_slab_run, mu_slab_run, pec}, brewster.* (tm_60), swap_ref_at_normal
(te_00), leak.* (graze_vac), grazing_pec.* (graze_pec), grazing_slab.* (graze_te),
meep.{present, k_point, ky_meep_rad_m, precheck, R_meep, T_meep, d*_meep_tmm, d*_rfx_meep,
window4_*, window5_*, mean/max_*, gates.*, e4_ok}}`, `verdict.{rfx_self_ok, meep_present,
exit_code, summary}`. Falsifiers `rfx__falsifier_<name>.json` (seven, each exit 1);
ladders `rfx__graze_pec_d{8,16,32}.json`, `rfx__<arm>_dx1.json` (five); Meep
`meep_<arm>.json` (six) and `meep_te_45__falsifier_k_2pi.json`. Public numbers only as
`path.json::key = value`.

## 10. What the VESSL run owes, and what would refute this note

Owed: `rfx.json` with exit 0 (every gate on all ten arms), seven falsifier artifacts each
exit 1 for the declared reason, six Meep legs with `precheck.passed`, the ladders, the gate
test green on the produced set. Refutations accepted before the run: (i) an E2 gate failing
on a primary arm — then either the μ path (TM arms) or the Bloch update has a defect the
lattice witness will locate (|rfx − lattice| ≫ 3e-4 says the solver differs from its own
discrete model; ≤ 3e-4 with a gate failure says the §3.2 term exceeds the a-priori
prediction, which cannot happen unless the model is wrong — either is a finding, no window
widens); (ii) G6 failing — the absorber differs from its transcribed recursion, or the
record; (iii) the −40 dB witness not met inside the gate after box growth to 4× (a slower
component than derived); (iv) a falsifier exiting 0 — the gate does not resolve that
defect and the case is not claims-bearing for it; (v) the leakage bar failing — the
transformed-frame injection is not exact on this rig.

## 11. Scope statement (the row this case carries)

One slab (ε = 4, 10 mm, lossless), 2-D TMz, the `bloch` TFSF path with `ez` only
(`methodB` not exercised), fixed k_y per arm, realized angles 0 and 22.8–69.7° on the wide
rig (dx on the normal controls, dx/2 on the oblique arms), 80–85° on the compact box at dx;
TE injected, TM through the exact ε ↔ μ duality (rfx's μ update, not a p-polarized source);
Meep with a k_point Bloch boundary at 40 px/cm on the six wide-rig arms; the CPML's grazing
reflection gated against the exact discrete model of rfx's own absorber and injector.
Nothing about 3-D, other thicknesses or materials, angles above 70° on the wide rig, or
`methodB`.

## 12. Local rig checks before any VESSL arm (2026-09-03) — not evidence; what they closed

All ≤ 6 s wall on this Mac, the cv24 §12 discipline. None is an artifact of this case.

### 12.1 The injector and the interior are exact on the lattice

Vacuum, compact box, θ₀ = 0: |scat/inc| = **3.6e-7**, |T − 1| = 2.8e-7 — the
transformed-frame TFSF is float32-clean (L = 1e-3 has 10³ headroom). cv04's 600-cell box,
ε-slab, θ₀ = 0, cv04's time gate: rfx vs the absorber-free lattice **9.0e-6 mean /
8.0e-5 max** in R (cv23's class), both 0.0098 from the transfer matrix.

### 12.2 Dead end 1 — the vacuum arm cannot see the absorber

The brief's "excess |R| beyond Fresnel attributable to the absorber" was first designed as
a vacuum compact-box arm measuring |r_pml| directly. The lattice gave r = 1e-14 at every
absorber strength including NO absorber (PEC/PMC ends): in a vacuum run the total-field
region holds the incident and the scattered-field region is identically zero — the TFSF
removes the wave at x_hi. The absorber is hit only by SCATTERED waves. The PEC arm replaced
it (§4.5) and the vacuum arm became the leakage witness.

### 12.3 Dead end 2 — the first absorber model missed the injector's own echo

With the PEC arm the compact-box run showed R − 1 excursions of −0.21…+0.29 at NORMAL
incidence (the 20-cell CPML reflects ~10 % in amplitude on this box: cv04 time-gates it out
for a reason), and my lattice with a plane-wave incident reproduced the class but not the
bins (mean |ΔR| 0.070). A numpy replica of rfx's step order (which matched rfx to 1e-6 once
its own bug — the missing 1/dx in the face coefficients — was found) and a CW steady-state
node-by-node comparison (5e-7 agreement inside the absorber cells) proved the CPML
transcription exact. The remaining term was the aux grid: its hi-end absorber (§3.2)
returns a −x echo into the injected incident, 0.7e-3–0.17 in R on the compact box, gated
out on every wide-rig record. Modelling the aux lattice exactly closed it: PEC@0°
**1.7e-4 mean / 6.5e-4 max**, slab@0° 1.3e-4, PEC@60° 6.8e-4 mean.

### 12.4 The grazing PEC arm itself, and the bandwidth rule

`graze_pec` (82°, 20-cell absorber) at dx: 22 001 steps (5 extensions), tails 6.5e-4 /
2.8e-4, purity 9.5e-4, **|R − R_lattice| 3.9e-4 mean / 1.7e-3 max** over the 69 gated
bins; measured excess |R − 1| up to 0.2725 against the a-priori 0.2724 (3-D 0.115, aux
0.174). With my first bandwidth rule (1e-2 at cutoff) the
60° PEC arm hit the record cap with the purity witness at 5.3e-3 (§2's reason); at 1e-3 it
settled at 9.7e-4 with 4 extensions.

### 12.5 The dx/2 rig

PEC@60° at dx/2: |R − R_lattice| **4.8e-4 mean / 3.3e-3 max**; the μ-slab (TM dual) at
60°, dx/2, compact: R 1.3e-4, T 1.0e-3 max — the μ update on the Bloch path equals its
lattice model. Two threading bugs were caught by this check and fixed before the freeze:
the grazing evaluators read dx = 1 mm regardless of the recipe, and the lattice took the
declared depth (20) where the refined rig runs 40 cells. Both are now derived from the rig
bookkeeping (`rig_cells(dx_div=K)`), never restated.

### 12.6 What the checks did NOT do

No wide-rig oblique arm was run locally (te_60 at dx/2 is ~30 s on the pod, above the
local budget); their predictions (§3.2, §6) stand untested until VESSL. The Meep leg was
not executed anywhere (no Meep on this Mac); its k_point mapping is unit-tested only.
