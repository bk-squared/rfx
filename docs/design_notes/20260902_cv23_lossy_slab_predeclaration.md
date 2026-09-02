# cv23 lossy-slab Fresnel — pre-declaration (gap lane 2, dielectric loss)

Date: 2026-09-02 · Lane: `agent/gap2-lossy-slab` (stacked on
`agent/gap1-dispersive-slab`, the cv22 lane) · Case:
`validation/crossval/23_lossy_slab_fresnel.py` (new; id
`23_lossy_slab_fresnel`; claims-bearing, E2 + E4).

**Append-only.** Corrections are added as new sections; nothing above a
correction is edited. Every number in this note is ANALYTIC (closed-form
ε(f), transfer matrix, the discrete-time transfer function of the σ update)
or READ from a committed cv04/cv22 artifact or the cv04 golden fixture. No
FDTD has been run for this case at the time of this commit; measured numbers
arrive later, by artifact key. The note follows the cv22 note
(`docs/design_notes/20260902_cv22_dispersive_slab_predeclaration.md`) and
inherits its rules verbatim where it says so — in particular §14's "the
witness is the gate, the derivation is the estimate" and §10's "a falsifier is
judged against the DECLARED material".

## 0. Why this case exists

Dielectric loss — `sigma` in `Simulation.add_material(name, eps_r=, sigma=)`
(`rfx/api/__init__.py:871`, documented at
`docs/public/guide/materials-geometry.mdx:57-64`), realized by
`materials.sigma` in `rfx/core/yee.py::update_e` — is never the gated
observable anywhere in the crossval campaign:

- cv15 (`15_patch_antenna_rt5880.py:361`, `add_material("sub", eps_r=EPS_R,
  sigma=SIGMA_SUB)`) is the only lossy crossval case and its loss is not
  gated (resonance / |S11| envelope only);
- the Leontovich-α and sheet-Q oracles are xfail;
- there is no external-solver (E4) evidence for a lossy dielectric at all;
- cv22 (the lane this one stacks on) explicitly excludes "conductivity-plus-
  pole" from its scope and never sets `materials.sigma`.

This case gates R(f), T(f) AND the absorption A(f) = 1 − R − T of a
conductive slab against the analytic transfer matrix with the complex
ε(f) = ε' − jσ/(ωε0) (E2) and against Meep's `D_conductivity` (E4), on the
cv22/cv04 rig where the rig's own discretization error is a committed number.

## 1. Rig — cv22's round-4 recipe, unchanged

`validation/crossval/22_dispersive_slab_fresnel.py` PART 1 = cv04 PART 1:
2-D TMz `Grid(freq_max=20e9, domain=(nx·dx, 0.004, dx), dx=1e-3,
cpml_layers=20, mode="2d_tmz")`, `dt = 2.335067793382187e-12 s` (Courant
0.700), TFSF `+x`, `polarization="ez"`, `f0 = 10 GHz`, `bandwidth = 0.5`,
differentiated-Gaussian incident (τ = 63.66 ps, t0 = 3τ; amplitude peak at
3.5 GHz), slab `d = 10 mm` at the domain centre, probes 30 cells either side,
1-D auxiliary incident reference, rFFT oversampled ×8, cv04's mask
(3 GHz < f < 15 GHz, incident amplitude > 2 % of peak), `nx_interior = 1000`
(cv22 §12: the smallest round box whose CPML round-trip gate, 1262 steps,
exceeds the longest derived record), record length DERIVED per arm from the
slab's own ring-down over the incident-weighted ring band (§8 below; cv22
§13), the −40 dB settling witness (`SETTLING_LIMIT = 1e-2`) with adaptive
extension (`RECORD_EXTEND_STEPS = 100`) and box growth (`NX_GROW_CELLS =
200`) instead of clipping, cv04's 1e-3 tail-purity witness, the stored
300-step tail envelope and its post-pulse fitted rate (cv22 §15.1). All of
this is imported from `validation/crossval/comparators/cv22_dispersive_gates.py`;
none of its constants is restated.

The ONLY change from cv22 is the slab material: `eps_r = ε' = 4` (cv04's
slab) and `sigma = σ` in the slab cells, with the ORDINARY `update_e`
(`rfx/core/yee.py:347`) in the loop — no ADE. Every arm is one FDTD run.
Meep: cv22's leg geometry (a = 1 cm, 60 × 0.4 cm cell, 2 cm PML in x,
periodic in y, slab 1 cm at the centre, flux monitors 3 cm either side,
two-run reference subtraction) with `Medium(epsilon=ε', D_conductivity=σ_D)`
at 40 px/cm as the primary reference (cv22 §12: the converged rung of the
measured first-order ladder) plus the 10/20/40 ladder. `eps_averaging` is
set False on both Meep passes (Meep's own documentation: conductivity "is
not compatible with subpixel averaging"); the slab faces fall on pixel
boundaries at all three resolutions (0.5 cm = 5 / 10 / 20 px), so this
changes nothing about where the interface is and is stated, not measured.

## 2. Material arms — σ chosen from tan δ at the band centre

Definition: `tan δ = σ/(ω ε0 ε')` at `f_c = 7 GHz` with `ε' = 4`, so
`σ = tan δ · 2π f_c ε0 ε' = tan δ × 1.5577 S/m`
(`dispersive_eps.sigma_from_tan_delta`; ε0 = 8.8541878128e-12 as in
`rfx/core/yee.py`). tan δ is then ∝ 1/f across the band.

| arm | tan δ @ 7 GHz | **σ (S/m)** | tan δ over 4–10 GHz | skin depth 1/(k0\|Im n\|) at 4 / 7 / 10 GHz | d/δ_skin @ 7 GHz | material path |
|---|---|---|---|---|---|---|
| `tand0p1` | 0.1 | **0.15577** | 0.174 → 0.070 | 68.4 / 68.2 / 68.2 mm | 0.15 | direct: `init_materials` + `.at[slab].set(σ)` (cv22's construction) |
| `tand1` | 1 | **1.5577** | 1.739 → 0.701 | 8.37 / 7.49 / 7.18 mm | 1.3 | **documented user path**: `Simulation.add_material("lossy_slab", eps_r=4.0, sigma=σ)` + `Simulation.add(Box(...), material=...)`, arrays assembled by the API |
| `tand3` | 3 | **4.6731** | 5.217 → 2.103 | 4.05 / 3.28 / 2.93 mm | 3.1 | documented user path, as `tand1` |

(For a conductor `k0·Im n ≈ σ Z0/(2√ε')` is frequency-independent, which is
why the low-loss skin depth barely moves across the band.)

Analytic R, T, A on the gated bins (§5), transfer matrix with the continuous
complex ε(f):

| arm | R | T | A = 1 − R − T | what the arm discriminates |
|---|---|---|---|---|
| `tand0p1` | 0.009 – 0.298 | 0.507 – 0.702 | 0.193 – 0.293 | δ_skin ≫ d: a lossy perturbation of cv04's lossless etalon — the fringes survive (R spans 0.01–0.30) and A is the absorption per pass integrated over the multiple reflections; a wrong σ shows first in T |
| `tand1` | 0.174 – 0.331 | 0.046 – 0.061 | 0.608 – 0.777 | δ_skin ≈ d: the transition regime where both faces still matter (double-pass attenuation e^{−2d/δ} ≈ 0.07); A is maximal; R and T both carry the discrimination |
| `tand3` | 0.333 – 0.512 | 0.0007 – 0.003 | 0.485 – 0.666 | δ_skin ≈ d/3: the back face is invisible (e^{−2d/δ} ≈ 2.5e-3), R is set by the surface impedance of a semi-infinite conductor, A = 1 − R. **The T gate on this arm is vacuous** (T < W_mean_T everywhere); R and A are the observables — stated here, not discovered later |

**Material paths, stated per arm.** `tand0p1` builds the arrays exactly as
cv22 and cv04 do (`init_materials(grid.shape)` then `.at[slab].set`) — it is
the control closest to cv04. `tand1` and `tand3` build them through the
documented `Simulation.add_material(...)` → `Simulation.add(Box, material=)`
→ `Simulation._assemble_materials(grid)` path (the registration, geometry
rasterization and CPML-pad extension the user's `run()` goes through) and
run the same loop. What the API arms exercise is therefore the material
REGISTRATION and FILL wiring; the time-stepping is the same `update_e` in all
three arms, and `Simulation.run()`'s own scan body is NOT exercised (its TFSF
source and probes are a different rig; the cv04 envelope would not transfer).
Checked analytically before any run and locked by the gate test: for every
arm the API-assembled `eps_r`, `sigma`, `mu_r` arrays equal the direct arrays
bit-for-bit on the nx-1000 grid (10 slab cells, [515, 525)), the `Grid` the
API builds has the same shape and dt, and no PEC cell is produced. The
`Box` is drawn in lattice arithmetic (the exact node coordinates of cells
515 and 525, half-open `[lo, hi)`, `rfx/geometry/csg.py:86-110`).

**No fourth arm.** rfx exposes no loss-tangent parameter: `add_material`'s
loss parameters are `sigma` only (`rfx/api/_spec.py:1062` `MaterialSpec`),
and the library laminates carry a fixed effective σ derived offline from a
datasheet tan δ (`materials-geometry.mdx:28-31,40`). A tan δ-at-f0 arm
would exercise nothing rfx has; the arm definition above IS "σ from tan δ at
a reference frequency", done in the comparator.

### 2.1 Stability and the size of the σ update coefficient

`update_e` (`rfx/core/yee.py:388-393`): `s = σ dt/(2 ε0 ε')`,
`ca = (1 − s)/(1 + s)`, `cb = (dt/ε)/(1 + s)`. For σ ≥ 0, `0 < ca < 1`:
unconditionally stable in the material term; the Courant condition is cv04's.

| arm | s | ca |
|---|---|---|
| `tand0p1` | 5.14e-3 | 0.9898 |
| `tand1` | 5.14e-2 | 0.9022 |
| `tand3` | 1.54e-1 | 0.7331 |

Meep's `D_conductivity` update is the same semi-implicit form (§7) and is
likewise unconditionally stable for σ_D ≥ 0; there is no pole, so cv22's
Nyquist-permittivity instability (F-B) has no analogue here.

## 3. The new physics term: temporal discretization of the σ update

Derived from the coefficients at `rfx/core/yee.py:388-393`, which are
algebraically

    ε0ε' (E^{n+1} − E^n)/dt + σ (E^{n+1} + E^n)/2 = curl H^{n+1/2}

(substitute `ca`, `cb`: `E^{n+1}(1 + s) = (1 − s) E^n + (dt/ε) C`). With
`z = e^{jω dt}` and `x = ω dt/2`:

    E/C = (dt/ε) / [(1 + s) z^{1/2} − (1 − s) z^{−1/2}]
        = 1 / [ε0ε' · 2j sin x/dt  +  σ cos x]
        = 1 / (j ω̂ ε0 ε_num),      ω̂ = 2 sin x/dt  (the Yee temporal factor)

so, measured against the same common Yee factor cv22 §3 uses for the ADEs,

    ε_num(ω) = ε' − j σ_eff(ω)/(ω ε0),    σ_eff = σ · x/tan x,   x = ω dt/2.

The semi-implicit average under-realizes σ by `1 − x/tan x ≈ x²/3`:
**−1.79e-3 at 10 GHz** (x = 0.0734), −8.8e-4 at 7 GHz, −2.9e-4 at 4 GHz.
Meep's update (§7) has the identical form at `dt_meep = 0.5·(a/40)/c =
4.170e-13 s`: x = 0.0131 at 10 GHz, `x²/3 = 5.7e-5`. The function
`dispersive_eps.sigma_warp` carries the factor; the gate test witnesses it
against the LIVE `update_e` (the coefficients are extracted from one update
on a tiny grid and the scalar recurrence driven sinusoidally reproduces
`1/(jω̂ ε0 ε_num)`).

The σ-update window term is the exact propagation of `ε_num − ε` through the
transfer matrix, per bin, for all three observables:

    W_σ,R(f) = |R_TMM(ε_num) − R_TMM(ε)|,  W_σ,T likewise,
    W_σ,A(f) = |A_TMM(ε_num) − A_TMM(ε)|,  A_TMM = 1 − R_TMM − T_TMM.

On the rfx bin grid (nfft 16384, 229 gated bins), rfx dt:

| arm | max W_σ,R | max W_σ,T | max W_σ,A | mean W_σ,R | mean W_σ,T | mean W_σ,A |
|---|---|---|---|---|---|---|
| `tand0p1` | 8.5e-5 | 2.8e-4 | 3.5e-4 | 2.0e-5 | 1.8e-4 | 2.0e-4 |
| `tand1` | 1.8e-4 | 2.1e-4 | 9.1e-5 | 1.2e-4 | 1.2e-4 | 2.5e-5 |
| `tand3` | 3.5e-4 | 6.4e-6 | 3.5e-4 | 1.9e-4 | 4.7e-6 | 1.9e-4 |

Meep side at 40 px/cm: max 1.1e-5 (R), 9.0e-6 (T), 1.1e-5 (A). As in cv22,
the term is 2 orders below the rig term at this dt and is named and carried
anyway; it is not negligible at a coarser dt or a higher band.

## 4. Windows — derived, not chosen

Same rig, same committed cv04 envelope, same repo rule
(`tests/_gate_policy.py::gate_from_envelope`, × 1.5, quantum 1000); the
numbers are cv22 §4's and are imported, not restated:

    W_bin    = 0.074   (per-bin, from cv04's max|R+T−1| = 0.0487, 04_multilayer_fresnel.py:309)
    W_mean,R = 0.010   (from multilayer_fresnel.json::mean_reflectance_error = 0.0066)
    W_mean,T = 0.017   (from ...::mean_transmittance_error = 0.011)

**Absorption windows.** A = 1 − R − T, so `|ΔA| ≤ |ΔR| + |ΔT|` bin by bin and
in the mean; the DECLARED A windows are the triangle-inequality sums:

    W_bin,A  = 2·W_bin            = 0.148
    W_mean,A = W_mean,R + W_mean,T = 0.027

plus `W_σ,A(f)` per bin and its gated mean. A tighter A window IS derivable
and is stated here without being chosen: cv04's closure `|R + T − 1|` on a
lossless slab is a direct measurement of ΔA (A ≡ 0 there), so its committed
per-bin envelope 0.0487 gives `gate_from_envelope = 0.074` per bin, and the
committed band-mean closure
`tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[id=mean_energy_closure_error].observed_baseline = 0.0091`
gives 0.014 band-mean — about half the triangle windows. Both derivations
rest on the same cv04 numbers; the triangle one is the rigorous bound given
the R and T gates and is the gate; the closure-based one assumes the
lossless closure error transfers to a lossy slab's A, which this case will
measure but has not. The artifact records the closure-based verdict as
`A_tight_ok` (per arm, ungated) so the next lane can decide from evidence.

### E2 gates (rfx vs analytic TMM with the continuous complex ε(f)), per arm

- **G1 per-bin**, every gated bin: `|R_rfx − R_TMM| ≤ W_bin + W_σ,R(f)`,
  T likewise, and `|A_rfx − A_TMM| ≤ W_bin,A + W_σ,A(f)`.
- **G2 band-mean**: `mean|ΔR| ≤ W_mean,R + mean W_σ,R`, T likewise,
  `mean|ΔA| ≤ W_mean,A + mean W_σ,A`. Numerically (R / T / A):
  `tand0p1` 0.01002 / 0.01718 / 0.02720; `tand1` 0.01012 / 0.01712 / 0.02702;
  `tand3` 0.01019 / 0.01700 / 0.02719.
- **G3 witnesses**: the −40 dB settling witness on both tails, cv04's 1e-3
  purity, passivity `R + T ≤ 1 + CONS_MAX_LIMIT` (0.06) at every masked
  bin, and the rig floor (incident amplitude ≥ 5 % of peak over the gated
  band — 8.6 % at 10 GHz, analytic).

### E4 gates (Meep present), per arm

- **G4 reference soundness**, Meep vs TMM: per-bin
  `|R_meep − R_TMM| ≤ W_bin + W_σ,meep,R(f)` (T likewise, A with `W_bin,A`);
  band-mean `≤ W_mean + mean W_σ,meep`. There is no mapping residual
  (`W_map ≡ 0`: `D_conductivity` realizes ε'(1 + iσ_D/ω) exactly, §7).
- **G5 rfx vs Meep** (triangle over G1 and G4): per-bin
  `≤ 2·W_bin + W_σ,R + W_σ,meep,R` (A: `2·W_bin,A + …`); band-mean
  `≤ 2·W_mean + …` (R 0.020, T 0.034, A 0.054 + small).
- `precheck.passed` (the leg's 1e-9 pre-run check, §7) is a gate, as in
  cv22 §15.2; the Meep-side windows are derived from the DECLARED material
  mapped by `to_meep`, never from what the JSON reports.

Exit contract (cv22's): 0 = G1–G5 pass on all three arms; 1 = any gate
fails; 2 = E2 passes but a Meep JSON is missing for any arm (inconclusive).

## 5. Gated bins

Every arm's derived record (§8) lands on `nfft = 16384` (df = 26.14 MHz):
**229 gated bins, 4.025–9.985 GHz**, one shared bin grid. cv04's mask edges
(3–4 and 10–11.9 GHz) are reported, not gated, for cv22 §5's reasons.

## 6. Falsifiers — pre-declared, with analytic margins

Each is a `--falsifier <name>` arm of the case script or of the Meep leg,
run on VESSL, and MUST exit 1. The FDTD is built with the defect and judged
against the DECLARED material (cv22 §10.1). Margins are the analytic
`mean|Δ|` of the defective vs the declared ε(f) through the same TMM on the
gated bins, over the band-mean window; "bins" counts gated bins where G1
must fail on its own.

**F1 — σ × 1.5.**

| name | mean\|ΔR\| / W (×) | mean\|ΔT\| / W (×) | mean\|ΔA\| / W (×) | bins > W (R, T, A) | worst bin |
|---|---|---|---|---|---|
| `tand0p1_sigma_x1p5` | 0.0099 (**0.99×, a coin toss — R does not carry this one**) | 0.0862 (5.1×) | 0.0928 (3.4×) | 0, 160, 0 | T 0.110 at 7.55 GHz |
| `tand1_sigma_x1p5` | 0.0655 (6.5×) | 0.0353 (2.1×) | 0.0302 (1.1×, coin toss on A) | 73, 0, 0 | R 0.079 at 6.59 GHz |
| `tand3_sigma_x1p5` | 0.0810 (8.1×) | 0.0012 (0.1×, T vacuous) | 0.0798 (3.0×) | 229, 0, 0 | R 0.083 at 9.98 GHz |

Each F1 arm fails G2 with ≥ 2× margin on at least one observable (the
cv22 coin-toss guard); the observable that carries it is named above.

**F2 — σ × 0** (cv04's lossless ε = 4 slab, judged as lossy; A_TMM > 0.19
everywhere while the run's A ≡ 0):

| name | mean\|ΔR\| (×) | mean\|ΔT\| (×) | mean\|ΔA\| (×) | bins > W (R, T, A) |
|---|---|---|---|---|
| `tand0p1_sigma_zero` | 0.031 (3.1×) | 0.219 (12.9×) | 0.248 (9.2×) | 0, 229, 229 |
| `tand1_sigma_zero` | 0.080 (8.0×) | 0.783 (46×) | 0.740 (27×) | 106, 229, 229 |
| `tand3_sigma_zero` | 0.247 (25×) | 0.835 (49×) | 0.588 (22×) | 218, 229, 229 |

**F3 — σ mapped into Meep with the WRONG scaling** (this repo's round-1
failure class), on the `tand1` arm of the Meep leg
(`scripts/crossval/meep_cv23_lossy_slab.py --falsifier …`); the 1e-9
pre-check is recorded as failed and the run proceeds so G4/G5 are exercised
on real Meep output:

| name | defect | pre-check rel. err (must be ≫ 1e-9) | mean\|ΔR\| / 2W_mean,R | mean\|ΔT\| / 2W_mean,T | mean\|ΔA\| / 2W_mean,A | bins > 2·W_bin (R) |
|---|---|---|---|---|---|---|
| `meep_tand1_sigma_2pi` | `σ_D × 2π` (σ_D taken in units of 2πc/a) | 4.58 | 0.347 (17×) | 0.053 (1.6×) | 0.294 (5.4×) | 229 |
| `meep_tand1_sigma_no_eps` | `σ_D = σ a/(c ε0)` (ε' division dropped: σ applied to E, not D) | 2.60 | 0.261 (13×) | 0.052 (1.5×) | 0.209 (3.9×) | 229 |

Unit level (`tests/test_cv23_lossy_eps_mapping.py`, no FDTD): the 1e-9
mapping test passes for the correct σ_D on all three arms and FAILS for
`× 2π`, the dropped ε', a dropped `a/c` unit scale, and a dropped
`e^{−iωt}` conjugation.

**F4 — passivity: an arm with R + T > 1 must fail.** Built as a GAIN
medium: `tand0p1` with `σ → −σ` (`tand0p1_sigma_neg`, σ = −0.15577 S/m).
Analytically `R + T = 1.25 – 1.49` at every gated bin (A = −0.25 … −0.49),
so G3_passivity fails at all 229 bins and the A gates fail by 23× in the
mean; it is stable: the gain slab is below its lasing threshold
(`σ d Z0/2 = 0.29 < 1`: the k = 0 slab mode radiates faster than it is
pumped; per round trip ρ = |r|² e^{+2k0|Im n|d} ≤ 0.21 across the ring
band, derived record 1106 steps, §8), and `ca = 1.0103` in the slab cells.
Refutation accepted: if the FDTD of this arm is non-finite the artifact is
not written and the arm is recorded as "unstable, not a falsifier" — that
would say the gain construction is not usable on this rig, not that the
passivity gate is wrong; a replacement passivity construction would then be
pre-declared in a new section before any run.

Found while deriving F4 (fixed in this lane, cv22 numbers unchanged):
`cv22_dispersive_gates.slab_ringdown_rates` selected the sqrt branch with
`where(n.imag > 0, −n, n)`, a no-op for every passive arm but a negation of
Re n for a gain medium (|r|² = 9, ρ = 6.7). It now keeps `Re n ≥ 0`, writes
ρ as `|r|² e^{2 k0 Im(n) d}`, and raises if ρ ≥ 1 anywhere in the ring band.
cv22's committed records replay unchanged (1108 / 1228 / 1168).

## 7. Meep mapping (the module is the authority: `dispersive_eps.to_meep`)

Meep (`e^{−iωt}`, frequencies in c/a, a = 1 cm): `Medium(epsilon=ε',
D_conductivity=σ_D)` realizes, in `python/geom.py::Medium._get_epsmu`,
`epsmu = (1 + 1j/(2π·freqs)·conductivity)·epsmu`, i.e.

    ε_meep(ω) = ε' (1 + i σ_D / ω_m),   ω_m = 2π f_m,  f_m = f·a/c,  σ_D in c/a.

Its time stepping (`src/step_generic.cpp` `step_curl`, with
`condinv = 1/(1 + σ_D·dt/2)` from `src/structure.cpp`):
`D ← ((1 − σ_D dt/2)·D + dt·curl H)/(1 + σ_D dt/2)` — the same semi-implicit
average as rfx's `update_e`, applied to D. Since D = ε'E for a dispersionless
ε', this is exactly rfx's update with σ = σ_D ε0 ε' c/a, and Meep's
discrete-time term is the same `x/tan x` factor at Meep's dt (§3).

Matching `conj(ε_rfx) = ε' + iσ/(ω ε0) = ε'(1 + iσ/(ω ε0 ε'))`:

    σ_D = σ · a / (c ε0 ε')           (dimensionless; units of c/a)

`tand0p1` 0.146709, `tand1` 1.467092, `tand3` 4.401275. The two traps are the
two F3 arms: dividing by ε' (D, not E) and the 2π (frequency unit c/a, not
2πc/a). The Meep leg evaluates the mapped ε at 4.5 / 7 / 9.5 GHz through
`Medium.epsilon(f)` (Meep's OWN evaluation of its OWN convention, which
includes the conductivity term) and through the comparator's
reconstruction, against `eps_analytic`, and aborts above 1e-9 relative
unless `--falsifier`. Comparison is always `conj(ε_meep) == ε_rfx`.

Ladder: 10 / 20 / 40 px/cm on all three arms (no instability witness is
expected — no pole); the summary records Meep's convergence order as in
cv22 §12.2, as evidence, not as a window term.

## 8. Record length — derived per arm before the run; the witness is the gate

cv22 §13.3's derivation (`derive_record_length`) with the loss included in
the etalon rate and NO material pole: `J = σE` is memoryless (no P
recurrence, no material ring-down mode; the charge-relaxation rate
σ/(ε0ε') is a longitudinal mode normal incidence does not excite), so
`rate(f) = etalon(f)` with `ρ(f) = |r|² e^{2 k0 Im(n) d}` — absorption per
pass is what the loss adds. Over the incident ring band (1.13–15 GHz,
w ≥ 0.5), each component needs `ln(100·w(f))/rate(f)`; the slowest is the
bottom of the band for every lossy arm (absorption per pass and |r|² both
favour low f there). At dt = 2.335 ps on the nx-1000 rig
(`n_pulse_end = 908`, CPML gate 1262):

| arm | governing component | ρ per round trip | rate (s⁻¹) | n_ring | **n_steps_min** | nfft |
|---|---|---|---|---|---|---|
| `tand0p1` | etalon at 1.13 GHz, w = 0.50 | 0.117 | 1.54e10 | 109 | **1067** | 16384 |
| `tand1` | etalon at 1.13 GHz | 0.118 | 8.40e9 | 200 | **1158** | 16384 |
| `tand3` | etalon at 1.13 GHz | 0.043 | 7.52e9 | 223 | **1181** | 16384 |
| falsifiers | `*_sigma_x1p5`: 1078 / 1169 / 1186; `*_sigma_zero`: 1078 (cv04's lossless etalon, 1.65e10 s⁻¹, governed by w = 1 at 3.54 GHz); `tand0p1_sigma_neg`: 1106 (ρ = 0.21, 1.14e10 s⁻¹) | | | | | |

(Ordering note: the higher-loss arms need LONGER records although they
absorb more per pass, because their |n| at 1.13 GHz is larger — the round
trip is slower and the low-frequency |r|² is higher; this is the estimate,
recorded so its miss can be measured.)

**Witness predictions (before the run):** every arm meets the −40 dB bar at
`n_steps_min` with **0 extensions**; the measured tails land BELOW the bar
by the ratio of the true starting level to the assumed `w` — the scattered
start is ≤ √R(1.13 GHz)·w and the transmitted start ≤ √T·w — so
**2e-3 – 7e-3 (−43 to −54 dB)**, `tand0p1` closest to the bar (its etalon is
the least damped, as cv04's), `tand3` far below on the transmitted side.
Fitted post-pulse tail rates: between the 1.13 GHz etalon rate above and the
7 GHz etalon rate (2–4e10 s⁻¹); a fitted rate BELOW the 1.13 GHz value means
a slower component than derived (cv22 §14.1's Debye class) and is a
finding; the adaptive extension, not the estimate, then guarantees the
record. If an extension fires the record grows by 100 to at most 1262; box
growth is not expected.

## 9. Artifacts and keys (to be filled by the VESSL run; prose only until then)

- `validation/crossval/_23_lossy_results/rfx.json` — schema
  `cv23-lossy-slab/v1`, cv22's schema plus the absorption: per arm
  `freqs_hz`, `gated`, `R_rfx`, `T_rfx`, `A_rfx`, `R_tmm`, `T_tmm`, `A_tmm`,
  `R_tmm_ade`, `T_tmm_ade`, `A_tmm_ade`, `dR`, `dT`, `dA`, `window_R`,
  `window_T`, `window_A`, `w_ade_{R,T,A}`, `max_d{R,T,A}_gated`,
  `mean_d{R,T,A}_gated`, `mean_window_{R,T,A}`, `worst_bin_{R,T,A}_hz`,
  `A_tight_ok`, `materials_path` ("direct" | "api"), `run.record.*`,
  `tail.*`, `gates.{G1_R,G1_T,G1_A,G2_R,G2_T,G2_A,G3_passivity,G3_tail}`,
  `meep.{present, precheck, R_meep, T_meep, A_meep, d{R,T,A}_meep_tmm,
  d{R,T,A}_rfx_meep, window4_*, window5_*, gates.{precheck_passed,
  band_covered, G4_*, G5_*}}`; top-level `verdict.{rfx_self_ok,
  meep_present, e4_ok, exit_code}` and `commit` (from `.staged_commit`).
- `…/rfx__falsifier_<name>.json` — seven rfx-side arms (F1, F2, F4) plus the
  two `meep_tand1_*` arms (rfx `tand1` correct, wrong-scaling Meep JSON
  read), each `verdict.exit_code = 1`.
- `…/meep_<arm>.json` (three, 40 px/cm), `…/meep_tand1__falsifier_<x>.json`
  (two), `…/meep_<arm>__res{10,20,40}.json` (nine), `meep_ladder_summary.json`.
- Public numbers, once measured, only as `path.json::key = value`.

## 10. What the VESSL run owes, and what would refute this note

Owed: the three Meep primaries with `precheck.passed = true`, `rfx.json`
with exit 0, nine falsifier `rfx.json`s each with exit 1, the ladder summary,
the gate test green on the committed set.

Refutations accepted: (i) the baseline fails G1/G2 on any arm or observable
— then either `update_e`'s σ path has a defect the unit tests do not see,
or the cv04 envelope does not transfer to a lossy slab's A as §4 assumed;
either is a finding and no window is widened; (ii) a falsifier exits 0 — the
gate does not resolve the declared defect and the case is not claims-bearing
for that observable; (iii) the pre-run ε check fails at 1e-9 on a
non-falsifier leg — the σ_D mapping is wrong and nothing downstream counts;
(iv) a −40 dB witness not met after growth to 4× the box — a slower mode
than derived; (v) `tand0p1_sigma_neg` non-finite — the passivity
construction is unusable here (§6, F4).
