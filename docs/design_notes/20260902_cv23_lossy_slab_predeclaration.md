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

## 11. Addendum (2026-09-02, same day, before any measurement) — three corrections found while implementing

None changes a window, an arm, a band, a falsifier or a record; all were
found while writing the comparator, the leg and the gate test against this
note, before the first FDTD run.

1. **§6 F3 pre-check column.** The 4.58 / 2.60 quoted there are the maximum
   relative ε errors over the 229 GATED bins; the leg's pre-check evaluates
   at its three fixed frequencies (4.5 / 7 / 9.5 GHz, cv22's
   `PRECHECK_FREQS_HZ`) and reads **4.44** (`σ_D × 2π`) and **2.52** (ε'
   dropped) there — both ≫ 1e-9, the verdict is unchanged. The unit-level
   twin asserts > 1 on the same seven test frequencies. Verified end-to-end
   on a structural Meep stand-in (Meep's `Medium.epsilon` formula, canned
   fluxes): the correct mapping pre-checks at 1.6e-16 through the same call
   the pod will make.
2. **§6 F1 `tand3_sigma_x1p5` R margin.** 8.1× is the ratio to `W_mean,R =
   0.010`; the gate's band-mean window includes the arm's mean `W_σ,R`
   (1.9e-4), so the ratio the test locks is **7.95×** of 0.01019. Same
   arithmetic, same verdict; the gate test asserts > 7.
3. **cv22's leg refactor.** `run_slab_two_pass` was extracted from
   `meep_cv22_dispersive_slab.py::main` so cv23's leg shares the geometry
   and the two-run subtraction rather than copying them; the cv22 leg's
   behaviour is unchanged (default `eps_averaging` left to Meep; the
   stand-in reproduces its committed `gamma_half` pre-check value 0.914 =
   `validation/crossval/_22_dispersive_results/meep_lorentz__falsifier_gamma_half.json::precheck.max_rel_err = 0.91`).
   Likewise `run_rfx_arm` became `comparators/slab_rig.py::run_slab_arm`
   with the material as a hook; cv22's smoke reproduces its rig line
   (grid, dt, slab cells, probes, record) exactly.

Smoke (`--smoke`, 200-cell box, not evidence): all three arms execute in
0.3 s each, the API-path arms assemble and match the direct arrays, the
derived record and the adaptive extension run, and the gain arm
`tand0p1_sigma_neg` is finite on the tiny box with `max R+T = 1.76` and
`G3_passivity = False` — the F4 construction is usable on this rig as far as
the smoke can say; the −40 dB witness necessarily fails there (the box's
CPML gate is 176 steps), as in cv22's smoke.

## 12. Round 1 read (VESSL 369367257813) — two gates fired; the rfx residual is the Yee lattice's own term, derived exactly; round 2 pre-declared

Artifacts (world-readable): `~/mnt/remilab-fs/personal-workspaces/claude-workspace/rfx/runs/cv23-lossy-20260902T115843Z/_23_lossy_results/` (`r1/` below); log `~/Documents/vessl-run-logs/369367257813_cv23-lossy-r1.log`; ran the tree at `04d6e88` (= `3400143` after the coordinator's rebase; `r1/rfx.json::commit`). Every number is by key. No artifact is committed by this section (the baseline fails; round 2 re-runs the full set); no window moves.

### 12.1 What fired

| arm | E2 (`r1/rfx.json::arms.<arm>`) | mean\|ΔR\| / \|ΔT\| / \|ΔA\| vs windows | signed ΔR at 4 / 7 / 10 GHz | witness | E4 (Meep 40 px/cm) |
|---|---|---|---|---|---|
| tand0p1 (direct) | PASS | 0.0039 / 0.0057 / 0.0020 vs 0.0100 / 0.0172 / 0.0272 (`A_tight_ok` true) | +0.0016 / −0.0007 / +0.0140 | 1067 steps, 0 ext., tails 8.9e-4 / 1.0e-3 | **G4_mean_R FAIL**: Meep-vs-TMM `::meep.mean_dR_meep_tmm_gated = 0.0157` vs 0.0100 (1.57×; the coordinator's 0.0106 was the leg's diagnostic on Meep's own bins); T 0.0113 / A 0.0066 pass; rfx-vs-Meep 0.0127 / 0.0076 / 0.0059 pass (G5) |
| tand1 (api) | PASS | 0.0051 / 0.0018 / 0.0033 | +0.0026 / +0.0050 / +0.0077 | 1158, 0 ext., 9.8e-5 / 2.8e-5 | all pass: Meep-vs-TMM 0.0031 / 0.0035 / 0.0033; rfx-vs-Meep 0.0045 / 0.0018 / 0.0063 |
| tand3 (api) | **G2_R FAIL**: `::mean_dR_gated = 0.0126` vs `::mean_window_R = 0.0102` (1.24×); per-bin max 0.0190 < 0.074; T 0.0001 (vacuous, as declared); A 0.0124 vs 0.0272 pass, `A_tight_ok` true | | **+0.0067 / +0.0124 / +0.0190** (monotone, ∝ f) | 1181, 0 ext., 8.7e-5 / 1.0e-5 | all pass — but read the numbers: **Meep-vs-TMM `::meep.mean_dR_meep_tmm_gated = 0.0009`**, rfx-vs-Meep `::meep.mean_dR_rfx_meep_gated = 0.0117`. Meep at 40 px/cm agrees with the transfer matrix to 0.001; rfx is 0.012 above BOTH. The premise "rfx and Meep sit together ~0.01 from TMM" does not hold; the TMM/interface model is not implicated |

Falsifiers: all nine `verdict.exit_code = 1` as pre-declared (`r1/rfx__falsifier_tand0p1_sigma_neg.json::arms.tand0p1.gates.G3_passivity = false`, mean|ΔA| 0.62; the σ×1.5 / σ = 0 arms reproduce §6's margins to two digits: 0.008/0.092, 0.036/0.214, 0.072/0.036, 0.081/0.778, 0.099/0.001, 0.242/0.830). The API path assembled and matched the direct arrays on both API arms (`::arms.tand1.materials.api_equals_direct = true`). Meep pre-checks 5.5e-17 / 1.6e-16 / 1.9e-16; both wrong-scaling legs pre-checked failed. Witness predictions of §8 held (0 extensions everywhere; tails −61 to −100 dB, well under the predicted −43 to −54: the starting-level assumption was again the conservative one). Meep ladder (`r1/meep_ladder_summary.json::arms.<arm>.orders`): tand0p1 R 1.00 / 1.01, tand1 1.03 / 1.00 (first order, as cv22), **tand3 R 1.95 / 1.90 (second order)** — Meep's first-order term is absent on the surface-impedance arm.

### 12.2 The rfx residual, derived a priori: the exact 1-D Yee lattice of the staircase slab

The rig at normal incidence with periodic y IS a 1-D Yee lattice: E nodes `515..524` carry (ε', σ), all others vacuum. Its time-harmonic solution is exact and needs no measurement. With `z = e^{jωdt}`, `ω̂ = 2 sin(ωdt/2)/dt`, `x = ωdt/2`, the two update equations become

    H_{i+1/2} − H_{i−1/2} = dx (jω̂ ε_i + σ_i cos x) E_i,      E_{i+1} − E_i = dx (jω̂ μ0) H_{i+1/2},

marched from a unit transmitted lattice plane wave (vacuum lattice wavenumber `k = (2/dx) asin(ω̂dx/2c)`) back to the incidence side, where two nodes are decomposed into incident + reflected (`dispersive_eps.yee_lattice_slab_rt`; it converges to `tmm_slab_rt` second-order: mean|ΔR| 0.019 → 1.2e-3 → 4.6e-5 → 1.8e-6 at dx = 1, 0.25, 0.05, 0.01 mm on tand3). It contains at once the slab's bulk numerical dispersion (`|n| k0 dx = 0.64` at 10 GHz on tand3 — ten cells per wavelength inside the conductor), the node interface, and the σ warp of §3.

Evaluated on the r1 bin grid at the rig's dx = 1 mm, dt = 2.335 ps, against the r1 measurement (`r1/rfx.json::arms.<arm>.{R_rfx,T_rfx}`):

| arm | lattice − TMM: mean\|ΔR\| / \|ΔT\| / \|ΔA\| (signed ΔR at 4 / 7 / 10 GHz) | measured rfx − TMM | **\|rfx − lattice\|** mean / max (R) |
|---|---|---|---|
| tand0p1 | 0.00392 / 0.00567 / 0.00194 (+0.0016 / −0.0007 / +0.0140) | 0.0039 / 0.0057 / 0.0020 | **3e-5 / 1.4e-4** |
| tand1 | 0.00509 / 0.00176 / 0.00334 (+0.0026 / +0.0050 / +0.0078) | 0.0051 / 0.0018 / 0.0033 | **2e-5 / 5e-5** |
| tand3 | 0.01256 / 0.00011 / 0.01244 (+0.0067 / +0.0124 / +0.0189) | 0.0126 / 0.0001 / 0.0124 | **3e-5 / 8e-5** |

The measurement is the lattice solution to 3e-5 in the mean on every arm and every observable, bin by bin (the ∝ f trend on tand3 is `(|n| k0 dx)² ∝ f²·(σ/ωε0) ∝ f`). What is left after the lattice term — the truncation / CPML / probe residual the cv04 envelope was meant to cover — is ≤ 1.4e-4. Read plainly: **cv04's committed 0.0066 was itself mostly the lattice term of the lossless ε = 4 slab** (|n| = 2), and §4's assumption that the cv04 window transfers to |n| = 3–4.6 is what fired, exactly as §10(i) allowed. rfx's σ path (§3) is confirmed at the 1e-4 level; nothing in `update_e`, the API assembly or the R extraction is implicated.

The coordinator's two candidate mechanisms, computed a priori as asked (gated-band mean|ΔR| through the TMM): a ±dx/2 thickness error gives 0.031 / 0.006 / **0.0005** on tand0p1 / tand1 / tand3 — the surface-impedance arm is thickness-blind, so an interface shift cannot be its 0.0126 — and half-weighted interface cells (1 | 8 | 1 mm at (ε'+1)/2, σ/2) give 0.058 / 0.017 / 0.022 with the **wrong sign** on tand3 (−0.022 vs the measured +0.0126). Neither is the mechanism; the lattice term is.

### 12.3 Meep: the tand3 ladder is the same lattice term; the tand0p1/tand1 first-order term is a one-cell thickness excess

The same lattice solution at Meep's (dx = a/res, Courant 0.5) predicts tand3 Meep-vs-TMM mean|ΔR| **0.0127 / 0.0031 / 0.0008** at 10 / 20 / 40 px/cm; measured `r1/meep_ladder_summary.json::arms.tand3.rungs.{10,20,40}.mean_dR_meep_tmm_gated` = **0.0131 / 0.0034 / 0.0009**. At the same dx = 1 mm, Meep (0.0131) and rfx (0.0126) carry the same term. For tand0p1 / tand1 the lattice predicts only 0.0040 / 0.0052 at 10 px/cm, while Meep measured 0.0633 / 0.0128 — the first-order excess. Hypothesis, a priori: Meep's block `[−d/2, d/2]` includes the E nodes ON both faces (inclusive containment, `eps_averaging` off), so it realizes **d + a/res**. TMM(d + a/res) − TMM(d), gated mean|ΔR| / |ΔT|:

| arm | 10 px/cm (pred / measured) | 20 | 40 | **80 (prediction)** |
|---|---|---|---|---|
| tand0p1 | 0.0593 / 0.0427 vs 0.0633 / 0.0473 | 0.0307 / 0.0219 vs 0.0316 / 0.0230 | 0.0155 / 0.0110 vs 0.0157 / 0.0113 | **0.0078 / 0.0055** |
| tand1 | 0.0100 / 0.0124 vs 0.0128 / 0.0140 | 0.0057 / 0.0067 vs 0.0063 / 0.0071 | 0.0030 / 0.0034 vs 0.0031 / 0.0035 | 0.0015 / 0.0018 |
| tand3 | 0.0008 vs 0.0131 (lattice-dominated) | 0.0002 | 0.0001 | lattice 0.0002 |

(The tand0p1 10 px/cm row is 6 % short because the lattice term, 0.004, adds to it there; the 20 and 40 rows agree to 3 %.) The reviewer's cv22 warning ("converged by one rung only") is now a mechanism: the 40 px/cm reference on the etalon arms carries a d + 0.25 mm slab.

### 12.4 Round 2 — pre-declared before the run (`scripts/vessl_cv23_lossy_slab_r2.yaml`)

**No window moves.** Every r1 leg is re-run unchanged (baseline expected to exit 1 again on tand3 G2_R and tand0p1 G4_mean_R; nine falsifiers each 1; Meep 40 px/cm primaries; 10/20/40 ladders), plus:

**(a) rfx dx ladder** (`--dx-div 2|4 --tag <arm>_dx<K>`, geometry identical in cells, record re-derived per rung): tand3 (the fired arm), tand1 (control), tand0p1 (completeness). Predictions from the lattice term alone (gated mean |ΔR| / |ΔT| / |ΔA|), locked in `tests/test_cv23_lossy_slab_gates.py::_R2_LATTICE_PRED`:

| arm | dx (r1 measured) | **dx/2** | **dx/4** |
|---|---|---|---|
| tand3 | 0.0126 / 0.0001 / 0.0124 | **0.0031 / 0.00003 / 0.0031** | **0.0008 / 0.00001 / 0.0008** |
| tand1 | 0.0051 / 0.0018 / 0.0033 | 0.0013 / 0.0004 / 0.0008 | 0.0003 / 0.0001 / 0.0002 |
| tand0p1 | 0.0039 / 0.0057 / 0.0020 | 0.0010 / 0.0014 / 0.0005 | 0.0002 / 0.0004 / 0.0001 |

Reading rules: **lattice-confirmed** iff mean|rfx − lattice(dx/K)| ≤ 3e-4 (10× the r1 residual) in R and T on every rung — the fall is then second order (×0.25, ×0.062) by construction; **first-order** iff the ratio to r1 is in [0.4, 0.6] at dx/2 and [0.2, 0.35] at dx/4 with the lattice residual above the bar (an interface term outside the lattice model — would contradict the 3e-5 match, so unlikely, but pre-declared); **no-fall** ≥ 0.7 (then the σ update or the extraction path in a strongly absorbing slab, and the r1 match would have been a coincidence); anything else "unresolved", not fitted. At dx/2 the tand3 arm's mean|ΔR| 0.0031 sits inside the 0.0102 window with 3× margin — a diagnostic reading, not a re-gate.

**(b) Meep 80 px/cm** for tand0p1 and tand3 (`--resolution 80 --tag res80`; Courant 0.5; ladder summary extended to 10/20/40/80). Predictions: tand0p1 Meep-vs-TMM mean|ΔR| **0.0078** (first order, thickness excess d + 0.125 mm) — inside the 0.0100 G4 mean by only 1.28×; if it lands there the residual is Meep's own model of the same interface and 80 px/cm passes for that reason, which is said here rather than by choosing the resolution; tand3 **0.0002–0.0003** (second order, lattice). **(c) The decisive thickness test**: tand0p1 at 40 px/cm with the block drawn **one Meep cell thinner** (`--thickness-offset-cells -1 --tag res40_thin1`, d − 0.25 mm; monitors and source unmoved). Prediction: Meep-vs-TMM falls from 0.0157 to the lattice level **≈ 0.0003 in R, 0.0004 in T**. If it does, Meep's first-order term (here and in cv22) is the inclusive-node thickness excess; if it stays ~0.016 or flips sign at the same size, the hypothesis is wrong and the 80 px/cm reading stands alone.

**(d) What §13 may propose, PI decides — nothing is changed by this run.** If (a) reads lattice-confirmed on all rungs: the term `W_lat(f) = |lattice(f; ε', σ, d, dx, dt) − TMM(f)|` is an a priori, measurement-free function of the declared material and the rig — the same class as `W_ADE` (cv22 §3) and `W_σ` (§3) — and §13 will put to the PI (i) carrying it as a named E2/E4 window term for R, T, A on every arm (the cv04 envelope then covers only the ≤ 1.4e-4 residual it was actually measured to cover), against (ii) declaring dx = 1 mm under-resolved for |n| ≥ 3 and running the tand3 arm at dx/2 as its primary with the windows untouched; and, if (c) confirms, (iii) drawing the Meep D_conductivity block at d − a/res on the etalon arms (a documented Meep-side model correction) against (iv) 80 px/cm with its 1.28× margin stated. Refutations accepted: (a) not lattice-confirmed on any rung (then §12.2's identification is withdrawn and the H3 branch reopens with the extraction path as the suspect); (c) not falling (then §12.3's mechanism is withdrawn); a −40 dB witness failing on a refined rung (record re-derived per rung, as in cv22 r2). Cost: dx/4 on the nx-1000 rig ≈ 64× the r1 arm (~30 s each); Meep 80 px/cm ≈ 4× the 40 px/cm leg.

## 13. Round 2 read (VESSL 369367257814) — lattice term confirmed on every rung; the Meep thickness leg mis-predicted, not refuted; round 3 (final) pre-declared

Artifacts: `~/mnt/remilab-fs/personal-workspaces/claude-workspace/rfx/runs/cv23-lossy-r2-20260902T121457Z/_23_lossy_results/` (`r2/`); log `~/Documents/vessl-run-logs/369367257814_cv23-lossy-r2.log`; tree `30701a1`. `pytest -k r2` rc 0; the full gate test 3 failed / 59 passed (the baseline replays, as expected with no window moved). No window moves in this section either.

### 13.1 rfx dx ladder — lattice-confirmed on every arm and rung (§12.4(a))

`r2/rfx__<arm>_dx<K>.json::arms.<arm>`: |rfx − lattice(dx/K)| ≤ 3.0e-5 in R and ≤ 7e-5 in T on all six rungs (bar 3e-4); ratios to dx ×0.244–0.248 (dx/2) and ×0.060–0.062 (dx/4) — second order, as the lattice predicts. tand3: `r2/rfx.json::arms.tand3.mean_dR_gated = 0.01256` → `r2/rfx__tand3_dx2.json::arms.tand3.mean_dR_gated = 0.00312` → `r2/rfx__tand3_dx4.json::arms.tand3.mean_dR_gated = 0.00078` (predicted 0.0031 / 0.0008), E2 PASS at both refinements. The dx/2 rung ran 2362 steps (`::run.record`: n_pulse_end 1816 + n_ring 446 + window 100; CPML gate 2524; 0 extensions; tails 8.6e-5 / 6.4e-6; nfft 32768; 8.6 s). rfx reproduces the exact Yee-lattice solution of its own staircase, and the lattice converges to the transfer matrix at second order: the only defect at dx = 1 mm was that cv04's |n| = 2 envelope does not carry the lattice term for |n| up to 4.6. `rfx__tand0p1_dx2` and `_dx4` exited 1 with every E2 gate true: the tagged arm reads the committed res-40 Meep leg and fails its E4 `G4_mean_R` (Meep-vs-TMM 0.0157 vs 0.0100) — the E4 consequence of the Meep primary, not a property of the refined rfx arm (`r2/rfx__tand0p1_dx2.json::verdict.summary = "E4 FAIL …"`, `::arms.tand0p1.meep.gates.G4_mean_R = false`).

### 13.2 Meep — 80 px/cm as predicted; the "thin block" leg was mis-predicted by my own node count, and its result confirms the thickness mechanism

`r2/meep_tand0p1__res80.json`: Meep-vs-TMM mean|ΔR| **0.00783** (predicted 0.0078; `G4_mean_R` passes by 1.28×); `r2/meep_tand3__res80.json`: **0.00025** (predicted 0.0002, second order). Ladder orders on tand0p1: 1.00–1.04 (first order, `r2/meep_ladder_summary.json::arms.tand0p1.orders`).

The block drawn d − a/40 at 40 px/cm (`r2/meep_tand0p1__res40_thin1.json`) gave mean|ΔR| **0.01537** against my §12.4(c) prediction of 0.0003. In magnitude it looks unchanged from the nominal 0.0157; in sign it is **mirrored**: its per-bin ΔR at 4 / 5.5 / 7 / 8.5 / 10 GHz is +0.0025 / +0.0164 / +0.0129 / −0.0208 / −0.0211 against the nominal leg's −0.0032 / −0.0170 / −0.0090 / +0.0238 / +0.0191, and it equals `TMM(d − a/40) − TMM(d)` bin by bin to 2.2e-4 mean (correlation 1.000; −0.99 against the nominal), while the nominal leg equals `TMM(d + a/40) − TMM(d)` to 2.5e-4. The two legs differ by 0.031 in mean — exactly two Meep cells of thickness. So the §12.3 mechanism holds in the form the numbers dictate: **Meep realizes d_eff = (number of E nodes inside the block) × a/res with the nodes at integer positions and inclusive containment** — the nominal block [−0.5, 0.5] cm holds 41 nodes (d + a/res), and a block of any drawn size in (d − a/res, d) holds 39 (d − a/res). My prediction assumed the thinner block would hold 40; a 40-node slab cannot be centred at 0 on an integer lattice. The prediction was wrong, the hypothesis was not; recorded as such, with the numbers.

Two Meep discriminators are pre-declared for round 3 at 40 px/cm on tand0p1, with predictions from the integer-node model:

| leg | tag | what it draws | nodes | **prediction** (mean\|ΔR\| / \|ΔT\| vs TMM) | if instead … |
|---|---|---|---|---|---|
| half-cell thinner (the coordinator's) | `res40_thin_half` (`--thickness-offset-cells -0.5`, d − a/80) | edges ±0.49375 cm | 39 | **0.0156 / 0.0110, mirrored** (= TMM(d − a/40)), NOT a collapse | it collapses → the half-pixel offset is the mechanism and the node-count model is wrong |
| centre shifted +½ pixel, nominal size | `res40_shift_half` (`--center-offset-cells 0.5`) | [−0.4875, 0.5125] cm | **40** | **≈ 0.00025 / 0.00036** — the lattice level (d_eff = d; monitors unmoved; \|R\|, \|T\| are translation-invariant) | it stays ~0.0156 → the node-count model is refuted and Meep's first-order term is left as an open note (the 80 px/cm reference then stands on its measured margin alone) |

The r2 thin-by-one leg is re-run in round 3 so it is part of the committed evidence.

### 13.3 Round 3 (final) recipe — resolution, not tolerance; windows unchanged

**rfx primaries** (`cv23_lossy_gates.ARM_DX_DIV`): **tand3 at dx/2** (the arm whose |n| k0 dx = 0.64 at the band top puts the lattice term outside a window derived at |n| = 2; predicted mean|ΔR| 0.0031 vs 0.0102, |ΔA| 0.0031 vs 0.027); **tand0p1 and tand1 at dx** (inside: 0.0039 / 0.0051 vs 0.010). Cost: the dx/2 arm is 2× cells × 2× steps ≈ 4× compute (8.6 s vs 0.7 s on the pod), 2362 steps on nfft 32768 (its own bin grid, ~458 gated bins; the gates are per arm). The nine falsifiers run at the same per-arm recipes (their analytic margins are unchanged; each must exit 1). Every window is as declared in §4.

The exact lattice solution is carried in the artifact as a **reported witness**, not a window term: per arm `lattice.{W_lat_{R,T,A}(f), R/T/A_lattice(f), mean_W_lat_*_gated, mean/max_d{R,T,A}_lattice_gated}` (|rfx − lattice| per bin), replayed by the gate test (it must reproduce, and |rfx − lattice| ≤ 3e-4 is asserted as a witness of the run's integrity, not as accuracy). Carrying `W_lat(f)` inside the E2 window is the alternative available to the PI and was not taken here: it would make the E2 gate test "rfx equals its own lattice model" — a different claim from "rfx reproduces the transfer matrix within a window derived from a committed envelope", which is what this case is for; refining the one under-resolved arm keeps that claim intact.

**Meep primaries** (`MEEP_PRIMARY_RESOLUTION_BY_ARM`): **tand0p1 at 80 px/cm** — Meep-vs-TMM 0.00783 / 0.00559 / 0.00332 against 0.0100 / 0.0170 / 0.0270 (R by **1.28×**, one rung under the window, as the cv22 reviewer insisted be stated; T 3.0×, A 8×); **tand1 at 40 px/cm** — 0.00314 / 0.00355 / 0.00328 (3.2× / 4.8× / 8.2×); **tand3 at 40 px/cm** — 0.00091 / 0.00019 / 0.00071 (11× / 89× / 38×). The 10/20/40/80 ladders and the three thickness legs are committed as evidence.

**Predictions for the round-3 baseline** (all gates pass, exit 0): E2 mean |ΔR| / |ΔT| / |ΔA| tand0p1 0.0039 / 0.0057 / 0.0019, tand1 0.0051 / 0.0018 / 0.0033, tand3 **0.0031 / 0.00003 / 0.0031**, each with |rfx − lattice| ≤ 1e-4; E4 rfx-vs-Meep tand0p1 0.0052 / 0.0031 / 0.0028, tand1 0.0045 / 0.0018 / 0.0063, tand3 0.0022 / 0.0002 / 0.0024 (windows 0.020 / 0.034 / 0.054); Meep-vs-TMM as measured in r2 (the primaries are re-run, not copied). Records: tand0p1 1067 / tand1 1158 / tand3 2362 steps, 0 extensions, tails at the r2 levels (≤ 1e-3, 1e-4, 1e-4). Refutations accepted: any E2/E4 gate failing at these recipes; a −40 dB witness not met; the centre-shift leg not collapsing (then §13.2's model is withdrawn, 80 px/cm stands on its margin alone).

After round 3 lands: artifacts committed under `validation/crossval/_23_lossy_results/`, `artifact_paths` filled, every public number by key, then the independent review.

## 14. Round 3 (VESSL 369367257817) — green; the claims-bearing result by key; the Meep node-count mechanism confirmed by the discriminator that predicted it

Artifacts committed under `validation/crossval/_23_lossy_results/` (this commit; the tree that ran was `796aa1e` = `validation/crossval/_23_lossy_results/rfx.json::commit` via `.staged_commit`; the run log `r3_rfx_baseline.log` and `r3_commit.txt` sit beside the JSONs). Gate test on the committed set: green, no skips. Verdict `validation/crossval/_23_lossy_results/rfx.json::verdict.exit_code = 0`.

### 14.1 Recipes and what they measured (E2, rfx vs the transfer matrix; gated 4–10 GHz, `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.n_bins_gated = 229` bins per arm)

| arm | recipe | mean \|ΔR\| / \|ΔT\| / \|ΔA\| vs windows | max \|ΔR\| / \|ΔA\| | record | lattice witness (reported) |
|---|---|---|---|---|---|
| tand0p1 (direct) | dx (`validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.run.dx_div = 1`) | `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.mean_dR_gated = 0.0039` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.mean_dT_gated = 0.0057` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.mean_dA_gated = 0.002` against `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.mean_window_R = 0.01` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.mean_window_T = 0.017` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.mean_window_A = 0.027` | `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.max_dR_gated = 0.014` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.max_dA_gated = 0.0039` | `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.run.n_steps = 1067` steps at dx/1 (`validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.run.record.extensions = 0` extensions; tails `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.tail.scat_refl_rel = 0.00089` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.tail.total_trans_rel = 0.001`) | |rfx − lattice| `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.lattice.mean_dR_lattice_gated = 3e-5` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.lattice.mean_dT_lattice_gated = 3.1e-5` (max R `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.lattice.max_dR_lattice_gated = 0.00014`), W_lat `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.lattice.mean_W_lat_R_gated = 0.0039` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.lattice.mean_W_lat_T_gated = 0.0057` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.lattice.mean_W_lat_A_gated = 0.0019` |
| tand1 (api) | dx (`validation/crossval/_23_lossy_results/rfx.json::arms.tand1.run.dx_div = 1`) | `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.mean_dR_gated = 0.0051` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.mean_dT_gated = 0.0018` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.mean_dA_gated = 0.0033` against `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.mean_window_R = 0.01` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.mean_window_T = 0.017` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.mean_window_A = 0.027` | `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.max_dR_gated = 0.0077` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.max_dA_gated = 0.0047` | `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.run.n_steps = 1158` steps at dx/1 (`validation/crossval/_23_lossy_results/rfx.json::arms.tand1.run.record.extensions = 0` extensions; tails `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.tail.scat_refl_rel = 9.8e-5` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.tail.total_trans_rel = 2.8e-5`) | |rfx − lattice| `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.lattice.mean_dR_lattice_gated = 2.1e-5` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.lattice.mean_dT_lattice_gated = 5.3e-6` (max R `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.lattice.max_dR_lattice_gated = 5.4e-5`), W_lat `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.lattice.mean_W_lat_R_gated = 0.0051` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.lattice.mean_W_lat_T_gated = 0.0018` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.lattice.mean_W_lat_A_gated = 0.0033` |
| tand3 (api) | **dx/2** (`validation/crossval/_23_lossy_results/rfx.json::arms.tand3.run.dx_div = 2`) | `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.mean_dR_gated = 0.0031` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.mean_dT_gated = 3e-5` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.mean_dA_gated = 0.0031` against `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.mean_window_R = 0.01` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.mean_window_T = 0.017` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.mean_window_A = 0.027` (R **3.2×** inside; T vacuous as declared) | `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.max_dR_gated = 0.0047` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.max_dA_gated = 0.0047` | `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.run.n_steps = 2362` steps at dx/2 (`validation/crossval/_23_lossy_results/rfx.json::arms.tand3.run.record.extensions = 0` extensions; tails `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.tail.scat_refl_rel = 8.6e-5` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.tail.total_trans_rel = 6.4e-6`) | |rfx − lattice| `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.lattice.mean_dR_lattice_gated = 3e-5` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.lattice.mean_dT_lattice_gated = 2.1e-7` (max R `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.lattice.max_dR_lattice_gated = 7.6e-5`), W_lat `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.lattice.mean_W_lat_R_gated = 0.0031` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.lattice.mean_W_lat_T_gated = 3e-5` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.lattice.mean_W_lat_A_gated = 0.0031` |

Every §13.3 prediction landed (0.0039 / 0.0057 / 0.0019; 0.0051 / 0.0018 / 0.0033; 0.0031 / 0.00003 / 0.0031). `A_tight_ok` is true on all three arms (`validation/crossval/_23_lossy_results/rfx.json::arms.tand3.A_tight_ok`): the closure-derived tighter A window would also have passed; it stays reported, not gated. The API-path arms assembled the direct arrays bit-for-bit (`validation/crossval/_23_lossy_results/rfx.json::arms.tand1.materials.api_equals_direct`, `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.materials.api_equals_direct`). Fitted tail rates: tand1 `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.tail.fitted_rate_scat_refl_1_s = 3.2e9` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.tail.fitted_rate_total_trans_1_s = 4.6e9` s⁻¹ (`validation/crossval/_23_lossy_results/rfx.json::arms.tand1.tail.fitted_rate_blocks = 4` blocks) and tand3 `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.tail.fitted_rate_scat_refl_1_s = 1e10` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.tail.fitted_rate_total_trans_1_s = 1.6e10` s⁻¹ (`validation/crossval/_23_lossy_results/rfx.json::arms.tand3.tail.fitted_rate_blocks = 9` blocks) against the derived etalon rates `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.run.record.rate_ring_1_s = 8.4e9` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.run.record.rate_ring_1_s = 7.5e9` — the witness is the gate, the estimate sized the record (0 extensions everywhere). **tand0p1's fit is a two-point estimate**: its record has 109 post-window steps, `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.tail.fitted_rate_blocks = 2` blocks of 50 after the pulse, so `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.tail.fit_reliable` is false and its `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.tail.fitted_rate_scat_refl_1_s = 2.3e10` s⁻¹ is not evidence of a rate; the r3 artifact stored NaN there and was post-processed with `--refit-tail-fits` (no rerun; `tail.fit_note` says so), the gate test accepting nb ≥ 2 with the flag. A longer stored envelope for future runs is a script change, not made here.

### 14.2 E4 — Meep primaries per arm, margins stated

| arm | Meep primary | pre-check | E4 means | Meep-vs-TMM R margin |
|---|---|---|---|---|
| tand0p1 | **80 px/cm** (`validation/crossval/_23_lossy_results/meep_tand0p1.json::resolution = 80`) | `validation/crossval/_23_lossy_results/meep_tand0p1.json::precheck.max_rel_err = 5.5e-17` | Meep-vs-TMM `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.meep.mean_dR_meep_tmm_gated = 0.0078` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.meep.mean_dT_meep_tmm_gated = 0.0056` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.meep.mean_dA_meep_tmm_gated = 0.0033`, rfx-vs-Meep `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.meep.mean_dR_rfx_meep_gated = 0.0052` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.meep.mean_dT_rfx_meep_gated = 0.0031` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.meep.mean_dA_rfx_meep_gated = 0.0028` (windows `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.meep.mean_window5_R = 0.02` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.meep.mean_window5_T = 0.034` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.meep.mean_window5_A = 0.054`) | **1.3×** — one rung under the window (the 40 px/cm rung `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand0p1.rungs.40.mean_dR_meep_tmm_gated = 0.016` is outside it) |
| tand1 | 40 px/cm (`validation/crossval/_23_lossy_results/meep_tand1.json::resolution = 40`) | `validation/crossval/_23_lossy_results/meep_tand1.json::precheck.max_rel_err = 1.6e-16` | Meep-vs-TMM `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.meep.mean_dR_meep_tmm_gated = 0.0031` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.meep.mean_dT_meep_tmm_gated = 0.0035` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.meep.mean_dA_meep_tmm_gated = 0.0033`, rfx-vs-Meep `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.meep.mean_dR_rfx_meep_gated = 0.0045` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.meep.mean_dT_rfx_meep_gated = 0.0018` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.meep.mean_dA_rfx_meep_gated = 0.0063` (windows `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.meep.mean_window5_R = 0.02` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.meep.mean_window5_T = 0.034` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand1.meep.mean_window5_A = 0.054`) | 3.2× |
| tand3 | 40 px/cm (`validation/crossval/_23_lossy_results/meep_tand3.json::resolution = 40`) | `validation/crossval/_23_lossy_results/meep_tand3.json::precheck.max_rel_err = 1.9e-16` | Meep-vs-TMM `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.meep.mean_dR_meep_tmm_gated = 0.00091` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.meep.mean_dT_meep_tmm_gated = 0.00019` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.meep.mean_dA_meep_tmm_gated = 0.00071`, rfx-vs-Meep `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.meep.mean_dR_rfx_meep_gated = 0.0022` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.meep.mean_dT_rfx_meep_gated = 0.00016` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.meep.mean_dA_rfx_meep_gated = 0.0024` (windows `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.meep.mean_window5_R = 0.02` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.meep.mean_window5_T = 0.034` / `validation/crossval/_23_lossy_results/rfx.json::arms.tand3.meep.mean_window5_A = 0.054`) | 11.1× |

Ladders (`validation/crossval/_23_lossy_results/meep_ladder_summary.json::resolutions.3 = 80` px/cm rungs present): tand0p1 first order, R `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand0p1.orders.order_dR_10_20 = 1` / `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand0p1.orders.order_dR_20_40 = 1` / `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand0p1.orders.order_dR_40_80 = 1` (`validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand0p1.rungs.10.mean_dR_meep_tmm_gated = 0.063` → `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand0p1.rungs.20.mean_dR_meep_tmm_gated = 0.032` → `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand0p1.rungs.40.mean_dR_meep_tmm_gated = 0.016` → `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand0p1.rungs.80.mean_dR_meep_tmm_gated = 0.0078`); tand1 first order `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand1.orders.order_dR_10_20 = 1` / `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand1.orders.order_dR_20_40 = 1`; tand3 second order `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand3.orders.order_dR_10_20 = 2` / `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand3.orders.order_dR_20_40 = 1.9` / `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand3.orders.order_dR_40_80 = 1.8` (`validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand3.rungs.10.mean_dR_meep_tmm_gated = 0.013` → `validation/crossval/_23_lossy_results/meep_ladder_summary.json::arms.tand3.rungs.80.mean_dR_meep_tmm_gated = 0.00025`), the lattice term of §12.2 at Meep's dx.

### 14.3 The node-count mechanism, confirmed by the discriminator that predicted it — an actionable note for anyone driving Meep without `eps_averaging`

tand0p1 at 40 px/cm (`meep_ladder_summary.json::diagnostics.tand0p1`): block drawn one cell thinner → `validation/crossval/_23_lossy_results/meep_ladder_summary.json::diagnostics.tand0p1.res40_thin1.mean_dR_meep_tmm_gated = 0.015` (predicted 0.0156, mirrored); half a cell thinner → `validation/crossval/_23_lossy_results/meep_ladder_summary.json::diagnostics.tand0p1.res40_thin_half.mean_dR_meep_tmm_gated = 0.015` (predicted 0.0156 mirrored, NOT a collapse — as the integer-node model said, and as the half-pixel-offset alternative did not); **nominal size with the centre shifted by half a pixel → `validation/crossval/_23_lossy_results/meep_ladder_summary.json::diagnostics.tand0p1.res40_shift_half.mean_dR_meep_tmm_gated = 0.00024`** (predicted 0.00025; `G4_mean_R` `validation/crossval/_23_lossy_results/meep_ladder_summary.json::diagnostics.tand0p1.res40_shift_half.G4_mean_R`), the lattice level of §12.2. Read plainly: with `eps_averaging=False`, a Meep `Block` realizes a slab of `(number of E-node positions inside it) × a/res`, the E nodes sit at integer multiples of `a/res`, and containment is inclusive — a block of nominal width `N·a/res` centred on a node holds `N + 1` nodes and is one cell too thick; drawn off-centre by half a pixel it holds `N` and is exact. The first-order "interface term" cv22 measured on its etalon arms and carried at 40 px/cm is this thickness excess; on the surface-impedance arm (tand3) it is invisible because R there does not depend on d. This is stated as a property of the Meep geometry as used by these legs (2-D TM, `Ez`, `Block` on a node-centred cell), witnessed by three legs and one prediction; it is not a statement about `eps_averaging=True`.

### 14.4 Falsifiers — each exit 1 against the passing baseline, at the arm's recipe

| arm | mean \|ΔR\| / \|ΔT\| / \|ΔA\| vs the DECLARED material, exit |
|---|---|
| `tand0p1_sigma_x1p5` | `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_x1p5.json::arms.tand0p1.mean_dR_gated = 0.008` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_x1p5.json::arms.tand0p1.mean_dT_gated = 0.092` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_x1p5.json::arms.tand0p1.mean_dA_gated = 0.095`, `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_x1p5.json::verdict.exit_code = 1` (T carries it; R was the declared coin toss) |
| `tand0p1_sigma_zero` | `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_zero.json::arms.tand0p1.mean_dR_gated = 0.036` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_zero.json::arms.tand0p1.mean_dT_gated = 0.21` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_zero.json::arms.tand0p1.mean_dA_gated = 0.25`, `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_zero.json::verdict.exit_code = 1` |
| `tand0p1_sigma_neg` (gain, passivity) | `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_neg.json::arms.tand0p1.mean_dR_gated = 0.1` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_neg.json::arms.tand0p1.mean_dT_gated = 0.52` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_neg.json::arms.tand0p1.mean_dA_gated = 0.62`, `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_neg.json::verdict.exit_code = 1`; `validation/crossval/_23_lossy_results/rfx__falsifier_tand0p1_sigma_neg.json::arms.tand0p1.gates.G3_passivity` is false |
| `tand1_sigma_x1p5` | `validation/crossval/_23_lossy_results/rfx__falsifier_tand1_sigma_x1p5.json::arms.tand1.mean_dR_gated = 0.072` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand1_sigma_x1p5.json::arms.tand1.mean_dT_gated = 0.036` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand1_sigma_x1p5.json::arms.tand1.mean_dA_gated = 0.036`, `validation/crossval/_23_lossy_results/rfx__falsifier_tand1_sigma_x1p5.json::verdict.exit_code = 1` |
| `tand1_sigma_zero` | `validation/crossval/_23_lossy_results/rfx__falsifier_tand1_sigma_zero.json::arms.tand1.mean_dR_gated = 0.081` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand1_sigma_zero.json::arms.tand1.mean_dT_gated = 0.78` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand1_sigma_zero.json::arms.tand1.mean_dA_gated = 0.74`, `validation/crossval/_23_lossy_results/rfx__falsifier_tand1_sigma_zero.json::verdict.exit_code = 1` |
| `tand3_sigma_x1p5` (at dx/2) | `validation/crossval/_23_lossy_results/rfx__falsifier_tand3_sigma_x1p5.json::arms.tand3.mean_dR_gated = 0.085` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand3_sigma_x1p5.json::arms.tand3.mean_dT_gated = 0.0012` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand3_sigma_x1p5.json::arms.tand3.mean_dA_gated = 0.084`, `validation/crossval/_23_lossy_results/rfx__falsifier_tand3_sigma_x1p5.json::verdict.exit_code = 1` |
| `tand3_sigma_zero` (at dx/2) | `validation/crossval/_23_lossy_results/rfx__falsifier_tand3_sigma_zero.json::arms.tand3.mean_dR_gated = 0.25` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand3_sigma_zero.json::arms.tand3.mean_dT_gated = 0.83` / `validation/crossval/_23_lossy_results/rfx__falsifier_tand3_sigma_zero.json::arms.tand3.mean_dA_gated = 0.59`, `validation/crossval/_23_lossy_results/rfx__falsifier_tand3_sigma_zero.json::verdict.exit_code = 1` |
| `meep_tand1_sigma_2pi` (rfx correct; Meep σ_D × 2π) | Meep-vs-TMM `validation/crossval/_23_lossy_results/rfx__falsifier_meep_tand1_sigma_2pi.json::arms.tand1.meep.mean_dR_meep_tmm_gated = 0.35`, pre-check `validation/crossval/_23_lossy_results/meep_tand1__falsifier_sigma_2pi.json::precheck.max_rel_err = 4.4`, `validation/crossval/_23_lossy_results/rfx__falsifier_meep_tand1_sigma_2pi.json::verdict.exit_code = 1` |
| `meep_tand1_sigma_no_eps` (ε' dropped) | Meep-vs-TMM `validation/crossval/_23_lossy_results/rfx__falsifier_meep_tand1_sigma_no_eps.json::arms.tand1.meep.mean_dR_meep_tmm_gated = 0.26`, pre-check `validation/crossval/_23_lossy_results/meep_tand1__falsifier_sigma_no_eps.json::precheck.max_rel_err = 2.5`, `validation/crossval/_23_lossy_results/rfx__falsifier_meep_tand1_sigma_no_eps.json::verdict.exit_code = 1` |

The §6 analytic margins are reproduced to two digits.

### 14.5 The rfx dx ladder, committed as evidence (re-run in round 3)

|rfx − lattice| ≤ `validation/crossval/_23_lossy_results/rfx__tand3_dx4.json::arms.tand3.lattice.mean_dR_lattice_gated = 2.9e-5` on every rung: tand3 `validation/crossval/_23_lossy_results/rfx__tand3_dx2.json::arms.tand3.mean_dR_gated = 0.0031` → `validation/crossval/_23_lossy_results/rfx__tand3_dx4.json::arms.tand3.mean_dR_gated = 0.00078`, tand1 `validation/crossval/_23_lossy_results/rfx__tand1_dx2.json::arms.tand1.mean_dR_gated = 0.0013` → `validation/crossval/_23_lossy_results/rfx__tand1_dx4.json::arms.tand1.mean_dR_gated = 0.00031`, tand0p1 `validation/crossval/_23_lossy_results/rfx__tand0p1_dx2.json::arms.tand0p1.mean_dR_gated = 0.00096` → `validation/crossval/_23_lossy_results/rfx__tand0p1_dx4.json::arms.tand0p1.mean_dR_gated = 0.00024` (second order, §12.4(a) predictions 0.0031 / 0.0008, 0.0013 / 0.0003, 0.0010 / 0.0002).

### 14.6 Put to the PI: the W_lat alternative

The case gates rfx against the transfer matrix inside windows derived from cv04's committed envelope, and resolves the one arm whose lattice term exceeds that envelope (dx/2 on tand3). The alternative — carrying `W_lat(f) = |lattice(f; ε', σ, d, dx, dt) − TMM(f)|` (a priori, measurement-free, `dispersive_eps.yee_lattice_slab_rt`) as a named window term on every arm and keeping tand3 at dx — would pass at dx by construction (`validation/crossval/_23_lossy_results/rfx.json::arms.tand3.lattice.mean_dR_lattice_gated = 3e-5`-class residuals) and is available; it was not taken because it changes the claim from "rfx reproduces the transfer matrix within a committed-envelope window" to "rfx equals its own lattice model", which is a different (weaker for the user, stronger for the solver) statement. Either way the artifact carries the lattice witness per bin, so the choice can be revisited without a rerun.

### 14.7 Chronology, r1 → r3

| round | VESSL | what fired | what decided it |
|---|---|---|---|
| r1 | 369367257813 | tand3 G2_R (0.0126 vs 0.0102); tand0p1 Meep G4_mean_R (0.0157 vs 0.010) | the exact Yee-lattice solution reproduces every rfx arm to 3e-5 (§12.2); Meep's first-order term = TMM(d + a/res) (§12.3) |
| r2 | 369367257814 | — (diagnostics) | rfx dx ladder lattice-confirmed on all six rungs; Meep 80 px/cm as predicted; thin-block leg mirrored, my node count corrected (§13.2) |
| r3 | 369367257817 | — (green) | tand3 at dx/2, tand0p1 Meep at 80 px/cm; centre-shift leg collapses to the lattice level as predicted; committed here |

No window moved at any round.
