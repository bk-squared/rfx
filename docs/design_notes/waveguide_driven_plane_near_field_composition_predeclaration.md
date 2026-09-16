# Pre-declaration — what the near field at the driven port's reference plane is made of

Lane: issue #873, option 1 of the three the leader session put to the PI on 2026-09-16.
A **new mechanism hypothesis** — the transverse composition of the source's own near
field — distinct from the two lanes already closed on this issue:

* **PR #880 (attempt 1)**: the excess modelled as a spurious *reflection*, three named
  factors of `Z_seen/Z_used`. Non-closing; and it ran on the N+1-cell port (#868) that
  PR #889 has since fixed.
* **PR #1081 (attempt 2)**: the excess modelled as a port-local *V/I transfer* error.
  Refuted by a bound, not a fit: `a₁` and `b₂` are the same `(V + Z·I)/2` combination
  read by two identical ports, so any common transfer error cancels in `|S21|` exactly
  (`rfx/sources/waveguide_port.py:1600-1601`, `:1667-1669`, `:2227`, `:2239`).

R2 accounting for this note: **attempts on THIS hypothesis = 1**, and it is the only one
authorised. The hypothesis is new because its mechanism (what the field at the plane is
made of) is not a member of either closed family; PR #1081's own closing paragraph names
it as the measurement that run did not make.

Written **before any projection coefficient of a simulated field was computed**. Section 2
states numbers that are properties of the *port code and the grid* — discrete cutoffs,
attenuation constants, and the extractor's own overlap rows. Those are inputs to the
derivation, not measurements of the observable, and how each was obtained is stated where
it appears. Nothing in this note was computed from an FDTD field.

## 0. The observable this has to explain

From PR #1081's committed artifact
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json`, stage `planes`.
Each waveguide port records modal V and I at two planes, so one drive gives four. On this
fixture (`tests/fixtures/waveguide_chain_battery/fixture_v18_close.json`, three thru cells,
`normalize=False`) they sit at fixed physical x at every rung. The driven source plane is
at 12.70 mm; the receiving source plane at 109.22 mm.

Modal power `P = ½·Re(V·I*)` at each plane, **relative to the receiving port's reference
plane**, at bin 16 (11.6 GHz, the worst bin):

| plane | x (mm) | from the DRIVEN source (mm) | coarse | mid | fine |
|---|---|---|---|---|---|
| driven reference | 20.32 | 7.62 | **−5.070e-3** | **−1.073e-3** | **−2.454e-4** |
| driven probe | 38.10 | 25.40 | +8.246e-4 | +1.717e-4 | +3.898e-5 |
| receiving probe | 83.82 | 71.12 | −4.196e-5 | −1.228e-5 | −4.590e-6 |
| receiving reference | 101.60 | 88.90 | 0 (reference) | 0 | 0 |

Call the driven-reference entry `D_meas(f)`. It is 82.6 / 92.4 / 96.9 % of that rung's
column-power excess at bin 16. The three planes at ≥25 mm agree to ≤8.2e-4 (coarse).
The **first-hop ratio** `|D_meas(7.62 mm)| / |offset(25.40 mm)|` is 6.15 / 6.25 / 6.29 —
mesh-independent to 2.5 % — while the amplitude falls ~4.5× per dx halving. The signs
alternate across the three planes, so PR #1081 explicitly did **not** claim a decay length;
this note does not inherit that claim either.

## 1. The hypothesis, stated so it can fail

> The driven port's reference plane, 7.62 mm downstream of an active TFSF launch, sits in
> the source's near field. That near field contains discrete **higher modes of the port
> cross-section**, all below cutoff in 8.4–11.6 GHz. Their content, read through the
> extractor's own aperture weighting, is what makes `a₁` wrong and therefore `|S21| =
> |b₂/a₁|` too large at the top of the band.

## 2. The basis, and the two overlap rows that decide what can leak

### 2.1 Basis

The discrete TE and TM eigenmodes of the port cross-section **on the arm's own grid**, not
analytic sines. They are built from the same module the port builds its own mode with,
`rfx/sources/_waveguide_modes.py`: the Galerkin stiffness/mass pair
`_galerkin_stiffness_mass_1d` (Neumann for TE `Hx`, Dirichlet for TM `Ex`), the transverse
gradient `_cell_centred_gradient`, the dual-mesh co-location `_shift_profile_to_dual` with
the port's own `h_offset = (0.5, 0.5)`, and the same amplitude normalisation
(`∫(e_y²+e_z²) dA = 1`, then `_scale_h_to_unit_cross`) that
`_discrete_te_mode_profiles` / `_discrete_tm_mode_profiles` apply.

One deviation from calling those two functions directly, and the reason for it: they pick
an eigenvector out of the 2-D spectrum by overlap with an analytic shape, and on this grid
several requested `(m, n)` land on the **same** eigenvector (at the coarse rung TE03, TE13,
TE23 and TE70 all return `kc = 945.01`). A basis with duplicated vectors is singular. The
operator here is separable, so the basis is instead assembled as the tensor product of the
two 1-D eigenproblems — `Hx_mn = φ_m(u) ⊗ χ_n(v)`, `kc²_mn = λ_m + μ_n` — which gives every
`(m, n)` a unique vector and an unambiguous label, with no picker. The TE10 profiles this
produces agree with the port's own stored `ez_profile` / `hy_profile` to 3.7e-8 relative
(float32 storage), which is the check that the two constructions are the same object.

TE: `m = 0 … nu−1`, `n = 0 … nv−1`, excluding `(0,0)`. TM: `m = 1 … nu`, `n = 1 … nv`.
`(nu, nv)` = (9, 4) / (18, 8) / (36, 16).

### 2.2 Cutoffs and attenuation constants

Computed from the basis above; `α_mn(f) = sqrt(kc_mn² − k²)`, `k = 2πf/c`. Analytic
cutoffs beside the discrete ones, `α` and the decay over the **first hop** (7.62 → 25.40
mm, i.e. 17.78 mm) at bin 16 (11.6 GHz):

| mode | f_c analytic | f_c discrete (c/m/f) | α(11.6 GHz), 1/m | 1/α, mm | `exp(+α·17.78 mm)` |
|---|---|---|---|---|---|
| TE10 | 6.5571 | 6.5239 / 6.5488 / 6.5551 | — (propagating) | — | — |
| TE20 | 13.1143 | 12.8496 / 13.0478 / 13.0976 | 115.84 / 125.20 / 127.46 | 8.63 / 7.99 / 7.85 | **7.84 / 9.26 / 9.64** |
| TE01 | 14.7536 | 14.3773 / 14.6589 / 14.7299 | 178.02 / 187.84 / 190.26 | 5.62 / 5.32 / 5.26 | 23.7 / 28.2 / 29.5 |
| TE11, TM11 | 16.1451 | 15.7882 / 16.0553 / 16.1226 | 224.47 / 232.64 / 234.68 | 4.46 / 4.30 / 4.26 | 54.1 / 62.6 / 64.9 |
| TE30 | 19.6714 | 18.7848 / 19.4475 / 19.6153 | 309.67 / 327.14 / 331.51 | 3.23 / 3.06 / 3.02 | **246 / 336 / 363** |
| TE50 | 32.7857 | 28.7800 / 31.7552 / 32.5262 | 552.02 / 619.55 / 636.87 | 1.81 / 1.61 / 1.57 | 1.83e4 / 6.08e4 / 8.28e4 |

(GHz. At the band bottom, 8.4 GHz, every `α` is larger: TE20 reads 203.8 / 209.3 / 210.6
1/m.)

### 2.3 The two overlap rows — what the extractor can and cannot see

The extractor forms exactly two aperture integrals
(`rfx/sources/waveguide_port.py:1337-1360`):

    V = Σ (E_y·ey₁₀ + E_z·ez₁₀) dA
    I = Σ (H̃_y·hy₁₀ + H̃_z·hz₁₀) dA · exp(+jω·dt/2)

with `H̃` the H field co-located onto the E plane by `_plane_h_field_at_dual` (a centred
average over the two half-cell planes along the normal, then the same edge-clamped
`[1,2,1]/4` transverse stencil the stored profile carries). So a higher mode `k` reaches
`V` only through `⟨e_k, e₁₀⟩` and reaches `I` only through `⟨h_k, h₁₀⟩`, both taken with the
aperture measure `dA`. Those two rows are properties of the code and the grid, and they are
computed here, before any field:

| row | result |
|---|---|
| `⟨e_k, e₁₀⟩`, every physical mode `k ≠ TE10` | **zero to 1.4e-15** at all three rungs |
| `⟨h_k, h₁₀⟩`, TE20, TE01, TE11, TM11, TE21, TE40 … | **zero to 2e-15** |
| `⟨h_k, h₁₀⟩`, odd TE_m0 only | TE30 **5.445e-2 / 6.508e-3 / 7.990e-4**; TE50 1.049e-1 / 1.133e-2 / 1.348e-3; TE70 1.881e-1 / 1.710e-2 / 1.922e-3 |

The one non-zero entry outside that family is the highest-order TM corner mode (TM94 /
TM188 / TM3616), `⟨e_k, e₁₀⟩` = 1.4e-3 / 5.0e-3 / 9.8e-4 — a lattice-scale checkerboard
whose `α` at 11.6 GHz is 1087 / 2214 / 4448 1/m, i.e. `exp(−α·7.62 mm)` ≤ 4e-4 even at the
coarse rung. It is carried in the tables but cannot be a carrier.

Why the family is what it is, stated as a derivation rather than as an observation:
`ez₁₀ ∝ sin(πu/a)` is **even about the guide centre and constant in v**, and `ey₁₀ = 0`.
So `V` is a projection onto one function that is u-even and v-flat: every mode with `n ≥ 1`
integrates to zero in v, and every u-odd mode (TE20, TE40, …) integrates to zero in u. What
survives is the odd `TE_m0` family — and on the E side even those come out at 1e-15,
because the discrete gradient makes the TE family `⟨e_k, e_l⟩`-orthogonal. **The extractor's
modal voltage is blind to every higher mode of this guide.** The H side is not, and the
reason is named: `hy = −_shift_profile_to_dual(ez, h_offset)` uses an **edge-clamped**
`[1,2,1]/4` stencil, which at the two aperture walls substitutes an even reflection where
the cell-centred sine has an odd one and adds `0.5·f₀` at each wall cell (the same term PR
#1081 §4 measured: predicted 8.057 against measured 8.140 at the coarse rung). That wall
term is u-even, so it has content in odd `TE_m0` and only there.

### 2.4 What §2.3 costs the hypothesis before a single field is read

`a₁ = (V + Z·I)/2`. With `V` blind, the **entire** channel by which higher-mode content can
corrupt `a₁` is

    δa₁ = (Z/2) · Σ_{m odd ≥ 3} B_m · ⟨h_m0, h₁₀⟩ ,

with `B_m` the amplitude of mode `m` in the co-located H at the plane. Two consequences
are fixed here, before the measurement:

1. **A single-TE20 explanation is excluded by the overlap row, not by a fit.** TE20 is the
   lowest evanescent mode and will carry the largest amplitude at the plane, but its weight
   into both `V` and `I` is zero to 2e-15. Whatever TE20 is doing at that plane, the
   extractor does not see it.
2. **The first-hop ratio the hypothesis predicts is 246 / 336 / 363, against a measured
   6.15 / 6.25 / 6.29.** TE30 is the lowest member of the only family that leaks. The
   measured ratio is mesh-independent to 2.5 %; TE30's prediction moves 48 % across the
   same ladder. Even the naive TE20 candidate — which cannot leak — predicts 7.84 / 9.26 /
   9.64, already above the measurement and moving 23 % across the ladder. **No discrete
   evanescent mode of this guide predicts a first-hop ratio as small as 6.2, and none
   predicts one that is mesh-independent.** For a single-mode explanation that is fatal
   unless the cross-term's phase rotation rescues it: the TE10 factor turns `β·17.78 mm` =
   3.567 rad ≈ 204° over that hop, so a *cross-term* ratio is `exp(α·Δ)·|cos ψ /
   cos(ψ+βΔ)|` and is not a pure exponential. That is why the decision rule below tests the
   **complex modal coefficient's own** ratio, where no phase factor enters, and treats the
   cross-term ratio as the secondary comparison.

This note is therefore written expecting the hypothesis to be hard to confirm. It is run
anyway because the measurement is cheap, because it is the one PR #1081 said was missing,
and because a refutation here is informative in a way the two closed lanes were not: if the
named modes are present and the extractor is blind to them, then the offset is in the TE10
coefficient itself, which points away from mode content and at the launch.

## 3. The measurement

One CPU re-run of the three thru cells (coarse ≈ 6 s, all three ≈ 2 min), `num_periods =
40`, `normalize=False`, identical to the fixture. The FDTD solve is byte-for-byte the solve
PR #1081 re-measured: the only change is **what is recorded**. `modal_voltage` and
`modal_current` (`rfx/sources/waveguide_port.py:1337`, `:1346`) are replaced by functions
that write the *whole aperture slice* they would otherwise contract — `(E_y, E_z)` at the
record plane and `(H̃_y, H̃_z)` from `_plane_h_field_at_dual` at the same plane — into the
port's existing per-step record buffers, widened from `(n_t,)` to `(n_t, 2, nu, nv)`.
Injection (`apply_waveguide_port_e` / `_h`) reads `cfg.ez_profile` / `cfg.hy_profile` and is
untouched, so the fields are unchanged.

**Reproduce-gate**, fixed here: a separate unpatched run of each rung must reproduce the
committed `fixture_v18_close.json` column power on the quantity under study, to the same
tolerance PR #1081 recorded for a float32 backend change (worst-bin `column power − 1`
within 2 % at the fine rung and 0.1 % at the coarse one). And, on the patched run, the
modal voltage reconstructed from the recorded aperture slice,
`Σ(E_y·ey₁₀ + E_z·ez₁₀) dA`, must equal the unpatched run's own `cfg.v_ref_t` to ≤1e-5
relative. Without that second check the capture is not proven to be the field the extractor
reads.

Post-scan, in float64: the same rectangular full-record DFT the port uses
(`_rect_dft`, `2·dt·Σ exp(−jωt)` over `n_steps_recorded`) is applied per aperture cell and
component, giving complex `E(u,v,f)` and `H̃(u,v,f)` at each of the four planes. Then, per
plane and per bin, least-squares projection onto the §2.1 basis in the `dA` inner product:

    A_k(f) = argmin ‖E − Σ A_k e_k‖_dA ,     B_k(f) = argmin ‖H̃ − Σ B_k h_k‖_dA

normalised so `Σ_k |A_k|²⟨e_k,e_k⟩` is the plane's total transverse-E energy; the
unexplained residual fraction is reported beside every fit and a fit with residual > 1 % is
reported as a failed measurement rather than used.

**Dumped per rung, per plane, per bin** (workspace rule R5 — the full trace, not the worst
bin): magnitude and phase of every mode with `|A_k|²⟨e_k,e_k⟩ > 1e-6` of the TE10 term, the
same on the H side, the two overlap rows, the per-mode contributions to `V` and `I`, and
the TE10 coefficient at all four planes. Figure: per-bin coefficient spectra, path recorded
in the results note.

## 4. The prediction that would confirm the hypothesis — derived

Write, at a plane, `V = Σ_k A_k⟨e_k,e₁₀⟩ = V₁₀ + V_hi` and `I = e^{jωdt/2}·Σ_k
B_k⟨h_k,h₁₀⟩ = I₁₀ + I_hi`, with `V₁₀ = A₁₀⟨e₁₀,e₁₀⟩`, `I₁₀ = e^{jωdt/2}B₁₀⟨h₁₀,h₁₀⟩` the
TE10-only parts. The extractor's modal power is `P = ½Re(V·I*)`; its TE10-only value is
`P₁₀ = ½Re(V₁₀·I₁₀*)`. The hypothesis says the offset at the driven plane *is* the
higher-mode part, so the predicted offset is

    D_pred(f) = [ ½Re(V·I*) − ½Re(V₁₀·I₁₀*) ] / ½Re(V₁₀·I₁₀*)                        (★)

computed at the driven reference plane from the measured `A_k`, `B_k` and the §2.3 overlap
rows — i.e. from the measured coefficients through the extractor's own aperture weighting,
with no free parameter. It is compared against `D_meas(f)` of §0.

`D_meas` is referenced to the *receiving* plane's power, while (★) is referenced to the
driven plane's own TE10-only power. Those two references differ by the clean-TE10 power
difference between the planes, `P₁₀(drive)/P(recv_ref) − 1`, which is reported as a third
column at every bin rather than assumed away. If it is not small compared with `D_meas`,
the comparison in §5 leg 1 is reported as inconclusive rather than as a pass or a fail.

## 5. Decision rule, fixed here

**CONFIRMED** requires both legs:

1. **Magnitude.** `|D_pred / D_meas − 1| ≤ 0.25` — a factor 1.25 — at the worst bin
   (bin 16, 11.6 GHz) on **all three** rungs, with the reference-difference column of §4
   below 25 % of `D_meas` at that bin.
2. **Decay.** The mode set that carries ≥80 % of `D_pred` at bin 16 is *named in advance*
   as the odd `TE_m0` family, TE30 first (§2.3 leaves no alternative). Its complex
   coefficient must fall between the 7.62 mm and 25.40 mm planes by `exp(α_m·17.78 mm)` —
   246 / 336 / 363 for TE30 — within **25 %**, at all three rungs. If more than one odd
   `TE_m0` carries ≥20 % of `D_pred`, the weighted sum of their predicted contamination at
   25.40 mm is used instead, and must reproduce the measured first-hop ratio 6.15 / 6.25 /
   6.29 within 25 %.

Anything else is **NOT CONFIRMED**, and this stops: no third attempt on this hypothesis, no
further candidate reached for in this run, no gate, tolerance or golden moved, and `rfx/`
carries no diff whatever the verdict. If it is confirmed, the fix is *described* and left
to the PI; no extractor code changes in this PR either way.

Two partial outcomes are named now so they are not presented as successes later:

* **Leg 1 passes, leg 2 fails** → the arithmetic reproduces the offset but the content does
  not behave like the mode it is labelled with. Reported as NOT CONFIRMED with the
  discrepancy, not as a confirmation with a caveat.
* **Both fail, and `D_pred ≪ D_meas`** → the named modes are present but the extractor is
  blind to them. Reported as NOT CONFIRMED plus the positive statement that follows from
  it: the offset lives in the TE10 coefficient itself. Whatever `A₁₀`, `B₁₀` and the
  implied forward/backward split do across the four planes is then the result, and it is
  reported as a measurement, not turned into a new hypothesis inside this run.

## 6. The cheap falsifier, with its numbers fixed before the run

Two extra coarse thru runs, ~6 s each, with the record plane moved off 3 cells (7.62 mm) to
**4 cells (10.16 mm)** and to **11 cells (27.94 mm)** from the port plane. 10.16 mm is the
smallest lattice-aligned distance beyond 3 decay lengths of TE30 at the coarse rung
(3/α = 9.69 mm); 27.94 mm is beyond 3 decay lengths of TE20 (3/α = 25.90 mm) and stays
clear of the probe plane at 25.40 mm and of the DUT window at 48.26 mm. The fixture builder
carries one `ref_offset` for both ports, so both planes move together; the receiving plane
is measured clean in §0 and its cleanliness at the new position is re-checked as a control.

Predicted **multiplicative** reduction of the contamination term `D_pred`, `exp(−α·Δd)`
with `Δd` the added distance, at bin 16 and the coarse rung — stated for each candidate so
the prediction stands whichever mode the measurement names:

| named mode | plane at 10.16 mm (Δd = 2.54 mm) | plane at 27.94 mm (Δd = 20.32 mm) |
|---|---|---|
| TE20 (α = 115.84 /m) | ×0.745 | ×0.0950 |
| TE30 (α = 309.67 /m) | **×0.455** | **×1.85e-3** |
| TE50 (α = 552.02 /m) | ×0.246 | ×1.34e-5 |

On a CONFIRMED verdict the same factors apply to the whole measured excess `D_meas`, since
the two are then equal within 1.25. On a NOT CONFIRMED verdict they apply only to `D_pred`,
where they test the projection measurement itself rather than the hypothesis; `D_meas` at
the moved planes is reported beside them without being fitted. Either way the falsifier is
run and its numbers are reported against this table.

## 7. Controls, fixed here

1. **Mirror-position control.** The receiving port's reference plane sits 7.62 mm from its
   own (inactive) port plane. The named modes' coefficients there must be **≤1e-4** of the
   driven plane's. A failure means the content is a property of the geometry, not of the
   active source, and the hypothesis is not about the source.
2. **Plane independence of TE10.** `A₁₀` and `B₁₀` at the three planes ≥25 mm from the
   driven source must agree to the level §0 records for the modal power there, ≤8.2e-4
   (coarse) relative, after the propagation phase `exp(−jβΔx)` is removed. A failure means
   the TE10 coefficient is not a stable reference and leg 1's denominator is unreliable.
3. **Every preflight line and warning of every run is quoted verbatim** beside its numbers
   (repo rule: preflight output is part of the result). On this fixture the record-length
   advisory fires on both ports at every rung (`T/tau_far = 2.139`); PR #1081 §5 already
   ran `num_periods` 40 → 160 and moved the excess ≤1 %, so it is quoted, not re-tested.

## 8. Artifact and provenance

`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json` — a NEW
file. Neither `suspects.json` (PR #880) nor `transmission_tilt.json` (PR #1081) is read for
its measurements or overwritten; §0's table is quoted from the latter and cited by key.
Provenance recorded: commit, JAX/NumPy versions and backend, precision, `dt`, `n_steps`,
grid shape, CPML layers, plane indices and positions, the source fixture's sha256.

## 9. Revisions

| date | change |
|---|---|
| 2026-09-16 | first version, committed before any projection coefficient was computed |
