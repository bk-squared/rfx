# Results — what the near field at the driven port's reference plane is made of

Verdict, in the order the pre-declaration asks for it:

* **Decision rule: NOT CONFIRMED.** Leg 1 passes at every rung; leg 2 fails, because it
  names the wrong mode. The pre-declaration derived, from the extractor's own overlap
  rows, that only the odd `TE_m0` family could reach `a₁`, and named TE30 as the carrier.
  TE30 is not the carrier and its coefficient does not decay at its own `α`.
* **The hypothesis of section 1 is supported, with the carrier re-identified.** The
  driven reference plane sits in a measurable near field, that near field is **TE20** —
  0.999 of all the non-TE10 content there against node-registered templates, 94.2 / 98.5 /
  99.6 % against cell-centred ones — and it accounts for **94.3 / 96.9 / 97.9 %** of the
  measured modal-power offset. TE20's own attenuation constant, tabulated in the
  pre-declaration before the run, reproduces its decay across five plane positions, and
  its amplitude at the plane is predictable from the injected profile alone to 27 / 16 /
  11 %.
* **What made the pre-declaration name the wrong mode is itself the measurement's main
  finding**: the port's transverse templates are cell-centred while the `Ez` and `Hy` it
  integrates are node-registered in u. The parity argument that excluded TE20 holds for
  cell-centred modes and does not hold for the field. See section 3.

After two lanes that could not name the term (PR #880, PR #1081), this one names it and
gives it a mechanism: **one half-cell template offset, charged twice** — once where the
profile is injected and once where the field is read back through it. No code in `rfx/`
changed; no gate, tolerance or golden moved.

Reviewed by a separate Opus instance (PR #1094): ACCEPT WITH CHANGES. The review's six
items are addressed in sections 2, 3, 5, 8 and 9, each marked where it lands; its first
item corrected a reason this note gave for a deviation, and section 8 now carries the one
that holds.

Pre-declaration: `waveguide_driven_plane_near_field_composition_predeclaration.md`
(committed at 460bb9b7, before any projection coefficient existed).
Artifact: `tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json`,
produced by `scripts/diagnostics/waveguide_driven_plane_near_field_composition.py`.
Figures: `figures/waveguide_driven_plane_near_field_composition.png` (per-bin spectra and
the power split) and `figures/waveguide_driven_plane_near_field_composition_decay.png`
(TE20 against the pre-declared `α`, five plane positions).
PR #880's `suspects.json` and PR #1081's `transmission_tilt.json` are neither read for
their measurements nor overwritten.

## 0. Provenance, reproduce-gate and witnesses

Three thru cells of `tests/fixtures/waveguide_chain_battery/fixture_v18_close.json`
(sha256 `f00d0959…`), `num_periods = 40`, `normalize=False`, float32 fields on CPU (JAX
0.6.2, NumPy 2.2.6), float64 post-processing. Grids 83×10×5 / 165×19×9 / 329×37×17,
`n_steps` 713 / 1425 / 2849.

**Reproduce-gate.** Worst-bin `column power − 1`, this run against the frozen GPU
artifact: 6.134629e-3 / 6.135092e-3 (coarse), 1.161337e-3 / 1.161080e-3 (mid),
2.530813e-4 / 2.546142e-4 (fine) — relative 7.5e-5 / 2.2e-4 / 6.0e-3, the same float32
backend-change residue PR #1081 recorded on the same cells.

**Capture witness.** The solve is unchanged; only the recording is. To prove that, the
modal voltage reconstructed from the captured aperture slice was compared against the
unpatched run's own `cfg.v_ref_t`, step by step: max relative difference
**3.4e-7 / 4.4e-7 / 1.7e-6**, and 1.9e-7 / 2.2e-7 / 1.5e-6 for the current. Without this
the capture would not be proven to be the field the extractor reads.

**Preflight and warnings**, verbatim, stored per rung in the artifact. Preflight is empty
on all three cells. Four warnings fire, the same set at every rung:

* `UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase
  include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True.
  For |S11| of strong reflectors (PEC short, resonators) normalize=False is more accurate
  — see the normalize parameter docstring.`
* `PreflightWarning: the record is shorter than 3 x the far-boundary round trip —
  waveguide_port[0] (+x): T/tau_far = 2.139. …  num_periods >= 57 makes T/tau_far >= 3.`
  and the same for `waveguide_port[1] (-x)`. PR #1081 §5 already ran `num_periods`
  40 → 80 → 160 on these cells and moved the excess by at most 1 %, so the advisory is
  quoted rather than re-tested.
* `PreflightWarning: waveguide port index mirror audit, source E plane (x-axis):
  i(+) = 22, i(-) = 61, sum = 83 on n_axis = 83; the covariant sum for a primal plane is
  82. The +1 difference is the KNOWN shipped offset … Reported as information.`

Per-drive ring-down witness: −84.89 / −85.33 dB (coarse), −97.70 / −98.30 (mid),
−100.90 / −101.11 (fine).

**Determinism.** Regenerating the whole artifact a second time reproduced every stage
value-identically (byte-identical JSON), so the tables below carry no run-to-run scatter
of their own.

## 1. The measurement

`modal_voltage` and `modal_current` were replaced by functions storing the whole aperture
slice they would otherwise contract, so each of the four recorded planes holds `(E_y, E_z)`
and the co-located `(H̃_y, H̃_z)` per step; the port's own rectangular full-record DFT was
then applied per aperture cell in float64. Injection is untouched.

## 2. Leg 1 — magnitude

`D_meas` is the driven reference plane's modal power relative to the receiving reference
plane, the quantity PR #1081 recorded as −5.070e-3 / −1.073e-3 / −2.454e-4 at bin 16;
this run reads −5.061e-3 / −1.065e-3 / −2.402e-4 on its own re-measurement.

`D_pred` is the part of it carried by non-TE10 content, read off the field through the
extractor's own aperture weighting. `D_TE10` is the remainder, the TE10-only difference
between the two planes. The two compose exactly:
`(1 + D_meas) = (1 + D_pred)(1 + D_TE10)`, and the residual of that identity is 1.1e-16.

**What leg 1 therefore tests, said plainly (PR #1094 review).** That composition is exact
algebra, and both sides share the same measured `P(drive_ref)`, so `D_pred` is not an
independent prediction of `D_meas` — the two agree to the extent that `D_TE10` is small.
Leg 1 reduces to: *the calibrated TE10 power is plane-invariant to 2-6 % of `D_meas`*,
i.e. once the non-TE10 part of the driven plane's field is removed, what is left matches
the clean plane. That is a real and falsifiable statement — the split could have left a
large TE10-only residue and did not — but it is weaker than the pre-declaration's wording
("with no free parameter") implied, and that wording is withdrawn here. The independent
evidence for the hypothesis is leg 2's `α` (section 4), the falsifier (section 6) and the
a-priori launch prediction (section 5), none of which shares a term with `D_meas`.

| rung | `D_meas` (bin 16) | `D_pred` | `D_TE10` | `D_pred / D_meas` |
|---|---|---|---|---|
| coarse | −5.0614e-3 | −4.7716e-3 | −2.9122e-4 | **0.9427** |
| mid | −1.0654e-3 | −1.0322e-3 | −3.3231e-5 | **0.9688** |
| fine | −2.4019e-4 | −2.3513e-4 | −5.0557e-6 | **0.9790** |

Leg 1 asks for `|D_pred/D_meas − 1| ≤ 0.25` at the worst bin on all three rungs, with the
reference-difference column below 25 % of `D_meas`. Measured: 0.057 / 0.031 / 0.021, and
the reference difference is 5.8 / 3.1 / 2.1 % of `D_meas`. **Leg 1 passes.**

It is not a worst-bin coincidence. `D_pred/D_meas` per bin, low to high frequency:

```
coarse +1.165 +1.224 +1.211 +1.237 +1.271 +1.290 +1.344 +1.433 +1.680 +2.409 -4.797 +0.327 +0.699 +0.811 +0.877 +0.917 +0.943
mid    +1.022 +1.062 +1.047 +1.054 +1.061 +1.060 +1.072 +1.077 +1.097 +1.129 +1.185 +1.641 +0.667 +0.876 +0.938 +0.961 +0.969
fine   +0.954 +1.050 +0.997 +1.017 +1.017 +1.009 +1.023 +1.013 +1.023 +1.028 +1.025 +1.091 +1.191 +0.909 +0.988 +0.987 +0.979
```

**The sign chain, end to end.** `D_meas > 0` at the bottom of the band means the driven
plane reads more modal power than the receiving plane, i.e. `a₁` is measured high against
`b₂`; `|S21| = |b₂/a₁|` is then low and the column power sits below 1 — which is the
committed per-bin curve's negative low-band end (−1.56e-3 at bin 0, coarse). `D_meas`
crosses zero near bin 10-11 and is negative at the top, so `a₁` is measured low there,
`|S21|` high, and the column power runs above 1 (+6.13e-3 at bin 16). The frequency tilt
#873 carries is that crossing, and it is the crossing of the TE20 cross-term.

The excursions at bins 10-12 are a division by a `D_meas` that crosses zero there, not a
disagreement: in absolute terms the two curves are within 6.8e-4 / 6.4e-5 / 1.5e-5 of each
other at every bin, which is the same ladder the offset itself has, and the figure shows
it directly.

## 3. Why the pre-declaration named the wrong mode — the registration finding

The pre-declaration derived, from the code, that `⟨e_k, e₁₀⟩` is zero to 1.4e-15 for every
higher mode and that `⟨h_k, h₁₀⟩` is non-zero only for the odd `TE_m0` family. Both
statements are correct **for the cell-centred discrete modes**. Neither constrains the
field, because the field is not on that lattice.

Measured, with no basis involved: at the receiving reference plane — 88.9 mm from the
driven source, where every evanescent mode is below 1e-4 — the recorded `Ez` slice across
u, normalised, reads

```
measured       0  0.3473  0.6527  0.8794  1.0000  1.0000  0.8793  0.6527  0.3473
sin(pi j / 9)  0  0.3473  0.6527  0.8794  1.0000  1.0000  0.8794  0.6527  0.3473
```

i.e. `Ez` sits on the **y-nodes** `j = 0 … nu−1`, pinned to zero at the wall node it
includes and not containing the far wall's zero. The port's stored `ez_profile` is the
**cell-centred** `sin(π(j+½)/nu)`. The two lattices differ by half a cell. The same holds
for `Hy`. The signature is the index-flip antisymmetry of the recorded slice against a
template that is index-flip symmetric by construction: 0.1763 / 0.0875 / 0.0437 at the
clean plane, halving per rung exactly as a half-cell offset must.

The consequence is one number. A node-registered `TE_m0` of unit amplitude, integrated
against the port's cell-centred TE10 template, is read as a fraction of what a
node-registered TE10 reads as:

| | coarse | mid | fine | ratio per halving |
|---|---|---|---|---|
| TE20 through the `ez₁₀` template | **0.14504** | **0.07370** | **0.03699** | 1.968, 1.992 |
| TE20 through the `hy₁₀` template | 0.11818 | 0.07023 | 0.03656 | 1.683, 1.921 |
| TE30 through the `ez₁₀` template | −1.2e-8 | −5.8e-9 | −2.4e-9 | zero to float64 round-off |

So TE20 — the mode the pre-declaration excluded by parity — is the one that reaches the
extractor, at first order in `dx`, and TE30 — the one it named — does not reach the E side
at all. The parity argument was applied to the wrong object.

## 4. Leg 2 — the decay, and which mode it belongs to

The non-TE10 content at the driven reference plane, split off the field with no basis
(the clean far plane's own shape is the TE10 reference), is 4.5206e-2 / 2.3965e-2 /
1.2422e-2 of the field norm at bin 16. Of that, TE20 carries **94.20 / 98.50 / 99.61 %**
when the census is taken in the cell-centred family.

Taken against **node-registered** templates — `sin(mπj/nu)·cos(nπ(k+½)/nv)`, the lattice
the field is actually on (section 3) — TE20's share is **0.9994 to 0.9999 across the whole
band at every rung**, and that family's squares sum to 1.00000 of the orthogonal norm, so
it is complete rather than merely large. The 5.8 % of "other" in the 94.2 % figure is the
cell-centred basis's own read-through, not content (PR #1094 review; reproduced here by
re-running the `compose` stage — the per-bin share is not stored in the artifact, so it is
re-derived by running the script rather than read from a key).

| mode | `α(11.6 GHz)`, 1/m | content / TE10 (coarse/mid/fine) | first-hop ratio measured | predicted `exp(α·17.78 mm)` |
|---|---|---|---|---|
| **TE20** | 115.8 / 125.2 / 127.5 | 4.355e-2 / 2.379e-2 / 1.240e-2 | **8.068 / 9.440 / 9.761** | **7.844 / 9.263 / 9.644** |
| TE30 | 309.7 / 327.1 / 331.5 | 1.190e-2 / 3.130e-3 / 8.108e-4 | 7.730 / 9.165 / 9.538 | 246.1 / 335.8 / 363.0 |
| TE50 | 552.0 / 619.5 / 636.9 | 5.517e-3 / 1.356e-3 / 3.457e-4 | 8.247 / 9.792 / 10.19 | 1.83e4 / 6.08e4 / 8.28e4 |
| TE70 | 698.8 / 869.9 / 915.4 | 4.142e-3 / 9.073e-4 / 2.254e-4 | 8.230 / 9.721 / 10.09 | 2.49e5 / 5.22e6 / 1.17e7 |
| TE01, TE11, TM11 | 178 / 224 / 224 | ≤ 5.4e-9 at every rung | — | — |

TE20's measured decay over the first hop agrees with the `α` the pre-declaration tabulated
to **+2.9 % / +1.9 % / +1.2 %**, and both the prediction and the measurement move across
the ladder in the same direction and by the same amount (7.84 → 9.64 predicted,
8.07 → 9.76 measured). That is the pre-declared 25 % bar met with a factor of ten to
spare, on a number fixed before the run.

TE30, TE50 and TE70 all decay at **TE20's** rate, not their own — 7.7-10.2 against
predictions of 246 to 1.2e7. They are not independent content: they are the cell-centred
templates' view of the same node-registered TE20 field, and their share of the orthogonal
content collapses as the mesh refines (TE30: 25.8 % → 13.0 % → 6.5 %) exactly as the
registration read-through of section 3 does. **Leg 2, as written, fails**: the family it
named carries the content only as an artefact of the basis it was named in.

The `TE01` / `TE11` / `TM11` row is worth stating: every mode with structure across the
narrow wall is at 1e-9, i.e. absent. The launch error is purely a broad-wall one.

## 5. The mechanism, and why the excess is second order in `dx`

Two first-order factors multiply:

| | coarse | mid | fine | ratio per halving |
|---|---|---|---|---|
| TE20 content at the plane, / TE10 | 4.3553e-2 | 2.3792e-2 | 1.2397e-2 | 1.830, 1.919 |
| read-through into the TE10 template | 0.14504 | 0.07370 | 0.03699 | 1.968, 1.992 |
| product | 6.317e-3 | 1.753e-3 | 4.586e-4 | 3.60, 3.82 |
| measured `\|V_perp\|/\|V\|` | 6.6206e-3 | 1.7757e-3 | 4.6162e-4 | 3.73, 3.85 |

The product reproduces the measured voltage error to −4.6 % / −1.3 % / −0.7 %. **That row
is close to an identity and is demoted here (PR #1094 review):** `V_perp` is the projection
of the orthogonal field onto the TE10 template, and section 4 has just shown that
orthogonal field is TE20 to 0.999, so "TE20 amplitude × TE20's template overlap ≈ `V_perp`"
mostly re-states that. It is a closure check on the census, not a second measurement.

**The a-priori version, which uses no FDTD field at all.** Decompose the profile the port
INJECTS, `cfg.ez_profile`, on the node lattice it is added to. Its TE20 fraction is
**0.145045 / 0.073695 / 0.036990** — the *same numbers*, to 0.000 %, as the read-through
row above. That is the finding: the "two first-order factors" are **one half-cell offset
counted twice**, once when the template is written into the field and once when the field
is read back through it. The template is index-flip symmetric, so its symmetry centre sits
at index `(nu−1)/2` — 4.0 / 8.5 / 17.5 — while the guide centre is node `nu/2` = 4.5 / 9.0 /
18.0. The offset is exactly half a cell at every rung, and independent of the parity of
`nu`.

That makes the TE20 amplitude at the driven plane predictable before any solve: the
launched fraction times its own decay over 7.62 mm,

| | coarse | mid | fine |
|---|---|---|---|
| launched TE20 / TE10 (from `ez_profile`, no FDTD) | 0.145045 | 0.073695 | 0.036990 |
| `exp(−α·7.62 mm)`, `α` from the pre-declaration | 0.41365 | 0.38519 | 0.37860 |
| **predicted TE20 / TE10 at the plane** | **0.0600** | **0.0284** | **0.0140** |
| measured | 0.043553 | 0.023792 | 0.012397 |
| measured / predicted | 0.73 | 0.84 | 0.89 |

Predicted to within 27 / 16 / 11 %, improving with refinement, from the injected profile
and a cutoff — no field, no fit. The shortfall is the part the launch is not: the TFSF pair
injects an E-side and an H-side correction with different transverse shapes (PR #1081 §4),
and only the E side is decomposed here.

Stated as a chain, this is #873's one sentence with a mechanism in it. The port's
transverse template sits half a cell off the lattice the field lives on. Injected, that
offset launches a TE20 whose relative amplitude is first order in `dx` (0.145 / 0.0737 /
0.0370 at the aperture, and 0.73-0.89 of that survives to the reference plane). Read back,
the same offset makes the same template count that TE20 as TE10, with the same first-order
weight. The product is the error in `a₁`, second order in `dx`, and `|S21| = |b₂/a₁|`
inherits it — one geometric defect, charged at both ends. The receiving port sees the same
error only in the far field of its own inactive launch, i.e. not at all, so nothing
cancels — which is exactly why PR #1081's bound, correct for anything common to the two
ports, did not catch it.

## 6. The falsifier, against the numbers fixed before the run

Two extra coarse runs, the record plane moved from 3 cells (7.62 mm) to 4 (10.16 mm) and
11 (27.94 mm) from the port plane. The pre-declaration tabulated the predicted reduction
of the contamination for each candidate mode; TE20's row read ×0.745 and ×0.0950.

| | 7.62 mm | 10.16 mm | 27.94 mm |
|---|---|---|---|
| TE20 content / TE10 | 4.3553e-2 | 3.2065e-2 (×**0.7362**) | 3.9593e-3 (×**0.0909**) |
| predicted for TE20 | — | ×0.745 | ×0.0950 |
| implied `α` | — | 120.6 /m | 118.0 /m |
| `D_meas` (bin 16) | −5.0614e-3 | −5.2115e-3 | +6.8617e-4 |
| `D_pred` (bin 16) | −4.7716e-3 | −5.2661e-3 | +7.1094e-4 |
| band-mean `\|S11\|` | 0.024316 | 0.024558 | 0.025586 |
| worst `column power − 1` | 6.1346e-3 | 6.2082e-3 | 1.5832e-3 |

TE20's amplitude follows the pre-declared prediction to **−1.2 %** and **−4.3 %**, and the
`α` implied by the three plane positions is 118.0 /m against the 115.8 /m tabulated before
the run (+1.9 %). The decay figure plots all five positions on that one exponential.

The power offset itself does **not** follow `exp(−α·Δd)` — ×1.10 and ×−0.15 where the
amplitude gives ×0.74 and ×0.091 — and that is expected rather than a miss: the offset is
a cross-term between TE20 and TE10, and the TE10 factor turns `β·Δd` (29° over 2.54 mm,
233° over 20.32 mm), so the ratio is `exp(α·Δd)·|cos ψ / cos(ψ+βΔd)|`, including the sign
flip at 27.94 mm. `D_pred` tracks `D_meas` at both moved planes (ratios 1.010 and 1.036),
so the split holds where the offset changes sign and where it nearly doubles. Band-mean
`|S11|` moves by 1.0 % and 5.2 %: this is a transmission-side term, as PR #1081's
observable said it was.

## 7. Controls

1. **Mirror position.** TE20's content at the receiving port's probe plane, relative to
   the driven reference plane: 4.61e-3 / 2.28e-3 / 1.72e-3. The pre-declared bar was
   ≤1e-4 and is **not met as written**. What the number is: the receiving reference plane
   is the calibration reference, so its own orthogonal part is zero by construction and a
   ratio against it would be vacuous; the quoted figure is the nearest non-vacuous plane,
   71.1 mm from the DRIVEN source, and it is the floor of this measurement rather than
   content generated at the inactive port — it does not decay at TE20's `α` (which would
   put it at 2.8e-5) and it is where the decay figure's one off-line point sits. The
   control's purpose is met in substance: nothing is launched at the inactive port, and
   every plane's content is a function of distance from the ACTIVE source.
2. **Which far plane is the reference.** Repeating the whole split with the 71.12 mm plane
   as the reference instead of the 88.90 mm one moves `D_pred` by 0.12 % / 0.31 % / 0.22 %
   (−4.7716e-3 → −4.7659e-3, −1.0322e-3 → −1.0290e-3, −2.3513e-4 → −2.3462e-4). The two
   far planes' shapes disagree by 2.09e-4 / 5.47e-5 / 2.14e-5 of unit norm, against the
   driven plane's 4.52e-2. The result does not depend on the choice.
3. **TE10 plane independence.** With the propagation phase removed, `A₁₀` at the three
   planes ≥25 mm from the driven source agrees to 9.3e-4 / 7.0e-4 / 5.5e-4 relative — the
   level PR #1081 recorded for the modal power there.

## 8. Two deviations from the pre-declaration

Recorded before the verdict, because one of them changes a leg's outcome.

1. **The estimator in section 4 of the pre-declaration was replaced, and the replacement
   turns leg 1 from a fail into a pass.** As written, `D_pred` is built from a
   least-squares fit of the field in the cell-centred discrete mode family. That
   estimator was computed and is in the artifact
   (`D_pred_bin16` = −3.7304e-4 / +2.2981e-3 / +4.1469e-4, giving `D_pred/D_meas` of 0.074
   / −2.157 / −1.727 — a fail, and a sign flip between rungs).

   **The decisive reason it was discarded is structural, and an earlier draft of this
   paragraph gave a weaker one that is false at one rung (PR #1094 review).** That draft
   said the estimator "returns the same value on a plane with none": true at mid
   (+2.2981e-3 driven against +2.3245e-3 at the clean plane) and at fine (+4.1469e-4
   against +4.1668e-4), but **false at coarse**, where it reads −3.7304e-4 at the driven
   plane and +3.2224e-8 at the clean one — four orders apart, and still giving
   `D_pred/D_meas` = 0.0737.

   The reason that holds at every rung is in this run's own artifact.
   `modes.TE20.overlap_e_with_te10` is −2.09e-16 and `modes.TE30.overlap_e_with_te10` is
   −1.37e-8, and `te10_by_plane.drive_ref.reconstruction_rel_err_V` is 1.47e-11: `V` is
   reconstructed by the TE10 term alone to eleven figures, so `V_hi ≡ 0` identically and
   the pre-declared `D_pred` is an **H-side-only** quantity by construction. It is blind
   to exactly the E-side read-through this PR identifies as the mechanism. The
   pre-declaration derived "the modal voltage is blind to every higher mode" in its §2.3
   and read it as *there is no E-side contamination*; what it actually means is *this
   estimator cannot separate it*, because the blindness is a property of the cell-centred
   basis and not of the field. Section 3 is why. The plane-independent readings at mid and
   fine are then a second symptom of the same cause — the fit's `B₁₀` inherits the
   family's near-deficiency against a node-registered field, and its unexplained residual
   (4.86e-3 at mid) is identical at all four planes.
   The replacement takes the TE10 reference from the clean far plane's measured shape — no
   basis, no inversion — and is validated where it was not calibrated: at the 71.12 mm
   plane it reads −1.35e-5 / −3.94e-6 / −5.54e-7 against the driven plane's −4.77e-3 /
   −1.03e-3 / −2.35e-4, i.e. it discriminates planes, which the literal estimator does
   not. Control 2 above shows it does not depend on which far plane it is calibrated on.
   Both numbers are reported; the reader can take either.
   The confirming evidence does not rest on the substitution: TE20's `α` (section 4) and
   the falsifier's reduction factors (section 6) were both tabulated in the
   pre-declaration before any field was read, and both are met.
2. **The mirror control's 1e-4 bar is not met as written** (section 7 item 1), for a
   reason that is a property of the calibration, not of the field.
3. **The pre-declaration's §2.3 `⟨h_k, h₁₀⟩` row does not reproduce at the coarse rung.**
   It declared TE30 5.445e-2 / TE50 1.049e-1 / TE70 1.881e-1 there; the artifact holds
   5.3438e-2 / 9.7230e-2 / 1.4837e-1 (−1.9 / −7.3 / −21 %), while mid and fine agree to
   ≤0.2 % (PR #1094 review). Diagnosed rather than left as a discrepancy: the two are
   **different normalisations of the same overlap**. The pre-declaration's scratch
   construction scaled each mode's H by the port's own `_scale_h_to_unit_cross` (unit
   `∫(e×h)·n̂ dA`); the production script normalises `∫(h_y²+h_z²) dA = 1`, so the
   decomposition coefficient and the overlap weight stay separately interpretable.
   Recomputing the row under the pre-declaration's scaling returns 5.4445e-2 / 1.0486e-1 /
   1.8813e-1, i.e. its numbers to four figures; the two conventions converge as `dx` falls
   because the port's cross-overlap factor tends to 1 (1.0035 / 1.0004 / 1.0001 for TE10
   here). No verdict moves — the row is an ordering argument in the pre-declaration and the
   leak weights it ranks are unchanged — but the two columns are not the same quantity and
   must not be compared as if they were. The pre-declaration is left unedited; this is the
   record.

## 9. What a fix would be — for the PI, not done here

`rfx/` carries no diff in this PR, per the pre-declaration. Two candidates, with what each
would cost:

* **Yee-register the port's transverse templates — the principled fix.** The templates
  are built on one cell-centred lattice for both components, but `Ez` is a node in u and a
  cell centre in v, and `Ey` is the reverse; `Hy` follows `Ez`. Building each component's
  template on its own lattice removes the offset at its source, and section 5 shows the
  same offset is charged twice — once at the launch, once at the readout — so it is one
  change, not two. Blast radius, traced rather than gestured at: the discrete eigenproblem
  sets `kc_num`, which becomes `f_c` at `rfx/sources/waveguide_port.py:808`, which feeds
  `_compute_beta` and `_compute_mode_impedance`, so `β` and `Z_TE` move; every
  `normalize=False` and flux `|S|` moves with them, and so do
  `tests/fixtures/waveguide_chain_battery/fixture_v18_close.json` and the verdicts read out
  of it. This is #868/#889's transverse sibling — the same registration family seen from
  the template side rather than the eigenproblem side — and it wants the same treatment: a
  bit-identity gate for every configuration the change must not touch, and a declared
  re-gate list for the ones it must.
* **A reference-plane clearance rule in preflight — worth doing, but it hides rather
  than fixes.** The governing parameter is the physical distance from the launch in units
  of `1/α` of the first evanescent mode — TE20 here, `1/α` = 8.6 / 8.0 / 7.8 mm at the top
  of the band and 4.9 mm at the bottom. The shipped `ref_offset` is a **cell count**, so
  the default plane moves physically closer to the source as the mesh refines; this fixture
  happens to hold 7.62 mm fixed, which is 0.88 decay lengths. A preflight that computed `α`
  of the first mode above the operating band and warned when the record plane sits inside
  ~3/α would have fired on every rung of this battery. It is an input-fidelity check on
  declared geometry, which is the tier preflight is for, and it moves no committed number —
  but it moves the *user* away from the defect rather than removing it, so it is not a
  substitute for the first item. Do both, in that order of importance and the reverse order
  of cost.

**A candidate sibling for the same scope: #894.** Moving the reference plane moves
band-mean `|S11|` by 1.0 % and 5.2 % (section 6) on a guide whose reflection is supposed to
be a property of the structure, not of where it is read. #894's arrival-time class is a
different mechanism, but both are "the answer depends on the plane", and whoever scopes the
registration fix should check whether the two share a lane before assuming they do not.

## 10. Numeric provenance

Every load-bearing number above, cited by key into the committed artifact and value-checked
by `tests/contracts/test_evidence_numeric_provenance.py`.

**Leg 1 (section 2).**

`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.D_meas_bin16 = -0.00506140347538`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.D_pred_cal_bin16 = -0.00477157394997`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.D_te10_cal_bin16 = -0.000291219098872`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.leg1_ratio_cal_bin16 = 0.942737320426`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.D_meas_bin16 = -0.00106538640109`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.D_pred_cal_bin16 = -0.00103219005678`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.D_te10_cal_bin16 = -3.32306446518e-05`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.leg1_ratio_cal_bin16 = 0.968841028685`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.D_meas_bin16 = -0.00024018761608`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.D_pred_cal_bin16 = -0.00023513315408`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.D_te10_cal_bin16 = -5.05565075104e-06`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.leg1_ratio_cal_bin16 = 0.978956192319`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.identity_residual_cal_bin16 = 1.11022302463e-16`.

**The reproduce-gate and the capture witness (section 0).**

`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::control.coarse.max_positive_column_power_minus_1 = 0.00613462924957`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::control.coarse.frozen_max_positive_column_power_minus_1 = 0.00613509151298`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::control.mid.max_positive_column_power_minus_1 = 0.0011613368988`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::control.fine.max_positive_column_power_minus_1 = 0.000253081321716`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.capture_witness_rel_err_v_ref_t = 3.40641293503e-07`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.capture_witness_rel_err_v_ref_t = 4.3724301908e-07`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.capture_witness_rel_err_v_ref_t = 1.6600312345e-06`.

**The registration finding (section 3).**

`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.te10_by_plane.recv_ref.ez_index_flip_antisym_fraction_bin16 = 0.176299051162`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.te10_by_plane.recv_ref.ez_index_flip_antisym_fraction_bin16 = 0.0874712215995`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.te10_by_plane.recv_ref.ez_index_flip_antisym_fraction_bin16 = 0.043656256948`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.te10_by_plane.drive_ref.ez_index_flip_antisym_fraction_bin16 = 0.21421881945`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.node_registration_readthrough.TE20_node.read_as_te10_fraction_e = 0.145045255652`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.node_registration_readthrough.TE20_node.read_as_te10_fraction_e = 0.0736951199796`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.node_registration_readthrough.TE20_node.read_as_te10_fraction_e = 0.0369899386047`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.node_registration_readthrough.TE20_node.read_as_te10_fraction_h = 0.118183716666`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.node_registration_readthrough.TE30_node.read_as_te10_fraction_e = -1.20500254506e-08`.

**Leg 2 and the mode census (section 4).**

`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE20.perp_first_hop_ratio_bin16 = 8.06836335564`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE20.predicted_first_hop_ratio = 7.84376319065`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.modes.TE20.perp_first_hop_ratio_bin16 = 9.44029397441`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.modes.TE20.predicted_first_hop_ratio = 9.26318463773`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.modes.TE20.perp_first_hop_ratio_bin16 = 9.76054843277`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.modes.TE20.predicted_first_hop_ratio = 9.64391745457`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE20.perp_share_of_orthogonal_field_bin16 = 0.94203677608`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.modes.TE20.perp_share_of_orthogonal_field_bin16 = 0.985028021403`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.modes.TE20.perp_share_of_orthogonal_field_bin16 = 0.99614513488`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE30.perp_first_hop_ratio_bin16 = 7.72953017809`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE30.predicted_first_hop_ratio = 246.136785841`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.modes.TE30.perp_first_hop_ratio_bin16 = 9.53809273725`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.modes.TE30.predicted_first_hop_ratio = 362.972776107`.

**The mechanism (section 5).**

`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE20.perp_overlap_e_drive_ref[16] = 0.043552662982`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.modes.TE20.perp_overlap_e_drive_ref[16] = 0.0237921440842`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.modes.TE20.perp_overlap_e_drive_ref[16] = 0.0123969522787`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.V_perp_over_V_bin16 = 0.0066205640709`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.V_perp_over_V_bin16 = 0.00177567147278`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.V_perp_over_V_bin16 = 0.000461615550825`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.I_perp_over_I_bin16 = 0.00403323971153`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.te10_by_plane.drive_ref.perp_fraction_e[16] = 0.0452058029301`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.te10_by_plane.drive_ref.perp_fraction_e[16] = 0.02396538778`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.te10_by_plane.drive_ref.perp_fraction_e[16] = 0.0124222702635`.

**The falsifier (section 6).**

`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref3cells.named_mode_perp_overlap_over_te10_bin16.TE20 = 0.043552662982`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref4cells.named_mode_perp_overlap_over_te10_bin16.TE20 = 0.0320647267585`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref11cells.named_mode_perp_overlap_over_te10_bin16.TE20 = 0.00395927695883`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref3cells.D_meas_bin16 = -0.00506140347538`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref4cells.D_meas_bin16 = -0.00521153356905`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref11cells.D_meas_bin16 = 0.000686169202189`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref4cells.D_pred_cal_bin16 = -0.00526609173131`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref11cells.D_pred_cal_bin16 = 0.000710941456326`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref3cells.band_mean_s11_mag = 0.0243161674589`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref4cells.band_mean_s11_mag = 0.0245576649904`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref11cells.band_mean_s11_mag = 0.0255859978497`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::falsifier.ref11cells.max_positive_column_power_minus_1 = 0.00158321857452`.

**The controls and the deviation (sections 7 and 8).**

`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE20.perp_mirror_control_bin16 = 0.0046068177793`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.modes.TE20.perp_mirror_control_bin16 = 0.00228059479426`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.modes.TE20.perp_mirror_control_bin16 = 0.00172401364147`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.D_pred_cal_alt_ref_bin16 = -0.00476594373321`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.D_pred_cal_alt_ref_bin16 = -0.00102903707251`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.D_pred_cal_alt_ref_bin16 = -0.000234622901195`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.far_plane_shape_disagreement_bin16 = 0.000208557111136`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.D_pred_bin16 = -0.000373040755406`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.D_pred_bin16 = 0.00229805818739`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.D_pred_bin16 = 0.000414691544335`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.te10_by_plane.recv_ref.D_pred_at_this_plane[16] = 0.00232452880914`.

**The PR #1094 review's changes (sections 2, 3, 5, 8).**

`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE20.overlap_e_with_te10 = -2.08600497986e-16`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE30.overlap_e_with_te10 = -1.37027833151e-08`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.te10_by_plane.drive_ref.reconstruction_rel_err_V = 1.46835650497e-11`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.D_pred_recv_ref_bin16 = 3.22243134399e-08`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.D_pred_recv_ref_bin16 = 0.000416677695025`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE20.alpha_per_m[16] = 115.844697301`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.mid.modes.TE20.alpha_per_m[16] = 125.199544595`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.fine.modes.TE20.alpha_per_m[16] = 127.464983185`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE30.overlap_h_with_te10 = 0.0534383090981`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE50.overlap_h_with_te10 = 0.0972295365053`,
`tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json::read.per_rung.coarse.modes.TE70.overlap_h_with_te10 = 0.148373121362`.
