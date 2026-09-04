# The auxiliary absorber: what it reflects, and what depth that costs

**Status:** MEASUREMENT + PRE-DECLARATION (no case re-run yet) · **Opened:** 2026-09-04
**Branch:** `agent/issue-888-aux-absorber` (worktree `~/Documents/rfx-worktrees/aux-absorber`, base `origin/main` @ `b59e1d9`)
**Issue:** #888 · **PI decision A (2026-09-04):** deepen the auxiliary absorber, re-derive
its profile from a reflection target, and fix `predict_settling`'s echo cancellation.
**PI directive carried in:** gates must map where the model is valid; no lock-in on one
frequency or geometry, and "a port that does not work everywhere" is an acceptable,
statable limit.

Inputs read: `docs/design_notes/20260903_cv26_oblique_defect_diagnosis.md` (#888),
`docs/design_notes/20260903_cv04_envelope_decomposition.md`,
`rfx/sources/tfsf.py`, `rfx/sources/tfsf_2d.py`, `rfx/boundaries/cpml.py`,
`validation/crossval/04_multilayer_fresnel.py`,
`tests/crossval/test_aux_echo_record_invariant.py` (#892).

---

## 0. Why this note exists

The #888 diagnosis proposed the fix as *"re-derive the aux profile from the reflection
target and deepen it: `n = 60`, m=3, kappa=1, `R_asym = 1e-6` gives `|B/A| = 6.98e-05`."*
That number came from the semi-analytic lattice model (`aux_lattice_field`), not from
the shipped code. **It does not reproduce.** Driven through
`rfx.sources.tfsf`/`tfsf_2d` themselves, `n = 60` at `R_asym = 1e-6` leaves
`6.8e-03` on cv04's own band -- seven times over `LEAK_BAR` -- and the reflection
target is not the lever the diagnosis took it for.

The corrected reading is in section 2. It does not change decision A (deepen), only
its size and its cost.

---

## 1. The instruments

Two independent measurements, each driving the SHIPPED code rather than a model of it.

**(a) Two-mode fit, 2-D aux grid.** `init_tfsf_2d` + `update_tfsf_2d` standalone; `Ez`
sampled at 96 x-positions strictly inside the interior (source + 40 to the far absorber
- 40); each position's record FFT'd; at every gated bin a least-squares fit of the two
lattice modes `exp(-j k_x x)` and `exp(+j k_x x)` at the numerical `k_x(f, k_y)`.
`|B/A|` is x-independent in magnitude, so the window choice does not enter.

*Validated:* against the #888 note's own aux measurement -- te_00 `|B/A|` mean
**4.1785e-02**, max **5.4886e-02** against the note's 4.1804e-02 / 5.5047e-02.
Four significant digits, and the note's number came from a different geometry.

*Floor:* a 200-cell `R_asym = 1e-12` absorber reads 9.5e-07 (0 deg), 2.3e-06 (45 deg),
6.9e-06 mean / 7.3e-05 max (60 deg). Every number below is well above its own floor.

**(b) Padded twin, 1-D aux grid.** The aux grid run twice at identical source and probe
geometry, once as shipped and once with its far end pushed 6000 cells out. Over a record
shorter than the padded twin's own echo arrival, the difference between the two probe
records IS the shipped grid's echo. No dispersion model and no mode fit -- and the
normalisation is the one `predict_settling`'s `e_absorber` already uses.

*Validated:* against the SHIPPED code itself. `origin/main` @ `b59e1d99`, driven in a
separate process through the un-parameterised `rfx/sources/tfsf.py`, reads **5.7846e-02**
on cv04's rig and band; the parameterised override that expresses the same profile
through `_cpml_profile` reads **5.7846e-02**. Five significant figures, so the
parameterisation is the shipped absorber and the falsifier below is the real one.

*Not* the cv04 envelope-decomposition note's 4.4023e-02. That note's number is a
different statistic -- its own in-band two-mode fit -- of the same phenomenon at the
same scale. An earlier draft of this note claimed the padded-twin measurement
"reproduced" it; two different instruments landing within 30 percent of each other is
not a reproduction, and the claim is withdrawn.

**Band matters, and the first attempt got it wrong.** The 1-D path's default waveform is
a *differentiated* Gaussian, whose spectrum peaks at 1.75 GHz for `bw = 0.25`, not at
`f0`. A first pass banded 6.2-13.8 GHz -- entirely in the spectral tail -- and produced
ratios that were noise over noise. Every 1-D number in this note is measured on cv04's
own mask: `(f > 3 GHz) & (f < 15 GHz) & (inc_power > 0.02 * max)`, `bw = 0.5`, the 2-D
TMz timestep, 233 bins.

---

## 2. The 1-D path (cv04's): depth-limited, not sigma-limited

`echo / inc` on cv04's own rig and band, by absorber depth and reflection target:

| n_cpml | R_asym = 1e-6 | 1e-8 | 1e-10 |
|---|---|---|---|
| 20 | 4.427e-02 | 4.853e-02 | 5.211e-02 |
| 30 | 2.483e-02 | 2.750e-02 | 2.974e-02 |
| 40 | 1.559e-02 | 1.754e-02 | 1.916e-02 |
| 60 | 6.796e-03 | 7.999e-03 | 9.017e-03 |
| 80 | 2.863e-03 | 3.656e-03 | 4.341e-03 |
| 100 | 1.019e-03 | 1.484e-03 | 1.916e-03 |
| 140 | **9.355e-05** | 1.331e-04 | 2.125e-04 |
| 200 | 9.430e-06 | 1.938e-05 | 3.431e-05 |
| 280 | 3.737e-06 | 4.110e-06 | 6.115e-06 |

The shipped 1-D profile is `sigma_max = 0.8 * 4 / (eta * dx_1d)`, which is
`_cpml_profile`'s law at `R_asym = exp(-32) = 1.3e-14` for `n = 20` -- off the tight end
of this table, and it reads **5.7846e-02**, consistent with the table's own trend (the
tighter the target at fixed depth, the worse). Three readings:

1. **Depth is worth more than any target.** Every row falls monotonically with depth;
   no target at `n = 20` reaches what the loosest target reaches at `n = 40`.
2. **Tightening the reflection target makes it WORSE, at every depth.** `R_asym = 1e-6`
   beats 1e-8 beats 1e-10 in all nine rows. A tighter target raises `sigma_max`, steepens
   the grading, and the absorber reflects off its own discretisation. So the echo is not
   set by an under-absorbing sigma; the diagnosis's "re-derive sigma from the reflection
   target" is necessary bookkeeping (it puts the aux grid on the same law as the 3-D
   absorber) but it is not the fix.
3. **Depth is the lever, and its law is not a single power.** Over n = 20..100 the fall is
   about `n^-2.3`; over n = 100..280 it is about `n^-5.4`. A single fitted power
   (`n^-3.78` over the whole range) would extrapolate wrongly in both halves, so the
   required depth is READ OFF the table, never fitted:

   - `echo <= LEAK_BAR = 1e-3`: **n = 100** (1.019e-03, at the bar).
   - `echo <= LEAK_BAR / 10`: **n = 140** (9.355e-05).

**Declared for the 1-D path:** `AUX_N_CPML_1D = 140`, `AUX_CPML_R_ASYMPTOTIC_1D = 1e-6`,
order 3, kappa 1 (the 1-D update carries no kappa, and an override is now refused rather
than silently ignored). Cost: the 1-D auxiliary array grows from 680 to 920 cells on
cv04's grid -- a 1-D array beside a 2-D/3-D solve, so the cost is not measurable.

---

## 3. The 2-D path (the oblique Bloch path's): an angle wall

`|B/A|` mean / max by (depth, target) at three incidence angles. Each arm carries its own
`bandwidth_for(theta)`, so the 70 deg band is narrow and its realized angles run from
70 deg up toward grazing at the cutoff `f_c = f0 sin theta0`.

| n | R_asym | 0 deg | 45 deg | 70 deg |
|---|---|---|---|---|
| 30 (shipped depth) | 1e-4 | 4.03e-03 / 1.19e-02 | 7.25e-03 / 1.11e-02 | 2.68e-02 / 7.47e-02 |
| 30 | 1e-6 | 6.09e-03 / 1.54e-02 | 1.04e-02 / 1.68e-02 | 1.67e-02 / 2.35e-02 |
| 60 | 1e-6 | 1.29e-04 / 6.47e-04 | 4.13e-04 / 1.99e-03 | 6.06e-03 / 1.89e-02 |
| 60 | 1e-8 | 2.03e-04 / 1.12e-03 | 6.12e-04 / 2.52e-03 | 3.01e-03 / 3.93e-03 |
| 80 | 1e-6 | 2.51e-05 / 1.45e-04 | 7.95e-05 / 5.33e-04 | 8.20e-03 / 2.45e-02 |
| 80 | 1e-10 | 5.25e-05 / 2.20e-04 | 1.33e-04 / 5.94e-04 | 1.50e-03 / 2.76e-03 |
| 100 | 1e-6 | 7.45e-06 / 1.77e-05 | 6.18e-05 / 3.68e-04 | 9.13e-03 / 2.80e-02 |
| 100 | 1e-12 | 1.78e-05 / 1.05e-04 | 4.78e-05 / 1.83e-04 | **5.87e-04 / 1.87e-03** |

(The shipped absorber is n = 30 with `sigma_max = 0.8 (m+1)/(eta dx) * kappa_max`, m = 4,
kappa_max = 7 -- equivalent to `R_asym = exp(-48)`, off the tight end of this table. It
reads 4.18e-02 / 5.49e-02 at 0 deg and 5.13e-02 / 5.35e-02 at 60 deg: flat in angle,
which is what a badly mis-parameterised absorber looks like. Its `sigma_max` is **74.32**
at `dx = 1 mm`; the #888 note's 148.647 is the same expression at the dx/2 rung, where
it was measured.)

**The optimum target inverts with angle.** At 0 and 45 deg the loosest target wins and
the tightest costs a factor of two. At 70 deg the loosest target is a catastrophe
(2.7e-02 at n=30, and it gets *worse* with depth: 4.32e-02 at n=100) while the tightest
is best. No single `(n, R_asym)` minimises the reflection at every angle, so the target
is derived from the WORST DECLARED ANGLE and the cost elsewhere is stated rather than
optimised away: at `n = 100, R_asym = 1e-12` the 0 deg reading is 1.78e-05 instead of the
7.45e-06 a normal-incidence-optimal target would give -- a factor of 2.4, three decades
under `LEAK_BAR`, and not worth a second constant.

Whether the 70 deg corner converges with further depth or is a wall is measured in
section 4.

---

## 4. The 70-degree corner is not a wall

`|B/A|` mean / max at 70 degrees, and the instrument's own floor there:

| n | R = 1e-10 | 1e-12 | 1e-14 |
|---|---|---|---|
| 100 | 7.22e-04 / 2.72e-03 | 5.87e-04 / 1.87e-03 | 6.70e-04 / 1.87e-03 |
| 140 | 5.34e-04 / 2.73e-03 | 1.48e-04 / 9.12e-04 | 9.16e-05 / 4.05e-04 |
| 200 | 6.11e-04 / 2.94e-03 | 1.49e-04 / 8.50e-04 | **3.83e-05 / 2.34e-04** |
| 280 | 6.88e-04 / 3.35e-03 | 1.77e-04 / 1.05e-03 | 4.91e-05 / 3.41e-04 |
| **floor** (n = 600, R = 1e-14) | | | 4.04e-05 / 1.68e-04 |

It converges. At `n = 200, R_asym = 1e-14` the reading is the instrument's own floor, so
the absorber has stopped being the limiting term of the measurement that judges it --
which is where this derivation stops, rather than at a round number.

**The declared 2-D setting, across angle** (`n = 200`, `R_asym = 1e-14`, order 3, kappa 1):

| theta | mean | max | floor at that angle | shipped |
|---|---|---|---|---|
| 0 | 1.13e-06 | 3.10e-06 | 9.5e-07 | 4.18e-02 |
| 30 | 1.84e-06 | 1.16e-05 | | |
| 45 | 2.56e-06 | 1.25e-05 | 2.3e-06 | 4.81e-02 |
| 60 | 5.78e-06 | 4.62e-05 | 3.8e-06 | 5.13e-02 |
| 70 | 3.83e-05 | 2.34e-04 | 4.0e-05 | |

At every angle the declared absorber sits at or within a factor of two of the floor.
`sigma_max` is **0.856 S/m** -- gentler than the 2.445 a 30-cell layer at `R = 1e-6`
would carry and 87 times gentler than the 74.32 that shipped. The fix is not a stronger
absorber; it is a longer, weaker one.

**Where the reflector is, and why that stops being a question.** The phase-slope fit
`B/A = rho e^{-2 j k_x L}` localises the shipped reflector 8.19 cells (0 deg) and 7.86
cells (45 deg) inside the 30-cell layer, reproducing the #888 note's 8.0. On the declared
absorber it returns 0.68 cells at 0 deg and NEGATIVE depths at 45 and 70 -- at `|B/A|` of
1e-06 the residual backward wave is no longer one specular echo from one place. So
`AUX_REFLECTOR_DEPTH_CELLS = 6.88` in `cv22_dispersive_gates.py`, a MEASURED constant of
the 20-cell absorber, cannot be re-measured; it becomes a BOUND of zero -- the absorber's
inner edge, the earliest place a reflection could originate and therefore the earliest it
could arrive. That makes the #892 arrival guard strictly more conservative, which is the
safe direction, and the amplitude statement this lane adds is what it was standing in for.

### 4.1 The gate, and where the GATE is valid

`tests/unit/sources/test_tfsf_aux_absorber_reflection.py` is the reflection gate whose
absence let a 6 percent absorber ship. Its bars are the measured maxima carried through
`tests/_gate_policy.py::gate_from_envelope`, and they are measured **at the rig the gate
itself runs on** -- not borrowed from the derivation above. A bar derived at `nx = 400,
12000 steps` and applied at `nx = 150, 2500 steps` would be six times too tight at 45
degrees; borrowing an envelope measured on one rig to judge another is the exact defect
decision B exists to remove from the slab family, and it is not going to be introduced
here.

The gate declares its own validity domain and asserts it:

| theta | fast rig fit residual | verdict |
|---|---|---|
| 0 | 3.2e-07 | measured, bar 4.6e-06 |
| 30 | 5.7e-04 | measured, bar 1.2e-04 |
| 45 | 6.8e-04 | measured, bar 1.1e-04 |
| 60 | 2.0e-03 | measured, bar 4.9e-04 |
| 70 | **2.4e-01** | NOT-APPLICABLE -- four bins; measured on the full rig under `slow` |

`FIT_RESID_LIMIT = 1e-2` sits two decades over the worst residual the rig resolves and
one decade under the one it does not. At 70 degrees the fast rig reads 2.3e-02 for an
absorber the full rig measures at 3.8e-05: applying a bar to that number would be judging
noise, so the angle is refused rather than passed. Falsifiers: the shipped profile must
fail the same bars at 0 and 45 degrees (it reads 5.5e-02 and 5.4e-02), and -- sharper --
a 30-cell layer carrying the DECLARED target must also fail, so the gate tests the depth
the derivation identified rather than the law it is written in.

---

## 5. What changes, and what has to be re-measured because of it

`n_cpml`, the grading order, kappa and sigma of both auxiliary grids set the injected
field for **every** consumer of the TF/SF path -- cv04, cv16, cv17, cv26, `compute_rcs`
and the oblique waveguide arms. Deepening them re-baselines committed numbers, which is
why this needed decision A rather than an in-lane patch.

The layout follows the depth: `n2x = n_cpml + margin + n_tfsf + margin + n_cpml` and
`i0_x = n_cpml + margin`, so a 100-cell absorber moves `i0_x` from 55 to 125 and the
source from 33 to 103. Three constants in `tests/crossval/test_aux_echo_record_invariant.py`
(#892) are pinned to the old layout (`n_aux = 3092`, `src_idx = 33`, `aux_n_cpml = 30`)
and are re-derived, not relaxed: the echo ARRIVAL is geometry, and a deeper absorber
pushes the reflector further from the probes, which lengthens the admissible record.

---

## 6. Reproduction

Local, `~/Documents/rfx/.venv/bin/python`, worktree `~/Documents/rfx-worktrees/aux-absorber`.
No VESSL run. Wall time: 2-D (depth x target x angle) grid 75 measurements in ~75 min;
1-D depth law 27 measurements in 1132 s.

Scripts (session scratchpad, to be committed as the rig check named in section 7):
`aux_refl.py` (two-mode fit, 2-D), `aux_echo_pad.py` (padded twin, both paths),
`cv04_echo.py` (cv04's own rig and band), `sweep_nr.py`, `sweep_depth_1d.py`, `sweep_70.py`.

---

## 7. cv04 re-run: what moved, and the gate that went red

cv04 re-run on the derived absorber, everything else untouched (719 steps, same
mesh, same slab). `origin/main`'s own numbers, measured in the same session by
pointing `PYTHONPATH` at the un-changed checkout, are the "before" column.

| | before | after | |
|---|---|---|---|
| mean \|dR\| vs analytic | 0.0066 | 0.0070 | |
| mean \|dT\| vs analytic | 0.0110 | 0.0105 | |
| **max \|R+T-1\| per bin** | 0.0487 | **0.0421** | -13.5 % |
| tail scat @ refl | 0.0363 | 0.0328 | |
| tail total @ trans | 0.0514 | 0.0475 | |
| tail window purity | 2.11e-04 | 1.46e-04 | |
| mean \|dR - lattice\| (gated) | 1.682e-03 | **1.417e-03** | -16 % |
| max \|dR - lattice\| (gated) | 5.636e-03 | **4.731e-03** | -16 % |
| mean \|dT - lattice\| (gated) | 6.251e-03 | 6.087e-03 | |
| aux-echo arrival | 1196 | 1176 | ratio 0.601 -> 0.611, still ok |
| **fringe gate** | ok | **FAIL** | |

On the gated band the run moved TOWARD the exact Yee lattice, which is the
strongest available statement: the exact lattice is what a correct discrete
solver on this mesh should produce, and rfx is now 16 percent closer to it in R.
The per-bin closure -- the term cv04's `W_BIN` is derived from -- fell 13.5 %.

**And the fringe gate went red.** Its three extrema:

| fringe | analytic | Yee lattice | before | after |
|---|---|---|---|---|
| 1 (max) | 3.7475 | 3.7440 | 3.7175 (-26.5 vs lattice) | 3.7287 (**-15.3**) |
| 2 (min) | 7.4950 | 7.4678 | 7.4706 (+2.8) | 7.4722 (+4.4) |
| 3 (max) | 11.2425 | 11.1509 | 11.3084 (+157.5) | 11.5015 (**+350.6**) |

(The Yee-lattice column solves `k_num d = (2m+1) pi/2` with
`sin(k dx/2)/dx = sqrt(eps) sin(w dt/2)/(c dt)`: the lattice moves every fringe
DOWN, by 3.4, 27.0 and 91.3 MHz. The gate's own window is built to absorb
exactly that, `W(f) = 2 (df_bin/2 + |lattice shift|)`.)

Fringes 1 and 2 moved toward the lattice truth. Fringe 3 moved away from it, and
from the analytic reference, by 193 MHz.

**It is not a band-edge artifact.** The obvious suspicion -- fringe 3 sits near
the top of the 2 percent incident-power band, whose edge moved 11.87 -> 11.81 GHz
when the injection got cleaner -- was tested and is wrong. Re-running the SHIPPED
fringe gate on the same run under a 1 percent mask (band top 12.60 GHz, 184 bins),
a 5 percent mask, and upper limits of 12, 12.5 and 13 GHz moves fringe 3 by
exactly zero in both the before and after runs. The extremum is stable in the
band; it is the physics of the measurement there that changed.

### 7.1 Where cv04 gates, and where cv04's fringe gate looks

`gated_mask` -- the case's own policy for which bins carry enough incident
amplitude to be judged, at 10 percent of peak -- admits **4.012 to 9.998 GHz**,
459 of 918 bins. The fringe gate runs on `freqs[mask]`, the 2 percent band,
3.03 to 11.87 GHz.

| fringe | inside the GATED band? |
|---|---|
| 1 (3.7475 GHz) | **no** |
| 2 (7.4950 GHz) | yes |
| 3 (11.2425 GHz) | **no** |

Two of the fringe gate's three extrema sit where the case's own gating policy
says the incident is too weak to judge -- between 2 and 10 percent of peak. The
exact-lattice witness never looks at either of them, which is why it can report
16 percent closer while fringe 3 moves 193 MHz further away: the two
instruments are not looking at the same band.

That is the same defect shape as decision B, in a different case: a window (here,
a whole gate) applied outside the domain where the quantity it judges is
measurable. It is NOT resolved by widening `W(f)` -- the working agreement
forbids that and it would be wrong anyway, since the window already carries the
lattice shift it was built for.

**Three resolutions, and this is a PI decision because it changes what cv04
gates:**

1. **Restrict the fringe gate to the gated band.** Only fringe 2 survives; the
   gate drops from three extrema to one, and its three audited criterion-(B)
   falsifiers must be re-measured for detection power against the reduced gate
   before it is accepted (the working agreement: a replacement gate must be
   shown to kill the same mutants, measured).
2. **Widen the rig, not the window.** Raise cv04's source bandwidth so 11.24 GHz
   carries at least 10 percent incident amplitude, putting fringe 3 inside the
   gated band and making it measurable. Physically the honest fix -- measure
   where you can measure -- but it re-baselines every cv04 number, including the
   envelope decision B is about to re-derive from.
3. **Declare fringes 1 and 3 REPORTED, not gated**, with the incident amplitude
   at each printed beside it, exactly as this case already does for its lattice
   witness (`gated_here = false`).

Until that is decided the fringe gate is left RED rather than adjusted. Nothing
in this lane widens it.

---

## 8. cv22 and cv23 re-run: unmoved, and one falsifier that went silent

Both cases re-run on the derived absorber, everything else untouched. **Both pass
every gate**, E2 (TMM) and E4 (Meep), on every arm and every observable.

The numbers barely move -- fourth or fifth significant figure:

| case / arm | `mean_dR_gated` before -> after |
|---|---|
| cv22 debye | 2.2918e-03 -> 2.2862e-03 |
| cv22 lorentz | 2.8348e-03 -> 2.8332e-03 |
| cv22 drude | 4.9471e-04 -> 4.9356e-04 |
| cv23 tand0p1 | 3.9189e-03 -> 3.9188e-03 |
| cv23 tand1 | 5.0940e-03 -> 5.0956e-03 |
| cv23 tand3 | 3.1152e-03 -> 3.1134e-03 |

That is the cv04 envelope-decomposition note's claim confirmed from the other
side: on the settled slab rungs the auxiliary echo contributes nothing
measurable, because those rungs run at 0.50 to 0.60 of their own echo arrival.
cv04 moved (section 7) and these do not, and the difference is the record.

### 8.1 The exception, and it is not a widening

`tests/crossval/test_lattice_witness_gates.py` pre-declares which falsifiers must
fire at which rung. One flipped: **cv23 `tand3`, `eps_continuum`, declared to
fire, now silent.**

It fired through `GL1_T` alone. `n_bins_R_over_window` is 0 both before and
after, and `separation_over_window_R` is 0.102 before and 0.194 after -- it never
had R or A detection on this arm, and its R separation actually IMPROVED.

The budget says why. Eleven terms, nine of them better:

| term | before | after | ratio |
|---|---|---|---|
| scat_tail_rel | 8.560e-05 | 4.886e-05 | 0.57 |
| purity_rel | 1.109e-04 | 2.721e-05 | 0.25 |
| mean_delta_scat_gated | 3.575e-04 | 1.936e-04 | 0.54 |
| mean_delta_inc_gated | 3.536e-05 | 7.760e-06 | 0.22 |
| **trans_tail_rel** | 6.411e-06 | 2.631e-05 | **4.10** |
| **mean_delta_trans_gated** | 2.678e-05 | 1.043e-04 | **3.90** |

`tand3` is tan delta = 3: `T ~ 3e-05`, and its transmitted tail sits at the
float32 floor. The two terms that got worse are the two that measure that floor,
and they loosened `W_witness_T` by 3.6x (3.634e-06 -> 1.299e-05) -- while
`|rfx - lattice|` on T got BETTER (2.148e-07 -> 1.243e-07). A window derived from
a noise floor moved with the noise.

**Detection power on this rung went UP, measured**, which is what the working
agreement asks to be shown before a changed gate is accepted:

| falsifier | separation / window (R), before -> after |
|---|---|
| continuum | 6.61 -> 12.52 |
| thickness_plus_cell | 0.94 -> 1.79 |
| thickness_minus_cell | 1.10 -> 2.09 |
| eps_x1p01 | 0.52 -> 0.99 |
| eps_continuum | 0.10 -> 0.19 (still under 1: no R detection either way) |

and `eps_continuum` still fires on `tand1` (0.147 -> 0.252).

So the expectation is updated to what is MEASURED, with the reason in the table
itself: cv23's most lossy arm cannot referee this defect, because the only
channel it was refereed through -- a transmitted signal of 3e-05 -- is not
measurable there. No window was widened and no falsifier was removed; one
pre-declared verdict is now recorded as false, and this section is the record.

---

## 9. cv04's rig: the band was never illuminated (PI decision, 2026-09-04)

Section 7 left the fringe gate red and put three resolutions to the PI. The PI
chose **widen the rig**. Measured, at cv04's own rig with only `bw` changed:

| bw | fringe 3 (GHz) | dev vs analytic | max \|R+T-1\| | T max err | R+T mean |
|---|---|---|---|---|---|
| **0.5** (shipped) | 11.5015 | **+259.0 FAIL** | 0.0421 | 0.0513 | 1.0026 |
| 0.7 | 11.2038 | -38.7 ok | 0.0099 | 0.0200 | 0.9997 |
| 0.9 | 11.1838 | -58.7 ok | 0.0035 | 0.0194 | 0.9999 |
| 1.1 | 11.1818 | -60.7 ok | 0.0015 | 0.0189 | 1.0000 |

Three things at once:

1. **Fringe 3 stabilises and converges on the LATTICE answer.** The Yee-lattice
   prediction is 11.1509 GHz. From bw 0.9 to 1.1 the measurement moves **2.0 MHz**,
   against the 193 MHz it swung at bw = 0.5 when the injection changed. A quantity
   that moves 193 MHz under a change of the injection and 2 MHz under a doubling of
   the illumination was not being measured at bw = 0.5; it is at bw >= 0.9.
2. **The closure envelope collapses by a factor of 28**: `max|R+T-1|` 0.0421 ->
   0.0015. cv04's committed `per_bin_max_RT_closure = 0.0487` -- the number
   `W_BIN = gate_from_envelope(0.0487) = 0.074` is derived from, for the WHOLE
   slab family -- is very largely an artifact of a source that never illuminated
   the top of its own analysis band.
3. **The mean errors barely move**: `mean|dR|` 0.0070 -> 0.0080, `mean|dT|`
   0.0105 -> 0.0080. So `W_MEAN_R`'s envelope really is the lattice term the cv04
   decomposition note said it was, while `W_BIN`'s envelope was illumination.

That separation is load-bearing for **decision B**, which is about to re-derive
this family's windows: the mean-window and the bin-window envelopes have
different causes, and only one of them is discretisation.

**Not yet done, and it is the next thing.** Raising `bw` also raises the analysis
band's top: at bw >= 0.7 the 2 percent incident-power floor no longer cuts the
band below 15 GHz, and 15 GHz on this mesh is 5 cells per wavelength inside the
slab. Gating there would be judging content the mesh does not resolve. So the rig
change is `bw` PLUS an analysis-band ceiling derived from a declared
cells-per-wavelength criterion -- not left at the 15 GHz literal. Until that
derivation exists, `bw` is unchanged in the code and this section is the
measurement that motivates it.

Note also what "widen the rig" does NOT mean here. `BAND_GATED_HZ = (4.0e9, 10.0e9)`
in `cv22_dispersive_gates.py` is a declared constant of the WHOLE slab family, not
an amplitude criterion, so moving it to admit 11.24 GHz would move cv22 and cv23
too, and would gate them at 13.3 cells per wavelength in the slab. cv04's fringe
gate runs on its own band and needs no such change; the family constant stays.

---

## 10. The cv04 rig, derived

Section 9 measured that widening the source fixes fringe 3 and left two things
undone: which bandwidth, and what caps the analysis band. Both are derived here,
and the first hypothesis for the mechanism turned out to be wrong.

### 10.1 The band ceiling: where the REFERENCE stops being meetable

cv04 gates rfx against the analytic **continuum** slab. That reference is usable
only while a correct **lattice** solver could still meet it, so the ceiling is a
property of the reference, not of rfx.

Two channels, and only one of them can give a ceiling:

* **Position.** `W(f) = SAFETY (df_bin/2 + |Yee dispersion shift at f|)` absorbs
  the lattice shift *by construction*. A perfect lattice solver never breaches
  it at any frequency, so this channel is self-referential and yields no ceiling.
* **Value.** `value_limit = 0.04` is a FIXED tolerance. So the ceiling is the
  frequency at which `|R_lattice - R_continuum|` reaches it:

  ```
  |R_lattice - R_continuum| < 0.04 for every f below   15.587 GHz
  ```

  (`de.yee_lattice_slab_rt_eps` against `de.tmm_slab_rt`, cv04's own dx and dt.)

cv04's shipped analysis-band literal is **15.0 GHz**, which sits inside that
ceiling with 587 MHz to spare, so **no new ceiling constant is needed** -- the
literal is now justified rather than assumed. The band-mean channel never binds:
`mean|R_lattice - R_continuum|` over 3-15 GHz is 0.0080 against a 0.05 limit.

**Correction to section 9.** That section said 15 GHz is "5 cells per wavelength
inside the slab" and used it to argue a new ceiling was needed. That was an
arithmetic slip: `lambda_slab = c/(f n)` = 9.99 mm at 15 GHz, i.e. **10.0 cells**,
not 5. The mesh's own Nyquist in the slab is 74.95 GHz. The concern that motivated
a new constant does not exist.

### 10.2 The mechanism: illumination, not cell truncation

The first hypothesis was that fringe 3's half-fringe **search cell** (9.369 to
13.116 GHz) was truncated by the band top (11.81 GHz at bw = 0.5, 66.7 percent
covered), and that containment was the criterion. It is not:

* at `bw = 0.55` the band reaches 13.017 GHz -- the cell is **94 percent**
  covered -- and the gate still fails by **+245.8 MHz**;
* from `bw = 0.65` to `1.10` the realized band top is **identical** (14.951 GHz)
  and the measured position still moves **81 MHz**, monotonically, as the
  illumination rises.

The band sets what can be searched. The incident power AT the extremum sets
whether the answer means anything. Measured on cv04's own rig, source bandwidth
swept and nothing else changed:

| bw | inc power @ 11.2425 GHz | f3 measured (GHz) | error vs converged | verdict |
|---|---|---|---|---|
| 0.50 | 0.0336 | 11.5015 | +319.7 MHz | FAIL |
| 0.55 | 0.0735 | 11.4883 | +306.5 MHz | FAIL |
| 0.60 | 0.1313 | 11.3690 | +187.2 MHz | ok |
| 0.65 | 0.2037 | 11.2627 | +80.9 MHz | ok |
| 0.70 | 0.2855 | 11.2038 | +22.0 MHz | ok |
| 0.80 | 0.4571 | 11.1845 | +2.7 MHz | ok |
| 0.90 | 0.6146 | 11.1838 | +2.0 MHz | ok |
| 1.10 | 0.8414 | 11.1818 | 0 (reference) | ok |

The error crosses the measurement's own resolution, `df_bin/2 = 26.14 MHz`,
between 0.2037 and 0.2855 -- an incident power of **0.277**, fourteen times the
2 percent mask floor that admits the bin to the band at all.

`FRINGE_INC_POWER_MIN = 0.42` is that measured threshold carried through the
shared gate policy's multiplier (`tests/_gate_policy.py`,
`ENVELOPE_GATE_MULTIPLIER = 1.5`): `0.277 * 1.5 = 0.416`, quantised up.

### 10.3 The bandwidth

`bw = 0.8`: the smallest measured value at which EVERY gated extremum clears the
bar -- 0.8761, 0.9093, 0.4571 against 0.42. (`bw = 0.7` leaves fringe 3 at
0.2855, under it.) The realized band top is then 14.951 GHz, inside the 15.587
GHz ceiling of section 10.1. No safety factor is applied twice: the 1.5 is
already inside the bar.

### 10.4 What the gate now does, and the hole that had to be closed

`compare_fringes` takes `inc_power_rel` and **refuses** an extremum below the
bar: it is reported N/A, never judged and never silently dropped.

That opens a hole, and closing it is the point. Refusing an extremum makes the
GATE smaller, and a smaller gate must not read as a greener case. So cv04
declares which extrema its rig is supposed to measure (`CV04_DECLARED_EXTREMA`)
and **fails** when one goes N/A. Run on the rig that shipped:

```
     max      3.7475       3.7287     -18.8     59.0   0.3600   0.3599  -0.0001        ok
     min      7.4950           --        --       --       --       --       --       N/A   incident power 0.3704 < 0.42
     max     11.2425           --        --       --       --       --       --       N/A   incident power 0.0336 < 0.42
  !! fringe ADMISSION -- this rig no longer measures the min at 7.4950 GHz, the max at 11.2425 GHz.
  rfx accuracy: FAIL
```

The shipped rig is now rejected for being unable to measure its own gate,
instead of reading the third fringe 320 MHz off and passing.

### 10.5 cv04 on the derived rig

| | shipped (bw 0.5, 20-cell absorber) | derived (bw 0.8, 200-cell absorber) |
|---|---|---|
| `max\|R+T-1\|` per bin | 0.0487 | **0.0043** |
| `mean\|dR\|` vs analytic | 0.0066 | 0.0080 |
| `mean\|dT\|` vs analytic | 0.0110 | 0.0081 |
| `mean\|dR - lattice\|` gated | 1.682e-03 | **1.98e-04** |
| tail window purity | 2.11e-04 | 7.67e-05 |
| fringe 1 / 2 / 3 | ok / ok / (+65.9, judged unlit) | ok / ok / **ok** (-58.0) |
| verdict | PASS (on an unmeasurable fringe) | **PASS** |

`mean|dR - lattice|` is 8.5 times smaller than it was before this lane started,
and the per-bin closure envelope is 11 times smaller.

### 10.6 What this hands decision B

`CV04_ENVELOPE`, which the slab family's windows are derived from, was measured
on a rig that did not illuminate the top of its own band:

| envelope term | committed | on the derived rig | window it feeds |
|---|---|---|---|
| `per_bin_max_RT_closure` | 0.0487 | **0.0043** | `W_BIN = 0.074` |
| `mean_dR` | 0.0066 | 0.0080 | `W_MEAN_R = 0.010` |
| `mean_dT` | 0.011 | 0.0081 | `W_MEAN_T = 0.017` |

The two envelopes have different causes and only one of them survives the rig
fix. `mean_dR` is the lattice term the cv04 decomposition note said it was --
it does not fall, it rises slightly. `per_bin_max_RT_closure` falls by 11x,
so `W_BIN` is roughly an order of magnitude looser than the rig it is derived
from now warrants. Decision B's re-derivation should take its envelope from the
derived rig, not from the committed one.
