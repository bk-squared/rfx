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

---

## 11. Two gates this lane made possible and did not turn on

Regenerating cv04, cv22 and cv23 moved 73 numbers that four documents cite, and
the #829 provenance gate caught every one. They were re-stated from the
artifacts by the gate's own collector, resolver and tolerance
(`tests/contracts/test_evidence_numeric_provenance.py`), not retyped -- 73
citations across `validation/crossval/manifest.json`, `validation/README.md`,
`docs/design_notes/20260903_lattice_witness_standard.md` and
`docs/design_notes/20260904_aux_echo_record_invariant.md`.

**Re-stating a number is not the same as re-stating the argument built on it**,
and two arguments in the tree are now false. Both are corrected in place; neither
gate is turned on here, because turning one on changes what a case gates.

### 11.1 cv04's lattice witness has become a gate on R

`docs/design_notes/20260903_lattice_witness_standard.md` §5.3 declared cv04's
exact-lattice witness REPORTED, not gated, and gave the reason: *"a gate that
cannot reject the continuum model is not a gate"*. On the rig it was written
for, the continuum falsifier separated 0.099 of the window on 0 of 115 bins.

Same case, same rung, same 719 steps, after the derived absorber and `bw = 0.8`:

| | then | now |
|---|---|---|
| tails, scat / trans | 0.036 / 0.051 | 0.0042 / 0.0121 |
| `W_witness,R` | 5.35e-02 | 1.98e-03 |
| exceeds its own ceiling | yes | **no** (2.4x inside) |
| vs cv04's own `W_MEAN_R` = 0.010 | looser | **5x tighter** |
| `continuum` falsifier, sep / window R | 0.099 (0 bins) | **2.68 (65 of 115)** |
| `eps_x1p01`, sep / window R | 0.082 (0 bins) | **2.20 (86 of 115)** |
| `thickness_plus / minus_cell` | -- | 38.99 / 39.08 (115 / 114) |

The witness now rejects the continuum model by 2.7x its own window. **T did not
follow**: `W_witness,T` = 1.39e-02 still exceeds its 1.17e-02 ceiling and the
continuum falsifier separates only 0.38 there, so T stays reported.

**TURNED ON for R** (PI decision, 2026-09-04), T and A stay reported.
`04_multilayer_fresnel.py` writes `gated_channels = ["R"]` and exits non-zero
when the R channel fails; `tests/crossval/test_cv04_fringe_measurability.py`
replays it from the committed artifact so it bites in CI, not only under the
`--lattice-witness` flag.

Teeth, measured rather than argued: a one-cell slab thickness error (10 mm ->
11 mm, everything else untouched) drives `|rfx - lattice|` on R from 1.98e-04 to
**7.72e-02** and fails both `GL1_R` and `GL2_R`.

**What the improvement actually is, stated plainly**, because the honest split
matters more than the headline. The tail drop that tightened `W_witness` is
mostly the RIG, not the solver: the absorber fix alone moved the tails
0.036 -> 0.033, and the bandwidth 0.5 -> 0.8 moved them 0.033 -> 0.0042. A
shorter pulse settles further inside the same 719-step record. No window
formula, limit or falsifier definition changed -- every constant is the one that
shipped, and the derived window got TIGHTER, which is the opposite direction
from a widening. The independent check that something real improved is
`|rfx - lattice|`, which carries no window at all and fell 1.42e-03 -> 1.98e-04.
Against that, `mean|dR|` versus the continuum went slightly the other way,
0.0066 -> 0.0080. It is a better measurement, not a uniformly better answer.

### 11.2 The slab family's `W_BIN` is derived from an envelope that no longer exists

Section 10.6: `per_bin_max_RT_closure` 0.0487 -> 0.0043 while `mean_dR` 0.0066 ->
0.0080. `W_BIN = gate_from_envelope(0.0487) = 0.074` is therefore about an order
of magnitude looser than the rig it is derived from now warrants, and
`W_MEAN_R = 0.010` is not -- the two envelopes have different causes and only one
of them was illumination. This is decision B's input and belongs to decision B's
lane, not this one.

### 11.3 Not a gate, but recorded

`cv23 tand3`'s `eps_continuum` falsifier is silent where it was declared to fire
(§8.1), and cv04's `eps_continuum` separation is identically zero -- structurally,
since cv04 is lossless and that defect IS the continuum permittivity there. Both
are recorded as measured verdicts rather than repaired.

---

## 12. Correction: the worst declared angle is 82 degrees, not 70 (2026-09-05, review)

Section 3 derived the 2-D target at "the worst declared angle, 70 degrees, cv26's gate
cap", and section 4 said the result was "at the instrument floor at every angle". Both
statements were true only for the angles measured, and the angles measured stopped at 70.
`THETA_GATE_MAX_DEG = 70` is cv26's **primary**-rig cap. Three of cv26's ten arms
(`graze_vac`, `graze_pec`, `graze_te`) are declared at `GRAZE_THETA0_DEG = 82` with
`GRAZE_THETA_GATE_DEG = (80, 85)`. The derivation never looked there. An independent
reviewer did, and this section is the re-measurement.

### 12.1 What the 1e-14 absorber does above 70 degrees

Same instrument (`measure_aux_reflection_2d`), full rig, 40000 steps so the grazing band
carries bins; shipped 30-cell absorber alongside:

| theta | declared (n=200, R=1e-14) mean / max | shipped 30-cell mean / max | gain (max) |
|---|---|---|---|
| 70 | 3.769e-05 / 2.297e-04 | 5.263e-02 / 5.300e-02 | x231 |
| 76 | 5.135e-04 / 2.749e-03 | 5.263e-02 / 5.277e-02 | x19 |
| 82 | 9.694e-03 / 2.793e-02 | 5.151e-02 / 5.212e-02 | **x1.9** |

At 82 degrees the fix buys a factor of 1.9. The defect #888 exists to remove is still
present on the arms that run nearest grazing, and no gate in cv26 can see it because the
grazing gates (G6, G7) carry the same echo on both sides of their difference.

### 12.2 kappa is not the lever; sigma is

The textbook remedy for grazing incidence is real stretching, kappa > 1. Measured at 82
degrees, n = 200, two series, so the two effects the law couples are pulled apart
(`_cpml_profile` multiplies sigma_max by kappa_max):

| kappa | R_asym | sigma_max (S/m) | mean / max |
|---|---|---|---|
| 1 | 1e-14 | 0.856 | 9.694e-03 / 2.793e-02 |
| 2 | 1e-14 | 1.711 | 3.915e-04 / 1.288e-03 |
| 4 | 1e-14 | 3.423 | 8.216e-04 / 1.391e-03 |
| 7 | 1e-14 | 5.990 | 1.372e-03 / 2.042e-03 |
| 12 | 1e-14 | 10.268 | 2.076e-03 / 2.811e-03 |

and with sigma_max **held** at 0.856 (R_asym loosened as 1e-14^(1/kappa)):

| kappa | R_asym | sigma_max | mean / max |
|---|---|---|---|
| 2 | 1e-07 | 0.856 | 1.091e-02 / 3.152e-02 |
| 4 | 3.16e-04 | 0.856 | 1.182e-02 / 3.479e-02 |
| 7 | 1e-02 | 0.856 | 1.211e-02 / 3.493e-02 |
| 12 | 6.81e-02 | 0.856 | 1.045e-02 / 2.383e-02 |

With sigma held, kappa from 1 to 12 changes nothing. The whole gain in the first table is
sigma_max going 0.856 -> 1.711. Then sigma alone, kappa = 1:

| R_asym | sigma_max | mean / max at 82 deg |
|---|---|---|
| 1e-21 | 1.284 | 1.132e-03 / 3.718e-03 |
| **1e-28** | **1.711** | **5.760e-04 / 1.228e-03** |
| 1e-35 | 2.139 | 6.995e-04 / 1.326e-03 |
| 1e-42 | 2.567 | 8.562e-04 / 1.512e-03 |

The optimum at 82 degrees is sigma_max near 1.7, and it reproduces the kappa = 2 row
within noise. Physically: at grazing incidence the wave barely penetrates the layer, the
absorption per unit depth scales with cos(theta), and the way to recover it is more loss
per unit depth -- not a longer electrical path, which is what kappa buys and which only
propagating-below-cutoff (evanescent) content needs. Too much sigma and the taper reflects
off its own gradient again, which is the 1e-35 and 1e-42 rows.

### 12.3 What sigma_max = 1.711 costs below 70 degrees

| theta | R=1e-14 mean / max | R=1e-28 mean / max |
|---|---|---|
| 0 | 1.130e-06 / 3.098e-06 | 2.285e-06 / 6.844e-06 |
| 45 | 2.561e-06 / 1.254e-05 | 4.365e-06 / 1.758e-05 |
| 70 | 3.832e-05 / 2.336e-04 | **2.920e-05 / 1.219e-04** |

A factor of two at 0 and 45 degrees, three decades under any bar; an improvement at 70.
So `AUX_CPML_R_ASYMPTOTIC` is re-derived to **1e-28** (sigma_max = 1.711 S/m). The 1-D
path is unchanged: it is normal incidence only, and section 2's derivation at 0 degrees
stands.

### 12.4 What it does NOT fix, stated as a domain

At the optimum, 82 degrees reads max 1.228e-03 -- **above LEAK_BAR = 1e-03**. A
polynomial-graded CPML does not get grazing incidence under that bar at this depth, and
the sweep above says pushing sigma further makes it worse. So the absorber's validity
domain is declared, not tuned: it meets LEAK_BAR up to an angle measured in 12.5 below,
and cv26's grazing arms sit outside it. Those arms already model the echo instead of
gating it out (their compact box cannot time-gate at all -- section 13); the residual
they must answer for is the 1.2e-03 measured here, not the 2.8e-02 they were carrying.

### 12.5 The validity domain, measured

`n = 200`, `R_asym = 1e-28`, kappa 1, order 3; full rig at 40000 steps so the grazing bands
carry bins; every fit residual under `FIT_RESID_LIMIT`:

| theta | bins | fit residual | mean / max | vs LEAK_BAR = 1e-3 |
|---|---|---|---|---|
| 70 | 65 | 5.2e-04 | 2.829e-05 / 1.246e-04 | under, x8.0 |
| 74 | 41 | 4.1e-04 | 5.467e-05 / 1.630e-04 | under, x6.1 |
| 76 | 32 | 4.6e-04 | 8.257e-05 / 2.239e-04 | under, x4.5 |
| 78 | 23 | 5.1e-04 | 1.438e-04 / 4.668e-04 | under, x2.1 |
| **80** | 17 | 4.7e-04 | 2.875e-04 / **9.063e-04** | under, **x1.10** |
| **82** | 11 | 2.9e-04 | 5.760e-04 / **1.228e-03** | **over**, x0.81 |

**Declared:** the 2-D auxiliary absorber meets `LEAK_BAR` on the gated band for
incidence angles **up to 80 degrees**, with 10 percent in hand at 80, and does not at 82.
cv26's grazing arms (theta0 = 82, gate 80-85) are **outside** the absorber's validity
domain and must keep answering for the residual echo through their model terms
(`e_absorber`, G6, G7) -- now a 1.2e-03 term, not the 2.8e-02 they carried before this
section. The domain is gated, not just stated:
`tests/unit/sources/test_tfsf_aux_absorber_reflection.py::test_the_domain_edge_is_where_the_note_says`
measures 70, 80 and 82 on this rig and asserts the inside rows under the bar, the 82 row
OVER it (so the domain claim is checked in both directions), and each within 10 percent
of the numbers above.

The 1-D path (`AUX_CPML_R_ASYMPTOTIC_1D = 1e-6`) is untouched: normal incidence only.

---

## 13. What the clean injection did to the RCS chain (2026-09-06)

CI caught what the lane's own suites did not: `tests/oracle/test_rcs_mie_fixture.py`
drifted 0.57 dB. RCS is a TF/SF consumer, and this is the third absorber of the same
class in this lane.

### 13.1 The old agreement with Mie was a cancellation

The committed fixture read **0.063 dB** from the exact Mie series. On the clean injection
the same rig reads **0.510 dB**. The direction of the depth ladder settles which one is
the accident -- `|monostatic - Mie|` against the rig's own CPML depth:

| main CPML | 8 | 16 | 24 | 32 | 40 |
|---|---|---|---|---|---|
| pre-#888 auxiliary (20 cells) | **0.097** | 0.191 | 0.231 | -- | -- |
| shipped auxiliary (200 cells) | 0.510 | 0.224 | **0.185** | 0.196 | 0.211 |

On the old injection the answer walks AWAY from Mie as its own absorber improves; on the
clean one it converges. Two errors were cancelling, and the fixture recorded the
cancellation as agreement.

### 13.2 The fixture rig's depth, derived

Mie is a continuum reference, so "closer to Mie" also rewards mesh error. The depth is
read off convergence of the ANSWER:

| cpml | 8 | 16 | 24 | 32 | 40 |
|---|---|---|---|---|---|
| monostatic (dBsm) | -24.8826 | -25.1685 | -25.2076 | -25.1971 | -25.1815 |
| step from the previous rung | -- | 0.286 | **0.039** | 0.011 | 0.016 |

Past 24 the value only wanders inside a ~0.02 dB floor. The same ladder at
`R_asymptotic = 1e-8` lands within 0.002 dB of it at 24, so depth is the lever here too
and the target stays at the repo default. **CPML_LAYERS 8 -> 24**, converged monostatic
-25.21 dBsm, 0.185 dB from Mie against a 1.0 dB physics gate. The sibling
`rcs280_reference_subtraction` fixture moved with it: its claim is that the uncorrected
path equals this fixture's monostatic on the SAME geometry, which only holds while the
two rigs match.

### 13.3 One thing this does NOT explain, and it is not this lane's to fix

`rcs280`'s corrected bistatic pattern agrees WORSE with Mie on the clean injection, and
the cause is the injection, not the depth -- both converge, to different values:

| aux depth | cpml 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| 200 (shipped) | 0.886 | 0.719 | **0.705** | 0.714 |
| 20 (pre-#888) | 0.481 | 0.404 | **0.408** | 0.412 |

mean `|corrected - Mie|` in dB over the 37-point H-plane cut. Everything else improves or
holds: correlation 0.977 (bar 0.95), backscatter delta -0.22 -> +0.185 (better in
magnitude), and the forward-oblique lobe the correction exists to remove still goes
10.66 -> 1.46 dB. Only the pattern MEAN moves, from 0.41 to 0.71, past the 0.6 bar that
was set at "measured ~0.42".

That says `subtract_incident_reference` was benefiting from a contaminated reference
field -- the subtraction cancels more of the pattern error when the incident carries the
4.4 % echo. It is a property of the #280 correction, measured here, not a regression this
lane introduced in it.

**PI decision, 2026-09-06: re-derive the bar from the clean-injection measurement.**
`gate_from_envelope(0.705, quantum=100) = 1.06`, through the shared policy, with the
measurement pinned beside it so a further degradation is visible. A bar that moved
because the measurement got worse must be shown to still kill what it killed, so
`test_the_pattern_bar_still_rejects_the_uncorrected_path` asserts the defect the #280
subtraction exists to remove still fails it: the uncorrected far field reads **3.10 dB**
mean, 2.9x the new bar, with correlation **-0.09** against Mie -- the shape is gone, not
merely offset. The other two assertions in that test are unchanged and both improved or
held (correlation 0.977 against a 0.95 bar; backscatter +0.185 dB against 0.5).

What is NOT closed by this: why a cleaner incident field makes the subtraction worse.
The measurement above is the whole of what this lane knows. It belongs to #280, not to
#888, and it wants its own lane.
