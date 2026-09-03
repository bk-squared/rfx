# cv26 oblique-Fresnel defect diagnosis (issue #888)

Branch `agent/issue-888-oblique-diagnosis` (off `origin/main` @ `8abdce2d`).
Lane code read from `origin/agent/gap4-oblique-fresnel` @ `93fd651`
(`validation/crossval/26_oblique_slab_fresnel.py`,
`validation/crossval/comparators/oblique_fresnel.py`), which is the commit the
round-2 artifacts carry
(`_26_oblique_results/rfx.json::commit = 93fd651d4faaf93137f76b83fe08672a69f07c93`).

Artifact root, abbreviated `R2/` below:
`~/mnt/remilab-fs/personal-workspaces/claude-workspace/rfx/runs/cv26-oblique-r2-20260903T070128Z/`.

**Nothing in this note changes a gate, a window, a declared record or a
committed number.** The one code experiment that perturbs rfx (section 6) was
run through a runtime monkey-patch in a scratch script; the repository is
untouched apart from this file.

---

## 0. Verdict up front

The defect is **not** angle-dependent and it is **not** in the Bloch phase, the
transverse operator, the TF/SF corrections or the probe extraction.

**The incident field that the Bloch TFSF injects is not a plane wave.** The 2-D
auxiliary grid in `rfx/sources/tfsf_2d.py` is terminated by its own CFS-CPML
(`n_cpml_2d = 30`, `cpml_order = 4`, `kappa_max = 7.0`,
`sigma_max = 0.8·(m+1)/(η·dx)·kappa_max`), and that absorber reflects **4-6 % in
amplitude** back into the grid as a −x wave. Measured in the rfx FDTD itself:

| θ₀ | measured \|B/A\| on the aux grid (steady state) |
|---|---|
| 0° | 4.18e-02 (mean) / 5.50e-02 (max) |
| 30° | 5.96e-02 |
| 45° | 5.95e-02 |
| 60° | 5.85e-02 |

The case defines `R = |E_tot − E_inc|²/|E_inc|²` and `T = |E_tot|²/|E_inc|²`
with `E_inc` read **from that same aux grid**
(`26_oblique_slab_fresnel.py:151`, the `aux.ez_2d[ref_r]`/`aux.ez_2d[ref_t]`
samples). The contamination therefore cancels *identically* in vacuum — which is
why discriminator 1 comes back at 1e-10 and why the leakage witness
(`LEAK_BAR = 1e-3`) has never seen it — but not with a slab: the −x component is
injected into the total-field region, strikes the slab from the transmitted
side, and the normalisation divides by an `|E_inc|` that is standing-wave
modulated by ±6 %.

**Why 45° and 60° and not 0° and 30°: the record, not the angle.** The echo
needs `2·L/v_gx` to travel from the aux hi-CPML back to the probes, and
`v_gx = c·cos θ` on the lattice. Round 2's record law (amplitude-gated, not
arrival-gated) lengthened the oblique records past that arrival time:

| arm | record (steps) | echo reaches the refl probe at | inside the record? |
|---|---|---|---|
| te_00 | 1512 | 3384 | no |
| te_30 | 6387 | 7643 | no |
| te_45 | 18083 (19 extensions) | 9358 | **yes** |
| te_60 | 27238 (4 extensions) | 13225 | **yes** |

Records from `R2/_26_oblique_results/rfx.json::arms.<arm>.run.record.n_steps`;
arrival = (aux source → measured reflector at index `n2x−22` → reflection probe)
at the lattice group velocity `yee_vgx(f0)` — 0.997 c, 0.866 c, 0.707 c, 0.500 c
respectively. Every passing arm's record ends before its arrival; every failing
arm's record runs 1.9-2.1× past it.

The onset is a **step**, not a smooth growth (section 5), and it sits exactly at
the record/arrival crossing.

A second, smaller term of the same kind rides on top: the **3-D grid's own
CPML** at the declared depth contributes mean \|ΔR\| = 0.0249 and closure 0.0640
at te_45 (section 7). Together they account for the whole 0.0950 / 0.3082.

---

## 1. Discriminator 1 — vacuum at 45° (excludes injection and extraction)

Ran the te_45 configuration with the slab removed (`spec["slab"] = False`,
`eps_slab_rfx = mu_slab_rfx = 1.0`), same rig, same probes, same record law,
through `26_oblique_slab_fresnel.py::run_rfx_arm` unmodified.

```
grid (3081, 85, 1) dt=1.1675e-12 s; theta0 45 deg (k_y 148.199 rad/m); bw 0.1114
record 11259 steps (28 ext); masked 7.61-12.12 GHz (692 bins), gated 517
  max |R|        gated = 1.981e-10
  max |T - 1|    gated = 1.399e-06
  max |R + T - 1| gated = 1.399e-06
```

Every number is 3-7 decades under `LEAK_BAR = 1e-3`. Fresnel's `R = 0, T = 1` is
reproduced.

**What this excludes.** The 4-edge TF/SF corrections, the Bloch phase on the 3-D
y-roll (`bloch_phase_tuple`), the `(e^{−j k_y dx} − 1)/dx` transverse operator in
both `tfsf_2d.py` and `rfx/core/yee.py::_diff_fwd_o/_diff_bwd_o`, the Yee
y-offset question (in TMz-with-x-propagation the two components that cross the
x-faces, `Ez` and `Hy`, both sit at integer y, so no half-cell transverse phase
is owed), and the probe/extraction arithmetic. All of that carries the aux field
into the 3-D grid and back out to 1.4e-06.

**What this does NOT exclude — and this is the trap.** With no slab the probe's
total field *is* the injected field, and `E_inc` in the R/T definitions is the
same aux field. Any defect in the *content* of the incident field cancels
exactly. The vacuum arm is blind to incident-field purity by construction. So
the issue's own "if the vacuum arm also fails closure at 45°, the defect is in
the injection" is a one-way test: it passing does **not** clear the injection.

## 2. Discriminator 2 — the exact 2-D lattice as referee

`oblique_fresnel.py::yee_lattice_full` is a closed-form time-harmonic solution of
exactly the rfx lattice at fixed `k_y` (E nodes 0..nx-1, Hy on links, the CPML
recursion as `apply_cpml_e/h` realise it, the TFSF face corrections as forcing
terms, and — with `aux="model"` — the *aux grid's own* field as the incident).
The round-2 run already carries it per arm:

| arm | \|rfx − lattice\| mean R | \|lattice − Fresnel\| mean R | \|rfx − Fresnel\| mean R |
|---|---|---|---|
| te_00 | 0.0442 | 0.0455 | **0.0098** |
| te_30 | 0.0786 | 0.0789 | **0.0031** |
| te_45 | **0.0006** | 0.0950 | **0.0950** |
| te_60 | **0.0007** | 0.0722 | **0.0723** |
| tm_45 | **0.0003** | 0.0468 | **0.0468** |
| tm_60 | **0.0001** | 0.0204 | **0.0204** |

`R2/_26_oblique_results/rfx.json::arms.te_45.lattice.mean_dR_lattice_gated = 0.0006`,
`::arms.te_45.lattice.mean_W_lat_R_gated = 0.0950`,
`::arms.te_45.mean_dR_gated = 0.0950`;
`::arms.te_60.lattice.mean_dR_lattice_gated = 0.0007`;
`::arms.tm_45.lattice.mean_dR_lattice_gated = 0.0003`.

Read this carefully — it is the whole diagnosis in one table:

* **On the failing arms rfx reproduces the exact lattice to 3-7 parts in 10⁴.**
  The FDTD is doing exactly what the algorithm says. There is no coding error in
  the time-stepping, no float32 drift, no branch, no mask bug.
* The lattice **itself** is 0.095 away from Fresnel. So the error is in the
  *model of the rig*, i.e. in something `yee_lattice_full` faithfully includes:
  its absorbers and its incident field.
* **On the passing arms rfx does NOT match the lattice** (0.044, 0.079) — because
  `yee_lattice_full` is a *steady-state* solution and those arms' records are far
  too short to reach steady state. The passing arms pass because they stop early.

The lattice's own decomposition (all from
`R2/_26_oblique_results/rfx.json::arms.te_45.lattice`):

```
absorber_term_R_gated_max  = 0.2959   (rig-with-everything  vs  ideal termination + clean plane wave)
aux_echo_term_R_gated_max  = 0.2431   (aux="model"          vs  aux="plane")
cpml3d_term_R_gated_max    = 0.0639   (3-D CPML             vs  ideal termination)
```

82 % of the term is the **aux grid's own field**. That is the localisation.

Re-derived independently here (fresh evaluation, gated band, dx/2 rung):

| variant | te_00 max\|ΔR\| | te_45 max\|ΔR\| | te_45 max\|R+T−1\| |
|---|---|---|---|
| `aux="model"`, CPML (= the rig) | 0.1569 | 0.2964 | 0.2449 |
| `aux="plane"`, CPML | 0.0925 | 0.0643 | 0.0790 |
| `aux="plane"`, ideal termination | **0.0185** | **0.0052** | **0.0000** |

The last row is the irreducible lattice-dispersion floor, and at te_00 it is
0.0185 against the run's measured `max_dR_gated = 0.0186` — the model and the
FDTD agree on the clean case to 1e-4. With a clean incident field and a clean
absorber the rig would sit at 0.005 against a 0.0139 window at 45°.

## 3. Where the contamination comes from — measured on the aux grid itself

The lattice model accuses the aux grid, so I measured the aux grid directly in
the FDTD, with no slab and no 3-D domain at all: `init_tfsf_2d` +
`update_tfsf_2d_h/e` exactly as `run_rfx_arm` drives them, sampling `ez_2d` at 96
x-positions spanning the 3-D TFSF box, then the lane's own conjugated DFT, then a
two-mode least-squares fit `E(x) = A e^{−j k_x x} + B e^{+j k_x x}` at the
lattice `k_x` (`oblique_fresnel.py::yee_kx`).

```
te_45, 18083 steps (the arm's declared record), n2x=3092, src_x=33, i0_x=55
  |B/A| gated:  mean 5.9498e-02   max 6.0142e-02      two-mode fit residual max 3.15e-03
  |E| ripple across the x window, gated: max 1.2757e-01   (= 2|B/A|)
  MODEL aux_lattice_field |B/A| gated: mean 5.9485e-02   max 6.0018e-02
  |FDTD aux - model| / max|aux|, gated: 1.66e-02

te_00, 20000 steps (long enough to reach steady state), n2x=1602
  |B/A| gated:  mean 4.1804e-02   max 5.5047e-02      fit residual max 3.17e-06
  MODEL aux_lattice_field |B/A| gated: mean 4.1803e-02   max 5.5048e-02
  |FDTD aux - model| / max|aux|, gated: 1.28e-05

te_60, 27238 steps:  FDTD 5.8488e-02   MODEL 5.8490e-02
te_30,  6387 steps:  FDTD 1.0181e-02   MODEL (steady state) 5.9640e-02
```

Three things follow.

1. **The comparator's `aux_lattice_field` is exact** — it reproduces the FDTD aux
   field to 1.3e-05 (te_00) and 1.7e-02 (te_45, where the record is not yet fully
   settled). Everything below can therefore be done in the model.
2. **The contamination is angle-independent**: 4.2e-02 at normal incidence,
   5.9e-02 at 30/45/60. It is not an oblique effect and not a grazing-CPML
   effect. Normal incidence has it too — it is just never inside a te_00 record.
3. **te_30 at its own record shows 1.02e-02, not the steady-state 5.96e-02.** The
   echo is *partly* arrived at 6387 steps. That is the record dependence, seen
   directly in the incident field.

**Locating the reflector.** `B/A = ρ·e^{−2 j k_x L}`, so the phase slope against
`k_x` gives the distance from the fit origin to the reflector. Over 241 bins
(fit residual ≤ 2.2e-12):

```
te_30: d(arg B/A)/d(k_x) = -2.9949 m  ->  L = 1.4975 m = 2994.9 cells -> aux index 3069.9
te_45: d(arg B/A)/d(k_x) = -2.9948 m  ->  L = 1.4974 m = 2994.8 cells -> aux index 3069.8
te_60: d(arg B/A)/d(k_x) = -2.9950 m  ->  L = 1.4975 m = 2995.0 cells -> aux index 3070.0
```

`n2x = 3092`; the hi CFS-CPML occupies indices 3062..3091 (inner edge 3062).
The reflector sits at 3070 — **8 cells inside the aux grid's own 30-cell
absorber**, at the same place for every angle. That is the source of the −x wave.

## 4. Why the aux absorber is this bad

Control first: the same model with an outgoing-wave termination in place of the
CPML gives `|B/A| = 9.4e-15` (te_00) and `6.9e-15` (te_45) — the fit and the
model are exact, and the 6 % is entirely the absorber.

`tfsf_2d.py:196-208` uses `cpml_order = 4`, `kappa_max = 7.0`,
`sigma_max = 0.8·(m+1)/(η·dx)·kappa_max`. The `0.8(m+1)/(η·dx)` factor is already
the optimum; multiplying it by `kappa_max = 7` puts σ_max **70×** above the value
that minimises reflection for this depth. Model scan, te_45 geometry, aux grid
only, 30 cells:

```
as-shipped (m=4, kappa_max=7, sigma_max=148.647)          |B/A| 5.951e-02
best over (m, kappa_max, sigma_factor) at n=30            |B/A| 8.735e-03
   at m=3, kappa_max=2.0, sigma_factor=0.05, sigma_max=2.124
```

and the normal-incidence depth/σ scan with the *main* rfx law
(`_cpml_profile`: m=3, κ=1, σ_max = −ln R·(m+1)/(2 η n dx)):

```
n= 20  R_asym=1e-15  sigma_max= 9.17   |B/A| 3.745e-02
n= 30  R_asym=1e-15  sigma_max= 6.11   |B/A| 1.287e-02
n= 60  R_asym=1e-15  sigma_max= 3.06   |B/A| 2.902e-04
n= 60  R_asym=1e-06  sigma_max= 1.22   |B/A| 6.975e-05
```

So the absorber is **convergent** — it is not structurally broken, it is
mis-parameterised, and 30 cells is too shallow for the σ it carries.

## 5. Discriminator 4 — closure versus angle: a STEP, at the record/arrival crossing

Same rig, cheap rung (dx, K=1), θ₀ swept 30-60° with the lane's own record law
(the closed-form fallback plus its extensions, since only the seven declared arms
have a declared record), everything else untouched. `v_gx(f0)` from
`oblique_fresnel.py::yee_vgx`; "echo reaches probe" is the aux source → measured
reflector (index n2x−22) → reflection probe path at `v_gx`.

| θ₀ | record | max\|R+T−1\| | mean\|ΔR\| | v_gx(f0) | echo arrival (steps) |
|---|---|---|---|---|---|
| 30 | 3172 | **0.0010** | 0.0124 | 0.865 c | 3902 |
| 33 | 3502 | **0.0012** | 0.0134 | 0.838 c | 4029 |
| 36 | 4851 | **0.2638** | 0.0687 | 0.808 c | 4175 |
| 39 | 5222 | 0.2596 | 0.0676 | 0.777 c | 4345 |
| 42 | 5614 | 0.2592 | 0.0690 | 0.743 c | 4542 |
| 45 | 7362 | 0.4074 | 0.0719 | 0.707 c | 4772 |
| 48 | 6603 | 0.2694 | 0.0684 | 0.669 c | 5041 |
| 52 | 7428 | 0.2831 | 0.0561 | 0.616 c | 5475 |
| 56 | 8477 | 0.3331 | 0.0550 | 0.560 c | 6023 |
| 60 | 9898 | 0.3189 | 0.0841 | 0.501 c | 6730 |

**A step, two decades wide, between 33° and 36°** — and the record first exceeds
the echo arrival between exactly those two rows (3502 < 4029; 4851 > 4175). Not a
smooth `(k_y dx)²` or `tan θ` growth; not a mode crossing (nothing in the gated
band changes character at 34°); a threshold in the *record law*.

The complement, which isolates the angle: same sweep, record **forced to 4000
steps** for every θ₀ (witness bars disabled so no extension fires).

| θ₀ | mean\|ΔR\| | max\|R+T−1\| |
|---|---|---|
| 30 | 0.0456 | 0.3097 | (echo already in: arrival 3902 < 4000) |
| 33 | 0.0189 | 0.4661 | (arrival 4029 ≈ 4000) |
| 36 | 0.0153 | 0.1596 |
| 39 | **0.0154** | **0.0133** |
| 42 | **0.0164** | **0.0033** |
| 45 | **0.0176** | **0.0044** |
| 48 | **0.0188** | **0.0086** |
| 52 | **0.0205** | **0.0132** |
| 56 | 0.0210 | 0.0396 | (record now too *short*) |
| 60 | 0.0288 | 0.1226 | (record too short) |

At a record that sits *between* "the pulse and the slab ring-down have cleared
the probes" and "the aux echo has arrived", **45° is as accurate as 30°**. The
angle is innocent. The window closes at both ends and round 2 walked out of the
top of it.

Sharpened on the declared te_45 arm at the primary rung (dx/2), record forced:

| record (steps) | mean\|ΔR\| | max\|ΔR\| | max\|R+T−1\| |
|---|---|---|---|
| 4000 | 0.4050 | 2.0941 | 2.1478 | (pulse has not cleared — too short) |
| 6000 | **0.0044** | **0.0069** | 0.0256 |
| 8000 | **0.0043** | **0.0061** | **0.0042** |
| 10000 | 0.0716 | 0.3949 | 0.3619 |
| 14283 (`n_steps_min`) | 0.0949 | 0.2935 | 0.3133 |
| 18083 (declared) | 0.0950 | 0.3012 | 0.3082 |

At 8000 steps the failing arm reproduces Fresnel at mean 0.0043 against its
0.0139 mean window and closes energy to 4.2e-3. The step lands between 8000 and
10000 steps, against the predicted echo arrival of 9358 steps. **Same rig, same
code, same angle, same slab — only the record changed.**

## 6. Discriminator 5 — perturbing the accused expression

Replaced *only* `cfg.b_cpml / c_cpml / kappa_cpml` after `init_tfsf_2d` (same
30-cell depth, same code path, same everything else) with the in-family optimum
of section 4 (m=3, κ_max=2, σ_factor=0.05), and re-ran the failing arms at their
**declared** records:

| arm | mean\|ΔR\| shipped → retuned | max\|R+T−1\| shipped → retuned |
|---|---|---|
| te_45 | 0.0950 → **0.0348** | 0.3082 → **0.1928** |
| tm_45 | 0.0468 → **0.0173** | 0.3019 → 0.1971 |
| te_60 | 0.0723 → 0.0751 | 0.2563 → 0.3748 |

Touching *only* the aux absorber's three profile arrays moves te_45's mean error
by 2.7× and tm_45's by 2.7×, with no other change anywhere. That is a direct
causal demonstration on the accused expression. It does not *fix* the arm,
because a 6.8× reduction in `|B/A|` still leaves ~9e-3 of counter-propagating
incident field (9× over `LEAK_BAR`), and because the second term below is
untouched. The records move under the change (te_45 18083 → 14483, te_60
27238 → 28038) because the witnesses settle differently once the incident field
is cleaner; te_60 ends slightly worse, its residual now dominated by the 3-D
absorber term of section 7, whose echo its still-longer record admits.

## 7. The second term: the 3-D grid's own CPML

Exact lattice, te_45, with a *clean* incident plane wave (`aux="plane"`) so only
the 3-D absorber varies:

```
n_cpml declared= 20 ( 40 cells at dx/2): mean|dR| 0.0249  max|dR| 0.0629  max closure 0.0640
n_cpml declared= 40 ( 80 cells at dx/2): mean|dR| 0.0041  max|dR| 0.0154  max closure 0.0211
n_cpml declared= 80 (160 cells at dx/2): mean|dR| 0.0045  max|dR| 0.0054  max closure 0.0012
n_cpml declared=160 (320 cells at dx/2): mean|dR| 0.0045  max|dR| 0.0052  max closure 0.0000
ideal outgoing termination             : mean|dR| 0.0045  max|dR| 0.0052  max closure 0.0000
```

The declared depth (`N_CPML = 20`) leaves mean 0.0249 / closure 0.0640 on this
rig once the record is long enough to contain the echo; it converges to the
dispersion floor by 80 declared cells. Same mechanism, `rfx/boundaries/cpml.py`
rather than `tfsf_2d.py`, and it is the term the round-2 record law *does* try to
account for.

**Full budget for te_45 at the declared record** (mean \|ΔR\| over the gated band):

```
irreducible 2-D Yee lattice dispersion         0.0045
+ 3-D CPML echo (N_CPML = 20)                  0.0249   (+0.0204)
+ aux-grid CPML echo (n_cpml_2d = 30)          0.0950   (+0.0701)
measured rfx                                   0.0950
```

## 8. Why the round-2 record law did not catch it

`oblique_fresnel.py::predict_settling` computes `e_absorber` as "the largest
probe-field difference over the record between the rig with its CPML and the same
lattice with an outgoing-wave termination". Both sides of that difference are
built with `aux="model"` (the default of `yee_lattice_full`), i.e. **both carry
the aux grid's echo, so it cancels exactly**. `e_absorber` therefore measures the
3-D CPML term only, and the dominant term is invisible to it:

```
R2/_26_oblique_results/rfx.json::arms.te_45.run.record.e_absorber       = 2.984e-02
R2/_26_oblique_results/rfx.json::arms.te_45.run.record.W_absorber_R_max = 0.0429   (< W_bin 0.074 -> "ok")
R2/_26_oblique_results/rfx.json::arms.te_45.lattice.aux_echo_term_R_gated_max = 0.2431   (never gated)
```

The `aux_echo_term_R_gated_max` number was computed, written to the artifact, and
reported — it is in `evaluate_e2`'s `lattice` block — but it is a *reported*
witness, never a gate, and nothing compares it to `W_bin`. The 0.2431 was sitting
in the round-2 artifact from the moment it was written.

The tail witnesses do not catch it either: at te_45 the run reports
`purity_inc_rel = 9.27e-04`, `scat_refl_rel = 1.33e-03`,
`total_trans_rel = 6.86e-04` (`R2/_26_oblique_results/rfx.json::arms.te_45.tail`)
— all inside their bars. A steady 6 % standing wave in the *incident* field has a
flat envelope; it does not decay, so a "has it settled" witness reads it as
settled.

## 9. Dead ends, recorded

* **Vacuum at 45° (discriminator 1).** Clean to 1.4e-06. Excluded injection and
  extraction *fidelity* — but it is structurally blind to incident-field
  *content*, because R and T are normalised by the same aux field. It cost 28 s
  and it removed half the search, but on its own it is nearly misleading.
* **The Bloch phase / transverse operator / Yee y-offset.** Checked by hand
  against the field-transformation algebra (`_diff_fwd_o` carries
  `exp(−j k_y dx)` on the forward roll, `_diff_bwd_o` its conjugate on the
  backward roll; `bloch_phase_tuple` matches `tfsf_2d`'s `pshift`;
  `k_transverse = −direction_sign·k0·sinθ` is consistent between the aux grid and
  the 3-D roll). All correct. And the vacuum arm at 1e-10 proves it empirically.
  In TMz with +x propagation the only components crossing the x-faces are `Ez`
  and `Hy`, both at integer y — there is no half-cell transverse phase owed, so
  the issue's "y-offset of the Yee components" suspect does not apply on this
  path.
* **Staggered CPML profile.** Hypothesised that the CFS profile is evaluated at
  nodes and reused for the half-cell H links without a ½-cell shift (true in both
  `tfsf_2d.py` and `rfx/boundaries/cpml.py`), and that this sets a
  depth-independent floor. **Wrong.** Adding the ½-cell shift makes it *worse*:
  te_00 4.19e-02 → 8.28e-02, te_45 5.95e-02 → 1.18e-01. Not the mechanism.
* **`alpha_max`.** Setting `alpha = 0` changes `|B/A|` from 4.19e-02 to 4.40e-02
  (te_00). Irrelevant.
* **Deeper aux absorber alone.** `n = 60` at the shipped σ gives 2.84e-02 (2×
  better); `n = 120` gives 2.84e-01 (worse) because σ_max is not rescaled with
  depth. Depth without re-deriving σ is not a fix.
* **Grazing-angle CPML failure.** The natural first guess, and it is wrong: the
  contamination is 4.18e-02 at *normal* incidence and 5.85e-02 at 60°. Flat.
* **Frequency masking / evanescent bins entering the gated band.** `gated_mask`
  is clean; the step in section 5 sits at 34°, where nothing enters or leaves the
  band. Excluded by the fixed-record sweep, which is flat across the same angles.

## 10. What is fixed, what is not, and what would settle the rest

**Established, with evidence:** the injected incident field carries a 4-6 %
counter-propagating component generated 8 cells inside the aux grid's 30-cell
CFS-CPML; the R/T normalisation hides it in vacuum; the round-2 record law admits
it above ≈34°; that fully accounts for the 45°/60° failures and the closure
violation; and the same class of term at the 3-D absorber accounts for the
remainder.

**No fix is committed here**, and this is not the one-line defect the brief
allowed me to fix in-lane:

* the accused expression is `tfsf_2d.py:200-208` (the profile constants), but
  section 6 shows that retuning them inside the current family at `n = 30`
  reduces the error 2.7× and does not clear the window;
* `n_cpml_2d`, `cpml_order`, `kappa_max` and `sigma_max` in `tfsf_2d.py` set the
  injected field for **every** consumer of the oblique Bloch path (the #404 lane,
  `compute_rcs`, the oblique waveguide arms), so changing them re-baselines
  committed numbers well outside cv26 — which the brief forbids;
* the 3-D `N_CPML = 20` term needs its own decision, and it is a declared rig
  constant.

**Fix candidates, in the order I would try them:**

1. **Make the incident field clean at source.** Re-derive the aux profile from
   the reflection target rather than from `0.8(m+1)/(η dx)·κ_max`, and deepen the
   aux absorber. Measured: `n = 60`, m=3, κ=1, `R_asym = 1e-6` gives
   `|B/A| = 6.98e-05` — 600× better than shipped and 14× under `LEAK_BAR`. The
   aux layout (`n2x = 30 + 25 + n_tfsf + 25 + 30`, `i0_x = 55`) must grow with
   the absorber, since 60 CPML cells would otherwise overlap `i0_x`.
2. **Bound the aux echo in the record law.** `predict_settling`'s `e_absorber`
   should difference the rig against `aux="plane", ideal_absorber=True` — i.e.
   against a clean incident wave — instead of letting the aux echo cancel. That
   turns the existing, already-computed `aux_echo_term_R_gated_max = 0.2431` into
   something the record has to answer for. This is a *comparator* change and it
   would have failed round 2 honestly instead of silently.
3. Normalise R and T by an analytic lattice plane wave rather than by the aux
   probe sample. This removes the standing-wave modulation from the denominator
   but not the −x wave injected into the total-field region, so it is a
   half-measure.

**Still unknown, and what would settle it:**

* Whether the 3-D CPML term (section 7) is the same mis-parameterisation. The
  aux-geometry scan at the *main* rfx law predicts ~1.3e-02 at 30 cells and
  2.9e-04 at 60, while the measured 3-D term at 40 cells (dx/2) is larger than
  that scaling suggests. **Settled by:** running te_45 at the declared record
  across the lane's existing `CPML_DEPTH_LADDER` on the *wide* rig (it is
  currently only run on the compact grazing rig) and checking that mean \|ΔR\|
  falls toward the 0.0045 floor.
* Whether cv04's committed `per_bin_max_RT_closure = 0.0487` envelope is the same
  effect at normal incidence, held down only by cv04's shorter records. The
  te_00 aux measurement (4.18e-02 backward amplitude in steady state) says the
  contamination exists there too. **Settled by:** re-running cv04 with a record
  extended past its own absorber echo arrival and seeing whether the closure
  grows to the same 0.2-0.3 scale.
* Whether the shipped `kappa_max = 7` in `tfsf_2d.py` was ever validated against
  a reflection measurement, or inherited. **Settled by:** git archaeology on the
  #404 lane and a one-off aux-reflection sweep committed as a rig check — the aux
  grid has no reflection gate today, which is why a 6 % absorber shipped.

---

## Reproduction

All local, `~/Documents/rfx/.venv/bin/python`, worktree
`~/Documents/rfx-worktrees/issue-888` with the lane files checked out from
`origin/agent/gap4-oblique-fresnel`. Wall time: vacuum arm 28 s, aux-purity
measurement 5-21 s per angle, θ sweep ~4 min, record sweep ~3 min, retuned-arm
runs ~2 min. No VESSL run was needed.

* §1 — `run_rfx_arm(dict(arm_spec("te_45"), slab=False, eps_slab_rfx=1.0, mu_slab_rfx=1.0), dx_div=2)`.
* §3 — `init_tfsf_2d` + `update_tfsf_2d_h/e` driven standalone; sample `ez_2d`
  at 96 x-positions across the box; the lane's `fft(conj(x))` convention; then a
  two-mode least-squares fit at `yee_kx`, and `L = −½·d(arg B/A)/d k_x`.
* §4, §7 — `oblique_fresnel.aux_lattice_field` / `yee_lattice_full` with the
  profile and `n_cpml` varied.
* §5 — `run_rfx_arm` with `O.derive_record` wrapped to force `n_steps` and the
  witness bars raised so no extension fires.
* §6 — `rfx.sources.tfsf_2d.init_tfsf_2d` wrapped to `_replace` the three profile
  arrays after construction.
