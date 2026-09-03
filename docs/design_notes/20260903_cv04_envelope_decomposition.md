# cv04's committed envelope: does it contain the auxiliary-grid echo? (issue #888 follow-up)

Branch `agent/cv04-aux-echo-measurement`, off `origin/main` @ `0141f39e`.
Reads the diagnosis of #888 (`docs/design_notes/20260903_cv26_oblique_defect_diagnosis.md`
on `agent/issue-888-oblique-diagnosis` @ `831ea3c`) and the lattice-witness
standard merged in #886 (`docs/design_notes/20260903_lattice_witness_standard.md`).

**Nothing in this note changes a gate, a window, a declared record or a
committed number.** Every measurement below was run from a scratch harness that
re-implements `validation/crossval/04_multilayer_fresnel.py` PART 1 + PART 2
verbatim; the repository is untouched apart from this file. The harness
reproduces cv04's committed numbers exactly (§1.1), which is what licenses the
rest.

---

## 0. Verdict up front

**No.** cv04's committed closure envelope does not contain the auxiliary-grid
echo. The echo term in cv04's committed `mean|ΔR| = 0.0066` and
`per_bin_max_RT_closure = 0.0487` is **exactly zero — bit-identical, not merely
small.**

The contamination is real and it is present in cv04's rig at the amplitude #888
predicted: the 1-D auxiliary grid of `rfx/sources/tfsf.py` reflects
**|B/A| = 4.40e-02** in steady state, from a reflector **6.9 cells inside its own
20-cell CPML**. But cv04's record ends **511 steps (0.58×) before the echo's
first arrival**, and over that record the backward content of the injected
incident field is **1.04e-05**, which is the float32 fit floor and not an echo
(the echo-free control gives the same 1.0358e-05 to every digit).

Replacing the auxiliary absorber with an echo-free one (the aux array padded
4000 cells at its hi end, `|B/A| = 1.0e-05`) changes cv04's four probe time
series by **0.000e+00** over all 719 steps, and `max|R_shipped − R_padded| =
0.000e+00` over all 170 mask bins.

So the family's windows rest on **one** artefact, not two. What cv04's envelope
carries is the Yee-lattice term #886 identified, and — new here — a *signed*
record-truncation term that partially cancels it in the band mean. The lattice
term alone over-explains the envelope (0.00728 against 0.0066, ratio 1.10), and
§5 shows why the two terms are not independent and which one is doing it.

The defect itself is confirmed at normal incidence, and it is one record length
away on every rung in the family: extend cv04's rig past its arrival and it
reproduces cv26's failure exactly — `mean|ΔR|` 0.0073 → 0.0517, `max|R+T−1|`
0.0004 → 0.258 (§3). **All 13 committed slab-family rungs run at 0.50–0.57 of
their own echo arrival** (§6), which is structural rather than lucky: the record
law is `0.95 × 2·dist(probe → 3-D CPML)/v` measured from t = 0, while the aux
echo's path is measured from the *source*, roughly twice as long.

---

## 1. The rig, and what differs from cv26

cv26 injects through `rfx/sources/tfsf_2d.py` — a 2-D auxiliary grid with a
30-cell CFS-CPML at `cpml_order = 4`, `kappa_max = 7.0`,
`sigma_max = 0.8(m+1)/(η dx)·κ_max`. **cv04 injects through
`rfx/sources/tfsf.py`, a different absorber**: a 1-D grid, `n_cpml_1d = 20`
hard-coded, cubic grading, no κ (`tfsf.py:264-287`):

```
n_cpml_1d = 20
sigma_max = 0.8 * 4.0 / (eta * dx_1d)      # = 0.8(m+1)/(eta dx), m = 3, kappa = 1
sigma_prof = sigma_max * rho**3
alpha_prof = 0.05 * (1 - rho)
```

This is the textbook optimum without the ×7 that #888 §4 identified as the
cv26 mis-parameterisation, so the answer was genuinely open: cv04's aux
absorber might have been fine. **It is not** (§4): 20 cells is too shallow for
this σ, and the measured reflection is the same 4-6 % class.

Geometry, from the committed script (`04_multilayer_fresnel.py:76-179`):

| quantity | value |
|---|---|
| `nx_interior` / `n_cpml` / grid `nx` | 600 / 20 / 641 |
| `dx` / `dt` | 1.0e-3 m / 2.335067793382187e-12 s |
| TFSF box `x_lo`, `x_hi`; `i0` | 25, 615; 30 |
| aux array length `n_1d`; `src_idx` | 652; 23 |
| aux hi CPML | indices 632..651 |
| slab cells | [315, 325) |
| probes (3-D) refl / trans | 285 / 355 |
| aux reference indices refl / trans | 290 / 360 |
| `dist_to_cpml_hi` / `lo` | 266 / 265 |
| `t_safe_steps` hi / lo | 721 / 719 |
| **committed record `n_steps`** | **719** |
| lattice `v_g(10 GHz)` | 0.69808 cells/step (0.99719 c) |

### 1.1 The harness reproduces the committed numbers exactly

At `n_steps = 719`, 170 mask bins over 3.032–11.867 GHz:

| quantity | committed | harness |
|---|---|---|
| `mean\|ΔR\|` vs TMM | `tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[0].observed_baseline = 0.0066` | 0.006618 |
| `mean\|ΔT\|` vs TMM | `tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[1].observed_baseline = 0.011` | 0.011023 |
| `mean\|R+T−1\|` | `tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[2].observed_baseline = 0.0091` | 0.009080 |
| `max\|R+T−1\|` | `validation/crossval/04_multilayer_fresnel.py:333` (code comment, rung C4, job 369367246779) `= 0.0487` — cited as `:309` in `comparators/cv22_dispersive_gates.py:121-124`, which is stale by 24 lines | 0.048736, worst bin 11.8669 GHz (bin 169 of 170) |
| `\|rfx − lattice\|` gated mean R | `validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.mean_dR_lattice_gated = 0.0016818844941439814` | 0.001682 |
| `\|rfx − lattice\|` gated mean T | `validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.mean_dT_lattice_gated = 0.006251101710953211` | 0.006251 |
| gated bins | `validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.n_bins_gated = 115` | 115 |

The worst-bin frequency (11.87 GHz, the mask edge) matches the committed code
comment's own attribution. The exact 1-D lattice is
`validation/crossval/comparators/dispersive_eps.py::yee_lattice_slab_rt(f, 4.0, 0.0, 0.010, 1e-3, dt)`;
the gated band is `validation/crossval/comparators/cv22_dispersive_gates.py::gated_mask`
(4–10 GHz).

---

## 2. Task 1 — the echo arrival at cv04's geometry, against its committed record

**Where the reflector is.** Driving the 1-D aux grid standalone
(`init_tfsf` + `update_tfsf_1d_h/e`, exactly as `run_rfx_arm` drives them),
sampling `e1d` at 96 x-positions spanning the mapped TFSF region, taking the
case's own `np.fft.rfft`, and least-squares fitting
`E(x) = A e^{−j k x} + B e^{+j k x}` at the 1-D Yee lattice
`k = (2/dx) asin(ŵ dx/2c)` over 676 bins:

```
d(arg B/A)/dk = -1.277755 m   ->   L = 0.638877 m = 638.88 cells
phase-fit residual 3.40e-03
```

`n_1d = 652`, hi CPML at 632..651, so the reflector sits at index **638.9 —
6.9 cells inside the 20-cell absorber**, the same picture as #888's "8 cells
inside 30" on the 2-D grid.

**When it arrives.** Path = (source → reflector) + (reflector → sample), at
`v_g(f0) = 0.69808` cells/step:

| target | path (cells) | predicted arrival (pulse centre) | predicted leading edge (−3τ = 82 steps) |
|---|---|---|---|
| aux trans reference (index 360) = 3-D trans probe 355 | 894.8 | 1282 | 1200 |
| aux refl reference (index 290) = 3-D refl probe 285 | 964.8 | 1382 | 1300 |

**Measured, not predicted.** Running the full cv04 rig twice — shipped aux and
an echo-free aux (§4.2) — and taking the first step at which the two diverge in
float32:

| series | first non-zero difference | >1e-4 of peak | >1e-3 | >1e-2 |
|---|---|---|---|---|
| aux incident @ trans ref | **1237** | 1277 | 1291 | 1310 |
| aux incident @ refl ref | 1333 | 1377 | 1391 | 1410 |
| 3-D total @ trans probe | **1230** | 1276 | 1290 | 1308 |
| 3-D total @ refl probe | 1350 | 1392 | 1406 | 1425 |

**Answer to task 1: the committed record ends BEFORE the arrival, by 511
steps.** 719 against the earliest measured arrival of 1230 — the record is
**0.58×** the arrival, i.e. it stops 41.5 % of the way short. Against the
predicted centre (1282) the margin is 563 steps; against the most conservative
predicted leading edge (1200) it is 481 steps. cv26's failing arms ran
**1.9–2.1× past** their arrivals; cv04 runs 0.58× of its own.

The pulse is not wide enough to close that gap: τ = 63.66 ps = 27.3 steps, and
the arrival numbers above already start the clock at t = 0 rather than at the
source centre `t0 = 81.8` steps, so 1200 is a hard lower bound.

---

## 3. Task 2 — record sweep across the crossing

cv04's rig, committed geometry, records from 719 to 3000. Two arms at every
record: **shipped** (the rig as committed) and **padded** (the aux array
extended 4000 cells at its hi end so its echo cannot return — §4.2). Full
committed mask (3–15 GHz ∩ incident amplitude > 2 % of peak); `nfft` doubles
above 1024 and again above 2048 steps, which is why the bin count changes.

| record | arm | bins | `mean\|ΔR\|` vs TMM | `mean\|ΔT\|` vs TMM | `max\|R+T−1\|` | `mean\|ΔR\|` vs LATTICE | `mean\|ΔT\|` vs LATTICE |
|---|---|---|---|---|---|---|---|
| **719 (committed)** | shipped | 170 | **0.00662** | **0.01102** | **0.04874** | 0.00268 | 0.01057 |
| 719 | padded | 170 | 0.00662 | 0.01102 | 0.04874 | 0.00268 | 0.01057 |
| 800 | shipped | 170 | 0.00725 | 0.00724 | 0.00406 | 0.00032 | 0.00026 |
| 800 | padded | 170 | 0.00725 | 0.00724 | 0.00406 | 0.00032 | 0.00026 |
| 900 | shipped | 170 | 0.00728 | 0.00730 | 0.00072 | 0.000023 | 0.00014 |
| 900 | padded | 170 | 0.00728 | 0.00730 | 0.00072 | 0.000023 | 0.00014 |
| 1000 | shipped | 170 | 0.00728 | 0.00730 | 0.00071 | 0.000015 | 0.00010 |
| 1000 | padded | 170 | 0.00728 | 0.00730 | 0.00071 | 0.000015 | 0.00010 |
| 1100 | shipped | 340 | 0.00727 | 0.00726 | 0.00044 | 0.000011 | 0.00007 |
| 1100 | padded | 340 | 0.00727 | 0.00726 | 0.00044 | 0.000011 | 0.00007 |
| 1200 | shipped | 340 | 0.00727 | 0.00726 | 0.00037 | 0.0000076 | 0.00005 |
| 1200 | padded | 340 | 0.00727 | 0.00726 | 0.00037 | 0.0000076 | 0.00005 |
| 1300 | shipped | 340 | 0.01487 | 0.01433 | 0.16536 | 0.01350 | 0.01332 |
| 1300 | padded | 340 | 0.01487 | 0.01887 | 0.21391 | 0.01350 | 0.01818 |
| 1400 | shipped | 340 | 0.01742 | 0.11100 | 0.31359 | 0.01537 | 0.11097 |
| 1400 | padded | 340 | 0.01892 | 0.07267 | 0.19945 | 0.01711 | 0.07282 |
| 1600 | shipped | 336 | 0.05162 | 0.10157 | 0.24984 | 0.05075 | 0.10163 |
| 1600 | padded | 340 | 0.02889 | 0.06564 | 0.14668 | 0.02813 | 0.06547 |
| 1800 | shipped | 336 | 0.05152 | 0.10158 | 0.24938 | 0.05066 | 0.10161 |
| 1800 | padded | 340 | 0.02883 | 0.06565 | 0.14737 | 0.02808 | 0.06547 |
| 2200 | shipped | 671 | 0.05156 | 0.10149 | 0.25193 | 0.05068 | 0.10150 |
| 2200 | padded | 679 | 0.02881 | 0.06557 | 0.14850 | 0.02799 | 0.06548 |
| 3000 | shipped | 671 | 0.05168 | 0.10155 | 0.25798 | 0.05071 | 0.10153 |
| 3000 | padded | 679 | 0.02887 | 0.06574 | 0.14161 | 0.02804 | 0.06555 |

Same, over the gated band (`cv22_dispersive_gates.gated_mask`, 115/229/459 bins):

| record | arm | `mean\|ΔR\|` TMM | `mean\|ΔT\|` TMM | `max\|R+T−1\|` | `mean\|ΔR\|` LATTICE |
|---|---|---|---|---|---|
| **719 (committed)** | shipped = padded | **0.00625** | **0.00872** | **0.01639** | **0.00168** |
| 900 | shipped = padded | 0.00531 | 0.00531 | 0.00033 | 0.0000074 |
| 1200 | shipped = padded | 0.00529 | 0.00529 | 0.00011 | 0.0000029 |
| 1600 | shipped | 0.04760 | 0.10783 | 0.24984 | 0.04722 |
| 1600 | padded | 0.02423 | 0.06966 | 0.14668 | 0.02381 |
| 3000 | shipped | 0.04763 | 0.10775 | 0.25798 | 0.04709 |
| 3000 | padded | 0.02433 | 0.06958 | 0.14161 | 0.02382 |

Read this in three pieces.

1. **Across cv04's own crossing there is nothing to step on.** At 719 the
   shipped and padded arms are *identical in every digit* — see §4.3, they are
   bit-identical in the time domain. The record ladder is therefore flat with
   respect to the aux echo from 719 all the way to 1200: `shipped − padded = 0`
   at every one of those records. **By the brief's own criterion — "if the
   residual is flat across the crossing, the echo is not in cv04's envelope" —
   the answer is no.**

2. **There IS a settled plateau, and it is the lattice.** From 800 to 1200 the
   residual against the continuum transfer matrix sits at **0.00727–0.00728**,
   which is `|lattice − TMM|` over the same mask to five decimals (§5), while
   `|rfx − lattice|` falls to **7.6e-06** and `max|R+T−1|` to 3.7e-04. In that
   window rfx *is* the exact 1-D Yee lattice, to seven parts in a million. This
   plateau is new: it did not exist in the committed record, which sits below it.

3. **Past 1230 the rig fails exactly as cv26 does, at normal incidence.**
   `mean|ΔR|` 0.0073 → 0.0517, `max|R+T−1|` 0.0004 → 0.258. The budget has the
   same shape as #888 §7's te_45 budget:

   | term | cv04 (θ = 0°, this note) | cv26 te_45 (#888 §7) |
   |---|---|---|
   | irreducible Yee lattice dispersion | 0.0073 | 0.0045 |
   | + 3-D CPML echo (`N_CPML = 20`) | 0.0289 | 0.0249 |
   | + aux-grid CPML echo | **0.0517** | 0.0950 |

   The angle was never the variable in cv26 and it is not the variable here.
   The only reason cv04 is green and cv26 te_45 is red is that cv04 stops at
   0.58× its arrival and round-2 te_45 ran to 1.9×.

A caveat on the ≥1300 rows, stated because it bounds what may be claimed from
them: at cv04's geometry **the two echoes arrive together**. The aux hi-CPML
sits only 6 cells outside the 3-D hi-CPML (aux index 632 maps to 3-D 627; the
3-D CPML starts at 621), and the aux echo's path is measured from the source
while the 3-D echo's is measured from the probe, so the two round trips differ
by ~14 cells out of ~2240. They cannot be separated by geometry in this rig
(dead end, §8); the `shipped − padded` difference is what separates them, and
that is what the padded column is for.

---

## 4. Task 3 — direct measurement of the contamination in cv04's own rig

### 4.1 The steady-state backward content

Two-mode fit on cv04's own aux grid (96 samples over aux indices 40–620,
2500-step drive so the echo has fully passed the window), 676 bins,
3.006–11.828 GHz:

```
|B/A| gated:  mean 4.4023e-02   max 5.5471e-02   min 2.9632e-02
two-mode fit residual max 1.81e-03
```

Convergence in the record used for the fit: 1.92e-03 at 1000 steps, 1.13e-02 at
1200, 3.18e-02 at 1600, 4.400e-02 at 2000, and 4.4023e-02 from 2500 on.

Cross-check at the cv22/cv23 geometry (`nx_interior = 1000`, aux `n_1d = 1053`):
`|B/A|` mean **4.4028e-02**, max 5.5432e-02, reflector at aux index 1038.9 —
same amplitude, same absorber depth. The number is a property of
`tfsf.py`'s profile, not of the box.

**This is the same 4-6 % class #888 measured on the 2-D aux grid**
(4.18e-02 at 0°, 5.85–5.96e-02 at 30/45/60°). Two absorbers with different
parameterisations — cv26's κ = 7 fourth-order 30-cell and cv04's κ = 1
third-order 20-cell — land within 5 % of each other. The mis-parameterisation
#888 accused (`kappa_max = 7`) is therefore **not the whole story**: cv04's
absorber does not carry it and still reflects 4.4 %. 20 cells is simply too
shallow at the shipped σ, which is what #888 §4's depth scan already showed
(`n = 20 → 3.7e-02`, `n = 60 → 2.9e-04`).

### 4.2 The echo-free control

A retuned 20-cell profile is *not* a usable control: scanning
(m, κ_max, σ_factor) over the aux grid, the family floor at n = 20 is
`|B/A| = 2.36e-02` (m = 3, κ = 1, σ_factor = 0.2) — only 1.9× better than
shipped (dead end, §8). Instead the control **pads the aux array 4000 cells at
its hi end**. Every index the 3-D TFSF corrections read (`i0`,
`i0 + (x_hi − x_lo)`, `i0 + (x_hi + 1 − x_lo)`, `i0 − 1`) and the source index
are at the lo end and are unchanged; the hi CPML simply moves 4000 cells
further out, so its echo cannot return inside any record run here. Measured on
the padded grid, 2500 steps:

```
padded aux, |B/A| mean 1.0028e-05   max 3.1932e-05     (4390x below shipped)
```

### 4.3 The echo term in cv04's committed numbers

**Over cv04's own 719-step record**, fitting the same two modes on a sample
window the direct pulse fully clears inside the record (aux indices 40–280,
64 samples, 170 bins):

| rig | `\|B/A\|` mean | `\|B/A\|` max |
|---|---|---|
| shipped aux, 719 steps | **1.0358e-05** | 2.5615e-05 |
| echo-free aux, 719 steps | **1.0358e-05** | 2.5615e-05 |
| shipped aux, 1500 steps | 6.3207e-03 | 1.2231e-02 |
| shipped aux, 2500 steps | 4.4024e-02 | 5.5502e-02 |

The shipped and echo-free numbers are equal to every printed digit, so the
1.04e-05 is the float32 fit floor, not an echo. **cv04's record contains 0.024 %
of the contamination that exists on its grid.**

And end to end, the whole rig at the committed record:

```
max |ts_inc_refl_shipped  - ts_inc_refl_padded |  = 0.000e+00   (719 steps)
max |ts_inc_trans_shipped - ts_inc_trans_padded|  = 0.000e+00
max |ts_refl_shipped      - ts_refl_padded      |  = 0.000e+00
max |ts_trans_shipped     - ts_trans_padded     |  = 0.000e+00
max |R_shipped - R_padded| = 0.000e+00   over all 170 mask bins
max |T_shipped - T_padded| = 0.000e+00

shipped: mean|dR| 0.006618  mean|dT| 0.011023  max|R+T-1| 0.048736
padded : mean|dR| 0.006618  mean|dT| 0.011023  max|R+T-1| 0.048736
```

**The echo term in R at cv04's committed record, in the form #888 reports
`aux_echo_term_R_gated_max` for the oblique arms:**

| record | `\|R_shipped − R_padded\|` mask mean / max | gated mean / max |
|---|---|---|
| **719 (committed)** | **0.00000 / 0.00000** | **0.00000 / 0.00000** |
| 800 – 1300 | 0.00000 / 0.00000 | 0.00000 / 0.00000 |
| 1400 | 0.00189 / 0.02337 | 0.00065 / 0.00324 |
| 1600 | 0.02919 / 0.09720 | 0.02608 / 0.08136 |
| 3000 | 0.02910 / 0.09625 | 0.02593 / 0.08491 |

**Answer to task 3.** `|B/A| = 4.40e-02` exists on cv04's grid; the echo term it
produces in R at cv04's committed record is **0.0000**. It accounts for
**0.0 % of `mean|ΔR| = 0.0066`** and **0.0 % of `per_bin_max_RT_closure =
0.0487`**. Had the record reached steady state on the same geometry it would
account for 0.029 in `mean|ΔR|` — **4.4× the whole committed envelope** — and
0.10 in the worst bin.

---

## 5. Task 4 — decomposing the envelope

Over cv04's committed 170-bin mask, all four terms measured on the same run:

| term | how it is measured | value (R) | value (T) |
|---|---|---|---|
| committed envelope | `tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[0].observed_baseline` / `[1]` | **0.0066** | **0.011** |
| lattice term `mean\|lattice − TMM\|` | `dispersive_eps.yee_lattice_slab_rt` vs `fresnel_slab_RT`, same mask | **0.00728** | **0.00728** |
| **aux-grid echo term** | `\|R_shipped − R_padded\|`, §4.3 | **0.00000** | **0.00000** |
| 3-D CPML echo term | flat plateau 800–1200 at the lattice value, §3 | **0.00000** | **0.00000** |
| record truncation `mean\|rfx − lattice\|` | measured, cv04's own rung | 0.00268 | 0.01057 |

(The #886 lane quotes 0.00727 for the lattice term over the same mask
— `docs/design_notes/20260903_lattice_witness_standard.md` §5.3; the harness
gets 0.007282. Gated, both give 0.00531.)

**The arithmetic does not close, and it over-explains.**

```
lattice 0.00728  +  echo 0.00000  +  3-D CPML 0.00000  =  0.00728
committed envelope                                     =  0.00662
                                                over-explained by 0.00066 (110 %)
```

Adding the truncation term as a magnitude makes it worse:

```
mean|lattice - TMM|   0.00728
mean|rfx - lattice|   0.00268
                sum   0.00997   against a total of 0.00662  -> 151 %
```

**Said plainly: the terms are not independent, and the one that is not
independent is the record truncation.** It is a *signed* per-bin perturbation
that partially cancels the lattice term in the band mean, so the means of
absolute values do not add. The evidence is direct:

* over the 170 mask bins, the residual `R_rfx − R_lattice` **opposes** the
  lattice term `R_lattice − R_TMM` in **85 of 170 bins (50 %)**, with Pearson
  `r = −0.203`;
* removing the truncation — running the same rig to 900–1200 steps and changing
  nothing else — makes `mean|ΔR|` go **up**, from 0.00662 to 0.00728, i.e.
  exactly onto the lattice term. A term that gets *larger* when a contamination
  is removed is a term whose contamination was cancelling, not adding.

So the honest decomposition of cv04's committed `mean|ΔR| = 0.0066` is:

```
Yee-lattice second-order term                    +0.00728   (110 % of the envelope)
record truncation, in the band mean              -0.00066   (signed, partially cancelling)
auxiliary-grid CPML echo                          0.00000   (  0 % )
3-D CPML echo                                     0.00000   (  0 % )
                                                  -------
committed envelope                                0.00662
```

**What is left for genuine physics: nothing measurable.** Once the record
settles (900–1200 steps, geometry unchanged) `|rfx − lattice|` is
**1.5e-05 over the mask / 5.3e-06 gated** at 1000 steps, falling to 7.6e-06 /
2.9e-06 at 1200. That is the solver's entire non-discretisation residual on this
case, and it is **440× smaller** than the envelope. cv04's committed envelope is
the lattice term and nothing else; the 10 % it falls short by is the truncation
cancelling, not physics.

This tightens #886 §5.3 rather than contradicting it. #886 said the
identification held for `|ΔR|` (lattice/envelope 1.10) and not for `|ΔT|`
(0.66, "more than half of the 0.011 envelope is this record's TRUNCATION"). The
settled plateau now measures the T side directly: at 900–1200 steps
`mean|ΔT|` = 0.00726–0.00730, against a committed 0.011. **Removing the
truncation and nothing else drops the T residual by a factor of 1.51, onto the
same 0.00728 the R side lands on** — so the committed T envelope is
truncation-dominated in the sense that a third of it disappears when the record
settles, while the R envelope goes the other way (0.00662 → 0.00728, up by
1.10). Same conclusion as #886 §5.3, now with a measured settled value rather
than an inference, and with the echo excluded as a candidate for either
direction. (The individual terms still must not be subtracted from each other:
`mean|ΔT|` is a mean of absolute values and the truncation is signed, exactly as
in the R decomposition above — `0.00728 + 0.01057 = 0.0179` against a total of
0.01102 over-explains by 63 %.)

---

## 6. Task 5 — the consequence for the family's windows

`W_MEAN_R = gate_from_envelope(0.0066, quantum=1000) = 0.010`
(`validation/crossval/comparators/cv22_dispersive_gates.py:126`;
`ENVELOPE_GATE_MULTIPLIER = 1.5`, `nu_cavity_gates.py:618-622`).

**How much of `W_MEAN_R = 0.010` is lattice, how much is echo:**

| component of the envelope | value | share of the 0.0066 envelope | share of the 0.010 window |
|---|---|---|---|
| Yee-lattice second-order term | 0.00728 | **110 %** | `1.5 × 0.00728 = 0.0110`, i.e. **the whole window and 10 % more** |
| auxiliary-grid echo | 0.00000 | **0 %** | **0.000** |
| 3-D CPML echo | 0.00000 | 0 % | 0.000 |
| remainder available to genuine physics | −0.00066 | **−10 %** | none |

**A window derived from the remainder.** cv04 now has its own settled rung (this
note, §3), which is sharper than the settled rung #886 had to borrow:

| source of the "non-lattice part" | `mean\|rfx − lattice\|` gated R | `gate_from_envelope(·, quantum=1000)` | `gate_from_envelope(·, quantum=1e6)` |
|---|---|---|---|
| #886's borrowed rung (cv23 `sigma_zero`, 1078 steps) | 1.69e-04 | **0.001** | 2.54e-04 |
| cv04's own settled rung, 1000 steps, this note | **5.31e-06** | **0.001** | **8e-06** |
| cv04's own settled rung, 1200 steps | 2.95e-06 | 0.001 | 5e-06 |

At `quantum = 1000` the re-derived window is **0.001** — identical to #886's,
now confirmed from cv04's own rig instead of borrowed from cv23's, and
**unchanged by this note's result, because the echo term is zero**. The 0.001
is entirely a quantisation floor: the underlying number is 8e-06, and at
`quantum = 1e6` the window would be 8e-06 rather than 0.001.

**Which cases a re-derived window would fail.** At `W_MEAN_R = 0.001` the list is
exactly #886 §5.4's, since the echo contributes nothing to move it — **7 of the
12 committed rungs**, using the numbers already committed in the artifacts:

| rung | committed `mean\|ΔR\|` vs TMM, gated | that rung's own lattice term | ≤ 0.001? |
|---|---|---|---|
| cv22 `debye` | 0.0023 | 0.0022 | **FAIL** |
| cv22 `lorentz` | 0.0028 | 0.0028 | **FAIL** |
| cv22 `drude` | 0.00049 | 0.00049 | pass |
| cv23 `tand0p1` | 0.0039 | 0.0039 | **FAIL** |
| cv23 `tand0p1_dx2` | 0.00096 | 0.00096 | pass |
| cv23 `tand0p1_dx4` | 0.00024 | 0.00024 | pass |
| cv23 `tand1` | 0.0051 | 0.0051 | **FAIL** |
| cv23 `tand1_dx2` | 0.0013 | 0.0013 | **FAIL** |
| cv23 `tand1_dx4` | 0.00031 | 0.00031 | pass |
| cv23 `tand3` | 0.0031 | 0.0031 | **FAIL** |
| cv23 `tand3_dx2` | 0.0031 | 0.0031 | **FAIL** |
| cv23 `tand3_dx4` | 0.00078 | 0.00078 | pass |

Primary arms verified against the committed artifacts here:
`validation/crossval/_22_dispersive_results/rfx.json::arms.debye.mean_dR_gated = 0.0022918393481320366`,
`::arms.lorentz.mean_dR_gated = 0.0028347900514427015`,
`::arms.drude.mean_dR_gated = 0.0004947077441103369`;
`validation/crossval/_23_lossy_results/rfx.json::arms.tand0p1.mean_dR_gated = 0.00391892210699406`,
`::arms.tand1.mean_dR_gated = 0.005093978067061866`,
`::arms.tand3.mean_dR_gated = 0.003115202439613799`. The refined `_dx2` / `_dx4`
rungs live in the VESSL run directory rather than in the repo artifacts and are
taken as tabulated in `docs/design_notes/20260903_lattice_witness_standard.md`
§5.4, together with the per-rung lattice terms in the middle column.

At the un-quantised 8e-06 **all twelve fail**, and so would cv04's own 0.0066 by
a factor of 800. That is the same statement as "the envelope is entirely the
lattice term" and it is the reason the re-derivation is a mesh decision, not a
tolerance decision. **Nothing is changed here.** `W_MEAN_R` stays 0.010,
`W_MEAN_T` stays 0.017, `W_BIN` stays 0.074, cv04's own gates
(`T_err.mean() < 0.05`, `R_err.mean() < 0.05`, `cons.mean() < 0.05`,
`CONS_MAX_LIMIT = 0.06`) are untouched, and cv04's committed record stays 719.

**What this note adds to the decision, and it is a subtraction:** the follow-up
lane proposed in #886 §5.4 does **not** need to carry an auxiliary-echo term.
The two-artefact scenario the brief posed is ruled out. It needs to carry the
lattice term (per rung, per mesh) and the record-truncation term, and it can
treat the incident field as clean at every committed record in the family — see
§6.1 for why that last clause holds beyond cv04.

### 6.1 Every committed rung in the family is on the safe side of its own arrival

The whole slab family runs `slab_rig.run_slab_arm`, which is cv04's rig with the
material factored out, so the same arithmetic applies at every rung. Aux echo
arrival computed as in §2 (reflector 13.1 cells from the aux array's hi end,
`v_g(10 GHz)`), records from the committed artifacts:

| rung | K | grid `nx` | committed record | arrival (trans) | margin | record / arrival |
|---|---|---|---|---|---|---|
| cv04 `slab_eps4` | 1 | 641 | 719 | 1282 | 563 | 0.56 |
| cv22 `debye` | 1 | 1041 | 1108 | 2141 | 1033 | 0.52 |
| cv22 `lorentz` | 1 | 1041 | 1228 | 2141 | 913 | 0.57 |
| cv22 `drude` | 1 | 1041 | 1168 | 2141 | 973 | 0.55 |
| cv23 `tand0p1` | 1 | 1041 | 1067 | 2141 | 1074 | 0.50 |
| cv23 `tand1` | 1 | 1041 | 1158 | 2141 | 983 | 0.54 |
| cv23 `tand3` | 2 | 2081 | 2362 | 4209 | 1847 | 0.56 |
| cv23 `tand0p1_dx2` | 2 | 2081 | 2134 | 4209 | 2075 | 0.51 |
| cv23 `tand1_dx2` | 2 | 2081 | 2315 | 4209 | 1894 | 0.55 |
| cv23 `tand3_dx2` | 2 | 2081 | 2362 | 4209 | 1847 | 0.56 |
| cv23 `tand0p1_dx4` | 4 | 4161 | 4267 | 8350 | 4083 | 0.51 |
| cv23 `tand1_dx4` | 4 | 4161 | 4629 | 8350 | 3721 | 0.55 |
| cv23 `tand3_dx4` | 4 | 4161 | 4723 | 8350 | 3627 | 0.57 |

Records from `validation/crossval/_22_dispersive_results/lattice_witness.json::rungs.<rung>.n_steps`
(debye 1108, lorentz 1228, drude 1168) and
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.<rung>.n_steps`
(tand0p1 1067, tand1 1158, tand3 2362, tand0p1_dx2 2134, tand0p1_dx4 4267,
tand1_dx2 2315, tand1_dx4 4629, tand3_dx2 2362, tand3_dx4 4723).

**Every rung sits at 0.50–0.57 of its own echo arrival**, and the clustering is
structural, not luck. The rig derives its record from
`t_safe = 0.95 × 2·dist(probe → 3-D CPML)/v` measured from `t = 0`; the aux echo
must travel source → aux reflector → probe, which is roughly twice that path
because the direct leg is counted from the source rather than from the probe.
The record law therefore buys a factor of ~1.8 against the aux echo for free,
without ever naming it. That is a load-bearing coincidence and it is written
down nowhere: nothing in `derive_record_length` or in `t_safe_steps` mentions
the auxiliary grid at all.

Verified directly at the cv22/cv23 geometry (`nx_interior = 1000`, cv04's slab,
shipped vs echo-free aux):

```
N=1108   max|shipped-padded| time series = 0.000e+00 ; mean|dR| 0.00727  max|R+T-1| 0.00084
N=1600   max|shipped-padded|             = 0.000e+00 ; mean|dR| 0.00727  max|R+T-1| 0.00021
N=2000   max|shipped-padded|             = 0.000e+00 ; mean|dR| 0.00727  max|R+T-1| 0.00008
N=2400   max|shipped-padded|             = 3.223e-02 ; mean|dR| 0.05338  max|R+T-1| 0.25935
```

Bit-identical up to 2000 steps; the arrival is at 2141; at 2400 the case breaks
the same way cv26 te_45 does.

---

## 7. What this says about issue #888

The #888 note's §10 listed as unsettled: *"Whether cv04's committed
`per_bin_max_RT_closure = 0.0487` envelope is the same effect at normal
incidence, held down only by cv04's shorter records… Settled by: re-running cv04
with a record extended past its own absorber echo arrival and seeing whether the
closure grows to the same 0.2-0.3 scale."*

That is done, and it splits into two answers that must not be confused:

* **The closure DOES grow to the 0.2–0.3 scale** past the arrival — 0.258 at
  3000 steps, against #888's 0.2563–0.3082 on the oblique arms. So yes, it is
  the same effect, and yes, cv04 is held down only by its record.
* **The committed 0.0487 is NOT that effect.** It is 511 steps on the safe side
  of the arrival, and it is bit-identically unchanged when the echo is removed.
  The committed number is record truncation of the order-2 etalon echo, exactly
  as `04_multilayer_fresnel.py:329-337` says it is ("the error is ENTIRELY
  order-2 etalon-echo truncation — widening to nx=1500/1940 steps collapses it
  to 0.0002"). This note reproduces that too, without widening the box: at the
  same nx=600, going from 719 to 1200 steps collapses `max|R+T−1|` from 0.0487
  to 0.00037.

One thing here does **not** follow #888's diagnosis. #888 attributes the aux
reflection to `kappa_max = 7.0` putting σ_max 70× above optimum. cv04's 1-D aux
carries **no κ at all** and still reflects 4.40e-02 against cv26's 4.18e-02 at
the same incidence. The κ = 7 is a real mis-parameterisation — #888 §4's own
scan shows retuning it helps — but the *dominant* cause in both grids is
**depth at the shipped σ**, and fixing κ alone would leave cv04's absorber where
it is. #888's fix candidate 1 (deepen and re-derive σ from a reflection target)
covers both; fix candidate "retune the profile constants" does not.

---

## 8. Dead ends, recorded

* **Two-mode fitting the aux field over the full mapped span at the committed
  record.** Fit residual 1.03 — meaningless. At 719 steps the direct pulse has
  not yet cleared the far samples (it reaches aux index 620 at ~937 steps), so
  the steady-state two-mode ansatz does not hold there and the fit reports a
  spurious `|B/A| = 2.1e-02`. The fix is to restrict the sample window to
  positions the pulse clears inside the record (aux 40–280), which gives
  1.04e-05 with residual 1.2e-03. **A two-mode purity fit is only valid where
  the record contains the whole direct pulse**, and reporting one without
  checking the residual would have produced a false positive here — the exact
  wrong answer to this brief.
* **Retuning the 20-cell aux profile as the echo-free control.** Scanned
  (m ∈ {2,3,4}) × (κ_max ∈ {1,2,3,5,7}) × (σ_factor ∈ {0.02…0.8}). The shipped
  point is reproduced exactly (m=3, κ=1, σ_factor=0.8 → σ_max 8.494,
  `|B/A|` 4.4003e-02). The family floor at n = 20 is 2.36e-02 (m=3, κ=1,
  σ_factor=0.2) — 1.9×, not enough to serve as a control; and raising κ is
  sharply worse (κ=2 → 9.0e-02, κ=3 → 2.6e-01). Abandoned in favour of padding
  the array.
* **Separating the aux echo from the 3-D CPML echo by geometry.** Impossible in
  this rig at any `nx`: `n_1d = nx + 12` locks the aux absorber ~6 cells outside
  the 3-D one, and the two round trips differ by ~14 cells out of ~2240, so the
  two echoes always arrive within ~20 steps of each other. The
  `shipped − padded` difference is the only separator.
* **Reading the record sweep alone as the answer.** The sweep does step —
  0.0073 → 0.0149 → 0.0517 between 1200 and 1600 — and a sweep without the
  padded control would have been read as "the echo is in the sweep", which is
  true, and then as "so it may be in the envelope", which is false. The
  committed record is on the flat side of the step; the step is 511 steps away.
* **Assuming the 1-D aux inherits cv26's defect parameters.** It does not:
  different file, different order, no κ. The reflection amplitude had to be
  measured, and it came out the same anyway — but for a different reason
  (depth, not κ).

---

## 9. What is still uncertain

* **Float32.** The bit-identity in §4.3 is a float32 statement. The whole
  campaign runs float32 (`jnp.result_type(float)`, `tfsf.py:320`), so it is the
  right statement for the committed numbers, but a float64 build would show a
  nonzero difference at some level. It cannot be large — the echo has not
  physically arrived — but "0.000e+00" is a property of the arithmetic as well
  as of the physics. **Settled by:** re-running §4.3 under
  `JAX_ENABLE_X64=1`; ~30 s.
* **The reflector depth is measured at two geometries, not derived.** 6.88 cells
  inside the absorber's inner edge at `nx_interior = 600` (aux index 638.88,
  `n_1d = 652`, hi CPML 632..651) and 6.88 at `nx_interior = 1000` (aux index
  1038.88, `n_1d = 1052`, hi CPML 1032..1051). §6.1's arrival table
  uses 13.1 cells from the array end for all 13 rungs. The profile is invariant
  in normalised units (σ ∝ 1/dx, dt ∝ dx, so σ dt/ε₀ is fixed), so the depth
  should not move with K, and the two measurements agree — but the dx/2 and dx/4
  rungs were not measured directly. The margins in §6.1 are 1847–4083 steps, so
  even a 100-cell error in the reflector position would not change the verdict.
  **Settled by:** one `drive_aux` fit at K = 4; ~1 min.
* **Only the R/T observables were checked.** cv04 also gates fringe positions
  and values (`fringe_gate.compare_fringes`) and a tail witness. Those are
  computed from the same `R_rfx` array, which is bit-identical between shipped
  and echo-free, so they cannot differ — but they were not exercised, because
  the harness reproduces PART 1 + PART 2 and not PART 4.
* **Meep was not run.** The harness has no Meep leg, so this note says nothing
  about the secondary reference. It does not need to: the question is whether
  rfx's own committed numbers contain an rfx-internal artefact.
* **The 3-D CPML echo term is inferred, not isolated.** §5 records it as
  0.00000 on the strength of the 800–1200 plateau sitting exactly on the lattice
  term (`|rfx − lattice|` 7.6e-06). That is strong, but it is an absence-of-
  effect argument rather than a controlled difference, because the 3-D absorber
  cannot be padded the way the aux array can. **Settled by:** the
  `CPML_DEPTH_LADDER` experiment #888 §10 already proposes, run at cv04's rung.
* **The record law's factor-of-1.8 safety margin against the aux echo is
  undocumented and unowned.** §6.1 shows every rung inherits it, but nothing in
  `derive_record_length`, `t_safe_steps` or any gate refers to the auxiliary
  grid. A future change that lengthens records (a tighter settling bar, a
  higher-Q material, an adaptive extension) can cross the arrival with no
  witness firing — cv26 round 2 is precisely that failure. This is a rig gap,
  not an uncertainty in the present answer. #888's fix candidate 2 (make
  `predict_settling`'s `e_absorber` difference against a clean incident wave)
  is the same gap in the oblique lane; the slab family needs the normal-incidence
  version of it.

---

## 10. Reproduction

All local, `~/Documents/rfx/.venv/bin/python`, worktree
`~/Documents/rfx-worktrees/cv04-echo` at `origin/main` @ `0141f39e`. No VESSL
run was needed. Wall time: one 719-step cv04 run 6.3 s; the 24-run record sweep
~4 min; the aux drive 5-9 s per record; the profile scan ~40 s for eight points;
the cv22-geometry robustness check ~2 min.

* **§1.1, §3, §5** — cv04 PART 1 + PART 2 re-implemented verbatim (same `Grid`,
  same `init_tfsf`, same probes, same `2 %` amplitude mask, same
  `nfft = 2^ceil(log2 N) · 8`), with `n_steps` and `nx_interior` as free
  parameters and an optional `pad_aux`. References:
  `fresnel_slab_RT` (the case's own transfer matrix) and
  `comparators/dispersive_eps.py::yee_lattice_slab_rt(f, 4.0, 0.0, 0.010, dx, dt)`.
* **§2 (reflector), §4.1, §4.3 (purity)** — `init_tfsf` + `update_tfsf_1d_h/e`
  driven standalone with no 3-D domain; sample `e1d` at 64-96 x-positions; the
  case's own `np.fft.rfft`; two-mode least squares at
  `k = (2/dx) asin(ŵ dx/2c)`; `L = −½ · d(arg B/A)/dk`.
* **§2 (arrival), §4.2, §4.3, §6.1** — the echo-free control is
  `state._replace(e1d=concat([e1d, zeros(4000)]), h1d=concat([h1d, zeros(4000)]))`
  applied once after `init_tfsf`. Nothing else changes: `i0`, `src_idx`,
  `x_lo`, `x_hi` and every index `apply_tfsf_e/h` reads are at the lo end.
* **§4.2 (scan)** — the `tfsf.py:281-287` profile expression with
  (m, κ_max, σ_factor) as parameters, `_replace`d into `b_cpml`/`c_cpml` after
  `init_tfsf`; σ_factor = 0.8, κ = 1, m = 3 reproduces the shipped arrays.
* **§6.1 (arrival table)** — `rfx.grid.Grid` plus the index arithmetic of
  `comparators/slab_rig.py:64-95` and `rfx/sources/tfsf.py:253-271`, with
  `v_g(10 GHz)` from the 1-D lattice dispersion relation.
