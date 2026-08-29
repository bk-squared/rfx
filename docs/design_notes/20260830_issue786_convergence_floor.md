# Issue #786 — attributing the W4R non-convergence floor

**Lane**: `agent/issue-786-convergence-floor`. **Contract**: SPEC-00 (common) + issue #786.
**Discipline**: physical accuracy first; pre-declaration before measurement; append-only.

This note is **append-only**. Sections are numbered in the order they were written;
a correction is a NEW section, never an edit of an old one.

---

## §1 Premise verification (2026-08-30, before anything else)

Re-checked against the current tree, not assumed:

1. `git merge-base main agent/multiband-nu-envelope` = `bdcf9ea` = `main` HEAD.
   `git diff main agent/multiband-nu-envelope -- rfx/ tests/` touches only
   `rfx/api/_preflight.py` (new advisories), `rfx/api/__init__.py` (re-exports)
   and tests. **No solver code differs**, so this lane's `main`-based worktree
   runs the same solver that produced `w4r_supraconvergence.json`.
2. The symptom table in issue #786 is the `err_hz` column of that JSON. Reading the
   same JSON's `f_target` column (which the issue does not quote) shows the uniform
   arm's *frequency* sequence is **monotone increasing** as `s` falls —
   5.009 / 5.405 / 5.502 / 5.533 / 5.542 / 5.5436 GHz for s = 3.0 … 0.5 — and that the
   s = 0.25 reference, 5.5208 GHz, sits **below** the s = 0.6 and s = 0.5 rungs, i.e.
   off the monotone trend. The non-monotone *error* column is therefore consistent
   with either (a) a floor in the ladder or (b) a bad reference. This lane must not
   assume which; it is exactly what the discriminators are for.
3. `rfx/simulation.py::make_soft_source` — the source the W4R port uses — is
   **additive**: `E += Cb * waveform(t)`. Verified by reading the code, not assumed.
4. `rfx/api/_spec.py::Result.find_resonances` subsamples the probe record with
   `step = len(w) // 10000; w_sub = w[::step][:10000]` — an **un-antialiased stride**
   plus a truncation — before handing it to the matrix-pencil estimator, which then
   applies its *own* anti-aliased decimation. `step` is an integer floor of a
   quantity that varies with the ladder scale, so the analysed record is a
   scale-dependent object. This is a fact about the instrument, recorded here
   because it makes D4 a live candidate; it is **not** yet evidence of anything.

## §2 The fixture copy

`validation/research/convergence_floor/fixture.py` is a functional copy of PR #785's
`fixtures.py` + `w4r_port_supraconvergence.py::build_sim/measure`. The copy exists so
this lane is self-contained and rerunnable from `main`; PR #785's files are **not**
modified. The copy is *checked*, not asserted: D0 re-runs the ladder and compares
every rung with the committed PR #785 numbers to 1 kHz.

Sole additions: the raw probe record and `dt` are returned (so an independent
estimator can be run on the identical record without paying for the FDTD twice), and
`with_trace` / `src_amp` / source & probe positions are parameters defaulting to the
PR #785 values, so D2 and D3 each change exactly one thing.

## §3 Pre-declared discriminators — frozen BEFORE any measurement

Machine-readable copy: `validation/research/convergence_floor/results/predeclared_windows_786.json`
(committed in the same commit as this section, ahead of every measurement commit).

**Burned data.** The issue's numbers — 116.3 / 18.7 / 12.2 / 21.6 / 22.7 MHz,
`u_ref = 7.58 MHz`, "~4.1e-3" — are the **symptom**. No window below is a function of
them. Every window carries a `derivation` key naming its source class:
`arithmetic`, `first_principles`, `wedge_theory`, `prior_provenance`.

### D0 — reproduction
Re-run the uniform and multiband ladders plus the s = 0.25 reference with the copied
fixture. **PASS** iff every rung matches PR #785's `f_target` to ≤ 1 kHz (determinism:
identical code, identical CPU JAX). **FAIL ⇒ STOP** and report the irreproducibility
as the finding.

### D1 — geometry quantization (issue candidate 1)
*Instrument.* Assemble the materials and PEC mask that each rung actually compiles
(`assemble_materials_nu`), and read back, on the run's own node coordinates: the
realized PEC node span in x/y/z and the realized substrate / upper-dielectric
interface planes. `delta_max(s) = max |realized − declared|`.

*Window — derivation `arithmetic`.* Every declared feature coordinate
(6.75, 9.0, 13.5, 18.0, 20.25, 22.5, 27 mm in-plane; 0.75, 1.5, 3.0, 7.5, 9.0, 13.5 mm
in z) is an exact integer multiple of `dx(s) = 0.75·s` mm and `dz(s) = 0.25·s` mm for
every ladder scale s ∈ {1.5, 1.0, 0.75, 0.6, 0.5, 0.25}. The pre-declaration script
computes those residuals in float64: the largest over all rungs is **3.5e-18 m**.
The geometry is therefore *predicted* to realize exactly at every rung — which is
precisely the issue's "decisive" version of the discriminator (a ladder built only on
scales where the declaration realizes exactly), already satisfied by construction.

- **EXONERATE (1)**: `delta_max(s) < 1e-12 m` at every rung including the reference.
- **ATTRIBUTE (1)**: `delta_max(s) ≥ 0.25·dx(s)` at ≥ 1 rung **and** |Pearson ρ| ≥ 0.8
  between the per-rung residual and the realized-vs-declared electrical-length delta.
- **INCONCLUSIVE**: `0 < delta_max < 0.25·dx(s)`, or ρ below 0.8.

### D2 — edge singularity (issue candidate 2)
*Instrument.* The identical box, dielectric stack, port pair, probe, `T_TOTAL`,
subpixel setting and scale set, with **only** the PEC trace deleted. The surviving
in-band line is a dielectric-loaded box mode: no metal edge in the interior, fields
analytic away from the flat coordinate-aligned PEC walls.

*Window — derivation `wedge_theory`.* The W4R trace is a **1.5 mm-thick** PEC block,
so its edges are 90° conductor corners: field wedge angle 3π/2, Meixner exponent
ν = π/(2π − θ) = 2/3, and the leading resonant-frequency error term from such an edge
is O(h^{2ν}) = O(h^{4/3}). A wedge singularity therefore predicts a **reduced order,
never a non-vanishing floor**.

- **ATTRIBUTE partial to (2)**: fitted `p_trace ∈ [1.0, 1.6]` (consistent with 4/3)
  **and** `p_smooth ≥ 1.8` for the trace-free control — removing the edge restores the
  smooth-field order.
- **EXONERATE (2) as the FLOOR mechanism**: `f(s)` of the with-trace ladder is
  monotone in s and its error against the D4b reference decreases at every rung, with
  `p_trace ≥ 1.0`.
- **INCONCLUSIVE**: `p_trace < 1.0`, or the trace-free ladder is itself non-monotone.

### D3 — port / probe loading (issue candidate 3)
*First principles.* rfx soft sources are **additive** (`E += Cb·w(t)`, §1.3). An
additive forcing term in a linear time-invariant system leaves the system operator —
hence every eigenfrequency — exactly unchanged. Predicted coupling-induced Δf = **0**.

*Instrument.* At s = 0.75 and s = 0.5: (a) `src_amp` 1.0, (b) 0.01, (c) 100.0,
(d) source pair moved to (9.0, 11.25, 0.75)/(18.0, 11.25, 0.75) mm — same symmetry
class, different physical coupling to the mode, (e) probe moved to
(15.75, 11.25, 0.75) mm.

- **EXONERATE (3)**: max pairwise |Δf| over (a)–(e) ≤ **1.0 MHz**.
- **ATTRIBUTE (3)**: monotone dependence of f on coupling strength, span ≥ **10 MHz**.
- **INCONCLUSIVE**: between.

### D4 — reference quality (issue candidate 4)

**The 1 MHz / 10 MHz pair — derivation `first_principles`.** Probe records are
float32 (ε₃₂ = 2⁻²⁴ = 5.96e-8). The Cramér–Rao bound for the frequency of a single
sinusoid in white noise of relative amplitude σ, N samples spanning T, is
σ_f = √6·σ/(π·T·√N). With T = 20 ns, N = 700 (the post-decimation sample count the
incumbent estimator actually uses) and σ = ε₃₂ this is **0.088 Hz**. Inflating by 10³
for round-off accumulated over ~10⁵ steps and for estimator inefficiency gives ~1 MHz;
a further decade gives 10 MHz. Declared:
**≤ 1 MHz = sound**, **≥ 10 MHz = instrument-limited**, between = inconclusive.
(Context, *not* the window's source: the ledger's measured staircase-edge envelope for
patch resonance is −dx/L_eff ≈ 2.9e-2 at dx = 1 mm on a 32 mm patch; an instrument
error at 10 MHz/5.5 GHz = 1.8e-3 would already be ~6 % of the physics envelope this
fixture class is used to bound.)

**D4a — exact-reference instrument twin.** Empty **vacuum** PEC box,
Lx = Ly = 38.25 mm, Lz = 1.5 mm, `dx = 0.75·s` mm, `dz = 0.25·s` mm — so `dt`,
`n_steps`, the analysis band and the record length are **identical** to the W4R
uniform rung at the same s. Mode TM₁₁₀ (Ez, no z variation),
f_exact = (c₀/2)·√2/38.25 mm = 5.5421 GHz, inside the same BAND (4–6.5 GHz), and the
next mode (TM₂₁₀) is at 8.76 GHz, outside it. The **exact** discrete leapfrog
eigenfrequency at each rung is
`f_disc = arcsin((c₀·dt/2)·√(μ_x+μ_y))/(π·dt)` with μ from
`analytic_dispersion.operator_eigenvalues` on the uniform axis profiles — the same
difference operator rfx builds. This removes the discretization error *analytically*,
leaving `eps_instr(s) = |f_extract(s) − f_disc(s)|` = the extraction instrument's own
error at that rung.
Admissibility gate: the twin's in-band line must be single with dominance ≥ 10 at
every rung.

**D4b — independent reference by Richardson.** Model `f(h) = f∞ − C·h^p`,
`h = dz_fine(s) = 0.25·s` mm, fitted by nonlinear least squares to the five ladder
rungs s ∈ {1.5, 1.0, 0.75, 0.6, 0.5} — **the s = 0.25 rung is not in the fit**.
- **OUTLIER (attribute 4)**: |f(0.25) − f_pred(0.25)| ≥ 5 × RMS residual **and**
  sign(f(0.25) − f(0.5)) = −sign(f(0.5) − f(0.6)).
- **VINDICATED**: |f(0.25) − f_pred(0.25)| ≤ 3 × RMS residual.
- **INCONCLUSIVE**: between.

**D4c — independent estimators on the identical record.** Re-extract f from the
*same* stored probe record of every rung with three estimators that do not share
`find_resonances`'s un-antialiased `[::step]` subsampling:
E2 = FFT bandpass over BAND + Hilbert analytic-signal phase-slope fit;
E3 = `rfx.harminv.harminv` on the full ring-down with anti-aliased decimation;
E4 = 4-parameter damped-sinusoid nonlinear least squares on the bandpassed ring-down.
E1 = the incumbent `find_resonances`.
- E2/E3/E4 **agree** iff max pairwise spread ≤ 1 MHz; their mean is then the
  trustworthy value at that rung.
- **ATTRIBUTE (4a)**: |E1 − consensus| ≥ 10 MHz at a rung.
- **EXONERATE (4a)**: |E1 − consensus| ≤ 1 MHz at a rung.

### Apportionment (frozen)
`Delta_total = |f_E1(0.25) − f∞|`. `Delta_instr = |f_E1(0.25) − f_consensus(0.25)|`
is charged to (4a); `Delta_phys = |f_consensus(0.25) − f∞|` is the remaining
solver/physics part, apportioned between (1), (2), (3) by their own verdicts. Both
reported as percentages of `Delta_total`. **A mechanism whose own discriminator
exonerates it may not be charged any part.**

### Remedy licence (frozen)
- (4a) dominates (`Delta_instr ≥ 0.5·Delta_total`) ⇒ the remedy is in the extraction
  instrument: fix the record handling and re-run the ladder; **the floor must be
  demonstrated to vanish**, not asserted.
- (1) dominates ⇒ scale-consistent geometry realization / ladder design rule.
- (3) dominates ⇒ fix the observable.
- (2) dominates and the floor is irreducible ⇒ **no code fix**; produce the exact
  envelope statement (number + scope + what it bounds) and state that #715's baseline
  must be quoted against it.

### STOP conditions
D0 fails ⇒ STOP. All four discriminators inconclusive ⇒ STOP, no remedy.

## §4 Cost design (this host is shared with concurrent lanes)
Smallest decisive fixtures, each arm well under the 20-minute CPU cap:
D0 ≈ 11 min (dominated by the one s = 0.25 rung); D1 ≈ 2 min (no time stepping);
D2 ≈ 2 min; D3 ≈ 1 min; D4a ≈ 3.5 min (the twin's s = 0.25 rung costs ~1/4 of the
W4R one because the twin needs only 24 z-cells to carry a z-invariant mode while
keeping `dz` — hence `dt` and `n_steps` — identical); D4c is free (it re-reads
records D0 already stored). No GPU arm is required.

---

## §5 Results, part 1 — D0–D4 (appended 2026-08-30, after the runs)

Artifacts: `validation/research/convergence_floor/results/{d0_reproduction.json,
d0_records.npz, d1_geometry.json, d2_edge.json, d3_port.json,
d4_reference_a.json, d4_reference_b.json, d4_reference_c.json}`.

### D0 — REPRODUCED, bit-identically
All eleven rungs match PR #785's `f_target` to **0.0000 Hz**. The symptom is a
property of the code, not of a run.

| s | 1.5 | 1.0 | 0.75 | 0.6 | 0.5 | **0.25 (ref)** |
|---|---|---|---|---|---|---|
| f (GHz), uniform | 5.404546 | 5.502115 | 5.533066 | 5.542444 | 5.543558 | **5.520821** |
| \|f − f_ref\| (MHz) | 116.3 | 18.7 | 12.2 | 21.6 | 22.7 | — |

The error column is the issue's table. The **frequency** column, which the issue
does not quote, is monotone increasing as the cell shrinks, and the reference sits
**below both of the two finest rungs**. Both statements are facts; which of the two
is wrong is what D2–D5 decide.

### D1 — geometry quantization EXONERATED
Realized PEC extents are exactly the declared 13.5 × 4.5 × 1.5 mm at every rung:
**12/18/24/30/36/72** cells in x and **4/6/8/10/12/24** in y and z. Worst
realized-vs-declared deviation over all PEC faces and all three dielectric
interfaces, over all six rungs: **4.3e-10 … 8.7e-10 m = 7.8e-7 … 6.8e-6 CELLS**
(window: < 1e-3 cells).

*Base-window letter verdict: INCONCLUSIVE, reported unchanged.* The base window
was written as `< 1e-12 m` in absolute metres, which is below the float32
mesh-storage floor (node coordinates are a cumsum of a float32 mesh:
ε₃₂·27 mm = 1.6e-9 m) — a specification error, disclosed in the addendum rather
than widened. The measured 1e-9 m **is** that floor.

*D1c (smoothed material) fired by the letter* — max relative spread 0.247 at
`comp2/upper_lo` offset +3 — and the post-hoc diagnosis (letter verdict left
standing) is that every differing entry is at s = 1.5 alone, at an offset that
reaches the **neighbouring** interface: h_sub = h_upper = 1.5 mm = 4 cells at
dz = 0.375 mm, so ±3 from one interface is the other's halo. Restricted to
offsets within ±2 cells the spread is 7.0e-6; restricted to rungs s ≤ 1.0 at all
offsets it is 7.0e-6. The smoothed offset→eps maps
(2.65 / 4.236475 / 1.063525 / 1.023100 / 1.600000 / 2.176900) are otherwise
bit-identical at every rung: the realized material is **self-similar under
refinement**.

### D3 — port / probe loading EXONERATED
Predicted coupling-induced Δf = **exactly 0** (additive source). Measured span
over drive amplitude ×0.01 … ×100, a moved port pair and a moved probe:
**3.540 kHz** at s = 0.75 and **0.623 kHz** at s = 0.5 — 3e-4 of the 1 MHz window.

### D4a — the ladder machinery is SOUND at every rung, reference included
Exact-reference twin (empty vacuum PEC box, TM₁₁₀ at 5.5421 GHz; dx, dz, dt,
n_steps, band and record length identical to the W4R uniform rung at the same s):

| s | 1.5 | 1.0 | 0.75 | 0.6 | 0.5 | 0.25 |
|---|---|---|---|---|---|---|
| ε_instr (kHz) | 12.7 | 16.5 | 8.3 | 6.0 | 7.9 | **8.4** |
| e_disc (MHz) | −1.6203 | −0.7201 | −0.4051 | −0.2592 | −0.1800 | −0.0450 |

ε_instr — the extraction's own error with the discretization removed
**analytically** — is 6–17 kHz everywhere, 60–120× inside the 1 MHz window, and
does **not** grow at the reference scale. e_disc falls at exactly second order
(1.6203/0.7201 = 2.2500 = 1.5² to five figures); fitted p = **2.0001** analytic,
1.9707 measured. So the grid, the solver, the time stepping, the additive port,
the probe and Harminv together deliver a clean p = 2 sequence with **no floor** and
a 45 kHz total error at s = 0.25.

### D4c — the incumbent extraction is right, on the reference record itself
Three estimators sharing no code with `find_resonances` (Hilbert phase slope;
anti-aliased `harminv` on the full ring-down; damped-sinusoid NLS), run on the
**identical stored record**:

| rung | E1 | E2 | E3 | E4 | spread | E1 − consensus |
|---|---|---|---|---|---|---|
| UC s=0.25 | 5.520821 | 5.520824 | 5.520824 | 5.520608 | 215 kHz | **+0.069 MHz** |

EXONERATE-4a at ten of eleven rungs (the exception is MB s = 0.6, where E4's NLS
locks onto a different line, spread 8.5 MHz, so no consensus is available; the
uniform arm and the reference rung are unaffected). **The s = 0.25 record really
does contain 5.520821 GHz.** Mechanism (4a) — a noisy or broken extraction — is
dead.

### D4b — judged as a single power law, the reference rung is a 17.6σ outlier
`f(h) = f∞ − C·h^p` fitted to the five ladder rungs with s = 0.25 held out:
f∞ = **5.553393 GHz**, p = 2.716, RMS residual 1.782 MHz; the reference deviates
**31.423 MHz = 17.6 × RMS** and breaks the trend → OUTLIER by the frozen 5× rule
(multiband arm: 27.699 MHz = 16.5 × RMS, same verdict).

That verdict is correct **under its declared model**. Whether the model is the
right one is exactly what D5 was pre-declared to decide.

### D2 — the edge cannot make a floor; the smooth controls converge cleanly
D2-B (primary control = the D4a twin): p_smooth = **2.00** analytic / 1.97
measured, no floor. D2-A (same box/stack/ports, PEC trace deleted; the 4–6.5 GHz
band is empty without it, so the control tracks the 11.79 GHz line in the
pre-declared 10–13 GHz control band): f = 11.790357 / 11.772714 / 11.761799 /
11.754559 / 11.749481 GHz — monotone and smooth, approaching its limit from
**above**. Its own 3-parameter fit converged poorly (p = 4.26, RMS 14.6 MHz) on
five points of a decreasing sequence, so no order is claimed from D2-A; reported
and not judged, its successive-triple ratios put its order near 0.4–0.5, i.e. the
trace-free dielectric stack at 11.8 GHz is itself low-order. Letter verdict:
**EXONERATED as the FLOOR mechanism** (p_trace = 2.61, f monotone, error
decreasing at every rung). The theory behind that window stands on its own: a
Meixner wedge exponent reduces the **order**; it cannot produce a non-vanishing
floor.

---

## §6 Results, part 2 — D5, D6, and the verdict (appended 2026-08-30)

Artifacts: `results/{d5_turnover.json, d6_two_term.json, verdict.json}`.

### D5 — TURN-OVER CONFIRMED: the ladder is pre-asymptotic
Three lattice-valid scales the ladder had skipped (3/s = 7, 8, 9 — the same
exact-realization arithmetic D1 verified), inserted between s = 0.5 and s = 0.25:

| 3/s | 2 | 3 | 4 | 5 | 6 | **7** | **8** | **9** | 12 |
|---|---|---|---|---|---|---|---|---|---|
| s | 1.5 | 1.0 | 0.75 | 0.6 | 0.5 | 0.428571 | 0.375 | 0.333333 | 0.25 |
| f (GHz) | 5.404546 | 5.502115 | 5.533066 | 5.542444 | **5.543558** | 5.541260 | 5.537579 | 5.533355 | 5.520821 |
| Δf (MHz) | | +97.57 | +30.95 | +9.38 | +1.11 | −2.30 | −3.68 | −4.22 | −12.53 |

**Exactly one sign change; the descending branch holds four rungs; the maximum is
at s = 0.5 (dz_fine = 0.125 mm).** Under the frozen structural window this is
TURN-OVER CONFIRMED: the s = 0.25 reference is **on-curve**, and the five-rung
ladder PR #785 fitted lay entirely on the ascending branch of a curve with an
interior maximum.

That settles the question D0 left open. f(h) is not converging monotonically and
then hitting a floor; it rises, peaks, and descends. `|f(s) − f(0.25)|` is
therefore not an error sequence at all — the "floor" is the arithmetic of
subtracting an anchor that lies four rungs down the far branch.

### D6 — the low-order term is real but its exponent is NOT identifiable
| model | f∞ (GHz) | RMS residual |
|---|---|---|
| M0 `f∞ − C h^p` (p = 3.83 fitted, 9 rungs) | 5.537648 | 7.687 MHz |
| M1 `f∞ + A h^{4/3} − B h²` (exponents fixed by theory) | 5.509492 | 4.361 MHz |
| free `f∞ + A h^a − B h^b` | 5.376805 | 0.231 MHz, **degenerate** (a = 0.872, b = 0.895) |

Letter verdict: **INCONCLUSIVE** — RMS_M1 = 4.36 MHz misses the 1 MHz window and is
only 1.76× better than M0 (3× required). The free fit reaches 0.23 MHz only by
collapsing its two exponents onto each other (a ≈ b, amplitudes +4.9/−5.7 that
cancel), which is a numerical derivative, not two physical terms; its f∞ is an
extrapolation artefact.

Reported and judged by nothing, with the bulk exponent held at 2 (the value D4a
*measured*, 2.0001):

| low-order exponent a | 0.5 | 2/3 | 1.0 | **4/3 (Meixner)** |
|---|---|---|---|---|
| RMS (MHz) | 2.642 | 3.006 | 3.706 | 4.361 |
| f∞ (GHz) | 5.440581 | 5.468147 | 5.495723 | 5.509492 |

The data mildly prefer a *lower* exponent than Meixner's 4/3, and no member of the
family reaches the instrument's accuracy class. **The Meixner edge exponent is not
confirmed**, and which feature supplies the low-order term — the trace's 3π/2
conductor wedges, the rasterized dielectric stack, or the subpixel-smoothing error
at the interfaces — is left open (see §8).

### Verdict on the four candidates
| candidate | verdict | discriminating number |
|---|---|---|
| **(1) geometry quantization** | **EXONERATED** | realized-vs-declared = **6.8e-6 cells** worst over 6 rungs, window 1e-3 cells; realized PEC extents are exact integer cell counts 12/18/24/30/36/72 × 4/6/8/10/12/24 |
| **(2) edge singularity** | **EXONERATED as the floor mechanism**; not confirmed as the low-order term | smooth controls give p = **2.0001** (exact-reference twin) with a 45 kHz error at s = 0.25; the theory-fixed 4/3 model fits at 4.36 MHz RMS, worse than a = 0.5 (2.64 MHz) |
| **(3) port / probe loading** | **EXONERATED** | span over ×0.01…×100 drive, moved ports, moved probe = **3.540 kHz** (s = 0.75) and **0.623 kHz** (s = 0.5), against a predicted exactly-zero |
| **(4) reference quality** | **ATTRIBUTED — and it is the whole effect**, but for a different reason than the issue supposed | the extraction is sound (ε_instr = **8.4 kHz** at s = 0.25; three independent estimators agree with it to **0.069 MHz** on the same record). The anchor is invalid because it lies **past the maximum** of a non-monotone error curve — D5 |

**Apportionment** (rule frozen before measurement; Δ_total measured against D4b's
independent reference f∞ = 5.553393 GHz):
Δ_total = 32.572 MHz; Δ_instr = **0.069 MHz = 0.21 %** charged to (4a);
Δ_phys = 32.641 MHz = **99.79 %**, of which (1) and (3) may be charged **nothing**
(their own discriminators exonerate them at 6.8e-6 cells and 3.5 kHz), leaving all
of it to the fixture's own two-term discretization error. `remedy_licensed_4a`
= **false**.

### Remedy — NONE licensed, and none implemented
Against the frozen licence table: (4a) does not dominate (0.21 %); (1) and (3) are
exonerated; (2) is exonerated as the floor mechanism. **No branch of the table
licenses a code change, and this lane makes none — `rfx/` is untouched.**

The observable-side remedy the table would have licensed for (4) — replace the
self-referential anchor with a model-based reference and re-run — *was attempted*
(D4b, D6) and **does not restore a convergence claim**: the fixture's error curve
turns over inside the ladder, so no single order exists to demonstrate. Saying so
is the finding.

What is left behind instead: the rerunnable harness (§7) and
`ladder_guard.py`, which states the precondition in code. It is **not** a fix for
any mechanism and touches no rfx code; it is the design rule D5 licenses. Its
self-check fires on PR #785's own ladder via P5 — *the anchor sits a factor 2.0
below the finest fit point with no rung in between, while the ladder's own largest
internal step is 1.5, so the entire turn hid in an interval the ladder never
sampled* — and on the extended ladder via P1/P2.

## §7 The accuracy envelope for this fixture class (the #715 deliverable)

**Scope.** A 13.5 × 4.5 × 1.5 mm PEC block on a 1.5 mm ε_r = 4.3 substrate with a
1.5 mm ε_r = 2.2 upper layer, inside a 27 × 22.5 × 13.5 mm PEC box; **uniform**
mesh, `subpixel_smoothing=True`, an additive mirror-selective Ez port pair and a
Harminv ring-down observable; the x-odd / y-even half-wave line near 5.5 GHz.
Resolved range: dz_fine = 0.25 → 0.0625 mm, dx = 0.75 → 0.1875 mm, i.e. **18 → 72
cells across the 13.5 mm trace**, 58,320 → 3,732,480 cells.

> **ENVELOPE (measured, model-free).** Over that range the extracted absolute
> resonance is **non-monotone**, with a maximum at dz_fine = 0.125 mm, and spans
> **5.502115 … 5.543558 GHz = 41.44 MHz = 7.49e-3 relative**. Refinement-induced
> variation alone therefore bounds any absolute-frequency claim on this fixture
> class at **≥ 7.5e-3** over the resolved range. (Over the narrower
> s ∈ [0.5, 0.25] window the issue quotes, it is 22.74 MHz = 4.1e-3 — the issue's
> number, which is a *lower* bound because it samples only part of the curve.)

> **ENVELOPE (continuum limit, model-dependent).** The limit is **not determined**
> by this ladder. Admissible extrapolations give f∞ = 5.5534 (single power law on
> the ascending branch), 5.5376 (single power law on all nine rungs), 5.5095
> (h^{4/3} + h²), 5.4957 (h¹ + h²), 5.4681 (h^{2/3} + h²), 5.4406 (h^{1/2} + h²) —
> a spread of **97.1 MHz = 1.8e-2** over the nine-rung model set (112.8 MHz =
> 2.0e-2 including the five-rung fit). The finest rung this lane could afford
> (3.7 M cells, 710 s CPU) is somewhere between **11 MHz above** and **17 MHz
> below** the limit depending on the model, i.e. its own absolute accuracy is
> **2e-3 … 3e-3 at best and 2e-2 at worst**.

> **WHAT IT BOUNDS.** #715's patch cross-validation baseline, and every absolute
> resonance quoted on a rasterized microstrip-class fixture in this class, must be
> stated against **≥ 7.5e-3** (measured refinement variation), not against the
> finest rung, and must not claim a continuum limit better than **~2e-2** without
> an external reference. The ledger's independently measured staircase envelope
> for the tutorial patch — f_res bias ≈ −dx/L_eff, which is −1.4 % to −5.6 % over
> this lane's dx range on a 13.5 mm resonant length — is the same order, so the
> two agree that this fixture class sits in the 10⁻²  accuracy band and not the
> 10⁻³ one.

**What is NOT bounded by this.** The solver machinery: on a smooth fixture in the
same box, with the same dt, record length, band, port and extraction, the error is
45 kHz at the reference resolution and falls at exactly p = 2.0001 (D4a). This
envelope is a statement about **rasterized geometry with a conductor edge and a
layered dielectric**, not about rfx's Yee core.

## §8 Left open (named, with a design a future lane can pick up)

1. **Which feature supplies the low-order term?** Three candidates remain: the
   trace's 3π/2 conductor wedges, the rasterized dielectric stack, the
   subpixel-smoothing error at the ε interfaces. The evidence so far: the empty
   box (no dielectric, no edge) is exactly p = 2; the trace-free P-C box
   (dielectric stack, no edge) has successive-triple ratios near order 0.4–0.5 at
   11.79 GHz; the with-trace fixture prefers a ≈ 0.5 over a = 4/3.
   **Pre-declarable discriminator (D7)** for that lane: keep the box, stack,
   ports, probe, T and scales, and draw the trace **full-width in y**
   (0 → 22.5 mm, touching both side walls). That removes the two 13.5 mm free
   edges and keeps the two 4.5 mm ones — 75 % of the singular edge length — while
   keeping the same stack and the same half-wave-in-x class. Window derived from
   the edge-length ratio: fit `f∞ + A h^a − B h²` to both ladders and compare the
   edge-term amplitude at a fixed h; **attribute to the edge iff
   |A_full-width| / |A_W4R| ≤ 0.5**, **exonerate the edge iff ≥ 0.9**. This lane
   did not run it: it needs its own bring-up (the full-width trace shorts to the
   side walls and moves the line), and the four candidates the issue asked about
   were already decided without it.
2. **A fixture in this class with an external reference.** Everything above is
   still self-referential in the weak sense that no independent solver or
   measurement pins f∞. Until one does, the 1.8e-2 model spread stands.
3. **D6's model class.** Nine rungs over a 6× span with a turning point inside do
   not identify two exponents. A lane that wants the exponent needs either a much
   wider h range (a GPU arm) or a fixture whose turn-over is outside the range.

---

## §9 The harness (rerunnable; how a future lane extends it)

`validation/research/convergence_floor/`. Nothing under `rfx/` is touched by this
lane; the only file outside the package is
`tests/_example_fidelity_lib.py` + its snapshot, where the #737
enumerate-and-classify gate requires an entry for every new script (three new
audited variants; the regeneration was pure addition, 878 lines, zero deletions,
so no existing example drifted).

```
PYTHONPATH=<worktree> .venv/bin/python -m validation.research.convergence_floor.<mod>

predeclare              windows, frozen (run first; every harness reads them)
predeclare_addendum     the D1b/D1c/D2-band addendum
fixture                 the W4R P-C fixture (library; `-m` prints the ladder table)
d0_reproduce            the ladder + reference; stores every raw record  (~13 min)
   ... --rejudge        re-score an existing run against PR #785 (no FDTD)
d1_geometry             realized-vs-declared, no time stepping           (~2 min)
d2_edge                 trace-free control ladder                        (~2 min)
d3_port                 coupling / probe sweep                           (~4 min)
d4_reference --part a   exact-reference empty-box twin                   (~4 min)
             --part b   Richardson reference from d0 (no FDTD)
             --part c   independent estimators on the stored records (no FDTD)
d5_predeclare_and_run   the three skipped lattice-valid scales           (~9 min)
   ... --with-0.3       one more descending rung (3/s = 10)
d6_two_term_model       model comparison + exponent grid (no FDTD)
verdict                 applies the frozen windows, apportions, envelope
ladder_guard            the precondition, with a self-check on the W4R ladder
```

Every window lives in `results/predeclared_windows_786*.json` and every harness
reads it from there, so a window cannot be widened by editing a script. `d5` and
`d6` refuse to run if the on-disk window file disagrees with their code.

To extend: add rungs by dropping lattice-valid scales (3/s and 6/s both integer)
into `d5`'s scale list — it caches, so previously-measured rungs are not re-run —
and re-run `d6` and `verdict`. §8's D7 is the next discriminator, with its window
already derived.

---

## §10 CORRECTION — narrowing the claims to what was measured (appended 2026-08-30, after independent adversarial review)

This section is a **correction**, appended per the append-only rule; §§1–9 are left
exactly as they were written. Where a sentence below contradicts an earlier one, this
section governs.

**What the review found.** The review reproduced this lane's measurements rather than
disputing them: it re-ran the three decisive D5 rungs in its own harness and matched them
to 0.0000 Hz, matched D0 against PR #785's committed JSON, re-fitted D6 to every digit,
verified the exact-realization arithmetic in exact rational arithmetic with no rfx code
involved, and confirmed `ladder_guard` fires on PR #785's ladder. **The measurements
stand. The conclusions drawn from them were wider than the evidence**, and that is what
this section fixes. Nothing below widens a window; every change narrows a claim.

### §10.1 WITHDRAWN: "there is no floor"

Commit `d5edc34`'s subject line ("the floor is not a floor") and the summary wording it
carried are **withdrawn**.

*Why.* This lane never measured a floor's **absence**. It has no external reference —
§8.2 of this note already concedes that "no independent solver or measurement pins f∞" —
and a consistency floor is fully compatible with every number here: **f(h) can turn over
and still converge to some f_floor ≠ f_physical**. A turn-over is a statement about the
shape of f(h) over the sampled range; it is not a statement about the limit.

*Worse, the authoritative ledger records a MEASURED floor for this very fixture class.*
`rfx-known-issues.md` (issue-#80 resolution research, 2026-05-30): **DIELECTRIC-interface
staircasing, ~49 % of the error, ε_r-coupled**, which "drops FDTD from 2nd→1st order
(Farjadpour/Johnson SPIE 6322, 2006), giving a **non-vanishing FLOOR for thin high-ε_r
layers** (Boriskin arXiv:1001.1733)". The W4R fixture is exactly that: a 1.5 mm ε_r = 4.3
substrate and a 1.5 mm ε_r = 2.2 upper layer, rasterized. **That entry is the standing
evidence that a real floor may exist here**, and nothing this lane measured contradicts
it.

*What replaces it — the narrower claim the data do support:*

> PR #785's `|f(s) − f(0.25)|` was **not an error sequence**, because the anchor sits four
> rungs past a turn-over in f(h) (D5: exactly one sign change, maximum at
> dz_fine = 0.125 mm, four descending rungs, s = 0.25 on-curve). **The 22.7 MHz figure is
> therefore not an error**, and the "~4e-3 floor" as the issue states it is not
> established.
>
> **Whether f(h) converges to the physical answer is NOT determined by this lane.**

That is still a real result — it invalidates the reading that produced the issue — but it
is a claim about a *ladder*, not about a *limit*.

*Scope note on the two surviving "no floor" phrases in §5.* "p = 2 sequence with no
floor" (§5, D4a) and "no floor" (§5, D2-B) are about the **empty vacuum twin**, which
carries an **exact** discrete eigenfrequency at every rung, so its error is measured
against a reference and its convergence to zero is observed, not inferred. Those two
statements are in scope as written. They say nothing about the P-C fixture.

### §10.2 WITHDRAWN: the ledger cross-check in §7's "WHAT IT BOUNDS" — and the discrepancy, as its own finding

§7's closing sentence used the ledger's staircase law to corroborate the envelope
("−1.4 % to −5.6 % over this lane's dx range … is the same order, so the two agree that
this fixture class sits in the 10⁻² band"). **That cross-check is withdrawn**: this lane's
own data refute it.

| | |
|---|---|
| ledger law (rfx-known-issues.md, 2026-07-12 patch-resonance envelope) | f_res bias ≈ −dx/L_eff |
| applied here, L_eff = 13.5 mm, dx = 0.75 → 0.1875 mm | bias −5.56 % → −1.39 % |
| ⇒ predicted **change** across this lane's dx range | **4.17 % = 230.0 MHz** |
| this lane's **measured** refinement variation over that same range | **41.44 MHz** |
| **over-prediction factor** | **5.55×** |

A law that over-predicts the measured variation by 5.6× does not corroborate the
envelope; it **disagrees with it**. The correct handling is to state the disagreement as
a finding in its own right and to stop using it to round the deliverable outward:

> **FINDING (new, unattributed).** A committed ledger law — the measured
> patch-resonance staircase envelope f_res bias ≈ −dx/L_eff — over-predicts this
> fixture's measured refinement variation by **~5.6×** (230 MHz predicted change vs
> 41.44 MHz measured, over dx = 0.75 → 0.1875 mm on a 13.5 mm resonant length).
> Candidate explanations, none tested here: the law was measured on an **open** patch
> radiator and this fixture is an **enclosed** 3-layer line; L_eff for a half-wave line
> under a 4.5 mm air gap and an ε_r = 2.2 superstrate is not the patch's L_eff; the
> ladder is pre-asymptotic so the measured variation is not the asymptotic bias. This
> deserves its own issue.

The envelope's numbers themselves are unchanged — they were always measured, never
derived from the ledger — and the band is **not** widened on the strength of a refuted
cross-check.

### §10.3 RE-TAKEN: D2's letter verdict on the nine-rung ladder is INCONCLUSIVE

§5's D2 paragraph and the §6 verdict table record D2 as "EXONERATED as the FLOOR
mechanism". That verdict was evaluated on the **five pre-D5 scales** only. On the
nine-rung ladder the lane ended up with, the frozen rule's exonerate branch fails in
both of its clauses. The rule is **not rewritten**; it is re-applied
(`d2_retake.py` → `results/d2_edge_retake.json`, no new FDTD; `d2_edge.json` is left
unedited as the five-rung record it is).

Frozen rule (`predeclared_windows_786.json` :: `D2_edge_singularity`): *EXONERATE as the
floor mechanism iff f(s) of the with-trace ladder is MONOTONE in s **and** its error
against the D4b reference DECREASES at every rung **and** p_trace ≥ 1.0.*

| 3/s | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| error vs D4b f∞ (MHz) | 148.85 | 51.28 | 20.33 | 10.95 | **9.83** | 12.13 | 15.81 | 20.04 | 32.57 |

| clause | five rungs | **nine rungs** |
|---|---|---|
| f(s) monotone | true | **false** (D5: one sign change) |
| error decreasing at every rung | true | **false** (rises at 3/s = 7, 8, 9, 12) |
| p_trace (log-log vs f∞) | 2.615 | **0.9495** |

`p_trace < 1.0` is the frozen rule's **own** INCONCLUSIVE clause. Neither branch is
satisfied:

> **D2 (re-taken, nine-rung ladder, same frozen rule): INCONCLUSIVE.** Candidate (2) is
> neither exonerated nor attributed by this lane.

*What survives, correctly labelled.* The wedge reasoning is an **ARGUMENT from theory,
not a measured exoneration**: the 1.5 mm-thick PEC trace presents 3π/2 conductor wedges,
Meixner exponent ν = 2/3, leading frequency-error term O(h^{4/3}) — a **reduced order**,
which by itself cannot produce a non-vanishing floor. That argument is derived in the
pre-declaration and is unaffected by any measurement here; it is also **not** a
measurement, and it does not rule out the *other* mechanism the ledger names (ε_r-coupled
dielectric-interface staircasing, which the ledger reports **does** produce a floor).
The smooth-field control stays **as measured**: p = 2.0001 analytic / 1.9707 measured on
the exact-reference vacuum twin. D6 independently declined to confirm the Meixner
exponent (RMS 4.36 MHz against a 1 MHz window; a = 0.5 fits better at 2.64 MHz).

### §10.4 The instrument check the D5 rungs shipped without

D4c ran on D0's eleven records only. The three rungs D5 **added** — 3/s = 7, 8, 9 — carry
**three of the four descending steps** the turn-over verdict rests on, and had no
instrument check at all. `d5_instrument_check.py` (committed before it ran) applies the
**unchanged** frozen D4c pair to them; the three rungs were re-solved to obtain records,
which also re-checks determinism.

| 3/s | s | E1 (GHz) | E2 | E3 | E4 | spread | E1 − consensus | E1 − committed D5 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 0.428571 | 5.541260 | 5.541330 | 5.541291 | 5.541421 | 129.8 kHz | −0.088 MHz | **0.0 Hz** | EXONERATE-4a |
| 8 | 0.375000 | 5.537579 | 5.537629 | 5.537577 | 5.537631 | 53.9 kHz | −0.033 MHz | **0.0 Hz** | EXONERATE-4a |
| 9 | 0.333333 | 5.533355 | 5.533405 | 5.533344 | 5.533293 | 112.1 kHz | +0.008 MHz | **0.0 Hz** | EXONERATE-4a |

**EXONERATE-4a at all three**, with `max |E1 − consensus| = 87.8 kHz` against the frozen
1 MHz exoneration window (11× margin) and all three estimator spreads inside the frozen
1 MHz consensus gate (7.7× margin at the worst rung). The re-solve reproduced
`d5_turnover.json`'s `f_target` **bit-identically (0.0 Hz)** at all three rungs, so the
turn-over verdict rests on rungs whose extraction has now been checked by three
estimators that share no code with the incumbent.

*Scope.* This bounds the **extraction's** disagreement on the identical record at those
rungs. It says nothing about the discretization error there, and it is not a claim about
the continuum limit. Artifacts: `results/d5_instrument_check.json`, `results/d5_records.npz`.

### §10.5 Instrument honesty: which number belongs to which object

The headline "ε_instr = **8.4 kHz**" is the **empty-vacuum-box twin's** extraction error
against an exact discrete eigenfrequency. It is a statement about the extraction
machinery on a clean single-line record, **not** about the W4R fixture. On the fixture's
**own** records the three independent estimators spread by:

| | reference rung (s = 0.25) | uniform arm | all D0 records with a consensus | D5 rungs (3/s = 7,8,9) |
|---|---|---|---|---|
| E2/E3/E4 spread | **215.5 kHz** | 81.5–215.5 kHz | 72.5–229.7 kHz | 53.9–129.8 kHz |

(The one record with no consensus, MB s = 0.6, spreads 8.5 MHz because E4's NLS locks
onto a different line; it is excluded from the ranges above and was already reported as
CONSENSUS-UNAVAILABLE.)

**Rule adopted for this lane's prose:** quote 8.4 kHz only for the twin; **wherever the
fixture's own extraction uncertainty is meant, quote 215.5 kHz at the reference rung and
72–230 kHz across records** — a factor 9–26 larger. This does not change any verdict:
the D4c window judges `|E1 − consensus|` (0.069 MHz at the reference rung), not the
spread, and the consensus gate (spread ≤ 1 MHz) passes with 4.6× margin. It does change
what may be claimed about the fixture's own precision, which is ~2e-4 relative, not
~1.5e-6.

### §10.6 Candidate (1): the prose now matches the machine verdict

§5's heading and §6's table read "(1) geometry quantization — **EXONERATED**", flatly,
while `verdict.json` records `D1b=EXONERATED D1c=ATTRIBUTED-CANDIDATE` and the **base**
window's letter verdict is **INCONCLUSIVE**. The body of §5 disclosed all three; the
headline did not. Corrected statement:

> **(1) geometry quantization — EXONERATED by the D1b addendum window** (worst
> realized-vs-declared 6.8e-6 cells over six rungs, window 1e-3 cells; realized PEC
> extents are exact integer cell counts 12/18/24/30/36/72 × 4/6/8/10/12/24).
> The **base** window (Δ < 1e-12 m) sits below the float32 mesh-storage floor
> (ε₃₂·27 mm = 1.6e-9 m); its letter verdict is **INCONCLUSIVE** and is reported
> unchanged rather than widened — the disclosed specification error. **D1c fired by the
> letter** and is recorded as **ATTRIBUTED-CANDIDATE**, with the post-hoc diagnosis (the
> only differing entries are at s = 1.5, at an offset that reaches the neighbouring
> interface) left standing as a diagnosis, not as a re-judgement.

### §10.7 The D4a window's stated derivation does not produce its declared number

`predeclare.py:44–53` computes **CRB = 0.0878 Hz**; the rationale string frozen into the
window file says "CRB is ~1.5 Hz", and claims the 1 MHz band comes from "inflating by
1e3". Both are wrong as arithmetic:

| quantity | value |
|---|---|
| CRB (√6/(π·T)·ε₃₂/√N, T = 20 ns, N = 700) | **0.0878 Hz** |
| 1e3 × CRB | 87.8 Hz |
| 1e3 × √(n_steps) × CRB, n_steps ~ 1e5 | **27.8 kHz** |
| declared SOUND band | **1 MHz = 1.1e7 × CRB** |

So the 1 MHz / 10 MHz pair is **not** a CRB-derived number: it is a declared instrument
band far above any noise-floor bound, and the CRB establishes only that float32 record
noise is not what limits the extraction. The frozen window file is **not edited** (it is
frozen, and `predeclare.py` still reproduces it byte for byte; the correction is a
comment in that file plus this section).

The declared band is **generous**, which is the direction that could flatter a verdict —
so the honest check is whether D4a survives the *tighter* derivable band. It does:
ε_instr = 6.0 / 7.9 / 8.3 / 8.4 / 12.7 / 16.5 kHz at s = 0.6 / 0.5 / 0.75 / 0.25 / 1.5 /
1.0, **every rung below 27.8 kHz**, by 1.7–4.6×. D4a's "SOUND at every rung" stands under
either band.

### §10.8 The continuum-limit envelope, made internally consistent

§7's second envelope block listed six admissible extrapolations and then said the finest
rung is "between **11 MHz above** and **17 MHz below**" the limit. Those two statements
are inconsistent: the range excludes four of the six models it just listed, including the
**best-fitting** one. Corrected, over the admissible nine-rung set (the free-exponent fit
stays excluded, and named as excluded, because its exponents collapse onto each other —
a = 0.872, b = 0.895, amplitudes +4.9/−5.7 cancelling; the five-rung fit stays excluded
because it was fitted on the ascending branch alone):

| admissible model (9 rungs) | f∞ (GHz) | RMS (MHz) | finest rung − f∞ (MHz) |
|---|---|---|---|
| f∞ − C h^p (p = 3.83 fitted) | 5.537648 | 7.69 | **−16.83** |
| f∞ + A h^{4/3} − B h² | 5.509492 | 4.36 | +11.33 |
| f∞ + A h¹ − B h² | 5.495723 | 3.71 | +25.10 |
| f∞ + A h^{2/3} − B h² | 5.468147 | 3.01 | +52.67 |
| **f∞ + A h^{1/2} − B h² (best RMS)** | 5.440581 | **2.64** | **+80.24** |

> **ENVELOPE (continuum limit, model-dependent) — corrected.** The limit is **not
> determined** by this ladder. Over the admissible nine-rung model set f∞ spans
> **97.07 MHz = 1.76e-2** (112.8 MHz = 2.04e-2 if the five-rung ascending-branch fit is
> included for comparison), and the finest rung this lane could afford lies between
> **16.8 MHz below** and **80.2 MHz above** the extrapolated limit — its own absolute
> accuracy is **2.1e-3 at best and 1.5e-2 at worst**. The best-fitting admissible model
> (h^{1/2} + h², RMS 2.64 MHz) is the one that puts the finest rung 80 MHz **above** the
> limit. The earlier "11 MHz above / 17 MHz below" range is **withdrawn**.

The **measured, model-free** envelope in §7 is unchanged and remains the number to quote:
5.502115 … 5.543558 GHz = **41.44 MHz = 7.49e-3** over dz_fine = 0.25 → 0.0625 mm.

### §10.9 The apportionment percentages do not partition — and why

`verdict.json` reported instr 0.2118 % and phys 100.2118 %, which sum to 100.42 %. That
is not a rounding error: the two terms are **magnitudes**, and the instrument step points
*against* the total. f∞ = 5553.3927, E1(0.25) = 5520.8208, consensus(0.25) = 5520.7518
MHz, so the consensus sits 0.069 MHz **further** from f∞ than E1 does. The signed
decomposition is exact:

    Δ_total = (f∞ − consensus) + (consensus − E1)
            = +32.6409 MHz    + (−0.0690 MHz)   = 32.5719 MHz
            = +100.2118 %     + (−0.2118 %)     = 100.0000 %

Both forms are now emitted (`apportionment.signed_decomposition`). The magnitude of the
(4a) share is 0.21 % either way, and the remedy-licence branch ("(4a) dominates iff
Δ_instr ≥ 0.5·Δ_total") is unaffected: `remedy_licensed_4a = false`.

### §10.10 What this lane established, stated once, at the width the evidence supports

1. **The symptom reproduces bit-identically** — 0.0000 Hz over all eleven rungs against
   PR #785's committed JSON. It is a property of the code, not of a run. (D0)
2. **f(h) turns over inside the ladder** — exactly one sign change, maximum at
   dz_fine = 0.125 mm, four descending rungs, s = 0.25 on-curve. PR #785's
   `|f(s) − f(0.25)|` was therefore never an error sequence, and **the 22.7 MHz "floor"
   figure is not an error**. (D5, and independently reproduced by the reviewer.)
3. **Three exonerations at 3–5 orders of margin.** Geometry quantization: 6.8e-6 cells
   against a 1e-3-cell window (D1b). Port/probe loading: 3.5 kHz and 0.6 kHz against a
   first-principles predicted **exact zero** and a 1 MHz window (D3). The extraction on
   the reference record: |E1 − consensus| = 0.069 MHz against a 1 MHz window, three
   estimators sharing no code with the incumbent (D4c) — now also checked on the three
   decisive D5 rungs (§10.4).
4. **A smooth-field control at p = 2.0001.** On an exact-reference vacuum twin with the
   same dt, band, record length and extraction, the error falls at exactly second order
   to 45 kHz at the reference scale. The Yee core, the additive port and the extraction
   are **not** what limits this fixture. (D4a / D2-B)
5. **A ladder-reading precondition that catches the original misreading.**
   `ladder_guard.py` fires on PR #785's own five-rung ladder via P5 — the anchor sits a
   factor 2.0 below the finest fit point with no rung between, while the ladder's largest
   internal step is 1.5, so the whole turn hid in an interval never sampled. No rfx code
   is touched by this lane.

And, equally, what it did **not** establish:

6. **Whether this fixture has a convergence floor is undetermined.** No external
   reference exists in this lane; every f∞ here is a fitted extrapolation and they span
   97 MHz. The ledger's measured ε_r-coupled dielectric-interface staircasing floor for
   this fixture class remains fully compatible with everything measured here, and D2 —
   re-taken on the nine-rung ladder — is **INCONCLUSIVE**, not an exoneration. Deciding
   it needs an external reference (§8.2) or the D7 edge-length discriminator (§8.1).

*Falsifier for §10.10's own claims, pre-declared here:* if a fixture in this class with
an **independent** reference (analytic, external solver, or measurement) shows
|f(h) − f_physical| bounded away from zero as h → 0 over a range that includes
dz_fine ≤ 0.0625 mm, then item 6 resolves to "there IS a floor" and the ledger entry is
confirmed on this fixture; if instead it shows |f(h) − f_physical| → 0 at any positive
order, item 6 resolves the other way. **Neither outcome is decided by anything in this
lane**, and no sentence in this note may be read as deciding it.

### §10.11 Harness addendum (extends §9; nothing in §9 is removed)

```
d2_retake            re-applies the FROZEN D2 rule to the nine-rung ladder   (no FDTD)
d5_instrument_check  the frozen D4c pair on the D5 rungs 3/s = 7, 8, 9       (~7 min)
```

Run order after `d5`: `d2_retake` → `d5_instrument_check` → `d6_two_term_model` →
`verdict`. Both new modules read their rule out of
`results/predeclared_windows_786.json`; neither defines a window of its own, and neither
can widen one. `rfx/` remains untouched by this lane
(`git diff main --name-only | grep '^rfx/'` is empty). The only file outside
`validation/research/convergence_floor/` is `tests/_example_fidelity_lib.py`, where the
#737 gate requires a classification entry per script (both new modules are
`no_simulation`).
