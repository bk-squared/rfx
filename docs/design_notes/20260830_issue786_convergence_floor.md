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
