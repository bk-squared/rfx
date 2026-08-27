# Phase-2 plan — dual-band notch (2026-08-27, revised same day)

**Operating rule for this phase (PI direction, 2026-08-27): do the research first and
decide the novelty framing afterwards. Do not shape experiments to fit a claim.**
The first draft of this plan violated that — it reorganized the arms around "what can we
publish" the moment the prior-art survey landed. This version is organized around the
open technical questions instead. Literature findings are kept only where they make the
experiment *better*: they told us which spec is genuinely hard for classical synthesis
(Part I) and which conductor interpolation the field already knows works (Part II). The
publication-positioning notes are demoted to an appendix to be revisited at write-up
time, after we know what is true.

---

# Part I — The benchmark: dual notch at the WLAN pair. Classical genuinely has no knob.

**Spec: a through-line rejecting WLAN 5.15–5.35 GHz and 5.725–5.825 GHz, inside a
bounded footprint, with the passband preserved across 3.1–10.6 GHz.**

Three independent classical walls stack on exactly this spec, each citable from a
full-text-read source:

1. **Frequency ratio.** 5.775/5.25 = **1.10**. The best dual-band SIR bandstop
   synthesis (Chin & Lung, PIER C 10:37–48, 2009) states its own limit: the realizable
   ratio range "is limited from 1.92 to 2.65" because the required high/low impedances
   must stay inside the practical 20–120 Ω microstrip window. Their three fabricated
   filters sit at 2.1, 2.417, 2.85. Our target is **below the floor**.
2. **Bandwidth asymmetry.** The one exact synthesis that explicitly targets *closely
   spaced* bands — Uchida et al., T-MTT 2004 (dual-band rejection filter, exact
   two-step frequency transformation) — carries **one shared rejection bandwidth Δωr
   for both bands** (their Eq. 3, Fig. 1(c)). The real WLAN spec is **200 MHz lower vs
   100 MHz upper, a 2:1 asymmetry**, which is outside that parameterization, not merely
   a hard case within it.
3. **Coupling when bands are close, quantified.** Rahman, Ko & Park (Sensors 17:2174,
   2017) designed four CSRR notches including this exact WLAN pair. Isolated-design
   prediction vs realized with coupling: 3.5→3.85, 4.5→5.2, 5.25→5.95, 5.65→6.25 GHz —
   **10–16 % centre-frequency error, i.e. several notch bandwidths off target.** Their
   words: "quite a challenging task … due to the strong coupling between each notching
   element when each resonant frequency is close to each other (e.g., lower and upper
   WLAN bands)"; and the remedy they name is "to separate away from each resonator
   physically."

**The footprint bound is what closes the classical escape hatches, and it is
principled rather than arbitrary.** Schiffman & Matthaei (T-MTT 1964 — the exact
bandstop synthesis) already prescribe separating stubs by **3λ0/4** "in order to avoid
undesirable interaction effects because of the fringing fields." At 5.5 GHz on our
substrate (ε_eff = 2.87, λg = 32.2 mm) that is **24 mm of line for spacing alone**.
Bounding the design region to ~12 mm along the line therefore forbids the only classical
cure for (3). The second classical escape — duplicating sections for depth, since
"a single coupled-resonator stopband section will realistically produce an attenuation
of approximately 16 dB" (Rambabu et al., T-MTT 2006) — is also area, and the same bound
closes it. One constraint, both escapes.

Two further classical dead ends, recorded because they bound what the baseline can do:
- **Narrow notches need impossible impedances.** Schiffman & Matthaei's own worked
  two-stub 5 %-stopband example needs Z1 = 949.9 Ω, Z2 = 899.9 Ω against a ~120 Ω
  microstrip ceiling.
- **Harmonics are structural.** A λ/4 stub notches at 3f0, 5f0 …; the synthesis has no
  knob for them. Rambabu's fix is a per-harmonic geometry trick, not a design equation.

**Every published route to this class of spec ends in full-wave parametric
optimization**, in the authors' own words: Ning (PIER 131, 2012) "by finely tuning the
dimensions … with the help of a full-wave EM solver"; Zheng et al. (2018) "The final
size of the filter is optimized by using HFSS with 0.1 GHz sweep step"; and even at a
friendly ratio Chin & Lung designed 2.4/5.8 and measured 2.35/5.58 (−2.1 %, −3.8 %).

## Why the single-frequency parity result was the expected outcome

Worth recording as a technical fact, not as rhetoric: for one notch, Schiffman &
Matthaei's exact synthesis **is** the λ/4 open stub. A 2 256-variable search cannot
beat a provably exact solution, so parity was the only outcome available, and our
cross-validated 30.5 vs 31.6 dB is a calibration of the pipeline rather than a
disappointment. It also sets the expectation for this phase: if the dual-band spec
likewise has an exact answer we have not found in the literature, parity is again the
ceiling — and finding that out is a legitimate result.

## Targets to aim at (all measured numbers, full text)

| source | bands (GHz) | ratio | rejection | notch BW | passband IL |
|---|---|---|---|---|---|
| Chin & Lung 2009 (B) | 2.35 / 5.58 | 2.37 | 46 / 40 dB | 40.4 % / 18 % | — |
| Ning 2012 (dual) | 2.37 / 3.54 | 1.49 | 31.4 / 36.7 dB | 6.3 % / 3.4 % | 0.12–0.82 dB |
| Zheng 2018 | 3.5 / 7.5 | 2.14 | 25.2 / 17.3 dB | 15.9 % / 9.6 % | < 1.2 dB |
| Basit 2022 (triple) | 5.1 / 6 / 8 | — | > 15 dB each | ~6 % each | < 0.85 dB |

**Working target (a yardstick, not a promise):** ≥ 20 dB in both bands; 3-dB notch
fractional bandwidths matching the spec asymmetry (≈ 3.8 % lower, 1.7 % upper);
passband IL < 1 dB; first spurious above 10.6 GHz; footprint below two 3λ/4-spaced stub
sections. Bandwidth asymmetry and spurious placement are the two objectives where
classical synthesis has no knob at all, so they are the most informative places to look
— whichever way the measurement comes out.

**Control experiment (do it):** also run 2.4 / 5.8 GHz (ratio 2.42), where Chin & Lung
give a synthesized, fabricated and measured baseline. It is the check that tells us
whether our pipeline and our baseline are both sound — at a ratio classical handles
well, we should tie. A win there would mean the baseline is broken, not that we are
good.

---

# Part II — What the prior art teaches us about the IMPLEMENTATION

The survey's value here is technical, not positional: an established line has already
worked out how to represent a conductor as a continuous design variable, and its
documented failure mode explains a result we already measured. That belongs in the
experiment design. (The positioning material is in the appendix.)

## The relevant line of work

- **Free-form binary metal topology optimization of planar PCB structures with
  S-parameter objectives, FDTD + adjoint gradient, fabricated and measured — is in
  T-MTT already.** Hassan, Scheiner, Michler, Röhrl, Berggren, Wadbro, "Multilayer
  Topology Optimization of Wideband SIW-to-Waveguide Transitions," IEEE T-MTT
  68(4):1326–1339, 2020 (10.1109/TMTT.2019.2959759). **105 960 design variables,
  < 400 Maxwell solves, CST cross-verification, fabricated + VNA-measured.** Its recipe
  is step-for-step what we were about to propose: per-edge density → blurring filter →
  conductivity interpolation → time-reversed adjoint → GCMMA → continuation → threshold
  → independent commercial solve → build → measure.
- **Metal TO with multi-band objectives**: Lu, Wadbro, Berggren, Hassan, EuCAP 2025 —
  density-based FDTD metal TO with a two-band (2.5/5.5 GHz) objective and minimum-size
  control.
- **Metal conductivity interpolation for TO**: Hassan TAP 2014/2015; Aage, Mortensen &
  Sigmund IJNME 2010; Shin & Yoo SMO 2017.
- **Threshold-then-verify-in-an-independent-solver**: standard in that line since 2015.
  Our tier-1 cross-check is good practice, not a contribution.
- **Pixelated dual-band microstrip filters**: Zhang & Xu, IEEE MWTL 34(1):29–32, 2024 —
  32×32 binary pixels, dual-band. Pixelated notch filters fabricated: Gomez et al.,
  Sci. Rep. 15, 2025 — 1 536 pixels, GA in CST, laser-ablated, −61 dB measured.

## The physics correction that changes our implementation

Hassan's scheme interpolates **conductivity** and deliberately uses **no Heaviside
projection**: intermediate conductivity is ohmically lossy, so an energy objective is
**self-penalizing** toward binary. The documented failure mode is the opposite of ours —
self-penalization becomes too aggressive and traps the optimizer — and the established
cure is **filter-radius continuation**, not projection sharpening.

Read our Phase-1 results against that: our arm B ("damped conductive gray") is
essentially a re-derivation of Hassan's interpolation, and **our observed collapse of
arm B to an empty region from a low-fill start is exactly the self-penalization trap
that line documents.** So Phase-2 should adopt the established scheme and test the
projection question rather than assume it:

- exponential σ(ρ) map between ~1e5 and ~1e-3 S/m (Hassan's numbers) instead of our
  ad-hoc linear RAMP;
- **filter-radius continuation** as the primary continuation mechanism;
- Heaviside projection demoted to an ablation arm — a reviewer will ask "why project at
  all when gray metal is already lossy?", and we should have the measurement.

---

# Part III — Experiment design

## Geometry and budget (computed on the current mesh)

Current validated setup: dx = 127 µm, grid 279×180×19 = 954 k cells, dt = 0.242 ps,
ε_eff = 2.87, 30 mm line, F_MAX = 9 GHz. Quarter-wave stubs: **8.51 mm at 5.2 GHz,
7.63 mm at 5.8 GHz.**

**Design region: 12 mm (along the line) × 9 mm (transverse) = 94 × 71 ≈ 6 700 binary
variables.** Transversely a λ/4 stub just fits (8.51 < 9 mm); along the line, 12 mm is
half the 24 mm that Schiffman & Matthaei's anti-coupling rule demands — the constraint
is exactly the one that breaks classical design.

**Window is the real cost driver, and it is a physics constraint, not a knob:**

| periods @ F_MAX | n_steps | record T | DFT resolution | cost/iteration |
|---|---|---|---|---|
| 10 (Phase-1) | 4 589 | 1.11 ns | 0.90 GHz | ~22 s |
| 20 | 9 178 | 2.22 ns | 0.45 GHz | ~44 s |
| 30 | 13 767 | 3.33 ns | 0.30 GHz | ~66 s |
| 45 | 20 650 | 5.00 ns | 0.20 GHz | ~98 s |

The two band centres are **0.525 GHz apart**, so the Phase-1 window (0.90 GHz) **cannot
separate them** — this alone invalidates reusing Phase-1 settings. Resolving the
100 MHz upper notch bandwidth would need ≈ 90 periods (~200 s/iteration, ~8 h per
150-iteration arm).

**Resolution — use our own paper's precedent:** size the descent window for *ranking
designs*, and quote every final number from an independent long-window evaluation. The
accepted paper states exactly this for its notch example ("the window is sized for
descent, not for resolving the final null … all final performance figures below are
therefore quoted from the independent long-window S-matrix evaluation"). Concretely:
**descent at 30 periods (~66 s/iter), verification at 90+ periods**, with rfx's
ring-down settling witness as the gate on the verification window (all Phase-1
cross-checks settled at ≈ −119 dB at 80 periods, so this is calibrated, not guessed).

## Arms — each one answers a question we do not know the answer to

| # | arm | the open question it answers |
|---|---|---|
| A | gradient, conductivity interpolation + filter-radius continuation | **Can a gradient meet this spec at all inside the bounded box?** Nobody has told us; the box was chosen precisely because classical cannot. |
| B | A + Heaviside projection | **Does projection help or hurt a conductor?** The established line says gray metal is already self-penalizing and uses no projection; our Phase-1 used projection and saw a collapse. Both cannot be right for our setup. |
| C | binary heuristic (BPSO / direct binary search), same grid, budget counted in Maxwell solves | **Does the gradient actually buy anything here?** If a binary search on the same grid matches it, that is a real finding about this problem class and we should know it before building more machinery on the gradient. |
| D | classical two open stubs, lengths + positions swept inside the same box | **Where exactly does classical fail, and by how much?** The premise of the whole benchmark. If a calibrated two-stub design meets the spec in this box, the benchmark is wrong and we pick another. |
| E | classical SIR at the 2.4/5.8 control ratio (Chin & Lung: synthesized, fabricated, measured) | **Is our whole pipeline sane?** At a ratio classical handles well, we should tie, not win. A win there would mean our baseline is broken. |

Arm D is the gate on the benchmark itself and runs FIRST — the Stage-0 window sweep
already carries its geometry, so its answer arrives before any optimization is spent.
Arm E is the sanity check that keeps us honest if A or C looks too good.

For C, the budget must be pre-registered and counted in **Maxwell solves, not
iterations** (a gradient iteration is 1 forward + 1 backward ≈ 2 solves), and the output
is a budget-vs-quality curve, not a single point. That curve is the honest form of the
comparison whichever way it comes out.

## Gates (all pre-registered, all already built or nearly so)

1. **Ring-down settling witness** < −40 dB on every quoted number (already wired).
2. **Independent hard-PEC re-solve** of every final design as real `Box` geometry with
   the imperative extractor (already built, tier-1).
3. **Mesh transferability**: re-evaluate at dx/2, re-optimize at production resolution
   if it degrades — our own paper's Sec. V-C recommendation, applied to ourselves.
4. **Filter radius fixed in millimetres, not cells** (Phase-1 bug: specifying it in
   cells makes the design problem change under mesh refinement; the library's own
   design-region API already takes metres).
5. **Baselines calibrated on the same mesh** (the Phase-1 retraction came from skipping
   this).

## Staging

- **Stage 0 (running).** Window adequacy on the classical two-stub design in the box:
  ring-down witness and notch resolution at 20/30/45/90/140 periods. Fixes the descent
  and verification windows by measurement. **It also answers arm D's first half** — a
  CPU check at the Phase-1 window already shows the two-stub design landing at 4.80 and
  5.60 GHz against 5.25/5.775 targets, in the direction the coupling literature
  predicts, though the window was too short to trust the numbers.
- **Stage 1.** Finish arm D properly: sweep stub lengths and positions inside the box at
  the measured verification window. **Decision gate: if a calibrated two-stub design
  meets the spec here, this benchmark is wrong and we choose a different one** rather
  than proceeding.
- **Stage 2.** Gradient arms A and B on the WLAN spec. Before optimizing, verify the
  gradient through the conductivity map with a directional-derivative Richardson check
  (the Phase-1 two-point finite-difference check was inconclusive and must not be
  repeated as-is).
- **Stage 3.** Arm C, budget-matched, with the budget-vs-quality curve.
- **Stage 4.** Control at 2.4/5.8 (arm E).
- **Stage 5.** Whatever the results turn out to need: mesh transferability at dx/2,
  spurious-response placement above 10.6 GHz, and the rfx core-patch proposal (first
  item: `topology_optimize()` still defaults to the legacy damping path when the
  foreground material is PEC — the path our probe measured as gradient-starved).

## Scope question to settle before Stage 3 (not now)

**Fabrication and measurement.** Relevant regardless of framing, because tolerance is a
physics question here: Gomez et al. laser-ablated a pixelated notch filter and still
measured a 160 MHz centre shift from ε_r tolerance and cutting overcut, and free-form
binary pixels have finer, more tolerance-sensitive features than a stub. So even for our
own understanding, a measured board would tell us how much of the simulated margin
survives manufacture. Cost and timing are the PI's call, and nothing before Stage 3
depends on it.

---

# Sources

Classical: Schiffman & Matthaei T-MTT 1964 (10.1109/TMTT.1964.1125744) · Uchida et al.
T-MTT 2004 (10.1109/TMTT.2004.837161) · Rambabu et al. T-MTT 2006
(10.1109/TMTT.2006.877813) · Chin & Lung PIER C 2009 (10.2528/PIERC09080306) · Ning
PIER 2012 (10.2528/PIER12072109) · Rahman, Ko & Park Sensors 2017 (10.3390/s17102174) ·
Zheng et al. 2018 (10.3390/mi9060280) · Basit et al. 2022 (10.1371/journal.pone.0268886).

Inverse design: Hassan et al. IEEE T-MTT 2020 (10.1109/TMTT.2019.2959759) · Hassan et
al. IEEE TAP 2014 (10.1109/TAP.2014.2309112) · Hassan et al. IEEE TAP 2015
(10.1109/TAP.2015.2449894) · Lu et al. EuCAP 2025 (10.23919/EuCAP63536.2025.10999941) ·
Aage, Mortensen & Sigmund IJNME 2010 (10.1002/nme.2837) · Shin & Yoo SMO 2017
(10.1007/s00158-017-1792-3) · Arsanjani et al. IEEE T-MTT 2025
(10.1109/TMTT.2024.3519274) · Zhang & Xu IEEE MWTL 2024 (10.1109/LMWT.2023.3329047) ·
Gomez et al. Sci. Rep. 2025 (10.1038/s41598-025-10666-y) · Parsaei et al. IEEE TCAS-I
2023 (10.1109/TCSI.2023.3314621).

Full texts read for the load-bearing quotes: Schiffman & Matthaei 1964, Uchida 2004,
Rambabu 2006, Chin & Lung 2009, Ning 2012, Rahman 2017, Zheng 2018, Basit 2022 (classical
side); Hassan T-MTT 2020, Hassan TAP 2015, Gomez 2025 (inverse-design side). Everything
else is metadata-verified but abstract-only and is flagged as such in the survey records.

---

# Appendix — positioning notes, to be revisited at write-up time

**Do not let anything in this appendix steer an experiment.** It is recorded now only so
the survey work is not lost, and it should be re-read after we know what the results are.

## Where the survey thought the open ground was

1. **[HIGH] An equal-budget, same-grid, same-solver head-to-head between the gradient
   and the binary heuristics that actually own this device class.** Hassan asserts a
   2–3 order-of-magnitude advantage over pixel-GA **by citation, not measurement**;
   Gomez spent ≈ 1 600 CST solves on 1 536 pixels; a gradient run costs ~2 solves per
   iteration regardless of variable count. Nobody has run them against each other on
   one grid, one objective, one solver, one budget. **This is also the natural extension
   of what our accepted paper already did for the dielectric taper** (budget-matched
   particle-swarm and GA trailing the gradient by ≥ 11.6 dB) — same experiment, moved
   to binary metal on the device class where the heuristics are the incumbent.
2. **[MEDIUM] A measured gray→binary rasterization penalty for conductors.** The Umeå
   line reports thresholding as benign; the pixel line is binary by construction and has
   nothing to report. We already have one datum from Phase-1: our stub-seeded design's
   notch moved 6.00 → 6.10 GHz when re-rasterized as real geometry, while the
   distributed design did not. "How many dB does binarization cost, which designs pay
   it, and does re-optimizing at production resolution recover it" is the direct
   analogue of the honesty move our accepted paper made for the Klopfenstein taper.
3. **[MEDIUM] Bridging the two literatures.** Arsanjani (T-MTT 2025, pixelated
   metasurface components) and Hassan (T-MTT 2020, density TO) do not cite each other.

## The framing that survives contact with the literature

> **"Gradient versus binary search on the pixel grid: an equal-budget comparison for
> free-form microstrip filters, on a dual-band spec classical synthesis cannot
> express."**

The dual-band notch becomes the **testbed chosen because it stresses the optimizer**,
not the claimed novelty. Hassan T-MTT 2020 gets cited in the **first paragraph of the
introduction**, not buried in related work.

## What a skeptical reviewer asked of the first draft

The prior-art survey listed the objections a T-MTT reviewer would raise against a
"first free-form metal TO" framing — chiefly Hassan T-MTT 2020 (105 960 variables,
fabricated and measured), Zhang & Xu MWTL 2024 (pixelated dual-band microstrip filter),
and "why Heaviside at all when gray conductor is self-penalizing?". The last one is a
physics question and has been promoted into the experiment as arm B. The rest are
positioning and wait for results.
