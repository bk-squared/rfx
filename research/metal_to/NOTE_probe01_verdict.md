# Probe #1 verdict — the gray-metal continuation landscape, measured (2026-08-25)

Branch `feat/metal-topology-ramp`, script `probe01_gray_landscape.py`,
outputs `out/probe01_{kottke,legacy}.{json,png}`. CPU, grid (279,180,19),
n_steps 4623, MSL notch geometry with the stub FIXED at the analytic
lambda/4 length (7.374 mm); uniform occupancy level swept 0 -> 1 in 11 steps
on both `pec_occupancy_override` consumers.

## Verdict: falsifier NOT triggered — the gray traversal is non-benign on
## both paths, and the two failure mechanisms are DIFFERENT

**Kottke inv-eps path (production).** J(level)=|S21(f_t)|^2 is flat from 0 to
0.5, rises at 0.6, has a SPURIOUS LOCAL MINIMUM at 0.7 (J=0.022 vs plateau
0.041), a BARRIER at 0.8 (J=0.037), and only then descends to the true notch
(J=0.0004 at 1.0, about -34 dB). The band plot shows why: the gray stub is a
lossless high-effective-eps dielectric resonator whose resonance SWEEPS DOWN
THROUGH THE BAND as occupancy grows — at level 0.7 the moving dip sits near
7 GHz and leaks into the 6 GHz target (false valley); by 0.8 it has moved
past (barrier); at 1.0 the true lambda/4 PEC notch lands at 5.5-6 GHz. The
surface-Ez maps confirm it: at level 0.3 the field does not see the stub at
all; at 0.7 the field is INSIDE the stub (dielectric-resonator mode); at 1.0
the field is expelled (real metal). A density optimizer following this
landscape parks in the false valley or is repelled by the barrier — a
concrete, measured mechanism for "preliminary density runs converged to
weaker optima" (paper Sec. V-C).

**Legacy damping path.** No mid-path resonance, but |S21(f_t)| slides only
0.203 -> 0.185 over levels 0 -> 0.9 (a ~90%-of-path PLATEAU: the damped gray
stub is nearly invisible to the through-line) and then cliffs to 0.055 at
1.0. Gradient signal is vanishingly small over most of the traversal. (Its
sub-beta wiggle pathology near high-Q resonances is separately documented in
the notch example header.)

**Prior expectation corrected.** The blanket "gray metal is a nonphysically
lossy medium" story (accurate for the legacy path and for direct linear
sigma interpolation) does NOT describe the production path: Kottke gray is
lossless, and its pathology is resonance detuning, not absorption. On the
side-stub geometry the legacy absorber also did NOT produce a mid-path
absorption bump in the through-signal — it produced signal starvation.

## Implication for the remedy (redirects the naive plan)

- Naive convex "delay the metal-ness" shaping of occ(rho) would WORSEN the
  dominant plateau problem.
- A concave RAMP-like shaping fights the plateau but, on the Kottke path,
  drags the spurious-resonance excursion earlier; a pure reparameterization
  of the same 1-D family cannot remove the excursion.
- The remedy must change WHAT gray means physically, not just where on the
  path a given rho sits. Leading candidate, Phase-1 A/B: **conductivity-aware
  gray** — intermediate rho gets a finite, RAMP-scheduled conductivity
  (sigma(rho) concave; PEC only at rho ~ 1 via the Kottke fold), so the gray
  resonator is deliberately DAMPED while it forms: the spurious dielectric
  resonance cannot ring, and the through-signal sees a monotone loss->metal
  progression. The paper's "lossy gray" intuition returns as the fix, in a
  controlled amount, rather than as the bug. (Skin-depth caveat from the
  2026-05-16 blocker note applies only at sigma ~ 1e7; mid-path RAMP sigma is
  orders lower.)

## Caveats (scope of validity)

- Absolute R/T/A here are NOT calibrated: short window (10 periods) and the
  deliberately-diverged plane extractor (defect pair documented in
  `rfx/probes/msl_wave_decomp.py::_v_from_plane`) — hence A ~ 0.87 even with
  no metal, and R slightly > 1 at level 1. Landscape SHAPES are valid: same
  window, same extractor, same grid across all levels (the notch example
  optimizes exactly this quantity).
- One geometry (side-coupled stub), one dx (127 um), uniform-level 1-D
  family. The free-form per-cell case is Phase-1's subject.

## Phase-1 plan (next)

Free-form per-cell density over the stub-region footprint, objective =
notch depth at f_t; arms: (A) Kottke linear (current), (B) Kottke +
RAMP-sigma gray damping, (C) legacy. Gate: AD-vs-FD gradient check on (B),
then binarized HARD-PEC re-evaluation of every final design (the notch
example's cross-check pattern); success = (B)'s binarized notch beats (A)'s
by a clear margin at equal iteration budget, at two dx values.
