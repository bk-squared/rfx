# Task battery — blind docs-only footgun test

Each task is a realistic request a docs-only user would make. Give the agent the
**Prompt** verbatim (plus the standing "public docs + docstrings only, no source"
constraint from the README). Score its output against **Ground truth** using the
rubric in the README. The **Footgun** column is for the scorer only — never put it
in the agent's prompt.

Rotate/extend this battery over time; add a task whenever a new footgun is found
so future runs regression-test the fix. Keep each task self-contained and
answerable from the public docs alone.

---

## T1 — cavity-Q

- **Prompt:** "Build a small rfx simulation of a closed rectangular PEC cavity and
  report its lowest resonant frequency and its quality factor Q."
- **Footgun:** a lossless closed cavity has *infinite* physical Q; a finite Q read
  from `harminv` on such a run is a window-length artifact. Frequency is fine.
- **Ground truth:** the resonant frequency for the chosen box (analytic
  `f = (c/2)·sqrt((m/a)²+(n/b)²+(p/d)²)`); Q should be reported as
  infinite/ill-defined for the lossless case (or the agent should add a realistic
  loss / open boundary before quoting a finite Q). A confidently-quoted finite Q
  with no caveat = `BITTEN`.

## T2 — pec-short-s11

- **Prompt:** "Terminate a rectangular waveguide (or a lumped port line) in a PEC
  short and compute |S11| across the band. Report the magnitude."
- **Footgun:** `normalize=True` is wrong for a strong total reflector; it can push
  |S11| well below the physical ~1.0. The docs steer to `normalize=False`.
- **Ground truth:** |S11| ≈ 1.0 across the band (a PEC short reflects all power).
  Values materially below 1.0 reported as physical = `BITTEN`.

## T3 — lumped-load-s11

- **Prompt:** "Place a lumped RLC load in an rfx simulation and compute the |S11|
  that a 50 Ω line sees looking into it."
- **Footgun:** `add_lumped_rlc` is a circuit element, not a reflection-referenced
  port; a single-cell lumped element read directly reports its own self-reflection
  (~1.0), not a `Z0`-referenced load Γ. The documented path is
  `add_port(..., impedance=50)` + `run(compute_s_params=True)` /
  `forward(port_s11_freqs=...)`. (Pinned by PR #294.)
- **Ground truth:** for a chosen `R` the analytic Γ = (R−Z0)/(R+Z0) (real load),
  or the full RLC Γ(f). Reporting the ~1.0 single-cell self-reflection as the load
  |S11| = `BITTEN`. Refusing / redirecting to `add_port(impedance=)` =
  `PROTECTED_BY_DOCS` (the target state after #294).

## T4 — rcs-pattern

- **Prompt:** "Compute the radar cross section of a conducting sphere in rfx and
  report the full bistatic RCS pattern."
- **Footgun:** only the monostatic (backscatter) bin is cross-validated against
  the exact Mie series; the off-backscatter bistatic pattern at the default NTFF
  box is not validated (a spurious near-field lobe). Enlarging `ntff_offset` alone
  does not fix it.
- **Ground truth:** monostatic RCS near the Mie value for the chosen `ka`;
  bistatic cuts labelled qualitative / not-validated. Presenting the full bistatic
  pattern as validated = `BITTEN`; trusting only monostatic = `WARNED_ADAPTED`.

## T5 — grad-optimize

- **Prompt:** "Use rfx's differentiable FDTD to optimize a dielectric matching
  block that minimizes reflection at a port. Set up the gradient loop, verify the
  gradient, and report the optimized reflection objective."
- **Footgun:** `minimize_reflected_energy` splits the probe series into
  early/late; if the round trip is short (or the run too short) the reflection
  never lands in the late window, the proxy collapses to `~1e-7`, and its gradient
  is numerical noise. An AD-vs-FD check **passes anyway** (both differentiate the
  same empty window). (Pinned by PR #294.)
- **Ground truth:** a meaningfully reflecting design gives a proxy of order
  `1e-2`–`1e-1`; a reported optimized loss of `~1e-7` accepted as a matched design
  = `BITTEN`. Sanity-checking the loss magnitude and fixing the window
  (raise `late_fraction` / enlarge `n_steps` / use an impedance-referenced port)
  = `PROTECTED_BY_DOCS` (the target state after #294).
