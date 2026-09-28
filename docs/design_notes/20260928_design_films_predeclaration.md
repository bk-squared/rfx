# Pre-declaration: design films from the current code (issue 1359)

**Status: PRE-DECLARATION. Written and pushed before any job of this batch was submitted.**
Nothing below comes from a run of this batch. The check rules and thresholds are the lane
leader's, from the brief for issue 1359 (2026-09-28); this note copies them and fixes the
choices the brief left open. It judges nothing itself.

Two gradient-descent design loops from the T-MTT examples are run again on the current code, and
every iterate is recorded. The films draw three things side by side on one clock: the design
being changed, the response it produces, and the gradient that sets the next change.

1. **WR-90 dielectric taper.** A TE10 wave in WR-90 meets an eps_r = 9 load that fills the guide.
   Thirty graded dielectric sections ahead of the load are the design variables. The objective
   is the band-mean reflected power over 8.2-12.4 GHz.
2. **Beam-steering dielectric cover.** A dipole a quarter wavelength above a finite reflector
   radiates through a dielectric cover. The cover permittivity is designed to raise directivity
   toward 30 degrees in the E-plane.

Before each descent starts, its gradient is checked against central differences and against
the same gradient taken from a record 1.5 times longer. When the descent ends, the final design
is read back from the record and solved again without the optimiser, both on the same mesh and
on a finer one.

Scripts: `scripts/showcase/{design_taper,design_beam,_design_common,render_design}.py`, reusing
`scripts/showcase/_record.py`. Jobs: `scripts/vessl_showcase_design_{taper,beam}.yaml`. Every job
writes `result.json` (schema `rfx-showcase-result/1`) beside its arrays. In it a number is either
**judged**, carrying a threshold named here, or **reported**.

## 1. Taper

**Structure.** The model comes from `validation/tmtt_paper/waveguide_dielectric_taper.py`
(`build_sim`, the `SMOKE=0` constants), imported by path and not edited. Its walls are PEC y/z
domain faces, CPML runs along x, both ports are waveguide ports, and the eps_r = 9 fill runs from
the end of the taper into the right absorber.

**Mesh.** dx = a/36 = 0.635 mm. It divides both WR-90 walls, so the guide solved is 22.86 × 10.16
mm (36 × 16 cells), with no rounding. The module's metre lengths are rounded to whole cells of
this mesh:

| length | module (m) | cells | mm |
|---|---|---|---|
| domain along x | 0.220 | 346 | 219.71 |
| port plane from each end | 0.030 | 47 | 29.845 |
| reference plane from each end | 0.040 | 63 | 40.005 |
| taper start | 0.060 | 94 | 59.69 |
| eps_r = 9 fill start | 0.140 | 220 | 139.70 |

This gives a taper of 126 cells = 80.01 mm. The section edges follow the module's `design_layout`
rule, `round(linspace(94, 220, 31))`. Sections 3, 8, 13, 18, 23 and 28 (counted from the vacuum
side) are 5 cells long and the other 24 are 4 cells. The script checks that its edges equal the
module's `design_layout` edges at this mesh, and it raises unless the grid built exactly
22.86 × 10.16 mm.

**Two departures from the module's settings.** The current code asks for both at run time. The
module's own values are also run, on the final design and on the Klopfenstein taper, and those
results are reported beside the others (§1.6).

- **Absorber: 72 CPML layers (45.72 mm) instead of 24 (15.24 mm).** With 24 layers,
  `compute_waveguide_s_matrix` warns on every call: "absorber on the x propagation axis is
  thinner than the documented 0.5 guide-wavelength far-port discipline at the lowest measured
  frequency 8.200 GHz ... 0.25 lambda_g". It also quotes issue #494: residual |S11| ripple of
  0.0706 at 0.30 lambda_g, 0.0366 at 0.50 and 0.0093 at 0.75. 72 layers is 0.75 lambda_g of the
  vacuum guide at 8.2 GHz. The absorber sits outside the domain, so no port, plane or section
  moves.
- **Record: 171 periods of f_max instead of 120.** The preflight record-length check
  (`_validate_cfg_record_vs_far_boundary`) compares the record T with the time tau_far for a wave
  to go from a port to the far outer wall and back. It states that T/tau_far >= 3 was needed on the
  WR-90 battery at a/18, 5 at a/36 and 8 at a/72, and treats 3 only as the floor. With the
  72-layer stack the far path from the left port is 235.6 mm, and at 8.2 GHz v_g = 0.601 c, so
  tau_far is 2.62 ns. 120 periods (T = 9.3 ns) gives 3.55. 171 periods, rounded up to a multiple
  of the 120 checkpoint segments, gives 10 920 steps, T = 13.2 ns, and a ratio of 5.05. The
  preflight uses the vacuum group velocity, so the part of the path that runs in the eps_r = 9
  fill makes the true round trip longer than this estimate.

**Band and the one preflight advisory.** The 21 bins cover 8.2-12.4 GHz, as in the module.
Preflight reports, for each port, "max measurement frequency 12.400 GHz exceeds 0.90 × fc_next =
11.803 GHz ... Evanescent TE20 contamination may exceed 1 %". Disposition: the band is kept. Every
section fills the whole guide cross-section, so no section can turn TE10 into TE20, and the port
launches TE10 only. The advisory text is kept in `preflight_*.json`.

### 1.1 Objective and optimiser (the module's)
- **J = mean over the 21 bins of |S11|^2.** S11 comes from `compute_waveguide_s_matrix`
  (`normalize=False`, the full record, left port), which is the module's `make_objective` call.
  Each iterate also records 20 log10(mean |S11|), the module's printed "band-avg" number, and
  max |S11|.
- Each section's eps_r is 1 + 8 sigmoid(theta_s). The start is theta = 0, so every section has
  eps_r = 5.
- Adam (`optax.adam`, lr 0.2), 120 steps, `checkpoint_segments` 120. The loop runs
  `jax.jit(jax.value_and_grad)`, where the module uses plain `value_and_grad`. Precision float32.

### 1.2 Recorded at every iterate (0 ... 120)
For each iterate: theta; the 30 section eps_r; complex S11 and |S11| on the 21 bins; J in linear
units and in dB; 20 log10 mean |S11|; dJ/dtheta and dJ/d eps_r per section
(dJ/d eps_r = dJ/dtheta / (8 sigma (1 - sigma))); and the wall time. Iterate 120 is the final
design, solved forward with no step taken. `iterations.npz` and the Adam state are rewritten
after every iterate.

### 1.3 Baselines at a/36 (reported)
- **Klopfenstein taper of the same 30 sections.** The module prints a Klopfenstein number but
  carries no Klopfenstein code, so it is built here, in closed form, with no FDTD:
  - ln Z_TE10(f0) follows Klopfenstein's profile (Pozar §5.9) along normalized electrical length
    at f0 = 10.3 GHz, the band centre. It runs from the vacuum guide to the eps_r = 9 guide.
  - Position follows from beta(f0) = k0 / Z (Z in units of eta0). Each section takes the
    profile's eps_r at its centre.
  - The one free parameter A is chosen so that J, computed by the closed-form TE10 section
    cascade on the same 21 bins and the same section lengths, is smallest. The search is a grid
    of step 0.25 on [0, A_pb], then a bounded refinement. A_pb is the textbook passband rule,
    electrical length at 8.2 GHz = A, and is recorded with its own cascade J.
  - The taper is then solved with the FDTD at a/36 by the same call as the design.
- **No taper** ("bare step"): vacuum up to the eps_r = 9 fill.
- **The start** (eps_r = 5 in every section), solved eagerly as a cross-check on iterate 0.

### 1.4 Gradient against central differences (judged)
Checked at the start (eps_r = 5 everywhere), for dJ/d eps_r of three sections chosen now:
section 1 (the vacuum end), section 16 (the middle) and section 30 (next to the load).
- **FD** is the central difference of J for a change of that section's eps_r by ±h, with
  h = 0.1, 0.05 and 0.025.
- **AD** is `jax.grad` of J with respect to the 30 section eps_r.
- **Judged:** |AD − FD| / |FD| ≤ 0.05 at h = 0.05, for each section whose |FD(0.05)| is at
  least 0.1 × the largest |FD(0.05)| of the three. The others are reported.
- **Float64 repeat.** A judged section's ladder counts as round-off-dominated when FD moves more
  between h = 0.05 and 0.025 than between 0.1 and 0.05. If that happens for any judged section,
  a separate process repeats the AD and the whole ladder with `JAX_ENABLE_X64=1` and a float64
  `Simulation`. The float64 comparison is then the judged one, and the float32 one is reported.
  *Amendment 1 (2026-09-28, after the timing runs and before either full run):*
  `compute_waveguide_s_matrix` ignores `Simulation(precision=...)`. Its
  `extract_waveguide_s_matrix` calls `rfx.simulation.run` without `field_dtype`, so the fields
  stay float32 even with x64 enabled. The taper's float64 process therefore wraps
  `rfx.simulation.run` for that process only, so that it passes `field_dtype=float64`. It records
  the dtype of every field state returned and refuses to write the record unless all of them are
  float64. The beam's forward lane does honour `precision`, and its float64 process records and
  checks the field dtype the same way.

### 1.5 Record-length witness (judged)
`rfx.gradient_record_length_witness` is taken at the start with factor 1.5, on the full 30-section
dJ/d eps_r.
- **Judged:** the helper's ||g_1.5 − g_1.0|| / ||g_1.5|| ≤ 0.05.
- **Judged:** |g_1.5 − g_1.0| / |g_1.5| ≤ 0.05 for each FD-judged section.
- **Reported:** J at 1.0× and 1.5×, and the cosine between the two gradients.
- On each arm, `checkpoint_segments` is the largest divisor of that arm's step count that is
  ≤ 120.
- **A failed witness stops the case before the descent** (exit 3). The record is still written,
  and nothing is lengthened after the fact.

### 1.6 Re-solve without the optimiser (reported)
A separate process reads the final eps_r and the Klopfenstein eps_r from the record. It solves
them with the same lane as the design, with no gradient and no checkpointing:
- at **a/36**, with 72 layers and 171 periods: final, Klopfenstein, start;
- at **a/72 = 0.3175 mm**, with 144 layers (the same 45.72 mm) and 273 periods (T/tau_far about 8):
  final, Klopfenstein, start. The section edges double, so each section keeps its physical extent;
- at **a/36 with the module's absorber and record** (24 layers, 120 periods): final and
  Klopfenstein.

Every record length follows the module's rule, `num_timesteps` rounded up to a multiple of 120, so
the a/36 solve has the same step count as the loop. Each solve reports 20 log10(mean |S11|),
max |S11| in dB, and J. Any run-time advisory is stored
with the solve.

## 2. Beam

**Structure and mesh.** `validation/tmtt_paper/beam_steering_superstrate.py`, `build_problem()`
with `SMOKE=0`, imported by path and not edited. The dipole points along x, 3 GHz, and sits
lambda/4 above a 149.9 mm square PEC sheet. The cover is 1.5 lambda square and 2 cells thick. The
mesh is dx = lambda/20 = 4.9965 mm, the grid is 93 × 93 × 84, and the record is 60 periods of
4 GHz = 1575 steps. The cover occupies nodes (31…61, 31…61, 48…50), 31 × 31 × 3. Preflight at this
setting prints the plate's sheet-edge advisory (+2.33 %, open issue #1138) and four NTFF faces at
exactly lambda/2 from the plate (49.97 mm). Both are recorded and left as the module has them.

**Parameterization (added here).** The module optimizes all 2883 cover cells. The issue and the
paper use 441 latent variables, and the module carries no such map, so it is added:
- a 21 × 21 grid of control points, each eps_r = 1 + 9 sigmoid(psi);
- linear interpolation in x and in y onto the 31 × 31 cover nodes (end points coincide, one
  control point every 1.5 cells), the same value through the 3 node layers.

The start is the module's ramp, eps_r 2 → 9 along x, placed at the control points. Linear
interpolation reproduces it exactly on the cover nodes, and the script checks this.

**Objective and optimiser (the module's).**
- L = −log U(30°, 0)/P + 0.3 log mean U(0, φ)/P + 0.5 log mean_{θ>90°} U/P, evaluated on its
  73 × 73 (θ, φ) grid.
- Adam (lr 0.08), 140 steps, float32.
- **Departure (Amendment 2, after the timing runs and before the beam's full run): the far-field
  call.** The module calls `compute_far_field`, which uses the numpy transform on an eager call and
  the JAX transform (`compute_far_field_jax`) under a gradient. Under `jax.jit` it fails, because
  the forward's own NTFF box carries traced frequencies that the transform reads with numpy. The
  unjitted gradient took 82 s per iteration on the timing run (RTX 2070 SUPER). Here
  `compute_far_field_jax` is used on every call, on a box built once by `make_ntff_box` from the
  simulation's NTFF declaration (the call `forward` makes) with its frequencies held as numpy, and
  the gradient step is jitted. On CPU at 300 steps and the start cover, the two paths differ by
  7.2e-7 of the pattern maximum and 8.8e-7 in gradient norm. The job records the module-path
  against script-path difference at the full record length (`pattern_equivalence.json`, reported).
- **Departure, memory only.** The forward runs with `checkpoint_segments` = the largest
  divisor of the step count not above its square root (35 for 1575 steps), in place of the
  module's per-step `checkpoint=True`. Per-step remat keeps the six field arrays of every step,
  1575 × 17.4 MB ≈ 27 GB at lambda/20, and the 1.5× witness arm needs half as much again.
  Segmenting changes what reverse mode stores, not what it computes. On CPU at 300 steps (15
  segments against per-step), the patterns are identical and the gradients differ by 1.3e-7 in
  norm. Apart from these two departures and a dtype, the pattern function is the module's
  `make_pattern_fn`, line for line.

**Recorded at every iterate (0 ... 140):** psi; the control-point eps_r; the cover eps_r on its
31 × 31 nodes; the directivity D(θ, φ) on 73 × 73 (so the E-plane and H-plane cuts are both
stored); D(30°); broadside D; the E-plane peak angle and value; L; dL/dpsi and dL/d eps_r at the
control points; and the wall time. Iterate 140 is the final design (a forward only). The record
is rewritten after every iterate.

**Baselines at lambda/20 (reported).** A uniform cover over the same footprint and thickness with
eps_r = 1, 2, …, 10 (eps_r = 1 is no cover). Each gets D(30°), its pattern, and the module's
ring-down settling witness, with the module's −40 dB verdict. The best uniform D(30°) is marked.

**Gradient against central differences (judged).** Checked at the start, for dL/d eps_r at three
control points on the centre line y = 0: (5, 10), (10, 10) and (15, 10), at x = −37.5, 0 and
+37.5 mm from the cover centre.
- FD steps h = 0.2, 0.1 and 0.05 in control-point eps_r.
- **Judged:** |AD − FD| / |FD| ≤ 0.05 at h = 0.1, for each point with |FD(0.1)| ≥ 0.1 × the
  largest. The others are reported.
- The float64 rule is the one in §1.4, with the steps 0.2 / 0.1 / 0.05.
- Both sides use the module's far-field call. AD differentiates its JAX transform, and FD
  evaluates its numpy transform on eager forwards.

**Record-length witness (judged).** Factor 1.5 and tol 0.05, judged as in §1.5: on the norm over
the 441 control points and on each FD-judged point. The stop rule is also the one in §1.5.

**Re-solve without the optimiser (reported).** A separate process reads the final control-point
eps_r from the record and solves:
- at **lambda/20**, eagerly;
- at **lambda/40** (dx = 2.498 mm, 20 CPML layers, the same thickness). The lambda/20 domain,
  plate, source, probes, NTFF box and cover region are kept, and every one of them lies on a
  lambda/20 node. The cover is 2 lambda/20 cells thick, now 5 nodes. The cover map is the same
  bilinear map sampled on 61 × 61 nodes, and the script checks that it equals the lambda/20 map on
  the shared nodes.

The designs solved are the final cover, no cover, and the best uniform cover. Each reports D(30°),
the E-plane peak angle, and the settling witness. At refine 1 the lambda/40 builder must reproduce
`build_problem()`'s grid, design indices and plate plane, or the job stops.

## 3. Compute
All runs are on VESSL, with a `Resource choice:` record per submission (rfx-archive
`RESOURCE_CHOICE.md`). The first job of each case is a timing run at full resolution with 2
iterations: first-call and steady-state forward and gradient times, device memory, and a forward on
the re-solve mesh. The projection formula is written into `timing.json`. If the projection is
≤ 6 GPU-hours for the case, the full run is submitted without waiting; otherwise the case stops and
the projection is reported.

## 4. Films and storyboard
`scripts/showcase/render_design.py` draws everything from the stored arrays. It computes nothing
but unit conversions and display normalization.
- **Three panels on one clock:**
  - Design: the section eps_r over a side view of the guide and load, or the cover eps_r map in
    plan view with the dipole and plate outline.
  - Result: |S11| in dB across the band with the Klopfenstein curve, or D(θ) in dBi on the
    E-plane with a line at 30°.
  - Gradient: dJ/d eps_r per section, or dL/d eps_r on the control grid, on a symmetric
    diverging scale.
- **Header:** the goal in plain words, `iteration k / N`, and the objective with its unit.
- **Timing:** a staged intro at iteration 0 (the panels appear about one second apart), fixed axes
  and fixed design and result colour scales, and a final hold with a start → final strip.
- **Gradient scale:** two variants are rendered, fixed from the first iterations and normalized
  per frame with the norm printed.
- **Formats:** site 1920 × 1080 and social 1080 × 1350, H.264 yuv420p, no audio.
- **Storyboard:** a contact sheet per case, with rows at iterations 0, ~25 %, ~50 % and final and
  columns for the three panels.
- Nothing goes to the share host or to a public page until the lane leader approves the storyboard.

## Amendment 3 (2026-09-28): the lane leader's decisions, before the beam rerun and any keyframe run

The first beam full run (369367265782, 441 control points) stays in the archive, marked superseded.
The taper's optimization record (369367265774) is kept as it is. None of the runs below has been
submitted yet.

**A3.1 The beam uses the module's own parameterization.** Every one of the 31 × 31 × 3 = 2883 cover
cells is a design variable, with εr = 1 + 9 sigmoid(ψ) and the module's ramp start. No
control-point layer is added.
- The FD variables are three cells on the centre line y = 0, in the middle layer: (7, 15, 1),
  (15, 15, 1) and (23, 15, 1), at x = −40, 0 and +40 mm. The steps, the judged step and the rule
  are those of §2.
- The witness runs over all 2883 cells.
- Recorded at every iterate: the cells' εr, dL/dψ, dL/dεr per cell, and dL/dεr summed over the
  3 layers (a 31 × 31 map).
- The λ/40 re-solve writes each λ/20 cell's εr onto the 2 × 2 × 2 λ/40 cells nested in it. A
  λ/20 εr index i covers [x_i, x_i+1). Every position is held.

**A3.2 Amendment 2's condition, judged.** At the start cover and with the full record, the
module's `make_pattern_fn` (per-step checkpoint, the `compute_far_field` dispatch, not jitted) is
compared with this script's path (`checkpoint_segments`, `compute_far_field_jax`, jitted).
- Judged: the largest pattern difference ÷ the pattern maximum ≤ 1e-4.
- Judged: ||g_script − g_module|| / ||g_module|| ≤ 1e-4, for the gradient with respect to the 2883
  cell εr.
- If either fails, the case stops (exit 4) and the record is written. The module's per-step tape
  needs about 27 GB, so the job runs on an RTX A6000.

**A3.3 Float32 matmuls without TF32 for the beam.** Every beam stage sets
`jax_default_matmul_precision = "highest"`. Measured on the superseded run, same commit:
- The same float32 FD ladder scattered on the RTX A6000, by up to 0.36 relative, but was clean on
  the RTX 2070 SUPER, where the judged rows were ≤ 1.05e-3.
- The module-path comparison read 1.97e-4 on the A6000 and 1.3e-6 on the 2070 SUPER.

The suspected cause is TF32, which Ampere cards use for float32 matmuls by default; it has not been
verified. The setting keeps float32 matmuls in float32 on every card. The taper's recorded descent
ran on an RTX 3080 without it (its float32 FD rows are in fd_start.json of run 369367265774) and is not re-run.

**A3.4 Film labels.**
- The baseline curve is labelled "Klopfenstein profile, A tuned for this band (closed form)", and
  A and its closed-form J are recorded.
- The taper header shows 10·log10 of the band-mean |S11|², in dB.
- The beam header shows D(30°) in dBi and names the objective: "raise D(30°) while holding
  broadside and back lobes down".
- Gradient panels are in εr units. The beam's gradient map is the per-cell dL/dεr summed over the
  3 layers. Its design map is the layer mean (all three layers are recorded).

**A3.5 Recorded for visualization, not judged.** These are extra solves of stored designs; the
optimization record is only read.
- **Taper, keyframes 0, 6, 18, 42, 72 and 120** (0, 5, 15, 35, 60 and 100 % of 120):
  - The complex Ez phasor on the longitudinal mid-plane z = b/2 (the x–y plane; the TE10 field lies
    along z and does not vary along it). It is taken at 8.2, 10.3 and 12.4 GHz, and its centre line
    y = a/2 is also saved.
  - These come from a forward run driven from the left port alone, with the same mesh, absorber and
    record as the loop. The right port is left out: the εr = 9 fill runs into the absorber, which
    terminates it.
  - At keyframes 0, 60 and 120, the complex Jacobian dS11(f)/dεr (21 bins × 30 sections), taken as
    `jax.jit(jax.jacfwd(...))` without checkpointing. The first keyframe is timed; if three would
    take more than 2 GPU-hours, the rest are not run and the projection is reported.
  - Reported, not judged: dJ/dεr formed from the Jacobian, 2 mean(Re(S11* dS11/dεr)), against the
    recorded reverse-mode gradient at the same keyframe.
- **Beam, keyframes 0, 7, 21, 49, 84 and 140:**
  - The complex Ex and Ez phasors at f0 on the plane y = cy, the E-plane through the dipole. The
    plane spans the whole grid, so it covers the reflector, the cover and more than 1 λ above it.
  - D(θ, φ) on a 2° grid over the whole sphere. The plate is finite and the objective holds the back
    hemisphere down, so the back is kept.
- Files are complex64 or float32, one npz per case (`keyframes.npz`, with `keyframes.json`).

## 5. What this batch does not do
- It changes no solver code and no file under `validation/tmtt_paper/`.
- It does not quote or reproduce the paper's numbers. Every number comes from these runs.
- It runs no external solver, and it gives no realized gain or efficiency (directivity only).
