# Pre-declaration: two showcase designs from printable dielectrics (issue 1359)

**Status: PRE-DECLARATION. Written and pushed before any run of this batch.** Nothing below comes
from a run of this batch. Choices and thresholds are the lane leader's (showcase session,
2026-10-04, after the PI chose the examples). The implementer reads this file and never edits it;
a change is appended as a dated amendment by the leader, before the runs it governs.

Two designs an RF engineer would build, each with permittivity that a 3-D printer can make:

- **A. A graded-index (GRIN) flat lens** in front of a small reflector-backed dipole, at X band.
  Without the lens, the feed's beam is broad. A lens whose permittivity is highest at the centre
  and falls toward the rim delays the central rays more than the edge rays, flattens the phase
  front and narrows the beam, so the boresight directivity rises. The lens is printed from PLA
  at variable infill, which limits it to 1 ≤ εr ≤ 2.7.
- **B. A WR-90 bandpass filter** made of a ceramic-filled printed insert. Dielectric sections in
  the guide reflect partially; spaced right, the reflections form coupled resonances that pass a
  band and reject the rest. The insert's permittivity map is designed until |S11| and |S21| fall
  inside a specification mask. Ceramic-filled filament limits it to 1 ≤ εr ≤ 10.

Each design records every iterate (design, response, gradient), so that one clock can drive a
promotional render and an engineering figure set from the same numbers. The order is fixed: this
file → a 30-iteration trial on the design mesh (§3) → the full run.

Memory consulted (R1): `research_stop_differentiable_iris_gradient_m2.md` (metal shape gradients
in a waveguide are not the continuum shape derivative) — consistent: B moves no metal, only cell
permittivity. `research_stop_loaded_two_pole_m6.md` / `research_stop_iris_coupled_cavity_stage1.md`
(coupling-coefficient extraction in coupled-resonator filters failed) — consistent: B claims no
coupling coefficient, only the two-port S curves against a mask. #1359 record (rfx-archive
`20260928-design-beam-cells`): the stored final iterate was not the best (D(30°) 11.07 vs 11.72 dBi)
— answered here by a decaying step and best-iterate selection (§1.3, §2.3).

Every number below is **judged** (carries a threshold declared here) or **reported**. A number that
appears in promotional material must come from a run whose judged witnesses all passed; otherwise
the promotional material shows curves without that number.

## 1. A — GRIN flat lens

### 1.1 Structure
All coordinates lie on a 3.0 mm lattice so that every mesh of the ladder (1.5, 1.0, 0.75 mm)
places every face on a node line.

| item | value |
|---|---|
| design frequency band | 9.5, 10.0, 10.5 GHz (λ0 = 29.98 mm at 10 GHz) |
| feed | x-directed point dipole (Ex source), 9.0 mm (0.30 λ0) above a PEC sheet |
| reflector | PEC sheet, 30.0 × 30.0 mm (1.0 λ0), declared as a zero-thickness `Box` |
| lens | 90 × 90 mm (3 λ0) square, 30.0 mm (1.0 λ0) thick, bottom face 45.0 mm (1.5 λ0) above the dipole |
| lens permittivity | εr = 1 + 1.7 σ(ψ) ∈ [1, 2.7], lossless |
| design pixels | 3.0 × 3.0 × 3.0 mm (printable cell), 30 × 30 × 10; mirror symmetry in x and in y about the lens axis → 15 × 15 × 10 = 2250 free variables |
| boundaries | CPML on all six faces (10 layers at 1.5 mm, same thickness on finer meshes) |
| NTFF box | ≥ λ0/2 (15 mm) from the lens and the reflector on every face |
| design mesh | dx = 1.0 mm (λ0/30; 17 cells per wavelength inside εr 2.7 at 10.5 GHz) |
| record | `num_timesteps(num_periods=60)` at f_max = 12.6 GHz, lengthened only by §1.5's rule |

The mirror symmetries are those of the feed: the x-dipole over a centred square sheet radiates an
intensity pattern symmetric under x → −x and y → −y, so a boresight objective has a symmetric
optimum and the symmetry removes nothing the objective can use.

Probes for the settling witness (Ex): lens centre at mid-thickness; lens corner pixel at
mid-thickness; 7.5 mm above the lens top on axis; reflector edge (+x) 3 mm above the sheet.

### 1.2 Objective
L = −(1/3) Σ_f log D_f(θ = 0), with D_f = 4π U_f / P_rad,f from `compute_far_field_jax` on a
73 × 73 (θ, φ) grid (the beam-cover module's grid and quadrature). This is **directivity, not
realized gain**: the feed is a soft source and its match is not modelled. Every film and figure
says "directivity".

### 1.3 Optimiser
Adam, step size 0.10 with cosine decay to 0.01 over 150 iterations, float32. **Primary start:**
ψ = 0 everywhere (εr = 1.85, a uniform slab) — the film. **Secondary start (reported):** the
textbook GRIN profile below. **The design reported is the best iterate** by L (each iterate's L is
its own forward); the last iterate is also stored and reported.

### 1.4 Baselines (reported, same mesh and record)
- no lens (feed alone);
- uniform slab of the lens size, εr = 1.0, 1.85, 2.7 (best marked);
- **textbook GRIN**: n(r) = √2.7 − (√(F² + r²) − F)/t with F = 45 mm, t = 30 mm, r the radial
  distance from the axis at the pixel centre, εr = n² clipped to [1, 2.7], the same through the
  thickness (the path-length-equalizing flat lens).

### 1.5 Witnesses
1. **Gradient against central differences (judged), at the primary start.** dL/dεr of three
   free variables, each perturbing its pixel and that pixel's mirror images together: the pixels
   centred at x = 1.5, 22.5 and 40.5 mm from the axis, y = 1.5 mm, in the fifth pixel layer from
   the lens bottom. FD steps h = 0.1, 0.05, 0.025 in εr. Judged: |AD − FD|/|FD| ≤ 0.05 at h = 0.05 for
   each pixel with |FD(0.05)| ≥ 0.1 × the largest of the three; others reported. If any judged
   ladder is round-off dominated (FD moves more between 0.05 and 0.025 than between 0.1 and 0.05),
   the comparison is repeated in float64 in its own process, which records the field dtype and
   refuses unless all are float64; that one is then judged.
2. **Gradient record-length witness (judged), at the start and at the reported design.**
   `rfx.gradient_record_length_witness`, factor 1.5: ‖g1.5 − g1.0‖/‖g1.5‖ ≤ 0.05 over all 2250
   variables, and per FD pixel ≤ 0.05. A failure at the start stops the case before the descent.
   A failure at the reported design removes the gradient panel's colour scale from the film
   (the panel is shown with "not witnessed") and is reported.
3. **Settling, per read bin (judged), for the start, the reported design and every baseline.**
   `rfx.sparams._tail_witness.tail_share_witness` on the four probe records, the source-free
   window after the source ends, at the three read bins: the tail beyond the record adds at most
   1e-2 in amplitude to each bin's in-record DFT (the #1426 rule). Status "undetermined" is a
   failure. If the start or any baseline fails, the record is lengthened by 1.5× once, before the
   descent, and the new length governs every run of the case; a second failure stops the case.
4. **Mesh trend (judged for any promotional number).** The reported design, the no-lens feed,
   the best uniform slab and the textbook GRIN are re-solved without the optimiser at
   dx = 1.5, 1.0 and 0.75 mm (pixels are 2, 3, 4 cells; the map is the same physical map). For
   each design and bin, D(0°) is reported at the three meshes with the trend stated. Judged: the
   change between 1.0 and 0.75 mm is at most 0.3 dB and no larger than the change between 1.5
   and 1.0 mm. A promotional number is the 0.75 mm value.
5. **Aperture bound (reported).** D(0°) against 4πA/λ² with A the lens area (20.5 dBi at 10 GHz).
   A design above it is a red flag for the comparator, not a result.
6. **Preflight** output at each mesh quoted verbatim in the record.

## 2. B — WR-90 printed-insert bandpass filter

### 2.1 Structure
| item | value |
|---|---|
| guide | WR-90, 22.86 × 10.16 mm (y × z), PEC walls as domain faces, CPML along x |
| mesh | dx = a/36 = 0.635 mm (divides both walls exactly); re-solves at a/54 = 0.4233 mm and a/72 = 0.3175 mm |
| absorber | 45.72 mm of CPML at every mesh (72 layers at a/36; 0.75 λg at 8.2 GHz) |
| ports | waveguide ports 47 cells (29.845 mm at a/36) from each end, reference planes 63 cells (40.005 mm) from each end |
| insert | x from 59.69 to 140.97 mm (81.28 mm), the full cross-section; domain 200.66 mm |
| insert permittivity | εr = 1 + 9 σ(ψ) ∈ [1, 10], lossless |
| design pixels | 2.54 mm along x and y (4, 6, 8 cells on the ladder), uniform along z: 32 × 9 pixels; mirror symmetry about the guide centre y = a/2 → 32 × 5 = 160 free variables |
| band | 85 bins, 8.2–12.4 GHz (50 MHz) |
| S extraction | `compute_waveguide_s_matrix(normalize="flux")`, full two-port matrix |

Why these symmetries: an insert uniform along z and even about y = a/2 cannot couple the incident
TE10 to TE20 (odd in y) or to any mode varying along z, so the ports see TE10 alone over the band.
Even modes (TE30) can propagate inside εr = 10 material but are cut off in the empty guide
(19.7 GHz); between the insert and either reference plane lies 19.7 mm of empty guide, where TE30
at 12.4 GHz decays by about e^−6.3. Preflight's TE20 advisory at 12.4 GHz (> 0.9 f_c,TE20) is
expected and kept in the record with this disposition.

### 2.2 Specification mask and objective
Mask (judged against): passband 10.0–10.6 GHz, |S11| ≤ −15 dB; stopbands 8.2–9.0 GHz and
11.6–12.4 GHz, |S21| ≤ −25 dB. The optimiser aims 2 dB inside the mask:
J = mean_pass relu(20 log10|S11| + 17)² + mean_stop relu(20 log10|S21| + 27)², with |S| floored at
1e-6 inside the logarithm.

### 2.3 Optimiser and start
Adam, step 0.15 with cosine decay to 0.015 over 200 iterations, float32. The design reported is
the best iterate by J (ties: the earlier); the last iterate is stored too. Two candidate starts are
tried in the trial (§3): **S1** εr = 1.5 everywhere (an almost empty guide); **S2** three
full-width εr = 6 slabs, each 2 pixels long, centred at pixels 6, 16 and 26. The full run uses the
start whose trial reaches the lower J at iteration 30 (S1 if equal) — the rule is fixed now.

### 2.4 Baselines (reported)
The empty guide; the S2 start itself.

### 2.5 Witnesses
1. **Gradient against central differences (judged)** at the chosen start: dJ/dεr of three pixels
   on the guide centre line, at x-pixels 6, 16, 26; steps and rule as §1.5.1. Because J is zero
   where the mask is met, if a pixel's |FD(0.05)| is below 0.1 × the largest it is reported only.
2. **Gradient record-length witness (judged)** at the start and at the reported design, as
   §1.5.2, on all 160 variables.
3. **Settling, per read bin (judged)** for the start, the reported design and the baselines: the
   per-bin tail-share witness that `compute_waveguide_s_matrix` records for both runs of each drive
   (`rfx.sparams._tail_witness`), at every one of the 85 bins: ≤ 1e-2, status "undetermined" is a
   failure. On failure at the start, the record is lengthened by 1.5× once before the descent; a
   second failure stops the case. At the reported design: the record is lengthened (1.5×, then
   2×) until the witness passes, and the design is re-solved there; if 2× still fails, the
   promotional material shows the curves without numbers. (Ring-down completion is a wire-port
   path today and is not used on the waveguide lane.)
4. **Power balance (judged)** for the reported design at every bin and every mesh:
   0.98 ≤ |S11|² + |S21|² ≤ 1.01 (lossless insert, PEC walls). A bin outside is a comparator
   problem first; the design is not reported from that mesh until it is explained.
5. **Mesh trend (judged for any promotional number).** The reported design at a/36, a/54, a/72:
   the passband edges (first and last bin with |S11| ≤ −15 dB), the worst passband |S11| and the
   worst stopband |S21| with the trend stated. Judged: the passband-edge frequencies move by at
   most 1 % between a/54 and a/72 and by no more than between a/36 and a/54; the mask verdict is
   stated per mesh. A promotional "inside the mask" claim needs the a/72 curve inside the mask.
6. **Cross-check of normalization (reported):** the reported design with `normalize=False`.
7. **Preflight** quoted verbatim at each mesh.

## 3. Trial before the full run (both cases)
On the design mesh, at the declared record, 30 iterations of the declared optimiser (step
schedule truncated, not rescaled), with every iterate recorded. Purpose: does the descent move?
This is diagnostic; no trial number is published. Continue to the full run only if:
- A: the band-mean boresight D at iteration 30 is ≥ 2 dB above the start's, no NaN;
- B: J at iteration 30 is ≤ 0.5 × J at the start for the chosen start, no NaN.
If not, the case stops and the leader writes an amendment before any further run.

## 4. Compute and records
VESSL only, with a `Resource choice:` record per submission (rfx-archive `RESOURCE_CHOICE.md`).
The lab rule holds: when the chosen preset has pending jobs, at most two GPU runs of this project
at a time. Estimated scale (to be replaced by the trial's timing): A ≈ 3.2 M cells × about 2 600
steps per gradient at 1.0 mm; B ≈ 0.12 M cells × about 2 × 15 000 steps per gradient at a/36.
Records go to rfx-archive `rfx/records/20261004-showcase-lens/` and `.../20261004-showcase-filter/`,
schema `rfx-showcase-result/1`, rewritten after every iterate. Promotional originals (4K stills,
video) go to the private `bk-squared/rfx-research` repository, branch `showcase`, not to this
repository.

## 5. Two outputs from one record
- **Promotional** (PyVista for clips and site loops, Blender Cycles for hero stills): geometry,
  response and gradient on one clock; colour and transparency of a pixel follow its εr; the lobe
  radius follows linear directivity; every non-data property (plate thickness, rod size, colours)
  is a display convention and is listed in the render manifest.
- **Document** (SciencePlots pubstyle, no titles, PDF + PNG; TikZ for the geometry drawing):
  geometry with dimensions; response with baselines and mask; objective history with the best
  iterate marked; gradient checks (FD and record length) in separate panels; mesh trend.

## Amendment 1 (2026-10-04, leader; before any run)
Two points the text left open, found by the implementer before any run:
1. **B's starting record.** `num_timesteps(num_periods=240)` at f_max = 1.05 × 12.4 GHz =
   13.02 GHz (about 18.4 ns, about 15 000 steps at a/36), rounded up to a multiple of the
   checkpoint-segment count (the largest divisor of the step count not above its square root).
   §2.5.3 governs any lengthening. The finer meshes use the same physical record length.
2. **Pixel indexing for B.** Pixels are numbered 1…32 along +x from the port-1 end of the insert.
   Start S2's slabs occupy pixels {6, 7}, {16, 17} and {26, 27} (mirror-symmetric about the
   insert centre), the full guide width. §2.5.1's FD pixels are pixels 6, 16 and 26 on the guide
   centre line (the centre pixel of the 9 across the width).

## Amendment 2 (2026-10-04, leader; after the CPU smokes, before any VESSL run)
1. **Record rounding (both cases).** The declared record `num_timesteps(...)` is rounded up to a
   multiple of ⌊√n⌋, so that reverse-mode checkpointing keeps about √n segments. Without it the
   lens records (2498 steps at 1.0 mm, 3331 at 0.75 mm) have no divisor near √n and would hold
   1–2 segments. Resulting records: lens 1680 / 2499 / 3363 steps at 1.5 / 1.0 / 0.75 mm; filter
   15 252 / 22 952 / 30 624 steps at a/36 / a/54 / a/72.
2. **B's ports and reference planes** move to 48 and 64 cells at a/36 (30.48 mm and 40.64 mm from
   each end), so that they lie on node lines at a/54 as well (47 and 63 cells fell on half nodes
   there). Between the reference planes and the insert lie 19.05 mm of empty guide; TE30 at
   12.4 GHz decays there by about e^−6.
3. **S2's background** between its slabs is εr = 1.5 (S1's value).
4. **Probes** of A are placed at the declared points and their realized positions are recorded;
   they are not required to lie on the 3 mm lattice (they read fields, they are not geometry).
5. **The feed reflector's solved size.** Preflight reports that the solver extends a PEC sheet's
   edge 0.35 cell past its last node, so the drawn 30.0 mm sheet is solved as 31.05 / 30.7 /
   30.52 mm on the 1.5 / 1.0 / 0.75 mm meshes (issue #1138's sheet-edge offset). The sheet is not
   moved; §1.5.4's mesh trend therefore includes a reflector converging toward its drawn size, and
   the record states this beside the trend.
6. **Run outputs** are written on NFS under `byungkwan-workspace/rfx-showcase-runs/` (outside every
   git clone) and copied into rfx-archive by the leader; the job clones the public repository from
   GitHub at the given commit.

## Amendment 3 (2026-10-05, leader; after the PI's review of the first showcase, before the runs it governs)
The PI judged that start S2 (three εr = 6 slabs) hands the optimiser the resonator layout, so a filter
grown from it does not show what the gradient finds. The S2 run and its three-mesh re-solve stay in the
record unchanged; this amendment adds a second filter run and a rule for the frame the showcase shows.
1. **Filter run F2, from the uniform start S1** (εr = 1.5 in every pixel). Same structure, mask,
   objective (§2.2), record (§2.5.3 rules), witnesses (§2.5) and three-mesh re-solve (a/36, a/54,
   a/72) as the first run. Optimiser: Adam, step 0.15 with cosine decay to 0.015 over **400**
   iterations (the S1 trial reached J = 24.3 at iteration 30, against 11.5 for S2, so it is given
   twice the budget); the design reported is the best iterate. The S1 start's FD and record-length
   witnesses were measured in the trial (369367267413, `S1/`) and are reused; they are not re-run.
   **Outcome rule, fixed now:** if F2's reported design is inside the mask on a/72 and passes §2.5's
   witnesses, it becomes the showcase filter. If not, the showcase shows F2's curves without the
   words "inside the mask", and the result is reported as it is; S2 is not substituted.
2. **The showcase frame's iterate k\*.** Geometry, response and gradient are shown at one iterate
   chosen by this rule, not by eye: with the progress fraction
   p_k = (L_0 − L_k)/(L_0 − L_best) for the lens (L is already a log-directivity) and
   p_k = (log J_0 − log J_k)/(log J_0 − log J_best) for the filter, k\* is the smallest k with
   p_k ≥ 0.5. The frame also shows the best design's response faintly as the destination.
3. **Gradient witness at k\*.** The gradient shown at k\* gets its own record-length witness
   (`rfx.gradient_record_length_witness`, factor 1.5, ≤ 0.05 in norm over all variables). If it
   fails, the frame uses the last iterate before k\* whose witness passes (searched downward in steps
   of 10, ending at the start, whose witness passed), and the frame's caption names the iterate used.

## Amendment 5 (2026-10-08, leader; before any run it governs) — two more objectives on the same lens
(Number 4 is not used in this note.) The PI asked that the showcase state its optimisation problem
plainly and show that the same code serves other objectives. The first lens run (§1) maximises
boresight directivity. This amendment adds two runs on the **same structure, variables, bounds, mesh,
record rules, optimiser and start** (§1.1, §1.3: 2,250 free values, 1 ≤ εr ≤ 2.7, Adam 0.10 → 0.01 over
150 iterations, primary start εr = 1.85) that differ only in the objective. Both objectives read the
directivity pattern D_f(θ, φ) in dBi that §1.2 already computes on the 73 × 73 grid (θ every 2.5°), at
the three frequencies 9.5, 10.0, 10.5 GHz; nothing in the solver or the far-field transform changes.

Where the first design stands (stored record, 10 GHz, 1.0 mm mesh): boresight 19.0 dBi; the largest
directivity in any direction with θ ≥ 25° is 3.8 dBi (15.3 dB below the peak); the pattern is 15.0–13.7 dBi
at θ = 10° and 9.2–6.0 dBi at θ = 15°.

**Notation.** relu(x) = max(x, 0). For a set of grid directions R, Q_R(v) = sqrt(mean_R relu(v)⁴ + δ⁴) − δ²
with δ = 0.1 dB: a smooth measure of the worst violations in R (dB²), zero when none.

1. **Objective M — a pencil beam under a flat sidelobe mask.**
   Mask (judged against, at each of the three frequencies): D(θ = 0) ≥ 18.0 dBi, and D ≤ −2.0 dBi in
   every grid direction with θ ≥ 25° (sidelobes and back radiation at least 20 dB below an 18 dBi beam).
   The optimiser aims 1 dB inside:
   J_M = mean_f [ relu(19.0 − D_f(0))² + Q_S(D_f + 3.0) ], S = {θ ≥ 25°}.
2. **Objective F — a flat-top beam.**
   Mask (judged against, each frequency): 13.0 ≤ D ≤ 15.0 dBi in every grid direction with θ ≤ 12.5°, and
   D ≤ 0.0 dBi in every grid direction with θ ≥ 32.5°; the band between is free. The optimiser aims
   0.5 dB inside the window and 1 dB inside the outer bound:
   J_F = mean_f [ mean_C (relu(13.5 − D_f)² + relu(D_f − 14.5)²) + Q_O(D_f + 1.0) ],
   C = {θ ≤ 12.5°}, O = {θ ≥ 32.5°}.
   (Power check: 14 dBi over a 12.5° cone carries 30 % of the radiated power, so the window is not
   excluded by conservation; whether a 3 λ0 aperture of εr ≤ 2.7 reaches it is what the run finds.)
3. **Design reported and frame iterate.** The best iterate by J. The progress fraction for the frame
   rule of Amendment 3 is p_k = (log(J_0 + ε) − log(J_k + ε)) / (log(J_0 + ε) − log(J_best + ε)) with
   ε = 1e-3 dB² (J can reach zero).
4. **Witnesses.** §1.5 items 1–3 and 6 unchanged in form, with L replaced by J_M or J_F: gradient
   against central differences at the start on the same three pixels; gradient record-length witness
   at the start and at the reported design (and at every iterate a film shows, Amendment 3.3);
   per-bin settling for the start and the reported design. A 30-iteration trial (§3) comes first and
   carries the start witnesses.
   **Mesh trend (judged for any promotional number), replacing §1.5 item 4 for these runs:** the
   reported design re-solved at 1.5, 1.0 and 0.75 mm. Reported per mesh and frequency: D(0), the
   largest D in the outer region (S or O), and for F the smallest and largest D in C. Judged: between
   1.0 and 0.75 mm, D(0) and the two window extremes change by at most 0.3 dB and the largest outer-region
   level by at most 1.0 dB, and no change is larger than the one between 1.5 and 1.0 mm. A promotional
   number is the 0.75 mm value. §1.5 item 5 (aperture bound) applies to D(0) of M.
5. **Outcome rule, fixed now.** A run "meets its mask" only if the reported design satisfies every
   mask inequality at all three frequencies on the 0.75 mm mesh and passes the witnesses. Otherwise the
   showcase shows the achieved pattern against the mask without those words, with the achieved worst
   values, and the result is reported as it is. The mask levels above are not adjusted after a run;
   a different mask is a new amendment decided with the PI.
6. **What is shown.** For each run: the frame at k\* and at the reported design, the pattern cut against
   its mask, the three-mesh values. The first run's film is unchanged; the two runs appear beside it as
   "the same two passes, another objective". Every showcase output names its objective in words before
   any result.
7. **Memory read (R1).** Ledger: the lens run's own entries (settled at the declared record on every
   mesh; gradient witnesses pass at all 151 iterates) — consistent, the runs reuse that record length
   and re-witness it; the filter's fixed-record limitation ("inverse design on a fixed record can descend
   into designs that do not settle on that record") — applies here too, which is why item 4 keeps the
   per-bin settling witness on the reported design and the gradient witness on every shown iterate.

**Amendment 5, clarifications (2026-10-08, leader; found while the driver was written, before any run).**
(a) The regions S, C, O are defined on the grid's nominal angles (multiples of 2.5°). The grid's rows
sit up to 1e-4 rad from those angles, and a literal comparison would have left the row at nominal
12.5° out of C; C is the six rows 0°–12.5° (432 directions), S the rows from 25°, O the rows from 32.5°.
(b) §3's continuation rule A (band-mean boresight ≥ 2 dB above the start's at iteration 30) belongs to
the boresight objective. The trials of objectives M and F continue to the full run by rule B's form:
J at iteration 30 ≤ 0.5 × J at the start, no NaN.

## Amendment 6 (2026-10-09, leader; before any run it governs) — a wide flat-top beam; objective F withdrawn
**Why.** Objective M (Amendment 5) ran and did not meet its mask: the largest directivity beyond 25° stayed
4.5–4.9 dB above the −2 dBi ceiling at 1.0 mm because the skirt of a 3 λ0 aperture's main beam itself crosses
25°; the mask had been written without reference to the aperture. Objective F never ran its descent (the
start check's float64 repeat did not fit a 48 GB card). The PI then asked that further objectives be derived
from the literature, and chose a **wide flat-top beam (wide field of view)**. Objective F (13–15 dBi inside
12.5°) is withdrawn; objective M stays reported as it ran. This amendment declares objective W on the
**same structure, variables, bounds, mesh, record rules, optimiser and start** as §1 (2,250 free values,
1 ≤ εr ≤ 2.7, Adam 0.10 → 0.01 over 150 iterations, start εr = 1.85), reading the same directivity pattern
D_f(θ, φ) in dBi on the 73 × 73 grid at 9.5, 10.0, 10.5 GHz.

**1. What "wide flat-top" means in the literature read** (23 sources, 14 in full text; record: private archive,
`rfx/records/20261004-showcase-lens/objective-survey/extraction_E_wide_flat_top.md`; two papers' numbers
re-read by the leader in the downloaded originals: Kumar & Ghosh, PIER Letters 100, 2021; Hong & Nguyen,
J. Electromagn. Eng. Sci., 2026). For lenses, metasurfaces, transmitarrays and superstrates of 1.8–18 λ over
one feed: the flat region is stated as the angular width within 1 dB (sometimes 2 dB) of the flat level;
achieved half-widths run from ±10° to ±30°, the widest read in full text being ±29°/±30° at −1 dB from a 1.76 λ
aperture with 7.5 dBi realised gain; ripple 1–2 dB; edge roll-off (−3 → −15 dB) 18°–29°; level outside the
sector −8 to −15 dB relative to the flat level; flat-region gain 7.5–14 dBi. Wider sectors (±45° constant-flux,
±60°) appear only as templates for an 18 λ aperture or as targets without achieved numbers.

**2. What this block can support (checked before declaring).**
(a) *Geometry.* The source sits 45 mm below the block's lower face and 75 mm below its upper face; the block
is 90 mm wide. In a principal plane, rays within atan(45/75) = 31° of boresight cross the full thickness,
rays between 31° and 45° leave through a side wall, and rays beyond 45° do not enter the block.
(b) *Path budget.* The block can add at most 30 mm × (√2.7 − 1) = 19.3 mm (0.64 λ0) of path difference.
Collimating the source fully needs √(45² + 45²) − 45 = 18.6 mm at the edge of the lower face (the first design
uses nearly all of the budget); a beam that stays spread needs less correction than full collimation.
The path budget does not limit a ±30° sector.
(c) *Aperture.* A 3 λ0 aperture gives an edge slope of about λ0/L = 19°; the sector edge cannot be sharper.
(d) *Power.* An ideal conical sector of half-angle 30° has 11.7 dBi; about 90 % of the radiated power goes
into the upper hemisphere (stored patterns), so the ceiling is about 11.3 dBi.
(e) *Where the designs stand* (stored 1.0 mm records; smallest / largest directivity over all grid directions
with θ ≤ 30°, at 9.5 / 10 / 10.5 GHz): source over its plate without the block 4.9 / 4.8 / 4.7 to 7.5 / 7.5 / 7.4 dBi;
uniform start 2.7 / 0.4 / −1.9 to 12.6 / 12.2 / 12.4 dBi; first (boresight) design −4.9 / −8.1 / −14.0 to
18.6 / 19.0 / 18.9 dBi.
(f) *What the block cannot do.* The source alone radiates 3.1–3.7 dBi in some directions with θ ≥ 60°
(the plane across the dipole), by rays that never meet the block. The level outside the sector is therefore
not something this block controls, and a sector wider than about ±31° asks the block to lift directions its
full thickness does not reach; ±45° would need a wider block or a closer source. **The sector declared is
±30°.**

**3. Objective W.** C = the 13 grid rows with nominal θ ≤ 30° × the 72 distinct φ columns = 936 grid
directions (the duplicated φ endpoint dropped, nominal angles as in Amendment 5 clarification (a); the
θ = 0 row enters as 72 copies of one direction, which only weights boresight slightly in the smooth minimum). For each frequency the smooth minimum over C,
m_f = −τ · ln( mean_C exp(−D_f / τ) ), τ = 0.5 dB,
and **J_W = − mean_f m_f** (dB, minimised): raise the weakest direction of the sector. Because the pattern
integrates to a fixed total, raising the minimum flattens the sector and moves power into it; no level is
prescribed and no term acts outside the sector (item 2(f)).

**4. Judged, at each of the three frequencies, on the 0.75 mm mesh** (hard minimum and maximum over all grid
directions of C, not the smooth m_f):
- **ripple:** max_C D − min_C D ≤ 2.0 dB (the literature's 1–2 dB);
- **level:** min_C D ≥ 7.5 dBi (every direction of the ±30° cone at least as high as the source's own
  boresight directivity, 7.5 dBi, and equal to the gain the widest flat-top read reports).
These two conditions hold over the whole cone, all φ, at three frequencies. The literature's numbers are per
principal plane at one frequency, so the bar is stricter than any paper read; the source alone has 2.6 dB of
ripple over this cone. A result slightly outside is reported with its numbers under item 6.
**Reported, not judged:** the largest D with θ ≥ 60° and its difference from min_C D (literature −8 to −15 dB;
item 2(f) says why it is not a bar here); the angle between the −3 dB and −15 dB points relative to min_C D
in the two principal planes (literature 18°–29°); boresight D; the fraction of radiated power inside the cone.

**5. Trial, witnesses, mesh trend.** As Amendment 5 items 3–4 with these replacements. Trial continuation:
J_W at iteration 30 ≤ J_W at the start − 2.0 dB, no NaN. Progress fraction for the frame rule:
p_k = (J_0 − J_k) / (J_0 − J_best). Mesh trend (1.5, 1.0, 0.75 mm): min_C D and max_C D change by at most
0.3 dB between 1.0 and 0.75 mm and by no more than between 1.5 and 1.0 mm; the reported outside level by at
most 1.0 dB. Gradient against central differences at the start on the same three pixels (float64 repeat
if a ladder is round-off dominated, on a card that holds it); gradient record-length witness at the start, at
the reported design and at every iterate a film shows; per-bin settling at the start and the reported design.

**6. Outcome rule, fixed now.** The words "a ±30° flat-top beam" are used only if both judged conditions of
item 4 hold at all three frequencies on the 0.75 mm mesh and the witnesses pass. Otherwise the showcase shows
the achieved pattern with its ripple and minimum, without those words. The sector, τ and the two levels are
not adjusted after a run; a different sector is a new amendment decided with the PI.

**7. What is shown.** The pattern cut over the ±30° sector with the 2 dB band drawn at the achieved level,
for the source alone, the first design and this design; the three-mesh values; the frames of Amendment 3.
The first design's film is unchanged; this design appears after it as "the same two simulations per iteration,
another objective", with its objective named in words before any result.

**8. Memory read (R1).** Session memory `objective-feasibility-before-predeclare` (a mask declared without
aperture physics failed; check beamwidth, geometry and the literature's own terms first) — item 2 is that
check. Ledger: the lens entries (settled at the declared record on every mesh; gradient witnesses pass at all
151 iterates of the first run) — consistent, the run reuses the record length and re-witnesses it.

**Amendment 6, note (2026-10-09, leader; geometry only, no run) — what a wider sector would take.**
Asked by the lead while the PI decides whether ±30° on the present block is the wanted "wide" field of view.
Full-thickness coverage half-angle θc of a block of width W and thickness t = 30 mm whose lower face is h above
the source: tan θc = (W/2) / (h + t). Present block: W = 90 mm, h = 45 mm, θc = 31°. Cell counts are for the
driver's domain rule (18 mm of air round the block and above it, plate 18 mm above the floor, source 9 mm above
the plate, 15 mm absorber on every face). Memory is scaled by cell count from the present 1.0 mm run (26.5 GB
of gradient arguments reported by the solver; about 40 GB for the 1.5× record of the gradient witness, derived);
both scaled columns are estimates, not measurements. The 0.75 mm mesh is used only for the forward re-solve of
the reported design (no gradient), which needs a few GB at any of these sizes.

| target θc | change | W (mm) | h (mm) | free values | cells at 1.0 mm (ratio) | gradient / witness memory (est.) | cells at 0.75 mm | fits |
|---|---|---|---|---|---|---|---|---|
| 31° (present) | none | 90 | 45 | 2,250 | 3.72 M (1.00) | 26.5 / 40 GB | 8.8 M | 48 GB card, except the float64 start check |
| 45° | wider block | 150 | 45 | 6,250 | 7.11 M (1.91) | 51 / 76 GB | 16.8 M | 141 GB card |
| 45° | closer source | 90 | 15 | 2,250 | 2.98 M (0.80) | 21 / 32 GB | 7.0 M | 48 GB card |
| 60° | wider block only | 260 | 45 | 18,490 | 16.2 M (4.34) | 115 / 174 GB | 38.0 M | no (witness exceeds 141 GB) |
| 60° | closer source and wider block | 156 | 15 | 6,760 | 6.02 M (1.62) | 43 / 65 GB | 14.2 M | 141 GB card |

The float64 repeat of the start check asked for one 29.8 GiB block on the present grid and did not fit 48 GB;
scaled by 1.9 it is about 57 GiB plus the rest of that run, which should fit 141 GB but is not measured.
Time per iteration scales with the cell count (11.3 s measured on the present grid).

What changes besides size. (i) Moving the source to 15 mm (0.5 λ0) below the block puts the block in the
source's near field: the ray picture above is then only a guide, and the first (boresight) design no longer
shares the structure, so "the same structure, another objective" would need the boresight run repeated on the
new geometry or the two shown as different structures. (ii) A wider block keeps the source distance and the
first design's structure class but not its size; the boresight run would also have to be repeated at 150 mm
for a like-for-like pair, and its directivity would rise with the aperture. (iii) Power: the ideal conical
sector is 8.3 dBi at ±45° and 6.0 dBi at ±60°, against 7.5 dBi at the source's own boresight and 1.5 dBi /
−3.4 dBi at its 45° / 60° directions (worst plane, stored pattern); a ±60° flat-top therefore lowers the
centre below the source's own boresight level to lift the edges by 8–9 dB. (iv) The literature read gives no
achieved flat-top wider than ±30° from an aperture under 5 λ0 in full text (wider ones are templates for 18 λ0
or targets without numbers). None of these geometries is declared; a change of geometry is a new amendment.

## Amendment 7 (2026-10-11, leader; declared before any run) — objective N: a fan beam, wide in one plane and narrow in the other

**1. Why.** Objective W (Amendment 6) ended with 4.5–5.3 dB of ripple over the ±30° cone on the 0.75 mm mesh and
is not shown as a flat-top beam (item 6 there). The PI then stated what "wide field of view" means for this
example (2026-10-11): a wide beamwidth, as in automotive radar, where the antenna covers a detection region
shaped on purpose — wide in azimuth, narrow in elevation — and the radome has to support that pattern. Objective
N is that pattern on the same block and source: flat over ±30° in one principal plane, narrow in the other.

**2. What was wrong with Amendment 6, kept here so it is not repeated.** (a) The objective raised a soft minimum
only while the judged quantity was the ripple; nothing pushed the maximum down, and the design kept a boresight
peak. (b) The level bar (7.5 dBi) came from the source's boresight directivity, not from the power the sector
can hold. (c) The sector was a cone, not the shaped region the PI meant. For objective N the objective contains
the judged ripple, and the levels below come from a power and aperture estimate made before the run.

**3. Estimate made before the run (no FDTD run; scripts and numbers in the private record).** A thin-screen model
of the block's top face (90 × 90 mm, amplitude from the stored source-alone pattern, a delay of 0–19.3 mm per
3 mm pixel, the same delay at the three frequencies, scalar far field) was calibrated against two stored FDTD
designs (model minus FDTD: 1.4–2.1 dB; 1.5 dB used). Driven with Amendment 6's objective and recipe the model
reproduces that run's boresight peak and ripple (3.5–4.8 dB against 4.6–5.5 dB), which is the evidence for 2(a).
Driven with objective N (item 4, τ = 0.2 dB, 300 steps from the uniform start) it gives, at 9.5 / 10 / 10.5 GHz:
strip ripple 1.2 / 0.7 / 0.8 dB; strip minimum 13.2 / 13.6 / 13.9 dBi in model terms, 11.7 / 12.1 / 12.4 dBi after
calibration; half-power beamwidth in the narrow plane 20.3° / 18.1° / 16.3° (a uniformly lit 90 mm aperture
gives about 17° at 10 GHz). The model ignores the block's thickness and side faces; it is an estimate.

**4. Objective N.** Wide plane: φ = 90° / 270° (the plane across the source current, where the source alone is
widest: 106°–114° at −3 dB against 68°–70° in the φ = 0° plane). Strip S_N: the 13 θ rows with nominal θ ≤ 30°
and the six φ columns 85°, 90°, 95°, 265°, 270°, 275° (indices 17, 18, 19, 53, 54, 55); 78 directions. With
D_f in dBi over S_N (linear floor 10⁻³⁰), τ = 0.2 dB,
smin_f = −τ·ln(mean exp(−D_f/τ)), smax_f = τ·ln(mean exp(D_f/τ)),
**J_N = mean over the three frequencies of (smax_f − smin_f) − smin_f.**
The first term is the smooth ripple on the strip; the second raises the strip's level, which by power
conservation narrows the other plane. τ and the weights are not adjusted after a run. Start, bounds, mirror
symmetry, optimiser and step schedule as the first design; **300 iterations**.

**5. Judged, on the 0.75 mm mesh, at each of the three frequencies:**
- **ripple:** max − min of D over S_N ≤ 2.0 dB;
- **level:** min of D over S_N ≥ 11.0 dBi (estimate 11.7–12.4 dBi; 0.7 dB allowed for the model);
- **narrow plane:** half-power beamwidth in the φ = 0° / 180° cut ≤ 23.0° (estimate 16°–20°), measured from the
  boresight value: the sum of the first −3 dB crossings on the two sides, linear in θ on the nominal grid.
**Reported, not judged:** the −3 dB width in the wide plane; the largest D with θ ≥ 60°; boresight D.

**6. Trial, witnesses, mesh trend.** As Amendment 6 item 5 with these replacements. Trial continuation: J_N at
iteration 30 ≤ J_N at the start − 3.0 dB, no NaN (model: −9 to −11 dB; the stored start gives J_N = 8.087).
Progress fraction p_k = (J_0 − J_k) / (J_0 − J_best). Mesh trend (1.5, 1.0, 0.75 mm): strip minimum and maximum
change by at most 0.3 dB between 1.0 and 0.75 mm and by no more than between 1.5 and 1.0 mm; the narrow-plane
beamwidth by at most 1.0°; the reported outside level by at most 1.0 dB.

**7. Outcome rule, fixed now.** The words "a fan beam, flat within 2 dB over ±30°" are used only if the three
judged conditions hold at all three frequencies on the 0.75 mm mesh and the witnesses pass. If the ripple
condition fails, the thick block does not behave as the thin screen of item 3 and no further objective is run on
this block without a new argument put to the PI (this is the third objective on it: Amendments 5, 6 and 7).
If only the level or the beamwidth is slightly outside, the pattern is shown with its numbers and without those
words.

**8. What is shown.** The two principal-plane cuts with the 2 dB band, for the source alone, the first design
and this design, and the pattern over the upper half space; the three-mesh values; the frames of Amendment 3.

**9. Memory read (R1).** Session memory `objective-feasibility-before-predeclare` (check aperture physics and the
literature before declaring) — items 2–3 are that check, and this time the objective itself was tried in a model
before declaring it. Ledger: lens entries as in Amendment 6 item 8 — consistent.
