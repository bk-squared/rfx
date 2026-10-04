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
