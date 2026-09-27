# Pre-declaration: showcase results from the current code (issue 1332, first batch)

**Status: PRE-DECLARATION. Written and pushed before any job of this batch was submitted.**
Nothing below comes from a run of this batch. The thresholds are the lane leader's, set in the
brief for issue 1332 on 2026-09-27; this note copies them. It defines no threshold of its own.

Four results, each with a check that does not use the gradient to check itself:

1. A probe-fed patch on RT/Duroid 5880. One reverse-mode pass through the 3-D FDTD solve gives
   the derivative of the reflected power |S11(f_t)|^2 with respect to the permittivity of each
   substrate cell. The derivative is checked against central finite differences on six cell
   blocks chosen here, before the run.
2. A three-layer anti-reflection coating on eps_r = 12 over 8-12 GHz. Adam descends through the
   FDTD solve toward the transfer-matrix (TMM) optimum. Checks: the example's own gates, the
   derivative against finite differences at the start point, and FDTD R(f) against TMM R(f).
3. What one gradient of the patch costs in wall time and device memory, for three design-box
   sizes on one pinned GPU model. The finite-difference cost is derived from the measured
   forward time, not measured.
4. The rfx curves behind three existing cross-validation comparisons, saved as arrays at the rung
   each test judges. The verdicts come from the unchanged pytest ladders run at the same commit.

Scripts: `scripts/showcase/{_record,patch_sensitivity,ar_coating,forward_curves,render}.py`.
Every job writes `result.json` (schema `rfx-showcase-result/1`, `scripts/showcase/_record.py`)
beside its arrays. In `result.json` a claim is either **judged**, with the threshold named here,
or **reported**.

## 1. Patch: sensitivity map of |S11(f_t)|^2

**Structure and mesh.** The board of `tests/crossval/rt5880_patch/test_rt5880_patch.py`, built
by that module's own `build(dx)` and imported by path, read-only. The board: a 39.7 × 49.25 mm
patch on 3.175 mm of eps_r 2.2 / tanδ 0.001, a 55.6 × 65.1 mm ground, and a 50 Ω wire-port probe
8.73125 mm off centre along the resonant length (x). The patch and the ground are zero-thickness
PEC sheets. The mesh is uniform, dx = h/4 = 793.75 µm, the ladder's coarsest rung: 4 cells
through the substrate, grid 145 × 161 × 85 including the CPML. The module's `assert_realized`
must pass before anything runs. The record length is the module's `NUM_PERIODS` = 60 periods of
4 GHz, 9912 steps. Precision is float32 unless §1.5 triggers the float64 repeat.

**Indices.** The patch centre is node (I, J) = (72, 80) at h/4. Cell i spans nodes i to i+1. The
patch covers cells I−25 … I+24 along x and J−31 … J+30 along y. The laminate covers cells 37 … 107
along x and 39 … 121 along y. The substrate is cells k = 36 … 39. The probe column is (61, 80).

### 1.1 Baseline (step a)
The module's own `run_rung(h/4)` supplies S11(f) on the module's 901 bins over 1.6-3.4 GHz, its
ring-down settling witness, and its passivity witness. Its preflight output is kept in the job
log. The job saves S11(f) and Zin(f), and writes a Touchstone `.s1p`.

The two resonance estimators of the module both run and are both reported:
- `resonance()`: the |S11| minimum, `refined_extremum` on log|S11|.
- `refined_remax`: the Re(Zin) peak, the frequency the cross-validation test judges.

Also reported: max |S11_forward − S11_run| over the 901 bins. This compares `forward(port_s11_freqs=…)`
on the same board with `run()`. It shows that the differentiable path reads the S11 that `run()`
reads.

### 1.2 f_t rule (step b)
**f_t = 1.01 × f_r.** f_r is the |S11| minimum from the module's `resonance()`, the objective
being |S11|^2. The Re(Zin) peak is reported beside it.

### 1.3 Objective and design region (step c)
**J = |S11(f_t)|^2.** J comes from `Simulation.forward(port_s11_freqs=[f_t])`: the wire port's
driven reflection, from its whole-gap voltage and its current.

**Design box.** The full substrate thickness (k = 36 … 39), over the patch footprint plus a
margin of 2·h = 8 cells on every side, clipped to the laminate. That is cells 39 … 104 along x
and 41 … 118 along y, 66 × 78 × 4 = 20 592 cells, every one of them laminate. The call is
`forward(design_box=…, design_eps_override=…, design_box_holds_ports=True, checkpoint=False)`.
The design values are the drawn eps_r = 2.2, and the conductivity stays the drawn one. The
probe's own edges are held: they keep the drawn coefficients, and their derivative is exactly
zero by construction.

**Gradients computed.** Each is saved with its cell coordinates.
- A1 check: a 600-step record. It must be finite and nonzero, otherwise the job stops and
  reports.
- The full record: 9912 steps.
- The record-length witness: 1.5 × the record, 14 868 steps. The rfx rule says a gradient is
  reportable only beside the same gradient from a longer record. **Reported**: the per-block
  ratio of the 1.5× sum to the 1.0× sum, and the Pearson correlation of the two maps.

### 1.4 Finite-difference blocks (step d)
Six blocks, each 3 × 3 cells in plane (2.38 × 2.38 mm, the whole-cell size nearest to 2 mm)
through all 4 substrate cells, so 36 cells each. Offsets are inclusive, in cells, from the patch
centre node (I, J):

| block | x cells | y cells | where |
|---|---|---|---|
| edge_minus_x | I−25 … I−23 | J−1 … J+1 | inside the patch, at the −x radiating edge |
| edge_plus_x | I+22 … I+24 | J−1 … J+1 | inside the patch, at the +x radiating edge |
| centre_plus_y | I−1 … I+1 | J+14 … J+16 | on the centre line between the radiating edges, +y side |
| centre_minus_y | I−1 … I+1 | J−17 … J−15 | on the centre line, −y side |
| margin_plus_x | I+25 … I+27 | J−1 … J+1 | outside the patch, beyond the +x radiating edge |
| margin_plus_y | I+9 … I+11 | J+31 … J+33 | outside the patch, beyond the +y edge |

A 3-cell block cannot be centred on a node. The two centre-line blocks therefore span −0.79 …
+1.59 mm about the centre line x = 0. None of the blocks contains the probe column.

**FD** is the central difference of J for a uniform eps_r change Δ of every cell in the block.
**AD** is the sum of the full-record gradient over the block's 36 cells. The step ladder is
Δ = 0.2, 0.1, 0.05.

**Judged:** |AD − FD| / |FD| ≤ 0.05 at Δ = 0.1, for every block whose |FD(0.1)| ≥ 0.1 × the
largest |FD(0.1)| of the six. The other blocks are **reported**.

### 1.5 Float64 repeat
A block's ladder counts as round-off-dominated when FD moves more between Δ = 0.1 and 0.05 than
between 0.2 and 0.1. The reason: truncation error falls as the step shrinks, and round-off error
grows.

If any judged block's ladder is round-off-dominated, the same job repeats the AD block sums and
the whole FD ladder for all six blocks with `JAX_ENABLE_X64=1` and float64 fields. The module's
builder is handed a `Simulation` defaulting to `precision="float64"`. The judged comparison is
then the float64 one, and the float32 numbers are reported. `result.json` names the precision of
every number.

### 1.6 Field map (step e), reported
|E_z(f_t)|^2 in the substrate comes from the same model. The method: two
`add_dft_plane_probe(axis="z", component="ez")` planes at the E_z edges 1.5 and 2.5 cells above
the ground (0.375 h and 0.625 h), in one `forward` run, averaged. **Reported:** the Pearson
correlation between the z-summed gradient map and that |E_z|^2 over the design box footprint,
E_z index (i, j) against design cell (i, j). The same correlation at f_r is also reported. No
bar, and no interpretation in the record.

### 1.7 Timing (step f; issue item 3)
Three design boxes, with the same record (9912 steps), J and settings as §1.3:
- `patch`: the patch footprint only, 50 × 62 × 4 = 12 400 cells.
- `design`: the box of §1.3, 20 592 cells.
- `laminate`: the whole laminate, 71 × 83 × 4 = 23 572 cells.

For each box and each of the two modes, `jit(J)` and `jit(value_and_grad(J))`, one process
records:
- the first call, which includes compilation;
- 3 timed calls after it (the median is the wall time; compile time = first call − median);
- the device's `peak_bytes_in_use`.

**Derived, labelled derived:** the central-difference cost of every cell of a box,
**2 × N_cells × forward wall time**. It is stored under `derived` with that formula, and it is
never measured. The GPU model is pinned to the preset recorded in `result.json` and in the
resource-choice record.

## 2. Anti-reflection coating

**Structure.** `examples/inverse_design/multilayer_ar_coating.py`, imported by path. Its
builders, constants, cost pipeline and TMM are the example's. The values that live inside its
`main()` are restated in `scripts/showcase/ar_coating.py`: the L-BFGS-B start (2.5, 4, 7), Adam
lr 0.15, betas 0.9 / 0.999, eps 1e-8, 60 iterations, and the geometric-ladder warm start. The
example's own `main()` is also run in the same job, and its printed trace is compared with this
script's (reported).

The model is 1-D-equivalent: periodic y and z, CPML on x, dx = 0.5 mm, eps_sub = 12, and three
layers of 11 cells each (5.5 mm; the TMM uses the example's 5.494 mm). The record is 3147 steps.
The cost is the mean of |R(f)|^2 over the FFT bins in 8-12 GHz, and the precision is float32.
`jax.jit(jax.value_and_grad)` is used; the example itself does not jit.

**Recorded per iterate** (0 … 60, where 60 is the final eps_r with no step taken):
- the three eps_r;
- the FDTD cost;
- the FDTD R(f) on the band bins;
- the gradient, in latent space and in eps_r;
- the TMM R(f) at the same eps_r on the same bins;
- the wall time.

Also recorded: the TMM optimum (eps_r, cost), the geometric-ladder cost, and the TMM optimum's
cost at the realized 5.5 mm thickness (reported).

**Judged, the example's gates, computed as the example computes them.** The final cost is the
cost at iterate 59, `losses[-1]`, and the final eps_r is iterate 60.
- final cost ≤ 1.2 × TMM optimum cost;
- final cost ≤ 0.7 × geometric-ladder cost;
- ≥ 2 layers with 1.5 < eps_r < 11.5.

**Judged:** AD against central FD at the start point (the geometric ladder, 1.8612 / 3.4641 /
6.4474), for each of the 3 components of d cost / d eps_r. The ladder is h = 0.02, 0.01, 0.005
in eps_r, and the bar is |AD − FD| / |FD| ≤ 0.05 at h = 0.01.

**Reported:**
- FDTD R(f) against TMM R(f) at the final iterate: mean and max |ΔR| over the band bins.
- The cost at the final eps_r.
- The record-length witness: AD at the start point for 1.0× and 1.5× the record, both with an
  8192-point FFT so only the record length differs, as a per-component ratio.
- The largest relative cost difference between the example's own `main()` and this script, at
  the iterations the example prints.

## 3. Forward curves behind the benchmarks

For the RT5880 patch, the MSL notch filter and the Sheen low-pass filter,
`scripts/showcase/forward_curves.py` does the following:
- imports `tests/crossval/<case>/test_<case>.py` by path;
- calls its `run_rung` at `LADDER_M[-1]` (patch h/12, notch h/6, Sheen h/12), the rung each test
  compares with the openEMS `stage_b_fine`;
- saves S11 (and S21, and for the patch Zin);
- saves the reference records' curves, with each record's path and sha256.

It judges nothing. A5: the modules expose `run_rung`, callable without pytest, but the verdict
exists only inside the pytest test body. So each case's verdict at the same commit comes from its
unchanged pytest ladder in its own job: `scripts/vessl_showcase_ladder_<case>.yaml`, which runs
`pytest tests/crossval/<case> -m gpu`. The judging rules are the tests' own, unchanged.

## 4. What this batch does not do
- It does not change solver code, `tests/crossval/` or `validation/crossval/`.
- It does not converge the sensitivity map with the mesh. The map is at one rung, h/4.
- It runs no external solver.
- It adds no page under `docs/public/`; the site pages are a later PR.
- Interpretation of any number is the lane leader's, after the records are read. Records and PR
  carry facts and a "Conclusion: (leader)" placeholder.

## Amendment 1: the lane leader's answers of 2026-09-27

The lane leader answered the three open questions on 2026-09-27. The answers arrived after the first
patch and AR runs had finished: patch 369367265539 and AR 369367265547, both archived as the first
runs. This amendment was committed before the reruns it governs were submitted. Its one new
threshold, 0.05 for the record-length witness, is the leader's. It is the value the public
Gradient Behavior page uses with `gradient_record_length_witness`, and it was not taken from
any run.

**Q1, f_t (unchanged).** f_r stays the |S11| minimum from the module's `resonance()`, and f_t
stays 1.01 × f_r. Added as REPORTED, from the baseline forward on the same board:
- f0 (`refined_remax`) and the |S11| minimum;
- |S11(f_t)| in dB;
- d|S11|²/df at f_t, by central difference over f_t ± 1 MHz from one `forward(port_s11_freqs=…)`
  run, and also over the 901 run() bins.

No sentence interprets the sign.

**Q2, the record-length witness (now JUDGED, tolerance 0.05).**
- Patch: `gradient_record_length_witness(objective, eps_box, 9912, tol=0.05, factor=1.5)`
  computes the design-box gradient at the base record and at 1.5 × it. For each of the six block
  sums, the job computes |S_1.5 − S_1.0| / |S_1.5|. It is judged ≤ 0.05 on the blocks above the
  same cutoff as §1.4 (|FD(0.1)| ≥ 0.1 × the largest of the six, from the precision the FD check
  was judged in). The other blocks are reported. Also reported: the helper's own norm-level
  verdict, ||g_1.5 − g_1.0|| / ||g_1.5|| over all box cells, with its cosine, and the Pearson
  correlation of the two z-summed maps.
- AR coating: the same helper, with objective(ε, n) = the example's cost from an n-step record,
  both arms on one 8192-point FFT so that only the record length differs. It runs at N_STEPS and
  1.5 × N_STEPS at the start point. Each of the three components is judged at
  |g_1.5 − g_1.0| / |g_1.5| ≤ 0.05. The helper's norm verdict is reported.
- If a witness fails, the case stops there and is reported. The record is not lengthened.

**Q3, timing box 3.** Box 3 becomes the full substrate z-layer (k = 36 … 39) across the whole
physical interior, kept out of the CPML: cells 24 … 119 along x and 24 … 135 along y,
96 × 112 × 4 = 43 008 cells at h/4. It includes the vacuum cells beside the laminate; `result.json`
records how many of its cells are laminate and how many are not. The `patch` and `design` boxes
are unchanged. All three are re-timed in the rerun on the same pinned GPU model (RTX A6000).
