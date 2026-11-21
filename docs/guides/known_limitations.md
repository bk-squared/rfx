# rfx Known Limitations

Defects and limits that a user can hit today, on the current `main`. This page
is the companion to the [support matrix](support_matrix.md): that page says what
is supported and within what limits, this one says what is known to be wrong or
unevidenced inside those limits.

Each subject has its own heading and a `Tracker:` line: issue/PR numbers, or
`none — accepted limitation (<reason>)` for a deliberate untracked limit.

Each entry gives the symptom you would see, the mechanism in one line, what to do
about it now, and the tracking issue. **The issue is the evidence**: measured
numbers, the fixture they came from, and the commit they were measured on live
there, not here. An entry disappears from this page when its defect is fixed and
a committed test pins the fix, not when someone believes it is better. An entry
whose wrong result is now refused, with a committed test pinning the refusal,
moves to the **Refused inputs** table of the [support matrix](support_matrix.md)
and leaves this page.

Nothing here is a substitute for the repo's standing rule: a warning's absence is
not an accuracy guarantee, and a preflight pass is not a convergence study.

## General constraints

### Traced full-tensor CPU builds on older jaxlib issue pieces serially

Tracker: #2095

On CPU with jaxlib < 0.11.1, `forward(subpixel_smoothing="full_tensor")` with traced geometry or permittivity
could hang. When one linear-algebra call holds more than 2^20 matrices (grids above about 1 M nodes), rfx now
issues its pieces one after another. At 1.48 M nodes the measured runs hung 0 of 8 times, against 6 of 11
before, at about 1.5× the build time and about 1.24× host memory. Smaller grids keep the previous code and
cost. This is a stopgap with an
unproven bound: 56 independent `_linalg` sites remain concurrent in one build. If a build still hangs,
set `PJRT_NPROC=1` before JAX starts; that workaround's cost is unmeasured. jaxlib ≥ 0.11.1 keeps full speed.
rfx's dependency range now installs jaxlib 0.11.1 on Python 3.12+; Python 3.11 installs jax 0.10.2.

### One Python process that compiles many distinct grids can abort inside the JAX compiler on Linux hosts with the default vm.max_map_count

Tracker: none — accepted limitation (warns at default: tests/contracts/test_map_count_warning.py::test_public_run_warns_once)

The process can die with a segmentation fault or abort inside `backend_compile_and_load` (JAX's compiler),
with no Python error and memory to spare. No wrong number is returned; unsaved results are lost.
This was measured at about 151 distinct grid shapes compiled in one Python process at the Linux default
limit of 65,530, for example a mesh or domain-size sweep in one notebook or script.
Each new grid shape compiles new programs and JAX keeps them mapped. Reusing the same grid shape is
expected not to add to the mapping count; not measured.

Check the limit with `cat /proc/sys/vm/max_map_count` or `sysctl vm.max_map_count`; 65530 is the default
on many distributions. Raise it with `sudo sysctl -w vm.max_map_count=262144`; to persist it, put
`vm.max_map_count = 262144` in a file under `/etc/sysctl.d/`.
Alternatively, inside the sweep call `jax.clear_caches()` and `gc.collect()` every few tens of grid
shapes. The measured mapping count stayed below 23,000 when cleared every 50 shapes; later calls recompile.
rfx warns once per process (`MapCountWarning`) when the limit is below 262,144 on Linux.
This limitation was measured on Linux. macOS, Windows and GPU backends were not measured.

### Graded lumped S: mixed records, periodic run requests and the run() default

Tracker: #1373

Graded `run()` and `forward(port_s11_freqs=...)` use the uniform lumped wave
decomposition for all-lumped records, with at most one driven port, and for
equal bins and step counts return the same S and lumped spectra. `run()` with no
S argument forms lumped S on the 50 default bins, as on a uniform mesh, when the
ports are all lumped with exactly one driven port, no plain source or 0-ohm port
and no periodic axis; with a plain source or 0-ohm port it warns and returns
fields and S=None, as a uniform `run()` does. One row still differs from a
uniform `run()` with no S argument: with two driven lumped ports the graded run
returns fields and S=None with the isolated-drive warning (uniform drives each
port in its own scan). The uniform and graded `run()` drive scans drive each
port in turn, so a passive port still gets its S column.
`forward(port_s11_freqs=...)` and the single-wire path read one record and,
with every port passive, refuse an explicit S request or warn and return no S
for a default request; the per-port V, I and V_ref spectra stay in
`lumped_port_sparams` / `wire_port_sparams` through `run(compute_s_params=False)`.
Records containing
any wire port, including mixed wire/lumped records, retain the wire
decomposition and the wire default: the remaining exception to one wave
definition per port kind. The periodic `run()` S refusal remains; graded
`forward()` may return S for the same periodic declaration.

### S-parameters from a run with several excited ports

Tracker: none — accepted limitation (refused at default: tests/contracts/test_lumped_wire_s_isolated_drives.py::test_several_excited_ports_refused_before_step)

Before 2.0.0rc1, graded run()/forward() and uniform forward() could return
mixed S-parameters from one record with several excited lumped or wire ports.
Each reported entry b_j/a_k was the sum, over every driven port m, of S_jm
times its incident amplitude a_m divided by the incident amplitude a_k.
Measured on a 50 Ω two-port through line driven at both ports by the same pulse:
every reported entry had magnitude 1.00 ± 0.01 across the band, because the
reported S11 was S11 + S12. At the line's match frequency, where S11 is -48 dB,
the reported S11 was 0 dB: an error of 100 % of max |S|.

Changed in 2.0.0rc1: explicit S requests, including ringdown, on these paths raise
before stepping when two or more impedance ports have excite=True. Graded
forward() with neither port_s11_freqs nor ringdown warns once and returns
s_params=None while the run proceeds.
Set excite=False on every port but the driven one and run once per driven port,
or use run(compute_s_params=True) on one device, which drives one port per run
on the uniform and the graded mesh. Uniform run(), including its multi-device S
driver, is unchanged. One excited port with passive receiver ports is unchanged.

### A partly metal cell edge (metal density between 0 and 1) is an artificial loss; a source on such an edge is scaled by the same weight

At metal density strictly between 0 and 1, the field on an edge is scaled every step by a weight; this is physical only at 0 and 1.
The loss acts as a conductivity -eps0 ln(1 - M)/dt on the edge (M = 1 - (1 - w)^n over its n incident cells), so it grows as the mesh is refined: in a 12 GHz PEC cavity a 2 x 2 mm post at metal density 0.01 lowers the mode Q from above 1e10 to 73 with 1 mm cells and to 36 with 0.5 mm cells. Only densities 0 and 1 are lossless, and a result with intermediate densities does not converge under mesh refinement. A density strictly between 0 and 1 comes only from a relaxed conductor: topology optimisation with PEC as one of its two materials (the projected density becomes the metal density; the default projection `beta_projection=1.0` leaves it gray, so binarise with a large beta before reading a result), or a `pec_occupancy_override` / `design_occupancy_override` array the user passes. A PEC body declared with `add(..., material='pec')` or a PEC sheet is always 0 or 1.
In 2.0 a source on such an edge is scaled by the same weight on single-device forward runs, as on multi-device runs.
A resistive sheet or lumped RLC element on such an edge is NOT scaled: its update replaces the field after the weight.
Tracker: #1373
→ [#1373](https://github.com/bk-squared/rfx/issues/1373)

### Lossy Box film area on a traced mesh is checked only on a concrete grid

Tracker: none — accepted limitation (refused at default: tests/contracts/test_film_drawn_size.py::test_t1_box_area_strict_refusal_and_declared_warning)

Box DC films retain their fold on traced coordinates; the area comparison runs on the concrete realized grid in
preflight. Its drawn rectangle is clipped to the sampled tangential extent, including absorber padding (one period
on periodic axes). Non-Box films retain their existing admission rule. Surface-impedance sheets use the same
`sheet_effective_size` check and solved-edge model as PEC sheets.

### A curved lossy film's absorption still changes with the mesh at its rim

Tracker: #1958

A lossy film (`add_thin_conductor` below 1e6 S/m) is now placed where it is drawn (cell-centre footprint), and a film that
fills the whole cross-section converges at second order to the resistive-sheet closed form (absorbed fraction 4/9 at
377 Ω/sq, error 6e-4 / 1.5e-4 / 4.5e-5 at 1 / 0.5 / 0.25 mm). A finite curved film does not yet converge that way: a 377 Ω/sq
disc of radius 3 mm in a 40 × 15 × 30 mm metal cavity changed the cavity's decay rate by −1 % from 1 to 0.5 mm and by −16 %
from 0.5 to 0.25 mm, with its realized area right to 1 % on both meshes. The change comes from the staircased rim; its
mechanism is not measured. Check a curved film's loss on three meshes before using it as a converged number.

### A surface-impedance sheet one cell in front of a PEC face gives no loss to a field whose E is normal to it

Tracker: #2429

Use `add_wall_loss(faces=...)` or `add_wall_loss(shape=...)` for opaque PEC wall conductor loss.

A surface-impedance sheet (`add_thin_conductor(..., surface_impedance_f0=...)`) is a penetrable film: it carries current
only through the tangential E on its edges. Two cases are measured. With a PEC face one cell behind the sheet, a field
with E normal to the sheet and H tangential passes into the one-cell gap, and the sheet adds no conductor loss.
Synthetic case: a closed 30 × 12 × 24 mm vacuum cavity, TE101 at 8.0 GHz, one wall pair replaced by f0 sheets one cell
inside the PEC faces. With sheets on the y faces (E normal to them), the mode's field decay rate was 22,134 1/s against
25,108 1/s with PEC sheets in the same place; a Leontovich wall of the same resistance gives about 1.55×10⁷ 1/s. With
sheets on the x or z faces (E tangential), the cavity Q matched the closed form within 4×10⁻³ at 1 mm cells. With only
air and the absorber behind the sheets, the sheets carry the conductor loss: a probe-fed patch and its ground plane,
both f0 sheets (copper, and ten times copper's resistance), with the E between them mostly normal to them, lost
1.17–1.22 times the cavity-model estimate in the sheets on the bins where |S11| is below −10 dB, at three meshes; input
power equalled radiated, substrate and sheet power within 3×10⁻⁴ of the input (both measured; records kept with the
project). Other spacings between a sheet and a lossless conductor behind it were not measured. For a wall one cell in
front of a PEC face that must carry loss for every field orientation, use a PEC wall or a thick lossy Box.
Preflight warns when realized conductors bound both sides, or realized PEC owns the sheet's tangential edges away
from its footprint boundary rows; domain PEC faces contribute only to the two-sided conductor check.

### Conductors join by drawn extents pair by pair: three cases a gap can still close

Tracker: #1958

Metal sheets and lossy films (`add_thin_conductor`) on one plane join only where their drawn extents touch, so a gap drawn
between two of them stays open at any width (a sub-cell gap is solved as one empty cell row). The rule decides per pair of
declarations, by bounding intervals, and across shared faces. Three cases are outside it:
- two conductors that touch at one place and are gapped at another along the same axis (two halves of a period meeting
  through a periodic seam; curved or non-convex shapes whose bounding boxes touch) are joined at both places;
- one conductor's gap to its own periodic image (a seam slot) is not a pair, so the rule does not cover it;
- two films whose cells meet only at a diagonal corner node are not separated.
Draw such gaps as separate declarations that do not touch anywhere along the gap's axis, at least one cell wide.

### Conductor thickness is not solved: a metal sheet has zero thickness

Tracker: #1523

`add_thin_conductor` with a metal (`sigma_bulk` at or above 1e6 S/m) builds a PEC
sheet on one node plane; its `thickness` argument is not read
(`rfx/api/__init__.py`, `Simulation.add_thin_conductor` docstring). A declared copper
thickness is therefore not part of the model, and a microstrip line built from such a
sheet reads slightly electrically long against the same line with its real thickness.
Widening the sheet does not correct it.
Preflight now reports declared metal-sheet thickness above 2 % of the nearest overlapping Box dielectric's height, with the expected electrical-length shift.
A PEC `Box` with zero extent along one axis also has zero thickness, and preflight does not report it: the check reads only `add_thin_conductor` thickness.
Against a reference with the real copper thickness, a zero-thickness microstrip reads +0.45, +0.57 and +0.99 % electrically long at thickness 4.5, 8.5 and 17.5 % of the substrate height.

---

### The AD memory estimate reads low with checkpoint levels and with several outputs at once

Tracker: #1739

`estimate_ad_memory` and `plan_ad_memory` remain static estimates. On one graded scene
(737,009 nodes, float32, RTX A6000, JAX 0.10.2), single-output device peaks were
1.00 to 1.14 times the estimate; leave 15 % above it with checkpoint levels.
For simultaneous pullbacks, `cotangents=` now adds 7.5 carries per extra output:
one boundary carry plus 6.5 carries of backward working arrays.
This calibration uses one graded whole-grid scene, not a runtime guarantee.
Design boxes use the same whole-grid adjoint workspace estimate and are unmeasured;
other scenes, backends, and JAX versions are also unmeasured for this coefficient.
With a lossless design box and `checkpoint=False`, a gradient keeps half the per-step field history it kept in rfx 2.0.0, and the estimate counts the cells the box writes.
For that field term the estimate allows up to twice the stored amount, so it can read high.
Measure a representative run or pull back one output at a time when memory is tight.
The former 2.2-times advice was exceeded by the measured 16-output row (2.25 times).
Since rfx 2.0, on a single-device graded run on a GPU, `estimate_ad_memory(..., compiled=True)` compiles the gradient program without running it and returns its device peak in `compiled_total_bytes`: 0.992 to 1.000 of the measured peak on four graded rows on two GPU models, for one compile of 9 to 31 s.
It refuses wherever it was not measured: uniform, multi-device and subgridded runs, other backends, a preallocated GPU memory pool (`XLA_PYTHON_CLIENT_PREALLOCATE` must be "false"), design boxes, float64, warmup steps, plans that do not segment the run, and outputs other than probe and waveguide-port records.

### jax 0.11.2 and 0.12.0 run the time step slower than jax 0.10.2, on CPU at every measured size and on GPU on a large grid

Tracker: #2125

rfx's install range (`jax>=0.10.2,<0.11.2`) does not install these versions; this entry applies where they are installed by hand.

Measured at two grid sizes, with and without one lumped RLC element. This is not a trend over grid size.
On one RTX A6000 at 3.05 M cells (CPML included), jax 0.11.2 takes 1.18 times as long per time step as jax 0.10.2.
On the GPU at 0.53 M cells, the two versions take the same time (1.00).
On both devices the slowdown appears with and without the lumped element.
Peak device memory is the same on both versions.
`estimate_ad_memory` still reads 1.007 to 1.009 of the measured gradient peak with `checkpoint=True`.
On CPU (one node, 8 cores, the same two grids), jax 0.11.2 takes 2.7 to 3.0 times as long per step at 0.53 M cells.
At 3.05 M cells it takes 1.6 to 2.4 times as long.
The compiled CPU step holds 18 more domain-sized copies on jax 0.11.2, with and without the lumped element.
Peak host memory is unchanged on CPU (0.91 to 1.01 times).
jax 0.11.0 and 0.11.1 run the CPU step at jax 0.10.2's speed (1.00 to 1.01 times, 8 domain-sized copies); jax 0.12.0 is slower (5.6 times at 0.53 M cells, 2.4 times at 3.05 M cells).
On CPU, every domain-sized copy that jaxlib 0.11.2 adds comes from the absorber (CPML) write-back. That write-back is a chain of partial updates of each field, and from 0.11.2 the compiler recomputes such chains (openxla/xla#50977). This holds in 40 of 43 measured scenes.
On the GPU the write-back takes another form, and the extra copies do not appear; the cause of the GPU's 1.18 times at 3.05 M cells is not measured.
An rfx-side change, a compiler barrier after each absorber update on CPU, removed the extra copies. With it, the step on 0.11.2 took 0.98 to 1.00 times jax 0.10.2's time, and the fields stayed bit-identical. It is not in 2.2; it is planned together with lifting the install cap.
These step times come from a second CPU node; on it, the unchanged step on 0.11.2 took 1.5 to 1.7 times jax 0.10.2's time, so the size of the slowdown depends on the machine.
No XLA flag found restores the step: `--xla_cpu_experimental_enable_tiling_propagation=false` leaves it 2.4 to 3.0 times slower, and jaxlib 0.11.2 rejects `--xla_cpu_use_fusion_emitters`.
Workaround: keep rfx's install range; on Python 3.12 or newer it installs jax 0.11.1.
jax 0.11 needs Python 3.12; on Python 3.11, rfx installs jax 0.10.

### Subpixel smoothing makes a curved high-contrast dielectric's resonance converge slowly, and can be worse than no smoothing.

`subpixel_smoothing=True` and `"kottke_pec"` keep only the diagonal of the smoothed inverse-permittivity tensor.
True gives first-order convergence at curved dielectric interfaces, worst at high contrast (eps ~30 and above).
`run()` warns (`CurvedSmoothingWarning`) only for a smoothed `Sphere` or `Cylinder` at 10 times the contrast of the background or of another declared dielectric, or more.
Below that no message is given, and convergence is still first order: a lossless eps-4 sphere's extinction converges first order against Mie over three meshes.
There smoothing is still closer to Mie than the staircase at each mesh.
A measured eps-38 rod resonance was off by 1–2% on practical meshes, and up to 8x further off than without smoothing.
Grid-aligned faces are not affected. Judge such resonances by at least three refinements and compare against
`subpixel_smoothing=False`; disagreement means neither series has converged.

`subpixel_smoothing="full_tensor"` uses full anisotropic averaging (Farjadpour 2006; Oskooi 2009;
Werner–Bauer–Cary 2013 node form) on single-device Yee meshes, uniform or graded. Extinction cross section on a synthetic eps-4
sphere measured second-order convergence over three meshes. This is not a convergence-order claim for high-contrast rods.
Multiple devices, subgrids, UPML and the waveguide S driver refuse this option.
Kerr and declared-shape gradients remain refused for every smoothing option.
The full_tensor diagonal can jump where an interface enters an absorber pad (triplet interior, Kottke midpoint pad); oblique TFSF refuses a noninvariant padded periodic axis; use a closed box (#1632).
Tracker: #1632

### A material gradient through `forward()` needs `skip_preflight=True` until preflight reads a traced permittivity.

Tracker: #2436

`jax.grad`, `jax.jit` and `jax.vmap` over a material's `eps_r` now run through `subpixel_smoothing=True` (uniform and graded mesh) and `"kottke_pec"` (uniform mesh); `"full_tensor"` refuses a traced `eps_r` (#2441; see the full-tensor entry on non-Box shapes).
With the default `skip_preflight=False`, `forward()` with a traced `eps_r` still stops in preflight with a `ConcretizationTypeError` at `rfx/preflight/mesh.py:295`, smoothing on or off: the mesh check reads the permittivity as a Python number.
Pass `forward(..., skip_preflight=True)` and run `sim.preflight()` once on a concrete value.
Under a trace the curved high-contrast check cannot judge a traced material; it warns (`CurvedSmoothingWarning`) that the check was not judged and names the material.
Two materials given different tracers are smoothed as two media even when their values are equal; a concrete run merges equal values into one medium.

### A material with permittivity below 0.98 grows without bound at the default time step, and nothing refuses it.

Tracker: #2772

A Yee cell of relative permittivity ε (μr 1) is stable only for a time step up to √ε times the vacuum Courant limit,
because its fastest waves travel at c/√ε, faster than light for ε below 1. In a Debye material the pole terms vanish at
the grid's Nyquist frequency, so the cell sees ε∞ there. No check refuses a step above a cell's own limit: the default
step is 0.99 of the vacuum limit, and the pole stability check skips a cell whose step is above that cell's limit
without its poles. A material with `eps_r` below 0.98 (0.99²), or a Debye material with ε∞ below 0.98 (such as a
`fit_debye` result before 2.3), therefore grows until the fields are non-finite at the default step when it fills cells that set
the step (every cell on a uniform mesh); this was measured at 0.5 on the uniform and the graded mesh. Between 0.98 and 1
the default step stays inside the cell's limit. Nothing refuses the run; it ends with an `AccuracyWarning` about
non-finite values.

Two material fits reach this entry, measured with a 12.3 × 11.5 × 10.8 mm box of the fitted material on 1 mm cells in
an open domain (uniform mesh, and a graded mesh with five 0.8 mm cells in x), at the default step:
- On a table with ε′ 4.4 and tan δ 0.02 constant over 1–40 GHz, `fit_debye` with a free ε∞ returned ε∞ 0.5 before
  2.3 (it now keeps ε∞ at 1 or above). `run()` admits the 2-pole fit on both meshes; the fields grow by a factor of 1.04 per step and become
  non-finite at step 2025 (uniform) and 2008 (graded).
- On a table with heavy loss at low ε′ (ε′ 1.1, tan δ 0.1, constant over 1–40 GHz), `fit_lorentz` returned ε∞ 0.1, its
  lower bound, with 2 and with 3 resonances before 2.3 (it now keeps ε∞ at 1 or above). `run()` admits both fits on both meshes; the fields grow by a factor of
  100 to 140 per step and become non-finite at step 22.

In both cases preflight reports only advisories, and the run ends with the `AccuracyWarning` about non-finite values.
Before building a material from a fit, check `fit.eps_inf`: below 0.98, the material is outside the default step's
stable range. The 2-pole `fit_debye` result above ran 17143 steps without growth at 0.61 of the vacuum Courant limit
of the smallest cell, on a graded board with a pinned `dt=`. No step was measured for the `fit_lorentz` result.

Keep `eps_r` and ε∞ at 1 or above. On a graded mesh, a pinned `dt=` below √ε times the vacuum Courant
limit, with ε the smallest permittivity, is inside the stable range: at 0.6 of the limit a material with ε 0.5 ran
without growth.

### `subpixel_smoothing="full_tensor"` keeps the one-plane rule next to a sphere, a cylinder or another non-Box shape, and refuses traced coordinates beside a Box.

Tracker: #2441

The full-tensor build gave each node one plane, the nearest face. At a box edge on a grid node, or on the bisector of a box edge, rounding chose the face and the model jumped: on main 5150ed340 a face moved by 1e-12 m changed the 20-step probe loss of an eps_r 10 box by 15 %.
Fixed for axis-aligned Boxes, also overlapping ones: a node within 1.25 times the reach of its edge supports (1.53 cells on a uniform mesh, from the realized cell sizes on a graded one) of a Box takes, on each support, the exact content of the Boxes that meet it, painted in declared order as the assembled model paints them. On the on-node and the bisector box scenes a face moved by 1e-12 m changes no diagonal value by more than one float32 step. Every node farther from all Boxes keeps the one-plane values bit for bit.
Next to a non-Box shape the exact and the one-plane triplets are blended over one cell, with the weight rising from 0 at the reach from the shape's bounding box to 1 one cell farther; within the reach the one-plane rule stays, with its jumps where a box face plane ties. Measured on an eps_r 10 box beside an eps_r 10 sphere of radius 1.5 mm whose bounding box sits at the reach, half a cell and one cell beyond it, the sphere moved by 2e-9 m: no diagonal value changes by more than the one-plane build changes it plus 1.7e-6 of the value (uniform and 1.0 / 0.8 mm graded meshes); 26 to 50 nodes are blended; the 20-step loss does not change in float32.
Within that reach the jump is large. A 12 mm PEC box with εr 10 shapes was measured on 1 mm cells (record kept with the project), uniform and graded in z (1.0 / 0.8 mm). Moving a shape coordinate by 1e-12 m changed the 20-step probe loss by 0.39 % to 39 % at a Cylinder rim on grid nodes, whether the cap was translated or the radius changed, along x, y or z. At a Box edge on a node with a sphere 1 cell away it changed by 10.7 % (uniform) and 24 % (graded); at a Box edge with nodes on its bisector, with those nodes 1.4 mm from the sphere's bounding box, by 7.6 % and 12 %. There the jump is the one-plane face flip at the bisector nodes; the nodes next to it, whose normal comes from the sphere and reaches the Box within one cell, count the Box's content along that normal, and that corrected model responds more to the same flip (measured: main's model with this fix's values at only those nodes gives this fix's losses exactly, on both meshes). The diagonal jumped by up to 6.5×10⁶ float32 steps in 20 to 56 cells. A gradient or a sweep through such a coordinate sees a step. Until one content rule covers every shape, use `subpixel_smoothing=True` near cylinders and spheres (measured for cylinder coordinates and for a Box face beside a sphere; a sphere's own coordinates were not moved). On the same scenes and meshes, `True` was measured continuous: at ±1e-12 m no cell moved by a float32 step and the loss did not change, and from 1e-9 to 1e-6 m its changes grow with the move. Its slope can differ on the two sides of a face that lies on a node (a kink, not a step). Its accuracy against `"full_tensor"` near curved shapes was not measured. A non-Box body thinner than a cell keeps the one-plane rule too: a `Cylinder` disc (εr 4) 0.3 of a cell thick reads a diagonal Ez of 1.951 where the series average is 1.290, on uniform and graded meshes, while `True` reads 1.290 there. `OrientedBox` is not reachable under `"full_tensor"`: it raises `NotImplementedError` by name.
Where a later Box of one permittivity sits inside an earlier Box of another, the triplets carry the later Box's value. A later shape whose permittivity repeats an earlier one is painted in declared order with `True`, `"kottke_pec"` and `"full_tensor"` alike: in an εr 4 Box nested in an εr 10.2 Box after another εr 4 shape, every mode reads 4.0 inside the nested Box.
A traced permittivity with `"full_tensor"` is refused by name in any scene. With a Box in the scene a traced shape coordinate or mesh profile is also refused by name: the scope of the exact content is decided on concrete coordinates and realized cell sizes, and under a trace the build runs in float32 (plain and `jax.jit` losses 9.7×10⁻⁴ apart with a traced sphere radius beside a concrete εr 2 box, 8.8×10⁻³ with a traced permittivity on an εr 4 box). This is a capability change: main admitted a traced Box coordinate, and its build equalled the host build in shared-face scenes and moved a plain-versus-jit loss by up to 9.65×10⁻⁴ elsewhere (a traced sphere radius beside a Box); restoring it with the exact content formed on traced coordinates is #2473. More than eight boxes meeting one edge support are refused by name. Use `subpixel_smoothing=True` or `"kottke_pec"` for a gradient over `eps_r`.
→ [#2473](https://github.com/bk-squared/rfx/issues/2473)

### Subpixel smoothing retains one source face and point rules for bodies without exact line intervals.

Tracker: #2624

The outside portion of each support now reads the declared content along its normal, including both faces of a thin neighbouring Box, Sphere or Cylinder. Fixed for planar bodies under `subpixel_smoothing=True` (uniform and graded mesh) and `"kottke_pec"`: where a `Box`, or a `Cylinder` cut cap to cap, has both faces inside one cell's averaging length, its fill is its own thickness and the medium outside it is read on both sides (an εr 4 slab 0.3 of a cell thick reads Ez 1.290, the series average, instead of 1.429).
A thin `Sphere`, a `Cylinder` crossed through its radius, and a run of touching bodies that includes either still use one face for the fill.
A dielectric `Sphere` or a `Cylinder` crossed through its radius that spans about two cells or fewer is over-polarized, with or without smoothing. Measured by the TE101 resonance shift (about 10.6 GHz) such a body causes in a 20 × 10 × 20 mm PEC cavity: the shift is proportional to the body's polarizability, and so to the field it scatters. The reference is the shift extrapolated from three finer meshes with smoothing (1, 0.5 and 0.25 mm cells: the sphere spans 2, 4 and 8 cells there, the rod 1, 2 and 4); it lies 3–4 % above the first-order perturbation closed form. Ratios to it, εr 4:

| body | cells across | `subpixel_smoothing=True` | no smoothing |
|---|---|---:|---:|
| sphere, radius 1 mm, 2 mm cells | 1 | 2.41 | 2.56 |
| rod across the field, radius 0.5 mm, 2 mm cells | 0.5 | 3.34 | 6.35 |
| rod across the field, radius 0.5 mm, 1 mm cells | 1 | 1.55 | 1.60 |
| sphere, radius 1 mm, 1.5 mm cells (cavity realized as 19.5 × 10.5 × 19.5 mm; its closed form uses those lengths) | 1.33 | 1.58 | 1.15 |
| sphere, radius 1 mm, 1 mm cells | 2 | 1.30 | 1.25 |
| rod across the field, radius 0.5 mm, 0.5 mm cells | 2 | 1.10 | 1.13 |

Scattered power goes as the square of the ratio, 20·log10 of it in dB: with smoothing, 7.6 dB high for the sphere one cell across, 10.5 dB for the rod half a cell across, 3.8 dB for the rod one cell across, and 0.9–2.3 dB at two cells; without smoothing 8.2, 16.1, 4.1 and 1.1–1.9 dB. Each row was taken at one position; the ratio is expected to depend on where the body sits on the mesh, which was not swept. With smoothing, put more cells across such a body until the result stops moving (without smoothing the measured curves do not approach the reference monotonically; measured on uniform meshes; a graded mesh that refines around the body adds cells there but was not measured).
A `Sphere`, or a `Cylinder` crossed through its radius, that is not PEC is refused wherever it is placed when its radius is below half the diagonal of the cell containing its centre: 0.866 of the cell on a uniform mesh, and for a `Cylinder` half the diagonal of that cell's cross-section, 0.707 of the cell. Such a body may contain no primal-cell centre, so before this rule it ran at some positions and was refused as "rasterizes to ZERO cells" at others. A sweep of, or a gradient through, its position could cross from a result to a refusal. The rule is not applied on a 2D grid, for a traced centre or traced cell widths, or for a centre outside the grid; there the empty-cell refusal alone decides. `preflight()` warns ("Small dielectric body:") for an admitted dielectric `Sphere`, or a `Cylinder` across its radius, under three cells across. A PEC `Sphere` or `Cylinder` keeps the earlier rule: below a cell, whether it runs depends on its position. Bodies just above the size threshold are still over-polarized, as in the table above.
Under `"full_tensor"` a thin non-Box body keeps the one-face rule: a `Cylinder` disc 0.3 of a cell thick (εr 4) reads Ez 1.951 on the diagonal where the series average is 1.290 (51 % high), on uniform and graded meshes.
Mask-only bodies and continued conductor sheet shapes retain the point rule for the outside medium.
On a curved face, full-tensor dual-face averages still use a first-order planar model.
A later group's pass with a crossed normal steps when its face enters the cell; measured and listed under subpixel smoothing in the support matrix.
Full-tensor spectral bounds include zero-weight portions, whose medium just beyond the face can step; a bound that clips then makes the tensor step too.
A traced shape on a periodic axis is admitted while its bounds lie inside [−L, 2L], but smoothing evaluates only its images −1, 0 and +1. If the shape is declared a full period outside the domain, with its upper bound less than half a cell below 2L, the smoothing samples at the domain's low face miss its copy across the seam and read the background. On 1 mm cells, 0.23 mm below 2L, Ey reads 1.000 where the body gives 1.810. The low end of the interval is affected only on a graded mesh with an oblique face normal: dx 1 mm, dy = dz 3 mm, 0.23 mm above −L, 0.318 off. A shape declared inside [0, L], or across one seam, is not affected. Declare traced periodic shapes there.

### A metal face resting on a dielectric between grid nodes makes `subpixel_smoothing=True` and `"kottke_pec"` worse than no smoothing.

Tracker: none — accepted limitation (warns at default: tests/contracts/test_kottke_pec_warning.py::test_kottke_pec_volume_warning)

Where a PEC body rests on a dielectric and the shared face lies between grid nodes, the cell under the face is not averaged by its content.
With less than half of that cell metal, the smoothed inverse permittivity is larger than the content gives. Under a PEC strip on εr 4.4, with a quarter of the cell metal, `"kottke_pec"` reads 0.315 and `True` 0.420, where the content gives 0.170.
With more than half of that cell metal, the field there is set to zero, where the content gives 0.085.
With exactly half, `"kottke_pec"` read 0.307 on 0.5 mm cells and zero on 0.125 mm cells, where the content gives 0.114.
On a 30 mm PEC strip resonator in a closed box (εr 4.4), with only the substrate face under the strip off the grid, the first resonance against a reference mesh on which every face lies on a node is off by:

| cells | `True` | `"kottke_pec"` | `False` (staircase) |
|---|---:|---:|---:|
| 0.5 mm | −4.28 % | −4.28 % | −1.97 % |
| 0.25 mm | +5.99 % | +3.15 % | −0.90 % |
| 0.125 mm | −1.40 % | −1.40 % | −0.23 % |

The error follows how much of the cell under the face is dielectric, not the cell size, so refining the mesh does not make it fall steadily.
The second resonance moves the same way.
With `"kottke_pec"`, an off-grid side edge of the strip also adds error that does not fall with refinement: −1.40 % at 0.125 mm with the edge half a cell off the grid, against −0.15 % for the staircase.
Faces on grid nodes are not affected: with every face on a node, the three settings read the same resonance.
The cause lies in the metal model itself, not in one coefficient.
Under `"kottke_pec"`, any part of a cell that holds metal freezes the field component parallel to the metal face; with `True` and `False` the edge ownership rule freezes the same row.
So, on the 0.5 mm and 0.125 mm meshes measured, the tangential field row just below the off-grid metal face is frozen in all three settings (on the 0.25 mm mesh that row lies clear of the metal).
That cuts the half cell of dielectric between that row and the drawn face off from the rest of the substrate, so the metal acts as if it began one row lower.
A correction to the field normal to the face was measured and cannot reach that row.
`"kottke_pec"` is therefore not recommended where a metal body rests on a dielectric face that lies between grid nodes.
Put the dielectric face under the metal and the metal's edges on grid nodes: choose the cell size so that they fall on whole cells, or build the profiles with `rfx.mesh_edges.edge_aware_profiles(..., solids=[...], axes="xyz")`.
Otherwise use `subpixel_smoothing=False`, which converged on every mesh measured.

### Subpixel smoothing with `"kottke_pec"` leaves a late field around a closed curved metal body, so its scattering depends on the record length.

Tracker: #1651, #2291

With `subpixel_smoothing="kottke_pec"`, a plane wave on a PEC sphere (ka 6, cells of λ/20, 8–11 GHz) leaves a field of 3–5 % of the peak outside the body after the pulse has passed.
This field stays to the end of the record instead of decaying.
The backscatter then depends on the record length: from 6 to 12 ns it changed by 2.3–2.7 dB at most frequencies and by up to 7.7 dB at a backscatter null.
The extinction from the optical theorem differs from Mie by 20–51 % across the band; the staircase (`subpixel_smoothing=False`) on the same mesh differs by 13–14 %.
A late field of 1e-2 to 5e-2 of the peak was also seen at ka 10 and on three meshes.
`run()` warns "no ring-down settling witness" and names the outputs read from that record,
but does not refuse the run.
Until a fix lands, use the staircase on a closed curved PEC body.
Conformal PEC was not measured on this sphere, and it cannot be combined with `"kottke_pec"` (next entry).

### Conformal PEC on a curved metal body converges only first order.

Tracker: #1651

`conformal_pec=True` (and `Boundary(conformal=True)`) gives each cut E edge eps/l̄ (l̄ is its PEC-free length) and each cut H face mu·s̄ (s̄ is its PEC-free area).
A stability clamp (Zagorodnov's SC) keeps the ordinary time step.
A flat PEC wall sits at its drawn plane within 0.01 cell.
On a PEC sphere (radius 9 mm, plane wave 8–12 GHz, cells 0.75 / 0.5 / 0.375 mm), the extinction fitted against Mie reads a sphere larger by 0.19 / 0.18 / 0.20 cell; the staircase reads 0.69 / 0.74 / 0.70 cell.
The backscatter errors are 0.56 / 0.34 / 0.28 dB, against 2.35 / 1.75 / 1.19 dB for the staircase.
The offset stays constant in cells, so the error falls only first order in the cell size.
A user gets conformal PEC only on request: `run(conformal_pec=True)`, or any axis declared `Boundary(conformal=True)`, which turns it on for every PEC body in that run. The default is the staircase. A run with conformal PEC warns.
Under conformal PEC these are refused: `conformal_min_weight` other than 0.1; PEC shapes with no signed distance (polyhedra, extrusions, volume wires); a body that holds no grid node strictly inside, except a Box with every face on a node plane or a domain face; a zero-thickness PEC sheet off the node lattice; and `subpixel_smoothing="kottke_pec"`, refused before any time step on every path.
Graded meshes, multiple devices and subgrids refuse conformal PEC. `forward()` has no `conformal_pec=` argument: on a uniform mesh on one device it takes conformal PEC from a `Boundary(conformal=True)` declaration, and it refuses one together with `pec_mask_override`, `pec_occupancy_override` or `design_occupancy_override`. `optimize()`, `topology_optimize()`, `compute_mixed_s_matrix()`, `differentiable_material_fit()`, the fast path of `vmap_material_sweep()` and a traced PEC shape refuse conformal PEC.
Judge a curved body by a mesh series of at least three refinements.
A closed-form check of the conformal update, the TM010 resonance of a hollow PEC cylinder (`tests/oracle/test_conformal_convergence.py`), stays marked as an expected failure until a fix lands.

The v2.1 opt-in `run(conformal_pec="cut_cell_floor")` with an explicit half- or quarter-Courant step reduces the measured sphere offset; the q = 0.5 result remains first order and misses the order >= 1.5 target (see the [support matrix](support_matrix.md#conformal-pec-cut-cell-floor-opt-in-v21)).

### Conformal PEC cut-cell areas and lengths deviate from the drawn surface where the mesh is nearly tangent to a curved body.

Tracker: #2274

With `conformal_pec=True`, the H update uses the open (PEC-free) area of each cut face. The E update uses the open
length of each cut edge, after a stability bound that caps it by the areas of the faces around it. Measured against
the drawn surface on 1 mm cells, before that bound, curved bodies are off by up to 0.44 of a face (cylinder side) and
0.63 of an edge (sphere), where a mesh face or edge is nearly tangent to the surface.

An axis-aligned Box takes its exact overlap with each cut cell, its edges and corners included. A cut cell that
another PEC body also reaches (a sphere, a cylinder, or such a body continued into the absorber; within one cell
diagonal of its surface) keeps that body's rule, so a Box edge there is not exact. Open areas and lengths in cells crossed only by a flat face are exact, and a Box face lying on a
mesh plane is realized at its drawn plane. In a closed PEC cavity, a PEC block whose faces lie off the mesh planes
resonates as the exact-area computation does on 1, 0.5 and 0.25 mm cells, and an empty cavity with two or three walls
off the mesh planes puts each wall within 0.01 cell of the closed form. Effect of the curved-body deviation on
S-parameters, resonance frequencies and RCS: not measured.

A finer mesh shrinks the region affected near an edge. On a sphere at radius 4.3, 8.6 and 17.2 cells it also shrinks
the share of nearly tangent cells: the faces off by more than 0.1 of a face were 17, 4 and 6, while the cut faces grew
16x. The largest deviation does not shrink. Judge a curved body by a mesh series of at least three refinements.
Measured 2026-10-31 with the conformal update now on main (open edge lengths and cut-face areas); the Box overlap
measured 2026-11-19; records kept with the project.

### Inside a user `jax.jit`, `run()` refuses a single port's S-parameters on a uniform mesh, `until_decay` and `report_every`, and a multi-device closed-box plane wave is refused

Tracker: none — accepted limitation (refused at default: tests/contracts/test_user_jit_drive_and_probe.py::test_refused_under_user_jit)

On one device, a fixed-step `run()` inside a user `jax.jit` stays within the cross-trace bar of plain `run()` on Linux CPU (9 float32 ULP of the probe peak per step, 1e-4 of the peak summed over the record) for a point current source and for a lumped or wire port, on uniform and graded meshes. The same holds for `forward()` with a traced `eps_override` and a point source or a lumped port, and for `run(devices=...)` and `forward(distributed=True)` with a point source or a lumped or wire port, on uniform and graded meshes (`tests/contracts/test_user_jit_drive_and_probe.py`). Four requests are refused under a user `jax.jit`: on a uniform mesh, `run()`'s S-parameters of a single lumped or wire port (use `forward(port_s11_freqs=...)` or a plain `run()`); a closed-box plane wave on several devices; `until_decay`; and `report_every` (`test_host_control_keeps_its_user_jit_behavior`). Call these without an enclosing `jax.jit`.

### One process with several GPUs: peak memory per GPU stops falling after two GPUs.

Tracker: #2150

In one process, `sim.run(devices=...)` needs about as much memory per GPU on three GPUs as on two. On three RTX A6000
GPUs a 1024 x 21 x 192-cell box peaks at 552.5 MB on one GPU and at 270.5 MB per GPU on both two and three, although
the compiled time step falls from 233.2 MB to 156.1 MB. The peak is reached on every GPU after the time loop; which
buffer holds it is not yet found. A third GPU therefore does not let a model fit that two GPUs cannot hold.
What to do: size a run on three or more GPUs in one process from its two-GPU peak.
→ [#2150](https://github.com/bk-squared/rfx/issues/2150)

### A closed PEC box with a TF/SF plane wave runs 11.6 % slower per step on two GPUs than before the shared multi-device time step.

Tracker: #2472

On two RTX A6000 GPUs a 236³-cell (13 M) closed PEC box driven by a TF/SF plane wave takes 2.010 ms per step, against
1.801 ms before the shared multi-device time step (medians of three processes). The same box with CPML is 3.4 %
slower (3.293 to 3.405 ms per step). Field values are unchanged. Two whole-field copies before the PEC wall write take
most of the difference; removing them recovered only 0.043 ms per step, for a reason not yet measured.

### A small Debye, Lorentz or Drude region makes every time step update the dispersion state on the whole grid.

Tracker: #1825

The polarization state of a dispersive material is held and updated on every cell of the grid, not only where the
material is. On the graded path a 16³-cell Debye block in a 160³-cell box (padding included) adds 23.1 ms per step,
69 % of the 33.7 ms bare step; Lorentz and Drude add 5.4 and 6.6 ms. The added time grows with the box, not with the
block: a Debye block adds 2.1 / 6.4 / 23.1 ms at 72³ / 104³ / 160³ cells (uniform path: 1.7 / 4.1 / 21.0 ms). Field
values are unchanged. A gradient run also saves this state at each checkpoint: from the array shapes (not measured as
memory), 12 bytes per grid cell per checkpoint for a one-pole Debye material and 24 for Lorentz or Drude.
What to do: budget step time and gradient memory for the whole grid as soon as a model has any dispersive region.

### Each lumped RLC element adds its own block to the compiled time step, so compile time grows with the number of elements.

Tracker: #1946

The time per step and the field values do not depend on the number of elements; the compile does. On a 40 x 40 x 49-cell
box run for 200 steps on CPU, a repeat `run()` on the same `Simulation` took 0.72 / 2.91 / 10.78 s with 0 / 100 / 400
series R+C loads, and the first `run()` 4.4 / 6.9 / 17.1 s; the growth is linear in the count (log-log slope 0.93 over
50 to 400 loads). `run()` compiles again on every call.
What to do: with many elements, call the solve inside one `jax.jit` (for example around `value_and_grad` of the
objective) so that the compile is paid once.

### Under a caller's `jax.jit`, a multi-device run holds more whole-domain setup arrays on every device for dispersive materials and graded meshes

Tracker: #2486

Wrapping a multi-device run in `jax.jit` (`run(devices=...)` or `forward(distributed=True)` inside a caller's jit) puts some of its setup arrays into every device's compiled program as constants of the whole domain, not of the device's slab (#2486). Field values are unchanged; device memory and program size grow. A plain multi-device call, with no enclosing `jax.jit`, holds none.

Each such array costs every device the same number of bytes:

    bytes per array = n_devices × (⌈nx / n_devices⌉ + 2) × ny × nz × 4

where nx is the cell count along x, the axis the run splits; the 2 counts one ghost row on each side of a device's slab, and each cell holds 4 bytes (float32). This cost does not fall as devices are added. The number of such arrays a model holds, measured on a 25 × 13 × 13-cell box:
- a material with one Debye pole: 13 arrays; with one Lorentz or Drude pole: 11 (Drude takes the Lorentz route; not measured). Materials with more poles: not measured.
- a graded mesh: each material array, that is the permittivity (also with a lumped port), or the two arrays of a lossy thin film.

Example, measured: a Debye box of 25 × 13 × 13 cells holds 13 × 20,280 = 263,640 bytes on each of two devices, and 13 × 22,308 = 290,004 bytes on each of three. By the formula, a model of 10⁸ cells with one Debye pole holds about 13 × 0.4 GB ≈ 5.2 GB on every device, whatever the device count.
What to do: if a model fits only when split over devices, call `run(devices=...)` or `forward(distributed=True)` without an enclosing `jax.jit`.

### A multi-device run that records snapshots holds more than twice the whole snapshot record on each device while it joins the record

Tracker: #2608

`run(devices=..., snapshot=SnapshotSpec(...))` joins the devices' snapshot histories into one record at the end of the
run. While it does so, each device holds more than twice the whole snapshot record, and after the run every device
keeps a full copy of the record. The same run on one device peaks at 1.0 to 1.05 times the record. So the largest
snapshot record that fits is smaller on two devices than on one.
Measured on two CPU host devices, all six components of a uniform 48³-cell box (CPML cells included) recorded for 100,
200 and 400 steps: 2.75 to 2.76 times the record on each device at all three lengths, growing by 6.96 MiB per recorded
step. Host memory rose by a further 1.0 times the record during the join, so these figures are lower bounds (record
kept with the project). GPUs, graded meshes, more than two devices, fewer components and slices were not measured.
From the code: the join needs at least twice the record on each device for any device count, so more devices do not
raise the limit; the multiple is larger when fewer components are recorded; and a slice normal to x (`slice_axis=0`) is
held whole on every device during the run, so its join grows with the device count.
What to do: record fewer frames (`SnapshotSpec(interval=...)`); the measured peak grew in proportion to the frame count.
A 2-D slice normal to y or z (`slice_axis=1` or `2` with `slice_index`) lowers the peak with its size; fewer components
lower it by less than the record shrinks. Or run a snapshot-heavy case on one device.

### `preflight()` and `realized_geometry()` keep the model's whole-domain inspection arrays until a solve or `clear_inspection_cache()`

Tracker: #1432

`preflight()` and `realized_geometry()` keep the model's assembly context, its realized-geometry record and the coax
stamp on the `Simulation`, so repeated inspection calls share them. A solve (`run()`, `forward()`, the S-matrix
calculators, `topology_optimize()`) releases them when it is dispatched; a refused solve keeps them, and
`Simulation.clear_inspection_cache()` frees them for a model you only inspect
(`tests/contracts/test_inspection_release.py`). A `realized_geometry()` record you keep still holds the arrays it
refers to. Measured before the thin-sheet change below, the held arrays took about 21 bytes per cell (21.1 B on
81 x 41 x 17 cells, 21.0 B on 161 x 81 x 33), and 1.43 GiB of device memory stayed reachable from a kept record's
`conductors` on a synthetic model of 161 thin sheets on 3.05 M cells; neither has been measured since. On main after 2.0.0, thin PEC sheets keep only
their plane masks, so the GPU memory these arrays hold no longer grows with the number of sheets (host memory not
measured): a board of 161 sheets on 3.05 M cells, with `preflight()`, `realized_geometry()` and a second model kept in
the same process, completes with the GPU pool capped at about 5.6 GiB at a 0.46 GiB peak, where 2.0.0 ran out of
memory. What to do: call `clear_inspection_cache()` after inspecting a model you will not solve, and drop any
`realized_geometry()` record before a large solve.

## Ports and extraction

### The staircased coax line can be 40–44 ohm for a drawn 48.6-ohm line.

Tracker: #1958

On these meshes the built coax line is 40–44 ohm for a drawn 48.6 ohm, so the 50-ohm |S11| of a matched coax through-line reads up to about 0.14.
The cause is the staircased line, not the port; `line_impedances` reports the built value.
Setting `impedance=` to that value requests the line's own reference; it does not correct the built geometry.
A finer mesh changes Z_built non-monotonically: 40.55 / 44.08 / 41.29 ohm at
0.4 / 0.3 / 0.2 mm on the SMA line (measured; record kept with the project).

### A microstrip strip drawn past an MSL port to the domain edge is an open stub that can short the port.

Tracker: #1512

rfx ends a two-conductor line at its port: the piece of strip between the port plane and an
absorbing face ends at the absorber's entrance, so it is an open-ended stub. Near the frequency
where that stub is a quarter guided wavelength long, its input shorts the port, the line between
the ports becomes a low-loss resonator and S is wrong. The bad band moves with the overhang length,
not with the mesh, the absorber or the simulation time.
The port refuses the stub, also with `skip_preflight=True`, when an odd quarter-wave resonance of
its effective length lies within a factor of 1.5 of the band the call reads
(`rfx/preflight/line_stub.py`); a stub resonating outside that band is only reported by
preflight, and S is not checked for it. The effective length is the node-to-node
overhang on the grid plus the open-end extension (Hammerstad's closed form for a microstrip; none is added
for a coax pin). The message reports the lengths on the grid and the declared lengths, the extension and the
effective length, and prints the port's grid-node coordinate along its axis.
Start the signal strip at that coordinate (the port's grid node), so it covers the port node
and nothing behind it. The absorber pads lie outside
the declared domain, so drawing the strip "into the absorber" is not possible today.
→ [#1512](https://github.com/bk-squared/rfx/issues/1512)

### The default wire port is a mesh-sized probe.

Tracker: #1373

`add_port(..., extent=...)`
with `radius=None` acts as a probe of radius approximately `0.20 * dx` on a
square transverse mesh. Refinement therefore changes its series inductance;
it is not a fixed-radius physical pin. Opt in to `radius=` in metres to use
the local thin-probe model, within `radius <= 0.20 * dx` and a locally uniform
square transverse stencil and uniform spacing along the pin. Axial grading
across the port refuses this model. It is carried by nondispersive 3-D
second-order Yee `run()` and `forward()`, on one device or several; other solvers,
Debye/Lorentz, tensor/design-box updates and traced mesh metrics refuse it.
Resolve larger wire ports geometrically with a coax feed or a volume wire.
Each halving of the transverse cell adds `ω·(μ0·ℓ/2π)·ln 2` of series reactance
to a port of length `ℓ` without `radius=`, about 5 Ω per halving at 5.8 GHz for a
1 mm port, so its S does not converge with the mesh. `run()` and `sim.preflight()`
report it (code `wire_port_no_radius`). A YAML lumped port takes `radius:` the
same way, with the same bound and the same refusal.

### A multi-cell wire port's load-plane reflection misses the 2 dB and 0.1 bars on two loads at 25 µm.

Tracker: #1614

A wire port (`add_port(..., extent=...)`) on a coaxial line closed by a load, read at the load plane
over 1–10 GHz with 25 µm cells, gives a reflection magnitude 2.31 dB off at 8.8 GHz on a 2·Zc load and
`|Γ|` 0.102 at 10 GHz on a matched load. The bars are 2 dB and 0.1. Short, open and Zc/2 loads stay inside them.
The error ripples over frequency, with maxima that grow toward 10 GHz.
No message reports it. Only one mesh was measured, so the trend with the cell and the cause are not known.
Check a wire-port reflection on three meshes before using it as a converged number.

### The coax→microstrip transition reads more power out than in.

Tracker: #1479

`compute_coax_msl_transition(...)` returns a matrix whose columns carry more power
out than the incident wave brings in, so it is not passive. The call refuses it by
default (`strict_passivity=True`, raising `ValueError`); pass `strict_passivity=False`
to get the diagnostic matrix back with a `UserWarning`. Transition S-parameters from
this path are not usable as an absolute power reference. Cross-family transition
calculators are not pursued further before 2.0, so the excess will not be attributed or
corrected; use the single-family extractors separately (`compute_msl_s_matrix(...)`,
the coaxial two-port and line-reflection calculators) for a result you can cite.
#838 was closed as not planned before 2.0 (decided 2026-09-20); this is a standing limitation.

Part of the excess came from the coax side's plane voltage, which counted the cells next
to the inner and outer conductors at half weight and so read 0.84 of the line voltage on
the 0.1 mm test mesh. With the whole-cell voltage, the short test fixture's |S21| falls
from 3.81 to 3.22 at 6 GHz and its column power from 14.6 to 10.5; the matrix is still
far from passive, and the rest of the excess is not attributed.

### The microstrip S-matrix can come back non-passive, and says so.

Tracker: none — accepted limitation (warns at default: tests/unit/autodiff/test_msl_forward_identity.py::test_a_nonpassive_extraction_is_named_bin_by_bin_in_one_warning)

`compute_msl_s_matrix(...)` returns the S it extracted; it no longer projects it
onto the passive set by default, so that the S a user reads and the S a gradient
differentiates are one function. When a bin's largest singular value exceeds 1
(beyond float32 rounding) the call warns with the bin count and the worst value,
and `result.sigma_max_excess` carries the per-bin amount. A passive structure
cannot do that: read it as a record that ended before ring-down or a mesh too
coarse for the geometry (`settling_db`, `reliable`), or as an absorber too close
above the line (next entry), not as gain.
`enforce_passivity=True` returns the projected matrix instead. On a reflecting fixture
a test checks the raw S: the microstrip chain battery holds its maximum column power at
or below 1.02 on the open-stub notch on the 25 µm mesh the claims rest on, in
`tests/oracle/test_msl_chain_battery.py`. Read the current fixture for the
measured notch and thru values. Coarser meshes are
reported, not tested.

### A microstrip S-matrix exceeds unit column power when the absorber above the line is close and thin.

Tracker: none — accepted limitation (warns at default: tests/unit/autodiff/test_msl_forward_identity.py::test_a_nonpassive_extraction_is_named_bin_by_bin_in_one_warning)

On a passive microstrip, `compute_msl_s_matrix(...)` can return S whose column
power is above 1 on nearly every bin. On a synthetic board with a 0.3 mm
substrate, the excess is about 0.2 % on a matched thru and 1 % on an open-stub
notch, 0.01–0.05 dB on `|S|`. It grows slightly as the mesh is refined. On that
board, with 1.5 mm of air above the line and 8 CPML layers, the extra power
enters the board region through the top face, from the absorber. With 4.5 mm of
air above the line, or with 16 layers (the default), the column power stays at
or below 1 on every bin from 1.5 to 7 GHz. A wider lateral margin does not
change it.

A power balance on a closed box around the board (diagnostic: one board, one
50 µm mesh, 1.5–7 GHz) measures the same thing directly. With 8 layers, the
top and side faces return up to 0.2 % of the driven power into the board
(2.0×10⁻³, near 5.8 GHz). On this board's |S21| that is at most 0.009 dB. A
record 1.5 times longer moves the balance by at most 0.1 %. With 16 layers on
every face the returned power falls to the noise level (+1.1×10⁻⁴), at 1.26
times the cells of the 8-layer run. Sampling E and H from one absorber profile
at their own Yee positions lowers the 8-layer excess to 0.72 of its size and
does not remove it. The absorber depth matters more than the E/H profile
offset. The returned power follows the strip, not the domain: moving the side
absorbers 6 mm farther from the strip ends changes it by at most 1.3 %, and its
swing over frequency has the strip's one-way transit time (within 3 %). Whether
the thin top absorber reflects the line's near field or is not passive for it
is not measured. Not measured either:
16 layers on the top face alone, other substrates and air heights, and how the
balance moves with the mesh.

An optimiser objective that rewards `|S21|` can exploit that 0.2 % of
returned power. Keep 16 layers (the default) when passivity matters.

The non-passive warning from this call tells you to suspect a record that ended
early or a coarse mesh. That advice does not fit this case: these records ring
down to −86 to −95 dB, and a finer mesh makes the excess slightly larger. Give
the line more air below the top absorber, or keep the default 16 layers.

### The fitted microstrip propagation constant differs from the Hammerstad–Jensen closed form, and the difference is not attributed.

Tracker: none — accepted limitation (diagnostic, does not enter the result: tests/unit/sparams/test_msl_fitted_beta_does_not_enter_s.py::test_s_is_bit_identical_when_the_fitted_beta_is_wrong)

The fitted `beta` sits off the
closed form on every in-band bin, by an amount that changes with the cell size;
no trend is claimed, because the strip was one cell thick in each measured run.
It is a diagnostic that does not enter S (pinned by a test), so it is not a bias
on the phase of an S-parameter. #830 is closed as characterized.

### Microstrip `Z0` and `beta` are unreadable when the probes sit near a reflector.

Tracker: none — accepted limitation (diagnostic, does not enter the result: tests/unit/sparams/test_msl_power_normalization.py::test_reference_metadata_is_neither_fitted_z0_nor_the_50_ohm_load)

The N-probe fit rides the standing wave instead of measuring the
line: the fitted `Z0` follows the reflection's ripple, while `reliable` can stay
True on every in-band bin. `S` itself is referred to each port's declared
`impedance=` (`reference_impedances`), with its waves formed at the analytic
Hammerstad-Jensen line impedance (`line_impedances`), not at the fitted `Z0`. Judge by
`probe_clearance` and `beta_railed`, not by `reliable`; the
fix is a longer uniform feed or a moved reference plane. Every warning about
this condition carries one shared text (`rfx.preflight.msl.MSL_PROBE_CLEARANCE_EFFECT`).
Settled in #726 (closed): the guard and preflight used to contradict each other
about this, and a measurement decided it.

### On a straight microstrip, the MSL port's |S21| exceeds 0 dB near 10–12 GHz and depends on where the first probe sits: the port misreads power within about 10 mm of its plane.

Tracker: #2270

On a synthetic matched straight microstrip (εr 2.2, h 0.8 mm, 60 mm, no discontinuity), `compute_msl_s_matrix(...)` with the first probe plane 5 mm from each feed plane returns `|S21|` up to +0.096 dB above 10 GHz (largest singular value of S up to 1.015), and `|S21|` spreads by 0.19 dB across first-probe positions of 5, 12 and 20 mm.
At default `compute_msl_s_matrix` warns that S is not passive, with the number of such bins and the worst one (`sigma_max_excess` holds the per-bin amounts).
The spread does not fall with the cell (0.186 dB at 0.2 mm, 0.172 dB at 0.1 mm), and doubling the lateral and top air margins and the absorber changes neither number by more than 5 %.
The solved field carries no gain: between the two first-probe planes the real power flux falls by 0.08 to 0.41 dB at those bins.
The excess comes from the port's V·I* reading of power near its plane: at 10–12 GHz it differs from the real flux by up to 10 % within about 10 mm of each feed plane, and by at most 2.5 % below 9 GHz.
The port splits V and I into waves at a constant closed-form Z0, and on matched lines that split moves the reading with the probe position (next entry, #2393).
Workaround on lines like this one: place the first probe plane 12 mm or more from the feed plane (`add_msl_port(..., n_probe_offset=...)`, in cells: 60 cells at 0.2 mm); with it at 12 mm the same line's `|S21|` exceeds 0 dB by at most 0.017 dB (largest singular value up to 1.0034).
This is one line's evidence (record kept with the project); a clearance rule or default waits for a sweep over several lines.
On a one-stub line, moving only the probes changes `|S21|` by up to 0.47 dB on the null's upper shoulder (1.28 dB on a 45 mm feed) and the null depth by 2.8–3.4 dB; at bins where S stays passive no message reports it.
The automatic first-probe offset is at least five substrate heights (4 mm at h = 0.8 mm), inside the zone above.

### On a matched microstrip line, the MSL port's |S11| and |S21| move with where its probes sit.

Tracker: #2393

The MSL port splits the recorded voltage and current into forward and backward waves at one constant impedance, the Hammerstad–Jensen Z0 of the declared strip.
The line's own impedance changes with frequency, so part of the forward wave is counted as reflection, and how much depends on the probe positions.
On three matched lines the test `tests/oracle/test_msl_probe_offset_spread.py` puts the first probe at 5h, 12h and 25h from each port.
The spread of |S11| across those three offsets, largest over the band, is 0.043 (εr 2.2, 0.8 mm substrate, 1–12 GHz), 0.110 (εr 3.66, 0.254 mm, 1–40 GHz) and 0.173 (εr 10.2, 1.27 mm, at 12 GHz).
The spread of |S21| on the same lines is 0.010, 0.009 and 0.023.
These are the numbers on the test's meshes (cells of h/4, h/3 and h/4); other meshes were not measured.
Until a fix lands, read the |S11| of a well-matched microstrip structure as uncertain by about this much, and keep the probe offset fixed when you compare designs.

### A source on a smoothed or conformal edge is injected with a coefficient that differs from the field update's.

Tracker: #1373

The field increment a source adds on its edge is the update's coefficient times the drive. A rectangular-waveguide port and a plane-wave source still compute that coefficient apart from the update; so do lumped, wire and MSL port drives on smoothed and conformal edges (next entry).

- A waveguide port beside a conformal PEC wall (`Boundary(..., conformal=True)`): where the wall cuts a driven aperture edge (the wall-to-wall length rounds up to the next whole cell), the port drives that edge with its own coefficient, 1.5 times the field update's on that edge. On a 40 × 20 mm guide (TE10, 5 to 7 GHz) the launched wave, and with it every absolute field and probe amplitude, reads 2.5 % high with 3 mm cells, 1.0 % with 1.2 mm cells and 0.63 % with 0.75 mm cells; the excess is proportional to the cell size, 0.83 % per millimetre of cell. |S11| and |S21| moved by at most 3e-4 at 1.2 and 0.75 mm, because the port measures its incident wave at its own planes. With the walls on mesh nodes nothing changes.
- A waveguide port with `subpixel_smoothing` is refused when smoothing changes the permittivity in the port section.
- A plane-wave source (`add_tfsf_source`) refuses a smoothed or conformal edge on its injection planes, on every path (`tests/contracts/test_tfsf_vacuum_planes.py::test_treated_edges_on_injection_planes`).

Not fixed in 2.2.0. Read absolute amplitudes from a port beside a conformal wall as high by these amounts, or put the walls on mesh nodes. Measured 2026-11-03 on the cut-cell conformal model; record kept with the project.

### Surface-impedance sheets and the Kerr term read a different permittivity on their edge than the field update.

Tracker: #1961

The field update divides the curl on each electric-field edge by one permittivity: on a plain edge, the mean of the four
cells around the edge; on a smoothed edge (`subpixel_smoothing=True` or `"full_tensor"`), the smoothed value; on an edge
that conformal PEC cuts (`conformal_pec=True` or `Boundary(conformal=True)`), the permittivity divided by the edge's
conformal weight (its free length, limited by the cut faces around it). The operators below act on the same edge every
step but form their own quantity from another permittivity, so they drive, load or correct the field as if the edge held
another material. Each number is |a − b| / |b|, where a is the operator's quantity and b is the same quantity formed
from the update's permittivity, on one edge of one test scene (an εr 4 sphere or a PEC sphere, 1 mm cells, uniform mesh,
one device; `run()` and `forward()` give the same value).

- A surface-impedance sheet (`add_thin_conductor(..., surface_impedance_f0=...)`) and the Kerr term
  (`add_material(..., chi3=...)`) read the permittivity of one cell; no option is needed for this. On an edge with one
  εr 4 cell and three εr 1 cells around it, both read 4 where the update reads 1.75. The sheet's per-step loss exponent
  σ·Δt/(ε0·εr) is then 0.44 of the update's (difference 0.56), and the Kerr term reads 2.29 times the update's
  permittivity (difference 1.29). In a Kerr medium on an edge next to a conformal cut, whose conformal weight a cut face
  beside it limits to 0.93, the Kerr term reads 4 where the update reads 4.30 (difference 0.069). A sheet with smoothing
  or conformal PEC, and the Kerr term with smoothing, are refused.
- A rectangular-waveguide port on a conformal edge has the same fault; the entry above gives its measured effect.

Lumped, wire and MSL port drives and lumped RLC elements (`add_port`, `add_msl_port`, `add_lumped_rlc`) read the
update's own coefficient on every edge: plain, smoothed and conformally cut. The sheet and the Kerr term read the
update's permittivity on a plain edge where the four cells around the edge hold one material. The effect of these differences
on S-parameters, fields and gradients is not measured. Measured 2026-11-15; record kept with the project.

### A waveguide port built without a uniform Grid is unchecked in a caller's own loop.

A configuration built with scalar `dx` or a graded grid warns when the aperture exceeds the declared guide size by more than half the last cell on either transverse axis; the walls remain unchecked when stepped through `inject_waveguide_port` and `update_waveguide_port_probe` in the caller's own loop.
A whole node plane there drives and reads the row outside the high electric wall; use the cell span `(0, n - 1)`, as `Simulation` builds it.

Tracker: none — accepted limitation (warns at default: tests/unit/ports/test_waveguide_whole_plane_aperture.py::test_initializers_warn_without_uniform_grid)

### WR-90 S-parameters near cutoff need a record of about 16 000 steps at a/45.

Tracker: none — accepted limitation (warns at default: tests/unit/sparams/test_settling_witness.py::test_settling_populated_and_truncation_warning_fires)

Below f/fc = 1.2 the guide's slow group velocity rings long after the source
ends. At a/45 cells the run's settling check passes from about 16 000 steps
(−51.6 dB); there |S11|²+|S21|² is within 1.8e-3 of a 64 000-step record, inside
the check's own per-bin bound. On 8 000 steps the sum moves by up to 9e-3 per bin
and the settling check fails that record (−39.7 dB against −40 dB). From
32 000 steps on, the remaining 1.2e-3 to 1.3e-3 of |1 − |S11|² − |S21|²| does
not fall with record length: it is the a/45 mesh. Measured on an empty WR-90 at 7.0–7.9 GHz
(f/fc 1.07–1.20) on 2026-10-28; record kept with the project.
Give `n_steps` explicitly when the band starts below 1.2 fc.

### When two modes of a waveguide cross-section share a cutoff (for example TE10 and TE01 on a square guide, or TE20 and TE01 on a guide twice as wide as it is high), `rfx.eigenmode.solve_waveguide_modes` returns that pair in an arbitrary basis chosen by the eigensolver, which can differ between machines.

Tracker: #2725

The waveguide ports' own mode profiles do not have this problem: they take the requested mode within the pair.

### MSL S-parameters under `eps_override` refuse an override that varies under a port line, and a call under `jax.jit`

Tracker: none — accepted limitation (refused at default: tests/unit/sparams/test_msl_feed_shift_eps_override.py::test_override_varying_along_the_shifted_length_is_refused_by_port_name)

`compute_msl_s_matrix(eps_override=...)` moves S from the probes to the feed planes along the line the override sets under each port (its beta and Z0; the gradient carries that term). Two cases are refused by name:

- an override whose permittivity varies under a port line between its feed plane and its first probe. The move uses one line over that length; keep the override uniform there or move the probes with `n_probe_offset`.
- a call under `jax.jit`, where the override has no values to check (`::test_override_without_values_is_refused_under_jit`). Call it outside `jax.jit`; `jax.grad` and `jax.jacfwd` work.

### A two-stub microstrip stopband filter's group delay misses the 1 % bar against HFSS

Tracker: #2203

On a two-stub microstrip stopband filter on the edge-aware non-uniform mesh (`rfx.mesh_edges.edge_aware_profiles`),
the passband group delay over 5.0–5.6 GHz is 1.06 % above an HFSS reference. With the mesh it changes little at the fine
end: 1.44, 1.14 and 1.06 % at h/4, h/8 and h/12, where h is the substrate thickness. Near the stopband edges the per-bin
difference is larger and not flat. The stopband minima are shallower in rfx: −35 and −41 dB, against −39 and −59 dB in
HFSS. The cause is not known. On a uniform mesh the gap is larger and falls first order with the cell (3.86, 2.44 and
1.96 %), because a free PEC sheet edge is solved 0.35 cell beyond its last node.
The stopband minima also move with the feed length: with longer feeds they are 5.8 and 16.6 dB deeper, and the
passband `|S21|` moves by at most 0.68 dB. No message reports it (largest singular value of S 1.001). The MSL port's
power reading near its plane (entry above) probably contributes, because the feed change moves the probes; this is not measured.

### On a lossy line or fill, |S| at the reported reference plane is |S| at the plane where it was measured

Tracker: #2944

A port measures its waves at one plane and moves them to the reported reference plane. The move multiplies them by
exp(∓jβs) with a real propagation constant β. The attenuation α over the shifted length s is left out.

Where the move happens:
- the microstrip calculators move S from probe 0 out to the feed plane;
- the waveguide port moves its waves from the modal record plane to `reference_plane`;
- the opt-in wire reference planes de-embed with phase only.

On a lossless line nothing is lost. On a lossy substrate, a lossy fill or a conductor with loss, the phase is right and
the magnitude is that of the measuring plane:
- a move outward, away from the device (the microstrip feed plane), reports |S11| high by exp(2αs) and |S21| high by
  exp(α(s₁ + s₂));
- a move inward reports them low by the same factors.

For example, a microstrip on h = 1.0 mm, εr 4.4, tan δ 0.02 with probe 0 at 5 mm from the feed has α_d ≈ 2.1 Np/m at
6 GHz. Its feed-plane |S11| reads about 0.18 dB high. This is an estimate from the dielectric-loss formula, not a
measurement.

Until it is fixed, take the reference plane at the measuring plane, or correct |S| by the line's α yourself. The fix
moves S with the complex propagation constant α + jβ for every port kind.

## Ring-down completion and the early stop

Ring-down completion (extrapolating the time signal past the end of the run from fitted decaying poles; cf. the CST AR filter).

### A completed forward ring-down value can have a failed lazy report without warning.

Tracker: #1961

A `forward(ringdown=...)` completion whose error witness, passivity or traced
witness fails returns finite `s_params` and warns nothing; read
`result.ringdown.report.ok` (the report is `None` while traced) and
`rfx.ringdown.gradient_witness` before trusting a completed value or its gradient.
`run(ringdown=...)` warns on the same failures.

### A pair of modes that no window of the record separates is completed as one mode, and every early-window vs late-window comparison agrees.

Tracker: #1514

`run(ringdown=...)` and
`forward(ringdown=...)` identify the ringing's poles on the record's second
half. Two resonances closer than the record can resolve come back as one
blended pole, and the completion from any window of that record makes the same
blend. The comparison of an early-window and a late-window completion then agrees while the completed S can be several
percent off near the pair. What to do: record longer. In the measured case, a
pair 0.03 % apart was resolved at about a fifth of its decay time.
#1381 was closed as a stated limit (decided 2026-09-30); this is a standing limitation.

### The early stop can end the run before a weak unresolved high-Q pair is resolved.

Tracker: #1514

The rule is `run(..., until_identified=True)`: stop once the early-window and late-window completions agree
twice and the record is half the decay time of the slowest mode that moves S.
A weakly coupled pair of very high-Q modes (Q about 5e4, each feature 0.02–0.03
in |S|, 40–60 MHz apart) beats inside a short record. The record reads the beat
as fast decay, the blended pole's weight falls below the bar, and the floor
does not hold. Measured on a synthetic one-port: the stop fires at 2250 steps
with the completed S up to 3e-2 off at the pair while that comparison reads about 1e-4.
Stronger or single weak modes do hold the stop. What to do: where weak high-Q
features matter, use a fixed record (`run(n_steps=..., ringdown=...)`) long
enough to resolve them. Pinned as a strict xfail in
`tests/unit/sparams/test_ringdown_early_stop.py`.

### A long identification window can be refused at stage 2, the original-rate residue fit, above the memory bound.

Tracker: #2874

`run(ringdown=...)` and `forward(ringdown=...)` identify poles in two stages:
- stage 1, the pencil, runs on the decimated record;
- stage 2 fits the residues at the original sample rate and needs `1.71 × 16 × 5 × n_win × r` bytes, where `n_win` is the window length in samples and `r` the kept rank.

When that exceeds `rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES` (4 GiB by default), the window is refused by name with `IdentificationMemoryRefused`. The message names the stage, the dimensions, the memory needed and the bound. For example, n_win = 80,000 at r = 600 needs 6.1 GiB. No number is returned for that window.

What to do:
- raise `rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES` when the device has the memory;
- or shorten the identification window.

#2874 plans a streamed stage-2 fit, whose memory scales with `r²`.

### A long identification window can exhaust device memory in the traced pole step of `forward(ringdown=...)`, with no refusal.

Tracker: #2874

`forward(ringdown=...)` moves the identified poles by one frozen Gauss-Newton step, so a derivative reaches them (`rfx/ringdown/_jax.py`, `completion` and `_pole_step`). For each identification window the step builds arrays of `M × C × K` complex values, 8 bytes each (complex64; 16 bytes with `jax_enable_x64`): the pole-derivative basis, and its projection, which is the step's Jacobian. It factors that Jacobian by QR once for the value and once more for the derivative, and each factor is about as large again.
- `M` is the window length in samples;
- `C` is the number of fitted channels: with ports, the voltage and the current of each port plus the identification probe columns; with probe spectra only, the identification probe columns;
- `K` is the window's pole budget, the smaller of the pencil's capacity and `rfx.ringdown.TRACED_POLE_BUDGET` (512 by default).

Two windows are identified: the main window `[window_start × T, T]` and the early-start comparison window `[window_start / 2 × T, T]`, which is 1.5 times longer at the default `window_start = 0.5`. For example, a 120,635-step record with 8 channels and K = 512 gives M = 60,317 and 90,476, so each array takes 60,317 × 8 × 512 × 8 B = 1.98 GB and 90,476 × 8 × 512 × 8 B = 2.96 GB. Nothing checks this memory before the step runs: a device without it ends the call with the allocator's `RESOURCE_EXHAUSTED`.

What to do:
- shorten the identification window: a larger `RingdownSpec.window_start`, or a shorter record;
- or lower the pole budget by assignment before the call is traced, for example `rfx.ringdown.TRACED_POLE_BUDGET = 256`; a function already compiled by `jax.jit` keeps its old budget until it is traced again. A window that keeps more poles than the budget then returns NaN, and its status reason names the kept count and the budget.

#2874's streamed fit is planned to cover this step too.

### A plain record shorter than a weakly coupled resonance's decay misreads S near it on any port, and the end-of-run decay check can miss it.

Tracker: #1514

A lossless PEC box with a one-cell port rings on its first
mode for far longer than a plain record; the port is its only loss. On a plain record,
|S11| near the mode departs from 1, and the per-frequency decay check (`settling_db`) at the end of the run can pass: its reference
peak is the direct response, and the weakly coupled mode rings below it.
On the same record, one-cell wire-port completion restores |S11| to 1 within 1e-3 with its early-window vs late-window comparisons passing
(`tests/unit/sparams/test_sparam.py::test_wire_port_pec_cavity_s11_around_the_first_mode`).
`run(ringdown=...)` and `forward(ringdown=...)` refuse one-cell lumped ports (`add_port` without `extent=`).
What to do: record well past the slowest mode's decay, or use a one-cell wire port
(`add_port(..., extent=dx)`) with `run(ringdown=...)`.
#1255 was closed as a stated limit (decided 2026-10-01); this is a standing limitation.

### A probe at the rounding level of a much stronger probe in the same cell is not judged by the decay check.

Tracker: none — accepted limitation (diagnostic, does not enter the result: tests/unit/sparams/test_settling_rounding_only_records.py::test_weak_ringing_record_at_rounding_tail_is_skipped)

When two point probes of the same field kind (two E components, or two H components) sit in one Yee cell, the
per-frequency decay check (`settling_db`, `settling_witness`) skips a record as rounding only when its peak is 52 dB or
more below the stronger record (2.4×10⁻³ in float32) and its level over the last 10 % of the record is at most
2.9×10⁻⁴ (−71 dB) of the stronger record's peak. Such a record cannot be told apart from float32 rounding at that point:
records on a field null were measured ending at 1.4×10⁻⁵ to 2.8×10⁻⁵ of the co-located peak, still about 0.2 of their own peak.
A weak record still ringing above −71 dB is scored like any other record. The skipped records are listed in
`result.settling_witness["skipped_records"]` with their `tail_ratio`.
What to do: to judge a component that weak, put a probe of its own at a neighbouring cell, or run in float64.

### The printed ring-down settling level can move by more than 10 dB when the fields change only at rounding level.

Tracker: #1687

`settling_db` is the share that the tail past the end of the record adds to the in-record transform, extrapolated from
poles fitted to the record after the source ends. A second fit over a check window (the first 90 % of the main
window, or in a long enough record a window that starts earlier) gives the identification-error indicator
`settling_witness["error_per_bin"]`; the check window does not enter the printed level. A rounding-level change of
the record (a code change, or another host's arithmetic) can change which poles the main-window fit returns, and the
printed level follows. Measured on `examples/tutorials/antenna_farfield_pattern.py` (a point source in a
60 mm vacuum cube with a CPML, 400 steps, one probe, read at 8 GHz) across one rfx change that moved the probe's field
by at most 1.1e-5 of its peak (7.5e-3 of the decayed field's peak inside the main window): the fit over the 0.55 ns
main window returned a slow non-oscillating pole (time constant 5.1 ns) before and not after, so the level moved from
-66.8 to -80.2 dB (share 4.6e-4 to 9.8e-5), while the check-window fit returned it on both sides (5.1046e-4 against
5.1053e-4) and the directivity stayed 1.761 dBi. After the change the error indicator read 4.15e-4, 4.2 times the share it qualifies,
and the status stayed "pass"; the same version read -80.2 dB on one host and -79.7 dB on another. At 800 and 1600
steps the two versions' shares agree within 15 % and 8 %. At the −40 dB rule (a share of 1e-2), if the main window
crosses it and the check window does not, the status is "undetermined" with `settling_db` nan, not the opposite
verdict.
The same rounding-level choice of poles reaches a gradient taken through the completion. For a weakly coupled
resonance (a lossy PEC box, eps_r 2.2, Q about 5000, its TM110 resonance at 12.425 GHz seen through a short corner wire
port), dS11/d ln(eps_r) with the poles identified from the port alone was 2.1 % off on one machine, and up to 6.8 % at
another source scale, while the gradient check (the same gradient through the completion from [T/4, T] against
[T/2, T], 1 % per bin) read 0.27 % and passed. With `RingdownSpec(identification_probes=...)` the error was 0.18 to
0.20 % on every run, and the check read it at 0.94 to 0.99 of its size.
What to do: do not read a `settling_db` difference between versions or hosts as a change of the field; read
`settling_witness["error_per_bin"]` beside it, or compare at a longer record. For a gradient at a weakly coupled resonance, pass
`identification_probes`.

### With an automatic mesh, `vmap_material_sweep`'s sequential fallback runs every swept value for the first value's step count.

Tracker: #1293

Without `dx=`, the fallback meshes each swept value on its own grid but runs every value for the first grid's step
count. Measured with `num_periods=2`: eps_r = 2 covers 200 ps and eps_r = 10 covers 80 ps, stacked on one time axis.
Each value's `settling_db`, `settling_witness` and `dt` come back per value and judge that value's own record.
What to do: pass `dx=` so every value shares one grid and time step; or use `parametric_sweep()`, which returns each
value's full `Result`.

### An open plane-wave scene can run `until_decay` far past its drive end.

Tracker: #1839

The energy stop takes its reference after the drive ends, when the pulse has already left, so the reference is the small
remainder itself. Measured on the slab tutorial's 2-D vacuum reference (drive end at step 406, `until_decay=1e-3`): the
run stops at step 25,500, against a fixed 506-step run. An empty 3-D open box stops at step 550 on the float32 floor.
What to do: on an open plane-wave scene, run a fixed `n_steps` (drive end plus the transit of the domain), or set a
lower `decay_max_steps`.

### `compute_rcs` judges the incident pulse's transit to its NTFF box, not the scatterer's own ringing after it.

Tracker: #2030

`compute_rcs` checks that its record covers the plane-wave pulse, its launch and the trip of the scattered field to the
farthest corner of its NTFF box. A shorter record returns `settling_witness["status"] == "undetermined"` with one
`AccuracyWarning`. After that trip a resonant scatterer can still radiate, and `compute_rcs` does not judge that decay:
`settling_witness["status"]` is `"absent"` and nothing warns. The witness's reason suggests an identification probe,
but `compute_rcs` takes no probe, so that advice cannot be followed here.
What to do: run `compute_rcs` with two record lengths, for example `n_steps` and twice `n_steps`, and use the RCS only
where the two agree.

## Absorbing boundaries

### Magnetic faces require the single-device second-order Yee image.

Tracker: none — accepted limitation (refused at default: tests/unit/sparams/test_pmc_waveguide_refusal.py::test_pmc_waveguide_mode_solver_refuses)

Uniform and graded run/forward place the wall on the declared E-node face.
The multi-device runners take a magnetic face through the same single-device
image (#1761). There, wire ports and lumped-port S-parameters refuse it, and on
the graded mesh so do every port, DFT planes, flux monitors, snapshots and current moments.
Subgridded paths refuse magnetic faces (the `pmc_face` row of
`tests/contracts/path_disposition.py`; the graded multi-device records:
`tests/contracts/test_distributed_graded_pmc_records.py`).
Waveguide-port execution also refuses magnetic faces: its aperture mode solver
implements PEC walls only. The fourth-order stencil refuses them because
its far neighbors have no magnetic image. Subpixel smoothing refuses a material surface within half
an adjacent cell of a magnetic face when the smoother would need the
material's even extension; extending the material at least a cell beyond
the face is supported. No issue plans these kernels or the smoothing
extension for 2.0; they stay refusals.

### A plane wave requires compatible transverse faces and an invariant absorber replacement.

Tracker: none — accepted limitation (refused at default: tests/contracts/test_tfsf_boundary_admission.py::test_finite_box_is_refused)

At normal incidence on both meshes, and at oblique Bloch incidence on the uniform mesh, dispatch refuses absorbing transverse faces for an open TF/SF plane wave when
materials on the grid, conductors or localized field updates vary along that axis,
including in the pads. The refusal also applies with `skip_preflight=True`.
For an array, declare its axes periodic; for a finite scatterer use
`closed_box=True` or `rfx.rcs.compute_rcs`. For an oblique finite scatterer, use
`add_tfsf_source(closed_box=True, theta_deg=…, phi_deg=…, polarization='s' or 'p')`
in 3-D with CPML on all six faces. On the graded mesh, for propagation along x with E along z, the
admitted wall pairs are PMC on y and PEC on z. Each
transverse axis takes only a matched pair: both faces absorbing, both the wall
that suits the polarization, or both periodic (`rfx/boundaries/tfsf.py::replacement_axes`).
A wall that does not suit the polarization is refused (for E along z, PEC on y or
PMC on z), and so is a pair that mixes a PMC face with an absorbing one. A wall
pair admits only an invariant model; finite transverse
structure with those walls is refused. UPML remains unsupported for TF/SF.
The uniform plane-wave path solves y and z (Method B: z) periodic, including their pads, when no
boundary or `boundary='cpml'` is given, and reports the affected faces in preflight. Its grid and period do not
change. On the uniform mesh it refuses absorbing or wall faces declared explicitly on those axes (a `BoundarySpec`,
`Boundary` or dict), since it would solve them periodic: declare them periodic or omit them. At normal incidence the graded mesh also keeps the declared side faces. Traced material overrides retain the legacy operator
with a warning when invariance cannot be judged. Open oblique Method B keeps its
absorbing y faces; its z faces remain periodic.
The graded mesh keeps its transverse absorbers under a plane wave (no wrap) but
applies the same rule: a laterally invariant structure runs, and on a graded profile with the uniform mesh's cells
its probe record equals the uniform mesh's within 2×10⁻⁶ of its peak
(`tests/contracts/test_plane_wave_admission_both_meshes.py`). Traced overrides keep the existing graded operator
with the same warning. A finite scatterer or an array, also one given to `forward()` as an override or a design
box, is refused there with the uniform mesh named as the remedy, since a closed box and a plane wave with periodic axes are
refused on the graded mesh (an operator that admits them is not planned for 2.0).

### With no declared side faces, a plane wave solves periodic side faces on the uniform mesh and absorbing ones on the graded mesh.

Tracker: #1961

`add_tfsf_source` without `closed_box`, with no boundary or `boundary='cpml'`, is one request that the two meshes
solve as two models: the uniform mesh makes the transverse faces periodic, the graded mesh keeps CPML on them. For
the laterally invariant structures a plane wave admits, both give the same field: on a graded profile with the uniform
mesh's cells the probe record equals the uniform mesh's within 2×10⁻⁶ of its peak
(`tests/contracts/test_plane_wave_admission_both_meshes.py`). A declaration of the side faces gives one model per
request, solved as declared or refused: periodic faces run on the uniform mesh only; CPML or the wall pair that suits
the polarization runs on the graded mesh, and on the uniform mesh only on an axis it does not wrap (Method B's y). One transverse operator for both meshes is planned after 2.0.

An oblique Bloch plane wave (|`angle_deg`| > 0.01°, `method='bloch'`) is refused before stepping with off-diagonal node coupling (`'full_tensor'` smoothing or a directly supplied node tensor), `'kottke_pec'` smoothing with a PEC shape, mask or occupancy, conformal PEC, or a PEC occupancy override with `pec_occupancy_model='kottke'`. The off-diagonal node coupling does not apply the Bloch phase, and these PEC combinations remain unsupported. Use normal incidence, omit the node coupling, turn smoothing or conformal PEC off, or omit `pec_occupancy_override`, as applicable.

### An oblique Bloch plane wave rings at its cutoff along the source plane, so a value read at f0 depends on the simulation time.

Tracker: none — accepted limitation (warns at default: tests/unit/sources/test_tfsf_bloch_cutoff_warning.py::test_bloch_cutoff_warning)

With `add_tfsf_source(angle_deg=θ, method="bloch")` (the default method), the part
of the pulse at the cutoff f_c = f0·sinθ does not
leave the source plane: it rings there for the whole record, and a rectangular-window
DFT of a probe's time series picks it up, so a value at f0 changes with the simulation
time. rfx warns when the pulse level at f_c is above −60 dB of its level at f0
(`rfx/api/__init__.py`, `Simulation.add_tfsf_source`); the warning names the angle,
bandwidth, f_c and level. Narrow the bandwidth, or taper the tail of the time series
before the DFT.

## Gradients and optimization

### A multi-device `forward()` on a uniform mesh takes no override or design argument

Tracker: #1373

`forward(distributed=True)` on a uniform mesh runs the same solve as `run(devices=...)` and returns its probe records. It refuses `eps_override`, `sigma_override`, `mu_r_override`, the PEC overrides, `design_box` and `rlc_values_override`. It runs inside a user `jax.jit`, where port S-parameters are refused by name. For a gradient with those arguments, run on one device, or use a graded mesh, where `forward(distributed=True)` takes traced permittivity, conductivity and PEC overrides.

### Port V/I and DFT plane time records (`sparam_time_records`, `dft_time_records`) carry no gradient, and nothing refuses or warns

Tracker: #2627

Every time step writes the per-step voltage and current records of lumped and wire ports, `sparam_time_records` on the `run()` and `forward()` results, under `jax.lax.stop_gradient`, on the uniform and graded meshes and on one device or several. So `jax.grad` of any quantity built from these records is exactly 0 with respect to every input, with no refusal and no warning, although a driven port's own waveform enters its V and I linearly. For the gradient of a port quantity, differentiate the S-parameters of `forward(port_s11_freqs=...)` (`ForwardResult.s_params`), which are formed from the frequency-domain accumulators, not from these records. That route runs on one device on a uniform or graded mesh, and with `distributed=True` on a graded mesh. Its gradient is refused with `distributed=True` on a uniform mesh, where the plain call runs. The route is refused when the model also has a plain source or a 0-ohm port, when one record has more than one driven port, and for a passive port with `distributed=True` on a uniform mesh. Field probes (`ForwardResult.time_series`) differentiate normally. The per-step DFT plane records, `dft_time_records` on the same results, are written the same way on one device (the multi-device paths refuse them) and carry the same zero gradient; for the gradient of a field quantity on a plane, place field probes at the cells it needs.

### A gradient with respect to conductivity does not settle with simulation time on lossless or nearly lossless design cells

Tracker: none — accepted limitation (warns at default: tests/unit/autodiff/test_conductivity_gradient_record.py::test_lossless_design)

`jax.grad` through `forward()` admits a conductivity design. Where T·σ/ε on
those design cells is not large, the gradient depends on the phase at each frequency bin set by the simulation time: a static sensitivity remains although the field
decays. A longer record does not help.
`gradient_record_length_witness` detects it; the mechanism was measured in #1571.
`gradient="adjoint"` refuses every
conductivity design (`design_sigma_override`), lossy or not, and returns
settled-spectrum gradients only.
`forward()` now attaches an advisory record `conductivity_gradient_record_length` to `result.diagnostics` when a conductivity design is supplied.

### Port-only ring-down completion can miss the gradient of a weakly coupled high-Q resonance.

Tracker: #1514

A pole and the completed S-parameter can be accurate while
its material derivative is tens of percent wrong at the resonance bin. Add
interior field channels with
`RingdownSpec(identification_probes=(((x, y, z), "ez"),))` on `run()` or
`forward()`. They inform pole identification; S still uses the port V/I
residues. The report lists snapped identification channels and per-pole
residue/RMS shares as information, without an observability threshold or a
check of their own for these channels.

Judge completed gradients with `gradient_witness(g, g_early, ringdown=result.ringdown,
bin_axis=k)` when axis `k` carries frequency: it normalizes each bin of each
parameter leaf independently and returns the worst bin, so pass only the bins
your objective reads (a near-zero gradient at an unused band edge reads large). Exactly zero reference
bins are skipped and counted. Without `bin_axis`, normalization is per leaf;
a scalar objective's gradient is already per objective. A passing early-window vs late-window value comparison
does not establish a passing early-window vs late-window gradient comparison. Where the early-window vs late-window gradient comparison
cannot be formed, compare against a record 1.5–2 times longer.
The weak-port cavity contract is in
`tests/unit/sparams/test_ringdown_identification.py` (#1419, fixed by the
identification-probe option; port-only identification remains the default).

### A gradient can be wrong by tens of percent on a record whose value is converged.

Tracker: #1514

Every frequency-domain quantity rfx differentiates is a DFT of a
finite time record. A structure still ringing when the record ends leaves a term
in every bin whose phase is `(ω − ω_r)·T`; a parameter that moves the resonance
spins that phase, and because the phase grows with the simulation time the spin
lands in the derivative magnified while staying nearly invisible in the value.
In the measured cases the sign of the slope was right and its size was not — an
optimizer reading them takes steps of the wrong length, and a check on a gradient
value passes or fails on the simulation time rather than on the physics.

The committed reproduction is a lossy-filled PEC cavity
(`tests/unit/autodiff/test_gradient_record_length_witness.py::test_cavity_gradient_needs_a_longer_record_than_its_value`):
its short record passes the −40 dB decay rule and the DFT power a user reads is
converged to 0.4 %, while the gradient with respect to the fill permittivity is more
than 10 % from the gradient of a record twice as long.

Neither existing check sees it. `settling_verdict` scores the end of the record
against its peak, which is a statement about the value. AD against a finite
difference agrees to the precision floor at every simulation time, because both
sides differentiate the same truncated record. Score a gradient you intend to
report or optimize with `gradient_record_length_witness(objective, params,
n_steps, tol=...)`: it takes the same gradient from a record `factor` times
longer (default 2) and reports, per frequency bin,
`‖g_long − g_short‖ / ‖g_long‖` over the whole parameter vector together with
the cosine between the two — so a change of step length reads differently from a
turn of the descent direction. Complex observables (an S-parameter, a DFT
phasor) are compared as complex sensitivities, since a parameter that rotates a
phasor at constant magnitude moves only its imaginary part. A per-element table
comes with the result to locate where a failure sits; it is not the pass/fail result,
because an element carrying a millionth of the dominant sensitivity moves by
100 % on its own rounding. The comparison with a longer run does not replace the −40 dB rule — a
record too short by more than `factor` can still pass it, because both runs then
carry a similar leftover — and it has no default tolerance, because the right bar
depends on the structure's Q and on what the gradient is for.
Measured in #1181; the comparison with a longer run above landed with #1186.

### An optimization on a fixed record can walk into a design that the record does not settle.

Tracker: #1514

The simulation time is usually chosen so that the starting design
settles. During the descent the optimizer can build a weakly coupled, high-Q mode:
for example a mode that the structure's symmetry keeps orthogonal to the port mode,
coupled only through a small asymmetry the descent itself introduces. Its ringing
outlasts the record, the truncated DFT reads a response the settled structure does
not have, and the objective can keep improving on the truncated record while the
settled response does not. A synthetic case: a dielectric insert in a rectangular
waveguide whose permittivity varies only along the guide keeps every higher mode
across the broad wall orthogonal to the dominant drive; once the permittivity may
also vary across the wall, a higher mode held inside a high-permittivity region
can couple to the drive very weakly and ring far longer than the record.

What to do: judge the per-frequency decay check (`settling_witness` on the
S-parameter result) on the final design, and periodically during the descent, on
a concrete solve rather than inside the traced objective. When it fails, lengthen
the record if the decay is within reach; when the decay time is far longer than any
affordable record, lengthening does not help, and constraining the design space so
that the mode cannot be excited (for instance tying parameters so the design keeps
the symmetry that makes the mode orthogonal to the drive) removes the cause. This
is a property of optimizing on a truncated record, not a solver defect. At default
settings nothing flags it during the descent, so it is tracked with the same cause
as the decay check blind spot near weakly coupled resonances.

## Examples and validation coverage

### A shipped paper example's headline numbers are not WR-90 numbers.

Tracker: #2531

`validation/tmtt_paper/waveguide_dielectric_taper.py` declares 22.86 × 10.16 mm.
Its default SMOKE test run meshes at a commensurate dx = 1.27 mm and realizes WR-90
exactly. Its paper run does not: dx = 0.5 mm realizes 23.00 × 10.50 mm
(46 × 21 cells, +0.61 % / +3.35 %) and the dx = 0.25 mm production mesh realizes
23.00 × 10.25 mm (92 × 41 cells, +0.61 % / +0.89 %). Both have a TE10 cutoff of
6.517 GHz against the declared 6.557 GHz. The −26.7 dB and −38.0 dB figures the
docstring quotes were measured on those guides.

Decided 2026-09-19 (#1122, closed): disclosed, not corrected — no erratum and no
re-mesh. The file states the guide on the grid in its docstring, in its README entry
and in the result block the run prints, and derives every analytic reference from
the walls on the grid, so a reader meets the right structure beside every number.
The SMOKE half was #1100.

### Committed examples are verified at build time, not by their results.

Tracker: #737

The
example-fidelity check builds every reachable example and pins the preflight rows
it emits; it does not step time, and it does not check that any example's printed
result is right. Six tutorials plus `hello_world` are additionally run end to end
with physics assertions. Everything else in `examples/` and `validation/` is
covered only as far as its build.
→ [#737](https://github.com/bk-squared/rfx/issues/737)

### The RT5880 patch-antenna accuracy figure is not claimed for 2.0.

Tracker: #1471

The patch-antenna accuracy figure is not claimed for 2.0 because
its simulation time predates the current decay check. It is re-measured after
2.0, with the MSL notch filter's, on a redesigned record length and mesh ladder
(decided 2026-10-04, recorded on
[#1471](https://github.com/bk-squared/rfx/issues/1471)).

---

## What this page is not

It is not the issue tracker: an entry here exists because a *user* can hit the
thing, not because work is scheduled on it. Internal bookkeeping, campaign
tracking, decisions pending, and research narrative are deliberately absent — for
open work see the
[issue tracker](https://github.com/bk-squared/rfx/issues), and for what is
supported at all see the [support matrix](support_matrix.md).

### Uniform coax ports remain experimental

Experimental uniform coax ports: same-line calculator equality, run()/forward() equality, bottom-feed, NTFF-leakage and occupancy-gradient checks define the validation scope; the antenna-level result against an independent solver is not yet shown. NTFF leakage and record-length gradient evidence must be checked for the antenna being reported.
