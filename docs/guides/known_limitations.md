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

### `plan_mesh` with a graded z mesh solves the board on the lower absorber, in a shorter domain than it reports

Tracker: #1941

When `plan_mesh(...)` grades z (a thin dielectric layer), its z cell profile runs from z = 0 to the top of the geometry
plus one margin, but the domain it reports is the geometry height plus two margins. The planner does not move the
geometry, so `to_sim_kwargs()` builds a model with the board's bottom face on the z-lo absorber and only the upper margin
of air: a 1.6 mm FR4 board planned for 1.5–3.5 GHz reports a 101.5 mm z domain and is solved over about 53.5 mm. The
lower margin, meant to keep the absorber away from the board, is missing. Until the fix lands, raise the geometry by
the margin before planning (about 50 mm in this example; the profile then spans the lower margin and matches the
reported domain), or build the model with a uniform z mesh and an explicit domain.

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

Before this change, graded run()/forward() and uniform forward() could return
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
Since this change, a source on such an edge is scaled by the same weight on single-device forward runs, as multi-device runs already did.
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

### Two lossy films with a sub-cell drawn gap can be joined

Tracker: #2146

Metal sheets join only where their drawn extents touch, so a gap drawn between two sheets stays open at any width. Lossy
films (`add_thin_conductor` below 1e6 S/m) do not follow that rule yet: a gap of less than about one cell between two films
on one plane stays open only when a cell centre lies inside it. Otherwise the edge across the gap carries the film's
conductance and the two films are solved as one. Draw such a gap at least one cell wide, or with its edges on mesh lines.

### Conductor thickness is not solved: a metal sheet has zero thickness

Tracker: #1523

`add_thin_conductor` with a metal (`sigma_bulk` at or above 1e6 S/m) builds a PEC
sheet on one node plane; its `thickness` argument is not read
(`rfx/api/__init__.py`, `Simulation.add_thin_conductor` docstring). A declared copper
thickness is therefore not part of the model, and a microstrip line built from such a
sheet reads slightly electrically long against the same line with its real thickness.
Widening the sheet does not correct it.
Preflight now reports declared metal-sheet thickness above 2 % of the nearest overlapping Box dielectric's height, with the expected electrical-length shift.

---

### Experimental ADI

Tracker: #1487

`solver="adi"` is outside the 2.0 supported scope and warns
(`ExperimentalADIWarning`) on every run. It refuses an absorbing boundary of any
kind (`boundary="cpml"`, `boundary="upml"`, or a `BoundarySpec` with an absorbing
face): its absorber is an unmatched conductivity sponge, not a CPML. It refuses a
dielectric or conductivity interface: it gives each E component its own cell's
material where the Yee paths take the four-cell edge mean, so a material face sits
half a cell from where Yee puts it; a homogeneous fill runs. Use Yee with CPML for
open structures and for any structure with a material interface; ADI runs only a
closed (`boundary="pec"`), homogeneously filled cavity. Both refusals are rows in
the support matrix's **Refused inputs** table.

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
Measure a representative run or pull back one output at a time when memory is tight.
The former 2.2-times advice was exceeded by the measured 16-output row (2.25 times).
An estimate from the compiled whole gradient is the planned replacement.

### jax 0.11.2 and 0.12.0 run the time step slower than jax 0.10.2, on CPU at every measured size and on GPU on a large grid

Tracker: #2125

Measured at two grid sizes, with and without one lumped RLC element. This is not a trend over grid size.
On one RTX A6000 at 3.05 M cells (CPML included), jax 0.11.2 takes 1.18 times as long per time step as jax 0.10.2.
On the GPU at 0.53 M cells, the two versions take the same time (1.00).
On both devices the slowdown appears with and without the lumped element, so it comes from the compiler version, not from rfx.
Peak device memory is the same on both versions.
`estimate_ad_memory` still reads 1.007 to 1.009 of the measured gradient peak with `checkpoint=True`.
On CPU (one node, 8 cores, the same two grids), jax 0.11.2 takes 2.7 to 3.0 times as long per step at 0.53 M cells.
At 3.05 M cells it takes 1.6 to 2.4 times as long.
The compiled CPU step holds 18 more domain-sized copies on jax 0.11.2, with and without the lumped element.
Peak host memory is unchanged on CPU (0.91 to 1.01 times).
jax 0.11.0 and 0.11.1 run the CPU step at jax 0.10.2's speed (1.00 to 1.01 times, 8 domain-sized copies); jax 0.12.0 is slower (5.6 times at 0.53 M cells, 2.4 times at 3.05 M cells).
The release and the change in the compiled program (domain-sized fusions 85 to 98, copies 8 to 26) match openxla/xla#50977: from jaxlib 0.11.2, chains of partial array updates are recomputed. This is a match by version and operation counts, not a proven cause.
No XLA flag found restores the step: `--xla_cpu_experimental_enable_tiling_propagation=false` leaves it 2.4 to 3.0 times slower, and jaxlib 0.11.2 rejects `--xla_cpu_use_fusion_emitters`.
Workaround on CPU with Python 3.12 or newer: install `jax==0.11.1 jaxlib==0.11.1`.
jax 0.11 needs Python 3.12; on Python 3.11, rfx installs jax 0.10.

### A finite flux box closes only when it is centred on its source

Tracker: #1628

A finite flux monitor integrates its node-centred field products over cells lo to hi-1 with full weight, half a cell off the box it reports, instead of with half weights at both ends.
The two end errors cancel on a box symmetric about the source and add otherwise: on the uniform mesh at 1 mm cells, a box reaching 4 mm from the source on one side and 6 mm on the other read the net radiated flux 0.8 % off, and boxes closed by a magnetic (PMC) symmetry face read it 4.7, 3.4, 2.4 and 1.8 % low when they reach 3, 4, 5 and 6 mm from the source; the error falls as the box grows relative to the cell (not measured over meshes).
A flux plane lying on a magnetic face reports non-zero normal flux, where the exact value is zero.
The graded mesh slices its node-centred dual widths the same way at a finite window's ends; its size there is not measured.
Until this is fixed, centre a flux box on its source and keep it off magnetic faces.

### Subpixel smoothing makes a curved high-contrast dielectric's resonance converge slowly, and can be worse than no smoothing.

`subpixel_smoothing=True` and `"kottke_pec"` keep only the diagonal of the smoothed inverse-permittivity tensor.
True gives first-order convergence at curved dielectric interfaces, worst at high contrast (eps ~30 and above).
A measured eps-38 rod resonance was off by 1–2% on practical meshes, and up to 8x further off than without smoothing.
Grid-aligned faces are not affected. Judge such resonances by at least three refinements and compare against
`subpixel_smoothing=False`; disagreement means neither series has converged.

`subpixel_smoothing="full_tensor"` uses full anisotropic averaging (Farjadpour 2006; Oskooi 2009;
Werner–Bauer–Cary 2013 node form) on uniform single-device meshes. Extinction cross section on a synthetic eps-4
sphere measured second-order convergence over three meshes. This is not a convergence-order claim for high-contrast rods.
Graded meshes, multiple devices, ADI, subgrids, UPML, and the waveguide S driver refuse this option.
Kerr and declared-shape gradients remain refused for every smoothing option.
The full_tensor diagonal can jump where an interface enters an absorber pad (triplet interior, Kottke midpoint pad); oblique TFSF refuses a noninvariant padded periodic axis; use a closed box (#1632).
Tracker: #1632

### Conformal PEC makes a curved metal body electrically larger, and is less accurate than the staircase.

`conformal_pec=True` (and `Boundary(conformal=True)`) gives each E edge cut by a metal surface the permittivity eps/w,
with w the edge's open fraction (down to 0.1); only edges with w = 0 are metal. The cut cells become a high-permittivity
skin about one cell thick, which makes the body electrically larger. On a PEC sphere against Mie, its extinction and
backscatter errors were larger than the staircase's at each of three meshes, and fell only first order (tracker #1651). A run with conformal PEC warns, except together with
`subpixel_smoothing="kottke_pec"`: that combination forms no conformal weights (the smoothed tensor treats the metal
edges instead) and gives no conformal warning. Until the Dey-Mittra update (cut H-face areas)
lands, use the staircase and judge a curved body by a mesh series of at least three refinements.
A closed-form check of the conformal update, the TM010 resonance of a hollow PEC cylinder (`tests/oracle/test_conformal_convergence.py`), fails and is marked as an expected failure until #1651.
`subpixel_smoothing="kottke_pec"` on a curved PEC body leaves a field near 1e-2 of the peak that does not decay (a PEC sphere, ka 6 and 10, three meshes), so its RCS depends on the record length (one bin moved 4.5 dB between 6 and 12 ns); `run()` warns "no ring-down settling witness" (#1651, #1632).
Tracker: #1651

## Ports and extraction

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
across the port refuses this model. It is carried by single-device,
nondispersive 3-D second-order Yee `run()` and `forward()`; other solvers,
Debye/Lorentz, tensor/design-box updates and traced mesh metrics refuse it.
Resolve larger wire ports geometrically with a coax feed or a volume wire.
Each halving of the transverse cell adds `ω·(μ0·ℓ/2π)·ln 2` of series reactance
to a port of length `ℓ` without `radius=`, about 5 Ω per halving at 5.8 GHz for a
1 mm port, so its S does not converge with the mesh. `run()` and `sim.preflight()`
report it (code `wire_port_no_radius`).

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
#838 was closed as not planned before 2.0 (PI decision, 2026-09-20); this is a standing limitation.

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
the raw S is gated: the microstrip chain battery holds its maximum column power at
or below 1.02 on the open-stub notch at its 25 µm claims rung, in
`tests/oracle/test_msl_chain_battery.py`. Read the current fixture for the
measured notch and thru values. Coarser meshes are
reported, not gated.

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
True on every in-band bin. `S` itself is normalized with the analytic
Hammerstad-Jensen `Z0` (`rfx/sparams/msl.py`, `reference_impedances`), not with
the fitted one. Gate on `probe_clearance` and `beta_railed`, not on `reliable`; the
fix is a longer uniform feed or a moved reference plane. Every warning about
this condition carries one shared text (`rfx.preflight.msl.MSL_PROBE_CLEARANCE_EFFECT`).
Settled in #726 (closed): the guard and preflight used to contradict each other
about this, and a measurement decided it.

### A source on a smoothed or conformal edge is injected with a coefficient that differs from the field update's.

Tracker: #1373

Open for 2.0 in #1373 (folded from #1524 on 2026-10-07): it is to be measured
against a run with the update's exact coefficient, and refused if the error is
outside the accuracy bar.

The field increment a source adds on its edge is the update's coefficient times the drive. One place still computes that coefficient apart from the update: a rectangular-waveguide port or plane-wave source whose edges touch `subpixel_smoothing` or conformal-PEC cells. The drive on those edges is scaled by the ratio of the two coefficients; S-parameters normalised to the incident wave are not known to move, absolute field and probe amplitudes are. Until it is fixed, read absolute amplitudes from such a run as uncalibrated, or declare the source on a plain edge.

### WR-90 S-parameters near cutoff need a record of about 32 000 steps at a/45.

Tracker: none — accepted limitation (warns at default: tests/unit/sparams/test_settling_witness.py::test_settling_populated_and_truncation_warning_fires)

Below f/fc = 1.2 the guide's slow group velocity rings long after the source
ends. At a/45 cells such a band needs about 32 000 steps before |S11|²+|S21|²
stops moving with record length; on 8 000 steps it moves by up to 1e-2 per bin,
and the run's settling check reports that record (−39 dB against −40 dB).
Give `n_steps` explicitly when the band starts below 1.2 fc.

## Ring-down completion and the early stop

Ring-down completion (extrapolating the time signal past the end of the run from fitted decaying poles; cf. the CST AR filter).

### A pair of modes that no window of the record separates is completed as one mode, and every early-window vs late-window comparison agrees.

Tracker: #1514

`run(ringdown=...)` and
`forward(ringdown=...)` identify the ringing's poles on the record's second
half. Two resonances closer than the record can resolve come back as one
blended pole, and the completion from any window of that record makes the same
blend. The early-window vs late-window comparison WE then agrees while the completed S can be several
percent off near the pair. What to do: record longer. In the measured case, a
pair 0.03 % apart was resolved at about a fifth of its decay time.
#1381 was closed as a stated limit (PI decision, 2026-09-30); this is a standing limitation.

### The early stop can end the run before a weak unresolved high-Q pair is resolved.

Tracker: #1514

The rule is `run(..., until_identified=True)`: stop once WE agrees
twice and the record is half the decay time of the slowest mode that moves S.
A weakly coupled pair of very high-Q modes (Q about 5e4, each feature 0.02–0.03
in |S|, 40–60 MHz apart) beats inside a short record. The record reads the beat
as fast decay, the blended pole's weight falls below the bar, and the floor
does not hold. Measured on a synthetic one-port: the stop fires at 2250 steps
with the completed S up to 3e-2 off at the pair while WE reads about 1e-4.
Stronger or single weak modes do hold the stop. What to do: where weak high-Q
features matter, use a fixed record (`run(n_steps=..., ringdown=...)`) long
enough to resolve them. Pinned as a strict xfail in
`tests/unit/sparams/test_ringdown_early_stop.py`.

### A plain record shorter than a weakly coupled resonance's decay misreads S near it on any port, and the end-of-run decay check can miss it.

Tracker: #1514

A lossless PEC box with a one-cell port rings on its first
mode for far longer than a plain record; the port is its only loss. On a plain record,
|S11| near the mode departs from 1, and the per-frequency decay check (`settling_db`) at the end of the run can pass: its reference
peak is the direct response, and the weakly coupled mode rings below it.
On the same record, one-cell wire-port completion restores |S11| to 1 within 1e-3 with its early-window vs late-window comparisons passing
(`tests/unit/sparams/test_sparam.py::test_wire_port_pec_cavity_s11_around_the_first_mode`).
`run(ringdown=...)` currently refuses one-cell lumped ports.
What to do: record well past the slowest mode's decay, or use a one-cell wire port
(`add_port(..., extent=dx)`) with `run(ringdown=...)`.
#1255 was closed as a stated limit (PI decision, 2026-10-01); this is a standing limitation.

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

### The end-of-run decay check does not compare the record with the field's transit through an NTFF box, a DFT plane or a flux monitor, and a plane wave's source end is too early for a probe in the scattered region.

Tracker: #1237

NTFF boxes, field-DFT planes and flux monitors keep no time series, so the end-of-run decay check (`settling_db`,
`settling_witness`) cannot read them. Nothing compares the record with the time the field from the source needs to
cross them. Measured on a soft source with an NTFF box 112 mm long at 2 mm cells: a 196-step record ends 45 steps
before that field reaches the box's far corners. The check still reports pass (-94.6 dB) and nothing warns.
`compute_rcs` reports `settling_witness["status"] == "absent"` and never compares `n_steps` with the transit.
With a plane wave, every probe's source end is the waveform's end plus the total-field box diagonal. A probe in the
scattered region receives a target's first echo later: measured, the echo peaks 42 steps after that end and stays
above 1 % of its peak for 86 steps more, so the check reads the echo as free ringing.
What to do: record at least the waveform's end plus the time light takes from the plane-wave source line to the far
corner of the total-field box, then from that corner to the farthest corner of the NTFF box, plane or probe.

### `vmap_material_sweep` returns no decay check per swept value, and with an automatic mesh each value covers a different duration.

Tracker: #1293

`VmapSweepResult` carries the time series and DFT planes but no `settling_db` or `settling_witness`. This holds on
both paths: the batched one computes none, and the sequential fallback runs `run()` per value and drops its check.
Without `dx=`, the fallback meshes each swept value on its own grid but runs every value for the first grid's step
count. Measured with `num_periods=2`: eps_r = 2 covers 200 ps and eps_r = 10 covers 80 ps, stacked on one time axis.
What to do: pass `dx=` so every value shares one grid and time step, and judge a value's record with `run()` on that
value; or use `parametric_sweep()`, which returns each value's full `Result` with its check.

## Absorbing boundaries

### Magnetic faces require the single-device second-order Yee image.

Tracker: none — accepted limitation (refused at default: tests/unit/sparams/test_pmc_waveguide_refusal.py::test_pmc_waveguide_mode_solver_refuses)

Uniform and graded run/forward place the wall on the declared E-node face.
The multi-device runners take a magnetic face through the same single-device
image (#1761). There, wire ports and lumped-port S-parameters refuse it, and on
the graded mesh so do DFT planes, flux monitors, snapshots and current moments.
Subgridded and ADI paths refuse magnetic faces (the `pmc_face` row of
`tests/contracts/path_disposition.py`; the graded multi-device records:
`tests/contracts/test_distributed_graded_pmc_records.py`).
Waveguide-port execution also refuses magnetic faces: its aperture mode solver
implements PEC walls only. The fourth-order stencil refuses them because
its far neighbors have no magnetic image. Subpixel smoothing refuses a material surface within half
an adjacent cell of a magnetic face when the smoother would need the
material's even extension; extending the material at least a cell beyond
the face is supported. No issue plans these kernels or the smoothing
extension for 2.0; they stay refusals.

### With the mesh as a design variable, a ground plane still ends at the absorber.

Tracker: #1230

A conductor drawn to an absorbing boundary is continued through the absorber, so
a grounded board stays grounded inside it. When any mesh axis is traced
(`dz_profile` and its siblings under `jax.grad`), nothing is continued and a
`UserWarning` says so: the conductor then stops at the absorber entrance. Draw the
ground past the domain face by the absorber thickness, which fills every absorber
cell except the outermost row on the +x and +y faces.
→ [#1230](https://github.com/bk-squared/rfx/issues/1230)

### A plane wave requires compatible transverse faces and an invariant absorber replacement.

Tracker: none — accepted limitation (refused at default: tests/contracts/test_tfsf_boundary_admission.py::test_finite_box_is_refused)

At normal incidence, on the uniform and the graded mesh alike, dispatch refuses absorbing transverse faces for an open TF/SF plane wave when
materials on the grid, conductors or localized field updates vary along that axis,
including in the pads. The refusal also applies with `skip_preflight=True`.
For an array, declare its axes periodic; for a finite scatterer use
`closed_box=True` or `rfx.rcs.compute_rcs`. On the graded mesh, for propagation along x with E along z, the
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
`Boundary` or dict), since it would solve them periodic: declare them periodic, omit them, or use the graded mesh. Traced material overrides retain the legacy operator
with a warning when invariance cannot be judged. At oblique incidence the tilt
axis keeps its operator and is reported as not judged for invariance.
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

### The absolute value of an oblique-incidence RCS is uncertain by about ±2 dB and moves with the mesh; its pattern shape and specular direction are validated.

Tracker: #1991

`compute_rcs(..., theta_inc=θ)` with θ ≠ 0 runs the oblique open-domain plane wave (2.5-D: open in y, a thin
periodic slab in z). On a 2.2-wavelength PEC plate at θ = 20° and 40°, the specular peak of the returned bistatic RCS
is +0.86 dB above a physical-optics uniform-aperture reference at λ/30 cells, and −1.55 dB below it at λ/60, a 2.4 dB
change from mesh refinement alone. Treat the absolute value as a ±2 dB-class number at λ/30 to λ/60, not a sub-dB one.
The far-field pattern shape and the direction of the specular peak are validated (peak within 3° at θ = 40°, gated at
±6°; tests/oracle/test_oblique_rcs_specular.py). The returned RCS is that of a strip whose height is the
two-cell far-field box in z, `h_eff = 2·dx`: for a physical strip of height `L_z`, scale it by `(L_z/h_eff)²`.
What to do: compare shapes and directions directly; for an absolute level, run two meshes (λ/30 and λ/60) and
report the spread as the uncertainty of the level.

## Gradients and optimization

### A multi-device `forward()` on a uniform mesh takes no override or design argument

Tracker: #1373

`forward(distributed=True)` on a uniform mesh runs the same solve as `run(devices=...)` and returns its probe records. It refuses `eps_override`, `sigma_override`, `mu_r_override`, the PEC overrides, `design_box` and `rlc_values_override`, and it refuses to run inside `jax.jit`. For a gradient with those arguments, run on one device, or use a graded mesh, where `forward(distributed=True)` takes traced permittivity, conductivity and PEC overrides.

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
optimizer reading them takes steps of the wrong length, and a gate on a gradient
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
comes with the result to locate where a failure sits; it is not the verdict,
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

Tracker: #1122

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
example-fidelity gate builds every reachable example and pins the preflight rows
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
(PI decision 2026-10-04, recorded on
[#1471](https://github.com/bk-squared/rfx/issues/1471)).

---

## What this page is not

It is not the issue tracker: an entry here exists because a *user* can hit the
thing, not because work is scheduled on it. Internal bookkeeping, campaign
tracking, decisions pending, and research narrative are deliberately absent — for
open work see the
[issue tracker](https://github.com/bk-squared/rfx/issues), and for what is
supported at all see the [support matrix](support_matrix.md).
