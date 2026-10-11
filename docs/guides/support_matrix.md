# rfx Support Matrix

This page describes what can be used now. It does not treat a passing unit test
as RF validation, and it does not infer support for one combination from support
for another.

Known defects that a user can hit inside these limits are listed separately in
[known limitations](known_limitations.md); this page states what is supported,
that one states what is currently wrong.

Status terms:

- **supported** — documented for routine use within the stated limits
- **limited** — documented only for the exact geometry, mode, mesh, or observable
  restrictions listed here
- **experimental** — the code path runs and has regression coverage, but the RF
  evidence is not sufficient for a documented result
- **not documented** — code may exist, but there is no current public workflow
- **unsupported** — the requested combination raises an error

Accuracy and evidence statements inside a row carry one more word:

- **validated** — a committed test or frozen fixture in this repository judges it
  against a closed form or a reference; the file is named beside it
- **diagnostic** — reported, not judged (a witness, or a fitted quantity that does
  not enter the result)
- **provisional** — measured, but on one mesh or before a change that is still moving
- **planned** — not implemented
- **refused** — the code raises before the first time step; the
  [Refused inputs](#refused-inputs) table names the message's remedy and the test
  that pins it

A row marked "Pending for 2.0 — tracker N" covers code that is being restructured
for 2.0; only what is refused today is stated there.

## Current documented baseline

The general baseline is a uniform Cartesian Yee grid with `pec`, `cpml`, or
`upml` where the selected feature is listed as supported. For open domains use
`cpml`: its plane-wave reflection is measured per face, depth and frequency
(**validated**, `tests/unit/boundaries/test_cpml_reflection_coefficient.py`).
`upml` absorbs (interior energy decays, `tests/oracle/test_pml_reflectivity_upml.py`),
but no plane-wave reflection of it is measured in this repository; its parallel-E
component is attenuated only indirectly through curl coupling (`rfx/boundaries/upml.py`).
Documented sources
include point/current sources and the port families listed in the
[S-parameter support matrix](sparameter_support_matrix.md).
Documented observables include time-series probes, flux monitors, Harminv
resonances, benchmarked NTFF cases, and the explicitly listed port results.

| Feature | Status | Current limits |
|---|---|---|
| Uniform Cartesian Yee workflows | **supported** | Primary documented grid; each source and observable still has its own restrictions. |
| Uniform precision modes | **limited by observable** | `precision="float32"` is the default. `"float64"` is uniform-single-device only and is **refused** unless JAX 64-bit mode is on in the process. `"mixed"` is also uniform-single-device only; with CPML its float16 field storage raises the absorber residual floor, so it is not a substitute for float32 near low-reflection or S-parameter floors. |
| Rectangular-waveguide S-matrix | **limited** | Use `compute_waveguide_s_matrix(...)`. Uniform single-mode magnitude results are covered by the documented WR-band tests; phase, junction and nonuniform results have narrower evidence. A full-aperture port needs PEC on its transverse faces: on a uniform mesh an omitted boundary is solved as PEC with a warning, and a declared absorbing or other non-PEC transverse face is **refused**; on a graded mesh an omitted boundary is **refused** (declare e.g. `boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}` for an x-directed guide). A waveguide port with any PMC face, `solver="adi"`, and `eps_override` together with `subpixel_smoothing` are **refused**. Uniform `precision="float64"` (with JAX x64 enabled) and `precision="mixed"` honor the requested field storage, independently of accumulator and result dtype; nonuniform meshes still require `precision="float32"`. Differentiation through `eps_override` / `sigma_override` runs on `normalize=False` and `normalize="flux"` only; `normalize=True` is host-assembled and raises `NotImplementedError` for a traced override. Multimode (`n_modes>1`) is host-assembled and outside the differentiable chain. S is a modal voltage-wave ratio referenced to each port's own discrete modal impedance, so it is a power-wave S only when all ports share one cross-section. The `normalize="flux"` result dtype follows `JAX_ENABLE_X64` (complex64 by default, complex128 under x64), as `normalize=False` does. **validated** — the WR-90 chain battery (thru, PEC short, `eps_r = 4` slab against the analytic Airy result) is gated by `tests/oracle/test_waveguide_chain_battery_v18_close.py` against the live fixtures `tests/fixtures/waveguide_chain_battery/fixture_931_realized_pec_forward2_run369367259427.json` and `fixture_1012_cpml_half_cell_run369367264100.json`; read the numbers there. The older fixtures in that directory are kept for history and are not current evidence (`tests/fixtures/waveguide_chain_battery/README.md`). Fences: junctions, multimode and nonuniform meshes are excluded. |
| Microstrip-line S-matrix | **limited** | Use `compute_msl_s_matrix(...)` with the laplace/quasi-TEM model, within its own documented envelope. Raw extracted S is the default (`enforce_passivity=False`) and is the value differentiated. Passivity projection is opt-in on the concrete measurement channel and is skipped for `eps_override`. Automatic probe offsets and spacing are recounted using the realized port-runway cells; a ladder crossing a grading ramp retains the old counts with a warning, and the routing checks retain their ramp refusals. `add_msl_port` accepts `+x`, `-x`, `+y`, and `-y`; the x-only eigenmode path remains outside the public support envelope. A signal strip drawn behind a line port is an open stub; it is **refused** when an odd quarter-wave resonance of its effective length lies within a factor of 1.5 of the band read, and only reported by preflight otherwise. S uses positive real per-port power-wave references, exposed as `reference_impedances` separately from fitted Z0 and load resistance. See the [MSL power and replay contract](msl_power_contract.md). The external notch comparison is characterized rather than a tight cross-solver match. The extractor also reports a fitted `beta`; it is **diagnostic** and does not enter S (`tests/unit/sparams/test_msl_fitted_beta_does_not_enter_s.py`). **validated** — the fit recovers a known `beta` to 0.05 % on synthetic data that is exactly the fitted model (`tests/unit/sparams/test_msl_beta_fit_accuracy_envelope.py`). The fitted `beta` of a solved line differs from the Hammerstad–Jensen closed form by an amount that is not attributed (see [known limitations](known_limitations.md)). Conductor thickness is not solved: a metal strip is a zero-thickness sheet (tracker 1523). |
| Surface-impedance sheets and lossy films | **limited** | `add_thin_conductor(..., surface_impedance_f0=...)` supports patterned shapes that are sheets — realized on ONE node plane, no thicker than one local cell along their normal — on its documented uniform and graded-mesh lanes. The f0 sheet loads the edge set of the PEC sheet on the same cells, with the edge rows on a drawn free edge at half weight; **validated** — a resistive strip whose drawn edges lie on grid nodes is solved at its drawn width by the f0 and the lossy-film model alike (`tests/unit/materials/test_sheet_footprint_width.py`). `add_thin_conductor` below the PEC threshold (`sigma_bulk` under 1e6 S/m, no `surface_impedance_f0`) is one resistive film on the tangential E edges of its declared node plane, on uniform, graded and two-device runs. A non-Box lossy film must occupy one cell layer and realize its drawn area within 1 % in equivalent length, or it is **refused** unless `snap="declared"` is passed; `eps_r != 1` on a lossy film, a film on a PMC face or normal to a 2-D model's invariant axis, an f0 sheet sharing edges with a lossy film, and a film edge inside a design box are **refused**. Neither an f0 sheet nor a Box lossy film is judged by `sheet_effective_size` (see [known limitations](known_limitations.md)). It is not a general dispersive or multi-runner conductor model. |
| Lumped and wire-port S-parameters | **limited** | Use the calculation API and evidence limits for the selected port family. Do not interpret these discrete feed models as calibrated transmission-line modes. Driven lumped ports use the post-injection terminal V/I pair and the Yee half-step current phase, with the wire-family decomposition for diagonal and off-diagonal entries; every existing lumped S result changes. This extraction correction does not calibrate the feed reference plane. `add_port(..., extent=h, radius=a)` declares a physical probe-pin radius up to 0.2 times the transverse cell (`tests/unit/ports/test_wire_port_radius.py::test_radius_requires_extent_and_resolved_pin_above_bound`); a larger radius, axial grading across the port, and paths other than single-device nondispersive 3-D Yee `run()`/`forward()` are **refused**; resolve a larger pin with a coax feed or a volume wire. Omitting `radius` retains the prior probe model. Positive subcell PEC `PolylineWire` radii below half a cell are **refused**; `radius=0` selects the legacy filament model, not an arbitrary physical radius. A lumped/wire S-matrix request on a model that also has a plain source or a 0-ohm port is **refused**. Pending for 2.0 — tracker 1373 (one port V/I measurement layer on every path). |
| Coaxial-line reflection | **limited** | `compute_coaxial_line_reflection(...)` requires float32 precision, a three-dimensional second-order uniform Yee grid, CPML tokens with positive thickness on every boundary face, `cpml_axes="xyz"` (the only accepted value), positive lateral absorber clearance, no periodic axes, and exactly one `face="top"` coaxial port. It is documented over the stated frequency, termination, impedance, and mesh ranges, not as a general coaxial-network solver. The coax drive launches the TEM mode of the realized cross-section. The line ends at its declared extent without the former open stub behind the termination; earlier loss and termination readings are not current accuracy evidence. |
| Coax-to-microstrip transition | **experimental** | `compute_coax_msl_transition(...)` is a diagnostic cross-family calculation, not a validated transition. Before stepping it runs the sheet-size verdict (unless `skip_preflight=True`) and the line-stub admission; the other preflight families are not run on this path (`rfx/sparams/coax.py::compute_coax_msl_transition`). The returned matrix carries more power out than in, so it is not passive; the call refuses it by default (`strict_passivity=True`, raising `ValueError`), and `strict_passivity=False` returns the diagnostic matrix with a `UserWarning` instead. No magnitude from this path is usable as an absolute power reference, and the lane is not scheduled for correction before 2.0. For a result you can cite, use `compute_msl_s_matrix(...)` and the coaxial lanes separately, each within its own documented envelope. Listed in [known limitations](known_limitations.md). |
| Mixed lumped/wire + microstrip S-parameters | **experimental** | `compute_mixed_s_matrix(...)` is diagnostic and not in the validated set. Off-diagonal magnitudes come from Poynting flux and absolute magnitude has no external-solver comparison; off-diagonal phase mixes two reference-plane conventions. With `enforce_passivity=True` (the default) the returned diagonal is a jointly SVD-projected value, not the raw per-family measurement — read `S_raw` / `passivity_correction` for what was measured. Like the transition row above, this cross-family lane is not pursued further before 2.0. |
| Nonuniform meshes | **limited by feature** | See the table below. Support for one observable does not imply support for another. |
| SBP-SAT/subgridding | **not documented** | No current public workflow. `add_refinement(...)` on a uniform mesh selects the subgridded lane, which is **refused** by default (`validation="production"`); `validation="research"` runs it with an `ExperimentalSubgridWarning`. Use a graded mesh (`dx_profile` / `dy_profile` / `dz_profile`) for local resolution. |
| ADI | **experimental** | The Zheng–Chen–Zhang ADI lane (`solver="adi"`) is outside the 2.0 supported scope and warns (`ExperimentalADIWarning`) on every run. It runs a closed (`boundary="pec"`), homogeneously filled cavity. Absorbing boundaries, dielectric or conductivity interfaces, graded meshes, interior PEC and ports are **refused**. No accuracy and no stiff-mesh speedup is claimed; use explicit Yee for claims-bearing physics. |
| Distributed execution | **not documented** | Pending for 2.0 — tracker 1402. Execution scaling is separate from numerical correctness. The available runner is `shard_map`-based `distributed_v2`; the retired `pmap` runner raises. Distributed E permittivity and conductivity use the same four-cell edge mean as the single-device lane. Uniform `run(devices=...)` returns the S-matrix of excited single-cell lumped ports and of wire ports (`tests/unit/runners/test_distributed_wire_ports.py::test_wire_parity`); wire ports with reference planes or a radius, passive ports, lumped RLC elements, Kerr materials, MSL ports, magnetic (PMC) faces and graded-mesh ports are **refused**. Uniform distributed runs return NTFF records for far-field and gain calculations; NTFF on a graded mesh and on `forward(distributed=True)`, flux monitors and DFT plane probes are **refused**. `exchange_interval` other than 1, and an enclosing `jax.jit` on GPUs, are **refused**. These implementations do not promote the release support status. |
| Floquet/Bloch S-parameters | **experimental** | Excitation and diagnostic helpers exist, but there is no documented high-level Floquet S-parameter API. |

Materials in the general baseline are isotropic linear materials, conductive
materials, and the dispersive subsets covered by the applicable validation
tests. Optimization examples are evidence for their named objective only; they
do not validate every result produced by the same simulation.

Microstrip performance depends on the board, mesh and differentiation tape.
The current microstrip chain battery recommends 100 µm cells for its board
(`tests/oracle/test_msl_chain_battery.py`,
`test_the_ladder_converges_and_sets_a_recommended_cell_size`). Its quarter-wave
notch formula omits open-end extension and is not the physical notch reference.
The chain-battery measurements in `tests/fixtures/msl_chain_battery/fixture.json`
include memory and runtime observations: finer meshes can exceed device memory
and reduce throughput. They are performance observations, not accuracy evidence.
The graded-notch runtime measurements are in
`validation/research/multiband_nu/results/msl_notch_graded_fz_after_1213.json`.
The earlier substrate-cell recommendation predates the material-interface
correction and must not be used as a current mesh rule; use the measured ladder
for the named board rather than extrapolating below its coarsest solved rung.

The absorber measurements in
`scripts/diagnostics/open_boundary_contract/option_b/` characterize CPML as a
line termination and changes to its alpha profile. Lines should end at their
ports (`terminates=`), not in the absorber. Increasing alpha can degrade the
low band; the measured Option B was not adopted.

Current-moment far fields have a tested envelope on thin planar boards with
wire or lumped ports. Radiating geometry must lie strictly inside the monitor
slab; its extent and material/boundary refusals still apply. See
`tests/unit/farfield/test_current_moment_monitor.py`.

The edge-fed patch resonance has a measured mesh ladder against openEMS in
`tests/locks/test_patch_edgefed_resonance_harminv.py`.
Its substrate-resolution requirement is specific to that board and observable.
Oblique TF/SF validation has an angular limit; near grazing incidence the
injected-wave error is not characterized, and the auxiliary-grid instrument's
reading must not be interpreted as a PML reflection coefficient. See
`tests/unit/sources/test_tfsf_aux_absorber_reflection.py`.

An outer `jax.jit` retains forward setup arrays in the device cache until
`jax.clear_caches()`. It adds compilation cost and can add iteration overhead;
see the internal record 20261001-1364-jit-farfield-bench.
`run(devices=...)` is for capacity first. It divides cells equally along x,
also on graded meshes. Measured speedups depend on the boundary and device
count; the internal record 20261002-distributed-split-rule
uses a fitted single-device baseline rather than a direct single-device timing
for the large box.

## Refused inputs

rfx refuses an input it would not solve, before the first time step, so a model
it cannot solve does not come back as a wrong number. This section lists every
such refusal. Defects that are not refused are in
[known limitations](known_limitations.md).

### Refused by solver path

A model runs on one of these solver paths: `run()` or `forward()` on a uniform
mesh or a graded mesh (`dx_profile` / `dy_profile` / `dz_profile`), on one device
or on several (`run(devices=...)`, `forward(distributed=True)`), and the
calculators that build and run a model themselves. Each cell says what that path
does with the input:

- **supported**: the path solves it;
- **refused**: the path raises before the first time step;
- **falls back**: the path hands the model to another path, with a warning;
- **n/a**: no model with the input reaches the path;
- **unclassified**: not yet classified (no row carries this today).

A cell can name a part of its column in parentheses: `(run)` or `(forward)` for
the entry point, `(uniform)` or `(graded)` for the mesh, and in the Calculators
column the calculators that refuse. A `*` after a word marks a cell whose
behaviour today contradicts it; the row names the open issue. The Row key is the
row of `tests/contracts/path_disposition.py` the cell words are checked against.

Calculator names: `s_matrix_scan` is the lumped and wire-port S-matrix of
`run(compute_s_params=True)`; `mixed_s_matrix` is `compute_mixed_s_matrix`;
`topology_optimize` is `topology_optimize`; `waveguide_s_matrix` is
`compute_waveguide_s_matrix`; `vmap_sweep_batched` is the batched kernel of
`vmap_material_sweep`; `material_fit` is `differentiable_material_fit`.

Left out of these tables, each for one reason:

- **Subgridded runs** (`add_refinement` on a uniform mesh) are refused by default;
  see the `admission:subgrid-default` row below. Tracker 1465.
- **ADI** (`solver="adi"`) is experimental: it runs a closed, homogeneously filled
  cavity with field sources and refuses the rest (see its row in the baseline table
  above and the two ADI rows below).
- **`compute_coaxial_line_reflection` and `compute_coaxial_two_port`** build their
  own coaxial line and accept no other declared structure.
- **`compute_coax_msl_transition`** is an experimental diagnostic calculation
  (baseline table above); it refuses dispersive materials, PEC faces, absorbing lids
  and probes as well as the inputs named in its cells.

#### Materials

| Input | Row key | Uniform | Graded | Multi-device | Calculators | What to do instead |
|---|---|---|---|---|---|---|
| A Kerr (χ³) material, `chi3 != 0` | `_materials/kerr` | supported | refused | refused | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, waveguide_s_matrix, vmap_sweep_batched, material_fit) | Run it with `run()` or `forward()` on a single-device uniform mesh. |

#### Conductors

| Input | Row key | Uniform | Graded | Multi-device | Calculators | What to do instead |
|---|---|---|---|---|---|---|
| A sheet declared with `add_thin_conductor(..., surface_impedance_f0=...)` | `_thin_conductors/surface_impedance` | supported | supported | refused | refused (mixed_s_matrix, topology_optimize, material_fit) | Run it on one device. The calculators named refuse it; use `run()` / `forward()`. |

#### Sources and ports

| Input | Row key | Uniform | Graded | Multi-device | Calculators | What to do instead |
|---|---|---|---|---|---|---|
| A plain source (`add_source`) in a model given to a port calculator | `_ports/source` | supported | supported | supported | refused (mixed_s_matrix, waveguide_s_matrix, material_fit) | Remove the plain source; the calculator drives its own ports. |
| A source with an explicit `amplitude_kind=` in a model given to a port calculator | `_ports/amplitude_kind` | supported | supported | supported | refused (mixed_s_matrix, waveguide_s_matrix, material_fit) | Remove the source; the calculator drives its own ports. |
| A lumped port (`add_port`, impedance > 0) | `_ports/lumped_port` | supported | supported | supported (uniform); refused (graded) | refused (waveguide_s_matrix) | On a graded mesh, run it on one device. For an S-matrix use `run(compute_s_params=True)`. |
| A wire port (`add_port(..., extent=...)`) | `_ports/wire_port` | supported | supported | supported (uniform); refused (graded) | refused (waveguide_s_matrix, material_fit) | On a graded mesh, run it on one device. For an S-matrix use `run(compute_s_params=True)`. |
| A port that is not excited (`excite=False`) | `_ports/passive_port` | supported | supported | refused | refused (waveguide_s_matrix, material_fit) | Run it on one device. |
| A microstrip port (`add_msl_port`) | `_msl_ports/msl_port` | supported | supported | refused | refused (waveguide_s_matrix, material_fit) | Run it on one device. Use `compute_msl_s_matrix(...)` for its S-matrix. |
| A rectangular-waveguide port (`add_waveguide_port`) | `_waveguide_ports/waveguide_port` | supported | supported | falls back (run); refused (forward) | refused (mixed_s_matrix, material_fit) | A multi-device `run()` hands the model to one device with a warning; `forward(distributed=True)` refuses it. Use `compute_waveguide_s_matrix(...)`. |
| A coaxial port (`add_coaxial_port`) in `run()` or `forward()` | `_coaxial_ports/coax_port` | refused | refused | refused | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, waveguide_s_matrix, material_fit) | Use `compute_coaxial_line_reflection(...)` or `compute_coaxial_two_port(...)`. |
| A Floquet port (`add_floquet_port`) | `_floquet_ports/floquet_port` | supported | refused | refused | refused (mixed_s_matrix, waveguide_s_matrix, material_fit) | Use a uniform mesh on one device. |
| A Floquet port scanned off normal, `scan_theta != 0` | `_floquet_ports/scan_angle` | refused | refused | refused | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, waveguide_s_matrix, material_fit) | Only normal incidence (`scan_theta=0`) is solved. |
| A lumped resistor, inductor or capacitor (`add_lumped_rlc`) | `_lumped_rlc/R` | supported | supported | refused | refused (waveguide_s_matrix, material_fit) | Run it on one device. |
| A series R–L element (`add_lumped_rlc`) | `_lumped_rlc/series_RL` | supported | supported | refused | refused (waveguide_s_matrix, material_fit) | Run it on one device. |
| A plane wave (`add_tfsf_source`) | `_tfsf/plane_wave` | supported | supported (run); refused (forward) | falls back (uniform); refused (graded) | refused (mixed_s_matrix, waveguide_s_matrix, material_fit) | On a graded mesh use `run()`, not `forward()`, on one device. A uniform multi-device `run()` hands the model to one device with a warning. |

#### Boundaries

| Input | Row key | Uniform | Graded | Multi-device | Calculators | What to do instead |
|---|---|---|---|---|---|---|
| `boundary='upml'` | `_boundary/upml` | supported | refused | refused | refused (waveguide_s_matrix) | Use `boundary='cpml'`. |
| A magnetic (PMC) face | `_boundary_spec/pmc_face` | supported | supported | refused | refused (waveguide_s_matrix, vmap_sweep_batched) | Run it on one device. |
| Conformal PEC (`Boundary(conformal=True)` or `conformal_pec=True`) | `_boundary_spec/conformal` | supported (run); refused (forward) | refused | refused | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, vmap_sweep_batched, material_fit) | It is solved by a single-device uniform `run()` only. |
| Conformal PEC together with a lumped or wire-port S-matrix request | `_boundary_spec/conformal_s_matrix` | refused | refused | refused | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, waveguide_s_matrix, vmap_sweep_batched, material_fit) | Request the S-matrix without conformal PEC. |
| A periodic axis | `_periodic_axes/periodic` | supported | refused | refused | refused (waveguide_s_matrix) | Use a uniform mesh on one device. |
| `cpml_kappa_max` other than 1 | `_cpml_kappa_max/kappa` | supported | refused | supported (uniform); refused (graded) | — | Leave `cpml_kappa_max=1` on a graded mesh. |

#### Mesh

| Input | Row key | Uniform | Graded | Multi-device | Calculators | What to do instead |
|---|---|---|---|---|---|---|
| A refined region (`add_refinement`) | `_refinement/slab` | n/a (run); refused (forward) | refused | refused | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, waveguide_s_matrix, vmap_sweep_batched, material_fit) | Use a graded mesh (`dx_profile` / `dy_profile` / `dz_profile`) for local resolution. |
| A graded mesh along x (`dx_profile`) | `_dx_profile/graded` | n/a | supported | n/a (uniform); supported (graded) | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, material_fit) | Use a uniform mesh with the calculators named, or `run()` / `forward()` on the graded model. |
| A graded mesh along y (`dy_profile`) | `_dy_profile/graded` | n/a | supported | n/a (uniform); supported (graded) | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, material_fit) | Use a uniform mesh with the calculators named, or `run()` / `forward()` on the graded model. |
| A graded mesh along z (`dz_profile`) | `_dz_profile/graded` | n/a | supported | n/a (uniform); supported (graded) | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, material_fit) | Use a uniform mesh with the calculators named, or `run()` / `forward()` on the graded model. |
| `interface_eps='dual_average'` | `_interface_eps/dual_average` | refused | supported | refused | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, waveguide_s_matrix, vmap_sweep_batched, material_fit) | It runs on a single-device graded mesh only; elsewhere leave the default. |

#### Observers

| Input | Row key | Uniform | Graded | Multi-device | Calculators | What to do instead |
|---|---|---|---|---|---|---|
| A frequency-domain plane (`add_dft_plane_probe`) | `_dft_planes/dft_plane` | supported | supported | refused | — | Run it on one device. |
| A flux monitor (`add_flux_monitor`) | `_flux_monitors/flux` | supported (run); n/a (forward) | supported (run); n/a (forward) | refused | — | Run it on one device. |
| A far-field box (`add_ntff_box`) | `_ntff/ntff_box` | supported | supported | supported (uniform); refused (graded) | — | On a graded mesh, run it on one device. |
| A current-moment monitor (`add_current_moment_monitor`) | `_current_moments/block_moments` | supported | supported | refused | refused (vmap_sweep_batched) | Run it on one device. The batched `vmap_material_sweep` refuses it. |

#### Run settings

| Input | Row key | Uniform | Graded | Multi-device | Calculators | What to do instead |
|---|---|---|---|---|---|---|
| `precision='float64'` or `'mixed'` | `_precision` | supported | refused | refused | refused (vmap_sweep_batched, material_fit) | Use `precision='float32'` (the default) off the single-device uniform mesh. |
| `stencil_order=4` | `_stencil_order` | supported | refused | refused | refused (waveguide_s_matrix, vmap_sweep_batched, material_fit) | Use the default second-order stencil off the single-device uniform mesh. |
| A 2-D model, `mode='2d_tmz'` or `'2d_tez'` | `_mode` | supported | refused | refused | — | Build the 2-D problem as a thin 3-D box (`mode='3d'`). |
| A time step given as `dt=` | `_dt_pin` | n/a | supported | refused (uniform); supported (graded) | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, waveguide_s_matrix, vmap_sweep_batched, material_fit) | Leave `dt=` out on a uniform multi-device run and with the calculators named; they take the step from the mesh. |
| `dt_min_cell=` | `_dt_min_cell` | n/a | supported | refused (uniform); supported (graded) | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, waveguide_s_matrix, vmap_sweep_batched, material_fit) | Leave `dt_min_cell=` out on a uniform multi-device run and with the calculators named. |

### Refused combinations

Each row is a combination of inputs that `run()`, `forward()` or the named
calculator refuses before the first time step. The fourth column is what the
error message tells you to do; the last names the test that pins the refusal.

| Input | Row key | Where | What to do instead | Pinned by |
|---|---|---|---|---|
| `exchange_interval` other than the integer 1 | `arg:exchange_interval` | `run(devices=...)`, `forward(distributed=True)` | Use `exchange_interval=1` (each skipped exchange updates seam cells from stale values and the field grows in a lossless box). | `tests/unit/runners/test_distributed.py::TestExchangeInterval::test_interval_refused` |
| `jax.jit` (or `lax.scan` / `jax.checkpoint`) enclosing a multi-device run on GPUs, with CUDA command buffers on. CPU multi-device runs are not refused. Pending for 2.0 — tracker 1402. | `admission:multi-device-outer-jit` | `run(devices=...)`, `forward(distributed=True)` and `jax.grad` of them | Call the run, and `jax.grad` of it, without `jax.jit` (the runner compiles its own time loop), or start the process with `XLA_FLAGS=--xla_gpu_enable_command_buffer=` set before importing jax. | `tests/contracts/test_multi_device_outer_jit_refusal.py::test_T1_outer_jit_refused` |
| A Drude or Lorentz medium above its own time-step limit. The cases the check does not judge are in known limitations. | `admission:pole-time-step` | every solver path, at model assembly | The message gives the largest stable step. Graded mesh: pass `dt=` that step, or raise `eps_inf`. Uniform mesh: use finer cells, or declare a `dx_profile`/`dy_profile`/`dz_profile` so that `dt=` can be passed, or raise `eps_inf`. | `tests/unit/materials/test_pole_stability_admission.py::test_medium_above_the_limit_is_refused_with_the_stable_step` |
| A PEC `PolylineWire` with radius `0 < a < 0.5·d_min` (`d_min` the smallest local cell at its vertices), also on a traced mesh | `admission:pec-wire-radius` | every path | Resolve the wire as a volume: refine the mesh until `a >= 0.5·d`. `radius=0` keeps the legacy filament, which declares no physical radius. | `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan` |
| A wire-port `radius` above 0.2 times the transverse cell, axial grading across the port, or a path other than single-device nondispersive 3-D Yee | `admission:wire-port-radius` | `run()` / `forward()` of a model with `add_port(..., extent=..., radius=...)` | Use a single-device, nondispersive Yee `run()`/`forward()`, or resolve the pin geometrically (a coax feed or a volume wire). | `tests/unit/ports/test_wire_port_radius.py::test_radius_requires_extent_and_resolved_pin_above_bound`, `::test_radius_refuses_axial_grading_before_step` |
| `solver="adi"` with any absorbing boundary (`cpml`, `upml`, or a `BoundarySpec` absorbing face) | `admission:adi-absorber` | `Simulation(...)` | Use `solver="yee"` with `boundary="cpml"` for open structures, or `solver="adi"` with `boundary="pec"` for closed cavities. | `tests/unit/runners/test_adi_experimental_and_sponge_refusal.py::test_absorber_refused_at_construction` |
| `solver="adi"` with a dielectric or conductivity interface, including a traced array override | `admission:adi-interface` | ADI `run()` / `forward()` | Use `solver='yee'`. A homogeneous fill runs. | `tests/unit/runners/test_adi_material_interface_refusal.py::test_adi_refuses_a_declared_interface_before_the_kernel` |
| `add_refinement(...)` (subgridded run) with the default `validation="production"` | `admission:subgrid-default` | `run()`, `forward()`, S-matrix calculators | Use a graded mesh (`dx`/`dy`/`dz` profiles) for local resolution, or pass `validation="research"` to run it anyway with a warning. | `tests/contracts/test_subgrid_requires_experimental.py::test_default_refused_before_runner` |
| A full-aperture waveguide port with a declared non-PEC transverse face, uniform mesh | `admission:waveguide-faces-uniform` | `run()`, `forward()`, `compute_waveguide_s_matrix`, also with `skip_preflight=True` | Declare the `boundary=` the message prints: PEC on the transverse faces, the absorber on the port axis. | `tests/contracts/test_waveguide_boundary_admission.py::test_explicit_absorber_is_refused_at_dispatch` |
| A full-aperture waveguide port on a graded mesh with `boundary` omitted | `admission:waveguide-faces-graded` | `run()`, `forward()`, multi-device runs, `compute_waveguide_s_matrix` | Declare the walls, e.g. `boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}` for an x-directed guide. | `tests/contracts/test_graded_guide_default.py::test_default_graded_refuses_before_field_step` |
| A waveguide port in a model with a PMC face | `admission:waveguide-pmc` | `run()`, `compute_waveguide_s_matrix` | Use a full PEC-walled guide (the aperture mode solver has PEC walls only). | `tests/unit/sparams/test_pmc_waveguide_refusal.py::test_pmc_waveguide_mode_solver_refuses` |
| `eps_override` together with `subpixel_smoothing`. Pending for 2.0 — tracker 1524. | `admission:waveguide-override-smoothing` | `compute_waveguide_s_matrix` | Declare the permittivity as a material instead of `eps_override`, or switch smoothing off. | `tests/unit/sparams/test_waveguide_override_with_smoothing_refused.py::test_eps_override_with_smoothing_is_refused_before_the_solve` |
| A TF/SF plane wave on a uniform mesh whose absorbing transverse faces see structure that varies along that axis; a transverse wall pair that does not suit the polarization or mixes a PMC face with an absorber | `admission:plane-wave-faces` | `run()`, `forward()`, also with `skip_preflight=True` | Declare the axis periodic to solve an array, or use `closed_box=True` for a finite scatterer. | `tests/contracts/test_tfsf_boundary_admission.py::test_finite_box_is_refused`, `::test_inadmissible_electric_wall_is_refused` |
| A soft source on an edge the grid realizes as PEC (inside a PEC volume, on a sheet or wire, or an H source whose whole curl loop is PEC) | `admission:source-on-pec` | uniform, graded and multi-device runs, also with `skip_preflight=True` | Move the source off the conductor, or use a port. | `tests/contracts/path_equivalence/test_source_on_pec.py::test_dead_e_source_refused` |
| A soft source or one-cell lumped port whose E edge lies tangential in a declared PEC wall face; a wire port lying entirely in such a wall | `admission:source-on-pec-wall` | every path | Move it at least one cell inside the domain, off the wall. | `tests/contracts/test_source_on_pec_wall.py::test_b3_soft_wall`, `::test_b3_port_wall` |
| A lumped/wire S-matrix request on a model that also has a plain source (`add_source`, `add_polarized_source`) or a 0-ohm port | `admission:plain-source-s-request` | `run(compute_s_params=...)`, `run(devices=...)`, `compute_lumped_wire_s_matrix_via_scan`, `forward(port_s11_freqs=...)` | Remove the plain source, or call `run(compute_s_params=False)` to get raw port waves with the source included. | `tests/unit/sparams/test_plain_source_refusal.py::test_plain_source_s_request_refused` |
| A PEC sheet solved more than 1 % off its drawing (preflight error `sheet_effective_size`) | `admission:sheet-size` | `run()`, `forward()`, `compute_coax_msl_transition` | Build the in-plane profiles with `rfx.mesh_edges.edge_aware_profiles`, or pass `Simulation(..., snap="declared")` to accept the difference as a warning. | `tests/contracts/test_sheet_size_verdict.py::test_strict_refuses_inset_and_on_node_sheets` |
| A non-Box lossy film that spans several cell layers, misses its drawn area by more than 1 % in equivalent length, or cannot state its area | `admission:film-area` | every mesh | Refine the in-plane mesh or draw a Box, or use `snap="declared"` to accept the declared geometry. | `tests/contracts/test_dc_film_admission.py::test_b4_small_disc_refusal_and_declared_finding` |
| A lossy film with `eps_r != 1`; on a PMC face; normal to a 2-D model's invariant axis; sharing edges with an f0 sheet; with an edge inside a design box | `admission:film-conflicts` | every path | The message names the film and the conflict. | `tests/contracts/test_film_one_plane.py::test_c4_permittivity`, `::test_c4_geometry_refusals`, `::test_c4_f0_overlap`, `::test_c4_occupancy_design_box_refuses_film` |
| `precision="float64"` while JAX 64-bit mode is off | `admission:float64-without-x64` | `run()`, `forward()` and gradient entries | Build and run the Simulation inside a scoped `with jax.enable_x64():` block (do not switch it on at module level). | `tests/contracts/test_precision_realized.py::test_off_refuses_before_allocation` |
| A microstrip or coax signal strip drawn behind its line port whose open-stub resonance lies within a factor of 1.5 of the band read | `admission:line-stub` | line ports, also with `skip_preflight=True` | Start the signal strip at the port's grid-node coordinate the message prints, so it covers the port node and nothing behind it. | `tests/contracts/test_line_stub_refusal_1512.py::test_in_band_and_inclusive_near_edges_refuse` |
| A non-passive coax→microstrip transition matrix (default `strict_passivity=True`) | `admission:transition-passivity` | `compute_coax_msl_transition` | Pass `strict_passivity=False` to get the diagnostic matrix back with a warning. | `tests/unit/sparams/test_coax_msl_transition.py::test_transition_default_raises_on_a_nonpassive_s_and_false_returns_it` |

## S-parameter API routing

- Lumped and wire `add_port(...)` simulations use
  `run(compute_s_params=True, s_param_freqs=...)` for full matrices.
  A plain source or zero-ohm port in the model makes a full S-matrix request
  refuse; `run(compute_s_params=False)` returns raw port responses instead.
- The same port family uses `forward(port_s11_freqs=...)` for uniform or graded
  single-device S11 vectors at the requested bins. Distributed forward still
  refuses explicit port frequencies.
- `add_msl_port(...)` uses `compute_msl_s_matrix(...)`.
- `add_waveguide_port(...)` uses `compute_waveguide_s_matrix(...)` for a full
  multi-port matrix. `run()` exposes only per-port `waveguide_sparams`.
- A float32, three-dimensional, second-order uniform Yee simulation with
  CPML tokens with positive thickness on every boundary face, positive lateral
  absorber clearance, `cpml_axes="xyz"` (the only accepted value), no periodic
  axes, and exactly one
  `add_coaxial_port(..., face="top")` uses
  `compute_coaxial_line_reflection(...)` for the documented result.
  The method builds its own line, source, DFT planes, and termination; do not
  register separate geometry, RLC elements, monitors, or termination helpers.
  It derives the axial layout internally. Use `feed_impedance=` for the feed and
  `dut_impedance=` only with `termination="matched"`.
  `probe_count` must be an integer of at least three, and every requested plane
  must fit between the DUT and source.
  `compute_coaxial_s_matrix(...)` is removed. Use
  `compute_coaxial_line_reflection(...)` for reflection or
  `compute_coaxial_two_port(...)` for a through line.
- Sources, TFSF, probes, and flux monitors do not define a port impedance or
  reference plane and therefore do not produce S-parameters.

The exact schemas, metrics, and RF evidence are in
`docs/guides/sparameter_support_matrix.md`. Evidence levels are defined in
`docs/guides/physics_validation_evidence_rule.md`.

## Nonuniform-mesh classification

Nonuniform support is determined per combination. In particular, a graded mesh
running successfully is not evidence that an S-parameter, flux, or far-field
result is accurate.

| Combination | Status | What is documented |
|---|---|---|
| Periodic/Floquet port + nonuniform mesh | **unsupported** | Preflight or the calculation must fail. |
| `boundary="upml"` + nonuniform mesh | **unsupported** | The run must fail. The non-uniform runner has no UPML implementation — its absorber dispatch keys on the CPML layer count and never reads the boundary type — so this combination was accepted and silently ran CPML while `sim._boundary` still reported `upml` (issue #680). Use `boundary="cpml"` on a graded mesh. |
| NTFF + graded z | **limited** | **provisional** — a short-dipole directivity case agrees with the 1.76 dBi theory within the test's ±0.3 dB bar on one mesh (`tests/unit/farfield/test_farfield_nonuniform.py`). Other source and geometry combinations need separate validation. |
| DFT plane or full-plane flux + graded z | **experimental** | The calculation runs, but no general RF-accuracy statement is documented. |
| Finite-region `add_flux_monitor(size=...)` + graded (`dz_profile`/`dy_profile`/`dx_profile`) mesh | **experimental** | Runs on uniform and graded grids. A shared geometry resolver selects a half-open CELL window inside the physical interior; neither CPML cells nor the trailing bounding-node slot are included. Every requested endpoint outside the interior emits a clamp warning, including sub-cell overflow. Empty windows are rejected. NU selection and metre bounds use the canonical float64 node spine; area weights retain the actual per-axis cell sizes. `preflight().flux_regions` reports requested/realized bounds and indices as metadata, separate from findings, so aligned inputs do not become strict-mode failures. `size` and `center` must each contain exactly two finite tangential values, with positive sizes. `size=None` retains the legacy full allocated plane. Tests cover physical windows/area, finite-versus-restricted integration, graded fields, and material AD through the raw port hook. These checks do not establish general RF accuracy; the existing graded-normal-axis H co-location (0.5/0.5) limitation remains. See [finite flux regions](flux_regions.md). |
| TFSF + graded z | **experimental** | Only normal incidence along `+x` or `-x` (`angle_deg=0`) runs. Oblique incidence and incidence along `+z` or `-z` raise. This qualification requires a uniform injection axis x; grading x triggers the `nonuniform_tfsf` advisory because the incident line and grid waves no longer match. |
| Rectangular-waveguide S-matrix + nonuniform mesh (`dx_profile`/`dy_profile`/`dz_profile`) | **experimental** | Single-mode `normalize=True` and `normalize="flux"` run. A full-aperture guide with `boundary` omitted is **refused** (the graded mesh keeps absorbers on the transverse faces); declare the walls, e.g. `boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}`. The calculator runs the graded `run()` kernel and refuses what the graded `run()` refuses. Port injection reads the local primal cell width for its H correction and the local dual spacing for its E correction. Old boundary-cell-derived injection readings do not apply to ports inside a refined band. Analytic fixtures cover grading ratios 1--3 and relative permittivity 2 and 4. A passed Palace magnitude comparison covers `normalize="flux"`, a graded-`dy` ratio of 2, WR-90 empty/PEC-short/dielectric-slab cases, and 8.2--12.4 GHz (`max_mag_abs_diff=0.008529`, improved from `0.07009` when the lane's absorber stopped being 0.33 of a guide wavelength — #576). Other profiles, bands, phase, multimode operation, and arbitrary junctions are not validated. `eps_override` and `sigma_override` differentiation is available only with `normalize="flux"`; only `eps_override` has a nonuniform AD-vs-FD regression test. Neither AD check establishes RF accuracy. Dispatch history: until #811 (fixed 2026-09-01) this API reached the nonuniform lane only for `dx_profile`/`dy_profile` — a `dz_profile`-only simulation was silently solved on the uniform grid built from the scalar `dx` while preflight described the graded mesh (dz-only meshes with different `dt` returned bit-identical S in the falsifier baseline, `scripts/diagnostics/wr90_dz_dispatch_falsifier.py`). A `dz_profile` now dispatches here under the same restrictions, and the WR-90 graded-propagation control is recorded by #1083. This is scoped evidence, not validation of arbitrary dz profiles. The lane emits the same energy-based ring-down settling witness (`settling_db`, -40 dB aggregate warning) as the uniform single-mode lane (#827 waveguide instance). Phase envelope on this lane (derived arithmetic, not a measurement): the reference-plane shift `exp(-/+ jβΔ)` and the modal impedance evaluate β at the grid's BOUNDARY cell (`NonUniformGrid.dx`), not the cell the plane sits in. `Z_TE` does not depend on the cell size at all (the discrete `sin(β·dx/2)` equals `s_x·dx/2`), and the β difference is the second-order Yee correction `(β·dx)²/24` at the two sizes: for the committed WR-90 fixture (boundary 1.5 mm, fine 0.75 mm, discrete cutoff 6.650 GHz; `tests/fixtures/waveguide_nu_beta_cell_size_envelope.json`, replayed by `tests/unit/sparams/test_waveguide_nu_beta_cell_size_envelope.py`) Δβ = 0.057 / 0.271 / 0.652 rad/m at 8 / 10 / 12 GHz, i.e. 0.07° / 0.31° / 0.75° over a 20 mm plane offset (0.86° worst at 12.4 GHz) and 0.02° over that fixture's own 0.5 mm applied shift. Both of that fixture's port-to-reference spans lie in uniform 1.5 mm cells, so the two evaluations coincide there and the fixture cannot exercise the difference. A span that crosses cells of more than one size now raises `ValueError` (`tests/unit/sparams/test_waveguide_nu_grading_zone.py`) instead of applying a single β; a fixture with a reference plane inside the graded region, and β integration over the span, are deferred to #854 item 1. This bounds one mechanism; it is not a phase validation of the lane. |
| Lumped-port S-parameters + nonuniform mesh | **limited** | Single-device `forward(port_s11_freqs=...)` accepts requested bins for lumped and wire ports, including their V/I accumulators. This does not establish arbitrary full-matrix support; distributed forward still refuses explicit port frequencies. |
| MSL S-matrix + nonuniform mesh | **experimental** | `mode="laplace"` and `mode="uniform"` have internal settled-S11 regression coverage. **diagnostic** — the stub-notch frequency over a substrate-cell ladder on a band-graded mesh is recorded in `validation/research/multiband_nu/results/msl_notch_graded_fz_after_1213.json` and replayed by `tests/unit/nonuniform/test_msl_notch_fz_after_1213_replay.py`; it is not an accuracy statement. `mode="eigenmode"` raises. |
| Coaxial port + nonuniform mesh | **unsupported** | The request must fail. |
| Lumped RLC update + nonuniform mesh | **limited** | R/L/C ADE elements participate in the field update. Material gradients through `eps_override` and `sigma_override` are implemented on graded meshes containing lumped RLC elements. Component-value gradients remain separately undocumented; requested-frequency port extraction has the scope stated above. |
| Multi-band graded mesh (fine bands on **z, x or y**, unequal sizes allowed, ratio <= 1.4) | **limited** | The mesh and solver have bounded evidence, not general port, flux or far-field validation. The z battery below covers conservation, transition reflection, round-trip amplitude, order and autodiff. Separate PEC-closed in-plane witnesses cover cavity accuracy (W7), unequal-band reflection (E5), and x-band width autodiff (E6), each within its stated envelope. Bands narrower than two cells, in-plane absorber/port/TFSF behavior, and resolutions outside those envelopes remain unvalidated. `rfx.make_band_profile` places interfaces on node planes and enforces the adjacent-ratio cap, including seams between protected bands. Read the per-witness scope below; the timestep still follows the global min-cell CFL. |
| Volumetric PEC scatterer + nonuniform waveguide | **experimental** | The device/reference handling is regression-tested, but no RF validation is documented for arbitrary iris, post, septum, branch, or T-junction geometries. The REALIZATION changed at 2.0 (#931): a PEC volume now realizes walls at both drawn faces and its drawn extent equals its realized extent, so the regression fixtures behind "regression-tested" are recomputed for the release and a fixture that has not been is not shipped. No accuracy claim moves with them, because this row makes none. |

### Multi-band graded mesh

**What the original z battery covers.** An explicit **`dz_profile`** vector holding
fine bands along z, in any order — fine-coarse-fine-coarse-fine and other
small-large-small-large patterns included — with **every adjacent cell
ratio <= 1.4**, abrupt (a single step at the cap) or smoothly ramped.
**What is witnessed is the transition law, and band count only sums.**
Each transition reflects at a level set by its ratio r (F-S2), by the local
cells per wavelength (the (dz/lambda)^2 law in that row), and — when the
band between two transitions is narrower than about a guide wavelength —
by the band width: W6 below measures fine bands of 2 to 64 cells between
two ratio-1.4 ramps and finds the two ramp reflections adding as a
Fabry-Perot sum, bounded by twice the single-ramp value and oscillating
with width (near-maximum at about lambda_g/4, a null at about lambda_g/2).
The widest fixture in the battery has 3 fine bands / 4 transitions
(`fixtures.py`, F-S4); the narrowest witnessed band is 2 fine cells (W6).
More bands or wider bands add transitions that each behave as measured —
that is the law, not a per-count witness. **Not witnessed:** bands narrower
than 2 cells, resolutions other than the two measured (fine 30 and coarse
15.3 cells per free-space wavelength), and in-plane grading.
Before this row, only a single fine band was documented.

**Building such a profile.** `rfx.make_band_profile(edges, cell_sizes,
max_ratio=1.4, protected=..., boundary_cell=...)` realizes a declared
stack with every interface on a node plane (1e-12 m), every adjacent
ratio <= the cap — a seam between two protected bands is served by refining
the coarser band, since no ramp can sit between them — and the sum exact;
`boundary_cell` pins both end cells for the in-plane CPML contract
(`tests/unit/nonuniform/test_band_profile_builder.py`; design note
`docs/design_notes/20260907_nu_band_profile_predeclaration.md`). The builder
is a mesh guarantee, not an accuracy witness: it produces profiles inside
this row's envelope, and the row's evidence is what covers them.

**The F-S1..F-S5 witnesses below grade z only** — that harness takes a
scalar transverse cell size. The in-plane axes are covered by the later
witnesses listed after them (W7 A2/A3, E5, E6), each with its own,
narrower domain; read the in-plane bullet in the exclusions before quoting
an x/y number, because the in-plane boundary contract still pins the end
cells and no in-plane witness has an absorber.

**The cap is 1.4 — on z.** Ratios above it still construct and run; what
is lost is the accuracy class below, not stability. On a `dz_profile`,
`Simulation(...)` warns above 1.4 and preflight adds
`nu_grading_ratio_beyond_validated_cap`, both advisory. 1.4 matches the
commercial default (Tidy3D `max_scale`). **On `dx_profile` / `dy_profile`
the threshold is the pre-existing 1.3**, deliberately NOT moved to 1.4:
no witness in this row grades an in-plane axis, so there is no in-plane
provenance for moving an in-plane lock (SPEC-00 §0.2-4). Both the
constructor warning and the preflight advisory carry the per-axis value,
and both are advisory on either axis.

**Evidence** (SPEC-01 lane, tracker #780; pre-declaration
`docs/design_notes/20260829_spec01_multiband_predeclaration.md` — every
window below was committed before its measurement; harness and raw results
in `validation/research/multiband_nu/`; regression packaging in
`tests/unit/nonuniform/test_multiband_nu_envelope.py`):

| Witness | What was measured | Result |
|---|---|---|
| F-S1 stability/energy (`results/w1_pa_1d.json`) | Remis-class dual-cell discrete energy over 10^6 steps, 1-D lossless multi-band, r in {1.0 control, 1.1, 1.2, 1.4, 1.5, 2.0} abrupt + {1.4, 2.0} smooth | drift <= 2.94e-6 relative on every arm, bounded inside the pre-declared float-accumulation envelope (1.19e-3 at 10^6 steps) at every sample. The judge's growth-trend clause did NOT evaluate on any arm: it is gated at a declared trend floor of 50u = 2.980e-6 below which a slope is quantization noise, and the largest drift reached 2.931e-6. What is established is boundedness within the envelope, not the absence of a trend |
| F-S1 3-D (`results/w1_pb_full_gpu.json`) | Same functional, 3-D PEC box, 10^6 steps on GPU (VESSL run 369367256892) | end drift +4.7e-9 (r=1.4) and +1.3e-7 (r=2.0), max 7.7e-8 / 2.0e-7, bounded inside the same envelope; the trend clause did not evaluate here either (both maxima are >1 order below the 2.980e-6 trend floor) |
| F-S2 per-transition reflection (`results/w2_w3.json`) | Gated two-run differencing against an exact discrete scattering-chain model, 10 GHz, 30 cells/wavelength in the fine band | r=1.4 abrupt -53.9 dB measured vs -54.0 dB modelled (r=1.1 -67.1 dB, r=1.2 -60.7 dB); every in-envelope arm inside its pre-declared window. **These dB figures are resolution-specific — quote the law, not the number.** Per-transition reflection scales as (dz/lambda)^2, i.e. -12.0 dB per doubling of fine-band resolution; the same chain model gives, for r=1.4 at 15 / 20 / 30 / 60 / 120 fine cells per free-space wavelength, -41.7 / -46.8 / -54.0 / -66.1 / -78.2 dB (the N^-2 law is asymptotic and under-predicts the reflection below ~30 cells/wavelength) |
| F-S3 round-trip amplitude (`results/w2_w3.json`) | Net amplitude drift across a symmetric rise-and-fall traversal (the Christ amplification/attenuation asymmetry) | \|T\|-deviation <= 7.6e-6 for r <= 1.4, against a 3e-4 floor; r=1.0 null control 2.4e-6 |
| F-S4 convergence order (`results/w4r3_zdominant_cavity.json`) | Resonance error vs the ANALYTIC TE_{1,0,4} of an empty PEC cavity (60 x 3 x 64 mm), multi-band z profile at the cap ratio r=1.4 vs a uniform-fine control, four scales, on a fixture DESIGNED so the graded axis carries the error budget: an exact discrete-dispersion decomposition (`analytic_dispersion.py`, certified to reproduce every arm to 0.033 MHz) puts 89 % (uniform) to 92 % (multi-band) of the modelled error on the z axis, with 36 % of the multi-band total in the grading-specific term | p_multiband = 2.01, p_uniform = 2.02 — 2nd-order supraconvergence preserved at the cap ratio (Monk & Suli 1994; Li & Shields 2016). The multi-band mesh's extra error at matched scale is measured, not inferred: 28.4 MHz (2.9e-3 relative) at the coarsest scale falling to 0.44 MHz (4.6e-5) at the finest — about 1.56x the uniform-fine error amplitude at equal fine cell size, at the same ORDER |
| F-S5 differentiability (`results/w5_ad.json`) | `jax.grad` of a multi-band profile observable vs central FD, dominant cells. **The only witness here that is not PEC-closed:** its grid is built with `cpml_layers = 4` on all six faces, and its profile's boundary runway is non-uniform, so this fixture draws the row's own `nu_grading_reaches_absorber` advisory. It is a gradient-consistency check, not an accuracy measurement — see the absorber exclusion | worst dominant-cell error 1.1e-4 relative (f32 path) and 1.7e-4 (x64 context), inside the existing NU AD convention (15 %) |
| Design-variable autodiff (`validation/research/multiband_nu/results/adq_*.json`; pre-declaration `docs/design_notes/20260913_nu_ad_designvar_predeclaration.md`) | `jax.grad` along PHYSICAL design variables — layer thickness at fixed total length and per-layer permittivity, through a fixed-topology stackup map (E4, PR #962) — judged by Taylor-remainder ORDER (AD-Q: R1 in [1.8, 2.2]; Lane 2b: R0 in [0.9, 1.1] and R1 >= 1.8, without an upper R1 cutoff) and by central FD inside three times the FD's own estimated error, not by a flat percentage. Revert-proof: the same test re-run with the CFL time step cut out of the gradient (`stop_gradient(dt)`) | Transient 120-step loss: h_core_left, **h_thin**, h_core_right, eps_core_right verified; h_thin R1 2.047 and FD inside a **2.8e-4** band; together they cover **88.3 %** of the cell-space gradient norm. **The dt path is verified, not assumed**: with it removed the h_thin test FIRES (R1 0.947); its share of the thickness gradient is 18.5 %. Resonant 8000-step DFT-power loss: eps_core_left/right and eps_air verified; **thickness NOT verified on that observable** — FD fires, the observable is rippled at ~1 % in thickness (central FD within -1.3..+4.5 % of AD for steps <= 0.4 %). Lane 2b (`results/e7_zsmooth.json`; `docs/design_notes/20260915_nu_lane2b_predeclaration.md`): the same stack and 8000 steps with a Hann-windowed comb spectral loss — **h_thin verified** (R1 1.981, FD inside a 7.8e-4 band, floor 3.6 ulp; the dt revert-proof fires, share -1.045), h_core_left / h_air verified; h_core_right FIRED on the one-sided R1 (1.721) along the exact negative of h_core_left's direction (HELD, 2.188) — diagnosed from the ladders as the statistic's cubic asymmetry (even remainder 1.99-2.01 on all four controls, central remainder h^3, gradient error along the line < 6.8e-4), recorded as FIRED. In-plane (E6 + Lane 2b `e7_y.json` / `e7_pos.json`): band **width** verified on x and on y (R1 1.94-1.96, dt share 0.59 / 0.77, revert fires); band **position** verified on x with the probe on the source side (R1 2.233 / 1.989, FD 6.7e-4 / 7.5e-3); with the probe across the strip (E6, y) position stays INCONCLUSIVE on order (lead/tail cancellation) and FD-supported. Per-cell gradients on tied-minimum cells are an equal-split subgradient (`g_ad = FD+ + (FD- - FD+)/n_tied`), not a derivative; F-S5's per-cell 15 % figure above is a smoke check, not this claim |
| Revert-proof (`results/revert_proof.json`) | Two deliberate defect injections (one witness weight, one solver transition coefficient) | baseline f64 drift 2.1e-16; corrupted witness 5.8e-3, corrupted solver transition coefficient 7.9e-3 — the witness does fire on the defect family it guards. **Scope limit:** the witness builds its weights from the same `grid.inv_*` arrays the solver steps with (`remis_energy.energy_weights`), so a defect INSIDE those arrays — a wrong dual spacing computed by `make_nonuniform_grid` itself — would corrupt witness and solver identically and stay invisible here. It guards the update path against a correct metric, not the metric |
| W6 narrow fine band (`results/w6_band_builder.json`; pre-declaration `docs/design_notes/20260907_nu_band_profile_predeclaration.md`) | The F-S2 method (2-run differencing, geometric time gating, 10 GHz, PEC-closed), incident from the COARSE side (1.96 mm cells, 15.3 cells/wavelength) onto a fine band of n_b = 2 / 4 / 8 / 16 / 32 / 64 cells of 1.0 mm (30 cells/wavelength) between two ratio-1.4 ramps (1.96 -> 1.4 -> 1.0 and back); every profile built by `make_band_profile` and asserted cell-for-cell against its declaration | gate row n_b = 4: measured 1.0063e-2 (-39.9 dB) vs 1.0141e-2 from the exact discrete chain model, inside the pre-declared window [8.0826e-3, 1.2199e-2]; the law rows n_b = 2 / 8 / 16 / 32 / 64 measured 7.4364e-3 / 1.1164e-2 / 1.2559e-3 / 1.4248e-3 / 6.4596e-3 vs modelled 7.4916e-3 / 1.1296e-2 / 1.2101e-3 / 1.5111e-3 / 6.5575e-3, every row inside its window. A single 1.96 -> 1.4 -> 1.0 ramp reflects 5.79e-3 (-44.7 dB) in the exact discrete chain model (a modelled value — W6 did not measure the single ramp on its own); a band of width L between two such ramps reflects about 2 x 5.79e-3 x abs(sin(k_g L_eff)) with L_eff = L + 1.87 mm (the two ramps' phase length, fitted on the chain model: the six W6 rows within 0.03-0.53 % (the two null rows 0.53 % and 0.42 %); with L alone the null rows are off by 2-3.5x) and the discrete k_g(1.0 mm) = 0.1816 /mm (lambda_g = 34.59 mm) — the chain model peaks at 7 / 24 / 41 / 59 / 76 cells (maximum 1.158e-2 = 2 x 5.79e-3 x 0.9999) and nulls at 15 / 33 / 50 / 67; 8 cells sits near the first maximum and 16 near the first null. Narrow is not worse: bounded by twice the single-transition value, oscillating with width. One resolution pair only — quote the law, not the dB |
| Order-witness revert-proof (`results/w4r3_revert_proof.json`) | One transition coefficient of the multi-band arm deliberately corrupted (E-update dual metric replaced by the primal cell width at a single coarse->fine node — an error that is identically zero on a uniform mesh) | the resonance moves -47.3 / -24.0 / -12.0 / -6.0 MHz at the four scales (20-160x the fit floor) and the fitted order drops to 1.38, so the committed F-S4 judge FIRES. The order gate above can fail for a grading reason |

**Exclusions — these are NOT covered by the row.**

- **Grading must not reach the absorber.** Every witness above that
  measures an accuracy observable — F-S1 (1-D and 3-D), F-S2, F-S3, F-S4
  and both revert-proofs — is PEC-closed with `cpml_layers = 0`; each
  builds its grid through `harness.build_pec_fixture`. **F-S5 is the
  exception and does NOT run PEC-closed:** `w5_ad_consistency.py` builds
  its grid with `cpml_layers = 4`, i.e. a 4-layer CPML on all six faces,
  so one witness in this row does grade z with an absorber present. Its
  own profile's boundary runway is not uniform either (the anti-tie
  jitter, ~1 % and ~1.5 % on the two z faces), so that fixture would
  itself draw the `nu_grading_reaches_absorber` advisory described here.
  What F-S5 measures is `jax.grad` against central FD on the profile
  vector; it compares no field, reflection, energy or resonance against
  any reference, so what it establishes is that a graded mesh beside an
  absorber constructs, runs and differentiates consistently — not that
  anything computed on it is accurate. **The exclusion therefore rests on
  the narrower true statement: no witness in this row measures an
  accuracy observable with an absorber present.** Nothing here says how a
  grading transition interacts with a CPML/UPML face, and the absorber pad
  replicates the outermost interior cell, so a transition in that runway
  changes the discrete medium in the boundary-normal direction exactly
  where the absorber begins — the documented PML breakdown class.
  Preflight emits `nu_grading_reaches_absorber` when an axis grades within
  the absorbing face's own layer count. Keep that runway uniform, or close
  the face with PEC/PMC.
- **`dt` is unchanged: still the global min-cell CFL** (0.99 of it,
  `rfx.nonuniform.make_nonuniform_grid`). Adding fine bands buys cell
  count, not time steps — the cost is the finest cell everywhere in time.
  Recovering that is not in this row and is not implemented; domain-wise
  `dt` and local time stepping are explicitly not pursued (late-time
  interface instability, Xiao et al. TAP 55(7):1981, 2007).
- **Ratios above 1.4** are advisory-flagged, not validated: r=1.5 measured
  -51.6 dB and r=2.0 -43.9 dB per transition at the F-S2 fixture's own
  30 cells/wavelength (the same resolution caveat and (dz/lambda)^2 law as
  the F-S2 row above), both consistent with the chain model but outside the
  claimed envelope.
- **In-plane grading is COVERED for three witness classes, and only
  those.** None of F-S1..F-S5 grades `dx_profile` or `dy_profile` (that
  harness takes a scalar transverse cell). What does exist, each with its
  own pre-declaration and replay test:
  - *Accuracy (W7 A2/A3, `docs/design_notes/20260907_nu_band_accuracy_ad_predeclaration.md`):*
    two fine bands on x AND y (4:1, caps 1.3 and 1.4) on a TM110 air cavity,
    |err| <= 0.028 %, graded-minus-uniform <= 0.016 pt; all three axes graded
    at once (TM111) err -0.035 %. PEC-closed, vacuum.
  - *Per-transition reflection (E5, `docs/design_notes/20260913_nu_lane1_multilevel_xyz_predeclaration.md`):*
    the W6/E1 two-run method run on x and y as well as z — 40 arms per axis,
    every one inside |R - R_model| <= 0.2 R_model + 3e-5, including unequal
    coarse sides (1.96 / 2.744 mm around a 1.0 mm band), a bare cap step on
    one side, and two fine sizes (1.0 and 0.5 mm) in one column. The law
    generalizes to two amplitudes: R = sqrt(R_L^2 + R_R^2 - 2 R_L R_R cos(2 k_g (n_b d + c))),
    bounded by R_L + R_R (87/87 band arms). On the unpinned relabel control,
    x and z traces are bit-identical; y trace differences are below 7e-7.
    The y reflection difference is 6.02e-6 relative, so its W5 1e-6
    relabel criterion FIRED (`e5_relabel.json`); W1-W3 still hold.
    Single numbers carry a ~3 % rectangular-DFT
    window systematic; quote the law and the bound, not a sub-3 % figure.
  - *Design-variable autodiff (E6, `docs/design_notes/20260913_nu_lane2_inplane_designvar_ad_predeclaration.md`):*
    d(loss)/d(band width) on x, Taylor order R1 1.945 (transient) / 1.962
    (smooth spectral), inside central FD's own error bar, CFL-dt path
    included (59 % of the gradient; the judges fail when it is removed).
    Lane 2b (`20260915_nu_lane2b_predeclaration.md`): the same on y (R1
    1.943 / 1.962, dt share 0.59 / 0.77); band position order-verified on
    x with the probe on the source side (R1 2.233 / 1.989), FD-supported
    only with the probe across the strip (E6, y).
  What is still NOT covered in-plane: profiles whose two end cells differ
  (the grid refuses them — `dx_profile[0] == dx_profile[-1] == dx`, and
  `dy_profile` ends must match). The pinned E5 runs use 1.0 mm end cells,
  E6 and E7 use 0.5 mm, and the E5 relabel control is unpinned with matching ends.
  Also not covered: any absorber on an in-plane-graded face; the
  energy / round-trip / order witnesses (F-S1, F-S3, F-S4) on x or y;
  ratios other than 1.3/1.4 in-plane; dielectric interiors in the
  reflection witness; ports, TFSF and waveguide observables on an
  in-plane-graded mesh. `make_band_profile(..., max_ratio=1.3,
  boundary_cell=dx)` constructs and preflights without the ratio advisory
  (`test_band_profile_builder.py`, F5) but draws `nu_grading_reaches_absorber`
  on every in-plane face: the pinned cell is one cell, and the plateau
  beside it (0.9606 mm next to a 1.0 mm pin on the F5 fixture, deviation
  0.039-0.041) is not the `cpml_layers`-deep uniform runway the absorber
  exclusion asks for.

**Honest scope — what the witnesses do and do NOT establish.** The original
battery's conservation, round-trip and global-order statements apply to
**grading along z**; in-plane coverage is limited to the three classes above.
On the z battery:
the scheme conserves its discrete energy, each transition reflects at the
modelled level, a symmetric traversal does not drift in amplitude, the
global order stays 2 at ~1.56x the uniform-fine error amplitude, and the
gradient path is intact. They are NOT a statement that any S-parameter,
flux, far-field, or port result computed on a multi-band mesh is accurate —
each of those keeps the status its own row in the table above gives it.

In particular the F-S4 order result was obtained on an EMPTY PEC cavity
against an analytic eigenfrequency, precisely because the theorem it tests
assumes smooth fields. Its fixture is deliberately proportioned so the
graded axis dominates the error budget (89-92 % of it), which is what makes
the order gate capable of failing for a grading reason — the earlier
`w4r2_analytic_cavity` fixture put only ~1 % of its error on that axis and
is retained in the design note as a recorded diagnostic, not as evidence
for this row. On a rasterized dielectric-loaded microstrip-class fixture
the same ladder was inconclusive. What THIS lane measured there is one
thing only: the ~20 MHz (~4e-3) spread was present identically in
uniform-mesh arms, so it is not a grading effect. The reading of that
spread in an earlier draft of this row (a "floor", and a
"geometry-realization limit") is WITHDRAWN — it was an interpretation this
lane never measured, and it is withdrawn on that ground alone.

Evidence taken later on the same fixture reports its `f(h)` as **non-monotone**,
with a turn at `dz_fine = 0.125 mm`, so this ladder's anchor lies past the turn and
the fitted quantity was never an error sequence; the same work found geometry
quantization and port loading not responsible (records in
`validation/research/convergence_floor/results/`). Whether that fixture's `f(h)`
converges to the physical answer is not settled. Convergence on your own geometry
still has to be demonstrated on your own geometry; `ladder_guard`
(`validation/research/convergence_floor/ladder_guard.py`, tested by
`tests/unit/misc/test_ladder_guard.py`) is there to catch a ladder read across a turn.

## Interpreting output and warnings

- A preflight pass checks compatibility; it is not a convergence study or an
  external-solver comparison.
- A warning or reliability mask identifies a result that should be excluded or
  investigated. Its absence is not an accuracy guarantee.
- Touchstone, HDF5, CSV, JSON, and plotting helpers preserve or display a result;
  they do not change its support status.
- Reports should state the API, source or port family, mesh type, frequency and
  geometry range, normalization, git SHA, and the cited comparison metric.
- Before using a result outside a listed range, repeat mesh/time convergence and
  compare against an analytic or independent solver reference suitable for that
  configuration.
