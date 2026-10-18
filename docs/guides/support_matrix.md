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
- **diagnostic** — reported, not judged (a check, or a fitted quantity that does
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
include point/current sources and the port families listed under
[S-parameters](#s-parameters).
Documented observables include time-series probes, flux monitors, resonances from the matrix pencil fit, benchmarked NTFF cases, and the explicitly listed port results.

| Feature | Status | Current limits |
|---|---|---|
| Uniform Cartesian Yee workflows | **supported** | Primary documented grid; each source and observable still has its own restrictions. |
| Uniform precision modes | **limited by observable** | `precision="float32"` is the default. `"float64"` is uniform-single-device only and is **refused** unless JAX 64-bit mode is on in the process. `"mixed"` is also uniform-single-device only; with CPML its float16 field storage raises the absorber residual floor, so it is not a substitute for float32 near low-reflection or S-parameter floors. |
| Rectangular-waveguide S-matrix | **limited** | Use `compute_waveguide_s_matrix(...)`. Uniform single-mode magnitude results are covered by the documented WR-band tests; phase, junction and nonuniform results have narrower evidence. A full-aperture port needs PEC on its transverse faces: on a uniform mesh an omitted boundary is solved as PEC with a warning, and a declared absorbing or other non-PEC transverse face is **refused**; on a graded mesh an omitted boundary is **refused** (declare e.g. `boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}` for an x-directed guide). A waveguide port with any PMC face, `solver="adi"`, and `eps_override` together with `subpixel_smoothing` are **refused**. Uniform `precision="float64"` (with JAX x64 enabled) and `precision="mixed"` honor the requested field storage, independently of accumulator and result dtype; nonuniform meshes still require `precision="float32"`. Differentiation through `eps_override` / `sigma_override` runs on `normalize=False` and `normalize="flux"` only; `normalize=True` is host-assembled and raises `NotImplementedError` for a traced override. Multimode (`n_modes>1`) is host-assembled and outside the differentiable chain. S is a modal voltage-wave ratio referenced to each port's own discrete modal impedance, so it is a power-wave S only when all ports share one cross-section. The `normalize="flux"` result dtype follows `JAX_ENABLE_X64` (complex64 by default, complex128 under x64), as `normalize=False` does. **validated** — the WR-90 chain battery (thru, PEC short, `eps_r = 4` slab against the analytic Airy result) is gated by `tests/oracle/test_waveguide_chain_battery_v18_close.py` against the live fixtures `tests/fixtures/waveguide_chain_battery/fixture_931_realized_pec_forward2_run369367259427.json` and `fixture_1012_cpml_half_cell_run369367264100.json`; read the numbers there. The older fixtures in that directory are kept for history and are not current evidence (`tests/fixtures/waveguide_chain_battery/README.md`). Fences: junctions, multimode and nonuniform meshes are excluded. |
| Microstrip-line S-matrix | **limited** | Use `compute_msl_s_matrix(...)` with the laplace/quasi-TEM model, within its own documented validated range. Raw extracted S is the default (`enforce_passivity=False`) and is the value differentiated. Passivity projection is opt-in on the concrete measurement channel and is skipped for `eps_override`. Automatic probe offsets and spacing are recounted using the feed-line cells on the grid; a ladder crossing a grading ramp retains the old counts with a warning, and the routing checks retain their ramp refusals. `add_msl_port` accepts `+x`, `-x`, `+y`, and `-y`; the x-only eigenmode path remains outside the public support range. A signal strip drawn behind a line port is an open stub; it is **refused** when an odd quarter-wave resonance of its effective length lies within a factor of 1.5 of the band read, and only reported by preflight otherwise. S uses positive real per-port power-wave references, exposed as `reference_impedances` separately from fitted Z0 and load resistance. See the [MSL power and replay contract](msl_power_contract.md). The external notch comparison is characterized rather than a tight cross-solver match. The extractor also reports a fitted `beta`; it is **diagnostic** and does not enter S (`tests/unit/sparams/test_msl_fitted_beta_does_not_enter_s.py`). **validated** — the fit recovers a known `beta` to 0.05 % on synthetic data that is exactly the fitted model (`tests/unit/sparams/test_msl_beta_fit_accuracy_envelope.py`). The fitted `beta` of a solved line differs from the Hammerstad–Jensen closed form by an amount that is not attributed (see [known limitations](known_limitations.md)). Conductor thickness is not solved: a metal strip is a zero-thickness sheet (tracker 1523). |
| Surface-impedance sheets and lossy films | **limited** | `add_thin_conductor(..., surface_impedance_f0=...)` supports patterned shapes that are sheets — placed on ONE node plane on the grid, no thicker than one local cell along their normal — on its documented uniform and graded-mesh paths. The f0 sheet loads the edge set of the PEC sheet on the same cells, with the edge rows on a drawn free edge at half weight; **validated** — a resistive strip whose drawn edges lie on grid nodes is solved at its drawn width by the f0 and the lossy-film model alike (`tests/unit/materials/test_sheet_footprint_width.py`). `add_thin_conductor` below the PEC threshold (`sigma_bulk` under 1e6 S/m, no `surface_impedance_f0`) is one resistive film on the tangential E edges of its declared node plane, on uniform, graded and multi-device runs. A non-Box lossy film must occupy one cell layer and realize its drawn area within 1 % in equivalent length, or it is **refused** unless `snap="declared"` is passed; `eps_r != 1` on a lossy film, a film on a PMC face or normal to a 2-D model's invariant axis, an f0 sheet sharing edges with a lossy film, and a film edge inside a design box are **refused**. Neither an f0 sheet nor a Box lossy film is judged by `sheet_effective_size` (see [known limitations](known_limitations.md)). It is not a general dispersive or multi-runner conductor model. |
| Lumped and wire-port S-parameters | **limited** | Use the calculation API and evidence limits for the selected port family. Do not interpret these discrete feed models as calibrated transmission-line modes. Driven lumped ports use the post-injection terminal V/I pair and the Yee half-step current phase, with the wire-family decomposition for diagonal and off-diagonal entries; every existing lumped S result changes. This extraction correction does not calibrate the feed reference plane. `add_port(..., extent=h, radius=a)` declares a physical probe-pin radius up to 0.2 times the transverse cell (`tests/unit/ports/test_wire_port_radius.py::test_radius_requires_extent_and_resolved_pin_above_bound`); a larger radius, axial grading across the port, and paths other than single-device nondispersive 3-D Yee `run()`/`forward()` are **refused**; resolve a larger pin with a coax feed or a volume wire. Omitting `radius` retains the prior probe model. Positive subcell PEC `PolylineWire` radii below half a cell are **refused**; `radius=0` selects the legacy filament model, not an arbitrary physical radius. A lumped/wire S-matrix request on a model that also has a plain source or a 0-ohm port is **refused**. Pending for 2.0 — tracker 1373 (one port V/I measurement layer on every path). |
| Coaxial-line reflection | **limited** | `compute_coaxial_line_reflection(...)` requires float32 precision, a three-dimensional second-order uniform Yee grid, CPML tokens with positive thickness on every boundary face, `cpml_axes="xyz"` (the only accepted value), positive lateral absorber clearance, no periodic axes, and exactly one `face="top"` coaxial port. It is documented over the stated frequency, termination, impedance, and mesh ranges, not as a general coaxial-network solver. The coax drive launches the TEM mode of the cross-section on the grid. The line ends at its declared extent without the former open stub behind the termination; earlier loss and termination readings are not current accuracy evidence. |
| Coax-to-microstrip transition | **experimental** | `compute_coax_msl_transition(...)` is a diagnostic cross-family calculation, not a validated transition. Before stepping it runs the sheet-size verdict (unless `skip_preflight=True`) and the line-stub admission; the other preflight families are not run on this path (`rfx/sparams/coax.py::compute_coax_msl_transition`). The returned matrix carries more power out than in, so it is not passive; the call refuses it by default (`strict_passivity=True`, raising `ValueError`), and `strict_passivity=False` returns the diagnostic matrix with a `UserWarning` instead. No magnitude from this path is usable as an absolute power reference, and the calculator is not scheduled for correction before 2.0. For a result you can cite, use `compute_msl_s_matrix(...)` and the coaxial calculators separately, each within its own documented validated range. Listed in [known limitations](known_limitations.md). |
| Mixed lumped/wire + microstrip S-parameters | **experimental** | `compute_mixed_s_matrix(...)` is diagnostic and not in the validated set. Off-diagonal magnitudes come from Poynting flux and absolute magnitude has no external-solver comparison; off-diagonal phase mixes two reference-plane conventions. With `enforce_passivity=True` (the default) the returned diagonal is a jointly SVD-projected value, not the raw per-family measurement — read `S_raw` / `passivity_correction` for what was measured. Like the transition row above, this cross-family calculator is not pursued further before 2.0. |
| Nonuniform meshes | **limited by feature** | See the table below. Support for one observable does not imply support for another. |
| SBP-SAT/subgridding | **not documented** | No current public workflow. `add_refinement(...)` on a uniform mesh selects the subgridded path, which is **refused** by default (`validation="production"`); `validation="research"` runs it with an `ExperimentalSubgridWarning`. Use a graded mesh (`dx_profile` / `dy_profile` / `dz_profile`) for local resolution. |
| ADI | **experimental** | The Zheng–Chen–Zhang ADI path (`solver="adi"`) is outside the 2.0 supported scope and warns (`ExperimentalADIWarning`) on every run. It runs a closed (`boundary="pec"`), homogeneously filled cavity. Absorbing boundaries, dielectric or conductivity interfaces, graded meshes, interior PEC and ports are **refused**. No accuracy and no stiff-mesh speedup is claimed; use explicit Yee for physics results you report. |
| Distributed execution | **not documented** | Pending for 2.0 — tracker 1402. Execution scaling is separate from numerical correctness. The available runner is `shard_map`-based `distributed_v2`; the retired `pmap` runner raises. Distributed E permittivity and conductivity use the same four-cell edge mean as the single-device path. Uniform `run(devices=...)` returns the S-matrix of excited single-cell lumped ports and of wire ports (`tests/unit/runners/test_distributed_wire_ports.py::test_wire_parity`); wire ports with reference planes or a radius, passive ports, lumped RLC elements, Kerr materials, MSL ports, magnetic (PMC) faces and graded-mesh ports are **refused**. Uniform distributed runs return NTFF records for far-field and gain calculations; NTFF on a graded mesh and on `forward(distributed=True)`, flux monitors and DFT plane probes are **refused**. `exchange_interval` other than 1, and an enclosing `jax.jit` on GPUs, are **refused**. These implementations do not promote the release support status. |
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

Current-moment far fields have a range the tests cover on thin planar boards with
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
- **n/a**: no model with the input reaches the path, or the path returns nothing
  the input would produce (a `forward()` run returns no flux);
- **unclassified**: not yet classified (no row carries this today).

A cell can name a part of its column in parentheses: `(run)` or `(forward)` for
the entry point, `(uniform)` or `(graded)` for the mesh, and in the Calculators
column the calculators that refuse. A `*` after a word marks a cell whose
behaviour today contradicts it; the row names the open issue. The Row key is the
row of `tests/contracts/path_disposition.py` the cell words are taken from.

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
| A plane wave (`add_tfsf_source`) | `_tfsf/plane_wave` | supported | supported (run, forward) | falls back (uniform); refused (graded) | refused (mixed_s_matrix, waveguide_s_matrix, material_fit) | Default pulse: DG for normal incidence, closed boxes and method B; MG for oblique Bloch, which refuses explicit DG. Method B honours pulse and bandwidth; every modulated Gaussian peaks at 5τ. An open plane wave refuses finite transverse structures on both meshes. On a graded mesh use one device; finite scatterers (`closed_box=True`) and arrays (explicit periodic boundaries) require the uniform mesh. A uniform multi-device `run()` hands the model to one device with a warning. |

#### Boundaries

| Input | Row key | Uniform | Graded | Multi-device | Calculators | What to do instead |
|---|---|---|---|---|---|---|
| `boundary='upml'` | `_boundary/upml` | supported | refused | refused | refused (waveguide_s_matrix) | Use `boundary='cpml'`. |
| A magnetic (PMC) face | `_boundary_spec/pmc_face` | supported | supported | refused | refused (waveguide_s_matrix, vmap_sweep_batched) | Run it on one device. |
| Conformal PEC (`Boundary(conformal=True)` or `conformal_pec=True`) | `_boundary_spec/conformal` | supported (run); refused (forward) | refused | refused | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, vmap_sweep_batched, material_fit) | It is solved by a single-device uniform `run()` only. |
| Conformal PEC together with a lumped or wire-port S-matrix request | `_boundary_spec/conformal_s_matrix` | refused | refused | refused | refused (s_matrix_scan, mixed_s_matrix, topology_optimize, waveguide_s_matrix, vmap_sweep_batched, material_fit) | Request the S-matrix without conformal PEC. |
| A periodic axis | `_periodic_axes/periodic` | supported | supported (constant cell size along the axis; refused with a graded profile along it, a plane wave or `interface_eps='dual_average'`) | refused | refused (waveguide_s_matrix) | Run it on one device; on a graded mesh keep the periodic axis's cell size constant. |
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
| `add_refinement(...)` (subgridded run) with the default `validation="production"` | `admission:subgrid-default` | `run()`, `forward()`, S-matrix calculators | Use a graded mesh (`dx`/`dy`/`dz` profiles) for local resolution. Only `run()` on a uniform mesh runs it, with a warning, when you pass `validation="research"`; `forward()` and the calculators refuse a refinement whatever `validation` says. | `tests/contracts/test_subgrid_requires_experimental.py::test_default_refused_before_runner` |
| A full-aperture waveguide port with a declared non-PEC transverse face, uniform mesh | `admission:waveguide-faces-uniform` | `run()`, `forward()`, `compute_waveguide_s_matrix`, also with `skip_preflight=True` | Declare the `boundary=` the message prints: PEC on the transverse faces, the absorber on the port axis. | `tests/contracts/test_waveguide_boundary_admission.py::test_explicit_absorber_is_refused_at_dispatch` |
| A full-aperture waveguide port on a graded mesh with `boundary` omitted | `admission:waveguide-faces-graded` | `run()`, `forward()`, multi-device runs, `compute_waveguide_s_matrix` | Declare the walls, e.g. `boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}` for an x-directed guide. | `tests/contracts/test_graded_guide_default.py::test_default_graded_refuses_before_field_step` |
| A waveguide port in a model with a PMC face | `admission:waveguide-pmc` | `run()`, `compute_waveguide_s_matrix` | Use a full PEC-walled guide (the aperture mode solver has PEC walls only). | `tests/unit/sparams/test_pmc_waveguide_refusal.py::test_pmc_waveguide_mode_solver_refuses` |
| `eps_override` together with `subpixel_smoothing`. Pending for 2.0 — tracker 1524. | `admission:waveguide-override-smoothing` | `compute_waveguide_s_matrix` | Declare the permittivity as a material instead of `eps_override`, or switch smoothing off. | `tests/unit/sparams/test_waveguide_override_with_smoothing_refused.py::test_eps_override_with_smoothing_is_refused_before_the_solve` |
| A TF/SF plane wave whose absorbing transverse faces see structure that varies along that axis, on the uniform or the graded mesh, including a finite block given to `forward()` as an override or a design box; a transverse wall pair that does not suit the polarization or mixes a PMC face with an absorber | `admission:plane-wave-faces` | `run()`, `forward()`, also with `skip_preflight=True` | Uniform mesh: declare the axis periodic to solve an array, or use `closed_box=True` for a finite scatterer. Graded mesh: run it on the uniform mesh. | `tests/contracts/test_tfsf_boundary_admission.py::test_finite_box_is_refused`, `::test_inadmissible_electric_wall_is_refused`, `tests/contracts/test_plane_wave_admission_both_meshes.py::test_finite_cube_refused`, `::test_finite_forward_override_refused` |
| A soft source on an edge the grid realizes as PEC (inside a PEC volume, on a sheet or wire, or an H source whose whole curl loop is PEC) | `admission:source-on-pec` | uniform, graded and multi-device runs, also with `skip_preflight=True` | Move the source off the conductor, or use a port. | `tests/contracts/path_equivalence/test_source_on_pec.py::test_dead_e_source_refused` |
| A soft source or one-cell lumped port whose E edge lies tangential in a declared PEC wall face; a wire port lying entirely in such a wall | `admission:source-on-pec-wall` | every path | Move it at least one cell inside the domain, off the wall. | `tests/contracts/test_source_on_pec_wall.py::test_b3_soft_wall`, `::test_b3_port_wall` |
| A lumped/wire S-matrix request on a model that also has a plain source (`add_source`, `add_polarized_source`) or a 0-ohm port | `admission:plain-source-s-request` | `run(compute_s_params=...)`, `run(devices=...)`, `compute_lumped_wire_s_matrix_via_scan`, `forward(port_s11_freqs=...)` | Remove the plain source, or call `run(compute_s_params=False)` to get raw port waves with the source included. | `tests/unit/sparams/test_plain_source_refusal.py::test_plain_source_s_request_refused` |
| A PEC sheet solved more than 1 % off its drawing (preflight error `sheet_effective_size`) | `admission:sheet-size` | `run()`, `forward()`, `compute_coax_msl_transition` | Build the in-plane profiles with `rfx.mesh_edges.edge_aware_profiles`, or pass `Simulation(..., snap="declared")` to accept the difference as a warning. | `tests/contracts/test_sheet_size_verdict.py::test_strict_refuses_inset_and_on_node_sheets` |
| A non-Box lossy film that spans several cell layers, misses its drawn area by more than 1 % in equivalent length, or cannot state its area | `admission:film-area` | every mesh | Refine the in-plane mesh or draw a Box, or use `snap="declared"` to accept the declared geometry. | `tests/contracts/test_dc_film_admission.py::test_b4_small_disc_refusal_and_declared_finding` |
| A lossy film with `eps_r != 1`; on a PMC face; normal to a 2-D model's invariant axis; sharing edges with an f0 sheet; with an edge inside a design box | `admission:film-conflicts` | every path | The message names the film and the conflict. | `tests/contracts/test_film_one_plane.py::test_c4_permittivity`, `::test_c4_geometry_refusals`, `::test_c4_f0_overlap`, `::test_c4_occupancy_design_box_refuses_film` |
| `precision="float64"` while JAX 64-bit mode is off | `admission:float64-without-x64` | `run()`, `forward()` and gradient entries | Build and run the Simulation inside a scoped `with jax.enable_x64():` block (do not switch it on at module level). | `tests/contracts/test_precision_realized.py::test_off_refuses_before_allocation` |
| A microstrip or coax signal strip drawn behind its line port whose open-stub resonance lies within a factor of 1.5 of the band read | `admission:line-stub` | line ports, also with `skip_preflight=True` | Start the signal strip at the port's grid-node coordinate the message prints, so it covers the port node and nothing behind it. | `tests/contracts/test_line_stub_refusal_1512.py::test_in_band_and_inclusive_near_edges_refuse` |
| A non-passive coax→microstrip transition matrix (default `strict_passivity=True`) | `admission:transition-passivity` | `compute_coax_msl_transition` | Pass `strict_passivity=False` to get the diagnostic matrix back with a warning. | `tests/unit/sparams/test_coax_msl_transition.py::test_transition_default_raises_on_a_nonpassive_s_and_false_returns_it` |
| An excited drive whose pulse, sampled at the time step, has no value above 10⁻⁶ of its amplitude (a `GaussianPulse` or `ModulatedGaussian` source or port, a coaxial port, a TF/SF source, waveguide or Floquet port); usually `bandwidth` given in Hz instead of as a fraction of `f0` | `admission:undriven-pulse` | `preflight()` (error `undriven_pulse`), `run()`, `forward()`, the S-parameter calculators | Give `bandwidth` as a fraction, e.g. `GaussianPulse(f0=5e9, bandwidth=0.8)`. | `tests/unit/sources/test_source_dc_floor_guards.py::test_undriven_pulse_is_refused_for_every_drive_kind`, `::test_undriven_pulse_is_refused_on_absolute_hz_bandwidth` |
| `GaussianPulse(cutoff=...)` below 2.5 | `admission:pulse-cutoff` | `GaussianPulse(...)` | Keep the default `cutoff=3`, or any value from 2.5 up. | `tests/unit/sources/test_source_dc_floor_guards.py::test_gaussian_pulse_refuses_a_cutoff_below_2p5` |
| An S-parameter request from one record in which two or more lumped or wire ports are excited, ring-down requests included | `admission:several-drives-s` | graded `run()` and `forward()`, uniform `forward()` | Run once per driven port with the others passive (`excite=False`), or use the uniform `run(compute_s_params=True)`, which drives one port per run. | `tests/contracts/test_lumped_wire_s_isolated_drives.py::test_several_excited_ports_refused_before_step` |
| A far-field box (`add_ntff_box`) in a model with a PEC, PMC or zero-depth absorbing face (a symmetry plane, an infinite ground plane, a waveguide port's guide walls); `compute_rcs(boundary="periodic")` at normal incidence | `admission:ntff-mirror-face` | `run()`, `forward()`, `compute_rcs` | Draw a finite ground plane as geometry and keep every face absorbing. Periodic faces are not refused. | `tests/contracts/test_ntff_boundary_admission.py::test_ntff_boundary_table`, `::test_rcs_boundary_table` |
| A gradient with respect to a declared shape's size or position (a sphere radius, a box corner) | `admission:shape-gradient` | `run()`, `forward()` under `jax.grad` or `jax.jacfwd` | Differentiate through material overrides or a design region. `jit` and `vmap` over a sphere or cylinder dimension still run. | `tests/unit/runners/test_shape_gradient_refusal.py::test_a_declared_dimension_gradient_refuses` |
| `precision="float64"`, or `float32` under JAX 64-bit mode, on the Apple-silicon `mps` JAX backend | `admission:mps-float64` | every run, forward, sweep and S-parameter entry | Run `float32` with JAX 64-bit mode off there, or run `float64` on CPU or CUDA. | `tests/contracts/test_precision_realized.py::test_mps_backend_refuses_float64_before_allocation`, `::test_mps_backend_refuses_float32_under_x64` |
| A flux monitor with `dft_window="hann"` or `"tukey"` in a run stopped by `until_decay`; `decay_energy_consecutive` below 2 | `admission:decay-stop-inputs` | `run(until_decay=...)`, uniform and graded mesh | Keep the default `dft_window="rect"` or run a fixed `n_steps`; pass `decay_energy_consecutive` of 2 or more. | `tests/contracts/test_decay_stop_one_rule.py::test_windowed_flux_monitor_is_refused_with_until_decay`, `::test_energy_criterion_refuses_fewer_than_two` |

## S-parameters

This part answers three questions for each public primitive: does it define a
port impedance and a reference plane, which calculator computes its
S-parameters, and what evidence and restrictions apply to that calculation now.
It replaces the former `sparameter_support_matrix.md`. The machine-readable
companion is [`sparameter_support_matrix.json`](sparameter_support_matrix.json),
published as `/rfx/api/support/sparameter_support_matrix.json`; it still
carries each family's evidence record in full. Raw voltage/current replay is
described in [`sparameter_dump_replay.md`](sparameter_dump_replay.md).

### Which calculator

If you are not sure which calculator your model needs, ask the simulation
instead of guessing from the table:

```python
sim.s_matrix_lane()          # -> 'compute_waveguide_s_matrix'
res = sim.compute_s_matrix() # calls that calculator with your keyword arguments
```

`Simulation.compute_s_matrix(**kwargs)` reads the registered ports, picks one
calculator and forwards every keyword argument to it unchanged: you get that
calculator's defaults, checks, warnings and result type, exactly as if you had
called it directly. `Simulation.s_matrix_lane()` returns the same choice as a
string and runs no simulation, so you can check it on a half-built model.

| Registered ports | Calculator |
|---|---|
| waveguide ports only | `compute_waveguide_s_matrix(...)` |
| microstrip ports only | `compute_msl_s_matrix(...)` |
| one coaxial port only | **ambiguous**: pass `method="compute_coaxial_line_reflection"` or `method="compute_coaxial_two_port"` |
| one coaxial + one microstrip port | `compute_coax_msl_transition(...)` |
| lumped/wire `add_port(...)` + microstrip | `compute_mixed_s_matrix(...)` |
| lumped/wire only | `s_matrix_lane()` returns `"run(compute_s_params=True)"`; `compute_s_matrix()` raises and names it, because `run()` returns a `Result`, not an S-matrix result |
| anything else | raises, naming the registered families and the calculator each one has |

`compute_s_matrix` never chooses between the two coaxial calculators (their
registrations are identical; only you know which measurement you meant), and it
never calls `run()` for you, because `run()` has no default simulation time to
supply. `method=` replaced `lane=` in 2.0; `lane=` still works through 2.x with
a `DeprecationWarning`.

### Result convention

Full matrices are indexed `S[receiver_port, driven_port, frequency_index]` with
shape `(n_ports, n_ports, n_freqs)`; frequencies are in Hz with shape
`(n_freqs,)`. A `*_mag_abs_diff` metric is an absolute difference in linear
magnitude unless it says dB.

Passivity, reciprocity, replay and warning checks are necessary, but none of
them alone shows that a result is accurate. The automatic checks also differ by
calculator: the microstrip and waveguide calculators check the power in every
driven column; lumped and wire results are checked only for non-finite or too
large individual `|S|` values; `compute_coaxial_line_reflection(...)` has no
shared passivity check (`compute_coaxial_two_port(...)` has). An absent warning
therefore means different things for different port families.

### API summary

| Primitive | Calculation API | Result | Current status |
|---|---|---|---|
| Lumped `add_port(..., extent=None)` | `run(compute_s_params=True, s_param_freqs=...)` | `Result.s_params`, `Result.freqs` | **limited**: a one-cell feed with a scalar reference impedance, not a transmission-line mode; uniform mesh only |
| Lumped `add_port(..., extent=None)` | `forward(port_s11_freqs=...)` | `ForwardResult.s_params`, `.freqs` (S11 vectors; on a graded mesh the driven column) | **limited**: one device, uniform or graded mesh |
| Wire `add_port(..., extent=...)` | `run(compute_s_params=True, s_param_freqs=...)` | `Result.s_params`, `Result.freqs` | **limited**: a feed spread over `extent`; on a graded mesh experimental |
| Wire `add_port(..., extent=...)` | `forward(port_s11_freqs=...)` | `ForwardResult.s_params`, `.freqs` (S11 vectors; on a graded mesh the driven column) | **limited**: one device, uniform or graded mesh |
| `add_msl_port(...)` | `compute_msl_s_matrix(...)` | `MSLSMatrixResult.S`, `.freqs`, `.Z0`, `.beta`, `.port_names`, `.reliable`, `.probe_clearance`, `.beta_railed`, `.reference_impedances` | **limited**: quasi-TEM (`mode="laplace"`) on a uniform mesh; on a graded mesh experimental; `mode="eigenmode"` refused |
| `add_waveguide_port(...)` | `compute_waveguide_s_matrix(...)` | `WaveguideSMatrixResult.s_params`, `.freqs`, `.port_names`, `.port_directions`, `.reference_planes` | **limited**: magnitude of single-mode rectangular guides on a uniform mesh; phase and junctions narrower; on a graded mesh experimental outside the cases listed below |
| `add_waveguide_port(...)` | `run(...)` | `Result.waveguide_sparams[name]` | **limited**: per-port output, not the full matrix |
| `add_coaxial_port(...)` | `compute_coaxial_line_reflection(...)` | `CoaxialLineReflectionResult` | **limited**: one `face="top"` port on a line the calculator builds itself |
| `add_coaxial_port(...)` | `compute_coaxial_two_port(...)` | `CoaxialTwoPortResult` | **limited**, validated with scope: one through-line family, symmetric (TM0n) discontinuities only |
| `add_coaxial_port(...)` + `add_msl_port(...)` | `compute_coax_msl_transition(...)` | `CoaxMSLTransitionResult` | **experimental**, diagnostic only: the matrix is not passive (mechanism unresolved) and is refused by default |
| `add_port(...)` + `add_msl_port(...)` | `compute_mixed_s_matrix(...)` | `MixedSMatrixResult` | **experimental**, diagnostic: magnitudes not checked against another solver; the default passivity projection changes the diagonal |
| `add_floquet_port(...)` | no documented high-level S-parameter API | none | **experimental**: diagnostic helpers at normal incidence only |
| Sources, plane wave, probes, DFT planes, flux monitors | none | field, resonance or flux results | **not a port**: no impedance and no reference plane |

### Lumped port

A one-cell feed or load with a positive scalar reference impedance. It is not a
transmission-line mode. Circuit elements use `add_lumped_rlc(...)` and do not
define an S-parameter port.

- Checked against a closed form: a one-cell port terminating a parallel-plate
  line in a known resistance, at three loads
  (`tests/unit/ports/test_lumped_port_known_load_line.py`). It agrees at the
  low end of the band; the residual near the top of the band fits one effective
  reference impedance and is not characterized. The fixture cannot be refined
  (the port must bridge a one-cell gap), so its convergence is not shown.
- A two-port line with a shunt port at each end is checked against its
  closed form (`|S11|` within 0.05, `|S21|` within 0.01), and the lumped and
  wire calculators give the same complex matrix on the same cells (to 1e-6) (`tests/unit/ports/test_lumped_two_port_matched_line.py`).
- The extractor's algebra and its replay from stored voltages and currents are
  checked (`tests/unit/sparams/test_sparam.py`,
  `tests/unit/sparams/test_port_dump_replay.py`). Algebra and replay checks do
  not show that the port is calibrated.

Analytic extractor and V/I replay checks validate the extraction algebra and its
reproducibility; they do not establish a generally calibrated lumped-port
result. `run(compute_s_params=True)` with a lumped port needs a uniform mesh;
on a graded mesh it raises (use a wire port there).
`forward(port_s11_freqs=...)` runs on one device, uniform or graded mesh. A plane wave, waveguide ports or another port family cannot share this
calculation.

### Wire port

A feed spread along `extent`. Use `add_msl_port(...)` when the model is a
distributed microstrip line.

- The same shunt-port line closed form as the lumped port, the same matrix as
  the lumped port on one cell (to 1e-6), and the same replay checks (`tests/unit/sparams/test_twoport_wire_port.py`,
  `tests/unit/sparams/test_wire_port_sparams_forward.py`).
- Absolute calibration is limited by the feed model; a replay check is not a
  modal-port calibration.

The nonuniform wire calculation is experimental: regression and gradient
coverage on a graded mesh are not RF validation.

### Microstrip-line port

`compute_msl_s_matrix(...)` with the quasi-TEM (`mode="laplace"`) model. It
runs one simulation per driven port and extracts S from voltage and current
probes downstream of each feed.

- **What `S` is.** S is the raw extracted power-wave matrix, referenced to the
  analytic (Hammerstad–Jensen) impedance in `result.reference_impedances`. The
  same S is returned on every channel, including under `jax.grad`. A raw S may
  exceed `‖S‖₂ ≤ 1`: `sigma_max_excess` records the excess per frequency point
  and a warning names the worst one. A passive structure cannot scatter more
  power than it receives, so such points are a measurement artifact; read
  `settling_db` and `reliable` to see which. `enforce_passivity=True` returns
  the projected matrix and keeps the measured one in `S_raw`; it is not
  applied under `jax.grad` or on the `eps_override` channel, where S stays raw.
- **The fitted `Z0` and `beta` are diagnostics and do not enter S**
  (`tests/unit/sparams/test_msl_fitted_beta_does_not_enter_s.py`).
- **`reliable`** is a per-point fit-quality mask, `None` under JAX tracing.
  Because S is solved jointly over the drives, a `False` entry at any port
  spoils the whole slice `S[:, :, k]`: filter with `np.all(reliable, axis=0)`.
  A `True` entry is not an accuracy guarantee.
- **Probes near a reflector.** `probe_clearance` reports whether the probes are
  clear of the next discontinuity. When it reports `insufficient`, the fitted
  `Z0` and `beta` cannot be read (`beta_railed` flags the symptom; `reliable`
  does not), and nothing bounds the error of S. Lengthen the uniform feed or
  move the reference plane; raising `n_probe_offset` alone moves the probes
  toward the reflector. `preflight()` and
  `preflight_sparameters(calculator="msl")` report the condition before a run.
- **Thru line.** A uniform thru line is held to a mean `|S21|` in
  (0.90, 1.05) and a mean `Re(Z0)` in (40, 65) Ω
  (`tests/unit/sparams/test_msl_port_integration.py`).
- **Cross-validation.** The open-stub notch filter is compared with frozen data
  from another solver in `tests/crossval/msl_notch_filter/`.

Nonuniform `mode="laplace"` and `mode="uniform"` have internal regression
coverage only and no comparison with another solver on a graded mesh; treat
them as experimental. `mode="eigenmode"` is refused. Subgridding, ADI, a plane
wave and other port families are refused for this calculation. The trace must
be hard PEC (a `Box`, a zero-thickness `Box` or a PEC `add_thin_conductor`
sheet); a lossy (`surface_impedance_f0=`) sheet may be used for other metal on
the board. Draw the trace at the width it is solved at: preflight refuses a
mismatch, and a trace end fed by a port is solved where it is drawn.

### Rectangular-waveguide port

`compute_waveguide_s_matrix(...)` for a full matrix of at least two ports;
`run()` gives only a per-port diagnostic. Prefer `normalize="flux"`: it uses a
matched reference run per port (`2 * n_ports` simulations).

- **Magnitude, uniform single-mode guides.** Dielectric slabs in five bands
  (WR-340, WR-62, WR-28, WR-15, WR-10) against the closed form, and the WR-90
  empty guide, PEC short and slab against frozen data from another solver
  (`tests/fixtures/waveguide_broad_e5/`). The port validation battery and the
  two-port contract are the authoritative checks: column power with a
  reflecting obstacle below 1.02, a PEC short's `|S11|` at least 0.99
  (`tests/oracle/test_waveguide_port_validation_battery.py`,
  `tests/unit/sparams/test_waveguide_twoport_contract_v1.py`).
- **Phase and group delay** are checked on the same slab cases and on an empty
  WR-340 guide near cutoff
  (`tests/oracle/test_waveguide_group_delay_near_cutoff.py`). PEC shorts,
  junctions, graded meshes and multimode extraction have no phase evidence.
- **The WR-90 chain**: single-mode S on a uniform mesh on the `normalize=False`
  and `normalize="flux"` paths, three meshes, thru / PEC short / slab, with
  gradients (`tests/oracle/test_waveguide_chain_battery_v18_close.py`). On an
  empty guide the `normalize=False` path reads a column power slightly above 1
  that falls with each mesh halving; it is a stated accuracy of that path, not
  a defect.

The calculation remains experimental on a graded mesh. What is checked there
is a dielectric slab in WR-90 with `normalize="flux"` and a graded transverse
mesh, against the closed form over 8.2–12.4 GHz
(`tests/fixtures/waveguide_nu_broad_e5/`); there is no comparison with another
solver on a graded mesh in this repository.
Gradients there (`eps_override`, `sigma_override`) need `normalize="flux"`.

Setup:

- Size the absorber from the guide wavelength at the lowest frequency, where
  it is longest; the calculator warns below half a guide wavelength, and that
  is a minimum, not a target.
- Draw interior metal (irises, septa, posts) at the size you mean and check the
  realized planes with `realized_wall_planes()` or `fidelity_report()`; a
  symmetric opening must be representable on the grid
  (`tests/oracle/test_wr90_inductive_iris_mode_matching.py`).
- Keep slab lengths an integer number of cells and frequency points away from
  Airy reflection nulls.
- Multimode `normalize=True` is refused. Branches, T-junctions and septa need a
  matched straight-guide reference per port and far-port placement; compact
  arbitrary junctions are not covered.

### Coaxial port

Use `compute_coaxial_line_reflection(...)` for the documented TEM-line result.
The model must contain exactly one `face="top"` coaxial port and no other port family, on a float32,
three-dimensional, second-order Yee uniform mesh with CPML of positive
thickness on every face (`cpml_axes="xyz"`) and no periodic axis. The
calculator builds the line, source, DFT planes and termination; do not
register separate geometry, RLC elements, monitors or termination helpers.
The registered port's z position, `pin_length` and `impedance` do not set the
line the calculator builds. Use `feed_impedance=` for the feed and `dut_impedance=` only with
`termination="matched"`. `probe_count` must be at least three and every plane
must fit between the load and the source.

- A short and an open are held to `|Γ|` within 0.05 of 1 on the current code
  (`tests/unit/sparams/test_coaxial_line_reflection.py`).
- Recorded fixtures compare it with the closed form over 4–12 GHz for short,
  open, matched and resistive loads at two line impedances, and with frozen
  data from another solver for short and open (`tests/fixtures/coax_broad_e5/`,
  `tests/fixtures/coax_broad_e4/`). These were taken before 2.0 changed the
  absorber to all three axes and were not taken again.
- Use about four or more cells across the annulus; below 3.5 the result is
  reported as under-resolved.
- Gradients through `eps_scale` are checked against finite differences
  (`tests/unit/autodiff/test_coax_end_to_end_ad.py`).

`compute_coaxial_two_port(...)` drives each end of one through line in turn and
solves for the full matrix. It is validated with scope: every DUT it can
currently gate against (no load, a matched feed, a coaxial dielectric plug) is
symmetric about the axis and excites only TM0n modes, and the evidence covers
one line geometry family
(`tests/unit/sparams/test_coax_two_port_smatrix.py`,
`tests/unit/autodiff/test_coax_two_port_ad.py`). On a coarse annulus the line
attenuates in the simulation, so `|S21|` falls with frequency even while `|S11|`
stays small; the four-cell rule above applies to transmission too.

`run()` and `forward()` refuse coaxial S-parameter requests. The former
`compute_coaxial_s_matrix(...)` is removed.

### Floquet/Bloch and non-port observables

`add_floquet_port(...)` has broadside modal bookkeeping, field-dump replay, and
a test that recovers a planted reflection and transmission from synthetic waves
(`tests/unit/ports/test_floquet_s_params_contract.py`); the comparisons with the
closed form for empty space and a homogeneous slab are diagnostic scripts
(`scripts/diagnostics/build_floquet_*_analytic_comparison.py`). These are not a
comparison with RCWA or another full-wave solver, and there is no documented
high-level Floquet S-parameter API. Treat the result as experimental. A graded `dz`
and a plane wave are refused with it.

Sources, polarized sources, the plane wave, point probes, DFT planes and flux
monitors define no port impedance or reference plane. Check their field,
resonance, far-field or flux result directly; do not use them as ports.

### Refusals and preflight

An S-parameter request outside its calculator raises; it never returns `None`
or drops a feature:

- `run(compute_s_params=True)` accepts only lumped and wire `add_port(...)`.
- `compute_msl_s_matrix(...)` accepts only microstrip ports.
- `compute_waveguide_s_matrix(...)` accepts only waveguide ports.
- `compute_coaxial_line_reflection(...)` accepts exactly one `face="top"`
  coaxial port within its line setup.
- `forward(port_s11_freqs=...)` accepts only lumped and wire ports on one
  device.

Check the routing before a long run:

```python
sim.preflight_sparameters(calculator="run")
sim.preflight_sparameters(calculator="forward")
sim.preflight_sparameters(calculator="msl")
sim.preflight_sparameters(calculator="waveguide")
```

These calls check that the API fits the model, not mesh or time convergence or
accuracy. Pass `strict=True` to raise on any reported issue.

### Mixed port families — experimental

`compute_mixed_s_matrix(freqs=...)` combines lumped or wire ports with
microstrip ports on a uniform mesh. It drives each port in turn, takes the
diagonal from each family's own extractor and the off-diagonal magnitudes from
Poynting flux.

| Aspect | Status |
|---|---|
| off-diagonal magnitude | experimental; its only independent internal check is reciprocity, which warns above `reciprocity_tol` (default 0.06) |
| absolute magnitude | not compared with another solver |
| off-diagonal phase | provisional: it mixes two reference-plane conventions |
| column power | an identity of the flux normalization, not a passivity check |

Neither diagonal is verified on this path. With `enforce_passivity=True` (the
default) the returned diagonal is a joint projection, not the measured value:
read `S_raw` and `passivity_correction` for what was measured
(`tests/unit/sparams/test_mixed_port_sparam.py`). Waveguide, Floquet, coaxial
and plane-wave registrations, bare sources, 0 Ω ports, mixed lumped+wire sets,
`reference_plane_cells`, lossy (`surface_impedance_f0=`) sheets, graded meshes,
subgridding and ADI are refused. The
flux channel also needs a PEC ground at `z_lo` and vertical (`component="ez"`)
lumped or wire ports.

### Coax-to-microstrip transition — experimental, diagnostic only

`compute_coax_msl_transition(...)` drives a coaxial port and a microstrip port
in turn. The returned matrix is not passive: on the same run, a Poynting flux
box measures about three times the microstrip-side power that the power-wave
extraction reports, and the mechanism is unresolved (run record in
`scripts/diagnostics/_coax_msl_transition_settled_run_logs/`; readings from
earlier attempts in that directory were taken through instrument defects since
fixed and must not be cited). The calculator therefore
refuses the matrix by default (`strict_passivity=True` raises `ValueError`);
`strict_passivity=False` returns it with a warning. Cross-family transitions
are not pursued before 2.0. For a result you can report, use the single-family
calculators: `compute_msl_s_matrix(...)` for the microstrip line and
`compute_coaxial_two_port(...)` or `compute_coaxial_line_reflection(...)` for
the coax. Gradients are not available on this path.

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
| Finite-region `add_flux_monitor(size=...)` + graded (`dz_profile`/`dy_profile`/`dx_profile`) mesh | **experimental** | Runs on uniform and graded grids. A shared geometry resolver selects a half-open CELL window inside the physical interior; neither CPML cells nor the trailing bounding-node slot are included. Every requested endpoint outside the interior emits a clamp warning, including sub-cell overflow. Empty windows are rejected. NU selection and metre bounds use the canonical float64 node spine; area weights retain the actual per-axis cell sizes. `preflight().flux_regions` reports requested bounds and bounds on the grid and indices as metadata, separate from findings, so aligned inputs do not become strict-mode failures. `size` and `center` must each contain exactly two finite tangential values, with positive sizes. `size=None` retains the legacy full allocated plane. Tests cover physical windows/area, finite-versus-restricted integration, graded fields, and material AD through the raw port hook. These checks do not establish general RF accuracy; the existing graded-normal-axis H co-location (0.5/0.5) limitation remains. See [finite flux regions](flux_regions.md). |
| TFSF + graded z | **experimental** | Only normal incidence along `+x` or `-x` (`angle_deg=0`) runs. Oblique incidence and incidence along `+z` or `-z` raise. This qualification requires a uniform injection axis x; grading x triggers the `nonuniform_tfsf` advisory because the incident line and grid waves no longer match. |
| Rectangular-waveguide S-matrix + nonuniform mesh (`dx_profile`/`dy_profile`/`dz_profile`) | **experimental** | Single-mode `normalize=True` and `normalize="flux"` run. A full-aperture guide with `boundary` omitted is **refused** (the graded mesh keeps absorbers on the transverse faces); declare the walls, e.g. `boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}`. The calculator runs the graded `run()` kernel and refuses what the graded `run()` refuses. Port injection reads the local primal cell width for its H correction and the local dual spacing for its E correction. Old boundary-cell-derived injection readings do not apply to ports inside a refined band. Analytic fixtures cover grading ratios 1--3 and relative permittivity 2 and 4. A passed Palace magnitude comparison covers `normalize="flux"`, a graded-`dy` ratio of 2, WR-90 empty/PEC-short/dielectric-slab cases, and 8.2--12.4 GHz (`max_mag_abs_diff=0.008529`, improved from `0.07009` when the path's absorber stopped being 0.33 of a guide wavelength — #576). Other profiles, bands, phase, multimode operation, and arbitrary junctions are not validated. `eps_override` and `sigma_override` differentiation is available only with `normalize="flux"`; only `eps_override` has a nonuniform AD-vs-FD regression test. Neither AD check establishes RF accuracy. Dispatch history: until #811 (fixed 2026-09-01) this API reached the nonuniform path only for `dx_profile`/`dy_profile` — a `dz_profile`-only simulation was silently solved on the uniform grid built from the scalar `dx` while preflight described the graded mesh (dz-only meshes with different `dt` returned bit-identical S in the negative test baseline, `scripts/diagnostics/wr90_dz_dispatch_falsifier.py`). A `dz_profile` now dispatches here under the same restrictions, and the WR-90 graded-propagation control is recorded by #1083. This is scoped evidence, not validation of arbitrary dz profiles. The path emits the same energy-based per-frequency decay check (`settling_db`, -40 dB aggregate warning) as the uniform single-mode path (#827 waveguide instance). Phase range on this path (derived arithmetic, not a measurement): the reference-plane shift `exp(-/+ jβΔ)` and the modal impedance evaluate β at the grid's BOUNDARY cell (`NonUniformGrid.dx`), not the cell the plane sits in. `Z_TE` does not depend on the cell size at all (the discrete `sin(β·dx/2)` equals `s_x·dx/2`), and the β difference is the second-order Yee correction `(β·dx)²/24` at the two sizes: for the committed WR-90 fixture (boundary 1.5 mm, fine 0.75 mm, discrete cutoff 6.650 GHz; `tests/fixtures/waveguide_nu_beta_cell_size_envelope.json`, replayed by `tests/unit/sparams/test_waveguide_nu_beta_cell_size_envelope.py`) Δβ = 0.057 / 0.271 / 0.652 rad/m at 8 / 10 / 12 GHz, i.e. 0.07° / 0.31° / 0.75° over a 20 mm plane offset (0.86° worst at 12.4 GHz) and 0.02° over that fixture's own 0.5 mm applied shift. Both of that fixture's port-to-reference spans lie in uniform 1.5 mm cells, so the two evaluations coincide there and the fixture cannot exercise the difference. A span that crosses cells of more than one size now raises `ValueError` (`tests/unit/sparams/test_waveguide_nu_grading_zone.py`) instead of applying a single β; a fixture with a reference plane inside the graded region, and β integration over the span, are deferred to #854 item 1. This bounds one mechanism; it is not a phase validation of the path. |
| Lumped-port S-parameters + nonuniform mesh | **limited** | Single-device `forward(port_s11_freqs=...)` accepts requested bins for lumped and wire ports, including their V/I accumulators. This does not establish arbitrary full-matrix support; distributed forward still refuses explicit port frequencies. |
| MSL S-matrix + nonuniform mesh | **experimental** | `mode="laplace"` and `mode="uniform"` have internal settled-S11 regression coverage. **diagnostic** — the stub-notch frequency over a substrate-cell ladder on a band-graded mesh is recorded in `validation/research/multiband_nu/results/msl_notch_graded_fz_after_1213.json` and replayed by `tests/unit/nonuniform/test_msl_notch_fz_after_1213_replay.py`; it is not an accuracy statement. `mode="eigenmode"` raises. |
| Coaxial port + nonuniform mesh | **unsupported** | The request must fail. |
| Lumped RLC update + nonuniform mesh | **limited** | R/L/C ADE elements participate in the field update. Material gradients through `eps_override` and `sigma_override` are implemented on graded meshes containing lumped RLC elements. Component-value gradients remain separately undocumented; requested-frequency port extraction has the scope stated above. |
| Multi-band graded mesh (fine bands on **z, x or y**, unequal sizes allowed, ratio <= 1.4) | **limited** | The mesh and solver have bounded evidence, not port, flux or far-field validation: energy, transition reflection, round trip, order and gradients on z; cavity accuracy, transition reflection and band-width gradients in-plane. Bands narrower than two cells, absorbers next to the grading, and in-plane ports or plane waves are not covered. See [Multi-band graded mesh](#multi-band-graded-mesh) below; the time step still follows the smallest cell. |
| Volumetric PEC scatterer + nonuniform waveguide | **experimental** | The device/reference handling is regression-tested, but no RF validation is documented for arbitrary iris, post, septum, branch, or T-junction geometries. The REALIZATION changed at 2.0 (#931): a PEC volume now realizes walls at both drawn faces and its drawn extent equals its extent on the grid, so the regression fixtures behind "regression-tested" are recomputed for the release and a fixture that has not been is not shipped. No accuracy claim moves with them, because this row makes none. |

### Multi-band graded mesh

**What is covered.** A graded profile with fine bands along z, x or y, in any
order, with every adjacent cell ratio at most 1.4, as one step or as a ramp.
`rfx.make_band_profile(edges, cell_sizes, max_ratio=1.4, protected=...,
boundary_cell=...)` builds one: every interface on a node plane, every
adjacent ratio within the cap, and the cells summing exactly to the declared
length (`tests/unit/nonuniform/test_band_profile_builder.py`). The builder
guarantees the mesh, not the accuracy of a result on it.

**The ratio warning.** Above 1.4 on `dz_profile` (1.3 on `dx_profile` /
`dy_profile`) the mesh still runs; `Simulation(...)` warns and preflight adds
`nu_grading_ratio_beyond_validated_cap`. Both are advisory.

**What the checks show.** The checks run in closed PEC boxes, except the
gradient check on the profile vector, which has an absorber and compares no
field. Scripts and results: `validation/research/multiband_nu/`; replayed by
`tests/unit/nonuniform/test_multiband_nu_envelope.py`.

- *Energy.* The discrete energy of a lossless graded box stays bounded over
  10^6 time steps, in 1-D and 3-D (`validation/research/multiband_nu/results/w1_pa_1d.json`,
  `w1_pb_full_gpu.json`).
- *Reflection at a transition.* Each transition reflects at the level an exact
  discrete scattering-chain model predicts, and the level falls as
  (cell / wavelength)^2, 12 dB per doubling of the resolution. At ratio 1.4 and
  30 fine cells per wavelength it is about −54 dB
  (`validation/research/multiband_nu/results/w2_w3.json`). A fine band between
  two transitions reflects as the sum of the two: at most twice one transition,
  oscillating with the band width (bands of 2 to 64 cells checked,
  `validation/research/multiband_nu/results/w6_band_builder.json`). Quote the
  law, not one dB value: it depends on the resolution.
- *Round trip.* A wave crossing up and back down the grading keeps its amplitude.
- *Order.* The resonance of an empty PEC cavity graded along z at ratio 1.4
  converges at second order, as on a uniform mesh, with an error about 1.6 times
  the uniform mesh's at the same fine cell
  (`validation/research/multiband_nu/results/w4r3_zdominant_cavity.json`).
- *In-plane.* With fine bands on x and y, a cavity frequency agrees with its
  closed form, and the transition reflection follows the same law on x and y
  as on z.
- *Gradients.* `jax.grad` with respect to layer thickness, permittivity, band
  width (x, y, z) and band position (x) agrees with central finite differences
  within their own error, with the time-step dependence included. The thickness
  gradient of a resonant DFT power is not verified.

**Not covered.**

- *Grading next to an absorber.* No check measures accuracy with an absorber
  present. Preflight emits `nu_grading_reaches_absorber` when an axis grades
  within the absorber's layer count; keep that many cells next to the absorber
  uniform, or close the face with PEC or PMC. `make_band_profile(...,
  boundary_cell=dx)` fixes only the two end cells, so it still draws this advisory on
  an in-plane absorbing face.
- *Time step.* The time step follows the smallest cell everywhere. A fine band
  buys cells, not time steps; local time stepping is not implemented.
- *Ratios above 1.4* (1.3 in-plane) are flagged, not validated.
- *In-plane:* the energy, round-trip and order checks; other ratios; dielectric
  interiors; ports, plane waves and waveguide results on an in-plane graded mesh.
- *Results.* None of this says an S-parameter, flux, far-field or port result on
  a multi-band mesh is accurate; each keeps the status of its own row above.
- *Convergence on a loaded structure.* On a dielectric-loaded microstrip-class
  structure the resonance did not move monotonically with the mesh, and whether
  it converges to the physical answer is not settled
  (`validation/research/convergence_floor/results/`). Show convergence on your
  own geometry; `ladder_guard`
  (`validation/research/convergence_floor/ladder_guard.py`, tested by
  `tests/unit/misc/test_ladder_guard.py`) catches a mesh ladder read across a turn.

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
