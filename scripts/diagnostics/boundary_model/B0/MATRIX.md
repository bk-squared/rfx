# Realized-boundary matrix

This is a boundary-call measurement; no S-parameter or resonance conclusion is assigned here.

E[n] denotes tangential E zeroing at node index n. H[n] denotes tangential H zeroing at Yee index n. CPML:n is the number of nonzero-sigma entries (8 allocated layers give 7 nonzero entries); UPML:E/H gives both counts. ADI-conductivity names the measured conductivity layer, not a CPML update. `none observed` means none of these operations was recorded, not a claim about a physical open boundary.

Two steps; 24 x 20 x 16 mm domain; dx = 1 mm; 8 absorber layers. JSON and adjacent logs preserve the declared spec, realized shape, pads, call arguments, preflight text and full exception. GPU jobs record their GPU fast-path dispatch. Distributed supplements use two emulated CPU devices. The nonuniform metadata correction is selected ahead of the original pre-entry metadata refusal. Sigma count replays for UPML/ADI are in `profile_measurement.json`; CPML profiles come directly from observed init returns.

## pec

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| wire-fast | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| forward | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| sweep | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| nonuniform | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| subgridded | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| distributed | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| adi | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| run [GPU] | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| wire-fast [GPU] | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| forward [GPU] | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| sweep [GPU] | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| adi [GPU] | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |

- run: `matrix/pec__run_cpu.json`
- wire-fast: `matrix/pec__wire-fast_cpu.json`
- forward: `matrix/pec__forward_cpu.json`
- sweep: `matrix/pec__sweep_cpu.json`
- nonuniform: `matrix/pec__nonuniform_nu_measured.json`
- subgridded: `matrix/pec__subgridded_subgrid_measured.json`
  - Fine grid [49, 41, 25]: `apply_pec_faces`, faces=['x_hi', 'x_lo', 'y_hi', 'y_lo', 'z_lo'], axes=None. Tangential-E low index 0; high index = axis extent minus 1. Refinement spans z = 0 to 12 mm; the fine z-high interface is internal.
  - Fine grid [49, 41, 25]: `apply_pec_faces`, faces=['x_hi', 'x_lo', 'y_hi', 'y_lo', 'z_lo'], axes=None. Tangential-E low index 0; high index = axis extent minus 1. Refinement spans z = 0 to 12 mm; the fine z-high interface is internal.
- distributed: `matrix/pec__distributed_cpu.json`
- adi: `matrix/pec__adi_cpu.json`
- run [GPU]: `matrix/pec__run.json`
- wire-fast [GPU]: `matrix/pec__wire-fast.json`
- forward [GPU]: `matrix/pec__forward.json`
- sweep [GPU]: `matrix/pec__sweep.json`
- adi [GPU]: `matrix/pec__adi.json`

## cpml

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| wire-fast | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| forward | CPML:7 | CPML:7 | CPML:7 | CPML:7 | CPML:7 | CPML:7 |
| sweep | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| nonuniform | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| subgridded | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| distributed | CPML:7 | CPML:7 | CPML:7 | CPML:7 | CPML:7 | CPML:7 |
| adi | E[0]; ADI-conductivity:7 | E[40]; ADI-conductivity:7 | E[0]; ADI-conductivity:7 | E[36]; ADI-conductivity:7 | E[0]; ADI-conductivity:7 | E[32]; ADI-conductivity:7 |
| run [GPU] | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| wire-fast [GPU] | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| forward [GPU] | CPML:7 | CPML:7 | CPML:7 | CPML:7 | CPML:7 | CPML:7 |
| sweep [GPU] | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| adi [GPU] | E[0]; ADI-conductivity:7 | E[40]; ADI-conductivity:7 | E[0]; ADI-conductivity:7 | E[36]; ADI-conductivity:7 | E[0]; ADI-conductivity:7 | E[32]; ADI-conductivity:7 |

- run: `matrix/cpml__run_cpu.json`
- wire-fast: `matrix/cpml__wire-fast_cpu.json`
- forward: `matrix/cpml__forward_cpu.json`
- sweep: `matrix/cpml__sweep_cpu.json`
- nonuniform: `matrix/cpml__nonuniform_nu_measured.json`
- subgridded: `matrix/cpml__subgridded_subgrid_measured.json` — ValueError: subgrid validation: supported=False mode=production level=production-z-slab-guarded-boundary-vacuum-envelope - ERROR [subgrid_overlaps_absorber] production subgrid z interfaces must stay outside CPML/UPML - ERROR [boundary_terminated_requires_pec_no_cpml] one-sided boundary-terminated production subgrid support requires the refined slab to touch a PEC z face and avoid any fine-grid face that would need CPML/UPML; only the opposite z face may be absorbing
- distributed: `matrix/cpml__distributed_cpu.json`
- adi: `matrix/cpml__adi_cpu.json`
- run [GPU]: `matrix/cpml__run.json`
- wire-fast [GPU]: `matrix/cpml__wire-fast.json`
- forward [GPU]: `matrix/cpml__forward.json`
- sweep [GPU]: `matrix/cpml__sweep.json`
- adi [GPU]: `matrix/cpml__adi.json`

## upml

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | E[0]; UPML:8/8 | E[40]; UPML:8/8 | E[0]; UPML:8/8 | E[36]; UPML:8/8 | E[0]; UPML:8/8 | E[32]; UPML:8/8 |
| wire-fast | E[0]; UPML:8/8 | E[40]; UPML:8/8 | E[0]; UPML:8/8 | E[36]; UPML:8/8 | E[0]; UPML:8/8 | E[32]; UPML:8/8 |
| forward | UPML:8/8 | UPML:8/8 | UPML:8/8 | UPML:8/8 | UPML:8/8 | UPML:8/8 |
| sweep | E[0]; UPML:8/8 | E[40]; UPML:8/8 | E[0]; UPML:8/8 | E[36]; UPML:8/8 | E[0]; UPML:8/8 | E[32]; UPML:8/8 |
| nonuniform | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| subgridded | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| distributed | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| adi | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| run [GPU] | E[0]; UPML:8/8 | E[40]; UPML:8/8 | E[0]; UPML:8/8 | E[36]; UPML:8/8 | E[0]; UPML:8/8 | E[32]; UPML:8/8 |
| wire-fast [GPU] | E[0]; UPML:8/8 | E[40]; UPML:8/8 | E[0]; UPML:8/8 | E[36]; UPML:8/8 | E[0]; UPML:8/8 | E[32]; UPML:8/8 |
| forward [GPU] | UPML:8/8 | UPML:8/8 | UPML:8/8 | UPML:8/8 | UPML:8/8 | UPML:8/8 |
| sweep [GPU] | E[0]; UPML:8/8 | E[40]; UPML:8/8 | E[0]; UPML:8/8 | E[36]; UPML:8/8 | E[0]; UPML:8/8 | E[32]; UPML:8/8 |

- run: `matrix/upml__run_cpu.json`
- wire-fast: `matrix/upml__wire-fast_cpu.json`
- forward: `matrix/upml__forward_cpu.json`
- sweep: `matrix/upml__sweep_cpu.json`
- nonuniform: `matrix/upml__nonuniform_nu_measured.json` — ValueError: boundary='upml' does not support the non-uniform run() lane: the non-uniform runner implements CPML only (rfx/nonuniform.py dispatches on cpml_layers > 0 and never reads the boundary type), so this configuration used to run CPML while sim._boundary kept reporting 'upml' (issue #680). Pass boundary='cpml' to run the absorber that is actually implemented here, or drop the mesh profile(s) to use the uniform lane, which does implement UPML.
- subgridded: `matrix/upml__subgridded_subgrid_measured.json` — ValueError: [run] preflight found 1 blocking error(s) (pass skip_preflight=True to bypass):   - ERROR: boundary='upml' does not support subgridding/refinement
- distributed: `matrix/upml__distributed_cpu.json` — ValueError: boundary='upml' does not support distributed execution
- adi: `matrix/upml__adi_cpu.json` — ValueError: solver='adi' does not support boundary='upml'
- run [GPU]: `matrix/upml__run.json`
- wire-fast [GPU]: `matrix/upml__wire-fast.json`
- forward [GPU]: `matrix/upml__forward.json`
- sweep [GPU]: `matrix/upml__sweep.json`

## pmc-pec

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | E[0]; H[0] | E[24]; H[23] | E[0] | E[20] | E[0] | E[16] |
| wire-fast | E[0]; H[0] | E[24]; H[23] | E[0] | E[20] | E[0] | E[16] |
| forward | E[0]; H[0] | E[24]; H[23] | E[0] | E[20] | E[0] | E[16] |
| sweep | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| nonuniform | E[0]; H[0] | E[24]; H[23] | E[0] | E[20] | E[0] | E[16] |
| subgridded | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| distributed | E[0]; H[0] | E[24]; H[23] | E[0] | E[20] | E[0] | E[16] |
| adi | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| run [GPU] | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| wire-fast [GPU] | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| forward [GPU] | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| sweep [GPU] | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |
| adi [GPU] | E[0] | E[24] | E[0] | E[20] | E[0] | E[16] |

- run: `matrix/pmc-pec__run_cpu.json`
- wire-fast: `matrix/pmc-pec__wire-fast_cpu.json`
- forward: `matrix/pmc-pec__forward_cpu.json`
- sweep: `matrix/pmc-pec__sweep_cpu.json`
- nonuniform: `matrix/pmc-pec__nonuniform_nu_measured.json`
- subgridded: `matrix/pmc-pec__subgridded_subgrid_measured.json`
  - Fine grid [49, 41, 25]: `apply_pec_faces`, faces=['x_hi', 'x_lo', 'y_hi', 'y_lo', 'z_lo'], axes=None. Tangential-E low index 0; high index = axis extent minus 1. Refinement spans z = 0 to 12 mm; the fine z-high interface is internal.
  - Fine grid [49, 41, 25]: `apply_pec_faces`, faces=['x_hi', 'x_lo', 'y_hi', 'y_lo', 'z_lo'], axes=None. Tangential-E low index 0; high index = axis extent minus 1. Refinement spans z = 0 to 12 mm; the fine z-high interface is internal.
- distributed: `matrix/pmc-pec__distributed_cpu.json`
- adi: `matrix/pmc-pec__adi_cpu.json`
- run [GPU]: `matrix/pmc-pec__run.json`
- wire-fast [GPU]: `matrix/pmc-pec__wire-fast.json`
- forward [GPU]: `matrix/pmc-pec__forward.json`
- sweep [GPU]: `matrix/pmc-pec__sweep.json`
- adi [GPU]: `matrix/pmc-pec__adi.json`

## pmc-cpml

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | E[0]; H[0] | E[24]; H[23] | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| wire-fast | E[0]; H[0] | E[24]; H[23] | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| forward | H[0] | H[23] | CPML:7 | CPML:7 | CPML:7 | CPML:7 |
| sweep | E[0] | E[24] | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| nonuniform | E[0]; H[0] | E[24]; H[23] | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| subgridded | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| distributed | H[0]; CPML:7 | H[23]; CPML:7 | CPML:7 | CPML:7 | CPML:7 | CPML:7 |
| adi | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| run [GPU] | E[0]; H[0] | E[24]; H[23] | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| wire-fast [GPU] | E[0]; H[0] | E[24]; H[23] | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| forward [GPU] | H[0] | H[23] | CPML:7 | CPML:7 | CPML:7 | CPML:7 |
| sweep [GPU] | E[0] | E[24] | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |

- run: `matrix/pmc-cpml__run_cpu.json`
- wire-fast: `matrix/pmc-cpml__wire-fast_cpu.json`
- forward: `matrix/pmc-cpml__forward_cpu.json`
- sweep: `matrix/pmc-cpml__sweep_cpu.json`
- nonuniform: `matrix/pmc-cpml__nonuniform_nu_measured.json`
- subgridded: `matrix/pmc-cpml__subgridded_subgrid_measured.json` — ValueError: subgrid validation: supported=False mode=production level=production-z-slab-guarded-boundary-vacuum-envelope - ERROR [subgrid_overlaps_absorber] production subgrid z interfaces must stay outside CPML/UPML - ERROR [boundary_terminated_requires_pec_no_cpml] one-sided boundary-terminated production subgrid support requires the refined slab to touch a PEC z face and avoid any fine-grid face that would need CPML/UPML; only the opposite z face may be absorbing
- distributed: `matrix/pmc-cpml__distributed_cpu.json`
- adi: `matrix/pmc-cpml__adi_cpu.json` — ValueError: solver='adi' supports only a uniform absorber: its absorbing layer is stamped on all six faces at the scalar cpml_layers, so a per-face boundary layout (x_hi='pmc', x_lo='pmc', y_hi='cpml', y_lo='cpml', z_hi='cpml', z_lo='cpml') would silently absorb on the faces you declared as reflectors. Use solver='yee' for per-face boundaries, or make every face 'cpml' with no per-face thickness override.
- run [GPU]: `matrix/pmc-cpml__run.json`
- wire-fast [GPU]: `matrix/pmc-cpml__wire-fast.json`
- forward [GPU]: `matrix/pmc-cpml__forward.json`
- sweep [GPU]: `matrix/pmc-cpml__sweep.json`

## pec-zlo

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0] | E[24]; CPML:7 |
| wire-fast | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0] | E[24]; CPML:7 |
| forward | CPML:7 | CPML:7 | CPML:7 | CPML:7 | E[0] | CPML:7 |
| sweep | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0] | E[24]; CPML:7 |
| nonuniform | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0] | E[24]; CPML:7 |
| subgridded | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| distributed | CPML:7 | CPML:7 | CPML:7 | CPML:7 | CPML:7 | CPML:7 |
| adi | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| run [GPU] | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0] | E[24]; CPML:7 |
| wire-fast [GPU] | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0] | E[24]; CPML:7 |
| forward [GPU] | CPML:7 | CPML:7 | CPML:7 | CPML:7 | E[0] | CPML:7 |
| sweep [GPU] | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0] | E[24]; CPML:7 |

- run: `matrix/pec-zlo__run_cpu.json`
- wire-fast: `matrix/pec-zlo__wire-fast_cpu.json`
- forward: `matrix/pec-zlo__forward_cpu.json`
- sweep: `matrix/pec-zlo__sweep_cpu.json`
- nonuniform: `matrix/pec-zlo__nonuniform_nu_measured.json`
- subgridded: `matrix/pec-zlo__subgridded_subgrid_measured.json` — ValueError: subgrid validation: supported=False mode=production level=production-z-slab-guarded-boundary-vacuum-envelope - ERROR [boundary_terminated_requires_pec_no_cpml] one-sided boundary-terminated production subgrid support requires the refined slab to touch a PEC z face and avoid any fine-grid face that would need CPML/UPML; only the opposite z face may be absorbing
- distributed: `matrix/pec-zlo__distributed_cpu.json`
- adi: `matrix/pec-zlo__adi_cpu.json` — ValueError: solver='adi' supports only a uniform absorber: its absorbing layer is stamped on all six faces at the scalar cpml_layers, so a per-face boundary layout (x_hi='cpml', x_lo='cpml', y_hi='cpml', y_lo='cpml', z_hi='cpml', z_lo='pec') would silently absorb on the faces you declared as reflectors. Use solver='yee' for per-face boundaries, or make every face 'cpml' with no per-face thickness override.
- run [GPU]: `matrix/pec-zlo__run.json`
- wire-fast [GPU]: `matrix/pec-zlo__wire-fast.json`
- forward [GPU]: `matrix/pec-zlo__forward.json`
- sweep [GPU]: `matrix/pec-zlo__sweep.json`

## periodic-xy

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | wrap | wrap | wrap | wrap | E[0]; CPML:7 | E[32]; CPML:7 |
| wire-fast | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| forward | wrap | wrap | wrap | wrap | CPML:7 | CPML:7 |
| sweep | wrap | wrap | wrap | wrap | E[0]; CPML:7 | E[32]; CPML:7 |
| nonuniform | E[0] | E[24] | E[0] | E[20] | E[0]; CPML:7 | E[32]; CPML:7 |
| subgridded | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| distributed | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| adi | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| run [GPU] | wrap | wrap | wrap | wrap | E[0]; CPML:7 | E[32]; CPML:7 |
| forward [GPU] | wrap | wrap | wrap | wrap | CPML:7 | CPML:7 |
| sweep [GPU] | wrap | wrap | wrap | wrap | E[0]; CPML:7 | E[32]; CPML:7 |

- run: `matrix/periodic-xy__run_cpu.json`
- wire-fast: `matrix/periodic-xy__wire-fast_cpu.json` — NotImplementedError: run(compute_s_params=True) for lumped/wire add_port(...) does not honor periodic axes: the S-parameter extraction re-run uses non-periodic boundaries, so the returned S-matrix would silently ignore set_periodic_axes('xy'). Remove the periodic axes for the S-parameter run, or use a port family that supports periodicity (e.g. a Floquet port).
- forward: `matrix/periodic-xy__forward_cpu.json`
- sweep: `matrix/periodic-xy__sweep_cpu.json`
- nonuniform: `matrix/periodic-xy__nonuniform_nu_measured.json`
- subgridded: `matrix/periodic-xy__subgridded_subgrid_measured.json` — ValueError: subgrid validation: supported=False mode=production level=production-z-slab-guarded-boundary-vacuum-envelope - ERROR [subgrid_overlaps_absorber] production subgrid z interfaces must stay outside CPML/UPML - ERROR [boundary_terminated_requires_pec_no_cpml] one-sided boundary-terminated production subgrid support requires the refined slab to touch a PEC z face and avoid any fine-grid face that would need CPML/UPML; only the opposite z face may be absorbing
- distributed: `matrix/periodic-xy__distributed_cpu.json` — NotImplementedError: periodic axes 'x', 'y': periodic / Bloch boundaries are not supported on the distributed multi-device run() path; the lane would use the declared non-periodic wall instead (rfx.runners.distributed._update_h_local / _update_e_local). Remove the periodic axes / Bloch phase, or omit devices=... (use a single-device run() instead).
- adi: `matrix/periodic-xy__adi_cpu.json` — ValueError: solver='adi' supports only a uniform absorber: its absorbing layer is stamped on all six faces at the scalar cpml_layers, so a per-face boundary layout (x_hi='periodic', x_lo='periodic', y_hi='periodic', y_lo='periodic', z_hi='cpml', z_lo='cpml') would silently absorb on the faces you declared as reflectors. Use solver='yee' for per-face boundaries, or make every face 'cpml' with no per-face thickness override.
- run [GPU]: `matrix/periodic-xy__run.json`
- forward [GPU]: `matrix/periodic-xy__forward.json`
- sweep [GPU]: `matrix/periodic-xy__sweep.json`

## tfsf

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | E[0]; CPML:7 | E[40]; CPML:7 | wrap | wrap | wrap | wrap |
| wire-fast | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| forward | CPML:7 | CPML:7 | wrap | wrap | wrap | wrap |
| sweep | E[0]; CPML:7 | E[40]; CPML:7 | wrap | wrap | wrap | wrap |
| nonuniform | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| subgridded | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| distributed | E[0]; CPML:7 | E[40]; CPML:7 | wrap | wrap | wrap | wrap |
| adi | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| run [GPU] | E[0]; CPML:7 | E[40]; CPML:7 | wrap | wrap | wrap | wrap |
| forward [GPU] | CPML:7 | CPML:7 | wrap | wrap | wrap | wrap |
| sweep [GPU] | E[0]; CPML:7 | E[40]; CPML:7 | wrap | wrap | wrap | wrap |

- run: `matrix/tfsf__run_cpu.json`
- wire-fast: `matrix/tfsf__wire-fast_cpu.json` — ValueError: Lumped ports are not supported together with the TFSF plane-wave source
- forward: `matrix/tfsf__forward_cpu.json`
- sweep: `matrix/tfsf__sweep_cpu.json`
- nonuniform: `matrix/tfsf__nonuniform_nu_measured.json`
- subgridded: `matrix/tfsf__subgridded_subgrid_measured.json` — ValueError: subgrid validation: supported=False mode=production level=production-z-slab-guarded-boundary-vacuum-envelope - ERROR [subgrid_overlaps_absorber] production subgrid z interfaces must stay outside CPML/UPML - ERROR [boundary_terminated_requires_pec_no_cpml] one-sided boundary-terminated production subgrid support requires the refined slab to touch a PEC z face and avoid any fine-grid face that would need CPML/UPML; only the opposite z face may be absorbing - ERROR [tfsf_unvalidated] TFSF is not validated with subgridding
- distributed: `matrix/tfsf__distributed_cpu.json`
- adi: `matrix/tfsf__adi_cpu.json` — ValueError: solver='adi' does not support TFSF sources yet
- run [GPU]: `matrix/tfsf__run.json`
- forward [GPU]: `matrix/tfsf__forward.json`
- sweep [GPU]: `matrix/tfsf__sweep.json`

## waveguide-cpml

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| wire-fast | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| forward | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| sweep | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| nonuniform | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; CPML:7 | E[36]; CPML:7 | E[0]; CPML:7 | E[32]; CPML:7 |
| subgridded | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| distributed | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| adi | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| run [GPU] | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| forward [GPU] | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| sweep [GPU] | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |

- run: `matrix/waveguide-cpml__run_cpu.json`
- wire-fast: `matrix/waveguide-cpml__wire-fast_cpu.json` — NotImplementedError: run(compute_s_params=True) has a single result schema for add_port(...) lumped/wire ports. Mixed or specialized port families must use their documented calculators: add_waveguide_port(...) uses compute_waveguide_s_matrix() for the full S-matrix; run() may return per-port result.waveguide_sparams but not Result.s_params.
- forward: `matrix/waveguide-cpml__forward_cpu.json`
- sweep: `matrix/waveguide-cpml__sweep_cpu.json`
- nonuniform: `matrix/waveguide-cpml__nonuniform_nu_measured.json`
- subgridded: `matrix/waveguide-cpml__subgridded_subgrid_measured.json` — ValueError: subgrid validation: supported=False mode=production level=production-z-slab-guarded-boundary-vacuum-envelope - ERROR [boundary_terminated_requires_pec_no_cpml] one-sided boundary-terminated production subgrid support requires the refined slab to touch a PEC z face and avoid any fine-grid face that would need CPML/UPML; only the opposite z face may be absorbing - ERROR [waveguide_port_unvalidated] waveguide ports are not validated with subgridding
- distributed: `matrix/waveguide-cpml__distributed_cpu.json`
- adi: `matrix/waveguide-cpml__adi_cpu.json` — ValueError: solver='adi' does not support waveguide or Floquet ports yet
- run [GPU]: `matrix/waveguide-cpml__run.json`
- forward [GPU]: `matrix/waveguide-cpml__forward.json`
- sweep [GPU]: `matrix/waveguide-cpml__sweep.json`

## waveguide-pmc

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | CPML:7 | CPML:7 | E[0]; H[0] | E[20]; H[19] | E[0]; H[0] | E[16]; H[15] |
| wire-fast | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| forward | CPML:7 | CPML:7 | E[0]; H[0] | E[20]; H[19] | E[0]; H[0] | E[16]; H[15] |
| sweep | CPML:7 | CPML:7 | E[0]; H[0] | E[20]; H[19] | E[0]; H[0] | E[16]; H[15] |
| nonuniform | E[0]; CPML:7 | E[40]; CPML:7 | E[0]; H[0] | E[20]; H[19] | E[0]; H[0] | E[16]; H[15] |
| subgridded | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| distributed | CPML:7 | CPML:7 | E[0]; H[0] | E[20]; H[19] | E[0]; H[0] | E[16]; H[15] |
| adi | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| run [GPU] | CPML:7 | CPML:7 | E[0]; H[0] | E[20]; H[19] | E[0]; H[0] | E[16]; H[15] |
| forward [GPU] | CPML:7 | CPML:7 | E[0]; H[0] | E[20]; H[19] | E[0]; H[0] | E[16]; H[15] |
| sweep [GPU] | CPML:7 | CPML:7 | E[0]; H[0] | E[20]; H[19] | E[0]; H[0] | E[16]; H[15] |

- run: `matrix/waveguide-pmc__run_cpu.json`
- wire-fast: `matrix/waveguide-pmc__wire-fast_cpu.json` — NotImplementedError: run(compute_s_params=True) has a single result schema for add_port(...) lumped/wire ports. Mixed or specialized port families must use their documented calculators: add_waveguide_port(...) uses compute_waveguide_s_matrix() for the full S-matrix; run() may return per-port result.waveguide_sparams but not Result.s_params.
- forward: `matrix/waveguide-pmc__forward_cpu.json`
- sweep: `matrix/waveguide-pmc__sweep_cpu.json`
- nonuniform: `matrix/waveguide-pmc__nonuniform_nu_measured.json`
- subgridded: `matrix/waveguide-pmc__subgridded_subgrid_measured.json` — ValueError: subgrid validation: supported=False mode=production level=production-z-slab-guarded-boundary-vacuum-envelope - ERROR [boundary_terminated_requires_pec_no_cpml] one-sided boundary-terminated production subgrid support requires the refined slab to touch a PEC z face and avoid any fine-grid face that would need CPML/UPML; only the opposite z face may be absorbing - ERROR [waveguide_port_unvalidated] waveguide ports are not validated with subgridding
- distributed: `matrix/waveguide-pmc__distributed_cpu.json`
- adi: `matrix/waveguide-pmc__adi_cpu.json` — ValueError: solver='adi' supports only a uniform absorber: its absorbing layer is stamped on all six faces at the scalar cpml_layers, so a per-face boundary layout (x_hi='cpml', x_lo='cpml', y_hi='pmc', y_lo='pmc', z_hi='pmc', z_lo='pmc') would silently absorb on the faces you declared as reflectors. Use solver='yee' for per-face boundaries, or make every face 'cpml' with no per-face thickness override.
- run [GPU]: `matrix/waveguide-pmc__run.json`
- forward [GPU]: `matrix/waveguide-pmc__forward.json`
- sweep [GPU]: `matrix/waveguide-pmc__sweep.json`

## waveguide-pec

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| wire-fast | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| forward | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| sweep | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| nonuniform | E[0]; CPML:7 | E[40]; CPML:7 | E[0] | E[20] | E[0] | E[16] |
| subgridded | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| distributed | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| adi | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| run [GPU] | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| forward [GPU] | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |
| sweep [GPU] | CPML:7 | CPML:7 | E[0] | E[20] | E[0] | E[16] |

- run: `matrix/waveguide-pec__run_cpu.json`
- wire-fast: `matrix/waveguide-pec__wire-fast_cpu.json` — NotImplementedError: run(compute_s_params=True) has a single result schema for add_port(...) lumped/wire ports. Mixed or specialized port families must use their documented calculators: add_waveguide_port(...) uses compute_waveguide_s_matrix() for the full S-matrix; run() may return per-port result.waveguide_sparams but not Result.s_params.
- forward: `matrix/waveguide-pec__forward_cpu.json`
- sweep: `matrix/waveguide-pec__sweep_cpu.json`
- nonuniform: `matrix/waveguide-pec__nonuniform_nu_measured.json`
- subgridded: `matrix/waveguide-pec__subgridded_subgrid_measured.json` — ValueError: subgrid validation: supported=False mode=production level=production-z-slab-guarded-boundary-vacuum-envelope - ERROR [boundary_terminated_requires_pec_no_cpml] one-sided boundary-terminated production subgrid support requires the refined slab to touch a PEC z face and avoid any fine-grid face that would need CPML/UPML; only the opposite z face may be absorbing - ERROR [waveguide_port_unvalidated] waveguide ports are not validated with subgridding
- distributed: `matrix/waveguide-pec__distributed_cpu.json`
- adi: `matrix/waveguide-pec__adi_cpu.json` — ValueError: solver='adi' supports only a uniform absorber: its absorbing layer is stamped on all six faces at the scalar cpml_layers, so a per-face boundary layout (x_hi='cpml', x_lo='cpml', y_hi='pec', y_lo='pec', z_hi='pec', z_lo='pec') would silently absorb on the faces you declared as reflectors. Use solver='yee' for per-face boundaries, or make every face 'cpml' with no per-face thickness override.
- run [GPU]: `matrix/waveguide-pec__run.json`
- forward [GPU]: `matrix/waveguide-pec__forward.json`
- sweep [GPU]: `matrix/waveguide-pec__sweep.json`

## floquet

| Entry | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi |
|---|---|---|---|---|---|---|
| run | wrap | wrap | wrap | wrap | E[0]; CPML:7 | E[32]; CPML:7 |
| wire-fast | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| forward | wrap | wrap | wrap | wrap | CPML:7 | CPML:7 |
| sweep | wrap | wrap | wrap | wrap | E[0]; CPML:7 | E[32]; CPML:7 |
| nonuniform | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| subgridded | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| distributed | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| adi | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| run [GPU] | wrap | wrap | wrap | wrap | E[0]; CPML:7 | E[32]; CPML:7 |
| forward [GPU] | wrap | wrap | wrap | wrap | CPML:7 | CPML:7 |
| sweep [GPU] | wrap | wrap | wrap | wrap | E[0]; CPML:7 | E[32]; CPML:7 |

- run: `matrix/floquet__run_cpu.json`
- wire-fast: `matrix/floquet__wire-fast_cpu.json` — NotImplementedError: run(compute_s_params=True) has a single result schema for add_port(...) lumped/wire ports. Mixed or specialized port families must use their documented calculators: add_floquet_port(...) is experimental and has no claims-bearing run(compute_s_params=True) S-matrix path.
- forward: `matrix/floquet__forward_cpu.json`
- sweep: `matrix/floquet__sweep_cpu.json`
- nonuniform: `matrix/floquet__nonuniform_nu_measured.json` — ValueError: Floquet ports do not support non-uniform z mesh (dz_profile). Set dx explicitly to prevent auto-mesh from creating NU grid.
- subgridded: `matrix/floquet__subgridded_subgrid_measured.json` — ValueError: subgrid validation: supported=False mode=production level=production-z-slab-guarded-boundary-vacuum-envelope - ERROR [subgrid_overlaps_absorber] production subgrid z interfaces must stay outside CPML/UPML - ERROR [boundary_terminated_requires_pec_no_cpml] one-sided boundary-terminated production subgrid support requires the refined slab to touch a PEC z face and avoid any fine-grid face that would need CPML/UPML; only the opposite z face may be absorbing - ERROR [floquet_port_unvalidated] Floquet ports are not validated with subgridding
- distributed: `matrix/floquet__distributed_cpu.json` — NotImplementedError: periodic axes 'x', 'y': periodic / Bloch boundaries are not supported on the distributed multi-device run() path; the lane would use the declared non-periodic wall instead (rfx.runners.distributed._update_h_local / _update_e_local). Remove the periodic axes / Bloch phase, or omit devices=... (use a single-device run() instead).
- adi: `matrix/floquet__adi_cpu.json` — ValueError: solver='adi' does not support waveguide or Floquet ports yet
- run [GPU]: `matrix/floquet__run.json`
- forward [GPU]: `matrix/floquet__forward.json`
- sweep [GPU]: `matrix/floquet__sweep.json`

## Entry-point disagreements

Every differing measured face/kind pair is listed; absolute high-array indices are normalized to the high endpoint for this comparison. Refused cells have no realized kind to compare.

- cpml, x_lo: run `E[0]; CPML:7`; forward `CPML:7`.
- cpml, x_hi: run `E[40]; CPML:7`; forward `CPML:7`.
- cpml, y_lo: run `E[0]; CPML:7`; forward `CPML:7`.
- cpml, y_hi: run `E[36]; CPML:7`; forward `CPML:7`.
- cpml, z_lo: run `E[0]; CPML:7`; forward `CPML:7`.
- cpml, z_hi: run `E[32]; CPML:7`; forward `CPML:7`.
- cpml, x_lo: run `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, x_hi: run `E[40]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, y_lo: run `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, y_hi: run `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, z_lo: run `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, z_hi: run `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, x_lo: run `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, x_hi: run `E[40]; CPML:7`; distributed `CPML:7`.
- cpml, y_lo: run `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, y_hi: run `E[36]; CPML:7`; distributed `CPML:7`.
- cpml, z_lo: run `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, z_hi: run `E[32]; CPML:7`; distributed `CPML:7`.
- cpml, x_lo: run `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, x_hi: run `E[40]; CPML:7`; adi `E[40]; ADI-conductivity:7`.
- cpml, y_lo: run `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, y_hi: run `E[36]; CPML:7`; adi `E[36]; ADI-conductivity:7`.
- cpml, z_lo: run `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, z_hi: run `E[32]; CPML:7`; adi `E[32]; ADI-conductivity:7`.
- cpml, x_lo: run `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, x_hi: run `E[40]; CPML:7`; adi [GPU] `E[40]; ADI-conductivity:7`.
- cpml, y_lo: run `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, y_hi: run `E[36]; CPML:7`; adi [GPU] `E[36]; ADI-conductivity:7`.
- cpml, z_lo: run `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, z_hi: run `E[32]; CPML:7`; adi [GPU] `E[32]; ADI-conductivity:7`.
- cpml, x_lo: run [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- cpml, x_hi: run [GPU] `E[40]; CPML:7`; forward `CPML:7`.
- cpml, y_lo: run [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- cpml, y_hi: run [GPU] `E[36]; CPML:7`; forward `CPML:7`.
- cpml, z_lo: run [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- cpml, z_hi: run [GPU] `E[32]; CPML:7`; forward `CPML:7`.
- cpml, x_lo: run [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, x_hi: run [GPU] `E[40]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, y_lo: run [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, y_hi: run [GPU] `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, z_lo: run [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, z_hi: run [GPU] `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, x_lo: run [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, x_hi: run [GPU] `E[40]; CPML:7`; distributed `CPML:7`.
- cpml, y_lo: run [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, y_hi: run [GPU] `E[36]; CPML:7`; distributed `CPML:7`.
- cpml, z_lo: run [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, z_hi: run [GPU] `E[32]; CPML:7`; distributed `CPML:7`.
- cpml, x_lo: run [GPU] `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, x_hi: run [GPU] `E[40]; CPML:7`; adi `E[40]; ADI-conductivity:7`.
- cpml, y_lo: run [GPU] `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, y_hi: run [GPU] `E[36]; CPML:7`; adi `E[36]; ADI-conductivity:7`.
- cpml, z_lo: run [GPU] `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, z_hi: run [GPU] `E[32]; CPML:7`; adi `E[32]; ADI-conductivity:7`.
- cpml, x_lo: run [GPU] `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, x_hi: run [GPU] `E[40]; CPML:7`; adi [GPU] `E[40]; ADI-conductivity:7`.
- cpml, y_lo: run [GPU] `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, y_hi: run [GPU] `E[36]; CPML:7`; adi [GPU] `E[36]; ADI-conductivity:7`.
- cpml, z_lo: run [GPU] `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, z_hi: run [GPU] `E[32]; CPML:7`; adi [GPU] `E[32]; ADI-conductivity:7`.
- cpml, x_lo: wire-fast `E[0]; CPML:7`; forward `CPML:7`.
- cpml, x_hi: wire-fast `E[40]; CPML:7`; forward `CPML:7`.
- cpml, y_lo: wire-fast `E[0]; CPML:7`; forward `CPML:7`.
- cpml, y_hi: wire-fast `E[36]; CPML:7`; forward `CPML:7`.
- cpml, z_lo: wire-fast `E[0]; CPML:7`; forward `CPML:7`.
- cpml, z_hi: wire-fast `E[32]; CPML:7`; forward `CPML:7`.
- cpml, x_lo: wire-fast `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, x_hi: wire-fast `E[40]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, y_lo: wire-fast `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, y_hi: wire-fast `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, z_lo: wire-fast `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, z_hi: wire-fast `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, x_lo: wire-fast `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, x_hi: wire-fast `E[40]; CPML:7`; distributed `CPML:7`.
- cpml, y_lo: wire-fast `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, y_hi: wire-fast `E[36]; CPML:7`; distributed `CPML:7`.
- cpml, z_lo: wire-fast `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, z_hi: wire-fast `E[32]; CPML:7`; distributed `CPML:7`.
- cpml, x_lo: wire-fast `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, x_hi: wire-fast `E[40]; CPML:7`; adi `E[40]; ADI-conductivity:7`.
- cpml, y_lo: wire-fast `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, y_hi: wire-fast `E[36]; CPML:7`; adi `E[36]; ADI-conductivity:7`.
- cpml, z_lo: wire-fast `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, z_hi: wire-fast `E[32]; CPML:7`; adi `E[32]; ADI-conductivity:7`.
- cpml, x_lo: wire-fast `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, x_hi: wire-fast `E[40]; CPML:7`; adi [GPU] `E[40]; ADI-conductivity:7`.
- cpml, y_lo: wire-fast `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, y_hi: wire-fast `E[36]; CPML:7`; adi [GPU] `E[36]; ADI-conductivity:7`.
- cpml, z_lo: wire-fast `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, z_hi: wire-fast `E[32]; CPML:7`; adi [GPU] `E[32]; ADI-conductivity:7`.
- cpml, x_lo: wire-fast [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- cpml, x_hi: wire-fast [GPU] `E[40]; CPML:7`; forward `CPML:7`.
- cpml, y_lo: wire-fast [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- cpml, y_hi: wire-fast [GPU] `E[36]; CPML:7`; forward `CPML:7`.
- cpml, z_lo: wire-fast [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- cpml, z_hi: wire-fast [GPU] `E[32]; CPML:7`; forward `CPML:7`.
- cpml, x_lo: wire-fast [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, x_hi: wire-fast [GPU] `E[40]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, y_lo: wire-fast [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, y_hi: wire-fast [GPU] `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, z_lo: wire-fast [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, z_hi: wire-fast [GPU] `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- cpml, x_lo: wire-fast [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, x_hi: wire-fast [GPU] `E[40]; CPML:7`; distributed `CPML:7`.
- cpml, y_lo: wire-fast [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, y_hi: wire-fast [GPU] `E[36]; CPML:7`; distributed `CPML:7`.
- cpml, z_lo: wire-fast [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, z_hi: wire-fast [GPU] `E[32]; CPML:7`; distributed `CPML:7`.
- cpml, x_lo: wire-fast [GPU] `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, x_hi: wire-fast [GPU] `E[40]; CPML:7`; adi `E[40]; ADI-conductivity:7`.
- cpml, y_lo: wire-fast [GPU] `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, y_hi: wire-fast [GPU] `E[36]; CPML:7`; adi `E[36]; ADI-conductivity:7`.
- cpml, z_lo: wire-fast [GPU] `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, z_hi: wire-fast [GPU] `E[32]; CPML:7`; adi `E[32]; ADI-conductivity:7`.
- cpml, x_lo: wire-fast [GPU] `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, x_hi: wire-fast [GPU] `E[40]; CPML:7`; adi [GPU] `E[40]; ADI-conductivity:7`.
- cpml, y_lo: wire-fast [GPU] `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, y_hi: wire-fast [GPU] `E[36]; CPML:7`; adi [GPU] `E[36]; ADI-conductivity:7`.
- cpml, z_lo: wire-fast [GPU] `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, z_hi: wire-fast [GPU] `E[32]; CPML:7`; adi [GPU] `E[32]; ADI-conductivity:7`.
- cpml, x_lo: forward `CPML:7`; sweep `E[0]; CPML:7`.
- cpml, x_hi: forward `CPML:7`; sweep `E[40]; CPML:7`.
- cpml, y_lo: forward `CPML:7`; sweep `E[0]; CPML:7`.
- cpml, y_hi: forward `CPML:7`; sweep `E[36]; CPML:7`.
- cpml, z_lo: forward `CPML:7`; sweep `E[0]; CPML:7`.
- cpml, z_hi: forward `CPML:7`; sweep `E[32]; CPML:7`.
- cpml, x_lo: forward `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- cpml, x_hi: forward `CPML:7`; sweep [GPU] `E[40]; CPML:7`.
- cpml, y_lo: forward `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- cpml, y_hi: forward `CPML:7`; sweep [GPU] `E[36]; CPML:7`.
- cpml, z_lo: forward `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- cpml, z_hi: forward `CPML:7`; sweep [GPU] `E[32]; CPML:7`.
- cpml, x_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- cpml, x_hi: forward `CPML:7`; nonuniform `E[40]; CPML:7`.
- cpml, y_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- cpml, y_hi: forward `CPML:7`; nonuniform `E[36]; CPML:7`.
- cpml, z_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- cpml, z_hi: forward `CPML:7`; nonuniform `E[32]; CPML:7`.
- cpml, x_lo: forward `CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, x_hi: forward `CPML:7`; adi `E[40]; ADI-conductivity:7`.
- cpml, y_lo: forward `CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, y_hi: forward `CPML:7`; adi `E[36]; ADI-conductivity:7`.
- cpml, z_lo: forward `CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, z_hi: forward `CPML:7`; adi `E[32]; ADI-conductivity:7`.
- cpml, x_lo: forward `CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, x_hi: forward `CPML:7`; adi [GPU] `E[40]; ADI-conductivity:7`.
- cpml, y_lo: forward `CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, y_hi: forward `CPML:7`; adi [GPU] `E[36]; ADI-conductivity:7`.
- cpml, z_lo: forward `CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, z_hi: forward `CPML:7`; adi [GPU] `E[32]; ADI-conductivity:7`.
- cpml, x_lo: forward [GPU] `CPML:7`; sweep `E[0]; CPML:7`.
- cpml, x_hi: forward [GPU] `CPML:7`; sweep `E[40]; CPML:7`.
- cpml, y_lo: forward [GPU] `CPML:7`; sweep `E[0]; CPML:7`.
- cpml, y_hi: forward [GPU] `CPML:7`; sweep `E[36]; CPML:7`.
- cpml, z_lo: forward [GPU] `CPML:7`; sweep `E[0]; CPML:7`.
- cpml, z_hi: forward [GPU] `CPML:7`; sweep `E[32]; CPML:7`.
- cpml, x_lo: forward [GPU] `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- cpml, x_hi: forward [GPU] `CPML:7`; sweep [GPU] `E[40]; CPML:7`.
- cpml, y_lo: forward [GPU] `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- cpml, y_hi: forward [GPU] `CPML:7`; sweep [GPU] `E[36]; CPML:7`.
- cpml, z_lo: forward [GPU] `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- cpml, z_hi: forward [GPU] `CPML:7`; sweep [GPU] `E[32]; CPML:7`.
- cpml, x_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- cpml, x_hi: forward [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- cpml, y_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- cpml, y_hi: forward [GPU] `CPML:7`; nonuniform `E[36]; CPML:7`.
- cpml, z_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- cpml, z_hi: forward [GPU] `CPML:7`; nonuniform `E[32]; CPML:7`.
- cpml, x_lo: forward [GPU] `CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, x_hi: forward [GPU] `CPML:7`; adi `E[40]; ADI-conductivity:7`.
- cpml, y_lo: forward [GPU] `CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, y_hi: forward [GPU] `CPML:7`; adi `E[36]; ADI-conductivity:7`.
- cpml, z_lo: forward [GPU] `CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, z_hi: forward [GPU] `CPML:7`; adi `E[32]; ADI-conductivity:7`.
- cpml, x_lo: forward [GPU] `CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, x_hi: forward [GPU] `CPML:7`; adi [GPU] `E[40]; ADI-conductivity:7`.
- cpml, y_lo: forward [GPU] `CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, y_hi: forward [GPU] `CPML:7`; adi [GPU] `E[36]; ADI-conductivity:7`.
- cpml, z_lo: forward [GPU] `CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, z_hi: forward [GPU] `CPML:7`; adi [GPU] `E[32]; ADI-conductivity:7`.
- cpml, x_lo: sweep `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, x_hi: sweep `E[40]; CPML:7`; distributed `CPML:7`.
- cpml, y_lo: sweep `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, y_hi: sweep `E[36]; CPML:7`; distributed `CPML:7`.
- cpml, z_lo: sweep `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, z_hi: sweep `E[32]; CPML:7`; distributed `CPML:7`.
- cpml, x_lo: sweep `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, x_hi: sweep `E[40]; CPML:7`; adi `E[40]; ADI-conductivity:7`.
- cpml, y_lo: sweep `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, y_hi: sweep `E[36]; CPML:7`; adi `E[36]; ADI-conductivity:7`.
- cpml, z_lo: sweep `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, z_hi: sweep `E[32]; CPML:7`; adi `E[32]; ADI-conductivity:7`.
- cpml, x_lo: sweep `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, x_hi: sweep `E[40]; CPML:7`; adi [GPU] `E[40]; ADI-conductivity:7`.
- cpml, y_lo: sweep `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, y_hi: sweep `E[36]; CPML:7`; adi [GPU] `E[36]; ADI-conductivity:7`.
- cpml, z_lo: sweep `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, z_hi: sweep `E[32]; CPML:7`; adi [GPU] `E[32]; ADI-conductivity:7`.
- cpml, x_lo: sweep [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, x_hi: sweep [GPU] `E[40]; CPML:7`; distributed `CPML:7`.
- cpml, y_lo: sweep [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, y_hi: sweep [GPU] `E[36]; CPML:7`; distributed `CPML:7`.
- cpml, z_lo: sweep [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, z_hi: sweep [GPU] `E[32]; CPML:7`; distributed `CPML:7`.
- cpml, x_lo: sweep [GPU] `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, x_hi: sweep [GPU] `E[40]; CPML:7`; adi `E[40]; ADI-conductivity:7`.
- cpml, y_lo: sweep [GPU] `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, y_hi: sweep [GPU] `E[36]; CPML:7`; adi `E[36]; ADI-conductivity:7`.
- cpml, z_lo: sweep [GPU] `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, z_hi: sweep [GPU] `E[32]; CPML:7`; adi `E[32]; ADI-conductivity:7`.
- cpml, x_lo: sweep [GPU] `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, x_hi: sweep [GPU] `E[40]; CPML:7`; adi [GPU] `E[40]; ADI-conductivity:7`.
- cpml, y_lo: sweep [GPU] `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, y_hi: sweep [GPU] `E[36]; CPML:7`; adi [GPU] `E[36]; ADI-conductivity:7`.
- cpml, z_lo: sweep [GPU] `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, z_hi: sweep [GPU] `E[32]; CPML:7`; adi [GPU] `E[32]; ADI-conductivity:7`.
- cpml, x_lo: nonuniform `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, x_hi: nonuniform `E[40]; CPML:7`; distributed `CPML:7`.
- cpml, y_lo: nonuniform `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, y_hi: nonuniform `E[36]; CPML:7`; distributed `CPML:7`.
- cpml, z_lo: nonuniform `E[0]; CPML:7`; distributed `CPML:7`.
- cpml, z_hi: nonuniform `E[32]; CPML:7`; distributed `CPML:7`.
- cpml, x_lo: nonuniform `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, x_hi: nonuniform `E[40]; CPML:7`; adi `E[40]; ADI-conductivity:7`.
- cpml, y_lo: nonuniform `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, y_hi: nonuniform `E[36]; CPML:7`; adi `E[36]; ADI-conductivity:7`.
- cpml, z_lo: nonuniform `E[0]; CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, z_hi: nonuniform `E[32]; CPML:7`; adi `E[32]; ADI-conductivity:7`.
- cpml, x_lo: nonuniform `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, x_hi: nonuniform `E[40]; CPML:7`; adi [GPU] `E[40]; ADI-conductivity:7`.
- cpml, y_lo: nonuniform `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, y_hi: nonuniform `E[36]; CPML:7`; adi [GPU] `E[36]; ADI-conductivity:7`.
- cpml, z_lo: nonuniform `E[0]; CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, z_hi: nonuniform `E[32]; CPML:7`; adi [GPU] `E[32]; ADI-conductivity:7`.
- cpml, x_lo: distributed `CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, x_hi: distributed `CPML:7`; adi `E[40]; ADI-conductivity:7`.
- cpml, y_lo: distributed `CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, y_hi: distributed `CPML:7`; adi `E[36]; ADI-conductivity:7`.
- cpml, z_lo: distributed `CPML:7`; adi `E[0]; ADI-conductivity:7`.
- cpml, z_hi: distributed `CPML:7`; adi `E[32]; ADI-conductivity:7`.
- cpml, x_lo: distributed `CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, x_hi: distributed `CPML:7`; adi [GPU] `E[40]; ADI-conductivity:7`.
- cpml, y_lo: distributed `CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, y_hi: distributed `CPML:7`; adi [GPU] `E[36]; ADI-conductivity:7`.
- cpml, z_lo: distributed `CPML:7`; adi [GPU] `E[0]; ADI-conductivity:7`.
- cpml, z_hi: distributed `CPML:7`; adi [GPU] `E[32]; ADI-conductivity:7`.
- upml, x_lo: run `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, x_hi: run `E[40]; UPML:8/8`; forward `UPML:8/8`.
- upml, y_lo: run `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, y_hi: run `E[36]; UPML:8/8`; forward `UPML:8/8`.
- upml, z_lo: run `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, z_hi: run `E[32]; UPML:8/8`; forward `UPML:8/8`.
- upml, x_lo: run `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, x_hi: run `E[40]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, y_lo: run `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, y_hi: run `E[36]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, z_lo: run `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, z_hi: run `E[32]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, x_lo: run [GPU] `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, x_hi: run [GPU] `E[40]; UPML:8/8`; forward `UPML:8/8`.
- upml, y_lo: run [GPU] `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, y_hi: run [GPU] `E[36]; UPML:8/8`; forward `UPML:8/8`.
- upml, z_lo: run [GPU] `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, z_hi: run [GPU] `E[32]; UPML:8/8`; forward `UPML:8/8`.
- upml, x_lo: run [GPU] `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, x_hi: run [GPU] `E[40]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, y_lo: run [GPU] `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, y_hi: run [GPU] `E[36]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, z_lo: run [GPU] `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, z_hi: run [GPU] `E[32]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, x_lo: wire-fast `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, x_hi: wire-fast `E[40]; UPML:8/8`; forward `UPML:8/8`.
- upml, y_lo: wire-fast `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, y_hi: wire-fast `E[36]; UPML:8/8`; forward `UPML:8/8`.
- upml, z_lo: wire-fast `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, z_hi: wire-fast `E[32]; UPML:8/8`; forward `UPML:8/8`.
- upml, x_lo: wire-fast `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, x_hi: wire-fast `E[40]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, y_lo: wire-fast `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, y_hi: wire-fast `E[36]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, z_lo: wire-fast `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, z_hi: wire-fast `E[32]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, x_lo: wire-fast [GPU] `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, x_hi: wire-fast [GPU] `E[40]; UPML:8/8`; forward `UPML:8/8`.
- upml, y_lo: wire-fast [GPU] `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, y_hi: wire-fast [GPU] `E[36]; UPML:8/8`; forward `UPML:8/8`.
- upml, z_lo: wire-fast [GPU] `E[0]; UPML:8/8`; forward `UPML:8/8`.
- upml, z_hi: wire-fast [GPU] `E[32]; UPML:8/8`; forward `UPML:8/8`.
- upml, x_lo: wire-fast [GPU] `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, x_hi: wire-fast [GPU] `E[40]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, y_lo: wire-fast [GPU] `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, y_hi: wire-fast [GPU] `E[36]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, z_lo: wire-fast [GPU] `E[0]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, z_hi: wire-fast [GPU] `E[32]; UPML:8/8`; forward [GPU] `UPML:8/8`.
- upml, x_lo: forward `UPML:8/8`; sweep `E[0]; UPML:8/8`.
- upml, x_hi: forward `UPML:8/8`; sweep `E[40]; UPML:8/8`.
- upml, y_lo: forward `UPML:8/8`; sweep `E[0]; UPML:8/8`.
- upml, y_hi: forward `UPML:8/8`; sweep `E[36]; UPML:8/8`.
- upml, z_lo: forward `UPML:8/8`; sweep `E[0]; UPML:8/8`.
- upml, z_hi: forward `UPML:8/8`; sweep `E[32]; UPML:8/8`.
- upml, x_lo: forward `UPML:8/8`; sweep [GPU] `E[0]; UPML:8/8`.
- upml, x_hi: forward `UPML:8/8`; sweep [GPU] `E[40]; UPML:8/8`.
- upml, y_lo: forward `UPML:8/8`; sweep [GPU] `E[0]; UPML:8/8`.
- upml, y_hi: forward `UPML:8/8`; sweep [GPU] `E[36]; UPML:8/8`.
- upml, z_lo: forward `UPML:8/8`; sweep [GPU] `E[0]; UPML:8/8`.
- upml, z_hi: forward `UPML:8/8`; sweep [GPU] `E[32]; UPML:8/8`.
- upml, x_lo: forward [GPU] `UPML:8/8`; sweep `E[0]; UPML:8/8`.
- upml, x_hi: forward [GPU] `UPML:8/8`; sweep `E[40]; UPML:8/8`.
- upml, y_lo: forward [GPU] `UPML:8/8`; sweep `E[0]; UPML:8/8`.
- upml, y_hi: forward [GPU] `UPML:8/8`; sweep `E[36]; UPML:8/8`.
- upml, z_lo: forward [GPU] `UPML:8/8`; sweep `E[0]; UPML:8/8`.
- upml, z_hi: forward [GPU] `UPML:8/8`; sweep `E[32]; UPML:8/8`.
- upml, x_lo: forward [GPU] `UPML:8/8`; sweep [GPU] `E[0]; UPML:8/8`.
- upml, x_hi: forward [GPU] `UPML:8/8`; sweep [GPU] `E[40]; UPML:8/8`.
- upml, y_lo: forward [GPU] `UPML:8/8`; sweep [GPU] `E[0]; UPML:8/8`.
- upml, y_hi: forward [GPU] `UPML:8/8`; sweep [GPU] `E[36]; UPML:8/8`.
- upml, z_lo: forward [GPU] `UPML:8/8`; sweep [GPU] `E[0]; UPML:8/8`.
- upml, z_hi: forward [GPU] `UPML:8/8`; sweep [GPU] `E[32]; UPML:8/8`.
- pmc-pec, x_lo: run `E[0]; H[0]`; run [GPU] `E[0]`.
- pmc-pec, x_hi: run `E[24]; H[23]`; run [GPU] `E[24]`.
- pmc-pec, x_lo: run `E[0]; H[0]`; wire-fast [GPU] `E[0]`.
- pmc-pec, x_hi: run `E[24]; H[23]`; wire-fast [GPU] `E[24]`.
- pmc-pec, x_lo: run `E[0]; H[0]`; forward [GPU] `E[0]`.
- pmc-pec, x_hi: run `E[24]; H[23]`; forward [GPU] `E[24]`.
- pmc-pec, x_lo: run `E[0]; H[0]`; sweep `E[0]`.
- pmc-pec, x_hi: run `E[24]; H[23]`; sweep `E[24]`.
- pmc-pec, x_lo: run `E[0]; H[0]`; sweep [GPU] `E[0]`.
- pmc-pec, x_hi: run `E[24]; H[23]`; sweep [GPU] `E[24]`.
- pmc-pec, x_lo: run `E[0]; H[0]`; subgridded `E[0]`.
- pmc-pec, x_hi: run `E[24]; H[23]`; subgridded `E[24]`.
- pmc-pec, x_lo: run `E[0]; H[0]`; adi `E[0]`.
- pmc-pec, x_hi: run `E[24]; H[23]`; adi `E[24]`.
- pmc-pec, x_lo: run `E[0]; H[0]`; adi [GPU] `E[0]`.
- pmc-pec, x_hi: run `E[24]; H[23]`; adi [GPU] `E[24]`.
- pmc-pec, x_lo: run [GPU] `E[0]`; wire-fast `E[0]; H[0]`.
- pmc-pec, x_hi: run [GPU] `E[24]`; wire-fast `E[24]; H[23]`.
- pmc-pec, x_lo: run [GPU] `E[0]`; forward `E[0]; H[0]`.
- pmc-pec, x_hi: run [GPU] `E[24]`; forward `E[24]; H[23]`.
- pmc-pec, x_lo: run [GPU] `E[0]`; nonuniform `E[0]; H[0]`.
- pmc-pec, x_hi: run [GPU] `E[24]`; nonuniform `E[24]; H[23]`.
- pmc-pec, x_lo: run [GPU] `E[0]`; distributed `E[0]; H[0]`.
- pmc-pec, x_hi: run [GPU] `E[24]`; distributed `E[24]; H[23]`.
- pmc-pec, x_lo: wire-fast `E[0]; H[0]`; wire-fast [GPU] `E[0]`.
- pmc-pec, x_hi: wire-fast `E[24]; H[23]`; wire-fast [GPU] `E[24]`.
- pmc-pec, x_lo: wire-fast `E[0]; H[0]`; forward [GPU] `E[0]`.
- pmc-pec, x_hi: wire-fast `E[24]; H[23]`; forward [GPU] `E[24]`.
- pmc-pec, x_lo: wire-fast `E[0]; H[0]`; sweep `E[0]`.
- pmc-pec, x_hi: wire-fast `E[24]; H[23]`; sweep `E[24]`.
- pmc-pec, x_lo: wire-fast `E[0]; H[0]`; sweep [GPU] `E[0]`.
- pmc-pec, x_hi: wire-fast `E[24]; H[23]`; sweep [GPU] `E[24]`.
- pmc-pec, x_lo: wire-fast `E[0]; H[0]`; subgridded `E[0]`.
- pmc-pec, x_hi: wire-fast `E[24]; H[23]`; subgridded `E[24]`.
- pmc-pec, x_lo: wire-fast `E[0]; H[0]`; adi `E[0]`.
- pmc-pec, x_hi: wire-fast `E[24]; H[23]`; adi `E[24]`.
- pmc-pec, x_lo: wire-fast `E[0]; H[0]`; adi [GPU] `E[0]`.
- pmc-pec, x_hi: wire-fast `E[24]; H[23]`; adi [GPU] `E[24]`.
- pmc-pec, x_lo: wire-fast [GPU] `E[0]`; forward `E[0]; H[0]`.
- pmc-pec, x_hi: wire-fast [GPU] `E[24]`; forward `E[24]; H[23]`.
- pmc-pec, x_lo: wire-fast [GPU] `E[0]`; nonuniform `E[0]; H[0]`.
- pmc-pec, x_hi: wire-fast [GPU] `E[24]`; nonuniform `E[24]; H[23]`.
- pmc-pec, x_lo: wire-fast [GPU] `E[0]`; distributed `E[0]; H[0]`.
- pmc-pec, x_hi: wire-fast [GPU] `E[24]`; distributed `E[24]; H[23]`.
- pmc-pec, x_lo: forward `E[0]; H[0]`; forward [GPU] `E[0]`.
- pmc-pec, x_hi: forward `E[24]; H[23]`; forward [GPU] `E[24]`.
- pmc-pec, x_lo: forward `E[0]; H[0]`; sweep `E[0]`.
- pmc-pec, x_hi: forward `E[24]; H[23]`; sweep `E[24]`.
- pmc-pec, x_lo: forward `E[0]; H[0]`; sweep [GPU] `E[0]`.
- pmc-pec, x_hi: forward `E[24]; H[23]`; sweep [GPU] `E[24]`.
- pmc-pec, x_lo: forward `E[0]; H[0]`; subgridded `E[0]`.
- pmc-pec, x_hi: forward `E[24]; H[23]`; subgridded `E[24]`.
- pmc-pec, x_lo: forward `E[0]; H[0]`; adi `E[0]`.
- pmc-pec, x_hi: forward `E[24]; H[23]`; adi `E[24]`.
- pmc-pec, x_lo: forward `E[0]; H[0]`; adi [GPU] `E[0]`.
- pmc-pec, x_hi: forward `E[24]; H[23]`; adi [GPU] `E[24]`.
- pmc-pec, x_lo: forward [GPU] `E[0]`; nonuniform `E[0]; H[0]`.
- pmc-pec, x_hi: forward [GPU] `E[24]`; nonuniform `E[24]; H[23]`.
- pmc-pec, x_lo: forward [GPU] `E[0]`; distributed `E[0]; H[0]`.
- pmc-pec, x_hi: forward [GPU] `E[24]`; distributed `E[24]; H[23]`.
- pmc-pec, x_lo: sweep `E[0]`; nonuniform `E[0]; H[0]`.
- pmc-pec, x_hi: sweep `E[24]`; nonuniform `E[24]; H[23]`.
- pmc-pec, x_lo: sweep `E[0]`; distributed `E[0]; H[0]`.
- pmc-pec, x_hi: sweep `E[24]`; distributed `E[24]; H[23]`.
- pmc-pec, x_lo: sweep [GPU] `E[0]`; nonuniform `E[0]; H[0]`.
- pmc-pec, x_hi: sweep [GPU] `E[24]`; nonuniform `E[24]; H[23]`.
- pmc-pec, x_lo: sweep [GPU] `E[0]`; distributed `E[0]; H[0]`.
- pmc-pec, x_hi: sweep [GPU] `E[24]`; distributed `E[24]; H[23]`.
- pmc-pec, x_lo: nonuniform `E[0]; H[0]`; subgridded `E[0]`.
- pmc-pec, x_hi: nonuniform `E[24]; H[23]`; subgridded `E[24]`.
- pmc-pec, x_lo: nonuniform `E[0]; H[0]`; adi `E[0]`.
- pmc-pec, x_hi: nonuniform `E[24]; H[23]`; adi `E[24]`.
- pmc-pec, x_lo: nonuniform `E[0]; H[0]`; adi [GPU] `E[0]`.
- pmc-pec, x_hi: nonuniform `E[24]; H[23]`; adi [GPU] `E[24]`.
- pmc-pec, x_lo: subgridded `E[0]`; distributed `E[0]; H[0]`.
- pmc-pec, x_hi: subgridded `E[24]`; distributed `E[24]; H[23]`.
- pmc-pec, x_lo: distributed `E[0]; H[0]`; adi `E[0]`.
- pmc-pec, x_hi: distributed `E[24]; H[23]`; adi `E[24]`.
- pmc-pec, x_lo: distributed `E[0]; H[0]`; adi [GPU] `E[0]`.
- pmc-pec, x_hi: distributed `E[24]; H[23]`; adi [GPU] `E[24]`.
- pmc-cpml, x_lo: run `E[0]; H[0]`; forward `H[0]`.
- pmc-cpml, x_hi: run `E[24]; H[23]`; forward `H[23]`.
- pmc-cpml, y_lo: run `E[0]; CPML:7`; forward `CPML:7`.
- pmc-cpml, y_hi: run `E[36]; CPML:7`; forward `CPML:7`.
- pmc-cpml, z_lo: run `E[0]; CPML:7`; forward `CPML:7`.
- pmc-cpml, z_hi: run `E[32]; CPML:7`; forward `CPML:7`.
- pmc-cpml, x_lo: run `E[0]; H[0]`; forward [GPU] `H[0]`.
- pmc-cpml, x_hi: run `E[24]; H[23]`; forward [GPU] `H[23]`.
- pmc-cpml, y_lo: run `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, y_hi: run `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, z_lo: run `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, z_hi: run `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, x_lo: run `E[0]; H[0]`; sweep `E[0]`.
- pmc-cpml, x_hi: run `E[24]; H[23]`; sweep `E[24]`.
- pmc-cpml, x_lo: run `E[0]; H[0]`; sweep [GPU] `E[0]`.
- pmc-cpml, x_hi: run `E[24]; H[23]`; sweep [GPU] `E[24]`.
- pmc-cpml, x_lo: run `E[0]; H[0]`; distributed `H[0]; CPML:7`.
- pmc-cpml, x_hi: run `E[24]; H[23]`; distributed `H[23]; CPML:7`.
- pmc-cpml, y_lo: run `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, y_hi: run `E[36]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_lo: run `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_hi: run `E[32]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, x_lo: run [GPU] `E[0]; H[0]`; forward `H[0]`.
- pmc-cpml, x_hi: run [GPU] `E[24]; H[23]`; forward `H[23]`.
- pmc-cpml, y_lo: run [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- pmc-cpml, y_hi: run [GPU] `E[36]; CPML:7`; forward `CPML:7`.
- pmc-cpml, z_lo: run [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- pmc-cpml, z_hi: run [GPU] `E[32]; CPML:7`; forward `CPML:7`.
- pmc-cpml, x_lo: run [GPU] `E[0]; H[0]`; forward [GPU] `H[0]`.
- pmc-cpml, x_hi: run [GPU] `E[24]; H[23]`; forward [GPU] `H[23]`.
- pmc-cpml, y_lo: run [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, y_hi: run [GPU] `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, z_lo: run [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, z_hi: run [GPU] `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, x_lo: run [GPU] `E[0]; H[0]`; sweep `E[0]`.
- pmc-cpml, x_hi: run [GPU] `E[24]; H[23]`; sweep `E[24]`.
- pmc-cpml, x_lo: run [GPU] `E[0]; H[0]`; sweep [GPU] `E[0]`.
- pmc-cpml, x_hi: run [GPU] `E[24]; H[23]`; sweep [GPU] `E[24]`.
- pmc-cpml, x_lo: run [GPU] `E[0]; H[0]`; distributed `H[0]; CPML:7`.
- pmc-cpml, x_hi: run [GPU] `E[24]; H[23]`; distributed `H[23]; CPML:7`.
- pmc-cpml, y_lo: run [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, y_hi: run [GPU] `E[36]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_lo: run [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_hi: run [GPU] `E[32]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, x_lo: wire-fast `E[0]; H[0]`; forward `H[0]`.
- pmc-cpml, x_hi: wire-fast `E[24]; H[23]`; forward `H[23]`.
- pmc-cpml, y_lo: wire-fast `E[0]; CPML:7`; forward `CPML:7`.
- pmc-cpml, y_hi: wire-fast `E[36]; CPML:7`; forward `CPML:7`.
- pmc-cpml, z_lo: wire-fast `E[0]; CPML:7`; forward `CPML:7`.
- pmc-cpml, z_hi: wire-fast `E[32]; CPML:7`; forward `CPML:7`.
- pmc-cpml, x_lo: wire-fast `E[0]; H[0]`; forward [GPU] `H[0]`.
- pmc-cpml, x_hi: wire-fast `E[24]; H[23]`; forward [GPU] `H[23]`.
- pmc-cpml, y_lo: wire-fast `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, y_hi: wire-fast `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, z_lo: wire-fast `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, z_hi: wire-fast `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, x_lo: wire-fast `E[0]; H[0]`; sweep `E[0]`.
- pmc-cpml, x_hi: wire-fast `E[24]; H[23]`; sweep `E[24]`.
- pmc-cpml, x_lo: wire-fast `E[0]; H[0]`; sweep [GPU] `E[0]`.
- pmc-cpml, x_hi: wire-fast `E[24]; H[23]`; sweep [GPU] `E[24]`.
- pmc-cpml, x_lo: wire-fast `E[0]; H[0]`; distributed `H[0]; CPML:7`.
- pmc-cpml, x_hi: wire-fast `E[24]; H[23]`; distributed `H[23]; CPML:7`.
- pmc-cpml, y_lo: wire-fast `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, y_hi: wire-fast `E[36]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_lo: wire-fast `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_hi: wire-fast `E[32]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, x_lo: wire-fast [GPU] `E[0]; H[0]`; forward `H[0]`.
- pmc-cpml, x_hi: wire-fast [GPU] `E[24]; H[23]`; forward `H[23]`.
- pmc-cpml, y_lo: wire-fast [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- pmc-cpml, y_hi: wire-fast [GPU] `E[36]; CPML:7`; forward `CPML:7`.
- pmc-cpml, z_lo: wire-fast [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- pmc-cpml, z_hi: wire-fast [GPU] `E[32]; CPML:7`; forward `CPML:7`.
- pmc-cpml, x_lo: wire-fast [GPU] `E[0]; H[0]`; forward [GPU] `H[0]`.
- pmc-cpml, x_hi: wire-fast [GPU] `E[24]; H[23]`; forward [GPU] `H[23]`.
- pmc-cpml, y_lo: wire-fast [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, y_hi: wire-fast [GPU] `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, z_lo: wire-fast [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, z_hi: wire-fast [GPU] `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- pmc-cpml, x_lo: wire-fast [GPU] `E[0]; H[0]`; sweep `E[0]`.
- pmc-cpml, x_hi: wire-fast [GPU] `E[24]; H[23]`; sweep `E[24]`.
- pmc-cpml, x_lo: wire-fast [GPU] `E[0]; H[0]`; sweep [GPU] `E[0]`.
- pmc-cpml, x_hi: wire-fast [GPU] `E[24]; H[23]`; sweep [GPU] `E[24]`.
- pmc-cpml, x_lo: wire-fast [GPU] `E[0]; H[0]`; distributed `H[0]; CPML:7`.
- pmc-cpml, x_hi: wire-fast [GPU] `E[24]; H[23]`; distributed `H[23]; CPML:7`.
- pmc-cpml, y_lo: wire-fast [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, y_hi: wire-fast [GPU] `E[36]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_lo: wire-fast [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_hi: wire-fast [GPU] `E[32]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, x_lo: forward `H[0]`; sweep `E[0]`.
- pmc-cpml, x_hi: forward `H[23]`; sweep `E[24]`.
- pmc-cpml, y_lo: forward `CPML:7`; sweep `E[0]; CPML:7`.
- pmc-cpml, y_hi: forward `CPML:7`; sweep `E[36]; CPML:7`.
- pmc-cpml, z_lo: forward `CPML:7`; sweep `E[0]; CPML:7`.
- pmc-cpml, z_hi: forward `CPML:7`; sweep `E[32]; CPML:7`.
- pmc-cpml, x_lo: forward `H[0]`; sweep [GPU] `E[0]`.
- pmc-cpml, x_hi: forward `H[23]`; sweep [GPU] `E[24]`.
- pmc-cpml, y_lo: forward `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- pmc-cpml, y_hi: forward `CPML:7`; sweep [GPU] `E[36]; CPML:7`.
- pmc-cpml, z_lo: forward `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- pmc-cpml, z_hi: forward `CPML:7`; sweep [GPU] `E[32]; CPML:7`.
- pmc-cpml, x_lo: forward `H[0]`; nonuniform `E[0]; H[0]`.
- pmc-cpml, x_hi: forward `H[23]`; nonuniform `E[24]; H[23]`.
- pmc-cpml, y_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- pmc-cpml, y_hi: forward `CPML:7`; nonuniform `E[36]; CPML:7`.
- pmc-cpml, z_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- pmc-cpml, z_hi: forward `CPML:7`; nonuniform `E[32]; CPML:7`.
- pmc-cpml, x_lo: forward `H[0]`; distributed `H[0]; CPML:7`.
- pmc-cpml, x_hi: forward `H[23]`; distributed `H[23]; CPML:7`.
- pmc-cpml, x_lo: forward [GPU] `H[0]`; sweep `E[0]`.
- pmc-cpml, x_hi: forward [GPU] `H[23]`; sweep `E[24]`.
- pmc-cpml, y_lo: forward [GPU] `CPML:7`; sweep `E[0]; CPML:7`.
- pmc-cpml, y_hi: forward [GPU] `CPML:7`; sweep `E[36]; CPML:7`.
- pmc-cpml, z_lo: forward [GPU] `CPML:7`; sweep `E[0]; CPML:7`.
- pmc-cpml, z_hi: forward [GPU] `CPML:7`; sweep `E[32]; CPML:7`.
- pmc-cpml, x_lo: forward [GPU] `H[0]`; sweep [GPU] `E[0]`.
- pmc-cpml, x_hi: forward [GPU] `H[23]`; sweep [GPU] `E[24]`.
- pmc-cpml, y_lo: forward [GPU] `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- pmc-cpml, y_hi: forward [GPU] `CPML:7`; sweep [GPU] `E[36]; CPML:7`.
- pmc-cpml, z_lo: forward [GPU] `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- pmc-cpml, z_hi: forward [GPU] `CPML:7`; sweep [GPU] `E[32]; CPML:7`.
- pmc-cpml, x_lo: forward [GPU] `H[0]`; nonuniform `E[0]; H[0]`.
- pmc-cpml, x_hi: forward [GPU] `H[23]`; nonuniform `E[24]; H[23]`.
- pmc-cpml, y_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- pmc-cpml, y_hi: forward [GPU] `CPML:7`; nonuniform `E[36]; CPML:7`.
- pmc-cpml, z_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- pmc-cpml, z_hi: forward [GPU] `CPML:7`; nonuniform `E[32]; CPML:7`.
- pmc-cpml, x_lo: forward [GPU] `H[0]`; distributed `H[0]; CPML:7`.
- pmc-cpml, x_hi: forward [GPU] `H[23]`; distributed `H[23]; CPML:7`.
- pmc-cpml, x_lo: sweep `E[0]`; nonuniform `E[0]; H[0]`.
- pmc-cpml, x_hi: sweep `E[24]`; nonuniform `E[24]; H[23]`.
- pmc-cpml, x_lo: sweep `E[0]`; distributed `H[0]; CPML:7`.
- pmc-cpml, x_hi: sweep `E[24]`; distributed `H[23]; CPML:7`.
- pmc-cpml, y_lo: sweep `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, y_hi: sweep `E[36]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_lo: sweep `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_hi: sweep `E[32]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, x_lo: sweep [GPU] `E[0]`; nonuniform `E[0]; H[0]`.
- pmc-cpml, x_hi: sweep [GPU] `E[24]`; nonuniform `E[24]; H[23]`.
- pmc-cpml, x_lo: sweep [GPU] `E[0]`; distributed `H[0]; CPML:7`.
- pmc-cpml, x_hi: sweep [GPU] `E[24]`; distributed `H[23]; CPML:7`.
- pmc-cpml, y_lo: sweep [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, y_hi: sweep [GPU] `E[36]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_lo: sweep [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_hi: sweep [GPU] `E[32]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, x_lo: nonuniform `E[0]; H[0]`; distributed `H[0]; CPML:7`.
- pmc-cpml, x_hi: nonuniform `E[24]; H[23]`; distributed `H[23]; CPML:7`.
- pmc-cpml, y_lo: nonuniform `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, y_hi: nonuniform `E[36]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_lo: nonuniform `E[0]; CPML:7`; distributed `CPML:7`.
- pmc-cpml, z_hi: nonuniform `E[32]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_lo: run `E[0]; CPML:7`; forward `CPML:7`.
- pec-zlo, x_hi: run `E[40]; CPML:7`; forward `CPML:7`.
- pec-zlo, y_lo: run `E[0]; CPML:7`; forward `CPML:7`.
- pec-zlo, y_hi: run `E[36]; CPML:7`; forward `CPML:7`.
- pec-zlo, z_hi: run `E[24]; CPML:7`; forward `CPML:7`.
- pec-zlo, x_lo: run `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, x_hi: run `E[40]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, y_lo: run `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, y_hi: run `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, z_hi: run `E[24]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, x_lo: run `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_hi: run `E[40]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_lo: run `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_hi: run `E[36]; CPML:7`; distributed `CPML:7`.
- pec-zlo, z_lo: run `E[0]`; distributed `CPML:7`.
- pec-zlo, z_hi: run `E[24]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_lo: run [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- pec-zlo, x_hi: run [GPU] `E[40]; CPML:7`; forward `CPML:7`.
- pec-zlo, y_lo: run [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- pec-zlo, y_hi: run [GPU] `E[36]; CPML:7`; forward `CPML:7`.
- pec-zlo, z_hi: run [GPU] `E[24]; CPML:7`; forward `CPML:7`.
- pec-zlo, x_lo: run [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, x_hi: run [GPU] `E[40]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, y_lo: run [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, y_hi: run [GPU] `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, z_hi: run [GPU] `E[24]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, x_lo: run [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_hi: run [GPU] `E[40]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_lo: run [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_hi: run [GPU] `E[36]; CPML:7`; distributed `CPML:7`.
- pec-zlo, z_lo: run [GPU] `E[0]`; distributed `CPML:7`.
- pec-zlo, z_hi: run [GPU] `E[24]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_lo: wire-fast `E[0]; CPML:7`; forward `CPML:7`.
- pec-zlo, x_hi: wire-fast `E[40]; CPML:7`; forward `CPML:7`.
- pec-zlo, y_lo: wire-fast `E[0]; CPML:7`; forward `CPML:7`.
- pec-zlo, y_hi: wire-fast `E[36]; CPML:7`; forward `CPML:7`.
- pec-zlo, z_hi: wire-fast `E[24]; CPML:7`; forward `CPML:7`.
- pec-zlo, x_lo: wire-fast `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, x_hi: wire-fast `E[40]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, y_lo: wire-fast `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, y_hi: wire-fast `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, z_hi: wire-fast `E[24]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, x_lo: wire-fast `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_hi: wire-fast `E[40]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_lo: wire-fast `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_hi: wire-fast `E[36]; CPML:7`; distributed `CPML:7`.
- pec-zlo, z_lo: wire-fast `E[0]`; distributed `CPML:7`.
- pec-zlo, z_hi: wire-fast `E[24]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_lo: wire-fast [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- pec-zlo, x_hi: wire-fast [GPU] `E[40]; CPML:7`; forward `CPML:7`.
- pec-zlo, y_lo: wire-fast [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- pec-zlo, y_hi: wire-fast [GPU] `E[36]; CPML:7`; forward `CPML:7`.
- pec-zlo, z_hi: wire-fast [GPU] `E[24]; CPML:7`; forward `CPML:7`.
- pec-zlo, x_lo: wire-fast [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, x_hi: wire-fast [GPU] `E[40]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, y_lo: wire-fast [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, y_hi: wire-fast [GPU] `E[36]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, z_hi: wire-fast [GPU] `E[24]; CPML:7`; forward [GPU] `CPML:7`.
- pec-zlo, x_lo: wire-fast [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_hi: wire-fast [GPU] `E[40]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_lo: wire-fast [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_hi: wire-fast [GPU] `E[36]; CPML:7`; distributed `CPML:7`.
- pec-zlo, z_lo: wire-fast [GPU] `E[0]`; distributed `CPML:7`.
- pec-zlo, z_hi: wire-fast [GPU] `E[24]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_lo: forward `CPML:7`; sweep `E[0]; CPML:7`.
- pec-zlo, x_hi: forward `CPML:7`; sweep `E[40]; CPML:7`.
- pec-zlo, y_lo: forward `CPML:7`; sweep `E[0]; CPML:7`.
- pec-zlo, y_hi: forward `CPML:7`; sweep `E[36]; CPML:7`.
- pec-zlo, z_hi: forward `CPML:7`; sweep `E[24]; CPML:7`.
- pec-zlo, x_lo: forward `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- pec-zlo, x_hi: forward `CPML:7`; sweep [GPU] `E[40]; CPML:7`.
- pec-zlo, y_lo: forward `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- pec-zlo, y_hi: forward `CPML:7`; sweep [GPU] `E[36]; CPML:7`.
- pec-zlo, z_hi: forward `CPML:7`; sweep [GPU] `E[24]; CPML:7`.
- pec-zlo, x_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- pec-zlo, x_hi: forward `CPML:7`; nonuniform `E[40]; CPML:7`.
- pec-zlo, y_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- pec-zlo, y_hi: forward `CPML:7`; nonuniform `E[36]; CPML:7`.
- pec-zlo, z_hi: forward `CPML:7`; nonuniform `E[24]; CPML:7`.
- pec-zlo, z_lo: forward `E[0]`; distributed `CPML:7`.
- pec-zlo, x_lo: forward [GPU] `CPML:7`; sweep `E[0]; CPML:7`.
- pec-zlo, x_hi: forward [GPU] `CPML:7`; sweep `E[40]; CPML:7`.
- pec-zlo, y_lo: forward [GPU] `CPML:7`; sweep `E[0]; CPML:7`.
- pec-zlo, y_hi: forward [GPU] `CPML:7`; sweep `E[36]; CPML:7`.
- pec-zlo, z_hi: forward [GPU] `CPML:7`; sweep `E[24]; CPML:7`.
- pec-zlo, x_lo: forward [GPU] `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- pec-zlo, x_hi: forward [GPU] `CPML:7`; sweep [GPU] `E[40]; CPML:7`.
- pec-zlo, y_lo: forward [GPU] `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- pec-zlo, y_hi: forward [GPU] `CPML:7`; sweep [GPU] `E[36]; CPML:7`.
- pec-zlo, z_hi: forward [GPU] `CPML:7`; sweep [GPU] `E[24]; CPML:7`.
- pec-zlo, x_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- pec-zlo, x_hi: forward [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- pec-zlo, y_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- pec-zlo, y_hi: forward [GPU] `CPML:7`; nonuniform `E[36]; CPML:7`.
- pec-zlo, z_hi: forward [GPU] `CPML:7`; nonuniform `E[24]; CPML:7`.
- pec-zlo, z_lo: forward [GPU] `E[0]`; distributed `CPML:7`.
- pec-zlo, x_lo: sweep `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_hi: sweep `E[40]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_lo: sweep `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_hi: sweep `E[36]; CPML:7`; distributed `CPML:7`.
- pec-zlo, z_lo: sweep `E[0]`; distributed `CPML:7`.
- pec-zlo, z_hi: sweep `E[24]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_lo: sweep [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_hi: sweep [GPU] `E[40]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_lo: sweep [GPU] `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_hi: sweep [GPU] `E[36]; CPML:7`; distributed `CPML:7`.
- pec-zlo, z_lo: sweep [GPU] `E[0]`; distributed `CPML:7`.
- pec-zlo, z_hi: sweep [GPU] `E[24]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_lo: nonuniform `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, x_hi: nonuniform `E[40]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_lo: nonuniform `E[0]; CPML:7`; distributed `CPML:7`.
- pec-zlo, y_hi: nonuniform `E[36]; CPML:7`; distributed `CPML:7`.
- pec-zlo, z_lo: nonuniform `E[0]`; distributed `CPML:7`.
- pec-zlo, z_hi: nonuniform `E[24]; CPML:7`; distributed `CPML:7`.
- periodic-xy, z_lo: run `E[0]; CPML:7`; forward `CPML:7`.
- periodic-xy, z_hi: run `E[32]; CPML:7`; forward `CPML:7`.
- periodic-xy, z_lo: run `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- periodic-xy, z_hi: run `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- periodic-xy, x_lo: run `wrap`; nonuniform `E[0]`.
- periodic-xy, x_hi: run `wrap`; nonuniform `E[24]`.
- periodic-xy, y_lo: run `wrap`; nonuniform `E[0]`.
- periodic-xy, y_hi: run `wrap`; nonuniform `E[20]`.
- periodic-xy, z_lo: run [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- periodic-xy, z_hi: run [GPU] `E[32]; CPML:7`; forward `CPML:7`.
- periodic-xy, z_lo: run [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- periodic-xy, z_hi: run [GPU] `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- periodic-xy, x_lo: run [GPU] `wrap`; nonuniform `E[0]`.
- periodic-xy, x_hi: run [GPU] `wrap`; nonuniform `E[24]`.
- periodic-xy, y_lo: run [GPU] `wrap`; nonuniform `E[0]`.
- periodic-xy, y_hi: run [GPU] `wrap`; nonuniform `E[20]`.
- periodic-xy, z_lo: forward `CPML:7`; sweep `E[0]; CPML:7`.
- periodic-xy, z_hi: forward `CPML:7`; sweep `E[32]; CPML:7`.
- periodic-xy, z_lo: forward `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- periodic-xy, z_hi: forward `CPML:7`; sweep [GPU] `E[32]; CPML:7`.
- periodic-xy, x_lo: forward `wrap`; nonuniform `E[0]`.
- periodic-xy, x_hi: forward `wrap`; nonuniform `E[24]`.
- periodic-xy, y_lo: forward `wrap`; nonuniform `E[0]`.
- periodic-xy, y_hi: forward `wrap`; nonuniform `E[20]`.
- periodic-xy, z_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- periodic-xy, z_hi: forward `CPML:7`; nonuniform `E[32]; CPML:7`.
- periodic-xy, z_lo: forward [GPU] `CPML:7`; sweep `E[0]; CPML:7`.
- periodic-xy, z_hi: forward [GPU] `CPML:7`; sweep `E[32]; CPML:7`.
- periodic-xy, z_lo: forward [GPU] `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- periodic-xy, z_hi: forward [GPU] `CPML:7`; sweep [GPU] `E[32]; CPML:7`.
- periodic-xy, x_lo: forward [GPU] `wrap`; nonuniform `E[0]`.
- periodic-xy, x_hi: forward [GPU] `wrap`; nonuniform `E[24]`.
- periodic-xy, y_lo: forward [GPU] `wrap`; nonuniform `E[0]`.
- periodic-xy, y_hi: forward [GPU] `wrap`; nonuniform `E[20]`.
- periodic-xy, z_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- periodic-xy, z_hi: forward [GPU] `CPML:7`; nonuniform `E[32]; CPML:7`.
- periodic-xy, x_lo: sweep `wrap`; nonuniform `E[0]`.
- periodic-xy, x_hi: sweep `wrap`; nonuniform `E[24]`.
- periodic-xy, y_lo: sweep `wrap`; nonuniform `E[0]`.
- periodic-xy, y_hi: sweep `wrap`; nonuniform `E[20]`.
- periodic-xy, x_lo: sweep [GPU] `wrap`; nonuniform `E[0]`.
- periodic-xy, x_hi: sweep [GPU] `wrap`; nonuniform `E[24]`.
- periodic-xy, y_lo: sweep [GPU] `wrap`; nonuniform `E[0]`.
- periodic-xy, y_hi: sweep [GPU] `wrap`; nonuniform `E[20]`.
- tfsf, x_lo: run `E[0]; CPML:7`; forward `CPML:7`.
- tfsf, x_hi: run `E[40]; CPML:7`; forward `CPML:7`.
- tfsf, x_lo: run `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- tfsf, x_hi: run `E[40]; CPML:7`; forward [GPU] `CPML:7`.
- tfsf, y_lo: run `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, y_hi: run `wrap`; nonuniform `E[36]; CPML:7`.
- tfsf, z_lo: run `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, z_hi: run `wrap`; nonuniform `E[32]; CPML:7`.
- tfsf, x_lo: run [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- tfsf, x_hi: run [GPU] `E[40]; CPML:7`; forward `CPML:7`.
- tfsf, x_lo: run [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- tfsf, x_hi: run [GPU] `E[40]; CPML:7`; forward [GPU] `CPML:7`.
- tfsf, y_lo: run [GPU] `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, y_hi: run [GPU] `wrap`; nonuniform `E[36]; CPML:7`.
- tfsf, z_lo: run [GPU] `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, z_hi: run [GPU] `wrap`; nonuniform `E[32]; CPML:7`.
- tfsf, x_lo: forward `CPML:7`; sweep `E[0]; CPML:7`.
- tfsf, x_hi: forward `CPML:7`; sweep `E[40]; CPML:7`.
- tfsf, x_lo: forward `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- tfsf, x_hi: forward `CPML:7`; sweep [GPU] `E[40]; CPML:7`.
- tfsf, x_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- tfsf, x_hi: forward `CPML:7`; nonuniform `E[40]; CPML:7`.
- tfsf, y_lo: forward `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, y_hi: forward `wrap`; nonuniform `E[36]; CPML:7`.
- tfsf, z_lo: forward `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, z_hi: forward `wrap`; nonuniform `E[32]; CPML:7`.
- tfsf, x_lo: forward `CPML:7`; distributed `E[0]; CPML:7`.
- tfsf, x_hi: forward `CPML:7`; distributed `E[40]; CPML:7`.
- tfsf, x_lo: forward [GPU] `CPML:7`; sweep `E[0]; CPML:7`.
- tfsf, x_hi: forward [GPU] `CPML:7`; sweep `E[40]; CPML:7`.
- tfsf, x_lo: forward [GPU] `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- tfsf, x_hi: forward [GPU] `CPML:7`; sweep [GPU] `E[40]; CPML:7`.
- tfsf, x_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- tfsf, x_hi: forward [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- tfsf, y_lo: forward [GPU] `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, y_hi: forward [GPU] `wrap`; nonuniform `E[36]; CPML:7`.
- tfsf, z_lo: forward [GPU] `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, z_hi: forward [GPU] `wrap`; nonuniform `E[32]; CPML:7`.
- tfsf, x_lo: forward [GPU] `CPML:7`; distributed `E[0]; CPML:7`.
- tfsf, x_hi: forward [GPU] `CPML:7`; distributed `E[40]; CPML:7`.
- tfsf, y_lo: sweep `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, y_hi: sweep `wrap`; nonuniform `E[36]; CPML:7`.
- tfsf, z_lo: sweep `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, z_hi: sweep `wrap`; nonuniform `E[32]; CPML:7`.
- tfsf, y_lo: sweep [GPU] `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, y_hi: sweep [GPU] `wrap`; nonuniform `E[36]; CPML:7`.
- tfsf, z_lo: sweep [GPU] `wrap`; nonuniform `E[0]; CPML:7`.
- tfsf, z_hi: sweep [GPU] `wrap`; nonuniform `E[32]; CPML:7`.
- tfsf, y_lo: nonuniform `E[0]; CPML:7`; distributed `wrap`.
- tfsf, y_hi: nonuniform `E[36]; CPML:7`; distributed `wrap`.
- tfsf, z_lo: nonuniform `E[0]; CPML:7`; distributed `wrap`.
- tfsf, z_hi: nonuniform `E[32]; CPML:7`; distributed `wrap`.
- waveguide-cpml, x_lo: run `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, x_hi: run `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-cpml, y_lo: run `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, y_hi: run `E[20]`; nonuniform `E[36]; CPML:7`.
- waveguide-cpml, z_lo: run `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, z_hi: run `E[16]`; nonuniform `E[32]; CPML:7`.
- waveguide-cpml, x_lo: run [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, x_hi: run [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-cpml, y_lo: run [GPU] `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, y_hi: run [GPU] `E[20]`; nonuniform `E[36]; CPML:7`.
- waveguide-cpml, z_lo: run [GPU] `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, z_hi: run [GPU] `E[16]`; nonuniform `E[32]; CPML:7`.
- waveguide-cpml, x_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, x_hi: forward `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-cpml, y_lo: forward `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, y_hi: forward `E[20]`; nonuniform `E[36]; CPML:7`.
- waveguide-cpml, z_lo: forward `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, z_hi: forward `E[16]`; nonuniform `E[32]; CPML:7`.
- waveguide-cpml, x_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, x_hi: forward [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-cpml, y_lo: forward [GPU] `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, y_hi: forward [GPU] `E[20]`; nonuniform `E[36]; CPML:7`.
- waveguide-cpml, z_lo: forward [GPU] `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, z_hi: forward [GPU] `E[16]`; nonuniform `E[32]; CPML:7`.
- waveguide-cpml, x_lo: sweep `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, x_hi: sweep `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-cpml, y_lo: sweep `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, y_hi: sweep `E[20]`; nonuniform `E[36]; CPML:7`.
- waveguide-cpml, z_lo: sweep `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, z_hi: sweep `E[16]`; nonuniform `E[32]; CPML:7`.
- waveguide-cpml, x_lo: sweep [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, x_hi: sweep [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-cpml, y_lo: sweep [GPU] `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, y_hi: sweep [GPU] `E[20]`; nonuniform `E[36]; CPML:7`.
- waveguide-cpml, z_lo: sweep [GPU] `E[0]`; nonuniform `E[0]; CPML:7`.
- waveguide-cpml, z_hi: sweep [GPU] `E[16]`; nonuniform `E[32]; CPML:7`.
- waveguide-cpml, x_lo: nonuniform `E[0]; CPML:7`; distributed `CPML:7`.
- waveguide-cpml, x_hi: nonuniform `E[40]; CPML:7`; distributed `CPML:7`.
- waveguide-cpml, y_lo: nonuniform `E[0]; CPML:7`; distributed `E[0]`.
- waveguide-cpml, y_hi: nonuniform `E[36]; CPML:7`; distributed `E[20]`.
- waveguide-cpml, z_lo: nonuniform `E[0]; CPML:7`; distributed `E[0]`.
- waveguide-cpml, z_hi: nonuniform `E[32]; CPML:7`; distributed `E[16]`.
- waveguide-pmc, x_lo: run `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pmc, x_hi: run `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pmc, x_lo: run [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pmc, x_hi: run [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pmc, x_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pmc, x_hi: forward `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pmc, x_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pmc, x_hi: forward [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pmc, x_lo: sweep `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pmc, x_hi: sweep `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pmc, x_lo: sweep [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pmc, x_hi: sweep [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pmc, x_lo: nonuniform `E[0]; CPML:7`; distributed `CPML:7`.
- waveguide-pmc, x_hi: nonuniform `E[40]; CPML:7`; distributed `CPML:7`.
- waveguide-pec, x_lo: run `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pec, x_hi: run `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pec, x_lo: run [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pec, x_hi: run [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pec, x_lo: forward `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pec, x_hi: forward `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pec, x_lo: forward [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pec, x_hi: forward [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pec, x_lo: sweep `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pec, x_hi: sweep `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pec, x_lo: sweep [GPU] `CPML:7`; nonuniform `E[0]; CPML:7`.
- waveguide-pec, x_hi: sweep [GPU] `CPML:7`; nonuniform `E[40]; CPML:7`.
- waveguide-pec, x_lo: nonuniform `E[0]; CPML:7`; distributed `CPML:7`.
- waveguide-pec, x_hi: nonuniform `E[40]; CPML:7`; distributed `CPML:7`.
- floquet, z_lo: run `E[0]; CPML:7`; forward `CPML:7`.
- floquet, z_hi: run `E[32]; CPML:7`; forward `CPML:7`.
- floquet, z_lo: run `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- floquet, z_hi: run `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- floquet, z_lo: run [GPU] `E[0]; CPML:7`; forward `CPML:7`.
- floquet, z_hi: run [GPU] `E[32]; CPML:7`; forward `CPML:7`.
- floquet, z_lo: run [GPU] `E[0]; CPML:7`; forward [GPU] `CPML:7`.
- floquet, z_hi: run [GPU] `E[32]; CPML:7`; forward [GPU] `CPML:7`.
- floquet, z_lo: forward `CPML:7`; sweep `E[0]; CPML:7`.
- floquet, z_hi: forward `CPML:7`; sweep `E[32]; CPML:7`.
- floquet, z_lo: forward `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- floquet, z_hi: forward `CPML:7`; sweep [GPU] `E[32]; CPML:7`.
- floquet, z_lo: forward [GPU] `CPML:7`; sweep `E[0]; CPML:7`.
- floquet, z_hi: forward [GPU] `CPML:7`; sweep `E[32]; CPML:7`.
- floquet, z_lo: forward [GPU] `CPML:7`; sweep [GPU] `E[0]; CPML:7`.
- floquet, z_hi: forward [GPU] `CPML:7`; sweep [GPU] `E[32]; CPML:7`.

## Declared-versus-observed cells

The list includes extra backing walls, simultaneous PEC/PMC calls, and unobserved declared operators; interpretation is reserved.

- cpml, run, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, run, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- cpml, run, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, run, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- cpml, run, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, run, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- cpml, wire-fast, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, wire-fast, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- cpml, wire-fast, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, wire-fast, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- cpml, wire-fast, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, wire-fast, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- cpml, sweep, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, sweep, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- cpml, sweep, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, sweep, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- cpml, sweep, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, sweep, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- cpml, nonuniform, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, nonuniform, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- cpml, nonuniform, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, nonuniform, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- cpml, nonuniform, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, nonuniform, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- cpml, adi, x_lo: declared `cpml`; observed `E[0]; ADI-conductivity:7`.
- cpml, adi, x_hi: declared `cpml`; observed `E[40]; ADI-conductivity:7`.
- cpml, adi, y_lo: declared `cpml`; observed `E[0]; ADI-conductivity:7`.
- cpml, adi, y_hi: declared `cpml`; observed `E[36]; ADI-conductivity:7`.
- cpml, adi, z_lo: declared `cpml`; observed `E[0]; ADI-conductivity:7`.
- cpml, adi, z_hi: declared `cpml`; observed `E[32]; ADI-conductivity:7`.
- upml, run, x_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, run, x_hi: declared `upml`; observed `E[40]; UPML:8/8`.
- upml, run, y_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, run, y_hi: declared `upml`; observed `E[36]; UPML:8/8`.
- upml, run, z_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, run, z_hi: declared `upml`; observed `E[32]; UPML:8/8`.
- upml, wire-fast, x_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, wire-fast, x_hi: declared `upml`; observed `E[40]; UPML:8/8`.
- upml, wire-fast, y_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, wire-fast, y_hi: declared `upml`; observed `E[36]; UPML:8/8`.
- upml, wire-fast, z_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, wire-fast, z_hi: declared `upml`; observed `E[32]; UPML:8/8`.
- upml, sweep, x_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, sweep, x_hi: declared `upml`; observed `E[40]; UPML:8/8`.
- upml, sweep, y_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, sweep, y_hi: declared `upml`; observed `E[36]; UPML:8/8`.
- upml, sweep, z_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, sweep, z_hi: declared `upml`; observed `E[32]; UPML:8/8`.
- pmc-pec, run, x_lo: declared `pmc`; observed `E[0]; H[0]`.
- pmc-pec, run, x_hi: declared `pmc`; observed `E[24]; H[23]`.
- pmc-pec, wire-fast, x_lo: declared `pmc`; observed `E[0]; H[0]`.
- pmc-pec, wire-fast, x_hi: declared `pmc`; observed `E[24]; H[23]`.
- pmc-pec, forward, x_lo: declared `pmc`; observed `E[0]; H[0]`.
- pmc-pec, forward, x_hi: declared `pmc`; observed `E[24]; H[23]`.
- pmc-pec, sweep, x_lo: declared `pmc`; observed `E[0]`.
- pmc-pec, sweep, x_hi: declared `pmc`; observed `E[24]`.
- pmc-pec, nonuniform, x_lo: declared `pmc`; observed `E[0]; H[0]`.
- pmc-pec, nonuniform, x_hi: declared `pmc`; observed `E[24]; H[23]`.
- pmc-pec, subgridded, x_lo: declared `pmc`; observed `E[0]`.
- pmc-pec, subgridded, x_hi: declared `pmc`; observed `E[24]`.
- pmc-pec, distributed, x_lo: declared `pmc`; observed `E[0]; H[0]`.
- pmc-pec, distributed, x_hi: declared `pmc`; observed `E[24]; H[23]`.
- pmc-pec, adi, x_lo: declared `pmc`; observed `E[0]`.
- pmc-pec, adi, x_hi: declared `pmc`; observed `E[24]`.
- pmc-cpml, run, x_lo: declared `pmc`; observed `E[0]; H[0]`.
- pmc-cpml, run, x_hi: declared `pmc`; observed `E[24]; H[23]`.
- pmc-cpml, run, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, run, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pmc-cpml, run, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, run, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- pmc-cpml, wire-fast, x_lo: declared `pmc`; observed `E[0]; H[0]`.
- pmc-cpml, wire-fast, x_hi: declared `pmc`; observed `E[24]; H[23]`.
- pmc-cpml, wire-fast, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, wire-fast, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pmc-cpml, wire-fast, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, wire-fast, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- pmc-cpml, sweep, x_lo: declared `pmc`; observed `E[0]`.
- pmc-cpml, sweep, x_hi: declared `pmc`; observed `E[24]`.
- pmc-cpml, sweep, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, sweep, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pmc-cpml, sweep, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, sweep, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- pmc-cpml, nonuniform, x_lo: declared `pmc`; observed `E[0]; H[0]`.
- pmc-cpml, nonuniform, x_hi: declared `pmc`; observed `E[24]; H[23]`.
- pmc-cpml, nonuniform, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, nonuniform, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pmc-cpml, nonuniform, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, nonuniform, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- pmc-cpml, distributed, x_lo: declared `pmc`; observed `H[0]; CPML:7`.
- pmc-cpml, distributed, x_hi: declared `pmc`; observed `H[23]; CPML:7`.
- pec-zlo, run, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, run, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- pec-zlo, run, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, run, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pec-zlo, run, z_hi: declared `cpml`; observed `E[24]; CPML:7`.
- pec-zlo, wire-fast, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, wire-fast, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- pec-zlo, wire-fast, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, wire-fast, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pec-zlo, wire-fast, z_hi: declared `cpml`; observed `E[24]; CPML:7`.
- pec-zlo, sweep, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, sweep, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- pec-zlo, sweep, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, sweep, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pec-zlo, sweep, z_hi: declared `cpml`; observed `E[24]; CPML:7`.
- pec-zlo, nonuniform, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, nonuniform, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- pec-zlo, nonuniform, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, nonuniform, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pec-zlo, nonuniform, z_hi: declared `cpml`; observed `E[24]; CPML:7`.
- pec-zlo, distributed, z_lo: declared `pec`; observed `CPML:7`.
- periodic-xy, run, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- periodic-xy, run, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- periodic-xy, sweep, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- periodic-xy, sweep, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- periodic-xy, nonuniform, x_lo: declared `periodic`; observed `E[0]`.
- periodic-xy, nonuniform, x_hi: declared `periodic`; observed `E[24]`.
- periodic-xy, nonuniform, y_lo: declared `periodic`; observed `E[0]`.
- periodic-xy, nonuniform, y_hi: declared `periodic`; observed `E[20]`.
- periodic-xy, nonuniform, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- periodic-xy, nonuniform, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- tfsf, run, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- tfsf, run, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- tfsf, run, y_lo: declared `cpml`; observed `wrap`.
- tfsf, run, y_hi: declared `cpml`; observed `wrap`.
- tfsf, run, z_lo: declared `cpml`; observed `wrap`.
- tfsf, run, z_hi: declared `cpml`; observed `wrap`.
- tfsf, forward, y_lo: declared `cpml`; observed `wrap`.
- tfsf, forward, y_hi: declared `cpml`; observed `wrap`.
- tfsf, forward, z_lo: declared `cpml`; observed `wrap`.
- tfsf, forward, z_hi: declared `cpml`; observed `wrap`.
- tfsf, sweep, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- tfsf, sweep, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- tfsf, sweep, y_lo: declared `cpml`; observed `wrap`.
- tfsf, sweep, y_hi: declared `cpml`; observed `wrap`.
- tfsf, sweep, z_lo: declared `cpml`; observed `wrap`.
- tfsf, sweep, z_hi: declared `cpml`; observed `wrap`.
- tfsf, nonuniform, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- tfsf, nonuniform, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- tfsf, nonuniform, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- tfsf, nonuniform, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- tfsf, nonuniform, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- tfsf, nonuniform, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- tfsf, distributed, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- tfsf, distributed, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- tfsf, distributed, y_lo: declared `cpml`; observed `wrap`.
- tfsf, distributed, y_hi: declared `cpml`; observed `wrap`.
- tfsf, distributed, z_lo: declared `cpml`; observed `wrap`.
- tfsf, distributed, z_hi: declared `cpml`; observed `wrap`.
- waveguide-cpml, run, y_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, run, y_hi: declared `cpml`; observed `E[20]`.
- waveguide-cpml, run, z_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, run, z_hi: declared `cpml`; observed `E[16]`.
- waveguide-cpml, forward, y_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, forward, y_hi: declared `cpml`; observed `E[20]`.
- waveguide-cpml, forward, z_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, forward, z_hi: declared `cpml`; observed `E[16]`.
- waveguide-cpml, sweep, y_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, sweep, y_hi: declared `cpml`; observed `E[20]`.
- waveguide-cpml, sweep, z_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, sweep, z_hi: declared `cpml`; observed `E[16]`.
- waveguide-cpml, nonuniform, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- waveguide-cpml, nonuniform, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- waveguide-cpml, nonuniform, y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- waveguide-cpml, nonuniform, y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- waveguide-cpml, nonuniform, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- waveguide-cpml, nonuniform, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- waveguide-cpml, distributed, y_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, distributed, y_hi: declared `cpml`; observed `E[20]`.
- waveguide-cpml, distributed, z_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, distributed, z_hi: declared `cpml`; observed `E[16]`.
- waveguide-pmc, run, y_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, run, y_hi: declared `pmc`; observed `E[20]; H[19]`.
- waveguide-pmc, run, z_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, run, z_hi: declared `pmc`; observed `E[16]; H[15]`.
- waveguide-pmc, forward, y_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, forward, y_hi: declared `pmc`; observed `E[20]; H[19]`.
- waveguide-pmc, forward, z_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, forward, z_hi: declared `pmc`; observed `E[16]; H[15]`.
- waveguide-pmc, sweep, y_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, sweep, y_hi: declared `pmc`; observed `E[20]; H[19]`.
- waveguide-pmc, sweep, z_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, sweep, z_hi: declared `pmc`; observed `E[16]; H[15]`.
- waveguide-pmc, nonuniform, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- waveguide-pmc, nonuniform, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- waveguide-pmc, nonuniform, y_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, nonuniform, y_hi: declared `pmc`; observed `E[20]; H[19]`.
- waveguide-pmc, nonuniform, z_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, nonuniform, z_hi: declared `pmc`; observed `E[16]; H[15]`.
- waveguide-pmc, distributed, y_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, distributed, y_hi: declared `pmc`; observed `E[20]; H[19]`.
- waveguide-pmc, distributed, z_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, distributed, z_hi: declared `pmc`; observed `E[16]; H[15]`.
- waveguide-pec, nonuniform, x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- waveguide-pec, nonuniform, x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- floquet, run, x_lo: declared `cpml`; observed `wrap`.
- floquet, run, x_hi: declared `cpml`; observed `wrap`.
- floquet, run, y_lo: declared `cpml`; observed `wrap`.
- floquet, run, y_hi: declared `cpml`; observed `wrap`.
- floquet, run, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- floquet, run, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- floquet, forward, x_lo: declared `cpml`; observed `wrap`.
- floquet, forward, x_hi: declared `cpml`; observed `wrap`.
- floquet, forward, y_lo: declared `cpml`; observed `wrap`.
- floquet, forward, y_hi: declared `cpml`; observed `wrap`.
- floquet, sweep, x_lo: declared `cpml`; observed `wrap`.
- floquet, sweep, x_hi: declared `cpml`; observed `wrap`.
- floquet, sweep, y_lo: declared `cpml`; observed `wrap`.
- floquet, sweep, y_hi: declared `cpml`; observed `wrap`.
- floquet, sweep, z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- floquet, sweep, z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- cpml, run [GPU], x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, run [GPU], x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- cpml, run [GPU], y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, run [GPU], y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- cpml, run [GPU], z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, run [GPU], z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- cpml, wire-fast [GPU], x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, wire-fast [GPU], x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- cpml, wire-fast [GPU], y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, wire-fast [GPU], y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- cpml, wire-fast [GPU], z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, wire-fast [GPU], z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- cpml, sweep [GPU], x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, sweep [GPU], x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- cpml, sweep [GPU], y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, sweep [GPU], y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- cpml, sweep [GPU], z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- cpml, sweep [GPU], z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- cpml, adi [GPU], x_lo: declared `cpml`; observed `E[0]; ADI-conductivity:7`.
- cpml, adi [GPU], x_hi: declared `cpml`; observed `E[40]; ADI-conductivity:7`.
- cpml, adi [GPU], y_lo: declared `cpml`; observed `E[0]; ADI-conductivity:7`.
- cpml, adi [GPU], y_hi: declared `cpml`; observed `E[36]; ADI-conductivity:7`.
- cpml, adi [GPU], z_lo: declared `cpml`; observed `E[0]; ADI-conductivity:7`.
- cpml, adi [GPU], z_hi: declared `cpml`; observed `E[32]; ADI-conductivity:7`.
- upml, run [GPU], x_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, run [GPU], x_hi: declared `upml`; observed `E[40]; UPML:8/8`.
- upml, run [GPU], y_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, run [GPU], y_hi: declared `upml`; observed `E[36]; UPML:8/8`.
- upml, run [GPU], z_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, run [GPU], z_hi: declared `upml`; observed `E[32]; UPML:8/8`.
- upml, wire-fast [GPU], x_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, wire-fast [GPU], x_hi: declared `upml`; observed `E[40]; UPML:8/8`.
- upml, wire-fast [GPU], y_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, wire-fast [GPU], y_hi: declared `upml`; observed `E[36]; UPML:8/8`.
- upml, wire-fast [GPU], z_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, wire-fast [GPU], z_hi: declared `upml`; observed `E[32]; UPML:8/8`.
- upml, sweep [GPU], x_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, sweep [GPU], x_hi: declared `upml`; observed `E[40]; UPML:8/8`.
- upml, sweep [GPU], y_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, sweep [GPU], y_hi: declared `upml`; observed `E[36]; UPML:8/8`.
- upml, sweep [GPU], z_lo: declared `upml`; observed `E[0]; UPML:8/8`.
- upml, sweep [GPU], z_hi: declared `upml`; observed `E[32]; UPML:8/8`.
- pmc-pec, run [GPU], x_lo: declared `pmc`; observed `E[0]`.
- pmc-pec, run [GPU], x_hi: declared `pmc`; observed `E[24]`.
- pmc-pec, wire-fast [GPU], x_lo: declared `pmc`; observed `E[0]`.
- pmc-pec, wire-fast [GPU], x_hi: declared `pmc`; observed `E[24]`.
- pmc-pec, forward [GPU], x_lo: declared `pmc`; observed `E[0]`.
- pmc-pec, forward [GPU], x_hi: declared `pmc`; observed `E[24]`.
- pmc-pec, sweep [GPU], x_lo: declared `pmc`; observed `E[0]`.
- pmc-pec, sweep [GPU], x_hi: declared `pmc`; observed `E[24]`.
- pmc-pec, adi [GPU], x_lo: declared `pmc`; observed `E[0]`.
- pmc-pec, adi [GPU], x_hi: declared `pmc`; observed `E[24]`.
- pmc-cpml, run [GPU], x_lo: declared `pmc`; observed `E[0]; H[0]`.
- pmc-cpml, run [GPU], x_hi: declared `pmc`; observed `E[24]; H[23]`.
- pmc-cpml, run [GPU], y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, run [GPU], y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pmc-cpml, run [GPU], z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, run [GPU], z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- pmc-cpml, wire-fast [GPU], x_lo: declared `pmc`; observed `E[0]; H[0]`.
- pmc-cpml, wire-fast [GPU], x_hi: declared `pmc`; observed `E[24]; H[23]`.
- pmc-cpml, wire-fast [GPU], y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, wire-fast [GPU], y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pmc-cpml, wire-fast [GPU], z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, wire-fast [GPU], z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- pmc-cpml, sweep [GPU], x_lo: declared `pmc`; observed `E[0]`.
- pmc-cpml, sweep [GPU], x_hi: declared `pmc`; observed `E[24]`.
- pmc-cpml, sweep [GPU], y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, sweep [GPU], y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pmc-cpml, sweep [GPU], z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pmc-cpml, sweep [GPU], z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- pec-zlo, run [GPU], x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, run [GPU], x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- pec-zlo, run [GPU], y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, run [GPU], y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pec-zlo, run [GPU], z_hi: declared `cpml`; observed `E[24]; CPML:7`.
- pec-zlo, wire-fast [GPU], x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, wire-fast [GPU], x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- pec-zlo, wire-fast [GPU], y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, wire-fast [GPU], y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pec-zlo, wire-fast [GPU], z_hi: declared `cpml`; observed `E[24]; CPML:7`.
- pec-zlo, sweep [GPU], x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, sweep [GPU], x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- pec-zlo, sweep [GPU], y_lo: declared `cpml`; observed `E[0]; CPML:7`.
- pec-zlo, sweep [GPU], y_hi: declared `cpml`; observed `E[36]; CPML:7`.
- pec-zlo, sweep [GPU], z_hi: declared `cpml`; observed `E[24]; CPML:7`.
- periodic-xy, run [GPU], z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- periodic-xy, run [GPU], z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- periodic-xy, sweep [GPU], z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- periodic-xy, sweep [GPU], z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- tfsf, run [GPU], x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- tfsf, run [GPU], x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- tfsf, run [GPU], y_lo: declared `cpml`; observed `wrap`.
- tfsf, run [GPU], y_hi: declared `cpml`; observed `wrap`.
- tfsf, run [GPU], z_lo: declared `cpml`; observed `wrap`.
- tfsf, run [GPU], z_hi: declared `cpml`; observed `wrap`.
- tfsf, forward [GPU], y_lo: declared `cpml`; observed `wrap`.
- tfsf, forward [GPU], y_hi: declared `cpml`; observed `wrap`.
- tfsf, forward [GPU], z_lo: declared `cpml`; observed `wrap`.
- tfsf, forward [GPU], z_hi: declared `cpml`; observed `wrap`.
- tfsf, sweep [GPU], x_lo: declared `cpml`; observed `E[0]; CPML:7`.
- tfsf, sweep [GPU], x_hi: declared `cpml`; observed `E[40]; CPML:7`.
- tfsf, sweep [GPU], y_lo: declared `cpml`; observed `wrap`.
- tfsf, sweep [GPU], y_hi: declared `cpml`; observed `wrap`.
- tfsf, sweep [GPU], z_lo: declared `cpml`; observed `wrap`.
- tfsf, sweep [GPU], z_hi: declared `cpml`; observed `wrap`.
- waveguide-cpml, run [GPU], y_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, run [GPU], y_hi: declared `cpml`; observed `E[20]`.
- waveguide-cpml, run [GPU], z_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, run [GPU], z_hi: declared `cpml`; observed `E[16]`.
- waveguide-cpml, forward [GPU], y_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, forward [GPU], y_hi: declared `cpml`; observed `E[20]`.
- waveguide-cpml, forward [GPU], z_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, forward [GPU], z_hi: declared `cpml`; observed `E[16]`.
- waveguide-cpml, sweep [GPU], y_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, sweep [GPU], y_hi: declared `cpml`; observed `E[20]`.
- waveguide-cpml, sweep [GPU], z_lo: declared `cpml`; observed `E[0]`.
- waveguide-cpml, sweep [GPU], z_hi: declared `cpml`; observed `E[16]`.
- waveguide-pmc, run [GPU], y_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, run [GPU], y_hi: declared `pmc`; observed `E[20]; H[19]`.
- waveguide-pmc, run [GPU], z_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, run [GPU], z_hi: declared `pmc`; observed `E[16]; H[15]`.
- waveguide-pmc, forward [GPU], y_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, forward [GPU], y_hi: declared `pmc`; observed `E[20]; H[19]`.
- waveguide-pmc, forward [GPU], z_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, forward [GPU], z_hi: declared `pmc`; observed `E[16]; H[15]`.
- waveguide-pmc, sweep [GPU], y_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, sweep [GPU], y_hi: declared `pmc`; observed `E[20]; H[19]`.
- waveguide-pmc, sweep [GPU], z_lo: declared `pmc`; observed `E[0]; H[0]`.
- waveguide-pmc, sweep [GPU], z_hi: declared `pmc`; observed `E[16]; H[15]`.
- floquet, run [GPU], x_lo: declared `cpml`; observed `wrap`.
- floquet, run [GPU], x_hi: declared `cpml`; observed `wrap`.
- floquet, run [GPU], y_lo: declared `cpml`; observed `wrap`.
- floquet, run [GPU], y_hi: declared `cpml`; observed `wrap`.
- floquet, run [GPU], z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- floquet, run [GPU], z_hi: declared `cpml`; observed `E[32]; CPML:7`.
- floquet, forward [GPU], x_lo: declared `cpml`; observed `wrap`.
- floquet, forward [GPU], x_hi: declared `cpml`; observed `wrap`.
- floquet, forward [GPU], y_lo: declared `cpml`; observed `wrap`.
- floquet, forward [GPU], y_hi: declared `cpml`; observed `wrap`.
- floquet, sweep [GPU], x_lo: declared `cpml`; observed `wrap`.
- floquet, sweep [GPU], x_hi: declared `cpml`; observed `wrap`.
- floquet, sweep [GPU], y_lo: declared `cpml`; observed `wrap`.
- floquet, sweep [GPU], y_hi: declared `cpml`; observed `wrap`.
- floquet, sweep [GPU], z_lo: declared `cpml`; observed `E[0]; CPML:7`.
- floquet, sweep [GPU], z_hi: declared `cpml`; observed `E[32]; CPML:7`.

Conclusion: leader fills.
