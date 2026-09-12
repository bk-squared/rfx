"""Static CV06b source/load support audit; no FDTD or field advancement."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from dataclasses import replace
import hashlib
import importlib.util
import json
import numpy as np
import jax
import rfx
import rfx.sources.msl_port as msl_module
from rfx.boundaries.pec import realized_wall_planes
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.sources.msl_port import (compute_msl_mode_profile, make_msl_port_sources,
                                  msl_cross_section_span, msl_port_from_entry,
                                  setup_msl_port)
from rfx.sources.sources import GaussianPulse

assert Path(rfx.__file__).resolve().is_relative_to(ROOT)
fixture_path = ROOT / 'scripts/diagnostics/msl_probe_clearance_bias.py'
spec = importlib.util.spec_from_file_location('support_fixture', fixture_path)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
cv, arms, inputs = fixture.prepare_inputs()
sim = arms['clean']
rz = fixture._realized(sim)
grid = rz.grid
nodes = coords_from_uniform_grid(grid)
edges = tuple(np.asarray(a, dtype=bool) for a in rz.edge_masks)
materials = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
eps = np.asarray(materials.eps_r)
sigma_before = np.asarray(materials.sigma)
results = []
for p, entry in enumerate(sim._msl_ports):
    port = msl_port_from_entry(entry)
    span = msl_cross_section_span(grid, port)
    feed = span['i_feed']
    ktop = span['n_hi']
    eps_sub = float(eps[feed, span['w_centre'], (span['n_lo'] + ktop) // 2])
    mode = compute_msl_mode_profile(grid, port, eps_sub)
    profile = np.asarray(mode['ez_profile'], dtype=float)
    cells = [tuple(c) for c in mode['cell_indices']]
    accepted = []
    pairs = []
    for cell in cells:
        jl = cell[mode['width_idx']] - mode['j_grid_lo']
        kl = cell[mode['normal_idx']] - mode['k_grid_lo']
        if 0 <= kl < mode['n_z_sub'] and 0 <= jl < profile.shape[0] and profile[jl, kl] != 0:
            accepted.append(cell)
            pairs.append((jl, kl))
    assert len(accepted) == len(set(accepted))
    actual_pairs = set(pairs)
    omitted_pairs = [(j, k) for j in range(profile.shape[0]) for k in range(profile.shape[1])
                     if profile[j, k] != 0 and (j, k) not in actual_pairs]
    # Uniform interior grid: Ez Joule/control volume is dx**3 for every cell.
    norm_full = float(np.sum(profile**2) * grid.dx**3)
    norm_active = float(sum(profile[j, k]**2 * grid.dx**3 for j, k in pairs))
    columns = []
    for j in sorted({cell[1] for cell in accepted}):
        col = [cell for cell in accepted if cell[1] == j]
        endpoint_k = max(cell[2] for cell in col) + 1
        planes = realized_wall_planes(edges, 2, ij=(feed, j), periodic=sim._periodic_flags())
        attached = endpoint_k in planes
        js = j - mode['j_grid_lo']
        energy_weight = sum(profile[js, cell[2] - mode['k_grid_lo']]**2 * grid.dx**3 for cell in col)
        columns.append(dict(j=j, y_m=float(nodes.y[j]), cell_count=len(col), k_indices=[cell[2] for cell in col], profile=profile[js].tolist(), upper_node_k=endpoint_k,
                            upper_node_z_m=float(nodes.z[endpoint_k]), upper_node_has_incident_pec=bool(attached),
                            incident_trace_ex_forward=bool(edges[0][feed,j,endpoint_k]),
                            incident_trace_ex_backward=bool(edges[0][feed-1,j,endpoint_k]),
                            norm_fraction=energy_weight/norm_full,
                            below_endpoint_eps=float(eps[feed,j,endpoint_k-1]),
                            above_endpoint_eps=float(eps[feed,j,endpoint_k])))
    loaded = setup_msl_port(grid, port, materials, mode_profile=mode)
    delta = np.asarray(loaded.sigma) - sigma_before
    stamped_cells = set(map(tuple, np.argwhere(delta != 0)))
    assert stamped_cells == set(accepted)
    driven = replace(port, excitation=(entry.waveform or GaussianPulse(f0=sim._freq_max/2, bandwidth=.8)))
    sources = make_msl_port_sources(grid, driven, loaded, 1, mode_profile=mode)
    source_cells = {(s.i,s.j,s.k) for s in sources}
    assert source_cells == set(accepted) and len(sources) == len(accepted)
    uniform = setup_msl_port(grid, port, materials)
    uniform_cells = set(map(tuple, np.argwhere(np.asarray(uniform.sigma) != sigma_before)))
    assert uniform_cells == set(map(tuple, span['cells']))
    uniform_sources = make_msl_port_sources(grid, driven, uniform, 1)
    assert {(s.i,s.j,s.k) for s in uniform_sources} == uniform_cells
    uniform_columns = sorted({c[1] for c in uniform_cells})
    uniform_outside = [j for j in uniform_columns
                       if ktop not in realized_wall_planes(edges,2,ij=(feed,j),periodic=sim._periodic_flags())]
    outside = [c for c in columns if not c['upper_node_has_incident_pec']]
    pads_lo = [getattr(grid,f'pad_{a}_lo') for a in 'xyz']
    pads_hi = [getattr(grid,f'pad_{a}_hi') for a in 'xyz']
    def in_pad(cell):
        return any(not (grid.interior[a].start <= cell[a] < grid.interior[a].stop) for a in range(3))
    pad_cells = [cell for cell in accepted if in_pad(cell)]
    pad_norm = sum(profile[cell[1]-mode['j_grid_lo'],cell[2]-mode['k_grid_lo']]**2*grid.dx**3 for cell in pad_cells)
    results.append(dict(port=p, direction=entry.direction, feed_i=feed, feed_x_m=float(nodes.x[feed]),
                        field_lane='float32', eps_r_sub=eps_sub, dx_m=float(grid.dx),
                        laplace_profile_shape=list(profile.shape), source_count=len(sources), load_cell_count=len(stamped_cells),
                        column_count=len(columns), attached_column_count=len(columns)-len(outside),
                        unattached_column_count=len(outside), unattached_cell_count=sum(c['cell_count'] for c in outside),
                        unattached_profile_weighted_squared_norm_fraction=sum(c['norm_fraction'] for c in outside),
                        profile_norm_full=norm_full, profile_norm_actual_loop=norm_active,
                        normalization_nonzero_entries_excluded=omitted_pairs,
                        raw_profile_nonzero_count=int(np.count_nonzero(profile)),
                        cell_ranges_inclusive=[[min(c[a] for c in accepted),max(c[a] for c in accepted)] for a in range(3)],
                        pad_lo=pads_lo,pad_hi=pads_hi,grid_shape=list(grid.shape),
                        grid_interior_half_open=[[s.start,s.stop] for s in grid.interior],
                        padded_source_load_cells=pad_cells,padded_cell_count=len(pad_cells),padded_norm_fraction=float(pad_norm/norm_full),
                        original_pec_owned_source_ez_cell_count=sum(bool(edges[2][cell]) for cell in accepted),
                        sigma_values=np.unique(delta[delta!=0]).astype(float).tolist(),
                        uniform=dict(columns=uniform_columns, source_count=len(uniform_sources), load_cell_count=len(uniform_cells),
                                     unattached_columns=uniform_outside, unattached_norm_fraction=0.0,
                                     padded_cell_count=sum(in_pad(cell) for cell in uniform_cells)), columns=columns))

report = dict(scope='Static actual clean CV06b builder, canonical PEC incident edges, real Laplace/source/load helpers; no field evolution.',
              rfx_import_path=rfx.__file__, msl_import_path=msl_module.__file__,
              case_sha256=hashlib.sha256((ROOT/'validation/crossval/06b_msl_notch_filter_uniform.py').read_bytes()).hexdigest(),
              msl_source_sha256=hashlib.sha256((ROOT/'rfx/sources/msl_port.py').read_bytes()).hexdigest(),
              jax_version=jax.__version__, x64_enabled=jax.config.x64_enabled,
              ground='z_lo domain PEC; outside-trace endpoints are unmetalized substrate/air interface, not a conductor contact.',
              norm_definition='sum(ez_profile**2 * dual_prop * dual_width * primal_normal); uniform interior weights are dx**3.',
              interpretation='Impressed fringe support is not automatically invalid or a galvanic/pure-mode certification.', results=results)
out=ROOT/'.git/issue726-source-support-audit.json'
text=json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
out.write_text(text+'\n')
print(text)
