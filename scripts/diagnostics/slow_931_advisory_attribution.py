"""Controlled PEC-realization attribution on the repaired short fixture.

Run serially, with PYTHONPATH=$PWD JAX_PLATFORMS=cpu. No alternate checkout.
Hypothesis: the old (2.25, 3] interval witnesses the retired node sigma fold.
Falsifier: restoring ONLY that fold to the current extractor fails to restore
the interval. The node_volume bridge separates sampling from edge ownership.
This is an investigative intervention, not a supported alternate PEC contract.
The legacy fold is documented at a3e4dba4^:rfx/api/_sparams.py:3012-3014.
The PI authorized fixture repair after the original attribution. This script
now imports that repaired fixture; older artifacts retain their source hashes.
The interval remains unchanged. The current arm qualifies the repaired planes;
historical arms remain diagnostic interventions, never supported semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import warnings
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import rfx
import rfx.api._sparams as api_sparams
import rfx.simulation as solver
from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.sources.waveguide_port import extract_waveguide_port_waves, waveguide_plane_positions


def fixture_builder():
    from tests._pec_short_advisory_fixture import build
    source = Path('tests/_pec_short_advisory_fixture.py').read_text()
    return build, hashlib.sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['current', 'node_volume', 'legacy_sigma'], required=True)
    parser.add_argument('--fine', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('.validation-931-closures'))
    args = parser.parse_args()
    assert Path(rfx.__file__).resolve().is_relative_to(Path.cwd()), rfx.__file__
    build, test_hash = fixture_builder()
    frequencies, dx, cpml, periods = ((np.linspace(5e9, 7e9, 6), 1e-3, 10, 40)
                                    if args.fine else (np.linspace(4e9, 6e9, 6), 2e-3, 8, 30))
    sim = build(frequencies, dx, cpml)
    grid = sim._build_grid()
    sheets, wires = [], []
    materials, _, _, current_cells, *_ = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
    assert not sheets and not wires and len(sim._geometry) == 1
    nodes = coords_from_uniform_grid(grid)
    box = sim._geometry[0].shape
    legacy_cells = box.mask(grid)  # actual pre-#931 uniform assembly spelling
    current_edges = realized_pec_edge_masks(current_cells, periodic=sim._periodic_flags())
    node_edges = realized_pec_edge_masks(legacy_cells, periodic=sim._periodic_flags())
    def census(cells, edges):
        indexes = np.where(np.any(np.asarray(cells), axis=(1, 2)))[0]
        return dict(cell_count=int(np.sum(cells)), occupied_x_indices=indexes.tolist(),
                    occupied_x_lower_nodes_m=np.asarray(nodes.x)[indexes].tolist(),
                    edge_counts=[int(np.sum(e)) for e in edges],
                    tangential_wall_x_m=np.asarray(nodes.x)[realized_wall_planes(edges, 0)].tolist())
    result = dict(arm=args.arm, fine=args.fine, source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  rfx_import=rfx.__file__, fixture_sha256=test_hash, grid_shape=list(grid.shape),
                  dx=dx, dt=float(grid.dt), num_periods=periods,
                  current=census(current_cells, current_edges), node_volume=census(legacy_cells, node_edges),
                  legacy_sigma=dict(cell_count=int(np.sum(legacy_cells)),
                                    damped_x_nodes_m=np.asarray(nodes.x)[np.where(np.any(np.asarray(legacy_cells), axis=(1, 2)))[0]].tolist()),
                  frequencies_hz=frequencies.tolist(),
                  fixture='931-fixture-repair-node-aligned-short',
                  declared_short_faces_m=[float(box.corner_lo[0]), float(box.corner_hi[0])])
    result['port_planes_m'] = [
        {k: float(v) for k, v in waveguide_plane_positions(
            sim._build_waveguide_port_config(port, grid, frequencies, 1)).items()}
        for port in sim._waveguide_ports
    ]
    for planes in result['port_planes_m']:
        assert (max(planes.values()) < box.corner_lo[0]
                or min(planes.values()) > box.corner_hi[0]), planes
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / ('advisory_' + args.arm + ('_fine' if args.fine else ''))
    if args.build_only:
        print(json.dumps(result, indent=2))
        output.with_suffix('.build.json').write_text(json.dumps(result, indent=2) + '\n')
        return
    original_extract, original_run = api_sparams.extract_waveguide_s_matrix, solver.run
    records = {}
    summaries = []
    ref_shifts = []
    def capture(*a, **kw):
        run = original_run(*a, **kw)
        drive = len(summaries)
        summaries.append([])
        for port, cfg in enumerate(run.waveguide_ports):
            summary = dict(port=port, dt=float(cfg.dt), n_steps=int(np.asarray(cfg.n_steps_recorded)))
            for name in ['v_probe_t', 'v_ref_t', 'i_probe_t', 'i_ref_t']:
                value = np.asarray(getattr(cfg, name))
                records[f'drive{drive}_port{port}_{name}'] = value
                summary[name] = dict(peak=float(np.max(np.abs(value))), tail=float(np.max(np.abs(value[-max(1, len(value)//10):]))))
            a_wave, b_wave = extract_waveguide_port_waves(cfg, ref_shift=ref_shifts[port])
            for name, value in [('a_wave', a_wave), ('b_wave', b_wave)]:
                value = np.asarray(value)
                summary[name] = dict(real=value.real.tolist(), imag=value.imag.tolist(), magnitude=np.abs(value).tolist())
            summaries[-1].append(summary)
        return run
    def intervention(g, m, cfgs, n_steps, **kw):
        result['n_steps'] = int(n_steps)
        ref_shifts[:] = list(kw.get('ref_shifts') or [0.0] * len(cfgs))
        result['port_planes_m'] = [{k: float(v) for k, v in waveguide_plane_positions(c).items()} for c in cfgs]
        result['ref_shifts_m'] = ref_shifts
        # All arms must use identical configured sources, mesh, timestep,
        # port apertures, background materials and downstream extraction.
        def digest(arr):
            value = np.ascontiguousarray(np.asarray(arr))
            if value.dtype.hasobject:
                raise TypeError("Cannot fingerprint object-array pointer bytes")
            return hashlib.sha256(str((value.shape, value.dtype)).encode() + value.tobytes()).hexdigest()
        result['material_input_sha256'] = {k: digest(v) for k, v in m._asdict().items()}
        result['port_input_sha256'] = [{k: digest(v) for k, v in c._asdict().items()} for c in cfgs]
        assert g.shape == grid.shape
        if args.arm == 'legacy_sigma':
            m = m._replace(sigma=jnp.where(legacy_cells, 1e10, m.sigma))
            kw['pec_edge_masks'] = None
        elif args.arm == 'node_volume':
            kw['pec_edge_masks'] = node_edges
        return original_extract(g, m, cfgs, n_steps, **kw)
    started = time.monotonic()
    with patch.object(api_sparams, 'extract_waveguide_s_matrix', intervention), patch.object(solver, 'run', capture), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        measured = sim.compute_waveguide_s_matrix(normalize=False, num_periods=periods)
    s = np.asarray(measured.s_params)
    column_power = np.sum(np.abs(s)**2, axis=0)
    result.update(elapsed_s=time.monotonic()-started, s_real=s.real.tolist(), s_imag=s.imag.tolist(),
                  s_abs=np.abs(s).tolist(), column_power=column_power.tolist(), max_column_power=float(column_power.max()),
                  in_original_witness_interval=bool(2.25 < column_power.max() <= 3),
                  driven_incident_nonzero=[
                      bool(np.all(np.asarray(summaries[d][d]['a_wave']['magnitude']) > 0))
                      for d in range(len(summaries))],
                  settling_db=np.asarray(measured.settling_db).tolist(), records_summary=summaries,
                  warnings=[str(w.message) for w in caught])
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(str(output) + '_records.npz', **records)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
