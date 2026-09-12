"""Record raw fields for independent port/flux power comparison, not calibration.

Fixed CV06b sources and DUT; two sequential drives, six x observation planes,
and a closed observation box. Every H face is saved on BOTH real neighbouring
planes. Spatial/time alignment and component-specific quadrature are offline.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

REPO = Path(__import__('rfx').__file__).resolve().parents[1]
COMPONENTS = {0: ('ey', 'ez', 'hy', 'hz'), 1: ('ez', 'ex', 'hz', 'hx'),
              2: ('ex', 'ey', 'hx', 'hy')}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def build(*, lead_through_cpml=False):
    path = REPO / 'scripts/diagnostics/msl_probe_clearance_bias.py'
    spec = importlib.util.spec_from_file_location('fixed_source_case', path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    cv, arms, inputs = fixture.prepare_inputs()
    sim = arms['clean']
    rz = fixture._realized(sim)
    grid = rz.grid
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    from rfx.sources.msl_port import msl_cross_section_span, msl_port_from_entry
    gc = coords_from_uniform_grid(grid)
    nodes = [np.asarray(getattr(gc, a)) for a in 'xyz']
    lead_change = None
    if lead_through_cpml:
        # This is a controlled CV06b input change, not automatic API geometry
        # extension. Rebuild before realizing the new geometry.
        from rfx import Box
        original_box = cv.Box
        replacements = []
        def explicit_lead(lo, hi):
            if (lo[0] == 0 and hi[0] == cv.L_LINE + 2*cv.PORT_MARGIN
                    and lo[2] == hi[2] == cv.H_SUB):
                corners = ((float(nodes[0][0]), lo[1], lo[2]),
                           (float(nodes[0][-1]), hi[1], hi[2]))
                replacements.append(dict(old=[list(lo), list(hi)], new=[list(c) for c in corners]))
                return Box(*corners)
            return original_box(lo, hi)
        cv.Box = explicit_lead
        try:
            candidate = cv._build_sim()
        finally:
            cv.Box = original_box
        assert len(replacements) == 1
        candidate._msl_ports = list(sim._msl_ports)
        candidate._msl_auto_offset_min = {}
        candidate._msl_auto_probe_spacing = {}
        changed = fixture._realized(candidate)
        assert changed.grid.shape == grid.shape and changed.grid.dt == grid.dt
        # Match the whole source-to-source span, which contains the DUT and
        # all observation planes. Only the exterior terminating leads move.
        from rfx.sources.msl_port import msl_cross_section_span, msl_port_from_entry
        feed_indices = [msl_cross_section_span(grid, msl_port_from_entry(p))['i_feed']
                        for p in sim._msl_ports]
        lo, hi = min(feed_indices)-2, max(feed_indices)+3
        for old_mask, new_mask in zip(rz.edge_masks, changed.edge_masks):
            np.testing.assert_array_equal(np.asarray(old_mask)[lo:hi], np.asarray(new_mask)[lo:hi])
        # Dielectric and permeability arrays remain identical over the entire
        # carrier, including pads. No source/load parameters change.
        m0 = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
        m1 = candidate._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
        for key in ('eps_r', 'mu_r', 'sigma'):
            np.testing.assert_array_equal(getattr(m0,key), getattr(m1,key))
        assert [asdict(p) for p in sim._msl_ports] == [asdict(p) for p in candidate._msl_ports]
        lead_change = dict(**replacements[0], unchanged_pec_x_slice=[lo,hi],
                           all_material_arrays_identical=True, source_load_declarations_identical=True,
                           requested_boundary_model='trace explicitly continuous through allocated x CPML')
        sim, rz = candidate, changed
    archive = REPO / 'docs/research_notes/issue726/collocation/gpu-369367260605/artifacts/clean-phasors.npz'
    with np.load(archive, allow_pickle=False) as old:
        all_freqs = np.asarray(old['freqs_hz'])
    # Prespecified bad bins and band controls; retain actual stored f32 bins.
    targets = [2e9, 3.653125e9, 3.77125e9, 3.968125e9, 5e9]
    selected = [int(np.argmin(abs(all_freqs - f))) for f in targets]
    freqs = np.asarray(all_freqs[selected], dtype=np.float32)
    def nearest_x(x):
        return int(np.argmin(abs(nodes[0] - x)))
    left = nearest_x(inputs['ports']['clean'][0]['probe_x_m'][0])
    right = nearest_x(inputs['ports']['clean'][1]['probe_x_m'][0])
    near = nearest_x(inputs['ports']['near'][0]['probe_x_m'][0])
    stub_lo, stub_hi = inputs['realized_metal']['stub_i']
    right_near = int(stub_hi + (stub_lo - near))
    station_indices = [left, (left+near)//2, near, right_near, (right_near+right)//2, right]
    assert station_indices == sorted(set(station_indices))
    box_lo = [left, grid.pad_y_lo+1, 0]
    box_hi = [right, grid.ny-grid.pad_y_hi-2, grid.nz-grid.pad_z_hi-2]
    assert grid.pad_z_lo == 0
    assert inputs['ports']['clean'][0]['feed_x_m'] < nodes[0][left]
    assert inputs['ports']['clean'][1]['feed_x_m'] > nodes[0][right]
    assert all(box_hi[a] > box_lo[a]+2 for a in range(3))
    faces = []
    for number, idx in enumerate(station_indices):
        faces.append(dict(name=f'x_station_{number}', axis=0, index=idx,
                          outward=(-1 if number == 0 else 1 if number == 5 else 0)))
    for axis in (1, 2):
        for sign in (-1, 1):
            faces.append(dict(name=f'{"xyz"[axis]}_{"lo" if sign == -1 else "hi"}',
                              axis=axis, index=(box_lo if sign == -1 else box_hi)[axis],
                              outward=sign))
    sim._dft_plane_regions = dict(getattr(sim, '_dft_plane_regions', {}))
    probe_specs = []
    for face in faces:
        axis, idx = face['axis'], face['index']
        tangents = [a for a in range(3) if a != axis]
        region = [box_lo[tangents[0]], box_hi[tangents[0]]+1,
                  box_lo[tangents[1]], box_hi[tangents[1]]+1]
        face.update(region=region, coordinate_m=float(nodes[axis][idx]), probes={},
                    tangential_axes=tangents)
        for component in COMPONENTS[axis]:
            # Bottom is an exact PEC E plane; don't invent a missing H[-1].
            sample_indices = [idx] if component.startswith('e') else [idx-1, idx]
            if axis == 2 and idx == 0 and component.startswith('h'):
                face['zero_flux_pec_boundary'] = True
                continue
            keys = []
            for sample in sample_indices:
                assert sample >= 0
                name = f'power_{face["name"]}_{component}_{sample}'
                coordinate = float(nodes[axis][sample])
                sim.add_dft_plane_probe(axis='xyz'[axis], coordinate=coordinate,
                                        component=component, freqs=freqs, name=name)
                sim._dft_plane_regions[name] = tuple(region)
                roundtrip = grid.position_to_index(tuple(coordinate if a == axis else 0 for a in range(3)))[axis]
                assert roundtrip == sample, (name, sample, roundtrip)
                probe_specs.append(dict(name=name, axis=axis, index=sample,
                                        component=component, coordinate_m=coordinate, region=region))
                keys.append(name)
            face['probes'][component] = keys
    plan = dict(scope='raw field diagnostic only; no geometry/source change or accuracy PASS',
                inputs=inputs, lead_change=lead_change,
                registered_geometry=[asdict(g) for g in sim._geometry],
                box_node_lo=box_lo, box_node_hi=box_hi,
                box_coordinates_m=[[float(nodes[a][v[a]]) for a in range(3)] for v in (box_lo, box_hi)],
                grid_shape=list(grid.shape), dt_s=float(grid.dt), dx_m=float(grid.dx),
                all_freqs_hz=all_freqs.tolist(), selected_indices=selected,
                field_freqs_hz=freqs.astype(float).tolist(), faces=faces, probes=probe_specs,
                port_spans=[{k: (int(v) if isinstance(v, (int, np.integer)) else v)
                             for k,v in msl_cross_section_span(grid, msl_port_from_entry(p)).items()}
                            for p in sim._msl_ports],
                baseline_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                num_periods=100, passivity_projection=False)
    return sim, plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--lead-through-cpml', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sim, plan = build(lead_through_cpml=args.lead_through_cpml)
    write_json(args.out/'plan.json', plan)
    if args.build_only:
        print(json.dumps(dict(probes=len(plan['probes']), faces=len(plan['faces']),
                              shape=plan['grid_shape'], frequencies=plan['field_freqs_hz'])))
        return
    assert jax.default_backend() == 'gpu', jax.devices()
    assert not jax.config.jax_enable_x64, 'preserve the original f32 field/geometry lane'
    original = sim.run
    records = []
    def capture(**kwargs):
        driven = len(records)
        assert driven < 2
        start = time.monotonic()
        result = original(**kwargs)
        arrays = {key: np.asarray(result.dft_planes[key].accumulator) for key in
                  [p['name'] for p in plan['probes']]}
        for spec in plan['probes']:
            probe = result.dft_planes[spec['name']]
            assert probe.index == spec['index'] and tuple(probe.region) == tuple(spec['region'])
            np.testing.assert_array_equal(probe.freqs, plan['field_freqs_hz'])
        assert all(np.all(np.isfinite(a)) for a in arrays.values())
        assert str(result.state.ez.dtype) == 'float32'
        np.savez_compressed(args.out/f'drive-{driven}-fields.npz', **arrays)
        box_slice = tuple(slice(a,b+1) for a,b in zip(plan['box_node_lo'],plan['box_node_hi']))
        endpoint = {}
        for component in ('ex','ey','ez','hx','hy','hz'):
            field = np.asarray(getattr(result.state,component)[box_slice], dtype=np.float64)
            endpoint[component] = dict(max_abs=float(np.max(abs(field))), sum_squared=float(np.sum(field**2)))
        write_json(args.out/f'drive-{driven}-endpoint.json', dict(
            scope='final instantaneous fields only; no bound on the finite-window power residual',
            initial_fields_zero=True, start_step=0, end_step=int(result.state.step), fields=endpoint))
        records.append(dict(drive=driven, wall_s=time.monotonic()-start,
                            state_dtype=str(result.state.ez.dtype),
                            accumulator_dtypes=sorted({str(a.dtype) for a in arrays.values()}),
                            n_steps=int(result.state.step)))
        write_json(args.out/'drives.json', records)
        return result
    sim.run = capture
    result = sim.compute_msl_s_matrix(freqs=np.asarray(plan['all_freqs_hz']),
                                      num_periods=plan['num_periods'], enforce_passivity=False,
                                      report_every=5000, raw_3probe_dump_path=str(args.out/'raw-vi.npz'))
    np.savez_compressed(args.out/'result.npz', **{k: np.asarray(getattr(result,k)) for k in
                        ['S','freqs','Z0','beta','beta_railed','cond_a','reliable','settling_db','reference_impedances']})
    assert len(records) == 2
    write_json(args.out/'outcome.json', dict(status='recorded', assembly=result.assembly,
                                            settling_db=np.asarray(result.settling_db).tolist(),
                                            physical_accuracy_verdict=None))


if __name__ == '__main__':
    main()
