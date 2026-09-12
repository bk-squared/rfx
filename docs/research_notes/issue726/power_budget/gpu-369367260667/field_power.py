"""Observe source-work variables on the unchanged633 model; no source change.

Stores post-update Ez at every actual source/load cell, source waveforms,
source-cell E DFTs, and the ordinary MSL result. The midpoint source-model
power observable is diagnostic only and must not replace the MSL S.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.sources.msl_port import (compute_msl_mode_profile,msl_port_from_entry,
                                  msl_cross_section_span,msl_cell,setup_msl_port)

REPO=Path(__import__('rfx').__file__).resolve().parents[1]


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def build():
    path=REPO/'scripts/diagnostics/msl_probe_clearance_bias.py'
    spec=importlib.util.spec_from_file_location('source_work_case',path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    _cv,arms,inputs=fixture.prepare_inputs();sim=arms['clean']
    grid=sim._build_grid();gc=coords_from_uniform_grid(grid)
    nodes=[np.asarray(getattr(gc,a)) for a in 'xyz']
    materials=sim._assemble_materials(grid,pec_sheets=[],pec_wires=[])[0]
    initial_sigma=np.asarray(materials.sigma)
    ports=[];profiles=[]
    archive=REPO/'docs/research_notes/issue726/collocation/gpu-369367260605/artifacts/clean-phasors.npz'
    with np.load(archive,allow_pickle=False) as data:freqs=np.asarray(data['freqs_hz'])
    sim._dft_plane_regions=dict(getattr(sim,'_dft_plane_regions',{}))
    for p,entry in enumerate(sim._msl_ports):
        assert entry.mode=='laplace'
        mp=msl_port_from_entry(entry);span=msl_cross_section_span(grid,mp)
        eps_cell=msl_cell(entry.direction,span['i_feed'],span['w_centre'],(span['n_lo']+span['n_hi'])//2)
        eps=float(entry.eps_r_sub) if entry.eps_r_sub is not None else float(np.asarray(materials.eps_r[eps_cell]))
        profile=compute_msl_mode_profile(grid,mp,eps)
        assert profile['prop_axis']=='x' and profile['normal_axis']=='z'
        cells=[];weights=[];columns=[];physical_coordinates=[];registration_coordinates=[]
        for cell in profile['cell_indices']:
            j=cell[1]-profile['j_grid_lo'];k=cell[2]-profile['k_grid_lo']
            if not (0<=k<profile['n_z_sub']):continue
            e=float(profile['ez_profile'][j,k])
            if e==0:continue
            assert all(s.start<=c<s.stop for c,s in zip(cell,grid.interior))
            coordinate=tuple(float(nodes[a][c]) for a,c in enumerate(cell))
            assert grid.position_to_index(coordinate)==tuple(cell)
            columns.append(len(sim._probes));sim.add_probe(coordinate,'ez')
            sim._internal_probe_indices.add(columns[-1])
            cells.append(list(cell));weights.append(e);registration_coordinates.append(coordinate)
            physical_coordinates.append((coordinate[0],coordinate[1],coordinate[2]+grid.dx/2))
        assert len(cells)==len(set(map(tuple,cells)))
        region=(min(c[1] for c in cells),max(c[1] for c in cells)+1,
                min(c[2] for c in cells),max(c[2] for c in cells)+1)
        names={}
        for component in ('ex','ey','ez'):
            name=f'source_work_p{p}_{component}'
            sim.add_dft_plane_probe(axis='x',coordinate=float(nodes[0][span['i_feed']]),
                                    component=component,freqs=freqs,name=name)
            sim._dft_plane_regions[name]=region;names[component]=name
        before=np.asarray(materials.sigma)
        materials=setup_msl_port(grid,mp,materials,mode_profile=profile)
        index=tuple(np.asarray(cells).T)
        delta=np.asarray(materials.sigma)[index]-before[index]
        assert np.all(delta>0) and np.max(delta)==np.min(delta)
        sigma=float(delta[0]);volumes=np.full(len(cells),grid.dx**3)
        norm=float(np.sum(np.asarray(weights)**2*volumes))
        ports.append(dict(name=entry.name,nominal_load_ohm=entry.impedance,
                          eps_r_sub_inferred=eps,cells=cells,profile_e=weights,volumes_m3=volumes.tolist(),
                          profile_norm_m=norm,added_sigma_s_per_m=sigma,
                          effective_reference_ohm=1/(sigma*norm),point_columns=columns,
                          point_registration_coordinates_m=registration_coordinates,
                          physical_ez_coordinates_m=physical_coordinates,dft_names=names,dft_region=list(region)))
        profiles.append(profile)
    assert not (set(map(tuple,ports[0]['cells']))&set(map(tuple,ports[1]['cells'])))
    for p in ports:
        index=tuple(np.asarray(p['cells']).T)
        np.testing.assert_array_equal(np.asarray(materials.sigma)[index]-initial_sigma[index],p['added_sigma_s_per_m'])
    plan=dict(scope='passive additional source-work observers on unchanged633 geometry/source; distinct diagnostic observable',
              original_fixture_receipt=inputs,grid_shape=list(grid.shape),dt_s=float(grid.dt),dx_m=float(grid.dx),
              ports=ports,requested_freqs_hz=freqs.tolist(),num_periods=100,
              point_sample='E after electric update and source addition at step n+1; initial E=0',
              source_time='u[n] as actually sampled by make_msl_port_sources; conjugate work uses E midpoint',
              field_source_model_changed=False,replace_msl_s=False,
              driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return sim,grid,plan


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--build-only',action='store_true');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    sim,grid,plan=build();write_json(args.out/'plan.json',plan)
    if args.build_only:
        print(json.dumps(dict(source_cells=[len(p['cells']) for p in plan['ports']],point_probes=len(sim._probes),
                              references=[p['effective_reference_ohm'] for p in plan['ports']])))
        return
    assert jax.default_backend()=='gpu' and not jax.config.jax_enable_x64
    original=sim.run;records=[]
    def capture(**kwargs):
        drive=len(records);assert drive<2
        steps=kwargs.get('n_steps') or grid.num_timesteps(kwargs['num_periods'])
        times=jnp.arange(steps,dtype=jnp.float32)*grid.dt
        u=[];active=[]
        for p,entry in enumerate(sim._msl_ports):
            on=bool(entry.excite and entry.waveform is not None);active.append(on)
            u.append(np.asarray(jax.vmap(entry.waveform)(times)) if on else np.zeros(steps,dtype=np.float32))
        assert active==[p==drive for p in range(2)],active
        start=time.monotonic();result=original(**kwargs)
        assert int(result.state.step)==steps and str(result.state.ez.dtype)=='float32'
        arrays=dict(source_u=np.column_stack(u))
        for p,port in enumerate(plan['ports']):
            point=np.asarray(result.time_series[:,port['point_columns']])
            assert point.shape==(steps,len(port['cells']))
            arrays[f'p{p}_ez_post']=point
            ix=tuple(np.asarray(port['cells']).T)
            np.testing.assert_array_equal(point[-1],np.asarray(result.state.ez)[ix])
            for component,name in port['dft_names'].items():
                probe=result.dft_planes[name]
                assert tuple(probe.region)==tuple(port['dft_region'])
                arrays[f'p{p}_{component}_dft']=np.asarray(probe.accumulator)
                arrays[f'p{p}_{component}_end']=np.asarray(getattr(result.state,component))[ix]
            actual_freqs=np.asarray(result.dft_planes[port['dft_names']['ez']].freqs)
            if p==0:arrays['actual_freqs_hz']=actual_freqs
            else:np.testing.assert_array_equal(actual_freqs,arrays['actual_freqs_hz'])
        assert all(np.all(np.isfinite(v)) for v in arrays.values())
        np.savez_compressed(args.out/f'drive-{drive}-source-work.npz',**arrays)
        records.append(dict(drive=drive,steps=steps,wall_s=time.monotonic()-start,active_ports=active,
                            field_dtype=str(result.state.ez.dtype),source_u_dtype=str(arrays['source_u'].dtype),
                            raw_point_dtype=str(arrays['p0_ez_post'].dtype),dfreq_dtype=str(arrays['actual_freqs_hz'].dtype)))
        write_json(args.out/'drives.json',records)
        return result
    sim.run=capture
    result=sim.compute_msl_s_matrix(freqs=np.asarray(plan['requested_freqs_hz']),num_periods=100,
                                    enforce_passivity=False,report_every=5000,
                                    raw_3probe_dump_path=str(args.out/'raw-vi.npz'))
    np.savez_compressed(args.out/'result.npz',**{k:np.asarray(getattr(result,k)) for k in
                        ('S','freqs','Z0','beta','beta_railed','cond_a','reliable','settling_db','reference_impedances')})
    assert len(records)==2
    write_json(args.out/'outcome.json',dict(status='recorded',assembly=result.assembly,
                                           settling_db=np.asarray(result.settling_db).tolist(),
                                           source_work_s_is_distinct=True,physical_accuracy_verdict=None))


if __name__=='__main__':main()
