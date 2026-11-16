"""Finite normal TFSF lens illumination, using a proper cyclic axis permutation.

Solver (x,y,z) = physical (z,x,y); physical Ex is solver Ey, physical -z
is solver -x. No oblique/Bloch excitation and no solver modifications.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import _design_common as dc


def build(dx,empty=False):
    from rfx import Simulation
    from rfx.geometry.csg import Box
    from rfx.sources.sources import CustomWaveform, GaussianPulse
    import _lens_filter_common as common
    sim=Simulation(freq_max=12.6e9,domain=(.120,.126,.126),dx=dx,
                   boundary='cpml',cpml_layers=common.integral_cells(.015,dx),precision='float32')
    # The original feed is solely add_source(..., 'ex'), not conductor geometry.
    if not empty:sim.add(Box((.018,.048,.048),(.018,.078,.078)),material='pec')
    sim.add_tfsf_source(f0=1e10,bandwidth=.5,polarization='ey',direction='-x',
                        angle_deg=0.,margin=3,closed_box=True,
                        waveform=CustomWaveform(GaussianPulse(f0=1e10,bandwidth=.5)))
    for plane,normal in [('xz','z'),('yz','y')]:
        for component in ('ex','ey','ez'):
            sim.add_dft_plane_probe(axis=normal,coordinate=.063,component=component,
                                   freqs=[1e10],name=f'{plane}_{component}')
    return sim,sim._build_grid()


def physical_bounds(sim,grid):
    indices=sim.tfsf_box_indices()
    return {physical:[float(grid.node_of(axis,i)) for i in indices[solver]]
            for physical,solver,axis in [('x','y',1),('y','z',2),('z','x',0)]}


def incident_spectrum(sim,grid,n):
    from rfx.sources.tfsf import tfsf_from_declaration
    from rfx.measurement.incident import measure_normal_incident_spectrum
    cfg,aux=tfsf_from_declaration(sim._tfsf,grid,sim._freq_max,line_steps=n)
    index=(cfg.x_lo+cfg.x_hi)//2
    spectrum=measure_normal_incident_spectrum(cfg,aux,n,np.array([1e10]),grid.dt,reference_index=index)
    return float(abs(np.asarray(spectrum)[0])),float(grid.node_of(0,index))


def plane_arrays(result,grid,eps):
    arrays={}
    for name,normal,transverse in [('xz',2,1),('yz',1,2)]:
        probes=[result.dft_planes[f'{name}_{c}'] for c in ('ey','ez','ex')]
        index=int(probes[0].index)
        if any(int(p.index)!=index for p in probes):raise ValueError('inconsistent component planes')
        E=np.stack([np.asarray(p.accumulator) for p in probes],axis=1).transpose(0,1,3,2).astype(np.complex64)
        if not np.isfinite(E).all():raise ValueError('nonfinite reciprocal field')
        arrays.update({f'{name}_E':E,f'{name}_axes':np.asarray(name),
          f'{name}_u_m':np.array([grid.node_of(transverse,i) for i in range(grid.shape[transverse])],np.float32),
          f'{name}_v_m':np.array([grid.node_of(0,i) for i in range(grid.shape[0])],np.float32),
          f'{name}_offset_m':np.float32(grid.node_of(normal,index)),
          f'{name}_eps':np.take(eps,index,axis=normal).T})
    return arrays


def witnesses(arrays,bounds,incident):
    if not np.isfinite(incident) or incident<=0:raise ValueError('zero/nonfinite incident spectrum')
    report={}
    for name in ('xz','yz'):
        u,v=np.meshgrid(arrays[f'{name}_u_m'],arrays[f'{name}_v_m'],indexing='ij')
        # Inclusive total-field nodes, no trimmed window or boundary exclusion.
        axis=name[0];tol=1e-8
        inside=(u>=bounds[axis][0]-tol)&(u<=bounds[axis][1]+tol)&(v>=bounds['z'][0]-tol)&(v<=bounds['z'][1]+tol)
        magnitude=np.linalg.norm(arrays[f'{name}_E'][0].astype(np.complex128),axis=0)
        inner=magnitude[inside];outer=magnitude[~inside]
        mean=float(inner.mean())
        report[name]=dict(inside_samples=int(inside.sum()),outside_samples=int((~inside).sum()),
          inside_mean_abs_E=mean,inside_min_abs_E=float(inner.min()),inside_max_abs_E=float(inner.max()),
          flatness_peak_to_peak_over_mean=float(np.ptp(inner)/mean) if mean else None,
          maximum_relative_deviation_from_mean=float(np.max(abs(inner-mean))/mean) if mean else None,
          outside_max_abs_E=float(outer.max()),outside_max_over_incident=float(outer.max()/incident))
    return report


def run(case,args,source):
    import jax.numpy as jnp
    import _lens_filter_common as common
    if (args.out/'field_reciprocal.json').exists() or any((args.out/f'field_reciprocal_{k}.npz').exists() for k in args.iterates):raise ValueError('reciprocal output already exists; refusing overwrite')
    metadata=dc.load_json(source/'result.json');saved=metadata['model'];length=dc.load_json(source/'record_length.json')
    if case.CASE!='lens' or metadata['id']!='design-lens':raise ValueError('reciprocal field is lens-only')
    if args.precision!='float32' or saved['precision']!=args.precision or bool(saved['smoke'])!=args.smoke:raise ValueError('record precision/smoke mismatch')
    n=int(length['n_steps'])
    if n<=0 or n!=length['n_steps']:raise ValueError('positive integer record length required')
    with np.load(source/'iterations.npz',allow_pickle=False) as z:designs=z['eps']
    if designs.shape[1:]!=case.SHAPE or any(k<0 or k>=len(designs) for k in args.iterates):raise ValueError('invalid design/iterate')
    if not np.any(np.isclose(saved['freqs_hz'],1e10,rtol=0,atol=1)):raise ValueError('10 GHz record required')
    sim,grid=build(saved['dx_m'],args.empty)
    if tuple(np.asarray(grid.shape)[[1,2,0]])!=tuple(saved['grid_shape']) or not np.isclose(grid.dt,length['dt_s'],rtol=1e-12,atol=0):raise ValueError('record grid/timestep mismatch')
    bounds=physical_bounds(sim,grid)
    for axis,low,high in [('x',.018,.108),('y',.018,.108),('z',.018,.102)]:
        if not bounds[axis][0]<low<=high<bounds[axis][1]:raise ValueError('TFSF boundary intersects lens or plate')
    drive='finite six-face normal TFSF; physical -z propagation, Ex polarization; CustomWaveform(GaussianPulse(f0=10GHz,bandwidth=0.5)); only source; raw rectangular DFT sum E(t)*exp(-2j*pi*f*t)*dt; native Yee components'
    amplitude,zref=incident_spectrum(sim,grid,n)
    details=dict(source=str(source.resolve()),n_steps=n,dt_s=grid.dt,dx_m=grid.dx,drive=drive,
      empty=args.empty,planes=['xz','yz'],freqs_hz=[1e10],tfsf_box_bounds_m=bounds,
      solver_axes_physical=['z','x','y'],physical_components_from_solver=['ey','ez','ex'],
      reflector='removed' if args.empty else 'inside total-field box; PEC plate is illuminated and scatters',
      feed='point-current excitation omitted; original model has no dipole metal geometry',
      incident_abs_E_dft=amplitude,incident_reference_physical_z_m=zref,
      witness_window='all inclusive total-field nodes; all remaining cut-plane nodes including CPML are outside',
      files=[],iterates={},display_only=True,field_monitor_tail_judged=False)
    for k in dict.fromkeys(args.iterates):
        eps=np.ones(grid.shape,np.float32)
        lo=grid.position_to_index((.072,.018,.018));hi=grid.position_to_index((.102,.108,.108))
        if not args.empty:eps[tuple(slice(a,b) for a,b in zip(lo,hi))]=case.expand_pixels(designs[k],grid.dx).transpose(2,0,1)
        result=sim.forward(eps_override=jnp.asarray(eps),n_steps=n,checkpoint=False,skip_preflight=True)
        arrays=plane_arrays(result,grid,eps)
        arrays.update(iterate=np.int64(k),freqs_hz=np.array([1e10],np.float32),planes=np.array(['xz','yz']),drive=np.asarray(drive),n_steps=np.int64(n),dx_m=np.float32(grid.dx),empty=np.bool_(args.empty))
        filename=f'field_reciprocal_{k}.npz';dc.save_npz(args.out/filename,**arrays)
        details['files'].append(filename)
        details['iterates'][str(k)]=dict(witnesses=witnesses(arrays,bounds,amplitude)) if args.empty else {}
        dc.save_json(args.out/'field_reciprocal.json',details)
        model=SimpleNamespace(precision=args.precision,smoke=args.smoke,default_steps=n,describe=lambda:dict(saved,reciprocal=details))
        common.record(args.out,case,model,'field',details)
    print(__import__('json').dumps(details,default=lambda x:np.asarray(x).tolist(),indent=2))
