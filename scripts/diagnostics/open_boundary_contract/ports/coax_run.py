from __future__ import annotations
import argparse
import contextlib
import time
import traceback
from port_run import Tee, result_dict
from census_v2 import *
from rfx import simulation as lowlevel
from rfx.sources import coaxial_port as coax
from rfx.geometry.csg import Cylinder

def copied_stamp(grid,materials,*,center_xy,z_lo_index,z_hi_index,pin_radius,outer_radius,continued):
    dx=float(grid.dx)
    zlo=(int(z_lo_index)-grid.pad_z_lo)*dx
    zhi=(int(z_hi_index)-grid.pad_z_lo)*dx
    original_lo=zlo-dx
    original_hi=zhi+dx
    new_hi=float(grid.domain[2])+(grid.pad_z_hi+1)*dx if continued else original_hi
    old_center=(*map(float,center_xy),0.5*(zlo+zhi))
    new_center=(*map(float,center_xy),0.5*(original_lo+new_hi))
    old_height=zhi-zlo+2*dx
    new_height=new_hi-original_lo
    shell_inner=float(outer_radius)-min(dx,0.5*(float(outer_radius)-float(pin_radius)))
    # Original dielectric annulus declaration.
    inner_d=Cylinder(center=old_center,radius=shell_inner,height=old_height,axis='z').mask(grid)
    pin_d=Cylinder(center=old_center,radius=pin_radius,height=old_height,axis='z').mask(grid)
    # Only the high-z bounds of the two conductor declarations differ.
    outer=Cylinder(center=new_center,radius=outer_radius,height=new_height,axis='z').mask(grid)
    inner=Cylinder(center=new_center,radius=shell_inner,height=new_height,axis='z').mask(grid)
    pin=Cylinder(center=new_center,radius=pin_radius,height=new_height,axis='z').mask(grid)
    eps=np.array(materials.eps_r)
    sig=np.array(materials.sigma)
    shell=outer & ~inner
    eps=np.where(shell,1.0,eps)
    sig=np.where(shell,coax.PEC_SIGMA,sig)
    dielectric=inner_d & ~pin_d
    eps=np.where(dielectric,float(coax.PTFE_EPS_R),eps)
    sig=np.where(dielectric,0.0,sig)
    eps=np.where(pin,1.0,eps)
    sig=np.where(pin,coax.PEC_SIGMA,sig)
    return materials._replace(eps_r=jnp.asarray(eps),sigma=jnp.asarray(sig)),shell_inner,dict(old_z_bounds_m=[original_lo,original_hi],new_z_bounds_m=[original_lo,new_hi])

def execute(variant,out,dry):
    status=dict(lane='coax_two_port',index=77,variant=variant,completed=0,dry_run=int(dry),solve_calls=0)
    start=time.perf_counter()
    stamp_original=coax.stamp_coaxial_line
    run_original=lowlevel.run
    expected={}
    try:
        row=catalog_v2()[77]
        sim,mod,_=build_v2(row)
        write_json(out/'settings.json',dict(method='compute_coaxial_two_port',kwargs=dict(n_steps=6000,freqs=mod.BAND)))
        with (out/'preflight.txt').open('x') as f:
            f.write('compute_coaxial_two_port public preflight calls: 0\n')
        def stamp(grid,materials,**kwargs):
            original,shell=stamp_original(grid,materials,**kwargs)
            copy0,copy_shell,bounds0=copied_stamp(grid,materials,**kwargs,continued=False)
            assert copy_shell==shell
            assert np.array_equal(np.asarray(copy0.eps_r),np.asarray(original.eps_r))
            assert np.array_equal(np.asarray(copy0.sigma),np.asarray(original.sigma))
            after,shell,bounds=copied_stamp(grid,materials,**kwargs,continued=variant=='continued')
            assert np.array_equal(np.asarray(after.eps_r),np.asarray(original.eps_r))
            old=np.asarray(original.sigma)
            new=np.asarray(after.sigma)
            stop=grid.shape[2]-grid.pad_z_hi
            assert np.array_equal(old[:,:,:stop],new[:,:,:stop])
            sigma_pec=np.asarray(after.sigma)>=sim._PEC_SIGMA_THRESHOLD
            expected['sigma_pec']=sigma_pec
            counts=face_counts(sigma_pec,grid)
            record=dict(bounds=bounds,counts=counts,dielectric_eps_equal=1,interior_sigma_equal=1,baseline_builder_copy_equal=1,changed_sigma_cells=int(np.count_nonzero(old!=new)),grid_shape=list(grid.shape),face_pads=pads(grid))
            save_npz(out/'assembly_before_solve.npz',sigma=np.asarray(after.sigma),eps_r=np.asarray(after.eps_r))
            write_json(out/'assembly_before_solve.json',record)
            print(json.dumps(record),flush=True)
            if variant=='continued' and counts['z_hi']['absorber']==0:
                raise RuntimeError('continuation_absorber_count=0: coax z_hi')
            return after,shell
        def observe(grid,materials,n_steps,**kwargs):
            n=status['solve_calls']
            sigma_pec=np.asarray(materials.sigma)>=sim._PEC_SIGMA_THRESHOLD
            equal=int(np.array_equal(sigma_pec,expected['sigma_pec']))
            record=dict(call=n,n_steps=int(n_steps),sigma_pec_equal_to_builder=equal,counts=face_counts(sigma_pec,grid),pec_mask_present=int(kwargs.get('pec_mask') is not None),pec_edge_masks_present=int(kwargs.get('pec_edge_masks') is not None),cpml_axes=kwargs.get('cpml_axes'))
            status['solve_calls']+=1
            save_npz(out/f'assembly_received_{n:02d}.npz',sigma=np.asarray(materials.sigma),eps_r=np.asarray(materials.eps_r),sigma_pec=sigma_pec)
            write_json(out/f'assembly_received_{n:02d}.json',record)
            print(json.dumps(record),flush=True)
            assert equal
            if dry:
                raise Captured()
            result=run_original(grid,materials,n_steps,**kwargs)
            rd=result_dict(result)
            if rd.get('time_series') is not None:
                save_npz(out/f'witness_series_{n:02d}.npz',time_series=np.asarray(rd['time_series']))
            return result
        coax.stamp_coaxial_line=stamp
        lowlevel.run=observe
        try:
            result=sim.compute_coaxial_two_port(n_steps=6000,freqs=mod.BAND)
        except Captured:
            if not dry:
                raise
            status['completed']=1
            return status
        rd=result_dict(result)
        S=np.asarray(rd['s_params'])
        freqs=np.asarray(rd['freqs'])
        save_npz(out/'s.npz',S=S,freqs=freqs)
        write_json(out/'diagnostics.json',rd)
        save_npz(out/'diagnostics.npz',**{k:np.asarray(v) for k,v in rd.items() if isinstance(v,(np.ndarray,jax.Array,np.generic,float,int,bool,complex))})
        status.update(completed=1,frequencies=len(freqs),max_column_power=float(np.max(np.sum(np.abs(S)**2,axis=0))))
        return status
    except BaseException as exc:
        status.update(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc())
        print(status['traceback'],flush=True)
        return status
    finally:
        coax.stamp_coaxial_line=stamp_original
        lowlevel.run=run_original
        status['wall_s']=time.perf_counter()-start
        write_json(out/'status.json',status)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--require-gpu',action='store_true')
    args=ap.parse_args()
    if args.require_gpu:
        assert all(d.platform=='gpu' for d in jax.devices()),jax.devices()
    dest=OUT/('dry_coax' if args.dry_run else 'coax_two_port')/'fixture_077'
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    write_json(dest/'provenance.json',dict(devices=[str(d) for d in jax.devices()],sha256={str(p.relative_to(BASE)):sha(p) for p in (Path(__file__),OUT/'port_run.py',OUT/'measure_common.py',OUT/'census_v2.py',SRC/'rfx/boundaries/cpml.py',SRC/'rfx/sources/coaxial_port.py')}))
    for variant in ('baseline','continued'):
        out=dest/variant
        out.mkdir()
        with (out/'run.log').open('x') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
                status=execute(variant,out,args.dry_run)
        print(json.dumps(status),flush=True)
        if not status['completed']:
            break

if __name__=='__main__':
    main()
