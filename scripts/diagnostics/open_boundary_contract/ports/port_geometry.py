from __future__ import annotations
import time
import traceback
from census_v2 import *
from rfx import simulation as lowlevel
from rfx.sources import coaxial_port as coax
from rfx.geometry.csg import Cylinder

def main():
    row=next(r for r in catalog_v2() if r['path'].endswith('test_coax_two_port_smatrix.py'))
    sim,mod,captured=build_v2(row)
    record=dict(index=86,id=row['id']+'::generated_port_geometry',kind='additional_port_geometry',path=row['path'],builder=row['builder'],label='public_driver_geometry',lane='coax_two_port',entities=[],completed=0,solve_calls=0,lowlevel_calls=0)
    start=time.perf_counter()
    orig_stamp=coax.stamp_coaxial_line
    orig_run=lowlevel.run
    def stamp(grid,materials,**kw):
        result=orig_stamp(grid,materials,**kw)
        dx=float(grid.dx)
        zlo=(kw['z_lo_index']-grid.pad_z_lo)*dx
        zhi=(kw['z_hi_index']-grid.pad_z_lo)*dx
        center=(*kw['center_xy'],0.5*(zlo+zhi))
        h=zhi-zlo+2*dx
        pin=Cylinder(center=center,radius=kw['pin_radius'],height=h,axis='z')
        outer=Cylinder(center=center,radius=kw['outer_radius'],height=h,axis='z')
        inner=Cylinder(center=center,radius=result[1],height=h,axis='z')
        pm,om,im=[np.asarray(s.mask(grid),bool) for s in (pin,outer,inner)]
        for name,shape,mask,isconductor in [('pin',pin,pm,1),('shell',outer,om&~im,1),('dielectric_annulus',inner,im&~pm,0)]:
            lo,hi=[list(map(float,x)) for x in shape.bounding_box()]
            counts=face_counts(mask,grid)
            reach={f:bool(lo['xyz'.index(f[0])]<=sim._domain['xyz'.index(f[0])]*1e-9 if f.endswith('lo') else hi['xyz'.index(f[0])]>=sim._domain['xyz'.index(f[0])]*(1-1e-9)) for f in counts}
            record['entities'].append(dict(name=name,conductor=isconductor,bounds_m=[lo,hi],reaches=reach,sigma_pec_cells=counts if isconductor else None,eps_not_1=counts if not isconductor else None,affected_faces=[f for f in counts if isconductor and reach[f] and counts[f]['absorber']==0]))
        record.update(grid_shape=list(grid.shape),cells=int(np.prod(grid.shape)),dt_s=float(grid.dt),face_pads=pads(grid),domain_m=sim._domain,boundary={a+'_'+s:getattr(getattr(sim._boundary_spec,a),s) for a in 'xyz' for s in ('lo','hi')},stamp_arguments=kw)
        return result
    def intercept(grid,materials,n_steps,**kwargs):
        record['lowlevel_calls']+=1
        record['steps']=int(n_steps)
        record['cpml_axes_received']=kwargs.get('cpml_axes')
        record['received_pec_sigma_cells']=face_counts(np.asarray(materials.sigma)>=sim._PEC_SIGMA_THRESHOLD,grid)
        record['received_eps_not_1']=face_counts(np.asarray(materials.eps_r)!=1,grid)
        save_npz(OUT/'coax_port_geometry_received.npz',eps_r=np.asarray(materials.eps_r),sigma=np.asarray(materials.sigma))
        raise Captured()
    try:
        coax.stamp_coaxial_line=stamp
        lowlevel.run=intercept
        sim.compute_coaxial_two_port(n_steps=6000,freqs=mod.BAND)
    except Captured:
        record['completed']=1
    except BaseException as exc:
        record.update(exception=str(exc),exception_type=type(exc).__name__,traceback=traceback.format_exc())
    finally:
        coax.stamp_coaxial_line=orig_stamp
        lowlevel.run=orig_run
    record['affected']=int(any(e['affected_faces'] for e in record['entities']))
    record['wall_s']=time.perf_counter()-start
    write_json(OUT/'census_records_v2'/'086.json',record)
    print(json.dumps(serial(record)),flush=True)

if __name__=='__main__':
    main()
