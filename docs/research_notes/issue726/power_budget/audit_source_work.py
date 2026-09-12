"""Cheap MSL source/load functional and electric-substep work audit.

Prescribed profile, no RF propagation or mode-accuracy claim. Calls production
loading/source construction/update_e, then checks independent discrete work.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64
from rfx.grid import Grid
from rfx.core.yee import EPS_0, MaterialArrays, init_state, update_e
from rfx.sources.msl_port import MSLPort, setup_msl_port, make_msl_port_sources


def main(out):
    with enable_x64():
        grid=Grid(freq_max=1e8,domain=(6.,6.,6.),dx=1.,cpml_layers=0,cpml_axes='')
        cells=[(3,2,2),(3,2,3),(3,3,2),(3,3,3)]
        e=np.array([.2,.1,.5,.5]); fringe=np.array([.1,-.2,0.,0.])
        assert abs(e@fringe)<1e-16 and np.sum(fringe[2:])==0
        u=.8; r=50.
        port=MSLPort(feed_x=3.,y_lo=2.,y_hi=3.,z_lo=2.,z_hi=4.,direction='+x',
                     impedance=r,excitation=lambda t:jnp.asarray(u,dtype=jnp.float64))
        profile=dict(ez_profile=e.reshape(2,2),cell_indices=cells,j_grid_lo=2,k_grid_lo=2,n_z_sub=2,
                     prop_idx=0,width_idx=1,normal_idx=2,prop_axis='x',width_axis='y',normal_axis='z')
        materials=MaterialArrays(jnp.full(grid.shape,2.25,dtype=jnp.float64),
                                jnp.zeros(grid.shape,dtype=jnp.float64),jnp.ones(grid.shape,dtype=jnp.float64))
        loaded=setup_msl_port(grid,port,materials,mode_profile=profile)
        sources=make_msl_port_sources(grid,port,loaded,1,mode_profile=profile)
        n=float(e@e)*grid.dx**3
        sigma=np.asarray(loaded.sigma)
        np.testing.assert_allclose([sigma[c] for c in cells],1/(r*n),rtol=1e-14)
        results=[]
        for label,ez_values,ex_value,ey_value in [('profile_only',e,0.,0.),
                                                ('orthogonal_fringe',e+fringe,0.,0.),
                                                ('transverse_field',e,.3,-.2)]:
            old=init_state(grid.shape,field_dtype=jnp.float64)
            for k,cell in enumerate(cells):
                old=old._replace(ez=old.ez.at[cell].set(ez_values[k]),
                                 ex=old.ex.at[cell].set(ex_value),ey=old.ey.at[cell].set(ey_value))
            before_load=float(sum(sigma[c]*(ez_values[k]**2+ex_value**2+ey_value**2)
                                  for k,c in enumerate(cells))*grid.dx**3)
            vcentre=float(np.sum(ez_values[2:])*grid.dx)
            vproj=float(e@ez_values*grid.dx**3/n)
            # One electric substep with prescribed H^{n+1/2}=0. This is not
            # a full Yee step; curl work is exactly zero in this check.
            new=update_e(old,loaded,grid.dt,grid.dx,periodic=(True,True,True))
            for source in sources:
                field=getattr(new,source.component)
                new=new._replace(**{source.component:field.at[source.i,source.j,source.k].add(source.waveform[0])})
            storage=0.;dissipation=0.;mid={}
            for comp in ('ex','ey','ez'):
                eo,en=np.asarray(getattr(old,comp)),np.asarray(getattr(new,comp))
                mid[comp]=(eo+en)/2
                storage+=float(np.sum(2.25*EPS_0*(en**2-eo**2)/2)*grid.dx**3/grid.dt)
                dissipation+=float(np.sum(sigma*mid[comp]**2)*grid.dx**3)
            q=float(sum(e[k]*mid['ez'][cell] for k,cell in enumerate(cells))*grid.dx**3)
            injected=u*q
            residual=storage+dissipation-injected
            assert abs(residual)<1e-13*max(abs(storage),abs(dissipation),abs(injected),1e-30)
            results.append(dict(case=label,initial_centre_voltage=vcentre,initial_projected_voltage=vproj,
                                initial_actual_load_power=before_load,initial_vcentre_squared_over_r=vcentre**2/r,
                                electric_storage_rate=storage,actual_ohmic_power=dissipation,
                                source_work_power=injected,relative_work_residual=abs(residual)/max(abs(injected),1e-30),
                                power_conjugate_mid_voltage=q/n,source_current=n*u,
                                actual_equivalent_thevenin_voltage=r*n*u))
        np.testing.assert_allclose([x['initial_centre_voltage'] for x in results],1,atol=1e-14)
        np.testing.assert_allclose([x['initial_projected_voltage'] for x in results],1,atol=1e-14)
        assert results[1]['initial_actual_load_power']>results[0]['initial_actual_load_power']
        assert results[2]['initial_actual_load_power']>results[0]['initial_actual_load_power']
        record=dict(scope='structural prescribed-field audit; no Maxwell mode or RF accuracy verdict',
                    grid_dx_m=grid.dx,dt_s=grid.dt,reference_resistance_ohm=r,profile_norm_n_m=n,
                    waveform_parameter_u=u,waveform_parameter_units_from_equation='A/m for profile in1/m',
                    source_rhs='e*u, not sigma_port*e*u',
                    results=results)
        assert not out.exists()
        out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    main(ap.parse_args().out)
