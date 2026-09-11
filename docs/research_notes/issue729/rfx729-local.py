import sys,json
import numpy as np
sys.path.insert(0,'/root/rfx-codex')
from rfx import Simulation, Box
from rfx.sources.msl_port import MSLPort,compute_msl_mode_profile,setup_msl_port
from rfx.boundaries.pec import realized_pec_edge_masks,realized_wall_planes
h=254e-6;dx=h/4; gnd=8*dx;top=gnd+h;Lx=32*dx;Ly=24*dx
sim=Simulation(freq_max=5e9,domain=(Lx,Ly,24*dx),dx=dx,cpml_layers=0,boundary='pec')
sim.add_material('substrate',eps_r=3.66)
sim.add(Box((0,0,gnd),(Lx,Ly,top)),material='substrate')
sim.add_thin_conductor(Box((0,0,gnd),(Lx,Ly,gnd)))
sim.add_thin_conductor(Box((0,8*dx,top),(Lx,16*dx,top)))
grid=sim._build_grid();sheets=[];wires=[]
mat,_,_,mask,_,_,_=sim._assemble_materials(grid,pec_sheets=sheets,pec_wires=wires)
edges=realized_pec_edge_masks(mask,sheets=tuple(sheets),wires=tuple(wires))
port=MSLPort(feed_x=4*dx,y_lo=8*dx,y_hi=16*dx,z_lo=gnd,z_hi=top,direction='+x',impedance=50,excitation=None)
profile=compute_msl_mode_profile(grid,port,3.66)
i,j,k=grid.position_to_index((port.feed_x,12*dx,gnd))
planes=realized_wall_planes(edges,2,ij=(i,j))
klo,khi=int(planes[0]),int(planes[-1]);ez=np.asarray(profile['ez_profile']);jc=j-profile['j_grid_lo']
vphys=np.sum(ez[jc,:khi-klo])*dx
vfull=np.sum(ez[jc])*dx
terminated=setup_msl_port(grid,port,mat,mode_profile=profile)
sig=np.asarray(terminated.sigma)-np.asarray(mat.sigma)
power=0.0
for ci,cj,ck in profile['cell_indices']:
 value=ez[cj-profile['j_grid_lo'],ck-profile['k_grid_lo']]
 power+=sig[ci,cj,ck]*value**2*dx**3
report=dict(dx=dx,ground_trace_planes=[int(v) for v in planes],expected_substrate_cells=khi-klo,mode_n_z_sub=profile['n_z_sub'],mode_height=profile['n_z_sub']*dx,physical_height=(khi-klo)*dx,z0_static=profile['z0_static'],eps_eff=profile['eps_eff'],source_k=sorted(set(c[2] for c in profile['cell_indices'])),voltage_between_actual_walls=float(vphys),voltage_over_all_source_cells=float(vfull),joule_power_for_mode=float(power),physical_voltage_squared_over_power=float(vphys**2/power))
print(json.dumps(report,indent=2))
open('/tmp/rfx729-local.json','w').write(json.dumps(report,indent=2)+'\n')
