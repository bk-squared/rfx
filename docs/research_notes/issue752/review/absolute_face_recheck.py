from dataclasses import replace
import numpy as np
from tests.unit.ports.test_msl_preflight_conductor_gap import _offset_stack,_messages,U
from rfx.sources.msl_port import _msl_grid_geometry
sim,g,t=_offset_stack('+x',False,'sheet','sheet')
pe=sim._msl_ports[0];pos=list(pe.position);pos[2]+=.2*U
sim._msl_ports[0]=replace(pe,position=tuple(pos));pe=sim._msl_ports[0]
a=sim._msl_assemble_once();face=sim._msl_declared_face_geometry(pe,a[0]);mat=sim._msl_realized_substrate(pe,2,a)
msgs=list(map(str,_messages(sim)))
assert abs(face['frac']-.2)<1e-14 and abs(mat['frac']-.2)<1e-14
mixed=[m for m in msgs if 'mixed-cell danger zone' in m]
assert len(mixed)==1 and 'set dx =' not in mixed[0]
print('Original offset-ground refutation closed:',face)
sim,g,t=_offset_stack('+x',True,'sheet','sheet');sim._geometry.pop(1);sim._dx=3*U
pe=sim._msl_ports[0];a=sim._msl_assemble_once();face=sim._msl_declared_face_geometry(pe,a[0])
assert sim._msl_realized_substrate(pe,2,a) is None and face['frac']==0 and face['nonuniform']
msgs=list(map(str,_messages(sim)))
assert not any('mixed-cell danger zone' in m or 'scalar estimate, run grid unavailable' in m for m in msgs)
print('Original absent-material NU refutation closed:',face)
count=0
for nonuniform in (False,True):
    sim,g,t=_offset_stack('-y',nonuniform,'sheet','sheet')
    grid=sim._msl_assemble_once()[0];nodes=_msl_grid_geometry(grid)[0][2]
    pe=sim._msl_ports[0]
    start=int(np.argmin(abs(nodes-g)))
    for index in range(start+1,len(nodes)-1):
        for frac in (0.,.125,.25,.5,.875):
            top=float(nodes[index]+frac*(nodes[index+1]-nodes[index]))
            other=replace(pe,height=top-pe.position[2])
            found=sim._msl_declared_face_geometry(other,grid)
            assert abs(found['frac']-frac)<1e-13,(nonuniform,index,frac,found)
            assert found['d_iface']==float(nodes[index+1]-nodes[index])
            count+=1
    upper=replace(pe,height=float(nodes[-1]-pe.position[2]))
    assert sim._msl_declared_face_geometry(upper,grid)['frac']==0
    for top in (nodes[-1]+.25*U,np.inf,np.nan):
        try:sim._msl_declared_face_geometry(replace(pe,height=float(top-pe.position[2])),grid)
        except ValueError:pass
        else:raise AssertionError(('out-of-range accepted',top))
print('Independent pure face-location cases:',count,'plus endpoint/range/nonfinite refusals')
