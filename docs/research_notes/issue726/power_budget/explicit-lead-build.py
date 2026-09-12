"""Build-only audit of explicitly continuing the declared trace through CPML."""
import importlib.util
import json
import numpy as np
from rfx import Box
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
sp=importlib.util.spec_from_file_location('case','scripts/diagnostics/msl_probe_clearance_bias.py')
f=importlib.util.module_from_spec(sp); sp.loader.exec_module(f)
cv,arms,receipt=f.prepare_inputs()
old=arms['clean']; ro=f._realized(old); grid=ro.grid
nodes=np.asarray(coords_from_uniform_grid(grid).x)
original_box=cv.Box
count=[0]
def explicit_lead(lo,hi):
    if lo[0]==0 and hi[0]==cv.L_LINE+2*cv.PORT_MARGIN and lo[2]==hi[2]==cv.H_SUB:
        count[0]+=1
        return Box((float(nodes[0]),lo[1],lo[2]),(float(nodes[-1]),hi[1],hi[2]))
    return original_box(lo,hi)
cv.Box=explicit_lead
new=cv._build_sim(); assert count[0]==1
new._msl_ports=list(old._msl_ports)
new._msl_auto_offset_min={}; new._msl_auto_probe_spacing={}
rn=f._realized(new)
assert rn.grid.shape==grid.shape
p=receipt['ports']['clean']; xlo=int(np.argmin(abs(nodes-p[0]['probe_x_m'][0]))); xhi=int(np.argmin(abs(nodes-p[1]['probe_x_m'][0])))
unchanged=[bool(np.array_equal(np.asarray(a)[xlo:xhi+1],np.asarray(b)[xlo:xhi+1])) for a,b in zip(ro.edge_masks,rn.edge_masks)]
assert all(unchanged)
k=receipt['realized_metal']['plane_k']; j=29
mx0=np.asarray(ro.edge_masks[0]); mx1=np.asarray(rn.edge_masks[0])
report=dict(scope='build-only; no field result and no attribution of the observed side inflow',
            old_trace_corners=[old._geometry[1].shape.corner_lo,old._geometry[1].shape.corner_hi],
            explicit_trace_corners=[new._geometry[1].shape.corner_lo,new._geometry[1].shape.corner_hi],
            unchanged_dut_box_pec_components=unchanged,
            old_x_pad_ex_counts=[int(mx0[:grid.pad_x_lo,j,k].sum()),int(mx0[-grid.pad_x_hi:,j,k].sum())],
            explicit_x_pad_ex_counts=[int(mx1[:grid.pad_x_lo,j,k].sum()),int(mx1[-grid.pad_x_hi:,j,k].sum())],
            explicit_center_ex_indices=np.flatnonzero(mx1[:,j,k]).tolist(),
            source_declarations_unchanged=[f._source_spec(a)==f._source_spec(b) for a,b in zip(old._msl_ports,new._msl_ports)])
print(json.dumps(report,indent=2))
