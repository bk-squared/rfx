from __future__ import annotations
import ast
import copy
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BASE = Path('/root/workspace/bk-workspace/.801-measure')
OUT = BASE / 'ports'
SRC = BASE / 'src-main'
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(SRC / 'tests'))
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'mpl_config'))
os.environ.setdefault('XDG_CACHE_HOME', str(OUT / 'cache'))

def mutation_guard(event, args):
    if event == 'open':
        path, mode, flags = args
        if not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            return
        if isinstance(path, int):
            return
        p = Path(os.fsdecode(path)).resolve()
        if not p.is_relative_to(OUT) or p.exists():
            raise PermissionError(f'NEW_FILES_ONLY: {event} {p}')
    elif event == 'os.mkdir':
        p = Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(OUT):
            raise PermissionError(f'PORTS_ONLY: {event} {p}')
    elif event in {'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink', 'os.chmod', 'os.chown'}:
        raise PermissionError(f'NO_MUTATION: {event} {args}')

sys.addaudithook(mutation_guard)
import numpy as np
import jax
import jax.numpy as jnp
from rfx import Simulation
from rfx.boundaries.pec import realized_pec_edge_masks
import _example_fidelity_lib as lib

def serial(v):
    if dataclasses.is_dataclass(v):
        return {f.name: serial(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, dict):
        return {str(k): serial(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [serial(x) for x in v]
    if isinstance(v, (np.ndarray, jax.Array)):
        return serial(np.asarray(v).tolist())
    if isinstance(v, np.generic):
        return serial(v.item())
    if isinstance(v, complex):
        return {'real': v.real, 'imag': v.imag}
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (str, float, int, bool)) or v is None:
        return v
    return repr(v)

def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(serial(value), f, indent=2)
        f.write('\n')

def save_npz(path, **arrays):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_module(rel):
    name = '_ports_' + hashlib.sha256(rel.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, SRC / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def catalog():
    rows = []
    for rel, entry in sorted(lib.CLASSIFICATION.items()):
        if entry.kind == 'audited':
            for b in entry.builders:
                for v in b.variants:
                    rows.append(dict(kind='audited', path=rel, builder=b.fn, label=v.label))
    def extra(path, fn, label='default', kwargs=None, **more):
        rows.append(dict(kind='additional', path=path, builder=fn, label=label, kwargs=kwargs or {}, **more))
    unit = 'tests/unit/sparams/'
    for stem, fn in [('test_msl_internal_probe_advisories','_thru'),
                     ('test_msl_passivity_enforcement','_thru'),
                     ('test_msl_plane_primitives_smoke','_build_thru_line'),
                     ('test_msl_probe_offset_interval','_open_thru_sim')]:
        extra(unit+stem+'.py', fn)
    extra(unit+'test_msl_sheet_threading.py','build_msl_thru')
    extra(unit+'test_msl_sheet_threading.py','build_msl_thru','pec_sheet',{'sheet': ['pec']})
    for axis in 'xy':
        for lane in ('uniform','nonuniform'):
            extra(unit+'test_msl_sparse_dft.py','_thru_sim',axis+'_'+lane,dict(axis=axis,lane=lane))
    for fn in ('test_msl_thru_line_passive_gate','test_msl_thru_line_eigenmode_gate'):
        extra(unit+'test_msl_port_integration.py',fn,capture=True)
    for length in (0.008,0.010,0.012):
        extra(unit+'test_msl_port_integration.py','_run_msl_thru',str(length),{'l_line':length},capture=True)
    extra(unit+'test_coax_two_port_smatrix.py','_sim')
    extra(unit+'test_coax_msl_transition.py','_build_coax_msl_transition_sim')
    extra(unit+'test_waveguide_lane_pad_continuation.py','_guide','WR90_slab',{'with_slab':True})
    extra('scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py','_build_sim','cv20_rfx')
    for i, row in enumerate(rows):
        row['index'] = i
        row['id'] = f"{row['path']}::{row['builder']}::{row['label']}"
    return rows

class Captured(BaseException):
    pass

def build(row):
    mod = load_module(row['path'])
    if row['kind'] == 'audited':
        b = next(x for x in lib.CLASSIFICATION[row['path']].builders if x.fn == row['builder'])
        v = next(x for x in b.variants if x.label == row['label'])
        with lib.build_only():
            result = getattr(mod,b.fn)(**v.kwargs(mod))
        return (result if b.result_index is None else result[b.result_index]), mod, None
    if row.get('capture'):
        from unittest.mock import patch
        captured = {}
        def intercept(sim, *a, **kw):
            captured.update(sim=sim,args=a,kwargs=kw)
            raise Captured()
        try:
            with patch.object(Simulation,'compute_msl_s_matrix',intercept):
                getattr(mod,row['builder'])(**row['kwargs'])
        except Captured:
            return captured['sim'], mod, captured['kwargs']
        raise RuntimeError('builder reached no compute_msl_s_matrix call')
    with lib.build_only():
        sim = getattr(mod,row['builder'])(**row['kwargs'])
    return sim, mod, None

def grid_for(sim):
    try:
        return sim._build_grid(), False, None
    except ValueError as exc:
        text = str(exc)
        if any(getattr(sim,a,None) is not None for a in ('_dx_profile','_dy_profile','_dz_profile')):
            return sim._build_nonuniform_grid(), True, text
        raise

def assemble(sim, grid, nu):
    sheets, wires, impedance_sheets = [], [], []
    fn = sim._assemble_materials_nu if nu else sim._assemble_materials
    result = fn(grid,pec_sheets=sheets,pec_wires=wires,sheet_specs=impedance_sheets)
    cells = result[3]
    if cells is None:
        cells = jnp.zeros(grid.shape, dtype=bool)
    edges = realized_pec_edge_masks(cells,sheets,wires,periodic=sim._periodic_flags())
    return result, np.asarray(cells), tuple(np.asarray(a) for a in edges), sheets, wires, impedance_sheets

def pads(grid):
    return {a+'_'+s:int(getattr(grid,'pad_'+a+'_'+s)) for a in 'xyz' for s in ('lo','hi')}

def face_counts(array, grid):
    arr = np.asarray(array, dtype=bool)
    ans = {}
    for ax,a in enumerate('xyz'):
        for side in ('lo','hi'):
            key=a+'_'+side
            p=pads(grid)[key]
            if not p:
                continue
            sl=[slice(None)]*3
            sl[ax] = p if side=='lo' else arr.shape[ax]-p-1
            last=int(arr[tuple(sl)].sum())
            sl[ax] = slice(0,p) if side=='lo' else slice(arr.shape[ax]-p,arr.shape[ax])
            ans[key]={'last_interior':last,'absorber':int(arr[tuple(sl)].sum())}
    return ans

def measurement(sim, grid, nu):
    result,cells,edges,sheets,wires,imp=assemble(sim,grid,nu)
    domain=tuple(float(x) for x in sim._domain)
    rows=[]
    for collection in ('_geometry','_thin_conductors'):
        for i,entry in enumerate(getattr(sim,collection)):
            solo=copy.copy(sim)
            solo._geometry=[entry] if collection=='_geometry' else []
            solo._thin_conductors=[entry] if collection=='_thin_conductors' else []
            a,c,e,ss,ww,ii=assemble(solo,grid,nu)
            bounds=entry.shape.bounding_box()
            lo,hi=[list(map(float,x)) for x in bounds]
            mat=sim._resolve_material(entry.material_name) if collection=='_geometry' else None
            conductor=collection=='_thin_conductors' or mat.sigma >= sim._PEC_SIGMA_THRESHOLD
            cc=face_counts(c,grid)
            ec={a:face_counts(v,grid) for a,v in zip('xyz',e)}
            dc=face_counts(np.asarray(a[0].eps_r)!=1,grid)
            reach={f:bool(lo['xyz'.index(f[0])] <= domain['xyz'.index(f[0])]*1e-9 if f.endswith('lo') else hi['xyz'.index(f[0])] >= domain['xyz'.index(f[0])]*(1-1e-9)) for f in cc}
            affected=[f for f in cc if conductor and reach[f] and cc[f]['absorber']==0 and sum(ec[a][f]['absorber'] for a in 'xyz')==0]
            rows.append(dict(collection=collection,index=i,shape_type=type(entry.shape).__name__,material=getattr(entry,'material_name',repr(entry)),conductor=int(conductor),bounds_m=[lo,hi],reaches=reach,volume_cells=cc,edges=ec,eps_not_1=dc,volume_total=int(c.sum()),edge_totals=[int(x.sum()) for x in e],sheet_count=len(ss),wire_count=len(ww),impedance_sheet_count=len(ii),affected_faces=affected))
    return dict(domain_m=domain,grid_shape=list(grid.shape),cells=int(np.prod(grid.shape)),dt_s=float(grid.dt),face_pads=pads(grid),boundary={a+'_'+s:getattr(getattr(sim._boundary_spec,a),s) for a in 'xyz' for s in ('lo','hi')},entities=rows,affected=int(any(r['affected_faces'] for r in rows)),assembly_tuple_length=len(result),total_cells=face_counts(cells,grid),total_edges={a:face_counts(e,grid) for a,e in zip('xyz',edges)},total_eps_not_1=face_counts(np.asarray(result[0].eps_r)!=1,grid))

def lane(sim):
    counts={key:len(getattr(sim,key,[])) for key in ('_msl_ports','_coaxial_ports','_waveguide_ports','_ports','_wire_ports')}
    if counts['_coaxial_ports'] and counts['_msl_ports']:
        return 'coax_msl_transition',counts
    if counts['_coaxial_ports']:
        return 'coax_two_port',counts
    if counts['_msl_ports']:
        return 'msl_two_port',counts
    if counts['_waveguide_ports']:
        return 'waveguide',counts
    if counts['_ports'] or counts['_wire_ports']:
        return 'lumped_wire',counts
    return 'none',counts
