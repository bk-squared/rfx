"""Read-only diagnosis of the rejected G4 candidate; no candidate tuning."""
import importlib.util
import os
from pathlib import Path
import jax
import numpy as np

root = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('g4_tests', root / 'tests/unit/boundaries/test_cpml_localization.py')
os.environ['RFX_G4_REJECTED_CANDIDATE'] = '1'
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
print('jax', jax.__version__, 'backend', jax.default_backend())
for implementation, label in [(t.old, 'baseline'), (t.cpml, 'candidate')]:
    fn = jax.jit(lambda: t.runner('uniform8', implementation))
    compiled = fn.lower().compile()
    (root / f'validation/research/nu_cost/g4/{label}.hlo.txt').write_text(compiled.as_text())

# Scan histories locate the first divergent step, without changing arithmetic.
a = jax.jit(lambda: t.runner('uniform8', t.old, history=True)[1])()
b = jax.jit(lambda: t.runner('uniform8', t.cpml, history=True)[1])()
first = 200
for group_a, group_b in zip(a, b):
    for name in group_a._fields:
        x, y = np.asarray(getattr(group_a, name)), np.asarray(getattr(group_b, name))
        locs = np.argwhere(x != y)
        if len(locs):
            index = tuple(locs[0])
            first = min(first, index[0])
            print(name, 'first', index, 'old', x[index], 'new', y[index], 'old_bits', x[index].view(np.uint32), 'new_bits', y[index].view(np.uint32))
print('first divergent completed step', first + 1)
np.savez(root / 'validation/research/nu_cost/g4/first_divergence.npz', **{
    f'{label}_{group}_{name}': np.asarray(getattr(data, name))[first - 1:first + 1]
    for label, results in [('baseline', a), ('candidate', b)]
    for group, data in zip(['field', 'psi'], results)
    for name in data._fields
})
# Replay the first step from the identical carry, separating H/Yee-E/CPML-E.
from rfx.core.yee import update_h, update_e, EPS_0
snap = np.load(root / 'validation/research/nu_cost/g4/first_divergence.npz')
st = t.init_state((29,) * 3)._replace(**{n: t.jnp.asarray(snap['baseline_field_' + n][0]) for n in t.init_state((29,) * 3)._fields})
grid = t.fixture('uniform8')
params, ps = t.cpml.init_cpml(grid)
ps = ps._replace(**{n: t.jnp.asarray(snap['baseline_psi_' + n][0]) for n in ps._fields})
mat = t.init_materials((29,) * 3)
h = jax.jit(lambda s: update_h(s, mat, grid.dt, grid.dx))(st)
h, ps = jax.jit(lambda s, p: t.old.apply_cpml_h(s, params, p, grid, materials=mat))(h, ps)
e = jax.jit(lambda s: update_e(s, mat, grid.dt, grid.dx))(h)
for impl, label in [(t.old, 'baseline'), (t.cpml, 'candidate')]:
    out, p = jax.jit(lambda s, p: impl.apply_cpml_e(s, params, p, grid, materials=mat))(e, ps)
    for field, coord, pn, pc in [('ex',(13,13,24),'psi_ex_zhi',(3,13,13)), ('ey',(12,13,24),'psi_ey_zhi',(3,13,12))]:
        val = np.float32(getattr(out,field)[coord]); prev = np.float32(getattr(e,field)[coord]); psi = np.float32(getattr(p,pn)[pc])
        ce = np.float32(grid.dt/EPS_0)
        sign = -1 if field == 'ex' else 1
        separate = np.float32(prev + np.float32(np.float32(sign*ce)*psi))
        fused = np.float32(np.float64(prev) + np.float64(np.float32(sign*ce))*np.float64(psi))
        print('isolated E', label, field, 'input', prev, 'psi', psi, 'ce', ce, 'output', val, 'bits', val.view(np.uint32), 'separate', separate, separate.view(np.uint32), 'fused', fused, fused.view(np.uint32))
from itertools import product
f = np.float32
fma = lambda x, y, z: f(np.float64(x)*np.float64(y)+np.float64(z))
for field, coord, fa, aa, fb, ab, pn, pc, sign in [
    ('ex',(13,13,24),'hz',1,'hy',2,'psi_ex_zhi',(3,13,13),-1),
    ('ey',(12,13,24),'hx',2,'hz',0,'psi_ey_zhi',(3,13,12),1),
]:
    hvals = {n: snap['baseline_field_'+n][1] for n in ('hx','hy','hz')}
    ca, cbidx = list(coord), list(coord)
    ca[aa]-=1; cbidx[ab]-=1
    d1 = f(hvals[fa][coord]-hvals[fa][tuple(ca)])
    d2 = f(hvals[fb][coord]-hvals[fb][tuple(cbidx)])
    inv = f(1/f(grid.dx)); ce = f(grid.dt/EPS_0)
    previous = f(snap['baseline_field_'+field][0][coord])
    psi = f(snap['baseline_psi_'+pn][1][pc])
    print('scalar operands',field,'d1',d1,'d2',d2,'inv',inv,'ce',ce,'previous',previous,'psi',psi)
    for curl_mode, yee_fused, correction_fused in product(range(3), (False,True), (False,True)):
        curl = [lambda: f(f(d1*inv)-f(d2*inv)),lambda: fma(d1,inv,-f(d2*inv)),lambda: fma(-d2,inv,f(d1*inv))][curl_mode]()
        yee = fma(ce,curl,previous) if yee_fused else f(previous+f(ce*curl))
        out = fma(f(sign*ce),psi,yee) if correction_fused else f(yee+f(f(sign*ce)*psi))
        print('scalar replay', field, curl_mode, yee_fused, correction_fused, out, out.view(np.uint32))
for impl, label in [(t.old, 'baseline'), (t.cpml, 'candidate')]:
    out = jax.jit(lambda: t.runner('uniform8', impl, steps=13)[0])()[0]
    for field, coord in [('ex',(13,13,24)),('ey',(12,13,24))]:
        val = np.float32(getattr(out,field)[coord])
        print('13-step final without history', label, field, val, val.view(np.uint32))
