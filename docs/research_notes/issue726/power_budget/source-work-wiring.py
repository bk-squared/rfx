"""Stop at the actual uniform runner entry; never advance a field."""
import importlib.util,inspect,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import jax,jax.numpy as jnp
import rfx.simulation as engine
from rfx.core.yee import EPS_0
p=Path('docs/research_notes/issue726/power_budget/record_source_work.py')
spec=importlib.util.spec_from_file_location('recorder',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sim,grid,plan=m.build();signature=inspect.signature(engine.run);receipt={}
class Recorded(BaseException):pass

def stop(*args,**kwargs):
    bound=signature.bind(*args,**kwargs).arguments
    assert bound['grid'].shape==tuple(plan['grid_shape'])
    materials=bound['materials'];n=bound['n_steps']
    assert n==grid.num_timesteps(100)
    probe_specs=bound['probes']
    for port in plan['ports']:
        for column,cell in zip(port['point_columns'],port['cells']):
            q=probe_specs[column]
            assert [q.i,q.j,q.k]==cell and q.component=='ez'
        index=tuple(np.asarray(port['cells']).T)
        np.testing.assert_array_equal(np.asarray(materials.sigma)[index],port['added_sigma_s_per_m'])
    assert not bound.get('mag_sources')
    src=bound['sources'];target=plan['ports'][0]
    assert len(src)==len(target['cells'])==72
    assert {(s.i,s.j,s.k) for s in src}==set(map(tuple,target['cells']))
    u=np.asarray(jax.vmap(sim._msl_ports[0].waveform)(jnp.arange(n,dtype=jnp.float32)*grid.dt),dtype=np.float64)
    expected_g={tuple(c):g for c,g in zip(target['cells'],target['profile_e'])}
    errors=[]
    for s in src:
        c=(s.i,s.j,s.k);assert s.component=='ez'
        eps=float(materials.eps_r[c])*EPS_0;sigma=float(materials.sigma[c])
        cb=(grid.dt/eps)/(1+sigma*grid.dt/(2*eps))
        expected=cb*expected_g[c]*u
        actual=np.asarray(s.waveform,dtype=np.float64)
        errors.append(float(np.max(abs(actual-expected))/np.max(abs(expected))))
    assert max(errors)<2e-6,max(errors)
    receipt.update(scope='actual runner setup intercepted before any field step; source shape equality only to f32 precision',
                   point_probes_checked=sum(len(p['cells']) for p in plan['ports']),
                   active_source_cells=len(src),source_relative_peak_error=max(errors),steps=n,
                   magnetic_source_count=0,source_sigmas_match=True,point_cell_roundtrips_match=True)
    raise Recorded

with patch.object(engine,'run',side_effect=stop):
    try:
        sim.compute_msl_s_matrix(freqs=np.asarray(plan['requested_freqs_hz']),num_periods=100,enforce_passivity=False)
    except Recorded:pass
assert receipt
Path('.git/issue726-source-work-wiring.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
