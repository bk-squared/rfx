"""Localized ADE state adds no domain carry/copy or new fusion outputs (E4)."""
import gc
import json
from pathlib import Path
import sys

import jax
from jax.extend.core import jaxpr_as_fun
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import init_lorentz, lorentz_pole
from rfx.nonuniform import make_nonuniform_grid, run_nonuniform
from rfx.simulation import run

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'perf'))
from step_cost_census import compiled_opcode_counts

# Measured on main 0bdf8ccd9, uniform and graded. Debye's three inherited
# domain-sized curl-pad fusions are also present before state cropping.
MAIN_FUSIONS = {None: 76, 'debye': 79, 'lorentz': 76}


def scans(jaxpr):
    jaxpr = getattr(jaxpr, 'jaxpr', jaxpr)
    for eq in jaxpr.eqns:
        if eq.primitive.name == 'scan':
            yield eq
        for value in eq.params.values():
            if hasattr(value, 'jaxpr') or hasattr(value, 'eqns'):
                yield from scans(value)


def step_counts(model, path):
    n, dx = 24, .001
    if path == 'uniform':
        grid = Grid(freq_max=3e9, domain=(n * dx,) * 3, dx=dx, cpml_layers=8)
    else:
        widths = dx * np.array([1.] + [1.05, .95] * 11 + [1.])
        grid = make_nonuniform_grid((n * dx,) * 2, widths, dx, cpml_layers=8)
    mats = init_materials(grid.shape)
    mask = np.zeros(grid.shape, bool)
    mask[12:28, 12:28, 12:28] = True
    options = {}
    if model == 'debye':
        options['debye'] = init_debye([DebyePole(2., 1e-10)], mats, grid.dt, mask)
    elif model == 'lorentz':
        options['lorentz'] = init_lorentz([lorentz_pole(2., 1e10, 1e9)], mats, grid.dt, mask)
    def simulate():
        if path == 'uniform':
            return run(grid, mats, 2, boundary='cpml', return_state=True, **options).state
        return run_nonuniform(grid, mats, 2, **options)['state']
    eq = max(scans(jax.make_jaxpr(simulate)()), key=lambda eq: eq.params['num_carry'])
    start, count = eq.params['num_consts'], eq.params['num_carry']
    cells = int(np.prod(np.asarray(grid.shape) - 1))
    carries = sum(int(np.prod(v.aval.shape)) >= cells for v in eq.invars[start:start + count])
    body = eq.params['jaxpr']
    arguments = [jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in body.jaxpr.invars]
    executable = jax.jit(jaxpr_as_fun(body)).lower(*arguments).compile()
    counts = compiled_opcode_counts(executable.as_text(), cells)
    return {'carry': carries, 'copy': counts.get('copy', 0), 'fusion': counts.get('fusion', 0)}


@pytest.mark.parametrize('path', ('uniform', 'graded'))
@pytest.mark.parametrize('model', ('debye', 'lorentz'))
def test_poles_add_no_domain_carry_or_copy(model, path):
    bare, pole = step_counts(None, path), step_counts(model, path)
    print('E4 ' + json.dumps(dict(model=model, path=path, bare=bare, pole=pole)), flush=True)
    assert pole['carry'] == bare['carry']
    assert pole['copy'] == bare['copy']
    assert bare['fusion'] <= MAIN_FUSIONS[None]
    assert pole['fusion'] <= MAIN_FUSIONS[model]
    jax.clear_caches()
    gc.collect()
