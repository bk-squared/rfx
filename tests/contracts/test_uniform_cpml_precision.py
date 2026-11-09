"""B2/B6: synthetic CPML box coefficients and original float32 bits."""
from pathlib import Path
import sys
import warnings

import jax
import numpy as np
import pytest

from tests.contracts.path_equivalence.builders import build


def solve(entry, precision):
    from rfx.stepping import uniform_loop
    states = []
    previous = sys.getprofile()

    def capture(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        if event == 'return' and frame.f_code is uniform_loop.run.__code__:
            states.append(frame.f_locals['final_carry']['fdtd'])

    sim = build(('_materials', 'eps'), 'run_uniform' if entry == 'run' else 'fwd_uniform',
                precision=precision)
    sys.setprofile(capture)
    try:
        result = (sim.run(n_steps=36, compute_s_params=False, skip_preflight=True) if entry == 'run'
                  else sim.forward(n_steps=36, checkpoint=False, skip_preflight=True))
        jax.block_until_ready(result.time_series)
        return result, states[-1]
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize('entry', ('run', 'forward'))
def test_uniform_cpml_float32_bitwise_original(entry):
    # Generated from git archive 95e66c479: uniform B1 dielectric CPML box,
    # explicit float32, 36 steps, checkpoint=False. Store only final fields
    # and probe series, viewed as uint32, for both entry points.
    reference = Path(__file__).with_name('uniform_cpml_float32_95e66c479.npz')
    try:
        with jax.enable_x64(False), warnings.catch_warnings(), np.load(reference) as expected:
            warnings.simplefilter('ignore', UserWarning)
            result, state = solve(entry, 'float32')
            for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz', 'time_series'):
                actual = np.asarray(result.time_series if name == 'time_series' else getattr(state, name))
                assert actual.dtype == np.float32
                np.testing.assert_array_equal(actual.view(np.uint32), expected[f'{entry}_{name}'],
                                              err_msg=f'B6 uniform {entry}.{name} differs from 95e66c479')
    finally:
        jax.clear_caches()


@pytest.mark.parametrize('entry', ('run', 'forward'))
def test_uniform_cpml_float64_coefficients(entry):
    from rfx.boundaries import cpml
    codes = {cpml.apply_cpml_e.__code__: ('E', ('_ce_ex', '_ce_ey', '_ce_ez')),
             cpml.apply_cpml_h.__code__: ('H', ('ch',))}
    observed = set()

    def inspect(frame, event, arg):
        if event == 'return' and frame.f_code in codes:
            half, names = codes[frame.f_code]
            assert getattr(frame.f_locals['dt'], 'dtype', None) == np.float64, f'B2 uniform {entry} CPML {half}.dt must be a float64 array'
            for name in names:
                for value in jax.tree.leaves(frame.f_locals[name]):
                    assert value.dtype == np.float64, f'B2 uniform {entry} CPML {half}.{name} dtype {value.dtype}; expected float64'
            observed.add(half)

    previous = sys.getprofile()
    try:
        with jax.enable_x64(True), warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            sys.setprofile(inspect)
            solve(entry, 'float64')
            assert observed == {'E', 'H'}, observed
    finally:
        sys.setprofile(previous)
        jax.clear_caches()
