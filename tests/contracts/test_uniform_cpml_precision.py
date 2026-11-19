"""Synthetic box coefficient precision and float32 field and probe rounding."""
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


def check_accumulated_law(result, state, expected, record_property):
    """Scale each field by its family peaks and both Ez probes by the running E peaks."""
    for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz', 'time_series'):
        actual = np.asarray(result.time_series if name == 'time_series' else getattr(state, name))
        reference = expected[name]
        assert actual.dtype == np.float32
        assert actual.shape == reference.shape
        delta = np.abs(actual.astype(np.float64) - reference)
        if name == 'time_series':
            error = delta.max(axis=1)
            allowance = 1.2e-7 * np.cumsum(expected['peak_E'], dtype=np.float64)
        else:
            error = delta.max()
            family = 'peak_E' if name.startswith('e') else 'peak_H'
            allowance = 1.2e-7 * np.sum(expected[family], dtype=np.float64)
        ratios = np.divide(error, allowance, where=allowance != 0,
                           out=np.where(error == 0, 0., np.inf))
        ratio = float(ratios.max())
        record_property(f'{name}_ratio', ratio)
        assert ratio <= 1, (name, ratio)


@pytest.mark.parametrize('entry', ('run', 'forward'))
def test_uniform_cpml_float32_within_accumulated_law(entry, record_property):
    """Bound field and probe errors by 1.2e-7 times accumulated float64 family peaks."""
    reference = Path(__file__).with_name('uniform_cpml_float64_reference.npz')
    try:
        with jax.enable_x64(False), warnings.catch_warnings(), np.load(reference) as expected:
            warnings.simplefilter('ignore', UserWarning)
            result, state = solve(entry, 'float32')
            jax.block_until_ready(result.time_series)
            check_accumulated_law(result, state, expected, record_property)
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


def test_coefficient_dt_without_float64_field_keeps_python_float():
    from rfx.boundaries.electric_coefficient import coefficient_dt
    with jax.enable_x64(True):
        for field_dtype in (None, np.dtype('float32')):
            assert type(coefficient_dt(np.float64(1e-12), field_dtype)) is float
