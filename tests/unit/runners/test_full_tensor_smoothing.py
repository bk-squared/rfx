"""#1632 public full-tensor routing and unsupported-lane admission."""
import inspect
import re

import jax
import numpy as np
import pytest

from rfx.model.smoothing_admission import FULL_TENSOR_MESSAGE
from tests.unit.runners.test_forward_subpixel_smoothing import sphere_sim


def test_full_tensor_run_forward_and_diagonal_differ():
    options = dict(n_steps=40, skip_preflight=True, subpixel_smoothing='full_tensor')
    expected = np.asarray(sphere_sim().run(**options).time_series)
    actual = np.asarray(sphere_sim().forward(checkpoint=False, **options).time_series)
    diagonal = np.asarray(sphere_sim().run(**dict(options, subpixel_smoothing=True)).time_series)
    np.testing.assert_array_equal(actual, expected)
    difference = np.max(np.abs(actual - diagonal))
    print(f'full tensor vs True max field difference={difference:.17g}')
    assert difference > 0


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('lane', ['distributed', 'adi', 'subgridded', 'upml'])
def test_full_tensor_refused(entry, lane):
    sim = sphere_sim(graded=lane == 'graded', solver='adi' if lane == 'adi' else 'yee')
    options = {}
    if lane == 'distributed':
        devices = jax.devices('cpu')[:2]
        assert len(devices) == 2
        options = dict(devices=devices) if entry == 'run' else dict(distributed=True, devices=devices)
    if lane == 'subgridded':
        sim.add_refinement(z_range=(.004, .008), ratio=2)
    if lane == 'upml':
        sim._boundary = 'upml'
    with pytest.raises(NotImplementedError, match=re.escape(FULL_TENSOR_MESSAGE)) as info:
        getattr(sim, entry)(n_steps=1, skip_preflight=True, subpixel_smoothing='full_tensor', **options)
    assert str(info.value) == FULL_TENSOR_MESSAGE


def test_waveguide_s_driver_refuses_full_tensor():
    from tests.unit.sparams.test_waveguide_override_with_smoothing_refused import _sim
    with pytest.raises(NotImplementedError, match=re.escape(FULL_TENSOR_MESSAGE)):
        _sim().compute_waveguide_s_matrix(n_steps=1, subpixel_smoothing='full_tensor')


def test_msl_s_driver_option_is_not_reachable():
    from rfx.sparams.msl import compute_msl_s_matrix
    assert 'subpixel_smoothing' not in inspect.signature(compute_msl_s_matrix).parameters


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_full_tensor_shape_gradient_still_refuses(entry):
    from tests.unit.runners.test_shape_gradient_refusal import objective
    with pytest.raises(NotImplementedError, match='declared shape dimension'):
        jax.grad(lambda r: objective(r, entry=entry, smoothing='full_tensor'))(.0023)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_full_tensor_kerr_refuses(entry):
    with pytest.raises(NotImplementedError, match='subpixel_smoothing.*kerr_chi3'):
        getattr(sphere_sim(chi3=1e-20), entry)(
            n_steps=1, skip_preflight=True, subpixel_smoothing='full_tensor')
