"""Graded lumped S uses the uniform wave definition and one entry pipeline."""
import warnings
from dataclasses import replace

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec
from tests.contracts.test_graded_port_reference import constant_scene
from tests._x64_compat import enable_x64

BINS = np.array([3e9, 7e9, 13e9], np.float32)


def evaluate(sim, entry, *, bins=BINS, devices=None):
    kwargs = dict(n_steps=32, skip_preflight=True)
    if entry == 'forward':
        return sim.forward(**kwargs, checkpoint=False, port_s11_freqs=bins,
                           distributed=devices is not None, devices=devices)
    return sim.run(**kwargs, compute_s_params=True, s_param_freqs=bins, devices=devices)


def lumped_pair(graded):
    # A passive lumped port beside the driven one, so the record has an
    # off-diagonal (receive) entry as well as the driven diagonal.
    sim = constant_scene(graded, 'lumped')
    sim.add_port((6/1024, 3/1024, 2/1024), 'ez', impedance=75., excite=False)
    return sim


@pytest.mark.parametrize('pair', [False, True], ids=['one', 'driven_passive'])
@pytest.mark.parametrize('entry', ['forward', 'run'])
def test_binary_constant_lumped_s(entry, pair, record_property):
    build = lumped_pair if pair else (lambda nu: constant_scene(nu, 'lumped'))
    with enable_x64(False):
        uniform, graded = [evaluate(build(nu), entry) for nu in (False, True)]
    # The driven column (port 0); the other column is unmeasured (NaN) on graded,
    # and uniform run() fills it from its own separately driven scan.
    a, b = np.asarray(uniform.s_params)[:, :1], np.asarray(graded.s_params)[:, :1]
    delta = float(np.max(np.abs(a.astype(np.complex128) - b.astype(np.complex128))))
    ulp = float(np.spacing(np.float32(np.max(np.abs(a)))))
    bits = a.tobytes() == b.tobytes()
    record_property('bit_equal', bits)
    record_property('max_abs_dS', delta)
    record_property('ulp_of_peak', delta / ulp)
    print(f'B1 {entry}: bit_equal={bits}, max_abs_dS={delta:.12g}, ULP={delta / ulp:.9g}')
    assert bits or delta / ulp <= 9


def scene(case):
    sim = constant_scene(True, 'wire' if case == 'wire_lumped' else 'lumped')
    if case == 'passive':
        sim._ports[0] = replace(sim._ports[0], excite=False)
        sim.add_source((7/1024, 4/1024, 3/1024), 'ez')
    if case in ('lumped_wire', 'wire_lumped', 'two_driven'):
        sim.add_port((6/1024, 3/1024, 2/1024), 'ez', impedance=75.,
                     extent=2/1024 if case == 'lumped_wire' else None,
                     excite=case == 'two_driven',
                     **({'waveform': sim._ports[0].waveform} if case == 'two_driven' else {}))
    return sim


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize('case', ['lumped', 'lumped_wire', 'wire_lumped', 'passive'])
@pytest.mark.parametrize('default_bins', [False, True])
def test_equal_requests(case, default_bins):
    run = evaluate(scene(case), 'run', bins=None if default_bins else BINS)
    forward = evaluate(scene(case), 'forward', bins=np.asarray(run.freqs) if default_bins else BINS)
    same_bits(run.freqs, forward.freqs)
    same_bits(run.s_params, forward.s_params)
    assert run.lumped_port_sparams and forward.lumped_port_sparams
    for field in ('lumped_port_sparams', 'wire_port_sparams'):
        left, right = getattr(run, field) or (), getattr(forward, field) or ()
        assert len(left) == len(right)
        for (_, a), (_, b) in zip(left, right, strict=True):
            for x, y in zip(a, b, strict=True):
                same_bits(x, y)


def test_two_driven_refusal_prefix():
    errors = []
    for entry in ('forward', 'run'):
        with pytest.raises(ValueError) as caught:
            evaluate(scene('two_driven'), entry)
        errors.append(caught.value)
    assert type(errors[0]) is type(errors[1])
    assert str(errors[1]).startswith(str(errors[0]))


def test_s_off_keeps_lumped_records_only():
    # compute_s_params=False: records only (the no-argument default forms S,
    # test_default_run_forms_lumped_s_as_an_explicit_request).
    result = constant_scene(True, 'lumped').run(n_steps=32, skip_preflight=True,
                                              compute_s_params=False)
    assert result.s_params is None and result.lumped_port_sparams is None
    assert result.sparam_time_records[0].samples.shape == (32, 3)


@pytest.mark.distributed
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_distributed_lumped_s_is_published(entry):
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('requires two virtual CPU devices')
    result = evaluate(scene('lumped'), entry, devices=devices[:2])
    other = evaluate(scene('lumped'), 'forward' if entry == 'run' else 'run', devices=devices[:2])
    run = result if entry == 'run' else other
    assert len(run.state.ez.devices()) == 2
    assert result.s_params.shape == (1, 1, 3)
    same_bits(result.s_params, other.s_params)
    for a, b in zip(result.lumped_port_sparams[0][1], other.lumped_port_sparams[0][1], strict=True):
        same_bits(a, b)


@pytest.mark.distributed
def test_distributed_run_default_bins_return_lumped_s():
    # An explicit S request with no bins reads the default S bins on two devices,
    # as on one; it is never dropped to s_params=None.
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('requires two virtual CPU devices')
    multi = scene('lumped').run(n_steps=32, skip_preflight=True, compute_s_params=True,
                                devices=devices[:2])
    one = scene('lumped').run(n_steps=32, skip_preflight=True, compute_s_params=True)
    assert multi.s_params is not None
    np.testing.assert_array_equal(np.asarray(multi.freqs), np.asarray(one.freqs))
    assert np.asarray(multi.s_params).shape == np.asarray(one.s_params).shape


@pytest.mark.distributed
def test_distributed_two_driven_lumped_ports_refused_before_the_solve():
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('requires two virtual CPU devices')
    with pytest.raises(ValueError, match='isolated drive'):
        scene('two_driven').run(n_steps=32, skip_preflight=True, compute_s_params=True,
                                s_param_freqs=BINS, devices=devices[:2])


@pytest.mark.parametrize('entry', ['forward', 'run'])
def test_all_lumped_graded_record_takes_the_lumped_decomposition(entry, monkeypatch):
    # decompose_lumped_s_matrix delegates to the wire decomposition at one live
    # cell, so the two give the same numbers: the route, not a value, is pinned.
    import rfx.probes.probes as probes
    calls = []
    original = probes.decompose_lumped_s_matrix

    def counted(*args, **kwargs):
        calls.append(kwargs.get('v_ref') is not None)
        return original(*args, **kwargs)

    monkeypatch.setattr(probes, 'decompose_lumped_s_matrix', counted)
    with enable_x64(False):
        evaluate(lumped_pair(True), entry)
    assert calls == [True]


def test_default_run_forms_lumped_s_as_an_explicit_request():
    # No S argument: one driven lumped port on a graded mesh reads the 50
    # default bins, as uniform run() does, and equals the explicit request.
    default = scene('lumped').run(n_steps=32, skip_preflight=True)
    uniform = constant_scene(False, 'lumped').run(n_steps=32, skip_preflight=True)
    assert np.asarray(default.s_params).shape == (1, 1, 50)
    # The graded S default has always used the float64 host grid, within 2
    # float32 ULP of uniform's device grid (default_frequency_grid docstring).
    np.testing.assert_allclose(np.asarray(default.freqs), np.asarray(uniform.freqs),
                               rtol=2 * np.finfo(np.float32).eps, atol=0)
    explicit = scene('lumped').run(n_steps=32, skip_preflight=True, compute_s_params=True,
                                   s_param_freqs=np.asarray(default.freqs))
    same_bits(default.s_params, explicit.s_params)


@pytest.mark.parametrize('case', ['two_driven', 'plain_source', 'periodic', 'all_passive'])
def test_default_run_never_refuses_for_s_it_was_not_asked_for(case):
    sim = scene('two_driven' if case == 'two_driven' else 'lumped')
    if case == 'all_passive':
        sim._ports[0] = replace(sim._ports[0], excite=False)
    if case == 'plain_source':
        sim.add_source((7/1024, 4/1024, 3/1024), 'ez')
    if case == 'periodic':
        sim = Simulation(freq_max=20e9, domain=(10/1024, 8/1024, 6/1024), dx=1/1024,
                         boundary=BoundarySpec(x='periodic', y='pec', z='pec'),
                         dz_profile=np.full(6, 1/1024))
        sim.add_port((3/1024, 3/1024, 2/1024), 'ez', impedance=50.)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = sim.run(n_steps=32, skip_preflight=True)
    assert result.s_params is None
    isolated = [w for w in caught if 'isolated drive' in str(w.message)]
    assert bool(isolated) == (case == 'two_driven')
    # A plain source warns as on a uniform mesh (fields, S=None).
    plain = [w for w in caught if 'do not support plain sources' in str(w.message)]
    assert bool(plain) == (case == 'plain_source')
    if case == 'two_driven':
        # An explicit request graded cannot form still refuses.
        with pytest.raises(ValueError, match='isolated drive'):
            scene('two_driven').run(n_steps=32, skip_preflight=True, compute_s_params=True)


@pytest.mark.distributed
def test_distributed_default_run_forms_lumped_s():
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('requires two virtual CPU devices')
    multi = scene('lumped').run(n_steps=32, skip_preflight=True, devices=devices[:2])
    one = scene('lumped').run(n_steps=32, skip_preflight=True)
    assert np.asarray(multi.s_params).shape == (1, 1, 50)
    np.testing.assert_array_equal(np.asarray(multi.freqs), np.asarray(one.freqs))
