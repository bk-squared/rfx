"""Issue 1683: rematerialization must not lengthen or shorten a record."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.probes.probes import flux_spectrum


STEPS = 60
CHUNKS = (7, 40, 30)
QUANTITIES = ('probe', 's11', 'dft', 'flux', 'flux_dfts', 'E', 'H')


def _scene(*, distributed=False):
    # Issue-1683 scene, retained at its original size and source duration.
    dx = 1e-3
    profile = np.full(24, dx)
    profile[10:14] = 0.5e-3
    sim = Simulation(freq_max=20e9, domain=(float(profile.sum()), 24*dx, 24*dx),
                     dx=dx, boundary='cpml', cpml_layers=6, dx_profile=profile)
    pulse = GaussianPulse(f0=8e9, bandwidth=0.8)
    sim.add_port(position=(6*dx, 12*dx, 12*dx), component='ez',
                 impedance=50., extent=2*dx, waveform=pulse)
    if not distributed:
        sim.add_dft_plane_probe(axis='x', coordinate=8*dx, component='ez',
                               freqs=jnp.asarray([6e9, 8e9, 10e9]))
        sim.add_flux_monitor(axis='x', coordinate=8*dx,
                             freqs=jnp.asarray([6e9, 8e9, 10e9]))
    sim.add_probe((8*dx, 12*dx, 12*dx), 'ez')
    return sim


def _forward(sim, chunk, *, distributed=False, eps=None):
    kwargs = dict(distributed=True, devices=jax.devices('cpu')[:2]) if distributed else {}
    kwargs['port_s11_freqs'] = jnp.asarray([6e9, 8e9, 10e9])
    return sim.forward(n_steps=STEPS, checkpoint=True, checkpoint_every=chunk,
                       eps_override=eps, skip_preflight=True, **kwargs)


@pytest.fixture(scope='module', params=('single', 'distributed'))
def records(request):
    distributed = request.param == 'distributed'
    if distributed and len(jax.devices('cpu')) < 2:
        pytest.skip('requires two host CPU devices, as configured by root conftest')
    if distributed:
        import rfx.runners._distributed_nu_wrapped as runner
        name = 'run_prepared'
    else:
        import rfx.runners.nonuniform as runner
        name = 'run_nonuniform'
    original = getattr(runner, name)
    captured = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    results = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, name, capture)
        sim = _scene(distributed=distributed)
        for chunk in (None, *CHUNKS):
            public = _forward(sim, chunk, distributed=distributed)
            raw = captured.pop()
            state = raw['final_state_sharded' if distributed else 'state']
            values = dict(probe=np.asarray(public.time_series), s11=np.asarray(public.s_params),
                          E=np.stack([np.asarray(getattr(state, k)) for k in ('ex', 'ey', 'ez')]),
                          H=np.stack([np.asarray(getattr(state, k)) for k in ('hx', 'hy', 'hz')]))
            if not distributed:
                mon = raw['flux_monitors'][0]
                values.update(dft=np.asarray(raw['dft_planes'][0].accumulator),
                              flux=np.asarray(flux_spectrum(mon)),
                              flux_dfts=np.stack([np.asarray(getattr(mon, k)) for k in
                                                  ('e1_dft', 'e2_dft', 'h1_dft', 'h2_dft')]))
            results[chunk] = values
    yield request.param, results
    jax.clear_caches()


@pytest.mark.parametrize('chunk', CHUNKS)
@pytest.mark.parametrize('quantity', QUANTITIES)
def test_forward_record_length(records, chunk, quantity, record_property):
    lane, values = records
    if quantity not in values[None]:
        pytest.skip('distributed NU forward refuses DFT planes and flux monitors')
    reference, actual = values[None][quantity], values[chunk][quantity]
    assert reference.shape == actual.shape
    assert np.all(np.isfinite(reference)) and np.all(np.isfinite(actual))
    peak = max(float(np.max(np.abs(reference))), float(np.max(np.abs(actual))))
    assert peak > 0, f'{quantity} must be excited'
    delta = float(np.max(np.abs(actual.astype(np.complex128)-reference.astype(np.complex128))))
    ulp = float(np.spacing(np.float32(peak)))
    bitwise = reference.tobytes() == actual.tobytes()
    measurement = dict(lane=lane, chunk=chunk, quantity=quantity, bitwise=bitwise,
                       peak=peak, difference=delta, peak_ulps=delta/ulp)
    record_property('measurement', json.dumps(measurement))
    print(json.dumps(measurement))
    # Declared before the run: the same steps in the same order, so bitwise.
    bar = 0.0
    assert delta <= bar, measurement


def test_s11_gradient_record_length(record_property):
    sim = _scene()
    grid = sim._build_nonuniform_grid()
    base = jnp.ones(grid.shape, dtype=jnp.float32)
    center = [grid.index_of(axis, pos) for axis, pos in enumerate((.008, .012, .012))]
    box = tuple(slice(i-1, i+2) for i in center)

    def loss(eps, chunk):
        result = _forward(sim, chunk, eps=base.at[box].set(eps))
        return jnp.abs(result.s_params.reshape(-1)[1]) ** 2

    reference = float(jax.grad(lambda a: loss(a, None))(jnp.float32(2.)))
    actual = float(jax.grad(lambda a: loss(a, 40))(jnp.float32(2.)))
    assert np.isfinite(reference) and np.isfinite(actual) and reference != 0
    relative = abs(actual-reference)/max(abs(reference), abs(actual))
    measurement = dict(lane='single', chunk=40, quantity='gradient',
                       reference=reference, actual=actual, relative=relative)
    record_property('measurement', json.dumps(measurement))
    print(json.dumps(measurement))
    assert relative <= 1e-4, measurement
    jax.clear_caches()


def test_distributed_s11_gradient_record_length():
    if len(jax.devices('cpu')) < 2:
        pytest.skip('requires two host devices')
    sim = _scene(distributed=True)
    base = jnp.ones(sim._build_nonuniform_grid().shape, dtype=jnp.float32)
    def loss(eps, chunk):
        result = _forward(sim, chunk, eps=eps, distributed=True)
        return jnp.sum(jnp.abs(result.s_params) ** 2)
    one = np.asarray(jax.grad(lambda e: loss(e, None))(base))
    multi = np.asarray(jax.grad(lambda e: loss(e, 40))(base))
    assert np.max(np.abs(one)) > 0
    assert np.max(np.abs(multi - one)) / np.max(np.abs(one)) <= 1e-4
    jax.clear_caches()


def test_a_chunk_above_half_the_record_still_leaves_two_full_segments():
    """One full segment plus a tail keeps the whole per-step tape (no memory
    saved), so the effective chunk always leaves at least two."""
    from rfx.stepping.remat import effective_chunk
    for n in range(2, 300):
        for chunk in range(1, n):
            used = effective_chunk(n, chunk)
            assert 1 <= used <= chunk and n // used >= 2, (n, chunk, used)
    assert effective_chunk(60, None) is None and effective_chunk(60, 60) is None
