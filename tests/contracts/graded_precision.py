"""Realized graded carry checks for the synthetic box precision contract."""
from functools import wraps
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Box, DebyePole, Simulation, drude_pole
from rfx.sources.sources import GaussianPulse


def check_graded_storage(monkeypatch, entry, precision):
    import rfx.stepping.graded as graded
    import rfx.stepping.graded_setup as setup

    expected = np.dtype(precision)
    complex_dtype = np.dtype('complex128' if precision == 'float64' else 'complex64')
    observed = set()
    initialize, make_step = setup.init_state, graded.make_graded_step

    def check(key, tree, dtype):
        arrays = [a for a in jax.tree.leaves(tree)
                  if hasattr(a, 'dtype') and np.issubdtype(a.dtype, np.inexact)]
        assert arrays, f'{entry}: missing {key} carry'
        for a in arrays:
            assert a.dtype == dtype, f'{entry}: {key} carry dtype {a.dtype}; expected {dtype}'
        observed.add(key)

    def init(*args, **kwargs):
        state = initialize(*args, **kwargs)
        check('initialized fields', state, expected)
        return state

    def builder(ctx, *args, **kwargs):
        if precision == 'float64':
            for name in ('dt', 'inv_dx', 'inv_dy', 'inv_dz', 'inv_dx_h', 'inv_dy_h', 'inv_dz_h'):
                assert getattr(ctx.grid, name).dtype == expected, name
            assert ctx.dt.dtype == expected
            assert all(a.dtype == expected for a in jax.tree.leaves(ctx.cpml_params))
        step = make_step(ctx, *args, **kwargs)

        @wraps(step)
        def checked(carry, xs, invariants):
            # Inspect the incoming carry, before the step can promote a leaf.
            for key in ('fdtd', 'cpml'):
                check(key, carry.get(key), expected)
            if entry == 'graded_tfsf':
                check('tfsf', carry.get('tfsf'), expected)
            else:
                for key in ('debye', 'lorentz', 'rlc_states'):
                    check(key, carry.get(key), expected)
                for key in ('wire_sparams', 'ntff'):
                    check(key, carry.get(key), complex_dtype)
            return step(carry, xs, invariants)
        return checked

    monkeypatch.setattr(setup, 'init_state', init)
    monkeypatch.setattr(graded, 'make_graded_step', builder)
    try:
        with jax.enable_x64(precision == 'float64'), warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            sim = Simulation(freq_max=8e9, domain=(.024, .018, .018), dx=.003,
                             dz_profile=np.full(6, .003), boundary='cpml',
                             cpml_layers=2, precision=precision)
            sim.add_probe((.012, .009, .009), 'ez')
            freqs = np.array([2e9, 3e9], dtype=np.float32)
            if entry == 'graded_tfsf':
                sim.add_tfsf_source(f0=4e9, bandwidth=.5, margin=1)
            else:
                sim.add_material('dispersive', eps_r=2.,
                                 debye_poles=[DebyePole(.2, 1e-11)],
                                 lorentz_poles=[drude_pole(2*np.pi*3e9, 2.300874e8)])
                sim.add(Box((.009, .006, .006), (.015, .012, .012)), material='dispersive')
                sim.add_port((.006, .009, .009), 'ez', impedance=50.,
                             waveform=GaussianPulse(f0=3e9, bandwidth=.8))
                sim.add_port((.006, .012, .009), 'ez', impedance=50., extent=.003, excite=False)
                sim.add_lumped_rlc((.018, .009, .009), 'ez', R=10., L=1e-9)
                sim.add_ntff_box((.003, .003, .003), (.021, .015, .015), freqs=freqs)
            if entry == 'forward_nonuniform':
                result = sim.forward(n_steps=8, port_s11_freqs=freqs, skip_preflight=True)
            elif entry == 'graded_decay':
                result = sim.run(until_decay=.01, decay_min_steps=8, decay_max_steps=16,
                                 decay_check_interval=4,
                                 s_param_freqs=freqs, skip_preflight=True)
            else:
                result = sim.run(n_steps=8, compute_s_params=entry != 'graded_tfsf',
                                 s_param_freqs=freqs if entry != 'graded_tfsf' else None,
                                 skip_preflight=True)
            jax.block_until_ready(result.time_series)
            assert result.time_series.dtype == expected
            required = {'initialized fields', 'fdtd', 'cpml'} | (
                {'tfsf'} if entry == 'graded_tfsf' else {'debye', 'lorentz', 'rlc_states', 'wire_sparams', 'ntff'})
            assert observed == required, observed
            print(entry, precision, 'incoming carry:', sorted(observed))
    finally:
        jax.clear_caches()
