"""Execute production sampling/phase statements without a field solve.

Flux sampling is currently inline in two large step bodies. AST extraction
executes those exact statements, including their branches; it does not restate
the stencil in a test helper. Missing/ambiguous statements fail closed.
"""
import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')


@lru_cache(None)
def tree(nu):
    path = nu if isinstance(nu, str) else ('rfx/nonuniform.py' if nu else 'rfx/simulation.py')
    result = ast.parse((ROOT / path).read_text())
    if nu is False:
        # Keep the original runner in scope and recognize its extracted phases.
        for step in sorted((ROOT / 'rfx/stepping').glob('*.py')):
            result.body.extend(ast.parse(step.read_text()).body)
    return result


def assignment(nu, name, dependency=None):
    candidates = [n for n in ast.walk(tree(nu)) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
                  and (dependency is None or dependency in {v.id for v in ast.walk(n.value) if isinstance(v, ast.Name)})]
    # Identical assignments can occur in multiple runner entry points.
    expressions = {ast.dump(n.value): n.value for n in candidates}
    assert len(expressions) == 1, (name, expressions)
    return compile(ast.Expression(next(iter(expressions.values()))), '<production phase>', 'eval')


def stamp(nu, family, kind):
    """Execute the actual scan's shared-accumulator call with a unit field."""
    from rfx.measurement.accumulators import planes, flux
    from rfx.measurement.plan import field_channel
    shape = (3, 3, 3)
    state = SimpleNamespace(step=jnp.int32(4), **{c: jnp.ones(shape) for c in COMPONENTS})
    bins = np.array([.01])
    acc = jnp.zeros((1, 3, 3), dtype=jnp.complex64)
    name = 'planes' if family == 'dft_plane' else 'flux'
    calls = [n for n in ast.walk(tree(nu)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == name]
    assert len(calls) == 1
    dft_meta = ((field_channel(kind.lower()+'z'), 0, 1, bins, None),)
    flux_meta = ((0, 1, bins, tuple(field_channel(c) for c in ('ey', 'ez', 'hy', 'hz')), 0, 3, 0, 3),)
    env = dict(planes=planes, flux=flux, st=state, dt=1., step_idx=jnp.int32(3),
               carry={'dft_planes': (acc,), 'flux_monitors': ((acc, acc, acc, acc),)},
               dft_meta=dft_meta, flux_meta=flux_meta,
               ctx=SimpleNamespace(dft_meta=dft_meta, flux_meta=flux_meta))
    env['frame'] = SimpleNamespace(st=state, carry=env['carry'], step_idx=env['step_idx'])
    value = eval(compile(ast.Expression(calls[0]), '<production accumulator>', 'eval'), env)
    spectrum = value[0][0] if name == 'planes' else value[0][0 if kind == 'E' else 2]
    return float(-jnp.angle(spectrum[0, 0, 0])/(2*jnp.pi*.01))


def flux_samples(nu, cfg, fields):
    from rfx.measurement.accumulators import flux_samples as sample
    from rfx.probes.probes import _FLUX_COMPONENTS
    return sample(fields, cfg.axis, cfg.index, _FLUX_COMPONENTS[cfg.axis],
                  (cfg.lo1, cfg.hi1, cfg.lo2, cfg.hi2))


def jacobian_nodes(function, shape):
    arrays = tuple(jnp.zeros(shape) for _ in COMPONENTS)
    def scalar(*values):
        return function(SimpleNamespace(**dict(zip(COMPONENTS, values))))
    gradients = jax.grad(scalar, argnums=tuple(range(6)))(*arrays)
    return {(component, *index): float(np.asarray(g)[index])
            for component, g in zip(COMPONENTS, gradients)
            for index in zip(*np.nonzero(np.asarray(g)))}


def judge_metadata(plan, actual):
    """Compare every channel's pair of offsets and record slot to production."""
    from rfx.core.dft_utils import port_dft_phase
    from rfx.sources.waveguide_port import (
        update_waveguide_port_probe, _rect_dft,
    )
    nu = 'nonuniform' in plan.path
    for owner in plan.owners:
        for channel in owner.channels:
            slot = 0
            if owner.kind in ('dft_plane', 'wire', 'lumped'):
                indices = [n.slice for n in ast.walk(tree(nu)) if isinstance(n, ast.Subscript)
                           and isinstance(n.value, ast.Attribute) and n.value.attr == 'at'
                           and isinstance(n.value.value, ast.Name) and n.value.value.id == 'record']
                assert indices
                slots = {int(eval(compile(ast.Expression(index), '<production record slot>', 'eval'),
                                  dict(step_idx=jnp.int32(3),
                                       frame=SimpleNamespace(step_idx=jnp.int32(3)))))-3 for index in indices}
                assert len(slots) == 1
                slot = slots.pop()
            if owner.kind in ('dft_plane', 'flux', 'msl', 'coax'):
                family = 'flux' if owner.kind == 'flux' else 'dft_plane'
                e, h = stamp(nu, family, 'E'), stamp(nu, family, 'H')
            elif owner.kind in ('wire', 'lumped'):
                bins = jnp.array([.01])
                env = dict(jnp=jnp, step_idx=jnp.int32(3), dt=1., sp_freqs=bins,
                           wp_meta=SimpleNamespace(freqs=bins), lp_meta=SimpleNamespace(freqs=bins),
                           port_dft_phase=port_dft_phase,
                           **{name: SimpleNamespace(dtype=jnp.complex64) for name in (
                               'v_dft', 'i_dft', 'v_ref_dft', 'v_ref_dft_l', 'i_dft_l')})
                env['frame'] = SimpleNamespace(step_idx=env['step_idx'])
                lumped_uniform = owner.kind == 'lumped' and not nu
                phase = eval(assignment(nu, 'phase_l' if lumped_uniform else 'phase',
                                        'lp_meta' if lumped_uniform else 'sp_freqs' if nu else 'wp_meta'), env)
                env['phase_l' if lumped_uniform else 'phase'] = phase
                current_phase = eval(assignment(nu, 'i_phase_l' if lumped_uniform else 'i_phase'), env)
                e = float(-jnp.angle(phase[0])/(2*jnp.pi*.01))
                h = float(-jnp.angle(current_phase[0])/(2*jnp.pi*.01))
            elif owner.kind == 'waveguide':
                cfg = actual['waveguide_ports'][int(owner.id.rsplit(':', 1)[1])]
                cfg = cfg._replace(freqs=jnp.array([.01]))
                state = SimpleNamespace(step=jnp.int32(4))
                # Distinct nonzero values reveal the actual write index, independent of the plan.
                from unittest.mock import patch
                with patch('rfx.sources.waveguide_port.modal_voltage', return_value=jnp.float32(2)), \
                     patch('rfx.sources.waveguide_port.modal_current', return_value=jnp.float32(3)):
                    recorded = update_waveguide_port_probe(cfg, state, 1., 1.)
                indices = np.flatnonzero(recorded.v_probe_t)
                assert len(indices) == 1
                slot = int(indices[0])-3
                phase = _rect_dft(recorded.v_probe_t, recorded.freqs, 1., recorded.n_steps_recorded)
                e = float(-jnp.angle(phase[0])/(2*jnp.pi*.01))
                h_phase = _rect_dft(recorded.v_probe_t, recorded.freqs, 1., recorded.n_steps_recorded, 'H')
                h = float(-jnp.angle(h_phase[0])/(2*jnp.pi*.01))
            elif owner.kind == 'current_moment':
                # The actual monitor supplies the accumulator's half-step stamp.
                e, h = 4., 3+float(actual['current_moments'].half_step)
                assert channel.kind == 'H'  # Contributions form the current at the H time.
            else:
                # Raw probes have no Fourier stamp. They return post-update state
                # in scan order, so their clock is the physical sample time.
                e, h = 4., 3.5
            np.testing.assert_allclose(channel.e_time_offset, e-4, atol=2e-6, rtol=0)
            np.testing.assert_allclose(channel.h_time_offset, h-3.5, atol=2e-6, rtol=0)
            assert channel.slot_offset == slot
            if (channel.e_time_offset if channel.kind == 'E' else channel.h_time_offset) != 0:
                assert any(flag.startswith(channel.name+': '+channel.kind+' stamp offset')
                           for flag in owner.known_differences)
        if owner.kind in ('wire', 'lumped'):
            expected = 'sampled'
            if owner.kind == 'lumped' and plan.path.startswith('run'):
                expected = ('V/I unavailable in run without wire extent' if nu else
                            'V/I unavailable in main scan; sampled in second scan when compute_s_params is on')
            assert owner.availability == expected
            assert ('NU missing pre-injection V_ref' in owner.known_differences) == nu
        elif owner.kind == 'flux':
            assert owner.availability == ('result missing' if plan.path.startswith('fwd') else 'sampled')
        elif owner.kind in ('msl', 'coax') and not owner.channels:
            assert owner.availability == 'calculator setup required; run() has no V/I projector'
        else:
            assert owner.availability == 'sampled'
