"""Restored sampler defects scoped to uniform distributed port measurements."""
import jax
import jax.numpy as jnp
from unittest.mock import patch


def mutate_port_sampler(monkeypatch, mutation):
    from rfx.stepping import slab_wrap_uniform as wrapped

    if mutation == 'own_slab':
        prepare = wrapped.prepare_uniform

        def wrong_owner(*args, **kwargs):
            # Keep every term's local index and weight, but give its payload
            # to another rank: that rank does not own the sampled global cell.
            carry, inputs, waves, advance = prepare(*args, **kwargs)
            if 'port_table' in inputs:
                inputs = dict(inputs, port_table=jax.tree.map(
                    lambda a: jnp.roll(a, 1, axis=0), inputs['port_table']))
            return carry, inputs, waves, advance

        monkeypatch.setattr(wrapped, 'prepare_uniform', wrong_owner)
    elif mutation == 'half_step':
        from rfx.measurement import projector
        from rfx.stepping.sequence import HookPoint
        make_step, project = wrapped.make_uniform_step, projector.project

        def early_step(context, invariants, *, hooks):
            before = {}

            def capture(frame):
                before['state'] = frame.st

            def early(state, table):
                # Only physical port channels read the pre-injection state.
                # Reference sampling, DFT stamps and user probes stay intact.
                return project(before['state'] if table is invariants.get('port_table')
                               else state, table)

            hooks = dict(hooks)
            hooks[HookPoint.BEFORE_SOURCES] = (*hooks.get(HookPoint.BEFORE_SOURCES, ()), capture)
            with patch.object(projector, 'project', early):
                return make_step(context, invariants, hooks=hooks)

        monkeypatch.setattr(wrapped, 'make_uniform_step', early_step)
    else:
        raise ValueError(mutation)
