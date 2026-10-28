"""Restored sampler defects scoped to uniform distributed port measurements."""
import jax
import jax.numpy as jnp


def mutate_port_sampler(monkeypatch, mutation):
    from rfx.stepping import uniform_slab_ports as ports

    if mutation == 'own_slab':
        cut = ports.cut_probe_table

        def wrong_owner(*args, **kwargs):
            # Keep every term's local index and weight, but give its payload
            # to another rank: that rank does not own the sampled global cell.
            return jax.tree.map(lambda a: jnp.roll(a, 1, axis=0), cut(*args, **kwargs))

        monkeypatch.setattr(ports, 'cut_probe_table', wrong_owner)
    elif mutation == 'half_step':
        references, update = ports.UniformSlabPorts.references, ports.UniformSlabPorts.update
        before = {}

        def capture(self, state, payload, *, ranks):
            before[self] = state
            return references(self, state, payload, ranks=ranks)

        def early(self, carry, state, refs, step, payload, *, ranks):
            # Physical completed-field channels now read the pre-injection
            # state. Reference sampling, DFT stamps and user probes stay intact.
            return update(self, carry, before[self], refs, step, payload, ranks=ranks)

        monkeypatch.setattr(ports.UniformSlabPorts, 'references', capture)
        monkeypatch.setattr(ports.UniformSlabPorts, 'update', early)
    else:
        raise ValueError(mutation)
