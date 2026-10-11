"""Assemble named port records at the result boundary; carries stay arrays."""
from .port_time import PortTimeRecord
from .projector import port_owners


def uniform_specs(ctx):
    return tuple((m.component, m.live_cells or ((m.mid_i, m.mid_j, m.mid_k),), 'wire',
                       (m.mid_i, m.mid_j, m.mid_k))
                       for m in ctx.wire_sparam_meta) + tuple(
                           (m.component, ((m.i, m.j, m.k),), 'lumped', (m.i, m.j, m.k)) for m in ctx.lumped_sparam_meta)


def from_owners(records, owners, n_steps=None):
    records, owners = tuple(records), tuple(owners)
    assert len(records) == len(owners), 'port record count differs from owner count'
    return tuple(PortTimeRecord(owner.id, owner.kind,
                               tuple(c.name for c in owner.channels), samples[:n_steps])
                 for owner, samples in zip(owners, records))


def uniform_records(carry, ctx, n_steps=None):
    from dataclasses import replace
    from .plan import Channel
    owners = list(port_owners(ctx.grid, uniform_specs(ctx), nu=False,
                              boundary=ctx.curl_boundary))
    plane_counts = {}
    for plane in ctx.wire_refplane_meta:
        port = owners[plane.port_index]
        p = plane_counts.get(port.id, 0)
        plane_counts[port.id] = p + 1
        owners.append(replace(port, id=f'{port.id}/plane{p}', channels=(
            Channel('V_plane', 'E', ()), Channel('I-_plane', 'H', ()),
            Channel('I+_plane', 'H', ()))))
    return from_owners(carry.get('sparam_time_records', ()), owners, n_steps)


def graded_records(setup, carry, n_steps=None):
    return from_owners(carry.get('sparam_time_records', ()),
                       setup.step_fn_inv.port_owners, n_steps)


def init_uniform_records(carry, n_steps, wires, lumped, planes, dtype):
    import jax.numpy as jnp
    if wires or lumped:
        carry['sparam_time_records'] = tuple(
            jnp.zeros((n_steps, count), dtype=dtype)
            for count in ([4] * len(wires) + [3] * len(lumped) + [3] * len(planes)))


def init_graded_records(carry, n_steps):
    import jax.numpy as jnp
    if 'wire_sparams' in carry:
        carry['sparam_time_records'] = tuple(
            jnp.zeros((n_steps, 3), dtype=carry['fdtd'].ex.dtype) for _ in carry['wire_sparams'])
