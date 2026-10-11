"""Assemble named port records at the result boundary; carries stay arrays."""
from .port_time import PortTimeRecord
from .projector import port_owners


def uniform_specs(ctx):
    if ctx.port_record_specs is not None:
        return ctx.port_record_specs
    return tuple((m.component, m.live_cells or ((m.mid_i, m.mid_j, m.mid_k),), 'wire',
                       (m.mid_i, m.mid_j, m.mid_k), i)
                       for i, m in enumerate(ctx.wire_sparam_meta)) + tuple(
                           (m.component, ((m.i, m.j, m.k),), 'lumped', (m.i, m.j, m.k), i + len(ctx.wire_sparam_meta)) for i, m in enumerate(ctx.lumped_sparam_meta))


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


def init_uniform_records(carry, n_steps, wires, lumped, planes, dtype, specs=None):
    import jax.numpy as jnp
    counts = ([4 if p[2] == 'wire' else 3 for p in specs] if specs is not None else
              [4] * len(wires) + [3] * len(lumped))
    if counts:
        carry['sparam_time_records'] = tuple(
            jnp.zeros((n_steps, count), dtype=dtype) for count in counts + [3] * len(planes))


def init_graded_records(carry, n_steps, setup):
    import jax.numpy as jnp
    owners = setup.step_fn_inv.port_owners
    if owners:
        carry['sparam_time_records'] = tuple(
            jnp.zeros((n_steps, len(o.channels)), dtype=carry['fdtd'].ex.dtype) for o in owners)


def with_graded_records(result, setup, carry, n_steps=None):
    from rfx.sparams._record_selection import SReadPorts
    result['sparam_time_records'] = graded_records(setup, carry, n_steps)
    result['sparam_read_ids'] = SReadPorts(tuple(
        setup.step_fn_inv.port_owners[i].id for i in setup.step_fn_inv.port_acc_indices)
        if result.get('s_params') is not None else ())
    return result


def uniform_samples(values, specs, wire_refs, lumped_refs):
    import jax.numpy as jnp
    refs = iter((*wire_refs, *lumped_refs))
    samples, offset = [], 0
    for _, _, kind, _, _ in specs:
        width = 3 if kind == 'wire' else 2
        samples.append(jnp.stack((*values[offset:offset+width], next(refs))))
        offset += width
    return samples
