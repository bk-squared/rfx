"""Publish port spectra by kind and arrange ring-down probe columns."""
import numpy as np


def pack_port_records(kinds, accumulators, metadata, *, graded=False, freqs=None):
    """Publish each kind in declaration order, without changing spectral bits.

    Uniform specs and accumulator tuples pass through unchanged. Graded wire
    metadata retain their static tuple schema; lumped records select slots
    0, 1 and 4 from the joint five-channel carry used internally for S.
    """
    from rfx._uniform_specs import LumpedPortSParamSpec

    wire, lumped = [], []
    for kind, accs, meta in zip(kinds, accumulators, metadata, strict=True):
        if kind == 'wire':
            wire.append((meta, accs))
        elif kind == 'lumped':
            if graded:
                meta = LumpedPortSParamSpec(*meta[:4], freqs, float(meta[4]), bool(meta[7]))
                accs = (accs[0], accs[1], accs[4])
            lumped.append((meta, accs))
        else:
            raise ValueError(f'Unknown port record kind: {kind}')
    return tuple(wire) or None, tuple(lumped) or None


def graded_record_pairs(result):
    """Restore the joint wire-then-lumped order for graded S diagnostics."""
    return ((getattr(result, 'wire_port_sparams', None) or ())
            + (getattr(result, 'lumped_port_sparams', None) or ()))


def recorded_port_vi(pm, e, hx, hy, hz):
    """Combine recorded samples in the shared table's component/term order.

    E is live-edge first; H arrays are the local 2x2x2 probe neighbourhood,
    with optional trailing time. _port_layout refuses a table node with no
    probe column; the 2x2x2 arrays are filled from the owner's H nodes.
    """
    from dataclasses import replace
    import jax.numpy as jnp
    from rfx.measurement.projector import sample_table, combine
    owner = replace(pm.owner, channels=tuple(c for c in pm.owner.channels
                                             if c.sample_stage == 'post-injection'))
    table = sample_table((owner,), dtype=e.dtype)
    h = dict(hx=hx, hy=hy, hz=hz)
    samples = []
    for terms in table.components:
        for node in zip(*terms.indices):
            if terms.component == pm.component:
                samples.append(e[pm.live_cells.index(tuple(node))])
            else:
                local = tuple(int(a-b+1) for a, b in zip(node, pm.mid))
                samples.append(h[terms.component][local])
    xp = np if isinstance(e, np.ndarray) else jnp
    values = combine(xp.stack(samples), table)
    return tuple(values[..., table.channel_names.index(name)] for name in ('V', 'V_port', 'I'))
