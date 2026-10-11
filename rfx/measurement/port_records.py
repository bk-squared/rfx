"""Arrange the ring-down probe columns for the shared point reduction."""
import numpy as np


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
