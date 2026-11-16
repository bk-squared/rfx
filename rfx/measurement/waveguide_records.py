"""Compile completed-field waveguide observations once for either solver."""
from dataclasses import fields

import numpy as np

from .ports import waveguide_owner
from .projector import sample_table


def check_waveguide_owner(owner, expected):
    """Require the attached channel map to equal the runner's map in bytes."""
    def signature(channel):
        metadata = tuple(getattr(channel, f.name) for f in fields(channel) if f.name != 'nodes')
        nodes = tuple((n.component, n.i, n.j, n.k,
                       np.asarray(n.weight).tobytes(),
                       None if n.area is None else np.asarray(n.area).tobytes()) for n in channel.nodes)
        return metadata, nodes
    if tuple(map(signature, owner.channels)) != tuple(map(signature, expected.channels)):
        raise ValueError(f'{owner.id}: waveguide channel map differs from runner config')


def compile_waveguide_table(grid, cfgs, *, measurement=None, dtype):
    """Keep plan order and validate every runner config, including passive modes."""
    owners = tuple(waveguide_owner(grid, cfg, port_id=i) for i, cfg in enumerate(cfgs))
    if measurement is not None:
        selected = tuple(o for o in measurement.owners if o.kind == 'waveguide')
        if len(selected) != len(cfgs):
            raise ValueError(f'waveguide owners: plan count {len(selected)} != runner count {len(cfgs)}')
        for owner, expected in zip(selected, owners):
            check_waveguide_owner(owner, expected)
        owners = selected
    return sample_table(owners, dtype=dtype)
