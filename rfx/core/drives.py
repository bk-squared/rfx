"""Realized source tables and component-grouped soft-source injection."""
from typing import NamedTuple

import jax.numpy as jnp


class Drives(NamedTuple):
    """Nodes and coefficients keyed by Yee component; waves are (wave, time).

    ``owner`` identifies the drive, not a device. ``source_end_step`` reserves
    per-waveform end metadata for the drives/measurement assemblers.
    """

    nodes: dict
    coef: dict
    wave_id: dict
    waves: object
    owner: dict
    source_end_step: object = None


class StepDrives(NamedTuple):
    """The two Yee injection stages, owned by one step-context field.

    Tables stay separate to retain their original sample dtypes and timing:
    magnetic drives after H, electric drives after E. Clearing the context's
    ``drives`` field suppresses both stages without changing the scan inputs.
    """

    electric: Drives
    magnetic: Drives


def drives_from_sources(meta, waves):
    """Adapt existing (i, j, k, component) sources without renormalizing them."""
    nodes, coef, wave_id, owner = {}, {}, {}, {}
    for component in dict.fromkeys(s[3] for s in meta):
        ids = [i for i, s in enumerate(meta) if s[3] == component]
        nodes[component] = tuple(
            jnp.asarray([meta[i][axis] for i in ids], dtype=jnp.int32)
            for axis in range(3))
        coef[component] = jnp.ones(len(ids), dtype=waves.dtype)
        wave_id[component] = jnp.asarray(ids, dtype=jnp.int32)
        owner[component] = wave_id[component]
    return Drives(nodes, coef, wave_id, waves, owner)


def inject_drives(state, drives, wave_t):
    """Add one scatter per populated component, retaining duplicate-node adds.

    ``wave_t`` is ``drives.waves[:, step]`` (or the equivalent scan input).
    Cast updates to the destination dtype, as required by mixed-precision runs.
    No uniqueness promise is made to XLA: overlapping drives must accumulate.
    """
    updates = {}
    for component, nodes in drives.nodes.items():
        field = getattr(state, component)
        values = drives.coef[component] * wave_t[drives.wave_id[component]]
        updates[component] = field.at[nodes].add(values.astype(field.dtype))
    return state._replace(**updates)
