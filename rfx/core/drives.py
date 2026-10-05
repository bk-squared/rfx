"""Realized source tables and component-grouped soft-source injection."""
from typing import NamedTuple

import numpy as np


class Drives(NamedTuple):
    """Nodes and coefficients keyed by Yee component; waves are (wave, time).

    ``owner`` identifies the drive, not a device. ``source_end_step`` reserves
    per-waveform end metadata for the drives/measurement assemblers. The
    builders use NumPy index/coefficient arrays; step layouts omit ``waves``
    because samples enter the step through scan arguments.
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


def drive_layout(meta, dtype, *, electric_only_path=None):
    """Static, host-side injection layout; never retains a waveform table.

    Scan bodies close over this layout and receive samples as scan arguments.
    NumPy indices/coefficients avoid capturing arrays pinned to one device in
    multi-process runners. ``Drives.waves`` is only populated by the assembler.
    """
    nodes, coef, wave_id, owner = {}, {}, {}, {}
    for component in dict.fromkeys(s[3] for s in meta):
        if electric_only_path and component not in ("ex", "ey", "ez"):
            raise ValueError(f"{electric_only_path}: unsupported source component {component}")
        ids = [i for i, s in enumerate(meta) if s[3] == component]
        nodes[component] = tuple(
            np.asarray([meta[i][axis] for i in ids], dtype=np.int32)
            for axis in range(3))
        coef[component] = np.ones(len(ids), dtype=dtype)
        wave_id[component] = np.asarray(ids, dtype=np.int32)
        owner[component] = np.asarray(ids, dtype=np.int32)
    return Drives(nodes, coef, wave_id, None, owner)


def drives_from_sources(meta, waves):
    """Assemble sources without renormalizing; step builders use drive_layout."""
    return drive_layout(meta, waves.dtype)._replace(waves=waves)


def inject_drives(state, drives, wave_t):
    """Add one scatter per populated component, retaining duplicate-node adds.

    ``wave_t`` is a column of the assembled waveform table, passed separately
    so step layouts need not retain that table. Cast updates to the destination dtype, as required by mixed-precision runs.
    No uniqueness promise is made to XLA: overlapping drives must accumulate.
    """
    updates = {}
    for component, nodes in drives.nodes.items():
        field = getattr(state, component)
        values = drives.coef[component] * wave_t[drives.wave_id[component]]
        updates[component] = field.at[nodes].add(values.astype(field.dtype))
    return state._replace(**updates)
