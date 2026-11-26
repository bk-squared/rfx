"""Per-read-bin pole-tail witness for recorded values.

The record-end rules live in ``rfx.measurement.record_end``; the private
names imported here stay importable from this module.
"""

import numpy as np
from collections.abc import Mapping
from functools import partial
from typing import NamedTuple

from rfx.ringdown import identify, judge_growing_poles, _static_pole, _fit_residues, RingdownPole
from rfx.measurement.record_end import (IdentificationOps, TailShareWitness,
                                         _pole_contributions as _record_pole_contributions,
                                         _rounding_only_records, merge_growing_poles, record_end_error)

_IDENTIFICATION = IdentificationOps(
    identify, judge_growing_poles, _static_pole, _fit_residues, RingdownPole)
_pole_contributions = partial(_record_pole_contributions, identification=_IDENTIFICATION)


def tail_share_witness(records, dt, source_end, freqs, *, freq_max, rounding_groups=None,
                       _record_results=None, known_drives=None, n_record=None,
                       scale_groups=None, _channel_labels=None, reader=None,
                       echo_bound: Mapping[str, float | str | None] | None = None):
    """Delegate the record-end rule, retaining the identity reader by default."""
    result = record_end_error(
        records, dt, source_end, freqs, identification=_IDENTIFICATION, freq_max=freq_max,
        rounding_groups=rounding_groups, _record_results=_record_results,
        known_drives=known_drives, n_record=n_record, scale_groups=scale_groups,
        _channel_labels=_channel_labels, reader=reader)
    from rfx.measurement.record_end import with_echo_bound
    _, detail = with_echo_bound(result.db, result._asdict(), echo_bound, dt)
    return result._replace(**{key: value for key, value in detail.items() if key in result._fields})


def result_read_bins(result):
    """Collect the frequency bins stored by a result's spectral channels."""
    from rfx.core.jax_utils import is_tracer

    values = []

    def add(value):
        if value is not None and not is_tracer(value):
            values.extend(np.asarray(value, dtype=float).reshape(-1))

    add(getattr(result, 'freqs', None))
    for field in ('ntff_box', 'current_moment_monitor'):
        add(getattr(getattr(result, field, None), 'freqs', None))
    for field in ('dft_planes', 'flux_monitors', 'waveguide_ports'):
        for obj in (getattr(result, field, None) or {}).values():
            add(getattr(obj, 'freqs', None))
    for field in ('wire_port_sparams', 'lumped_port_sparams'):
        for meta, _ in (getattr(result, field, None) or ()):
            add(getattr(meta, 'freqs', None))
    return np.unique(values)


class SettlingChannels(NamedTuple):
    """Named time channels and their spectral read coordinates."""

    records: tuple
    dt: float
    source_end: int | None
    freqs: object
    freq_max: float

    def witness(self, *, scale_groups=None, _channel_labels=None, reader=None):
        return tail_share_witness(self.records, self.dt, self.source_end,
                                  self.freqs, freq_max=self.freq_max,
                                  scale_groups=scale_groups, _channel_labels=_channel_labels, reader=reader)


def port_record_witness(records, dt, source_end, freqs, *, freq_max, reader=None,
                        echo_bound: Mapping[str, float | str | None] | None = None):
    """Judge recorded S-channel matrices, one matrix per port.

    Drive witnesses are host reports: their bound retains scalar values,
    including the exact float32 values returned by the realized reduction.
    """
    from rfx.sparams._settling_groups import port_column_groups

    records = tuple(records)
    groups, labels = port_column_groups(records)
    channels = SettlingChannels(
        tuple((record.id, record.samples) for record in records),
        dt, source_end, freqs, freq_max)
    result = channels.witness(scale_groups=groups, _channel_labels=labels, reader=reader)
    from rfx.measurement.record_end import with_echo_bound
    if echo_bound is not None:
        import jax
        if any(isinstance(value, jax.core.Tracer) for value in jax.tree.leaves(echo_bound)):
            echo_bound = dict(tau_max=None, D=None, n_max=None, g=None,
                              reason='cannot judge: traced echo delay bound')
        else:
            echo_bound = {key: value.item() if value is not None and hasattr(value, 'item') else value
                          for key, value in echo_bound.items()}
    db, detail = with_echo_bound(result.db, result._asdict(), echo_bound, dt)
    return db, {**detail, 'route': 's_channels',
                       'worst_record': None, 'per_record_db': {},
                       'skipped_records': list(result.skipped_records), 'source_dominated': False,
                       'reader_frame': None if reader is None else reader.get('frame'),
                       'scored_port_record_ids': [r.id for r in records],
                       'skipped_port_records': [],
                       'source_dominated_records': [], 'qualifier': ''}


def msl_time_channels(plane_records, voltage_names, current_names, stencils,
                      metas, trace_planes, dz, directions, *, regions=None):
    """Project recorded planes onto the MSL voltage ladders and loop currents."""
    from types import SimpleNamespace

    from rfx.sparams._common import _collocated_msl_h, msl_modal_voltage
    from rfx.sources.msl_port import msl_loop_current

    regions = regions or {}
    planes = {name: SimpleNamespace(accumulator=value) for name, value in plane_records.items()}
    records = []
    for p, meta in enumerate(metas):
        channels = []
        for name in voltage_names[p]:
            value = plane_records[name]
            region = regions.get(name)
            cropped = region is not None and value.shape[1:] == (region[1]-region[0], region[3]-region[2])
            channels.append(np.asarray(msl_modal_voltage(
                value, j_centre=0 if cropped else meta['j_centre'],
                k_lo=0 if cropped else meta['k_lo'],
                k_hi=region[3]-region[2] if cropped else trace_planes[p][0],
                dz_arr=dz[region[2]:region[3]] if cropped else dz, dtype=np.complex128)))
        ha, hb = _collocated_msl_h(planes, current_names[p], stencils[p]['weights'])
        ha, hb = np.asarray(ha), np.asarray(hb)
        region = regions.get(current_names[p][0][1])
        if region is not None and ha.shape[1:] == (region[1]-region[0], region[3]-region[2]):
            w0, w1, n0, n1 = region
        else:
            w0, w1, n0, n1 = 0, ha.shape[1], 0, ha.shape[2]
        j0, j1 = meta['j_lo']-w0, meta['j_hi']-w0
        k0, k1 = trace_planes[p][0]-n0, trace_planes[p][1]-n0
        if meta.get('a_is_width', True):
            da = meta.get('a_arr', meta.get('dy_arr'))
            db = meta.get('b_arr', dz)
            da, db = da[w0:w1], db[n0:n1]
        else:
            ha, hb = ha.transpose(0, 2, 1), hb.transpose(0, 2, 1)
            j0, j1, k0, k1 = k0, k1, j0, j1
            da, db = meta['a_arr'][n0:n1], meta['b_arr'][w0:w1]
        channels.append(np.asarray(msl_loop_current(
            ha, hb, j_lo=j0, j_hi=j1, k_trace_lo=k0, k_trace_hi=k1,
            dy_arr=da, dz_arr=db, direction=directions[p])))
        records.append((f'msl{p}/V_I', np.column_stack(channels)))
    return records


def combine_witness_details(*rows):
    """Combine witnesses for spectra assembled from multiple runs."""
    missing = [row for row in rows if row['status'] in {'absent', 'undetermined'}]
    worst = missing[0] if missing else max(rows, key=lambda row: row['db'])
    return {**worst, 'reason': '; '.join(row['reason'] for row in missing),
            'growing_poles': merge_growing_poles(*(row.get('growing_poles', ()) for row in rows)),
            'runs': rows}
