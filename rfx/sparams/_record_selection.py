"""Settling uses a port record only when the returned S reads that port."""


from dataclasses import dataclass

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SReadPorts:
    """Static ids of ports whose same-run accumulators formed returned S."""
    ids: tuple[str, ...] = ()

    def tree_flatten(self):
        return (), self.ids

    @classmethod
    def tree_unflatten(cls, ids, leaves):
        return cls(ids)


def read_ids(records, kind=None):
    return SReadPorts(tuple(r.id for r in records or ()
                           if kind is None or r.kind == kind))


def finish_forward(result):
    """Publish the ids selected by uniform forward's accumulator branch."""
    from rfx.measurement.port_s_contract import uniform_record_pairs
    pairs = uniform_record_pairs(result)
    kind = 'wire' if pairs is result.wire_port_sparams else 'lumped'
    selected = read_ids(result.sparam_time_records, kind) if result.s_params is not None else SReadPorts()
    return result._replace(sparam_read_ids=selected)


def s_read_records(result):
    """Read only explicitly identified, same-run S records, in record order."""
    ids = getattr(result, 'sparam_read_ids', None)
    if getattr(result, 's_params', None) is None or ids is None:
        return ()
    records = tuple(r for r in getattr(result, 'sparam_time_records', None) or () if r.id in ids.ids)
    if hasattr(getattr(result, 'grid', None), 'dx_arr'):
        # Graded S still reads only the physical channels. Its new reference
        # is retained for record parity, not as another settling-fit input.
        from dataclasses import replace
        records = tuple(replace(r, channels=r.channels[:-1], samples=r.samples[..., :-1])
                        if r.channels[-1:] == ('V_ref',) else r for r in records)
    return records


def record_selection_detail(result):
    records = tuple(getattr(result, 'sparam_time_records', None) or ())
    scored = [r.id for r in s_read_records(result)]
    return {'scored_port_record_ids': scored,
            'skipped_port_records': [{'id': r.id, 'reason': 'S not read'}
                                     for r in records if r.id not in scored]}


def forward_settling_witness(result):
    from rfx.sparams._tail_witness import result_read_bins

    bins = result_read_bins(result)
    freq_max = (result.settling_freq_max or getattr(result.grid, 'freq_max', None)
                or (max(bins) if len(bins) else None))
    return build_settling_witness(
        result, bins=bins, dt=result.dt, freq_max=freq_max,
        probe_info=result.settling_probe_info,
        # Transit is to the records judged: probes for the probe row, ports for the port row.
        source_end=(result.settling_probe_source_end_index
                    if result.settling_probe_source_end_index is not None
                    else result.settling_source_end_index),
        port_source_end=result.settling_source_end_index, warn=False)


def build_settling_witness(result, *, bins, dt, freq_max, probe_info,
                           source_end, port_source_end, prior_rows=(), warn=True):
    """Judge selected records once, retaining each contributing witness row.

    The caller supplies timing and any verdict from separately driven scans;
    record selection and aggregation are shared by both public entries.
    """
    import numpy as np
    from rfx.probes.settling import probe_record_settling_witness
    from rfx.sparams._tail_witness import combine_witness_details, port_record_witness
    from rfx.measurement.settling import settling_db_from_port_records

    rows = list(prior_rows)
    db = rows[0]['db'] if rows else None
    records = s_read_records(result)
    if records:
        db, detail = port_record_witness(
            records, dt, port_source_end, bins, freq_max=freq_max)
        rows.append(detail)
    if getattr(result, 'waveguide_ports', None):
        db, detail = settling_db_from_port_records(
            result.waveguide_ports.values(), freqs=bins,
            freq_max=freq_max, return_detail=True)
        rows.append(detail)
    probe_db, probe_detail = probe_record_settling_witness(
        getattr(result, 'time_series', None), probe_info, warn=warn,
        source_end_index=source_end, dt=dt, freqs=bins, freq_max=freq_max)
    if probe_info:
        db = probe_db
        rows.append(probe_detail)
    elif rows and _reads_unrecorded_fields(result):
        rows.append({**probe_detail, 'status': 'absent', 'db': None,
                     'worst_freq_hz': None,
                     'share_per_bin': np.zeros(bins.size),
                     'error_per_bin': np.zeros(bins.size),
                     'reason': 'far-field or DFT-plane frequencies are read but no '
                               'probe records the fields they come from'})
    if not rows:
        return probe_db, {**probe_detail, **record_selection_detail(result)}
    detail = combine_witness_details(*rows) if len(rows) > 1 else rows[0]
    if len(rows) > 1:
        db = detail['db']
        if bins.size:
            shares = np.maximum.reduce([row['share_per_bin'] for row in rows])
            errors = np.maximum.reduce([row['error_per_bin'] for row in rows])
            winning = np.argmax([row['share_per_bin'] for row in rows], axis=0)
            drive_shares = np.array([row['drive_share_per_bin']
                                     if len(row.get('drive_share_per_bin', ())) else np.zeros(bins.size)
                                     for row in rows])[winning, np.arange(bins.size)]
            detail = {**detail, 'share_per_bin': shares, 'error_per_bin': errors,
                      'drive_share_per_bin': (drive_shares if any(
                          len(row.get('drive_share_per_bin', ())) for row in rows) else ()),
                      'worst_freq_hz': (float(bins[np.argmax(shares)])
                                        if np.any(shares) or detail['status'] == 'pass'
                                        else float('nan'))}
    return db, {**detail, **record_selection_detail(result)}


def _reads_unrecorded_fields(result):
    """True when NTFF, DFT-plane, flux or current-moment outputs are read."""
    if getattr(result, 'ntff_box', None) is not None:
        return True
    if getattr(result, 'current_moment_monitor', None) is not None:
        return True
    return any(getattr(result, f, None) for f in ('dft_planes', 'flux_monitors'))
