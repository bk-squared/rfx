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
    from rfx.model.realization_transport import forward_arrays_match
    if not forward_arrays_match(result):
        raise ValueError(
            "ForwardResult settling reads one primal result; its arrays are not the arrays "
            "it was built with (a vmap output, a Jacobian, or a sliced or mapped result). "
            "For a batch, take one element first: "
            "jax.tree_util.tree_map(lambda a: a[i], result).")
    from rfx.sparams._tail_witness import result_read_bins

    bins = result_read_bins(result)
    freq_max = (result.settling_freq_max or getattr(result.grid, 'freq_max', None)
                or (max(bins) if len(bins) else None))
    db, witness = build_settling_witness(
        result, bins=bins, dt=result.dt, freq_max=freq_max,
        probe_info=result.settling_probe_info,
        # Transit is to the records judged: probes for the probe row, ports for the port row.
        source_end=(result.settling_probe_source_end_index
                    if result.settling_probe_source_end_index is not None
                    else result.settling_source_end_index),
        port_source_end=result.settling_source_end_index, warn=False)
    from rfx.measurement.coverage import AccumulatorCoverage, with_coverage
    from rfx.measurement.record_end import with_echo_bound
    db, witness = with_coverage(db, witness, result.settling_coverage or AccumulatorCoverage())
    db, witness = with_echo_bound(db, witness, result.settling_echo_bound, result.dt)
    return db, {key: value for key, value in witness.items() if key != 'groups'}


def build_settling_witness(result, *, bins, dt, freq_max, probe_info,
                           source_end, port_source_end, prior_rows=(), warn=True):
    """Judge selected records once, retaining each contributing witness row.

    The caller supplies timing and any verdict from separately driven scans;
    record selection and aggregation are shared by both public entries.
    """
    import numpy as np
    from rfx.probes.settling import probe_record_settling_witness
    from rfx.sparams._tail_witness import port_record_witness
    from rfx.measurement.settling import settling_db_from_port_records

    rows = [{**row, 'reads': ('S-parameters',)} for row in prior_rows]
    db = rows[0]['db'] if rows else None
    records = s_read_records(result)
    if records:
        from rfx.measurement.record_readers import result_port_s_reader
        db, detail = port_record_witness(
            records, dt, port_source_end, bins, freq_max=freq_max,
            reader=result_port_s_reader(result, records))
        rows.append({**detail, 'reads': ('S-parameters',)})
    if getattr(result, 'waveguide_ports', None):
        db, detail = settling_db_from_port_records(
            result.waveguide_ports.values(), freqs=bins,
            freq_max=freq_max, return_detail=True)
        rows.append({**detail, 'reads': ('S-parameters',)})
    probe_db, probe_detail = probe_record_settling_witness(
        getattr(result, 'time_series', None), probe_info, warn=warn,
        source_end_index=source_end, dt=dt, freqs=bins, freq_max=freq_max)
    field_reads = field_outputs_read(result)
    probe_detail = {**probe_detail, 'reads': field_reads}
    if probe_info:
        db = probe_db
        rows.append(probe_detail)
    elif field_reads:
        rows.append({**probe_detail, 'status': 'absent', 'db': None,
                     'worst_freq_hz': None,
                     'share_per_bin': np.zeros(bins.size),
                     'error_per_bin': np.zeros(bins.size),
                     'reason': 'far-field or DFT-plane frequencies are read but no '
                               'probe records the fields they come from'})
    if not rows:
        return probe_db, {**probe_detail, 'groups': [probe_detail],
                          **record_selection_detail(result)}
    detail = _combine_rows(rows, bins)
    if len(rows) > 1:
        db = detail['db']
    by_reads: dict[tuple, list] = {}
    for row in rows:
        by_reads.setdefault(tuple(row.get('reads', ())), []).append(row)
    groups = [{**{k: v for k, v in _combine_rows(members, bins).items() if k != 'groups'},
               'reads': reads}
              for reads, members in by_reads.items()]
    return db, {**detail, 'groups': groups, **record_selection_detail(result)}


def _combine_rows(rows, bins):
    """One witness from rows: status by combine_witness_details, per-bin
    shares and errors by maximum, worst bin by the largest share."""
    import numpy as np
    from rfx.sparams._tail_witness import combine_witness_details

    detail = combine_witness_details(*rows) if len(rows) > 1 else rows[0]
    if len(rows) > 1 and bins.size:
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
    quantity_rows = [row for row in rows if len(row.get('record_end_error_per_bin', ()))]
    if quantity_rows:
        quantity = np.maximum.reduce([row['record_end_error_per_bin'] for row in quantity_rows])
        detail = {**detail, 'record_end_error_per_bin': quantity,
                  'kappa_per_bin': np.maximum.reduce([row['kappa_per_bin'] for row in quantity_rows]),
                  'record_end_floor_per_bin': np.maximum.reduce([
                      row['record_end_floor_per_bin'] for row in quantity_rows]),
                  'status_per_entry': {key: value for row in quantity_rows
                                       for key, value in row['status_per_entry'].items()},
                  'record_end_error_per_entry': {key: value for row in quantity_rows
                                                for key, value in row['record_end_error_per_entry'].items()},
                  'record_end_error_second_per_entry': {key: value for row in quantity_rows
                                                for key, value in (row.get('record_end_error_second_per_entry') or {}).items()}}
        if bins.size and detail['status'] in {'pass', 'fail'}:
            detail['worst_freq_hz'] = float(bins[np.argmax(quantity)])
    return detail


def field_outputs_read(result):
    """Field outputs judged by the probe record, in reporting order."""
    return tuple(name for name, present in (
        ("NTFF far fields", getattr(result, 'ntff_box', None) is not None),
        ("field-DFT planes", bool(getattr(result, 'dft_planes', None))),
        ("flux monitors", bool(getattr(result, 'flux_monitors', None))),
        ("current-moment far fields", getattr(result, 'current_moment_monitor', None) is not None),
    ) if present)


def _reads_unrecorded_fields(result):
    """True when NTFF, DFT-plane, flux or current-moment outputs are read."""
    return bool(field_outputs_read(result))
