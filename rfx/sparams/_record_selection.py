"""Settling uses a port record only when the returned S reads that port."""


def s_read_records(result):
    """Follow the result's S extraction, independently of record presence."""
    records = tuple(getattr(result, 'sparam_time_records', None) or ())
    if getattr(result, 's_params', None) is None:
        return ()
    # Uniform forward chooses lumped S before wire S. Graded forward uses
    # its single wire accumulator collection for both kinds.
    lumped = getattr(result, 'lumped_port_sparams', None)
    if lumped:
        wires = getattr(result, 'wire_port_sparams', None) or ()
        return records[len(wires):len(wires) + len(lumped)]
    return records


def record_selection_detail(result):
    """Add record identities without changing the numeric verdict fields."""
    records = tuple(getattr(result, 'sparam_time_records', None) or ())
    selected = s_read_records(result)
    ids = [getattr(record, 'id', f'port:{i}:mode:0')
           for i, record in enumerate(records)]
    scored = [name for name, record in zip(ids, records)
              if any(record is chosen for chosen in selected)]
    return {'scored_port_record_ids': scored,
            'skipped_port_records': [{'id': name, 'reason': 'S not read'}
                                     for name in ids if name not in scored]}


def forward_settling_witness(result):
    from rfx.probes.settling import probe_record_settling_witness
    from rfx.sparams._tail_witness import port_record_witness, result_read_bins

    bins = result_read_bins(result)
    freq_max = (result.settling_freq_max or getattr(result.grid, 'freq_max', None)
                or (max(bins) if len(bins) else None))
    records = s_read_records(result)
    if records:
        db, detail = port_record_witness(
            records, result.dt, result.settling_source_end_index,
            bins, freq_max=freq_max)
    else:
        db, detail = probe_record_settling_witness(
            result.time_series, result.settling_probe_info, warn=False,
            source_end_index=result.settling_source_end_index, dt=result.dt,
            freqs=bins, freq_max=freq_max)
    return db, {**detail, **record_selection_detail(result)}
