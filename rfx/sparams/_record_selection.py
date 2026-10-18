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
    kind = 'lumped' if result.lumped_port_sparams else 'wire'
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
