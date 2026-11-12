"""Linear maps from recorded channels to the quantities a port S reader uses.

This module supplies weights and sampling conventions, never verdicts.
"""
import numpy as np


def port_s_reader(records, metadata, *, drive_id=None, whole_port=None):
    """Build a column of S from metadata keyed by the record's port identity.

    Metadata values are (Z0, live cell count, excited). Graded production S
    uses whole-port V_port (wire) or V (lumped), even without V_ref.
    """
    records = tuple(records)
    sampling = {f'{r.id}[{c}]': ('H' if c == 'I' else 'E', 1.)
                for r in records for c in r.channels}
    if any(r.id not in metadata for r in records):
        return dict(entries={}, sampling=sampling, reason='realized port metadata unavailable')
    drives = [r for r in records if r.id == drive_id] if drive_id else [
        r for r in records if metadata[r.id][2]]
    if len(drives) != 1:
        return dict(entries={}, sampling=sampling, reason='S record does not have exactly one driven port')
    drive = drives[0]
    if whole_port is None:
        whole_port = ('V_port' in drive.channels if drive.kind == 'wire'
                      else 'V_ref' in drive.channels)
    def wave(record, voltage, sign, impedance):
        return {f'{record.id}[{voltage}]': sign, f'{record.id}[I]': impedance}
    def voltage(record):
        return 'V_port' if record.kind == 'wire' and 'V_port' in record.channels else 'V'
    entries = {}
    zd, nd, _ = metadata[drive.id]
    ref = 'V_ref' if 'V_ref' in drive.channels else 'V'
    for record in records:
        zr, nr, _ = metadata[record.id]
        if whole_port:
            b = wave(record, voltage(record), 1., -zr)
            a = wave(drive, voltage(drive), 1., zd)
        elif record.id == drive.id:
            b = wave(drive, ref, -1., -zd)
            a = wave(drive, ref, -1., zd)
        else:
            b = wave(record, 'V', 1., -zr / max(nr, 1))
            a = wave(drive, ref, -1., zd / max(nd, 1))
        entries[f'{record.id}/{drive.id}'] = (b, a)
    return dict(entries=entries, sampling=sampling,
                frame='whole-port' if whole_port else 'legacy per-cell')


def realized_port_metadata(records, pairs):
    """Join records to realized S specs by kind and midpoint identity, never order."""
    by_identity = {}
    for kind, meta in pairs:
        if meta is None:
            continue
        if hasattr(meta, 'impedance'):
            mid = tuple(getattr(meta, a, getattr(meta, 'mid_' + a, None)) for a in 'ijk')
            identity = (meta.component, *mid)
            value = (meta.impedance, len(getattr(meta, 'live_cells', ())) or 1, meta.excite)
        else:
            identity = (meta[3], *meta[:3])
            value = (meta[4], len(meta[13]), meta[7])
        key = (kind, identity)
        if key in by_identity:
            raise ValueError(f'ambiguous realized port identity: {key}')
        by_identity[key] = value
    return {r.id: by_identity[r.kind, r.identity] for r in records
            if (r.kind, r.identity) in by_identity}


def result_port_s_reader(result, records):
    pairs = [(kind, meta) for kind in ('wire', 'lumped')
             for meta, _ in getattr(result, f'{kind}_port_sparams', None) or ()]
    return port_s_reader(records, realized_port_metadata(records, pairs),
                         whole_port=True if hasattr(getattr(result, 'grid', None), 'dx_arr') else None)


def waveguide_s_reader(configs, drive_index):
    """Reference-plane modal ratio; probe channels have zero weight."""
    from rfx.measurement.waveguide_s import _compute_mode_impedance
    from rfx.core.jax_utils import is_tracer
    if any(is_tracer(cfg.freqs) or is_tracer(getattr(cfg, 'v_ref_t', None)) for cfg in configs):
        return dict(entries={}, sampling={}, reason='record is traced')
    sampling, incoming, outgoing = {}, [], []
    for k, cfg in enumerate(configs):
        z = np.asarray(_compute_mode_impedance(cfg.freqs, cfg.f_cutoff, cfg.mode_type,
                                               medium=cfg.medium, dt=cfg.dt, dx=cfg.dx))
        direction = 1 if cfg.direction.startswith('+') else -1
        v, i = f'port{k}/v_ref_t', f'port{k}/i_ref_t'
        incoming.append({v: .5, i: .5 * direction * z})
        outgoing.append({v: .5, i: -.5 * direction * z})
        for name in ('v_ref_t', 'i_ref_t', 'v_probe_t', 'i_probe_t'):
            sampling[f'port{k}/{name}'] = ('E' if name.startswith('v') else 'H', 2.)
    return dict(entries={f'S{k+1}{drive_index+1}': (b, incoming[drive_index])
                         for k, b in enumerate(outgoing)}, sampling=sampling,
                frame='reference-plane modal waves')
