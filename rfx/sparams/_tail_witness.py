"""Per-read-bin pole-tail witness for recorded values."""

from typing import NamedTuple

import numpy as np


_POLE_RULE = (
    "For each discarded growing, kept non-static growing, or non-decaying "
    "sinusoidal pole: abs(DFT of its fitted term inside the identification "
    "window) / abs(whole-record plain DFT), per read bin and channel; "
    "> 1e-2 is undetermined. Static poles are exempt from these checks."
)


def _pole_contributions(model, series, bins, denominator, record_s):
    """Return per-bin maxima over fitted pole terms and channels by rule."""
    from rfx.ringdown import RingdownPole, _fit_residues, _static_pole, plain_dft

    window = series[model.n_ref:model.n_ref + model.n_window]
    indices = {'discarded growing': [], 'kept non-static growing': [],
               'non-decaying sinusoidal': []}
    for k in range(len(model.s)):
        pole = model.pole(k)
        if _static_pole(pole, record_s, 1e-6):
            continue
        if model.s[k].real > 0:
            indices['kept non-static growing'].append(k)
        if abs(model.s[k].real * model.dt) < 1e-12:
            indices['non-decaying sinusoidal'].append(k)
    for k, pole_s in enumerate(model.s_growing):
        pole = RingdownPole(float(pole_s.imag / (2*np.pi)), 0.,
                            float(-pole_s.real), float(abs(np.exp(pole_s*model.dt))), 0.)
        if not _static_pole(pole, record_s, 1e-6):
            indices['discarded growing'].append(k)
    joint = None
    if indices['discarded growing']:
        joint = _fit_residues(window, np.concatenate([model.s, model.s_growing]),
                              model.dt, column_reference=True)[len(model.s):]
    time = np.arange(len(window)) * model.dt
    values = {}
    for rule, selected in indices.items():
        value = np.zeros(len(bins))
        for k in selected:
            if rule == 'discarded growing':
                samples = np.exp((time - time[-1]) * model.s_growing[k])[:, None] * joint[k]
            else:
                samples = np.exp(time * model.s[k])[:, None] * model.c[k]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.abs(plain_dft(samples, model.dt, bins)) / denominator
            value = np.maximum(value, np.max(ratio, axis=1))
        values[rule] = value
    return values


class TailShareWitness(NamedTuple):
    db: float
    share_per_bin: np.ndarray
    worst_freq_hz: float
    status: str
    reason: str
    error_per_bin: np.ndarray
    pole_contributions: dict | None = None
    pole_rule: str = _POLE_RULE


def tail_share_witness(records, dt, source_end, freqs, *, freq_max, _record_results=None):
    """Judge per-bin pole tails by agreement between two source-free windows.

    sensitivities are judged by a separate witness
    """
    from dataclasses import replace

    from rfx.core.jax_utils import is_tracer
    from rfx.ringdown import identify, plain_dft, tail_dft, _static_pole

    records = tuple(records)
    if is_tracer(freqs):
        return TailShareWitness(float('nan'), np.array([]), float('nan'),
                                'undetermined', 'read bins are traced', np.array([]))
    bins = np.asarray([] if freqs is None else freqs, dtype=float).reshape(-1)
    share = np.zeros(bins.size)
    error = np.zeros(bins.size)
    contributions = {}

    def unavailable(status, reason):
        return TailShareWitness(float('nan'), share, float(bins[np.argmax(share)]) if bins.size and np.any(share) else float('nan'), status, reason, error, contributions)

    if not records or not bins.size:
        return unavailable('absent', 'no records or no read bins')
    if source_end is None or is_tracer(source_end):
        return unavailable('undetermined', 'source end is unavailable')
    if dt is None or freq_max is None or is_tracer(dt) or is_tracer(freq_max):
        return unavailable('undetermined', 'dt or identification freq_max is unavailable')
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(freq_max) or freq_max <= 0 or not np.isfinite(bins).all():
        return unavailable('undetermined', 'invalid time step, read bins, or identification freq_max')
    reasons = []
    zero_reasons = []
    groups = {}
    for name, raw in sorted(records, key=lambda item: item[0]):
        if _record_results is not None:
            _record_results[name] = float("nan")
        if is_tracer(raw):
            reasons.append(f"{name}: record is traced")
            continue
        try:
            y = np.asarray(raw)
            if y.ndim == 0 or not y.size or not np.isfinite(y).all():
                raise ValueError("empty or non-finite record")
            y = y.reshape(y.shape[0], -1).astype(np.complex128)
            groups.setdefault(len(y), []).append((name, y))
        except (ValueError, FloatingPointError) as exc:
            reasons.append(f"{name}: {exc}")

    for n, group in sorted(groups.items()):
        label = ", ".join(name for name, _ in group)
        n_reasons = len(reasons)
        try:
            if int(source_end) != source_end or not 0 <= source_end < n:
                raise ValueError("source end outside record")
            start = max(int(source_end), round(.5 * n))
            use_earlier = start // 2 >= source_end
            check_start = start // 2 if use_earlier else start
            check_stop = n if use_earlier else start + round(.9 * (n - start))
            if min(n - start, check_stop - check_start) < 10:
                raise ValueError("post-source window too short for the identification check")
            group = [(name, array[:, np.any(array[int(source_end):] != array[int(source_end)], axis=0)])
                     for name, array in group]
            y = np.concatenate([array for _, array in group], axis=1)
            if y.shape[1] == 0:
                zero_reasons.append(f"{label}: no post-source variation")
                if _record_results is not None:
                    for name, _ in group:
                        _record_results[name] = float("-inf")
                continue
            scale = np.max(np.maximum(np.abs(y.real), np.abs(y.imag)), axis=0)
            y = y / scale
            denom = np.abs(plain_dft(y, dt, bins))

            def identify_window(lo, hi):
                centered = y - np.mean(y[lo:hi], axis=0)
                active = np.any(centered[lo:hi] != 0, axis=0)
                if not np.any(active):
                    raise ValueError("no post-source variation in identification window only")
                model = identify(centered[:, active], dt, lo, hi,
                                 freq_max=max(freq_max, float(np.max(bins))))
                residues = np.zeros((len(model.s), y.shape[1]), dtype=complex)
                residues[:, active] = model.c
                model = replace(model, c=residues)
                return model, centered, np.where(active, denom, np.inf)

            main, main_series, main_denom = identify_window(start, n)
            other, other_series, other_denom = identify_window(check_start, check_stop)
            for model, series, window_denom in ((main, main_series, main_denom),
                                                 (other, other_series, other_denom)):
                if not len(model.s):
                    reasons.append(f"{label}: no kept poles identified")
                for rule, value in _pole_contributions(model, series, bins, window_denom, n*dt).items():
                    contributions[rule] = np.maximum(contributions.get(rule, np.zeros(len(bins))), value)
                    for k in np.flatnonzero((value > 1e-2) | ~np.isfinite(value)):
                        comparison = '> 1e-2' if np.isfinite(value[k]) else 'is non-finite'
                        reasons.append(
                            f"{label}: {rule} pole at read bin {bins[k]:.12g} Hz; "
                            f"identification-window term DFT / whole-record DFT = {value[k]:.12g} {comparison}")
            def without_static(model):
                keep = np.array([not _static_pole(model.pole(k), n*dt, 1e-6)
                                 for k in range(len(model.s))], dtype=bool)
                return replace(model, s=model.s[keep], c=model.c[keep])

            main, other = without_static(main), without_static(other)
            tail = tail_dft(main, n - 1, bins)
            alt = tail_dft(other, n - 1, bins)
            with np.errstate(divide="ignore", invalid="ignore"):
                per_channel = np.abs(tail) / main_denom
                other_channel = np.abs(alt) / other_denom
                err_channel = np.abs(tail - alt) / np.minimum(main_denom, other_denom)
            if not np.isfinite(per_channel).all() or not np.isfinite(err_channel).all() or not np.isfinite(other_channel).all():
                reasons.append(f"{label}: non-finite tail or zero in-record DFT")
                continue
            share = np.maximum(share, per_channel.max(axis=1))
            error = np.maximum(error, err_channel.max(axis=1))
            uncertain = (per_channel > 1e-2) != (other_channel > 1e-2)
            if _record_results is not None and len(reasons) == n_reasons:
                offset = 0
                for name, array in group:
                    if array.shape[1] == 0:
                        _record_results[name] = float('-inf')
                        continue
                    columns = slice(offset, offset + array.shape[1])
                    peak = np.max(per_channel[:, columns])
                    if not np.any(uncertain[:, columns]):
                        _record_results[name] = float(20*np.log10(max(peak, np.finfo(float).tiny)))
                    offset += array.shape[1]
            if np.any(uncertain):
                for k in np.flatnonzero(np.any(uncertain, axis=1)):
                    reasons.append(f"{label}: identification windows disagree at {bins[k]:.12g} Hz")
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            reasons.append(f"{label}: {exc}")
    if reasons:
        return unavailable('undetermined', '; '.join(reasons))
    worst = int(np.argmax(share))
    db = float(20 * np.log10(max(share[worst], np.finfo(float).tiny)))
    return TailShareWitness(db, share, float(bins[worst]),
                            'pass' if share[worst] <= 1e-2 else 'fail', '; '.join(zero_reasons), error, contributions)


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

    def witness(self):
        return tail_share_witness(self.records, self.dt, self.source_end,
                                  self.freqs, freq_max=self.freq_max)


def port_record_witness(records, dt, source_end, freqs, *, freq_max):
    """Judge recorded S-channel matrices, one matrix per port."""
    channels = SettlingChannels(
        tuple((f'port{p}', record) for p, record in enumerate(records)),
        dt, source_end, freqs, freq_max)
    result = channels.witness()
    return result.db, {**result._asdict(), 'route': 's_channels',
                       'worst_record': None, 'per_record_db': {},
                       'skipped_records': [], 'source_dominated': False,
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
            'runs': rows}
