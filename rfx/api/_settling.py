"""Select the records actually read by returned spectra for settling."""
import numpy as np

from rfx.api._spec import ForwardResult
from rfx.sparams._record_selection import s_read_records, record_selection_detail


def run_settling_witness(self, result):
    """Judge retained records at the result's spectral read bins.

    sensitivities are judged by a separate witness
    """
    from rfx.probes.settling import probe_record_settling_witness
    from rfx.sparams._tail_witness import (
        combine_witness_details, port_record_witness, result_read_bins,
    )
    from rfx.sources.waveguide_port import settling_db_from_port_records

    prior = (getattr(result, "settling_witness", None)
             if not isinstance(result, ForwardResult) else None)
    if prior and prior.get("run_records_scored"):
        return result.settling_db, prior

    bins = result_read_bins(result)
    dt = getattr(result, "dt", None) or getattr(getattr(result, "grid", None), "dt", None)
    rows = []
    if prior:
        prior_bins = np.asarray(getattr(result, "freqs", None), dtype=float).reshape(-1)
        if len(prior['share_per_bin']) != bins.size:
            aligned = {}
            for key in ('share_per_bin', 'error_per_bin'):
                values = np.zeros(bins.size)
                np.maximum.at(values, np.searchsorted(bins, prior_bins), prior[key])
                aligned[key] = values
            prior = {**prior, **aligned}
        rows.append(prior)
    port_records = s_read_records(result)
    if port_records:
        _, detail = port_record_witness(
            port_records, dt,
            self._settling_source_end(result, records=self._ports),
            bins, freq_max=self._freq_max)
        rows.append(detail)
    if getattr(result, "waveguide_ports", None):
        _, detail = settling_db_from_port_records(
            result.waveguide_ports.values(), freqs=bins,
            freq_max=self._freq_max, return_detail=True)
        rows.append(detail)
    series = getattr(result, "time_series", None)
    selection = self._settling_probe_selection(result)
    probe_db, probe_detail = probe_record_settling_witness(
        series, selection,
        source_end_index=self._settling_source_end(result),
        dt=dt, freqs=bins, freq_max=self._freq_max)
    if selection:
        rows.append(probe_detail)
    elif rows and _reads_unrecorded_fields(result):
        rows.append({**probe_detail, 'status': 'absent', 'db': None,
                     'worst_freq_hz': None, 'floor_amplitude': None,
                     'share_per_bin': np.zeros(bins.size),
                     'error_per_bin': np.zeros(bins.size),
                     'reason': 'far-field or DFT-plane frequencies are read but no '
                               'probe records the fields they come from'})
    if not rows:
        return probe_db, {**probe_detail, **record_selection_detail(result), 'run_records_scored': True}
    detail = combine_witness_details(*rows) if len(rows) > 1 else rows[0]
    if len(rows) > 1 and bins.size:
        shares = np.maximum.reduce([row['share_per_bin'] for row in rows])
        errors = np.maximum.reduce([row['error_per_bin'] for row in rows])
        detail = {**detail, 'share_per_bin': shares, 'error_per_bin': errors,
                  'worst_freq_hz': (float(bins[np.argmax(shares)])
                                    if np.any(shares) or detail['status'] == 'pass'
                                    else float('nan'))}
    return detail['db'], {**detail, **record_selection_detail(result), 'run_records_scored': True}

def settling_source_end(self, result, records=None):
    from rfx.probes.settling import simulation_source_end_step

    series = getattr(result, "time_series", None)
    shape = getattr(series, "shape", ())
    port_records = s_read_records(result)
    if port_records and (records is not None or not shape):
        series = port_records[0].samples
        shape = series.shape
    if not shape:
        return None
    grid = getattr(result, "grid", None)
    dt = getattr(result, "dt", None)
    if dt is None:
        dt = getattr(grid, "dt", None)
    if records is None:
        internal = getattr(self, "_internal_probe_indices", ()) or ()
        count = 1 if len(shape) == 1 else shape[1]
        records = [p for i, p in enumerate(self._probes[:count]) if i not in internal]
    return simulation_source_end_step(
        self, series.shape[0], dt, records, grid=grid,
        waveguide_configs=getattr(result, "waveguide_ports", None))

def _reads_unrecorded_fields(result) -> bool:
    """True when NTFF, DFT-plane, flux or current-moment outputs are read."""
    if getattr(result, "ntff_box", None) is not None:
        return True
    if getattr(result, "current_moment_monitor", None) is not None:
        return True
    return any(getattr(result, f, None) for f in ("dft_planes", "flux_monitors"))
