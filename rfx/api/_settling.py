"""Select the records actually read by returned spectra for settling."""
import numpy as np

from rfx.api._spec import ForwardResult
from rfx.sparams._record_selection import s_read_records, build_settling_witness


def run_settling_witness(self, result):
    """Judge retained records at the result's spectral read bins.

    sensitivities are judged by a separate witness
    """
    from rfx.sparams._tail_witness import result_read_bins

    prior = (getattr(result, "settling_witness", None)
             if not isinstance(result, ForwardResult) else None)
    if prior and prior.get("run_records_scored"):
        return result.settling_db, prior

    # run()'s port records repeat its S bins in float32; reading them again
    # would split each bin of a float64 request in two.
    read = result
    if not isinstance(result, ForwardResult) and hasattr(result, "_replace"):
        read = result._replace(wire_port_sparams=None, lumped_port_sparams=None)
    bins = result_read_bins(read)
    dt = getattr(result, "dt", None) or getattr(getattr(result, "grid", None), "dt", None)
    rows = []
    if prior:
        prior_bins = np.asarray(getattr(result, "freqs", None), dtype=float).reshape(-1)
        if len(prior['share_per_bin']) != bins.size:
            aligned = {}
            for key in ('share_per_bin', 'error_per_bin', 'drive_share_per_bin'):
                if key == 'drive_share_per_bin' and not len(prior.get(key, ())):
                    aligned[key] = ()
                    continue
                values = np.zeros(bins.size)
                if len(prior.get(key, ())):
                    np.maximum.at(values, np.searchsorted(bins, prior_bins), prior[key])
                aligned[key] = values
            prior = {**prior, **aligned}
        rows.append(prior)
    db, detail = build_settling_witness(
        result, bins=bins, dt=dt, freq_max=self._freq_max,
        probe_info=self._settling_probe_selection(result),
        source_end=self._settling_source_end(result),
        port_source_end=self._settling_source_end(result, records=self._ports),
        prior_rows=rows)
    return db, {**detail, 'run_records_scored': True}

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
