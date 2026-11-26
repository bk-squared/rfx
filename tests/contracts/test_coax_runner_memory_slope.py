"""Count runner input and output array growth against stored source bytes."""

import gc
import json
import sys

import jax
import numpy as np
import pytest

from rfx.api import Simulation


K_BOUND = 3.6


class CensusInstrumentError(RuntimeError):
    """The runner byte census could not establish its measurement conditions."""


def _case(row, n):
    if row == "one":
        sim = Simulation(freq_max=10e9, domain=(.008, .008, .030),
                         dx=.0004, boundary="cpml")
        sim.add_coaxial_port((.004, .004, .015), pin_radius=.000635,
                            outer_radius=.002055)
        return sim.compute_coaxial_line_reflection(
            termination="matched", n_steps=n, freqs=np.linspace(4e9, 6e9, 11))
    from tests.unit.autodiff.test_coax_two_port_ad import (
        _build_small_two_port_sim, FREQ, PROBE_COUNT, PROBE_START_CELLS,
        PROBE_SPACING_CELLS,
    )
    return _build_small_two_port_sim().compute_coaxial_two_port(
        n_steps=n, freqs=FREQ, probe_count=PROBE_COUNT,
        probe_start_cells=PROBE_START_CELLS, probe_spacing_cells=PROBE_SPACING_CELLS)


def _census(args, kw, result):
    sources = tuple(kw.get("sources", ())) + tuple(kw.get("mag_sources", ()))
    leaves, _ = jax.tree_util.tree_flatten_with_path((args, kw, result))
    arrays = [(path, int(value.nbytes)) for path, value in leaves
              if isinstance(value, (jax.Array, np.ndarray))]
    waveform_bytes = sum(size for path, size in arrays
                         if len(path) >= 4
                         and path[0] == jax.tree_util.SequenceKey(1)
                         and path[1] in (jax.tree_util.DictKey("sources"),
                                         jax.tree_util.DictKey("mag_sources"))
                         and path[-1] == jax.tree_util.GetAttrKey("waveform"))
    return dict(source_bytes=sum(int(s.waveform.nbytes) for s in sources),
                shapes=[tuple(s.waveform.shape) for s in sources],
                waveform_bytes=waveform_bytes,
                census=sum(size for _, size in arrays),
                paths={jax.tree_util.keystr(path): size for path, size in arrays})


def measure_row(row, *, install=None):
    """Measure N then 2N, printing the receipt even if measurement raises."""
    import rfx.simulation as simulation

    n, expected_calls = {"one": (1000, 1), "two": (600, 2)}[row]
    receipt = dict(row=row, n=n, calls=[], k=None, growing=[], instrument=[])
    instrument = receipt["instrument"]
    runs = []

    def check(condition, message):
        if not condition:
            instrument.append("census instrument: " + message)

    try:
        with pytest.MonkeyPatch.context() as mp:
            if install is not None:
                install(mp)
            original = simulation.run
            calls = []

            def runner(*args, **kw):
                result = original(*args, **kw)
                calls.append(_census(args, kw, result))
                return result

            mp.setattr(simulation, "run", runner)
            for steps in (n, 2 * n):
                calls = []
                _case(row, steps)
                runs.append(calls)
                check(len(calls) == expected_calls,
                      f"{row} at {steps}: runner call count {len(calls)} != {expected_calls}")
                for index, call in enumerate(calls):
                    label = f"{row} at {steps}, call {index}"
                    check(all(shape == (steps,) for shape in call["shapes"]),
                          f"{label}: waveform sample count differs from {steps}")
                    check(call["waveform_bytes"] == call["source_bytes"],
                          f"{label}: waveform path byte sum differs from source bytes")
        growing = []
        paths_equal = len(runs[0]) == len(runs[1])
        for index, (short, long) in enumerate(zip(*runs)):
            source_bytes = short["source_bytes"]
            check(source_bytes > 0, f"{row}, call {index}: source bytes must be positive")
            check(long["source_bytes"] == 2 * source_bytes,
                  f"{row}, call {index}: source bytes do not double")
            s_src = source_bytes / n
            slope = (long["census"] - short["census"]) / n
            k = slope / s_src if s_src else None
            check(k is not None and k >= 1,
                  f"{row}, call {index}: waveform growth coverage k is below 1 or undefined")
            receipt["calls"].append(dict(
                source_bytes=[source_bytes, long["source_bytes"]],
                census=[short["census"], long["census"]], s_src=s_src, slope=slope, k=k))
            if short["paths"].keys() != long["paths"].keys():
                paths_equal = False
            else:
                growing.extend(dict(path=f"call[{index}]{path}", bytes_n=size,
                                    bytes_2n=long["paths"][path])
                               for path, size in sorted(short["paths"].items())
                               if size != long["paths"][path])
        ks = [call["k"] for call in receipt["calls"] if call["k"] is not None]
        receipt["k"] = max(ks) if ks else None
        receipt["growing"] = growing if paths_equal else "paths differ"
    finally:
        print("KROW " + json.dumps(receipt), file=sys.stderr, flush=True)
        jax.clear_caches()
        gc.collect()
    if instrument:
        raise CensusInstrumentError("; ".join(instrument))
    return receipt


def test_one_port_k():
    row = measure_row("one")
    assert row["k"] <= K_BOUND, f"k pin: one-port k={row['k']} > {K_BOUND}"


def test_two_port_k():
    row = measure_row("two")
    assert row["k"] <= K_BOUND, f"k pin: two-port k={row['k']} > {K_BOUND}"
