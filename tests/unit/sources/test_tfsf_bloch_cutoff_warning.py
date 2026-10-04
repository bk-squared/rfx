"""Construction-only contract for the oblique Bloch cutoff warning."""

import math
import warnings

import pytest

from rfx import Simulation


@pytest.mark.parametrize("angle,bandwidth,waveform,method,f0", [
    (43.6, 0.3, "modulated_gaussian", "bloch", 2.9e9),
    (43.6, 0.1, "modulated_gaussian", "bloch", 2.9e9),
    (20.0, 0.3, "modulated_gaussian", "bloch", 2.9e9),   # -41.8 dB: warns
    (43.6, 0.12, "modulated_gaussian", "bloch", 2.9e9),  # -58.1 dB: warns
    (43.6, 0.11, "modulated_gaussian", "bloch", 2.9e9),  # -69 dB: silent
    (0.0, 0.3, "modulated_gaussian", "bloch", 2.9e9),
    (0.005, 0.5, "modulated_gaussian", "bloch", 2.9e9),
    (43.6, 0.3, "modulated_gaussian", "methodB", 2.9e9),
    (70.0, 0.5, None, "bloch", None),
    (-43.6, 0.3, "modulated_gaussian", "bloch", 2.9e9),
])
def test_bloch_cutoff_warning(angle, bandwidth, waveform, method, f0):
    sim = Simulation(freq_max=5.8e9, domain=(0.285, 0.020, 0.010),
                     dx=0.005, cpml_layers=8)
    # Both names reach the SAME analytic modulated drive in tfsf_2d.py.
    # Fourier transform: |S(f)| / |S(f0)| = exp(-[pi*tau*(f-f0)]**2).
    source_f0 = 2.9e9
    cutoff = source_f0 * abs(math.sin(math.radians(angle)))
    tau = 1 / (math.pi * source_f0 * bandwidth)
    ratio = math.exp(-(math.pi * tau * (cutoff - source_f0)) ** 2)
    expected_db = 20 * math.log10(ratio)
    if waveform is None:
        assert ratio == pytest.approx(0.9855573897392684, rel=1e-10)
    kwargs = {} if waveform is None else {"waveform": waveform}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.add_tfsf_source(f0=f0, angle_deg=angle, bandwidth=bandwidth,
                            method=method, **kwargs)
    should_warn = method == "bloch" and abs(angle) > 0.01 and ratio > 1e-3
    assert len(caught) == int(should_warn)
    if should_warn:
        assert caught[0].category is UserWarning
        message = str(caught[0].message)
        assert f"angle={angle:g} deg" in message
        assert f"bandwidth={bandwidth:g}" in message
        assert f"{expected_db:.2f} dB relative to f0" in message
        assert f"f_c={cutoff / 1e9:.6g} GHz" in message
        assert "a rectangular-window DFT of this path's probe time series picks up" in message
        assert "non-decaying component at f_c = f0·sinθ" in message
        assert "depends on the record length" in message
        assert "Narrow the bandwidth" in message
        assert "taper the tail of the time series before the DFT" in message
