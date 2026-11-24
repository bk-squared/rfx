"""Completed probe spectra divided by the closed-box plane wave's incident spectrum.

Physical anchor: in a closed TF/SF box at normal incidence along x the 3-D
lattice carries the 1-D line's plane wave, so in the empty box the E field
at a node equals the line's E along the polarization at that x plane. The
field a structure produces at a probe, over that incident field, is its
transfer function. The divisor is checked against the empty box's own field;
the normalized spectra are the raw ones divided by it; the error witness is
read on the divided spectra; and ``run``, ``forward`` and its traced forms
return the same quotient.

With ``polarization="s"`` or ``"p"`` at ``phi_deg != 0`` (theta 0) the
divisor is the incident E along e_hat = (0, -sin phi, cos phi), so an empty
box's quotient at an ez channel is cos phi, not 1. The scenes here use the
``ez`` polarization, where it is 1.
"""
from functools import lru_cache
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._closed_box_plane_wave import EZ_EZ, F0, declare, slotted_box
from rfx import ringdown as rd

BINS = np.linspace(0.3 * F0, 2.4 * F0, 101)
WINDOW_START = 0.5


def declared_record_length():
    """Smallest half-record window covering the scale-1.01 fill's delay."""
    sim, _, _ = slotted_box(channels=EZ_EZ)
    grid = sim._build_realized_grid()
    lengths = []
    for axis in 'xyz':
        cells = np.asarray(grid.cells(axis), np.float64)[:-1]
        lo, hi = getattr(grid, f'pad_{axis}_lo'), getattr(grid, f'pad_{axis}_hi')
        lengths.append(np.sum(cells[lo:len(cells)-hi], dtype=np.float64))
    fill = sim._materials['fill']
    index = np.sqrt(1.01 * fill.eps_r * fill.mu_r)
    required = 2.05 * 2 * np.linalg.norm(lengths) * index / 299792458.0 / float(grid.dt)
    upper = 2 * int(np.ceil(required))
    return next(n for n in range(upper - 2, upper + 2)
                if n - round(WINDOW_START * n) >= required)


N = declared_record_length()


def spec(probes, normalization="incident"):
    return rd.RingdownSpec(identification_probes=probes, probe_freqs=BINS,
                           probe_normalization=normalization, window_start=WINDOW_START)


def channel_error(a, b):
    """max_f |a - b| / max_f |b| per channel."""
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b), axis=0) / np.max(np.abs(b), axis=0)


def own_spectrum(series, dt):
    """dt sum_n y_n exp(-j 2 pi f (n + 1) dt) per column, numpy float64."""
    y = np.asarray(series, dtype=np.float64)
    t = (np.arange(y.shape[0], dtype=np.float64) + 1.0) * dt
    return dt * (np.exp(-2j * np.pi * BINS[:, None] * t[None, :]) @ y)


@lru_cache(maxsize=None)
def scene_run(normalization):
    """``run`` on the slotted box with two ez channels, and the raw spectra its witnesses were built from."""
    sim, probes, _ = slotted_box(channels=EZ_EZ)
    captured = {}
    two, early = rd.two_window_witness, rd.early_start_witness

    def two_spy(*args, **kwargs):
        captured["two"] = two(*args, **kwargs)
        return captured["two"]

    def early_spy(*args, **kwargs):
        captured["early"] = early(*args, **kwargs)
        return captured["early"]

    with patch.object(rd, "two_window_witness", two_spy), patch.object(rd, "early_start_witness", early_spy):
        result = sim.run(n_steps=N, ringdown=spec(probes, normalization), skip_preflight=True)
    assert result.ringdown.report.completed, result.ringdown.report.failure
    return result, captured, probes, float(sim._build_grid().dt)


def test_the_divisor_is_the_incident_field_of_the_empty_box():
    result, _, probes, dt = scene_run("incident")
    divisor = result.ringdown.report.incident_spectra
    twin = declare()
    for position, component in probes:
        twin.add_probe(position=position, component=component)
    assert float(twin._build_grid().dt) == dt
    empty = own_spectrum(twin.run(n_steps=N, skip_preflight=True).time_series, dt)
    per_bin = np.abs(empty - divisor) / np.abs(divisor)
    at_peak = np.max(np.abs(empty - divisor), axis=0) / np.max(np.abs(divisor), axis=0)
    print(f"divisor against the empty box: per bin {np.max(per_bin, axis=0)}, at the peak {at_peak}",
          file=sys.stderr)
    assert divisor.shape == (BINS.size, 2) and divisor.dtype == np.complex128
    assert np.all(per_bin <= 1e-3), np.max(per_bin, axis=0)
    assert np.all(at_peak <= 1e-4), at_peak


def test_the_normalized_spectra_are_the_raw_ones_over_the_reported_divisor():
    raw, _, _, _ = scene_run(None)
    norm, _, _, _ = scene_run("incident")
    a, b = raw.ringdown.report, norm.ringdown.report
    assert a.probe_normalization is None and a.incident_spectra is None
    assert b.probe_normalization == "incident"
    expected = np.asarray(raw.ringdown.probe_spectra) / b.incident_spectra
    assert np.array_equal(np.asarray(norm.ringdown.probe_spectra), expected)
    assert repr(a.poles) == repr(b.poles)
    assert a.identification_channels == b.identification_channels
    assert a.identification_amplitude_shares == b.identification_amplitude_shares
    assert (a.rank, a.n_kept, a.decimation_factors) == (b.rank, b.n_kept, b.decimation_factors)
    assert a.witness("source_off") == b.witness("source_off")


def test_an_ey_channel_shares_the_ez_divisor_of_its_x_plane():
    sim, probes, _ = slotted_box(channels=EZ_EZ)
    p1 = probes[0][0]
    plan = rd.RingdownRun(sim, spec(((p1, "ez"), (p1, "ey"))), lane="uniform", n_steps=N,
                          grid=sim._build_grid())
    (_, ez_index), (_, ey_index) = plan.identification_keys
    assert ez_index[0] == ey_index[0]
    assert np.array_equal(plan.incident_spectra[:, 0], plan.incident_spectra[:, 1])


def test_we_reads_the_completion_error_on_the_normalized_scale():
    result, captured, _, _ = scene_run("incident")
    report = result.ringdown.report
    divisor = report.incident_spectra
    spread = np.max(np.abs(divisor), axis=0) / np.min(np.abs(divisor), axis=0)
    main, early = np.asarray(captured["two"].spectra), np.asarray(captured["early"].spectra_long)
    expected = float(np.max(channel_error(early / divisor, main / divisor)))
    raw_scale = float(np.max(channel_error(early, main)))
    we = report.witness("WE")
    print(f"WE {we.value!r}, own normalized {expected!r}, raw scale {raw_scale!r}, |I| spread {spread}",
          file=sys.stderr)
    assert np.all(spread >= 10), spread
    assert we.judged and "normalized by the incident spectrum" in we.rule
    assert we.value == pytest.approx(expected, rel=1e-12, abs=0)
    assert abs(raw_scale - expected) > 0.1 * expected


@lru_cache(maxsize=None)
def scaled_forward():
    """``forward`` on the slotted box with two ez channels, the fill's permittivity scaled."""
    sim, probes, inner = slotted_box(channels=EZ_EZ)
    eps = jnp.asarray(np.asarray(sim.realized_geometry().materials.eps_r), jnp.float32)
    mask = np.zeros(eps.shape, dtype=bool)
    mask[tuple(slice(*inner[a]) for a in "xyz")] = True
    fill = jnp.asarray(mask)

    def evaluate(scale):
        return sim.forward(n_steps=N, ringdown=spec(probes), skip_preflight=True,
                           eps_override=jnp.where(fill, eps * scale, eps)).ringdown
    return evaluate


SCALES = (1.0, 1.01)


@lru_cache(maxsize=None)
def untraced(scale):
    return scaled_forward()(jnp.float32(scale))


@pytest.mark.parametrize('scale', SCALES)
def test_scaled_fill_completes_at_the_declared_record(scale):
    result = untraced(scale)
    assert result.report.completed, result.report.failure


def test_run_and_forward_return_the_same_normalized_spectra():
    run, _, _, _ = scene_run("incident")
    fwd = untraced(1.0)
    assert np.array_equal(fwd.report.incident_spectra, run.ringdown.report.incident_spectra)
    error = channel_error(fwd.probe_spectra, run.ringdown.probe_spectra)
    print(f"forward against run: {error}", file=sys.stderr)
    assert np.all(error <= 1e-4), error


@pytest.mark.parametrize("transform", ["jit", "vmap"])
def test_a_traced_forward_returns_the_untraced_normalized_spectra(transform):
    evaluate = scaled_forward()
    if transform == "jit":
        got = [jax.jit(evaluate)(jnp.float32(SCALES[0])).probe_spectra]
    else:
        batched = jax.jit(jax.vmap(evaluate))(jnp.asarray(SCALES, jnp.float32)).probe_spectra
        got = [batched[i] for i in range(len(SCALES))]
    for scale, spectra in zip(SCALES, got):
        error = channel_error(spectra, untraced(scale).probe_spectra)
        print(f"{transform} at {scale}: {error}", file=sys.stderr)
        assert np.all(error <= 1e-4), (scale, error)
