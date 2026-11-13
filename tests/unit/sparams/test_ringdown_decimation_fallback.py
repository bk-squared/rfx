"""Closed-form poles and plan invariants for memory-refused windows (#2704)."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from rfx import ringdown as rd
from rfx.ringdown._pencil import _check_run_identification_memory, _identification_plan


# Positive-frequency poles; the real record also contains their conjugates.
TRUE_POLES = -1.0 / np.array([30e-9, 45e-9, 25e-9]) + 2j * np.pi * np.array([180e6, 310e6, 470e6])


def _record():
    rng = np.random.default_rng(2704)
    amplitudes = rng.normal(size=(3, 72)) + 1j * rng.normal(size=(3, 72))
    return 2 * (np.exp(np.outer(np.arange(1400) * 20e-12, TRUE_POLES)) @ amplitudes).real


def _stage1_bytes(kept, channels):
    # Independent transcription of the stage-1 allocation bound.
    columns = max(4, min(int(kept * 0.33), kept - 2))
    rows = kept - columns
    rank = min(rows, columns * channels)
    return 1.87 * 16 * (3 * rows * columns * channels + rows * rank
                        + 2 * rank * columns * channels + 4 * rank * rank)


def test_b1_refused_window_keeps_capacity_and_fits():
    assert rd.decimation_plan(1400, 20e-12, 600e6) == ((), 1400)
    assert _stage1_bytes(1400, 72) > rd.IDENTIFY_MEMORY_BOUND_BYTES
    model = rd.identify(_record(), 20e-12, 0, 1400, freq_max=600e6)
    factors, kept = _identification_plan(1400, 20e-12, 600e6, 72)
    assert (factors, kept) == ((8,), 155)
    samples = 1400
    for factor in factors:
        assert 2 <= factor <= 13
        samples = (samples + factor - 1) // factor - 20
        columns = max(4, min(int(samples * 0.33), samples - 2))
        assert samples >= 10 and min(columns, samples - columns) >= 51
    assert samples == kept
    assert _stage1_bytes(kept, 72) <= rd.IDENTIFY_MEMORY_BOUND_BYTES
    assert (model.factors, model.n_decimated) == ((8,), 155)


def test_b2_fallback_recovers_generating_poles():
    model = rd.identify(_record(), 20e-12, 0, 1400, freq_max=600e6)
    for true in TRUE_POLES:
        pole = model.s[np.argmin(np.abs(model.s - true))]
        assert abs(pole.imag / true.imag - 1) < 1e-4
        assert abs(pole.real / true.real - 1) < 1e-3


def test_b3_accepted_default_plans_do_not_move():
    accepted = rescued = 0
    for n_window, dt, freq_max, channels in product(
        (64, 100, 155, 300, 1000, 1400, 2262, 5000, 20000),
        (1e-15, 1e-13, 5e-12, 2e-11, 1e-10),
        (1e8, 6e8, 3e9, 3e10), (1, 8, 72, 256),
    ):
        default = rd.decimation_plan(n_window, dt, freq_max)
        actual = _identification_plan(n_window, dt, freq_max, channels)
        if _stage1_bytes(default[1], channels) <= rd.IDENTIFY_MEMORY_BOUND_BYTES:
            assert actual == default, (n_window, dt, freq_max, channels)
            accepted += 1
        elif actual != default:
            assert _stage1_bytes(actual[1], channels) <= rd.IDENTIFY_MEMORY_BOUND_BYTES
            rescued += 1
    assert accepted > 0
    assert rescued > 0


def test_b4_identify_preflight_and_static_bound_share_plan(monkeypatch):
    factors, kept = _identification_plan(1400, 20e-12, 600e6, 72)
    model = rd.identify(_record(), 20e-12, 0, 1400, freq_max=600e6)
    assert (model.factors, model.n_decimated) == (factors, kept)
    checked = []
    check_memory = rd._check_identification_memory

    def capture(n_win, shape, **kwargs):
        checked.append((n_win, shape))
        return check_memory(n_win, shape, **kwargs)

    monkeypatch.setattr(rd, "_check_identification_memory", capture)
    plan = SimpleNamespace(
        bins=None, sim=SimpleNamespace(_freq_max=600e6), freq_max=600e6,
        dt=20e-12, n_steps=1400, n_start=0, wire_entries=tuple(range(36)),
        identification_keys=(), lane="uniform",
    )
    _check_run_identification_memory(plan)
    assert checked == [(1400, (kept, 72))]
    columns = max(4, min(int(kept * 0.33), kept - 2))
    assert rd.static_pole_bound(1400, 72, 20e-12, 600e6) == min(kept - columns, 72 * columns)


@pytest.mark.parametrize("bound", [1, float("inf")], ids=["no_fit", "unbounded"])
def test_configured_memory_bound_preserves_default_when_no_fallback(monkeypatch, bound):
    monkeypatch.setattr(rd, "IDENTIFY_MEMORY_BOUND_BYTES", bound)
    assert _identification_plan(1400, 20e-12, 600e6, 72) == ((), 1400)
    if bound == 1:
        with pytest.raises(rd.IdentificationMemoryRefused) as refused:
            rd.identify(_record(), 20e-12, 0, 1400, freq_max=600e6)
        assert refused.value.stage == 1
        assert refused.value.Nd == 1400
        assert refused.value.needed_bytes == _stage1_bytes(1400, 72)
