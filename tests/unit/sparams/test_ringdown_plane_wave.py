"""Ring-down completion of field probes under a closed-box plane wave.

Physical anchor: a closed TF/SF box injects only through its six faces, by
adding values its faces read from the 1-D incident line. Once every value read
is below source_off_tol of its peak (E and H operands each by its own peak),
the field inside rings freely, which is what the completion's windows need.

The admission rows: a closed box in probe mode, driven by the differentiated
Gaussian at a realized normal direction (+-1, 0, 0), alone; every other plane
wave is refused by name. The check: on a lossy slotted PEC box under the
plane wave, the plane wave's source-off row is the maximum over the main
window of the level formed from the line replay, and the main window that
still holds the pulse refuses.
"""
from functools import lru_cache
import re
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._closed_box_plane_wave import (F0, declare, level, line, pulse_end, recorded_maxima,
                                          same_level, scene_k, windows)
from rfx import GaussianPulse, ringdown as rd
from rfx.errors import RfxUnsupported
from rfx.ringdown._admission import refuse_forward_request, refuse_run_request
from rfx.ringdown._plane_wave import PLANE_WAVE_LABEL, PlaneWaveDrive
from rfx.sources.sources import CustomWaveform
from rfx.stepping.incident_line import face_read_maxima

BINS = np.linspace(0.3 * F0, 2.4 * F0, 101)
CENTRE = (((0.0275, 0.0275, 0.02), "ez"),)
ENTRIES = ("run", "forward")


def probe_spec(probes=CENTRE):
    return rd.RingdownSpec(identification_probes=probes, probe_freqs=BINS)


def refusal(sim_factory, spec, error, n=400):
    """The refusal both entries raise, with the caller's name taken out."""
    messages = []
    for entry in ENTRIES:
        with pytest.raises(error) as caught:
            getattr(sim_factory(), entry)(n_steps=n, ringdown=spec, skip_preflight=True)
        text = str(caught.value)
        assert text.startswith(f"{entry}(ringdown=...)"), text
        messages.append(text[len(entry):])
    assert messages[0] == messages[1]
    return messages[0]


def test_a_plane_wave_alone_is_a_drive():
    sim = declare()
    assert refuse_run_request(sim, probe_spec(), devices=None, until_decay=None,
                              compute_s_params=None) is None
    assert refuse_forward_request(sim, probe_spec(), distributed=False,
                                  emit_time_series=True) is None


def test_the_slab_is_refused_first():
    message = refusal(lambda: declare({"closed_box": False}), probe_spec(), RfxUnsupported)
    assert "the source-off level is measured for the closed-box plane wave (closed_box=True) only" in message


def test_port_mode_is_refused():
    message = refusal(declare, rd.RingdownSpec(), RfxUnsupported)
    assert "completes wire-port S-parameters; a TF/SF plane wave drives this model" in message
    assert "Complete field probes instead (RingdownSpec(probe_freqs=...))" in message


def test_a_continuous_wave_is_refused():
    message = refusal(lambda: declare({"waveform": "continuous_wave"}), probe_spec(), RfxUnsupported)
    assert "driven by a continuous wave, which never turns off" in message


@pytest.mark.parametrize("waveform, name", [
    ("modulated_gaussian", "'modulated_gaussian'"),
    (CustomWaveform(lambda t: jnp.exp(-((t - 2e-9) / 4e-10) ** 2)), "a CustomWaveform"),
], ids=["modulated", "custom"])
def test_another_waveform_is_refused_by_name(waveform, name):
    message = refusal(lambda: declare({"waveform": waveform}), probe_spec(), RfxUnsupported)
    assert ("the source-off level of a closed-box plane wave is measured for the "
            f"differentiated Gaussian only; this plane wave is driven by {name}") in message


def test_oblique_incidence_is_refused():
    message = refusal(lambda: declare({"polarization": "s", "theta_deg": 30, "phi_deg": 20}),
                      probe_spec(), RfxUnsupported)
    assert "source-off not measured for oblique closed-box incidence" in message


def test_incidence_along_y_is_refused():
    message = refusal(lambda: declare({"polarization": "s", "theta_deg": 89.95, "phi_deg": 0}),
                      probe_spec(), RfxUnsupported)
    assert "source-off not measured for closed-box incidence along y" in message


def test_a_plane_wave_beside_a_soft_source_is_refused():
    def model():
        sim = declare()
        sim.add_source((0.02, 0.0275, 0.02), "ez", amplitude_kind="current",
                       waveform=GaussianPulse(f0=F0, bandwidth=0.8))
        return sim
    message = refusal(model, probe_spec(), RfxUnsupported)
    assert ("the source-off check of a TF/SF plane wave together with a soft source is not "
            "measured; complete a model driven by the plane wave alone") in message


def test_a_traced_plane_wave_field_is_refused_upstream():
    def loss(amplitude):
        sim = declare({"amplitude": amplitude})
        res = sim.forward(n_steps=400, ringdown=probe_spec(), skip_preflight=True)
        return jnp.sum(jnp.abs(res.ringdown.probe_spectra) ** 2)
    with pytest.raises(ValueError, match="amplitude: differentiating with respect to the incident source is not supported"):
        jax.grad(loss)(1.0)


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_line_that_cannot_be_formed_is_refused_by_the_plane_wave_label(entry, monkeypatch):
    from rfx._traced_refusal import TracedStructureRefusal
    from rfx.ringdown import _plane_wave
    monkeypatch.setattr(_plane_wave, "plane_wave_entries", lambda plan: [PlaneWaveDrive(None)])
    with pytest.raises(TracedStructureRefusal, match=re.escape(f"the {PLANE_WAVE_LABEL} cannot be judged")):
        getattr(declare(), entry)(n_steps=400, ringdown=probe_spec(), skip_preflight=True)


@lru_cache(maxsize=None)
def completed(entry, n):
    """``entry`` on the slotted box for ``n`` steps, with the source-off rows the check returned."""
    sim, probes, _ = scene_k()
    seen = []
    original = rd.RingdownRun._check_sources_off

    def capture(plan):
        rows = original(plan)
        seen.append(rows)
        return rows

    with patch.object(rd.RingdownRun, "_check_sources_off", capture):
        result = getattr(sim, entry)(n_steps=n, ringdown=probe_spec(probes), skip_preflight=True)
    return result, seen[-1]


def own_row(n):
    """The main-window maximum of r formed here from the line replay, as the check is told to form it."""
    sim, _, _ = scene_k()
    cfg, state, grid = line(sim, n)
    i_h, i_e = face_read_maxima(cfg, state, grid.dx, grid.dt)
    r = level(np.asarray(i_h), np.asarray(i_e))
    return float(np.max(r[int(round(0.5 * n)):n]))


@pytest.mark.parametrize("n", [4000, 4003])
def test_the_plane_wave_row_is_its_face_read_level_over_the_main_window(n):
    result, (rows, _) = completed("run", n)
    report = result.ringdown.report
    assert report.completed, report.failure
    expected = own_row(n)
    assert rows == ((PLANE_WAVE_LABEL, expected),)
    assert report.witness("source_off").value == expected
    assert expected <= 1e-6


def test_the_replayed_level_agrees_with_the_record_at_the_check_length():
    n = 4000
    sim, _, _ = scene_k()
    cfg, state, grid = line(sim, n)
    got_h, got_e = (np.asarray(v) for v in face_read_maxima(cfg, state, grid.dx, grid.dt))
    rec_h, rec_e, _ = recorded_maxima(sim, n)
    got = windows(got_h, got_e, (n,), pulse_end(level(got_h, got_e)))
    rec = windows(rec_h, rec_e, (n,), pulse_end(level(rec_h, rec_e)))
    for key in rec:
        assert same_level(got[key], rec[key]), (key, got[key], rec[key])


def test_a_main_window_holding_the_pulse_refuses():
    message = refusal(lambda: scene_k()[0], probe_spec(scene_k()[1]), ValueError, n=1000)
    assert "the values the TF/SF box faces read from its incident line reach" in message
    assert "(E and H operands each by its own peak)" in message


def test_an_early_window_holding_the_pulse_is_not_formed():
    result, (rows, ratio_long) = completed("run", 2000)
    report = result.ringdown.report
    assert rows[0][0] == PLANE_WAVE_LABEL and rows[0][1] <= 1e-6
    assert ratio_long > 1e-6
    assert not report.witness("WE").judged and report.witness("WE").note.startswith("not formed")
    assert report.witness("W2").judged


def test_run_and_forward_read_the_same_rows_and_spectra():
    run, run_rows = completed("run", 4000)
    fwd, fwd_rows = completed("forward", 4000)
    assert fwd_rows == run_rows
    a = np.asarray(fwd.ringdown.probe_spectra)
    b = np.asarray(run.ringdown.probe_spectra)
    error = np.max(np.abs(a - b), axis=0) / np.max(np.abs(b), axis=0)
    assert np.all(error <= 1e-3), error
