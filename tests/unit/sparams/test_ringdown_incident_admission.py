"""Admission of the incident normalization of completed probe channels.

Physical anchor: in a closed TF/SF box at normal incidence along x, the
incident wave is uniform in y and z, so the incident E at any total-field
node is the 1-D line's E along the polarization at that x plane. Only a
transverse E channel inside the total-field box has that incident field to
divide by, and only a bin where its spectrum stands clear of the line's
float32 rounding can be divided. Every other request is refused by name,
before the run, the same way on ``run`` and ``forward``.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._closed_box_plane_wave import DX, EZ_EZ, F0, declare, slotted_box
from rfx import GaussianPulse, Simulation, ringdown as rd
from rfx.errors import RfxRefusal, RfxUnsupported
from rfx.ringdown._admission import (refuse_incident_bins, refuse_incident_channels,
                                     refuse_run_request)

BINS = np.linspace(0.3 * F0, 2.4 * F0, 101)
CENTRE = (((0.0275, 0.0275, 0.02), "ez"),)
ENTRIES = ("run", "forward")
BOUNDS = {"x": (11, 27), "y": (11, 27), "z": (11, 21)}


def normalized_spec(probes=CENTRE, bins=BINS):
    return rd.RingdownSpec(identification_probes=probes, probe_freqs=bins,
                           probe_normalization="incident")


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


def soft_source_box():
    sim = Simulation(freq_max=1.5e9, domain=(.055, .055, .040), dx=DX, boundary="cpml",
                     cpml_layers=8, snap="declared")
    sim.add_source((0.02, 0.0275, 0.02), "ez", amplitude_kind="current",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    return sim


def test_normalization_without_probe_freqs_is_refused():
    from tests.unit.sparams.test_ringdown_run import _box
    with pytest.raises(RfxRefusal, match=re.escape(
            "run(ringdown=...): probe_normalization applies to completed probe channels (probe_freqs=)")):
        refuse_run_request(_box("uniform"), rd.RingdownSpec(probe_normalization="incident"),
                           devices=None, until_decay=None, compute_s_params=True)
    message = refusal(declare, rd.RingdownSpec(probe_normalization="incident"), RfxRefusal)
    assert "probe_normalization applies to completed probe channels (probe_freqs=)" in message


def test_an_unknown_normalization_is_refused_by_the_spec():
    with pytest.raises(ValueError, match=re.escape("probe_normalization must be 'incident' or None")):
        rd.RingdownSpec(identification_probes=CENTRE, probe_freqs=BINS, probe_normalization="scattered")


def test_normalization_without_a_plane_wave_is_refused():
    reason = ("the incident normalization divides by the plane wave's incident spectrum; "
              "this model has no plane wave")
    with pytest.raises(RfxRefusal, match=re.escape(f"run(ringdown=...): {reason}")):
        refuse_run_request(soft_source_box(), normalized_spec(), devices=None, until_decay=None,
                           compute_s_params=None)
    assert reason in refusal(soft_source_box, normalized_spec(), RfxRefusal)


def test_a_plane_wave_beside_a_soft_source_keeps_its_own_refusal():
    def model():
        sim = declare()
        sim.add_source((0.02, 0.0275, 0.02), "ez", amplitude_kind="current",
                       waveform=GaussianPulse(f0=F0, bandwidth=0.8))
        return sim
    message = refusal(model, normalized_spec(), RfxUnsupported)
    assert ("the source-off check of a TF/SF plane wave together with a soft source is not "
            "measured; complete a model driven by the plane wave alone") in message


def test_a_channel_other_than_ey_or_ez_is_refused():
    keys = (("ez", (20, 18, 16)), ("hy", (17, 21, 16)))
    with pytest.raises(RfxUnsupported, match=re.escape(
            "forward(ringdown=...): the incident normalization divides transverse E channels (ey, ez) "
            "by the incident E; hy at cell (17, 21, 16) is not one.")):
        refuse_incident_channels("forward", keys, BOUNDS)


@pytest.mark.parametrize("x", [11, 27])
def test_a_channel_on_an_x_face_of_the_box_is_admitted(x):
    assert refuse_incident_channels("run", (("ey", (x, 18, 16)), ("ez", (x, 18, 16))), BOUNDS) is None


@pytest.mark.parametrize("index", [(10, 18, 16), (28, 18, 16), (20, 10, 16), (20, 18, 22)])
def test_a_channel_outside_the_box_is_refused(index):
    with pytest.raises(RfxUnsupported, match=re.escape(
            f"run(ringdown=...): ez at cell {index} lies outside the total-field box")) as caught:
        refuse_incident_channels("run", (("ez", (20, 18, 16)), ("ez", index)), BOUNDS)
    assert "where the incident field is not present" in str(caught.value)


def gaussian_record():
    """A sampled Gaussian; its spectrum relative to dt sum |i| is close to exp(-(8 pi f dt)^2)."""
    n = np.arange(200, dtype=np.float64)
    return np.exp(-((n - 60.0) / 8.0) ** 2).astype(np.float32)


def own_rho(record, freqs, dt):
    i = np.asarray(record, dtype=np.float64)
    t = (np.arange(i.size) + 1.0) * dt
    spectrum = dt * (np.exp(-2j * np.pi * np.asarray(freqs)[:, None] * t[None, :]) @ i)
    return np.abs(spectrum) / (dt * np.sum(np.abs(i)))


def test_a_bin_where_the_incident_spectrum_is_below_its_limit_is_refused():
    dt = 1e-12
    bins = np.asarray([1e10, 1.48e11])
    rho = own_rho(gaussian_record(), bins, dt)
    assert rho[0] > 0.5 and 1e-8 < rho[1] < 1e-5, rho
    with pytest.raises(RfxUnsupported, match=re.escape(
            f"run(ringdown=...): the incident spectrum at {bins[1]:.6g} Hz is {rho[1]:.1e} of its "
            "bound at ez cell (20, 18, 16); the normalized value there is not formed.")):
        refuse_incident_bins("run", (("ez", (20, 18, 16)),), [gaussian_record()], bins, dt)


def test_a_traced_record_is_refused_by_name():
    def divide(record):
        return refuse_incident_bins("forward", (("ez", (20, 18, 16)),), [record], BINS, 4.77e-12)
    with pytest.raises(RfxUnsupported, match=re.escape(
            "forward(ringdown=...): the incident normalization forms its divisor on the host from the "
            "plane wave's line, which is traced here.")):
        jax.jit(divide)(jnp.asarray(gaussian_record()))


@pytest.fixture
def no_record(monkeypatch):
    """Fail the test if an incident record is replayed."""
    from rfx.sources import tfsf_discrete_plane_wave as pw
    calls = []
    original = pw.normal_incident_record

    def spy(*args, **kwargs):
        calls.append(args[-1])
        return original(*args, **kwargs)
    monkeypatch.setattr(pw, "normal_incident_record", spy)
    return calls


def test_an_h_channel_is_refused_on_both_entries_before_any_record(no_record):
    sim, probes, _ = slotted_box()
    message = refusal(lambda: slotted_box()[0], normalized_spec(probes), RfxUnsupported, n=4000)
    assert "by the incident E; hy at cell" in message and "is not one." in message
    assert no_record == []


def test_a_channel_outside_the_box_is_refused_on_both_entries(no_record):
    sim, probes, _ = slotted_box(channels=EZ_EZ)
    grid = sim._build_grid()
    outside = ((grid.node_of("x", 10), probes[0][0][1], probes[0][0][2]), "ez")
    message = refusal(lambda: slotted_box(channels=EZ_EZ)[0], normalized_spec((probes[0], outside)),
                      RfxUnsupported, n=4000)
    assert "ez at cell (10, " in message and "lies outside the total-field box" in message
    assert no_record == []


def test_a_bin_below_the_limit_is_refused_on_both_entries():
    message = refusal(declare, normalized_spec(bins=np.append(BINS, 3e9)), RfxUnsupported, n=4000)
    assert re.search(r"the incident spectrum at 3e\+09 Hz is \S+ of its bound at ez cell", message), message
