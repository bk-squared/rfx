"""#580: the magnitude-preserving reference_probe loss mode.

Acceptance criterion 2 lives here at the arithmetic level, where it is
deterministic and cheap: a pure magnitude change on the RESPONSE side is
invisible to the legacy per-probe-max proxy (the documented blindness that
made loss/conductivity fits near-degenerate) and visible to the
reference-probe proxy — while a shared drive-amplitude change cancels in
the reference mode (that invariance is the desired calibration, not a
defect). The FDTD-in-the-loop recovery test (acceptance criterion 1) is
test_differentiable_material_fit.py::test_recover_debye_reference_mode_public_entry
(gpu lane, same file as the other FDTD fit tests).
"""
import numpy as np
import jax.numpy as jnp
import pytest

from rfx.differentiable_material_fit import (
    _normalize_probe_spectra,
    differentiable_material_fit,
)


def _synthetic_raw(seed=7, n_probes=3, n_freqs=8):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.normal(size=(n_probes, n_freqs))
        + 1j * rng.normal(size=(n_probes, n_freqs)),
        dtype=jnp.complex64,
    )


def test_per_probe_max_is_blind_to_response_magnitude():
    """The legacy proxy is INVARIANT to scaling a response probe — the
    exact blindness #580 reports (conductivity acts on magnitude)."""
    s_raw = _synthetic_raw()
    scaled = s_raw.at[0].multiply(0.37)
    a = _normalize_probe_spectra(s_raw, 1, (1, 1, 8))
    b = _normalize_probe_spectra(scaled, 1, (1, 1, 8))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                               rtol=1e-6, atol=1e-7)


def test_reference_probe_sees_response_magnitude():
    """The reference mode scales with a response-side magnitude change —
    the discriminating case (pure magnitude offset) from the issue."""
    s_raw = _synthetic_raw()
    scaled = s_raw.at[0].multiply(0.37)
    kw = dict(normalization="reference_probe", reference_probe=2)
    a = np.asarray(_normalize_probe_spectra(s_raw, 1, (1, 1, 8), **kw))
    b = np.asarray(_normalize_probe_spectra(scaled, 1, (1, 1, 8), **kw))
    np.testing.assert_allclose(b[0, 0], 0.37 * a[0, 0], rtol=1e-5)
    assert not np.allclose(a, b, rtol=1e-3)


def test_reference_probe_cancels_shared_drive_amplitude():
    """Scaling ALL probes together (drive amplitude) cancels in the
    reference mode — that invariance is the calibration, kept by design."""
    s_raw = _synthetic_raw()
    kw = dict(normalization="reference_probe", reference_probe=2)
    a = np.asarray(_normalize_probe_spectra(s_raw, 1, (1, 1, 8), **kw))
    b = np.asarray(_normalize_probe_spectra(2.9 * s_raw, 1, (1, 1, 8), **kw))
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_reference_probe_preserves_phase():
    """Complex per-frequency division: a phase ramp applied to the
    response probe appears in the proxy (per-probe-max keeps it too, but
    reference mode must not lose it while fixing magnitude)."""
    s_raw = _synthetic_raw()
    ramp = jnp.exp(1j * jnp.linspace(0.0, 1.2, 8)).astype(jnp.complex64)
    kw = dict(normalization="reference_probe", reference_probe=2)
    a = np.asarray(_normalize_probe_spectra(s_raw, 1, (1, 1, 8), **kw))
    b = np.asarray(_normalize_probe_spectra(s_raw.at[0].multiply(ramp),
                                            1, (1, 1, 8), **kw))
    np.testing.assert_allclose(np.angle(b[0, 0] / a[0, 0]),
                               np.linspace(0.0, 1.2, 8), atol=1e-5)


# ---------------------------------------------------------------------------
# Public-entry validation (no FDTD reached: the mode checks raise before
# the optimization loop; only a tiny grid build runs)
# ---------------------------------------------------------------------------

def _tiny_factory(n_probes):
    from rfx.api import Simulation
    from rfx import GaussianPulse

    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = Simulation(freq_max=5e9, domain=(0.012, 0.012, 0.012))
        sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles)
        sim.add_port((0.003, 0.006, 0.006), "ez",
                     waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
        xs = np.linspace(0.006, 0.010, n_probes)
        for x in xs:
            sim.add_probe((float(x), 0.006, 0.006), component="ez")
        return sim

    return factory


def _call(n_probes, **kw):
    freqs = np.linspace(2e9, 4e9, 4)
    s_meas = np.zeros((1, 1, 4), dtype=complex)
    return differentiable_material_fit(
        _tiny_factory(n_probes), s_meas, freqs, n_debye_poles=1,
        n_iterations=1, verbose=False, **kw,
    )


def test_validation_bad_mode_name():
    with pytest.raises(ValueError, match="per_probe_max"):
        _call(2, normalization="typo_mode")


def test_validation_reference_mode_requires_index():
    with pytest.raises(ValueError, match="requires reference_probe"):
        _call(2, normalization="reference_probe")


def test_validation_reference_must_not_be_response_probe():
    with pytest.raises(ValueError, match="response"):
        _call(2, normalization="reference_probe", reference_probe=0)


def test_validation_reference_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        _call(2, normalization="reference_probe", reference_probe=5)
