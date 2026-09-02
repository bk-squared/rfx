"""Passivity enforcement on compute_msl_s_matrix: strict ||S||_2 <= 1, loudly.

The bound is enforced by per-frequency singular-value clipping (nearest
passive matrix in spectral norm). It is constraint enforcement, not a
physics fix: the raw extraction is preserved in S_raw, the per-bin clip in
passivity_correction, a warning names the touched bins, and the raw-value
self-check still audits what was measured. Measured context (PR #462 Sheen
study): the raw coarse-mesh extraction exceeds the bound by up to sigma~3 in
the stopband where the standing-wave-null mask already marks the bins
unreliable — the projection bounds those bins, it does not bless them.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.api._sparams import _project_passive


def _sigma_max(S):
    S = np.asarray(S)
    return np.array([
        np.linalg.svd(S[:, :, k], compute_uv=False)[0]
        for k in range(S.shape[2])
    ])


# ---------------------------------------------------------------------------
# Unit level: the projection helper itself.
# ---------------------------------------------------------------------------

def test_projection_clips_only_the_nonpassive_bins():
    S = np.zeros((2, 2, 3), dtype=complex)
    S[:, :, 0] = [[0.3, 0.5], [0.5, 0.3]]     # passive
    S[:, :, 1] = [[0.0, 3.0], [0.2, 0.0]]     # sigma_max ~ 3.007
    S[:, :, 2] = np.eye(2)                    # exactly at the bound

    S_pass, corr = _project_passive(jnp.asarray(S))
    S_pass = np.asarray(S_pass)
    corr = np.asarray(corr)

    assert np.allclose(S_pass[:, :, 0], S[:, :, 0]), "passive bin must be untouched"
    assert corr[0] == 0.0
    assert corr[1] == pytest.approx(np.linalg.svd(S[:, :, 1], compute_uv=False)[0] - 1.0)
    # Strict bound including the reconstruction round-trip.
    assert np.all(_sigma_max(S_pass) <= 1.0)


@pytest.mark.parametrize("n_ports", [2, 8, 32])
def test_projection_bound_is_strict_at_float32(n_ports):
    """min(sigma, 1) alone reconstructs to 1 + O(eps), and the error GROWS
    with n_ports — an 8*eps margin measured 1.0000000255 at n=8 and
    1.0000006584 at n=32. The 64*eps margin must hold the strict bound after
    the float32 round-trip across port counts."""
    rng = np.random.default_rng(7)
    S = (rng.normal(size=(n_ports, n_ports, 64))
         + 1j * rng.normal(size=(n_ports, n_ports, 64))).astype(np.complex64)
    S *= 2.0  # comfortably non-passive everywhere
    S_pass, _ = _project_passive(jnp.asarray(S))
    assert np.all(_sigma_max(np.asarray(S_pass)) <= 1.0)


# ---------------------------------------------------------------------------
# End to end on a tiny thru line (~seconds of FDTD). A deliberately truncated
# record is the worst case: its raw S carries genuine truncation artifacts.
# ---------------------------------------------------------------------------

def _thru():
    sim = Simulation(freq_max=20e9, domain=(0.012, 0.008, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0.0008)), material="sub")
    sim.add(Box((0.0, 0.0034, 0.0008), (0.012, 0.0046, 0.0010)), material="pec")
    sim.add_msl_port(position=(0.002, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    return sim


FREQS = jnp.linspace(2e9, 18e9, 16)


def test_default_result_is_strictly_passive_and_loud():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _thru().compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)

    assert np.all(_sigma_max(res.S) <= 1.0), "the default S must satisfy the bound at every frequency"

    # Nothing hidden: raw kept, correction recorded, warning fired.
    assert res.S_raw is not None
    assert res.passivity_correction is not None
    assert float(np.max(np.asarray(res.passivity_correction))) > 0.0
    messages = [str(w.message) for w in caught]
    assert any("projected onto the passive set" in m for m in messages)
    # The projection must not silence the artifact diagnoses.
    assert any("settling witness" in m for m in messages)


def test_enforce_passivity_false_returns_the_raw_extraction():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _thru().compute_msl_s_matrix(freqs=FREQS, num_periods=2.0,
                                           enforce_passivity=False)
    assert res.S_raw is None and res.passivity_correction is None
    # The truncated raw extraction genuinely violates the bound — that is
    # exactly what the default projects away.
    assert float(_sigma_max(res.S).max()) > 1.0


def test_projection_agrees_with_offline_projection_of_the_raw():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _thru().compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)
    S_off, corr_off = _project_passive(jnp.asarray(np.asarray(res.S_raw)))
    assert np.allclose(np.asarray(res.S), np.asarray(S_off), atol=1e-6)
    assert np.allclose(np.asarray(res.passivity_correction),
                       np.asarray(corr_off), atol=1e-6)
