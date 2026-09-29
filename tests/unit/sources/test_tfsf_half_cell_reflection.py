"""Plane waves measure each Bloch and open-domain auxiliary absorber."""
import jax
import numpy as np
import pytest

from tests._absorber_witness import (
    tfsf_methodb_auxiliary_reflection, tfsf_oblique_auxiliary_reflection,
)
from tests._gate_policy import gate_from_envelope


@pytest.mark.parametrize("polarization", ("ez", "ey"))
@pytest.mark.parametrize("direction", ("+x", "-x"))
def test_oblique_auxiliary_return(polarization, direction):
    with jax.default_device(jax.devices("cpu")[0]):
        reflected_db, fit_residual = tfsf_oblique_auxiliary_reflection(
            polarization=polarization, direction=direction)
    assert fit_residual < 1e-2, f"unresolved forward/backward modes: {fit_residual}"
    # The 2500-step record has a 7.48e-5 envelope at both 30 and 60 layers.
    # Keep its measured duration/depth dependence; use the shared margin.
    amplitude_limit = gate_from_envelope(7.48e-5, quantum=1e6)
    assert np.all(10**(reflected_db / 20) < amplitude_limit), f"{polarization} {direction}: R = {reflected_db} dB"


@pytest.mark.parametrize("theta_deg", (0, 30, 60))
@pytest.mark.parametrize("face", ("lo", "hi"))
def test_methodb_auxiliary_return(theta_deg, face):
    with jax.default_device(jax.devices("cpu")[0]):
        reflected_db = tfsf_methodb_auxiliary_reflection(
            theta_deg, face, np.arange(2, 30.01, 0.5) * 1e9)
    jax.clear_caches()
    # The 20-layer sampled envelope is 5.61422e-6 over 2--30 GHz, both faces,
    # at 0/30/60 degrees. At 40 layers the worst return is -117.260 dB.
    # The shared policy gives ceil(1.5*5.61422e-6*1e8)/1e8 = 8.43e-6.
    amplitude_limit = gate_from_envelope(5.61422e-6, quantum=1e8)
    assert np.all(np.isfinite(reflected_db)), f"{theta_deg} {face}: nonfinite return"
    assert np.all(10**(reflected_db / 20) < amplitude_limit), f"{theta_deg} {face}: R = {reflected_db} dB"
