"""A 30-degree plane wave measures each auxiliary absorber in TMz and TEz."""
import jax
import numpy as np
import pytest

from tests._absorber_witness import tfsf_oblique_auxiliary_reflection
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
