"""A guided TMz pulse measures the optional ADI absorber at a small timestep."""
import jax
import numpy as np
import pytest

from tests._absorber_witness import adi_cpml_reflection


@pytest.mark.parametrize("axis", ("x", "y"))
@pytest.mark.parametrize("n_layers", (8, 16))
def test_adi_cpml_return_at_small_timestep(n_layers, axis):
    with jax.default_device(jax.devices("cpu")[0]):
        reflected_db = adi_cpml_reflection(n_layers, axis, np.array([2e9, 10e9, 30e9]))
    jax.clear_caches()
    # The matched-absorber goal is -60 dB. The 8/16-layer, per-face records
    # and PEC calibration resolve that goal at CFL factor 0.01 only.
    assert np.all(np.isfinite(reflected_db)), f"{axis}: nonfinite return"
    assert np.all(reflected_db < -60), f"{axis} low/high R at 2/10/30 GHz: {reflected_db} dB"
