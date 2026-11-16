"""Every completed waveguide step is recorded, independent of run length."""

import jax
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.measurement.modal import recorded


N = 40
RECORDS = ("v_probe_t", "v_ref_t", "i_probe_t", "i_ref_t", "v_inc_t")


def _guide(graded):
    sim = Simulation(
        freq_max=12.4e9, domain=(0.06, 0.02286, 11 * 1e-3), dx=1e-3,
        boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=8,
        dz_profile=np.array([1e-3] * 5 + [0.5e-3] * 12) if graded else None,
    )
    sim.add_waveguide_port(
        0.012, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=np.linspace(8.2e9, 12.4e9, 3), f0=10.3e9,
        bandwidth=0.5, name="wg1",
    )
    return sim


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_every_completed_step_matches_the_longer_run(graded, entry):
    def run(steps):
        result = getattr(_guide(graded), entry)(n_steps=steps)
        return result.waveguide_ports["wg1"]

    short, long = run(N), run(N + 7)
    assert int(short.n_steps_recorded) == N
    assert int(long.n_steps_recorded) == N + 7
    for cfg, size in ((short, N), (long, N + 7)):
        for name in RECORDS:
            assert getattr(cfg, name).shape == (size,), name
        assert np.asarray(cfg.v_inc_t)[-1] != 0
        assert np.asarray(cfg.v_probe_t)[-1] != 0
    for name in RECORDS:
        actual = np.asarray(getattr(short, name))
        expected = np.asarray(getattr(long, name))[:N]
        difference = float(np.max(np.abs(actual - expected)))
        peak = float(np.max(np.abs(expected)))
        relative = difference / peak if peak else difference
        assert actual.dtype == expected.dtype, name
        np.testing.assert_array_equal(
            actual.view(np.uint8), expected.view(np.uint8),
            err_msg=f"{name}: max difference={difference}, peak={peak}, relative={relative}",
        )


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_jitted_forward_record_has_the_final_sample(graded):
    sim = _guide(graded)

    @jax.jit
    def drive_record():
        cfg = sim.forward(n_steps=N).waveguide_ports["wg1"]
        return recorded(cfg, "v_inc_t")

    drive = np.asarray(drive_record())
    assert drive.shape == (N,)
    assert drive[-1] != 0
