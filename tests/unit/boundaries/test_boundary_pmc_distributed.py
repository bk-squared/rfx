"""Distributed PMC preserves physical half-cell H and the declared-face image."""
import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


def _require_two_devices():
    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2
    return devices


def test_pmc_distributed_nu_z_lo():
    devices = _require_two_devices()
    sim = Simulation(
        freq_max=15e9, domain=(.012, .010, .008), dx=.001,
        dx_profile=np.full(12, .001),
        boundary=BoundarySpec(x="pec", y="pec", z=Boundary("pmc", "pec")),
    )
    sim.add_source((.003, .004, .002), "ez", amplitude_kind="field")
    single = sim.run(n_steps=20, compute_s_params=False, skip_preflight=True)
    multi = sim.run(n_steps=20, devices=devices, compute_s_params=False, skip_preflight=True)
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        np.testing.assert_array_equal(getattr(multi.state, component), getattr(single.state, component))
    for field in (multi.state.hx, multi.state.hy):
        assert np.max(np.abs(np.asarray(field)[:, :, 0])) > 0


@pytest.mark.parametrize("face", ["x_lo", "z_lo"])
@pytest.mark.parametrize("skip_preflight", [False, True])
def test_pmc_distributed_v2_admits_declared_face(face, skip_preflight):
    """Public admission accepts the same image with or without preflight."""
    devices = _require_two_devices()
    spec = BoundarySpec(x=Boundary("pmc", "cpml") if face == "x_lo" else "cpml",
                        y="cpml", z=Boundary("pmc", "pec") if face == "z_lo" else "cpml")
    sim = Simulation(freq_max=5e9, domain=(.08, .04, .12), dx=.005, boundary=spec)
    sim.add_source((.04, .02, .06), "ex")
    sim.add_probe((.045, .02, .06), "ex")
    result = sim.run(n_steps=30, devices=devices, skip_preflight=skip_preflight)
    assert np.isfinite(np.asarray(result.time_series)).all()
    assert np.max(np.abs(np.asarray(result.time_series))) > 0
