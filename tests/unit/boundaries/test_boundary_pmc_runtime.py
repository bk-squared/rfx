"""PMC lives on E nodes; adjacent H samples are physical half-cell samples.

B3b replaces the former H[0]=0 mechanism pins with odd-image checks.
Distributed kernels refuse until they implement the same image.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


def test_tangential_h_has_odd_image_at_pmc_face():
    """The odd-image pair, not the interior Yee sample, averages to zero."""
    spec = BoundarySpec(
        x="cpml", y="cpml",
        z=Boundary(lo="pmc", hi="cpml"),
    )
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
        boundary=spec,
    )
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.005), "ez")
    res = sim.run(n_steps=40)

    from rfx.core.yee import CurlBoundary, h_neighbor
    boundary = CurlBoundary(pmc_faces=frozenset({"z_lo"}))
    for field in (res.state.hx, res.state.hy):
        # H[0] is half a cell INSIDE: preserve it. Interpolation between
        # that sample and its odd image puts H_t=0 on the declared face.
        assert np.max(np.abs(np.asarray(field)[:, :, 0])) > 0
        image = h_neighbor(field, 2, boundary=boundary)
        np.testing.assert_array_equal(np.asarray(field + image)[:, :, 0], 0.)


def test_pmc_runtime_produces_finite_nonzero_trace():
    """Sanity: a PMC + CPML sim injects energy and produces a
    non-zero finite probe trace."""
    spec = BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pmc", hi="cpml"))
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
        boundary=spec,
    )
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.006), "ez")
    ts = np.asarray(sim.run(n_steps=80).time_series)
    assert np.all(np.isfinite(ts))
    assert float(np.max(np.abs(ts))) > 1e-9


def test_mixed_pmc_cpml_seam_is_finite():
    """Mixed-face regression: PMC on z_lo, CPML on z_hi. No NaN / Inf
    in the probe trace. The late-time stability bound + quantitative
    reflection measurements live in the physics harness rather than
    as unit tests (too sensitive to source waveform + probe placement)."""
    spec = BoundarySpec(
        x="cpml", y="cpml",
        z=Boundary(lo="pmc", hi="cpml"),
    )
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
        boundary=spec,
    )
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.006), "ez")
    ts = np.asarray(sim.run(n_steps=100).time_series)[:, 0]
    assert np.all(np.isfinite(ts))


def test_pmc_plus_distributed_forward_refuses_until_image_lands():
    """The old distributed H-zero rule is not the declared-face image."""
    import jax
    dz = np.array([0.5e-3] * 10, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, float(np.sum(dz))),
        dx=0.5e-3, dz_profile=dz, cpml_layers=4,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pmc", hi="cpml")),
    )
    sim.add_source((0.005, 0.005, 0.002), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    # Use a real single-device list. Multi-device sharded-PMC evidence
    # lives in tests/unit/boundaries/test_boundary_pmc_distributed.py.
    devices = [jax.devices()[0]]
    with pytest.raises(NotImplementedError, match="z_lo.*distributed_nu.*magnetic image"):
        sim.forward(n_steps=20, distributed=True, devices=devices,
                    skip_preflight=True)


def test_pmc_and_pec_produce_physically_different_h_at_face():
    """PEC and PMC both retain physical H but solve different wall fields."""
    def _final_state(spec):
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
            boundary=spec,
        )
        sim.add_source((0.005, 0.005, 0.005), "ez")
        sim.add_probe((0.005, 0.005, 0.005), "ez")
        return sim.run(n_steps=40).state

    # Both nearest H planes lie inside the domain; neither is the wall.
    st_pmc = _final_state(BoundarySpec(x="cpml", y="cpml",
                                       z=Boundary(lo="pmc", hi="cpml")))
    st_pec = _final_state(BoundarySpec(x="cpml", y="cpml",
                                       z=Boundary(lo="pec", hi="cpml")))
    hx_pmc_face = float(np.max(np.abs(np.asarray(st_pmc.hx)[:, :, 0])))
    hx_pec_face = float(np.max(np.abs(np.asarray(st_pec.hx)[:, :, 0])))
    assert hx_pmc_face > 0., "the physical half-cell H sample must survive"
    assert hx_pec_face > 0.
    assert not np.array_equal(np.asarray(st_pmc.hx), np.asarray(st_pec.hx))
