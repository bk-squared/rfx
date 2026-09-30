"""Requested graded forward bins, numerical subset witness and AD (#1410)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.sources.sources import GaussianPulse


def _sim(*, lumped=False):
    sim = Simulation(freq_max=10e9, domain=(8e-3, 8e-3, 8e-3), dx=1e-3,
                     dz_profile=np.linspace(0.8e-3, 1.2e-3, 8), boundary="pec")
    sim.add_port(position=(4e-3, 4e-3, 3e-3), component="ez", impedance=50,
                 extent=None if lumped else 2e-3, excite=True,
                 waveform=GaussianPulse(f0=2e9, bandwidth=0.9))
    return sim


def _forward(sim, **kwargs):
    return sim.forward(n_steps=100, skip_preflight=True, **kwargs)


@pytest.mark.parametrize("lumped", [False, True], ids=["wire", "lumped"])
def test_requested_port_bins(lumped):
    request = np.array([1.1e9, 1.7e9, 2.3e9], dtype=np.float64)
    result = _forward(_sim(lumped=lumped), port_s11_freqs=request)
    np.testing.assert_array_equal(result.freqs, request.astype(np.float32))
    assert result.freqs.dtype == jnp.float32
    assert result.s_params.shape == (1, 1, len(request))
    assert np.isfinite(result.s_params).all()
    assert len(result.wire_port_sparams) == 1
    for accumulator in result.wire_port_sparams[0][1]:
        assert accumulator.shape == (len(request),)
        assert np.isfinite(accumulator).all()


def test_port_bins_subset_matches_default():
    sim = _sim()
    default = _forward(sim)
    assert default.s_params.shape == (1, 1, 50)
    np.testing.assert_array_equal(default.freqs,
                                  np.linspace(1e9, 10e9, 50).astype(np.float32))
    indices = np.array([0, 7, 19, 31, 49])
    request = np.asarray(default.freqs)[indices]
    selected = _forward(sim, port_s11_freqs=request)
    np.testing.assert_array_equal(selected.freqs, request)
    expected = np.asarray(default.s_params)[..., indices]
    delta = np.max(np.abs(np.asarray(selected.s_params) - expected))
    print(f"subset max |dS|={delta:.9g}")
    np.testing.assert_allclose(selected.s_params, expected, rtol=0, atol=2e-7)
    for actual, baseline in zip(selected.wire_port_sparams[0][1],
                                default.wire_port_sparams[0][1]):
        np.testing.assert_allclose(actual, baseline[indices], rtol=2e-7, atol=0)


def test_narrow_port_band_gradient():
    sim = _sim()
    request = np.linspace(1.95e9, 2.05e9, 21, dtype=np.float32)
    eps = jnp.full(sim._build_nonuniform_grid().shape, 2.0)

    def loss(eps_override):
        result = _forward(sim, port_s11_freqs=request, eps_override=eps_override)
        return jnp.sum(jnp.abs(result.s_params) ** 2), result.s_params

    (value, spectrum), gradient = jax.value_and_grad(loss, has_aux=True)(eps)
    assert spectrum.shape == (1, 1, 21)
    assert np.isfinite(spectrum).all()
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    norm = float(jnp.linalg.norm(gradient))
    print(f"narrow-band gradient norm={norm:.9g}")
    assert norm > 0


def test_lumped_default_stays_without_sparams():
    result = _forward(_sim(lumped=True))
    assert result.s_params is None
    assert result.wire_port_sparams is None


def _known_load_line(load_ratio, *, profile):
    """One-cell TEM gap: Zc = eta0, port at x=1 mm, load at x=3 mm.

    See scripts/diagnostics/lumped_port_known_load_line.py. PEC plates on z
    and PMC walls on x/y leave |Gamma| = |(R - Zc)/(R + Zc)| at low frequency.
    """
    from rfx.boundaries.spec import Boundary, BoundarySpec

    dx = 1e-3
    zc = 376.730313668
    profiles = {}
    if profile is not None:
        widths = [1.0, 0.8, 1.2, 1.0] if profile == "graded" else [1.0] * 4
        profiles = dict(dx_profile=np.array(widths) * dx,
                        dz_profile=np.array([dx]))
    sim = Simulation(
        freq_max=10e9, domain=(4 * dx, dx, dx), dx=dx,
        boundary=BoundarySpec(x=Boundary(lo="pmc", hi="pmc"),
                              y=Boundary(lo="pmc", hi="pmc"),
                              z=Boundary(lo="pec", hi="pec")), **profiles,
    )
    sim.add_port(position=(dx, 0., 0.), component="ez", impedance=zc,
                 waveform=GaussianPulse(f0=5e9, bandwidth=1.6))
    sim.add_lumped_rlc(position=(3 * dx, 0., 0.), component="ez",
                       R=load_ratio * zc, topology="parallel")
    return sim


@pytest.mark.parametrize("load_ratio,expected", [(2.0, 1 / 3), (1.0, 0.0)])
def test_lumped_known_load_graded(load_ratio, expected):
    result = _known_load_line(load_ratio, profile="graded").forward(
        port_s11_freqs=[1e9], num_periods=20, skip_preflight=True)
    magnitude = float(np.abs(result.s_params[0, 0, 0]))
    print(f"graded R/Zc={load_ratio:g}: |S11|(1 GHz)={magnitude:.9g}")
    assert abs(magnitude - expected) < 0.01


def test_lumped_uniform_profile_lane_parity():
    request = np.array([1.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float32) * 1e9
    uniform = _known_load_line(2.0, profile=None).forward(
        port_s11_freqs=request, num_periods=20, skip_preflight=True)
    graded = _known_load_line(2.0, profile="flat").forward(
        port_s11_freqs=request, num_periods=20, skip_preflight=True)
    np.testing.assert_array_equal(graded.freqs, request)
    np.testing.assert_array_equal(uniform.freqs, request)
    # Uniform squeezes a single lumped diagonal to (nf,); graded is a matrix.
    uniform_s11 = np.asarray(uniform.s_params).reshape(1, -1)[0]
    delta = np.max(np.abs(np.asarray(graded.s_params[0, 0]) - uniform_s11))
    print(f"lumped lane parity max |dS|={delta:.9g}")
    np.testing.assert_allclose(graded.s_params[0, 0], uniform_s11,
                               rtol=0, atol=1e-5)


def test_lumped_before_passive_wire_preserves_wire_index():
    sim = _sim(lumped=True)
    sim.add_port(position=(5e-3, 4e-3, 3e-3), component="ez", impedance=75,
                 extent=2e-3, excite=False)
    default = _forward(sim)
    selected = _forward(sim, port_s11_freqs=np.asarray(default.freqs))
    assert default.s_params.shape == (1, 1, 50)
    assert selected.s_params.shape == (2, 2, 50)
    # These ordered specs/accumulators define the matrix's port indices.
    # The default has only a passive-port diagnostic; explicit bins include
    # the driven lumped column, so their S values need not be equal.
    default_spec, default_acc = default.wire_port_sparams[0]
    selected_spec, selected_acc = selected.wire_port_sparams[0]
    assert selected_spec == default_spec
    assert selected_spec[4] == 75
    assert not selected_spec[7]
    assert selected.wire_port_sparams[1][0][7]
    for actual, expected in zip(selected_acc, default_acc):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
