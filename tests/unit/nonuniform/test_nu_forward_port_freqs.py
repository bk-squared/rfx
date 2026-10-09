"""Requested graded forward bins, numerical subset witness and AD (#1410)."""

import jax
import sys
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
    print(f"subset max |dS|={delta:.9g}", file=sys.stderr)
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
    print(f"narrow-band gradient norm={norm:.9g}", file=sys.stderr)
    assert norm > 0


def test_lumped_default_stays_without_sparams():
    result = _forward(_sim(lumped=True))
    assert result.s_params is None
    assert result.wire_port_sparams is None


def _known_load_line(load_ratio, *, profile):
    """Internal lattice coax, asymmetric transverse cells: Zc=eta0/3.75.

    dx=0.25 mm; PEC sheets y=2/5 and z=2/4, inner filament (3,3).
    Unequal exterior margins; the line continues through longitudinal CPML. Grading changes x cells only and keeps
    this realized cross-section. Both declarations are shared mesh nodes,
    and their realized separation is asserted before the solve.
    """
    from tests._interior_tem_line import build

    widths = None
    if profile is not None:
        widths = np.array([1, .8, 1.2, .9, 1.1, .8, 1.2, 1, 1]
                          if profile == "graded" else [1] * 9) * .25e-3
    d = 1.2e-3 if profile == "graded" else 1.25e-3
    return build(ratio=load_ratio, profile=widths, axial_positions=(.25e-3, .25e-3+d),
                 declared_separation=d)[0]


@pytest.mark.parametrize("load_ratio", [2.0, 1.0])
def test_lumped_known_load_graded(load_ratio, record_property):
    from tests._interior_tem_line import (build, input_reflection, element_inductance,
                                          assert_predicted_residual, assert_solved_ports)

    widths = np.array([1, .8, 1.2, .9, 1.1, .8, 1.2, 1, 1]) * .25e-3
    sim, line = build(ratio=load_ratio, profile=widths,
                      axial_positions=(.25e-3, 1.45e-3), declared_separation=1.2e-3)
    record_property("realized_d_m", line.length)
    result = sim.forward(port_s11_freqs=[1e9], num_periods=20, skip_preflight=True)
    assert_solved_ports(result, line)
    magnitude = float(np.abs(result.s_params[0, 0, 0]))
    expected = float(abs(input_reflection(line, [1e9], load_ratio * line.zc)[0]))
    print(f"graded R/Zc={load_ratio:g}: |S11|={magnitude:.9g}, TEM={expected:.9g}", file=sys.stderr)
    record_property("closed_form_s11_magnitude", expected)
    record_property("new_s11_magnitude", magnitude)
    import json
    measured = np.asarray(result.s_params).reshape(-1)
    record_property("s11", json.dumps(np.stack([measured.real, measured.imag], axis=-1).tolist()))
    assert abs(magnitude - expected) < 0.01
    # Retain the original magnitude bar and add D1(ii), on the same requested
    # 1 GHz bin and snapped Ez planes; local transverse cell size stays dx.
    pure = input_reflection(line, [1e9], load_ratio * line.zc)
    predicted = input_reflection(line, [1e9], load_ratio * line.zc,
                                 element_l=element_inductance(line))
    assert_predicted_residual(np.asarray(result.s_params).reshape(-1), pure, predicted)


def test_lumped_uniform_profile_lane_parity(record_property):
    request = np.array([1.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float32) * 1e9
    from tests._interior_tem_line import build, assert_solved_ports

    # Lane parity, not a continuum-accuracy solve: five axial 1 mm cells,
    # 8x8 transverse cells, 20 CPML cells per end (46x9x9 nodes), 100 steps.
    # Source/load are distinct interior nodes, with the same shunt load and
    # all five requested frequencies; retain the original complex 1e-5 bar.
    uniform_sim, line = build(dx=1e-3, cells=5, ratio=2.,
                              axial_positions=(1e-3, 3e-3), declared_separation=2e-3)
    graded_sim, graded_line = build(dx=1e-3, cells=5, ratio=2., profile=np.full(5, 1e-3),
                                    axial_positions=(1e-3, 3e-3), declared_separation=2e-3)
    uniform = uniform_sim.forward(port_s11_freqs=request, n_steps=100, skip_preflight=True)
    graded = graded_sim.forward(port_s11_freqs=request, n_steps=100, skip_preflight=True)
    assert_solved_ports(uniform, line)
    assert_solved_ports(graded, graded_line)
    np.testing.assert_array_equal(graded.freqs, request)
    np.testing.assert_array_equal(uniform.freqs, request)
    # Uniform squeezes a single lumped diagonal to (nf,); graded is a matrix.
    uniform_s11 = np.asarray(uniform.s_params).reshape(1, -1)[0]
    assert np.isfinite(uniform_s11).all() and np.isfinite(graded.s_params).all()
    record_property("interior_cells", "5x8x8; 20 CPML cells at each x end")
    record_property("n_steps", 100)
    record_property("uniform_abs_s11", np.abs(uniform_s11).tolist())
    record_property("graded_abs_s11", np.abs(graded.s_params[0, 0]).tolist())
    delta = np.max(np.abs(np.asarray(graded.s_params[0, 0]) - uniform_s11))
    print(f"lumped lane parity max |dS|={delta:.9g}", file=sys.stderr)
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


@pytest.mark.slow
@pytest.mark.parametrize("load_ratio", [1., 2.])
def test_graded_known_load_three_mesh_trend(load_ratio, record_property):
    """D1(i) on the graded fixture itself, at its always-on 1 GHz bin.

    Subdivide each axial interval by 1/2/4 and scale the transverse lattice
    with dx. Physical declarations and the total axial domain stay fixed;
    each oracle asserts the same 1.2 mm realized separation. No coefficient is refitted.
    """
    import json
    from tests._interior_tem_line import (
        build, input_reflection, element_inductance, assert_predicted_residual,
        assert_first_order, residuals, assert_solved_ports,
    )

    widths = np.array([1, .8, 1.2, .9, 1.1, .8, 1.2, 1, 1]) * .25e-3
    meshes = np.array([.25e-3, .125e-3, .0625e-3])
    freqs = np.array([1e9])
    errors, phases, curves = [], [], []
    record_property("freqs_hz", json.dumps(freqs.tolist()))
    for dx, refinement in zip(meshes, (1, 2, 4)):
        sim, line = build(dx=dx, cells=9 * refinement, ratio=load_ratio,
                          profile=np.repeat(widths / refinement, refinement),
                          axial_positions=(.25e-3, 1.45e-3), declared_separation=1.2e-3)
        record_property(f"realized_d_{dx}", line.length)
        result = sim.forward(port_s11_freqs=freqs, num_periods=40, skip_preflight=True)
        assert_solved_ports(result, line)
        measured = np.asarray(result.s_params).reshape(-1)
        pure = input_reflection(line, freqs, load_ratio * line.zc)
        predicted = input_reflection(line, freqs, load_ratio * line.zc,
                                     element_l=element_inductance(line))
        curves.append((measured, pure, predicted))
        record_property(f"residuals_{dx}", json.dumps(residuals(measured, pure, predicted)))
        errors.append(abs(measured - pure))
        phases.append(abs(np.angle(measured / pure)))
    record_property("complex_order", json.dumps(assert_first_order(meshes, errors).tolist()))
    record_property("phase_order", json.dumps(assert_first_order(meshes, phases).tolist()))
    for measured, pure, predicted in curves:
        assert_predicted_residual(measured, pure, predicted)
