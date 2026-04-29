"""Phase XV native Strategy B one-port S-parameter observable tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation


def _make_one_port_sim(*, boundary: str = "pec") -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.009, 0.009, 0.009),
        dx=0.001,
        boundary=boundary,
        cpml_layers=0 if boundary == "pec" else 2,
    )
    sim.add_port(
        (0.004, 0.004, 0.004),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8),
    )
    sim.add_probe((0.006, 0.004, 0.004), "ez")
    return sim


def _single_cell_eps(grid, base_eps: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    i, j, k = grid.position_to_index((0.006, 0.006, 0.006))
    return base_eps.at[i, j, k].add(alpha)


def test_phase15_one_port_strategy_b_returns_native_s11_without_standard_extractors(monkeypatch):
    sim = _make_one_port_sim()
    freqs = jnp.array([2.0e9, 3.0e9, 4.0e9], dtype=jnp.float32)

    def _fail_standard_extractor(*_args, **_kwargs):
        raise AssertionError("standard S-parameter extractor must not be used for native Strategy B S11")

    import rfx.probes.probes as probes_mod

    monkeypatch.setattr(probes_mod, "extract_s_matrix", _fail_standard_extractor)
    monkeypatch.setattr(Simulation, "run", _fail_standard_extractor)

    inputs = sim.build_hybrid_phase1_inputs(n_steps=24, s_param_freqs=freqs)
    assert inputs.s_param_request is not None
    report = sim.inspect_hybrid_strategy_b_phase6_from_inputs(inputs, checkpoint_every=8)
    assert report.supported, report.reason_text

    result = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=8,
    )

    assert result.s_params is not None
    assert result.freqs is not None
    assert result.s_params.shape == (1, 1, len(freqs))
    assert result.freqs.shape == (len(freqs),)
    np.testing.assert_allclose(np.asarray(result.freqs), np.asarray(freqs), rtol=0.0, atol=0.0)
    assert np.iscomplexobj(np.asarray(result.s_params))
    assert np.isfinite(np.asarray(result.s_params)).all()


def test_phase15_cpml_one_port_strategy_b_returns_native_s11():
    sim = _make_one_port_sim(boundary="cpml")
    freqs = jnp.array([2.0e9, 3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=18, s_param_freqs=freqs)

    result = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=6,
    )

    assert result.s_params is not None
    assert result.s_params.shape == (1, 1, len(freqs))
    np.testing.assert_allclose(np.asarray(result.freqs), np.asarray(freqs), rtol=0.0, atol=0.0)
    assert np.isfinite(np.asarray(result.s_params)).all()


def test_phase15_no_sparam_request_preserves_legacy_null_contract():
    sim = _make_one_port_sim()
    inputs = sim.build_hybrid_phase1_inputs(n_steps=16)

    result = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=8,
    )

    assert result.s_params is None
    assert result.freqs is None


def test_phase15_native_s11_matches_standard_one_port_reference():
    sim = _make_one_port_sim()
    freqs = jnp.array([2.0e9, 3.0e9, 4.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=32, s_param_freqs=freqs)

    native = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=8,
    )
    standard = sim.run(n_steps=32, compute_s_params=True, s_param_freqs=freqs)

    # Strategy B uses the fast uniform replay kernel, while the standard
    # extractor uses the Python-loop Yee path.  The native S11 should stay
    # tightly correlated without requiring bitwise-identical internals.
    np.testing.assert_allclose(
        np.asarray(native.s_params),
        np.asarray(standard.s_params),
        rtol=2e-3,
        atol=2e-4,
    )


def test_phase15_direct_forward_api_threads_explicit_sparam_freqs():
    sim = _make_one_port_sim()
    freqs = jnp.array([2.5e9, 3.5e9], dtype=jnp.float32)

    result = sim.forward_hybrid_phase1(
        n_steps=16,
        fallback="raise",
        strategy="b",
        checkpoint_every=8,
        s_param_freqs=freqs,
    )

    assert result.s_params.shape == (1, 1, 2)
    np.testing.assert_allclose(np.asarray(result.freqs), np.asarray(freqs), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "bad_freqs, message",
    [
        ([], "non-empty"),
        ([[1.0e9]], "1-D"),
        ([1.0e9, float("nan")], "finite"),
        ([0.0, 1.0e9], "positive"),
    ],
)
def test_phase15_sparam_freqs_validate_fail_closed(bad_freqs, message):
    sim = _make_one_port_sim()

    with pytest.raises(ValueError, match=message):
        sim.build_hybrid_phase1_inputs(n_steps=8, s_param_freqs=bad_freqs)


def test_phase15_passive_two_port_request_remains_unsupported():
    sim = _make_one_port_sim()
    sim.add_port((0.007, 0.004, 0.004), "ez", impedance=50.0, excite=False)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=16, s_param_freqs=jnp.array([3.0e9]))

    report = sim.inspect_hybrid_strategy_b_phase6_from_inputs(inputs, checkpoint_every=8)

    assert not report.supported
    assert "Phase XV native Strategy B S-parameters do not support passive/two-port workflows" in report.reason_text
    with pytest.raises(ValueError, match="passive/two-port"):
        sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b", checkpoint_every=8)


@pytest.mark.parametrize(
    "configure, expected",
    [
        (
            lambda sim: sim.add_port(
                (0.002, 0.004, 0.004),
                "ez",
                impedance=50.0,
                extent=0.002,
                waveform=GaussianPulse(f0=3e9, bandwidth=0.8),
            ),
            "wire ports",
        ),
        (
            lambda sim: (sim.add_source((0.002, 0.004, 0.004), "ez"), sim.add_port(
                (0.004, 0.004, 0.004),
                "ez",
                impedance=50.0,
                waveform=GaussianPulse(f0=3e9, bandwidth=0.8),
            )),
            "mixed soft-source workflows",
        ),
    ],
)
def test_phase15_wire_and_mixed_soft_source_requests_remain_unsupported(configure, expected):
    sim = Simulation(freq_max=5e9, domain=(0.009, 0.009, 0.009), dx=0.001, boundary="pec", cpml_layers=0)
    configure(sim)
    sim.add_probe((0.006, 0.004, 0.004), "ez")
    inputs = sim.build_hybrid_phase1_inputs(n_steps=16, s_param_freqs=jnp.array([3.0e9]))

    report = sim.inspect_hybrid_strategy_b_phase6_from_inputs(inputs, checkpoint_every=8)

    assert not report.supported
    assert expected in report.reason_text


def test_phase15_sparams_are_stop_gradient_sidecar_and_do_not_perturb_time_series_gradient():
    sim = _make_one_port_sim()
    freqs = jnp.array([3.0e9], dtype=jnp.float32)
    no_sp_inputs = sim.build_hybrid_phase1_inputs(n_steps=12)
    sp_inputs = sim.build_hybrid_phase1_inputs(n_steps=12, s_param_freqs=freqs)
    grid = sp_inputs.grid
    assert grid is not None
    assert sp_inputs.materials is not None

    def time_loss(inputs, alpha):
        eps = _single_cell_eps(grid, inputs.materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1_from_inputs(
            inputs,
            eps_override=eps,
            strategy="b",
            checkpoint_every=6,
        )
        return jnp.sum(result.time_series**2)

    alpha0 = jnp.float32(0.05)
    grad_without = jax.grad(lambda a: time_loss(no_sp_inputs, a))(alpha0)
    grad_with = jax.grad(lambda a: time_loss(sp_inputs, a))(alpha0)
    np.testing.assert_allclose(np.asarray(grad_with), np.asarray(grad_without), rtol=1e-5, atol=1e-8)

    def sidecar_loss(alpha):
        eps = _single_cell_eps(grid, sp_inputs.materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1_from_inputs(
            sp_inputs,
            eps_override=eps,
            strategy="b",
            checkpoint_every=6,
        )
        assert result.s_params is not None
        return jnp.real(jnp.sum(jnp.abs(result.s_params) ** 2))

    sidecar_grad = jax.grad(sidecar_loss)(alpha0)
    np.testing.assert_allclose(np.asarray(sidecar_grad), np.asarray(0.0, dtype=np.float32), atol=0.0, rtol=0.0)
