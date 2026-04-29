"""Fast contract tests for the mode-orthogonal waveguide-port extractor.

These tests encode the public/internal hooks specified in
``docs/research_notes/2026-04-26_codex_implementation_spec.md`` without
running a full FDTD simulation.  They intentionally exercise the modal
profile builders and extractor algebra under the same ``aperture_dA`` that
``modal_voltage``, ``modal_current``, and ``overlap_modal_amplitude`` use.
"""

from __future__ import annotations

from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.sources.waveguide_port import (
    WaveguidePort,
    _aperture_dA,
    _discrete_te_mode_profiles,
    init_waveguide_port,
    modal_current,
    modal_voltage,
    overlap_modal_amplitude,
)
from rfx.api import Simulation


State = namedtuple("State", "ex ey ez hx hy hz")


def _dropped_plus_face_aperture(u_widths: np.ndarray, v_widths: np.ndarray) -> np.ndarray:
    """Face-aware dA for the P0 contract: zero +u and +v PEC cells."""
    u_weights = np.ones_like(u_widths, dtype=np.float64)
    v_weights = np.ones_like(v_widths, dtype=np.float64)
    u_weights[-1] = 0.0
    v_weights[-1] = 0.0
    return (u_widths * u_weights)[:, None] * (v_widths * v_weights)[None, :]


def _call_discrete_te_with_aperture(
    a: float,
    b: float,
    mode: tuple[int, int],
    u_widths: np.ndarray,
    v_widths: np.ndarray,
    aperture_dA: np.ndarray,
):
    """Call the intended P0 hook and fail with a hook-specific message if absent."""
    try:
        return _discrete_te_mode_profiles(
            a,
            b,
            mode[0],
            mode[1],
            u_widths,
            v_widths,
            aperture_dA=aperture_dA,
        )
    except TypeError as exc:
        pytest.fail(
            "_discrete_te_mode_profiles must accept aperture_dA= and solve/normalise "
            "with that same mass measure for mode-orthogonal extraction"
            f" (got TypeError: {exc})"
        )


def _mode_profiles(
    a: float,
    b: float,
    mode: tuple[int, int],
    u_widths: np.ndarray,
    v_widths: np.ndarray,
    aperture_dA: np.ndarray,
):
    ey, ez, hy, hz, kc = _call_discrete_te_with_aperture(
        a, b, mode, u_widths, v_widths, aperture_dA
    )
    return {
        "ey": np.asarray(ey, dtype=np.float64),
        "ez": np.asarray(ez, dtype=np.float64),
        "hy": np.asarray(hy, dtype=np.float64),
        "hz": np.asarray(hz, dtype=np.float64),
        "kc": float(kc),
    }


def _e_overlap(lhs: dict[str, np.ndarray], rhs: dict[str, np.ndarray], dA: np.ndarray) -> float:
    return float(np.sum((lhs["ey"] * rhs["ey"] + lhs["ez"] * rhs["ez"]) * dA))


def _lorentz_self_overlap(profile: dict[str, np.ndarray], dA: np.ndarray) -> float:
    return float(np.sum((profile["ey"] * profile["hz"] - profile["ez"] * profile["hy"]) * dA))


def _base_cfg_for_profile(profile: dict[str, np.ndarray], aperture_dA: np.ndarray):
    """Build a lightweight x-normal config, then replace profiles with test profiles."""
    nu, nv = aperture_dA.shape
    dx = 1.0e-3
    port = WaveguidePort(
        x_index=2,
        y_slice=(0, nu),
        z_slice=(0, nv),
        a=nu * dx,
        b=nv * dx,
        mode=(1, 0),
        mode_type="TE",
        direction="+x",
    )
    cfg = init_waveguide_port(port, dx=dx, freqs=jnp.array([12.0e9]), mode_profile="analytic")
    return cfg._replace(
        ey_profile=jnp.asarray(profile["ey"], dtype=jnp.float32),
        ez_profile=jnp.asarray(profile["ez"], dtype=jnp.float32),
        hy_profile=jnp.asarray(profile["hy"], dtype=jnp.float32),
        hz_profile=jnp.asarray(profile["hz"], dtype=jnp.float32),
        aperture_dA=jnp.asarray(aperture_dA, dtype=jnp.float32),
        h_offset=(0.0, 0.0),
    )


def _state_with_xnormal_fields(
    e_u: np.ndarray,
    e_v: np.ndarray,
    h_u: np.ndarray,
    h_v: np.ndarray,
    *,
    x_index: int = 2,
) -> State:
    """Create arrays where H is identical on x_index and x_index-1 after averaging."""
    nu, nv = e_u.shape
    shape = (x_index + 2, nu, nv)
    zeros = jnp.zeros(shape, dtype=jnp.float32)
    ey = zeros.at[x_index, :, :].set(jnp.asarray(e_u, dtype=jnp.float32))
    ez = zeros.at[x_index, :, :].set(jnp.asarray(e_v, dtype=jnp.float32))
    hy = zeros.at[x_index, :, :].set(jnp.asarray(h_u, dtype=jnp.float32))
    hy = hy.at[x_index - 1, :, :].set(jnp.asarray(h_u, dtype=jnp.float32))
    hz = zeros.at[x_index, :, :].set(jnp.asarray(h_v, dtype=jnp.float32))
    hz = hz.at[x_index - 1, :, :].set(jnp.asarray(h_v, dtype=jnp.float32))
    return State(ex=zeros, ey=ey, ez=ez, hx=zeros, hy=hy, hz=hz)


def test_multimode_te10_te20_te30_cross_overlap_uses_aperture_dA():
    """TE10/TE20/TE30 profiles are orthogonal under the extractor aperture dA."""
    a, b = 24.0e-3, 8.0e-3
    u_widths = np.full(24, a / 24, dtype=np.float64)
    v_widths = np.full(8, b / 8, dtype=np.float64)
    aperture_dA = _dropped_plus_face_aperture(u_widths, v_widths)

    modes = [
        _mode_profiles(a, b, mode, u_widths, v_widths, aperture_dA)
        for mode in [(1, 0), (2, 0), (3, 0)]
    ]
    overlap = np.array([[_e_overlap(lhs, rhs, aperture_dA) for rhs in modes] for lhs in modes])

    assert np.all(np.diag(overlap) > 0.5)
    offdiag = overlap - np.diag(np.diag(overlap))
    assert np.max(np.abs(offdiag)) < 1.0e-6


def test_synthetic_two_mode_field_separates_under_aperture_dA():
    """Lorentz projection recovers each coefficient from a synthetic two-mode field."""
    a, b = 24.0e-3, 8.0e-3
    u_widths = np.full(24, a / 24, dtype=np.float64)
    v_widths = np.full(8, b / 8, dtype=np.float64)
    aperture_dA = _dropped_plus_face_aperture(u_widths, v_widths)
    te10 = _mode_profiles(a, b, (1, 0), u_widths, v_widths, aperture_dA)
    te20 = _mode_profiles(a, b, (2, 0), u_widths, v_widths, aperture_dA)

    coeff_10 = 0.75
    coeff_20 = -0.30
    e_y = coeff_10 * te10["ey"] + coeff_20 * te20["ey"]
    e_z = coeff_10 * te10["ez"] + coeff_20 * te20["ez"]
    h_y = coeff_10 * te10["hy"] + coeff_20 * te20["hy"]
    h_z = coeff_10 * te10["hz"] + coeff_20 * te20["hz"]

    cfg10 = _base_cfg_for_profile(te10, aperture_dA)
    cfg20 = _base_cfg_for_profile(te20, aperture_dA)
    state = _state_with_xnormal_fields(e_y, e_z, h_y, h_z)

    a10, b10 = overlap_modal_amplitude(state, cfg10, cfg10.x_index, cfg10.dx)
    a20, b20 = overlap_modal_amplitude(state, cfg20, cfg20.x_index, cfg20.dx)

    np.testing.assert_allclose(float(a10), coeff_10, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(float(a20), coeff_20, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(float(b10), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(float(b20), 0.0, atol=1.0e-6)


def test_lorentz_and_vi_paths_agree_on_controlled_pure_mode_field():
    """For a pure forward mode, Lorentz amplitude equals modal V and I to 1e-5."""
    a, b = 24.0e-3, 8.0e-3
    u_widths = np.full(24, a / 24, dtype=np.float64)
    v_widths = np.full(8, b / 8, dtype=np.float64)
    aperture_dA = _dropped_plus_face_aperture(u_widths, v_widths)
    te10 = _mode_profiles(a, b, (1, 0), u_widths, v_widths, aperture_dA)
    cfg = _base_cfg_for_profile(te10, aperture_dA)

    amplitude = 0.625
    state = _state_with_xnormal_fields(
        amplitude * te10["ey"],
        amplitude * te10["ez"],
        amplitude * te10["hy"],
        amplitude * te10["hz"],
    )

    a_fwd, b_bwd = overlap_modal_amplitude(state, cfg, cfg.x_index, cfg.dx)
    voltage = modal_voltage(state, cfg, cfg.x_index, cfg.dx)
    current = modal_current(state, cfg, cfg.x_index, cfg.dx)
    c_mode = _lorentz_self_overlap(te10, aperture_dA)

    np.testing.assert_allclose(float(c_mode), 1.0, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(float(a_fwd), amplitude, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(float(b_bwd), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(float(voltage), amplitude, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(float(current), amplitude, rtol=1.0e-5, atol=1.0e-5)


def test_zero_weight_aperture_cells_are_dropped_and_backfilled_as_zero_profiles():
    """Zero-dA cells are dropped from stored modal profiles."""
    a, b = 24.0e-3, 8.0e-3
    u_widths = np.full(24, a / 24, dtype=np.float64)
    v_widths = np.full(8, b / 8, dtype=np.float64)
    aperture_dA = _dropped_plus_face_aperture(u_widths, v_widths)

    profile = _mode_profiles(a, b, (3, 0), u_widths, v_widths, aperture_dA)

    assert np.isfinite(profile["kc"])
    dropped = aperture_dA == 0.0
    assert np.any(dropped)
    for component in ("ey", "ez", "hy", "hz"):
        assert np.all(np.isfinite(profile[component]))
        assert np.all(profile[component][dropped] == 0.0)
    np.testing.assert_allclose(
        np.asarray(_aperture_dA(_base_cfg_for_profile(profile, aperture_dA))),
        aperture_dA,
        rtol=1.0e-6,
        atol=1.0e-12,
    )


def test_multimode_normalize_true_api_dispatch_smoke():
    """The public API no longer rejects normalize=True for multi-mode ports."""
    sim = Simulation(freq_max=10e9, domain=(0.06, 0.04, 0.02), dx=0.002, boundary="cpml")
    common = dict(
        y_range=(0.0, 0.04),
        z_range=(0.0, 0.02),
        freqs=np.array([6.0e9]),
        f0=6.0e9,
        bandwidth=0.4,
        n_modes=2,
    )
    sim.add_waveguide_port(0.004, direction="+x", name="left", **common)
    sim.add_waveguide_port(0.052, direction="-x", name="right", **common)

    result = sim.compute_waveguide_s_matrix(n_steps=4, normalize=True)

    assert result.s_params.shape == (4, 4, 1)
    assert result.port_names == (
        "left_mode0_TE10",
        "left_mode1_TE01",
        "right_mode0_TE10",
        "right_mode1_TE01",
    )
    np.testing.assert_allclose(result.reference_planes, np.array([0.004, 0.052]))


def test_biortho_dot_product_solve_orientation_matches_recorder_contract():
    """BIORTHO solves raw=G.T@coeff for separate E/H recorder Grams."""
    from scripts import _phase2_orthonormal_spike as spike

    gram_e = np.array([[2.0, 0.25], [-0.10, 1.5]], dtype=np.float64)
    gram_h = np.array([[1.25, -0.20], [0.15, 0.90]], dtype=np.float64)
    coeff = np.array([[0.75, 0.10], [-0.30, 0.40]], dtype=np.float64)

    raw_e = gram_e.T @ coeff
    raw_h = gram_h.T @ coeff

    np.testing.assert_allclose(
        spike._solve_biorthogonal_coefficients(gram_e, raw_e, label="E"),
        coeff,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        spike._solve_biorthogonal_coefficients(gram_h, raw_h, label="H"),
        coeff,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_biortho_rejects_ill_conditioned_dot_gram():
    """The spike must stop instead of silently inverting singular Grams."""
    from scripts import _phase2_orthonormal_spike as spike

    singular = np.array([[1.0, 1.0], [1.0, 1.0 + 1.0e-14]], dtype=np.float64)
    raw = np.array([1.0, 1.0], dtype=np.float64)

    with pytest.raises(spike.SpikeUnavailable, match="ill-conditioned"):
        spike._solve_biorthogonal_coefficients(
            singular,
            raw,
            cond_threshold=1.0e6,
            label="test Gram",
        )


def test_biortho_single_mode_solve_reduces_to_scalar_normalisation():
    """A one-mode BIORTHO solve is exactly raw/G, so it is no-benefit safe."""
    from scripts import _phase2_orthonormal_spike as spike

    recovered = spike._solve_biorthogonal_coefficients(
        np.array([[2.5]], dtype=np.float64),
        np.array([1.25], dtype=np.float64),
    )
    np.testing.assert_allclose(recovered, np.array([0.5]), rtol=1.0e-12, atol=1.0e-12)


def test_biortho_full_yee_profiles_are_finite_for_both_h_offsets():
    """BIORTHO algebra covers the old centered and current dual-H fixtures."""
    from scripts import _phase2_orthonormal_spike as spike

    candidate = next(c for c in spike.CANDIDATES if c.name == "BIORTHO_FULL_YEE")
    a, b = 22.86e-3, 10.16e-3
    dx = 1.0e-3
    u_widths = np.full(int(round(a / dx)), dx, dtype=np.float64)
    v_widths = np.full(int(round(b / dx)), dx, dtype=np.float64)
    dA = spike._candidate_dA(candidate, u_widths, v_widths)

    for h_offset in [(0.0, 0.0), (0.5, 0.5)]:
        profiles = [
            spike._profile_for_candidate(candidate, a, b, mode, u_widths, v_widths, h_offset=h_offset)
            for mode in [(1, 0), (0, 1), (2, 0)]
        ]
        gram_e = spike._mode_dot_gram_np([(p[0], p[1]) for p in profiles], dA)
        gram_h = spike._mode_dot_gram_np([(p[2], p[3]) for p in profiles], dA)

        assert np.all(np.isfinite(gram_e))
        assert np.all(np.isfinite(gram_h))
        assert np.linalg.cond(gram_e) < 10.0
        assert np.linalg.cond(gram_h) < 10.0
