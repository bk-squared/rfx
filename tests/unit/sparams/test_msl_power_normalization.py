"""Public MSL power-wave contract on manufactured two-port DFT records.

Unequal positive analytic references must retain a planted reciprocal passive
power S. The generic wave solve, spatial/temporal alignment and transverse
current integral run normally. No FDTD or physical accuracy claim is involved.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from rfx import Box, Simulation
import rfx.api._sparams as sparams
from rfx.probes.probes import DFTPlaneProbe
from rfx.validation import load_port_vi_dump_npz, replay_smatrix_from_port_vi_dump

U = 2.0**-12
FREQS = np.array([1e9, 2e9, 3e9])
RHO = np.array([0.96, 0.92, 0.88])
PHASE = np.array([-0.1, 0.2, -0.3])
THETA = 1.2
INCIDENT = np.array([[1 + 0.05j, 0.18 - 0.08j], [-0.12 + 0.1j, 0.9 - 0.03j]])


def _oracle(theta):
    """Symmetric power S with singular values RHO < 1, independently in NumPy."""
    c, s = np.cos(theta), 1j * np.sin(theta)
    return np.array([[c, s], [s, c]])[:, :, None] * (RHO * np.exp(1j * PHASE))


def _references(widths):
    # Independent u>1 thin-strip formula for h=2U, eps_r=2. No production
    # reference or normalization helper is used to create the field oracle.
    u = np.asarray(widths) / (2 * U)
    eps_eff = 1.5 + 0.5 / np.sqrt(1 + 12 / u)
    return 120 * np.pi / (np.sqrt(eps_eff) * (u + 1.393 + 0.667 * np.log(u + 1.444)))


def _case(monkeypatch, *, equal=False, incident=INCIDENT):
    widths = (4 * U, (4 if equal else 8) * U)
    references = _references(widths)
    sim = Simulation(freq_max=20e9, domain=(40 * U, 16 * U, 8 * U), dx=U,
                     cpml_layers=2, boundary="cpml")
    sim.add_material("substrate", eps_r=2.0)
    sim.add(Box((0, 0, 0), (40 * U, 16 * U, 2 * U)), material="substrate")
    sim.add(Box((0, 0, 0), (40 * U, 16 * U, 0)), material="pec")
    for p, width in enumerate(widths):
        sim.add(Box((p * 20 * U, 8 * U - width / 2, 2 * U),
                    ((p + 1) * 20 * U, 8 * U + width / 2, 2 * U)), material="pec")
        sim.add_msl_port(position=((2 if p == 0 else 38) * U, 8 * U, 0),
                         width=width, height=2 * U, direction="+x" if p == 0 else "-x",
                         impedance=50.0, eps_r_sub=2.0, mode="uniform",
                         n_probe_offset=10, n_probe_spacing=2, n_probes=3)
    grid = sim._build_realized_grid()
    calls = []
    names = re.compile(r"_msl_run(?P<drive>\d+)_p(?P<port>\d+)_(?:ez\d+|hy|hz)(?:_left)?")

    def record(theta):
        dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
        c, s = jnp.cos(theta), 1j * jnp.sin(theta)
        power_s = jnp.stack((jnp.stack((c, s)), jnp.stack((s, c))))[:, :, None]
        power_s = power_s * jnp.asarray(RHO * np.exp(1j * PHASE), dtype=dtype)
        a = jnp.broadcast_to(jnp.asarray(incident, dtype=dtype)[:, :, None], (2, 2, 3))
        b = jnp.einsum("prf,rdf->pdf", power_s, a)
        root_r = jnp.asarray(np.sqrt(references), dtype=dtype)[:, None, None]
        voltage, current = root_r * (a + b), (a - b) / root_r
        planes = {}
        for entry in sim._dft_planes:
            match = names.fullmatch(entry.name)
            assert match is not None and entry.axis == "x"
            drive, port = int(match["drive"]), int(match["port"])
            index = grid.position_to_index((entry.coordinate, 0, 0))[0]
            region = sim._dft_plane_regions[entry.name]
            w0, w1, k0, k1 = region
            shape = (w1 - w0, k1 - k0)
            if entry.component == "ez":
                field = np.ones(shape) / (2 * U)
                amplitude = voltage[port, drive]
            else:
                x_h = (index - grid.pad_x_lo + 0.5) * U
                target = (12 if port == 0 else 28) * U
                factor = 1 + (0.5 + 0.25j) * (x_h - target) / U
                if entry.component == "hy":
                    z_h = np.arange(k0, k1) - grid.pad_z_lo + 0.5
                    profile = np.broadcast_to(z_h[None, :], shape)
                else:
                    assert entry.component == "hz"
                    y_h = np.arange(w0, w1) - grid.pad_y_lo + 0.5
                    profile = np.broadcast_to(0.25 * y_h[:, None], shape)
                # The closed contour has 5 or 9 width-node faces. Its
                # Hy/Hz contributions are +N*U and -0.25*N*U after orientation.
                n_width = (5, 5 if equal else 9)[port]
                sign = 1 if port == 0 else -1
                field = sign * factor * profile / (0.75 * n_width * U)
                phase = jnp.asarray(np.exp(-1j * np.pi * FREQS * grid.dt), dtype=dtype)
                amplitude = current[port, drive] * phase
            planes[entry.name] = DFTPlaneProbe(
                accumulator=jnp.asarray(field, dtype=dtype)[None, :, :] * amplitude[:, None, None],
                freqs=entry.freqs, component=entry.component, axis=0, index=index,
                total_steps=1, window="rect", window_alpha=0.25, region=region,
            )
        return SimpleNamespace(dft_planes=planes)

    def fake_run(**kwargs):
        calls.append("run")
        return record(jnp.asarray(THETA))

    def fake_forward(*, eps_override, **kwargs):
        calls.append("forward")
        assert eps_override.shape == grid.shape
        return record(eps_override[tuple(pad + 1 for pad in grid.axis_pads)])

    monkeypatch.setattr(sim, "run", fake_run)
    monkeypatch.setattr(sim, "forward", fake_forward)
    return sim, grid, references, calls


def _compute(sim, **kwargs):
    return sim.compute_msl_s_matrix(freqs=FREQS, n_steps=1, num_periods=1, **kwargs)


@pytest.mark.parametrize("equal", [False, True], ids=["unequal_references", "equal_references"])
@pytest.mark.parametrize("route", ["run", "forward"])
def test_passive_reciprocal_power_s_survives_raw_and_default_projection(
    monkeypatch, tmp_path, equal, route,
):
    sim, grid, references, calls = _case(monkeypatch, equal=equal)
    expected = _oracle(THETA)
    # This plant is demonstrably passive in the standard power inner product.
    for k, rho in enumerate(RHO):
        np.testing.assert_allclose(expected[:, :, k].conj().T @ expected[:, :, k],
                                   rho**2 * np.eye(2), atol=1e-14)
    if not equal:
        # The retired voltage-wave S breaks reciprocity and looks active
        # for these same passive waves; a later SVD clip cannot repair it.
        voltage_s = np.sqrt(references)[:, None, None] * expected / np.sqrt(references)[None, :, None]
        assert np.max(abs(voltage_s[0, 1] - voltage_s[1, 0])) > 0.1
        assert max(np.linalg.eigvalsh(voltage_s[:, :, k].conj().T @ voltage_s[:, :, k]).max()
                   for k in range(3)) > 1.05
    waves = []
    real_solve = sparams.msl_solve_s_from_waves

    def capture(a, b):
        waves.append((jnp.asarray(a), jnp.asarray(b)))
        return real_solve(a, b)

    monkeypatch.setattr(sparams, "msl_solve_s_from_waves", capture)
    route_kwargs = {"eps_override": jnp.full(grid.shape, THETA)} if route == "forward" else {}
    dump = tmp_path / "raw.npz"
    raw = _compute(sim, enforce_passivity=False, raw_3probe_dump_path=str(dump), **route_kwargs)
    default = _compute(sim, **route_kwargs)
    assert calls == [route] * 4  # Both ports driven on each complete extraction.
    for result in (raw, default):
        assert result.assembly == "multi_drive_solve" and np.max(result.cond_a) < 3
        np.testing.assert_allclose(result.reference_impedances, references, rtol=1e-12)
        np.testing.assert_allclose(result.S, expected, rtol=5e-6, atol=1e-6)
        np.testing.assert_allclose(result.S[0, 1], result.S[1, 0], rtol=5e-6, atol=1e-6)
        assert result.S_raw is None and result.passivity_correction is None
    np.testing.assert_array_equal(default.S, raw.S)
    replay = replay_smatrix_from_port_vi_dump(load_port_vi_dump_npz(dump))
    np.testing.assert_allclose(replay.s_params, expected, rtol=5e-6, atol=1e-6)
    if equal:
        # Equal references retain the existing voltage-wave scale exactly;
        # an unnecessary absolute rescaling could move conditioning/roundoff.
        with np.load(dump) as saved:
            v, i = jnp.asarray(saved["raw_v"][:, :, 0]), jnp.asarray(saved["raw_i1"])
        np.testing.assert_array_equal(waves[0][0], 0.5 * (v + float(references[0]) * i))
        np.testing.assert_array_equal(waves[0][1], 0.5 * (v - float(references[0]) * i))


def test_reference_metadata_is_neither_fitted_z0_nor_the_50_ohm_load(monkeypatch, tmp_path):
    import rfx.probes.msl_wave_decomp as decomp

    sim, _, references, _ = _case(monkeypatch)

    def diagnostic_fit(v, x, i1, beta0, *, z0_hj=None):
        return dict(z0=jnp.full(v.shape[0], 777 + 13j), beta=jnp.full(v.shape[0], 123),
                    q=jnp.ones(v.shape[0]), beta_railed=jnp.zeros(v.shape[0], dtype=bool))

    monkeypatch.setattr(decomp, "extract_msl_nprobe", diagnostic_fit)
    dump = tmp_path / "metadata.npz"
    result = _compute(sim, enforce_passivity=False, raw_3probe_dump_path=str(dump))
    assert np.shape(result.reference_impedances) == (2,)
    assert np.all(np.abs(references - 50) > 1)
    assert np.all(np.abs(result.Z0) > 700)
    np.testing.assert_allclose(result.S, _oracle(THETA), rtol=5e-6, atol=1e-6)
    with np.load(dump) as saved:
        meta = json.loads(str(saved["metadata_json"]))
        assert meta["schema_version"] == 4 and meta["s_wave_convention"] == "power"
        assert meta["current_convention"] == "native_msl_loop_current"
        np.testing.assert_allclose(meta["s_reference_impedances_ohm"], references, rtol=1e-12)
        assert [p["impedance_ohm"] for p in meta["port_definitions"]] == [50, 50]
        np.testing.assert_array_equal(saved["production_smatrix"], result.S)
        a = np.broadcast_to(INCIDENT[:, :, None], (2, 2, 3))
        b = np.stack([_oracle(THETA)[:, :, k] @ INCIDENT for k in range(3)], axis=2)
        voltage = np.sqrt(references)[:, None, None] * (a + b)
        current = (a - b) / np.sqrt(references)[:, None, None]
        np.testing.assert_allclose(saved["raw_v"][:, :, 0], voltage.transpose(1, 0, 2),
                                   rtol=5e-6, atol=1e-6)
        np.testing.assert_allclose(saved["raw_i1"], current.transpose(1, 0, 2),
                                   rtol=5e-6, atol=1e-7)


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_forward_power_objective_preserves_nonzero_material_tape_gradient(monkeypatch, x64):
    with enable_x64(x64):
        sim, grid, references, calls = _case(monkeypatch)

        def objective(theta):
            result = _compute(sim, eps_override=jnp.full(grid.shape, theta))
            np.testing.assert_allclose(result.reference_impedances, references, rtol=1e-12)
            return jnp.mean(jnp.abs(result.S[1, 0])**2)

        value, derivative = jax.value_and_grad(objective)(jnp.asarray(THETA))
        assert calls == ["forward", "forward"]
        expected_loss = float(np.mean(RHO**2) * np.sin(THETA)**2)
        expected_grad = float(np.mean(RHO**2) * np.sin(2 * THETA))
        assert abs(expected_grad) > 0.1
        np.testing.assert_allclose(value, expected_loss, rtol=5e-6, atol=1e-6)
        np.testing.assert_allclose(derivative, expected_grad, rtol=1e-5, atol=1e-6)


def test_fallback_keeps_power_units_after_a_solver_failure(monkeypatch, tmp_path):
    """Inject a nonfinite solve result only to reach the existing fallback branch."""
    sim, _, _, _ = _case(monkeypatch)
    real_solve = sparams.msl_solve_s_from_waves
    solved = []

    def failed_solve(a, b):
        s, cond = real_solve(a, b)
        solved.append(np.asarray(s))
        return jnp.full_like(s, jnp.nan), cond

    monkeypatch.setattr(sparams, "msl_solve_s_from_waves", failed_solve)
    dump = tmp_path / "fallback.npz"
    result = _compute(sim, enforce_passivity=False, raw_3probe_dump_path=str(dump))
    assert result.assembly == "single_ratio_fallback"
    np.testing.assert_allclose(solved[0], _oracle(THETA), rtol=5e-6, atol=1e-6)
    b = np.stack([_oracle(THETA)[:, :, k] @ INCIDENT for k in range(3)], axis=2)
    expected_ratio = b / np.diag(INCIDENT)[None, :, None]
    np.testing.assert_allclose(result.S, expected_ratio, rtol=5e-6, atol=1e-6)
    replay = replay_smatrix_from_port_vi_dump(load_port_vi_dump_npz(dump))
    np.testing.assert_allclose(replay.s_params, result.S, rtol=5e-6, atol=1e-6)
