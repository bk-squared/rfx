"""A line port's declared impedance= is its S reference (#2014).

``renormalize_power_waves`` refers a power-wave S from one real reference per
port to another: S' = C (S - R)(I - R S)^-1 C^-1. Every expected value here is
built WITHOUT the helper: through the impedance matrix (T1, T4), from a
closed-form circuit (T1b), or from the pre-change S of the same run (T2, T3).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
from rfx.sparams._port_reference import renormalize_power_waves
from tests.unit.sparams.test_msl_passivity_enforcement import _thru
from tests._x64_compat import enable_x64


def _route_via_z(S, z_from, z_to):
    """Independent route: S at z_from -> impedance matrix -> S at z_to.

    Z = F (I + S)(I - S)^-1 F, F = diag(sqrt z_from);
    S' = G^-1 (Z - Zt)(Z + Zt)^-1 G, G = diag(sqrt z_to), Zt = diag(z_to).
    ``S`` is (n, n, n_freqs); float64 throughout.
    """
    S = np.asarray(S, dtype=np.complex128)
    n = S.shape[0]
    eye = np.eye(n)
    f = np.diag(np.sqrt(np.asarray(z_from, dtype=np.float64)))
    g = np.diag(np.sqrt(np.asarray(z_to, dtype=np.float64)))
    zt = np.diag(np.asarray(z_to, dtype=np.float64))
    out = np.empty_like(S)
    for k in range(S.shape[2]):
        s = S[:, :, k]
        z = f @ (eye + s) @ np.linalg.inv(eye - s) @ f
        out[:, :, k] = np.linalg.inv(g) @ (z - zt) @ np.linalg.inv(z + zt) @ g
    return out


def _random_passive(n, n_freqs, rng):
    a = rng.normal(size=(n, n, n_freqs)) + 1j * rng.normal(size=(n, n, n_freqs))
    for k in range(n_freqs):
        a[:, :, k] /= 1.1 * np.linalg.norm(a[:, :, k], 2)
    return a


# ---- T1: the helper against the impedance-matrix route -----------------------

def test_t1_quarter_wave_section_matches_impedance_route():
    """A lossless 90 deg section of a 52.91 ohm line at 10 GHz, S at its own
    Zc, referred to 50 ohm. Also checked against the section's ABCD matrix
    terminated in 50 ohm (closed form, no S at Zc involved)."""
    zc, z_to = 52.91, 50.0
    theta = np.pi / 2
    s_line = np.zeros((2, 2, 1), dtype=np.complex128)
    s_line[0, 1, 0] = s_line[1, 0, 0] = np.exp(-1j * theta)
    got = renormalize_power_waves(s_line, [zc, zc], [z_to, z_to])
    want = _route_via_z(s_line, [zc, zc], [z_to, z_to])
    assert np.max(np.abs(got - want)) < 1e-12

    # ABCD of a line section: [[cos, j Zc sin], [j sin / Zc, cos]].
    a, b = np.cos(theta), 1j * zc * np.sin(theta)
    c, d = 1j * np.sin(theta) / zc, np.cos(theta)
    den = a + b / z_to + c * z_to + d
    s11 = (a + b / z_to - c * z_to - d) / den
    s21 = 2.0 / den
    assert abs(got[0, 0, 0] - s11) < 1e-12
    assert abs(got[1, 0, 0] - s21) < 1e-12
    assert abs(got[1, 1, 0] - s11) < 1e-12


def test_t1_random_passive_three_port_matches_impedance_route():
    rng = np.random.default_rng(2014)
    s = _random_passive(3, 7, rng)
    z_from = rng.uniform(20.0, 120.0, size=3)
    z_to = rng.uniform(20.0, 120.0, size=3)
    got = renormalize_power_waves(s, z_from, z_to)
    want = _route_via_z(s, z_from, z_to)
    assert got.shape == s.shape
    assert np.max(np.abs(got - want)) < 1e-12


def test_t1_partial_change_matches_route_and_confines_nan_columns():
    """Ports whose reference does not move (lumped/wire in the mixed lane):
    the result equals the full route, and a NaN column driven at such a port
    stays in that column."""
    rng = np.random.default_rng(7)
    s = _random_passive(3, 4, rng)
    z_from = np.array([50.0, 47.9, 52.9])
    z_to = np.array([50.0, 50.0, 50.0])
    got = renormalize_power_waves(s, z_from, z_to)
    assert np.max(np.abs(got - _route_via_z(s, z_from, z_to))) < 1e-12

    s_nan = s.copy()
    s_nan[:, 0, :] = np.nan
    got_nan = renormalize_power_waves(s_nan, z_from, z_to)
    assert np.all(np.isnan(got_nan[:, 0, :]))
    assert np.max(np.abs(got_nan[:, 1:, :] - got[:, 1:, :])) < 1e-12


# ---- T1b: the issue's circuit (closed form, no FDTD) --------------------------

_C0 = 299792458.0


def _hj(w, h, er):
    u = w / h
    aa = (1 + math.log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) / 49
          + math.log(1 + (u / 18.1) ** 3) / 18.7)
    bb = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    ee = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-aa * bb)
    fu = 6 + (2 * math.pi - 6) * math.exp(-((30.666 / u) ** 0.7528))
    return (376.73 / (2 * math.pi * math.sqrt(ee))) * math.log(
        fu / u + math.sqrt(1 + (2 / u) ** 2)), ee


def _stub_matched_load():
    """Line w 1.15 mm, h 0.5 mm, eps_r 3.0, lossless; 75 ohm load; single
    open-stub match at 10 GHz designed on the line's own Z0 (Pozar 5.2).
    Returns (f, Zc, Zin(f)) at the load-side port plane."""
    w, h, er, rl, f0 = 1.15e-3, 0.5e-3, 3.0, 75.0, 10e9
    zc, ee = _hj(w * 1e3, h * 1e3, er)
    f = np.linspace(8e9, 12e9, 4001)
    beta = 2 * np.pi * f * math.sqrt(ee) / _C0
    b0 = 2 * np.pi * f0 * math.sqrt(ee) / _C0
    t_s = np.linspace(0.001, math.pi - 0.001, 200000) / b0

    def yin(d, b):
        z = (rl + 1j * zc * np.tan(b * d)) / (zc + 1j * rl * np.tan(b * d))
        return 1 / z

    y = yin(t_s, b0)
    i = int(np.argmin(np.where(t_s * b0 > 0.05, abs(y.real - 1), 9)))
    d, b_stub = t_s[i], y[i].imag
    l_stub = (math.atan(-b_stub) % math.pi) / b0
    zl = zc * (rl + 1j * zc * np.tan(beta * d)) / (zc + 1j * rl * np.tan(beta * d))
    ystub = 1j * np.tan(beta * l_stub) / zc
    zin = 1 / (1 / zl + ystub)
    return f, zc, zin


def _dip(f, s11):
    db = 20 * np.log10(np.abs(s11))
    k = int(np.argmin(db))
    return f[k], db[k]


def test_t1b_stub_matched_load_dip_moves_with_the_reference():
    f, zc, zin = _stub_matched_load()
    assert abs(zc - 52.91) < 0.005
    s_line = (zin - zc) / (zin + zc)
    s_50 = (zin - 50.0) / (zin + 50.0)

    f_line, db_line = _dip(f, s_line)
    f_50, db_50 = _dip(f, s_50)
    assert abs(f_line - 10.000e9) <= 1e6 and db_line < -100.0
    assert abs(f_50 - 10.034e9) <= 1e6 and -35.0 < db_50 < -28.0

    # The helper reproduces the 50 ohm curve from the line-Z0 curve.
    got = renormalize_power_waves(s_line[None, None, :], [zc], [50.0])[0, 0]
    assert np.max(np.abs(got - s_50)) < 1e-12


# ---- T2: bit identity where nothing moves -------------------------------------

def test_t2_equal_references_return_the_same_object():
    rng = np.random.default_rng(0)
    s_np = _random_passive(2, 3, rng)
    assert renormalize_power_waves(s_np, [47.9, 50.0], [47.9, 50.0]) is s_np
    s_j = jnp.asarray(s_np.astype(np.complex64))
    assert renormalize_power_waves(s_j, [47.9, 50.0], [47.9, 50.0]) is s_j


def _thru_at(impedance):
    # The geometry of tests/unit/sparams/test_msl_passivity_enforcement.py's
    # _thru(), with the ports' impedance= as the variable.
    sim = Simulation(snap="declared", freq_max=20e9, domain=(0.012, 0.008, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0.0008)), material="sub")
    sim.add(Box((0., 0., 0.), (0.012, 0.008, 0.)), material="pec")
    sim.add(Box((0.002, 0.0034, 0.0008), (0.010, 0.0046, 0.0008)), material="pec")
    sim.add_msl_port(position=(0.002, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=impedance, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=impedance, eps_r_sub=2.2, name="p2")
    return sim


FREQS = jnp.linspace(2e9, 18e9, 16)
Z_HJ = float(hammerstad_jensen_z0_eps_eff(0.0012, 0.0008, 2.2)[0])


def _unpatched(monkeypatch):
    monkeypatch.setattr(
        "rfx.sparams.msl.renormalize_power_waves", lambda s, zf, zt: s)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_t2_msl_thru_at_its_line_z0_is_bit_identical(monkeypatch):
    new = _thru_at(Z_HJ).compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)
    _unpatched(monkeypatch)
    old = _thru_at(Z_HJ).compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)
    assert np.asarray(new.S).tobytes() == np.asarray(old.S).tobytes()
    np.testing.assert_array_equal(new.reference_impedances, [Z_HJ, Z_HJ])
    np.testing.assert_array_equal(new.line_impedances, [Z_HJ, Z_HJ])


# ---- T3: end to end, declared 50 ohm on the lock fixture ----------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_t3_msl_thru_is_referenced_to_the_declared_impedance(monkeypatch):
    res = _thru().compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)
    _unpatched(monkeypatch)
    at_line = _thru().compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)

    np.testing.assert_array_equal(res.reference_impedances, [50.0, 50.0])
    np.testing.assert_array_equal(res.line_impedances, [Z_HJ, Z_HJ])
    want = _route_via_z(at_line.S, [Z_HJ, Z_HJ], [50.0, 50.0])
    got = np.asarray(res.S, dtype=np.complex128)
    assert np.max(np.abs(got - want)) < 1e-6
    # The reference really moved: the two S differ by more than rounding.
    assert np.max(np.abs(got - np.asarray(at_line.S))) > 1e-3


# ---- T4: the product helper and the test route on unequal references ----------

def test_t4_helper_agrees_with_test_route_on_random_two_port():
    rng = np.random.default_rng(4)
    s = rng.normal(size=(2, 2, 5)) + 1j * rng.normal(size=(2, 2, 5))
    s *= 0.45
    z_from = np.array([52.91, 31.7])
    z_to = np.array([50.0, 75.0])
    got = renormalize_power_waves(s, z_from, z_to)
    assert np.max(np.abs(got - _route_via_z(s, z_from, z_to))) < 1e-12


def test_jnp_path_matches_numpy_and_is_differentiable():
    rng = np.random.default_rng(5)
    s = _random_passive(3, 4, rng)
    z_from = np.array([52.91, 47.9, 50.0])
    z_to = np.array([50.0, 50.0, 50.0])
    with enable_x64():
        got = renormalize_power_waves(jnp.asarray(s), z_from, z_to)
        assert np.max(np.abs(np.asarray(got) - _route_via_z(s, z_from, z_to))) < 1e-12

        def objective(t):
            return jnp.sum(jnp.abs(renormalize_power_waves(
                jnp.asarray(s) * t, z_from, z_to)) ** 2)

        grad = float(jax.grad(objective)(0.7))
        h = 1e-6
        fd = (float(objective(0.7 + h)) - float(objective(0.7 - h))) / (2 * h)
        assert abs(grad - fd) < 1e-6 * max(1.0, abs(fd))


# ---- Coax two-port: the line under each probe ladder is the wave reference ----
#
# No FDTD: the run, the plane voltage and the ladder assembly are replaced, and
# the assembly returns a planted voltage-wave S. What is left is exactly the
# producer's reference handling, compared with the impedance-matrix route.

_PLANTED_COAX = np.array([[[0.21 + 0.05j], [0.62 - 0.31j]],
                          [[0.58 - 0.36j], [-0.17 + 0.12j]]])


def _coax_two_port(monkeypatch, eps_scale=None, impedance=50.0):
    from rfx.sources.coaxial_port import coaxial_tem_characteristic_impedance
    from rfx.sources import GaussianPulse
    import rfx.simulation as simulation_mod
    import rfx.sources.coaxial_port as coax_mod
    import rfx.sparams.coax as coax_lane

    class _Acc:
        accumulator = np.ones(1, dtype=np.complex128)

    class _Run:
        def __init__(self, n_planes, n_probes, dt):
            self.dft_planes = [_Acc() for _ in range(n_planes)]
            self.time_series = np.full((10, n_probes), 1e-6)
            self.dt = dt

    def fake_run(grid_, materials, n_steps, *, probes, dft_planes, **_):
        seen["eps_r"] = materials.eps_r
        return _Run(len(dft_planes), len(probes), grid_.dt)

    def fake_assemble(**kwargs):
        n_f = _PLANTED_COAX.shape[-1]
        return (_PLANTED_COAX, np.ones(n_f), np.zeros((2, 2, n_f)),
                np.zeros((2, 2, n_f)), np.zeros((2, 2, n_f), dtype=complex))

    monkeypatch.setattr(simulation_mod, "run", fake_run)
    monkeypatch.setattr(coax_mod, "coaxial_line_plane_voltage",
                        lambda grid_, ex, ey, **_: np.asarray(ex))
    monkeypatch.setattr(coax_mod, "coaxial_line_plane_voltage_jnp",
                        lambda grid_, ex, **_: jnp.asarray(ex))
    monkeypatch.setattr(coax_lane, "_assemble_coaxial_two_port_from_voltages",
                        fake_assemble)
    sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9,
                     boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         impedance=impedance,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    port = sim._coaxial_ports[0]
    z_tem = coaxial_tem_characteristic_impedance(port.pin_radius,
                                                 port.outer_radius)
    run = lambda scale: sim.compute_coaxial_two_port(  # noqa: E731
        n_steps=10, freqs=np.array([8.0e9]), probe_count=12, eps_scale=scale)
    return run, sim, z_tem


seen: dict = {}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_coax_two_port_is_referenced_to_the_declared_impedance(monkeypatch):
    run, _, z_tem = _coax_two_port(monkeypatch)
    res = run(None)
    np.testing.assert_array_equal(res.reference_impedances, [50.0, 50.0])
    np.testing.assert_array_equal(res.line_impedances, [z_tem, z_tem])
    want = _route_via_z(_PLANTED_COAX, [z_tem, z_tem], [50.0, 50.0])
    assert np.max(np.abs(np.asarray(res.s_params) - want)) < 1e-12
    assert np.max(np.abs(want - _PLANTED_COAX)) > 1e-3


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_coax_two_port_at_its_line_z0_returns_the_ratio_unchanged(monkeypatch):
    from rfx.sources.coaxial_port import (
        SMA_OUTER_RADIUS, SMA_PIN_RADIUS, coaxial_tem_characteristic_impedance)
    z_tem = coaxial_tem_characteristic_impedance(SMA_PIN_RADIUS, SMA_OUTER_RADIUS)
    run, _, _ = _coax_two_port(monkeypatch, impedance=z_tem)
    res = run(None)
    assert np.asarray(res.s_params).tobytes() == _PLANTED_COAX.tobytes()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_coax_two_port_eps_scale_moves_the_line_under_the_ladders(monkeypatch):
    # A uniform scale s over the whole line: the line under both ladders is
    # z_tem / sqrt(s), and that is the impedance the waves sit at.
    run, _, z_tem = _coax_two_port(monkeypatch)
    res = run(jnp.asarray(1.21))
    z_line = z_tem / 1.1
    np.testing.assert_allclose(res.line_impedances, [z_line, z_line], rtol=1e-7)  # float32 1.21
    want = _route_via_z(_PLANTED_COAX, [z_line, z_line], [50.0, 50.0])
    assert np.max(np.abs(np.asarray(res.s_params) - want)) < 1e-6


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_coax_two_port_unequal_ladder_lines_are_power_waves(monkeypatch):
    # The top half filled 1.21x: port 1 (top) reads a z_tem/1.1 line, port 2
    # (bottom) z_tem. The voltage ratio between them is a power-wave S only
    # after sqrt(Z_driven / Z_receiver).
    run, sim, z_tem = _coax_two_port(monkeypatch)
    nz = sim._build_grid().shape[2]
    scale = np.ones(sim._build_grid().shape)
    scale[:, :, nz // 2:] = 1.21
    res = run(jnp.asarray(scale))
    z_line = np.array([z_tem / 1.1, z_tem])
    np.testing.assert_allclose(res.line_impedances, z_line, rtol=1e-7)  # float32 1.21
    root = np.sqrt(z_line)
    s_power = _PLANTED_COAX * root[None, :, None] / root[:, None, None]
    want = _route_via_z(s_power, z_line, [50.0, 50.0])
    assert np.max(np.abs(np.asarray(res.s_params) - want)) < 1e-6


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_coax_two_port_refuses_a_scale_that_varies_over_a_ladder(monkeypatch):
    run, sim, _ = _coax_two_port(monkeypatch)
    shape = sim._build_grid().shape
    scale = 1.0 + 0.2 * (np.arange(shape[0]) % 2)[:, None, None] * np.ones(shape)
    with pytest.raises(ValueError, match="varies over the annulus"):
        run(jnp.asarray(scale))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_coax_two_port_gradient_carries_the_line_impedance_term(monkeypatch):
    # With the planted S fixed, everything d|S'|^2/ds holds comes from the
    # line under the ladders moving against the fixed 50 ohm reference.
    run, _, z_tem = _coax_two_port(monkeypatch)

    def route(s):
        z = z_tem / math.sqrt(s)
        return float(np.sum(np.abs(_route_via_z(_PLANTED_COAX, [z, z],
                                                [50.0, 50.0])) ** 2))

    with enable_x64():
        grad = float(jax.grad(
            lambda s: jnp.sum(jnp.abs(run(s).s_params) ** 2))(jnp.asarray(1.21)))
    h = 1e-6
    fd = (route(1.21 + h) - route(1.21 - h)) / (2 * h)
    assert abs(fd) > 1e-2
    assert abs(grad - fd) < 1e-6 * max(1.0, abs(fd))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_coax_two_port_fill_only_scale_reads_the_stamped_fill(monkeypatch):
    # A scale on exactly the cells the stamp filled (read from the materials
    # the run received, not rebuilt here) is one uniform line under each
    # ladder: no refusal, and z_tem / 1.1 at both ends.
    from rfx.sources.coaxial_port import PTFE_EPS_R
    run, _, z_tem = _coax_two_port(monkeypatch)
    run(None)
    fill = np.asarray(seen["eps_r"]) == np.float32(PTFE_EPS_R)
    assert fill.any()
    res = run(jnp.asarray(np.where(fill, 1.21, 1.0)))
    z_line = z_tem / 1.1
    np.testing.assert_allclose(res.line_impedances, [z_line, z_line], rtol=1e-7)
