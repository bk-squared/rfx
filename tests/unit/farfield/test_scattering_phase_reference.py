"""Complex scattering needs the incident field's location AND E-update clock.

The incident witness reads actual 3-D probe samples independently of the
auxiliary replay. Delay invariance then exercises target/reference solves and
NTFF together. These are phase-bookkeeping contracts, not a continuum
accuracy certification for all scatterers.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import rfx.rcs as rcs
from rfx.core.yee import MaterialArrays
from rfx.grid import C0, Grid
from rfx.simulation import ProbeSpec, run
from rfx.sources.tfsf import init_tfsf, measure_normal_incident_spectrum
from tests._x64_compat import enable_x64

F0 = 6e9
FREQS = F0 * np.array([0.8, 1.0, 1.2])
STEPS = 360


def _rig(*, dtype=jnp.float32, **grid_options):
    dx = C0 / F0 / 20
    grid = Grid(freq_max=1.5 * F0, domain=(20 * dx,) * 3, dx=dx, cpml_layers=8,
                **grid_options)
    vacuum = MaterialArrays(jnp.ones(grid.shape, dtype), jnp.zeros(grid.shape, dtype),
                            jnp.ones(grid.shape, dtype))
    return grid, vacuum


def _reference(grid):
    return tuple(grid.node_of(a, grid.shape[a] // 2) for a in range(3))


@pytest.mark.parametrize("polarization", ["ey", "ez"])
def test_incident_phase_matches_actual_3d_e_clock_and_position(polarization):
    grid, vacuum = _rig()
    cfg, state = init_tfsf(nx=grid.nx, dx=grid.dx, dt=grid.dt, cpml_layers=8,
                           f0=F0, polarization=polarization, tfsf_margin=3)
    indices = [cfg.x_lo + 2, grid.nx // 2, cfg.x_hi - 2]
    probes = [ProbeSpec(i, grid.ny // 2, grid.nz // 2, polarization) for i in indices]
    result = run(grid, vacuum, STEPS, boundary="cpml", tfsf=(cfg, state), probes=probes)
    samples = np.asarray(result.time_series, dtype=np.float64)
    # Probe samples are the post-update E field, not samples at n*dt.
    times = (np.arange(STEPS) + 1) * float(grid.dt)
    measured = np.exp(-2j * np.pi * FREQS[:, None] * times) @ samples * float(grid.dt)
    assert np.max(np.abs(samples[-20:])) < 1e-3 * np.max(np.abs(samples))
    for column, index in enumerate(indices):
        replay = measure_normal_incident_spectrum(
            cfg, state, STEPS, FREQS, grid.dt, reference_index=index,
        )
        np.testing.assert_allclose(replay, measured[:, column], rtol=2e-3, atol=0)
    # A common DFT step error is large despite identical |E| and RCS.
    assert np.max(np.abs(np.exp(2j * np.pi * FREQS * grid.dt) - 1)) > 0.1


def _scatter(grid, materials, **kw):
    options = dict(f0=F0, freqs=FREQS,
        theta_obs=np.array([np.pi / 3, np.pi / 2]),
        phi_obs=np.array([0.0, np.pi / 2, np.pi]),
        subtract_incident_reference=True, phase_reference=_reference(grid),
    )
    options.update(kw)
    return rcs.compute_rcs(grid, materials, STEPS, **options)


def test_source_delay_cancels_from_complex_response(monkeypatch):
    grid, vacuum = _rig()
    i, j, k = (n // 2 for n in grid.shape)
    target = vacuum._replace(eps_r=vacuum.eps_r.at[i-2:i+3, j-1:j+2, k-2:k+3].set(2.5))
    original = _scatter(grid, target)
    real_init = rcs.init_tfsf
    delay = 27 * float(grid.dt)
    source_config = []

    def delayed_source(*args, **kwargs):
        cfg, state = real_init(*args, **kwargs)
        source_config.append(cfg)
        return cfg._replace(src_t0=cfg.src_t0 + delay), state

    monkeypatch.setattr(rcs, "init_tfsf", delayed_source)
    delayed = _scatter(grid, target)
    assert isinstance(original, rcs.ScatteringResponse)
    # The Gaussian starts at 3*tau, not minus infinity. Delaying it changes
    # the small omitted leading tail, so an ideal exp(-jw*delay) would be
    # the wrong comparator. DFT the two actual finite input waveforms.
    cfg = source_config[0]
    times = np.arange(STEPS) * float(grid.dt)
    spectra = []
    for t0 in (cfg.src_t0, cfg.src_t0 + delay):
        arg = (times - t0) / cfg.src_tau
        pulse = -2 * arg * np.exp(-arg**2)
        spectra.append(np.exp(-2j * np.pi * FREQS[:, None] * times) @ pulse)
    np.testing.assert_allclose(delayed.incident_spectrum / original.incident_spectrum,
                               spectra[1] / spectra[0], rtol=2e-3, atol=0)
    for key in ("F_theta", "F_phi"):
        before, after = getattr(original, key), getattr(delayed, key)
        assert np.max(np.abs(before)) > 1e-5  # the test cannot pass on zero scattering
        assert np.max(np.abs(after - before)) / np.max(np.abs(before)) < 2e-3
    sigma = 4 * np.pi * (np.abs(original.F_theta)**2 + np.abs(original.F_phi)**2)
    np.testing.assert_allclose(sigma, original.rcs.rcs_linear, rtol=2e-14, atol=0)
    np.testing.assert_allclose(10*np.log10(sigma[:,1,2]), original.rcs.monostatic_rcs,
                               rtol=0, atol=1e-12)


def test_moving_the_reference_along_the_outgoing_ray_has_retarded_phase():
    grid, vacuum = _rig()
    i, j, k = (n // 2 for n in grid.shape)
    target = vacuum._replace(eps_r=vacuum.eps_r.at[i-2:i+3, j-1:j+2, k-2:k+3].set(2.5))
    original = _scatter(grid, target)
    position = np.asarray(_reference(grid))
    distance = 4 * grid.dx
    position[1] += distance
    moved = _scatter(grid, target, phase_reference=tuple(position))
    # The input is uniform along y. An output reference moved toward +y
    # sees exp(-jk*d) of the old amplitude under exp(+jwt), from the
    # retarded Green function. The actual object/source/grid do not move.
    np.testing.assert_array_equal(moved.incident_spectrum, original.incident_spectrum)
    expected = np.cos(2*np.pi*FREQS*distance/C0) - 1j*np.sin(2*np.pi*FREQS*distance/C0)
    np.testing.assert_allclose(moved.F_theta[:,1,1] / original.F_theta[:,1,1],
                               expected, rtol=1e-13, atol=0)


def test_vacuum_subtraction_has_zero_complex_response():
    grid, vacuum = _rig()
    response = _scatter(grid, vacuum)
    np.testing.assert_array_equal(response.F_theta, 0)
    np.testing.assert_array_equal(response.F_phi, 0)


def test_float64_vacuum_subtraction_matches_material_precision():
    with enable_x64():
        grid, vacuum = _rig(dtype=jnp.float64)
        response = _scatter(grid, vacuum)
        np.testing.assert_array_equal(response.F_theta, 0)
        np.testing.assert_array_equal(response.F_phi, 0)


@pytest.mark.parametrize("grid_options", [
    {"cpml_axes": "xy"}, {"pec_faces": {"y_lo"}}, {"pmc_faces": {"z_lo"}},
])
def test_incomplete_realized_cpml_refuses_before_solving(monkeypatch, grid_options):
    grid, vacuum = _rig(**grid_options)
    # These legal grids retain nominal face_layers=8 but omit actual padding.
    assert set(grid.face_layers.values()) == {8}

    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete CPML reached the solver")

    monkeypatch.setattr(rcs, "run", forbidden)
    with pytest.raises(NotImplementedError, match="symmetric CPML"):
        _scatter(grid, vacuum)


@pytest.mark.parametrize("change,exception,match", [
    ({"subtract_incident_reference": False}, ValueError, "subtract_incident_reference"),
    ({"theta_inc": 20.0}, NotImplementedError, "normal"),
    ({"phi_inc": 10.0}, NotImplementedError, "normal"),
    ({"boundary": "pec"}, NotImplementedError, "CPML"),
    ({"cpml_layers": 6}, NotImplementedError, "CPML"),
    ({"phase_reference": (np.nan, 0.02, 0.02)}, ValueError, "finite"),
    ({"phase_reference": (0.02, 0.02)}, ValueError, "finite"),
    ({"phase_reference": (0.0, 0.02, 0.02)}, ValueError, "total-field"),
    ({"phase_reference": (0.02001, 0.02, 0.02)}, ValueError, "E-node"),
    ({"freqs": [0.0]}, ValueError, "frequencies"),
])
def test_unsupported_phase_contract_refuses_before_solving(monkeypatch, change, exception, match):
    grid, vacuum = _rig()
    kwargs = dict(f0=F0, freqs=FREQS, subtract_incident_reference=True,
                  phase_reference=_reference(grid))
    kwargs.update(change)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported phase input reached the solver")

    monkeypatch.setattr(rcs, "run", forbidden)
    with pytest.raises(exception, match=match):
        rcs.compute_rcs(grid, vacuum, STEPS, **kwargs)


@pytest.mark.parametrize("index", [-1, 0.5, True, 999])
def test_incident_reference_index_refuses_outside_slab(index):
    grid, _ = _rig()
    cfg, state = init_tfsf(nx=grid.nx, dx=grid.dx, dt=grid.dt, cpml_layers=8, f0=F0)
    with pytest.raises(ValueError, match="integer E-node"):
        measure_normal_incident_spectrum(cfg, state, STEPS, FREQS, grid.dt, reference_index=index)
