"""WR-90 window deformation: PREDECLARE D4, B1 and B2 (2.54 mm).

Parameters are copied from the predeclared structure, not imported from the
chain battery. Only total-preserving tangents keep port planes fixed in metres.
"""

import math
import time

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box
from rfx.model.graded_admission import NominalProfile

C0 = 299792458.0
DX = 2.54e-3
CELLS = np.full(48, DX, dtype=np.float64)
WINDOW = (22, 30)
FD_STEP = 0.2e-3
D_MIN = DX - FD_STEP / 4
DT = float(0.9 / (C0 * np.sqrt(1 / D_MIN**2 + 2 / DX**2)))
FREQS = np.linspace(8.4e9, 11.6e9, 17)
# The realized wall-to-wall aperture is 9*DX by 4*DX. The battery's
# cpml_layers_for rule uses its TE10 cutoff and 0.75 guide wavelengths.
FC = C0 / (2 * 22.86e-3)
CPML = math.ceil(0.75 * C0 / FREQS[0] / np.sqrt(1 - (FC / FREQS[0])**2) / DX)


def section(delta):
    traced = is_tracer(delta)
    if traced:
        widths = jnp.asarray(CELLS, dtype=jnp.float32)
        widths = widths.at[22:26].add(delta / 4).at[26:30].add(-delta / 4)
    else:
        widths = CELLS.copy()
        widths[22:26] += float(delta) / 4
        widths[26:30] -= float(delta) / 4
    declaration = {"x": NominalProfile(CELLS, traced_cells=WINDOW)} if traced else None
    sim = Simulation(
        freq_max=float(FREQS[-1]), domain=(48 * DX, 9 * DX, 4 * DX), dx=DX,
        boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=CPML,
        dx_profile=widths, dt=DT, dt_min_cell=D_MIN, nominal_profiles=declaration,
    )
    sim.add_material("slab", eps_r=4.0)
    # Concrete FD faces follow the realized nodes; all cell centres stay on
    # their original side of the fixed box throughout the traced stencil.
    end = 26 * DX if traced else 26 * DX + float(delta)
    sim.add(Box((22 * DX, 0, 0), (end, 9 * DX, 4 * DX)), material="slab")
    for k, direction, name, ref in ((5, "+x", "left", 8), (43, "-x", "right", 40)):
        sim.add_waveguide_port(
            k * DX, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=FREQS, f0=10e9, bandwidth=0.5, name=name,
            ref_offset=3, probe_offset=10, reference_plane=ref * DX,
        )
    return sim


def realized_eps(delta):
    sim = section(delta)
    grid = sim._build_nonuniform_grid()
    return sim._assemble_materials_nu(grid)[0].eps_r


def test_d4_realized_eps_derivative_is_exactly_zero():
    derivative = np.asarray(jax.jacfwd(realized_eps)(0.0))
    assert np.count_nonzero(derivative) == 0
    concrete = np.asarray(realized_eps(0.0))
    assert set(np.unique(concrete)) == {1.0, 4.0}
    for delta in (-FD_STEP, FD_STEP):
        np.testing.assert_array_equal(realized_eps(delta), concrete)


@pytest.mark.parametrize("store", [False, True])
def test_d5_host_cells_match_each_sites_previous_precision(store):
    from rfx.model.graded_admission import host_cells
    grid = section(FD_STEP)._build_nonuniform_grid()
    for axis, old_store in zip("xyz", (grid.dx_arr, grid.dy_arr, grid.dz)):
        previous = np.asarray(old_store) if store else grid.cells(axis)
        actual = host_cells(grid, axis, store=store)
        assert actual.dtype == previous.dtype
        assert actual.tobytes() == previous.tobytes()

    def traced(delta):
        grid = section(delta)._build_nonuniform_grid()
        expected = np.full(48 + 2 * CPML + 1, DX, dtype=np.float32 if store else np.float64)
        actual = host_cells(grid, "x", store=store)
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
        return jnp.sum(grid.dx_arr)

    jax.jacfwd(traced)(0.0)


def s_matrix(delta, n_steps):
    return jnp.asarray(section(delta).compute_waveguide_s_matrix(
        n_steps=n_steps, normalize="flux").s_params)


def test_window_flux_jacfwd_reaches_extraction():
    gradient = jax.jacfwd(lambda d: jnp.real(s_matrix(d, 2)))(0.0)
    assert gradient.shape == (2, 2, 17)


def test_flux_ratio_tiny_reference_power_has_finite_quotient_derivative():
    from rfx.sparams.waveguide import _flux_power_ratio

    def ratio(delta):
        return _flux_power_ratio((2.0 + delta) * jnp.float32(1e-24),
                                 (3.0 + delta) * jnp.float32(1e-24))

    # Analytic derivative of (2+d)/(3+d) at zero, independent of the scale.
    for transform in (jax.jacfwd, jax.grad):
        assert float(transform(ratio)(0.0)) == pytest.approx(1 / 9, rel=1e-6)
    numerator, denominator = jnp.float32(2e-24), jnp.float32(3e-24)
    assert np.asarray(_flux_power_ratio(numerator, denominator)).tobytes() == np.asarray(
        numerator / denominator).tobytes()


def test_b1_b2_window_gradient_all_17_bins():
    from rfx.stepping.step_count import steps_for_periods
    sim = section(0.0)
    grid = sim._build_nonuniform_grid()
    # NonUniformGrid has no num_timesteps method: this is the same helper
    # Grid.num_timesteps(40) calls, using this concrete build's pinned dt.
    n_steps = steps_for_periods(grid.dt, sim._freq_max, 40)
    started = time.perf_counter()

    def amplitudes_and_primal(delta):
        s = s_matrix(delta, n_steps)
        return jnp.stack((jnp.abs(s[1, 0]), jnp.abs(s[0, 0]))), s

    gradient, traced = jax.jacfwd(amplitudes_and_primal, has_aux=True)(0.0)
    gradient, traced = np.asarray(gradient), np.asarray(traced)
    concrete = np.asarray(s_matrix(0.0, n_steps))
    minus = np.asarray(s_matrix(-FD_STEP, n_steps))
    plus = np.asarray(s_matrix(FD_STEP, n_steps))
    fd = np.stack(((np.abs(plus[1, 0]) - np.abs(minus[1, 0])) / (2 * FD_STEP),
                   (np.abs(plus[0, 0]) - np.abs(minus[0, 0])) / (2 * FD_STEP)))
    primal_error = float(np.max(np.abs(traced - concrete)))
    primal_bar = float(1e-4 * np.max(np.abs(concrete)))
    print(f"B1 n_steps={n_steps}: max_abs={primal_error:.12g}, bar={primal_bar:.12g}")
    for row in range(2):
        for column in range(2):
            print(f"B1 S{row+1}{column+1}: max_abs={np.max(np.abs(traced[row,column]-concrete[row,column])):.12g}")
    errors = np.max(np.abs(gradient - fd), axis=1)
    bars = 2e-2 * np.max(np.abs(fd), axis=1)
    for name, error, bar, ad, central in zip(("S21", "S11"), errors, bars, gradient, fd):
        print(f"B2 abs({name}): max_abs={float(error):.12g}, bar={float(bar):.12g}")
        print(f"B2 abs({name}) AD={ad.tolist()} FD={central.tolist()}")
    print(f"B1/B2 wall_seconds={time.perf_counter()-started:.6f}")
    assert np.isfinite(gradient).all() and np.isfinite(fd).all()
    assert primal_error <= primal_bar
    assert np.all(errors <= bars), (errors, bars)
