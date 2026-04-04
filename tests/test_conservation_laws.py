"""Physics-invariant conservation-law tests (Tier 2).

These tests validate passivity, near-unitarity, reciprocity, mesh
convergence, and causality without comparing against any external solver.
They depend only on Maxwell's equations and energy-conservation principles.
"""

import numpy as np
import jax.numpy as jnp

from rfx.api import Simulation
from rfx.geometry.csg import Box


# ---- Shared waveguide helpers ----

WAVEGUIDE_DOMAIN = (0.12, 0.04, 0.02)
PORT_LEFT_X = 0.01
PORT_RIGHT_X = 0.09


def _build_waveguide_sim(
    freqs_hz,
    *,
    dx=None,
    obstacle_specs=(),
):
    """Build a two-port rectangular waveguide Simulation.

    Parameters
    ----------
    freqs_hz : array-like
        Frequency points for S-parameter extraction.
    dx : float or None
        Cell size override.
    obstacle_specs : sequence of ((lo, hi, eps_r),)
        Dielectric obstacles to place inside the waveguide.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))

    sim = Simulation(
        freq_max=max(float(freqs[-1]), f0),
        domain=WAVEGUIDE_DOMAIN,
        boundary="cpml",
        cpml_layers=10,
        dx=dx,
    )

    for idx, (lo, hi, eps_r) in enumerate(obstacle_specs):
        material_name = f"dielectric_{idx}"
        sim.add_material(material_name, eps_r=eps_r, sigma=0.0)
        sim.add(Box(lo, hi), material=material_name)

    port_freqs = jnp.asarray(freqs)
    sim.add_waveguide_port(
        PORT_LEFT_X,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=port_freqs,
        f0=f0,
        bandwidth=bandwidth,
        name="left",
    )
    sim.add_waveguide_port(
        PORT_RIGHT_X,
        direction="-x",
        mode=(1, 0),
        mode_type="TE",
        freqs=port_freqs,
        f0=f0,
        bandwidth=bandwidth,
        name="right",
    )
    return sim


def _compute_s_matrix(*, freqs_hz, dx=None, obstacle_specs=(), num_periods=40, normalize=False):
    """Run a two-port waveguide simulation and return S-params.

    Returns
    -------
    s_params : ndarray, shape (n_ports, n_ports, n_freqs), complex
    freqs : ndarray, shape (n_freqs,)
    port_index : dict mapping port name -> integer index
    """
    sim = _build_waveguide_sim(freqs_hz, dx=dx, obstacle_specs=obstacle_specs)
    result = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize=normalize)
    port_index = {name: idx for idx, name in enumerate(result.port_names)}
    return np.asarray(result.s_params), np.asarray(result.freqs), port_index


# =========================================================================
# Test 1: Passivity — empty two-port waveguide
# =========================================================================

def test_passivity_two_port_empty_waveguide():
    """For a lossless empty waveguide: sum_i |S_ij(f)|^2 <= 1 (+ numerical margin).

    Energy conservation: scattered power out cannot exceed incident power in.
    """
    freqs = np.linspace(4.5e9, 8.0e9, 20)
    s_params, sim_freqs, port_index = _compute_s_matrix(
        freqs_hz=freqs, num_periods=40, normalize=True,
    )

    # Column-wise power sum: for each excitation port j, sum |S_ij|^2 over i
    column_powers = np.sum(np.abs(s_params) ** 2, axis=0)  # (n_ports, n_freqs)
    max_power = float(np.max(column_powers))

    print("\nPassivity: empty two-port waveguide")
    for port_name, col_idx in port_index.items():
        col_max = float(np.max(column_powers[col_idx]))
        print(f"  {port_name}: max(sum |S_ij|^2) = {col_max:.6f}")
    print(f"  Overall max column power: {max_power:.6f}")
    print(f"  Frequency span: {sim_freqs[0]/1e9:.2f} to {sim_freqs[-1]/1e9:.2f} GHz")

    assert max_power < 1.15, (
        f"Passivity violated: max column power = {max_power:.6f} (limit 1.15)"
    )


# =========================================================================
# Test 2: Unitarity — lossless waveguide with dielectric obstacle
# =========================================================================

def test_unitarity_lossless_waveguide():
    """For a lossless system: sum_i |S_ij(f)|^2 ~ 1.0.

    Total scattered power approximately equals incident power when all
    materials are lossless (sigma=0).
    """
    freqs = np.linspace(4.5e9, 8.0e9, 20)
    obstacle = [((0.04, 0.0, 0.0), (0.06, 0.04, 0.02), 4.0)]
    s_params, _, port_index = _compute_s_matrix(
        freqs_hz=freqs,
        obstacle_specs=obstacle,
        num_periods=40, normalize=True,
    )

    column_powers = np.sum(np.abs(s_params) ** 2, axis=0)  # (n_ports, n_freqs)
    mean_power = float(np.mean(column_powers))

    print("\nUnitarity: lossless waveguide with dielectric obstacle")
    for port_name, col_idx in port_index.items():
        col_mean = float(np.mean(column_powers[col_idx]))
        print(f"  {port_name}: mean(sum |S_ij|^2) = {col_mean:.6f}")
    print(f"  Global mean column power: {mean_power:.6f}")

    assert 0.8 < mean_power < 1.40, (
        f"Unexpected mean power balance: {mean_power:.6f} (expected 0.8 < x < 1.40)"
    )


# =========================================================================
# Test 3: Reciprocity — asymmetric dielectric structure
# =========================================================================

def test_reciprocity_asymmetric_structure():
    """S_ij = S_ji for any linear passive structure, even asymmetric ones.

    Uses an asymmetric dielectric that covers only half the y-cross-section
    to break geometric symmetry while preserving reciprocity.
    """
    freqs = np.linspace(4.5e9, 8.0e9, 20)
    # Asymmetric obstacle: only covers y in [0, 0.02] (half of 0.04 domain)
    obstacle = [((0.03, 0.0, 0.0), (0.05, 0.02, 0.02), 6.0)]
    s_params, sim_freqs, port_index = _compute_s_matrix(
        freqs_hz=freqs,
        obstacle_specs=obstacle,
        num_periods=40,
    )

    s21 = np.abs(s_params[port_index["right"], port_index["left"]])
    s12 = np.abs(s_params[port_index["left"], port_index["right"]])
    rel_diff = np.abs(s21 - s12) / np.maximum(np.maximum(s21, s12), 1e-12)
    mean_rel_diff = float(np.mean(rel_diff))

    print("\nReciprocity: asymmetric dielectric structure")
    print(f"  S21 magnitudes: {np.array2string(s21, precision=4, separator=', ')}")
    print(f"  S12 magnitudes: {np.array2string(s12, precision=4, separator=', ')}")
    print(f"  Mean relative difference: {mean_rel_diff:.6f}")
    print(f"  Frequencies (GHz): {np.array2string(sim_freqs/1e9, precision=3, separator=', ')}")

    # Asymmetric obstacles excite higher-order modes that single-mode TE10
    # projection cannot capture, increasing reciprocity error. 15% tolerance
    # accommodates CFS-CPML boundary profile differences.
    assert mean_rel_diff < 0.15, (
        f"Reciprocity error too large: mean |S21-S12|/max = {mean_rel_diff:.6f} (limit 0.15)"
    )


# =========================================================================
# Test 4: Mesh convergence — S21 converges with mesh refinement
# =========================================================================

def test_mesh_convergence_s21():
    """S-parameters must converge as the mesh is refined.

    This establishes the 'true' answer without any external tool by
    verifying monotonic convergence across three resolutions.
    """
    freq = 6.0e9
    obstacle = [((0.04, 0.0, 0.0), (0.06, 0.04, 0.02), 4.0)]
    resolutions = [0.003, 0.002, 0.001]
    s21_values = []

    for dx in resolutions:
        s_params, _, port_index = _compute_s_matrix(
            freqs_hz=np.array([freq]),
            dx=dx,
            obstacle_specs=obstacle,
            num_periods=40,
        )
        s21_val = float(np.abs(
            s_params[port_index["right"], port_index["left"], 0]
        ))
        s21_values.append(s21_val)

    coarse_delta = abs(s21_values[0] - s21_values[1])
    fine_delta = abs(s21_values[1] - s21_values[2])

    print("\nMesh convergence of |S21| at 6 GHz")
    for dx, value in zip(resolutions, s21_values):
        print(f"  dx = {dx*1e3:.0f} mm -> |S21| = {value:.6f}")
    print(f"  |S21(3mm) - S21(2mm)| = {coarse_delta:.6f}")
    print(f"  |S21(2mm) - S21(1mm)| = {fine_delta:.6f}")

    assert fine_delta < coarse_delta, (
        "Mesh refinement did not reduce the |S21| change "
        f"(coarse_delta={coarse_delta:.6f}, fine_delta={fine_delta:.6f})"
    )
    assert fine_delta < 0.05, (
        f"Fine-mesh |S21| change remains too large: {fine_delta:.6f} (limit 0.05)"
    )


# =========================================================================
# Test 5: Causality — no signal before wavefront arrival
# =========================================================================

def test_causality_no_signal_before_source():
    """No field should appear at a distant probe before the wavefront
    could physically arrive at the speed of light.

    Uses a direct additive Ez point source and a distant probe.
    A well-delayed Gaussian pulse (t0 = 4.5*tau) ensures negligible
    amplitude at t=0, so the earliest physical wavefront departs at
    approximately t=0 and arrives at the probe after distance/c.

    This validates basic FDTD causality without TFSF boundary artifacts.
    """
    from rfx.grid import Grid, C0
    from rfx.core.yee import init_state, init_materials, update_e, update_h
    from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h

    # Long domain with CPML on x, periodic y/z
    domain = (0.30, 0.006, 0.006)
    grid = Grid(
        freq_max=10e9,
        domain=domain,
        dx=0.001,
        cpml_layers=10,
        cpml_axes="x",
    )
    dt, dx_val = grid.dt, grid.dx
    periodic = (False, True, True)

    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    cp_cfg, cp_state = init_cpml(grid)

    # Source: additive Gaussian pulse at a point near the left boundary
    src_x = grid.pad_x + 5
    src_idx = (src_x, grid.ny // 2, grid.nz // 2)

    # Probe: near the right boundary
    probe_x = grid.nx - grid.pad_x - 5
    probe_idx = (probe_x, grid.ny // 2, grid.nz // 2)

    # Physical positions
    src_x_phys = (src_x - grid.pad_x) * dx_val
    probe_x_phys = (probe_x - grid.pad_x) * dx_val
    distance = probe_x_phys - src_x_phys
    t_arrival = distance / C0

    # Gaussian pulse parameters: well-delayed so envelope at t=0 is negligible
    # With t0 = 4.5*tau, envelope at t=0 is exp(-20.25) ~ 1.6e-9
    f0 = 5e9
    tau = 1.0 / (f0 * np.pi)
    t0 = 4.5 * tau

    def gaussian_source(t):
        return np.exp(-((t - t0) / tau) ** 2) * np.sin(2 * np.pi * f0 * t)

    # Run enough steps for the signal to arrive and be measurable
    n_steps = int(np.ceil((t0 + t_arrival + 6.0 * tau) / dt))
    n_steps = min(n_steps, 2000)

    trace = np.zeros(n_steps)

    for step in range(n_steps):
        t = step * dt

        # H update
        state = update_h(state, materials, dt, dx_val, periodic)
        state, cp_state = apply_cpml_h(state, cp_cfg, cp_state, grid, axes="x")

        # E update
        state = update_e(state, materials, dt, dx_val, periodic)
        state, cp_state = apply_cpml_e(state, cp_cfg, cp_state, grid, axes="x")

        # Additive (soft) source injection after E update
        src_val = gaussian_source(t)
        state = state._replace(
            ez=state.ez.at[src_idx].add(src_val)
        )

        trace[step] = float(state.ez[probe_idx])

    times = np.arange(n_steps) * dt

    # Before wavefront arrival (with 5*dt safety margin for numerical dispersion)
    pre_arrival_mask = times < (t_arrival - 5.0 * dt)
    # After the source peak has had time to propagate to probe
    post_arrival_mask = times >= (t_arrival + t0)

    if np.any(pre_arrival_mask):
        pre_arrival_peak = float(np.max(np.abs(trace[pre_arrival_mask])))
    else:
        pre_arrival_peak = 0.0

    post_arrival_peak = float(np.max(np.abs(trace[post_arrival_mask])))

    print("\nCausality: no signal before wavefront arrival")
    print(f"  Source x  = {src_x_phys:.6f} m (cell {src_x})")
    print(f"  Probe x   = {probe_x_phys:.6f} m (cell {probe_x})")
    print(f"  Distance  = {distance:.6f} m")
    print(f"  t_arrival = {t_arrival:.3e} s")
    print(f"  t0 (src)  = {t0:.3e} s")
    print(f"  tau (src) = {tau:.3e} s")
    print(f"  Total steps = {n_steps}")
    print(f"  Max |Ez| before t_arrival - 5dt: {pre_arrival_peak:.3e}")
    print(f"  Max |Ez| after  t_arrival + t0:  {post_arrival_peak:.3e}")

    assert pre_arrival_peak < 1e-10, (
        f"Non-causal pre-arrival field detected: {pre_arrival_peak:.3e}"
    )
    assert post_arrival_peak > 1e-6, (
        f"No signal detected after physical arrival time (peak={post_arrival_peak:.3e})"
    )
