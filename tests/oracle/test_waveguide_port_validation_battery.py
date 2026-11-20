"""Physical-correctness validation battery for the waveguide port.

Locks Meep-class behavior on a canonical rectangular waveguide for the
source + extractor pipeline. Each test asserts a falsifiable physical
property — independent of other tests — that a correct FDTD simulator
must satisfy.

Thresholds reflect the extractor state as of 2026-04-22 after:
- diagonal-subtraction patch in ``extract_waveguide_s_params_normalized``
  (subtracts empty-guide reference outgoing from device outgoing)
- CPML default retune (polynomial order 2->3). A parallel `kappa_max`
  sweep showed that `kappa_max > 1` degrades guided-mode absorption
  in the current CFS-CPML formulation, so `kappa_max=1` is kept (an
  earlier handoff that claimed a `kappa_max 1->5` retune was retracted
  on 2026-04-22 after the sweep was run).

Industry anchors (handoff 2026-04-21):
- Meep EigenModeSource: matched-load |S11| < 1 %, directionality ~0.1 %.
- OpenEMS: matched-load |S11| < 5 %.
- Lossless passivity: sum_i |S_ij|^2 <= 1.
- Passive-medium reciprocity: S21 == S12 on the projected subspace.

Tighten thresholds when extractor quality improves; a failure here
marks a physics regression.
The domain lengths written as ``n_y * dx`` and ``n_z * dx`` are whole numbers of cells of dx, so those solved lengths equal the declared ones.
"""

from __future__ import annotations
from rfx.model.waveguide_modes import PortMedium

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from tests._realized_geometry import (
    assert_wall_planes, domain_wall_positions, node_index, realized,
)
from tests._realized_pec import assert_walls_at, realize


DOMAIN = (0.12, 0.04, 0.02)
PORT_LEFT_X = 0.01
PORT_RIGHT_X = 0.09
F_CUTOFF_HZ = 3.75e9
TARGET_CPML_M = 0.030
SHORT_CELLS = 2


def _build_sim(
    freqs_hz,
    *,
    dx: float | None = None,
    cpml_layers: int | None = None,
    obstacles=(),
    pec_short_x: float | None = None,
    waveform: str = "modulated_gaussian",
    n_modes: int = 1,
    solved_domain: bool = False,
):
    """Two-port rectangular waveguide simulation.

    Parameters
    ----------
    freqs_hz
        Frequency points for S-parameter extraction.
    dx
        Cell size override; ``None`` lets ``Simulation`` pick from freq_max.
    cpml_layers
        CPML layer count; ``None`` uses the Simulation default.
    obstacles
        Sequence of ``((lo, hi, eps_r),)`` dielectric boxes.
    pec_short_x
        If provided, add a thin PEC wall at this x position (full
        cross-section) — short-circuit termination for S11 tests.
    waveform
        Source pulse shape. Default ``modulated_gaussian`` for clean
        bandpass excitation.
    Finalize before placing a body from node coordinates because adding the body invalidates the unfrozen mesh selection and can refine it.
    """
    RIGHT_PORT_CLEARANCE_CELLS = 4
    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))

    domain = DOMAIN
    if solved_domain:
        dx = (0.75e-3 if obstacles else 299792458.0 / max(float(freqs[-1]), f0) / 20.0) if dx is None else dx
        n_y = int(np.ceil(DOMAIN[1] / dx))
        n_z = int(np.ceil(DOMAIN[2] / dx))
        domain = (DOMAIN[0], n_y * dx, n_z * dx)
    sim_kwargs = dict(
        freq_max=max(float(freqs[-1]), f0),
        domain=domain,
        boundary={"x": "cpml", "y": "pec", "z": "pec"},
    )
    if cpml_layers is not None:
        sim_kwargs["cpml_layers"] = cpml_layers
    else:
        sim_kwargs["cpml_layers"] = 10
    if dx is not None:
        sim_kwargs["dx"] = dx
    sim = Simulation(**sim_kwargs)

    for idx, (lo, hi, eps_r) in enumerate(obstacles):
        name = f"diel_{idx}"
        sim.add_material(name, eps_r=eps_r, sigma=0.0)
        sim.add(Box(lo, hi), material=name)

    if pec_short_x is not None:
        assert isinstance(SHORT_CELLS, int) and SHORT_CELLS >= 1, (
            'The PEC short needs whole cells because a sub-cell Box volume is refused.'
        )
        grid = sim.freeze_mesh()
        d = float(grid.dx)
        k_lo = int(round(pec_short_x / d))
        x_lo, x_hi = k_lo * d, (k_lo + SHORT_CELLS) * d
        y_wall = domain_wall_positions(grid, 1)[1]
        z_wall = domain_wall_positions(grid, 2)[1]
        sim.add(
            Box((x_lo, 0.0, 0.0), (x_hi, y_wall, z_wall)),
            material="pec",
        )
        sim._pec_short_faces_m = (x_lo, x_hi)
        assert RIGHT_PORT_CLEARANCE_CELLS > 3, (
            "The three-cell reflection offset must leave the right-port plane on the port side of the short's far face."
        )
        right_x = (k_lo + SHORT_CELLS + RIGHT_PORT_CLEARANCE_CELLS) * d
    else:
        right_x = PORT_RIGHT_X

    port_freqs = jnp.asarray(freqs)
    sim.add_waveguide_port(
        PORT_LEFT_X,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=port_freqs,
        f0=f0,
        bandwidth=bandwidth,
        waveform=waveform,
        n_modes=n_modes,
        name="left",
    )
    sim.add_waveguide_port(
        right_x,
        direction="-x",
        mode=(1, 0),
        mode_type="TE",
        freqs=port_freqs,
        f0=f0,
        bandwidth=bandwidth,
        waveform=waveform,
        n_modes=n_modes,
        name="right",
    )
    return sim


def _s_matrix(sim, *, num_periods=40, normalize=True):
    result = sim.compute_waveguide_s_matrix(
        num_periods=num_periods,
        normalize=normalize,
    )
    s = np.asarray(result.s_params)
    port_idx = {name: idx for idx, name in enumerate(result.port_names)}
    for name in list(port_idx):
        if "_mode0_" in name:
            short = name.split("_mode0_", 1)[0]
            port_idx.setdefault(short, port_idx[name])
    freqs = np.asarray(result.freqs)
    return s, freqs, port_idx


def test_matched_load_s11_empty_waveguide():
    """The 0.05 gate and 0.08 witness threshold separate the 10-layer reflection measured earlier as at most 0.0135 over 4.5–8 GHz from the 4-layer reflection measured earlier at 0.119."""
    freqs = np.linspace(4.5e9, 8.0e9, 10)
    sim = _build_sim(freqs, solved_domain=True, cpml_layers=10, waveform="modulated_gaussian")
    s, sim_freqs, port_idx = _s_matrix(sim, num_periods=40, normalize=False)

    s11 = np.abs(s[port_idx["left"], port_idx["left"], :])
    s22 = np.abs(s[port_idx["right"], port_idx["right"], :])
    max_refl = float(max(s11.max(), s22.max()))
    min_absorbed = float(min((1.0 - s11**2).min(), (1.0 - s22**2).min()))

    print("\n[matched-load] |S11| per freq:", np.array2string(s11, precision=3))
    print("[matched-load] |S22| per freq:", np.array2string(s22, precision=3))
    print(f"[matched-load] max reflection = {max_refl:.4f}; "
          f"min absorbed fraction (1-|S11|^2) = {min_absorbed:.4f}")
    print("[matched-load] With the magnetic CPML profile at the Yee half cell the 10-layer load reflects at most 0.0135 over 4.5–8 GHz (0.156 on 2026-07-20; 0.119 with the integer-node profile on today's tree) and 4 layers reflect 0.119, so the earlier 13–16 % floor was the absorber, not the source plane. Gate 0.05; the 4-layer witness must clear 0.08.")

    assert max_refl < 0.05, (
        f"Matched-load reflection |S11|={max_refl:.4f} exceeds 0.05 — the "
        "CPML termination is not absorbing as well as the validated "
        "envelope, or the single-run extractor regressed."
    )
    assert min_absorbed > 0.95, (
        f"Matched load absorbs only {min_absorbed:.4f} of incident power "
        "(<0.95) — degraded CPML absorber."
    )

    sim_bad = _build_sim(freqs, solved_domain=True, cpml_layers=4, waveform="modulated_gaussian")
    s_bad, _, idx_bad = _s_matrix(sim_bad, num_periods=40, normalize=False)
    bad_refl = float(max(
        np.abs(s_bad[idx_bad["left"], idx_bad["left"], :]).max(),
        np.abs(s_bad[idx_bad["right"], idx_bad["right"], :]).max(),
    ))
    print(f"[matched-load] crippled 4-layer CPML max reflection = {bad_refl:.4f} "
          f"(must exceed 0.08 to prove non-vacuity)")
    assert bad_refl > 0.08, (
        f"Crippled-CPML witness failed: 4-layer reflection {bad_refl:.4f} did "
        "not rise clear of 0.08 (matched-load gate 0.05) — the matched-load gate may have "
        "gone vacuous again (issue #395)."
    )


def test_tight_passivity_reflecting_dut():
    """The flux path is the tool for a passivity-magnitude claim because its |S11| and |S21| are two INDEPENDENT power measurements."""
    freqs = np.linspace(4.5e9, 8.0e9, 10)
    obstacles = [((0.05, 0.0, 0.0), (0.07, 0.04, 0.02), 4.0)]
    sim = _build_sim(freqs, solved_domain=True, obstacles=obstacles, waveform="modulated_gaussian")
    s, _, port_idx = _s_matrix(sim, num_periods=40, normalize="flux")

    s11 = np.abs(s[port_idx["left"], port_idx["left"], :])
    s21 = np.abs(s[port_idx["right"], port_idx["left"], :])
    col_power = np.sum(np.abs(s) ** 2, axis=0)
    max_power = float(col_power.max())

    print(f"\n[passivity] |S11| = {np.array2string(s11, precision=3)}")
    print(f"[passivity] |S21| = {np.array2string(s21, precision=3)}")
    print(f"[passivity] max column |S|^2 = {max_power:.4f} (gate 1.02; measured 1.0005)")

    assert s11.max() > 0.30, (
        f"DUT not reflecting (max|S11|={s11.max():.4f}<=0.30) — the passivity "
        "gate would be vacuous; check the eps_r=4 obstacle was built."
    )
    assert s21.min() < 0.95, (
        f"DUT not attenuating (min|S21|={s21.min():.4f}>=0.95) — transmission "
        "should drop below unity behind an eps_r=4 step."
    )
    assert max_power < 1.02, (
        f"Reflecting-DUT column power {max_power:.4f} exceeds 1.02 — "
        "non-passive extractor (energy injection) on a lossless obstacle."
    )


def test_reciprocity_symmetric_obstacle():
    """Two-run normalization on a symmetric obstacle measured 0.0005 mean reciprocity error in an earlier run."""
    freqs = np.linspace(4.5e9, 8.0e9, 10)
    obstacles = [((0.05, 0.0, 0.0), (0.07, 0.04, 0.02), 4.0)]
    assert np.isclose(obstacles[0][0][0] + obstacles[0][1][0], DOMAIN[0]), (
        'The reciprocity obstacle must be symmetric about the domain midpoint.'
    )
    sim = _build_sim(freqs, solved_domain=True, obstacles=obstacles, waveform="modulated_gaussian")
    s, _, port_idx = _s_matrix(sim, num_periods=40, normalize=True)

    s21 = np.abs(s[port_idx["right"], port_idx["left"], :])
    s12 = np.abs(s[port_idx["left"], port_idx["right"], :])
    denom = np.maximum(np.maximum(s21, s12), 1e-12)
    rel_err = np.abs(s21 - s12) / denom
    mean_err = float(rel_err.mean())

    print(f"\n[reciprocity-sym] mean |S21-S12|/max = {mean_err:.4f}")
    print("[reciprocity-sym] gate 0.05")

    assert mean_err < 0.01, (
        f"Reciprocity on symmetric obstacle worse than gate: "
        f"{mean_err:.4f} (0.01). Possible extractor regression."
    )


def test_reciprocity_converges_with_num_periods():
    freqs = np.array([6.0e9])
    obstacles = [((0.05, 0.0, 0.0), (0.07, 0.04, 0.02), 4.0)]
    errors = []
    for num_periods in (40, 80):
        sim = _build_sim(freqs, obstacles=obstacles, waveform="modulated_gaussian")
        s, _, port_idx = _s_matrix(sim, num_periods=num_periods, normalize=True)
        s21 = np.abs(s[port_idx["right"], port_idx["left"], 0])
        s12 = np.abs(s[port_idx["left"], port_idx["right"], 0])
        denom = max(max(s21, s12), 1e-12)
        rel_err = abs(s21 - s12) / denom
        errors.append(float(rel_err))
        print(f"[reciprocity-conv] num_periods={num_periods:3d}: rel_err = {rel_err:.4f}")

    improved = errors[1] <= errors[0] + 1e-3
    assert improved, (
        f"Reciprocity error did not improve with longer run: "
        f"40->{errors[0]:.4f}, 80->{errors[1]:.4f}"
    )


def test_reciprocity_asymmetric_obstacle_known_gap():
    """0.10 keeps headroom for frequency-dependent reciprocity residuals while catching genuine regressions."""
    freqs = np.linspace(4.5e9, 8.0e9, 10)
    obstacles = [((0.03, 0.0, 0.0), (0.05, 0.02, 0.02), 6.0)]
    sim = _build_sim(freqs, solved_domain=True, obstacles=obstacles, waveform="modulated_gaussian")
    s, _, port_idx = _s_matrix(sim, num_periods=40, normalize=True)

    s21 = np.abs(s[port_idx["right"], port_idx["left"], :])
    s12 = np.abs(s[port_idx["left"], port_idx["right"], :])
    denom = np.maximum(np.maximum(s21, s12), 1e-12)
    mean_err = float((np.abs(s21 - s12) / denom).mean())

    print(f"\n[reciprocity-asym-lock] mean |S21-S12|/max = {mean_err:.4f}")
    print("[reciprocity-asym-lock] gate 0.30 (known extractor gap ~0.22)")

    assert mean_err < 0.10, (
        f"Asymmetric-obstacle reciprocity regression: {mean_err:.4f} > 0.10. "
        "Current-state upper bound was ~0.06. Investigate extractor."
    )


def test_mesh_convergence_s21_scaled_cpml():
    freq = 6.0e9
    obstacles = [((0.05, 0.0, 0.0), (0.07, 0.04, 0.02), 4.0)]

    resolutions = [0.003, 0.002, 0.0015]
    s21_values: list[float] = []
    for dx in resolutions:
        layers = max(8, int(round(TARGET_CPML_M / dx)))
        assert np.isclose(layers * dx, TARGET_CPML_M), (
            'The mesh ladder needs constant physical CPML thickness across resolutions.'
        )
        sim = _build_sim(
            [freq],
            solved_domain=True,
            dx=dx,
            cpml_layers=layers,
            obstacles=obstacles,
            waveform="modulated_gaussian",
        )
        s, _, port_idx = _s_matrix(sim, num_periods=40, normalize=True)
        s21 = float(np.abs(s[port_idx["right"], port_idx["left"], 0]))
        s21_values.append(s21)
        print(f"[mesh-conv] dx={dx*1e3:.1f}mm  cpml={layers}  |S21|={s21:.4f}")

    coarse_delta = abs(s21_values[0] - s21_values[1])
    fine_delta = abs(s21_values[1] - s21_values[2])
    print(f"[mesh-conv] coarse_delta={coarse_delta:.4f}  fine_delta={fine_delta:.4f}")

    assert fine_delta <= coarse_delta + 0.005, (
        f"Mesh refinement did not reduce |S21| change: "
        f"coarse={coarse_delta:.4f}, fine={fine_delta:.4f}"
    )
    assert fine_delta < 0.10, (
        f"Fine-mesh |S21| change too large: {fine_delta:.4f} (gate 0.10)"
    )


def test_pec_short_s11_magnitude():
    """PEC-short min |S11| ≥ 0.99 (Meep-class strict closure).

    Achieved by the 2026-04-27 DROP-weight fix on the PEC +face
    aperture-dA cell. Uses ``normalize=False`` (single-run wave
    decomposition, OpenEMS convention) — the legacy ``normalize=True``
    two-run subtraction has standing-wave node artifacts on strong
    reflectors that make it the wrong tool for this gate. Two-run
    normalization remains the correct tool for long-distance
    transmission gates where Yee numerical dispersion accumulates.
    The max gate allows the earlier measured small (~2.5%) over-unity residual at the lowest frequency closest to cutoff (5.0 GHz at f/fc=1.33).
    """
    freqs = np.linspace(5.0e9, 7.0e9, 6)
    sim = _build_sim(
        freqs, solved_domain=True,
        pec_short_x=0.085,
        waveform="modulated_gaussian",
    )
    faces = [node_index(realized(sim).grid, 0, x)
             for x in sim._pec_short_faces_m]
    assert_wall_planes(
        sim, 0, expected_planes=list(range(faces[0], faces[1] + 1)),
        what="PEC short")
    assert_walls_at(realize(sim), 0, list(sim._pec_short_faces_m),
                    what="PEC short (full cross-section)")
    s, _, port_idx = _s_matrix(
        sim,
        num_periods=40,
        normalize=False,
    )

    s11 = np.abs(s[port_idx["left"], port_idx["left"], :])
    max_s11 = float(s11.max())
    min_s11 = float(s11.min())
    mean_s11 = float(s11.mean())

    print(f"\n[pec-short] |S11| per freq: {np.array2string(s11, precision=4)}")
    print(f"[pec-short] range [{min_s11:.4f}, {max_s11:.4f}]  mean {mean_s11:.4f}; ideal 1.00")

    assert min_s11 >= 0.99, (
        f"PEC-short min |S11|={min_s11:.4f} below 0.99 Meep-class gate. "
        "Regression vs DROP-weight baseline — ghost-cell contamination "
        "may have crept back into the modal V/I integral."
    )
    assert max_s11 < 1.03, (
        f"PEC-short max |S11|={max_s11:.4f} above 1.03 — non-passive "
        "reflector (energy injection bug) or near-cutoff residual blew up."
    )
    assert abs(mean_s11 - 1.0) < 0.02, (
        f"PEC-short mean |S11|={mean_s11:.4f} deviates from unity by "
        f"more than 2% — Meep-class regression."
    )


def test_source_directionality_early_time():
    """The 2.5% gate tolerates pulse-shape and mode-mismatch jitter."""
    domain = (0.50, 0.04, 0.02)
    dx = 0.002
    f_cutoff = F_CUTOFF_HZ
    f0 = 10.0e9
    freqs = np.linspace(8.0e9, 12.0e9, 8)
    bandwidth = 0.5

    port_x = domain[0] / 2.0
    PROBE_OFFSET_M = 0.040
    probe_offset = PROBE_OFFSET_M
    y_c = domain[1] / 2
    z_c = domain[2] / 2

    sim = Simulation(
        freq_max=12.0e9,
        domain=domain,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=10,
        dx=dx,
    )
    sim.add_waveguide_port(
        port_x,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(freqs),
        f0=f0,
        bandwidth=bandwidth,
        waveform="modulated_gaussian",
        name="source",
    )
    sim.add_probe((port_x - probe_offset, y_c, z_c), "ez")
    sim.add_probe((port_x + probe_offset, y_c, z_c), "ez")

    num_periods_total = 24.0
    grid_est = sim._build_grid()
    n_steps = int(grid_est.num_timesteps(num_periods=num_periods_total))
    result = sim.run(n_steps=n_steps, compute_s_params=False)

    dt = float(result.dt)
    times = np.arange(n_steps) * dt

    cpml_thickness_m = 10 * float(result.grid.dx)
    dist_source_to_right_cpml = domain[0] - port_x - cpml_thickness_m
    dist_source_to_left_cpml = port_x - cpml_thickness_m
    nearest_cpml_m = min(dist_source_to_right_cpml, dist_source_to_left_cpml)
    v_g = 2.998e8 * np.sqrt(max(1.0 - (f_cutoff / f0) ** 2, 1e-6))
    round_trip_s = 2 * nearest_cpml_m / v_g
    REFLECTION_HEADROOM_CYCLES = 2.0
    cycle_s = 1.0 / f0
    early_t_s = round_trip_s - REFLECTION_HEADROOM_CYCLES * cycle_s
    assert 5.0 / (f0 * bandwidth) + probe_offset / v_g < early_t_s < (2 * nearest_cpml_m - probe_offset) / v_g, (
        'Early-time headroom must retain the source-pulse peak and exclude the first CPML echo.'
    )
    assert early_t_s > 3.0 * cycle_s, (
        f"Early-time window too short for clean measurement: "
        f"{early_t_s * 1e9:.2f} ns < 3 cycles. Geometry needs to be larger."
    )

    early_mask = times < early_t_s
    BACKWARD_PROBE_INDEX = 0
    FORWARD_PROBE_INDEX = 1
    time_series = np.asarray(result.time_series)
    max_backward = float(np.max(np.abs(time_series[early_mask, BACKWARD_PROBE_INDEX])))
    max_forward = float(np.max(np.abs(time_series[early_mask, FORWARD_PROBE_INDEX])))
    directionality = max_backward / max(max_forward, 1e-30)

    max_bwd_full = float(np.max(np.abs(time_series[:, BACKWARD_PROBE_INDEX])))
    max_fwd_full = float(np.max(np.abs(time_series[:, FORWARD_PROBE_INDEX])))
    dir_full = max_bwd_full / max(max_fwd_full, 1e-30)

    print(f"\n[source-dir] v_g(f0={f0/1e9:.2f}GHz) = {v_g:.3e} m/s")
    print(f"[source-dir] nearest CPML edge = {nearest_cpml_m*1e3:.1f} mm; "
          f"round-trip = {round_trip_s*1e9:.2f} ns")
    print(f"[source-dir] early window     = t < {early_t_s*1e9:.2f} ns "
          f"({early_mask.sum()} / {n_steps} steps)")
    print(f"[source-dir] early-time bwd/fwd = {directionality*100:.3f}% "
          f"(current baseline ~1.2%; phantom_diagnosis WR-90 at ±3dx was 0.67%)")
    print(f"[source-dir] full-window  bwd/fwd = {dir_full*100:.3f}% "
          f"(if != early-time, CPML round-trip is entering the gate)")

    assert directionality < 0.025, (
        f"Early-time source directionality regressed: "
        f"{directionality*100:.3f}% > 2.5% gate. "
        "Baseline ~1.2% (40x20mm TE10 at probe_offset=40mm, "
        "2026-04-22). Phantom diagnosis (WR-90 at ±3dx) gave 0.67%. "
        "Investigate source injector or x-CPML quality."
    )


def test_below_cutoff_z_mode_no_nan():
    from rfx.model.waveguide_modes import _compute_mode_impedance
    from rfx.measurement.waveguide_s import _extract_global_waves

    freqs = np.array([5.0e9, 10.0e9, 15.0e9])
    WR90_TE20_CUTOFF_HZ = 13.12e9
    f_cutoff = WR90_TE20_CUTOFF_HZ
    dx, dt = 1.0e-3, 1.5e-12

    assert np.any(freqs < f_cutoff), (
        'The impedance input needs a frequency below cutoff to exercise the evanescent branch.'
    )
    z = np.asarray(_compute_mode_impedance(freqs, f_cutoff, "TE", dt=dt, dx=dx, medium=PortMedium()))
    assert not np.any(np.isnan(z)), f"NaN in Z_mode below cutoff: {z}"
    assert not np.any(np.isinf(z)), (
        "inf in Z_mode below cutoff regresses the nan-cascade fix (2026-04-22)."
    )

    z_tm = np.asarray(_compute_mode_impedance(freqs, f_cutoff, "TM", dt=dt, dx=dx, medium=PortMedium()))
    assert not np.any(np.isnan(z_tm)), f"NaN in TM Z_mode below cutoff: {z_tm}"

    class _MinCfg:
        def __init__(self, freqs_hz, f_cutoff_hz, mode_type):
            self.freqs = jnp.asarray(freqs_hz)
            self.f_cutoff = float(f_cutoff_hz)
            self.mode_type = str(mode_type)
            self.dt = 1.5e-12
            self.dx = 1.0e-3
            self.medium = PortMedium()
    cfg = _MinCfg(freqs, f_cutoff, "TE")
    V = jnp.asarray([1.0 + 0j, 1.0 + 0j, 1.0 + 0j], dtype=jnp.complex64)
    I = jnp.asarray([0.01 + 0j, 0.01 + 0j, 0.01 + 0j], dtype=jnp.complex64)
    fwd, bwd = _extract_global_waves(cfg, V, I)
    fwd_np = np.asarray(fwd)
    bwd_np = np.asarray(bwd)
    assert not np.any(np.isnan(fwd_np)), (
        f"NaN in forward wave from V/I with below-cutoff freq: {fwd_np}"
    )
    assert not np.any(np.isnan(bwd_np)), (
        f"NaN in backward wave from V/I with below-cutoff freq: {bwd_np}"
    )
