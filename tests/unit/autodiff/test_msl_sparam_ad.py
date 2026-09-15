"""WI-1 regression tests for JAX-native MSL S-matrix assembly (S1 base)."""

from __future__ import annotations

import warnings

from pathlib import Path
from types import MethodType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from rfx.probes.probes import DFTPlaneProbe
from rfx.sources import GaussianPulse
from tests._msl_ad_objective import msl_band_mean_s21_sq
from tests.unit.sparams.test_msl_port_integration import (
    DX,
    EPS_R,
    F_MAX,
    H_SUB,
    L_LINE,
    LX,
    LY,
    LZ,
    PORT_MARGIN,
    W_TRACE,
)

# ---------------------------------------------------------------------------
# Historical S1-base evidence; incomplete for the current H-collocated algorithm.
# Preserve these binaries. Current structural gates use manufactured data below.
# ---------------------------------------------------------------------------

REPLAY_ACCUMULATORS_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "msl_replay_accumulators.npz"
)
REPLAY_GOLDEN_F64_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "msl_replay_golden_f64.npy"
)

# Slow end-to-end golden: a capture of the CURRENT pipeline (re-baselined
# 2026-09-12, #726), regenerable ONLY with a written reason in
# test_compute_msl_s_matrix_end_to_end_matches_historical_base's docstring
# (scripts/capture_msl_e2e_golden.py). The original "pre-change S1 capture,
# must NOT be regenerated" contract this comment used to state was retired
# by that re-baseline; the docstring is the contract.
E2E_GOLDEN_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "msl_s_matrix_golden.npy"
)


def _build_thru_line_sim(*, dx=DX, ly=LY) -> Simulation:
    sim = Simulation(
        freq_max=F_MAX,
        domain=(LX, ly, LZ),
        dx=dx,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml",
            y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (LX, ly, H_SUB)), material="ro4350b")

    y_centre = ly / 2.0
    trace_y_lo = y_centre - W_TRACE / 2.0
    trace_y_hi = y_centre + W_TRACE / 2.0
    # #931 §1.3: this one-cell foil declaration is a SHEET, not a volume.
    # On the current dx = H_SUB/3 mesh its midpoint is at 3.5 cells:
    # the exact half-cell tie resolves LOWER, onto the laminate face (3).
    # The historical capture uses dx = 80 um: midpoint/dx = 3.675, so
    # its nearest node is 4 (320 um). These are different replay geometries.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sim.add_thin_conductor(
            Box((0.0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + dx)),
            sigma_bulk=5.8e7, thickness=35e-6,
        )
    sim.add_msl_port(
        position=(PORT_MARGIN, y_centre, 0.0),
        width=W_TRACE,
        height=H_SUB,
        direction="+x",
        impedance=50.0,
    )
    sim.add_msl_port(
        position=(PORT_MARGIN + L_LINE, y_centre, 0.0),
        width=W_TRACE,
        height=H_SUB,
        direction="-x",
        impedance=50.0,
    )
    return sim


# The live coupon keeps the authorized 254 um laminate, 600 um etched
# trace and 10 mm launch-plane separation. Fixed profiles give a graded Yee
# grid: a scalar 84.667 um mesh cannot align its width. The 50 um CPML
# boundary spacing matches every axis. Two 50 um and four 38.5 um cells
# size the substrate exactly; the air ends in 16 uniform 50 um cells.
# Fixed physical margins survive refinement; 35 um copper is idealized as
# a PEC sheet, not a one-cell metal body. Profiles select the supported NU
# runner, so the old uniform-field replay/AD fixture remains separate.
E2E_SPACING = (50e-6, 50e-6, 50e-6)
E2E_DZ_PROFILE = np.repeat(
    np.asarray([100e-6] + [77e-6] * 14 + [100e-6] * 8) / 2, 2)
E2E_DOMAIN = (14e-3, 3.4e-3, 1.978e-3)
E2E_PERIODS = 12
E2E_REPAIR_ID = "931-msl-254um-600um-aligned-v2"
# The implicit f0=F_MAX/2 pulse falls below the API low-signal screen at
# 4.5/5 GHz. This explicit drive covers the entire declared measurement band.
E2E_WAVEFORM = GaussianPulse(f0=F_MAX, bandwidth=.8)


def _build_aligned_e2e_sim(*, refinement=1) -> Simulation:
    if refinement not in (1, 2):
        raise ValueError("qualification uses the fixed drawing on 1x or 2x mesh")
    spacing = tuple(d / refinement for d in E2E_SPACING)
    profiles = {
        f"d{axis}_profile": np.full(round(length / step), step)
        for axis, length, step in zip("xyz", E2E_DOMAIN, spacing)
    }
    profiles["dz_profile"] = np.repeat(E2E_DZ_PROFILE / refinement, refinement)
    sim = Simulation(
        freq_max=F_MAX, domain=E2E_DOMAIN, dx=min(spacing),
        cpml_layers=16 * refinement,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        **profiles,
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0., 0., 0.), (E2E_DOMAIN[0], E2E_DOMAIN[1], H_SUB)),
            material="ro4350b")
    centre = E2E_DOMAIN[1] / 2
    sim.add_thin_conductor(
        Box((0., centre - W_TRACE / 2, H_SUB),
            (E2E_DOMAIN[0], centre + W_TRACE / 2, H_SUB)),
        sigma_bulk=5.8e7, thickness=35e-6,
    )
    for x, direction in ((PORT_MARGIN, "+x"),
                         (PORT_MARGIN + L_LINE, "-x")):
        sim.add_msl_port(position=(x, centre, 0.), width=W_TRACE,
                         height=H_SUB, direction=direction, impedance=50.,
                         waveform=E2E_WAVEFORM,
                         n_probe_offset=60 * refinement,
                         n_probe_spacing=20 * refinement, n_probes=5)
    return sim


def _build_historical_capture_sim() -> Simulation:
    """Interpret the unchanged PR #516 captures on their original mesh.

    The replay binaries predate #931's on-lattice redraw.
    DX and LY imported from the live integration fixture now describe a
    different grid; using them to index old DFT planes moves both the V span
    and Ampere contour. Freeze the capture's 80 um mesh and lateral extent
    here as historical geometry evidence. The capture lacks the left H
    planes required by #726 and is not a current-algorithm oracle. The
    declared 254 um port height also differs from its realized 320 um foil.
    """
    dx = 80e-6
    return _build_thru_line_sim(
        dx=dx, ly=W_TRACE + 2 * (2 * H_SUB + 8 * dx),
    )


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _make_replay_fake_run(acc_data: dict, run_idx: int):
    """Return a sim.run replacement that replays captured accumulators.

    The real FDTD scanner is NEVER called. The complex64 arrays are retained
    only for historical geometry and missing-data rejection witnesses; no
    missing neighbour is synthesized for the current extraction algorithm.
    """
    def fake_run(self, *, n_steps=None, num_periods=1.0, compute_s_params=False):
        del n_steps, num_periods, compute_s_params
        grid = self._build_grid()
        planes = {}
        for key, arr in acc_data.items():
            if not key.startswith(f"run{run_idx}__"):
                continue
            assert arr.shape[1:] == (grid.ny, grid.nz), (
                f"Replay plane {key} has transverse shape {arr.shape[1:]}, "
                f"but the simulation grid has {(grid.ny, grid.nz)}; "
                "captured fields and geometry must be updated together"
            )
            plane_name = key[len(f"run{run_idx}__"):]
            planes[plane_name] = SimpleNamespace(accumulator=arr)
        return SimpleNamespace(dft_planes=planes)
    return fake_run


def _load_replay_accumulators():
    """Load captured accumulator .npz and return (data_dict, freqs)."""
    with np.load(REPLAY_ACCUMULATORS_PATH) as npz:
        freqs = npz["freqs"]
        data = {k: npz[k] for k in npz.files if k != "freqs"}
    return data, freqs


def _run_manufactured_assembly(*, use_x64: bool):
    """Complete c64 DFT records -> production assembly -> independent S.

    Plant B=S*A for a nonsingular A, then manufacture V=A+B and I=(A-B)/Zref.
    Transverse H ramps have a nonzero, analytically known contour; affine
    longitudinal variation requires both bracketing samples to recover I.
    Neither the production current primitive nor its S solve forms the oracle.
    """
    import re
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

    try:
        from jax import enable_x64
    except ImportError:
        from tests._x64_compat import enable_x64

    u = 2.0**-12
    freqs = np.array([0.8e9, 1.2e9, 1.6e9])
    expected = np.array([[0.2 + 0.05j, 0.55 - 0.07j],
                         [0.65 + 0.1j, -0.15 + 0.03j]])[:, :, None]
    expected = expected * np.array([1.0, 0.9 + 0.1j, 0.8 - 0.15j])
    a = np.array([[1.0 + 0.1j, 0.2 - 0.05j], [-0.1 + 0.04j, 0.9 - 0.1j]])
    assert np.linalg.cond(a) < 2
    b = np.stack([expected[:, :, k] @ a for k in range(len(freqs))], axis=-1)
    zref = float(hammerstad_jensen_z0_eps_eff(4 * u, 2 * u, 2.0)[0])
    voltage, current = a[:, :, None] + b, (a[:, :, None] - b) / zref
    assert np.min(np.abs(current)) > 1e-3
    name_re = re.compile(r"_msl_run(?P<run>\d+)_p(?P<port>\d+)_(?:ez\d+|hy|hz)(?:_left)?")
    seen_h = {}

    def fake_run(self, **kwargs):
        grid = self._build_grid()
        planes = {}
        for entry in self._dft_planes:
            match = name_re.fullmatch(entry.name)
            assert match is not None
            driven, port = int(match["run"]), int(match["port"])
            index = grid.position_to_index((entry.coordinate, 0, 0))[0]
            region = self._dft_plane_regions[entry.name]
            w_lo, w_hi, z_lo, z_hi = region
            shape = (w_hi - w_lo, z_hi - z_lo)
            if entry.component == "ez":
                field = np.ones(shape) / (2 * u)
                amplitude = voltage[port, driven]
            else:
                x_h = (index - grid.pad_x_lo + 0.5) * u
                target = (12 if port == 0 else 28) * u
                seen_h.setdefault((driven, port, entry.component), []).append(x_h)
                factor = 1 + (0.5 + 0.25j) * (x_h - target) / u
                if entry.component == "hy":
                    z_h = np.arange(z_lo, z_hi) - grid.pad_z_lo + 0.5
                    profile = np.broadcast_to(z_h[None, :], shape)
                else:
                    y_h = np.arange(w_lo, w_hi) - grid.pad_y_lo + 0.5
                    profile = np.broadcast_to(0.25 * y_h[:, None], shape)
                sign = 1 if port == 0 else -1
                field = sign * factor * profile / (0.75 * 5 * u)
                amplitude = current[port, driven] * np.exp(-1j * np.pi * freqs * grid.dt)
            accumulator = (jnp.asarray(field, dtype=jnp.complex64)[None, :, :]
                           * jnp.asarray(amplitude, dtype=jnp.complex64)[:, None, None])
            planes[entry.name] = DFTPlaneProbe(
                accumulator=accumulator, freqs=entry.freqs, component=entry.component,
                axis=0, index=index, total_steps=1, window="rect", window_alpha=0.25,
                region=region,
            )
        return SimpleNamespace(dft_planes=planes)

    with enable_x64(use_x64):
        sim = Simulation(freq_max=20e9, domain=(40 * u, 12 * u, 8 * u),
                         dx=u, cpml_layers=2, boundary="cpml")
        sim.add_material("substrate", eps_r=2.0)
        sim.add(Box((0, 0, 0), (40 * u, 12 * u, 2 * u)), material="substrate")
        sim.add(Box((0, 0, 0), (40 * u, 12 * u, 0)), material="pec")
        sim.add(Box((0, 4 * u, 2 * u), (40 * u, 8 * u, 2 * u)), material="pec")
        for feed, direction in ((2 * u, "+x"), (38 * u, "-x")):
            sim.add_msl_port(position=(feed, 6 * u, 0), width=4 * u, height=2 * u,
                             direction=direction, mode="uniform", eps_r_sub=2.0,
                             n_probe_offset=10, n_probe_spacing=2, n_probes=3)
        sim.run = MethodType(fake_run, sim)
        result = sim.compute_msl_s_matrix(
            n_steps=1, freqs=freqs, num_periods=1.0, enforce_passivity=False)
        assert result.S.dtype == (jnp.complex128 if use_x64 else jnp.complex64)
        assert result.assembly == "multi_drive_solve"
        assert np.max(result.cond_a) < 2
        actual = np.asarray(result.S)
    for driven in range(2):
        for port, target in enumerate((12 * u, 28 * u)):
            for component in ("hy", "hz"):
                np.testing.assert_array_equal(sorted(seen_h[driven, port, component]),
                                              [target - u / 2, target + u / 2])
    return actual, expected


# ---------------------------------------------------------------------------
# Test A: manufactured float64 structural equivalence (FAST)
# ---------------------------------------------------------------------------

def test_replay_float64_equivalence():
    """Keep the historical test name and 1e-5 gate, using complete current data.

    The old captures omit the left H samples and remain historical evidence.
    The new oracle is a planted three-frequency, non-diagonal S matrix;
    scoped x64 assembly consumes complete manufactured complex64 DFTs.
    """
    actual, expected = _run_manufactured_assembly(use_x64=True)
    max_abs_dev = float(np.max(np.abs(actual - expected)))
    print(f"\n[manufactured float64 assembly] max_abs_dev = {max_abs_dev:.3e}")
    assert max_abs_dev < 1e-5
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Test B: float32 deployment delta documented and bounded (FAST)
# ---------------------------------------------------------------------------

def test_float32_deployment_delta_bounded():
    """Keep the 1e-3 precision gate and independently check both outputs.

    This bounds assembly arithmetic on manufactured data, not RF error.
    x64 is scoped and both output dtypes are checked before host conversion.
    """
    s_f64, expected = _run_manufactured_assembly(use_x64=True)
    s_f32, expected_f32 = _run_manufactured_assembly(use_x64=False)
    np.testing.assert_array_equal(expected, expected_f32)
    np.testing.assert_allclose(s_f64, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(s_f32, expected, rtol=1e-3, atol=1e-3)
    delta = float(np.max(np.abs(s_f64 - s_f32)))
    print(f"\n[manufactured float32-vs-float64 assembly] max_abs_dev = {delta:.3e}")
    assert delta < 1e-3


def test_historical_capture_cannot_supply_current_bracketing_h_data():
    """Reject the incomplete capture rather than inventing a left H plane."""
    from rfx.api._sparams import _collocated_msl_h

    acc_data, freqs = _load_replay_accumulators()
    assert np.load(REPLAY_GOLDEN_F64_PATH).shape == (2, 2, len(freqs))
    frame = _make_replay_fake_run(acc_data, 0)(_build_historical_capture_sim())
    assert not any(name.endswith("_left") for name in frame.dft_planes)
    pairs = (("_msl_run0_p0_hy_left", "_msl_run0_p0_hy"),
             ("_msl_run0_p0_hz_left", "_msl_run0_p0_hz"))
    assert all(right in frame.dft_planes for _, right in pairs)
    with pytest.raises(ValueError, match="requires bracketing H-plane data"):
        _collocated_msl_h(frame.dft_planes, pairs, (0.5, 0.5))


def test_replay_geometry_matches_the_captured_lattice():
    """Frozen fields need their captured indices, not the live board's DX.

    Independent witnesses: the stored plane dimensions, the realized trace
    wall, and the ground-to-trace voltage / Ampere-loop indexing metadata.
    """
    from rfx.probes.msl_wave_decomp import register_msl_plane_probes
    from tests._realized_geometry import (
        assert_sheet_planes, assert_wall_planes, realized,
    )

    sim = _build_historical_capture_sim()
    grid = sim._build_grid()
    assert (grid.nx, grid.ny, grid.nz) == (192, 54, 31)
    assert_sheet_planes(sim, 2, [320e-6], what="captured MSL trace")
    assert_wall_planes(sim, 2, [320e-6], what="captured MSL trace")
    rz = realized(sim)
    assert rz.pec_mask is None
    assert not np.any(rz.edge_masks[2]), "normal Ez through the sheet stays live"
    acc_data, freqs = _load_replay_accumulators()
    assert {arr.shape for arr in acc_data.values()} == {(len(freqs), 54, 31)}
    probes = register_msl_plane_probes(sim, port_index=0, freqs=freqs)
    assert (probes.j_centre, probes.k_lo, probes.k_hi) == (26, 0, 4)
    assert (probes.j_lo, probes.j_hi, probes.k_trace_lo, probes.k_trace_hi) == (
        22, 30, 4, 4,
    )
    # Fail at the replay boundary, before silently sampling a different mesh.
    with pytest.raises(AssertionError, match="captured fields and geometry"):
        _make_replay_fake_run(acc_data, 0)(_build_thru_line_sim())


# ---------------------------------------------------------------------------
# Test C: AD smoke test (FAST)
# ---------------------------------------------------------------------------

def _synthetic_v_n(alpha: jnp.ndarray, gamma: jnp.ndarray,
                    n_probes: int = 5) -> list:
    """Synthetic voltage probes: V_n = alpha * q^n + gamma * q^(-n)."""
    q = jnp.asarray(0.82 - 0.25j, dtype=jnp.complex64)
    return [alpha * q**n + gamma / q**n for n in range(n_probes)]


def _fake_run_for_theta(theta: jnp.ndarray, n_probes: int = 5):
    """Return a monkeypatched sim.run that injects synthetic phasors.

    Generates n_probes Ez planes (ez0..ez{n-1}) plus both hy/hz neighbours
    per port/run. H is constant along propagation in this tape smoke;
    longitudinal sampling has its own manufactured driver regressions.
    """
    import re
    plane_name_re = re.compile(
        r"_msl_run(?P<run>\d+)_p(?P<port>\d+)_(?P<kind>ez(?P<ez>\d+)|hy|hz)(?:_left)?"
    )

    def fake_run(self, *, n_steps=None, num_periods=1.0, compute_s_params=False):
        del n_steps, num_periods, compute_s_params
        grid = self._build_grid()
        planes = {}
        ones = jnp.ones((1, grid.ny, grid.nz), dtype=jnp.complex64)
        # Linear z/y ramps for the Hy/Hz planes (issue #515 root cause).
        # msl_loop_current's closed Ampere loop is
        #   I = (Hy[bottom] - Hy[top])*dy_sum + (Hz[right] - Hz[left])*dz_sum
        # A spatially UNIFORM Hy/Hz plane (the old ``ones * amp``) makes the
        # opposing legs cancel EXACTLY, so I ~ 0 regardless of theta. With
        # I ~ 0, every a=(V+Z0*I)/2 and b=(V-Z0*I)/2 collapses to a~b~V/2,
        # so the multi-drive S=B*A^-1 solve returns ~Identity independent of
        # V (and therefore of theta) -- empirically confirmed: the old
        # fixture read S ~ [[1,~1e-9],[~1e-9,1]] at every theta in [0.5,0.9].
        # A non-uniform ramp gives Hy/Hz a genuine difference between the
        # loop's opposing legs, so I is non-zero and the V,I -> (a,b)
        # decomposition is no longer degenerate.
        z_ramp = (0.5 + jnp.arange(grid.nz, dtype=jnp.float32) / grid.nz)[None, None, :]
        y_ramp = (0.5 + jnp.arange(grid.ny, dtype=jnp.float32) / grid.ny)[None, :, None]
        for entry in self._dft_planes:
            match = plane_name_re.fullmatch(entry.name)
            assert match is not None, f"Unexpected probe name: {entry.name!r}"
            run_idx = int(match.group("run"))
            port_idx = int(match.group("port"))
            is_driven_port = port_idx == run_idx
            kind = match.group("kind")

            if is_driven_port:
                alpha = jnp.asarray(1.0 + 0.15j, dtype=jnp.complex64)
                gamma = jnp.asarray(0.08 - 0.03j, dtype=jnp.complex64)
            elif run_idx == 0 and port_idx == 1:
                alpha = theta * jnp.asarray(0.42 + 0.07j, dtype=jnp.complex64)
                gamma = jnp.asarray(0.02 + 0.01j, dtype=jnp.complex64)
            else:
                alpha = jnp.asarray(0.35 - 0.04j, dtype=jnp.complex64)
                gamma = jnp.asarray(0.01 - 0.02j, dtype=jnp.complex64)

            if kind.startswith("ez"):
                ez_idx = int(match.group("ez"))
                vs = _synthetic_v_n(alpha, gamma, n_probes=n_probes)
                field = ones * vs[ez_idx]
            elif kind == "hy":
                amp = jnp.asarray(0.018 + 0.004j, dtype=jnp.complex64)
                field = ones * amp * z_ramp
            else:  # hz
                amp = jnp.asarray(0.005 + 0.001j, dtype=jnp.complex64)
                field = ones * amp * y_ramp

            planes[entry.name] = DFTPlaneProbe(
                accumulator=field,
                freqs=entry.freqs,
                component=entry.component,
                axis=0,
                index=0,
                total_steps=1,
                window="rect",
                window_alpha=0.25,
            )
        return SimpleNamespace(dft_planes=planes)

    return fake_run


def test_compute_msl_s_matrix_ad_smoke_has_finite_gradient():
    """AD smoke: jax.grad through V·I assembly must produce finite non-NaN gradient.

    NOTE (M1, 2026-05-24): this test uses SYNTHETIC accumulators (``_fake_run``),
    so the honesty-guard fires a huge-Z0 warning (e.g. ~1e26 ohm) — that is
    EXPECTED for non-physical synthetic data and is NOT a regression. This test
    gates only that the assembly is differentiable; it deliberately does NOT
    assert physical S/Z0 (the data isn't physical). The real physics-sanity gate
    (forward S in [0, ~1.2] + finite-difference cross-check) belongs on the REAL
    end-to-end path and is a required acceptance criterion of G-AD-WIRE
    (docs/research_notes/2026-05-24_next_goals_and_miss_audit.md). Real-path
    passivity is already gated by test_msl_thru_line_passive_gate.

    REBUILT (issue #515, 2026-08-04; CORRECTED 2026-08-04 per adversarial
    review of PR #559 — see "CORRECTION" below). ONE root cause, one
    additional drift-prevention change:

    **Root cause: the synthetic fixture was degenerate.**
    ``_fake_run_for_theta`` used to synthesize Hy/Hz DFT-plane data as a
    spatially UNIFORM field (``ones * amp``). ``msl_loop_current``'s
    closed Ampère loop is
    ``I = (Hy[bottom] − Hy[top])·Σdy + (Hz[right] − Hz[left])·Σdz`` — a
    uniform Hy/Hz plane makes the opposing legs cancel EXACTLY, so the
    synthetic trace current I ≈ 0 at every port, every run, every theta.
    With I ≈ 0, every wave pair ``a = (V + Z0·I)/2``, ``b = (V − Z0·I)/2``
    collapses to ``a ≈ b ≈ V/2``, and the multi-drive ``S = B·A⁻¹`` solve
    then returns ``S ≈ Identity`` REGARDLESS of V — and therefore
    regardless of theta or WHICH objective reads S21. Measured directly:
    the old fixture gave ``S ≈ [[1, ~1e-9], [~1e-9, 1]]`` at every
    theta ∈ [0.5, 0.9]. Fixed by giving Hy a linear z-ramp and Hz a linear
    y-ramp (see the ramps built in ``_fake_run_for_theta``) so the loop's
    opposing legs no longer cancel and I is genuinely non-zero — a
    physically-motivated shape (a real trace current does not thread a
    spatially uniform H field either), not a numerology patch.

    CORRECTION (adversarial review of PR #559): the first version of this
    docstring claimed "two independent defects", the second being that
    ``Re(S21)`` (the old objective) was "structurally flat on current
    main". That is FALSIFIED. Measured directly: NEW (fixed) fixture +
    OLD ``Re(S21)`` objective gives ``grad = -2.973442e-02`` — nonzero,
    and 6.4x LARGER in magnitude than the new objective's
    ``-4.681417e-03``. ``Re(S21)`` was never structurally flat; its
    ``grad = 0.0`` on main was a CONSEQUENCE of the degenerate uniform-
    Hy/Hz fixture above, the same single root cause, not a second,
    independent one. The objective was ADDITIONALLY switched from
    ``Re(S21)`` to the shared ``msl_band_mean_s21_sq``
    (``tests/_msl_ad_objective.py``) purely so this smoke and the #530
    tight gate cannot drift onto two hand-written reductions — that switch
    was not itself required to fix the zero gradient, and is recorded here
    to keep the claim honest.

    Measured after the fixture fix (this test, CPU float32, theta0=0.7):
    ``grad = -4.681417e-03``, ``loss = 6.008485e-03``.

    CROSS-CHECK CORRECTION (adversarial review of PR #559 — BLOCKING
    finding). The first version of this docstring quoted an informal
    central-FD cross-check reading ``g_fd = -2.842303e-03`` and called it
    "agreeing in sign and order of magnitude" — that hid a 65% gap
    (rel_err 0.6471) whose cause the review named: ``compute_msl_s_matrix``
    gates its passivity projection on
    ``enforce_passivity and eps_override is None and not is_tracer(S)``
    (``rfx/api/_sparams.py``, near the ``_project_passive`` call). This
    smoke's ``objective()`` does not pass ``eps_override``, so under
    ``jax.grad`` the traced call has ``is_tracer(S) == True`` and the
    projection is SKIPPED (AD sees the RAW S), while an EAGER call — like
    the informal FD cross-check — has a concrete ``S`` and DOES get
    projected (default ``enforce_passivity=True``). That is the PR #468
    defect class the comment at that call site warns about
    ("a finite-difference objective sees the projected function while
    jax.grad sees the raw one"), recurring here on the non-``eps_override``
    channel: the FD comparator and the AD tape were evaluating two
    DIFFERENT functions.

    Fixed by passing ``enforce_passivity=False`` explicitly in
    ``objective()`` below, so it returns the identical (raw) reduction
    whether called under trace or eagerly — a default-arg-dependent FD
    cross-check is not a valid comparator on this path precisely because
    the two call modes silently take different projection branches; an
    explicit, mode-independent flag is. Re-measured with both sides on
    ``enforce_passivity=False``: ``g_fd = -4.681409e-03`` at h=0.01
    (rel_err 0.0000) and ``g_fd = -4.679663e-03`` at h=0.001 (rel_err
    0.0004) — agreeing with ``grad = -4.681417e-03`` to 4 significant
    figures, not merely in sign and order of magnitude. (Passing
    ``enforce_passivity=False`` does not change the committed ``grad``
    value above: under tracing the projection was already skipped via the
    ``is_tracer`` branch regardless of the flag, so only the previously-
    mismatched EAGER/FD side moves.)

    The floor below (1e-4) sits ~47x under the measured magnitude —
    comfortable headroom for CPU/GPU or JAX-version float32 noise while
    still rejecting a severed tape (which reads exactly 0.0, not a small
    nonzero number).

    FALSIFIER (PR body records the run; measured against THIS test's actual
    code path, i.e. with ``enforce_passivity=False`` in ``objective()``
    below): reproducing the OLD uniform-field fixture with the NEW shared
    objective gives ``grad`` = exactly ``0.0`` at theta = 0.7 and 0.9, and
    ``1.909939e-17`` at theta = 0.5 (float32 noise floor — an order of
    magnitude below a single ULP of a ~1e-3-scale gradient, not a small
    physical signal) — many orders of magnitude under the ``1e-4`` floor —
    confirming this floor rejects the exact severed construction issue
    #515 was filed against.
    """
    freqs = jnp.asarray([1.0e9], dtype=jnp.float32)

    # A strictly-positive floor, not merely isfinite/not-NaN (the #515
    # defect: those assertions cannot fail on a severed tape, so the old
    # test passed on a dead grad = 0.0). Derived from the measured
    # -4.681417e-03 with ~47x headroom below its magnitude.
    _GRAD_FLOOR = 1.0e-4

    def objective(theta):
        sim = _build_thru_line_sim()
        sim.run = MethodType(_fake_run_for_theta(theta), sim)
        result = sim.compute_msl_s_matrix(
            n_steps=1,
            freqs=freqs,
            num_periods=1.0,
            # Explicit, mode-independent — do NOT rely on the default. The
            # default gates passivity projection on
            # `eps_override is None and not is_tracer(S)` (rfx/api/_sparams.py),
            # so a traced call (jax.grad) and an eager call (e.g. an FD
            # cross-check) silently take DIFFERENT branches on this
            # eps_override=None path and would compare two different
            # functions (PR #468 defect class; see this test's docstring,
            # "CROSS-CHECK CORRECTION"). This flag makes objective() return
            # the identical (raw) reduction either way.
            enforce_passivity=False,
        )
        return msl_band_mean_s21_sq(result.S)

    grad = jax.grad(objective)(jnp.asarray(0.7, dtype=jnp.float32))
    print(f"\n[AD smoke] grad = {float(grad):.6e}")
    assert bool(jnp.isfinite(grad)), f"Gradient is not finite: {grad}"
    assert not bool(jnp.isnan(grad)), f"Gradient is NaN: {grad}"
    assert abs(float(grad)) > _GRAD_FLOOR, (
        f"[AD smoke] gradient is effectively zero ({float(grad):.3e}, floor "
        f"{_GRAD_FLOOR:.0e}): the tape may be severed again. isfinite/not-NaN "
        "alone cannot catch this — see issue #515."
    )
    print("[test_compute_msl_s_matrix_ad_smoke_has_finite_gradient] PASS")


# ---------------------------------------------------------------------------
# Test D: physically qualified live coupon and secondary complex drift lock
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_compute_msl_s_matrix_end_to_end_matches_historical_base():
    """End-to-end drift lock: full FDTD + assembly vs the committed golden.

    RE-PINNED 2026-09-12 (#726): the authorized H interpolation puts the
    current on the existing voltage E-node plane. The board, source/load,
    normalization, duration and physical/drift thresholds are unchanged.
    Pinned source ec3bedfe2fc1a8c8be12ebd2b7054562a5fcf096, GPU base
    369367260604, passes the existing raw physical checks but differs from
    the old-algorithm golden by 0.002851. Reconstructing the old same-index
    current on those same fields restores the old golden within 1.784e-5.
    Independent confirmation 369367260608 is bit-identical in raw and
    projected S. Its 2x refinement passes the same physical checks and
    differs by at most 0.007077362 in raw complex S (unchanged budget .02).
    The measured frequency vector is identical on all three records, and
    actual sampled voltage planes remain 5 and 9 mm. Import the base S
    after these checks; retain rtol=.005/atol=.002 below. Current provenance
    is tests/fixtures/msl_s_matrix_golden.json; the prior golden and full
    manifest are preserved under docs/research_notes/issue726/collocation/
    pre-collocation-golden/. The earlier migration records below are history.

    FIXTURE REPAIRED 2026-09-08 (#931): retain the PI-authorized 254 um
    substrate, 600 um width and 10 mm launch-plane separation; align every declared face on
    dx=dy=50 um and dz=(50,50,38.5,38.5,38.5,38.5) um through the
    substrate, declare the foil at its actual interface, and hold domain/absorber thickness fixed under refinement.
    An explicit f0=5 GHz / bandwidth=.8 differentiated Gaussian covers all
    ten measurement bins; the former implicit 2.5 GHz pulse undersupplied
    the final two bins. All four named runs qualify this repaired drive.
    Raw physical qualification precedes the unchanged complex drift gate.

    RE-PINNED 2026-09-09 (#931, T11 acceptance condition met): offline import
    of base run 369367259636 S_projected, complex64 (2,2,10), from exact SHA
    b36fc46cdf21d1c57f221e6a057654bcad60bae2. Confirmation 369367259638 is
    array-identical; long 369367259637 differs by at most 5.331202e-7 projected
    and 2.457562e-7 raw. Refinement 369367259648 halves every cell with fixed
    geometry: all 40 raw entries differ by <=0.007206652 (budget 0.02),
    projected <=0.007206839. Every physical screen passes, both drives settle
    below -103 dB on base/refine, and refinement finishes with 11 pytest passes.
    The old-pin gaps remain 0.2030584601 (base) and 0.1995657504 (refine).
    This qualifies convergence within the declared mesh budget for a changed
    board: the old 80 um lattice / 320 um trace plane was replaced by the
    aligned graded mesh above / 254 um trace sheet. The persistent old-pin
    gap is not numerical integration drift. The final live mesh is NOT the
    intermediate scalar 254/3 um mesh. Historical 80 um replay binaries stay
    frozen. No rfx/ code or physical/drift gate changes accompany this re-pin.
    Full base values, source hashes, named reports/run IDs, per-entry
    comparisons and offline base/confirmation verification are recorded in
    tests/fixtures/msl_s_matrix_golden.json. The transfer uses
    scripts/diagnostics/import_931_msl_golden.py; capture_msl_e2e_golden.py
    --write-golden would run a new solve and was not used for this import.

    RE-BASELINED 2026-07-30 (PR #516; decision required by that PR's review,
    finding F4, and recorded here + in issue #509). The original golden was a
    pre-jnp-conversion S1 capture whose contract said it "must NOT be
    regenerated from post-change code" — that contract existed to prove a pure
    REFACTOR was precision-only. Two deliberate algorithm changes then made
    the premise unsatisfiable:

      * #468 made ``enforce_passivity=True`` the default (projection moves S
        by up to 5.2e-2 on this fixture — the first red, 2026-07-27, #509);
      * #511/#507 (PR #516) corrected the modal-voltage span and replaced the
        single-ratio assembly with the multi-drive solve (measured deviation
        vs the old golden: 1.07e-1 as shipped (an intermediate PR state
        measured 2.93e-1 before review finding F2 re-anchored the V span) —
        not f32 rounding, and not
        meant to be).

    A frozen old-algorithm capture cannot survive deliberate algorithm
    changes, and keeping it red hides real regressions behind an expected
    one. The golden is therefore now a capture of the CURRENT pipeline
    (fixture, freqs, ``num_periods=12``, default ``enforce_passivity`` —
    exactly this test's invocation; capture script recorded in PR #516), and
    this test's job is narrower and honest: catch UNINTENDED cross-version
    drift. Any future red here is a real change to the MSL S-matrix and must
    be either root-caused or re-baselined with its reason written in this
    docstring — never silently.

    Structural assembly is checked separately by test_replay_float64_equivalence
    using complete manufactured bracketing H records and an independent
    planted two-port S matrix. One-sided historical captures cannot test
    the current interpolated algorithm.
    Tolerance stays rtol=5e-3/atol=2e-3: it must absorb cross-machine float
    noise on a CI runner (the PR #119 lesson: tolerances, not bit-equality).
    The #726 correction exceeded that unchanged bound at reflection entries;
    the lock distinguished the deliberate algorithm change from roundoff.

    MEASURED, NOT RE-BASELINED (2026-09-02, closing the #803 remainder).
    After the exact-coordinate rasterization fix (#802/#807, PR #834) this
    fixture realizes the substrate box at 25900 cells (x nodes 0..174) and
    the trace at 1225, identical at both x64 flags (``_build_thru_line_sim``,
    no solve, measured here); the pre-#834 base 06cf29f0 realized 26048
    (x nodes 0..175) / 1232 at x64=0 (measured by the #803 remainder
    assessment) — one x node plane fewer on each, the x-hi face. Re-running
    scripts/capture_msl_e2e_golden.py on main b5605391 at JAX_ENABLE_X64=0
    gave old-vs-new max|dS| = 1.2307e-4 (mean|S11| 0.0852 -> 0.0851,
    mean|S21| 0.9954 unchanged), 16x inside atol, so the golden was left as
    captured by PR #516. The lock's remaining headroom against that S is
    2e-3 - 1.2e-4; the next deliberate re-baseline should quote this number
    as its starting point.
    """
    from tests._msl_fixture_qualification import qualify_result

    golden = np.load(E2E_GOLDEN_PATH)
    sim = _build_aligned_e2e_sim()
    _assert_trace_sheet_realized(sim)
    _assert_e2e_preflight(sim)
    freqs = jnp.linspace(F_MAX / 10, F_MAX, golden.shape[-1], dtype=jnp.float32)

    result = sim.compute_msl_s_matrix(freqs=freqs, num_periods=E2E_PERIODS)
    qualification = qualify_result(result)
    s_actual = np.asarray(result.S).astype(np.complex128)

    print("\n[WI-1 MSL end-to-end full S trace]")
    for idx, freq in enumerate(np.asarray(result.freqs)):
        vals = s_actual[:, :, idx]
        print(
            f"f[{idx:02d}]={freq:.9e} "
            f"S00={vals[0, 0].real:+.12e}{vals[0, 0].imag:+.12e}j "
            f"S01={vals[0, 1].real:+.12e}{vals[0, 1].imag:+.12e}j "
            f"S10={vals[1, 0].real:+.12e}{vals[1, 0].imag:+.12e}j "
            f"S11={vals[1, 1].real:+.12e}{vals[1, 1].imag:+.12e}j"
        )

    raw = s_actual if result.S_raw is None else np.asarray(result.S_raw)
    print("[WI-1 MSL raw full S, matrix axes port/drive/frequency]", raw.tolist())
    print("[WI-1 MSL raw qualification]", qualification)
    assert not qualification["failures"], qualification["failures"]
    max_dev = float(np.max(np.abs(s_actual - golden)))
    print(f"[WI-1 MSL] max_abs_dev(aligned live coupon vs committed drift golden) = {max_dev:.3e}")
    # Unchanged drift tolerance against the qualified repaired-board base.
    np.testing.assert_allclose(s_actual, golden, rtol=5e-3, atol=2e-3)


def _assert_trace_sheet_realized(sim_sim):
    """Check the drawing, sheet footprint and dielectric interface, without FDTD."""
    from tests._realized_geometry import realized, _node_line, node_index

    rz = realized(sim_sim)
    assert rz.pec_mask is None, "a sheet owns no cell"
    assert len(rz.sheets) == 1
    assert not np.any(rz.edge_masks[2]), "normal Ez through the sheet stays live"
    lines = [_node_line(rz.grid, axis) for axis in range(3)]
    sheet = rz.sheets[0]
    assert int(sheet.normal_axis) == 2
    np.testing.assert_allclose(lines[2][sheet.plane], H_SUB, atol=2e-9, rtol=0)
    footprint = np.argwhere(np.asarray(sheet.footprint))
    expected = ((0., E2E_DOMAIN[0]), (1.4e-3, 2.0e-3), (H_SUB, H_SUB))
    for axis, bounds in enumerate(expected):
        actual = lines[axis][[footprint[:, axis].min(), footprint[:, axis].max()]]
        np.testing.assert_allclose(actual, bounds, atol=2e-9, rtol=0)
    for axis, points in enumerate(((0., PORT_MARGIN, PORT_MARGIN + L_LINE,
                                    E2E_DOMAIN[0]),
                                   (0., 1.4e-3, 1.7e-3, 2.0e-3, E2E_DOMAIN[1]),
                                   (0., H_SUB, E2E_DOMAIN[2]))):
        for point in points:
            np.testing.assert_allclose(lines[axis][node_index(rz.grid, axis, point)],
                                       point, atol=2e-9, rtol=0)
    mats = sim_sim._assemble_materials_nu(rz.grid, pec_sheets=[], pec_wires=[])[0]
    i = node_index(rz.grid, 0, PORT_MARGIN)
    j = node_index(rz.grid, 1, E2E_DOMAIN[1] / 2)
    k0 = node_index(rz.grid, 2, 0.)
    k1 = node_index(rz.grid, 2, H_SUB)
    np.testing.assert_allclose(np.asarray(mats.eps_r)[i, j, k0:k1], EPS_R)
    np.testing.assert_allclose(np.asarray(mats.eps_r)[i, j, k1], 1.)
    from rfx.sources.msl_port import msl_port_from_entry, msl_probe_x_coords_n
    from tests._msl_fixture_qualification import REFERENCE_PLANES_M
    expected_planes = ([5e-3, 6e-3, 7e-3, 8e-3, 9e-3],
                       [9e-3, 8e-3, 7e-3, 6e-3, 5e-3])
    np.testing.assert_allclose(REFERENCE_PLANES_M,
                               [planes[0] for planes in expected_planes], atol=1e-12, rtol=0)
    for entry, expected_x in zip(sim_sim._msl_ports, expected_planes):
        actual_x = msl_probe_x_coords_n(
            rz.grid, msl_port_from_entry(entry), n_probes=entry.n_probes,
            n_offset_cells=entry.n_probe_offset,
            n_spacing_cells=entry.n_probe_spacing)
        np.testing.assert_allclose(actual_x, expected_x, rtol=0, atol=2e-9)
    return rz


def _assert_e2e_preflight(sim):
    """Prevent known invalid geometry/absorber setups before an expensive run.

    Lossless-material Q advice does not condemn a transmission coupon. The
    boundary-touching dielectric can produce an overlap warning at machine
    roundoff; the exact domain/material-node checks above own that boundary.
    All preflight messages remain recorded, including those benign notices.
    """
    report = sim.preflight()
    failures = [issue.to_dict() for issue in report
                if issue.severity == "error" or issue.code in
                {"nu_grading_reaches_absorber", "msl_port_geometry",
                 "nonuniform_cpml_thin"}]
    assert not failures, failures
    return report


@pytest.mark.parametrize("refinement", [1, 2])
def test_migrated_trace_is_a_sheet_on_the_declared_plane(refinement):
    sim = _build_aligned_e2e_sim(refinement=refinement)
    _assert_trace_sheet_realized(sim)
    _assert_e2e_preflight(sim)


def _ideal_qualification_result():
    """An exactly matched lossless line is the physical limiting case."""
    from tests._msl_fixture_qualification import quasi_static_line
    freqs = np.linspace(.5e9, 5e9, 10)
    z0, beta, transmission = quasi_static_line(freqs)
    s = np.zeros((2, 2, len(freqs)), dtype=complex)
    s[0, 1] = s[1, 0] = transmission
    return SimpleNamespace(
        freqs=freqs, S=s.copy(), S_raw=s.copy(),
        Z0=np.full((2, len(freqs)), z0, dtype=complex), beta=beta,
        assembly="multi_drive_solve", reliable=np.ones((2, len(freqs)), bool),
        settling_db=np.array([-60., -60.]), cond_a=np.ones(len(freqs)),
        beta_railed=np.zeros((2, len(freqs)), bool),
    )


def test_aligned_coupon_qualification_accepts_matched_line_limit():
    from tests._msl_fixture_qualification import qualify_result
    assert not qualify_result(_ideal_qualification_result())["failures"]


@pytest.mark.parametrize("defect, expected", [
    ("phase_conjugation", "electrical-length phase"),
    ("zero_transmission", "transmission misses"),
    ("excess_reflection", "reflection exceeds"),
    ("projection_hides_gain", "singular value exceeds"),
    ("unsettled", "ringdown exceeds"),
    ("nonfinite", "non-finite raw S"),
])
def test_aligned_coupon_qualification_rejects_wrong_observables(defect, expected):
    from tests._msl_fixture_qualification import qualify_result
    result = _ideal_qualification_result()
    if defect == "phase_conjugation":
        result.S_raw = result.S_raw.conj()
    elif defect == "zero_transmission":
        result.S_raw[0, 1] = result.S_raw[1, 0] = 0.
    elif defect == "excess_reflection":
        result.S_raw[0, 0] = .5
    elif defect == "projection_hides_gain":
        result.S_raw *= 2.
    elif defect == "unsettled":
        result.settling_db[1] = -20.
    elif defect == "nonfinite":
        result.S_raw[0, 0, 7] = np.nan
    failures = qualify_result(result)["failures"]
    assert any(expected in failure for failure in failures), failures


def test_aligned_coupon_drive_covers_the_declared_measurement_band():
    """The actual sampled drive must clear the fixed low-signal screen.

    This checks excitation before a solve, not the resulting S-matrix. It
    cannot certify output reliability; that remains a live qualification.
    """
    sim = _build_aligned_e2e_sim()
    grid = sim._build_nonuniform_grid()
    times = np.arange(sim._nu_n_steps(E2E_PERIODS)) * grid.dt
    freqs = np.linspace(F_MAX / 10, F_MAX, 10)
    transform = np.exp(-2j * np.pi * freqs[:, None] * times[None, :])
    for entry in sim._msl_ports:
        assert entry.waveform is E2E_WAVEFORM
        spectrum = np.abs(transform @ np.asarray(entry.waveform(times)))
        # Predeclared 2.5x margin over the existing 10% reliability floor.
        assert np.min(spectrum / np.median(spectrum)) > .25
    old = np.abs(transform @ np.asarray(GaussianPulse(f0=F_MAX / 2, bandwidth=.8)(times)))
    np.testing.assert_array_equal(old / np.median(old) < .1,
                                   [False] * 8 + [True, True])


def test_qualification_checks_the_float32_rounded_band_endpoint():
    from tests._msl_fixture_qualification import qualify_result
    result = _ideal_qualification_result()
    result.freqs[8] = 4_500_000_256.  # actual float32 linspace endpoint
    result.S_raw[0, 0, 8] = .3
    report = qualify_result(result)
    assert report["physics_band_indices"] == [5, 6, 7, 8]
    assert any("reflection exceeds" in failure for failure in report["failures"])
