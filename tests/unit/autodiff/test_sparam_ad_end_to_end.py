"""G-AD-WIRE: End-to-end autodiff through S-parameter extractors.

G-AD-WIRE (2026-05-25) connected jax.grad end-to-end through both
``compute_msl_s_matrix`` and ``compute_waveguide_s_matrix`` by:

MSL fix:
  1. Added ``eps_override`` kwarg to ``compute_msl_s_matrix``.  When set,
     the method calls ``self.forward(eps_override=...)`` instead of
     ``self.run()``, so the FDTD scan is differentiable.
  2. Replaced ``hy_plane = np.asarray(...)`` / ``hz_plane = np.asarray(...)``
     with ``jnp.asarray`` so field-plane reads stay on the JAX tape.

Waveguide fix (G-AD-WIRE, now upgraded to G-AD-WIRE-WG2):
  1. Replaced the ``np.zeros`` + ``np.array`` item-assign accumulator in
     ``extract_waveguide_s_matrix`` (rfx/sources/waveguide_port.py) with
     a functional ``jnp.stack`` build so ``b_recv / safe_a`` stays on the
     JAX tape.
  2. Added public ``eps_override`` / ``sigma_override`` kwargs to
     ``compute_waveguide_s_matrix`` (rfx/api/_sparams.py) so jax.grad
     flows end-to-end through the PUBLIC API without internal monkeypatching.
     NU path excluded (uses run_nonuniform_path, different material channel).

Both paths have finite/nonzero-gradient and bounded-output smoke checks.
The waveguide test also performs FD comparisons at two epsilon points.
The MSL accuracy gate is test_msl_ad_fd_converged.py; the short MSL record
here checks tape connectivity and a fixture-specific gradient sign.

Positive controls (forward-only differentiable path) are preserved below.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from rfx.simulation import _suggest_checkpoint_segments


RO4350B_RELATIVE_PERMITTIVITY = 3.66
_MSL_EPS_R = RO4350B_RELATIVE_PERMITTIVITY
SUBSTRATE_THICKNESS_M = 254e-6
_MSL_H_SUB = SUBSTRATE_THICKNESS_M
TRACE_WIDTH_M = 600e-6
_MSL_W_TRACE = TRACE_WIDTH_M
_MSL_DX = _MSL_H_SUB / 3
PROBE_LINE_LENGTH_M = 6e-3
_MSL_L_LINE = PROBE_LINE_LENGTH_M
PROBE_MARGIN_M = 2e-3
_MSL_PORT_MARGIN = PROBE_MARGIN_M
_MSL_F_MAX = 5e9


def _build_msl_sim() -> Simulation:
    """Tiny MSL thru-line sim (2 ports, minimal domain).
    The ground and trace lie on the bounding nodes of three substrate cells.
    """
    lx = _MSL_L_LINE + 2 * _MSL_PORT_MARGIN
    ly = _MSL_W_TRACE + 2 * (2 * _MSL_H_SUB + 8 * _MSL_DX)
    AIR_LAYER_THICKNESS_M = 0.5e-3
    lz = _MSL_H_SUB + AIR_LAYER_THICKNESS_M

    sim = Simulation(snap="declared",
        freq_max=_MSL_F_MAX,
        domain=(lx, ly, lz),
        dx=_MSL_DX,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )

    sim.add_material("ro4350b", eps_r=_MSL_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _MSL_H_SUB)), material="ro4350b")

    y_centre = ly / 2.0
    trace_y_lo = y_centre - _MSL_W_TRACE / 2.0
    trace_y_hi = y_centre + _MSL_W_TRACE / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sim.add_thin_conductor(
            Box((0.0, trace_y_lo, _MSL_H_SUB), (lx, trace_y_hi, _MSL_H_SUB)),
            sigma_bulk=5.8e7, thickness=35e-6,
        )

    sim.add_msl_port(
        position=(_MSL_PORT_MARGIN, y_centre, 0.0),
        width=_MSL_W_TRACE,
        height=_MSL_H_SUB,
        direction="+x",
        impedance=50.0,
    )
    sim.add_msl_port(
        position=(_MSL_PORT_MARGIN + _MSL_L_LINE, y_centre, 0.0),
        width=_MSL_W_TRACE,
        height=_MSL_H_SUB,
        direction="-x",
        impedance=50.0,
    )
    return sim


WR90_BROAD_WALL_M = 22.86e-3
_WR90_A = WR90_BROAD_WALL_M
WR90_NARROW_WALL_M = 10.16e-3
_WR90_B = WR90_NARROW_WALL_M
WAVEGUIDE_CELL_SIZE_M = 2e-3
_WR90_DX = WAVEGUIDE_CELL_SIZE_M
WAVEGUIDE_DOMAIN_LENGTH_M = 0.05
_WR90_LX = WAVEGUIDE_DOMAIN_LENGTH_M


_WR90_FREQS = jnp.linspace(8e9, 12e9, 8)


def _build_wg_sim(*, n_ports: int = 2) -> Simulation:
    """Tiny 2-port WR-90 sim (minimal domain, CPU-fast, above-cutoff freqs).
    The waveguide domain lengths (_WR90_LX, 0.024, 0.012) are whole numbers of cells of _WR90_DX, so the solved domain equals the declared one.
    """
    assert np.all(np.asarray(_WR90_FREQS) > 299792458.0 / (2 * _WR90_A)), (
        "waveguide fixture frequencies must be above the WR-90 TE10 cutoff"
    )
    sim = Simulation(
        freq_max=12e9,
        domain=(_WR90_LX, 0.024, 0.012),
        dx=_WR90_DX,
        boundary=BoundarySpec(x="cpml", y="pec", z="pec"),
        cpml_layers=8,
    )
    sim.add_waveguide_port(
        direction="+x",
        x_position=0.01,
        y_range=(0.0, 0.024),
        z_range=(0.0, 0.012),
        n_modes=1,
        freqs=_WR90_FREQS,
    )
    if n_ports == 2:
        sim.add_waveguide_port(
            direction="-x",
            x_position=_WR90_LX - 0.01,
            y_range=(0.0, 0.024),
            z_range=(0.0, 0.012),
            n_modes=1,
            freqs=_WR90_FREQS,
        )
    return sim


def test_msl_s_matrix_ad_end_to_end():
    """G-AD-WIRE M1: jax.grad flows end-to-end through compute_msl_s_matrix.

    Injects a differentiable scalar ``alpha`` via ``eps_override`` (added by
    G-AD-WIRE), runs a real (tiny) FDTD forward, assembles the MSL S-matrix,
    and checks:
    1. Gradient is finite and non-zero.
    2. Forward |S| remains within the retained smoke envelope [0, 1.2].
    3. The finite-record gradient retains its measured sign. This is not
       a monotonicity law or an FD-accuracy qualification.
    4. WI-1 replay golden is unaffected (tested separately).
    An earlier measurement with segmented checkpointing capped the peak to a few GB.
    """
    sim = _build_msl_sim()
    grid = sim._build_realized_grid()
    num_periods = 3
    n_steps = grid.num_timesteps(num_periods=num_periods)
    checkpoint_segments = _suggest_checkpoint_segments(n_steps)
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)

    def objective(alpha: jnp.ndarray) -> jnp.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.compute_msl_s_matrix(
                n_freqs=8,
                n_steps=n_steps,
                num_periods=num_periods,
                eps_override=eps_base * alpha,
                checkpoint_segments=checkpoint_segments,
            )
        S = result.S
        k0 = S.shape[-1] // 2
        return jnp.real(jnp.sum(jnp.abs(S[:, :, k0]) ** 2))

    alpha0 = jnp.float32(1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd_result = sim.compute_msl_s_matrix(
            n_freqs=8,
            n_steps=n_steps,
            num_periods=num_periods,
            eps_override=eps_base * alpha0,
        )
    S_fwd = np.asarray(fwd_result.S)
    s_max = float(np.max(np.abs(S_fwd)))
    assert s_max <= 1.2, (
        f"Forward |S| = {s_max:.4f} exceeds the retained 1.2 smoke envelope. "
        "Check the MSL forward path or geometry."
    )
    assert s_max > 0.0, "Forward |S| = 0 everywhere — likely a broken forward pass."

    loss_val, g = jax.value_and_grad(objective)(alpha0)

    assert jnp.isfinite(g), f"AD gradient is not finite: {g}"
    assert float(jnp.abs(g)) > 1e-10, (
        f"AD gradient is effectively zero ({g:.3e}): tape is still broken. "
        "Run jax.make_jaxpr(objective)(alpha0) and grep for eps_r primitive."
    )

    g_ad = float(g)
    print(f"\n[G-AD-WIRE MSL] loss={float(loss_val):.6e}  |S|_max={s_max:.4f}")
    print(f"  AD grad = {g_ad:.6e}")
    assert g_ad < 0, (
        f"MSL AD gradient sign unexpected: {g_ad:.4e}. "
        "This is the observed sign of a fixed finite-record fixture, not a "
        "law that increasing epsilon increases dissipation. Check an independent "
        "finite difference before changing the sign assertion."
    )
    print("[test_msl_s_matrix_ad_end_to_end] PASS")


def test_waveguide_s_matrix_ad_end_to_end():
    """G-AD-WIRE-WG2 M1: jax.grad flows through PUBLIC compute_waveguide_s_matrix(eps_override=...).

    G-AD-WIRE-WG2 adds a public ``eps_override`` kwarg to
    ``compute_waveguide_s_matrix`` (rfx/api/_sparams.py) so that
    jax.grad can flow end-to-end without monkeypatching internals.

    The earlier fix (G-AD-WIRE) jnp-ified the accumulator in
    ``extract_waveguide_s_matrix`` so b_recv/safe_a stays on the JAX tape.
    This test exercises the PUBLIC channel end-to-end and checks:
    1. Gradient is finite and non-zero.
    2. Forward |S| is physically sane ([0, 1.2]).
    3. Finite-difference cross-check (sign + magnitude, 2 eps points).

    NU path: excluded from eps_override scope (uses run_nonuniform_path,
    a different material-injection channel). Only the uniform lane is tested.
    Both vacuum modal segments stay fixed when differentiating the mid-guide slab.
    """
    sim = _build_wg_sim()
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)
    lo, hi = (grid.index_of("x", x) for x in (.020, .030))
    design = jnp.zeros(grid.shape, dtype=jnp.float32).at[lo:hi].set(1.)

    def objective(alpha: jnp.ndarray) -> jnp.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.compute_waveguide_s_matrix(
                n_steps=200,
                normalize=False,
                eps_override=eps_base + design * (alpha - 1.),
            )
        S = result.s_params
        k0 = S.shape[-1] // 2
        return jnp.real(jnp.sum(jnp.abs(S[:, :, k0]) ** 2))

    alpha0 = jnp.float32(1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd_result = sim.compute_waveguide_s_matrix(
            n_steps=200,
            normalize=False,
            eps_override=eps_base + design * (alpha0 - 1.),
        )
    S_fwd = np.asarray(fwd_result.s_params)
    s_max = float(np.max(np.abs(S_fwd)))
    assert s_max <= 1.2, (
        f"Forward |S| = {s_max:.4f} exceeds 1.2 — physically implausible."
    )
    assert s_max > 0.0, "Forward |S| = 0 everywhere — likely a broken forward pass."

    loss_val, g = jax.value_and_grad(objective)(alpha0)

    assert jnp.isfinite(g), f"AD gradient is not finite: {g}"
    assert float(jnp.abs(g)) > 1e-10, (
        f"AD gradient is effectively zero ({g:.3e}): eps_override not reaching the scan. "
        "Run jax.make_jaxpr(objective)(alpha0) and grep for eps_r primitive."
    )

    h = 1e-3
    f_plus = float(objective(jnp.float32(alpha0 + h)))
    f_minus = float(objective(jnp.float32(alpha0 - h)))
    g_fd = (f_plus - f_minus) / (2.0 * h)
    g_ad = float(g)
    print(f"\n[G-AD-WIRE-WG2] loss={float(loss_val):.6e}  |S|_max={s_max:.4f}")
    print(f"  AD grad = {g_ad:.6e}")
    print(f"  FD grad = {g_fd:.6e}  (h={h})")
    rel_err = abs(g_ad - g_fd) / (abs(g_fd) + 1e-12)
    print(f"  rel_err(AD vs FD) = {rel_err:.3e}")
    assert rel_err < 0.05, (
        f"AD vs FD gradient mismatch: AD={g_ad:.4e} FD={g_fd:.4e} "
        f"rel_err={rel_err:.3e} (threshold 5%)."
    )
    assert g_ad * g_fd > 0, (
        f"AD and FD gradients have opposite signs: AD={g_ad:.4e} FD={g_fd:.4e}"
    )
    print("[test_waveguide_s_matrix_ad_end_to_end] PASS")


def test_forward_eps_override_is_differentiable_msl():
    """Positive control: jax.grad through forward(eps_override=) + probe loss.

    This does NOT go through compute_msl_s_matrix — it uses the validated
    forward() path to confirm that the non-S-matrix differentiable path
    still works on the MSL geometry.
    Gradient may be zero on a trivial geometry, so finite is the minimal bar.
    """
    PROBE_CONTROL_CELL_SIZE_M = 80e-6
    sim = Simulation(
        freq_max=_MSL_F_MAX,
        domain=(6e-3, 3e-3, 2e-3),
        dx=PROBE_CONTROL_CELL_SIZE_M,
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
        cpml_layers=4,
    )
    sim.add_port((3e-3, 1.5e-3, 1e-3), "ez", impedance=50.0)
    sim.add_probe((3e-3, 1.5e-3, 1e-3), "ez")

    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)

    def loss(alpha):
        r = sim.forward(eps_override=eps_base * alpha, n_steps=20)
        return jnp.sum(jnp.abs(r.time_series) ** 2)

    alpha0 = jnp.float32(1.0)
    val = float(loss(alpha0))
    assert np.isfinite(val), f"forward() loss is not finite: {val}"

    g = float(jax.grad(loss)(alpha0))
    assert np.isfinite(g), f"forward() gradient is not finite: {g}"


def test_forward_eps_override_is_differentiable_waveguide():
    """Positive control: jax.grad through forward(eps_override=) on WR-90 sim.

    Mirrors tests/unit/autodiff/test_waveguide_forward.py::test_forward_with_waveguide_port_is_differentiable.
    Confirms the forward() tape is intact on the waveguide geometry used in
    the xfail test above.
    forward() refuses two waveguide ports in one record (#2449), and this control needs one.
    """
    sim = _build_wg_sim(n_ports=1)
    sim.add_probe(position=(_WR90_LX / 2, _WR90_A / 2, _WR90_B / 2), component="ey")
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)

    def loss(alpha):
        r = sim.forward(eps_override=eps_base * alpha, n_steps=20)
        return jnp.sum(jnp.abs(r.time_series) ** 2)

    alpha0 = jnp.float32(1.0)
    val = float(loss(alpha0))
    assert np.isfinite(val), f"forward() loss is not finite: {val}"

    g = float(jax.grad(loss)(alpha0))
    assert np.isfinite(g), f"forward() gradient is not finite: {g}"


def _assert_trace_sheet_realized(sim_sim):
    """Build-time check (no solve): the migrated trace realizes on the node
    plane shared by the laminate and both ports, and owns no cell (#931).

    Every migrated conductor on this branch owes this assertion; the shared
    spelling is tests/_realized_geometry.py, so a fixture never re-derives
    the rule it is checking.
    """
    from tests._realized_geometry import assert_sheet_planes, realized
    rz = realized(sim_sim)
    assert rz.pec_mask is None, "a sheet owns no cell"
    assert len(rz.sheets) == 1
    for port in sim_sim._msl_ports:
        assert port.position[2]+port.height == _MSL_H_SUB
    return assert_sheet_planes(sim_sim, 2, [_MSL_H_SUB], what="MSL trace")


def test_migrated_trace_is_a_sheet_on_the_declared_plane():
    _assert_trace_sheet_realized(_build_msl_sim())
