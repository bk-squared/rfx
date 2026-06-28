"""API-level contract tests for the (2,4) fourth-order-in-space stencil
(``stencil_order=`` on ``Simulation`` / threaded through ``run`` / ``forward``).

PR-1a added the kernel-level option (``stencil_order=2|4`` on
``rfx.core.yee.update_e`` / ``update_h``); the kernel contract + bit-identity
tests live in ``tests/test_fourth_order_stencil.py``. PR-1b (this file) wires
that option through the ``Simulation`` API, derates the CFL for order=4, and
FENCES order=4 against every unsupported path.

Two absolute contracts a reviewer checks adversarially:

1. ``stencil_order=2`` (the default) is BYTE-IDENTICAL end-to-end — threading
   the new parameter must not perturb the existing default at any call site
   (``test_order2_default_byte_identical``).
2. ``stencil_order=4`` is REACHABLE ONLY on the plain uniform-Cartesian
   vacuum/dielectric path (pec/periodic boundary, default solver, no
   dispersion/anisotropy/conformal/Kerr, uniform mesh, single device); every
   other configuration raises ``NotImplementedError`` rather than silently
   running 2nd order (``test_order4_fenced``).

These tests deliberately do NOT enable jax x64 — a process-global flip would
turn every pytest-split shard red (see feedback_jax_x64_module_level_tests).
"""
import numpy as np
import pytest

from rfx import Simulation


def _uniform_sim(stencil_order, *, boundary="pec", **kw):
    """Small uniform-Cartesian PEC sim with a single soft point source."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.016, 0.016, 0.016),
        dx=0.001,
        boundary=boundary,
        stencil_order=stencil_order,
        **kw,
    )
    sim.add_source((0.008, 0.008, 0.008), component="ez")
    return sim


def test_order2_default_byte_identical():
    """The default (no stencil_order arg) is BIT-IDENTICAL to an explicit
    stencil_order=2 — threading the new param perturbs nothing on the default
    path."""
    sim_default = Simulation(
        freq_max=10e9, domain=(0.016, 0.016, 0.016), dx=0.001, boundary="pec",
    )
    sim_default.add_source((0.008, 0.008, 0.008), component="ez")
    r_default = sim_default.run(n_steps=40, skip_preflight=True)

    r_explicit2 = _uniform_sim(2).run(n_steps=40, skip_preflight=True)

    for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
        a = np.array(getattr(r_default.state, comp))
        b = np.array(getattr(r_explicit2.state, comp))
        assert np.array_equal(a, b), f"default vs order=2 differ in {comp}"


def test_order4_runs_on_supported_path():
    """stencil_order=4 runs on the supported uniform PEC path, produces finite
    fields, and differs from order=2 (proves order=4 is actually applied, not
    silently downgraded)."""
    r2 = _uniform_sim(2).run(n_steps=40, skip_preflight=True)
    r4 = _uniform_sim(4).run(n_steps=40, skip_preflight=True)

    ez4 = np.array(r4.state.ez)
    assert np.all(np.isfinite(ez4)), "order=4 produced non-finite fields"
    assert float(np.max(np.abs(ez4))) > 0.0, "order=4 fields are all zero"

    ez2 = np.array(r2.state.ez)
    assert not np.array_equal(ez2, ez4), (
        "order=4 result is byte-identical to order=2 — the 4th-order stencil "
        "was NOT applied (silent downgrade)."
    )


def test_order4_runs_on_periodic_path():
    """stencil_order=4 is also reachable on a periodic-boundary uniform sim."""
    from rfx.boundaries.spec import BoundarySpec

    spec = BoundarySpec(x="periodic", y="periodic", z="pec")
    sim = Simulation(
        freq_max=10e9, domain=(0.016, 0.016, 0.016), dx=0.001,
        boundary=spec, stencil_order=4,
    )
    sim.add_source((0.008, 0.008, 0.008), component="ez")
    r4 = sim.run(n_steps=40, skip_preflight=True)
    assert np.all(np.isfinite(np.array(r4.state.ez)))


def test_order4_fenced_cpml():
    """stencil_order=4 with an absorbing (cpml) boundary raises."""
    sim = _uniform_sim(4, boundary="cpml")
    with pytest.raises(NotImplementedError, match="stencil_order=4"):
        sim.run(n_steps=10, skip_preflight=True)


def test_order4_fenced_nonuniform():
    """stencil_order=4 on a non-uniform (graded) mesh raises."""
    dz = np.full(20, 0.001)
    sim = Simulation(
        freq_max=10e9, domain=(0.016, 0.016, 0.02), dx=0.001,
        dz_profile=dz, boundary="pec", stencil_order=4,
    )
    sim.add_source((0.008, 0.008, 0.01), component="ez")
    with pytest.raises(NotImplementedError, match="stencil_order=4"):
        sim.run(n_steps=10, skip_preflight=True)


def test_order4_fenced_adi():
    """stencil_order=4 with the ADI solver raises."""
    sim = Simulation(
        freq_max=10e9, domain=(0.016, 0.016, 0.016), dx=0.001,
        boundary="pec", solver="adi", mode="2d_tmz", stencil_order=4,
    )
    sim.add_source((0.008, 0.008, 0.0), component="ez")
    with pytest.raises(NotImplementedError, match="stencil_order=4"):
        sim.run(n_steps=10, skip_preflight=True)


def test_order4_fenced_dispersive():
    """stencil_order=4 with a Debye dispersive material raises."""
    from rfx.geometry.csg import Box
    from rfx.materials.debye import DebyePole

    sim = Simulation(
        freq_max=5e9, domain=(0.02, 0.02, 0.02), dx=0.001,
        boundary="pec", stencil_order=4,
    )
    sim.add_material(
        "disp", eps_r=4.0, debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)],
    )
    sim.add(Box((0, 0, 0), (0.02, 0.02, 0.02)), material="disp")
    sim.add_source((0.005, 0.005, 0.01), component="ez")
    with pytest.raises(NotImplementedError, match="stencil_order=4"):
        sim.run(n_steps=10, skip_preflight=True)


def test_order4_cfl_derated():
    """The timestep used for order=4 is ~0.857x the order=2 dt (the (2,4) CFL
    bound). Inspect the dt resolved inside ``_build_step_setup`` directly."""
    from rfx.grid import Grid
    from rfx.core.yee import init_materials
    from rfx.simulation import _build_step_setup, _ORDER4_CFL_FACTOR

    grid = Grid(freq_max=10e9, domain=(0.016, 0.016, 0.016), dx=0.001)
    mats = init_materials(grid.shape)
    common = dict(
        boundary="pec", cpml_axes="", pec_axes="xyz", periodic=None,
        debye=None, lorentz=None, tfsf=None, sources=[], probes=[],
        dft_planes=[], flux_monitors=[], waveguide_ports=[], ntff=None,
        aniso_eps=None, aniso_inv_eps=None, aniso_inv_eps_smooth=False,
        pec_mask=None, pec_occupancy=None, conformal_weights=None,
        wire_port_sparams=[], lumped_port_sparams=[], lumped_rlc=[],
        kerr_chi3=None, field_dtype=None, mag_sources=[],
    )
    setup2 = _build_step_setup(grid, mats, stencil_order=2, **common)
    setup4 = _build_step_setup(grid, mats, stencil_order=4, **common)

    assert setup2.dt == grid.dt, "order=2 dt must equal the undertated grid.dt"
    assert setup4.dt == pytest.approx(grid.dt * _ORDER4_CFL_FACTOR, rel=1e-9)
    assert setup4.dt < setup2.dt, "order=4 dt must be derated below order=2"


def test_invalid_order_raises():
    """stencil_order other than 2 or 4 raises at construction time."""
    with pytest.raises(ValueError, match="stencil_order must be 2 or 4"):
        Simulation(
            freq_max=10e9, domain=(0.016, 0.016, 0.016), dx=0.001,
            boundary="pec", stencil_order=3,
        )
