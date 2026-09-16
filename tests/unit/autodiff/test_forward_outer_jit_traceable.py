"""forward()/optimize() must be wrappable in an OUTER jax.jit.

Regression for the blind-docs finding: ``_assemble_materials`` decided whether to
return the PEC mask via ``bool(jnp.any(pec_mask))`` -- a host-side boolean
conversion on a geometry-derived device array. That is fine eagerly (and under a
bare ``jax.grad``, where materials stay concrete), but when the whole
``forward()`` is wrapped in an *outer* ``jax.jit`` the geometry-derived
``pec_mask`` becomes a tracer and the ``bool(...)`` raised
``TracerBoolConversionError`` deep in material assembly -- so a user JITing their
optimization step crashed with an opaque error.

Fix: the eager path keeps the exact ``jnp.any`` test (bit-identical results,
including the corner where a PEC shape's mask is empty); only under trace, where a
host bool is impossible, does it fall back to a static Python ``has_pec``
predicate. These tests lock the capability (outer-jit works) and the eager
invariant (a PEC shape entirely outside the grid stays a no-op).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx import GaussianPulse, Simulation
from rfx.geometry import Box

from tests._gate_policy import gate_from_envelope


def _pec_sim():
    s = Simulation(freq_max=6e9, domain=(0.04, 0.02, 0.02), boundary="pec")
    s.add_source((0.02, 0.01, 0.01), "ez",
                 waveform=GaussianPulse(f0=3e9, bandwidth=3e9))
    s.add_probe((0.012, 0.008, 0.008), "ez")
    return s


def _reflected_energy_loss(sim):
    def loss(eps):
        r = sim.forward(eps_override=eps, n_steps=60, checkpoint=True,
                        skip_preflight=True)
        ts = r.time_series[:, 0]
        n = ts.shape[0]
        return jnp.sum(ts[n // 2:] ** 2) / (jnp.sum(ts[:n // 2] ** 2) + 1e-30)
    return loss


def test_forward_wrappable_in_outer_jit():
    """jax.jit(loss) must run (previously TracerBoolConversionError) and match eager."""
    sim = _pec_sim()
    shape = sim.run(n_steps=1, skip_preflight=True).grid.shape
    eps = jnp.ones(shape, dtype=jnp.float32) * 1.5
    loss = _reflected_energy_loss(sim)

    eager = float(loss(eps))
    jitted = float(jax.jit(loss)(eps))  # must not raise
    assert np.isfinite(jitted)
    assert abs(eager - jitted) <= 1e-6 * (abs(eager) + 1e-12), (eager, jitted)


def test_grad_wrappable_in_outer_jit():
    """jax.jit(jax.grad(loss)) must run and match the un-jitted gradient."""
    sim = _pec_sim()
    shape = sim.run(n_steps=1, skip_preflight=True).grid.shape
    eps = jnp.ones(shape, dtype=jnp.float32) * 1.5
    loss = _reflected_energy_loss(sim)

    g_plain = jax.grad(loss)(eps)
    g_jit = jax.jit(jax.grad(loss))(eps)  # must not raise
    assert np.all(np.isfinite(np.asarray(g_jit)))
    # float32 gradients through 60 FDTD steps: jit fuses ops differently, so
    # allow the usual XLA-reordering slack (agreement is to ~6 sig figs).
    assert np.allclose(np.asarray(g_plain), np.asarray(g_jit),
                       rtol=3e-4, atol=1e-6)


def test_real_interior_pec_under_outer_jit_matches_eager():
    """The point of the fix on the hot path: a sim with a REAL interior PEC
    obstacle (so ``has_pec_cells`` is True and the trace fallback returns the
    actual mask) must still be outer-jit-wrappable and match eager. `_pec_sim`
    has only a PEC *boundary* (separate grid path), which leaves ``pec_mask``
    empty — this exercises the non-trivial fallback branch."""
    s = Simulation(freq_max=6e9, domain=(0.04, 0.02, 0.02), boundary="pec")
    s.add(Box((0.018, 0.006, 0.006), (0.024, 0.014, 0.014)), material="pec")
    s.add_source((0.008, 0.01, 0.01), "ez",
                 waveform=GaussianPulse(f0=3e9, bandwidth=3e9))
    s.add_probe((0.012, 0.008, 0.008), "ez")
    shape = s.run(n_steps=1, skip_preflight=True).grid.shape
    # confirm this sim really populates pec_mask (the branch under test)
    assert s._assemble_materials(s._build_grid())[3] is not None

    eps = jnp.ones(shape, dtype=jnp.float32) * 1.5
    loss = _reflected_energy_loss(s)
    eager = float(loss(eps))
    jitted = float(jax.jit(loss)(eps))
    assert abs(eager - jitted) <= 1e-5 * (abs(eager) + 1e-12), (eager, jitted)
    g_plain = jax.grad(loss)(eps)
    g_jit = jax.jit(jax.grad(loss))(eps)
    assert np.allclose(np.asarray(g_plain), np.asarray(g_jit), rtol=3e-4, atol=1e-6)


def test_a_pec_shape_that_realizes_nothing_is_an_error_not_a_silent_none():
    """The empty-PEC corner, restated under the lattice ownership contract.

    Before #931 a PEC shape entirely outside the grid rasterized to an
    empty mask and ``_assemble_materials`` returned ``pec_mask=None`` —
    silently, which is the #369 vaporized-metal class. §1.5 makes it a
    refusal: a PEC volume that no primal-cell centre falls inside raises,
    naming ``PolylineWire`` for a filament and the minimum radius for a
    volume. The invariant the old test protected (the empty case does not
    over-approximate into a spurious mask) is kept by the refusal being
    reached at all; what changed is that the user is told.

    The second half is unchanged: a real interior obstacle returns a
    non-empty mask.
    """
    import pytest

    s_empty = Simulation(freq_max=6e9, domain=(0.04, 0.02, 0.02), boundary="pec")
    s_empty.add(Box((10.0, 10.0, 10.0), (11.0, 11.0, 11.0)), material="pec")
    with pytest.raises(ValueError, match="ZERO cells"):
        s_empty._assemble_materials(s_empty._build_grid())

    s_real = Simulation(freq_max=6e9, domain=(0.04, 0.02, 0.02), boundary="pec")
    s_real.add(Box((0.018, 0.006, 0.006), (0.024, 0.014, 0.014)), material="pec")
    grid_r = s_real._build_grid()
    pec_mask_r = s_real._assemble_materials(grid_r)[3]
    assert pec_mask_r is not None and bool(pec_mask_r.any()), \
        "interior PEC obstacle must return a non-empty mask"
    # ... and a run with NO conductor at all still returns None, so "None"
    # keeps meaning "no conductor" rather than "a conductor vanished".
    s_none = Simulation(freq_max=6e9, domain=(0.04, 0.02, 0.02), boundary="pec")
    assert s_none._assemble_materials(s_none._build_grid())[3] is None


# ---------------------------------------------------------------------------
# Issue #1091 — the NTFF and MSL-port stages of the SAME capability.
#
# The two tests above lock ``forward()`` + a time-series objective. They pass
# on a tree where ``compute_far_field`` and any MSL-port forward still die at
# trace time, because neither exercises an NTFF box or an MSL port:
#
#   rfx/farfield.py   ``int(np.asarray(box.freqs).shape[0])`` — a STATIC shape
#                     read through a host materialisation of a traced array
#                     (``box.freqs`` leaves ``forward()`` as a tracer).
#   rfx/sources/msl_port.py  ``np.asarray(m, dtype=bool)`` on the realized PEC
#                     edge masks, which descend from the traced
#                     eps_override / pec_occupancy_override.
#   rfx/api/_execute.py      ``float(np.asarray(_static_eps_483[eps_cell]))``
#                     — the array is concrete, but INDEXING it under an outer
#                     trace stages the read, so the scalar came back traced.
#   rfx/sparams/_common.py   ``float(dz_arr[k])`` in ``msl_modal_voltage``,
#                     same shape one layer out in the MSL extractor.
#
# These two cases cover each stage end to end and compare eager against jit.
# ---------------------------------------------------------------------------

_THETA_1091 = jnp.asarray(np.linspace(1e-3, np.pi - 1e-3, 7))
_PHI_1091 = jnp.asarray([0.0, np.pi / 2])

# Gates for the eager-vs-outer-jit agreement below, derived through the shared
# policy from MEASURED envelopes (CPU, jax 0.10.2, float32, 2026-09-16), swept
# over eps_override in {1.0, 1.3, 2.0, 3.0} on BOTH fixtures:
#
#   worst |loss_jit - loss_eager|                 = 3.576279e-07
#       (both objectives are logs, so an ABSOLUTE difference of the loss IS
#        the relative difference of the quantity inside the log)
#   worst max|g_jit - g_eager| / max|g_eager|     = 1.875460e-05
#
# The residual is XLA reassociation: under jit the whole forward + NTFF /
# extraction chain is one fused module, so the float32 sums are accumulated in
# a different order. Nothing here is a physics claim -- these are 26^3 / 40-step
# traceability locks with no settling witness; they gate CAPABILITY (the stage
# traces at all) and AGREEMENT (jit does not silently change the answer).
_JIT_VALUE_GATE_1091 = gate_from_envelope(3.576279e-07, quantum=1e8)   # 5.4e-07
_JIT_GRAD_GATE_1091 = gate_from_envelope(1.875460e-05, quantum=1e6)    # 2.9e-05


def _ntff_sim():
    """Open (CPML) box with a point dipole and an NTFF surface around it."""
    s = Simulation(freq_max=6e9, domain=(0.03, 0.03, 0.03),
                   boundary="cpml", cpml_layers=6)
    s.add_source((0.015, 0.015, 0.015), "ez",
                 waveform=GaussianPulse(f0=3e9, bandwidth=3e9))
    s.add_ntff_box((0.009, 0.009, 0.009), (0.021, 0.021, 0.021),
                   freqs=jnp.asarray([3e9]))
    return s


def _ntff_pattern_loss(sim, n_steps=48):
    from rfx.farfield import compute_far_field

    def loss(eps):
        r = sim.forward(eps_override=eps, n_steps=n_steps, checkpoint=True,
                        skip_preflight=True)
        ff = compute_far_field(r.ntff_data, r.ntff_box, r.grid,
                               _THETA_1091, _PHI_1091)
        # MAGNITUDES, not |E|^2: rfx's spectral NTFF convention puts
        # |E| ~ 1e-19 here, so squaring in float32 underflows to 0 and the
        # objective would be the +epsilon floor -- a vacuous lock. The 1e20
        # rescale keeps the log argument O(1e2).
        mag = jnp.abs(ff.E_theta[0]) + jnp.abs(ff.E_phi[0])
        return jnp.log(jnp.sum(mag) * 1e20 + 1e-30)
    return loss


def test_ntff_far_field_wrappable_in_outer_jit():
    """forward() -> compute_far_field under an outer jax.jit (issue #1091).

    Before the fix this raised ``TracerArrayConversionError: float32[1]`` at
    trace time, and the traceback pointed at the numpy fallback
    (``compute_far_field``'s dispatcher swallowed the real failure with
    ``except Exception: pass``), not at the direction-chunking budget that
    actually read ``box.freqs`` through numpy.

    Gate: derived from the measured eager-vs-jit envelope through
    ``tests._gate_policy.gate_from_envelope`` (float32 XLA reassociation --
    jit fuses the NTFF surface sum differently from the eager dispatch).
    """
    sim = _ntff_sim()
    shape = sim._build_grid().shape
    eps = jnp.ones(shape, dtype=jnp.float32) * 1.3
    loss = _ntff_pattern_loss(sim)

    eager = float(loss(eps))
    jitted = float(jax.jit(loss)(eps))      # must not raise
    assert np.isfinite(eager) and np.isfinite(jitted), (eager, jitted)
    # Guard the guard: a far field that underflowed to the log's epsilon floor
    # would make the agreement assertion vacuous.
    assert eager > -60.0, f"NTFF objective collapsed to its floor: {eager}"

    assert abs(jitted - eager) <= _JIT_VALUE_GATE_1091, (
        eager, jitted, abs(jitted - eager), _JIT_VALUE_GATE_1091)

    g_plain = np.asarray(jax.grad(loss)(eps))
    g_jit = np.asarray(jax.jit(jax.grad(loss))(eps))   # must not raise
    assert np.all(np.isfinite(g_jit))
    scale = max(float(np.max(np.abs(g_plain))), 1e-30)
    assert float(np.max(np.abs(g_plain - g_jit))) <= _JIT_GRAD_GATE_1091 * scale, (
        float(np.max(np.abs(g_plain - g_jit))), scale, _JIT_GRAD_GATE_1091)


def _msl_sim():
    """Small microstrip thru-line: ground sheet, substrate, trace sheet, two
    MSL ports and one interior probe.

    ``eps_r_sub`` is deliberately NOT given, so the port takes the AUTO
    branch that reads the registered substrate permittivity out of a
    re-assembled material array -- the ``rfx/api/_execute.py`` site above.
    """
    from rfx.api import Simulation as _Sim
    domain_y, y_c = 0.008, 0.004
    s = _Sim(freq_max=20e9, domain=(0.012, domain_y, 0.0032),
             dx=4e-4, boundary="cpml", cpml_layers=6)
    s.add_material("sub", eps_r=2.2)
    s.add(Box((0, 0, 0), (0.012, domain_y, 0.0008)), material="sub")
    s.add(Box((0, 0, 0), (0.012, domain_y, 0)), material="pec")          # ground
    s.add(Box((0.0, y_c - 0.0008, 0.0008),
              (0.012, y_c + 0.0008, 0.0008)), material="pec")            # trace
    s.add_msl_port(position=(0.002, y_c, 0.0), width=0.0016, height=0.0008,
                   direction="+x", impedance=50.0, name="p1")
    s.add_msl_port(position=(0.010, y_c, 0.0), width=0.0016, height=0.0008,
                   direction="-x", impedance=50.0, name="p2")
    s.add_probe((0.006, y_c, 0.0004), "ez")
    return s


def _msl_energy_loss(sim, n_steps=40):
    def loss(eps):
        r = sim.forward(eps_override=eps, n_steps=n_steps, checkpoint=True,
                        skip_preflight=True)
        ts = r.time_series[:, 0]
        return jnp.log(jnp.sum(ts ** 2) + 1e-30)
    return loss


def test_msl_port_forward_wrappable_in_outer_jit():
    """forward() on a simulation carrying MSL ports, under an outer jax.jit.

    Before the fix this raised ``TracerArrayConversionError: bool[...]`` from
    ``validate_msl_port_geometry``, which converts the realized PEC edge masks
    to host booleans to census the conductor surfaces -- impossible on a
    tracer. The conductor census is now DEFERRED under trace, with a warning
    that names what did not run; everything reading only concrete mesh/port
    geometry still runs, and the eager path is untouched.
    """
    sim = _msl_sim()
    shape = sim._build_grid().shape
    eps = jnp.ones(shape, dtype=jnp.float32)
    loss = _msl_energy_loss(sim)

    import warnings
    with warnings.catch_warnings(record=True) as eager_w:
        warnings.simplefilter("always")
        eager = float(loss(eps))
    assert not [w for w in eager_w if "validation SKIPPED" in str(w.message)], \
        "the eager path must still run the full conductor-surface census"

    with warnings.catch_warnings(record=True) as jit_w:
        warnings.simplefilter("always")
        jitted = float(jax.jit(loss)(eps))          # must not raise
    skipped = [w for w in jit_w if "validation SKIPPED" in str(w.message)]
    assert skipped, (
        "a deferred geometry check must say so -- a silent skip is the #303 "
        "class")
    assert "jax tracers" in str(skipped[0].message).lower()

    assert np.isfinite(eager) and np.isfinite(jitted), (eager, jitted)
    assert abs(jitted - eager) <= _JIT_VALUE_GATE_1091, (
        eager, jitted, abs(jitted - eager), _JIT_VALUE_GATE_1091)

    g_plain = np.asarray(jax.grad(loss)(eps))
    g_jit = np.asarray(jax.jit(jax.grad(loss))(eps))   # must not raise
    assert np.all(np.isfinite(g_jit))
    scale = max(float(np.max(np.abs(g_plain))), 1e-30)
    assert float(np.max(np.abs(g_plain - g_jit))) <= _JIT_GRAD_GATE_1091 * scale, (
        float(np.max(np.abs(g_plain - g_jit))), scale, _JIT_GRAD_GATE_1091)


def test_msl_port_geometry_validation_still_refuses_a_bad_port_eagerly():
    """The deferral must not become a hole: with CONCRETE masks the census
    still raises on a port whose declared trace plane has no conductor."""
    import pytest
    from dataclasses import replace
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.sources.msl_port import msl_port_from_entry, validate_msl_port_geometry

    sim = _msl_sim()
    grid = sim._build_grid()
    sheets, wires = [], []
    _, _, _, pec_mask, *_ = sim._assemble_materials(
        grid, pec_sheets=sheets, pec_wires=wires)
    masks = realized_pec_edge_masks(pec_mask, sheets=sheets, wires=wires,
                                    periodic=(False, False, False))
    port = msl_port_from_entry(sim._msl_ports[0])
    validate_msl_port_geometry(grid, port, pec_edge_masks=masks, name="p1")
    with pytest.raises(ValueError, match="no longitudinal conductor"):
        validate_msl_port_geometry(
            grid, replace(port, z_hi=port.z_hi + 2 * 4e-4),
            pec_edge_masks=masks, name="p1")
