"""AD status for ``Simulation.compute_coaxial_two_port`` (issue #489, active
dev track leg 3).

``tests/test_ad_surface_contract.py`` already classifies this method
``NOT_TRACEABLE`` at the prose level. This file pins that classification
EMPIRICALLY (per this repo's own "the load-bearing coaxial case is
verified empirically, not by prose" contract stated in
``test_ad_surface_contract.py``'s own module docstring) and documents
exactly which op breaks the tape, with a file:line citation, so a future
session that wires AD through this method has a concrete starting point
instead of re-diagnosing the block from scratch.

WHAT WAS TRIED (mirroring the #559/MSL AD pattern per issue #489's own
instructions): the plan was a shared band-mean ``|S21|**2`` objective
(``tests/_coax_ad_objective.py``, mirrors ``tests/_msl_ad_objective.py``
exactly), an AD smoke asserting ``|grad| > floor`` with a severed-tape
falsifier, and — if that worked — a converged AD-vs-FD measurement, same
as ``tests/test_msl_sparam_ad.py``.

WHY IT STOPS AT THE SMOKE STAGE, TWO INDEPENDENT REASONS:

1. **No traced-input channel exists at the API surface at all.**
   ``Simulation.compute_coaxial_two_port``'s signature (``rfx/api/
   _sparams.py``, ``def compute_coaxial_two_port(self, *, n_steps=...,
   freqs=..., n_freqs=..., field_scale=..., cpml_axes=..., probe_count=...,
   probe_start_cells=..., probe_spacing_cells=..., feed_impedance=...,
   cond_warn=..., strict_passivity=...)``) has no ``eps_scale``/
   ``eps_override``-shaped parameter — unlike its 1-port sibling
   ``compute_coaxial_line_reflection(..., eps_scale=...)``, which is
   GRAD_SAFE precisely because that parameter exists and threads a jnp
   array into the FDTD materials. There is nothing to call ``jax.grad``
   with respect to yet — see
   ``test_compute_coaxial_two_port_has_no_traced_input_channel`` below.

2. **Even with a channel, the post-FDTD extraction concretizes before the
   two-drive solve.** ``compute_coaxial_two_port`` calls
   ``rfx.sources.coaxial_port.coaxial_line_plane_voltage`` (NOT the
   already-existing differentiable twin ``coaxial_line_plane_voltage_jnp``
   the 1-port ``eps_scale`` path uses) to turn each probe plane's DFT
   accumulator into a modal voltage. That function's body converts its
   inputs unconditionally::

       ex = np.asarray(ex_dft, dtype=np.complex128)
       ey = np.asarray(ey_dft, dtype=np.complex128)

   (``rfx/sources/coaxial_port.py:1499-1500``, inside
   ``coaxial_line_plane_voltage``, ``def`` at line 1479). Under
   ``jax.grad`` this raises ``jax.errors.TracerArrayConversionError`` — a
   traced array cannot be materialized to a concrete ``numpy.ndarray`` —
   empirically reproduced by
   ``test_coaxial_line_plane_voltage_severs_the_jax_tape`` below. The
   post-FDTD pipeline downstream of this call
   (``_assemble_coaxial_two_port_from_voltages`` ->
   ``coaxial_line_reflection_from_plane_voltages`` ->
   ``solve_two_port_from_wave_amplitudes``, all in
   ``rfx/api/_sparams.py`` / ``rfx/sources/coaxial_port.py``) is ALSO
   concrete ``np.asarray(..., dtype=np.complex128)`` throughout, so even a
   hypothetical fix at ``coaxial_line_plane_voltage`` alone would not be
   sufficient — this is the first break, not the only one.

CONCLUSION (per issue #489's own instruction: "if AD is structurally
blocked, that is a RESULT... skip the gate, do NOT force a workaround"):
no AD smoke or AD-vs-FD gate is added in this pass. ``tests/
_coax_ad_objective.py``'s reduction is committed unused, ready for the
session that wires a traced-input channel through. Filed as the concrete
de-experimental blocker on issue #489 (see the PR body / issue comment for
the filed text) — the two fixes needed are (a) add an ``eps_scale``-shaped
traced-input parameter to ``compute_coaxial_two_port`` mirroring the
1-port method's own design, and (b) route the two-port path through
``coaxial_line_plane_voltage_jnp`` (already differentiable, already used
by the 1-port path) instead of the concrete ``coaxial_line_plane_voltage``,
then re-derive the two-drive solve (``solve_two_port_from_wave_
amplitudes``) in jnp as well, since it currently also concretizes.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from tests._coax_ad_objective import coax_band_mean_s21_sq


def test_coax_band_mean_s21_sq_reduction_is_correct():
    """Unit-check the shared reduction on a synthetic S-matrix: correctness
    is independent of whether any AD gate exercises it yet."""
    n_f = 3
    s21 = np.array([0.9 + 0.1j, 0.7 - 0.2j, 0.5 + 0.0j])
    S = np.zeros((2, 2, n_f), dtype=np.complex128)
    S[1, 0, :] = s21
    got = float(coax_band_mean_s21_sq(jnp.asarray(S)))
    want = float(np.mean(np.abs(s21) ** 2))
    # float32-precision tolerance: jnp.asarray on a complex128 array
    # downcasts to complex64 by default (no x64 enabled in this test).
    assert abs(got - want) < 1e-6, (got, want)


def test_compute_coaxial_two_port_has_no_traced_input_channel():
    """Regression guard for AD-blocker reason (1): if a future change adds
    a traced-input parameter (``eps_scale``/``eps_override``-shaped) to
    ``compute_coaxial_two_port``, this test must be updated in the SAME
    change (and the AD smoke this file currently skips should be added) —
    it fails loud instead of letting the classification in
    ``tests/test_ad_surface_contract.py`` go stale silently.
    """
    sig = inspect.signature(Simulation.compute_coaxial_two_port)
    # Matches the naming convention used by every OTHER traced-input
    # parameter in this codebase (compute_coaxial_line_reflection's
    # eps_scale, compute_waveguide_s_matrix's eps_override/sigma_override):
    # a name starting with "eps_"/"sigma_", or an explicit "eps_scale"-style
    # design-variable name. Deliberately narrow (field_scale, a plain float
    # source-amplitude knob, must NOT match).
    traced_input_names = {
        name for name in sig.parameters
        if name.lower().startswith(("eps_", "sigma_")) or name.lower() == "eps_scale"
    }
    assert not traced_input_names, (
        f"compute_coaxial_two_port() now has apparent traced-input "
        f"parameter(s) {traced_input_names!r} -- update tests/"
        f"test_coax_two_port_ad.py and tests/test_ad_surface_contract.py's "
        f"NOT_TRACEABLE classification, and add the AD smoke this module's "
        f"docstring describes."
    )


def test_coaxial_line_plane_voltage_severs_the_jax_tape():
    """Empirical pin for AD-blocker reason (2): ``coaxial_line_plane_
    voltage`` (``rfx/sources/coaxial_port.py``, ``np.asarray(ex_dft, ...)``
    / ``np.asarray(ey_dft, ...)`` at lines 1499-1500) cannot be
    differentiated through — a JAX tracer cannot be materialized to a
    concrete ``numpy.ndarray``. This is the function
    ``compute_coaxial_two_port`` actually calls (NOT the already-existing
    differentiable twin ``coaxial_line_plane_voltage_jnp``, which the
    GRAD_SAFE 1-port ``compute_coaxial_line_reflection(..., eps_scale=...)``
    path uses instead).

    SEVERED-TAPE FALSIFIER: this test asserts the CURRENT broken state
    (``jax.grad`` raises). If a future change makes
    ``coaxial_line_plane_voltage`` traceable (e.g. by switching its body to
    ``jnp.asarray``), this test goes RED — which is the intended signal
    that AD-blocker reason (2) has cleared and
    ``test_compute_coaxial_two_port_has_no_traced_input_channel``'s
    docstring instructions should be followed.
    """
    from rfx.api import Simulation
    from rfx.sources.coaxial_port import coaxial_line_plane_voltage
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(domain=(0.008, 0.008, 0.020), freq_max=40e9, boundary="cpml")
    sim.add_coaxial_port(
        (0.004, 0.004, 0.010), face="top", pin_length=5.0e-3,
        waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2),
    )
    grid = sim._build_grid()

    def f(scale):
        ex = jnp.ones((grid.nx, grid.ny), dtype=jnp.complex64) * scale
        ey = jnp.zeros((grid.nx, grid.ny), dtype=jnp.complex64)
        v = coaxial_line_plane_voltage(
            grid, ex, ey, center_xy=(0.004, 0.004),
            pin_radius=0.635e-3, outer_radius=2.055e-3,
        )
        return jnp.real(jnp.sum(v))

    # Non-firing control: the CONCRETE (non-traced) call works fine and
    # returns a finite value -- proving the function itself is not simply
    # broken, only non-traceable.
    concrete_val = f(1.0 + 0.0j)
    assert np.isfinite(np.asarray(concrete_val)), (
        "coaxial_line_plane_voltage's concrete path itself is broken -- "
        "fix that before drawing any AD conclusion from it."
    )

    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.grad(f)(1.0 + 0.0j)
