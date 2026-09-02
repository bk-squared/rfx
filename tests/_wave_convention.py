"""tests/_wave_convention.py

The ONE planting helper for probe-ladder modal voltages, shared by every
test that feeds a planted field to an ``rfx.api._sparams`` wave-amplitude
assembler (issue #822).

Why this module exists
----------------------
Both assemblers' pre-existing planted fixtures built ``V(axis)`` by
INVERTING the extractor's own extrapolation: they asked
``coaxial_line_reflection_from_plane_voltages`` which exponential branch it
would call ``forward_amp`` (via its ``load_below`` rule) and planted the
incident wave onto the branch the assembler's own constant would read as
incident. Those fixtures are label ROUND-TRIPS: helper and assembler encode
the same constant, so the pair stays green for whatever constant is chosen
as long as both are chosen together. (Measured 2026-09-01 on a hard-linked
probe tree: flipping ONLY the two-port assembler's constant while leaving
the retired helper's ``load_below`` literals alone does red its planted
test, max|S - S_true| = 0.688/1.19/1.37 -- so "passes under either
assignment", as issue #822 phrases it, holds for the COUPLED change, not
for a one-sided one.)

What such a fixture can never detect is the defect that actually shipped:
that the constant is wrong FOR THE GEOMETRY. Geometry never enters the
planting, so when ``_assemble_coax_msl_transition_from_voltages`` copied
both the helper and the two-port lane's constant onto a lane whose
reference planes sit on the OTHER side of the probes, the two halves stayed
consistent with each other while both were backwards about the physics --
``S_code = inv(S_true)`` behind a green planted test (issue #822).

The contract frozen here instead: plant from GEOMETRY only. The incident
wave travels TOWARD the DUT, the outgoing wave AWAY from it, and which
axial direction that is comes from ``dut_sign`` -- a statement about where
the DUT sits relative to the ladder, checkable against the fixture's own
picture and never against the extractor's labels.

Precedent for a shared tests/ module: ``tests/_gate_policy.py`` (issue
#528), imported as ``from tests._gate_policy import ...`` by
``tests/unit/sparams/test_coax_msl_transition.py`` among others. Keeping ONE copy is
what makes "the convention is stated once" checkable; three byte-identical
copies would be a convention nothing enforces.

Consumers:
  * tests/unit/sparams/test_coax_msl_transition_wave_roles.py  (#822 gate, both ports)
  * tests/unit/sparams/test_coax_two_port_fdtd.py              (planted two-port fixtures)
  * tests/unit/sparams/test_coax_msl_transition.py             (planted unequal-Z0 fixture)
"""
from __future__ import annotations

import numpy as np

__all__ = ["plant_ladder_voltages_physical"]


def plant_ladder_voltages_physical(a, b, *, gamma, planes_m, ref_m, dut_sign):
    """Plant ``V(axis)`` at ``planes_m`` from GEOMETRY, never from labels.

    Parameters
    ----------
    a, b : (n_freqs,) complex
        Wave amplitudes AT the reference plane: ``a`` = the wave travelling
        toward the DUT (incident on the network), ``b`` = the wave
        travelling away from it (outgoing). Units are the caller's -- pass
        ``sqrt(Z0) * a`` to plant modal VOLTAGES for a power-wave ``a``.
    gamma : complex or (n_freqs,) complex
        Propagation constant ``alpha + 1j*beta`` (may differ per frequency,
        which de-degenerates the fixture against a per-frequency indexing
        bug in an assembly loop).
    planes_m : (n_planes,) float
        Probe-plane positions on the ladder's own axis (metres).
    ref_m : float
        The reference plane on that same axis.
    dut_sign : float
        ``+1`` when the DUT lies at LARGER coordinate than the ladder,
        ``-1`` when smaller. This is the only place the geometry enters,
        and it is a fact about the fixture's picture -- assert it against
        the fixture's realized plane positions rather than writing it as a
        bare literal.

    Returns
    -------
    (n_planes, n_freqs) complex
        ``V(s) = a exp(-dut_sign*gamma*s) + b exp(+dut_sign*gamma*s)`` with
        ``s = plane - ref``. With the repo's ``exp(-j 2 pi f t)`` DFT kernel
        (``rfx/probes/probes.py``) a wave travelling in ``+axis`` carries
        ``exp(-gamma s)``, so for ``dut_sign=+1`` (DUT above) the incident
        wave travels ``+axis`` and DECAYS as it approaches the DUT, and for
        ``dut_sign=-1`` it travels ``-axis``. At ``s = 0`` this reduces to
        ``V(ref) = a + b`` for either sign.
    """
    s = np.asarray(planes_m, dtype=np.float64) - float(ref_m)
    a = np.atleast_1d(np.asarray(a, dtype=np.complex128))
    b = np.atleast_1d(np.asarray(b, dtype=np.complex128))
    g = np.broadcast_to(np.asarray(gamma, dtype=np.complex128), a.shape)
    sg = float(dut_sign)
    if sg not in (1.0, -1.0):
        raise ValueError(f"dut_sign must be +1 or -1, got {dut_sign}")
    return (
        np.exp(-sg * np.multiply.outer(s, g)) * a[None, :]
        + np.exp(+sg * np.multiply.outer(s, g)) * b[None, :]
    )
