"""tests/_coax_ad_objective.py

Shared coax two-port AD-gate objective (issue #489, active dev track leg
3), mirroring ``tests/_msl_ad_objective.py``'s shape exactly (issue #530):
one reduction, imported by every consumer, instead of each test hand-
rolling its own copy that can drift.

STATUS (2026-08-04): this reduction is NOT yet exercised end-to-end by any
AD gate. ``Simulation.compute_coaxial_two_port`` has no traced-input
channel at all (no ``eps_scale``/``eps_override`` parameter — unlike
``compute_coaxial_line_reflection``), and even if one were added, its
internal voltage extraction (``rfx.sources.coaxial_port.
coaxial_line_plane_voltage``) unconditionally concretizes its input via
``np.asarray(..., dtype=np.complex128)`` (``rfx/sources/coaxial_port.py``,
the two lines directly under "concrete path: NumPy (float64), byte-
identical to the pre-AD code" in ``coaxial_line_plane_voltage``'s body),
which raises ``jax.errors.TracerArrayConversionError`` under
``jax.grad`` — empirically confirmed and pinned by
``tests/unit/autodiff/test_coax_two_port_ad.py::
test_coaxial_line_plane_voltage_severs_the_jax_tape``. See that test file
for the full blocker writeup and issue #489's comment thread for the filed
finding.

This module is committed now (rather than deferred until the blocker
clears) so a future session that wires a traced-input channel through
``compute_coaxial_two_port`` reuses this EXACT reduction instead of
inventing a second one that could drift from the MSL gate's own choice —
the same rationale ``tests/_msl_ad_objective.py`` documents for
``tests/_gate_policy.py``'s envelope->gate multiplier (issue #528).

WHY BAND-MEAN |S21|**2 (mirrors issue #530's MSL reasoning): a full-matrix
``sum_ij|S_ij|**2`` reduction is passivity-pinned (``S^dagger S <= I``
caps it near a structural constant), which let issue #527's float32
comparator go blind once an extractor fix shrank the differentiated
residue. Band-mean ``|S21|**2`` has real dynamic range and moves directly
with whatever the eventual traced design variable controls (guided
wavelength / attenuation via the line's electrical length), the same
argument ``tests/_msl_ad_objective.py`` makes for the MSL gate.
"""

from __future__ import annotations

import jax.numpy as jnp


def coax_band_mean_s21_sq(S: "jnp.ndarray") -> "jnp.ndarray":
    """Band-mean ``|S21|**2`` from a coax two-port S-matrix.

    Parameters
    ----------
    S : (2, 2, n_freqs) complex
        As returned by ``CoaxialTwoPortResult.s_params`` /
        ``compute_coaxial_two_port(...).s_params``. Indexing follows
        ``solve_two_port_from_wave_amplitudes``'s own documented
        convention: ``S[j, i]`` is the response at port ``j`` when driving
        port ``i`` — S21 (port1 -> port2, i.e. response at port 2 when
        driving port 1) is therefore ``S[1, 0, :]``.

    Returns
    -------
    Real scalar: ``mean_f |S21(f)|**2`` over whatever frequency bins are in
    ``S`` — mirrors ``tests/_msl_ad_objective.py::msl_band_mean_s21_sq``'s
    reduction exactly (down to the single-bin case, mean of one element is
    that element), so a shared gate helper generalizes across both S-matrix
    layouts without a second hand-written reduction.
    """
    s21 = S[1, 0, :]
    return jnp.real(jnp.mean(jnp.abs(s21) ** 2))
