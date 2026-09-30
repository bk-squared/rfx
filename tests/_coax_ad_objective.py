"""Shared coax two-port transmitted-power reduction.

For a matched lossless thru this is physically constant. After the realized
TEM source fix (#1356), the two-port AD gate uses transmission phase for its
nonzero sensitivity and retains this reduction only as a small-power-slope
sanity check. It is also used by the standalone power diagnostic referee.
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
