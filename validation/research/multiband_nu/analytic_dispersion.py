"""Exact discrete eigenfrequency of an empty PEC box on an rfx tensor grid,
with a per-axis error decomposition. Design + validity instrument for W4R3.

Model class: FIRST PRINCIPLES (SPEC-00 0.2-2 window-source rule). It is a
frequency-domain eigenvalue computation on the SAME difference operators
``rfx.nonuniform`` builds; it runs no FDTD time-stepping and shares no data
with the measurements it certifies.

Derivation
----------
For the empty PEC box the TE_{m,0,p} family has ``Ey`` as its only E
component and no y variation, so the rfx update equations
(``update_e_nu`` with ``inv_d_e[k] = 2/(d[k-1]+d[k])``, ``update_h_nu``
with ``inv_d_h[k] = 1/d[k]``) reduce to

    d2 Ey / dt2  =  -c0^2 (A_x + A_z) Ey

with the SAME 1-D operator on each axis a (Dirichlet ends: apply_pec pins
tangential Ey at the x and z faces),

    (A E)[k] = -inv_e[k] ( inv_h[k](E[k+1]-E[k]) - inv_h[k-1](E[k]-E[k-1]) )

on the PADDED profile, i.e. including the #562 trailing bounding-node
duplicate (``_append_bounding_node``) and ``inv_h[-1] = 0``. ``A`` is
similar to the symmetric ``sqrt(D) T sqrt(D)`` (``D = diag(inv_e)``,
``T`` the symmetric second-difference with ``inv_h``), so its eigenvalues
``mu`` are real and positive and are computed here by ``eigvalsh``.

Leapfrog in time turns ``mu`` into the exact discrete eigenfrequency

    sin(omega dt / 2) = (c0 dt / 2) sqrt(mu_x + mu_z)

which is the frequency an infinitely long, noise-free run of the solver
would report. Everything the fitted convergence order sees is contained in
the three ways that frequency differs from the continuum eigenfrequency
f_exact = (c0/2) sqrt((m/a)^2 + (p/L)^2):

    e_z = f(mu_x, mu_z, dt) - f(mu_x, kz^2, dt)     graded-axis dispersion
    e_x = f(mu_x, kz^2, dt) - f(kx^2, kz^2, dt)     transverse dispersion
    e_t = f(kx^2, kz^2, dt) - f_exact               time dispersion
    e_total = e_z + e_x + e_t  (exactly, by construction)

``e_z`` is the ONLY term that knows the z profile is graded; ``e_x`` and
``e_t`` are identical in a multiband arm and in its matched uniform
control (same dx, same dt), so a fixture in which they dominate cannot
fail its order gate for a grading reason. That is the W4R3 design
criterion; ``z_fraction`` below is what the note's fixture-validity gate
G1 is applied to.
"""

from __future__ import annotations

import numpy as np

C0 = 299792458.0


def padded_profile(profile: np.ndarray) -> np.ndarray:
    """The cell array the grid actually carries: profile + the #562
    trailing bounding-node duplicate (cpml_layers = 0, no pads)."""
    d = np.asarray(profile, dtype=np.float64)
    return np.concatenate([d, d[-1:]])


def inv_arrays(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float64 mirror of ``rfx.nonuniform._profile_to_inv_arrays`` on the
    padded profile: (inv_e, inv_h)."""
    dfull = padded_profile(profile)
    inv_local = 1.0 / dfull
    inv_h = np.concatenate([inv_local[:-1], [0.0]])
    inv_e = np.concatenate([inv_local[:1], 2.0 / (dfull[:-1] + dfull[1:])])
    return inv_e, inv_h


def operator_eigenvalues(profile: np.ndarray, n_modes: int,
                         inv_e_override: np.ndarray | None = None
                         ) -> np.ndarray:
    """The n_modes smallest eigenvalues mu of the 1-D rfx operator with
    PEC (Dirichlet) ends. ``inv_e_override`` injects a corrupted metric
    (revert-proof use)."""
    inv_e, inv_h = inv_arrays(profile)
    if inv_e_override is not None:
        inv_e = np.asarray(inv_e_override, dtype=np.float64)
    n_cells = len(np.asarray(profile))
    ks = np.arange(1, n_cells)                 # interior nodes 1..N-1
    s = np.sqrt(inv_e[ks])
    diag = s * (inv_h[ks] + inv_h[ks - 1]) * s
    off = -inv_h[ks[:-1]] * s[:-1] * s[1:]
    m = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return np.sort(np.linalg.eigvalsh(m))[:n_modes]


def leapfrog_freq(mu_x: float, mu_z: float, dt: float) -> float:
    """sin(w dt/2) = (c0 dt/2) sqrt(mu_x + mu_z)."""
    arg = C0 * dt * np.sqrt(mu_x + mu_z) / 2.0
    if not 0.0 < arg < 1.0:
        raise ValueError(f"non-propagating / unstable argument {arg}")
    return float(np.arcsin(arg) / (np.pi * dt))


def decompose(dz_profile: np.ndarray, dx: float, a_x: float, dt: float,
              m_x: int, p_z: int, l_z: float | None = None,
              inv_e_z_override: np.ndarray | None = None) -> dict:
    """Predicted frequency and the nested (e_z, e_x, e_t) decomposition.

    ``dx`` is the uniform transverse cell size, ``a_x`` the x extent,
    ``dt`` the grid's realized time step (pass ``float(grid.dt)`` so the
    model matches the run exactly).
    """
    dz = np.asarray(dz_profile, dtype=np.float64)
    l_z = float(dz.sum()) if l_z is None else float(l_z)
    n_x = int(round(a_x / dx))
    mu_x = float(operator_eigenvalues(np.full(n_x, dx), m_x)[m_x - 1])
    mu_z = float(operator_eigenvalues(dz, p_z, inv_e_z_override)[p_z - 1])
    k_x2 = (m_x * np.pi / a_x) ** 2
    k_z2 = (p_z * np.pi / l_z) ** 2
    f_exact = C0 * np.sqrt(k_x2 + k_z2) / (2 * np.pi)
    f_full = leapfrog_freq(mu_x, mu_z, dt)
    f_no_z = leapfrog_freq(mu_x, k_z2, dt)
    f_no_xz = leapfrog_freq(k_x2, k_z2, dt)
    e_z, e_x, e_t = f_full - f_no_z, f_no_z - f_no_xz, f_no_xz - f_exact
    denom = abs(e_z) + abs(e_x) + abs(e_t)
    return {
        "f_exact": f_exact, "f_model": f_full,
        "e_z": e_z, "e_x": e_x, "e_t": e_t, "e_total": f_full - f_exact,
        "z_fraction": abs(e_z) / denom if denom else float("nan"),
        "mu_x": mu_x, "mu_z": mu_z, "dt": float(dt),
    }
