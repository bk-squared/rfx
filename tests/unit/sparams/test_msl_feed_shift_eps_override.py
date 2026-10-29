"""The MSL feed-plane shift under ``eps_override`` moves S along the SOLVED line (#2199).

``compute_msl_s_matrix`` measures S at each port's probe 0 and refers it to the
declared feed plane with a closed-form line: a phase ``exp(-j beta L)`` applied
in the line's own impedance, then S referred to ``impedance=``. When an
``eps_override`` changes the permittivity under a port line, that line is the
one the fields were solved with, not the declared one. Before #2199 the shift
used the declared permittivity: on a 20 mm, 0.8 mm-thick line with 4 mm
offsets the S21 phase was off by up to 0.10 rad, |S11| by up to 0.06, and
d(arg S21)/d(eps) came out 64-66 % of the correct value.

Planted records (the harness of ``tests/unit/autodiff/test_msl_forward_identity.py``):
the probe-0 V/I do not depend on ``eps_override``, so the whole effect of the
override on S here is the plane shift and its reference. The expected S is
written out from the planted probe-0 S with the NumPy closed forms
(``hammerstad_jensen_z0_eps_eff``, ``msl_closed_form_beta``), not the traced
``msl_line_beta_z0`` the product now calls, and the impedance-matrix
renormalization is written out here as well.

Mutations (run by hand, 2026-10-29): (a) the shift back on the declared line
(the line permittivity never read): all four rows fail; (b) the solved line's
beta kept but the shift left in the declared Z0 reference: the value and
gradient rows fail; (c) the uniformity check removed: the varying-override
refusal row fails.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx._traced_refusal import TracedStructureRefusal
from rfx.measurement.msl_planes import msl_closed_form_beta
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
from tests.unit.autodiff.test_msl_forward_identity import (
    FREQS,
    U,
    _build_planted_sim,
    _compute,
    _noop_eps_override,
    _planted_s,
)

W, H, L = 4 * U, 2 * U, 10 * U  # trace width, substrate, feed-to-probe-0 (n_probe_offset=10, dx=U)
DECLARED = 2.0


def _renorm(S, z_from, z_to):
    """Power-wave S from one real reference (all ports) to another, via Z."""
    eye = np.eye(S.shape[0])
    out = np.empty_like(S)
    for k in range(S.shape[2]):
        z = z_from * (eye + S[:, :, k]) @ np.linalg.inv(eye - S[:, :, k])
        out[:, :, k] = (z - z_to * eye) @ np.linalg.inv(z + z_to * eye)
    return out


def _expected(eps_line: float) -> np.ndarray:
    """Planted probe-0 S (formed at the declared line's Z0) moved along a line of
    permittivity ``eps_line`` to the feed planes and referred to 50 ohm."""
    planted = np.asarray(_planted_s(1.0), dtype=np.complex128)
    z_decl = hammerstad_jensen_z0_eps_eff(W, H, DECLARED)[0]
    z_line = hammerstad_jensen_z0_eps_eff(W, H, eps_line)[0]
    d = np.exp(-1j * msl_closed_form_beta(W, H, eps_line, FREQS) * L)
    S = _renorm(planted, z_decl, z_line) * d[None, None, :] * d[None, None, :]
    return _renorm(S, z_line, 50.0)


def _override(sim, eps_line: float):
    """The sim's own eps with every substrate cell set to ``eps_line``."""
    eps = np.asarray(_noop_eps_override(sim)).copy()
    eps[eps == DECLARED] = eps_line
    return jnp.asarray(eps, dtype=jnp.float32)


def test_shift_follows_the_solved_line_not_the_declaration():
    sim, _ = _build_planted_sim(1.0)
    S = np.asarray(_compute(sim, eps_override=_override(sim, 3.0)).S, dtype=np.complex128)
    want, frozen = _expected(3.0), _expected(DECLARED)
    # The fixture can tell the two lines apart by far more than the tolerance.
    assert np.max(np.abs(want - frozen)) > 1e-3
    np.testing.assert_allclose(S, want, atol=2e-5 * np.max(np.abs(want)))


def test_gradient_carries_the_plane_term():
    sim, _ = _build_planted_sim(1.0)
    eps0 = _noop_eps_override(sim)
    k = len(FREQS) - 1

    def phase21(alpha):
        return jnp.angle(_compute(sim, eps_override=eps0 * alpha).S[1, 0, k])

    g_ad = float(jax.grad(phase21)(jnp.float32(1.0)))
    h = 1e-3
    g_fd = (np.angle(_expected(DECLARED * (1 + h))[1, 0, k])
            - np.angle(_expected(DECLARED * (1 - h))[1, 0, k])) / (2 * h)
    assert abs(g_fd) > 1e-3, g_fd
    assert abs(g_ad - g_fd) <= 2e-3 * abs(g_fd), (g_ad, g_fd)


def test_override_varying_along_the_shifted_length_is_refused_by_port_name():
    sim, _ = _build_planted_sim(1.0)
    eps = np.asarray(_override(sim, DECLARED)).copy()
    grid = sim._build_grid()
    i_mid = grid.position_to_index((2 * U + 5 * U, 0, 0))[0]  # between port 1's feed and probe 0
    eps[i_mid, :, :][eps[i_mid, :, :] == DECLARED] = 2.6
    with pytest.raises(NotImplementedError, match=r"sees a permittivity from 2 to 2\.6 under its line"):
        _compute(sim, eps_override=jnp.asarray(eps))


def test_override_without_values_is_refused_under_jit():
    sim, _ = _build_planted_sim(1.0)
    eps0 = _noop_eps_override(sim)
    with pytest.raises(TracedStructureRefusal, match="must check that it is uniform"):
        jax.jit(lambda e: _compute(sim, eps_override=e).S)(eps0)

