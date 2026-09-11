"""Passivity enforcement on compute_msl_s_matrix: strict ||S||_2 <= 1, loudly.

The bound is enforced by per-frequency singular-value clipping (nearest
passive matrix in spectral norm). It is constraint enforcement, not a
physics fix: the raw extraction is preserved in S_raw, the per-bin clip in
passivity_correction, a warning names the touched bins, and the raw-value
self-check still audits what was measured. Measured context (PR #462 Sheen
study): the raw coarse-mesh extraction exceeds the bound by up to sigma~3 in
the stopband where the standing-wave-null mask already marks the bins
unreliable — the projection bounds those bins, it does not bless them.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.api._sparams import _project_passive
from tests._realized_geometry import assert_sheet_planes, assert_wall_planes


def _sigma_max(S):
    # Verify the stored matrix, not a singular value rounded back to f32.
    # A tiny off-diagonal term can make sigma_max > 1 even when that
    # singular value rounds to exactly 1 in the matrix's storage dtype.
    S = np.asarray(S, dtype=np.complex128)
    return np.array([
        np.linalg.svd(S[:, :, k], compute_uv=False)[0]
        for k in range(S.shape[2])
    ])


def test_passivity_referee_resolves_gain_below_one_float32_ulp():
    coupling = np.float32(1e-8)
    matrix = np.array([[1., coupling], [coupling, 1.]], dtype=np.complex64)[:, :, None]
    # The symmetric common-mode vector has the analytic gain 1+coupling.
    measured = float(_sigma_max(matrix)[0])
    assert measured > 1.
    assert measured == pytest.approx(1.+float(coupling), rel=0.,
                                     abs=8.*np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# Unit level: the projection helper itself.
# ---------------------------------------------------------------------------

def test_projection_clips_only_the_nonpassive_bins():
    S = np.zeros((2, 2, 3), dtype=complex)
    S[:, :, 0] = [[0.3, 0.5], [0.5, 0.3]]     # passive
    S[:, :, 1] = [[0.0, 3.0], [0.2, 0.0]]     # sigma_max ~ 3.007
    S[:, :, 2] = np.eye(2)                    # exactly at the bound

    S_pass, corr = _project_passive(jnp.asarray(S))
    S_pass = np.asarray(S_pass)
    corr = np.asarray(corr)

    assert np.allclose(S_pass[:, :, 0], S[:, :, 0]), "passive bin must be untouched"
    assert corr[0] == 0.0
    assert corr[1] == pytest.approx(np.linalg.svd(S[:, :, 1], compute_uv=False)[0] - 1.0)
    # Strict bound including the reconstruction round-trip.
    assert np.all(_sigma_max(S_pass) <= 1.0)


@pytest.mark.parametrize("n_ports", [2, 8, 32])
def test_projection_bound_is_strict_at_float32(n_ports):
    """min(sigma, 1) alone reconstructs to 1 + O(eps), and the error GROWS
    with n_ports — an 8*eps margin measured 1.0000000255 at n=8 and
    1.0000006584 at n=32. The 64*eps margin must hold the strict bound after
    the float32 round-trip across port counts."""
    rng = np.random.default_rng(7)
    S = (rng.normal(size=(n_ports, n_ports, 64))
         + 1j * rng.normal(size=(n_ports, n_ports, 64))).astype(np.complex64)
    S *= 2.0  # comfortably non-passive everywhere
    S_pass, _ = _project_passive(jnp.asarray(S))
    assert np.all(_sigma_max(np.asarray(S_pass)) <= 1.0)


@pytest.mark.parametrize("backend", [
    "cpu", pytest.param("gpu", marks=pytest.mark.gpu),
])
@pytest.mark.parametrize("n_ports", [2, 8, 32])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_near_passive_complex_projection_preserves_the_measured_matrix(backend, n_ports, dtype):
    """Small clipping must not introduce a larger reconstruction error.

    Nearly unitary matrices model a settled low-loss multiport. Known
    orthonormal factors define the analytic answer without another SVD.
    Exercise the GPU's lower-precision ambient matmul setting explicitly:
    the physical projection must honor its output dtype's error budget.
    """
    if backend == "gpu" and jax.default_backend() != "gpu":
        pytest.skip("requires an actual GPU")
    rng = np.random.default_rng(729)
    eps = np.finfo(np.empty((), dtype=dtype).real.dtype).eps
    matrices, references = [], []
    for _ in range(12):
        u, _ = np.linalg.qr(rng.normal(size=(n_ports, n_ports))
                            + 1j*rng.normal(size=(n_ports, n_ports)))
        v, _ = np.linalg.qr(rng.normal(size=(n_ports, n_ports))
                            + 1j*rng.normal(size=(n_ports, n_ports)))
        singular = 1. + np.linspace(-1e-3, 1e-3, n_ports)
        matrices.append((u*singular) @ v.conj().T)
        references.append((u*np.minimum(singular, 1.-64.*eps)) @ v.conj().T)
    raw = np.stack(matrices, axis=-1).astype(dtype)
    expected = np.stack(references)
    with (jax.experimental.enable_x64(), jax.default_device(jax.devices(backend)[0]),
          jax.default_matmul_precision("tensorfloat32")):
        device_raw = jnp.asarray(raw)
        projected, correction = _project_passive(device_raw)
        assert projected.dtype == device_raw.dtype
        assert correction.dtype == device_raw.real.dtype
        assert projected.devices() == correction.devices() == device_raw.devices()
        projected = np.asarray(projected).transpose(2, 0, 1).astype(np.complex128)
        correction = np.asarray(correction)
    # The existing clipping radius leaves 64 output ULPs for reconstruction.
    # Spending more than that on multiplication defeats its strict bound.
    assert np.max(np.abs(projected-expected)) <= 64.*eps
    # Independent power-operator eigensolve, not the production SVD again.
    power = projected.conj().transpose(0, 2, 1) @ projected
    assert np.linalg.eigvalsh(power).max() <= 1.
    # The constructed sigma_max is 1.001; input rounding is within this
    # dtype-scaled bound. Do not derive the expected clip from production.
    assert np.max(np.abs(correction-1e-3)) <= 32.*eps


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_projection_preserves_nonfinite_bins_for_the_finiteness_audit(bad_value):
    raw = np.zeros((2, 2, 3), dtype=np.complex64)
    raw[:, :, 0] = [[.3, .5], [.5, .3]]
    raw[:, :, 1] = [[0., bad_value], [1., 0.]]
    raw[:, :, 2] = [[0., 2.], [2., 0.]]
    result, correction = map(np.asarray, _project_passive(jnp.asarray(raw)))
    assert np.all(np.isnan(result[:, :, 1]))
    assert np.isnan(correction[1])
    np.testing.assert_allclose(result[:, :, 0], raw[:, :, 0], rtol=1e-6)
    assert _sigma_max(result[:, :, [0, 2]]).max() <= 1.
    assert correction[0] == 0.
    assert correction[2] == pytest.approx(1.)


def test_projection_preserves_f64_arrays_after_their_creation_context_exits():
    with jax.experimental.enable_x64():
        raw = jnp.asarray([[[.2], [1.]], [[1.], [.2]]], dtype=jnp.complex128)
    with jax.experimental.disable_x64():
        projected, correction = _project_passive(raw)
        assert not jax.config.x64_enabled
        assert projected.dtype == np.dtype(np.complex128)
        assert correction.dtype == np.dtype(np.float64)
        assert projected.devices() == raw.devices()
        assert _sigma_max(np.asarray(projected)).max() <= 1.


# ---------------------------------------------------------------------------
# End to end on a tiny thru line (~seconds of FDTD). A deliberately truncated
# record is the worst case: its raw S carries genuine truncation artifacts.
# ---------------------------------------------------------------------------

def _thru():
    sim = Simulation(freq_max=20e9, domain=(0.012, 0.008, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0.0008)), material="sub")
    # 35 um foil: a SHEET (#931 §1.3), declared by a zero-thickness Box
    # on the laminate top. h_sub / dx = 0.8 mm / 0.2 mm = 4, so the substrate face is a
    # node line and the sheet lands on it exactly. Drawn one cell thick
    # before the contract, it would now be a VOLUME — walls at BOTH z
    # faces and the Ez edge between them shorted.
    sim.add(Box((0.0, 0.0034, 0.0008), (0.012, 0.0046, 0.0008)), material="pec")
    sim.add_msl_port(position=(0.002, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    return sim


FREQS = jnp.linspace(2e9, 18e9, 16)


def test_default_result_is_strictly_passive_and_loud():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _thru().compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)

    assert np.all(_sigma_max(res.S) <= 1.0), "the default S must satisfy the bound at every frequency"

    # Nothing hidden: raw kept, correction recorded, warning fired.
    assert res.S_raw is not None
    assert res.passivity_correction is not None
    assert float(np.max(np.asarray(res.passivity_correction))) > 0.0
    messages = [str(w.message) for w in caught]
    assert any("projected onto the passive set" in m for m in messages)
    # The projection must not silence the artifact diagnoses.
    assert any("settling witness" in m for m in messages)


def test_enforce_passivity_false_returns_the_raw_extraction():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _thru().compute_msl_s_matrix(freqs=FREQS, num_periods=2.0,
                                           enforce_passivity=False)
    assert res.S_raw is None and res.passivity_correction is None
    # The truncated raw extraction genuinely violates the bound — that is
    # exactly what the default projects away.
    assert float(_sigma_max(res.S).max()) > 1.0


def test_projection_agrees_with_offline_projection_of_the_raw():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _thru().compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)
    S_off, corr_off = _project_passive(jnp.asarray(np.asarray(res.S_raw)))
    assert np.allclose(np.asarray(res.S), np.asarray(S_off), atol=1e-6)
    assert np.allclose(np.asarray(res.passivity_correction),
                       np.asarray(corr_off), atol=1e-6)


def test_realized_conductor_planes_equal_the_declaration():
    """Build-time witness (no solve) for the #931 ownership contract.

    The foil is declared as a SHEET, so the lattice must give it exactly
    ONE wall plane, on the node line of the laminate face it was drawn on,
    with the normal Ez edge through it left live. Drawn one cell thick it
    was a volume: two walls, and the Ez edge between them shorted. This
    assertion is what keeps the declaration and the realization the same
    statement.
    """
    sim = _thru()
    assert_sheet_planes(sim, 2, [0.0008], what="MSL thru foil")
    assert_wall_planes(sim, 2, [0.0008], what="MSL thru foil")
