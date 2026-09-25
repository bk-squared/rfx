"""Actual CPML operator mirror symmetry on staggered Yee coordinates."""

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h, init_cpml
from rfx.core.yee import init_state
from rfx.grid import Grid


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("magnetic", [False, True])
def test_cpml_correction_respects_physical_mirror(axis: int, magnetic: bool) -> None:
    """E reflects about (N-1)/2, transverse H about (N-2)/2."""
    h = 1 / 512
    grid = Grid(freq_max=5e9, domain=(23 * h,) * 3, dx=h, cpml_layers=12)
    params, auxiliary = init_cpml(grid)
    state = init_state(grid.shape)
    values = np.zeros(grid.shape, np.float32)
    low: list[int | slice] = [slice(None)] * 3
    high: list[int | slice] = [slice(None)] * 3
    count = grid.shape[axis]
    if magnetic:
        low[axis], high[axis] = 6, count - 7
        values[tuple(low)] = values[tuple(high)] = 1
        source_component = "e" + "xyz"[(axis + 2) % 3]
        result_component = "h" + "xyz"[(axis + 1) % 3]
        state = state._replace(**{source_component: jnp.asarray(values)})
        result, _ = apply_cpml_h(state, params, auxiliary, grid, axes="xyz"[axis])
        indices = np.arange(count - 1)
        actual = np.take(np.asarray(getattr(result, result_component)), indices, axis=axis)
        expected = -np.flip(actual, axis=axis)
    else:
        low[axis], high[axis] = 5, count - 7
        values[tuple(low)] = 1
        values[tuple(high)] = -1
        source_component = "h" + "xyz"[(axis + 1) % 3]
        result_component = "e" + "xyz"[(axis + 2) % 3]
        state = state._replace(**{source_component: jnp.asarray(values)})
        result, _ = apply_cpml_e(state, params, auxiliary, grid, axes="xyz"[axis])
        actual = np.asarray(getattr(result, result_component))
        expected = np.flip(actual, axis=axis)
    assert np.any(actual != 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("budget", [1, 4])
def test_one_sample_faces_retain_legacy_coefficient_convention(budget: int) -> None:
    """One sample has no resolved grading interval; do not invent a new origin."""
    h = 1 / 512
    grid = Grid(freq_max=5e9, domain=(17 * h,) * 3, dx=h, cpml_layers=budget,
                face_layers={axis + side: 1 for axis in "xyz" for side in ("_lo", "_hi")})
    params, _ = init_cpml(grid)
    assert params.magnetic is not None
    for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"):
        electric = getattr(params, face)
        magnetic = getattr(params.magnetic, face)
        for name in ("sigma", "kappa", "alpha", "b", "c"):
            np.testing.assert_array_equal(getattr(electric, name), getattr(magnetic, name))


# --------------------------------------------------------------------------
# Falsifiers for the mirror tests above (independent review, 2026-09-14).
#
# The mirror tests discriminate STAGGERING but not absorber ORIENTATION: an
# in-memory swap of the lo/hi magnetic pair reverses the grading and still
# passes all three physical mirrors with residual exactly 0, because swapping
# a mirror-consistent pair is itself mirror-consistent. The mirror must
# therefore not be the only gate on the magnetic profiles. The three tests
# below close that: an absolute orientation pin on the profiles themselves,
# a swap falsifier on the field correction, and two mirror arms that actually
# exercise b's memory term and nontrivial kappa grading (the original arms
# start from psi = 0 and run at kappa_max = 1, so neither was reached).
# --------------------------------------------------------------------------

# Leading psi axis is always the face's own axis, so the mirror is a flip
# along axis 0 whatever the trailing permutation is. Names follow the
# component the mirror tests read: magnetic axis a drives h[(a+1) % 3],
# electric axis a drives e[(a+2) % 3].
_MAGNETIC_PSI = {
    0: ("psi_hy_xlo", "psi_hy_xhi"),
    1: ("psi_hz_ylo", "psi_hz_yhi"),
    2: ("psi_hx_zlo", "psi_hx_zhi"),
}
_ELECTRIC_PSI = {
    0: ("psi_ez_xlo", "psi_ez_xhi"),
    1: ("psi_ex_ylo", "psi_ex_yhi"),
    2: ("psi_ey_zlo", "psi_ey_zhi"),
}


def _stagger_grid() -> Grid:
    """The fixture the mirror tests above use: 12 uniform layers, h = 1/512."""
    h = 1 / 512
    return Grid(freq_max=5e9, domain=(23 * h,) * 3, dx=h, cpml_layers=12)


def _one_update_mirror(grid, params, auxiliary, axis: int, magnetic: bool,
                       offset: int | None = None):
    """One actual CPML update from a mirror-consistent source.

    Returns ``(actual, expected)`` where *expected* is *actual* under the
    physical mirror: E reflects about (N-1)/2, transverse H about (N-2)/2.
    This is the body of ``test_cpml_correction_respects_physical_mirror``,
    reused so the falsifiers below measure the same operator. *offset* is
    the lo-side source index; its default reproduces that test exactly.
    A forward difference reaches absorber layer ``offset - 1``, so only a
    small offset drives the OUTERMOST layer -- the one the orientation of
    the grading is about.
    """
    state = init_state(grid.shape)
    values = np.zeros(grid.shape, np.float32)
    low: list[int | slice] = [slice(None)] * 3
    high: list[int | slice] = [slice(None)] * 3
    count = grid.shape[axis]
    if magnetic:
        offset = 6 if offset is None else offset
        low[axis], high[axis] = offset, count - 1 - offset
        values[tuple(low)] = values[tuple(high)] = 1
        source_component = "e" + "xyz"[(axis + 2) % 3]
        result_component = "h" + "xyz"[(axis + 1) % 3]
        state = state._replace(**{source_component: jnp.asarray(values)})
        result, _ = apply_cpml_h(state, params, auxiliary, grid, axes="xyz"[axis])
        indices = np.arange(count - 1)
        actual = np.take(np.asarray(getattr(result, result_component)), indices, axis=axis)
        expected = -np.flip(actual, axis=axis)
    else:
        offset = 5 if offset is None else offset
        low[axis], high[axis] = offset, count - 2 - offset
        values[tuple(low)] = 1
        values[tuple(high)] = -1
        source_component = "h" + "xyz"[(axis + 1) % 3]
        result_component = "e" + "xyz"[(axis + 2) % 3]
        state = state._replace(**{source_component: jnp.asarray(values)})
        result, _ = apply_cpml_e(state, params, auxiliary, grid, axes="xyz"[axis])
        actual = np.asarray(getattr(result, result_component))
        expected = np.flip(actual, axis=axis)
    return actual, expected


def _mirror_consistent_psi(auxiliary, axis: int, magnetic: bool, seed: int):
    """Seed the face's psi pair with a mirror-consistent random carry.

    The mirror holds on a nonzero carry only if the hi-face coefficients are
    the lo-face coefficients read backwards at the face's own stagger. The
    H carry is antisymmetric under i -> depth-2-i (its last sample lands on
    the excluded global index N-1); the E carry is symmetric under
    i -> depth-1-i.
    """
    lo_name, hi_name = (_MAGNETIC_PSI if magnetic else _ELECTRIC_PSI)[axis]
    template = np.asarray(getattr(auxiliary, lo_name))
    rng = np.random.default_rng(seed)
    lo = rng.standard_normal(template.shape).astype(template.dtype)
    hi = np.zeros_like(lo)
    if magnetic:
        # depth-1 pairs with depth-2-j; the innermost lo sample (global
        # index depth-1) mirrors to a node no face buffer covers, and the
        # innermost hi sample lands on the excluded global index N-1. Both
        # are zeroed so the arm asserts the mirror, not a coincidence.
        lo[-1] = 0.0
        hi[:-1] = -np.flip(lo[:-1], axis=0)
    else:
        hi[:] = np.flip(lo, axis=0)
    return auxiliary._replace(**{lo_name: jnp.asarray(lo), hi_name: jnp.asarray(hi)})


@pytest.mark.parametrize("axis_name", ["x", "y", "z"])
def test_magnetic_profiles_pin_the_absorber_orientation(axis_name: str) -> None:
    """rho itself, not only its mirror: lo grades DOWN inward from (n-1.5)/(n-1).

    The mirror tests pass on a reversed pair, so the profiles need an
    absolute pin. Every value here is derived from ``n``, and the two scales
    (sigma_max, the alpha slope) are read off the E profile this change
    leaves bit-identical -- nothing is pasted from a previous run.
    """
    grid = _stagger_grid()
    params, _ = init_cpml(grid)
    assert params.magnetic is not None
    n = grid.cpml_layers
    index = np.arange(n, dtype=np.float64)
    # H samples the E grading half a cell further into the domain on the lo
    # face, and half a cell further out before the hi-face reversal.
    rho_lo = np.clip((n - 1 - index - 0.5) / (n - 1), 0.0, 1.0)
    rho_hi = np.clip((index + 0.5) / (n - 1), 0.0, 1.0)

    electric_lo = getattr(params, f"{axis_name}_lo")
    magnetic_lo = getattr(params.magnetic, f"{axis_name}_lo")
    magnetic_hi = getattr(params.magnetic, f"{axis_name}_hi")
    # alpha = alpha_slope * (1 - rho) is linear, so it recovers rho without
    # assuming the grading order. Both scales come from the E profile: its
    # lo face runs rho = 1 at index 0 down to rho = 0 at index n-1.
    sigma_max = float(np.asarray(electric_lo.sigma)[0])
    alpha_slope = float(np.asarray(electric_lo.alpha)[-1])
    assert sigma_max > 0 and alpha_slope > 0
    measured_lo = 1.0 - np.asarray(magnetic_lo.alpha, dtype=np.float64) / alpha_slope
    measured_hi = 1.0 - np.asarray(magnetic_hi.alpha, dtype=np.float64) / alpha_slope

    # The four endpoints, each derived from n (n = 12 -> 0.9545454..., 0,
    # 0.0454545..., 1). A reversed pair fails all four.
    assert measured_lo[0] == pytest.approx((n - 1.5) / (n - 1), rel=1e-6)
    assert measured_lo[-1] == pytest.approx(0.0, abs=1e-6)
    assert measured_hi[0] == pytest.approx(0.5 / (n - 1), rel=1e-6)
    assert measured_hi[-1] == pytest.approx(1.0, rel=1e-6)
    np.testing.assert_allclose(measured_lo, rho_lo, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(measured_hi, rho_hi, rtol=1e-6, atol=1e-7)

    # sigma follows the same rho. The grading order is not pasted either: it
    # is the exponent the untouched E profile already obeys.
    rho_electric = 1.0 - index / (n - 1)
    order = int(round(float(np.log(
        np.asarray(electric_lo.sigma, dtype=np.float64)[1] / sigma_max,
    ) / np.log(rho_electric[1]))))
    np.testing.assert_allclose(
        np.asarray(electric_lo.sigma, dtype=np.float64),
        sigma_max * rho_electric**order, rtol=1e-6, atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(magnetic_lo.sigma, dtype=np.float64),
        sigma_max * rho_lo**order, rtol=1e-6, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(magnetic_hi.sigma, dtype=np.float64),
        sigma_max * rho_hi**order, rtol=1e-6, atol=1e-12,
    )
    # Direction, stated once without reference to any mirror.
    assert np.all(np.diff(np.asarray(magnetic_lo.sigma)) < 0)
    assert np.all(np.diff(np.asarray(magnetic_hi.sigma)) > 0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_swapped_magnetic_face_pair_passes_the_mirror_but_moves_the_field(axis: int) -> None:
    """The swap falsifier: the mirror cannot be the only gate.

    Exchanging the lo and hi magnetic profiles reverses the absorber -- the
    outermost lo layer now carries the weakest grading -- yet the swapped
    pair still satisfies the physical H mirror exactly, because a swapped
    mirror-consistent pair is mirror-consistent. What DOES move is the
    field: the correction at the outer lo-face layer is a different number.
    """
    grid = _stagger_grid()
    params, auxiliary = init_cpml(grid)
    assert params.magnetic is not None
    face = "xyz"[axis]
    magnetic = params.magnetic
    swapped = params._replace(magnetic=magnetic._replace(**{
        f"{face}_lo": getattr(magnetic, f"{face}_hi"),
        f"{face}_hi": getattr(magnetic, f"{face}_lo"),
    }))

    # offset=1 so the forward difference reaches absorber layer 0.
    correct, correct_mirror = _one_update_mirror(
        grid, params, auxiliary, axis, True, offset=1)
    reversed_, reversed_mirror = _one_update_mirror(
        grid, swapped, auxiliary, axis, True, offset=1)
    # The blind spot, asserted rather than described: both pass the mirror.
    np.testing.assert_allclose(correct, correct_mirror, rtol=1e-6, atol=0)
    np.testing.assert_allclose(reversed_, reversed_mirror, rtol=1e-6, atol=0)

    outer_correct = np.take(correct, [0], axis=axis)
    outer_reversed = np.take(reversed_, [0], axis=axis)
    assert np.any(outer_correct != 0)
    # The falsifier: a reversed absorber is a different operator at the
    # outermost layer, and the difference is not a rounding artefact.
    residual = float(np.max(np.abs(outer_correct - outer_reversed)))
    assert residual > 0.0
    # Direction, not merely difference. The correct orientation puts the
    # strongest grading on the outermost layer, so its correction there is
    # the larger one by orders of magnitude (measured 0.0014300662 vs
    # 4.6407476e-07 A/m, ratio ~3.1e3, on all three axes). The 100x bound
    # is loose on purpose: the claim is the direction, not the ratio.
    assert float(np.max(np.abs(outer_correct))) > 100.0 * float(
        np.max(np.abs(outer_reversed)))

    # The same falsifier on the fixture the mirror tests themselves use
    # (offset 6, which drives layers 5 and 6, not layer 0): the peak
    # correction over the whole component moves 0.0005091045 -> 0.0007448532
    # A/m under the swap while the mirror residual stays exactly 0. This is
    # the reviewer's own measurement, reproduced here so the blind spot is
    # pinned at the geometry where it was found.
    default_correct, _ = _one_update_mirror(grid, params, auxiliary, axis, True)
    default_reversed, default_mirror = _one_update_mirror(
        grid, swapped, auxiliary, axis, True)
    np.testing.assert_allclose(default_reversed, default_mirror, rtol=1e-6, atol=0)
    peak_correct = float(np.max(np.abs(default_correct)))
    peak_reversed = float(np.max(np.abs(default_reversed)))
    assert peak_correct > 0 and peak_reversed != peak_correct
    assert abs(peak_reversed - peak_correct) > 1e-6


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("magnetic", [False, True])
@pytest.mark.parametrize("arm", ["memory", "kappa"])
def test_physical_mirror_holds_with_memory_and_kappa_grading(
    axis: int, magnetic: bool, arm: str,
) -> None:
    """The mirror arms above start at psi = 0 and kappa_max = 1.

    Neither reaches b (the recursive-convolution memory term multiplies a
    zero carry) nor a nontrivial kappa grading. These two arms do, on the
    same operator and the same rtol=1e-6/atol=0 mirror: the ``memory`` arm
    carries a seeded nonzero psi, the ``kappa`` arm runs kappa_max = 3.
    """
    grid = _stagger_grid()
    kappa_max = 3.0 if arm == "kappa" else None
    params, auxiliary = init_cpml(grid, kappa_max=kappa_max)
    assert params.magnetic is not None
    # The profile this arm actually exercises: H reads .magnetic, E does not.
    profile = getattr(params.magnetic if magnetic else params, "xyz"[axis] + "_lo")
    if arm == "memory":
        auxiliary = _mirror_consistent_psi(auxiliary, axis, magnetic, seed=20260914)
        carry = getattr(auxiliary, (_MAGNETIC_PSI if magnetic else _ELECTRIC_PSI)[axis][0])
        assert float(np.max(np.abs(np.asarray(carry)))) > 0
        # b actually multiplies something: the memory term is live.
        b = np.asarray(profile.b)
        assert np.all(b > 0) and np.any(b < 1)
    else:
        kappa = np.asarray(profile.kappa)
        assert kappa[0] > 1.0 and np.any(np.diff(kappa) != 0)

    actual, expected = _one_update_mirror(grid, params, auxiliary, axis, magnetic)
    assert np.any(actual != 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)
