"""Closed-form contracts only; no FDTD solves and no global precision changes."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "showcase"))
import _lens_filter_common as common
import design_filter as filt
import design_lens as lens


@pytest.mark.parametrize("case,pitch", [(lens, .003), (filt, .00254)], ids=["lens", "filter"])
def test_pixel_cell_physical_map(case, pitch):
    free = np.arange(np.prod(case.SHAPE)).reshape(case.SHAPE)
    # Independently locate several physical pixel centres in each mesh.
    for dx in case.MESHES:
        full = case.expand_pixels(free, dx)
        n = round(pitch / dx)
        if case is lens:
            assert full.shape == (30 * n, 30 * n, 10 * n)
            for i, j, k in ((0, 0, 0), (7, 3, 4), (14, 14, 9)):
                cell = tuple(int(np.floor(v / dx + 1e-9)) for v in
                             ((15 + i + .5) * pitch, (15 + j + .5) * pitch, (k + .5) * pitch))
                assert full[cell] == free[i, j, k]
            recovered = full.reshape(30, n, 30, n, 10, n).mean(axis=(1, 3, 5))
        else:
            assert full.shape == (32 * n, 9 * n, round(.01016 / dx))
            for i, j in ((0, 0), (5, 4), (15, 2), (31, 3)):
                cell = tuple(int(np.floor(v / dx + 1e-9)) for v in
                             ((i + .5) * pitch, (j + .5) * pitch, .00508))
                assert full[cell] == free[i, j]
            recovered = full[:, :, 0].reshape(32, n, 9, n).mean(axis=(1, 3))
        if dx == case.MESHES[0]:
            reference = recovered
        np.testing.assert_array_equal(recovered, reference)


@pytest.mark.parametrize("case", [lens, filt], ids=["lens", "filter"])
def test_expanded_mirror_symmetry(case):
    free = np.random.default_rng(1).normal(size=case.SHAPE)
    for dx in case.MESHES:
        full = case.expand_pixels(free, dx)
        np.testing.assert_array_equal(full, full[:, ::-1, :])
        if case is lens:
            np.testing.assert_array_equal(full, full[::-1, :, :])
        else:
            np.testing.assert_array_equal(full[:, :, 0], full[:, :, -1])
            assert full.shape[1] == round(.02286 / dx)  # centre pixel appears once


def test_grin_centre_and_clipping():
    assert lens.textbook_grin(0.) == pytest.approx(2.7)
    # n=1 at this radius; beyond it the declared n^2 profile clips to one.
    radius = np.sqrt((.045 + .030 * (np.sqrt(2.7) - 1)) ** 2 - .045 ** 2)
    assert lens.textbook_grin(radius) == pytest.approx(1.)
    assert lens.textbook_grin(.065) == 1.
    values = lens.textbook_grin(np.linspace(0, .064, 40))
    assert np.all((values >= 1) & (values <= 2.7))
    assert np.all(np.diff(values) <= 1e-12)


def test_mask_objective_inside_outside():
    # The optimization mask is the specified 2 dB inset (-17/-27).
    s11 = np.full(85, 10 ** (-18 / 20))
    s21 = np.full(85, 10 ** (-28 / 20))
    assert filt.mask_objective(s11, s21) == 0.
    s11[filt.PASS] = 10 ** (-16 / 20)
    assert filt.mask_objective(s11, s21) == pytest.approx(1.)
    s21[filt.STOP] = 10 ** (-25 / 20)
    assert filt.mask_objective(s11, s21) == pytest.approx(5.)
    assert filt.mask_objective(np.zeros(85), np.zeros(85)) == 0.
    assert np.count_nonzero(filt.PASS) == 13 and np.count_nonzero(filt.STOP) == 34


def test_best_iterate_earliest_minimum():
    assert common.best_iterate([8., 2., 5., 2., 6.]) == 1
    assert common.best_iterate([1., 1., 1.]) == 0
    assert common.best_iterate([5., 4., 2.]) == 2
    with pytest.raises(ValueError):
        common.best_iterate([1., np.nan])


@pytest.mark.parametrize("case", [lens, filt], ids=["lens", "filter"])
def test_cosine_schedule_endpoints(case):
    rates = common.cosine_schedule(np.arange(case.ITERATIONS), *case.LR, case.ITERATIONS)
    assert rates[0] == pytest.approx(case.LR[0])
    assert rates[-1] == pytest.approx(case.LR[1])
    assert np.all(np.diff(rates) < 0)
    assert common.cosine_schedule(29, *case.LR, case.ITERATIONS) > case.LR[1]


def test_filter_amendment1_pixel_indices():
    s2 = filt.starts()["S2"]
    np.testing.assert_array_equal(np.flatnonzero(np.all(s2 == 6, axis=1)) + 1,
                                  [6, 7, 16, 17, 26, 27])
    assert set(filt.FD_PIXELS.values()) == {(5, 4), (15, 4), (25, 4)}
    np.testing.assert_array_equal(s2, s2[::-1])
