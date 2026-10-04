"""Closed-form contracts only; no FDTD solves and no global precision changes."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "showcase"))
import _lens_filter_common as common
import design_filter as filt
import design_lens as lens
import _visual_data as visual


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


# Visual helpers have no solver dependency; coordinate-coded arrays catch flips.


def test_visual_lens_quarter_coordinates():
    q = np.arange(2250).reshape(15, 15, 10)
    full = visual.mirror_lens(q)
    assert full.shape == (30, 30, 10)
    for i in range(30):
        for j in range(30):
            np.testing.assert_array_equal(full[i, j], q[abs(i - 14.5).astype(int) if isinstance(i, np.ndarray) else int(abs(i - 14.5)), int(abs(j - 14.5))])
    np.testing.assert_array_equal(full[14, 14], q[0, 0])
    np.testing.assert_array_equal(full[15, 15], q[0, 0])
    with pytest.raises(ValueError):
        visual.mirror_lens(np.zeros((15, 15, 9)))


def test_visual_filter_centre_occurs_once():
    half = np.tile(np.arange(5), (32, 1)) + 10 * np.arange(32)[:, None]
    full = visual.mirror_filter(half)
    assert full.shape == (32, 9)
    for i in range(32):
        np.testing.assert_array_equal(full[i], i * 10 + np.array([0, 1, 2, 3, 4, 3, 2, 1, 0]))
    with pytest.raises(ValueError):
        visual.mirror_filter(np.zeros((32, 4)))


@pytest.mark.parametrize('plane,pos,negative', [('E', 0, 36), ('H', 18, 54)])
def test_visual_signed_plane_cut(plane, pos, negative):
    response = 1000 * np.arange(73)[:, None] + np.arange(73)[None, :]
    theta, cut = visual.plane_cut(response, plane)
    assert len(theta) == len(cut) == 145
    assert np.count_nonzero(theta == 0) == 1
    assert np.all(np.diff(theta) > 0)
    assert theta[0] == pytest.approx(-180 + np.rad2deg(1e-4))
    assert theta[73] == pytest.approx(np.rad2deg(np.linspace(1e-4, np.pi - 1e-4, 73)[1]))
    np.testing.assert_array_equal(cut[:72], response[72:0:-1, negative])
    np.testing.assert_array_equal(cut[72:], response[:, pos])
    _, batched = visual.plane_cut(np.stack([response, response + 7]), plane)
    np.testing.assert_array_equal(batched[1], cut + 7)


def test_visual_dbi_power_not_amplitude():
    np.testing.assert_allclose(visual.dbi([.01, 1, 10, 100]), [-20, 0, 10, 20])
    assert np.isneginf(visual.dbi(0))
    assert visual.dbi(4 * np.pi * .09**2 / (299792458 / 1e10)**2) == pytest.approx(20.54053477)
    with pytest.raises(ValueError):
        visual.dbi(-1)
    np.testing.assert_allclose(visual.amplitude_db([0, .1j, 1]), [-120, -20, 0])


def test_visual_lens_quarter_orientation():
    from _visual_data import lens_mirror
    q = np.arange(2250).reshape(15, 15, 10)
    full = lens_mirror(q)
    assert full.shape == (30, 30, 10)
    np.testing.assert_array_equal(full[15:, 15:], q)
    assert full[0, 0, 9] == q[14, 14, 9]
    np.testing.assert_array_equal(full, full[::-1])
    np.testing.assert_array_equal(full, full[:, ::-1])


def test_visual_filter_centre_not_duplicated():
    from _visual_data import filter_mirror
    q = np.arange(160).reshape(32, 5)
    full = filter_mirror(q)
    assert full.shape == (32, 9)
    np.testing.assert_array_equal(full[:, :5], q)
    np.testing.assert_array_equal(full[0], [0, 1, 2, 3, 4, 3, 2, 1, 0])


@pytest.mark.parametrize('plane,positive,negative', [('E', 0, 36), ('H', 18, 54)])
def test_visual_cuts_opposite_phi_branch(plane, positive, negative):
    from _visual_data import plane_cut
    d = np.arange(73*73).reshape(73, 73)
    theta, cut = plane_cut(d, plane)
    assert theta.shape == cut.shape == (145,)
    assert theta[72] == 0
    assert np.all(np.diff(theta) > 0)
    np.testing.assert_array_equal(cut[:72], d[:0:-1, negative])
    np.testing.assert_array_equal(cut[72:], d[:, positive])
    assert theta[-1] == pytest.approx(np.rad2deg(np.pi-1e-4))


def test_visual_dbi_is_power_logarithm():
    from _visual_data import dbi
    np.testing.assert_allclose(dbi([.1, 1, 10, 100]), [-10, 0, 10, 20])
    assert np.isneginf(dbi(0))



def test_final_hold_lens_uses_fine_mesh_at_10ghz():
    trend = {'reported': {'boresight_dbi': [[1, 2, 3], [4, 5, 6], [18.5, 19.00765, 18.8]]}}
    assert visual.final_hold_text('lens', trend, [9.5e9, 1e10, 10.5e9], {'promotional_number_eligible': True}, 150, 150) == 'final: 19.0 dBi at 10 GHz, three-mesh converged'


@pytest.mark.parametrize('passed,expected', [([True]*3, 'inside the mask on three meshes'),
                                            ([True, False, True], ''), ([], ''), ([True], '')])
def test_final_hold_filter_requires_all_three_masks(passed, expected):
    assert visual.final_hold_text('filter', {'mask_each_mesh': passed}, [], {'promotional_number_eligible': True}, 200, 200) == expected


def test_fd_judge_fraction_reports_small_gradients_only():
    # A pixel whose |FD| is below 0.1 x the largest is reported, not judged:
    # its 300 % error must not fail the verdict, and the large pixel's 1 %
    # error must pass at the 5e-2 bar (pre-declaration §1.5.1).
    import _design_common as dc
    steps = (0.1, 0.05, 0.025)
    ladder = {"big": {str(h): {"fd": 1.0} for h in steps},
              "small": {str(h): {"fd": 0.01} for h in steps}}
    verdict = dc.judge_fd({"big": 1.01, "small": 0.04}, ladder, steps, 0.05, 0.05, 0.1)
    assert verdict["judged"] == ["big"]
    assert verdict["rows"]["small"]["passed"] is None
    assert verdict["all_judged_passed"] is True
    # The same small pixel above the fraction would be judged and fail.
    ladder["small"] = {str(h): {"fd": 0.2} for h in steps}
    verdict = dc.judge_fd({"big": 1.01, "small": 0.8}, ladder, steps, 0.05, 0.05, 0.1)
    assert "small" in verdict["judged"] and verdict["all_judged_passed"] is False


def test_final_hold_note_requires_gates_and_same_iterate():
    trend = {'reported': {'boresight_dbi': [[0, 0, 0], [0, 0, 0], [18.5, 19.008, 18.8]]},
             'mask_each_mesh': [True, True, True]}
    freqs = [9.5e9, 10e9, 10.5e9]
    ok = {'promotional_number_eligible': True}
    assert visual.final_hold_text('lens', trend, freqs, ok, 150, 150).startswith('final: 19.0 dBi')
    assert visual.final_hold_text('lens', trend, freqs, {'promotional_number_eligible': False}, 150, 150) == ''
    assert visual.final_hold_text('lens', trend, freqs, ok, 149, 150) == ''
    assert visual.final_hold_text('filter', trend, freqs, ok, 200, 200) == 'inside the mask on three meshes'
    assert visual.lens_label('uniform_1.85') == 'uniform slab, εr = 1.85'
