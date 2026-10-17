"""Driver contracts and one coarse field smoke; no global precision changes."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "showcase"))
import _lens_filter_common as common
import design_filter as filt
import design_lens as lens
import _visual_data as visual


def test_wide_stored_record():
    import jax.numpy as jnp
    path = Path('/Users/byungkwankim/mnt/remilab-fs/personal-workspaces/byungkwan-workspace/rfx-showcase-runs/20261004-showcase-lens/20261004T182324Z-b184fc8e-main/iterations.npz')
    if not path.is_file():
        pytest.skip(f'stored record absent: {path}')
    # Open the real target once; expected numbers are the independent float64 read.
    with np.load(path, allow_pickle=False) as stored:
        response = stored['response']
    assert response.shape == (151, 3, 73, 73)
    assert response.dtype == np.float32
    first = lens.summary(response[0], 'wide')
    np.testing.assert_allclose(first['smooth_min_dbi'], [5.01894048, 2.85652214, .51121956], atol=1e-4, rtol=0)
    np.testing.assert_allclose(first['sector_min_dbi'], [2.71386773, .38507152, -1.90578981], atol=1e-4, rtol=0)
    np.testing.assert_allclose(first['sector_max_dbi'], [12.62092062, 12.18094386, 12.35116166], atol=1e-4, rtol=0)
    np.testing.assert_allclose(first['outside_max_dbi'], [5.23780215, 5.68598373, 4.41078657], atol=1e-4, rtol=0)
    assert first['objective'] == pytest.approx(-2.79556073, abs=1e-4)
    for i, expected in [(0, -2.79556073), (150, 6.55280328)]:
        assert float(lens.loss(jnp.asarray(response[i]), 'wide')) == pytest.approx(expected, abs=1e-4)


def test_wide_region_and_constant():
    import jax.numpy as jnp
    spec = lens.objective_spec('wide')
    region = spec['regions']['C_W']
    assert spec['objective_words'] == 'Wide flat-top beam (±30°)'
    assert spec['tau_db'] == .5
    assert spec['judged'] == {'ripple_max_db': 2., 'level_min_dbi': 7.5}
    assert spec['linear_floor'] == 1e-30
    assert region['directions'] == 936
    assert region['predicate'] == 'nominal theta <= 30 degrees'
    np.testing.assert_array_equal(region['theta_indices'], np.arange(13))
    np.testing.assert_array_equal(region['phi_indices'], np.arange(72))
    assert lens.WIDE_REGION[12] and not lens.WIDE_REGION[13]
    pattern = np.full((3, 73, 73), 1e-20, np.float32)
    constants = np.array([1., 10., 100.], np.float32)
    pattern[:, :13, :72] = constants[:, None, None]
    pattern[:, :, 72] = 1e20  # Duplicated endpoint cannot change sector metrics.
    result = lens.summary(pattern, 'wide')
    np.testing.assert_allclose(result['smooth_min_dbi'], [0., 10., 20.], atol=1e-5)
    np.testing.assert_array_equal(result['ripple_db'], np.zeros(3))
    np.testing.assert_array_equal(result['meets_ripple'], [True, True, True])
    np.testing.assert_array_equal(result['meets_level'], [False, True, True])
    value = lens.loss(jnp.asarray(pattern), 'wide')
    assert value.shape == () and value.dtype == jnp.float32
    assert float(value) == pytest.approx(-10., abs=1e-5)
    pattern[:, 13, :72] = 1e20
    assert float(lens.loss(jnp.asarray(pattern), 'wide')) == float(value)
    for function in (lens.objective_spec, lens.with_objective):
        with pytest.raises(ValueError, match='unknown lens objective'):
            function('unknown')
    for function in (lens.loss, lens.summary):
        with pytest.raises(ValueError, match='unknown lens objective'):
            function(pattern, 'unknown')


def test_wide_floor_gradient():
    import jax
    import jax.numpy as jnp
    pattern = jnp.ones((3, 73, 73), dtype=jnp.float32).at[:, 12, 71].set(1e-30)
    value, gradient = jax.value_and_grad(lambda x: lens.loss(x, 'wide'))(pattern)
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    # One -300 dBi direction dominates; the remaining exponentials underflow.
    expected = 300. - .5 * np.log(936.)
    assert float(value) == pytest.approx(expected, abs=1e-4)


def test_wide_summary_power_and_rolloff():
    theta = np.r_[0., np.linspace(1e-4, np.pi - 1e-4, 73)[1:]]
    degrees = np.rad2deg(theta)
    db = np.full((3, 73, 73), 8.)
    # Linear cuts with crossings at 36/60 degrees and 33/45 degrees.
    for phi, slope in [(0, .5), (18, 1.)]:
        db[:, 13:, phi] = 8. - slope * (degrees[13:] - 30.)
    db[1, :, 18] = 8.  # Neither crossing reached.
    db[2, 13:, 0] = 4.  # Only the -3 dB crossing reached.
    pattern = 10 ** (db / 10)
    result = lens.summary(pattern, 'wide')
    np.testing.assert_allclose(result['rolloff_deg'][0], [24., 12.], atol=1e-12)
    assert np.isnan(result['rolloff_deg'][1, 1])
    assert np.isnan(result['rolloff_deg'][2, 0])
    weights = np.sin(theta)[None, :, None]
    expected = (pattern[:, :13, :72] * weights[:, :13]).sum(axis=(1, 2)) / (pattern[:, :, :72] * weights).sum(axis=(1, 2))
    np.testing.assert_allclose(result['cone_power_fraction'], expected)
    np.testing.assert_allclose(result['outside_minus_sector_min_db'], 0., atol=1e-12)
    np.testing.assert_allclose(result['boresight_dbi'], 10 * np.log10(pattern[:, 0].mean(axis=1)))
    # Both thresholds in a single interval require separate interpolations.
    db[:] = 8.
    db[:, 13:, 0] = -22.
    result = lens.summary(10 ** (db / 10), 'wide')
    np.testing.assert_allclose(result['rolloff_deg'][:, 0], .4 * (degrees[13] - degrees[12]), atol=1e-12)


@pytest.mark.parametrize('last,passed', [(-4.9, True), (-4.7, False), (np.nan, False)])
def test_wide_trial_gate(last, passed):
    from types import SimpleNamespace
    store = SimpleNamespace(rows={'objective': [-2.8, last]})
    assert common.trial_pass(lens.with_objective('wide'), store)['passed'] is passed


def test_wide_progress_and_nan_history():
    from types import SimpleNamespace
    values = [-2., -5., -8.]
    np.testing.assert_array_equal(common.frame_progress(values, 'wide'), [0., .5, 1.])
    assert common.frame_iterate(values, 'wide') == 1
    store = SimpleNamespace(rows={'objective': [-2.8, np.nan, -4.9]})
    assert not common.trial_pass(lens.with_objective('wide'), store)['passed']


def test_wide_describe_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace
    captured, meshes = [], []
    def setup(case, out, dx, precision, smoke):
        meshes.append(dx)
        return SimpleNamespace(precision=precision, smoke=smoke, default_steps=10,
                               describe=lambda: {'dx_m': dx})
    monkeypatch.setattr(common, 'setup', setup)
    monkeypatch.setattr(common._record, 'source_block', lambda *a: {})
    monkeypatch.setattr(common._record, 'write_result', lambda out, rec, files: captured.append(rec))
    assert common.main(lens, ['--stage', 'describe', '--objective', 'wide', '--out', str(tmp_path)]) == 0
    assert meshes == [1.5e-3, 1e-3, .75e-3]
    assert captured[0]['question'] == 'Wide flat-top beam (±30°)'
    assert captured[0]['objective'] == 'wide'
    stage = common.dc.load_json(tmp_path / 'stage.json')
    assert stage['regions']['C_W']['directions'] == 936
    assert stage['build_only'] is True


@pytest.mark.parametrize('stage', ['trial', 'main', 'resolve', 'frame', 'fd', 'rlw', 'describe'])
def test_wide_cli_mismatch(tmp_path, stage):
    source = tmp_path / 'source'
    source.mkdir()
    common.dc.save_json(source / 'result.json', {'objective': 'mask'})
    with pytest.raises(SystemExit, match='2'):
        common.main(lens, ['--stage', stage, '--objective', 'wide', '--record', str(source),
                           '--out', str(tmp_path / 'out')])


@pytest.mark.parametrize('failure', [None, 'fd', 'start', 'reported', 'mesh', 'ripple', 'level', 'rlw'])
def test_wide_resolve_eligibility(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    case = lens.with_objective('wide')
    source, out = tmp_path / 'source', tmp_path / 'out'
    source.mkdir()
    out.mkdir()
    common.dc.save_npz(source / 'iterations.npz', objective=[-2., -8.], eps=np.zeros((2, *lens.SHAPE)))
    common.dc.save_json(source / 'record_length.json', {'n_steps': 10, 'dt_s': 1., 'passed': failure != 'start'})
    common.dc.save_json(source / 'reported.json', {'reported': {'settled': failure != 'reported'}})
    common.dc.save_json(source / 'fd_float32.json', {'judgement': {'all_judged_passed': failure != 'fd'}})
    for label in ('start', 'reported'):
        common.dc.save_json(source / f'rlw_{label}.json', {'all_passed': failure != 'rlw'})
    monkeypatch.setattr(common, 'setup', lambda *a: SimpleNamespace(grid=SimpleNamespace(dt=1.)))
    monkeypatch.setattr(common, 'record', lambda *a: None)
    calls = []
    def evaluate(case, model, designs, n, directory, label):
        assert list(designs) == ['reported']
        i = len(calls)
        calls.append(i)
        row = dict(sector_min_dbi=np.full(3, 8. + [0., .2, .6 if failure == 'mesh' else .3][i]),
                   sector_max_dbi=np.full(3, 9.), outside_max_dbi=np.full(3, 4.),
                   meets_ripple=[True, not (failure == 'ripple' and i == 2), True],
                   meets_level=[True, True, not (failure == 'level' and i == 2)], settled=True)
        return {'reported': row}
    monkeypatch.setattr(common, 'evaluate', evaluate)
    common.resolved(case, SimpleNamespace(out=out, precision='float32'), source)
    result = common.dc.load_json(out / 'eligibility.json')
    assert result['meets_wide'] is (failure is None)
    assert 'meets_mask' not in result
    trend = common.dc.load_json(out / 'mesh_trend.json')
    assert set(trend) == {'sector_min_dbi', 'sector_max_dbi', 'outside_max_dbi'}
    for key, limit in [('sector_min_dbi', .3), ('sector_max_dbi', .3), ('outside_max_dbi', 1.)]:
        assert trend[key]['limit_db'] == limit
        assert np.shape(trend[key]['values_dbi']) == (3, 3)


@pytest.mark.parametrize('last,passed', [(-5.8, True), (-5.8001, True), (-5.7999, False), (np.nan, False)])
def test_fan_trial_gate(last, passed):
    from types import SimpleNamespace
    store = SimpleNamespace(rows={'objective': [-2.8, last]})
    assert common.trial_pass(lens.with_objective('fan'), store)['passed'] is passed


def test_fan_progress_and_nan_history():
    from types import SimpleNamespace
    values = [-2., -5., -8.]
    np.testing.assert_array_equal(common.frame_progress(values, 'fan'), [0., .5, 1.])
    assert common.frame_iterate(values, 'fan') == 1
    store = SimpleNamespace(rows={'objective': [-2.8, np.nan, -6.]})
    assert not common.trial_pass(lens.with_objective('fan'), store)['passed']


def test_fan_describe_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace
    captured, meshes = [], []
    def setup(case, out, dx, precision, smoke):
        meshes.append(dx)
        return SimpleNamespace(precision=precision, smoke=smoke, default_steps=10,
                               describe=lambda: {'dx_m': dx})
    monkeypatch.setattr(common, 'setup', setup)
    monkeypatch.setattr(common._record, 'source_block', lambda *a: {})
    monkeypatch.setattr(common._record, 'write_result', lambda out, rec, files: captured.append(rec))
    assert common.main(lens, ['--stage', 'describe', '--objective', 'fan', '--out', str(tmp_path)]) == 0
    assert meshes == [1.5e-3, 1e-3, .75e-3]
    assert captured[0]['question'] == 'Fan beam (flat over ±30° in one plane, narrow in the other)'
    assert captured[0]['objective'] == 'fan'
    stage = common.dc.load_json(tmp_path / 'stage.json')
    assert stage['regions']['S_N']['directions'] == 78
    assert stage['build_only'] is True


@pytest.mark.parametrize('stage', ['trial', 'main', 'resolve', 'frame', 'fd', 'rlw', 'describe'])
def test_fan_cli_mismatch(tmp_path, stage):
    source = tmp_path / 'source'
    source.mkdir()
    common.dc.save_json(source / 'result.json', {'objective': 'mask'})
    with pytest.raises(SystemExit, match='2'):
        common.main(lens, ['--stage', stage, '--objective', 'fan', '--record', str(source),
                           '--out', str(tmp_path / 'out')])


@pytest.mark.parametrize('failure', [None, 'fd', 'start', 'reported', 'mesh', 'ripple', 'level', 'narrow', 'rlw'])
def test_fan_resolve_eligibility(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    case = lens.with_objective('fan')
    source, out = tmp_path / 'source', tmp_path / 'out'
    source.mkdir()
    out.mkdir()
    common.dc.save_npz(source / 'iterations.npz', objective=[-2., -8.], eps=np.zeros((2, *lens.SHAPE)))
    common.dc.save_json(source / 'record_length.json', {'n_steps': 10, 'dt_s': 1., 'passed': failure != 'start'})
    common.dc.save_json(source / 'reported.json', {'reported': {'settled': failure != 'reported'}})
    common.dc.save_json(source / 'fd_float32.json', {'judgement': {'all_judged_passed': failure != 'fd'}})
    for label in ('start', 'reported'):
        common.dc.save_json(source / f'rlw_{label}.json', {'all_passed': failure != 'rlw'})
    monkeypatch.setattr(common, 'setup', lambda *a: SimpleNamespace(grid=SimpleNamespace(dt=1.)))
    monkeypatch.setattr(common, 'record', lambda *a: None)
    calls = []
    def evaluate(case, model, designs, n, directory, label):
        assert list(designs) == ['reported']
        i = len(calls)
        calls.append(i)
        row = dict(strip_min_dbi=np.full(3, 8. + [0., .2, .6 if failure == 'mesh' else .3][i]),
                   strip_max_dbi=np.full(3, 9.), outside_max_dbi=np.full(3, 4.), narrow_hpbw_deg=np.full(3, 20.),
                   meets_narrow=[not (failure == 'narrow' and i == 2), True, True],
                   meets_ripple=[True, not (failure == 'ripple' and i == 2), True],
                   meets_level=[True, True, not (failure == 'level' and i == 2)], settled=True)
        return {'reported': row}
    monkeypatch.setattr(common, 'evaluate', evaluate)
    common.resolved(case, SimpleNamespace(out=out, precision='float32'), source)
    result = common.dc.load_json(out / 'eligibility.json')
    assert result['meets_fan'] is (failure is None)
    assert 'meets_mask' not in result
    for condition in ('ripple', 'level', 'narrow'):
        assert np.shape(result[f'meets_{condition}_each_mesh']) == (3, 3)
    trend = common.dc.load_json(out / 'mesh_trend.json')
    assert set(trend) == {'strip_min_dbi', 'strip_max_dbi', 'narrow_hpbw_deg', 'outside_max_dbi'}
    for key, limit in [('strip_min_dbi', .3), ('strip_max_dbi', .3), ('outside_max_dbi', 1.), ('narrow_hpbw_deg', 1.)]:
        assert trend[key]['limit_db'] == limit
        assert np.shape(trend[key]['values_dbi']) == (3, 3)


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


def test_frame_iterate_lens():
    # Half progress is exactly at 2; later reversals and ties do not change it.
    assert common.frame_iterate([2., 1., 0., 1., -2., 0.], "lens") == 2


def test_frame_iterate_filter():
    # Linear progress would select 1, log progress first reaches half at 2.
    assert common.frame_iterate([256., 64., 16., 32., 1.], "filter") == 2


@pytest.mark.parametrize("kind,values", [
    ("lens", [1., np.nan]), ("filter", [2., np.inf]),
    ("lens", [1., 1.]), ("filter", [1., 2.]),
    ("lens", []), ("lens", [1.]), ("lens", [[2., 1.]]),
    ("filter", [2., 0.]), ("filter", [2., -1.]),
    ("unknown", [2., 1.]),
])
def test_frame_iterate_refuses(kind, values):
    with pytest.raises(ValueError):
        common.frame_iterate(values, kind)


def test_main_iterations_schedule(tmp_path, monkeypatch):
    pytest.importorskip("optax")  # optional extra "optimization"; CI and the local gate install .[dev] only
    from types import SimpleNamespace
    import jax.numpy as jnp
    case = SimpleNamespace(UPPER=2.7, LR=(.15, .015), ITERATIONS=200,
                           FREQS=np.array([1.]), loss=lambda x: jnp.sum(x ** 2))
    model = SimpleNamespace(response_fn=lambda n: lambda e: e)
    monkeypatch.setattr(common, "record", lambda *a: None)
    store = common.optimize(case, model, np.array([1.5], dtype=np.float32),
                            10, 4, tmp_path, "main")
    rates = np.asarray(store.rows["lr"])
    assert len(rates) == 5
    assert rates[0] == pytest.approx(.15)
    assert rates[3] == pytest.approx(.015)
    assert rates[1] == pytest.approx(.11625)
    assert common.cosine_schedule(3, *case.LR, case.ITERATIONS) > .14


def test_reuse_start_witnesses(tmp_path):
    source, out = tmp_path / "trial", tmp_path / "main"
    origin = source / "S1"
    origin.mkdir(parents=True)
    files = {"fd_float32.json": {"judgement": {"roundoff": ["p"]}},
             "fd_float64.json": {"judgement": {"all_judged_passed": True}},
             "x64_needed.json": {"pixels": ["p"]},
             "rlw_start.json": {"all_passed": True, "n_steps": 15252}}
    for name, data in files.items():
        common.dc.save_json(origin / name, data)
    assert common.reuse_start_witnesses(source, "S1", out) == files["rlw_start.json"]
    for name in files:
        assert (out / name).read_bytes() == (origin / name).read_bytes()
    provenance = common.dc.load_json(out / "start_witness_reuse.json")
    assert provenance["reused"] is True and provenance["start"] == "S1"
    assert provenance["source"] == str(origin.resolve())
    assert set(provenance["files"]) == set(files)


@pytest.mark.parametrize("y_uniform", [False, True])
def test_main_start_and_iterations_cli(tmp_path, monkeypatch, y_uniform):
    from types import SimpleNamespace
    import jax
    source, out = tmp_path / "trial", tmp_path / "main"
    source.mkdir()
    common.dc.save_json(source / "trial_selection.json",
                        {"passed": True, "chosen_start": "S2"})
    common.dc.save_json(source / "record_length.json", {"n_steps": 10, "dt_s": 1.})
    model = SimpleNamespace(grid=SimpleNamespace(dt=1.), default_steps=10)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(common, "setup", lambda *a: model)
    prepared, reused, counts = [], [], []
    def prepare(case, model, names, n, out, allow_extension):
        prepared.append((names, allow_extension))
        return n
    monkeypatch.setattr(common, "prepare", prepare)
    def reuse(source, name, out):
        reused.append(name)
        return {"all_passed": True}
    monkeypatch.setattr(common, "reuse_start_witnesses", reuse)
    def witness(case, model, e, n, out, label):
        assert label == "reported"
        return {"all_passed": True}
    monkeypatch.setattr(common, "rlw", witness)
    monkeypatch.setattr(common, "fd", lambda *a: pytest.fail("FD must be reused"))
    def optimize(case, model, e, n, count, out, stage, **kwargs):
        assert kwargs == ({"y_uniform": True} if y_uniform else {})
        np.testing.assert_array_equal(e, filt.starts()["S1"])
        counts.append(count)
        return None
    monkeypatch.setattr(common, "optimize", optimize)
    monkeypatch.setattr(common, "trial_pass", lambda *a: {})
    monkeypatch.setattr(common, "selected_design", lambda *a: (filt.starts()["S1"], 0))
    monkeypatch.setattr(common, "evaluate", lambda *a, **kw: {"reported": {"settled": True, "n_steps": 10}})
    records = []
    monkeypatch.setattr(common, "record", lambda *a: records.append(a[-1]))
    assert common.main(filt, ["--stage", "main", "--record", str(source), "--out",
                              str(out), "--start", "S1", "--iterations", "400"]
                       + (["--y-uniform"] if y_uniform else [])) == 0
    assert prepared == [(["S1"], False)]
    assert reused == ["S1"] and counts == [400]
    assert records[-1]["chosen_start"] == "S1"
    assert records[-1]["start_witnesses_reused"] is True
    assert records[-1].get("y_uniform", False) is y_uniform


def test_frame_fallback_and_gradient(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source, out = tmp_path / "main", tmp_path / "frame"
    source.mkdir()
    out.mkdir()
    objectives = np.arange(50., -1., -1.)
    common.dc.save_npz(source / "iterations.npz", objective=objectives,
                       eps=np.arange(51.)[:, None])
    common.dc.save_json(source / "record_length.json", {"n_steps": 123, "dt_s": .25})
    # A reported-design extension must not change the descent frame's record.
    common.dc.save_json(source / "reported_record_length.json", {"n_steps": 246, "dt_s": .25})
    monkeypatch.setattr(common, "setup", lambda *a: SimpleNamespace(grid=SimpleNamespace(dt=.25)))
    monkeypatch.setattr(common, "record", lambda *a: None)
    tried = []
    def witness(case, model, e, n, out, label):
        assert n == 123
        k = int(e[0])
        tried.append(k)
        return {"passed": k == 0, "all_passed": False,
                "grad": {"": np.array([[42.]])}}
    monkeypatch.setattr(common, "rlw", witness)
    common.frame(lens, SimpleNamespace(out=out, precision="float32"), source)
    result = common.dc.load_json(out / "frame.json")
    assert tried == [25, 15, 5, 0]
    assert result["k_star"] == 25 and result["k_used"] == 0
    assert result["progress"][25] == .5
    assert result["objective_values"] == objectives.tolist()
    assert set(result["witnesses"]) == {"25", "15", "5", "0"}
    with np.load(out / "frame_gradient.npz") as data:
        np.testing.assert_array_equal(data["grad_eps"], [42.])
        assert data["iteration"] == 0 and data["n_steps"] == 123


def _toy_filter(monkeypatch):
    from types import SimpleNamespace
    import jax.numpy as jnp
    weights = jnp.arange(1, 161, dtype=jnp.float32).reshape(32, 5) / 160
    case = SimpleNamespace(CASE="filter", UPPER=filt.UPPER, LR=filt.LR,
                           ITERATIONS=filt.ITERATIONS, FREQS=filt.FREQS,
                           loss=lambda e: jnp.sum(weights * (e - 2) ** 2))
    model = SimpleNamespace(response_fn=lambda n: lambda e: e)
    monkeypatch.setattr(common, "record", lambda out, case, model, stage, details:
                        common.dc.save_json(out / "stage.json", details))
    return case, model


@pytest.mark.parametrize("count", [2, 7])
def test_y_uniform_broadcast_and_storage(tmp_path, monkeypatch, count):
    pytest.importorskip("optax")  # optional extra "optimization"; CI and the local gate install .[dev] only
    case, model = _toy_filter(monkeypatch)
    start = np.broadcast_to(np.linspace(1.2, 6., 32, dtype=np.float32)[:, None], (32, 5))
    common.optimize(case, model, start, 10, count, tmp_path, "main", y_uniform=True)
    with np.load(tmp_path / "iterations.npz") as data:
        for key in ("eps", "psi"):
            assert data[key].shape == (count + 1, 32, 5)
            np.testing.assert_array_equal(data[key], np.repeat(data[key][:, :, :1], 5, axis=2))
        assert data["adam_mu"].shape == data["adam_nu"].shape == (count + 1, 32)
        assert np.any(data["eps"][0] != data["eps"][-1])
        chosen, index = common.selected_design(tmp_path)
        np.testing.assert_array_equal(chosen, data["eps"][index])
    for filename in ("selection.json", "stage.json"):
        assert common.dc.load_json(tmp_path / filename)["y_uniform"] is True


def test_y_uniform_gradient_column_sum(tmp_path, monkeypatch):
    import jax
    import jax.numpy as jnp
    optax = pytest.importorskip("optax")  # optional extra "optimization"; CI and the local gate install .[dev] only
    from types import SimpleNamespace
    case, model = _toy_filter(monkeypatch)
    original_adam = optax.adam
    seen = []
    def adam(schedule):
        opt = original_adam(schedule)
        def update(gradient, state):
            seen.append(np.asarray(gradient))
            return opt.update(gradient, state)
        return SimpleNamespace(init=opt.init, update=update)
    monkeypatch.setattr(optax, "adam", adam)
    store = common.optimize(case, model, filt.starts()["S2"], 10, 4, tmp_path,
                            "main", y_uniform=True)
    for i, actual in enumerate(seen):
        psi = jnp.asarray(store.rows["psi"][i])
        fraction = jax.nn.sigmoid(psi)
        g = jnp.asarray(store.rows["grad_eps"][i])
        expected = jnp.sum(g * (case.UPPER - 1) * fraction * (1 - fraction), axis=1)
        np.testing.assert_array_equal(actual, expected)
        def objective(latent):
            eps = 1 + (case.UPPER - 1) * jax.nn.sigmoid(latent)
            return case.loss(jnp.broadcast_to(eps[:, None], (32, 5)))
        np.testing.assert_allclose(actual, jax.grad(objective)(psi[:, 0]), rtol=3e-7)
    assert len(seen) == 4


@pytest.mark.parametrize("case,stage", [(lens, "main")] + [
    (filt, stage) for stage in ("timing", "fd", "rlw", "trial", "resolve", "baselines", "describe", "frame")])
def test_y_uniform_cli_refuses_case_and_stage(tmp_path, monkeypatch, capsys, case, stage):
    monkeypatch.setattr(common, "setup", lambda *a: pytest.fail("must reject before setup"))
    with pytest.raises(SystemExit) as error:
        common.main(case, ["--stage", stage, "--out", str(tmp_path), "--y-uniform"])
    assert error.value.code == 2
    assert "--y-uniform requires filter --stage main" in capsys.readouterr().err


def test_y_uniform_refuses_nonuniform_start(tmp_path, monkeypatch):
    case, model = _toy_filter(monkeypatch)
    start = filt.starts()["S1"]
    start[3, 2] = np.nextafter(start[3, 2], np.float32(2.))
    with pytest.raises(ValueError, match="requires each pixel row"):
        common.optimize(case, model, start, 10, 2, tmp_path, "main", y_uniform=True)
    assert not (tmp_path / "iterations.npz").exists()


def test_default_optimizer_bit_identical(tmp_path, monkeypatch):
    import jax
    import jax.numpy as jnp
    optax = pytest.importorskip("optax")  # optional extra "optimization"; CI and the local gate install .[dev] only
    case, model = _toy_filter(monkeypatch)
    start = np.linspace(1.2, 6., 160, dtype=np.float32).reshape(32, 5)
    count = 4
    # Original per-pixel update, including initialization and arithmetic order.
    p = np.clip((np.asarray(start, dtype=float) - 1) / (case.UPPER - 1), 1e-3, 1 - 1e-3)
    psi = jnp.asarray(np.log(p / (1 - p)), dtype=jnp.float32)
    def schedule(i):
        return common.cosine_schedule(i, *case.LR, count, xp=jnp)
    opt = optax.adam(schedule)
    state = opt.init(psi)
    vg = jax.jit(jax.value_and_grad(lambda eps: (case.loss(eps), eps), has_aux=True))
    expected = []
    for i in range(count + 1):
        fraction = jax.nn.sigmoid(psi)
        eps = 1 + (case.UPPER - 1) * fraction
        (value, response), g = vg(eps)
        expected.append(dict(eps=eps, psi=psi, objective=value, response=response,
                             grad_eps=g, lr=schedule(i), **common.dc.adam_state_arrays(state)))
        if i != count:
            updates, state = opt.update(g * (case.UPPER - 1) * fraction * (1 - fraction), state)
            psi = optax.apply_updates(psi, updates)
    store = common.optimize(case, model, start, 10, count, tmp_path, "main")
    for i, row in enumerate(expected):
        for key, value in row.items():
            assert np.asarray(store.rows[key][i]).tobytes() == np.asarray(value).tobytes(), (i, key)
    assert "y_uniform" not in common.dc.load_json(tmp_path / "selection.json")


def test_y_uniform_cli_refuses_nonuniform_start(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import jax
    source, out = tmp_path / "trial", tmp_path / "main"
    source.mkdir()
    common.dc.save_json(source / "trial_selection.json",
                        {"passed": True, "chosen_start": "S1"})
    starts = filt.starts()
    starts["S1"][0, 4] = 1.6
    monkeypatch.setattr(filt, "starts", lambda: starts)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(common, "setup", lambda *a: SimpleNamespace(default_steps=10))
    monkeypatch.setattr(common, "prepare", lambda *a, **kw: pytest.fail("must reject before prepare"))
    with pytest.raises(ValueError, match="requires each pixel row"):
        common.main(filt, ["--stage", "main", "--record", str(source),
                           "--out", str(out), "--y-uniform"])


def test_frame_video_iterates_witnessed_once_each(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source, out = tmp_path / "main", tmp_path / "frame"
    source.mkdir()
    out.mkdir()
    common.dc.save_npz(source / "iterations.npz", objective=np.arange(50., -1., -1.),
                       eps=np.arange(51.)[:, None])
    common.dc.save_json(source / "record_length.json", {"n_steps": 123, "dt_s": .25})
    monkeypatch.setattr(common, "setup", lambda *a: SimpleNamespace(grid=SimpleNamespace(dt=.25)))
    monkeypatch.setattr(common, "record", lambda *a: None)
    tried = []
    def witness(case, model, e, n, out, label):
        k = int(e[0])
        tried.append(k)
        return {"passed": k != 7, "worst": float(k), "grad": {"": np.array([[1.]])}}
    monkeypatch.setattr(common, "rlw", witness)
    common.frame(lens, SimpleNamespace(out=out, precision="float32", iterates=[0, 7, 25, 50]), source)
    result = common.dc.load_json(out / "frame.json")
    # k* = 25 passes first; the film list reuses it and witnesses the rest once.
    assert tried == [25, 0, 7, 50]
    assert result["video_witnesses"] == {"0": {"passed": True, "worst": 0.0}, "7": {"passed": False, "worst": 7.0},
                                         "25": {"passed": True, "worst": 25.0}, "50": {"passed": True, "worst": 50.0}}


@pytest.mark.parametrize("extra", [[], ["--iterates", ""], ["--iterates", "-1"],
    ["--iterates", "0", "--start", "S1"], ["--iterates", "0", "--reported"],
    ["--iterates", "0", "--precision", "float64"], ["--iterates", "0", "--iterations", "2"]])
def test_field_cli_validation(tmp_path, extra):
    with pytest.raises(SystemExit, match="2"):
        common.main(filt, ["--stage", "field", "--record", str(tmp_path / "source"),
                           "--out", str(tmp_path / "out"), *extra])


@pytest.mark.parametrize("same", [False, True])
def test_field_requires_distinct_record(tmp_path, same):
    argv = ["--stage", "field", "--iterates", "0", "--out", str(tmp_path)]
    if same:
        argv += ["--record", str(tmp_path)]
    with pytest.raises(SystemExit, match="2"):
        common.main(filt, argv)


def test_field_filter_smoke(tmp_path, monkeypatch):
    from types import SimpleNamespace
    model = filt.Model(filt.SMOKE_DX, smoke=True)
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    out.mkdir()
    common.dc.save_json(source / "result.json", {"id": "design-filter", "model": model.describe()})
    common.dc.save_json(source / "record_length.json", {"n_steps": 240, "dt_s": model.grid.dt})
    common.dc.save_npz(source / "iterations.npz", eps=filt.starts()["S1"][None])
    # Count forward calls and reject any accidental response/gradient solve.
    from rfx import Simulation
    forward = Simulation.forward
    calls = []
    def counted(self, **kwargs):
        calls.append(kwargs["n_steps"])
        return forward(self, **kwargs)
    monkeypatch.setattr(Simulation, "forward", counted)
    monkeypatch.setattr(filt.Model, "response_fn", lambda *a, **k: pytest.fail("response solve"))
    args = SimpleNamespace(out=out, precision="float32", smoke=True, iterates=[0])
    assert common.main(filt, ["--stage", "field", "--record", str(source),
                              "--iterates", "0", "--out", str(out), "--smoke"]) == 0
    assert calls == [240]
    with np.load(out / "field_0.npz", allow_pickle=False) as data:
        assert set(data.files) == {"iterate", "freqs_hz", "planes", "drive", "n_steps", "dx_m",
                                   "xy_u_m", "xy_v_m", "xy_axes", "xy_offset_m", "xy_E", "xy_eps"}
        assert data["planes"].tolist() == ["xy"]
        assert data["xy_axes"].item() == "xy"
        assert data["xy_E"].shape == (3, 3, len(data["xy_u_m"]), len(data["xy_v_m"]))
        assert data["xy_eps"].shape == data["xy_E"].shape[2:]
        assert data["xy_E"].dtype == np.complex64
        assert data["xy_eps"].dtype == np.float32
        assert np.isfinite(data["xy_E"]).all()
        assert np.max(abs(data["xy_E"])) > 0
        np.testing.assert_allclose(data["freqs_hz"], [10.3e9, 8.6e9, 12e9])
        np.testing.assert_allclose(data["xy_u_m"], [model.grid.node_of(0, i) for i in range(model.grid.shape[0])])
        assert data["xy_u_m"][0] < 0  # CPML coordinates retain their physical origin.
    details = common.dc.load_json(out / "field.json")
    assert details["files"] == ["field_0.npz"]
    assert details["settling"]["field_monitor_tail_judged"] is False
    assert len(details["iterates"]["0"]["peak_abs_E"]["xy"]) == 3
    args.iterates = [1]
    with pytest.raises(ValueError, match="outside stored designs"):
        common.field(filt, args, source)
    assert calls == [240]


def _aim_pattern(objective):
    db = np.full((3, 73, 73), -3. if objective == 'mask' else -1.)
    if objective == 'mask':
        db[:, 0, :] = 19.
    else:
        db[:, lens.REGIONS['C'], :] = 13.5
    return 10 ** (db / 10)


@pytest.mark.parametrize('objective,region,offset', [('mask', 'S', 3), ('flattop', 'O', 1)])
def test_lens_extra_objective_answers(objective, region, offset):
    import jax
    import jax.numpy as jnp
    pattern = _aim_pattern(objective)
    assert float(lens.loss(jnp.asarray(pattern), objective)) == pytest.approx(0, abs=2e-7)
    row = np.flatnonzero(lens.REGIONS[region])[0]
    pattern[0, row, 7] = 10 ** ((2 - offset) / 10)
    n = lens.REGIONS[region].sum() * 72
    expected = (np.sqrt(2 ** 4 / n + .1 ** 4) - .1 ** 2) / 3
    assert float(lens.loss(jnp.asarray(pattern), objective)) == pytest.approx(expected, rel=2e-5)
    gradient = np.asarray(jax.grad(lambda x: lens.loss(x, objective))(jnp.asarray(pattern)))
    assert np.isfinite(gradient).all() and gradient[0, row, 7] != 0
    original = float(lens.loss(jnp.asarray(pattern), objective))
    pattern[:, row, 72] = 1e9
    assert float(lens.loss(jnp.asarray(pattern), objective)) == original
    assert lens.summary(pattern, objective)['mask_met'] is False


def test_lens_region_coordinates_and_counts():
    assert {k: int(v.sum() * 72) for k, v in lens.REGIONS.items()} == {'S': 4536, 'C': 432, 'O': 4320}
    assert np.flatnonzero(lens.REGIONS['S'])[0] == 10
    assert np.flatnonzero(lens.REGIONS['O'])[0] == 13
    assert np.flatnonzero(lens.REGIONS['C'])[-1] == 5
    assert np.rad2deg(lens.THETA[5]) > 12.5


@pytest.mark.parametrize('objective', ['mask', 'flattop'])
def test_lens_mask_verdict_and_extrema(objective):
    pattern = _aim_pattern(objective)
    s = lens.summary(pattern, objective)
    assert s['mask_met'] and np.all(s['mask_met_per_frequency'])
    region = 'S' if objective == 'mask' else 'O'
    row = np.flatnonzero(lens.REGIONS[region])[1]
    pattern[1, row, 4] = 10 ** ((-1.9 if objective == 'mask' else .1) / 10)
    s = lens.summary(pattern, objective)
    assert s['mask_met_per_frequency'].tolist() == [True, False, True]
    assert s['outer_max_theta_rad'][1] == lens.THETA[row]
    assert s['outer_max_phi_rad'][1] == lens.PHI[4]
    pattern = _aim_pattern(objective)
    if objective == 'mask':
        pattern[2, 0, :] = 10 ** (17.9 / 10)
    else:
        pattern[2, 2, 5] = 10 ** (15.1 / 10)
        assert lens.summary(pattern, objective)['window_max_dbi'][2] == pytest.approx(15.1)
    assert lens.summary(pattern, objective)['mask_met_per_frequency'].tolist() == [True, True, False]


@pytest.mark.parametrize('objective', ['mask', 'flattop'])
def test_lens_zero_progress(objective):
    values = [10., 1., 0., 2.]
    transformed = np.log(np.asarray(values) + .001)
    expected = (transformed[0] - transformed) / (transformed[0] - transformed[2])
    np.testing.assert_array_equal(common.frame_progress(values, objective), expected)
    assert common.frame_iterate(values, objective) == 2


@pytest.mark.parametrize('stage', ['fd', 'rlw', 'trial', 'main', 'resolve', 'frame', 'baselines', 'describe', 'timing', 'field'])
def test_lens_objective_mismatch_refused(tmp_path, monkeypatch, stage):
    source = tmp_path / 'source'
    source.mkdir()
    common.dc.save_json(source / 'result.json', {'objective': 'mask'})
    monkeypatch.setattr(common, 'setup', lambda *a: pytest.fail('must reject before setup'))
    with pytest.raises(SystemExit, match='2'):
        common.main(lens, ['--stage', stage, '--record', str(source), '--out', str(tmp_path / 'out'), '--objective', 'flattop'])
    assert not (tmp_path / 'out').exists()


def test_lens_boresight_original_arithmetic():
    import jax.numpy as jnp
    response = np.random.default_rng(4).uniform(.001, 100., (3, 73, 73)).astype(np.float32)
    expected = -jnp.mean(jnp.log(jnp.maximum(jnp.asarray(response)[:, 0, :].mean(axis=1), 1e-30)))
    assert np.asarray(lens.loss(jnp.asarray(response))).tobytes() == np.asarray(expected).tobytes()
    np.testing.assert_array_equal(lens.summary(response)['boresight_dbi'],
                                  10 * np.log10(np.maximum(response[:, 0, :].mean(axis=1), 1e-30)))
    assert lens.with_objective('boresight').objective_metadata == {'objective': 'boresight'}


def test_lens_window_single_violation():
    import jax.numpy as jnp
    pattern = _aim_pattern('flattop')
    pattern[0, 2, 8] = 10 ** (16.5 / 10)
    assert float(lens.loss(jnp.asarray(pattern), 'flattop')) == pytest.approx(4 / (432 * 3), rel=2e-5)


@pytest.mark.parametrize('objective', ['mask', 'flattop'])
@pytest.mark.parametrize('failure', [None, 'fd', 'start', 'reported', 'mesh', 'mask'])
def test_lens_resolve_eligibility(tmp_path, monkeypatch, objective, failure):
    from types import SimpleNamespace
    case = lens.with_objective(objective)
    source, out = tmp_path / 'source', tmp_path / 'out'
    source.mkdir()
    out.mkdir()
    common.dc.save_npz(source / 'iterations.npz', objective=[1., 0.], eps=np.zeros((2, *lens.SHAPE)))
    common.dc.save_json(source / 'record_length.json', {'n_steps': 10, 'dt_s': 1., 'passed': failure != 'start'})
    common.dc.save_json(source / 'reported.json', {'reported': {'settled': failure != 'reported'}})
    common.dc.save_json(source / 'fd_float32.json', {'judgement': {'all_judged_passed': failure != 'fd'}})
    for label in ('start', 'reported'):
        common.dc.save_json(source / f'rlw_{label}.json', {'all_passed': True})
    monkeypatch.setattr(common, 'setup', lambda *a: SimpleNamespace(grid=SimpleNamespace(dt=1.)))
    monkeypatch.setattr(common, 'record', lambda *a: None)
    calls = []
    def evaluate(case, model, designs, n, directory, label):
        assert list(designs) == ['reported']
        i = len(calls)
        calls.append(i)
        row = case.summary(_aim_pattern(objective))
        row['boresight_dbi'] = np.array([19., 19., 19.]) + [0., .2, .3 if failure != 'mesh' else .6][i]
        row['settled'] = True
        if failure == 'mask' and i == 2:
            row['mask_met'] = False
            row['mask_met_per_frequency'] = [True, False, True]
        return {'reported': row}
    monkeypatch.setattr(common, 'evaluate', evaluate)
    common.resolved(case, SimpleNamespace(out=out, precision='float32'), source)
    result = common.dc.load_json(out / 'eligibility.json')
    assert result['meets_mask'] is (failure is None)
    trend = common.dc.load_json(out / 'mesh_trend.json')
    assert trend['outer_max_dbi']['limit_db'] == 1.
    assert trend['boresight_dbi']['limit_db'] == .3
    assert np.shape(trend['boresight_dbi']['values_dbi']) == (3, 3)
    assert ('window_min_dbi' in trend) is (objective == 'flattop')


def test_lens_boresight_smoke_original(tmp_path):
    import jax.numpy as jnp
    model = lens.Model(lens.SMOKE_DX, smoke=True)
    response = model.response_fn(model.default_steps)(lens.starts()['primary'])
    original = -jnp.mean(jnp.log(jnp.maximum(response[:, 0, :].mean(axis=1), 1e-30)))
    assert np.asarray(lens.with_objective('boresight').loss(response)).tobytes() == np.asarray(original).tobytes()
    d = np.asarray(response)
    bore = 10 * np.log10(np.maximum(d[:, 0, :].mean(axis=1), 1e-30))
    bound = 10 * np.log10(4 * np.pi * .09 ** 2 / (299792458. / lens.FREQS) ** 2)
    expected = {'boresight_dbi': bore, 'mean_boresight_dbi': float(bore.mean()),
                'aperture_bound_dbi': bound, 'above_aperture_bound': bore > bound}
    for key, value in expected.items():
        assert np.asarray(lens.summary(response)[key]).tobytes() == np.asarray(value).tobytes()


@pytest.mark.parametrize('index,objective,lo,hi,narrow,wide', [
    (0, 8.087480770, [3.7335, 1.6399, -1.3261], [12.6209, 12.1809, 12.3512],
     [22.750, 40.684, 39.463], [47.679, 47.056, 45.445]),
    (150, 27.239873978, [-1.7962, -5.2562, -7.9461], [18.5735, 19.0357, 18.8805],
     [17.703, 16.588, 15.517], [16.433, 15.273, 14.494]),
])
def test_fan_stored_record(index, objective, lo, hi, narrow, wide):
    import jax.numpy as jnp
    path = Path('/Users/byungkwankim/mnt/remilab-fs/personal-workspaces/byungkwan-workspace/rfx-showcase-runs/20261004-showcase-lens/20261004T182324Z-b184fc8e-main/iterations.npz')
    if not path.is_file():
        pytest.skip(f'stored record absent: {path}')
    with np.load(path, allow_pickle=False) as stored:
        response = stored['response'][index]
    result = lens.summary(response, 'fan')
    for key, expected in [('strip_min_dbi', lo), ('strip_max_dbi', hi)]:
        np.testing.assert_allclose(result[key], expected, atol=2e-3, rtol=0)
    for key, expected in [('narrow_hpbw_deg', narrow), ('wide_width_deg', wide)]:
        np.testing.assert_allclose(result[key], expected, atol=5e-3, rtol=0)
    assert result['objective'] == pytest.approx(objective, abs=1e-4)
    assert float(lens.loss(jnp.asarray(response), 'fan')) == pytest.approx(objective, abs=1e-4)


def test_fan_region_and_constant(monkeypatch):
    import jax.numpy as jnp
    monkeypatch.setattr(lens, 'WIDE_HALF_DEG', 60)
    spec = lens.objective_spec('fan')
    region = spec['regions']['S_N']
    assert spec['tau_db'] == .2
    assert spec['linear_floor'] == 1e-30
    assert spec['judged'] == {'ripple_max_db': 2., 'level_min_dbi': 11., 'narrow_hpbw_max_deg': 23.}
    assert region['directions'] == 78
    np.testing.assert_array_equal(lens.FAN_ROWS, np.arange(13))
    np.testing.assert_array_equal(lens.FAN_COLS, [17, 18, 19, 53, 54, 55])
    np.testing.assert_array_equal(region['theta_indices'], np.arange(13))
    np.testing.assert_array_equal(region['phi_indices'], [17, 18, 19, 53, 54, 55])
    np.testing.assert_array_equal(region['theta_rad'], lens.THETA[:13])
    pattern = np.full((3, 73, 73), 10 ** (12. / 10), np.float32)
    result = lens.summary(pattern, 'fan')
    np.testing.assert_array_equal(result['ripple_db'], np.zeros(3))
    for key in ('strip_min_dbi', 'strip_max_dbi', 'smooth_min_dbi', 'smooth_max_dbi'):
        np.testing.assert_allclose(result[key], 12., atol=1e-6, rtol=0)
        assert result[key].dtype == np.float64
    assert np.isnan(result['narrow_hpbw_deg']).all()
    assert np.isnan(result['wide_width_deg']).all()
    assert result['meets_ripple'].all() and result['meets_level'].all()
    assert not result['meets_narrow'].any()
    assert result['objective'] == pytest.approx(-12., abs=1e-5)
    assert float(lens.loss(jnp.asarray(pattern), 'fan')) == pytest.approx(-12., abs=1e-5)
    # Independently paint the declared strip; no FAN_COLS helper in the fixture.
    pattern[:] = 1e-20
    pattern[:, :13, [17, 18, 19, 53, 54, 55]] = 10.
    pattern[:, :, 72] = 1e20
    assert float(lens.loss(jnp.asarray(pattern), 'fan')) == pytest.approx(-10., abs=1e-5)
    np.testing.assert_allclose(lens.summary(pattern, 'fan')['strip_min_dbi'], 10.)


def test_fan_floor_gradient():
    import jax
    import jax.numpy as jnp
    pattern = jnp.ones((3, 73, 73), dtype=jnp.float32).at[:, 12, 55].set(0.)
    value, gradient = jax.value_and_grad(lambda x: lens.loss(x, 'fan'))(pattern)
    expected = .2 * np.log(77 / 78) + 600 - .4 * np.log(78)
    assert float(value) == pytest.approx(expected, abs=1e-4)
    assert np.isfinite(gradient).all() and np.any(np.asarray(gradient) != 0)
    np.testing.assert_array_equal(gradient[:, 12, 55], 0.)


def test_fan_summary_cuts_and_outside():
    db = np.full((3, 73, 73), 12.)
    for phi, slope in [(0, .5), (36, .25), (18, .1), (54, .2)]:
        db[:, :, phi] = 12. - slope * lens.NOMINAL_DEG
    db[:, 0, 1] = 100.  # Boresight uses column 0, not the row mean.
    db[:, 23, 1] = 90.  # Outside begins at row 24 (60 degrees).
    db[:, 24, 71] = 14.
    db[:, :, 72] = 200.
    db[1, :, 54] = 12.  # A missing crossing propagates NaN.
    result = lens.summary(10 ** (db / 10), 'fan')
    np.testing.assert_allclose(result['boresight_dbi'], 12., atol=1e-12)
    np.testing.assert_allclose(result['narrow_hpbw_deg'], 18., atol=1e-12)
    np.testing.assert_allclose(result['wide_width_deg'][[0, 2]], 45., atol=1e-12)
    assert np.isnan(result['wide_width_deg'][1])
    np.testing.assert_allclose(result['outside_max_dbi'], 14., atol=1e-12)
    assert result['meets_narrow'].all()
