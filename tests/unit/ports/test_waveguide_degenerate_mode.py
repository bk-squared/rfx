"""#2707, predeclaration §§1–3,8: basis-independent degenerate port modes."""
from contextlib import nullcontext
import json

import numpy as np
import pytest
from jax import enable_x64

from rfx import Simulation
from rfx.errors import RfxRefusal
from rfx.model import waveguide_modes as wm

B = 1e-12
FIELDS = ('ey_profile', 'ez_profile', 'hy_profile', 'hz_profile')
ROWS = ('D1', 'D2a', 'D2b', 'D3', 'D7', 'D8a', 'D8b', 'D9-uniform', 'D9-graded-x', 'D10')


def _shape(u, v, mode, kind='TE', transverse=False):
    uc, vc = np.cumsum(u) - u / 2, np.cumsum(v) - v / 2
    m, n = mode
    trig = np.cos if kind == 'TE' else np.sin
    return (trig(m * np.pi * uc / u.sum())[:, None]
            * (np.cos if transverse else trig)(n * np.pi * vc / v.sum())[None, :])


def _axis_values(widths, kind):
    # Independent numpy finite-volume stiffness, with reflected PEC ghosts.
    size = len(widths)
    stiffness = np.zeros((size, size))
    for i in range(size - 1):
        c = 2 / (widths[i] + widths[i + 1])
        stiffness[i:i + 2, i:i + 2] += c * np.array([[1., -1.], [-1., 1.]])
    if kind == 'TM':
        stiffness[0, 0] += 2 / widths[0]
        stiffness[-1, -1] += 2 / widths[-1]
    root = np.sqrt(widths)
    return np.linalg.eigvalsh(stiffness / root[:, None] / root[None, :])


def _scene(row):
    nu = 20 if row in ('D2a', 'D2b', 'D8b', 'D10') else 10
    u, v = np.full(nu, .002), np.full(10, .002)
    if row == 'D7':
        u = v = np.array([.0015, .0025] * 5)
    kind = 'TM' if row == 'D3' else 'TE'
    target = (1, 2) if kind == 'TM' else ((2, 0) if nu == 20 else (1, 0))
    guard_u, guard_v = (u.astype(np.float32).astype(np.float64), v.astype(np.float32).astype(np.float64)) if row == 'D9-graded-x' else (u, v)
    values_u, values_v = _axis_values(guard_u, kind), _axis_values(guard_v, kind)
    offset = 1 if kind == 'TM' else 0
    cutoff = values_u[target[0] - offset] + values_v[target[1] - offset]
    return u, v, kind, cutoff


def _install_wrapper(patch, build, u, v, kind, target):
    """Wrap every eigh call, including those made by TE lower-mode recursion."""
    eigh = np.linalg.eigh
    active = []
    changes = []
    root = np.sqrt(np.kron(u, v))
    start = 0 if kind == 'TE' else 1
    modes = [(m, n) for m in range(start, len(u) + start)
             for n in range(start, len(v) + start)]
    shapes = np.array([root * _shape(u, v, mode, kind).ravel() for mode in modes])
    shapes /= np.linalg.norm(shapes, axis=1)[:, None]
    rotation = np.array([[np.cos(np.pi / 6), -np.sin(np.pi / 6)],
                         [np.sin(np.pi / 6), np.cos(np.pi / 6)]])

    def wrapped_eigh(*args, **kwargs):
        values, vectors = eigh(*args, **kwargs)
        if vectors.shape[0] != len(root):
            return values, vectors
        original = vectors.copy()
        first = 0
        while first < len(values):
            end = first + 1
            while end < len(values) and values[end] - values[first] <= 1e-9 * abs(values[first]):
                end += 1
            if build != 'u':
                for column in range(first, end - 1):
                    vectors[:, column:column + 2] = vectors[:, column:column + 2] @ rotation
                if build == 'rr':
                    basis = vectors[:, first:end]
                    coefficients = shapes @ basis
                    scores = np.sum(coefficients**2, axis=1)
                    if active:
                        scores[modes.index(active[-1])] = -1
                    p = shapes[np.argmax(scores)]
                    q = basis @ (basis.T @ p)
                    q /= np.linalg.norm(q)
                    vectors[:, first:end] = basis - 2 * np.outer(q, q @ basis)
            if end - first > 1 and abs(values[first] - target) <= 1e-9 * abs(target):
                changes.append(float(np.linalg.norm(vectors[:, first:end] - original[:, first:end])
                                     / np.linalg.norm(original[:, first:end])))
            first = end
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(len(values)), atol=1e-12, rtol=0)
        return values, vectors

    def track(original):
        def tracked(a, b, m, n, *args, **kwargs):
            active.append((m, n))
            try:
                return original(a, b, m, n, *args, **kwargs)
            finally:
                active.pop()
        return tracked

    for name in ('_discrete_te_mode_profiles', '_discrete_tm_mode_profiles'):
        patch.setattr(wm, name, track(getattr(wm, name)))
    patch.setattr(np.linalg, 'eigh', wrapped_eigh)
    return changes


def _config_arrays(cfg):
    return tuple(np.asarray(getattr(cfg, key)) for key in FIELDS) + (float(cfg.f_cutoff),)


def _build(row, u, v):
    mode = (1, 2) if row == 'D3' else ((3, 0) if row in ('D8b', 'D10') else (1, 0))
    kind = 'TM' if row == 'D3' else 'TE'
    dA = u[:, None] * v[None, :]
    if row in ('D8a', 'D8b'):
        dA[-1, :] = 0
        dA[:, -1] = 0
    if row in ('D7', 'D8a', 'D8b'):
        result = wm._discrete_te_mode_profiles(u.sum(), v.sum(), *mode, u, v, aperture_dA=dA)
        return [result], [mode], dA
    if row.startswith('D9'):
        mesh = {'dx_profile': np.array([.002] * 20 + [.001] * 20 + [.002] * 20)} if row.endswith('graded-x') else {}
        sim = Simulation(12e9, (.1, .02, .02), dx=.002, cpml_layers=4,
                         boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}, **mesh)
        sim.add_waveguide_port(.02, name='launch')
        result = sim.run(n_steps=2)
        cfg = result.waveguide_ports['launch']
        assert np.asarray(cfg.ey_profile).dtype == np.float32
        return [_config_arrays(cfg)], [mode], np.asarray(cfg.aperture_dA)
    port = wm.WaveguidePort(15, (0, len(u)), (0, len(v)), u.sum(), v.sum(), mode=mode, mode_type=kind)
    kwargs = dict(f0=20e9, field_dtype=np.float64)
    if row in ('D2a', 'D2b'):
        cfgs = wm.init_multimode_waveguide_port(port, .002, np.array([20e9]),
                                               n_modes=2 if row == 'D2a' else 3, **kwargs)
        assert cfgs[1].mode_indices == (0, 1)
        if row == 'D2b':
            assert cfgs[2].mode_indices == (2, 0)
    else:
        cfgs = [wm.init_waveguide_port(port, .002, np.array([20e9]), **kwargs)]
    return [_config_arrays(c) for c in cfgs], [c.mode_indices for c in cfgs], np.asarray(cfgs[0].aperture_dA)


def _overlap(a, b, weights):
    return abs(np.sum(a * b * weights)) / np.sqrt(np.sum(a*a*weights) * np.sum(b*b*weights))


@pytest.mark.parametrize('row', ROWS)
@pytest.mark.parametrize('build', ('u', 'r', 'rr'))
def test_declared_row(row, build, monkeypatch):
    u, v, kind, target = _scene(row)
    with nullcontext() if row.startswith('D9') else enable_x64():
        baseline, _, _ = _build(row, u, v)
        with monkeypatch.context() as patch:
            changes = _install_wrapper(patch, build, u, v, kind, target)
            stored, modes, dA = _build(row, u, v)
    guard = max(changes, default=0.)
    assert changes, 'numpy-computed degenerate cluster was never observed'
    if build != 'u':
        assert guard > 1e-3, f'{row}/{build}: non-vacuity guard {guard}'
    cross_guard = None
    if row == 'D8b':
        uc = np.cumsum(u) - u / 2
        s3 = np.broadcast_to(np.sin(3 * np.pi * uc / u.sum())[:, None], dA.shape)
        s2 = np.broadcast_to(np.sin(2 * np.pi * uc / u.sum())[:, None], dA.shape)
        cross_guard = _overlap(s3, s2, dA)
        assert cross_guard >= 1e-3, 'D8b dropped-face cross-overlap guard'
    for index, (actual, reference, mode) in enumerate(zip(stored, baseline, modes)):
        ey, ez, hy, hz, cutoff = actual
        epeak = max(np.max(np.abs(reference[0])), np.max(np.abs(reference[1])))
        hpeak = max(np.max(np.abs(reference[2])), np.max(np.abs(reference[3])))
        differences = [float(np.max(np.abs(a - b)) / peak)
                       for a, b, peak in zip(actual[:4], reference[:4], (epeak, epeak, hpeak, hpeak))]
        cutoff_difference = abs(cutoff - reference[4]) / abs(reference[4])
        quantity = None
        if row == 'D3':
            self_overlap = _overlap(ez, _shape(u, v, (1, 2), 'TM', True), dA)
            cross_overlap = _overlap(ez, _shape(u, v, (2, 1), 'TM', True), dA)
            quantity = [float(self_overlap), float(cross_overlap)]
        elif row != 'D10':
            quantity = float(np.max(np.abs(ez)) / np.max(np.abs(ey)) if mode == (0, 1)
                             else np.max(np.abs(ey)) / np.max(np.abs(ez)))
        print('MEASURE ' + json.dumps(dict(row=row, build=build, mode=mode, quantity=quantity,
              differences=differences, cutoff_difference=cutoff_difference,
              guard=guard, cross_guard=cross_guard, B=B)))
        if row == 'D3':
            assert self_overlap >= 1 - B, f'{row}/{build}: TM12 self overlap {self_overlap}'
            assert cross_overlap <= B, f'{row}/{build}: TM21 cross overlap {cross_overlap}'
        elif row != 'D10':
            assert quantity <= B, f'{row}/{build}/{mode}: purity {quantity} > B={B}'
        assert max(differences) <= B, f'{row}/{build}/{index}: E/H joint-peak differences {differences}'
        assert cutoff_difference <= B, f'{row}/{build}: cutoff difference {cutoff_difference}'


def test_D5_helper():
    mass = np.array([1., 2., 3., 4.])
    basis = np.eye(4) / np.sqrt(mass)[:, None]
    values = np.array([1., 1. + 1e-10, 3., 4.])
    with pytest.raises(RfxRefusal):
        wm._project_degenerate_mode(values, basis, 0, np.array([0., 0., 1., 1.]), mass)
    template = np.array([1., 2., 0., 0.])
    result, kc2 = wm._project_degenerate_mode(values, basis, 1, template, mass)
    difference = float(np.max(np.abs(result - template / np.sqrt(np.sum(mass * template**2)))))
    print('MEASURE ' + json.dumps(dict(row='D5', difference=difference, kc2=kc2, B=B)))
    assert difference <= B
    assert kc2 == min(values[:2])


@pytest.mark.parametrize('kind,mode,nu,lower', [
    ('TE', (1, 0), 10, None), ('TM', (1, 2), 10, None), ('TE', (3, 0), 20, 'TE20'),
])
def test_D5_refusal_context(kind, mode, nu, lower, monkeypatch):
    def refuse(*args):
        raise RfxRefusal('synthetic near-orthogonal group')

    monkeypatch.setattr(wm, '_project_degenerate_mode', refuse)
    u, v = np.full(nu, .002), np.full(10, .002)
    builder = wm._discrete_te_mode_profiles if kind == 'TE' else wm._discrete_tm_mode_profiles
    with pytest.raises(RfxRefusal) as caught:
        builder(u.sum(), v.sum(), *mode, u, v, aperture_dA=u[:, None] * v[None, :])
    message = str(caught.value)
    assert f'{kind}{mode[0]}{mode[1]}' in message
    assert f'{nu} x 10 cells' in message
    if lower:
        assert f'lower mode {lower}' in message
    print('D5_CONTEXT', message)
