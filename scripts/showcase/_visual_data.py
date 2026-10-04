"""Stored-array transforms only; no solver imports."""
import json
from pathlib import Path
import numpy as np


def lens_mirror(eps):
    e = np.asarray(eps)
    if e.shape != (15, 15, 10):
        raise ValueError('lens quarter must be (15, 15, 10)')
    e = np.concatenate((e[::-1], e), axis=0)
    return np.concatenate((e[:, ::-1], e), axis=1)


def filter_mirror(eps):
    e = np.asarray(eps)
    if e.shape != (32, 5):
        raise ValueError('filter half must be (32, 5)')
    return np.concatenate((e, e[:, -2::-1]), axis=1)


def dbi(d):
    """Linear power directivity to dBi; zero is negative infinity."""
    d = np.asarray(d, dtype=float)
    if np.any(d < 0):
        raise ValueError("directivity cannot be negative")
    with np.errstate(divide="ignore"):
        return 10 * np.log10(d)


def plane_cut(d, plane):
    """Signed theta: E uses phi=0/pi; H uses phi=pi/2,3pi/2."""
    d = np.asarray(d)
    if d.shape[-2:] != (73, 73) or plane not in ('E', 'H'):
        raise ValueError('expected 73 x 73 grid and E or H plane')
    theta = np.linspace(1e-4, np.pi - 1e-4, 73)
    theta[0] = 0
    pos, neg = (0, 36) if plane == 'E' else (18, 54)
    return np.rad2deg(np.r_[-theta[:0:-1], theta]), np.concatenate((d[..., :0:-1, neg], d[..., :, pos]), axis=-1)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_record(path):
    with np.load(Path(path) / 'iterations.npz', allow_pickle=False) as a:
        return {k: a[k] for k in ('eps', 'response', 'objective', 'iteration', 'freqs_hz')}


def write_manifest(out, name, content):
    Path(out, name).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')


mirror_lens = lens_mirror
mirror_filter = filter_mirror


def amplitude_db(s):
    """Amplitude dB with the declared 1e-6 amplitude floor."""
    return 20 * np.log10(np.maximum(np.abs(s), 1e-6))


LENS_LABELS = {
    'no_lens': 'feed alone',
    'grin': 'textbook GRIN',
    'uniform_2.7': 'uniform slab, εr = 2.7',
    'best_uniform': 'best uniform slab',
    'reported': 'designed lens',
}


def lens_label(key):
    """Plain-language legend label for a stored lens design key."""
    if key.startswith('uniform_'):
        return f'uniform slab, εr = {key[len("uniform_"):]}'
    return LENS_LABELS.get(key, key)


FILTER_MESHES = [('0.000635', 'a/36 (0.635 mm)'),
                 ('0.000423333333', 'a/54 (0.423 mm)'),
                 ('0.0003175', 'a/72 (0.318 mm)')]


def final_hold_text(kind, trend, freqs_hz, eligibility=None, resolved_iteration=None, best_iteration=None):
    """Separate fine-mesh annotation from the design-mesh iterate HUD.

    Empty unless the re-solve record declares every promotional gate passed
    (eligibility.json) and re-solved the same iterate the film ends on.
    """
    if not (eligibility and eligibility.get('promotional_number_eligible') is True):
        return ''
    if resolved_iteration is None or best_iteration is None or int(resolved_iteration) != int(best_iteration):
        return ''
    if kind == 'lens':
        fi = int(np.argmin(abs(np.asarray(freqs_hz) - 1e10)))
        value = trend['reported']['boresight_dbi'][2][fi]
        return f'final: {value:.1f} dBi at 10 GHz, three-mesh converged'
    passed = trend.get('mask_each_mesh', [])
    return 'inside the mask on three meshes' if len(passed) == 3 and all(passed) else ''
