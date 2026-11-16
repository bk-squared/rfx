"""Diagnosis of the vector-pad cases outside the per-step bar (not a test; asserts nothing).

For a case, run the scene unpadded and padded with every field snapshotted each
step, and return as a dict:

- ``r_H``, ``r_E``: max |padded - unpadded| over the family's components and cells,
  divided by 1.2e-7 x S(n), at every step n (S from the unpadded run);
- at the step where ``r_H`` peaks: the component and cell of the largest H
  difference, its distance in cells from each padded high face, both values and
  the difference in float32 ULPs of the unpadded value;
- ``r_H`` restricted to cells at least 2 cells from every padded high face;
- the first step with any H difference, and its cell;
- a witness: the same scene unpadded in two compiled forms (the plain time step,
  and the same call inside an outer ``jax.jit``), judged the same way.

Run through ``tests/unit/runners/test_vector_pad_diag_tmp.py`` (skipped unless
``RFX_VPAD_DIAG=1``).
"""

import contextlib

import jax
import numpy as np

from rfx.stepping import vector_pad
from tests.unit.runners.test_vector_pad import FIELDS, STEP_BAR, family_sums, run_record, scene
from rfx import SnapshotSpec

FAMILIES = {'h': ('hx', 'hy', 'hz'), 'e': ('ex', 'ey', 'ez')}


@contextlib.contextmanager
def one_axis_pad(pad):
    """Every forced plan pads exactly ``pad`` (the per-axis test's plan)."""
    real = vector_pad.plan_pad

    def plan(shape, *, target, periodic_axes, two_d=False):
        base = real(shape, target='off', periodic_axes=periodic_axes, two_d=two_d)
        if target == 'off':
            return base
        return vector_pad.PadPlan(base.shape, tuple(n + p for n, p in zip(base.shape, pad)), pad, target)

    vector_pad.plan_pad = plan
    try:
        yield
    finally:
        vector_pad.plan_pad = real


def _snapshots(out):
    return {c: np.asarray(out[f"snapshots['{c}']"]) for c in FIELDS}


def _curve(ref, got, sums, family, mask=None):
    """r(n) for one family; ``mask`` keeps only the cells it marks True."""
    worst = np.zeros(len(sums[family]))
    for c in FAMILIES[family]:
        diff = np.abs(got[c].astype(np.float64) - ref[c].astype(np.float64))
        if mask is not None:
            diff = np.where(mask[None], diff, 0.)
        worst = np.maximum(worst, diff.reshape(len(diff), -1).max(axis=1))
    return worst / (STEP_BAR * sums[family])


def _ulps(ref, got):
    ref32, got32 = np.float32(ref), np.float32(got)
    return float((np.float64(got32) - np.float64(ref32)) / np.spacing(np.abs(ref32)))


def _compare(ref, got, plan_shape, pad):
    """Every item of the diagnosis between two per-step snapshot records."""
    sums = family_sums(ref)
    r_h, r_e = _curve(ref, got, sums, 'h'), _curve(ref, got, sums, 'e')
    shape = ref['hx'].shape[1:]
    peak = int(np.argmax(r_h))
    best = None
    for c in FAMILIES['h']:
        diff = np.abs(got[c][peak].astype(np.float64) - ref[c][peak].astype(np.float64))
        cell = np.unravel_index(int(np.argmax(diff)), shape)
        if best is None or diff[cell] > best[0]:
            best = (float(diff[cell]), c, tuple(int(i) for i in cell))
    _, component, cell = best
    distance = {'xyz'[a]: int(plan_shape[a] - 1 - cell[a]) for a in range(3) if pad[a]}
    ref_value, got_value = float(ref[component][peak][cell]), float(got[component][peak][cell])
    # Cells at least 2 cells from every padded high face.
    mask = np.ones(shape, dtype=bool)
    for a in range(3):
        if pad[a]:
            index = np.arange(shape[a]).reshape([-1 if b == a else 1 for b in range(3)])
            mask &= (plan_shape[a] - 1 - index) >= 2
    r_h_inner = _curve(ref, got, sums, 'h', mask)
    first = None
    for n in range(len(r_h)):
        for c in FAMILIES['h']:
            changed = np.argwhere(got[c][n] != ref[c][n])
            if len(changed):
                first = dict(step=n + 1, component=c, cell=[int(i) for i in changed[0]],
                             cells_differing=int(len(changed)))
                break
        if first:
            break
    return dict(
        r_H=r_h.tolist(), r_E=r_e.tolist(),
        peak_step=peak + 1, peak_r_H=float(r_h[peak]),
        peak_r_E=float(r_e.max()), peak_r_E_step=int(np.argmax(r_e)) + 1,
        peak_detail=dict(component=component, cell=list(cell),
                         distance_from_padded_high_face=distance,
                         unpadded=ref_value, padded=got_value,
                         ulps_of_unpadded_value=_ulps(ref_value, got_value)),
        inner_peak_r_H=float(r_h_inner.max()), inner_peak_step=int(np.argmax(r_h_inner)) + 1,
        first_H_difference=first)


def padded_vs_unpadded(name, target, n_steps, pad=None):
    snap = SnapshotSpec(interval=1, components=FIELDS)
    with (one_axis_pad(pad) if pad else contextlib.nullcontext()):
        ref, _ = run_record(scene(name), 'off', n_steps=n_steps, snapshot=snap)
        got, plans = run_record(scene(name), target, n_steps=n_steps, snapshot=snap)
    plan = plans[0]
    assert plan.active, plan
    out = _compare(_snapshots(ref), _snapshots(got), plan.shape, plan.pad)
    out['plan'] = dict(shape=list(plan.shape), padded=list(plan.padded), pad=list(plan.pad))
    return out


def witness(name, n_steps, pad_faces):
    """The unpadded scene in two compiled forms: the plain call, and inside an outer jit."""
    import rfx.simulation as simulation
    real, calls = simulation.run, []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    simulation.run = capture
    try:
        run_record(scene(name), 'off', n_steps=n_steps,
                   snapshot=SnapshotSpec(interval=1, components=FIELDS))
    finally:
        simulation.run = real
    assert len(calls) == 1, f'{len(calls)} time-step calls in one run()'
    args, kwargs = calls[0]

    def snaps(result):
        return {c: np.asarray(v) for c, v in result.items()}

    plain = snaps(real(*args, **kwargs).snapshots)
    form = 'outer jax.jit around the same call'
    try:
        other = snaps(jax.jit(lambda: real(*args, **kwargs).snapshots)())
    except Exception as exc:  # report which compiled form was used
        form = f'jax.disable_jit (outer jit failed: {type(exc).__name__}: {exc})'[:300]
        with jax.disable_jit():
            other = snaps(real(*args, **kwargs).snapshots)
    shape = plain['hx'].shape[1:]
    out = _compare(plain, other, shape, pad_faces)
    out['form'] = form
    out['bitwise'] = all(plain[c].tobytes() == other[c].tobytes() for c in FIELDS)
    return out


CASES = {
    'vacuum y pad (0,1,0)': dict(name='vacuum', target='aligned', pad=(0, 1, 0)),
    'pec_pmc_slab/odd': dict(name='pec_pmc_slab', target='odd', pad=None),
}


def diagnose(n_steps=120):
    report = {}
    for label, case in CASES.items():
        result = padded_vs_unpadded(case['name'], case['target'], n_steps, case['pad'])
        result['witness'] = witness(case['name'], n_steps, tuple(result['plan']['pad']))
        report[label] = result
        jax.clear_caches()
    return report
