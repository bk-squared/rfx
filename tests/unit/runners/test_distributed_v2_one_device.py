"""One device through ``distributed_v2.run_distributed`` (#1296).

A direct call with one device used to be handed to the ``jax.pmap`` runner,
which never applied the declared-PEC cell mask and therefore refused a PEC
volume -- so the same call refused a metal block at one device and ran it at
two. #1296 removed that runner. One device now runs v2's own shard_map path on
a one-device mesh: the same kernels, the same stage order and the same
refusals as at two devices, with no seam. ``Simulation.run(devices=...)`` never
takes this path (it dispatches to v2 only for two or more devices); a direct
caller does, including ``devices=None`` on a one-device host.

The reference is the same model through the uniform single-device lane. The
geometry is ``test_distributed_v2_pec_body_seam.py``'s: a 31 x 15 x 15 mm box
at 1 mm cells, a field source 8 mm from x_lo, Ez probes 4 mm and 12 mm further
along x, and a 5-cell PEC block between them (x 16-21 mm, y and z 6-11 mm).

GATE DERIVATION
---------------
Measured 2026-09-25 on this pod (one of 2 virtual CPU devices, jax 0.10.2,
float32, 300 steps), ``max|one device - single| / peak(|single|)`` per probe:

    boundary  body      probe0      probe1      mask stage dropped
    pec       none      0           0           (no mask)
    pec       block     0           0           1.049e-01 / 1.375e+00
    cpml      none      2.127e-06   3.161e-05   (no mask)
    cpml      block     1.512e-06   1.285e-05   1.513e-02 / 4.246e-01

With PEC walls the one-device mesh reproduces the uniform lane bit for bit on
this host, probes and all six final fields; with CPML the two lanes run
different absorber kernels (x-slab CPML against the uniform lane's), which is
the 1e-5 floor. Bit identity is not promised on another host or XLA version,
so the tests read a tolerance: ``GATE = 1e-3``, the gate
``test_distributed_v2_pec_body_seam.py`` derives for the same kernel pair at
two devices, 32x the worst number above. The block moves the single-device
trace by more than 100x the gate (asserted), and running the block with the
mask stage dropped -- the pmap runner's behaviour -- lands 15x to 1375x above
it (asserted), so the gate separates a realized block from a missing one.
"""

import warnings

import jax
import numpy as np
import pytest

import rfx.runners.distributed_v2 as _v2
from tests.unit.runners.test_distributed_v2_pec_body_seam import (
    N_STEPS,
    SEAM_FACE_BODY_X,
    _build,
)

GATE = 1e-3
BLOCK_X = SEAM_FACE_BODY_X     # one device has no seam; any block will do

_CACHE: dict = {}


def _run(boundary, body_x, *, one_device, drop_stage=False):
    """One run, cached: the single-device references are shared."""
    key = (boundary, body_x, one_device, drop_stage)
    if key in _CACHE:
        return _CACHE[key]
    saved_stage = _v2.apply_pec_mask_shmap
    if drop_stage:
        _v2.apply_pec_mask_shmap = lambda st, *a, **k: st
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = _build(boundary, body_x)
            if one_device:
                res = _v2.run_distributed(sim, n_steps=N_STEPS,
                                          devices=jax.devices()[:1])
            else:
                res = sim.run(n_steps=N_STEPS)
    finally:
        _v2.apply_pec_mask_shmap = saved_stage
    _CACHE[key] = res
    return res


def _rel(boundary, body_x, *, drop_stage=False):
    """max|one device - single| / peak(|single|), per probe."""
    got = np.asarray(_run(boundary, body_x, one_device=True,
                          drop_stage=drop_stage).time_series)
    ref = np.asarray(_run(boundary, body_x, one_device=False).time_series)
    assert got.shape == ref.shape == (N_STEPS, 2), (got.shape, ref.shape)
    peaks = np.max(np.abs(ref), axis=0)
    assert np.all(peaks > 1e-6), f"vacuous fixture: probe peaks {peaks}"
    return np.max(np.abs(got - ref), axis=0) / peaks


@pytest.mark.parametrize("body_x", [None, BLOCK_X], ids=["empty", "block"])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_one_device_matches_the_single_device_lane(boundary, body_x):
    """The parity gate. With the block, that the run happens at all is half
    of it: the pmap runner raised ``NotImplementedError`` here."""
    if body_x is not None:
        ref = np.asarray(_run(boundary, body_x, one_device=False).time_series)
        empty = np.asarray(_run(boundary, None, one_device=False).time_series)
        effect = (np.max(np.abs(ref - empty), axis=0)
                  / np.max(np.abs(empty), axis=0))
        assert np.max(effect) > 100 * GATE, (
            f"boundary={boundary}: deleting the block moves the single-device "
            f"trace by only {effect}; a path that dropped it would pass.")
    rel = _rel(boundary, body_x)
    assert np.all(rel < GATE), (
        f"boundary={boundary}, body {body_x}: one device through "
        f"distributed_v2 deviates from the single-device lane by {rel} "
        f"(gate {GATE:.0e}).")


@pytest.mark.parametrize("body_x", [None, BLOCK_X], ids=["empty", "block"])
def test_one_device_final_e_field_matches_in_a_closed_box(body_x):
    """All of E at the last step, PEC walls: nothing leaves the box, so the
    final field is as large as the traces. Each component against its own
    peak. (With CPML the final E is what the absorber leaves, about 1 % of
    the probe peak, and the two lanes' absorber kernels differ there by up to
    1.3e-3 of that residue; the traces above are the CPML gate.)"""
    got = _run("pec", body_x, one_device=True).state
    ref = _run("pec", body_x, one_device=False).state
    for comp in ("ex", "ey", "ez"):
        a = np.asarray(getattr(got, comp))
        b = np.asarray(getattr(ref, comp))
        assert a.shape == b.shape, (comp, a.shape, b.shape)
        peak = np.max(np.abs(b))
        assert peak > 0, f"vacuous fixture: final {comp} is zero"
        rel = np.max(np.abs(a - b)) / peak
        assert rel < GATE, (
            f"body {body_x}: final {comp} deviates by {rel:.3e} of its peak "
            f"(gate {GATE:.0e})")


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_dropping_the_mask_stage_reds_the_one_device_gate(boundary):
    """The gate binds. Without the mask stage the block is absent -- the
    pmap runner's defect -- and the traces move by 1.5e-02 to 1.375e+00."""
    rel = _rel(boundary, BLOCK_X, drop_stage=True)
    assert np.max(rel) > GATE, (
        f"boundary={boundary}: with the PEC-mask stage dropped the one-device "
        f"path still agrees to {rel}, inside the gate {GATE:.0e}.")
