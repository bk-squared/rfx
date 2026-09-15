"""#1053 physics gate: distributed_v2 realizes a declared PEC body at the seam.

``rfx/runners/distributed_v2.py`` assembled ``pec_mask`` and dropped it: its
step bodies applied the DOMAIN-FACE PEC alone, so no declared conductor reached
the field update. Rather than run with the metal missing, the lane refused
(``NotImplementedError``). #1053 leg 2 ported ``distributed_nu``'s realized-PEC
mask stage across; leg 4 narrowed the refusal to sheets and wires, which own no
cell and are still absent from this lane.

So every test here now drives the PUBLIC route, ``sim.run(devices=...)``. They
were written in leg 0 as ``xfail(raises=NotImplementedError, strict=True)``
beside a second family that reached the same physics with the volume half of
the refusal bypassed in-test; leg 4 lifted the refusal, so the xfail is gone
and the bypassed duplicates are gone with it -- their assertions live in
``test_a_declared_pec_body_matches_the_single_device_lane`` below.

The reference is always the SAME model on ONE device through the uniform lane,
the only lane in the repo with no seam. Not a tolerance-free comparison: the
two lanes evaluate the same physics through different kernels and different
float32 fusions.

WHICH BODY WITNESSES THE HOOK POINT (measured, and it is not the obvious one)
----------------------------------------------------------------------------
The stage must run after source injection and IMMEDIATELY BEFORE the E ghost
exchange, so the exchange hands the neighbour a finished row. A body "straddling
the seam" turns out NOT to prove that. Measured on this fixture at 300 steps, by
moving the stage to after the exchange and re-running (relative error against
the single-device lane, correct order vs wrong order):

    body x span     cells owned            pec, correct   pec, WRONG order
    13-18 mm        rank 0 and rank 1      4.362e-05      4.362e-05  (blind)
    14-19 mm        rank 0 and rank 1      3.483e-05      3.483e-05  (blind)
    15-20 mm        rank 0 and rank 1      3.961e-05      3.961e-05  (blind)
    16-21 mm        rank 1 only            3.074e-05      3.203e-01  <-- WITNESS
    17-22 mm        rank 1 only            5.197e-05      5.197e-05  (blind)

Only the body whose metal FACE lands on rank 1's first real cell sees it, and
the mechanism says why. Rank 0's right E ghost is the plane rank 1 owns and
zeroes; rank 0's H at its LAST REAL cell consumes that ghost. If the exchange
runs first, rank 0 imports the un-zeroed plane. That error then has to escape
into live field to be observable -- and when the body has two or more metal
cells on rank 0's side, rank 0's last real cell is itself buried in the
conductor, its own E edges are re-zeroed every step, and the perturbation never
leaves the metal. With the body's face exactly on the seam, rank 0's last real
cell is vacuum and the error propagates.

So this file keeps THREE bodies and says what each one is for: the seam-face
body is the ordering gate, the straddling body proves the mask is realized on
both ranks at once, and the interior body is the no-seam control.

GATE DERIVATION
---------------
Measured on this host (2026-09-15, 2 virtual CPU devices, jax 0.10.2, float32,
300 steps), ``max|multi - single| / peak(|single|)`` per probe, stage ON:

    body              pec probe0   pec probe1   cpml probe0  cpml probe1
    none (floor)      5.100e-06    8.889e-05    1.991e-06    3.603e-05
    seam-face 16-21   4.332e-06    3.074e-05    2.199e-06    1.954e-05
    straddle 14-19    4.743e-06    3.483e-05    2.927e-06    1.385e-05
    interior 22-27    5.243e-06    4.148e-05    1.721e-06    3.826e-05

and the two defect signatures the gate has to reject, same runs: the stage
DROPPED to a no-op (the metal-missing lane this port exists to end) and the
stage moved to AFTER the E ghost exchange (the wrong hook point):

    body              stage OFF, pec         stage OFF, cpml        after-exch, pec
    seam-face 16-21   1.049e-01 / 1.375e+00  1.505e-02 / 4.424e-01  2.093e-02 / 3.203e-01
    straddle 14-19    1.041e-01 / 1.435e+00  8.251e-02 / 3.293e-01  (blind, see above)
    interior 22-27    1.273e-01 / 1.062e+00  2.240e-03 / 1.962e-01  (no seam involved)

``GATE = 1e-3`` is 11x the worst stage-ON number, 8.889e-05 -- which is the
NO-BODY floor; every body-bearing case is smaller. Read the way the tests read
it, max over the two probes, the weakest defect signature is 1.962e-01, 196x
the gate, and the strongest is 1.435e+00, 1435x. The wrong-hook-point signature
on the seam-face body is 3.203e-01 (pec) and 1.786e-01 (cpml), 179x to 320x the
gate. The same 1e-3 is #1041's gate on this geometry, derived the same way from
the same lane floor.

Why NOT 5e-5, the number the nu lane uses: that is the Class B final-step
tolerance of ``tests/_distributed_nu_tolerances.py:101-106``, and it gates the
NU DISTRIBUTED lane against the NU SINGLE-DEVICE lane -- the same kernels on the
same grid type, where the only difference is the decomposition. This file
compares v2's shard_map kernels against the uniform single-device stepper, and
the measured floor for that pair on this geometry is 8.889e-05, ABOVE 5e-5. A
5e-5 gate here would pin float32 fusion rather than the PEC stage, and would be
red on the day it was written.

PREFLIGHT, quoted verbatim -- the body is 5 cells on every axis and preflight
says so on every build that carries one, single-device and distributed alike:

    [PREFLIGHT] PEC 'pec' x-extent 5mm = 5.0 cells - volume under-resolved (a
    PEC volume's curved/edge features need >=5 cells; a 1-2 cell slab is
    realized as drawn, with walls on both faces).

and the same line for y and z. It is advisory and it is about absolute
accuracy, not lane parity: both sides of every comparison here carry the
identical under-resolved body, so it cancels in the difference. The
empty-domain builds report ``All checks passed (NTFF advisory tier; the
PEC-overlap error check runs on forward()/preflight())``.
"""

# Simulate 2 devices on CPU. Must be set BEFORE importing JAX.
import os  # noqa: I001

os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import jax  # noqa: E402

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2,
    reason=(
        "the #1053 PEC-body seam gate needs >=2 devices. Run with "
        "XLA_FLAGS=--xla_force_host_platform_device_count=2. "
        "tests/unit/runners/test_device_count_sentinel.py FAILS rather than "
        "skips in that environment, so this skip is not a silent hole."
    ),
)

import rfx.runners.distributed_v2 as _v2  # noqa: E402
from rfx import Box, Simulation  # noqa: E402

DX = 1e-3
NX_CELLS = 31        # -> nx = 32 nodes, so nx_per_rank = 16 and pad_x = 0
NYZ_CELLS = 15       # -> ny = nz = 16
CENTER = 8e-3
SOURCE = (8e-3, CENTER, CENTER)       # 8 cells from the seam, inside rank 0
PROBE_LO = (12e-3, CENTER, CENTER)    # 4 cells inside rank 0
#: Offset in z, OUTSIDE the body's 6-11 mm span, so the rank-1 probe is not in
#: the block's geometric shadow. On the axis its peak drops to 24% and the
#: same absolute float32 difference then divides by a much smaller number --
#: 2.651e-04 instead of 3.483e-05, for no physical reason.
PROBE_HI = (20e-3, CENTER, 13e-3)     # 4 cells inside rank 1, off-axis

#: x span of the PEC body, metres. The rank seam sits between global node 15
#: (rank 0's last real cell) and node 16 (rank 1's first).
SEAM_FACE_BODY_X = (16e-3, 21e-3)   # face ON the seam: the ordering witness
STRADDLE_BODY_X = (14e-3, 19e-3)    # metal owned by BOTH ranks
INTERIOR_BODY_X = (22e-3, 27e-3)    # wholly inside rank 1, 6 cells clear
BODY_YZ = (6e-3, 11e-3)             # 5 cells, centred on the source line

GATE = 1e-3

#: Measured stage-ON relative error per (boundary, body), probe0 then probe1.
#: Kept as data so the tests assert against the numbers the docstring derives
#: the gate from, and a drift shows up as a number rather than as a pass.
MEASURED = {
    ("pec", None): (5.100e-06, 8.889e-05),
    ("pec", SEAM_FACE_BODY_X): (4.332e-06, 3.074e-05),
    ("pec", STRADDLE_BODY_X): (4.743e-06, 3.483e-05),
    ("pec", INTERIOR_BODY_X): (5.243e-06, 4.148e-05),
    ("cpml", None): (1.991e-06, 3.603e-05),
    ("cpml", SEAM_FACE_BODY_X): (2.199e-06, 1.954e-05),
    ("cpml", STRADDLE_BODY_X): (2.927e-06, 1.385e-05),
    ("cpml", INTERIOR_BODY_X): (1.721e-06, 3.826e-05),
}

N_STEPS = 300

_TRACE_CACHE: dict = {}


def _build(boundary, body_x, *, nu=False):
    """The fixture. ``body_x=None`` deletes the conductor.

    ``nu=True`` adds a UNIFORM-VALUED ``dz_profile``, which is the same
    lattice through a different code path: it flips ``run_distributed``'s
    ``is_nu`` branch, so the run assembles through ``_assemble_materials_nu``
    and steps through the NU kernels. The PEC-mask stage is outside that
    dispatch and runs on both, so the branch needs a witness of its own.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(
            freq_max=15e9,
            domain=(NX_CELLS * DX, NYZ_CELLS * DX, NYZ_CELLS * DX),
            dx=DX, boundary=boundary,
            cpml_layers=6 if boundary == "cpml" else 0,
            **({"dz_profile": np.full(NYZ_CELLS, DX)} if nu else {}))
        if body_x is not None:
            sim.add(Box((body_x[0], BODY_YZ[0], BODY_YZ[0]),
                        (body_x[1], BODY_YZ[1], BODY_YZ[1])),
                    material="pec")
        sim.add_source(SOURCE, "ez", amplitude_kind="field")
        sim.add_probe(PROBE_LO, "ez")
        sim.add_probe(PROBE_HI, "ez")
    return sim


def _run(boundary, body_x, *, distributed, drop_stage=False, nu=False):
    """One trace. Cached: several tests share the same single-device run."""
    key = (boundary, body_x, distributed, drop_stage, nu)
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]

    saved_stage = _v2.apply_pec_mask_shmap
    if drop_stage:
        _v2.apply_pec_mask_shmap = lambda st, *a, **k: st
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kw = dict(devices=jax.devices()[:2]) if distributed else {}
            ts = np.asarray(
                _build(boundary, body_x, nu=nu).run(
                    n_steps=N_STEPS, **kw).time_series)
    finally:
        _v2.apply_pec_mask_shmap = saved_stage

    _TRACE_CACHE[key] = ts
    return ts


def _rel(boundary, body_x, *, drop_stage=False, nu=False):
    """max|multi - single| / peak(|single|), per probe.

    The distributed run goes FIRST: when this lane refuses a model it must do
    so before the test pays for a single-device reference it will not use.
    """
    ts_multi = _run(boundary, body_x, distributed=True,
                    drop_stage=drop_stage, nu=nu)
    ts_single = _run(boundary, body_x, distributed=False, nu=nu)
    assert ts_multi.shape == ts_single.shape == (N_STEPS, 2), (
        f"shapes: multi {ts_multi.shape}, single {ts_single.shape}")
    peaks = np.max(np.abs(ts_single), axis=0)
    assert np.all(peaks > 1e-6), (
        f"vacuous fixture: single-device probe peaks {peaks}")
    return np.max(np.abs(ts_multi - ts_single), axis=0) / peaks


def _assert_the_body_is_visible(boundary, body_x):
    """The conductor must move the single-device trace by far more than GATE.

    Without this, a lane that silently dropped the body could pass the
    comparison by arithmetic rather than by physics.
    """
    ts_body = _run(boundary, body_x, distributed=False)
    ts_empty = _run(boundary, None, distributed=False)
    effect = (np.max(np.abs(ts_body - ts_empty), axis=0)
              / np.max(np.abs(ts_empty), axis=0))
    assert np.max(effect) > 100 * GATE, (
        f"boundary={boundary}, body {body_x}: deleting the conductor moves "
        f"the single-device trace by only {effect}, not >> the gate "
        f"{GATE:.0e}. A lane that dropped the body would pass; re-place the "
        "body or the probes.")


_BODIES = [
    pytest.param(SEAM_FACE_BODY_X, id="seam-face"),
    pytest.param(STRADDLE_BODY_X, id="straddle"),
    pytest.param(INTERIOR_BODY_X, id="interior"),
]


# ===========================================================================
# The gate, through the public route.
# ===========================================================================

@pytest.mark.parametrize("body_x", _BODIES)
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_a_declared_pec_body_matches_the_single_device_lane(boundary, body_x):
    """The gate. Three bodies, each testing something different.

    ``seam-face``: the block's metal face is rank 1's first real cell, so the
    PEC edge rank 0 must import through the E ghost exchange sits next to live
    field. This is the one placement that witnesses the stage's hook point --
    moving the stage after the exchange takes it to 2.093e-01 / 3.203e-01
    (pec). ``straddle``: metal owned by both ranks at once, which proves each
    rank realizes its own slab. ``interior``: no seam involvement at all.

    That this runs at all is half the assertion. Before #1053 leg 4,
    ``sim.run(devices=...)`` raised ``NotImplementedError`` on any declared
    PEC volume and this test was ``xfail(raises=NotImplementedError,
    strict=True)``.
    """
    _assert_the_body_is_visible(boundary, body_x)
    rel = _rel(boundary, body_x)
    assert np.all(rel < GATE), (
        f"boundary={boundary}, body {body_x}: the distributed lane deviates "
        f"from the single-device lane by {rel} (gate {GATE:.0e}). For the "
        "seam-face body suspect the stage POSITION first: it must run after "
        "source injection and immediately BEFORE the E ghost exchange, so a "
        "ghost row is a copy of the owner's finished real row (#1053 leg 2).")
    recorded = np.asarray(MEASURED[(boundary, body_x)])
    assert np.all(rel < 4 * recorded), (
        f"boundary={boundary}, body {body_x}: relative error moved from the "
        f"recorded {recorded} to {rel}. Still under the gate, but the gate's "
        "derivation is stale; re-measure it rather than editing this bound.")


def test_the_nu_branch_realizes_the_body_too():
    """``run_distributed``'s ``is_nu`` dispatch, which leg 4 also opened.

    The PEC-mask stage sits OUTSIDE that dispatch -- one call site per step
    body, guarded only on ``sharded_pec_mask is not None`` -- so both branches
    apply it, and the narrowed refusal admits a declared volume on both. A
    uniform-valued ``dz_profile`` makes this the same lattice reached through
    ``_assemble_materials_nu`` and the NU kernels, so the classification must
    agree even though the steppers differ numerically.

    One case, not the full matrix: the seam-face body, ``boundary="pec"``
    (v2 refuses NU + CPML separately). Measured 2026-09-15: 6.851e-06 /
    3.213e-05 with the body against a 4.282e-06 / 6.163e-05 no-body NU lane
    floor -- the same envelope as the uniform rows above, under the same gate.
    """
    rel = _rel("pec", SEAM_FACE_BODY_X, nu=True)
    assert np.all(rel < GATE), (
        f"the NU branch of the shard_map lane deviates from the NU "
        f"single-device lane by {rel}, gate {GATE:.0e}")
    floor = _rel("pec", None, nu=True)
    assert np.all(floor < GATE / 10), (
        f"the NU no-body lane floor is {floor}, within a decade of the gate "
        f"{GATE:.0e}. Re-derive the gate rather than raising it.")


@pytest.mark.parametrize("body_x", _BODIES)
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_dropping_the_mask_stage_reds_the_gate(boundary, body_x):
    """The gate can bind. Without the stage, the body is simply absent.

    This is the state ``distributed_v2`` was in before #1053 and the reason it
    refused rather than ran: measured here at 1.049e-01 to 1.435e+00 relative
    (pec), i.e. 100x to 1400x the gate. If this ever passes, the comparison
    above has stopped measuring whether the conductor is realized.
    """
    rel = _rel(boundary, body_x, drop_stage=True)
    assert np.max(rel) > GATE, (
        f"boundary={boundary}, body {body_x}: with the PEC-mask stage dropped "
        f"to a no-op the lane still agrees to {rel}, inside the gate "
        f"{GATE:.0e}. The gate is not binding on this fixture.")


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_the_empty_domain_control_sets_the_lane_floor(boundary):
    """The floor ``GATE`` is derived from.

    Same domain, same source, same probes, no conductor, no stage to run.
    What is left is the v2-vs-uniform lane difference itself. If this ever
    approaches ``GATE`` the gate has stopped separating a PEC-stage defect
    from float32 noise and must be re-derived, not raised.
    """
    rel = _rel(boundary, None)
    assert np.all(rel < GATE / 10), (
        f"boundary={boundary}: the empty-domain lane floor is {rel}, within a "
        f"decade of the gate {GATE:.0e}. Re-derive the gate.")
    recorded = np.asarray(MEASURED[(boundary, None)])
    assert np.all(rel < 4 * recorded), (
        f"boundary={boundary}: the lane floor moved from the recorded "
        f"{recorded} to {rel}. The gate's derivation is stale; re-measure it "
        "rather than editing this bound.")
