"""#1053 physics gate: a realized PEC body straddling the shard seam.

``rfx/runners/distributed_v2.py`` assembles ``pec_mask`` and drops it: its step
bodies apply the DOMAIN-FACE PEC alone, so no declared conductor reaches the
field update. Rather than run with the metal missing, the lane refuses
(``NotImplementedError``, ``distributed_v2.py:573``). #1053 ports
``distributed_nu``'s realized-PEC mask stage across, which makes the refusal
unnecessary for a volume and lets this file measure what it is for.

Until leg 4 narrows that refusal to sheets and wires the two comparison tests
here are ``xfail(raises=NotImplementedError, strict=True)``. They are red for
exactly one reason, and ``strict`` means they fail the moment they become red
for a different one -- or green for the wrong one.

The reference is the SAME model on ONE device through the uniform lane, which
is the only lane in the repo with no seam. This is not a tolerance-free
comparison: the two lanes evaluate the same physics through different kernels
and different float32 fusions.

GATE DERIVATION
---------------
Measured on this host (2026-09-15, 2 virtual CPU devices, jax 0.10.2, float32,
300 steps) as ``max|multi - single| / peak(|single|)`` per probe, on the
geometry below with the PEC body DELETED -- the lane difference with no
conductor anywhere, which is the floor a body-parity gate has to clear:

    boundary   probe 12mm (rank 0)   probe 20mm (rank 1)
    pec        5.100e-06             7.450e-05
    cpml       1.991e-06             2.284e-05

Those four numbers reproduce #1041's interior-source control to four digits
(``test_distributed_v2_seam_source_order.py``, 5.100e-06 / 7.450e-05 /
1.629e-06 / 2.529e-05) because it is the same domain, the same source cell and
the same probe cells. The rank-1 probe carries the larger figure because its
own peak is 7% of the rank-0 probe's, so the same absolute float32 difference
divides by a much smaller number.

``GATE = 1e-3`` is 13x the worst floor (7.450e-05) and 210x below the defect it
has to catch: ``ac782d4f`` (#931 T3) measured a seam-cell PEC body under the
WRONG stage order at 2.107e-01 final-step relative error on the nu lane, and
7.773e-08 once the E ghost exchange became the last stage of the E half-step.
The same 1e-3 is #1041's gate on this geometry, for the same reason.

Why NOT 5e-5, the number the nu lane uses: 5e-5 is the Class B final-step
tolerance of ``tests/_distributed_nu_tolerances.py:101-106``, and it gates the
NU DISTRIBUTED lane against the NU SINGLE-DEVICE lane -- the same kernels on
the same grid type, where the only difference is the decomposition. This file
compares v2's shard_map kernels against the uniform single-device stepper, and
the measured floor for that pair on this geometry is 7.450e-05, ABOVE 5e-5. A
5e-5 gate here would pin float32 fusion rather than the PEC stage, and would be
red on the day it was written. The floor is measured by
``test_the_empty_domain_control_sets_the_lane_floor`` below, which runs and is
green TODAY, so the derivation stays live rather than becoming a comment.

PREFLIGHT, quoted verbatim -- the body is 5 cells on every axis and preflight
says so on every build in this file (single-device and distributed alike):

    [PREFLIGHT] PEC 'pec' x-extent 5mm = 5.0 cells - volume under-resolved (a
    PEC volume's curved/edge features need >=5 cells; a 1-2 cell slab is
    realized as drawn, with walls on both faces).

and the same line for y and z. It is advisory and it is about absolute
accuracy, not about lane parity: BOTH sides of every comparison here carry the
identical under-resolved body, so the advisory cancels in the difference. The
empty-domain control reports ``All checks passed (NTFF advisory tier; the
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

from rfx import Box, Simulation  # noqa: E402

DX = 1e-3
NX_CELLS = 31        # -> nx = 32 nodes, so nx_per_rank = 16 and pad_x = 0
NYZ_CELLS = 15       # -> ny = nz = 16
CENTER = 8e-3
SOURCE_X = 8e-3      # 8 cells from the seam, inside rank 0
PROBE_LO_X = 12e-3   # 4 cells inside rank 0
PROBE_HI_X = 20e-3   # 4 cells inside rank 1
N_STEPS = 300

#: x span of the PEC body, in metres. The rank seam sits between global node
#: 15 (rank 0's last real cell) and node 16 (rank 1's first real cell).
SEAM_BODY_X = (14e-3, 19e-3)      # nodes 14-19: STRADDLES the seam
INTERIOR_BODY_X = (22e-3, 27e-3)  # nodes 22-27: wholly inside rank 1
BODY_YZ = (6e-3, 11e-3)           # 5 cells, centred on the source/probe line

GATE = 1e-3

#: The measured no-body lane floor, per boundary, per probe. Kept as data so
#: the control test below asserts against the same numbers the docstring
#: derives the gate from.
FLOOR = {
    "pec": (5.100e-06, 7.450e-05),
    "cpml": (1.991e-06, 2.284e-05),
}


def _build(boundary, body_x):
    """The fixture. ``body_x=None`` deletes the conductor."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(
            freq_max=15e9,
            domain=(NX_CELLS * DX, NYZ_CELLS * DX, NYZ_CELLS * DX),
            dx=DX, boundary=boundary,
            cpml_layers=6 if boundary == "cpml" else 0)
        if body_x is not None:
            sim.add(Box((body_x[0], BODY_YZ[0], BODY_YZ[0]),
                        (body_x[1], BODY_YZ[1], BODY_YZ[1])),
                    material="pec")
        sim.add_source((SOURCE_X, CENTER, CENTER), "ez",
                       amplitude_kind="field")
        sim.add_probe((PROBE_LO_X, CENTER, CENTER), "ez")
        sim.add_probe((PROBE_HI_X, CENTER, CENTER), "ez")
    return sim


def _single(boundary, body_x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(
            _build(boundary, body_x).run(n_steps=N_STEPS).time_series)


def _multi(boundary, body_x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(
            _build(boundary, body_x).run(
                n_steps=N_STEPS, devices=jax.devices()[:2]).time_series)


def _rel_per_probe(boundary, body_x):
    """max|multi - single| / peak(|single|) for each of the two probes.

    Also proves the fixture is not vacuous in the ONE way that matters here:
    the body must change the single-device trace by far more than ``GATE``,
    or a lane that silently drops the conductor would pass by arithmetic.
    """
    ts_single = _single(boundary, body_x)
    peaks = np.max(np.abs(ts_single), axis=0)
    assert np.all(peaks > 1e-6), (
        f"vacuous fixture: single-device probe peaks {peaks}")

    if body_x is not None:
        ts_empty = _single(boundary, None)
        body_effect = (np.max(np.abs(ts_single - ts_empty), axis=0)
                       / np.max(np.abs(ts_empty), axis=0))
        assert np.max(body_effect) > 100 * GATE, (
            f"boundary={boundary}, body {body_x}: deleting the conductor "
            f"moves the single-device trace by only {body_effect}, which is "
            f"not >> the gate {GATE:.0e}. A lane that dropped the body would "
            "pass this comparison; re-place the body or the probes.")

    ts_multi = _multi(boundary, body_x)
    assert ts_multi.shape == ts_single.shape == (N_STEPS, 2), (
        f"shapes: multi {ts_multi.shape}, single {ts_single.shape}")
    return np.max(np.abs(ts_multi - ts_single), axis=0) / peaks


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
@pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason=(
        "#1053 leg 4 has not lifted the declared-PEC-volume refusal in "
        "rfx/runners/distributed_v2.py. Until it does this lane cannot run "
        "the fixture at all, and that refusal -- not a numeric failure -- is "
        "what this xfail records. strict=True so the test fails if it goes "
        "green, or if it goes red for any other reason."),
)
def test_pec_body_straddling_the_seam_matches_the_single_device_lane(boundary):
    """The gate. A realized PEC block owned by BOTH ranks.

    This is the fixture #1041 could not build. Its failure mode is specific:
    the mask stage acts on real cells only, so rank 0's right E ghost is a
    copy of rank 1's finished first real row -- and only if the E ghost
    exchange runs AFTER the mask stage. With the exchange first, ``ac782d4f``
    measured 2.107e-01 against a 5e-5 gate on the nu lane.
    """
    rel = _rel_per_probe(boundary, SEAM_BODY_X)
    assert rel[0] < GATE, (
        f"boundary={boundary}: PEC body across the seam, probe 4 cells into "
        f"rank 0 deviates from the single-device lane by {rel[0]:.3e} "
        f"(gate {GATE:.0e}). Suspect the PEC-mask stage position: the E ghost "
        "exchange must be the LAST stage of the E half-step (#1053 leg 2).")
    assert rel[1] < GATE, (
        f"boundary={boundary}: PEC body across the seam, probe inside rank 1 "
        f"deviates by {rel[1]:.3e} (gate {GATE:.0e})")


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
@pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason=(
        "same refusal as the seam case above; #1053 leg 4 lifts it."),
)
def test_pec_body_inside_one_rank_matches_the_single_device_lane(boundary):
    """The control that makes the seam result readable.

    The same block moved 6 cells clear of the seam, wholly inside rank 1. No
    ghost row carries a PEC cell, so the stage's position relative to the E
    exchange cannot matter here. If this reds while the seam case is green,
    the defect is in the mask stage itself and not in where it is hooked.
    """
    rel = _rel_per_probe(boundary, INTERIOR_BODY_X)
    assert np.all(rel < GATE), (
        f"boundary={boundary}: PEC body inside one rank deviates by {rel} "
        f"from the single-device lane (gate {GATE:.0e}). This placement is "
        "insensitive to the E-exchange hook point, so a failure here is the "
        "mask stage itself.")


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_the_empty_domain_control_sets_the_lane_floor(boundary):
    """The floor ``GATE`` is derived from. Runs and is GREEN today.

    Same domain, same source, same probes, no conductor. What is left is the
    v2-vs-uniform lane difference itself. If this ever approaches ``GATE`` the
    gate above has stopped separating a PEC-stage defect from float32 noise
    and must be re-derived, not raised.
    """
    rel = _rel_per_probe(boundary, None)
    expected = np.asarray(FLOOR[boundary])
    assert np.all(rel < GATE / 10), (
        f"boundary={boundary}: the empty-domain lane floor is {rel}, within a "
        f"decade of the gate {GATE:.0e}. Re-derive the gate before trusting "
        "test_pec_body_straddling_the_seam_matches_the_single_device_lane.")
    assert np.all(rel < 4 * expected), (
        f"boundary={boundary}: the lane floor moved from the recorded "
        f"{expected} to {rel}. The gate's derivation is stale; re-measure it "
        "rather than editing this bound.")
