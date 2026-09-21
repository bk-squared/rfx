"""What each distributed lane does with a declared PEC conductor.

Both distributed runners used to assemble ``pec_mask`` and never apply it:
their step bodies called the DOMAIN-FACE PEC alone. That gap predates #931.
What #931 added was a refusal whose remedy told the user to redraw an
unsupported sheet as a VOLUME — advice that on these lanes produced a run with
the metal still missing and nothing to show for it. Measured before that fix:
a two-device run probing inside a declared PEC Box returned a trace
bit-identical to the same model with the Box deleted. So the refusal was
widened to cover volumes too, and the remedy named what worked.

#1053 closed the gap on the shard_map lane only. ``distributed_v2`` now shards
``pec_mask`` and applies it in both step bodies at the #1041 ordering, so a
declared PEC VOLUME runs there and is gated against the single-device lane by
``test_distributed_v2_pec_body_seam.py``. Sheets and sub-cell wires own no
cell, nothing on that lane carries them, and they are still refused.

The pmap lane in ``rfx/runners/distributed.py`` still drops the mask, so its
refusal — a separate string — still covers all three kinds (#1055). The two
messages are therefore checked separately: asserting one set of substrings
against both is what would let a stale sentence survive on the lane it has
stopped being true of.
"""
# Simulate 2 devices on CPU. Must be set BEFORE importing JAX.
import os  # noqa: I001

os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2"
)

import warnings  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rfx import Box, PolylineWire, Simulation  # noqa: E402

DOMAIN = (24e-3, 12e-3, 12e-3)
DX = 1e-3


def _build(kind):
    """``kind`` in ``{"none", "volume", "sheet", "wire"}``.

    ``volume`` is the #931 fixture: a PEC ``Box`` whose x span 10–14 mm on a
    24 mm domain straddles the 2-rank seam at 12 mm. ``sheet`` is the same
    footprint drawn with zero thickness, ``wire`` a sub-cell filament through
    the same region — one of each kind the lane can be handed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, boundary="pec")
        if kind == "volume":
            sim.add(Box((10e-3, 2e-3, 2e-3), (14e-3, 10e-3, 10e-3)),
                    material="pec")
        elif kind == "sheet":
            sim.add(Box((10e-3, 2e-3, 6e-3), (14e-3, 10e-3, 6e-3)),
                    material="pec")
        elif kind == "wire":
            sim.add(PolylineWire(((10e-3, 6e-3, 6e-3), (14e-3, 6e-3, 6e-3)),
                                 radius=0.2e-3), material="pec")
        sim.add_source(position=(4e-3, 6e-3, 6e-3), component="ez",
                       amplitude_kind="field")
        sim.add_probe(position=(12e-3, 6e-3, 6e-3), component="ez")
    return sim


def _assert_pmap_remedy_is_honest(msg):
    """The pmap lane's message. It still drops the mask for all three kinds,
    so it still names the volume and still has to say that redrawing a sheet
    as one buys nothing there (#1055)."""
    assert "PEC volume" in msg, msg
    assert "does NOT help" in msg, (
        "the refusal must say that redrawing as a volume is not a remedy "
        "on this lane")
    assert "sim.run()" in msg, msg


def _assert_shmap_remedy_is_honest(msg):
    """The shard_map lane's message, after #1053.

    Deliberately NOT the assertions above. "does NOT help" became false here
    the moment the lane started realizing volumes, and a shared helper would
    have forced the sentence to stay. What the message must still do is name
    the kinds it refuses, say what DOES run, and point at a lane that realizes
    everything.
    """
    assert "SHEETS" in msg and "WIRES" in msg, msg
    assert "does NOT help" not in msg, (
        "this lane realizes a declared PEC volume since #1053, so the message "
        "must not still tell the user that redrawing as one is useless")
    assert "#1053" in msg and "VOLUME" in msg, (
        "the refusal must say that a declared PEC volume DOES run here, "
        "otherwise it reads as a refusal of all declared PEC")
    assert "sim.run()" in msg, msg


def test_the_shmap_distributed_lane_realizes_a_declared_volume():
    """#1053 leg 4: the volume half of the refusal is gone because the metal
    is now there.

    The numeric gate lives in ``test_distributed_v2_pec_body_seam.py`` —
    three bodies, two boundary kinds, each against the same model on one
    device. This test reuses its seam-face body, the one placement whose
    metal face lands on rank 1's first real cell and therefore witnesses the
    stage's hook point, so that the file which used to assert "this raises"
    asserts "this runs, and agrees".
    """
    if len(jax.devices()) < 2:
        pytest.skip("need 2 virtual devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    from tests.unit.runners.test_distributed_v2_pec_body_seam import (
        GATE, SEAM_FACE_BODY_X, _rel)

    rel = _rel("pec", SEAM_FACE_BODY_X)
    assert np.all(rel < GATE), (
        f"a declared PEC body on sim.run(devices=...) deviates from the "
        f"single-device lane by {rel}, gate {GATE:.0e}")


@pytest.mark.parametrize("kind", ["sheet", "wire"])
def test_the_shmap_distributed_lane_refuses_a_sheet_and_a_wire(kind):
    """Neither owns a cell, and the cell mask is the only carrier this lane
    has, so both would be absent from every rank with no sign of it."""
    if len(jax.devices()) < 2:
        pytest.skip("need 2 virtual devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    from rfx.runners.distributed_v2 import run_distributed

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError) as excinfo:
            run_distributed(_build(kind), n_steps=4)
    msg = str(excinfo.value)
    _assert_shmap_remedy_is_honest(msg)
    assert ("PEC sheet(s)" if kind == "sheet" else "sub-cell wire(s)") in msg, (
        f"the refusal must name what was declared; got: {msg}")

    # the control: without the conductor the same model runs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_distributed(_build("none"), n_steps=4)
    assert res.time_series is not None


def test_the_one_device_fast_path_still_refuses_a_volume():
    """The seam between the two runners, pinned rather than left to surprise.

    ``distributed_v2.run_distributed`` delegates to the pmap runner at
    ``n_devices == 1``, and that runner still drops the mask — so the same
    call refuses a declared volume at one device and runs it at two. Nobody
    reaches this through the public API: ``rfx/api/_execute.py`` routes to
    this lane only for ``len(devices) > 1``, and one device takes the ordinary
    single-device lane, which realizes the volume. It is reachable by calling
    the runner directly, and #1055 is where it closes.
    """
    from rfx.runners.distributed_v2 import run_distributed as shmap_run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError) as excinfo:
            shmap_run(_build("volume"), n_steps=4, devices=jax.devices()[:1])
    # the pmap runner's message, not this lane's
    _assert_pmap_remedy_is_honest(str(excinfo.value))


def test_the_pmap_distributed_lane_refuses_a_declared_volume():
    """The lane ``run_distributed_v2`` delegates to at one device. It still
    drops the mask, so its refusal still covers volumes (#1055)."""
    from rfx.runners.distributed import run_distributed as pmap_run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError) as excinfo:
            pmap_run(_build("volume"), n_steps=4, devices=jax.devices()[:1])
    _assert_pmap_remedy_is_honest(str(excinfo.value))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pmap_run(_build("none"), n_steps=4, devices=jax.devices()[:1])
    assert res.time_series is not None
