"""Declared PEC volumes, sheets and wires on one- and two-device meshes."""
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
                                 radius=0.), material="pec")  # legacy PEC filament
        sim.add_source(position=(4e-3, 6e-3, 6e-3), component="ez",
                       amplitude_kind="field")
        sim.add_probe(position=(12e-3, 6e-3, 6e-3), component="ez")
    return sim


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
@pytest.mark.parametrize("n_devices", [1, 2])
def test_distributed_sheet_and_wire_match_single_device(kind, n_devices):
    from rfx.runners.distributed_v2 import run_distributed
    from tests.unit.runners.test_distributed_v2_pec_body_seam import GATE

    if len(jax.devices()) < n_devices:
        pytest.skip("need two virtual CPU devices")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = np.asarray(_build(kind).run(n_steps=300, skip_preflight=True).time_series)
        actual = np.asarray(run_distributed(_build(kind), n_steps=300,
                            devices=jax.devices()[:n_devices]).time_series)
    peak = np.max(np.abs(reference), axis=0)
    assert np.all(peak > 0)
    assert np.all(np.max(np.abs(actual - reference), axis=0) / peak < GATE)


def test_the_one_device_path_runs_a_declared_volume():
    """#1296: one device used to be handed to the pmap runner, which refused
    this volume (#1055). It now runs the shard_map path on a one-device mesh,
    which realizes the volume as it does at two devices. The probe sits
    inside the metal, so the trace reads zero when the volume is realized and
    the field of the empty box when it is dropped -- the #931 witness.
    Numeric parity with the single-device lane is gated in
    ``test_distributed_v2_one_device.py``.
    """
    from rfx.runners.distributed_v2 import run_distributed as shmap_run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inside = np.asarray(shmap_run(_build("volume"), n_steps=40,
                                      devices=jax.devices()[:1]).time_series)
        empty = np.asarray(shmap_run(_build("none"), n_steps=40,
                                     devices=jax.devices()[:1]).time_series)
    assert np.max(np.abs(empty)) > 0, "vacuous fixture: the empty box reads zero"
    assert np.max(np.abs(inside)) <= 1e-6 * np.max(np.abs(empty)), (
        f"the probe inside the declared PEC volume reads "
        f"{np.max(np.abs(inside)):.3e} against {np.max(np.abs(empty)):.3e} "
        "with the volume deleted: the one-device path dropped the metal")
