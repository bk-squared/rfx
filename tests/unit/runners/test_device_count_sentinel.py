"""Device-count sentinel (roadmap W1.6).

conftest.py injects ``XLA_FLAGS=--xla_force_host_platform_device_count=2``
via ``os.environ.setdefault`` and only ``warnings.warn``s when fewer than
two devices are visible. Every 2-device test then ``pytest.skip``s
individually, so a late/lost flag silently degrades the whole
distributed/2-device lane to green skips.

This test FAILS (not skips) when the 2-device environment is missing,
unless a lane explicitly opts out by exporting ``RFX_ALLOW_SINGLE_DEVICE=1``.
"""

import os

import jax


def test_two_devices_visible_unless_opted_out():
    if os.environ.get("RFX_ALLOW_SINGLE_DEVICE") == "1":
        return
    n = len(jax.devices())
    assert n >= 2, (
        f"Only {n} JAX device(s) visible — the XLA_FLAGS 2-device injection "
        "from conftest.py did not take effect, so every 2-device/distributed "
        "test in this session is silently skipping. Either fix the flag "
        "injection (it must be set before jax initializes) or export "
        "RFX_ALLOW_SINGLE_DEVICE=1 for lanes that legitimately run "
        "single-device."
    )
