"""Step A / #1844: graded assembly under a user jit, both public entries."""

import contextlib

import jax
import numpy as np
import pytest

from rfx.core import jax_utils
from tests.contracts.multi_device_assembly import make_call, outputs

pytestmark = pytest.mark.distributed


@pytest.fixture
def devices():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two host CPU devices")
    return devices[:2]


@pytest.mark.parametrize("lane", ["graded_forward", "graded_run"])
@pytest.mark.parametrize("scene", ["S1", "S2", "S3", "S4"])
def test_user_jit_preserves_results(devices, lane, scene):
    call, eps, _ = make_call(lane, scene, devices)
    args = (eps,) if lane.endswith("forward") else ()

    def projected(*args):
        return outputs(call(*args))

    eager = jax.tree.map(np.asarray, projected(*args))
    compiled = jax.tree.map(np.asarray, jax.jit(projected)(*args))
    assert eager.keys() == compiled.keys()
    assert np.max(np.abs(eager["time_series"])) > 0
    for name, expected in eager.items():
        actual = compiled[name]
        assert np.isfinite(actual).all(), name
        if scene == "S3":
            # Ledger: rfx-archive rfx/records/20261021-stepA-multidevice-setup/.
            # S3 eager/jit wire-port trap: 4.5 float32 ULP of the peak.
            peak = np.float32(np.max(np.abs(expected), initial=0))
            tolerance = 9 * float(np.spacing(peak))
            np.testing.assert_allclose(
                actual, expected, rtol=0, atol=tolerance, err_msg=name
            )
        else:
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes(), name


@pytest.mark.parametrize("lane", ["graded_forward", "graded_run"])
def test_assembly_region_once_at_trace_time(devices, monkeypatch, lane):
    original = jax_utils.declaration_setup
    entries = []

    @contextlib.contextmanager
    def counted():
        entries.append(True)
        with original():
            yield

    monkeypatch.setattr(jax_utils, "declaration_setup", counted)
    call, eps, _ = make_call(lane, "S2", devices)
    args = (eps,) if lane.endswith("forward") else ()

    def projected(*args):
        return outputs(call(*args))

    jax.block_until_ready(projected(*args))
    assert len(entries) == 0
    jax.block_until_ready(jax.jit(projected)(*args))
    assert len(entries) == 1
