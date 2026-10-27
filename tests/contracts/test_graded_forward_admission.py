"""Graded run/forward must refuse unsupported ports at the same cause (#2076)."""
from pathlib import Path

import jax
import pytest

from tests.contracts.run_forward_census import BUILDERS, exception_record


@pytest.mark.parametrize("declaration", ("P-msl", "R-rlc"))
def test_graded_multi_device_refusal_has_one_cause(declaration):
    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2, "Root conftest must provide two host devices"
    source_root = Path(__file__).resolve().parents[2] / "rfx"
    sites = []
    with jax.default_device(devices[0]):
        for entry in ("run", "forward"):
            sim = BUILDERS[declaration]("multi-device-graded")
            options = {"distributed": True, "checkpoint": False} if entry == "forward" else {
                "compute_s_params": False,
            }
            with pytest.raises(NotImplementedError) as raised:
                getattr(sim, entry)(n_steps=1, devices=devices, skip_preflight=True, **options)
            # Reuse the census's innermost traceback frame and source-root rule.
            # The run exception supplies the expectation, never a product helper.
            record = exception_record(raised.value)
            filename, line = record["raising_site"].rsplit(":", 1)
            path = Path(filename).resolve()
            assert path.is_relative_to(source_root), record
            sites.append((path, int(line)))
    assert sites[0] == sites[1], sites
