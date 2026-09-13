"""Paper figures retain the same concrete records that judge their settling."""
import hashlib
import json

import numpy as np
import pytest

from rfx.api._spec import ForwardResult
from rfx.probes.settling import probe_record_settling_witness
from validation.tmtt_paper._settling_record import retain_observation


@pytest.mark.parametrize("decay,status", [(0.001, "fail"), (0.1, "pass")])
def test_saved_observable_and_witness_replay_from_the_same_record(tmp_path, decay, status):
    time_series = np.exp(-decay * np.arange(200))[:, None]
    result = ForwardResult(time_series=time_series, settling_probe_info=((0, 2),))
    observable = np.array([1+2j, 3-4j])
    record = retain_observation(tmp_path, "point", result, observable)
    saved = json.loads((tmp_path / "point.json").read_text())
    assert record == saved
    assert saved["settling_verdict"] == status
    arrays = tmp_path / saved["arrays"]
    assert hashlib.sha256(arrays.read_bytes()).hexdigest() == saved["arrays_sha256"]
    with np.load(arrays) as raw:
        np.testing.assert_array_equal(raw["observable"], observable)
        db, witness = probe_record_settling_witness(
            raw["time_series"], tuple(map(tuple, raw["settling_probe_info"])), warn=False)
    assert db == saved["settling_db"]
    assert witness == saved["settling_witness"]


def test_missing_probe_coverage_is_retained_as_absent(tmp_path):
    result = ForwardResult(time_series=np.ones((20, 1)), settling_probe_info=())
    record = retain_observation(tmp_path, "absent", result, np.array([1.]))
    assert record["settling_verdict"] == "absent"
    assert record["settling_db"] is None
