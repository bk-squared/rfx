"""#1237: actual Mie records, shared -40 dB bar, and unchanged sigma."""
import importlib.util
from pathlib import Path
import warnings

import numpy as np
import pytest

import rfx.rcs as rcs


@pytest.fixture
def mie_rig(monkeypatch):
    directory = Path(__file__).parents[1] / "fixtures" / "rcs_sphere_mie"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("rcs_settling_mie_rig", directory / "generate_fixture.py")
    rig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rig)
    return rig


@pytest.mark.parametrize("steps, expected_db, suspect", [
    (60, -1.5509324, True),
    (350, -46.6075916, False),
])
def test_mie_record_settling(mie_rig, steps, expected_db, suspect):
    mie_rig.N_STEPS = steps
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mie_rig.run_rfx()[4]
    ringdown = [w for w in caught if "ring-down" in str(w.message)]
    assert bool(ringdown) == suspect
    assert (result.settling_db > -40) == suspect
    # Reject a witness fed the wrong temporal window, even if it still fails.
    assert result.settling_db == pytest.approx(expected_db, abs=0.001)


def test_mie_sigma_bit_identical_without_recorders(mie_rig, monkeypatch):
    mie_rig.N_STEPS = 60
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        instrumented = mie_rig.run_rfx()[4]
        original_run = rcs.run

        def without_recorders(*args, **kwargs):
            kwargs.pop("probes")
            return original_run(*args, **kwargs)

        monkeypatch.setattr(rcs, "run", without_recorders)
        monkeypatch.setattr(rcs, "_rcs_settling_db", lambda _: None)
        baseline = mie_rig.run_rfx()[4]
    for name in ("rcs_linear", "rcs_dbsm", "monostatic_rcs"):
        before, after = getattr(baseline, name), getattr(instrumented, name)
        assert before.dtype == after.dtype
        assert before.shape == after.shape
        assert before.tobytes() == after.tobytes(), name


def test_result_defaults_and_scattering_response():
    arrays = [np.zeros(1)] * 6
    result = rcs.RCSResult(*arrays)
    response = rcs.ScatteringResponse(result, *arrays[:4], "ez")
    assert result.settling_db is response.settling_db is None
