"""The #498 rfx-artifact -> referee-contract adapter re-shapes and never re-computes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "scripts" / "diagnostics" / "i498_rfx_artifact_to_referee_contract.py"
_ARTIFACT = (
    _REPO / "scripts" / "diagnostics" / "_i498_mixed_refplane_logs" / "measurement_369367257597_60p.json"
)

_spec = importlib.util.spec_from_file_location("i498_adapter", _SRC)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

# The per-bin magnitudes the committed run printed in its own table (run.log), first and last bin.
PUBLISHED = {
    0: {"S00": 0.3814, "S10": 0.9133, "S22": 0.0199},
    4: {"S00": 0.4027, "S10": 0.9057, "S22": 0.0341},
}


def _doc():
    if not _ARTIFACT.exists():  # pragma: no cover - artifact is committed
        pytest.skip(f"measurement artifact not present: {_ARTIFACT}")
    return json.loads(_ARTIFACT.read_text())


def test_adapter_reproduces_the_runs_own_published_magnitudes():
    """The re-shaping is correct in the only way that matters: it agrees with the run's own table."""
    view = _mod.adapt(_doc(), source=str(_ARTIFACT))
    s = np.asarray([[[complex(*p) for p in row] for row in fam] for fam in view["s_raw"]])
    assert s.shape == (2, 2, len(view["freqs_hz"]))
    for k, want in PUBLISHED.items():
        assert abs(s[0, 0, k]) == pytest.approx(want["S00"], abs=5e-5)
        assert abs(s[1, 0, k]) == pytest.approx(want["S10"], abs=5e-5)
        assert abs(s[1, 1, k]) == pytest.approx(want["S22"], abs=5e-5)


def test_adapter_carries_every_source_value_unchanged():
    """Not a single number is recomputed: every entry is the source entry."""
    doc = _doc()
    view = _mod.adapt(doc, source="x")
    flat = doc["s_matrix"]["S_raw"]
    n = len(view["freqs_hz"])
    for i in range(2):
        for j in range(2):
            for k in range(n):
                assert view["s_raw"][i][j][k] == list(map(float, flat[(i * 2 + j) * n + k]))
    assert view["freqs_hz"] == list(doc["fixture"]["freqs_hz"])


def test_adapter_refuses_a_transposed_port_family_order():
    doc = _doc()
    doc["s_matrix"]["port_families"] = ["msl", "wire"]
    with pytest.raises(SystemExit, match="port_families"):
        _mod.adapt(doc, source="x")


def test_adapter_refuses_a_shape_that_disagrees_with_the_frequency_count():
    doc = _doc()
    doc["s_matrix"]["S_raw_shape"] = [2, 2, 4]
    with pytest.raises(SystemExit, match="S_raw_shape"):
        _mod.adapt(doc, source="x")


def test_adapter_never_carries_the_projected_s():
    """result.S is an SVD projection; the contract is S_raw only."""
    view = _mod.adapt(_doc(), source="x")
    blob = json.dumps(view)
    assert "S_shipped_post_passivity" not in blob
    assert "S_wave" not in view
    assert view["projected_S_deliberately_omitted"]
