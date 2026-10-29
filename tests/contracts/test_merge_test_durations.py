"""merge_test_durations rounds prices to two significant figures and never to zero."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("merge_test_durations", ROOT / "scripts/ci/merge_test_durations.py")
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


@pytest.mark.parametrize(("seconds", "price"), [
    (1234.567, 1200.0), (12.3456, 12.0), (0.98765, 0.99), (0.0123456, 0.012),
    (0.000123, 0.00012), (2.0e-7, 2.0e-7), (0.0, 0.0), (95.0, 95.0), (99.6, 100.0),
])
def test_round_price_keeps_two_significant_figures(seconds, price):
    assert merge.round_price(seconds) == price


def test_merged_file_holds_rounded_prices_and_later_inputs_win(tmp_path):
    a, b, out = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "out.json"
    a.write_text(json.dumps({"t.py::x": 0.0004567, "t.py::y": 3.14159}))
    b.write_text(json.dumps({"t.py::y": 27.777}))
    assert merge.main(["merge", str(out), str(a), str(b)]) == 0
    assert json.loads(out.read_text()) == {"t.py::x": 0.00046, "t.py::y": 28.0}
