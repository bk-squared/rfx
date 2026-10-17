"""Weekly item 5 debye box at 800 steps; one file bounds CPU runtime."""
import pytest
from tests._distributed_dispersive_cpml import parity

pytestmark = [pytest.mark.distributed, pytest.mark.slow]


def test_law_c_and_bar_s(monkeypatch):
    parity('debye', 800, monkeypatch)
