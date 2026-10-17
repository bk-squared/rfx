"""Always-on item 5 debye box: law C and spectrum bar S at 400 steps."""
import pytest
from tests._distributed_dispersive_cpml import parity

pytestmark = pytest.mark.distributed


def test_law_c_and_bar_s(monkeypatch):
    parity('debye', 400, monkeypatch)
