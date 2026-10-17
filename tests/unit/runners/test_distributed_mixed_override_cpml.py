"""#1834 mixed graded CPML drawn-epsilon override: fields and gradient."""
import pytest
from tests._distributed_dispersive_cpml import override_parity

pytestmark = pytest.mark.distributed


def test_mixed_override_matches_one_device():
    override_parity()
