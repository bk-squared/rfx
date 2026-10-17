"""Always-on pole snapshots using the shared one-device law C/bar S body."""
import pytest

from tests._distributed_cpml_cross_trace import PR_PREFIXES
from tests._distributed_dispersive_cpml import snapshot_parity
from tests.unit.runners.test_distributed_edge_averaged_materials import _debye_box

pytestmark = pytest.mark.distributed


@pytest.mark.parametrize("kind", ["debye", "lorentz"])
def test_cpml_dispersive_pr_snapshots(kind, monkeypatch):
    """Use law C/bar S for accumulated pole rounding (rfx-archive records/20261017-item5-dispersive)."""
    snapshot_parity(kind, PR_PREFIXES, monkeypatch, entry='snapshot_pr', scene_builder=_debye_box)
