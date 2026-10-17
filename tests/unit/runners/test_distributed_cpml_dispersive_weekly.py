"""Weekly pole law C/bar S records and unchanged R15l prefix sweeps."""
import pytest

from tests._distributed_cpml_cross_trace import WEEKLY_PREFIXES, cross_trace
from tests.unit.runners.test_distributed_edge_averaged_materials import (
    _cpml_1302_box,
    _debye_box,
    _devices,
)

pytestmark = [pytest.mark.distributed, pytest.mark.slow]


@pytest.mark.parametrize("boundary,lane", [("cpml", "fwd")])
@pytest.mark.parametrize("kind", ["debye", "lorentz", "mixed"])
def test_a_dispersive_model_matches_one_device(boundary, lane, kind, monkeypatch):
    """Debye/Lorentz use law C/bar S for accumulated pole rounding (rfx-archive records/20261017-item5-dispersive).

    The wrapped mixed (Debye+Lorentz) row measured at most 6.04 peak ULP over
    200/400/800 steps on Mac, leaving 2.96 ULP below the unchanged 9-ULP bar
    (review B, rfx-archive records/20261017-item5-dispersive).
    """
    if kind in ('debye', 'lorentz'):
        from tests._distributed_dispersive_cpml import snapshot_parity
        snapshot_parity(kind, (400, 800, 1600), monkeypatch, entry='snapshot_weekly')
        return
    cross_trace(lambda: _debye_box(boundary, lane, kind), lane, _devices(2),
                f"{kind}/{boundary}/{lane}", prefixes=WEEKLY_PREFIXES)


# Graded law C + bar S rows (800/1600) live in the box/length-specific
# test_distributed_dispersive_cpml_*_weekly.py files to bound CPU file time.
@pytest.mark.parametrize('kind', ['debye', 'lorentz'])
def test_uniform_dispersive_cpml_box_matches_one_device(kind):
    cross_trace(lambda: _cpml_1302_box(kind, 'fwd'), 'fwd', _devices(2),
                f'cpml1302/{kind}/fwd', prefixes=WEEKLY_PREFIXES)
