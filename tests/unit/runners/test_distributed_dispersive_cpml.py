"""Always-on item 5 law C, spectrum bar S, and its tau negative control."""
import numpy as np
import pytest

from tests._distributed_dispersive_cpml import check, measure, record, scene

pytestmark = pytest.mark.distributed


def test_tau_negative_control():
    sim = scene('debye')
    one = np.asarray(sim.forward(n_steps=400, skip_preflight=True, checkpoint=False).time_series)
    changed = scene('debye', tau_factor=1.0001)
    multi = np.asarray(changed.forward(n_steps=400, skip_preflight=True, checkpoint=False).time_series)
    row = dict(kind='debye', entry='negative_control', devices=1, n=400, tau_factor=1.0001,
               **measure(one, multi, float(sim._build_nonuniform_grid().dt)))
    record(row)
    with pytest.raises(AssertionError, match='bar S') as failure:
        check(row, spectral_only=True)
    record(dict(entry='M2', assertion=str(failure.value), spectral=row['spectral']))
