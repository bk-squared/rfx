"""Unused decay monitor options warn once at the caller, only off default."""

import inspect
import warnings

import pytest

from rfx import Simulation
from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.simulation import run_until_decay


MESSAGE = (
    "decay_monitor_component/decay_monitor_position no longer affect until_decay; "
    "the stop reads the field energy after the drive. They will be removed."
)


@pytest.mark.parametrize("entry", ["run", "run_fixed", "run_until_decay"])
@pytest.mark.parametrize("options", [
    {}, {"component": "ez", "position": None},
    {"component": "hy"}, {"position": (.002, .002, .002)},
    {"component": "hy", "position": (.002, .002, .002)},
])
def test_unused_decay_monitor_warns_once_at_user_call(entry, options):
    if entry in ("run", "run_fixed"):
        sim = Simulation(domain=(.004,) * 3, dx=.001, freq_max=10e9, boundary="pec")
        call = sim.run
        kwargs = dict(until_decay=1e-3, decay_check_interval=2, decay_min_steps=2,
                      decay_max_steps=4, skip_preflight=True, compute_s_params=False)
        if entry == "run_fixed":
            kwargs = dict(n_steps=4, skip_preflight=True, compute_s_params=False)
        kwargs.update({"decay_monitor_" + k: v for k, v in options.items()})
        args = ()
    else:
        grid = Grid(domain=(.004,) * 3, dx=.001, freq_max=10e9, cpml_layers=0)
        call = run_until_decay
        args = (grid, init_materials(grid.shape))
        kwargs = dict(check_interval=2, min_steps=2, max_steps=4)
        kwargs.update({"monitor_" + k: v for k, v in options.items()})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        call(*args, **kwargs)
    found = [w for w in caught if w.category is DeprecationWarning and str(w.message) == MESSAGE]
    changed = options.get("component", "ez") != "ez" or options.get("position") is not None
    assert len(found) == int(changed), [str(w.message) for w in caught]
    if changed:
        assert (found[0].filename, found[0].lineno) == (__file__, line)
