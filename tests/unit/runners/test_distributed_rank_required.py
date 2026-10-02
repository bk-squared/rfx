"""Missing device rank must fail before a slab silently loses its x faces."""
import inspect

import pytest

from rfx.runners import _distributed_common as common, distributed_nu as nu, distributed_v2 as uniform


_SHMAPS = sorted({fn for module in (common, nu, uniform) for name, fn in vars(module).items()
                  if name.endswith("_shmap") and inspect.isfunction(fn)},
                 key=lambda fn: fn.__name__)


@pytest.mark.parametrize("helper", _SHMAPS, ids=lambda fn: fn.__name__)
def test_shard_helpers_require_rank_input(helper):
    # Supply every other required parameter; missing rank must fail at binding,
    # before these sentinel values can enter a numerical operation.
    args = {name: None for name, p in inspect.signature(helper).parameters.items()
            if p.default is inspect.Parameter.empty and name != "ranks"}
    with pytest.raises(TypeError, match="required keyword-only argument: 'ranks'"):
        helper(**args)


@pytest.mark.parametrize("helper", [common._apply_cpml_e_distributed,
                                     common._apply_cpml_h_distributed,
                                     nu._apply_cpml_e_local_nu,
                                     nu._apply_cpml_h_local_nu])
def test_cpml_rejects_absent_rank(helper):
    args = {name: None for name, p in inspect.signature(helper).parameters.items()
            if p.default is inspect.Parameter.empty}
    with pytest.raises(ValueError, match="slab rank must be supplied as data"):
        helper(**args)
