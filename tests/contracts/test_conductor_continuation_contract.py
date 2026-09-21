"""Absorber columns must contain the adjacent solved cross-section."""
import jax
import pytest

import _example_fidelity_lib as lib
from _conductor_continuation_contract import assembled_arrays, violations


CASES = [(path, builder, variant)
         for path, entry in sorted(lib.CLASSIFICATION.items()) if entry.kind == "audited"
         for builder in entry.builders for variant in builder.variants]


@pytest.mark.parametrize("path,builder,variant", CASES,
                         ids=[f"{p}:{b.fn}:{v.label}" for p, b, v in CASES])
def test_absorbing_columns_equal_the_face(path, builder, variant):
    with lib.build_only():
        module = lib.load_module(path)
        result = getattr(module, builder.fn)(**variant.kwargs(module))
        sim = result if builder.result_index is None else result[builder.result_index]
        try:
            grid, arrays, poles, nodes = assembled_arrays(sim)
            bad, _ = violations(sim, grid, arrays, poles, nodes)
            assert not bad, bad
        finally:
            jax.clear_caches()


def test_reached_conductor_is_a_live_array_witness():
    from rfx import Box, Simulation
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=2)
    sim.add(Box((0., 2., 2.), (8., 6., 6.)), material="pec")
    grid, arrays, poles, nodes = assembled_arrays(sim)
    bad, exceptions = violations(sim, grid, arrays, poles, nodes)
    assert not exceptions
    assert not bad, bad
    assert arrays["pec_mask"][:2].any()
    assert arrays["fenced_y"][:2].any()
    assert (arrays["tensor_y"][:2] == 0.).any()
    # A single removed pad cell is visible without re-running any helper.
    arrays["pec_mask"] = arrays["pec_mask"].copy()
    arrays["pec_mask"][0, 4, 4] = False
    bad, _ = violations(sim, grid, arrays, poles, nodes)
    assert any(item["array"] == "pec_mask" and item["axis"] == 0
               and item["side"] == 0 and item["differences"] == 1 for item in bad)
