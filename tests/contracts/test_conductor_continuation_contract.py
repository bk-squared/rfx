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


def test_occupied_patch_ground_continues_on_both_high_faces():
    module = lib.load_module("tests/oracle/test_lossless_open_domain_ringdown_does_not_grow.py")
    sim = module._build(n=2, pad_h=10, cpml=4)
    try:
        grid, arrays, poles, nodes = assembled_arrays(sim)
        bad, exceptions = violations(sim, grid, arrays, poles, nodes)
        assert not exceptions
        assert not bad, bad
        for axis in (0, 1):
            assert arrays["pec_mask"].take(-2, axis=axis).any()
    finally:
        jax.clear_caches()


def test_port_exception_is_only_the_terminal_entry():
    import numpy as np
    from rfx import Box, Simulation
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=2, pec_faces={"z_lo"})
    sim.add(Box((0., 2., 3.), (8., 3., 3.)), material="pec")
    sim.add_port((4., 2., 0.), component="ez", extent=3.)
    sim.add(Box((.1, 5., 3.), (7.9, 7., 5.)), material="pec")
    grid, arrays, poles, nodes = assembled_arrays(sim)
    bad, exceptions = violations(sim, grid, arrays, poles, nodes)
    assert not bad, bad
    assert {item[2] for item in exceptions} == {"port-terminal:pec"}
    assert arrays["sheet_0"][2].any()
    assert not arrays["sheet_0"][:2].any()
    assert arrays["pec_mask"][:2].any()
    # Corrupt the other entry outside the terminal sheet's transverse support.
    arrays["pec_mask"] = arrays["pec_mask"].copy()
    occupied = np.argwhere(arrays["pec_mask"][0])
    j, k = occupied[0]
    arrays["pec_mask"][0, j, k] = False
    bad, _ = violations(sim, grid, arrays, poles, nodes)
    assert any(b["array"] == "pec_mask" and b["differences"] == 1 for b in bad)
