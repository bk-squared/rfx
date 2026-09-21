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
