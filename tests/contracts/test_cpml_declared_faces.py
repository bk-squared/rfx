"""Only declared CPML faces enter the E/H absorber windows."""

import inspect
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.contracts.boundary_cases import build
from tests.contracts.boundary_fields import COMPONENTS, is_aux


EXPECTED = {
    "pmc-cpml": {"y_lo", "y_hi", "z_lo", "z_hi"},
    "pec-zlo": {"x_lo", "x_hi", "y_lo", "y_hi", "z_hi"},
    "cpml": {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"},
}


def _run(case, entry, *, seed_psi=False):
    import rfx.boundaries.cpml as cpml

    jax.clear_caches()
    sim, declared = build(case, "nonuniform" if entry == "distributed_nu" else entry)
    assert {face for face, token in declared.items() if token == "cpml"} == EXPECTED[case]
    observed = {"apply_cpml_e": [], "apply_cpml_h": []}
    original_windows, original_scan = cpml.face_windows, jax.lax.scan
    seeded = []

    def windows(*args, **kwargs):
        result = original_windows(*args, **kwargs)
        caller = inspect.currentframe().f_back.f_code.co_name
        assert caller in observed, caller
        observed[caller].append({face for face, depth in result[3].items() if depth > 0})
        return result

    def scan(function, initial, *args, **kwargs):
        def seed(value):
            if is_aux(value):
                seeded.append(True)
                return value._replace(**{
                    name: jnp.full_like(array, .125)
                    for name, array in value._asdict().items()
                    if name.endswith(("_xlo", "_xhi"))})
            return value
        if seed_psi:
            initial = jax.tree_util.tree_map(seed, initial, is_leaf=is_aux)
        return original_scan(function, initial, *args, **kwargs)

    kwargs = dict(n_steps=1, compute_s_params=False)
    if entry in ("distributed", "distributed_nu"):
        assert len(jax.devices("cpu")) >= 2, "two CPU devices required"
        kwargs["devices"] = jax.devices("cpu")[:2]
    with patch.object(cpml, "face_windows", windows), patch.object(jax.lax, "scan", scan):
        result = sim.run(**kwargs)
        jax.block_until_ready(result.time_series)
        jax.effects_barrier()
    if seed_psi:
        assert seeded, "non-CPML psi was not seeded in the production scan"
    return result, observed


@pytest.mark.parametrize("entry", ["run", "nonuniform", "distributed", "distributed_nu"])
@pytest.mark.parametrize("case", ["pmc-cpml", "pec-zlo", "cpml"])
def test_only_declared_faces_are_windowed(case, entry):
    result, observed = _run(case, entry)
    for kernel, calls in observed.items():
        assert calls, f"{case}/{entry}: {kernel} was not reached"
        for faces in calls:
            assert faces == EXPECTED[case], (
                f"{case}/{entry}/{kernel}: windowed {sorted(faces)}; "
                f"declared CPML {sorted(EXPECTED[case])}")
    if case == "pmc-cpml":
        seeded, _ = _run(case, entry, seed_psi=True)
        for component in COMPONENTS:
            np.testing.assert_array_equal(
                np.asarray(getattr(seeded.state, component)).view(np.uint32),
                np.asarray(getattr(result.state, component)).view(np.uint32),
                err_msg=f"{entry}/{component}: non-CPML psi changed the one-step field")
        np.testing.assert_array_equal(
            np.asarray(seeded.time_series).view(np.uint32),
            np.asarray(result.time_series).view(np.uint32))
