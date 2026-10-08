"""Kept finishing-round defects; each invocation must make its judge red."""
import sys

import pytest


def main(mode):
    if mode == "explicit-periodic-falls-back":
        from rfx import Simulation
        from rfx.boundaries.spec import BoundarySpec
        original = Simulation._build_grid

        def implicit(sim, **kwargs):
            saved = sim._boundary_spec
            sim._boundary_spec = BoundarySpec.uniform("cpml")
            try:
                return original(sim, **kwargs)
            finally:
                sim._boundary_spec = saved

        Simulation._build_grid = implicit
        path = "tests/contracts/test_boundary_finishing.py"
        judge = "test_explicit_periodic_keeps_declared_ring"
    elif mode == "refused-registry-row-admits":
        from tests.contracts import test_realized_boundary as registry
        original = registry.measure

        def admit(case, entry):
            if case == "waveguide-cpml":
                case = "waveguide-default"
            return original(case, entry)

        registry.measure = admit
        path = "tests/contracts/test_realized_boundary.py"
        judge = "test_no_unlisted_departures and waveguide-cpml and not nonuniform and not wire-fast and not adi"
    elif mode == "default-twin-treated-as-explicit":
        from rfx.boundaries import features
        original = features.boundary_was_explicit

        def explicit(sim):
            original(sim)
            return True

        features.boundary_was_explicit = explicit
        path = "tests/contracts/test_realized_boundary.py"
        judge = "test_default_waveguide_twin_records_its_walls"
    elif mode == "default-wrap-invariance-unconditional":
        from rfx.boundaries import tfsf
        original = tfsf.invariant

        def invariant(*args):
            original(*args)
            return True, ""

        tfsf.invariant = invariant
        path = "tests/contracts/test_boundary_finishing.py"
        judge = "test_default_wrap_accepts_slab_and_refuses_original_box"
    else:
        raise ValueError(mode)
    return pytest.main([path, "-q", "-k", judge, "--tb=short",
                        "--basetemp=.pr3/finish/mutant-" + mode,
                        "-o", "cache_dir=.pr3/pytest-cache"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
