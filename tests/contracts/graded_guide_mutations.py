"""Addendum 5d defects, preserving production helper calls."""
import sys

import pytest


def main(mode):
    from rfx.boundaries import features
    if mode == "graded-refusal-removed":
        original = features._admit_graded_guide

        def admit(sim, lane):
            try:
                return original(sim, lane)
            except ValueError:
                return None

        features._admit_graded_guide = admit
        judge = "test_default_graded_refuses_before_field_step"
    elif mode == "graded-explicit-walls-dropped":
        from rfx import Simulation
        from rfx.boundaries.spec import BoundarySpec
        original = Simulation._build_nonuniform_grid

        def drop(sim, **kwargs):
            saved = sim._boundary_spec
            sim._boundary_spec = BoundarySpec.uniform("cpml")
            try:
                return original(sim, **kwargs)
            finally:
                sim._boundary_spec = saved

        Simulation._build_nonuniform_grid = drop
        judge = "test_declared_walls_match_uniform_guide"
    else:
        raise ValueError(mode)
    return pytest.main(["tests/contracts/test_graded_guide_default.py", "-q", "-k", judge,
                        "--tb=short", "-o", "cache_dir=.pr3/pytest-cache"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
