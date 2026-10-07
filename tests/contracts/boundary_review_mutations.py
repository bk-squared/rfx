"""Kept Addendum 5c mutations; each selected judge must fail."""
import sys

import numpy as np
import pytest

from rfx.boundaries import serialization, tfsf


def main():
    mode = sys.argv[1]
    if mode == "walls-ignored":
        original = tfsf.replacement_axes

        def ignore(cfg, grid, **kwargs):
            from dataclasses import replace
            from rfx.boundaries.depths import Kind, grid_face_depths
            records = kwargs.pop("records", None) or grid_face_depths(grid)
            records = tuple(replace(f, kind=Kind.PERIODIC)
                            if f.name[0] in "yz" and f.kind in (Kind.PEC, Kind.PMC)
                            else f for f in records)
            return original(cfg, grid, records=records, **kwargs)

        tfsf.replacement_axes = ignore
        judge = "test_declared_walls_refuse_a_finite_box"
    elif mode == "localized-ignored":
        original = tfsf.check_arrays

        def ignore(*args, **kwargs):
            kwargs["localized"] = ()
            return original(*args, **kwargs)

        tfsf.check_arrays = ignore
        judge = "test_localized_updates_are_not_invariant"
    elif mode == "wall-operator-changed":
        tfsf.boundary_flags = lambda *args, **kwargs: ((False, False, False), "xyz")
        judge = "test_declared_invariant_walls_keep_the_reported_legacy_wrap"
    elif mode == "rebuild-loses-default":
        serialization.constructor_boundary = lambda sim: sim._boundary_spec
        judge = "test_default_provenance_survives"
    elif mode == "upml-guard-lifted":
        tfsf.validate_source_boundary = lambda *args, **kwargs: None
        judge = "test_upml_remains_refused"
    elif mode in ("seam-exemption-removed", "seam-exemption-widened"):
        def mask(name, host, axis, index):
            if mode.endswith("widened") and "pec_edge_masks" in name:
                return np.zeros_like(host)
            return host

        tfsf._without_duplicated_end_edge = mask
        judge = "test_only_the_duplicated_end_edge_is_exempt"
    elif mode == "traced-finding-lost":
        original = tfsf.invariant

        def assume(*args):
            result = original(*args)
            return (True, "") if result[0] is None else result

        tfsf.invariant = assume
        judge = "test_traced_override_is_admitted_with_unjudged_finding"
    elif mode == "legacy-document-becomes-explicit":
        serialization.restore_legacy_boundary = lambda sim: sim
        judge = "test_main_document_without_provenance"
    else:
        raise ValueError(mode)
    return pytest.main([
        "tests/contracts/test_boundary_review_regressions.py", "-q", "-k", judge,
        "--basetemp=.pr3/mutation-" + mode, "-o", "cache_dir=.pr3/pytest-cache",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
