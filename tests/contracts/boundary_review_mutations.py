"""Kept Addendum 5c mutations; each selected judge must fail."""
import sys

import numpy as np
import pytest

from rfx.boundaries import serialization, tfsf


def main():
    mode = sys.argv[1]
    path = "tests/contracts/test_boundary_review_regressions.py"
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
    elif mode == "base-assembly-rejudged":
        original = tfsf.admit_simulation

        def ignore(sim, **kwargs):
            kwargs.pop("material_overrides", None)
            kwargs.pop("materials", None)
            return original(sim, **kwargs)

        tfsf.admit_simulation = ignore
        judge = "test_concrete_override_replaces_finite_base_for_admission"
    elif mode == "legacy-document-becomes-explicit":
        serialization.restore_legacy_boundary = lambda sim: sim
        judge = "test_main_document_without_provenance"
    elif mode == "graded-guide-refused":
        from rfx.boundaries import features
        original = features.admit_waveguide

        def refuse(sim, **kwargs):
            if sim._uses_nonuniform_mesh and sim._waveguide_ports:
                raise ValueError("mutant refuses the graded guide")
            return original(sim, **kwargs)

        features.admit_waveguide = refuse
        judge = "test_nonuniform_explicit_absorbers_keep_real_pads_without_default_warning"
    elif mode == "propagation-wall-refused":
        original = tfsf.replacement_axes

        def refuse(cfg, grid, **kwargs):
            from rfx.boundaries.depths import Kind, grid_face_depths
            records = kwargs.get("records") or grid_face_depths(grid)
            if any(f.name.startswith("x_") and f.kind != Kind.ABSORBER for f in records):
                raise ValueError("mutant refuses the propagation-axis wall")
            return original(cfg, grid, **kwargs)

        tfsf.replacement_axes = refuse
        judge = "test_propagation_axis_wall_keeps_main_admission"
    elif mode == "export-adds-format-key":
        from rfx.interop import _design
        original = _design._dump_boundary
        _design._dump_boundary = lambda sim: dict(original(sim), explicit=False)
        judge = "test_default_export_does_not_extend_the_frozen_schema"
    elif mode == "caller-wrap-rejudged":
        original = tfsf.admit_setup

        def rejudge(**kwargs):
            kwargs["feature_owned"] = True
            return original(**kwargs)

        tfsf.admit_setup = rejudge
        path = "tests/contracts/test_tfsf_boundary_admission.py"
        judge = "test_low_level_explicit_wrap_keeps_caller_declaration"
    elif mode == "realized-kind-ignored":
        from dataclasses import replace
        from rfx.boundaries import model
        original = model.realize

        def declared(boundary, grid):
            result = original(boundary, grid)
            return replace(result, faces=tuple(replace(f, kind=boundary.face(f.name).kind)
                                                for f in result.faces))

        model.realize = declared
        judge = "test_realized_kind_agrees_with_default_grid_record"
    elif mode == "unrelated-kinds-relabeled":
        from dataclasses import replace
        from rfx.boundaries import model
        original = model.realize

        def relabel(boundary, grid):
            result = original(boundary, grid)
            kinds = {f.name: f.kind for f in grid.boundary_depths}
            return replace(result, faces=tuple(replace(f, kind=kinds[f.name])
                                                for f in result.faces))

        model.realize = relabel
        judge = "test_guide_relabel_does_not_relabel_the_floquet_declaration"
    else:
        raise ValueError(mode)
    return pytest.main([
        path, "-q", "-k", judge,
        "--basetemp=.pr3/mutation-" + mode, "-o", "cache_dir=.pr3/pytest-cache",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
