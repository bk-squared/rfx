"""Addendum 5e defects; production helpers remain called in every mode."""
import sys

import pytest


def main(mode):
    from rfx.boundaries import features, tfsf
    path = "tests/contracts/test_tfsf_boundary_admission.py"
    if mode == "graded-uniform-rule":
        original = tfsf.admit_setup

        def restored(**kwargs):
            if kwargs["nonuniform"]:
                kwargs.update(nonuniform=False, periodic=(False, True, True))
            return original(**kwargs)

        tfsf.admit_setup = restored
        judge = "test_graded_plane_wave_keeps_absorbers"
    elif mode == "oblique-no-axes":
        original = tfsf.replacement_axes

        def empty(cfg, *args, **kwargs):
            axes = original(cfg, *args, **kwargs)
            return () if cfg.angle_deg != 0 else axes

        tfsf.replacement_axes = empty
        judge = "test_oblique_non_tilt_finite_box_refuses"
    elif mode == "oblique-tilt-unjudged":
        original = tfsf.replacement_axes

        def unjudged(cfg, grid, **kwargs):
            from dataclasses import replace
            if getattr(cfg, "angle_deg", 0) != 0:
                electric = getattr(cfg, "electric_component", getattr(cfg, "polarization", "ez"))[-1]
                tilt = "y" if electric == "z" else "z"
                records = kwargs.get("records")
                records = tfsf.grid_face_depths(grid) if records is None else records
                # Bypass both the requirement loop and the array judge on tilt.
                kwargs["records"] = tuple(replace(face, kind=tfsf.Kind.PERIODIC)
                                          if face.name.startswith(tilt + "_") else face
                                          for face in records)
            return original(cfg, grid, **kwargs)

        tfsf.replacement_axes = unjudged
        judge = "test_oblique_tilt_finite_box_refuses"
    elif mode in ("explicit-full-admitted", "partial-refused"):
        original = features.full_aperture

        def wrong(sim, entry, grid):
            full = original(sim, entry, grid)
            explicit_range = entry.y_range is not None
            if sim._uses_nonuniform_mesh and explicit_range:
                return mode == "partial-refused"
            return full

        features.full_aperture = wrong
        path = "tests/contracts/test_graded_guide_default.py"
        judge = "test_explicit_aperture_ranges_on_graded_mesh"
    else:
        raise ValueError(mode)
    return pytest.main([path, "-q", "-k", judge, "--tb=short",
                        "-o", "cache_dir=.pr3/pytest-cache"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
