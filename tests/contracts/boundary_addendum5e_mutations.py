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
    elif mode == "oblique-tilt-not-reported":
        original = tfsf.report

        def omit(sim, issues, **kwargs):
            from rfx.preflight._common import PreflightIssue
            original(sim, issues, **kwargs)
            if sim._tfsf is not None and sim._tfsf.angle_deg != 0:
                tilt = "y" if sim._tfsf.polarization == "ez" else "z"
                for index, issue in enumerate(issues):
                    if issue.code == "tfsf_transverse_periodic":
                        issues[index] = PreflightIssue(
                            str(issue).replace(f"{tilt}_lo, {tilt}_hi", ""),
                            severity=issue.severity, code=issue.code)

        tfsf.report = omit
        judge = "test_oblique_tilt_finite_box_reports_unjudged"
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
