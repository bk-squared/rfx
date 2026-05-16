"""Production-envelope validation for the subgridding API."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


def _vacuum_subgrid_sim(*, validation: str = "production") -> Simulation:
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=0.004)
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2, validation=validation)
    sim.add_source((0.02, 0.02, 0.020), "ez")
    sim.add_probe((0.024, 0.024, 0.020), "ez")
    return sim


def _guarded_boundary_vacuum_subgrid_sim() -> Simulation:
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=0.004)
    sim.add_refinement(z_range=(0.004, 0.040), ratio=2, validation="production")
    zlo, zhi = (0.004, 0.040)
    span = zhi - zlo
    sim.add_source((0.02, 0.02, zlo + 0.45 * span), "ez")
    sim.add_probe((0.024, 0.024, zlo + 0.55 * span), "ez")
    return sim


def test_production_subgrid_validation_rejects_centered_vacuum_source_probe_case():
    sim = _vacuum_subgrid_sim(validation="production")

    report = sim.validate_subgrid()

    assert not report.supported
    assert any(issue.code == "z_slab_requires_guarded_boundary" for issue in report.errors)

    with pytest.raises(ValueError, match="z_slab_requires_guarded_boundary"):
        report.raise_if_unsupported()


def test_disjoint_topology_contract_is_research_only_and_distinct_from_overlap_guard():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=0.004)
    sim.add_refinement(
        z_range=(0.012, 0.028),
        ratio=2,
        validation="production",
        topology="stage2_disjoint_3d",
    )
    sim.add_source((0.02, 0.02, 0.020), "ez")
    sim.add_probe((0.024, 0.024, 0.020), "ez")

    report = sim.validate_subgrid()

    assert not report.supported
    assert report.support_level == (
        "unsupported-production-stage2-disjoint-3d-integration-pending"
    )
    codes = [issue.code for issue in report.errors]
    assert "disjoint_topology_public_runner_unintegrated" in codes
    assert "z_slab_requires_guarded_boundary" not in codes


def test_disjoint_topology_research_contract_runs_finite_smoke_trace():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=0.004)
    sim.add_refinement(
        z_range=(0.012, 0.028),
        ratio=2,
        validation="research",
        topology="stage2_disjoint_3d",
    )
    sim.add_source((0.02, 0.02, 0.020), "ez")
    sim.add_probe((0.024, 0.024, 0.020), "ez")

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert report.support_level == "research-stage2-disjoint-3d-public-contract"
    assert report.issues[0].code == "stage2_disjoint_public_contract"
    result = sim.run(n_steps=4, compute_s_params=False)
    assert result.time_series.shape == (4, 1)
    assert np.all(np.isfinite(np.asarray(result.time_series)))
    assert result.dt is not None


def test_production_subgrid_validation_accepts_guarded_boundary_vacuum_source_probe_case():
    sim = _guarded_boundary_vacuum_subgrid_sim()

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert report.errors == ()
    assert report.support_level == "production-z-slab-guarded-boundary-vacuum-envelope"
    assert report.raise_if_unsupported() is report

    artifact = report.to_dict()
    assert artifact["supported"] is True
    assert artifact["support_level"] == "production-z-slab-guarded-boundary-vacuum-envelope"
    assert artifact["region"]["ratio"] == 2
    assert artifact["issues"][0]["code"] == "support_envelope"
    assert json.loads(report.to_json()) == artifact


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("use_material_sat", False),
        ("material_sat_scale", 0.75),
        ("material_sat_coarse_scale", 0.75),
        ("material_sat_fine_scale", 0.75),
        ("material_sat_e_coarse_scale", 0.75),
        ("material_sat_e_fine_scale", 0.75),
        ("material_sat_h_coarse_scale", 0.75),
        ("material_sat_h_fine_scale", 0.75),
        ("material_sat_zlo_scale", 0.75),
        ("material_sat_zhi_scale", 0.75),
        ("material_sat_e_zlo_scale", 0.75),
        ("material_sat_e_zhi_scale", 0.75),
        ("material_sat_h_zlo_scale", 0.75),
        ("material_sat_h_zhi_scale", 0.75),
        ("material_sat_pair_a_zlo_scale", 0.75),
        ("material_sat_pair_b_zlo_scale", 0.75),
        ("material_sat_zlo_common_trace_projection", "coarse"),
        ("material_sat_zhi_common_trace_projection", "coarse"),
        ("material_sat_normal_e_scale", 0.75),
        ("material_sat_zhi_coarse_eps_blend", 0.75),
        ("defer_material_h_sat_until_after_e", True),
        ("coarse_shadow_source_scale", 0.75),
        ("fine_source_scale", 0.75),
        ("coarse_shadow_source_projection", "fine_node_nearest"),
        ("material_sat_face_projection", "sample"),
        ("sync_coarse_interface_from_fine", True),
        ("sync_coarse_shadow_from_fine", True),
        ("sync_box_coarse_shadow_from_fine", True),
        ("mask_coarse_shadow_interior", True),
        ("use_exterior_z_interfaces", True),
        ("use_boundary_terminated_exterior_z_interfaces", True),
        ("use_exterior_box_interfaces", True),
        ("ghost_exterior_coarse_shadow_from_fine", True),
        ("inject_sources_before_e_coupling", True),
        ("inject_sources_on_coarse_shadow", True),
        ("diagnostic_lumped_sparam_freqs", (3.0e9,)),
    ),
)
def test_production_subgrid_validation_rejects_diagnostic_overrides(key, value):
    sim = _vacuum_subgrid_sim(validation="production")
    sim._refinement[key] = value

    report = sim.validate_subgrid()

    assert not report.supported
    assert any(
        issue.code == "diagnostic_subgrid_override_unvalidated"
        for issue in report.errors
    )
    assert key in report.format()

    with pytest.raises(ValueError, match="diagnostic_subgrid_override_unvalidated"):
        report.raise_if_unsupported()


def test_production_subgrid_run_rejects_diagnostic_overrides_before_execution():
    sim = _vacuum_subgrid_sim(validation="production")
    sim._refinement["sync_coarse_shadow_from_fine"] = True

    with pytest.raises(ValueError, match="diagnostic_subgrid_override_unvalidated"):
        sim.run(n_steps=1, compute_s_params=False)


def test_subgrid_auto_timesteps_preserve_requested_physical_duration():
    sim = Simulation(freq_max=4e9, domain=(0.012, 0.012, 0.012), boundary="pec", dx=0.004)
    sim.add_refinement(z_range=(0.004, 0.012), ratio=3, validation="production")
    sim.add_source((0.004, 0.004, 0.008), "ez")
    sim.add_probe((0.008, 0.008, 0.008), "ez")

    coarse_steps = sim._build_grid().num_timesteps(num_periods=1.0)
    result = sim.run(num_periods=1.0, compute_s_params=False)

    assert len(result.time_series) == coarse_steps * 3


def test_subgrid_explicit_timesteps_remain_low_level_escape_hatch():
    sim = Simulation(freq_max=4e9, domain=(0.012, 0.012, 0.012), boundary="pec", dx=0.004)
    sim.add_refinement(z_range=(0.004, 0.012), ratio=3, validation="production")
    sim.add_source((0.004, 0.004, 0.008), "ez")
    sim.add_probe((0.008, 0.008, 0.008), "ez")

    result = sim.run(n_steps=7, compute_s_params=False)

    assert len(result.time_series) == 7


def test_research_subgrid_validation_allows_diagnostic_overrides():
    sim = _vacuum_subgrid_sim(validation="research")
    sim._refinement["sync_coarse_interface_from_fine"] = True

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert not any(
        issue.code == "diagnostic_subgrid_override_unvalidated"
        for issue in report.errors
    )


def test_production_subgrid_validation_rejects_xy_window_observables_near_xy_interface():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add_refinement(
        z_range=(0.002, 0.024),
        ratio=2,
        xy_margin=0.004,
        validation="production",
    )
    sim.add_source((0.04 / 3.0, 0.04 / 3.0, 0.012), "ez")
    sim.add_probe((2.0 * 0.04 / 3.0, 2.0 * 0.04 / 3.0, 0.014), "ez")

    report = sim.validate_subgrid()

    assert not report.supported
    assert any(
        issue.code == "xy_windowed_margin_too_close"
        for issue in report.errors
    )


def test_production_subgrid_validation_accepts_central_source_xy_window_envelope():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    zlo, zhi = (0.002, 0.024)
    span = zhi - zlo
    sim.add_refinement(
        z_range=(zlo, zhi),
        ratio=2,
        xy_margin=0.008,
        validation="production",
    )
    sim.add_source((0.020, 0.020, zlo + 0.45 * span), "ez")
    sim.add_probe((0.020, 0.020, zlo + 0.55 * span), "ez")

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert report.errors == ()
    assert report.support_level == "production-local-xy-window-central-source-envelope"


def test_production_subgrid_validation_rejects_offcenter_source_xy_window():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    zlo, zhi = (0.002, 0.024)
    span = zhi - zlo
    sim.add_refinement(
        z_range=(zlo, zhi),
        ratio=2,
        xy_margin=0.008,
        validation="production",
    )
    sim.add_source((0.0188, 0.020, zlo + 0.45 * span), "ez")
    sim.add_probe((0.020, 0.020, zlo + 0.55 * span), "ez")

    report = sim.validate_subgrid()

    assert not report.supported
    assert any(
        issue.code == "xy_windowed_external_crossval_blocked"
        for issue in report.errors
    )
    assert report.support_level == "unsupported-production-local-xy-window-external-crossval-blocked"


def test_research_central_xy_window_vacuum_margin_runs_smoke_after_production_block():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    zlo, zhi = (0.002, 0.024)
    span = zhi - zlo
    sim.add_refinement(
        z_range=(zlo, zhi),
        ratio=2,
        xy_margin=0.008,
        validation="research",
    )
    sim.add_source((0.020, 0.020, zlo + 0.45 * span), "ez")
    sim.add_probe((0.020, 0.020, zlo + 0.55 * span), "ez")

    report = sim.validate_subgrid(mode="research")
    assert report.supported, report.format()
    result = sim.run(n_steps=3, compute_s_params=False)

    assert result.time_series.shape[0] == 3


def test_research_xy_window_builds_local_region_and_runs_smoke():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.004)
    sim.add_refinement(
        z_range=(0.004, 0.020),
        ratio=2,
        xy_margin=0.008,
        validation="research",
    )
    sim.add_source((0.020, 0.020, 0.012), "ez")
    sim.add_probe((0.024, 0.020, 0.012), "ez")

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert report.region is not None
    assert report.region.fi_lo > 0
    assert report.region.fi_hi < sim._build_grid().nx
    result = sim.run(n_steps=3, compute_s_params=False)
    assert result.time_series.shape[0] == 3


def test_production_subgrid_validation_rejects_centered_dielectric_outside_guarded_envelope():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=0.004)
    sim.add_material("dielectric", eps_r=2.25)
    sim.add(Box((0.0, 0.0, 0.0), (0.04, 0.04, 0.04)), material="dielectric")
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2, validation="production")
    sim.add_source((0.02, 0.02, 0.020), "ez")
    sim.add_probe((0.024, 0.024, 0.020), "ez")

    report = sim.validate_subgrid()

    assert not report.supported
    assert any(issue.code == "material_weighted_sat_missing" for issue in report.errors)

    with pytest.raises(ValueError, match="material_weighted_sat_missing"):
        report.raise_if_unsupported()

    artifact = json.loads(report.to_json())
    assert artifact["supported"] is False
    assert artifact["issues"][0]["severity"] == "error"


def test_production_boundary_terminated_accepts_static_material_guarded_envelope():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add_material("dielectric", eps_r=2.25)
    sim.add(Box((-0.001, -0.001, -0.001), (0.041, 0.041, 0.025)), material="dielectric")
    sim.add_refinement(z_range=(0.002, 0.024), ratio=2, validation="production")
    zlo, zhi = (0.002, 0.024)
    span = zhi - zlo
    sim.add_source((0.04 / 3.0, 0.04 / 3.0, zlo + 0.45 * span), "ez")
    sim.add_probe((0.02, 0.02, zlo + 0.35 * span), "ez")

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert report.errors == ()
    assert report.support_level == "production-z-slab-guarded-boundary-static-material-envelope"


def test_production_boundary_terminated_accepts_contained_static_material_layer():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add_material("dielectric", eps_r=2.25)
    sim.add(Box((-0.001, -0.001, 0.012), (0.041, 0.041, 0.018)), material="dielectric")
    sim.add_refinement(z_range=(0.002, 0.024), ratio=2, validation="production")
    zlo, zhi = (0.002, 0.024)
    span = zhi - zlo
    sim.add_source((0.04 / 3.0, 0.04 / 3.0, zlo + 0.45 * span), "ez")
    sim.add_probe((0.02, 0.02, zlo + 0.35 * span), "ez")

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert not any(
        issue.code == "material_transition_near_artificial_interface"
        for issue in report.errors
    )


def test_production_boundary_terminated_rejects_static_material_jump_at_interface():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add_material("dielectric", eps_r=2.25)
    sim.add(
        Box((-0.001, -0.001, -0.001), (0.041, 0.041, 0.017)),
        material="dielectric",
    )
    zlo, zhi = (0.0, 0.016)
    span = zhi - zlo
    sim.add_refinement(z_range=(zlo, zhi), ratio=2, validation="production")
    sim.add_source((0.013, 0.013, zlo + 0.45 * span), "ez")
    sim.add_probe((0.027, 0.027, zlo + 0.55 * span), "ez")

    report = sim.validate_subgrid()

    assert not report.supported
    assert any(issue.code == "material_jump_at_zhi_interface" for issue in report.errors)

    with pytest.raises(ValueError, match="material_jump_at_zhi_interface"):
        sim.run(n_steps=3, compute_s_params=False)


def test_production_boundary_terminated_rejects_static_material_transition_near_interface():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add_material("dielectric", eps_r=2.25)
    sim.add(
        Box((-0.001, -0.001, -0.001), (0.041, 0.041, 0.023)),
        material="dielectric",
    )
    zlo, zhi = (0.0, 0.016)
    span = zhi - zlo
    sim.add_refinement(z_range=(zlo, zhi), ratio=2, validation="production")
    sim.add_source((0.013, 0.013, zlo + 0.45 * span), "ez")
    sim.add_probe((0.027, 0.027, zlo + 0.55 * span), "ez")

    report = sim.validate_subgrid()

    assert not report.supported
    codes = [issue.code for issue in report.errors]
    assert "material_jump_at_zhi_interface" not in codes
    assert "material_transition_near_artificial_interface" in codes

    with pytest.raises(ValueError, match="material_transition_near_artificial_interface"):
        sim.run(n_steps=3, compute_s_params=False)


def test_production_boundary_terminated_accepts_guarded_pec_at_interface():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add(Box((0.010, 0.010, 0.015), (0.030, 0.030, 0.017)), material="pec")
    zlo, zhi = (0.0, 0.016)
    span = zhi - zlo
    sim.add_refinement(z_range=(zlo, zhi), ratio=2, validation="production")
    sim.add_source((0.006, 0.006, zlo + 0.45 * span), "ez")
    sim.add_probe((0.008, 0.006, zlo + 0.55 * span), "ez")

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert not any(issue.code.startswith("pec_at_") for issue in report.errors)
    result = sim.run(n_steps=20, compute_s_params=False)
    assert np.all(np.isfinite(result.time_series))


def test_production_subgrid_validation_rejects_material_jump_at_interface():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=0.004)
    sim.add_material("dielectric", eps_r=4.0)
    sim.add(Box((0.0, 0.0, 0.0), (0.04, 0.04, 0.016)), material="dielectric")
    sim.add_refinement(z_range=(0.016, 0.032), ratio=2, validation="production")
    sim.add_source((0.02, 0.02, 0.024), "ez")
    sim.add_probe((0.024, 0.024, 0.024), "ez")

    report = sim.validate_subgrid()

    assert not report.supported
    assert any(issue.code == "material_jump_at_zlo_interface" for issue in report.errors)
    assert any(issue.code == "material_weighted_sat_missing" for issue in report.errors)


def test_production_boundary_terminated_requires_guarded_margin():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add_refinement(z_range=(0.002, 0.024), ratio=2, validation="production")
    zlo, zhi = (0.002, 0.024)
    span = zhi - zlo
    sim.add_source((0.04 / 3.0, 0.04 / 3.0, zlo + 0.45 * span), "ez")
    sim.add_probe((0.02, 0.02, zlo + 0.25 * span), "ez")

    report = sim.validate_subgrid()

    assert not report.supported
    assert any(
        issue.code == "boundary_terminated_margin_too_close"
        for issue in report.errors
    )


def test_production_boundary_terminated_accepts_guarded_vacuum_margin():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add_refinement(z_range=(0.002, 0.024), ratio=2, validation="production")
    zlo, zhi = (0.002, 0.024)
    span = zhi - zlo
    sim.add_source((0.04 / 3.0, 0.04 / 3.0, zlo + 0.45 * span), "ez")
    sim.add_probe((0.02, 0.02, zlo + 0.35 * span), "ez")

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert report.errors == ()


def test_production_boundary_terminated_accepts_opposite_z_cpml_boundary():
    sim = Simulation(
        freq_max=8e9,
        domain=(0.04, 0.04, 0.024),
        boundary=BoundarySpec(x="pec", y="pec", z=Boundary(lo="pec", hi="cpml")),
        dx=0.002,
        cpml_layers=2,
    )
    zlo, zhi = (0.0, 0.016)
    span = zhi - zlo
    sim.add_refinement(z_range=(zlo, zhi), ratio=2, validation="production")
    sim.add_source((0.020, 0.020, zlo + 0.45 * span), "ez")
    sim.add_probe((0.022, 0.020, zlo + 0.55 * span), "ez")

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert report.errors == ()
    result = sim.run(n_steps=3, compute_s_params=False)
    assert result.time_series.shape[0] == 3


def test_production_boundary_terminated_rejects_refined_face_touching_cpml():
    sim = Simulation(
        freq_max=8e9,
        domain=(0.04, 0.04, 0.024),
        boundary=BoundarySpec(x="pec", y="pec", z=Boundary(lo="cpml", hi="pec")),
        dx=0.002,
        cpml_layers=2,
    )
    zlo, zhi = (0.0, 0.016)
    span = zhi - zlo
    with pytest.warns(UserWarning, match="overlaps PML"):
        sim.add_refinement(z_range=(zlo, zhi), ratio=2, validation="production")
    sim.add_source((0.020, 0.020, zlo + 0.45 * span), "ez")
    sim.add_probe((0.022, 0.020, zlo + 0.55 * span), "ez")

    report = sim.validate_subgrid()

    assert not report.supported
    assert any(issue.code == "subgrid_overlaps_absorber" for issue in report.errors)
    assert any(
        issue.code == "boundary_terminated_requires_pec_no_cpml"
        for issue in report.errors
    )


def test_production_boundary_terminated_accepts_guarded_ntff_box():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add_refinement(z_range=(0.002, 0.024), ratio=2, validation="production")
    zlo, zhi = (0.002, 0.024)
    span = zhi - zlo
    sim.add_source((0.020, 0.020, zlo + 0.55 * span), "ez")
    sim.add_probe((0.022, 0.020, zlo + 0.55 * span), "ez")
    sim.add_ntff_box(
        (0.014, 0.014, zlo + 0.40 * span),
        (0.026, 0.026, zlo + 0.75 * span),
        freqs=[4.0e9],
    )

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert report.errors == ()


def test_production_opposite_z_cpml_accepts_guarded_ntff_box():
    boundary = BoundarySpec(
        x="pec",
        y="pec",
        z=Boundary(lo="pec", hi="cpml"),
    )
    sim = Simulation(
        freq_max=8e9,
        domain=(0.04, 0.04, 0.024),
        boundary=boundary,
        dx=0.002,
        cpml_layers=2,
    )
    sim.add_refinement(z_range=(0.0, 0.016), ratio=2, validation="production")
    zlo, zhi = (0.0, 0.016)
    span = zhi - zlo
    sim.add_source((0.020, 0.020, zlo + 0.45 * span), "ez")
    sim.add_probe((0.022, 0.020, zlo + 0.55 * span), "ez")
    sim.add_ntff_box(
        (0.014, 0.014, zlo + 0.35 * span),
        (0.026, 0.026, zlo + 0.70 * span),
        freqs=[4.0e9],
    )

    report = sim.validate_subgrid()

    assert report.supported, report.format()
    assert report.errors == ()


def test_production_boundary_terminated_rejects_ntff_box_near_artificial_interface():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add_refinement(z_range=(0.002, 0.024), ratio=2, validation="production")
    zlo, zhi = (0.002, 0.024)
    span = zhi - zlo
    sim.add_source((0.020, 0.020, zlo + 0.55 * span), "ez")
    sim.add_probe((0.022, 0.020, zlo + 0.55 * span), "ez")
    sim.add_ntff_box(
        (0.014, 0.014, zlo + 0.05 * span),
        (0.026, 0.026, zlo + 0.40 * span),
        freqs=[4.0e9],
    )

    report = sim.validate_subgrid()

    assert not report.supported
    assert any(issue.code == "ntff_box_margin_too_close" for issue in report.errors)
