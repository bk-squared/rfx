"""A geometry entry contributes to every absorber axis it crosses."""

import pytest

from rfx import Box, Simulation


@pytest.mark.parametrize(
    "lo,hi,expected_axes",
    [
        ((-0.001, -0.002, 0.001), (0.031, 0.032, 0.002),
         ["x-axis", "y-axis"]),
        ((0.010, -0.002, 0.001), (0.012, 0.032, 0.002),
         ["y-axis"]),
    ],
    ids=["crosses-x-and-y", "crosses-only-y"],
)
def test_geometry_absorber_finding_names_every_crossed_axis(lo, hi, expected_axes):
    sim = Simulation(freq_max=10e9, domain=(0.030, 0.030, 0.004),
                     dx=0.0005, cpml_layers=8, boundary="cpml")
    sim.add_material("substrate", eps_r=3.55)
    sim.add(Box(lo, hi), material="substrate")
    sim.add_source((0.015, 0.015, 0.002), "ez")

    findings = [issue for issue in sim.preflight(strict=False)
                if issue.code == "geometry_in_absorber"]

    assert [str(issue).split(" region along ", 1)[1].split(":", 1)[0]
            for issue in findings] == expected_axes
