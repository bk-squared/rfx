"""Wire-port diagnostics sample the selected grid, including graded axes."""
import numpy as np
import pytest

from rfx import Simulation


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_graded_wire_port_centers_and_unavailable_classification(axis):
    profiles = {f"d{'xyz'[axis]}_profile": np.array([0.001, 0.002, 0.001, 0.001, 0.001])}
    sim = Simulation(freq_max=10e9, domain=(0.006, 0.006, 0.006),
                     dx=0.001, cpml_layers=0, boundary="pec", **profiles)
    position = [0.002, 0.002, 0.002]
    position[axis] = 0.001
    sim.add_port(tuple(position), component=f"e{'xyz'[axis]}",
                 impedance=50, extent=0.003)

    centers, midpoint = sim._wire_port_cell_centers(sim._ports[0])
    expected = np.tile(position, (2, 1)).astype(float)
    expected[:, axis] = [0.002, 0.0035]
    np.testing.assert_allclose(centers, expected, rtol=0, atol=1e-9)
    assert midpoint == 1

    # NU rasterization is known; the separate dead-cell classifier still
    # discloses its unsupported lane rather than silently omitting the note.
    findings = sim.preflight()
    hits = [finding for finding in findings
            if finding.code == "wire_port_dead_cell_classification_unavailable"]
    assert len(hits) == 1
    assert hits[0].severity == "warning"
    assert "non-uniform mesh" in hits[0]
