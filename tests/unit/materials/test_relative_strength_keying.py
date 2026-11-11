"""T3: value keying conserves pole entries and realized susceptibility."""
import numpy as np
import pytest

from rfx import Box, LorentzPole, RelativeStrength, Simulation
from rfx.core.yee import EPS_0


@pytest.mark.parametrize("second", [1., .5], ids=["equal", "different_value"])
def test_relative_strength_keying(second):
    w0 = 2e10
    sim = Simulation(freq_max=8e9, domain=(.020, .010, .010), dx=.001, boundary="pec")
    for name, value, bounds in (("a", 1., ((0., 0., 0.), (.012, .010, .010))),
                                ("b", second, ((.008, 0., 0.), (.020, .010, .010)))):
        sim.add_material(name, eps_r=1., lorentz_poles=[LorentzPole(w0, 1e9, RelativeStrength(2 * w0**2, value))])
        sim.add(Box(*bounds), material=name)
    grid = sim._build_grid()
    materials, _, lorentz = sim._build_materials(grid)
    coefficients, _ = lorentz
    entries = coefficients.damping.shape[0]
    delta_eps = np.asarray(coefficients.c[0], dtype=np.float64).sum(axis=0) * (1 + 1e9 * grid.dt) / (EPS_0 * grid.dt**2 * w0**2)
    permittivity = np.asarray(materials.eps_r) + delta_eps
    cells = (permittivity[4, 4, 4], permittivity[10, 4, 4], permittivity[16, 4, 4])
    print(f"T3 second={second}: entries={entries} epsilon(a,overlap,b)={cells}")
    assert entries == (1 if second == 1 else 2)
    np.testing.assert_allclose(cells, [3., 1 + 2 * second, 1 + 2 * second], rtol=1e-6)
