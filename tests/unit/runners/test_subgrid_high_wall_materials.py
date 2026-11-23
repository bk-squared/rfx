"""#1966: capture the production fine material handoff before any step."""
import pytest

from rfx import Box, Simulation
from rfx.core.yee import component_e_materials
from rfx.model import source_coefficients


@pytest.mark.parametrize("end,expected", [(3, 2.5), (2, 1.75)],
                         ids=["global-high-wall", "interior-fine-end"])
def test_production_fine_terminal_material(monkeypatch, end, expected):
    """The global-wall mean is (4+4+1+1)/4 = 2.5, versus the interior-end mean (1+4+1+1)/4 = 1.75 with its real vacuum sample."""
    h = .001
    sim = Simulation(freq_max=8e9, domain=(2*h, 2*h, 3*h), dx=h,
                     boundary="pec", cpml_layers=0)
    sim.add_refinement((h, end*h), ratio=2, validation="off")
    sim.add_material("body", eps_r=4.)
    sim.add(Box((0, .5*h, h), (2*h, h, end*h)), material="body")
    captured = []

    class Prepared(Exception):
        pass

    def capture(requests, grid, materials, n_steps, **kwargs):
        captured.append((grid, materials, kwargs["high_wall"]))
        raise Prepared

    monkeypatch.setattr(source_coefficients, "subgrid_source_tables", capture)
    with pytest.raises(Prepared):
        sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    assert len(captured) == 1
    grid, materials, high_wall = captured[0]
    assert high_wall == ((True, True, True) if end == 3 else (True, True, False))
    assert grid.shape == (5, 5, 2*end-1)
    assert grid.dx == h/2
    index = (1, 1, -1)
    assert float(materials.eps_r[1, 1, -2]) == 4.
    assert float(component_e_materials(materials, high_wall=high_wall)[0][0][index]) == expected
    assert float(materials.eps_r[index]) == 1.0
