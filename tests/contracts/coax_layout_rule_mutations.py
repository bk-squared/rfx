"""Coaxial port layout, band limits and loading with altered behavior."""
import os
import subprocess
import sys

import pytest


CASES = {
    'm1': ('defect_restored', 'tests/unit/model/test_coax_layout_rule.py::test_plane_wiring', "rewrite(coax_line, 'coax_port_layout', '(rule.start + rule.spacing * k)', '(8 + 4 * k)')"),
    'm2': ('refusal_removed', 'tests/unit/model/test_coax_layout_rule.py::test_length_remedies', "rewrite(coax_line, 'coax_port_layout', 'if distance < rule.min_cells:', 'if False:')"),
    'm2b': ('defect_restored', 'tests/unit/model/test_coax_layout_rule.py::test_length_edge_sma', "rewrite(coax_line, 'coax_port_layout', 'if distance < rule.min_cells:', 'if distance < 60 + setback:')"),
    'm3': ('defect_restored', 'tests/unit/model/test_coax_layout_rule.py::test_source_clearance', "rewrite(coax_line, 'coax_port_layout', 'source -= direction * setback', 'source -= direction * (setback + 1)')"),
    'm4': ('refusal_removed', 'tests/unit/model/test_coax_layout_rule.py::test_refuses_nonuniform_grid', "rewrite(coax_line, 'coax_port_layout', 'if not isinstance(grid, Grid):', 'if False:')"),
    'm5': ('defect_restored', 'tests/unit/model/test_coax_layout_rule.py::test_length_remedies', "rewrite(coax_line, 'coax_port_layout', 'freq_min=port.freq_min)', 'freq_min=freq_max / 10)')"),
    'm6': ('defect_restored', 'tests/unit/model/test_coax_plane_rule.py::test_phase_remedy_kappa_only', 'rewrite(coax_planes, \'_coax_plane_rule\', \'+ " and ".join(remedies)\', \'+ "a smaller kappa or a finer dz"\')'),
    'm8': ('defect_restored', 'tests/unit/model/test_coax_layout_rule.py::test_ntff_end', 'rewrite(coax_line, \'coax_port_layout\', "end = \'NTFF box\'", "end = \'z absorber\'")'),
    'm9': ('defect_restored', 'tests/contracts/test_coax_shielded_end.py::test_gap', "rewrite(coax_shield, 'shield_coax_end', 'fill_gap_cells(outer_radius, grid.dx)', 'fill_gap_cells(outer_radius, grid.dx) + 1')"),
    "m7a": ("defect_restored", "tests/unit/ports/test_coax_freq_min.py::test_default_freq_min", "coaxial_port.FREQ_MIN_DIVISOR = 9"),
    "m7b": ("refusal_removed", "tests/unit/ports/test_coax_freq_min.py::test_refuses_freq_min_outside_band", "rewrite(coaxial_port, 'coax_freq_min', 'if not 0 < value < freq_max:', 'if False:')"),
    "m10": ("defect_restored", "tests/interop/test_interop_design_document.py::test_coaxial_freq_min_survives_the_design_round_trip[explicit]", "rewrite(design, 'simulation_from_design', '            freq_min=values[\"freq_min\"],\\n', ''); interop.simulation_from_design = design.simulation_from_design"),
}


@pytest.mark.parametrize("case", CASES)
def test_layout_rule_mutation(case, capsys):
    mode, witness, defect = CASES[case]
    script = '''
import inspect
import pytest
from rfx.sources import coaxial_port
from rfx import interop
from rfx.interop import _design as design
from rfx.model import coax_line, coax_planes, coax_shield
from rfx.model import coax_port as model_coax_port

def rewrite(owner, name, old, new):
    """Keep altered coaxial functions in their own globals."""
    fn = getattr(owner, name)
    source = inspect.getsource(fn)
    assert source.count(old) == 1, (name, old, source.count(old))
    scope = inspect.unwrap(fn).__globals__
    exec(source.replace(old, new), scope)
    setattr(owner, name, scope[name])
'''
    script += defect + "\n"
    script += "model_coax_port.coax_port_layout = coax_line.coax_port_layout\n"
    script += f"raise SystemExit(pytest.main([{witness!r}, '-q', '-o', 'addopts=', '--tb=short']))\n"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=""))
    with capsys.disabled():
        print(f"{case} {mode}: exit={result.returncode}\n{result.stdout}\n{result.stderr}", file=sys.stderr)
    assert result.returncode == 1
    assert "FAILED " + witness in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s", "-o", "addopts="]))
