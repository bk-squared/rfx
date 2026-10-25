"""#1826 rule 4: the declared mode requires realized conductors on its rim."""
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.features import admit_waveguide


def one_bare_edge(graded, boundary="cpml"):
    kwargs = {"dz_profile": np.full(8, .002)} if graded else {}
    sim = Simulation(30e9, (.032, .020, .016), dx=.002,
                     boundary=boundary, cpml_layers=4, **kwargs)
    for lo, hi in (
        ((0, .004, .002), (.032, .006, .014)),
        ((0, .004, .002), (.032, .016, .004)),
        ((0, .004, .012), (.032, .016, .014)),
    ):
        sim.add(Box(lo, hi), material="pec")
    sim.add_waveguide_port(.008, direction="+x", y_range=(.006, .014),
                           z_range=(.004, .012), f0=25e9, freqs=[24e9, 26e9],
                           ref_offset=1, probe_offset=2, name="bare-rim")
    return sim


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("multiple_devices", [False, True])
def test_f3_one_bare_absorbing_edge_refuses_before_solve(graded, multiple_devices):
    devices = {"devices": [jax.devices()[0]] * 2} if multiple_devices else {}
    with pytest.raises(ValueError) as caught:
        one_bare_edge(graded).run(n_steps=2, skip_preflight=True, **devices)
    message = str(caught.value)
    for part in ("bare-rim", "bare aperture edges", "y_hi", "node plane 14 mm",
                 "nearest conductor distance 8 mm", "Draw a PEC wall", "domain cross-section",
                 "equal to the aperture", "transverse faces PEC"):
        assert part in message
    for closed in ("y_lo node plane", "z_lo node plane", "z_hi node plane"):
        assert closed not in message


@pytest.mark.parametrize("graded", [False, True])
def test_f1b_remote_pec_face_does_not_close_a_bare_aperture(graded):
    sim = one_bare_edge(graded, {"x": "cpml", "y": "pec", "z": "pec"})
    with pytest.raises(ValueError) as caught:
        admit_waveguide(sim)
    assert "y_hi node plane 14 mm" in str(caught.value)
    assert "nearest conductor distance 6 mm" in str(caught.value)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("margin", [6, 10])
def test_f2_drawn_ci_guide_remains_admitted(graded, margin):
    from tests.contracts.test_guide_partial_aperture_parity import scene
    admit_waveguide(scene(graded, margin=margin))


def test_traced_pec_operand_is_refused_by_name(monkeypatch):
    from rfx.boundaries import waveguide_edges
    sim = one_bare_edge(False)
    grid = sim._build_grid()

    def check(mask):
        monkeypatch.setattr(waveguide_edges, "realized_operands", lambda *args: (
            SimpleNamespace(name="pec_mask", array=mask),))
        waveguide_edges.refuse_bare_waveguide_edges(sim, grid, lane="contract")
        return mask

    with pytest.raises(ValueError, match="traced PEC operand pec_mask"):
        jax.make_jaxpr(check)(jnp.zeros(grid.shape, dtype=bool))


@pytest.mark.docs_consistency
def test_rule4_support_refusal_row():
    text = (Path(__file__).parents[2] / "docs/guides/support_matrix.md").read_text()
    row, = [line for line in text.splitlines() if "waveguide_aperture_conductors" in line]
    for part in ("refused", "both meshes", "before", "PEC", "aperture"):
        assert part in row
