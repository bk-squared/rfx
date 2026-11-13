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
    if multiple_devices:
        # No multi-device lane steps a waveguide port, so that refusal comes
        # first on every entry, before the aperture is judged.
        with pytest.raises(NotImplementedError, match="waveguide ports .* not supported on"):
            one_bare_edge(graded).run(n_steps=2, skip_preflight=True,
                                      devices=[jax.devices()[0]] * 2)
        return
    with pytest.raises(ValueError) as caught:
        one_bare_edge(graded).run(n_steps=2, skip_preflight=True)
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


def conformal_guide(conformal=True):
    """A WR-90 aperture (22.86 x 10.16 mm) bounded only by Dey-Mittra walls.

    dx = 1 mm snaps the rims to 23 and 10 mm; the conformal half-spaces
    start at 22.86 and 10.16 mm, inside the cells holding those rims. The
    PEC domain faces at 25 and 12 mm are 2 mm away.
    """
    from rfx.boundaries.spec import Boundary, BoundarySpec
    walls = Boundary(lo="pec", hi="pec", conformal=conformal)
    sim = Simulation(10e9, (.06, .025, .012), dx=.001, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y=walls, z=walls))
    sim.add_waveguide_port(.02, direction="+x", y_range=(0., .02286),
                           z_range=(0., .01016), f0=8e9, bandwidth=.5, name="conformal-rim")
    return sim


def _edge_refusal(sim, **kwargs):
    from rfx.boundaries.waveguide_edges import refuse_bare_waveguide_edges
    with pytest.raises(ValueError) as caught:
        refuse_bare_waveguide_edges(sim, sim._build_realized_grid(), lane="contract", **kwargs)
    return str(caught.value)


def test_conformal_walls_bound_the_aperture():
    # The realized conformal weights close both snapped rims, at dispatch and in the runner.
    admit_waveguide(conformal_guide())
    message = _edge_refusal(conformal_guide(conformal=False))
    assert "y_hi node plane 23 mm, nearest conductor distance 2 mm" in message
    assert "z_hi node plane 10 mm, nearest conductor distance 2 mm" in message


def test_mutation_conformal_weights_ignored_refuses():
    message = _edge_refusal(conformal_guide(), conformal_weights=None)
    assert "y_hi node plane 23 mm" in message and "z_hi node plane 10 mm" in message


def test_mutation_conformal_wall_two_cells_out_refuses():
    # Production weights from the same half-spaces, the y wall moved 2 cells out.
    from rfx.geometry.conformal import clamp_conformal_weights, compute_conformal_weights_sdf
    sim = conformal_guide()
    big = 10.
    shapes = [Box((-big, .02486, -big), (big, big, big)),
              Box((-big, -big, .01016), (big, big, big))]
    weights = clamp_conformal_weights(
        *compute_conformal_weights_sdf(sim._build_realized_grid(), shapes), .1)
    message = _edge_refusal(sim, conformal_weights=weights)
    assert "y_hi node plane 23 mm" in message
    assert "z_hi" not in message


def test_runner_judges_the_weights_it_formed():
    # conformal_pec=False runs the binary walls, whose rims are bare.
    with pytest.raises(ValueError, match="run_uniform: Waveguide port 'conformal-rim' has bare"):
        conformal_guide().run(n_steps=2, skip_preflight=True, conformal_pec=False)


@pytest.mark.docs_consistency
def test_rule4_support_refusal_row():
    text = (Path(__file__).parents[2] / "docs/guides/support_matrix.md").read_text()
    row, = [line for line in text.splitlines() if "waveguide_aperture_conductors" in line]
    for part in ("refused", "both meshes", "before", "PEC", "aperture"):
        assert part in row


@pytest.mark.parametrize("absorbing", ["lo", "hi"])
def test_omitted_range_spans_the_domain_with_unequal_pads(absorbing):
    # z_range omitted, z faces PEC on one side and CPML on the other: the
    # guide must span the 16 mm domain height, neither reaching into the
    # absorber pad nor stopping one pad short of the PEC face. A drawn wall
    # closes the rim on the absorbing side; y is a partial, walled aperture.
    from rfx.boundaries.spec import Boundary, BoundarySpec
    faces = {"lo": "pec", "hi": "pec", absorbing: "cpml"}
    sim = Simulation(30e9, (.032, .020, .016), dx=.002, cpml_layers=4,
                     boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(**faces)))
    rim = (0., .002) if absorbing == "lo" else (.014, .016)
    for lo, hi in (((0, .004, 0), (.032, .006, .016)),
                   ((0, .014, 0), (.032, .016, .016)),
                   ((0, .004, rim[0]), (.032, .016, rim[1]))):
        sim.add(Box(lo, hi), material="pec")
    sim.add_waveguide_port(.008, direction="+x", y_range=(.006, .014),
                           f0=25e9, freqs=[24e9, 26e9], ref_offset=1, probe_offset=2)
    admit_waveguide(sim)
    grid = sim._build_grid()
    assert getattr(grid, f"pad_z_{absorbing}") == 4
    cfg = sim._build_waveguide_port_config(
        sim._waveguide_ports[0], grid, jnp.asarray([24e9, 26e9]), 8)
    assert (cfg.v_lo, cfg.v_hi) == (grid.pad_z_lo, grid.nz - 1 - grid.pad_z_hi)
    assert cfg.b == pytest.approx(.016, rel=1e-9)
    assert cfg.a == pytest.approx(.008, rel=1e-9)
