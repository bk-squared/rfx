"""Only assembled continuations replace a conductor's drawn end by the wall."""
import numpy as np
import pytest

from rfx import Box, Simulation


DX = .0005
DOMAIN_CELLS = 10.05


def continued_strip(kind="msl", *, shifted=False, graded=False, thin=False):
    """Translate the identical strip one cell inward to remove its sole reached face."""
    sim = Simulation(domain=(DOMAIN_CELLS*DX, 8*DX, 6*DX), dx=DX, freq_max=1e9,
                     cpml_layers=3, boundary="cpml", snap="declared",
                     **({"dz_profile": np.full(6, DX)} if graded else {}))
    offset = -DX if shifted else 0.
    strip = Box((3.36*DX + offset, 2*DX, 2*DX),
                (DOMAIN_CELLS*DX + offset, 6*DX, 2*DX))
    if thin:
        sim.add_thin_conductor(strip)
    else:
        sim.add(strip, material="pec")
    if kind == "msl":
        sim.add_msl_port((7*DX, 4*DX, 0), width=4*DX, height=2*DX,
                         direction="-x", mode="uniform", n_probe_offset=3,
                         n_probe_spacing=2, n_probes=3)
    else:
        sim.add_port((7*DX, 4*DX, DX), component="ez", terminates=())
    return sim


def realization_advisories(sim):
    """Read only this strip's table verdict; other preflight findings remain visible."""
    return [str(issue) for issue in sim.preflight()
            if issue.code == "realization" and issue.loc in ("geometry[0]", "thin_conductor[0]")]


@pytest.mark.parametrize("kind", ["msl", "lumped"])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("thin", [False, True])
def test_continued_strip_has_no_realization_advisory(kind, graded, thin):
    sim = continued_strip(kind, graded=graded, thin=thin)
    record = sim.realized_geometry()
    assert record.entities[0].continued_faces == ("x-hi",)
    assert realization_advisories(sim) == []


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("thin", [False, True])
def test_shifted_strip_keeps_realization_advisory(graded, thin):
    sim = continued_strip(shifted=True, graded=graded, thin=thin)
    assert sim.realized_geometry().entities[0].continued_faces == ()
    assert any("centroid shift" in issue for issue in realization_advisories(sim))
