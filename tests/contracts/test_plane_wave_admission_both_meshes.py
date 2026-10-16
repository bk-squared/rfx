"""One open plane-wave declaration has one invariance admission on both meshes."""
import sys

import numpy as np
import pytest

from rfx import Box, Simulation


FIRST_SENTENCE = (
    "TF/SF y_lo, y_hi: update.materials.eps_r prevents an invariant periodic replacement"
)
UNIFORM_MESSAGE = (
    FIRST_SENTENCE + "; declare boundary with y='periodic' to solve an array, "
    "or use closed_box=True for a finite scatterer."
)
GRADED_MESSAGE = (
    FIRST_SENTENCE + "; on the graded mesh a finite scatterer or an array under a "
    "plane wave is not available: run it on the uniform mesh (closed_box=True, "
    "or y='periodic')"
)


def scene(graded, *, finite):
    mesh = {"dz_profile": np.full(16, .001)} if graded else {}
    sim = Simulation(freq_max=15e9, domain=(.024, .016, .016), dx=.001,
                     boundary="cpml", cpml_layers=6, **mesh)
    sim.add_material("dielectric", eps_r=4)
    corners = ((.009, .005, .005), (.015, .011, .011)) if finite else (
        (.013, 0., 0.), (.016, .016, .016))
    sim.add(Box(*corners), material="dielectric")
    sim.add_tfsf_source(f0=8e9, polarization="ez", direction="+x", margin=3,
                        method="bloch")
    sim.add_probe((.0105, .0065, .0095), "ez")
    sim.add_probe((.0195, .0095, .0065), "ez")
    return sim


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_finite_cube_refused(graded, entry):
    """A finite cube breaks the lateral invariance required by the open plane wave.

    The declaration must be refused on both meshes before any field step.
    """
    with pytest.raises(ValueError) as caught:
        getattr(scene(graded, finite=True), entry)(n_steps=20, skip_preflight=True)
    assert str(caught.value) == (GRADED_MESSAGE if graded else UNIFORM_MESSAGE)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_full_span_slab_equal_fields(entry):
    """A full-span slab produces a laterally invariant field.

    Transverse CPML has nothing to act on, so both meshes give the same probes.
    """
    uniform = np.asarray(getattr(scene(False, finite=False), entry)(
        n_steps=200, skip_preflight=True).time_series)
    graded = np.asarray(getattr(scene(True, finite=False), entry)(
        n_steps=200, skip_preflight=True).time_series)
    assert uniform.shape == graded.shape == (200, 2)
    assert np.isfinite(uniform).all() and np.isfinite(graded).all()
    for probe in range(2):
        peak = np.max(np.abs(uniform[:, probe]))
        assert peak > 0
        relative = np.max(np.abs(graded[:, probe] - uniform[:, probe])) / peak
        print(f"{entry} probe{probe}: max|graded-uniform|/max|uniform|={relative:.9g}", file=sys.stderr)
        assert relative <= 2e-6
