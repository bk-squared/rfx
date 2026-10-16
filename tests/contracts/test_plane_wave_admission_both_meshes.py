"""One open plane-wave declaration has one invariance admission on both meshes."""
import sys

import numpy as np
import pytest

from rfx import Box, Simulation


UNIFORM_REMEDY = (
    "; declare boundary with y='periodic' to solve an array, "
    "or use closed_box=True for a finite scatterer."
)
GRADED_REMEDY = (
    "; on the graded mesh a finite scatterer or an array under a "
    "plane wave is not available: run it on the uniform mesh (closed_box=True, "
    "or y='periodic')"
)


def refusal(feature, graded):
    return (f"TF/SF y_lo, y_hi: {feature} prevents an invariant periodic replacement"
            + (GRADED_REMEDY if graded else UNIFORM_REMEDY))


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
    assert str(caught.value) == refusal("update.materials.eps_r", graded)



@pytest.mark.parametrize("override", ["pec_mask", "pec_occupancy", "design_box"])
@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_finite_forward_override_refused(graded, override):
    """forward() inputs that place a finite block on the slab break the same invariance.

    The block (x 9-12 mm, y and z 5-9 mm) is away from the injection planes.
    """
    import jax.numpy as jnp
    sim = scene(graded, finite=False)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    block = (slice(15, 18), slice(11, 15), slice(11, 15))
    if override == "design_box":
        corners = ((.009, .005, .005), (.012, .009, .009))
        bounds = sim._design_box_bounds_from_corners(grid, corners)
        shape = tuple(bounds[2 * d + 1] - bounds[2 * d] for d in range(3))
        options = dict(design_box=corners, design_eps_override=jnp.full(shape, 4.))
        feature = "design_box"
    else:
        array = np.zeros(grid.shape, bool if override == "pec_mask" else np.float32)
        array[block] = 1
        options = {f"{override}_override": jnp.asarray(array)}
        feature = f"update.{override}"
    with pytest.raises(ValueError) as caught:
        sim.forward(n_steps=5, skip_preflight=True, **options)
    assert str(caught.value) == refusal(feature, graded)

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
