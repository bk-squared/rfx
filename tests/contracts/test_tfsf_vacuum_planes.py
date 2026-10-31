"""The slab TF/SF vacuum rule applies to concrete and traced public entries."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


VACUUM_MESSAGE = (
    "TFSF plane-wave source requires vacuum on and adjacent to the TFSF x "
    "boundaries; non-vacuum material found at x_lo-1"
)
# The refusal names the operand the step would apply on the plane (#1676).
PERMITTIVITY_MESSAGE = VACUUM_MESSAGE + ": permittivity (materials.eps_r)"
DESIGN_BOX_MESSAGE = VACUUM_MESSAGE + ": caller-supplied non-vacuum mask (nonvacuum_masks.0)"
TRACED_MESSAGE = (
    "TFSF plane-wave source requires vacuum on and adjacent to the TFSF x "
    "boundaries; non-vacuum material found at x_lo"
)


def _scene(graded):
    profile = 1e-3 * np.array([1.2, 1.1, 1., .9, .8, .7, .6,
                              .6, .7, .8, .9, 1., 1.1, .6])
    mesh = {"dz_profile": profile} if graded else {}
    sim = Simulation(freq_max=20e9, domain=(.030, .020, .012), dx=.001,
                     cpml_layers=4, **mesh)
    sim.add_tfsf_source(margin=3)
    sim.add_probe((.023, .012, .007), "ez")
    return sim


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_box_on_vacuum_planes_refused(graded, entry):
    sim = _scene(graded)
    sim.add_material("fill", eps_r=4.)
    sim.add(Box((0., 0., 0.), (.030, .020, .012)), material="fill")
    with pytest.raises(ValueError) as caught:
        getattr(sim, entry)(n_steps=8)
    assert str(caught.value) == PERMITTIVITY_MESSAGE


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_forward_override_on_vacuum_planes_refused(graded):
    sim = _scene(graded)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    with pytest.raises(ValueError) as caught:
        sim.forward(n_steps=8, eps_override=jnp.full(grid.shape, 4.))
    assert str(caught.value) == PERMITTIVITY_MESSAGE


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_traced_forward_override_on_x_lo_refused(graded):
    sim = _scene(graded)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()

    def energy(eps):
        # Four pad cells plus margin three: x_lo is node seven.
        # Keep its neighbours vacuum so the literal must name x_lo.
        material = jnp.ones(grid.shape).at[7, :, :].set(eps)
        return jnp.sum(sim.forward(n_steps=8, eps_override=material).time_series ** 2)

    with pytest.raises(Exception) as caught:
        jax.grad(energy)(4.)
    assert TRACED_MESSAGE in str(caught.value)
    print(f"traced graded={graded}: {type(caught.value).__name__}: {caught.value}")


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_interior_dielectric_with_vacuum_planes_admitted(graded, entry):
    sim = _scene(graded)
    sim.add_material("slab", eps_r=4.)
    sim.add(Box((.013, 0., 0.), (.017, .020, .012)), material="slab")
    result = getattr(sim, entry)(n_steps=120)
    values = np.asarray(result.time_series)
    assert np.isfinite(values).all()
    assert np.max(np.abs(values)) > 0


@pytest.mark.parametrize("crossing", [True, False], ids=["crossing", "interior"])
def test_graded_design_box_vacuum_planes(crossing):
    sim = _scene(graded=True)
    x_lo, x_hi = (.001, .029) if crossing else (.013, .017)
    box = ((x_lo, .008, .005), (x_hi, .012, .007))
    grid = sim._build_nonuniform_grid()
    bounds = sim._design_box_bounds_from_corners(grid, box)
    shape = tuple(bounds[2 * d + 1] - bounds[2 * d] for d in range(3))
    eps = jnp.full(shape, 4.)
    if crossing:
        with pytest.raises(ValueError) as caught:
            sim.forward(n_steps=120, design_box=box, design_eps_override=eps)
        assert str(caught.value) == DESIGN_BOX_MESSAGE
    else:
        # The vacuum rule admits an interior box; the plane wave's invariance
        # admission then refuses the finite box, as on the uniform mesh (#1794).
        with pytest.raises(ValueError) as caught:
            sim.forward(n_steps=120, design_box=box, design_eps_override=eps)
        assert str(caught.value) == (
            "TF/SF y_lo, y_hi: design_box prevents an invariant periodic "
            "replacement; on the graded mesh a finite scatterer or an array "
            "under a plane wave is not available: run it on the uniform mesh "
            "(closed_box=True, or y='periodic')")


def test_traced_x_cell_sizes_refused():
    def energy(profile):
        sim = Simulation(freq_max=20e9, domain=(.030, .020, .012), dx=.001,
                         dx_profile=profile, cpml_layers=4)
        sim.add_tfsf_source()
        sim.add_probe((.023, .012, .007), "ez")
        return jnp.sum(sim.forward(n_steps=8, skip_preflight=True).time_series ** 2)

    with pytest.raises(ValueError) as caught:
        jax.grad(energy)(jnp.full(30, .001))
    assert str(caught.value) == (
        "TF/SF requires host x cell widths; a traced propagation "
        "profile (mesh as design variable) is not supported")


@pytest.mark.parametrize("mesh,source,treatment,entry,placement", [
    pytest.param(mesh, source, treatment, entry, "crossing",
                 id=f"{mesh}-{source}-{treatment}-{entry}-crossing")
    for mesh, source, treatments in (
        ("uniform", "slab", ("subpixel", "kottke_pec", "conformal_pec", "full_tensor")),
        ("uniform", "methodB", ("subpixel", "kottke_pec", "conformal_pec",
                                "boundary_conformal", "full_tensor")),
        ("graded", "slab", ("subpixel", "full_tensor")),
        ("uniform", "closed_box", ("subpixel", "kottke_pec", "conformal_pec", "full_tensor")),
    )
    for treatment in treatments
    for entry in (("run",) if source == "closed_box" or treatment in ("conformal_pec", "boundary_conformal")
                  else ("run", "forward"))
] + [
    pytest.param("uniform", "closed_box", treatment, "run", "inside",
                 id=f"uniform-closed_box-{treatment}-run-inside")
    for treatment in ("subpixel", "kottke_pec", "conformal_pec",
                      "boundary_conformal", "full_tensor")
])
def test_treated_edges_on_injection_planes(mesh, source, treatment, entry, placement):
    pec_crossing = source != "closed_box" and treatment in ("kottke_pec", "conformal_pec")
    if placement == "crossing" and not pec_crossing:
        # The identical staircase scene must reach stepping with treatment off.
        sim, kwargs = _treated_edge_scene(mesh, source, treatment, entry, placement, False)
        result = getattr(sim, entry)(n_steps=2, skip_preflight=True, **kwargs)
        assert np.isfinite(np.asarray(result.time_series)).all()

    sim, kwargs = _treated_edge_scene(mesh, source, treatment, entry, placement, True)
    if placement == "inside":
        result = getattr(sim, entry)(n_steps=2, skip_preflight=True, **kwargs)
        assert np.isfinite(np.asarray(result.time_series)).all()
    else:
        # Literal treated labels; closed-box realization converts treatment
        # arrays to non-vacuum masks before checking the plane.
        label = {
            "subpixel": "anisotropic permittivity" if mesh == "graded" else "tensor permittivity",
            "full_tensor": "node tensor",
            "kottke_pec": r"[^()]+",
            "conformal_pec": r"[^()]+",
            "boundary_conformal": "component permeability",
        }[treatment]
        if source == "closed_box":
            label = "caller-supplied non-vacuum mask"
        plane = r"x=(?:7|8)" if source == "closed_box" else r"x_lo(?:-1)?"
        with pytest.raises(ValueError, match=rf"requires vacuum.*{plane}: {label} \(") as caught:
            getattr(sim, entry)(n_steps=2, skip_preflight=True, **kwargs)
        print(f"treated refusal: {caught.value}")


def _treated_edge_scene(mesh, source, treatment, entry, placement, treated):
    inside = placement == "inside"
    wall = treatment == "boundary_conformal" and not inside
    options = {"dz_profile": [.00018] * 12 + [.00022] * 12} if mesh == "graded" else {}
    boundary = (BoundarySpec(x="cpml", y=Boundary("pec", "pec", conformal=treated),
                             z="periodic") if wall else "cpml")
    sim = Simulation(freq_max=100e9, domain=(.0048, .00487 if wall else .0048, .0048),
                     dx=.0002, cpml_layers=4, boundary=boundary, snap="declared", **options)
    source_options = ({"method": "methodB", "angle_deg": 20.} if source == "methodB"
                      else {"closed_box": True} if source == "closed_box" else {})
    sim.add_tfsf_source(f0=50e9, bandwidth=.5, margin=4, polarization="ez",
                        direction="+x", **source_options)
    if not wall:
        # Four pad cells and margin four put x_lo at x=0.8 mm (node 8).
        # Dielectric at node 8.35 leaves the plain plane vacuum; smoothing reaches it.
        dielectric = treatment in ("subpixel", "full_tensor")
        # Open-slab/Method-B PEC has no treated-only crossing: its half-cell skin also puts staircase Ex on the plane.
        x_face = .00087 if dielectric else .00091 if source == "closed_box" else .00067
        if inside:
            lo, hi = (.00227,) * 3, (.00267,) * 3
        else:
            lo, hi = (x_face, -.002, -.002), (.00147, .007, .007)
            if not dielectric and source != "closed_box":
                lo, hi = (.00067, -.002, -.002), (.00107, .007, .007)
            if source == "methodB":
                lo, hi = (x_face, .00227, -.002), (hi[0], .00267, .007)
            elif source == "closed_box":
                lo, hi = (x_face, .00227, .00227), (.00147, .00267, .00267)
        if dielectric:
            sim.add_material("dielectric", eps_r=4.)
        sim.add(Box(lo, hi), material="dielectric" if dielectric else "pec")
    kwargs = {"compute_s_params": False} if entry == "run" else {"checkpoint": False}
    if not treated:
        return sim, kwargs
    if treatment in ("subpixel", "kottke_pec", "full_tensor"):
        kwargs["subpixel_smoothing"] = True if treatment == "subpixel" else treatment
    elif treatment == "conformal_pec" or inside:
        # A domain wall cannot move inside TF; its control is an interior
        # conformal PEC box. Closed box also admits the full-tensor control.
        kwargs["conformal_pec"] = True
    return sim, kwargs
