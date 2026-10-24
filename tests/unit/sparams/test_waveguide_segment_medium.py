"""Vacuum modal ports must never measure a loaded launch segment (#1994)."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.features import waveguide_segment_cells

DX, A, B, L = .002, .024, .012, .12


def guide(graded=False, *, direction="+x", axis="x", **port):
    domain = [A, B]
    domain.insert("xyz".index(axis), L)
    boundary = {a: "cpml" if a == axis else "pec" for a in "xyz"}
    sim = Simulation(freq_max=12e9, domain=tuple(domain), dx=DX,
                     boundary=boundary, cpml_layers=8,
                     **({f"d{axis}_profile": np.full(60, DX)} if graded else {}))
    sim.add_waveguide_port(.02 if direction.startswith("+") else .10,
                          direction=direction[0]+axis, name="launch", f0=9e9,
                          freqs=np.array([8e9, 10e9]), ref_offset=3, probe_offset=10, **port)
    return sim


def execute(sim, route="run", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if route == "compute_waveguide_s_matrix" and sim._uses_nonuniform_mesh:
            kw.setdefault("normalize", "flux")
        return getattr(sim, route)(n_steps=2, skip_preflight=True, **kw)


def fill(sim, lo, hi, material="fill", **props):
    if material != "pec":
        sim.add_material(material, **(props or dict(eps_r=2.2)))
    sim.add(Box((lo, 0, 0), (hi, A, B)), material=material)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("route", ["run", "forward", "compute_waveguide_s_matrix"])
def test_filled_guide_refused(graded, route):
    sim = guide(graded)
    fill(sim, -.1, L+.1)
    if route == "compute_waveguide_s_matrix":
        sim.add_waveguide_port(.10, direction="-x", name="receive", f0=9e9,
                              freqs=np.array([8e9, 10e9]))
    with pytest.raises(ValueError, match=r"launch.*eps.*2\.2.*rfx#1994"):
        execute(sim, route)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("placement", ["h", "declared", "pec"])
def test_segment_extents(graded, placement):
    sim = guide(graded, **({"reference_plane": .07} if placement == "declared" else {}))
    lo, hi = {"h": (.018, .0199), "declared": (.06, .062), "pec": (.022, .024)}[placement]
    fill(sim, lo, hi, "pec" if placement == "pec" else "fill")
    with pytest.raises(ValueError, match=r"launch.*(eps|PEC).*cell.*rfx#1994"):
        execute(sim)


@pytest.mark.parametrize("graded", [False, True])
def test_between_reference_and_probe_admitted(graded):
    """S-matrix extraction reads reference waves only, not the probe plane."""
    sim = guide(graded)
    fill(sim, .032, .034)
    execute(sim)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("medium", ["empty", "vacuum", "empty_poles", "beyond"])
def test_vacuum_and_beyond_admitted(graded, medium, record_property, monkeypatch):
    from rfx.boundaries import features
    sim = guide(graded)
    if medium == "beyond":
        fill(sim, .028, .036)
    elif medium == "vacuum":
        sim.add(Box((0, 0, 0), (L, A, B)), material="vacuum")
    elif medium == "empty_poles":
        fill(sim, 0, L, eps_r=1., debye_poles=[], lorentz_poles=[])
    original = features.refuse_waveguide_segment_medium
    measured = []

    def observe(ports, grid, materials, *args, **kwargs):
        cfg = ports[0]
        measured.append(float(np.asarray(materials.eps_r)[cfg.ref_x, cfg.u_lo, cfg.v_lo]))
        return original(ports, grid, materials, *args, **kwargs)

    monkeypatch.setattr(features, "refuse_waveguide_segment_medium", observe)
    execute(sim)
    assert measured == [1.0]
    record_property("last_segment_eps", measured)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("operand", ["eps_override", "sigma_override"])
def test_override_refused(graded, operand):
    sim = guide(graded)
    grid = sim._build_realized_grid()
    vacuum = 1. if operand == "eps_override" else 0.
    value = jnp.full(grid.shape, vacuum).at[grid.index_of("x", .022), :, :].set(2.2)
    with pytest.raises(ValueError, match=r"launch.*(eps|sigma).*2\.2"):
        execute(sim, "forward", **{operand: value})


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("near", [True, False])
def test_design_box_window(graded, near):
    sim = guide(graded)
    box = ((.022 if near else .06, .004, .004),
           (.024 if near else .064, .006, .006))
    grid = sim._build_realized_grid()
    lo, hi = [tuple(grid.index_of(axis, value) for axis, value in zip("xyz", p)) for p in box]
    eps = jnp.full(tuple(b-a+1 for a, b in zip(lo, hi)), 2.2)
    if graded:
        with pytest.raises(NotImplementedError, match="design box.*waveguide"):
            execute(sim, "forward", design_box=box, design_eps_override=eps)
    elif near:
        with pytest.raises(ValueError, match=r"launch.*design_box update window"):
            execute(sim, "forward", design_box=box, design_eps_override=eps)
    else:
        execute(sim, "forward", design_box=box, design_eps_override=eps)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("direction", ["+x", "-x"])
def test_axes_and_h_index(graded, axis, direction):
    sim = guide(graded, axis=axis, direction=direction)
    grid = sim._build_realized_grid()
    entry = sim._waveguide_ports[0]
    if graded:
        from rfx.runners.nonuniform import _build_waveguide_port_config_nu
        cfg = _build_waveguide_port_config_nu(sim, entry, grid, entry.freqs, 2)
    else:
        cfg = sim._build_waveguide_port_config(entry, grid, entry.freqs, 2)
    bounds = waveguide_segment_cells(cfg, grid)
    a = "xyz".index(axis)
    assert bounds[a] == ((cfg.x_index-1, cfg.ref_x+1) if direction.startswith("+")
                         else (cfg.ref_x, cfg.x_index+1))
    execute(sim)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("profile", ["analytic", "discrete"])
def test_profiles_and_multimode(graded, profile):
    sim = guide(graded, n_modes=2, mode_profile=profile)
    fill(sim, 0, L)
    with pytest.raises(ValueError, match="launch.*eps"):
        execute(sim, "forward")


def test_traced_override_retains_callback():
    sim = guide()
    grid = sim._build_realized_grid()
    # jit executes a non-vacuum traced operand, rather than merely inspecting jaxpr.
    def call(eps):
        return execute(sim, "forward", eps_override=eps).time_series
    with pytest.raises(Exception, match="traced override over the segment"):
        jax.jit(call)(jnp.full(grid.shape, 2.2)).block_until_ready()


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("behind", [False, True])
def test_declared_reference_extends_segment(graded, behind):
    sim = guide(graded, reference_plane=.006 if behind else .07)
    fill(sim, .01 if behind else .06, .012 if behind else .062)
    with pytest.raises(ValueError, match="launch.*eps"):
        execute(sim)


@pytest.mark.parametrize("graded", [False, True])
def test_declared_probe_does_not_extend_segment(graded):
    sim = guide(graded, probe_plane=.07)
    fill(sim, .06, .062)
    execute(sim)
    sim.add_waveguide_port(.10, direction="-x", f0=9e9, freqs=np.array([8e9, 10e9]))
    with pytest.raises(ValueError, match="does not use per-port probe_plane"):
        execute(sim, "compute_waveguide_s_matrix")


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("behind", [False, True])
def test_declared_reference_excludes_its_cell_centre(graded, behind):
    from rfx.geometry.rasterize_grid import _axis_node_positions
    sim = guide(graded)
    grid = sim._build_realized_grid()
    entry = sim._waveguide_ports[0]
    if graded:
        from rfx.runners.nonuniform import _build_waveguide_port_config_nu
        cfg = _build_waveguide_port_config_nu(sim, entry, grid, entry.freqs, 2)
    else:
        cfg = sim._build_waveguide_port_config(entry, grid, entry.freqs, 2)
    nodes = np.asarray(_axis_node_positions(grid.cells("x"), grid.pad_x_lo))
    index = cfg.x_index - 4 if behind else cfg.ref_x + 4
    plane = nodes[index] + grid.cells("x")[index] / 2
    cfg = cfg._replace(segment_planes=(plane,))
    lo, hi = waveguide_segment_cells(cfg, grid)[0]
    assert (lo if behind else hi) == (index+1 if behind else index)


@pytest.mark.parametrize("graded", [False, True])
def test_cpml_material_sigma_only(graded):
    from dataclasses import replace
    sim = guide(graded)
    sim._waveguide_ports[0] = replace(sim._waveguide_ports[0], x_position=.002)
    execute(sim)


@pytest.mark.parametrize("graded", [False, True])
def test_isolated_h_operand(graded):
    """Direct setup material isolates H from the assembler's E-edge averaging."""
    from rfx.boundaries.features import refuse_waveguide_segment_medium
    from rfx.core.yee import MaterialArrays
    sim = guide(graded)
    grid = sim._build_realized_grid()
    entry = sim._waveguide_ports[0]
    if graded:
        from rfx.runners.nonuniform import _build_waveguide_port_config_nu
        cfg = _build_waveguide_port_config_nu(sim, entry, grid, entry.freqs, 2)
    else:
        cfg = sim._build_waveguide_port_config(entry, grid, entry.freqs, 2)
    eps = jnp.ones(grid.shape).at[cfg.x_index-1, :, :].set(2.2)
    materials = MaterialArrays(eps, jnp.zeros(grid.shape), jnp.ones(grid.shape))
    with pytest.raises(ValueError, match="launch.*eps"):
        refuse_waveguide_segment_medium([cfg], grid, materials, {}, ())


def test_scalar_low_level_medium():
    from rfx.boundaries.features import refuse_waveguide_segment_medium
    from rfx.core.yee import MaterialArrays
    sim = guide()
    grid = sim._build_realized_grid()
    entry = sim._waveguide_ports[0]
    cfg = sim._build_waveguide_port_config(entry, grid, entry.freqs, 2)
    with pytest.raises(ValueError, match="permeability.*mu_r.*2"):
        refuse_waveguide_segment_medium([cfg], grid, MaterialArrays(1., 0., 2.), {}, ())


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("kind", ["mu", "sigma", "debye", "lorentz", "chi3"])
def test_other_operands(graded, kind):
    from rfx import DebyePole
    from rfx.materials.lorentz import lorentz_pole
    properties = {
        "mu": dict(mu_r=2.), "sigma": dict(sigma=.1),
        "debye": dict(debye_poles=[DebyePole(delta_eps=1., tau=1e-10)]),
        "lorentz": dict(lorentz_poles=[lorentz_pole(delta_eps=1., omega_0=2*np.pi*5e9, delta=1e10)]),
        "chi3": dict(chi3=1e-3),
    }
    sim = guide(graded)
    fill(sim, .022, .024, eps_r=1., **properties[kind])
    if graded and kind == "chi3":
        with pytest.raises(NotImplementedError, match="Kerr|chi3"):
            execute(sim)
    else:
        with pytest.raises(ValueError, match="launch.*rfx#1994"):
            execute(sim)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("route", ["run", "forward"])
def test_two_device_routes_refuse(graded, route, record_property):
    sim = guide(graded)
    fill(sim, 0, L)
    devices = jax.devices("cpu")
    assert len(devices) >= 2
    kwargs = dict(devices=devices[:2])
    if route == "forward":
        kwargs["distributed"] = True
    with pytest.raises((ValueError, NotImplementedError), match="waveguide|Waveguide|launch") as caught:
        execute(sim, route, **kwargs)
    record_property("refusal", str(caught.value))


@pytest.mark.parametrize("validation", ["production", "research"])
def test_subgridded_refuses(validation, record_property):
    sim = guide()
    sim.add_refinement(z_range=(.004, .008), ratio=2, validation=validation)
    with pytest.raises((ValueError, NotImplementedError), match="waveguide|Waveguide|subgridded") as caught:
        execute(sim)
    record_property("refusal", str(caught.value))
