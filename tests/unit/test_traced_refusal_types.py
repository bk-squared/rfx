"""Census refusals raise rfx's own error types with their old messages.

A site moved from ``ValueError`` raises ``RfxRefusal``; a site moved from
``NotImplementedError`` raises ``RfxUnsupported``. Each test reaches its
site through the call named in its row, asserts the exact new type, the old
builtin and ``RfxError`` as bases, and the old message. A kept
``NotImplementedError`` site stays exactly ``NotImplementedError``.

Expected messages are literals, or are formatted here from the inputs the
test passes. Where a message embeds a tracer's repr, the literal text on
each side is asserted and the middle is the repr of the tracer the test
passed, read inside the same trace.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import (Box, GaussianPulse, OrientedBox, RelativeStrength, Simulation, Sphere,
                 drude_pole_map, lorentz_pole, lorentz_pole_map)
from rfx.boundaries.graded_periodic import admit_graded_periodic
from rfx.boundaries.spec import BoundarySpec
from rfx.core.yee import init_materials
from rfx.errors import RfxError, RfxRefusal, RfxUnsupported
from rfx.geometry.rasterize_grid import GridCoords, _static_wire_min_cell, sheet_spec_from_shape
from rfx.geometry.smoothing import _reaches_pad_face
from rfx.materials.thin_conductor import PinnedSheet, ThinConductor
from rfx.model import NominalProfile
from rfx.model.graded_admission import (admit_graded_profiles, admit_nominal_profiles,
                                        admit_traced_windows)
from rfx.model.graded_grid import make_nonuniform_grid
from rfx.model.lumped_stamp import _nu_cell_arrays
from rfx.runners._exchange_interval import validate_exchange_interval
from rfx.runners.nonuniform import assemble_interface_eps_nu
from rfx.sources.msl_port import _msl_grid_geometry
from rfx.sources.sources import CustomWaveform
from rfx.sources.tfsf_discrete_plane_wave import source_scalar
from rfx.sources.wire_radius import stamp_wire_radius
from rfx.stepping.progress import validate_report_every

D = .001
NOMINAL = D * np.array([1., .8, 1.2] + [1.] * 13)
CELL = (7, 7, 7)
MM = 1e-3
DZ = [0.5e-3] * 6 + [0.3e-3] * 5
PERIODIC_FACTORS = np.array([1., .8, 1.2, .9, 1.1, .8, 1.2, .9, 1.1, 1., 1., 1.])


def assert_moved(exc, new, old, message):
    assert type(exc) is new
    assert isinstance(exc, old)
    assert isinstance(exc, RfxError)
    assert str(exc) == message


def assert_moved_around(exc, new, old, prefix, tracer_repr, suffix):
    assert type(exc) is new
    assert isinstance(exc, old)
    assert isinstance(exc, RfxError)
    message = str(exc)
    assert message.startswith(prefix)
    assert message.endswith(suffix)
    assert message == prefix + tracer_repr + suffix


def assert_kept(exc, message):
    assert type(exc) is NotImplementedError
    assert not isinstance(exc, RfxError)
    assert str(exc) == message


def raised(call, *args):
    with pytest.raises(Exception) as caught:
        call(*args)
    return caught.value


def traced_graded_grid(profile, nominal_profiles=None):
    return make_nonuniform_grid((.016, .016), np.full(16, D), D, 0, dx_profile=profile,
                                dy_profile=NOMINAL, nominal_profiles=nominal_profiles)


def in_traced_grid(build):
    """Run ``build(profile)`` with the x cell widths traced under ``jax.jvp``."""
    def objective(profile):
        build(profile)
        return jnp.sum(profile)

    return raised(jax.jvp, objective, (jnp.asarray(NOMINAL),), (jnp.ones(16),))


def test_auto_mesh_refuses_traced_geometry():
    def selected_dx(width):
        sim = Simulation(freq_max=10e9, domain=(0.008, 0.008, 0.008), boundary="pec")
        sim.add(Box((0.001, 0.001, 0.001), (width, 0.006, 0.006)), material="fr4")
        return sim._dx

    exc = raised(jax.jit(selected_dx), jnp.float32(0.006))
    assert_moved(exc, RfxRefusal, ValueError,
                 "Auto mesh requires static geometry and materials; supply dx= explicitly "
                 "(and fixed profiles) before tracing design inputs.")


def coax_sim(**mesh):
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=40.0e9, boundary="cpml", dx=1.0e-3, **mesh)
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim


def test_coaxial_line_refuses_a_graded_profile():
    sim = coax_sim(dz_profile=np.full(40, 1.0e-3))
    exc = raised(lambda: sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1))
    assert_moved(exc, RfxRefusal, ValueError,
                 "compute_coaxial_line_reflection() supports only a uniform Yee grid; "
                 "dx_profile, dy_profile, and dz_profile are not supported.")


def test_coaxial_line_refuses_registered_geometry():
    sim = coax_sim()
    sim.add_material("test_dielectric", eps_r=2.0)
    sim.add(Box((0.001, 0.001, 0.010), (0.002, 0.002, 0.011)), material="test_dielectric")
    exc = raised(lambda: sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1))
    assert_moved(exc, RfxRefusal, ValueError,
                 "compute_coaxial_line_reflection() constructs the complete line geometry; "
                 "registered geometry and thin conductors are not supported. Use the "
                 "documented Simulation, port, and method arguments instead.")


def test_coaxial_line_refuses_a_fractional_probe_count():
    sim = coax_sim()
    exc = raised(lambda: sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1, probe_count=3.5))
    assert_moved(exc, RfxRefusal, ValueError, "probe_count must be an integer of at least 3.")


def test_coaxial_two_port_refuses_a_graded_profile():
    sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9, boundary="cpml",
                     dz_profile=np.full(60, .001))
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    exc = raised(lambda: sim.compute_coaxial_two_port(n_steps=1, freqs=np.array([8e9])))
    assert_moved(exc, RfxRefusal, ValueError,
                 "compute_coaxial_two_port() supports only a uniform Yee grid; "
                 "dx_profile, dy_profile, and dz_profile are not supported.")


def waveguide_sim():
    return Simulation(freq_max=10e9, domain=(.02, .02, .02), dx=.001, boundary="cpml", cpml_layers=2)


@pytest.mark.parametrize("label", ["x_position", "reference_plane", "probe_plane", "f0",
                                   "bandwidth", "amplitude"])
def test_waveguide_port_refuses_a_traced_scalar(label):
    sim = waveguide_sim()
    seen = {}

    def declare(t):
        seen["repr"] = repr(t)
        sim.add_waveguide_port(**{"x_position": .005, label: t})
        return t

    exc = raised(jax.make_jaxpr(declare), .005)
    assert_moved_around(exc, RfxRefusal, ValueError,
                        f"{label} must be a finite scalar, got ", seen["repr"], "")


@pytest.mark.parametrize("label", ["y_range", "z_range"])
def test_waveguide_port_refuses_a_traced_range(label):
    sim = waveguide_sim()
    seen = {}

    def declare(t):
        seen["repr"] = repr(t)
        sim.add_waveguide_port(**{"x_position": .005, label: (t, .015)})
        return t

    exc = raised(jax.make_jaxpr(declare), .005)
    assert_moved_around(exc, RfxRefusal, ValueError,
                        f"{label} must contain finite scalars, got (", seen["repr"], ", 0.015)")


@pytest.mark.parametrize("keyword,value,message", [
    ("mode", (1.0, 0), "mode must be a tuple of two integers, got (1.0, 0)"),
    ("n_modes", 0, "n_modes must be a positive integer, got 0"),
    ("n_freqs", 2.0, "n_freqs must be a positive integer, got 2.0"),
])
def test_waveguide_port_refuses_a_non_integer_count(keyword, value, message):
    sim = waveguide_sim()
    exc = raised(lambda: sim.add_waveguide_port(.005, **{keyword: value}))
    assert_moved(exc, RfxRefusal, ValueError, message)


def test_floquet_port_refuses_a_graded_z_profile():
    sim = Simulation(freq_max=10e9, domain=(.015, .015, .060), dx=.0015,
                     dz_profile=np.array([.001] * 20 + [.002] * 20), boundary="pec")
    exc = raised(lambda: sim.add_floquet_port(.015, axis="z", f0=10e9, freqs=np.array([10e9]),
                                              name="lateral"))
    assert_moved(exc, RfxRefusal, ValueError,
                 "Floquet ports do not support non-uniform z mesh (dz_profile). "
                 "Set dx explicitly to prevent auto-mesh from creating NU grid.")


@pytest.mark.parametrize("call,message", [
    (lambda: drude_pole_map(lambda x, y, z: x, 1e9, density=1.0),
     "drude_pole_map density requires a concrete reference omega_p"),
    (lambda: lorentz_pole_map(1.0, 0.0, 1.0, relative=True),
     "relative lorentz_pole_map requires concrete omega_0 > 0; use drude_pole_map for omega_0=0"),
    (lambda: lorentz_pole(1.0, 0.0, 1.0, relative=True),
     "relative lorentz_pole requires concrete omega_0 > 0; use drude_pole for omega_0=0"),
    (lambda: RelativeStrength(1, 1.0),
     "RelativeStrength reference must be a positive finite Python/numpy float"),
], ids=["drude-map-density", "lorentz-map-relative", "lorentz-pole-relative", "relative-strength"])
def test_pole_declarations_refuse_a_non_concrete_reference(call, message):
    assert_moved(raised(call), RfxRefusal, ValueError, message)


SHEET = Box((0.0, 0.0, 0.002), (0.004, 0.004, 0.002))


def test_pec_thin_conductor_refuses_a_permittivity():
    exc = raised(lambda: ThinConductor(SHEET, 1e8, 35e-6, eps_r=2.0))
    assert_moved(exc, RfxRefusal, ValueError,
                 "add_thin_conductor sheet in PEC / surface-impedance mode: eps_r=2.0 is not "
                 "modelled; the sheet's update uses the permittivity of the cells it lies in. "
                 "Declare the dielectric around it with add(..., material=...); leave eps_r at 1.")


def test_thin_conductor_class_refuses_an_abstract_conductivity():
    exc = raised(jax.make_jaxpr(lambda sigma: ThinConductor(SHEET, sigma, 35e-6).is_pec), 1e8)
    assert_moved(exc, RfxRefusal, ValueError,
                 "sigma_bulk: the PEC/lossy class cannot be decided for an abstract value; "
                 "keep sigma_bulk concrete or differentiate without jit.")


def test_pinned_sheet_refuses_a_surface_impedance():
    exc = raised(lambda: PinnedSheet(normal_axis=2, plane_index=3, i_range=(1, 4), j_range=(1, 4),
                                     surface_impedance_f0=1e9))
    assert_moved(exc, RfxRefusal, ValueError,
                 "PinnedSheet: surface_impedance_f0 (Leontovich loss) is not supported on the "
                 "node-pinned declaration — its sheet conductivity is folded per cell from the "
                 "local mesh spacing. Declare that sheet with add_thin_conductor(shape, "
                 "surface_impedance_f0=...) on a pinned mesh.")


def test_pinned_sheet_refuses_a_permittivity():
    exc = raised(lambda: PinnedSheet(normal_axis=2, plane_index=3, i_range=(1, 4), j_range=(1, 4),
                                     eps_r=2.0))
    assert_moved(exc, RfxRefusal, ValueError,
                 "add_pinned_sheet: eps_r=2.0 is not modelled; the sheet's update uses the "
                 "permittivity of the cells it lies in. Declare the dielectric around it with "
                 "add(..., material=...); leave eps_r at 1.")


@pytest.mark.parametrize("axis,message", [
    (0, "dx_profile is a JAX tracer; provide a concrete domain[0] "
        "(profile sum cannot be host-coerced)."),
    (1, "dy_profile is a JAX tracer; provide a concrete domain[1] "
        "(profile sum cannot be host-coerced)."),
    (2, "dz_profile is a JAX tracer; provide a concrete domain=(..., ..., dz_total) extent "
        "(the profile sum cannot be host-coerced during tracing)."),
])
def test_graded_admission_refuses_a_traced_profile_without_its_extent(axis, message):
    def admit(profile):
        domain = [.01, .01, .01]
        domain[axis] = 0.0
        profiles = [None, None, None]
        profiles[axis] = profile
        admit_graded_profiles(tuple(domain), *profiles)
        return profile

    exc = raised(jax.make_jaxpr(admit), jnp.full(10, 1e-3))
    assert_moved(exc, RfxRefusal, ValueError, message)


def test_nominal_profile_refuses_traced_widths():
    exc = raised(jax.make_jaxpr(lambda p: admit_nominal_profiles({"x": p}, (p, None, None))),
                 jnp.full(4, 1e-3))
    assert_moved(exc, RfxRefusal, ValueError,
                 "nominal_profiles['x'] is a JAX tracer; declare the concrete x cell widths "
                 "at the primal point.")


def test_traced_window_refuses_a_float_index():
    exc = raised(admit_traced_windows, {"x": NominalProfile(cells=(1e-3,) * 4, traced_cells=(1.0, 2))})
    assert_moved(exc, RfxRefusal, ValueError,
                 "nominal_profiles['x'].traced_cells must be a 2-tuple of Python ints (not bool), "
                 "or a non-empty tuple of them.")


def test_pinned_time_step_refuses_a_traced_step():
    exc = raised(jax.make_jaxpr(lambda dt: make_nonuniform_grid(
        (.016, .016), np.full(16, D), D, 0, dx_profile=NOMINAL, dy_profile=NOMINAL, dt=dt).dt),
        1e-12)
    assert_moved(exc, RfxRefusal, ValueError,
                 "make_nonuniform_grid(dt=...) must be a concrete float; a traced dt puts the "
                 "time step itself into the derivative, which is the thing pinning it is meant "
                 "to remove.")


def test_pinned_time_step_on_a_traced_profile_needs_its_floor():
    exc = in_traced_grid(lambda profile: make_nonuniform_grid(
        (.016, .016), np.full(16, D), D, 0, dx_profile=profile, dy_profile=NOMINAL, dt=1e-12))
    assert_moved(exc, RfxRefusal, ValueError,
                 "make_nonuniform_grid(dt=1.000000e-12) with a traced x profile also needs "
                 "dt_min_cell=: a tracer carries no host cell size, so the Courant limit of the "
                 "realized mesh cannot be measured here. Pass the smallest cell the traced axes "
                 "reach anywhere in the family you will run.")


def test_pinned_time_step_refuses_a_negative_floor():
    exc = raised(lambda: make_nonuniform_grid((.016, .016), np.full(16, D), D, 0, dx_profile=NOMINAL,
                                              dy_profile=NOMINAL, dt=1e-12, dt_min_cell=-1.0))
    assert_moved(exc, RfxRefusal, ValueError,
                 "make_nonuniform_grid(dt_min_cell=...) must be a concrete positive float (m), "
                 "got -1.0.")


def test_index_of_refuses_a_traced_axis():
    exc = in_traced_grid(lambda profile: traced_graded_grid(profile).index_of("x", .007))
    assert_moved(exc, RfxRefusal, ValueError,
                 "index_of: axis 'x' carries a TRACED cell profile (mesh-as-design-variable), "
                 "so a physical coordinate cannot be resolved to an index on the host. The nominal "
                 "uniform fallback the old position_to_index used returns an index for a mesh the "
                 "solver is not running (gap G9). Resolve structural positions before tracing the "
                 "mesh, or pass the index directly.")


def test_sheet_plane_refuses_a_traced_node_line():
    nodes = np.arange(5) * 1e-3

    def place(x):
        coords = GridCoords(x=x, y=nodes, z=nodes, shape=(5, 5, 5))
        sheet_spec_from_shape(SHEET, coords, name="s")
        return x

    exc = raised(jax.make_jaxpr(place), jnp.asarray(nodes))
    assert_moved(exc, RfxRefusal, ValueError,
                 "sheet 's': a sheet's plane is a static integer, and this mesh is a JAX tracer "
                 "(mesh-as-design-variable), so the nearest node plane cannot be resolved. Pin the "
                 "mesh, or declare the conductor as a volume.")


def test_dual_average_interface_refuses_a_traced_profile():
    def assemble(dz):
        sim = Simulation(freq_max=20e9, domain=(6 * MM, 6 * MM, float(np.sum(DZ))), dx=0.5 * MM,
                         boundary="pec", dz_profile=dz, interface_eps="dual_average")
        assemble_interface_eps_nu(sim, None, None, pec_mask=None, high_wall=(False, False, False))
        return jnp.sum(dz)

    exc = raised(jax.grad(assemble), jnp.asarray(DZ))
    assert_moved(exc, RfxRefusal, ValueError,
                 "interface_eps='dual_average' requires concrete profiles, not traced profiles")


def test_exchange_interval_refuses_two():
    assert_moved(raised(validate_exchange_interval, 2), RfxRefusal, ValueError,
                 "exchange_interval=2 is refused: each skipped exchange updates seam cells from "
                 "stale neighbour values and the field grows exponentially in a lossless box; "
                 "use exchange_interval=1.")


def test_progress_reporting_refuses_a_traced_interval():
    exc = raised(jax.make_jaxpr(lambda every: validate_report_every(every, n_steps=10)), 5)
    assert_moved(exc, RfxRefusal, ValueError,
                 "report_every is a host-side progress feature and cannot run under jax.jit / "
                 "jax.grad / jax.vmap: the chunk loop reads the host wall clock and prints, so a "
                 "traced call would emit one line at TRACE time with a fabricated elapsed and rate. "
                 "Drop report_every on the differentiable path (forward() / optimize() / "
                 "eps_override=...), or call the eager entry point (run() / run_until_decay()) "
                 "outside the transform.")


def test_incident_source_refuses_a_traced_scalar():
    exc = raised(jax.make_jaxpr(lambda value: source_scalar("amplitude", value)), 1.0)
    assert_moved(exc, RfxRefusal, ValueError,
                 "amplitude: differentiating with respect to the incident source is not supported")


def test_oriented_box_refuses_a_traced_centre():
    exc = raised(jax.make_jaxpr(lambda t: OrientedBox((t, 0.0, 0.0), (1e-3, 1e-3, 1e-3)).center[1]), 0.0)
    assert_moved(exc, RfxUnsupported, NotImplementedError,
                 "OrientedBox requires concrete geometry parameters")


def test_forward_refuses_a_shape_dimension_gradient():
    def objective(radius):
        sim = Simulation(freq_max=20e9, domain=(.008,) * 3, dx=.001, cpml_layers=4,
                         dz_profile=.001 * np.array([1, 1, 1.5, 1.5, 1, 1, 1]))
        sim.add_material('d', eps_r=4.)
        sim.add(Sphere((.0041, .0038, .0043), radius), material='d')
        sim.add(Box((.001, .001, .001), (.006, .002, .002)), material='d')
        sim.add_source((.0021, .0022, .0023), 'ez', waveform=GaussianPulse(f0=10e9, bandwidth=.8))
        sim.add_probe((.0052, .0041, .0044), 'ez')
        result = sim.forward(n_steps=20, skip_preflight=True, subpixel_smoothing='full_tensor')
        return jnp.sum(result.time_series ** 2)

    exc = raised(jax.grad(objective), .0023)
    assert_moved(exc, RfxUnsupported, NotImplementedError,
                 "A gradient with respect to a declared shape dimension (a size or position of a "
                 "shape added with Simulation.add) is not supported: run() and forward() draw "
                 "shapes onto grid cells, so the result is a step function of that dimension and "
                 "its gradient would be zero almost everywhere. Differentiate a material value "
                 "through eps_override, or use a design region (design_box / design_eps_override) "
                 "for shape optimisation. jax.jit and jax.vmap over a dielectric Sphere or Cylinder dimension "
                 "are supported (with subpixel_smoothing only in a closed PEC domain).")


def test_wire_port_refuses_a_traced_radius():
    sim = Simulation(freq_max=10e9, domain=(.02, .02, .02), dx=.001, boundary="pec")

    def declare(t):
        sim.add_port((.01, .01, .01), extent=.003, radius=t)
        return t

    exc = raised(jax.make_jaxpr(declare), .0001)
    assert_moved(exc, RfxUnsupported, NotImplementedError,
                 "wire-port radius must be a static declaration")


def test_forward_refuses_rlc_values_off_the_uniform_lane():
    sim = Simulation(freq_max=20e9, domain=(.008,) * 3, dx=.001, cpml_layers=4,
                     dz_profile=.001 * np.array([1, 1, 1.5, 1.5, 1, 1, 1]))
    sim.add_source((.0021, .0022, .0023), 'ez', waveform=GaussianPulse(f0=10e9, bandwidth=.8))
    sim.add_probe((.0052, .0041, .0044), 'ez')
    exc = raised(lambda: sim.forward(n_steps=2, skip_preflight=True,
                                     rlc_values_override={0: {"R": 50.0}}))
    assert_moved(exc, RfxUnsupported, NotImplementedError,
                 "forward(rlc_values_override=...) is only supported on the uniform single-device "
                 "forward lane, not 'fwd_nonuniform'. Lumped RLC component-value gradients require "
                 "the uniform mesh path (non-uniform / distributed forward do not process "
                 "add_lumped_rlc elements).")


def test_adjoint_forward_refuses_a_call_without_a_design_box():
    sim = Simulation(freq_max=6e9, domain=(.006,) * 3, dx=.0005, boundary="pec")
    exc = raised(lambda: sim.forward(n_steps=4, gradient="adjoint", skip_preflight=True))
    assert_moved(exc, RfxUnsupported, NotImplementedError,
                 "gradient='adjoint' does not support calls without design_box and "
                 "design_eps_override")


def test_graded_periodic_axis_refuses_a_varying_profile():
    sim = Simulation(10e9, (12 * D,) * 3, dx=D, cpml_layers=0,
                     boundary=BoundarySpec(x='periodic', y='pec', z='pec'),
                     dx_profile=PERIODIC_FACTORS * D)
    assert_moved(raised(admit_graded_periodic, sim), RfxUnsupported, NotImplementedError,
                 "Graded periodic axis 'x' requires a constant profile; use a constant spacing "
                 "along a periodic axis")


@pytest.mark.parametrize("kind", ["closed_box", "custom"])
def test_run_refuses_an_extended_plane_wave_under_a_transform(kind):
    sim = Simulation(freq_max=30e9, domain=(0.018, 0.016, 0.014), dx=0.001,
                     boundary="cpml", cpml_layers=4)
    waveform = "differentiated_gaussian"
    if kind == "custom":
        waveform = CustomWaveform(lambda t: jnp.sin(2 * jnp.pi * 15e9 * t))
    sim.add_tfsf_source(f0=15e9, bandwidth=0.8, polarization="ez", direction="+x", margin=3,
                        waveform=waveform, closed_box=kind == "closed_box")
    sim.add_probe((0.003, 0.005, 0.004), component="ez")

    def trace(eps):
        return jnp.sum(sim.run(n_steps=8, skip_preflight=True).time_series ** 2) * eps

    assert_moved(raised(jax.grad(trace), 1.0), RfxUnsupported, NotImplementedError,
                 "CustomWaveform and closed_box TFSF cannot be passed through run() under external "
                 "JAX transformations; differentiate forward() instead.")


def test_kept_pad_reach_refusal_on_a_traced_shape_bound():
    exc = raised(jax.make_jaxpr(
        lambda t: _reaches_pad_face(((t, 0.0, 0.0), (1.0, 1.0, 1.0)), 0, "lo", 0.0, 1.0, None)), 0.0)
    assert_kept(exc, "jax.jit or jax.vmap over a declared shape dimension is not supported with "
                     "subpixel_smoothing in a domain with an absorbing boundary: whether the shape "
                     "reaches the absorber, and is continued into it, depends on that dimension. "
                     "Call run()/forward() with a concrete shape, run without subpixel_smoothing, "
                     "or use a closed PEC domain, which has no absorber.")


def test_kept_polyline_wire_refusal_on_traced_cell_sizes():
    exc = raised(jax.make_jaxpr(
        lambda sizes: _static_wire_min_cell(None, (None, None, None),
                                            (sizes, np.ones(3), np.ones(3)))), jnp.ones(3))
    assert_kept(exc, "PEC PolylineWire radius requires static lattice metrics: x cell sizes are "
                     "traced and no nominal untraced profile is available to decide "
                     "0 < a < 0.5*d_min.")


def test_kept_periodic_film_refusal_on_a_traced_graded_mesh():
    dz = np.array([1e-3] * 3 + [.5e-3] * 4 + [1e-3] * 3, dtype=np.float32)

    def fold(profile):
        sim = Simulation(20e9, (.008, .010, float(np.sum(dz.astype(np.float64)))), dx=1e-3,
                         cpml_layers=0, boundary=BoundarySpec(x="pec", y="periodic", z="pec"),
                         snap="declared", dz_profile=profile)
        sim.add_thin_conductor(Box((.002, .005, .004), (.006, .010, .004)), sigma_bulk=1e3,
                               thickness=1e-4)
        grid = sim._build_nonuniform_grid()
        return sim._assemble_materials_nu(grid, pec_sheets=[], pec_wires=[])[0].sigma_film

    exc = raised(jax.jit(fold), jnp.asarray(dz))
    assert_kept(exc, "a lossy thin-conductor film on a traced graded mesh with periodic axes 'y' is "
                     "not implemented (its footprint would get no seam image); use a concrete mesh "
                     "profile")


def test_kept_msl_plane_refusal_on_a_traced_mesh():
    exc = in_traced_grid(lambda profile: _msl_grid_geometry(traced_graded_grid(profile)))
    assert_kept(exc, "MSL conductor-plane validation requires concrete mesh coordinates")


def test_kept_wire_radius_refusal_without_a_nominal_profile():
    def stamp(profile):
        grid = traced_graded_grid(profile)
        stamp_wire_radius(grid, init_materials(grid.shape), "ez", .0001, [CELL])

    exc = in_traced_grid(stamp)
    assert_kept(exc, "wire-port radius does not support traced mesh metrics; a nominal profile "
                     "with traced_cells is needed")


def test_kept_wire_radius_refusal_on_an_overlapping_window():
    window = (6, 7)

    def objective(profile):
        grid = traced_graded_grid(profile, {"x": NominalProfile(NOMINAL, window)})
        stamp_wire_radius(grid, init_materials(grid.shape), "ez", .0001, [CELL])
        return jnp.sum(profile)

    exc = raised(jax.grad(objective), jnp.asarray(NOMINAL))
    assert_kept(exc, f"wire-port radius at port cell {CELL}, axis x reads stencil cells 5..8 "
                     f"(padded); declared traced range {window} (half-open interior) overlaps the "
                     "stencil or its margin. A pin inside or next to a moving range needs the "
                     "off-square Hankel bound and AD-vs-FD work, which is not implemented.")


class _TracedCells:
    """A grid stand-in whose per-cell spacings are the traced array."""

    def __init__(self, cells):
        self.dx_arr = self.dy_arr = self.dz = cells


def test_kept_port_metric_refusal_on_traced_spacings():
    exc = raised(jax.make_jaxpr(lambda cells: _nu_cell_arrays(_TracedCells(cells))[0]), jnp.ones(4))
    assert_kept(exc, "port metrics (port_sigma / port_dual_transverse) need CONCRETE per-cell mesh "
                     "spacings: the realized conductance of a stamped sigma is fixed by the E node's "
                     "discrete Ampere control volume, which is built from the cell sizes AT the port "
                     "node. This grid carries a TRACED profile (mesh-as-design-variable), which is "
                     "not supported on this path. Differentiate with respect to the port/material "
                     "values on a concrete mesh instead.\n\nBefore this check, the path did not "
                     "raise: it silently used the BOUNDARY cell size for the two transverse axes and "
                     "returned a gradient of exactly 0.0 w.r.t. the mesh. Raising is intentional.")
