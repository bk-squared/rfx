"""Lane admission: a lane solves only the inputs it declares it carries.

A ``Simulation`` is a declaration: materials, conductors, ports, sources,
boundaries, a mesh and observers. Nine time-stepping lanes run it, and each
one solves only part of it. Until this module each lane listed the inputs it
refused and solved every other one as if it had not been declared: ADI solved
a μr = 4 block as vacuum, the graded ``run()`` solved a Kerr block as linear,
a Floquet port on a graded mesh launched nothing. Here each lane lists what
it carries, and :func:`admit` refuses every other declared input before the
first time step (the PI's rule of 2026-09-24: a physics input a path does not
implement is refused).

``DETECTORS`` says, for every input row of the path-disposition table
(``tests/contracts/path_disposition.py``), whether a declaration switches that
input on. A detector reads only the static declaration: it never reads a
traced array, and a value that is traced (a design permittivity, say) counts
as declared. For a setting, on means a value other than the constructor's
default. ``ADMITS`` says which rows each lane admits: the rows whose table
cell on that lane is ``carries`` or ``ignorable``. The table keeps the notes
and the open issues; ``tests/contracts/test_path_disposition.py`` fails when
its cell kinds and ``ADMITS`` disagree.

Each lane calls :func:`admit` at the last point before its kernel, after its
own specific refusals (#1240, #1241, #1297 and the rest), so those keep their
messages and admission answers only for what they do not name. The lane
checked is the lane that runs: a multi-device ``run()`` that falls back to one
device is judged on the uniform lane it falls back to.
"""

from __future__ import annotations

import functools
import inspect
from typing import Callable

import numpy as np

from rfx.core.jax_utils import is_tracer

Row = tuple[str, str]

LANES = (
    "run_uniform", "run_nonuniform", "run_subgridded", "run_adi",
    "run_distributed", "fwd_uniform", "fwd_nonuniform", "fwd_distributed_nu",
    "fwd_adi",
)

LANE_WORDS = {
    "run_uniform": "uniform run()",
    "run_nonuniform": "graded run()",
    "run_subgridded": "subgridded run()",
    "run_adi": "ADI run()",
    "run_distributed": "multi-device run(devices=...)",
    "fwd_uniform": "uniform forward()",
    "fwd_nonuniform": "graded forward()",
    "fwd_distributed_nu": "forward(distributed=True)",
    "fwd_adi": "ADI forward()",
}

_PEC_SIGMA = 1e6   # Simulation._PEC_SIGMA_THRESHOLD


# ----------------------------------------------------------------- helpers

@functools.lru_cache(maxsize=None)
def _defaults() -> dict:
    """The defaults of rfx's own ``Simulation`` constructor, read once. Not
    ``type(sim)``'s: a subclass may take ``**kwargs``."""
    from rfx.api import Simulation
    return {name: p.default for name, p in inspect.signature(Simulation.__init__).parameters.items()
            if p.default is not inspect.Parameter.empty}


def _differs(value, default) -> bool:
    """Whether a declared value is not ``default``; a traced value is declared."""
    if value is None or default is None:
        return value is not default
    if is_tracer(value):
        return True
    if isinstance(value, str) or isinstance(default, str):
        return value != default
    return bool(np.any(np.asarray(value) != default))


def _setting(attr: str, arg: str) -> Callable:
    def active(sim) -> bool:
        return _differs(getattr(sim, attr), _defaults()[arg])
    return active


def _is_pec(sigma) -> bool:
    return not is_tracer(sigma) and float(sigma) >= _PEC_SIGMA


def _placed(sim):
    """The materials the geometry places, PEC excluded."""
    for entry in sim._geometry:
        material = sim._resolve_material(entry.material_name)
        if not _is_pec(material.sigma):
            yield material


def _pec_shapes(sim):
    for entry in sim._geometry:
        if _is_pec(sim._resolve_material(entry.material_name).sigma):
            yield entry.shape


def _wire_lattice(sim):
    """The grid the lanes build, with the node axes and cell sizes the
    assembler classifies a wire against, or None when the mesh is traced."""
    from rfx.geometry.rasterize_grid import (
        cell_sizes_from_nonuniform_grid, cell_sizes_from_uniform_grid,
        coords_from_nonuniform_grid, coords_from_uniform_grid)
    grid = sim._build_realized_grid()
    if hasattr(grid, "dx_arr"):
        coords, sizes = coords_from_nonuniform_grid(grid), cell_sizes_from_nonuniform_grid(grid)
    else:
        coords, sizes = coords_from_uniform_grid(grid), cell_sizes_from_uniform_grid(grid)
    axes = (coords.x, coords.y, coords.z)
    if any(is_tracer(a) for a in (*axes, *sizes)):
        return None
    return grid, axes, sizes


def _pec_kind(sim, shape, lattice) -> str:
    """How the assembler realizes a PEC shape (``classify_pec_entry``): a
    Box with one zero-extent axis is a sheet, anything else but a wire a
    volume. A PolylineWire is judged by the assembler's own rule,
    ``wire_filament_nodes``, after the same conductor continuation, on the
    grid the lanes build."""
    lo, hi = getattr(shape, "corner_lo", None), getattr(shape, "corner_hi", None)
    if lo is not None and hi is not None:
        if not (is_tracer(lo) or is_tracer(hi)) and sum(
                float(a) == float(b) for a, b in zip(lo, hi)) == 1:
            return "pec_sheet"
        return "pec_volume"
    radius = getattr(shape, "radius", None)
    if (getattr(shape, "points", None) is None or radius is None or is_tracer(radius)
            or lattice is None):
        return "pec_volume"
    from rfx.geometry.rasterize_grid import wire_filament_nodes
    from rfx.geometry.smoothing import continued_conductor_shape
    grid, axes, sizes = lattice
    solved = continued_conductor_shape(sim, grid, shape, unextendable=[])
    if getattr(solved, "points", None) is None or getattr(solved, "radius", None) is None:
        return "pec_volume"   # continued into a shape the assembler takes as a volume
    filament = wire_filament_nodes(solved.points, solved.radius, axes, sizes, grid=grid)
    return "pec_wire" if filament is not None else "pec_volume"


def _pec(kind: str) -> Callable:
    def active(sim) -> bool:
        shapes = list(_pec_shapes(sim))
        if not shapes:
            return False
        wires = any(getattr(shape, "points", None) is not None for shape in shapes)
        lattice = _wire_lattice(sim) if wires else None
        return any(_pec_kind(sim, shape, lattice) == kind for shape in shapes)
    return active


def _lorentz_like(pole, *, drude: bool) -> bool:
    omega_0 = pole.omega_0
    if is_tracer(omega_0):
        return not drude
    return (float(omega_0) == 0.0) == drude


def _thin(conductor) -> str:
    if conductor.surface_impedance_f0 is not None:
        return "surface_impedance"
    return "pec_sheet" if _is_pec(conductor.sigma_bulk) else "lossy_sheet"


def _soft(port) -> bool:
    return not is_tracer(port.impedance) and float(port.impedance) == 0.0


def _ports(sim, *, extent: bool | None = None, excite: bool | None = None):
    return [p for p in sim._ports if not _soft(p)
            and (extent is None or (p.extent is not None) == extent)
            and (excite is None or bool(p.excite) == excite)]


def _series_with_inductor(element) -> bool:
    return element.topology == "series" and _differs(element.L, 0.0)


def _absorber(sim) -> bool:
    return sim._cpml_layers > 0


def _faces(sim) -> dict:
    return {f"{axis}_{side}": token for axis, side, token in sim._boundary_spec.faces()}


def _absorbing_lid(sim) -> bool:
    """A closed PEC box whose one absorbing face is a z face, opposite a PEC z face."""
    faces = _faces(sim)
    z = {faces["z_lo"], faces["z_hi"]}
    return (_absorber(sim) and all(faces[f] == "pec" for f in ("x_lo", "x_hi", "y_lo", "y_hi"))
            and "pec" in z and len(z & {"cpml", "upml"}) == 1)


def _guarded_lid(sim, grid) -> bool:
    """Whether production subgrid validation accepts this box's absorber: the
    refined slab touches a PEC z face, the opposite z face may absorb, the
    x/y faces are closed PEC, and the slab stays clear of the absorber. The
    validator's own checks decide, from the coarse grid the subgridded lane
    builds."""
    refinement = sim._refinement
    if refinement is None or sim._uses_nonuniform_mesh:
        return False
    from rfx.subgridding.validation import (
        _guarded_boundary_production_allowed, _one_sided_physical_z_boundary,
        _slab_overlaps_absorber, build_subgrid_region)
    grid = sim._build_grid() if grid is None else grid
    region = build_subgrid_region(sim, grid)
    if region is None:
        return False
    return (bool(_guarded_boundary_production_allowed(
                sim, grid, _one_sided_physical_z_boundary(sim, region, grid),
                refinement.get("xy_margin")))
            and not _slab_overlaps_absorber(sim, region, grid))


# Inputs that production subgrid validation refuses and 'research'/'off'
# run with the input dropped (#1286).
_RELAXED_DROPS = (("_materials", "debye"), ("_materials", "lorentz"),
                  ("_materials", "drude"), ("_materials", "kerr"),
                  ("_lumped_rlc", "R"), ("_lumped_rlc", "series_RL"))


def _relaxed_validation(sim) -> bool:
    refinement = sim._refinement
    if refinement is None or refinement.get("validation", "production") == "production":
        return False
    return any(DETECTORS[row](sim) for row in _RELAXED_DROPS)


def _s_matrix_ports(sim) -> bool:
    """run()'s lumped/wire S-matrix comes from the scan driver
    (``_forward_from_materials``) for one lumped port or two wire ports."""
    return bool(_ports(sim, extent=False)) or len(_ports(sim, extent=True)) >= 2


# --------------------------------------------------------------- detectors

DETECTORS: dict[Row, Callable] = {
    # settings: a value other than the constructor's default
    ("_freq_max", ""): lambda sim: True,
    ("_domain", ""): lambda sim: True,
    ("_dx", ""): _setting("_dx", "dx"),
    ("_dt_pin", ""): _setting("_dt_pin", "dt"),
    ("_dt_min_cell", ""): _setting("_dt_min_cell", "dt_min_cell"),
    ("_precision", ""): _setting("_precision", "precision"),
    ("_solver", ""): _setting("_solver", "solver"),
    ("_adi_cfl_factor", ""): _setting("_adi_cfl_factor", "adi_cfl_factor"),
    ("_stencil_order", ""): _setting("_stencil_order", "stencil_order"),
    ("_mode", ""): _setting("_mode", "mode"),
    # materials the geometry places
    ("_materials", "eps"): lambda sim: any(_differs(m.eps_r, 1.0) for m in _placed(sim)),
    ("_materials", "sigma"): lambda sim: any(_differs(m.sigma, 0.0) for m in _placed(sim)),
    ("_materials", "mu"): lambda sim: any(_differs(m.mu_r, 1.0) for m in _placed(sim)),
    ("_materials", "debye"): lambda sim: any(m.debye_poles for m in _placed(sim)),
    ("_materials", "lorentz"): lambda sim: any(
        _lorentz_like(p, drude=False) for m in _placed(sim) for p in (m.lorentz_poles or ())),
    ("_materials", "drude"): lambda sim: any(
        _lorentz_like(p, drude=True) for m in _placed(sim) for p in (m.lorentz_poles or ())),
    ("_materials", "kerr"): lambda sim: any(_differs(m.chi3, 0.0) for m in _placed(sim)),
    # PEC geometry
    ("_geometry", "pec_volume"): _pec("pec_volume"),
    ("_geometry", "pec_sheet"): _pec("pec_sheet"),
    ("_geometry", "pec_wire"): _pec("pec_wire"),
    ("_thin_conductors", "lossy_sheet"): lambda sim: any(
        _thin(c) == "lossy_sheet" for c in sim._thin_conductors),
    ("_thin_conductors", "pec_sheet"): lambda sim: any(
        _thin(c) == "pec_sheet" for c in sim._thin_conductors),
    ("_thin_conductors", "surface_impedance"): lambda sim: any(
        _thin(c) == "surface_impedance" for c in sim._thin_conductors),
    ("_pinned_sheets", "pec_sheet"): lambda sim: bool(sim._pinned_sheets),
    # ports and sources
    ("_ports", "source"): lambda sim: any(_soft(p) for p in sim._ports),
    ("_ports", "amplitude_kind"): lambda sim: any(
        _soft(p) and p.amplitude_kind == "current" for p in sim._ports),
    ("_ports", "lumped_port"): lambda sim: bool(_ports(sim, extent=False, excite=True)),
    ("_ports", "passive_port"): lambda sim: bool(_ports(sim, excite=False)),
    ("_ports", "wire_port"): lambda sim: bool(_ports(sim, extent=True)),
    ("_msl_ports", "msl_port"): lambda sim: bool(sim._msl_ports),
    ("_waveguide_ports", "waveguide_port"): lambda sim: bool(sim._waveguide_ports),
    ("_coaxial_ports", "coax_port"): lambda sim: bool(sim._coaxial_ports),
    ("_floquet_ports", "floquet_port"): lambda sim: bool(sim._floquet_ports),
    ("_floquet_ports", "scan_angle"): lambda sim: any(
        _differs(p.scan_theta, 0.0) for p in sim._floquet_ports),
    ("_lumped_rlc", "R"): lambda sim: any(
        not _series_with_inductor(e) for e in sim._lumped_rlc),
    ("_lumped_rlc", "series_RL"): lambda sim: any(
        _series_with_inductor(e) for e in sim._lumped_rlc),
    ("_tfsf", "plane_wave"): lambda sim: sim._tfsf is not None,
    ("_refinement", "slab"): lambda sim: sim._refinement is not None,
    ("_refinement", "relaxed_validation"): _relaxed_validation,
    # boundaries
    ("_boundary", "cpml"): lambda sim: sim._boundary == "cpml" and _absorber(sim),
    ("_boundary", "upml"): lambda sim: sim._boundary == "upml" and _absorber(sim),
    ("_pec_faces", "pec_face"): lambda sim: bool(sim._pec_faces) and _absorber(sim),
    ("_boundary_spec", "pmc_face"): lambda sim: bool(sim._boundary_spec.pmc_faces()),
    ("_boundary_spec", "conformal"): lambda sim: bool(sim._boundary_spec.conformal_faces()),
    ("_boundary_spec", "conformal_s_matrix"): lambda sim: (
        bool(sim._boundary_spec.conformal_faces()) and _s_matrix_ports(sim)),
    ("_boundary_spec", "absorbing_lid"): _absorbing_lid,
    ("_periodic_axes", "periodic"): lambda sim: bool(sim._periodic_axes),
    ("_cpml_layers", "layers"): _absorber,
    ("_cpml_kappa_max", "kappa"): lambda sim: _absorber(sim) and _differs(
        sim._cpml_kappa_max, _defaults()["cpml_kappa_max"]),
    ("_interface_eps", "dual_average"): _setting("_interface_eps", "interface_eps"),
    # mesh profiles, as declared
    ("_dx_profile", "graded"): lambda sim: sim._dx_profile is not None,
    ("_dy_profile", "graded"): lambda sim: sim._dy_profile is not None,
    ("_dz_profile", "graded"): lambda sim: sim._dz_profile is not None,
    # observers
    ("_probes", "probe"): lambda sim: bool(sim._probes),
    ("_dft_planes", "dft_plane"): lambda sim: bool(sim._dft_planes),
    ("_flux_monitors", "flux"): lambda sim: bool(sim._flux_monitors),
    ("_ntff", "ntff_box"): lambda sim: sim._ntff is not None,
    ("_current_moments", "block_moments"): lambda sim: sim._current_moments is not None,
}

# What the error message calls each input.
ROW_WORDS: dict[Row, str] = {
    ("_freq_max", ""): "freq_max",
    ("_domain", ""): "the domain",
    ("_dx", ""): "an explicit dx",
    ("_dt_pin", ""): "a pinned time step (dt=)",
    ("_dt_min_cell", ""): "dt_min_cell=",
    ("_precision", ""): "a precision other than 'float32'",
    ("_solver", ""): "solver='adi'",
    ("_adi_cfl_factor", ""): "adi_cfl_factor",
    ("_stencil_order", ""): "stencil_order=4",
    ("_mode", ""): "a 2-D mode (mode='2d_tmz' or '2d_tez')",
    ("_materials", "eps"): "a dielectric (eps_r != 1)",
    ("_materials", "sigma"): "a conductive material (sigma > 0)",
    ("_materials", "mu"): "a magnetic material (mu_r != 1)",
    ("_materials", "debye"): "a Debye pole",
    ("_materials", "lorentz"): "a Lorentz pole",
    ("_materials", "drude"): "a Drude pole",
    ("_materials", "kerr"): "a Kerr χ³ material (chi3 != 0)",
    ("_geometry", "pec_volume"): "a PEC volume",
    ("_geometry", "pec_sheet"): "a PEC sheet (a zero-thickness PEC Box)",
    ("_geometry", "pec_wire"): "a sub-cell PEC wire (PolylineWire)",
    ("_thin_conductors", "lossy_sheet"): "a lossy thin conductor (add_thin_conductor)",
    ("_thin_conductors", "pec_sheet"): "a PEC thin conductor (add_thin_conductor)",
    ("_thin_conductors", "surface_impedance"): "a surface-impedance sheet (surface_impedance_f0)",
    ("_pinned_sheets", "pec_sheet"): "a pinned sheet (add_pinned_sheet)",
    ("_ports", "source"): "a soft source (add_source)",
    ("_ports", "amplitude_kind"): "a soft source with amplitude_kind='current'",
    ("_ports", "lumped_port"): "a driven lumped port (add_port)",
    ("_ports", "passive_port"): "a passive port (add_port(excite=False))",
    ("_ports", "wire_port"): "a wire port (add_port(extent=...))",
    ("_msl_ports", "msl_port"): "a microstrip port (add_msl_port)",
    ("_waveguide_ports", "waveguide_port"): "a waveguide port (add_waveguide_port)",
    ("_coaxial_ports", "coax_port"): "a coaxial port (add_coaxial_port)",
    ("_floquet_ports", "floquet_port"): "a Floquet port (add_floquet_port)",
    ("_floquet_ports", "scan_angle"): "a Floquet port scanned off normal (scan_theta != 0)",
    ("_lumped_rlc", "R"): "a lumped RLC element (add_lumped_rlc)",
    ("_lumped_rlc", "series_RL"): "a series lumped RLC element with an inductance",
    ("_tfsf", "plane_wave"): "a TFSF plane-wave source (add_tfsf_source)",
    ("_refinement", "slab"): "a subgrid refinement (add_refinement)",
    ("_refinement", "relaxed_validation"): (
        "validation='research'/'off' with a dispersive pole, Kerr χ³ or a lumped RLC element"),
    ("_boundary", "cpml"): "a CPML absorber",
    ("_boundary", "upml"): "a UPML absorber",
    ("_pec_faces", "pec_face"): "PEC faces on an absorbing box",
    ("_boundary_spec", "pmc_face"): "a PMC (magnetic wall) face",
    ("_boundary_spec", "conformal"): "Boundary(conformal=True) or conformal_pec=True",
    ("_boundary_spec", "conformal_s_matrix"): (
        "Boundary(conformal=True) or conformal_pec=True with a lumped/wire S-matrix"),
    ("_boundary_spec", "absorbing_lid"): "an absorbing z lid on a closed PEC box",
    ("_periodic_axes", "periodic"): "a periodic axis",
    ("_cpml_layers", "layers"): "an absorber thickness (cpml_layers)",
    ("_cpml_kappa_max", "kappa"): "cpml_kappa_max != 1",
    ("_interface_eps", "dual_average"): "interface_eps='dual_average'",
    ("_dx_profile", "graded"): "a dx_profile (graded x mesh)",
    ("_dy_profile", "graded"): "a dy_profile (graded y mesh)",
    ("_dz_profile", "graded"): "a dz_profile (graded z mesh)",
    ("_probes", "probe"): "a point probe",
    ("_dft_planes", "dft_plane"): "a DFT plane probe",
    ("_flux_monitors", "flux"): "a flux monitor",
    ("_ntff", "ntff_box"): "an NTFF box",
    ("_current_moments", "block_moments"): "a current-moment monitor",
}


# ------------------------------------------------------------------ admits
# Per input row, the lanes that admit it: those whose cell in the table is
# ``carries`` or ``ignorable``. ADMITS below is the same list read by lane.

_ALL = frozenset(LANES)
_ADI = frozenset({"run_adi", "fwd_adi"})
_GRADED = frozenset({"run_nonuniform", "fwd_nonuniform", "fwd_distributed_nu"})
_UNIFORM_YEE = frozenset({"run_uniform", "fwd_uniform"})
_MESH_PINNED = _GRADED | {"run_distributed"}   # the lanes a dx/dy/dz profile reaches
_NO_SHEETS = _ALL - {"run_subgridded", "run_adi", "run_distributed", "fwd_distributed_nu", "fwd_adi"}

_ADMITTED_ON: dict[Row, frozenset] = {
    ("_freq_max", ""): _ALL,
    ("_domain", ""): _ALL,
    ("_dx", ""): _ALL,
    ("_dt_pin", ""): _MESH_PINNED,
    ("_dt_min_cell", ""): _MESH_PINNED,
    ("_precision", ""): _UNIFORM_YEE,
    ("_solver", ""): _ADI,
    ("_adi_cfl_factor", ""): _ALL,
    ("_stencil_order", ""): _UNIFORM_YEE,
    ("_mode", ""): _UNIFORM_YEE | _ADI | {"run_distributed"},
    ("_materials", "eps"): _ALL,
    ("_materials", "sigma"): _ALL,
    ("_materials", "mu"): _ALL - _ADI,
    ("_materials", "debye"): _ALL - _ADI - {"run_subgridded"},
    ("_materials", "lorentz"): _ALL - _ADI - {"run_subgridded"},
    ("_materials", "drude"): _ALL - _ADI - {"run_subgridded"},
    ("_materials", "kerr"): _UNIFORM_YEE,
    ("_geometry", "pec_volume"): _ALL - _ADI,
    ("_geometry", "pec_sheet"): _NO_SHEETS,
    ("_geometry", "pec_wire"): _NO_SHEETS,
    ("_thin_conductors", "lossy_sheet"): _ALL - _ADI - {"run_subgridded"},
    ("_thin_conductors", "pec_sheet"): _NO_SHEETS,
    ("_thin_conductors", "surface_impedance"): _NO_SHEETS,
    ("_pinned_sheets", "pec_sheet"): _NO_SHEETS,
    ("_ports", "source"): _ALL,
    ("_ports", "amplitude_kind"): _ALL - _ADI,
    ("_ports", "lumped_port"): _ALL - _ADI - {"fwd_distributed_nu"},
    ("_ports", "passive_port"): _ALL - _ADI - {"run_distributed", "fwd_distributed_nu"},
    ("_ports", "wire_port"): _ALL - _ADI - {"run_distributed", "fwd_distributed_nu"},
    ("_msl_ports", "msl_port"): _NO_SHEETS,
    ("_waveguide_ports", "waveguide_port"): _NO_SHEETS,
    ("_coaxial_ports", "coax_port"): frozenset(),
    ("_floquet_ports", "floquet_port"): _UNIFORM_YEE,
    ("_floquet_ports", "scan_angle"): frozenset(),
    ("_lumped_rlc", "R"): _NO_SHEETS,
    ("_lumped_rlc", "series_RL"): _NO_SHEETS,
    ("_tfsf", "plane_wave"): frozenset({"run_uniform", "run_nonuniform", "fwd_uniform"}),
    ("_refinement", "slab"): frozenset({"run_subgridded"}),
    ("_refinement", "relaxed_validation"): frozenset(),
    ("_boundary", "cpml"): _ALL - {"run_subgridded"},
    ("_boundary", "upml"): _UNIFORM_YEE,
    ("_pec_faces", "pec_face"): _ALL - _ADI - {"run_subgridded"},
    ("_boundary_spec", "pmc_face"): _ALL - _ADI - {"run_subgridded"},
    ("_boundary_spec", "conformal"): frozenset({"run_uniform"}),
    ("_boundary_spec", "conformal_s_matrix"): frozenset(),
    ("_boundary_spec", "absorbing_lid"): _ALL - _ADI,
    ("_periodic_axes", "periodic"): _UNIFORM_YEE,
    ("_cpml_layers", "layers"): _ALL - {"run_subgridded"},
    ("_cpml_kappa_max", "kappa"): frozenset({"run_uniform", "run_distributed", "fwd_uniform"}),
    ("_interface_eps", "dual_average"): frozenset({"run_nonuniform", "fwd_nonuniform"}),
    ("_dx_profile", "graded"): _MESH_PINNED,
    ("_dy_profile", "graded"): _MESH_PINNED,
    ("_dz_profile", "graded"): _MESH_PINNED,
    ("_probes", "probe"): _ALL,
    ("_dft_planes", "dft_plane"): _NO_SHEETS,
    ("_flux_monitors", "flux"): frozenset({"run_uniform", "run_nonuniform", "fwd_uniform",
                                          "fwd_nonuniform", "fwd_adi"}),
    ("_ntff", "ntff_box"): _ALL - _ADI - {"fwd_distributed_nu"},
    ("_current_moments", "block_moments"): _NO_SHEETS,
}

ADMITS: dict[str, frozenset] = {
    lane: frozenset(row for row, lanes in _ADMITTED_ON.items() if lane in lanes)
    for lane in LANES
}


# A row the call decides as well as the declaration. run() computes the
# lumped/wire S-matrix, through a path with no conformal update (#1299), only
# when compute_s_params is not False (rfx/runners/uniform.py); called with
# compute_s_params=False it computes none, and the conformal walls it does
# carry are all there is. Called with conformal_pec=False, the override
# run() documents, the fields are staircase as well as the S-matrix, so
# nothing declared is dropped. The gate reads the call's static arguments,
# never a traced value, and a call that passes none is judged as the default
# call. An explicit conformal_pec=True on a model with no PEC to conform is a
# no-op on every lane (Simulation._has_pec_to_conform), so it asks for nothing.
def _conformal_requested(sim, run_args) -> bool:
    conformal = run_args.get("conformal_pec")
    if conformal is None:
        return bool(sim._boundary_spec.conformal_faces())
    return bool(conformal) and sim._has_pec_to_conform()


def _conformal_s_matrix_requested(sim, run_args) -> bool:
    return bool(_conformal_requested(sim, run_args) and _s_matrix_ports(sim)
                and run_args.get("compute_s_params") is not False)


CALL_GATES: dict[Row, Callable] = {
    ("_boundary_spec", "conformal"): _conformal_requested,
    ("_boundary_spec", "conformal_s_matrix"): _conformal_s_matrix_requested,
}


# Rows a lane admits for some declarations only, decided by that lane's own
# check. The subgridded lane's production validation accepts one absorbing z
# face, a lid opposite the PEC z face its refined slab touches, with closed
# PEC x/y faces; there the lane reads a CPML absorber, its thickness and
# kappa_max (#1355 review, measured). Every validation mode gets exactly that
# envelope: 'research' and 'off' do not widen it. A UPML lid is not gated: the
# lane runs CPML in its place, bit for bit (measured), so it stays refused.
# tests/contracts/path_disposition.py lists the same gates as LANE_GATES.
_LID_ROWS = (("_boundary", "cpml"), ("_pec_faces", "pec_face"),
             ("_cpml_layers", "layers"), ("_cpml_kappa_max", "kappa"))
LANE_GATES: dict[str, dict[Row, Callable]] = {
    "run_subgridded": {row: _guarded_lid for row in _LID_ROWS},
}


# ---------------------------------------------------------------- admission

def active(sim, run_args=None) -> list[Row]:
    """The input rows this declaration switches on, in table order, for a
    call with ``run_args``."""
    run_args = run_args or {}
    return [row for row, on in DETECTORS.items()
            if (CALL_GATES[row](sim, run_args) if row in CALL_GATES else on(sim))]


def refused(sim, lane: str, run_args=None, grid=None) -> list[Row]:
    """The declared inputs ``lane`` does not admit. ``grid`` is the grid the
    lane built, for its ``LANE_GATES``; without it a gate builds its own."""
    admits, gates, decided = ADMITS[lane], LANE_GATES.get(lane, {}), {}
    out = []
    for row in active(sim, run_args):
        if row in admits:
            continue
        gate = gates.get(row)
        if gate is not None:
            if gate not in decided:
                decided[gate] = gate(sim, grid)
            if decided[gate]:
                continue
        out.append(row)
    return out


# The rows that choose the lane: each is admitted only by its own family, so
# a lane that carries everything else is still named as an alternative.
LANE_SELECTORS = frozenset({("_solver", ""), ("_dx_profile", "graded"), ("_dy_profile", "graded"),
                            ("_dz_profile", "graded"), ("_refinement", "slab"), ("_dt_pin", ""),
                            ("_dt_min_cell", "")})


def message(lane: str, rows, sim, run_args=None) -> str:
    """The refusal, derived from the same table: each input the lane does not
    carry, and the lanes that carry every other input of this model."""
    lines = [f"  - {ROW_WORDS[row]} is not carried by the {LANE_WORDS[lane]} lane."
             for row in rows]
    carriers = [LANE_WORDS[other] for other in LANES if other != lane
                and not set(refused(sim, other, run_args)) - LANE_SELECTORS]
    where = ("Lanes that carry every input of this model apart from the ones that choose the "
             "lane (solver, mesh profiles, refinement): " + ", ".join(carriers) + "."
             if carriers else
             "No time-stepping lane carries every input of this model apart from the ones that "
             "choose the lane (solver, mesh profiles, refinement).")
    if ("_boundary_spec", "conformal_s_matrix") in rows:
        where += (" Use compute_s_params=False for conformal probe fields, or "
                  "conformal_pec=False for a staircase S-matrix.")
    return (f"The {LANE_WORDS[lane]} lane would solve this Simulation as if "
            f"{'these inputs were' if len(lines) > 1 else 'this input was'} not "
            "declared, so it is refused before the first time step:\n"
            + "\n".join(lines) + "\n" + where
            + "\nRemove the input, or declare the model so that a lane that "
            "carries it runs.")


def admit(sim, lane: str, *, run_args=None, grid=None) -> None:
    """Raise ``NotImplementedError``, as the lanes' own refusals do, if
    ``lane`` does not carry every input ``sim`` declares. ``run_args`` are
    the call's static arguments that ``CALL_GATES`` read; ``grid`` is the
    grid the lane built, for its ``LANE_GATES``."""
    from rfx.sources.wire_radius import require_radius_support
    require_radius_support(sim, lane)
    rows = refused(sim, lane, run_args, grid)
    if rows:
        raise NotImplementedError(message(lane, rows, sim, run_args))
