"""Lane admission: a lane solves only the inputs it declares it carries.

A ``Simulation`` is a declaration: materials, conductors, ports, sources,
boundaries, a mesh and observers. Ten time-stepping lanes run it, and each
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
checked is the lane that runs; unsupported multi-device sources are refused
before any step.
"""

from __future__ import annotations

import functools
import inspect
import threading
from dataclasses import fields, is_dataclass, replace
from typing import Callable

import jax
import numpy as np

from rfx.core.jax_utils import is_tracer

Row = tuple[str, str]

LANES = (
    "run_uniform", "run_nonuniform", "run_subgridded", "run_adi",
    "run_distributed", "run_distributed_nu", "fwd_distributed", "fwd_uniform", "fwd_nonuniform", "fwd_distributed_nu",
    "fwd_adi",
)

CALCULATORS = (
    "s_matrix_scan", "mixed_s_matrix", "topology_optimize",
    "waveguide_s_matrix", "coaxial_line_reflection", "coaxial_two_port",
    "coax_msl_transition", "vmap_sweep_batched",
    "material_fit",
)

CALCULATOR_WORDS = {
    "s_matrix_scan": "lumped/wire S-matrix scan",
    "mixed_s_matrix": "compute_mixed_s_matrix()",
    "topology_optimize": "topology_optimize()",
    "waveguide_s_matrix": "compute_waveguide_s_matrix()",
    "coaxial_line_reflection": "compute_coaxial_line_reflection()",
    "coaxial_two_port": "compute_coaxial_two_port()",
    "coax_msl_transition": "compute_coax_msl_transition()",
    "vmap_sweep_batched": "vmap_material_sweep() batched kernel",
    "material_fit": "differentiable_material_fit()",
}

LANE_WORDS = {
    "run_uniform": "uniform run()",
    "run_nonuniform": "graded run()",
    "run_subgridded": "subgridded run()",
    "run_adi": "ADI run()",
    "run_distributed": "multi-device run(devices=...)",
    "run_distributed_nu": "graded multi-device run(devices=...)",
    "fwd_uniform": "uniform forward()",
    "fwd_nonuniform": "graded forward()",
    "fwd_distributed": "forward(distributed=True)",
    "fwd_distributed_nu": "forward(distributed=True)",
    "fwd_adi": "ADI forward()",
}

_PEC_SIGMA = 1e6   # Simulation._PEC_SIGMA_THRESHOLD

# An owner-only H writer must supply the right Hx ghost consumed by NTFF.
# Extend this inventory when admitting one (including a magnetic source or
# lumped element); the adjacent admission tests require a collected invariant
# witness for every name. Curl/CPML-H already update the ghost like its owner.
SLAB_OWNER_ONLY_H_WRITERS = ("closed_box_tfsf",)


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
    ("_nominal_profiles", ""): _setting("_nominal_profiles", "nominal_profiles"),
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
    ("_waveguide_ports", "n_modes"): lambda sim: any(
        p.n_modes > 1 for p in sim._waveguide_ports),
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
    ("_nominal_profiles", ""): "nominal_profiles=",
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
    ("_waveguide_ports", "n_modes"): "a waveguide port with several modes (add_waveguide_port(n_modes>1))",
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

_DISTRIBUTED_NU = frozenset({"run_distributed_nu", "fwd_distributed_nu"})
_ALL = frozenset(LANES)
_OTHER_RUNNERS = _ALL - _DISTRIBUTED_NU
_ADI = frozenset({"run_adi", "fwd_adi"})
_GRADED = frozenset({"run_nonuniform", "fwd_nonuniform"}) | _DISTRIBUTED_NU
_UNIFORM_YEE = frozenset({"run_uniform", "fwd_uniform"})
_MESH_PINNED = _GRADED   # the lanes a dx/dy/dz profile reaches
_NO_SHEETS = _OTHER_RUNNERS - {"run_subgridded", "run_adi", "run_distributed", "fwd_adi"}
# PEC edges are carried by both graded entries and uniform distributed runs.
_PEC_EDGES = _NO_SHEETS | {"run_distributed"} | _DISTRIBUTED_NU

_ADMITTED_ON: dict[Row, frozenset] = {
    ("_freq_max", ""): _OTHER_RUNNERS,
    ("_domain", ""): _OTHER_RUNNERS,
    ("_dx", ""): _OTHER_RUNNERS,
    ("_dt_pin", ""): _MESH_PINNED - _DISTRIBUTED_NU,
    ("_dt_min_cell", ""): _MESH_PINNED - _DISTRIBUTED_NU,
    ("_nominal_profiles", ""): _MESH_PINNED - _DISTRIBUTED_NU,
    ("_precision", ""): _UNIFORM_YEE,
    ("_solver", ""): _ADI,
    ("_adi_cfl_factor", ""): _OTHER_RUNNERS,
    ("_stencil_order", ""): _UNIFORM_YEE,
    ("_mode", ""): _UNIFORM_YEE | _ADI,
    # ADI carries a homogeneous fill only: LANE_GATES decides (#1373).
    ("_materials", "eps"): _OTHER_RUNNERS - _ADI,
    ("_materials", "sigma"): _OTHER_RUNNERS - _ADI,
    ("_materials", "mu"): _OTHER_RUNNERS - _ADI,
    ("_materials", "debye"): _OTHER_RUNNERS - _ADI - {"run_subgridded"},
    ("_materials", "lorentz"): _OTHER_RUNNERS - _ADI - {"run_subgridded"},
    ("_materials", "drude"): _OTHER_RUNNERS - _ADI - {"run_subgridded"},
    ("_materials", "kerr"): _UNIFORM_YEE,
    ("_geometry", "pec_volume"): _OTHER_RUNNERS - _ADI,
    ("_geometry", "pec_sheet"): _PEC_EDGES,
    ("_geometry", "pec_wire"): _PEC_EDGES,
    ("_thin_conductors", "lossy_sheet"): _OTHER_RUNNERS - _ADI - {"run_subgridded"},
    ("_thin_conductors", "pec_sheet"): _PEC_EDGES,
    ("_thin_conductors", "surface_impedance"): _NO_SHEETS,
    ("_pinned_sheets", "pec_sheet"): _PEC_EDGES,
    ("_ports", "source"): _OTHER_RUNNERS,
    ("_ports", "amplitude_kind"): _OTHER_RUNNERS - _ADI,
    ("_ports", "lumped_port"): _OTHER_RUNNERS - _ADI,
    ("_ports", "passive_port"): _OTHER_RUNNERS - _ADI - {"run_distributed"},
    ("_ports", "wire_port"): _OTHER_RUNNERS - _ADI,
    ("_msl_ports", "msl_port"): _NO_SHEETS,
    ("_waveguide_ports", "waveguide_port"): _NO_SHEETS - {"fwd_distributed"},
    # One configuration per port is all the single-run assembly takes; the
    # per-mode list is driven only by compute_waveguide_s_matrix() (#2070).
    ("_waveguide_ports", "n_modes"): frozenset(),
    ("_coaxial_ports", "coax_port"): frozenset(),
    ("_floquet_ports", "floquet_port"): _UNIFORM_YEE,
    ("_floquet_ports", "scan_angle"): frozenset(),
    ("_lumped_rlc", "R"): _NO_SHEETS,
    ("_lumped_rlc", "series_RL"): _NO_SHEETS,
    ("_tfsf", "plane_wave"): frozenset({"run_uniform", "run_nonuniform", "fwd_uniform", "fwd_nonuniform", "run_distributed", "fwd_distributed"}),
    ("_refinement", "slab"): frozenset(),
    ("_refinement", "relaxed_validation"): frozenset(),
    ("_boundary", "cpml"): _OTHER_RUNNERS - _ADI - {"run_subgridded"},
    ("_boundary", "upml"): _UNIFORM_YEE,
    ("_pec_faces", "pec_face"): _OTHER_RUNNERS - _ADI - {"run_subgridded"},
    ("_boundary_spec", "pmc_face"): _OTHER_RUNNERS - _ADI - {"run_subgridded"},
    ("_boundary_spec", "conformal"): frozenset({"run_uniform"}),
    ("_boundary_spec", "conformal_s_matrix"): frozenset(),
    ("_boundary_spec", "absorbing_lid"): _OTHER_RUNNERS - _ADI,
    ("_periodic_axes", "periodic"): _UNIFORM_YEE | {"run_nonuniform", "fwd_nonuniform"},
    ("_cpml_layers", "layers"): _OTHER_RUNNERS - _ADI - {"run_subgridded"},
    ("_cpml_kappa_max", "kappa"): frozenset({"run_uniform", "run_distributed", "fwd_uniform"}),
    ("_interface_eps", "dual_average"): frozenset({"run_nonuniform", "fwd_nonuniform"}),
    ("_dx_profile", "graded"): _MESH_PINNED - _DISTRIBUTED_NU,
    ("_dy_profile", "graded"): _MESH_PINNED - _DISTRIBUTED_NU,
    ("_dz_profile", "graded"): _MESH_PINNED - _DISTRIBUTED_NU,
    ("_probes", "probe"): _OTHER_RUNNERS,
    ("_dft_planes", "dft_plane"): _NO_SHEETS | {"run_distributed"},
    ("_flux_monitors", "flux"): frozenset({"run_uniform", "run_nonuniform", "fwd_uniform",
                                          "fwd_nonuniform", "run_distributed"}),
    ("_ntff", "ntff_box"): _OTHER_RUNNERS - _ADI - {"run_subgridded"},  # Its x and y faces are always PEC (only a z lid may absorb).
    ("_current_moments", "block_moments"): _NO_SHEETS | {"run_distributed"},
}

# Both public entries use distributed_nu.run_nonuniform_distributed_pec.
# Define that runner's carried rows once; neither entry inherits uniform v2.
_DISTRIBUTED_NU_ROWS = frozenset({
    ('_dft_planes', 'dft_plane'),
    ('_flux_monitors', 'flux'),
    ('_ntff', 'ntff_box'),
    ('_current_moments', 'block_moments'),
    ('_boundary_spec', 'pmc_face'),
    ('_adi_cfl_factor', ''),
    ('_boundary', 'cpml'),
    ('_boundary_spec', 'absorbing_lid'),
    ('_cpml_layers', 'layers'),
    ('_domain', ''),
    ('_dt_min_cell', ''),
    ('_dt_pin', ''),
    ('_dx', ''),
    ('_dx_profile', 'graded'),
    ('_dy_profile', 'graded'),
    ('_dz_profile', 'graded'),
    ('_freq_max', ''),
    ('_geometry', 'pec_volume'),
    ('_geometry', 'pec_sheet'),
    ('_geometry', 'pec_wire'),
    ('_thin_conductors', 'pec_sheet'),
    ('_pinned_sheets', 'pec_sheet'),
    ('_materials', 'debye'),
    ('_materials', 'drude'),
    ('_materials', 'eps'),
    ('_materials', 'lorentz'),
    ('_materials', 'mu'),
    ('_materials', 'sigma'),
    ('_pec_faces', 'pec_face'),
    ('_ports', 'amplitude_kind'),
    ('_ports', 'source'),
    ('_ports', 'lumped_port'),
    ('_ports', 'wire_port'),
    ('_ports', 'passive_port'),
    ('_probes', 'probe'),
    ('_thin_conductors', 'lossy_sheet'),
})
for _row in _DISTRIBUTED_NU_ROWS:
    _ADMITTED_ON[_row] = _ADMITTED_ON[_row] | _DISTRIBUTED_NU


# Calculator memberships are independent of every shared lane set. A new
# lane capability therefore never silently expands a calculator's contract.
# waveguide_s_matrix drops NTFF with the #704 warning; material_fit drops it
# without a warning (pre-existing behaviour, reported for its owner).
NTFF_UNFORMED_LANES = frozenset({"waveguide_s_matrix", "material_fit"})

# The waveguide extractor assembles these materials and PEC operators and
# passes them to its device scans; its graded branch uses run_nonuniform_path.
for _row in (
    ("_freq_max", ""), ("_domain", ""), ("_dx", ""),
    ("_precision", ""), ("_mode", ""), ("_adi_cfl_factor", ""),
    ("_materials", "eps"), ("_materials", "sigma"), ("_materials", "mu"),
    ("_materials", "debye"), ("_materials", "lorentz"), ("_materials", "drude"),
    ("_geometry", "pec_volume"), ("_geometry", "pec_sheet"), ("_geometry", "pec_wire"),
    ("_thin_conductors", "lossy_sheet"), ("_thin_conductors", "pec_sheet"),
    ("_thin_conductors", "surface_impedance"), ("_pinned_sheets", "pec_sheet"),
    ("_waveguide_ports", "waveguide_port"), ("_waveguide_ports", "n_modes"),
    ("_boundary", "cpml"), ("_pec_faces", "pec_face"),
    ("_boundary_spec", "conformal"),
    ("_boundary_spec", "absorbing_lid"),
    ("_cpml_layers", "layers"), ("_cpml_kappa_max", "kappa"),
    ("_dx_profile", "graded"), ("_dy_profile", "graded"), ("_dz_profile", "graded"),
    # Preserve observer handling: the calculator warns for NTFF (#704).
    ("_probes", "probe"), ("_dft_planes", "dft_plane"),
    ("_flux_monitors", "flux"), ("_ntff", "ntff_box"),
    ("_current_moments", "block_moments"),
):
    _ADMITTED_ON[_row] = _ADMITTED_ON[_row] | {"waveguide_s_matrix"}


# Explicit memberships for calculators that assemble or step their own model.
_CALCULATOR_ROWS = {
    's_matrix_scan': (
        ('_freq_max', ''),
        ('_adi_cfl_factor', ''),
        ('_domain', ''),
        ('_dx', ''),
        ('_precision', ''),
        ('_stencil_order', ''),
        ('_mode', ''),
        ('_materials', 'eps'),
        ('_materials', 'sigma'),
        ('_materials', 'mu'),
        ('_materials', 'debye'),
        ('_materials', 'lorentz'),
        ('_materials', 'drude'),
        ('_geometry', 'pec_volume'),
        ('_geometry', 'pec_sheet'),
        ('_geometry', 'pec_wire'),
        ('_thin_conductors', 'lossy_sheet'),
        ('_thin_conductors', 'pec_sheet'),
        ('_thin_conductors', 'surface_impedance'),
        ('_pinned_sheets', 'pec_sheet'),
        ('_ports', 'source'),
        ('_ports', 'amplitude_kind'),
        ('_ports', 'lumped_port'),
        ('_ports', 'passive_port'),
        ('_ports', 'wire_port'),
        ('_msl_ports', 'msl_port'),
        ('_waveguide_ports', 'waveguide_port'),
        ('_floquet_ports', 'floquet_port'),
        ('_lumped_rlc', 'R'),
        ('_lumped_rlc', 'series_RL'),
        ('_tfsf', 'plane_wave'),
        ('_boundary', 'cpml'),
        ('_boundary', 'upml'),
        ('_pec_faces', 'pec_face'),
        ('_boundary_spec', 'pmc_face'),
        ('_boundary_spec', 'absorbing_lid'),
        ('_periodic_axes', 'periodic'),
        ('_cpml_layers', 'layers'),
        ('_cpml_kappa_max', 'kappa'),
        ('_probes', 'probe'),
        ('_dft_planes', 'dft_plane'),
        ('_flux_monitors', 'flux'),
        ('_ntff', 'ntff_box'),
        ('_current_moments', 'block_moments'),
    ),
    'mixed_s_matrix': (
        ('_freq_max', ''),
        ('_adi_cfl_factor', ''),
        ('_domain', ''),
        ('_dx', ''),
        ('_precision', ''),
        ('_stencil_order', ''),
        ('_mode', ''),
        ('_materials', 'eps'),
        ('_materials', 'sigma'),
        ('_materials', 'mu'),
        ('_materials', 'debye'),
        ('_materials', 'lorentz'),
        ('_materials', 'drude'),
        ('_geometry', 'pec_volume'),
        ('_geometry', 'pec_sheet'),
        ('_geometry', 'pec_wire'),
        ('_thin_conductors', 'lossy_sheet'),
        ('_thin_conductors', 'pec_sheet'),
        ('_pinned_sheets', 'pec_sheet'),
        ('_ports', 'lumped_port'),
        ('_ports', 'passive_port'),
        ('_ports', 'wire_port'),
        ('_msl_ports', 'msl_port'),
        ('_lumped_rlc', 'R'),
        ('_lumped_rlc', 'series_RL'),
        ('_boundary', 'cpml'),
        ('_boundary', 'upml'),
        ('_pec_faces', 'pec_face'),
        ('_boundary_spec', 'pmc_face'),
        ('_boundary_spec', 'absorbing_lid'),
        ('_periodic_axes', 'periodic'),
        ('_cpml_layers', 'layers'),
        ('_cpml_kappa_max', 'kappa'),
        ('_probes', 'probe'),
        ('_dft_planes', 'dft_plane'),
        ('_flux_monitors', 'flux'),
        ('_ntff', 'ntff_box'),
        ('_current_moments', 'block_moments'),
    ),
    'topology_optimize': (
        ('_freq_max', ''),
        ('_adi_cfl_factor', ''),
        ('_domain', ''),
        ('_dx', ''),
        ('_solver', ''),
        ('_precision', ''),
        ('_stencil_order', ''),
        ('_mode', ''),
        ('_materials', 'eps'),
        ('_materials', 'sigma'),
        ('_materials', 'mu'),
        ('_materials', 'debye'),
        ('_materials', 'lorentz'),
        ('_materials', 'drude'),
        ('_geometry', 'pec_volume'),
        ('_geometry', 'pec_sheet'),
        ('_geometry', 'pec_wire'),
        ('_thin_conductors', 'lossy_sheet'),
        ('_thin_conductors', 'pec_sheet'),
        ('_pinned_sheets', 'pec_sheet'),
        ('_ports', 'source'),
        ('_ports', 'amplitude_kind'),
        ('_ports', 'lumped_port'),
        ('_ports', 'passive_port'),
        ('_ports', 'wire_port'),
        ('_msl_ports', 'msl_port'),
        ('_waveguide_ports', 'waveguide_port'),
        ('_floquet_ports', 'floquet_port'),
        ('_lumped_rlc', 'R'),
        ('_lumped_rlc', 'series_RL'),
        ('_tfsf', 'plane_wave'),
        ('_boundary', 'cpml'),
        ('_boundary', 'upml'),
        ('_pec_faces', 'pec_face'),
        ('_boundary_spec', 'pmc_face'),
        ('_boundary_spec', 'absorbing_lid'),
        ('_periodic_axes', 'periodic'),
        ('_cpml_layers', 'layers'),
        ('_cpml_kappa_max', 'kappa'),
        ('_probes', 'probe'),
        ('_dft_planes', 'dft_plane'),
        ('_flux_monitors', 'flux'),
        ('_ntff', 'ntff_box'),
        ('_current_moments', 'block_moments'),
    ),
    'coaxial_line_reflection': (
        ('_freq_max', ''),
        ('_adi_cfl_factor', ''),
        ('_domain', ''),
        ('_dx', ''),
        ('_coaxial_ports', 'coax_port'),
        ('_boundary', 'cpml'),
        ('_cpml_layers', 'layers'),
        ('_cpml_kappa_max', 'kappa'),
        ('_current_moments', 'block_moments'),
    ),
    'coaxial_two_port': (
        ('_freq_max', ''),
        ('_adi_cfl_factor', ''),
        ('_domain', ''),
        ('_dx', ''),
        ('_coaxial_ports', 'coax_port'),
        ('_boundary', 'cpml'),
        ('_cpml_layers', 'layers'),
        ('_cpml_kappa_max', 'kappa'),
        ('_current_moments', 'block_moments'),
    ),
    'coax_msl_transition': (
        ('_freq_max', ''),
        ('_adi_cfl_factor', ''),
        ('_domain', ''),
        ('_dx', ''),
        ('_materials', 'eps'),
        ('_materials', 'sigma'),
        ('_materials', 'mu'),
        ('_geometry', 'pec_volume'),
        ('_geometry', 'pec_sheet'),
        ('_geometry', 'pec_wire'),
        ('_thin_conductors', 'lossy_sheet'),
        ('_thin_conductors', 'pec_sheet'),
        ('_pinned_sheets', 'pec_sheet'),
        ('_coaxial_ports', 'coax_port'),
        ('_msl_ports', 'msl_port'),
        ('_boundary', 'cpml'),
        ('_cpml_layers', 'layers'),
        ('_cpml_kappa_max', 'kappa'),
        ('_current_moments', 'block_moments'),
    ),
    'vmap_sweep_batched': (
        ('_freq_max', ''),
        ('_adi_cfl_factor', ''),
        ('_domain', ''),
        ('_dx', ''),
        ('_mode', ''),
        ('_materials', 'eps'),
        ('_materials', 'sigma'),
        ('_materials', 'mu'),
        ('_geometry', 'pec_volume'),
        ('_geometry', 'pec_sheet'),
        ('_geometry', 'pec_wire'),
        ('_thin_conductors', 'lossy_sheet'),
        ('_thin_conductors', 'pec_sheet'),
        ('_pinned_sheets', 'pec_sheet'),
        ('_ports', 'source'),
        ('_ports', 'amplitude_kind'),
        ('_boundary', 'cpml'),
        ('_pec_faces', 'pec_face'),
        ('_boundary_spec', 'absorbing_lid'),
        ('_periodic_axes', 'periodic'),
        ('_cpml_layers', 'layers'),
        ('_cpml_kappa_max', 'kappa'),
        ('_probes', 'probe'),
        ('_dft_planes', 'dft_plane'),
    ),
    'material_fit': (
        ('_freq_max', ''),
        ('_adi_cfl_factor', ''),
        ('_domain', ''),
        ('_dx', ''),
        ('_mode', ''),
        ('_materials', 'eps'),
        ('_materials', 'sigma'),
        ('_materials', 'mu'),
        ('_materials', 'debye'),
        ('_materials', 'lorentz'),
        ('_materials', 'drude'),
        ('_geometry', 'pec_volume'),
        ('_geometry', 'pec_sheet'),
        ('_geometry', 'pec_wire'),
        ('_thin_conductors', 'lossy_sheet'),
        ('_thin_conductors', 'pec_sheet'),
        ('_pinned_sheets', 'pec_sheet'),
        ('_ports', 'lumped_port'),
        ('_boundary', 'cpml'),
        ('_boundary', 'upml'),
        ('_pec_faces', 'pec_face'),
        ('_boundary_spec', 'pmc_face'),
        ('_boundary_spec', 'absorbing_lid'),
        ('_cpml_layers', 'layers'),
        ('_cpml_kappa_max', 'kappa'),
        ('_periodic_axes', 'periodic'),
        ('_probes', 'probe'),
        ('_dft_planes', 'dft_plane'),
        ('_flux_monitors', 'flux'),
        ('_ntff', 'ntff_box'),
        ('_current_moments', 'block_moments'),
    ),
}
for _calculator, _rows in _CALCULATOR_ROWS.items():
    for _row in _rows:
        _ADMITTED_ON[_row] = _ADMITTED_ON[_row] | {_calculator}

# Both uniform distributed entries reach the same runner and admission.
for _row, _lanes in _ADMITTED_ON.items():
    _ADMITTED_ON[_row] = (_lanes - {"fwd_distributed"}) | (
        {"fwd_distributed"} if "run_distributed" in _lanes else set())

ADMITS: dict[str, frozenset] = {
    lane: frozenset(row for row, lanes in _ADMITTED_ON.items() if lane in lanes)
    for lane in LANES + CALCULATORS
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
def _adi_homogeneous(sim, grid) -> bool:
    """Whether the declared materials realize one eps_r and one sigma over the
    whole grid, the only fill the ADI lane carries (#1373): its update reads
    each cell's own value, so an interface sits half a cell from where the Yee
    lanes put it. The realized arrays decide, assembled from the declaration."""
    from rfx.adi import adi_material_interface_refusal
    if sim._uses_nonuniform_mesh:
        return False   # ADI refuses a graded mesh first; no uniform grid to read
    grid = sim._build_grid() if grid is None else grid
    # The gate reads eps_r/sigma cells only. Sheets and wires are refused on
    # ADI by their own (ungated) rows; collecting and dropping them is the
    # assembler's explicit "I read cells only" (_refuse_uncollected_pec).
    # Without it every refusal of a sheet/wire model ran this gate (message()
    # lists carriers through refused() on every lane) and ended in the
    # assembler's internal ValueError instead of the refusal.
    materials = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    return adi_material_interface_refusal(materials.eps_r, materials.sigma) is None


# tests/contracts/path_disposition.py lists the same gates as LANE_GATES.
_LID_ROWS = (("_boundary", "cpml"), ("_pec_faces", "pec_face"),
             ("_cpml_layers", "layers"), ("_cpml_kappa_max", "kappa"))


def _waveguide_profiled(sim, grid) -> bool:
    """Only the graded waveguide builder passes the declared dt controls."""
    return any(getattr(sim, attr) is not None
               for attr in ("_dx_profile", "_dy_profile", "_dz_profile"))


_ADI_MATERIAL_ROWS = (("_materials", "eps"), ("_materials", "sigma"))


def _subgrid_opted_in(sim) -> bool:
    """Whether the refinement opts in to the subgridded lane (#1465, #1468:
    validation='research' or 'off')."""
    ref = sim._refinement
    return ref is not None and ref.get("validation", "production") in {"research", "off"}

LANE_GATES: dict[str, dict[Row, Callable]] = {
    "run_subgridded": {
        **{row: _guarded_lid for row in _LID_ROWS},
        ("_refinement", "slab"): lambda sim, grid: (
            sim._refinement.get("validation", "production") in {"research", "off"}),
    },
    "run_adi": {row: _adi_homogeneous for row in _ADI_MATERIAL_ROWS},
    "fwd_adi": {row: _adi_homogeneous for row in _ADI_MATERIAL_ROWS},
    "waveguide_s_matrix": {row: _waveguide_profiled for row in (
        ("_dt_pin", ""), ("_dt_min_cell", ""), ("_nominal_profiles", ""))},
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
    if lane == "run_subgridded" and ("_refinement", "slab") in rows:
        from rfx.subgridding._notice import SUBGRID_NOTICE
        return SUBGRID_NOTICE
    # Both conformal rows describe one input; display its more specific refusal.
    message_rows = [row for row in rows if row != ("_boundary_spec", "conformal")
                    or ("_boundary_spec", "conformal_s_matrix") not in rows]
    if lane in CALCULATORS:
        lines = ([] if lane == "material_fit" else
                 [f"  - {ROW_WORDS[row]}" for row in message_rows])
        if ("_refinement", "slab") in rows and lane != "material_fit":
            lines.append("  - add_refinement(): this calculator has no subgridded lane (#1240).")
            if sim._uses_nonuniform_mesh:
                lines.append("  - subgridding on a non-uniform mesh is not supported.")
        if lane == "material_fit":
            # Preserve the input names and remedies of #1380 while the
            # refusal itself is owned solely by the admission membership.
            names = {
                ("_ports", "wire_port"): "add_port(..., extent=...) (wire port)",
                ("_ports", "source"): "add_source() (plain source)",
                ("_ports", "passive_port"): "add_port(..., excite=False) (passive port)",
                ("_msl_ports", "msl_port"): "add_msl_port()",
                ("_lumped_rlc", "R"): "add_lumped_rlc()",
                ("_lumped_rlc", "series_RL"): "add_lumped_rlc()",
                ("_tfsf", "plane_wave"): "add_tfsf_source()",
                ("_waveguide_ports", "waveguide_port"): "add_waveguide_port()",
                ("_coaxial_ports", "coax_port"): "add_coaxial_port()",
                ("_floquet_ports", "floquet_port"): "add_floquet_port()",
                ("_materials", "kerr"): "add_material(..., chi3=...) (Kerr material)",
                ("_precision", ""): f"precision={sim._precision!r}",
            }
            remedies = {
                ("_solver", ""): "Use solver='yee'.",
                ("_precision", ""): "Use precision='float32'.",
                ("_stencil_order", ""): "Use stencil_order=2.",
                ("_interface_eps", "dual_average"): "Use interface_eps='sampled'.",
                ("_refinement", "slab"): "This calculator has no subgridded lane (#1240). Remove the refinement.",
            }
            lines.extend(f"  - {ROW_WORDS[row]}"
                         + (f" — {names[row]}" if row in names else "")
                         + " (#1290). "
                         + remedies.get(row, "Remove this input from sim_factory.")
                         for row in message_rows)
        if lane == "waveguide_s_matrix" and ("_solver", "") in rows:
            lines.append("  - solver='adi' (#1300). Use solver='yee'.")
        if lane == "waveguide_s_matrix" and any(row[0] == "_lumped_rlc" for row in rows):
            lines.append("  - lumped RLC elements are not implemented (#1263). Use run() / forward().")
        return (f"{CALCULATOR_WORDS[lane]} does not carry these declared inputs "
                "and refuses them before the first time step:\n"
                + "\n".join(lines)
                + "\nUse run() / forward() on a model those paths support.")
    # On ADI a dielectric or conductor is carried as a homogeneous fill; what
    # is refused is the interface, so the interface text stands for both rows.
    lines = [f"  - {ROW_WORDS[row]} is not carried by the {LANE_WORDS[lane]} lane."
             for row in message_rows if not (lane in _ADI and row in _ADI_MATERIAL_ROWS)]
    if lane in _ADI and any(row in rows for row in _ADI_MATERIAL_ROWS):
        from rfx.adi import ADI_INTERFACE_MESSAGE
        lines.append(f"  - {ADI_INTERFACE_MESSAGE}")
    if lane in _ADI and ("_ports", "amplitude_kind") in rows:
        lines.append(
            "ADI implements only amplitude_kind='field'; 'current' is the default "
            "when amplitude_kind is not given (2.0); declare amplitude_kind='field' "
            "to run on ADI (the earlier ADI behaviour).")
    # The subgridded lane is unstable and unverified (#1465): it is not
    # offered as a carrier unless the model already opts in to it.
    carriers = [LANE_WORDS[other] for other in LANES if other != lane
                and (other != "run_subgridded" or _subgrid_opted_in(sim))
                and not set(refused(sim, other, run_args)) - LANE_SELECTORS]
    where = ("Lanes that carry every input of this model apart from the ones that choose the "
             "lane (solver, mesh profiles, refinement): " + ", ".join(carriers) + "."
             if carriers else
             "No time-stepping lane carries every input of this model apart from the ones that "
             "choose the lane (solver, mesh profiles, refinement).")
    if ("_boundary_spec", "conformal_s_matrix") in rows:
        where += (" Use compute_s_params=False for conformal probe fields, or "
                  "conformal_pec=False for a staircase S-matrix.")
    if ("_waveguide_ports", "n_modes") in rows:
        where += (" run() and forward() drive and record one mode per waveguide port (n_modes=1); "
                  "compute_waveguide_s_matrix() on a uniform mesh measures several modes per port (#2070).")
    return (f"The {LANE_WORDS[lane]} lane would solve this Simulation as if "
            f"{'these inputs were' if len(lines) > 1 else 'this input was'} not "
            "declared, so it is refused before the first time step:\n"
            + "\n".join(lines) + "\n" + where
            + "\nRemove the input, or declare the model so that a lane that "
            "carries it runs.")


class _NoWarnings:
    """Drops the unresolved-pulse advisory; admission keeps only its refusal."""

    @staticmethod
    def warn(*_args, **_kwargs):
        return None


def refuse_undriven_drives(sim) -> None:
    """Issue 1668: refuse a drive whose pulse is not resolved by dt (its
    table, sampled at dt over the pulse, holds no value above 1e-6 of its
    amplitude), on entries that run no preflight. The same check preflight
    runs, ``_validate_cfg_unresolved_pulse``; raises ``PreflightConfigError``
    (a ``ValueError``)."""
    from rfx.core.jax_utils import is_tracer
    from rfx.grid import C0
    check = getattr(sim, "_validate_cfg_unresolved_pulse", None)
    if check is None or is_tracer(getattr(sim, "_dx", None)):
        return
    check(_NoWarnings, sim._dx or C0 / sim._freq_max / 20.0)


def refuse_distributed_sources(sim, lane):
    """Name sources that cannot be stepped on the requested slab lane."""
    if getattr(sim, "_waveguide_ports", ()):
        raise NotImplementedError(f"waveguide ports (add_waveguide_port) are not supported on {lane}; use one device")
    entry = getattr(sim, "_tfsf", None)
    if entry is None:
        return
    if getattr(sim, "_uses_nonuniform_mesh", False) or lane.endswith("_nu"):
        raise NotImplementedError(f"TFSF plane-wave sources (add_tfsf_source) are not supported on the graded multi-device lane ({lane}); use one device")
    refuse_slab_tfsf(entry, lane)


def refuse_slab_tfsf(entry, lane="uniform multi-device"):
    """Accept declarations or realized configs; face replays never enter a slab scan."""
    from rfx.sources.tfsf import is_tfsf_2d, is_tfsf_methodB
    if getattr(entry, "face_record", None) is not None:
        kind = "TFSF face_record replay"
    elif is_tfsf_methodB(entry) or (abs(getattr(entry, "angle_deg", 0.)) > .01
                                    and getattr(entry, "method", None) == "methodB"):
        kind = "TFSF Method B oblique (is_tfsf_methodB)"
    elif is_tfsf_2d(entry) or abs(getattr(entry, "angle_deg", 0.)) > .01:
        kind = "TFSF 2-D Bloch line (is_tfsf_2d)"
    elif getattr(entry, "closed_box", False) and (
        getattr(entry, "theta_deg", 0.) != 0 or getattr(entry, "phi_deg", 0.) != 0
    ):
        kind = "oblique closed-box TFSF (theta_deg or phi_deg != 0)"
    elif not getattr(entry, "closed_box", False):
        kind = "open-slab normal-incidence TFSF (periodic y/z is not on slabs)"
    else:
        return
    raise NotImplementedError(f"{kind} is not supported on {lane}; use one device")


#: Closed-box TF/SF validation covers three seams, a padded last slab, a box face
#: on a seam, probe and far-field parity, and PEC sphere backscatter vs Mie.
SLAB_TFSF_MAX_DEVICES = 4


def refuse_slab_tfsf_devices(entry, n_devices, lane="uniform multi-device"):
    """Refuse a closed-box TF/SF on more devices than its multi-seam validation covers."""
    if entry is not None and getattr(entry, "closed_box", False) and n_devices > SLAB_TFSF_MAX_DEVICES:
        raise NotImplementedError(
            f"closed-box TFSF (add_tfsf_source(closed_box=True)) on {n_devices} devices is not supported on "
            f"{lane}: it is validated on at most {SLAB_TFSF_MAX_DEVICES} devices (three seams); "
            f"5 or more are refused. Use devices=jax.devices()[:{SLAB_TFSF_MAX_DEVICES}] or one device")


_COAX_ADVICE = {
    "run": ("for a single add_coaxial_port(), compute_s_matrix(method=...) "
            "dispatches to either. Use add_port(..., extent=...) for "
            "probe-feed S-parameters."),
    "forward": ("both take eps_scale= for jax.grad. Use add_port(..., extent=...) "
                "for differentiable probe-feed S11 objectives."),
}


def refuse_coaxial_port(sim, entry: str) -> None:
    """The one coaxial-port refusal of ``run()`` and ``forward()`` (#2075),
    raised before mesh planning while no time-stepping lane admits the row;
    ``entry`` selects the advice."""
    row = ("_coaxial_ports", "coax_port")
    if sim._coaxial_ports and not any(row in ADMITS[lane] or row in LANE_GATES.get(lane, {})
                                      for lane in LANES):
        raise NotImplementedError(
            f"add_coaxial_port() is not wired into Simulation.{entry}(). A "
            "coaxial port's S-parameters come from "
            "compute_coaxial_line_reflection(...) (one-port reflection) "
            "or compute_coaxial_two_port(...) (through line); " + _COAX_ADVICE[entry])


def admit(sim, lane: str, *, run_args=None, grid=None, diagnostics=None) -> None:
    """Raise ``NotImplementedError``, as the lanes' own refusals do, if
    ``lane`` does not carry every input ``sim`` declares. ``run_args`` are
    the call's static arguments that ``CALL_GATES`` read; ``grid`` is the
    grid the lane built, for its ``LANE_GATES``. A drive whose pulse is not
    resolved by dt is refused first (``refuse_undriven_drives``)."""
    if lane in {"run_distributed", "fwd_distributed", "run_distributed_nu", "fwd_distributed_nu"}:
        refuse_distributed_sources(sim, lane)
    refuse_undriven_drives(sim)
    from rfx.sources.wire_radius import require_radius_support
    require_radius_support(sim, lane)
    rows = refused(sim, lane, run_args, grid)
    if rows:
        if diagnostics is not None:
            from rfx._diagnostic_transport import diagnostic_refusal
            raise diagnostic_refusal(NotImplementedError(message(lane, rows, sim, run_args)), diagnostics)
        raise NotImplementedError(message(lane, rows, sim, run_args))
    from rfx.boundaries.model import refresh_boundary_model
    refresh_boundary_model(sim)
    from rfx.boundaries.features import admit_ntff, admit_waveguide
    from rfx.boundaries.tfsf import admit_simulation
    if lane not in NTFF_UNFORMED_LANES:
        admit_ntff(sim, lane=lane)
    admit_waveguide(sim, lane=lane)
    from rfx.measurement import admit_waveguide_entries
    admit_waveguide_entries(sim)
    admit_simulation(sim, materials=(run_args or {}).get("tfsf_materials"))


def _warn_implicit_s(message):
    """Attribute implicit S warnings to the first caller outside rfx."""
    import sys
    import warnings
    stacklevel = 2
    frame = sys._getframe(1)
    try:
        while frame is not None and (
            frame.f_globals.get("__name__", "") == "rfx"
            or frame.f_globals.get("__name__", "").startswith("rfx.")
        ):
            stacklevel += 1
            frame = frame.f_back
    finally:
        del frame
    warnings.warn(message, UserWarning, stacklevel=stacklevel)


def require_isolated_s_drive(sim, *, implicit=False, fields_only=False):
    """Admit S from one record, or omit implicit S when drives are mixed."""
    excited = [(index, port) for index, port in enumerate(sim._ports)
               if port.impedance > 0 and port.excite]
    if len(excited) < 2:
        return True
    ports = "; ".join(f"port {index} at {port.position}" for index, port in excited)
    message = (
        f"S-parameters from one record require an isolated drive; excited ports: {ports}. "
        "set excite=False on every port but the driven one and run once per driven port, "
        "or use run(compute_s_params=True) on one device, which drives one port per run"
    )
    if fields_only:
        message += "; use run(compute_s_params=False) for fields only"
    if implicit:
        _warn_implicit_s(message + "; returning no S-parameters.")
        return False
    raise ValueError(message)


def graded_lumped_default_s(sim):
    """Graded run() with no S argument forms lumped S when one record can.

    Only for an all-lumped port set, with exactly one driven port, no plain
    source or 0-ohm port and no periodic axis; otherwise the run returns
    fields with S=None (two driven ports also warn, as forward() does; with no
    driven port there is no incident wave to form S from). A port
    set with a wire port keeps the wire default. Never a refusal: the user did
    not ask for S (rfx-lead D:6b5bd8, #1373).
    """
    ports = [p for p in sim._ports if p.impedance != 0.0]
    if (not ports or any(p.extent is not None for p in ports)
            or any(p.impedance == 0.0 for p in sim._ports) or sim._periodic_axes
            or not any(p.excite for p in ports)):
        return False
    return require_isolated_s_drive(sim, implicit=True)


def refuse_plain_sources_s_matrix(sim, *, main_record=False, implicit=False):
    """A port drive must be the only excitation of a lumped/wire S solve.

    Routes that build S drive by drive (the scan driver, run()'s scan S, the
    multi-device S, subgridding) drive EVERY impedance port, passive ones too,
    so any plain source fires in every drive: refused whenever an impedance port
    exists. Routes that read S from the main run's own record (``main_record``:
    forward(port_s11_freqs=), the single-wire fast path on either mesh)
    require a driven impedance port, regardless of other excitations.
    Passive records remain available as termination diagnostics, without S.
    """
    ports = [p for p in sim._ports if p.impedance != 0.0]
    if main_record and ports:
        from rfx.measurement.port_s_contract import record_ports, require_driven_record_s
        if not require_driven_record_s(
                (p.excite for p in record_ports(sim, default=implicit)), implicit=implicit):
            return False
    if not ports or not any(p.impedance == 0.0 for p in sim._ports):
        return True
    reason = (
        "Lumped/wire S-matrix requests do not support plain sources / "
        "0-ohm ports (add_source, add_polarized_source or "
        "add_port(impedance=0)): they fire in every port drive. ")
    if implicit:
        _warn_implicit_s(
            reason + "Remove the plain source, or pass compute_s_params=False "
            "to silence this warning; returning no S-parameters.")
        return False
    raise NotImplementedError(
        reason + "Remove the plain source, or drop the S request to run with the "
        "source included: run(compute_s_params=False) returns the raw port waves, "
        "forward() without port_s11_freqs the fields and probe records.")


def resolve_plain_source_s_request(sim, compute_s_params, *, requested, devices=None):
    """run()'s S request after the plain-source rule: an explicit request with a
    plain source or 0-ohm port is refused; the default request warns and
    resolves to False (fields, S=None), as graded run() does (rfx-lead D:c70245)."""
    # Subgridded runs opt into S explicitly; uniform/graded runs resolve
    # None from lumped/wire ports, graded lumped ports only when one is driven.
    # Passive single-wire main records warn and keep raw spectra with S disabled.
    default_requested = (
        compute_s_params is None
        and sim._refinement is None
        and sim._solver != "adi"
        and any(pe.impedance != 0.0
                and (not sim._uses_nonuniform_mesh or pe.extent is not None
                     or getattr(pe, "excite", True)) for pe in sim._ports)
    )
    if compute_s_params is False or not (requested or default_requested):
        return compute_s_params
    _one_device = devices is None or len(devices) <= 1
    # S read from the main run's own record: the single-wire fast path on
    # either mesh (no scan, _s_matrix_ports False).
    _main_record = (
        _one_device and sim._refinement is None and sim._solver != "adi"
        and not _s_matrix_ports(sim))
    if (not _one_device and sim._uses_nonuniform_mesh
            and any(p.impedance != 0.0 and not getattr(p, "excite", True) for p in sim._ports)):
        reason = ("A passive port's S column needs a drive of its own: graded run() drives each "
                  "port in turn on one device; on several devices it reads S from one record. ")
        if compute_s_params is None and not requested:
            _warn_implicit_s(reason + "Run on one device for every column; returning no S-parameters.")
            return False
        raise NotImplementedError(reason + "Run on one device (omit devices=), or call "
                                  "run(compute_s_params=False) for fields only.")
    if not refuse_plain_sources_s_matrix(
            sim, main_record=_main_record,
            implicit=compute_s_params is None and not requested):
        return False
    return compute_s_params


def admit_forward_s_matrix(sim):
    """Check explicit forward S requests before dispatch or stepping."""
    from rfx.preflight._msl_probe_rules import admit_msl_auto_probes
    # MSL automatic probe counts from the full declaration (#1584).
    admit_msl_auto_probes(sim)
    from rfx.measurement import admit_waveguide_entries
    admit_waveguide_entries(sim)
    refuse_plain_sources_s_matrix(sim, main_record=True)
    require_isolated_s_drive(sim)


def admit_run_s_matrix(sim, *, compute_s_params=None, conformal_pec=None,
                       distributed=False):
    """Check the requested scan extraction before run() starts its field solve."""
    uses_scan = (False if distributed and sim._uses_nonuniform_mesh else
                 any(p.impedance != 0.0 for p in sim._ports) if distributed else _s_matrix_ports(sim))
    if compute_s_params is not False:
        # Without the scan, S is read from the main run's own record.
        refuse_plain_sources_s_matrix(sim, main_record=not uses_scan)
    if compute_s_params is False or not uses_scan:
        return
    try:
        admit(sim, "s_matrix_scan", run_args={"conformal_pec": conformal_pec})
    except NotImplementedError as error:
        rows = refused(sim, "s_matrix_scan", {"conformal_pec": conformal_pec})
        inputs = ", ".join(ROW_WORDS[row] for row in rows) or str(error)
        raise NotImplementedError(
            "run() cannot compute its lumped/wire S-matrix with these inputs: "
            + inputs + ". Use compute_s_params=False for fields only, or remove "
            "the unsupported input.") from error


def _eager_trace_state(get_opaque_trace_state):
    """The trace state of a thread no transformation has entered (state is per thread)."""
    state = []
    thread = threading.Thread(target=lambda: state.append(get_opaque_trace_state()))
    thread.start()
    thread.join()
    return state[0]


def _under_external_jax_transform() -> bool:
    """Compare the caller's trace state with an untransformed thread's (jax 0.10 and 0.11)."""
    try:
        from jax.extend.core import get_opaque_trace_state
    except ImportError:
        try:  # JAX 0.6 export layout.
            from jax.core import get_opaque_trace_state
        except ImportError:  # The JAX 0.4.20 export layout.
            from jax.core import trace_state_clean
            return not trace_state_clean()
    return get_opaque_trace_state() != _eager_trace_state(get_opaque_trace_state)


def _refuse_transformed_extended_tfsf(entry, until_decay=None, decay_max_steps=None) -> None:
    """Keep run()'s source fence before setup can hide caller tracing."""
    from rfx.sources.tfsf_discrete_plane_wave import validate_run
    validate_run(entry, until_decay, decay_max_steps, _under_external_jax_transform)


@jax.custom_jvp
def _no_shape_tangent(value):
    """Preserve jit/vmap values while refusing differentiation of a shape."""
    return value


@_no_shape_tangent.defjvp
def _no_shape_tangent_jvp(primals, tangents):
    raise NotImplementedError(
        "A gradient with respect to a declared shape dimension (a size or position "
        "of a shape added with Simulation.add) is not supported: run() and "
        "forward() draw shapes onto grid cells, so the result is a step function "
        "of that dimension and its gradient would be zero almost everywhere. "
        "Differentiate a material value through eps_override, or use a design "
        "region (design_box / design_eps_override) for shape optimisation. "
        "jax.jit and jax.vmap over a Sphere or Cylinder dimension are supported "
        "(with subpixel_smoothing only in a closed PEC domain).")


def guard_shape_tangents(sim):
    """Attach the tangent refusal before any lane realizes declared shapes.

    Concrete leaves and entries retain their identity. Dataclass fields and
    tuples (centres, corners) are walked; other shape classes are left as they are.
    """
    def guard(value):
        if is_tracer(value):
            return _no_shape_tangent(value)
        if is_dataclass(value) and not isinstance(value, type):
            changes = {}
            for field in fields(value):
                leaf = getattr(value, field.name)
                guarded = guard(leaf)
                if guarded is not leaf:
                    changes[field.name] = guarded
            return replace(value, **changes) if changes else value
        if isinstance(value, tuple):
            guarded = tuple(guard(leaf) for leaf in value)
            return guarded if any(a is not b for a, b in zip(value, guarded)) else value
        return value

    for index, entry in enumerate(sim._geometry):
        if is_dataclass(entry.shape):
            shape = guard(entry.shape)
            if shape is not entry.shape:
                sim._geometry[index] = replace(entry, shape=shape)


def admit_traced_run(sim, until_decay=None, decay_max_steps=None, radiated_flux_box=None):
    """Refuse an unused flux stop and preserve source/shape fences before dispatch."""
    from rfx.measurement.decay import refuse_radiated_flux_stop
    refuse_radiated_flux_stop(radiated_flux_box, until_decay, sim._boundary)
    _refuse_transformed_extended_tfsf(sim._tfsf, until_decay, decay_max_steps)
    guard_shape_tangents(sim)


def refuse_smoothing_overrides(*, subpixel_smoothing, eps_override=None,
                               pec_occupancy=None, design_occupancy=None):
    """A geometry tensor cannot be combined with another eps operator."""
    if not subpixel_smoothing:
        return
    if eps_override is not None:
        raise NotImplementedError(
            "subpixel_smoothing builds the permittivity tensor from declared shapes; "
            "eps_override would not reach the solve. Omit smoothing or eps_override.")
    if pec_occupancy is not None or design_occupancy is not None:
        raise NotImplementedError(
            "subpixel_smoothing cannot combine with pec_occupancy / design_occupancy: "
            "they supply a second inverse-permittivity operator.")


from rfx.runners._smoothing_admission import refuse_forward_smoothing


def _refuse_conformal_boundary(self, path_name: str, *, entry: str,
                               instead: str) -> None:
    """Refuse ``Boundary(conformal=True)`` on a path with no Dey-Mittra update.

        ``run()`` on the uniform lane and the uniform waveguide S-matrix
        lane turn a conformal axis into Dey-Mittra weights on every PEC
        shape; a path without that update solves staircased PEC instead.
        A conformal axis always has PEC shapes to act on: both of its faces
        are PEC, and the hi face's half-space wall is always one of the
        uniform assembler's ``pec_shapes`` (``rfx/api/_compile.py``).
        """
    self._refuse_unsupported_run_kwargs(
        path_name,
        {"conformal_pec": bool(self._boundary_spec.conformal_faces())},
        instead=instead,
        entry=entry,
        reason_overrides={"conformal_pec":
            "Boundary(conformal=True) asks for Dey-Mittra conformal "
            "PEC, and this path has no conformal update, so the walls "
            "and PEC shapes would be staircased"},
        remedy_overrides={"conformal_pec":
            "drop Boundary(conformal=True) (staircase PEC)"},
    )


def _reject_upml_on_nonuniform(self, lane: str) -> None:
    """Refuse ``boundary='upml'`` on a non-uniform lane (#680).

        ``rfx/nonuniform.py`` has no UPML implementation at all: its
        absorber dispatch is ``use_cpml = grid.cpml_layers > 0``, which
        keys on the layer COUNT and never reads ``sim._boundary``. So a
        non-uniform run asked for UPML silently got CPML instead —
        measured on a 4x4x3 mm ez-dipole domain, otherwise identical
        configs: ``apply_upml_e`` called 1x on the uniform lane and 0x on
        the NU lane, with ``sim._boundary`` still reading ``'upml'``
        afterwards, so even a post-hoc audit reported the wrong absorber.

        Refuse at lane entry, in the style of the sibling distributed
        guard, rather than substituting: the two absorbers have different
        reflection and different material-in-pad semantics, and a silently
        swapped one is exactly the class that made a "UPML + non-uniform"
        regression test pass while never touching UPML.
        """
    if self._boundary != "upml" or self._cpml_layers <= 0:
        return
    raise ValueError(
        f"boundary='upml' does not support the {lane} lane: the "
        "non-uniform runner implements CPML only (rfx/nonuniform.py "
        "dispatches on cpml_layers > 0 and never reads the boundary "
        "type), so this configuration used to run CPML while "
        "sim._boundary kept reporting 'upml' (issue #680). Pass "
        "boundary='cpml' to run the absorber that is actually "
        "implemented here, or drop the mesh profile(s) to use the "
        "uniform lane, which does implement UPML."
    )


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly (as rfx/preflight/sources.py
# does). The functions above were ``def``s in ``_ExecuteMixin``'s body, so their
# qualnames read ``_ExecuteMixin.<name>``; ``rfx/api/__init__.py``
# rewrites exactly ``<mixin>.<name>`` -> ``Simulation.<name>`` at class
# composition and skips a bare name, so without these lines a TypeError would no
# longer name ``Simulation.<name>()``.
# ``tests/contracts/test_execute_split_qualnames.py`` pins them.
# ---------------------------------------------------------------------------
_refuse_conformal_boundary.__qualname__ = "_ExecuteMixin._refuse_conformal_boundary"
_reject_upml_on_nonuniform.__qualname__ = "_ExecuteMixin._reject_upml_on_nonuniform"


def admit_sheet_layout_lane(sim, *, distributed):
    """Refuse before static assembly can mistake a switch for fixed metal."""
    if distributed:
        from rfx.runners._distributed_entry import refuse_sheet_layout
        refuse_sheet_layout()
    if not sim._uses_nonuniform_mesh or sim._solver == 'adi' or sim._refinement is not None:
        from rfx.runners.uniform_forward import refuse_sheet_layout
        refuse_sheet_layout()
