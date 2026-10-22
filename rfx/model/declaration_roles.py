"""Declaration roles from Step B, including the leader's 2026-10-21 amendments.

Containers retain their kind/count on the host; nested records follow their
own registry entries. Missing classes/fields are never assigned by a fallback.
"""
from dataclasses import fields, is_dataclass
import functools
import inspect

import jax

from rfx.api._spec import (
    MaterialSpec, _GeometryEntry, _PortEntry, _ProbeEntry, _DFTPlaneEntry,
    _FluxMonitorEntry, _WaveguidePortEntry, _FloquetPortEntry, _MSLPortEntry,
)
from rfx.boundaries.model import BoundaryModel, Face, Axis, Features
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.farfield import NTFFBox
from rfx.geometry.csg import Box, OrientedBox, PolylineWire, Sphere, Cylinder
from rfx.lumped import LumpedRLCSpec
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.materials.thin_conductor import PinnedSheet, ThinConductor
from rfx.sources.coaxial_port import CoaxialPort
from rfx.sources.sources import GaussianPulse, ModulatedGaussian, CWSource, CustomWaveform
from rfx.sources.tfsf_discrete_plane_wave import _TFSFEntry

STRUCTURE = "STRUCTURE"
VALUE = "VALUE"
STRUCTURE_TRACEABLE = "STRUCTURE_TRACEABLE"


_SCHEMA = {
    MaterialSpec: ('eps_r', 'sigma', 'mu_r', 'debye_poles', 'lorentz_poles', 'chi3'),
    _GeometryEntry: ('shape', 'material_name'),
    _PortEntry: ('position', 'component', 'impedance', 'waveform', 'extent', 'excite', 'direction', 'reference_plane_cells', 'amplitude_kind', 'terminates', 'radius'),
    _ProbeEntry: ('position', 'component'),
    _DFTPlaneEntry: ('name', 'axis', 'coordinate', 'component', 'freqs', 'n_freqs'),
    _FluxMonitorEntry: ('name', 'axis', 'coordinate', 'freqs', 'n_freqs', 'size', 'center', 'dft_window', 'dft_window_alpha'),
    _WaveguidePortEntry: ('name', 'x_position', 'y_range', 'z_range', 'x_range', 'mode', 'mode_type', 'direction', 'freqs', 'n_freqs', 'f0', 'bandwidth', 'amplitude', 'probe_offset', 'ref_offset', 'calibration_preset', 'reference_plane', 'probe_plane', 'n_modes', 'waveform', 'mode_profile'),
    _FloquetPortEntry: ('name', 'position', 'axis', 'scan_theta', 'scan_phi', 'polarization', 'n_modes', 'freqs', 'n_freqs', 'f0', 'bandwidth', 'amplitude'),
    _MSLPortEntry: ('name', 'position', 'width', 'height', 'direction', 'impedance', 'waveform', 'excite', 'n_probe_offset', 'n_probe_spacing', 'n_probes', 'mode', 'eps_r_sub', 'terminates'),
    BoundaryModel: ('faces', 'axes', 'absorber_type', 'absorber_parameters', 'requirements', 'departures', 'mode', 'declaration', 'k_t'),
    Face: ('name', 'kind', 'origin', 'layers', 'conformal'),
    Axis: ('name', 'invariant', 'pairing', 'bloch'),
    Features: ('requirements', 'layers', 'absorber_parameters', 'explicit_faces', 'origin', 'face_origins', 'bloch_axes', 'domain'),
    Boundary: ('lo', 'hi', 'lo_thickness', 'hi_thickness', 'conformal'),
    BoundarySpec: ('x', 'y', 'z'),
    NTFFBox: ('i_lo', 'i_hi', 'j_lo', 'j_hi', 'k_lo', 'k_hi', 'freqs', 'cpml_lo_x', 'cpml_hi_x', 'cpml_lo_y', 'cpml_hi_y', 'cpml_lo_z', 'cpml_hi_z', 'face_centre', 'w_x_lo', 'w_x_hi', 'w_y_lo', 'w_y_hi', 'w_z_lo', 'w_z_hi'),
    Box: ('corner_lo', 'corner_hi'),
    OrientedBox: ('center', 'size', 'rotation'),
    PolylineWire: ('points', 'radius'),
    Sphere: ('center', 'radius'),
    Cylinder: ('center', 'radius', 'height', 'axis'),
    LumpedRLCSpec: ('R', 'L', 'C', 'topology', 'position', 'component'),
    DebyePole: ('delta_eps', 'tau'),
    LorentzPole: ('omega_0', 'delta', 'kappa'),
    PinnedSheet: ('normal_axis', 'plane_index', 'i_range', 'j_range', 'sigma_bulk', 'thickness', 'eps_r', 'surface_impedance_f0', 'name'),
    ThinConductor: ('shape', 'sigma_bulk', 'thickness', 'eps_r', 'surface_impedance_f0'),
    CoaxialPort: ('position', 'face', 'pin_length', 'pin_radius', 'outer_radius', 'impedance', 'excitation', 'terminates'),
    GaussianPulse: ('f0', 'bandwidth', 'amplitude', 'cutoff'),
    ModulatedGaussian: ('f0', 'bandwidth', 'amplitude', 'cutoff'),
    CWSource: ('f0', 'amplitude', 'ramp_steps'),
    CustomWaveform: ('func',),
    _TFSFEntry: ('f0', 'bandwidth', 'amplitude', 'margin', 'polarization', 'direction', 'angle_deg', 'waveform', 'method', 'closed_box', 'theta_deg', 'phi_deg', 'direction_tol_deg', 'max_component'),
}

def record_fields(cls):
    return tuple(f.name for f in fields(cls)) if is_dataclass(cls) else cls._fields


def _roles(cls, values=(), traceable=()):
    return {(cls, name): (VALUE if name in values else
                         STRUCTURE_TRACEABLE if name in traceable else STRUCTURE)
            for name in _SCHEMA[cls]}


ROLES = {}
for _cls, _values, _traceable in (
    (MaterialSpec, ("eps_r", "sigma", "mu_r", "chi3"), ()),
    (DebyePole, ("delta_eps", "tau"), ()),
    (LorentzPole, ("omega_0", "delta", "kappa"), ()),
    (_GeometryEntry, (), ()), (Box, (), ("corner_lo", "corner_hi")), (OrientedBox, (), ()), (PolylineWire, (), ()),
    (Sphere, (), ("radius",)), (Cylinder, (), ("radius", "height")),
    (_PortEntry, ("impedance",), ()),
    (_WaveguidePortEntry, ("f0", "bandwidth", "amplitude"), ()),
    (_FloquetPortEntry, ("f0", "bandwidth", "amplitude"), ()),
    (CoaxialPort, ("impedance",), ()), (_MSLPortEntry, ("impedance",), ()),
    (LumpedRLCSpec, ("R", "L", "C"), ()),
    (PinnedSheet, (), ()),  # Finding: PinnedSheet.eps_r accepted and ignored.
    # Finding: ThinConductor.eps_r accepted and ignored in f0 mode; DC refuses it.
    (ThinConductor, ("sigma_bulk", "thickness", "surface_impedance_f0"), ()),
    (_TFSFEntry, ("f0", "bandwidth", "amplitude", "waveform"), ()),
    (_ProbeEntry, (), ()), (_DFTPlaneEntry, (), ()), (_FluxMonitorEntry, (), ()),
    (NTFFBox, (), ()),
    (BoundaryModel, (), ()), (Face, (), ()), (Axis, (), ()), (Features, (), ()),
    (BoundarySpec, (), ()), (Boundary, (), ()),
    (GaussianPulse, ("f0", "bandwidth", "amplitude"), ()),
    (ModulatedGaussian, ("f0", "bandwidth", "amplitude"), ()),
    (CWSource, ("f0", "amplitude"), ()), (CustomWaveform, (), ()),
):
    ROLES.update(_roles(_cls, _values, _traceable))

COLLECTIONS = (
    "_materials", "_geometry", "_ports", "_waveguide_ports", "_floquet_ports",
    "_coaxial_ports", "_msl_ports", "_lumped_rlc", "_thin_conductors",
    "_pinned_sheets", "_tfsf", "_probes", "_dft_planes", "_flux_monitors",
    "_ntff", "_current_moments",
)
SIMULATION_STRUCTURE = (
    "_domain", "_dx", "_freq_max",
    "_dt_pin", "_dt_min_cell", "_nominal_profiles", "_cpml_layers", "_cpml_kappa_max",
    "_precision", "_mode", "_solver", "_adi_cfl_factor", "_stencil_order", "_snap", "_interface_eps",
    "_boundary", "_boundary_spec", "_boundary_model", "_boundary_explicit",
    "_periodic_axes", "_pec_faces", "_refinement", "_internal_probe_indices",
    "_msl_auto_offset_min", "_msl_auto_probe_spacing", "_msl_auto_probe_lengths",
)
# Cell widths enter the graded solver through the nominal-profile route.
SIMULATION_VALUE = ("_dx_profile", "_dy_profile", "_dz_profile")
SIMULATION_ROLES = {**dict.fromkeys(SIMULATION_STRUCTURE, STRUCTURE),
                    **dict.fromkeys(SIMULATION_VALUE, VALUE)}


def _traced(value):
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def refuse_staircase_box(shape):
    """Fence concrete Box rasterizers; the SDF route never calls this."""
    if isinstance(shape, Box):
        for field in ("corner_lo", "corner_hi"):
            if _traced(getattr(shape, field)):
                raise NotImplementedError(
                    f"Box.{field} is traced; the staircase mask needs concrete corners. "
                    "Use subpixel smoothing, design_box or eps_override.")


def no_effect_text(cls, name, mode=None):
    context = f" in this mode: {mode}" if mode else ""
    return (f"{cls.__name__}.{name} has no effect on the realized model{context}; "
            "a traced value would have a zero gradient. Declare it concrete.")


def refuse_thin_mode(record):
    """Conditional VALUE admission before constructors, entries and direct folds."""
    from rfx.core.jax_utils import is_tracer
    from rfx.materials.thin_conductor import _PEC_SIGMA_THRESHOLD
    f0, sigma = record.surface_impedance_f0, record.sigma_bulk
    mode = ("surface-impedance" if f0 is not None else
            "PEC" if not is_tracer(sigma) and sigma is not None and sigma >= _PEC_SIGMA_THRESHOLD
            else "DC lossy")
    if _traced(record.eps_r):
        text = no_effect_text(ThinConductor, "eps_r", mode)
        if mode == "DC lossy":
            from rfx.model.thin_conductors import DCFilmAdmissionError
            raise DCFilmAdmissionError(
                text + " a film on one node plane has no volume; its eps_r is not modelled",
                code="dc_film_permittivity")
        raise NotImplementedError(text)
    if mode != "DC lossy" and _traced(record.thickness):
        raise NotImplementedError(no_effect_text(ThinConductor, "thickness", mode))


def refuse_field(cls, name, value):
    """Refuse one direct structure operand before a constructor can convert it."""
    if _traced(value):
        if (cls is PinnedSheet and name in ("eps_r", "thickness", "sigma_bulk")) or (
                cls is ThinConductor and name == "eps_r") or (
                cls is _TFSFEntry and name == "direction_tol_deg"):
            raise NotImplementedError(no_effect_text(cls, name))
        if (cls is _TFSFEntry and name in ("theta_deg", "phi_deg")) or (
                cls is _FloquetPortEntry and name == "scan_phi"):
            raise NotImplementedError(
                f"{cls.__name__}.{name} changes the solved fields, but rfx has no traced route "
                "for it yet (rfx #1961). Declare it concrete.")
        shape_field = cls in (Box, OrientedBox, PolylineWire, Sphere, Cylinder) or (
            cls is _GeometryEntry and name == "shape")
        route = "use design_box / eps_override" if shape_field else "declare it concrete"
        detail = (" OrientedBox requires concrete geometry." if cls is OrientedBox else
                  " A gradient with respect to a declared shape dimension is not supported."
                  if shape_field else "")
        raise NotImplementedError(f"{cls.__name__}.{name} is STRUCTURE; {route}.{detail}")


def refuse_traced_structure(sim):
    """Walk only declarations, before dtype, mesh, admission, or realization."""
    def walk(value, owner, field):
        cls = type(value)
        names = record_fields(cls) if is_dataclass(value) or hasattr(cls, "_fields") else ()
        if names:
            if cls is ThinConductor:
                refuse_thin_mode(value)
            for name in names:
                leaf = getattr(value, name)
                role = ROLES.get((cls, name))
                if role == STRUCTURE and not (cls is ThinConductor and name == "eps_r"):
                    walk(leaf, cls, name)
                elif is_dataclass(leaf) or hasattr(type(leaf), "_fields"):
                    walk(leaf, cls, name)
                elif role is None:
                    # No inferred role for records absent from the leader's table.
                    refuse_field(cls, name, leaf)
        elif isinstance(value, dict):
            for leaf in value.values():
                walk(leaf, owner, field)
        elif isinstance(value, (tuple, list)):
            for leaf in value:
                walk(leaf, owner, field)
        else:
            refuse_field(owner, field, value)

    declaration = vars(sim)
    for name in SIMULATION_STRUCTURE:
        # Mesh descriptors store their declarations under _unresolved_*.
        key = "_unresolved" + name if "_unresolved" + name in declaration else name
        if key in declaration:
            walk(declaration[key], type(sim), name)
    for name in COLLECTIONS:
        if name in declaration:
            walk(declaration[name], type(sim), name)


def guard_declaration_calls(simulation):
    """Apply the same registry before declaration constructors validate values.

    Some declarations convert structure operands while being constructed (node
    indices, boundary thicknesses). Entry refusal alone is too late for them.
    These guards inspect call operands only, never realized or cached state.
    """
    def guard_value(value, owner, field):
        if is_dataclass(value) or hasattr(type(value), "_fields"):
            cls = type(value)
            if cls is ThinConductor:
                refuse_thin_mode(value)
            for name in record_fields(cls):
                if ROLES.get((cls, name)) == STRUCTURE and not (cls is ThinConductor and name == "eps_r"):
                    guard_value(getattr(value, name), cls, name)
            return
        if isinstance(value, dict):
            for leaf in value.values():
                guard_value(leaf, owner, field)
        elif isinstance(value, (list, tuple)):
            for leaf in value:
                guard_value(leaf, owner, field)
        else:
            refuse_field(owner, field, value)

    def guarded(call, owner, aliases=None):
        signature = inspect.signature(call)

        @functools.wraps(call)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            if owner is ThinConductor:
                from types import SimpleNamespace
                bound.apply_defaults()
                refuse_thin_mode(SimpleNamespace(**{name: bound.arguments[name] for name in
                    ("sigma_bulk", "thickness", "eps_r", "surface_impedance_f0")}))
            for name, value in bound.arguments.items():
                field = (aliases or {}).get(name, name)
                if owner is simulation and field == "_nominal_profiles":
                    # Main's constructor already refuses traced nominal widths
                    # with its axis-specific ValueError; retain that validation.
                    continue
                role = (SIMULATION_ROLES.get(field) if owner is simulation
                        else ROLES.get((owner, field)))
                if role == STRUCTURE and not (owner is ThinConductor and field == "eps_r"):
                    # Registered nested records own the roles of their leaves.
                    guard_value(value, owner, field)
                elif role in (VALUE, STRUCTURE_TRACEABLE) and (
                        is_dataclass(value) or hasattr(type(value), "_fields")):
                    guard_value(value, owner, field)
            # Declaration-only constants (frequency bins, geometry arrays)
            # stay concrete even when the caller constructs a model in jit.
            with jax.ensure_compile_time_eval():
                return call(*args, **kwargs)
        return wrapper

    for cls in dict.fromkeys(cls for cls, _ in ROLES):
        if cls in (PinnedSheet, BoundarySpec, Boundary, BoundaryModel, Face, Axis, Features, GaussianPulse, OrientedBox):
            cls.__init__ = guarded(cls.__init__, cls)
    simulation.__init__ = guarded(simulation.__init__, simulation,
                                  {**{name.lstrip("_"): name for name in SIMULATION_ROLES}, "dt": "_dt_pin"})
    for method, cls, aliases in (
        ("add", _GeometryEntry, {"material": "material_name"}),
        ("add_source", _PortEntry, {}), ("add_port", _PortEntry, {}),
        ("add_probe", _ProbeEntry, {}), ("add_dft_plane_probe", _DFTPlaneEntry, {}),
        ("add_flux_monitor", _FluxMonitorEntry, {}),
        ("add_waveguide_port", _WaveguidePortEntry, {}),
        ("add_floquet_port", _FloquetPortEntry, {}),
        ("add_coaxial_port", CoaxialPort, {}), ("add_msl_port", _MSLPortEntry, {}),
        ("add_lumped_rlc", LumpedRLCSpec, {}),
        ("add_thin_conductor", ThinConductor, {}), ("add_pinned_sheet", PinnedSheet, {}),
        ("add_tfsf_source", _TFSFEntry, {}),
        ("add_ntff_box", NTFFBox, {"corner_lo": "i_lo", "corner_hi": "i_hi", "n_freqs": "freqs"}),
    ):
        setattr(simulation, method, guarded(getattr(simulation, method), cls, aliases))
