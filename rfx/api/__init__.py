"""Agent-friendly high-level simulation API.

Provides a declarative ``Simulation`` builder that wraps the low-level
functional primitives.  Designed so that AI agents (and RF engineers)
can construct simulations from natural-language-like descriptions
without touching grid indices, NamedTuples, or JAX internals.

Usage
-----
>>> sim = Simulation(freq_max=10e9, domain=(0.05, 0.05, 0.025))
>>> sim.add_material("substrate", eps_r=4.4, sigma=0.02)
>>> sim.add(Box((0, 0, 0), (0.05, 0.05, 0.001)), material="substrate")
>>> sim.add_port((0.01, 0.025, 0.001), "ez", impedance=50, waveform=GaussianPulse(f0=5e9))
>>> sim.add_probe((0.04, 0.025, 0.001), "ez")
>>> result = sim.run()
>>> result.s_params   # (n_ports, n_ports, n_freqs) complex
"""

from __future__ import annotations

import math
import jax
from numbers import Integral

import jax.numpy as jnp
import numpy as np


from rfx.grid import Grid, C0  # noqa: F401
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box, Shape  # noqa: F401
from rfx.nonuniform import NonUniformGrid  # noqa: F401
from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import CoaxialPort
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.lumped import LumpedRLCSpec
from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor  # noqa: F401
from rfx.sources.waveguide_port import (
    WaveguidePort,  # noqa: F401
    extract_waveguide_s_matrix,  # noqa: F401
    extract_waveguide_s_matrix_flux,  # noqa: F401
    extract_waveguide_s_params_normalized,  # noqa: F401
    init_waveguide_port,  # noqa: F401
    init_multimode_waveguide_port,  # noqa: F401
    extract_multimode_s_matrix,  # noqa: F401
    waveguide_plane_positions,  # noqa: F401
)
from rfx.boundaries.spec import BoundarySpec

# ---------------------------------------------------------------------------
# Leaf data structures — moved to rfx/api/_spec.py (Part B Stage 0).
# `_spec.py` is a leaf module; never import Simulation back into it.
# Re-exported here so `from rfx.api import <name>` keeps working.
# ---------------------------------------------------------------------------

from rfx.api._spec import (  # noqa: E402
    MATERIAL_LIBRARY,
    AD_MemoryEstimate,
    ADMemoryPlan,
    ADMemoryComponent,
    ADMemoryExplainabilityReport,
    MeshIntelligenceReport,
    Result,
    ForwardResult,
    MaterialSpec,
    _GeometryEntry,
    _PortEntry,
    _ProbeEntry,
    _TFSFEntry,
    _DFTPlaneEntry,
    _FluxMonitorEntry,
    _WaveguidePortEntry,
    _FloquetPortEntry,
    WaveguideSParamResult,
    WaveguideSMatrixResult,
    CoaxialSMatrixResult,
    CoaxialLineReflectionResult,
    _MSLPortEntry,
    MSLSMatrixResult,
)
from rfx.mesh_planner import MeshPlan, plan_simulation_mesh  # noqa: E402,F401

def _require_integral_param(name: str, value: object) -> int:
    """Return ``value`` as int after rejecting bools and non-integral values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _require_positive_finite_scalar(name: str, value: object) -> float:
    """Return ``value`` as a finite positive Python float."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite real scalar, got {value!r}")
    arr = np.asarray(value)
    if arr.shape != ():
        raise TypeError(
            f"{name} must be a finite real scalar, got array shape {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(
        arr.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must be a finite real scalar, got {value!r}")
    out = float(arr.item())
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _positive_divisors(value: int) -> list[int]:
    """Return positive divisors of ``value`` sorted ascending."""
    small: list[int] = []
    large: list[int] = []
    root = math.isqrt(value)
    for divisor in range(1, root + 1):
        if value % divisor == 0:
            small.append(divisor)
            paired = value // divisor
            if paired != divisor:
                large.append(paired)
    return small + large[::-1]

AD_MEMORY_FIT_SAFETY_FACTOR = 1.30


def _format_memory_gb(value: float) -> str:
    """Render GB values without hiding sub-10MB estimates as ``0.00 GB``."""
    if value < 0.01:
        return f"{value * 1000.0:.1f} MB"
    return f"{value:.2f} GB"


def _format_memory_with_safety(value: float, safety_factor: float) -> str:
    """Render raw and safety-adjusted memory when the safety factor is active."""
    if safety_factor == 1.0:
        return _format_memory_gb(value)
    return (
        f"{_format_memory_gb(value)} "
        f"({_format_memory_gb(value * safety_factor)} with {safety_factor:.2f}x safety)"
    )

# ---------------------------------------------------------------------------
# Preflight / validation methods — moved to rfx/api/_preflight.py
# (Part B Stage 1a). `_preflight.py` is a transitional mixin importing only
# `_spec` + external rfx.*/stdlib/jax/numpy; never import Simulation into it.
# ---------------------------------------------------------------------------

from rfx.api._preflight import _PreflightMixin  # noqa: E402

# ---------------------------------------------------------------------------
# S-parameter extraction methods — moved to rfx/api/_sparams.py
# (Part B Stage 2). `_sparams.py` is a transitional mixin importing only
# `_spec` + external rfx.*/stdlib/jax/numpy; never import Simulation into it.
# ---------------------------------------------------------------------------

from rfx.api._sparams import _SparamMixin  # noqa: E402
from rfx.api._compile import _CompileMixin  # noqa: E402

# ---------------------------------------------------------------------------
# Execute methods (forward / run dispatch + per-path runners) — moved to
# rfx/api/_execute.py (Part B Stage 4, final). `_execute.py` is a transitional
# mixin importing only `_spec` + external rfx.*/stdlib/jax/numpy; never import
# Simulation into it. With this stage the God class is fully dissolved into a
# 5-module package facade (_spec, _compile, _preflight, _sparams, _execute).
# ---------------------------------------------------------------------------

from rfx.api._execute import _ExecuteMixin  # noqa: E402
from rfx.api._artifacts import _ArtifactsMixin  # noqa: E402


_DebyeSpec = tuple[list[DebyePole], list[jnp.ndarray]]
_LorentzSpec = tuple[list[LorentzPole], list[jnp.ndarray]]


# ---------------------------------------------------------------------------
# Simulation builder
# ---------------------------------------------------------------------------

class Simulation(
    _PreflightMixin,
    _SparamMixin,
    _CompileMixin,
    _ExecuteMixin,
    _ArtifactsMixin,
):
    """Declarative FDTD simulation builder.

    Parameters
    ----------
    freq_max : float
        Maximum simulation frequency (Hz).
    domain : (Lx, Ly, Lz) in metres
        Physical domain size.  For 2D modes Lz is ignored.
    boundary : "pec", "cpml", or "upml"
        Boundary condition. Default "cpml".
    cpml_layers : int
        Number of CPML layers per face. Default 12 (ignored for "pec").
    dx : float or None
        Cell size override (metres). Auto-computed if None.
    mode : str
        ``"3d"`` (default), ``"2d_tmz"`` (Ez, Hx, Hy), or
        ``"2d_tez"`` (Hz, Ex, Ey).
    precision : str
        ``"float32"`` (default) or ``"mixed"``.  When ``"mixed"``,
        field arrays (E, H) use float16 for ~2x memory reduction while
        material coefficients and DFT accumulators stay float32.
        Arithmetic is performed in float32 and cast back to float16
        for storage.
    solver : str
        ``"yee"`` (default) for the standard explicit scheme or
        ``"adi"`` for the current 2D TMz ADI-FDTD path.
    adi_cfl_factor : float
        Timestep multiplier relative to the standard 2D CFL limit when
        ``solver="adi"``.
    """

    def __init__(
        self,
        freq_max: float,
        domain: tuple[float, float, float],
        *,
        boundary: str | BoundarySpec | dict = "cpml",
        cpml_layers: int = 16,
        cpml_kappa_max: float = 1.0,
        pec_faces: set[str] | list[str] | None = None,
        dx: float | None = None,
        mode: str = "3d",
        dz_profile: np.ndarray | None = None,
        dx_profile: np.ndarray | None = None,
        dy_profile: np.ndarray | None = None,
        precision: str = "float32",
        solver: str = "yee",
        adi_cfl_factor: float = 5.0,
    ):
        from rfx.boundaries.spec import normalize_boundary

        # T7-B: accept BoundarySpec directly or normalise a legacy scalar
        # boundary=<str>. A BoundarySpec provided here is authoritative;
        # concurrent legacy kwargs (pec_faces) conflict with it.
        _explicit_spec = isinstance(boundary, (BoundarySpec, dict))
        if _explicit_spec:
            if pec_faces is not None:
                raise ValueError(
                    "pec_faces= cannot be combined with a BoundarySpec "
                    "boundary= argument; encode PEC faces inside the "
                    "BoundarySpec (e.g. z=Boundary(lo='pec', hi='cpml'))."
                )
            spec = normalize_boundary(boundary)
        else:
            # Legacy scalar path — validated below, lifted to BoundarySpec
            # after pec_faces / set_periodic_axes() have been resolved.
            if boundary not in ("pec", "cpml", "upml"):
                raise ValueError(
                    f"boundary must be 'pec', 'cpml', or 'upml', got {boundary!r}"
                )
            spec = None  # deferred; folded in after legacy fields settle
        if freq_max <= 0:
            raise ValueError(f"freq_max must be positive, got {freq_max}")
        if precision not in ("float32", "mixed"):
            raise ValueError(f"precision must be 'float32' or 'mixed', got {precision!r}")
        if solver not in ("yee", "adi"):
            raise ValueError(f"solver must be 'yee' or 'adi', got {solver!r}")
        if adi_cfl_factor <= 0:
            raise ValueError(f"adi_cfl_factor must be positive, got {adi_cfl_factor}")
        # Synthesize domain extents from axis profiles before validation.
        # dx_profile / dy_profile set x / y; dz_profile sets z.
        # Tracer-valued profiles require the caller to supply a concrete
        # domain extent for that axis since the profile sum cannot be
        # host-coerced during tracing.
        if dx_profile is not None:
            if is_tracer(dx_profile):
                if len(domain) < 1 or domain[0] <= 0:
                    raise ValueError(
                        "dx_profile is a JAX tracer; provide a concrete "
                        "domain[0] (profile sum cannot be host-coerced)."
                    )
            else:
                domain = (float(np.sum(dx_profile)),
                          domain[1] if len(domain) >= 2 else 0.0,
                          domain[2] if len(domain) >= 3 else 0.0)
        if dy_profile is not None:
            if is_tracer(dy_profile):
                if len(domain) < 2 or domain[1] <= 0:
                    raise ValueError(
                        "dy_profile is a JAX tracer; provide a concrete "
                        "domain[1] (profile sum cannot be host-coerced)."
                    )
            else:
                domain = (domain[0],
                          float(np.sum(dy_profile)),
                          domain[2] if len(domain) >= 3 else 0.0)
        if dz_profile is not None:
            if any(d <= 0 for d in domain[:2]):
                raise ValueError(f"domain x/y must be positive, got {domain}")
            if is_tracer(dz_profile):
                if len(domain) < 3 or domain[2] <= 0:
                    raise ValueError(
                        "dz_profile is a JAX tracer; provide a concrete "
                        "domain=(..., ..., dz_total) extent (the profile "
                        "sum cannot be host-coerced during tracing)."
                    )
            else:
                dz_total = float(np.sum(dz_profile))
                if len(domain) < 3 or domain[2] <= 0:
                    domain = (domain[0], domain[1], dz_total)
        elif any(d <= 0 for d in domain):
            raise ValueError(f"domain dimensions must be positive, got {domain}")

        # P2: Warn on abrupt grading in user-supplied dz_profile.
        # Tracer profiles skip the warning — adjacent ratios can't be
        # computed host-side during tracing.
        if (dz_profile is not None and not is_tracer(dz_profile)
                and len(dz_profile) > 1):
            import warnings as _w
            ratios = np.array(dz_profile[1:]) / np.array(dz_profile[:-1])
            max_ratio = float(np.max(np.maximum(ratios, 1.0 / ratios)))
            if max_ratio > 1.3 + 1e-6:
                _w.warn(
                    f"dz_profile has max adjacent cell ratio {max_ratio:.2f} "
                    f"(> 1.3). This may cause numerical reflections. "
                    f"Use rfx.smooth_grading(dz_profile) to fix.",
                    stacklevel=2,
                )

        _valid_faces = {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}
        if _explicit_spec:
            # BoundarySpec is authoritative; derive the legacy views so
            # downstream code that has not migrated continues to work.
            self._pec_faces = spec.pec_faces()
            legacy_boundary = spec.absorber_type
            if legacy_boundary is None:
                # No absorbing face: pick 'pec' (matches historic all-PEC).
                legacy_boundary = "pec"
            boundary = legacy_boundary  # feed the rest of __init__
        else:
            if pec_faces is not None:
                import warnings as _w
                _w.warn(
                    "pec_faces= kwarg is deprecated; encode PEC faces in "
                    "BoundarySpec instead (e.g. "
                    "boundary=BoundarySpec(x='cpml', y='cpml', "
                    "z=Boundary(lo='pec', hi='cpml'))). The kwarg will be "
                    "removed in rfx v2.0.",
                    DeprecationWarning, stacklevel=2,
                )
            self._pec_faces = set(pec_faces) if pec_faces else set()
        if self._pec_faces - _valid_faces:
            raise ValueError(
                f"pec_faces must be subset of {_valid_faces}, "
                f"got invalid: {self._pec_faces - _valid_faces}")
        # Legacy guard: pec_faces= kwarg alongside boundary="pec" is
        # redundant in the pre-BoundarySpec API. The explicit-spec path
        # (T7) bypasses this — a BoundarySpec with z=Boundary(lo='pmc',
        # hi='pec') legitimately derives self._pec_faces={"z_hi"} even
        # when the effective scalar boundary is 'pec' (no cpml/upml
        # face anywhere).
        if not _explicit_spec and boundary == "pec" and self._pec_faces:
            raise ValueError("pec_faces is only meaningful with boundary='cpml' or boundary='upml'")

        # Non-uniform xy profiles require an explicit dx (boundary cell
        # size) so the CPML profiles have a defined edge spacing.
        if (dx_profile is not None or dy_profile is not None) and dx is None:
            raise ValueError("dx_profile / dy_profile require an explicit dx (boundary cell size)")

        self._freq_max = freq_max
        self._domain = domain
        self._boundary = boundary
        self._cpml_layers = cpml_layers if boundary in ("cpml", "upml") else 0
        self._cpml_kappa_max = cpml_kappa_max
        self._dx = dx
        self._mode = mode
        self._dz_profile = dz_profile
        self._dx_profile = dx_profile
        self._dy_profile = dy_profile
        self._precision = precision
        self._solver = solver
        self._adi_cfl_factor = adi_cfl_factor

        if self._solver == "adi":
            if self._mode not in ("2d_tmz", "3d"):
                raise ValueError("solver='adi' supports mode='3d' or mode='2d_tmz'")
            if self._boundary == "upml":
                raise ValueError("solver='adi' does not support boundary='upml'")
            if self._boundary not in ("pec", "cpml"):
                raise ValueError("solver='adi' supports boundary='pec' or 'cpml'")
            if self._dz_profile is not None:
                raise ValueError("solver='adi' does not support nonuniform dz_profile")
            if self._dx_profile is not None or self._dy_profile is not None:
                raise ValueError("solver='adi' does not support nonuniform dx/dy profile")

        # Registered items
        self._materials: dict[str, MaterialSpec] = {}
        self._geometry: list[_GeometryEntry] = []
        self._ports: list[_PortEntry] = []
        self._probes: list[_ProbeEntry] = []
        self._thin_conductors: list[ThinConductor] = []
        self._coaxial_ports: list[CoaxialPort] = []
        # Terminations applied after setup_coaxial_port: each entry is
        # (port_index, target_impedance, axial_offset_cells). Stamped during
        # compute_coaxial_s_matrix's materials build for the targeted port.
        self._coaxial_terminations: list[tuple[int, float, int]] = []
        # Open-circuit terminations: each entry is (port_index,
        # pin_retract_cells). Retracts the inner pin to create a circular-
        # waveguide-like open termination beyond the new pin tip.
        self._coaxial_open_terminations: list[tuple[int, int]] = []
        # PEC end-cap closures: each entry is (port_index, axial_offset_cells).
        # Closes the outer shell with a PEC disk to isolate the line from the
        # surrounding cavity (used together with open termination).
        self._coaxial_pec_end_caps: list[tuple[int, int]] = []
        self._ntff: tuple | None = None  # (corner_lo, corner_hi, freqs)
        self._tfsf: _TFSFEntry | None = None
        self._dft_planes: list[_DFTPlaneEntry] = []
        self._flux_monitors: list[_FluxMonitorEntry] = []
        self._waveguide_ports: list[_WaveguidePortEntry] = []
        self._msl_ports: list[_MSLPortEntry] = []
        self._periodic_axes: str = ""
        self._refinement: dict | None = None
        self._lumped_rlc: list[LumpedRLCSpec] = []
        self._floquet_ports: list[_FloquetPortEntry] = []

        # T7-B: canonical BoundarySpec. When the caller supplies a
        # BoundarySpec directly it is authoritative; otherwise compose
        # one from the legacy triad (scalar boundary + pec_faces +
        # periodic_axes='' at construction). set_periodic_axes() and any
        # future legacy mutation rebuilds via _build_spec_from_legacy.
        if _explicit_spec:
            self._boundary_spec = spec
            self._periodic_axes = spec.periodic_axes()
        else:
            self._boundary_spec = self._build_spec_from_legacy()

        # T7 Phase 2 PR2: per-face CPML thickness now runs end-to-end
        # via the padded-profile engine in rfx/boundaries/cpml.py.
        # The Phase 1 guard _check_thickness_uniformity_phase1 has
        # been removed; Boundary.lo_thickness / .hi_thickness are
        # consumed by _build_grid → Grid.face_layers → init_cpml.

        # T7-E Phase 2 PR3: PMC runtime lands in rfx/boundaries/pmc.py
        # and is hooked into the uniform scan body after each H update
        # (rfx/simulation.py). The Phase 1 construction guard has been
        # removed; PMC faces flow through Grid.pmc_faces and are applied
        # in-trace.

    # ---- refinement (subgridding) ----

    def add_refinement(
        self,
        z_range: tuple[float, float],
        *,
        ratio: int = 4,
        xy_margin: float | None = None,
        tau: float = 0.5,
        validation: str = "production",
        topology: str = "overlap_z_slab",
    ) -> "Simulation":
        """Add a z-axis refinement region for SBP-SAT subgridding.

        The promoted production runner covers the specified z-range across the
        full x/y interior at ``dx_fine = dx_coarse / ratio``.  ``xy_margin``
        enables an experimental research-only local x/y window whose fine
        region is inset from the physical x/y boundaries by that distance.

        Parameters
        ----------
        z_range : (z_lo, z_hi) in metres
            Physical z-range for the fine region.
        ratio : int
            Refinement ratio (fine cells per coarse cell). Default 4.
        xy_margin : float or None
            Experimental x/y inset in metres.  ``None`` keeps the full x/y
            interior.  A finite non-negative value creates a local window
            spanning ``[xy_margin, Lx - xy_margin]`` and
            ``[xy_margin, Ly - xy_margin]``.  Production validation still
            rejects this lane until waveform gates and crossval pass.
        tau : float
            SAT penalty coefficient (default 0.5). Higher values give
            stronger coupling but more dissipation.
        validation : {"production", "research", "off"}
            Validation envelope for the run path. ``"production"`` rejects
            unsupported material/interface/source configurations before FDTD
            execution. ``"research"`` preserves the experimental legacy lane
            for internal diagnostics. ``"off"`` is reserved for low-level
            debugging and should not be used for claims-bearing results.
        topology : {"overlap_z_slab", "stage2_disjoint_3d"}
            Internal topology selector. ``"overlap_z_slab"`` is the current
            public runner. ``"stage2_disjoint_3d"`` records the selected
            centered/two-interface integration lane but remains a research-only
            contract until its public runner wiring and waveform gates pass.
        """
        if self._refinement is not None:
            raise ValueError("Only one refinement region is supported")
        if validation not in {"production", "research", "off"}:
            raise ValueError(
                "validation must be one of 'production', 'research', or 'off'"
            )
        if topology not in {"overlap_z_slab", "stage2_disjoint_3d"}:
            raise ValueError(
                "topology must be one of 'overlap_z_slab' or "
                "'stage2_disjoint_3d'"
            )
        # Warn if subgrid overlaps PML region.
        # PML operates on the coarse grid only; the fine grid has no PML.
        # Overlapping causes late-time energy growth (SAT coupling feeds
        # energy into the PML boundary faster than it can absorb).
        if self._boundary in ("cpml", "upml") and self._cpml_layers > 0:
            import warnings
            dx = self._dx or (2.998e8 / self._freq_max / 10)
            face_layers = self._resolve_face_layers()
            z_lo_kind = self._boundary_spec.z.lo if self._boundary_spec else self._boundary
            z_hi_kind = self._boundary_spec.z.hi if self._boundary_spec else self._boundary
            pml_zlo_thickness = (
                face_layers.get("z_lo", self._cpml_layers) * dx
                if z_lo_kind in ("cpml", "upml")
                else 0.0
            )
            pml_zhi_thickness = (
                face_layers.get("z_hi", self._cpml_layers) * dx
                if z_hi_kind in ("cpml", "upml")
                else 0.0
            )
            domain_z = self._domain[2] if len(self._domain) > 2 else 0
            z_lo, z_hi = z_range
            if z_lo < pml_zlo_thickness or (
                domain_z > 0 and z_hi > domain_z - pml_zhi_thickness
            ):
                warnings.warn(
                    f"Subgrid z_range=({z_lo*1e3:.1f}, {z_hi*1e3:.1f})mm overlaps "
                    f"PML region (zlo={pml_zlo_thickness*1e3:.1f}mm, "
                    f"zhi={pml_zhi_thickness*1e3:.1f}mm). "
                    f"This causes late-time energy growth. "
                    f"Move z_range inside the PML boundary for stable results.",
                    stacklevel=2,
                )

        self._refinement = {
            "z_range": z_range,
            "ratio": ratio,
            "xy_margin": xy_margin,
            "tau": tau,
            "validation": validation,
            "topology": topology,
        }
        return self

    def validate_subgrid(self, *, mode: str | None = None):
        """Return the production-envelope validation report for subgridding.

        This is a physics-support report, not a numerical smoke test.  It checks
        whether the configured refinement lies inside the currently derived
        guarded one-sided z-slab support envelope: static materials,
        source/probe or single-cell lumped-port observables, no
        material/PEC discontinuity at artificial coarse/fine interfaces, and
        no unsupported RF post-processing features.
        """
        if self._refinement is None:
            from rfx.subgridding.validation import validate_subgrid_setup
            grid = self._build_grid()
            mats, _, _, pec_mask, *_ = self._assemble_materials(grid)
            return validate_subgrid_setup(
                self, grid, mats, pec_mask, mode=mode or "production",
            )
        grid = self._build_grid()
        mats, _, _, pec_mask, *_ = self._assemble_materials(grid)
        from rfx.subgridding.validation import validate_subgrid_setup
        return validate_subgrid_setup(
            self,
            grid,
            mats,
            pec_mask,
            mode=mode or self._refinement.get("validation", "production"),
        )

    # ---- material registration ----

    def add_material(
        self,
        name: str,
        *,
        eps_r: float = 1.0,
        sigma: float = 0.0,
        mu_r: float = 1.0,
        debye_poles: list[DebyePole] | None = None,
        lorentz_poles: list[LorentzPole] | None = None,
        chi3: float = 0.0,
    ) -> "Simulation":
        """Register a named material.

        Parameters
        ----------
        chi3 : float
            Third-order (Kerr) susceptibility in m^2/V^2.  When
            non-zero, an ADE correction is applied after each E-field
            update inside the scan body.
        """
        self._materials[name] = MaterialSpec(
            eps_r=eps_r, sigma=sigma, mu_r=mu_r,
            debye_poles=debye_poles, lorentz_poles=lorentz_poles,
            chi3=chi3,
        )
        return self

    def _resolve_material(self, name: str) -> MaterialSpec:
        """Look up a material by name (user-defined first, then library)."""
        if name in self._materials:
            return self._materials[name]
        if name in MATERIAL_LIBRARY:
            lib = MATERIAL_LIBRARY[name]
            return MaterialSpec(
                eps_r=lib.get("eps_r", 1.0),
                sigma=lib.get("sigma", 0.0),
                mu_r=lib.get("mu_r", 1.0),
                debye_poles=lib.get("debye_poles"),
                lorentz_poles=lib.get("lorentz_poles"),
            )
        raise KeyError(
            f"Unknown material {name!r}. "
            f"Register with add_material() or use a library name: "
            f"{list(MATERIAL_LIBRARY.keys())}"
        )

    # ---- geometry ----

    def add(self, shape: Shape, *, material: str) -> "Simulation":
        """Add a geometric shape filled with a named material."""
        self._resolve_material(material)  # validate early
        self._geometry.append(_GeometryEntry(shape=shape, material_name=material))
        return self

    # ---- sources (non-port) ----

    def add_source(
        self,
        position: tuple[float, float, float],
        component: str = "ez",
        *,
        waveform=None,
    ) -> "Simulation":
        """Add a soft point source (no impedance loading).

        Unlike ``add_port()``, this does NOT add a resistive load.
        Ideal for resonance characterization where port loading would
        damp the cavity response.

        Uses ModulatedGaussian by default (zero DC content, like Meep).
        This prevents static charge accumulation on PEC surfaces.

        Parameters
        ----------
        position : (x, y, z) in metres
        component : "ex", "ey", or "ez"
        waveform : excitation pulse (default: ModulatedGaussian at freq_max/2)
        """
        if component not in ("ex", "ey", "ez"):
            raise ValueError(f"component must be ex/ey/ez, got {component!r}")
        if waveform is None:
            waveform = GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)

        # Store as a source entry (reuse _PortEntry with impedance=None flag)
        self._ports.append(_PortEntry(
            position=position, component=component,
            impedance=0.0,  # 0 = no port impedance (soft source)
            waveform=waveform, extent=None,
        ))
        return self

    def add_polarized_source(
        self,
        position: tuple[float, float, float],
        *,
        polarization: str | tuple = "ez",
        waveform=None,
    ) -> "Simulation":
        """Add a polarized point source.

        Parameters
        ----------
        position : (x, y, z) in metres
        polarization : str or tuple
            - "ez", "ex", "ey" — linear single-component
            - "circular" or "rhcp" — right-hand circular (Ex + jEy)
            - "lhcp" — left-hand circular (Ex - jEy)
            - (Ex, Ey) tuple — arbitrary Jones vector (complex)
            - "slant45" — 45° linear (Ex = Ey)
        waveform : excitation pulse (default: GaussianPulse)
        """
        if waveform is None:
            waveform = GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)

        if isinstance(polarization, str):
            if polarization in ("ex", "ey", "ez"):
                self.add_source(position, polarization, waveform=waveform)
                return self
            pol_map = {
                "circular": (1.0, 1j),
                "rhcp": (1.0, 1j),
                "lhcp": (1.0, -1j),
                "slant45": (1.0, 1.0),
            }
            if polarization not in pol_map:
                raise ValueError(f"Unknown polarization: {polarization!r}")
            jones = pol_map[polarization]
        else:
            jones = tuple(polarization)

        # Normalize Jones vector
        import numpy as _np
        norm = _np.sqrt(abs(jones[0])**2 + abs(jones[1])**2)
        if norm < 1e-30:
            raise ValueError("Jones vector magnitude is zero")
        jx, jy = jones[0] / norm, jones[1] / norm

        # For complex Jones (circular): use two sources with phase-shifted waveforms
        if _np.isreal(jx) and _np.isreal(jy):
            # Real Jones — simple amplitude scaling
            if abs(float(jx.real)) > 1e-10:
                from rfx.sources.sources import GaussianPulse as GP
                wx = GP(f0=waveform.f0, bandwidth=waveform.bandwidth,
                        amplitude=waveform.amplitude * float(jx.real))
                self.add_source(position, "ex", waveform=wx)
            if abs(float(jy.real)) > 1e-10:
                wy = GP(f0=waveform.f0, bandwidth=waveform.bandwidth,
                        amplitude=waveform.amplitude * float(jy.real))
                self.add_source(position, "ey", waveform=wy)
        else:
            # Complex Jones — use CW-modulated source for phase control
            from rfx.sources.sources import ModulatedGaussian as MG
            # Ex component (reference phase)
            if abs(jx) > 1e-10:
                wx = MG(f0=waveform.f0, bandwidth=waveform.bandwidth,
                        amplitude=waveform.amplitude * float(abs(jx)))
                self.add_source(position, "ex", waveform=wx)
            # Ey component (90° phase shift for circular)
            if abs(jy) > 1e-10:
                # Phase of jy relative to jx
                import cmath
                phase = cmath.phase(jy) - cmath.phase(jx) if abs(jx) > 1e-10 else 0
                # For ±90° (circular): use cosine instead of sine carrier
                from rfx.sources.sources import CustomWaveform
                import jax.numpy as _jnp
                tau = 1.0 / (waveform.f0 * waveform.bandwidth * 3.14159)
                t0 = 3.0 * tau
                amp_y = waveform.amplitude * float(abs(jy))
                def _ey_func(t):
                    envelope = _jnp.exp(-((t - t0) / tau)**2)
                    carrier = _jnp.cos(2.0 * _jnp.pi * waveform.f0 * t + float(phase))
                    return amp_y * carrier * envelope
                self.add_source(position, "ey", waveform=CustomWaveform(func=_ey_func))

        return self

    # ---- coaxial ports ----

    def add_coaxial_port(
        self,
        position: tuple[float, float, float],
        face: str = "top",
        *,
        pin_length: float = 5e-3,
        pin_radius: float = 0.635e-3,
        outer_radius: float = 2.055e-3,
        impedance: float = 50.0,
        waveform=None,
    ) -> "Simulation":
        """Add an SMA-style coaxial probe port.

        Parameters
        ----------
        position : (x, y, z) — center of port on cavity wall (metres)
        face : "top", "bottom", "front", "back", "left", "right"
        pin_length : float — pin protrusion into cavity (default 5mm)
        pin_radius : float — center pin radius (default 0.635mm, SMA)
        outer_radius : float — outer conductor radius (default 2.055mm)
        impedance : float — port impedance (default 50 ohm)
        waveform : excitation pulse (default GaussianPulse at freq_max/2)
        """
        if waveform is None:
            waveform = GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)
        self._coaxial_ports.append(CoaxialPort(
            position=position,
            face=face,
            pin_length=pin_length,
            pin_radius=pin_radius,
            outer_radius=outer_radius,
            impedance=impedance,
            excitation=waveform,
        ))
        return self

    def add_coaxial_pec_end_cap(
        self,
        port_index: int = 0,
        *,
        axial_offset_cells: int = 0,
    ) -> "Simulation":
        """Close the outer shell of a coaxial port with a PEC end-cap.

        Stamps a PEC disk one cell past the original shell tip in the
        forward direction, isolating the coax line from the surrounding
        cavity. Combined with :meth:`add_coaxial_open_termination`,
        this completes a proper open-circuit cup geometry.

        Parameters
        ----------
        port_index : int
        axial_offset_cells : int
            Optional offset from the default (one cell past the shell
            tip). Negative moves the cap into the line.
        """
        if port_index < 0 or port_index >= len(self._coaxial_ports):
            raise ValueError(
                f"port_index {port_index} is out of range "
                f"(have {len(self._coaxial_ports)} coaxial ports registered)"
            )
        self._coaxial_pec_end_caps.append(
            (int(port_index), int(axial_offset_cells))
        )
        return self

    def add_coaxial_open_termination(
        self,
        port_index: int = 0,
        *,
        pin_retract_cells: int = 1,
    ) -> "Simulation":
        """Register an open-circuit termination on a coaxial port.

        Retracts the inner pin by ``pin_retract_cells`` Yee cells from
        its original ``pin_length`` end. The resulting cross-section
        beyond the new pin tip is a PTFE-filled circular waveguide whose
        TE/TM modes have cutoff frequencies far above the design band,
        so the wave reaching the step is below cutoff and reflects
        evanescently — an open-circuit-like termination with ``|Γ| ≈ 1``.

        Parameters
        ----------
        port_index : int
            Index into the order of ``add_coaxial_port`` calls.
        pin_retract_cells : int
            Number of Yee cells to remove from the pin's far end.
            Default 1 cuts back exactly one cell.
        """

        if port_index < 0 or port_index >= len(self._coaxial_ports):
            raise ValueError(
                f"port_index {port_index} is out of range "
                f"(have {len(self._coaxial_ports)} coaxial ports registered)"
            )
        self._coaxial_open_terminations.append(
            (int(port_index), int(pin_retract_cells))
        )
        return self

    def add_coaxial_matched_load(
        self,
        port_index: int = 0,
        *,
        target_impedance: float | None = None,
        axial_offset_cells: int = 1,
    ) -> "Simulation":
        """Register a distributed annular matched termination on a coaxial port.

        Stamps a single-z-cell annular conductor in the PTFE region
        between the pin and outer-shell of the targeted port, with sigma
        chosen so the radial pin-to-shell resistance equals
        ``target_impedance``. Use this to build a matched-load test
        without needing a full lumped pin↔shell resistor model.

        Parameters
        ----------
        port_index : int
            Index into the order of ``add_coaxial_port`` calls. Default 0
            targets the first registered coaxial port.
        target_impedance : float or None
            Resistance of the radial pin-to-shell path in ohms. ``None``
            uses the closed-form ``coaxial_tem_characteristic_impedance``
            for the port's geometry — the canonical "matched load".
        axial_offset_cells : int
            Number of Yee cells past the pin tip (in the forward
            direction) where the load slice sits. Default 1 places the
            load one cell beyond the pin tip.
        """

        if port_index < 0 or port_index >= len(self._coaxial_ports):
            raise ValueError(
                f"port_index {port_index} is out of range "
                f"(have {len(self._coaxial_ports)} coaxial ports registered)"
            )
        if target_impedance is None:
            from rfx.sources.coaxial_port import (
                coaxial_tem_characteristic_impedance,
                PTFE_EPS_R,
            )
            p = self._coaxial_ports[port_index]
            target_impedance = float(
                coaxial_tem_characteristic_impedance(
                    p.pin_radius, p.outer_radius, PTFE_EPS_R
                )
            )
        self._coaxial_terminations.append(
            (int(port_index), float(target_impedance), int(axial_offset_cells))
        )
        return self

    # ---- thin conductors ----

    def add_thin_conductor(
        self,
        shape: Shape,
        *,
        sigma_bulk: float = 5.8e7,
        thickness: float = 35e-6,
        eps_r: float = 1.0,
    ) -> "Simulation":
        """Add a thin conductor with subcell correction.

        Parameters
        ----------
        shape : Shape
            Geometric region of the conductor.
        sigma_bulk : float
            Bulk conductivity (S/m). Default: copper (5.8e7).
        thickness : float
            Physical thickness (metres). Default: 35 µm (1 oz copper).
        eps_r : float
            Relative permittivity (default 1.0).
        """
        tc = ThinConductor(
            shape=shape, sigma_bulk=sigma_bulk,
            thickness=thickness, eps_r=eps_r,
        )
        # P0.1: Warn if thin conductor will be routed to PEC mask
        if tc.is_pec:
            import warnings
            dx_est = self._dx or C0 / self._freq_max / 20.0
            sigma_eff = sigma_bulk * thickness / dx_est
            warnings.warn(
                f"Thin conductor sigma_eff={sigma_eff:.2e} S/m exceeds PEC "
                f"threshold (1e6). Will be routed to PEC mask, not "
                f"conductivity. Use lower sigma_bulk for lossy behavior.",
                stacklevel=2,
            )
        self._thin_conductors.append(tc)
        return self

    # ---- ports ----

    def add_port(
        self,
        position: tuple[float, float, float],
        component: str = "ez",
        *,
        impedance: float = 50.0,
        waveform: GaussianPulse | None = None,
        extent: float | None = None,
        excite: bool = True,
        direction: str | None = None,
    ) -> "Simulation":
        """Add a lumped port (single-cell) or wire port (multi-cell).

        Parameters
        ----------
        position : (x, y, z) in metres
        component : "ex", "ey", or "ez"
        impedance : port impedance in ohms (default 50)
        waveform : excitation pulse (default: GaussianPulse at freq_max/2).
            Ignored when ``excite=False``.
        extent : float or None
            When provided, the port spans from *position* along the port
            axis by this distance (metres), creating a multi-cell WirePort.
            For example ``component="ez", extent=0.0015`` spans the port
            from ``z`` to ``z + 0.0015``.
        excite : bool (default True)
            When True the port has BOTH a resistive termination AND a
            time-domain source (legacy behaviour).
            When False the port is a passive matched load only — no
            source, just the σ=1/(Z0·A) resistive termination.
            Passive ports are required for multi-port S-parameter
            extraction: excite one port at a time and probe V/I at the
            others to fill off-diagonal entries of the S matrix.
        direction : {"+x", "-x", "+y", "-y"} or None
            Outward-normal direction of the port (from the port cell
            into the external world). Used by the S-matrix post-
            processing to orient the V/I → (incoming, outgoing) wave
            decomposition. When None, the runner auto-detects from the
            port's position (closest boundary face).
        """
        if self._tfsf is not None:
            raise ValueError(
                "Lumped ports are not supported together with the TFSF plane-wave source"
            )
        if component not in ("ex", "ey", "ez"):
            raise ValueError(f"component must be ex/ey/ez, got {component!r}")
        if impedance <= 0:
            raise ValueError(f"impedance must be positive, got {impedance}")
        if direction is not None and direction not in ("+x", "-x", "+y", "-y"):
            raise ValueError(
                f"direction must be one of '+x','-x','+y','-y' (or None), got {direction!r}"
            )

        if waveform is None and excite:
            waveform = GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)
        if waveform is not None and not excite:
            import warnings as _w
            _w.warn(
                "waveform is ignored when excite=False (passive matched load). "
                "Remove the waveform argument to suppress this warning.",
                stacklevel=2,
            )

        self._ports.append(_PortEntry(
            position=position, component=component,
            impedance=impedance, waveform=waveform,
            extent=extent, excite=excite, direction=direction,
        ))
        return self

    def add_msl_port(
        self,
        position: tuple[float, float, float],
        *,
        width: float,
        height: float,
        direction: str = "+x",
        impedance: float = 50.0,
        waveform: GaussianPulse | None = None,
        excite: bool = True,
        n_probe_offset: int | None = None,
        n_probe_spacing: int | None = None,
        n_probes: int = 5,
        name: str | None = None,
        mode: str = "laplace",
        eps_r_sub: float | None = None,
    ) -> "Simulation":
        """Add a microstrip-line (MSL) port spanning the full trace cross-section.

        Unlike :meth:`add_port` with ``extent=...`` (a one-cell-transverse
        wire port), this port covers the full ``width × height``
        cross-section under the trace and uses 3-probe numerical
        de-embedding to extract β and Z0 empirically downstream of the
        feed plane.

        Parameters
        ----------
        position : (x_feed, y_centre, z_lo)
            Feed plane x, trace centre y, and substrate bottom z.
        width : float
            Trace width in metres (y extent of the port).
        height : float
            Substrate thickness in metres (z extent of the port).
        direction : "+x" or "-x"
            Direction the launched wave propagates.
        impedance : float
            Target Z0 in ohms (default 50). Used for the σ distribution.
        waveform : GaussianPulse or callable, optional
            Excitation waveform. Defaults to a band-limited Gaussian
            centred at ``freq_max/2``. Ignored when ``excite=False``.
        excite : bool
            ``True`` → resistive termination + active source; ``False``
            → passive matched termination only.
        n_probe_offset : int, optional
            Distance (cells) from the feed plane to the first probe plane.
            When ``None``, bound to the LARGER of two near-field clearances:
            (a) the wavelength reactive far-field (issue #80 Fix B),
            ``round(0.5 * lam_min_eff / (2*pi) / dx)`` with
            ``lam_min_eff = c / freq_max / sqrt(eps_r_sub_estimate)``; and
            (b) the source FRINGING transient ``round(5 * h_sub / dx)``,
            which decays over a few substrate thicknesses (``h_sub`` = the
            port ``height``), NOT over λ. For a thin high-εr substrate (b)
            dominates: clearing only (a) leaves probe 0 inside the fringing
            transient and corrupts the V·I-split S11 of a high-Q resonant
            load (issue #80 patch: |S11|=8.94/1.11 → passive ~0.99 once
            cleared). Pass an explicit value to override; ``< 5*h_sub/dx``
            triggers a preflight near-field warning.
        n_probe_spacing : int, optional
            Distance (cells) between consecutive probe planes. When ``None``,
            bound so the total N-probe array span stays ~``lam_min_eff/8``
            independent of ``n_probes``:
            ``round(lam_min_eff / 8 / (n_probes - 1) / dx)``. For
            ``n_probes=3`` this is exactly ``lam_min_eff/16`` (issue #80
            Fix B). Shrinking the adjacent spacing as ``n_probes`` grows
            keeps the probe array from running off short lines.
        n_probes : int
            Number of equally-spaced voltage probe planes registered for
            the N-probe least-squares wave-decomposition extractor (issue
            #80 Fix C). Probe ``n`` sits at ``offset + n*spacing`` cells
            from the feed plane. ``N >= 3`` is required; the default 5
            over-determines the 2-unknown ``(alpha, gamma)`` fit, which
            removes the 3-probe quadratic's q→1 singularity. The current
            probe at probe 0 is always recorded for the absolute Z0.
        name : str, optional
            Port label used in result dicts. Auto-generated when omitted.
        eps_r_sub : float, optional
            Substrate relative permittivity. Used both for the eigenmode
            source and as ``eps_r_sub_estimate`` for the wavelength-bound
            probe-placement defaults (issue #80 Fix B). When ``None`` the
            estimate is resolved from the dielectric geometry already
            registered under the port (a shape whose bounding box contains
            the substrate mid-point and whose material has ``eps_r > 1``);
            if no such dielectric is found it falls back to 1.0, which is
            conservative (largest ``lam_min_eff`` → largest offset).
        """
        if direction not in ("+x", "-x"):
            raise ValueError(f"direction must be '+x' or '-x', got {direction!r}")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")
        if impedance <= 0:
            raise ValueError(f"impedance must be positive, got {impedance}")
        if mode not in ("eigenmode", "laplace", "uniform"):
            raise ValueError(f"mode must be 'eigenmode', 'laplace', or 'uniform', got {mode!r}")
        # Wavelength-bound probe-placement defaults (issue #80 Fix B).
        # Cell-counted and fixed-µm defaults both placed probe 1 inside the
        # source reactive zone and the 3-probe quadratic at the q→1
        # singularity. Bind the defaults to the shortest in-substrate
        # wavelength so β·Δ stays ≈ π/8 and probe 1 sits in the far-field.
        # An explicit user-supplied value is always honoured.
        _dx = float(self._dx or (C0 / self._freq_max / 10))
        # Resolve the substrate permittivity for the wavelength estimate.
        # Precedence: explicit eps_r_sub kwarg > the dielectric registered
        # under the port (a shape containing the substrate mid-point whose
        # material has eps_r > 1) > conservative 1.0 fallback (largest
        # lam_min_eff, hence largest offset — safe but coarse).
        if eps_r_sub is not None:
            eps_r_sub_estimate = float(eps_r_sub)
        else:
            eps_r_sub_estimate = 1.0
            x_feed, y_centre, z_lo = position
            probe_pt = (x_feed, y_centre, z_lo + height / 2.0)
            for ge in self._geometry:
                try:
                    (lo0, lo1, lo2), (hi0, hi1, hi2) = ge.shape.bounding_box()
                except Exception:
                    # Shape without a bounding_box() (or a degenerate one):
                    # skip it for the estimate rather than fail port setup.
                    continue
                if not (
                    lo0 <= probe_pt[0] <= hi0
                    and lo1 <= probe_pt[1] <= hi1
                    and lo2 <= probe_pt[2] <= hi2
                ):
                    continue
                mat_eps = float(self._resolve_material(ge.material_name).eps_r)
                if mat_eps > 1.0:
                    eps_r_sub_estimate = max(eps_r_sub_estimate, mat_eps)
        lam_min_eff = C0 / self._freq_max / math.sqrt(eps_r_sub_estimate)
        # Probe 0 must clear BOTH near-field scales of the MSL launch:
        #  (a) λ/(2π) reactive far-field of the quasi-TEM mode, and
        #  (b) the SOURCE FRINGING transient, which decays over a few
        #      substrate thicknesses (~5·h_sub), NOT over λ.  For a thin
        #      high-εr substrate (b) dominates; clearing only (a) leaves
        #      probe 0 inside the fringing transient and corrupts the
        #      V·I-split S11 of a high-Q resonant load — the issue #80
        #      edge-fed patch read |S11|=8.94/1.11 at the (a)-only offset
        #      (~5 cells) and a passive ~0.99 once cleared to ~5·h_sub.
        #      ``height`` is the port cross-section height = substrate h_sub.
        _lam_cells = int(round(0.5 * lam_min_eff / (2.0 * math.pi) / _dx))
        _hsub_cells = int(round(5.0 * height / _dx))
        if n_probe_offset is None:
            n_probe_offset = max(3, _lam_cells, _hsub_cells)
        if n_probe_spacing is None:
            # Bind the default so the TOTAL N-probe array span stays
            # ~lam/8 (the original Fix B 3-probe span of 2*lam/16),
            # independent of n_probes. For n_probes=3 this is exactly
            # lam/16 — bit-identical to Fix B. As n_probes grows the
            # adjacent spacing shrinks so the probe array does not run
            # off short lines (issue #80 Fix C).
            span_total = lam_min_eff / 8.0
            n_probe_spacing = max(
                2, int(round(span_total / (n_probes - 1) / _dx))
            )
        if n_probe_offset < 3:
            raise ValueError(
                f"n_probe_offset must be >= 3 to avoid near-field, got {n_probe_offset}"
            )
        if n_probe_spacing < 2:
            raise ValueError(
                f"n_probe_spacing must be >= 2 to avoid the q->1 extractor "
                f"singularity, got {n_probe_spacing}"
            )
        if n_probes < 3:
            raise ValueError(
                f"n_probes must be >= 3 for the N-probe least-squares "
                f"wave-decomposition extractor (issue #80 Fix C), got "
                f"{n_probes}"
            )

        if waveform is None and excite:
            waveform = GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)

        if name is None:
            name = f"msl_{len(self._msl_ports)}"

        self._msl_ports.append(_MSLPortEntry(
            name=name,
            position=position,
            width=width,
            height=height,
            direction=direction,
            impedance=impedance,
            waveform=waveform,
            excite=excite,
            n_probe_offset=n_probe_offset,
            n_probe_spacing=n_probe_spacing,
            n_probes=n_probes,
            mode=mode,
            eps_r_sub=eps_r_sub,
        ))
        return self

    def add_lumped_rlc(
        self,
        position: tuple[float, float, float],
        component: str = "ez",
        *,
        R: float = 0.0,
        L: float = 0.0,
        C: float = 0.0,
        topology: str = "series",
    ) -> "Simulation":
        """Add a lumped RLC element at a single cell.

        The element is modelled via Auxiliary Differential Equations
        (ADE) that are updated each timestep alongside the standard
        Yee update.  Any combination of R, L, C is valid (set unused
        components to 0).

        Parameters
        ----------
        position : (x, y, z) in metres
        component : "ex", "ey", or "ez"
        R : float
            Resistance in ohms (default 0).
        L : float
            Inductance in henries (default 0).
        C : float
            Capacitance in farads (default 0).
        topology : "series" or "parallel"

        Notes
        -----
        For series topology with a single component (e.g., pure L with
        R=0 and C=0), the element is handled via material folding, not
        the full series ADE current tracker.  To ensure the series ADE
        path is used, specify at least two non-zero components.
        """
        if component not in ("ex", "ey", "ez"):
            raise ValueError(f"component must be ex/ey/ez, got {component!r}")
        if topology not in ("series", "parallel"):
            raise ValueError(f"topology must be 'series' or 'parallel', got {topology!r}")
        if R < 0 or L < 0 or C < 0:
            raise ValueError(f"R, L, C must be non-negative, got R={R}, L={L}, C={C}")
        if R == 0 and L == 0 and C == 0:
            raise ValueError("At least one of R, L, C must be non-zero")

        # Warn if series topology with single component — falls back to parallel behavior
        if topology == "series":
            n_comp = (R > 0) + (L > 0) + (C > 0)
            if n_comp == 1:
                import warnings
                active = "R" if R > 0 else ("L" if L > 0 else "C")
                warnings.warn(
                    f"Series topology with single component ({active}) uses material "
                    f"folding, not the series ADE. Add a second component or use "
                    f"topology='parallel' to suppress this warning.",
                    stacklevel=2,
                )

        self._lumped_rlc.append(LumpedRLCSpec(
            R=R, L=L, C=C,
            topology=topology,
            position=position,
            component=component,
        ))
        return self

    def add_tfsf_source(
        self,
        *,
        f0: float | None = None,
        bandwidth: float = 0.5,
        amplitude: float = 1.0,
        margin: int = 3,
        polarization: str = "ez",
        direction: str = "+x",
        angle_deg: float = 0.0,
        waveform: str = "differentiated_gaussian",
    ) -> "Simulation":
        """Add a normal-incidence plane-wave TFSF source.

        Current scope is intentionally narrow: x-directed propagation,
        ``ez``/``ey`` polarization, 3D mode, and CPML boundaries. For
        oblique incidence, only the single transverse-axis plane implied
        by the chosen polarization is supported.

        Parameters
        ----------
        waveform : {"differentiated_gaussian", "modulated_gaussian"}
            Pulse shape injected into the 1D auxiliary grid.

            ``differentiated_gaussian`` (default, legacy rfx): baseband
            pulse with no carrier, DC-free, spectrum peaks near
            ``f0·bandwidth``. Good for broadband measurements but
            spectrum extends well below ``f0``.

            ``modulated_gaussian``: carrier-modulated Gaussian matching
            Meep's ``GaussianSource(frequency=f0, fwidth=f0·bandwidth)``.
            Spectrum is a Gaussian centered at ``f0`` with 1/e
            half-width ``f0·bandwidth``. Use this for matched rfx-vs-Meep
            crossval comparisons.
        """
        if self._boundary != "cpml":
            raise ValueError("TFSF plane-wave source requires boundary='cpml'")
        if self._cpml_layers <= 0:
            raise ValueError("TFSF plane-wave source requires cpml_layers > 0")
        if self._mode not in ("3d", "2d_tmz", "2d_tez"):
            raise ValueError(
                f"TFSF plane-wave source requires mode='3d', '2d_tmz', or '2d_tez', got {self._mode!r}"
            )
        if self._periodic_axes:
            raise ValueError("TFSF plane-wave source is not supported with manual periodic-axis overrides")
        if self._ports:
            raise ValueError(
                "TFSF plane-wave source is not supported together with lumped ports"
            )
        if f0 is not None and f0 <= 0:
            raise ValueError(f"f0 must be positive when provided, got {f0}")
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth}")
        if margin < 1:
            raise ValueError(f"margin must be >= 1, got {margin}")
        if polarization not in ("ez", "ey"):
            raise ValueError(f"polarization must be 'ez' or 'ey', got {polarization!r}")
        if direction not in ("+x", "-x"):
            raise ValueError(f"direction must be '+x' or '-x', got {direction!r}")
        if abs(angle_deg) >= 90.0:
            raise ValueError(f"abs(angle_deg) must be < 90, got {angle_deg}")
        if waveform not in ("differentiated_gaussian", "modulated_gaussian"):
            raise ValueError(
                f"waveform must be 'differentiated_gaussian' or "
                f"'modulated_gaussian', got {waveform!r}"
            )

        self._tfsf = _TFSFEntry(
            f0=f0,
            bandwidth=bandwidth,
            amplitude=amplitude,
            margin=margin,
            polarization=polarization,
            direction=direction,
            angle_deg=angle_deg,
            waveform=waveform,
        )
        return self

    def add_waveguide_port(
        self,
        x_position: float,
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
        mode: tuple[int, int] = (1, 0),
        mode_type: str = "TE",
        direction: str = "+x",
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 50,
        f0: float | None = None,
        bandwidth: float = 0.5,
        amplitude: float = 1.0,
        probe_offset: int = 10,
        ref_offset: int = 3,
        calibration_preset: str | None = None,
        reference_plane: float | None = None,
        probe_plane: float | None = None,
        name: str | None = None,
        n_modes: int = 1,
        waveform: str = "modulated_gaussian",
        mode_profile: str = "discrete",
    ) -> "Simulation":
        """Add a rectangular waveguide port.

        `x_position` is interpreted along the selected port-normal axis from
        `direction`. For example, `direction='-y'` uses `x_position` as the
        physical y-coordinate of the port plane.

        Current scope supports axis-normal boundary ports using rectangular
        apertures, with `boundary='cpml'` and `mode='3d'`.

        Calibration options:
        - `reference_plane` / `probe_plane`: explicit reported planes in
          physical coordinates along the port normal axis
        - `calibration_preset='source_to_probe'`: auto-report `S11` at the
          snapped source plane and `S21` from source to the snapped probe plane
        - `calibration_preset=None` or `'measured'`: use the snapped stored
          reference/probe planes directly

        Sampling still occurs on the nearest snapped grid planes, and the
        result metadata reports those actual measurement planes explicitly.

        ``waveform`` selects the source pulse shape. Default
        ``"differentiated_gaussian"`` matches historical rfx behaviour.
        ``"modulated_gaussian"`` is the Meep-style bandpass pulse — no
        sub-cutoff DC content, so the in-band TFSF filter collapses to
        identity, reducing directional leakage in the H+E injection pair.
        Preliminary measurement shows directionality backward/forward
        ratio 13.3 % → 8.2 % for a WR-90 at f₀=10 GHz; further gains
        await the discrete-eigenmode profile work (P3).
        """
        scalar_checks = [
            ("x_position", x_position, False),
            ("bandwidth", bandwidth, True),
            ("amplitude", amplitude, False),
        ]
        if f0 is not None:
            scalar_checks.append(("f0", f0, True))
        if reference_plane is not None:
            scalar_checks.append(("reference_plane", reference_plane, False))
        if probe_plane is not None:
            scalar_checks.append(("probe_plane", probe_plane, False))
        for label, value, require_positive in scalar_checks:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{label} must be a finite scalar, got {value!r}") from None
            if not math.isfinite(numeric):
                raise ValueError(f"{label} must be finite, got {value!r}")
            if require_positive and numeric <= 0:
                raise ValueError(f"{label} must be positive, got {value}")

        if self._boundary != "cpml":
            raise ValueError("Waveguide port requires boundary='cpml'")
        if self._cpml_layers <= 0:
            raise ValueError("Waveguide port requires cpml_layers > 0")
        if self._mode != "3d":
            raise ValueError("Waveguide port currently supports only mode='3d'")
        if self._periodic_axes:
            raise ValueError("Waveguide port is not supported with manual periodic-axis overrides")
        if self._ports:
            raise ValueError("Waveguide port is not supported together with lumped ports")
        if self._tfsf is not None:
            raise ValueError("Waveguide port is not supported together with TFSF")
        if direction not in ("+x", "-x", "+y", "-y", "+z", "-z"):
            raise ValueError(
                "direction must be one of '+x', '-x', '+y', '-y', '+z', or '-z', "
                f"got {direction!r}"
            )
        if waveform not in ("differentiated_gaussian", "modulated_gaussian"):
            raise ValueError(
                "waveform must be 'differentiated_gaussian' or "
                f"'modulated_gaussian', got {waveform!r}"
            )
        if mode_profile not in ("analytic", "discrete"):
            raise ValueError(
                "mode_profile must be 'analytic' or 'discrete', "
                f"got {mode_profile!r}"
            )
        axis_name = direction[1]
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name]
        if x_position < 0 or x_position > self._domain[axis_idx]:
            raise ValueError(
                f"x_position {x_position} m is outside the {axis_name}-domain [0, {self._domain[axis_idx]}]"
            )
        for label, rng, domain_max in (
            ("x_range", x_range, self._domain[0]),
            ("y_range", y_range, self._domain[1]),
            ("z_range", z_range, self._domain[2]),
        ):
            if rng is None:
                continue
            if not isinstance(rng, tuple) or len(rng) != 2:
                raise ValueError(f"{label} must be a (lo, hi) tuple when provided")
            lo, hi = rng
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except (TypeError, ValueError):
                raise ValueError(f"{label} must contain finite scalars, got {rng!r}") from None
            if not math.isfinite(lo_f) or not math.isfinite(hi_f):
                raise ValueError(f"{label} must contain finite scalars, got {rng!r}")
            if lo_f < 0.0 or hi_f > domain_max or hi_f <= lo_f:
                raise ValueError(
                    f"{label} {rng!r} must satisfy 0 <= lo < hi <= {domain_max}"
                )
        if (
            not isinstance(mode, tuple)
            or len(mode) != 2
            or any(not isinstance(idx, Integral) for idx in mode)
        ):
            raise ValueError(f"mode must be a tuple of two integers, got {mode!r}")
        if any(idx < 0 for idx in mode):
            raise ValueError(f"mode indices must be non-negative, got {mode!r}")
        if mode_type not in ("TE", "TM"):
            raise ValueError(f"mode_type must be 'TE' or 'TM', got {mode_type!r}")
        unused_range_by_axis = {
            "x": ("x_range", x_range, "y_range/z_range"),
            "y": ("y_range", y_range, "x_range/z_range"),
            "z": ("z_range", z_range, "x_range/y_range"),
        }
        unused_label, unused_value, replacement = unused_range_by_axis[axis_name]
        if unused_value is not None:
            raise ValueError(
                f"{unused_label} is not used for {axis_name}-normal ports; use {replacement} instead"
            )
        if not isinstance(probe_offset, Integral) or not isinstance(ref_offset, Integral):
            raise ValueError("probe_offset and ref_offset must be positive integers")
        if probe_offset <= 0 or ref_offset <= 0:
            raise ValueError("probe_offset and ref_offset must be positive integers")
        if not isinstance(n_modes, Integral) or n_modes < 1:
            raise ValueError(f"n_modes must be a positive integer, got {n_modes!r}")
        if calibration_preset not in (None, "measured", "source_to_probe"):
            raise ValueError(
                "calibration_preset must be one of None, 'measured', or 'source_to_probe'"
            )
        if calibration_preset is not None and (reference_plane is not None or probe_plane is not None):
            raise ValueError(
                "calibration_preset cannot be combined with explicit reference_plane/probe_plane"
            )
        grid = self._build_grid(extra_waveguide_axes=axis_name)
        pos_vec = [0.0, 0.0, 0.0]
        pos_vec[axis_idx] = x_position
        x_index = grid.position_to_index(tuple(pos_vec))[axis_idx]
        axis_pad = grid.axis_pads[axis_idx]
        snapped_source_plane = (x_index - axis_pad) * grid.dx
        step_sign = 1 if direction.startswith("+") else -1
        measured_reference_plane = snapped_source_plane + step_sign * ref_offset * grid.dx
        measured_probe_plane = snapped_source_plane + step_sign * probe_offset * grid.dx
        axis_domain = self._domain[axis_idx]
        if (
            measured_reference_plane < 0.0
            or measured_reference_plane > axis_domain
            or measured_probe_plane < 0.0
            or measured_probe_plane > axis_domain
            or x_index + step_sign * ref_offset < 0
            or x_index + step_sign * ref_offset >= grid.shape[axis_idx]
            or x_index + step_sign * probe_offset < 0
            or x_index + step_sign * probe_offset >= grid.shape[axis_idx]
        ):
            raise ValueError(
                "Waveguide port measurement planes exceed the physical "
                f"{axis_name}-domain after grid snapping; reduce ref_offset/probe_offset, "
                "flip direction, or move x_position inward"
            )
        if reference_plane is not None and not (0.0 <= reference_plane <= axis_domain):
            raise ValueError(
                f"reference_plane {reference_plane} m is outside the {axis_name}-domain [0, {axis_domain}]"
            )
        if probe_plane is not None and not (0.0 <= probe_plane <= axis_domain):
            raise ValueError(
                f"probe_plane {probe_plane} m is outside the {axis_name}-domain [0, {axis_domain}]"
            )
        if (
            reference_plane is not None
            and probe_plane is not None
            and probe_plane < reference_plane
        ):
            raise ValueError("probe_plane must be >= reference_plane when both are provided")
        if freqs is None:
            if not isinstance(n_freqs, Integral):
                raise ValueError(f"n_freqs must be a positive integer, got {n_freqs!r}")
            if n_freqs <= 0:
                raise ValueError(f"n_freqs must be positive, got {n_freqs}")
            freqs_arr = None
        else:
            freqs_arr = jnp.asarray(freqs)
            if freqs_arr.ndim != 1 or freqs_arr.size == 0:
                raise ValueError("freqs must be a non-empty 1-D array")
            freqs_np = np.asarray(freqs_arr, dtype=float)
            if not np.all(np.isfinite(freqs_np)):
                raise ValueError("freqs must contain only finite values")
            if np.any(freqs_np <= 0):
                raise ValueError("freqs must contain only positive values")

        if name is None:
            name = f"waveguide_{len(self._waveguide_ports)}"

        self._waveguide_ports.append(_WaveguidePortEntry(
            name=name,
            x_position=x_position,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            mode=mode,
            mode_type=mode_type,
            direction=direction,
            freqs=freqs_arr,
            n_freqs=n_freqs,
            f0=f0,
            bandwidth=bandwidth,
            amplitude=amplitude,
            probe_offset=probe_offset,
            ref_offset=ref_offset,
            calibration_preset=calibration_preset,
            reference_plane=reference_plane,
            probe_plane=probe_plane,
            n_modes=n_modes,
            waveform=waveform,
            mode_profile=mode_profile,
        ))
        return self

    def _build_spec_from_legacy(self):
        """T7-B: compose a canonical BoundarySpec from the legacy triad.

        Called once at ``__init__`` time and whenever the legacy fields
        change (``set_periodic_axes``, mutation of ``pec_faces``). The
        spec is the single source of truth for T7-D preflight and
        T7-C / T7-E runner integration; the legacy fields remain as
        derived views for code that has not yet migrated.
        """
        from rfx.boundaries.spec import BoundarySpec, Boundary
        default = self._boundary  # 'cpml' | 'upml' | 'pec'
        axes = {}
        for axis_name in "xyz":
            if axis_name in self._periodic_axes:
                axes[axis_name] = Boundary(lo="periodic", hi="periodic")
            else:
                lo_tok = "pec" if f"{axis_name}_lo" in self._pec_faces else default
                hi_tok = "pec" if f"{axis_name}_hi" in self._pec_faces else default
                axes[axis_name] = Boundary(lo=lo_tok, hi=hi_tok)
        return BoundarySpec(x=axes["x"], y=axes["y"], z=axes["z"])

    def set_periodic_axes(self, axes: str = "xyz") -> "Simulation":
        """Set periodic boundary axes for high-level runs.

        .. deprecated:: 1.6.3
            Encode periodic axes directly in :class:`BoundarySpec`:
            ``boundary=BoundarySpec(x='periodic', y='cpml', z='cpml')``.
            This method will be removed in v2.0.

        Parameters
        ----------
        axes : str
            Any combination of ``x``, ``y``, ``z``. Empty string disables
            manual periodic overrides.
        """
        import warnings as _w
        _w.warn(
            "Simulation.set_periodic_axes() is deprecated; pass a "
            "BoundarySpec to Simulation(..., boundary=BoundarySpec(...)) "
            "with periodic tokens on the desired axes instead. "
            "The method will be removed in rfx v2.0.",
            DeprecationWarning, stacklevel=2,
        )
        normalized = "".join(axis for axis in "xyz" if axis in axes)
        invalid = sorted(set(axes) - set("xyz"))
        if invalid:
            raise ValueError(f"periodic axes must be drawn from 'xyz', got invalid axes {invalid}")
        if self._tfsf is not None:
            raise ValueError("Manual periodic-axis overrides are not supported together with TFSF")
        if self._waveguide_ports:
            raise ValueError("Manual periodic-axis overrides are not supported together with waveguide ports")
        self._periodic_axes = normalized
        # Rebuild the canonical BoundarySpec so downstream code that
        # consults it (T7-C/D/E) sees the updated periodic axes.
        self._boundary_spec = self._build_spec_from_legacy()
        return self

    # ---- Floquet ports ----

    def add_floquet_port(
        self,
        position: float,
        *,
        axis: str = "z",
        scan_theta: float = 0.0,
        scan_phi: float = 0.0,
        polarization: str = "te",
        n_modes: int = 1,
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 50,
        f0: float | None = None,
        bandwidth: float = 0.5,
        amplitude: float = 1.0,
        name: str | None = None,
    ) -> "Simulation":
        """Add a Floquet port for periodic structure / phased array analysis.

        The Floquet port injects a plane wave at the given scan angle
        and extracts Floquet mode amplitudes (S-parameters) from the
        unit cell response.  Requires periodic BC on the two axes
        perpendicular to the port normal.

        Parameters
        ----------
        position : float
            Physical coordinate along the port normal axis (metres).
        axis : str
            Port normal axis: ``"x"``, ``"y"``, or ``"z"``.
        scan_theta : float
            Scan angle theta from broadside (degrees). Default 0.
        scan_phi : float
            Scan angle phi in the transverse plane (degrees). Default 0.
        polarization : str
            ``"te"`` or ``"tm"``. Default ``"te"``.
        n_modes : int
            Number of Floquet modes to extract (default 1 = specular).
        freqs : array or None
            Analysis frequencies. Auto-generated if None.
        n_freqs : int
            Number of frequency points when ``freqs`` is None.
        f0 : float or None
            Source center frequency. Default: ``freq_max / 2``.
        bandwidth : float
            Source fractional bandwidth. Default 0.5.
        amplitude : float
            Source amplitude. Default 1.0.
        name : str or None
            Optional name for the port. Auto-generated if None.
        """
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        if polarization not in ("te", "tm"):
            raise ValueError(f"polarization must be 'te' or 'tm', got {polarization!r}")
        if scan_theta < 0 or scan_theta >= 90:
            raise ValueError(f"scan_theta must be in [0, 90), got {scan_theta}")
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")
        if self._tfsf is not None:
            raise ValueError(
                "Floquet ports are not supported together with TFSF sources"
            )

        # Auto-set periodic axes for the two transverse directions
        transverse = "".join(a for a in "xyz" if a != axis)
        if not self._periodic_axes:
            self._periodic_axes = transverse
        else:
            for a in transverse:
                if a not in self._periodic_axes:
                    raise ValueError(
                        f"Floquet port on axis={axis!r} requires periodic BC on {transverse!r}, "
                        f"but periodic_axes={self._periodic_axes!r}"
                    )

        # P0.3: Floquet port requires uniform mesh
        if self._dz_profile is not None:
            raise ValueError(
                "Floquet ports do not support non-uniform z mesh (dz_profile). "
                "Set dx explicitly to prevent auto-mesh from creating NU grid."
            )

        if name is None:
            name = f"floquet_{len(self._floquet_ports)}"

        self._floquet_ports.append(_FloquetPortEntry(
            name=name,
            position=position,
            axis=axis,
            scan_theta=scan_theta,
            scan_phi=scan_phi,
            polarization=polarization,
            n_modes=n_modes,
            freqs=freqs,
            n_freqs=n_freqs,
            f0=f0,
            bandwidth=bandwidth,
            amplitude=amplitude,
        ))
        return self

    # ---- probes ----

    def add_probe(
        self,
        position: tuple[float, float, float],
        component: str = "ez",
    ) -> "Simulation":
        """Add a point field probe."""
        if component not in ("ex", "ey", "ez", "hx", "hy", "hz"):
            raise ValueError(f"component must be a field name, got {component!r}")
        self._probes.append(_ProbeEntry(position=position, component=component))
        return self

    def add_vector_probe(
        self,
        position: tuple[float, float, float],
    ) -> "Simulation":
        """Add a vector probe that records ALL 6 field components.

        Records Ex, Ey, Ez, Hx, Hy, Hz at the same position.
        Results accessible via result.time_series columns [0..5].
        """
        for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
            self._probes.append(_ProbeEntry(position=position, component=comp))
        return self

    def add_dft_plane_probe(
        self,
        *,
        axis: str,
        coordinate: float,
        component: str = "ez",
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 50,
        name: str | None = None,
    ) -> "Simulation":
        """Add a frequency-domain 2D plane probe.

        Parameters
        ----------
        axis : "x", "y", or "z"
            Plane normal axis.
        coordinate : float
            Physical coordinate in metres along the selected axis.
        component : field component name
            One of ex/ey/ez/hx/hy/hz.
        freqs : array or None
            Probe frequencies in Hz. Default: linspace(freq_max/10, freq_max, n_freqs).
        n_freqs : int
            Number of frequencies if freqs is None.
        name : str or None
            Optional result key.
        """
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        if component not in ("ex", "ey", "ez", "hx", "hy", "hz"):
            raise ValueError(f"component must be a field name, got {component!r}")

        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        if coordinate < 0 or coordinate > self._domain[axis_idx]:
            raise ValueError(
                f"coordinate {coordinate} m is outside the {axis}-domain [0, {self._domain[axis_idx]}]"
            )
        if freqs is None:
            if n_freqs <= 0:
                raise ValueError(f"n_freqs must be positive, got {n_freqs}")
            freqs_arr = None
        else:
            freqs_arr = jnp.asarray(freqs)
            if freqs_arr.ndim != 1 or freqs_arr.size == 0:
                raise ValueError("freqs must be a non-empty 1-D array")

        if name is None:
            name = f"{component}_{axis}_{len(self._dft_planes)}"

        self._dft_planes.append(_DFTPlaneEntry(
            name=name,
            axis=axis,
            coordinate=coordinate,
            component=component,
            freqs=freqs_arr,
            n_freqs=n_freqs,
        ))
        return self

    def add_flux_monitor(
        self,
        *,
        axis: str,
        coordinate: float,
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 50,
        size: tuple[float, float] | None = None,
        center: tuple[float, float] | None = None,
        name: str | None = None,
        dft_window: str = "rect",
        dft_window_alpha: float = 0.25,
    ) -> "Simulation":
        """Add a Poynting flux monitor on a plane (Meep flux-region equivalent).

        Accumulates frequency-domain E and H tangential components to
        compute ``integral Re(E x H*) . n_hat dA`` at each frequency.

        Parameters
        ----------
        axis : "x", "y", or "z"
            Plane normal axis.
        coordinate : float
            Physical coordinate in metres along the selected axis.
        freqs : array or None
            Monitor frequencies in Hz.
        n_freqs : int
            Number of frequencies if freqs is None.
        size : (float, float) or None
            Physical extent in the two tangential directions.  ``None``
            means the full plane (legacy behaviour).
        center : (float, float) or None
            Physical centre of the flux region in the two tangential
            directions.  ``None`` defaults to the domain midpoint.
            For example, for an x-normal monitor the two tangential
            axes are (y, z).
        name : str or None
            Result key. Default: ``flux_{axis}_{idx}``.
        """
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        if coordinate < 0 or coordinate > self._domain[axis_idx]:
            raise ValueError(
                f"coordinate {coordinate} m is outside the {axis}-domain "
                f"[0, {self._domain[axis_idx]}]"
            )
        if freqs is not None:
            freqs_arr = jnp.asarray(freqs)
        else:
            freqs_arr = None

        if name is None:
            name = f"flux_{axis}_{len(self._flux_monitors)}"

        self._flux_monitors.append(_FluxMonitorEntry(
            name=name,
            axis=axis,
            coordinate=coordinate,
            freqs=freqs_arr,
            n_freqs=n_freqs,
            size=size,
            center=center,
            dft_window=dft_window,
            dft_window_alpha=dft_window_alpha,
        ))
        return self

    # ---- NTFF ----

    def add_ntff_box(
        self,
        corner_lo: tuple[float, float, float],
        corner_hi: tuple[float, float, float],
        freqs=None,
        n_freqs: int = 50,
    ) -> "Simulation":
        """Add a near-to-far-field transform box for radiation patterns.

        Parameters
        ----------
        corner_lo, corner_hi : (x, y, z) in metres
            Opposite corners of the Huygens box.
        freqs : array or None
            Frequencies (Hz). Default: n_freqs points from freq_max/10
            to freq_max.
        n_freqs : int
            Number of frequencies if freqs is None.
        """
        if freqs is None:
            freqs = jnp.linspace(self._freq_max / 10, self._freq_max, n_freqs)
        self._ntff = (corner_lo, corner_hi, freqs)
        return self

    # ---- build helpers ----

    # ---- AD memory estimation (issue #30 CHECK 4) ----
    def _ad_memory_static_accounting(self) -> dict[str, int]:
        """Return shared static byte accounting for AD memory artifacts."""
        dx = self._dx or (C0 / self._freq_max / 20.0)

        def _nx(extent: float, prof) -> int:
            if prof is not None:
                return len(prof) + 1 + 2 * self._cpml_layers
            return int(math.ceil(extent / dx)) + 1 + 2 * self._cpml_layers

        nx = _nx(self._domain[0], self._dx_profile)
        ny = _nx(self._domain[1], self._dy_profile)
        nz = _nx(self._domain[2], self._dz_profile)
        cells = int(nx * ny * nz)

        # Forward working set: 6 field + ~6 material + ~4 CPML psi (~15%)
        bytes_per_cell = 4  # float32
        field_bytes = cells * 6 * bytes_per_cell
        material_bytes = cells * 6 * bytes_per_cell
        cpml_bytes = (
            int(cells * 0.15 * 24 * bytes_per_cell)
            if self._cpml_layers > 0
            else 0
        )
        forward_bytes = field_bytes + material_bytes + cpml_bytes

        # NTFF DFT state: 6 faces × n_freqs × face cells × (3 E + 3 H) × complex64.
        ntff_bytes = 0
        if self._ntff is not None:
            _, _, freqs = self._ntff
            n_freqs = int(len(freqs)) if freqs is not None else 10
            face_est = 2 * ((nx * ny) + (ny * nz) + (nx * nz))
            ntff_bytes = face_est * n_freqs * 6 * 8

        return {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "cells": cells,
            "bytes_per_cell": bytes_per_cell,
            "field_bytes": field_bytes,
            "material_bytes": material_bytes,
            "cpml_bytes": cpml_bytes,
            "forward_bytes": forward_bytes,
            "ntff_bytes": ntff_bytes,
        }


    def estimate_ad_memory(
        self,
        n_steps: int,
        *,
        available_memory_gb: float | None = None,
        checkpoint_every: int | None = None,
        checkpoint_segments: int | None = None,
        n_warmup: int = 0,
        design_mask: jnp.ndarray | np.ndarray | None = None,
    ) -> "AD_MemoryEstimate":
        """Estimate reverse-mode AD memory for this simulation.

        Returns a static-estimate AD_MemoryEstimate with forward,
        checkpointed-AD, and non-checkpointed-AD sizes in GB, plus a
        best-effort warning if estimated AD memory exceeds 85% of available
        VRAM. This artifact is planning evidence, not a certificate of peak
        runtime memory.

        When ``checkpoint_every`` is provided, the returned
        ``ad_segmented_gb`` reflects the non-uniform segmented scan-of-scan
        chunk-size path from issue #31. When ``checkpoint_segments`` is
        provided, it reflects the uniform segmented-scan segment-count path
        from issue #73. ``n_warmup`` must be an integer with
        ``0 <= n_warmup < n_steps``. ``design_mask`` must be a boolean array
        matching the simulation grid shape and selecting at least one cell.
        ``n_warmup`` reduces reported reverse-mode tape time. ``design_mask``
        is reported as active-design metadata but does not reduce primary
        memory estimates until masked-state memory has observed calibration.
        The legacy
        ``ad_checkpointed_gb`` field keeps its old (optimistic) heuristic
        for backwards compatibility; it
        is NOT accurate for FDTD on the non-uniform path — see the
        class docstring.
        """
        n_steps_i = _require_integral_param("n_steps", n_steps)
        n_warmup_i = _require_integral_param("n_warmup", n_warmup)
        checkpoint_every_i = (
            None
            if checkpoint_every is None
            else _require_integral_param("checkpoint_every", checkpoint_every)
        )
        checkpoint_segments_i = (
            None
            if checkpoint_segments is None
            else _require_integral_param("checkpoint_segments", checkpoint_segments)
        )
        avail_gb = (
            None
            if available_memory_gb is None
            else _require_positive_finite_scalar(
                "available_memory_gb", available_memory_gb
            )
        )

        if n_steps_i <= 0:
            raise ValueError("n_steps must be positive")
        if n_warmup_i < 0:
            raise ValueError("n_warmup must be >= 0")
        if n_warmup_i >= n_steps_i:
            raise ValueError(f"n_warmup ({n_warmup_i}) must be < n_steps ({n_steps_i})")
        if checkpoint_every_i is not None and checkpoint_segments_i is not None:
            raise ValueError(
                "checkpoint_every and checkpoint_segments are mutually exclusive"
            )
        if checkpoint_every_i is not None and checkpoint_every_i <= 0:
            raise ValueError(
                f"checkpoint_every must be positive when provided, got {checkpoint_every_i}"
            )
        if checkpoint_segments_i is not None:
            if checkpoint_segments_i < 1:
                raise ValueError(
                    f"checkpoint_segments must be ≥ 1, got {checkpoint_segments_i}"
                )
            if n_steps_i % checkpoint_segments_i != 0:
                raise ValueError(
                    f"checkpoint_segments={checkpoint_segments_i} does not divide "
                    f"n_steps={n_steps_i}"
                )
        accounting = self._ad_memory_static_accounting()
        nx = accounting["nx"]
        ny = accounting["ny"]
        nz = accounting["nz"]
        field_bytes = accounting["field_bytes"]
        forward_bytes = accounting["forward_bytes"]
        ntff_bytes = accounting["ntff_bytes"]

        # Legacy "checkpointed" estimate: remat-recomputes internals of
        # step_fn. NOT valid for the NU path after issue #31 because the
        # scan carry itself is not rematerialised.
        ad_ckpt_bytes = 4 * forward_bytes + ntff_bytes
        active_steps = n_steps_i - n_warmup_i
        active_design_fraction = 1.0
        if design_mask is not None:
            mask = np.asarray(design_mask)
            if mask.size == 0:
                raise ValueError("design_mask must be non-empty")
            if mask.shape == ():
                raise ValueError("design_mask must be a boolean array matching grid shape")
            if mask.dtype != np.bool_:
                raise TypeError("design_mask must have boolean dtype")
            grid_shape = (nx, ny, nz)
            if mask.shape != grid_shape:
                raise ValueError(
                    f"design_mask shape {mask.shape} must match simulation grid shape {grid_shape}"
                )
            selected_cells = int(np.count_nonzero(mask))
            if selected_cells == 0:
                raise ValueError("design_mask must select at least one cell")
            active_design_fraction = float(selected_cells) / float(mask.size)
        active_tape_field_bytes = field_bytes

        # Non-checkpointed AD: O(active_steps) full-grid field tape. ``n_warmup``
        # reduces active reverse-mode time, while ``design_mask`` is retained as
        # metadata only until masked-state memory has observed calibration.
        ad_full_bytes = active_steps * active_tape_field_bytes + ntff_bytes + forward_bytes

        # Segmented scan paths. ``checkpoint_every`` is the non-uniform
        # scan-of-scan chunk length (issue #31); ``checkpoint_segments`` is
        # the uniform segmented-scan segment count (issue #73). Both store
        # carry + cotangent at segment boundaries.
        ad_seg_bytes: int | None = None
        segmented_active_segments: int | None = None
        if checkpoint_segments_i is not None:
            segment_len = n_steps_i // checkpoint_segments_i
            first_active_segment = n_warmup_i // segment_len
            segmented_active_segments = checkpoint_segments_i - first_active_segment
            ad_seg_bytes = (
                2 * segmented_active_segments * active_tape_field_bytes + forward_bytes + ntff_bytes
            )
        elif checkpoint_every_i is not None:
            total_segments = math.ceil(n_steps_i / checkpoint_every_i)
            inactive_segments = n_warmup_i // checkpoint_every_i
            segmented_active_segments = total_segments - inactive_segments
            ad_seg_bytes = (
                2 * segmented_active_segments * active_tape_field_bytes + forward_bytes + ntff_bytes
            )

        # VRAM detection (best effort)
        if avail_gb is None:
            try:
                devs = jax.local_devices()
                for d in devs:
                    if d.platform == "gpu":
                        stats = d.memory_stats() if hasattr(d, "memory_stats") else None
                        if stats and "bytes_limit" in stats:
                            avail_gb = stats["bytes_limit"] / 1e9
                            break
            except Exception:
                avail_gb = None

        to_gb = 1.0 / 1e9
        # Pick the most realistic estimate for the warning: segmented if
        # requested, otherwise full-AD (since the legacy "checkpointed"
        # number is unreliable — see class docstring).
        primary_bytes = ad_seg_bytes if ad_seg_bytes is not None else ad_full_bytes
        warning = None
        if avail_gb is not None and primary_bytes * to_gb > avail_gb * 0.85:
            label = "segmented" if ad_seg_bytes is not None else "non-checkpointed"
            if checkpoint_segments_i is not None:
                if segmented_active_segments == 1:
                    action = "Reduce grid size, reduce n_steps, or use a more aggressive memory-reduction lane."
                else:
                    action = "Reduce checkpoint_segments, reduce grid size, or reduce n_steps."
            elif ad_seg_bytes is not None:
                if segmented_active_segments == 1:
                    action = "Reduce grid size, reduce n_steps, or use a more aggressive memory-reduction lane."
                else:
                    action = "Increase checkpoint_every, reduce grid size, or reduce n_steps."
            else:
                action = "Use plan_ad_memory() to choose a segmented checkpoint setting, reduce grid size, or reduce n_steps."
            warning = (
                f"AD memory estimate {_format_memory_gb(primary_bytes * to_gb)} ({label}) "
                f"exceeds 85% of {_format_memory_gb(avail_gb)} available VRAM. "
                f"{action}"
            )
        return AD_MemoryEstimate(
            forward_gb=forward_bytes * to_gb,
            ad_checkpointed_gb=ad_ckpt_bytes * to_gb,
            ad_full_gb=ad_full_bytes * to_gb,
            ntff_dft_gb=ntff_bytes * to_gb,
            available_gb=avail_gb,
            warning=warning,
            ad_segmented_gb=(ad_seg_bytes * to_gb) if ad_seg_bytes is not None else None,
            checkpoint_every=checkpoint_every_i,
            checkpoint_segments=checkpoint_segments_i,
            ad_active_steps=active_steps,
            ad_active_design_fraction=active_design_fraction,
            ad_segmented_active_segments=segmented_active_segments,
        )

    def explain_ad_memory(
        self,
        n_steps: int,
        *,
        available_memory_gb: float | None = None,
        checkpoint_every: int | None = None,
        checkpoint_segments: int | None = None,
        n_warmup: int = 0,
        design_mask: jnp.ndarray | np.ndarray | None = None,
    ) -> ADMemoryExplainabilityReport:
        """Explain the static reverse-mode AD memory estimate.

        This method uses the same conservative accounting as
        :meth:`estimate_ad_memory`, then decomposes the selected AD memory
        number into named contributors. It is meant to answer "what is making
        this AD run large?" without claiming profiler evidence or a runtime
        peak bound.
        """
        n_steps_i = _require_integral_param("n_steps", n_steps)
        estimate = self.estimate_ad_memory(
            n_steps_i,
            available_memory_gb=available_memory_gb,
            checkpoint_every=checkpoint_every,
            checkpoint_segments=checkpoint_segments,
            n_warmup=n_warmup,
            design_mask=design_mask,
        )

        accounting = self._ad_memory_static_accounting()
        field_bytes = accounting["field_bytes"]
        material_bytes = accounting["material_bytes"]
        cpml_bytes = accounting["cpml_bytes"]
        ntff_bytes = accounting["ntff_bytes"]
        cells = accounting["cells"]
        bytes_per_cell = accounting["bytes_per_cell"]

        active_steps = estimate.ad_active_steps if estimate.ad_active_steps is not None else n_steps_i
        if estimate.ad_segmented_gb is not None:
            selected_field = "ad_segmented_gb"
            selected_gb = float(estimate.ad_segmented_gb)
            strategy = (
                "segmented_checkpoint_segments"
                if estimate.checkpoint_segments is not None
                else "segmented_checkpoint_every"
            )
            active_segments = (
                estimate.ad_segmented_active_segments
                if estimate.ad_segmented_active_segments is not None
                else 0
            )
            tape_count = int(2 * active_segments)
            tape_bytes = tape_count * field_bytes
            tape_name = "segmented_boundary_field_tape"
            tape_explanation = (
                "Segmented reverse-mode AD stores field carry and cotangent "
                "state at active segment boundaries instead of every active "
                "time step."
            )
            tape_unit = "carry-or-cotangent-field-state"
        else:
            selected_field = "ad_full_gb"
            selected_gb = float(estimate.ad_full_gb)
            strategy = "full_reverse_ad_static_tape"
            tape_count = int(active_steps)
            tape_bytes = tape_count * field_bytes
            tape_name = "full_reverse_field_tape"
            tape_explanation = (
                "Full reverse-mode AD stores a field tape entry for each "
                "active post-warmup time step."
            )
            tape_unit = "active-time-step-field-state"

        def _share(memory_bytes: int) -> float:
            if selected_gb <= 0.0:
                return 0.0
            return float(memory_bytes / 1e9 / selected_gb)

        def _component(
            name: str,
            memory_bytes: int,
            kind: str,
            *,
            unit: str | None,
            count: int | None,
            bytes_per_unit: int | None,
            explanation: str,
        ) -> ADMemoryComponent:
            return ADMemoryComponent(
                name=name,
                kind=kind,
                memory_gb=memory_bytes / 1e9,
                share_of_selected=_share(memory_bytes),
                unit=unit,
                count=count,
                bytes_per_unit_gb=(
                    None if bytes_per_unit is None else bytes_per_unit / 1e9
                ),
                explanation=explanation,
            )

        components = (
            _component(
                "field_state",
                field_bytes,
                "forward_working_set",
                unit="field-array",
                count=6,
                bytes_per_unit=cells * bytes_per_cell,
                explanation="Six E/H field arrays carried by the forward solve.",
            ),
            _component(
                "material_auxiliary_state",
                material_bytes,
                "forward_working_set",
                unit="material-array",
                count=6,
                bytes_per_unit=cells * bytes_per_cell,
                explanation=(
                    "Static material/ADE auxiliary allocation used by the "
                    "planner's forward working-set model."
                ),
            ),
            _component(
                "cpml_auxiliary_state",
                cpml_bytes,
                "forward_working_set",
                unit="cpml-overhead",
                count=None,
                bytes_per_unit=None,
                explanation=(
                    "CPML auxiliary state estimate; zero when the simulation "
                    "does not use CPML layers."
                ),
            ),
            _component(
                tape_name,
                tape_bytes,
                "reverse_ad_saved_state",
                unit=tape_unit,
                count=tape_count,
                bytes_per_unit=field_bytes,
                explanation=tape_explanation,
            ),
            _component(
                "ntff_dft_state",
                ntff_bytes,
                "monitor_state",
                unit="ntff-dft-state",
                count=None,
                bytes_per_unit=None,
                explanation=(
                    "Near-to-far-field DFT monitor state retained in forward "
                    "and AD planning estimates."
                ),
            ),
        )

        dominant = max(components, key=lambda component: component.memory_gb)
        recommendations: list[str] = [
            f"Dominant AD memory contributor is {dominant.name}.",
            "Treat ad_checkpointed_gb as a legacy heuristic; use the selected memory field in this report for planning.",
        ]
        if estimate.ad_segmented_gb is None:
            recommendations.append(
                "Use plan_ad_memory() to choose a supported segmented checkpoint knob when full reverse-mode AD is too large."
            )
        else:
            recommendations.append(
                f"Selected strategy stores {tape_count} {tape_unit} unit(s) instead of {active_steps} active time-step tape entries."
            )
        if estimate.ad_active_design_fraction is not None and estimate.ad_active_design_fraction < 1.0:
            recommendations.append(
                "design_mask is recorded as active-design metadata only; it does not reduce the selected memory estimate."
            )
        if estimate.warning:
            recommendations.append(estimate.warning)

        return ADMemoryExplainabilityReport(
            n_steps=n_steps_i,
            strategy=strategy,
            selected_memory_gb=selected_gb,
            selected_memory_field=selected_field,
            estimate=estimate,
            components=components,
            dominant_component=dominant.name,
            recommendations=tuple(recommendations),
        )

    def plan_ad_memory(
        self,
        n_steps: int,
        available_memory_gb: float,
        *,
        target_fraction: float = 0.85,
        n_warmup: int = 0,
        design_mask: jnp.ndarray | np.ndarray | None = None,
        safety_factor: float = AD_MEMORY_FIT_SAFETY_FACTOR,
    ) -> ADMemoryPlan:
        """Choose a segmented-AD memory plan for a memory budget.

        The planner reuses :meth:`estimate_ad_memory` and returns a
        calibrated conservative planning artifact rather than mutating the
        simulation or certifying runtime peak memory. If ordinary reverse-mode
        AD already fits under ``target_fraction * available_memory_gb``, both
        checkpoint knobs are ``None``. Otherwise it returns a segmented
        candidate: ``checkpoint_every`` for non-uniform grids or
        ``checkpoint_segments`` for uniform grids. Only wire the candidate when
        ``segmented_fits`` is true; a non-fitting plan keeps the least-memory
        candidate as diagnostics.
        ``n_warmup`` must be an integer with ``0 <= n_warmup < n_steps``.
        ``design_mask`` must be a boolean array matching the simulation grid
        shape and selecting at least one cell. ``n_warmup`` reduces active
        reverse-mode time; ``design_mask`` is metadata only for conservative
        budgeting. Raw estimates must also fit after multiplying by
        ``safety_factor`` before fit flags are set, so near-boundary plans stay
        conservative.
        """
        n_steps_i = _require_integral_param("n_steps", n_steps)
        n_warmup_i = _require_integral_param("n_warmup", n_warmup)
        available_memory_gb_f = _require_positive_finite_scalar(
            "available_memory_gb", available_memory_gb
        )
        target_fraction_f = _require_positive_finite_scalar(
            "target_fraction", target_fraction
        )
        safety_factor_f = _require_positive_finite_scalar(
            "safety_factor", safety_factor
        )

        if n_steps_i <= 0:
            raise ValueError("n_steps must be positive")
        if n_warmup_i < 0:
            raise ValueError("n_warmup must be >= 0")
        if n_warmup_i >= n_steps_i:
            raise ValueError(f"n_warmup ({n_warmup_i}) must be < n_steps ({n_steps_i})")
        if target_fraction_f > 1.0:
            raise ValueError("target_fraction must be in the interval (0, 1]")
        if safety_factor_f < 1.0:
            raise ValueError("safety_factor must be >= 1")

        target_memory_gb = available_memory_gb_f * target_fraction_f
        full_estimate = self.estimate_ad_memory(
            n_steps_i,
            available_memory_gb=available_memory_gb_f,
            n_warmup=n_warmup_i,
            design_mask=design_mask,
        )
        if full_estimate.ad_full_gb * safety_factor_f <= target_memory_gb:
            return ADMemoryPlan(
                n_steps=n_steps_i,
                available_memory_gb=available_memory_gb_f,
                target_fraction=target_fraction_f,
                target_memory_gb=target_memory_gb,
                checkpoint_every=None,
                checkpoint_segments=None,
                checkpoint_mode=None,
                fit_safety_factor=safety_factor_f,
                selected_estimate=full_estimate,
                full_ad_fits=True,
                segmented_fits=False,
                recommendation=(
                    f"full reverse-mode AD estimate ({_format_memory_with_safety(full_estimate.ad_full_gb, safety_factor_f)}) "
                    f"fits within the {_format_memory_gb(target_memory_gb)} target; "
                    "segmented checkpointing is optional for memory"
                ),
            )

        uses_nonuniform = (
            self._dx_profile is not None
            or self._dy_profile is not None
            or self._dz_profile is not None
        )
        if not uses_nonuniform:
            divisors = _positive_divisors(n_steps_i)
            sqrt_steps = math.sqrt(n_steps_i)
            recommended_segments = min(
                divisors,
                key=lambda divisor: (
                    abs(float(divisor) - sqrt_steps),
                    -divisor,
                ),
            )
            best_segments: int | None = None
            best_estimate: AD_MemoryEstimate | None = None
            for segments in sorted(
                (divisor for divisor in divisors if divisor <= recommended_segments),
                reverse=True,
            ):
                estimate = self.estimate_ad_memory(
                    n_steps_i,
                    available_memory_gb=available_memory_gb_f,
                    checkpoint_segments=segments,
                    n_warmup=n_warmup_i,
                    design_mask=design_mask,
                )
                segmented_gb = estimate.ad_segmented_gb
                if segmented_gb is not None and segmented_gb * safety_factor_f <= target_memory_gb:
                    best_segments = segments
                    best_estimate = estimate
                    break

            if best_segments is not None and best_estimate is not None:
                return ADMemoryPlan(
                    n_steps=n_steps_i,
                    available_memory_gb=available_memory_gb_f,
                    target_fraction=target_fraction_f,
                    target_memory_gb=target_memory_gb,
                    checkpoint_every=None,
                    checkpoint_segments=best_segments,
                    checkpoint_mode="checkpoint_segments",
                    fit_safety_factor=safety_factor_f,
                    selected_estimate=best_estimate,
                    full_ad_fits=False,
                    segmented_fits=True,
                    recommendation=(
                        f"use checkpoint_segments={best_segments}: segmented AD estimate "
                        f"{_format_memory_with_safety(best_estimate.ad_segmented_gb, safety_factor_f)} fits within the "
                        f"{_format_memory_gb(target_memory_gb)} target"
                    ),
                )

            minimum_segment_estimate = self.estimate_ad_memory(
                n_steps_i,
                available_memory_gb=available_memory_gb_f,
                checkpoint_segments=1,
                n_warmup=n_warmup_i,
                design_mask=design_mask,
            )
            return ADMemoryPlan(
                n_steps=n_steps_i,
                available_memory_gb=available_memory_gb_f,
                target_fraction=target_fraction_f,
                target_memory_gb=target_memory_gb,
                checkpoint_every=None,
                checkpoint_segments=1,
                checkpoint_mode="checkpoint_segments",
                fit_safety_factor=safety_factor_f,
                selected_estimate=minimum_segment_estimate,
                full_ad_fits=False,
                segmented_fits=False,
                recommendation=(
                    "even checkpoint_segments=1 is estimated at "
                    f"{_format_memory_with_safety(minimum_segment_estimate.ad_segmented_gb, safety_factor_f)}, above the "
                    f"{_format_memory_gb(target_memory_gb)} target; reduce mesh size, reduce "
                    "n_steps, or use a more aggressive memory-reduction lane"
                ),
            )

        best_checkpoint: int | None = None
        best_estimate: AD_MemoryEstimate | None = None
        for checkpoint in range(1, n_steps_i + 1):
            estimate = self.estimate_ad_memory(
                n_steps_i,
                available_memory_gb=available_memory_gb_f,
                checkpoint_every=checkpoint,
                n_warmup=n_warmup_i,
                design_mask=design_mask,
            )
            segmented_gb = estimate.ad_segmented_gb
            if segmented_gb is not None and segmented_gb * safety_factor_f <= target_memory_gb:
                best_checkpoint = checkpoint
                best_estimate = estimate
                break

        if best_checkpoint is not None and best_estimate is not None:
            return ADMemoryPlan(
                n_steps=n_steps_i,
                available_memory_gb=available_memory_gb_f,
                target_fraction=target_fraction_f,
                target_memory_gb=target_memory_gb,
                checkpoint_every=best_checkpoint,
                checkpoint_segments=None,
                checkpoint_mode="checkpoint_every",
                fit_safety_factor=safety_factor_f,
                selected_estimate=best_estimate,
                full_ad_fits=False,
                segmented_fits=True,
                recommendation=(
                    f"use checkpoint_every={best_checkpoint}: segmented AD estimate "
                    f"{_format_memory_with_safety(best_estimate.ad_segmented_gb, safety_factor_f)} fits within the "
                    f"{_format_memory_gb(target_memory_gb)} target"
                ),
            )

        largest_chunk_estimate = self.estimate_ad_memory(
            n_steps_i,
            available_memory_gb=available_memory_gb_f,
            checkpoint_every=n_steps_i,
            n_warmup=n_warmup_i,
            design_mask=design_mask,
        )
        return ADMemoryPlan(
            n_steps=n_steps_i,
            available_memory_gb=available_memory_gb_f,
            target_fraction=target_fraction_f,
            target_memory_gb=target_memory_gb,
            checkpoint_every=n_steps_i,
            checkpoint_segments=None,
            checkpoint_mode="checkpoint_every",
            fit_safety_factor=safety_factor_f,
            selected_estimate=largest_chunk_estimate,
            full_ad_fits=False,
            segmented_fits=False,
            recommendation=(
                f"even checkpoint_every={n_steps_i} is estimated at "
                f"{_format_memory_with_safety(largest_chunk_estimate.ad_segmented_gb, safety_factor_f)}, above the "
                f"{_format_memory_gb(target_memory_gb)} target; reduce mesh size, reduce "
                "n_steps, or use a more aggressive memory-reduction lane"
            ),
        )

    def mesh_intelligence_report(
        self,
        *,
        n_steps: int | None = None,
        checkpoint_every: int | None = None,
        checkpoint_segments: int | None = None,
        available_memory_gb: float | None = None,
        n_warmup: int = 0,
        design_mask: jnp.ndarray | np.ndarray | None = None,
        check_ntff: bool = True,
        check_resolution: bool = True,
    ) -> MeshIntelligenceReport:
        """Return a consolidated mesh-quality and memory-planning report.

        The report is intentionally advisory: it reuses ``preflight()``
        for physics/geometry warnings, reuses ``estimate_ad_memory()``
        when ``n_steps`` is provided, and adds a uniform-fine comparator
        that estimates how many cells a globally fine mesh would need if
        it used the minimum cell size present in any non-uniform profile.

        This is the Stage-1 production-near "subgrid-like" planning
        surface: it helps users decide whether existing non-uniform mesh
        plus segmented checkpointing is enough before attempting
        research-only true subgridding.

        ``checkpoint_every`` and ``checkpoint_segments`` are forwarded to the
        same AD estimator used by :meth:`plan_ad_memory`; they are mutually
        exclusive. ``n_warmup`` and ``design_mask`` are forwarded as
        reverse-mode tape metadata only, matching ``estimate_ad_memory``.
        """
        import contextlib
        import io

        if any(
            p is not None and is_tracer(p)
            for p in (self._dx_profile, self._dy_profile, self._dz_profile)
        ):
            raise ValueError(
                "mesh_intelligence_report requires concrete mesh profiles; "
                "tracer-valued mesh-as-design-variable profiles cannot be "
                "summarized host-side."
            )

        dx = self._dx or (C0 / self._freq_max / 20.0)

        def _axis_cells(extent: float, profile) -> int:
            if profile is not None:
                return int(len(profile)) + 1 + 2 * self._cpml_layers
            return int(math.ceil(extent / dx)) + 1 + 2 * self._cpml_layers

        grid_shape = (
            _axis_cells(self._domain[0], self._dx_profile),
            _axis_cells(self._domain[1], self._dy_profile),
            _axis_cells(self._domain[2], self._dz_profile),
        )
        cells = int(grid_shape[0] * grid_shape[1] * grid_shape[2])

        axis_min = [
            float(np.min(self._dx_profile)) if self._dx_profile is not None else dx,
            float(np.min(self._dy_profile)) if self._dy_profile is not None else dx,
            float(np.min(self._dz_profile)) if self._dz_profile is not None else dx,
        ]
        min_cell_size = min(axis_min)
        uniform_fine_shape = tuple(
            int(math.ceil(extent / min_cell_size)) + 1 + 2 * self._cpml_layers
            for extent in self._domain
        )
        uniform_fine_cells = int(
            uniform_fine_shape[0] * uniform_fine_shape[1] * uniform_fine_shape[2]
        )
        cell_savings_factor = (
            float(uniform_fine_cells / cells) if cells > 0 else float("inf")
        )

        # preflight() prints a summary by design; suppress it so the
        # report method remains a pure information-returning API.
        with contextlib.redirect_stdout(io.StringIO()):
            preflight_issues = tuple(
                self.preflight(
                    strict=False,
                    check_ntff=check_ntff,
                    check_resolution=check_resolution,
                    check_ad_memory=False,
                )
            )

        ad_memory = None
        if n_steps is not None:
            ad_memory = self.estimate_ad_memory(
                n_steps,
                available_memory_gb=available_memory_gb,
                checkpoint_every=checkpoint_every,
                checkpoint_segments=checkpoint_segments,
                n_warmup=n_warmup,
                design_mask=design_mask,
            )

        uses_nonuniform = any(
            p is not None
            for p in (self._dx_profile, self._dy_profile, self._dz_profile)
        )
        recommendation_parts: list[str] = []
        if preflight_issues:
            recommendation_parts.append(
                f"resolve {len(preflight_issues)} preflight issue(s) before "
                "trusting physics results"
            )
        elif uses_nonuniform and cell_savings_factor >= 2.0:
            recommendation_parts.append(
                f"non-uniform mesh is useful here: ~{cell_savings_factor:.1f}x "
                "fewer cells than a uniform mesh at the finest spacing"
            )
        elif uses_nonuniform:
            recommendation_parts.append(
                "non-uniform mesh gives limited cell savings; verify that "
                "the refinement profile is worth the added validation burden"
            )
        else:
            recommendation_parts.append(
                "uniform mesh: use preflight/memory estimates as baseline; "
                "consider non-uniform profiles before research-only subgrid"
            )

        if ad_memory is not None:
            if ad_memory.warning:
                recommendation_parts.append(ad_memory.warning)
            elif (checkpoint_every or checkpoint_segments) and ad_memory.ad_segmented_gb is not None:
                recommendation_parts.append(
                    f"use segmented AD estimate ({_format_memory_gb(ad_memory.ad_segmented_gb)}) "
                    "rather than the legacy step-checkpoint heuristic"
                )
            else:
                recommendation_parts.append(
                    "estimated full reverse-mode AD memory is "
                    f"{_format_memory_gb(ad_memory.ad_full_gb)}"
                )

        return MeshIntelligenceReport(
            grid_shape=grid_shape,
            cells=cells,
            uniform_fine_shape=uniform_fine_shape,
            uniform_fine_cells=uniform_fine_cells,
            cell_savings_factor=cell_savings_factor,
            min_cell_size=float(min_cell_size),
            nominal_dx=float(dx),
            uses_nonuniform=uses_nonuniform,
            preflight_issues=preflight_issues,
            ad_memory=ad_memory,
            recommendation="; ".join(recommendation_parts) + ".",
        )
    def _mesh_planner_state(self) -> dict[str, object]:
        """Return a narrow internal snapshot consumed by ``rfx.mesh_planner``.

        This keeps the planner from scattering direct ``Simulation`` private
        attribute reads while avoiding a larger public accessor surface.
        """
        grid = self._build_grid()
        return {
            "freq_max": float(self._freq_max),
            "domain": tuple(float(v) for v in self._domain),
            "boundary": str(self._boundary),
            "cpml_layers": int(self._cpml_layers),
            "pec_faces": tuple(sorted(getattr(self, "_pec_faces", set()))),
            "dx": None if self._dx is None else float(self._dx),
            "dx_profile": self._dx_profile,
            "dy_profile": self._dy_profile,
            "dz_profile": self._dz_profile,
            "dt": float(grid.dt),
        }
    def plan_mesh(
        self,
        *,
        n_steps: int | None = None,
        checkpoint_every: int | None = None,
        available_memory_gb: float | None = None,
        sparameter_calculator: str | None = None,
        artifact_root: str | None = None,
    ) -> MeshPlan:
        """Return an advisory mesh plan for this configured simulation."""
        return plan_simulation_mesh(
            self,
            n_steps=n_steps,
            checkpoint_every=checkpoint_every,
            available_memory_gb=available_memory_gb,
            sparameter_calculator=sparameter_calculator,
            artifact_root=artifact_root,
        )

    def __repr__(self) -> str:
        grid = self._build_grid()
        return (
            f"Simulation(\n"
            f"  freq_max={self._freq_max:.2e} Hz,\n"
            f"  domain={self._domain},\n"
            f"  grid={grid},\n"
            f"  boundary={self._boundary!r},\n"
            f"  materials={len(self._materials)} custom + library,\n"
            f"  geometry={len(self._geometry)} shapes,\n"
            f"  ports={len(self._ports)},\n"
            f"  probes={len(self._probes)},\n"
            f"  dft_planes={len(self._dft_planes)},\n"
            f"  waveguide_ports={len(self._waveguide_ports)},\n"
            f"  floquet_ports={len(self._floquet_ports)},\n"
            f"  periodic_axes={self._periodic_axes!r},\n"
            f"  tfsf={self._tfsf is not None},\n"
            f"  precision={self._precision!r},\n"
            f"  solver={self._solver!r},\n"
            f")"
        )

    @classmethod
    def auto(
        cls,
        freq_range: tuple[float, float],
        *,
        accuracy: str = "standard",
        **kwargs,
    ) -> "Simulation":
        """Create a Simulation with auto-derived parameters.

        Parameters
        ----------
        freq_range : (f_min, f_max) in Hz
            Analysis frequency range. All parameters (dx, domain, CPML,
            n_steps) are derived from this.
        accuracy : "draft", "standard", or "high"
        **kwargs : additional Simulation constructor args (override auto values)

        Returns
        -------
        Simulation with optimal configuration for the frequency range.

        Example
        -------
        >>> sim = Simulation.auto(freq_range=(1.5e9, 3.5e9))
        >>> sim.add(Box(...), material="pec")
        >>> result = sim.run(until_decay=1e-5)
        >>> modes = result.find_resonances()
        """
        import warnings
        warnings.warn(
            "Simulation.auto() called without geometry — the auto-derived "
            "domain and grid may not match your intended structure. "
            "Add geometry after construction or pass domain/dx overrides.",
            stacklevel=2,
        )

        from rfx.auto_config import auto_configure
        config = auto_configure([], freq_range, accuracy=accuracy)

        if config.warnings:
            for w in config.warnings:
                warnings.warn(f"auto_config: {w}")

        sim_kwargs = config.to_sim_kwargs()
        sim_kwargs.update(kwargs)
        return cls(**sim_kwargs)


# ---------------------------------------------------------------------------
# Public API surface for the rfx.api package.
# Data structures are defined in rfx/api/_spec.py and re-exported above;
# Simulation is defined in this module.
# ---------------------------------------------------------------------------
__all__ = [
    "Simulation",
    "MATERIAL_LIBRARY",
    "AD_MemoryEstimate",
    "ADMemoryPlan",
    "ADMemoryComponent",
    "ADMemoryExplainabilityReport",
    "MeshIntelligenceReport",
    "Result",
    "ForwardResult",
    "MaterialSpec",
    "WaveguideSParamResult",
    "WaveguideSMatrixResult",
    "CoaxialSMatrixResult",
    "CoaxialLineReflectionResult",
    "MSLSMatrixResult",
]
