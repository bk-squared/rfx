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

import inspect
import math
import json
import jax
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any

import jax.numpy as jnp
import numpy as np


from rfx.grid import Grid, C0  # noqa: F401
from rfx.geometry.csg import Box, Shape  # noqa: F401
from rfx.nonuniform import NonUniformGrid  # noqa: F401
from rfx.sources.sources import GaussianPulse
from rfx.sources import tfsf_discrete_plane_wave as _discrete_plane_wave
from rfx.realized_geometry import realized_geometry as _realized_geometry_fn
from rfx.fidelity import fidelity_report as _fidelity_report
from rfx.sources.coaxial_port import CoaxialPort
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.lumped import LumpedRLCSpec
from rfx.materials.thin_conductor import (  # noqa: F401
    _PEC_SIGMA_THRESHOLD,
    ThinConductor,
    apply_thin_conductor,
    has_f0_sheets,
)
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
from rfx.boundaries import refresh_boundary_model, require_floquet_lateral_faces
from rfx.boundaries.model import DEFAULT_BOUNDARY

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
    ADMemoryActionHint,
    ADMemoryExplainabilityReport,
    ADMemoryPreflightReport,
    ADCompiledMemoryCertificate,
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
    CoaxialLineReflectionResult,
    CoaxialTwoPortResult,
    _MSLPortEntry,
    MSLProbeClearance,
    MSLSMatrixResult,
    MixedSMatrixResult,
)
from rfx.mesh_planner import MeshPlan, mesh_intelligence_report, plan_simulation_mesh  # noqa: E402,F401
from rfx.auto_config import simulation_auto  # noqa: E402


def _bloch_cutoff_level_db(angle_deg: float, bandwidth: float) -> float:
    """Amplitude at |f0 sin(theta)| relative to f0 for the Bloch drive."""
    # init_tfsf dispatches BOTH Gaussian waveform names to init_tfsf_2d,
    # whose drive is exp(-i*2*pi*f0*t) * exp(-(t/tau)**2),
    # tau = 1/(pi*f0*bandwidth). Thus |S(fc)/S(f0)| = exp(-offset**2).
    # Work directly in dB to avoid underflow for narrow pulses.
    offset = (1.0 - abs(math.sin(math.radians(angle_deg)))) / bandwidth
    return -(20.0 / math.log(10.0)) * offset * offset


from rfx.api._validators import (  # noqa: E402
    _require_integral_param,
    _require_positive_finite_scalar,
    _positive_divisors,
    _preflight_json_safe,
    _validate_residual_context,
)

from rfx.api._compiled_memory import (  # noqa: E402
    _canonicalize_exact_scope,
    _environment_summary,
    _normalize_compiled_memory_analysis,
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
from rfx.api._mesh import _MeshMixin  # noqa: E402
from rfx.api._artifacts import _ArtifactsMixin  # noqa: E402
from rfx.api._ad_memory import (  # noqa: E402
    _AdMemoryMixin,
    _format_memory_gb,
    AD_MEMORY_FIT_SAFETY_FACTOR,
    AD_MEMORY_PREFLIGHT_EVIDENCE_BOUNDARIES,
    AD_COMPILED_MEMORY_CERTIFICATE_EVIDENCE_BOUNDARIES,
)

# ---------------------------------------------------------------------------
# Record-length witness for GRADIENTS, the companion to the -40 dB settling
# witness for VALUES. A free function, not a Simulation method: it takes the
# caller's differentiable objective and the run length it depends on, and
# nothing about it needs this class.
# ---------------------------------------------------------------------------

from rfx.api._gradient_witness import (  # noqa: E402
    GradientRecordLengthWitness,
    gradient_record_length_witness,
)


_DebyeSpec = tuple[list[DebyePole], list[jnp.ndarray]]
_LorentzSpec = tuple[list[LorentzPole], list[jnp.ndarray]]


# ---------------------------------------------------------------------------
# Simulation builder
# ---------------------------------------------------------------------------

class Simulation(
    _MeshMixin,
    _PreflightMixin,
    _SparamMixin,
    _CompileMixin,
    _ExecuteMixin,
    _ArtifactsMixin,
    _AdMemoryMixin,
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
        Number of CPML layers per face. Default 16 (ignored for "pec").
    dx : float or None
        Cell size override (metres). Auto-computed if None.
    mode : str
        ``"3d"`` (default), ``"2d_tmz"`` (Ez, Hx, Hy), or
        ``"2d_tez"`` (Hz, Ex, Ey).
    snap : {"strict", "declared"}
        Strict (default) refuses PEC sheets solved more than 1% off their
        drawn in-plane size. "declared" accepts the difference as a warning
        and records the choice in realized geometry; no solved numbers change.
    dt : float or None
        Concrete time step (s) for the NON-UNIFORM lane, used instead of
        the Courant step derived from the smallest cell. ``None`` (the
        default) keeps the derived step and is byte-identical to before;
        passing it without any of ``dz_profile`` / ``dx_profile`` /
        ``dy_profile`` is refused, because the uniform lane derives its own
        step in ``Grid.courant_dt``.

        Why it exists: the derived step is a function of ``min(cell)``, so a
        mesh that is a design variable moves ``dt`` as it deforms and every
        observable read off the run carries the step change alongside the
        geometry change. One pinned step makes a deformation sweep one board
        at one step. A step above the Courant limit of the REALIZED cells is
        refused, never clipped.
    dt_min_cell : float or None
        Smallest cell size (m) the caller's TRACED axes reach anywhere in
        the family they will run. Required with ``dt`` when a profile is a
        JAX tracer, because a tracer carries no host cell size for the
        limit to be measured against at build time. Concrete axes are always
        measured from their realized profile; a traced axis is measured
        against this declared floor when the run builds its mesh, and the
        run is refused if the realized cells fall below it.
    nominal_profiles : dict or None
        ``{"x" | "y" | "z": concrete cell widths}`` of a TRACED profile at the
        primal point. Ports, probes and sources given in metres on that axis
        resolve to nodes of this profile; the run refuses a node it moves.
    precision : str
        ``"float32"`` (default), ``"mixed"``, or ``"float64"``.  When
        ``"mixed"``, field arrays (E, H) use float16 for ~2x memory
        reduction while material coefficients and DFT accumulators stay
        float32.  When ``"float64"``, field arrays use float64 storage
        AND the Yee update arithmetic runs at float64 (issue #630 — prior
        releases hard-pinned the Yee curl/update arithmetic to float32
        regardless of storage dtype, so float64 fields were silently
        re-quantized to float32 every timestep; the arithmetic dtype is
        now ``jnp.promote_types(field_dtype, jnp.float32)``, so
        ``"float32"``/``"mixed"`` are numerically unaffected and
        byte-identical to before).  ``"float64"`` requires JAX's x64 mode
        to already be enabled by the caller (``jax.config.update(
        "jax_enable_x64", True)`` or ``jax.experimental.enable_x64()``) —
        without it JAX silently downcasts float64 arrays back to
        float32, and ``preflight()`` warns if this mismatch is detected.
        Material coefficients (``eps_r``/``sigma``/``mu_r``) and
        precomputed CPML profile arrays stay float32 either way (fixed
        setup-time constants measurably bias the primal but do not
        create the per-timestep rounding-lattice noise floor that the
        compute-dtype pin did — see issue #630's f64-lift measurement).
        ``"mixed"``/``"float64"`` are UNIFORM-SINGLE-DEVICE-LANE ONLY
        today: ``field_dtype`` is threaded only by
        ``rfx/runners/uniform.py`` — the non-uniform-mesh, distributed
        (``devices=``/``distributed=True``), and subgridded
        (``refinement``) runners do not thread it, so a non-float32
        precision there would silently run float32 fields. ``run()`` /
        ``forward()`` raise ``NotImplementedError`` rather than do that
        silently; ``preflight()`` also warns in advance for the
        non-uniform-mesh case (the distributed case is a call-time
        ``run()``/``forward()`` kwarg, invisible to preflight, so its
        `NotImplementedError` at dispatch is the only enforcement point).

        ACCURACY CAVEAT for ``"mixed"`` with a CPML boundary (issue #644):
        float16 field storage is fine for transient and field-shape work
        (measured 0.20% relative L2 error on Ez vs ``"float32"`` at 50
        steps), but it raises the absorber's residual floor. On the
        fixture in ``tests/unit/grid/test_mixed_precision.py`` — a 40**3 domain,
        dx = 3.0 mm, point source, 400 steps — total field energy settles
        to about -76 dB below peak under ``"float32"`` but only about
        -59.5 dB under ``"mixed"``: roughly 17 dB worse, consistent with
        float16's ~1e-3 machine epsilon quantizing the stored fields.
        Those two numbers are from that one fixture, not a universal
        bound; the floor scales with float16 epsilon, so expect the same
        order wherever the field storage is float16. Practical rule:
        ``"mixed"`` is not a drop-in ``"float32"`` substitute when the
        quantity of interest is a reflection coefficient, an S-parameter,
        or anything else living near the absorber floor — use
        ``"float32"`` (the default) for those.
    solver : str
        ``"yee"`` (default) for the standard explicit scheme or
        ``"adi"`` for the experimental ADI-FDTD path. The homogeneous,
        lossless split with compatible domain boundaries removes the
        explicit CFL stability restriction. Interior PEC (sheets, wires,
        volumes) is refused at every factor on both lanes: its projection
        does not inherit that guarantee. Cavity accuracy without interior
        PEC is tested in 2D TMz (``mode="2d_tmz"``, 2% resonance gate at
        5x CFL, ``test_adi_cavity_resonance``) and 3D
        (full Zheng–Chen–Zhang two-sub-step scheme since
        2026-07-13, issue #338: 2% PEC-cavity eigenfrequency gate at 2x
        CFL, ~15 cells/wavelength). 3D dispersion error grows ~dt^2, so
        preflight emits an ``adi_3d_accuracy`` envelope advisory when
        ``adi_cfl_factor > 2`` on a 3D grid — large factors trade
        wavelength-scale accuracy for stiff-mesh throughput.
    adi_cfl_factor : float
        Timestep multiplier relative to the grid's Yee CFL timestep when
        ``solver="adi"``. Default 2.0; for quantitative wavelength-scale
        3D results prefer <= 2.0 (see ``solver``).
    stencil_order : int
        Spatial finite-difference order for the explicit Yee update: ``2``
        (default, the standard (2,2) scheme — BYTE-IDENTICAL to prior
        releases) or ``4`` (the (2,4) fourth-order-in-space stencil). Order
        4 is supported ONLY on the plain uniform Cartesian vacuum/dielectric
        path: boundary ``pec``/``periodic`` (NOT cpml/upml), default ``yee``
        solver (NOT adi), uniform mesh (no dz/dx/dy profile), no dispersive
        materials, no anisotropic/subpixel eps, no conformal PEC, no Kerr,
        not distributed. Any unsupported combination with
        ``stencil_order=4`` raises ``NotImplementedError``. Order 4 derates
        the timestep by ~0.857x for stability (the (2,4) CFL bound).
    """

    def __init__(
        self,
        freq_max: float,
        domain: tuple[float, float, float],
        *,
        boundary: str | BoundarySpec | dict = DEFAULT_BOUNDARY,
        cpml_layers: int = 16,
        cpml_kappa_max: float = 1.0,
        dx: float | None = None,
        mode: str = "3d",
        dz_profile: np.ndarray | None = None,
        dx_profile: np.ndarray | None = None,
        dy_profile: np.ndarray | None = None,
        dt: float | None = None,
        dt_min_cell: float | None = None,
        nominal_profiles: dict | None = None,
        precision: str = "float32",
        solver: str = "yee",
        adi_cfl_factor: float = 2.0,
        stencil_order: int = 2,
        interface_eps: str = "sampled",
        snap: str = "strict",
        **_removed_kwargs,
    ):
        if "pec_faces" in _removed_kwargs:
            raise TypeError(
                "Simulation(pec_faces=...) was removed; use BoundarySpec PEC faces "
                "via boundary=BoundarySpec(...)."
            )
        if _removed_kwargs:
            raise TypeError(
                f"Simulation() got unexpected keyword arguments: {sorted(_removed_kwargs)}"
            )
        from rfx.boundaries.spec import normalize_boundary
        from rfx.runners.nonuniform import INTERFACE_EPS_RULES

        if snap not in ("strict", "declared"):
            raise ValueError(f"snap must be 'strict' or 'declared', got {snap!r}")
        self._snap = snap

        if interface_eps not in INTERFACE_EPS_RULES:
            raise ValueError(f"interface_eps must be one of {INTERFACE_EPS_RULES}, got {interface_eps!r}")
        self._interface_eps = interface_eps

        # T7-B: accept BoundarySpec directly or normalise a legacy scalar
        # boundary=<str>. A BoundarySpec provided here is authoritative;
        # removed keywords are rejected above.
        _explicit_spec = isinstance(boundary, (BoundarySpec, dict))
        self._boundary_explicit = boundary is not DEFAULT_BOUNDARY
        _boundary_origin = "declared" if self._boundary_explicit else "default"
        if boundary is DEFAULT_BOUNDARY:
            boundary = "cpml"
        if _explicit_spec:
            spec = normalize_boundary(boundary)
        else:
            # Legacy scalar path — validated below, lifted to BoundarySpec
            # after scalar boundary fields have been resolved.
            if boundary not in ("pec", "cpml", "upml"):
                raise ValueError(
                    f"boundary must be 'pec', 'cpml', or 'upml', got {boundary!r}"
                )
            spec = None  # deferred; folded in after legacy fields settle
        if freq_max <= 0:
            raise ValueError(f"freq_max must be positive, got {freq_max}")
        if precision not in ("float32", "mixed", "float64"):
            raise ValueError(
                f"precision must be 'float32', 'mixed', or 'float64', got {precision!r}"
            )
        if solver not in ("yee", "adi"):
            raise ValueError(f"solver must be 'yee' or 'adi', got {solver!r}")
        if adi_cfl_factor <= 0:
            raise ValueError(f"adi_cfl_factor must be positive, got {adi_cfl_factor}")
        if stencil_order not in (2, 4):
            raise ValueError(f"stencil_order must be 2 or 4, got {stencil_order}")
        from rfx.model import admit_graded_profiles
        # Caps come from _PreflightMixin (single source of truth; preflight's
        # _validate_cfg_multiband_grading reads the same two attributes).
        domain, self._nominal_profiles = admit_graded_profiles(
            domain, dx_profile, dy_profile, dz_profile, dt=dt,
            dt_min_cell=dt_min_cell, nominal_profiles=nominal_profiles,
            multiband_ratio_cap=self._MULTIBAND_RATIO_CAP,
            inplane_ratio_cap=self._INPLANE_RATIO_CAP)

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
            self._pec_faces = set()

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
        # Pinned time step (non-uniform lane only); admitted above.
        self._dt_pin = dt
        self._dt_min_cell = dt_min_cell
        self._precision = precision
        self._solver = solver
        self._adi_cfl_factor = adi_cfl_factor
        # (2,4) fourth-order-in-space stencil (PR-1b). 2 = default,
        # byte-identical. 4 is reachable ONLY on the plain uniform Cartesian
        # vacuum/dielectric path; the comprehensive API fence
        # (_check_stencil_order_supported) rejects every unsupported runner.
        self._stencil_order = stencil_order

        if self._solver == "adi":
            if self._mode not in ("2d_tmz", "3d"):
                raise ValueError("solver='adi' supports mode='3d' or mode='2d_tmz'")
            if self._boundary == "upml":
                raise ValueError("solver='adi' does not support boundary='upml'")
            from rfx._adi_notice import require_closed_boundary
            require_closed_boundary(self)
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
        # Node-pinned PEC sheets (add_pinned_sheet): declared by node index,
        # so they are the same conductor on every mesh — including one that
        # is a JAX tracer, where a metric sheet cannot be resolved at all.
        self._pinned_sheets: list = []
        self._coaxial_ports: list[CoaxialPort] = []
        self._ntff: tuple | None = None  # (corner_lo, corner_hi, freqs)
        # (corner_lo, corner_hi, block_size, freqs, order, extra) — the
        # declaration only; the slab and the block map are realized against
        # the built grid in ``rfx.current_moments.monitor_for_simulation``.
        self._current_moments: tuple | None = None
        self._tfsf: _TFSFEntry | None = None
        self._dft_planes: list[_DFTPlaneEntry] = []
        # ``_dft_plane_regions`` (runtime-only crop metadata for the internal
        # MSL DFT planes) is deliberately NOT set here: it exists only while
        # ``compute_msl_s_matrix`` runs and is removed again on exit, so it
        # never enters the design-IR ledger (rfx/interop/_design.py rule 4).
        # Public add_dft_plane_probe registrations stay full-plane.
        self._flux_monitors: list[_FluxMonitorEntry] = []
        self._waveguide_ports: list[_WaveguidePortEntry] = []
        self._msl_ports: list[_MSLPortEntry] = []
        # Issue #469/#470 runtime bookkeeping. Deliberately NOT entry
        # fields: _MSLPortEntry's and _ProbeEntry's field sets are pinned
        # by the interop design-IR contract (rfx/interop/_design.py
        # _PINNED_RECORDS), and neither of these is design state.
        #   _msl_auto_offset_min: port name -> the upstream-only lower
        #     edge stored when n_probe_offset was auto-resolved; the #469
        #     interval solve in compute_msl_s_matrix starts from it.
        #   _msl_auto_probe_spacing: port name -> the registration-time
        #     Hammerstad-Jensen eps_eff estimate, stored when
        #     n_probe_spacing was auto. The #681 span solve in
        #     _resolve_msl_auto_offsets widens the auto spacing toward
        #     span = (N-1)*lambda_g(f_max)/4 where the geometry allows;
        #     the entry keeps the conservative registration default so
        #     resolver-skip paths (a ladder crossing a grading ramp) never
        #     overrun the feed.
        #   _msl_auto_probe_lengths: port name -> the two LENGTHS the
        #     automatic probe defaults are counted from at registration,
        #     (lambda_eff/(4*pi), lambda_eff/8) at f_max, stored for every
        #     port. On a graded propagation axis _resolve_msl_auto_offsets
        #     counts them in the runway cell at the port, and preflight asks
        #     from them what an automatic offset would be (#810).
        #   _internal_probe_indices: indices into self._probes of
        #     library-registered diagnostic probes (MSL settling
        #     witnesses); probe-placement preflight advisories and the
        #     #332 tail advisory skip them (issue #470 self-noise).
        self._msl_auto_offset_min: dict[str, int] = {}
        self._msl_auto_probe_spacing: dict[str, float] = {}
        self._msl_auto_probe_lengths: dict[str, tuple[float, float]] = {}
        self._internal_probe_indices: set[int] = set()
        self._periodic_axes: str = ""
        self._refinement: dict | None = None
        self._lumped_rlc: list[LumpedRLCSpec] = []
        self._floquet_ports: list[_FloquetPortEntry] = []

        # Canonical BoundarySpec, with scalar boundary compatibility.
        if _explicit_spec:
            self._boundary_spec = spec
            self._periodic_axes = spec.periodic_axes()
        else:
            self._boundary_spec = self._build_spec_from_legacy()

        from rfx.boundaries.model import Features, resolve_kinds
        self._boundary_model = resolve_kinds(
            self._boundary_spec, mode=mode,
            features=Features(layers=cpml_layers,
                              absorber_parameters=(("kappa_max", cpml_kappa_max),),
                              explicit_faces=_explicit_spec, origin=_boundary_origin,
                              face_origins=(),
                              domain=domain),
        )

        # solver='adi' has no per-face absorber. Its absorbing layer is a
        # graded sigma stamped on ALL SIX faces at the scalar cpml_layers
        # (rfx/adi.py make_adi_absorbing_sigma{,_3d}, reached from
        # rfx/api/_execute.py under `self._boundary == "cpml"`), and
        # `self._boundary` is 'cpml' as soon as ANY face absorbs. So a
        # per-face spec with PEC faces, or with a per-face thickness
        # override, would silently get an absorber where the caller asked
        # for a reflector — the SILENT_WRONG class preflight exists to
        # remove (issue #647 grep sweep). Reject it instead of absorbing
        # into a wall.
        if self._solver == "adi" and self._cpml_layers > 0:
            _adi_faces = {
                f"{ax}_{side}":
                    getattr(getattr(self._boundary_spec, ax), side)
                for ax in "xyz" for side in ("lo", "hi")
            }
            from rfx.boundaries.depths import adi_uniform_faces
            if not adi_uniform_faces(self._boundary_spec, cpml_layers):
                _layout = ", ".join(
                    f"{face}={tok!r}"
                    for face, tok in sorted(_adi_faces.items())
                )
                raise ValueError(
                    "solver='adi' supports only a uniform absorber: its "
                    "absorbing layer is stamped on all six faces at the "
                    "scalar cpml_layers, so a per-face boundary layout "
                    f"({_layout}) would silently absorb on the faces you "
                    "declared as reflectors. Use solver='yee' for per-face "
                    "boundaries, or make every face 'cpml' with no per-face "
                    "thickness override."
                )

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

        Refinement acts only through ``run()`` on a uniform mesh. Other
        solve entry points, including ``forward()`` and optimization, refuse it.

        The experimental runner covers the specified z-range across the
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
            Production refuses the unstable, unverified subgridded lane before
            stepping (#1465). An empty PEC cavity's first resonance is 8 % high
            and its field grows. Use dx/dy/dz profiles for local resolution.
            Explicit ``"research"`` or ``"off"`` runs with a warning; neither
            mode establishes physics support.
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
        # Warn if subgrid overlaps the PML region.
        # PML operates on the coarse grid only; the fine grid has no PML.
        # Overlapping causes late-time energy growth (SAT coupling feeds
        # energy into the PML boundary faster than it can absorb).
        #
        # Coordinate frame (issue #466): the grid builder places the absorber
        # OUTSIDE the user domain — ``domain`` is entirely physical, the CPML
        # cells are extra padding beyond it, and ``(idx - axis_pads)*dx``
        # recovers user coordinates (rfx/grid.py). The pre-#466 check used the
        # opposite frame (absorber inside the first/last N domain cells) and
        # so fired on geometry that was clear of the absorber. In the
        # builder's frame a z_range meets the absorber only when it reaches
        # the physical-domain edge of an ABSORBING face: z_lo <= 0 (touching
        # or entering the z_lo padding) or z_hi >= domain_z. Touching counts —
        # the SAT interface then sits directly against the CPML interface,
        # and the authoritative static validator flags the same geometry as
        # ``subgrid_overlaps_absorber`` (see
        # test_production_boundary_terminated_rejects_refined_face_touching_cpml,
        # which pins this warning for the touching case). What the frame fix
        # REMOVES is the interior false positive: a z_range strictly inside
        # (0, domain_z) is clear of the absorber no matter how close to the
        # edge, because the absorber cells are padding outside. The subgrid
        # lane stays EXPERIMENTAL (PR #90).
        if self._boundary in ("cpml", "upml") and self._cpml_layers > 0:
            import warnings
            z_lo_kind = self._boundary_spec.z.lo if self._boundary_spec else self._boundary
            z_hi_kind = self._boundary_spec.z.hi if self._boundary_spec else self._boundary
            zlo_absorbing = z_lo_kind in ("cpml", "upml")
            zhi_absorbing = z_hi_kind in ("cpml", "upml")
            domain_z = self._domain[2] if len(self._domain) > 2 else 0
            z_lo, z_hi = z_range
            if (zlo_absorbing and z_lo <= 0.0) or (
                zhi_absorbing and domain_z > 0 and z_hi >= domain_z
            ):
                warnings.warn(
                    f"Subgrid z_range=({z_lo*1e3:.1f}, {z_hi*1e3:.1f})mm overlaps PML "
                    f"absorber padding: it reaches the edge of the physical domain "
                    f"(0.0, {domain_z*1e3:.1f})mm on an absorbing z face (the "
                    f"absorber lies OUTSIDE the domain — rfx/grid.py convention). "
                    f"This causes late-time energy growth. Keep z_range strictly "
                    f"inside the physical domain or use PEC z faces.",
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

    from rfx.subgridding.validation import validate_subgrid

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
        """Add a geometric shape filled with a named material.

        Parameters
        ----------
        shape : Shape
        material : str
            Registered or library material name.

        Notes
        -----
        A PEC shape is a VOLUME under the lattice ownership contract
        (#931): its primal cells are sampled at cell centres and every E
        edge incident to an occupied cell is shorted, so a Box drawn
        ``z_a -> z_b`` on node planes realizes walls at BOTH planes with
        realized thickness = drawn thickness. A PEC Box with exactly one
        zero-extent axis (``lo == hi``) is a SHEET declaration, realized
        exactly as :meth:`add_thin_conductor` would realize it. A PEC Box
        thinner than one local cell (but not zero) is refused at
        assembly; so is any PEC shape that rasterizes to zero cells.
        There is no per-entry realization knob.
        """
        self._resolve_material(material)  # validate early
        self._geometry.append(_GeometryEntry(
            shape=shape, material_name=material))
        return self

    def realized_geometry(self):
        """Return the immutable host record of this configuration's built geometry.

        Build only: entities, solved sheet spans, signed face residuals in
        metres, domain padding, and driven port edges. Dense diagnostic arrays
        are built on request here. ``run()`` attaches a compact record built
        independently from its own assembly as ``Result.realized_geometry``.
        """
        return _realized_geometry_fn(self)

    def fidelity_report(self, print_report: bool = True):
        """Input-fidelity audit: declared vs solved, per entity, in input
        units — run BEFORE any solve. See :func:`rfx.fidelity.fidelity_report`
        (design rule: never predicts result-side impact; rasterization,
        realization class and materialization only)."""
        return _fidelity_report(self, print_report=print_report)

    # ---- sources (non-port) ----

    def add_source(
        self,
        position: tuple[float, float, float],
        component: str = "ez",
        *,
        waveform=None,
        amplitude_kind: str | None = None,
    ) -> "Simulation":
        """Add a soft point source (no impedance loading).

        Unlike ``add_port()``, this does NOT add a resistive load.
        Ideal for resonance characterization where port loading would
        damp the cavity response.

        Uses a differentiated ``GaussianPulse`` by default (near-zero DC
        content, ~1e-4). This prevents static charge accumulation on PEC
        surfaces.

        Parameters
        ----------
        position : (x, y, z) in metres
        component : "ex", "ey", or "ez"
        waveform : excitation pulse
            (default: ``GaussianPulse(f0=freq_max/2, bandwidth=0.8)``)
        amplitude_kind : 'field' | 'current' | None
            What the waveform amplitude MEANS — issue #571. Boundary- and
            mesh-independent once explicit:

            - ``'current'``: the amplitude is a current moment I(t) in A·m,
              realized as ``E += Cb * I / dV`` on every path and boundary
              (Yee ``Cb = (dt/eps)/(1 + sigma*dt/(2*eps))``; ADI refuses
              ``'current'`` and requires explicit ``'field'``,
              ``dV`` = local cell volume). Resolution-independent injected
              power; Meep's convention; the declaration default.
            - ``'field'``: the amplitude is a raw E-field increment per
              step, ``E += w(t)`` on every path and boundary.

            Cell-volume convention: ``dV = dx * dy * dz`` at the source
            cell; a 2D grid is treated as ONE CELL DEEP (``dz`` one cell,
            duck-typed to ``dx``), so ``dV = dx**3`` on a cubic 2D grid,
            not the per-unit-length ``dx**2``
            (``rfx/api/_source_semantics.py``). On a non-uniform mesh
            ``dV`` is the E node's control volume — the primal per-cell
            width on the component's own axis and the DUAL spacing
            ``(d[k-1]+d[k])/2`` on the two transverse axes (issue #672);
            the two coincide on a uniform profile.

            ``None`` means ``'current'`` (E += Cb*I/dV, I is a current moment in A·m)
            on every path and emits one
            :class:`DeprecationWarning` per Simulation. Pass the kind
            explicitly to silence it.

            The open-boundary + ``subpixel_smoothing`` cross-path residual
            (0.18 % amplitude, issue #582) is a solver defect independent
            of this parameter.
        """
        if component not in ("ex", "ey", "ez"):
            raise ValueError(f"component must be ex/ey/ez, got {component!r}")
        from rfx.api._source_semantics import (
            resolve_amplitude_kind, legacy_kind_description)
        resolved_kind = resolve_amplitude_kind(amplitude_kind)
        if amplitude_kind is None and not getattr(
                self, "_amplitude_kind_warned", False):
            self._amplitude_kind_warned = True
            import warnings
            warnings.warn(
                "add_source(..., amplitude_kind=None) now means "
                f"{legacy_kind_description(False, self._boundary)}. "
                "Pass amplitude_kind explicitly to silence this warning.",
                DeprecationWarning, stacklevel=2)
        if waveform is None:
            waveform = GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)

        # Store as a source entry (reuse _PortEntry with impedance=None flag)
        self._ports.append(_PortEntry(
            position=position, component=component,
            impedance=0.0,  # 0 = no port impedance (soft source)
            waveform=waveform, extent=None,
            amplitude_kind=resolved_kind,
        ))
        return self

    def add_polarized_source(
        self,
        position: tuple[float, float, float],
        *,
        polarization: str | tuple = "ez",
        waveform=None,
        amplitude_kind: str | None = None,
    ) -> "Simulation":
        """Add a polarized point source.

        Parameters
        ----------
        position : (x, y, z) in metres
        polarization : str or tuple
            - "ez", "ex", "ey" — linear single-component
            - "circular" or "rhcp" — accepted complex-component shortcut;
              time-domain quadrature is not independently verified
            - "lhcp" — accepted complex-component shortcut; same limitation
            - (Ex, Ey) tuple — normalized component weights; real tuples are
              the documented linear-polarization scope
            - "slant45" — 45° linear (Ex = Ey)
        waveform : excitation pulse (default: GaussianPulse)
        amplitude_kind : 'field' | 'current' | None
            Threaded through to every internal :meth:`add_source` call —
            see ``add_source`` for the semantics. ``None`` resolves to
            ``'current'`` and warns once per simulation; pass the kind explicitly
            to silence the warning.
        """
        if waveform is None:
            waveform = GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)

        if isinstance(polarization, str):
            if polarization in ("ex", "ey", "ez"):
                self.add_source(position, polarization, waveform=waveform,
                                amplitude_kind=amplitude_kind)
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

        # Complex weights use two separately constructed waveforms. Their
        # quadrature is not part of the verified public polarization scope.
        if _np.isreal(jx) and _np.isreal(jy):
            # Real Jones — simple amplitude scaling. Thread the source
            # waveform's cutoff through the reconstruction (post-#392
            # review): dropping it silently reverted the #388/#392
            # deposited-DC remedy (cutoff=4.5) to the default 3.0 on
            # this path. getattr default 3.0 = GaussianPulse's default,
            # so cutoff-less waveforms are unchanged.
            if abs(float(jx.real)) > 1e-10:
                from rfx.sources.sources import GaussianPulse as GP
                wx = GP(f0=waveform.f0, bandwidth=waveform.bandwidth,
                        amplitude=waveform.amplitude * float(jx.real),
                        cutoff=getattr(waveform, 'cutoff', 3.0))
                self.add_source(position, "ex", waveform=wx,
                                amplitude_kind=amplitude_kind)
            if abs(float(jy.real)) > 1e-10:
                from rfx.sources.sources import GaussianPulse as GP
                wy = GP(f0=waveform.f0, bandwidth=waveform.bandwidth,
                        amplitude=waveform.amplitude * float(jy.real),
                        cutoff=getattr(waveform, 'cutoff', 3.0))
                self.add_source(position, "ey", waveform=wy,
                                amplitude_kind=amplitude_kind)
        else:
            # Complex Jones — build carrier-modulated components.
            from rfx.sources.sources import ModulatedGaussian as MG
            # Ex component (reference phase)
            if abs(jx) > 1e-10:
                wx = MG(f0=waveform.f0, bandwidth=waveform.bandwidth,
                        amplitude=waveform.amplitude * float(abs(jx)))
                self.add_source(position, "ex", waveform=wx,
                                amplitude_kind=amplitude_kind)
            # Ey component (90° phase shift for circular)
            if abs(jy) > 1e-10:
                # Phase of jy relative to jx
                import cmath
                phase = cmath.phase(jy) - cmath.phase(jx) if abs(jx) > 1e-10 else 0
                # Construct the second carrier from the requested relative phase.
                from rfx.sources.sources import CustomWaveform
                import jax.numpy as _jnp
                tau = 1.0 / (waveform.f0 * waveform.bandwidth * 3.14159)
                t0 = 3.0 * tau
                amp_y = waveform.amplitude * float(abs(jy))
                def _ey_func(t):
                    envelope = _jnp.exp(-((t - t0) / tau)**2)
                    carrier = _jnp.cos(2.0 * _jnp.pi * waveform.f0 * t + float(phase))
                    return amp_y * carrier * envelope
                self.add_source(position, "ey", waveform=CustomWaveform(func=_ey_func),
                                amplitude_kind=amplitude_kind)

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
        terminates=None,
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
        from rfx.geometry.port_termination import resolve_terminates
        terminated = resolve_terminates(
            self, terminates, port=f"add_coaxial_port at {position}")

        self._coaxial_ports.append(CoaxialPort(
            position=position,
            face=face,
            pin_length=pin_length,
            pin_radius=pin_radius,
            outer_radius=outer_radius,
            impedance=impedance,
            excitation=waveform,
            terminates=terminated,
        ))
        return self

    # ---- thin conductors ----

    def add_thin_conductor(
        self,
        shape: Shape,
        *,
        sigma_bulk: float = 5.8e7,
        thickness: float = 35e-6,
        eps_r: float = 1.0,
        surface_impedance_f0: float | None = None,
    ) -> "Simulation":
        """Add a thin conductor with subcell correction.

        Parameters
        ----------
        shape : Shape
            Geometric region of the conductor.
        sigma_bulk : float
            Bulk conductivity (S/m). Default: copper (5.8e7).

            **This single value decides the model.** At or above 1e6 S/m the
            shape becomes a PEC sheet and ``thickness`` and ``eps_r`` are
            never read (:func:`rfx.materials.thin_conductor.apply_thin_conductor`
            returns the material arrays untouched and only ORs the mask). Below
            1e6 S/m the shape becomes a lossy sheet whose conductivity is
            ``sigma_bulk * thickness / d_norm``, where ``thickness`` IS
            load-bearing and ``d_norm`` is the local sheet-normal spacing
            (see ``surface_impedance_f0`` below for the graded-mesh case).
            Every real metal — copper 5.8e7, aluminium 3.5e7, even stainless
            steel 1.4e6 — is on the PEC side, so by default
            (``surface_impedance_f0`` unset) for metals this call models a
            lossless perfect sheet and nothing else. Pass
            ``surface_impedance_f0`` to opt in to band-centre Leontovich
            conductor loss instead (see below).
        thickness : float
            Physical thickness (metres). Default: 35 µm (1 oz copper).
            **Only used on the lossy path** (``sigma_bulk < 1e6``); ignored
            entirely for metals — see ``sigma_bulk`` above.
        eps_r : float
            Relative permittivity (default 1.0). Lossy path only, same as
            ``thickness``.
        surface_impedance_f0 : float or None
            Opt-in band-centre Leontovich surface-impedance loss
            (issue #669). Default ``None`` = exact legacy behaviour on every
            path. When set to a positive frequency (Hz), the sheet — for ANY
            ``sigma_bulk > 0``, metals included — is modelled as a resistive
            sheet of sheet resistance ``Rs0 = sqrt(pi*f0*mu0/sigma_bulk)``
            (the thick-conductor surface resistance at ``f0``), realized
            NODE-THIN (#677): a per-step operator applies an exact
            exponential-stepping resistive update ``E = A*E + B*curlH``
            with sheet conductivity ``sigma_sheet = (1/Rs0)/d_norm`` on
            exactly the tangential E edges a PEC sheet on the same cells
            would zero — so toggling ``surface_impedance_f0`` on a metal
            changes LOSS, never the conductor's surface position (before #677 the sheet folded into ``materials.sigma`` as a full-cell
            slab, moving resonances by geometry). ``d_norm`` is the local
            spacing normal to the sheet: the cell size on a uniform grid,
            and on a graded one the E-node DUAL spacing ``(d[k-1]+d[k])/2``
            (#671). Both the uniform and the non-uniform (``dz_profile``)
            ``run()``/S-parameter lanes apply the operator; lanes that do
            not (distributed, subgridded, ADI, UPML, dispersive-overlap,
            multimode-waveguide, MSL-junction/mixed, optimize/topology
            drivers) refuse f0 sheets loudly rather than silently dropping
            them; ``compute_msl_s_matrix`` applies the sheet through its
            ``run()``/``forward()`` device dispatches (#679). The sheet
            contributes NO PEC cells and no ``materials.sigma``/``eps_r``
            entries.

            **Consequence for material inspection (#695): an f0 sheet is
            INVISIBLE to the assembled arrays.** It is absent from
            ``pec_mask`` and it adds nothing to ``materials.sigma``, so a
            conductor test written the obvious way --
            ``pec_mask | (sigma > 1e3)`` -- finds no metal at all on a
            board whose traces are all f0 sheets, and a connectivity
            check built on it reports a perfectly healthy model as
            disconnected. Use :meth:`Simulation.conductor_mask`, which
            returns ``pec_mask | (sigma > threshold) | union(f0 sheet
            cell masks)`` -- the whole conductor footprint in one
            spelling, on the grid the run will actually use. The runtime
            edge masks (``mask_ex/ey/ez``) live on the
            ``SheetImpedanceCtx`` the runners build; the accessor returns
            the CELL footprint, which is what an occupancy or
            connectivity check wants.

            Model scope (be precise about what you get):

            - **Magnitude loss only.** The reactive part ``Xs = Rs``
              (internal inductance) is NOT modelled, so resonances read
              slightly high in frequency.
            - **Frequency-flat within a run**: ``Rs`` is frozen at ``f0``
              with relative band error ``|sqrt(f/f0) - 1|`` (about +-11%
              across an 8-12 GHz band centred at 10 GHz).
            - **Thick-conductor model**: valid for ``thickness >= ~3``
              skin depths at ``f0``. For thin films the DC path (omit this
              kwarg, ``sigma_bulk < 1e6``) is the correct model;
              ``thickness`` deliberately does NOT enter ``sigma_eff`` in f0
              mode (Leontovich loss is thickness-independent).
            - **Transmission leakage** ``(2*Rs/eta0)**2`` makes the sheet
              unusable for shielding-effectiveness claims beyond ~60 dB.
              (Validated against the closed form ``T = 2Rs/(2Rs+eta0)``
              to within a few percent, frequency-flat —
              tests/oracle/test_leontovich_alpha_oracle.py.)
            - The operator touches TANGENTIAL sheet edges only; the
              sheet-normal E component is never loaded (#677 removed the
              pre-existing isotropic-fold caveat).
            - ``f0`` and ``sigma_bulk`` are differentiable DoFs in f0 mode
              (closed forms ``d(sigma_sheet)/df0 = -sigma_sheet/(2*f0)``,
              ``d(sigma_sheet)/d(sigma_bulk) =
              +sigma_sheet/(2*sigma_bulk)``). Geometry (the rasterized
              mask) stays non-differentiable. Traced values skip the
              concrete-value validation below.

            Sheet shapes (issue #674): ANY shape implementing
            ``mask_on_coords`` and ``bounding_box`` may carry a
            surface-impedance sheet — a patterned ground plane with
            clearance holes, a meandered arm, an imported CAD outline
            (:class:`~rfx.geometry.mesh_import.MeshShape`), a disc, a
            strip. The realization is per occupied cell, so a hole is
            simply a cell the operator never touches. What is NOT supported
            is a body with HEIGHT: the sheet must rasterize to exactly ONE
            cell layer along its normal, because ``sigma_sheet`` is the
            conductance of a single E node. A 3-D solid, a bent/L-shaped sheet, or a slab
            thicker than a cell raises ``ValueError`` when the grid is
            built — folding such a body per cell would multiply the sheet
            conductance by the layer count while still reporting ``Rs0``.
            A sheet that rasterizes to ZERO cells raises too: a sub-cell
            mesh slab registers only where a grid node falls inside it, and
            silently vaporized metal is the #369 class. (A ``Box`` drawn
            with equal lo/hi on its normal axis snaps to the nearest node by
            construction, so it always satisfies both.)

            Validation at add time (concrete values only): ``f0 <= 0`` and
            ``sigma_bulk <= 0`` with ``f0`` set each raise ``ValueError``,
            as does a shape with no ``mask_on_coords`` or no bounding box.
            The rasterized one-layer and non-empty checks need the grid, so
            they raise at build time, on BOTH lanes. Non-Box DC films use
            the same layer and footprint admission on both grids.

        Notes
        -----
        A PEC sheet occupying ONE cell is modelled as a surface, not as a
        conductor of that cell's thickness: :func:`rfx.boundaries.pec.apply_pec_mask`
        zeroes a tangential E component only where the mask has a neighbour
        along that component's axis, so the normal component survives as
        surface charge. Stamping the same conductor TWO cells thick therefore
        changes the model rather than the thickness. Measured on an MSL thru at
        dx = 84.67 µm: one cell gives Re(Z0) = 44.1 Ω and two cells 41.9 Ω.
        Reproduce with
        ``scripts/diagnostics/thin_conductor_cell_thickness_probe.py``.

        Note this pair no longer *proves* the model change on its own. Until
        issue #511 was fixed the one-cell reading was 38.8 Ω, so the thicker
        stamp appeared to read the HIGHER impedance — backwards for a
        fringing-capacitance effect, which is how the model change was first
        argued. That inversion was an extractor artifact: the modal voltage
        summed one Ez edge too many, and a two-cell stamp happens to mask that
        edge, so only the one-cell number was biased. With both corrected the
        ordering is the ordinary one (thicker → lower Z0) and the 2.25 Ω step
        is consistent with either a model change or a genuine thickness
        effect. The model change is still real — it follows from
        ``apply_pec_mask`` directly — but this measurement no longer
        discriminates between the two.

        Conductor LOSS is therefore unavailable for metals through this API.
        On the lanes this repo gates that is a small omission — copper loss on
        a 10 mm 50 Ω microstrip is 0.04-0.05 dB at 3-4.5 GHz against a 0.45 dB
        gate budget, and the metal is 26-36 skin depths thick so its geometry
        is electromagnetically a half-space — but it is a real limit for
        anything whose loss budget IS the conductor, e.g. resonator Q.
        """
        from rfx.model import admit_thin_conductor
        tc = admit_thin_conductor(
            shape, sigma_bulk=sigma_bulk, thickness=thickness, eps_r=eps_r,
            surface_impedance_f0=surface_impedance_f0,
        )
        self._thin_conductors.append(tc)
        return self

    def add_pinned_sheet(
        self,
        *,
        plane_index: int,
        i_range: tuple[int, int],
        j_range: tuple[int, int],
        normal_axis: int = 2,
        sigma_bulk: float = 5.8e7,
        thickness: float = 35e-6,
        eps_r: float = 1.0,
        surface_impedance_f0: float | None = None,
        name: str | None = None,
    ) -> "Simulation":
        """Declare a thin PEC conductor by NODE INDICES instead of metres.

        ``add_thin_conductor`` draws metal in metres, and the grid then
        decides which node lines that reaches. This draws it on the node
        lines directly: the conductor occupies node ``plane_index`` along
        ``normal_axis`` and the inclusive node ranges ``i_range`` /
        ``j_range`` on the other two axes, in increasing axis order — for
        the default ``normal_axis=2`` that is ``i_range`` on x and
        ``j_range`` on y.

        Indices are INTERIOR (unpadded) node indices: interior node ``i``
        sits at ``i * dx`` from the domain origin on a uniform mesh, and is
        the ``i``-th interior node on a graded one. Absorber padding is
        added when the sheet is realized.

        Why it exists: the conductor's PHYSICAL size is then whatever the
        cells between those nodes add up to, so stretching the cells makes
        the metal longer without moving it off the lattice and without any
        sub-cell metal. That is what lets ``jax.grad`` reach a patch edge
        through ``dx_profile`` / ``dy_profile`` / ``dz_profile`` — a metric
        sheet on a mesh that is a JAX tracer is still refused, because its
        corners cannot be read off a traced node line.

        The declaration is mesh-independent by construction, so the same
        call gives the same footprint on the uniform and non-uniform lanes,
        traced or not. A range spanning a single node line is refused: one
        node line carries no E edge, so the metal would carry no current.

        Only the lossless PEC sheet is supported (``sigma_bulk >= 1e6``,
        ``surface_impedance_f0`` unset); ``thickness`` and ``eps_r`` are
        accepted for signature compatibility with
        :meth:`add_thin_conductor` and are not read, exactly as they are
        not read there for a metal.
        """
        from rfx.materials.thin_conductor import PinnedSheet
        self._pinned_sheets.append(PinnedSheet(
            normal_axis=normal_axis, plane_index=plane_index,
            i_range=tuple(i_range), j_range=tuple(j_range),
            sigma_bulk=sigma_bulk, thickness=thickness, eps_r=eps_r,
            surface_impedance_f0=surface_impedance_f0, name=name))
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
        radius: float | None = None,
        excite: bool = True,
        direction: str | None = None,
        reference_plane_cells: int | None = None,
        terminates=None,
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
        radius : float or None
            Opt-in thin-probe radius in metres (requires ``extent``).
            ``None`` preserves the legacy filament, whose effective radius
            is approximately 0.20 times the transverse cell size. The
            local self-field model requires a square, locally uniform
            transverse mesh, uniform spacing along the pin, and radius
            <= 0.2 times the transverse spacing. Supported
            on single-device, nondispersive 3-D second-order Yee run/forward;
            unsupported solvers and material updates raise before stepping.
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
        reference_plane_cells : int or None
            Opt-in reference-plane port waves for the wire-port S-matrix
            OFF-diagonal extraction (issue #313). When set to an integer
            N >= 1, ``run(compute_s_params=True)`` registers TWO line
            V/I reference planes for this port at N and 2N cells
            outboard (into the DUT along the line axis, i.e. opposite
            ``direction``): gap-voltage line integrals at integer Yee
            planes plus dual adjacent Ampere loops around the PEC signal
            trace, accumulated in the production scan. Off-diagonal
            ``S[i, j]`` entries whose ports BOTH opt in are then computed
            from the plane waves — forward/backward split with the
            MEASURED two-plane line impedance
            ``Zc^2 = (V1^2 - V2^2)/(I1^2 - I2^2)`` (never the nominal
            port impedance) and phase-only de-embedding to the port plane
            with the MEASURED per-bin beta from the same two planes.
            The Phase-0 closed-box flux referee showed the port-cell wave
            pair does not conserve power (the port plane is near-field
            dominated) while the plane waves close the power budget at
            all bins; the plane path removes the drive-side |S21|
            deflation kappa(f) = 1.49-1.86 of issue #313.
            ``None`` (default) keeps the shipped port-cell behaviour,
            byte-identical. The DIAGONAL ``S_jj`` always stays on the
            byte-frozen legacy path either way, so ``forward()`` /
            1-port S11 results are unaffected. Requires a wire port
            (``extent=``) with an explicit ``direction`` transverse to
            the port component, a PEC signal trace uniform across both
            planes, and the uniform-mesh ``run()`` lane: the
            non-uniform and subgridded lanes raise
            ``NotImplementedError`` with the opt-in set, and the
            distributed lane does not support ``compute_s_params`` at
            all (it warns that the kwarg is unsupported and ignores
            it).

            Choosing N: place BOTH planes (N and 2N cells) roughly >= 10
            cells from every port so the two-plane Zc/beta measurement
            sits outside the port near-fields — the Phase-0
            pre-registration rule. Measured on the canonical 16 mm thru
            (dx = 0.5 mm, gap-trimmed V, 2026-07-10 battery): N=3 planes
            read beta/(w/c) = 1.16-1.20 and Zc = 52-53 ohm with Im/Re up
            to 8.2% (near-field contaminated; |S21| closed-box-referee
            residual -3.1% at 7 GHz, row energy |S11|^2+|S21|^2 up to
            1.019), while N=10 planes read the clean mid-line
            beta/(w/c) = 1.046-1.059 and Zc = 47.9-48.6 ohm (Im/Re <=
            1.2%) and closed the closed-box referee within 0.9% at all
            bins.
        """
        if self._tfsf is not None:
            raise ValueError(
                "Lumped ports are not supported together with the TFSF plane-wave source"
            )
        if component not in ("ex", "ey", "ez"):
            raise ValueError(f"component must be ex/ey/ez, got {component!r}")
        from rfx.sources.wire_radius import validate_radius
        validate_radius(radius)
        if radius is not None and extent is None:
            raise ValueError("radius requires a wire port: provide extent=...")
        if impedance <= 0:
            raise ValueError(f"impedance must be positive, got {impedance}")
        if direction is not None and direction not in ("+x", "-x", "+y", "-y"):
            raise ValueError(
                f"direction must be one of '+x','-x','+y','-y' (or None), got {direction!r}"
            )
        if reference_plane_cells is not None:
            if extent is None:
                raise NotImplementedError(
                    "reference_plane_cells is only supported on wire ports "
                    "(extent=...): the plane V/I method needs a gap-voltage "
                    "line integral and a PEC signal trace. Single-cell "
                    "lumped ports have no Phase-0 evidence (issue #313)."
                )
            if int(reference_plane_cells) != reference_plane_cells \
                    or int(reference_plane_cells) < 1:
                raise ValueError(
                    "reference_plane_cells must be an integer >= 1, got "
                    f"{reference_plane_cells!r}"
                )
            if direction is None:
                raise ValueError(
                    "reference_plane_cells requires an explicit direction= "
                    "('+x'/'-x'/'+y'/'-y') — the reference planes go "
                    "outboard (into the DUT), opposite the port's outward "
                    "normal, and auto-detection is not accepted for a "
                    "measurement-plane choice."
                )
            if direction[1] == component[1]:
                raise ValueError(
                    f"reference_plane_cells: port component {component!r} "
                    f"lies along the line axis {direction!r} — the gap-V "
                    "line integral must be transverse to the line."
                )
            reference_plane_cells = int(reference_plane_cells)

        if waveform is None and excite:
            waveform = GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)
        if waveform is not None and not excite:
            import warnings as _w
            _w.warn(
                "waveform is ignored when excite=False (passive matched load). "
                "Remove the waveform argument to suppress this warning.",
                stacklevel=2,
            )

        from rfx.geometry.port_termination import resolve_terminates
        terminated = resolve_terminates(
            self, terminates, port=f"add_port at {position}")

        self._ports.append(_PortEntry(
            position=position, component=component,
            impedance=impedance, waveform=waveform,
            extent=extent, excite=excite, direction=direction,
            radius=radius,
            reference_plane_cells=reference_plane_cells,
            terminates=terminated,
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
        terminates=None,
    ) -> "Simulation":
        """Add a microstrip-line (MSL) port spanning the full trace cross-section.

        Unlike :meth:`add_port` with ``extent=...`` (a one-cell-transverse
        wire port), this port covers the full ``width × height``
        cross-section under the trace and uses 3-probe numerical
        de-embedding to extract β and Z0 empirically downstream of the
        feed plane.

        Parameters
        ----------
        position : (x, y, z_lo)
            Physical feed-plane point. The coordinate on the PROPAGATION
            axis (named by ``direction``) is the feed plane; the
            coordinate on the other in-board axis is the trace centre;
            ``z`` is the substrate bottom. So a ``"+x"`` port reads it as
            ``(x_feed, y_centre, z_lo)`` and a ``"+y"`` port as
            ``(x_centre, y_feed, z_lo)``.
        width : float
            Trace width in metres, spanning the in-board axis that is NOT
            the propagation axis (y for an x-directed port, x for a
            y-directed one).
        height : float
            Substrate thickness in metres — always the z extent, because
            the substrate normal is fixed to z (see ``direction``).
        direction : str
            One of ``"+x"``, ``"-x"``, ``"+y"``, ``"-y"`` — the direction
            the launched wave propagates. The board lies in the xy-plane
            and the substrate normal is always z, so the feed may run
            along either in-board axis (issue #661: a CAD-imported board
            does not get to choose its orientation).

            ``"+z"`` / ``"-z"`` are REJECTED with a clear error rather
            than supported: a z-propagating microstrip needs its
            substrate normal along x or y, which ``position`` + scalar
            ``height`` cannot express, and which the Laplace mode solve,
            the ``"ez"`` source component, the modal voltage
            ``V = Σ Ez·dz`` and the trace-conductor PEC scan all assume.
            Accepting it would return a z-normal answer for a board that
            is not oriented that way.

            ``mode="eigenmode"`` remains ``"+x"``/``"-x"`` only.
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
            Automatic values are estimated from the full declaration when
            S, run, forward, preflight or export resolves the port. Counts are recomputed
            in the cells under the port, including graded runways, uniform
            ``dy_profile`` different from ``dx``, and auto meshes. A ladder
            crossing a grading ramp retains counts based on scalar ``dx``.
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
            cleared). These two terms are only the UPSTREAM (lower) edge:
            at ``compute_msl_s_matrix`` time the auto default additionally
            solves the DOWNSTREAM constraint (issue #469) — the deepest
            probe must stay ≥ λ_g/4 (at ``freq_max``) clear of the nearest
            reflector — picking the midpoint of the compliant interval
            when a reflector bounds it, keeping the upstream edge when
            none does, and warning loudly when the interval is empty (the
            feed line is too short for a clean N-probe measurement).
            Pass an explicit value to override (explicit values are never
            adjusted); ``< 5*h_sub/dx`` triggers a preflight near-field
            warning.
        n_probe_spacing : int, optional
            Distance (cells) between consecutive probe planes. Automatic
            counts use the same driver-time runway recount as the offset.
            When ``None``,
            bound so the total N-probe array span stays ~``lam_min_eff/8``
            independent of ``n_probes``:
            ``round(lam_min_eff / 8 / (n_probes - 1) / dx)``. For
            ``n_probes=3`` this is exactly ``lam_min_eff/16`` (issue #80
            Fix B). Shrinking the adjacent spacing as ``n_probes`` grows
            keeps the probe array from running off short lines.

            An AUTO spacing is additionally WIDENED at driver time
            (issue #681): ``compute_msl_s_matrix`` /
            ``compute_mixed_s_matrix`` re-solve it toward a total span of
            ``(n_probes−1)·λ_g(f_max)/4`` (λ_g from the HJ ε_eff), capped
            by the downstream-reflector interval and the absorbing
            boundary — a ~0.1·λ_g span leaves the β fit noise-fragile
            (measured: median β error 5.5% at 0.10 λ_g vs 0.8% at
            0.30 λ_g under 1% probe noise). An explicit value is never
            touched.
        n_probes : int
            Number of equally-spaced voltage probe planes registered for
            the N-probe least-squares wave-decomposition extractor (issue #80 Fix C). Probe ``n`` sits at ``offset + n*spacing`` cells
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
            estimate is resolved from the full dielectric declaration when
            S, run, forward, preflight or export resolves the port (a shape
            under the port whose bounding box contains
            the substrate mid-point and whose material has ``eps_r > 1``);
            if no such dielectric is found it falls back to 1.0, which is
            conservative (largest ``lam_min_eff`` → largest offset).
        """
        from rfx.sources.msl_port import msl_axis_roles as _msl_axis_roles
        # Raises with the substrate-normal explanation for "+z"/"-z" and a
        # plain domain error otherwise (issue #661).
        _msl_axis_roles(direction)
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")
        if impedance <= 0:
            raise ValueError(f"impedance must be positive, got {impedance}")
        if mode not in ("eigenmode", "laplace", "uniform"):
            raise ValueError(f"mode must be 'eigenmode', 'laplace', or 'uniform', got {mode!r}")
        if mode == "eigenmode" and direction not in ("+x", "-x"):
            raise NotImplementedError(
                f"add_msl_port(mode='eigenmode', direction={direction!r}) is "
                "not supported: the Schelkunoff J+M launch hard-codes the "
                "x-axis TFSF correction pair and rides on the FDFD eigenmode "
                "solver, which is a documented dead-end kept under a strict "
                "xfail. Use mode='laplace' (the default) for a y-directed "
                "port (issue #661)."
            )
        from rfx.preflight._msl_probe_rules import msl_auto_probe_declaration
        (_, _near_field_m, span_total, _eps_eff_hj,
         _auto_offset, _auto_spacing) = msl_auto_probe_declaration(
            self, position=position, width=width, height=height,
            n_probes=n_probes, eps_r_sub=eps_r_sub)
        _offset_is_auto = n_probe_offset is None
        if _offset_is_auto:
            n_probe_offset = _auto_offset
        _spacing_is_auto = n_probe_spacing is None
        if _spacing_is_auto:
            n_probe_spacing = _auto_spacing
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

        if _offset_is_auto:
            # Runtime bookkeeping only (NOT an entry field — _MSLPortEntry's
            # field set is pinned by the interop design-IR contract): the
            # #469 interval solve in compute_msl_s_matrix re-derives the
            # effective offset from THIS stored lower edge every call, so
            # repeated calls do not drift.
            self._msl_auto_offset_min[name] = int(n_probe_offset)
        if _spacing_is_auto:
            # Same bookkeeping pattern for the auto spacing (issue #681):
            # store the registration-time HJ eps_eff so the driver-time
            # span solve can size λ_g(f_max)/4 per probe step from the
            # SAME permittivity estimate every call (idempotent, and the
            # design-IR dump resolves through the same function).
            self._msl_auto_probe_spacing[name] = float(_eps_eff_hj)
        # The two lengths the automatic defaults were counted from, for every
        # port: the resolver recounts an automatic offset or spacing in the
        # runway cell on a graded axis, and preflight asks what leaving an
        # explicit offset None would give on this port's runway (#810).
        self._msl_auto_probe_lengths[name] = (
            float(_near_field_m), float(span_total))

        from rfx.geometry.port_termination import resolve_terminates
        terminated = (None if terminates is None else resolve_terminates(
            self, terminates, port=f"add_msl_port at {position}"))

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
            terminates=terminated,
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

    from rfx.sources.tfsf_discrete_plane_wave import add_tfsf_source, tfsf_incidence

    def tfsf_box_indices(self) -> dict[str, tuple[int, int]]:
        """Realized inclusive total-field node bounds on the padded grid.

        The normal slab returns x only; ``closed_box=True`` returns x/y/z.
        NTFF cell-centre interpolation must clear these bounds, including
        the adjacent Yee samples (the NTFF placement check enforces this).
        """
        return _discrete_plane_wave.tfsf_box_indices(self)

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
        - `reference_plane` / `probe_plane`: explicit reported coordinates;
          an unset reference uses the source plane, an unset probe the measured probe
        - `calibration_preset=None` or `'source_to_probe'`: report `S11` at the
          snapped source plane and `S21` from source to the measured probe plane
        - `calibration_preset='measured'`: use the snapped stored
          reference/probe planes directly

        Sampling still occurs on the nearest snapped grid planes, and the
        result metadata reports those actual measurement planes explicitly.

        ``waveform`` selects the source pulse shape. Default
        ``"modulated_gaussian"`` is the Meep-style bandpass pulse — no
        sub-cutoff DC content, so the in-band TFSF filter collapses to
        identity, reducing directional leakage in the H+E injection pair.
        ``"differentiated_gaussian"`` is refused on waveguide ports: its
        zero-phase incident filter wraps the precursor to the record end
        and re-injects there (#1942). Use ``"modulated_gaussian"``
        (the default).
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
        if waveform != "modulated_gaussian":
            from rfx.model import waveguide_waveform_error
            raise ValueError(
                waveguide_waveform_error(waveform)
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
        axis_domain = self._domain[axis_idx]
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

    from rfx.boundaries.serialization import legacy_spec as _build_spec_from_legacy

    def __getattr__(self, name):
        if name == "set_periodic_axes":
            raise AttributeError(
                "Simulation.set_periodic_axes was removed; use BoundarySpec "
                "per-face periodic via boundary=BoundarySpec(x='periodic', ...)."
            )
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def boundary_model(self, *, rcs: bool = False):
        """Return the B1 face declaration with the current feature requirements.

        No runner consumes this descriptor in B1. ``rcs=True`` collects the
        RCS requirement for a caller preparing that operation.
        """
        return refresh_boundary_model(self, rcs=rcs)

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
        unit cell response. Scalar CPML/UPML (including the default) absorbs
        along the port axis and makes the transverse axes periodic.

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
        # Only the specular (0,0) TE mode is implemented in extract_floquet_modes; n_modes>1 and
        # TM were silently accepted and returned wrong results (higher-order lobes dropped; TM read
        # the TE field pair + impedance). Fail loud until implemented (RF-audit 2026-07-23).
        if n_modes > 1:
            raise NotImplementedError(
                f"add_floquet_port(n_modes={n_modes}): only the specular (0,0) Floquet mode is "
                f"extracted; higher-order grating lobes are not implemented. Use n_modes=1."
            )
        if polarization == "tm":
            raise NotImplementedError(
                "add_floquet_port(polarization='tm'): TM Floquet S-parameter extraction is not "
                "implemented — the extractor is hardwired to the TE (Ex,Hy) pair and TE wave "
                "impedance, so a TM drive yields wrong S-parameters. Use polarization='te'. "
                "(For a TM plane-wave field study without S-parameters, use add_tfsf_source, "
                "which supports oblique TM incidence.)"
            )
        if self._tfsf is not None:
            raise ValueError(
                "Floquet ports are not supported together with TFSF sources"
            )

        # Reject an explicit incompatible declaration at registration. Auto
        # mesh depends on the completed model and is checked by preflight.
        if self._declared_mesh["_dz_profile"] is not None:
            raise ValueError(
                "Floquet ports do not support non-uniform z mesh (dz_profile). "
                "Set dx explicitly to prevent auto-mesh from creating NU grid."
            )

        transverse = require_floquet_lateral_faces(self, axis)
        if not self._periodic_axes:
            self._periodic_axes = transverse

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

    def _validate_declared_plane_coordinate(self, axis: str, coordinate: float) -> None:
        """Validate a builder coordinate without resolving unfinished geometry."""
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        extent = self._declared_mesh["_domain"][axis_idx]
        if coordinate < 0 or coordinate > extent:
            raise ValueError(
                f"coordinate {coordinate} m is outside the {axis}-domain [0, {extent}]"
            )

    def add_dft_plane_probe(
        self,
        *,
        axis: str,
        coordinate: float,
        component: str = "ez",
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 50,
        name: str | None = None,
        region: tuple[int, int, int, int] | None = None,
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
        region : tuple or None
            Half-open transverse array-index crop (lo1, hi1, lo2, hi2).
            A 1 by 1 crop is a point DFT at that Yee component.
        name : str or None
            Optional result key.
        """
        from rfx.measurement.setup import register_plane
        return register_plane(self, axis, coordinate, component, freqs, n_freqs, name, region)

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
            Positive finite extents in the two tangential directions. A finite
            window snaps to cell edges, clamps to the physical interior with
            a warning on any clamp, and excludes CPML/bounding-node slots.
            ``None`` means the full allocated plane (legacy behaviour).
        center : (float, float) or None
            Physical centre of the flux region in the two tangential
            directions. ``None`` defaults to the declared domain midpoint
            on uniform grids and the realized interior midpoint on graded grids.
            For example, for an x-normal monitor the two tangential
            axes are (y, z). Exactly two finite values are required.
            Preflight reports finite requested/realized bounds in metres
            through ``report.flux_regions``.
        name : str or None
            Result key. Default: ``flux_{axis}_{idx}``.
        """
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        self._validate_declared_plane_coordinate(axis, coordinate)
        from rfx.probes.flux_region import validate_flux_region_inputs
        validate_flux_region_inputs(size, center)
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
        from rfx.measurement import default_frequency_grid
        freqs = default_frequency_grid(self._freq_max, n_freqs) if freqs is None else freqs
        self._ntff = (corner_lo, corner_hi, freqs)
        return self

    def add_current_moment_monitor(
        self,
        corner_lo: tuple[float, float, float],
        corner_hi: tuple[float, float, float],
        block_size: float,
        freqs,
        order: int = 2,
        margin_cells=0,
        off_cells: int = 0,
    ) -> "Simulation":
        """Accumulate the structure's own current as block moments, in-loop.

        The radiation of a structure is what the current in it radiates. On
        the Yee lattice that current is an identity the solver already
        enforces — ``J = curl_h H - eps0 dE/dt`` at every electric-field edge
        — so it can be read off the fields the step is holding, with nothing
        modelled and nothing fitted. This monitor sums it over the slab
        between ``corner_lo`` and ``corner_hi`` into a few numbers per
        in-plane block (the total current moment and its first two spatial
        moments about the block's own centre) and DFTs those, instead of
        accumulating tangential E and H over a Huygens surface.

        Parameters
        ----------
        corner_lo, corner_hi : (x, y, z) in metres
            Opposite corners of the slab, in the same frame as
            :meth:`add_ntff_box`. The z range picks the node planes; the
            whole thickness goes into one block.
        block_size : float
            In-plane block side in metres, rounded to a whole number of
            cells. The realized side is what the monitor reports.
        freqs : array
            Frequencies (Hz).
        order : int
            2 (the only accepted value) keeps P, Q and T — 30 numbers per
            block. Lower orders are refused; the low-level
            ``rfx.current_moments.build_current_moment_monitor`` keeps them.
        margin_cells : int or (mx, my, mz)
            Extra cells around the declared corners.
        off_cells : int
            Shift of the in-plane partition origin, in cells. 0 and half a
            block are the two the block rule was measured with.

        Notes
        -----
        The slab must lie in the interior: inside the absorber the E update
        is not Ampere's law, so a current read there is the absorber's
        fiction. Periodic/Bloch axes, TFSF sources, ``stencil_order=4``, a
        graded or dx != dy in-plane mesh, a traced mesh profile, and every
        lane whose scan body does not accumulate the monitor are refused
        rather than approximated. So is a model the pattern would silently
        leave out: magnetic material (``mu_r != 1``), a PEC or PMC domain
        face, a waveguide, coaxial or Floquet port, a microstrip port with
        ``mode="eigenmode"``, and any dielectric, conductor, dispersive cell,
        port, lumped element or source that is not inside the slab (its
        outermost edge layer counts as outside). So are the Kottke PEC paths
        and blocks with k·|delta| > 0.5 at the top frequency (|delta|: edge to
        block centre, slab thickness included): use smaller blocks or NTFF.

        The named arguments here are the whole public surface. The low-level
        builder additionally takes deliberately wrong metrics, centres, curl
        signs and time stamps so the mutation harness can measure what the
        declared checks catch; those never reach a user's declaration.
        """
        if not float(block_size) > 0.0:
            raise ValueError(
                f"add_current_moment_monitor(block_size={block_size}): the "
                "block side must be positive.")
        if int(order) != 2:
            raise ValueError(
                f"add_current_moment_monitor(order={order}): only order=2 is "
                "supported. Each block is expanded about the mean position of "
                "its edges, which sits half a Yee cell off the x-directed "
                "edges' own centroid; the second moment T absorbs that offset, "
                "while order 1 leaves a floor of about 0.5-0.9 % in the "
                "pattern (measured on the tutorial patch) and order 0 drops "
                "the first moment Q altogether.")
        self._current_moments = (corner_lo, corner_hi, float(block_size),
                                 freqs, int(order),
                                 {"margin_cells": margin_cells,
                                  "off_cells": int(off_cells)})
        return self

    # ---- build helpers ----

    def freeze_mesh(self):
        """Finalize mesh spacing/profiles and extent, returning the selected grid.

        Call after adding mesh-driving materials/geometry and before placing
        lattice-aligned conductors. Later geometry cannot refine this mesh;
        unresolved features still raise under the normal conductor contract.
        Repeated calls preserve the same mesh. Build a new Simulation to remesh.

        Grid previews (including preflight) alone do not finalize the mesh.
        Register boundary conditions and ports before retaining grid indices:
        port registration may change padding, though physical node positions
        in the domain stay fixed. Caller declarations remain available unchanged.
        """
        return self._freeze_mesh()

    def mesh_intelligence_report(
        self,
        *,
        n_steps: int | None = None,
        checkpoint_every: int | None = None,
        checkpoint_segments: int | None = None,
        available_memory_gb: float | None = None,
        n_warmup: int = 0,
        check_ntff: bool = True,
        check_resolution: bool = True,
    ) -> MeshIntelligenceReport:
        """Return a consolidated mesh-quality and memory-planning report."""
        return mesh_intelligence_report(
            self, n_steps=n_steps, checkpoint_every=checkpoint_every,
            checkpoint_segments=checkpoint_segments,
            available_memory_gb=available_memory_gb, n_warmup=n_warmup,
            check_ntff=check_ntff, check_resolution=check_resolution,
        )
    def _mesh_planner_state(self) -> dict[str, object]:
        """Return a narrow internal snapshot consumed by ``rfx.mesh_planner``.

        This keeps the planner from scattering direct ``Simulation`` private
        attribute reads while avoiding a larger public accessor surface.
        """
        grid = self._build_realized_grid()
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
        grid = self._build_realized_grid()
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

    auto = classmethod(simulation_auto)


# ---------------------------------------------------------------------------
# Arc-audit follow-up (verification round): every method Simulation
# inherits from its five mixins keeps that mixin's name in its
# __qualname__ (Python sets __qualname__ from the class body where a
# function is DEFINED, not where it ends up bound via inheritance), which
# leaks into user-facing TypeError messages for an unrecognised keyword
# argument -- e.g. `TypeError: _ExecuteMixin.forward() got an unexpected
# keyword argument 'design_mask'` instead of naming the actual public
# surface, `Simulation.forward()`. This was fixed for forward()
# specifically via an explicit **_removed_kwargs shim
# (_reject_removed_forward_kwargs in rfx/api/_execute.py), but the
# verifier found it is not one-off: ten public Simulation methods leak
# the same way (e.g. `sim.run(n_stepss=2)` still says `_ExecuteMixin.
# run() got an unexpected keyword argument`). Rebind __qualname__ on
# every inherited method here, once, at class composition time, so all
# of them read `Simulation.<method>` regardless of which mixin module
# happens to define them. Structural only: does not change behaviour,
# identity, or MRO -- only the __qualname__ string attribute Python's
# own error formatting (and tracebacks, repr, etc.) reads from. Each of
# these mixin classes is used ONLY by Simulation (verified: no other
# `class ...(_ExecuteMixin` etc. anywhere in rfx/), so mutating the
# function objects in place cannot affect any other class.
# ---------------------------------------------------------------------------
for _mixin in (
    _PreflightMixin, _SparamMixin, _CompileMixin, _ExecuteMixin, _ArtifactsMixin,
    _AdMemoryMixin,
):
    for _name, _member in vars(_mixin).items():
        if (
            inspect.isfunction(_member)
            and _member.__qualname__ == f"{_mixin.__name__}.{_name}"
        ):
            _member.__qualname__ = f"Simulation.{_name}"
del _mixin, _name, _member


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
    "ADMemoryActionHint",
    "ADMemoryExplainabilityReport",
    "ADMemoryPreflightReport",
    "ADCompiledMemoryCertificate",
    "MeshIntelligenceReport",
    "Result",
    "ForwardResult",
    "MaterialSpec",
    "WaveguideSParamResult",
    "WaveguideSMatrixResult",
    "CoaxialLineReflectionResult",
    "CoaxialTwoPortResult",
    "MSLProbeClearance",
    "MSLSMatrixResult",
    "MixedSMatrixResult",
    "GradientRecordLengthWitness",
    "gradient_record_length_witness",
]
