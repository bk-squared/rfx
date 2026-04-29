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

from dataclasses import dataclass
import math
import jax
from numbers import Integral
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid, C0
from rfx.core.yee import MaterialArrays, EPS_0
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Shape
from rfx.nonuniform import NonUniformGrid
from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import CoaxialPort
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import LorentzPole, init_lorentz
from rfx.lumped import LumpedRLCSpec
from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor
from rfx.sources.waveguide_port import (
    WaveguidePort,
    extract_waveguide_s_matrix,
    extract_waveguide_s_params_normalized,
    init_waveguide_port,
    init_multimode_waveguide_port,
    extract_multimode_s_matrix,
    waveguide_plane_positions,
)
from rfx.simulation import (
    SnapshotSpec,
)
from rfx.adi import ADIState2D, run_adi_2d


# ---------------------------------------------------------------------------
# Named material library — common RF/microwave materials
# ---------------------------------------------------------------------------

MATERIAL_LIBRARY: dict[str, dict] = {
    "vacuum":      {"eps_r": 1.0, "sigma": 0.0},
    "air":         {"eps_r": 1.0006, "sigma": 0.0},
    "fr4":         {"eps_r": 4.4, "sigma": 0.025},
    "rogers4003c": {"eps_r": 3.55, "sigma": 0.0027 * 2 * np.pi * 5e9 * 3.55 * EPS_0},
    "alumina":     {"eps_r": 9.8, "sigma": 0.0},
    "silicon":     {"eps_r": 11.9, "sigma": 0.01},
    "ptfe":        {"eps_r": 2.1, "sigma": 0.0},
    "copper":      {"eps_r": 1.0, "sigma": 5.8e7},
    "aluminum":    {"eps_r": 1.0, "sigma": 3.5e7},
    "pec":         {"eps_r": 1.0, "sigma": 1e10},
    "water_20c":   {
        "eps_r": 4.9, "sigma": 0.0,
        "debye_poles": [DebyePole(delta_eps=74.1, tau=8.3e-12)],
    },
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class AD_MemoryEstimate(NamedTuple):
    """Reverse-mode AD memory estimate (issue #30 CHECK 4 / #39).

    All sizes are in gigabytes. ``warning`` is populated when the
    selected estimate exceeds 85% of ``available_gb``.

    ``ad_checkpointed_gb`` is the legacy ``jax.checkpoint(step_fn)``
    estimate; for FDTD on the non-uniform path this is *not* a
    realistic memory number because the scan carry itself is not
    rematerialised (see issue #31 VESSL 369367233490). Use
    ``ad_segmented_gb`` when ``checkpoint_every`` is set — that matches
    the observed memory on RTX 4090 within ~15%.
    """
    forward_gb: float
    ad_checkpointed_gb: float
    ad_full_gb: float
    ntff_dft_gb: float
    available_gb: float | None
    warning: str | None
    ad_segmented_gb: float | None = None
    checkpoint_every: int | None = None


class Result(NamedTuple):
    """Structured simulation result.

    Attributes
    ----------
    state : FDTDState
        Final field state (useful for visualization).
    time_series : (n_steps, n_probes) float array
        Probe recordings over time.
    s_params : (n_ports, n_ports, n_freqs) complex or None
        S-parameter matrix (computed only when ports are present and
        ``compute_s_params=True``).
    freqs : (n_freqs,) float or None
        Frequency array for S-parameters.
    ntff_data : NTFFData or None
        Raw NTFF DFT data (use ``compute_far_field`` for radiation pattern).
    ntff_box : NTFFBox or None
        NTFF box specification (needed for ``compute_far_field``).
    dft_planes : dict[str, DFTPlaneProbe] or None
        Frequency-domain plane probes keyed by name.
    waveguide_ports : dict[str, WaveguidePortConfig] or None
        Final accumulated waveguide-port configs keyed by name.
    waveguide_sparams : dict[str, WaveguideSParamResult] or None
        High-level calibrated waveguide S-parameters keyed by port name.
    snapshots : dict[str, ndarray] or None
        Field snapshots keyed by component name.
    grid : Grid or None
        Grid metadata for post-processing helpers and advanced objectives.
    """
    state: object
    time_series: jnp.ndarray
    s_params: np.ndarray | None
    freqs: np.ndarray | None
    ntff_data: object = None
    ntff_box: object = None
    dft_planes: dict | None = None
    flux_monitors: dict | None = None
    waveguide_ports: dict | None = None
    waveguide_sparams: dict | None = None
    snapshots: dict | None = None
    grid: object = None
    dt: float | None = None
    freq_range: tuple | None = None

    def find_resonances(self, freq_range=None, probe_idx=0,
                         source_decay_time=None, bandpass=None):
        """Extract resonant modes from probe time series via Harminv.

        Parameters
        ----------
        freq_range : (f_min, f_max) in Hz, or None to use stored range
        probe_idx : which probe to analyze
        source_decay_time : float or None
            Time (s) after which source has decayed. If None, auto-
            computed as 2×(3/π/f_center/bandwidth) — skips the Gaussian
            excitation region for clean ring-down analysis.
        bandpass : bool or None
            Apply FFT bandpass before Harminv. Default: auto (True for
            CPML results where DC/surface-wave artifacts exist, False
            for PEC cavities where signal is clean).

        Returns
        -------
        list of HarminvMode
        """
        from rfx.harminv import harminv, harminv_from_probe
        ts = np.asarray(self.time_series)
        if ts.ndim == 2:
            ts = ts[:, probe_idx]
        ts = ts.ravel()
        if self.dt is None:
            raise ValueError("dt not available in Result — run with store_dt=True")
        fr = freq_range
        if fr is None:
            fr = self.freq_range
        if fr is None:
            raise ValueError("freq_range not specified")
        stored_boundary = 'cpml'
        if self.freq_range is not None and len(self.freq_range) > 2:
            stored_boundary = self.freq_range[2]
        if len(fr) > 2:
            stored_boundary = fr[2]
            fr = (fr[0], fr[1])

        if bandpass is None:
            bandpass = stored_boundary == 'cpml'

        if source_decay_time is None:
            f_center = (fr[0] + fr[1]) / 2
            bw = 0.8
            tau = 1.0 / (f_center * bw * np.pi)
            source_decay_time = 2.0 * 3.0 * tau

        start = int(np.ceil(source_decay_time / self.dt))
        start = min(start, max(len(ts) - 20, 0))
        w = ts[start:] - np.mean(ts[start:])

        max_direct = 10000
        if len(w) > max_direct:
            step = len(w) // max_direct
            w_sub = w[::step][:max_direct]
            dt_h = self.dt * step
        else:
            w_sub = w
            dt_h = self.dt

        modes = harminv(w_sub, dt_h, fr[0], fr[1])

        if not modes and bandpass:
            modes = harminv_from_probe(ts, self.dt, fr,
                                        source_decay_time=source_decay_time)

        return modes


class ForwardResult(NamedTuple):
    """Minimal differentiable simulation result.

    Carries only the observables needed by gradient-based objectives,
    avoiding the broader stateful surface of :class:`Result`.

    ``lumped_port_sparams`` exposes the raw per-port (V_dft, I_dft) tuples
    accumulated inside the JIT scan body when ``forward(port_s11_freqs=...)``
    is used.  Single-port objectives can keep using ``s_params`` (which is
    populated with per-port |S11| via :func:`extract_lumped_s11`).  Multi-
    port AD objectives (e.g. 2-port |S21| topology optimisation) read raw
    V/I from this field and compose their own wave decomposition, since
    ``extract_lumped_s11`` collapses each port to its self-reflection only.
    """
    time_series: jnp.ndarray
    ntff_data: object = None
    ntff_box: object = None
    grid: object = None
    s_params: object = None
    freqs: object = None
    lumped_port_sparams: object = None


# ---------------------------------------------------------------------------
# Material specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialSpec:
    """Material definition with optional Debye/Lorentz dispersion."""
    eps_r: float = 1.0
    sigma: float = 0.0
    mu_r: float = 1.0
    debye_poles: list[DebyePole] | None = None
    lorentz_poles: list[LorentzPole] | None = None
    chi3: float = 0.0  # Third-order Kerr susceptibility (m^2/V^2)


# ---------------------------------------------------------------------------
# Internal bookkeeping types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _GeometryEntry:
    shape: Shape
    material_name: str


@dataclass(frozen=True)
class _PortEntry:
    position: tuple[float, float, float]
    component: str
    impedance: float
    waveform: GaussianPulse
    extent: float | None = None
    # Port excitation mode:
    #   excite=True  → resistive termination + source (legacy behaviour)
    #   excite=False → resistive termination only (matched passive load)
    # Passive ports are essential for multi-port S-parameter extraction
    # where only one port drives the DUT at a time.
    excite: bool = True
    # Port outward-normal direction (port face → external world). Used by
    # the S-matrix extraction to orient the V/I wave decomposition: at a
    # port looking in +x, the incoming (into-DUT) wave is the +x-moving
    # wave, so `a = (V + Z·I)/2`. At a -x port, signs flip. Valid
    # values: "+x", "-x", "+y", "-y". None → auto-detect from position
    # at sim-build time.
    direction: str | None = None


@dataclass(frozen=True)
class _ProbeEntry:
    position: tuple[float, float, float]
    component: str


@dataclass(frozen=True)
class _TFSFEntry:
    f0: float | None
    bandwidth: float
    amplitude: float
    margin: int
    polarization: str
    direction: str
    angle_deg: float
    waveform: str = "differentiated_gaussian"


@dataclass(frozen=True)
class _DFTPlaneEntry:
    name: str
    axis: str
    coordinate: float
    component: str
    freqs: jnp.ndarray | None
    n_freqs: int


@dataclass(frozen=True)
class _FluxMonitorEntry:
    name: str
    axis: str
    coordinate: float
    freqs: jnp.ndarray | None
    n_freqs: int
    size: tuple[float, float] | None = None    # tangential extent (dim1, dim2)
    center: tuple[float, float] | None = None  # tangential center (dim1, dim2)
    dft_window: str = "rect"                    # streaming DFT window
    dft_window_alpha: float = 0.25              # Tukey shape parameter


@dataclass(frozen=True)
class _WaveguidePortEntry:
    name: str
    x_position: float
    y_range: tuple[float, float] | None
    z_range: tuple[float, float] | None
    x_range: tuple[float, float] | None
    mode: tuple[int, int]
    mode_type: str
    direction: str
    freqs: jnp.ndarray | None
    n_freqs: int
    f0: float | None
    bandwidth: float
    amplitude: float
    probe_offset: int
    ref_offset: int
    calibration_preset: str | None
    reference_plane: float | None
    probe_plane: float | None
    n_modes: int = 1
    waveform: str = "differentiated_gaussian"
    mode_profile: str = "analytic"


@dataclass(frozen=True)
class _FloquetPortEntry:
    """Internal bookkeeping for a Floquet port."""
    name: str
    position: float
    axis: str
    scan_theta: float
    scan_phi: float
    polarization: str
    n_modes: int
    freqs: jnp.ndarray | None
    n_freqs: int
    f0: float | None
    bandwidth: float
    amplitude: float


class WaveguideSParamResult(NamedTuple):
    """High-level calibrated waveguide S-parameter data."""
    freqs: np.ndarray
    s11: np.ndarray
    s21: np.ndarray
    calibration_preset: str
    source_plane: float
    measured_reference_plane: float
    measured_probe_plane: float
    reference_plane: float
    probe_plane: float


class WaveguideSMatrixResult(NamedTuple):
    """Waveguide scattering result assembled one driven port at a time."""
    s_params: np.ndarray
    freqs: np.ndarray
    port_names: tuple[str, ...]
    port_directions: tuple[str, ...]
    reference_planes: np.ndarray


_DebyeSpec = tuple[list[DebyePole], list[jnp.ndarray]]
_LorentzSpec = tuple[list[LorentzPole], list[jnp.ndarray]]


# ---------------------------------------------------------------------------
# Phase 3 (issue #44 V3 §M6): module-level flag so the distributed=True
# UserWarning fires exactly once per process.  Reset to False on import,
# flipped to True the first time ``Simulation.forward(distributed=True, ...)``
# is invoked.
# ---------------------------------------------------------------------------
_DISTRIBUTED_FIRST_CALL_WARNED: bool = False


# ---------------------------------------------------------------------------
# Simulation builder
# ---------------------------------------------------------------------------

class Simulation:
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
        boundary: "str | BoundarySpec | dict" = "cpml",
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
        from rfx.boundaries.spec import BoundarySpec, Boundary, normalize_boundary

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
        self._ntff: tuple | None = None  # (corner_lo, corner_hi, freqs)
        self._tfsf: _TFSFEntry | None = None
        self._dft_planes: list[_DFTPlaneEntry] = []
        self._flux_monitors: list[_FluxMonitorEntry] = []
        self._waveguide_ports: list[_WaveguidePortEntry] = []
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
    ) -> "Simulation":
        """Add a z-axis refinement region for SBP-SAT subgridding.

        The fine grid covers the specified z-range (plus xy_margin around
        geometry) at dx_fine = dx_coarse / ratio.

        Parameters
        ----------
        z_range : (z_lo, z_hi) in metres
            Physical z-range for the fine region.
        ratio : int
            Refinement ratio (fine cells per coarse cell). Default 4.
        xy_margin : float or None
            Extra xy margin around geometry for the fine region.
            Default: 2 * dx_coarse.
        tau : float
            SAT penalty coefficient (default 0.5). Higher values give
            stronger coupling but more dissipation.
        """
        if self._refinement is not None:
            raise ValueError("Only one refinement region is supported")

        # Warn if subgrid overlaps PML region.
        # PML operates on the coarse grid only; the fine grid has no PML.
        # Overlapping causes late-time energy growth (SAT coupling feeds
        # energy into the PML boundary faster than it can absorb).
        if self._boundary in ("cpml", "upml") and self._cpml_layers > 0:
            import warnings
            dx = self._dx or (2.998e8 / self._freq_max / 10)
            pml_thickness = self._cpml_layers * dx
            domain_z = self._domain[2] if len(self._domain) > 2 else 0
            z_lo, z_hi = z_range
            if z_lo < pml_thickness or (domain_z > 0 and z_hi > domain_z - pml_thickness):
                warnings.warn(
                    f"Subgrid z_range=({z_lo*1e3:.1f}, {z_hi*1e3:.1f})mm overlaps "
                    f"PML region (thickness={pml_thickness*1e3:.1f}mm). "
                    f"This causes late-time energy growth. "
                    f"Move z_range inside the PML boundary for stable results.",
                    stacklevel=2,
                )

        self._refinement = {
            "z_range": z_range,
            "ratio": ratio,
            "xy_margin": xy_margin,
            "tau": tau,
        }
        return self

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

        .. deprecated:: 1.7.0
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

    def _waveguide_cpml_axes(self, extra_axes: str = "") -> str:
        axes_in_use = {
            entry.direction[1]
            for entry in self._waveguide_ports
        }
        axes_in_use.update(axis for axis in extra_axes if axis in "xyz")
        return "".join(axis for axis in "xyz" if axis in axes_in_use) or "x"

    def _build_grid(self, *, extra_waveguide_axes: str = "") -> Grid:
        # Remove periodic axes from CPML allocation — CPML on a periodic
        # axis fights the wrap-around and corrupts the physics
        # (issue #68). Default is "xyz"; the waveguide-port path overrides
        # with a port-normal-PEC filter.
        def _filter_periodic(axes: str) -> str:
            if not self._periodic_axes:
                return axes
            return "".join(ax for ax in axes if ax not in self._periodic_axes)

        face_layers = self._resolve_face_layers()

        if self._waveguide_ports or extra_waveguide_axes:
            cpml_axes = _filter_periodic(
                self._waveguide_cpml_axes(extra_waveguide_axes)
            )
            return Grid(
                freq_max=self._freq_max,
                domain=self._domain,
                dx=self._dx,
                cpml_layers=self._cpml_layers,
                cpml_axes=cpml_axes,
                mode=self._mode,
                kappa_max=self._cpml_kappa_max,
                pec_faces=self._pec_faces,
                pmc_faces=self._boundary_spec.pmc_faces(),
                face_layers=face_layers,
            )
        return Grid(
            freq_max=self._freq_max,
            domain=self._domain,
            dx=self._dx,
            cpml_layers=self._cpml_layers,
            cpml_axes=_filter_periodic("xyz"),
            mode=self._mode,
            kappa_max=self._cpml_kappa_max,
            pec_faces=self._pec_faces,
            pmc_faces=self._boundary_spec.pmc_faces(),
            face_layers=face_layers,
        )

    def _resolve_face_layers(self) -> dict:
        """T7 Phase 2 PR2: per-face active CPML layer counts from the
        canonical ``BoundarySpec``. Faces without an explicit
        ``lo_thickness`` / ``hi_thickness`` default to the scalar
        ``cpml_layers`` (the symmetric common case — no padding).
        """
        n_default = self._cpml_layers
        out = {}
        for axis_name, boundary in (("x", self._boundary_spec.x),
                                    ("y", self._boundary_spec.y),
                                    ("z", self._boundary_spec.z)):
            out[f"{axis_name}_lo"] = boundary.resolved_lo_thickness(n_default)
            out[f"{axis_name}_hi"] = boundary.resolved_hi_thickness(n_default)
        return out

    # Threshold above which sigma is treated as PEC (use mask instead).
    _PEC_SIGMA_THRESHOLD = 1e6

    def _assemble_materials(
        self,
        grid: Grid,
    ) -> tuple[MaterialArrays, _DebyeSpec | None, _LorentzSpec | None, jnp.ndarray | None, list, jnp.ndarray | None]:
        """Build material arrays plus per-pole dispersion masks.

        Returns
        -------
        materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, kerr_chi3
            pec_mask is a boolean array (True at PEC cells) or None.
            pec_shapes is a list of Shape objects that are PEC.
            kerr_chi3 is a float32 array of chi3 values or None.
        """
        # Start with vacuum
        eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
        sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
        mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
        chi3_arr = jnp.zeros(grid.shape, dtype=jnp.float32)
        pec_mask = jnp.zeros(grid.shape, dtype=jnp.bool_)
        pec_shapes = []
        has_kerr = False

        # Collect per-pole masks so distinct materials do not inherit
        # each other's dispersion poles.
        debye_masks_by_pole: dict[DebyePole, jnp.ndarray] = {}
        lorentz_masks_by_pole: dict[LorentzPole, jnp.ndarray] = {}

        for entry in self._geometry:
            mat = self._resolve_material(entry.material_name)
            mask = entry.shape.mask(grid)

            if mat.sigma >= self._PEC_SIGMA_THRESHOLD:
                # True PEC: mark in mask, keep eps/sigma at vacuum values
                pec_mask = pec_mask | mask
                pec_shapes.append(entry.shape)
            else:
                eps_r = jnp.where(mask, mat.eps_r, eps_r)
                sigma = jnp.where(mask, mat.sigma, sigma)
                mu_r = jnp.where(mask, mat.mu_r, mu_r)

            if mat.chi3 != 0.0:
                chi3_arr = jnp.where(mask, mat.chi3, chi3_arr)
                has_kerr = True

            if mat.debye_poles:
                for pole in mat.debye_poles:
                    if pole in debye_masks_by_pole:
                        debye_masks_by_pole[pole] = debye_masks_by_pole[pole] | mask
                    else:
                        debye_masks_by_pole[pole] = mask

            if mat.lorentz_poles:
                for pole in mat.lorentz_poles:
                    if pole in lorentz_masks_by_pole:
                        lorentz_masks_by_pole[pole] = lorentz_masks_by_pole[pole] | mask
                    else:
                        lorentz_masks_by_pole[pole] = mask

        # Extend material properties into CPML padding so that guided
        # modes in dielectric waveguides see an impedance-matched absorber
        # (equivalent to UPML).  Each CPML face copies the interior-edge
        # slice outward, as if the geometry continued beyond the domain.
        if self._boundary in ("cpml", "upml") and self._cpml_layers > 0:
            # v1.7.5: per-face allocation (pad_{axis}_lo / _hi). Reflector /
            # periodic faces have pad=0 on that side and the corresponding
            # replicate step is skipped so the interior cells are not
            # overwritten. The replicate depth matches the actual
            # allocation on that face (``pad_*_lo`` or ``pad_*_hi``).
            plx, phx = grid.pad_x_lo, grid.pad_x_hi
            ply, phy = grid.pad_y_lo, grid.pad_y_hi
            plz, phz = grid.pad_z_lo, grid.pad_z_hi
            eps_r_ext = eps_r
            sigma_ext = sigma
            mu_r_ext = mu_r
            if plx > 0:
                eps_r_ext = eps_r_ext.at[:plx,:,:].set(eps_r_ext[plx:plx+1,:,:])
                sigma_ext = sigma_ext.at[:plx,:,:].set(sigma_ext[plx:plx+1,:,:])
                mu_r_ext = mu_r_ext.at[:plx,:,:].set(mu_r_ext[plx:plx+1,:,:])
            if phx > 0:
                eps_r_ext = eps_r_ext.at[-phx:,:,:].set(eps_r_ext[-phx-1:-phx,:,:])
                sigma_ext = sigma_ext.at[-phx:,:,:].set(sigma_ext[-phx-1:-phx,:,:])
                mu_r_ext = mu_r_ext.at[-phx:,:,:].set(mu_r_ext[-phx-1:-phx,:,:])
            if ply > 0:
                eps_r_ext = eps_r_ext.at[:,:ply,:].set(eps_r_ext[:,ply:ply+1,:])
                sigma_ext = sigma_ext.at[:,:ply,:].set(sigma_ext[:,ply:ply+1,:])
                mu_r_ext = mu_r_ext.at[:,:ply,:].set(mu_r_ext[:,ply:ply+1,:])
            if phy > 0:
                eps_r_ext = eps_r_ext.at[:,-phy:,:].set(eps_r_ext[:,-phy-1:-phy,:])
                sigma_ext = sigma_ext.at[:,-phy:,:].set(sigma_ext[:,-phy-1:-phy,:])
                mu_r_ext = mu_r_ext.at[:,-phy:,:].set(mu_r_ext[:,-phy-1:-phy,:])
            if plz > 0:
                eps_r_ext = eps_r_ext.at[:,:,:plz].set(eps_r_ext[:,:,plz:plz+1])
                sigma_ext = sigma_ext.at[:,:,:plz].set(sigma_ext[:,:,plz:plz+1])
                mu_r_ext = mu_r_ext.at[:,:,:plz].set(mu_r_ext[:,:,plz:plz+1])
            if phz > 0:
                eps_r_ext = eps_r_ext.at[:,:,-phz:].set(eps_r_ext[:,:,-phz-1:-phz])
                sigma_ext = sigma_ext.at[:,:,-phz:].set(sigma_ext[:,:,-phz-1:-phz])
                mu_r_ext = mu_r_ext.at[:,:,-phz:].set(mu_r_ext[:,:,-phz-1:-phz])
            eps_r, sigma, mu_r = eps_r_ext, sigma_ext, mu_r_ext

        materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

        # Apply thin conductors (P4: PEC thin sheets go to pec_mask)
        for tc in self._thin_conductors:
            materials, pec_mask = apply_thin_conductor(
                grid, tc, materials, pec_mask=pec_mask)
            if tc.is_pec:
                pec_shapes.append(tc.shape)

        debye_spec = None
        if debye_masks_by_pole:
            debye_poles = list(debye_masks_by_pole)
            debye_masks = [debye_masks_by_pole[pole] for pole in debye_poles]
            debye_spec = (debye_poles, debye_masks)

        lorentz_spec = None
        if lorentz_masks_by_pole:
            lorentz_poles = list(lorentz_masks_by_pole)
            lorentz_masks = [lorentz_masks_by_pole[pole] for pole in lorentz_poles]
            lorentz_spec = (lorentz_poles, lorentz_masks)

        has_pec = bool(jnp.any(pec_mask))
        kerr_chi3 = chi3_arr if has_kerr else None
        return materials, debye_spec, lorentz_spec, pec_mask if has_pec else None, pec_shapes, kerr_chi3

    @staticmethod
    def _init_dispersion(
        materials: MaterialArrays,
        dt: float,
        debye_spec: _DebyeSpec | None,
        lorentz_spec: _LorentzSpec | None,
    ) -> tuple[MaterialArrays, tuple | None, tuple | None]:
        """Initialize Debye/Lorentz coefficients for the given materials."""
        debye = None
        if debye_spec is not None:
            debye_poles, debye_masks = debye_spec
            debye = init_debye(debye_poles, materials, dt, mask=debye_masks)

        lorentz = None
        if lorentz_spec is not None:
            lorentz_poles, lorentz_masks = lorentz_spec
            lorentz = init_lorentz(lorentz_poles, materials, dt, mask=lorentz_masks)

        return materials, debye, lorentz

    def _build_materials(self, grid: Grid) -> tuple[MaterialArrays, tuple | None, tuple | None]:
        """Build material arrays and optional Debye/Lorentz coefficients."""
        materials, debye_spec, lorentz_spec, _, _, _ = self._assemble_materials(grid)
        _, debye, lorentz = self._init_dispersion(
            materials, grid.dt, debye_spec, lorentz_spec)
        return materials, debye, lorentz

    @staticmethod
    def _range_to_slice(
        value_range: tuple[float, float] | None,
        domain_max: float,
        dx: float,
        grid_size: int,
        axis_pad: int,
    ) -> tuple[tuple[int, int], float]:
        """Convert a physical range to a grid slice and actual physical span."""
        if value_range is None:
            return (axis_pad, grid_size - axis_pad), domain_max
        lo, hi = value_range
        lo_idx = int(round(lo / dx)) + axis_pad
        hi_idx = int(round(hi / dx)) + axis_pad + 1
        if lo_idx < axis_pad or hi_idx > grid_size - axis_pad or hi_idx - lo_idx < 2:
            raise ValueError(
                f"range {value_range!r} does not resolve to a valid aperture on the current grid"
            )
        actual_span = (hi_idx - lo_idx - 1) * dx
        if actual_span <= 0.0 or actual_span > domain_max + 1e-12:
            raise ValueError(
                f"range {value_range!r} resolves to an invalid physical aperture span {actual_span}"
            )
        return (lo_idx, hi_idx), actual_span

    def _build_waveguide_port_config(
        self,
        entry: _WaveguidePortEntry,
        grid: Grid,
        freqs: jnp.ndarray,
        n_steps: int,
    ):
        normal_axis = entry.direction[1]
        axis_idx = {"x": 0, "y": 1, "z": 2}[normal_axis]
        pos_vec = [0.0, 0.0, 0.0]
        pos_vec[axis_idx] = entry.x_position
        x_index = grid.position_to_index(tuple(pos_vec))[axis_idx]
        snapped_source_plane = (x_index - grid.axis_pads[axis_idx]) * grid.dx
        step_sign = 1 if entry.direction.startswith("+") else -1
        measured_reference_plane = snapped_source_plane + step_sign * entry.ref_offset * grid.dx
        measured_probe_plane = snapped_source_plane + step_sign * entry.probe_offset * grid.dx
        axis_domain = self._domain[axis_idx]
        if (
            measured_reference_plane < 0.0
            or measured_reference_plane > axis_domain
            or measured_probe_plane < 0.0
            or measured_probe_plane > axis_domain
            or x_index + step_sign * entry.ref_offset < 0
            or x_index + step_sign * entry.ref_offset >= grid.shape[axis_idx]
            or x_index + step_sign * entry.probe_offset < 0
            or x_index + step_sign * entry.probe_offset >= grid.shape[axis_idx]
        ):
            raise ValueError(
                "Waveguide port measurement planes exceed the physical "
                f"{normal_axis}-domain after grid snapping; reduce ref_offset/probe_offset, "
                "flip direction, or move x_position inward"
            )
        if normal_axis == "x":
            u_slice, a_span = self._range_to_slice(entry.y_range, self._domain[1], grid.dx, grid.ny, grid.axis_pads[1])
            v_slice, b_span = self._range_to_slice(entry.z_range, self._domain[2], grid.dx, grid.nz, grid.axis_pads[2])
        elif normal_axis == "y":
            u_slice, a_span = self._range_to_slice(entry.x_range, self._domain[0], grid.dx, grid.nx, grid.axis_pads[0])
            v_slice, b_span = self._range_to_slice(entry.z_range, self._domain[2], grid.dx, grid.nz, grid.axis_pads[2])
        else:
            u_slice, a_span = self._range_to_slice(entry.x_range, self._domain[0], grid.dx, grid.nx, grid.axis_pads[0])
            v_slice, b_span = self._range_to_slice(entry.y_range, self._domain[1], grid.dx, grid.ny, grid.axis_pads[1])
        port = WaveguidePort(
            x_index=x_index,
            y_slice=None,
            z_slice=None,
            a=a_span,
            b=b_span,
            mode=entry.mode,
            mode_type=entry.mode_type,
            direction=entry.direction,
            x_position=snapped_source_plane,
            normal_axis=normal_axis,
            u_slice=u_slice,
            v_slice=v_slice,
        )
        if entry.n_modes > 1:
            cfgs = init_multimode_waveguide_port(
                port,
                grid.dx,
                freqs,
                n_modes=entry.n_modes,
                f0=entry.f0 if entry.f0 is not None else self._freq_max / 2,
                bandwidth=entry.bandwidth,
                amplitude=entry.amplitude,
                probe_offset=entry.probe_offset,
                ref_offset=entry.ref_offset,
                dft_total_steps=n_steps,
                dt=float(grid.dt),
                waveform=entry.waveform,
                mode_profile=entry.mode_profile,
                grid=grid,
            )
            return cfgs
        cfg = init_waveguide_port(
            port,
            grid.dx,
            freqs,
            f0=entry.f0 if entry.f0 is not None else self._freq_max / 2,
            bandwidth=entry.bandwidth,
            amplitude=entry.amplitude,
            probe_offset=entry.probe_offset,
            ref_offset=entry.ref_offset,
            dft_total_steps=n_steps,
            dt=float(grid.dt),
            waveform=entry.waveform,
            mode_profile=entry.mode_profile,
            grid=grid,
        )
        return cfg

    def compute_waveguide_s_matrix(
        self,
        *,
        n_steps: int | None = None,
        num_periods: float = 20.0,
        normalize: bool = False,
        subpixel_smoothing: bool = False,
    ) -> WaveguideSMatrixResult:
        """Compute a theoretically clean axis-normal boundary-aperture waveguide S-matrix.

        Parameters
        ----------
        num_periods : float
            Length of the FDTD run (in source-period multiples) used to
            derive ``n_steps`` when ``n_steps`` is not given. The
            spectra are computed POST-SCAN from the recorded modal V/I
            time series via a rectangular full-record DFT (matching
            OpenEMS's ``utilities.DFT_time2freq``); ``num_periods``
            therefore governs both the CPML drain horizon AND the DFT
            integration window. Phase 2 cleanup (2026-04-25) removed
            the legacy ``num_periods_dft`` early-gate knob — the rect
            full-record DFT is finite-energy on the recorded transient
            so no gating is needed even on strong reflectors.
        normalize : bool
            When True, run a two-run normalization that cancels Yee-grid
            numerical dispersion.  A reference simulation with vacuum (no
            user geometry) is run automatically, and the device S-params
            are divided by the reference incident waves.

            **Recommendation:** Always use ``normalize=True`` for accurate
            S-parameters, reciprocity (S12==S21), and comparison with
            measurements.  Use ``normalize=False`` only for fast relative
            comparisons (e.g., optimizer inner loop).
        """
        if not normalize:
            import warnings
            warnings.warn(
                "compute_waveguide_s_matrix(normalize=False): S-parameters "
                "include Yee numerical dispersion artifacts. For accurate "
                "results, reciprocity, or measurement comparison, use "
                "normalize=True.",
                stacklevel=2,
            )
        if self._ports or self._tfsf:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported together with lumped ports or TFSF"
            )
        if self._periodic_axes:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported with manual periodic-axis overrides"
            )
        if len(self._waveguide_ports) < 2:
            raise ValueError(
                "compute_waveguide_s_matrix() requires at least two waveguide ports"
            )

        entries = list(self._waveguide_ports)
        if any(entry.probe_plane is not None for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() does not use per-port probe_plane; use reference_plane only or leave probe_plane unset"
            )
        if any(entry.calibration_preset not in (None, "measured") for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() currently supports only measured/default reference planes or explicit reference_plane overrides"
            )

        # Non-uniform-mesh dispatch. Earlier the uniform scan ran with
        # the coarse boundary dx and silently ignored ``dx_profile`` /
        # ``dy_profile`` (handover v2 experiment 12). The dedicated NU
        # two-run extractor below is enabled when its supported scope
        # is met (``normalize=True``, single-mode ports); otherwise
        # raise so the user is not given silently-wrong numbers.
        if self._dx_profile is not None or self._dy_profile is not None:
            unsupported = []
            if not normalize:
                unsupported.append("normalize=True is required")
            if any(entry.n_modes > 1 for entry in entries):
                unsupported.append("multi-mode ports (n_modes>1) are not supported")
            if unsupported:
                raise NotImplementedError(
                    "compute_waveguide_s_matrix() on a non-uniform mesh "
                    "(dx_profile / dy_profile) supports normalize=True "
                    "and single-mode ports. "
                    + "; ".join(unsupported)
                    + ". Drop the dx/dy profile to use the uniform lane."
                )
            return self._compute_waveguide_s_matrix_nu(
                n_steps=n_steps,
                num_periods=num_periods,
                normalize=normalize,
            )

        grid = self._build_grid()
        base_materials, debye_spec, lorentz_spec, pec_mask_wg, _, _ = self._assemble_materials(grid)
        # Waveguide S-matrix runner doesn't support pec_mask yet.
        # Fold PEC mask back into high sigma for compatibility.
        if pec_mask_wg is not None:
            base_materials = base_materials._replace(
                sigma=jnp.where(pec_mask_wg, 1e10, base_materials.sigma))
        materials = base_materials
        if n_steps is None:
            n_steps = grid.num_timesteps(num_periods=num_periods)
        _, debye, lorentz = self._init_dispersion(materials, grid.dt, debye_spec, lorentz_spec)

        def _resolve_freqs(entry: _WaveguidePortEntry) -> jnp.ndarray:
            if entry.freqs is not None:
                return entry.freqs
            return jnp.linspace(self._freq_max / 10, self._freq_max, entry.n_freqs)

        freqs = _resolve_freqs(entries[0])
        for entry in entries[1:]:
            entry_freqs = _resolve_freqs(entry)
            if entry_freqs.shape != freqs.shape or not np.allclose(np.asarray(entry_freqs), np.asarray(freqs)):
                raise ValueError("waveguide S-matrix requires matching frequency grids on all ports")

        # Build configs — may be a single config or a list of configs per port
        has_multimode = any(entry.n_modes > 1 for entry in entries)
        raw_cfgs = [self._build_waveguide_port_config(entry, grid, freqs, n_steps) for entry in entries]

        # Unify source waveform across all ports so that the S-matrix
        # extraction uses identical excitation.  Different source spectra
        # (from mismatched f0/bandwidth) cause S11 ≠ S22 artifacts in the
        # unnormalized path because V/I decomposition error varies with
        # frequency.  Use port 0's waveform as the canonical source.
        def _flatten_cfgs(cfgs):
            out = []
            for c in cfgs:
                if isinstance(c, list):
                    out.extend(c)
                else:
                    out.append(c)
            return out

        flat0 = _flatten_cfgs(raw_cfgs)
        ref_t0 = flat0[0].src_t0
        ref_tau = flat0[0].src_tau
        need_unify = any(
            c.src_t0 != ref_t0 or c.src_tau != ref_tau for c in flat0[1:]
        )
        if need_unify:
            raw_cfgs = [
                cfg._replace(src_t0=ref_t0, src_tau=ref_tau)
                if not isinstance(cfg, list)
                else [c._replace(src_t0=ref_t0, src_tau=ref_tau) for c in cfg]
                for cfg in raw_cfgs
            ]

        if has_multimode:
            # Multi-mode path: each raw_cfg is a list of WaveguidePortConfig
            port_mode_cfgs: list[list] = []
            for entry, raw in zip(entries, raw_cfgs):
                if isinstance(raw, list):
                    port_mode_cfgs.append(raw)
                else:
                    port_mode_cfgs.append([raw])

            ref_shifts_mm = []
            for entry, mode_cfgs in zip(entries, port_mode_cfgs):
                first_cfg = mode_cfgs[0]
                planes = waveguide_plane_positions(first_cfg)
                desired_ref = (
                    entry.reference_plane
                    if entry.reference_plane is not None
                    else planes["source"]
                )
                ref_shifts_mm.append(desired_ref - planes["reference"])

            if normalize:
                raise ValueError(
                    "compute_waveguide_s_matrix(normalize=True) is not yet "
                    "supported with n_modes > 1"
                )

            s_params, mode_map = extract_multimode_s_matrix(
                grid,
                materials,
                port_mode_cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes="".join(axis for axis in "xyz" if axis not in grid.cpml_axes),
                debye=debye,
                lorentz=lorentz,
                ref_shifts=ref_shifts_mm,
            )
            reference_planes = np.array(ref_shifts_mm, dtype=float)
            # Build port names including mode indices
            port_names_mm = []
            port_directions_mm = []
            for port_idx, mode_idx, mtype, m_n in mode_map:
                entry = entries[port_idx]
                port_names_mm.append(f"{entry.name}_mode{mode_idx}_{mtype}{m_n[0]}{m_n[1]}")
                port_directions_mm.append(entry.direction)
            return WaveguideSMatrixResult(
                s_params=np.array(s_params),
                freqs=np.array(freqs),
                port_names=tuple(port_names_mm),
                port_directions=tuple(port_directions_mm),
                reference_planes=reference_planes,
            )

        # Single-mode path (original behavior)
        cfgs = raw_cfgs

        def _slices_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
            return max(a[0], b[0]) < min(a[1], b[1])

        by_direction = {}
        for entry, cfg in zip(entries, cfgs):
            by_direction.setdefault(entry.direction, []).append(cfg)

        for direction, side_cfgs in by_direction.items():
            plane_indices = {cfg.x_index for cfg in side_cfgs}
            if len(plane_indices) != 1:
                raise ValueError(
                    f"waveguide ports on boundary {direction} must share one boundary plane"
                )
            for i in range(len(side_cfgs)):
                for j in range(i + 1, len(side_cfgs)):
                    if _slices_overlap((side_cfgs[i].u_lo, side_cfgs[i].u_hi), (side_cfgs[j].u_lo, side_cfgs[j].u_hi)) and _slices_overlap((side_cfgs[i].v_lo, side_cfgs[i].v_hi), (side_cfgs[j].v_lo, side_cfgs[j].v_hi)):
                        raise ValueError(
                            f"waveguide ports on the same {direction} boundary must have disjoint apertures"
                        )

        ref_shifts = []
        for entry, cfg in zip(entries, cfgs):
            # Default reference plane = the user-facing port plane
            # (snapped x_position). Previously defaulted to the internal
            # ``reference_x_m`` (= source + ref_offset·dx) which left the
            # returned S-matrix phase-shifted by `exp(-jβ·ref_offset·dx)`
            # relative to the physical port — a silent convention mismatch
            # vs. Meep, OpenEMS, and any analytic formula the user would
            # compare against. Keep the ``entry.reference_plane`` override
            # for explicit user control.
            planes = waveguide_plane_positions(cfg)
            desired_ref = (
                entry.reference_plane
                if entry.reference_plane is not None
                else planes["source"]
            )
            ref_shifts.append(desired_ref - planes["reference"])

        # Compute Kottke per-component smoothed permittivity if requested.
        # Mirrors rfx/runners/uniform.py: shape_eps_pairs from sim geometry,
        # then compute_smoothed_eps. The reference run is vacuum and has no
        # ε interfaces, so it always passes aniso_eps=None inside the
        # extractor.
        aniso_eps = None
        if subpixel_smoothing:
            from rfx.geometry.smoothing import compute_smoothed_eps
            shape_eps_pairs = [
                (entry.shape, self._resolve_material(entry.material_name).eps_r)
                for entry in self._geometry
            ]
            if shape_eps_pairs:
                aniso_eps = compute_smoothed_eps(
                    grid, shape_eps_pairs, background_eps=1.0,
                )

        if normalize:
            # Build reference materials: vacuum everywhere (no user geometry).
            from rfx.core.yee import init_materials as _init_vacuum_materials
            ref_materials = _init_vacuum_materials(grid.shape)
            s_params = extract_waveguide_s_params_normalized(
                grid,
                materials,
                ref_materials,
                cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes="".join(axis for axis in "xyz" if axis not in grid.cpml_axes),
                debye=debye,
                lorentz=lorentz,
                ref_debye=None,
                ref_lorentz=None,
                ref_shifts=ref_shifts,
                aniso_eps=aniso_eps,
            )
        else:
            s_params = extract_waveguide_s_matrix(
                grid,
                materials,
                cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes="".join(axis for axis in "xyz" if axis not in grid.cpml_axes),
                debye=debye,
                lorentz=lorentz,
                ref_shifts=ref_shifts,
                aniso_eps=aniso_eps,
            )
        reference_planes = np.array(
            [
                entry.reference_plane
                if entry.reference_plane is not None
                else waveguide_plane_positions(cfg)["reference"]
                for entry, cfg in zip(entries, cfgs)
            ],
            dtype=float,
        )
        return WaveguideSMatrixResult(
            s_params=np.array(s_params),
            freqs=np.array(freqs),
            port_names=tuple(entry.name for entry in entries),
            port_directions=tuple(entry.direction for entry in entries),
            reference_planes=reference_planes,
        )

    def _compute_waveguide_s_matrix_nu(
        self,
        *,
        n_steps: int | None,
        num_periods: float,
        normalize: bool,
    ) -> WaveguideSMatrixResult:
        """Non-uniform-mesh two-run S-matrix extraction.

        Drives each port in turn, running device + vacuum-reference
        scans through ``run_nonuniform_path`` so ``dx_profile`` /
        ``dy_profile`` actually flow into the Yee update. The per-port
        drive is implemented by temporarily zeroing ``amplitude`` on
        non-driven entries; the original port list is restored in a
        ``finally`` block. Reference run uses ``eps_override`` /
        ``sigma_override`` to replace the assembled materials with
        vacuum before the scan launches.

        Current scope (matches the uniform path minus a few niceties):
          - ``normalize=True`` only.
          - Single-mode ports (``n_modes == 1``) only.

        Extracts ``a_inc`` / ``b_out`` via the same
        ``extract_waveguide_port_waves`` helper as the uniform path and
        applies the same diagonal-subtraction + off-diagonal-division
        normalisation (see ``extract_waveguide_s_params_normalized``
        in ``rfx/sources/waveguide_port.py``).
        """
        from dataclasses import replace as _dc_replace
        from rfx.runners.nonuniform import (
            run_nonuniform_path,
            assemble_materials_nu,
        )
        from rfx.sources.waveguide_port import (
            extract_waveguide_port_waves,
            waveguide_plane_positions,
        )

        if not normalize:
            raise NotImplementedError(
                "compute_waveguide_s_matrix(normalize=False) is not yet "
                "supported on the non-uniform mesh path; use normalize=True "
                "or drop dx/dy_profile to stay on the uniform lane."
            )

        entries = list(self._waveguide_ports)
        if any(entry.n_modes > 1 for entry in entries):
            raise NotImplementedError(
                "Multi-mode waveguide ports are not yet supported on the "
                "non-uniform mesh path."
            )

        n_ports = len(entries)

        # ``_build_nonuniform_grid`` requires a concrete dz_profile.
        # Synthesise one from the scalar dx when the user did not supply
        # a dz_profile (same semantics as the uniform lane's implicit
        # z-resolution). Restored in the ``finally`` below.
        _dz_profile_saved = self._dz_profile
        if self._dz_profile is None:
            _nz = int(round(float(self._domain[2]) / float(self._dx)))
            self._dz_profile = np.full(max(_nz, 1), float(self._dx))

        # Build the grid directly so we can restrict ``cpml_axes`` to
        # axes that are not fully PEC/PMC-bounded. The rasteriser (see
        # ``rfx/geometry/rasterize.py::coords_from_nonuniform_grid``)
        # uses a single ``grid.cpml_layers`` offset for every axis;
        # when a fully PEC-bounded axis is shorter than
        # ``cpml_layers + 1`` cells the offset slice hits IndexError.
        # Dropping that axis from ``cpml_axes`` keeps the physical
        # grid identical (PEC faces already have pad=0) but zeroes the
        # offset so the rasteriser snaps cells to 0 cleanly.
        from rfx.runners.nonuniform import build_nonuniform_grid
        pec_set = (self._boundary_spec.pec_faces()
                   if self._boundary_spec is not None else None) or set()
        pmc_set = (self._boundary_spec.pmc_faces()
                   if self._boundary_spec is not None else None) or set()

        def _axis_fully_closed(ax: str) -> bool:
            return {f"{ax}_lo", f"{ax}_hi"}.issubset(pec_set | pmc_set)

        cpml_axes = "".join(
            ax for ax in "xyz"
            if ax not in (self._periodic_axes or "")
            and not _axis_fully_closed(ax)
        )
        try:
            grid = build_nonuniform_grid(
                self._freq_max, self._domain, self._dx, self._cpml_layers,
                self._dz_profile,
                dx_profile=self._dx_profile,
                dy_profile=self._dy_profile,
                pec_faces=pec_set or None,
                pmc_faces=pmc_set or None,
                cpml_axes=cpml_axes,
            )
        except Exception:
            self._dz_profile = _dz_profile_saved
            raise
        if n_steps is None:
            # ``NonUniformGrid`` does not expose ``num_timesteps`` (known
            # asymmetry vs. ``Grid``); inline the same formula here.
            n_steps = int(np.ceil(num_periods / self._freq_max / float(grid.dt)))

        # Assemble device materials once to learn the full array shape;
        # vacuum reference is shape-matched onto that same array.
        dev_materials_concrete, _, _, _ = assemble_materials_nu(self, grid)
        vacuum_eps = jnp.ones_like(dev_materials_concrete.eps_r)
        vacuum_sigma = jnp.zeros_like(dev_materials_concrete.sigma)

        # Frequency grid must match across ports.
        port_freqs = entries[0].freqs
        if port_freqs is None:
            port_freqs = jnp.linspace(
                self._freq_max / 10, self._freq_max, entries[0].n_freqs,
            )
        for entry in entries[1:]:
            other = entry.freqs if entry.freqs is not None else jnp.linspace(
                self._freq_max / 10, self._freq_max, entry.n_freqs,
            )
            if other.shape != port_freqs.shape or not np.allclose(
                np.asarray(other), np.asarray(port_freqs)
            ):
                raise ValueError(
                    "waveguide S-matrix requires matching frequency grids on all ports"
                )
        n_freqs = int(port_freqs.shape[0])

        s_matrix = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex64)
        ref_shifts: tuple[float, ...] | None = None
        reference_planes_out: np.ndarray | None = None

        original_entries = list(entries)
        try:
            for drive_idx in range(n_ports):
                self._waveguide_ports = [
                    _dc_replace(
                        e,
                        amplitude=(e.amplitude if idx == drive_idx else 0.0),
                    )
                    for idx, e in enumerate(original_entries)
                ]

                dev_result = run_nonuniform_path(self, n_steps=n_steps)
                ref_result = run_nonuniform_path(
                    self,
                    n_steps=n_steps,
                    eps_override=vacuum_eps,
                    sigma_override=vacuum_sigma,
                )

                dev_wg = dev_result.waveguide_ports or {}
                ref_wg = ref_result.waveguide_ports or {}
                if len(dev_wg) != n_ports or len(ref_wg) != n_ports:
                    raise RuntimeError(
                        "NU waveguide S-matrix expected one final cfg per "
                        "port on both device and reference runs"
                    )

                # Compute ref_shifts from the first drive's configs (same
                # measured planes for every drive / run).
                if ref_shifts is None:
                    shifts = []
                    planes_out = []
                    for entry in original_entries:
                        cfg = dev_wg[entry.name]
                        planes = waveguide_plane_positions(cfg)
                        desired = (
                            entry.reference_plane
                            if entry.reference_plane is not None
                            else planes["source"]
                        )
                        shifts.append(desired - planes["reference"])
                        planes_out.append(desired)
                    ref_shifts = tuple(shifts)
                    reference_planes_out = np.asarray(planes_out, dtype=float)

                drive_name = original_entries[drive_idx].name
                a_inc_ref, _ = extract_waveguide_port_waves(
                    ref_wg[drive_name], ref_shift=ref_shifts[drive_idx],
                )
                a_inc_ref_np = np.asarray(a_inc_ref)
                safe_a_inc = np.where(
                    np.abs(a_inc_ref_np) > 1e-30,
                    a_inc_ref_np,
                    np.ones_like(a_inc_ref_np),
                )

                for recv_idx in range(n_ports):
                    recv_name = original_entries[recv_idx].name
                    _, b_ref = extract_waveguide_port_waves(
                        ref_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                    )
                    _, b_dev = extract_waveguide_port_waves(
                        dev_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                    )
                    b_ref_np = np.asarray(b_ref)
                    b_dev_np = np.asarray(b_dev)

                    if recv_idx == drive_idx:
                        s_matrix[recv_idx, drive_idx, :] = (
                            b_dev_np - b_ref_np
                        ) / safe_a_inc
                    else:
                        safe_b = np.where(
                            np.abs(b_ref_np) > 1e-30,
                            b_ref_np,
                            np.ones_like(b_ref_np),
                        )
                        s_matrix[recv_idx, drive_idx, :] = b_dev_np / safe_b
        finally:
            self._waveguide_ports = original_entries
            self._dz_profile = _dz_profile_saved

        return WaveguideSMatrixResult(
            s_params=np.asarray(s_matrix),
            freqs=np.asarray(port_freqs),
            port_names=tuple(e.name for e in original_entries),
            port_directions=tuple(e.direction for e in original_entries),
            reference_planes=reference_planes_out
            if reference_planes_out is not None
            else np.array(
                [
                    e.reference_plane if e.reference_plane is not None
                    else 0.0
                    for e in original_entries
                ],
                dtype=float,
            ),
        )

    @staticmethod
    def _validate_tfsf_vacuum_boundary(materials: MaterialArrays, tfsf_cfg) -> None:
        """Ensure the TFSF x-boundary planes remain vacuum.

        The current TFSF correction assumes vacuum on and immediately
        adjacent to the TFSF x boundaries. Fail loudly instead of
        allowing silently wrong scattered fields.
        """
        boundary_slices = (
            ("x_lo-1", slice(tfsf_cfg.x_lo - 1, tfsf_cfg.x_lo)),
            ("x_lo", slice(tfsf_cfg.x_lo, tfsf_cfg.x_lo + 1)),
            ("x_hi", slice(tfsf_cfg.x_hi, tfsf_cfg.x_hi + 1)),
            ("x_hi+1", slice(tfsf_cfg.x_hi + 1, tfsf_cfg.x_hi + 2)),
        )

        for plane_name, xs in boundary_slices:
            eps = np.asarray(materials.eps_r[xs, :, :])
            sigma = np.asarray(materials.sigma[xs, :, :])
            mu = np.asarray(materials.mu_r[xs, :, :])
            if not (
                np.allclose(eps, 1.0)
                and np.allclose(sigma, 0.0)
                and np.allclose(mu, 1.0)
            ):
                raise ValueError(
                    "TFSF plane-wave source requires vacuum on and adjacent to "
                    f"the TFSF x boundaries; non-vacuum material found at {plane_name}"
                )

    # ---- subgridded run ----

    def _run_subgridded(self, grid_coarse, base_materials_coarse, pec_mask_coarse,
                        n_steps):
        """Run simulation using SBP-SAT subgridding (JIT-compiled)."""
        from rfx.runners.subgridded import run_subgridded_path
        return run_subgridded_path(self, grid_coarse, base_materials_coarse,
                                   pec_mask_coarse, n_steps)

    # ---- non-uniform mesh run path ----

    def _build_nonuniform_grid(self) -> NonUniformGrid:
        """Build a NonUniformGrid from stored dz_profile (and optional
        dx_profile / dy_profile). A uniform profile on any axis is
        synthesised from the scalar ``dx`` when the profile is not set.
        """
        from rfx.runners.nonuniform import build_nonuniform_grid
        return build_nonuniform_grid(
            self._freq_max, self._domain, self._dx, self._cpml_layers,
            self._dz_profile,
            dx_profile=self._dx_profile,
            dy_profile=self._dy_profile,
            pec_faces=self._boundary_spec.pec_faces()
                if self._boundary_spec is not None else None,
            pmc_faces=self._boundary_spec.pmc_faces()
                if self._boundary_spec is not None else None,
            cpml_axes="".join(
                ax for ax in "xyz"
                if ax not in (self._periodic_axes or "")
            ),
        )

    def _assemble_materials_nu(
        self, grid: NonUniformGrid,
    ) -> tuple[MaterialArrays, object, object, jnp.ndarray | None]:
        """Build material arrays and dispersion specs for non-uniform grid."""
        from rfx.runners.nonuniform import assemble_materials_nu
        return assemble_materials_nu(self, grid)

    def _pos_to_nu_index(self, grid: NonUniformGrid, pos):
        """Convert physical (x, y, z) to non-uniform grid indices."""
        from rfx.runners.nonuniform import pos_to_nu_index
        return pos_to_nu_index(grid, pos)

    def _run_nonuniform(self, *, n_steps, compute_s_params=None,
                        s_param_freqs=None, subpixel_smoothing: bool = False,
                        checkpoint: bool = False):
        """Run simulation on non-uniform grid with graded dz."""
        from rfx.runners.nonuniform import run_nonuniform_path
        return run_nonuniform_path(
            self,
            n_steps=n_steps,
            compute_s_params=compute_s_params,
            s_param_freqs=s_param_freqs,
            subpixel_smoothing=subpixel_smoothing,
            checkpoint=checkpoint,
        )

    @staticmethod
    def _warn_unsupported_run_kwargs(path_name: str,
                                     unsupported_kwargs: dict) -> None:
        """Emit ``UserWarning`` for any Simulation.run kwarg that a given
        dispatch path drops. Only non-default values are surfaced.

        Pre-v1.7.5 the distributed / non-uniform / subgridded paths
        silently dropped most of the ``run`` kwargs; this helper makes
        the drop explicit at the API boundary so users can tell their
        request was not honoured. See GitHub tracking issue for the
        feature-request backlog to actually propagate these kwargs.
        """
        import warnings as _w
        # Per-kwarg "silent" values — the kwarg is dropped only if the
        # user set it to a value that asks the dispatch path to do
        # something it does not support. ``compute_s_params=False``
        # is silent because it matches the path's actual behaviour
        # (no S-matrix assembly), while ``compute_s_params=True``
        # warns because the user asked for something that will not
        # happen.
        silent_values = {
            "subpixel_smoothing": (False,),
            "checkpoint": (False,),
            "snapshot": (None,),
            "until_decay": (None,),
            "conformal_pec": (False,),
            "compute_s_params": (None, False),
            "s_param_freqs": (None,),
            "s_param_n_steps": (None,),
        }
        reasons = {
            "subpixel_smoothing":
                "per-component anisotropic eps is not wired on this path",
            "checkpoint":
                "reverse-mode AD will store the full tape (no "
                "checkpoint-every support here)",
            "snapshot":
                "scan-body field snapshotting is not wired on this path",
            "until_decay":
                "scan body runs for exactly n_steps; no decay-based "
                "termination on this path (use the uniform-mesh path)",
            "conformal_pec":
                "Dey-Mittra conformal weights are computed on a uniform "
                "staircase mesh only",
            "compute_s_params":
                "S-matrix assembly is not plumbed through this path",
            "s_param_freqs":
                "S-matrix assembly is not plumbed through this path",
            "s_param_n_steps":
                "S-matrix assembly is not plumbed through this path",
        }
        for kw, val in unsupported_kwargs.items():
            silent = silent_values.get(kw, (None,))
            if val in silent:
                continue
            reason = reasons.get(kw, "not propagated")
            _w.warn(
                f"{kw}={val!r} is silently ignored on the {path_name} run "
                f"path ({reason}).",
                UserWarning, stacklevel=3,
            )

    def _auto_configure_mesh(self) -> None:
        """P1: Auto-detect features and set dx/dz_profile when dx=None.

        Uses the existing auto_configure() infrastructure to derive cell size
        from geometry dimensions and material properties.  Runs only once per
        simulation — subsequent calls are no-ops.
        """
        import warnings as _w
        from rfx.auto_config import auto_configure

        geometry_pairs = [
            (entry.shape, entry.material_name)
            for entry in self._geometry
        ]
        materials_dict = {}
        for name, spec in self._materials.items():
            materials_dict[name] = {
                "eps_r": spec.eps_r,
                "sigma": spec.sigma,
            }
        # Include library materials used by geometry but not explicitly registered
        for entry in self._geometry:
            mname = entry.material_name
            if mname not in materials_dict and mname in MATERIAL_LIBRARY:
                materials_dict[mname] = MATERIAL_LIBRARY[mname]

        config = auto_configure(
            geometry=geometry_pairs,
            freq_range=(self._freq_max / 10, self._freq_max),
            materials=materials_dict,
            boundary=self._boundary,
        )

        self._dx = config.dx
        if config.dz_profile is not None and self._dz_profile is None:
            self._dz_profile = config.dz_profile
            # Update domain z from dz_profile
            dz_total = float(np.sum(config.dz_profile))
            self._domain = (self._domain[0], self._domain[1], dz_total)

        _w.warn(
            f"Auto mesh: dx={config.dx*1e3:.3f}mm "
            f"({config.cells_per_wavelength:.0f} cells/λ)"
            + (f", non-uniform z ({len(config.dz_profile)} cells)"
               if config.dz_profile is not None else "")
            + ". Set dx= explicitly to suppress.",
            stacklevel=3,
        )
        for w in config.warnings:
            _w.warn(w, stacklevel=3)

    def _validate_mesh_quality(self) -> None:
        """Pre-simulation mesh quality check (P0).

        Scans all geometry elements against the grid cell size and warns
        about under-resolved features. Prevents silent garbage results
        from mesh-related setup errors.
        """
        import warnings as _w

        # Tracer-valued profiles (mesh-as-design-variable gradient) cannot
        # participate in host-side min/len/indexing. Advisory warnings
        # are skipped in that case — correctness is preserved downstream.
        if any(
            p is not None and is_tracer(p)
            for p in (self._dx_profile, self._dy_profile, self._dz_profile)
        ):
            return

        dx = self._dx
        if dx is None:
            dx = C0 / self._freq_max / 20.0

        # Determine minimum cell size per axis — use profile min when
        # non-uniform xy is active, so we don't flag features that are
        # actually well-resolved in their local fine-mesh region.
        min_dx = float(min(self._dx_profile)) if self._dx_profile is not None else dx
        min_dy = float(min(self._dy_profile)) if self._dy_profile is not None else dx
        if self._dz_profile is not None:
            min_dz = min(self._dz_profile)
        else:
            min_dz = dx

        for entry in self._geometry:
            shape = entry.shape
            mat_name = entry.material_name

            # Get bounding box dimensions
            if hasattr(shape, "bounding_box"):
                try:
                    c1, c2 = shape.bounding_box()
                    dims = [abs(c2[i] - c1[i]) for i in range(3)]
                except (NotImplementedError, TypeError):
                    continue
            else:
                continue

            cell_sizes = [min_dx, min_dy, min_dz]

            for axis, (dim, cell) in enumerate(zip(dims, cell_sizes)):
                if dim <= 0:
                    # Zero-thickness geometry
                    axis_name = "xyz"[axis]
                    _w.warn(
                        f"Zero-thickness geometry '{mat_name}' along "
                        f"{axis_name}-axis. On non-uniform mesh this may "
                        f"produce empty rasterization. Consider giving it "
                        f"at least one cell of thickness ({cell*1e3:.2f}mm).",
                        stacklevel=3,
                    )
                elif dim < cell:
                    axis_name = "xyz"[axis]
                    cells_count = dim / cell
                    # Check if this is a PEC material that could use thin sheet
                    mat = self._resolve_material(mat_name)
                    is_pec = mat.sigma >= self._PEC_SIGMA_THRESHOLD
                    hint = (
                        " Use add_thin_conductor() for sub-cell PEC sheet."
                        if is_pec else
                        " Use non-uniform mesh or reduce dx."
                    )
                    _w.warn(
                        f"'{mat_name}' {axis_name}-extent {dim*1e3:.2f}mm = "
                        f"{cells_count:.1f} cells — below 1 cell resolution."
                        + hint,
                        stacklevel=3,
                    )
                else:
                    # Physics-based resolution thresholds (issue #37).
                    # PEC with extent <3 cells is a thin sheet — 1-cell
                    # rasterization is canonical. Only warn on partial
                    # volume: 3-5 cells thick PEC slabs.
                    # Dielectric: cells per local λ_eff, not cells per
                    # geometry extent.
                    mat = self._resolve_material(mat_name)
                    is_pec = mat.sigma >= self._PEC_SIGMA_THRESHOLD
                    axis_name = "xyz"[axis]
                    cells = dim / cell
                    if is_pec:
                        if 3.0 <= cells < 5.0:
                            _w.warn(
                                f"PEC '{mat_name}' {axis_name}-extent "
                                f"{dim*1e3:.2f}mm = {cells:.1f} cells — "
                                "volume under-resolved (PEC volume needs "
                                "≥5 cells; thin sheets <3 cells are fine).",
                                stacklevel=3,
                            )
                    else:
                        eps_r = float(mat.eps_r) if mat.eps_r else 1.0
                        lam_eff = (
                            C0 / self._freq_max / math.sqrt(max(eps_r, 1.0))
                        )
                        cells_per_lam = lam_eff / cell
                        # rfx's Yee update is 2nd-order in bulk but
                        # degrades to 1st-order at ε-discontinuities
                        # because subpixel smoothing is default OFF
                        # (Meep ships it ON and stays 2nd-order). For
                        # phase-accurate propagation we need ≥15 cells
                        # per λ_eff — the traditional λ/10 rule applies
                        # to subpixel-smoothed codes. S-parameter
                        # extraction with a port or flux monitor
                        # amplifies dielectric-interface phase error
                        # into |S| magnitude error (see
                        # examples/crossval/11 rfx-vs-analytic audit,
                        # 2026-04-24): at 17.7 cells/λ_eff we measure
                        # ~5% |S21| deficit at Fabry-Perot peaks; at
                        # 35 cells/λ_eff (dx halved) it halves to ~2%.
                        # Require 20 cells/λ_eff when S-param
                        # extraction is active.
                        sparam_active = bool(
                            self._waveguide_ports
                            or self._flux_monitors
                        )
                        threshold = 20.0 if sparam_active else 15.0
                        if cells_per_lam < threshold:
                            suffix = (
                                " S-parameter extraction amplifies "
                                "ε-interface phase error into |S| "
                                "magnitude error; ~5% |S21| deficit "
                                "expected at 17 cells/λ_eff."
                                if sparam_active else
                                " Yee without subpixel smoothing has "
                                "1st-order convergence at ε interfaces."
                            )
                            _w.warn(
                                f"dielectric '{mat_name}' on {axis_name}: "
                                f"{cells_per_lam:.1f} cells per λ_eff "
                                f"(eps_r={eps_r:.2f}, freq_max="
                                f"{self._freq_max/1e9:.2f}GHz, "
                                f"dx={cell*1e3:.3f}mm). Need ≥"
                                f"{threshold:.0f} cells/λ_eff for "
                                f"phase-accurate propagation."
                                f"{suffix}",
                                stacklevel=3,
                            )

        # Check gaps between PEC structures
        pec_entries = [e for e in self._geometry if e.material_name == "pec"]
        if len(pec_entries) >= 2:
            for i in range(len(pec_entries)):
                for j in range(i + 1, min(i + 5, len(pec_entries))):
                    try:
                        c1a, c2a = pec_entries[i].shape.bounding_box()
                        c1b, c2b = pec_entries[j].shape.bounding_box()
                        # Min gap along each axis
                        for ax in range(3):
                            gap = max(0, max(c1b[ax] - c2a[ax], c1a[ax] - c2b[ax]))
                            cell = [dx, dx, min_dz][ax]
                            if 0 < gap < 3 * cell:
                                _w.warn(
                                    f"Gap between PEC structures: "
                                    f"{gap*1e3:.2f}mm = {gap/cell:.1f} cells "
                                    f"along {'xyz'[ax]} — coupling may be "
                                    f"under-resolved.",
                                    stacklevel=3,
                                )
                    except (NotImplementedError, TypeError, AttributeError):
                        continue

        # Physics-based numerical dispersion check (Taflove Ch. 4).
        # Instead of a fixed aspect-ratio heuristic, compute the actual
        # per-axis phase velocity error at freq_max from the FDTD
        # dispersion relation. This is application-independent.
        self._check_numerical_dispersion()

        # Thin-metal-on-NU-mesh symmetry (Meep/OpenEMS convention — issue #48).
        self._validate_thin_metal_on_nu_mesh()

    def _check_numerical_dispersion(self) -> None:
        """Warn when per-axis FDTD phase velocity error at freq_max
        exceeds a threshold (Taflove Ch. 4 dispersion relation).

        For each axis the worst-case phase velocity is:
            v_ph = (omega·dt) / (2·arcsin(nu_i · sin(k·d_i/2)))
        where nu_i = c·dt/d_i, k = 2π/λ, d_i = cell size along axis i.

        Reports the per-axis error so the user sees which axis is under-
        resolved or has Courant mismatch — no arbitrary ratio threshold.
        """
        import warnings as _w

        # Skip host-side min when any profile is a tracer. The dispersion
        # warning is advisory only; mesh-as-design-variable optimisation
        # runs under tracing and the warning cannot fire correctly there.
        if any(
            p is not None and is_tracer(p)
            for p in (self._dx_profile, self._dy_profile, self._dz_profile)
        ):
            return

        dx_nom = self._dx or (C0 / self._freq_max / 20.0)
        d = [dx_nom, dx_nom, dx_nom]
        if self._dx_profile is not None:
            d[0] = float(np.min(self._dx_profile))
        if self._dy_profile is not None:
            d[1] = float(np.min(self._dy_profile))
        if self._dz_profile is not None:
            d[2] = float(np.min(self._dz_profile))

        inv_sq = sum(1.0 / di ** 2 for di in d)
        dt_cfl = 0.99 / (C0 * math.sqrt(inv_sq))
        omega = 2.0 * math.pi * self._freq_max
        k0 = omega / C0

        errors = {}
        sin_wdt2 = math.sin(omega * dt_cfl / 2.0)
        for ax, (name, di) in enumerate(zip("xyz", d)):
            # Taflove Eq. 4.44: v_ph along axis i
            # = omega * d_i / (2 * arcsin(d_i * sin(omega*dt/2) / (c*dt)))
            arg = di * sin_wdt2 / (C0 * dt_cfl)
            if abs(arg) >= 1.0:
                errors[name] = float("inf")
                continue
            v_ph = omega * di / (2.0 * math.asin(arg))
            errors[name] = abs(v_ph - C0) / C0

        max_err = max(errors.values())
        if max_err > 0.02:
            parts = ", ".join(
                f"{name}={err*100:.1f}%" for name, err in errors.items()
            )
            worst = max(errors, key=errors.get)
            _w.warn(
                f"FDTD numerical dispersion at freq_max="
                f"{self._freq_max/1e9:.2f}GHz exceeds 2%: {parts}. "
                f"Worst axis: {worst} (cell {d['xyz'.index(worst)]*1e3:.3f}mm). "
                f"Phase velocity error causes resonance frequency bias. "
                f"Refine the coarse axis or co-refine all axes together "
                f"(Taflove Ch. 4).",
                stacklevel=4,
            )

    def _validate_thin_metal_on_nu_mesh(self) -> None:
        """Warn when a thin PEC sheet sits on a NU axis without symmetric
        neighbouring cells (Meep/OpenEMS require equal dz on both sides of
        a metal plane, else surface currents pick up O(1) error and the
        far-field pattern is corrupted — issue #48).
        """
        import warnings as _w
        profiles = (
            ("x", self._dx_profile),
            ("y", self._dy_profile),
            ("z", self._dz_profile),
        )
        for axis_name, prof in profiles:
            if prof is None:
                continue
            if is_tracer(prof):
                # Tracer profiles can't be host-scanned for edge / ratio
                # checks. The warning is advisory only; correctness is
                # preserved downstream.
                continue
            prof_arr = np.asarray(prof, dtype=np.float64)
            if len(prof_arr) < 3:
                continue
            axis_idx = "xyz".index(axis_name)
            for entry in self._geometry:
                mat = self._resolve_material(entry.material_name)
                if mat.sigma < self._PEC_SIGMA_THRESHOLD:
                    continue
                try:
                    c1, c2 = entry.shape.bounding_box()
                except Exception:
                    continue
                lo, hi = float(c1[axis_idx]), float(c2[axis_idx])
                extent = hi - lo
                min_d = float(prof_arr.min())
                if extent > min_d * 1.5:
                    continue
                # _dz_profile is the user's interior profile (no CPML
                # padding). Geometry coordinates are in interior space
                # starting at 0, so cumsum gives the cell edges directly.
                edges = np.concatenate([[0.0], np.cumsum(prof_arr)])
                mid = 0.5 * (lo + hi)
                k = int(np.searchsorted(edges, mid) - 1)
                if k < 0 or k + 1 >= len(prof_arr) or k - 1 < 0:
                    continue
                dz_here = prof_arr[k]
                dz_above = prof_arr[k + 1]
                dz_below = prof_arr[k - 1]
                # Check ratio both directions — metal-in-coarse-cell
                # next to a fine region is just as bad as the reverse.
                def _ratio(a, b):
                    return max(a, b) / min(a, b)
                ratio_above = _ratio(dz_above, dz_here)
                ratio_below = _ratio(dz_below, dz_here)
                if max(ratio_above, ratio_below) > 1.5:
                    _w.warn(
                        f"Thin PEC '{entry.material_name}' on axis "
                        f"{axis_name} sits in a cell of dz={dz_here*1e3:.3f}"
                        f"mm with asymmetric neighbours "
                        f"(below {dz_below*1e3:.3f}, above "
                        f"{dz_above*1e3:.3f} mm). Meep/OpenEMS require "
                        f"equal cell sizes across a metal plane; "
                        f"radiation pattern may be corrupted (issue #48). "
                        f"Put the metal on a preserved-region boundary "
                        f"or refine the neighbouring cell.",
                        stacklevel=4,
                    )

    def preflight(
        self,
        *,
        strict: bool = False,
        check_ntff: bool = True,
        check_resolution: bool = True,
        check_ad_memory: bool = False,
        n_steps_for_memory: int | None = None,
        available_memory_gb: float | None = None,
    ) -> list[str]:
        """Run all pre-simulation checks and return warnings.

        Parameters
        ----------
        strict : bool
            If True, raise ValueError on the first issue instead of
            collecting warnings.
        check_ntff : bool
            Run inverse-design NTFF checks (PEC overlap hard-error,
            λ/4 near-field gap warning). Default True.
        check_resolution : bool
            Run the tightened resolution check (existing _validate_mesh_quality
            uses per-material thresholds already — this flag kept for
            symmetry and future tightening). Default True.
        check_ad_memory : bool
            Run AD memory estimate and warn if > 85% of available VRAM.
            Requires n_steps_for_memory. Default False (diagnostic only).
        n_steps_for_memory : int or None
            Step count for AD memory sizing. Required when check_ad_memory.
        available_memory_gb : float or None
            Override VRAM detection. If None, best-effort via JAX devices.

        Returns
        -------
        list of str
            Warning messages. Empty if no issues found.
        """
        import warnings
        issues = []

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if check_resolution:
                    self._validate_mesh_quality()
                self._validate_simulation_config()
                if check_ntff:
                    self._validate_ntff_inverse_design()
            except ValueError as e:
                if strict:
                    raise
                issues.append(f"ERROR: {e}")

        for w in caught:
            msg = str(w.message)
            if strict:
                raise ValueError(msg)
            issues.append(msg)

        if check_ad_memory:
            if n_steps_for_memory is None:
                raise ValueError("check_ad_memory=True requires n_steps_for_memory")
            est = self.estimate_ad_memory(
                n_steps_for_memory,
                available_memory_gb=available_memory_gb,
            )
            if est.warning:
                msg = est.warning
                if strict:
                    raise ValueError(msg)
                issues.append(msg)

        if issues:
            for iss in issues:
                print(f"  [PREFLIGHT] {iss}")
        else:
            print("  [PREFLIGHT] All checks passed.")

        return issues

    def _auto_preflight(self, *, skip: bool = False, context: str = "forward") -> None:
        """Emit a UserWarning if preflight finds issues (issue #66).

        Called automatically at the start of ``forward()``, ``optimize()``,
        and ``topology_optimize()`` so users discover physics violations
        (under-resolved mesh, geometry in CPML, probe in PEC, ...) before
        spending minutes of GPU compute. Pass ``skip_preflight=True`` at
        the call site to opt out (tests, already-validated configs).
        """
        if skip:
            return
        try:
            issues = self.preflight(strict=False)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"[{context}] auto-preflight raised {type(exc).__name__}: "
                f"{exc}. Call sim.preflight() manually to investigate.",
                UserWarning, stacklevel=3,
            )
            return
        if not issues:
            return
        import warnings
        body = "\n  - ".join(issues)
        warnings.warn(
            f"[{context}] preflight found {len(issues)} issue(s) - "
            f"pass skip_preflight=True to suppress:\n  - {body}",
            UserWarning, stacklevel=3,
        )

    # ---- Inverse-design preflight extensions (issue #30) ----

    def _validate_ntff_inverse_design(self) -> None:
        """NTFF inverse-design checks: PEC overlap (error) and λ/4 gap (warn).

        CHECK 2: NTFF face plane strictly intersecting a PEC bbox.
        CHECK 3: NTFF face closer than λ/4 to any geometry/source/probe.
        """
        import warnings as _w

        if self._ntff is None:
            return

        corner_lo, corner_hi, freqs = self._ntff
        # face = (axis, sign, coord, tangential bbox: [(lo_a, hi_a), (lo_b, hi_b)])
        faces = []
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            tang = ((corner_lo[other[0]], corner_hi[other[0]]),
                    (corner_lo[other[1]], corner_hi[other[1]]))
            faces.append(("lo", axis, corner_lo[axis], tang))
            faces.append(("hi", axis, corner_hi[axis], tang))

        # CHECK 2: strict PEC intersection
        pec_entries = [e for e in self._geometry if e.material_name == "pec"]
        for side, axis, coord, tang in faces:
            for entry in pec_entries:
                try:
                    c1, c2 = entry.shape.bounding_box()
                except (NotImplementedError, TypeError, AttributeError):
                    continue
                # Strict interior along normal axis
                if not (c1[axis] < coord < c2[axis]):
                    continue
                # Tangential overlap along the other two axes
                other = [a for a in range(3) if a != axis]
                overlap = True
                for idx, (tlo, thi) in zip(other, tang):
                    if c2[idx] <= tlo or c1[idx] >= thi:
                        overlap = False
                        break
                if overlap:
                    raise ValueError(
                        f"NTFF face {'xyz'[axis]}_{side} at {coord*1e3:.2f}mm "
                        f"intersects PEC geometry '{entry.material_name}' "
                        f"(bbox {c1}–{c2}). NTFF box must enclose all radiators "
                        f"with no PEC crossing any face. Shrink or move the NTFF box."
                    )

        # CHECK 3: λ/4 near-field gap to any geometry/source/probe
        if freqs is None:
            return
        try:
            f_max = float(jnp.max(jnp.asarray(freqs)))
        except Exception:
            f_max = float(self._freq_max)
        lam_min = C0 / max(f_max, 1.0)
        gap_thresh = lam_min / 4.0

        # Collect candidate bboxes and point positions
        bboxes: list[tuple[str, tuple, tuple]] = []
        for entry in self._geometry:
            try:
                c1, c2 = entry.shape.bounding_box()
                bboxes.append((entry.material_name, c1, c2))
            except (NotImplementedError, TypeError, AttributeError):
                continue
        points: list[tuple[str, tuple]] = []
        for pe in self._ports:
            points.append(("port/source", tuple(pe.position)))
        for pe in self._probes:
            points.append(("probe", tuple(pe.position)))

        for side, axis, coord, tang in faces:
            other = [a for a in range(3) if a != axis]
            min_gap = float("inf")
            culprit = None
            # bbox distances
            for name, c1, c2 in bboxes:
                # tangential overlap check — only meaningful gap if the face
                # is "above" the feature in the normal direction
                overlap = True
                for idx, (tlo, thi) in zip(other, tang):
                    if c2[idx] <= tlo or c1[idx] >= thi:
                        overlap = False
                        break
                if not overlap:
                    continue
                if coord <= c1[axis]:
                    d = c1[axis] - coord
                elif coord >= c2[axis]:
                    d = coord - c2[axis]
                else:
                    d = 0.0  # already handled by CHECK 2 for PEC; skip
                    continue
                if d < min_gap:
                    min_gap, culprit = d, f"geometry '{name}'"
            # points
            for name, pos in points:
                # require tangential in-box for relevance
                in_tang = all(
                    tang[i][0] <= pos[other[i]] <= tang[i][1] for i in range(2)
                )
                if not in_tang:
                    continue
                d = abs(coord - pos[axis])
                if d < min_gap:
                    min_gap, culprit = d, f"{name} at {pos}"

            if culprit is not None and min_gap < gap_thresh:
                _w.warn(
                    f"NTFF face {'xyz'[axis]}_{side} is {min_gap*1e3:.2f}mm "
                    f"from {culprit} — below λ/4 = {gap_thresh*1e3:.2f}mm at "
                    f"f_max={f_max/1e9:.2f}GHz. Evanescent near-field may "
                    f"contaminate the far-field pattern. Move NTFF box further "
                    f"from radiating/scattering structures.",
                    stacklevel=3,
                )

    # ---- AD memory estimation (issue #30 CHECK 4) ----

    def estimate_ad_memory(
        self,
        n_steps: int,
        *,
        available_memory_gb: float | None = None,
        checkpoint_every: int | None = None,
    ) -> "AD_MemoryEstimate":
        """Estimate reverse-mode AD memory for this simulation.

        Returns an AD_MemoryEstimate with forward, checkpointed-AD, and
        non-checkpointed-AD sizes in GB, plus a best-effort warning if
        estimated AD memory exceeds 85% of available VRAM.

        When ``checkpoint_every`` is provided, the returned
        ``ad_segmented_gb`` reflects the segmented scan-of-scan path
        from issue #31. The legacy ``ad_checkpointed_gb`` field keeps
        its old (optimistic) heuristic for backwards compatibility; it
        is NOT accurate for FDTD on the non-uniform path — see the
        class docstring.
        """
        dx = self._dx or (C0 / self._freq_max / 20.0)

        def _nx(extent: float, prof) -> int:
            if prof is not None:
                return len(prof) + 1 + 2 * self._cpml_layers
            return int(math.ceil(extent / dx)) + 1 + 2 * self._cpml_layers

        nx = _nx(self._domain[0], self._dx_profile)
        ny = _nx(self._domain[1], self._dy_profile)
        nz = _nx(self._domain[2], self._dz_profile)
        cells = nx * ny * nz

        # Forward working set: 6 field + ~6 material + ~4 CPML psi (~15%)
        bytes_per_cell = 4  # float32
        field_bytes = cells * 6 * bytes_per_cell
        mat_bytes = cells * 6 * bytes_per_cell
        cpml_bytes = int(cells * 0.15 * 24 * bytes_per_cell) if self._cpml_layers > 0 else 0
        forward_bytes = field_bytes + mat_bytes + cpml_bytes

        # NTFF DFT state: 6 faces × n_freqs × face_cells × 4 (3 E + 3 H) × complex64 (8B)
        ntff_bytes = 0
        if self._ntff is not None:
            _, _, freqs = self._ntff
            n_freqs = int(len(freqs)) if freqs is not None else 10
            # Approx face cells from domain extents / dx
            face_est = 2 * ((nx * ny) + (ny * nz) + (nx * nz))
            ntff_bytes = face_est * n_freqs * 6 * 8

        # Legacy "checkpointed" estimate: remat-recomputes internals of
        # step_fn. NOT valid for the NU path after issue #31 because the
        # scan carry itself is not rematerialised.
        ad_ckpt_bytes = 4 * forward_bytes + ntff_bytes
        # Non-checkpointed AD: O(n_steps) tape — 6 field arrays per step
        ad_full_bytes = n_steps * field_bytes + ntff_bytes + forward_bytes

        # Segmented scan-of-scan (issue #31). Outer scan wraps
        # jax.checkpoint around an inner scan of length K; the tape
        # stores carry + cotangent at (n_steps/K) segment boundaries.
        # Formula fit to VESSL 369367233490 data on 608k cells within
        # ~15% (see issue #39).
        ad_seg_bytes: int | None = None
        if checkpoint_every is not None and checkpoint_every > 0:
            n_segments = math.ceil(n_steps / checkpoint_every)
            # 2x per segment: primal carry + cotangent stored for backward
            ad_seg_bytes = 2 * n_segments * field_bytes + forward_bytes + ntff_bytes

        # VRAM detection (best effort)
        avail_gb = available_memory_gb
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
            warning = (
                f"AD memory estimate {primary_bytes*to_gb:.2f}GB ({label}) "
                f"exceeds 85% of {avail_gb:.2f}GB available VRAM. "
                f"Reduce grid size, reduce n_steps, or lower checkpoint_every."
            )
        return AD_MemoryEstimate(
            forward_gb=forward_bytes * to_gb,
            ad_checkpointed_gb=ad_ckpt_bytes * to_gb,
            ad_full_gb=ad_full_bytes * to_gb,
            ntff_dft_gb=ntff_bytes * to_gb,
            available_gb=avail_gb,
            warning=warning,
            ad_segmented_gb=(ad_seg_bytes * to_gb) if ad_seg_bytes is not None else None,
            checkpoint_every=checkpoint_every,
        )

    def _validate_simulation_config(self) -> None:
        """Comprehensive pre-simulation configuration validation.

        Checks for common setup mistakes that produce silent wrong results:
        probe/source in CPML, boundary type mismatch, feature compatibility,
        NTFF precision, normalize defaults.

        Called from run() after _validate_mesh_quality().
        """
        import warnings as _w

        dx = self._dx or C0 / self._freq_max / 20.0
        cpml_thickness = self._cpml_layers * dx if self._boundary in ("cpml", "upml") else 0

        # Warn about pec_faces + finite PEC objects co-existing.
        # pec_faces creates an INFINITE PEC boundary face across the whole
        # domain side. Users building antennas or finite-GP structures
        # often use pec_faces thinking it's a "ground plane" — but it's
        # a full-domain boundary condition, not a finite structure.
        if self._pec_faces and self._geometry:
            has_finite_pec = any(
                entry.material_name == "pec"
                for entry in self._geometry
            )
            if has_finite_pec:
                pec_face_list = ", ".join(sorted(self._pec_faces))
                _w.warn(
                    f"pec_faces={{{pec_face_list}}} creates an INFINITE PEC "
                    f"boundary AND the geometry contains finite PEC objects. "
                    f"For antennas or finite-GP structures, the pec_faces "
                    f"boundary makes the ground plane cover the entire domain "
                    f"face, which changes the physics (cavity vs radiating "
                    f"antenna). If you need a finite ground plane, remove "
                    f"pec_faces and use an explicit PEC Box instead.",
                    stacklevel=3,
                )

        if self._boundary == "upml" and self._refinement is not None:
            raise ValueError("boundary='upml' does not support subgridding/refinement")

        # Per-face CPML thickness (v1.7.5). Mirrors Grid._face_pad:
        # pec_faces / pmc_faces / periodic-axis faces consume 0 cells;
        # remaining faces get the axis CPML thickness (non-uniform z
        # aggregates the leading dz_profile entries). Under asymmetric
        # composition (half-symmetric PMC + CPML, one-sided reflector)
        # the lo and hi sides of a single axis can differ — the legacy
        # symmetric scalar forced both sides to the max and produced
        # false positives on the reflector face.
        _pmc_faces_set = set(self._boundary_spec.pmc_faces())
        _axis_thickness = [cpml_thickness, cpml_thickness, cpml_thickness]
        if (self._dz_profile is not None
                and not is_tracer(self._dz_profile)
                and self._boundary in ("cpml", "upml")
                and self._cpml_layers > 0):
            n = min(self._cpml_layers, len(self._dz_profile))
            _axis_thickness[2] = float(sum(self._dz_profile[:n]))

        def _face_thickness(ax_idx: int, side: str) -> float:
            ax_name = "xyz"[ax_idx]
            face = f"{ax_name}_{side}"
            if self._boundary not in ("cpml", "upml"):
                return 0.0
            if face in self._pec_faces or face in _pmc_faces_set:
                return 0.0
            if ax_name in self._periodic_axes:
                return 0.0
            return _axis_thickness[ax_idx]

        cpml_thick_lo = [_face_thickness(ax, "lo") for ax in range(3)]
        cpml_thick_hi = [_face_thickness(ax, "hi") for ax in range(3)]
        # Legacy symmetric scalar kept for readers that treat it as
        # "nominal CPML thickness on this axis"; per-side checks below
        # use cpml_thick_lo / cpml_thick_hi.
        cpml_thick_xyz = [
            max(cpml_thick_lo[i], cpml_thick_hi[i]) for i in range(3)
        ]

        # P1.1: Floquet + non-uniform mesh — no silent fallback allowed
        if self._floquet_ports and self._dz_profile is not None:
            raise ValueError(
                "Floquet ports do not support non-uniform z mesh (dz_profile). "
                "Use the uniform reference lane and set dx explicitly."
            )

        # P1.2/P1.3: Probe or source inside absorber region
        absorber_label = "UPML" if self._boundary == "upml" else "CPML"
        if cpml_thickness > 0:
            for pe in self._probes:
                pos = pe.position
                for ax, coord in enumerate(pos):
                    domain_extent = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                    ax_i = min(ax, 2)
                    ct_lo = cpml_thick_lo[ax_i]
                    ct_hi = cpml_thick_hi[ax_i]
                    if coord < ct_lo * 0.5 or coord > domain_extent - ct_hi * 0.5:
                        _w.warn(
                            f"Probe at {pos} is near/inside {absorber_label} region "
                            f"({absorber_label} {'xyz'[ax]}-thickness: "
                            f"lo={ct_lo*1e3:.1f}mm, hi={ct_hi*1e3:.1f}mm). "
                            f"Signal will be attenuated. Move probe to interior.",
                            stacklevel=3,
                        )
                        break

            for pe in self._ports:
                pos = pe.position
                for ax, coord in enumerate(pos):
                    domain_extent = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                    ax_i = min(ax, 2)
                    ct_lo = cpml_thick_lo[ax_i]
                    ct_hi = cpml_thick_hi[ax_i]
                    if coord < ct_lo * 0.5 or coord > domain_extent - ct_hi * 0.5:
                        _w.warn(
                            f"Source/port at {pos} is near/inside {absorber_label} region "
                            f"({absorber_label} {'xyz'[ax]}-thickness: "
                            f"lo={ct_lo*1e3:.1f}mm, hi={ct_hi*1e3:.1f}mm). "
                            f"Energy will be absorbed. Move source to interior.",
                            stacklevel=3,
                        )
                        break

        # P1.6: Source / port placed ON a PEC or PMC face plane. Both
        # reflectors zero specific field components at the plane every
        # time step (PEC: tangential E; PMC: tangential H); a source
        # that drives a zeroed component is silently discarded. A
        # source that drives a component forced to zero by the mirror
        # image (e.g. normal E on a PMC face) fights the symmetry and
        # yields numerically inconsistent results.
        #
        # Component-specific rule:
        #   PEC face (axis = ax_name): tangential E (Ex/Ey/Ez with
        #     component axis != ax_name) is zeroed every E update.
        #     Normal E (component axis == ax_name) is the legitimate
        #     way to drive a PEC mirror.
        #   PMC face (axis = ax_name): tangential H (Hx/Hy/Hz with
        #     component axis != ax_name) is zeroed; the outgoing
        #     wave from an on-plane tangential E source is killed via
        #     this H zeroing. Normal E (component axis == ax_name) is
        #     odd-symmetric and must be zero at the plane by image,
        #     so injecting it fights the mirror.
        #
        # See docs/research_notes/2026-04-20_source_on_symmetry_plane_industry_survey.md
        # for the industry survey behind this rule (Meep / OpenEMS /
        # Tidy3D all follow the same convention).
        _all_reflector_faces = set(self._pec_faces) | set(_pmc_faces_set)
        if _all_reflector_faces:
            _dx_axis = [float(dx), float(dx), float(dx)]
            if (self._dz_profile is not None
                    and not is_tracer(self._dz_profile)):
                _dx_axis[2] = float(self._dz_profile[0])
            for face in _all_reflector_faces:
                ax_name = face[0]
                side = face[2:]
                ax_i = "xyz".index(ax_name)
                face_kind = "PMC" if face in _pmc_faces_set else "PEC"
                d_ext = self._domain[ax_i] if ax_i < len(self._domain) else self._domain[-1]
                plane_coord = 0.0 if side == "lo" else float(d_ext)
                tol = 0.5 * _dx_axis[ax_i]
                for pe in self._ports:
                    pos = pe.position
                    coord = pos[ax_i]
                    if abs(coord - plane_coord) > tol:
                        continue
                    # Classify the source component vs. the face axis.
                    comp = pe.component.lower()
                    comp_field = comp[0]       # 'e' or 'h'
                    comp_axis = comp[1:]       # 'x' / 'y' / 'z'
                    is_tangential = (comp_axis != ax_name)
                    if face_kind == "PMC":
                        if comp_field == "e" and is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PMC {face} plane. The outgoing "
                                f"tangential H is zeroed every step by "
                                f"apply_pmc_faces, so no wave radiates — the "
                                f"probe records silent zero field. Offset by "
                                f"one cell ({_dx_axis[ax_i]*1e3:.3g} mm) off "
                                f"the plane to let the Yee curl run normally."
                            )
                        elif comp_field == "e" and not is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PMC {face} plane and drives the "
                                f"NORMAL E component. PMC imposes odd symmetry "
                                f"on normal E (it must be zero at the plane), "
                                f"so the source fights the mirror image. Use a "
                                f"tangential E source offset by one cell "
                                f"({_dx_axis[ax_i]*1e3:.3g} mm) off the plane."
                            )
                        elif comp_field == "h" and is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PMC {face} plane and drives a "
                                f"tangential H. apply_pmc_faces zeros this "
                                f"component at the plane every step, so the "
                                f"source has no effect."
                            )
                        else:
                            msg = None      # normal H on PMC plane is legit
                    else:                    # PEC
                        if comp_field == "e" and is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PEC {face} plane and drives a "
                                f"tangential E. PEC zeros E_tan at the plane "
                                f"every step, so the source is silently "
                                f"discarded. Use a normal E source at this "
                                f"face, or offset by one cell "
                                f"({_dx_axis[ax_i]*1e3:.3g} mm) off the plane."
                            )
                        elif comp_field == "h" and not is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PEC {face} plane and drives the "
                                f"NORMAL H component. PEC imposes odd symmetry "
                                f"on normal H (it must be zero at the plane). "
                                f"Use a tangential H source or offset by one "
                                f"cell ({_dx_axis[ax_i]*1e3:.3g} mm) off the plane."
                            )
                        else:
                            msg = None      # tangential H or normal E on PEC is legit
                    if msg is not None:
                        _w.warn(msg, stacklevel=3)

        # P1.4: NTFF box overlap with absorber
        if self._ntff is not None and cpml_thickness > 0:
            corner_lo, corner_hi, _ = self._ntff
            for ax in range(3):
                domain_ext = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                ax_i = min(ax, 2)
                ct_lo = cpml_thick_lo[ax_i]
                ct_hi = cpml_thick_hi[ax_i]
                if corner_lo[ax] < ct_lo or corner_hi[ax] > domain_ext - ct_hi:
                    _w.warn(
                        f"NTFF box extends into {absorber_label} region along "
                        f"{'xyz'[ax]}-axis. Far-field results will be "
                        f"corrupted. Shrink NTFF box to interior.",
                        stacklevel=3,
                    )
                    break

        # P1.5: (merged into P2.1 — non-uniform + NTFF is unsupported)

        # P1.7: NTFF with too few steps
        if self._ntff is not None:
            _, _, ntff_freqs = self._ntff
            if ntff_freqs is not None:
                min_freq = float(min(ntff_freqs))
                period = 1.0 / max(min_freq, 1.0)
                dt_est = dx / (C0 * 1.732) * 0.99  # CFL estimate
                min_steps_for_ntff = int(10 * period / dt_est)
                # Can't check n_steps here (not known yet), but store hint
                self._ntff_min_steps_hint = min_steps_for_ntff

        # P1.9: Geometry (dielectric OR PEC) extending into CPML region.
        # CPML modifies field-update equations with absorbing coefficients;
        # any structure placed there is effectively eaten by the absorber
        # and produces physically meaningless results (issue #61).
        # Periodic axes have no CPML (see _build_grid — issue #68), so
        # the per-axis thresholds above already carry `cpml_thick_xyz[ax]
        # == 0` on those axes and the check naturally skips.
        if cpml_thickness > 0 and self._boundary == "cpml":
            for entry in self._geometry:
                if hasattr(entry.shape, "bounding_box"):
                    try:
                        c1, c2 = entry.shape.bounding_box()
                        for ax in range(min(3, len(self._domain))):
                            thick_lo = cpml_thick_lo[ax]
                            thick_hi = cpml_thick_hi[ax]
                            if thick_lo <= 0 and thick_hi <= 0:
                                continue
                            d = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                            lo_hit = thick_lo > 0 and c1[ax] < thick_lo * 0.3
                            hi_hit = thick_hi > 0 and c2[ax] > d - thick_hi * 0.3
                            if lo_hit or hi_hit:
                                _w.warn(
                                    f"Material '{entry.material_name}' extends "
                                    f"into CPML region along {'xyz'[ax]}-axis. "
                                    f"{absorber_label} modifies field updates — "
                                    f"geometry inside the absorber is physically "
                                    f"meaningless (issue #61).",
                                    stacklevel=3,
                                )
                                break
                    except (NotImplementedError, TypeError):
                        pass

        # P1.8: Port/source/probe inside PEC geometry
        for pe in list(self._ports) + list(self._probes):
            pos = pe.position
            for entry in self._geometry:
                if entry.material_name != "pec":
                    continue
                if hasattr(entry.shape, "bounding_box"):
                    try:
                        c1, c2 = entry.shape.bounding_box()
                        inside = all(c1[ax] <= pos[ax] <= c2[ax] for ax in range(3))
                        if inside:
                            _w.warn(
                                f"Port/source at {pos} is inside PEC geometry "
                                f"'{entry.material_name}'. Field will be zero. "
                                f"Move source outside PEC.",
                                stacklevel=3,
                            )
                    except (NotImplementedError, TypeError):
                        pass

        # P1.9: Single-cell port in dielectric with no adjacent PEC pin
        # (issue #71). A single-cell LumpedPort placed mid-substrate with
        # no conducting pin or microstrip does not couple to patch-antenna
        # TM modes — the optimiser reads a nonsense loss from the
        # floating Ez source. Recommend extent=<substrate_height> to
        # promote to a WirePort spanning ground → patch.
        _PORT_COMP_AXIS = {"ex": 0, "ey": 1, "ez": 2}
        for pe in self._ports:
            # Filter: only true ports (impedance > 0), single-cell
            # (extent is None), actively excited (excite is True).
            # add_source() creates _PortEntry with impedance=0.0 and is
            # intentionally a soft source — not a port footgun.
            if not pe.impedance or pe.impedance <= 0.0:
                continue
            if pe.extent is not None:
                continue
            if not pe.excite:
                continue
            pos = pe.position
            # Find the dielectric geometry enclosing the port cell.
            enclosing_eps_r = None
            enclosing_name = None
            for entry in self._geometry:
                if entry.material_name == "pec":
                    continue
                if not hasattr(entry.shape, "bounding_box"):
                    continue
                try:
                    c1, c2 = entry.shape.bounding_box()
                except (NotImplementedError, TypeError):
                    continue
                inside = all(c1[ax] <= pos[ax] <= c2[ax] for ax in range(3))
                if not inside:
                    continue
                mspec = self._materials.get(entry.material_name)
                if mspec is not None and float(mspec.eps_r) > 1.0 + 1e-3:
                    enclosing_eps_r = float(mspec.eps_r)
                    enclosing_name = entry.material_name
                    break
            if enclosing_eps_r is None:
                continue
            # Check for a PEC geometry one cell away along the port's
            # component axis (coax-style pin or microstrip feed edge).
            # Without such a pin, the port cell cannot drive a vertical
            # current that couples to the patch TM mode.
            comp_axis = _PORT_COMP_AXIS.get(pe.component)
            if comp_axis is None:
                continue
            nudge = float(self._dx or 0.0) * 1.01
            adj_positions = (
                tuple(pos[i] + (nudge if i == comp_axis else 0.0) for i in range(3)),
                tuple(pos[i] - (nudge if i == comp_axis else 0.0) for i in range(3)),
            )
            has_adjacent_pec = False
            for apos in adj_positions:
                for entry in self._geometry:
                    if entry.material_name != "pec":
                        continue
                    if not hasattr(entry.shape, "bounding_box"):
                        continue
                    try:
                        c1, c2 = entry.shape.bounding_box()
                    except (NotImplementedError, TypeError):
                        continue
                    if all(c1[ax] <= apos[ax] <= c2[ax] for ax in range(3)):
                        has_adjacent_pec = True
                        break
                if has_adjacent_pec:
                    break
            if has_adjacent_pec:
                continue
            _w.warn(
                f"Single-cell port at {pos} ({pe.component}) sits inside "
                f"dielectric '{enclosing_name}' (eps_r={enclosing_eps_r:.2f}) "
                f"with no adjacent PEC along the {pe.component[1]}-axis. A "
                f"floating single-cell port inside substrate does not "
                f"couple to patch-antenna TM modes. Pass "
                f"extent=<substrate_height> to create a WirePort spanning "
                f"ground → patch plane (issue #71).",
                stacklevel=3,
            )

        # P0.4: PEC boundary on likely open structure
        if self._boundary == "pec" and self._ntff is not None:
            _w.warn(
                "PEC boundary with NTFF far-field: PEC reflects all energy "
                "back into domain. Use boundary='cpml' or boundary='upml' for open structures "
                "(antennas, scatterers).",
                stacklevel=3,
            )

        # P0.5: No sources configured
        if not self._ports and self._tfsf is None and not self._waveguide_ports and not self._floquet_ports:
            _w.warn(
                "No sources, ports, TFSF, or waveguide/Floquet ports configured. "
                "Simulation will produce zero fields.",
                stacklevel=3,
            )

        # ================================================================
        # P2: Non-uniform mesh shadow-lane limitations
        # ================================================================
        if self._dz_profile is not None:
            # P2.3: TFSF on nonuniform mesh — narrowed scope.
            # Axis-aligned ±x incidence with angle_deg=0 runs the 1D
            # auxiliary along the uniform x axis and is supported. The
            # z-directed and oblique cases would need a z-nonuniform 1D
            # aux (resp. nonuniform 2D aux) and are deferred.
            if self._tfsf is not None:
                if self._tfsf.direction in ("+z", "-z"):
                    raise ValueError(
                        "TFSF z-directed incidence is not yet supported on "
                        "nonuniform z mesh. Axis-aligned incidence along x "
                        "(direction='+x' or '-x') is supported."
                    )
                if abs(self._tfsf.angle_deg) > 0.01:
                    raise ValueError(
                        "TFSF oblique incidence is not yet supported on "
                        "nonuniform z mesh. Use angle_deg=0."
                    )

            # P2.6: CPML z-thickness on non-uniform mesh.
            # Skip on tracer profiles — advisory warning only.
            if (self._boundary == "cpml"
                    and self._cpml_layers > 0
                    and not is_tracer(self._dz_profile)):
                cpml_z_thick = sum(float(d) for d in self._dz_profile[:self._cpml_layers])
                if cpml_z_thick < cpml_thickness * 0.3:
                    _w.warn(
                        f"CPML z-thickness is {cpml_z_thick*1e3:.1f}mm "
                        f"({self._cpml_layers} cells), much thinner than "
                        f"xy-thickness {cpml_thickness*1e3:.1f}mm. "
                        f"Absorbing performance may be asymmetric. "
                        f"Consider more z cells or fewer CPML layers.",
                        stacklevel=3,
                    )

        # ================================================================
        # P3: Distributed path limitations
        # ================================================================
        # (Distributed warnings are emitted at run() dispatch time in
        #  distributed_v2.py, but we add preflight hints here too.)

        # ================================================================
        # P4: Subgridded path limitations
        # ================================================================
        if self._refinement is not None:
            if self._ntff is not None:
                _w.warn(
                    "NTFF far-field is not supported with SBP-SAT subgridding. "
                    "The NTFF box will be ignored.",
                    stacklevel=3,
                )
            if self._dft_planes:
                _w.warn(
                    "DFT plane probes are not supported with SBP-SAT "
                    "subgridding.",
                    stacklevel=3,
                )
            if self._waveguide_ports:
                _w.warn(
                    "Waveguide ports are not supported with SBP-SAT "
                    "subgridding.",
                    stacklevel=3,
                )
            if self._floquet_ports:
                _w.warn(
                    "Floquet ports are not supported with SBP-SAT subgridding.",
                    stacklevel=3,
                )
            if self._tfsf is not None:
                _w.warn(
                    "TFSF source is not supported with SBP-SAT subgridding.",
                    stacklevel=3,
                )
            if self._lumped_rlc:
                _w.warn(
                    "Lumped RLC elements are not supported with SBP-SAT "
                    "subgridding.",
                    stacklevel=3,
                )

        # P2.7 (obsolete): PMC / PEC + CPML on the same axis used to emit
        # a warning for the architectural offset between the reflector
        # plane and the user domain edge. v1.7.5 closed that gap on both
        # the uniform (rfx/grid.py) and non-uniform (rfx/nonuniform.py)
        # paths via per-face ``pad_{axis}_{lo,hi}`` allocation. The
        # warning is retained as a no-op anchor so external references
        # ("[P2.7]") don't break and as a reminder that the fix is
        # regression-locked via tests/test_silent_drop_warnings.py and
        # tests/test_boundary_pmc_hi_faces.py.

        # P2.8: Waveguide-port reference plane sanity.
        # The S-matrix returned by ``compute_waveguide_s_matrix`` is
        # evaluated AT the reference plane (either ``entry.reference_plane``
        # if user-specified, or the port's ``x_position`` by default after
        # 2026-04-22). The phase of reported S-params is therefore tied to
        # that plane. Physical correctness requires the plane lies inside
        # the simulation domain, outside the CPML absorbing region, and
        # preferably inside a uniform-cross-section segment of guide so the
        # modal decomposition is defined.
        if self._waveguide_ports:
            axis_map = {"x": 0, "y": 1, "z": 2}
            for entry in self._waveguide_ports:
                direction = entry.direction  # e.g., "+x", "-x"
                ax_i = axis_map[direction[-1]]
                domain_ext = self._domain[ax_i]
                ct_lo = cpml_thick_lo[ax_i]
                ct_hi = cpml_thick_hi[ax_i]
                effective = (entry.reference_plane if entry.reference_plane is not None
                             else entry.x_position)
                if effective < 0 or effective > domain_ext:
                    raise ValueError(
                        f"waveguide_port reference plane = {effective:.4g} m is "
                        f"outside the {direction[-1]}-domain [0, {domain_ext:.4g}] m. "
                        f"Check x_position / reference_plane."
                    )
                if effective < ct_lo or effective > domain_ext - ct_hi:
                    _w.warn(
                        f"waveguide_port reference plane = {effective*1e3:.3g} mm is "
                        f"inside the CPML absorbing region along the "
                        f"{direction[-1]}-axis (CPML extent: "
                        f"[0, {ct_lo*1e3:.3g}] and "
                        f"[{(domain_ext - ct_hi)*1e3:.3g}, {domain_ext*1e3:.3g}] mm). "
                        f"S-matrix phase will be distorted by CPML stretching. "
                        f"Move x_position / reference_plane to the interior or "
                        f"reduce cpml_layers.",
                        stacklevel=3,
                    )
                # Device overlap warning: check if any geometry box spans
                # the port's x-plane.
                if self._geometry:
                    for g in self._geometry:
                        try:
                            lo, hi = g.bounds
                        except Exception:
                            continue
                        if lo[ax_i] <= effective <= hi[ax_i]:
                            _w.warn(
                                f"waveguide_port reference plane at "
                                f"{effective*1e3:.3g} mm intersects geometry "
                                f"'{getattr(g, 'material', '?')}' "
                                f"(bounds {lo[ax_i]*1e3:.3g}–{hi[ax_i]*1e3:.3g} mm "
                                f"on {direction[-1]}). Modal decomposition "
                                f"assumes a uniform cross-section at the port "
                                f"plane; reported S-params will mix modes. Move "
                                f"the reference plane into the empty-guide region.",
                                stacklevel=3,
                            )
                            break

    def _validate_adi_configuration(self, materials: MaterialArrays, debye_spec, lorentz_spec) -> None:
        """Validate that the current simulation is compatible with the ADI path."""
        if self._mode not in ("2d_tmz", "3d"):
            raise ValueError("solver='adi' supports mode='3d' or mode='2d_tmz'")
        if self._boundary == "upml":
            raise ValueError("solver='adi' does not support boundary='upml'")
        if self._boundary not in ("pec", "cpml"):
            raise ValueError("solver='adi' supports boundary='pec' or 'cpml'")
        if self._refinement is not None:
            raise ValueError("solver='adi' does not support subgridding yet")
        if self._tfsf is not None:
            raise ValueError("solver='adi' does not support TFSF sources yet")
        if self._waveguide_ports or self._floquet_ports:
            raise ValueError("solver='adi' does not support waveguide or Floquet ports yet")
        if self._periodic_axes:
            raise ValueError("solver='adi' does not support manual periodic axes yet")
        if self._dft_planes:
            raise ValueError("solver='adi' does not support DFT plane probes yet")
        if self._ntff is not None:
            raise ValueError("solver='adi' does not support NTFF accumulation yet")
        if self._coaxial_ports:
            raise ValueError("solver='adi' does not support coaxial ports yet")
        if self._lumped_rlc:
            raise ValueError("solver='adi' does not support lumped RLC elements yet")
        if self._thin_conductors:
            raise ValueError("solver='adi' does not support thin-conductor corrections yet")
        if debye_spec is not None or lorentz_spec is not None:
            raise ValueError("solver='adi' does not support dispersive materials yet")
        # Conductivity is now supported: implicit sigma in ADI tridiagonal.
        # Internal absorbing layers also use sigma, so no restriction needed.
        for pe in self._ports:
            if pe.impedance != 0.0 or pe.extent is not None:
                raise ValueError("solver='adi' currently supports only add_source()-style soft sources")
            if self._mode == "2d_tmz" and pe.component != "ez":
                raise ValueError("solver='adi' in 2D TMz mode supports only Ez soft sources")
        _valid_adi_probes = {"ez", "hx", "hy"} if self._mode == "2d_tmz" else {"ex", "ey", "ez", "hx", "hy", "hz"}
        for probe in self._probes:
            if probe.component not in _valid_adi_probes:
                raise ValueError(f"solver='adi' supports probes on {_valid_adi_probes} only")

    def _run_adi_from_materials(
        self,
        grid: Grid,
        materials: MaterialArrays,
        debye_spec,
        lorentz_spec,
        *,
        n_steps: int,
        pec_mask: jnp.ndarray | None = None,
        return_state: bool = True,
    ):
        """Run the integrated ADI solver path (2D TMz or 3D)."""
        import copy

        self._validate_adi_configuration(materials, debye_spec, lorentz_spec)

        dt = float(grid.dt * self._adi_cfl_factor)
        times = jnp.arange(n_steps, dtype=jnp.float32) * dt

        grid_out = copy.copy(grid)
        grid_out.dt = dt

        # ---- 3D path ----
        if self._mode == "3d":
            from rfx.adi import run_adi_3d, ADIState3D, make_adi_absorbing_sigma_3d

            sources_3d = []
            for pe in self._ports:
                i, j, k = grid.position_to_index(pe.position)
                waveform = jax.vmap(pe.waveform)(times)
                sources_3d.append((i, j, k, pe.component, waveform))

            probes_3d = []
            for pe in self._probes:
                i, j, k = grid.position_to_index(pe.position)
                probes_3d.append((i, j, k, pe.component))

            eps_r_3d = materials.eps_r
            sigma_3d = materials.sigma

            if self._boundary == "cpml" and self._cpml_layers > 0:
                nx, ny, nz = grid.shape
                absorb_sigma = make_adi_absorbing_sigma_3d(
                    nx, ny, nz, self._cpml_layers, grid.dx, grid.dx, grid.dx)
                sigma_3d = sigma_3d + absorb_sigma

            shape = grid.shape
            zeros = jnp.zeros(shape, dtype=jnp.float32)
            ex_f, ey_f, ez_f, hx_f, hy_f, hz_f, probe_data = run_adi_3d(
                zeros, zeros, zeros, zeros, zeros, zeros,
                eps_r_3d, sigma_3d,
                dt, grid.dx, grid.dx, grid.dx,
                n_steps,
                sources=sources_3d,
                probes=probes_3d,
                pec_mask=pec_mask,
            )
            if probe_data is None:
                probe_data = jnp.zeros((n_steps, 0), dtype=jnp.float32)

            if return_state:
                state = ADIState3D(
                    ex=ex_f, ey=ey_f, ez=ez_f,
                    hx=hx_f, hy=hy_f, hz=hz_f,
                    step=jnp.asarray(n_steps, dtype=jnp.int32),
                )
                return Result(
                    state=state,
                    time_series=probe_data,
                    s_params=None, freqs=None,
                    grid=grid_out, dt=dt,
                    freq_range=(self._freq_max / 10, self._freq_max, self._boundary),
                )
            return ForwardResult(
                time_series=probe_data,
                ntff_data=None, ntff_box=None,
                grid=grid_out,
            )

        # ---- 2D TMz path ----
        sources = []
        for pe in self._ports:
            i, j, _ = grid.position_to_index(pe.position)
            waveform = jax.vmap(pe.waveform)(times)
            sources.append((i, j, waveform))

        probes = []
        for pe in self._probes:
            i, j, _ = grid.position_to_index(pe.position)
            probes.append((i, j, pe.component))

        eps_r_2d = materials.eps_r[:, :, 0]
        sigma_2d = materials.sigma[:, :, 0]

        # Add implicit absorbing sigma layer for CPML boundary
        if self._boundary == "cpml" and self._cpml_layers > 0:
            from rfx.adi import make_adi_absorbing_sigma
            nx_2d, ny_2d = eps_r_2d.shape
            absorb_sigma = make_adi_absorbing_sigma(
                nx_2d, ny_2d, self._cpml_layers, grid.dx)
            sigma_2d = sigma_2d + absorb_sigma

        pec_mask_2d = None
        if pec_mask is not None:
            pec_mask_2d = pec_mask[:, :, 0]
        ez0 = jnp.zeros_like(eps_r_2d)
        hx0 = jnp.zeros_like(eps_r_2d)
        hy0 = jnp.zeros_like(eps_r_2d)
        ez_f, hx_f, hy_f, probe_data = run_adi_2d(
            ez0,
            hx0,
            hy0,
            eps_r_2d,
            sigma_2d,
            dt,
            grid.dx,
            grid.dx,
            n_steps,
            sources=sources,
            probes=probes,
            pec_mask=pec_mask_2d,
        )
        if probe_data is None:
            probe_data = jnp.zeros((n_steps, 0), dtype=ez_f.dtype)

        if return_state:
            state = ADIState2D(
                ez=ez_f,
                hx=hx_f,
                hy=hy_f,
                step=jnp.asarray(n_steps, dtype=jnp.int32),
            )
            return Result(
                state=state,
                time_series=probe_data,
                s_params=None,
                freqs=None,
                grid=grid_out,
                dt=dt,
                freq_range=(self._freq_max / 10, self._freq_max, self._boundary),
            )

        return ForwardResult(
            time_series=probe_data,
            ntff_data=None,
            ntff_box=None,
            grid=grid_out,
        )

    def _forward_from_materials(
        self,
        grid: Grid,
        materials: MaterialArrays,
        debye_spec: tuple | None,
        lorentz_spec: tuple | None,
        *,
        n_steps: int,
        checkpoint: bool = True,
        checkpoint_segments: int | None = None,
        pec_mask: jnp.ndarray | None = None,
        pec_occupancy: jnp.ndarray | None = None,
        port_s11_freqs: object | None = None,
    ) -> ForwardResult:
        """Run a minimal differentiable forward path from explicit materials."""
        if self._solver == "adi":
            return self._run_adi_from_materials(
                grid,
                materials,
                debye_spec,
                lorentz_spec,
                n_steps=n_steps,
                pec_mask=pec_mask,
                return_state=False,
            )

        from rfx.simulation import (
            run as _run,
            make_source,
            make_probe,
            make_port_source,
            make_wire_port_sources,
            LumpedPortSParamSpec,
        )
        from rfx.sources.sources import (
            LumpedPort,
            WirePort,
            setup_lumped_port,
            setup_wire_port,
            _wire_port_cells,
        )

        sources = []
        probes = []
        pec_mask_local = pec_mask
        pec_occupancy_local = pec_occupancy
        lumped_port_sparam_specs: list = []
        # Resolve a freq array once for downstream auto-build (issue #72)
        if port_s11_freqs is not None:
            _s11_freqs_arr = jnp.asarray(port_s11_freqs, dtype=jnp.float32)
        else:
            _s11_freqs_arr = None

        for pe in self._ports:
            if pe.impedance == 0.0:
                from rfx.simulation import make_j_source
                sources.append(
                    make_j_source(grid, pe.position, pe.component,
                                  pe.waveform, n_steps, materials)
                )
                continue

            if pe.extent is not None:
                axis_map = {"ex": 0, "ey": 1, "ez": 2}
                axis = axis_map[pe.component]
                end = list(pe.position)
                end[axis] += pe.extent
                wp = WirePort(
                    start=pe.position,
                    end=tuple(end),
                    component=pe.component,
                    impedance=pe.impedance,
                    excitation=pe.waveform,
                )
                materials = setup_wire_port(grid, wp, materials)
                if pe.excite:
                    sources.extend(make_wire_port_sources(grid, wp, materials, n_steps))
                for cell in _wire_port_cells(grid, wp):
                    if pec_mask_local is not None:
                        pec_mask_local = pec_mask_local.at[cell[0], cell[1], cell[2]].set(False)
                    if pec_occupancy_local is not None:
                        pec_occupancy_local = pec_occupancy_local.at[cell[0], cell[1], cell[2]].set(0.0)
                continue

            lp = LumpedPort(
                position=pe.position,
                component=pe.component,
                impedance=pe.impedance,
                excitation=pe.waveform,
            )
            materials = setup_lumped_port(grid, lp, materials)
            if pe.excite:
                sources.append(make_port_source(grid, lp, materials, n_steps))
            idx = grid.position_to_index(pe.position)
            if pec_mask_local is not None:
                pec_mask_local = pec_mask_local.at[idx[0], idx[1], idx[2]].set(False)
            if pec_occupancy_local is not None:
                pec_occupancy_local = pec_occupancy_local.at[idx[0], idx[1], idx[2]].set(0.0)
            # Register a JIT-integrated S-param accumulator for this
            # lumped port when the user requested forward(port_s11_freqs=...)
            # (issue #72). Skipping passive ports keeps S11 indexing
            # aligned with the active-port list.
            if _s11_freqs_arr is not None:
                lumped_port_sparam_specs.append(LumpedPortSParamSpec(
                    i=int(idx[0]), j=int(idx[1]), k=int(idx[2]),
                    component=pe.component,
                    freqs=_s11_freqs_arr,
                    impedance=float(pe.impedance),
                ))

        for pe in self._probes:
            probes.append(make_probe(grid, pe.position, pe.component))

        if not probes and self._ports:
            for pe in self._ports:
                probes.append(make_probe(grid, pe.position, pe.component))

        _, debye, lorentz = self._init_dispersion(
            materials, grid.dt, debye_spec, lorentz_spec,
        )

        ntff_box = None
        if self._ntff is not None:
            from rfx.farfield import make_ntff_box
            corner_lo, corner_hi, freqs = self._ntff
            ntff_box = make_ntff_box(grid, corner_lo, corner_hi, freqs)

        # Waveguide ports (differentiable DFT accumulation inside scan)
        waveguide_ports = []
        if self._waveguide_ports:
            wg_freqs = None
            for pe in self._waveguide_ports:
                if pe.freqs is not None:
                    wg_freqs = jnp.asarray(pe.freqs, dtype=jnp.float32)
                    break
            if wg_freqs is None:
                wg_freqs = jnp.linspace(
                    self._freq_max * 0.5, self._freq_max, 20, dtype=jnp.float32)
            for pe in self._waveguide_ports:
                waveguide_ports.append(
                    self._build_waveguide_port_config(pe, grid, wg_freqs, n_steps))

        # Floquet ports — inject soft source, same as run_uniform.py:274-327
        periodic = None
        if self._periodic_axes:
            periodic = tuple(axis in self._periodic_axes for axis in "xyz")

        if self._floquet_ports:
            axis_map_str = {"x": 0, "y": 1, "z": 2}
            for fpe in self._floquet_ports:
                axis_idx = axis_map_str[fpe.axis]
                fp_f0 = fpe.f0 if fpe.f0 is not None else self._freq_max / 2
                from rfx.sources.sources import GaussianPulse as _GP
                wf = _GP(f0=fp_f0, bandwidth=fpe.bandwidth, amplitude=fpe.amplitude)
                center = [self._domain[i] / 2.0 for i in range(3)]
                center[axis_idx] = fpe.position
                if fpe.polarization == "te":
                    comp = {"z": "ex", "x": "ey", "y": "ex"}[fpe.axis]
                else:
                    comp = {"z": "ey", "x": "ez", "y": "ez"}[fpe.axis]
                from rfx.simulation import make_source as _make_src
                sources.append(_make_src(grid, tuple(center), comp, wf, n_steps))
            if periodic is None:
                periodic = (True, True, False)  # default x-y periodic for Floquet

        periodic_bool = periodic if periodic is not None else (False, False, False)

        # Forward cpml_axes from the grid — when waveguide ports are
        # present the grid restricts CPML to the non-propagation axes.
        # The default _run cpml_axes="xyz" builds CPML state for axes
        # that have no padding, producing shape-broadcast errors like
        # (8,1,1) vs (nx,ny,nz) during the scan (issue #29). The run()
        # path forwards these explicitly at api.py:2012, so does the
        # waveguide compute path at :2077 / :2092.
        cpml_axes_run = grid.cpml_axes
        pec_axes_run = "".join(a for a in "xyz" if a not in cpml_axes_run)

        result = _run(
            grid,
            materials,
            n_steps,
            boundary=self._boundary,
            cpml_axes=cpml_axes_run,
            pec_axes=pec_axes_run,
            periodic=periodic_bool,
            debye=debye,
            lorentz=lorentz,
            sources=sources,
            probes=probes,
            waveguide_ports=waveguide_ports if waveguide_ports else None,
            ntff=ntff_box,
            checkpoint=checkpoint,
            checkpoint_segments=checkpoint_segments,
            pec_mask=pec_mask_local,
            pec_occupancy=pec_occupancy_local,
            lumped_port_sparams=lumped_port_sparam_specs or None,
            return_state=False,
        )

        s_params_out = getattr(result, "s_params", None)
        freqs_out = getattr(result, "freqs", None)
        if result.lumped_port_sparams:
            from rfx.probes.probes import extract_lumped_s11
            s_list = []
            for spec, accs in result.lumped_port_sparams:
                v_dft, i_dft = accs
                s_list.append(extract_lumped_s11(v_dft, i_dft, z0=spec.impedance))
            # Stacked as (n_ports, n_freqs); single-port shapes (n_freqs,).
            s_params_out = s_list[0] if len(s_list) == 1 else jnp.stack(s_list, axis=0)
            freqs_out = result.lumped_port_sparams[0][0].freqs

        return ForwardResult(
            time_series=result.time_series,
            ntff_data=result.ntff_data,
            ntff_box=result.ntff_box,
            grid=result.grid,
            s_params=s_params_out,
            freqs=freqs_out,
            lumped_port_sparams=result.lumped_port_sparams,
        )

    def _forward_nonuniform_from_materials(
        self,
        *,
        eps_override: jnp.ndarray | None = None,
        sigma_override: jnp.ndarray | None = None,
        pec_mask_override: jnp.ndarray | None = None,
        pec_occupancy_override: jnp.ndarray | None = None,
        n_steps: int,
        checkpoint: bool = True,
        emit_time_series: bool = True,
        checkpoint_every: int | None = None,
        n_warmup: int = 0,
        design_mask: jnp.ndarray | None = None,
    ) -> ForwardResult:
        """Differentiable forward on the non-uniform mesh path.

        Routes through ``run_nonuniform_path`` with optimisation overrides
        applied after material assembly, then repackages the returned
        ``Result`` into the minimal ``ForwardResult`` schema.

        When ``checkpoint`` is True (the default), the NU scan body is
        wrapped in ``jax.checkpoint`` so reverse-mode AD memory scales
        with ``sqrt(n_steps)`` instead of ``n_steps``.
        """
        from rfx.runners.nonuniform import run_nonuniform_path

        result = run_nonuniform_path(
            self,
            n_steps=n_steps,
            eps_override=eps_override,
            sigma_override=sigma_override,
            pec_mask_override=pec_mask_override,
            pec_occupancy_override=pec_occupancy_override,
            checkpoint=checkpoint,
            emit_time_series=emit_time_series,
            checkpoint_every=checkpoint_every,
            n_warmup=n_warmup,
            design_mask=design_mask,
        )
        return ForwardResult(
            time_series=result.time_series,
            ntff_data=result.ntff_data,
            ntff_box=result.ntff_box,
            grid=result.grid,
            s_params=getattr(result, "s_params", None),
            freqs=getattr(result, "freqs", None),
        )

    def _forward_distributed_nonuniform_from_materials(
        self,
        *,
        eps_override: jnp.ndarray | None = None,
        sigma_override: jnp.ndarray | None = None,
        pec_mask_override: jnp.ndarray | None = None,
        pec_occupancy_override: jnp.ndarray | None = None,
        n_steps: int,
        checkpoint: bool = True,
        emit_time_series: bool = True,
        checkpoint_every: int | None = None,
        n_warmup: int = 0,
        design_mask: jnp.ndarray | None = None,
        devices: list | None = None,
        exchange_interval: int = 1,
        skip_preflight: bool = False,
    ) -> ForwardResult:
        """Phase 3 (issue #44): differentiable forward on the **distributed**
        non-uniform mesh path.

        Mirrors :meth:`_forward_nonuniform_from_materials` but routes
        through the sharded NU runner
        (``rfx.runners.distributed_nu.run_nonuniform_distributed_pec``)
        with x-axis 1-D slab decomposition across ``devices``.  Performs
        the V3 §M4 distributed-specific preflight (5 checks) before any
        trace build, then assembles materials, builds the
        :class:`ShardedNUGrid`, shards every input array, and calls the
        Phase 2F runner.

        See :meth:`forward` for the public-facing kwarg semantics.
        """
        if self._flux_monitors:
            raise NotImplementedError(
                "add_flux_monitor() is not supported on the distributed "
                "non-uniform forward path; the sharded NU scan body does "
                "not accumulate flux DFTs. Drop flux monitors (use "
                "add_ntff_box() for far-field observables) or use the "
                "uniform lane."
            )
        import warnings as _w
        from rfx.runners.distributed_nu import (
            build_sharded_nu_grid,
            init_cpml_for_sharded_nu,
            run_nonuniform_distributed_pec,
            shard_cpml_state_x_slab,
            shard_debye_coeffs_x_slab,
            shard_debye_state_x_slab,
            shard_design_mask_x_slab,
            shard_lorentz_coeffs_x_slab,
            shard_lorentz_state_x_slab,
            shard_pec_mask_x_slab,
            shard_pec_occupancy_x_slab,
        )
        from rfx.core.yee import MaterialArrays
        from rfx.materials.debye import init_debye
        from rfx.materials.lorentz import init_lorentz
        from rfx.nonuniform import (
            position_to_index as _nu_pos_to_idx,
            make_current_source as _nu_make_current_source,
        )
        from rfx.simulation import ProbeSpec, SourceSpec

        # ---- Resolve devices (V3 §5 semantics) ----
        if devices is None:
            devices = list(jax.devices())
        else:
            available = list(jax.devices())
            # Reject lists longer than the available device count.  This
            # catches both "more devices than exist" and the duplicate-
            # entry case (a 4-element list built from 2 real devices)
            # before JAX errors out deep inside ``device_put`` with an
            # opaque ``safe_zip`` traceback.
            if len(devices) > len(available):
                raise ValueError(
                    f"forward(distributed=True, devices=...): requested "
                    f"{len(devices)} devices but only {len(available)} are "
                    "available in jax.devices()."
                )
            # Reject duplicate entries — jax.sharding.Mesh requires each
            # device to appear at most once.
            if len(set(map(id, devices))) != len(devices):
                raise ValueError(
                    "forward(distributed=True, devices=...): duplicate "
                    "device entries are not allowed; each device must "
                    "appear at most once."
                )
            for d in devices:
                if d not in available:
                    raise ValueError(
                        f"forward(distributed=True, devices=...): device "
                        f"{d!r} is not in jax.devices() "
                        f"(available={len(available)} devices)."
                    )
        n_devices = len(devices)

        # ---- Build the NU grid up front for preflight metrics ----
        grid = self._build_nonuniform_grid()

        # ---- V3 §M4 distributed-specific preflight (5 checks).  Skipped
        # entirely when the caller requested skip_preflight=True. ----
        if not skip_preflight:
            # Check 1 — device count.
            if n_devices < 2:
                raise ValueError(
                    f"forward(distributed=True) requires at least 2 "
                    f"devices; found {n_devices} (devices={devices!r}). "
                    "Use distributed=False on a single device, or pass an "
                    "explicit devices list with len>=2."
                )

            # Check 2 — grading ratio (hard error at >5).  Use the same
            # max-over-all-axes definition as Simulation.run().
            _max_ratio = 1.0
            for _prof in (
                self._dx_profile, self._dy_profile, self._dz_profile,
            ):
                if _prof is not None and len(_prof) > 0:
                    _pa = np.asarray(_prof, dtype=np.float64)
                    if float(_pa.min()) > 0.0:
                        _max_ratio = max(
                            _max_ratio,
                            float(_pa.max()) / float(_pa.min()),
                        )
            if _max_ratio > 5.0:
                raise ValueError(
                    f"grading_ratio={_max_ratio:.2f} exceeds 5.0; "
                    "distributed NU forward requires grading_ratio <= 5.0 "
                    "for shared-dt stability and x-face CPML calibration. "
                    "Reduce the cell-size variation or omit distributed=True."
                )

            # Check 3 — ghost width vs local slab.  Mirror
            # ``ShardedNUGrid`` arithmetic for nx_per_rank.
            nx = grid.nx
            pad_x = 0
            if nx % n_devices != 0:
                pad_x = n_devices - (nx % n_devices)
            nx_padded = nx + pad_x
            nx_per_rank = nx_padded // n_devices
            ghost_width = math.floor(exchange_interval / 2) + 1
            for rank in range(n_devices):
                if ghost_width > nx_per_rank:
                    raise ValueError(
                        f"ghost_width={ghost_width} exceeds "
                        f"nx_per_rank={nx_per_rank} for rank {rank}; "
                        "reduce exchange_interval or increase nx."
                    )

            # Check 4 — CPML vs local slab on outer boundary ranks.
            cpml_layers = int(getattr(self, "_cpml_layers", 0) or 0)
            if self._boundary == "cpml" and cpml_layers > 0:
                for rank in (0, n_devices - 1):
                    nx_local_real = nx_per_rank
                    if cpml_layers * 2 >= nx_local_real:
                        raise ValueError(
                            f"cpml_layers*2={cpml_layers * 2} >= "
                            f"nx_local={nx_local_real} on boundary rank "
                            f"{rank}; reduce cpml_layers (or set per-face "
                            f"lo_thickness/hi_thickness on the x Boundary) "
                            f"or increase nx."
                        )

            # Check 5 — segmented remat overhead warning.
            if (
                n_warmup == 0
                and checkpoint_every is not None
                and n_steps < 1000
            ):
                _w.warn(
                    f"checkpoint_every={checkpoint_every} with "
                    f"n_warmup=0 and n_steps={n_steps} < 1000 may spend "
                    "more time on recomputation overhead than it saves "
                    "in memory; consider checkpoint_every=None for small "
                    "runs.",
                    UserWarning,
                    stacklevel=3,
                )

        # ---- Assemble full-domain materials ----
        materials, debye_spec, lorentz_spec, pec_mask = (
            self._assemble_materials_nu(grid)
        )

        # ``eps_override`` / ``sigma_override`` may be JAX tracers (the
        # caller is differentiating w.r.t. eps/sigma).  Keep the original
        # concrete materials for ``make_current_source`` so source
        # normalisation stays Python-float (matches the single-device NU
        # runner's ``materials_concrete`` pattern in run_nonuniform_path).
        materials_concrete = materials
        if eps_override is not None or sigma_override is not None:
            materials = MaterialArrays(
                eps_r=(
                    eps_override if eps_override is not None
                    else materials.eps_r
                ),
                sigma=(
                    sigma_override if sigma_override is not None
                    else materials.sigma
                ),
                mu_r=materials.mu_r,
            )
        if pec_mask_override is not None:
            pec_mask = (
                pec_mask_override if pec_mask is None
                else (pec_mask | pec_mask_override)
            )

        # ---- Initialise Debye / Lorentz state on the full domain BEFORE
        # sharding (distributed_nu shard helpers expect the full-domain
        # arrays produced by init_debye / init_lorentz). ----
        debye = None
        if debye_spec is not None:
            debye_poles, debye_masks = debye_spec
            debye = init_debye(
                debye_poles, materials, grid.dt, mask=debye_masks,
            )
        lorentz = None
        if lorentz_spec is not None:
            lorentz_poles, lorentz_masks = lorentz_spec
            lorentz = init_lorentz(
                lorentz_poles, materials, grid.dt, mask=lorentz_masks,
            )

        # ---- Build sharded grid + mesh ----
        sharded_grid = build_sharded_nu_grid(
            grid, n_devices, exchange_interval=exchange_interval,
        )
        from jax.sharding import Mesh
        mesh = Mesh(np.array(devices), axis_names=("x",))

        # ---- Shard materials.  ``_split_materials`` lives in
        # rfx.runners.distributed and pads the high-x end before slabbing. ----
        from rfx.runners.distributed import _split_materials
        from jax.sharding import NamedSharding, PartitionSpec as _P
        shd = NamedSharding(mesh, _P("x"))
        nx = grid.nx
        pad_x = sharded_grid.pad_x
        if pad_x > 0:
            _pad_widths = ((0, pad_x), (0, 0), (0, 0))
            materials_padded = MaterialArrays(
                eps_r=jnp.pad(materials.eps_r, _pad_widths,
                              constant_values=1.0),
                sigma=jnp.pad(materials.sigma, _pad_widths,
                              constant_values=0.0),
                mu_r=jnp.pad(materials.mu_r, _pad_widths,
                             constant_values=1.0),
            )
        else:
            materials_padded = materials

        ghost = sharded_grid.ghost_width
        materials_slabs = _split_materials(
            materials_padded, n_devices, ghost,
        )

        def _shard_3d_stacked(arr):
            n_dev = arr.shape[0]
            rest = arr.shape[1:]
            return jax.device_put(
                arr.reshape(n_dev * rest[0], *rest[1:]), shd,
            )

        sharded_materials = MaterialArrays(
            eps_r=_shard_3d_stacked(materials_slabs.eps_r),
            sigma=_shard_3d_stacked(materials_slabs.sigma),
            mu_r=_shard_3d_stacked(materials_slabs.mu_r),
        )

        # ---- Shard PEC mask / occupancy / design mask via Phase 2 helpers. ----
        sharded_pec_mask = shard_pec_mask_x_slab(pec_mask, sharded_grid)
        sharded_pec_occupancy = shard_pec_occupancy_x_slab(
            pec_occupancy_override, sharded_grid,
        )
        sharded_design_mask = shard_design_mask_x_slab(
            design_mask, sharded_grid,
        )

        # ---- CPML init + sharding (Phase 2C). ----
        cpml_params = None
        cpml_state_sharded = None
        cpml_layers = int(getattr(self, "_cpml_layers", 0) or 0)
        if self._boundary == "cpml" and cpml_layers > 0:
            cpml_params, cpml_state_stacked = init_cpml_for_sharded_nu(
                sharded_grid, n_devices,
                pec_faces=getattr(self, "_pec_faces", None),
            )
            cpml_state_sharded = shard_cpml_state_x_slab(
                cpml_state_stacked, sharded_grid, mesh,
            )

        # ---- Shard Debye / Lorentz dispersion (Phase 2D). ----
        sharded_debye = None
        if debye is not None:
            db_coeffs, db_state = debye
            sharded_debye = (
                shard_debye_coeffs_x_slab(db_coeffs, sharded_grid, mesh),
                shard_debye_state_x_slab(db_state, sharded_grid, mesh),
            )
        sharded_lorentz = None
        if lorentz is not None:
            lr_coeffs, lr_state = lorentz
            sharded_lorentz = (
                shard_lorentz_coeffs_x_slab(lr_coeffs, sharded_grid, mesh),
                shard_lorentz_state_x_slab(lr_state, sharded_grid, mesh),
            )

        # ---- Sources / probes (lumped/wire/coax ports unsupported here). ----
        if self._lumped_rlc:
            raise NotImplementedError(
                "Lumped RLC ports are not yet supported on the "
                "distributed=True forward path; remove the lumped_rlc "
                "spec or omit distributed=True."
            )
        sources: list[SourceSpec] = []
        for pe in self._ports:
            if pe.impedance > 0.0:
                raise NotImplementedError(
                    "Lumped / wire ports (impedance > 0) are not yet "
                    "supported on the distributed=True forward path; "
                    "use distributed=False or replace with a current "
                    "source (impedance=0)."
                )
            idx = _nu_pos_to_idx(grid, pe.position)
            # Use concrete materials so make_current_source can resolve
            # eps / sigma to Python floats (it calls ``float(...)`` on
            # both for the source-cell normalisation).
            si, sj, sk, sc, wf = _nu_make_current_source(
                grid, idx, pe.component, pe.waveform, n_steps,
                materials_concrete,
            )
            sources.append(SourceSpec(
                i=int(si), j=int(sj), k=int(sk),
                component=sc, waveform=jnp.asarray(wf),
            ))

        probes: list[ProbeSpec] = []
        for pe in self._probes:
            idx = _nu_pos_to_idx(grid, pe.position)
            probes.append(ProbeSpec(
                i=int(idx[0]), j=int(idx[1]), k=int(idx[2]),
                component=pe.component,
            ))

        # ---- Launch the sharded NU runner. ----
        result = run_nonuniform_distributed_pec(
            sharded_grid,
            sharded_materials,
            sharded_pec_mask,
            n_steps,
            sources=sources,
            probes=probes,
            n_devices=n_devices,
            exchange_interval=exchange_interval,
            debye=sharded_debye,
            lorentz=sharded_lorentz,
            devices=devices,
            cpml_params=cpml_params,
            cpml_state=cpml_state_sharded,
            sharded_pec_occupancy=sharded_pec_occupancy,
            checkpoint_every=checkpoint_every,
            n_warmup=n_warmup,
            sharded_design_mask=sharded_design_mask,
            emit_time_series=emit_time_series,
            pmc_faces=frozenset(self._boundary_spec.pmc_faces()),
        )

        # ---- Repackage into ForwardResult.
        # Both the distributed runner and the single-device NU runner
        # return time_series with layout ``(n_steps, n_probes)``; we
        # surface that schema unchanged so vmap_sweep / decay_convergence
        # / lumped_rlc / etc. continue to work.
        ts = result.get("time_series")
        return ForwardResult(
            time_series=ts,
            ntff_data=None,
            ntff_box=None,
            grid=grid,
            s_params=None,
            freqs=None,
        )

    # ---- forward (differentiable) ----

    def forward(
        self,
        *,
        eps_override: jnp.ndarray | None = None,
        sigma_override: jnp.ndarray | None = None,
        pec_mask_override: jnp.ndarray | None = None,
        pec_occupancy_override: jnp.ndarray | None = None,
        n_steps: int | None = None,
        num_periods: float = 20.0,
        checkpoint: bool = True,
        checkpoint_segments: int | None = None,
        emit_time_series: bool = True,
        checkpoint_every: int | None = None,
        n_warmup: int = 0,
        skip_preflight: bool = False,
        design_mask: jnp.ndarray | None = None,
        distributed: bool = False,
        devices: list | None = None,
        exchange_interval: int = 1,
        port_s11_freqs: object | None = None,
    ) -> ForwardResult:
        """Run a minimal differentiable forward simulation.

        This path is designed for ``jax.grad`` / ``jax.value_and_grad`` and
        intentionally returns only the observables needed by differentiable
        objectives instead of the broader stateful :meth:`run` result.

        Parameters
        ----------
        eps_override : jnp.ndarray or None
            Replacement permittivity array with shape ``grid.shape``.
        sigma_override : jnp.ndarray or None
            Replacement conductivity array with shape ``grid.shape``.
        pec_mask_override : jnp.ndarray or None
            Additional hard PEC mask to merge with geometry-defined PEC.
        pec_occupancy_override : jnp.ndarray or None
            Relaxed conductor occupancy field in ``[0, 1]`` for
            differentiable PEC-style optimisation.
        n_steps : int or None
            Number of timesteps. If None, auto-computed from *num_periods*.
        num_periods : float
            Number of periods at freq_max for auto step count.
        checkpoint : bool
            Enable gradient checkpointing (default True).
        distributed : bool, optional
            **Opt-in, unstable, pending GPU evidence (issue #44).**  When
            ``True`` and a non-uniform mesh is configured, route the
            differentiable forward through the sharded NU runner
            (``rfx.runners.distributed_nu.run_nonuniform_distributed_pec``)
            using a 1-D x-slab decomposition across ``devices``.  Defaults
            to ``False`` (single-device path, no behaviour change).  In
            v1.6.2 the distributed forward path is **NU-only** (DP3): a
            uniform mesh raises ``NotImplementedError``.  TFSF sources
            and waveguide ports are unsupported on this path and raise
            ``NotImplementedError`` at preflight.
        devices : list of jax.Device or None, optional
            Devices for the distributed run.  When ``distributed=True``
            and ``devices=None``, defaults to ``jax.devices()``.  When
            an explicit list is supplied, every device must already exist
            in ``jax.devices()`` (otherwise ``ValueError``).  Passing
            ``devices=`` *without* ``distributed=True`` raises
            ``ValueError`` — there is no silent activation of the
            distributed lane.
        exchange_interval : int, optional
            Ghost-cell exchange interval for the distributed runner
            (default ``1``).  Only ``1`` is currently supported by
            ``run_nonuniform_distributed_pec``; other values are
            forward-compatible reservations and will raise inside the
            runner.

        Returns
        -------
        ForwardResult
            Minimal differentiable observables (time series and optional NTFF).
        """
        # Phase 3 (issue #44 V3 §M6): one-shot UserWarning for the opt-in
        # distributed=True path so users know the path is opt-in / unstable
        # / pending GPU evidence.  The flag lives at module scope so we
        # warn exactly once per process, not per Simulation instance.
        global _DISTRIBUTED_FIRST_CALL_WARNED
        if distributed and not _DISTRIBUTED_FIRST_CALL_WARNED:
            import warnings as _w
            _w.warn(
                "Simulation.forward(distributed=True) is opt-in and pending "
                "GPU evidence (see issue #44). Set distributed=True only "
                "after reading "
                "docs/research_notes/2026-04-16_issue44_v3_plan.md.",
                UserWarning,
                stacklevel=2,
            )
            _DISTRIBUTED_FIRST_CALL_WARNED = True

        # Phase 3 (V3 dispatch rule): devices= without distributed=True
        # must be rejected with a clear ValueError.  No silent activation.
        if devices is not None and not distributed:
            raise ValueError(
                "forward(devices=...) requires distributed=True; "
                "passing devices without distributed=True is rejected to "
                "avoid silent activation of the distributed lane. "
                "Either set distributed=True or omit devices."
            )

        self._auto_preflight(skip=skip_preflight, context="forward")

        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )

        # Issue #72: forward(port_s11_freqs=...) is currently wired only on
        # the uniform single-device path. Reject loudly elsewhere so users
        # don't get a silent s_params=None.
        if port_s11_freqs is not None and (distributed or is_nonuniform):
            raise NotImplementedError(
                "forward(port_s11_freqs=...) is currently wired only on the "
                "uniform single-device forward path (issue #72). Drop "
                "port_s11_freqs or run on a uniform mesh without "
                "distributed=True."
            )

        # Issue #73: forward(checkpoint_segments=...) is currently wired only
        # on the uniform single-device path. Reject loudly elsewhere — both
        # for distributed=True and for non-uniform meshes — so users don't
        # get a silent fall-back to the linear-memory scan that this kwarg
        # was meant to fix. NU follow-up will mirror the pattern in
        # run_nonuniform; track on issue #73.
        if checkpoint_segments is not None and (distributed or is_nonuniform):
            raise NotImplementedError(
                "forward(checkpoint_segments=...) is currently wired only "
                "on the uniform single-device forward path (issue #73). "
                "Drop checkpoint_segments or run on a uniform mesh without "
                "distributed=True. NU support is tracked as a follow-up."
            )

        # Phase 3: distributed dispatch (V3 lines 842-847).
        if distributed:
            # NU-only in v1.6.2 (DP3 locked decision).
            if not is_nonuniform:
                raise NotImplementedError(
                    "distributed=True on forward() is currently implemented "
                    "only for non-uniform meshes; use run(..., devices=...) "
                    "for the uniform distributed path."
                )
            # Reject TFSF / waveguide ports up front (V3 §3 unsupported).
            if self._tfsf is not None:
                raise NotImplementedError(
                    "TFSF sources are not supported on the distributed "
                    "forward path; remove the TFSF source or omit "
                    "distributed=True."
                )
            if self._waveguide_ports:
                raise NotImplementedError(
                    "Waveguide ports are not supported on the distributed "
                    "forward path; remove waveguide ports or omit "
                    "distributed=True."
                )
            # v1.7.4 T8: PMC is now wired across all three sharded
            # runners (distributed_nu, distributed_v2, distributed).
            # The reject guard that used to live here (introduced in
            # f3cab7c) has been removed. The single-device PMC runtime
            # hook lives in rfx/simulation.py:703-705 and the sharded
            # PMC helpers live in each runner next to their PEC analog.
            # Synthesise missing dz profile so the NU grid build always
            # sees all three axes (mirrors Simulation.run() and the
            # single-device NU forward path).
            if self._dz_profile is None:
                nz_phys = max(1, int(round(self._domain[2] / self._dx)))
                self._dz_profile = np.full(nz_phys, float(self._dx))
            if n_steps is None:
                grid_probe = self._build_nonuniform_grid()
                period = 1.0 / float(self._freq_max)
                n_steps = int(np.ceil(num_periods * period / float(grid_probe.dt)))
            return self._forward_distributed_nonuniform_from_materials(
                eps_override=eps_override,
                sigma_override=sigma_override,
                pec_mask_override=pec_mask_override,
                pec_occupancy_override=pec_occupancy_override,
                n_steps=n_steps,
                checkpoint=checkpoint,
                emit_time_series=emit_time_series,
                checkpoint_every=checkpoint_every,
                n_warmup=n_warmup,
                design_mask=design_mask,
                devices=devices,
                exchange_interval=exchange_interval,
                skip_preflight=skip_preflight,
            )

        if is_nonuniform:
            # Synthesise missing dz profile so the NU grid build always
            # sees all three axes (mirrors Simulation.run()'s pre-build
            # synthesis at api.py ~4268).  Required for sims that set
            # only dx/dy_profile and leave dz uniform.
            if self._dz_profile is None:
                nz_phys = max(1, int(round(self._domain[2] / self._dx)))
                self._dz_profile = np.full(nz_phys, float(self._dx))
            # Let the NU runner build grid/materials so it can apply the
            # NU-aware pec_mask and port/source setup against per-axis widths.
            if n_steps is None:
                grid_probe = self._build_nonuniform_grid()
                period = 1.0 / float(self._freq_max)
                n_steps = int(np.ceil(num_periods * period / float(grid_probe.dt)))
            return self._forward_nonuniform_from_materials(
                eps_override=eps_override,
                sigma_override=sigma_override,
                pec_mask_override=pec_mask_override,
                pec_occupancy_override=pec_occupancy_override,
                n_steps=n_steps,
                checkpoint=checkpoint,
                emit_time_series=emit_time_series,
                checkpoint_every=checkpoint_every,
                n_warmup=n_warmup,
                design_mask=design_mask,
            )
        if not emit_time_series:
            raise NotImplementedError(
                "emit_time_series=False is currently only supported on the "
                "non-uniform forward path. Frequency-domain objectives "
                "(NTFF, S-params) on uniform meshes still emit time series."
            )
        if checkpoint_every is not None:
            raise NotImplementedError(
                "checkpoint_every (segmented remat) is currently only "
                "supported on the non-uniform forward path. For the "
                "uniform path, use checkpoint_segments instead (issue #73)."
            )
        if design_mask is not None:
            raise NotImplementedError(
                "design_mask (issue #41) is currently only supported on the "
                "non-uniform forward path. Ping #41 if you need it on the "
                "uniform path — the same step_fn stop_gradient pattern applies."
            )
        grid = self._build_grid()
        materials, debye_spec, lorentz_spec, pec_mask, _, _ = self._assemble_materials(grid)

        if eps_override is not None or sigma_override is not None:
            materials = MaterialArrays(
                eps_r=eps_override if eps_override is not None else materials.eps_r,
                sigma=sigma_override if sigma_override is not None else materials.sigma,
                mu_r=materials.mu_r,
            )

        if pec_mask_override is not None:
            pec_mask = pec_mask_override if pec_mask is None else (pec_mask | pec_mask_override)

        if n_steps is None:
            n_steps = grid.num_timesteps(num_periods=num_periods)

        return self._forward_from_materials(
            grid,
            materials,
            debye_spec,
            lorentz_spec,
            n_steps=n_steps,
            checkpoint=checkpoint,
            checkpoint_segments=checkpoint_segments,
            pec_mask=pec_mask,
            pec_occupancy=pec_occupancy_override,
            port_s11_freqs=port_s11_freqs,
        )

    # ---- run ----

    def run(
        self,
        *,
        n_steps: int | None = None,
        num_periods: float = 20.0,
        until_decay: float | None = None,
        decay_check_interval: int = 50,
        decay_min_steps: int = 100,
        decay_max_steps: int = 50_000,
        decay_monitor_component: str = "ez",
        decay_monitor_position: tuple[float, float, float] | None = None,
        checkpoint: bool = False,
        compute_s_params: bool | None = None,
        s_param_freqs: jnp.ndarray | None = None,
        s_param_n_steps: int | None = None,
        snapshot: SnapshotSpec | None = None,
        subpixel_smoothing: bool = False,
        conformal_pec: bool = False,
        conformal_min_weight: float = 0.1,
        devices: list | None = None,
        exchange_interval: int = 1,
    ) -> Result:
        """Run the simulation.

        Parameters
        ----------
        n_steps : int or None
            Number of timesteps. If None, auto-computed from num_periods.
        num_periods : float
            Number of periods at freq_max for auto timestep count.
        checkpoint : bool
            Enable gradient checkpointing for reverse-mode AD.
        compute_s_params : bool or None
            Compute S-parameter matrix. Default: True when ports exist.
        s_param_freqs : array or None
            Frequencies for S-parameters. Default: 50 points from
            freq_max/10 to freq_max.
        s_param_n_steps : int or None
            Timesteps for S-parameter simulation (may need more than
            the main simulation for frequency resolution).
        until_decay : float or None
            When provided, overrides *n_steps* and runs until field
            energy decays to this fraction of peak. E.g. ``1e-3``.
        decay_check_interval : int
            Check decay every N steps (default 50).
        decay_min_steps : int
            Always run at least this many steps (default 100).
        decay_max_steps : int
            Hard upper limit on steps (default 50000).
        decay_monitor_component : str
            Field component to monitor (default ``"ez"``).
        decay_monitor_position : tuple or None
            Physical position to monitor. If None, use domain center.
        conformal_pec : bool
            Enable Dey-Mittra conformal PEC for second-order accuracy
            on curved PEC surfaces. Default False (staircase PEC).
        conformal_min_weight : float
            Minimum conformal weight for CFL stability clamping.
            Default 0.1. Recommended range: 0.05-0.3.
        devices : list of jax.Device or None
            When a list with len > 1 is provided, run the simulation
            distributed across those devices using 1D slab decomposition
            along the x-axis (via ``jax.pmap``).  Phase 1 supports PEC
            boundary, soft sources, and point probes.
        exchange_interval : int, optional
            How often (in timesteps) to perform ghost cell exchange in
            the distributed runner.  Default 1 (every step).  Higher
            values (2-4) reduce synchronization overhead at the cost of
            O(interval * dt) boundary error.

        Returns
        -------
        Result
        """
        # ---- P1: Auto mesh when dx not specified and geometry exists ----
        if self._dx is None and self._geometry:
            self._auto_configure_mesh()

        # ---- P0: Pre-simulation validation ----
        self._validate_mesh_quality()
        self._validate_simulation_config()

        if self._solver == "adi" and devices is not None and len(devices) > 1:
            raise ValueError("solver='adi' does not support distributed execution")
        if self._boundary == "upml" and devices is not None and len(devices) > 1:
            raise ValueError("boundary='upml' does not support distributed execution")

        # ---- Distributed + nonuniform (Phase B guardrail).
        # Phase B permits the combination for PEC boundary with grading
        # ratio <= 5 and no TFSF. The distributed_v2 runner dispatches to
        # the NU kernels in distributed_nu.py; dispersion and CPML on the
        # distributed NU path are Phase C items and still raise below.
        _nu_profile = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )
        if (devices is not None and len(devices) > 1 and _nu_profile):
            import warnings as _wmod
            # Grading ratio check (shared single dt) across provided profiles.
            _max_ratio = 1.0
            for _prof in (
                self._dx_profile, self._dy_profile, self._dz_profile
            ):
                if _prof is not None and len(_prof) > 0:
                    _pa = np.asarray(_prof, dtype=np.float64)
                    if float(_pa.min()) > 0.0:
                        _max_ratio = max(
                            _max_ratio,
                            float(_pa.max()) / float(_pa.min()),
                        )
            if _max_ratio > 5.0:
                raise ValueError(
                    "Distributed + non-uniform requires grading ratio "
                    "<= 5:1 for shared-dt stability; got "
                    f"{_max_ratio:.2f}:1."
                )
            if self._tfsf is not None:
                raise ValueError(
                    "Distributed + non-uniform does not support TFSF "
                    "plane-wave sources (Phase B scope)."
                )
            if self._solver == "adi":
                raise ValueError(
                    "Distributed + non-uniform does not support solver='adi'."
                )
            if _max_ratio > 3.0:
                _wmod.warn(
                    f"Distributed + non-uniform grading ratio {_max_ratio:.2f}"
                    ":1 exceeds the 3:1 stability caution threshold. "
                    "Monitor for numerical dispersion / late-time drift.",
                    stacklevel=2,
                )

        # ---- Distributed multi-device path ----
        if devices is not None and len(devices) > 1:
            self._warn_unsupported_run_kwargs("distributed multi-device", {
                "subpixel_smoothing": subpixel_smoothing,
                "checkpoint": checkpoint,
                "snapshot": snapshot,
                "until_decay": until_decay,
                "conformal_pec": conformal_pec,
                "compute_s_params": compute_s_params,
                "s_param_freqs": s_param_freqs,
                "s_param_n_steps": s_param_n_steps,
            })
            if n_steps is None:
                if _nu_profile:
                    # Synthesise missing dz profile so _build_nonuniform_grid
                    # has all three axes available.
                    if self._dz_profile is None:
                        nz_phys = max(
                            1, int(round(self._domain[2] / self._dx)))
                        self._dz_profile = np.full(
                            nz_phys, float(self._dx))
                    _ngrid = self._build_nonuniform_grid()
                    n_steps = int(np.ceil(
                        num_periods / (self._freq_max * _ngrid.dt)))
                else:
                    grid = self._build_grid()
                    n_steps = grid.num_timesteps(num_periods=num_periods)
            from rfx.runners.distributed_v2 import run_distributed
            return run_distributed(
                self, n_steps=n_steps, devices=devices,
                exchange_interval=exchange_interval,
            )

        # ---- Non-uniform mesh path ----
        if (self._dz_profile is not None
                or self._dx_profile is not None
                or self._dy_profile is not None):
            self._warn_unsupported_run_kwargs("non-uniform mesh", {
                "snapshot": snapshot,
                "until_decay": until_decay,
                "conformal_pec": conformal_pec,
            })
            # Synthesize a uniform dz profile from the scalar domain_z
            # when only dx/dy are non-uniform, so the shared non-uniform
            # runner has all three axes available.
            if self._dz_profile is None:
                nz_phys = max(1, int(round(self._domain[2] / self._dx)))
                self._dz_profile = np.full(nz_phys, float(self._dx))
            nu_grid = self._build_nonuniform_grid()
            if n_steps is None:
                n_steps = int(np.ceil(
                    num_periods / (self._freq_max * nu_grid.dt)))
            return self._run_nonuniform(
                n_steps=n_steps,
                compute_s_params=compute_s_params,
                s_param_freqs=s_param_freqs,
                subpixel_smoothing=subpixel_smoothing,
                checkpoint=checkpoint,
            )

        grid = self._build_grid()
        base_materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, kerr_chi3 = self._assemble_materials(grid)

        if self._solver == "adi":
            if until_decay is not None:
                raise ValueError("solver='adi' does not support until_decay yet")
            if snapshot is not None:
                raise ValueError("solver='adi' does not support snapshots yet")
            if n_steps is None:
                n_steps = grid.num_timesteps(num_periods=num_periods)
            return self._run_adi_from_materials(
                grid,
                base_materials,
                debye_spec,
                lorentz_spec,
                n_steps=n_steps,
                pec_mask=pec_mask,
                return_state=True,
            )

        # ---- Subgridded path ----
        if self._refinement is not None:
            self._warn_unsupported_run_kwargs("subgridded (SBP-SAT)", {
                "subpixel_smoothing": subpixel_smoothing,
                "checkpoint": checkpoint,
                "snapshot": snapshot,
                "until_decay": until_decay,
                "conformal_pec": conformal_pec,
                "compute_s_params": compute_s_params,
                "s_param_freqs": s_param_freqs,
                "s_param_n_steps": s_param_n_steps,
            })
            return self._run_subgridded(
                grid, base_materials, pec_mask,
                n_steps=n_steps or grid.num_timesteps(num_periods=num_periods),
            )

        # ---- Uniform path ----
        if n_steps is None:
            n_steps = grid.num_timesteps(num_periods=num_periods)

        from rfx.runners.uniform import run_uniform
        _field_dtype = jnp.float16 if self._precision == "mixed" else None
        return run_uniform(
            self,
            n_steps=n_steps,
            until_decay=until_decay,
            decay_check_interval=decay_check_interval,
            decay_min_steps=decay_min_steps,
            decay_max_steps=decay_max_steps,
            decay_monitor_component=decay_monitor_component,
            decay_monitor_position=decay_monitor_position,
            checkpoint=checkpoint,
            compute_s_params=compute_s_params,
            s_param_freqs=s_param_freqs,
            s_param_n_steps=s_param_n_steps,
            snapshot=snapshot,
            subpixel_smoothing=subpixel_smoothing,
            conformal_pec=conformal_pec,
            conformal_min_weight=conformal_min_weight,
            pec_shapes=pec_shapes,
            grid=grid,
            base_materials=base_materials,
            debye_spec=debye_spec,
            lorentz_spec=lorentz_spec,
            pec_mask=pec_mask,
            kerr_chi3=kerr_chi3,
            field_dtype=_field_dtype,
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
