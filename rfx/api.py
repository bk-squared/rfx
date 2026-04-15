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
    """
    time_series: jnp.ndarray
    ntff_data: object = None
    ntff_box: object = None
    grid: object = None
    s_params: object = None
    freqs: object = None


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
        boundary: str = "cpml",
        cpml_layers: int = 8,
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
        if boundary not in ("pec", "cpml", "upml"):
            raise ValueError(f"boundary must be 'pec', 'cpml', or 'upml', got {boundary!r}")
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
        if dx_profile is not None:
            domain = (float(np.sum(dx_profile)),
                      domain[1] if len(domain) >= 2 else 0.0,
                      domain[2] if len(domain) >= 3 else 0.0)
        if dy_profile is not None:
            domain = (domain[0],
                      float(np.sum(dy_profile)),
                      domain[2] if len(domain) >= 3 else 0.0)
        if dz_profile is not None:
            if any(d <= 0 for d in domain[:2]):
                raise ValueError(f"domain x/y must be positive, got {domain}")
            dz_total = float(np.sum(dz_profile))
            if len(domain) < 3 or domain[2] <= 0:
                domain = (domain[0], domain[1], dz_total)
        elif any(d <= 0 for d in domain):
            raise ValueError(f"domain dimensions must be positive, got {domain}")

        # P2: Warn on abrupt grading in user-supplied dz_profile
        if dz_profile is not None and len(dz_profile) > 1:
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
        self._pec_faces = set(pec_faces) if pec_faces else set()
        if self._pec_faces - _valid_faces:
            raise ValueError(
                f"pec_faces must be subset of {_valid_faces}, "
                f"got invalid: {self._pec_faces - _valid_faces}")
        if boundary == "pec" and self._pec_faces:
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
        ))
        return self

    def set_periodic_axes(self, axes: str = "xyz") -> "Simulation":
        """Set periodic boundary axes for high-level runs.

        Parameters
        ----------
        axes : str
            Any combination of ``x``, ``y``, ``z``. Empty string disables
            manual periodic overrides.
        """
        normalized = "".join(axis for axis in "xyz" if axis in axes)
        invalid = sorted(set(axes) - set("xyz"))
        if invalid:
            raise ValueError(f"periodic axes must be drawn from 'xyz', got invalid axes {invalid}")
        if self._tfsf is not None:
            raise ValueError("Manual periodic-axis overrides are not supported together with TFSF")
        if self._waveguide_ports:
            raise ValueError("Manual periodic-axis overrides are not supported together with waveguide ports")
        self._periodic_axes = normalized
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
        if self._waveguide_ports or extra_waveguide_axes:
            cpml_axes = self._waveguide_cpml_axes(extra_waveguide_axes)
            return Grid(
                freq_max=self._freq_max,
                domain=self._domain,
                dx=self._dx,
                cpml_layers=self._cpml_layers,
                cpml_axes=cpml_axes,
                mode=self._mode,
                kappa_max=self._cpml_kappa_max,
                pec_faces=self._pec_faces,
            )
        return Grid(
            freq_max=self._freq_max,
            domain=self._domain,
            dx=self._dx,
            cpml_layers=self._cpml_layers,
            mode=self._mode,
            kappa_max=self._cpml_kappa_max,
            pec_faces=self._pec_faces,
        )

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
            n = self._cpml_layers
            for arr_name in ("eps_r", "sigma", "mu_r"):
                arr = locals()[arr_name]
                if grid.pad_x > 0:
                    arr = arr.at[:n, :, :].set(arr[n:n+1, :, :])      # x-lo
                    arr = arr.at[-n:, :, :].set(arr[-n-1:-n, :, :])    # x-hi
                if grid.pad_y > 0:
                    arr = arr.at[:, :n, :].set(arr[:, n:n+1, :])      # y-lo
                    arr = arr.at[:, -n:, :].set(arr[:, -n-1:-n, :])    # y-hi
                if grid.pad_z > 0:
                    arr = arr.at[:, :, :n].set(arr[:, :, n:n+1])      # z-lo
                    arr = arr.at[:, :, -n:].set(arr[:, :, -n-1:-n])    # z-hi
                locals()[arr_name]  # reassign through locals won't work; use explicit
            # Re-bind after modification (locals() trick doesn't persist)
            eps_r_ext = eps_r
            sigma_ext = sigma
            mu_r_ext = mu_r
            if grid.pad_x > 0:
                eps_r_ext = eps_r_ext.at[:n,:,:].set(eps_r_ext[n:n+1,:,:])
                eps_r_ext = eps_r_ext.at[-n:,:,:].set(eps_r_ext[-n-1:-n,:,:])
                sigma_ext = sigma_ext.at[:n,:,:].set(sigma_ext[n:n+1,:,:])
                sigma_ext = sigma_ext.at[-n:,:,:].set(sigma_ext[-n-1:-n,:,:])
                mu_r_ext = mu_r_ext.at[:n,:,:].set(mu_r_ext[n:n+1,:,:])
                mu_r_ext = mu_r_ext.at[-n:,:,:].set(mu_r_ext[-n-1:-n,:,:])
            if grid.pad_y > 0:
                eps_r_ext = eps_r_ext.at[:,:n,:].set(eps_r_ext[:,n:n+1,:])
                eps_r_ext = eps_r_ext.at[:,-n:,:].set(eps_r_ext[:,-n-1:-n,:])
                sigma_ext = sigma_ext.at[:,:n,:].set(sigma_ext[:,n:n+1,:])
                sigma_ext = sigma_ext.at[:,-n:,:].set(sigma_ext[:,-n-1:-n,:])
                mu_r_ext = mu_r_ext.at[:,:n,:].set(mu_r_ext[:,n:n+1,:])
                mu_r_ext = mu_r_ext.at[:,-n:,:].set(mu_r_ext[:,-n-1:-n,:])
            if grid.pad_z > 0:
                eps_r_ext = eps_r_ext.at[:,:,:n].set(eps_r_ext[:,:,n:n+1])
                eps_r_ext = eps_r_ext.at[:,:,-n:].set(eps_r_ext[:,:,-n-1:-n])
                sigma_ext = sigma_ext.at[:,:,:n].set(sigma_ext[:,:,n:n+1])
                sigma_ext = sigma_ext.at[:,:,-n:].set(sigma_ext[:,:,-n-1:-n])
                mu_r_ext = mu_r_ext.at[:,:,:n].set(mu_r_ext[:,:,n:n+1])
                mu_r_ext = mu_r_ext.at[:,:,-n:].set(mu_r_ext[:,:,-n-1:-n])
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
        )
        return cfg

    def compute_waveguide_s_matrix(
        self,
        *,
        n_steps: int | None = None,
        num_periods: float = 20.0,
        normalize: bool = False,
    ) -> WaveguideSMatrixResult:
        """Compute a theoretically clean axis-normal boundary-aperture waveguide S-matrix.

        Parameters
        ----------
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
                desired_ref = (
                    entry.reference_plane
                    if entry.reference_plane is not None
                    else waveguide_plane_positions(first_cfg)["reference"]
                )
                ref_shifts_mm.append(desired_ref - waveguide_plane_positions(first_cfg)["reference"])

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
            desired_ref = (
                entry.reference_plane
                if entry.reference_plane is not None
                else waveguide_plane_positions(cfg)["reference"]
            )
            ref_shifts.append(desired_ref - waveguide_plane_positions(cfg)["reference"])

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
                        s_param_freqs=None):
        """Run simulation on non-uniform grid with graded dz."""
        from rfx.runners.nonuniform import run_nonuniform_path
        return run_nonuniform_path(
            self,
            n_steps=n_steps,
            compute_s_params=compute_s_params,
            s_param_freqs=s_param_freqs,
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
                    # Tightened thresholds (issue #30 CHECK 1):
                    # - PEC features carrying current need ≥5 cells
                    # - Dielectric features need ≥10 cells for accurate fields
                    mat = self._resolve_material(mat_name)
                    is_pec = mat.sigma >= self._PEC_SIGMA_THRESHOLD
                    thresh = 5 if is_pec else 10
                    if dim < thresh * cell:
                        axis_name = "xyz"[axis]
                        cells = dim / cell
                        kind = "PEC" if is_pec else "dielectric"
                        _w.warn(
                            f"'{mat_name}' {axis_name}-extent {dim*1e3:.2f}mm = "
                            f"{cells:.1f} cells — under-resolved ({kind} needs "
                            f"≥{thresh} cells). Consider finer mesh.",
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

        # Thin-metal-on-NU-mesh symmetry (Meep/OpenEMS convention — issue #48).
        self._validate_thin_metal_on_nu_mesh()

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

        # Per-axis CPML thickness (z may differ on non-uniform mesh)
        # Faces with pec_faces override have zero effective CPML thickness
        cpml_thick_xyz = [cpml_thickness, cpml_thickness, cpml_thickness]
        if self._dz_profile is not None and self._boundary in ("cpml", "upml") and self._cpml_layers > 0:
            n = min(self._cpml_layers, len(self._dz_profile))
            cpml_thick_xyz[2] = float(sum(self._dz_profile[:n]))
        # Zero out thickness for PEC-overridden faces
        if "z_lo" in self._pec_faces:
            cpml_thick_xyz[2] = 0  # z_lo is PEC, no CPML absorption there

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
                    ct = cpml_thick_xyz[min(ax, 2)]
                    if coord < ct * 0.5 or coord > domain_extent - ct * 0.5:
                        _w.warn(
                            f"Probe at {pos} is near/inside {absorber_label} region "
                            f"({absorber_label} {'xyz'[ax]}-thickness={ct*1e3:.1f}mm). "
                            f"Signal will be attenuated. Move probe to interior.",
                            stacklevel=3,
                        )
                        break

            for pe in self._ports:
                pos = pe.position
                for ax, coord in enumerate(pos):
                    domain_extent = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                    ct = cpml_thick_xyz[min(ax, 2)]
                    if coord < ct * 0.5 or coord > domain_extent - ct * 0.5:
                        _w.warn(
                            f"Source/port at {pos} is near/inside {absorber_label} region "
                            f"({absorber_label} {'xyz'[ax]}-thickness={ct*1e3:.1f}mm). "
                            f"Energy will be absorbed. Move source to interior.",
                            stacklevel=3,
                        )
                        break

        # P1.4: NTFF box overlap with absorber
        if self._ntff is not None and cpml_thickness > 0:
            corner_lo, corner_hi, _ = self._ntff
            for ax in range(3):
                domain_ext = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                ct = cpml_thick_xyz[min(ax, 2)]
                if corner_lo[ax] < ct or corner_hi[ax] > domain_ext - ct:
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

        # P1.9: Dielectric material extending into CPML region
        if cpml_thickness > 0 and self._boundary == "cpml":
            for entry in self._geometry:
                if entry.material_name == "pec":
                    continue
                if hasattr(entry.shape, "bounding_box"):
                    try:
                        c1, c2 = entry.shape.bounding_box()
                        for ax in range(min(3, len(self._domain))):
                            d = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                            if c1[ax] < cpml_thickness * 0.3 or c2[ax] > d - cpml_thickness * 0.3:
                                _w.warn(
                                    f"Material '{entry.material_name}' extends "
                                    f"into CPML region along {'xyz'[ax]}-axis. "
                                    f"{absorber_label} assumes vacuum — dielectric in absorber "
                                    f"causes increased reflections.",
                                    stacklevel=3,
                                )
                                break
                    except (NotImplementedError, TypeError):
                        pass

        # P1.8: Port/source inside PEC geometry
        for pe in self._ports:
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

            # P2.6: CPML z-thickness on non-uniform mesh
            if self._boundary == "cpml" and self._cpml_layers > 0:
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
        pec_mask: jnp.ndarray | None = None,
        pec_occupancy: jnp.ndarray | None = None,
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
            pec_mask=pec_mask_local,
            pec_occupancy=pec_occupancy_local,
            return_state=False,
        )
        return ForwardResult(
            time_series=result.time_series,
            ntff_data=result.ntff_data,
            ntff_box=result.ntff_box,
            grid=result.grid,
            s_params=getattr(result, "s_params", None),
            freqs=getattr(result, "freqs", None),
        )

    def _forward_nonuniform_from_materials(
        self,
        *,
        eps_override: jnp.ndarray | None = None,
        sigma_override: jnp.ndarray | None = None,
        pec_mask_override: jnp.ndarray | None = None,
        n_steps: int,
        checkpoint: bool = True,
        emit_time_series: bool = True,
        checkpoint_every: int | None = None,
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
            checkpoint=checkpoint,
            emit_time_series=emit_time_series,
            checkpoint_every=checkpoint_every,
        )
        return ForwardResult(
            time_series=result.time_series,
            ntff_data=result.ntff_data,
            ntff_box=result.ntff_box,
            grid=result.grid,
            s_params=getattr(result, "s_params", None),
            freqs=getattr(result, "freqs", None),
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
        emit_time_series: bool = True,
        checkpoint_every: int | None = None,
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

        Returns
        -------
        ForwardResult
            Minimal differentiable observables (time series and optional NTFF).
        """
        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )
        if is_nonuniform:
            if pec_occupancy_override is not None:
                raise ValueError(
                    "pec_occupancy_override is not yet supported on the "
                    "non-uniform forward path (run_nonuniform has no soft-PEC "
                    "occupancy field). Use pec_mask_override for hard PEC."
                )
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
                n_steps=n_steps,
                checkpoint=checkpoint,
                emit_time_series=emit_time_series,
                checkpoint_every=checkpoint_every,
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
                "supported on the non-uniform forward path."
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
            pec_mask=pec_mask,
            pec_occupancy=pec_occupancy_override,
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
