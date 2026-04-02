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

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid, C0
from rfx.core.yee import MaterialArrays, init_materials, EPS_0
from rfx.geometry.csg import Shape, Box, Sphere, Cylinder, rasterize
from rfx.sources.sources import GaussianPulse, LumpedPort, setup_lumped_port, WirePort, setup_wire_port
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import LorentzPole, init_lorentz, drude_pole, lorentz_pole
from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor
from rfx.probes.probes import extract_s_matrix, extract_s_matrix_wire, init_dft_plane_probe
from rfx.sources.waveguide_port import (
    WaveguidePort,
    extract_waveguide_s_matrix,
    extract_waveguide_s_params_normalized,
    extract_waveguide_sparams,
    init_waveguide_port,
    waveguide_plane_positions,
)
from rfx.simulation import (
    make_source, make_probe, make_port_source, make_wire_port_sources,
    run as _run, run_until_decay as _run_until_decay,
    SimResult, SourceSpec, ProbeSpec, SnapshotSpec,
)
from rfx.farfield import (
    NTFFBox, make_ntff_box, compute_far_field, FarFieldResult,
)


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
    """
    state: object
    time_series: jnp.ndarray
    s_params: np.ndarray | None
    freqs: np.ndarray | None
    ntff_data: object = None
    ntff_box: object = None
    dft_planes: dict | None = None
    waveguide_ports: dict | None = None
    waveguide_sparams: dict | None = None
    snapshots: dict | None = None


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


@dataclass(frozen=True)
class _DFTPlaneEntry:
    name: str
    axis: str
    coordinate: float
    component: str
    freqs: jnp.ndarray | None
    n_freqs: int


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
    boundary : "pec" or "cpml"
        Boundary condition. Default "cpml".
    cpml_layers : int
        Number of CPML layers per face. Default 10 (ignored for "pec").
    dx : float or None
        Cell size override (metres). Auto-computed if None.
    mode : str
        ``"3d"`` (default), ``"2d_tmz"`` (Ez, Hx, Hy), or
        ``"2d_tez"`` (Hz, Ex, Ey).
    """

    def __init__(
        self,
        freq_max: float,
        domain: tuple[float, float, float],
        *,
        boundary: str = "cpml",
        cpml_layers: int = 10,
        dx: float | None = None,
        mode: str = "3d",
    ):
        if boundary not in ("pec", "cpml"):
            raise ValueError(f"boundary must be 'pec' or 'cpml', got {boundary!r}")
        if freq_max <= 0:
            raise ValueError(f"freq_max must be positive, got {freq_max}")
        if any(d <= 0 for d in domain):
            raise ValueError(f"domain dimensions must be positive, got {domain}")

        self._freq_max = freq_max
        self._domain = domain
        self._boundary = boundary
        self._cpml_layers = cpml_layers if boundary == "cpml" else 0
        self._dx = dx
        self._mode = mode

        # Registered items
        self._materials: dict[str, MaterialSpec] = {}
        self._geometry: list[_GeometryEntry] = []
        self._ports: list[_PortEntry] = []
        self._probes: list[_ProbeEntry] = []
        self._thin_conductors: list[ThinConductor] = []
        self._ntff: tuple | None = None  # (corner_lo, corner_hi, freqs)
        self._tfsf: _TFSFEntry | None = None
        self._dft_planes: list[_DFTPlaneEntry] = []
        self._waveguide_ports: list[_WaveguidePortEntry] = []
        self._periodic_axes: str = ""
        self._refinement: dict | None = None

    # ---- refinement (subgridding) ----

    def add_refinement(
        self,
        z_range: tuple[float, float],
        *,
        ratio: int = 4,
        xy_margin: float | None = None,
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
        """
        if self._refinement is not None:
            raise ValueError("Only one refinement region is supported")
        self._refinement = {
            "z_range": z_range,
            "ratio": ratio,
            "xy_margin": xy_margin,
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
    ) -> "Simulation":
        """Register a named material."""
        self._materials[name] = MaterialSpec(
            eps_r=eps_r, sigma=sigma, mu_r=mu_r,
            debye_poles=debye_poles, lorentz_poles=lorentz_poles,
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
        self._thin_conductors.append(ThinConductor(
            shape=shape, sigma_bulk=sigma_bulk,
            thickness=thickness, eps_r=eps_r,
        ))
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
    ) -> "Simulation":
        """Add a lumped port (single-cell) or wire port (multi-cell).

        Parameters
        ----------
        position : (x, y, z) in metres
        component : "ex", "ey", or "ez"
        impedance : port impedance in ohms (default 50)
        waveform : excitation pulse (default: GaussianPulse at freq_max/2)
        extent : float or None
            When provided, the port spans from *position* along the port
            axis by this distance (metres), creating a multi-cell WirePort.
            For example ``component="ez", extent=0.0015`` spans the port
            from ``z`` to ``z + 0.0015``.
        """
        if self._tfsf is not None:
            raise ValueError(
                "Lumped ports are not supported together with the TFSF plane-wave source"
            )
        if component not in ("ex", "ey", "ez"):
            raise ValueError(f"component must be ex/ey/ez, got {component!r}")
        if impedance <= 0:
            raise ValueError(f"impedance must be positive, got {impedance}")

        if waveform is None:
            waveform = GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)

        self._ports.append(_PortEntry(
            position=position, component=component,
            impedance=impedance, waveform=waveform,
            extent=extent,
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
    ) -> "Simulation":
        """Add a normal-incidence plane-wave TFSF source.

        Current scope is intentionally narrow: x-directed propagation,
        ``ez``/``ey`` polarization, 3D mode, and CPML boundaries. For
        oblique incidence, only the single transverse-axis plane implied
        by the chosen polarization is supported.
        """
        if self._boundary != "cpml":
            raise ValueError("TFSF plane-wave source requires boundary='cpml'")
        if self._cpml_layers <= 0:
            raise ValueError("TFSF plane-wave source requires cpml_layers > 0")
        if self._mode != "3d":
            raise ValueError("TFSF plane-wave source currently supports only mode='3d'")
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

        self._tfsf = _TFSFEntry(
            f0=f0,
            bandwidth=bandwidth,
            amplitude=amplitude,
            margin=margin,
            polarization=polarization,
            direction=direction,
            angle_deg=angle_deg,
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
            )
        return Grid(
            freq_max=self._freq_max,
            domain=self._domain,
            dx=self._dx,
            cpml_layers=self._cpml_layers,
            mode=self._mode,
        )

    # Threshold above which sigma is treated as PEC (use mask instead).
    _PEC_SIGMA_THRESHOLD = 1e6

    def _assemble_materials(
        self,
        grid: Grid,
    ) -> tuple[MaterialArrays, _DebyeSpec | None, _LorentzSpec | None, jnp.ndarray | None]:
        """Build material arrays plus per-pole dispersion masks.

        Returns
        -------
        materials, debye_spec, lorentz_spec, pec_mask
            pec_mask is a boolean array (True at PEC cells) or None.
        """
        # Start with vacuum
        eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
        sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
        mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
        pec_mask = jnp.zeros(grid.shape, dtype=jnp.bool_)

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
            else:
                eps_r = jnp.where(mask, mat.eps_r, eps_r)
                sigma = jnp.where(mask, mat.sigma, sigma)
                mu_r = jnp.where(mask, mat.mu_r, mu_r)

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

        materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

        # Apply thin conductors
        for tc in self._thin_conductors:
            materials = apply_thin_conductor(grid, tc, materials)

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
        return materials, debye_spec, lorentz_spec, pec_mask if has_pec else None

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
        materials, debye_spec, lorentz_spec, _ = self._assemble_materials(grid)
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
        """
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
        base_materials, debye_spec, lorentz_spec, _ = self._assemble_materials(grid)
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

        cfgs = [self._build_waveguide_port_config(entry, grid, freqs, n_steps) for entry in entries]

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
        """Run simulation using SBP-SAT subgridding."""
        from rfx.subgridding.sbp_sat_3d import SubgridConfig3D, SubgridState3D
        from rfx.subgridding.runner import run_subgridded as _run_sg

        ref = self._refinement
        ratio = ref["ratio"]
        z_lo, z_hi = ref["z_range"]
        dx_c = grid_coarse.dx
        dx_f = dx_c / ratio
        xy_margin = ref["xy_margin"] if ref["xy_margin"] is not None else 2 * dx_c

        # Map z_range to coarse grid indices
        cpml = grid_coarse.cpml_layers
        fk_lo = max(int(round(z_lo / dx_c)) + cpml, cpml)
        fk_hi = min(int(round(z_hi / dx_c)) + cpml + 1, grid_coarse.nz - cpml)

        # Fine region covers full x,y for simplicity (with cpml margin)
        fi_lo = cpml
        fi_hi = grid_coarse.nx - cpml
        fj_lo = cpml
        fj_hi = grid_coarse.ny - cpml

        # Fine grid dimensions
        nx_f = (fi_hi - fi_lo) * ratio
        ny_f = (fj_hi - fj_lo) * ratio
        nz_f = (fk_hi - fk_lo) * ratio

        # Global timestep (limited by fine grid CFL)
        import numpy as np_
        from rfx.core.yee import EPS_0, MU_0
        C0_val = 1.0 / np_.sqrt(float(EPS_0) * float(MU_0))
        dt = 0.45 * dx_f / (C0_val * np_.sqrt(3))

        config = SubgridConfig3D(
            nx_c=grid_coarse.nx, ny_c=grid_coarse.ny, nz_c=grid_coarse.nz,
            dx_c=dx_c,
            fi_lo=fi_lo, fi_hi=fi_hi,
            fj_lo=fj_lo, fj_hi=fj_hi,
            fk_lo=fk_lo, fk_hi=fk_hi,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
            dx_f=dx_f, dt=float(dt), ratio=ratio, tau=0.5,
        )

        # Build fine-grid materials by rasterizing geometry at fine resolution
        shape_f = (nx_f, ny_f, nz_f)

        # Create a Grid for fine region (for position_to_index utility)
        fine_domain = (nx_f * dx_f, ny_f * dx_f, nz_f * dx_f)
        fine_grid = Grid(
            freq_max=self._freq_max,
            domain=fine_domain,
            dx=dx_f,
            cpml_layers=0,
        )
        # Override shape to match exactly (Grid may add +1 rounding)
        fine_grid._shape_override = shape_f

        # Rasterize geometry into fine grid materials
        eps_r_f = jnp.ones(shape_f, dtype=jnp.float32)
        sigma_f = jnp.zeros(shape_f, dtype=jnp.float32)
        mu_r_f = jnp.ones(shape_f, dtype=jnp.float32)
        pec_mask_f = jnp.zeros(shape_f, dtype=jnp.bool_)

        # Offset: fine grid origin in physical coords
        x_off = (fi_lo - cpml) * dx_c
        y_off = (fj_lo - cpml) * dx_c
        z_off = (fk_lo - cpml) * dx_c

        for entry in self._geometry:
            mat = self._resolve_material(entry.material_name)
            shape = entry.shape
            if hasattr(shape, 'corner1') and hasattr(shape, 'corner2'):
                from rfx.geometry.csg import Box
                # Offset shape to fine grid local coordinates
                c1 = shape.corner1
                c2 = shape.corner2
                # Map physical coords to fine grid indices
                i0 = max(0, int(round((c1[0] - x_off) / dx_f)))
                i1 = min(nx_f, int(round((c2[0] - x_off) / dx_f)))
                j0 = max(0, int(round((c1[1] - y_off) / dx_f)))
                j1 = min(ny_f, int(round((c2[1] - y_off) / dx_f)))
                k0 = max(0, int(round((c1[2] - z_off) / dx_f)))
                k1 = min(nz_f, int(round((c2[2] - z_off) / dx_f)))
                if i0 < i1 and j0 < j1 and k0 < k1:
                    mask = jnp.zeros(shape_f, dtype=jnp.bool_)
                    mask = mask.at[i0:i1, j0:j1, k0:k1].set(True)
                    if mat.sigma >= self._PEC_SIGMA_THRESHOLD:
                        pec_mask_f = pec_mask_f | mask
                    else:
                        eps_r_f = jnp.where(mask, mat.eps_r, eps_r_f)
                        sigma_f = jnp.where(mask, mat.sigma, sigma_f)
                        mu_r_f = jnp.where(mask, mat.mu_r, mu_r_f)

        mats_f = MaterialArrays(eps_r=eps_r_f, sigma=sigma_f, mu_r=mu_r_f)
        has_pec_f = bool(jnp.any(pec_mask_f))

        # Helper: convert physical position to fine-grid index
        def _pos_to_fine_idx(pos):
            return (
                int(round((pos[0] - x_off) / dx_f)),
                int(round((pos[1] - y_off) / dx_f)),
                int(round((pos[2] - z_off) / dx_f)),
            )

        # Build sources on fine grid
        import jax
        sources_f = []
        times = jnp.arange(n_steps, dtype=jnp.float32) * dt

        for pe in self._ports:
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            axis = axis_map[pe.component]

            if pe.extent is not None:
                # Wire port: compute cells manually
                pos_f = (pe.position[0] - x_off, pe.position[1] - y_off,
                         pe.position[2] - z_off)
                idx_start = _pos_to_fine_idx(pe.position)
                end_pos = list(pe.position)
                end_pos[axis] += pe.extent
                idx_end = _pos_to_fine_idx(tuple(end_pos))

                lo = min(idx_start[axis], idx_end[axis])
                hi = max(idx_start[axis], idx_end[axis])
                cells = []
                for a in range(lo, hi + 1):
                    cell = list(idx_start)
                    cell[axis] = a
                    cells.append(tuple(cell))

                n_cells = max(len(cells), 1)
                # Distribute port impedance
                sigma_port_per_cell = n_cells / (pe.impedance * dx_f)
                for cell in cells:
                    i, j, k = cell
                    mats_f = mats_f._replace(
                        sigma=mats_f.sigma.at[i, j, k].add(sigma_port_per_cell))
                    pec_mask_f = pec_mask_f.at[i, j, k].set(False)

                # Precompute Cb-corrected waveforms
                for cell in cells:
                    i, j, k = cell
                    eps = float(mats_f.eps_r[i, j, k]) * EPS_0
                    sigma_val = float(mats_f.sigma[i, j, k])
                    loss = sigma_val * dt / (2.0 * eps)
                    cb = (dt / eps) / (1.0 + loss)
                    waveform = (cb / dx_f) * jax.vmap(pe.waveform)(times) / n_cells
                    sources_f.append((i, j, k, pe.component, np.array(waveform)))
            else:
                # Lumped port
                idx = _pos_to_fine_idx(pe.position)
                i, j, k = idx
                sigma_port = 1.0 / (pe.impedance * dx_f)
                mats_f = mats_f._replace(
                    sigma=mats_f.sigma.at[i, j, k].add(sigma_port))
                pec_mask_f = pec_mask_f.at[i, j, k].set(False)

                eps = float(mats_f.eps_r[i, j, k]) * EPS_0
                sigma_val = float(mats_f.sigma[i, j, k])
                loss = sigma_val * dt / (2.0 * eps)
                cb = (dt / eps) / (1.0 + loss)
                waveform = (cb / dx_f) * jax.vmap(pe.waveform)(times)
                sources_f.append((i, j, k, pe.component, np.array(waveform)))

        # Build probes on fine grid
        probe_indices_f = []
        probe_components = []
        for pe in self._probes:
            idx = _pos_to_fine_idx(pe.position)
            probe_indices_f.append(idx)
            probe_components.append(pe.component)

        result = _run_sg(
            grid_coarse,
            base_materials_coarse,
            None,  # fine_grid not used directly
            mats_f,
            config,
            n_steps,
            pec_mask_c=pec_mask_coarse,
            pec_mask_f=pec_mask_f if has_pec_f else None,
            sources_f=sources_f,
            probe_indices_f=probe_indices_f,
            probe_components=probe_components,
        )

        return Result(
            state=result["state_f"],
            time_series=result["time_series"],
            s_params=None,
            freqs=None,
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

        Returns
        -------
        Result
        """
        grid = self._build_grid()
        base_materials, debye_spec, lorentz_spec, pec_mask = self._assemble_materials(grid)
        materials = base_materials

        # ---- Subgridded path ----
        if self._refinement is not None:
            return self._run_subgridded(
                grid, base_materials, pec_mask,
                n_steps=n_steps or grid.num_timesteps(num_periods=num_periods),
            )

        # Compute per-component smoothed permittivity if requested
        aniso_eps = None
        if subpixel_smoothing:
            from rfx.geometry.smoothing import compute_smoothed_eps
            shape_eps_pairs = [
                (entry.shape, self._resolve_material(entry.material_name).eps_r)
                for entry in self._geometry
            ]
            if shape_eps_pairs:
                aniso_eps = compute_smoothed_eps(grid, shape_eps_pairs, background_eps=1.0)

        if n_steps is None:
            n_steps = grid.num_timesteps(num_periods=num_periods)

        # Build sources and probes for the compiled runner
        sources: list[SourceSpec] = []
        probes: list[ProbeSpec] = []
        dft_planes = []
        waveguide_ports = []
        periodic = None
        cpml_axes = "xyz"
        pec_axes = None
        tfsf = None

        if self._tfsf is not None and self._ports:
            raise ValueError(
                "TFSF plane-wave source is not supported together with lumped ports"
            )
        if self._waveguide_ports and (self._ports or self._tfsf):
            raise ValueError(
                "Waveguide ports are not supported together with lumped ports or TFSF"
            )
        if len(self._waveguide_ports) > 1:
            raise ValueError(
                "Simulation.run() supports only a single waveguide port; use compute_waveguide_s_matrix() for the multiport waveguide scattering workflow"
            )
        if self._periodic_axes:
            periodic = tuple(axis in self._periodic_axes for axis in "xyz")

        # Port sources — fold impedances into materials first
        lumped_ports: list[LumpedPort] = []
        wire_ports: list[WirePort] = []
        for pe in self._ports:
            if pe.extent is not None:
                # Multi-cell wire port
                axis_map = {"ex": 0, "ey": 1, "ez": 2}
                axis = axis_map[pe.component]
                end = list(pe.position)
                end[axis] += pe.extent
                wp = WirePort(
                    start=pe.position, end=tuple(end),
                    component=pe.component,
                    impedance=pe.impedance, excitation=pe.waveform,
                )
                wire_ports.append(wp)
                materials = setup_wire_port(grid, wp, materials)
                sources.extend(make_wire_port_sources(grid, wp, materials, n_steps))
                # Clear PEC mask at wire cells (probe pierces ground plane)
                if pec_mask is not None:
                    from rfx.sources.sources import _wire_port_cells
                    for cell in _wire_port_cells(grid, wp):
                        pec_mask = pec_mask.at[cell[0], cell[1], cell[2]].set(False)
            else:
                # Single-cell lumped port
                lp = LumpedPort(
                    position=pe.position, component=pe.component,
                    impedance=pe.impedance, excitation=pe.waveform,
                )
                lumped_ports.append(lp)
                materials = setup_lumped_port(grid, lp, materials)
                sources.append(make_port_source(grid, lp, materials, n_steps))
                # Clear PEC mask at lumped port cell
                if pec_mask is not None:
                    idx = grid.position_to_index(pe.position)
                    pec_mask = pec_mask.at[idx[0], idx[1], idx[2]].set(False)

        for pe in self._probes:
            probes.append(make_probe(grid, pe.position, pe.component))

        axis_to_index = {"x": 0, "y": 1, "z": 2}
        for pe in self._dft_planes:
            axis_idx = axis_to_index[pe.axis]
            plane_pos = [0.0, 0.0, 0.0]
            plane_pos[axis_idx] = pe.coordinate
            grid_index = grid.position_to_index(tuple(plane_pos))[axis_idx]
            freqs_arr = (
                pe.freqs
                if pe.freqs is not None
                else jnp.linspace(self._freq_max / 10, self._freq_max, pe.n_freqs)
            )
            dft_planes.append(
                init_dft_plane_probe(
                    axis=axis_idx,
                    index=grid_index,
                    component=pe.component,
                    freqs=freqs_arr,
                    grid_shape=grid.shape,
                    dft_total_steps=n_steps,
                )
            )

        if self._waveguide_ports:
            cpml_axes = grid.cpml_axes
            pec_axes = "".join(axis for axis in "xyz" if axis not in cpml_axes)
            for pe in self._waveguide_ports:
                freqs_arr = (
                    pe.freqs
                    if pe.freqs is not None
                    else jnp.linspace(self._freq_max / 10, self._freq_max, pe.n_freqs)
                )
                waveguide_ports.append(self._build_waveguide_port_config(pe, grid, freqs_arr, n_steps))

        if self._tfsf is not None:
            from rfx.sources.tfsf import init_tfsf

            tfsf = init_tfsf(
                grid.nx,
                grid.dx,
                grid.dt,
                cpml_layers=grid.cpml_layers,
                tfsf_margin=self._tfsf.margin,
                f0=self._tfsf.f0 if self._tfsf.f0 is not None else self._freq_max / 2,
                bandwidth=self._tfsf.bandwidth,
                amplitude=self._tfsf.amplitude,
                polarization=self._tfsf.polarization,
                direction=self._tfsf.direction,
                angle_deg=self._tfsf.angle_deg,
            )
            self._validate_tfsf_vacuum_boundary(materials, tfsf[0])
            periodic = (False, True, True)
            cpml_axes = "x"

        _, debye, lorentz = self._init_dispersion(
            materials, grid.dt, debye_spec, lorentz_spec)

        # NTFF box
        ntff_box = None
        if self._ntff is not None:
            corner_lo, corner_hi, freqs = self._ntff
            ntff_box = make_ntff_box(grid, corner_lo, corner_hi, freqs)

        # Main simulation
        if until_decay is not None:
            sim_result = _run_until_decay(
                grid, materials,
                decay_by=until_decay,
                check_interval=decay_check_interval,
                min_steps=decay_min_steps,
                max_steps=decay_max_steps,
                monitor_component=decay_monitor_component,
                monitor_position=decay_monitor_position,
                boundary=self._boundary,
                cpml_axes=cpml_axes,
                pec_axes=pec_axes,
                periodic=periodic,
                debye=debye,
                lorentz=lorentz,
                tfsf=tfsf,
                sources=sources,
                probes=probes,
                dft_planes=dft_planes,
                waveguide_ports=waveguide_ports,
                ntff=ntff_box,
                snapshot=snapshot,
                checkpoint=checkpoint,
                aniso_eps=aniso_eps,
                pec_mask=pec_mask,
            )
        else:
            sim_result = _run(
                grid, materials, n_steps,
                boundary=self._boundary,
                cpml_axes=cpml_axes,
                pec_axes=pec_axes,
                periodic=periodic,
                debye=debye,
                lorentz=lorentz,
                tfsf=tfsf,
                sources=sources,
                probes=probes,
                dft_planes=dft_planes,
                waveguide_ports=waveguide_ports,
                ntff=ntff_box,
                snapshot=snapshot,
                checkpoint=checkpoint,
                aniso_eps=aniso_eps,
                pec_mask=pec_mask,
            )

        # S-parameters (separate Python-loop simulation for accuracy)
        if compute_s_params is None:
            compute_s_params = len(lumped_ports) > 0 or len(wire_ports) > 0

        s_params = None
        freqs_out = None

        if compute_s_params and (lumped_ports or wire_ports):
            if s_param_freqs is None:
                s_param_freqs = jnp.linspace(
                    self._freq_max / 10, self._freq_max, 50,
                )
            freqs_out = np.array(s_param_freqs)

            if wire_ports:
                s_params = extract_s_matrix_wire(
                    grid, base_materials, wire_ports, s_param_freqs,
                    n_steps=s_param_n_steps,
                    boundary=self._boundary,
                    debye_spec=debye_spec,
                    lorentz_spec=lorentz_spec,
                    pec_mask=pec_mask,
                )
            else:
                s_params = extract_s_matrix(
                    grid, base_materials, lumped_ports, s_param_freqs,
                    n_steps=s_param_n_steps,
                    boundary=self._boundary,
                    debye_spec=debye_spec,
                    lorentz_spec=lorentz_spec,
                )

        waveguide_ports_result = (
            {
                entry.name: cfg
                for entry, cfg in zip(self._waveguide_ports, sim_result.waveguide_ports or ())
            }
            if self._waveguide_ports
            else None
        )
        waveguide_sparams_result = None
        if self._waveguide_ports:
            waveguide_sparams_result = {}
            for entry, cfg in zip(self._waveguide_ports, sim_result.waveguide_ports or ()):
                plane_positions = waveguide_plane_positions(cfg)
                source_plane = plane_positions["source"]
                measured_reference_plane = plane_positions["reference"]
                measured_probe_plane = plane_positions["probe"]
                if entry.calibration_preset == "source_to_probe":
                    reference_plane = source_plane
                    probe_plane = measured_probe_plane
                    calibration_preset = "source_to_probe"
                elif entry.reference_plane is not None or entry.probe_plane is not None:
                    reference_plane = (
                        entry.reference_plane
                        if entry.reference_plane is not None
                        else measured_reference_plane
                    )
                    probe_plane = (
                        entry.probe_plane
                        if entry.probe_plane is not None
                        else measured_probe_plane
                    )
                    calibration_preset = "explicit"
                else:
                    reference_plane = measured_reference_plane
                    probe_plane = measured_probe_plane
                    calibration_preset = "measured"
                s11, s21 = extract_waveguide_sparams(
                    cfg,
                    ref_shift=reference_plane - measured_reference_plane,
                    probe_shift=probe_plane - measured_probe_plane,
                )
                waveguide_sparams_result[entry.name] = WaveguideSParamResult(
                    freqs=np.array(cfg.freqs),
                    s11=np.array(s11),
                    s21=np.array(s21),
                    calibration_preset=calibration_preset,
                    source_plane=float(source_plane),
                    measured_reference_plane=measured_reference_plane,
                    measured_probe_plane=measured_probe_plane,
                    reference_plane=reference_plane,
                    probe_plane=probe_plane,
                )

        return Result(
            state=sim_result.state,
            time_series=sim_result.time_series,
            s_params=s_params,
            freqs=freqs_out,
            ntff_data=sim_result.ntff_data,
            ntff_box=ntff_box,
            dft_planes=(
                {
                    entry.name: probe
                    for entry, probe in zip(self._dft_planes, sim_result.dft_planes or ())
                }
                if self._dft_planes
                else None
            ),
            waveguide_ports=waveguide_ports_result,
            waveguide_sparams=waveguide_sparams_result,
            snapshots=sim_result.snapshots,
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
            f"  periodic_axes={self._periodic_axes!r},\n"
            f"  tfsf={self._tfsf is not None},\n"
            f")"
        )
