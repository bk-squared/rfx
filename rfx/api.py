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
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid, C0
from rfx.core.yee import MaterialArrays, init_materials, EPS_0
from rfx.geometry.csg import Shape, Box, Sphere, Cylinder, rasterize
from rfx.sources.sources import GaussianPulse, LumpedPort, setup_lumped_port
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import LorentzPole, init_lorentz, drude_pole, lorentz_pole
from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor
from rfx.probes.probes import extract_s_matrix
from rfx.simulation import (
    make_source, make_probe, make_port_source,
    run as _run, SimResult, SourceSpec, ProbeSpec,
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
    """
    state: object
    time_series: jnp.ndarray
    s_params: np.ndarray | None
    freqs: np.ndarray | None
    ntff_data: object = None
    ntff_box: object = None


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


@dataclass(frozen=True)
class _ProbeEntry:
    position: tuple[float, float, float]
    component: str


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
        Physical domain size.
    boundary : "pec" or "cpml"
        Boundary condition. Default "cpml".
    cpml_layers : int
        Number of CPML layers per face. Default 10 (ignored for "pec").
    dx : float or None
        Cell size override (metres). Auto-computed if None.
    """

    def __init__(
        self,
        freq_max: float,
        domain: tuple[float, float, float],
        *,
        boundary: str = "cpml",
        cpml_layers: int = 10,
        dx: float | None = None,
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

        # Registered items
        self._materials: dict[str, MaterialSpec] = {}
        self._geometry: list[_GeometryEntry] = []
        self._ports: list[_PortEntry] = []
        self._probes: list[_ProbeEntry] = []
        self._thin_conductors: list[ThinConductor] = []
        self._ntff: tuple | None = None  # (corner_lo, corner_hi, freqs)

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
    ) -> "Simulation":
        """Add a lumped port.

        Parameters
        ----------
        position : (x, y, z) in metres
        component : "ex", "ey", or "ez"
        impedance : port impedance in ohms (default 50)
        waveform : excitation pulse (default: GaussianPulse at freq_max/2)
        """
        if component not in ("ex", "ey", "ez"):
            raise ValueError(f"component must be ex/ey/ez, got {component!r}")
        if impedance <= 0:
            raise ValueError(f"impedance must be positive, got {impedance}")

        if waveform is None:
            waveform = GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)

        self._ports.append(_PortEntry(
            position=position, component=component,
            impedance=impedance, waveform=waveform,
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

    def _build_grid(self) -> Grid:
        return Grid(
            freq_max=self._freq_max,
            domain=self._domain,
            dx=self._dx,
            cpml_layers=self._cpml_layers,
        )

    def _assemble_materials(
        self,
        grid: Grid,
    ) -> tuple[MaterialArrays, _DebyeSpec | None, _LorentzSpec | None]:
        """Build material arrays plus per-pole dispersion masks."""
        # Start with vacuum
        eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
        sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
        mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

        # Collect per-pole masks so distinct materials do not inherit
        # each other's dispersion poles.
        debye_masks_by_pole: dict[DebyePole, jnp.ndarray] = {}
        lorentz_masks_by_pole: dict[LorentzPole, jnp.ndarray] = {}

        for entry in self._geometry:
            mat = self._resolve_material(entry.material_name)
            mask = entry.shape.mask(grid)
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

        return materials, debye_spec, lorentz_spec

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
        materials, debye_spec, lorentz_spec = self._assemble_materials(grid)
        _, debye, lorentz = self._init_dispersion(
            materials, grid.dt, debye_spec, lorentz_spec)
        return materials, debye, lorentz

    # ---- run ----

    def run(
        self,
        *,
        n_steps: int | None = None,
        num_periods: float = 20.0,
        checkpoint: bool = False,
        compute_s_params: bool | None = None,
        s_param_freqs: jnp.ndarray | None = None,
        s_param_n_steps: int | None = None,
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

        Returns
        -------
        Result
        """
        grid = self._build_grid()
        base_materials, debye_spec, lorentz_spec = self._assemble_materials(grid)
        materials = base_materials

        if n_steps is None:
            n_steps = grid.num_timesteps(num_periods=num_periods)

        # Build sources and probes for the compiled runner
        sources: list[SourceSpec] = []
        probes: list[ProbeSpec] = []

        # Port sources — fold impedances into materials first
        lumped_ports: list[LumpedPort] = []
        for pe in self._ports:
            lp = LumpedPort(
                position=pe.position, component=pe.component,
                impedance=pe.impedance, excitation=pe.waveform,
            )
            lumped_ports.append(lp)
            materials = setup_lumped_port(grid, lp, materials)
            sources.append(make_port_source(grid, lp, materials, n_steps))

        for pe in self._probes:
            probes.append(make_probe(grid, pe.position, pe.component))

        _, debye, lorentz = self._init_dispersion(
            materials, grid.dt, debye_spec, lorentz_spec)

        # NTFF box
        ntff_box = None
        if self._ntff is not None:
            corner_lo, corner_hi, freqs = self._ntff
            ntff_box = make_ntff_box(grid, corner_lo, corner_hi, freqs)

        # Main simulation
        sim_result = _run(
            grid, materials, n_steps,
            boundary=self._boundary,
            debye=debye,
            lorentz=lorentz,
            sources=sources,
            probes=probes,
            ntff=ntff_box,
            checkpoint=checkpoint,
        )

        # S-parameters (separate Python-loop simulation for accuracy)
        if compute_s_params is None:
            compute_s_params = len(lumped_ports) > 0

        s_params = None
        freqs_out = None

        if compute_s_params and lumped_ports:
            if s_param_freqs is None:
                s_param_freqs = jnp.linspace(
                    self._freq_max / 10, self._freq_max, 50,
                )
            freqs_out = np.array(s_param_freqs)

            s_params = extract_s_matrix(
                grid, base_materials, lumped_ports, s_param_freqs,
                n_steps=s_param_n_steps,
                boundary=self._boundary,
                debye_spec=debye_spec,
                lorentz_spec=lorentz_spec,
            )

        return Result(
            state=sim_result.state,
            time_series=sim_result.time_series,
            s_params=s_params,
            freqs=freqs_out,
            ntff_data=sim_result.ntff_data,
            ntff_box=ntff_box,
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
            f")"
        )
