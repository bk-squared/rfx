"""RIS (Reconfigurable Intelligent Surface) unit cell design workflow.

Wraps rfx's Floquet port + periodic BC infrastructure to automate
RIS unit cell characterisation: sweep varactor bias or scan angle and
extract reflection phase / amplitude.

Typical usage
-------------
>>> cell = RISUnitCell(
...     cell_size=(10e-3, 10e-3),
...     substrate_thickness=1.5e-3,
...     substrate_material="fr4",
...     freq_range=(4e9, 8e9),
... )
>>> cell.add_element(Box((3e-3, 3e-3, z), (7e-3, 7e-3, z)), material="pec")
>>> cell.add_varactor((5e-3, 5e-3), capacitance_range=(0.1e-12, 1.0e-12))
>>> result = cell.sweep_capacitance([0.1e-12, 0.5e-12, 1.0e-12])
>>> cell.plot_phase_diagram(result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from rfx.api import Simulation
from rfx.geometry.csg import Box


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RISSweepResult:
    """Structured result from an RIS parameter sweep.

    Attributes
    ----------
    phases : (n_params, n_freqs) ndarray
        Reflection phase in degrees.
    amplitudes : (n_params, n_freqs) ndarray
        Reflection amplitude (linear, 0-1).
    freqs : (n_freqs,) ndarray
        Frequency points in Hz.
    capacitances : (n_params,) ndarray or None
        Swept capacitance values (None for angle sweeps).
    angles : (n_params,) ndarray or None
        Swept scan angles in degrees (None for capacitance sweeps).
    """

    phases: np.ndarray
    amplitudes: np.ndarray
    freqs: np.ndarray
    capacitances: np.ndarray | None = None
    angles: np.ndarray | None = None

    @property
    def phase_range_deg(self) -> float:
        """Maximum achievable phase range across sweep at any frequency."""
        if self.phases.shape[0] < 2:
            return 0.0
        phase_span = np.ptp(self.phases, axis=0)  # per frequency
        return float(np.max(phase_span))


# ---------------------------------------------------------------------------
# Internal element descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ElementEntry:
    """Conducting element on the unit cell surface."""
    shape: Box
    material: str


@dataclass(frozen=True)
class _VaractorEntry:
    """Tunable varactor across a gap in the unit cell."""
    position: tuple[float, float]
    capacitance_range: tuple[float, float]
    component: str  # "ex" or "ey" — field component the varactor bridges


# ---------------------------------------------------------------------------
# RISUnitCell
# ---------------------------------------------------------------------------

class RISUnitCell:
    """RIS unit cell simulation wrapper.

    Automates: periodic BC + Floquet port + sweep bias/geometry
    to produce reflection phase / amplitude curves.

    Parameters
    ----------
    cell_size : (Lx, Ly) in metres
        Unit cell lateral dimensions.
    substrate_thickness : float
        Substrate thickness in metres.
    substrate_material : str
        Substrate material name (must be in ``MATERIAL_LIBRARY`` or
        registered as custom). Default ``"fr4"``.
    freq_range : (f_min, f_max) in Hz
        Frequency range for analysis.
    n_freqs : int
        Number of frequency points. Default 21.
    scan_angles : list of (theta, phi) or None
        Default scan angles for characterisation. If None, broadside only.
    polarization : str
        ``"te"`` or ``"tm"``. Default ``"te"``.
    n_steps : int
        Number of FDTD timesteps per simulation. Default 300.
    dx : float or None
        Cell size override. Auto-computed if None.
    cpml_layers : int
        CPML layers on the port-normal axis. Default 8.
    ground_plane : bool
        Whether to include a PEC ground plane below the substrate.
        Default True (standard RIS backed by metal).
    """

    def __init__(
        self,
        *,
        cell_size: tuple[float, float],
        substrate_thickness: float,
        substrate_material: str = "fr4",
        freq_range: tuple[float, float],
        n_freqs: int = 21,
        scan_angles: list[tuple[float, float]] | None = None,
        polarization: str = "te",
        n_steps: int = 300,
        dx: float | None = None,
        cpml_layers: int = 8,
        ground_plane: bool = True,
    ):
        if len(cell_size) != 2 or any(d <= 0 for d in cell_size):
            raise ValueError(f"cell_size must be two positive numbers, got {cell_size}")
        if substrate_thickness <= 0:
            raise ValueError(f"substrate_thickness must be positive, got {substrate_thickness}")
        if len(freq_range) != 2 or freq_range[0] >= freq_range[1]:
            raise ValueError(f"freq_range must be (f_min, f_max) with f_min < f_max, got {freq_range}")
        if polarization not in ("te", "tm"):
            raise ValueError(f"polarization must be 'te' or 'tm', got {polarization!r}")

        self._cell_size = cell_size
        self._substrate_thickness = substrate_thickness
        self._substrate_material = substrate_material
        self._freq_range = freq_range
        self._n_freqs = n_freqs
        self._scan_angles = scan_angles or [(0.0, 0.0)]
        self._polarization = polarization
        self._n_steps = n_steps
        self._dx = dx
        self._cpml_layers = cpml_layers
        self._ground_plane = ground_plane

        self._elements: list[_ElementEntry] = []
        self._varactors: list[_VaractorEntry] = []

    # ---- builders ----

    def add_element(self, shape: Box, material: str = "pec") -> "RISUnitCell":
        """Add a conducting element (patch, cross, etc.) to the unit cell.

        Parameters
        ----------
        shape : Box
            Geometry of the element. Coordinates are in the unit cell
            frame: x in [0, Lx], y in [0, Ly], z relative to the
            substrate top surface.
        material : str
            Material name. Default ``"pec"``.
        """
        self._elements.append(_ElementEntry(shape=shape, material=material))
        return self

    def add_varactor(
        self,
        position: tuple[float, float],
        capacitance_range: tuple[float, float],
        component: str = "ez",
    ) -> "RISUnitCell":
        """Add a tunable varactor for reconfigurability.

        The varactor is modelled as a lumped capacitor bridging a gap
        at the given (x, y) position on the substrate surface.

        Parameters
        ----------
        position : (x, y) in metres
            Location in the unit cell plane.
        capacitance_range : (C_min, C_max) in farads
            Tuning range of the varactor diode.
        component : str
            E-field component the varactor bridges. Default ``"ez"``.
        """
        if len(capacitance_range) != 2 or capacitance_range[0] >= capacitance_range[1]:
            raise ValueError(
                f"capacitance_range must be (C_min, C_max) with C_min < C_max, "
                f"got {capacitance_range}"
            )
        self._varactors.append(_VaractorEntry(
            position=position,
            capacitance_range=capacitance_range,
            component=component,
        ))
        return self

    # ---- internal simulation builder ----

    def _build_sim(
        self,
        capacitance_override: float | None = None,
        theta: float = 0.0,
        phi: float = 0.0,
    ) -> Simulation:
        """Build a fully configured ``Simulation`` for one parameter point."""
        Lx, Ly = self._cell_size
        h_sub = self._substrate_thickness
        # Z domain: ground plane at z=0, substrate [0, h_sub],
        # elements at z=h_sub, plus space above for CPML.
        # Use enough vertical space for wave propagation + CPML.
        freq_max = self._freq_range[1]
        c0 = 299_792_458.0
        lam_min = c0 / freq_max
        z_total = h_sub + max(lam_min, 4 * h_sub)

        sim = Simulation(
            freq_max=freq_max,
            domain=(Lx, Ly, z_total),
            boundary="cpml",
            cpml_layers=self._cpml_layers,
            dx=self._dx,
        )

        # Substrate
        sim.add_material("substrate", eps_r=_substrate_eps(self._substrate_material))
        sim.add(
            Box((0, 0, 0), (Lx, Ly, h_sub)),
            material="substrate",
        )

        # Ground plane (PEC at z=0)
        if self._ground_plane:
            sim.add(
                Box((0, 0, 0), (Lx, Ly, 0)),
                material="pec",
            )

        # User elements
        for elem in self._elements:
            sim.add(elem.shape, material=elem.material)

        # Varactors as substrate permittivity modulation.
        #
        # In a real RIS, the varactor changes the effective capacitance
        # of the unit cell, which shifts the resonant frequency and
        # thus the reflection phase.  In FDTD this is modelled by
        # varying the substrate permittivity.  The mapping is:
        #
        #     eps_eff = eps_substrate + C / (eps_0 * h_sub)
        #
        # where the added capacitance per unit area augments the
        # substrate's effective permittivity. This produces the desired
        # resonance shift without requiring lumped-element placement
        # that could collide with PEC cells.
        if self._varactors:
            from rfx.core.yee import EPS_0
            C = capacitance_override if capacitance_override is not None else self._varactors[0].capacitance_range[0]
            eps_base = _substrate_eps(self._substrate_material)
            dx_est = sim._dx or (c0 / freq_max / 20.0)
            # Capacitance contribution to substrate permittivity
            eps_add = C / (EPS_0 * dx_est)
            eps_loaded = eps_base + eps_add
            # Re-register the substrate material with the loaded value
            sim._materials["substrate"] = sim._materials["substrate"].__class__(
                eps_r=eps_loaded
            )

        # Floquet port above the structure
        port_z = h_sub + (z_total - h_sub) * 0.4
        f_center = (self._freq_range[0] + self._freq_range[1]) / 2.0
        sim.add_floquet_port(
            port_z,
            axis="z",
            scan_theta=theta,
            scan_phi=phi,
            polarization=self._polarization,
            n_freqs=self._n_freqs,
            f0=f_center,
        )

        # Probe above the patch
        probe_z = h_sub + (z_total - h_sub) * 0.3
        component = "ex" if self._polarization == "te" else "ey"
        sim.add_probe((Lx / 2, Ly / 2, probe_z), component=component)

        return sim

    # ---- sweep methods ----

    def sweep_capacitance(
        self,
        values: Sequence[float],
        freq: np.ndarray | None = None,
    ) -> RISSweepResult:
        """Sweep varactor capacitance and extract reflection phase/amplitude.

        Parameters
        ----------
        values : sequence of float
            Capacitance values in farads to sweep over.
        freq : array or None
            Frequency points. If None, auto-generated from freq_range.

        Returns
        -------
        RISSweepResult
        """
        values = list(values)
        if not values:
            raise ValueError("values must not be empty")
        if not self._varactors:
            raise ValueError(
                "No varactors added. Use add_varactor() before sweeping capacitance."
            )

        all_phases = []
        all_amps = []
        result_freqs = None

        for cap_val in values:
            sim = self._build_sim(capacitance_override=cap_val)
            result = sim.run(n_steps=self._n_steps)

            s11, freqs = _extract_reflection(result, self._freq_range, self._n_freqs)
            phase_deg = np.angle(s11, deg=True)
            amplitude = np.abs(s11)

            all_phases.append(phase_deg)
            all_amps.append(amplitude)
            if result_freqs is None:
                result_freqs = freqs

        return RISSweepResult(
            phases=np.array(all_phases),
            amplitudes=np.array(all_amps),
            freqs=result_freqs,
            capacitances=np.array(values),
        )

    def sweep_angle(
        self,
        theta_values: Sequence[float],
        freq: np.ndarray | None = None,
        phi: float = 0.0,
    ) -> RISSweepResult:
        """Sweep scan angle and extract reflection vs angle.

        Parameters
        ----------
        theta_values : sequence of float
            Scan angles theta in degrees to sweep over.
        freq : array or None
            Frequency points. If None, auto-generated from freq_range.
        phi : float
            Azimuth angle in degrees. Default 0.

        Returns
        -------
        RISSweepResult
        """
        theta_values = list(theta_values)
        if not theta_values:
            raise ValueError("theta_values must not be empty")

        all_phases = []
        all_amps = []
        result_freqs = None
        cap_val = self._varactors[0].capacitance_range[0] if self._varactors else None

        for theta in theta_values:
            sim = self._build_sim(
                capacitance_override=cap_val,
                theta=theta,
                phi=phi,
            )
            result = sim.run(n_steps=self._n_steps)

            s11, freqs = _extract_reflection(result, self._freq_range, self._n_freqs)
            phase_deg = np.angle(s11, deg=True)
            amplitude = np.abs(s11)

            all_phases.append(phase_deg)
            all_amps.append(amplitude)
            if result_freqs is None:
                result_freqs = freqs

        return RISSweepResult(
            phases=np.array(all_phases),
            amplitudes=np.array(all_amps),
            freqs=result_freqs,
            angles=np.array(theta_values),
        )

    # ---- visualization ----

    def plot_phase_diagram(self, result: RISSweepResult) -> object:
        """Plot reflection phase vs frequency vs bias state.

        Parameters
        ----------
        result : RISSweepResult
            Output of ``sweep_capacitance`` or ``sweep_angle``.

        Returns
        -------
        matplotlib Figure
        """
        if not HAS_MPL:
            raise ImportError("matplotlib is required for RIS visualization")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        freqs_ghz = result.freqs / 1e9

        # Phase plot
        ax_phase = axes[0]
        for i in range(result.phases.shape[0]):
            if result.capacitances is not None:
                label = f"C = {result.capacitances[i]*1e12:.2f} pF"
            elif result.angles is not None:
                label = f"theta = {result.angles[i]:.1f} deg"
            else:
                label = f"state {i}"
            ax_phase.plot(freqs_ghz, result.phases[i], label=label)
        ax_phase.set_xlabel("Frequency (GHz)")
        ax_phase.set_ylabel("Reflection Phase (deg)")
        ax_phase.set_title("Reflection Phase")
        ax_phase.legend(fontsize=8)
        ax_phase.grid(True, alpha=0.3)

        # Amplitude plot
        ax_amp = axes[1]
        for i in range(result.amplitudes.shape[0]):
            amp_db = 20.0 * np.log10(np.maximum(result.amplitudes[i], 1e-10))
            if result.capacitances is not None:
                label = f"C = {result.capacitances[i]*1e12:.2f} pF"
            elif result.angles is not None:
                label = f"theta = {result.angles[i]:.1f} deg"
            else:
                label = f"state {i}"
            ax_amp.plot(freqs_ghz, amp_db, label=label)
        ax_amp.set_xlabel("Frequency (GHz)")
        ax_amp.set_ylabel("Reflection Amplitude (dB)")
        ax_amp.set_title("Reflection Amplitude")
        ax_amp.legend(fontsize=8)
        ax_amp.grid(True, alpha=0.3)

        fig.suptitle(
            f"RIS Unit Cell: {self._cell_size[0]*1e3:.1f} x {self._cell_size[1]*1e3:.1f} mm",
            fontsize=13,
        )
        fig.tight_layout()
        return fig

    # ---- repr ----

    def __repr__(self) -> str:
        return (
            f"RISUnitCell(\n"
            f"  cell_size=({self._cell_size[0]*1e3:.1f}, {self._cell_size[1]*1e3:.1f}) mm,\n"
            f"  substrate={self._substrate_material} {self._substrate_thickness*1e3:.2f} mm,\n"
            f"  freq_range=({self._freq_range[0]/1e9:.1f}, {self._freq_range[1]/1e9:.1f}) GHz,\n"
            f"  elements={len(self._elements)},\n"
            f"  varactors={len(self._varactors)},\n"
            f"  polarization={self._polarization!r},\n"
            f"  ground_plane={self._ground_plane},\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _substrate_eps(material: str) -> float:
    """Resolve substrate material to a relative permittivity value."""
    from rfx.api import MATERIAL_LIBRARY
    if material in MATERIAL_LIBRARY:
        return MATERIAL_LIBRARY[material]["eps_r"]
    # Common RIS substrates not in the default library
    _EXTRA = {
        "rogers5880": 2.2,
        "rogers4350b": 3.66,
        "rogers3003": 3.0,
        "duroid": 2.2,
    }
    if material in _EXTRA:
        return _EXTRA[material]
    raise ValueError(
        f"Unknown substrate material {material!r}. "
        f"Known: {sorted(list(MATERIAL_LIBRARY.keys()) + list(_EXTRA.keys()))}"
    )


def _extract_reflection(
    result,
    freq_range: tuple[float, float],
    n_freqs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract S11 reflection from a simulation result.

    Tries Floquet S-params first; falls back to time-domain FFT
    of the probe recording.

    Returns
    -------
    s11 : (n_freqs,) complex array
    freqs : (n_freqs,) float array
    """
    # If the result has S-parameters from ports (rare for Floquet-only),
    # use them.
    if result.s_params is not None and result.freqs is not None:
        s11 = np.asarray(result.s_params)[0, 0, :]
        freqs = np.asarray(result.freqs)
        # Filter to freq_range
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if np.sum(mask) >= 2:
            return s11[mask], freqs[mask]

    # Fallback: FFT of probe time series
    ts = np.asarray(result.time_series).ravel()
    if result.dt is not None:
        dt = result.dt
    else:
        # Estimate dt from the simulation parameters
        dt = 1.0 / (freq_range[1] * 40)  # rough estimate

    n = len(ts)
    freqs_fft = np.fft.rfftfreq(n, d=dt)
    spectrum = np.fft.rfft(ts)

    # Interpolate to desired frequency points
    target_freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    s11 = np.interp(target_freqs, freqs_fft, spectrum, left=0, right=0)

    # Normalize by peak
    peak = np.max(np.abs(s11))
    if peak > 0:
        s11 = s11 / peak

    return s11, target_freqs
