"""Leaf data structures for the rfx high-level API.

Module-level data containers and named material library shared by the
:class:`rfx.api.Simulation` builder.  This is a **leaf** module: it must
import only external ``rfx.*`` submodules / stdlib / jax / numpy.

IMPORT CONTRACT
---------------
NEVER write ``from rfx.api import Simulation`` (or import any other name
from the ``rfx.api`` package) in this file.  Doing so creates a circular
import: ``rfx/api/__init__.py`` imports *from* this module.  The same
rule applies to any future ``rfx/api/_*.py`` mixin module.

``Result.find_resonances`` keeps its ``rfx.harminv`` import lazy (inside
the method body) — do not hoist it to module scope.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0
from rfx.geometry.csg import Shape
from rfx.sources.sources import GaussianPulse
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole


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

    def to_dict(self) -> dict[str, float | int | None | str]:
        """Return a stable JSON-serializable AD memory artifact."""
        return {
            "forward_gb": float(self.forward_gb),
            "ad_checkpointed_gb": float(self.ad_checkpointed_gb),
            "ad_full_gb": float(self.ad_full_gb),
            "ntff_dft_gb": float(self.ntff_dft_gb),
            "available_gb": (
                None if self.available_gb is None else float(self.available_gb)
            ),
            "warning": self.warning,
            "ad_segmented_gb": (
                None if self.ad_segmented_gb is None else float(self.ad_segmented_gb)
            ),
            "checkpoint_every": self.checkpoint_every,
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the estimate for research-note and CI artifacts."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        return json.dumps(self.to_dict(), **options)


class ADMemoryPlan(NamedTuple):
    """Checkpoint planning result for reverse-mode AD memory.

    ``checkpoint_every`` is the smallest segmented-scan chunk length estimated
    to fit under ``target_fraction * available_memory_gb``.  ``None`` means the
    full reverse-mode estimate already fits and segmented checkpointing is not
    required for memory.
    """

    n_steps: int
    available_memory_gb: float
    target_fraction: float
    target_memory_gb: float
    checkpoint_every: int | None
    selected_estimate: AD_MemoryEstimate
    full_ad_fits: bool
    segmented_fits: bool
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable AD memory plan artifact."""
        return {
            "n_steps": int(self.n_steps),
            "available_memory_gb": float(self.available_memory_gb),
            "target_fraction": float(self.target_fraction),
            "target_memory_gb": float(self.target_memory_gb),
            "checkpoint_every": self.checkpoint_every,
            "selected_estimate": self.selected_estimate.to_dict(),
            "full_ad_fits": bool(self.full_ad_fits),
            "segmented_fits": bool(self.segmented_fits),
            "recommendation": self.recommendation,
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the plan for memory-budget and CI artifacts."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        return json.dumps(self.to_dict(), **options)


class MeshIntelligenceReport(NamedTuple):
    """Consolidated mesh/memory preflight summary.

    This is a lightweight user-facing planning object for the
    "subgrid-like" non-uniform lane: it combines existing preflight
    advisories with cell-count and AD-memory estimates, including a
    uniform-fine comparator for non-uniform meshes.
    """
    grid_shape: tuple[int, int, int]
    cells: int
    uniform_fine_shape: tuple[int, int, int]
    uniform_fine_cells: int
    cell_savings_factor: float
    min_cell_size: float
    nominal_dx: float
    uses_nonuniform: bool
    preflight_issues: tuple[str, ...]
    ad_memory: AD_MemoryEstimate | None
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable mesh/memory planning artifact."""
        return {
            "grid_shape": list(self.grid_shape),
            "cells": int(self.cells),
            "uniform_fine_shape": list(self.uniform_fine_shape),
            "uniform_fine_cells": int(self.uniform_fine_cells),
            "cell_savings_factor": float(self.cell_savings_factor),
            "min_cell_size": float(self.min_cell_size),
            "nominal_dx": float(self.nominal_dx),
            "uses_nonuniform": bool(self.uses_nonuniform),
            "preflight_issues": list(self.preflight_issues),
            "ad_memory": (
                None if self.ad_memory is None else self.ad_memory.to_dict()
            ),
            "recommendation": self.recommendation,
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the report for memory-reduction evidence artifacts."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        return json.dumps(self.to_dict(), **options)


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

    ``dft_planes`` exposes the JIT-scan-accumulated complex DFT plane
    probes registered via :meth:`Simulation.add_dft_plane_probe`.  Each
    entry is a :class:`DFTPlaneProbe` whose ``accumulator`` field is a
    JAX-traceable complex array shaped ``(n_freqs, *plane_shape)``, with
    plane-resolved field values usable by gradient-based objectives that
    need plane-integrated V/I (e.g. waveguide-port or microstrip-port
    line-integrated voltage / closed-loop current).  ``None`` when no
    plane probes were registered.
    """
    time_series: jnp.ndarray
    ntff_data: object = None
    ntff_box: object = None
    grid: object = None
    s_params: object = None
    freqs: object = None
    lumped_port_sparams: object = None
    wire_port_sparams: object = None
    dft_planes: object = None


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


class CoaxialSMatrixResult(NamedTuple):
    """Coaxial scattering result from the experimental TEM plane-source API.

    The result schema mirrors :class:`WaveguideSMatrixResult` so the
    validation/replay infrastructure (``validate_port_smatrix``,
    ``compare_sparameter_datasets``) can consume both. The status field flags
    whether any per-frequency V/I sample fell below the configured signal
    floor; downstream tools should treat ``"degraded"`` rows with care.

    The reference plane is the cross-section that was injected on; ``z_tem``
    is the analytic ``Z_TEM`` used both for the source amplitude and for the
    power-wave decomposition.
    """

    s_params: np.ndarray
    freqs: np.ndarray
    port_names: tuple[str, ...]
    port_faces: tuple[str, ...]
    reference_planes: np.ndarray
    z_tem_ohm: np.ndarray
    voltages: np.ndarray
    currents: np.ndarray
    status: str


@dataclass(frozen=True)
class _MSLPortEntry:
    """Internal bookkeeping for a microstrip line port.

    The port covers the trace cross-section ``width × height`` at feed
    plane ``position[0]``; ``position[1]`` is the trace y-centre and
    ``position[2]`` the substrate bottom.
    """
    name: str
    position: tuple[float, float, float]
    width: float
    height: float
    direction: str
    impedance: float
    waveform: object
    excite: bool = True
    n_probe_offset: int = 5
    n_probe_spacing: int = 3
    mode: str = "eigenmode"
    eps_r_sub: float | None = None


@dataclass
class MSLSMatrixResult:
    """MSL S-matrix result.

    Attributes
    ----------
    S : (n_ports, n_ports, n_freqs) complex
        Full S-matrix.
    freqs : (n_freqs,) float
        Frequency grid in Hz.
    Z0 : (n_ports, n_freqs) complex
        Characteristic impedance extracted via 3-probe de-embedding,
        per driven-port run (``Z0[i, :]`` is from run with port i driven).
    beta : (n_freqs,) complex
        Propagation constant β = -ln(q) / Δ at the first port's run.
    port_names : tuple[str, ...]
    """
    S: np.ndarray
    freqs: np.ndarray
    Z0: np.ndarray
    beta: np.ndarray
    port_names: tuple[str, ...] = ()


__all__ = [
    "MATERIAL_LIBRARY",
    "AD_MemoryEstimate",
    "ADMemoryPlan",
    "MeshIntelligenceReport",
    "Result",
    "ForwardResult",
    "MaterialSpec",
    "_GeometryEntry",
    "_PortEntry",
    "_ProbeEntry",
    "_TFSFEntry",
    "_DFTPlaneEntry",
    "_FluxMonitorEntry",
    "_WaveguidePortEntry",
    "_FloquetPortEntry",
    "WaveguideSParamResult",
    "WaveguideSMatrixResult",
    "CoaxialSMatrixResult",
    "_MSLPortEntry",
    "MSLSMatrixResult",
]
